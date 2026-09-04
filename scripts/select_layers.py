#!/usr/bin/env python3
"""Fit the paper's single-source OLS probes and select top-8 layers.

This is the production Phase-3 pipeline for Qwen3-14B -> Qwen3-32B.  For
every target/source-layer pair it fits eight independent 128 -> 128 affine
OLS probes, one for each same-numbered KV head.  Keys are read from the
content-space ``k_content`` activation store (RoPE has already been removed)
and values are read from ``v``.

The implementation streams BF16 mmaps and saves enough sufficient statistics
to reproduce every score without rereading model activations:

* FP32 source means and centered Gram matrices;
* FP32 target fitting means plus FP64 target channel energies/SSTs;
* FP32 centered source/target cross-products for every layer pair.

Per-channel R2 is evaluated in FP64.  Channels are averaged uniformly within
each head, then all eight heads and K/V are averaged with equal weight.  The
pooled/global R2 values written to the artifacts are diagnostics only and are
never passed to the ranking function.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.activation_store import (  # noqa: E402
    ActivationMetadata,
    open_bf16_activations,
    read_activation_metadata,
)
from kvcache_transfer.config import load_yaml  # noqa: E402
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
)
from kvcache_transfer.select_layers import select_from_head_scores  # noqa: E402


CACHE_TYPES = ("k_content", "v")
SCHEMA_VERSION = 1
METHOD = "same-head-single-source-ols-affine-uniform-channel-r2"
SOLVER_CHOLESKY = 0
SOLVER_CPU_FP64_LSTSQ = 1


@dataclass(frozen=True)
class Architecture:
    source_layers: int
    target_layers: int
    kv_heads: int
    head_dim: int
    top_k: int


@dataclass(frozen=True)
class CatalogEntry:
    model_role: str
    cache_type: str
    layer: int
    path: str
    shape: tuple[int, ...]
    nbytes: int
    sha256: str
    provenance: Mapping[str, Any]


def _json_fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sidecar_for_npz(path: Path) -> Path:
    return path.with_suffix(".json")


def _atomic_save_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    *,
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> dict[str, Any]:
    """Atomically publish an uncompressed NPZ, then its JSON commit marker."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _sidecar_for_npz(path)
    if not overwrite and (path.exists() or sidecar.exists()):
        raise FileExistsError(f"refusing to replace checkpoint: {path} / {sidecar}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        checksum = sha256_file(temporary)
        os.replace(temporary, path)
        sidecar_value = {
            "schema_version": SCHEMA_VERSION,
            "payload": path.name,
            "sha256": checksum,
            "nbytes": path.stat().st_size,
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in sorted(arrays.items())
            },
            **dict(metadata),
        }
        atomic_write_json(sidecar, sidecar_value)
        return sidecar_value
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_npz_checkpoint(
    path: Path,
    *,
    expected: Mapping[str, Any] | None = None,
    verify_checksum: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sidecar = _sidecar_for_npz(path)
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    if int(raw.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {sidecar}")
    if path.stat().st_size != int(raw["nbytes"]):
        raise ValueError(f"checkpoint byte count mismatch: {path}")
    if verify_checksum and sha256_file(path) != raw["sha256"]:
        raise ValueError(f"checkpoint SHA256 mismatch: {path}")
    if expected:
        for key, wanted in expected.items():
            if raw.get(key) != wanted:
                raise ValueError(
                    f"checkpoint {path} has {key}={raw.get(key)!r}, expected {wanted!r}"
                )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    for name, description in raw["arrays"].items():
        if name not in arrays:
            raise ValueError(f"checkpoint array {name!r} is missing: {path}")
        if list(arrays[name].shape) != description["shape"]:
            raise ValueError(f"checkpoint array shape mismatch for {name}: {path}")
        if str(arrays[name].dtype) != description["dtype"]:
            raise ValueError(f"checkpoint array dtype mismatch for {name}: {path}")
    return arrays, raw


def _checkpoint_is_valid(path: Path, expected: Mapping[str, Any]) -> bool:
    if not path.is_file() or not _sidecar_for_npz(path).is_file():
        return False
    try:
        _load_npz_checkpoint(path, expected=expected, verify_checksum=True)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return True


def _activation_path(root: Path, cache_type: str, layer: int) -> Path:
    return root / cache_type / f"layer_{layer:03d}.bf16.bin"


def _catalog_activations(
    root: Path,
    *,
    model_role: str,
    layers: int,
    kv_heads: int,
    head_dim: int,
    verify_payload_sha256: bool,
    formal_protocol: bool,
) -> tuple[list[CatalogEntry], int]:
    entries: list[CatalogEntry] = []
    sample_count: int | None = None
    token_hashes: set[str] = set()
    for cache_type in CACHE_TYPES:
        for layer in range(layers):
            path = _activation_path(root, cache_type, layer)
            metadata: ActivationMetadata = read_activation_metadata(path)
            if path.stat().st_size != metadata.nbytes:
                raise ValueError(f"activation byte count mismatch: {path}")
            if verify_payload_sha256 and sha256_file(path) != metadata.sha256:
                raise ValueError(f"activation SHA256 mismatch: {path}")
            if len(metadata.shape) != 3:
                raise ValueError(f"expected [tokens,heads,dim] at {path}, got {metadata.shape}")
            expected_tail = (kv_heads, head_dim)
            if tuple(metadata.shape[1:]) != expected_tail:
                raise ValueError(
                    f"activation shape mismatch at {path}: expected [N,{kv_heads},{head_dim}], "
                    f"got {metadata.shape}"
                )
            if sample_count is None:
                sample_count = metadata.shape[0]
            elif metadata.shape[0] != sample_count:
                raise ValueError(f"activation token counts differ at {path}")
            provenance = dict(metadata.provenance)
            if provenance.get("cache_type") != cache_type:
                raise ValueError(f"cache_type provenance mismatch: {path}")
            if int(provenance.get("layer", -1)) != layer:
                raise ValueError(f"layer provenance mismatch: {path}")
            if formal_protocol:
                if provenance.get("split") != "train":
                    raise ValueError(f"selection must use calibration train activations: {path}")
                if int(provenance.get("stride", -1)) != 4:
                    raise ValueError(f"paper selection requires activation stride=4: {path}")
                if int(provenance.get("stride_start", -1)) != 0:
                    raise ValueError(f"paper selection requires stride_start=0: {path}")
                if int(provenance.get("sequences", -1)) != 500:
                    raise ValueError(f"paper selection requires 500 sequences: {path}")
                expected_model = "qwen3_14b" if model_role == "source" else "qwen3_32b"
                if provenance.get("model_name") != expected_model:
                    raise ValueError(
                        f"formal {model_role} activations must be {expected_model}: {path}"
                    )
                if provenance.get("forward_dtype") != "bfloat16":
                    raise ValueError(f"paper extraction requires BF16 forward: {path}")
                if provenance.get("content_key_inverse_dtype") != "float32":
                    raise ValueError(f"paper key inverse-RoPE must use FP32: {path}")
                token_hash = provenance.get("token_sha256")
                if (
                    not isinstance(token_hash, str)
                    or len(token_hash) != 64
                    or any(character not in "0123456789abcdef" for character in token_hash)
                ):
                    raise ValueError(f"formal activation is missing a valid token SHA256: {path}")
            if provenance.get("token_sha256"):
                token_hashes.add(str(provenance["token_sha256"]))
            entries.append(
                CatalogEntry(
                    model_role=model_role,
                    cache_type=cache_type,
                    layer=layer,
                    path=str(path.resolve()),
                    shape=metadata.shape,
                    nbytes=metadata.nbytes,
                    sha256=metadata.sha256,
                    provenance=provenance,
                )
            )
    if formal_protocol and sample_count != 500 * (1024 // 4):
        raise ValueError(
            f"paper selection requires 128000 sampled tokens, observed {sample_count}"
        )
    if len(token_hashes) > 1:
        raise ValueError(f"activation files reference different token payloads: {token_hashes}")
    assert sample_count is not None
    return entries, sample_count


def _catalog_json(entries: Sequence[CatalogEntry]) -> list[dict[str, Any]]:
    return [asdict(entry) for entry in entries]


@contextmanager
def _strict_fp32_matmul(device: torch.device) -> Iterator[None]:
    if device.type != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _configure_worker_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    torch.set_grad_enabled(False)
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA worker devices must have an explicit index")
        # Each CUDA worker also performs the FP64 R2 analysis on CPU.  PyTorch
        # otherwise gives *every* spawned worker the whole host thread pool
        # (112 logical CPUs on this server), so eight workers create roughly
        # 900 runnable threads and can be much slower than one worker.  Divide
        # the process affinity evenly across all visible GPUs.  This is only a
        # resource-scheduling choice; it does not alter any covariance, solve,
        # or ranking arithmetic.
        try:
            logical_cpus = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux fallback
            logical_cpus = os.cpu_count() or 1
        visible_cuda_workers = max(1, torch.cuda.device_count())
        torch.set_num_threads(max(1, logical_cpus // visible_cuda_workers))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Fresh spawned workers take the normal path.  The guard keeps
            # direct in-process tests usable if another test initialized the
            # inter-op pool first.
            pass
        torch.cuda.set_device(device.index)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    return device


def _iter_slices(samples: int, chunk_tokens: int) -> Iterator[slice]:
    for start in range(0, samples, chunk_tokens):
        yield slice(start, min(start + chunk_tokens, samples))


@torch.no_grad()
def _fp32_mean(mapped: Any, *, chunk_tokens: int, device: torch.device) -> torch.Tensor:
    mean: torch.Tensor | None = None
    count = 0
    for selection in _iter_slices(mapped.metadata.shape[0], chunk_tokens):
        chunk = mapped.to_torch(selection, device=device, dtype=torch.float32)
        chunk_count = chunk.shape[0]
        if mean is None:
            mean = torch.zeros(chunk.shape[1:], dtype=torch.float32, device=device)
        total = count + chunk_count
        mean.add_((chunk.mean(dim=0) - mean) * (float(chunk_count) / float(total)))
        count = total
    if mean is None:
        raise ValueError("cannot calculate a mean over empty activations")
    return mean


@torch.no_grad()
def _fp64_mean_and_sst(
    mapped: Any, *, chunk_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None
    count = 0
    for selection in _iter_slices(mapped.metadata.shape[0], chunk_tokens):
        chunk = mapped.to_torch(selection, device=device, dtype=torch.float32).to(torch.float64)
        chunk_count = chunk.shape[0]
        chunk_mean = chunk.mean(dim=0)
        centered = chunk - chunk_mean
        chunk_m2 = torch.sum(centered * centered, dim=0)
        if mean is None:
            mean = torch.zeros_like(chunk_mean)
            m2 = torch.zeros_like(chunk_m2)
        assert m2 is not None
        total = count + chunk_count
        delta = chunk_mean - mean
        m2.add_(chunk_m2 + delta.square() * (float(count * chunk_count) / float(total)))
        mean.add_(delta * (float(chunk_count) / float(total)))
        count = total
    if mean is None or m2 is None:
        raise ValueError("cannot calculate target moments over empty activations")
    return mean, m2


def _moment_path(
    statistics_root: Path, model_role: str, cache_type: str, layer: int
) -> Path:
    return statistics_root / "moments" / model_role / cache_type / f"layer_{layer:03d}.npz"


@torch.no_grad()
def _compute_moment_checkpoint(
    task: Mapping[str, Any], *, device: torch.device
) -> dict[str, Any]:
    output = Path(task["output"])
    expected = {
        "kind": task["kind"],
        "input_fingerprint": task["input_fingerprint"],
        "model_role": task["model_role"],
        "cache_type": task["cache_type"],
        "layer": task["layer"],
    }
    if not task["overwrite"] and _checkpoint_is_valid(output, expected):
        return {"path": str(output), "status": "skipped"}
    if not task["overwrite"] and (output.exists() or _sidecar_for_npz(output).exists()):
        raise ValueError(f"invalid existing moment checkpoint; inspect or use --overwrite: {output}")

    started = time.monotonic()
    activation_path = Path(task["activation_path"])
    chunk_tokens = int(task["chunk_tokens"])
    with open_bf16_activations(activation_path, verify_checksum=False) as mapped:
        mean32 = _fp32_mean(mapped, chunk_tokens=chunk_tokens, device=device)
        samples = mapped.metadata.shape[0]
        if task["kind"] == "source_moment":
            heads, dimension = mean32.shape
            gram = torch.zeros(
                (heads, dimension, dimension), dtype=torch.float32, device=device
            )
            with _strict_fp32_matmul(device):
                for selection in _iter_slices(samples, chunk_tokens):
                    chunk = mapped.to_torch(
                        selection, device=device, dtype=torch.float32
                    )
                    centered = chunk - mean32
                    gram.add_(torch.einsum("nhi,nhj->hij", centered, centered))
            arrays = {
                "mean_fp32": mean32.cpu().numpy(),
                "centered_gram_fp32": gram.cpu().numpy(),
            }
        else:
            mean64, sst64 = _fp64_mean_and_sst(
                mapped, chunk_tokens=chunk_tokens, device=device
            )
            energy_fit64 = torch.zeros_like(sst64)
            for selection in _iter_slices(samples, chunk_tokens):
                chunk = mapped.to_torch(
                    selection, device=device, dtype=torch.float32
                )
                centered64 = (chunk - mean32).to(torch.float64)
                energy_fit64.add_(torch.sum(centered64.square(), dim=0))
            arrays = {
                "mean_fp32": mean32.cpu().numpy(),
                "mean_fp64": mean64.cpu().numpy(),
                "energy_about_fit_mean_fp64": energy_fit64.cpu().numpy(),
                "sst_about_true_mean_fp64": sst64.cpu().numpy(),
            }
    metadata = {
        **expected,
        "n_samples": samples,
        "compute_dtype": "float32",
        "analysis_dtype": "float64",
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_save_npz(output, arrays, metadata=metadata, overwrite=bool(task["overwrite"]))
    return {"path": str(output), "status": "computed"}


def _run_moment_worker(
    worker_index: int, device_name: str, tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    device = _configure_worker_device(device_name)
    results = []
    for task in tasks:
        result = _compute_moment_checkpoint(task, device=device)
        result.update({"worker": worker_index, "device": device_name})
        print(json.dumps({"selection_moment": result}, sort_keys=True), flush=True)
        results.append(result)
    return results


@torch.no_grad()
def _centered_cross_product(
    source_path: Path,
    target_path: Path,
    source_mean: torch.Tensor,
    target_mean: torch.Tensor,
    *,
    chunk_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    with open_bf16_activations(source_path, verify_checksum=False) as source, \
        open_bf16_activations(target_path, verify_checksum=False) as target:
        if source.metadata.shape != target.metadata.shape:
            raise ValueError(
                f"source/target activation shapes differ: {source_path}, {target_path}"
            )
        heads, dimension = source.metadata.shape[1:]
        cross = torch.zeros(
            (heads, dimension, dimension), dtype=torch.float32, device=device
        )
        with _strict_fp32_matmul(device):
            for selection in _iter_slices(source.metadata.shape[0], chunk_tokens):
                x = source.to_torch(selection, device=device, dtype=torch.float32)
                y = target.to_torch(selection, device=device, dtype=torch.float32)
                cross.add_(
                    torch.einsum(
                        "nhi,nhj->hij",
                        x - source_mean,
                        y - target_mean,
                    )
                )
    return cross


@torch.no_grad()
def _solve_batched_ols(
    gram: torch.Tensor,
    cross: torch.Tensor,
    *,
    fallback: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve batched unregularized normal equations without silent ridge."""

    factors, info = torch.linalg.cholesky_ex(gram, check_errors=False)
    weights = torch.empty_like(cross)
    solver_codes = torch.full(
        (gram.shape[0],), SOLVER_CHOLESKY, dtype=torch.int8, device=gram.device
    )
    good = info == 0
    if bool(good.any()):
        weights[good] = torch.cholesky_solve(cross[good], factors[good])
    failed = torch.nonzero(~good, as_tuple=False).flatten().tolist()
    if failed and fallback == "error":
        raise RuntimeError(
            "unregularized OLS Gram is not positive definite for heads "
            f"{failed}; rerun explicitly with --ols-fallback cpu-fp64-lstsq "
            "after recording this failure"
        )
    for head in failed:
        # This fallback is explicit in the command/config and checkpoint.  It
        # does not add an undocumented epsilon/ridge term.
        solution = torch.linalg.lstsq(
            gram[head].detach().cpu().to(torch.float64),
            cross[head].detach().cpu().to(torch.float64),
            driver="gelsd",
        ).solution
        weights[head] = solution.to(device=gram.device, dtype=torch.float32)
        solver_codes[head] = SOLVER_CPU_FP64_LSTSQ
    return weights, solver_codes


@torch.no_grad()
def score_from_sufficient_statistics(
    gram: torch.Tensor | np.ndarray,
    cross: torch.Tensor | np.ndarray,
    weight: torch.Tensor | np.ndarray,
    target_energy_about_fit_mean: torch.Tensor | np.ndarray,
    target_sst_about_true_mean: torch.Tensor | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return uniform-channel and pooled R2 from centered OLS statistics.

    All calculations in this function are FP64.  The first returned array has
    one uniform-channel score per head and is the only score used by layer
    selection.  The pooled score and raw minimum SSE are diagnostics.
    """

    g = torch.as_tensor(gram, device="cpu", dtype=torch.float64)
    c = torch.as_tensor(cross, device="cpu", dtype=torch.float64)
    w = torch.as_tensor(weight, device="cpu", dtype=torch.float64)
    y_energy = torch.as_tensor(
        target_energy_about_fit_mean, device="cpu", dtype=torch.float64
    )
    y_sst = torch.as_tensor(target_sst_about_true_mean, device="cpu", dtype=torch.float64)
    if g.ndim != 3 or c.shape != w.shape or c.ndim != 3:
        raise ValueError("expected Gram [H,D,D] and cross/weight [H,D,D]")
    if y_energy.shape != c.shape[::2] or y_sst.shape != c.shape[::2]:
        # c.shape[::2] is (heads, output_dimension).
        raise ValueError("target energy/SST must have [H,D] shape")
    linear = torch.sum(w * c, dim=1)
    quadratic = torch.einsum("hio,hij,hjo->ho", w, g, w)
    raw_sse = y_energy - 2.0 * linear + quadratic
    # Exact SSE is non-negative.  FP32 covariance products can leave tiny
    # negative round-off after the FP64 subtraction; retain the raw minimum as
    # a diagnostic and clamp only for the definition of R2.
    raw_minimum = raw_sse.min(dim=1).values
    sse = torch.clamp(raw_sse, min=0.0)
    nonconstant = y_sst != 0
    per_channel = torch.empty_like(y_sst)
    per_channel[nonconstant] = 1.0 - sse[nonconstant] / y_sst[nonconstant]
    per_channel[~nonconstant] = torch.where(
        sse[~nonconstant] == 0,
        torch.ones_like(sse[~nonconstant]),
        torch.zeros_like(sse[~nonconstant]),
    )
    uniform = per_channel.mean(dim=1)
    sse_sum = sse.sum(dim=1)
    sst_sum = y_sst.sum(dim=1)
    pooled = torch.where(
        sst_sum != 0,
        1.0 - sse_sum / sst_sum,
        torch.where(sse_sum == 0, torch.ones_like(sse_sum), torch.zeros_like(sse_sum)),
    )
    return uniform.numpy(), pooled.numpy(), raw_minimum.numpy()


def _pair_path(statistics_root: Path, target_layer: int) -> Path:
    return statistics_root / "single_source" / f"target_layer_{target_layer:03d}.npz"


@torch.no_grad()
def _run_target_layer(
    task: Mapping[str, Any],
    *,
    device: torch.device,
    source_state: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    target_layer = int(task["target_layer"])
    output = Path(task["output"])
    expected = {
        "kind": "single_source_target_layer",
        "input_fingerprint": task["input_fingerprint"],
        "target_layer": target_layer,
        "method": METHOD,
    }
    if not task["overwrite"] and _checkpoint_is_valid(output, expected):
        return {"target_layer": target_layer, "path": str(output), "status": "skipped"}
    if not task["overwrite"] and (output.exists() or _sidecar_for_npz(output).exists()):
        raise ValueError(f"invalid existing pair checkpoint; inspect or use --overwrite: {output}")

    architecture = Architecture(**task["architecture"])
    key_head_r2 = np.empty((architecture.source_layers, architecture.kv_heads), np.float64)
    value_head_r2 = np.empty_like(key_head_r2)
    key_global_r2 = np.empty_like(key_head_r2)
    value_global_r2 = np.empty_like(key_head_r2)
    key_min_raw_sse = np.empty_like(key_head_r2)
    value_min_raw_sse = np.empty_like(key_head_r2)
    key_cross = np.empty(
        (architecture.source_layers, architecture.kv_heads, architecture.head_dim, architecture.head_dim),
        np.float32,
    )
    value_cross = np.empty_like(key_cross)
    solver_codes = np.empty((2, architecture.source_layers, architecture.kv_heads), np.int8)
    started = time.monotonic()

    for cache_index, cache_type in enumerate(CACHE_TYPES):
        target_moment_path = _moment_path(
            Path(task["statistics_root"]), "target", cache_type, target_layer
        )
        target_arrays, _ = _load_npz_checkpoint(
            target_moment_path,
            expected={
                "kind": "target_moment",
                "input_fingerprint": task["input_fingerprint"],
                "model_role": "target",
                "cache_type": cache_type,
                "layer": target_layer,
            },
        )
        target_mean = torch.from_numpy(target_arrays["mean_fp32"]).to(device=device)
        target_path = _activation_path(Path(task["target_root"]), cache_type, target_layer)
        state = source_state[cache_type]
        for source_layer in range(architecture.source_layers):
            cross = _centered_cross_product(
                _activation_path(Path(task["source_root"]), cache_type, source_layer),
                target_path,
                state["means"][source_layer],
                target_mean,
                chunk_tokens=int(task["chunk_tokens"]),
                device=device,
            )
            weight, codes = _solve_batched_ols(
                state["grams"][source_layer], cross, fallback=str(task["ols_fallback"])
            )
            uniform, pooled, raw_minimum = score_from_sufficient_statistics(
                state["grams"][source_layer],
                cross,
                weight,
                target_arrays["energy_about_fit_mean_fp64"],
                target_arrays["sst_about_true_mean_fp64"],
            )
            solver_codes[cache_index, source_layer] = codes.cpu().numpy()
            if cache_type == "k_content":
                key_cross[source_layer] = cross.cpu().numpy()
                key_head_r2[source_layer] = uniform
                key_global_r2[source_layer] = pooled
                key_min_raw_sse[source_layer] = raw_minimum
            else:
                value_cross[source_layer] = cross.cpu().numpy()
                value_head_r2[source_layer] = uniform
                value_global_r2[source_layer] = pooled
                value_min_raw_sse[source_layer] = raw_minimum

    arrays = {
        "key_cross_fp32": key_cross,
        "value_cross_fp32": value_cross,
        "key_head_uniform_r2_fp64": key_head_r2,
        "value_head_uniform_r2_fp64": value_head_r2,
        "key_head_global_r2_fp64": key_global_r2,
        "value_head_global_r2_fp64": value_global_r2,
        "key_min_raw_sse_fp64": key_min_raw_sse,
        "value_min_raw_sse_fp64": value_min_raw_sse,
        "solver_code": solver_codes,
    }
    metadata = {
        **expected,
        "architecture": asdict(architecture),
        "n_samples": int(task["n_samples"]),
        "covariance_dtype": "float32",
        "analysis_dtype": "float64",
        "solver_codes": {
            str(SOLVER_CHOLESKY): "cholesky_unregularized_normal_equations",
            str(SOLVER_CPU_FP64_LSTSQ): "explicit_cpu_fp64_lstsq_fallback",
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_save_npz(output, arrays, metadata=metadata, overwrite=bool(task["overwrite"]))
    return {"target_layer": target_layer, "path": str(output), "status": "computed"}


def _load_source_state(
    statistics_root: Path,
    architecture: Architecture,
    input_fingerprint: str,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    state: dict[str, dict[str, torch.Tensor]] = {}
    for cache_type in CACHE_TYPES:
        means = []
        grams = []
        for layer in range(architecture.source_layers):
            arrays, _ = _load_npz_checkpoint(
                _moment_path(statistics_root, "source", cache_type, layer),
                expected={
                    "kind": "source_moment",
                    "input_fingerprint": input_fingerprint,
                    "model_role": "source",
                    "cache_type": cache_type,
                    "layer": layer,
                },
            )
            means.append(torch.from_numpy(arrays["mean_fp32"]))
            grams.append(torch.from_numpy(arrays["centered_gram_fp32"]))
        state[cache_type] = {
            "means": torch.stack(means).to(device=device, dtype=torch.float32),
            "grams": torch.stack(grams).to(device=device, dtype=torch.float32),
        }
    return state


def _run_pair_worker(
    worker_index: int, device_name: str, tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    device = _configure_worker_device(device_name)
    if not tasks:
        return []
    first = tasks[0]
    architecture = Architecture(**first["architecture"])
    source_state = _load_source_state(
        Path(first["statistics_root"]),
        architecture,
        str(first["input_fingerprint"]),
        device,
    )
    results = []
    for task in tasks:
        result = _run_target_layer(task, device=device, source_state=source_state)
        result.update({"worker": worker_index, "device": device_name})
        print(json.dumps({"selection_target": result}, sort_keys=True), flush=True)
        results.append(result)
    return results


def _execute_workers(
    function: Any,
    assignments: Sequence[tuple[int, str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    nonempty = [assignment for assignment in assignments if assignment[2]]
    if len(nonempty) == 1 and nonempty[0][1] == "cpu":
        return function(*nonempty[0])
    context = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(nonempty), mp_context=context) as executor:
        futures = [executor.submit(function, *assignment) for assignment in nonempty]
        for future in futures:
            results.extend(future.result())
    return results


def _round_robin_assign(
    devices: Sequence[str], tasks: Sequence[Mapping[str, Any]]
) -> list[tuple[int, str, list[Mapping[str, Any]]]]:
    assignments: list[tuple[int, str, list[Mapping[str, Any]]]] = [
        (index, device, []) for index, device in enumerate(devices)
    ]
    for index, task in enumerate(tasks):
        assignments[index % len(assignments)][2].append(task)
    return assignments


def _write_heatmap(path: Path, scores: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(scores, origin="lower", aspect="auto", interpolation="nearest")
    axis.set_xlabel("source layer (Qwen3-14B)")
    axis.set_ylabel("target layer (Qwen3-32B)")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="uniform-channel R2")
    figure.tight_layout()
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, dpi=180)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def _parse_devices(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("--devices cannot be empty")
    if values == ["cpu"]:
        return values
    if "cpu" in values:
        raise ValueError("CPU cannot be mixed with CUDA workers")
    result = [value if value.startswith("cuda:") else f"cuda:{int(value)}" for value in values]
    if len(set(result)) != len(result):
        raise ValueError("--devices contains duplicates")
    return result


def _parse_target_layers(raw: str | None, total: int) -> list[int]:
    if raw is None:
        return list(range(total))
    values = sorted({int(value) for value in raw.split(",") if value.strip()})
    if not values or values[0] < 0 or values[-1] >= total:
        raise ValueError("--only-target-layers contains an out-of-range layer")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-conformant single-source OLS R2 and top-k selection."
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs/mapper_qwen3_14b_to_32b.yaml"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-activations", type=Path)
    parser.add_argument("--target-activations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--source-layers", type=int)
    parser.add_argument("--target-layers", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument(
        "--only-target-layers",
        help="debug/resume aid: comma-separated target layers; no final selection until all exist",
    )
    parser.add_argument(
        "--ols-fallback",
        choices=("error", "cpu-fp64-lstsq"),
        default="error",
        help="explicit behavior if an unregularized source Gram is singular",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--trust-sidecar-checksums",
        action="store_true",
        help="skip rereading activation payload SHA256 (still records committed hashes)",
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument(
        "--debug-allow-nonpaper-architecture",
        action="store_true",
        help="permit tiny synthetic layouts; artifacts are marked debug-only",
    )
    return parser.parse_args(argv)


def _resolve(path: Path | None, default: Path, project_root: Path) -> Path:
    value = default if path is None else path
    return value.resolve() if value.is_absolute() else (project_root / value).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    config = load_yaml(args.config)
    config_arch = config["architecture"]
    architecture = Architecture(
        source_layers=int(args.source_layers or config_arch["source_layers"]),
        target_layers=int(args.target_layers or config_arch["target_layers"]),
        kv_heads=int(args.kv_heads or config_arch["source_kv_heads"]),
        head_dim=int(args.head_dim or config_arch["source_head_dim"]),
        top_k=int(args.top_k or config["paper"]["selected_source_layers"]),
    )
    formal = not args.debug_allow_nonpaper_architecture
    if formal and architecture != Architecture(40, 64, 8, 128, 8):
        raise ValueError(
            "formal Qwen3-14B -> Qwen3-32B selection must use 40/64 layers, "
            "8 KV heads, head_dim=128, top_k=8"
        )
    if not (0 < architecture.top_k <= architecture.source_layers):
        raise ValueError("top_k must be in [1, source_layers]")
    chunk_tokens = int(args.chunk_tokens or config["runtime"]["statistics_chunk_tokens"])
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    devices = _parse_devices(args.devices)
    source_root = _resolve(
        args.source_activations,
        Path("data/activations/qwen3_14b/train"),
        project_root,
    )
    target_root = _resolve(
        args.target_activations,
        Path("data/activations/qwen3_32b/train"),
        project_root,
    )
    output = _resolve(
        args.output,
        Path("artifacts/selection/qwen3_14b_to_32b"),
        project_root,
    )
    statistics_root = _resolve(
        args.statistics,
        Path("artifacts/statistics/qwen3_14b_to_32b/selection"),
        project_root,
    )
    output.mkdir(parents=True, exist_ok=True)
    statistics_root.mkdir(parents=True, exist_ok=True)

    run = make_run_directory(project_root, "p3", "single_source_ols_top8")
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "architecture": asdict(architecture),
        "formal_protocol": formal,
        "source_activations": str(source_root),
        "target_activations": str(target_root),
        "output": str(output),
        "statistics": str(statistics_root),
        "devices": devices,
        "chunk_tokens": chunk_tokens,
        "covariance_dtype": "float32",
        "analysis_dtype": "float64",
        "ols_fallback": args.ols_fallback,
        "activation_sha256_verified": not args.trust_sidecar_checksums,
    }
    record_run_start(run, list(sys.argv if argv is None else [sys.argv[0], *argv]), run_config)
    atomic_write_json(run.path / "status.json", {"status": "cataloging"})

    try:
        source_catalog, source_samples = _catalog_activations(
            source_root,
            model_role="source",
            layers=architecture.source_layers,
            kv_heads=architecture.kv_heads,
            head_dim=architecture.head_dim,
            verify_payload_sha256=not args.trust_sidecar_checksums,
            formal_protocol=formal,
        )
        target_catalog, target_samples = _catalog_activations(
            target_root,
            model_role="target",
            layers=architecture.target_layers,
            kv_heads=architecture.kv_heads,
            head_dim=architecture.head_dim,
            verify_payload_sha256=not args.trust_sidecar_checksums,
            formal_protocol=formal,
        )
        if source_samples != target_samples:
            raise ValueError(
                f"source/target activation token counts differ: {source_samples}, {target_samples}"
            )
        source_token_hashes = {
            str(entry.provenance.get("token_sha256"))
            for entry in source_catalog
            if entry.provenance.get("token_sha256")
        }
        target_token_hashes = {
            str(entry.provenance.get("token_sha256"))
            for entry in target_catalog
            if entry.provenance.get("token_sha256")
        }
        if source_token_hashes != target_token_hashes:
            raise ValueError(
                "source and target activations were not extracted from the same token payload: "
                f"{source_token_hashes} vs {target_token_hashes}"
            )
        catalog_value = {
            "schema_version": SCHEMA_VERSION,
            "architecture": asdict(architecture),
            "source": _catalog_json(source_catalog),
            "target": _catalog_json(target_catalog),
        }
        input_fingerprint = _json_fingerprint(catalog_value)
        catalog_value["input_fingerprint"] = input_fingerprint
        activation_manifest = statistics_root / "activation_inputs.json"
        atomic_write_json(activation_manifest, catalog_value)

        if args.verify_only:
            selection_path = output / "selected_layers.json"
            selected = json.loads(selection_path.read_text(encoding="utf-8"))
            if selected["input_fingerprint"] != input_fingerprint:
                raise ValueError("selected layer manifest belongs to different activations")
            score_path = output / "scores.npz"
            if sha256_file(score_path) != selected["score_artifacts"]["npz"]["sha256"]:
                raise ValueError("selection score artifact checksum mismatch")
            atomic_write_json(run.path / "status.json", {"status": "verified"})
            print(json.dumps({"status": "verified", "selection": str(selection_path)}))
            return 0

        moment_tasks: list[dict[str, Any]] = []
        for model_role, root, layers, kind in (
            ("source", source_root, architecture.source_layers, "source_moment"),
            ("target", target_root, architecture.target_layers, "target_moment"),
        ):
            for cache_type in CACHE_TYPES:
                for layer in range(layers):
                    moment_tasks.append(
                        {
                            "kind": kind,
                            "input_fingerprint": input_fingerprint,
                            "model_role": model_role,
                            "cache_type": cache_type,
                            "layer": layer,
                            "activation_path": str(_activation_path(root, cache_type, layer)),
                            "output": str(
                                _moment_path(statistics_root, model_role, cache_type, layer)
                            ),
                            "chunk_tokens": chunk_tokens,
                            "overwrite": args.overwrite,
                        }
                    )
        atomic_write_json(run.path / "status.json", {"status": "moments"})
        _execute_workers(_run_moment_worker, _round_robin_assign(devices, moment_tasks))

        moment_entries = []
        for task in moment_tasks:
            path = Path(task["output"])
            _, sidecar = _load_npz_checkpoint(
                path,
                expected={
                    "kind": task["kind"],
                    "input_fingerprint": input_fingerprint,
                    "model_role": task["model_role"],
                    "cache_type": task["cache_type"],
                    "layer": task["layer"],
                },
            )
            moment_entries.append(
                {
                    "path": str(path),
                    "sha256": sidecar["sha256"],
                    "kind": task["kind"],
                    "model_role": task["model_role"],
                    "cache_type": task["cache_type"],
                    "layer": task["layer"],
                }
            )
        moments_manifest = statistics_root / "moments_manifest.json"
        atomic_write_json(
            moments_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "n_samples": source_samples,
                "entries": moment_entries,
            },
        )

        target_layers = _parse_target_layers(args.only_target_layers, architecture.target_layers)
        pair_tasks = [
            {
                "target_layer": target_layer,
                "output": str(_pair_path(statistics_root, target_layer)),
                "statistics_root": str(statistics_root),
                "source_root": str(source_root),
                "target_root": str(target_root),
                "input_fingerprint": input_fingerprint,
                "architecture": asdict(architecture),
                "n_samples": source_samples,
                "chunk_tokens": chunk_tokens,
                "ols_fallback": args.ols_fallback,
                "overwrite": args.overwrite,
            }
            for target_layer in target_layers
        ]
        atomic_write_json(run.path / "status.json", {"status": "pair_ols"})
        _execute_workers(_run_pair_worker, _round_robin_assign(devices, pair_tasks))

        missing = [
            layer
            for layer in range(architecture.target_layers)
            if not _pair_path(statistics_root, layer).is_file()
            or not _sidecar_for_npz(_pair_path(statistics_root, layer)).is_file()
        ]
        if missing:
            atomic_write_json(
                run.path / "status.json",
                {"status": "partial", "missing_target_layers": missing},
            )
            print(
                json.dumps(
                    {
                        "status": "partial",
                        "missing_target_layers": missing,
                        "resume_command": "rerun without --overwrite",
                    },
                    indent=2,
                )
            )
            return 0

        shape = (architecture.target_layers, architecture.source_layers, architecture.kv_heads)
        key_head = np.empty(shape, dtype=np.float64)
        value_head = np.empty(shape, dtype=np.float64)
        key_global = np.empty(shape, dtype=np.float64)
        value_global = np.empty(shape, dtype=np.float64)
        solver_codes = np.empty((architecture.target_layers, 2, architecture.source_layers, architecture.kv_heads), dtype=np.int8)
        pair_entries = []
        for target_layer in range(architecture.target_layers):
            path = _pair_path(statistics_root, target_layer)
            arrays, sidecar = _load_npz_checkpoint(
                path,
                expected={
                    "kind": "single_source_target_layer",
                    "input_fingerprint": input_fingerprint,
                    "target_layer": target_layer,
                    "method": METHOD,
                },
            )
            key_head[target_layer] = arrays["key_head_uniform_r2_fp64"]
            value_head[target_layer] = arrays["value_head_uniform_r2_fp64"]
            key_global[target_layer] = arrays["key_head_global_r2_fp64"]
            value_global[target_layer] = arrays["value_head_global_r2_fp64"]
            solver_codes[target_layer] = arrays["solver_code"]
            pair_entries.append(
                {"target_layer": target_layer, "path": str(path), "sha256": sidecar["sha256"]}
            )
        pair_manifest = statistics_root / "single_source_manifest.json"
        atomic_write_json(
            pair_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "method": METHOD,
                "entries": pair_entries,
            },
        )

        # This is the sole selection call.  It receives only uniform-channel
        # head scores; the global diagnostic tensors are intentionally absent.
        selection = select_from_head_scores(key_head, value_head, k=architecture.top_k)
        score_path = output / "scores.npz"
        score_arrays = {
            "key_head_uniform_r2": key_head,
            "value_head_uniform_r2": value_head,
            "key_head_global_r2_diagnostic": key_global,
            "value_head_global_r2_diagnostic": value_global,
            "key_head_averaged_uniform_r2": selection.key_scores,
            "value_head_averaged_uniform_r2": selection.value_scores,
            "combined_equal_kv_uniform_r2": selection.combined_scores,
            "solver_code": solver_codes,
        }
        score_expected = {
            "kind": "selection_scores",
            "input_fingerprint": input_fingerprint,
            "method": METHOD,
            "selection_score": "uniform_channel_r2_only",
            "global_r2_role": "diagnostic_only_never_used_for_ranking",
        }
        if not args.overwrite and _checkpoint_is_valid(score_path, score_expected):
            existing_arrays, score_sidecar = _load_npz_checkpoint(
                score_path, expected=score_expected
            )
            for name, expected_array in score_arrays.items():
                if not np.array_equal(existing_arrays[name], expected_array, equal_nan=True):
                    raise ValueError(
                        f"existing immutable selection scores differ for array {name}; "
                        "inspect or use --overwrite"
                    )
        else:
            if not args.overwrite and (
                score_path.exists() or _sidecar_for_npz(score_path).exists()
            ):
                raise ValueError(
                    f"invalid existing selection score artifact; inspect or use --overwrite: {score_path}"
                )
            score_sidecar = _atomic_save_npz(
                score_path,
                score_arrays,
                metadata=score_expected,
                overwrite=args.overwrite,
            )
        heatmaps = {}
        for name, scores, title in (
            ("key", selection.key_scores, "K-content single-source OLS"),
            ("value", selection.value_scores, "V single-source OLS"),
            ("combined", selection.combined_scores, "Equal-weight K/V selection score"),
        ):
            path = output / f"heatmap_{name}.png"
            _write_heatmap(path, scores, title)
            heatmaps[name] = {"path": str(path), "sha256": sha256_file(path)}

        selected_path = output / "selected_layers.json"
        selected_manifest = {
            "schema_version": SCHEMA_VERSION,
            "direction": "qwen3_14b_to_32b",
            "formal_protocol": formal,
            "method": METHOD,
            "source_layers": architecture.source_layers,
            "target_layers": architecture.target_layers,
            "kv_heads": architecture.kv_heads,
            "head_dim": architecture.head_dim,
            "k": architecture.top_k,
            "n_samples": source_samples,
            "input_fingerprint": input_fingerprint,
            "token_sha256": sorted(source_token_hashes),
            "selected_source_layers_rank_order": [
                list(layers) for layers in selection.selected_layers
            ],
            "feature_order": "selected-source-layer rank, then KV head 0..7, then head dimension 0..127",
            "selection_rule": {
                "probe": "independent affine OLS 128->128 for same-numbered KV head",
                "key_space": "content K after inverse RoPE",
                "channel_reduction": "uniform arithmetic mean of 128 per-channel R2 values",
                "head_reduction": "uniform arithmetic mean over 8 KV heads",
                "kv_reduction": "equal arithmetic mean of K and V scores",
                "tie_break": "lower source layer index",
                "global_r2": "diagnostic only; never used for ranking",
                "benchmark_data_used": False,
            },
            "score_artifacts": {
                "npz": {"path": str(score_path), "sha256": score_sidecar["sha256"]},
                "heatmaps": heatmaps,
            },
            "sufficient_statistics": {
                "activation_inputs": {
                    "path": str(activation_manifest),
                    "sha256": sha256_file(activation_manifest),
                },
                "moments_manifest": {
                    "path": str(moments_manifest),
                    "sha256": sha256_file(moments_manifest),
                },
                "single_source_manifest": {
                    "path": str(pair_manifest),
                    "sha256": sha256_file(pair_manifest),
                },
            },
        }
        manifest_reused = False
        if selected_path.exists() and not args.overwrite:
            existing = json.loads(selected_path.read_text(encoding="utf-8"))
            if existing != selected_manifest:
                raise FileExistsError(
                    f"selection manifest exists and differs; inspect or use --overwrite: {selected_path}"
                )
            manifest_reused = True
        else:
            atomic_write_json(selected_path, selected_manifest)

        metrics = {
            "mean_best_single_key_uniform_r2": float(selection.key_scores.max(axis=1).mean()),
            "mean_best_single_value_uniform_r2": float(selection.value_scores.max(axis=1).mean()),
            "mean_best_combined_uniform_r2": float(selection.combined_scores.max(axis=1).mean()),
            "explicit_ols_fallback_head_count": int(
                np.count_nonzero(solver_codes == SOLVER_CPU_FP64_LSTSQ)
            ),
            "selected_source_layers_rank_order": [
                list(layers) for layers in selection.selected_layers
            ],
        }
        atomic_write_json(run.path / "metrics.json", metrics)
        atomic_write_json(
            run.path / "manifest.json",
            {
                "status": "complete",
                "selection": str(selected_path),
                "selection_sha256": sha256_file(selected_path),
                "scores": str(score_path),
                "scores_sha256": score_sidecar["sha256"],
                "input_fingerprint": input_fingerprint,
            },
        )
        atomic_write_json(run.path / "status.json", {"status": "complete"})
        if not args.no_ledger and not manifest_reused:
            append_reproduction_log(
                project_root,
                heading="P3-SEL-001：single-source OLS 与 top-8 选层",
                bullets=[
                    "仅使用 500×1024 FineWeb-Edu train calibration（stride=4，共 128000 token positions），未使用 benchmark 数据选层。",
                    "逐同编号 KV head 拟合无 ridge 的 128→128 affine OLS；K 使用反 RoPE 后 content key，V 直接使用。",
                    "逐 channel FP64 R² 等权平均，再对 8 heads 与 K/V 等权平均；global R² 只保存为诊断。",
                    f"完整 top-{architecture.top_k} 结果：{selected_path}；scores SHA256={score_sidecar['sha256']}。",
                    f"显式 OLS fallback heads={metrics['explicit_ols_fallback_head_count']}；run={run.run_id}。",
                ],
            )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "selection": str(selected_path),
                    "metrics": metrics,
                    "run": str(run.path),
                },
                indent=2,
            )
        )
        return 0
    except BaseException as error:
        atomic_write_json(
            run.path / "status.json",
            {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
