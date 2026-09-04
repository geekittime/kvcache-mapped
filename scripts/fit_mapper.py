#!/usr/bin/env python3
"""Fit the complete Qwen3-14B -> Qwen3-32B closed-form mapper.

The production run is embarrassingly parallel across target layers.  Worker
``i`` owns layers whose index modulo the number of selected devices equals
``i``; no NCCL collectives or cross-GPU statistics are involved.  Each cache
type is fitted by replaying the BF16 mmap activations twice (means, then
centered sufficient statistics), followed by an FP64 diagnostic pass.

Per-layer safetensors plus their JSON sidecars are checkpoints.  A sidecar is
accepted on resume only after its payload SHA256 and complete fit fingerprint
match the current run.  The aggregate manifest is published only after all 64
paper layers have independently committed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.activation_store import (  # noqa: E402
    open_bf16_activations,
    sidecar_path as activation_sidecar_path,
)
from kvcache_transfer.artifacts import save_layer_mapper  # noqa: E402
from kvcache_transfer.config import load_yaml  # noqa: E402
from kvcache_transfer.fit_mapper import MMapLayerChunkFactory  # noqa: E402
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.mapper import LayerMapper  # noqa: E402
from kvcache_transfer.metrics import R2Accumulator  # noqa: E402
from kvcache_transfer.ridge import (  # noqa: E402
    RidgeSolution,
    fit_centered_ridge_streaming,
    normal_equation_relative_residual,
)
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
)


DEFAULT_CONFIG = ROOT / "configs/mapper_qwen3_14b_to_32b.yaml"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FitArchitecture:
    """All dimensions that define one mapper family."""

    source_layers: int
    target_layers: int
    kv_heads: int
    head_dim: int
    selected_source_layers: int
    n_samples: int

    @property
    def source_features(self) -> int:
        return self.selected_source_layers * self.kv_heads * self.head_dim

    @property
    def target_features(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def expected_weight_count(self) -> int:
        return 2 * self.target_layers * self.source_features * self.target_features

    @property
    def expected_bias_count(self) -> int:
        return 2 * self.target_layers * self.target_features


@dataclass(frozen=True)
class WorkerJob:
    rank: int
    device: str
    target_layers: tuple[int, ...]
    selected_layers: tuple[tuple[int, ...], ...]
    source_activation_root: str
    target_activation_root: str
    output_directory: str
    run_directory: str
    fit_fingerprint: str
    selection_sha256: str
    architecture: FitArchitecture
    ridge_lambda: float
    chunk_tokens: int
    solver: str
    analysis_device: str
    resume: bool
    cpu_threads: int


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _resolve(project_root: Path, path: str | Path) -> Path:
    result = Path(path)
    return result if result.is_absolute() else project_root / result


def activation_path(root: str | Path, cache_type: str, layer: int) -> Path:
    if cache_type not in ("k_content", "v"):
        raise ValueError(f"unknown cache type: {cache_type}")
    return Path(root) / cache_type / f"layer_{layer:03d}.bf16.bin"


def layer_mapper_path(output_directory: str | Path, target_layer: int) -> Path:
    return Path(output_directory) / "layers" / f"layer_{target_layer:03d}.safetensors"


def component_mapper_path(
    output_directory: str | Path, target_layer: int, cache_type: str
) -> Path:
    if cache_type not in ("k_content", "v"):
        raise ValueError(f"unknown mapper cache type: {cache_type}")
    return (
        Path(output_directory)
        / "components"
        / cache_type
        / f"layer_{target_layer:03d}.safetensors"
    )


def _save_mapper_component(
    output_directory: str | Path,
    *,
    target_layer: int,
    cache_type: str,
    selected_layers: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    fit_fingerprint: str,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically checkpoint one independently fitted K/V regression."""

    destination = component_mapper_path(output_directory, target_layer, cache_type)
    sidecar_path = destination.with_suffix(".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    try:
        save_file(
            {
                "weight": weight.detach().cpu().to(torch.float32).contiguous(),
                "bias": bias.detach().cpu().to(torch.float32).contiguous(),
            },
            temporary_name,
            metadata={
                "target_layer": str(target_layer),
                "cache_type": cache_type,
                "source_layers_rank_order": ",".join(map(str, selected_layers)),
                "fit_fingerprint": fit_fingerprint,
            },
        )
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "target_layer": target_layer,
        "cache_type": cache_type,
        "source_layers_rank_order": list(map(int, selected_layers)),
        "fit_fingerprint": fit_fingerprint,
        "dtype": "float32",
        "weight_shape": list(weight.shape),
        "bias_shape": list(bias.shape),
        "weight_numel": int(weight.numel()),
        "bias_numel": int(bias.numel()),
        "sha256": sha256_file(destination),
        "diagnostics": dict(diagnostics),
    }
    # The sidecar is the commit marker and is deliberately published last.
    atomic_write_json(sidecar_path, sidecar)
    return sidecar


def _load_mapper_component(
    output_directory: str | Path,
    *,
    target_layer: int,
    cache_type: str,
    selected_layers: Sequence[int],
    architecture: FitArchitecture,
    fit_fingerprint: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
    """Load a validated K/V checkpoint, or return None if uncommitted."""

    payload = component_mapper_path(output_directory, target_layer, cache_type)
    sidecar_path = payload.with_suffix(".json")
    if not sidecar_path.exists():
        return None
    if not payload.exists():
        raise ValueError(f"component commit marker has no payload: {sidecar_path}")
    sidecar = _strict_json(sidecar_path)
    expectations = {
        "schema_version": SCHEMA_VERSION,
        "target_layer": target_layer,
        "cache_type": cache_type,
        "source_layers_rank_order": list(map(int, selected_layers)),
        "fit_fingerprint": fit_fingerprint,
        "dtype": "float32",
        "weight_shape": [architecture.source_features, architecture.target_features],
        "bias_shape": [architecture.target_features],
        "weight_numel": architecture.source_features * architecture.target_features,
        "bias_numel": architecture.target_features,
    }
    for field, expected in expectations.items():
        if sidecar.get(field) != expected:
            raise ValueError(
                f"{sidecar_path}: {field}={sidecar.get(field)!r}, expected {expected!r}"
            )
    if sha256_file(payload) != sidecar.get("sha256"):
        raise ValueError(f"component mapper SHA256 mismatch: {payload}")
    tensors = load_file(payload, device="cpu")
    if set(tensors) != {"weight", "bias"}:
        raise ValueError(f"component mapper tensor set is invalid: {payload}")
    weight = tensors["weight"]
    bias = tensors["bias"]
    if weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise ValueError(f"component mapper is not FP32: {payload}")
    diagnostics = sidecar.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError(f"component mapper lacks diagnostics: {sidecar_path}")
    return weight, bias, diagnostics


def load_selected_layers(
    path: str | Path, architecture: FitArchitecture
) -> tuple[tuple[int, ...], ...]:
    """Load and strictly validate the rank-ordered top-k selection manifest."""

    manifest_path = Path(path)
    raw = _strict_json(manifest_path)
    rows = raw.get("selected_source_layers_rank_order")
    if not isinstance(rows, list):
        raise ValueError(
            f"{manifest_path} lacks selected_source_layers_rank_order"
        )
    metadata_expectations = {
        "source_layers": architecture.source_layers,
        "target_layers": architecture.target_layers,
        "kv_heads": architecture.kv_heads,
        "head_dim": architecture.head_dim,
        "k": architecture.selected_source_layers,
        "n_samples": architecture.n_samples,
    }
    for field, expected in metadata_expectations.items():
        if field in raw and int(raw[field]) != expected:
            raise ValueError(
                f"selection {field}={raw[field]!r}, expected {expected!r}"
            )
    if len(rows) != architecture.target_layers:
        raise ValueError(
            f"selection has {len(rows)} target rows, expected {architecture.target_layers}"
        )
    selected: list[tuple[int, ...]] = []
    for target_layer, row in enumerate(rows):
        if not isinstance(row, list):
            raise ValueError(f"target layer {target_layer} selection is not a list")
        try:
            indices = tuple(int(index) for index in row)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target layer {target_layer} has non-integer indices") from exc
        if len(indices) != architecture.selected_source_layers:
            raise ValueError(
                f"target layer {target_layer} selects {len(indices)} layers, "
                f"expected {architecture.selected_source_layers}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"target layer {target_layer} contains duplicate source layers")
        if any(index < 0 or index >= architecture.source_layers for index in indices):
            raise ValueError(f"target layer {target_layer} has an out-of-range source layer")
        selected.append(indices)
    return tuple(selected)


def assign_target_layers(
    target_layers: Sequence[int], devices: Sequence[str]
) -> dict[str, tuple[int, ...]]:
    """Assign layer ``t`` to worker ``t % len(devices)`` deterministically."""

    if not devices:
        raise ValueError("at least one compute device is required")
    if len(set(devices)) != len(devices):
        raise ValueError("compute devices must be unique")
    assignments: list[list[int]] = [[] for _ in devices]
    for target_layer in sorted(int(value) for value in target_layers):
        assignments[target_layer % len(devices)].append(target_layer)
    return {
        str(device): tuple(assignments[rank]) for rank, device in enumerate(devices)
    }


def assert_paper_fit(architecture: FitArchitecture, ridge_lambda: float) -> None:
    expected = FitArchitecture(
        source_layers=40,
        target_layers=64,
        kv_heads=8,
        head_dim=128,
        selected_source_layers=8,
        n_samples=128_000,
    )
    if architecture != expected:
        raise ValueError(f"paper architecture mismatch: {architecture!r} != {expected!r}")
    if ridge_lambda != 0.01:
        raise ValueError(f"paper ridge lambda must be 0.01, observed {ridge_lambda}")
    if architecture.source_features != 8192 or architecture.target_features != 1024:
        raise AssertionError("paper design matrix dimensions must be 8192 -> 1024")
    if architecture.expected_weight_count != 1_073_741_824:
        raise AssertionError("paper mapper weight count invariant failed")
    if architecture.expected_bias_count != 131_072:
        raise AssertionError("paper mapper bias count invariant failed")


def collect_activation_inventory(
    source_root: str | Path,
    target_root: str | Path,
    architecture: FitArchitecture,
    *,
    verify_payload_checksum: bool,
) -> dict[str, Any]:
    """Validate every source/target K/V payload and record immutable hashes."""

    entries: list[dict[str, Any]] = []
    provenance_checks: dict[str, set[str]] = {
        "token_sha256": set(),
        "split": set(),
        "stride": set(),
        "stride_start": set(),
        "sequences": set(),
    }
    expected_shape = (
        architecture.n_samples,
        architecture.kv_heads,
        architecture.head_dim,
    )
    for role, root, layer_count in (
        ("source", Path(source_root), architecture.source_layers),
        ("target", Path(target_root), architecture.target_layers),
    ):
        for cache_type in ("k_content", "v"):
            for layer in range(layer_count):
                payload = activation_path(root, cache_type, layer)
                with open_bf16_activations(
                    payload, verify_checksum=verify_payload_checksum
                ) as mapped:
                    metadata = mapped.metadata
                    if metadata.shape != expected_shape:
                        raise ValueError(
                            f"{payload}: shape {metadata.shape}, expected {expected_shape}"
                        )
                    provenance = dict(metadata.provenance)
                provenance_expectations = {
                    "cache_type": cache_type,
                    "layer": layer,
                    "model_name": "qwen3_14b" if role == "source" else "qwen3_32b",
                    "forward_dtype": "bfloat16",
                    "content_key_inverse_dtype": "float32",
                }
                for field, expected in provenance_expectations.items():
                    if field not in provenance:
                        raise ValueError(f"{payload}: provenance lacks required {field}")
                    observed = provenance[field]
                    if field == "layer":
                        try:
                            matches = int(observed) == expected
                        except (TypeError, ValueError):
                            matches = False
                    else:
                        matches = str(observed) == expected
                    if not matches:
                        raise ValueError(
                            f"{payload}: provenance {field}={observed!r}, expected {expected!r}"
                        )
                for key in provenance_checks:
                    if key in provenance:
                        provenance_checks[key].add(str(provenance[key]))
                metadata_path = activation_sidecar_path(payload)
                entries.append(
                    {
                        "role": role,
                        "cache_type": cache_type,
                        "layer": layer,
                        "payload_path": str(payload.resolve()),
                        "nbytes": metadata.nbytes,
                        "payload_sha256": metadata.sha256,
                        "metadata_path": str(metadata_path.resolve()),
                        "metadata_sha256": sha256_file(metadata_path),
                        "shape": list(metadata.shape),
                        "provenance": provenance,
                        "provenance_sha256": _canonical_sha256(provenance),
                    }
                )
    # These fields must bind source and target extraction to the same tokens.
    for field, values in provenance_checks.items():
        if not values:
            raise ValueError(f"activation provenance lacks required shared field {field}")
        if len(values) > 1:
            raise ValueError(f"activation provenance disagrees on {field}: {sorted(values)}")
    shared = {
        field: next(iter(values)) if values else None
        for field, values in provenance_checks.items()
    }
    expected_shared = {
        "split": "train",
        "stride": "4",
        "stride_start": "0",
        # Each 1024-token calibration window contributes 256 stride-4 rows.
        "sequences": str(architecture.n_samples // 256),
    }
    for field, expected in expected_shared.items():
        if shared[field] != expected:
            raise ValueError(
                f"activation provenance {field}={shared[field]!r}, expected {expected!r}"
            )
    if len(str(shared["token_sha256"])) != 64:
        raise ValueError("activation token_sha256 must be a SHA256 hex digest")
    if any(character not in "0123456789abcdef" for character in str(shared["token_sha256"])):
        raise ValueError("activation token_sha256 must be lowercase hexadecimal")
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "payload_checksums_verified": bool(verify_payload_checksum),
        "expected_shape": list(expected_shape),
        "shared_provenance": shared,
        "entries": entries,
    }
    inventory["inventory_digest"] = _canonical_sha256(inventory)
    return inventory


def validate_selection_provenance(
    selection_path: str | Path,
    activation_inventory: Mapping[str, Any],
    *,
    require_formal_protocol: bool,
) -> dict[str, Any]:
    """Bind the selected top-k rows to the exact activation payload catalog.

    Phase 3 records the activation-sidecar catalog that its OLS statistics
    consumed.  Merely checking the top-k JSON hash is insufficient: a valid
    top-k file copied from another capture has the same dimensions.  This
    routine follows the catalog link, verifies its checksum/fingerprint, and
    compares every source/target K/V payload and provenance object against the
    Phase-4 preflight inventory.
    """

    selection_path = Path(selection_path)
    selection = _strict_json(selection_path)
    if require_formal_protocol and selection.get("formal_protocol") is not True:
        raise ValueError("production fit requires a formal-protocol selection manifest")
    if selection.get("direction") != "qwen3_14b_to_32b":
        raise ValueError("selection direction is not qwen3_14b_to_32b")
    rule = selection.get("selection_rule")
    if not isinstance(rule, dict):
        raise ValueError("selection manifest lacks its ranking rule")
    if rule.get("benchmark_data_used") is not False:
        raise ValueError("selection must explicitly record that benchmarks were not used")
    if "uniform arithmetic mean" not in str(rule.get("channel_reduction", "")):
        raise ValueError("selection was not ranked by uniform per-channel R2")
    if "uniform arithmetic mean" not in str(rule.get("head_reduction", "")):
        raise ValueError("selection did not average KV heads uniformly")
    if "equal arithmetic mean" not in str(rule.get("kv_reduction", "")):
        raise ValueError("selection did not weight K and V equally")
    expected_feature_order = (
        "selected-source-layer rank, then KV head 0..7, then head dimension 0..127"
    )
    if selection.get("feature_order") != expected_feature_order:
        raise ValueError("selection feature order is not the production rank/head/dim order")
    statistics = selection.get("sufficient_statistics")
    if not isinstance(statistics, dict):
        raise ValueError("selection manifest lacks sufficient-statistics provenance")
    catalog_reference = statistics.get("activation_inputs")
    if not isinstance(catalog_reference, dict):
        raise ValueError("selection manifest lacks its activation input catalog")
    catalog_path = Path(str(catalog_reference.get("path", "")))
    if not catalog_path.is_absolute():
        catalog_path = selection_path.parent / catalog_path
    if not catalog_path.is_file():
        raise ValueError(f"selection activation catalog is missing: {catalog_path}")
    catalog_sha = sha256_file(catalog_path)
    if catalog_sha != catalog_reference.get("sha256"):
        raise ValueError("selection activation catalog SHA256 mismatch")
    catalog = _strict_json(catalog_path)
    recorded_fingerprint = catalog.get("input_fingerprint")
    catalog_without_fingerprint = dict(catalog)
    catalog_without_fingerprint.pop("input_fingerprint", None)
    if recorded_fingerprint != _canonical_sha256(catalog_without_fingerprint):
        raise ValueError("selection activation catalog fingerprint is invalid")
    if selection.get("input_fingerprint") != recorded_fingerprint:
        raise ValueError("selection and activation catalog fingerprints disagree")

    expected_entries: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for entry in activation_inventory.get("entries", []):
        if not isinstance(entry, Mapping):
            raise ValueError("fit activation inventory contains a malformed entry")
        key = (str(entry["role"]), str(entry["cache_type"]), int(entry["layer"]))
        expected_entries[key] = entry
    observed_entries: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for role in ("source", "target"):
        role_entries = catalog.get(role)
        if not isinstance(role_entries, list):
            raise ValueError(f"selection activation catalog lacks {role} entries")
        for entry in role_entries:
            if not isinstance(entry, Mapping):
                raise ValueError("selection activation catalog contains a malformed entry")
            key = (role, str(entry.get("cache_type")), int(entry.get("layer", -1)))
            if key in observed_entries:
                raise ValueError(f"selection activation catalog duplicates {key}")
            observed_entries[key] = entry
    if set(expected_entries) != set(observed_entries):
        raise ValueError("selection and fit activation catalogs cover different payloads")
    for key, expected in expected_entries.items():
        observed = observed_entries[key]
        comparisons = {
            "path": (str(observed.get("path")), str(expected["payload_path"])),
            "shape": (list(observed.get("shape", ())), list(expected["shape"])),
            "nbytes": (int(observed.get("nbytes", -1)), int(expected["nbytes"])),
            "sha256": (str(observed.get("sha256")), str(expected["payload_sha256"])),
            "provenance": (observed.get("provenance"), expected["provenance"]),
        }
        for field, (observed_value, expected_value) in comparisons.items():
            if observed_value != expected_value:
                raise ValueError(
                    f"selection activation catalog mismatch for {key} field {field}"
                )
    token_hashes = selection.get("token_sha256")
    shared_token_hash = activation_inventory.get("shared_provenance", {}).get(
        "token_sha256"
    )
    if token_hashes != [shared_token_hash]:
        raise ValueError("selection token hash does not match fitted activations")

    scores = selection.get("score_artifacts", {}).get("npz", {})
    score_path = Path(str(scores.get("path", "")))
    if not score_path.is_absolute():
        score_path = selection_path.parent / score_path
    if not score_path.is_file() or sha256_file(score_path) != scores.get("sha256"):
        raise ValueError("selection score artifact is absent or fails SHA256")
    return {
        "selection_input_fingerprint": recorded_fingerprint,
        "activation_catalog_path": str(catalog_path.resolve()),
        "activation_catalog_sha256": catalog_sha,
        "score_path": str(score_path.resolve()),
        "score_sha256": str(scores["sha256"]),
    }


def make_fit_fingerprint(
    *,
    architecture: FitArchitecture,
    ridge_lambda: float,
    allow_tf32: bool,
    solver: str,
    chunk_tokens: int,
    selection_sha256: str,
    activation_inventory_digest: str,
) -> tuple[str, dict[str, Any]]:
    """Hash only settings and inputs that can change fitted weights."""

    definition = {
        "schema_version": SCHEMA_VERSION,
        "method": "two-pass-centered-ridge",
        "architecture": asdict(architecture),
        "ridge_lambda_added_directly_to_unnormalized_gram": ridge_lambda,
        "standardize_features": False,
        "regularize_bias": False,
        "covariance_dtype": "float32",
        "mapper_dtype": "float32",
        "allow_tf32": bool(allow_tf32),
        "solver": solver,
        # FP32 streaming reductions are order-sensitive, so the chunk size is
        # part of the fitted-weight identity even though the closed-form
        # equation is mathematically unchanged.
        "statistics_chunk_tokens": int(chunk_tokens),
        "torch_version": torch.__version__,
        "feature_order": "selected_layer_rank,kv_head,head_dim",
        "output_layout": "target_kv_head,target_head_dim",
        "multioutput_solve": (
            "joint 1024-channel RHS; mathematically equivalent to eight independent "
            "128-channel target-head solves"
        ),
        "selection_sha256": selection_sha256,
        "activation_inventory_digest": activation_inventory_digest,
    }
    return _canonical_sha256(definition), definition


def _analysis_device(requested: str, compute_device: torch.device) -> torch.device:
    return compute_device if requested == "compute" else torch.device(requested)


@torch.no_grad()
def _fit_cache_type(
    factory: MMapLayerChunkFactory,
    *,
    ridge_lambda: float,
    compute_device: torch.device,
    solver: str,
    analysis_device: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Fit one independent K or V map and compute FP64 train diagnostics."""

    started = time.perf_counter()
    solution, statistics = fit_centered_ridge_streaming(
        factory,
        ridge_lambda,
        compute_device=compute_device,
        allow_tf32=False,
        solver=solver,  # type: ignore[arg-type]
    )
    fit_seconds = time.perf_counter() - started
    tensors = (statistics.mean_x, statistics.mean_y, statistics.xtx, statistics.xty)
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise FloatingPointError("ridge sufficient statistics contain NaN or Inf")
    if not bool(torch.isfinite(solution.weight).all().item()) or not bool(
        torch.isfinite(solution.bias).all().item()
    ):
        raise FloatingPointError("ridge solution contains NaN or Inf")

    diag_device = _analysis_device(analysis_device, compute_device)
    diagnostics_started = time.perf_counter()
    residual = normal_equation_relative_residual(
        statistics, solution, analysis_device=diag_device
    )
    r2_accumulator = R2Accumulator(analysis_device=diag_device)
    for x, y in factory():
        prediction = solution.predict(x)
        r2_accumulator.update(y, prediction)
    r2 = r2_accumulator.finalize()
    diagnostic_seconds = time.perf_counter() - diagnostics_started

    gram_diagonal = statistics.xtx.diagonal().to(dtype=torch.float64)
    diagnostics = {
        "n_samples": statistics.n_samples,
        "source_features": statistics.source_features,
        "target_features": statistics.target_features,
        "ridge_lambda": float(ridge_lambda),
        "gram_normalized_by_n": False,
        "feature_standardization": False,
        "bias_regularized": False,
        "statistics_dtype": "float32",
        "solution_dtype": "float32",
        "analysis_dtype": "float64",
        "allow_tf32": False,
        "solver": solution.solver,
        "cholesky_info": 0 if solution.solver == "cholesky" else None,
        "normal_equation_relative_residual_fp64": residual,
        "train_r2_uniform_channel_average_fp64": r2.uniform_average,
        "train_r2_global_fp64": r2.global_r2,
        "train_r2_flattened_global_fp64": r2.flattened_global_r2,
        "train_mse_fp64": r2.mse,
        "train_global_sse_fp64": r2.global_sse,
        "train_global_sst_fp64": r2.global_sst,
        "regularized_gram_diagonal_min_fp64": float(
            (gram_diagonal.min() + ridge_lambda).item()
        ),
        "regularized_gram_diagonal_max_fp64": float(
            (gram_diagonal.max() + ridge_lambda).item()
        ),
        "fit_seconds": fit_seconds,
        "diagnostic_seconds": diagnostic_seconds,
    }
    return (
        solution.weight.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        solution.bias.detach().to(device="cpu", dtype=torch.float32).contiguous(),
        diagnostics,
    )


def validate_completed_layer(
    output_directory: str | Path,
    target_layer: int,
    *,
    fit_fingerprint: str,
    selected_layers: Sequence[int],
    architecture: FitArchitecture,
    verify_payload_checksum: bool = True,
) -> dict[str, Any] | None:
    """Return a committed layer sidecar, or ``None`` when no checkpoint exists."""

    payload = layer_mapper_path(output_directory, target_layer)
    sidecar = payload.with_suffix(".json")
    if not payload.exists() and not sidecar.exists():
        return None
    if not sidecar.exists():
        # A payload without its commit marker is an interrupted, replaceable
        # checkpoint.  It is deliberately not accepted on resume.
        return None
    if not payload.exists():
        raise ValueError(f"committed mapper sidecar has no payload: {sidecar}")
    raw = _strict_json(sidecar)
    if raw.get("target_layer") != target_layer:
        raise ValueError(f"{sidecar}: target layer does not match filename")
    if tuple(raw.get("source_layers_rank_order", ())) != tuple(selected_layers):
        raise ValueError(f"{sidecar}: selected source layers do not match current manifest")
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("fit_fingerprint") != fit_fingerprint:
        raise ValueError(f"{sidecar}: fit fingerprint does not match current run")
    if raw.get("dtype") != "float32":
        raise ValueError(f"{sidecar}: production mapper is not FP32")
    if int(raw.get("source_heads", -1)) != architecture.kv_heads:
        raise ValueError(f"{sidecar}: KV-head count mismatch")
    if int(raw.get("head_dim", -1)) != architecture.head_dim:
        raise ValueError(f"{sidecar}: head dimension mismatch")
    expected_weight = 2 * architecture.source_features * architecture.target_features
    expected_bias = 2 * architecture.target_features
    if int(raw.get("weight_numel", -1)) != expected_weight:
        raise ValueError(f"{sidecar}: mapper weight count mismatch")
    if int(raw.get("bias_numel", -1)) != expected_bias:
        raise ValueError(f"{sidecar}: mapper bias count mismatch")
    if verify_payload_checksum and sha256_file(payload) != raw.get("sha256"):
        raise ValueError(f"{payload}: mapper SHA256 mismatch")
    return raw


def fit_one_target_layer(
    *,
    target_layer: int,
    selected_layers: Sequence[int],
    source_activation_root: str | Path,
    target_activation_root: str | Path,
    output_directory: str | Path,
    architecture: FitArchitecture,
    ridge_lambda: float,
    chunk_tokens: int,
    compute_device: str | torch.device,
    solver: str,
    analysis_device: str,
    fit_fingerprint: str,
    selection_sha256: str,
    resume_components: bool = True,
) -> dict[str, Any]:
    """Fit K and V for one target layer and atomically publish one checkpoint."""

    if len(selected_layers) != architecture.selected_source_layers:
        raise ValueError("selected-layer count does not match fit architecture")
    device = torch.device(compute_device)
    cache_results: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    diagnostics: dict[str, Any] = {}
    component_artifacts: dict[str, Any] = {}
    layer_started = time.perf_counter()
    for cache_type in ("k_content", "v"):
        restored = _load_mapper_component(
            output_directory,
            target_layer=target_layer,
            cache_type=cache_type,
            selected_layers=selected_layers,
            architecture=architecture,
            fit_fingerprint=fit_fingerprint,
        )
        if restored is not None:
            if not resume_components:
                raise FileExistsError(
                    f"layer {target_layer} {cache_type} component exists and resume is disabled"
                )
            weight, bias, cache_diagnostics = restored
            key = "key" if cache_type == "k_content" else "value"
            cache_results[key] = (weight, bias)
            diagnostics[key] = cache_diagnostics
            component = component_mapper_path(output_directory, target_layer, cache_type)
            component_artifacts[key] = {
                "path": str(component.resolve()),
                "sha256": sha256_file(component),
                "resumed": True,
            }
            continue
        source_paths = [
            activation_path(source_activation_root, cache_type, source_layer)
            for source_layer in selected_layers
        ]
        target_path = activation_path(target_activation_root, cache_type, target_layer)
        with MMapLayerChunkFactory(
            source_paths,
            target_path,
            chunk_samples=chunk_tokens,
            verify_checksum=False,
        ) as factory:
            expected_shape = (
                architecture.n_samples,
                architecture.kv_heads,
                architecture.head_dim,
            )
            if factory.activation_shape != expected_shape:
                raise ValueError(
                    f"layer {target_layer} {cache_type}: activation shape "
                    f"{factory.activation_shape}, expected {expected_shape}"
                )
            if factory.source_features != architecture.source_features:
                raise AssertionError("source design width is not the configured width")
            if factory.target_features != architecture.target_features:
                raise AssertionError("target design width is not the configured width")
            weight, bias, cache_diagnostics = _fit_cache_type(
                factory,
                ridge_lambda=ridge_lambda,
                compute_device=device,
                solver=solver,
                analysis_device=analysis_device,
            )
        key = "key" if cache_type == "k_content" else "value"
        cache_results[key] = (weight, bias)
        diagnostics[key] = cache_diagnostics
        component_sidecar = _save_mapper_component(
            output_directory,
            target_layer=target_layer,
            cache_type=cache_type,
            selected_layers=selected_layers,
            weight=weight,
            bias=bias,
            fit_fingerprint=fit_fingerprint,
            diagnostics=cache_diagnostics,
        )
        component = component_mapper_path(output_directory, target_layer, cache_type)
        component_artifacts[key] = {
            "path": str(component.resolve()),
            "sha256": component_sidecar["sha256"],
            "resumed": False,
        }
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mapper = LayerMapper(
        source_layers=tuple(int(value) for value in selected_layers),
        key_weight=cache_results["key"][0],
        key_bias=cache_results["key"][1],
        value_weight=cache_results["value"][0],
        value_bias=cache_results["value"][1],
    )
    mapper.validate(source_heads=architecture.kv_heads, head_dim=architecture.head_dim)
    layer_metadata = {
        "schema_version": SCHEMA_VERSION,
        "fit_fingerprint": fit_fingerprint,
        "selection_manifest_sha256": selection_sha256,
        "feature_order": "selected_layer_rank,kv_head,head_dim",
        "selected_source_layers_rank_order": list(map(int, selected_layers)),
        "ridge_lambda_added_directly_to_unnormalized_gram": ridge_lambda,
        "standardize_features": False,
        "regularize_bias": False,
        "forward_activation_dtype": "bfloat16",
        "covariance_dtype": "float32",
        "mapper_dtype": "float32",
        "analysis_dtype": "float64",
        "allow_tf32": False,
        "diagnostics": diagnostics,
        "component_checkpoints": component_artifacts,
        "layer_wall_seconds": time.perf_counter() - layer_started,
    }
    payload = save_layer_mapper(
        Path(output_directory) / "layers",
        target_layer,
        mapper,
        metadata=layer_metadata,
        source_heads=architecture.kv_heads,
        head_dim=architecture.head_dim,
    )
    committed = validate_completed_layer(
        output_directory,
        target_layer,
        fit_fingerprint=fit_fingerprint,
        selected_layers=selected_layers,
        architecture=architecture,
        verify_payload_checksum=True,
    )
    assert committed is not None
    return {
        "target_layer": target_layer,
        "payload": str(payload),
        "sha256": committed["sha256"],
        "metadata": layer_metadata,
    }


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _worker_entry(job: WorkerJob) -> None:
    """Multiprocessing entry point; always writes a structured result file."""

    result_path = Path(job.run_directory) / "workers" / f"worker_{job.rank:02d}.json"
    events_path = Path(job.run_directory) / "workers" / f"worker_{job.rank:02d}.jsonl"
    result: dict[str, Any] = {
        "rank": job.rank,
        "device": job.device,
        "assigned_layers": list(job.target_layers),
        "completed": [],
        "skipped": [],
        "errors": [],
    }
    try:
        torch.set_num_threads(job.cpu_threads)
        device = torch.device(job.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        for target_layer in job.target_layers:
            event_base = {
                "target_layer": target_layer,
                "device": job.device,
            }
            try:
                committed = validate_completed_layer(
                    job.output_directory,
                    target_layer,
                    fit_fingerprint=job.fit_fingerprint,
                    selected_layers=job.selected_layers[target_layer],
                    architecture=job.architecture,
                    verify_payload_checksum=True,
                )
                if committed is not None:
                    if not job.resume:
                        raise FileExistsError(
                            f"layer {target_layer} is already committed and resume is disabled"
                        )
                    result["skipped"].append(target_layer)
                    _append_jsonl(
                        events_path,
                        {
                            **event_base,
                            "event": "resume-skip",
                            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        },
                    )
                    continue
                _append_jsonl(
                    events_path,
                    {
                        **event_base,
                        "event": "fit-start",
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    },
                )
                fitted = fit_one_target_layer(
                    target_layer=target_layer,
                    selected_layers=job.selected_layers[target_layer],
                    source_activation_root=job.source_activation_root,
                    target_activation_root=job.target_activation_root,
                    output_directory=job.output_directory,
                    architecture=job.architecture,
                    ridge_lambda=job.ridge_lambda,
                    chunk_tokens=job.chunk_tokens,
                    compute_device=device,
                    solver=job.solver,
                    analysis_device=job.analysis_device,
                    fit_fingerprint=job.fit_fingerprint,
                    selection_sha256=job.selection_sha256,
                    resume_components=job.resume,
                )
                result["completed"].append(target_layer)
                _append_jsonl(
                    events_path,
                    {
                        **event_base,
                        "event": "fit-complete",
                        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "sha256": fitted["sha256"],
                    },
                )
            except BaseException as exc:
                failure = {
                    **event_base,
                    "event": "fit-error",
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "fit_fingerprint": job.fit_fingerprint,
                }
                cholesky_info = getattr(exc, "cholesky_info", None)
                if cholesky_info is not None:
                    # The production runner never silently changes lambda or
                    # falls back to another solver.  This persisted LAPACK/
                    # cuSOLVER info is what authorizes a later, explicit
                    # diagnostic rerun with --solver lstsq.
                    failure["cholesky_info"] = int(cholesky_info)
                atomic_write_json(
                    Path(job.output_directory)
                    / "failures"
                    / f"layer_{target_layer:03d}.json",
                    failure,
                )
                result["errors"].append(failure)
                _append_jsonl(events_path, failure)
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    except BaseException as exc:
        result["worker_fatal_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write_json(result_path, result)


def collect_layer_entries(
    output_directory: str | Path,
    *,
    selected_layers: Sequence[Sequence[int]],
    architecture: FitArchitecture,
    fit_fingerprint: str,
    target_layers: Sequence[int],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for target_layer in target_layers:
        sidecar = validate_completed_layer(
            output_directory,
            target_layer,
            fit_fingerprint=fit_fingerprint,
            selected_layers=selected_layers[target_layer],
            architecture=architecture,
            verify_payload_checksum=True,
        )
        if sidecar is None:
            raise ValueError(f"target layer {target_layer} has no committed mapper")
        payload = layer_mapper_path(output_directory, target_layer)
        entries.append(
            {
                "target_layer": target_layer,
                "source_layers_rank_order": list(selected_layers[target_layer]),
                "path": str(payload.resolve()),
                "sha256": sidecar["sha256"],
                "sidecar_path": str(payload.with_suffix(".json").resolve()),
                "sidecar_sha256": sha256_file(payload.with_suffix(".json")),
                "weight_numel": sidecar["weight_numel"],
                "bias_numel": sidecar["bias_numel"],
                "diagnostics": sidecar["metadata"]["diagnostics"],
            }
        )
    return entries


def publish_complete_manifest(
    output_directory: str | Path,
    *,
    architecture: FitArchitecture,
    fit_fingerprint: str,
    fit_definition: Mapping[str, Any],
    selection_path: Path,
    selection_sha256: str,
    activation_inventory_path: Path,
    activation_inventory: Mapping[str, Any],
    selected_layers: Sequence[Sequence[int]],
    ridge_lambda: float,
    solver: str,
    selection_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify all target layers and publish the final aggregate commit marker."""

    entries = collect_layer_entries(
        output_directory,
        selected_layers=selected_layers,
        architecture=architecture,
        fit_fingerprint=fit_fingerprint,
        target_layers=range(architecture.target_layers),
    )
    weight_count = sum(int(entry["weight_numel"]) for entry in entries)
    bias_count = sum(int(entry["bias_numel"]) for entry in entries)
    if weight_count != architecture.expected_weight_count:
        raise AssertionError(
            f"aggregate weight count {weight_count}, expected {architecture.expected_weight_count}"
        )
    if bias_count != architecture.expected_bias_count:
        raise AssertionError(
            f"aggregate bias count {bias_count}, expected {architecture.expected_bias_count}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "direction": "qwen3_14b_to_32b",
        "method": "closed-form-centered-ridge-kv-cache-transfer",
        "fit_fingerprint": fit_fingerprint,
        "fit_definition": dict(fit_definition),
        "architecture": asdict(architecture),
        "ridge_lambda": ridge_lambda,
        "solver": solver,
        "allow_tf32": False,
        "weight_count": weight_count,
        "bias_count": bias_count,
        "dtype": "float32",
        "selection": {
            "path": str(selection_path.resolve()),
            "source_file_sha256_used_by_fit": selection_sha256,
            "snapshot_sha256": sha256_file(selection_path),
            "selected_source_layers_rank_order": [list(row) for row in selected_layers],
            "phase3_provenance": dict(selection_provenance or {}),
        },
        "activation_inventory": {
            "path": str(activation_inventory_path.resolve()),
            "sha256": sha256_file(activation_inventory_path),
            "inventory_digest": activation_inventory["inventory_digest"],
            "payload_checksums_verified": activation_inventory[
                "payload_checksums_verified"
            ],
        },
        "layers": entries,
    }
    destination = Path(output_directory) / "manifest.json"
    atomic_write_json(destination, manifest)
    atomic_write_json(
        Path(output_directory) / "manifest.sha256.json",
        {"path": str(destination.resolve()), "sha256": sha256_file(destination)},
    )
    return manifest


def _parse_target_layers(value: str | None, count: int) -> tuple[int, ...]:
    if value is None:
        return tuple(range(count))
    result: set[int] = set()
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            begin_text, end_text = piece.split("-", 1)
            begin, end = int(begin_text), int(end_text)
            if end < begin:
                raise ValueError(f"invalid target-layer range {piece!r}")
            result.update(range(begin, end + 1))
        else:
            result.add(int(piece))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"target layers must be within [0,{count - 1}]")
    return tuple(sorted(result))


def _parse_devices(value: str) -> tuple[str, ...]:
    pieces = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not pieces:
        raise ValueError("--devices cannot be empty")
    if pieces == ("cpu",):
        return pieces
    devices = tuple(piece if piece.startswith("cuda:") else f"cuda:{int(piece)}" for piece in pieces)
    if len(set(devices)) != len(devices):
        raise ValueError("--devices contains duplicates")
    for value in devices:
        index = torch.device(value).index
        if index is None or index >= torch.cuda.device_count():
            raise ValueError(
                f"requested {value}, but torch reports {torch.cuda.device_count()} CUDA devices"
            )
    return devices


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--source-activations", type=Path)
    parser.add_argument("--target-activations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--target-layers", help="comma/range subset, e.g. 0-7,16")
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--ridge-lambda", type=float)
    parser.add_argument("--solver", choices=("cholesky", "lstsq"), default="cholesky")
    parser.add_argument(
        "--analysis-device",
        default="compute",
        help="compute (same GPU), cpu, or an explicit torch device for FP64 diagnostics",
    )
    parser.add_argument("--cpu-threads-per-worker", type=int, default=2)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-payload-checksum", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_yaml(args.config)
    project_root = Path(raw.get("project_root", ROOT)).resolve()
    paper = raw["paper"]
    architecture_raw = raw["architecture"]
    runtime = raw["runtime"]
    if not bool(runtime.get("disable_tf32_for_master_fit", False)):
        raise ValueError("production config must disable TF32 for the master fit")
    dtype_expectations = {
        "forward_dtype": "bfloat16",
        "covariance_dtype": "float32",
        "analysis_dtype": "float64",
    }
    for field, expected in dtype_expectations.items():
        if str(paper.get(field)) != expected:
            raise ValueError(
                f"paper config {field} must be {expected!r}, got {paper.get(field)!r}"
            )
    architecture = FitArchitecture(
        source_layers=int(architecture_raw["source_layers"]),
        target_layers=int(architecture_raw["target_layers"]),
        kv_heads=int(architecture_raw["source_kv_heads"]),
        head_dim=int(architecture_raw["source_head_dim"]),
        selected_source_layers=int(paper["selected_source_layers"]),
        n_samples=int(paper["calibration_sequences"])
        * (int(paper["sequence_length"]) // int(paper["position_stride"])),
    )
    if int(architecture_raw["target_kv_heads"]) != architecture.kv_heads:
        raise ValueError("source and target KV-head counts must match")
    if int(architecture_raw["target_head_dim"]) != architecture.head_dim:
        raise ValueError("source and target head dimensions must match")
    ridge_lambda = (
        float(args.ridge_lambda)
        if args.ridge_lambda is not None
        else float(paper["ridge_lambda"])
    )
    chunk_tokens = (
        int(args.chunk_tokens)
        if args.chunk_tokens is not None
        else int(runtime["statistics_chunk_tokens"])
    )
    if chunk_tokens <= 0:
        raise ValueError("chunk tokens must be positive")
    direction = str(raw.get("direction", "qwen3_14b_to_32b"))
    paths = {
        "selection": args.selection
        or project_root / "artifacts/selection" / direction / "selected_layers.json",
        "source_activations": args.source_activations
        or project_root / "data/activations/qwen3_14b/train",
        "target_activations": args.target_activations
        or project_root / "data/activations/qwen3_32b/train",
        "output": args.output
        or project_root / "artifacts/mappers" / direction / "k8_lambda0.01",
    }
    paths = {name: _resolve(project_root, path) for name, path in paths.items()}
    return {
        "raw": raw,
        "project_root": project_root,
        "architecture": architecture,
        "ridge_lambda": ridge_lambda,
        "chunk_tokens": chunk_tokens,
        "paths": paths,
    }


def _ensure_json_snapshot(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        existing = _strict_json(path)
        if _canonical_sha256(existing) != _canonical_sha256(value):
            raise ValueError(f"existing immutable input snapshot disagrees: {path}")
        return
    atomic_write_json(path, value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configuration = _configuration(args)
    project_root: Path = configuration["project_root"]
    architecture: FitArchitecture = configuration["architecture"]
    ridge_lambda: float = configuration["ridge_lambda"]
    chunk_tokens: int = configuration["chunk_tokens"]
    paths: dict[str, Path] = configuration["paths"]
    assert_paper_fit(architecture, ridge_lambda)
    if args.skip_payload_checksum:
        raise ValueError("production runs may not skip activation payload checksums")
    target_layers = _parse_target_layers(args.target_layers, architecture.target_layers)
    devices = _parse_devices(args.devices)
    selected_layers = load_selected_layers(paths["selection"], architecture)
    selection_sha256 = sha256_file(paths["selection"])

    run = make_run_directory(project_root, "p4", "fit_mapper")
    record_config = {
        "config_path": str(args.config.resolve()),
        "selection": str(paths["selection"].resolve()),
        "source_activations": str(paths["source_activations"].resolve()),
        "target_activations": str(paths["target_activations"].resolve()),
        "output": str(paths["output"].resolve()),
        "architecture": asdict(architecture),
        "ridge_lambda": ridge_lambda,
        "chunk_tokens": chunk_tokens,
        "solver": args.solver,
        "analysis_device": args.analysis_device,
        "devices": list(devices),
        "target_layers": list(target_layers),
        "resume": not args.no_resume,
        "payload_checksums": not args.skip_payload_checksum,
        "dry_run": args.dry_run,
    }
    record_run_start(run, list(sys.argv if argv is None else [sys.argv[0], *argv]), record_config)
    output = paths["output"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        inventory = collect_activation_inventory(
            paths["source_activations"],
            paths["target_activations"],
            architecture,
            verify_payload_checksum=not args.skip_payload_checksum,
        )
        selection_provenance = validate_selection_provenance(
            paths["selection"],
            inventory,
            require_formal_protocol=True,
        )
        fit_fingerprint, fit_definition = make_fit_fingerprint(
            architecture=architecture,
            ridge_lambda=ridge_lambda,
            allow_tf32=False,
            solver=args.solver,
            chunk_tokens=chunk_tokens,
            selection_sha256=selection_sha256,
            activation_inventory_digest=inventory["inventory_digest"],
        )
        inputs_directory = output / "inputs"
        selection_snapshot = inputs_directory / "selected_layers.json"
        inventory_snapshot = inputs_directory / "activation_inventory.json"
        fit_snapshot = inputs_directory / "fit_definition.json"
        _ensure_json_snapshot(selection_snapshot, _strict_json(paths["selection"]))
        _ensure_json_snapshot(inventory_snapshot, inventory)
        _ensure_json_snapshot(fit_snapshot, fit_definition)
        assignments = assign_target_layers(target_layers, devices)
        state = {
            "schema_version": SCHEMA_VERSION,
            "status": "dry-run" if args.dry_run else "running",
            "run_id": run.run_id,
            "fit_fingerprint": fit_fingerprint,
            "target_layers": list(target_layers),
            "assignments": {key: list(value) for key, value in assignments.items()},
            "selection_provenance": selection_provenance,
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_write_json(output / "run_state.json", state)
        atomic_write_json(run.path / "preflight.json", {**state, "inventory": inventory})
        if args.dry_run:
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0

        jobs = [
            WorkerJob(
                rank=rank,
                device=device,
                target_layers=assignments[device],
                selected_layers=selected_layers,
                source_activation_root=str(paths["source_activations"]),
                target_activation_root=str(paths["target_activations"]),
                output_directory=str(output),
                run_directory=str(run.path),
                fit_fingerprint=fit_fingerprint,
                selection_sha256=selection_sha256,
                architecture=architecture,
                ridge_lambda=ridge_lambda,
                chunk_tokens=chunk_tokens,
                solver=args.solver,
                analysis_device=args.analysis_device,
                resume=not args.no_resume,
                cpu_threads=args.cpu_threads_per_worker,
            )
            for rank, device in enumerate(devices)
            if assignments[device]
        ]
        context = mp.get_context("spawn")
        processes = [context.Process(target=_worker_entry, args=(job,)) for job in jobs]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join()
        except KeyboardInterrupt:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
            state.update(
                status="interrupted",
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            atomic_write_json(output / "run_state.json", state)
            atomic_write_json(run.path / "result.json", state)
            raise

        worker_results = []
        for job, process in zip(jobs, processes, strict=True):
            result_path = run.path / "workers" / f"worker_{job.rank:02d}.json"
            if not result_path.exists():
                worker_results.append(
                    {
                        "rank": job.rank,
                        "device": job.device,
                        "worker_fatal_error": f"no result file; exit code {process.exitcode}",
                    }
                )
            else:
                worker_results.append(_strict_json(result_path))
        errors = [
            result
            for result in worker_results
            if result.get("errors") or result.get("worker_fatal_error")
        ]
        if errors:
            state.update(
                status="failed",
                finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                worker_results=worker_results,
            )
            atomic_write_json(output / "run_state.json", state)
            atomic_write_json(run.path / "result.json", state)
            append_reproduction_log(
                project_root,
                heading=f"FIT-FAIL-{run.run_id}",
                bullets=[
                    f"Production mapper fit stopped with {len(errors)} worker error group(s).",
                    f"Run evidence: `{run.path}`; committed per-layer checkpoints remain resumable.",
                ],
            )
            return 1

        requested_entries = collect_layer_entries(
            output,
            selected_layers=selected_layers,
            architecture=architecture,
            fit_fingerprint=fit_fingerprint,
            target_layers=target_layers,
        )
        if target_layers == tuple(range(architecture.target_layers)):
            manifest = publish_complete_manifest(
                output,
                architecture=architecture,
                fit_fingerprint=fit_fingerprint,
                fit_definition=fit_definition,
                selection_path=selection_snapshot,
                selection_sha256=selection_sha256,
                activation_inventory_path=inventory_snapshot,
                activation_inventory=inventory,
                selected_layers=selected_layers,
                ridge_lambda=ridge_lambda,
                solver=args.solver,
                selection_provenance=selection_provenance,
            )
            status = "complete"
        else:
            manifest = {"layers": requested_entries}
            status = "partial"
        state.update(
            status=status,
            finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            committed_layers=[entry["target_layer"] for entry in requested_entries],
            worker_results=worker_results,
        )
        atomic_write_json(output / "run_state.json", state)
        atomic_write_json(
            run.path / "result.json",
            {
                **state,
                "manifest": str((output / "manifest.json").resolve())
                if status == "complete"
                else None,
            },
        )
        append_reproduction_log(
            project_root,
            heading=f"FIT-{status.upper()}-{run.run_id}",
            bullets=[
                f"Mapper fit status: {status}; committed {len(requested_entries)} target layer(s).",
                "Fit used centered, unnormalized FP32 sufficient statistics; lambda was added "
                "directly, bias was unregularized, and TF32 was disabled.",
                f"Run evidence: `{run.path}`; artifact directory: `{output}`.",
            ],
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "fit_fingerprint": fit_fingerprint,
                    "committed_layers": len(requested_entries),
                    "weight_count": manifest.get("weight_count"),
                    "bias_count": manifest.get("bias_count"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as exc:
        failure = {
            "status": "preflight-or-driver-failed",
            "run_id": run.run_id,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_write_json(run.path / "result.json", failure)
        atomic_write_json(output / "run_state.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
