#!/usr/bin/env python3
"""Fit and publish an isolated head-wise k4/k20 shrinkage mapper.

The script intentionally does not import the reverse experiment.  Its only
dependencies on that experiment are immutable input files whose identities
are verified and recorded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def artifact(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": (
            str(resolved.relative_to(relative_to.resolve()))
            if relative_to is not None
            else str(resolved)
        ),
        "nbytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def load_config() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    configured_outputs = config.get("outputs", {})
    if not isinstance(configured_outputs, Mapping):
        raise ValueError("config outputs are malformed")
    for value in configured_outputs.values():
        Path(str(value)).resolve().relative_to(ROOT)
    return config


def code_inventory() -> dict[str, Any]:
    paths = [ROOT / "README.md", CONFIG_PATH]
    paths.extend(sorted((ROOT / "src").glob("*.py")))
    paths.extend(sorted((ROOT / "tests").glob("*.py")))
    entries = [artifact(path, relative_to=ROOT) for path in paths]
    return {"files": entries, "digest": canonical_sha256(entries)}


def safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        size_raw = handle.read(8)
        if len(size_raw) != 8:
            raise ValueError(f"truncated safetensors header: {path}")
        size = struct.unpack("<Q", size_raw)[0]
        if size <= 2 or size > 16 * 1024 * 1024:
            raise ValueError(f"invalid safetensors header length {size}: {path}")
        raw = handle.read(size)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"malformed safetensors header: {path}")
    return value


def _expected_shapes(k: int) -> dict[str, list[int]]:
    features = k * 8 * 128
    return {
        "key_weight": [features, 1024],
        "key_bias": [1024],
        "value_weight": [features, 1024],
        "value_bias": [1024],
    }


def _selection_rows(root: Path) -> list[list[int]]:
    value = read_json(root / "inputs" / "selected_layers.json")
    rows = value.get("selected_source_layers_rank_order")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError(f"selection does not contain 40 rows: {root}")
    return [[int(item) for item in row] for row in rows]


def _validate_mapper_root(root: Path, *, k: int, deep_hash: bool) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    sidecar_path = root / "manifest.sha256.json"
    manifest = read_json(manifest_path)
    marker = read_json(sidecar_path)
    manifest_hash = sha256_file(manifest_path)
    if marker.get("sha256") != manifest_hash:
        raise ValueError(f"manifest marker mismatch: {root}")
    expected_architecture = {
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "selected_source_layers": k,
        "n_samples": 128000,
    }
    if manifest.get("status") != "complete":
        raise ValueError(f"upstream mapper is incomplete: {root}")
    if manifest.get("direction") != "qwen3_32b_to_14b":
        raise ValueError(f"upstream mapper direction changed: {root}")
    if manifest.get("dtype") != "float32":
        raise ValueError(f"upstream mapper dtype changed: {root}")
    if manifest.get("architecture") != expected_architecture:
        raise ValueError(f"upstream architecture changed: {root}")
    rows = _selection_rows(root)
    if any(len(row) != k or len(set(row)) != k for row in rows):
        raise ValueError(f"invalid selection cardinality: {root}")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ValueError(f"upstream mapper layer inventory changed: {root}")
    expected_shapes = _expected_shapes(k)
    payloads: list[dict[str, Any]] = []
    for target_layer, entry in enumerate(layers):
        if int(entry.get("target_layer", -1)) != target_layer:
            raise ValueError(f"unordered mapper layer {target_layer}: {root}")
        if entry.get("source_layers_rank_order") != rows[target_layer]:
            raise ValueError(f"mapper/selection row mismatch at {target_layer}: {root}")
        stem = f"layer_{target_layer:03d}"
        tensor_path = root / "layers" / f"{stem}.safetensors"
        json_path = root / "layers" / f"{stem}.json"
        header = safetensors_header(tensor_path)
        tensor_header = {key: value for key, value in header.items() if key != "__metadata__"}
        if set(tensor_header) != set(expected_shapes):
            raise ValueError(f"tensor keys changed at {target_layer}: {root}")
        for name, shape in expected_shapes.items():
            value = tensor_header[name]
            if value.get("dtype") != "F32" or value.get("shape") != shape:
                raise ValueError(f"tensor shape/dtype changed for {name}: {tensor_path}")
        json_value = read_json(json_path)
        if json_value.get("sha256") != entry.get("sha256"):
            raise ValueError(f"payload hashes disagree at layer {target_layer}: {root}")
        if json_value.get("tensor_shapes") != expected_shapes:
            raise ValueError(f"sidecar shapes changed at layer {target_layer}: {root}")
        sidecar_hash = sha256_file(json_path)
        if sidecar_hash != entry.get("sidecar_sha256"):
            raise ValueError(f"layer sidecar hash mismatch at {target_layer}: {root}")
        if deep_hash and sha256_file(tensor_path) != entry.get("sha256"):
            raise ValueError(f"layer payload hash mismatch at {target_layer}: {root}")
        payloads.append(
            {
                "target_layer": target_layer,
                "payload": str(tensor_path.resolve()),
                "payload_sha256": str(entry.get("sha256")),
                "sidecar": str(json_path.resolve()),
                "sidecar_sha256": sidecar_hash,
            }
        )
    return {
        "root": str(root),
        "k": k,
        "manifest": artifact(manifest_path),
        "manifest_marker": artifact(sidecar_path),
        "fit_fingerprint": manifest.get("fit_fingerprint"),
        "rows": rows,
        "payloads_deep_hashed": deep_hash,
        "payload_inventory_digest": canonical_sha256(payloads),
    }


def _activation_metadata(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if value.get("dtype") != "bfloat16" or value.get("storage_dtype") != "uint16":
        raise ValueError(f"activation storage dtype changed: {path}")
    if value.get("shape") != [12800, 8, 128]:
        raise ValueError(f"activation shape changed: {path}")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"activation provenance missing: {path}")
    expected = {
        "split": "heldout",
        "sequences": 50,
        "stride": 4,
        "stride_start": 0,
    }
    for field, observed in expected.items():
        if provenance.get(field) != observed:
            raise ValueError(f"activation provenance {field} changed: {path}")
    return value


def _needed_activation_files(
    config: Mapping[str, Any], rows: Sequence[Sequence[int]]
) -> list[tuple[str, str, int, Path, Path]]:
    upstream = config["upstream"]
    source_root = Path(str(upstream["heldout_source_activations"]))
    target_root = Path(str(upstream["heldout_target_activations"]))
    source_layers = sorted({layer for row in rows for layer in row})
    result: list[tuple[str, str, int, Path, Path]] = []
    for component in ("k_content", "v"):
        for layer in source_layers:
            stem = source_root / component / f"layer_{layer:03d}"
            result.append(("source", component, layer, stem.with_suffix(".json"), stem.with_suffix(".bf16.bin")))
        for layer in range(40):
            stem = target_root / component / f"layer_{layer:03d}"
            result.append(("target", component, layer, stem.with_suffix(".json"), stem.with_suffix(".bf16.bin")))
    return result


def build_split_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    token_path = Path(str(config["upstream"]["heldout_tokens"])).resolve()
    tokens = np.load(token_path, allow_pickle=False)
    if tokens.dtype != np.dtype("int32") or tokens.shape != (50, 1024):
        raise ValueError(f"heldout token matrix changed: {tokens.dtype} {tokens.shape}")
    row_hashes = [
        hashlib.sha256(np.asarray(row, dtype="<i4").tobytes(order="C")).hexdigest()
        for row in tokens
    ]
    ordered = sorted(range(50), key=lambda index: (row_hashes[index], index))
    locked = ordered[:10]
    tuning = ordered[10:]
    folds: list[list[int]] = [[] for _ in range(5)]
    for rank, sequence_id in enumerate(tuning):
        folds[rank % 5].append(sequence_id)
    if sorted(locked + [value for fold in folds for value in fold]) != list(range(50)):
        raise AssertionError("split is not a partition")
    if len(locked) != 10 or any(len(fold) != 8 for fold in folds):
        raise AssertionError("split cardinalities changed")
    return {
        "schema_version": 1,
        "rule": config["protocol"]["split_rule"],
        "token_file": artifact(token_path),
        "row_sha256": row_hashes,
        "locked_audit_sequence_ids": locked,
        "tuning_sequence_ids": tuning,
        "cv_folds": [
            {"fold": fold, "validation_sequence_ids": ids}
            for fold, ids in enumerate(folds)
        ],
        "locked_audit_used_for_alpha": False,
    }


def validate_inputs(*, deep_hash: bool) -> dict[str, Any]:
    config = load_config()
    k20_root = Path(str(config["upstream"]["k20_mapper"]))
    k4_root = Path(str(config["upstream"]["k4_mapper"]))
    k20 = _validate_mapper_root(k20_root, k=20, deep_hash=deep_hash)
    k4 = _validate_mapper_root(k4_root, k=4, deep_hash=deep_hash)
    mismatches = [
        layer
        for layer, (row20, row4) in enumerate(zip(k20["rows"], k4["rows"], strict=True))
        if row20[:4] != row4
    ]
    if mismatches:
        raise ValueError(f"k4 selection cannot be embedded in k20: {mismatches}")
    split = build_split_manifest(config)
    token_sha = split["token_file"]["sha256"]
    activation_entries = []
    for role, component, layer, metadata_path, payload_path in _needed_activation_files(
        config, k20["rows"]
    ):
        metadata = _activation_metadata(metadata_path)
        if metadata["provenance"].get("token_sha256") != token_sha:
            raise ValueError(f"activation token identity mismatch: {metadata_path}")
        expected_bytes = 12800 * 8 * 128 * 2
        if payload_path.stat().st_size != expected_bytes or metadata.get("nbytes") != expected_bytes:
            raise ValueError(f"activation byte count changed: {payload_path}")
        if deep_hash and sha256_file(payload_path) != metadata.get("sha256"):
            raise ValueError(f"activation payload hash mismatch: {payload_path}")
        activation_entries.append(
            {
                "role": role,
                "component": component,
                "layer": layer,
                "metadata": artifact(metadata_path),
                "payload": str(payload_path.resolve()),
                "payload_nbytes": payload_path.stat().st_size,
                "payload_sha256": metadata.get("sha256"),
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "completed_utc": utc_now(),
        "deep_hash": deep_hash,
        "config": artifact(CONFIG_PATH),
        "code": code_inventory(),
        "k20": {key: value for key, value in k20.items() if key != "rows"},
        "k4": {key: value for key, value in k4.items() if key != "rows"},
        "selection_embedding": {
            "status": "passed",
            "checked_rows": 40,
            "rule": "k4_row == k20_row[:4]",
            "mismatches": mismatches,
            "k20_rows_digest": canonical_sha256(k20["rows"]),
            "k4_rows_digest": canonical_sha256(k4["rows"]),
        },
        "split": split,
        "activations": {
            "entries": len(activation_entries),
            "payloads_deep_hashed": deep_hash,
            "inventory_digest": canonical_sha256(activation_entries),
        },
    }


def publish_preflight(*, deep_hash: bool) -> dict[str, Any]:
    config = load_config()
    destination = Path(str(config["outputs"]["preflight"])).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite preflight: {destination}")
    temporary = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    try:
        result = validate_inputs(deep_hash=deep_hash)
        atomic_write_json(temporary / "preflight.json", result)
        atomic_write_json(temporary / "split_manifest.json", result["split"])
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "files": [
                artifact(temporary / "preflight.json", relative_to=temporary),
                artifact(temporary / "split_manifest.json", relative_to=temporary),
            ],
        }
        atomic_write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return read_json(destination / "preflight.json")


def fit_alpha(e_dot_d: Any, d2: Any, sequence_ids: Sequence[int]) -> Any:
    """Analytic per-head alpha, aggregating channels and selected sequences."""
    import numpy as np

    ids = np.asarray(sequence_ids, dtype=np.int64)
    numerator = np.asarray(e_dot_d, dtype=np.float64)[ids].sum(axis=(0, 2))
    denominator = np.asarray(d2, dtype=np.float64)[ids].sum(axis=(0, 2))
    alpha = np.zeros(8, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).tiny
    alpha[valid] = numerator[valid] / denominator[valid]
    return np.clip(alpha, 0.0, 1.0)


def _merge_target_moments(stats: Mapping[str, Any], sequence_ids: Sequence[int]) -> tuple[Any, Any, float]:
    import numpy as np

    ids = np.asarray(sequence_ids, dtype=np.int64)
    means = np.asarray(stats["target_mean"], dtype=np.float64)[ids]
    m2 = np.asarray(stats["target_m2"], dtype=np.float64)[ids]
    mean = means.mean(axis=0)
    sst = m2.sum(axis=0) + 256.0 * np.square(means - mean).sum(axis=0)
    flat_means = np.asarray(stats["flat_mean"], dtype=np.float64)[ids]
    flat_m2 = np.asarray(stats["flat_m2"], dtype=np.float64)[ids]
    flat_mean = float(flat_means.mean())
    flat_sst = float(
        flat_m2.sum() + (256 * 1024) * np.square(flat_means - flat_mean).sum()
    )
    return mean, sst, flat_sst


def metrics_for_alpha(
    stats: Mapping[str, Any], sequence_ids: Sequence[int], alpha: Any, *, per_channel: bool
) -> dict[str, Any]:
    import numpy as np

    ids = np.asarray(sequence_ids, dtype=np.int64)
    alpha_array = np.asarray(alpha, dtype=np.float64).reshape(1, 8, 1)
    e2 = np.asarray(stats["e2"], dtype=np.float64)[ids].sum(axis=0)
    ed = np.asarray(stats["e_dot_d"], dtype=np.float64)[ids].sum(axis=0)
    d2 = np.asarray(stats["d2"], dtype=np.float64)[ids].sum(axis=0)
    sse = e2 - 2.0 * alpha_array[0] * ed + np.square(alpha_array[0]) * d2
    tolerance = 1e-9 * max(1.0, float(e2.max()))
    if float(sse.min()) < -tolerance:
        raise ValueError(f"negative SSE beyond roundoff: {float(sse.min())}")
    sse = np.maximum(sse, 0.0)
    _, sst, flattened_sst = _merge_target_moments(stats, sequence_ids)
    nonconstant = sst != 0.0
    r2 = np.empty_like(sst)
    r2[nonconstant] = 1.0 - sse[nonconstant] / sst[nonconstant]
    r2[~nonconstant] = np.where(sse[~nonconstant] == 0.0, 1.0, 0.0)
    global_sse = float(sse.sum())
    global_sst = float(sst.sum())
    observations = len(ids) * 256
    result: dict[str, Any] = {
        "sequence_count": len(ids),
        "observations": observations,
        "channels": 1024,
        "alpha": np.asarray(alpha, dtype=np.float64).tolist(),
        "uniform_channel_r2_fp64": float(r2.mean()),
        "global_r2_fp64": 1.0 - global_sse / global_sst if global_sst else float(global_sse == 0.0),
        "flattened_global_r2_fp64": (
            1.0 - global_sse / flattened_sst
            if flattened_sst
            else float(global_sse == 0.0)
        ),
        "mse_fp64": global_sse / (observations * 1024),
        "global_sse_fp64": global_sse,
        "global_sst_fp64": global_sst,
        "flattened_global_sst_fp64": flattened_sst,
    }
    if per_channel:
        result["per_channel_sse_fp64"] = sse.reshape(-1).tolist()
        result["per_channel_sst_fp64"] = sst.reshape(-1).tolist()
    return result


def cross_validate_component(
    stats: Mapping[str, Any], split: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    tuning = [int(value) for value in split["tuning_sequence_ids"]]
    fold_rows = []
    oof_sse = np.zeros((8, 128), dtype=np.float64)
    for fold in split["cv_folds"]:
        validation = [int(value) for value in fold["validation_sequence_ids"]]
        validation_set = set(validation)
        training = [value for value in tuning if value not in validation_set]
        alpha = fit_alpha(stats["e_dot_d"], stats["d2"], training)
        metrics = metrics_for_alpha(stats, validation, alpha, per_channel=False)
        ids = np.asarray(validation, dtype=np.int64)
        e2 = np.asarray(stats["e2"])[ids].sum(axis=0)
        ed = np.asarray(stats["e_dot_d"])[ids].sum(axis=0)
        d2 = np.asarray(stats["d2"])[ids].sum(axis=0)
        expanded = alpha.reshape(8, 1)
        oof_sse += e2 - 2.0 * expanded * ed + np.square(expanded) * d2
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "training_sequence_ids": training,
                "validation_sequence_ids": validation,
                "alpha": alpha.tolist(),
                "validation_metrics": metrics,
            }
        )
    _, tune_sst, tune_flat_sst = _merge_target_moments(stats, tuning)
    tune_global_sse = float(oof_sse.sum())
    nonconstant = tune_sst != 0.0
    oof_r2 = np.empty_like(tune_sst)
    oof_r2[nonconstant] = 1.0 - oof_sse[nonconstant] / tune_sst[nonconstant]
    oof_r2[~nonconstant] = np.where(oof_sse[~nonconstant] == 0.0, 1.0, 0.0)
    final_alpha = fit_alpha(stats["e_dot_d"], stats["d2"], tuning)
    cv = {
        "folds": fold_rows,
        "oof": {
            "sequence_count": 40,
            "observations": 10240,
            "channels": 1024,
            "uniform_channel_r2_fp64": float(oof_r2.mean()),
            "global_r2_fp64": 1.0 - tune_global_sse / float(tune_sst.sum()),
            "flattened_global_r2_fp64": 1.0 - tune_global_sse / tune_flat_sst,
            "mse_fp64": tune_global_sse / (10240 * 1024),
            "global_sse_fp64": tune_global_sse,
            "global_sst_fp64": float(tune_sst.sum()),
            "flattened_global_sst_fp64": tune_flat_sst,
        },
        "refit": {
            "sequence_ids": tuning,
            "alpha": final_alpha.tolist(),
            "metrics": metrics_for_alpha(stats, tuning, final_alpha, per_channel=False),
        },
        "locked_audit_consulted": False,
    }
    return final_alpha, cv


def _mmap_activation(path: Path) -> Any:
    import torch

    raw = torch.from_file(
        str(path), shared=False, size=12800 * 8 * 128, dtype=torch.uint16
    )
    return raw.view(torch.bfloat16).reshape(12800, 8, 128)


def _load_mapper_layer(root: Path, target_layer: int) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(
        str(root / "layers" / f"layer_{target_layer:03d}.safetensors"),
        device="cpu",
    )


def collect_component_stats(
    *,
    source_root: Path,
    target_root: Path,
    component: str,
    target_layer: int,
    selected_layers: Sequence[int],
    weight20: Any,
    bias20: Any,
    weight4: Any,
    bias4: Any,
    device: Any,
    chunk_sequences: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    source = [
        _mmap_activation(source_root / component / f"layer_{layer:03d}.bf16.bin")
        for layer in selected_layers
    ]
    target = _mmap_activation(
        target_root / component / f"layer_{target_layer:03d}.bf16.bin"
    )
    w20 = weight20.to(device=device, dtype=torch.float32)
    b20 = bias20.to(device=device, dtype=torch.float32)
    w4 = weight4.to(device=device, dtype=torch.float32)
    b4 = bias4.to(device=device, dtype=torch.float32)
    arrays = {
        "e2": np.empty((50, 8, 128), dtype=np.float64),
        "e_dot_d": np.empty((50, 8, 128), dtype=np.float64),
        "d2": np.empty((50, 8, 128), dtype=np.float64),
        "target_mean": np.empty((50, 8, 128), dtype=np.float64),
        "target_m2": np.empty((50, 8, 128), dtype=np.float64),
        "flat_mean": np.empty((50,), dtype=np.float64),
        "flat_m2": np.empty((50,), dtype=np.float64),
    }
    observations_per_sequence = 256
    with torch.inference_mode():
        for first_sequence in range(0, 50, chunk_sequences):
            sequence_count = min(chunk_sequences, 50 - first_sequence)
            first = first_sequence * observations_per_sequence
            last = (first_sequence + sequence_count) * observations_per_sequence
            x_cpu = torch.stack([value[first:last] for value in source], dim=1).reshape(
                last - first, 20480
            )
            y = target[first:last].reshape(last - first, 1024).to(
                device=device, dtype=torch.float32
            )
            x = x_cpu.to(device=device, dtype=torch.float32)
            p20 = torch.addmm(b20, x, w20)
            p4 = torch.addmm(b4, x[:, :4096], w4)
            y64 = y.reshape(sequence_count, 256, 8, 128).to(torch.float64)
            e = (y - p4).reshape(sequence_count, 256, 8, 128).to(torch.float64)
            d = (p20 - p4).reshape(sequence_count, 256, 8, 128).to(torch.float64)
            means = y64.mean(dim=1)
            flat = y64.reshape(sequence_count, -1)
            flat_means = flat.mean(dim=1)
            sl = slice(first_sequence, first_sequence + sequence_count)
            arrays["e2"][sl] = torch.sum(e * e, dim=1).cpu().numpy()
            arrays["e_dot_d"][sl] = torch.sum(e * d, dim=1).cpu().numpy()
            arrays["d2"][sl] = torch.sum(d * d, dim=1).cpu().numpy()
            arrays["target_mean"][sl] = means.cpu().numpy()
            arrays["target_m2"][sl] = torch.sum(
                torch.square(y64 - means[:, None]), dim=1
            ).cpu().numpy()
            arrays["flat_mean"][sl] = flat_means.cpu().numpy()
            arrays["flat_m2"][sl] = torch.sum(
                torch.square(flat - flat_means[:, None]), dim=1
            ).cpu().numpy()
            del x_cpu, x, y, y64, e, d, p20, p4, means, flat, flat_means
    del w20, b20, w4, b4, source, target
    return arrays


def _driver_state() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    gpus = []
    for line in query.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        gpus.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
                "utilization_gpu_percent": int(fields[4]),
            }
        )
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    process_rows = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    return {"gpus": gpus, "compute_process_rows": process_rows}


def acquire_idle_gate(config: Mapping[str, Any], run_root: Path) -> Any:
    runtime = config["runtime"]
    lock_path = Path(str(runtime["global_gpu_lock"]))
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_handle.close()
        raise RuntimeError(f"global GPU lock is held: {lock_path}")
    samples = []
    required = int(runtime["idle_gate_samples"])
    maximum = int(runtime["idle_gate_max_memory_mib"])
    interval = int(runtime["idle_gate_interval_seconds"])
    try:
        for sample_index in range(required):
            state = _driver_state()
            idle = (
                len(state["gpus"]) == 8
                and not state["compute_process_rows"]
                and max(row["memory_used_mib"] for row in state["gpus"]) <= maximum
            )
            sample = {
                "sample": sample_index,
                "timestamp_utc": utc_now(),
                "idle": idle,
                **state,
            }
            samples.append(sample)
            atomic_write_json(
                run_root / "idle_gate.partial.json",
                {
                    "schema_version": 1,
                    "required_consecutive_samples": required,
                    "interval_seconds": interval,
                    "max_memory_mib": maximum,
                    "samples": samples,
                },
            )
            if not idle:
                raise RuntimeError(f"GPU idle gate failed at sample {sample_index}")
            if sample_index + 1 < required:
                time.sleep(interval)
        gate = {
            "schema_version": 1,
            "status": "passed",
            "global_lock": str(lock_path.resolve()),
            "lock_held": True,
            "required_consecutive_samples": required,
            "observed_consecutive_samples": len(samples),
            "interval_seconds": interval,
            "max_memory_mib": maximum,
            "samples": samples,
        }
        atomic_write_json(run_root / "idle_gate.json", gate)
        (run_root / "idle_gate.partial.json").unlink()
        return lock_handle
    except BaseException:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()
        raise


def _save_tuning_stats(path: Path, stats: Mapping[str, Any], tuning_ids: Sequence[int]) -> None:
    import numpy as np

    ids = np.asarray(tuning_ids, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            sequence_ids=ids,
            **{key: np.asarray(value)[ids] for key, value in stats.items()},
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _alpha_records(
    alpha_by_layer: Sequence[Mapping[str, Any]], split: Mapping[str, Any], preflight: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "frozen_before_locked_audit",
        "method": "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers",
        "formula": "f4 + alpha * (f20 - f4)",
        "bounds": [0.0, 1.0],
        "granularity": "target_layer,cache_component,target_kv_head",
        "fit_sequence_ids": split["tuning_sequence_ids"],
        "locked_audit_sequence_ids": split["locked_audit_sequence_ids"],
        "locked_audit_used_for_alpha": False,
        "split_digest": canonical_sha256(split),
        "input_preflight_digest": canonical_sha256(preflight),
        "layers": list(alpha_by_layer),
    }


def _fuse_component(weight20: Any, weight4: Any, alpha: Any) -> Any:
    import torch

    if tuple(weight20.shape) != (20480, 1024) or tuple(weight4.shape) != (4096, 1024):
        raise ValueError("cannot fuse unexpected mapper shapes")
    result = weight20.clone().to(dtype=torch.float32)
    for head in range(8):
        columns = slice(head * 128, (head + 1) * 128)
        value = float(alpha[head])
        result[:, columns].mul_(value)
        result[:4096, columns].add_(weight4[:, columns], alpha=1.0 - value)
    return result


def _fuse_bias(bias20: Any, bias4: Any, alpha: Any) -> Any:
    import torch

    result = bias20.clone().to(dtype=torch.float32)
    for head in range(8):
        columns = slice(head * 128, (head + 1) * 128)
        value = float(alpha[head])
        result[columns].mul_(value)
        result[columns].add_(bias4[columns], alpha=1.0 - value)
    return result


def publish_mapper(
    *, config: Mapping[str, Any], alpha_manifest: Mapping[str, Any], run_root: Path
) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    destination = Path(str(config["outputs"]["mapper"])).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite mapper: {destination}")
    staging = destination.with_name(f".{destination.name}.staging.{uuid.uuid4().hex}")
    (staging / "layers").mkdir(parents=True)
    (staging / "inputs").mkdir()
    k20_root = Path(str(config["upstream"]["k20_mapper"]))
    k4_root = Path(str(config["upstream"]["k4_mapper"]))
    rows20 = _selection_rows(k20_root)
    try:
        shutil.copy2(k20_root / "inputs" / "selected_layers.json", staging / "inputs" / "selected_layers_k20.json")
        shutil.copy2(k4_root / "inputs" / "selected_layers.json", staging / "inputs" / "selected_layers_k4.json")
        shutil.copy2(run_root / "split_manifest.json", staging / "inputs" / "split_manifest.json")
        shutil.copy2(run_root / "alpha_manifest.json", staging / "inputs" / "alpha_manifest.json")
        method_definition = {
            "schema_version": 1,
            "method": "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers",
            "formula": "f4 + alpha * (f20 - f4)",
            "alpha_bounds": [0.0, 1.0],
            "alpha_granularity": "target_layer,cache_component,target_kv_head",
            "k4_embedded_rows": "k20 feature rows 0:4096 because selection_k4 == selection_k20[:4] for all target layers",
            "output_layout": "target_kv_head,target_head_dim",
            "feature_order": "selected_layer_rank,kv_head,head_dim",
            "upstream_k20_manifest_sha256": sha256_file(k20_root / "manifest.json"),
            "upstream_k4_manifest_sha256": sha256_file(k4_root / "manifest.json"),
            "split_manifest_sha256": sha256_file(staging / "inputs" / "split_manifest.json"),
            "alpha_manifest_sha256": sha256_file(staging / "inputs" / "alpha_manifest.json"),
            "locked_audit_used_for_alpha": False,
        }
        mapper_fingerprint = canonical_sha256(method_definition)
        layer_entries = []
        payload_validation_entries = []
        for target_layer in range(40):
            tensors20 = load_file(str(k20_root / "layers" / f"layer_{target_layer:03d}.safetensors"), device="cpu")
            tensors4 = load_file(str(k4_root / "layers" / f"layer_{target_layer:03d}.safetensors"), device="cpu")
            alpha_row = alpha_manifest["layers"][target_layer]
            key_alpha = alpha_row["key"]["alpha"]
            value_alpha = alpha_row["value"]["alpha"]
            output_tensors = {
                "key_weight": _fuse_component(tensors20["key_weight"], tensors4["key_weight"], key_alpha),
                "key_bias": _fuse_bias(tensors20["key_bias"], tensors4["key_bias"], key_alpha),
                "value_weight": _fuse_component(tensors20["value_weight"], tensors4["value_weight"], value_alpha),
                "value_bias": _fuse_bias(tensors20["value_bias"], tensors4["value_bias"], value_alpha),
            }
            tensor_path = staging / "layers" / f"layer_{target_layer:03d}.safetensors"
            save_file(
                output_tensors,
                str(tensor_path),
                metadata={
                    "schema_version": "1",
                    "method": method_definition["method"],
                    "mapper_fingerprint": mapper_fingerprint,
                    "target_layer": str(target_layer),
                    "source_layers_rank_order": ",".join(map(str, rows20[target_layer])),
                },
            )
            tensor_sha = sha256_file(tensor_path)
            sidecar = {
                "schema_version": 1,
                "target_layer": target_layer,
                "source_layers_rank_order": rows20[target_layer],
                "dtype": "float32",
                "source_heads": 8,
                "head_dim": 128,
                "tensor_shapes": _expected_shapes(20),
                "weight_numel": 2 * 20480 * 1024,
                "bias_numel": 2 * 1024,
                "sha256": tensor_sha,
                "metadata": {
                    "mapper_fingerprint": mapper_fingerprint,
                    "key_alpha": key_alpha,
                    "value_alpha": value_alpha,
                    "upstream_k20_layer_sha256": sha256_file(k20_root / "layers" / f"layer_{target_layer:03d}.safetensors"),
                    "upstream_k4_layer_sha256": sha256_file(k4_root / "layers" / f"layer_{target_layer:03d}.safetensors"),
                },
            }
            sidecar_path = staging / "layers" / f"layer_{target_layer:03d}.json"
            atomic_write_json(sidecar_path, sidecar)
            entry = {
                "target_layer": target_layer,
                "path": f"layers/layer_{target_layer:03d}.safetensors",
                "sha256": tensor_sha,
                "sidecar_path": f"layers/layer_{target_layer:03d}.json",
                "sidecar_sha256": sha256_file(sidecar_path),
                "source_layers_rank_order": rows20[target_layer],
                "weight_numel": 2 * 20480 * 1024,
                "bias_numel": 2 * 1024,
            }
            layer_entries.append(entry)
            payload_validation_entries.append(
                {
                    "target_layer": target_layer,
                    "sha256": tensor_sha,
                    "tensor_shapes": _expected_shapes(20),
                }
            )
            del tensors20, tensors4, output_tensors
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": utc_now(),
            "direction": "qwen3_32b_to_14b",
            "method": method_definition["method"],
            "dtype": "float32",
            "allow_tf32": False,
            "architecture": {
                "source_layers": 64,
                "target_layers": 40,
                "kv_heads": 8,
                "head_dim": 128,
                "selected_source_layers": 20,
            },
            "weight_count": 40 * 2 * 20480 * 1024,
            "bias_count": 40 * 2 * 1024,
            "method_definition": method_definition,
            "mapper_fingerprint": mapper_fingerprint,
            "selection": {
                "selected_source_layers_rank_order": rows20,
                "k20_snapshot": artifact(staging / "inputs" / "selected_layers_k20.json", relative_to=staging),
                "k4_snapshot": artifact(staging / "inputs" / "selected_layers_k4.json", relative_to=staging),
                "embedding_validated": True,
            },
            "alpha": artifact(staging / "inputs" / "alpha_manifest.json", relative_to=staging),
            "split": artifact(staging / "inputs" / "split_manifest.json", relative_to=staging),
            "upstream": {
                "k20_manifest": artifact(k20_root / "manifest.json"),
                "k4_manifest": artifact(k4_root / "manifest.json"),
            },
            "layers": layer_entries,
            "payload_validation": {
                "status": "passed",
                "layers": 40,
                "tensors": 160,
                "entries_digest": canonical_sha256(payload_validation_entries),
            },
        }
        atomic_write_json(staging / "manifest.json", manifest)
        atomic_write_json(
            staging / "manifest.sha256.json",
            {"path": "manifest.json", "sha256": sha256_file(staging / "manifest.json")},
        )
        validate_mapper(staging, deep_hash=True)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return validate_mapper(destination, deep_hash=True)


def validate_mapper(root: Path, *, deep_hash: bool) -> dict[str, Any]:
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    marker = read_json(root / "manifest.sha256.json")
    manifest_sha = sha256_file(root / "manifest.json")
    if marker != {"path": "manifest.json", "sha256": manifest_sha}:
        raise ValueError("published mapper manifest marker mismatch")
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "method": "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers",
        "dtype": "float32",
        "allow_tf32": False,
        "weight_count": 1677721600,
        "bias_count": 81920,
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise ValueError(f"published mapper {field} mismatch")
    definition = manifest.get("method_definition")
    if canonical_sha256(definition) != manifest.get("mapper_fingerprint"):
        raise ValueError("mapper fingerprint mismatch")
    alpha_path = root / manifest["alpha"]["path"]
    if artifact(alpha_path, relative_to=root) != manifest["alpha"]:
        raise ValueError("alpha identity mismatch")
    alpha = read_json(alpha_path)
    if alpha.get("locked_audit_used_for_alpha") is not False:
        raise ValueError("locked audit leaked into alpha")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ValueError("published mapper layer inventory mismatch")
    entries = []
    for target_layer, entry in enumerate(layers):
        if entry.get("target_layer") != target_layer:
            raise ValueError("published mapper layers unordered")
        payload = root / entry["path"]
        sidecar_path = root / entry["sidecar_path"]
        if sha256_file(sidecar_path) != entry["sidecar_sha256"]:
            raise ValueError(f"published sidecar hash mismatch at {target_layer}")
        sidecar = read_json(sidecar_path)
        if sidecar.get("sha256") != entry["sha256"]:
            raise ValueError(f"published payload hash disagreement at {target_layer}")
        header = safetensors_header(payload)
        shapes = {
            key: value for key, value in header.items() if key != "__metadata__"
        }
        for name, expected_shape in _expected_shapes(20).items():
            if shapes.get(name, {}).get("dtype") != "F32" or shapes[name].get("shape") != expected_shape:
                raise ValueError(f"published tensor mismatch at {target_layer} {name}")
        observed_hash = sha256_file(payload) if deep_hash else entry["sha256"]
        if observed_hash != entry["sha256"]:
            raise ValueError(f"published payload hash mismatch at {target_layer}")
        entries.append({"target_layer": target_layer, "sha256": observed_hash, "tensor_shapes": _expected_shapes(20)})
    if canonical_sha256(entries) != manifest["payload_validation"]["entries_digest"]:
        raise ValueError("published payload digest mismatch")
    return {
        "schema_version": 1,
        "status": "passed",
        "root": str(root),
        "manifest_sha256": manifest_sha,
        "mapper_fingerprint": manifest["mapper_fingerprint"],
        "payloads_deep_hashed": deep_hash,
        "layers": 40,
        "tensors": 160,
    }


def _aggregate_audit_summary(layers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"target_layers": 40}
    for component in ("key", "value"):
        for candidate in ("k4", "k20", "shrinkage"):
            values = [float(layer[component]["audit"][candidate]["uniform_channel_r2_fp64"]) for layer in layers]
            result[f"{component}_{candidate}_uniform_channel_r2_mean"] = sum(values) / len(values)
            result[f"{component}_{candidate}_uniform_channel_r2_min"] = min(values)
    result["key_shrinkage_minus_k20_pp"] = 100.0 * (
        result["key_shrinkage_uniform_channel_r2_mean"]
        - result["key_k20_uniform_channel_r2_mean"]
    )
    result["value_shrinkage_minus_k20_pp"] = 100.0 * (
        result["value_shrinkage_uniform_channel_r2_mean"]
        - result["value_k20_uniform_channel_r2_mean"]
    )
    return result


def prior_failed_attempt_evidence() -> list[dict[str, Any]]:
    """Bind the two allocator-probe failures that preceded the formal run."""
    result = []
    for version in (1, 2):
        root = ROOT / "runs" / f"activation_cv_v{version}"
        console = ROOT / "runs" / f"activation_cv_v{version}.console.log"
        status = read_json(root / "status.json")
        gate = read_json(root / "idle_gate.json")
        if status.get("status") != "failed" or status.get("error") != "Invalid device argument ":
            raise ValueError(f"unexpected prior-attempt status: {root}")
        if gate.get("status") != "passed" or gate.get("observed_consecutive_samples") != 12:
            raise ValueError(f"prior-attempt idle gate is incomplete: {root}")
        forbidden = [
            root / "tuning_stats",
            root / "alpha_manifest.json",
            root / "alpha_freeze_receipt.json",
            root / "locked_audit.json",
            root / "receipt.json",
        ]
        if any(path.exists() for path in forbidden):
            raise ValueError(f"prior attempt progressed beyond allocator reset: {root}")
        result.append(
            {
                "attempt": f"activation_cv_v{version}",
                "status": "failed_before_any_activation_matmul",
                "failure_call": (
                    "torch.cuda.reset_peak_memory_stats(torch.device('cuda:0'))"
                    if version == 1
                    else "torch.cuda.reset_peak_memory_stats(0)"
                ),
                "models_loaded": [],
                "tuning_statistics_created": False,
                "alpha_created": False,
                "locked_audit_evaluated": False,
                "evidence": {
                    "status": artifact(root / "status.json"),
                    "run_definition": artifact(root / "run_definition.json"),
                    "idle_gate": artifact(root / "idle_gate.json"),
                    "console": artifact(console),
                },
            }
        )
    return result


def run_activation_cv() -> dict[str, Any]:
    # Set visibility before importing torch.  Logical cuda:0 is physical GPU 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = load_config()
    run_root = Path(str(config["outputs"]["activation_cv_run"])).resolve()
    mapper_root = Path(str(config["outputs"]["mapper"])).resolve()
    if run_root.exists() or mapper_root.exists():
        raise FileExistsError("refusing to overwrite the formal run or mapper")
    run_root.mkdir(parents=True)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    definition = {
        "schema_version": 1,
        "status": "frozen",
        "created_utc": utc_now(),
        "experiment_id": config["experiment_id"],
        "command": command,
        "cwd": str(Path.cwd().resolve()),
        "cuda_visible_devices": "0",
        "code": code_inventory(),
        "config": artifact(CONFIG_PATH),
        "protocol": config["protocol"],
        "runtime": config["runtime"],
        "prior_failed_attempts": prior_failed_attempt_evidence(),
    }
    atomic_write_json(run_root / "run_definition.json", definition)
    atomic_write_json(
        run_root / "status.json",
        {"schema_version": 1, "status": "preflight", "updated_utc": utc_now()},
    )
    lock_handle = None
    started = time.perf_counter()
    try:
        # Deep verification is intentionally completed before any CUDA matmul.
        preflight = validate_inputs(deep_hash=True)
        atomic_write_json(run_root / "input_preflight.json", preflight)
        atomic_write_json(run_root / "split_manifest.json", preflight["split"])
        lock_handle = acquire_idle_gate(config, run_root)
        import numpy as np
        import torch

        if torch.cuda.device_count() != 1:
            raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
        device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        # This pinned NGC build requires a CUDA context before its allocator
        # statistics reset.  The one-element probe is after the idle gate and
        # before the reset; it is not an activation or model computation.
        context_probe = torch.empty(1, device=device)
        torch.cuda.synchronize(0)
        del context_probe
        torch.cuda.reset_peak_memory_stats(0)
        split = preflight["split"]
        tuning_ids = [int(value) for value in split["tuning_sequence_ids"]]
        audit_ids = [int(value) for value in split["locked_audit_sequence_ids"]]
        k20_root = Path(str(config["upstream"]["k20_mapper"]))
        k4_root = Path(str(config["upstream"]["k4_mapper"]))
        source_root = Path(str(config["upstream"]["heldout_source_activations"]))
        target_root = Path(str(config["upstream"]["heldout_target_activations"]))
        rows20 = _selection_rows(k20_root)
        alpha_layers = []
        # Locked rows remain memory-only until every component's final alpha
        # has been fit and the complete alpha manifest has been frozen.
        locked_stats: dict[tuple[int, str], dict[str, Any]] = {}
        driver_peak_mib = 0
        atomic_write_json(
            run_root / "status.json",
            {"schema_version": 1, "status": "activation_cv", "updated_utc": utc_now()},
        )
        for target_layer in range(40):
            tensors20 = _load_mapper_layer(k20_root, target_layer)
            tensors4 = _load_mapper_layer(k4_root, target_layer)
            alpha_row: dict[str, Any] = {"target_layer": target_layer}
            for component, prefix in (("key", "key"), ("value", "value")):
                cache_type = "k_content" if component == "key" else "v"
                stats = collect_component_stats(
                    source_root=source_root,
                    target_root=target_root,
                    component=cache_type,
                    target_layer=target_layer,
                    selected_layers=rows20[target_layer],
                    weight20=tensors20[f"{prefix}_weight"],
                    bias20=tensors20[f"{prefix}_bias"],
                    weight4=tensors4[f"{prefix}_weight"],
                    bias4=tensors4[f"{prefix}_bias"],
                    device=device,
                    chunk_sequences=int(config["runtime"]["chunk_sequences"]),
                )
                # Only tuning sufficient statistics are persisted.  The locked
                # rows are not exposed before alpha is frozen.
                stats_path = run_root / "tuning_stats" / f"layer_{target_layer:03d}_{component}.npz"
                _save_tuning_stats(stats_path, stats, tuning_ids)
                final_alpha, cv = cross_validate_component(stats, split)
                alpha_row[component] = {
                    "alpha": final_alpha.tolist(),
                    "cv": cv,
                    "tuning_stats": artifact(stats_path, relative_to=run_root),
                }
                locked_index = np.asarray(audit_ids, dtype=np.int64)
                locked_stats[(target_layer, component)] = {
                    key: np.asarray(value)[locked_index].copy()
                    for key, value in stats.items()
                }
                del stats
                state = _driver_state()
                driver_peak_mib = max(
                    driver_peak_mib,
                    max(row["memory_used_mib"] for row in state["gpus"] if row["index"] == 0),
                )
            alpha_layers.append(alpha_row)
            del tensors20, tensors4
            torch.cuda.empty_cache()
            atomic_write_json(
                run_root / "status.json",
                {
                    "schema_version": 1,
                    "status": "activation_cv",
                    "completed_target_layers": target_layer + 1,
                    "updated_utc": utc_now(),
                },
            )
        alpha_manifest = _alpha_records(alpha_layers, split, preflight)
        atomic_write_json(run_root / "alpha_manifest.json", alpha_manifest)
        alpha_sha = sha256_file(run_root / "alpha_manifest.json")
        # This marker proves every alpha was frozen before the audit artifact.
        atomic_write_json(
            run_root / "alpha_freeze_receipt.json",
            {
                "schema_version": 1,
                "status": "frozen",
                "completed_utc": utc_now(),
                "alpha_manifest_sha256": alpha_sha,
                "locked_audit_used_for_alpha": False,
            },
        )
        # No locked-audit metric is computed until the complete 640-alpha
        # artifact above has been atomically frozen and content-hashed.
        audit_layers = []
        for target_layer in range(40):
            audit_row: dict[str, Any] = {"target_layer": target_layer}
            for component in ("key", "value"):
                stats = locked_stats[(target_layer, component)]
                # The memory-only arrays contain ten rows, hence local ids are
                # 0..9.  Original heldout sequence identities remain recorded
                # separately in the audit manifest.
                local_ids = list(range(10))
                final_alpha = np.asarray(
                    alpha_manifest["layers"][target_layer][component]["alpha"],
                    dtype=np.float64,
                )
                audit_row[component] = {
                    "audit": {
                        "k4": metrics_for_alpha(stats, local_ids, np.zeros(8), per_channel=True),
                        "k20": metrics_for_alpha(stats, local_ids, np.ones(8), per_channel=True),
                        "shrinkage": metrics_for_alpha(stats, local_ids, final_alpha, per_channel=True),
                    }
                }
            audit_layers.append(audit_row)
        del locked_stats
        audit_result = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": utc_now(),
            "scope": "ten-sequence locked heldout activation reconstruction audit; not downstream accuracy",
            "alpha_manifest_sha256": alpha_sha,
            "locked_audit_sequence_ids": audit_ids,
            "locked_audit_evaluations": 1,
            "locked_audit_used_for_alpha": False,
            "layers": audit_layers,
            "summary": _aggregate_audit_summary(audit_layers),
        }
        atomic_write_json(run_root / "locked_audit.json", audit_result)
        mapper_validation = publish_mapper(
            config=config, alpha_manifest=alpha_manifest, run_root=run_root
        )
        environment = {
            "python": sys.version,
            "executable": sys.executable,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "visible_device_count": torch.cuda.device_count(),
            "logical_cuda0_name": torch.cuda.get_device_name(0),
            "logical_cuda0_uuid": torch.cuda.get_device_properties(0).uuid,
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        }
        gpu_usage = {
            "physical_gpu_indices": [0],
            "models_loaded": [],
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "max_driver_memory_used_mib_observed": driver_peak_mib,
        }
        files = []
        for path in sorted(run_root.rglob("*")):
            if path.is_file() and path.name not in {"receipt.json", "receipt.sha256.json", "status.json"}:
                files.append(artifact(path, relative_to=run_root))
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "command": command,
            "code_digest": definition["code"]["digest"],
            "input_preflight_sha256": sha256_file(run_root / "input_preflight.json"),
            "idle_gate_sha256": sha256_file(run_root / "idle_gate.json"),
            "alpha_manifest_sha256": alpha_sha,
            "locked_audit_sha256": sha256_file(run_root / "locked_audit.json"),
            "mapper": mapper_validation,
            "environment": environment,
            "gpu_usage": gpu_usage,
            "run_files": files,
            "run_files_digest": canonical_sha256(files),
            "locked_audit_used_for_alpha": False,
            "prior_failed_attempts": definition["prior_failed_attempts"],
        }
        atomic_write_json(run_root / "receipt.json", receipt)
        atomic_write_json(
            run_root / "receipt.sha256.json",
            {"path": "receipt.json", "sha256": sha256_file(run_root / "receipt.json")},
        )
        atomic_write_json(
            run_root / "status.json",
            {
                "schema_version": 1,
                "status": "complete",
                "completed_utc": utc_now(),
                "receipt_sha256": sha256_file(run_root / "receipt.json"),
            },
        )
        return receipt
    except BaseException as exc:
        atomic_write_json(
            run_root / "status.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failed_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if lock_handle is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def validate_formal_outputs() -> dict[str, Any]:
    config = load_config()
    run_root = Path(str(config["outputs"]["activation_cv_run"])).resolve()
    mapper_root = Path(str(config["outputs"]["mapper"])).resolve()
    status = read_json(run_root / "status.json")
    if status.get("status") != "complete":
        raise ValueError("activation-CV run is not complete")
    receipt_path = run_root / "receipt.json"
    receipt = read_json(receipt_path)
    marker = read_json(run_root / "receipt.sha256.json")
    if marker != {"path": "receipt.json", "sha256": sha256_file(receipt_path)}:
        raise ValueError("receipt marker mismatch")
    if receipt.get("locked_audit_used_for_alpha") is not False:
        raise ValueError("receipt indicates audit leakage")
    alpha = read_json(run_root / "alpha_manifest.json")
    audit = read_json(run_root / "locked_audit.json")
    if sha256_file(run_root / "alpha_manifest.json") != audit.get("alpha_manifest_sha256"):
        raise ValueError("audit is not bound to frozen alpha")
    if audit.get("locked_audit_evaluations") != 1 or audit.get("locked_audit_used_for_alpha") is not False:
        raise ValueError("locked audit protocol changed")
    if len(alpha.get("layers", [])) != 40 or len(audit.get("layers", [])) != 40:
        raise ValueError("alpha/audit layer coverage changed")
    summary = _aggregate_audit_summary(audit["layers"])
    if canonical_bytes(summary) != canonical_bytes(audit.get("summary")):
        raise ValueError("audit summary does not derive from layer rows")
    mapper = validate_mapper(mapper_root, deep_hash=True)
    files = receipt.get("run_files")
    if not isinstance(files, list) or canonical_sha256(files) != receipt.get("run_files_digest"):
        raise ValueError("receipt run inventory digest mismatch")
    for entry in files:
        path = (run_root / entry["path"]).resolve()
        path.relative_to(run_root)
        if artifact(path, relative_to=run_root) != entry:
            raise ValueError(f"run artifact changed: {entry['path']}")
    return {
        "schema_version": 1,
        "status": "passed",
        "completed_utc": utc_now(),
        "run": str(run_root),
        "receipt_sha256": sha256_file(receipt_path),
        "mapper": mapper,
        "locked_audit_summary": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--deep-hash", action="store_true")
    subparsers.add_parser("run")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--no-deep-hash", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        result = publish_preflight(deep_hash=bool(args.deep_hash))
    elif args.command == "run":
        result = run_activation_cv()
    elif args.command == "validate":
        if args.no_deep_hash:
            config = load_config()
            result = validate_mapper(Path(config["outputs"]["mapper"]), deep_hash=False)
        else:
            result = validate_formal_outputs()
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
