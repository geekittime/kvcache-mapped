"""Held-out activation reconstruction for the reverse top-20 mapper."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from pathlib import Path
import time
from typing import Any

import torch

from .protocol import (
    DIRECTION,
    EXPERIMENT_ROOT,
    SOURCE,
    TARGET,
    TOP_K,
    atomic_write_json,
    canonical_sha256,
    read_json,
    sha256_file,
    validate_capture_pair,
    validate_config,
)
from .provenance import load_verified_train_inventory
from .runtime import ReverseMapperBundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPOSITORY_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.fit_mapper import MMapLayerChunkFactory  # noqa: E402
from kvcache_transfer.metrics import R2Accumulator  # noqa: E402


_ACCEPTANCE_RULE = "complete_finite_all_40_layers_K_and_V_with_exact_bindings"
_NO_THRESHOLD_REASON = "paper does not disclose a held-out activation threshold"


def _activation(root: Path, cache_type: str, layer: int) -> Path:
    return root / cache_type / f"layer_{layer:03d}.bf16.bin"


def _mapper_training_binding(mapper_root: Path) -> dict[str, Any]:
    manifest_path = mapper_root / "manifest.json"
    checksum_path = mapper_root / "manifest.sha256.json"
    manifest = read_json(manifest_path)
    checksum = read_json(checksum_path)
    manifest_sha = sha256_file(manifest_path)
    if checksum.get("sha256") != manifest_sha:
        raise ValueError("mapper manifest SHA256 mismatch during held-out preflight")
    if manifest.get("status") != "complete" or manifest.get("direction") != DIRECTION:
        raise ValueError("held-out validation requires the complete reverse mapper")
    inventory, source_identity, target_identity = load_verified_train_inventory(
        mapper_root, manifest
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "fit_fingerprint": manifest.get("fit_fingerprint"),
        "activation_inventory_sha256": manifest["activation_inventory"]["sha256"],
        "activation_inventory_digest": inventory["inventory_digest"],
        "training_payload_checksums_verified": True,
        "source_capture_model_identity": source_identity,
        "target_capture_model_identity": target_identity,
        "train_token_sha256": inventory["shared_provenance"]["token_sha256"],
    }


def heldout_preflight(*, verify_payloads: bool = True) -> dict[str, Any]:
    if verify_payloads is not True:
        raise ValueError("formal held-out validation requires full payload SHA256 verification")
    configuration = validate_config()
    source, target = validate_capture_pair(
        split="heldout", verify_payloads=verify_payloads
    )
    train_source, _ = validate_capture_pair(split="train", verify_payloads=False)
    mapper_root = configuration["outputs"]["mapper"]
    mapper_binding = _mapper_training_binding(mapper_root)
    train_token_sha = train_source["token_sha256"]
    if mapper_binding["train_token_sha256"] != train_token_sha:
        raise ValueError("mapper inventory does not match frozen train activations")
    if train_token_sha == source["token_sha256"]:
        raise ValueError("held-out activations reuse mapper training tokens")
    return {
        "schema_version": 1,
        "status": "passed",
        "direction": DIRECTION,
        "protocol": {
            "sequences": 50,
            "sequence_length": 1024,
            "stride": 4,
            "observations": 12_800,
            "selected_source_layers": TOP_K,
            "mapper_compute_dtype": "float32",
            "analysis_dtype": "float64",
        },
        "source": {
            "model_name": SOURCE.model_name,
            "capture_native_role": SOURCE.capture_native_role,
            "manifest_sha256": source["manifest_sha256"],
            "token_sha256": source["token_sha256"],
        },
        "target": {
            "model_name": TARGET.model_name,
            "capture_native_role": TARGET.capture_native_role,
            "manifest_sha256": target["manifest_sha256"],
            "token_sha256": target["token_sha256"],
        },
        "train_token_sha256": train_token_sha,
        "train_heldout_tokens_differ": train_token_sha != source["token_sha256"],
        "payload_checksums_verified": verify_payloads,
        "mapper_root": str(mapper_root),
        "mapper_training_binding": mapper_binding,
    }


def _one_component(
    *,
    mapper: Any,
    target_layer: int,
    cache_type: str,
    chunk_samples: int,
    compute_device: torch.device,
    analysis_device: torch.device,
) -> dict[str, Any]:
    selected = mapper.source_layers
    source_paths = [_activation(SOURCE.heldout_root, cache_type, layer) for layer in selected]
    target_path = _activation(TARGET.heldout_root, cache_type, target_layer)
    weight = mapper.key_weight if cache_type == "k_content" else mapper.value_weight
    bias = mapper.key_bias if cache_type == "k_content" else mapper.value_bias
    weight = weight.to(device=compute_device, dtype=torch.float32)
    bias = bias.to(device=compute_device, dtype=torch.float32)
    metrics = R2Accumulator(analysis_device=analysis_device)
    started = time.perf_counter()
    with MMapLayerChunkFactory(
        source_paths,
        target_path,
        chunk_samples=chunk_samples,
        verify_checksum=False,
    ) as chunks:
        if chunks.activation_shape != (12_800, 8, 128):
            raise ValueError(f"held-out shape mismatch: {chunks.activation_shape}")
        if chunks.source_features != 20_480 or chunks.target_features != 1_024:
            raise ValueError("held-out design matrix dimensions are not 20480 -> 1024")
        with torch.inference_mode():
            for x, y in chunks():
                prediction = torch.addmm(
                    bias,
                    x.to(device=compute_device, dtype=torch.float32),
                    weight,
                )
                metrics.update(y, prediction)
    result = metrics.finalize()
    return {
        "uniform_channel_r2_fp64": result.uniform_average,
        "global_r2_fp64": result.global_r2,
        "flattened_global_r2_fp64": result.flattened_global_r2,
        "mse_fp64": result.mse,
        "global_sse_fp64": result.global_sse,
        "global_sst_fp64": result.global_sst,
        # Persist the sufficient FP64 statistics needed to derive every
        # reported reconstruction score without rerunning the GPU matmul.
        "per_channel_sse_fp64": [
            float(value) for value in result.sse_per_channel.cpu().tolist()
        ],
        "per_channel_sst_fp64": [
            float(value) for value in result.sst_per_channel.cpu().tolist()
        ],
        "flattened_global_sst_fp64": result.flattened_global_sst,
        "observations": result.n_samples,
        "channels": result.n_channels,
        "seconds": time.perf_counter() - started,
    }


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"held-out {field} is not a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"held-out {field} is non-finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"held-out {field} is negative")
    return result


def _expected_global_r2(sse: float, sst: float) -> float:
    # Match R2Accumulator's force-finite constant-channel policy exactly.
    if sst == 0.0:
        return 1.0 if sse == 0.0 else 0.0
    return 1.0 - sse / sst


def validate_heldout_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """Validate a committed k=20 held-out result from current raw identities.

    The checksum marker is only an accidental-corruption check.  This routine
    additionally reconstructs the frozen preflight, reopens the current mapper
    bundle, checks every target-layer selection, verifies all stored metric
    algebra, and derives the summary/acceptance receipt from the 40 raw rows.
    Consequently, rewriting both JSON and its checksum cannot bypass the
    scientific validation.
    """

    configuration = validate_config()
    expected_path = Path(configuration["outputs"]["heldout_validation"]).resolve()
    destination = Path(path).resolve() if path is not None else expected_path
    if destination != expected_path:
        raise ValueError("held-out artifact path differs from the frozen output")
    try:
        destination.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("held-out output escaped reverse experiment") from exc

    marker_path = destination.with_suffix(".sha256.json")
    marker = read_json(marker_path)
    expected_marker = {"path": str(destination), "sha256": sha256_file(destination)}
    if marker != expected_marker:
        raise ValueError("held-out SHA256 marker changed")
    value = read_json(destination)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "direction": DIRECTION,
        "metric_scope": "held-out activation reconstruction; not a benchmark score",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"held-out {field} changed")

    # Reconstruct capture, token-separation, payload-checksum, and mapper-fit
    # identities rather than trusting the copy embedded in the result.
    expected_preflight = heldout_preflight(verify_payloads=True)
    if value.get("preflight") != expected_preflight:
        raise ValueError("held-out preflight differs from current frozen inputs")

    mapper_root = Path(configuration["outputs"]["mapper"]).resolve()
    bundle = ReverseMapperBundle.open(mapper_root)
    expected_mapper = {
        "manifest": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "fit_fingerprint": bundle.fit_fingerprint,
    }
    if value.get("mapper") != expected_mapper:
        raise ValueError("held-out mapper identity differs from the current bundle")
    if (
        bundle.architecture.source_layers != 64
        or bundle.architecture.target_layers != 40
        or bundle.architecture.source_features != 20_480
        or bundle.architecture.target_features != 1_024
        or bundle.architecture.selected_source_layers != TOP_K
    ):
        raise ValueError("held-out mapper architecture changed")
    binding = expected_preflight["mapper_training_binding"]
    if (
        binding.get("manifest_sha256") != bundle.manifest_sha256
        or binding.get("fit_fingerprint") != bundle.fit_fingerprint
    ):
        raise ValueError("held-out preflight and mapper bundle disagree")

    manifest = read_json(bundle.manifest_path)
    selected_rows = manifest.get("selection", {}).get(
        "selected_source_layers_rank_order"
    )
    if not isinstance(selected_rows, list) or len(selected_rows) != 40:
        raise ValueError("held-out mapper selection rows are malformed")
    layers = value.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ValueError("held-out result must contain exactly 40 target layers")
    metric_fields = {
        "uniform_channel_r2_fp64",
        "global_r2_fp64",
        "flattened_global_r2_fp64",
        "mse_fp64",
        "global_sse_fp64",
        "global_sst_fp64",
        "per_channel_sse_fp64",
        "per_channel_sst_fp64",
        "flattened_global_sst_fp64",
        "observations",
        "channels",
        "seconds",
    }
    for target_layer, row in enumerate(layers):
        if not isinstance(row, Mapping) or set(row) != {
            "target_layer",
            "selected_source_layers_rank_order",
            "key",
            "value",
        }:
            raise ValueError(f"held-out target layer {target_layer} row is malformed")
        if row.get("target_layer") != target_layer:
            raise ValueError("held-out target layers are not ordered and complete")
        if row.get("selected_source_layers_rank_order") != selected_rows[target_layer]:
            raise ValueError(f"held-out layer {target_layer} selection changed")
        for component in ("key", "value"):
            metrics = row.get(component)
            if not isinstance(metrics, Mapping) or set(metrics) != metric_fields:
                raise ValueError(
                    f"held-out layer {target_layer} {component} metrics are malformed"
                )
            if metrics.get("observations") != 12_800 or metrics.get("channels") != 1_024:
                raise ValueError("held-out metric dimensions changed")
            uniform = _finite_number(
                metrics.get("uniform_channel_r2_fp64"),
                f"layer {target_layer} {component} uniform R2",
            )
            flattened = _finite_number(
                metrics.get("flattened_global_r2_fp64"),
                f"layer {target_layer} {component} flattened R2",
            )
            sse = _finite_number(
                metrics.get("global_sse_fp64"),
                f"layer {target_layer} {component} SSE",
                nonnegative=True,
            )
            sst = _finite_number(
                metrics.get("global_sst_fp64"),
                f"layer {target_layer} {component} SST",
                nonnegative=True,
            )
            mse = _finite_number(
                metrics.get("mse_fp64"),
                f"layer {target_layer} {component} MSE",
                nonnegative=True,
            )
            global_r2 = _finite_number(
                metrics.get("global_r2_fp64"),
                f"layer {target_layer} {component} global R2",
            )
            channel_sse_raw = metrics.get("per_channel_sse_fp64")
            channel_sst_raw = metrics.get("per_channel_sst_fp64")
            if (
                not isinstance(channel_sse_raw, list)
                or not isinstance(channel_sst_raw, list)
                or len(channel_sse_raw) != 1_024
                or len(channel_sst_raw) != 1_024
            ):
                raise ValueError(
                    f"held-out layer {target_layer} {component} per-channel statistics changed"
                )
            channel_sse = [
                _finite_number(
                    item,
                    f"layer {target_layer} {component} channel SSE",
                    nonnegative=True,
                )
                for item in channel_sse_raw
            ]
            channel_sst = [
                _finite_number(
                    item,
                    f"layer {target_layer} {component} channel SST",
                    nonnegative=True,
                )
                for item in channel_sst_raw
            ]
            if not math.isclose(
                math.fsum(channel_sse), sse, rel_tol=1e-12, abs_tol=1e-12
            ) or not math.isclose(
                math.fsum(channel_sst), sst, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(
                    f"held-out layer {target_layer} {component} channel/global sums changed"
                )
            expected_uniform = math.fsum(
                _expected_global_r2(channel_error, channel_variance)
                for channel_error, channel_variance in zip(
                    channel_sse, channel_sst, strict=True
                )
            ) / 1_024
            if not math.isclose(
                uniform, expected_uniform, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError(
                    f"held-out layer {target_layer} {component} uniform R2 changed"
                )
            flattened_sst = _finite_number(
                metrics.get("flattened_global_sst_fp64"),
                f"layer {target_layer} {component} flattened SST",
                nonnegative=True,
            )
            if not math.isclose(
                flattened,
                _expected_global_r2(sse, flattened_sst),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"held-out layer {target_layer} {component} flattened R2 changed"
                )
            expected_mse = sse / (12_800 * 1_024)
            if not math.isclose(mse, expected_mse, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(
                    f"held-out layer {target_layer} {component} MSE/SSE algebra changed"
                )
            if not math.isclose(
                global_r2,
                _expected_global_r2(sse, sst),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"held-out layer {target_layer} {component} R2/SSE/SST algebra changed"
                )
            _finite_number(
                metrics.get("seconds"),
                f"layer {target_layer} {component} seconds",
                nonnegative=True,
            )

    key_values = [float(row["key"]["uniform_channel_r2_fp64"]) for row in layers]
    value_values = [float(row["value"]["uniform_channel_r2_fp64"]) for row in layers]
    expected_summary = {
        "target_layers": 40,
        "key_uniform_channel_r2_fp64_mean_across_layers": sum(key_values) / 40,
        "value_uniform_channel_r2_fp64_mean_across_layers": sum(value_values) / 40,
        "key_uniform_channel_r2_fp64_min": min(key_values),
        "value_uniform_channel_r2_fp64_min": min(value_values),
    }
    if value.get("summary") != expected_summary:
        raise ValueError("held-out summary differs from the 40 layer metrics")
    expected_acceptance = {
        "status": "passed",
        "rule": _ACCEPTANCE_RULE,
        "threshold_applied": False,
        "reason_no_numeric_threshold": _NO_THRESHOLD_REASON,
        "raw_layers_digest": canonical_sha256(layers),
    }
    if value.get("acceptance") != expected_acceptance:
        raise ValueError("held-out acceptance differs from the raw layer metrics")
    _finite_number(value.get("elapsed_seconds"), "elapsed seconds", nonnegative=True)
    completed = value.get("completed_utc")
    if not isinstance(completed, str):
        raise ValueError("held-out completion timestamp is missing")
    try:
        timestamp = dt.datetime.fromisoformat(completed)
    except ValueError as exc:
        raise ValueError("held-out completion timestamp is malformed") from exc
    if timestamp.tzinfo is None:
        raise ValueError("held-out completion timestamp is not timezone-aware")
    if not isinstance(value.get("compute_device"), str) or not value["compute_device"]:
        raise ValueError("held-out compute device evidence is missing")
    if not isinstance(value.get("analysis_device"), str) or not value["analysis_device"]:
        raise ValueError("held-out analysis device evidence is missing")
    if isinstance(value.get("chunk_samples"), bool) or not isinstance(
        value.get("chunk_samples"), int
    ) or value["chunk_samples"] <= 0:
        raise ValueError("held-out chunk size is invalid")
    return value


def evaluate_heldout(
    *,
    compute_device: str = "cuda:0",
    analysis_device: str = "cpu",
    chunk_samples: int = 1024,
    output: Path | None = None,
) -> dict[str, Any]:
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    preflight = heldout_preflight(verify_payloads=True)
    configuration = validate_config()
    mapper_root = configuration["outputs"]["mapper"]
    bundle = ReverseMapperBundle.open(mapper_root)
    compute = torch.device(compute_device)
    analysis = torch.device(analysis_device)
    layers = []
    started = time.perf_counter()
    for target_layer in range(40):
        mapper = bundle.load_layer(target_layer, device="cpu")
        layer = {
            "target_layer": target_layer,
            "selected_source_layers_rank_order": list(mapper.source_layers),
            "key": _one_component(
                mapper=mapper,
                target_layer=target_layer,
                cache_type="k_content",
                chunk_samples=chunk_samples,
                compute_device=compute,
                analysis_device=analysis,
            ),
            "value": _one_component(
                mapper=mapper,
                target_layer=target_layer,
                cache_type="v",
                chunk_samples=chunk_samples,
                compute_device=compute,
                analysis_device=analysis,
            ),
        }
        layers.append(layer)
        del mapper
        if compute.type == "cuda":
            torch.cuda.empty_cache()
    key_values = [float(layer["key"]["uniform_channel_r2_fp64"]) for layer in layers]
    value_values = [float(layer["value"]["uniform_channel_r2_fp64"]) for layer in layers]
    result = {
        "schema_version": 1,
        "status": "complete",
        "direction": DIRECTION,
        "metric_scope": "held-out activation reconstruction; not a benchmark score",
        "preflight": preflight,
        "mapper": {
            "manifest": str(bundle.manifest_path),
            "manifest_sha256": bundle.manifest_sha256,
            "fit_fingerprint": bundle.fit_fingerprint,
        },
        "compute_device": str(compute),
        "analysis_device": str(analysis),
        "chunk_samples": chunk_samples,
        "layers": layers,
        "summary": {
            "target_layers": 40,
            "key_uniform_channel_r2_fp64_mean_across_layers": sum(key_values) / 40,
            "value_uniform_channel_r2_fp64_mean_across_layers": sum(value_values) / 40,
            "key_uniform_channel_r2_fp64_min": min(key_values),
            "value_uniform_channel_r2_fp64_min": min(value_values),
        },
        "acceptance": {
            "status": "passed",
            "rule": _ACCEPTANCE_RULE,
            "threshold_applied": False,
            "reason_no_numeric_threshold": _NO_THRESHOLD_REASON,
            "raw_layers_digest": canonical_sha256(layers),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    destination = (output or configuration["outputs"]["heldout_validation"]).resolve()
    try:
        destination.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("held-out output escaped reverse experiment") from exc
    atomic_write_json(destination, result)
    atomic_write_json(
        destination.with_suffix(".sha256.json"),
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    return validate_heldout_artifact(destination)


__all__ = ["evaluate_heldout", "heldout_preflight", "validate_heldout_artifact"]
