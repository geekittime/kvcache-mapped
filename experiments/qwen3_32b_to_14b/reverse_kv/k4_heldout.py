"""Independent held-out activation audit required before the k=4 CoQA curve."""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Mapping
from pathlib import Path
import time
from typing import Any

import torch

from .k4_protocol import (
    K4_BIAS_COUNT,
    K4_MAPPER_ROOT,
    K4_SOURCE_FEATURES,
    K4_TARGET_FEATURES,
    K4_TOP_K,
    K4_WEIGHT_COUNT,
)
from .k4_runtime import K4ReverseMapperBundle
from .protocol import (
    DIRECTION,
    EXPERIMENT_ROOT,
    SOURCE,
    TARGET,
    atomic_write_json,
    canonical_sha256,
    read_json,
    sha256_file,
    validate_capture_pair,
)
from .provenance import load_verified_train_inventory

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPOSITORY_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.fit_mapper import MMapLayerChunkFactory  # noqa: E402
from kvcache_transfer.metrics import R2Accumulator  # noqa: E402


K4_HELDOUT = (
    EXPERIMENT_ROOT
    / "artifacts/validation/qwen3_32b_to_14b/k4_retuned/heldout_activation_reconstruction.json"
)


def _activation(root: Path, cache_type: str, layer: int) -> Path:
    return root / cache_type / f"layer_{layer:03d}.bf16.bin"


def _training_binding() -> dict[str, Any]:
    manifest_path = K4_MAPPER_ROOT / "manifest.json"
    checksum_path = K4_MAPPER_ROOT / "manifest.sha256.json"
    manifest = read_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    if read_json(checksum_path) != {
        "path": str(manifest_path),
        "sha256": manifest_sha,
    }:
        raise ValueError("k4 mapper manifest checksum changed during held-out preflight")
    if (
        manifest.get("status") != "complete"
        or manifest.get("direction") != DIRECTION
        or manifest.get("curve_variant") != "figure4_large_to_small_retuned_k4"
        or manifest.get("weight_count") != K4_WEIGHT_COUNT
        or manifest.get("bias_count") != K4_BIAS_COUNT
    ):
        raise ValueError("k4 held-out preflight received the wrong mapper family")
    inventory, source_identity, target_identity = load_verified_train_inventory(
        K4_MAPPER_ROOT, manifest
    )
    return {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha,
        "fit_fingerprint": manifest.get("fit_fingerprint"),
        "curve_variant": "figure4_large_to_small_retuned_k4",
        "selected_source_layers": K4_TOP_K,
        "source_features": K4_SOURCE_FEATURES,
        "target_features": K4_TARGET_FEATURES,
        "selection_snapshot_sha256": sha256_file(
            K4_MAPPER_ROOT / "inputs/selected_layers.json"
        ),
        "activation_inventory_sha256": manifest["activation_inventory"]["sha256"],
        "activation_inventory_digest": inventory["inventory_digest"],
        "training_payload_checksums_verified": True,
        "source_capture_model_identity": source_identity,
        "target_capture_model_identity": target_identity,
        "train_token_sha256": inventory["shared_provenance"]["token_sha256"],
    }


def k4_heldout_preflight(*, verify_payloads: bool = True) -> dict[str, Any]:
    if verify_payloads is not True:
        raise ValueError("formal k4 held-out audit requires full payload SHA256 verification")
    source, target = validate_capture_pair(split="heldout", verify_payloads=True)
    train_source, _ = validate_capture_pair(split="train", verify_payloads=False)
    binding = _training_binding()
    train_sha = train_source["token_sha256"]
    if binding["train_token_sha256"] != train_sha:
        raise ValueError("k4 mapper is not bound to the frozen train tokens")
    if train_sha == source["token_sha256"]:
        raise ValueError("k4 held-out activations reuse training tokens")
    return {
        "schema_version": 1,
        "status": "passed",
        "direction": DIRECTION,
        "curve_variant": "figure4_large_to_small_retuned_k4",
        "protocol": {
            "scope": "local_validation_not_disclosed_by_paper",
            "purpose": "mandatory_pre_CoQA_k4_mapper_acceptance_gate",
            "sequences": 50,
            "sequence_length": 1024,
            "stride": 4,
            "observations": 12_800,
            "selected_source_layers": K4_TOP_K,
            "source_features": K4_SOURCE_FEATURES,
            "target_features": K4_TARGET_FEATURES,
            "mapper_compute_dtype": "float32",
            "analysis_dtype": "float64",
            "quality_threshold": None,
            "acceptance_rule": "complete_finite_all_40_layers_K_and_V_with_exact_bindings",
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
        "train_token_sha256": train_sha,
        "train_heldout_tokens_differ": True,
        "payload_checksums_verified": True,
        "mapper_root": str(K4_MAPPER_ROOT),
        "mapper_training_binding": binding,
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
    selected = tuple(int(value) for value in mapper.source_layers)
    if len(selected) != K4_TOP_K:
        raise ValueError("held-out k4 mapper selected-layer count changed")
    source_paths = [
        _activation(SOURCE.heldout_root, cache_type, layer) for layer in selected
    ]
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
            raise ValueError("k4 held-out activation shape changed")
        if (
            chunks.source_features != K4_SOURCE_FEATURES
            or chunks.target_features != K4_TARGET_FEATURES
        ):
            raise ValueError("k4 held-out design matrix is not 4096 -> 1024")
        with torch.inference_mode():
            for x, y in chunks():
                prediction = torch.addmm(
                    bias,
                    x.to(device=compute_device, dtype=torch.float32),
                    weight,
                )
                metrics.update(y, prediction)
    result = metrics.finalize()
    values = {
        "uniform_channel_r2_fp64": result.uniform_average,
        "global_r2_fp64": result.global_r2,
        "flattened_global_r2_fp64": result.flattened_global_r2,
        "mse_fp64": result.mse,
        "global_sse_fp64": result.global_sse,
        "global_sst_fp64": result.global_sst,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise FloatingPointError("k4 held-out reconstruction produced NaN/Inf")
    return {
        **values,
        # Persist the FP64 sufficient statistics needed to derive every
        # reported score during resume/final-completion validation.  A JSON
        # checksum alone must not make a coordinated metric rewrite valid.
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
        raise ValueError(f"k4 held-out {field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"k4 held-out {field} is non-finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"k4 held-out {field} is negative")
    return result


def _expected_r2(sse: float, sst: float) -> float:
    # Match R2Accumulator's force-finite constant-channel policy.
    if sst == 0.0:
        return 1.0 if sse == 0.0 else 0.0
    return 1.0 - sse / sst


def validate_k4_heldout_artifact(path: str | Path = K4_HELDOUT) -> dict[str, Any]:
    destination = Path(path).resolve()
    if destination != K4_HELDOUT.resolve():
        raise ValueError("k4 held-out artifact path changed")
    marker_path = destination.with_suffix(".sha256.json")
    marker = read_json(marker_path)
    if marker != {"path": str(destination), "sha256": sha256_file(destination)}:
        raise ValueError("k4 held-out SHA256 marker changed")
    value = read_json(destination)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "direction": DIRECTION,
        "curve_variant": "figure4_large_to_small_retuned_k4",
        "metric_scope": "held-out activation reconstruction; not a benchmark score",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"k4 held-out {field} changed")
    # Re-run the full payload-hashing preflight.  This proves current train
    # and held-out capture identities, token separation, and the mapper's
    # activation-inventory/model bindings instead of trusting copied fields.
    expected_preflight = k4_heldout_preflight(verify_payloads=True)
    preflight = value.get("preflight")
    if preflight != expected_preflight:
        raise ValueError("k4 held-out preflight differs from current frozen inputs")
    binding = expected_preflight["mapper_training_binding"]
    mapper = value.get("mapper")
    if not isinstance(mapper, Mapping):
        raise ValueError("k4 held-out lacks mapper binding")
    bundle = K4ReverseMapperBundle.open(K4_MAPPER_ROOT)
    manifest_path = K4_MAPPER_ROOT / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    expected_mapper = {
        "manifest": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "fit_fingerprint": bundle.fit_fingerprint,
    }
    if mapper != expected_mapper or (
        binding.get("manifest_sha256") != manifest_sha
        or bundle.manifest_sha256 != manifest_sha
        or mapper.get("fit_fingerprint") != binding.get("fit_fingerprint")
        or bundle.architecture.source_layers != 64
        or bundle.architecture.target_layers != 40
        or bundle.architecture.source_features != K4_SOURCE_FEATURES
        or bundle.architecture.target_features != K4_TARGET_FEATURES
        or bundle.architecture.selected_source_layers != K4_TOP_K
    ):
        raise ValueError("k4 held-out mapper identity drifted")
    layers = value.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ValueError("k4 held-out must contain exactly 40 layers")
    manifest = read_json(manifest_path)
    selected_rows = manifest.get("selection", {}).get(
        "selected_source_layers_rank_order"
    )
    if not isinstance(selected_rows, list) or len(selected_rows) != 40:
        raise ValueError("k4 mapper selection rows are malformed")
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
    for expected_layer, row in enumerate(layers):
        if not isinstance(row, Mapping) or set(row) != {
            "target_layer",
            "selected_source_layers_rank_order",
            "key",
            "value",
        } or row.get("target_layer") != expected_layer:
            raise ValueError("k4 held-out target layers are not ordered/complete")
        if row.get("selected_source_layers_rank_order") != selected_rows[expected_layer]:
            raise ValueError(f"k4 held-out layer {expected_layer} selection changed")
        for cache_type in ("key", "value"):
            metrics = row.get(cache_type)
            if not isinstance(metrics, Mapping) or set(metrics) != metric_fields:
                raise ValueError(f"k4 held-out layer {expected_layer} lacks {cache_type}")
            if metrics.get("observations") != 12_800 or metrics.get("channels") != 1_024:
                raise ValueError("k4 held-out metric dimensions changed")
            uniform = _finite_number(
                metrics.get("uniform_channel_r2_fp64"), "uniform R2"
            )
            global_r2 = _finite_number(metrics.get("global_r2_fp64"), "global R2")
            flattened = _finite_number(
                metrics.get("flattened_global_r2_fp64"), "flattened R2"
            )
            mse = _finite_number(metrics.get("mse_fp64"), "MSE", nonnegative=True)
            sse = _finite_number(
                metrics.get("global_sse_fp64"), "global SSE", nonnegative=True
            )
            sst = _finite_number(
                metrics.get("global_sst_fp64"), "global SST", nonnegative=True
            )
            channel_sse_raw = metrics.get("per_channel_sse_fp64")
            channel_sst_raw = metrics.get("per_channel_sst_fp64")
            if (
                not isinstance(channel_sse_raw, list)
                or not isinstance(channel_sst_raw, list)
                or len(channel_sse_raw) != 1_024
                or len(channel_sst_raw) != 1_024
            ):
                raise ValueError("k4 held-out per-channel statistics changed")
            channel_sse = [
                _finite_number(item, "channel SSE", nonnegative=True)
                for item in channel_sse_raw
            ]
            channel_sst = [
                _finite_number(item, "channel SST", nonnegative=True)
                for item in channel_sst_raw
            ]
            if not math.isclose(
                math.fsum(channel_sse), sse, rel_tol=1e-12, abs_tol=1e-12
            ) or not math.isclose(
                math.fsum(channel_sst), sst, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("k4 held-out channel/global sums changed")
            expected_uniform = math.fsum(
                _expected_r2(error, variance)
                for error, variance in zip(channel_sse, channel_sst, strict=True)
            ) / 1_024
            if not math.isclose(
                uniform, expected_uniform, rel_tol=1e-12, abs_tol=1e-15
            ):
                raise ValueError("k4 held-out uniform R2 changed")
            flattened_sst = _finite_number(
                metrics.get("flattened_global_sst_fp64"),
                "flattened SST",
                nonnegative=True,
            )
            if not math.isclose(
                flattened,
                _expected_r2(sse, flattened_sst),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("k4 held-out flattened R2 changed")
            if not math.isclose(
                global_r2,
                _expected_r2(sse, sst),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("k4 held-out global R2 changed")
            if not math.isclose(
                mse,
                sse / (12_800 * 1_024),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError("k4 held-out MSE/SSE algebra changed")
            _finite_number(metrics.get("seconds"), "seconds", nonnegative=True)
    key = [float(row["key"]["uniform_channel_r2_fp64"]) for row in layers]
    val = [float(row["value"]["uniform_channel_r2_fp64"]) for row in layers]
    expected_summary = {
        "target_layers": 40,
        "key_uniform_channel_r2_fp64_mean_across_layers": sum(key) / 40,
        "value_uniform_channel_r2_fp64_mean_across_layers": sum(val) / 40,
        "key_uniform_channel_r2_fp64_min": min(key),
        "value_uniform_channel_r2_fp64_min": min(val),
    }
    if value.get("summary") != expected_summary:
        raise ValueError("k4 held-out summary differs from raw layer metrics")
    acceptance = value.get("acceptance")
    expected_acceptance = {
        "status": "passed",
        "rule": "complete_finite_all_40_layers_K_and_V_with_exact_bindings",
        "threshold_applied": False,
        "reason_no_numeric_threshold": "paper does not disclose a held-out activation threshold",
        "raw_layers_digest": canonical_sha256(layers),
    }
    if acceptance != expected_acceptance:
        raise ValueError("k4 held-out acceptance receipt changed")
    _finite_number(value.get("elapsed_seconds"), "elapsed seconds", nonnegative=True)
    completed = value.get("completed_utc")
    if not isinstance(completed, str):
        raise ValueError("k4 held-out completion timestamp is missing")
    try:
        timestamp = dt.datetime.fromisoformat(completed)
    except ValueError as exc:
        raise ValueError("k4 held-out completion timestamp is malformed") from exc
    if timestamp.tzinfo is None:
        raise ValueError("k4 held-out completion timestamp is not timezone-aware")
    if value.get("compute_device") != "cuda:0":
        raise ValueError("k4 held-out compute device changed")
    if value.get("analysis_device") != "cpu":
        raise ValueError("k4 held-out analysis device changed")
    if value.get("chunk_samples") != 1_024:
        raise ValueError("k4 held-out chunk size changed")
    return value


def evaluate_k4_heldout(
    *,
    compute_device: str = "cuda:0",
    analysis_device: str = "cpu",
    chunk_samples: int = 1024,
) -> dict[str, Any]:
    if K4_HELDOUT.is_file():
        return validate_k4_heldout_artifact()
    if chunk_samples <= 0:
        raise ValueError("k4 held-out chunk size must be positive")
    preflight = k4_heldout_preflight(verify_payloads=True)
    bundle = K4ReverseMapperBundle.open(K4_MAPPER_ROOT)
    compute = torch.device(compute_device)
    analysis = torch.device(analysis_device)
    layers = []
    started = time.perf_counter()
    for target_layer in range(40):
        mapper = bundle.load_layer(target_layer, device="cpu")
        layers.append(
            {
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
        )
        del mapper
        if compute.type == "cuda":
            torch.cuda.empty_cache()
    key = [float(row["key"]["uniform_channel_r2_fp64"]) for row in layers]
    value = [float(row["value"]["uniform_channel_r2_fp64"]) for row in layers]
    result = {
        "schema_version": 1,
        "status": "complete",
        "direction": DIRECTION,
        "curve_variant": "figure4_large_to_small_retuned_k4",
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
            "key_uniform_channel_r2_fp64_mean_across_layers": sum(key) / 40,
            "value_uniform_channel_r2_fp64_mean_across_layers": sum(value) / 40,
            "key_uniform_channel_r2_fp64_min": min(key),
            "value_uniform_channel_r2_fp64_min": min(value),
        },
        "acceptance": {
            "status": "passed",
            "rule": "complete_finite_all_40_layers_K_and_V_with_exact_bindings",
            "threshold_applied": False,
            "reason_no_numeric_threshold": "paper does not disclose a held-out activation threshold",
            "raw_layers_digest": canonical_sha256(layers),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    K4_HELDOUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(K4_HELDOUT, result)
    atomic_write_json(
        K4_HELDOUT.with_suffix(".sha256.json"),
        {"path": str(K4_HELDOUT), "sha256": sha256_file(K4_HELDOUT)},
    )
    return validate_k4_heldout_artifact()


__all__ = [
    "K4_HELDOUT",
    "evaluate_k4_heldout",
    "k4_heldout_preflight",
    "validate_k4_heldout_artifact",
]
