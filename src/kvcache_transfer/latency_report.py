"""Fail-closed reporting for the production Qwen3 latency experiment.

This module is intentionally independent from the benchmark implementation.  It
uses only the Python standard library, never imports torch, and treats the
benchmark run directories as untrusted input.  A report is produced only after
the endpoint gate and the ten-length paper-main run have been checked through
their SHA sidecars, manifests, request fingerprints, checkpoints, and redundant
JSON/CSV representations.

The local measurements are from RTX 4090 GPUs connected without NVLink/P2P.
The paper references are H100 measurements and are included as context, not as
directly comparable absolute latency baselines.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import statistics
import tempfile
from typing import Any, Mapping, Sequence

from .io import sha256_file


REPORT_SCHEMA_VERSION = 1
INPUT_SCHEMA_VERSION = 3
CUDA_PEAK_SCHEMA_VERSION = 2

PAPER_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
ENDPOINT_LENGTHS = (64, 32768)
SOURCE_DEVICE_IDS = (4, 5)
TARGET_DEVICE_IDS = (0, 1, 2, 3)
SELECTED_DEVICE_IDS = (*SOURCE_DEVICE_IDS, *TARGET_DEVICE_IDS)
CUDA_VISIBILITY = "4,5,6,7,0,1"
ENDPOINT_MINIMUM_HEADROOM_BYTES = 1 << 30
MAPPER_PARAMETER_BYTES = 4_295_491_584
MAPPER_MANIFEST_SHA256 = "785e2d57c8953a0c630f3044c4c06381f81a88799ca651960586f584f569ec08"
MAPPER_FIT_FINGERPRINT = "860adcba99ae9c61bdf39c084145bc8d65cedb6ebac0454d07ff939b7b85f0f9"
LATENCY_CODE_FILES_DIGEST = "00b2a0c71664e062a777ff5b4f1299e744e63229a86f2115d570b92fdfa181f3"
BENCHMARK_LATENCY_SHA256 = "85b31d2e0eb430107c364df3c5c03327e0b7597cd0b5788eac175de0fe047b75"
LATENCY_LIBRARY_SHA256 = "1566894a35cdf52d947d78e9499efcd13d4d0fd5ea530b5c5e73af858b57e582"

CUDA_PEAK_MEASUREMENT_SCOPE = (
    "entire_length_cell_from_source_cache_build_through_final_allocator_cleanup"
)
STAGED_CUDA_PEAK_MEASUREMENT_SCOPE = (
    "entire_length_cell_from_preexisting_gpu_source_cache_through_final_allocator_cleanup"
)
SOURCE_CACHE_STAGING_POLICY = "untimed_cpu_staging_then_gpu_restage_v1"
SOURCE_CACHE_BF16_BYTES_PER_TOKEN = 40 * 2 * 8 * 128 * 2
CUDA_PEAK_ALLOCATOR_SCOPE = "current_process_pytorch_cuda_allocator"
CUDA_DRIVER_FREE_SCOPE = "whole_device_snapshots_before_and_after_not_historical_peak"
CUDA_PEAK_DISABLED_REASON = "preserve_caller_owned_outer_cuda_peak_counters"
MULTI_GPU_TIMING_BACKEND = "host_perf_counter_with_multi_cuda_synchronize"
PRELOAD_GPU_IDLE_GATE_POLICY = {
    "schema_version": 1,
    "scope": "post_identity_hash_pre_cuda_model_load",
    "cuda_visible_devices_environment": CUDA_VISIBILITY,
    "selected_logical_device_ids": list(SELECTED_DEVICE_IDS),
    "maximum_memory_used_mib": 64,
    "required_consecutive_sandwich_samples": 12,
    "sandwich_interval_seconds": 5.0,
    "timeout_seconds": 86400.0,
    "memory_query": "index,uuid,memory.used",
    "compute_apps_query": "gpu_uuid,pid,process_name,used_memory",
    "selected_compute_apps_required_empty": True,
    "streak_resets_on_any_failed_observation": True,
    "nvidia_smi_error_fails_closed": True,
}

# Only the audited original placement and the capacity-driven all-eight-card
# fallback are valid for this reproduction.  Dynamic extraction must not turn
# into permission for arbitrary smaller layouts that change both memory risk
# and interconnect timing.
ALLOWED_DEVICE_LAYOUTS = {
    "legacy_4target_2source": {
        "source_devices": (4, 5),
        "target_devices": (0, 1, 2, 3),
        "cuda_visibility": "4,5,6,7,0,1",
        "selected_physical_devices": (0, 1, 4, 5, 6, 7),
        "source_layer_counts": None,
        "target_layer_counts": None,
    },
    "balanced_5target_3source_v3": {
        "source_devices": (5, 6, 7),
        "target_devices": (0, 1, 2, 3, 4),
        "cuda_visibility": "3,4,5,6,7,0,1,2",
        "selected_physical_devices": (0, 1, 2, 3, 4, 5, 6, 7),
        "source_layer_counts": None,
        "target_layer_counts": None,
    },
    "balanced_5target_3source_low_first_v4": {
        "source_devices": (5, 6, 7),
        "target_devices": (0, 1, 2, 3, 4),
        "cuda_visibility": "3,4,5,6,7,0,1,2",
        "selected_physical_devices": (0, 1, 2, 3, 4, 5, 6, 7),
        "source_layer_counts": (14, 13, 13),
        "target_layer_counts": (8, 15, 14, 14, 13),
    },
    "balanced_5target_3source_embedding_relief_v5": {
        "source_devices": (5, 6, 7),
        "target_devices": (0, 1, 2, 3, 4),
        "cuda_visibility": "3,4,5,6,7,0,1,2",
        "selected_physical_devices": (0, 1, 2, 3, 4, 5, 6, 7),
        "source_layer_counts": (14, 13, 13),
        "target_layer_counts": (4, 15, 15, 15, 15),
    },
    "balanced_5target_3source_single_first_v6": {
        "source_devices": (5, 6, 7),
        "target_devices": (0, 1, 2, 3, 4),
        "cuda_visibility": "3,4,5,6,7,0,1,2",
        "selected_physical_devices": (0, 1, 2, 3, 4, 5, 6, 7),
        "source_layer_counts": (14, 13, 13),
        "target_layer_counts": (1, 16, 16, 16, 15),
    },
    "balanced_5target_3source_equal_v7": {
        "source_devices": (5, 6, 7),
        "target_devices": (0, 1, 2, 3, 4),
        "cuda_visibility": "3,4,5,6,7,0,1,2",
        "selected_physical_devices": (0, 1, 2, 3, 4, 5, 6, 7),
        "source_layer_counts": (14, 13, 13),
        "target_layer_counts": (13, 13, 13, 13, 12),
    },
    "balanced_5target_3source_inference_low_first_v8": {
        "source_devices": (5, 6, 7),
        "target_devices": (0, 1, 2, 3, 4),
        "cuda_visibility": "3,4,5,6,7,0,1,2",
        "selected_physical_devices": (0, 1, 2, 3, 4, 5, 6, 7),
        "source_layer_counts": (14, 13, 13),
        "target_layer_counts": (6, 15, 15, 15, 13),
    },
    "staged_6target_2cache_v9": {
        "source_devices": (6, 7),
        "target_devices": (0, 1, 2, 3, 4, 5),
        "cuda_visibility": "2,3,4,5,6,7,0,1",
        "selected_physical_devices": (0, 1, 2, 3, 4, 5, 6, 7),
        "source_layer_counts": (20, 20),
        "target_layer_counts": (10, 11, 11, 11, 11, 10),
        "source_prefill_devices": (5, 6, 7),
        "source_prefill_layer_counts": (14, 13, 13),
        "staged_source_lifecycle": True,
    },
}

PAPER_MAIN_PROTOCOL = {
    "name": "paper_main_only",
    "paper_main_metrics": ["target_reprefill", "mapper_handoff_end_to_end"],
    "supplemental_diagnostics_enabled": False,
    "component_profile_in_cell": False,
    "boundary_decode_in_cell": False,
    "protocol_complete_in_cell": False,
    "paper_headline_speedup_uses_supplemental_metrics": False,
}
NESTED_PEAK_CONTRACT = {
    "outer_cell_peak_measurement_active": True,
    "internal_metric_peak_collection_enabled": False,
    "internal_peak_counter_resets_disabled": True,
}
TIMED_FAST_PATH_CONTRACT = {
    "enabled": True,
    "strict_preflight_required_and_passed": True,
    "strict_validation_argument": False,
    "diagnostic_rope_and_finite_reductions_in_timed_region": False,
    "shape_dtype_architecture_and_output_validation_retained": True,
}
SUPPLEMENTAL_FIELDS = (
    "mapper_synchronized_component_profile",
    "mapper_kernel_only_profile",
    "boundary_one_token_decode",
    "protocol_complete",
    "median_speedup_target_reprefill_over_protocol_complete",
)

# The paper's displayed table has only these three reference rows.  At length
# 64 it displays ``4x`` even though the ratio of the displayed 61.7/14.0 ms
# values is about 4.41.  Both facts are retained rather than silently choosing
# one interpretation.
PAPER_REFERENCE = {
    64: {
        "mapper_handoff_ms": 14.0,
        "target_reprefill_ms": 61.7,
        "reported_speedup_x": 4.0,
    },
    8192: {
        "mapper_handoff_ms": 67.8,
        "target_reprefill_ms": 1154.8,
        "reported_speedup_x": 17.0,
    },
    32768: {
        "mapper_handoff_ms": 277.6,
        "target_reprefill_ms": 6975.3,
        "reported_speedup_x": 25.1,
    },
}

INPUT_CSV_COLUMNS = (
    "prefix_length",
    "metric",
    "count",
    "mean_ms",
    "median_ms",
    "p10_ms",
    "p50_ms",
    "p90_ms",
    "std_population_ms",
    "min_ms",
    "max_ms",
    "timing_backends",
    "source_prefill_included",
)

OUTPUT_CSV_COLUMNS = (
    "prefix_length",
    "target_reprefill_mean_ms",
    "target_reprefill_median_ms",
    "target_reprefill_p10_ms",
    "target_reprefill_p90_ms",
    "mapper_handoff_mean_ms",
    "mapper_handoff_median_ms",
    "mapper_handoff_p10_ms",
    "mapper_handoff_p90_ms",
    "median_speedup_target_over_mapper_x",
    "minimum_peak_reserved_headroom_bytes",
    "minimum_peak_reserved_headroom_gib",
    "paper_mapper_handoff_ms",
    "paper_target_reprefill_ms",
    "paper_reported_speedup_x",
    "paper_ratio_from_displayed_ms_x",
    "absolute_ms_directly_comparable_to_paper",
)


class LatencyReportError(ValueError):
    """Raised when input artifacts cannot support the formal latency report."""


@dataclass(frozen=True)
class DeviceLayout:
    """Logical/physical CUDA placement bound by a production run definition."""

    source_devices: tuple[int, ...]
    target_devices: tuple[int, ...]
    selected_devices: tuple[int, ...]
    cuda_visibility: str
    visible_physical_devices: tuple[int, ...]
    profile: str
    source_layer_counts: tuple[int, ...] | None
    target_layer_counts: tuple[int, ...] | None
    source_prefill_devices: tuple[int, ...] = ()
    source_prefill_layer_counts: tuple[int, ...] | None = None
    staged_source_lifecycle: bool = False

    @property
    def selected_physical_devices(self) -> tuple[int, ...]:
        return tuple(self.visible_physical_devices[index] for index in self.selected_devices)

    @property
    def preload_policy(self) -> dict[str, Any]:
        return {
            **PRELOAD_GPU_IDLE_GATE_POLICY,
            "cuda_visible_devices_environment": self.cuda_visibility,
            "selected_logical_device_ids": list(self.selected_devices),
        }

    @property
    def requires_exact_layer_evidence(self) -> bool:
        return (
            self.source_layer_counts is not None
            and self.target_layer_counts is not None
        )


@dataclass(frozen=True)
class ValidatedRun:
    root: Path
    definition: Mapping[str, Any]
    configuration: Mapping[str, Any]
    result: Mapping[str, Any]
    manifest: Mapping[str, Any]
    hardware: Mapping[str, Any]
    attention: Mapping[str, Any]
    mapper_setup: Mapping[str, Any]
    metrics: Mapping[str, Any]
    checkpoints: tuple[Mapping[str, Any], ...]
    checkpoint_sha256: Mapping[int, str]
    minimum_headroom_by_length: Mapping[int, int]
    layout: DeviceLayout


def canonical_digest(value: Any) -> str:
    """Return the benchmark's canonical finite-JSON SHA256 digest."""

    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LatencyReportError(f"value is not canonical finite JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _reject_constant(value: str) -> None:
    raise LatencyReportError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LatencyReportError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise LatencyReportError(f"required JSON artifact is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                parse_constant=_reject_constant,
                object_pairs_hook=_reject_duplicate_pairs,
            )
    except LatencyReportError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LatencyReportError(f"cannot read strict JSON from {path}: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LatencyReportError(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise LatencyReportError(f"{label} must be a JSON array")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise LatencyReportError(f"{label} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise LatencyReportError(f"{label} must be >= {minimum}, got {value}")
    return value


def _number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LatencyReportError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise LatencyReportError(f"{label} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise LatencyReportError(f"{label} must be >= {minimum}, got {result}")
    return result


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LatencyReportError(f"{label} must be a lowercase SHA256 digest")
    return value


def _same_number(actual: Any, expected: float, label: str) -> None:
    value = _number(actual, label)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise LatencyReportError(f"{label} is inconsistent ({value!r} != {expected!r})")


def _device_tuple(value: Any, label: str) -> tuple[int, ...]:
    raw = _list(value, label)
    devices = tuple(
        _integer(item, f"{label}[{index}]", minimum=0)
        for index, item in enumerate(raw)
    )
    if not devices:
        raise LatencyReportError(f"{label} must not be empty")
    if len(set(devices)) != len(devices):
        raise LatencyReportError(f"{label} contains duplicate CUDA indices")
    return devices


def _optional_layer_count_tuple(
    value: Any,
    label: str,
    *,
    device_count: int,
    total_layers: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    raw = _list(value, label)
    counts = tuple(
        _integer(item, f"{label}[{index}]", minimum=1)
        for index, item in enumerate(raw)
    )
    if len(counts) != device_count:
        raise LatencyReportError(
            f"{label} must contain exactly one positive count per CUDA device"
        )
    if sum(counts) != total_layers:
        raise LatencyReportError(
            f"{label} must sum to {total_layers}, got {sum(counts)}"
        )
    return counts


def _device_layout_from_definition(
    definition: Mapping[str, Any], *, label: str
) -> DeviceLayout:
    """Derive the placement contract from fingerprint-bound run fields.

    The report intentionally does not choose a topology.  It accepts either
    the original 4-target/2-source placement or a fresh capacity fallback only
    when the CLI, topology, preload evidence, config, hardware, checkpoints,
    endpoint receipt, and formal run all bind the same non-overlapping layout.
    """

    cli = _mapping(definition.get("cli"), f"{label}.cli")
    topology = _mapping(definition.get("topology"), f"{label}.topology")
    source = _device_tuple(cli.get("source_devices"), f"{label}.cli.source_devices")
    target = _device_tuple(cli.get("target_devices"), f"{label}.cli.target_devices")
    if set(source).intersection(target):
        raise LatencyReportError(f"{label} source/target CUDA devices overlap")
    selected = (*source, *target)
    if topology.get("selected_devices") != list(selected):
        raise LatencyReportError(
            f"{label}.topology.selected_devices must equal source+target order"
        )
    visibility = topology.get("cuda_visible_devices_environment")
    if not isinstance(visibility, str) or not visibility:
        raise LatencyReportError(f"{label}.topology lacks CUDA visibility")
    raw_tokens = visibility.split(",")
    if any(not token or token.strip() != token for token in raw_tokens):
        raise LatencyReportError(f"{label}.topology CUDA visibility is not canonical")
    try:
        physical = tuple(int(token) for token in raw_tokens)
    except ValueError as exc:
        raise LatencyReportError(
            f"{label}.topology CUDA visibility must use physical integer indices"
        ) from exc
    if any(device < 0 for device in physical) or len(set(physical)) != len(physical):
        raise LatencyReportError(
            f"{label}.topology CUDA visibility has negative/duplicate devices"
        )
    if any(device >= len(physical) for device in selected):
        raise LatencyReportError(
            f"{label} selects a logical CUDA device outside CUDA_VISIBLE_DEVICES"
        )
    selected_physical = tuple(physical[index] for index in selected)
    source_counts = _optional_layer_count_tuple(
        cli.get("source_layer_counts"),
        f"{label}.cli.source_layer_counts",
        device_count=len(source),
        total_layers=40,
    )
    target_counts = _optional_layer_count_tuple(
        cli.get("target_layer_counts"),
        f"{label}.cli.target_layer_counts",
        device_count=len(target),
        total_layers=64,
    )
    if (source_counts is None) != (target_counts is None):
        raise LatencyReportError(
            f"{label} must declare source and target layer counts together"
        )
    staged = cli.get("stage_source_caches_on_cpu_before_cells") is True
    if staged:
        source_prefill = _device_tuple(
            cli.get("source_prefill_devices"),
            f"{label}.cli.source_prefill_devices",
        )
        source_prefill_counts = _optional_layer_count_tuple(
            cli.get("source_prefill_layer_counts"),
            f"{label}.cli.source_prefill_layer_counts",
            device_count=len(source_prefill),
            total_layers=40,
        )
        if source_prefill_counts is None:
            raise LatencyReportError(f"{label} staged source prefill needs exact counts")
    else:
        source_prefill = ()
        source_prefill_counts = None
    matching_profiles = [
        name
        for name, contract in ALLOWED_DEVICE_LAYOUTS.items()
        if source == contract["source_devices"]
        and target == contract["target_devices"]
        and visibility == contract["cuda_visibility"]
        and selected_physical == contract["selected_physical_devices"]
        and source_counts == contract["source_layer_counts"]
        and target_counts == contract["target_layer_counts"]
        and source_prefill == contract.get("source_prefill_devices", ())
        and source_prefill_counts == contract.get("source_prefill_layer_counts")
        and staged == contract.get("staged_source_lifecycle", False)
    ]
    if len(matching_profiles) != 1:
        raise LatencyReportError(
            f"{label} CUDA placement is not an audited latency layout"
        )
    return DeviceLayout(
        source_devices=source,
        target_devices=target,
        selected_devices=selected,
        cuda_visibility=visibility,
        visible_physical_devices=physical,
        profile=matching_profiles[0],
        source_layer_counts=source_counts,
        target_layer_counts=target_counts,
        source_prefill_devices=source_prefill,
        source_prefill_layer_counts=source_prefill_counts,
        staged_source_lifecycle=staged,
    )


def _legacy_even_contiguous_counts(total_layers: int, device_count: int) -> tuple[int, ...]:
    counts = [0] * device_count
    for layer in range(total_layers):
        counts[min(layer * device_count // total_layers, device_count - 1)] += 1
    return tuple(counts)


def _expected_staged_runtime_topology(layout: DeviceLayout) -> dict[str, Any]:
    if not layout.staged_source_lifecycle:
        raise AssertionError("staged topology requested for a legacy layout")
    physical = layout.visible_physical_devices
    return {
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "cuda_visible_devices_environment": layout.cuda_visibility,
        "logical_to_physical": {
            str(index): device for index, device in enumerate(physical)
        },
        "source_model_prefill": {
            "logical_devices": list(layout.source_prefill_devices),
            "physical_devices": [physical[index] for index in layout.source_prefill_devices],
            "layer_counts": list(layout.source_prefill_layer_counts or ()),
        },
        "source_cache_staging": {
            "logical_devices": list(layout.source_devices),
            "physical_devices": [physical[index] for index in layout.source_devices],
            "layer_counts": list(layout.source_layer_counts or ()),
        },
        "target_timed": {
            "logical_devices": list(layout.target_devices),
            "physical_devices": [physical[index] for index in layout.target_devices],
            "layer_counts": list(layout.target_layer_counts or ()),
        },
        "source_prefill_target_logical_overlap": [5],
        "source_prefill_target_physical_overlap": [7],
        "source_cache_target_logical_overlap": [],
        "source_cache_target_physical_overlap": [],
        "logical_5_unique_sequential_reuse_physical_device": 7,
    }


def _resolved_counts_for_role(layout: DeviceLayout, role: str) -> tuple[int, ...]:
    if role == "source":
        return layout.source_layer_counts or _legacy_even_contiguous_counts(
            40, len(layout.source_devices)
        )
    if role == "target":
        return layout.target_layer_counts or _legacy_even_contiguous_counts(
            64, len(layout.target_devices)
        )
    raise AssertionError(f"unknown model role: {role}")


def _expected_layer_devices(
    devices: Sequence[int], counts: Sequence[int]
) -> tuple[int, ...]:
    return tuple(
        device
        for device, count in zip(devices, counts, strict=True)
        for _ in range(count)
    )


def _validate_layer_sharding_contract(
    value: Any,
    *,
    layout: DeviceLayout,
    label: str,
) -> Mapping[str, Any]:
    contract = _mapping(value, label)
    if layout.staged_source_lifecycle:
        expected_roles = {
            "source",
            "target",
            "source_model_prefill",
            "source_cache_staging",
        }
        if set(contract) != expected_roles:
            raise LatencyReportError(
                f"{label} must contain all three staged topology roles"
            )
        if contract["source"] != contract["source_cache_staging"]:
            raise LatencyReportError(f"{label} source/cache-staging aliases differ")
        cache_view = DeviceLayout(
            source_devices=layout.source_devices,
            target_devices=layout.target_devices,
            selected_devices=layout.selected_devices,
            cuda_visibility=layout.cuda_visibility,
            visible_physical_devices=layout.visible_physical_devices,
            profile=layout.profile,
            source_layer_counts=layout.source_layer_counts,
            target_layer_counts=layout.target_layer_counts,
        )
        _validate_layer_sharding_contract(
            {"source": contract["source"], "target": contract["target"]},
            layout=cache_view,
            label=f"{label}.cache_and_target",
        )
        prefill_view = DeviceLayout(
            source_devices=layout.source_prefill_devices,
            target_devices=layout.target_devices,
            selected_devices=layout.selected_devices,
            cuda_visibility=layout.cuda_visibility,
            visible_physical_devices=layout.visible_physical_devices,
            profile=layout.profile,
            source_layer_counts=layout.source_prefill_layer_counts,
            target_layer_counts=layout.target_layer_counts,
        )
        _validate_layer_sharding_contract(
            {
                "source": contract["source_model_prefill"],
                "target": contract["target"],
            },
            layout=prefill_view,
            label=f"{label}.prefill_and_target",
        )
        return contract
    if set(contract) != {"source", "target"}:
        raise LatencyReportError(f"{label} must contain exactly source and target")
    for role, devices, total in (
        ("source", layout.source_devices, 40),
        ("target", layout.target_devices, 64),
    ):
        item = _mapping(contract.get(role), f"{label}.{role}")
        declared = (
            layout.source_layer_counts if role == "source" else layout.target_layer_counts
        )
        expected_counts = _resolved_counts_for_role(layout, role)
        expected = {
            "mode": "explicit_exact_counts" if declared is not None else "legacy_even_contiguous",
            "devices": list(devices),
            "declared_layer_counts": list(declared) if declared is not None else None,
            "resolved_layer_counts": list(expected_counts),
            "total_layers": total,
        }
        if canonical_digest(item) != canonical_digest(expected):
            raise LatencyReportError(f"{label}.{role} does not prove the audited layer split")
    return contract


def _validate_runtime_model_layer_sharding(
    value: Any,
    *,
    layout: DeviceLayout,
    label: str,
) -> Mapping[str, Any]:
    evidence = _mapping(value, label)
    if layout.staged_source_lifecycle:
        expected_roles = {
            "source",
            "source_model_prefill",
            "source_cache_staging",
            "target",
        }
        if set(evidence) != expected_roles:
            raise LatencyReportError(
                f"{label} must contain staged prefill/cache/target roles"
            )
        if evidence["source"] != evidence["source_model_prefill"]:
            raise LatencyReportError(f"{label} staged source aliases differ")
        source_item = dict(
            _mapping(evidence["source_model_prefill"], f"{label}.source_model_prefill")
        )
        if source_item.get("role") != "source_model_prefill":
            raise LatencyReportError(f"{label} prefill runtime role mismatch")
        source_item["role"] = "source"
        prefill_view = DeviceLayout(
            source_devices=layout.source_prefill_devices,
            target_devices=layout.target_devices,
            selected_devices=layout.selected_devices,
            cuda_visibility=layout.cuda_visibility,
            visible_physical_devices=layout.visible_physical_devices,
            profile=layout.profile,
            source_layer_counts=layout.source_prefill_layer_counts,
            target_layer_counts=layout.target_layer_counts,
        )
        _validate_runtime_model_layer_sharding(
            {"source": source_item, "target": evidence["target"]},
            layout=prefill_view,
            label=f"{label}.prefill_and_target",
        )
        staging = _mapping(
            evidence["source_cache_staging"], f"{label}.source_cache_staging"
        )
        counts = layout.source_layer_counts or ()
        layer_devices = list(_expected_layer_devices(layout.source_devices, counts))
        ranges: list[dict[str, int]] = []
        start = 0
        for device, count in zip(layout.source_devices, counts, strict=True):
            ranges.append(
                {
                    "device": device,
                    "start_layer": start,
                    "end_layer_exclusive": start + count,
                    "count": count,
                }
            )
            start += count
        expected_staging = {
            "status": "passed",
            "role": "source_cache_staging_without_source_model",
            "mode": "explicit_exact_counts",
            "devices": list(layout.source_devices),
            "declared_layer_counts": list(counts),
            "expected_layer_counts": list(counts),
            "actual_layer_counts": list(counts),
            "total_layers": 40,
            "layer_devices": layer_devices,
            "contiguous_ranges": ranges,
            "source_model_parameters_present": False,
            "validated_per_cell_from_actual_cache_tensors": True,
        }
        if staging != expected_staging:
            raise LatencyReportError(f"{label} cache-staging evidence mismatch")
        return evidence
    if set(evidence) != {"source", "target"}:
        raise LatencyReportError(f"{label} must contain exactly source and target")
    for role, devices, total in (
        ("source", layout.source_devices, 40),
        ("target", layout.target_devices, 64),
    ):
        item = _mapping(evidence.get(role), f"{label}.{role}")
        declared = (
            layout.source_layer_counts if role == "source" else layout.target_layer_counts
        )
        counts = _resolved_counts_for_role(layout, role)
        layer_devices = _expected_layer_devices(devices, counts)
        ranges: list[dict[str, int]] = []
        start = 0
        for device, count in zip(devices, counts, strict=True):
            ranges.append(
                {
                    "device": device,
                    "start_layer": start,
                    "end_layer_exclusive": start + count,
                    "count": count,
                }
            )
            start += count
        expected = {
            "status": "passed",
            "role": role,
            "mode": "explicit_exact_counts" if declared is not None else "legacy_even_contiguous",
            "devices": list(devices),
            "declared_layer_counts": list(declared) if declared is not None else None,
            "expected_layer_counts": list(counts),
            "actual_layer_counts": list(counts),
            "total_layers": total,
            "layer_devices": list(layer_devices),
            "contiguous_ranges": ranges,
            "actual_parameters_inspected_for_every_layer": True,
            "auxiliary_parameters_inspected_for_every_module": True,
        }
        allowed = {
            *expected,
            "parameter_tensors_inspected",
            "auxiliary_modules",
            "auxiliary_parameter_tensors_inspected",
        }
        if set(item) != allowed or any(
            canonical_digest(item.get(field)) != canonical_digest(wanted)
            for field, wanted in expected.items()
        ):
            raise LatencyReportError(
                f"{label}.{role} does not prove the actual audited layer split"
            )
        expected_parameter_tensors = total * 11
        if _integer(
            item.get("parameter_tensors_inspected"),
            f"{label}.{role}.parameter_tensors_inspected",
            minimum=1,
        ) != expected_parameter_tensors:
            raise LatencyReportError(
                f"{label}.{role} decoder parameter inspection count mismatch"
            )
        expected_aux_devices = {
            "model.embed_tokens": devices[0],
            "model.norm": devices[-1],
            "lm_head": devices[-1],
        }
        auxiliary = _mapping(
            item.get("auxiliary_modules"), f"{label}.{role}.auxiliary_modules"
        )
        if set(auxiliary) != set(expected_aux_devices):
            raise LatencyReportError(
                f"{label}.{role} auxiliary module coverage mismatch"
            )
        auxiliary_tensor_count = 0
        for module_name, expected_device in expected_aux_devices.items():
            module = _mapping(
                auxiliary[module_name],
                f"{label}.{role}.auxiliary_modules[{module_name!r}]",
            )
            if set(module) != {
                "expected_device",
                "actual_device",
                "parameter_tensors_inspected",
            }:
                raise LatencyReportError(
                    f"{label}.{role} auxiliary module fields mismatch"
                )
            declared_aux_device = _integer(
                module.get("expected_device"),
                f"{label}.{role}.auxiliary_modules[{module_name!r}].expected_device",
                minimum=0,
            )
            actual_aux_device = _integer(
                module.get("actual_device"),
                f"{label}.{role}.auxiliary_modules[{module_name!r}].actual_device",
                minimum=0,
            )
            if declared_aux_device != expected_device or actual_aux_device != expected_device:
                raise LatencyReportError(
                    f"{label}.{role} auxiliary module device mismatch"
                )
            auxiliary_tensor_count += _integer(
                module.get("parameter_tensors_inspected"),
                f"{label}.{role}.auxiliary_modules[{module_name!r}].parameter_tensors_inspected",
                minimum=1,
            )
        if _integer(
            item.get("auxiliary_parameter_tensors_inspected"),
            f"{label}.{role}.auxiliary_parameter_tensors_inspected",
            minimum=1,
        ) != auxiliary_tensor_count:
            raise LatencyReportError(
                f"{label}.{role} auxiliary parameter count is inconsistent"
            )
        if auxiliary_tensor_count != 3:
            raise LatencyReportError(
                f"{label}.{role} auxiliary parameter inspection count mismatch"
            )
    return evidence


def _mapper_target_layer_counts(
    setup: Mapping[str, Any], layout: DeviceLayout
) -> tuple[int, ...]:
    counts = {device: 0 for device in layout.target_devices}
    for raw in _list(setup.get("per_layer"), "mapper_setup.per_layer"):
        item = _mapping(raw, "mapper_setup.per_layer[]")
        device_text = item.get("device")
        if not isinstance(device_text, str) or not device_text.startswith("cuda:"):
            raise LatencyReportError("mapper_setup.per_layer has a malformed CUDA device")
        try:
            device = int(device_text.removeprefix("cuda:"))
        except ValueError as exc:
            raise LatencyReportError(
                "mapper_setup.per_layer has a malformed CUDA device"
            ) from exc
        if device not in counts:
            raise LatencyReportError("mapper_setup.per_layer has an unexpected CUDA device")
        counts[device] += 1
    return tuple(counts[device] for device in layout.target_devices)


def _validate_sha_sidecar(
    path: Path,
    *,
    schema_version: int = INPUT_SCHEMA_VERSION,
    sidecar_path: Path | None = None,
) -> Mapping[str, Any]:
    sidecar = sidecar_path or path.with_name(f"{path.name}.sha256.json")
    value = _mapping(_load_json(sidecar), str(sidecar))
    _integer(value.get("schema_version"), f"{sidecar}.schema_version")
    expected = {
        "schema_version": schema_version,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
    }
    if value != expected:
        raise LatencyReportError(f"artifact SHA256 sidecar mismatch: {path}")
    return value


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_latency_summary(
    value: Any,
    *,
    label: str,
    warmup: int,
    repeats: int,
) -> Mapping[str, Any]:
    summary = _mapping(value, label)
    samples_raw = _list(summary.get("samples"), f"{label}.samples")
    if len(samples_raw) != repeats:
        raise LatencyReportError(
            f"{label}.samples has {len(samples_raw)} values, expected {repeats}"
        )
    samples = [
        _number(sample, f"{label}.samples[{index}]", minimum=0.0)
        for index, sample in enumerate(samples_raw)
    ]
    if summary.get("unit") != "ms":
        raise LatencyReportError(f"{label}.unit must be 'ms'")
    if _integer(summary.get("count"), f"{label}.count", minimum=1) != repeats:
        raise LatencyReportError(f"{label}.count must be {repeats}")
    if _integer(summary.get("timed_repeats"), f"{label}.timed_repeats", minimum=1) != repeats:
        raise LatencyReportError(f"{label}.timed_repeats must be {repeats}")
    if _integer(summary.get("warmup"), f"{label}.warmup", minimum=0) != warmup:
        raise LatencyReportError(f"{label}.warmup must be {warmup}")
    if summary.get("synchronize_each_iteration") is not True:
        raise LatencyReportError(f"{label} did not synchronize every iteration")
    if summary.get("timing_backends") != [MULTI_GPU_TIMING_BACKEND]:
        raise LatencyReportError(
            f"{label}.timing_backends must prove synchronized multi-GPU host timing"
        )
    if summary.get("cuda_peak_memory_collected") is not False:
        raise LatencyReportError(f"{label} performed an internal CUDA peak collection")
    if summary.get("cuda_peak_memory_disabled_reason") != CUDA_PEAK_DISABLED_REASON:
        raise LatencyReportError(f"{label} lacks the nested peak-disable reason")
    if summary.get("inference_mode_enabled") is not True:
        raise LatencyReportError(f"{label} lacks inference-mode evidence")
    if summary.get("autograd_graph_disabled") is not True:
        raise LatencyReportError(f"{label} lacks disabled-autograd evidence")
    for forbidden in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        if forbidden in summary:
            raise LatencyReportError(f"{label} contains forbidden internal field {forbidden}")

    expected = {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p10": _percentile(samples, 0.10),
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "std_population": statistics.pstdev(samples),
        "min": min(samples),
        "max": max(samples),
    }
    for field, expected_value in expected.items():
        _same_number(summary.get(field), expected_value, f"{label}.{field}")
    return summary


def _validate_project_code(value: Any, *, label: str) -> Mapping[str, Any]:
    project = _mapping(value, label)
    files = _mapping(project.get("files"), f"{label}.files")
    if not files:
        raise LatencyReportError(f"{label}.files is empty")
    for name, identity_raw in files.items():
        if not isinstance(name, str) or not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise LatencyReportError(f"{label}.files contains an unsafe path: {name!r}")
        identity = _mapping(identity_raw, f"{label}.files[{name!r}]")
        _sha256(identity.get("sha256"), f"{label}.files[{name!r}].sha256")
        _integer(identity.get("nbytes"), f"{label}.files[{name!r}].nbytes", minimum=1)
    expected = canonical_digest(files)
    if project.get("files_digest") != expected:
        raise LatencyReportError(f"{label}.files_digest does not hash files")
    if expected != LATENCY_CODE_FILES_DIGEST:
        raise LatencyReportError(f"{label}.files_digest is not the frozen schema-3 code digest")
    frozen_files = {
        "scripts/benchmark_latency.py": BENCHMARK_LATENCY_SHA256,
        "src/kvcache_transfer/latency.py": LATENCY_LIBRARY_SHA256,
    }
    for name, digest in frozen_files.items():
        identity = _mapping(files.get(name), f"{label}.files[{name!r}]")
        if identity.get("sha256") != digest:
            raise LatencyReportError(f"{label}.files[{name!r}] differs from frozen code")
    return project


def _validate_model_identity(value: Any, *, label: str) -> Mapping[str, Any]:
    model = _mapping(value, label)
    files = _mapping(model.get("content_hashed_files"), f"{label}.content_hashed_files")
    if not {"config.json", "model.safetensors.index.json", "tokenizer.json"} <= set(files):
        raise LatencyReportError(f"{label} lacks required content-hashed identity files")
    for name, identity_raw in files.items():
        identity = _mapping(identity_raw, f"{label}.content_hashed_files[{name!r}]")
        _sha256(identity.get("sha256"), f"{label}.{name}.sha256")
        _integer(identity.get("nbytes"), f"{label}.{name}.nbytes", minimum=1)
    shards = _list(model.get("weight_shards"), f"{label}.weight_shards")
    if not shards:
        raise LatencyReportError(f"{label}.weight_shards is empty")
    total = 0
    names: set[str] = set()
    for index, raw in enumerate(shards):
        item = _mapping(raw, f"{label}.weight_shards[{index}]")
        name = item.get("path")
        if not isinstance(name, str) or not name or name in names:
            raise LatencyReportError(f"{label}.weight_shards has an invalid/duplicate path")
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise LatencyReportError(f"{label}.weight_shards path escapes the model root")
        names.add(name)
        total += _integer(item.get("nbytes"), f"{label}.weight_shards[{index}].nbytes", minimum=1)
        _sha256(item.get("sha256"), f"{label}.weight_shards[{index}].sha256")
    if _integer(model.get("weight_shard_count"), f"{label}.weight_shard_count", minimum=1) != len(shards):
        raise LatencyReportError(f"{label}.weight_shard_count is inconsistent")
    if _integer(model.get("weight_shard_total_nbytes"), f"{label}.weight_shard_total_nbytes", minimum=1) != total:
        raise LatencyReportError(f"{label}.weight_shard_total_nbytes is inconsistent")
    if model.get("weight_shards_digest") != canonical_digest(shards):
        raise LatencyReportError(f"{label}.weight_shards_digest is inconsistent")
    if model.get("weight_shard_contents_hashed") is not True:
        raise LatencyReportError(f"{label} does not claim content-hashed weight shards")
    bound_config = model.get("bound_config_fields")
    if bound_config is not None:
        bound = _mapping(bound_config, f"{label}.bound_config_fields")
        expected_fields = {
            "model_type",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_size",
        }
        if set(bound) != expected_fields:
            raise LatencyReportError(f"{label}.bound_config_fields coverage mismatch")
        if not isinstance(bound.get("model_type"), str) or not bound["model_type"]:
            raise LatencyReportError(f"{label}.bound_config_fields.model_type is invalid")
        for field in expected_fields - {"model_type"}:
            _integer(
                bound.get(field),
                f"{label}.bound_config_fields.{field}",
                minimum=1,
            )
    return model


def _validate_runtime_artifacts(value: Any, *, label: str) -> Mapping[str, Any]:
    runtime = _mapping(value, label)
    if set(runtime) != {"source_model", "target_model", "mapper"}:
        raise LatencyReportError(f"{label} must contain exactly source_model/target_model/mapper")
    _validate_model_identity(runtime["source_model"], label=f"{label}.source_model")
    _validate_model_identity(runtime["target_model"], label=f"{label}.target_model")
    mapper = _mapping(runtime["mapper"], f"{label}.mapper")
    if _sha256(mapper.get("manifest_sha256"), f"{label}.mapper.manifest_sha256") != MAPPER_MANIFEST_SHA256:
        raise LatencyReportError(f"{label} is not the audited Qwen3 mapper manifest")
    if _sha256(mapper.get("fit_fingerprint"), f"{label}.mapper.fit_fingerprint") != MAPPER_FIT_FINGERPRINT:
        raise LatencyReportError(f"{label} is not the audited Qwen3 mapper fit")
    if _integer(mapper.get("layer_count"), f"{label}.mapper.layer_count") != 64:
        raise LatencyReportError(f"{label}.mapper.layer_count must be 64")
    return runtime


def _validate_attention(
    value: Any, *, maximum_length: int, staged_source_lifecycle: bool = False
) -> Mapping[str, Any]:
    attention = _mapping(value, "attention_backend.json")
    if attention.get("schema_version") != INPUT_SCHEMA_VERSION or attention.get("status") != "passed":
        raise LatencyReportError("production attention backend assertion is not complete/passing")
    if attention.get("persisted_before_any_length") is not True or attention.get("covers_32768_before_execution") is not True:
        raise LatencyReportError("attention backend was not asserted before the 32K sweep")
    if staged_source_lifecycle and (
        attention.get("source_model_released_before_target_model_load") is not True
        or attention.get("source_attention_asserted_during_untimed_cache_preparation")
        is not True
    ):
        raise LatencyReportError("staged attention evidence lacks source-release binding")
    for role, layers in (("source", 40), ("target", 64)):
        item = _mapping(attention.get(role), f"attention_backend.{role}")
        expected_fields = {
            "status": "passed",
            "role": role,
            "expected": "flash_attention_2",
            "config_attn_implementation": "flash_attention_2",
            "dispatcher_verified_for_every_layer": True,
            "backend_callable_module": "transformers.integrations.flash_attention",
            "backend_callable_name": "flash_attention_forward",
            "flash_attention_2_available": True,
            "maximum_requested_length": maximum_length,
            "decoder_layer_count": layers,
        }
        for field, expected in expected_fields.items():
            if item.get(field) != expected:
                raise LatencyReportError(f"attention_backend.{role}.{field} mismatch")
        if _integer(item.get("max_position_embeddings"), f"attention_backend.{role}.max_position_embeddings", minimum=maximum_length) < maximum_length:
            raise LatencyReportError(f"attention_backend.{role} cannot cover {maximum_length}")
        classes = _list(item.get("attention_module_classes"), f"attention_backend.{role}.attention_module_classes")
        if not classes or not all(isinstance(name, str) and name for name in classes):
            raise LatencyReportError(f"attention_backend.{role} lacks module classes")
    return attention


def _validate_mapper_setup(
    value: Any, *, runtime: Mapping[str, Any], layout: DeviceLayout
) -> Mapping[str, Any]:
    setup = _mapping(value, "mapper_setup.json")
    expected = {
        "scope": "one_time_setup_excluded_from_per_request_latency",
        "resident_dtype": "float32",
        "execution": "resident_eager_fp32",
        "resident_parameter_bytes": MAPPER_PARAMETER_BYTES,
        "manifest_sha256": MAPPER_MANIFEST_SHA256,
        "fit_fingerprint": MAPPER_FIT_FINGERPRINT,
        "target_layer_count": 64,
    }
    for field, wanted in expected.items():
        if setup.get(field) != wanted:
            raise LatencyReportError(f"mapper_setup.{field} mismatch")
    mapper_identity = _mapping(runtime.get("mapper"), "runtime_artifacts.mapper")
    if setup.get("manifest_sha256") != mapper_identity.get("manifest_sha256") or setup.get("fit_fingerprint") != mapper_identity.get("fit_fingerprint"):
        raise LatencyReportError("mapper setup identity differs from the run definition")
    model_sharding = setup.get("model_layer_sharding")
    if layout.requires_exact_layer_evidence or model_sharding is not None:
        _validate_runtime_model_layer_sharding(
            model_sharding,
            layout=layout,
            label="mapper_setup.model_layer_sharding",
        )
    if layout.staged_source_lifecycle:
        if (
            setup.get("source_model_live_during_timed_cells") is not False
            or setup.get("source_cache_staging_policy")
            != SOURCE_CACHE_STAGING_POLICY
            or setup.get("runtime_topology")
            != _expected_staged_runtime_topology(layout)
        ):
            raise LatencyReportError("mapper setup lacks staged source-release contract")
    per_layer = _list(setup.get("per_layer"), "mapper_setup.per_layer")
    if len(per_layer) != 64:
        raise LatencyReportError("mapper_setup.per_layer must contain all 64 target layers")
    observed_devices: set[str] = set()
    for index, raw in enumerate(per_layer):
        item = _mapping(raw, f"mapper_setup.per_layer[{index}]")
        if _integer(
            item.get("target_layer"),
            f"mapper_setup.per_layer[{index}].target_layer",
            minimum=0,
        ) != index:
            raise LatencyReportError("mapper_setup.per_layer target layers are incomplete/reordered")
        if item.get("device") not in {
            f"cuda:{device}" for device in layout.target_devices
        }:
            raise LatencyReportError(f"mapper_setup.per_layer[{index}] has an unexpected device")
        observed_devices.add(str(item["device"]))
        _number(item.get("host_load_and_hash_ms"), f"mapper_setup.per_layer[{index}].host_load_and_hash_ms", minimum=0.0)
        _number(item.get("weight_host_to_device_ms"), f"mapper_setup.per_layer[{index}].weight_host_to_device_ms", minimum=0.0)
    if observed_devices != {f"cuda:{device}" for device in layout.target_devices}:
        raise LatencyReportError("mapper_setup.per_layer does not cover every target device")
    mapper_counts = _mapper_target_layer_counts(setup, layout)
    if layout.requires_exact_layer_evidence:
        expected_counts = _resolved_counts_for_role(layout, "target")
        if mapper_counts != expected_counts:
            raise LatencyReportError(
                "mapper_setup.per_layer actual target distribution differs from exact counts"
            )
        expected_devices = _expected_layer_devices(
            layout.target_devices, expected_counts
        )
        actual_devices = tuple(
            int(str(_mapping(raw, "mapper_setup.per_layer[]")["device"]).removeprefix("cuda:"))
            for raw in per_layer
        )
        if actual_devices != expected_devices:
            raise LatencyReportError(
                "mapper_setup.per_layer is not the exact contiguous target split"
            )
    return setup


def _validate_hardware(
    value: Any, *, layout: DeviceLayout
) -> tuple[Mapping[str, Any], dict[int, int]]:
    hardware = _mapping(value, "hardware.json")
    if hardware.get("cuda_available") is not True:
        raise LatencyReportError("formal production run did not record CUDA availability")
    if hardware.get("selected_devices") != list(layout.selected_devices):
        raise LatencyReportError("hardware selected-device order differs from the protocol")
    if hardware.get("cuda_visible_devices_environment") != layout.cuda_visibility:
        raise LatencyReportError("CUDA visibility differs from the frozen logical/physical mapping")
    devices = _list(hardware.get("devices"), "hardware.devices")
    if len(devices) != len(layout.selected_devices):
        raise LatencyReportError("hardware.devices does not cover every selected GPU")
    capacities: dict[int, int] = {}
    for raw in devices:
        item = _mapping(raw, "hardware.devices[]")
        index = _integer(item.get("index"), "hardware.devices[].index", minimum=0)
        if index in capacities or index not in layout.selected_devices:
            raise LatencyReportError("hardware.devices has duplicate/unexpected device indices")
        name = item.get("name")
        if not isinstance(name, str) or "RTX 4090" not in name:
            raise LatencyReportError("formal local hardware is not NVIDIA RTX 4090")
        capacities[index] = _integer(
            item.get("total_memory_bytes"),
            f"hardware.devices[{index}].total_memory_bytes",
            minimum=1,
        )
    if tuple(capacities) != layout.selected_devices:
        raise LatencyReportError("hardware.devices order differs from selected-device order")
    if hardware.get("all_selected_pairs_peer_access_disabled") is not True:
        raise LatencyReportError("local no-P2P topology assertion is not true")
    convention = _mapping(hardware.get("local_topology_convention"), "hardware.local_topology_convention")
    if convention.get("nvlink") is not False or convention.get("cuda_peer_access_expected") is not False:
        raise LatencyReportError("local topology is not the expected no-NVLink/no-P2P setup")
    warning = convention.get("warning")
    if not isinstance(warning, str) or "not directly comparable" not in warning:
        raise LatencyReportError("hardware provenance lacks the H100 comparability warning")
    return hardware, capacities


def _validate_peak_memory(
    value: Any,
    *,
    length: int,
    fingerprint: str,
    capacities: Mapping[int, int],
    layout: DeviceLayout,
    label: str,
) -> tuple[int, dict[str, int]]:
    peak = _mapping(value, f"{label}.cuda_peak_memory")
    expected_fields = {
        "schema_version": CUDA_PEAK_SCHEMA_VERSION,
        "status": "captured",
        "prefix_length": length,
        "request_fingerprint": fingerprint,
        "selected_device_ids": list(layout.selected_devices),
        "measurement_scope": (
            STAGED_CUDA_PEAK_MEASUREMENT_SCOPE
            if layout.staged_source_lifecycle
            else CUDA_PEAK_MEASUREMENT_SCOPE
        ),
        "allocator_peak_scope": CUDA_PEAK_ALLOCATOR_SCOPE,
        "driver_free_scope": CUDA_DRIVER_FREE_SCOPE,
        "synchronized_before_reset": True,
        "peak_counters_reset_after_baseline": True,
        "synchronized_after_cell": True,
    }
    for field, expected in expected_fields.items():
        if peak.get(field) != expected:
            raise LatencyReportError(f"{label}.cuda_peak_memory.{field} mismatch")
    devices = _mapping(peak.get("devices"), f"{label}.cuda_peak_memory.devices")
    if set(devices) != {str(device) for device in layout.selected_devices} or len(devices) != len(
        layout.selected_devices
    ):
        raise LatencyReportError(f"{label}.cuda_peak_memory.devices has wrong coverage")
    integer_fields = (
        "capacity_bytes",
        "driver_free_bytes_before",
        "driver_total_bytes_before",
        "driver_used_bytes_before",
        "allocated_bytes_before",
        "reserved_bytes_before",
        "peak_allocated_bytes_immediately_after_reset",
        "peak_reserved_bytes_immediately_after_reset",
        "driver_free_bytes_after",
        "driver_total_bytes_after",
        "driver_used_bytes_after",
        "allocated_bytes_after",
        "reserved_bytes_after",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "allocated_growth_to_peak_bytes",
        "reserved_growth_to_peak_bytes",
        "capacity_headroom_at_peak_reserved_bytes",
    )
    headrooms: dict[str, int] = {}
    for device in layout.selected_devices:
        item = _mapping(devices[str(device)], f"{label}.cuda_peak_memory.devices.{device}")
        for field in integer_fields:
            _integer(item.get(field), f"{label}.cuda_peak_memory.devices.{device}.{field}", minimum=0)
        capacity = int(item["capacity_bytes"])
        if capacity != capacities[device]:
            raise LatencyReportError(f"{label} peak capacity differs from hardware on device {device}")
        for suffix in ("before", "after"):
            total = int(item[f"driver_total_bytes_{suffix}"])
            free = int(item[f"driver_free_bytes_{suffix}"])
            used = int(item[f"driver_used_bytes_{suffix}"])
            allocated = int(item[f"allocated_bytes_{suffix}"])
            reserved = int(item[f"reserved_bytes_{suffix}"])
            if total != capacity or used != total - free or free > total:
                raise LatencyReportError(f"{label} driver memory arithmetic failed on device {device}")
            if allocated > reserved:
                raise LatencyReportError(f"{label} allocator endpoint arithmetic failed on device {device}")
        allocated_before = int(item["allocated_bytes_before"])
        reserved_before = int(item["reserved_bytes_before"])
        allocated_after = int(item["allocated_bytes_after"])
        reserved_after = int(item["reserved_bytes_after"])
        peak_allocated = int(item["peak_allocated_bytes"])
        peak_reserved = int(item["peak_reserved_bytes"])
        if item["peak_allocated_bytes_immediately_after_reset"] != allocated_before or item["peak_reserved_bytes_immediately_after_reset"] != reserved_before:
            raise LatencyReportError(f"{label} peak reset baseline is inconsistent on device {device}")
        if peak_allocated < max(allocated_before, allocated_after):
            raise LatencyReportError(f"{label} allocated peak is inconsistent on device {device}")
        if peak_reserved < max(reserved_before, reserved_after, peak_allocated) or peak_reserved > capacity:
            raise LatencyReportError(f"{label} reserved peak is inconsistent on device {device}")
        if item["allocated_growth_to_peak_bytes"] != peak_allocated - allocated_before or item["reserved_growth_to_peak_bytes"] != peak_reserved - reserved_before:
            raise LatencyReportError(f"{label} peak-growth arithmetic failed on device {device}")
        headroom = capacity - peak_reserved
        if item["capacity_headroom_at_peak_reserved_bytes"] != headroom:
            raise LatencyReportError(f"{label} peak headroom is inconsistent on device {device}")
        for numerator, fraction_name in (
            (peak_allocated, "peak_allocated_fraction_of_capacity"),
            (peak_reserved, "peak_reserved_fraction_of_capacity"),
        ):
            _same_number(
                item.get(fraction_name),
                numerator / capacity,
                f"{label}.cuda_peak_memory.devices.{device}.{fraction_name}",
            )
        headrooms[str(device)] = headroom
    return min(headrooms.values()), headrooms


def _validate_staged_cache_structure(
    value: Any, *, length: int, devices: Sequence[str], label: str
) -> None:
    records = _list(value, label)
    if len(records) != 40 or len(devices) != 40:
        raise LatencyReportError(f"{label} must cover exactly 40 source layers")
    shape = [1, 8, length, 128]
    tensor_nbytes = 1 * 8 * length * 128 * 2
    for layer, (raw, device) in enumerate(zip(records, devices, strict=True)):
        item = _mapping(raw, f"{label}[{layer}]")
        expected = {
            "layer": layer,
            "device": device,
            "key_shape": shape,
            "value_shape": shape,
            "key_dtype": "torch.bfloat16",
            "value_dtype": "torch.bfloat16",
            "key_nbytes": tensor_nbytes,
            "value_nbytes": tensor_nbytes,
        }
        if item != expected:
            raise LatencyReportError(f"{label}[{layer}] structure mismatch")


def _validate_staged_checkpoint_lifecycle(
    checkpoint: Mapping[str, Any],
    *,
    root: Path,
    length: int,
    fingerprint: str,
    attention_sha256: str,
    layout: DeviceLayout,
    label: str,
) -> Mapping[str, Any]:
    if checkpoint.get("source_model_live_during_timed_cell") is not False:
        raise LatencyReportError(f"{label} retained source model during timing")
    if checkpoint.get("source_cache_origin") != "preexisting_before_run_one_length":
        raise LatencyReportError(f"{label} did not start from a preexisting source cache")
    lifecycle = _mapping(
        checkpoint.get("source_cache_lifecycle"), f"{label}.source_cache_lifecycle"
    )
    session_id = lifecycle.get("staging_session_id")
    session_name = lifecycle.get("session_evidence_path")
    if (
        not isinstance(session_id, str)
        or len(session_id) != 32
        or any(character not in "0123456789abcdef" for character in session_id)
        or session_name != f"source_cache_staging_session_{session_id}.json"
    ):
        raise LatencyReportError(f"{label} has invalid staging-session binding")
    session_path = root / str(session_name)
    session_sidecar = _validate_sha_sidecar(session_path)
    if lifecycle.get("session_evidence_sha256") != session_sidecar.get("sha256"):
        raise LatencyReportError(f"{label} staging-session SHA mismatch")
    session = _mapping(_load_json(session_path), str(session_path))
    topology = _expected_staged_runtime_topology(layout)
    session_expected = {
        "schema_version": 1,
        "status": "source_released_before_target_load",
        "request_fingerprint": fingerprint,
        "staging_session_id": session_id,
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "runtime_topology": topology,
        "runtime_topology_sha256": canonical_digest(topology),
        "target_model_loaded_when_session_evidence_written": False,
        "source_model_live_when_session_evidence_written": False,
    }
    for field, wanted in session_expected.items():
        if session.get(field) != wanted:
            raise LatencyReportError(f"{session_path}.{field} mismatch")
    preparations = _mapping(session.get("preparations"), f"{session_path}.preparations")
    prepared_lengths = _list(
        session.get("prepared_missing_lengths"), f"{session_path}.prepared_missing_lengths"
    )
    if set(preparations) != {str(item) for item in prepared_lengths} or length not in prepared_lengths:
        raise LatencyReportError(f"{session_path} prepared-length map mismatch")
    prepared_total = sum(
        SOURCE_CACHE_BF16_BYTES_PER_TOKEN * int(item)
        for item in prepared_lengths
    )
    if session.get("cpu_staging_total_bytes") != prepared_total:
        raise LatencyReportError(f"{session_path} CPU staging total mismatch")
    host_gate = _mapping(
        session.get("host_memory_staging_gate"), f"{session_path}.host_memory_gate"
    )
    host_lengths = _list(
        host_gate.get("staged_prefix_lengths"), f"{session_path}.host_lengths"
    )
    host_cache_bytes = sum(
        SOURCE_CACHE_BF16_BYTES_PER_TOKEN * int(item) for item in host_lengths
    )
    host_required = (
        _integer(host_gate.get("source_weight_shard_bytes"), "host.source_weights", minimum=1)
        + _integer(host_gate.get("target_weight_shard_bytes"), "host.target_weights", minimum=1)
        + host_cache_bytes
        + 16 * (1 << 30)
    )
    if (
        host_gate.get("schema_version") != INPUT_SCHEMA_VERSION
        or host_gate.get("status") != "passed"
        or host_gate.get("scope") != "before_source_model_load"
        or host_gate.get("all_requested_source_caches_cpu_bytes") != host_cache_bytes
        or host_gate.get("conservative_operating_overhead_bytes") != 16 * (1 << 30)
        or host_gate.get("required_available_bytes") != host_required
        or _integer(host_gate.get("mem_available_bytes"), "host.available", minimum=1)
        < host_required
    ):
        raise LatencyReportError(f"{session_path} host-memory gate mismatch")
    preparation = _mapping(
        preparations.get(str(length)), f"{session_path}.preparations.{length}"
    )
    if lifecycle.get("preparation_evidence_sha256") != canonical_digest(preparation):
        raise LatencyReportError(f"{label} preparation digest mismatch")
    cache_bytes = SOURCE_CACHE_BF16_BYTES_PER_TOKEN * length
    prefix = _mapping(
        preparation.get("deterministic_prefix_identity"),
        f"{session_path}.preparations.{length}.prefix",
    )
    if (
        prefix.get("shape") != [1, length]
        or prefix.get("dtype") != "torch.int64"
        or prefix.get("device") != "cpu"
        or prefix.get("nbytes") != 8 * length
    ):
        raise LatencyReportError(f"{label} deterministic prefix structure mismatch")
    _sha256(prefix.get("sha256"), f"{label}.prefix.sha256")
    raw_hashes = (
        preparation.get("source_gpu_raw_bit_sha256"),
        preparation.get("cpu_staged_raw_bit_sha256"),
        preparation.get("raw_bit_sha256"),
    )
    for index, digest in enumerate(raw_hashes):
        _sha256(digest, f"{label}.preparation.raw_hash[{index}]")
    if len(set(raw_hashes)) != 1 or preparation.get(
        "gpu_to_cpu_raw_bit_hash_matches"
    ) is not True:
        raise LatencyReportError(f"{label} GPU/CPU preparation hashes differ")
    if (
        preparation.get("prefix_length") != length
        or preparation.get("cpu_cache_total_bytes") != cache_bytes
        or preparation.get("expected_cache_total_bytes") != cache_bytes
    ):
        raise LatencyReportError(f"{label} preparation length/bytes mismatch")
    prefill_devices = list(
        _expected_layer_devices(
            layout.source_prefill_devices,
            layout.source_prefill_layer_counts or (),
        )
    )
    _validate_staged_cache_structure(
        preparation.get("source_prefill_cache_structure"),
        length=length,
        devices=[f"cuda:{device}" for device in prefill_devices],
        label=f"{label}.source_prefill_cache_structure",
    )
    _validate_staged_cache_structure(
        preparation.get("cpu_cache_structure"),
        length=length,
        devices=["cpu"] * 40,
        label=f"{label}.cpu_cache_structure",
    )
    attention_path = root / "attention_backend.json"
    if sha256_file(attention_path) != attention_sha256:
        raise LatencyReportError(f"{label} attention SHA binding mismatch")
    attention = _mapping(_load_json(attention_path), str(attention_path))
    if session.get("source_attention") != attention.get("source"):
        raise LatencyReportError(f"{label} session source FA2 evidence differs")
    release = _mapping(session.get("source_model_release"), f"{session_path}.release")
    if (
        release.get("status") != "passed"
        or release.get("source_model_live") is not False
        or release.get("source_model_weakref_is_none") is not True
        or release.get("all_source_gpu_cache_references_deleted") is not True
        or release.get("gc_completed") is not True
        or release.get("cuda_synchronized_and_empty_cache_completed") is not True
        or release.get("all_prefill_devices_allocated_bytes_zero") is not True
        or release.get("all_prefill_devices_reserved_bytes_zero") is not True
        or release.get("target_model_loaded_at_proof_time") is not False
        or release.get("source_prefill_logical_devices")
        != list(layout.source_prefill_devices)
        or release.get("source_prefill_target_overlap_exact") != [5]
        or release.get("logical_5_reused_physical_device") != 7
        or release.get("runtime_topology_sha256")
        != canonical_digest(_expected_staged_runtime_topology(layout))
    ):
        raise LatencyReportError(f"{label} source-model release proof failed")
    workspace = _mapping(
        release.get("cuda_workspace_and_allocator_release"),
        f"{session_path}.release.workspace",
    )
    if (
        workspace.get("cublas_workspaces_clear_api")
        != "torch._C._cuda_clearCublasWorkspaces"
        or workspace.get("status") != "passed"
        or workspace.get("cublas_workspaces_clear_called") is not True
        or workspace.get("cublas_workspaces_clear_succeeded") is not True
        or workspace.get("source_model_deleted_and_gc_completed_before_clear")
        is not True
        or workspace.get("gc_completed_after_workspace_clear") is not True
        or workspace.get("empty_cache_completed_for_every_selected_device")
        is not True
        or workspace.get("synchronized_before_and_after") is not True
        or workspace.get("selected_device_ids")
        != list(layout.source_prefill_devices)
        or workspace.get("call_order")
        != [
            "source_model_deleted_and_gc_collected",
            "synchronize_before_cublas_workspace_clear",
            "torch._C._cuda_clearCublasWorkspaces",
            "gc_collect_after_cublas_workspace_clear",
            "torch.cuda.empty_cache_each_selected_device",
            "synchronize_after_empty_cache",
        ]
    ):
        raise LatencyReportError(f"{label} cuBLAS workspace release proof failed")
    release_before = _mapping(
        release.get("allocator_driver_before_release"),
        f"{session_path}.release.before",
    )
    release_after = _mapping(
        release.get("allocator_driver_after_release_before_target_load"),
        f"{session_path}.release.after",
    )
    expected_prefill_keys = {
        str(device) for device in layout.source_prefill_devices
    }
    if set(release_before) != expected_prefill_keys or set(
        release_after
    ) != expected_prefill_keys:
        raise LatencyReportError(f"{label} source release device coverage mismatch")
    for device, raw in release_before.items():
        item = _mapping(raw, f"{label}.source_release_before.{device}")
        allocated = _integer(
            item.get("allocated_bytes"), f"{label}.before.{device}.allocated", minimum=1
        )
        reserved = _integer(
            item.get("reserved_bytes"), f"{label}.before.{device}.reserved", minimum=1
        )
        free = _integer(
            item.get("driver_free_bytes"), f"{label}.before.{device}.free", minimum=0
        )
        total = _integer(
            item.get("driver_total_bytes"), f"{label}.before.{device}.total", minimum=1
        )
        used = _integer(
            item.get("driver_used_bytes"), f"{label}.before.{device}.used", minimum=0
        )
        if reserved < allocated or free > total or total - free != used:
            raise LatencyReportError(f"{label} source release before arithmetic failed")
    for device, raw in release_after.items():
        item = _mapping(raw, f"{label}.source_release_after.{device}")
        allocated = _integer(
            item.get("allocated_bytes"), f"{label}.after.{device}.allocated", minimum=0
        )
        reserved = _integer(
            item.get("reserved_bytes"), f"{label}.after.{device}.reserved", minimum=0
        )
        free = _integer(
            item.get("driver_free_bytes"), f"{label}.after.{device}.free", minimum=0
        )
        total = _integer(
            item.get("driver_total_bytes"), f"{label}.after.{device}.total", minimum=1
        )
        used = _integer(
            item.get("driver_used_bytes"), f"{label}.after.{device}.used", minimum=0
        )
        if allocated != 0 or reserved != 0 or free > total or total - free != used:
            raise LatencyReportError(f"{label} source model was not allocator-empty")
    lifecycle_expected = {
        "status": "complete",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "source_model_live_during_timed_cells": False,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "outer_peak_reset_after_restage_validation": True,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "mapper_timer_started": True,
        "deterministic_prefix_identity": prefix,
        "source_gpu_raw_bit_sha256": raw_hashes[0],
        "cpu_staged_raw_bit_sha256": raw_hashes[1],
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "cache_total_bytes": cache_bytes,
    }
    for field, wanted in lifecycle_expected.items():
        if lifecycle.get(field) != wanted:
            raise LatencyReportError(f"{label}.source_cache_lifecycle.{field} mismatch")
    if checkpoint.get("deterministic_prefix_identity") != prefix:
        raise LatencyReportError(f"{label} top-level prefix identity differs")
    restage = _mapping(lifecycle.get("restage"), f"{label}.restage")
    cache_layer_devices = list(
        _expected_layer_devices(layout.source_devices, layout.source_layer_counts or ())
    )
    if (
        restage.get("status") != "passed"
        or restage.get("policy") != SOURCE_CACHE_STAGING_POLICY
        or restage.get("prefix_length") != length
        or restage.get("untimed") is not True
        or restage.get("completed_before_outer_peak_reset") is not True
        or restage.get("source_model_live") is not False
        or restage.get("source_cache_gpu_resident_before_mapper_timer") is not True
        or restage.get("copy_semantics") != "cpu_to_cuda_blocking_no_dtype_cast"
        or restage.get("raw_bit_sha256") != raw_hashes[1]
        or restage.get("raw_bit_hash_matches_cpu_staging") is not True
        or restage.get("layer_devices")
        != [f"cuda:{device}" for device in cache_layer_devices]
        or restage.get("layer_counts") != list(layout.source_layer_counts or ())
        or restage.get("cache_total_bytes") != cache_bytes
        or restage.get("cuda_synchronized_and_empty_cache_before_outer_peak_reset")
        is not True
    ):
        raise LatencyReportError(f"{label} GPU restage proof failed")
    _validate_staged_cache_structure(
        restage.get("cache_structure"),
        length=length,
        devices=[f"cuda:{device}" for device in cache_layer_devices],
        label=f"{label}.restage.cache_structure",
    )
    cache_release = _mapping(
        lifecycle.get("restaged_cache_release"), f"{label}.restaged_cache_release"
    )
    release_snapshot = _mapping(
        cache_release.get("allocator_driver_after_release"),
        f"{label}.restaged_cache_release.snapshot",
    )
    if (
        cache_release.get("status") != "passed"
        or cache_release.get("restaged_cache_weakref_is_none") is not True
        or cache_release.get("released_before_outer_peak_finish") is not True
        or cache_release.get("cuda_synchronized_and_empty_cache_completed") is not True
        or cache_release.get("source_cache_staging_devices")
        != list(layout.source_devices)
        or set(release_snapshot) != {str(device) for device in layout.source_devices}
    ):
        raise LatencyReportError(f"{label} restaged-cache release proof failed")
    for device, raw in release_snapshot.items():
        item = _mapping(raw, f"{label}.restaged_release.{device}")
        allocated = _integer(
            item.get("allocated_bytes"), f"{label}.restaged.{device}.allocated", minimum=0
        )
        reserved = _integer(
            item.get("reserved_bytes"), f"{label}.restaged.{device}.reserved", minimum=0
        )
        free = _integer(
            item.get("driver_free_bytes"), f"{label}.restaged.{device}.free", minimum=0
        )
        total = _integer(
            item.get("driver_total_bytes"), f"{label}.restaged.{device}.total", minimum=1
        )
        used = _integer(
            item.get("driver_used_bytes"), f"{label}.restaged.{device}.used", minimum=0
        )
        if allocated != 0 or reserved != 0 or free > total or total - free != used:
            raise LatencyReportError(f"{label} restaged-cache release proof failed")
    return lifecycle


def _validate_staged_sessions(
    *,
    root: Path,
    lengths: Sequence[int],
    checkpoints: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    attention: Mapping[str, Any],
    mapper_setup: Mapping[str, Any],
    metrics: Mapping[str, Any],
    fingerprint: str,
    layout: DeviceLayout,
) -> None:
    """Validate each staged preparation session once and bind resume semantics."""

    if len(checkpoints) != len(lengths):
        raise LatencyReportError("staged session validation lacks every checkpoint")
    source_model = _mapping(runtime.get("source_model"), "runtime.source_model")
    target_model = _mapping(runtime.get("target_model"), "runtime.target_model")
    source_config = _mapping(
        source_model.get("bound_config_fields"),
        "runtime.source_model.bound_config_fields",
    )
    target_config = _mapping(
        target_model.get("bound_config_fields"),
        "runtime.target_model.bound_config_fields",
    )
    source_vocab = _integer(
        source_config.get("vocab_size"), "runtime.source_model.vocab_size", minimum=1
    )
    target_vocab = _integer(
        target_config.get("vocab_size"), "runtime.target_model.vocab_size", minimum=1
    )
    if source_vocab != target_vocab:
        raise LatencyReportError("source and target vocabularies differ")
    if (
        source_config.get("model_type") != "qwen3"
        or target_config.get("model_type") != "qwen3"
        or source_config.get("num_hidden_layers") != 40
        or target_config.get("num_hidden_layers") != 64
        or source_config.get("num_key_value_heads") != 8
        or target_config.get("num_key_value_heads") != 8
        or source_config.get("head_dim") != 128
        or target_config.get("head_dim") != 128
    ):
        raise LatencyReportError("staged runtime model architecture binding mismatch")

    ordered_session_ids: list[str] = []
    first_index_by_session: dict[str, int] = {}
    session_names: dict[str, str] = {}
    previous_session: str | None = None
    closed_sessions: set[str] = set()
    for index, checkpoint in enumerate(checkpoints):
        lifecycle = _mapping(
            checkpoint.get("source_cache_lifecycle"),
            f"staged checkpoint[{index}].source_cache_lifecycle",
        )
        session_id = lifecycle.get("staging_session_id")
        if not isinstance(session_id, str):
            raise LatencyReportError("staged checkpoint session id is malformed")
        if session_id != previous_session:
            if previous_session is not None:
                closed_sessions.add(previous_session)
            if session_id in closed_sessions:
                raise LatencyReportError("staged session A-B-A re-entry is forbidden")
            ordered_session_ids.append(session_id)
            first_index_by_session[session_id] = index
            previous_session = session_id
        session_names[session_id] = str(lifecycle.get("session_evidence_path"))

    topology = _expected_staged_runtime_topology(layout)
    expected_prefill_devices = list(
        _expected_layer_devices(
            layout.source_prefill_devices,
            layout.source_prefill_layer_counts or (),
        )
    )
    expected_source_sharding = _mapping(
        _mapping(
            mapper_setup.get("model_layer_sharding"),
            "mapper_setup.model_layer_sharding",
        ).get("source_model_prefill"),
        "mapper_setup.model_layer_sharding.source_model_prefill",
    )
    all_length_list = list(lengths)
    source_weight_bytes = _integer(
        source_model.get("weight_shard_total_nbytes"),
        "runtime.source_model.weight_shard_total_nbytes",
        minimum=1,
    )
    target_weight_bytes = _integer(
        target_model.get("weight_shard_total_nbytes"),
        "runtime.target_model.weight_shard_total_nbytes",
        minimum=1,
    )
    all_cache_bytes = sum(
        SOURCE_CACHE_BF16_BYTES_PER_TOKEN * length for length in lengths
    )

    loaded_sessions: dict[str, Mapping[str, Any]] = {}
    for session_id in ordered_session_ids:
        session_path = root / session_names[session_id]
        session = _mapping(_load_json(session_path), str(session_path))
        loaded_sessions[session_id] = session
        first_index = first_index_by_session[session_id]
        expected_prepared_lengths = list(lengths[first_index:])
        raw_prepared_lengths = _list(
            session.get("prepared_missing_lengths"),
            f"{session_path}.prepared_missing_lengths",
        )
        prepared_lengths = [
            _integer(item, f"{session_path}.prepared_missing_lengths[]", minimum=1)
            for item in raw_prepared_lengths
        ]
        if (
            prepared_lengths != expected_prepared_lengths
            or len(set(prepared_lengths)) != len(prepared_lengths)
        ):
            raise LatencyReportError(
                f"{session_path} does not contain the canonical missing suffix"
            )
        if (
            session.get("request_fingerprint") != fingerprint
            or session.get("runtime_topology") != topology
            or session.get("runtime_topology_sha256")
            != canonical_digest(topology)
            or session.get("source_vocabulary_size") != source_vocab
            or session.get("source_attention") != attention.get("source")
            or session.get("source_model_prefill_sharding")
            != expected_source_sharding
        ):
            raise LatencyReportError(f"{session_path} immutable source binding mismatch")

        host_gate = _mapping(
            session.get("host_memory_staging_gate"),
            f"{session_path}.host_memory_staging_gate",
        )
        required_host_bytes = (
            source_weight_bytes
            + target_weight_bytes
            + all_cache_bytes
            + 16 * (1 << 30)
        )
        expected_host_fields = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "status": "passed",
            "scope": "before_source_model_load",
            "required_available_bytes": required_host_bytes,
            "source_weight_shard_bytes": source_weight_bytes,
            "target_weight_shard_bytes": target_weight_bytes,
            "all_requested_source_caches_cpu_bytes": all_cache_bytes,
            "staged_prefix_lengths": all_length_list,
            "conservative_operating_overhead_bytes": 16 * (1 << 30),
        }
        if any(host_gate.get(field) != wanted for field, wanted in expected_host_fields.items()):
            raise LatencyReportError(f"{session_path} host-memory gate binding mismatch")
        if _integer(
            host_gate.get("mem_available_bytes"),
            f"{session_path}.host_memory_staging_gate.mem_available_bytes",
            minimum=1,
        ) < required_host_bytes:
            raise LatencyReportError(f"{session_path} host-memory gate did not pass")

        preparations = _mapping(
            session.get("preparations"), f"{session_path}.preparations"
        )
        if set(preparations) != {str(length) for length in prepared_lengths}:
            raise LatencyReportError(f"{session_path} preparation coverage mismatch")
        prepared_total = 0
        for length in prepared_lengths:
            preparation = _mapping(
                preparations[str(length)],
                f"{session_path}.preparations.{length}",
            )
            cache_bytes = SOURCE_CACHE_BF16_BYTES_PER_TOKEN * length
            prefix = _mapping(
                preparation.get("deterministic_prefix_identity"),
                f"{session_path}.preparations.{length}.prefix",
            )
            expected_preparation_fields = {
                "schema_version": INPUT_SCHEMA_VERSION,
                "status": "passed",
                "policy": SOURCE_CACHE_STAGING_POLICY,
                "prefix_length": length,
                "source_prefill_timed": False,
                "cpu_staging_timed": False,
                "copy_semantics": "detach_to_cpu_blocking_exact_clone",
                "full_raw_bit_sha256_computed": True,
                "gpu_to_cpu_raw_bit_hash_matches": True,
                "source_prefill_layer_devices": [
                    f"cuda:{device}" for device in expected_prefill_devices
                ],
                "cpu_cache_total_bytes": cache_bytes,
                "expected_cache_total_bytes": cache_bytes,
            }
            if any(
                preparation.get(field) != wanted
                for field, wanted in expected_preparation_fields.items()
            ):
                raise LatencyReportError(
                    f"{session_path}.preparations.{length} contract mismatch"
                )
            if prefix != {
                "shape": [1, length],
                "dtype": "torch.int64",
                "device": "cpu",
                "nbytes": 8 * length,
                "sha256": prefix.get("sha256"),
            }:
                raise LatencyReportError(
                    f"{session_path}.preparations.{length} prefix mismatch"
                )
            _sha256(prefix.get("sha256"), f"{session_path}.prefix.sha256")
            hashes = [
                _sha256(
                    preparation.get(field),
                    f"{session_path}.preparations.{length}.{field}",
                )
                for field in (
                    "source_gpu_raw_bit_sha256",
                    "cpu_staged_raw_bit_sha256",
                    "raw_bit_sha256",
                )
            ]
            if len(set(hashes)) != 1:
                raise LatencyReportError(
                    f"{session_path}.preparations.{length} raw hashes differ"
                )
            _validate_staged_cache_structure(
                preparation.get("source_prefill_cache_structure"),
                length=length,
                devices=[f"cuda:{device}" for device in expected_prefill_devices],
                label=f"{session_path}.preparations.{length}.gpu_structure",
            )
            _validate_staged_cache_structure(
                preparation.get("cpu_cache_structure"),
                length=length,
                devices=["cpu"] * 40,
                label=f"{session_path}.preparations.{length}.cpu_structure",
            )
            prepared_total += cache_bytes
        if session.get("cpu_staging_total_bytes") != prepared_total:
            raise LatencyReportError(f"{session_path} CPU staging byte total mismatch")

    reused = _list(
        metrics.get("reused_checkpoint_lengths"),
        "metrics.reused_checkpoint_lengths",
    )
    active = metrics.get("active_staging_session")
    if reused == all_length_list:
        if active is not None:
            raise LatencyReportError(
                "finalization-only staged resume must not claim an active session"
            )
    else:
        if not ordered_session_ids:
            raise LatencyReportError("staged run has no preparation session")
        latest_id = ordered_session_ids[-1]
        latest_session = loaded_sessions[latest_id]
        if first_index_by_session[latest_id] != len(reused):
            raise LatencyReportError(
                "metrics reused prefix does not start the active staging session"
            )
        expected_active = {
            "staging_session_id": latest_id,
            "session_evidence_path": session_names[latest_id],
            "session_evidence_sha256": sha256_file(root / session_names[latest_id]),
            "prepared_missing_lengths": list(lengths[len(reused) :]),
            "host_memory_staging_gate": latest_session["host_memory_staging_gate"],
        }
        if active != expected_active:
            raise LatencyReportError("metrics.active_staging_session binding mismatch")


def _validate_checkpoint(
    value: Any,
    *,
    length: int,
    fingerprint: str,
    attention_sha256: str,
    warmup: int,
    repeats: int,
    capacities: Mapping[int, int],
    layout: DeviceLayout,
    root: Path,
    label: str,
) -> tuple[Mapping[str, Any], int, Mapping[str, int]]:
    checkpoint = _mapping(value, label)
    expected_fields = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "attention_backend_assertion_sha256": attention_sha256,
        "prefix_length": length,
        "batch_size": 1,
        "source_prefill_timed": False,
        "source_cache_length": length,
        "latency_protocol": PAPER_MAIN_PROTOCOL,
        "nested_cuda_peak_memory_contract": NESTED_PEAK_CONTRACT,
        "timed_transfer_fast_path": TIMED_FAST_PATH_CONTRACT,
    }
    for field, expected in expected_fields.items():
        if checkpoint.get(field) != expected:
            raise LatencyReportError(f"{label}.{field} mismatch")
    lifecycle: Mapping[str, Any] | None = None
    if layout.staged_source_lifecycle:
        lifecycle = _validate_staged_checkpoint_lifecycle(
            checkpoint,
            root=root,
            length=length,
            fingerprint=fingerprint,
            attention_sha256=attention_sha256,
            layout=layout,
            label=label,
        )
    contaminated = [field for field in SUPPLEMENTAL_FIELDS if field in checkpoint]
    if contaminated:
        raise LatencyReportError(f"{label} mixes supplemental fields into paper-main cell: {contaminated}")

    preflight = _mapping(checkpoint.get("strict_transfer_preflight"), f"{label}.strict_transfer_preflight")
    preflight_expected = {
        "status": "passed",
        "untimed": True,
        "strict_validation": True,
        "rope_singularity_reduction": True,
        "source_content_finite_reduction": True,
        "mapped_kv_finite_reduction": True,
        "validated_target_layers": 64,
        "source_cache_length": length,
        "mapped_cache_length": length,
        "mapped_cache_dtypes": ["torch.bfloat16"],
    }
    for field, expected in preflight_expected.items():
        if preflight.get(field) != expected:
            raise LatencyReportError(f"{label}.strict_transfer_preflight.{field} mismatch")
    _number(
        preflight.get("wall_ms_diagnostic_not_latency"),
        f"{label}.strict_transfer_preflight.wall_ms_diagnostic_not_latency",
        minimum=0.0,
    )

    target = _validate_latency_summary(
        checkpoint.get("target_reprefill"),
        label=f"{label}.target_reprefill",
        warmup=warmup,
        repeats=repeats,
    )
    target_expected = {
        "measurement_scope": "target_transformer_reprefill",
        "batch_size": 1,
        "input_length": length,
        "lm_head_included": False,
        "kv_cache_output": True,
    }
    for field, expected in target_expected.items():
        if target.get(field) != expected:
            raise LatencyReportError(f"{label}.target_reprefill.{field} mismatch")

    mapper = _validate_latency_summary(
        checkpoint.get("mapper_handoff_end_to_end"),
        label=f"{label}.mapper_handoff_end_to_end",
        warmup=warmup,
        repeats=repeats,
    )
    mapper_expected = {
        "measurement_scope": "source_cache_to_mapped_target_cache",
        "source_prefill_included": False,
        "includes_cache_build": True,
        "includes_inverse_and_target_rope": True,
        "includes_source_cache_device_handoff": True,
        "mapper_weights_resident_on_target_devices": True,
    }
    for field, expected in mapper_expected.items():
        if mapper.get(field) != expected:
            raise LatencyReportError(f"{label}.mapper_handoff_end_to_end.{field} mismatch")
    mapper_median = _number(mapper.get("median"), f"{label}.mapper_handoff_end_to_end.median", minimum=0.0)
    if mapper_median <= 0.0:
        raise LatencyReportError(f"{label} mapper median must be positive")
    speedup = _number(
        checkpoint.get("median_speedup_target_reprefill_over_mapper"),
        f"{label}.median_speedup_target_reprefill_over_mapper",
        minimum=0.0,
    )
    expected_speedup = float(target["median"]) / mapper_median
    if not math.isclose(speedup, expected_speedup, rel_tol=1e-12, abs_tol=1e-12):
        raise LatencyReportError(f"{label} median speedup is not the ratio of medians")
    minimum_headroom, headrooms = _validate_peak_memory(
        checkpoint.get("cuda_peak_memory"),
        length=length,
        fingerprint=fingerprint,
        capacities=capacities,
        layout=layout,
        label=label,
    )
    if lifecycle is not None:
        peak_devices = _mapping(
            checkpoint["cuda_peak_memory"]["devices"], f"{label}.peak.devices"
        )
        release_devices = _mapping(
            lifecycle["restaged_cache_release"]["allocator_driver_after_release"],
            f"{label}.release.devices",
        )
        total = int(lifecycle["cache_total_bytes"])
        for device, count in zip(
            layout.source_devices, layout.source_layer_counts or (), strict=True
        ):
            item = _mapping(peak_devices[str(device)], f"{label}.peak.{device}")
            expected_before = total * count // 40
            if (
                item.get("allocated_bytes_before") != expected_before
                or item.get("reserved_bytes_before", -1) < expected_before
                or item.get("allocated_bytes_after") != 0
                or item.get("reserved_bytes_after") != 0
                or release_devices[str(device)].get("allocated_bytes")
                != item.get("allocated_bytes_after")
                or release_devices[str(device)].get("reserved_bytes")
                != item.get("reserved_bytes_after")
            ):
                raise LatencyReportError(
                    f"{label} staged peak/release binding failed on cuda:{device}"
                )
    return checkpoint, minimum_headroom, headrooms


def _manifest_names(root: Path, lengths: Sequence[int]) -> list[str]:
    session_names: list[str] = []
    for length in lengths:
        checkpoint = _mapping(
            _load_json(root / f"length_{length:05d}.json"),
            f"length_{length:05d}.json",
        )
        lifecycle = checkpoint.get("source_cache_lifecycle")
        if lifecycle is not None:
            item = _mapping(lifecycle, f"length_{length:05d}.source_cache_lifecycle")
            name = item.get("session_evidence_path")
            if (
                not isinstance(name, str)
                or PurePosixPath(name).is_absolute()
                or len(PurePosixPath(name).parts) != 1
                or not name.startswith("source_cache_staging_session_")
                or not name.endswith(".json")
            ):
                raise LatencyReportError("checkpoint staging-session path is unsafe")
            if name not in session_names:
                session_names.append(name)
    return [
        "run_definition.json",
        "preload_gpu_idle_gate.json",
        "hardware.json",
        "attention_backend.json",
        "mapper_setup.json",
        *session_names,
        "telemetry_before.json",
        "telemetry_after.json",
        "metrics.json",
        "metrics.csv",
        *(f"length_{length:05d}.json" for length in lengths),
    ]


def _validate_manifest(
    root: Path,
    *,
    fingerprint: str,
    lengths: Sequence[int],
) -> Mapping[str, Any]:
    path = root / "results.manifest.json"
    manifest = _mapping(_load_json(path), str(path))
    _validate_sha_sidecar(path)
    expected_fields = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "lengths": list(lengths),
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise LatencyReportError(f"{path}.{field} mismatch")
    artifacts = _list(manifest.get("artifacts"), f"{path}.artifacts")
    expected_names = _manifest_names(root, lengths)
    names: list[str] = []
    for index, raw in enumerate(artifacts):
        item = _mapping(raw, f"{path}.artifacts[{index}]")
        if set(item) != {"path", "sha256", "nbytes"}:
            raise LatencyReportError(f"{path}.artifacts[{index}] has malformed fields")
        name = item.get("path")
        if not isinstance(name, str):
            raise LatencyReportError(f"{path}.artifacts[{index}].path must be a string")
        pure = PurePosixPath(name)
        if pure.is_absolute() or len(pure.parts) != 1 or ".." in pure.parts:
            raise LatencyReportError(f"{path} contains an unsafe artifact path: {name!r}")
        names.append(name)
        artifact_path = root / name
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise LatencyReportError(f"manifest artifact is missing or a symlink: {artifact_path}")
        digest = _sha256(item.get("sha256"), f"{path}.artifacts[{index}].sha256")
        nbytes = _integer(item.get("nbytes"), f"{path}.artifacts[{index}].nbytes", minimum=0)
        if sha256_file(artifact_path) != digest or artifact_path.stat().st_size != nbytes:
            raise LatencyReportError(f"manifest identity mismatch for {artifact_path}")
        sidecar = _validate_sha_sidecar(artifact_path)
        if sidecar.get("sha256") != digest or sidecar.get("nbytes") != nbytes:
            raise LatencyReportError(f"manifest and sidecar disagree for {artifact_path}")
    if names != expected_names or len(names) != len(set(names)):
        raise LatencyReportError(
            f"results manifest artifact whitelist/order mismatch: {names!r} != {expected_names!r}"
        )
    if manifest.get("artifacts_digest") != canonical_digest(artifacts):
        raise LatencyReportError("results manifest artifacts_digest is inconsistent")
    return manifest


def _validate_input_csv(
    path: Path,
    *,
    lengths: Sequence[int],
    checkpoints: Sequence[Mapping[str, Any]],
) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != INPUT_CSV_COLUMNS:
                raise LatencyReportError(f"{path} has an unexpected header")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise LatencyReportError(f"cannot read latency metrics CSV {path}: {exc}") from exc
    expected_pairs = [
        (length, metric)
        for length in lengths
        for metric in ("target_reprefill", "mapper_handoff_end_to_end")
    ]
    pairs: list[tuple[int, str]] = []
    by_length = {int(checkpoint["prefix_length"]): checkpoint for checkpoint in checkpoints}
    for row_index, row in enumerate(rows):
        if None in row or set(row) != set(INPUT_CSV_COLUMNS):
            raise LatencyReportError(f"{path} row {row_index + 2} is malformed")
        try:
            length_text = row["prefix_length"]
            count_text = row["count"]
            if str(int(length_text)) != length_text or str(int(count_text)) != count_text:
                raise ValueError
            length = int(length_text)
            count = int(count_text)
        except (TypeError, ValueError) as exc:
            raise LatencyReportError(f"{path} row {row_index + 2} has invalid integers") from exc
        metric = row["metric"]
        pairs.append((length, metric))
        if (length, metric) not in expected_pairs:
            raise LatencyReportError(f"{path} has an unexpected row {(length, metric)!r}")
        summary = _mapping(by_length[length].get(metric), f"checkpoint[{length}].{metric}")
        if count != summary.get("count"):
            raise LatencyReportError(f"{path} count disagrees for {(length, metric)!r}")
        for csv_field, summary_field in (
            ("mean_ms", "mean"),
            ("median_ms", "median"),
            ("p10_ms", "p10"),
            ("p50_ms", "p50"),
            ("p90_ms", "p90"),
            ("std_population_ms", "std_population"),
            ("min_ms", "min"),
            ("max_ms", "max"),
        ):
            try:
                parsed = float(row[csv_field])
            except (TypeError, ValueError) as exc:
                raise LatencyReportError(f"{path} has invalid {csv_field}") from exc
            if not math.isfinite(parsed) or not math.isclose(
                parsed, float(summary[summary_field]), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise LatencyReportError(f"{path} {csv_field} disagrees for {(length, metric)!r}")
        if row["timing_backends"] != ";".join(summary["timing_backends"]):
            raise LatencyReportError(f"{path} timing backend disagrees for {(length, metric)!r}")
        if row["source_prefill_included"] != "False":
            raise LatencyReportError(f"{path} claims source prefill is included")
    if pairs != expected_pairs or len(pairs) != len(set(pairs)):
        raise LatencyReportError(f"{path} rows are missing, duplicated, extra, or reordered")


def _validate_preload_gate_evidence(
    value: Any, *, label: str, layout: DeviceLayout
) -> Mapping[str, Any]:
    evidence = _mapping(value, label)
    expected_flags = {
        "schema_version": 1,
        "status": "passed",
        "policy": layout.preload_policy,
        "all_expensive_identity_hashes_completed_before_gate": True,
        "completed_before_cuda_model_load": True,
    }
    for field, expected in expected_flags.items():
        if evidence.get(field) != expected:
            raise LatencyReportError(f"{label}.{field} mismatch")
    bindings = _list(evidence.get("selected_gpu_bindings"), f"{label}.selected_gpu_bindings")
    expected_physical = list(layout.selected_physical_devices)
    if len(bindings) != len(layout.selected_devices):
        raise LatencyReportError(f"{label} selected GPU coverage mismatch")
    uuids: list[str] = []
    for index, (raw, logical, physical) in enumerate(
        zip(bindings, layout.selected_devices, expected_physical, strict=True)
    ):
        item = _mapping(raw, f"{label}.selected_gpu_bindings[{index}]")
        if set(item) != {"logical_index", "physical_index", "gpu_uuid"}:
            raise LatencyReportError(f"{label} selected GPU binding fields mismatch")
        if item.get("logical_index") != logical or item.get("physical_index") != physical:
            raise LatencyReportError(f"{label} selected logical/physical GPU mapping mismatch")
        uuid = item.get("gpu_uuid")
        if not isinstance(uuid, str) or not uuid:
            raise LatencyReportError(f"{label} selected GPU UUID is invalid")
        uuids.append(uuid)
    if len(set(uuids)) != len(uuids):
        raise LatencyReportError(f"{label} selected GPU UUIDs are duplicated")
    attempts = _integer(evidence.get("attempt_count"), f"{label}.attempt_count", minimum=12)
    rejected = _integer(evidence.get("rejected_attempt_count"), f"{label}.rejected_attempt_count", minimum=0)
    if rejected != attempts - 12:
        raise LatencyReportError(f"{label} attempt accounting mismatch")
    samples = _list(
        evidence.get("final_consecutive_sandwich_samples"),
        f"{label}.final_consecutive_sandwich_samples",
    )
    if len(samples) != 12:
        raise LatencyReportError(f"{label} final idle streak must contain 12 samples")
    expected_sequences = list(range(attempts - 11, attempts + 1))
    if [item.get("sequence") if isinstance(item, Mapping) else None for item in samples] != expected_sequences:
        raise LatencyReportError(f"{label} sample sequence mismatch")
    for sample_index, raw in enumerate(samples):
        sample = _mapping(raw, f"{label}.samples[{sample_index}]")
        if sample.get("selected_compute_apps_before") != [] or sample.get("selected_compute_apps_after") != []:
            raise LatencyReportError(f"{label} selected compute app contamination")
        for phase in ("memory_before", "memory_after"):
            memory = _list(sample.get(phase), f"{label}.samples[{sample_index}].{phase}")
            if len(memory) != len(bindings):
                raise LatencyReportError(f"{label} memory coverage mismatch")
            for device_index, (raw_memory, binding) in enumerate(zip(memory, bindings, strict=True)):
                item = _mapping(raw_memory, f"{label}.samples[{sample_index}].{phase}[{device_index}]")
                expected_identity = dict(binding)
                if set(item) != {*expected_identity, "memory_used_mib"} or any(
                    item.get(field) != expected_identity[field] for field in expected_identity
                ):
                    raise LatencyReportError(f"{label} memory row is not bound to selected GPU")
                used = _integer(
                    item.get("memory_used_mib"), f"{label}.memory_used_mib", minimum=0
                )
                if used > 64:
                    raise LatencyReportError(f"{label} memory limit was exceeded")
    for field in ("started_utc", "completed_utc"):
        if not isinstance(evidence.get(field), str) or not evidence[field]:
            raise LatencyReportError(f"{label}.{field} is missing")
    return evidence


def _validate_preload_gate_record(
    value: Any,
    *,
    definition: Mapping[str, Any],
    label: str,
    layout: DeviceLayout,
) -> Mapping[str, Any]:
    record = _mapping(value, label)
    if set(record) != {
        "schema_version",
        "request_fingerprint",
        "run_definition_gate_evidence_sha256",
        "active_preload_gpu_idle_gate",
        "immediate_preload_gpu_idle_sandwich",
    }:
        raise LatencyReportError(f"{label} fields mismatch")
    definition_evidence = _mapping(
        definition.get("preload_gpu_idle_gate"), "run_definition.preload_gpu_idle_gate"
    )
    if (
        record.get("schema_version") != 1
        or record.get("request_fingerprint") != definition.get("request_fingerprint")
        or record.get("run_definition_gate_evidence_sha256")
        != canonical_digest(definition_evidence)
    ):
        raise LatencyReportError(f"{label} definition/fingerprint binding mismatch")
    active = _validate_preload_gate_evidence(
        record.get("active_preload_gpu_idle_gate"),
        label=f"{label}.active_preload_gpu_idle_gate",
        layout=layout,
    )
    immediate = _mapping(
        record.get("immediate_preload_gpu_idle_sandwich"),
        f"{label}.immediate_preload_gpu_idle_sandwich",
    )
    expected_immediate_fields = {
        "status": "passed",
        "scope": "immediately_before_cuda_model_load",
        "sandwich_interval_seconds": 1.0,
        "selected_compute_apps_before": [],
        "selected_compute_apps_after": [],
    }
    for field, expected in expected_immediate_fields.items():
        if immediate.get(field) != expected:
            raise LatencyReportError(f"{label}.immediate.{field} mismatch")
    bindings = _list(active.get("selected_gpu_bindings"), f"{label}.bindings")
    for phase in ("memory_before", "memory_after"):
        memory = _list(immediate.get(phase), f"{label}.immediate.{phase}")
        if len(memory) != len(bindings):
            raise LatencyReportError(f"{label}.immediate memory coverage mismatch")
        for raw, binding in zip(memory, bindings, strict=True):
            item = _mapping(raw, f"{label}.immediate.{phase}.row")
            if set(item) != {*binding, "memory_used_mib"} or any(
                item.get(field) != binding[field] for field in binding
            ):
                raise LatencyReportError(f"{label}.immediate memory identity mismatch")
            used = _integer(item.get("memory_used_mib"), f"{label}.immediate.memory", minimum=0)
            if used > 64:
                raise LatencyReportError(f"{label}.immediate memory limit exceeded")
    if not isinstance(immediate.get("completed_utc"), str) or not immediate["completed_utc"]:
        raise LatencyReportError(f"{label}.immediate.completed_utc missing")
    return record


def _validate_definition(
    value: Any,
    *,
    label: str,
    lengths: Sequence[int],
    warmup: int,
    repeats: int,
) -> tuple[Mapping[str, Any], str, DeviceLayout]:
    definition = _mapping(value, label)
    if definition.get("schema_version") != INPUT_SCHEMA_VERSION or definition.get("kind") != "qwen3_kv_cache_transfer_latency":
        raise LatencyReportError(f"{label} is not a schema-{INPUT_SCHEMA_VERSION} latency definition")
    fingerprint = _sha256(definition.get("request_fingerprint"), f"{label}.request_fingerprint")
    unhashed = dict(definition)
    unhashed.pop("request_fingerprint", None)
    if canonical_digest(unhashed) != fingerprint:
        raise LatencyReportError(f"{label}.request_fingerprint does not hash the definition")
    layout = _device_layout_from_definition(definition, label=label)
    cli = _mapping(definition.get("cli"), f"{label}.cli")
    expected_cli = {
        "mode": "production",
        "lengths": list(lengths),
        "warmup": warmup,
        "repeats": repeats,
        "supplemental_diagnostics": False,
        "profile_warmup": None,
        "profile_repeats": None,
        "mapper_token_chunk": 1024,
        "attention": "flash_attention_2",
        "seed": 20260813,
        "source_devices": list(layout.source_devices),
        "target_devices": list(layout.target_devices),
        "preload_idle_memory_limit_mib": 64,
        "preload_idle_samples": 12,
        "preload_idle_sample_interval_seconds": 5.0,
        "preload_idle_timeout_seconds": 86400.0,
    }
    for field, expected in expected_cli.items():
        if cli.get(field) != expected:
            raise LatencyReportError(f"{label}.cli.{field} mismatch")
    if layout.staged_source_lifecycle:
        staged_cli = {
            "stage_source_caches_on_cpu_before_cells": True,
            "source_prefill_devices": list(layout.source_prefill_devices),
            "source_prefill_layer_counts": list(
                layout.source_prefill_layer_counts or ()
            ),
        }
        for field, expected in staged_cli.items():
            if cli.get(field) != expected:
                raise LatencyReportError(f"{label}.cli.{field} mismatch")
    if layout.requires_exact_layer_evidence:
        if cli.get("source_layer_counts") != list(layout.source_layer_counts or ()):
            raise LatencyReportError(f"{label}.cli.source_layer_counts mismatch")
        if cli.get("target_layer_counts") != list(layout.target_layer_counts or ()):
            raise LatencyReportError(f"{label}.cli.target_layer_counts mismatch")
    else:
        for field in ("source_layer_counts", "target_layer_counts"):
            if field in cli and cli[field] is not None:
                raise LatencyReportError(
                    f"{label}.cli.{field} is not valid for the historical profile"
                )
    resolved = _mapping(definition.get("resolved"), f"{label}.resolved")
    resolved_base = {
        "lengths": list(lengths),
        "latency_protocol": "paper_main_only",
        "profile_warmup": None,
        "profile_repeats": None,
    }
    if any(resolved.get(field) != expected for field, expected in resolved_base.items()):
        raise LatencyReportError(f"{label}.resolved is not the paper-main-only protocol")
    extra_resolved = set(resolved).difference(resolved_base)
    if extra_resolved not in (set(), {"layer_sharding"}):
        raise LatencyReportError(f"{label}.resolved has unexpected fields")
    if layout.requires_exact_layer_evidence and "layer_sharding" not in resolved:
        raise LatencyReportError(f"{label}.resolved lacks exact layer_sharding")
    if "layer_sharding" in resolved:
        _validate_layer_sharding_contract(
            resolved["layer_sharding"],
            layout=layout,
            label=f"{label}.resolved.layer_sharding",
        )
    _validate_preload_gate_evidence(
        definition.get("preload_gpu_idle_gate"),
        label=f"{label}.preload_gpu_idle_gate",
        layout=layout,
    )
    _validate_project_code(definition.get("project_code"), label=f"{label}.project_code")
    runtime = _validate_runtime_artifacts(
        definition.get("runtime_artifacts"),
        label=f"{label}.runtime_artifacts",
    )
    backend = _mapping(definition.get("backend"), f"{label}.backend")
    if backend.get("requested_attention") != "flash_attention_2" or backend.get("flash_attention_2_available") is not True:
        raise LatencyReportError(f"{label}.backend does not establish FlashAttention-2")
    topology = _mapping(definition.get("topology"), f"{label}.topology")
    if topology.get("cuda_visible_devices_environment") != layout.cuda_visibility or topology.get("selected_devices") != list(layout.selected_devices):
        raise LatencyReportError(f"{label}.topology device mapping mismatch")
    if topology.get("all_selected_pairs_peer_access_disabled") is not True:
        raise LatencyReportError(f"{label}.topology does not establish no-P2P execution")
    if layout.staged_source_lifecycle and topology.get(
        "staged_source_lifecycle"
    ) != _expected_staged_runtime_topology(layout):
        raise LatencyReportError(f"{label}.topology staged lifecycle mismatch")
    analytical = _mapping(definition.get("analytical_memory"), f"{label}.analytical_memory")
    if analytical.get("paper_main_cell_only") is not True or analytical.get("supplemental_diagnostics_enabled") is not False:
        raise LatencyReportError(f"{label}.analytical_memory is not paper-main-only")
    if analytical.get("resident_mapper_fp32_bytes") != MAPPER_PARAMETER_BYTES:
        raise LatencyReportError(f"{label}.analytical_memory mapper size mismatch")
    if layout.requires_exact_layer_evidence:
        expected_analytical_sharding: dict[str, Any] = {
            role: {
                "mode": "explicit_exact_counts",
                "resolved_layer_counts": list(_resolved_counts_for_role(layout, role)),
                "total_layers": 40 if role == "source" else 64,
            }
            for role in ("source", "target")
        }
        if layout.staged_source_lifecycle:
            expected_analytical_sharding.update(
                {
                    "source_model_prefill": {
                        "mode": "explicit_exact_counts",
                        "resolved_layer_counts": list(
                            layout.source_prefill_layer_counts or ()
                        ),
                        "total_layers": 40,
                    },
                    "source_cache_staging": dict(
                        expected_analytical_sharding["source"]
                    ),
                }
            )
        if analytical.get("layer_sharding") != expected_analytical_sharding:
            raise LatencyReportError(
                f"{label}.analytical_memory layer sharding mismatch"
            )
        for field, counts, maximum_field in (
            (
                "approx_source_per_device_lower_bound_bytes_by_layer_fraction",
                layout.source_layer_counts or (),
                "approx_source_per_device_lower_bound_bytes",
            ),
            (
                "approx_target_cell_per_device_lower_bound_bytes_by_layer_fraction",
                layout.target_layer_counts or (),
                "approx_target_cell_per_device_lower_bound_bytes",
            ),
        ):
            values = _list(analytical.get(field), f"{label}.analytical_memory.{field}")
            if len(values) != len(counts):
                raise LatencyReportError(
                    f"{label}.analytical_memory.{field} has wrong device coverage"
                )
            checked = [
                _integer(
                    value,
                    f"{label}.analytical_memory.{field}[{index}]",
                    minimum=1,
                )
                for index, value in enumerate(values)
            ]
            if _integer(
                analytical.get(maximum_field),
                f"{label}.analytical_memory.{maximum_field}",
                minimum=1,
            ) != max(checked):
                raise LatencyReportError(
                    f"{label}.analytical_memory.{maximum_field} disagrees with per-device bounds"
                )
        if layout.staged_source_lifecycle:
            expected_cpu_bytes = sum(
                SOURCE_CACHE_BF16_BYTES_PER_TOKEN * length for length in lengths
            )
            maximum_length = max(lengths)
            source_cache_bytes = SOURCE_CACHE_BF16_BYTES_PER_TOKEN * maximum_length
            source_inverse_bytes = 40 * 8 * maximum_length * 128 * 4
            timed_source_bounds = [
                int(
                    (source_cache_bytes + source_inverse_bytes)
                    * count
                    / 40
                )
                for count in layout.source_layer_counts or ()
            ]
            source_weight_bytes = _integer(
                _mapping(
                    runtime.get("source_model"),
                    f"{label}.runtime_artifacts.source_model",
                ).get("weight_shard_total_nbytes"),
                f"{label}.runtime_artifacts.source_model.weight_shard_total_nbytes",
                minimum=1,
            )
            untimed_prefill_bounds = [
                int(
                    source_weight_bytes * count / 40
                    + source_cache_bytes * count / 40
                )
                for count in layout.source_prefill_layer_counts or ()
            ]
            staged_expected = {
                "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
                "source_model_live_during_timed_cells": False,
                "all_prefix_lengths_cpu_staging_exact_bytes": expected_cpu_bytes,
                "all_prefix_lengths_cpu_staging_lengths": list(lengths),
                "source_lower_bound_phase": (
                    "timed_mapper_source_cache_plus_inverse_without_source_model"
                ),
                "timed_mapper_source_includes_source_model_weights": False,
                "untimed_source_prefill_includes_source_model_weights": True,
                "source_cache_bf16_bytes": source_cache_bytes,
                "source_inverse_content_keys_fp32_bytes": source_inverse_bytes,
                "approx_source_per_device_lower_bound_bytes_by_layer_fraction": timed_source_bounds,
                "approx_source_per_device_lower_bound_bytes": max(timed_source_bounds),
                "untimed_source_prefill_per_device_lower_bound_bytes_by_layer_fraction": untimed_prefill_bounds,
                "untimed_source_prefill_per_device_lower_bound_bytes": max(
                    untimed_prefill_bounds
                ),
            }
            for field, wanted in staged_expected.items():
                if analytical.get(field) != wanted:
                    raise LatencyReportError(
                        f"{label}.analytical_memory.{field} mismatch"
                    )
    return definition, fingerprint, layout


def _validate_configuration(
    value: Any,
    *,
    label: str,
    fingerprint: str,
    lengths: Sequence[int],
    warmup: int,
    repeats: int,
    layout: DeviceLayout,
) -> Mapping[str, Any]:
    configuration = _mapping(value, label)
    expected = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "request_fingerprint": fingerprint,
        "mode": "production",
        "source_devices": list(layout.source_devices),
        "target_devices": list(layout.target_devices),
        "cuda_visible_devices_environment": layout.cuda_visibility,
        "prefix_lengths": list(lengths),
        "batch_size": 1,
        "warmup": warmup,
        "timed_repeats": repeats,
        "latency_protocol": "paper_main_only",
        "supplemental_diagnostics_enabled": False,
        "profile_warmup": None,
        "profile_timed_repeats": None,
        "mapper_token_chunk": 1024,
        "attention_implementation": "flash_attention_2",
        "seed": 20260813,
        "expected_target_layers": 64,
        "expected_mapped_cache_dtypes": ["torch.bfloat16"],
        "preload_gpu_idle_gate_policy": layout.preload_policy,
    }
    for field, wanted in expected.items():
        if configuration.get(field) != wanted:
            raise LatencyReportError(f"{label}.{field} mismatch")
    if layout.staged_source_lifecycle:
        staged_expected = {
            "stage_source_caches_on_cpu_before_cells": True,
            "source_prefill_devices": list(layout.source_prefill_devices),
            "source_prefill_layer_counts": list(
                layout.source_prefill_layer_counts or ()
            ),
            "staged_runtime_topology": _expected_staged_runtime_topology(layout),
        }
        for field, wanted in staged_expected.items():
            if configuration.get(field) != wanted:
                raise LatencyReportError(f"{label}.{field} mismatch")
    if layout.requires_exact_layer_evidence:
        if configuration.get("source_layer_counts") != list(
            layout.source_layer_counts or ()
        ):
            raise LatencyReportError(f"{label}.source_layer_counts mismatch")
        if configuration.get("target_layer_counts") != list(
            layout.target_layer_counts or ()
        ):
            raise LatencyReportError(f"{label}.target_layer_counts mismatch")
        _validate_layer_sharding_contract(
            configuration.get("layer_sharding"),
            layout=layout,
            label=f"{label}.layer_sharding",
        )
    else:
        for field in ("source_layer_counts", "target_layer_counts"):
            if field in configuration and configuration[field] is not None:
                raise LatencyReportError(
                    f"{label}.{field} is not valid for the historical profile"
                )
        if "layer_sharding" in configuration:
            _validate_layer_sharding_contract(
                configuration["layer_sharding"],
                layout=layout,
                label=f"{label}.layer_sharding",
            )
    protocol = _mapping(configuration.get("local_conventions"), f"{label}.local_conventions")
    protocol_expected = {
        "source_prefill_included": False,
        "target_reprefill_excludes_lm_head": True,
        "paper_main_metrics": ["target_reprefill", "mapper_handoff_end_to_end"],
        "boundary_decode_excludes_lm_head": None,
        "boundary_and_protocol_are_supplemental_not_paper_main": True,
        "mapper_master_dtype": "float32",
        "model_and_cache_dtype": "bfloat16",
        "mapper_execution": "eager",
        "warmup_default": 50,
        "timed_repeats_default": 30,
        "hardware": "8xRTX4090_without_NVLink_or_P2P",
        "paper_hardware": "8xH100_with_NVLink",
        "absolute_latency_directly_comparable_to_paper": False,
        "per_length_cuda_peak_memory_evidence_required": True,
        "internal_metric_cuda_peak_resets_disabled": True,
    }
    for field, wanted in protocol_expected.items():
        if protocol.get(field) != wanted:
            raise LatencyReportError(f"{label}.local_conventions.{field} mismatch")
    if layout.staged_source_lifecycle:
        staged_protocol = {
            "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
            "source_model_live_during_timed_cells": False,
            "source_cpu_staging_and_gpu_restage_in_latency": False,
            "mapper_timer_includes_source_to_target_transfer": True,
            "author_model_residency_and_sharding_published": False,
            "local_staged_sharding_is_4090_capacity_configuration": True,
        }
        for field, wanted in staged_protocol.items():
            if protocol.get(field) != wanted:
                raise LatencyReportError(
                    f"{label}.local_conventions.{field} mismatch"
                )
    return configuration


def _validate_run(
    run_dir: str | Path,
    *,
    lengths: Sequence[int],
    warmup: int,
    repeats: int,
) -> ValidatedRun:
    root = Path(run_dir).resolve()
    if not root.is_dir() or root.is_symlink():
        raise LatencyReportError(f"latency run directory is missing or a symlink: {root}")
    definition_path = root / "run_definition.json"
    definition, fingerprint, layout = _validate_definition(
        _load_json(definition_path),
        label=str(definition_path),
        lengths=lengths,
        warmup=warmup,
        repeats=repeats,
    )
    # The definition is part of the manifest too, but checking its sidecar now
    # makes the request-fingerprint trust chain explicit.
    _validate_sha_sidecar(definition_path)
    configuration = _validate_configuration(
        _load_json(root / "config.json"),
        label=str(root / "config.json"),
        fingerprint=fingerprint,
        lengths=lengths,
        warmup=warmup,
        repeats=repeats,
        layout=layout,
    )
    definition_sharding = _mapping(definition.get("resolved"), "run_definition.resolved").get(
        "layer_sharding"
    )
    configuration_sharding = configuration.get("layer_sharding")
    if definition_sharding != configuration_sharding:
        raise LatencyReportError(
            "config layer_sharding differs from fingerprint-bound run definition"
        )
    status = _mapping(_load_json(root / "status.json"), str(root / "status.json"))
    if status != {"status": "complete", "request_fingerprint": fingerprint}:
        raise LatencyReportError(f"{root / 'status.json'} is not the exact complete marker")

    manifest = _validate_manifest(root, fingerprint=fingerprint, lengths=lengths)
    preload_path = root / "preload_gpu_idle_gate.json"
    preload_gate_record = _validate_preload_gate_record(
        _load_json(preload_path),
        definition=definition,
        label=str(preload_path),
        layout=layout,
    )
    _validate_sha_sidecar(preload_path)
    result_path = root / "result.json"
    result = _mapping(_load_json(result_path), str(result_path))
    _validate_sha_sidecar(result_path)
    result_expected = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "results_manifest": "results.manifest.json",
        "results_manifest_sha256": sha256_file(root / "results.manifest.json"),
        "metrics_sha256": sha256_file(root / "metrics.json"),
        "metrics_csv_sha256": sha256_file(root / "metrics.csv"),
    }
    for field, wanted in result_expected.items():
        if result.get(field) != wanted:
            raise LatencyReportError(f"{result_path}.{field} mismatch")

    hardware_raw = _load_json(root / "hardware.json")
    hardware, capacities = _validate_hardware(hardware_raw, layout=layout)
    topology = _mapping(definition.get("topology"), "run_definition.topology")
    for field in ("devices", "peer_access", "all_selected_pairs_peer_access_disabled"):
        if topology.get(field) != hardware.get(field):
            raise LatencyReportError(f"run_definition.topology.{field} differs from hardware.json")
    attention = _validate_attention(
        _load_json(root / "attention_backend.json"),
        maximum_length=max(lengths),
        staged_source_lifecycle=layout.staged_source_lifecycle,
    )
    attention_sha256 = sha256_file(root / "attention_backend.json")
    runtime = _mapping(definition.get("runtime_artifacts"), "run_definition.runtime_artifacts")
    mapper_setup = _validate_mapper_setup(
        _load_json(root / "mapper_setup.json"), runtime=runtime, layout=layout
    )

    metrics_path = root / "metrics.json"
    metrics = _mapping(_load_json(metrics_path), str(metrics_path))
    metric_expected = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "protocol": configuration["local_conventions"],
        "hardware": hardware,
        "attention_backend": attention,
        "attention_backend_assertion_sha256": attention_sha256,
        "mapper_setup": mapper_setup,
        "preload_gpu_idle_gate": preload_gate_record,
    }
    for field, wanted in metric_expected.items():
        if metrics.get(field) != wanted:
            raise LatencyReportError(f"{metrics_path}.{field} mismatch")
    if layout.staged_source_lifecycle:
        if type(metrics.get("resumed")) is not bool:
            raise LatencyReportError("staged metrics.resumed must be boolean")
        reused = _list(
            metrics.get("reused_checkpoint_lengths"),
            f"{metrics_path}.reused_checkpoint_lengths",
        )
        if reused != list(lengths[: len(reused)]):
            raise LatencyReportError("staged reused checkpoints are not a prefix")
        if reused and metrics.get("resumed") is not True:
            raise LatencyReportError(
                "staged reused checkpoints require resumed=true"
            )
        if metrics.get("resumed") is False and reused:
            raise LatencyReportError(
                "fresh staged metrics cannot claim reused checkpoints"
            )
        if metrics.get("source_cache_staging_policy") != SOURCE_CACHE_STAGING_POLICY:
            raise LatencyReportError("metrics lacks staged source-cache policy")
    elif metrics.get("resumed") is not False or metrics.get(
        "reused_checkpoint_lengths"
    ) != []:
        raise LatencyReportError("formal legacy run unexpectedly resumed checkpoints")
    math_contract = _mapping(metrics.get("math"), f"{metrics_path}.math")
    math_expected = {
        "model_forward_dtype": "bfloat16",
        "mapper_dtype": "float32",
        "allow_tf32_cuda_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "mapper_execution": "eager",
        "torch_compile": False,
        "cuda_graph": False,
    }
    if math_contract != math_expected:
        raise LatencyReportError("metrics math contract is not strict FP32 eager mapper execution")
    telemetry_before = _load_json(root / "telemetry_before.json")
    telemetry_after = _load_json(root / "telemetry_after.json")
    if metrics.get("telemetry_before") != telemetry_before or metrics.get("telemetry_after") != telemetry_after:
        raise LatencyReportError("metrics telemetry snapshots differ from standalone artifacts")

    embedded = _list(metrics.get("length_results"), f"{metrics_path}.length_results")
    if [item.get("prefix_length") if isinstance(item, Mapping) else None for item in embedded] != list(lengths):
        raise LatencyReportError("metrics.length_results is missing, duplicated, extra, or reordered")
    checkpoints: list[Mapping[str, Any]] = []
    checkpoint_sha256: dict[int, str] = {}
    minimum_headroom: dict[int, int] = {}
    for length, embedded_value in zip(lengths, embedded, strict=True):
        path = root / f"length_{length:05d}.json"
        disk_value = _load_json(path)
        if embedded_value != disk_value:
            raise LatencyReportError(f"metrics.length_results differs from checkpoint {path}")
        sidecar = _validate_sha_sidecar(path)
        checkpoint, headroom, _ = _validate_checkpoint(
            disk_value,
            length=length,
            fingerprint=fingerprint,
            attention_sha256=attention_sha256,
            warmup=warmup,
            repeats=repeats,
            capacities=capacities,
            layout=layout,
            root=root,
            label=str(path),
        )
        checkpoints.append(checkpoint)
        checkpoint_sha256[length] = str(sidecar["sha256"])
        minimum_headroom[length] = headroom
    if layout.staged_source_lifecycle:
        _validate_staged_sessions(
            root=root,
            lengths=lengths,
            checkpoints=checkpoints,
            runtime=runtime,
            attention=attention,
            mapper_setup=mapper_setup,
            metrics=metrics,
            fingerprint=fingerprint,
            layout=layout,
        )
    _validate_input_csv(root / "metrics.csv", lengths=lengths, checkpoints=checkpoints)
    return ValidatedRun(
        root=root,
        definition=definition,
        configuration=configuration,
        result=result,
        manifest=manifest,
        hardware=hardware,
        attention=attention,
        mapper_setup=mapper_setup,
        metrics=metrics,
        checkpoints=tuple(checkpoints),
        checkpoint_sha256=checkpoint_sha256,
        minimum_headroom_by_length=minimum_headroom,
        layout=layout,
    )


def _validate_endpoint_receipt(endpoint: ValidatedRun) -> Mapping[str, Any]:
    path = endpoint.root / "endpoint_gate_receipt.json"
    receipt = _mapping(_load_json(path), str(path))
    _validate_sha_sidecar(
        path,
        schema_version=1,
        sidecar_path=endpoint.root / "endpoint_gate_receipt.sha256.json",
    )
    allowed = {
        "schema_version",
        "status",
        "request_fingerprint",
        "project_code_files_digest",
        "runtime_artifacts_digest",
        "run_definition_sha256",
        "results_manifest_sha256",
        "preload_gpu_idle_gate_sha256",
        "cuda_visible_devices",
        "logical_device_order",
        "minimum_required_peak_reserved_headroom_bytes",
        "latency_protocol",
        "lengths",
    }
    if endpoint.layout.requires_exact_layer_evidence:
        allowed.update(
            {
                "device_layout_profile",
                "source_layer_counts",
                "target_layer_counts",
                "model_layer_sharding",
                "actual_mapper_target_layer_counts",
            }
        )
    if endpoint.layout.staged_source_lifecycle:
        allowed.update(
            {
                "source_model_prefill_devices",
                "source_model_prefill_layer_counts",
                "source_cache_staging_policy",
                "runtime_topology",
                "staging_session_bindings",
            }
        )
    if set(receipt) != allowed:
        raise LatencyReportError("endpoint receipt fields are missing or unexpected")
    _integer(receipt.get("schema_version"), f"{path}.schema_version")
    expected = {
        "schema_version": 1,
        "status": "passed",
        "request_fingerprint": endpoint.definition["request_fingerprint"],
        "project_code_files_digest": endpoint.definition["project_code"]["files_digest"],
        "runtime_artifacts_digest": canonical_digest(endpoint.definition["runtime_artifacts"]),
        "run_definition_sha256": sha256_file(endpoint.root / "run_definition.json"),
        "results_manifest_sha256": sha256_file(endpoint.root / "results.manifest.json"),
        "preload_gpu_idle_gate_sha256": sha256_file(
            endpoint.root / "preload_gpu_idle_gate.json"
        ),
        "cuda_visible_devices": endpoint.layout.cuda_visibility,
        "logical_device_order": list(endpoint.layout.selected_devices),
        "minimum_required_peak_reserved_headroom_bytes": ENDPOINT_MINIMUM_HEADROOM_BYTES,
        "latency_protocol": "paper_main_only",
    }
    for field, wanted in expected.items():
        if receipt.get(field) != wanted:
            raise LatencyReportError(f"endpoint receipt {field} mismatch")
    if endpoint.layout.requires_exact_layer_evidence:
        exact_layout_expected = {
            "device_layout_profile": endpoint.layout.profile,
            "source_layer_counts": list(endpoint.layout.source_layer_counts or ()),
            "target_layer_counts": list(endpoint.layout.target_layer_counts or ()),
            "model_layer_sharding": endpoint.mapper_setup["model_layer_sharding"],
            "actual_mapper_target_layer_counts": list(
                _mapper_target_layer_counts(endpoint.mapper_setup, endpoint.layout)
            ),
        }
        for field, wanted in exact_layout_expected.items():
            if canonical_digest(receipt.get(field)) != canonical_digest(wanted):
                raise LatencyReportError(f"endpoint receipt {field} mismatch")
    if endpoint.layout.staged_source_lifecycle:
        session_bindings = {
            str(item["prefix_length"]): {
                "staging_session_id": item["source_cache_lifecycle"][
                    "staging_session_id"
                ],
                "session_evidence_path": item["source_cache_lifecycle"][
                    "session_evidence_path"
                ],
                "session_evidence_sha256": item["source_cache_lifecycle"][
                    "session_evidence_sha256"
                ],
                "preparation_evidence_sha256": item["source_cache_lifecycle"][
                    "preparation_evidence_sha256"
                ],
            }
            for item in endpoint.checkpoints
        }
        staged_expected = {
            "source_model_prefill_devices": list(
                endpoint.layout.source_prefill_devices
            ),
            "source_model_prefill_layer_counts": list(
                endpoint.layout.source_prefill_layer_counts or ()
            ),
            "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
            "runtime_topology": _expected_staged_runtime_topology(endpoint.layout),
            "staging_session_bindings": session_bindings,
        }
        for field, wanted in staged_expected.items():
            if canonical_digest(receipt.get(field)) != canonical_digest(wanted):
                raise LatencyReportError(f"endpoint receipt {field} mismatch")
    lengths = _mapping(receipt.get("lengths"), "endpoint receipt lengths")
    if set(lengths) != {str(length) for length in ENDPOINT_LENGTHS} or len(lengths) != len(
        ENDPOINT_LENGTHS
    ):
        raise LatencyReportError("endpoint receipt must contain exactly 64 and 32768")
    checkpoints = {int(item["prefix_length"]): item for item in endpoint.checkpoints}
    for length in ENDPOINT_LENGTHS:
        item = _mapping(lengths[str(length)], f"endpoint receipt lengths.{length}")
        expected_checkpoint = f"length_{length:05d}.json"
        if item.get("checkpoint") != expected_checkpoint or item.get("checkpoint_sha256") != endpoint.checkpoint_sha256[length]:
            raise LatencyReportError(f"endpoint receipt checkpoint binding failed at {length}")
        peak_devices = _mapping(checkpoints[length]["cuda_peak_memory"]["devices"], f"endpoint checkpoint {length} devices")
        headrooms = {
            device: int(peak_devices[device]["capacity_headroom_at_peak_reserved_bytes"])
            for device in peak_devices
        }
        minimum = min(headrooms.values())
        if minimum < ENDPOINT_MINIMUM_HEADROOM_BYTES:
            raise LatencyReportError(f"endpoint {length} has less than 1 GiB peak-reserved headroom")
        if item.get("per_device_peak_reserved_headroom_bytes") != headrooms or item.get("minimum_peak_reserved_headroom_bytes") != minimum:
            raise LatencyReportError(f"endpoint receipt headroom disagrees at {length}")
        checkpoint_speedup = float(checkpoints[length]["target_reprefill"]["median"]) / float(
            checkpoints[length]["mapper_handoff_end_to_end"]["median"]
        )
        _same_number(
            item.get("median_speedup_target_reprefill_over_mapper"),
            checkpoint_speedup,
            f"endpoint receipt speedup[{length}]",
        )
    return receipt


def _run_provenance(run: ValidatedRun) -> dict[str, Any]:
    return {
        "path": str(run.root),
        "run_id": run.result.get("run_id"),
        "request_fingerprint": run.definition["request_fingerprint"],
        "run_definition_sha256": sha256_file(run.root / "run_definition.json"),
        "results_manifest_sha256": sha256_file(run.root / "results.manifest.json"),
        "metrics_sha256": sha256_file(run.root / "metrics.json"),
    }


def build_latency_report(endpoint_run_dir: str | Path, formal_run_dir: str | Path) -> dict[str, Any]:
    """Validate endpoint/formal runs and build the deterministic report payload."""

    endpoint = _validate_run(
        endpoint_run_dir,
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
    )
    receipt = _validate_endpoint_receipt(endpoint)
    formal = _validate_run(
        formal_run_dir,
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
    )
    if formal.root == endpoint.root:
        raise LatencyReportError("endpoint and formal run directories must be distinct")
    if formal.layout != endpoint.layout:
        raise LatencyReportError("formal CUDA device layout differs from the endpoint gate")
    for field in ("runtime_artifacts", "project_code"):
        if endpoint.definition.get(field) != formal.definition.get(field):
            raise LatencyReportError(f"formal {field} identity differs from the endpoint gate")
    if formal.definition["project_code"]["files_digest"] != receipt["project_code_files_digest"]:
        raise LatencyReportError("formal code identity differs from the frozen endpoint receipt")
    if canonical_digest(formal.definition["runtime_artifacts"]) != receipt["runtime_artifacts_digest"]:
        raise LatencyReportError("formal model/mapper identity differs from the endpoint receipt")
    if endpoint.hardware.get("devices") != formal.hardware.get("devices"):
        raise LatencyReportError("formal GPU identity/capacity differs from endpoint")
    if formal.layout.requires_exact_layer_evidence:
        if canonical_digest(
            formal.mapper_setup.get("model_layer_sharding")
        ) != canonical_digest(receipt.get("model_layer_sharding")):
            raise LatencyReportError(
                "formal runtime model layer split differs from endpoint receipt"
            )
        if list(_mapper_target_layer_counts(formal.mapper_setup, formal.layout)) != receipt.get(
            "actual_mapper_target_layer_counts"
        ):
            raise LatencyReportError(
                "formal mapper target distribution differs from endpoint receipt"
            )

    rows: list[dict[str, Any]] = []
    for checkpoint in formal.checkpoints:
        length = int(checkpoint["prefix_length"])
        target = checkpoint["target_reprefill"]
        mapper = checkpoint["mapper_handoff_end_to_end"]
        peak_devices = checkpoint["cuda_peak_memory"]["devices"]
        paper = PAPER_REFERENCE.get(length)
        row = {
            "prefix_length": length,
            "target_reprefill_mean_ms": float(target["mean"]),
            "target_reprefill_median_ms": float(target["median"]),
            "target_reprefill_p10_ms": float(target["p10"]),
            "target_reprefill_p90_ms": float(target["p90"]),
            "mapper_handoff_mean_ms": float(mapper["mean"]),
            "mapper_handoff_median_ms": float(mapper["median"]),
            "mapper_handoff_p10_ms": float(mapper["p10"]),
            "mapper_handoff_p90_ms": float(mapper["p90"]),
            "median_speedup_target_over_mapper_x": float(
                checkpoint["median_speedup_target_reprefill_over_mapper"]
            ),
            "minimum_peak_reserved_headroom_bytes": formal.minimum_headroom_by_length[length],
            "minimum_peak_reserved_headroom_gib": formal.minimum_headroom_by_length[length]
            / float(1 << 30),
            "per_device_peak_reserved_headroom_bytes": {
                device: int(item["capacity_headroom_at_peak_reserved_bytes"])
                for device, item in peak_devices.items()
            },
            "paper_mapper_handoff_ms": paper["mapper_handoff_ms"] if paper else None,
            "paper_target_reprefill_ms": paper["target_reprefill_ms"] if paper else None,
            "paper_reported_speedup_x": paper["reported_speedup_x"] if paper else None,
            "paper_ratio_from_displayed_ms_x": (
                paper["target_reprefill_ms"] / paper["mapper_handoff_ms"] if paper else None
            ),
            "absolute_ms_directly_comparable_to_paper": False,
        }
        rows.append(row)

    paper_reference = {
        str(length): {
            **values,
            "ratio_from_displayed_ms_x": (
                values["target_reprefill_ms"] / values["mapper_handoff_ms"]
            ),
        }
        for length, values in PAPER_REFERENCE.items()
    }
    completed_utc = formal.metrics.get("completed_utc")
    if not isinstance(completed_utc, str) or not completed_utc:
        raise LatencyReportError("formal metrics lacks completed_utc")
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_14b_to_32b_paper_main_latency_report",
        "generated_from_formal_completion_utc": completed_utc,
        "protocol": {
            "latency_protocol": "paper_main_only",
            "prefix_lengths": list(PAPER_LENGTHS),
            "warmup": 50,
            "timed_repeats": 30,
            "source_prefill_included": False,
            "headline_metrics": ["target_reprefill", "mapper_handoff_end_to_end"],
            "timed_main_metrics_inference_mode_enabled": True,
            "timed_main_metrics_autograd_graph_disabled": True,
            "speedup_definition": "ratio_of_medians_target_reprefill_over_mapper_handoff",
            "speedup_label": "paper-aligned TTFT prefill-component acceleration",
            "speedup_is_full_end_to_end_ttft": False,
            "supplemental_diagnostics_in_formal_cell": False,
            "attention_backend": "flash_attention_2",
            "mapper_dtype": "float32",
            "mapper_execution": "resident_eager_fp32",
            "source_devices": list(formal.layout.source_devices),
            "target_devices": list(formal.layout.target_devices),
            "selected_device_order": list(formal.layout.selected_devices),
            "cuda_visible_devices_environment": formal.layout.cuda_visibility,
            "device_layout_profile": formal.layout.profile,
            "source_layer_counts": (
                list(formal.layout.source_layer_counts)
                if formal.layout.source_layer_counts is not None
                else None
            ),
            "target_layer_counts": (
                list(formal.layout.target_layer_counts)
                if formal.layout.target_layer_counts is not None
                else None
            ),
            "actual_mapper_target_layer_counts": list(
                _mapper_target_layer_counts(formal.mapper_setup, formal.layout)
            ),
        },
        "hardware_comparison": {
            "local_machine": (
                "8x NVIDIA GeForce RTX 4090; "
                f"{len(formal.layout.selected_devices)} selected by the model protocol"
            ),
            "local_interconnect": "PCIe without NVLink/P2P",
            "paper_machine": "8x NVIDIA H100 with NVLink",
            "absolute_latency_directly_comparable": False,
            "relative_speedup_directly_comparable": False,
            "paper_reference_role": "context_only_not_a_hardware_equivalent_baseline",
            "warning": (
                "The paper does not disclose per-model GPU sharding for this pair. The "
                "local audited 5-target/3-source placement uses RTX 4090 "
                "PCIe without P2P, so both absolute milliseconds and the local speedup ratio "
                "are not directly comparable to the paper's H100/NVLink values."
            ),
        },
        "input_runs": {
            "endpoint": _run_provenance(endpoint),
            "formal": _run_provenance(formal),
            "endpoint_gate_receipt_sha256": sha256_file(
                endpoint.root / "endpoint_gate_receipt.json"
            ),
        },
        "endpoint_gate": {
            "status": "passed",
            "lengths": list(ENDPOINT_LENGTHS),
            "warmup": 1,
            "timed_repeats": 2,
            "minimum_required_peak_reserved_headroom_bytes": ENDPOINT_MINIMUM_HEADROOM_BYTES,
            "purpose": "capacity gate only; endpoint pilot latency is excluded from the formal table",
        },
        "paper_reference": paper_reference,
        "paper_reference_scope": (
            "Only lengths 64, 8192, and 32768 are represented. The paper table reports "
            "4x at length 64 while its displayed 61.7/14.0 ms values imply about 4.41x. "
            "These rows are context only because pair-level GPU sharding is undisclosed and "
            "the local hardware/interconnect and placement differ."
        ),
        "rows": rows,
        "minimum_peak_reserved_headroom_bytes_across_formal_lengths": min(
            formal.minimum_headroom_by_length.values()
        ),
    }
    if formal.layout.staged_source_lifecycle:
        report["protocol"].update(
            {
                "source_model_prefill_devices": list(
                    formal.layout.source_prefill_devices
                ),
                "source_model_prefill_layer_counts": list(
                    formal.layout.source_prefill_layer_counts or ()
                ),
                "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
                "runtime_topology": _expected_staged_runtime_topology(
                    formal.layout
                ),
                "source_model_live_during_timed_cells": False,
                "cpu_staging_and_gpu_restage_included_in_mapper_timing": False,
                "mapper_timing_starts_from_preexisting_gpu_source_cache": True,
                "author_model_residency_and_sharding_published": False,
                "local_staged_sharding_is_4090_capacity_configuration": True,
            }
        )
        report["hardware_comparison"]["warning"] = (
            "The paper does not disclose per-model GPU sharding or model-residency "
            "details for this pair. The local staged 6-target/2-source-cache timed "
            "placement, with a separate untimed 3-card source-prefill phase, is an "
            "RTX 4090 PCIe/no-P2P capacity configuration; neither absolute latency "
            "nor its speedup ratio is directly hardware-comparable to H100/NVLink."
        )
    report["report_payload_sha256"] = canonical_digest(report)
    return report


def render_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=OUTPUT_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in report["rows"]:
        writer.writerow({field: row.get(field) for field in OUTPUT_CSV_COLUMNS})
    return buffer.getvalue()


def _display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    protocol = _mapping(report.get("protocol"), "report.protocol")
    source_counts = protocol.get("source_layer_counts")
    target_counts = protocol.get("target_layer_counts")
    mapper_counts = protocol.get("actual_mapper_target_layer_counts")
    lines = [
        "# Qwen3-14B → Qwen3-32B paper-main latency",
        "",
        f"- Device-layout profile: `{protocol.get('device_layout_profile')}`.",
        (
            "- Explicit decoder-layer counts: "
            f"source=`{source_counts}`, target=`{target_counts}`; "
            f"actual resident mapper target distribution=`{mapper_counts}`."
            if source_counts is not None and target_counts is not None
            else "- Decoder-layer counts: historical legacy-even placement (no explicit weighted counts)."
        ),
        "- Formal protocol: source cache already exists; source prefill is excluded.",
        "- Both timed main metrics run under `torch.inference_mode()` with autograd graphs disabled.",
        "- Each formal length uses 50 warmups and 30 timed repeats.",
        (
            "- Headline speedup is the **paper-aligned TTFT prefill-component acceleration**: "
            "target re-prefill median divided by the unsplit mapper cache-handoff median. "
            "It is not full end-to-end TTFT."
        ),
        "- Mapper execution is resident eager FP32; model/cache dtype is BF16; attention is FlashAttention-2.",
        "- Supplemental component, boundary-decode, and protocol-complete diagnostics are not part of the formal cell.",
        "- **The paper does not disclose pair-level GPU sharding. This local capacity-driven RTX 4090/PCIe/no-P2P placement differs from H100/NVLink, so absolute milliseconds and the local speedup ratio are not directly comparable hardware-equivalent paper results; paper rows are context only.**",
        "",
        "| Prefix | Target re-prefill median (ms) | Mapper handoff median (ms) | Paper-aligned TTFT prefill-component acceleration (×; not full end-to-end TTFT) | Min peak-reserved headroom (GiB) | Paper mapper / target (ms) | Paper reported / displayed-ratio speedup (×) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if protocol.get("source_cache_staging_policy") == SOURCE_CACHE_STAGING_POLICY:
        lines[3:3] = [
            (
                "- Untimed source-model prefill placement: "
                f"devices=`{protocol.get('source_model_prefill_devices')}`, "
                f"decoder counts=`{protocol.get('source_model_prefill_layer_counts')}`."
            ),
            (
                "- Timed mapper cells start from a bitwise-verified preexisting GPU "
                "source cache; CPU staging, GPU restaging, and source-model prefill "
                "are outside the timer, and the source model is released before target load."
            ),
            (
                "- The paper does not publish model residency or sharding; this local "
                "three-phase topology is an RTX 4090 capacity configuration."
            ),
        ]
    for row in report["rows"]:
        paper_pair = (
            "—"
            if row["paper_mapper_handoff_ms"] is None
            else f"{row['paper_mapper_handoff_ms']:.1f} / {row['paper_target_reprefill_ms']:.1f}"
        )
        paper_speedup = (
            "—"
            if row["paper_reported_speedup_x"] is None
            else f"{row['paper_reported_speedup_x']:g} / {row['paper_ratio_from_displayed_ms_x']:.3f}"
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["prefix_length"]),
                    _display(row["target_reprefill_median_ms"]),
                    _display(row["mapper_handoff_median_ms"]),
                    _display(row["median_speedup_target_over_mapper_x"]),
                    _display(row["minimum_peak_reserved_headroom_gib"]),
                    paper_pair,
                    paper_speedup,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The endpoint 64/32768 run is a 1-warmup/2-repeat capacity gate only and is not mixed into the formal table.",
            "",
            "Paper references appear only at prefix lengths 64, 8192, and 32768. At 64 tokens, the paper displays 4× while the displayed latency values imply 4.407×.",
            "",
        ]
    )
    return "\n".join(lines)


def _strict_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise LatencyReportError(f"report is not strict finite JSON: {exc}") from exc


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_latency_report_outputs(
    report: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Atomically write deterministic JSON/CSV/Markdown and a SHA manifest.

    Existing identical outputs are accepted for idempotence.  A partial or
    different managed output set is refused instead of being silently mixed
    with the new report.
    """

    output = Path(output_dir).resolve()
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise LatencyReportError(f"output path is not a real directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    primary = {
        "latency_report.json": _strict_json_text(report).encode("utf-8"),
        "latency_table.csv": render_csv(report).encode("utf-8"),
        "latency_table.md": render_markdown(report).encode("utf-8"),
    }
    generated = report.get("generated_from_formal_completion_utc")
    manifest = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "complete",
        "algorithm": "sha256",
        "generated_from_formal_completion_utc": generated,
        "files": {
            name: {
                "sha256": hashlib.sha256(content).hexdigest(),
                "nbytes": len(content),
            }
            for name, content in sorted(primary.items())
        },
    }
    contents = {
        **primary,
        "sha256_manifest.json": _strict_json_text(manifest).encode("utf-8"),
    }
    existing = {name: (output / name).exists() for name in contents}
    if any(existing.values()):
        if not all(existing.values()):
            raise LatencyReportError("refusing a partial pre-existing latency report output set")
        for name, content in contents.items():
            path = output / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise LatencyReportError(f"refusing to overwrite different report output: {path}")
        return manifest
    for name, content in contents.items():
        _atomic_write_bytes(output / name, content)
    return manifest


__all__ = [
    "ENDPOINT_LENGTHS",
    "LatencyReportError",
    "OUTPUT_CSV_COLUMNS",
    "PAPER_LENGTHS",
    "PAPER_REFERENCE",
    "build_latency_report",
    "canonical_digest",
    "render_csv",
    "render_markdown",
    "write_latency_report_outputs",
]
