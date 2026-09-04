from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

import pytest

from kvcache_transfer.io import atomic_write_json, sha256_file
from kvcache_transfer.latency_report import (
    BENCHMARK_LATENCY_SHA256,
    CUDA_PEAK_ALLOCATOR_SCOPE,
    CUDA_PEAK_DISABLED_REASON,
    CUDA_PEAK_MEASUREMENT_SCOPE,
    STAGED_CUDA_PEAK_MEASUREMENT_SCOPE,
    CUDA_DRIVER_FREE_SCOPE,
    CUDA_VISIBILITY,
    ENDPOINT_LENGTHS,
    ENDPOINT_MINIMUM_HEADROOM_BYTES,
    INPUT_CSV_COLUMNS,
    INPUT_SCHEMA_VERSION,
    LATENCY_CODE_FILES_DIGEST,
    LATENCY_LIBRARY_SHA256,
    LatencyReportError,
    MAPPER_FIT_FINGERPRINT,
    MAPPER_MANIFEST_SHA256,
    MAPPER_PARAMETER_BYTES,
    MULTI_GPU_TIMING_BACKEND,
    NESTED_PEAK_CONTRACT,
    PAPER_LENGTHS,
    PAPER_MAIN_PROTOCOL,
    PRELOAD_GPU_IDLE_GATE_POLICY,
    SELECTED_DEVICE_IDS,
    SOURCE_DEVICE_IDS,
    TARGET_DEVICE_IDS,
    TIMED_FAST_PATH_CONTRACT,
    SOURCE_CACHE_STAGING_POLICY,
    build_latency_report,
    canonical_digest,
    render_markdown,
    write_latency_report_outputs,
)
from scripts import summarize_latency as cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LATENCY_CODE_FILES = (
    "scripts/benchmark_latency.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/transfer.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/io.py",
    "src/kvcache_transfer/runlog.py",
)

LEGACY_LAYOUT = {
    "source": SOURCE_DEVICE_IDS,
    "target": TARGET_DEVICE_IDS,
    "selected": SELECTED_DEVICE_IDS,
    "visibility": CUDA_VISIBILITY,
}
BALANCED_LAYOUT = {
    "source": (5, 6, 7),
    "target": (0, 1, 2, 3, 4),
    "selected": (5, 6, 7, 0, 1, 2, 3, 4),
    "visibility": "3,4,5,6,7,0,1,2",
}
V4_LAYOUT = {
    **BALANCED_LAYOUT,
    "profile": "balanced_5target_3source_low_first_v4",
    "source_layer_counts": (14, 13, 13),
    "target_layer_counts": (8, 15, 14, 14, 13),
}
V5_LAYOUT = {
    **BALANCED_LAYOUT,
    "profile": "balanced_5target_3source_embedding_relief_v5",
    "source_layer_counts": (14, 13, 13),
    "target_layer_counts": (4, 15, 15, 15, 15),
}
V6_LAYOUT = {
    **BALANCED_LAYOUT,
    "profile": "balanced_5target_3source_single_first_v6",
    "source_layer_counts": (14, 13, 13),
    "target_layer_counts": (1, 16, 16, 16, 15),
}
V7_LAYOUT = {
    **BALANCED_LAYOUT,
    "profile": "balanced_5target_3source_equal_v7",
    "source_layer_counts": (14, 13, 13),
    "target_layer_counts": (13, 13, 13, 13, 12),
}
V8_LAYOUT = {
    **BALANCED_LAYOUT,
    "profile": "balanced_5target_3source_inference_low_first_v8",
    "source_layer_counts": (14, 13, 13),
    "target_layer_counts": (6, 15, 15, 15, 13),
}
V9_LAYOUT = {
    "source": (6, 7),
    "target": (0, 1, 2, 3, 4, 5),
    "selected": (6, 7, 0, 1, 2, 3, 4, 5),
    "visibility": "2,3,4,5,6,7,0,1",
    "profile": "staged_6target_2cache_v9",
    "source_layer_counts": (20, 20),
    "target_layer_counts": (10, 11, 11, 11, 11, 10),
    "source_prefill_devices": (5, 6, 7),
    "source_prefill_layer_counts": (14, 13, 13),
    "staged": True,
}


def _legacy_counts(total_layers: int, device_count: int):
    counts = [0] * device_count
    for layer in range(total_layers):
        counts[min(layer * device_count // total_layers, device_count - 1)] += 1
    return tuple(counts)


def _role_counts(layout, role):
    total = 40 if role == "source" else 64
    devices = layout[role]
    return tuple(
        layout.get(f"{role}_layer_counts")
        or _legacy_counts(total, len(devices))
    )


def _layer_sharding(layout):
    result = {}
    for role, total in (("source", 40), ("target", 64)):
        devices = layout[role]
        declared = layout.get(f"{role}_layer_counts")
        counts = _role_counts(layout, role)
        result[role] = {
            "mode": "explicit_exact_counts" if declared is not None else "legacy_even_contiguous",
            "devices": list(devices),
            "declared_layer_counts": list(declared) if declared is not None else None,
            "resolved_layer_counts": list(counts),
            "total_layers": total,
        }
    if layout.get("staged"):
        result["source_model_prefill"] = {
            "mode": "explicit_exact_counts",
            "devices": list(layout["source_prefill_devices"]),
            "declared_layer_counts": list(layout["source_prefill_layer_counts"]),
            "resolved_layer_counts": list(layout["source_prefill_layer_counts"]),
            "total_layers": 40,
        }
        result["source_cache_staging"] = dict(result["source"])
    return result


def _runtime_layer_sharding(layout):
    if layout.get("staged"):
        prefill_layout = {
            "source": layout["source_prefill_devices"],
            "target": layout["target"],
            "source_layer_counts": layout["source_prefill_layer_counts"],
            "target_layer_counts": layout["target_layer_counts"],
        }
        prefill = _runtime_layer_sharding(prefill_layout)["source"]
        prefill["role"] = "source_model_prefill"
        target = _runtime_layer_sharding(prefill_layout)["target"]
        counts = layout["source_layer_counts"]
        layer_devices = [
            device
            for device, count in zip(layout["source"], counts, strict=True)
            for _ in range(count)
        ]
        ranges = []
        start = 0
        for device, count in zip(layout["source"], counts, strict=True):
            ranges.append(
                {
                    "device": device,
                    "start_layer": start,
                    "end_layer_exclusive": start + count,
                    "count": count,
                }
            )
            start += count
        staging = {
            "status": "passed",
            "role": "source_cache_staging_without_source_model",
            "mode": "explicit_exact_counts",
            "devices": list(layout["source"]),
            "declared_layer_counts": list(counts),
            "expected_layer_counts": list(counts),
            "actual_layer_counts": list(counts),
            "total_layers": 40,
            "layer_devices": layer_devices,
            "contiguous_ranges": ranges,
            "source_model_parameters_present": False,
            "validated_per_cell_from_actual_cache_tensors": True,
        }
        return {
            "source": prefill,
            "source_model_prefill": prefill,
            "source_cache_staging": staging,
            "target": target,
        }
    result = {}
    for role, total in (("source", 40), ("target", 64)):
        devices = layout[role]
        declared = layout.get(f"{role}_layer_counts")
        counts = _role_counts(layout, role)
        layer_devices = [
            device
            for device, count in zip(devices, counts, strict=True)
            for _ in range(count)
        ]
        ranges = []
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
        result[role] = {
            "status": "passed",
            "role": role,
            "mode": "explicit_exact_counts" if declared is not None else "legacy_even_contiguous",
            "devices": list(devices),
            "declared_layer_counts": list(declared) if declared is not None else None,
            "expected_layer_counts": list(counts),
            "actual_layer_counts": list(counts),
            "total_layers": total,
            "layer_devices": layer_devices,
            "contiguous_ranges": ranges,
            "actual_parameters_inspected_for_every_layer": True,
            "parameter_tensors_inspected": total * 11,
            "auxiliary_modules": {
                name: {
                    "expected_device": device,
                    "actual_device": device,
                    "parameter_tensors_inspected": 1,
                }
                for name, device in (
                    ("model.embed_tokens", devices[0]),
                    ("model.norm", devices[-1]),
                    ("lm_head", devices[-1]),
                )
            },
            "auxiliary_parameter_tensors_inspected": 3,
            "auxiliary_parameters_inspected_for_every_module": True,
        }
    return result


def _preload_policy(layout):
    return {
        **PRELOAD_GPU_IDLE_GATE_POLICY,
        "cuda_visible_devices_environment": layout["visibility"],
        "selected_logical_device_ids": list(layout["selected"]),
    }


def _staged_topology(layout=V9_LAYOUT):
    physical = tuple(int(item) for item in layout["visibility"].split(","))
    return {
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "cuda_visible_devices_environment": layout["visibility"],
        "logical_to_physical": {
            str(index): device for index, device in enumerate(physical)
        },
        "source_model_prefill": {
            "logical_devices": list(layout["source_prefill_devices"]),
            "physical_devices": [physical[index] for index in layout["source_prefill_devices"]],
            "layer_counts": list(layout["source_prefill_layer_counts"]),
        },
        "source_cache_staging": {
            "logical_devices": list(layout["source"]),
            "physical_devices": [physical[index] for index in layout["source"]],
            "layer_counts": list(layout["source_layer_counts"]),
        },
        "target_timed": {
            "logical_devices": list(layout["target"]),
            "physical_devices": [physical[index] for index in layout["target"]],
            "layer_counts": list(layout["target_layer_counts"]),
        },
        "source_prefill_target_logical_overlap": [5],
        "source_prefill_target_physical_overlap": [7],
        "source_cache_target_logical_overlap": [],
        "source_cache_target_physical_overlap": [],
        "logical_5_unique_sequential_reuse_physical_device": 7,
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary(base: float, *, length: int, warmup: int, repeats: int, metric: str):
    samples = [base + index * 0.01 for index in range(repeats)]
    value = {
        "unit": "ms",
        "count": repeats,
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p10": _percentile(samples, 0.10),
        "p50": _percentile(samples, 0.50),
        "p90": _percentile(samples, 0.90),
        "std_population": statistics.pstdev(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
        "warmup": warmup,
        "timed_repeats": repeats,
        "timing_backends": [MULTI_GPU_TIMING_BACKEND],
        "synchronize_each_iteration": True,
        "cuda_peak_memory_collected": False,
        "cuda_peak_memory_disabled_reason": CUDA_PEAK_DISABLED_REASON,
        "inference_mode_enabled": True,
        "autograd_graph_disabled": True,
    }
    if metric == "target_reprefill":
        value.update(
            {
                "measurement_scope": "target_transformer_reprefill",
                "batch_size": 1,
                "input_length": length,
                "lm_head_included": False,
                "kv_cache_output": True,
            }
        )
    else:
        value.update(
            {
                "measurement_scope": "source_cache_to_mapped_target_cache",
                "source_prefill_included": False,
                "includes_cache_build": True,
                "includes_inverse_and_target_rope": True,
                "includes_source_cache_device_handoff": True,
                "mapper_weights_resident_on_target_devices": True,
            }
        )
    return value


def _sha_text(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _model_identity(label: str):
    files = {
        name: {"sha256": _sha_text(f"{label}-{name}"), "nbytes": len(name) + 10}
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
        )
    }
    shards = [
        {
            "path": f"model-{index:05d}-of-00002.safetensors",
            "nbytes": 1000 + index,
            "sha256": _sha_text(f"{label}-shard-{index}"),
        }
        for index in (1, 2)
    ]
    layers = 40 if label == "Qwen3-14B" else 64
    hidden = 5120 if label == "Qwen3-14B" else 5120
    return {
        "path": f"/models/{label}",
        "content_hashed_files": files,
        "bound_config_fields": {
            "model_type": "qwen3",
            "vocab_size": 151936,
            "num_hidden_layers": layers,
            "num_attention_heads": 40 if label == "Qwen3-14B" else 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": hidden,
        },
        "weight_shards": shards,
        "weight_shard_count": len(shards),
        "weight_shard_total_nbytes": sum(item["nbytes"] for item in shards),
        "weight_shards_digest": canonical_digest(shards),
        "weight_shard_identity_scope": "fixture",
        "weight_shard_contents_hashed": True,
    }


def _runtime_identity():
    return {
        "source_model": _model_identity("Qwen3-14B"),
        "target_model": _model_identity("Qwen3-32B"),
        "mapper": {
            "root": "/artifacts/mapper",
            "manifest": "/artifacts/mapper/manifest.json",
            "manifest_sha256": MAPPER_MANIFEST_SHA256,
            "checksum": "/artifacts/mapper/manifest.sha256.json",
            "checksum_sha256": _sha_text("mapper-checksum"),
            "fit_fingerprint": MAPPER_FIT_FINGERPRINT,
            "layer_count": 64,
            "selection": {"fixture": True},
        },
    }


def _project_code():
    files = {
        relative: {
            "sha256": sha256_file(PROJECT_ROOT / relative),
            "nbytes": (PROJECT_ROOT / relative).stat().st_size,
        }
        for relative in LATENCY_CODE_FILES
    }
    assert files["scripts/benchmark_latency.py"]["sha256"] == BENCHMARK_LATENCY_SHA256
    assert files["src/kvcache_transfer/latency.py"]["sha256"] == LATENCY_LIBRARY_SHA256
    assert canonical_digest(files) == LATENCY_CODE_FILES_DIGEST
    return {"files": files, "files_digest": canonical_digest(files)}


def _hardware(layout=LEGACY_LAYOUT):
    devices = [
        {
            "index": device,
            "name": "NVIDIA GeForce RTX 4090",
            "total_memory_bytes": 24 << 30,
            "capability": [8, 9],
        }
        for device in layout["selected"]
    ]
    return {
        "torch_version": "fixture",
        "torch_cuda_version": "fixture",
        "cuda_available": True,
        "cuda_visible_devices_environment": layout["visibility"],
        "visible_cuda_devices": len(layout["visibility"].split(",")),
        "selected_devices": list(layout["selected"]),
        "local_topology_convention": {
            "accelerators": "8x NVIDIA GeForce RTX 4090",
            "nvlink": False,
            "cuda_peer_access_expected": False,
            "warning": (
                "The paper used 8xH100 with NVLink; local absolute latency and PCIe "
                "handoff cost are not directly comparable."
            ),
        },
        "devices": devices,
        "peer_access": {},
        "all_selected_pairs_peer_access_disabled": True,
        "nvidia_smi_topology": {"returncode": 0},
        "nvidia_smi_p2p_read": {"returncode": 0},
    }


def _attention(layout=LEGACY_LAYOUT):
    def role(name: str, layers: int):
        return {
            "status": "passed",
            "role": name,
            "expected": "flash_attention_2",
            "config_attn_implementation": "flash_attention_2",
            "config_attn_implementation_internal": "flash_attention_2",
            "decoder_layer_count": layers,
            "attention_module_classes": ["Qwen3Attention"],
            "dispatcher_verified_for_every_layer": True,
            "backend_callable_module": "transformers.integrations.flash_attention",
            "backend_callable_name": "flash_attention_forward",
            "flash_attention_2_available": True,
            "max_position_embeddings": 40960,
            "maximum_requested_length": 32768,
        }

    result = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "passed",
        "source": role("source", 40),
        "target": role("target", 64),
        "persisted_before_any_length": True,
        "covers_32768_before_execution": True,
    }
    if layout.get("staged"):
        result.update(
            {
                "source_model_released_before_target_model_load": True,
                "source_attention_asserted_during_untimed_cache_preparation": True,
            }
        )
    return result


def _mapper_setup(layout=LEGACY_LAYOUT):
    target_devices = [
        device
        for device, count in zip(
            layout["target"], _role_counts(layout, "target"), strict=True
        )
        for _ in range(count)
    ]
    result = {
        "scope": "one_time_setup_excluded_from_per_request_latency",
        "host_load_and_hash": {"unit": "ms", "count": 64},
        "weight_host_to_device": {"unit": "ms", "count": 64},
        "per_layer": [
            {
                "target_layer": layer,
                "device": f"cuda:{target_devices[layer]}",
                "host_load_and_hash_ms": 1.0 + layer / 100,
                "weight_host_to_device_ms": 2.0 + layer / 100,
            }
            for layer in range(64)
        ],
        "resident_parameter_bytes": MAPPER_PARAMETER_BYTES,
        "resident_dtype": "float32",
        "manifest_sha256": MAPPER_MANIFEST_SHA256,
        "fit_fingerprint": MAPPER_FIT_FINGERPRINT,
        "execution": "resident_eager_fp32",
        "target_layer_count": 64,
        **(
            {"model_layer_sharding": _runtime_layer_sharding(layout)}
            if layout.get("source_layer_counts") is not None
            else {}
        ),
    }
    if layout.get("staged"):
        result.update(
            {
                "source_model_live_during_timed_cells": False,
                "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
                "runtime_topology": _staged_topology(layout),
            }
        )
    return result


def _sidecar(path: Path, *, schema: int = INPUT_SCHEMA_VERSION, special: Path | None = None):
    atomic_write_json(
        special or path.with_name(f"{path.name}.sha256.json"),
        {
            "schema_version": schema,
            "artifact": path.name,
            "sha256": sha256_file(path),
            "nbytes": path.stat().st_size,
        },
    )


def _peak(length: int, fingerprint: str, layout=LEGACY_LAYOUT):
    capacity = 24 << 30
    before_allocated = 10 << 20
    before_reserved = 20 << 20
    peak_allocated = (3 << 30) + length
    peak_reserved = (4 << 30) + length
    after_allocated = 2 << 30
    after_reserved = 3 << 30
    devices = {}
    for device in layout["selected"]:
        driver_free_before = capacity - (1 << 30)
        driver_free_after = capacity - (2 << 30)
        current_before_allocated = before_allocated
        current_before_reserved = before_reserved
        current_after_allocated = after_allocated
        current_after_reserved = after_reserved
        if layout.get("staged") and device in layout["source"]:
            current_before_allocated = 163_840 * length // 2
            current_before_reserved = current_before_allocated
            current_after_allocated = 0
            current_after_reserved = 0
        devices[str(device)] = {
            "capacity_bytes": capacity,
            "driver_free_bytes_before": driver_free_before,
            "driver_total_bytes_before": capacity,
            "driver_used_bytes_before": capacity - driver_free_before,
            "allocated_bytes_before": current_before_allocated,
            "reserved_bytes_before": current_before_reserved,
            "peak_allocated_bytes_immediately_after_reset": current_before_allocated,
            "peak_reserved_bytes_immediately_after_reset": current_before_reserved,
            "driver_free_bytes_after": driver_free_after,
            "driver_total_bytes_after": capacity,
            "driver_used_bytes_after": capacity - driver_free_after,
            "allocated_bytes_after": current_after_allocated,
            "reserved_bytes_after": current_after_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_growth_to_peak_bytes": peak_allocated - current_before_allocated,
            "reserved_growth_to_peak_bytes": peak_reserved - current_before_reserved,
            "capacity_headroom_at_peak_reserved_bytes": capacity - peak_reserved,
            "peak_allocated_fraction_of_capacity": peak_allocated / capacity,
            "peak_reserved_fraction_of_capacity": peak_reserved / capacity,
        }
    return {
        "schema_version": 2,
        "status": "captured",
        "prefix_length": length,
        "request_fingerprint": fingerprint,
        "selected_device_ids": list(layout["selected"]),
        "measurement_scope": (
            STAGED_CUDA_PEAK_MEASUREMENT_SCOPE
            if layout.get("staged")
            else CUDA_PEAK_MEASUREMENT_SCOPE
        ),
        "allocator_peak_scope": CUDA_PEAK_ALLOCATOR_SCOPE,
        "driver_free_scope": CUDA_DRIVER_FREE_SCOPE,
        "synchronized_before_reset": True,
        "peak_counters_reset_after_baseline": True,
        "synchronized_after_cell": True,
        "devices": devices,
    }


def _staged_structure(length: int, devices):
    tensor_bytes = 1 * 8 * length * 128 * 2
    return [
        {
            "layer": layer,
            "device": str(device),
            "key_shape": [1, 8, length, 128],
            "value_shape": [1, 8, length, 128],
            "key_dtype": "torch.bfloat16",
            "value_dtype": "torch.bfloat16",
            "key_nbytes": tensor_bytes,
            "value_nbytes": tensor_bytes,
        }
        for layer, device in enumerate(devices)
    ]


def _allocator_snapshot(devices, *, allocated: int):
    capacity = 24 << 30
    return {
        str(device): {
            "allocated_bytes": allocated,
            "reserved_bytes": allocated,
            "driver_free_bytes": capacity - (1 << 30),
            "driver_total_bytes": capacity,
            "driver_used_bytes": 1 << 30,
        }
        for device in devices
    }


def _staged_session_payload(
    *, fingerprint: str, lengths, runtime, attention, layout=V9_LAYOUT
):
    session_id = "a" * 32
    prefill_layer_devices = [
        f"cuda:{device}"
        for device, count in zip(
            layout["source_prefill_devices"],
            layout["source_prefill_layer_counts"],
            strict=True,
        )
        for _ in range(count)
    ]
    preparations = {}
    for length in lengths:
        raw_hash = _sha_text(f"staged-cache-{length}")
        prefix = {
            "shape": [1, length],
            "dtype": "torch.int64",
            "device": "cpu",
            "nbytes": 8 * length,
            "sha256": _sha_text(f"prefix-{length}"),
        }
        cache_bytes = 163_840 * length
        preparations[str(length)] = {
            "schema_version": INPUT_SCHEMA_VERSION,
            "status": "passed",
            "policy": SOURCE_CACHE_STAGING_POLICY,
            "prefix_length": length,
            "source_prefill_timed": False,
            "cpu_staging_timed": False,
            "copy_semantics": "detach_to_cpu_blocking_exact_clone",
            "full_raw_bit_sha256_computed": True,
            "source_gpu_raw_bit_sha256": raw_hash,
            "cpu_staged_raw_bit_sha256": raw_hash,
            "raw_bit_sha256": raw_hash,
            "gpu_to_cpu_raw_bit_hash_matches": True,
            "deterministic_prefix_identity": prefix,
            "source_prefill_layer_devices": prefill_layer_devices,
            "source_prefill_cache_structure": _staged_structure(
                length, prefill_layer_devices
            ),
            "cpu_cache_structure": _staged_structure(length, ["cpu"] * 40),
            "cpu_cache_total_bytes": cache_bytes,
            "expected_cache_total_bytes": cache_bytes,
        }
    source_weight = runtime["source_model"]["weight_shard_total_nbytes"]
    target_weight = runtime["target_model"]["weight_shard_total_nbytes"]
    all_cache_bytes = sum(163_840 * length for length in lengths)
    host_gate = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "passed",
        "scope": "before_source_model_load",
        "mem_available_bytes": 500 << 30,
        "required_available_bytes": source_weight
        + target_weight
        + all_cache_bytes
        + (16 << 30),
        "source_weight_shard_bytes": source_weight,
        "target_weight_shard_bytes": target_weight,
        "all_requested_source_caches_cpu_bytes": all_cache_bytes,
        "staged_prefix_lengths": list(lengths),
        "conservative_operating_overhead_bytes": 16 << 30,
    }
    before = _allocator_snapshot(layout["source_prefill_devices"], allocated=1 << 30)
    after = _allocator_snapshot(layout["source_prefill_devices"], allocated=0)
    release = {
        "status": "passed",
        "source_model_live": False,
        "source_model_weakref_is_none": True,
        "all_source_gpu_cache_references_deleted": True,
        "gc_completed": True,
        "cuda_synchronized_and_empty_cache_completed": True,
        "cuda_workspace_and_allocator_release": {
            "status": "passed",
            "selected_device_ids": list(layout["source_prefill_devices"]),
            "cublas_workspaces_clear_api": "torch._C._cuda_clearCublasWorkspaces",
            "cublas_workspaces_clear_called": True,
            "cublas_workspaces_clear_succeeded": True,
            "gc_completed_after_workspace_clear": True,
            "empty_cache_completed_for_every_selected_device": True,
            "synchronized_before_and_after": True,
            "source_model_deleted_and_gc_completed_before_clear": True,
            "call_order": [
                "source_model_deleted_and_gc_collected",
                "synchronize_before_cublas_workspace_clear",
                "torch._C._cuda_clearCublasWorkspaces",
                "gc_collect_after_cublas_workspace_clear",
                "torch.cuda.empty_cache_each_selected_device",
                "synchronize_after_empty_cache",
            ],
        },
        "source_prefill_logical_devices": list(layout["source_prefill_devices"]),
        "allocator_driver_before_release": before,
        "allocator_driver_after_release_before_target_load": after,
        "all_prefill_devices_allocated_bytes_zero": True,
        "all_prefill_devices_reserved_bytes_zero": True,
        "target_model_loaded_at_proof_time": False,
        "source_prefill_target_overlap_exact": [5],
        "logical_5_reused_physical_device": 7,
        "runtime_topology_sha256": canonical_digest(_staged_topology(layout)),
    }
    return {
        "schema_version": 1,
        "status": "source_released_before_target_load",
        "created_utc": "2026-08-18T00:00:00+00:00",
        "request_fingerprint": fingerprint,
        "staging_session_id": session_id,
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "runtime_topology": _staged_topology(layout),
        "runtime_topology_sha256": canonical_digest(_staged_topology(layout)),
        "host_memory_staging_gate": host_gate,
        "prepared_missing_lengths": list(lengths),
        "source_vocabulary_size": 151936,
        "source_attention": attention["source"],
        "source_model_prefill_sharding": _runtime_layer_sharding(layout)[
            "source_model_prefill"
        ],
        "preparations": preparations,
        "cpu_staging_total_bytes": all_cache_bytes,
        "source_model_release": release,
        "target_model_loaded_when_session_evidence_written": False,
        "source_model_live_when_session_evidence_written": False,
    }


def _apply_staged_checkpoint(
    checkpoint, *, session, session_name: str, session_sha256: str, layout=V9_LAYOUT
):
    length = checkpoint["prefix_length"]
    preparation = session["preparations"][str(length)]
    prefix = preparation["deterministic_prefix_identity"]
    cache_bytes = preparation["cpu_cache_total_bytes"]
    cache_layer_devices = [
        f"cuda:{device}"
        for device, count in zip(
            layout["source"], layout["source_layer_counts"], strict=True
        )
        for _ in range(count)
    ]
    release_snapshot = _allocator_snapshot(layout["source"], allocated=0)
    lifecycle = {
        "status": "complete",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "staging_session_id": session["staging_session_id"],
        "session_evidence_path": session_name,
        "session_evidence_sha256": session_sha256,
        "preparation_evidence_sha256": canonical_digest(preparation),
        "source_model_live_during_timed_cells": False,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "outer_peak_reset_after_restage_validation": True,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "mapper_timer_started": True,
        "deterministic_prefix_identity": prefix,
        "source_gpu_raw_bit_sha256": preparation["source_gpu_raw_bit_sha256"],
        "cpu_staged_raw_bit_sha256": preparation["cpu_staged_raw_bit_sha256"],
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "cache_total_bytes": cache_bytes,
        "restage": {
            "status": "passed",
            "policy": SOURCE_CACHE_STAGING_POLICY,
            "prefix_length": length,
            "untimed": True,
            "completed_before_outer_peak_reset": True,
            "source_model_live": False,
            "source_cache_gpu_resident_before_mapper_timer": True,
            "copy_semantics": "cpu_to_cuda_blocking_no_dtype_cast",
            "raw_bit_sha256": preparation["raw_bit_sha256"],
            "raw_bit_hash_matches_cpu_staging": True,
            "layer_devices": cache_layer_devices,
            "layer_counts": list(layout["source_layer_counts"]),
            "cache_structure": _staged_structure(length, cache_layer_devices),
            "cache_total_bytes": cache_bytes,
            "cuda_synchronized_and_empty_cache_before_outer_peak_reset": True,
        },
        "restaged_cache_release": {
            "status": "passed",
            "restaged_cache_weakref_is_none": True,
            "released_before_outer_peak_finish": True,
            "cuda_synchronized_and_empty_cache_completed": True,
            "source_cache_staging_devices": list(layout["source"]),
            "allocator_driver_after_release": release_snapshot,
        },
    }
    checkpoint.update(
        {
            "source_model_live_during_timed_cell": False,
            "source_cache_origin": "preexisting_before_run_one_length",
            "deterministic_prefix_identity": prefix,
            "source_cache_lifecycle": lifecycle,
        }
    )
    return checkpoint


def _checkpoint(
    length: int,
    fingerprint: str,
    attention_sha: str,
    warmup: int,
    repeats: int,
    layout=LEGACY_LAYOUT,
):
    target = _summary(length / 10 + 10, length=length, warmup=warmup, repeats=repeats, metric="target_reprefill")
    mapper = _summary(length / 100 + 2, length=length, warmup=warmup, repeats=repeats, metric="mapper_handoff_end_to_end")
    return {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "attention_backend_assertion_sha256": attention_sha,
        "prefix_length": length,
        "batch_size": 1,
        "source_prefill_timed": False,
        "source_cache_length": length,
        "latency_protocol": dict(PAPER_MAIN_PROTOCOL),
        "strict_transfer_preflight": {
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
            "wall_ms_diagnostic_not_latency": 1.0,
        },
        "timed_transfer_fast_path": dict(TIMED_FAST_PATH_CONTRACT),
        "nested_cuda_peak_memory_contract": dict(NESTED_PEAK_CONTRACT),
        "target_reprefill": target,
        "mapper_handoff_end_to_end": mapper,
        "median_speedup_target_reprefill_over_mapper": target["median"] / mapper["median"],
        "cuda_peak_memory": _peak(length, fingerprint, layout),
    }


def _protocol(layout=LEGACY_LAYOUT):
    result = {
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
        "synthetic_exact_length_token_ids": True,
        "strict_transfer_preflight_untimed": True,
        "timed_transfer_diagnostic_reductions": False,
        "timed_transfer_shape_dtype_architecture_validation": True,
        "checkpoint_resume_requires_exact_request_fingerprint": True,
        "per_length_cuda_peak_memory_evidence_required": True,
        "cuda_peak_counters_reset_after_synchronized_baseline": True,
        "internal_metric_cuda_peak_resets_disabled": True,
        "driver_free_memory_is_before_after_snapshot_not_historical_peak": True,
    }
    if layout.get("staged"):
        result.update(
            {
                "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
                "source_model_live_during_timed_cells": False,
                "source_cpu_staging_and_gpu_restage_in_latency": False,
                "mapper_timer_includes_source_to_target_transfer": True,
                "author_model_residency_and_sharding_published": False,
                "local_staged_sharding_is_4090_capacity_configuration": True,
            }
        )
    return result


def _write_metrics_csv(path: Path, checkpoints):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for checkpoint in checkpoints:
            for metric in ("target_reprefill", "mapper_handoff_end_to_end"):
                summary = checkpoint[metric]
                writer.writerow(
                    {
                        "prefix_length": checkpoint["prefix_length"],
                        "metric": metric,
                        "count": summary["count"],
                        "mean_ms": summary["mean"],
                        "median_ms": summary["median"],
                        "p10_ms": summary["p10"],
                        "p50_ms": summary["p50"],
                        "p90_ms": summary["p90"],
                        "std_population_ms": summary["std_population"],
                        "min_ms": summary["min"],
                        "max_ms": summary["max"],
                        "timing_backends": ";".join(summary["timing_backends"]),
                        "source_prefill_included": False,
                    }
                )


def _preload_gate_fixture(layout=LEGACY_LAYOUT):
    visible = tuple(int(item) for item in layout["visibility"].split(","))
    physical = [visible[logical] for logical in layout["selected"]]
    bindings = [
        {
            "logical_index": logical,
            "physical_index": physical_index,
            "gpu_uuid": f"GPU-fixture-{physical_index}",
        }
        for logical, physical_index in zip(layout["selected"], physical, strict=True)
    ]
    memory = [{**binding, "memory_used_mib": 4} for binding in bindings]
    samples = [
        {
            "sequence": sequence,
            "memory_before": memory,
            "selected_compute_apps_before": [],
            "memory_after": memory,
            "selected_compute_apps_after": [],
        }
        for sequence in range(1, 13)
    ]
    return {
        "schema_version": 1,
        "status": "passed",
        "policy": _preload_policy(layout),
        "selected_gpu_bindings": bindings,
        "started_utc": "2026-08-18T00:00:00+00:00",
        "completed_utc": "2026-08-18T00:01:00+00:00",
        "attempt_count": 12,
        "rejected_attempt_count": 0,
        "final_consecutive_sandwich_samples": samples,
        "all_expensive_identity_hashes_completed_before_gate": True,
        "completed_before_cuda_model_load": True,
    }


def _make_run(
    root: Path,
    *,
    lengths,
    warmup: int,
    repeats: int,
    shared,
    layout=LEGACY_LAYOUT,
):
    root.mkdir()
    hardware, runtime, project = shared
    preload_evidence = _preload_gate_fixture(layout)
    definition = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "kind": "qwen3_kv_cache_transfer_latency",
        "cli": {
            "mode": "production",
            "source_model": "/models/Qwen3-14B",
            "target_model": "/models/Qwen3-32B",
            "mapper": "/artifacts/mapper",
            "source_devices": list(layout["source"]),
            "target_devices": list(layout["target"]),
            "tiny_device": "cpu",
            "lengths": list(lengths),
            "warmup": warmup,
            "repeats": repeats,
            "profile_warmup": None,
            "profile_repeats": None,
            "supplemental_diagnostics": False,
            "mapper_token_chunk": 1024,
            "attention": "flash_attention_2",
            "seed": 20260813,
            "preload_idle_memory_limit_mib": 64,
            "preload_idle_samples": 12,
            "preload_idle_sample_interval_seconds": 5.0,
            "preload_idle_timeout_seconds": 86400.0,
            "output_dir": str(root),
            "no_ledger": False,
            **(
                {
                    "source_layer_counts": list(layout["source_layer_counts"]),
                    "target_layer_counts": list(layout["target_layer_counts"]),
                }
                if layout.get("source_layer_counts") is not None
                else {}
            ),
            **(
                {
                    "stage_source_caches_on_cpu_before_cells": True,
                    "source_prefill_devices": list(layout["source_prefill_devices"]),
                    "source_prefill_layer_counts": list(
                        layout["source_prefill_layer_counts"]
                    ),
                }
                if layout.get("staged")
                else {}
            ),
        },
        "resolved": {
            "lengths": list(lengths),
            "latency_protocol": "paper_main_only",
            "profile_warmup": None,
            "profile_repeats": None,
            **(
                {"layer_sharding": _layer_sharding(layout)}
                if layout.get("source_layer_counts") is not None
                else {}
            ),
        },
        "runtime_artifacts": runtime,
        "project_code": project,
        "backend": {
            "requested_attention": "flash_attention_2",
            "flash_attention_2_available": True,
        },
        "analytical_memory": {
            "paper_main_cell_only": True,
            "supplemental_diagnostics_enabled": False,
            "resident_mapper_fp32_bytes": MAPPER_PARAMETER_BYTES,
            "boundary_branch_strategy": "not_run",
            "boundary_parent_plus_decode_output_cache_bytes": 0,
            "target_cache_copies_in_analytic_cell_bound": 1,
            **(
                {
                    "layer_sharding": {
                        role: {
                            "mode": "explicit_exact_counts",
                            "resolved_layer_counts": list(_role_counts(layout, role)),
                            "total_layers": 40 if role == "source" else 64,
                        }
                        for role in ("source", "target")
                    },
                    "approx_source_per_device_lower_bound_bytes_by_layer_fraction": [
                        count * 1000 for count in _role_counts(layout, "source")
                    ],
                    "approx_target_cell_per_device_lower_bound_bytes_by_layer_fraction": [
                        count * 2000 for count in _role_counts(layout, "target")
                    ],
                    "approx_source_per_device_lower_bound_bytes": max(
                        count * 1000 for count in _role_counts(layout, "source")
                    ),
                    "approx_target_cell_per_device_lower_bound_bytes": max(
                        count * 2000 for count in _role_counts(layout, "target")
                    ),
                }
                if layout.get("source_layer_counts") is not None
                else {}
            ),
        },
        "topology": {
            "cuda_visible_devices_environment": layout["visibility"],
            "selected_devices": list(layout["selected"]),
            "devices": hardware["devices"],
            "peer_access": hardware["peer_access"],
            "all_selected_pairs_peer_access_disabled": True,
            "nvidia_smi_topology": hardware["nvidia_smi_topology"],
            "nvidia_smi_p2p_read": hardware["nvidia_smi_p2p_read"],
            **(
                {"staged_source_lifecycle": _staged_topology(layout)}
                if layout.get("staged")
                else {}
            ),
        },
        "preload_gpu_idle_gate": preload_evidence,
    }
    if layout.get("staged"):
        maximum_length = max(lengths)
        source_cache_bytes = 163_840 * maximum_length
        inverse_bytes = 163_840 * maximum_length
        source_weight = runtime["source_model"]["weight_shard_total_nbytes"]
        timed_bounds = [
            int((source_cache_bytes + inverse_bytes) * count / 40)
            for count in layout["source_layer_counts"]
        ]
        untimed_bounds = [
            int(source_weight * count / 40 + source_cache_bytes * count / 40)
            for count in layout["source_prefill_layer_counts"]
        ]
        definition["analytical_memory"].update(
            {
                "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
                "source_model_live_during_timed_cells": False,
                "all_prefix_lengths_cpu_staging_exact_bytes": sum(
                    163_840 * length for length in lengths
                ),
                "all_prefix_lengths_cpu_staging_lengths": list(lengths),
                "source_lower_bound_phase": "timed_mapper_source_cache_plus_inverse_without_source_model",
                "timed_mapper_source_includes_source_model_weights": False,
                "untimed_source_prefill_includes_source_model_weights": True,
                "source_cache_bf16_bytes": source_cache_bytes,
                "source_inverse_content_keys_fp32_bytes": inverse_bytes,
                "approx_source_per_device_lower_bound_bytes_by_layer_fraction": timed_bounds,
                "approx_source_per_device_lower_bound_bytes": max(timed_bounds),
                "untimed_source_prefill_per_device_lower_bound_bytes_by_layer_fraction": untimed_bounds,
                "untimed_source_prefill_per_device_lower_bound_bytes": max(
                    untimed_bounds
                ),
            }
        )
        definition["analytical_memory"]["layer_sharding"].update(
            {
                "source_model_prefill": {
                    "mode": "explicit_exact_counts",
                    "resolved_layer_counts": list(
                        layout["source_prefill_layer_counts"]
                    ),
                    "total_layers": 40,
                },
                "source_cache_staging": dict(
                    definition["analytical_memory"]["layer_sharding"]["source"]
                ),
            }
        )
    fingerprint = canonical_digest(definition)
    definition["request_fingerprint"] = fingerprint
    atomic_write_json(root / "run_definition.json", definition)
    _sidecar(root / "run_definition.json")
    configuration = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "request_fingerprint": fingerprint,
        "mode": "production",
        "source_devices": list(layout["source"]),
        "target_devices": list(layout["target"]),
        "cuda_visible_devices_environment": layout["visibility"],
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
        "preload_gpu_idle_gate_policy": _preload_policy(layout),
        "local_conventions": _protocol(layout),
        **(
            {
                "source_layer_counts": list(layout["source_layer_counts"]),
                "target_layer_counts": list(layout["target_layer_counts"]),
                "layer_sharding": _layer_sharding(layout),
            }
            if layout.get("source_layer_counts") is not None
            else {}
        ),
        **(
            {
                "stage_source_caches_on_cpu_before_cells": True,
                "source_prefill_devices": list(layout["source_prefill_devices"]),
                "source_prefill_layer_counts": list(
                    layout["source_prefill_layer_counts"]
                ),
                "staged_runtime_topology": _staged_topology(layout),
            }
            if layout.get("staged")
            else {}
        ),
    }
    atomic_write_json(root / "config.json", configuration)
    atomic_write_json(root / "status.json", {"status": "complete", "request_fingerprint": fingerprint})
    atomic_write_json(root / "hardware.json", hardware)
    attention = _attention(layout)
    atomic_write_json(root / "attention_backend.json", attention)
    mapper_setup = _mapper_setup(layout)
    atomic_write_json(root / "mapper_setup.json", mapper_setup)
    preload_record = {
        "schema_version": 1,
        "request_fingerprint": fingerprint,
        "run_definition_gate_evidence_sha256": canonical_digest(preload_evidence),
        "active_preload_gpu_idle_gate": preload_evidence,
        "immediate_preload_gpu_idle_sandwich": {
            "status": "passed",
            "scope": "immediately_before_cuda_model_load",
            "sandwich_interval_seconds": 1.0,
            "memory_before": preload_evidence["final_consecutive_sandwich_samples"][-1]["memory_after"],
            "selected_compute_apps_before": [],
            "memory_after": preload_evidence["final_consecutive_sandwich_samples"][-1]["memory_after"],
            "selected_compute_apps_after": [],
            "completed_utc": "2026-08-18T00:01:01+00:00",
        },
    }
    atomic_write_json(root / "preload_gpu_idle_gate.json", preload_record)
    before, after = {"fixture": "before"}, {"fixture": "after"}
    atomic_write_json(root / "telemetry_before.json", before)
    atomic_write_json(root / "telemetry_after.json", after)
    attention_sha = sha256_file(root / "attention_backend.json")
    staged_session = None
    staged_session_name = None
    staged_session_sha = None
    if layout.get("staged"):
        staged_session = _staged_session_payload(
            fingerprint=fingerprint,
            lengths=lengths,
            runtime=runtime,
            attention=attention,
            layout=layout,
        )
        staged_session_name = (
            "source_cache_staging_session_"
            f"{staged_session['staging_session_id']}.json"
        )
        staged_session_path = root / staged_session_name
        atomic_write_json(staged_session_path, staged_session)
        _sidecar(staged_session_path)
        staged_session_sha = sha256_file(staged_session_path)
    checkpoints = [
        _checkpoint(length, fingerprint, attention_sha, warmup, repeats, layout)
        for length in lengths
    ]
    if layout.get("staged"):
        assert staged_session is not None
        assert staged_session_name is not None
        assert staged_session_sha is not None
        checkpoints = [
            _apply_staged_checkpoint(
                checkpoint,
                session=staged_session,
                session_name=staged_session_name,
                session_sha256=staged_session_sha,
                layout=layout,
            )
            for checkpoint in checkpoints
        ]
    for checkpoint in checkpoints:
        path = root / f"length_{checkpoint['prefix_length']:05d}.json"
        atomic_write_json(path, checkpoint)
        _sidecar(path)
    metrics = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "created_utc": "2026-08-18T00:00:00+00:00",
        "completed_utc": "2026-08-18T00:10:00+00:00",
        "run_id": root.name,
        "request_fingerprint": fingerprint,
        "resumed": False,
        "protocol": configuration["local_conventions"],
        "math": {
            "model_forward_dtype": "bfloat16",
            "mapper_dtype": "float32",
            "allow_tf32_cuda_matmul": False,
            "allow_tf32_cudnn": False,
            "float32_matmul_precision": "highest",
            "mapper_execution": "eager",
            "torch_compile": False,
            "cuda_graph": False,
        },
        "hardware": hardware,
        "attention_backend": attention,
        "attention_backend_assertion_sha256": attention_sha,
        "mapper_setup": mapper_setup,
        "preload_gpu_idle_gate": preload_record,
        "cuda_memory_baseline_after_runtime_load": {},
        "source_cache_staging_policy": (
            SOURCE_CACHE_STAGING_POLICY if layout.get("staged") else None
        ),
        "active_staging_session": (
            {
                "staging_session_id": staged_session["staging_session_id"],
                "session_evidence_path": staged_session_name,
                "session_evidence_sha256": staged_session_sha,
                "prepared_missing_lengths": list(lengths),
                "host_memory_staging_gate": staged_session[
                    "host_memory_staging_gate"
                ],
            }
            if layout.get("staged")
            else None
        ),
        "reused_checkpoint_lengths": [],
        "length_results": checkpoints,
        "telemetry_before": before,
        "telemetry_after": after,
    }
    atomic_write_json(root / "metrics.json", metrics)
    _write_metrics_csv(root / "metrics.csv", checkpoints)
    for name in (
        "hardware.json",
        "preload_gpu_idle_gate.json",
        "attention_backend.json",
        "mapper_setup.json",
        "telemetry_before.json",
        "telemetry_after.json",
        "metrics.json",
        "metrics.csv",
    ):
        _sidecar(root / name)
    _rebuild_manifest_and_result(root, lengths=lengths, fingerprint=fingerprint)
    return root


def _rebuild_manifest_and_result(root: Path, *, lengths=None, fingerprint=None):
    definition = json.loads((root / "run_definition.json").read_text())
    fingerprint = fingerprint or definition["request_fingerprint"]
    lengths = tuple(lengths or definition["resolved"]["lengths"])
    session_names = []
    for length in lengths:
        checkpoint = json.loads((root / f"length_{length:05d}.json").read_text())
        lifecycle = checkpoint.get("source_cache_lifecycle")
        if isinstance(lifecycle, dict):
            name = lifecycle["session_evidence_path"]
            if name not in session_names:
                session_names.append(name)
    names = [
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
    for name in names:
        _sidecar(root / name)
    artifacts = [
        {"path": name, "sha256": sha256_file(root / name), "nbytes": (root / name).stat().st_size}
        for name in names
    ]
    manifest = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "lengths": list(lengths),
        "artifacts": artifacts,
        "artifacts_digest": canonical_digest(artifacts),
    }
    atomic_write_json(root / "results.manifest.json", manifest)
    _sidecar(root / "results.manifest.json")
    result = {
        "schema_version": INPUT_SCHEMA_VERSION,
        "status": "complete",
        "run_id": root.name,
        "request_fingerprint": fingerprint,
        "results_manifest": "results.manifest.json",
        "results_manifest_sha256": sha256_file(root / "results.manifest.json"),
        "metrics_sha256": sha256_file(root / "metrics.json"),
        "metrics_csv_sha256": sha256_file(root / "metrics.csv"),
    }
    atomic_write_json(root / "result.json", result)
    _sidecar(root / "result.json")


def _resign_staged_session(root: Path, mutate) -> None:
    definition = json.loads((root / "run_definition.json").read_text())
    lengths = tuple(definition["resolved"]["lengths"])
    first_checkpoint = json.loads(
        (root / f"length_{lengths[0]:05d}.json").read_text()
    )
    session_name = first_checkpoint["source_cache_lifecycle"][
        "session_evidence_path"
    ]
    session_path = root / session_name
    session = json.loads(session_path.read_text())
    mutate(session)
    atomic_write_json(session_path, session)
    _sidecar(session_path)
    session_sha = sha256_file(session_path)
    checkpoints = []
    for length in lengths:
        path = root / f"length_{length:05d}.json"
        checkpoint = json.loads(path.read_text())
        lifecycle = checkpoint.get("source_cache_lifecycle")
        if lifecycle and lifecycle.get("session_evidence_path") == session_name:
            lifecycle["session_evidence_sha256"] = session_sha
            preparation = session.get("preparations", {}).get(str(length))
            if preparation is not None:
                lifecycle["preparation_evidence_sha256"] = canonical_digest(
                    preparation
                )
            atomic_write_json(path, checkpoint)
            _sidecar(path)
        checkpoints.append(checkpoint)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["length_results"] = checkpoints
    active = metrics.get("active_staging_session")
    if isinstance(active, dict) and active.get("session_evidence_path") == session_name:
        active["session_evidence_sha256"] = session_sha
        active["prepared_missing_lengths"] = session.get(
            "prepared_missing_lengths"
        )
        active["host_memory_staging_gate"] = session.get(
            "host_memory_staging_gate"
        )
    atomic_write_json(metrics_path, metrics)
    _write_metrics_csv(root / "metrics.csv", checkpoints)
    _rebuild_manifest_and_result(root, lengths=lengths)


def _resign_checkpoint(root: Path, length: int, mutate) -> None:
    definition = json.loads((root / "run_definition.json").read_text())
    lengths = tuple(definition["resolved"]["lengths"])
    path = root / f"length_{length:05d}.json"
    checkpoint = json.loads(path.read_text())
    mutate(checkpoint)
    atomic_write_json(path, checkpoint)
    _sidecar(path)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["length_results"] = [
        json.loads((root / f"length_{item:05d}.json").read_text())
        for item in lengths
    ]
    atomic_write_json(metrics_path, metrics)
    _write_metrics_csv(root / "metrics.csv", metrics["length_results"])
    _rebuild_manifest_and_result(root, lengths=lengths)


def _convert_to_partial_staged_resume(root: Path, *, split_index: int = 3) -> None:
    definition = json.loads((root / "run_definition.json").read_text())
    lengths = tuple(definition["resolved"]["lengths"])
    first = json.loads((root / f"length_{lengths[0]:05d}.json").read_text())
    first_name = first["source_cache_lifecycle"]["session_evidence_path"]
    first_session = json.loads((root / first_name).read_text())
    second_session = json.loads(json.dumps(first_session))
    second_session["staging_session_id"] = "b" * 32
    missing = list(lengths[split_index:])
    second_session["prepared_missing_lengths"] = missing
    second_session["preparations"] = {
        str(length): second_session["preparations"][str(length)]
        for length in missing
    }
    second_session["cpu_staging_total_bytes"] = sum(
        163_840 * length for length in missing
    )
    second_name = (
        "source_cache_staging_session_"
        f"{second_session['staging_session_id']}.json"
    )
    second_path = root / second_name
    atomic_write_json(second_path, second_session)
    _sidecar(second_path)
    second_sha = sha256_file(second_path)
    checkpoints = []
    for index, length in enumerate(lengths):
        path = root / f"length_{length:05d}.json"
        checkpoint = json.loads(path.read_text())
        if index >= split_index:
            _apply_staged_checkpoint(
                checkpoint,
                session=second_session,
                session_name=second_name,
                session_sha256=second_sha,
                layout=V9_LAYOUT,
            )
            atomic_write_json(path, checkpoint)
            _sidecar(path)
        checkpoints.append(checkpoint)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["resumed"] = True
    metrics["reused_checkpoint_lengths"] = list(lengths[:split_index])
    metrics["length_results"] = checkpoints
    metrics["active_staging_session"] = {
        "staging_session_id": second_session["staging_session_id"],
        "session_evidence_path": second_name,
        "session_evidence_sha256": second_sha,
        "prepared_missing_lengths": missing,
        "host_memory_staging_gate": second_session["host_memory_staging_gate"],
    }
    atomic_write_json(metrics_path, metrics)
    _write_metrics_csv(root / "metrics.csv", checkpoints)
    _rebuild_manifest_and_result(root, lengths=lengths)


def _endpoint_receipt(root: Path, layout=LEGACY_LAYOUT):
    definition = json.loads((root / "run_definition.json").read_text())
    lengths = {}
    for length in ENDPOINT_LENGTHS:
        path = root / f"length_{length:05d}.json"
        checkpoint = json.loads(path.read_text())
        headrooms = {
            device: item["capacity_headroom_at_peak_reserved_bytes"]
            for device, item in checkpoint["cuda_peak_memory"]["devices"].items()
        }
        lengths[str(length)] = {
            "checkpoint": path.name,
            "checkpoint_sha256": sha256_file(path),
            "minimum_peak_reserved_headroom_bytes": min(headrooms.values()),
            "per_device_peak_reserved_headroom_bytes": headrooms,
            "median_speedup_target_reprefill_over_mapper": checkpoint[
                "median_speedup_target_reprefill_over_mapper"
            ],
        }
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "request_fingerprint": definition["request_fingerprint"],
        "project_code_files_digest": definition["project_code"]["files_digest"],
        "runtime_artifacts_digest": canonical_digest(definition["runtime_artifacts"]),
        "run_definition_sha256": sha256_file(root / "run_definition.json"),
        "results_manifest_sha256": sha256_file(root / "results.manifest.json"),
        "preload_gpu_idle_gate_sha256": sha256_file(
            root / "preload_gpu_idle_gate.json"
        ),
        "cuda_visible_devices": layout["visibility"],
        "logical_device_order": list(layout["selected"]),
        "minimum_required_peak_reserved_headroom_bytes": ENDPOINT_MINIMUM_HEADROOM_BYTES,
        "latency_protocol": "paper_main_only",
        "lengths": lengths,
        **(
            {
                "device_layout_profile": layout["profile"],
                "source_layer_counts": list(layout["source_layer_counts"]),
                "target_layer_counts": list(layout["target_layer_counts"]),
                "model_layer_sharding": _runtime_layer_sharding(layout),
                "actual_mapper_target_layer_counts": list(
                    _role_counts(layout, "target")
                ),
            }
            if layout.get("source_layer_counts") is not None
            else {}
        ),
        **(
            {
                "source_model_prefill_devices": list(
                    layout["source_prefill_devices"]
                ),
                "source_model_prefill_layer_counts": list(
                    layout["source_prefill_layer_counts"]
                ),
                "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
                "runtime_topology": _staged_topology(layout),
                "staging_session_bindings": {
                    str(length): {
                        "staging_session_id": json.loads(
                            (root / f"length_{length:05d}.json").read_text()
                        )["source_cache_lifecycle"]["staging_session_id"],
                        "session_evidence_path": json.loads(
                            (root / f"length_{length:05d}.json").read_text()
                        )["source_cache_lifecycle"]["session_evidence_path"],
                        "session_evidence_sha256": json.loads(
                            (root / f"length_{length:05d}.json").read_text()
                        )["source_cache_lifecycle"]["session_evidence_sha256"],
                        "preparation_evidence_sha256": json.loads(
                            (root / f"length_{length:05d}.json").read_text()
                        )["source_cache_lifecycle"]["preparation_evidence_sha256"],
                    }
                    for length in ENDPOINT_LENGTHS
                },
            }
            if layout.get("staged")
            else {}
        ),
    }
    path = root / "endpoint_gate_receipt.json"
    atomic_write_json(path, receipt)
    _sidecar(path, schema=1, special=root / "endpoint_gate_receipt.sha256.json")


@pytest.fixture
def latency_runs(tmp_path):
    shared = (_hardware(), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
    )
    _endpoint_receipt(endpoint)
    formal = _make_run(
        tmp_path / "formal",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
    )
    return endpoint, formal


def _make_layout_runs(tmp_path: Path, layout):
    shared = (_hardware(layout), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=layout,
    )
    _endpoint_receipt(endpoint, layout)
    formal = _make_run(
        tmp_path / "formal",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=layout,
    )
    return endpoint, formal


def _launcher_layout_environment(endpoint: Path, formal: Path, layout):
    result = {
        **os.environ,
        "LATENCY_ENDPOINT_ROOT": str(endpoint),
        "LATENCY_FORMAL_ROOT": str(formal),
        "LATENCY_CUDA_VISIBILITY": layout["visibility"],
        "LATENCY_SOURCE_DEVICES": ",".join(str(item) for item in layout["source"]),
        "LATENCY_TARGET_DEVICES": ",".join(str(item) for item in layout["target"]),
        "LATENCY_SOURCE_LAYER_COUNTS": ",".join(
            str(item) for item in layout["source_layer_counts"]
        ),
        "LATENCY_TARGET_LAYER_COUNTS": ",".join(
            str(item) for item in layout["target_layer_counts"]
        ),
        "LATENCY_DEVICE_LAYOUT_PROFILE": layout["profile"],
    }
    if layout.get("staged"):
        result.update(
            {
                "LATENCY_STAGE_SOURCE_CACHES": "1",
                "LATENCY_SOURCE_PREFILL_DEVICES": ",".join(
                    str(item) for item in layout["source_prefill_devices"]
                ),
                "LATENCY_SOURCE_PREFILL_LAYER_COUNTS": ",".join(
                    str(item) for item in layout["source_prefill_layer_counts"]
                ),
            }
        )
    return result


def _resign_definition_with_layout(root: Path, *, source, target, visibility):
    path = root / "run_definition.json"
    definition = json.loads(path.read_text())
    definition.pop("request_fingerprint")
    definition["cli"]["source_devices"] = list(source)
    definition["cli"]["target_devices"] = list(target)
    definition["topology"]["cuda_visible_devices_environment"] = visibility
    definition["topology"]["selected_devices"] = [*source, *target]
    definition["request_fingerprint"] = canonical_digest(definition)
    atomic_write_json(path, definition)
    _sidecar(path)


def test_build_and_write_paper_main_latency_report(latency_runs, tmp_path):
    endpoint, formal = latency_runs
    report = build_latency_report(endpoint, formal)
    assert report["status"] == "complete"
    assert report["protocol"]["latency_protocol"] == "paper_main_only"
    assert report["protocol"]["device_layout_profile"] == "legacy_4target_2source"
    assert report["protocol"]["warmup"] == 50
    assert report["protocol"]["timed_repeats"] == 30
    assert (
        report["protocol"]["speedup_label"]
        == "paper-aligned TTFT prefill-component acceleration"
    )
    assert report["protocol"]["speedup_is_full_end_to_end_ttft"] is False
    assert report["protocol"]["timed_main_metrics_inference_mode_enabled"] is True
    assert report["protocol"]["timed_main_metrics_autograd_graph_disabled"] is True
    assert [row["prefix_length"] for row in report["rows"]] == list(PAPER_LENGTHS)
    references = {
        row["prefix_length"] for row in report["rows"] if row["paper_mapper_handoff_ms"] is not None
    }
    assert references == {64, 8192, 32768}
    row64 = report["rows"][0]
    assert row64["paper_mapper_handoff_ms"] == 14.0
    assert row64["paper_target_reprefill_ms"] == 61.7
    assert row64["paper_reported_speedup_x"] == 4.0
    assert row64["paper_ratio_from_displayed_ms_x"] == pytest.approx(61.7 / 14.0)
    assert row64["median_speedup_target_over_mapper_x"] == pytest.approx(
        row64["target_reprefill_median_ms"] / row64["mapper_handoff_median_ms"]
    )

    output = tmp_path / "report"
    manifest = write_latency_report_outputs(report, output)
    assert set(manifest["files"]) == {
        "latency_report.json",
        "latency_table.csv",
        "latency_table.md",
    }
    for name, identity in manifest["files"].items():
        assert sha256_file(output / name) == identity["sha256"]
        assert (output / name).stat().st_size == identity["nbytes"]
    markdown = (output / "latency_table.md").read_text()
    assert "not directly comparable" in markdown
    assert "paper-aligned TTFT prefill-component acceleration" in markdown
    assert "not full end-to-end TTFT" in markdown
    assert "Supplemental" in markdown
    assert "4×" in markdown
    assert len((output / "latency_table.csv").read_text().splitlines()) == 11
    # Identical inputs are deterministic and idempotent.
    assert write_latency_report_outputs(report, output) == manifest


def test_balanced_5target_3source_layout_is_fully_bound(tmp_path):
    endpoint, formal = _make_layout_runs(tmp_path, BALANCED_LAYOUT)
    report = build_latency_report(endpoint, formal)
    protocol = report["protocol"]
    assert protocol["device_layout_profile"] == "balanced_5target_3source_v3"
    assert protocol["source_devices"] == [5, 6, 7]
    assert protocol["target_devices"] == [0, 1, 2, 3, 4]
    assert protocol["selected_device_order"] == [5, 6, 7, 0, 1, 2, 3, 4]
    assert protocol["cuda_visible_devices_environment"] == "3,4,5,6,7,0,1,2"
    assert report["hardware_comparison"]["local_machine"].endswith(
        "8 selected by the model protocol"
    )


def test_weighted_v4_layer_sharding_is_fully_bound(tmp_path):
    endpoint, formal = _make_layout_runs(tmp_path, V4_LAYOUT)
    report = build_latency_report(endpoint, formal)
    protocol = report["protocol"]
    assert (
        protocol["device_layout_profile"]
        == "balanced_5target_3source_low_first_v4"
    )
    assert protocol["source_layer_counts"] == [14, 13, 13]
    assert protocol["target_layer_counts"] == [8, 15, 14, 14, 13]
    assert protocol["actual_mapper_target_layer_counts"] == [8, 15, 14, 14, 13]
    comparison = report["hardware_comparison"]
    assert comparison["absolute_latency_directly_comparable"] is False
    assert comparison["relative_speedup_directly_comparable"] is False
    assert comparison["paper_reference_role"].startswith("context_only")
    markdown = render_markdown(report)
    assert "balanced_5target_3source_low_first_v4" in markdown
    assert "source=`[14, 13, 13]`" in markdown
    assert "target=`[8, 15, 14, 14, 13]`" in markdown
    assert "absolute milliseconds and the local speedup ratio are not directly comparable" in markdown
    assert "context only" in markdown


def test_embedding_relief_v5_layer_sharding_is_fully_bound(tmp_path):
    endpoint, formal = _make_layout_runs(tmp_path, V5_LAYOUT)
    report = build_latency_report(endpoint, formal)
    protocol = report["protocol"]
    assert (
        protocol["device_layout_profile"]
        == "balanced_5target_3source_embedding_relief_v5"
    )
    assert protocol["source_layer_counts"] == [14, 13, 13]
    assert protocol["target_layer_counts"] == [4, 15, 15, 15, 15]
    assert protocol["actual_mapper_target_layer_counts"] == [4, 15, 15, 15, 15]
    markdown = render_markdown(report)
    assert "balanced_5target_3source_embedding_relief_v5" in markdown
    assert "source=`[14, 13, 13]`" in markdown
    assert "target=`[4, 15, 15, 15, 15]`" in markdown


def test_single_first_v6_layer_sharding_is_fully_bound(tmp_path):
    endpoint, formal = _make_layout_runs(tmp_path, V6_LAYOUT)
    report = build_latency_report(endpoint, formal)
    protocol = report["protocol"]
    assert (
        protocol["device_layout_profile"]
        == "balanced_5target_3source_single_first_v6"
    )
    assert protocol["source_layer_counts"] == [14, 13, 13]
    assert protocol["target_layer_counts"] == [1, 16, 16, 16, 15]
    assert protocol["actual_mapper_target_layer_counts"] == [1, 16, 16, 16, 15]
    markdown = render_markdown(report)
    assert "balanced_5target_3source_single_first_v6" in markdown
    assert "source=`[14, 13, 13]`" in markdown
    assert "target=`[1, 16, 16, 16, 15]`" in markdown


def test_equal_v7_layer_sharding_is_fully_bound(tmp_path):
    endpoint, formal = _make_layout_runs(tmp_path, V7_LAYOUT)
    report = build_latency_report(endpoint, formal)
    protocol = report["protocol"]
    assert protocol["device_layout_profile"] == "balanced_5target_3source_equal_v7"
    assert protocol["source_layer_counts"] == [14, 13, 13]
    assert protocol["target_layer_counts"] == [13, 13, 13, 13, 12]
    assert protocol["actual_mapper_target_layer_counts"] == [13, 13, 13, 13, 12]
    markdown = render_markdown(report)
    assert "balanced_5target_3source_equal_v7" in markdown
    assert "source=`[14, 13, 13]`" in markdown
    assert "target=`[13, 13, 13, 13, 12]`" in markdown


def test_inference_low_first_v8_layer_sharding_and_launcher_chain_are_fully_bound(
    tmp_path,
):
    endpoint, formal = _make_layout_runs(tmp_path, V8_LAYOUT)
    report = build_latency_report(endpoint, formal)
    protocol = report["protocol"]
    assert (
        protocol["device_layout_profile"]
        == "balanced_5target_3source_inference_low_first_v8"
    )
    assert protocol["source_layer_counts"] == [14, 13, 13]
    assert protocol["target_layer_counts"] == [6, 15, 15, 15, 13]
    assert protocol["actual_mapper_target_layer_counts"] == [6, 15, 15, 15, 13]
    markdown = render_markdown(report)
    assert "balanced_5target_3source_inference_low_first_v8" in markdown
    assert "source=`[14, 13, 13]`" in markdown
    assert "target=`[6, 15, 15, 15, 13]`" in markdown

    environment = _launcher_layout_environment(endpoint, formal, V8_LAYOUT)
    for verifier in (
        "verify_endpoint_receipt",
        "verify_formal_identity_against_endpoint",
    ):
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f"source scripts/launch_latency_after_idle.sh; {verifier}",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


def test_inference_low_first_v8_profile_tamper_fails_closed(tmp_path):
    endpoint, formal = _make_layout_runs(tmp_path, V8_LAYOUT)
    path = endpoint / "endpoint_gate_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["device_layout_profile"] = "balanced_5target_3source_low_first_v4"
    atomic_write_json(path, receipt)
    _sidecar(path, schema=1, special=endpoint / "endpoint_gate_receipt.sha256.json")
    with pytest.raises(LatencyReportError, match="device_layout_profile mismatch"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_launcher_formal_identity_verifier_accepts_exact_fixture(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/launch_latency_after_idle.sh; verify_formal_identity_against_endpoint",
        ],
        cwd=PROJECT_ROOT,
        env=_launcher_layout_environment(endpoint, formal, layout),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("layout", (V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v5", "v6", "v7"))
def test_recent_exact_launcher_receipt_verifier_rechecks_runtime_totals(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    receipt_path = endpoint / "endpoint_gate_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["model_layer_sharding"]["target"]["parameter_tensors_inspected"] = 703
    atomic_write_json(receipt_path, receipt)
    _sidecar(
        receipt_path,
        schema=1,
        special=endpoint / "endpoint_gate_receipt.sha256.json",
    )
    setup_path = endpoint / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    setup["model_layer_sharding"]["target"]["parameter_tensors_inspected"] = 703
    atomic_write_json(setup_path, setup)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/launch_latency_after_idle.sh; verify_endpoint_receipt",
        ],
        cwd=PROJECT_ROOT,
        env=_launcher_layout_environment(endpoint, formal, layout),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "inspection total drifted" in completed.stderr


@pytest.mark.parametrize("layout", (V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v5", "v6", "v7"))
def test_recent_exact_launcher_receipt_verifier_accepts_exact_fixture(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/launch_latency_after_idle.sh; verify_endpoint_receipt",
        ],
        cwd=PROJECT_ROOT,
        env=_launcher_layout_environment(endpoint, formal, layout),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("status", "drifted field status"),
        ("threshold", "minimum_required_peak_reserved_headroom_bytes"),
        ("minimum_headroom", "headroom drifted"),
        ("peak_headroom_arithmetic", "peak headroom arithmetic drifted"),
        ("checkpoint_name", "name drifted"),
        ("speedup", "speedup drifted"),
    ),
)
@pytest.mark.parametrize("layout", (V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v5", "v6", "v7"))
def test_recent_exact_launcher_receipt_semantic_tamper_fails_closed(
    tmp_path, tamper, message, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    receipt_path = endpoint / "endpoint_gate_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if tamper == "status":
        receipt["status"] = "failed"
    elif tamper == "threshold":
        receipt["minimum_required_peak_reserved_headroom_bytes"] = 0
    elif tamper == "minimum_headroom":
        receipt["lengths"]["32768"][
            "minimum_peak_reserved_headroom_bytes"
        ] += 1
    elif tamper == "peak_headroom_arithmetic":
        checkpoint_path = endpoint / "length_32768.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["cuda_peak_memory"]["devices"]["5"]["capacity_bytes"] += 1
        atomic_write_json(checkpoint_path, checkpoint)
        receipt["lengths"]["32768"]["checkpoint_sha256"] = sha256_file(
            checkpoint_path
        )
    elif tamper == "checkpoint_name":
        receipt["lengths"]["32768"]["checkpoint"] = "length_00064.json"
    elif tamper == "speedup":
        receipt["lengths"]["32768"][
            "median_speedup_target_reprefill_over_mapper"
        ] += 0.5
    else:
        raise AssertionError(tamper)
    atomic_write_json(receipt_path, receipt)
    _sidecar(
        receipt_path,
        schema=1,
        special=endpoint / "endpoint_gate_receipt.sha256.json",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/launch_latency_after_idle.sh; verify_endpoint_receipt",
        ],
        cwd=PROJECT_ROOT,
        env=_launcher_layout_environment(endpoint, formal, layout),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("strict_preflight", "strict preflight drifted"),
        ("checkpoint_speedup", "checkpoint 32768 speedup drifted"),
        ("inference_mode", "target_reprefill inference-mode evidence drifted"),
        ("autograd", "mapper_handoff_end_to_end autograd evidence drifted"),
    ),
)
@pytest.mark.parametrize("layout", (V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v5", "v6", "v7"))
def test_recent_exact_launcher_resigned_checkpoint_semantics_fail_closed(
    tmp_path, tamper, message, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    checkpoint_path = endpoint / "length_32768.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    if tamper == "strict_preflight":
        checkpoint["strict_transfer_preflight"]["status"] = "failed"
    elif tamper == "checkpoint_speedup":
        checkpoint["median_speedup_target_reprefill_over_mapper"] += 0.5
    elif tamper == "inference_mode":
        checkpoint["target_reprefill"]["inference_mode_enabled"] = False
    elif tamper == "autograd":
        checkpoint["mapper_handoff_end_to_end"]["autograd_graph_disabled"] = False
    else:
        raise AssertionError(tamper)
    atomic_write_json(checkpoint_path, checkpoint)
    metrics_path = endpoint / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["length_results"][1] = checkpoint
    atomic_write_json(metrics_path, metrics)
    _write_metrics_csv(endpoint / "metrics.csv", metrics["length_results"])
    _rebuild_manifest_and_result(endpoint)
    receipt_path = endpoint / "endpoint_gate_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["results_manifest_sha256"] = sha256_file(
        endpoint / "results.manifest.json"
    )
    receipt["lengths"]["32768"]["checkpoint_sha256"] = sha256_file(
        checkpoint_path
    )
    atomic_write_json(receipt_path, receipt)
    _sidecar(
        receipt_path,
        schema=1,
        special=endpoint / "endpoint_gate_receipt.sha256.json",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/launch_latency_after_idle.sh; verify_endpoint_receipt",
        ],
        cwd=PROJECT_ROOT,
        env=_launcher_layout_environment(endpoint, formal, layout),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert message in completed.stderr


def test_shared_launcher_has_single_endpoint_policy_and_mapper_increment():
    launcher = (PROJECT_ROOT / "scripts/launch_latency_after_idle.sh").read_text(
        encoding="utf-8"
    )
    assert launcher.count("\n  verify_endpoint_policy\n") == 1
    assert launcher.count("mapper_counts[device] += 1") == 1
    assert launcher.count(
        "4,15,15,15,15|balanced_5target_3source_embedding_relief_v5"
    ) == 1
    assert launcher.count(
        "1,16,16,16,15|balanced_5target_3source_single_first_v6"
    ) == 1
    assert launcher.count(
        "13,13,13,13,12|balanced_5target_3source_equal_v7"
    ) == 1
    formal_tail = launcher.split(
        'grep -q \'"status": "reused_complete"\' "${FORMAL_VERIFY_LOG}"', 1
    )[1]
    ordered_postformal_gate = (
        'run_logged "${ENDPOINT_POSTFORMAL_VERIFY_LOG}" "${ENDPOINT_ARGS[@]}"',
        'grep -q \'"status": "reused_complete"\' "${ENDPOINT_POSTFORMAL_VERIFY_LOG}"',
        "verify_endpoint_receipt",
        "verify_formal_identity_against_endpoint",
        'write_state "formal_passed"',
    )
    positions = [formal_tail.index(token) for token in ordered_postformal_gate]
    assert positions == sorted(positions)


@pytest.mark.parametrize("layout", (V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v5", "v6", "v7"))
def test_recent_exact_launcher_report_verifier_rebuilds_outputs(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    report_root = tmp_path / "report"
    assert cli.main(
        [str(endpoint), str(formal), "--output-dir", str(report_root)]
    ) == 0
    report_path = report_root / "latency_report.json"
    report = json.loads(report_path.read_text())
    report["rows"][0]["target_reprefill_median_ms"] += 12345.0
    atomic_write_json(report_path, report)
    manifest_path = report_root / "sha256_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["latency_report.json"] = {
        "sha256": sha256_file(report_path),
        "nbytes": report_path.stat().st_size,
    }
    atomic_write_json(manifest_path, manifest)
    environment = _launcher_layout_environment(endpoint, formal, layout)
    environment.update(
        {
            "LATENCY_REPORT_ROOT": str(report_root),
            "LATENCY_REPORT_LOG": str(tmp_path / "report.log"),
        }
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/launch_latency_after_idle.sh; verify_report_outputs_against_inputs",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "refusing to overwrite different report output" in completed.stdout


def test_v5_wrapper_preserves_frozen_old_shared_launcher_sha():
    launcher_path = PROJECT_ROOT / "scripts/launch_latency_after_idle.sh"
    wrapper_path = (
        PROJECT_ROOT / "scripts/launch_latency_embedding_relief_5plus3_after_idle.sh"
    )
    wrapper = wrapper_path.read_text(encoding="utf-8")
    frozen_v5_base_sha = "71607c4056dcd3de60d679baa875e07b4e93d0d11f29da72ece24783b8b56b1d"
    assert f'EXPECTED_BASE_LAUNCHER_SHA256="{frozen_v5_base_sha}"' in wrapper
    assert sha256_file(launcher_path) != frozen_v5_base_sha
    expected_exports = (
        'export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v5_embedding_relief_5plus3"',
        'export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3"',
        'export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v5_embedding_relief_5plus3"',
        'export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v5_embedding_relief_5plus3"',
        'export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.log"',
        'export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.verify.log"',
        'export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.preformal.verify.log"',
        'export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.postformal.verify.log"',
        'export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v5_embedding_relief_5plus3.log"',
        'export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v5_embedding_relief_5plus3.verify.log"',
        'export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v5_embedding_relief_5plus3.log"',
        'export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"',
        'export LATENCY_SOURCE_DEVICES="5,6,7"',
        'export LATENCY_TARGET_DEVICES="0,1,2,3,4"',
        'export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"',
        'export LATENCY_TARGET_LAYER_COUNTS="4,15,15,15,15"',
        'export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_embedding_relief_v5"',
    )
    latency_export_lines = tuple(
        line for line in wrapper.splitlines() if line.startswith("export LATENCY_")
    )
    assert latency_export_lines == expected_exports
    latency_exports = "\n".join(latency_export_lines)
    assert "_v4_" not in latency_exports
    assert "8,15,14,14,13" not in latency_exports
    assert wrapper_path.stat().st_mode & 0o111


def test_v6_wrapper_preserves_frozen_old_shared_launcher_sha():
    launcher_path = PROJECT_ROOT / "scripts/launch_latency_after_idle.sh"
    wrapper_path = (
        PROJECT_ROOT / "scripts/launch_latency_single_first_5plus3_after_idle.sh"
    )
    wrapper = wrapper_path.read_text(encoding="utf-8")
    frozen_v6_base_sha = "b2369daa3c377e5aebf45df4472834757e511423518b3d5defeddbb4ec570595"
    assert f'EXPECTED_BASE_LAUNCHER_SHA256="{frozen_v6_base_sha}"' in wrapper
    assert sha256_file(launcher_path) != frozen_v6_base_sha
    expected_exports = (
        'export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v6_single_first_5plus3"',
        'export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3"',
        'export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v6_single_first_5plus3"',
        'export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v6_single_first_5plus3"',
        'export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.log"',
        'export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.verify.log"',
        'export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.preformal.verify.log"',
        'export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.postformal.verify.log"',
        'export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v6_single_first_5plus3.log"',
        'export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v6_single_first_5plus3.verify.log"',
        'export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v6_single_first_5plus3.log"',
        'export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"',
        'export LATENCY_SOURCE_DEVICES="5,6,7"',
        'export LATENCY_TARGET_DEVICES="0,1,2,3,4"',
        'export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"',
        'export LATENCY_TARGET_LAYER_COUNTS="1,16,16,16,15"',
        'export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_single_first_v6"',
    )
    latency_export_lines = tuple(
        line for line in wrapper.splitlines() if line.startswith("export LATENCY_")
    )
    assert latency_export_lines == expected_exports
    latency_exports = "\n".join(latency_export_lines)
    assert "_v5_" not in latency_exports
    assert "4,15,15,15,15" not in latency_exports
    assert wrapper_path.stat().st_mode & 0o111


def test_v7_wrapper_preserves_frozen_old_shared_launcher_sha():
    launcher_path = PROJECT_ROOT / "scripts/launch_latency_after_idle.sh"
    wrapper_path = (
        PROJECT_ROOT
        / "scripts/launch_latency_inference_mode_equal_5plus3_after_idle.sh"
    )
    wrapper = wrapper_path.read_text(encoding="utf-8")
    frozen_v7_base_sha = "dea19e35c15b2208b1be25f24f3ae207e70a6bedc92d6854a961fac34680b19f"
    assert f'EXPECTED_BASE_LAUNCHER_SHA256="{frozen_v7_base_sha}"' in wrapper
    assert sha256_file(launcher_path) != frozen_v7_base_sha
    expected_exports = (
        'export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v7_inference_mode_equal_5plus3"',
        'export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3"',
        'export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v7_inference_mode_equal_5plus3"',
        'export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v7_inference_mode_equal_5plus3"',
        'export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.log"',
        'export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.verify.log"',
        'export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.preformal.verify.log"',
        'export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.postformal.verify.log"',
        'export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v7_inference_mode_equal_5plus3.log"',
        'export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v7_inference_mode_equal_5plus3.verify.log"',
        'export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v7_inference_mode_equal_5plus3.log"',
        'export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"',
        'export LATENCY_SOURCE_DEVICES="5,6,7"',
        'export LATENCY_TARGET_DEVICES="0,1,2,3,4"',
        'export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"',
        'export LATENCY_TARGET_LAYER_COUNTS="13,13,13,13,12"',
        'export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_equal_v7"',
    )
    latency_export_lines = tuple(
        line for line in wrapper.splitlines() if line.startswith("export LATENCY_")
    )
    assert latency_export_lines == expected_exports
    assert "_v6_" not in "\n".join(latency_export_lines)
    assert wrapper_path.stat().st_mode & 0o111


def test_v8_wrapper_binds_fresh_paths_profile_and_current_shared_launcher_sha():
    launcher_path = PROJECT_ROOT / "scripts/launch_latency_after_idle.sh"
    wrapper_path = (
        PROJECT_ROOT
        / "scripts/launch_latency_inference_mode_low_first_5plus3_after_idle.sh"
    )
    wrapper = wrapper_path.read_text(encoding="utf-8")
    assert (
        f'EXPECTED_BASE_LAUNCHER_SHA256="{sha256_file(launcher_path)}"' in wrapper
    )
    expected_exports = (
        'export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v8_inference_mode_low_first_5plus3"',
        'export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3"',
        'export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v8_inference_mode_low_first_5plus3"',
        'export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v8_inference_mode_low_first_5plus3"',
        'export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.log"',
        'export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.verify.log"',
        'export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.preformal.verify.log"',
        'export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.postformal.verify.log"',
        'export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v8_inference_mode_low_first_5plus3.log"',
        'export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v8_inference_mode_low_first_5plus3.verify.log"',
        'export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v8_inference_mode_low_first_5plus3.log"',
        'export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"',
        'export LATENCY_SOURCE_DEVICES="5,6,7"',
        'export LATENCY_TARGET_DEVICES="0,1,2,3,4"',
        'export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"',
        'export LATENCY_TARGET_LAYER_COUNTS="6,15,15,15,13"',
        'export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_inference_low_first_v8"',
    )
    latency_export_lines = tuple(
        line for line in wrapper.splitlines() if line.startswith("export LATENCY_")
    )
    assert latency_export_lines == expected_exports
    assert "_v7_" not in "\n".join(latency_export_lines)
    assert wrapper_path.stat().st_mode & 0o111


def test_v5_documented_launch_checks_wrapper_sha_and_propagates_tee_failure():
    wrapper_path = (
        PROJECT_ROOT / "scripts/launch_latency_embedding_relief_5plus3_after_idle.sh"
    )
    reproduction = (PROJECT_ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")
    session = "nvidia_p8_latency_queue_v5_embedding_relief_5plus3"
    matching = [line for line in reproduction.splitlines() if session in line]
    assert len(matching) == 1
    launch = matching[0]
    assert "set -Eeuo pipefail" in launch
    assert "set -o noclobber" in launch
    assert sha256_file(wrapper_path) in launch
    assert "sha256sum -c" in launch
    assert "test ! -e runs/20260818_p8_latency_main_only_queue_v5" in launch
    assert ": > runs/logs/20260818_p8_latency_main_only_queue_v5" in launch
    assert "tee -a runs/logs/20260818_p8_latency_main_only_queue_v5" in launch
    assert "latency_main_only_queue_v5_embedding_relief_5plus3.log" in launch


def test_v6_documented_launch_checks_wrapper_sha_and_propagates_tee_failure():
    wrapper_path = (
        PROJECT_ROOT / "scripts/launch_latency_single_first_5plus3_after_idle.sh"
    )
    reproduction = (PROJECT_ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")
    session = "nvidia_p8_latency_queue_v6_single_first_5plus3"
    matching = [line for line in reproduction.splitlines() if session in line]
    assert len(matching) == 1
    launch = matching[0]
    assert "set -Eeuo pipefail" in launch
    assert "set -o noclobber" in launch
    assert sha256_file(wrapper_path) in launch
    assert "sha256sum -c" in launch
    assert "test ! -e runs/20260818_p8_latency_main_only_queue_v6" in launch
    assert ": > runs/logs/20260818_p8_latency_main_only_queue_v6" in launch
    assert "tee -a runs/logs/20260818_p8_latency_main_only_queue_v6" in launch
    assert "latency_main_only_queue_v6_single_first_5plus3.log" in launch


def test_v7_documented_launch_checks_wrapper_sha_and_propagates_tee_failure():
    wrapper_path = (
        PROJECT_ROOT
        / "scripts/launch_latency_inference_mode_equal_5plus3_after_idle.sh"
    )
    reproduction = (PROJECT_ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")
    session = "nvidia_p8_latency_queue_v7_inference_mode_equal_5plus3"
    matching = [line for line in reproduction.splitlines() if session in line]
    assert len(matching) == 1
    launch = matching[0]
    assert "set -Eeuo pipefail" in launch
    assert "set -o noclobber" in launch
    assert sha256_file(wrapper_path) in launch
    assert "sha256sum -c" in launch
    assert "test ! -e runs/20260818_p8_latency_main_only_queue_v7" in launch
    assert ": > runs/logs/20260818_p8_latency_main_only_queue_v7" in launch
    assert "tee -a runs/logs/20260818_p8_latency_main_only_queue_v7" in launch
    assert "latency_main_only_queue_v7_inference_mode_equal_5plus3.log" in launch


def test_v8_documented_launch_checks_wrapper_sha_and_propagates_tee_failure():
    wrapper_path = (
        PROJECT_ROOT
        / "scripts/launch_latency_inference_mode_low_first_5plus3_after_idle.sh"
    )
    reproduction = (PROJECT_ROOT / "REPRODUCTION.md").read_text(encoding="utf-8")
    session = "nvidia_p8_latency_queue_v8_inference_mode_low_first_5plus3"
    matching = [line for line in reproduction.splitlines() if session in line]
    assert len(matching) == 1
    launch = matching[0]
    assert "set -Eeuo pipefail" in launch
    assert "set -o noclobber" in launch
    assert sha256_file(wrapper_path) in launch
    assert "sha256sum -c" in launch
    assert "test ! -e runs/20260818_p8_latency_main_only_queue_v8" in launch
    assert ": > runs/logs/20260818_p8_latency_main_only_queue_v8" in launch
    assert "tee -a runs/logs/20260818_p8_latency_main_only_queue_v8" in launch
    assert "latency_main_only_queue_v8_inference_mode_low_first_5plus3.log" in launch


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_wrong_declared_counts_fail_closed(tmp_path, layout):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    path = endpoint / "run_definition.json"
    definition = json.loads(path.read_text())
    definition.pop("request_fingerprint")
    definition["cli"]["target_layer_counts"] = [12, 13, 13, 13, 13]
    definition["request_fingerprint"] = canonical_digest(definition)
    atomic_write_json(path, definition)
    _sidecar(path)
    with pytest.raises(LatencyReportError, match="not an audited latency layout"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_config_counts_tamper_fails_closed(tmp_path, layout):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    path = formal / "config.json"
    configuration = json.loads(path.read_text())
    configuration["target_layer_counts"] = [12, 13, 13, 13, 13]
    atomic_write_json(path, configuration)
    with pytest.raises(LatencyReportError, match="target_layer_counts mismatch"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_actual_model_distribution_tamper_fails_closed(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    setup_path = formal / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    setup["model_layer_sharding"]["target"]["actual_layer_counts"] = [9, 13, 14, 14, 14]
    atomic_write_json(setup_path, setup)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["mapper_setup"] = setup
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="actual audited layer split"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_auxiliary_module_device_tamper_fails_closed(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    setup_path = formal / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    setup["model_layer_sharding"]["target"]["auxiliary_modules"]["lm_head"][
        "actual_device"
    ] = 0
    atomic_write_json(setup_path, setup)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["mapper_setup"] = setup
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="auxiliary module device mismatch"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_target_parameter_inspection_count_tamper_fails_closed(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    setup_path = formal / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    setup["model_layer_sharding"]["target"]["parameter_tensors_inspected"] = 703
    atomic_write_json(setup_path, setup)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["mapper_setup"] = setup
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="inspection count mismatch"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_source_parameter_inspection_count_tamper_fails_closed(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    setup_path = formal / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    setup["model_layer_sharding"]["source"]["parameter_tensors_inspected"] = 439
    atomic_write_json(setup_path, setup)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["mapper_setup"] = setup
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="inspection count mismatch"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_auxiliary_parameter_total_tamper_fails_closed(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    setup_path = formal / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    setup["model_layer_sharding"]["target"][
        "auxiliary_parameter_tensors_inspected"
    ] = 2
    atomic_write_json(setup_path, setup)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["mapper_setup"] = setup
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="auxiliary parameter count"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_actual_mapper_distribution_tamper_fails_closed(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    setup_path = formal / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    second_shard_last_layer = sum(layout["target_layer_counts"][:2]) - 1
    setup["per_layer"][second_shard_last_layer]["device"] = "cuda:2"
    atomic_write_json(setup_path, setup)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["mapper_setup"] = setup
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="actual target distribution"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    "layout", (V4_LAYOUT, V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v4", "v5", "v6", "v7")
)
def test_exact_count_endpoint_receipt_counts_tamper_fails_closed(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    path = endpoint / "endpoint_gate_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["target_layer_counts"] = [12, 13, 13, 13, 13]
    atomic_write_json(path, receipt)
    _sidecar(path, schema=1, special=endpoint / "endpoint_gate_receipt.sha256.json")
    with pytest.raises(LatencyReportError, match="target_layer_counts mismatch"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize("layout", (V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v5", "v6", "v7"))
def test_recent_exact_receipt_profile_tamper_fails_closed(tmp_path, layout):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    path = endpoint / "endpoint_gate_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["device_layout_profile"] = "balanced_5target_3source_low_first_v4"
    atomic_write_json(path, receipt)
    _sidecar(path, schema=1, special=endpoint / "endpoint_gate_receipt.sha256.json")
    with pytest.raises(LatencyReportError, match="device_layout_profile mismatch"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize("layout", (V5_LAYOUT, V6_LAYOUT, V7_LAYOUT), ids=("v5", "v6", "v7"))
def test_recent_exact_launcher_rejects_formal_runtime_identity_tamper(
    tmp_path, layout
):
    endpoint, formal = _make_layout_runs(tmp_path, layout)
    setup_path = formal / "mapper_setup.json"
    setup = json.loads(setup_path.read_text())
    setup["model_layer_sharding"]["target"]["auxiliary_modules"]["lm_head"][
        "actual_device"
    ] = 0
    atomic_write_json(setup_path, setup)
    completed = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/launch_latency_after_idle.sh; verify_formal_identity_against_endpoint",
        ],
        cwd=PROJECT_ROOT,
        env=_launcher_layout_environment(endpoint, formal, layout),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "actual model layer sharding differs" in completed.stderr


@pytest.mark.parametrize(
    ("source", "target", "visibility"),
    (
        ((1,), (0,), "0,1"),
        ((5, 6, 7), (0, 1, 2, 3, 4), "0,1,2,3,4,5,6,7"),
        ((5, 6, 7), (0, 1, 2, 3), "3,4,5,6,7,0,1,2"),
    ),
    ids=("one-plus-one", "wrong-cvd", "missing-target-device"),
)
def test_non_audited_device_layouts_fail_closed(
    latency_runs, source, target, visibility
):
    endpoint, formal = latency_runs
    _resign_definition_with_layout(
        endpoint, source=source, target=target, visibility=visibility
    )
    with pytest.raises(LatencyReportError, match="not an audited latency layout"):
        build_latency_report(endpoint, formal)


def test_cli_writes_report(latency_runs, tmp_path, capsys):
    endpoint, formal = latency_runs
    output = tmp_path / "cli-output"
    assert cli.main([str(endpoint), str(formal), "--output-dir", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "complete"
    assert printed["formal_lengths"] == 10
    assert (output / "sha256_manifest.json").is_file()


def test_missing_formal_sidecar_fails_closed_without_outputs(latency_runs, tmp_path):
    endpoint, formal = latency_runs
    (formal / "length_32768.json.sha256.json").unlink()
    output = tmp_path / "must-not-exist"
    with pytest.raises(LatencyReportError, match="missing"):
        build_latency_report(endpoint, formal)
    assert not output.exists()


def test_metrics_checkpoint_drift_is_rejected_even_when_hashes_are_resigned(latency_runs):
    endpoint, formal = latency_runs
    path = formal / "length_00064.json"
    checkpoint = json.loads(path.read_text())
    checkpoint["target_reprefill"]["samples"][0] += 1.0
    atomic_write_json(path, checkpoint)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="metrics.length_results differs"):
        build_latency_report(endpoint, formal)


def test_resigned_supplemental_contamination_is_rejected(latency_runs):
    endpoint, formal = latency_runs
    checkpoint_path = formal / "length_00064.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["protocol_complete"] = checkpoint["mapper_handoff_end_to_end"]
    atomic_write_json(checkpoint_path, checkpoint)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["length_results"][0] = checkpoint
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="supplemental fields"):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    ("metric", "field", "message"),
    (
        ("target_reprefill", "inference_mode_enabled", "inference-mode evidence"),
        (
            "mapper_handoff_end_to_end",
            "autograd_graph_disabled",
            "disabled-autograd evidence",
        ),
    ),
)
def test_resigned_missing_main_metric_inference_evidence_is_rejected(
    latency_runs, metric, field, message
):
    endpoint, formal = latency_runs
    checkpoint_path = formal / "length_00064.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint[metric][field] = False
    atomic_write_json(checkpoint_path, checkpoint)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["length_results"][0] = checkpoint
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match=message):
        build_latency_report(endpoint, formal)


def test_resigned_invalid_sample_statistics_are_rejected(latency_runs):
    endpoint, formal = latency_runs
    checkpoint_path = formal / "length_00064.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["target_reprefill"]["median"] += 0.5
    atomic_write_json(checkpoint_path, checkpoint)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["length_results"][0] = checkpoint
    atomic_write_json(metrics_path, metrics)
    _write_metrics_csv(formal / "metrics.csv", metrics["length_results"])
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="median is inconsistent"):
        build_latency_report(endpoint, formal)


def test_formal_50_30_contract_is_required(latency_runs):
    endpoint, formal = latency_runs
    config_path = formal / "config.json"
    config = json.loads(config_path.read_text())
    config["warmup"] = 49
    atomic_write_json(config_path, config)
    with pytest.raises(LatencyReportError, match="warmup mismatch"):
        build_latency_report(endpoint, formal)


def test_endpoint_receipt_semantics_are_rechecked_after_resigning(latency_runs):
    endpoint, formal = latency_runs
    path = endpoint / "endpoint_gate_receipt.json"
    receipt = json.loads(path.read_text())
    receipt["lengths"]["32768"]["minimum_peak_reserved_headroom_bytes"] += 1
    atomic_write_json(path, receipt)
    _sidecar(path, schema=1, special=endpoint / "endpoint_gate_receipt.sha256.json")
    with pytest.raises(LatencyReportError, match="headroom disagrees"):
        build_latency_report(endpoint, formal)


def test_output_refuses_partial_or_different_existing_set(latency_runs, tmp_path):
    report = build_latency_report(*latency_runs)
    output = tmp_path / "partial"
    output.mkdir()
    atomic_write_json(output / "latency_report.json", {"stale": True})
    with pytest.raises(LatencyReportError, match="partial pre-existing"):
        write_latency_report_outputs(report, output)


def test_staged_v9_endpoint_and_formal_full_report_happy_path(tmp_path):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)
    report = build_latency_report(endpoint, formal)
    protocol = report["protocol"]
    assert protocol["device_layout_profile"] == "staged_6target_2cache_v9"
    assert protocol["source_model_prefill_devices"] == [5, 6, 7]
    assert protocol["source_model_prefill_layer_counts"] == [14, 13, 13]
    assert protocol["source_cache_staging_policy"] == SOURCE_CACHE_STAGING_POLICY
    assert protocol["source_model_live_during_timed_cells"] is False
    assert protocol["cpu_staging_and_gpu_restage_included_in_mapper_timing"] is False
    endpoint_definition = json.loads((endpoint / "run_definition.json").read_text())
    formal_definition = json.loads((formal / "run_definition.json").read_text())
    assert endpoint_definition["analytical_memory"][
        "all_prefix_lengths_cpu_staging_exact_bytes"
    ] == 5_379_194_880
    assert formal_definition["analytical_memory"][
        "all_prefix_lengths_cpu_staging_exact_bytes"
    ] == 10_726_932_480
    assert formal_definition["analytical_memory"][
        "approx_source_per_device_lower_bound_bytes_by_layer_fraction"
    ] == [5 << 30, 5 << 30]
    assert "model-residency" in report["hardware_comparison"]["warning"]


def test_staged_v9_launcher_receipt_and_formal_identity_full_chain(tmp_path):
    endpoint, formal = _make_layout_runs(tmp_path, V9_LAYOUT)
    environment = _launcher_layout_environment(endpoint, formal, V9_LAYOUT)
    for verifier in (
        "verify_endpoint_policy",
        "verify_endpoint_receipt",
        "verify_formal_identity_against_endpoint",
    ):
        completed = subprocess.run(
            [
                "bash",
                "-c",
                f"source scripts/launch_latency_after_idle.sh; {verifier}",
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("prepared_order", "canonical missing suffix"),
        ("prepared_duplicate", "prepared-length map mismatch"),
        ("host_weight", "host-memory gate binding"),
        ("vocabulary", "immutable source binding"),
        ("source_sharding", "immutable source binding"),
        ("release_gc", "source-model release proof"),
        ("workspace_status", "cuBLAS workspace release proof"),
        ("gpu_cpu_hash", "GPU/CPU preparation hashes differ"),
        ("cpu_structure", "structure mismatch"),
    ),
)
def test_staged_v9_resigned_session_tamper_fails_closed(tmp_path, case, expected):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)

    def mutate(session):
        if case == "prepared_order":
            session["prepared_missing_lengths"] = list(
                reversed(session["prepared_missing_lengths"])
            )
        elif case == "prepared_duplicate":
            session["prepared_missing_lengths"] = [64, 64]
            session["preparations"] = {"64": session["preparations"]["64"]}
            session["cpu_staging_total_bytes"] = 2 * 163_840 * 64
        elif case == "host_weight":
            gate = session["host_memory_staging_gate"]
            gate["source_weight_shard_bytes"] += 1
            gate["required_available_bytes"] += 1
        elif case == "vocabulary":
            session["source_vocabulary_size"] += 1
        elif case == "source_sharding":
            session["source_model_prefill_sharding"]["auxiliary_modules"][
                "lm_head"
            ]["actual_device"] = 6
        elif case == "release_gc":
            session["source_model_release"]["gc_completed"] = False
        elif case == "workspace_status":
            session["source_model_release"][
                "cuda_workspace_and_allocator_release"
            ]["status"] = "failed"
        elif case == "gpu_cpu_hash":
            session["preparations"]["64"]["cpu_staged_raw_bit_sha256"] = _sha_text(
                "tampered-cpu-cache"
            )
        elif case == "cpu_structure":
            session["preparations"]["64"]["cpu_cache_structure"][0][
                "device"
            ] = "cuda:0"
        else:  # pragma: no cover
            raise AssertionError(case)

    _resign_staged_session(formal, mutate)
    with pytest.raises(LatencyReportError, match=expected):
        build_latency_report(endpoint, formal)


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("restage_policy", "GPU restage proof"),
        ("restage_sync", "GPU restage proof"),
        ("release_driver", "restaged-cache release proof"),
        ("peak_baseline", "staged peak/release binding"),
        ("peak_after", "staged peak/release binding"),
    ),
)
def test_staged_v9_resigned_lifecycle_peak_tamper_fails_closed(
    tmp_path, case, expected
):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)

    def mutate(checkpoint):
        lifecycle = checkpoint["source_cache_lifecycle"]
        peak = checkpoint["cuda_peak_memory"]["devices"]["6"]
        if case == "restage_policy":
            lifecycle["restage"]["policy"] = "tampered"
        elif case == "restage_sync":
            lifecycle["restage"][
                "cuda_synchronized_and_empty_cache_before_outer_peak_reset"
            ] = False
        elif case == "release_driver":
            lifecycle["restaged_cache_release"]["allocator_driver_after_release"][
                "6"
            ]["driver_used_bytes"] += 1
        elif case == "peak_baseline":
            peak["allocated_bytes_before"] += 512
            peak["reserved_bytes_before"] += 512
            peak["peak_allocated_bytes_immediately_after_reset"] += 512
            peak["peak_reserved_bytes_immediately_after_reset"] += 512
            peak["allocated_growth_to_peak_bytes"] -= 512
            peak["reserved_growth_to_peak_bytes"] -= 512
        elif case == "peak_after":
            peak["allocated_bytes_after"] = 512
            peak["reserved_bytes_after"] = 512
        else:  # pragma: no cover
            raise AssertionError(case)

    _resign_checkpoint(formal, 64, mutate)
    with pytest.raises(LatencyReportError, match=expected):
        build_latency_report(endpoint, formal)


def test_staged_v9_resigned_active_session_tamper_fails_closed(tmp_path):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["active_staging_session"]["prepared_missing_lengths"] = list(
        PAPER_LENGTHS[1:]
    )
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="active_staging_session binding"):
        build_latency_report(endpoint, formal)


def test_staged_v9_partial_resume_two_sessions_is_accepted(tmp_path):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)
    _convert_to_partial_staged_resume(formal, split_index=3)
    report = build_latency_report(endpoint, formal)
    assert report["status"] == "complete"


def test_staged_v9_session_a_b_a_reentry_is_rejected(tmp_path):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)
    _convert_to_partial_staged_resume(formal, split_index=3)
    first_checkpoint = json.loads(
        (formal / "length_00064.json").read_text()
    )
    first_name = first_checkpoint["source_cache_lifecycle"]["session_evidence_path"]
    first_session = json.loads((formal / first_name).read_text())
    first_sha = sha256_file(formal / first_name)

    def mutate(checkpoint):
        _apply_staged_checkpoint(
            checkpoint,
            session=first_session,
            session_name=first_name,
            session_sha256=first_sha,
            layout=V9_LAYOUT,
        )

    _resign_checkpoint(formal, 32768, mutate)
    with pytest.raises(LatencyReportError, match="A-B-A re-entry"):
        build_latency_report(endpoint, formal)


def test_staged_v9_old_session_unreferenced_preparation_is_still_validated(tmp_path):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)
    _convert_to_partial_staged_resume(formal, split_index=3)

    def mutate(session):
        session["preparations"]["32768"]["cpu_cache_structure"][0][
            "key_dtype"
        ] = "torch.float16"

    _resign_staged_session(formal, mutate)
    with pytest.raises(LatencyReportError, match="structure mismatch"):
        build_latency_report(endpoint, formal)


def test_staged_v9_finalization_only_resume_requires_no_active_session(tmp_path):
    shared = (_hardware(V9_LAYOUT), _runtime_identity(), _project_code())
    endpoint = _make_run(
        tmp_path / "endpoint_v9",
        lengths=ENDPOINT_LENGTHS,
        warmup=1,
        repeats=2,
        shared=shared,
        layout=V9_LAYOUT,
    )
    formal = _make_run(
        tmp_path / "formal_v9",
        lengths=PAPER_LENGTHS,
        warmup=50,
        repeats=30,
        shared=shared,
        layout=V9_LAYOUT,
    )
    _endpoint_receipt(endpoint, V9_LAYOUT)
    metrics_path = formal / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["resumed"] = True
    metrics["reused_checkpoint_lengths"] = list(PAPER_LENGTHS)
    metrics["active_staging_session"] = None
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    assert build_latency_report(endpoint, formal)["status"] == "complete"

    metrics = json.loads(metrics_path.read_text())
    metrics["resumed"] = False
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="require resumed=true"):
        build_latency_report(endpoint, formal)

    metrics = json.loads(metrics_path.read_text())
    metrics["resumed"] = True
    first = json.loads((formal / "length_00064.json").read_text())
    session_name = first["source_cache_lifecycle"]["session_evidence_path"]
    session = json.loads((formal / session_name).read_text())
    metrics["active_staging_session"] = {
        "staging_session_id": session["staging_session_id"],
        "session_evidence_path": session_name,
        "session_evidence_sha256": sha256_file(formal / session_name),
        "prepared_missing_lengths": list(PAPER_LENGTHS),
        "host_memory_staging_gate": session["host_memory_staging_gate"],
    }
    atomic_write_json(metrics_path, metrics)
    _rebuild_manifest_and_result(formal)
    with pytest.raises(LatencyReportError, match="finalization-only"):
        build_latency_report(endpoint, formal)
