"""Independent paper-style latency runner for Qwen3 32B -> 14B.

The source model creates one cache per sequence length before either timer is
entered.  The timed baseline calls only the 14B transformer body (no LM head),
and the timed mapper call includes cache handoff, inverse/source and target
RoPE work, eager FP32 affine maps, BF16 conversion, and target cache build.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, BinaryIO
import uuid
import weakref

import yaml

from .eval_common import (
    EVALUATION_SCHEMA_VERSION,
    assert_evaluation_output,
    code_identity,
    command_evidence,
    hashed_json,
    immutable_hashed_json,
    mapper_manifest_identity,
    model_artifact_identity,
    verify_hashed_json,
)
from .protocol import (
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    canonical_sha256,
    read_json,
    sha256_file,
)


DEFAULT_CONFIG = EXPERIMENT_ROOT / "config/latency.yaml"
PAPER_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
WARMUP = 50
REPEATS = 30
GLOBAL_GPU_LOCK = REPOSITORY_ROOT / "runs/gpu8_experiment.lock"
SOURCE_CACHE_STAGING_POLICY = (
    "untimed_6gpu_prefill_cpu_raw_bit_stage_release_5gpu_restage"
)
SOURCE_PREFILL_DEVICES = (0, 1, 2, 3, 4, 5)
SOURCE_PREFILL_LAYER_COUNTS = (10, 11, 11, 11, 11, 10)
SOURCE_CACHE_DEVICES = (3, 4, 5, 6, 7)
SOURCE_CACHE_LAYER_COUNTS = (13, 13, 13, 13, 12)
TARGET_DEVICES = (0, 1, 2)
TARGET_LAYER_COUNTS = (13, 14, 13)
MAPPER_PARAMETER_BYTES = 6_711_214_080
HOST_STAGING_OVERHEAD_BYTES = 34_359_738_368
LATENCY_PACKAGE_DISTRIBUTIONS = {
    "transformers": "transformers",
    "accelerate": "accelerate",
    "flash_attention": "flash-attn",
    "safetensors": "safetensors",
}
CODE_FILES = (
    "reverse_kv/__init__.py",
    "reverse_kv/reverse_latency.py",
    "reverse_kv/eval_common.py",
    "reverse_kv/provenance.py",
    "reverse_kv/runtime.py",
    "reverse_kv/protocol.py",
    "scripts/benchmark_latency.py",
    "config/latency.yaml",
    "artifacts/inputs/code_freeze.json",
)
SHARED_CODE_FILES = (
    "src/kvcache_transfer/__init__.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/transfer.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/io.py",
)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_latency_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("latency config must be a mapping")
    if raw.get("schema_version") != 1 or raw.get("direction") != "qwen3_32b_to_14b":
        raise ValueError("unsupported reverse latency protocol")
    protocol = raw.get("protocol")
    paths = raw.get("paths")
    runtime = raw.get("runtime")
    reference = raw.get("paper_h100_reference_only")
    paper_reference = raw.get("paper_reference")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol, paths, runtime, reference, paper_reference)
    ):
        raise ValueError("latency config lacks a required mapping")
    if dict(paper_reference) != {
        "paper": "arXiv:2608.03893v1",
        "sections": ["4.7", "Appendix G"],
        "table": 16,
        "pair": "Qwen3 32B->14B (k=20)",
    }:
        raise ValueError("reverse latency paper reference changed")
    fixed_protocol = {
        "lengths": list(PAPER_LENGTHS),
        "warmup": WARMUP,
        "timed_repeats": REPEATS,
        "batch_size": 1,
        "seed": 20260813,
        "dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "mapper_execution": "eager",
        "torch_compile": False,
        "cuda_graphs": False,
        "mapper_token_chunk": 1024,
        "source_prefill_in_timed_region": False,
        "target_reprefill_scope": "transformer_body_with_kv_excluding_lm_head",
        "mapper_scope": (
            "source_cache_handoff_inverse_rope_fp32_affine_target_rope_bf16_cache_build"
        ),
        "source_cache_lifecycle": SOURCE_CACHE_STAGING_POLICY,
        "cpu_staging_in_timed_region": False,
        "gpu_restage_in_timed_region": False,
        "mapper_timer_start": "preexisting_gpu_resident_source_kv",
        "endpoint_memory_smoke_length": 32768,
        "minimum_post_operation_headroom_mib": 2048,
    }
    for key, expected in fixed_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"reverse latency invariant {key} changed")
    fixed_runtime = {
        "source_prefill_devices": list(SOURCE_PREFILL_DEVICES),
        "source_prefill_layer_counts": list(SOURCE_PREFILL_LAYER_COUNTS),
        "source_cache_devices": list(SOURCE_CACHE_DEVICES),
        "source_cache_layer_counts": list(SOURCE_CACHE_LAYER_COUNTS),
        "target_devices": list(TARGET_DEVICES),
        "target_layer_counts": list(TARGET_LAYER_COUNTS),
        "host_staging_overhead_bytes": HOST_STAGING_OVERHEAD_BYTES,
        "require_cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "idle_memory_limit_mib": 64,
        "idle_consecutive_samples": 12,
        "idle_sample_interval_seconds": 5,
        "idle_timeout_seconds": 86400,
    }
    for key, expected in fixed_runtime.items():
        if runtime.get(key) != expected:
            raise ValueError(f"reverse latency topology invariant {key} changed")
    if reference.get("hardware") != (
        "8xH100 NVLink; do not compare absolute 4090 latency as bit-exact"
    ) or reference.get("anchors") != {
        "64": {"mapper_ms": 11.6, "target_reprefill_ms": 39.2},
        "8192": {"mapper_ms": 101.9, "target_reprefill_ms": 501.0},
        "32768": {"mapper_ms": 427.1, "target_reprefill_ms": 2952.7},
    } or reference.get("reported_speedup_range") != [3.3, 6.9]:
        raise ValueError("reverse H100 reference anchors changed")
    result = dict(raw)
    result["config_path"] = config_path
    result["protocol"] = dict(protocol)
    result["runtime"] = dict(runtime)
    result["paths"] = dict(paths)
    for key in ("source_model", "target_model", "mapper_manifest"):
        result["paths"][key] = _resolve(config_path.parent, str(paths[key]))
    result["paper_h100_reference_only"] = dict(reference)
    return result


def _gpu_query() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memory = command_evidence(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = command_evidence(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    if memory.get("returncode") != 0 or processes.get("returncode") != 0:
        raise RuntimeError("nvidia-smi GPU idle query failed")
    memory_rows: list[dict[str, Any]] = []
    for line in str(memory["stdout"]).splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("malformed nvidia-smi memory row")
        memory_rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "memory_used_mib": int(fields[2]),
                "memory_total_mib": int(fields[3]),
            }
        )
    process_rows: list[dict[str, Any]] = []
    for line in str(processes["stdout"]).splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("malformed nvidia-smi process row")
        process_rows.append(
            {
                "uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mib": (
                    int(fields[3]) if fields[3].isdigit() else fields[3]
                ),
            }
        )
    return memory_rows, process_rows


def wait_for_idle_gpus(configuration: Mapping[str, Any]) -> dict[str, Any]:
    runtime = configuration["runtime"]
    required_visible = str(runtime["require_cuda_visible_devices"])
    observed_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if observed_visible != required_visible:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly {required_visible!r}, got {observed_visible!r}"
        )
    required_physical = {int(value) for value in required_visible.split(",")}
    needed = int(runtime["idle_consecutive_samples"])
    limit = int(runtime["idle_memory_limit_mib"])
    interval = float(runtime["idle_sample_interval_seconds"])
    deadline = time.monotonic() + float(runtime["idle_timeout_seconds"])
    samples: list[dict[str, Any]] = []
    consecutive = 0
    while consecutive < needed:
        if time.monotonic() > deadline:
            raise TimeoutError("timed out waiting for all eight latency GPUs to become idle")
        memory, processes = _gpu_query()
        selected = [row for row in memory if row["physical_index"] in required_physical]
        selected_uuids = {row["uuid"] for row in selected}
        busy_processes = [row for row in processes if row["uuid"] in selected_uuids]
        passed = (
            len(selected) == len(required_physical)
            and not busy_processes
            and all(row["memory_used_mib"] <= limit for row in selected)
        )
        sample = {
            "sample_index": len(samples),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "passed": passed,
            "memory": selected,
            "compute_processes": busy_processes,
        }
        samples.append(sample)
        consecutive = consecutive + 1 if passed else 0
        if consecutive < needed:
            time.sleep(interval)
    return {
        "status": "passed",
        "required_consecutive_samples": needed,
        "memory_limit_mib": limit,
        "samples": samples,
        "final_consecutive_samples": consecutive,
        "cuda_visible_devices": observed_visible,
    }


def _validate_gpu_idle_gate(path: Path) -> dict[str, Any]:
    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "status": "passed",
        "required_consecutive_samples": 12,
        "memory_limit_mib": 64,
        "final_consecutive_samples": 12,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"latency GPU idle gate {key} changed")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) < 12:
        raise ValueError("latency GPU idle gate lacks twelve passing samples")
    trailing = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or sample.get("sample_index") != index:
            raise ValueError("latency GPU idle gate sample ordering changed")
        memory = sample.get("memory")
        processes = sample.get("compute_processes")
        passed = (
            isinstance(memory, list)
            and [row.get("physical_index") for row in memory] == list(range(8))
            and all(
                type(row.get("memory_used_mib")) is int
                and 0 <= row["memory_used_mib"] <= 64
                and type(row.get("memory_total_mib")) is int
                and row["memory_total_mib"] > 0
                and isinstance(row.get("uuid"), str)
                and bool(row["uuid"])
                for row in memory
            )
            and processes == []
        )
        if sample.get("passed") is not passed:
            raise ValueError("latency GPU idle gate sample verdict changed")
        trailing = trailing + 1 if passed else 0
    if trailing != 12:
        raise ValueError("latency GPU idle gate trailing pass count changed")
    return dict(value)


def hardware_snapshot() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise RuntimeError("reverse production latency requires exactly eight visible CUDA GPUs")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "logical_index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
            }
        )
    peer_access = {
        f"{left}->{right}": bool(torch.cuda.can_device_access_peer(left, right))
        for left in range(8)
        for right in range(8)
        if left != right
    }
    topology = command_evidence(["nvidia-smi", "topo", "-m"])
    p2p_read = command_evidence(["nvidia-smi", "topo", "-p2p", "r"])
    driver_identity = nvidia_driver_evidence()
    if any(
        evidence.get("returncode") != 0 for evidence in (topology, p2p_read)
    ):
        raise RuntimeError("formal latency requires successful nvidia-smi topology/P2P evidence")
    topology_text = str(topology.get("stdout", ""))
    nvlink_present = any(f"NV{links}" in topology_text for links in range(1, 19))
    names = [str(row["name"]) for row in devices]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "python_version": sys.version,
        "runtime_packages": latency_package_versions(),
        "nvidia_driver_version": driver_identity["version"],
        "nvidia_smi_driver_query": driver_identity["query"],
        "nvidia_smi_driver_rows": driver_identity["rows"],
        "devices": devices,
        "peer_access": peer_access,
        "all_cross_device_peer_access": all(peer_access.values()),
        "nvlink_present": nvlink_present,
        "paper_h100_nvlink_hardware_match": bool(
            nvlink_present and all("H100" in name for name in names)
        ),
        "absolute_latency_requires_local_hardware_label": True,
        "nvidia_smi_topology": topology,
        "nvidia_smi_p2p_read": p2p_read,
    }


def nvidia_driver_evidence() -> dict[str, Any]:
    """Query and parse one driver version for all eight physical GPUs."""

    driver = command_evidence(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if driver.get("returncode") != 0:
        raise RuntimeError("formal latency requires successful nvidia-smi driver evidence")
    driver_rows = []
    for line in str(driver.get("stdout", "")).splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError("malformed nvidia-smi driver-version row")
        driver_rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
            }
        )
    if [row["physical_index"] for row in driver_rows] != list(range(8)):
        raise RuntimeError("driver-version evidence does not cover physical GPUs 0..7")
    driver_versions = sorted({str(row["driver_version"]) for row in driver_rows})
    if len(driver_versions) != 1:
        raise RuntimeError("formal latency requires one NVIDIA driver version across all GPUs")
    return {
        "version": driver_versions[0],
        "query": driver,
        "rows": driver_rows,
    }


def latency_package_versions() -> dict[str, dict[str, str]]:
    """Return fail-closed distribution identities included in the run digest."""

    result: dict[str, dict[str, str]] = {}
    for role, distribution in LATENCY_PACKAGE_DISTRIBUTIONS.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"formal latency dependency is not installed: {distribution}"
            ) from exc
        if not version:
            raise RuntimeError(f"formal latency dependency has no version: {distribution}")
        result[role] = {"distribution": distribution, "version": version}
    return result


def cuda_memory_snapshot(label: str) -> dict[str, Any]:
    import torch

    memory, processes = _gpu_query()
    return {
        "label": label,
        "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "torch": {
            str(index): {
                "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
            }
            for index in range(torch.cuda.device_count())
        },
        "nvidia_smi_memory": memory,
        "nvidia_smi_compute_processes": processes,
    }


def deterministic_prefix(length: int, vocabulary_size: int, seed: int) -> Any:
    import torch

    if length not in PAPER_LENGTHS:
        raise ValueError("prefix length is not in the frozen paper sweep")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + length)
    low = min(100, max(vocabulary_size - 1, 0))
    if low >= vocabulary_size:
        low = 0
    return torch.randint(low, vocabulary_size, (1, length), generator=generator)


def prefix_identity(prefix: Any) -> dict[str, Any]:
    import torch

    if prefix.device.type != "cpu" or prefix.dtype != torch.int64 or prefix.ndim != 2:
        raise ValueError("latency prefix must be CPU int64 [1,T]")
    contiguous = prefix.contiguous()
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "nbytes": contiguous.numel() * contiguous.element_size(),
        "sha256": hashlib.sha256(memoryview(contiguous.numpy()).cast("B")).hexdigest(),
    }


def _expanded_layer_devices(
    devices: Sequence[int], layer_counts: Sequence[int], *, expected_layers: int
) -> tuple[int, ...]:
    if len(devices) != len(layer_counts) or any(type(value) is not int for value in layer_counts):
        raise ValueError("device/count topology is malformed")
    if any(int(value) <= 0 for value in layer_counts) or sum(layer_counts) != expected_layers:
        raise ValueError("device/count topology does not cover the exact layer count")
    result = tuple(
        int(device)
        for device, count in zip(devices, layer_counts, strict=True)
        for _ in range(int(count))
    )
    if len(result) != expected_layers:
        raise AssertionError("expanded layer topology has the wrong length")
    return result


def staged_runtime_topology(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable sequential-reuse topology used by formal latency."""

    runtime = configuration["runtime"]
    prefill_devices = tuple(int(value) for value in runtime["source_prefill_devices"])
    prefill_counts = tuple(int(value) for value in runtime["source_prefill_layer_counts"])
    cache_devices = tuple(int(value) for value in runtime["source_cache_devices"])
    cache_counts = tuple(int(value) for value in runtime["source_cache_layer_counts"])
    target_devices = tuple(int(value) for value in runtime["target_devices"])
    target_counts = tuple(int(value) for value in runtime["target_layer_counts"])
    _expanded_layer_devices(prefill_devices, prefill_counts, expected_layers=64)
    _expanded_layer_devices(cache_devices, cache_counts, expected_layers=64)
    _expanded_layer_devices(target_devices, target_counts, expected_layers=40)
    if set(cache_devices) & set(target_devices):
        raise ValueError("timed source-cache and target devices must be disjoint")
    if set(prefill_devices) | set(target_devices) | set(cache_devices) != set(range(8)):
        raise ValueError("staged latency topology must account for all eight GPUs")
    return {
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "sequential_gpu_reuse": True,
        "source_model_live_with_target_or_mapper": False,
        "source_model_prefill": {
            "devices": list(prefill_devices),
            "layer_counts": list(prefill_counts),
        },
        "source_cache_restage": {
            "devices": list(cache_devices),
            "layer_counts": list(cache_counts),
        },
        "target_model_and_mapper": {
            "devices": list(target_devices),
            "layer_counts": list(target_counts),
        },
        "timed_cache_target_disjoint": True,
        "prefill_target_sequential_overlap": sorted(set(prefill_devices) & set(target_devices)),
        "prefill_cache_sequential_overlap": sorted(set(prefill_devices) & set(cache_devices)),
        "cuda_visible_devices": runtime["require_cuda_visible_devices"],
    }


def source_cache_exact_bytes(length: int) -> int:
    if type(length) is not int or length <= 0:
        raise ValueError("source-cache length must be an exact positive integer")
    return 64 * 2 * 8 * length * 128 * 2


def host_memory_staging_gate(
    *,
    lengths: Sequence[int],
    source_weight_bytes: int,
    target_weight_bytes: int,
    overhead_bytes: int,
) -> dict[str, Any]:
    """Require enough host RAM for every staged cache and conservative setup slack."""

    fields: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        pieces = raw.strip().split()
        if len(pieces) >= 2 and pieces[1] == "kB":
            fields[name] = int(pieces[0]) * 1024
    available = fields.get("MemAvailable")
    if available is None:
        raise RuntimeError("/proc/meminfo lacks MemAvailable")
    cache_bytes = sum(source_cache_exact_bytes(int(length)) for length in lengths)
    required = (
        int(source_weight_bytes)
        + int(target_weight_bytes)
        + MAPPER_PARAMETER_BYTES
        + cache_bytes
        + int(overhead_bytes)
    )
    if available < required:
        raise RuntimeError(f"insufficient host RAM for staged latency: {available} < {required}")
    return {
        "status": "passed",
        "scope": "before_source_model_load",
        "mem_available_bytes": available,
        "required_available_bytes": required,
        "source_weight_shard_bytes": int(source_weight_bytes),
        "target_weight_shard_bytes": int(target_weight_bytes),
        "mapper_parameter_bytes": MAPPER_PARAMETER_BYTES,
        "staged_cache_bytes": cache_bytes,
        "staged_lengths": [int(value) for value in lengths],
        "conservative_overhead_bytes": int(overhead_bytes),
    }


def _cache_raw_sha256(cache: Any) -> str:
    """Hash cache metadata and every raw tensor byte with bounded host scratch."""

    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import cache_to_legacy

    digest = hashlib.sha256()
    for layer, (key, value) in enumerate(cache_to_legacy(cache)):
        for kind, tensor in (("key", key), ("value", value)):
            metadata = json.dumps(
                {
                    "layer": layer,
                    "kind": kind,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(metadata).to_bytes(8, "little"))
            digest.update(metadata)
            host = tensor.detach().to(device="cpu", copy=True).contiguous()
            digest.update(memoryview(host.view(torch.uint8).numpy()).cast("B"))
            del host
    return digest.hexdigest()


def _cache_structure(
    cache: Any,
    *,
    expected_length: int,
    expected_layer_devices: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import cache_sequence_length, cache_to_legacy

    pairs = cache_to_legacy(cache)
    if len(pairs) != 64 or cache_sequence_length(cache) != expected_length:
        raise RuntimeError("staged source cache has the wrong layer/sequence count")
    if len(expected_layer_devices) != 64:
        raise ValueError("source cache topology must name exactly 64 layer devices")
    rows: list[dict[str, Any]] = []
    total = 0
    for layer, ((key, value), expected_device) in enumerate(
        zip(pairs, expected_layer_devices, strict=True)
    ):
        expected_shape = (1, 8, expected_length, 128)
        if (
            tuple(key.shape) != expected_shape
            or tuple(value.shape) != expected_shape
            or key.dtype != torch.bfloat16
            or value.dtype != torch.bfloat16
            or str(key.device) != expected_device
            or str(value.device) != expected_device
        ):
            raise RuntimeError(f"staged source cache layer {layer} violates its contract")
        key_bytes = key.numel() * key.element_size()
        value_bytes = value.numel() * value.element_size()
        total += key_bytes + value_bytes
        rows.append(
            {
                "layer": layer,
                "device": expected_device,
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
                "key_nbytes": key_bytes,
                "value_nbytes": value_bytes,
            }
        )
    if total != source_cache_exact_bytes(expected_length):
        raise RuntimeError("staged source cache exact byte count changed")
    return rows, total


def stage_source_cache_on_cpu(
    cache: Any,
    *,
    length: int,
    identity: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Copy one 32B cache to CPU and prove full raw-bit identity."""

    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import build_dynamic_cache

    runtime = configuration["runtime"]
    prefill_layers = _expanded_layer_devices(
        runtime["source_prefill_devices"],
        runtime["source_prefill_layer_counts"],
        expected_layers=64,
    )
    gpu_devices = tuple(f"cuda:{value}" for value in prefill_layers)
    gpu_structure, gpu_bytes = _cache_structure(
        cache, expected_length=length, expected_layer_devices=gpu_devices
    )
    gpu_hash = _cache_raw_sha256(cache)
    staged = build_dynamic_cache(
        cache,
        layer_devices=("cpu",) * 64,
        dtype=torch.bfloat16,
        clone=True,
        detach=True,
        non_blocking=False,
    )
    cpu_structure, cpu_bytes = _cache_structure(
        staged, expected_length=length, expected_layer_devices=("cpu",) * 64
    )
    cpu_hash = _cache_raw_sha256(staged)
    if gpu_bytes != cpu_bytes or gpu_hash != cpu_hash:
        raise RuntimeError("GPU-to-CPU source-cache staging changed raw bits")
    return staged, {
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": length,
        "deterministic_prefix": dict(identity),
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "copy_semantics": "blocking_detach_to_cpu_bf16_exact_clone",
        "full_raw_bit_sha256_computed": True,
        "source_gpu_raw_bit_sha256": gpu_hash,
        "cpu_staged_raw_bit_sha256": cpu_hash,
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "source_prefill_layer_devices": list(gpu_devices),
        "source_gpu_cache_structure": gpu_structure,
        "cpu_cache_structure": cpu_structure,
        "cache_total_bytes": cpu_bytes,
    }


def restage_source_cache_on_gpus(
    staged: Any,
    *,
    preparation: Mapping[str, Any],
    configuration: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Copy one CPU cache onto the five disjoint timed cache GPUs, outside timing."""

    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import build_dynamic_cache
    from kvcache_transfer.latency import synchronize_devices

    runtime = configuration["runtime"]
    length = int(preparation["prefix_length"])
    _cache_structure(staged, expected_length=length, expected_layer_devices=("cpu",) * 64)
    layer_ids = _expanded_layer_devices(
        runtime["source_cache_devices"],
        runtime["source_cache_layer_counts"],
        expected_layers=64,
    )
    layer_devices = tuple(f"cuda:{value}" for value in layer_ids)
    restored = build_dynamic_cache(
        staged,
        layer_devices=layer_devices,
        dtype=torch.bfloat16,
        clone=False,
        detach=True,
        non_blocking=False,
    )
    synchronize_devices(runtime["source_cache_devices"])
    structure, total = _cache_structure(
        restored, expected_length=length, expected_layer_devices=layer_devices
    )
    restored_hash = _cache_raw_sha256(restored)
    if restored_hash != preparation.get("cpu_staged_raw_bit_sha256"):
        raise RuntimeError("CPU-to-GPU source-cache restage changed raw bits")
    synchronize_devices(runtime["source_cache_devices"])
    return restored, {
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": length,
        "untimed": True,
        "source_model_live": False,
        "source_cache_gpu_resident_before_mapper_timer": True,
        "copy_semantics": "blocking_cpu_to_cuda_bf16_no_value_transform",
        "raw_bit_sha256": restored_hash,
        "raw_bit_hash_matches_cpu_staging": True,
        "layer_devices": list(layer_devices),
        "layer_counts": list(runtime["source_cache_layer_counts"]),
        "cache_structure": structure,
        "cache_total_bytes": total,
    }


def _cuda_allocator_snapshot(devices: Sequence[int]) -> dict[str, Any]:
    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.latency import synchronize_devices

    synchronize_devices(devices)
    result: dict[str, Any] = {}
    for device in devices:
        free, total = torch.cuda.mem_get_info(int(device))
        result[str(device)] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(int(device))),
            "reserved_bytes": int(torch.cuda.memory_reserved(int(device))),
            "driver_free_bytes": int(free),
            "driver_total_bytes": int(total),
        }
    return result


def _clear_cuda_allocators(devices: Sequence[int]) -> dict[str, Any]:
    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.latency import synchronize_devices

    synchronize_devices(devices)
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if not callable(clear):
        raise RuntimeError("frozen Torch lacks _cuda_clearCublasWorkspaces")
    clear()
    gc.collect()
    for device in devices:
        with torch.cuda.device(int(device)):
            torch.cuda.empty_cache()
    synchronize_devices(devices)
    return {
        "status": "passed",
        "devices": [int(value) for value in devices],
        "cublas_workspaces_cleared": True,
        "gc_completed": True,
        "empty_cache_each_device": True,
        "synchronized_before_and_after": True,
    }


def prove_source_model_released(
    *,
    model_ref: weakref.ReferenceType[Any],
    configuration: Mapping[str, Any],
    allocator_before: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    devices = configuration["runtime"]["source_prefill_devices"]
    all_devices = tuple(range(torch.cuda.device_count()))
    if all_devices != tuple(range(8)):
        raise RuntimeError("source release proof requires all eight visible GPUs")
    gc.collect()
    if model_ref() is not None:
        raise RuntimeError("32B source model remains referenced before target load")
    clear = _clear_cuda_allocators(all_devices)
    gc.collect()
    if model_ref() is not None:
        raise RuntimeError("32B source model survived allocator cleanup")
    after = _cuda_allocator_snapshot(all_devices)
    if any(
        row["allocated_bytes"] != 0 or row["reserved_bytes"] != 0
        for row in after.values()
    ):
        raise RuntimeError("source-prefill allocators are not empty before target load")
    return {
        "status": "passed",
        "source_model_live": False,
        "source_model_weakref_is_none": True,
        "all_source_gpu_cache_references_deleted": True,
        "target_model_loaded_at_proof_time": False,
        "source_prefill_devices": list(devices),
        "all_visible_devices": list(all_devices),
        "allocator_before_release": dict(allocator_before),
        "allocator_after_release_before_target_load": after,
        "allocator_after_release_covers_all_visible_devices": True,
        "all_prefill_allocated_and_reserved_bytes_zero": True,
        "all_visible_allocated_and_reserved_bytes_zero": True,
        "workspace_release": clear,
    }


def release_restaged_source_cache(
    *, cache_ref: weakref.ReferenceType[Any], configuration: Mapping[str, Any]
) -> dict[str, Any]:
    devices = configuration["runtime"]["source_cache_devices"]
    gc.collect()
    if cache_ref() is not None:
        raise RuntimeError("restaged source cache remains strongly referenced")
    clear = _clear_cuda_allocators(devices)
    after = _cuda_allocator_snapshot(devices)
    if any(
        row["allocated_bytes"] != 0 or row["reserved_bytes"] != 0
        for row in after.values()
    ):
        raise RuntimeError("source-cache staging GPUs are not allocator-empty after cell")
    return {
        "status": "passed",
        "restaged_cache_weakref_is_none": True,
        "source_cache_devices": list(devices),
        "allocated_and_reserved_bytes_zero": True,
        "workspace_release": clear,
        "allocator_after_release": after,
    }


def source_cache_evidence(
    cache: Any,
    expected_length: int,
    *,
    expected_layer_devices: Sequence[str] | None = None,
) -> dict[str, Any]:
    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import cache_sequence_length, cache_to_legacy

    pairs = cache_to_legacy(cache)
    if len(pairs) != 64 or cache_sequence_length(cache) != expected_length:
        raise RuntimeError("preexisting 32B source cache has the wrong shape")
    if expected_layer_devices is not None and len(expected_layer_devices) != 64:
        raise ValueError("source-cache evidence requires exactly 64 expected devices")
    rows = []
    total = 0
    for layer, (key, value) in enumerate(pairs):
        if tuple(key.shape) != (1, 8, expected_length, 128) or key.shape != value.shape:
            raise RuntimeError(f"source cache layer {layer} has an unexpected shape")
        if str(key.dtype) != "torch.bfloat16" or value.dtype != key.dtype:
            raise RuntimeError("source cache is not BF16")
        if expected_layer_devices is not None and (
            str(key.device) != expected_layer_devices[layer]
            or str(value.device) != expected_layer_devices[layer]
        ):
            raise RuntimeError(f"source cache layer {layer} is on the wrong staging GPU")
        nbytes = (key.numel() + value.numel()) * key.element_size()
        total += nbytes
        rows.append(
            {
                "layer": layer,
                "key_device": str(key.device),
                "value_device": str(value.device),
                "shape": list(key.shape),
                "dtype": str(key.dtype),
                "bytes": nbytes,
            }
        )
    return {
        "status": "passed",
        "source_model": "Qwen3-32B",
        "source_prefill_timed": False,
        "sequence_length": expected_length,
        "layers": len(rows),
        "total_tensor_bytes": total,
        "per_layer": rows,
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _frozen_topology() -> dict[str, Any]:
    return staged_runtime_topology(
        {
            "runtime": {
                "source_prefill_devices": list(SOURCE_PREFILL_DEVICES),
                "source_prefill_layer_counts": list(SOURCE_PREFILL_LAYER_COUNTS),
                "source_cache_devices": list(SOURCE_CACHE_DEVICES),
                "source_cache_layer_counts": list(SOURCE_CACHE_LAYER_COUNTS),
                "target_devices": list(TARGET_DEVICES),
                "target_layer_counts": list(TARGET_LAYER_COUNTS),
                "require_cuda_visible_devices": "0,1,2,3,4,5,6,7",
            }
        }
    )


def validate_latency_run_definition(path: Path) -> dict[str, Any]:
    """Verify the immutable request fingerprint and staged topology contract."""

    verify_hashed_json(path)
    value = read_json(path)
    fingerprint = value.get("request_fingerprint")
    if not _is_sha256(fingerprint):
        raise ValueError("latency run definition lacks a request fingerprint")
    unsigned = dict(value)
    unsigned.pop("request_fingerprint", None)
    if canonical_sha256(unsigned) != fingerprint:
        raise ValueError("latency run definition fingerprint changed")
    frozen = load_latency_config(DEFAULT_CONFIG)
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "qwen3_32b_to_14b_prefill_latency",
        "direction": "qwen3_32b_to_14b",
        "paper_reference": frozen["paper_reference"],
        "protocol": frozen["protocol"],
        "runtime": frozen["runtime"],
        "staged_runtime_topology": _frozen_topology(),
        "staged_runtime_topology_sha256": canonical_sha256(_frozen_topology()),
        "paper_h100_reference_only": frozen["paper_h100_reference_only"],
        "absolute_h100_bit_exact_claimed": False,
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"latency run definition {key} changed")
    if value.get("idle_gate_policy") != {
        "memory_limit_mib": 64,
        "required_consecutive_samples": 12,
        "sample_interval_seconds": 5,
        "timeout_seconds": 86400,
    }:
        raise ValueError("latency run definition idle-gate policy changed")
    expected_code = {
        "reverse_experiment": code_identity(CODE_FILES),
        "shared_read_only_runtime": code_identity(
            SHARED_CODE_FILES, root=REPOSITORY_ROOT
        ),
    }
    if value.get("code") != expected_code:
        raise ValueError("latency run definition code identity changed")
    hardware = value.get("hardware")
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("cuda_visible_devices") != "0,1,2,3,4,5,6,7"
        or not isinstance(hardware.get("devices"), list)
        or [row.get("logical_index") for row in hardware["devices"]] != list(range(8))
        or hardware.get("absolute_latency_requires_local_hardware_label") is not True
    ):
        raise ValueError("latency run definition hardware identity is malformed")
    lock = value.get("gpu_lock")
    if not isinstance(lock, Mapping) or lock != {
        "path": str(GLOBAL_GPU_LOCK),
        "scope": "repository_wide_forward_and_reverse_eight_gpu_experiments",
        "acquired_before_idle_gate": True,
        "held_through_final_manifest": True,
    }:
        raise ValueError("latency run definition GPU-lock contract changed")
    artifacts = value.get("runtime_artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "source_model",
        "target_model",
        "mapper",
    }:
        raise ValueError("latency run definition artifact inventory changed")
    source = artifacts["source_model"]
    target = artifacts["target_model"]
    mapper = artifacts["mapper"]
    if not all(isinstance(item, Mapping) for item in (source, target, mapper)):
        raise ValueError("latency run definition artifact identity is malformed")
    if (
        source.get("path") != str(frozen["paths"]["source_model"])
        or target.get("path") != str(frozen["paths"]["target_model"])
        or mapper.get("manifest") != str(frozen["paths"]["mapper_manifest"])
        or mapper.get("target_layer_count") != 40
    ):
        raise ValueError("latency run definition artifact endpoint changed")
    return {
        **value,
        "source_model_artifact_sha256": canonical_sha256(source),
        "target_model_artifact_sha256": canonical_sha256(target),
        "mapper_artifact_sha256": canonical_sha256(mapper),
    }


def _validate_prefix_identity(value: Any, *, length: int) -> dict[str, Any]:
    expected = {
        "shape": [1, length],
        "dtype": "torch.int64",
        "nbytes": 8 * length,
    }
    if not isinstance(value, Mapping) or any(value.get(key) != wanted for key, wanted in expected.items()):
        raise ValueError("staged latency prefix identity changed")
    if not _is_sha256(value.get("sha256")) or set(value) != {*expected, "sha256"}:
        raise ValueError("staged latency prefix identity is incomplete")
    return dict(value)


def _validate_cache_structure_rows(
    value: Any,
    *,
    length: int,
    layer_devices: Sequence[str],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 64 or len(layer_devices) != 64:
        raise ValueError(f"{label} does not cover 64 source layers")
    shape = [1, 8, length, 128]
    tensor_bytes = 1 * 8 * length * 128 * 2
    for layer, (row, device) in enumerate(zip(value, layer_devices, strict=True)):
        expected = {
            "layer": layer,
            "device": device,
            "key_shape": shape,
            "value_shape": shape,
            "key_dtype": "torch.bfloat16",
            "value_dtype": "torch.bfloat16",
            "key_nbytes": tensor_bytes,
            "value_nbytes": tensor_bytes,
        }
        if row != expected:
            raise ValueError(f"{label} layer {layer} changed")
    return [dict(row) for row in value]


def _validate_preparation(value: Any, *, length: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source-cache preparation is malformed")
    fixed = {
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": length,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "copy_semantics": "blocking_detach_to_cpu_bf16_exact_clone",
        "full_raw_bit_sha256_computed": True,
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "cache_total_bytes": source_cache_exact_bytes(length),
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"source-cache preparation {key} changed")
    elapsed = value.get("source_prefill_untimed_diagnostic_ms")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(float(elapsed)) or float(elapsed) <= 0:
        raise ValueError("source-cache untimed prefill diagnostic is invalid")
    hashes = (
        value.get("source_gpu_raw_bit_sha256"),
        value.get("cpu_staged_raw_bit_sha256"),
    )
    if not all(_is_sha256(item) for item in hashes) or len(set(hashes)) != 1:
        raise ValueError("source-cache GPU/CPU raw-bit hashes disagree")
    _validate_prefix_identity(value.get("deterministic_prefix"), length=length)
    prefill_ids = _expanded_layer_devices(
        SOURCE_PREFILL_DEVICES, SOURCE_PREFILL_LAYER_COUNTS, expected_layers=64
    )
    prefill_devices = tuple(f"cuda:{value}" for value in prefill_ids)
    if value.get("source_prefill_layer_devices") != list(prefill_devices):
        raise ValueError("source-cache preparation prefill devices changed")
    _validate_cache_structure_rows(
        value.get("source_gpu_cache_structure"),
        length=length,
        layer_devices=prefill_devices,
        label="source GPU cache structure",
    )
    _validate_cache_structure_rows(
        value.get("cpu_cache_structure"),
        length=length,
        layer_devices=("cpu",) * 64,
        label="CPU-staged cache structure",
    )
    return dict(value)


def _validate_zero_allocator_snapshot(
    value: Any, *, devices: Sequence[int], label: str
) -> dict[str, Any]:
    expected_keys = {str(value) for value in devices}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{label} device coverage changed")
    for device, row in value.items():
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} device {device} is malformed")
        if row.get("allocated_bytes") != 0 or row.get("reserved_bytes") != 0:
            raise ValueError(f"{label} device {device} was not allocator-empty")
        free = row.get("driver_free_bytes")
        total = row.get("driver_total_bytes")
        if any(type(item) is not int or item < 0 for item in (free, total)) or free > total:
            raise ValueError(f"{label} device {device} driver memory is invalid")
    return {str(key): dict(row) for key, row in value.items()}


def _validate_layer_sharding(
    value: Any,
    *,
    devices: Sequence[int],
    layer_counts: Sequence[int],
    total_layers: int,
    native_role: str,
    label: str,
) -> dict[str, Any]:
    expected_layer_devices = list(
        _expanded_layer_devices(devices, layer_counts, expected_layers=total_layers)
    )
    fixed = {
        "status": "passed",
        "role": native_role,
        "mode": "explicit_exact_counts",
        "devices": list(devices),
        "declared_layer_counts": list(layer_counts),
        "expected_layer_counts": list(layer_counts),
        "actual_layer_counts": list(layer_counts),
        "total_layers": total_layers,
        "layer_devices": expected_layer_devices,
        "actual_parameters_inspected_for_every_layer": True,
        "auxiliary_parameters_inspected_for_every_module": True,
    }
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} sharding proof is missing")
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"{label} sharding proof {key} changed")
    expected_ranges = []
    start = 0
    for device, count in zip(devices, layer_counts, strict=True):
        expected_ranges.append(
            {
                "device": int(device),
                "start_layer": start,
                "end_layer_exclusive": start + int(count),
                "count": int(count),
            }
        )
        start += int(count)
    if value.get("contiguous_ranges") != expected_ranges:
        raise ValueError(f"{label} contiguous sharding ranges changed")
    inspected = value.get("parameter_tensors_inspected")
    auxiliary = value.get("auxiliary_parameter_tensors_inspected")
    if (
        type(inspected) is not int
        or inspected <= 0
        or type(auxiliary) is not int
        or auxiliary <= 0
    ):
        raise ValueError(f"{label} sharding parameter inspection is invalid")
    return dict(value)


def _validate_model_instance_identity(
    value: Any, *, artifact: Mapping[str, Any], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "model_path",
        "model_config_sha256",
        "model_tokenizer_sha256",
        "model_weight_index_sha256",
    }:
        raise ValueError(f"{label} loaded-model identity is malformed")
    files = artifact.get("content_hashed_files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{label} artifact identity lacks hashed metadata")
    expected = {
        "model_path": artifact.get("path"),
        "model_config_sha256": files.get("config.json", {}).get("sha256"),
        "model_tokenizer_sha256": files.get("tokenizer.json", {}).get("sha256"),
        "model_weight_index_sha256": files.get(
            "model.safetensors.index.json", {}
        ).get("sha256"),
    }
    if dict(value) != expected or not all(
        _is_sha256(expected[key])
        for key in (
            "model_config_sha256",
            "model_tokenizer_sha256",
            "model_weight_index_sha256",
        )
    ):
        raise ValueError(f"{label} loaded model differs from its run artifact")
    return dict(value)


def verify_staging_session(
    path: Path,
    *,
    fingerprint: str,
    required_length: int | None = None,
    source_artifact: Mapping[str, Any] | None = None,
    target_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sidecar = verify_hashed_json(path)
    value = read_json(path)
    stem = path.name.removeprefix("source_cache_staging_").removesuffix(".json")
    if (
        path.parent.name != "staging_sessions"
        or path.name != f"source_cache_staging_{stem}.json"
        or len(stem) != 32
        or any(character not in "0123456789abcdef" for character in stem)
    ):
        raise ValueError("source-cache staging receipt path is invalid")
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "source_released_before_target_load",
        "request_fingerprint": fingerprint,
        "session_id": stem,
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "runtime_topology": _frozen_topology(),
        "runtime_topology_sha256": canonical_sha256(_frozen_topology()),
        "target_model_loaded_when_receipt_written": False,
        "source_model_live_when_receipt_written": False,
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"source-cache staging receipt {key} changed")
    if (
        type(value.get("source_vocabulary_size")) is not int
        or value["source_vocabulary_size"] <= 100
        or not _is_sha256(value.get("source_model_artifact_sha256"))
        or not _is_sha256(value.get("target_model_artifact_sha256"))
    ):
        raise ValueError("source-cache staging model identity fields are invalid")
    lengths = value.get("prepared_lengths")
    preparations = value.get("preparations")
    if (
        not isinstance(lengths, list)
        or not lengths
        or lengths != sorted(set(lengths))
        or any(length not in PAPER_LENGTHS for length in lengths)
        or not isinstance(preparations, Mapping)
        or set(preparations) != {str(length) for length in lengths}
    ):
        raise ValueError("source-cache staging receipt length coverage is invalid")
    if required_length is not None and required_length not in lengths:
        raise ValueError("source-cache staging receipt lacks the checkpoint length")
    validated = {
        str(length): _validate_preparation(preparations[str(length)], length=length)
        for length in lengths
    }
    expected_total = sum(source_cache_exact_bytes(length) for length in lengths)
    if value.get("cpu_staging_total_bytes") != expected_total:
        raise ValueError("source-cache staging receipt byte total changed")
    gate = value.get("host_memory_staging_gate")
    weight_fields = ("source_weight_shard_bytes", "target_weight_shard_bytes")
    if (
        not isinstance(gate, Mapping)
        or gate.get("status") != "passed"
        or gate.get("scope") != "before_source_model_load"
        or gate.get("staged_lengths") != lengths
        or gate.get("staged_cache_bytes") != expected_total
        or gate.get("mapper_parameter_bytes") != MAPPER_PARAMETER_BYTES
        or gate.get("conservative_overhead_bytes") != HOST_STAGING_OVERHEAD_BYTES
        or type(gate.get("mem_available_bytes")) is not int
        or type(gate.get("required_available_bytes")) is not int
        or any(type(gate.get(field)) is not int or gate[field] <= 0 for field in weight_fields)
        or gate["required_available_bytes"]
        != gate["source_weight_shard_bytes"]
        + gate["target_weight_shard_bytes"]
        + MAPPER_PARAMETER_BYTES
        + expected_total
        + HOST_STAGING_OVERHEAD_BYTES
        or gate["mem_available_bytes"] < gate["required_available_bytes"]
    ):
        raise ValueError("source-cache host-memory staging gate is invalid")
    if source_artifact is not None:
        if (
            value.get("source_model_artifact_sha256")
            != canonical_sha256(source_artifact)
            or gate["source_weight_shard_bytes"]
            != source_artifact.get("weight_shard_total_nbytes")
        ):
            raise ValueError("source-cache staging source artifact changed")
        _validate_model_instance_identity(
            value.get("source_model_instance_identity"),
            artifact=source_artifact,
            label="source",
        )
    if target_artifact is not None and (
        value.get("target_model_artifact_sha256")
        != canonical_sha256(target_artifact)
        or gate["target_weight_shard_bytes"]
        != target_artifact.get("weight_shard_total_nbytes")
    ):
        raise ValueError("source-cache staging target artifact changed")
    release = value.get("source_model_release")
    if not isinstance(release, Mapping):
        raise ValueError("source-model release receipt is absent")
    release_fixed = {
        "status": "passed",
        "source_model_live": False,
        "source_model_weakref_is_none": True,
        "all_source_gpu_cache_references_deleted": True,
        "target_model_loaded_at_proof_time": False,
        "source_prefill_devices": list(SOURCE_PREFILL_DEVICES),
        "all_visible_devices": list(range(8)),
        "allocator_after_release_covers_all_visible_devices": True,
        "all_prefill_allocated_and_reserved_bytes_zero": True,
        "all_visible_allocated_and_reserved_bytes_zero": True,
    }
    for key, wanted in release_fixed.items():
        if release.get(key) != wanted:
            raise ValueError(f"source-model release proof {key} changed")
    _validate_zero_allocator_snapshot(
        release.get("allocator_after_release_before_target_load"),
        devices=tuple(range(8)),
        label="source-model release",
    )
    _validate_layer_sharding(
        value.get("source_model_prefill_sharding"),
        devices=SOURCE_PREFILL_DEVICES,
        layer_counts=SOURCE_PREFILL_LAYER_COUNTS,
        total_layers=64,
        native_role="target",
        label="source-model prefill",
    )
    if not isinstance(value.get("source_model_instance_identity"), Mapping):
        raise ValueError("source-model instance identity is missing")
    return {
        **value,
        "preparations": validated,
        "artifact_sha256": sidecar["sha256"],
        "artifact_nbytes": sidecar["nbytes"],
    }


def _validate_runtime_setup(
    path: Path,
    *,
    fingerprint: str,
    source_artifact: Mapping[str, Any],
    target_artifact: Mapping[str, Any],
    mapper_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "ready",
        "request_fingerprint": fingerprint,
        "semantic_role_binding": {
            "source": {
                "model": "Qwen3-32B",
                "capture_native_forward_role": "target",
            },
            "target": {
                "model": "Qwen3-14B",
                "capture_native_forward_role": "source",
            },
        },
        "staged_runtime_topology": _frozen_topology(),
        "source_model_live_during_target_mapper_runtime": False,
        "resident_mapper_parameter_bytes": MAPPER_PARAMETER_BYTES,
        "mapper_resident_before_all_timed_cells": True,
        "math_contract": {
            "model_forward_dtype": "bfloat16",
            "mapper_dtype": "float32",
            "allow_tf32_cuda_matmul": False,
            "allow_tf32_cudnn": False,
            "float32_matmul_precision": "highest",
            "mapper_execution": "eager",
            "torch_compile": False,
            "cuda_graph": False,
        },
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"staged latency runtime setup {key} changed")
    source_sharding = _validate_layer_sharding(
        value.get("source_sharding"),
        devices=SOURCE_PREFILL_DEVICES,
        layer_counts=SOURCE_PREFILL_LAYER_COUNTS,
        total_layers=64,
        native_role="target",
        label="runtime source",
    )
    _validate_layer_sharding(
        value.get("target_sharding"),
        devices=TARGET_DEVICES,
        layer_counts=TARGET_LAYER_COUNTS,
        total_layers=40,
        native_role="source",
        label="runtime target",
    )
    binding = value.get("staging_session")
    if not isinstance(binding, Mapping) or set(binding) != {
        "session_receipt",
        "session_receipt_sha256",
        "prepared_lengths",
    }:
        raise ValueError("staged latency runtime session binding changed")
    relative = binding.get("session_receipt")
    if not isinstance(relative, str):
        raise ValueError("staged latency runtime session path is malformed")
    session_path = (path.parent / relative).resolve()
    try:
        safe = session_path.relative_to(path.parent.resolve())
    except ValueError as exc:
        raise ValueError("staged latency runtime session escaped the run") from exc
    if len(safe.parts) != 2 or safe.parts[0] != "staging_sessions":
        raise ValueError("staged latency runtime session directory changed")
    session = verify_staging_session(
        session_path,
        fingerprint=fingerprint,
        source_artifact=source_artifact,
        target_artifact=target_artifact,
    )
    if (
        binding.get("session_receipt_sha256") != session["artifact_sha256"]
        or binding.get("prepared_lengths") != session["prepared_lengths"]
        or value.get("source_model_release") != session["source_model_release"]
        or source_sharding != session["source_model_prefill_sharding"]
    ):
        raise ValueError("staged latency runtime session evidence drifted")
    provenance = value.get("mapper_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("staged latency mapper provenance is absent")
    mapper_fixed = {
        "direction": "qwen3_32b_to_14b",
        "manifest": mapper_artifact.get("manifest"),
        "manifest_sha256": mapper_artifact.get("manifest_sha256"),
        "fit_fingerprint": mapper_artifact.get("fit_fingerprint"),
        "layers": 40,
        "selected_source_layers": 20,
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
        "resident_parameter_bytes": MAPPER_PARAMETER_BYTES,
        "source_model_live": False,
    }
    for key, wanted in mapper_fixed.items():
        if provenance.get(key) != wanted:
            raise ValueError(f"staged latency mapper provenance {key} changed")
    target_layer_ids = _expanded_layer_devices(
        TARGET_DEVICES, TARGET_LAYER_COUNTS, expected_layers=40
    )
    target_layer_devices = [f"cuda:{item}" for item in target_layer_ids]
    if provenance.get("target_layer_devices") != target_layer_devices:
        raise ValueError("staged latency mapper target-layer devices changed")
    _validate_model_instance_identity(
        provenance.get("source_model_artifact"),
        artifact=source_artifact,
        label="runtime source",
    )
    _validate_model_instance_identity(
        provenance.get("target_model_artifact"),
        artifact=target_artifact,
        label="runtime target",
    )
    setup = value.get("mapper_setup")
    if (
        not isinstance(setup, Mapping)
        or provenance.get("materialization") != setup
        or setup.get("scope") != "one_time_setup_excluded_from_per_request_latency"
        or not isinstance(setup.get("per_layer"), list)
        or len(setup["per_layer"]) != 40
    ):
        raise ValueError("staged latency mapper materialization proof changed")
    for layer, (row, device) in enumerate(
        zip(setup["per_layer"], target_layer_devices, strict=True)
    ):
        if (
            not isinstance(row, Mapping)
            or row.get("target_layer") != layer
            or row.get("device") != device
            or any(
                isinstance(row.get(field), bool)
                or not isinstance(row.get(field), (int, float))
                or not math.isfinite(float(row[field]))
                or float(row[field]) < 0.0
                for field in ("host_load_and_hash_ms", "weight_host_to_device_ms")
            )
        ):
            raise ValueError(f"staged latency mapper setup layer {layer} changed")
    memory = value.get("memory_after_staged_runtime_load")
    if not isinstance(memory, Mapping):
        raise ValueError("staged latency runtime memory evidence is absent")
    torch_rows = memory.get("torch")
    if not isinstance(torch_rows, Mapping) or set(torch_rows) != {
        str(value) for value in range(8)
    }:
        raise ValueError("staged latency runtime allocator evidence changed")
    for device in SOURCE_CACHE_DEVICES:
        row = torch_rows[str(device)]
        if (
            not isinstance(row, Mapping)
            or row.get("allocated_bytes") != 0
            or row.get("reserved_bytes") != 0
        ):
            raise ValueError(
                "source-cache restage device was not empty before timed runtime"
            )
    return dict(value)


def _resolve_session_from_lifecycle(
    *, output: Path, lifecycle: Mapping[str, Any], fingerprint: str, length: int
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    relative = lifecycle.get("session_receipt")
    if not isinstance(relative, str):
        raise ValueError("source-cache lifecycle lacks a staging receipt")
    session_path = (output / relative).resolve()
    try:
        safe = session_path.relative_to(output.resolve())
    except ValueError as exc:
        raise ValueError("source-cache staging receipt escaped the run") from exc
    if len(safe.parts) != 2 or safe.parts[0] != "staging_sessions":
        raise ValueError("source-cache staging receipt is outside its sealed directory")
    session = verify_staging_session(
        session_path, fingerprint=fingerprint, required_length=length
    )
    if lifecycle.get("session_receipt_sha256") != session["artifact_sha256"]:
        raise ValueError("source-cache lifecycle staging receipt SHA changed")
    preparation = session["preparations"][str(length)]
    if lifecycle.get("preparation_sha256") != canonical_sha256(preparation):
        raise ValueError("source-cache lifecycle preparation digest changed")
    return session_path, session, preparation


def _all_staging_session_paths(
    output: Path, *, referenced: Sequence[str]
) -> list[Path]:
    directory = output / "staging_sessions"
    payloads = (
        {
            path.resolve()
            for path in directory.glob("source_cache_staging_*.json")
            if not path.name.endswith(".sha256.json")
        }
        if directory.is_dir()
        else set()
    )
    sidecar_payloads = (
        {
            path.with_name(path.name.removesuffix(".sha256.json")).resolve()
            for path in directory.glob("source_cache_staging_*.json.sha256.json")
        }
        if directory.is_dir()
        else set()
    )
    if payloads != sidecar_payloads:
        raise ValueError("source-cache staging session is only partially committed")
    resolved_references = {(output / value).resolve() for value in referenced}
    if not resolved_references <= payloads:
        raise ValueError("source-cache lifecycle references an absent staging session")
    return sorted(payloads)


def _validate_source_cache_evidence_payload(
    value: Any, *, length: int, layer_devices: Sequence[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("preexisting source-cache evidence is absent")
    fixed = {
        "status": "passed",
        "source_model": "Qwen3-32B",
        "source_prefill_timed": False,
        "sequence_length": length,
        "layers": 64,
        "total_tensor_bytes": source_cache_exact_bytes(length),
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"preexisting source-cache evidence {key} changed")
    rows = value.get("per_layer")
    if not isinstance(rows, list) or len(rows) != 64:
        raise ValueError("preexisting source-cache evidence lacks 64 layers")
    shape = [1, 8, length, 128]
    layer_bytes = 2 * 8 * length * 128 * 2
    for layer, (row, device) in enumerate(zip(rows, layer_devices, strict=True)):
        if row != {
            "layer": layer,
            "key_device": device,
            "value_device": device,
            "shape": shape,
            "dtype": "torch.bfloat16",
            "bytes": layer_bytes,
        }:
            raise ValueError(f"preexisting source-cache evidence layer {layer} changed")
    return dict(value)


def validate_source_cache_lifecycle(
    value: Any, *, output: Path, fingerprint: str, length: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("latency checkpoint lacks staged source-cache lifecycle")
    _, _, preparation = _resolve_session_from_lifecycle(
        output=output, lifecycle=value, fingerprint=fingerprint, length=length
    )
    fixed = {
        "status": "complete",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "source_model_live_during_timed_cells": False,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "mapper_timer_started": True,
        "cpu_staged_raw_bit_sha256": preparation["cpu_staged_raw_bit_sha256"],
        "cache_total_bytes": source_cache_exact_bytes(length),
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"source-cache lifecycle {key} changed")
    if value.get("deterministic_prefix") != preparation["deterministic_prefix"]:
        raise ValueError("source-cache lifecycle deterministic prefix changed")
    restage = value.get("restage")
    if not isinstance(restage, Mapping):
        raise ValueError("source-cache lifecycle lacks restage proof")
    restage_fixed = {
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": length,
        "untimed": True,
        "source_model_live": False,
        "source_cache_gpu_resident_before_mapper_timer": True,
        "copy_semantics": "blocking_cpu_to_cuda_bf16_no_value_transform",
        "raw_bit_sha256": preparation["cpu_staged_raw_bit_sha256"],
        "raw_bit_hash_matches_cpu_staging": True,
        "layer_counts": list(SOURCE_CACHE_LAYER_COUNTS),
        "cache_total_bytes": source_cache_exact_bytes(length),
    }
    for key, wanted in restage_fixed.items():
        if restage.get(key) != wanted:
            raise ValueError(f"source-cache restage proof {key} changed")
    cache_ids = _expanded_layer_devices(
        SOURCE_CACHE_DEVICES, SOURCE_CACHE_LAYER_COUNTS, expected_layers=64
    )
    cache_devices = tuple(f"cuda:{value}" for value in cache_ids)
    if restage.get("layer_devices") != list(cache_devices):
        raise ValueError("source-cache restage layer devices changed")
    _validate_cache_structure_rows(
        restage.get("cache_structure"),
        length=length,
        layer_devices=cache_devices,
        label="GPU-restaged cache structure",
    )
    release = value.get("restaged_cache_release")
    if (
        not isinstance(release, Mapping)
        or release.get("status") != "passed"
        or release.get("restaged_cache_weakref_is_none") is not True
        or release.get("source_cache_devices") != list(SOURCE_CACHE_DEVICES)
        or release.get("allocated_and_reserved_bytes_zero") is not True
    ):
        raise ValueError("restaged source-cache release proof changed")
    _validate_zero_allocator_snapshot(
        release.get("allocator_after_release"),
        devices=SOURCE_CACHE_DEVICES,
        label="restaged source-cache release",
    )
    return dict(value)


def _minimum_driver_headroom_mib(snapshot: Mapping[str, Any]) -> int:
    rows = snapshot.get("nvidia_smi_memory")
    if not isinstance(rows, Sequence) or len(rows) != 8:
        raise RuntimeError("memory smoke did not observe all eight GPUs")
    if [row.get("physical_index") for row in rows if isinstance(row, Mapping)] != list(
        range(8)
    ):
        raise RuntimeError("memory smoke GPU inventory is not physical devices 0..7")
    headroom = [
        int(row["memory_total_mib"]) - int(row["memory_used_mib"])
        for row in rows
        if isinstance(row, Mapping)
    ]
    if len(headroom) != 8:
        raise RuntimeError("memory smoke has malformed nvidia-smi rows")
    return min(headroom)


def endpoint_memory_smoke(
    *,
    target: Any,
    resident: Any,
    source_config: Any,
    staged_cache: Any,
    preparation: Mapping[str, Any],
    session_binding: Mapping[str, Any],
    configuration: Mapping[str, Any],
    fingerprint: str,
) -> dict[str, Any]:
    """Run the staged, untimed 32K endpoint gate before the full sweep."""

    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import cache_sequence_length, qwen3_layer_devices
    from kvcache_transfer.latency import synchronize_devices
    from kvcache_transfer.transfer import transfer_source_cache

    protocol = configuration["protocol"]
    length = int(protocol["endpoint_memory_smoke_length"])
    minimum = int(protocol["minimum_post_operation_headroom_mib"])
    if length != max(PAPER_LENGTHS) or minimum != 2048:
        raise ValueError("endpoint memory smoke contract changed")
    prefix = deterministic_prefix(length, int(source_config.vocab_size), int(protocol["seed"]))
    identity = prefix_identity(prefix)
    if identity != preparation.get("deterministic_prefix"):
        raise RuntimeError("32K endpoint prefix differs from staged source cache")
    source_cache, restage = restage_source_cache_on_gpus(
        staged_cache, preparation=preparation, configuration=configuration
    )
    source_cache_ref = weakref.ref(source_cache)
    cache_ids = _expanded_layer_devices(
        SOURCE_CACHE_DEVICES, SOURCE_CACHE_LAYER_COUNTS, expected_layers=64
    )
    cache_devices = tuple(f"cuda:{value}" for value in cache_ids)
    cache_evidence = source_cache_evidence(
        source_cache, length, expected_layer_devices=cache_devices
    )

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    target_base = getattr(target, "model", target)
    target_device = target_base.embed_tokens.weight.device
    with torch.inference_mode():
        target_output = target_base(
            input_ids=prefix.to(target_device), use_cache=True, return_dict=True
        )
    synchronize_devices(qwen3_layer_devices(target))
    if cache_sequence_length(target_output.past_key_values) != length:
        raise RuntimeError("32K target re-prefill smoke produced the wrong cache length")
    target_snapshot = cuda_memory_snapshot("endpoint_32k_target_reprefill_output_live")
    target_headroom = _minimum_driver_headroom_mib(target_snapshot)
    del target_output
    for device in configuration["runtime"]["target_devices"]:
        with torch.cuda.device(int(device)):
            torch.cuda.empty_cache()

    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    with torch.inference_mode():
        mapped_cache = transfer_source_cache(
            source_cache,
            source_config=source_config,
            target_model=target,
            mapper_bundle=resident,
            token_chunk=int(protocol["mapper_token_chunk"]),
            strict_validation=True,
        )
    synchronize_devices(qwen3_layer_devices(target))
    if cache_sequence_length(mapped_cache) != length or len(mapped_cache) != 40:
        raise RuntimeError("32K mapper smoke produced an invalid target cache")
    mapper_snapshot = cuda_memory_snapshot("endpoint_32k_mapped_cache_output_live")
    mapper_headroom = _minimum_driver_headroom_mib(mapper_snapshot)
    del mapped_cache, source_cache, prefix
    release = release_restaged_source_cache(
        cache_ref=source_cache_ref, configuration=configuration
    )
    for device in configuration["runtime"]["target_devices"]:
        with torch.cuda.device(int(device)):
            torch.cuda.empty_cache()
    cleanup = cuda_memory_snapshot("endpoint_32k_smoke_after_cleanup")
    observed_minimum = min(target_headroom, mapper_headroom)
    if observed_minimum < minimum:
        raise RuntimeError(
            "32K endpoint memory smoke has insufficient driver headroom: "
            f"{observed_minimum} MiB < {minimum} MiB"
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "passed",
        "request_fingerprint": fingerprint,
        "timed": False,
        "source_prefill_timed": False,
        "source_model_live": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "prefix_length": length,
        "minimum_required_headroom_mib": minimum,
        "minimum_observed_headroom_mib": observed_minimum,
        "target_reprefill_headroom_mib": target_headroom,
        "mapper_headroom_mib": mapper_headroom,
        "source_cache": cache_evidence,
        "source_cache_lifecycle": {
            "status": "complete",
            "policy": SOURCE_CACHE_STAGING_POLICY,
            **dict(session_binding),
            "preparation_sha256": canonical_sha256(preparation),
            "source_model_live_during_timed_cells": False,
            "source_prefill_timed": False,
            "cpu_staging_timed": False,
            "gpu_restage_timed": False,
            "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
            "mapper_timer_started": True,
            "deterministic_prefix": identity,
            "cpu_staged_raw_bit_sha256": preparation["cpu_staged_raw_bit_sha256"],
            "cache_total_bytes": source_cache_exact_bytes(length),
            "restage": restage,
            "restaged_cache_release": release,
        },
        "target_reprefill_output_live": target_snapshot,
        "mapped_cache_output_live": mapper_snapshot,
        "after_cleanup": cleanup,
    }


def verify_endpoint_memory_smoke(path: Path, *, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file() or not path.with_name(f"{path.name}.sha256.json").is_file():
        return None
    verify_hashed_json(path)
    value = read_json(path)
    length = max(PAPER_LENGTHS)
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "passed",
        "request_fingerprint": fingerprint,
        "timed": False,
        "source_prefill_timed": False,
        "source_model_live": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "prefix_length": length,
        "minimum_required_headroom_mib": 2048,
    }
    for key, wanted in fixed.items():
        if value.get(key) != wanted:
            raise ValueError(f"endpoint memory smoke {key} changed")
    lifecycle = validate_source_cache_lifecycle(
        value.get("source_cache_lifecycle"),
        output=path.parent,
        fingerprint=fingerprint,
        length=length,
    )
    cache_ids = _expanded_layer_devices(
        SOURCE_CACHE_DEVICES, SOURCE_CACHE_LAYER_COUNTS, expected_layers=64
    )
    _validate_source_cache_evidence_payload(
        value.get("source_cache"),
        length=length,
        layer_devices=tuple(f"cuda:{item}" for item in cache_ids),
    )
    target_headroom = _minimum_driver_headroom_mib(
        value.get("target_reprefill_output_live", {})
    )
    mapper_headroom = _minimum_driver_headroom_mib(
        value.get("mapped_cache_output_live", {})
    )
    observed = min(target_headroom, mapper_headroom)
    if (
        value.get("target_reprefill_headroom_mib") != target_headroom
        or value.get("mapper_headroom_mib") != mapper_headroom
        or value.get("minimum_observed_headroom_mib") != observed
        or observed < 2048
    ):
        raise ValueError("endpoint memory smoke headroom evidence changed")
    return {**value, "source_cache_lifecycle": lifecycle}


def _checkpoint(output: Path, length: int) -> Path:
    return output / "lengths" / f"length_{length:05d}.json"


def verify_length_checkpoint(
    path: Path, *, fingerprint: str, length: int
) -> dict[str, Any] | None:
    if not path.is_file() or not path.with_name(f"{path.name}.sha256.json").is_file():
        return None
    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "prefix_length": length,
        "batch_size": 1,
        "source_prefill_timed": False,
        "source_cache_existed_before_each_timed_metric": True,
        "source_model_reference_passed_to_timed_mapper": False,
        "source_model_live_during_timed_cells": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "timed_mapper_strict_validation": False,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise ValueError(f"latency checkpoint invariant changed: {path} ({key})")
    for metric in ("target_reprefill", "mapper_handoff_end_to_end"):
        result = value.get(metric)
        if not isinstance(result, Mapping):
            raise ValueError(f"latency checkpoint lacks {metric}")
        if result.get("warmup") != WARMUP or result.get("timed_repeats") != REPEATS:
            raise ValueError(f"latency checkpoint {metric} trial count changed")
        _validate_latency_summary(result, label=metric)
    if value["target_reprefill"].get("lm_head_included") is not False:
        raise ValueError("target re-prefill accidentally includes the LM head")
    target = value["target_reprefill"]
    required_target_flags = {
        "measurement_scope": "target_transformer_reprefill",
        "kv_cache_output": True,
        "input_length": length,
        "batch_size": 1,
        "inference_mode_enabled": True,
        "autograd_graph_disabled": True,
        "synchronize_each_iteration": True,
        "timing_backends": ["host_perf_counter_with_multi_cuda_synchronize"],
    }
    for key, expected in required_target_flags.items():
        if target.get(key) != expected:
            raise ValueError(f"target re-prefill timing scope {key} changed")
    mapper = value["mapper_handoff_end_to_end"]
    required_mapper_flags = {
        "measurement_scope": "source_cache_to_mapped_target_cache",
        "source_prefill_included": False,
        "mapper_weights_resident_on_target_devices": True,
        "includes_source_cache_device_handoff": True,
        "includes_inverse_and_target_rope": True,
        "includes_cache_build": True,
        "inference_mode_enabled": True,
        "autograd_graph_disabled": True,
        "synchronize_each_iteration": True,
        "timing_backends": ["host_perf_counter_with_multi_cuda_synchronize"],
    }
    for key, expected in required_mapper_flags.items():
        if mapper.get(key) != expected:
            raise ValueError(f"mapper timing scope {key} changed")
    cache_ids = _expanded_layer_devices(
        SOURCE_CACHE_DEVICES, SOURCE_CACHE_LAYER_COUNTS, expected_layers=64
    )
    _validate_source_cache_evidence_payload(
        value.get("source_cache"),
        length=length,
        layer_devices=tuple(f"cuda:{item}" for item in cache_ids),
    )
    validate_source_cache_lifecycle(
        value.get("source_cache_lifecycle"),
        output=path.parent.parent,
        fingerprint=fingerprint,
        length=length,
    )
    preflight = value.get("untimed_strict_mapper_preflight")
    if not isinstance(preflight, Mapping) or {
        "status": preflight.get("status"),
        "target_cache_layers": preflight.get("target_cache_layers"),
        "target_cache_sequence_length": preflight.get("target_cache_sequence_length"),
        "strict_validation": preflight.get("strict_validation"),
    } != {
        "status": "passed",
        "target_cache_layers": 40,
        "target_cache_sequence_length": length,
        "strict_validation": True,
    }:
        raise ValueError("untimed strict mapper preflight evidence changed")
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


def _validate_latency_summary(value: Mapping[str, Any], *, label: str) -> None:
    samples_raw = value.get("samples")
    if not isinstance(samples_raw, list) or len(samples_raw) != REPEATS:
        raise ValueError(f"latency checkpoint {label} raw samples are incomplete")
    samples: list[float] = []
    for item in samples_raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"latency checkpoint {label} has a non-numeric sample")
        sample = float(item)
        if not math.isfinite(sample) or sample <= 0.0:
            raise ValueError(f"latency checkpoint {label} samples must be finite and positive")
        samples.append(sample)
    expected: dict[str, float | int | str] = {
        "unit": "ms",
        "count": REPEATS,
        "mean": float(statistics.fmean(samples)),
        "median": float(statistics.median(samples)),
        "p10": float(_percentile(samples, 0.10)),
        "p50": float(_percentile(samples, 0.50)),
        "p90": float(_percentile(samples, 0.90)),
        "std_population": float(statistics.pstdev(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
    }
    for field, wanted in expected.items():
        observed = value.get(field)
        if isinstance(wanted, str) or isinstance(wanted, int):
            if observed != wanted:
                raise ValueError(f"latency checkpoint {label} {field} changed")
            continue
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ValueError(f"latency checkpoint {label} {field} is not numeric")
        if not math.isclose(float(observed), wanted, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                f"latency checkpoint {label} {field} disagrees with raw samples"
            )


def summarize_checkpoints(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(int(value["prefix_length"]) for value in checkpoints) != PAPER_LENGTHS:
        raise ValueError("latency checkpoints do not cover the exact ten paper lengths")
    rows = []
    for value in checkpoints:
        _validate_latency_summary(value["target_reprefill"], label="target_reprefill")
        _validate_latency_summary(
            value["mapper_handoff_end_to_end"], label="mapper_handoff_end_to_end"
        )
        reprefill = float(value["target_reprefill"]["median"])
        mapper = float(value["mapper_handoff_end_to_end"]["median"])
        if not all(math.isfinite(item) and item > 0 for item in (reprefill, mapper)):
            raise ValueError("latency checkpoint has a non-positive median")
        rows.append(
            {
                "prefix_length": int(value["prefix_length"]),
                "target_reprefill_median_ms": reprefill,
                "mapper_median_ms": mapper,
                "speedup_reprefill_over_mapper": reprefill / mapper,
                "target_reprefill_raw_samples_ms": value["target_reprefill"]["samples"],
                "mapper_raw_samples_ms": value["mapper_handoff_end_to_end"]["samples"],
            }
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "lengths": list(PAPER_LENGTHS),
        "warmup": WARMUP,
        "timed_repeats": REPEATS,
        "source_prefill_timed": False,
        "headline_speedup": "target_transformer_reprefill_median / mapper_handoff_median",
        "rows": rows,
    }


def verify_latency_completion(output: Path) -> dict[str, Any] | None:
    """Recompute the complete staged-latency receipt from committed raw artifacts."""

    manifest_path = output / "manifest.json"
    receipt_path = output / "formal_latency_receipt.json"
    present = (
        manifest_path.is_file(),
        manifest_path.with_name(f"{manifest_path.name}.sha256.json").is_file(),
        receipt_path.is_file(),
        receipt_path.with_name(f"{receipt_path.name}.sha256.json").is_file(),
    )
    if not any(present):
        return None
    if not all(present):
        raise ValueError("formal latency completion is only partially committed")
    manifest_sidecar = verify_hashed_json(manifest_path)
    verify_hashed_json(receipt_path)
    definition = validate_latency_run_definition(output / "run_definition.json")
    fingerprint = definition["request_fingerprint"]
    artifacts = definition["runtime_artifacts"]
    manifest = read_json(manifest_path)
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "direction": "qwen3_32b_to_14b",
        "lengths": list(PAPER_LENGTHS),
        "warmup": WARMUP,
        "timed_repeats": REPEATS,
        "source_prefill_timed": False,
        "source_model_live_during_timed_cells": False,
        "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
        "cpu_staging_and_gpu_restage_timed": False,
        "mapper_timer_start": "preexisting_gpu_resident_source_kv",
        "staged_runtime_topology": _frozen_topology(),
        "staged_runtime_topology_sha256": canonical_sha256(_frozen_topology()),
    }
    for key, wanted in fixed.items():
        if manifest.get(key) != wanted:
            raise ValueError(f"formal latency manifest {key} changed")
    checkpoints = []
    checkpoint_inventory = []
    for length in PAPER_LENGTHS:
        path = _checkpoint(output, length)
        checkpoint = verify_length_checkpoint(path, fingerprint=fingerprint, length=length)
        if checkpoint is None:
            raise ValueError(f"formal latency checkpoint is missing: {length}")
        checkpoints.append(checkpoint)
        checkpoint_inventory.append(
            {
                "prefix_length": length,
                "path": str(path.relative_to(output)),
                "sha256": sha256_file(path),
                "nbytes": path.stat().st_size,
            }
        )
    if (
        manifest.get("checkpoints") != checkpoint_inventory
        or manifest.get("checkpoint_manifest_sha256")
        != canonical_sha256(checkpoint_inventory)
    ):
        raise ValueError("formal latency checkpoint inventory changed")
    verify_hashed_json(output / "summary.json")
    expected_summary = summarize_checkpoints(checkpoints)
    if read_json(output / "summary.json") != expected_summary:
        raise ValueError("formal latency summary differs from raw samples")
    smoke = verify_endpoint_memory_smoke(
        output / "endpoint_memory_smoke.json", fingerprint=fingerprint
    )
    if smoke is None:
        raise ValueError("formal latency endpoint memory smoke is missing")
    referenced_sessions = {
        str(row["source_cache_lifecycle"]["session_receipt"])
        for row in checkpoints
    }
    referenced_sessions.add(
        str(smoke["source_cache_lifecycle"]["session_receipt"])
    )
    staging_receipts = []
    for path in _all_staging_session_paths(
        output, referenced=sorted(referenced_sessions)
    ):
        session = verify_staging_session(path, fingerprint=fingerprint)
        if (
            session.get("source_model_artifact_sha256")
            != definition["source_model_artifact_sha256"]
            or session.get("target_model_artifact_sha256")
            != definition["target_model_artifact_sha256"]
        ):
            raise ValueError("formal latency staging session artifact binding changed")
        verify_staging_session(
            path,
            fingerprint=fingerprint,
            source_artifact=artifacts["source_model"],
            target_artifact=artifacts["target_model"],
        )
        staging_receipts.append(
            {
                "session_id": session["session_id"],
                "path": str(path.relative_to(output)),
                "sha256": session["artifact_sha256"],
                "nbytes": session["artifact_nbytes"],
                "prepared_lengths": session["prepared_lengths"],
            }
        )
    if (
        manifest.get("staging_receipts") != staging_receipts
        or manifest.get("staging_receipts_sha256") != canonical_sha256(staging_receipts)
    ):
        raise ValueError("formal latency staging-receipt inventory changed")
    summary_sidecar = verify_hashed_json(output / "summary.json")
    if manifest.get("summary") != summary_sidecar:
        raise ValueError("formal latency summary manifest binding changed")
    for name, field in (
        ("gpu_idle_gate.json", "gpu_idle_gate"),
        ("hardware.json", "hardware"),
        ("runtime_setup.json", "runtime_setup"),
    ):
        sidecar = verify_hashed_json(output / name)
        if manifest.get(field) != {
            "sha256": sidecar["sha256"],
            "nbytes": sidecar["nbytes"],
        }:
            raise ValueError(f"formal latency {field} manifest binding changed")
    _validate_gpu_idle_gate(output / "gpu_idle_gate.json")
    if read_json(output / "hardware.json") != definition["hardware"]:
        raise ValueError("formal latency hardware artifact differs from run definition")
    _validate_runtime_setup(
        output / "runtime_setup.json",
        fingerprint=fingerprint,
        source_artifact=artifacts["source_model"],
        target_artifact=artifacts["target_model"],
        mapper_artifact=artifacts["mapper"],
    )
    endpoint_entry = manifest.get("endpoint_memory_smoke")
    if not isinstance(endpoint_entry, Mapping) or endpoint_entry != {
        "sha256": sha256_file(output / "endpoint_memory_smoke.json"),
        "nbytes": (output / "endpoint_memory_smoke.json").stat().st_size,
        "minimum_required_headroom_mib": 2048,
        "minimum_observed_headroom_mib": smoke["minimum_observed_headroom_mib"],
    }:
        raise ValueError("formal latency endpoint manifest binding changed")
    receipt = read_json(receipt_path)
    expected_receipt = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "request_fingerprint": fingerprint,
        "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
        "staged_runtime_topology_sha256": canonical_sha256(_frozen_topology()),
        "manifest": manifest_sidecar,
        "summary_sha256": sha256_file(output / "summary.json"),
        "endpoint_memory_smoke_sha256": sha256_file(
            output / "endpoint_memory_smoke.json"
        ),
        "checkpoint_manifest_sha256": canonical_sha256(checkpoint_inventory),
        "staging_receipts_sha256": canonical_sha256(staging_receipts),
        "all_ten_lengths_complete": True,
        "source_model_release_proven_for_every_session": True,
        "mapper_timers_started_from_gpu_resident_source_kv": True,
    }
    if receipt != expected_receipt:
        raise ValueError("formal latency completion receipt changed")
    return manifest


def prepare_only(configuration: Mapping[str, Any], output: Path) -> dict[str, Any]:
    missing = []
    for key in ("source_model", "target_model"):
        if not Path(configuration["paths"][key]).is_dir():
            missing.append(key)
    mapper = Path(configuration["paths"]["mapper_manifest"])
    if not mapper.is_file():
        missing.append("mapper_manifest")
    plan: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "awaiting_artifacts" if missing else "ready_to_launch_after_gpu_idle_gate",
        "missing": missing,
        "protocol": configuration["protocol"],
        "runtime": configuration["runtime"],
        "paths": {key: str(value) for key, value in configuration["paths"].items()},
        "launch_environment": {
            "CUDA_VISIBLE_DEVICES": configuration["runtime"]["require_cuda_visible_devices"]
        },
        "gpu_started": False,
    }
    if not missing:
        plan["artifacts"] = {
            "source_model": model_artifact_identity(configuration["paths"]["source_model"]),
            "target_model": model_artifact_identity(configuration["paths"]["target_model"]),
            "mapper": mapper_manifest_identity(configuration["paths"]["mapper_manifest"]),
        }
    hashed_json(output / "latency_prepare_plan.json", plan)
    return plan


def _default_output(prepare: bool) -> Path:
    if prepare:
        return EXPERIMENT_ROOT / "artifacts/inputs/reverse_latency_prepare_v1"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return EXPERIMENT_ROOT / "runs" / f"{stamp}_reverse_latency_k20"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def acquire_gpu_lock(*, blocking: bool = False) -> BinaryIO:
    # This is intentionally the same repository-wide lock used by the forward
    # experiments.  It is acquired before the idle gate and held through the
    # final manifest, closing the managed-job idle-check/model-load TOCTOU race.
    path = GLOBAL_GPU_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(handle.fileno(), operation)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("another repository-managed job owns the eight-GPU lock") from exc
    return handle


def _prefill_and_stage_caches(
    source: Any,
    *,
    lengths: Sequence[int],
    configuration: Mapping[str, Any],
) -> tuple[dict[int, Any], dict[str, dict[str, Any]]]:
    import torch

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.latency import synchronize_devices

    runtime = configuration["runtime"]
    protocol = configuration["protocol"]
    staged: dict[int, Any] = {}
    preparations: dict[str, dict[str, Any]] = {}
    for raw_length in lengths:
        length = int(raw_length)
        prefix = deterministic_prefix(
            length, int(source.config.vocab_size), int(protocol["seed"])
        )
        identity = prefix_identity(prefix)
        base = getattr(source, "model", source)
        device = base.embed_tokens.weight.device
        started = time.perf_counter_ns()
        with torch.inference_mode():
            output = base(
                input_ids=prefix.to(device), use_cache=True, return_dict=True
            )
        gpu_cache = output.past_key_values
        del output, base
        synchronize_devices(runtime["source_prefill_devices"])
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        cpu_cache, preparation = stage_source_cache_on_cpu(
            gpu_cache,
            length=length,
            identity=identity,
            configuration=configuration,
        )
        preparation["source_prefill_untimed_diagnostic_ms"] = elapsed_ms
        gpu_ref = weakref.ref(gpu_cache)
        del gpu_cache, prefix
        gc.collect()
        for gpu in runtime["source_prefill_devices"]:
            with torch.cuda.device(int(gpu)):
                torch.cuda.empty_cache()
        synchronize_devices(runtime["source_prefill_devices"])
        if gpu_ref() is not None:
            raise RuntimeError(f"source GPU cache {length} remained referenced after staging")
        staged[length] = cpu_cache
        preparations[str(length)] = preparation
    return staged, preparations


def prepare_staged_runtime(
    configuration: Mapping[str, Any],
    *,
    output: Path,
    fingerprint: str,
    lengths: Sequence[int],
    source_artifact: Mapping[str, Any],
    target_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Prefill 32B, stage exact CPU caches, prove release, then load 14B+mapper."""

    import torch
    from transformers import AutoConfig

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import qwen3_layer_devices
    from kvcache_transfer.latency import materialize_mapper_on_target
    from kvcache_transfer.modeling import load_qwen3
    from .provenance import assert_model_instance_identity
    from .runtime import ReverseMapperBundle

    requested = tuple(sorted({int(value) for value in lengths}))
    if not requested or any(value not in PAPER_LENGTHS for value in requested):
        raise ValueError("staged runtime requested invalid latency lengths")
    runtime = configuration["runtime"]
    protocol = configuration["protocol"]
    topology = staged_runtime_topology(configuration)
    gate = host_memory_staging_gate(
        lengths=requested,
        source_weight_bytes=int(source_artifact["weight_shard_total_nbytes"]),
        target_weight_bytes=int(target_artifact["weight_shard_total_nbytes"]),
        overhead_bytes=int(runtime["host_staging_overhead_bytes"]),
    )
    bundle = ReverseMapperBundle.open(Path(configuration["paths"]["mapper_manifest"]).parent)
    source = load_qwen3(
        configuration["paths"]["source_model"],
        devices=runtime["source_prefill_devices"],
        layer_counts=runtime["source_prefill_layer_counts"],
        role="target",
        attention_implementation=protocol["attention_implementation"],
    )
    source_instance = assert_model_instance_identity(
        source, bundle.source_model_identity, semantic_role="source"
    )
    source_sharding = getattr(source, "_kvcache_transfer_layer_sharding_evidence", None)
    if not isinstance(source_sharding, Mapping):
        raise RuntimeError("loaded source lacks exact sharding evidence")
    source_vocabulary_size = int(source.config.vocab_size)
    staged, preparations = _prefill_and_stage_caches(
        source, lengths=requested, configuration=configuration
    )
    expected_total = sum(source_cache_exact_bytes(length) for length in requested)
    if sum(int(row["cache_total_bytes"]) for row in preparations.values()) != expected_total:
        raise RuntimeError("CPU-staged source-cache aggregate bytes changed")
    allocator_before = _cuda_allocator_snapshot(runtime["source_prefill_devices"])
    source_ref = weakref.ref(source)
    del source
    gc.collect()
    release = prove_source_model_released(
        model_ref=source_ref,
        configuration=configuration,
        allocator_before=allocator_before,
    )

    session_id = uuid.uuid4().hex
    session_dir = output / "staging_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_path = session_dir / f"source_cache_staging_{session_id}.json"
    session_payload = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "source_released_before_target_load",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request_fingerprint": fingerprint,
        "session_id": session_id,
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "runtime_topology": topology,
        "runtime_topology_sha256": canonical_sha256(topology),
        "host_memory_staging_gate": gate,
        "prepared_lengths": list(requested),
        "source_vocabulary_size": source_vocabulary_size,
        "source_model_artifact_sha256": canonical_sha256(source_artifact),
        "target_model_artifact_sha256": canonical_sha256(target_artifact),
        "source_model_instance_identity": source_instance,
        "source_model_prefill_sharding": dict(source_sharding),
        "preparations": preparations,
        "cpu_staging_total_bytes": expected_total,
        "source_model_release": release,
        "target_model_loaded_when_receipt_written": False,
        "source_model_live_when_receipt_written": False,
    }
    session_sidecar = hashed_json(session_path, session_payload)
    verified_session = verify_staging_session(
        session_path, fingerprint=fingerprint
    )
    if verified_session["artifact_sha256"] != session_sidecar["sha256"]:
        raise AssertionError("new staging-session receipt did not verify")
    session_binding = {
        "session_receipt": str(session_path.relative_to(output)),
        "session_receipt_sha256": session_sidecar["sha256"],
    }

    # This must remain after the immutable source-release receipt above.
    target = load_qwen3(
        configuration["paths"]["target_model"],
        devices=runtime["target_devices"],
        layer_counts=runtime["target_layer_counts"],
        role="source",
        attention_implementation=protocol["attention_implementation"],
    )
    target_instance = assert_model_instance_identity(
        target, bundle.target_model_identity, semantic_role="target"
    )
    target_sharding = getattr(target, "_kvcache_transfer_layer_sharding_evidence", None)
    if not isinstance(target_sharding, Mapping):
        raise RuntimeError("loaded target lacks exact sharding evidence")
    resident, mapper_setup = materialize_mapper_on_target(
        bundle, qwen3_layer_devices(target)
    )
    if resident.resident_parameter_bytes != MAPPER_PARAMETER_BYTES:
        raise RuntimeError("resident reverse mapper byte count changed")
    source_config = AutoConfig.from_pretrained(
        configuration["paths"]["source_model"], local_files_only=True
    )
    if (
        int(source_config.vocab_size) != source_vocabulary_size
        or int(target.config.vocab_size) != source_vocabulary_size
    ):
        raise RuntimeError("source/target vocabulary differs from staged prefixes")
    mapper_provenance = {
        "direction": "qwen3_32b_to_14b",
        "manifest": str(bundle.manifest_path),
        "manifest_sha256": resident.manifest_sha256,
        "fit_fingerprint": resident.fit_fingerprint,
        "layers": len(resident.layers),
        "selected_source_layers": 20,
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
        "target_layer_devices": [str(value) for value in qwen3_layer_devices(target)],
        "resident_parameter_bytes": resident.resident_parameter_bytes,
        "materialization": mapper_setup,
        "source_model_artifact": source_instance,
        "target_model_artifact": target_instance,
        "source_model_live": False,
    }
    return {
        "target": target,
        "resident": resident,
        "source_config": source_config,
        "staged_caches": staged,
        "preparations": preparations,
        "session_binding": session_binding,
        "session_receipt": verified_session,
        "source_sharding": dict(source_sharding),
        "target_sharding": dict(target_sharding),
        "mapper_setup": mapper_setup,
        "mapper_provenance": mapper_provenance,
        "source_model_release": release,
        "topology": topology,
    }


def run_latency(configuration: Mapping[str, Any], output: Path) -> int:
    # The formal pipeline is persistent: a repository-managed predecessor is
    # a queue condition.  Once acquired, the in-lock idle gate below still
    # rejects any unmanaged GPU owner before model loading.
    lock = acquire_gpu_lock(blocking=True)
    try:
        source_identity = model_artifact_identity(configuration["paths"]["source_model"])
        target_identity = model_artifact_identity(configuration["paths"]["target_model"])
        mapper_identity = mapper_manifest_identity(configuration["paths"]["mapper_manifest"])
        idle_gate = wait_for_idle_gpus(configuration)
        hardware = hardware_snapshot()
        topology = staged_runtime_topology(configuration)
        run_definition = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "kind": "qwen3_32b_to_14b_prefill_latency",
            "direction": "qwen3_32b_to_14b",
            "paper_reference": configuration["paper_reference"],
            "protocol": configuration["protocol"],
            "runtime": configuration["runtime"],
            "staged_runtime_topology": topology,
            "staged_runtime_topology_sha256": canonical_sha256(topology),
            "paper_h100_reference_only": configuration["paper_h100_reference_only"],
            "absolute_h100_bit_exact_claimed": False,
            "gpu_lock": {
                "path": str(GLOBAL_GPU_LOCK),
                "scope": "repository_wide_forward_and_reverse_eight_gpu_experiments",
                "acquired_before_idle_gate": True,
                "held_through_final_manifest": True,
            },
            "runtime_artifacts": {
                "source_model": source_identity,
                "target_model": target_identity,
                "mapper": mapper_identity,
            },
            "hardware": hardware,
            # The gate observations contain UTC timestamps and are fresh on
            # every resume.  The immutable request binds the policy; the
            # latest passed observations are separately hashed in the run.
            "idle_gate_policy": {
                "memory_limit_mib": configuration["runtime"]["idle_memory_limit_mib"],
                "required_consecutive_samples": configuration["runtime"][
                    "idle_consecutive_samples"
                ],
                "sample_interval_seconds": configuration["runtime"][
                    "idle_sample_interval_seconds"
                ],
                "timeout_seconds": configuration["runtime"]["idle_timeout_seconds"],
            },
            "code": {
                "reverse_experiment": code_identity(CODE_FILES),
                "shared_read_only_runtime": code_identity(
                    SHARED_CODE_FILES, root=REPOSITORY_ROOT
                ),
            },
        }
        fingerprint = canonical_sha256(run_definition)
        run_definition = {**run_definition, "request_fingerprint": fingerprint}
        immutable_hashed_json(output / "run_definition.json", run_definition)
        hashed_json(output / "gpu_idle_gate.json", idle_gate)
        immutable_hashed_json(output / "hardware.json", hardware)

        src_root = REPOSITORY_ROOT / "src"
        if str(src_root) not in sys.path:
            sys.path.insert(0, str(src_root))
        import torch
        from kvcache_transfer.cache_adapter import qwen3_layer_devices
        from kvcache_transfer.latency import (
            benchmark_mapper_end_to_end,
            benchmark_target_reprefill,
            configure_strict_latency_math,
            synchronize_devices,
        )
        from kvcache_transfer.transfer import transfer_source_cache

        math_contract = configure_strict_latency_math()
        protocol = configuration["protocol"]
        checkpoints: list[dict[str, Any]] = []
        missing_lengths: list[int] = []
        for length in PAPER_LENGTHS:
            destination = _checkpoint(output, length)
            payload_exists = destination.exists()
            sidecar_exists = destination.with_name(
                f"{destination.name}.sha256.json"
            ).exists()
            if payload_exists != sidecar_exists:
                raise ValueError(f"partial latency checkpoint is not resumable: {destination}")
            completed = verify_length_checkpoint(
                destination, fingerprint=fingerprint, length=length
            )
            if completed is not None:
                checkpoints.append(completed)
            else:
                missing_lengths.append(length)

        smoke_path = output / "endpoint_memory_smoke.json"
        smoke_payload_exists = smoke_path.exists()
        smoke_sidecar_exists = smoke_path.with_name(
            f"{smoke_path.name}.sha256.json"
        ).exists()
        if smoke_payload_exists != smoke_sidecar_exists:
            raise ValueError("partial endpoint memory-smoke artifact is not resumable")
        smoke = verify_endpoint_memory_smoke(smoke_path, fingerprint=fingerprint)
        staged_runtime: dict[str, Any] | None = None
        staging_lengths = sorted(
            set(missing_lengths) | ({max(PAPER_LENGTHS)} if smoke is None else set())
        )
        if staging_lengths:
            staged_runtime = prepare_staged_runtime(
                configuration,
                output=output,
                fingerprint=fingerprint,
                lengths=staging_lengths,
                source_artifact=source_identity,
                target_artifact=target_identity,
            )
            target = staged_runtime["target"]
            resident = staged_runtime["resident"]
            source_config = staged_runtime["source_config"]
            runtime_setup = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "ready",
                "request_fingerprint": fingerprint,
                "semantic_role_binding": {
                    "source": {"model": "Qwen3-32B", "capture_native_forward_role": "target"},
                    "target": {"model": "Qwen3-14B", "capture_native_forward_role": "source"},
                },
                "staged_runtime_topology": topology,
                "source_sharding": staged_runtime["source_sharding"],
                "target_sharding": staged_runtime["target_sharding"],
                "source_model_release": staged_runtime["source_model_release"],
                "source_model_live_during_target_mapper_runtime": False,
                "staging_session": {
                    **staged_runtime["session_binding"],
                    "prepared_lengths": staging_lengths,
                },
                "mapper_setup": staged_runtime["mapper_setup"],
                "mapper_provenance": staged_runtime["mapper_provenance"],
                "resident_mapper_parameter_bytes": resident.resident_parameter_bytes,
                "mapper_resident_before_all_timed_cells": True,
                "math_contract": math_contract,
                "memory_after_staged_runtime_load": cuda_memory_snapshot(
                    "after_source_release_target_and_mapper_load"
                ),
            }
            hashed_json(output / "runtime_setup.json", runtime_setup)
            if smoke is None:
                endpoint_length = max(PAPER_LENGTHS)
                preparation = staged_runtime["preparations"][str(endpoint_length)]
                smoke = endpoint_memory_smoke(
                    target=target,
                    resident=resident,
                    source_config=source_config,
                    staged_cache=staged_runtime["staged_caches"][endpoint_length],
                    preparation=preparation,
                    session_binding=staged_runtime["session_binding"],
                    configuration=configuration,
                    fingerprint=fingerprint,
                )
                hashed_json(smoke_path, smoke)
                smoke = verify_endpoint_memory_smoke(smoke_path, fingerprint=fingerprint)
                assert smoke is not None
        else:
            runtime_setup_path = output / "runtime_setup.json"
            if not runtime_setup_path.is_file():
                raise ValueError("complete latency checkpoints lack runtime setup")
            verify_hashed_json(runtime_setup_path)
            if smoke is None:
                raise ValueError("complete latency checkpoints lack endpoint memory smoke")

        cache_ids = _expanded_layer_devices(
            SOURCE_CACHE_DEVICES, SOURCE_CACHE_LAYER_COUNTS, expected_layers=64
        )
        cache_devices = tuple(f"cuda:{value}" for value in cache_ids)
        for length in missing_lengths:
            if staged_runtime is None:
                raise AssertionError("missing latency length lacks a staged runtime")
            destination = _checkpoint(output, length)
            prefix = deterministic_prefix(
                length, int(source_config.vocab_size), int(protocol["seed"])
            )
            identity = prefix_identity(prefix)
            preparation = staged_runtime["preparations"][str(length)]
            if identity != preparation["deterministic_prefix"]:
                raise RuntimeError("timed-cell prefix differs from staged source cache")
            cpu_cache = staged_runtime["staged_caches"].pop(length)
            source_cache, restage = restage_source_cache_on_gpus(
                cpu_cache,
                preparation=preparation,
                configuration=configuration,
            )
            del cpu_cache
            source_cache_ref = weakref.ref(source_cache)
            cache_evidence = source_cache_evidence(
                source_cache,
                length,
                expected_layer_devices=cache_devices,
            )
            before = cuda_memory_snapshot(
                f"length_{length}_after_verified_gpu_cache_restage"
            )
            with torch.inference_mode():
                preflight_cache = transfer_source_cache(
                    source_cache,
                    source_config=source_config,
                    target_model=target,
                    mapper_bundle=resident,
                    token_chunk=int(protocol["mapper_token_chunk"]),
                    strict_validation=True,
                )
                synchronize_devices(qwen3_layer_devices(target))
                preflight_length = int(preflight_cache.get_seq_length())
                if preflight_length != length or len(preflight_cache) != 40:
                    raise RuntimeError("untimed strict mapper preflight produced an invalid cache")
                del preflight_cache
                for device in configuration["runtime"]["target_devices"]:
                    with torch.cuda.device(int(device)):
                        torch.cuda.empty_cache()
                target_reprefill = benchmark_target_reprefill(
                    target,
                    prefix,
                    warmup=WARMUP,
                    repeats=REPEATS,
                    collect_cuda_peak_memory=True,
                )
                mapper = benchmark_mapper_end_to_end(
                    source_cache,
                    source_config=source_config,
                    target_model=target,
                    mapper_bundle=resident,
                    token_chunk=int(protocol["mapper_token_chunk"]),
                    warmup=WARMUP,
                    repeats=REPEATS,
                    strict_validation=False,
                    collect_cuda_peak_memory=True,
                )
            after = cuda_memory_snapshot(f"length_{length}_after_both_metrics")
            reprefill_median = float(target_reprefill["median"])
            mapper_median = float(mapper["median"])
            if mapper_median <= 0 or reprefill_median <= 0:
                raise RuntimeError("latency medians must be positive")
            del source_cache
            release = release_restaged_source_cache(
                cache_ref=source_cache_ref, configuration=configuration
            )
            lifecycle = {
                "status": "complete",
                "policy": SOURCE_CACHE_STAGING_POLICY,
                **staged_runtime["session_binding"],
                "preparation_sha256": canonical_sha256(preparation),
                "source_model_live_during_timed_cells": False,
                "source_prefill_timed": False,
                "cpu_staging_timed": False,
                "gpu_restage_timed": False,
                "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
                "mapper_timer_started": True,
                "deterministic_prefix": identity,
                "cpu_staged_raw_bit_sha256": preparation[
                    "cpu_staged_raw_bit_sha256"
                ],
                "cache_total_bytes": source_cache_exact_bytes(length),
                "restage": restage,
                "restaged_cache_release": release,
            }
            checkpoint = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "complete",
                "request_fingerprint": fingerprint,
                "prefix_length": length,
                "batch_size": 1,
                "source_prefill_timed": False,
                "source_cache_existed_before_each_timed_metric": True,
                "source_prefill_untimed_diagnostic_ms": preparation[
                    "source_prefill_untimed_diagnostic_ms"
                ],
                "source_model_reference_passed_to_timed_mapper": False,
                "source_model_live_during_timed_cells": False,
                "cpu_staging_timed": False,
                "gpu_restage_timed": False,
                "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
                "deterministic_prefix": identity,
                "source_cache": cache_evidence,
                "source_cache_lifecycle": lifecycle,
                "untimed_strict_mapper_preflight": {
                    "status": "passed",
                    "target_cache_layers": 40,
                    "target_cache_sequence_length": preflight_length,
                    "strict_validation": True,
                },
                "timed_mapper_strict_validation": False,
                "target_reprefill": target_reprefill,
                "mapper_handoff_end_to_end": mapper,
                "median_speedup_reprefill_over_mapper": reprefill_median / mapper_median,
                "memory_evidence": {
                    "before_metrics": before,
                    "after_metrics_before_source_cache_release": after,
                    "source_cache_release": release,
                },
            }
            hashed_json(destination, checkpoint)
            completed = verify_length_checkpoint(
                destination, fingerprint=fingerprint, length=length
            )
            assert completed is not None
            checkpoints.append(completed)
            del prefix

        if staged_runtime is not None:
            staged_runtime["staged_caches"].clear()
            gc.collect()

        checkpoints.sort(key=lambda value: int(value["prefix_length"]))
        summary = summarize_checkpoints(checkpoints)
        summary_sidecar = hashed_json(output / "summary.json", summary)
        checkpoint_manifest = [
            {
                "prefix_length": length,
                "path": str(_checkpoint(output, length).relative_to(output)),
                "sha256": sha256_file(_checkpoint(output, length)),
                "nbytes": _checkpoint(output, length).stat().st_size,
            }
            for length in PAPER_LENGTHS
        ]
        if smoke is None:
            raise AssertionError("latency finalization lacks verified endpoint smoke")
        referenced_sessions = {
            str(value["source_cache_lifecycle"]["session_receipt"])
            for value in checkpoints
        }
        referenced_sessions.add(
            str(smoke["source_cache_lifecycle"]["session_receipt"])
        )
        staging_receipts: list[dict[str, Any]] = []
        for session_path in _all_staging_session_paths(
            output, referenced=sorted(referenced_sessions)
        ):
            session = verify_staging_session(session_path, fingerprint=fingerprint)
            staging_receipts.append(
                {
                    "session_id": session["session_id"],
                    "path": str(session_path.relative_to(output)),
                    "sha256": session["artifact_sha256"],
                    "nbytes": session["artifact_nbytes"],
                    "prepared_lengths": session["prepared_lengths"],
                }
            )
        manifest = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "complete",
            "request_fingerprint": fingerprint,
            "direction": "qwen3_32b_to_14b",
            "lengths": list(PAPER_LENGTHS),
            "warmup": WARMUP,
            "timed_repeats": REPEATS,
            "source_prefill_timed": False,
            "source_model_live_during_timed_cells": False,
            "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
            "cpu_staging_and_gpu_restage_timed": False,
            "mapper_timer_start": "preexisting_gpu_resident_source_kv",
            "staged_runtime_topology": topology,
            "staged_runtime_topology_sha256": canonical_sha256(topology),
            "summary": summary_sidecar,
            "gpu_idle_gate": {
                "sha256": sha256_file(output / "gpu_idle_gate.json"),
                "nbytes": (output / "gpu_idle_gate.json").stat().st_size,
            },
            "hardware": {
                "sha256": sha256_file(output / "hardware.json"),
                "nbytes": (output / "hardware.json").stat().st_size,
            },
            "runtime_setup": {
                "sha256": sha256_file(output / "runtime_setup.json"),
                "nbytes": (output / "runtime_setup.json").stat().st_size,
            },
            "endpoint_memory_smoke": {
                "sha256": sha256_file(output / "endpoint_memory_smoke.json"),
                "nbytes": (output / "endpoint_memory_smoke.json").stat().st_size,
                "minimum_required_headroom_mib": 2048,
                "minimum_observed_headroom_mib": smoke[
                    "minimum_observed_headroom_mib"
                ],
            },
            "staging_receipts": staging_receipts,
            "staging_receipts_sha256": canonical_sha256(staging_receipts),
            "checkpoint_manifest_sha256": canonical_sha256(checkpoint_manifest),
            "checkpoints": checkpoint_manifest,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        manifest_sidecar = hashed_json(output / "manifest.json", manifest)
        completion_receipt = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "complete",
            "direction": "qwen3_32b_to_14b",
            "request_fingerprint": fingerprint,
            "source_cache_staging_policy": SOURCE_CACHE_STAGING_POLICY,
            "staged_runtime_topology_sha256": canonical_sha256(topology),
            "manifest": manifest_sidecar,
            "summary_sha256": sha256_file(output / "summary.json"),
            "endpoint_memory_smoke_sha256": sha256_file(smoke_path),
            "checkpoint_manifest_sha256": canonical_sha256(checkpoint_manifest),
            "staging_receipts_sha256": canonical_sha256(staging_receipts),
            "all_ten_lengths_complete": len(checkpoint_manifest) == len(PAPER_LENGTHS),
            "source_model_release_proven_for_every_session": True,
            "mapper_timers_started_from_gpu_resident_source_kv": True,
        }
        hashed_json(output / "formal_latency_receipt.json", completion_receipt)
        if verify_latency_completion(output) is None:
            raise AssertionError("formal latency completion receipt did not verify")
        return 0
    finally:
        lock.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configuration = load_latency_config(args.config)
    output = assert_evaluation_output(args.output_dir or _default_output(args.prepare_only))
    if args.prepare_only:
        plan = prepare_only(configuration, output)
        print(json.dumps(plan, ensure_ascii=False, indent=2, default=str))
        return 0
    return run_latency(configuration, output)


__all__ = [
    "DEFAULT_CONFIG",
    "GLOBAL_GPU_LOCK",
    "LATENCY_PACKAGE_DISTRIBUTIONS",
    "MAPPER_PARAMETER_BYTES",
    "PAPER_LENGTHS",
    "REPEATS",
    "SOURCE_CACHE_DEVICES",
    "SOURCE_CACHE_LAYER_COUNTS",
    "SOURCE_CACHE_STAGING_POLICY",
    "SOURCE_PREFILL_DEVICES",
    "SOURCE_PREFILL_LAYER_COUNTS",
    "TARGET_DEVICES",
    "TARGET_LAYER_COUNTS",
    "WARMUP",
    "deterministic_prefix",
    "endpoint_memory_smoke",
    "hardware_snapshot",
    "host_memory_staging_gate",
    "load_latency_config",
    "latency_package_versions",
    "nvidia_driver_evidence",
    "main",
    "prefix_identity",
    "prepare_only",
    "source_cache_exact_bytes",
    "source_cache_evidence",
    "staged_runtime_topology",
    "summarize_checkpoints",
    "validate_latency_run_definition",
    "validate_source_cache_lifecycle",
    "verify_endpoint_memory_smoke",
    "verify_latency_completion",
    "verify_length_checkpoint",
    "verify_staging_session",
    "wait_for_idle_gpus",
]
