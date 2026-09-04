#!/usr/bin/env python3
"""Benchmark Qwen3 target re-prefill and closed-form KV-cache transfer.

Production mode follows the paper's known latency contract where the source
cache already exists: source prefill is performed once per length outside all
timed regions.  The default paper-main protocol measures only target
transformer re-prefill (no LM head) and end-to-end cache mapping.  Mapper
component profiling, one-token boundary decode, and mapper-plus-boundary
latency are explicitly opt-in supplemental diagnostics and never participate
in the paper-main cell or its peak-memory gate.

The local server has eight RTX 4090 GPUs without NVLink/P2P.  Absolute H100
numbers from the paper are reference context only and are never copied into
measured output.  Multi-GPU regions use a host clock bracketed by explicit
CUDA synchronization because a CUDA event cannot span devices.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence
import uuid
import weakref

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.utils import is_flash_attn_2_available

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.cache_adapter import (
    build_dynamic_cache,
    cache_sequence_length,
    cache_to_legacy,
)  # noqa: E402
from kvcache_transfer.config import ArchitectureSpec, PAPER_ARCHITECTURE  # noqa: E402
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.latency import (  # noqa: E402
    COMPONENT_NAMES,
    ResidentMapperBundle,
    benchmark_boundary_decode,
    benchmark_mapper_components,
    benchmark_mapper_end_to_end,
    benchmark_protocol_complete,
    benchmark_target_reprefill,
    configure_strict_latency_math,
    materialize_mapper_on_target,
    synchronize_devices,
)
from kvcache_transfer.mapper import LayerMapper  # noqa: E402
from kvcache_transfer.modeling import (  # noqa: E402
    load_qwen3,
    model_layer_sharding_evidence,
    resolve_contiguous_layer_counts,
)
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
    RunDirectory,
)
from kvcache_transfer.transfer import MapperBundle, transfer_source_cache  # noqa: E402


PAPER_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
TINY_DEFAULT_LENGTHS = (8, 16)
SCHEMA_VERSION = 3
CUDA_PEAK_MEMORY_SCHEMA_VERSION = 2
CUDA_PEAK_MEASUREMENT_SCOPE = (
    "entire_length_cell_from_source_cache_build_through_final_allocator_cleanup"
)
STAGED_CUDA_PEAK_MEASUREMENT_SCOPE = (
    "entire_length_cell_from_preexisting_gpu_source_cache_through_final_allocator_cleanup"
)
SOURCE_CACHE_STAGING_POLICY = "untimed_cpu_staging_then_gpu_restage_v1"
SOURCE_CACHE_STAGING_SESSION_SCHEMA_VERSION = 1
STAGED_CUDA_VISIBLE_DEVICES = "2,3,4,5,6,7,0,1"
STAGED_SOURCE_MODEL_PREFILL_DEVICES = (5, 6, 7)
STAGED_SOURCE_MODEL_PREFILL_LAYER_COUNTS = (14, 13, 13)
STAGED_SOURCE_CACHE_DEVICES = (6, 7)
STAGED_SOURCE_CACHE_LAYER_COUNTS = (20, 20)
STAGED_TARGET_DEVICES = (0, 1, 2, 3, 4, 5)
STAGED_TARGET_LAYER_COUNTS = (10, 11, 11, 11, 11, 10)
STAGED_HOST_MEMORY_OVERHEAD_BYTES = 16 * (1 << 30)
CUDA_PEAK_ALLOCATOR_SCOPE = "current_process_pytorch_cuda_allocator"
CUDA_DRIVER_FREE_SCOPE = "whole_device_snapshots_before_and_after_not_historical_peak"
CUDA_PEAK_DISABLED_REASON = "preserve_caller_owned_outer_cuda_peak_counters"
PRELOAD_GPU_IDLE_GATE_SCHEMA_VERSION = 1
PRELOAD_GPU_IDLE_MEMORY_QUERY = "index,uuid,memory.used"
PRELOAD_GPU_IDLE_COMPUTE_QUERY = "gpu_uuid,pid,process_name,used_memory"
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


def parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def parse_device_csv(value: str) -> tuple[int, ...]:
    """Parse CUDA indices, where device zero is a valid index."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("device indices must be non-negative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return values


def parse_layer_counts_csv(value: str) -> tuple[int, ...]:
    """Parse exact positive layer counts; repeated counts are valid."""

    parts = value.split(",")
    if not parts or any(not item.strip() for item in parts):
        raise argparse.ArgumentTypeError(
            "layer counts must be a non-empty comma-separated list without empty fields"
        )
    try:
        values = tuple(int(item.strip()) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("layer counts must be positive")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("production", "tiny"), default="production")
    parser.add_argument(
        "--source-model", type=Path, default=Path("/home/wby/models/Qwen3-14B")
    )
    parser.add_argument(
        "--target-model", type=Path, default=Path("/home/wby/models/Qwen3-32B")
    )
    parser.add_argument(
        "--mapper",
        type=Path,
        default=ROOT / "artifacts/mappers/qwen3_14b_to_32b/k8_lambda0.01",
    )
    parser.add_argument("--source-devices", type=parse_device_csv, default=(4, 5))
    parser.add_argument("--target-devices", type=parse_device_csv, default=(0, 1, 2, 3))
    parser.add_argument(
        "--source-layer-counts",
        type=parse_layer_counts_csv,
        default=None,
        help=(
            "optional exact contiguous decoder-layer counts, one positive count per "
            "--source-devices entry; counts must sum to 40"
        ),
    )
    parser.add_argument(
        "--target-layer-counts",
        type=parse_layer_counts_csv,
        default=None,
        help=(
            "optional exact contiguous decoder-layer counts, one positive count per "
            "--target-devices entry; counts must sum to 64"
        ),
    )
    parser.add_argument(
        "--stage-source-caches-on-cpu-before-cells",
        action="store_true",
        help=(
            "production-only sequential lifecycle: load the source model first, "
            "prepare every missing source KV cache outside timing, clone it exactly "
            "to CPU, release the source model, then load target+mapper and restage "
            "one source cache on --source-devices before each outer peak reset"
        ),
    )
    parser.add_argument(
        "--source-prefill-devices",
        type=parse_device_csv,
        default=None,
        help=(
            "CUDA devices used only for untimed source-model prefill when CPU cache "
            "staging is enabled; --source-devices remain the timed cache-staging "
            "devices"
        ),
    )
    parser.add_argument(
        "--source-prefill-layer-counts",
        type=parse_layer_counts_csv,
        default=None,
        help=(
            "exact 40-layer source-model split on --source-prefill-devices; valid "
            "only with --stage-source-caches-on-cpu-before-cells"
        ),
    )
    parser.add_argument("--tiny-device", default="cpu")
    parser.add_argument(
        "--lengths",
        type=parse_int_csv,
        default=PAPER_LENGTHS,
        help="prefix lengths; production default is the paper's 64..32768 sweep",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=None,
        help="defaults to --warmup; retained separately because profiling synchronizes components",
    )
    parser.add_argument("--profile-repeats", type=int, default=None)
    parser.add_argument(
        "--supplemental-diagnostics",
        action="store_true",
        help=(
            "also run synchronized mapper-component, one-token boundary, and "
            "mapper-plus-boundary diagnostics; disabled for the paper-main latency sweep"
        ),
    )
    parser.add_argument("--mapper-token-chunk", type=int, default=1024)
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--preload-idle-memory-limit-mib",
        type=int,
        default=64,
        help="maximum nvidia-smi memory.used on every selected GPU before production model load",
    )
    parser.add_argument(
        "--preload-idle-samples",
        type=int,
        default=12,
        help="required consecutive post-identity-hash idle sandwich samples",
    )
    parser.add_argument(
        "--preload-idle-sample-interval-seconds",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--preload-idle-timeout-seconds",
        type=float,
        default=86400.0,
        help="fail-closed deadline while waiting for the post-hash production GPU gate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "exact run directory; an existing directory is resumed only when its "
            "immutable request fingerprint and checkpoint hashes match"
        ),
    )
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("--warmup must be non-negative and --repeats positive")
    if args.profile_warmup is not None and args.profile_warmup < 0:
        raise ValueError("--profile-warmup must be non-negative")
    if args.profile_repeats is not None and args.profile_repeats <= 0:
        raise ValueError("--profile-repeats must be positive")
    if args.mapper_token_chunk <= 0:
        raise ValueError("--mapper-token-chunk must be positive")
    if args.mode == "production":
        # This validates explicit declarations against the paper architecture
        # before any identity hashing or CUDA allocation.  load_qwen3 repeats
        # the check against each checkpoint's actual config and post-load map.
        resolve_contiguous_layer_counts(
            num_hidden_layers=PAPER_ARCHITECTURE.source_layers,
            devices=args.source_devices,
            layer_counts=args.source_layer_counts,
        )
        if args.stage_source_caches_on_cpu_before_cells:
            if (
                args.source_prefill_devices is None
                or args.source_prefill_layer_counts is None
            ):
                raise ValueError(
                    "CPU source-cache staging requires --source-prefill-devices "
                    "and --source-prefill-layer-counts"
                )
            resolve_contiguous_layer_counts(
                num_hidden_layers=PAPER_ARCHITECTURE.source_layers,
                devices=args.source_prefill_devices,
                layer_counts=args.source_prefill_layer_counts,
            )
        elif (
            args.source_prefill_devices is not None
            or args.source_prefill_layer_counts is not None
        ):
            raise ValueError(
                "source-prefill placement is valid only with CPU source-cache staging"
            )
        resolve_contiguous_layer_counts(
            num_hidden_layers=PAPER_ARCHITECTURE.target_layers,
            devices=args.target_devices,
            layer_counts=args.target_layer_counts,
        )
        if args.preload_idle_memory_limit_mib < 0:
            raise ValueError("--preload-idle-memory-limit-mib must be non-negative")
        if args.preload_idle_samples < 2:
            raise ValueError("--preload-idle-samples must be at least two")
        if (
            not math.isfinite(args.preload_idle_sample_interval_seconds)
            or args.preload_idle_sample_interval_seconds < 0
        ):
            raise ValueError(
                "--preload-idle-sample-interval-seconds must be finite and non-negative"
            )
        if (
            not math.isfinite(args.preload_idle_timeout_seconds)
            or args.preload_idle_timeout_seconds <= 0
        ):
            raise ValueError(
                "--preload-idle-timeout-seconds must be finite and positive"
            )
        if args.attention != "flash_attention_2":
            raise ValueError(
                "production paper latency requires --attention flash_attention_2"
            )
        if set(args.source_devices) & set(args.target_devices):
            raise ValueError(
                "source and target device sets must be disjoint in the local protocol"
            )
        if args.stage_source_caches_on_cpu_before_cells:
            prefill_overlap = set(args.source_prefill_devices) & set(
                args.target_devices
            )
            if prefill_overlap != {5}:
                raise ValueError(
                    "audited staged lifecycle requires source-prefill/target "
                    "sequential overlap to be exactly logical CUDA device 5"
                )
            if not set(args.source_devices).issubset(
                set(args.source_prefill_devices)
            ):
                raise ValueError(
                    "cache-staging devices must be a subset of source-prefill devices"
                )
            expected_staged_topology = {
                "source prefill devices": (
                    tuple(args.source_prefill_devices),
                    STAGED_SOURCE_MODEL_PREFILL_DEVICES,
                ),
                "source prefill layer counts": (
                    tuple(args.source_prefill_layer_counts),
                    STAGED_SOURCE_MODEL_PREFILL_LAYER_COUNTS,
                ),
                "source cache devices": (
                    tuple(args.source_devices),
                    STAGED_SOURCE_CACHE_DEVICES,
                ),
                "source cache layer counts": (
                    tuple(args.source_layer_counts or ()),
                    STAGED_SOURCE_CACHE_LAYER_COUNTS,
                ),
                "target devices": (
                    tuple(args.target_devices),
                    STAGED_TARGET_DEVICES,
                ),
                "target layer counts": (
                    tuple(args.target_layer_counts or ()),
                    STAGED_TARGET_LAYER_COUNTS,
                ),
            }
            for label, (observed, expected) in expected_staged_topology.items():
                if observed != expected:
                    raise ValueError(
                        f"audited staged v9 requires {label}={expected}, got {observed}"
                    )
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible != STAGED_CUDA_VISIBLE_DEVICES:
                raise ValueError(
                    "audited staged v9 requires CUDA_VISIBLE_DEVICES="
                    f"{STAGED_CUDA_VISIBLE_DEVICES}, got {visible!r}"
                )
        if not torch.cuda.is_available():
            raise RuntimeError("production latency requires CUDA")
        selected = args.source_devices + args.target_devices
        if max(selected) >= torch.cuda.device_count():
            raise ValueError("a requested CUDA device is unavailable")
        for path, label in (
            (args.source_model, "source model"),
            (args.target_model, "target model"),
            (args.mapper, "mapper"),
        ):
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
    elif args.source_layer_counts is not None or args.target_layer_counts is not None:
        raise ValueError("explicit layer counts are supported only in production mode")
    elif (
        args.stage_source_caches_on_cpu_before_cells
        or args.source_prefill_devices is not None
        or args.source_prefill_layer_counts is not None
    ):
        raise ValueError("source-cache staging is supported only in production mode")


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def model_artifact_identity(model_root: Path) -> dict[str, Any]:
    """Cryptographically bind a latency run to every local weight shard."""

    root = model_root.resolve()
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    files: dict[str, dict[str, Any]] = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required model identity file is missing: {path}")
        files[name] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    for optional in ("tokenizer_config.json", "generation_config.json"):
        path = root / optional
        if path.is_file():
            files[optional] = {
                "sha256": sha256_file(path),
                "nbytes": path.stat().st_size,
            }

    config_value = _strict_json(root / "config.json")
    config_fields = (
        "model_type",
        "vocab_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "hidden_size",
    )
    bound_config_fields: dict[str, Any] | None = None
    if any(field in config_value for field in config_fields):
        bound_config_fields = {}
        for field in config_fields:
            value = config_value.get(field)
            if field == "model_type":
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"model config has invalid {field}: {root / 'config.json'}"
                    )
            elif type(value) is not int or value <= 0:
                raise ValueError(
                    f"model config has invalid {field}: {root / 'config.json'}"
                )
            bound_config_fields[field] = value

    index_path = root / "model.safetensors.index.json"
    index = _strict_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"model safetensors index has no weight_map: {index_path}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    shards: list[dict[str, Any]] = []
    for name in shard_names:
        shard = (root / name).resolve()
        try:
            relative = str(shard.relative_to(root))
        except ValueError as exc:
            raise ValueError(f"model shard escapes model directory: {name}") from exc
        if not shard.is_file():
            raise FileNotFoundError(f"model shard named by index is missing: {shard}")
        shards.append(
            {
                "path": relative,
                "nbytes": shard.stat().st_size,
                "sha256": sha256_file(shard),
            }
        )
    result = {
        "path": str(root),
        "content_hashed_files": files,
        "weight_shards": shards,
        "weight_shard_count": len(shards),
        "weight_shard_total_nbytes": sum(item["nbytes"] for item in shards),
        "weight_shards_digest": _canonical_digest(shards),
        "weight_shard_identity_scope": (
            "index_sha256_plus_relative_filename_nbytes_and_per_shard_sha256"
        ),
        "weight_shard_contents_hashed": True,
    }
    if bound_config_fields is not None:
        result["bound_config_fields"] = bound_config_fields
    return result


def mapper_artifact_identity(mapper_root: Path) -> dict[str, Any]:
    manifest = mapper_root.resolve() / "manifest.json"
    checksum = mapper_root.resolve() / "manifest.sha256.json"
    if not manifest.is_file() or not checksum.is_file():
        raise FileNotFoundError(
            f"mapper manifest and checksum commit marker are required: {manifest}, {checksum}"
        )
    manifest_sha = sha256_file(manifest)
    checksum_value = _strict_json(checksum)
    if checksum_value.get("sha256") != manifest_sha:
        raise ValueError(
            "mapper aggregate manifest SHA256 disagrees with commit marker"
        )
    manifest_value = _strict_json(manifest)
    if manifest_value.get("status") != "complete":
        raise ValueError("production latency requires a complete mapper manifest")
    selection = manifest_value.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("mapper manifest lacks selection provenance")
    local_selection = mapper_root.resolve() / "inputs" / "selected_layers.json"
    if not local_selection.is_file():
        raise FileNotFoundError(
            f"mapper selection snapshot is missing: {local_selection}"
        )
    local_selection_sha = sha256_file(local_selection)
    declared_selection_sha = selection.get("source_file_sha256_used_by_fit")
    if local_selection_sha != declared_selection_sha:
        raise ValueError("mapper selection snapshot SHA256 disagrees with manifest")
    phase3 = selection.get("phase3_provenance")
    if not isinstance(phase3, Mapping):
        raise ValueError("mapper manifest lacks Phase-3 selection provenance")
    phase3_artifacts: dict[str, dict[str, Any]] = {}
    for stem in ("activation_catalog", "score"):
        path_value = phase3.get(f"{stem}_path")
        recorded_sha = phase3.get(f"{stem}_sha256")
        path = Path(str(path_value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Phase-3 {stem} artifact is missing: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != recorded_sha:
            raise ValueError(f"Phase-3 {stem} SHA256 disagrees with mapper manifest")
        phase3_artifacts[stem] = {
            "path": str(path),
            "sha256": actual_sha,
            "nbytes": path.stat().st_size,
        }
    return {
        "root": str(mapper_root.resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "checksum": str(checksum),
        "checksum_sha256": sha256_file(checksum),
        "fit_fingerprint": manifest_value.get("fit_fingerprint"),
        "layer_count": len(manifest_value.get("layers", [])),
        "selection": {
            "embedded_manifest_selection_digest": _canonical_digest(selection),
            "snapshot": str(local_selection),
            "snapshot_sha256": local_selection_sha,
            "selection_input_fingerprint": phase3.get("selection_input_fingerprint"),
            "phase3_artifacts": phase3_artifacts,
        },
    }


def project_code_provenance() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative in LATENCY_CODE_FILES:
        path = CODE_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"formal latency code file is missing: {path}")
        files[relative] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    return {"files": files, "files_digest": _canonical_digest(files)}


def backend_identity(attention: str) -> dict[str, Any]:
    try:
        nccl_version_raw = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        nccl_version: int | list[int] | str | None = None
    else:
        if isinstance(nccl_version_raw, tuple):
            nccl_version = [int(item) for item in nccl_version_raw]
        elif isinstance(nccl_version_raw, (int, str)):
            nccl_version = nccl_version_raw
        else:
            nccl_version = str(nccl_version_raw)
    runtime_environment = {
        name: os.environ.get(name)
        for name in (
            "PYTORCH_CUDA_ALLOC_CONF",
            "PYTORCH_ALLOC_CONF",
            "CUBLAS_WORKSPACE_CONFIG",
            "NVIDIA_TF32_OVERRIDE",
        )
    }
    return {
        "python_version": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nccl_version": nccl_version,
        "transformers": _package_version("transformers"),
        "accelerate": _package_version("accelerate"),
        "flash_attn": _package_version("flash-attn"),
        "requested_attention": attention,
        "flash_attention_2_available": bool(is_flash_attn_2_available()),
        "nvidia_smi_driver_version": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
        "runtime_environment": runtime_environment,
    }


def analytical_memory_contract(
    *,
    maximum_length: int,
    token_chunk: int,
    artifacts: Mapping[str, Any],
    hardware: Mapping[str, Any],
    source_device_count: int,
    target_device_count: int,
    supplemental_diagnostics: bool,
    source_layer_counts: Sequence[int] | None = None,
    target_layer_counts: Sequence[int] | None = None,
    staged_source_lifecycle: bool = False,
    source_prefill_device_count: int | None = None,
    source_prefill_layer_counts: Sequence[int] | None = None,
    staged_prefix_lengths: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Persist the exact cache/mapper byte arithmetic for the 32K risk gate.

    Model/cache/mapper bytes are apportioned by decoder-layer fraction only as
    an estimate.  It does not isolate fixed embedding, norm, or LM-head bytes;
    Accelerate's actual allocations and every benchmark's measured peak remain
    the authoritative memory observations.
    """

    source_layers, target_layers, heads, head_dim = 40, 64, 8, 128
    resolved_source_counts = resolve_contiguous_layer_counts(
        num_hidden_layers=source_layers,
        devices=range(source_device_count),
        layer_counts=source_layer_counts,
    )
    resolved_target_counts = resolve_contiguous_layer_counts(
        num_hidden_layers=target_layers,
        devices=range(target_device_count),
        layer_counts=target_layer_counts,
    )
    resolved_prefill_counts: tuple[int, ...] | None = None
    if staged_source_lifecycle:
        if source_prefill_device_count is None or source_prefill_layer_counts is None:
            raise ValueError(
                "staged source lifecycle requires its source-prefill device/count contract"
            )
        resolved_prefill_counts = resolve_contiguous_layer_counts(
            num_hidden_layers=source_layers,
            devices=range(source_prefill_device_count),
            layer_counts=source_prefill_layer_counts,
        )
        if not staged_prefix_lengths:
            raise ValueError("staged source lifecycle requires staged prefix lengths")
    bf16_bytes, fp32_bytes, selected_layers = 2, 4, 8
    source_cache = source_layers * 2 * heads * maximum_length * head_dim * bf16_bytes
    target_cache = target_layers * 2 * heads * maximum_length * head_dim * bf16_bytes
    source_inverse_keys = source_layers * heads * maximum_length * head_dim * fp32_bytes
    mapper = (
        target_layers
        * 2
        * (selected_layers * heads * head_dim * heads * head_dim + heads * head_dim)
        * fp32_bytes
    )
    chunk_features = 2 * token_chunk * selected_layers * heads * head_dim * fp32_bytes
    target_layer_fp32_kv = 2 * maximum_length * heads * head_dim * fp32_bytes
    cpu_staging_total = sum(
        source_layers * 2 * heads * int(length) * head_dim * bf16_bytes
        for length in (staged_prefix_lengths or ())
    )
    result: dict[str, Any] = {
        "scope": "analytic_capacity_risk_gate_not_a_measured_peak",
        "maximum_prefix_length": maximum_length,
        "source_cache_bf16_bytes": source_cache,
        "mapped_target_cache_bf16_bytes": target_cache,
        "source_inverse_content_keys_fp32_bytes": source_inverse_keys,
        "resident_mapper_fp32_bytes": mapper,
        "one_token_chunk_packed_k_plus_v_fp32_bytes": chunk_features,
        "one_target_layer_mapped_k_plus_v_fp32_bytes": target_layer_fp32_kv,
        "paper_main_cell_only": not supplemental_diagnostics,
        "supplemental_diagnostics_enabled": supplemental_diagnostics,
        "boundary_branch_strategy": (
            "read_only_shallow_prefix_container"
            if supplemental_diagnostics
            else "not_run"
        ),
        "boundary_prefix_deep_copy_bytes": 0,
        "boundary_parent_plus_decode_output_cache_bytes": (
            2 * target_cache if supplemental_diagnostics else 0
        ),
        "temporary_feature_width": selected_layers * heads * head_dim,
        "temporary_token_chunk": token_chunk,
        "explicit_t_by_t_attention_allocation_expected": False,
        "flash_attention_2_asserted_before_execution": True,
        "formal_acceptance_requires_per_length_cuda_peak_memory_evidence": True,
        "source_cache_staging_policy": (
            SOURCE_CACHE_STAGING_POLICY if staged_source_lifecycle else None
        ),
        "source_model_live_during_timed_cells": (
            False if staged_source_lifecycle else True
        ),
        "all_prefix_lengths_cpu_staging_exact_bytes": (
            cpu_staging_total if staged_source_lifecycle else 0
        ),
        "all_prefix_lengths_cpu_staging_lengths": (
            list(staged_prefix_lengths or ()) if staged_source_lifecycle else []
        ),
        "layer_sharding": {
            "source": {
                "mode": (
                    "explicit_exact_counts"
                    if source_layer_counts is not None
                    else "legacy_even_contiguous"
                ),
                "resolved_layer_counts": list(resolved_source_counts),
                "total_layers": source_layers,
            },
            "target": {
                "mode": (
                    "explicit_exact_counts"
                    if target_layer_counts is not None
                    else "legacy_even_contiguous"
                ),
                "resolved_layer_counts": list(resolved_target_counts),
                "total_layers": target_layers,
            },
        },
    }
    if staged_source_lifecycle:
        assert resolved_prefill_counts is not None
        result["layer_sharding"].update(
            {
                "source_model_prefill": {
                    "mode": "explicit_exact_counts",
                    "resolved_layer_counts": list(resolved_prefill_counts),
                    "total_layers": source_layers,
                },
                "source_cache_staging": dict(result["layer_sharding"]["source"]),
            }
        )
    if "source_model" in artifacts and "target_model" in artifacts:
        source_weights = int(artifacts["source_model"]["weight_shard_total_nbytes"])
        target_weights = int(artifacts["target_model"]["weight_shard_total_nbytes"])
        # Approximate worst live phases by the declared layer fractions while
        # excluding fixed-module imbalance and allocator/framework/attention
        # workspace. Formal measured peaks remain authoritative.
        if staged_source_lifecycle:
            assert resolved_prefill_counts is not None
            # The source model exists only during untimed preparation.  The
            # timed mapper phase begins with a raw BF16 source cache already on
            # the two source-cache cards, so source weights must not be charged
            # to that formal lower bound.
            source_lower_bounds = [
                (source_cache + source_inverse_keys) * count / source_layers
                for count in resolved_source_counts
            ]
            source_prefill_lower_bounds = [
                source_weights * count / source_layers
                + source_cache * count / source_layers
                for count in resolved_prefill_counts
            ]
        else:
            source_lower_bounds = [
                source_weights * count / source_layers
                + (source_cache + source_inverse_keys) * count / source_layers
                for count in resolved_source_counts
            ]
            source_prefill_lower_bounds = source_lower_bounds
        live_target_cache_copies = 2 if supplemental_diagnostics else 1
        target_lower_bounds = [
            target_weights * count / target_layers
            + mapper * count / target_layers
            + live_target_cache_copies * target_cache * count / target_layers
            for count in resolved_target_counts
        ]
        capacities = [
            int(item["total_memory_bytes"]) for item in hardware.get("devices", [])
        ]
        result.update(
            {
                "model_weight_shards_source_bytes": source_weights,
                "model_weight_shards_target_bytes": target_weights,
                "approx_source_per_device_lower_bound_bytes": int(
                    max(source_lower_bounds)
                ),
                "approx_target_cell_per_device_lower_bound_bytes": int(
                    max(target_lower_bounds)
                ),
                "approx_source_per_device_lower_bound_bytes_by_layer_fraction": [
                    int(value) for value in source_lower_bounds
                ],
                "source_lower_bound_phase": (
                    "timed_mapper_source_cache_plus_inverse_without_source_model"
                    if staged_source_lifecycle
                    else "source_model_plus_cache_plus_inverse"
                ),
                "untimed_source_prefill_per_device_lower_bound_bytes": int(
                    max(source_prefill_lower_bounds)
                ),
                "untimed_source_prefill_per_device_lower_bound_bytes_by_layer_fraction": [
                    int(value) for value in source_prefill_lower_bounds
                ],
                "untimed_source_prefill_includes_source_model_weights": True,
                "timed_mapper_source_includes_source_model_weights": (
                    not staged_source_lifecycle
                ),
                "approx_target_cell_per_device_lower_bound_bytes_by_layer_fraction": [
                    int(value) for value in target_lower_bounds
                ],
                "target_cache_copies_in_analytic_cell_bound": live_target_cache_copies,
                "minimum_selected_gpu_capacity_bytes": (
                    min(capacities) if capacities else None
                ),
                "capacity_interpretation": (
                    "Layer-fraction estimates do not isolate embedding, norm, or LM-head weights "
                    "and omit framework/FA2 workspace. Formal per-cell CUDA peak fields are the "
                    "acceptance evidence."
                ),
            }
        )
    return result


def _cli_identity(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(vars(args).items()):
        if isinstance(value, Path):
            result[key] = str(value.resolve())
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def production_layer_sharding_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Return the fingerprinted model/cache layer-sharding contract."""

    result: dict[str, Any] = {}
    for role, devices, declared, total_layers in (
        (
            "source",
            args.source_devices,
            args.source_layer_counts,
            PAPER_ARCHITECTURE.source_layers,
        ),
        (
            "target",
            args.target_devices,
            args.target_layer_counts,
            PAPER_ARCHITECTURE.target_layers,
        ),
    ):
        resolved = resolve_contiguous_layer_counts(
            num_hidden_layers=total_layers,
            devices=devices,
            layer_counts=declared,
        )
        result[role] = {
            "mode": (
                "explicit_exact_counts"
                if declared is not None
                else "legacy_even_contiguous"
            ),
            "devices": list(devices),
            "declared_layer_counts": list(declared) if declared is not None else None,
            "resolved_layer_counts": list(resolved),
            "total_layers": total_layers,
        }
    if args.stage_source_caches_on_cpu_before_cells:
        prefill_counts = resolve_contiguous_layer_counts(
            num_hidden_layers=PAPER_ARCHITECTURE.source_layers,
            devices=args.source_prefill_devices,
            layer_counts=args.source_prefill_layer_counts,
        )
        result["source_model_prefill"] = {
            "mode": "explicit_exact_counts",
            "devices": list(args.source_prefill_devices),
            "declared_layer_counts": list(args.source_prefill_layer_counts),
            "resolved_layer_counts": list(prefill_counts),
            "total_layers": PAPER_ARCHITECTURE.source_layers,
        }
        result["source_cache_staging"] = dict(result["source"])
    return result


def build_run_definition(
    args: argparse.Namespace,
    *,
    lengths: Sequence[int],
    profile_warmup: int,
    profile_repeats: int,
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    layer_sharding = (
        production_layer_sharding_contract(args) if args.mode == "production" else None
    )
    if args.mode == "production":
        artifacts: dict[str, Any] = {
            "source_model": model_artifact_identity(args.source_model),
            "target_model": model_artifact_identity(args.target_model),
            "mapper": mapper_artifact_identity(args.mapper),
        }
    else:
        artifacts = {
            "tiny_runtime": {
                "seed": 20260813,
                "architecture": {
                    "layers": 2,
                    "attention_heads": 4,
                    "kv_heads": 2,
                    "head_dim": 16,
                    "mapper": "identity_fp32",
                },
            }
        }
    if args.mode == "production":
        memory_contract = analytical_memory_contract(
            maximum_length=max(lengths),
            token_chunk=args.mapper_token_chunk,
            artifacts=artifacts,
            hardware=hardware,
            source_device_count=len(args.source_devices),
            target_device_count=len(args.target_devices),
            supplemental_diagnostics=args.supplemental_diagnostics,
            source_layer_counts=args.source_layer_counts,
            target_layer_counts=args.target_layer_counts,
            staged_source_lifecycle=args.stage_source_caches_on_cpu_before_cells,
            source_prefill_device_count=(
                len(args.source_prefill_devices)
                if args.stage_source_caches_on_cpu_before_cells
                else None
            ),
            source_prefill_layer_counts=args.source_prefill_layer_counts,
            staged_prefix_lengths=(
                lengths if args.stage_source_caches_on_cpu_before_cells else None
            ),
        )
    else:
        memory_contract = {
            "scope": "skipped_for_tiny_diagnostic",
            "formal_production_memory_contract_built_only_in_production_mode": True,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "qwen3_kv_cache_transfer_latency",
        "cli": _cli_identity(args),
        "resolved": {
            "lengths": list(lengths),
            "latency_protocol": (
                "paper_main_plus_supplemental"
                if args.supplemental_diagnostics
                else "paper_main_only"
            ),
            "profile_warmup": profile_warmup if args.supplemental_diagnostics else None,
            "profile_repeats": (
                profile_repeats if args.supplemental_diagnostics else None
            ),
            "layer_sharding": layer_sharding,
        },
        "runtime_artifacts": artifacts,
        "project_code": project_code_provenance(),
        "backend": backend_identity(args.attention),
        "analytical_memory": memory_contract,
        "topology": {
            "cuda_visible_devices_environment": hardware.get(
                "cuda_visible_devices_environment"
            ),
            "selected_devices": hardware.get("selected_devices"),
            "devices": hardware.get("devices", []),
            "peer_access": hardware.get("peer_access", {}),
            "all_selected_pairs_peer_access_disabled": hardware.get(
                "all_selected_pairs_peer_access_disabled"
            ),
            "nvidia_smi_topology": hardware.get("nvidia_smi_topology"),
            "nvidia_smi_p2p_read": hardware.get("nvidia_smi_p2p_read"),
            "staged_source_lifecycle": (
                staged_runtime_topology(args)
                if args.mode == "production"
                and args.stage_source_caches_on_cpu_before_cells
                else None
            ),
        },
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _strict_json(path) != dict(value):
            raise ValueError(
                f"immutable run artifact differs from current request: {path}"
            )
        return
    atomic_write_json(path, dict(value))


def _sha_sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256.json")


def write_sha_sidecar(path: Path) -> dict[str, Any]:
    value = {
        "schema_version": SCHEMA_VERSION,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
    }
    atomic_write_json(_sha_sidecar_path(path), value)
    return value


def verify_sha_sidecar(path: Path) -> dict[str, Any]:
    sidecar = _strict_json(_sha_sidecar_path(path))
    expected = {
        "schema_version": SCHEMA_VERSION,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
    }
    if sidecar != expected:
        raise ValueError(f"artifact SHA256 sidecar mismatch: {path}")
    return sidecar


def write_hashed_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    atomic_write_json(path, dict(value))
    return write_sha_sidecar(path)


def prepare_run_directory(
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    run_definition: Mapping[str, Any],
    request_fingerprint: str,
) -> tuple[RunDirectory, bool]:
    definition = {**dict(run_definition), "request_fingerprint": request_fingerprint}
    if args.output_dir is None:
        run = make_run_directory(ROOT, "p8", f"{args.mode}_latency")
        record_run_start(run, sys.argv, dict(configuration))
        _write_immutable_json(run.path / "run_definition.json", definition)
        write_sha_sidecar(run.path / "run_definition.json")
        return run, False

    path = args.output_dir.resolve()
    resumed = path.exists()
    if resumed:
        if not path.is_dir():
            raise ValueError(f"--output-dir exists but is not a directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=False)
    run = RunDirectory(path=path, run_id=path.name)
    if resumed:
        if _strict_json(path / "run_definition.json") != definition:
            raise ValueError("latency resume request fingerprint/definition mismatch")
        if _sha_sidecar_path(path / "run_definition.json").exists():
            verify_sha_sidecar(path / "run_definition.json")
        else:
            # The definition itself matches byte-independent canonical values;
            # complete a sidecar write interrupted between the two atomics.
            write_sha_sidecar(path / "run_definition.json")
        if _strict_json(path / "config.json") != dict(configuration):
            raise ValueError("latency resume configuration mismatch")
    else:
        record_run_start(run, sys.argv, dict(configuration))
        _write_immutable_json(path / "run_definition.json", definition)
        write_sha_sidecar(path / "run_definition.json")
    return run, resumed


def _command_output(command: Sequence[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(command), "error": str(exc)}
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _strict_nvidia_smi_rows(*, scope: str, query: str) -> list[list[str]]:
    command = [
        "nvidia-smi",
        f"--query-{scope}={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"pre-load GPU idle gate could not run nvidia-smi: {exc}"
        ) from exc
    if completed.returncode != 0:
        raise RuntimeError(
            "pre-load GPU idle gate nvidia-smi query failed: "
            f"returncode={completed.returncode}, stderr={completed.stderr.strip()!r}"
        )
    try:
        return [
            [field.strip() for field in row]
            for row in csv.reader(completed.stdout.splitlines())
            if row and any(field.strip() for field in row)
        ]
    except csv.Error as exc:
        raise RuntimeError(
            "pre-load GPU idle gate received malformed nvidia-smi CSV"
        ) from exc


def _preload_gpu_memory_rows() -> list[dict[str, Any]]:
    rows = _strict_nvidia_smi_rows(scope="gpu", query=PRELOAD_GPU_IDLE_MEMORY_QUERY)
    result: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise RuntimeError(
                "pre-load GPU idle gate memory query has the wrong column count"
            )
        try:
            physical_index = int(row[0])
            memory_used_mib = int(row[2])
        except ValueError as exc:
            raise RuntimeError(
                "pre-load GPU idle gate memory query is not integral"
            ) from exc
        gpu_uuid = row[1]
        if (
            physical_index < 0
            or memory_used_mib < 0
            or not gpu_uuid
            or physical_index in seen_indices
            or gpu_uuid in seen_uuids
        ):
            raise RuntimeError(
                "pre-load GPU idle gate memory query has invalid/duplicate GPUs"
            )
        seen_indices.add(physical_index)
        seen_uuids.add(gpu_uuid)
        result.append(
            {
                "physical_index": physical_index,
                "gpu_uuid": gpu_uuid,
                "memory_used_mib": memory_used_mib,
            }
        )
    if not result:
        raise RuntimeError("pre-load GPU idle gate memory query returned no GPUs")
    return result


def _preload_gpu_compute_rows() -> list[dict[str, Any]]:
    rows = _strict_nvidia_smi_rows(
        scope="compute-apps", query=PRELOAD_GPU_IDLE_COMPUTE_QUERY
    )
    result: list[dict[str, Any]] = []
    for row in rows:
        if len(row) != 4:
            raise RuntimeError(
                "pre-load GPU idle gate compute query has the wrong column count"
            )
        try:
            pid = int(row[1])
        except ValueError as exc:
            raise RuntimeError(
                "pre-load GPU idle gate compute PID is not integral"
            ) from exc
        try:
            memory_used_mib: int | None = int(row[3])
        except ValueError:
            # nvidia-smi can race process teardown and print ``[Not Found]``.
            # Retain the row as a busy observation instead of treating it as
            # empty or silently discarding it.
            memory_used_mib = None
        if (
            not row[0]
            or pid <= 0
            or (memory_used_mib is not None and memory_used_mib < 0)
        ):
            raise RuntimeError(
                "pre-load GPU idle gate compute query has invalid values"
            )
        result.append(
            {
                "gpu_uuid": row[0],
                "pid": pid,
                "process_name": row[2],
                "memory_used_mib": memory_used_mib,
            }
        )
    return result


def _preload_gpu_bindings(
    selected_devices: Sequence[int], memory_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        raise RuntimeError(
            "production pre-load GPU idle gate requires explicit numeric CUDA_VISIBLE_DEVICES"
        )
    tokens = [token.strip() for token in visible.split(",")]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise RuntimeError(
            "production pre-load GPU idle gate requires numeric CUDA_VISIBLE_DEVICES entries"
        )
    physical_indices = [int(token) for token in tokens]
    if len(set(physical_indices)) != len(physical_indices):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES contains duplicate physical GPU indices"
        )
    inventory = {int(row["physical_index"]): row for row in memory_rows}
    selected = _strict_cuda_device_ids(
        selected_devices, label="pre-load selected CUDA devices"
    )
    if any(logical >= len(physical_indices) for logical in selected):
        raise RuntimeError("a selected logical GPU is outside CUDA_VISIBLE_DEVICES")
    bindings: list[dict[str, Any]] = []
    for logical in selected:
        physical = physical_indices[logical]
        row = inventory.get(physical)
        if row is None:
            raise RuntimeError(
                f"physical GPU {physical} is absent from nvidia-smi inventory"
            )
        bindings.append(
            {
                "logical_index": logical,
                "physical_index": physical,
                "gpu_uuid": str(row["gpu_uuid"]),
            }
        )
    return bindings


def preload_gpu_idle_gate_policy(
    args: argparse.Namespace, selected_devices: Sequence[int]
) -> dict[str, Any]:
    return {
        "schema_version": PRELOAD_GPU_IDLE_GATE_SCHEMA_VERSION,
        "scope": "post_identity_hash_pre_cuda_model_load",
        "cuda_visible_devices_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "selected_logical_device_ids": list(selected_devices),
        "maximum_memory_used_mib": args.preload_idle_memory_limit_mib,
        "required_consecutive_sandwich_samples": args.preload_idle_samples,
        "sandwich_interval_seconds": args.preload_idle_sample_interval_seconds,
        "timeout_seconds": args.preload_idle_timeout_seconds,
        "memory_query": PRELOAD_GPU_IDLE_MEMORY_QUERY,
        "compute_apps_query": PRELOAD_GPU_IDLE_COMPUTE_QUERY,
        "selected_compute_apps_required_empty": True,
        "streak_resets_on_any_failed_observation": True,
        "nvidia_smi_error_fails_closed": True,
    }


def wait_for_preload_gpu_idle_gate(
    args: argparse.Namespace, selected_devices: Sequence[int]
) -> dict[str, Any]:
    """Wait for a post-hash, pre-model-load stable idle window on selected GPUs."""

    policy = preload_gpu_idle_gate_policy(args, selected_devices)
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    first_memory = _preload_gpu_memory_rows()
    bindings = _preload_gpu_bindings(selected_devices, first_memory)
    selected_uuids = {item["gpu_uuid"] for item in bindings}
    attempts = 0
    streak: list[dict[str, Any]] = []

    def selected_memory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        by_uuid = {str(row["gpu_uuid"]): row for row in rows}
        if len(by_uuid) != len(rows):
            raise RuntimeError("pre-load GPU idle gate observed duplicate GPU UUIDs")
        values: list[dict[str, Any]] = []
        for binding in bindings:
            row = by_uuid.get(binding["gpu_uuid"])
            if row is None or int(row["physical_index"]) != binding["physical_index"]:
                raise RuntimeError(
                    "selected GPU identity changed during pre-load idle gate"
                )
            values.append({**binding, "memory_used_mib": int(row["memory_used_mib"])})
        return values

    while len(streak) < args.preload_idle_samples:
        if time.monotonic() - started_monotonic > args.preload_idle_timeout_seconds:
            raise TimeoutError(
                "pre-load GPU idle gate timed out before a stable selected-GPU window"
            )
        attempts += 1
        memory_before = selected_memory(
            first_memory if attempts == 1 else _preload_gpu_memory_rows()
        )
        apps_before = [
            row
            for row in _preload_gpu_compute_rows()
            if row["gpu_uuid"] in selected_uuids
        ]
        time.sleep(args.preload_idle_sample_interval_seconds)
        memory_after = selected_memory(_preload_gpu_memory_rows())
        apps_after = [
            row
            for row in _preload_gpu_compute_rows()
            if row["gpu_uuid"] in selected_uuids
        ]
        sample = {
            "sequence": attempts,
            "memory_before": memory_before,
            "selected_compute_apps_before": apps_before,
            "memory_after": memory_after,
            "selected_compute_apps_after": apps_after,
        }
        passed = (
            not apps_before
            and not apps_after
            and all(
                item["memory_used_mib"] <= args.preload_idle_memory_limit_mib
                for item in (*memory_before, *memory_after)
            )
        )
        if passed:
            streak.append(sample)
        else:
            streak.clear()
            print(
                "pre-load GPU idle gate observed a busy selected GPU; idle streak reset",
                file=sys.stderr,
                flush=True,
            )

    return {
        "schema_version": PRELOAD_GPU_IDLE_GATE_SCHEMA_VERSION,
        "status": "passed",
        "policy": policy,
        "selected_gpu_bindings": bindings,
        "started_utc": started_utc,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "attempt_count": attempts,
        "rejected_attempt_count": attempts - len(streak),
        "final_consecutive_sandwich_samples": streak,
        "all_expensive_identity_hashes_completed_before_gate": True,
        "completed_before_cuda_model_load": True,
    }


def immediate_preload_gpu_idle_sandwich(
    args: argparse.Namespace, selected_devices: Sequence[int]
) -> dict[str, Any]:
    """Perform the final double observation immediately before ``load_runtime``."""

    before_rows = _preload_gpu_memory_rows()
    bindings = _preload_gpu_bindings(selected_devices, before_rows)
    selected_uuids = {item["gpu_uuid"] for item in bindings}

    def selected_memory(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        by_uuid = {str(row["gpu_uuid"]): row for row in rows}
        result: list[dict[str, Any]] = []
        for binding in bindings:
            row = by_uuid.get(binding["gpu_uuid"])
            if row is None or int(row["physical_index"]) != binding["physical_index"]:
                raise RuntimeError(
                    "selected GPU identity changed in immediate pre-load gate"
                )
            result.append({**binding, "memory_used_mib": int(row["memory_used_mib"])})
        return result

    memory_before = selected_memory(before_rows)
    apps_before = [
        row for row in _preload_gpu_compute_rows() if row["gpu_uuid"] in selected_uuids
    ]
    interval = min(float(args.preload_idle_sample_interval_seconds), 1.0)
    time.sleep(interval)
    memory_after = selected_memory(_preload_gpu_memory_rows())
    apps_after = [
        row for row in _preload_gpu_compute_rows() if row["gpu_uuid"] in selected_uuids
    ]
    if (
        apps_before
        or apps_after
        or any(
            item["memory_used_mib"] > args.preload_idle_memory_limit_mib
            for item in (*memory_before, *memory_after)
        )
    ):
        raise RuntimeError(
            "selected GPU became busy immediately before production CUDA model load"
        )
    return {
        "status": "passed",
        "scope": "immediately_before_cuda_model_load",
        "sandwich_interval_seconds": interval,
        "memory_before": memory_before,
        "selected_compute_apps_before": apps_before,
        "memory_after": memory_after,
        "selected_compute_apps_after": apps_after,
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def verify_preload_gpu_idle_gate_evidence(
    value: Any,
    *,
    expected_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(
            "production run definition lacks pre-load GPU idle-gate evidence"
        )
    evidence = dict(value)
    if (
        evidence.get("schema_version") != PRELOAD_GPU_IDLE_GATE_SCHEMA_VERSION
        or evidence.get("status") != "passed"
        or evidence.get("policy") != dict(expected_policy)
        or evidence.get("all_expensive_identity_hashes_completed_before_gate")
        is not True
        or evidence.get("completed_before_cuda_model_load") is not True
    ):
        raise ValueError("production pre-load GPU idle-gate evidence contract mismatch")
    bindings = evidence.get("selected_gpu_bindings")
    visible_tokens = str(expected_policy["cuda_visible_devices_environment"]).split(",")
    expected_physical = [
        int(visible_tokens[index])
        for index in expected_policy["selected_logical_device_ids"]
    ]
    if (
        not isinstance(bindings, list)
        or any(not isinstance(item, Mapping) for item in bindings)
        or [item.get("logical_index") for item in bindings]
        != list(expected_policy["selected_logical_device_ids"])
        or [item.get("physical_index") for item in bindings] != expected_physical
        or any(
            set(item) != {"logical_index", "physical_index", "gpu_uuid"}
            or type(item.get("physical_index")) is not int
            or not isinstance(item.get("gpu_uuid"), str)
            or not item["gpu_uuid"]
            for item in bindings
        )
        or len({item["physical_index"] for item in bindings}) != len(bindings)
        or len({item["gpu_uuid"] for item in bindings}) != len(bindings)
    ):
        raise ValueError("production pre-load GPU idle-gate device bindings mismatch")
    samples = evidence.get("final_consecutive_sandwich_samples")
    required = int(expected_policy["required_consecutive_sandwich_samples"])
    maximum = int(expected_policy["maximum_memory_used_mib"])
    attempts = evidence.get("attempt_count")
    rejected = evidence.get("rejected_attempt_count")
    if (
        type(attempts) is not int
        or attempts < required
        or type(rejected) is not int
        or rejected != attempts - required
    ):
        raise ValueError(
            "production pre-load GPU idle-gate attempt accounting mismatch"
        )
    if not isinstance(samples, list) or len(samples) != required:
        raise ValueError("production pre-load GPU idle-gate sample streak mismatch")
    expected_sequences = list(range(attempts - required + 1, attempts + 1))
    if [
        sample.get("sequence") if isinstance(sample, Mapping) else None
        for sample in samples
    ] != expected_sequences:
        raise ValueError("production pre-load GPU idle-gate sample sequence mismatch")
    for sample in samples:
        if (
            not isinstance(sample, Mapping)
            or sample.get("selected_compute_apps_before") != []
            or sample.get("selected_compute_apps_after") != []
        ):
            raise ValueError(
                "production pre-load GPU idle gate observed selected compute apps"
            )
        for field in ("memory_before", "memory_after"):
            memory = sample.get(field)
            if not isinstance(memory, list) or len(memory) != len(bindings):
                raise ValueError(
                    "production pre-load GPU idle-gate memory coverage mismatch"
                )
            if any(not isinstance(item, Mapping) for item in memory):
                raise ValueError(
                    "production pre-load GPU idle-gate memory rows are malformed"
                )
            for item, binding in zip(memory, bindings, strict=True):
                identity = {
                    key: item.get(key)
                    for key in ("logical_index", "physical_index", "gpu_uuid")
                }
                if (
                    set(item)
                    != {
                        "logical_index",
                        "physical_index",
                        "gpu_uuid",
                        "memory_used_mib",
                    }
                    or identity != dict(binding)
                    or type(item.get("memory_used_mib")) is not int
                    or item["memory_used_mib"] < 0
                    or item["memory_used_mib"] > maximum
                ):
                    raise ValueError(
                        "production pre-load GPU idle-gate memory binding/limit mismatch"
                    )
    return evidence


def bind_preload_gpu_idle_gate(
    args: argparse.Namespace,
    run_definition: Mapping[str, Any],
    selected_devices: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Bind fresh gate evidence, or reuse it only to verify an existing run."""

    policy = preload_gpu_idle_gate_policy(args, selected_devices)
    existing_path = (
        args.output_dir.resolve() / "run_definition.json"
        if args.output_dir is not None
        else None
    )
    if existing_path is not None and existing_path.is_file():
        stored = _strict_json(existing_path)
        stored_without_fingerprint = dict(stored)
        stored_without_fingerprint.pop("request_fingerprint", None)
        evidence = stored_without_fingerprint.pop("preload_gpu_idle_gate", None)
        if stored_without_fingerprint != dict(run_definition):
            raise ValueError("latency resume request fingerprint/definition mismatch")
        verified = verify_preload_gpu_idle_gate_evidence(
            evidence, expected_policy=policy
        )
        return (
            {**dict(run_definition), "preload_gpu_idle_gate": verified},
            verified,
            False,
        )
    evidence = wait_for_preload_gpu_idle_gate(args, selected_devices)
    verified = verify_preload_gpu_idle_gate_evidence(evidence, expected_policy=policy)
    return {**dict(run_definition), "preload_gpu_idle_gate": verified}, verified, True


def hardware_snapshot(device_ids: Sequence[int]) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    result: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "cuda_visible_devices_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_cuda_devices": torch.cuda.device_count() if cuda else 0,
        "selected_devices": list(device_ids),
        "local_topology_convention": {
            "accelerators": "8x NVIDIA GeForce RTX 4090",
            "nvlink": False,
            "cuda_peer_access_expected": False,
            "warning": (
                "The paper used 8xH100 with NVLink; local absolute latency and PCIe "
                "handoff cost are not directly comparable."
            ),
        },
    }
    if cuda:
        result["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    index
                ).total_memory,
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in device_ids
        ]
        result["peer_access"] = {
            f"{left}->{right}": bool(torch.cuda.can_device_access_peer(left, right))
            for left in device_ids
            for right in device_ids
            if left != right
        }
        result["all_selected_pairs_peer_access_disabled"] = not any(
            result["peer_access"].values()
        )
        result["nvidia_smi_topology"] = _command_output(["nvidia-smi", "topo", "-m"])
        result["nvidia_smi_p2p_read"] = _command_output(
            ["nvidia-smi", "topo", "-p2p", "r"]
        )
    return result


def assert_attention_backend(
    model: torch.nn.Module,
    *,
    role: str,
    expected: str,
    maximum_requested_length: int,
) -> dict[str, Any]:
    """Assert the Transformers dispatcher will call the requested FA2 backend."""

    config = model.config
    configured = getattr(config, "_attn_implementation", None)
    configured_internal = getattr(config, "_attn_implementation_internal", None)
    if configured != expected:
        raise RuntimeError(
            f"{role} attention implementation resolved to {configured!r}, expected {expected!r}"
        )
    if int(config.max_position_embeddings) < maximum_requested_length:
        raise RuntimeError(
            f"{role} max_position_embeddings={config.max_position_embeddings} is smaller than "
            f"requested length {maximum_requested_length}"
        )
    base = getattr(model, "model", model)
    layers = tuple(base.layers)
    if not layers:
        raise RuntimeError(f"{role} has no decoder attention layers")
    attention_modules = tuple(layer.self_attn for layer in layers)
    layer_implementations = {
        getattr(module.config, "_attn_implementation", None)
        for module in attention_modules
    }
    if layer_implementations != {expected}:
        raise RuntimeError(
            f"{role} decoder layers disagree on attention backend: {layer_implementations}"
        )
    dispatch_names = set(
        type(module).forward.__code__.co_names for module in attention_modules
    )
    if len(dispatch_names) != 1 or not all(
        "ALL_ATTENTION_FUNCTIONS" in names and "_attn_implementation" in names
        for names in dispatch_names
    ):
        raise RuntimeError(
            f"{role} attention module does not use the Transformers backend dispatcher"
        )

    backend = ALL_ATTENTION_FUNCTIONS.get(expected)
    backend_module = getattr(backend, "__module__", None)
    backend_name = getattr(backend, "__name__", None)
    if expected == "flash_attention_2":
        if not is_flash_attn_2_available():
            raise RuntimeError("Transformers reports flash_attention_2 unavailable")
        if (
            backend is None
            or backend_module != "transformers.integrations.flash_attention"
            or (backend_name != "flash_attention_forward")
        ):
            raise RuntimeError(
                "flash_attention_2 did not resolve to transformers.integrations.flash_attention."
                "flash_attention_forward"
            )
    return {
        "status": "passed",
        "role": role,
        "expected": expected,
        "config_attn_implementation": configured,
        "config_attn_implementation_internal": configured_internal,
        "decoder_layer_count": len(attention_modules),
        "attention_module_classes": sorted(
            {type(module).__qualname__ for module in attention_modules}
        ),
        "dispatcher_verified_for_every_layer": True,
        "backend_callable_module": backend_module,
        "backend_callable_name": backend_name,
        "flash_attention_2_available": bool(is_flash_attn_2_available()),
        "max_position_embeddings": int(config.max_position_embeddings),
        "maximum_requested_length": int(maximum_requested_length),
    }


def runtime_attention_contract(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    *,
    mode: str,
    expected: str,
    lengths: Sequence[int],
) -> dict[str, Any]:
    if mode == "tiny":
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "skipped",
            "reason": "tiny diagnostic uses an in-memory eager CPU/CUDA model",
            "requested_production_backend": expected,
        }
    maximum = max(lengths)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "source": assert_attention_backend(
            source_model,
            role="source",
            expected=expected,
            maximum_requested_length=maximum,
        ),
        "target": assert_attention_backend(
            target_model,
            role="target",
            expected=expected,
            maximum_requested_length=maximum,
        ),
        "persisted_before_any_length": True,
        "covers_32768_before_execution": 32768 not in lengths or maximum >= 32768,
    }


def staged_runtime_attention_contract(
    *,
    source_assertion: Mapping[str, Any],
    target_model: torch.nn.Module,
    expected: str,
    lengths: Sequence[int],
) -> dict[str, Any]:
    """Compose FA2 evidence after the source model was deliberately released."""

    maximum = max(lengths)
    source = dict(source_assertion)
    if (
        source.get("status") != "passed"
        or source.get("role") != "source"
        or source.get("expected") != expected
        or source.get("maximum_requested_length") != maximum
    ):
        raise ValueError("staged source attention assertion is not reusable")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "source": source,
        "target": assert_attention_backend(
            target_model,
            role="target",
            expected=expected,
            maximum_requested_length=maximum,
        ),
        "persisted_before_any_length": True,
        "covers_32768_before_execution": 32768 not in lengths or maximum >= 32768,
        "source_model_released_before_target_model_load": True,
        "source_attention_asserted_during_untimed_cache_preparation": True,
    }


def telemetry_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "gpu": _command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
                "utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        ),
        "compute_processes": _command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def peak_memory_baseline(device_ids: Sequence[int]) -> dict[str, dict[str, int]]:
    if not torch.cuda.is_available():
        return {}
    return {
        str(index): {
            "allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.memory_reserved(index)),
        }
        for index in device_ids
    }


def _strict_cuda_device_ids(value: Sequence[int], *, label: str) -> tuple[int, ...]:
    try:
        selected = tuple(value)
    except TypeError as exc:
        raise ValueError(
            f"{label} must be a sequence of exact integer CUDA IDs"
        ) from exc
    if any(type(index) is not int for index in selected):
        raise ValueError(f"{label} must contain exact integers, not bools or floats")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{label} must be unique")
    if any(index < 0 for index in selected):
        raise ValueError(f"{label} must be non-negative")
    return selected


def _validate_cuda_peak_binding(
    *, prefix_length: Any, request_fingerprint: Any
) -> None:
    if type(prefix_length) is not int or prefix_length <= 0:
        raise ValueError(
            "CUDA peak-memory prefix_length must be an exact positive integer"
        )
    if (
        type(request_fingerprint) is not str
        or len(request_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in request_fingerprint)
    ):
        raise ValueError(
            "CUDA peak-memory request_fingerprint must be 64 lowercase hex digits"
        )


def begin_cuda_peak_memory_measurement(
    device_ids: Sequence[int],
    *,
    prefix_length: int,
    request_fingerprint: str,
    measurement_scope: str = CUDA_PEAK_MEASUREMENT_SCOPE,
) -> dict[str, Any]:
    """Synchronize, snapshot, then reset allocator peaks for one length cell.

    The PyTorch peak counters are process-local.  CUDA ``mem_get_info`` snapshots
    are also persisted so the checkpoint exposes whole-device free/total memory
    immediately before and after the cell.  There is no CUDA API for historical
    minimum driver-free memory, so the latter must not be presented as a peak.
    """

    selected = _strict_cuda_device_ids(device_ids, label="CUDA peak-memory device IDs")
    _validate_cuda_peak_binding(
        prefix_length=prefix_length,
        request_fingerprint=request_fingerprint,
    )
    if measurement_scope not in (
        CUDA_PEAK_MEASUREMENT_SCOPE,
        STAGED_CUDA_PEAK_MEASUREMENT_SCOPE,
    ):
        raise ValueError("unsupported CUDA peak-memory measurement scope")
    common = {
        "schema_version": CUDA_PEAK_MEMORY_SCHEMA_VERSION,
        "prefix_length": prefix_length,
        "request_fingerprint": request_fingerprint,
        "measurement_scope": measurement_scope,
        "allocator_peak_scope": CUDA_PEAK_ALLOCATOR_SCOPE,
        "driver_free_scope": CUDA_DRIVER_FREE_SCOPE,
        "selected_device_ids": list(selected),
    }
    if not selected:
        return {
            **common,
            "status": "not_applicable",
            "reason": "no CUDA devices selected for this length cell",
            "devices": {},
            "synchronized_before_reset": False,
            "peak_counters_reset_after_baseline": False,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA devices were selected but CUDA is unavailable")

    # All asynchronous work from runtime loading or a preceding length must be
    # complete before the baseline is observed and the counters are reset.
    synchronize_devices(selected)
    devices: dict[str, dict[str, int]] = {}
    for index in selected:
        properties = torch.cuda.get_device_properties(index)
        driver_free, driver_total = torch.cuda.mem_get_info(index)
        allocated = int(torch.cuda.memory_allocated(index))
        reserved = int(torch.cuda.memory_reserved(index))
        devices[str(index)] = {
            "capacity_bytes": int(properties.total_memory),
            "driver_free_bytes_before": int(driver_free),
            "driver_total_bytes_before": int(driver_total),
            "driver_used_bytes_before": int(driver_total - driver_free),
            "allocated_bytes_before": allocated,
            "reserved_bytes_before": reserved,
        }
        # reset_peak_memory_stats sets both allocator peaks to the current
        # synchronized allocation/reservation watermark.
        torch.cuda.reset_peak_memory_stats(index)
        peak_allocated_after_reset = int(torch.cuda.max_memory_allocated(index))
        peak_reserved_after_reset = int(torch.cuda.max_memory_reserved(index))
        if (
            peak_allocated_after_reset != allocated
            or peak_reserved_after_reset != reserved
        ):
            raise RuntimeError(
                f"CUDA peak-memory reset did not land on the baseline for device {index}"
            )
        devices[str(index)][
            "peak_allocated_bytes_immediately_after_reset"
        ] = peak_allocated_after_reset
        devices[str(index)][
            "peak_reserved_bytes_immediately_after_reset"
        ] = peak_reserved_after_reset
    return {
        **common,
        "status": "measuring",
        "devices": devices,
        "synchronized_before_reset": True,
        "peak_counters_reset_after_baseline": True,
    }


def finish_cuda_peak_memory_measurement(state: Mapping[str, Any]) -> dict[str, Any]:
    """Synchronize and turn a started measurement into checkpoint evidence."""

    if (
        type(state.get("schema_version")) is not int
        or state.get("schema_version") != CUDA_PEAK_MEMORY_SCHEMA_VERSION
    ):
        raise ValueError("CUDA peak-memory state has an unsupported schema")
    _validate_cuda_peak_binding(
        prefix_length=state.get("prefix_length"),
        request_fingerprint=state.get("request_fingerprint"),
    )
    measurement_scope = state.get("measurement_scope")
    if measurement_scope not in (
        CUDA_PEAK_MEASUREMENT_SCOPE,
        STAGED_CUDA_PEAK_MEASUREMENT_SCOPE,
    ):
        raise ValueError("CUDA peak-memory state has invalid measurement_scope")
    for field, expected in (
        ("allocator_peak_scope", CUDA_PEAK_ALLOCATOR_SCOPE),
        ("driver_free_scope", CUDA_DRIVER_FREE_SCOPE),
    ):
        if state.get(field) != expected:
            raise ValueError(f"CUDA peak-memory state has invalid {field}")
    if state.get("status") == "not_applicable":
        return {
            **dict(state),
            "synchronized_after_cell": False,
        }
    if state.get("status") != "measuring":
        raise ValueError(
            "CUDA peak-memory state was not started by this implementation"
        )
    selected_raw = state.get("selected_device_ids")
    devices_raw = state.get("devices")
    if not isinstance(selected_raw, list) or not isinstance(devices_raw, Mapping):
        raise ValueError("CUDA peak-memory state is malformed")
    selected = _strict_cuda_device_ids(
        selected_raw,
        label="CUDA peak-memory state selected_device_ids",
    )

    # Peaks are meaningful only after every selected stream has completed the
    # full length cell.  run_one_length may empty the cache during cleanup, but
    # the reset peak counters retain the high-water marks.
    synchronize_devices(selected)
    devices: dict[str, dict[str, Any]] = {}
    for index in selected:
        baseline = devices_raw.get(str(index))
        if not isinstance(baseline, Mapping):
            raise ValueError(f"missing CUDA peak-memory baseline for device {index}")
        driver_free, driver_total = torch.cuda.mem_get_info(index)
        allocated_after = int(torch.cuda.memory_allocated(index))
        reserved_after = int(torch.cuda.memory_reserved(index))
        peak_allocated = int(torch.cuda.max_memory_allocated(index))
        peak_reserved = int(torch.cuda.max_memory_reserved(index))
        allocated_before = int(baseline["allocated_bytes_before"])
        reserved_before = int(baseline["reserved_bytes_before"])
        capacity = int(baseline["capacity_bytes"])
        if peak_allocated < max(allocated_before, allocated_after):
            raise RuntimeError(
                f"CUDA allocated peak is below an endpoint on device {index}"
            )
        if peak_reserved < max(reserved_before, reserved_after, peak_allocated):
            raise RuntimeError(
                f"CUDA reserved peak is below an endpoint on device {index}"
            )
        if peak_reserved > capacity:
            raise RuntimeError(
                f"CUDA reserved peak exceeds device capacity on device {index}"
            )
        devices[str(index)] = {
            **dict(baseline),
            "driver_free_bytes_after": int(driver_free),
            "driver_total_bytes_after": int(driver_total),
            "driver_used_bytes_after": int(driver_total - driver_free),
            "allocated_bytes_after": allocated_after,
            "reserved_bytes_after": reserved_after,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "allocated_growth_to_peak_bytes": peak_allocated - allocated_before,
            "reserved_growth_to_peak_bytes": peak_reserved - reserved_before,
            "capacity_headroom_at_peak_reserved_bytes": capacity - peak_reserved,
            "peak_allocated_fraction_of_capacity": peak_allocated / capacity,
            "peak_reserved_fraction_of_capacity": peak_reserved / capacity,
        }
    return {
        **dict(state),
        "status": "captured",
        "devices": devices,
        "synchronized_after_cell": True,
    }


def cuda_peak_memory_capture_failure(
    state: Mapping[str, Any],
    error: BaseException,
    *,
    stage: str,
) -> dict[str, Any]:
    """Build explicit, non-accepting evidence when final peak capture fails."""

    return {
        "schema_version": CUDA_PEAK_MEMORY_SCHEMA_VERSION,
        "status": "capture_failed",
        "capture_failure_stage": stage,
        "prefix_length": state.get("prefix_length"),
        "request_fingerprint": state.get("request_fingerprint"),
        "measurement_scope": state.get("measurement_scope"),
        "allocator_peak_scope": state.get("allocator_peak_scope"),
        "driver_free_scope": state.get("driver_free_scope"),
        "selected_device_ids": state.get("selected_device_ids"),
        "devices_baseline": state.get("devices"),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def verify_cuda_peak_memory_evidence(
    value: Any,
    *,
    expected_device_ids: Sequence[int],
    expected_prefix_length: int,
    expected_request_fingerprint: str,
    expected_measurement_scope: str = CUDA_PEAK_MEASUREMENT_SCOPE,
) -> dict[str, Any]:
    """Strictly validate the per-length CUDA memory gate persisted in a checkpoint."""

    if not isinstance(value, Mapping):
        raise ValueError("latency checkpoint lacks CUDA peak-memory evidence")
    expected = _strict_cuda_device_ids(
        expected_device_ids,
        label="expected CUDA peak-memory device IDs",
    )
    _validate_cuda_peak_binding(
        prefix_length=expected_prefix_length,
        request_fingerprint=expected_request_fingerprint,
    )
    if (
        type(value.get("schema_version")) is not int
        or value.get("schema_version") != CUDA_PEAK_MEMORY_SCHEMA_VERSION
    ):
        raise ValueError("CUDA peak-memory evidence has an unsupported schema")
    if value.get("prefix_length") != expected_prefix_length:
        raise ValueError(
            "CUDA peak-memory evidence prefix_length does not match the checkpoint"
        )
    if value.get("request_fingerprint") != expected_request_fingerprint:
        raise ValueError(
            "CUDA peak-memory evidence request_fingerprint does not match the run"
        )
    selected_raw = value.get("selected_device_ids")
    if not isinstance(selected_raw, list):
        raise ValueError("CUDA peak-memory evidence selected_device_ids must be a list")
    selected = _strict_cuda_device_ids(
        selected_raw,
        label="CUDA peak-memory evidence selected_device_ids",
    )
    if selected != expected:
        raise ValueError(
            "CUDA peak-memory evidence selected devices do not match the run"
        )
    if expected_measurement_scope not in (
        CUDA_PEAK_MEASUREMENT_SCOPE,
        STAGED_CUDA_PEAK_MEASUREMENT_SCOPE,
    ):
        raise ValueError("unsupported expected CUDA peak-memory measurement scope")
    for field, expected_scope in (
        ("measurement_scope", expected_measurement_scope),
        ("allocator_peak_scope", CUDA_PEAK_ALLOCATOR_SCOPE),
        ("driver_free_scope", CUDA_DRIVER_FREE_SCOPE),
    ):
        if value.get(field) != expected_scope:
            raise ValueError(f"CUDA peak-memory evidence has invalid {field}")
    devices = value.get("devices")
    if not isinstance(devices, Mapping):
        raise ValueError("CUDA peak-memory evidence has malformed devices")
    if not expected:
        if (
            value.get("status") != "not_applicable"
            or value.get("reason") != "no CUDA devices selected for this length cell"
            or devices
            or value.get("synchronized_before_reset") is not False
            or value.get("peak_counters_reset_after_baseline") is not False
            or value.get("synchronized_after_cell") is not False
        ):
            raise ValueError(
                "CPU latency checkpoint has invalid CUDA peak-memory evidence"
            )
        return dict(value)
    if value.get("status") != "captured":
        raise ValueError("CUDA peak-memory evidence was not captured")
    if value.get("synchronized_before_reset") is not True:
        raise ValueError("CUDA peak-memory counters were not synchronized before reset")
    if value.get("peak_counters_reset_after_baseline") is not True:
        raise ValueError("CUDA peak-memory counters were not reset after the baseline")
    if value.get("synchronized_after_cell") is not True:
        raise ValueError(
            "CUDA peak-memory counters were not synchronized after the cell"
        )
    if set(devices) != {str(index) for index in expected}:
        raise ValueError(
            "CUDA peak-memory evidence does not cover every selected device"
        )

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
    for index in expected:
        item = devices[str(index)]
        if not isinstance(item, Mapping):
            raise ValueError(f"CUDA peak-memory device {index} evidence is malformed")
        for field in integer_fields:
            field_value = item.get(field)
            if type(field_value) is not int or field_value < 0:
                raise ValueError(
                    f"CUDA peak-memory device {index} has invalid non-negative integer {field}"
                )
        capacity = int(item["capacity_bytes"])
        if capacity <= 0:
            raise ValueError(f"CUDA peak-memory device {index} has zero capacity")
        for suffix in ("before", "after"):
            free = int(item[f"driver_free_bytes_{suffix}"])
            total = int(item[f"driver_total_bytes_{suffix}"])
            used = int(item[f"driver_used_bytes_{suffix}"])
            allocated = int(item[f"allocated_bytes_{suffix}"])
            reserved = int(item[f"reserved_bytes_{suffix}"])
            if free > total or used != total - free:
                raise ValueError(
                    f"CUDA driver memory arithmetic failed on device {index}"
                )
            if allocated > reserved:
                raise ValueError(
                    f"CUDA allocator endpoint arithmetic failed on device {index}"
                )
        allocated_before = int(item["allocated_bytes_before"])
        reserved_before = int(item["reserved_bytes_before"])
        if item["peak_allocated_bytes_immediately_after_reset"] != allocated_before:
            raise ValueError(
                f"CUDA allocated peak reset is inconsistent on device {index}"
            )
        if item["peak_reserved_bytes_immediately_after_reset"] != reserved_before:
            raise ValueError(
                f"CUDA reserved peak reset is inconsistent on device {index}"
            )
        allocated_after = int(item["allocated_bytes_after"])
        reserved_after = int(item["reserved_bytes_after"])
        peak_allocated = int(item["peak_allocated_bytes"])
        peak_reserved = int(item["peak_reserved_bytes"])
        if peak_allocated < max(allocated_before, allocated_after):
            raise ValueError(f"CUDA allocated peak is inconsistent on device {index}")
        if peak_reserved < max(reserved_before, reserved_after, peak_allocated):
            raise ValueError(f"CUDA reserved peak is inconsistent on device {index}")
        if peak_reserved > capacity:
            raise ValueError(f"CUDA reserved peak exceeds capacity on device {index}")
        if item["allocated_growth_to_peak_bytes"] != peak_allocated - allocated_before:
            raise ValueError(
                f"CUDA allocated peak growth is inconsistent on device {index}"
            )
        if item["reserved_growth_to_peak_bytes"] != peak_reserved - reserved_before:
            raise ValueError(
                f"CUDA reserved peak growth is inconsistent on device {index}"
            )
        if item["capacity_headroom_at_peak_reserved_bytes"] != capacity - peak_reserved:
            raise ValueError(f"CUDA peak headroom is inconsistent on device {index}")
        for numerator_field, fraction_field in (
            ("peak_allocated_bytes", "peak_allocated_fraction_of_capacity"),
            ("peak_reserved_bytes", "peak_reserved_fraction_of_capacity"),
        ):
            fraction = item.get(fraction_field)
            if not isinstance(fraction, (int, float)) or not math.isfinite(
                float(fraction)
            ):
                raise ValueError(f"CUDA peak fraction is invalid on device {index}")
            expected_fraction = int(item[numerator_field]) / capacity
            if not math.isclose(
                float(fraction), expected_fraction, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(
                    f"CUDA peak fraction is inconsistent on device {index}"
                )
    return dict(value)


def tiny_model(device: torch.device) -> Qwen3ForCausalLM:
    torch.manual_seed(20260813)
    config = Qwen3Config(
        vocab_size=127,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=None,
        _attn_implementation="eager",
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def tiny_identity_mapper(model: Qwen3ForCausalLM) -> ResidentMapperBundle:
    config = model.config
    architecture = ArchitectureSpec(
        source_layers=config.num_hidden_layers,
        target_layers=config.num_hidden_layers,
        source_kv_heads=config.num_key_value_heads,
        target_kv_heads=config.num_key_value_heads,
        source_head_dim=config.head_dim,
        target_head_dim=config.head_dim,
        selected_source_layers=1,
    )
    device = next(model.parameters()).device
    features = architecture.target_features
    identity = torch.eye(features, device=device, dtype=torch.float32)
    layers = tuple(
        LayerMapper(
            source_layers=(index,),
            key_weight=identity.clone(),
            key_bias=torch.zeros(features, device=device, dtype=torch.float32),
            value_weight=identity.clone(),
            value_bias=torch.zeros(features, device=device, dtype=torch.float32),
        )
        for index in range(architecture.target_layers)
    )
    return ResidentMapperBundle(architecture=architecture, layers=layers)


def deterministic_ids(
    length: int,
    *,
    vocabulary_size: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + length)
    # Avoid control-token-heavy IDs at the bottom of the vocabulary while
    # preserving an exact synthetic token length.
    low = min(100, max(vocabulary_size - 1, 0))
    if low >= vocabulary_size:
        low = 0
    return torch.randint(low, vocabulary_size, (1, length + 1), generator=generator)


@torch.inference_mode()
def build_source_cache(source_model: torch.nn.Module, prefix_ids: torch.Tensor):
    """Create the already-existing source cache outside every timed region."""

    base = getattr(source_model, "model", source_model)
    device = base.embed_tokens.weight.device
    output = base(
        input_ids=prefix_ids.to(device),
        use_cache=True,
        return_dict=True,
    )
    cache = output.past_key_values
    if cache_sequence_length(cache) != prefix_ids.shape[1]:
        raise RuntimeError("source cache length differs from requested prefix length")
    del output
    return cache


def deterministic_prefix_identity(prefix_ids: torch.Tensor) -> dict[str, Any]:
    """Bind one synthetic prefix without hashing any model-generated cache bytes."""

    if prefix_ids.device.type != "cpu" or prefix_ids.dtype != torch.int64:
        raise ValueError("deterministic prefix identity requires CPU int64 token IDs")
    contiguous = prefix_ids.detach().contiguous()
    payload = memoryview(contiguous.numpy()).cast("B")
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "device": "cpu",
        "nbytes": contiguous.numel() * contiguous.element_size(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _cache_raw_sha256(cache: Any) -> str:
    """Hash cache structure and raw tensor bits with one tensor of host scratch."""

    digest = hashlib.sha256()
    pairs = cache_to_legacy(cache)
    for layer, (key, value) in enumerate(pairs):
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
    expected_devices: Sequence[torch.device | str],
    expected_length: int,
    expected_dtype: torch.dtype,
) -> tuple[list[dict[str, Any]], int]:
    pairs = cache_to_legacy(cache)
    if len(pairs) != len(expected_devices):
        raise RuntimeError(
            f"source cache has {len(pairs)} layers, expected {len(expected_devices)}"
        )
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for layer, ((key, value), expected_raw) in enumerate(
        zip(pairs, expected_devices, strict=True)
    ):
        expected = torch.device(expected_raw)
        if (
            key.device != expected
            or value.device != expected
            or key.dtype != expected_dtype
            or value.dtype != expected_dtype
            or key.shape != value.shape
            or key.ndim != 4
            or key.shape[0] != 1
            or key.shape[1] != PAPER_ARCHITECTURE.source_kv_heads
            or key.shape[2] != expected_length
            or key.shape[3] != PAPER_ARCHITECTURE.source_head_dim
        ):
            raise RuntimeError(f"source cache layer {layer} violates staging contract")
        key_bytes = key.numel() * key.element_size()
        value_bytes = value.numel() * value.element_size()
        total_bytes += key_bytes + value_bytes
        records.append(
            {
                "layer": layer,
                "device": str(expected),
                "key_shape": list(key.shape),
                "value_shape": list(value.shape),
                "key_dtype": str(key.dtype),
                "value_dtype": str(value.dtype),
                "key_nbytes": key_bytes,
                "value_nbytes": value_bytes,
            }
        )
    return records, total_bytes


def source_cache_exact_bytes(length: int) -> int:
    if type(length) is not int or length <= 0:
        raise ValueError("source cache length must be an exact positive integer")
    return (
        PAPER_ARCHITECTURE.source_layers
        * 2
        * PAPER_ARCHITECTURE.source_kv_heads
        * length
        * PAPER_ARCHITECTURE.source_head_dim
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )


def staged_runtime_topology(args: argparse.Namespace) -> dict[str, Any]:
    """Return the exact three-topology v9 contract, including physical binding."""

    if not args.stage_source_caches_on_cpu_before_cells:
        raise ValueError("staged runtime topology requested for a non-staged run")
    physical = tuple(int(item) for item in STAGED_CUDA_VISIBLE_DEVICES.split(","))
    logical_to_physical = {
        str(logical): device for logical, device in enumerate(physical)
    }
    result = {
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "cuda_visible_devices_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_to_physical": logical_to_physical,
        "source_model_prefill": {
            "logical_devices": list(args.source_prefill_devices),
            "physical_devices": [
                physical[device] for device in args.source_prefill_devices
            ],
            "layer_counts": list(args.source_prefill_layer_counts),
        },
        "source_cache_staging": {
            "logical_devices": list(args.source_devices),
            "physical_devices": [physical[device] for device in args.source_devices],
            "layer_counts": list(args.source_layer_counts),
        },
        "target_timed": {
            "logical_devices": list(args.target_devices),
            "physical_devices": [physical[device] for device in args.target_devices],
            "layer_counts": list(args.target_layer_counts),
        },
        "source_prefill_target_logical_overlap": [5],
        "source_prefill_target_physical_overlap": [7],
        "source_cache_target_logical_overlap": [],
        "source_cache_target_physical_overlap": [],
        "logical_5_unique_sequential_reuse_physical_device": 7,
    }
    expected = {
        "source_model_prefill": {
            "logical_devices": list(STAGED_SOURCE_MODEL_PREFILL_DEVICES),
            "physical_devices": [7, 0, 1],
            "layer_counts": list(STAGED_SOURCE_MODEL_PREFILL_LAYER_COUNTS),
        },
        "source_cache_staging": {
            "logical_devices": list(STAGED_SOURCE_CACHE_DEVICES),
            "physical_devices": [0, 1],
            "layer_counts": list(STAGED_SOURCE_CACHE_LAYER_COUNTS),
        },
        "target_timed": {
            "logical_devices": list(STAGED_TARGET_DEVICES),
            "physical_devices": [2, 3, 4, 5, 6, 7],
            "layer_counts": list(STAGED_TARGET_LAYER_COUNTS),
        },
    }
    for name, contract in expected.items():
        if result[name] != contract:
            raise RuntimeError(f"staged v9 {name} topology changed")
    return result


def host_memory_staging_gate(
    *,
    lengths: Sequence[int],
    source_weight_bytes: int,
    target_weight_bytes: int,
) -> dict[str, Any]:
    """Fail before model load unless host RAM covers the conservative staging plan."""

    fields: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="utf-8") as handle:
        for line in handle:
            name, raw = line.split(":", 1)
            pieces = raw.strip().split()
            if len(pieces) >= 2 and pieces[1] == "kB":
                fields[name] = int(pieces[0]) * 1024
    available = fields.get("MemAvailable")
    if available is None:
        raise RuntimeError("/proc/meminfo does not expose MemAvailable")
    staging_bytes = sum(source_cache_exact_bytes(int(length)) for length in lengths)
    required = (
        int(source_weight_bytes)
        + int(target_weight_bytes)
        + staging_bytes
        + STAGED_HOST_MEMORY_OVERHEAD_BYTES
    )
    if available < required:
        raise RuntimeError(
            "insufficient host MemAvailable for staged source caches: "
            f"{available} < {required}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "scope": "before_source_model_load",
        "mem_available_bytes": available,
        "required_available_bytes": required,
        "source_weight_shard_bytes": int(source_weight_bytes),
        "target_weight_shard_bytes": int(target_weight_bytes),
        "all_requested_source_caches_cpu_bytes": staging_bytes,
        "staged_prefix_lengths": list(lengths),
        "conservative_operating_overhead_bytes": STAGED_HOST_MEMORY_OVERHEAD_BYTES,
    }


def _cuda_allocator_driver_snapshot(device_ids: Sequence[int]) -> dict[str, Any]:
    synchronize_devices(device_ids)
    result: dict[str, Any] = {}
    for device in device_ids:
        driver_free, driver_total = torch.cuda.mem_get_info(device)
        result[str(device)] = {
            "allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "driver_free_bytes": int(driver_free),
            "driver_total_bytes": int(driver_total),
            "driver_used_bytes": int(driver_total - driver_free),
        }
    return result


def clear_cuda_workspaces_and_allocator_cache(
    device_ids: Sequence[int],
) -> dict[str, Any]:
    """Synchronize, clear persistent cuBLAS workspaces, then empty allocators."""

    selected = _strict_cuda_device_ids(
        device_ids, label="CUDA workspace-clear device IDs"
    )
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
    if not callable(clear):
        raise RuntimeError(
            "this frozen Torch runtime lacks _cuda_clearCublasWorkspaces"
        )
    call_order: list[str] = []
    synchronize_devices(selected)
    call_order.append("synchronize_before_cublas_workspace_clear")
    clear()
    call_order.append("torch._C._cuda_clearCublasWorkspaces")
    gc.collect()
    call_order.append("gc_collect_after_cublas_workspace_clear")
    for device in selected:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    call_order.append("torch.cuda.empty_cache_each_selected_device")
    synchronize_devices(selected)
    call_order.append("synchronize_after_empty_cache")
    return {
        "status": "passed",
        "selected_device_ids": list(selected),
        "cublas_workspaces_clear_api": "torch._C._cuda_clearCublasWorkspaces",
        "cublas_workspaces_clear_called": True,
        "cublas_workspaces_clear_succeeded": True,
        "gc_completed_after_workspace_clear": True,
        "empty_cache_completed_for_every_selected_device": True,
        "synchronized_before_and_after": True,
        "call_order": call_order,
    }


def prove_source_model_released_before_target_load(
    *,
    source_model_ref: weakref.ReferenceType[Any],
    source_prefill_devices: Sequence[int],
    allocator_before_release: Mapping[str, Any],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove the source model and every source GPU cache reference are gone."""

    gc.collect()
    if source_model_ref() is not None:
        raise RuntimeError("source model remains strongly referenced before target load")
    workspace_release = clear_cuda_workspaces_and_allocator_cache(
        source_prefill_devices
    )
    workspace_release["source_model_deleted_and_gc_completed_before_clear"] = True
    workspace_release["call_order"] = [
        "source_model_deleted_and_gc_collected",
        *workspace_release["call_order"],
    ]
    gc.collect()
    if source_model_ref() is not None:
        raise RuntimeError("source model remains strongly referenced before target load")
    after = _cuda_allocator_driver_snapshot(source_prefill_devices)
    nonzero = {
        device: {
            "allocated_bytes": item["allocated_bytes"],
            "reserved_bytes": item["reserved_bytes"],
        }
        for device, item in after.items()
        if item["allocated_bytes"] != 0 or item["reserved_bytes"] != 0
    }
    if nonzero:
        raise RuntimeError(
            "source-prefill CUDA allocator was not fully empty before target load: "
            f"{nonzero}"
        )
    return {
        "status": "passed",
        "source_model_live": False,
        "source_model_weakref_is_none": True,
        "all_source_gpu_cache_references_deleted": True,
        "gc_completed": True,
        "cuda_synchronized_and_empty_cache_completed": True,
        "cuda_workspace_and_allocator_release": workspace_release,
        "source_prefill_logical_devices": list(source_prefill_devices),
        "allocator_driver_before_release": dict(allocator_before_release),
        "allocator_driver_after_release_before_target_load": after,
        "all_prefill_devices_allocated_bytes_zero": True,
        "all_prefill_devices_reserved_bytes_zero": True,
        "target_model_loaded_at_proof_time": False,
        "source_prefill_target_overlap_exact": [5],
        "logical_5_reused_physical_device": 7,
        "runtime_topology_sha256": _canonical_digest(topology),
    }


@torch.inference_mode()
def stage_source_cache_on_cpu(
    source_cache: Any,
    *,
    length: int,
    prefix_identity: Mapping[str, Any],
    source_prefill_devices: Sequence[int],
    source_prefill_layer_counts: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    """Clone one completed BF16 source cache to CPU outside every timed cell."""

    prefill_layer_devices = tuple(
        torch.device("cuda", device)
        for device, count in zip(
            source_prefill_devices, source_prefill_layer_counts, strict=True
        )
        for _ in range(count)
    )
    gpu_structure, gpu_bytes = _cache_structure(
        source_cache,
        expected_devices=prefill_layer_devices,
        expected_length=length,
        expected_dtype=torch.bfloat16,
    )
    source_gpu_raw_sha256 = _cache_raw_sha256(source_cache)
    staged = build_dynamic_cache(
        source_cache,
        layer_devices="cpu",
        dtype=torch.bfloat16,
        clone=True,
        detach=True,
        non_blocking=False,
    )
    cpu_structure, cpu_bytes = _cache_structure(
        staged,
        expected_devices=("cpu",) * PAPER_ARCHITECTURE.source_layers,
        expected_length=length,
        expected_dtype=torch.bfloat16,
    )
    if gpu_bytes != cpu_bytes:
        raise RuntimeError("CPU-staged source cache changed its exact byte count")
    expected_bytes = source_cache_exact_bytes(length)
    if cpu_bytes != expected_bytes:
        raise RuntimeError(
            f"CPU-staged source cache has {cpu_bytes} bytes, expected {expected_bytes}"
        )
    raw_sha256 = _cache_raw_sha256(staged)
    if source_gpu_raw_sha256 != raw_sha256:
        raise RuntimeError("CPU-staged source cache differs bitwise from source GPU cache")
    return staged, {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": length,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "copy_semantics": "detach_to_cpu_blocking_exact_clone",
        "full_raw_bit_sha256_computed": True,
        "source_gpu_raw_bit_sha256": source_gpu_raw_sha256,
        "cpu_staged_raw_bit_sha256": raw_sha256,
        "raw_bit_sha256": raw_sha256,
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "deterministic_prefix_identity": dict(prefix_identity),
        "source_prefill_layer_devices": [str(device) for device in prefill_layer_devices],
        "source_prefill_cache_structure": gpu_structure,
        "cpu_cache_structure": cpu_structure,
        "cpu_cache_total_bytes": cpu_bytes,
        "expected_cache_total_bytes": expected_bytes,
    }


@torch.inference_mode()
def restage_source_cache_on_gpus(
    staged_cache: Any,
    *,
    preparation: Mapping[str, Any],
    source_devices: Sequence[int],
    source_layer_counts: Sequence[int],
) -> tuple[Any, dict[str, Any]]:
    """Restore one CPU cache to source-cache GPUs and prove raw-bit identity."""

    length = int(preparation["prefix_length"])
    layer_devices = tuple(
        torch.device("cuda", device)
        for device, count in zip(source_devices, source_layer_counts, strict=True)
        for _ in range(count)
    )
    _cache_structure(
        staged_cache,
        expected_devices=("cpu",) * PAPER_ARCHITECTURE.source_layers,
        expected_length=length,
        expected_dtype=torch.bfloat16,
    )
    restored = build_dynamic_cache(
        staged_cache,
        layer_devices=layer_devices,
        dtype=torch.bfloat16,
        clone=False,
        detach=True,
        non_blocking=False,
    )
    synchronize_devices(source_devices)
    structure, total_bytes = _cache_structure(
        restored,
        expected_devices=layer_devices,
        expected_length=length,
        expected_dtype=torch.bfloat16,
    )
    restored_sha256 = _cache_raw_sha256(restored)
    if restored_sha256 != preparation.get("raw_bit_sha256"):
        raise RuntimeError("GPU-restaged source cache differs bitwise from CPU staging")
    synchronize_devices(source_devices)
    for device in source_devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    synchronize_devices(source_devices)
    return restored, {
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": length,
        "untimed": True,
        "completed_before_outer_peak_reset": True,
        "source_model_live": False,
        "source_cache_gpu_resident_before_mapper_timer": True,
        "copy_semantics": "cpu_to_cuda_blocking_no_dtype_cast",
        "raw_bit_sha256": restored_sha256,
        "raw_bit_hash_matches_cpu_staging": True,
        "layer_devices": [str(device) for device in layer_devices],
        "layer_counts": list(source_layer_counts),
        "cache_structure": structure,
        "cache_total_bytes": total_bytes,
        "cuda_synchronized_and_empty_cache_before_outer_peak_reset": True,
    }


@torch.inference_mode()
def strict_transfer_preflight(
    source_cache: Any,
    *,
    source_config: Any,
    target_model: torch.nn.Module,
    mapper: ResidentMapperBundle,
    token_chunk: int,
) -> dict[str, Any]:
    """Untimed numerical gate required before using the no-reduction fast path."""

    devices = tuple(key.device for key, _ in cache_to_legacy(source_cache)) + tuple(
        next(layer.parameters()).device for layer in target_model.model.layers
    )
    synchronize_devices(devices)
    started = time.perf_counter_ns()
    mapped = transfer_source_cache(
        source_cache,
        source_config=source_config,
        target_model=target_model,
        mapper_bundle=mapper,  # type: ignore[arg-type]
        token_chunk=token_chunk,
        strict_validation=True,
    )
    finite_layers = 0
    for key, value in cache_to_legacy(mapped):
        if not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all()):
            raise RuntimeError("strict transfer preflight produced NaN/Inf")
        finite_layers += 1
    synchronize_devices(devices)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    source_length = cache_sequence_length(source_cache)
    mapped_length = cache_sequence_length(mapped)
    if mapped_length != source_length:
        raise RuntimeError("strict transfer preflight changed the cache length")
    output_dtypes = sorted({str(key.dtype) for key, _ in cache_to_legacy(mapped)})
    del mapped
    synchronize_devices(devices)
    return {
        "status": "passed",
        "untimed": True,
        "strict_validation": True,
        "rope_singularity_reduction": True,
        "source_content_finite_reduction": True,
        "mapped_kv_finite_reduction": True,
        "validated_target_layers": finite_layers,
        "source_cache_length": source_length,
        "mapped_cache_length": mapped_length,
        "mapped_cache_dtypes": output_dtypes,
        "wall_ms_diagnostic_not_latency": float(elapsed_ms),
    }


def run_one_length(
    *,
    length: int,
    source_cache: Any,
    source_config: Any,
    prefix_ids: torch.Tensor,
    boundary_ids: torch.Tensor,
    target_model: torch.nn.Module,
    mapper: ResidentMapperBundle,
    warmup: int,
    repeats: int,
    profile_warmup: int,
    profile_repeats: int,
    token_chunk: int,
    supplemental_diagnostics: bool = False,
    collect_internal_cuda_peak_memory: bool = True,
    expected_prefix_identity: Mapping[str, Any] | None = None,
    source_model_live_during_timed_cell: bool = True,
) -> dict[str, Any]:
    if int(source_config.vocab_size) != int(target_model.config.vocab_size):
        raise ValueError(
            "source and target vocabularies differ; exact shared token IDs are required"
        )
    if (
        prefix_ids.device.type != "cpu"
        or prefix_ids.dtype != torch.int64
        or tuple(prefix_ids.shape) != (1, length)
        or boundary_ids.device.type != "cpu"
        or boundary_ids.dtype != torch.int64
        or tuple(boundary_ids.shape) != (1, 1)
    ):
        raise ValueError("length cell requires exact CPU int64 prefix/boundary IDs")
    prefix_identity = deterministic_prefix_identity(prefix_ids)
    if (
        expected_prefix_identity is not None
        and prefix_identity != dict(expected_prefix_identity)
    ):
        raise RuntimeError(
            "timed-cell deterministic prefix differs from source-cache preparation"
        )

    # This is the paper's premise and a hard timing boundary: callers supply
    # an already-existing source cache.  This function has no source-model
    # reference and therefore cannot accidentally execute source prefill.
    if cache_sequence_length(source_cache) != length:
        raise RuntimeError("preexisting source cache length differs from the cell")
    synchronize_devices(tuple(key.device for key in source_cache.key_cache))

    strict_preflight = strict_transfer_preflight(
        source_cache,
        source_config=source_config,
        target_model=target_model,
        mapper=mapper,
        token_chunk=token_chunk,
    )

    target_reprefill = benchmark_target_reprefill(
        target_model,
        prefix_ids,
        warmup=warmup,
        repeats=repeats,
        collect_cuda_peak_memory=collect_internal_cuda_peak_memory,
    )
    mapper_end_to_end = benchmark_mapper_end_to_end(
        source_cache,
        source_config=source_config,
        target_model=target_model,
        mapper_bundle=mapper,
        token_chunk=token_chunk,
        warmup=warmup,
        repeats=repeats,
        strict_validation=False,
        collect_cuda_peak_memory=collect_internal_cuda_peak_memory,
    )
    speedup = target_reprefill["median"] / mapper_end_to_end["median"]
    result = {
        "prefix_length": length,
        "batch_size": 1,
        "source_prefill_timed": False,
        "source_model_live_during_timed_cell": source_model_live_during_timed_cell,
        "source_cache_origin": "preexisting_before_run_one_length",
        "deterministic_prefix_identity": prefix_identity,
        "source_cache_length": cache_sequence_length(source_cache),
        "latency_protocol": {
            "name": (
                "paper_main_plus_supplemental"
                if supplemental_diagnostics
                else "paper_main_only"
            ),
            "paper_main_metrics": [
                "target_reprefill",
                "mapper_handoff_end_to_end",
            ],
            "supplemental_diagnostics_enabled": supplemental_diagnostics,
            "component_profile_in_cell": supplemental_diagnostics,
            "boundary_decode_in_cell": supplemental_diagnostics,
            "protocol_complete_in_cell": supplemental_diagnostics,
            "paper_headline_speedup_uses_supplemental_metrics": False,
        },
        "strict_transfer_preflight": strict_preflight,
        "timed_transfer_fast_path": {
            "enabled": True,
            "strict_preflight_required_and_passed": True,
            "strict_validation_argument": False,
            "diagnostic_rope_and_finite_reductions_in_timed_region": False,
            "shape_dtype_architecture_and_output_validation_retained": True,
        },
        "nested_cuda_peak_memory_contract": {
            "outer_cell_peak_measurement_active": not collect_internal_cuda_peak_memory,
            "internal_metric_peak_collection_enabled": collect_internal_cuda_peak_memory,
            "internal_peak_counter_resets_disabled": not collect_internal_cuda_peak_memory,
        },
        "target_reprefill": target_reprefill,
        "mapper_handoff_end_to_end": mapper_end_to_end,
        "median_speedup_target_reprefill_over_mapper": speedup,
    }
    if supplemental_diagnostics:
        mapper_profile = benchmark_mapper_components(
            source_cache,
            source_config=source_config,
            target_model=target_model,
            mapper_bundle=mapper,
            token_chunk=token_chunk,
            warmup=profile_warmup,
            repeats=profile_repeats,
            strict_validation=False,
            collect_cuda_peak_memory=collect_internal_cuda_peak_memory,
        )
        mapped_once = transfer_source_cache(
            source_cache,
            source_config=source_config,
            target_model=target_model,
            mapper_bundle=mapper,  # type: ignore[arg-type]
            token_chunk=token_chunk,
            strict_validation=False,
        )
        synchronize_devices(tuple(key.device for key in mapped_once.key_cache))
        boundary_decode = benchmark_boundary_decode(
            target_model,
            boundary_ids,
            mapped_once,
            warmup=warmup,
            repeats=repeats,
            include_lm_head=False,
            collect_cuda_peak_memory=collect_internal_cuda_peak_memory,
        )
        del mapped_once
        protocol_complete = benchmark_protocol_complete(
            source_cache,
            source_config=source_config,
            target_model=target_model,
            boundary_input_ids=boundary_ids,
            mapper_bundle=mapper,
            token_chunk=token_chunk,
            warmup=warmup,
            repeats=repeats,
            include_lm_head=False,
            strict_validation=False,
            collect_cuda_peak_memory=collect_internal_cuda_peak_memory,
        )
        result.update(
            {
                "mapper_synchronized_component_profile": mapper_profile,
                "mapper_kernel_only_profile": mapper_profile["components"][
                    "mapper_fp32_matmul_and_bias"
                ],
                "boundary_one_token_decode": boundary_decode,
                "protocol_complete": protocol_complete,
                "median_speedup_target_reprefill_over_protocol_complete": (
                    target_reprefill["median"] / protocol_complete["median"]
                ),
            }
        )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _summary_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for length_result in result.get("length_results", []):
        length = length_result["prefix_length"]
        metrics = {
            "target_reprefill": length_result["target_reprefill"],
            "mapper_handoff_end_to_end": length_result["mapper_handoff_end_to_end"],
        }
        if length_result["latency_protocol"]["supplemental_diagnostics_enabled"]:
            metrics.update(
                {
                    "mapper_kernel_only_profile": length_result[
                        "mapper_kernel_only_profile"
                    ],
                    "boundary_one_token_decode": length_result[
                        "boundary_one_token_decode"
                    ],
                    "protocol_complete": length_result["protocol_complete"],
                }
            )
            components = length_result["mapper_synchronized_component_profile"][
                "components"
            ]
            metrics.update(
                {f"profile_{name}": value for name, value in components.items()}
            )
        for metric, summary in metrics.items():
            rows.append(
                {
                    "prefix_length": length,
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
                    "timing_backends": ";".join(summary.get("timing_backends", [])),
                    "source_prefill_included": False,
                }
            )
    return rows


def write_csv(path: Path, result: dict[str, Any]) -> None:
    rows = _summary_rows(result)
    fields = list(rows[0]) if rows else ["prefix_length", "metric"]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_hashed_csv(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    write_csv(path, result)
    return write_sha_sidecar(path)


def checkpoint_prefix_partition(
    run: RunDirectory, lengths: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the only unambiguous resume split: existing prefix + missing suffix."""

    existing: list[int] = []
    missing: list[int] = []
    missing_seen = False
    for length in lengths:
        path = run.path / f"length_{length:05d}.json"
        sidecar = _sha_sidecar_path(path)
        if path.exists() != sidecar.exists():
            raise ValueError(
                f"latency checkpoint/sidecar is only partially present: {path}"
            )
        if path.exists():
            if missing_seen:
                raise ValueError(
                    "latency checkpoints are not a contiguous prefix; refusing an ambiguous resume"
                )
            existing.append(int(length))
        else:
            missing_seen = True
            missing.append(int(length))
    return tuple(existing), tuple(missing)


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def verify_source_cache_preparation_record(
    value: Any,
    *,
    expected_length: int,
    expected_prefill_layer_devices: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("source-cache preparation evidence is malformed")
    expected_fields = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": expected_length,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "copy_semantics": "detach_to_cpu_blocking_exact_clone",
        "full_raw_bit_sha256_computed": True,
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "expected_cache_total_bytes": source_cache_exact_bytes(expected_length),
        "cpu_cache_total_bytes": source_cache_exact_bytes(expected_length),
    }
    for field, expected in expected_fields.items():
        if value.get(field) != expected:
            raise ValueError(
                f"source-cache preparation has invalid {field}: {value.get(field)!r}"
            )
    hashes = (
        value.get("source_gpu_raw_bit_sha256"),
        value.get("cpu_staged_raw_bit_sha256"),
        value.get("raw_bit_sha256"),
    )
    if not all(_is_sha256(item) for item in hashes) or len(set(hashes)) != 1:
        raise ValueError("source-cache GPU/CPU raw-bit hashes do not match")
    prefix = value.get("deterministic_prefix_identity")
    if (
        not isinstance(prefix, Mapping)
        or prefix.get("shape") != [1, expected_length]
        or prefix.get("dtype") != "torch.int64"
        or prefix.get("device") != "cpu"
        or prefix.get("nbytes") != 8 * expected_length
        or not _is_sha256(prefix.get("sha256"))
    ):
        raise ValueError("source-cache preparation has invalid deterministic prefix")
    if value.get("source_prefill_layer_devices") != list(
        expected_prefill_layer_devices
    ):
        raise ValueError("source-cache preparation prefill layer devices changed")
    source_structure = value.get("source_prefill_cache_structure")
    cpu_structure = value.get("cpu_cache_structure")
    if (
        not isinstance(source_structure, list)
        or not isinstance(cpu_structure, list)
        or len(source_structure) != PAPER_ARCHITECTURE.source_layers
        or len(cpu_structure) != PAPER_ARCHITECTURE.source_layers
    ):
        raise ValueError("source-cache preparation structure does not cover 40 layers")
    tensor_shape = [
        1,
        PAPER_ARCHITECTURE.source_kv_heads,
        expected_length,
        PAPER_ARCHITECTURE.source_head_dim,
    ]
    tensor_nbytes = (
        math.prod(tensor_shape)
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    for layer, (source_item, cpu_item, expected_device) in enumerate(
        zip(
            source_structure,
            cpu_structure,
            expected_prefill_layer_devices,
            strict=True,
        )
    ):
        for item, device in ((source_item, expected_device), (cpu_item, "cpu")):
            expected_item = {
                "layer": layer,
                "device": device,
                "key_shape": tensor_shape,
                "value_shape": tensor_shape,
                "key_dtype": "torch.bfloat16",
                "value_dtype": "torch.bfloat16",
                "key_nbytes": tensor_nbytes,
                "value_nbytes": tensor_nbytes,
            }
            if item != expected_item:
                raise ValueError(
                    f"source-cache preparation layer {layer} structure changed"
                )
    return dict(value)


def verify_source_cache_staging_session(
    path: Path,
    *,
    request_fingerprint: str,
    expected_session_id: str,
    expected_runtime_topology: Mapping[str, Any],
    expected_prepared_lengths: Sequence[int] | None = None,
    expected_all_lengths: Sequence[int] | None = None,
    expected_source_weight_bytes: int | None = None,
    expected_target_weight_bytes: int | None = None,
    expected_vocabulary_size: int | None = None,
) -> dict[str, Any]:
    sidecar = verify_sha_sidecar(path)
    value = _strict_json(path)
    if (
        type(expected_session_id) is not str
        or len(expected_session_id) != 32
        or any(character not in "0123456789abcdef" for character in expected_session_id)
        or path.name != f"source_cache_staging_session_{expected_session_id}.json"
    ):
        raise ValueError("source-cache staging session filename/id mismatch")
    expected = {
        "schema_version": SOURCE_CACHE_STAGING_SESSION_SCHEMA_VERSION,
        "status": "source_released_before_target_load",
        "request_fingerprint": request_fingerprint,
        "staging_session_id": expected_session_id,
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "runtime_topology": dict(expected_runtime_topology),
        "runtime_topology_sha256": _canonical_digest(expected_runtime_topology),
        "target_model_loaded_when_session_evidence_written": False,
        "source_model_live_when_session_evidence_written": False,
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"source-cache staging session has invalid {field}")
    missing_lengths = value.get("prepared_missing_lengths")
    preparations = value.get("preparations")
    if (
        not isinstance(missing_lengths, list)
        or not missing_lengths
        or any(type(length) is not int or length <= 0 for length in missing_lengths)
        or len(set(missing_lengths)) != len(missing_lengths)
        or not isinstance(preparations, Mapping)
        or set(preparations) != {str(length) for length in missing_lengths}
    ):
        raise ValueError("source-cache staging session has invalid prepared lengths")
    if expected_prepared_lengths is not None and missing_lengths != list(
        expected_prepared_lengths
    ):
        raise ValueError(
            "source-cache staging session does not contain the canonical missing suffix"
        )
    prefill = expected_runtime_topology["source_model_prefill"]
    expected_layer_devices = [
        f"cuda:{device}"
        for device, count in zip(
            prefill["logical_devices"], prefill["layer_counts"], strict=True
        )
        for _ in range(count)
    ]
    verified_preparations = {
        str(length): verify_source_cache_preparation_record(
            preparations[str(length)],
            expected_length=length,
            expected_prefill_layer_devices=expected_layer_devices,
        )
        for length in missing_lengths
    }
    total = sum(
        int(item["cpu_cache_total_bytes"])
        for item in verified_preparations.values()
    )
    if value.get("cpu_staging_total_bytes") != total:
        raise ValueError("source-cache staging session CPU byte total is invalid")
    host_gate = value.get("host_memory_staging_gate")
    if (
        not isinstance(host_gate, Mapping)
        or host_gate.get("schema_version") != SCHEMA_VERSION
        or host_gate.get("status") != "passed"
        or host_gate.get("scope") != "before_source_model_load"
    ):
        raise ValueError("source-cache staging session lacks host-memory gate")
    host_lengths = host_gate.get("staged_prefix_lengths")
    if (
        not isinstance(host_lengths, list)
        or not host_lengths
        or any(type(length) is not int or length <= 0 for length in host_lengths)
        or not set(missing_lengths).issubset(set(host_lengths))
    ):
        raise ValueError("source-cache staging host-memory lengths are invalid")
    if expected_all_lengths is not None and host_lengths != list(expected_all_lengths):
        raise ValueError("source-cache staging host-memory lengths changed")
    host_staging_bytes = sum(source_cache_exact_bytes(length) for length in host_lengths)
    required_host_bytes = (
        int(host_gate.get("source_weight_shard_bytes", -1))
        + int(host_gate.get("target_weight_shard_bytes", -1))
        + host_staging_bytes
        + STAGED_HOST_MEMORY_OVERHEAD_BYTES
    )
    if (
        host_gate.get("all_requested_source_caches_cpu_bytes")
        != host_staging_bytes
        or host_gate.get("conservative_operating_overhead_bytes")
        != STAGED_HOST_MEMORY_OVERHEAD_BYTES
        or host_gate.get("required_available_bytes") != required_host_bytes
        or type(host_gate.get("mem_available_bytes")) is not int
        or host_gate.get("mem_available_bytes") < required_host_bytes
    ):
        raise ValueError("source-cache staging host-memory arithmetic is invalid")
    if (
        expected_source_weight_bytes is not None
        and host_gate.get("source_weight_shard_bytes")
        != int(expected_source_weight_bytes)
    ):
        raise ValueError("source-cache staging source weight binding changed")
    if (
        expected_target_weight_bytes is not None
        and host_gate.get("target_weight_shard_bytes")
        != int(expected_target_weight_bytes)
    ):
        raise ValueError("source-cache staging target weight binding changed")
    if (
        type(value.get("source_vocabulary_size")) is not int
        or value.get("source_vocabulary_size") <= 0
    ):
        raise ValueError("source-cache staging session vocabulary is invalid")
    if (
        expected_vocabulary_size is not None
        and value.get("source_vocabulary_size") != int(expected_vocabulary_size)
    ):
        raise ValueError("source-cache staging session vocabulary binding changed")
    source_attention = value.get("source_attention")
    if (
        not isinstance(source_attention, Mapping)
        or source_attention.get("status") != "passed"
        or source_attention.get("role") != "source"
        or source_attention.get("expected") != "flash_attention_2"
        or source_attention.get("config_attn_implementation")
        != "flash_attention_2"
        or source_attention.get("decoder_layer_count")
        != PAPER_ARCHITECTURE.source_layers
        or source_attention.get("dispatcher_verified_for_every_layer") is not True
        or source_attention.get("backend_callable_module")
        != "transformers.integrations.flash_attention"
        or source_attention.get("backend_callable_name")
        != "flash_attention_forward"
        or source_attention.get("flash_attention_2_available") is not True
        or source_attention.get("maximum_requested_length") != max(host_lengths)
        or type(source_attention.get("max_position_embeddings")) is not int
        or source_attention.get("max_position_embeddings") < max(host_lengths)
        or not isinstance(source_attention.get("attention_module_classes"), list)
        or not source_attention.get("attention_module_classes")
    ):
        raise ValueError("source-cache staging session lacks source FA2 evidence")
    sharding = value.get("source_model_prefill_sharding")
    expected_prefill_devices = list(prefill["logical_devices"])
    expected_prefill_counts = list(prefill["layer_counts"])
    expected_prefill_layer_ids = [
        device
        for device, count in zip(
            expected_prefill_devices, expected_prefill_counts, strict=True
        )
        for _ in range(count)
    ]
    if (
        not isinstance(sharding, Mapping)
        or sharding.get("status") != "passed"
        or sharding.get("role") != "source_model_prefill"
        or sharding.get("devices") != expected_prefill_devices
        or sharding.get("expected_layer_counts") != expected_prefill_counts
        or sharding.get("actual_layer_counts") != expected_prefill_counts
        or sharding.get("total_layers") != PAPER_ARCHITECTURE.source_layers
        or sharding.get("layer_devices") != expected_prefill_layer_ids
        or sharding.get("actual_parameters_inspected_for_every_layer") is not True
        or sharding.get("auxiliary_parameters_inspected_for_every_module") is not True
    ):
        raise ValueError("source-cache staging session prefill sharding changed")
    auxiliary = sharding.get("auxiliary_modules")
    expected_aux_devices = {
        "model.embed_tokens": expected_prefill_devices[0],
        "model.norm": expected_prefill_devices[-1],
        "lm_head": expected_prefill_devices[-1],
    }
    if not isinstance(auxiliary, Mapping) or set(auxiliary) != set(
        expected_aux_devices
    ):
        raise ValueError("source prefill auxiliary sharding is malformed")
    for module, expected_device in expected_aux_devices.items():
        item = auxiliary[module]
        if (
            not isinstance(item, Mapping)
            or item.get("expected_device") != expected_device
            or item.get("actual_device") != expected_device
            or type(item.get("parameter_tensors_inspected")) is not int
            or item.get("parameter_tensors_inspected") <= 0
        ):
            raise ValueError(f"source prefill auxiliary {module} device changed")
    release = value.get("source_model_release")
    if not isinstance(release, Mapping):
        raise ValueError("source-cache staging session lacks source release evidence")
    if (
        release.get("status") != "passed"
        or release.get("gc_completed") is not True
        or release.get("cuda_synchronized_and_empty_cache_completed") is not True
    ):
        raise ValueError("source-cache staging release completion evidence is invalid")
    workspace_release = release.get("cuda_workspace_and_allocator_release")
    expected_workspace_order = [
        "source_model_deleted_and_gc_collected",
        "synchronize_before_cublas_workspace_clear",
        "torch._C._cuda_clearCublasWorkspaces",
        "gc_collect_after_cublas_workspace_clear",
        "torch.cuda.empty_cache_each_selected_device",
        "synchronize_after_empty_cache",
    ]
    if (
        not isinstance(workspace_release, Mapping)
        or workspace_release.get("status") != "passed"
        or workspace_release.get("selected_device_ids") != expected_prefill_devices
        or workspace_release.get("cublas_workspaces_clear_api")
        != "torch._C._cuda_clearCublasWorkspaces"
        or workspace_release.get("cublas_workspaces_clear_called") is not True
        or workspace_release.get("cublas_workspaces_clear_succeeded") is not True
        or workspace_release.get("source_model_deleted_and_gc_completed_before_clear")
        is not True
        or workspace_release.get("gc_completed_after_workspace_clear") is not True
        or workspace_release.get("empty_cache_completed_for_every_selected_device")
        is not True
        or workspace_release.get("synchronized_before_and_after") is not True
        or workspace_release.get("call_order") != expected_workspace_order
    ):
        raise ValueError("source-cache staging cuBLAS workspace release is invalid")
    for field in (
        "source_model_live",
        "source_model_weakref_is_none",
        "all_source_gpu_cache_references_deleted",
        "all_prefill_devices_allocated_bytes_zero",
        "all_prefill_devices_reserved_bytes_zero",
        "target_model_loaded_at_proof_time",
    ):
        wanted = field not in ("source_model_live", "target_model_loaded_at_proof_time")
        if release.get(field) is not wanted:
            raise ValueError(f"source-cache staging release has invalid {field}")
    if (
        release.get("source_prefill_logical_devices")
        != expected_prefill_devices
        or release.get("source_prefill_target_overlap_exact") != [5]
        or release.get("logical_5_reused_physical_device") != 7
        or release.get("runtime_topology_sha256")
        != _canonical_digest(expected_runtime_topology)
    ):
        raise ValueError("source-cache staging release topology changed")
    after = release.get("allocator_driver_after_release_before_target_load")
    before = release.get("allocator_driver_before_release")
    expected_devices = {
        str(device)
        for device in expected_runtime_topology["source_model_prefill"][
            "logical_devices"
        ]
    }
    if (
        not isinstance(before, Mapping)
        or set(before) != expected_devices
        or not isinstance(after, Mapping)
        or set(after) != expected_devices
    ):
        raise ValueError("source release does not cover every prefill device")
    for device, item in before.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"source release before snapshot {device} is malformed")
        for field in (
            "allocated_bytes",
            "reserved_bytes",
            "driver_free_bytes",
            "driver_total_bytes",
            "driver_used_bytes",
        ):
            if type(item.get(field)) is not int or item.get(field) < 0:
                raise ValueError(
                    f"source release before snapshot {device}.{field} is invalid"
                )
        if (
            item["driver_total_bytes"] - item["driver_free_bytes"]
            != item["driver_used_bytes"]
            or item["allocated_bytes"] <= 0
            or item["reserved_bytes"] < item["allocated_bytes"]
            or item["driver_free_bytes"] > item["driver_total_bytes"]
        ):
            raise ValueError(f"source release before snapshot {device} arithmetic failed")
    for device, item in after.items():
        if (
            not isinstance(item, Mapping)
            or item.get("allocated_bytes") != 0
            or item.get("reserved_bytes") != 0
        ):
            raise ValueError(f"source release device {device} was not allocator-empty")
        if (
            type(item.get("driver_free_bytes")) is not int
            or type(item.get("driver_total_bytes")) is not int
            or type(item.get("driver_used_bytes")) is not int
            or item["driver_total_bytes"] - item["driver_free_bytes"]
            != item["driver_used_bytes"]
        ):
            raise ValueError(f"source release after snapshot {device} arithmetic failed")
    return {**value, "artifact_sha256": sidecar["sha256"]}


def release_restaged_source_cache(
    *,
    cache_ref: weakref.ReferenceType[Any],
    source_devices: Sequence[int],
) -> dict[str, Any]:
    gc.collect()
    synchronize_devices(source_devices)
    for device in source_devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    synchronize_devices(source_devices)
    gc.collect()
    if cache_ref() is not None:
        raise RuntimeError("restaged source cache remains strongly referenced")
    snapshot = _cuda_allocator_driver_snapshot(source_devices)
    if any(
        item["allocated_bytes"] != 0 or item["reserved_bytes"] != 0
        for item in snapshot.values()
    ):
        raise RuntimeError("source-cache staging cards were not empty after cell")
    return {
        "status": "passed",
        "restaged_cache_weakref_is_none": True,
        "released_before_outer_peak_finish": True,
        "cuda_synchronized_and_empty_cache_completed": True,
        "source_cache_staging_devices": list(source_devices),
        "allocator_driver_after_release": snapshot,
    }


def verify_source_cache_lifecycle(
    value: Any,
    *,
    run_path: Path,
    request_fingerprint: str,
    expected_length: int,
    expected_runtime_topology: Mapping[str, Any],
    attention_assertion_sha256: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("staged latency checkpoint lacks source-cache lifecycle")
    session_id = value.get("staging_session_id")
    session_name = value.get("session_evidence_path")
    session_sha256 = value.get("session_evidence_sha256")
    if (
        type(session_id) is not str
        or len(session_id) != 32
        or any(character not in "0123456789abcdef" for character in session_id)
        or type(session_name) is not str
        or session_name != f"source_cache_staging_session_{session_id}.json"
        or not _is_sha256(session_sha256)
    ):
        raise ValueError("staged latency checkpoint has invalid session binding")
    session_path = run_path / session_name
    session = verify_source_cache_staging_session(
        session_path,
        request_fingerprint=request_fingerprint,
        expected_session_id=session_id,
        expected_runtime_topology=expected_runtime_topology,
    )
    if session["artifact_sha256"] != session_sha256:
        raise ValueError("staged latency checkpoint session SHA256 mismatch")
    attention_path = run_path / "attention_backend.json"
    attention_sidecar = verify_sha_sidecar(attention_path)
    attention = _strict_json(attention_path)
    if (
        attention_sidecar.get("sha256") != attention_assertion_sha256
        or attention.get("source") != session.get("source_attention")
        or attention.get("source_model_released_before_target_model_load") is not True
        or attention.get("source_attention_asserted_during_untimed_cache_preparation")
        is not True
    ):
        raise ValueError("staging session source FA2 evidence is not bound to the run")
    preparation = session["preparations"].get(str(expected_length))
    if preparation is None:
        raise ValueError("staging session did not prepare this checkpoint length")
    preparation_sha256 = _canonical_digest(preparation)
    if value.get("preparation_evidence_sha256") != preparation_sha256:
        raise ValueError("checkpoint preparation evidence digest mismatch")
    expected_prefix = preparation["deterministic_prefix_identity"]
    expected_raw_sha = preparation["cpu_staged_raw_bit_sha256"]
    expected_fields = {
        "status": "complete",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "source_model_live_during_timed_cells": False,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "outer_peak_reset_after_restage_validation": True,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "mapper_timer_started": True,
        "deterministic_prefix_identity": expected_prefix,
        "source_gpu_raw_bit_sha256": preparation[
            "source_gpu_raw_bit_sha256"
        ],
        "cpu_staged_raw_bit_sha256": expected_raw_sha,
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "cache_total_bytes": source_cache_exact_bytes(expected_length),
    }
    for field, wanted in expected_fields.items():
        if value.get(field) != wanted:
            raise ValueError(f"checkpoint source-cache lifecycle has invalid {field}")
    restage = value.get("restage")
    if not isinstance(restage, Mapping):
        raise ValueError("checkpoint lacks GPU restage evidence")
    restage_expected = {
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": expected_length,
        "untimed": True,
        "completed_before_outer_peak_reset": True,
        "source_model_live": False,
        "source_cache_gpu_resident_before_mapper_timer": True,
        "copy_semantics": "cpu_to_cuda_blocking_no_dtype_cast",
        "raw_bit_sha256": expected_raw_sha,
        "raw_bit_hash_matches_cpu_staging": True,
        "layer_counts": list(
            expected_runtime_topology["source_cache_staging"]["layer_counts"]
        ),
        "cache_total_bytes": source_cache_exact_bytes(expected_length),
        "cuda_synchronized_and_empty_cache_before_outer_peak_reset": True,
    }
    for field, wanted in restage_expected.items():
        if restage.get(field) != wanted:
            raise ValueError(f"checkpoint GPU restage has invalid {field}")
    expected_layer_devices = [
        f"cuda:{device}"
        for device, count in zip(
            expected_runtime_topology["source_cache_staging"]["logical_devices"],
            expected_runtime_topology["source_cache_staging"]["layer_counts"],
            strict=True,
        )
        for _ in range(count)
    ]
    if restage.get("layer_devices") != expected_layer_devices:
        raise ValueError("checkpoint GPU restage layer devices changed")
    restaged_structure = restage.get("cache_structure")
    tensor_shape = [
        1,
        PAPER_ARCHITECTURE.source_kv_heads,
        expected_length,
        PAPER_ARCHITECTURE.source_head_dim,
    ]
    tensor_nbytes = (
        math.prod(tensor_shape)
        * torch.tensor([], dtype=torch.bfloat16).element_size()
    )
    if (
        not isinstance(restaged_structure, list)
        or len(restaged_structure) != PAPER_ARCHITECTURE.source_layers
    ):
        raise ValueError("checkpoint GPU restage structure is malformed")
    for layer, (item, device) in enumerate(
        zip(restaged_structure, expected_layer_devices, strict=True)
    ):
        if item != {
            "layer": layer,
            "device": device,
            "key_shape": tensor_shape,
            "value_shape": tensor_shape,
            "key_dtype": "torch.bfloat16",
            "value_dtype": "torch.bfloat16",
            "key_nbytes": tensor_nbytes,
            "value_nbytes": tensor_nbytes,
        }:
            raise ValueError(f"checkpoint GPU restage layer {layer} changed")
    release = value.get("restaged_cache_release")
    if (
        not isinstance(release, Mapping)
        or release.get("status") != "passed"
        or release.get("restaged_cache_weakref_is_none") is not True
        or release.get("released_before_outer_peak_finish") is not True
        or release.get("cuda_synchronized_and_empty_cache_completed") is not True
        or release.get("source_cache_staging_devices")
        != list(
            expected_runtime_topology["source_cache_staging"]["logical_devices"]
        )
    ):
        raise ValueError("checkpoint lacks restaged-cache release evidence")
    release_snapshot = release.get("allocator_driver_after_release")
    expected_release_devices = {
        str(device)
        for device in expected_runtime_topology["source_cache_staging"][
            "logical_devices"
        ]
    }
    if (
        not isinstance(release_snapshot, Mapping)
        or set(release_snapshot) != expected_release_devices
    ):
        raise ValueError("restaged-cache release does not cover source devices")
    for device, item in release_snapshot.items():
        if (
            not isinstance(item, Mapping)
            or item.get("allocated_bytes") != 0
            or item.get("reserved_bytes") != 0
        ):
            raise ValueError(
                f"restaged-cache release device {device} was not allocator-empty"
            )
    return dict(value)


def verify_staged_peak_lifecycle_binding(
    *,
    lifecycle: Mapping[str, Any],
    peak_evidence: Mapping[str, Any],
    expected_runtime_topology: Mapping[str, Any],
) -> None:
    cache_total_bytes = int(lifecycle["cache_total_bytes"])
    release_devices = lifecycle["restaged_cache_release"][
        "allocator_driver_after_release"
    ]
    peak_devices = peak_evidence["devices"]
    source_topology = expected_runtime_topology["source_cache_staging"]
    for device, layer_count in zip(
        source_topology["logical_devices"],
        source_topology["layer_counts"],
        strict=True,
    ):
        expected_cache_bytes = (
            cache_total_bytes * int(layer_count)
            // PAPER_ARCHITECTURE.source_layers
        )
        peak_item = peak_devices[str(device)]
        release_item = release_devices[str(device)]
        if peak_item.get("allocated_bytes_before") != expected_cache_bytes:
            raise ValueError(
                f"staged CUDA peak baseline on device {device} does not equal its raw cache"
            )
        if peak_item.get("reserved_bytes_before", -1) < expected_cache_bytes:
            raise ValueError(
                f"staged CUDA peak reservation on device {device} is below its raw cache"
            )
        for field in ("allocated_bytes_after", "reserved_bytes_after"):
            if peak_item.get(field) != 0:
                raise ValueError(
                    f"staged CUDA peak endpoint {field} is nonzero on device {device}"
                )
        if (
            peak_item.get("allocated_bytes_after")
            != release_item.get("allocated_bytes")
            or peak_item.get("reserved_bytes_after")
            != release_item.get("reserved_bytes")
        ):
            raise ValueError(
                f"staged CUDA peak/release allocator snapshots disagree on device {device}"
            )


def make_length_checkpoint(
    current: Mapping[str, Any],
    *,
    request_fingerprint: str,
    attention_assertion_sha256: str,
) -> dict[str, Any]:
    return {
        **dict(current),
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": request_fingerprint,
        "attention_backend_assertion_sha256": attention_assertion_sha256,
    }


def verify_length_checkpoint(
    path: Path,
    *,
    request_fingerprint: str,
    expected_length: int,
    expected_repeats: int,
    expected_profile_repeats: int,
    expected_supplemental_diagnostics: bool,
    attention_assertion_sha256: str,
    expected_device_ids: Sequence[int],
    expected_target_layers: int,
    expected_mapped_cache_dtypes: Sequence[str],
    expected_staged_source_lifecycle: bool = False,
    expected_runtime_topology: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verify_sha_sidecar(path)
    value = _strict_json(path)
    expected_fields = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": request_fingerprint,
        "prefix_length": expected_length,
        "attention_backend_assertion_sha256": attention_assertion_sha256,
    }
    for field, expected in expected_fields.items():
        if value.get(field) != expected:
            raise ValueError(
                f"latency checkpoint {path} has {field}={value.get(field)!r}, expected {expected!r}"
            )
    if value.get("source_prefill_timed") is not False:
        raise ValueError(f"latency checkpoint timed the source prefill: {path}")
    if expected_staged_source_lifecycle:
        if expected_runtime_topology is None:
            raise ValueError("staged checkpoint verification lacks runtime topology")
        if value.get("source_model_live_during_timed_cell") is not False:
            raise ValueError(
                f"staged latency checkpoint retained source model during timing: {path}"
            )
        lifecycle = verify_source_cache_lifecycle(
            value.get("source_cache_lifecycle"),
            run_path=path.parent,
            request_fingerprint=request_fingerprint,
            expected_length=expected_length,
            expected_runtime_topology=expected_runtime_topology,
            attention_assertion_sha256=attention_assertion_sha256,
        )
        if value.get("deterministic_prefix_identity") != lifecycle.get(
            "deterministic_prefix_identity"
        ):
            raise ValueError(
                f"staged latency checkpoint token identity is inconsistent: {path}"
            )
    if type(value.get("batch_size")) is not int or value.get("batch_size") != 1:
        raise ValueError(f"latency checkpoint has an invalid batch size: {path}")
    if (
        type(value.get("source_cache_length")) is not int
        or value.get("source_cache_length") != expected_length
    ):
        raise ValueError(
            f"latency checkpoint has an invalid source cache length: {path}"
        )
    preflight = value.get("strict_transfer_preflight")
    fast_path = value.get("timed_transfer_fast_path")
    if not isinstance(preflight, Mapping) or preflight.get("status") != "passed":
        raise ValueError(f"latency checkpoint lacks a passing strict preflight: {path}")
    for field in (
        "untimed",
        "strict_validation",
        "rope_singularity_reduction",
        "source_content_finite_reduction",
        "mapped_kv_finite_reduction",
    ):
        if preflight.get(field) is not True:
            raise ValueError(
                f"latency checkpoint strict preflight lacks required {field}: {path}"
            )
    for field in ("source_cache_length", "mapped_cache_length"):
        if (
            type(preflight.get(field)) is not int
            or preflight.get(field) != expected_length
        ):
            raise ValueError(
                f"latency checkpoint strict preflight has invalid {field}: {path}"
            )
    validated_layers = preflight.get("validated_target_layers")
    if type(validated_layers) is not int or validated_layers != expected_target_layers:
        raise ValueError(
            f"latency checkpoint strict preflight has invalid validated_target_layers: {path}"
        )
    mapped_dtypes = preflight.get("mapped_cache_dtypes")
    if mapped_dtypes != list(expected_mapped_cache_dtypes):
        raise ValueError(
            f"latency checkpoint strict preflight has invalid mapped_cache_dtypes: {path}"
        )
    preflight_wall_ms = preflight.get("wall_ms_diagnostic_not_latency")
    if (
        type(preflight_wall_ms) not in (int, float)
        or not math.isfinite(float(preflight_wall_ms))
        or preflight_wall_ms < 0
    ):
        raise ValueError(
            f"latency checkpoint strict preflight has invalid diagnostic wall time: {path}"
        )
    expected_fast_path = {
        "enabled": True,
        "strict_preflight_required_and_passed": True,
        "strict_validation_argument": False,
        "diagnostic_rope_and_finite_reductions_in_timed_region": False,
        "shape_dtype_architecture_and_output_validation_retained": True,
    }
    if fast_path != expected_fast_path:
        raise ValueError(
            f"latency checkpoint lacks the validated timed fast-path contract: {path}"
        )
    nested_peak_contract = value.get("nested_cuda_peak_memory_contract")
    expected_nested_peak_contract = {
        "outer_cell_peak_measurement_active": True,
        "internal_metric_peak_collection_enabled": False,
        "internal_peak_counter_resets_disabled": True,
    }
    if nested_peak_contract != expected_nested_peak_contract:
        raise ValueError(
            f"latency checkpoint did not disable nested CUDA peak resets: {path}"
        )
    expected_latency_protocol = {
        "name": (
            "paper_main_plus_supplemental"
            if expected_supplemental_diagnostics
            else "paper_main_only"
        ),
        "paper_main_metrics": ["target_reprefill", "mapper_handoff_end_to_end"],
        "supplemental_diagnostics_enabled": expected_supplemental_diagnostics,
        "component_profile_in_cell": expected_supplemental_diagnostics,
        "boundary_decode_in_cell": expected_supplemental_diagnostics,
        "protocol_complete_in_cell": expected_supplemental_diagnostics,
        "paper_headline_speedup_uses_supplemental_metrics": False,
    }
    if value.get("latency_protocol") != expected_latency_protocol:
        raise ValueError(
            f"latency checkpoint has the wrong paper-main protocol: {path}"
        )
    for metric in ("target_reprefill", "mapper_handoff_end_to_end"):
        summary = value.get(metric)
        if (
            not isinstance(summary, Mapping)
            or type(summary.get("count")) is not int
            or summary.get("count") != expected_repeats
        ):
            raise ValueError(
                f"latency checkpoint {path} has an invalid {metric} repeat count"
            )
        samples = summary.get("samples")
        if (
            not isinstance(samples, list)
            or len(samples) != expected_repeats
            or any(
                type(sample) not in (int, float)
                or not math.isfinite(float(sample))
                or float(sample) < 0
                for sample in samples
            )
        ):
            raise ValueError(f"latency checkpoint {path} has invalid {metric} samples")
        for field in ("mean", "median", "p10", "p50", "p90", "min", "max"):
            number = summary.get(field)
            if (
                type(number) not in (int, float)
                or not math.isfinite(float(number))
                or float(number) < 0
            ):
                raise ValueError(
                    f"latency checkpoint {path} has invalid {metric}.{field}"
                )
        if summary.get("cuda_peak_memory_collected") is not False or any(
            field in summary
            for field in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
        ):
            raise ValueError(
                f"latency checkpoint {path} has an internal peak reset/claim for {metric}"
            )
        if summary.get("cuda_peak_memory_disabled_reason") != (
            "preserve_caller_owned_outer_cuda_peak_counters"
        ):
            raise ValueError(
                f"latency checkpoint {path} lacks the internal peak disable reason for {metric}"
            )
        if summary.get("inference_mode_enabled") is not True:
            raise ValueError(
                f"latency checkpoint {path} lacks inference-mode evidence for {metric}"
            )
        if summary.get("autograd_graph_disabled") is not True:
            raise ValueError(
                f"latency checkpoint {path} lacks disabled-autograd evidence for {metric}"
            )
    target_median = float(value["target_reprefill"]["median"])
    mapper_median = float(value["mapper_handoff_end_to_end"]["median"])
    speedup = value.get("median_speedup_target_reprefill_over_mapper")
    if (
        mapper_median <= 0
        or type(speedup) not in (int, float)
        or not math.isfinite(float(speedup))
        or not math.isclose(
            float(speedup), target_median / mapper_median, rel_tol=1e-12, abs_tol=0.0
        )
    ):
        raise ValueError(f"latency checkpoint {path} has an invalid paper-main speedup")
    supplemental_fields = (
        "mapper_synchronized_component_profile",
        "mapper_kernel_only_profile",
        "boundary_one_token_decode",
        "protocol_complete",
        "median_speedup_target_reprefill_over_protocol_complete",
    )
    if not expected_supplemental_diagnostics:
        present = [field for field in supplemental_fields if field in value]
        if present:
            raise ValueError(
                f"paper-main latency checkpoint contains supplemental fields {present}: {path}"
            )
    else:
        for metric in ("boundary_one_token_decode", "protocol_complete"):
            summary = value.get(metric)
            if (
                not isinstance(summary, Mapping)
                or summary.get("count") != expected_repeats
            ):
                raise ValueError(
                    f"latency checkpoint {path} has an invalid {metric} repeat count"
                )
            if summary.get("cuda_peak_memory_collected") is not False or any(
                field in summary
                for field in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
            ):
                raise ValueError(
                    f"latency checkpoint {path} has an internal peak reset/claim for {metric}"
                )
            if summary.get("cuda_peak_memory_disabled_reason") != (
                "preserve_caller_owned_outer_cuda_peak_counters"
            ):
                raise ValueError(
                    f"latency checkpoint {path} lacks the internal peak disable reason for {metric}"
                )
        profile_record = value.get("mapper_synchronized_component_profile")
        if not isinstance(profile_record, Mapping):
            raise ValueError(
                f"latency checkpoint {path} has an invalid component profile"
            )
        if profile_record.get("cuda_peak_memory_collected") is not False or any(
            field in profile_record
            for field in ("cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes")
        ):
            raise ValueError(
                f"latency checkpoint {path} component profile reset the outer peak"
            )
        if profile_record.get("cuda_peak_memory_disabled_reason") != (
            "preserve_caller_owned_outer_cuda_peak_counters"
        ):
            raise ValueError(
                f"latency checkpoint {path} lacks the component peak disable reason"
            )
        profile = profile_record.get("components")
        if not isinstance(profile, Mapping) or set(profile) != set(COMPONENT_NAMES):
            raise ValueError(
                f"latency checkpoint {path} has an invalid component profile"
            )
        if any(
            not isinstance(summary, Mapping)
            or summary.get("count") != expected_profile_repeats
            for summary in profile.values()
        ):
            raise ValueError(
                f"latency checkpoint {path} has an invalid profile repeat count"
            )
    peak_evidence = verify_cuda_peak_memory_evidence(
        value.get("cuda_peak_memory"),
        expected_device_ids=expected_device_ids,
        expected_prefix_length=expected_length,
        expected_request_fingerprint=request_fingerprint,
        expected_measurement_scope=(
            STAGED_CUDA_PEAK_MEASUREMENT_SCOPE
            if expected_staged_source_lifecycle
            else CUDA_PEAK_MEASUREMENT_SCOPE
        ),
    )
    if expected_staged_source_lifecycle:
        assert expected_runtime_topology is not None
        verify_staged_peak_lifecycle_binding(
            lifecycle=value["source_cache_lifecycle"],
            peak_evidence=peak_evidence,
            expected_runtime_topology=expected_runtime_topology,
        )
    return value


def write_results_manifest(
    run: RunDirectory,
    *,
    request_fingerprint: str,
    lengths: Sequence[int],
) -> tuple[Path, dict[str, Any]]:
    session_names: list[str] = []
    for length in lengths:
        checkpoint = _strict_json(run.path / f"length_{length:05d}.json")
        lifecycle = checkpoint.get("source_cache_lifecycle")
        if isinstance(lifecycle, Mapping):
            name = lifecycle.get("session_evidence_path")
            if (
                type(name) is not str
                or Path(name).name != name
                or not name.startswith("source_cache_staging_session_")
                or not name.endswith(".json")
            ):
                raise ValueError("checkpoint has an unsafe staging-session path")
            if name not in session_names:
                session_names.append(name)
    names = [
        "run_definition.json",
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
    preload_gate_path = run.path / "preload_gpu_idle_gate.json"
    if preload_gate_path.is_file():
        names.insert(1, preload_gate_path.name)
    artifacts: list[dict[str, Any]] = []
    for name in names:
        path = run.path / name
        if not path.is_file():
            raise FileNotFoundError(
                f"cannot finalize latency run; artifact is missing: {path}"
            )
        sidecar = verify_sha_sidecar(path)
        artifacts.append(
            {"path": name, "sha256": sidecar["sha256"], "nbytes": sidecar["nbytes"]}
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": request_fingerprint,
        "lengths": list(lengths),
        "artifacts": artifacts,
        "artifacts_digest": _canonical_digest(artifacts),
    }
    path = run.path / "results.manifest.json"
    write_hashed_json(path, manifest)
    return path, manifest


def verify_completed_run(
    run: RunDirectory,
    *,
    request_fingerprint: str,
    lengths: Sequence[int],
    expected_repeats: int,
    expected_profile_repeats: int,
    expected_supplemental_diagnostics: bool,
    expected_device_ids: Sequence[int],
    expected_target_layers: int,
    expected_mapped_cache_dtypes: Sequence[str],
    expected_staged_source_lifecycle: bool = False,
    expected_runtime_topology: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    result_path = run.path / "result.json"
    if not result_path.exists():
        return None
    result = _strict_json(result_path)
    if result.get("status") != "complete":
        return None
    verify_sha_sidecar(result_path)
    if result.get("request_fingerprint") != request_fingerprint:
        raise ValueError("completed latency result fingerprint mismatch")
    attention_sidecar = verify_sha_sidecar(run.path / "attention_backend.json")
    for length in lengths:
        verify_length_checkpoint(
            run.path / f"length_{length:05d}.json",
            request_fingerprint=request_fingerprint,
            expected_length=length,
            expected_repeats=expected_repeats,
            expected_profile_repeats=expected_profile_repeats,
            expected_supplemental_diagnostics=expected_supplemental_diagnostics,
            attention_assertion_sha256=attention_sidecar["sha256"],
            expected_device_ids=expected_device_ids,
            expected_target_layers=expected_target_layers,
            expected_mapped_cache_dtypes=expected_mapped_cache_dtypes,
            expected_staged_source_lifecycle=expected_staged_source_lifecycle,
            expected_runtime_topology=expected_runtime_topology,
        )
    manifest_path = run.path / "results.manifest.json"
    verify_sha_sidecar(manifest_path)
    manifest = _strict_json(manifest_path)
    if manifest.get("request_fingerprint") != request_fingerprint or manifest.get(
        "lengths"
    ) != list(lengths):
        raise ValueError("completed latency results manifest request mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or manifest.get(
        "artifacts_digest"
    ) != _canonical_digest(artifacts):
        raise ValueError("completed latency results manifest digest mismatch")
    definition = _strict_json(run.path / "run_definition.json")
    session_names: list[str] = []
    for length in lengths:
        checkpoint = _strict_json(run.path / f"length_{length:05d}.json")
        lifecycle = checkpoint.get("source_cache_lifecycle")
        if isinstance(lifecycle, Mapping):
            name = lifecycle.get("session_evidence_path")
            if type(name) is not str or Path(name).name != name:
                raise ValueError("completed checkpoint has an unsafe session path")
            if name not in session_names:
                session_names.append(name)
    expected_names = [
        "run_definition.json",
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
    if definition.get("cli", {}).get("mode") == "production":
        expected_names.insert(1, "preload_gpu_idle_gate.json")
    if [
        item.get("path") if isinstance(item, Mapping) else None for item in artifacts
    ] != expected_names:
        raise ValueError(
            "completed latency results manifest artifact whitelist/order mismatch"
        )
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise ValueError("completed latency results manifest has a malformed entry")
        path = run.path / str(artifact.get("path"))
        if (
            not path.is_file()
            or sha256_file(path) != artifact.get("sha256")
            or path.stat().st_size != artifact.get("nbytes")
        ):
            raise ValueError(
                f"completed latency artifact no longer matches manifest: {path}"
            )
    if definition.get("cli", {}).get("mode") == "production":
        gate_path = run.path / "preload_gpu_idle_gate.json"
        verify_sha_sidecar(gate_path)
        gate_record = _strict_json(gate_path)
        definition_gate = definition.get("preload_gpu_idle_gate")
        if (
            gate_record.get("request_fingerprint") != request_fingerprint
            or gate_record.get("run_definition_gate_evidence_sha256")
            != _canonical_digest(definition_gate)
            or gate_record.get("immediate_preload_gpu_idle_sandwich", {}).get("status")
            != "passed"
        ):
            raise ValueError(
                "completed latency pre-load GPU idle-gate binding mismatch"
            )
        verify_preload_gpu_idle_gate_evidence(
            gate_record.get("active_preload_gpu_idle_gate"),
            expected_policy=definition_gate["policy"],
        )
    return result


def load_runtime(args: argparse.Namespace):
    if args.mode == "tiny":
        device = torch.device(args.tiny_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("tiny CUDA mode requested but CUDA is unavailable")
        model = tiny_model(device)
        mapper = tiny_identity_mapper(model)
        return (
            model,
            model,
            mapper,
            {
                "scope": "tiny_identity_mapper_in_memory",
                "host_load_and_hash": None,
                "weight_host_to_device": None,
                "resident_parameter_bytes": mapper.resident_parameter_bytes,
                "resident_dtype": "float32",
                "manifest_sha256": None,
                "fit_fingerprint": None,
                "execution": "resident_eager_fp32",
                "model_layer_sharding": {
                    "status": "skipped",
                    "reason": "tiny mode shares one in-memory model",
                },
            },
        )

    target = load_qwen3(
        args.target_model,
        devices=args.target_devices,
        layer_counts=args.target_layer_counts,
        role="target",
        attention_implementation=args.attention,
    )
    source = load_qwen3(
        args.source_model,
        devices=args.source_devices,
        layer_counts=args.source_layer_counts,
        role="source",
        attention_implementation=args.attention,
    )
    target_sharding = model_layer_sharding_evidence(
        target,
        devices=args.target_devices,
        layer_counts=args.target_layer_counts,
        role="target",
    )
    source_sharding = model_layer_sharding_evidence(
        source,
        devices=args.source_devices,
        layer_counts=args.source_layer_counts,
        role="source",
    )
    bundle = MapperBundle.open(args.mapper)
    resident, setup = materialize_mapper_on_target(
        bundle,
        tuple(next(layer.parameters()).device for layer in target.model.layers),
    )
    setup["resident_parameter_bytes"] = resident.resident_parameter_bytes
    setup["resident_dtype"] = "float32"
    setup["manifest_sha256"] = resident.manifest_sha256
    setup["fit_fingerprint"] = resident.fit_fingerprint
    setup["execution"] = "resident_eager_fp32"
    setup["target_layer_count"] = resident.architecture.target_layers
    setup["model_layer_sharding"] = {
        "source": source_sharding,
        "target": target_sharding,
    }
    return source, target, resident, setup


def source_cache_staging_sharding_evidence(
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts = resolve_contiguous_layer_counts(
        num_hidden_layers=PAPER_ARCHITECTURE.source_layers,
        devices=args.source_devices,
        layer_counts=args.source_layer_counts,
    )
    layer_devices = [
        device
        for device, count in zip(args.source_devices, counts, strict=True)
        for _ in range(count)
    ]
    ranges: list[dict[str, int]] = []
    start = 0
    for device, count in zip(args.source_devices, counts, strict=True):
        ranges.append(
            {
                "device": device,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "count": count,
            }
        )
        start += count
    return {
        "status": "passed",
        "role": "source_cache_staging_without_source_model",
        "mode": "explicit_exact_counts",
        "devices": list(args.source_devices),
        "declared_layer_counts": list(args.source_layer_counts),
        "expected_layer_counts": list(counts),
        "actual_layer_counts": list(counts),
        "total_layers": PAPER_ARCHITECTURE.source_layers,
        "layer_devices": layer_devices,
        "contiguous_ranges": ranges,
        "source_model_parameters_present": False,
        "validated_per_cell_from_actual_cache_tensors": True,
    }


def load_staged_source_runtime(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not args.stage_source_caches_on_cpu_before_cells:
        raise ValueError("staged source runtime requested without staging policy")
    source = load_qwen3(
        args.source_model,
        devices=args.source_prefill_devices,
        layer_counts=args.source_prefill_layer_counts,
        role="source",
        attention_implementation=args.attention,
    )
    source_sharding = model_layer_sharding_evidence(
        source,
        devices=args.source_prefill_devices,
        layer_counts=args.source_prefill_layer_counts,
        role="source_model_prefill",
    )
    return source, source_sharding


def load_staged_target_mapper_runtime(
    args: argparse.Namespace,
    *,
    source_prefill_sharding: Mapping[str, Any],
) -> tuple[torch.nn.Module, ResidentMapperBundle, dict[str, Any]]:
    """Load target+mapper only after the caller proved source release."""

    target = load_qwen3(
        args.target_model,
        devices=args.target_devices,
        layer_counts=args.target_layer_counts,
        role="target",
        attention_implementation=args.attention,
    )
    target_sharding = model_layer_sharding_evidence(
        target,
        devices=args.target_devices,
        layer_counts=args.target_layer_counts,
        role="target",
    )
    bundle = MapperBundle.open(args.mapper)
    resident, setup = materialize_mapper_on_target(
        bundle,
        tuple(next(layer.parameters()).device for layer in target.model.layers),
    )
    setup["resident_parameter_bytes"] = resident.resident_parameter_bytes
    setup["resident_dtype"] = "float32"
    setup["manifest_sha256"] = resident.manifest_sha256
    setup["fit_fingerprint"] = resident.fit_fingerprint
    setup["execution"] = "resident_eager_fp32"
    setup["target_layer_count"] = resident.architecture.target_layers
    staging = source_cache_staging_sharding_evidence(args)
    setup["model_layer_sharding"] = {
        # Retain the legacy key for downstream readers while making the three
        # different v9 roles explicit and independently auditable.
        "source": dict(source_prefill_sharding),
        "source_model_prefill": dict(source_prefill_sharding),
        "source_cache_staging": staging,
        "target": target_sharding,
    }
    setup["source_model_live_during_timed_cells"] = False
    setup["source_cache_staging_policy"] = SOURCE_CACHE_STAGING_POLICY
    setup["runtime_topology"] = staged_runtime_topology(args)
    return target, resident, setup


def prepare_staged_runtime(
    args: argparse.Namespace,
    *,
    run: RunDirectory,
    request_fingerprint: str,
    all_lengths: Sequence[int],
    missing_lengths: Sequence[int],
    runtime_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Prepare missing CPU caches, prove release, then load target+mapper."""

    if not missing_lengths:
        raise ValueError("staged runtime preparation requires at least one missing length")
    topology = staged_runtime_topology(args)
    host_gate = host_memory_staging_gate(
        lengths=all_lengths,
        source_weight_bytes=int(
            runtime_artifacts["source_model"]["weight_shard_total_nbytes"]
        ),
        target_weight_bytes=int(
            runtime_artifacts["target_model"]["weight_shard_total_nbytes"]
        ),
    )
    source, source_sharding = load_staged_source_runtime(args)
    source_attention = assert_attention_backend(
        source,
        role="source",
        expected=args.attention,
        maximum_requested_length=max(all_lengths),
    )
    source_config = source.config
    source_vocabulary_size = int(source.config.vocab_size)
    source_bound_config = runtime_artifacts["source_model"].get(
        "bound_config_fields"
    )
    target_bound_config = runtime_artifacts["target_model"].get(
        "bound_config_fields"
    )
    if (
        not isinstance(source_bound_config, Mapping)
        or not isinstance(target_bound_config, Mapping)
        or source_bound_config.get("vocab_size") != source_vocabulary_size
        or target_bound_config.get("vocab_size") != source_vocabulary_size
        or source_bound_config.get("model_type") != "qwen3"
        or target_bound_config.get("model_type") != "qwen3"
    ):
        raise RuntimeError(
            "loaded source vocabulary/model type differs from fingerprint-bound models"
        )
    staged_caches: dict[int, Any] = {}
    preparations: dict[str, dict[str, Any]] = {}
    for length in missing_lengths:
        current_attention = assert_attention_backend(
            source,
            role="source",
            expected=args.attention,
            maximum_requested_length=max(all_lengths),
        )
        if current_attention != source_attention:
            raise RuntimeError("source FA2 contract changed during cache preparation")
        input_ids = deterministic_ids(
            int(length),
            vocabulary_size=source_vocabulary_size,
            seed=args.seed,
        )
        prefix_ids = input_ids[:, : int(length)]
        prefix_identity = deterministic_prefix_identity(prefix_ids)
        gpu_cache = build_source_cache(source, prefix_ids)
        synchronize_devices(args.source_prefill_devices)
        staged, preparation = stage_source_cache_on_cpu(
            gpu_cache,
            length=int(length),
            prefix_identity=prefix_identity,
            source_prefill_devices=args.source_prefill_devices,
            source_prefill_layer_counts=args.source_prefill_layer_counts,
        )
        gpu_cache_ref = weakref.ref(gpu_cache)
        del gpu_cache
        gc.collect()
        synchronize_devices(args.source_prefill_devices)
        for device in args.source_prefill_devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        synchronize_devices(args.source_prefill_devices)
        if gpu_cache_ref() is not None:
            raise RuntimeError(
                f"source GPU cache for length {length} remained referenced after staging"
            )
        staged_caches[int(length)] = staged
        preparations[str(int(length))] = preparation

    staged_total = sum(
        int(preparation["cpu_cache_total_bytes"])
        for preparation in preparations.values()
    )
    expected_missing_total = sum(
        source_cache_exact_bytes(int(length)) for length in missing_lengths
    )
    if staged_total != expected_missing_total:
        raise RuntimeError("prepared CPU source-cache byte total changed")

    allocator_before_release = _cuda_allocator_driver_snapshot(
        args.source_prefill_devices
    )
    source_model_ref = weakref.ref(source)
    del source
    release = prove_source_model_released_before_target_load(
        source_model_ref=source_model_ref,
        source_prefill_devices=args.source_prefill_devices,
        allocator_before_release=allocator_before_release,
        topology=topology,
    )
    session_id = uuid.uuid4().hex
    session_name = f"source_cache_staging_session_{session_id}.json"
    session_path = run.path / session_name
    session_payload = {
        "schema_version": SOURCE_CACHE_STAGING_SESSION_SCHEMA_VERSION,
        "status": "source_released_before_target_load",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request_fingerprint": request_fingerprint,
        "staging_session_id": session_id,
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "runtime_topology": topology,
        "runtime_topology_sha256": _canonical_digest(topology),
        "host_memory_staging_gate": host_gate,
        "prepared_missing_lengths": list(missing_lengths),
        "source_vocabulary_size": source_vocabulary_size,
        "source_attention": source_attention,
        "source_model_prefill_sharding": source_sharding,
        "preparations": preparations,
        "cpu_staging_total_bytes": staged_total,
        "source_model_release": release,
        "target_model_loaded_when_session_evidence_written": False,
        "source_model_live_when_session_evidence_written": False,
    }
    session_sidecar = write_hashed_json(session_path, session_payload)
    verify_source_cache_staging_session(
        session_path,
        request_fingerprint=request_fingerprint,
        expected_session_id=session_id,
        expected_runtime_topology=topology,
        expected_prepared_lengths=missing_lengths,
        expected_all_lengths=all_lengths,
        expected_source_weight_bytes=int(
            runtime_artifacts["source_model"]["weight_shard_total_nbytes"]
        ),
        expected_target_weight_bytes=int(
            runtime_artifacts["target_model"]["weight_shard_total_nbytes"]
        ),
        expected_vocabulary_size=source_vocabulary_size,
    )

    # This call is deliberately after the zero-allocator proof and immutable
    # session write.  logical cuda:5 / physical GPU 7 is reused only now.
    target, mapper, mapper_setup = load_staged_target_mapper_runtime(
        args,
        source_prefill_sharding=source_sharding,
    )
    attention_contract = staged_runtime_attention_contract(
        source_assertion=source_attention,
        target_model=target,
        expected=args.attention,
        lengths=all_lengths,
    )
    return {
        "source_config": source_config,
        "source_vocabulary_size": source_vocabulary_size,
        "staged_caches": staged_caches,
        "preparations": preparations,
        "source_attention": source_attention,
        "source_model_release": release,
        "target": target,
        "mapper": mapper,
        "mapper_setup": mapper_setup,
        "attention_contract": attention_contract,
        "runtime_topology": topology,
        "staging_session_id": session_id,
        "session_evidence_path": session_name,
        "session_evidence_sha256": session_sidecar["sha256"],
        "host_memory_staging_gate": host_gate,
    }


def expected_checkpoint_contract(
    args: argparse.Namespace,
) -> tuple[int, tuple[str, ...]]:
    """Return the exact architecture/dtype contract a checkpoint must prove.

    Production is deliberately specific to the requested Qwen3-14B -> Qwen3-32B
    reproduction.  Tiny mode uses the fixed two-layer fixture and its device-
    dependent model/cache dtype.  This contract is available before loading the
    models, so an already-complete run can be rejected safely during resume.
    """

    if args.mode == "production":
        return 64, ("torch.bfloat16",)
    tiny_dtype = (
        "torch.bfloat16"
        if torch.device(args.tiny_device).type == "cuda"
        else "torch.float32"
    )
    return 2, (tiny_dtype,)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    lengths = args.lengths
    if args.mode == "tiny" and lengths == PAPER_LENGTHS:
        lengths = TINY_DEFAULT_LENGTHS
    profile_warmup = args.warmup if args.profile_warmup is None else args.profile_warmup
    profile_repeats = (
        args.repeats if args.profile_repeats is None else args.profile_repeats
    )
    math_contract = configure_strict_latency_math()
    if args.mode == "production":
        selected_devices = args.source_devices + args.target_devices
    else:
        tiny_device = torch.device(args.tiny_device)
        selected_devices = (
            (
                (
                    tiny_device.index
                    if tiny_device.index is not None
                    else torch.cuda.current_device()
                ),
            )
            if tiny_device.type == "cuda"
            else ()
        )
    runtime_topology_contract = (
        staged_runtime_topology(args)
        if args.mode == "production"
        and args.stage_source_caches_on_cpu_before_cells
        else None
    )
    expected_target_layers, expected_mapped_cache_dtypes = expected_checkpoint_contract(
        args
    )
    hardware = hardware_snapshot(selected_devices)
    run_definition = build_run_definition(
        args,
        lengths=lengths,
        profile_warmup=profile_warmup,
        profile_repeats=profile_repeats,
        hardware=hardware,
    )
    preload_gate_evidence: dict[str, Any] | None = None
    preload_gate_sampled_now = False
    if args.mode == "production":
        run_definition, preload_gate_evidence, preload_gate_sampled_now = (
            bind_preload_gpu_idle_gate(args, run_definition, selected_devices)
        )
    request_fingerprint = _canonical_digest(run_definition)
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "request_fingerprint": request_fingerprint,
        "mode": args.mode,
        "source_model": str(args.source_model.resolve()),
        "target_model": str(args.target_model.resolve()),
        "mapper": str(args.mapper.resolve()),
        "source_devices": list(args.source_devices),
        "target_devices": list(args.target_devices),
        "source_layer_counts": (
            list(args.source_layer_counts)
            if args.source_layer_counts is not None
            else None
        ),
        "target_layer_counts": (
            list(args.target_layer_counts)
            if args.target_layer_counts is not None
            else None
        ),
        "stage_source_caches_on_cpu_before_cells": (
            args.stage_source_caches_on_cpu_before_cells
        ),
        "source_prefill_devices": (
            list(args.source_prefill_devices)
            if args.source_prefill_devices is not None
            else None
        ),
        "source_prefill_layer_counts": (
            list(args.source_prefill_layer_counts)
            if args.source_prefill_layer_counts is not None
            else None
        ),
        "staged_runtime_topology": runtime_topology_contract,
        "layer_sharding": (
            production_layer_sharding_contract(args)
            if args.mode == "production"
            else None
        ),
        "cuda_visible_devices_environment": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tiny_device": args.tiny_device,
        "prefix_lengths": list(lengths),
        "batch_size": 1,
        "warmup": args.warmup,
        "timed_repeats": args.repeats,
        "latency_protocol": (
            "paper_main_plus_supplemental"
            if args.supplemental_diagnostics
            else "paper_main_only"
        ),
        "supplemental_diagnostics_enabled": args.supplemental_diagnostics,
        "profile_warmup": profile_warmup if args.supplemental_diagnostics else None,
        "profile_timed_repeats": (
            profile_repeats if args.supplemental_diagnostics else None
        ),
        "mapper_token_chunk": args.mapper_token_chunk,
        "attention_implementation": args.attention,
        "seed": args.seed,
        "expected_target_layers": expected_target_layers,
        "expected_mapped_cache_dtypes": list(expected_mapped_cache_dtypes),
        "preload_gpu_idle_gate_policy": (
            preload_gpu_idle_gate_policy(args, selected_devices)
            if args.mode == "production"
            else None
        ),
        "local_conventions": {
            "source_prefill_included": False,
            "target_reprefill_excludes_lm_head": True,
            "paper_main_metrics": [
                "target_reprefill",
                "mapper_handoff_end_to_end",
            ],
            "boundary_decode_excludes_lm_head": (
                True if args.supplemental_diagnostics else None
            ),
            "boundary_and_protocol_are_supplemental_not_paper_main": True,
            "mapper_master_dtype": "float32",
            "model_and_cache_dtype": (
                "bfloat16" if args.mode == "production" else "device_dependent_tiny"
            ),
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
            "per_length_cuda_peak_memory_evidence_required": bool(selected_devices),
            "cuda_peak_counters_reset_after_synchronized_baseline": True,
            "internal_metric_cuda_peak_resets_disabled": True,
            "driver_free_memory_is_before_after_snapshot_not_historical_peak": True,
            "source_cache_staging_policy": (
                SOURCE_CACHE_STAGING_POLICY
                if args.stage_source_caches_on_cpu_before_cells
                else None
            ),
            "source_model_live_during_timed_cells": (
                False if args.stage_source_caches_on_cpu_before_cells else True
            ),
            "source_cpu_staging_and_gpu_restage_in_latency": False,
            "mapper_timer_includes_source_to_target_transfer": True,
            "author_model_residency_and_sharding_published": False,
            "local_staged_sharding_is_4090_capacity_configuration": bool(
                args.stage_source_caches_on_cpu_before_cells
            ),
        },
    }
    run, resumed = prepare_run_directory(
        args,
        configuration,
        run_definition,
        request_fingerprint,
    )
    completed = verify_completed_run(
        run,
        request_fingerprint=request_fingerprint,
        lengths=lengths,
        expected_repeats=args.repeats,
        expected_profile_repeats=profile_repeats,
        expected_supplemental_diagnostics=args.supplemental_diagnostics,
        expected_device_ids=selected_devices,
        expected_target_layers=expected_target_layers,
        expected_mapped_cache_dtypes=expected_mapped_cache_dtypes,
        expected_staged_source_lifecycle=args.stage_source_caches_on_cpu_before_cells,
        expected_runtime_topology=runtime_topology_contract,
    )
    if completed is not None:
        print(
            json.dumps(
                {
                    "status": "reused_complete",
                    "run": str(run.path),
                    "request_fingerprint": request_fingerprint,
                },
                ensure_ascii=False,
            )
        )
        return 0
    existing_checkpoint_lengths, missing_lengths = checkpoint_prefix_partition(
        run, lengths
    )
    if args.mode == "production" and not preload_gate_sampled_now:
        # A completed run never loads models and returned above.  A partial
        # production resume must establish a new post-hash gate immediately
        # before reloading CUDA runtime state.
        preload_gate_evidence = wait_for_preload_gpu_idle_gate(args, selected_devices)
        verify_preload_gpu_idle_gate_evidence(
            preload_gate_evidence,
            expected_policy=preload_gpu_idle_gate_policy(args, selected_devices),
        )
    atomic_write_json(
        run.path / "status.json",
        {
            "status": "resuming" if resumed else "starting",
            "request_fingerprint": request_fingerprint,
        },
    )
    failure_prefix_length: int | None = None
    failure_peak_memory: dict[str, Any] | None = None
    failure_source_cache_lifecycle: dict[str, Any] | None = None
    try:
        if args.mode == "production":
            preload_gate_record = {
                "schema_version": PRELOAD_GPU_IDLE_GATE_SCHEMA_VERSION,
                "request_fingerprint": request_fingerprint,
                "run_definition_gate_evidence_sha256": _canonical_digest(
                    run_definition["preload_gpu_idle_gate"]
                ),
                "active_preload_gpu_idle_gate": preload_gate_evidence,
            }
        else:
            preload_gate_record = None
        hardware_path = run.path / "hardware.json"
        _write_immutable_json(hardware_path, hardware)
        if _sha_sidecar_path(hardware_path).exists():
            verify_sha_sidecar(hardware_path)
        else:
            write_sha_sidecar(hardware_path)
        before_path = run.path / "telemetry_before.json"
        if before_path.exists() and _sha_sidecar_path(before_path).exists():
            verify_sha_sidecar(before_path)
            before = _strict_json(before_path)
        else:
            before = telemetry_snapshot()
            write_hashed_json(before_path, before)
        atomic_write_json(
            run.path / "status.json",
            {"status": "loading_runtime", "request_fingerprint": request_fingerprint},
        )
        if args.mode == "production":
            preload_gate_record["immediate_preload_gpu_idle_sandwich"] = (
                immediate_preload_gpu_idle_sandwich(args, selected_devices)
            )
            write_hashed_json(
                run.path / "preload_gpu_idle_gate.json", preload_gate_record
            )
        staged_runtime: dict[str, Any] | None = None
        staged_caches: dict[int, Any] = {}
        preparations: dict[str, dict[str, Any]] = {}
        source_config: Any | None = None
        source_vocabulary_size: int | None = None
        source_attention: Mapping[str, Any] | None = None
        if args.stage_source_caches_on_cpu_before_cells and missing_lengths:
            staged_runtime = prepare_staged_runtime(
                args,
                run=run,
                request_fingerprint=request_fingerprint,
                all_lengths=lengths,
                missing_lengths=missing_lengths,
                runtime_artifacts=run_definition["runtime_artifacts"],
            )
            source = None
            target = staged_runtime["target"]
            mapper = staged_runtime["mapper"]
            mapper_setup = staged_runtime["mapper_setup"]
            attention_contract = staged_runtime["attention_contract"]
            staged_caches = staged_runtime["staged_caches"]
            preparations = staged_runtime["preparations"]
            source_config = staged_runtime["source_config"]
            source_vocabulary_size = staged_runtime["source_vocabulary_size"]
            source_attention = staged_runtime["source_attention"]
        elif args.stage_source_caches_on_cpu_before_cells:
            # All length checkpoints already exist; a previous interruption
            # happened only during finalization.  Resume validates referenced
            # sessions and never reloads either model or recomputes a cache.
            source = target = mapper = None
            attention_contract = _strict_json(run.path / "attention_backend.json")
            mapper_setup = _strict_json(run.path / "mapper_setup.json")
        else:
            source, target, mapper, mapper_setup = load_runtime(args)
            attention_contract = runtime_attention_contract(
                source,
                target,
                mode=args.mode,
                expected=args.attention,
                lengths=lengths,
            )
        attention_path = run.path / "attention_backend.json"
        _write_immutable_json(attention_path, attention_contract)
        if _sha_sidecar_path(attention_path).exists():
            attention_sidecar = verify_sha_sidecar(attention_path)
        else:
            attention_sidecar = write_sha_sidecar(attention_path)
        mapper_setup_path = run.path / "mapper_setup.json"
        if mapper_setup_path.exists():
            recorded_setup = _strict_json(mapper_setup_path)
            for field in (
                "resident_parameter_bytes",
                "resident_dtype",
                "manifest_sha256",
                "fit_fingerprint",
                "execution",
                "model_layer_sharding",
                "source_model_live_during_timed_cells",
                "source_cache_staging_policy",
                "runtime_topology",
            ):
                if recorded_setup.get(field) != mapper_setup.get(field):
                    raise ValueError(f"resumed mapper setup changed field {field}")
            if _sha_sidecar_path(mapper_setup_path).exists():
                verify_sha_sidecar(mapper_setup_path)
            else:
                write_sha_sidecar(mapper_setup_path)
            mapper_setup_record = recorded_setup
        else:
            write_hashed_json(mapper_setup_path, mapper_setup)
            mapper_setup_record = mapper_setup
        memory_baseline = peak_memory_baseline(selected_devices)

        result: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "run_id": run.run_id,
            "request_fingerprint": request_fingerprint,
            "resumed": resumed,
            "protocol": configuration["local_conventions"],
            "math": math_contract,
            "hardware": hardware,
            "attention_backend": attention_contract,
            "attention_backend_assertion_sha256": attention_sidecar["sha256"],
            "mapper_setup": mapper_setup_record,
            "preload_gpu_idle_gate": preload_gate_record,
            "cuda_memory_baseline_after_runtime_load": memory_baseline,
            "source_cache_staging_policy": (
                SOURCE_CACHE_STAGING_POLICY
                if args.stage_source_caches_on_cpu_before_cells
                else None
            ),
            "active_staging_session": (
                {
                    "staging_session_id": staged_runtime["staging_session_id"],
                    "session_evidence_path": staged_runtime[
                        "session_evidence_path"
                    ],
                    "session_evidence_sha256": staged_runtime[
                        "session_evidence_sha256"
                    ],
                    "prepared_missing_lengths": list(missing_lengths),
                    "host_memory_staging_gate": staged_runtime[
                        "host_memory_staging_gate"
                    ],
                }
                if staged_runtime is not None
                else None
            ),
            "reused_checkpoint_lengths": [],
            "length_results": [],
        }
        for length in lengths:
            checkpoint_path = run.path / f"length_{length:05d}.json"
            if length in existing_checkpoint_lengths:
                current = verify_length_checkpoint(
                    checkpoint_path,
                    request_fingerprint=request_fingerprint,
                    expected_length=length,
                    expected_repeats=args.repeats,
                    expected_profile_repeats=profile_repeats,
                    expected_supplemental_diagnostics=args.supplemental_diagnostics,
                    attention_assertion_sha256=attention_sidecar["sha256"],
                    expected_device_ids=selected_devices,
                    expected_target_layers=expected_target_layers,
                    expected_mapped_cache_dtypes=expected_mapped_cache_dtypes,
                    expected_staged_source_lifecycle=(
                        args.stage_source_caches_on_cpu_before_cells
                    ),
                    expected_runtime_topology=runtime_topology_contract,
                )
                result["reused_checkpoint_lengths"].append(length)
                result["length_results"].append(current)
                continue
            failure_prefix_length = length
            failure_peak_memory = None
            failure_source_cache_lifecycle = None
            atomic_write_json(
                run.path / "status.json",
                {
                    "status": "benchmarking",
                    "prefix_length": length,
                    "request_fingerprint": request_fingerprint,
                    "completed_lengths": [
                        item["prefix_length"] for item in result["length_results"]
                    ],
                },
            )
            # Reassert the persisted dispatch contract immediately before the
            # 32K cell (and every other newly computed cell).  This prevents a
            # silent runtime mutation/fallback after model loading.
            if args.stage_source_caches_on_cpu_before_cells:
                if target is None or mapper is None or source_attention is None:
                    raise RuntimeError("staged runtime is absent for a missing length")
                current_attention = staged_runtime_attention_contract(
                    source_assertion=source_attention,
                    target_model=target,
                    expected=args.attention,
                    lengths=lengths,
                )
                vocabulary_size = int(source_vocabulary_size)
            else:
                current_attention = runtime_attention_contract(
                    source,
                    target,
                    mode=args.mode,
                    expected=args.attention,
                    lengths=lengths,
                )
                vocabulary_size = int(source.config.vocab_size)
            if current_attention != attention_contract:
                raise RuntimeError(
                    "attention backend contract changed after it was persisted"
                )
            input_ids = deterministic_ids(
                length,
                vocabulary_size=vocabulary_size,
                seed=args.seed,
            )
            prefix_ids = input_ids[:, :length]
            boundary_ids = input_ids[:, length:]
            prefix_identity = deterministic_prefix_identity(prefix_ids)
            source_cache: Any | None = None
            source_cache_ref: weakref.ReferenceType[Any] | None = None
            restage_evidence: dict[str, Any] | None = None
            preparation: Mapping[str, Any] | None = None
            if args.stage_source_caches_on_cpu_before_cells:
                preparation = preparations.get(str(length))
                if preparation is None or prefix_identity != preparation.get(
                    "deterministic_prefix_identity"
                ):
                    raise RuntimeError(
                        "cell token IDs differ from staged source-cache preparation"
                    )
                assert staged_runtime is not None
                failure_source_cache_lifecycle = {
                    "status": "gpu_restage_not_yet_complete",
                    "policy": SOURCE_CACHE_STAGING_POLICY,
                    "staging_session_id": staged_runtime["staging_session_id"],
                    "session_evidence_path": staged_runtime[
                        "session_evidence_path"
                    ],
                    "session_evidence_sha256": staged_runtime[
                        "session_evidence_sha256"
                    ],
                    "preparation_evidence_sha256": _canonical_digest(preparation),
                    "source_model_live_during_timed_cells": False,
                    "source_prefill_timed": False,
                    "cpu_staging_timed": False,
                    "gpu_restage_timed": False,
                    "outer_peak_reset_after_restage_validation": False,
                    "mapper_timer_starts_from_preexisting_gpu_source_cache": False,
                    "mapper_timer_started": False,
                    "deterministic_prefix_identity": prefix_identity,
                    "source_gpu_raw_bit_sha256": preparation[
                        "source_gpu_raw_bit_sha256"
                    ],
                    "cpu_staged_raw_bit_sha256": preparation[
                        "cpu_staged_raw_bit_sha256"
                    ],
                    "gpu_to_cpu_raw_bit_hash_matches": True,
                    "cache_total_bytes": source_cache_exact_bytes(length),
                }
                cpu_staged_cache = staged_caches.pop(length)
                try:
                    source_cache, restage_evidence = restage_source_cache_on_gpus(
                        cpu_staged_cache,
                        preparation=preparation,
                        source_devices=args.source_devices,
                        source_layer_counts=args.source_layer_counts,
                    )
                except BaseException as restage_exc:
                    traceback.clear_frames(restage_exc.__traceback__)
                    gc.collect()
                    synchronize_devices(args.source_devices)
                    for device in args.source_devices:
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                    synchronize_devices(args.source_devices)
                    failure_source_cache_lifecycle.update(
                        {
                            "status": "gpu_restage_failed_before_outer_peak_reset",
                            "restage_error_type": type(restage_exc).__name__,
                            "restage_error": str(restage_exc),
                            "restage_failure_cleanup": _cuda_allocator_driver_snapshot(
                                args.source_devices
                            ),
                        }
                    )
                    del cpu_staged_cache
                    raise
                del cpu_staged_cache
                gc.collect()
                source_cache_ref = weakref.ref(source_cache)
                failure_source_cache_lifecycle.update(
                    {
                        "status": "gpu_restage_verified_before_outer_peak_reset",
                        "restage": restage_evidence,
                        "outer_peak_reset_after_restage_validation": True,
                    }
                )
                # All restage copies, raw-bit D2H validation, synchronization,
                # and allocator cleanup above are explicitly outside the peak.
                try:
                    peak_memory_state = begin_cuda_peak_memory_measurement(
                        selected_devices,
                        prefix_length=length,
                        request_fingerprint=request_fingerprint,
                        measurement_scope=STAGED_CUDA_PEAK_MEASUREMENT_SCOPE,
                    )
                except BaseException:
                    del source_cache
                    source_cache = None
                    cache_release = release_restaged_source_cache(
                        cache_ref=source_cache_ref,
                        source_devices=args.source_devices,
                    )
                    failure_source_cache_lifecycle.update(
                        {
                            "status": "outer_peak_begin_failed_after_cache_release",
                            "restaged_cache_release": cache_release,
                        }
                    )
                    raise
                failure_source_cache_lifecycle.update(
                    {
                        "status": "outer_peak_reset_after_verified_gpu_restage",
                        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
                    }
                )
            else:
                peak_memory_state = begin_cuda_peak_memory_measurement(
                    selected_devices,
                    prefix_length=length,
                    request_fingerprint=request_fingerprint,
                )
            try:
                if source_cache is None:
                    source_cache = build_source_cache(source, prefix_ids)
                    source_cache_ref = weakref.ref(source_cache)
                current = run_one_length(
                    length=length,
                    source_cache=source_cache,
                    source_config=(
                        source_config
                        if args.stage_source_caches_on_cpu_before_cells
                        else source.config
                    ),
                    prefix_ids=prefix_ids,
                    boundary_ids=boundary_ids,
                    target_model=target,
                    mapper=mapper,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    profile_warmup=profile_warmup,
                    profile_repeats=profile_repeats,
                    token_chunk=args.mapper_token_chunk,
                    supplemental_diagnostics=args.supplemental_diagnostics,
                    collect_internal_cuda_peak_memory=False,
                    expected_prefix_identity=prefix_identity,
                    source_model_live_during_timed_cell=(
                        not args.stage_source_caches_on_cpu_before_cells
                    ),
                )
            except BaseException as cell_exc:
                traceback.clear_frames(cell_exc.__traceback__)
                del source_cache
                source_cache = None
                release_error: BaseException | None = None
                if args.stage_source_caches_on_cpu_before_cells:
                    try:
                        assert source_cache_ref is not None
                        failed_cell_release = release_restaged_source_cache(
                            cache_ref=source_cache_ref,
                            source_devices=args.source_devices,
                        )
                        assert failure_source_cache_lifecycle is not None
                        failure_source_cache_lifecycle.update(
                            {
                                "status": "cell_failed_cache_released_before_peak_finish",
                                "restaged_cache_release": failed_cell_release,
                            }
                        )
                    except BaseException as release_exc:
                        release_error = release_exc
                        assert failure_source_cache_lifecycle is not None
                        failure_source_cache_lifecycle.update(
                            {
                                "status": "cell_failed_cache_release_failed",
                                "cache_release_error_type": type(release_exc).__name__,
                                "cache_release_error": str(release_exc),
                            }
                        )
                else:
                    gc.collect()
                    synchronize_devices(selected_devices)
                    for device in selected_devices:
                        with torch.cuda.device(device):
                            torch.cuda.empty_cache()
                try:
                    failure_peak_memory = finish_cuda_peak_memory_measurement(
                        peak_memory_state
                    )
                except BaseException as peak_exc:
                    failure_peak_memory = cuda_peak_memory_capture_failure(
                        peak_memory_state,
                        peak_exc,
                        stage="after_cell_error",
                    )
                if release_error is not None:
                    raise release_error from cell_exc
                raise
            del source_cache
            source_cache = None
            if args.stage_source_caches_on_cpu_before_cells:
                assert source_cache_ref is not None
                try:
                    cache_release = release_restaged_source_cache(
                        cache_ref=source_cache_ref,
                        source_devices=args.source_devices,
                    )
                except BaseException as release_exc:
                    assert failure_source_cache_lifecycle is not None
                    failure_source_cache_lifecycle.update(
                        {
                            "status": "successful_cell_cache_release_failed",
                            "cache_release_error_type": type(release_exc).__name__,
                            "cache_release_error": str(release_exc),
                        }
                    )
                    try:
                        failure_peak_memory = finish_cuda_peak_memory_measurement(
                            peak_memory_state
                        )
                    except BaseException as peak_exc:
                        failure_peak_memory = cuda_peak_memory_capture_failure(
                            peak_memory_state,
                            peak_exc,
                            stage="after_restaged_cache_release_error",
                        )
                    raise release_exc
                assert preparation is not None and restage_evidence is not None
                assert failure_source_cache_lifecycle is not None
                current["source_cache_lifecycle"] = {
                    "status": "complete",
                    "policy": SOURCE_CACHE_STAGING_POLICY,
                    "staging_session_id": staged_runtime["staging_session_id"],
                    "session_evidence_path": staged_runtime[
                        "session_evidence_path"
                    ],
                    "session_evidence_sha256": staged_runtime[
                        "session_evidence_sha256"
                    ],
                    "preparation_evidence_sha256": _canonical_digest(preparation),
                    "source_model_live_during_timed_cells": False,
                    "source_prefill_timed": False,
                    "cpu_staging_timed": False,
                    "gpu_restage_timed": False,
                    "outer_peak_reset_after_restage_validation": True,
                    "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
                    "mapper_timer_started": True,
                    "deterministic_prefix_identity": prefix_identity,
                    "source_gpu_raw_bit_sha256": preparation[
                        "source_gpu_raw_bit_sha256"
                    ],
                    "cpu_staged_raw_bit_sha256": preparation[
                        "cpu_staged_raw_bit_sha256"
                    ],
                    "gpu_to_cpu_raw_bit_hash_matches": True,
                    "cache_total_bytes": source_cache_exact_bytes(length),
                    "restage": restage_evidence,
                    "restaged_cache_release": cache_release,
                }
                failure_source_cache_lifecycle = dict(
                    current["source_cache_lifecycle"]
                )
            else:
                gc.collect()
                synchronize_devices(selected_devices)
                for device in selected_devices:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
            try:
                current["cuda_peak_memory"] = finish_cuda_peak_memory_measurement(
                    peak_memory_state
                )
            except BaseException as peak_exc:
                failure_peak_memory = cuda_peak_memory_capture_failure(
                    peak_memory_state,
                    peak_exc,
                    stage="after_successful_cell",
                )
                raise
            failure_peak_memory = current["cuda_peak_memory"]
            checkpoint = make_length_checkpoint(
                current,
                request_fingerprint=request_fingerprint,
                attention_assertion_sha256=attention_sidecar["sha256"],
            )
            write_hashed_json(checkpoint_path, checkpoint)
            verified = verify_length_checkpoint(
                checkpoint_path,
                request_fingerprint=request_fingerprint,
                expected_length=length,
                expected_repeats=args.repeats,
                expected_profile_repeats=profile_repeats,
                expected_supplemental_diagnostics=args.supplemental_diagnostics,
                attention_assertion_sha256=attention_sidecar["sha256"],
                expected_device_ids=selected_devices,
                expected_target_layers=expected_target_layers,
                expected_mapped_cache_dtypes=expected_mapped_cache_dtypes,
                expected_staged_source_lifecycle=(
                    args.stage_source_caches_on_cpu_before_cells
                ),
                expected_runtime_topology=runtime_topology_contract,
            )
            result["length_results"].append(verified)
            write_hashed_json(run.path / "metrics.partial.json", result)
            write_hashed_csv(run.path / "metrics.partial.csv", result)
            failure_prefix_length = None
            failure_peak_memory = None
            failure_source_cache_lifecycle = None

        if args.stage_source_caches_on_cpu_before_cells and staged_caches:
            raise RuntimeError(
                "completed staged sweep retained unconsumed CPU source caches: "
                f"{sorted(staged_caches)}"
            )
        result["status"] = "complete"
        result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        after = telemetry_snapshot()
        result["telemetry_before"] = before
        result["telemetry_after"] = after
        write_hashed_json(run.path / "telemetry_after.json", after)
        write_hashed_json(run.path / "metrics.json", result)
        write_hashed_csv(run.path / "metrics.csv", result)
        manifest_path, _ = write_results_manifest(
            run,
            request_fingerprint=request_fingerprint,
            lengths=lengths,
        )
        final_result = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "run_id": run.run_id,
            "request_fingerprint": request_fingerprint,
            "results_manifest": manifest_path.name,
            "results_manifest_sha256": sha256_file(manifest_path),
            "metrics_sha256": sha256_file(run.path / "metrics.json"),
            "metrics_csv_sha256": sha256_file(run.path / "metrics.csv"),
        }
        write_hashed_json(run.path / "result.json", final_result)
        atomic_write_json(
            run.path / "status.json",
            {"status": "complete", "request_fingerprint": request_fingerprint},
        )
        if not args.no_ledger:
            append_reproduction_log(
                ROOT,
                heading=f"P8-LATENCY-{run.run_id}",
                bullets=[
                    f"完成 `{args.mode}` 延迟评测，prefix lengths={list(lengths)}。",
                    "source prefill 未计时；论文主协议只保存 target transformer re-prefill "
                    "与 mapper handoff；component/boundary/protocol 仅在显式 supplemental 模式运行。",
                    "每个长度先通过不计时 strict 数值 preflight；正式计时路径关闭诊断 reduction，"
                    "保留 shape/dtype/architecture 校验。",
                    "每个长度在同步后重置所有选定 CUDA device 的 allocator peak counters；"
                    "所有内部 metric/profile 禁止再次 reset；checkpoint 强制保存"
                    " allocated/reserved peak、容量及 cell 前后 driver free。",
                    f"request fingerprint={request_fingerprint}；JSON/CSV/逐长度结果均有 SHA256，"
                    "并支持同一 `--output-dir` 严格续跑。",
                    "本机为无 NVLink/P2P 的 RTX 4090；绝对时间不直接与论文 8×H100 数值比较。",
                    f"JSON/CSV 与逐长度 checkpoint：`{run.path}`。",
                ],
            )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "run": str(run.path),
                    "request_fingerprint": request_fingerprint,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed",
            "request_fingerprint": request_fingerprint,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if failure_prefix_length is not None:
            failure["prefix_length"] = failure_prefix_length
        if failure_peak_memory is not None:
            failure["cuda_peak_memory"] = failure_peak_memory
        if failure_source_cache_lifecycle is not None:
            failure["source_cache_lifecycle"] = failure_source_cache_lifecycle
        write_hashed_json(run.path / "result.json", failure)
        atomic_write_json(run.path / "status.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
