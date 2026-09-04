#!/usr/bin/env python3
"""Measure mapped-vs-ground-truth target attention-output cosine.

This is a post-hoc numerical diagnostic, not an accuracy/PPL benchmark and not
part of the formal evaluation code fingerprint.  It uses a deterministic small
slice of the local FineWeb-Edu held-out tokens and the paper's completed mapper.
See ``docs/ATTENTION_OUTPUT_COSINE.md`` for the exact local convention.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.attention_output_diagnostic import (  # noqa: E402
    LayerCosineAccumulator,
    TargetTrajectoryAttentionCosineProbe,
    aggregate_layer_accumulators,
)
from kvcache_transfer.evaluation import MappedPrefixCacheBuilder  # noqa: E402
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.modeling import load_qwen3  # noqa: E402
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
)
from kvcache_transfer.transfer import MapperBundle  # noqa: E402


DEFAULT_SOURCE = Path("/home/wby/models/Qwen3-14B")
DEFAULT_TARGET = Path("/home/wby/models/Qwen3-32B")
DEFAULT_MAPPER = ROOT / "artifacts/mappers/qwen3_14b_to_32b/k8_lambda0.01/manifest.json"
DEFAULT_CALIBRATION = ROOT / "data/calibration/qwen3_fineweb_edu_500x1024_v1"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/validation/qwen3_14b_to_32b/attention_output_cosine_heldout.json"
)
DIAGNOSTIC_CODE_FILES = (
    "scripts/diagnose_attention_output_cosine.py",
    "src/kvcache_transfer/attention_output_diagnostic.py",
)


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("integer list cannot be empty")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("integer list cannot contain duplicates")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-model", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--mapper-manifest", type=Path, default=DEFAULT_MAPPER)
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-indices",
        type=_csv_ints,
        default=(0, 1, 2, 3),
        help="held-out sequence row indices (default: 0,1,2,3)",
    )
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument(
        "--query-token-indices",
        type=_csv_ints,
        default=(63, 127, 191, 255),
        help="absolute token positions to probe (default: 63,127,191,255)",
    )
    parser.add_argument("--source-devices", type=_csv_ints, default=(4, 5))
    parser.add_argument("--target-devices", type=_csv_ints, default=(0, 1, 2, 3))
    parser.add_argument("--mapper-token-chunk", type=int, default=256)
    parser.add_argument("--attention-query-chunk", type=int, default=4)
    parser.add_argument(
        "--attention-implementation",
        choices=("flash_attention_2", "eager", "sdpa"),
        default="flash_attention_2",
        help="target/source prefill implementation; diagnostic attention itself is explicit FP32",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "verify full mapper/data hashes, model config/tokenizer/index content hashes, "
            "model shard filenames/sizes, and software identity without loading models or "
            "using a GPU"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacing the default aggregate artifact; every invocation still has a new run",
    )
    return parser.parse_args(argv)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("result JSON contains NaN or Inf")
    if isinstance(value, dict):
        for nested in value.values():
            _finite_json(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_json(nested)


def _write_immutable_payload(path: Path, value: Any) -> tuple[str, Path]:
    """Atomically write one full payload and a content-hash commit marker."""

    destination = path.resolve()
    atomic_write_json(destination, value)
    payload_sha256 = sha256_file(destination)
    marker = destination.with_suffix(".sha256.json")
    atomic_write_json(
        marker,
        {
            "schema_version": 1,
            "path": str(destination),
            "sha256": payload_sha256,
            "nbytes": destination.stat().st_size,
        },
    )
    if sha256_file(destination) != payload_sha256:
        raise RuntimeError(f"payload changed while committing marker: {destination}")
    return payload_sha256, marker


def _model_identity(model_root: Path) -> dict[str, Any]:
    """Bind small metadata content and shard names/sizes without huge weight hashes.

    This identity intentionally has the same scope as the formal benchmark:
    weight shard *contents* are not hashed.  A same-size in-place shard change
    is therefore outside the identity guarantee and this limitation is stated
    explicitly in every result rather than implying cryptographic weight
    provenance.
    """

    root = model_root.resolve()
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    content_hashed: dict[str, Any] = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required model identity file is missing: {path}")
        content_hashed[name] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    for name in ("tokenizer_config.json", "special_tokens_map.json"):
        path = root / name
        if path.is_file():
            content_hashed[name] = {
                "sha256": sha256_file(path),
                "nbytes": path.stat().st_size,
            }
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"model index has no weight_map: {root}")
    shards: list[dict[str, Any]] = []
    for name in sorted({str(item) for item in weight_map.values()}):
        path = (root / name).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError as exc:
            raise ValueError(f"model shard escapes root: {name}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"model shard is missing: {path}")
        shards.append({"path": relative, "nbytes": path.stat().st_size})
    return {
        "path": str(root),
        "content_hashed_files": content_hashed,
        "weight_shards": shards,
        "weight_shard_count": len(shards),
        "weight_shard_total_nbytes": sum(row["nbytes"] for row in shards),
        "weight_shard_identity_scope": "index_sha256_plus_relative_filename_and_nbytes",
        "weight_shard_contents_hashed": False,
        "cryptographic_weight_content_identity": False,
        "weight_identity_limitation": (
            "a weight-shard content change that preserves its relative filename and byte "
            "size is not detected; config/tokenizer/safetensors-index files are content-hashed"
        ),
    }


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _software_identity() -> dict[str, Any]:
    """Record the software/kernel identity that can affect attention values."""

    from transformers.models.qwen3 import modeling_qwen3
    from transformers.utils import is_flash_attn_2_available

    modeling_path = Path(modeling_qwen3.__file__).resolve()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": _distribution_version("transformers"),
        "accelerate": _distribution_version("accelerate"),
        "flash_attention": _distribution_version("flash-attn", "flash_attn"),
        "torch_cuda_build": torch.version.cuda,
        "cudnn_runtime": torch.backends.cudnn.version(),
        "cuda_available_at_preflight": bool(torch.cuda.is_available()),
        "cuda_visible_device_count_at_preflight": int(torch.cuda.device_count()),
        "transformers_flash_attention_2_available": bool(is_flash_attn_2_available()),
        "qwen3_modeling_implementation": {
            "path": str(modeling_path),
            "sha256": sha256_file(modeling_path),
            "nbytes": modeling_path.stat().st_size,
            "module": "transformers.models.qwen3.modeling_qwen3",
        },
        "identity_scope": (
            "installed Python package versions, Torch CUDA build/runtime visibility, and "
            "content hash of the exact Transformers Qwen3 implementation file"
        ),
    }


def _code_identity() -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative in DIAGNOSTIC_CODE_FILES:
        path = ROOT / relative
        files[relative] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    return {"files": files, "files_digest": _canonical_digest(files)}


def _formal_evaluation_code_identity() -> dict[str, Any]:
    """Read the formal file list without importing or modifying its script."""

    evaluate_path = ROOT / "scripts/evaluate.py"
    tree = ast.parse(evaluate_path.read_text(encoding="utf-8"), filename=str(evaluate_path))
    formal_files: tuple[str, ...] | None = None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "FORMAL_EVALUATION_CODE_FILES"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            formal_files = tuple(str(item) for item in value)
            break
    if not formal_files:
        raise ValueError("cannot resolve FORMAL_EVALUATION_CODE_FILES from scripts/evaluate.py")
    files: dict[str, Any] = {}
    for relative in formal_files:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"formal evaluation code file is missing: {path}")
        files[relative] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    overlap = sorted(set(formal_files) & set(DIAGNOSTIC_CODE_FILES))
    if overlap:
        raise ValueError(f"diagnostic code overlaps formal evaluation identity: {overlap}")
    return {
        "files": files,
        "files_digest": _canonical_digest(files),
        "diagnostic_code_files_disjoint": True,
    }


def _load_heldout_tokens(
    calibration_dir: Path,
    *,
    sample_indices: Sequence[int],
    sequence_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    root = calibration_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("calibration manifest is not complete")
    split = manifest.get("splits", {}).get("heldout")
    if not isinstance(split, dict):
        raise ValueError("calibration manifest has no heldout split")
    token_path = (root / str(split.get("file"))).resolve()
    expected_hash = manifest.get("artifacts", {}).get(token_path.name, {}).get("sha256")
    observed_hash = sha256_file(token_path)
    if expected_hash != observed_hash:
        raise ValueError("heldout token file SHA256 does not match calibration manifest")
    matrix = np.load(token_path, mmap_mode="r", allow_pickle=False)
    expected_shape = tuple(int(value) for value in split.get("shape", ()))
    if matrix.shape != expected_shape or matrix.dtype != np.dtype(str(split.get("dtype"))):
        raise ValueError(
            f"heldout token matrix disagrees with manifest: {matrix.shape}/{matrix.dtype}"
        )
    if sequence_length <= 0 or sequence_length > matrix.shape[1]:
        raise ValueError(f"sequence length must lie in [1,{matrix.shape[1]}]")
    if any(index < 0 or index >= matrix.shape[0] for index in sample_indices):
        raise ValueError(f"sample indices must lie in [0,{matrix.shape[0]})")
    selected = np.asarray(matrix[list(sample_indices), :sequence_length], dtype=np.int64)
    token_payload_sha = hashlib.sha256(selected.astype("<i8", copy=False).tobytes()).hexdigest()
    return selected, {
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_sha256": sha256_file(manifest_path),
        "heldout_token_file": str(token_path),
        "heldout_token_file_sha256": observed_hash,
        "heldout_full_shape": list(matrix.shape),
        "heldout_full_dtype": str(matrix.dtype),
        "selected_sample_indices": [int(value) for value in sample_indices],
        "selected_sequence_length": sequence_length,
        "selected_tokens_little_endian_int64_sha256": token_payload_sha,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.mapper_token_chunk <= 0 or args.attention_query_chunk <= 0:
        raise ValueError("chunk sizes must be positive")
    if any(position < 0 or position >= args.sequence_length for position in args.query_token_indices):
        raise ValueError(
            f"query token indices must lie in [0,{args.sequence_length})"
        )
    source_devices = tuple(int(value) for value in args.source_devices)
    target_devices = tuple(int(value) for value in args.target_devices)
    if any(value < 0 for value in (*source_devices, *target_devices)):
        raise ValueError("CUDA device indices must be non-negative")
    if set(source_devices) & set(target_devices):
        raise ValueError("source and target CUDA device sets must be disjoint")
    if not source_devices or not target_devices:
        raise ValueError("source and target device lists cannot be empty")
    for label, path in (
        ("output", args.output),
        ("mapper manifest", args.mapper_manifest),
        ("calibration directory", args.calibration_dir),
    ):
        try:
            path.resolve().relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"{label} must stay inside the reproduction project: {path}") from exc


def _preflight(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    _validate_args(args)
    tokens, data_identity = _load_heldout_tokens(
        args.calibration_dir,
        sample_indices=args.sample_indices,
        sequence_length=args.sequence_length,
    )
    manifest_path = args.mapper_manifest.resolve()
    mapper_root = manifest_path if manifest_path.is_dir() else manifest_path.parent
    mapper = MapperBundle.open(mapper_root)
    if manifest_path.is_file() and mapper.manifest_path != manifest_path:
        raise ValueError("resolved mapper manifest differs from requested path")
    source_identity = _model_identity(args.source_model)
    target_identity = _model_identity(args.target_model)
    identity = {
        "schema_version": 1,
        "status": "passed",
        "method": "mapped-vs-ground-truth-target-attention-output-cosine",
        "paper_scope": (
            "paper section 4.5 defines cosine between mapped-KV and ground-truth-KV "
            "attention outputs, averaged over layers and heads"
        ),
        "local_convention": {
            "query_trajectory": "target ground-truth prefill hidden states",
            "query_space": "post-q_norm and post-target-RoPE",
            "ground_truth_k_space": "post-k_norm and post-target-RoPE",
            "mapped_k_space": "production mapper post-target-RoPE BF16 cache",
            "value_space": "target/mapped cache value space",
            "attention_output_location": "per query head before target o_proj",
            "attention_math": "explicit causal GQA softmax in FP32",
            "cosine_math": "rowwise FP64 with both-zero=1 and one-zero=0",
            "aggregation": (
                "explicit equal arithmetic mean of target-layer means; identical per-head "
                "count vectors are mandatory across layers and identical head counts are "
                "mandatory within each layer"
            ),
            "domain": "local FineWeb-Edu heldout diagnostic, not paper HellaSwag analysis",
            "padding": "none; contiguous key positions 0..T-1",
        },
        "protocol": {
            "sample_indices": [int(value) for value in args.sample_indices],
            "sequence_length": int(args.sequence_length),
            "query_token_indices": [int(value) for value in args.query_token_indices],
            "source_devices": [int(value) for value in args.source_devices],
            "target_devices": [int(value) for value in args.target_devices],
            "mapper_token_chunk": int(args.mapper_token_chunk),
            "attention_query_chunk": int(args.attention_query_chunk),
            "prefill_attention_implementation": args.attention_implementation,
        },
        "data": data_identity,
        "source_model": source_identity,
        "target_model": target_identity,
        "software_runtime": _software_identity(),
        "mapper": {
            "manifest": str(mapper.manifest_path),
            "manifest_sha256": mapper.manifest_sha256,
            "fit_fingerprint": mapper.fit_fingerprint,
            "target_layers": mapper.architecture.target_layers,
            "selected_source_layers": mapper.architecture.selected_source_layers,
        },
        "diagnostic_code": _code_identity(),
        "formal_evaluation_code_unchanged_guard": _formal_evaluation_code_identity(),
    }
    identity["diagnostic_fingerprint"] = _canonical_digest(identity)
    return tokens, identity


@torch.inference_mode()
def _run_diagnostic(
    args: argparse.Namespace,
    tokens: np.ndarray,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("formal attention-output diagnostic requires CUDA")
    requested = (*args.source_devices, *args.target_devices)
    if max(requested) >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested CUDA device {max(requested)}, only {torch.cuda.device_count()} visible"
        )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_model = load_qwen3(
        args.source_model,
        devices=args.source_devices,
        role="source",
        attention_implementation=args.attention_implementation,
    )
    target_model = load_qwen3(
        args.target_model,
        devices=args.target_devices,
        role="target",
        attention_implementation=args.attention_implementation,
    )
    cache_builder = MappedPrefixCacheBuilder(
        source_model=source_model,
        target_model=target_model,
        mapper_manifest=args.mapper_manifest,
        token_chunk=args.mapper_token_chunk,
    )
    target_base = getattr(target_model, "model", target_model)
    target_input_device = target_base.embed_tokens.weight.device
    target_layers = int(target_model.config.num_hidden_layers)
    query_heads = int(target_model.config.num_attention_heads)
    aggregate = [LayerCosineAccumulator(query_heads=query_heads) for _ in range(target_layers)]
    per_sample: list[dict[str, Any]] = []
    sample_seconds: list[float] = []
    for local_index, token_row in enumerate(tokens):
        started = time.perf_counter()
        prefix = [int(value) for value in token_row]
        mapped_cache = cache_builder(prefix)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=target_input_device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=target_input_device
        ).unsqueeze(0)
        probe = TargetTrajectoryAttentionCosineProbe(
            target_model,
            mapped_cache,
            query_positions=args.query_token_indices,
            key_valid_mask=attention_mask,
            query_chunk=args.attention_query_chunk,
        )
        with probe:
            target_base(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=torch.arange(
                    input_ids.shape[1], dtype=torch.long, device=target_input_device
                ),
                use_cache=False,
                return_dict=True,
            )
        probe.validate_forward(expected_calls_per_layer=1)
        for destination, source in zip(aggregate, probe.accumulators, strict=True):
            destination.merge(source)
        sample_summary = aggregate_layer_accumulators(probe.accumulators)
        elapsed = time.perf_counter() - started
        sample_seconds.append(elapsed)
        per_sample.append(
            {
                "heldout_sample_index": int(args.sample_indices[local_index]),
                "mean_cosine_fp64": sample_summary["mean_cosine_fp64"],
                "min_row_cosine_fp64": sample_summary["min_row_cosine_fp64"],
                "max_row_cosine_fp64": sample_summary["max_row_cosine_fp64"],
                "row_observations": sample_summary["row_observations"],
                "elapsed_seconds_diagnostic_not_benchmark": elapsed,
            }
        )
        del mapped_cache, input_ids, attention_mask, position_ids, probe
    summary = aggregate_layer_accumulators(aggregate)
    expected_rows = (
        len(args.sample_indices)
        * len(args.query_token_indices)
        * target_layers
        * query_heads
    )
    if summary["row_observations"] != expected_rows:
        raise AssertionError(
            f"diagnostic row coverage {summary['row_observations']} != {expected_rows}"
        )
    return {
        "schema_version": 1,
        "status": "complete",
        "diagnostic_fingerprint": preflight["diagnostic_fingerprint"],
        "summary": summary,
        "per_sample": per_sample,
        "coverage": {
            "samples": len(args.sample_indices),
            "selected_query_tokens_per_sample": len(args.query_token_indices),
            "target_layers": target_layers,
            "query_heads": query_heads,
            "expected_and_observed_rows": expected_rows,
        },
        "mapper_runtime": cache_builder.provenance,
        "runtime": {
            "sample_elapsed_seconds_diagnostic_not_benchmark": sample_seconds,
            "total_sample_elapsed_seconds_diagnostic_not_benchmark": sum(sample_seconds),
            "cuda_device_names": {
                str(index): torch.cuda.get_device_name(index) for index in sorted(set(requested))
            },
            "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "interpretation": (
            "post-hoc local numerical diagnostic; it does not replace benchmark retention, "
            "does not reproduce the paper's undisclosed HellaSwag token sample, and its wall "
            "time is not a mapper latency measurement"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run = make_run_directory(ROOT, "p9", "attention_output_cosine")
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    record_run_start(
        run,
        list(sys.argv if argv is None else [sys.argv[0], *argv]),
        {
            "source_model": str(args.source_model.resolve()),
            "target_model": str(args.target_model.resolve()),
            "mapper_manifest": str(args.mapper_manifest.resolve()),
            "calibration_dir": str(args.calibration_dir.resolve()),
            "output": str(args.output.resolve()),
            "sample_indices": list(args.sample_indices),
            "sequence_length": args.sequence_length,
            "query_token_indices": list(args.query_token_indices),
            "source_devices": list(args.source_devices),
            "target_devices": list(args.target_devices),
            "mapper_token_chunk": args.mapper_token_chunk,
            "attention_query_chunk": args.attention_query_chunk,
            "attention_implementation": args.attention_implementation,
            "preflight_only": args.preflight_only,
            "formal_accuracy_or_ppl_protocol": False,
        },
    )
    started = time.perf_counter()
    try:
        output = args.output.resolve()
        if not args.preflight_only and output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
        tokens, preflight = _preflight(args)
        preflight_path = run.path / "preflight.json"
        preflight_sha, preflight_marker = _write_immutable_payload(preflight_path, preflight)
        if args.preflight_only:
            result = {
                "schema_version": 1,
                "status": "preflight-complete",
                "diagnostic_fingerprint": preflight["diagnostic_fingerprint"],
                "preflight_payload": str(preflight_path),
                "preflight_payload_sha256": preflight_sha,
                "preflight_commit_marker": str(preflight_marker),
                "elapsed_seconds": time.perf_counter() - started,
                "run": str(run.path),
            }
            atomic_write_json(run.path / "result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        result = _run_diagnostic(args, tokens, preflight)
        result["started_utc"] = started_utc
        result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        result["preflight"] = preflight
        _finite_json(result)
        # The run-local payload is the immutable scientific record.  The
        # aggregate output is a byte-identical convenience copy that may be
        # explicitly replaced by a later --overwrite run without erasing this
        # run's full payload.
        run_payload = run.path / "attention_output_cosine.json"
        run_payload_sha, run_payload_marker = _write_immutable_payload(
            run_payload, result
        )
        output_sha, output_marker = _write_immutable_payload(output, result)
        if output_sha != run_payload_sha or output.read_bytes() != run_payload.read_bytes():
            raise RuntimeError(
                "aggregate output is not byte-identical to the immutable run payload"
            )
        run_result = {
            "schema_version": 1,
            "status": "complete",
            "immutable_run_payload": str(run_payload),
            "immutable_run_payload_sha256": run_payload_sha,
            "immutable_run_payload_commit_marker": str(run_payload_marker),
            "aggregate_output": str(output),
            "aggregate_output_sha256": output_sha,
            "aggregate_output_commit_marker": str(output_marker),
            "aggregate_overwrite_requested": bool(args.overwrite),
            "preflight_payload": str(preflight_path),
            "preflight_payload_sha256": preflight_sha,
            "preflight_commit_marker": str(preflight_marker),
            "diagnostic_fingerprint": preflight["diagnostic_fingerprint"],
            "mean_cosine_fp64": result["summary"]["mean_cosine_fp64"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(run.path / "result.json", run_result)
        append_reproduction_log(
            ROOT,
            heading=f"P9-ATTENTION-COSINE-{run.run_id}",
            bullets=[
                "状态：completed；这是论文 4.5 attention-output cosine 的本地 held-out 数值诊断，"
                "不替代 accuracy/PPL retention。",
                "本地冻结口径：target ground-truth trajectory Q；同一 causal prefix；"
                "o_proj 前逐 query-head FP32 explicit attention；FP64 row cosine；"
                "headline 为 64 个 target-layer mean 的严格等权算术平均，且每层/每 head "
                "observation count 完全相同。",
                f"mean cosine={result['summary']['mean_cosine_fp64']:.10f}；"
                f"rows={result['summary']['row_observations']}；NaN/Inf=0。",
                f"run 内完整不可变 payload：`{run_payload.relative_to(ROOT)}`；"
                f"SHA256={run_payload_sha}；aggregate=`{output.relative_to(ROOT)}`；"
                f"run=`{run.path.relative_to(ROOT)}`。",
            ],
        )
        print(json.dumps(run_result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_utc": started_utc,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        failure_path = run.path / "failure.json"
        failure_sha, failure_marker = _write_immutable_payload(failure_path, failure)
        atomic_write_json(
            run.path / "result.json",
            {
                "schema_version": 1,
                "status": "failed",
                "failure_payload": str(failure_path),
                "failure_payload_sha256": failure_sha,
                "failure_commit_marker": str(failure_marker),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
