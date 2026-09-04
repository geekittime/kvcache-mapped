#!/usr/bin/env python3
"""Resumably extract paper calibration KV activations from one Qwen3 endpoint.

The 500-sequence production capture is a long multi-GPU job.  All layer files
are therefore written into a private ``.<output>.inprogress`` tree.  A sequence
is considered durable only after every layer payload has been flushed/fsynced
and ``progress.json`` is atomically advanced.  Restart treats that common
sequence count as authoritative: short payloads are rejected, while bytes
written after the last durable checkpoint are truncated and fsynced before
the model is loaded.

Once all sequences are present, complete payloads receive normal activation
sidecars, a cross-file manifest is committed, and the staging directory is
atomically renamed to the requested output.  The final directory never exposes
a mixture of committed and incomplete layers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import transformers
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.activation_store import (  # noqa: E402
    commit_existing_bf16_payload,
    read_activation_metadata,
    sidecar_path,
)
from kvcache_transfer.capture import capture_sampled_content_kv  # noqa: E402
from kvcache_transfer.config import assert_paper_architecture  # noqa: E402
from kvcache_transfer.data import verify_calibration_output  # noqa: E402
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.modeling import load_qwen3  # noqa: E402
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
)


SCHEMA_VERSION = 2
SAMPLED_PER_SEQUENCE = 256
KV_HEADS = 8
HEAD_DIM = 128
ROW_NBYTES = KV_HEADS * HEAD_DIM * 2
CACHE_TYPES = ("k_content", "v")
MIN_CAPTURE_HEADROOM_BYTES = 2 << 30
_ACTIVE_RUN_PATH: Path | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-name", required=True, choices=["qwen3_14b", "qwen3_32b"]
    )
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "heldout"])
    parser.add_argument("--devices", required=True, help="comma-separated CUDA indices")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-limit", type=int, help="debug-only prefix limit")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--fsync-every", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args(argv)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _model_identity(model_path: Path, role: str) -> dict[str, Any]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert_paper_architecture(config, role=role)
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = _strict_json(index_path)
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError(f"model index has no weight shards: {index_path}")
    shards = []
    for name in shard_names:
        path = model_path / str(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        # Hashing every 28/62 GiB checkpoint twice is wasteful.  The immutable
        # index hash plus exact shard names/sizes is the local checkpoint
        # identity; content hashes supplied by the original download are not
        # available in these model directories.
        shards.append({"name": str(name), "bytes": path.stat().st_size})
    tokenizer_path = model_path / "tokenizer.json"
    config_path = model_path / "config.json"
    return {
        "path": str(model_path.resolve()),
        "role": role,
        "config_sha256": sha256_file(config_path),
        "weight_index_sha256": sha256_file(index_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "weight_shards": shards,
        "architecture": {
            "model_type": config.model_type,
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.head_dim),
            "rope_theta": float(config.rope_theta),
        },
    }


def _calibration_identity(tokens: Path, split: str) -> tuple[dict[str, Any], int]:
    manifest_path = tokens.parent / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"formal capture requires calibration manifest beside tokens: {manifest_path}"
        )
    manifest = verify_calibration_output(tokens.parent)
    split_metadata = manifest.get("splits", {}).get(split)
    if not isinstance(split_metadata, Mapping):
        raise ValueError(f"calibration manifest has no split {split!r}")
    expected_file = tokens.parent / str(split_metadata["file"])
    if tokens.resolve() != expected_file.resolve():
        raise ValueError(
            f"--tokens does not match calibration manifest split {split}: {expected_file}"
        )
    token_sha = sha256_file(tokens)
    artifact = manifest.get("artifacts", {}).get(tokens.name)
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != token_sha:
        raise ValueError("token payload SHA256 is not bound by calibration manifest")
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_fingerprint": str(manifest["dataset_fingerprint"]),
        "config_fingerprint": str(manifest["config_fingerprint"]),
        "token_path": str(tokens.resolve()),
        "token_sha256": token_sha,
        "split": split,
    }, int(split_metadata["shape"][0])


def _payload_relative(cache_type: str, layer: int) -> Path:
    return Path(cache_type) / f"layer_{layer:03d}.bf16.bin"


def _stage_for(output: Path) -> Path:
    return output.parent / f".{output.name}.inprogress"


def _expected_payloads(stage: Path, layers: int) -> list[Path]:
    return [
        stage / _payload_relative(cache_type, layer)
        for cache_type in CACHE_TYPES
        for layer in range(layers)
    ]


def _capture_storage_preflight(
    stage: Path, layers: int, sequences: int
) -> dict[str, int]:
    """Prove that the filesystem can finish the unpublished BF16 payloads."""

    per_payload_bytes = sequences * SAMPLED_PER_SEQUENCE * ROW_NBYTES
    payloads = _expected_payloads(stage, layers)
    existing_bytes = sum(
        min(path.stat().st_size, per_payload_bytes) for path in payloads if path.exists()
    )
    final_payload_bytes = len(payloads) * per_payload_bytes
    remaining_payload_bytes = max(final_payload_bytes - existing_bytes, 0)
    free_bytes = shutil.disk_usage(stage.parent).free
    required_free_bytes = remaining_payload_bytes + MIN_CAPTURE_HEADROOM_BYTES
    if free_bytes < required_free_bytes:
        raise OSError(
            "insufficient free space for capture: "
            f"free={free_bytes}, remaining_payload={remaining_payload_bytes}, "
            f"required_with_headroom={required_free_bytes}"
        )
    return {
        "payload_files": len(payloads),
        "per_payload_bytes": per_payload_bytes,
        "estimated_final_payload_bytes": final_payload_bytes,
        "existing_staging_payload_bytes": existing_bytes,
        "remaining_payload_bytes": remaining_payload_bytes,
        "filesystem_free_bytes": free_bytes,
        "minimum_headroom_bytes": MIN_CAPTURE_HEADROOM_BYTES,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bf16_chunk(handle: Any, value: torch.Tensor) -> None:
    tensor = value.detach().squeeze(0).to(device="cpu", dtype=torch.bfloat16).contiguous()
    if tuple(tensor.shape) != (SAMPLED_PER_SEQUENCE, KV_HEADS, HEAD_DIM):
        raise ValueError(f"unexpected captured activation shape: {tuple(tensor.shape)}")
    payload = memoryview(tensor.view(torch.uint16).numpy()).cast("B")
    written = 0
    while written < len(payload):
        count = handle.write(payload[written:])
        if count is None or count <= 0:
            raise OSError(
                f"activation payload short write: wrote {written} of {len(payload)} bytes"
            )
        written += count


def _validate_progress_payloads(
    stage: Path, layers: int, completed_sequences: int
) -> None:
    """Restore staging payloads to the last common durable checkpoint.

    A process can stop after appending any byte of the next sequence, or while
    fsyncing the layer files but before advancing ``progress.json``.  Those
    suffixes are uncommitted even when they contain one or more whole
    sequences.  Conversely, a payload shorter than the durable checkpoint
    means acknowledged data was lost and must never be papered over.

    Validate every length before changing anything.  If repair is needed,
    invalidate staging-only commit markers before truncating payloads so a
    second crash cannot leave metadata that claims a now-modified payload.
    The published output directory never reaches this recovery path.
    """

    expected_bytes = completed_sequences * SAMPLED_PER_SEQUENCE * ROW_NBYTES
    payloads = _expected_payloads(stage, layers)
    sizes: dict[Path, int] = {}
    for payload in payloads:
        if not payload.is_file():
            raise ValueError(f"resume payload is missing: {payload}")
        actual = payload.stat().st_size
        sizes[payload] = actual
        if actual < expected_bytes:
            raise ValueError(
                f"resume payload length mismatch {payload}: {actual} is shorter than "
                f"durable checkpoint {expected_bytes}"
            )

    overlong = [payload for payload in payloads if sizes[payload] > expected_bytes]
    if not overlong:
        return

    # Sidecars and the aggregate manifest are commit markers only inside this
    # unpublished staging tree.  Invalidate all of them before changing a
    # payload: a stale progress rename can coexist with a partially completed
    # sidecar pass after a host crash.
    changed_directories: set[Path] = set()
    for marker in [stage / "manifest.json", *(sidecar_path(path) for path in payloads)]:
        try:
            marker.unlink()
        except FileNotFoundError:
            continue
        changed_directories.add(marker.parent)
    for directory in sorted(changed_directories):
        _fsync_directory(directory)

    for payload in overlong:
        with payload.open("r+b", buffering=0) as handle:
            actual = os.fstat(handle.fileno()).st_size
            if actual < expected_bytes:
                raise ValueError(
                    f"resume payload length mismatch {payload}: {actual} is shorter than "
                    f"durable checkpoint {expected_bytes}"
                )
            if actual > expected_bytes:
                handle.truncate(expected_bytes)
                handle.flush()
                os.fsync(handle.fileno())


def _open_append_handles(stage: Path, layers: int) -> list[Any]:
    handles = []
    payload_directories: set[Path] = set()
    try:
        for payload in _expected_payloads(stage, layers):
            payload.parent.mkdir(parents=True, exist_ok=True)
            handles.append(payload.open("ab", buffering=0))
            payload_directories.add(payload.parent)
        # File fsync alone does not make a newly created directory entry
        # durable.  Publish every payload entry before a progress checkpoint
        # is allowed to acknowledge bytes in it.
        for directory in sorted(payload_directories):
            _fsync_directory(directory)
        _fsync_directory(stage)
    except BaseException:
        for handle in handles:
            handle.close()
        raise
    return handles


def _close_handles(handles: Sequence[Any]) -> None:
    for handle in handles:
        handle.close()


def _durable_checkpoint(
    handles: Sequence[Any], progress_path: Path, progress: Mapping[str, Any]
) -> None:
    for handle in handles:
        handle.flush()
        os.fsync(handle.fileno())
    atomic_write_json(progress_path, dict(progress))
    # ``progress.json`` is the cross-file durability boundary.  Persist its
    # atomic directory entry before any later append or per-payload commit.
    _fsync_directory(progress_path.parent)


def _activation_provenance(
    run_definition: Mapping[str, Any], cache_type: str, layer: int
) -> dict[str, Any]:
    calibration = run_definition["calibration"]
    model = run_definition["model"]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_path": model["path"],
        "model_name": run_definition["model_name"],
        "model_role": model["role"],
        "model_config_sha256": model["config_sha256"],
        "model_weight_index_sha256": model["weight_index_sha256"],
        "model_tokenizer_sha256": model["tokenizer_sha256"],
        "token_path": calibration["token_path"],
        "token_sha256": calibration["token_sha256"],
        "calibration_manifest_sha256": calibration["manifest_sha256"],
        "dataset_fingerprint": calibration["dataset_fingerprint"],
        "split": calibration["split"],
        "stride": 4,
        "stride_start": 0,
        "forward_dtype": "bfloat16",
        "content_key_inverse_dtype": "float32",
        "rope_convention": "transformers-4.51-qwen3-split-half-default",
        "sequences": run_definition["sequences"],
        "cache_type": cache_type,
        "layer": layer,
        "capture_run_fingerprint": run_definition["run_fingerprint"],
    }


def _commit_stage(stage: Path, output: Path, run_definition: Mapping[str, Any]) -> dict[str, Any]:
    layers = int(run_definition["model"]["architecture"]["num_hidden_layers"])
    sequences = int(run_definition["sequences"])
    shape = (sequences * SAMPLED_PER_SEQUENCE, KV_HEADS, HEAD_DIM)
    entries = []
    for cache_type in CACHE_TYPES:
        for layer in range(layers):
            payload = stage / _payload_relative(cache_type, layer)
            metadata_path = sidecar_path(payload)
            if metadata_path.exists():
                metadata = read_activation_metadata(payload)
                if metadata.shape != shape:
                    raise ValueError(f"committed staging sidecar has wrong shape: {metadata_path}")
                if sha256_file(payload) != metadata.sha256:
                    raise ValueError(f"committed staging payload fails SHA256: {payload}")
            else:
                metadata = commit_existing_bf16_payload(
                    payload,
                    shape,
                    provenance=_activation_provenance(run_definition, cache_type, layer),
                )
            entries.append(
                {
                    "cache_type": cache_type,
                    "layer": layer,
                    "path": str(_payload_relative(cache_type, layer)),
                    "shape": list(metadata.shape),
                    "nbytes": metadata.nbytes,
                    "sha256": metadata.sha256,
                    "sidecar_sha256": sha256_file(metadata_path),
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_definition": dict(run_definition),
        "entries": entries,
    }
    manifest["inventory_sha256"] = _canonical_sha256(entries)
    atomic_write_json(stage / "manifest.json", manifest)
    _fsync_directory(stage)
    if output.exists():
        raise FileExistsError(f"refusing to replace existing capture output: {output}")
    os.replace(stage, output)
    _fsync_directory(output.parent)
    return _strict_json(output / "manifest.json")


def verify_capture_output(output: Path, expected_fingerprint: str | None = None) -> dict[str, Any]:
    manifest = _strict_json(output / "manifest.json")
    if manifest.get("status") != "complete" or int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"capture manifest is incomplete or unsupported: {output}")
    definition = manifest.get("run_definition")
    if not isinstance(definition, Mapping):
        raise ValueError("capture manifest lacks run_definition")
    if expected_fingerprint is not None and definition.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("capture output belongs to a different run definition")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("inventory_sha256") != _canonical_sha256(entries):
        raise ValueError("capture inventory fingerprint is invalid")
    for entry in entries:
        payload = output / str(entry["path"])
        metadata = read_activation_metadata(payload)
        if list(metadata.shape) != entry["shape"] or payload.stat().st_size != int(entry["nbytes"]):
            raise ValueError(f"capture entry shape/length mismatch: {payload}")
        if sha256_file(payload) != entry["sha256"]:
            raise ValueError(f"capture payload SHA256 mismatch: {payload}")
        if sha256_file(sidecar_path(payload)) != entry["sidecar_sha256"]:
            raise ValueError(f"capture sidecar SHA256 mismatch: {payload}")
    return manifest


def _main(argv: Sequence[str] | None = None) -> int:
    global _ACTIVE_RUN_PATH

    args = parse_args(argv)
    output = args.output.resolve()
    if args.verify_only:
        manifest = verify_capture_output(output)
        print(json.dumps({"status": "verified", "output": str(output), "inventory_sha256": manifest["inventory_sha256"]}, indent=2))
        return 0
    if args.fsync_every <= 0:
        raise ValueError("--fsync-every must be positive")
    tokens = args.tokens.resolve()
    token_ids = np.load(tokens, mmap_mode="r", allow_pickle=False)
    if token_ids.ndim != 2 or token_ids.shape[1] != 1024 or token_ids.dtype != np.int32:
        raise ValueError(f"expected int32 [sequences,1024] calibration tokens, got {token_ids.shape}/{token_ids.dtype}")
    calibration, manifest_count = _calibration_identity(tokens, args.split)
    if token_ids.shape[0] != manifest_count:
        raise ValueError("token payload row count differs from calibration manifest")
    count = token_ids.shape[0] if args.sequence_limit is None else min(args.sequence_limit, token_ids.shape[0])
    if args.sequence_limit is not None and args.sequence_limit <= 0:
        raise ValueError("--sequence-limit must be positive")
    if args.split == "train" and args.sequence_limit is None and count != 500:
        raise ValueError(f"formal train capture requires exactly 500 sequences, got {count}")
    role = "source" if args.model_name == "qwen3_14b" else "target"
    model_identity = _model_identity(args.model.resolve(), role)
    devices = [int(value) for value in args.devices.split(",") if value.strip()]
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("--devices must contain unique CUDA indices")
    if not torch.cuda.is_available():
        raise RuntimeError("formal activation capture requires CUDA")
    cuda_device_count = torch.cuda.device_count()
    if any(device < 0 or device >= cuda_device_count for device in devices):
        raise ValueError(
            f"--devices must be within [0,{cuda_device_count - 1}], got {devices}"
        )
    if transformers.__version__ != "4.51.0":
        raise RuntimeError(
            "capture was validated with transformers==4.51.0, observed "
            f"{transformers.__version__}"
        )
    definition_base = {
        "schema_version": SCHEMA_VERSION,
        "model_name": args.model_name,
        "model": model_identity,
        "calibration": calibration,
        "sequences": int(count),
        "sequence_length": 1024,
        "sample_stride": 4,
        "sample_stride_start": 0,
        "sampled_positions_per_sequence": SAMPLED_PER_SEQUENCE,
        "forward_dtype": "bfloat16",
        "inverse_rope_dtype": "float32",
        "attention_implementation": args.attention,
        "transformers_version": transformers.__version__,
    }
    run_definition = dict(definition_base)
    run_definition["run_fingerprint"] = _canonical_sha256(definition_base)
    fingerprint = str(run_definition["run_fingerprint"])
    if output.exists():
        verified = verify_capture_output(output, fingerprint)
        print(json.dumps({"status": "reused", "output": str(output), "inventory_sha256": verified["inventory_sha256"]}, indent=2))
        return 0

    run = make_run_directory(PROJECT_ROOT, "p2", f"capture_{args.model_name}_{args.split}")
    _ACTIVE_RUN_PATH = run.path
    record_run_start(
        run,
        list(sys.argv if argv is None else [sys.argv[0], *argv]),
        {
            **run_definition,
            "operational": {
                "devices": devices,
                "fsync_every_sequences": int(args.fsync_every),
                "cuda_device_count": cuda_device_count,
            },
        },
    )
    atomic_write_json(run.path / "status.json", {"status": "preparing", "output": str(output)})
    stage = _stage_for(output)
    stage.mkdir(parents=True, exist_ok=True)
    definition_path = stage / "run_definition.json"
    progress_path = stage / "progress.json"
    if definition_path.exists():
        existing = _strict_json(definition_path)
        if existing != run_definition:
            raise ValueError(f"in-progress capture belongs to a different run: {stage}")
    else:
        atomic_write_json(definition_path, run_definition)
    layers = int(model_identity["architecture"]["num_hidden_layers"])
    if progress_path.exists():
        progress = _strict_json(progress_path)
        if progress.get("run_fingerprint") != fingerprint:
            raise ValueError("capture progress fingerprint mismatch")
        completed = int(progress["completed_sequences"])
        if not 0 <= completed <= count:
            raise ValueError("capture progress sequence count is out of range")
        _validate_progress_payloads(stage, layers, completed)
    else:
        completed = 0
        # A missing progress marker means zero durable sequences, even if an
        # earlier invocation left payload bytes behind before the marker's
        # directory entry became durable.  First materialize every empty file,
        # then apply the same rollback rule used for an older checkpoint.
        handles = _open_append_handles(stage, layers)
        try:
            _validate_progress_payloads(stage, layers, completed)
            _durable_checkpoint(handles, progress_path, {"schema_version": SCHEMA_VERSION, "run_fingerprint": fingerprint, "completed_sequences": 0})
        finally:
            _close_handles(handles)

    storage = _capture_storage_preflight(stage, layers, count)
    atomic_write_json(
        run.path / "preflight.json",
        {
            "status": "passed",
            "model_name": args.model_name,
            "layers": layers,
            "sequences": count,
            "observations": count * SAMPLED_PER_SEQUENCE,
            "devices": devices,
            "storage": storage,
        },
    )

    if completed < count:
        atomic_write_json(run.path / "status.json", {"status": "loading_model", "completed_sequences": completed})
        model = load_qwen3(
            args.model,
            devices=devices,
            role=role,
            attention_implementation=args.attention,
        )
        embedding_device = model.model.embed_tokens.weight.device
        handles = _open_append_handles(stage, layers)
        started = time.monotonic()
        try:
            atomic_write_json(run.path / "status.json", {"status": "capturing", "completed_sequences": completed})
            for sequence_index in range(completed, count):
                inputs = torch.from_numpy(
                    np.asarray(token_ids[sequence_index], dtype=np.int64)
                ).unsqueeze(0).to(embedding_device)
                captured = capture_sampled_content_kv(model, inputs, stride=4)
                if len(captured) != layers:
                    raise ValueError(f"captured {len(captured)} layers, expected {layers}")
                for layer, item in enumerate(captured):
                    _write_bf16_chunk(handles[layer], item.k_content)
                    _write_bf16_chunk(handles[layers + layer], item.value)
                durable_count = sequence_index + 1
                if durable_count % args.fsync_every == 0 or durable_count == count:
                    progress = {
                        "schema_version": SCHEMA_VERSION,
                        "run_fingerprint": fingerprint,
                        "completed_sequences": durable_count,
                        "elapsed_seconds_this_invocation": time.monotonic() - started,
                    }
                    _durable_checkpoint(handles, progress_path, progress)
                    atomic_write_json(run.path / "status.json", {"status": "capturing", **progress})
                    print(json.dumps({"completed_sequences": durable_count, "total": count}), flush=True)
        finally:
            _close_handles(handles)
        _validate_progress_payloads(stage, layers, count)

    atomic_write_json(run.path / "status.json", {"status": "committing", "completed_sequences": count})
    manifest = _commit_stage(stage, output, run_definition)
    verified = verify_capture_output(output, fingerprint)
    atomic_write_json(
        run.path / "manifest.json",
        {
            "status": "complete",
            "capture_output": str(output),
            "capture_manifest_sha256": sha256_file(output / "manifest.json"),
            "inventory_sha256": verified["inventory_sha256"],
        },
    )
    atomic_write_json(run.path / "status.json", {"status": "complete"})
    if not args.no_ledger:
        append_reproduction_log(
            PROJECT_ROOT,
            heading=f"P2-CAP：{args.model_name} {args.split} KV 激活提取",
            bullets=[
                f"状态：completed；{count}×1024，stride=4，共 {count * SAMPLED_PER_SEQUENCE} observations。",
                "模型 forward 使用 BF16；K 从真实 cache 读取并用 FP32 inverse-RoPE，V 直接读取。",
                f"输入 dataset fingerprint={calibration['dataset_fingerprint']}；tokens SHA256={calibration['token_sha256']}。",
                f"输出：`{output.relative_to(PROJECT_ROOT)}`；inventory SHA256={manifest['inventory_sha256']}。",
                f"逐序列 durable checkpoint，run=`runs/{run.run_id}`。",
            ],
        )
    print(json.dumps({"status": "complete", "output": str(output), "inventory_sha256": manifest["inventory_sha256"], "run": str(run.path)}, indent=2))
    _ACTIVE_RUN_PATH = None
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run capture while making post-run-creation failures auditable."""

    try:
        return _main(argv)
    except SystemExit:
        # argparse --help and parse errors are not experiment failures.
        raise
    except BaseException as error:
        global _ACTIVE_RUN_PATH
        active_run_path = _ACTIVE_RUN_PATH
        _ACTIVE_RUN_PATH = None
        if active_run_path is not None:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            atomic_write_json(
                active_run_path / "status.json",
                {
                    "status": status,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "resume": "rerun the identical command; staging progress is authoritative",
                },
            )
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    # ``SystemExit(0)`` is the normal successful CLI path and must not be
    # reported as a failed capture.  KeyboardInterrupt likewise should retain
    # its conventional signal semantics; durable progress makes it resumable.
    except Exception as error:
        print(f"activation capture failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise
