"""Small, fail-closed provenance helpers for reverse-direction evaluations.

This module intentionally owns only evaluation artefacts.  It does not alter
the mapper fit, selection, held-out validation, or runtime implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from .protocol import (
    EXPERIMENT_ROOT,
    canonical_sha256,
    atomic_write_json,
    read_json,
    sha256_file,
)


EVALUATION_SCHEMA_VERSION = 1


def assert_evaluation_output(path: str | Path) -> Path:
    """Resolve an output below the isolated experiment and create it."""

    resolved = Path(path).resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("evaluation output must stay inside the reverse experiment") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def hashed_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically write JSON and an independently verifiable SHA sidecar."""

    destination = Path(path)
    atomic_write_json(destination, value)
    sidecar = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact": destination.name,
        "sha256": sha256_file(destination),
        "nbytes": destination.stat().st_size,
    }
    atomic_write_json(destination.with_name(f"{destination.name}.sha256.json"), sidecar)
    return sidecar


def verify_hashed_json(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    sidecar = read_json(destination.with_name(f"{destination.name}.sha256.json"))
    expected = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact": destination.name,
        "sha256": sha256_file(destination),
        "nbytes": destination.stat().st_size,
    }
    if sidecar != expected:
        raise ValueError(f"JSON SHA sidecar mismatch: {destination}")
    return sidecar


def immutable_hashed_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        if read_json(destination) != dict(value):
            raise ValueError(f"immutable evaluation artifact changed: {destination}")
        return verify_hashed_json(destination)
    return hashed_json(destination, value)


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(
            dict(row),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in rows
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def hashed_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_jsonl_bytes(rows)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    sidecar = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact": destination.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "nbytes": len(payload),
        "rows": payload.count(b"\n"),
    }
    atomic_write_json(destination.with_name(f"{destination.name}.sha256.json"), sidecar)
    return sidecar


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in token_ids:
        integer = int(value)
        if integer < 0:
            raise ValueError("token IDs must be non-negative")
        digest.update(integer.to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def model_artifact_identity(model_root: str | Path) -> dict[str, Any]:
    """Content-bind every model metadata file and safetensors weight shard.

    Names and byte sizes are not a model identity: a shard can be changed in
    place without either field moving.  Formal reverse evaluations therefore
    pay the one-time sequential read cost and put every shard SHA256 into the
    request fingerprint before model loading or checkpoint reuse.
    """

    root = Path(model_root).resolve()
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    files: dict[str, dict[str, Any]] = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    for name in ("tokenizer_config.json", "generation_config.json"):
        path = root / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    index = read_json(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"invalid model index: {root}")
    shards: list[dict[str, Any]] = []
    for name in sorted({str(value) for value in weight_map.values()}):
        path = (root / name).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError as exc:
            raise ValueError("model shard escapes checkpoint directory") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        shards.append(
            {
                "path": relative,
                "nbytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "path": str(root),
        "content_hashed_files": files,
        "weight_shards": shards,
        "weight_shard_count": len(shards),
        "weight_shard_total_nbytes": sum(row["nbytes"] for row in shards),
        "weight_shards_sha256": canonical_sha256(shards),
        "weight_identity_scope": "index_sha256_plus_every_weight_shard_content_sha256",
        "weight_shard_contents_hashed": True,
    }


def mapper_manifest_identity(mapper_manifest: str | Path) -> dict[str, Any]:
    manifest = Path(mapper_manifest).resolve()
    root = manifest if manifest.is_dir() else manifest.parent
    manifest = root / "manifest.json"
    checksum = root / "manifest.sha256.json"
    if not manifest.is_file() or not checksum.is_file():
        raise FileNotFoundError("mapper manifest and commit checksum are required")
    digest = sha256_file(manifest)
    checksum_value = read_json(checksum)
    if checksum_value.get("sha256") != digest:
        raise ValueError("mapper manifest disagrees with its commit checksum")
    value = read_json(manifest)
    if value.get("status") != "complete" or value.get("direction") != "qwen3_32b_to_14b":
        raise ValueError("evaluation requires a complete reverse mapper")
    return {
        "root": str(root),
        "manifest": str(manifest),
        "manifest_sha256": digest,
        "checksum": str(checksum),
        "checksum_sha256": sha256_file(checksum),
        "fit_fingerprint": value.get("fit_fingerprint"),
        "target_layer_count": len(value.get("layers", [])),
    }


def code_identity(
    relative_paths: Sequence[str | Path], *, root: str | Path = EXPERIMENT_ROOT
) -> dict[str, Any]:
    identity_root = Path(root).resolve()
    files: dict[str, dict[str, Any]] = {}
    for raw in sorted({str(Path(value)) for value in relative_paths}):
        path = (identity_root / raw).resolve()
        try:
            path.relative_to(identity_root)
        except ValueError as exc:
            raise ValueError(f"code identity path escapes its declared root: {raw}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        files[raw] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    return {
        "root": str(identity_root),
        "files": files,
        "files_sha256": canonical_sha256(files),
    }


def command_evidence(command: Sequence[str], *, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command), capture_output=True, text=True, check=False, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(command), "error": str(exc)}
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


__all__ = [
    "EVALUATION_SCHEMA_VERSION",
    "assert_evaluation_output",
    "canonical_jsonl_bytes",
    "code_identity",
    "command_evidence",
    "hashed_json",
    "hashed_jsonl",
    "immutable_hashed_json",
    "mapper_manifest_identity",
    "model_artifact_identity",
    "token_ids_sha256",
    "verify_hashed_json",
]
