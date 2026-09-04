from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from .common import (
    ContractError,
    artifact_ref,
    atomic_write_json,
    canonical_json_bytes,
    ensure_no_symlink_components,
    sha256_file,
    load_json_strict,
    require_exact_keys,
    require_int,
    validate_artifact_ref,
)


_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _safe_tensor_map(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not tensors:
        raise ContractError("checkpoint must contain at least one tensor")
    output: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or "/" in name or ".." in name:
            raise ContractError(f"invalid tensor name: {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise ContractError(f"checkpoint tensor {name!r} is not a torch Tensor")
        if tensor.is_cuda:
            raise ContractError("checkpoint serialization requires CPU tensors")
        if not bool(torch.isfinite(tensor).all()):
            raise ContractError(f"checkpoint tensor {name!r} contains NaN/Inf")
        output[name] = tensor.detach().contiguous().cpu()
    return output


def save_checkpoint(
    root: Path,
    checkpoint_id: str,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically publish a safe-tensors checkpoint and immutable receipt."""

    if not isinstance(checkpoint_id, str) or not _CHECKPOINT_ID.fullmatch(checkpoint_id):
        raise ContractError("invalid checkpoint_id")
    root = ensure_no_symlink_components(root)
    if not root.is_dir():
        raise ContractError("checkpoint root must be an existing directory")
    final = root / checkpoint_id
    if final.exists() or final.is_symlink():
        raise ContractError(f"refusing to overwrite checkpoint: {final}")
    safe_tensors = _safe_tensor_map(tensors)
    if not isinstance(metadata, Mapping):
        raise ContractError("checkpoint metadata must be an object")
    # Round-trip through canonical JSON to reject non-serializable values and
    # make the exact metadata part of the checkpoint hash boundary.
    metadata_copy = json.loads(canonical_json_bytes(dict(metadata)).decode("utf-8"))
    staging = root / f".{checkpoint_id}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    staging.mkdir(mode=0o700)
    try:
        from safetensors.torch import save_file

        save_file(safe_tensors, str(staging / "weights.safetensors"), metadata={"format": "pt"})
        atomic_write_json(staging / "metadata.json", metadata_copy)
        # The manifest is written before the staging directory is renamed.
        # References therefore point to their eventual immutable location,
        # never to a temporary path that would become dangling after commit.
        def future_ref(staged: Path, final_path: Path) -> dict[str, Any]:
            return {"path": str(final_path), "nbytes": staged.stat().st_size, "sha256": sha256_file(staged)}

        weights = future_ref(staging / "weights.safetensors", final / "weights.safetensors")
        metadata_ref = future_ref(staging / "metadata.json", final / "metadata.json")
        manifest = {
            "schema_version": 1,
            "kind": "functional_residual_checkpoint_manifest_v1",
            "checkpoint_id": checkpoint_id,
            "tensor_names": sorted(safe_tensors),
            "tensor_count": len(safe_tensors),
            "parameter_count": int(sum(tensor.numel() for tensor in safe_tensors.values())),
            "files": {"weights": weights, "metadata": metadata_ref},
        }
        atomic_write_json(staging / "manifest.json", manifest)
        manifest_ref = future_ref(staging / "manifest.json", final / "manifest.json")
        committed = {
            "schema_version": 1,
            "kind": "functional_residual_checkpoint_commit_v1",
            "status": "COMMITTED",
            "checkpoint_id": checkpoint_id,
            "manifest": manifest_ref,
            "weights": weights,
            "metadata": metadata_ref,
        }
        atomic_write_json(staging / "COMMITTED.json", committed)
        for child in staging.iterdir():
            os.chmod(child, 0o444)
        os.chmod(staging, 0o555)
        os.replace(staging, final)
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return final


def validate_checkpoint(path: Path) -> dict[str, Any]:
    path = ensure_no_symlink_components(path)
    if not path.is_dir():
        raise ContractError("checkpoint path is not a directory")
    children = {child.name for child in path.iterdir()}
    expected = {"weights.safetensors", "metadata.json", "manifest.json", "COMMITTED.json"}
    if children != expected:
        raise ContractError(f"checkpoint file set mismatch: {sorted(children)}")
    commit = load_json_strict(path / "COMMITTED.json")
    require_exact_keys(commit, ("schema_version", "kind", "status", "checkpoint_id", "manifest", "weights", "metadata"), "checkpoint commit")
    if commit["schema_version"] != 1 or commit["kind"] != "functional_residual_checkpoint_commit_v1" or commit["status"] != "COMMITTED":
        raise ContractError("invalid checkpoint commit marker")
    if not isinstance(commit["checkpoint_id"], str) or path.name != commit["checkpoint_id"]:
        raise ContractError("checkpoint id mismatch")
    for name in ("manifest", "weights", "metadata"):
        validate_artifact_ref(commit[name], f"checkpoint commit.{name}")
    manifest = load_json_strict(path / "manifest.json")
    require_exact_keys(manifest, ("schema_version", "kind", "checkpoint_id", "tensor_names", "tensor_count", "parameter_count", "files"), "checkpoint manifest")
    if manifest["schema_version"] != 1 or manifest["kind"] != "functional_residual_checkpoint_manifest_v1" or manifest["checkpoint_id"] != path.name:
        raise ContractError("invalid checkpoint manifest")
    if not isinstance(manifest["tensor_names"], list) or manifest["tensor_names"] != sorted(manifest["tensor_names"]):
        raise ContractError("tensor_names must be sorted list")
    require_int(manifest["tensor_count"], "manifest.tensor_count", minimum=1)
    require_int(manifest["parameter_count"], "manifest.parameter_count", minimum=1)
    require_exact_keys(manifest["files"], ("weights", "metadata"), "checkpoint manifest.files")
    for name in ("weights", "metadata"):
        validate_artifact_ref(manifest["files"][name], f"checkpoint manifest.files.{name}")
    if commit["manifest"] != artifact_ref(path / "manifest.json"):
        raise ContractError("commit manifest reference mismatch")
    if commit["weights"] != artifact_ref(path / "weights.safetensors"):
        raise ContractError("commit weights reference mismatch")
    if commit["metadata"] != artifact_ref(path / "metadata.json"):
        raise ContractError("commit metadata reference mismatch")
    metadata = load_json_strict(path / "metadata.json")
    if not isinstance(metadata, Mapping):
        raise ContractError("checkpoint metadata must be an object")
    from safetensors.torch import load_file

    tensors = load_file(str(path / "weights.safetensors"), device="cpu")
    if sorted(tensors) != manifest["tensor_names"] or len(tensors) != manifest["tensor_count"]:
        raise ContractError("checkpoint tensor inventory mismatch")
    parameter_count = sum(int(tensor.numel()) for tensor in tensors.values())
    if parameter_count != manifest["parameter_count"]:
        raise ContractError("checkpoint parameter count mismatch")
    if any(not bool(torch.isfinite(tensor).all()) for tensor in tensors.values()):
        raise ContractError("checkpoint tensor became non-finite")
    return {
        "checkpoint_id": path.name,
        "tensor_count": len(tensors),
        "parameter_count": parameter_count,
        "manifest_sha256": sha256_file(path / "manifest.json"),
        "commit_sha256": sha256_file(path / "COMMITTED.json"),
    }


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    validation = validate_checkpoint(path)
    from safetensors.torch import load_file

    tensors = load_file(str(path / "weights.safetensors"), device="cpu")
    metadata = load_json_strict(path / "metadata.json")
    commit = load_json_strict(path / "COMMITTED.json")
    return tensors, metadata, {"validation": validation, "commit": commit}
