from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .common import (
    ContractError,
    ensure_no_symlink_components,
    load_json_strict,
    require_exact_keys,
    require_int,
    sha256_file,
    validate_artifact_ref,
)


def _relative_ref(
    root: Path,
    relative_path: Any,
    expected_sha: Any,
    name: str,
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    if not isinstance(relative_path, str) or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ContractError(f"{name} has an unsafe relative path")
    target = root / relative_path
    target = ensure_no_symlink_components(target)
    if not target.is_file():
        raise ContractError(f"{name} is not a regular file")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ContractError(f"{name} has invalid expected SHA")
    actual = sha256_file(target) if verify_hash else expected_sha
    if verify_hash and actual != expected_sha:
        raise ContractError(f"{name} hash mismatch: expected {expected_sha}, got {actual}")
    return {"path": str(target), "nbytes": target.stat().st_size, "sha256": str(expected_sha)}


def validate_anchor_manifest(manifest_ref: Mapping[str, Any], *, deep: bool = False) -> dict[str, Any]:
    """Validate the frozen dir02 shrinkage manifest without loading tensors.

    `deep=True` hashes every manifest-listed payload and every regular file
    under the anchor directory (including unlisted files only for symlink
    safety). This is intentionally CPU/file I/O only.
    """

    manifest_path = validate_artifact_ref(manifest_ref, "anchor.manifest")
    root = manifest_path.parent
    document = load_json_strict(manifest_path)
    required = {
        "allow_tf32",
        "alpha",
        "architecture",
        "bias_count",
        "completed_utc",
        "direction",
        "dtype",
        "layers",
        "mapper_fingerprint",
        "method",
        "method_definition",
        "payload_validation",
        "schema_version",
        "selection",
        "split",
        "status",
        "upstream",
        "weight_count",
    }
    if set(document) != required:
        raise ContractError("anchor manifest top-level key set mismatch")
    if document["schema_version"] != 1 or document["status"] != "complete" or document["direction"] != "qwen3_32b_to_14b":
        raise ContractError("anchor manifest is not a complete Qwen3 32B->14B artifact")
    if document["dtype"] != "float32" or document["method"] != "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers":
        raise ContractError("unexpected anchor method/dtype")
    architecture = document["architecture"]
    expected_arch = {"head_dim": 128, "kv_heads": 8, "selected_source_layers": 20, "source_layers": 64, "target_layers": 40}
    if architecture != expected_arch:
        raise ContractError("anchor architecture mismatch")
    if document["payload_validation"].get("status") != "passed" or document["payload_validation"].get("layers") != 40:
        raise ContractError("anchor payload validation is not passed for 40 layers")
    layers = document["layers"]
    if not isinstance(layers, list) or len(layers) != 40:
        raise ContractError("anchor must enumerate exactly 40 layers")
    listed: set[Path] = {manifest_path}
    for index, layer in enumerate(layers):
        require_exact_keys(layer, ("bias_numel", "path", "sha256", "sidecar_path", "sidecar_sha256", "source_layers_rank_order", "target_layer", "weight_numel"), f"anchor.layers[{index}]")
        if layer["target_layer"] != index or layer["bias_numel"] != 2048 or layer["weight_numel"] != 41943040:
            raise ContractError(f"anchor layer {index} shape/count mismatch")
        order = layer["source_layers_rank_order"]
        if not isinstance(order, list) or len(order) != 20 or len(set(order)) != 20 or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 64 for value in order):
            raise ContractError(f"anchor layer {index} source order invalid")
        payload_ref = _relative_ref(
            root,
            layer["path"],
            layer["sha256"],
            f"anchor.layers[{index}].payload",
            verify_hash=deep,
        )
        sidecar_ref = _relative_ref(root, layer["sidecar_path"], layer["sidecar_sha256"], f"anchor.layers[{index}].sidecar")
        sidecar = load_json_strict(Path(sidecar_ref["path"]))
        if not isinstance(sidecar, Mapping) or sidecar.get("sha256") != payload_ref["sha256"] or sidecar.get("target_layer") != index:
            raise ContractError(f"anchor layer {index} sidecar does not bind payload")
        listed.update((Path(payload_ref["path"]), Path(sidecar_ref["path"])))

    for name in ("alpha", "split"):
        item = document[name]
        if not isinstance(item, Mapping) or set(item) != {"nbytes", "path", "sha256"}:
            raise ContractError(f"anchor {name} reference malformed")
        _relative_ref(root, item["path"], item["sha256"], f"anchor.{name}", verify_hash=deep)
        listed.add(root / item["path"])

    if deep:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ContractError(f"symlink under anchor directory: {path}")
            if path.is_file():
                ensure_no_symlink_components(path)
        # Ensure no listed file silently changed between per-layer checks and
        # inventory traversal.
        for path in listed:
            if not path.is_file():
                raise ContractError(f"listed anchor file disappeared: {path}")
            ensure_no_symlink_components(path)
    return {
        "manifest": dict(manifest_ref),
        "manifest_sha256": sha256_file(manifest_path),
        "layers": len(layers),
        "listed_payload_files": len(listed),
        "deep_hash_performed": bool(deep),
        "mapper_fingerprint": document["mapper_fingerprint"],
    }
