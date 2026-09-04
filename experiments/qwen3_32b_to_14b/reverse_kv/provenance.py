"""Fail-closed mapper/capture/model identity binding for the reverse runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import (
    DIRECTION,
    SOURCE,
    TARGET,
    canonical_sha256,
    fit_activation_inventory,
    read_json,
    sha256_file,
)


IDENTITY_FILES = {
    "model_config_sha256": "config.json",
    "model_tokenizer_sha256": "tokenizer.json",
    "model_weight_index_sha256": "model.safetensors.index.json",
}


def local_model_identity(model_root: str | Path) -> dict[str, str]:
    root = Path(model_root).resolve()
    result = {"model_path": str(root)}
    for field, filename in IDENTITY_FILES.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"model identity file is missing: {path}")
        result[field] = sha256_file(path)
    return result


def _identity_from_entries(
    entries: list[dict[str, Any]], *, semantic_role: str
) -> dict[str, str]:
    values: dict[str, set[str]] = {
        field: set()
        for field in (
            "model_name",
            "model_path",
            "model_role",
            "model_config_sha256",
            "model_tokenizer_sha256",
            "model_weight_index_sha256",
            "capture_run_fingerprint",
        )
    }
    role_entries = [entry for entry in entries if entry.get("role") == semantic_role]
    expected_count = 128 if semantic_role == "source" else 80
    if len(role_entries) != expected_count:
        raise ValueError(
            f"mapper inventory {semantic_role} coverage is {len(role_entries)}, expected {expected_count}"
        )
    for entry in role_entries:
        provenance = entry.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"mapper inventory {semantic_role} entry lacks provenance")
        for field in values:
            value = provenance.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"mapper inventory {semantic_role} provenance lacks {field}"
                )
            values[field].add(value)
    inconsistent = {field: sorted(items) for field, items in values.items() if len(items) != 1}
    if inconsistent:
        raise ValueError(
            f"mapper inventory {semantic_role} model identity is inconsistent: {inconsistent}"
        )
    return {field: next(iter(items)) for field, items in values.items()}


def load_verified_train_inventory(
    mapper_root: str | Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    """Verify the local mapper input snapshot and bind it to current captures/models."""

    root = Path(mapper_root).resolve()
    reference = manifest.get("activation_inventory")
    if not isinstance(reference, Mapping):
        raise ValueError("mapper manifest lacks activation inventory provenance")
    expected_path = (root / "inputs/activation_inventory.json").resolve()
    recorded_path = Path(str(reference.get("path", ""))).resolve()
    if recorded_path != expected_path or not expected_path.is_file():
        raise ValueError("mapper activation inventory is not the self-contained input snapshot")
    observed_sha = sha256_file(expected_path)
    if reference.get("sha256") != observed_sha:
        raise ValueError("mapper activation inventory SHA256 mismatch")
    if reference.get("payload_checksums_verified") is not True:
        raise ValueError("mapper was not fitted after full activation payload SHA256 verification")
    inventory = read_json(expected_path)
    recorded_digest = inventory.get("inventory_digest")
    digest_value = dict(inventory)
    digest_value.pop("inventory_digest", None)
    if recorded_digest != canonical_sha256(digest_value):
        raise ValueError("mapper activation inventory digest is invalid")
    if reference.get("inventory_digest") != recorded_digest:
        raise ValueError("mapper manifest and activation inventory digest disagree")
    exact = {
        "schema_version": 1,
        "direction": DIRECTION,
        "semantic_capture_role_binding": {"source": "target", "target": "source"},
        "payload_checksums_verified": True,
        "expected_shape": [128_000, 8, 128],
    }
    for field, expected in exact.items():
        if inventory.get(field) != expected:
            raise ValueError(f"mapper activation inventory {field} mismatch")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or len(entries) != 208:
        raise ValueError("mapper activation inventory must contain 208 source/target K/V entries")

    # Re-read all capture manifests and sidecars now.  The training fit already
    # hashed every payload; this pass proves the mapper snapshot still names the
    # exact current capture inventory without redundantly rereading ~57 GiB.
    current = fit_activation_inventory(verify_payloads=False)
    for field in (
        "direction",
        "semantic_capture_role_binding",
        "expected_shape",
        "shared_provenance",
        "entries",
    ):
        if inventory.get(field) != current.get(field):
            raise ValueError(f"mapper inventory no longer matches current captures: {field}")

    source_identity = _identity_from_entries(entries, semantic_role="source")
    target_identity = _identity_from_entries(entries, semantic_role="target")
    expected_identities = (
        (
            source_identity,
            SOURCE.model_name,
            str(SOURCE.model_path.resolve()),
            SOURCE.capture_native_role,
        ),
        (
            target_identity,
            TARGET.model_name,
            str(TARGET.model_path.resolve()),
            TARGET.capture_native_role,
        ),
    )
    for identity, model_name, model_path, native_role in expected_identities:
        if identity["model_name"] != model_name:
            raise ValueError("mapper capture model name does not match semantic endpoint")
        if Path(identity["model_path"]).resolve() != Path(model_path):
            raise ValueError("mapper capture model path does not match semantic endpoint")
        if identity["model_role"] != native_role:
            raise ValueError("mapper capture-native model role is invalid")
        current_model = local_model_identity(model_path)
        for field, value in current_model.items():
            if identity.get(field) != value:
                raise ValueError(
                    f"current model artifact differs from mapper training capture: {field}"
                )
    return inventory, source_identity, target_identity


def assert_model_instance_identity(
    model: Any, expected: Mapping[str, str], *, semantic_role: str
) -> dict[str, str]:
    """Bind an actually loaded HF model instance to its capture artifact identity."""

    candidate = getattr(model, "name_or_path", None)
    if not candidate:
        candidate = getattr(getattr(model, "config", None), "_name_or_path", None)
    if not candidate:
        raise ValueError(f"loaded {semantic_role} model does not expose name_or_path")
    observed = local_model_identity(candidate)
    for field, value in observed.items():
        if expected.get(field) != value:
            raise ValueError(
                f"loaded {semantic_role} model differs from mapper training capture: {field}"
            )
    return observed


__all__ = [
    "assert_model_instance_identity",
    "load_verified_train_inventory",
    "local_model_identity",
]
