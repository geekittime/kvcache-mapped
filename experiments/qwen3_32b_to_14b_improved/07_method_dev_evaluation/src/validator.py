"""CPU-only validator for a completed paired method-development run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ArtifactRef,
    ContractError,
    MODES,
    canonical_sha256,
    code_inventory,
    load_config,
    read_json,
    require_child,
    sha256_file,
)
from .runner import (
    build_run_definition,
    load_prepared,
    result_path,
    validate_checkpoint,
)
from .runtime_adapter import validate_method_bindings
from .statistics import aggregate_paired_results


def _verify_sidecar(path: Path) -> None:
    sidecar_path = path.with_name(path.name + ".sha256.json")
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise ContractError(f"SHA sidecar is missing: {sidecar_path}")
    expected = {
        "schema_version": 1,
        "path": path.name,
        "nbytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if read_json(sidecar_path) != expected:
        raise ContractError(f"SHA sidecar mismatch: {path}")


def _validated_confirmation_prerequisite(
    config_path: str | Path,
    config: Mapping[str, Any],
    frozen_definition: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Revalidate the pre-registered screen gate embedded in a confirm definition.

    A confirm run definition is deliberately tied to the exact, independently
    validated screen completion receipt that authorized it.  Recompute that
    validation rather than trusting serialized validation output, then return
    the original canonical payload for run-definition reproduction.
    """

    profile = frozen_definition.get("profile")
    has_prerequisite = "confirmation_prerequisite" in frozen_definition
    if profile != "confirm":
        if has_prerequisite:
            raise ContractError("non-confirm run definition contains a confirmation prerequisite")
        return None
    if not has_prerequisite:
        raise ContractError("confirm run definition lacks a confirmation prerequisite")

    prerequisite = frozen_definition["confirmation_prerequisite"]
    if not isinstance(prerequisite, Mapping) or set(prerequisite) != {
        "rule_id",
        "validated_screen",
        "screen_completion_receipt",
    }:
        raise ContractError("confirm confirmation prerequisite schema changed")
    gate = config["confirmation_gate"]
    if prerequisite.get("rule_id") != gate["rule_id"]:
        raise ContractError("confirm confirmation prerequisite rule changed")
    serialized_validation = prerequisite.get("validated_screen")
    if not isinstance(serialized_validation, Mapping):
        raise ContractError("confirm confirmation prerequisite lacks screen validation evidence")

    receipt_ref = ArtifactRef.parse(
        prerequisite.get("screen_completion_receipt"),
        label="confirm screen completion receipt",
    )
    receipt_path = require_child(
        config["outputs"]["runs_root"], receipt_ref.path, label="confirm screen completion receipt"
    )
    if receipt_path.name != "completion_receipt.json":
        raise ContractError("confirm prerequisite does not bind a screen completion receipt")
    receipt_ref.verify(label="confirm screen completion receipt")
    _verify_sidecar(receipt_path)

    observed_validation = validate_completed_run(
        config_path,
        profile=gate["required_screen_profile"],
        output_dir=receipt_path.parent,
    )
    if observed_validation.get("status") != gate["required_screen_validation_status"]:
        raise ContractError("confirm prerequisite screen validation status changed")
    if observed_validation.get("completion_receipt_sha256") != receipt_ref.sha256:
        raise ContractError("confirm prerequisite screen receipt hash changed")
    if dict(serialized_validation) != observed_validation:
        raise ContractError("confirm prerequisite validated screen evidence changed")
    return dict(prerequisite)


def validate_completed_run(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = load_config(config_path, verify_inputs=True)
    output = require_child(config["outputs"]["runs_root"], output_dir, label="run output")
    rows, _prepared, prepared_ref = load_prepared(config, profile)
    bindings = validate_method_bindings(config, deep_model_hash=False)
    definition_path = output / "run_definition.json"
    _verify_sidecar(definition_path)
    frozen_definition = read_json(definition_path)
    if not isinstance(frozen_definition, Mapping):
        raise ContractError("run definition must be an object")
    confirmation_prerequisite = _validated_confirmation_prerequisite(
        config_path, config, frozen_definition
    )
    expected_definition = build_run_definition(
        config,
        profile,
        rows,
        prepared_ref,
        bindings,
        confirmation_prerequisite=confirmation_prerequisite,
    )
    if frozen_definition != expected_definition:
        raise ContractError("run definition does not reproduce from current frozen inputs")
    fingerprint = expected_definition["request_fingerprint"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["runtime"]["target_model"], local_files_only=True, use_fast=True
    )
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for row in rows:
        target_path = result_path(output, row, "native14")
        baseline_path = result_path(output, row, "nvidia_k20")
        if not target_path.is_file() or not baseline_path.is_file():
            raise ContractError("completed run is missing target or baseline checkpoint")
        target_hash = sha256_file(target_path)
        baseline_hash = sha256_file(baseline_path)
        for mode in MODES:
            path = result_path(output, row, mode)
            expected_paths.add(path.resolve())
            value = validate_checkpoint(
                path,
                row=row,
                mode=mode,
                request_fingerprint=fingerprint,
                method_binding=expected_definition["method_binding_sha256"][mode],
                tokenizer=tokenizer,
                expected_target_sha256=target_hash if mode != "native14" else None,
                expected_baseline_sha256=(
                    baseline_hash
                    if mode in {"shrinkage", "tail_refresh_r8", "tail_refresh_r32"}
                    else None
                ),
            )
            records.append(value)
            inventory.append(
                ArtifactRef(
                    path=str(path), nbytes=path.stat().st_size, sha256=sha256_file(path)
                ).to_dict()
            )
    observed_paths = {
        path.resolve()
        for path in (output / "results").rglob("*.json")
        if path.is_file() and not path.name.endswith(".failure.json")
    }
    failures = [path for path in (output / "results").rglob("*.failure.json") if path.is_file()]
    if observed_paths != expected_paths or failures:
        raise ContractError("result tree has missing, unexpected, or failure checkpoints")

    expected_summary = aggregate_paired_results(
        records,
        profile=profile,
        request_fingerprint=fingerprint,
        bootstrap_seed=int(config["metrics"]["bootstrap_seed"]),
        bootstrap_replicates=int(config["metrics"]["bootstrap_replicates"]),
        confidence_level=float(config["metrics"]["confidence_level"]),
    )
    summary_path = output / "summary.json"
    _verify_sidecar(summary_path)
    if read_json(summary_path) != expected_summary:
        raise ContractError("paired summary does not recompute from atomic results")
    receipt_path = output / "completion_receipt.json"
    _verify_sidecar(receipt_path)
    receipt = read_json(receipt_path)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "profile": profile,
        "result_count": len(inventory),
        "expected_result_count": len(rows) * len(MODES),
        "result_inventory": inventory,
        "result_inventory_sha256": canonical_sha256(inventory),
        "runtime_binding_digest": bindings.digest,
        "paired_complete": True,
        "official_dev_opened": False,
        "claim_boundary": config["claim_boundary"],
    }
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise ContractError(f"completion receipt {field} changed")
    if receipt.get("run_definition") != ArtifactRef(
        path=str(definition_path),
        nbytes=definition_path.stat().st_size,
        sha256=sha256_file(definition_path),
    ).to_dict():
        raise ContractError("completion receipt run-definition binding changed")
    if receipt.get("summary") != ArtifactRef(
        path=str(summary_path), nbytes=summary_path.stat().st_size, sha256=sha256_file(summary_path)
    ).to_dict():
        raise ContractError("completion receipt summary binding changed")
    gate = receipt.get("gpu_idle_gate")
    gate_ref = ArtifactRef.parse(gate, label="GPU idle gate")
    if Path(gate_ref.path).resolve() != (output / "gpu_idle_gate.json").resolve():
        raise ContractError("completion receipt GPU gate path changed")
    gate_ref.verify(label="GPU idle gate")
    _verify_sidecar(Path(gate_ref.path))
    status = read_json(output / "status.json")
    if (
        status.get("status") != "complete"
        or status.get("request_fingerprint") != fingerprint
        or status.get("completion_receipt_sha256") != sha256_file(receipt_path)
        or status.get("completed_atomic_results") != len(inventory)
    ):
        raise ContractError("final status is not bound to the completion receipt")
    return {
        "schema_version": 1,
        "status": "validated_complete",
        "gpu_used_by_validator": False,
        "profile": profile,
        "request_fingerprint": fingerprint,
        "result_count": len(inventory),
        "completion_receipt_sha256": sha256_file(receipt_path),
        "code_inventory_digest": code_inventory()["digest"],
        "official_dev_opened": False,
    }


__all__ = ["validate_completed_run"]
