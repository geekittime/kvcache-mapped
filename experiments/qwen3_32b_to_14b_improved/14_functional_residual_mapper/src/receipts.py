from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import ContractError, artifact_ref, atomic_write_json, load_json_strict, require_exact_keys
from .checkpoint import validate_checkpoint


def build_training_receipt(
    path: Path,
    *,
    config_ref: Mapping[str, Any],
    code_inventory: Mapping[str, Any],
    data_refs: Mapping[str, Any],
    anchor_ref: Mapping[str, Any],
    oracle_ref: Mapping[str, Any],
    capture_ref: Mapping[str, Any],
    checkpoint_path: Path,
    selected: Mapping[str, Any],
    metrics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    system_class: str = "functional_residual_r1",
    locked_outcomes_opened: bool = False,
) -> Path:
    """Publish an immutable method-dev receipt; no locked claim is inferred."""

    if system_class not in {"functional_residual_r1", "functional_residual_r32"}:
        raise ContractError("unknown system class")
    if locked_outcomes_opened is not False:
        raise ContractError("pre-confirmation receipt cannot open locked outcomes")
    checkpoint_path = Path(checkpoint_path)
    validate_checkpoint(checkpoint_path)
    payload = {
        "schema_version": 1,
        "kind": "functional_residual_training_receipt_v1",
        "status": "METHOD_DEV_COMPLETE",
        "claim": "candidate_method_dev_only",
        "locked_outcomes_opened": False,
        "system_class": system_class,
        "tail_refresh": system_class == "functional_residual_r32",
        "config": dict(config_ref),
        "code_inventory": dict(code_inventory),
        "data_refs": dict(data_refs),
        "anchor": dict(anchor_ref),
        "oracle": dict(oracle_ref),
        "capture": dict(capture_ref),
        "checkpoint": artifact_ref(checkpoint_path / "COMMITTED.json"),
        "selected": dict(selected),
        "metrics": dict(metrics),
        "runtime": dict(runtime),
        "unfinished_boundaries": [
            "prospective_R1_locked_test_not_run",
            "optional_R32_not_run",
            "GPU_memory_and_latency_not_measured",
        ],
    }
    atomic_write_json(path, payload)
    return path


def validate_training_receipt(path: Path) -> dict[str, Any]:
    receipt = load_json_strict(path)
    required = (
        "schema_version",
        "kind",
        "status",
        "claim",
        "locked_outcomes_opened",
        "system_class",
        "tail_refresh",
        "config",
        "code_inventory",
        "data_refs",
        "anchor",
        "oracle",
        "capture",
        "checkpoint",
        "selected",
        "metrics",
        "runtime",
        "unfinished_boundaries",
    )
    require_exact_keys(receipt, required, "training receipt")
    if receipt["schema_version"] != 1 or receipt["kind"] != "functional_residual_training_receipt_v1":
        raise ContractError("unsupported training receipt schema")
    if receipt["status"] != "METHOD_DEV_COMPLETE" or receipt["claim"] != "candidate_method_dev_only":
        raise ContractError("invalid training receipt status/claim")
    if receipt["locked_outcomes_opened"] is not False:
        raise ContractError("training receipt opened locked outcomes")
    if receipt["system_class"] not in {"functional_residual_r1", "functional_residual_r32"}:
        raise ContractError("invalid training receipt system class")
    if receipt["tail_refresh"] is not (receipt["system_class"] == "functional_residual_r32"):
        raise ContractError("tail_refresh/system class mismatch")
    if not isinstance(receipt["unfinished_boundaries"], list) or not receipt["unfinished_boundaries"]:
        raise ContractError("receipt must disclose unfinished GPU/confirmation boundaries")
    validate_checkpoint(Path(receipt["checkpoint"]["path"]).parent)
    return receipt

