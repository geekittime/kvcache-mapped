"""CPU-only, from-scratch validator for sealed prospective v2 runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .configuration import load_config
from .contracts import (
    ALL_MODES,
    ContractError,
    artifact_ref,
    canonical_sha256,
    code_inventory,
    immutable_json,
    read_json,
    verify_sidecar,
    write_sidecar,
)
from .promotion import evaluate_confirm_evidence, evaluate_screen_promotion
from .run_protocol import (
    canonical_run_output,
    collect_and_validate_results,
    config_ref,
    reproduce_run_definition,
)
from .statistics import aggregate_paired_results


def validate_completed_run(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, verify_inputs=True)
    output = canonical_run_output(config, profile, output_dir)
    definition, rows = reproduce_run_definition(config, profile, output)
    from .gpu_runner import validate_raw_completion_receipt

    if tokenizer is None:
        from .gpu_runner import load_frozen_decode_tokenizer

        tokenizer = load_frozen_decode_tokenizer(config)
    validate_raw_completion_receipt(
        config, definition, rows, output, tokenizer=tokenizer
    )
    raw_completion_ref = artifact_ref(output / "raw_completion_receipt.json")
    records, inventory = collect_and_validate_results(
        definition, rows, output, tokenizer=tokenizer, write_derived=False
    )
    bootstrap = config["metrics"]["bootstrap"]
    expected_summary = aggregate_paired_results(
        records,
        profile=profile,
        expected_dataset_counts=definition["profile_dataset_counts"],
        request_fingerprint=definition["request_fingerprint"],
        seed=bootstrap["seed"],
        replicates=bootstrap["replicates"],
        confidence_level=bootstrap["confidence_level"],
    )
    summary_path = output / "summary.json"
    verify_sidecar(summary_path)
    if read_json(summary_path) != expected_summary:
        raise ContractError("summary does not recompute from atomic paired results")
    if profile == "screen":
        expected_decision = evaluate_screen_promotion(expected_summary)
    elif profile == "confirm":
        expected_decision = evaluate_confirm_evidence(expected_summary)
    else:
        expected_decision = {
            "schema_version": 2,
            "profile": "smoke",
            "status": "not_evaluated",
            "reason": "smoke checks execution integrity only and cannot promote",
            "authorizes_confirm": False,
            "authorizes_locked_test_or_paper_superiority_claim": False,
        }
    decision_path = output / "decision.json"
    verify_sidecar(decision_path)
    if read_json(decision_path) != expected_decision:
        raise ContractError("decision/promotion gate does not recompute")

    definition_ref = artifact_ref(output / "run_definition.json")
    summary_ref = artifact_ref(summary_path)
    decision_ref = artifact_ref(decision_path)
    expected_receipt = {
        "schema_version": 2,
        "status": "complete",
        "profile": profile,
        "request_fingerprint": definition["request_fingerprint"],
        "config": config_ref(config),
        "run_definition": definition_ref.to_dict(),
        "raw_completion_receipt": raw_completion_ref.to_dict(),
        "summary": summary_ref.to_dict(),
        "decision": decision_ref.to_dict(),
        "result_count": len(inventory),
        "expected_result_count": len(rows) * len(ALL_MODES),
        "result_inventory": inventory,
        "result_inventory_sha256": canonical_sha256(inventory),
        "raw_grid_complete": True,
        "derived_grid_complete": True,
        "paired_complete": True,
        "promotion_status": expected_decision.get("status"),
        "authorizes_confirm": expected_decision.get("authorizes_confirm", False),
        "complete_g0_v2_receipt": config["g0"]["completion_receipt"],
        "registry_binding": definition["registry_binding"],
        "official_dev_or_test_opened": False,
        "claim_boundary": config["claim_boundary"],
    }
    receipt_path = output / "completion_receipt.json"
    verify_sidecar(receipt_path)
    if read_json(receipt_path) != expected_receipt:
        raise ContractError("completion receipt does not reproduce")
    status_path = output / "status.json"
    verify_sidecar(status_path)
    expected_status = {
        "schema_version": 2,
        "status": "complete",
        "profile": profile,
        "request_fingerprint": definition["request_fingerprint"],
        "completed_results": len(inventory),
        "completion_receipt_sha256": artifact_ref(receipt_path).sha256,
    }
    if read_json(status_path) != expected_status:
        raise ContractError("run status is not bound to completion receipt")
    return {
        "schema_version": 2,
        "status": "validated_complete",
        "validator_used_gpu": False,
        "profile": profile,
        "request_fingerprint": definition["request_fingerprint"],
        "config": config_ref(config),
        "completion_receipt": artifact_ref(receipt_path).to_dict(),
        "decision": decision_ref.to_dict(),
        "promotion_status": expected_decision.get("status"),
        "authorizes_confirm": expected_decision.get("authorizes_confirm", False),
        "complete_g0_v2_receipt": config["g0"]["completion_receipt"],
        "registry_binding": definition["registry_binding"],
        "result_count": len(inventory),
        "code_inventory_digest": code_inventory()["digest"],
        "official_dev_or_test_opened": False,
        "claim_boundary": config["claim_boundary"],
    }


def write_validation_receipt(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    value = validate_completed_run(
        config_path, profile=profile, output_dir=output_dir, tokenizer=tokenizer
    )
    destination = Path(output_dir).resolve() / "validation_receipt.json"
    reference = immutable_json(destination, value)
    write_sidecar(reference)
    return {**value, "validation_receipt_sha256": reference.sha256}


__all__ = ["validate_completed_run", "write_validation_receipt"]
