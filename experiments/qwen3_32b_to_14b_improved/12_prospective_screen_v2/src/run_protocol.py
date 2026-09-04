"""Freeze run definitions and seal complete raw grids; never launches a GPU."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoints import result_path, validate_raw_checkpoint
from .configuration import load_config
from .contracts import (
    ALL_MODES,
    ArtifactRef,
    ContractError,
    RAW_MODES,
    artifact_ref,
    canonical_sha256,
    code_inventory,
    immutable_json,
    read_json,
    require_child,
    sha256_file,
    verify_sidecar,
    write_sidecar,
)
from .prepare_data import load_prepared
from .promotion import evaluate_confirm_evidence, evaluate_screen_promotion
from .statistics import aggregate_paired_results
from .two_pass_selector import derive_two_pass_result, validate_derived_result
from .gpu_runtime import validate_method_bindings
from .run_state import run_state_lock, write_terminal_status


def config_ref(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(config["config_path"])
    return {"path": str(path), "nbytes": path.stat().st_size, "sha256": config["config_sha256"]}


def canonical_run_output(
    config: Mapping[str, Any], profile: str, output_dir: str | Path
) -> Path:
    output = require_child(config["outputs"]["runs_root"], output_dir, label="run output")
    expected = Path(config["outputs"]["runs_root"]).resolve() / f"{profile}_v2"
    if output != expected:
        raise ContractError(f"{profile} has one canonical run directory: {expected}")
    return output


def method_binding_sha256(
    config: Mapping[str, Any], mode: str, validated_mode_digest: str
) -> str:
    if mode not in RAW_MODES:
        raise ContractError(f"unknown raw method: {mode}")
    return canonical_sha256(
        {
            "method": config["methods"][mode],
            "direction": config["direction"],
            "source_model": config["runtime"]["source_model"],
            "target_model": config["runtime"]["target_model"],
            "generation": config["generation"],
            "config_sha256": config["config_sha256"],
            "runtime_provenance_receipt": config["runtime_provenance_receipt"],
            "validated_mode_digest": validated_mode_digest,
        }
    )


def _validate_screen_receipt_for_confirm(config: Mapping[str, Any], reference_value: Any) -> dict[str, Any]:
    reference = ArtifactRef.parse(reference_value, label="screen validation receipt")
    path = require_child(config["outputs"]["runs_root"], reference.path, label="screen validation receipt")
    if path.name != "validation_receipt.json":
        raise ContractError("confirm prerequisite must be a screen validation receipt")
    reference.verify(label="screen validation receipt")
    verify_sidecar(path)
    serialized = read_json(path)
    if not isinstance(serialized, dict):
        raise ContractError("screen validation receipt root is malformed")
    from .validator import validate_completed_run

    observed = validate_completed_run(config["config_path"], profile="screen", output_dir=path.parent)
    if serialized != observed:
        raise ContractError("screen validation receipt does not reproduce")
    fixed = {
        "status": "validated_complete",
        "profile": "screen",
        "promotion_status": "pass",
        "authorizes_confirm": True,
        "config": config_ref(config),
        "official_dev_or_test_opened": False,
    }
    for field, expected in fixed.items():
        if serialized.get(field) != expected:
            raise ContractError(f"screen confirm prerequisite {field} changed")
    return {
        "rule_id": config["confirmation_gate"]["rule_id"],
        "screen_validation_receipt": reference.to_dict(),
        "validated_screen": serialized,
    }


def build_run_definition(
    config: Mapping[str, Any],
    profile: str,
    rows: Sequence[Mapping[str, Any]],
    prepared_manifest_ref: ArtifactRef,
    *,
    confirmation_prerequisite: Mapping[str, Any] | None,
    runtime_binding_digest: str,
    validated_method_bindings_digest: str,
    validated_mode_digests: Mapping[str, str],
) -> dict[str, Any]:
    if profile == "confirm":
        if confirmation_prerequisite is None:
            raise ContractError("confirm run lacks validated screen prerequisite")
    elif confirmation_prerequisite is not None:
        raise ContractError("non-confirm run cannot contain a confirm prerequisite")
    body: dict[str, Any] = {
        "schema_version": 2,
        "protocol_id": config["protocol_id"],
        "direction": config["direction"],
        "tier": config["tier"],
        "profile": profile,
        "sample_count": len(rows),
        "profile_dataset_counts": dict(config["profiles"][profile]),
        "expected_raw_results": len(rows) * len(RAW_MODES),
        "expected_derived_results": len(rows),
        "raw_modes": list(RAW_MODES),
        "derived_modes": ["two_pass_r1_then_r32"],
        "sample_bindings_sha256": canonical_sha256([row["sample_binding_sha256"] for row in rows]),
        "prepared_manifest": prepared_manifest_ref.to_dict(),
        "config": config_ref(config),
        "code_inventory": code_inventory(),
        "method_binding_sha256": {
            mode: method_binding_sha256(config, mode, validated_mode_digests[mode])
            for mode in RAW_MODES
        },
        "validated_method_bindings_digest": validated_method_bindings_digest,
        "validated_mode_digests": dict(validated_mode_digests),
        "registry_binding": {
            "published_marker": config["registry"]["published_marker"],
            "registry_manifest": config["registry"]["registry_manifest"],
            "publication_receipt": config["registry"]["publication_receipt"],
            "split_artifacts": config["registry"]["split_artifacts"],
        },
        "complete_g0_v2_receipt": config["g0"]["completion_receipt"],
        "generation": config["generation"],
        "selector": config["selector"],
        "promotion_gate": config["promotion_gate"],
        "methods": config["methods"],
        "runtime": config["runtime"],
        "runtime_provenance_receipt": config["runtime_provenance_receipt"],
        "runtime_binding_digest": runtime_binding_digest,
        "hypothesis_provenance": config["hypothesis_provenance"],
        "confirmation_prerequisite": confirmation_prerequisite,
        "official_dev_or_test_opened": False,
        "claim_boundary": config["claim_boundary"],
    }
    return {**body, "request_fingerprint": canonical_sha256(body)}


def freeze_run_definition(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
    screen_validation_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, verify_inputs=True)
    method_bindings = validate_method_bindings(config, deep_model_hash=False)
    output = canonical_run_output(config, profile, output_dir)
    rows, _manifest, manifest_ref = load_prepared(config, profile)
    prerequisite = None
    if profile == "confirm":
        if screen_validation_receipt is None:
            raise ContractError("confirm freeze requires an explicit screen validation receipt ref")
        prerequisite = _validate_screen_receipt_for_confirm(config, screen_validation_receipt)
    elif screen_validation_receipt is not None:
        raise ContractError("screen/smoke freeze cannot accept a confirm prerequisite")
    definition = build_run_definition(
        config,
        profile,
        rows,
        manifest_ref,
        confirmation_prerequisite=prerequisite,
        runtime_binding_digest=method_bindings.runtime["binding_digest"],
        validated_method_bindings_digest=method_bindings.digest,
        validated_mode_digests=method_bindings.mode_digests,
    )
    reference = immutable_json(output / "run_definition.json", definition)
    write_sidecar(reference)
    return {**definition, "run_definition_sha256": reference.sha256}


def reproduce_run_definition(
    config: Mapping[str, Any], profile: str, output: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = canonical_run_output(config, profile, output)
    definition_path = output / "run_definition.json"
    verify_sidecar(definition_path)
    frozen = read_json(definition_path)
    if not isinstance(frozen, dict):
        raise ContractError("run definition root is malformed")
    prerequisite = frozen.get("confirmation_prerequisite")
    if profile == "confirm":
        if not isinstance(prerequisite, Mapping) or set(prerequisite) != {
            "rule_id",
            "screen_validation_receipt",
            "validated_screen",
        }:
            raise ContractError("confirm prerequisite schema changed")
        observed = _validate_screen_receipt_for_confirm(
            config, prerequisite["screen_validation_receipt"]
        )
        if observed != prerequisite:
            raise ContractError("confirm prerequisite changed")
    elif prerequisite is not None:
        raise ContractError("non-confirm definition contains a prerequisite")
    rows, _manifest, manifest_ref = load_prepared(config, profile)
    method_bindings = validate_method_bindings(config, deep_model_hash=False)
    expected = build_run_definition(
        config,
        profile,
        rows,
        manifest_ref,
        confirmation_prerequisite=prerequisite,
        runtime_binding_digest=method_bindings.runtime["binding_digest"],
        validated_method_bindings_digest=method_bindings.digest,
        validated_mode_digests=method_bindings.mode_digests,
    )
    if frozen != expected:
        raise ContractError("run definition does not reproduce from current frozen inputs")
    return frozen, rows


def _validate_raw_grid(
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    tokenizer: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[Path]]:
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    expected_paths: set[Path] = set()
    for row in rows:
        native_path = result_path(output, row, "native14")
        baseline_path = result_path(output, row, "nvidia_k20")
        if not native_path.is_file() or not baseline_path.is_file():
            raise ContractError("raw grid is missing native14/nvidia_k20")
        native_sha = sha256_file(native_path)
        baseline_sha = sha256_file(baseline_path)
        for mode in RAW_MODES:
            path = result_path(output, row, mode)
            expected_paths.add(path.resolve())
            verify_sidecar(path)
            value = validate_raw_checkpoint(
                path,
                row=row,
                mode=mode,
                request_fingerprint=definition["request_fingerprint"],
                method_binding_sha256=definition["method_binding_sha256"][mode],
                tokenizer=tokenizer,
                expected_native_file_sha256=native_sha if mode != "native14" else None,
                expected_baseline_file_sha256=(
                    baseline_sha if mode in {"shrinkage", "tail_refresh_r32"} else None
                ),
            )
            records.append(value)
            inventory.append(artifact_ref(path).to_dict())
    return records, inventory, expected_paths


def _derive_grid(
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    write: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[Path]]:
    records: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    paths: set[Path] = set()
    for row in rows:
        r1_path = result_path(output, row, "nvidia_k20")
        r32_path = result_path(output, row, "tail_refresh_r32")
        r1 = read_json(r1_path)
        r32 = read_json(r32_path)
        value = derive_two_pass_result(
            r1,
            r32,
            r1_file_sha256=sha256_file(r1_path),
            r32_file_sha256=sha256_file(r32_path),
        )
        path = result_path(output, row, "two_pass_r1_then_r32", derived=True)
        if write:
            reference = immutable_json(path, value)
            write_sidecar(reference)
        verify_sidecar(path)
        frozen = read_json(path)
        validate_derived_result(
            frozen,
            r1,
            r32,
            r1_file_sha256=sha256_file(r1_path),
            r32_file_sha256=sha256_file(r32_path),
        )
        records.append(frozen)
        inventory.append(artifact_ref(path).to_dict())
        paths.add(path.resolve())
    return records, inventory, paths


def collect_and_validate_results(
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    tokenizer: Any | None,
    write_derived: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw, raw_inventory, raw_paths = _validate_raw_grid(
        definition, rows, output, tokenizer=tokenizer
    )
    derived, derived_inventory, derived_paths = _derive_grid(
        definition, rows, output, write=write_derived
    )
    results_root = output / "results"
    observed = {
        path.resolve()
        for path in results_root.rglob("*.json")
        if path.is_file() and not path.name.endswith(".sha256.json") and not path.name.endswith(".failure.json")
    }
    failures = [path for path in results_root.rglob("*.failure.json") if path.is_file()]
    if observed != raw_paths | derived_paths or failures:
        raise ContractError("result tree contains missing/unexpected/failure checkpoints")
    return raw + derived, raw_inventory + derived_inventory


def _seal_completed_run_locked(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    config = load_config(config_path, verify_inputs=True)
    output = canonical_run_output(config, profile, output_dir)
    definition, rows = reproduce_run_definition(config, profile, output)
    # Local import avoids a module cycle: the real GPU runner reproduces the
    # frozen definition through this module.  A CPU-authored raw grid is not a
    # formal run; sealing requires the content-addressed real-execution proof.
    from .gpu_runner import validate_raw_completion_receipt

    if tokenizer is None:
        from .gpu_runner import load_frozen_decode_tokenizer

        tokenizer = load_frozen_decode_tokenizer(config)
    validate_raw_completion_receipt(
        config,
        definition,
        rows,
        output,
        repair_missing_sidecar=True,
        tokenizer=tokenizer,
    )
    raw_completion_ref = artifact_ref(output / "raw_completion_receipt.json")
    records, inventory = collect_and_validate_results(
        definition, rows, output, tokenizer=tokenizer, write_derived=True
    )
    bootstrap = config["metrics"]["bootstrap"]
    summary = aggregate_paired_results(
        records,
        profile=profile,
        expected_dataset_counts=definition["profile_dataset_counts"],
        request_fingerprint=definition["request_fingerprint"],
        seed=bootstrap["seed"],
        replicates=bootstrap["replicates"],
        confidence_level=bootstrap["confidence_level"],
    )
    summary_ref = immutable_json(output / "summary.json", summary)
    write_sidecar(summary_ref)
    if profile == "screen":
        decision = evaluate_screen_promotion(summary)
    elif profile == "confirm":
        decision = evaluate_confirm_evidence(summary)
    else:
        decision = {
            "schema_version": 2,
            "profile": "smoke",
            "status": "not_evaluated",
            "reason": "smoke checks execution integrity only and cannot promote",
            "authorizes_confirm": False,
            "authorizes_locked_test_or_paper_superiority_claim": False,
        }
    decision_ref = immutable_json(output / "decision.json", decision)
    write_sidecar(decision_ref)
    definition_ref = artifact_ref(output / "run_definition.json")
    receipt = {
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
        "promotion_status": decision.get("status"),
        "authorizes_confirm": decision.get("authorizes_confirm", False),
        "complete_g0_v2_receipt": config["g0"]["completion_receipt"],
        "registry_binding": definition["registry_binding"],
        "official_dev_or_test_opened": False,
        "claim_boundary": config["claim_boundary"],
    }
    receipt_ref = immutable_json(output / "completion_receipt.json", receipt)
    write_sidecar(receipt_ref)
    status = {
        "schema_version": 2,
        "status": "complete",
        "profile": profile,
        "request_fingerprint": definition["request_fingerprint"],
        "completed_results": len(inventory),
        "completion_receipt_sha256": receipt_ref.sha256,
    }
    # The live GPU runner owns a mutable status file.  Completion is the one
    # terminal transition, after which its immutable hash sidecar is created.
    write_terminal_status(output / "status.json", status)
    return {**receipt, "completion_receipt_sha256": receipt_ref.sha256}


def seal_completed_run(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Serialize CPU sealing against all GPU writers for this canonical run."""

    config = load_config(config_path, verify_inputs=True)
    output = canonical_run_output(config, profile, output_dir)
    with run_state_lock(output):
        return _seal_completed_run_locked(
            config_path,
            profile=profile,
            output_dir=output,
            tokenizer=tokenizer,
        )


__all__ = [
    "build_run_definition",
    "canonical_run_output",
    "collect_and_validate_results",
    "config_ref",
    "freeze_run_definition",
    "method_binding_sha256",
    "reproduce_run_definition",
    "seal_completed_run",
]
