from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from .common import (
    ContractError,
    ensure_no_symlink_components,
    load_json_strict,
    placeholders,
    require_exact_keys,
    require_int,
    validate_artifact_ref,
)
from .anchor import validate_anchor_manifest
from .tier_gate import validate_answer_free_oracle_receipt, validate_capture_manifest, validate_registry_document


ROOT_KEYS = (
    "schema_version",
    "experiment_id",
    "status",
    "direction",
    "method_id",
    "execution_policy",
    "experiment_root",
    "anchor",
    "integrity_inputs",
    "data_policy",
    "architecture",
    "layer_selection",
    "runtime_guards",
    "functional_capture",
    "loss",
    "optimization",
    "system_comparisons",
    "comparison_policy",
    "formal_runtime",
    "outputs",
    "claim_policy",
)


def _pure_inside(path_text: str, root: Path, name: str) -> Path:
    if not isinstance(path_text, str) or not path_text.startswith("/"):
        raise ContractError(f"{name} must be an absolute path")
    path = Path(path_text)
    try:
        relative = path.relative_to(root)
    except ValueError:
        raise ContractError(f"{name} escapes experiment root") from None
    if ".." in relative.parts:
        raise ContractError(f"{name} escapes experiment root via lexical '..'")
    # Pure lexical containment catches `..` even when the leaf does not yet
    # exist. Existing components are checked for symlinks separately.
    current = root
    for component in relative.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            ensure_no_symlink_components(current)
    return path


def _validate_common(config: Mapping[str, Any], path: Path) -> dict[str, Any]:
    require_exact_keys(config, ROOT_KEYS, "config")
    if config["schema_version"] != 1 or config["direction"] != "qwen3_32b_to_14b":
        raise ContractError("unsupported config schema or direction")
    if config["method_id"] != "frozen_anchor_head_structured_functional_residual_v1":
        raise ContractError("unexpected method_id")
    root = Path(config["experiment_root"])
    if not root.is_absolute() or root != path.parent:
        raise ContractError("experiment_root must equal config parent")
    ensure_no_symlink_components(root)
    require_exact_keys(config["outputs"], ("runs_root", "artifacts_root", "checkpoints_root"), "outputs")
    for key, value in config["outputs"].items():
        if not isinstance(value, str):
            raise ContractError(f"outputs.{key} must be a path string")
        _pure_inside(value, root, f"outputs.{key}")

    execution = config["execution_policy"]
    require_exact_keys(
        execution,
        (
            "formal_config",
            "gpu_execution_authorized_by_this_config",
            "locked_outcomes_must_remain_unopened",
            "locked_or_test_examples_must_remain_unopened",
            "answers_may_influence_selection",
            "source_32b_logits_as_teacher_allowed",
            "writes_must_resolve_under_experiment_root",
        ),
        "execution_policy",
    )
    for key in (
        "formal_config",
        "gpu_execution_authorized_by_this_config",
        "locked_outcomes_must_remain_unopened",
        "locked_or_test_examples_must_remain_unopened",
        "answers_may_influence_selection",
        "source_32b_logits_as_teacher_allowed",
        "writes_must_resolve_under_experiment_root",
    ):
        if not isinstance(execution[key], bool):
            raise ContractError(f"execution_policy.{key} must be a bool")
    if execution["locked_outcomes_must_remain_unopened"] is not True or execution["locked_or_test_examples_must_remain_unopened"] is not True:
        raise ContractError("locked/test protection must be enabled")
    if execution["answers_may_influence_selection"] is not False or execution["source_32b_logits_as_teacher_allowed"] is not False:
        raise ContractError("answer-free/native-teacher policy violated")

    anchor = config["anchor"]
    require_exact_keys(
        anchor,
        (
            "id",
            "selection_state",
            "manifest",
            "deep_payload_hash_required",
            "mutable",
            "fallback",
            "dev_metric_based_anchor_switching_allowed",
            "k4_is_only_a_separately_preregistered_future_control",
        ),
        "anchor",
    )
    if anchor["id"] != "headwise_shrinkage_v1" or anchor["selection_state"] != "frozen_before_functional_training":
        raise ContractError("anchor identity/state mismatch")
    if anchor["deep_payload_hash_required"] is not True or anchor["mutable"] is not False:
        raise ContractError("anchor mutability/deep-hash policy invalid")
    if anchor["dev_metric_based_anchor_switching_allowed"] is not False or anchor["k4_is_only_a_separately_preregistered_future_control"] is not True:
        raise ContractError("anchor switching policy invalid")
    anchor_path = validate_artifact_ref(anchor["manifest"], "anchor.manifest")
    # Shallow anchor validation reads only manifest/sidecars and checks file
    # presence. The expensive layer-payload SHA walk is reserved for formal
    # pre-torch validation (`validate_formal`).
    anchor_summary = validate_anchor_manifest(anchor["manifest"], deep=False)

    architecture = config["architecture"]
    for key in (
        "source_layers",
        "target_layers",
        "selected_source_layers_per_target",
        "kv_heads",
        "query_heads",
        "head_dim",
        "input_dim",
    ):
        require_int(architecture[key], f"architecture.{key}", minimum=1)
    expected_arch = {
        "source_layers": 64,
        "target_layers": 40,
        "selected_source_layers_per_target": 20,
        "kv_heads": 8,
        "query_heads": 40,
        "head_dim": 128,
        "input_dim": 20480,
    }
    for key, expected in expected_arch.items():
        if architecture[key] != expected:
            raise ContractError(f"architecture.{key} must be {expected}")
    if architecture["components"] != ["key", "value"] or architecture["decoder_zero_initialized"] is not True or architecture["delta_bias_zero_initialized"] is not True or architecture["step_zero_is_exact_anchor"] is not True:
        raise ContractError("residual structural contract mismatch")
    if architecture["rank_candidates_in_order"] != [64, 128, 256]:
        raise ContractError("rank candidate order is not frozen")
    if architecture["oracle_candidate_layers"] != [7, 8, 9, 10, 34, 35]:
        raise ContractError("oracle layer catalog mismatch")

    policy = config["data_policy"]
    if policy["parameter_fit_tier"] != "mapper-fit" or policy["hyperparameter_selection_tier"] != "method-dev" or policy["confirmatory_tier"] != "locked-test":
        raise ContractError("data tier order mismatch")
    if policy["answer_fields_in_mapper_fit_or_selection_records_allowed"] is not False or policy["model_outcomes_in_selection_records_allowed"] is not False:
        raise ContractError("data answer/outcome policy invalid")
    if policy["teacher"] != "native_qwen3_14b" or policy["source_32b_logits_as_teacher_allowed"] is not False:
        raise ContractError("teacher policy invalid")

    systems = config["system_comparisons"]
    if not isinstance(systems, list) or len(systems) != 2:
        raise ContractError("exactly R1 and R32 system classes are required")
    if systems[0] != {
        "id": "functional_residual_r1",
        "target_replay_tokens": 1,
        "claim_class": "pure_mapper_primary",
        "tail_refresh": False,
    }:
        raise ContractError("R1 system contract mismatch")
    if systems[1] != {
        "id": "functional_residual_r32",
        "target_replay_tokens": 32,
        "claim_class": "optional_hybrid_quality_latency_secondary",
        "tail_refresh": True,
    }:
        raise ContractError("R32 system contract mismatch")
    if config["comparison_policy"]["pure_r1_must_be_evaluated_without_tail_refresh"] is not True or config["comparison_policy"]["r32_may_not_rescue_or_be_merged_with_r1_claim"] is not True:
        raise ContractError("R1/R32 separation policy invalid")
    if config["claim_policy"]["may_claim_better_than_nvidia_now"] is not False:
        raise ContractError("premature comparative claim enabled")
    return {"experiment_root": str(root), "anchor_manifest": str(anchor_path), "anchor_summary": anchor_summary}


def validate_template(path: Path) -> dict[str, Any]:
    path = ensure_no_symlink_components(path)
    config = load_json_strict(path)
    common = _validate_common(config, path)
    if config["status"] != "preformal_template":
        raise ContractError("template status must be preformal_template")
    if config["execution_policy"]["formal_config"] is not False or config["execution_policy"]["gpu_execution_authorized_by_this_config"] is not False:
        raise ContractError("template must be non-formal and GPU-disabled")
    found = placeholders(config)
    required_fragments = {
        "$.integrity_inputs.complete_g0_v2",
        "$.integrity_inputs.decontaminated_registry_receipt",
        "$.integrity_inputs.train_only_oracle_receipt",
        "$.integrity_inputs.capture_manifest",
        "$.integrity_inputs.tier_registry",
        "$.architecture.rank",
        "$.architecture.active_layers",
        "$.architecture.layer_loss_weights",
        "$.formal_runtime.gpu_plan",
        "$.formal_runtime.capture_runner",
        "$.formal_runtime.target_receiver_adapter",
    }
    missing = [
        fragment
        for fragment in required_fragments
        if not any(item == fragment or item.startswith(fragment + ".") for item in found)
    ]
    if missing:
        raise ContractError(f"template missing required placeholders: {sorted(missing)}")
    return {
        "status": "PRE_FORMAL_TEMPLATE_VALID",
        "formal_execution_blocked": True,
        "gpu_execution_performed": False,
        "locked_outcomes_opened": False,
        "placeholder_count": len(found),
        "placeholder_paths": found,
        **common,
    }


def validate_formal(path: Path) -> dict[str, Any]:
    path = ensure_no_symlink_components(path)
    config = load_json_strict(path)
    common = _validate_common(config, path)
    if config["status"] != "formal_frozen":
        raise ContractError("formal config status must be formal_frozen")
    if placeholders(config):
        raise ContractError("formal config still contains placeholders")
    if config["execution_policy"]["formal_config"] is not True or config["execution_policy"]["gpu_execution_authorized_by_this_config"] is not True:
        raise ContractError("formal config is not explicitly authorized")
    active = config["architecture"]["active_layers"]
    if not isinstance(active, list) or not active or active != sorted(set(active)) or any(layer not in [7, 8, 9, 10, 34, 35] for layer in active):
        raise ContractError("formal active layer set invalid")
    rank = config["architecture"]["rank"]
    if rank not in [64, 128, 256]:
        raise ContractError("formal rank invalid")
    weights = config["architecture"]["layer_loss_weights"]
    if not isinstance(weights, Mapping) or set(weights) != {str(layer) for layer in active}:
        raise ContractError("formal layer weights key order invalid")

    # Re-run the full payload walk only after the formal config is fully
    # frozen. This is file hashing, not model/GPU execution.
    anchor_summary = validate_anchor_manifest(config["anchor"]["manifest"], deep=True)

    refs = config["integrity_inputs"]
    for name, value in refs.items():
        validate_artifact_ref(value, f"integrity_inputs.{name}")
    registry = load_json_strict(Path(refs["tier_registry"]["path"]))
    registry_summary = validate_registry_document(registry)
    capture = load_json_strict(Path(refs["capture_manifest"]["path"]))
    capture_summary = validate_capture_manifest(capture, expected_tier="mapper-fit")
    oracle = load_json_strict(Path(refs["train_only_oracle_receipt"]["path"]))
    oracle_summary = validate_answer_free_oracle_receipt(oracle, expected_candidate_layers=[7, 8, 9, 10, 34, 35])
    if oracle_summary["active_layers"] != active:
        raise ContractError("formal active layers do not match oracle")
    return {
        "status": "FORMAL_CONFIG_VALID",
        "formal_execution_blocked": False,
        "gpu_execution_performed": False,
        "locked_outcomes_opened": False,
        "registry": registry_summary,
        "capture": capture_summary,
        "oracle": oracle_summary,
        "anchor": anchor_summary,
        **common,
    }


def validate_config(path: Path, *, expect_template_blocked: bool = False) -> dict[str, Any]:
    config = load_json_strict(path)
    if config.get("status") == "preformal_template":
        report = validate_template(path)
        if not expect_template_blocked:
            raise ContractError("template is intentionally blocked; pass --expect-template-blocked")
        return report
    return validate_formal(path)
