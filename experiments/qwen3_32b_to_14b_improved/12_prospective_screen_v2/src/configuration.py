"""Strict configuration loading and immutable protocol validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ArtifactRef,
    ContractError,
    canonical_sha256,
    DATASETS,
    PROFILE_MAX_COUNTS,
    PROFILES,
    RAW_MODES,
    RUNTIME_PACKAGE_CLOSURE_FILES,
    RUNTIME_PACKAGE_CLOSURE_ROOT,
    ROOT,
    SMOKE_COUNTS,
    SPLIT_KEYS,
    assert_no_placeholders,
    read_json,
    sha256_file,
)


STATIC_GENERATION = {
    "max_length": 16384,
    "max_new_tokens": 256,
    "batch_size": 1,
    "do_sample": False,
    "temperature": 0.0,
    "chat_mode": "qwen_chat_no_think",
    "enable_thinking": False,
    "stop_sequences": ["\nQuestion:", "\nContext:"],
    "overlength_policy": "exact_token_id_head_tail_before_runtime",
}
STATIC_SELECTOR = {
    "selector_id": "complete_r1_then_gold_blind_choose_r32_v2",
    "derived_mode": "two_pass_r1_then_r32",
    "initial_mode": "nvidia_k20",
    "repair_mode": "tail_refresh_r32",
    "stall_budget_generated_tokens": 32,
    "trigger": "r1_generated_tokens_gt_32_or_cap_hit",
    "requires_complete_r1": True,
    "requires_complete_r32_raw_for_every_sample": True,
    "gold_blind": True,
    "dataset_blind": True,
    "online_watchdog": False,
    "execution_semantics": "offline_two_pass_selector_over_complete_raw_runs",
}
STATIC_PROMOTION = {
    "rule_id": "prospective_screen_two_pass_vs_nvidia_k20_v2",
    "candidate_mode": "two_pass_r1_then_r32",
    "baseline_mode": "nvidia_k20",
    "target_mode": "native14",
    "primary_metric": "official_answer_f1",
    "per_dataset_delta_vs_baseline_min": 0.0,
    "pooled_delta_vs_baseline_strictly_positive": True,
    "per_dataset_cap_rate_must_not_increase": True,
    "per_dataset_effective_termination_must_not_decrease": True,
    "per_dataset_r1_correct_em_retention_min": 0.98,
    "requires_zero_failed_or_missing_cells": True,
    "screen_is_single_go_no_go": True,
    "screen_threshold_tuning_forbidden": True,
    "confirm_requires_screen_promotion_pass": True,
    "confirm_superiority_requires_pooled_f1_ci_low_gt_zero": True,
}
STATIC_CONFIRMATION = {
    "rule_id": "screen_pass_then_disjoint_confirm_v2",
    "required_profile": "screen",
    "required_validation_status": "validated_complete",
    "required_promotion_status": "pass",
    "requires_explicit_content_addressed_screen_receipt": True,
    "outcome_threshold_changes_after_screen_forbidden": True,
}
STATIC_PROMPT = {
    "template": (
        "Answer the multi-hop question using only the supplied context. "
        "Return only the shortest answer span and no explanation.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ),
    "paragraph_order": "official_order_hotpotqa; official_idx_ascending_musique",
    "answer_dereference_stage": "only_after_frozen_registry_ids_and_complete_g0_validation",
}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _not_locked_outcome_path(path: str, label: str) -> None:
    parts = [part.casefold() for part in Path(path).parts]
    banned_exact = {"dev", "test", "validation", "official_test", "official_validation"}
    banned_fragments = ("locked_dev", "dev_locked", "official_dev", "locked_test", "test_locked")
    if any(part in banned_exact or any(fragment in part for fragment in banned_fragments) for part in parts):
        raise ContractError(f"{label} points at a locked/dev/test outcome path: {path}")


def _artifact(value: Any, label: str, verify: bool) -> ArtifactRef:
    reference = ArtifactRef.parse(value, label=label)
    if verify:
        reference.verify(label=label)
    return reference


def validated_profile_counts(value: Any) -> dict[str, dict[str, int]]:
    """Validate final experiment-10-derived counts without pre-freezing them.

    Smoke remains 2+2.  Screen/confirm counts are exact formal-config inputs
    copied from the final atomic publication and are bounded by the original
    prospective design, so feasibility repair cannot silently expand scope.
    """

    profiles = _require_mapping(value, "profiles")
    if set(profiles) != set(PROFILES):
        raise ContractError("profile set changed")
    result: dict[str, dict[str, int]] = {}
    for profile in PROFILES:
        counts = _require_mapping(profiles[profile], f"profiles.{profile}")
        if set(counts) != set(DATASETS):
            raise ContractError(f"profiles.{profile} dataset set changed")
        parsed: dict[str, int] = {}
        for dataset in DATASETS:
            count = counts[dataset]
            maximum = PROFILE_MAX_COUNTS[profile][dataset]
            if type(count) is not int or count <= 0 or count > maximum:
                raise ContractError(
                    f"profiles.{profile}.{dataset} must be an exact positive int <= {maximum}"
                )
            parsed[dataset] = count
        result[profile] = parsed
    if result["smoke"] != SMOKE_COUNTS:
        raise ContractError("smoke profile must remain exact 2+2")
    for dataset in DATASETS:
        if result["screen"][dataset] < SMOKE_COUNTS[dataset]:
            raise ContractError(f"screen/{dataset} cannot be smaller than smoke")
    return result


def load_config(path: str | Path, *, verify_inputs: bool = True) -> dict[str, Any]:
    """Load a *formal frozen* config.

    Placeholder detection happens before any referenced path is touched.  Thus
    ``config.template.json`` can never read an in-progress registry by mistake.
    """

    config_path = Path(path).resolve()
    value = read_json(config_path)
    if not isinstance(value, dict):
        raise ContractError("config root must be an object")
    assert_no_placeholders(value)
    fixed = {
        "schema_version": 2,
        "protocol_id": "qwen3_32b_to_14b_prospective_decontaminated_v2",
        "status": "frozen_ready",
        "runnable": True,
        "direction": "qwen3_32b_to_14b",
        "tier": "method_dev_prospective",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ContractError(f"config {field} must equal {expected!r}")
    if value.get("generation") != STATIC_GENERATION:
        raise ContractError("generation contract changed")
    validated_profile_counts(value.get("profiles"))
    if value.get("prompt") != STATIC_PROMPT:
        raise ContractError("prompt/data-dereference contract changed")
    if value.get("raw_modes") != list(RAW_MODES):
        raise ContractError("raw modes or ordering changed")
    if value.get("selector") != STATIC_SELECTOR:
        raise ContractError("two-pass selector contract changed")
    if value.get("promotion_gate") != STATIC_PROMOTION:
        raise ContractError("promotion gate changed")
    if value.get("confirmation_gate") != STATIC_CONFIRMATION:
        raise ContractError("confirmation prerequisite changed")

    registry = _require_mapping(value.get("registry"), "registry")
    if registry.get("publication_state_required") != "published_complete":
        raise ContractError("registry must require native published_complete state")
    if registry.get("contains_ids_hashes_ranks_only") is not True:
        raise ContractError("registry must be answer/outcome free")
    expected_strata = {
        "hotpotqa_train": [
            "bridge/easy",
            "bridge/medium",
            "bridge/hard",
            "comparison/easy",
            "comparison/medium",
            "comparison/hard",
        ],
        "musique_train": ["2hop", "3hop", "4hop"],
    }
    if registry.get("strata_order") != expected_strata:
        raise ContractError("native registry strata order changed")
    # Only the atomic last marker may be opened before publication is proven.
    _artifact(registry.get("published_marker"), "registry.published_marker", verify_inputs)
    _artifact(registry.get("registry_manifest"), "registry.registry_manifest", False)
    _artifact(registry.get("publication_receipt"), "registry.publication_receipt", False)
    split_artifacts = _require_mapping(registry.get("split_artifacts"), "registry.split_artifacts")
    if set(split_artifacts) != set(SPLIT_KEYS):
        raise ContractError("registry must bind exactly four ordered split artifacts")
    for key in SPLIT_KEYS:
        _artifact(split_artifacts[key], f"registry.split_artifacts.{key}", False)

    g0 = _require_mapping(value.get("g0"), "g0")
    if g0.get("required_scope") != "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval":
        raise ContractError("G0 scope changed")
    if g0.get("requires_zero_flags_all_registered_directions") is not True:
        raise ContractError("G0 zero-collision gate was weakened")
    _artifact(g0.get("completion_receipt"), "g0.completion_receipt", False)

    sources = _require_mapping(value.get("train_sources"), "train_sources")
    if set(sources) != set(DATASETS):
        raise ContractError("train source ordering/content changed")
    expected = {
        "hotpotqa_train": (90447, "official_train_v1.1", "_id", "json"),
        "musique_train": (19938, "official_answerable_train_v1.0", "id", "jsonl"),
    }
    for dataset_id, contract in expected.items():
        source = _require_mapping(sources[dataset_id], f"train_sources.{dataset_id}")
        observed = (
            source.get("rows"), source.get("split"), source.get("id_field"), source.get("format")
        )
        if observed != contract or source.get("role") != "method_dev_train":
            raise ContractError(f"{dataset_id} train-source contract changed")
        ref = _artifact(source.get("artifact"), f"train_sources.{dataset_id}.artifact", verify_inputs)
        _not_locked_outcome_path(ref.path, f"train_sources.{dataset_id}.artifact")

    metrics = _require_mapping(value.get("metrics"), "metrics")
    if metrics.get("headline") != ["official_answer_f1", "official_answer_em"]:
        raise ContractError("headline metrics changed")
    if metrics.get("bootstrap") != {
        "seed": 20260830,
        "replicates": 10000,
        "confidence_level": 0.95,
        "resampling_unit": "paired_sample_id",
    }:
        raise ContractError("bootstrap contract changed")
    _artifact(metrics.get("hotpotqa_reference"), "metrics.hotpotqa_reference", verify_inputs)
    _artifact(metrics.get("musique_reference"), "metrics.musique_reference", verify_inputs)

    methods = _require_mapping(value.get("methods"), "methods")
    if set(methods) != set(RAW_MODES):
        raise ContractError("method bindings must match raw modes exactly")
    for mode in RAW_MODES:
        method = _require_mapping(methods[mode], f"methods.{mode}")
        if method.get("state") != "ready" or not isinstance(method.get("method_id"), str):
            raise ContractError(f"method {mode} is not frozen ready")
        for label, candidate in method.items():
            if label.endswith("_artifact") or label.endswith("_receipt") or label.endswith("_interface"):
                _artifact(candidate, f"methods.{mode}.{label}", verify_inputs)
    if {
        key: methods["shrinkage"].get(key)
        for key in ("module_name", "symbol", "factory", "verify_mapper_payloads")
    } != {
        "module_name": "qwen32to14_shrinkage_runtime_prospective_v2",
        "symbol": "ShrinkageMappedPrefixCacheBuilder",
        "factory": "constructor",
        "verify_mapper_payloads": True,
    }:
        raise ContractError("shrinkage runtime factory contract changed")
    if {
        key: methods["tail_refresh_r32"].get(key)
        for key in ("module_name", "symbol", "factory", "replay_tokens")
    } != {
        "module_name": "qwen32to14_tail_refresh_runtime_prospective_v2",
        "symbol": "TailRefreshMappedPrefixCacheBuilder",
        "factory": "from_shared_mapper",
        "replay_tokens": 32,
    }:
        raise ContractError("R32 runtime factory contract changed")

    provenance = _require_mapping(value.get("hypothesis_provenance"), "hypothesis_provenance")
    if provenance.get("role") != "post_hoc_hypothesis_provenance_only_not_runtime_binding":
        raise ContractError("experiment 11 provenance role changed")
    for name in ("config", "method", "protocol", "receipt"):
        _artifact(provenance.get(name), f"hypothesis_provenance.{name}", verify_inputs)
    _artifact(value.get("design_reference_inventory"), "design_reference_inventory", verify_inputs)

    runtime = _require_mapping(value.get("runtime"), "runtime")
    expected_runtime = {
        "source_model": "/home/wby/models/Qwen3-32B",
        "target_model": "/home/wby/models/Qwen3-14B",
        "source_devices": [0, 1, 2, 3, 4],
        "source_layer_counts": [12, 13, 14, 13, 12],
        "target_devices": [5, 6, 7],
        "target_layer_counts": [13, 14, 13],
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "python_executable": "/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python",
        "required_python_dont_write_bytecode": True,
        "required_python_pycache_prefix": "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/12_prospective_screen_v2/.formal_empty_pycache",
        "required_cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "global_gpu_lock": "/home/wby/llm-cache/nvidia-kvcache/runs/gpu8_experiment.lock",
        "idle_gate": {
            "memory_used_limit_mib": 64,
            "consecutive_samples": 12,
            "sample_interval_seconds": 5,
            "timeout_seconds": 86400,
        },
    }
    if runtime != expected_runtime:
        raise ContractError("runtime/topology contract changed")
    _artifact(
        value.get("runtime_provenance_receipt"),
        "runtime_provenance_receipt",
        verify_inputs,
    )
    package_closure = _require_mapping(
        value.get("runtime_package_closure"), "runtime_package_closure"
    )
    if set(package_closure) != {"root", "files", "files_sha256"}:
        raise ContractError("runtime package closure schema changed")
    if package_closure.get("root") != RUNTIME_PACKAGE_CLOSURE_ROOT:
        raise ContractError("runtime package closure root changed")
    package_files = package_closure.get("files")
    if package_files != RUNTIME_PACKAGE_CLOSURE_FILES:
        raise ContractError("runtime package closure file bindings changed")
    # The inventory digest is checked independently of the filesystem so a
    # malformed formal config cannot replace the frozen two-file closure.
    if package_closure.get("files_sha256") != canonical_sha256(package_files):
        raise ContractError("runtime package closure inventory digest changed")
    for relative, entry in package_files.items():
        _artifact(
            {
                "path": str(Path(RUNTIME_PACKAGE_CLOSURE_ROOT) / relative),
                **entry,
            },
            f"runtime_package_closure.files.{relative}",
            verify_inputs,
        )

    outputs = _require_mapping(value.get("outputs"), "outputs")
    if set(outputs) != {"prepared_root", "runs_root"}:
        raise ContractError("output schema changed")
    for name, output in outputs.items():
        output_path = Path(str(output)).resolve()
        try:
            relative = output_path.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ContractError(f"{name} escapes independent experiment 12") from exc
        if not relative.parts:
            raise ContractError(f"{name} cannot equal experiment root")

    if value.get("claim_boundary") != (
        "prospective method-dev only; official dev/test remains sealed; "
        "screen pass is not a locked-test or paper-superiority claim"
    ):
        raise ContractError("claim boundary changed")
    return {**value, "config_path": str(config_path), "config_sha256": sha256_file(config_path)}


__all__ = [
    "STATIC_CONFIRMATION",
    "STATIC_GENERATION",
    "STATIC_PROMOTION",
    "STATIC_PROMPT",
    "STATIC_SELECTOR",
    "load_config",
    "validated_profile_counts",
]
