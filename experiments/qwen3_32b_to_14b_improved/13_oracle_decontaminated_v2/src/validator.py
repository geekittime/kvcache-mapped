"""Independent CPU validation for atomic oracle checkpoints and receipts."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping

from .contracts import (
    ContractError,
    artifact_ref,
    assert_finite_tree,
    canonical_sha256,
    contract_preflight,
    ensure_formal_config_path,
    ensure_inside_output_root,
    intervention_catalog,
    load_config,
    read_json,
    sha256_file,
    verify_ref,
)


HEX64 = re.compile(r"[0-9a-f]{64}")


def result_path(output_dir: Path, sample_binding: Mapping[str, Any]) -> Path:
    dataset = sample_binding.get("dataset")
    sample_id = sample_binding.get("sample_id")
    if dataset not in {"hotpotqa", "musique"} or not isinstance(sample_id, str):
        raise ContractError("sample binding cannot name a result path")
    return output_dir / "results" / dataset / f"{sample_id}.json"


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def validate_atomic_result(
    path: str | Path,
    *,
    request_fingerprint: str,
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != 1 or value.get("status") != "complete":
        raise ContractError("atomic result is not schema-v1 complete")
    if value.get("request_fingerprint") != request_fingerprint:
        raise ContractError("atomic result request fingerprint changed")
    for key in (
        "dataset",
        "sample_id",
        "source_index",
        "selection_rank",
        "stratum",
        "source_projection_sha256",
        "input_commitment_sha256",
        "prompt_token_count",
        "serialized_prompt_sha256",
        "prompt_token_ids_sha256",
        "prefix_token_ids_sha256",
        "references_sha256",
        "gold_first_token_ids_sha256",
    ):
        if value.get(key) != plan_binding.get(key):
            raise ContractError(f"atomic result binding changed for {key}")
    for key in (
        "input_commitment_sha256",
        "prompt_token_ids_sha256",
        "prefix_token_ids_sha256",
        "serialized_prompt_sha256",
        "references_sha256",
        "gold_first_token_ids_sha256",
    ):
        if not _valid_sha(value.get(key)):
            raise ContractError(f"atomic result {key} is malformed")
    if value.get("mapped_cache_materializations") != 1:
        raise ContractError("k20 cache was not materialized exactly once for this prompt")
    boundary = value.get("causal_boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("prefix"),
        boundary.get("replay"),
        boundary.get("full_prompt_identity_passed"),
        boundary.get("native_boundary_qkv_identity_passed"),
    ) != ("C[:-1]", "C[-1]", True, True):
        raise ContractError("causal boundary identity proof is absent or failed")
    envelope = value.get("exact_cache_envelope")
    if not isinstance(envelope, Mapping) or (
        envelope.get("layers"),
        envelope.get("batch"),
        envelope.get("kv_heads"),
        envelope.get("head_dim"),
        envelope.get("positions"),
        envelope.get("key_space"),
        envelope.get("value_space"),
        envelope.get("finite_checked"),
        envelope.get("native_mapped_shape_dtype_device_equal"),
    ) != (
        40,
        1,
        8,
        128,
        "contiguous_zero_based_C_prefix",
        "post_target_qwen3_rope_cache",
        "target_cache_value_space",
        True,
        True,
    ):
        raise ContractError("exact cache-envelope evidence drifted")
    if envelope.get("prefix_length") != value.get("prompt_token_count") - 1:
        raise ContractError("cache prefix length does not equal |C|-1")
    context_sha = envelope.get("context_sha256")
    if not _valid_sha(context_sha):
        raise ContractError("cache context commitment is malformed")
    expected_ids = [item["id"] for item in intervention_catalog()]
    paired = value.get("paired")
    if not isinstance(paired, list) or [item.get("intervention_id") for item in paired] != expected_ids:
        raise ContractError("atomic paired metric intervention order changed")
    metric_keys = {
        "first_token_kl_native_to_candidate",
        "native_answer_margin",
        "candidate_answer_margin",
        "answer_margin_delta_candidate_minus_native",
        "native_answer_em",
        "candidate_answer_em",
        "answer_em_delta_candidate_minus_native",
        "native_answer_f1",
        "candidate_answer_f1",
        "answer_f1_delta_candidate_minus_native",
        "attention_output_cosine_uniform_layer_mean",
    }
    for item in paired:
        missing = metric_keys - set(item)
        if missing:
            raise ContractError(f"paired metric row is missing required metrics: {sorted(missing)}")
        if item.get("context_sha256") != context_sha:
            raise ContractError("paired metric row context commitment changed")
        if item["first_token_kl_native_to_candidate"] < -1e-9:
            raise ContractError("first-token KL is negative")
        for key in ("native_answer_em", "candidate_answer_em", "native_answer_f1", "candidate_answer_f1"):
            if not 0.0 <= item[key] <= 1.0:
                raise ContractError(f"paired metric {key} is outside [0,1]")
        if not -1.000001 <= item["attention_output_cosine_uniform_layer_mean"] <= 1.000001:
            raise ContractError("attention-output cosine is outside [-1,1]")
        layers = item.get("attention_layers")
        if not isinstance(layers, list) or [layer.get("layer") for layer in layers] != list(range(40)):
            raise ContractError("paired metric row lacks all 40 ordered attention layers")
        for layer in layers:
            if set(("attention_output_cosine", "attention_output_normalized_mse", "attention_probability_js")) - set(layer):
                raise ContractError("attention layer lacks cosine/NMSE/JS metrics")
            if not -1.000001 <= layer["attention_output_cosine"] <= 1.000001:
                raise ContractError("per-layer attention cosine is outside [-1,1]")
            if layer["attention_output_normalized_mse"] < 0 or layer["attention_probability_js"] < -1e-9:
                raise ContractError("per-layer attention divergence is negative")
    traces = value.get("intervention_generation")
    if not isinstance(traces, list) or [item.get("intervention_id") for item in traces] != expected_ids:
        raise ContractError("generation trace intervention order changed")
    for trace in traces:
        ids = trace.get("generated_token_ids")
        if not isinstance(ids, list) or any(type(item) is not int or item < 0 for item in ids):
            raise ContractError("generation trace has malformed token IDs")
        if trace.get("generated_token_ids_sha256") != canonical_sha256(ids):
            raise ContractError("generation token commitment changed")
        if trace.get("generated_token_count") != len(ids):
            raise ContractError("generation token count changed")
        if trace.get("cap") != 256 or trace.get("stop_reason") not in {"eos", "cap"}:
            raise ContractError("generation cap or stop policy changed")
        if trace["stop_reason"] == "cap" and len(ids) != 256:
            raise ContractError("cap stop reason disagrees with token count")
        if trace["stop_reason"] == "eos" and len(ids) > 255:
            raise ContractError("EOS trace cannot contain the full content-token cap")
        if not _valid_sha(trace.get("decoded_text_sha256")):
            raise ContractError("decoded-text commitment is malformed")
    compositions = value.get("compositions")
    if not isinstance(compositions, list) or [item.get("intervention_id") for item in compositions] != expected_ids:
        raise ContractError("cache-composition receipt order changed")
    for item, spec in zip(compositions, intervention_catalog(), strict=True):
        if item.get("cloned") is not True:
            raise ContractError("intervention cache branch was not isolated by cloning")
        if len(item.get("k_sources", [])) != 40 or len(item.get("v_sources", [])) != 40:
            raise ContractError("cache-composition receipt does not cover 40 layers")
        expected_k = []
        expected_v = []
        native_layers = set(spec["native_kv_layers"])
        for layer_index in range(40):
            expected_k.append("native" if layer_index in native_layers else spec["default_k"])
            expected_v.append("native" if layer_index in native_layers else spec["default_v"])
        if item["k_sources"] != expected_k or item["v_sources"] != expected_v:
            raise ContractError("cache-composition K/V sources changed")
        if item.get("context_sha256") != context_sha or not _valid_sha(item.get("mapper_manifest_sha256")):
            raise ContractError("cache-composition provenance commitment changed")
    assert_finite_tree(value)
    return value


def _validate_definition(
    definition_path: Path,
    *,
    config_path: Path,
    profile: str,
    expected_selected_sha256: str,
) -> dict[str, Any]:
    value = read_json(definition_path)
    if value.get("schema_version") != 1 or value.get("status") != "frozen_run_definition":
        raise ContractError("run definition is not frozen schema-v1")
    config_ref = value.get("config")
    if not isinstance(config_ref, Mapping) or verify_ref(config_ref, label="run config") != config_path:
        raise ContractError("run definition config identity changed")
    fingerprint = value.get("request_fingerprint")
    unsigned = dict(value)
    unsigned.pop("request_fingerprint", None)
    if not _valid_sha(fingerprint) or canonical_sha256(unsigned) != fingerprint:
        raise ContractError("run-definition fingerprint changed")
    if value.get("profile") != profile or value.get("selected_bindings_sha256") != expected_selected_sha256:
        raise ContractError("run definition profile or selected registry binding changed")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ContractError("run definition has no tokenized sample plans")
    required = {
        "dataset",
        "sample_id",
        "source_index",
        "stratum",
        "input_commitment_sha256",
        "prompt_token_ids_sha256",
        "prefix_token_ids_sha256",
        "serialized_prompt_sha256",
        "references_sha256",
        "gold_first_token_ids_sha256",
        "prompt_token_count",
    }
    for item in samples:
        if not isinstance(item, Mapping) or required - set(item):
            raise ContractError("run-definition sample binding is incomplete")
        for key in required - {"dataset", "sample_id", "source_index", "stratum", "prompt_token_count"}:
            if not _valid_sha(item[key]):
                raise ContractError(f"run-definition sample {key} is malformed")
        if isinstance(item["prompt_token_count"], bool) or not 2 <= item["prompt_token_count"] <= 8192:
            raise ContractError("run-definition prompt length is outside causal oracle bounds")
    return value


def validate_run(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Rebuild CPU gates, then verify every atomic result and final receipt."""

    config_path = ensure_formal_config_path(config_path)
    config = load_config(config_path)
    output = ensure_inside_output_root(config, output_dir)
    preflight, selected = contract_preflight(config_path, profile=profile)
    definition = _validate_definition(
        output / "run_definition.json",
        config_path=config_path,
        profile=profile,
        expected_selected_sha256=preflight["selected_bindings_sha256"],
    )
    samples = definition["samples"]
    expected_public = [item.public_binding() for item in selected]
    for plan, public in zip(samples, expected_public, strict=True):
        for key, expected in public.items():
            if plan.get(key) != expected:
                raise ContractError(f"tokenized plan changed registry binding {key}")
    fingerprint = definition["request_fingerprint"]
    inventory: list[dict[str, Any]] = []
    for plan in samples:
        path = result_path(output, plan)
        validate_atomic_result(path, request_fingerprint=fingerprint, plan_binding=plan)
        inventory.append(artifact_ref(path))
    receipt_path = output / "completion_receipt.json"
    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != 1 or receipt.get("status") != "complete":
        raise ContractError("completion receipt is not schema-v1 complete")
    if receipt.get("request_fingerprint") != fingerprint:
        raise ContractError("completion receipt request fingerprint changed")
    if receipt.get("atomic_prompt_results") != len(samples) or receipt.get("paired_metric_rows") != len(samples) * 11:
        raise ContractError("completion counts changed")
    if receipt.get("result_inventory") != inventory or receipt.get("result_inventory_sha256") != canonical_sha256(inventory):
        raise ContractError("completion result inventory changed")
    if (
        receipt.get("method_dev_train_only") is not True
        or receipt.get("locked_outcomes_opened") is not False
        or receipt.get("answers_used_for_selection") is not False
        or receipt.get("mapped_cache_materializations_per_prompt") != 1
    ):
        raise ContractError("completion data-isolation or single-materialization claim changed")
    idle_ref = receipt.get("final_session_idle_gate")
    if not isinstance(idle_ref, Mapping):
        raise ContractError("completion receipt has no final-session idle gate")
    idle_path = verify_ref(idle_ref, label="final-session GPU idle gate")
    try:
        idle_path.relative_to(output / "gpu_idle_gates")
    except ValueError as exc:
        raise ContractError("final-session idle gate escapes the run output") from exc
    idle = read_json(idle_path)
    if (
        idle.get("status") != "passed"
        or idle.get("limit_mib") != 64
        or idle.get("required_consecutive_samples") != 12
        or idle.get("observed_consecutive_samples", 0) < 12
        or idle.get("sample_interval_seconds") != 5
    ):
        raise ContractError("final-session GPU idle gate is not a 12-sample pass")
    window = idle.get("final_window")
    if not isinstance(window, list) or len(window) != 12:
        raise ContractError("final-session GPU idle window is incomplete")
    for row in window:
        if not isinstance(row, Mapping):
            raise ContractError("final-session GPU idle sample is malformed")
        memory = row.get("memory_used_mib")
        if row.get("passed") is not True or not isinstance(memory, list) or len(memory) != 8:
            raise ContractError("final-session GPU idle sample is malformed")
        if any(isinstance(value, bool) or not isinstance(value, int) or value > 64 for value in memory):
            raise ContractError("final-session GPU idle sample exceeds the memory limit")
    identities = (output / "deep_input_identity.json", output / "runtime_provenance.json")
    identity_refs = receipt.get("run_artifacts")
    if not isinstance(identity_refs, Mapping) or set(identity_refs) != {"deep_input_identity", "runtime_provenance", "run_definition"}:
        raise ContractError("completion run-artifact inventory changed")
    expected_run_refs = {
        "deep_input_identity": artifact_ref(identities[0]),
        "runtime_provenance": artifact_ref(identities[1]),
        "run_definition": artifact_ref(output / "run_definition.json"),
    }
    if identity_refs != expected_run_refs:
        raise ContractError("completion run-artifact hashes changed")
    return {
        "schema_version": 1,
        "status": "passed_cpu_receipt_validation",
        "gpu_used": False,
        "cuda_api_called": False,
        "profile": profile,
        "prompts": len(samples),
        "paired_metric_rows": len(samples) * 11,
        "request_fingerprint": fingerprint,
        "completion_receipt_sha256": sha256_file(receipt_path),
    }


__all__ = ["result_path", "validate_atomic_result", "validate_run"]
