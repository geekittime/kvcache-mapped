"""Gold-blind selector over *complete* R1 and R32 raw results.

This is deliberately not the online early-abort watchdog from experiment 11.
R1 must have reached an ordinary registered termination first.  The selector
then reads only ``generated_tokens`` and ``cap_hit``; if triggered, it chooses
the independently complete R32 result.
"""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import ContractError, canonical_sha256, finite_float


SELECTOR_ID = "complete_r1_then_gold_blind_choose_r32_v2"


def should_choose_r32(generation: Mapping[str, Any], *, threshold: int = 32) -> bool:
    if type(threshold) is not int or threshold != 32:
        raise ContractError("selector threshold is frozen at 32")
    if not isinstance(generation, Mapping):
        raise ContractError("R1 generation evidence is missing")
    count = generation.get("generated_tokens")
    cap_hit = generation.get("cap_hit")
    stop_reason = generation.get("stop_reason")
    if type(count) is not int or count < 0 or type(cap_hit) is not bool:
        raise ContractError("R1 selector inputs are malformed")
    if stop_reason not in {"eos", "stop_sequence", "max_tokens"}:
        raise ContractError("R1 is not a complete registered run")
    if (stop_reason == "max_tokens") != cap_hit:
        raise ContractError("R1 cap/termination evidence is inconsistent")
    return count > threshold or cap_hit


def derive_two_pass_result(
    r1: Mapping[str, Any],
    r32: Mapping[str, Any],
    *,
    r1_file_sha256: str,
    r32_file_sha256: str,
) -> dict[str, Any]:
    required_equal = (
        "request_fingerprint",
        "dataset_id",
        "sample_id",
        "sample_binding_sha256",
        "source_projection_sha256",
        "prompt_token_ids_sha256",
        "references",
        "truncation",
    )
    for field in required_equal:
        if r1.get(field) != r32.get(field):
            raise ContractError(f"R1/R32 pairing differs at {field}")
    if r1.get("status") != "complete" or r32.get("status") != "complete":
        raise ContractError("two-pass selector requires two complete raw runs")
    if r1.get("mode") != "nvidia_k20" or r32.get("mode") != "tail_refresh_r32":
        raise ContractError("two-pass selector received wrong raw modes")
    trigger = should_choose_r32(r1.get("generation", {}))
    selected = r32 if trigger else r1
    selected_mode = "tail_refresh_r32" if trigger else "nvidia_k20"
    r1_wall = finite_float(r1.get("runtime_cost", {}).get("wall_seconds"), label="R1 wall")
    r32_wall = finite_float(r32.get("runtime_cost", {}).get("wall_seconds"), label="R32 wall")
    result = {
        "schema_version": 2,
        "status": "complete",
        "result_kind": "derived_gold_blind_selector",
        "request_fingerprint": r1["request_fingerprint"],
        "dataset_id": r1["dataset_id"],
        "sample_id": r1["sample_id"],
        "sample_binding_sha256": r1["sample_binding_sha256"],
        "source_projection_sha256": r1["source_projection_sha256"],
        "prompt_token_ids_sha256": r1["prompt_token_ids_sha256"],
        "mode": "two_pass_r1_then_r32",
        "references": r1["references"],
        "truncation": r1["truncation"],
        "prediction": selected["prediction"],
        "generation": selected["generation"],
        "metrics": selected["metrics"],
        "runtime_cost": {
            "selected_raw_wall_seconds": finite_float(
                selected["runtime_cost"]["wall_seconds"], label="selected wall"
            ),
            "sequential_policy_wall_seconds_proxy": r1_wall + (r32_wall if trigger else 0.0),
            "proxy_only_from_separate_complete_raw_runs": True,
            "online_end_to_end_latency_measured": False,
            "peak_gpu_allocated_bytes_proxy": max(
                max(r1["runtime_cost"]["peak_gpu_allocated_bytes"].values()),
                max(r32["runtime_cost"]["peak_gpu_allocated_bytes"].values()) if trigger else 0,
            ),
        },
        "selector_audit": {
            "selector_id": SELECTOR_ID,
            "threshold_generated_tokens": 32,
            "selection_inputs": {
                "r1_generated_tokens": r1["generation"]["generated_tokens"],
                "r1_cap_hit": r1["generation"]["cap_hit"],
            },
            "triggered": trigger,
            "selected_mode": selected_mode,
            "r1_was_complete_before_selection": True,
            "r32_raw_was_complete_before_derivation": True,
            "gold_or_metric_read_for_selection": False,
            "dataset_read_for_selection": False,
            "online_watchdog": False,
            "execution_semantics": "offline_two_pass_selector_over_complete_raw_runs",
            "r1_raw_file_sha256": r1_file_sha256,
            "r32_raw_file_sha256": r32_file_sha256,
            "selected_result_content_sha256": canonical_sha256(selected),
        },
    }
    result["derived_binding_sha256"] = canonical_sha256(result)
    return result


def validate_derived_result(
    value: Mapping[str, Any],
    r1: Mapping[str, Any],
    r32: Mapping[str, Any],
    *,
    r1_file_sha256: str,
    r32_file_sha256: str,
) -> None:
    expected = derive_two_pass_result(
        r1,
        r32,
        r1_file_sha256=r1_file_sha256,
        r32_file_sha256=r32_file_sha256,
    )
    if dict(value) != expected:
        raise ContractError("derived two-pass checkpoint does not reproduce exactly")


__all__ = ["SELECTOR_ID", "derive_two_pass_result", "should_choose_r32", "validate_derived_result"]
