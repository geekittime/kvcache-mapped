from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import analyze


def audit(prompt: int, r_value: int) -> dict:
    prefix = prompt - 1
    mapped = prompt - r_value
    value = {
        "full_prompt_tokens": prompt,
        "builder_prefix_tokens": prefix,
        "requested_total_native_prompt_tokens": r_value,
        "builder_native_tail_tokens": r_value - 1,
        "engine_boundary_tokens": 1,
        "engine_boundary_position_id": prefix,
        "mapped_prefix_tokens": mapped,
        "mapped_token_slice": [0, mapped],
        "native_tail_token_slice": [mapped, prefix],
        "cache_length_before_refresh": mapped,
        "cache_length_after_refresh": prefix,
        "native_tail_position_ids": list(range(mapped, prefix)),
        "target_forward_calls": 0 if r_value == 1 else 1,
        "target_forward_query_tokens": r_value - 1,
        "r1_returned_underlying_cache_object_directly": r_value == 1,
    }
    if r_value > 1:
        value.update(
            attention_mask_shape=[1, prefix],
            cache_position=list(range(mapped, prefix)),
            position_ids=list(range(mapped, prefix)),
        )
    return value


@pytest.mark.parametrize("mode,r_value", [("mapped_r1", 1), ("tail_refresh_r8", 8), ("tail_refresh_r32", 32)])
def test_exact_tail_contract(mode: str, r_value: int) -> None:
    analyze.validate_tail_audit(audit(100, r_value), 100, mode)


def test_tail_contract_rejects_off_by_one() -> None:
    value = audit(100, 8)
    value["engine_boundary_position_id"] -= 1
    with pytest.raises(analyze.ValidationError):
        analyze.validate_tail_audit(value, 100, "tail_refresh_r8")


def test_r1_rejects_hidden_forward() -> None:
    value = audit(100, 1)
    value["target_forward_calls"] = 1
    with pytest.raises(analyze.ValidationError):
        analyze.validate_tail_audit(value, 100, "mapped_r1")


def test_bootstrap_is_deterministic_and_paired() -> None:
    first = analyze.paired_bootstrap_delta([1.0, 0.0], [0.0, 0.0], "unit")
    second = analyze.paired_bootstrap_delta([1.0, 0.0], [0.0, 0.0], "unit")
    assert first == second
    assert first["estimate"] == 0.5
    assert first["resampling_unit"] == "paired_sample_id_cluster"


def test_gap_closure_undefined_if_r1_beats_native() -> None:
    assert analyze.gap_closure_bootstrap([1, 1], [1, 1], [0, 0], "unit") is None


def test_transition_table_is_exhaustive() -> None:
    def item(em: float) -> dict:
        return {"metrics": {"relaxed_qa_exact_match": em}}

    observed = analyze.transitions([item(1), item(1), item(0), item(0)], [item(0), item(1), item(1), item(0)])
    assert observed == {"correct_to_wrong": 1, "wrong_to_correct": 1, "correct_to_correct": 1, "wrong_to_wrong": 1}


def test_pinned_receipt_if_present() -> None:
    path = ROOT / "outputs/validation_receipt.json"
    if not path.exists():
        pytest.skip("receipt is written after helper tests")
    value = analyze.read_json(path)
    assert value["status"] == "passed_strict_cpu_validation"
    assert value["validated_atomic_results"] == 384
    assert value["failures"] == 0
