import pytest

from src.contracts import ContractError
from src.metrics import score_answer
from src.promotion import evaluate_screen_promotion
from src.statistics import paired_bootstrap_mean_delta


def _passing_summary():
    datasets = ("hotpotqa_train", "musique_train")
    aggregates = {}
    comparisons = {}
    for dataset in (*datasets, "pooled"):
        aggregates[dataset] = {
            "nvidia_k20": {"cap_rate": 0.1, "effective_termination_rate": 0.9},
            "two_pass_r1_then_r32": {"cap_rate": 0.05, "effective_termination_rate": 0.95},
        }
        comparisons[dataset] = {
            "two_pass_r1_then_r32": {
                "official_answer_f1": {"mean_delta": 0.01},
                "r1_correct_em_retention": {"rate": 1.0},
            }
        }
    return {
        "status": "complete",
        "profile": "screen",
        "paired_complete": True,
        "failed_or_missing_cells": 0,
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
    }


def test_paired_bootstrap_is_deterministic_and_paired():
    first = paired_bootstrap_mean_delta([1.0, 0.0, 0.5], [0.0, 0.0, 0.0], seed=7, replicates=100, confidence_level=0.95)
    second = paired_bootstrap_mean_delta([1.0, 0.0, 0.5], [0.0, 0.0, 0.0], seed=7, replicates=100, confidence_level=0.95)
    assert first == second
    assert first["mean_delta"] == 0.5
    assert first["n"] == 3


def test_screen_gate_passes_only_all_registered_checks():
    decision = evaluate_screen_promotion(_passing_summary())
    assert decision["status"] == "pass"
    assert decision["authorizes_confirm"] is True
    assert decision["authorizes_locked_test_or_paper_superiority_claim"] is False


def test_screen_gate_fails_on_either_dataset_even_if_pooled_positive():
    summary = _passing_summary()
    summary["paired_comparisons"]["hotpotqa_train"]["two_pass_r1_then_r32"]["official_answer_f1"]["mean_delta"] = -0.0001
    decision = evaluate_screen_promotion(summary)
    assert decision["status"] == "fail"
    assert decision["authorizes_confirm"] is False


def test_smoke_cannot_be_used_as_screen_gate():
    summary = _passing_summary()
    summary["profile"] = "smoke"
    with pytest.raises(ContractError, match="only accepts"):
        evaluate_screen_promotion(summary)


def test_pinned_hotpot_and_musique_normalized_empty_semantics_differ():
    assert score_answer("hotpotqa_train", "the", ["the"])["official_answer_f1"] == 0.0
    assert score_answer("musique_train", "the", ["the"])["official_answer_f1"] == 1.0
