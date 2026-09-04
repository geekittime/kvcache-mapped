from __future__ import annotations

import torch

from scripts.oracle_smoke import (
    compare_last_hidden,
    compare_last_logits,
    diagnostic_context_lengths,
    target_local_case_passed,
)


def test_logit_metrics_report_relative_distribution_and_rank_controls() -> None:
    reference = torch.tensor([[[4.0, 2.0, 1.0, -1.0, -3.0]]])
    candidate = torch.tensor([[[3.95, 2.02, 1.01, -0.98, -3.2]]])
    metrics = compare_last_logits(reference, candidate)
    assert metrics["top1_equal"] is True
    assert metrics["minimum_top5_overlap"] == 5
    assert 0.0 < metrics["relative_l2_logit"] < 0.1
    assert 0.0 < metrics["kl_reference_to_candidate"] < 0.01
    assert 0.0 < metrics["total_variation_probability"] < 0.05


def test_target_local_gate_rejects_rank_flip_despite_small_vector_error() -> None:
    reference = torch.zeros((1, 1, 1000))
    candidate = reference.clone()
    reference[..., 0] = 1.0
    reference[..., 1] = 0.999
    candidate[..., 0] = 0.999
    candidate[..., 1] = 1.0
    replay = compare_last_logits(reference, candidate)
    hidden = compare_last_hidden(
        torch.ones((1, 1, 8)), torch.ones((1, 1, 8))
    )
    assert replay["relative_l2_logit"] < 0.01
    assert replay["top1_equal"] is False
    assert target_local_case_passed(replay, hidden) is False


def test_target_local_gate_does_not_use_one_tail_logit_max_abs() -> None:
    reference = torch.full((1, 1, 200_000), -20.0)
    candidate = reference.clone()
    reference[..., 0] = 8.0
    candidate[..., 0] = 8.0
    candidate[..., -1] = -19.0  # Large raw delta on negligible probability mass.
    replay = compare_last_logits(reference, candidate)
    hidden = compare_last_hidden(
        torch.ones((1, 1, 8)), torch.ones((1, 1, 8))
    )
    assert replay["max_abs_logit"] == 1.0
    assert replay["top1_equal"] is True
    assert target_local_case_passed(replay, hidden) is True


def test_diagnostic_context_lengths_include_short_cases_and_full_length() -> None:
    assert diagnostic_context_lengths(30) == (2, 4, 8, 15, 30)
    assert diagnostic_context_lengths(5) == (2, 4, 5)
