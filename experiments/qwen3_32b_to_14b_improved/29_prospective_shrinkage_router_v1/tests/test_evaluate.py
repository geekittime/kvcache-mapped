from __future__ import annotations

import pytest

from src.evaluate import ArmMetrics, PairedOutcome, evaluate_promotion, arm_means, paired_bootstrap_ci


def metrics(f1: float, em: float, *, cap: bool = False, complete: bool = True) -> ArmMetrics:
    return ArmMetrics(f1=f1, em=em, cap_hit=cap, complete=complete)


def rows(gain: float = 0.10, *, route_all: bool = False) -> tuple[PairedOutcome, ...]:
    result = []
    for dataset in ("hotpotqa_train", "musique_train"):
        for index in range(4):
            base = 0.50 + 0.02 * index
            result.append(
                PairedOutcome(
                    sample_id=f"{dataset}-{index}",
                    dataset_id=dataset,
                    split="sealed_eval",
                    route_action="shrinkage_r1" if route_all or index == 0 else "nvidia_k20",
                    route_no_go=False,
                    native14=metrics(base + 0.15, 0.70),
                    nvidia_k20=metrics(base, 0.50),
                    shrinkage_r1=metrics(base + gain, 0.50),
                )
            )
    return tuple(result)


def test_arm_means_and_deterministic_bootstrap() -> None:
    data = rows()
    means = arm_means(data)
    assert means["native14"]["f1"] > means["nvidia_k20"]["f1"]
    first = paired_bootstrap_ci(data, lambda row: row.shrinkage_r1.f1 - row.nvidia_k20.f1, replicates=200, seed=7)
    second = paired_bootstrap_ci(data, lambda row: row.shrinkage_r1.f1 - row.nvidia_k20.f1, replicates=200, seed=7)
    assert first == second
    assert first.point == pytest.approx(0.10)


def test_promotion_report_requires_positive_ci() -> None:
    report = evaluate_promotion(rows(route_all=True), budget_limit=8, bootstrap_replicates=200, bootstrap_seed=7)
    assert report.passed is True
    assert report.overall_f1_delta == pytest.approx(0.10)
    assert report.f1_ci.lower > 0.0
    assert set(report.arm_means) == {"native14", "nvidia_k20", "shrinkage_r1"}


def test_hotpot_safety_gate_and_budget() -> None:
    bad = list(rows(gain=-0.10))
    # Route the first Hotpot row to shrinkage, causing a safety failure.
    bad[0] = PairedOutcome(
        sample_id=bad[0].sample_id,
        dataset_id=bad[0].dataset_id,
        split=bad[0].split,
        route_action="shrinkage_r1",
        route_no_go=False,
        native14=bad[0].native14,
        nvidia_k20=bad[0].nvidia_k20,
        shrinkage_r1=bad[0].shrinkage_r1,
    )
    report = evaluate_promotion(tuple(bad), budget_limit=2, bootstrap_replicates=200, bootstrap_seed=7)
    assert report.passed is False
    assert "f1_regression:hotpotqa_train" in report.reasons
    assert "hotpot_f1_safety" in report.reasons

    with pytest.raises(ValueError, match="exceed"):
        evaluate_promotion(rows(), budget_limit=0, bootstrap_replicates=200)


def test_invalid_rows_fail_closed() -> None:
    with pytest.raises(ValueError):
        PairedOutcome(
            sample_id="x",
            dataset_id="hotpotqa_train",
            split="calibration",
            route_action="nvidia_k20",
            route_no_go=False,
            native14=metrics(0.5, 0.5),
            nvidia_k20=metrics(0.5, 0.5),
            shrinkage_r1=metrics(0.5, 0.5),
        )
    with pytest.raises(ValueError, match="exactly HotpotQA"):
        evaluate_promotion(tuple(rows()[:4]), bootstrap_replicates=200)
