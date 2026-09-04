"""Pre-registered single screen go/no-go and confirm evidence rules."""

from __future__ import annotations

from typing import Any, Mapping

from .configuration import STATIC_PROMOTION
from .contracts import ContractError, DATASETS, canonical_sha256, finite_float


def evaluate_screen_promotion(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("profile") != "screen" or summary.get("status") != "complete":
        raise ContractError("promotion gate only accepts a complete screen summary")
    if summary.get("failed_or_missing_cells") != 0 or summary.get("paired_complete") is not True:
        raise ContractError("screen is incomplete")
    comparisons = summary.get("paired_comparisons")
    aggregates = summary.get("aggregates")
    if not isinstance(comparisons, Mapping) or not isinstance(aggregates, Mapping):
        raise ContractError("screen summary lacks paired evidence")
    candidate = STATIC_PROMOTION["candidate_mode"]
    baseline = STATIC_PROMOTION["baseline_mode"]
    checks: list[dict[str, Any]] = []
    for dataset in DATASETS:
        try:
            comparison = comparisons[dataset][candidate]
            candidate_aggregate = aggregates[dataset][candidate]
            baseline_aggregate = aggregates[dataset][baseline]
        except (KeyError, TypeError) as exc:
            raise ContractError(f"screen summary lacks gate evidence for {dataset}") from exc
        delta = finite_float(
            comparison["official_answer_f1"]["mean_delta"], label=f"{dataset} F1 delta"
        )
        checks.append(
            {
                "check": f"{dataset}.f1_delta_vs_nvidia_k20_ge_0",
                "observed": delta,
                "threshold": 0.0,
                "pass": delta >= 0.0,
            }
        )
        candidate_cap = finite_float(candidate_aggregate["cap_rate"], label="candidate cap")
        baseline_cap = finite_float(baseline_aggregate["cap_rate"], label="baseline cap")
        checks.append(
            {
                "check": f"{dataset}.cap_rate_not_increased",
                "observed_candidate": candidate_cap,
                "observed_baseline": baseline_cap,
                "pass": candidate_cap <= baseline_cap,
            }
        )
        candidate_termination = finite_float(
            candidate_aggregate["effective_termination_rate"], label="candidate termination"
        )
        baseline_termination = finite_float(
            baseline_aggregate["effective_termination_rate"], label="baseline termination"
        )
        checks.append(
            {
                "check": f"{dataset}.effective_termination_not_decreased",
                "observed_candidate": candidate_termination,
                "observed_baseline": baseline_termination,
                "pass": candidate_termination >= baseline_termination,
            }
        )
        retention = comparison["r1_correct_em_retention"]["rate"]
        retention_value = finite_float(retention, label=f"{dataset} R1-correct retention") if retention is not None else -1.0
        checks.append(
            {
                "check": f"{dataset}.r1_correct_em_retention_ge_0.98",
                "observed": retention if retention is not None else None,
                "threshold": 0.98,
                "pass": retention is not None and retention_value >= 0.98,
            }
        )
    pooled_delta = finite_float(
        comparisons["pooled"][candidate]["official_answer_f1"]["mean_delta"],
        label="pooled F1 delta",
    )
    checks.append(
        {
            "check": "pooled.f1_delta_vs_nvidia_k20_gt_0",
            "observed": pooled_delta,
            "threshold": 0.0,
            "strict": True,
            "pass": pooled_delta > 0.0,
        }
    )
    checks.append(
        {
            "check": "paired_zero_failed_or_missing_cells",
            "observed": summary["failed_or_missing_cells"],
            "threshold": 0,
            "pass": summary["failed_or_missing_cells"] == 0,
        }
    )
    passed = all(check["pass"] is True for check in checks)
    return {
        "schema_version": 2,
        "gate_rule": STATIC_PROMOTION,
        "gate_rule_sha256": canonical_sha256(STATIC_PROMOTION),
        "profile": "screen",
        "candidate_mode": candidate,
        "baseline_mode": baseline,
        "status": "pass" if passed else "fail",
        "all_checks_pass": passed,
        "checks": checks,
        "screen_is_single_go_no_go": True,
        "threshold_tuning_after_outcomes_forbidden": True,
        "authorizes_confirm": passed,
        "authorizes_locked_test_or_paper_superiority_claim": False,
    }


def evaluate_confirm_evidence(summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("profile") != "confirm" or summary.get("status") != "complete":
        raise ContractError("confirm evidence requires a complete confirm summary")
    candidate = STATIC_PROMOTION["candidate_mode"]
    comparison = summary["paired_comparisons"]["pooled"][candidate]["official_answer_f1"]
    ci_low = finite_float(comparison["ci_low"], label="confirm pooled F1 CI low")
    mean = finite_float(comparison["mean_delta"], label="confirm pooled F1 delta")
    per_dataset_nonnegative = all(
        finite_float(
            summary["paired_comparisons"][dataset][candidate]["official_answer_f1"]["mean_delta"],
            label=f"{dataset} confirm F1 delta",
        )
        >= 0.0
        for dataset in DATASETS
    )
    supported = ci_low > 0.0 and mean > 0.0 and per_dataset_nonnegative
    return {
        "schema_version": 2,
        "profile": "confirm",
        "candidate_mode": candidate,
        "baseline_mode": "nvidia_k20",
        "pooled_f1_mean_delta": mean,
        "pooled_f1_ci_low": ci_low,
        "per_dataset_f1_nonnegative": per_dataset_nonnegative,
        "confirm_superiority_supported_within_method_dev": supported,
        "authorizes_locked_test_or_paper_superiority_claim": False,
    }


__all__ = ["evaluate_confirm_evidence", "evaluate_screen_promotion"]
