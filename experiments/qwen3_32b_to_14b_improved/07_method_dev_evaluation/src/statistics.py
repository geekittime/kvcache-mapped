"""Paired method-dev aggregation, bootstrap intervals, and fidelity retention."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from .contracts import ContractError, DATASETS, MODES, finite_float


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ContractError("quantile requires non-empty values")
    if not 0.0 <= probability <= 1.0:
        raise ContractError("quantile probability is outside [0,1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def paired_bootstrap_mean_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    seed: int,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ContractError("paired bootstrap requires equal non-empty samples")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ContractError("bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ContractError("confidence level must be in (0,1)")
    deltas = [finite_float(a, label="left") - finite_float(b, label="right") for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    count = len(deltas)
    bootstrap: list[float] = []
    for _ in range(replicates):
        bootstrap.append(sum(deltas[rng.randrange(count)] for _ in range(count)) / count)
    bootstrap.sort()
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "n": count,
        "mean_delta": sum(deltas) / count,
        "confidence_level": confidence_level,
        "ci_low": _quantile(bootstrap, alpha),
        "ci_high": _quantile(bootstrap, 1.0 - alpha),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "method": "paired_percentile_bootstrap",
    }


def _derived_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{base}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _metric(record: Mapping[str, Any], name: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ContractError("result lacks metrics")
    value = finite_float(metrics.get(name), label=f"metrics.{name}")
    if not 0.0 <= value <= 1.0:
        raise ContractError(f"metrics.{name} is outside [0,1]")
    return value


def _mode_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ContractError("mode summary cannot be empty")
    generations = [record.get("generation") for record in records]
    if any(not isinstance(value, Mapping) for value in generations):
        raise ContractError("result lacks generation evidence")
    stop_counts = {
        reason: sum(value.get("stop_reason") == reason for value in generations)
        for reason in ("eos", "stop_sequence", "max_tokens")
    }
    count = len(records)
    return {
        "count": count,
        "official_answer_f1": sum(_metric(record, "official_answer_f1") for record in records) / count,
        "official_answer_em": sum(_metric(record, "official_answer_em") for record in records) / count,
        "mean_generated_tokens": sum(int(value.get("generated_tokens", -1)) for value in generations) / count,
        "cap_rate": sum(value.get("cap_hit") is True for value in generations) / count,
        "stop_counts": stop_counts,
        "stop_rates": {reason: value / count for reason, value in stop_counts.items()},
        "prompt_truncation_rate": sum(bool(record.get("truncation", {}).get("truncated")) for record in records) / count,
    }


def _flip_table(
    candidate: Sequence[float], reference: Sequence[float]
) -> dict[str, int]:
    table = {"both_correct": 0, "reference_only": 0, "candidate_only": 0, "both_wrong": 0}
    for candidate_value, reference_value in zip(candidate, reference, strict=True):
        candidate_correct = candidate_value == 1.0
        reference_correct = reference_value == 1.0
        if candidate_correct and reference_correct:
            table["both_correct"] += 1
        elif reference_correct:
            table["reference_only"] += 1
        elif candidate_correct:
            table["candidate_only"] += 1
        else:
            table["both_wrong"] += 1
    return table


def _paired_comparison(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    *,
    dataset_id: str,
    mode: str,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("official_answer_f1", "official_answer_em"):
        candidate_values = [_metric(record, metric) for record in candidate]
        baseline_values = [_metric(record, metric) for record in baseline]
        target_values = [_metric(record, metric) for record in target]
        candidate_mean = sum(candidate_values) / len(candidate_values)
        baseline_mean = sum(baseline_values) / len(baseline_values)
        target_mean = sum(target_values) / len(target_values)
        label = f"{dataset_id}/{mode}/{metric}/vs_k20"
        comparison = paired_bootstrap_mean_delta(
            candidate_values,
            baseline_values,
            seed=_derived_seed(bootstrap_seed, label),
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        comparison["delta_vs_native14"] = candidate_mean - target_mean
        comparison["raw_score_retention_percent"] = (
            100.0 * candidate_mean / target_mean if target_mean > 0.0 else None
        )
        comparison["gap_closure_vs_k20"] = (
            (candidate_mean - baseline_mean) / (target_mean - baseline_mean)
            if target_mean > baseline_mean
            else None
        )
        result[metric] = comparison

    candidate_em = [_metric(record, "official_answer_em") for record in candidate]
    baseline_em = [_metric(record, "official_answer_em") for record in baseline]
    target_em = [_metric(record, "official_answer_em") for record in target]
    target_correct = sum(value == 1.0 for value in target_em)
    result["exact_match_fidelity"] = {
        "target_correct_count": target_correct,
        "target_correct_retention": (
            sum(c == 1.0 and t == 1.0 for c, t in zip(candidate_em, target_em, strict=True)) / target_correct
            if target_correct
            else None
        ),
        "candidate_vs_k20_2x2": _flip_table(candidate_em, baseline_em),
        "candidate_vs_native14_2x2": _flip_table(candidate_em, target_em),
    }
    candidate_f1 = [_metric(record, "official_answer_f1") for record in candidate]
    baseline_f1 = [_metric(record, "official_answer_f1") for record in baseline]
    result["f1_pairwise_direction"] = {
        "wins_vs_k20": sum(c > b for c, b in zip(candidate_f1, baseline_f1, strict=True)),
        "ties_vs_k20": sum(c == b for c, b in zip(candidate_f1, baseline_f1, strict=True)),
        "losses_vs_k20": sum(c < b for c, b in zip(candidate_f1, baseline_f1, strict=True)),
    }
    return result


def aggregate_paired_results(
    records: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    request_fingerprint: str,
    bootstrap_seed: int = 20260830,
    bootstrap_replicates: int = 10_000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if not records:
        raise ContractError("cannot aggregate an empty run")
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    sample_ids: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    for record in records:
        if record.get("request_fingerprint") != request_fingerprint:
            raise ContractError("result request fingerprint mismatch")
        dataset = str(record.get("dataset_id"))
        sample_id = str(record.get("sample_id"))
        mode = str(record.get("mode"))
        if dataset not in DATASETS or mode not in MODES or not sample_id:
            raise ContractError("result dataset/sample/mode is malformed")
        key = (dataset, sample_id, mode)
        if key in by_key:
            raise ContractError(f"duplicate result: {key}")
        by_key[key] = record
        sample_ids[dataset].add(sample_id)
    missing: list[dict[str, str]] = []
    for dataset, ids in sample_ids.items():
        for sample_id in sorted(ids):
            for mode in MODES:
                if (dataset, sample_id, mode) not in by_key:
                    missing.append({"dataset_id": dataset, "sample_id": sample_id, "mode": mode})
    if missing:
        raise ContractError(f"paired aggregation has {len(missing)} missing cells")

    aggregates: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for dataset in DATASETS:
        ordered_ids = sorted(sample_ids[dataset])
        if not ordered_ids:
            raise ContractError(f"paired aggregation lacks {dataset}")
        by_mode = {
            mode: [by_key[(dataset, sample_id, mode)] for sample_id in ordered_ids]
            for mode in MODES
        }
        aggregates[dataset] = {
            mode: _mode_summary(mode_records) for mode, mode_records in by_mode.items()
        }
        target = by_mode["native14"]
        baseline = by_mode["nvidia_k20"]
        paired[dataset] = {
            mode: _paired_comparison(
                by_mode[mode],
                baseline,
                target,
                dataset_id=dataset,
                mode=mode,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
            )
            for mode in ("shrinkage", "tail_refresh_r8", "tail_refresh_r32")
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "profile": profile,
        "request_fingerprint": request_fingerprint,
        "paired_complete": True,
        "resampling_unit": "complete_question_record",
        "bootstrap": {
            "base_seed": bootstrap_seed,
            "replicates": bootstrap_replicates,
            "confidence_level": confidence_level,
            "method": "paired_percentile_bootstrap",
        },
        "aggregates": aggregates,
        "paired_comparisons": paired,
        "claim_boundary": "method-dev only; no official-dev or locked-test superiority claim",
    }


__all__ = ["aggregate_paired_results", "paired_bootstrap_mean_delta"]
