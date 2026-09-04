"""Paired aggregation, bootstrap intervals, retention, and system diagnostics."""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Mapping, Sequence

from .contracts import ALL_MODES, ContractError, DATASETS, PROFILES, finite_float


CANDIDATE_MODES = ("shrinkage", "tail_refresh_r32", "two_pass_r1_then_r32")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ContractError("quantile requires non-empty values")
    if not 0.0 <= probability <= 1.0:
        raise ContractError("quantile probability is outside [0,1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _derived_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def paired_bootstrap_mean_delta(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int,
    replicates: int,
    confidence_level: float,
) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate:
        raise ContractError("paired bootstrap requires equal non-empty samples")
    if type(replicates) is not int or replicates <= 0:
        raise ContractError("bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ContractError("confidence level must be in (0,1)")
    deltas = [
        finite_float(left, label="candidate") - finite_float(right, label="baseline")
        for left, right in zip(candidate, baseline, strict=True)
    ]
    rng = random.Random(seed)
    count = len(deltas)
    draws = [sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(replicates)]
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "n": count,
        "mean_delta": sum(deltas) / count,
        "confidence_level": confidence_level,
        "ci_low": _quantile(draws, alpha),
        "ci_high": _quantile(draws, 1.0 - alpha),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "method": "paired_percentile_bootstrap",
    }


def _metric(record: Mapping[str, Any], metric: str) -> float:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ContractError("result lacks metrics")
    value = finite_float(metrics.get(metric), label=f"metrics.{metric}")
    if not 0.0 <= value <= 1.0:
        raise ContractError(f"metrics.{metric} is outside [0,1]")
    return value


def _wall(record: Mapping[str, Any]) -> float:
    cost = record.get("runtime_cost")
    if not isinstance(cost, Mapping):
        raise ContractError("result lacks runtime cost")
    field = "sequential_policy_wall_seconds_proxy" if record.get("mode") == "two_pass_r1_then_r32" else "wall_seconds"
    return finite_float(cost.get(field), label=f"runtime_cost.{field}")


def _mode_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ContractError("mode summary is empty")
    generations = [record.get("generation") for record in records]
    if any(not isinstance(item, Mapping) for item in generations):
        raise ContractError("result lacks generation evidence")
    count = len(records)
    stop_counts = {
        reason: sum(generation.get("stop_reason") == reason for generation in generations)
        for reason in ("eos", "stop_sequence", "max_tokens")
    }
    walls = [_wall(record) for record in records]
    result = {
        "count": count,
        "official_answer_f1": sum(_metric(record, "official_answer_f1") for record in records) / count,
        "official_answer_em": sum(_metric(record, "official_answer_em") for record in records) / count,
        "mean_generated_tokens": sum(int(item.get("generated_tokens", -1)) for item in generations) / count,
        "cap_rate": stop_counts["max_tokens"] / count,
        "effective_termination_rate": (stop_counts["eos"] + stop_counts["stop_sequence"]) / count,
        "stop_counts": stop_counts,
        "mean_wall_seconds": sum(walls) / count,
        "p95_wall_seconds": _quantile(walls, 0.95),
        "prompt_truncation_rate": sum(
            bool(record.get("truncation", {}).get("truncated")) for record in records
        ) / count,
    }
    if records[0].get("mode") == "two_pass_r1_then_r32":
        audits = [record.get("selector_audit") for record in records]
        if any(not isinstance(item, Mapping) for item in audits):
            raise ContractError("derived result lacks selector audit")
        result["selector_trigger_rate"] = sum(item.get("triggered") is True for item in audits) / count
        result["online_end_to_end_latency_measured"] = False
    return result


def _comparison(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    *,
    label: str,
    seed: int,
    replicates: int,
    confidence_level: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("official_answer_f1", "official_answer_em"):
        candidate_values = [_metric(record, metric) for record in candidate]
        baseline_values = [_metric(record, metric) for record in baseline]
        target_values = [_metric(record, metric) for record in target]
        paired = paired_bootstrap_mean_delta(
            candidate_values,
            baseline_values,
            seed=_derived_seed(seed, f"{label}/{metric}"),
            replicates=replicates,
            confidence_level=confidence_level,
        )
        candidate_mean = sum(candidate_values) / len(candidate_values)
        baseline_mean = sum(baseline_values) / len(baseline_values)
        target_mean = sum(target_values) / len(target_values)
        paired.update(
            {
                "candidate_mean": candidate_mean,
                "baseline_mean": baseline_mean,
                "native14_mean": target_mean,
                "delta_vs_native14": candidate_mean - target_mean,
                "native14_drop_reduction_vs_k20": candidate_mean - baseline_mean,
                "raw_score_retention_percent": 100.0 * candidate_mean / target_mean if target_mean else None,
            }
        )
        result[metric] = paired
    candidate_em = [_metric(record, "official_answer_em") for record in candidate]
    baseline_em = [_metric(record, "official_answer_em") for record in baseline]
    correct = sum(value == 1.0 for value in baseline_em)
    result["r1_correct_em_retention"] = {
        "r1_correct_count": correct,
        "retained_count": sum(
            left == 1.0 and right == 1.0
            for left, right in zip(candidate_em, baseline_em, strict=True)
        ),
        "rate": (
            sum(
                left == 1.0 and right == 1.0
                for left, right in zip(candidate_em, baseline_em, strict=True)
            )
            / correct
            if correct
            else None
        ),
        "r1_correct_to_candidate_wrong": sum(
            left != 1.0 and right == 1.0
            for left, right in zip(candidate_em, baseline_em, strict=True)
        ),
    }
    candidate_f1 = [_metric(record, "official_answer_f1") for record in candidate]
    baseline_f1 = [_metric(record, "official_answer_f1") for record in baseline]
    result["f1_pairwise_direction"] = {
        "wins": sum(left > right for left, right in zip(candidate_f1, baseline_f1, strict=True)),
        "ties": sum(left == right for left, right in zip(candidate_f1, baseline_f1, strict=True)),
        "losses": sum(left < right for left, right in zip(candidate_f1, baseline_f1, strict=True)),
    }
    return result


def aggregate_paired_results(
    records: Sequence[Mapping[str, Any]],
    *,
    profile: str,
    expected_dataset_counts: Mapping[str, int],
    request_fingerprint: str,
    seed: int = 20260830,
    replicates: int = 10000,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ContractError(f"unknown profile: {profile}")
    if (
        not isinstance(expected_dataset_counts, Mapping)
        or set(expected_dataset_counts) != set(DATASETS)
        or any(
            type(expected_dataset_counts[dataset]) is not int
            or expected_dataset_counts[dataset] <= 0
            for dataset in DATASETS
        )
    ):
        raise ContractError("expected dataset counts are malformed")
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for record in records:
        dataset = record.get("dataset_id")
        sample_id = record.get("sample_id")
        mode = record.get("mode")
        if record.get("request_fingerprint") != request_fingerprint:
            raise ContractError("result request fingerprint changed")
        if dataset not in DATASETS or mode not in ALL_MODES or not isinstance(sample_id, str) or not sample_id:
            raise ContractError("result key is malformed")
        key = (dataset, sample_id, mode)
        if key in by_key:
            raise ContractError(f"duplicate result cell: {key}")
        by_key[key] = record

    dataset_ids: dict[str, list[str]] = {}
    for dataset in DATASETS:
        ids = sorted({sample_id for ds, sample_id, _mode in by_key if ds == dataset})
        if len(ids) != expected_dataset_counts[dataset]:
            raise ContractError(f"{profile}/{dataset} sample count changed")
        for sample_id in ids:
            for mode in ALL_MODES:
                if (dataset, sample_id, mode) not in by_key:
                    raise ContractError(f"missing paired cell: {(dataset, sample_id, mode)}")
        dataset_ids[dataset] = ids
    expected = sum(expected_dataset_counts.values()) * len(ALL_MODES)
    if len(by_key) != expected:
        raise ContractError("unexpected result cells exist")

    aggregates: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    pooled_by_mode: dict[str, list[Mapping[str, Any]]] = {mode: [] for mode in ALL_MODES}
    for dataset in DATASETS:
        mode_records = {
            mode: [by_key[(dataset, sample_id, mode)] for sample_id in dataset_ids[dataset]]
            for mode in ALL_MODES
        }
        for mode in ALL_MODES:
            pooled_by_mode[mode].extend(mode_records[mode])
        aggregates[dataset] = {
            mode: _mode_summary(values) for mode, values in mode_records.items()
        }
        comparisons[dataset] = {
            mode: _comparison(
                mode_records[mode],
                mode_records["nvidia_k20"],
                mode_records["native14"],
                label=f"{dataset}/{mode}",
                seed=seed,
                replicates=replicates,
                confidence_level=confidence_level,
            )
            for mode in CANDIDATE_MODES
        }
    aggregates["pooled"] = {mode: _mode_summary(values) for mode, values in pooled_by_mode.items()}
    comparisons["pooled"] = {
        mode: _comparison(
            pooled_by_mode[mode],
            pooled_by_mode["nvidia_k20"],
            pooled_by_mode["native14"],
            label=f"pooled/{mode}",
            seed=seed,
            replicates=replicates,
            confidence_level=confidence_level,
        )
        for mode in CANDIDATE_MODES
    }
    return {
        "schema_version": 2,
        "status": "complete",
        "profile": profile,
        "expected_dataset_counts": dict(expected_dataset_counts),
        "request_fingerprint": request_fingerprint,
        "paired_complete": True,
        "failed_or_missing_cells": 0,
        "raw_modes": ["native14", "nvidia_k20", "shrinkage", "tail_refresh_r32"],
        "derived_modes": ["two_pass_r1_then_r32"],
        "bootstrap": {
            "seed": seed,
            "replicates": replicates,
            "confidence_level": confidence_level,
            "resampling_unit": "paired_sample_id",
            "method": "paired_percentile_bootstrap",
        },
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "official_dev_or_test_opened": False,
        "claim_boundary": "method-dev prospective evidence only",
    }


__all__ = ["CANDIDATE_MODES", "aggregate_paired_results", "paired_bootstrap_mean_delta"]
