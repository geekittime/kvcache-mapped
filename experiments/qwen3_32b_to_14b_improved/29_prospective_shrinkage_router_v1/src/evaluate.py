"""CPU-only evaluator for the frozen three-arm router contract.

Quality fields are accepted here only *after* routing and only for offline
analysis.  They are intentionally absent from :mod:`src.router` and from the
telemetry projection in :mod:`src.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Mapping, Sequence


ArmName = str
VALID_ARMS = ("native14", "nvidia_k20", "shrinkage_r1")


@dataclass(frozen=True)
class ArmMetrics:
    f1: float
    em: float
    cap_hit: bool
    complete: bool

    def __post_init__(self) -> None:
        for name in ("f1", "em"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if type(self.cap_hit) is not bool or type(self.complete) is not bool:
            raise ValueError("cap_hit and complete must be bool")


@dataclass(frozen=True)
class PairedOutcome:
    """One fresh sealed sample with all three arms and an immutable route."""

    sample_id: str
    dataset_id: str
    split: str
    route_action: ArmName
    route_no_go: bool
    native14: ArmMetrics
    nvidia_k20: ArmMetrics
    shrinkage_r1: ArmMetrics

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if type(self.dataset_id) is not str or not self.dataset_id:
            raise ValueError("dataset_id must be non-empty")
        if self.split != "sealed_eval":
            raise ValueError("quality evaluator accepts sealed_eval rows only")
        if self.route_action not in VALID_ARMS[1:]:
            raise ValueError("route_action must be nvidia_k20 or shrinkage_r1")
        if type(self.route_no_go) is not bool:
            raise ValueError("route_no_go must be bool")

    def arm(self, name: ArmName) -> ArmMetrics:
        if name not in VALID_ARMS:
            raise ValueError(f"unknown arm: {name}")
        return getattr(self, name)


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    replicates: int
    seed: int


@dataclass(frozen=True)
class PromotionReport:
    passed: bool
    reasons: tuple[str, ...]
    overall_f1_delta: float
    overall_em_delta: float
    f1_ci: BootstrapCI
    per_dataset_f1: Mapping[str, float]
    per_dataset_em: Mapping[str, float]
    cap_rate_delta: float
    complete_rate_delta: float
    no_go_fraction: float
    escalation_fraction: float
    arm_means: Mapping[str, Mapping[str, float]]


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        raise ValueError("cannot average an empty sequence")
    return sum(vals) / len(vals)


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    position = fraction * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return float(sorted_values[lo])
    weight = position - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def _validate_rows(rows: Sequence[PairedOutcome]) -> tuple[PairedOutcome, ...]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    if not rows:
        raise ValueError("rows must not be empty")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, PairedOutcome):
            raise TypeError("rows must contain PairedOutcome objects")
        if row.sample_id in seen:
            raise ValueError(f"duplicate sample_id: {row.sample_id}")
        seen.add(row.sample_id)
    return tuple(rows)


def paired_bootstrap_ci(
    rows: Sequence[PairedOutcome],
    value: Callable[[PairedOutcome], float],
    *,
    replicates: int = 10000,
    seed: int = 2901,
    stratify: bool = True,
) -> BootstrapCI:
    """Return a deterministic percentile paired-bootstrap confidence interval.

    With ``stratify=True`` (the preregistered mode), rows are sampled within
    each dataset and the original dataset weights are preserved.  The random
    generator is local, so calling this function cannot alter global state.
    """

    checked = _validate_rows(rows)
    if type(replicates) is not int or replicates < 100:
        raise ValueError("replicates must be an integer >= 100")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    values = [float(value(row)) for row in checked]
    if any(not math.isfinite(item) for item in values):
        raise ValueError("bootstrap values must be finite")
    point = _mean(values)
    groups: list[tuple[int, ...]] = []
    if stratify:
        by_dataset: dict[str, list[int]] = {}
        for index, row in enumerate(checked):
            by_dataset.setdefault(row.dataset_id, []).append(index)
        groups = [tuple(indices) for _, indices in sorted(by_dataset.items())]
    else:
        groups = [tuple(range(len(checked)))]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        chosen: list[int] = []
        for indices in groups:
            chosen.extend(rng.choice(indices) for _ in indices)
        samples.append(_mean(values[index] for index in chosen))
    samples.sort()
    return BootstrapCI(
        point=point,
        lower=_percentile(samples, 0.025),
        upper=_percentile(samples, 0.975),
        replicates=replicates,
        seed=seed,
    )


def arm_means(rows: Sequence[PairedOutcome]) -> dict[str, dict[str, float]]:
    checked = _validate_rows(rows)
    return {
        arm: {
            "f1": _mean(row.arm(arm).f1 for row in checked),
            "em": _mean(row.arm(arm).em for row in checked),
            "cap_rate": _mean(float(row.arm(arm).cap_hit) for row in checked),
            "complete_rate": _mean(float(row.arm(arm).complete) for row in checked),
        }
        for arm in VALID_ARMS
    }


def _policy_f1(row: PairedOutcome) -> float:
    return row.arm(row.route_action).f1


def _policy_em(row: PairedOutcome) -> float:
    return row.arm(row.route_action).em


def evaluate_promotion(
    rows: Sequence[PairedOutcome],
    *,
    budget_limit: int | None = None,
    per_dataset_f1_min: float = 0.0,
    hotpot_f1_safety_min: float = -0.01,
    hotpot_em_safety_min: float = -0.02,
    per_dataset_em_min: float = -0.02,
    max_cap_rate_increase: float = 0.01,
    max_complete_rate_decrease: float = 0.01,
    max_no_go_fraction: float = 0.05,
    bootstrap_replicates: int = 10000,
    bootstrap_seed: int = 2901,
) -> PromotionReport:
    """Apply the frozen promotion/no-go gates to sealed paired outcomes."""

    checked = _validate_rows(rows)
    datasets = sorted({row.dataset_id for row in checked})
    if set(datasets) != {"hotpotqa_train", "musique_train"}:
        raise ValueError("sealed rows must contain exactly HotpotQA and MuSiQue")
    policy_f1_deltas = [_policy_f1(row) - row.nvidia_k20.f1 for row in checked]
    policy_em_deltas = [_policy_em(row) - row.nvidia_k20.em for row in checked]
    overall_f1 = _mean(policy_f1_deltas)
    overall_em = _mean(policy_em_deltas)
    f1_ci = paired_bootstrap_ci(
        checked,
        lambda row: _policy_f1(row) - row.nvidia_k20.f1,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
        stratify=True,
    )

    per_f1: dict[str, float] = {}
    per_em: dict[str, float] = {}
    for dataset in datasets:
        subset = [row for row in checked if row.dataset_id == dataset]
        per_f1[dataset] = _mean(_policy_f1(row) - row.nvidia_k20.f1 for row in subset)
        per_em[dataset] = _mean(_policy_em(row) - row.nvidia_k20.em for row in subset)

    nvidia_cap = _mean(float(row.nvidia_k20.cap_hit) for row in checked)
    policy_cap = _mean(float(row.arm(row.route_action).cap_hit) for row in checked)
    nvidia_complete = _mean(float(row.nvidia_k20.complete) for row in checked)
    policy_complete = _mean(float(row.arm(row.route_action).complete) for row in checked)
    cap_delta = policy_cap - nvidia_cap
    complete_delta = policy_complete - nvidia_complete
    no_go_fraction = _mean(float(row.route_no_go) for row in checked)
    escalation_fraction = _mean(float(row.route_action == "shrinkage_r1") for row in checked)
    if budget_limit is not None:
        if type(budget_limit) is not int or budget_limit < 0:
            raise ValueError("budget_limit must be a non-negative integer")
        if sum(row.route_action == "shrinkage_r1" for row in checked) > budget_limit:
            raise ValueError("sealed outcomes exceed declared escalation budget")

    reasons: list[str] = []
    for dataset, delta in per_f1.items():
        if delta < per_dataset_f1_min:
            reasons.append(f"f1_regression:{dataset}")
    for dataset, delta in per_em.items():
        if delta < per_dataset_em_min:
            reasons.append(f"em_regression:{dataset}")
    if per_f1.get("hotpotqa_train", 0.0) < hotpot_f1_safety_min:
        reasons.append("hotpot_f1_safety")
    if per_em.get("hotpotqa_train", 0.0) < hotpot_em_safety_min:
        reasons.append("hotpot_em_safety")
    if f1_ci.lower <= 0.0:
        reasons.append("combined_f1_ci_not_positive")
    if cap_delta > max_cap_rate_increase:
        reasons.append("cap_rate_increase")
    if complete_delta < -max_complete_rate_decrease:
        reasons.append("complete_rate_decrease")
    if no_go_fraction > max_no_go_fraction:
        reasons.append("no_go_fraction")

    return PromotionReport(
        passed=not reasons,
        reasons=tuple(reasons),
        overall_f1_delta=overall_f1,
        overall_em_delta=overall_em,
        f1_ci=f1_ci,
        per_dataset_f1=per_f1,
        per_dataset_em=per_em,
        cap_rate_delta=cap_delta,
        complete_rate_delta=complete_delta,
        no_go_fraction=no_go_fraction,
        escalation_fraction=escalation_fraction,
        arm_means=arm_means(checked),
    )

