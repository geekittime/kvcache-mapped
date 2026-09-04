"""Deterministic, gold-free router for a prospective shrinkage comparison.

The module intentionally has no model/runtime imports.  Its public
``Observation`` contains only telemetry that is available after a completed
NVIDIA/R1 pass.  In particular, dataset labels, output text, answers, scores,
timings, cap flags, and GPU state cannot be supplied to the decision function.

This is a frozen CPU contract.  It is not an accuracy claim and should be
used with a newly frozen registry only after the pre-registration gates in
``PROTOCOL.md`` have been satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Literal, Sequence


Action = Literal["nvidia_k20", "shrinkage_r1"]


@dataclass(frozen=True)
class RouterConfig:
    """Immutable route thresholds and resource limits.

    The intersection gate is deliberate: both input and first-pass generation
    signals must be high.  This is stricter than the historical OR router and
    is intended to protect HotpotQA from unnecessary mapper changes.
    """

    prompt_enter_tokens: int = 2800
    r1_generated_enter_tokens: int = 16
    max_prompt_tokens: int = 32768
    max_r1_generated_tokens: int = 256
    max_escalation_fraction: float = 0.30
    max_escalations_absolute: int = 32

    def __post_init__(self) -> None:
        integer_fields = (
            "prompt_enter_tokens",
            "r1_generated_enter_tokens",
            "max_prompt_tokens",
            "max_r1_generated_tokens",
            "max_escalations_absolute",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.prompt_enter_tokens > self.max_prompt_tokens:
            raise ValueError("prompt threshold exceeds prompt limit")
        if self.r1_generated_enter_tokens > self.max_r1_generated_tokens:
            raise ValueError("generation threshold exceeds generation limit")
        if (
            type(self.max_escalation_fraction) not in (int, float)
            or not isfinite(float(self.max_escalation_fraction))
            or not 0.0 <= float(self.max_escalation_fraction) <= 1.0
        ):
            raise ValueError("max_escalation_fraction must be finite and in [0, 1]")


DEFAULT_CONFIG = RouterConfig()


@dataclass(frozen=True)
class Observation:
    """The complete runtime input to the router.

    Deliberately do not add fields here for dataset, text, answer, quality,
    timing, cap, or device state.  Callers should create one object only after
    the first pass has completed (or mark ``first_pass_complete=False``).
    """

    prompt_tokens: int
    r1_generated_tokens: int
    first_pass_complete: bool = True


@dataclass(frozen=True)
class Assessment:
    """Validated signals and the frozen intersection-gate result."""

    valid: bool
    prompt_signal: bool
    generation_signal: bool
    gate: bool
    no_go: str | None = None


@dataclass(frozen=True)
class Budget:
    """Mutable-by-replacement explicit escalation budget.

    The class is frozen so a route operation cannot mutate a shared budget by
    accident.  ``reserve`` returns a new budget and a success bit.
    """

    limit: int
    used: int = 0

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit < 0:
            raise ValueError("budget limit must be a non-negative integer")
        if type(self.used) is not int or not 0 <= self.used <= self.limit:
            raise ValueError("budget used must be an integer in [0, limit]")

    @classmethod
    def for_batch(cls, batch_size: int, config: RouterConfig = DEFAULT_CONFIG) -> "Budget":
        if type(batch_size) is not int or batch_size < 0:
            raise ValueError("batch_size must be a non-negative integer")
        limit = min(
            config.max_escalations_absolute,
            int(floor(float(config.max_escalation_fraction) * batch_size)),
        )
        return cls(limit=limit)

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def reserve(self) -> tuple["Budget", bool]:
        if self.used >= self.limit:
            return self, False
        return Budget(limit=self.limit, used=self.used + 1), True


@dataclass(frozen=True)
class RouteDecision:
    """A route outcome containing no quality or reference information."""

    action: Action
    escalated: bool
    eligible: bool
    reason: str
    budget_remaining: int


def _uint(value: object) -> bool:
    return type(value) is int and value >= 0


def assess(observation: Observation, config: RouterConfig = DEFAULT_CONFIG) -> Assessment:
    """Validate telemetry and evaluate the fixed AND gate.

    Invalid or incomplete telemetry is never eligible.  The function has no
    path to inspect arbitrary mappings, so forbidden fields cannot be smuggled
    into routing through a dictionary.
    """

    if not isinstance(observation, Observation):
        return Assessment(False, False, False, False, "observation_type")
    if not _uint(observation.prompt_tokens) or observation.prompt_tokens == 0:
        return Assessment(False, False, False, False, "prompt_tokens_invalid")
    if observation.prompt_tokens > config.max_prompt_tokens:
        return Assessment(False, False, False, False, "prompt_tokens_over_limit")
    if not _uint(observation.r1_generated_tokens):
        return Assessment(False, False, False, False, "r1_generated_tokens_invalid")
    if observation.r1_generated_tokens > config.max_r1_generated_tokens:
        return Assessment(False, False, False, False, "r1_generated_tokens_over_limit")
    if type(observation.first_pass_complete) is not bool:
        return Assessment(False, False, False, False, "first_pass_complete_invalid")
    if not observation.first_pass_complete:
        return Assessment(False, False, False, False, "first_pass_incomplete")
    prompt_signal = observation.prompt_tokens >= config.prompt_enter_tokens
    generation_signal = observation.r1_generated_tokens >= config.r1_generated_enter_tokens
    return Assessment(
        valid=True,
        prompt_signal=prompt_signal,
        generation_signal=generation_signal,
        gate=prompt_signal and generation_signal,
    )


def route_one(
    observation: Observation,
    budget: Budget,
    *,
    config: RouterConfig = DEFAULT_CONFIG,
) -> tuple[RouteDecision, Budget]:
    """Route one request and return ``(decision, updated_budget)``.

    The caller must retain the returned budget.  This explicit state transition
    prevents hidden cross-request counters and makes replay/audit deterministic.
    """

    if not isinstance(budget, Budget):
        raise TypeError("budget must be a Budget")
    assessment = assess(observation, config)
    if not assessment.valid:
        return (
            RouteDecision(
                action="nvidia_k20",
                escalated=False,
                eligible=False,
                reason=f"no_go:{assessment.no_go}",
                budget_remaining=budget.remaining,
            ),
            budget,
        )
    if not assessment.gate:
        return (
            RouteDecision(
                action="nvidia_k20",
                escalated=False,
                eligible=False,
                reason="below_dual_threshold",
                budget_remaining=budget.remaining,
            ),
            budget,
        )
    updated, reserved = budget.reserve()
    if not reserved:
        return (
            RouteDecision(
                action="nvidia_k20",
                escalated=False,
                eligible=True,
                reason="no_go:budget_exhausted",
                budget_remaining=budget.remaining,
            ),
            budget,
        )
    return (
        RouteDecision(
            action="shrinkage_r1",
            escalated=True,
            eligible=True,
            reason="escalate:prompt_and_r1_generation",
            budget_remaining=updated.remaining,
        ),
        updated,
    )


def route_batch(
    observations: Sequence[Observation],
    *,
    budget: Budget | None = None,
    config: RouterConfig = DEFAULT_CONFIG,
) -> tuple[tuple[RouteDecision, ...], Budget]:
    """Route a batch with a frozen budget and stable original-order tie break."""

    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence")
    local = budget if budget is not None else Budget.for_batch(len(observations), config)
    if not isinstance(local, Budget):
        raise TypeError("budget must be a Budget")

    # First evaluate all rows, then reserve in input order.  This makes the
    # policy deterministic even when a caller supplies a generator-like list
    # with expensive telemetry access, and ensures budget overflow is explicit.
    decisions: list[RouteDecision] = []
    for observation in observations:
        decision, local = route_one(observation, local, config=config)
        decisions.append(decision)
    return tuple(decisions), local


def observation_to_log(observation: Observation) -> dict[str, int | bool]:
    """Return the complete *allowed* telemetry record for an audit log."""

    if not isinstance(observation, Observation):
        raise TypeError("observation must be an Observation")
    return {
        "prompt_tokens": observation.prompt_tokens,
        "r1_generated_tokens": observation.r1_generated_tokens,
        "first_pass_complete": observation.first_pass_complete,
    }

