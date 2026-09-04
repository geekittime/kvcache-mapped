#!/usr/bin/env python3
"""A small, deterministic, CPU-only escalation router.

The router deliberately accepts only measurements that are available without
looking at a reference or evaluating an output: input token counts and the
length of the completed R=1 generation.  It does not import a model runtime,
read files, inspect text, or use timing information.

This module is a prospective *routing contract*, not an accuracy claim.  The
thresholds were frozen after inspecting the historical method-development
pilot and must be tested on a newly frozen registry before any scientific
interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor, isfinite
from typing import Literal, Sequence


Action = Literal["r1", "r32"]


@dataclass(frozen=True)
class RouterConfig:
    """Immutable routing thresholds and resource limits.

    The two enter/exit pairs provide hysteresis.  A caller should create a
    fresh latch for each request (or for one request's changing telemetry), so
    state cannot leak between examples.
    """

    prompt_enter_tokens: int = 2800
    prompt_exit_tokens: int = 2600
    context_enter_tokens: int = 2800
    context_exit_tokens: int = 2600
    r1_generated_enter_tokens: int = 16
    r1_generated_exit_tokens: int = 12
    max_prompt_tokens: int = 32768
    max_context_tokens: int = 32768
    max_r1_generated_tokens: int = 256
    max_escalation_fraction: float = 0.30
    max_escalations_absolute: int = 32

    def __post_init__(self) -> None:
        integer_names = (
            "prompt_enter_tokens",
            "prompt_exit_tokens",
            "context_enter_tokens",
            "context_exit_tokens",
            "r1_generated_enter_tokens",
            "r1_generated_exit_tokens",
            "max_prompt_tokens",
            "max_context_tokens",
            "max_r1_generated_tokens",
            "max_escalations_absolute",
        )
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.prompt_exit_tokens > self.prompt_enter_tokens:
            raise ValueError("prompt exit threshold exceeds enter threshold")
        if self.context_exit_tokens > self.context_enter_tokens:
            raise ValueError("context exit threshold exceeds enter threshold")
        if self.r1_generated_exit_tokens > self.r1_generated_enter_tokens:
            raise ValueError("generation exit threshold exceeds enter threshold")
        if self.prompt_enter_tokens > self.max_prompt_tokens:
            raise ValueError("prompt enter threshold exceeds prompt limit")
        if self.context_enter_tokens > self.max_context_tokens:
            raise ValueError("context enter threshold exceeds context limit")
        if self.r1_generated_enter_tokens > self.max_r1_generated_tokens:
            raise ValueError("generation enter threshold exceeds generation limit")
        if (
            type(self.max_escalation_fraction) not in (float, int)
            or not isfinite(float(self.max_escalation_fraction))
            or not 0.0 <= float(self.max_escalation_fraction) <= 1.0
        ):
            raise ValueError("max_escalation_fraction must be in [0, 1]")


DEFAULT_CONFIG = RouterConfig()


@dataclass(frozen=True)
class Observation:
    """The complete gold-free input to the router.

    ``context_tokens`` is optional because some serving paths expose only the
    total prompt count.  If supplied, it must be a non-negative count no
    larger than the prompt count.  ``first_pass_complete`` is a transport
    safety bit, not a quality signal: an incomplete first pass always falls
    back to R=1.
    """

    prompt_tokens: int
    r1_generated_tokens: int
    context_tokens: int | None = None
    first_pass_complete: bool = True


@dataclass(frozen=True)
class RiskAssessment:
    """Validated signals and deterministic priority for one observation."""

    valid: bool
    enter: bool
    priority: int
    prompt_signal: bool
    context_signal: bool
    generation_signal: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    no_go: str | None = None


@dataclass(frozen=True)
class RouteDecision:
    """A routing outcome; no quality or reference information is attached."""

    action: Action
    escalated: bool
    priority: int
    reason: str
    budget_remaining: int


@dataclass
class Budget:
    """Mutable, explicit escalation budget.

    The budget is intentionally separate from observations so its limit can be
    declared before a batch is processed.  ``used`` counts accepted R=32
    reservations only.
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
        """Create the frozen 30%-of-batch budget, capped at 32 requests.

        A floor is used deliberately: tiny batches do not silently exceed the
        declared fraction.  A batch whose computed limit is zero is all
        fallback, which is safer than borrowing from a future batch.
        """

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

    def reserve(self) -> bool:
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


@dataclass
class HysteresisLatch:
    """Per-request latch preventing threshold chatter.

    Call ``update`` with successive measurements from the same request.  The
    latch enters on any enter signal and clears only after *all available*
    dimensions are below their exit thresholds.  Invalid telemetry clears the
    latch and is never eligible for escalation.
    """

    active: bool = False

    def update(self, assessment: RiskAssessment, observation: Observation, config: RouterConfig = DEFAULT_CONFIG) -> bool:
        if not assessment.valid:
            self.active = False
            return False
        if self.active:
            prompt_clear = observation.prompt_tokens < config.prompt_exit_tokens
            context_clear = observation.context_tokens is None or observation.context_tokens < config.context_exit_tokens
            generation_clear = observation.r1_generated_tokens < config.r1_generated_exit_tokens
            if prompt_clear and context_clear and generation_clear:
                self.active = False
        elif assessment.enter:
            self.active = True
        return self.active


def _is_uint(value: object) -> bool:
    return type(value) is int and value >= 0


def assess(observation: Observation, config: RouterConfig = DEFAULT_CONFIG) -> RiskAssessment:
    """Validate one observation and compute gold-free risk signals."""

    if not isinstance(observation, Observation):
        return RiskAssessment(False, False, 0, False, False, False, no_go="observation_type")
    if not _is_uint(observation.prompt_tokens) or observation.prompt_tokens == 0:
        return RiskAssessment(False, False, 0, False, False, False, no_go="prompt_tokens_invalid")
    if observation.prompt_tokens > config.max_prompt_tokens:
        return RiskAssessment(False, False, 0, False, False, False, no_go="prompt_tokens_over_limit")
    if not _is_uint(observation.r1_generated_tokens):
        return RiskAssessment(False, False, 0, False, False, False, no_go="r1_generated_tokens_invalid")
    if observation.r1_generated_tokens > config.max_r1_generated_tokens:
        return RiskAssessment(False, False, 0, False, False, False, no_go="r1_generated_tokens_over_limit")
    if type(observation.first_pass_complete) is not bool or not observation.first_pass_complete:
        return RiskAssessment(False, False, 0, False, False, False, no_go="first_pass_incomplete")
    if observation.context_tokens is not None:
        if not _is_uint(observation.context_tokens):
            return RiskAssessment(False, False, 0, False, False, False, no_go="context_tokens_invalid")
        if observation.context_tokens > config.max_context_tokens:
            return RiskAssessment(False, False, 0, False, False, False, no_go="context_tokens_over_limit")
        if observation.context_tokens > observation.prompt_tokens:
            return RiskAssessment(False, False, 0, False, False, False, no_go="context_exceeds_prompt")

    prompt_signal = observation.prompt_tokens >= config.prompt_enter_tokens
    context_signal = (
        observation.context_tokens is not None
        and observation.context_tokens >= config.context_enter_tokens
    )
    generation_signal = observation.r1_generated_tokens >= config.r1_generated_enter_tokens
    enter = prompt_signal or context_signal or generation_signal
    priority = int(generation_signal) * 2 + int(prompt_signal or context_signal)
    reasons = tuple(
        reason
        for reason, active in (
            ("r1_generation_long", generation_signal),
            ("prompt_long", prompt_signal),
            ("context_long", context_signal),
        )
        if active
    )
    return RiskAssessment(
        valid=True,
        enter=enter,
        priority=priority,
        prompt_signal=prompt_signal,
        context_signal=context_signal,
        generation_signal=generation_signal,
        reasons=reasons,
    )


def _fallback(assessment: RiskAssessment, reason: str, budget: Budget) -> RouteDecision:
    return RouteDecision(
        action="r1",
        escalated=False,
        priority=assessment.priority,
        reason=reason,
        budget_remaining=budget.remaining,
    )


def route_one(
    observation: Observation,
    budget: Budget,
    *,
    latch: HysteresisLatch | None = None,
    config: RouterConfig = DEFAULT_CONFIG,
) -> RouteDecision:
    """Route one request, reserving budget only for an accepted escalation."""

    if not isinstance(budget, Budget):
        raise TypeError("budget must be a Budget")
    assessment = assess(observation, config)
    if not assessment.valid:
        if latch is not None:
            latch.active = False
        return _fallback(assessment, f"no_go:{assessment.no_go}", budget)
    active_latch = latch if latch is not None else HysteresisLatch()
    active = active_latch.update(assessment, observation, config)
    if not active:
        return _fallback(assessment, "below_enter_threshold", budget)
    if not budget.reserve():
        return _fallback(assessment, "no_go:budget_exhausted", budget)
    return RouteDecision(
        action="r32",
        escalated=True,
        priority=assessment.priority,
        reason="escalate:" + ",".join(assessment.reasons),
        budget_remaining=budget.remaining,
    )


def route_batch(
    observations: Sequence[Observation],
    *,
    budget: Budget | None = None,
    config: RouterConfig = DEFAULT_CONFIG,
) -> tuple[RouteDecision, ...]:
    """Route a batch with deterministic priority under a declared budget.

    Candidates are ranked by fixed priority: both long-generation and
    long-input signals (3), long generation (2), then long input (1).  Ties
    retain input order.  Each observation receives a fresh latch; callers who
    need telemetry hysteresis should use ``route_one`` for that request.
    """

    if not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence")
    local_budget = budget if budget is not None else Budget.for_batch(len(observations), config)
    if not isinstance(local_budget, Budget):
        raise TypeError("budget must be a Budget")
    assessments = [assess(observation, config) for observation in observations]
    eligible: list[int] = []
    for index, (observation, assessment) in enumerate(zip(observations, assessments)):
        if not assessment.valid:
            continue
        latch = HysteresisLatch()
        if latch.update(assessment, observation, config):
            eligible.append(index)
    eligible.sort(key=lambda index: (-assessments[index].priority, index))
    selected = set(eligible[: local_budget.remaining])
    decisions: list[RouteDecision] = []
    for index, assessment in enumerate(assessments):
        if not assessment.valid:
            decisions.append(_fallback(assessment, f"no_go:{assessment.no_go}", local_budget))
            continue
        if index not in eligible:
            decisions.append(_fallback(assessment, "below_enter_threshold", local_budget))
            continue
        if index not in selected:
            decisions.append(_fallback(assessment, "no_go:budget_priority_deferred", local_budget))
            continue
        if not local_budget.reserve():
            decisions.append(_fallback(assessment, "no_go:budget_exhausted", local_budget))
            continue
        decisions.append(
            RouteDecision(
                action="r32",
                escalated=True,
                priority=assessment.priority,
                reason="escalate:" + ",".join(assessment.reasons),
                budget_remaining=local_budget.remaining,
            )
        )
    return tuple(decisions)


__all__ = [
    "DEFAULT_CONFIG",
    "Budget",
    "HysteresisLatch",
    "Observation",
    "RouteDecision",
    "RouterConfig",
    "RiskAssessment",
    "assess",
    "route_batch",
    "route_one",
]
