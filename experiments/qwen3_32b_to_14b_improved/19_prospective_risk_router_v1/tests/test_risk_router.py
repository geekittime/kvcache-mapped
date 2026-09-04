from __future__ import annotations

import inspect

import pytest

from src.risk_router import (
    DEFAULT_CONFIG,
    Budget,
    HysteresisLatch,
    Observation,
    RouterConfig,
    assess,
    route_batch,
    route_one,
)


def obs(prompt: int = 1000, generated: int = 4, context: int | None = None, complete: bool = True) -> Observation:
    return Observation(
        prompt_tokens=prompt,
        r1_generated_tokens=generated,
        context_tokens=context,
        first_pass_complete=complete,
    )


def test_frozen_thresholds_and_limits() -> None:
    assert DEFAULT_CONFIG.prompt_enter_tokens == 2800
    assert DEFAULT_CONFIG.prompt_exit_tokens == 2600
    assert DEFAULT_CONFIG.context_enter_tokens == 2800
    assert DEFAULT_CONFIG.context_exit_tokens == 2600
    assert DEFAULT_CONFIG.r1_generated_enter_tokens == 16
    assert DEFAULT_CONFIG.r1_generated_exit_tokens == 12
    assert DEFAULT_CONFIG.max_escalation_fraction == pytest.approx(0.30)
    assert DEFAULT_CONFIG.max_escalations_absolute == 32
    assert Budget.for_batch(96).limit == 28
    assert Budget.for_batch(3).limit == 0
    assert Budget.for_batch(110).limit == 32


def test_signal_boundaries_and_priority() -> None:
    assert assess(obs()).enter is False
    assert assess(obs(prompt=2800)).priority == 1
    assert assess(obs(generated=16)).priority == 2
    both = assess(obs(prompt=2800, generated=16, context=2800))
    assert both.enter is True
    assert both.priority == 3
    assert both.reasons == ("r1_generation_long", "prompt_long", "context_long")
    assert assess(obs(prompt=2799, generated=15, context=2799)).enter is False


def test_context_is_optional_but_consistency_is_checked() -> None:
    assert assess(obs(prompt=3000, generated=4)).context_signal is False
    invalid = assess(obs(prompt=2000, generated=4, context=2001))
    assert invalid.valid is False
    assert invalid.no_go == "context_exceeds_prompt"


def test_incomplete_or_malformed_observations_are_no_go() -> None:
    for observation, reason in (
        (obs(prompt=0), "prompt_tokens_invalid"),
        (obs(prompt=40000), "prompt_tokens_over_limit"),
        (obs(generated=257), "r1_generated_tokens_over_limit"),
        (obs(complete=False), "first_pass_incomplete"),
        (obs(context=-1), "context_tokens_invalid"),
    ):
        result = assess(observation)
        assert result.valid is False
        assert result.no_go == reason
    not_an_observation = assess({"prompt_tokens": 3000, "r1_generated_tokens": 20})
    assert not_an_observation.valid is False
    assert not_an_observation.no_go == "observation_type"


def test_hysteresis_latch_enters_and_clears_only_below_exit() -> None:
    latch = HysteresisLatch()
    low = obs(prompt=2700, generated=13)
    high = obs(prompt=2800, generated=16)
    mid = obs(prompt=2700, generated=13)
    clear = obs(prompt=2599, generated=11)
    assert latch.update(assess(low), low) is False
    assert latch.update(assess(high), high) is True
    assert latch.update(assess(mid), mid) is True
    assert latch.update(assess(clear), clear) is False
    assert latch.active is False


def test_route_one_reserves_budget_and_falls_back_when_exhausted() -> None:
    budget = Budget(limit=1)
    candidate = obs(prompt=2800, generated=16)
    first = route_one(candidate, budget)
    second = route_one(candidate, budget)
    assert first.action == "r32"
    assert first.escalated is True
    assert first.priority == 3
    assert budget.used == 1
    assert second.action == "r1"
    assert second.reason == "no_go:budget_exhausted"


def test_route_batch_uses_fixed_priority_then_stable_tie_order() -> None:
    observations = (
        obs(prompt=2800, generated=16),  # priority 3
        obs(prompt=2800, generated=4),  # priority 1
        obs(prompt=1000, generated=16),  # priority 2
        obs(prompt=3000, generated=4),  # priority 1, later tie
        obs(),
    )
    decisions = route_batch(observations, budget=Budget(limit=2))
    assert [decision.action for decision in decisions] == ["r32", "r1", "r32", "r1", "r1"]
    assert decisions[1].reason == "no_go:budget_priority_deferred"
    assert decisions[3].reason == "no_go:budget_priority_deferred"
    assert decisions[4].reason == "below_enter_threshold"
    assert decisions[0].priority == 3
    assert decisions[2].priority == 2


def test_route_batch_is_deterministic_and_handles_invalid_rows() -> None:
    observations = (obs(prompt=2800), obs(prompt=0), obs(generated=16))
    first = route_batch(observations, budget=Budget(limit=2))
    second = route_batch(observations, budget=Budget(limit=2))
    assert first == second
    assert first[0].action == "r32"
    assert first[1].action == "r1"
    assert first[1].reason == "no_go:prompt_tokens_invalid"
    assert first[2].action == "r32"


def test_budget_rejects_invalid_state() -> None:
    with pytest.raises(ValueError):
        Budget(limit=-1)
    with pytest.raises(ValueError):
        Budget(limit=1, used=2)
    with pytest.raises(ValueError):
        Budget.for_batch(-1)


def test_config_rejects_reversed_hysteresis() -> None:
    with pytest.raises(ValueError):
        RouterConfig(prompt_enter_tokens=10, prompt_exit_tokens=11)
    with pytest.raises(ValueError):
        RouterConfig(r1_generated_enter_tokens=10, r1_generated_exit_tokens=11)


def test_gold_free_surface_has_no_forbidden_measurements() -> None:
    fields = set(Observation.__dataclass_fields__)
    assert fields == {"prompt_tokens", "r1_generated_tokens", "context_tokens", "first_pass_complete"}
    forbidden_fragments = ("wall", "answer", "reference", "quality", "dataset", "gpu")
    for name in fields:
        assert not any(fragment in name.lower() for fragment in forbidden_fragments)
    with pytest.raises(TypeError):
        Observation(prompt_tokens=3000, r1_generated_tokens=20, wall_seconds=1.0)  # type: ignore[call-arg]
    assert "torch" not in inspect.getsource(__import__("src.risk_router", fromlist=["risk_router"]))


def test_route_one_does_not_mutate_a_latch_on_low_observation() -> None:
    budget = Budget(limit=1)
    latch = HysteresisLatch()
    decision = route_one(obs(), budget, latch=latch)
    assert decision.action == "r1"
    assert latch.active is False
