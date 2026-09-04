from __future__ import annotations

import pytest

from src.router import Budget, Observation, RouterConfig, assess, observation_to_log, route_batch, route_one


def test_dual_gate_requires_both_signals() -> None:
    rows = (
        Observation(2799, 16),
        Observation(2800, 15),
        Observation(2800, 16),
    )
    decisions, budget = route_batch(rows, budget=Budget(limit=1))
    assert [item.action for item in decisions] == ["nvidia_k20", "nvidia_k20", "shrinkage_r1"]
    assert budget.used == 1
    assert decisions[2].reason == "escalate:prompt_and_r1_generation"


def test_budget_is_floor_and_capped() -> None:
    assert Budget.for_batch(0).limit == 0
    assert Budget.for_batch(3).limit == 0
    assert Budget.for_batch(96).limit == 28
    assert Budget.for_batch(1000).limit == 32


def test_budget_state_is_explicit_and_immutable() -> None:
    original = Budget(limit=1)
    decision, updated = route_one(Observation(3000, 20), original)
    assert decision.escalated is True
    assert original.used == 0
    assert updated.used == 1
    second, same = route_one(Observation(3000, 20), updated)
    assert second.action == "nvidia_k20"
    assert second.reason == "no_go:budget_exhausted"
    assert same == updated


@pytest.mark.parametrize(
    ("observation", "reason"),
    [
        (Observation(0, 16), "prompt_tokens_invalid"),
        (Observation(-1, 16), "prompt_tokens_invalid"),
        (Observation(32769, 16), "prompt_tokens_over_limit"),
        (Observation(2800, -1), "r1_generated_tokens_invalid"),
        (Observation(2800, 257), "r1_generated_tokens_over_limit"),
        (Observation(2800, 16, first_pass_complete=False), "first_pass_incomplete"),
        (Observation(2800, 16, first_pass_complete=1), "first_pass_complete_invalid"),
    ],
)
def test_invalid_telemetry_fails_closed(observation: Observation, reason: str) -> None:
    result = assess(observation)
    assert result.valid is False
    assert result.no_go == reason
    decisions, budget = route_batch((observation,))
    assert decisions[0].action == "nvidia_k20"
    assert decisions[0].reason == f"no_go:{reason}"
    assert budget.used == 0


def test_strict_integer_validation() -> None:
    with pytest.raises(ValueError):
        RouterConfig(prompt_enter_tokens=True)  # bool is not an integer here
    with pytest.raises(ValueError):
        RouterConfig(max_escalation_fraction=float("nan"))
    with pytest.raises(ValueError):
        Budget(limit=True)


def test_forbidden_fields_have_no_route_api() -> None:
    # The dataclass constructor itself rejects arbitrary quality/dataset fields.
    with pytest.raises(TypeError):
        Observation(prompt_tokens=3000, r1_generated_tokens=20, dataset_id="hotpotqa_train")  # type: ignore[call-arg]
    assert observation_to_log(Observation(3000, 20)) == {
        "prompt_tokens": 3000,
        "r1_generated_tokens": 20,
        "first_pass_complete": True,
    }


def test_batch_tie_break_is_original_order() -> None:
    rows = tuple(Observation(3000 + index, 20) for index in range(4))
    decisions, _ = route_batch(rows, budget=Budget(limit=2))
    assert [item.action for item in decisions] == ["shrinkage_r1", "shrinkage_r1", "nvidia_k20", "nvidia_k20"]
    assert decisions[2].reason == "no_go:budget_exhausted"


def test_strings_are_not_sequences_for_batch() -> None:
    with pytest.raises(TypeError):
        route_batch("bad")  # type: ignore[arg-type]
