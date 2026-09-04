from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.watchdog import State, TokenEvent, WatchdogTailRepair, trigger_from_generation


class FakeSession:
    def __init__(self, events, outcome):
        self.events = iter(events)
        self.outcome = outcome
        self.aborted = False
        self.finalized = False

    def step(self):
        return next(self.events)

    def abort(self):
        self.aborted = True

    def finalize(self):
        self.finalized = True
        return self.outcome


class FakeEngine:
    def __init__(self, sessions):
        self.sessions = sessions
        self.starts = []

    def start(self, request, *, cache_mode):
        self.starts.append((request, cache_mode))
        return self.sessions[cache_mode]


def events_until_terminal(content_tokens: int):
    assert content_tokens >= 0
    return [TokenEvent(i) for i in range(content_tokens)] + [TokenEvent(None, "eos")]


def test_no_trigger_returns_exact_r1_object_identity():
    r1_outcome = object()
    r32_outcome = object()
    r1 = FakeSession(events_until_terminal(3), r1_outcome)
    r32 = FakeSession(events_until_terminal(1), r32_outcome)
    engine = FakeEngine({"mapped_r1": r1, "tail_refresh_r32": r32})

    run = WatchdogTailRepair(engine).run({"gold": "must remain opaque", "dataset_id": "poison"})

    assert run.output is r1_outcome
    assert run.audit.triggered is False
    assert run.audit.r1_observed_tokens == 3
    assert run.audit.state_path == (State.R1_DECODING, State.COMPLETE)
    assert engine.starts == [({"gold": "must remain opaque", "dataset_id": "poison"}, "mapped_r1")]
    assert not r1.aborted and r1.finalized
    assert not r32.finalized


def test_nonterminal_token_32_aborts_and_restarts_r32():
    r1_outcome = object()
    r32_outcome = object()
    r1 = FakeSession([TokenEvent(i) for i in range(40)] + [TokenEvent(None, "eos")], r1_outcome)
    r32 = FakeSession(events_until_terminal(2), r32_outcome)
    engine = FakeEngine({"mapped_r1": r1, "tail_refresh_r32": r32})

    request = object()
    run = WatchdogTailRepair(engine).run(request)

    assert run.output is r32_outcome
    assert run.audit.triggered is True
    assert run.audit.r1_observed_tokens == 32
    assert run.audit.state_path == (State.R1_DECODING, State.R32_RESTART, State.COMPLETE)
    assert r1.aborted and not r1.finalized
    assert r32.finalized
    assert engine.starts == [(request, "mapped_r1"), (request, "tail_refresh_r32")]


def test_terminal_on_content_token_32_does_not_trigger():
    outcome = object()
    events = [TokenEvent(i) for i in range(31)] + [TokenEvent(31, "stop_sequence")]
    r1 = FakeSession(events, outcome)
    engine = FakeEngine({"mapped_r1": r1, "tail_refresh_r32": FakeSession([], object())})
    run = WatchdogTailRepair(engine).run(object())
    assert run.output is outcome
    assert run.audit.r1_observed_tokens == 32
    assert run.audit.triggered is False


@pytest.mark.parametrize(
    "generation,expected",
    [
        ({"generated_tokens": 32, "cap_hit": False}, False),
        ({"generated_tokens": 33, "cap_hit": False}, True),
        ({"generated_tokens": 1, "cap_hit": True}, True),
    ],
)
def test_offline_gate_exact_boundary(generation, expected):
    assert trigger_from_generation(generation) is expected


def test_offline_gate_ignores_gold_dataset_prediction_and_metrics():
    base = {"generated_tokens": 12, "cap_hit": False}
    poisoned = {
        **base,
        "references": ["gold"],
        "dataset_id": "hotpotqa_train",
        "prediction": "anything",
        "metrics": {"qa_f1": 0.0},
    }
    assert trigger_from_generation(base) == trigger_from_generation(poisoned) is False
