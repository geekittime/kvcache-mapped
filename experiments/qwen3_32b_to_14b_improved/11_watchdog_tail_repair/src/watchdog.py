"""Gold-blind online state machine for Watchdog-Gated Tail Repair R32.

The runtime deliberately has no metric, reference, or dataset API.  A request is
opaque and is only forwarded to the engine.  The sole gate signal is whether an
R1 session has terminated after at most ``stall_budget`` generated tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class WatchdogError(RuntimeError):
    """The engine violated the online decoding contract."""


class State(str, Enum):
    R1_DECODING = "r1_decoding"
    R32_RESTART = "r32_restart"
    COMPLETE = "complete"


@dataclass(frozen=True)
class TokenEvent:
    """Optional content token and optional terminal signal.

    EOS may be reported without exposing the special EOS ID as a generated
    content token, hence ``token_id=None, stop_reason='eos'`` is valid.
    """

    token_id: int | None
    stop_reason: str | None = None


@runtime_checkable
class DecodeSession(Protocol):
    def step(self) -> TokenEvent: ...

    def abort(self) -> None: ...

    def finalize(self) -> Any: ...


@runtime_checkable
class RestartableDecodeEngine(Protocol):
    def start(self, request: Any, *, cache_mode: str) -> DecodeSession: ...


@dataclass(frozen=True)
class WatchdogAudit:
    triggered: bool
    r1_observed_tokens: int
    selected_mode: str
    state_path: tuple[State, ...]


@dataclass(frozen=True)
class WatchdogRun:
    """The output is the exact engine-owned object, never copied or rewritten."""

    output: Any
    audit: WatchdogAudit


def trigger_from_generation(generation: dict[str, Any], *, stall_budget: int = 32) -> bool:
    """Pure offline equivalent of the online gate; it cannot inspect gold data."""

    if set(generation).isdisjoint({"generated_tokens", "cap_hit"}):
        raise WatchdogError("generation evidence is missing")
    count = generation.get("generated_tokens")
    cap_hit = generation.get("cap_hit")
    if type(count) is not int or count < 0 or type(cap_hit) is not bool:
        raise WatchdogError("generation evidence is malformed")
    if type(stall_budget) is not int or stall_budget <= 0:
        raise WatchdogError("stall_budget must be a positive integer")
    return cap_hit or count > stall_budget


class WatchdogTailRepair:
    """Run R1 first and restart from an R32 cache only after 32 nonterminal tokens."""

    def __init__(self, engine: RestartableDecodeEngine, *, stall_budget: int = 32):
        if not isinstance(engine, RestartableDecodeEngine):
            raise TypeError("engine does not implement RestartableDecodeEngine")
        if type(stall_budget) is not int or stall_budget <= 0:
            raise ValueError("stall_budget must be a positive integer")
        self.engine = engine
        self.stall_budget = stall_budget

    @staticmethod
    def _validate_event(event: TokenEvent) -> None:
        if not isinstance(event, TokenEvent):
            raise WatchdogError("engine emitted an invalid token event")
        if event.token_id is not None and (type(event.token_id) is not int or event.token_id < 0):
            raise WatchdogError("engine emitted an invalid token ID")
        if event.stop_reason not in {None, "eos", "stop_sequence", "max_tokens"}:
            raise WatchdogError("engine emitted an invalid stop reason")
        if event.token_id is None and event.stop_reason is None:
            raise WatchdogError("engine emitted neither a content token nor termination")

    def run(self, request: Any) -> WatchdogRun:
        """Decode without inspecting any field of the opaque ``request`` object."""

        states = [State.R1_DECODING]
        r1 = self.engine.start(request, cache_mode="mapped_r1")
        observed = 0
        while observed < self.stall_budget:
            event = r1.step()
            self._validate_event(event)
            observed += event.token_id is not None
            if event.stop_reason is not None:
                # Critical identity property: return the engine-owned R1 object.
                output = r1.finalize()
                states.append(State.COMPLETE)
                return WatchdogRun(
                    output=output,
                    audit=WatchdogAudit(False, observed, "mapped_r1", tuple(states)),
                )

        # Token 32 was nonterminal.  Abort, build an R32 cache, and decode afresh.
        r1.abort()
        states.append(State.R32_RESTART)
        r32 = self.engine.start(request, cache_mode="tail_refresh_r32")
        while True:
            event = r32.step()
            self._validate_event(event)
            if event.stop_reason is not None:
                output = r32.finalize()
                states.append(State.COMPLETE)
                return WatchdogRun(
                    output=output,
                    audit=WatchdogAudit(True, observed, "tail_refresh_r32", tuple(states)),
                )
