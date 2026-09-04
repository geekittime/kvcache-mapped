"""CPU-only prospective shrinkage-router contract."""

from .router import (
    DEFAULT_CONFIG,
    Assessment,
    Budget,
    Observation,
    RouteDecision,
    RouterConfig,
    assess,
    observation_to_log,
    route_batch,
    route_one,
)

__all__ = [
    "DEFAULT_CONFIG",
    "Assessment",
    "Budget",
    "Observation",
    "RouteDecision",
    "RouterConfig",
    "assess",
    "observation_to_log",
    "route_batch",
    "route_one",
]
