"""Dependency-injected execution harness for the prepared A1 oracle screen.

The harness owns pairing and cache composition but intentionally does not own
model loading.  A later authorized runner supplies two prefix-cache providers
and one native-14B boundary/decode executor.  This keeps CPU preparation from
silently opening CUDA while making the causal runtime interface executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch

from cache_oracle import (
    CacheContext,
    CacheEnvelope,
    CompositionReceipt,
    OracleContractError,
    OracleObservation,
    compose_legacy_cache,
    intervention_catalog,
    score_paired_observations,
)


class PrefixCacheProvider(Protocol):
    """Return a fully validated cache envelope for the exact frozen prefix."""

    def __call__(
        self,
        *,
        context: CacheContext,
        prefix_token_ids: torch.Tensor,
    ) -> CacheEnvelope: ...


class BoundaryDecodeExecutor(Protocol):
    """Replay one boundary token, greedily decode <=256, and capture attention."""

    def __call__(
        self,
        *,
        context: CacheContext,
        boundary_token_id: int,
        composed_prefix_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        max_new_tokens: int,
        intervention_id: str,
    ) -> OracleObservation: ...


@dataclass(frozen=True)
class SamplePlan:
    context: CacheContext
    prompt_token_ids: tuple[int, ...]
    references: tuple[str, ...]
    gold_first_token_ids: tuple[int, ...]
    max_new_tokens: int = 256

    @classmethod
    def from_prepared_row(cls, row: Mapping[str, Any]) -> "SamplePlan":
        if row.get("schema_version") != 1 or row.get("tier") != "method-dev":
            raise OracleContractError("runtime received a non-method-dev prepared row")
        raw_context = row.get("cache_context")
        if not isinstance(raw_context, Mapping):
            raise OracleContractError("prepared row has no cache context")
        try:
            context = CacheContext(**dict(raw_context))
        except (TypeError, ValueError) as exc:
            raise OracleContractError(f"prepared cache context is malformed: {exc}") from exc
        context.validate()
        raw_ids = row.get("prompt_token_ids")
        raw_references = row.get("references")
        raw_gold = row.get("gold_first_token_ids")
        if not isinstance(raw_ids, list) or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in raw_ids
        ):
            raise OracleContractError("prepared prompt token IDs are malformed")
        if len(raw_ids) != context.prefix_length + 1 or raw_ids[-1] != context.boundary_token_id:
            raise OracleContractError("prepared tokens disagree with cache boundary")
        if not isinstance(raw_references, list) or not raw_references or not all(
            isinstance(value, str) and value for value in raw_references
        ):
            raise OracleContractError("prepared answer references are malformed")
        if not isinstance(raw_gold, list) or not raw_gold or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in raw_gold
        ):
            raise OracleContractError("prepared gold first-token IDs are malformed")
        if row.get("max_new_tokens") != 256:
            raise OracleContractError("A1 generation cap drifted from 256")
        return cls(
            context=context,
            prompt_token_ids=tuple(raw_ids),
            references=tuple(raw_references),
            gold_first_token_ids=tuple(raw_gold),
            max_new_tokens=256,
        )


@dataclass(frozen=True)
class HarnessResult:
    paired_records: tuple[dict[str, Any], ...]
    compositions: tuple[CompositionReceipt, ...]


class OracleHarness:
    """Run all 11 preregistered interventions with one frozen prompt/cap."""

    def __init__(
        self,
        *,
        native_cache_provider: PrefixCacheProvider,
        mapped_cache_provider: PrefixCacheProvider,
        boundary_decode_executor: BoundaryDecodeExecutor,
    ) -> None:
        self.native_cache_provider = native_cache_provider
        self.mapped_cache_provider = mapped_cache_provider
        self.boundary_decode_executor = boundary_decode_executor

    def run(self, plan: SamplePlan) -> HarnessResult:
        plan.context.validate()
        if plan.max_new_tokens != 256:
            raise OracleContractError("A1 requires an exact 256-token generation cap")
        prefix = torch.tensor([plan.prompt_token_ids[:-1]], dtype=torch.long)
        native_cache = self.native_cache_provider(context=plan.context, prefix_token_ids=prefix)
        mapped_cache = self.mapped_cache_provider(context=plan.context, prefix_token_ids=prefix)
        reference: OracleObservation | None = None
        paired: list[dict[str, Any]] = []
        receipts: list[CompositionReceipt] = []
        for intervention in intervention_catalog():
            composed, receipt = compose_legacy_cache(
                native_cache,
                mapped_cache,
                intervention,
                clone=True,
            )
            observation = self.boundary_decode_executor(
                context=plan.context,
                boundary_token_id=plan.context.boundary_token_id,
                composed_prefix_cache=composed,
                max_new_tokens=plan.max_new_tokens,
                intervention_id=intervention.id,
            )
            receipts.append(receipt)
            if intervention.id == "native_k_native_v":
                reference = observation
            if reference is None:
                raise OracleContractError("native-K/native-V reference must execute first")
            paired.append(
                score_paired_observations(
                    context=plan.context,
                    intervention=intervention,
                    references=plan.references,
                    gold_first_token_ids=plan.gold_first_token_ids,
                    native=reference,
                    candidate=observation,
                )
            )
        return HarnessResult(paired_records=tuple(paired), compositions=tuple(receipts))


__all__ = [
    "BoundaryDecodeExecutor",
    "HarnessResult",
    "OracleHarness",
    "PrefixCacheProvider",
    "SamplePlan",
]
