"""Native target-tail refresh layered over the immutable reverse k20 mapper.

The shared evaluation engine owns the final causal boundary token.  If the
full prompt is ``C`` and the requested replay window is ``R``, the engine asks
this builder for a cache of ``C[:-1]``.  This builder therefore maps
``C[:-(R)]`` and appends exactly ``C[-R:-1]`` with native Qwen3-14B.  The
engine subsequently replays ``C[-1]``.  Counting the boundary in both places
would silently implement ``R+1`` and is explicitly guarded here.

No baseline code or mapper payload is modified.  ``R=1`` is a true zero-op
wrapper: it returns the exact cache object produced by
``ReverseMappedPrefixCacheBuilder`` without cloning, casting, or forwarding
the target model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
IMPROVED_ROOT = CANDIDATE_ROOT.parent
EXPERIMENTS_ROOT = IMPROVED_ROOT.parent
REPOSITORY_ROOT = EXPERIMENTS_ROOT.parent
BASELINE_EXPERIMENT_ROOT = EXPERIMENTS_ROOT / "qwen3_32b_to_14b"
REPOSITORY_SRC = REPOSITORY_ROOT / "src"

for _path in (BASELINE_EXPERIMENT_ROOT, REPOSITORY_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from kvcache_transfer.cache_adapter import (  # noqa: E402
    cache_batch_size,
    cache_sequence_length,
    full_attention_mask,
    replay_cache_position,
    replay_position_ids,
)
from reverse_kv.runtime import (  # noqa: E402
    ReverseMappedPrefixCacheBuilder,
)


ALLOWED_REPLAY_TOKENS = (1, 8, 32)
METHOD_ID = "native_target_tail_refresh_k20_v1"


class TailRefreshValidationError(ValueError):
    """A replay request or runtime object violates the method contract."""


@dataclass(frozen=True)
class TailRefreshPlan:
    """Exact split of the builder input and the engine-owned boundary."""

    builder_prefix_tokens: int
    full_prompt_tokens: int
    requested_total_native_prompt_tokens: int
    mapped_prefix_tokens: int
    builder_native_tail_tokens: int
    engine_boundary_tokens: int
    mapped_token_slice: tuple[int, int]
    native_tail_token_slice: tuple[int, int]
    native_tail_position_ids: tuple[int, ...]
    engine_boundary_position_id: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mapped_token_slice"] = list(self.mapped_token_slice)
        value["native_tail_token_slice"] = list(self.native_tail_token_slice)
        value["native_tail_position_ids"] = list(self.native_tail_position_ids)
        return value


def plan_tail_refresh(
    builder_prefix_tokens: int, replay_tokens: int
) -> TailRefreshPlan:
    """Plan ``R-1`` native appends for a builder input of ``C[:-1]``.

    ``builder_prefix_tokens`` is deliberately *not* the full prompt length.
    The shared engine has removed the boundary token before invoking a prefix
    builder.  At least one token must remain in the mapped region, both to
    preserve the hybrid method definition and because the production mapper
    rejects an empty prefix.  Method-dev prompts are preflighted against this
    requirement before GPUs are loaded.
    """

    if isinstance(builder_prefix_tokens, bool) or not isinstance(
        builder_prefix_tokens, int
    ):
        raise TypeError("builder_prefix_tokens must be an integer")
    if isinstance(replay_tokens, bool) or not isinstance(replay_tokens, int):
        raise TypeError("replay_tokens must be an integer")
    if replay_tokens not in ALLOWED_REPLAY_TOKENS:
        raise TailRefreshValidationError(
            f"replay_tokens must be one of {ALLOWED_REPLAY_TOKENS}, got {replay_tokens}"
        )
    if builder_prefix_tokens <= 0:
        raise TailRefreshValidationError("builder prefix must contain at least one token")

    builder_native_tail_tokens = replay_tokens - 1
    mapped_prefix_tokens = builder_prefix_tokens - builder_native_tail_tokens
    if mapped_prefix_tokens <= 0:
        raise TailRefreshValidationError(
            "prompt is too short for the registered hybrid split: builder receives "
            f"{builder_prefix_tokens} tokens but R={replay_tokens} requires at least "
            f"{replay_tokens} so one token remains mapped"
        )
    full_prompt_tokens = builder_prefix_tokens + 1
    positions = tuple(range(mapped_prefix_tokens, builder_prefix_tokens))
    plan = TailRefreshPlan(
        builder_prefix_tokens=builder_prefix_tokens,
        full_prompt_tokens=full_prompt_tokens,
        requested_total_native_prompt_tokens=replay_tokens,
        mapped_prefix_tokens=mapped_prefix_tokens,
        builder_native_tail_tokens=builder_native_tail_tokens,
        engine_boundary_tokens=1,
        mapped_token_slice=(0, mapped_prefix_tokens),
        native_tail_token_slice=(mapped_prefix_tokens, builder_prefix_tokens),
        native_tail_position_ids=positions,
        engine_boundary_position_id=builder_prefix_tokens,
    )
    if (
        plan.mapped_prefix_tokens
        + plan.builder_native_tail_tokens
        + plan.engine_boundary_tokens
        != plan.full_prompt_tokens
    ):
        raise AssertionError("tail-refresh causal partition is not exhaustive")
    if (
        plan.builder_native_tail_tokens + plan.engine_boundary_tokens
        != replay_tokens
    ):
        raise AssertionError("tail-refresh plan double-counted or lost the boundary")
    return plan


def _input_device(model: torch.nn.Module) -> torch.device:
    try:
        parameter = next(model.get_input_embeddings().parameters())
    except StopIteration as exc:
        raise TailRefreshValidationError(
            "target input embeddings contain no parameters"
        ) from exc
    if parameter.device.type == "meta":
        raise TailRefreshValidationError("target input embeddings remain on meta")
    return parameter.device


def _validate_token_ids(
    token_ids: Sequence[int], *, vocabulary_size: int | None
) -> list[int]:
    values: list[int] = []
    for token in token_ids:
        if isinstance(token, bool):
            raise TypeError("boolean values are not token IDs")
        value = int(token)
        if value < 0:
            raise TailRefreshValidationError("token IDs must be non-negative")
        if vocabulary_size is not None and value >= vocabulary_size:
            raise TailRefreshValidationError(
                f"token ID {value} is outside target vocab_size={vocabulary_size}"
            )
        values.append(value)
    return values


class TailRefreshMappedPrefixCacheBuilder:
    """Map a prefix with k20, then natively refresh its final ``R-1`` tokens.

    Construct one instance normally to validate both model artifacts and
    materialize the immutable k20 mapper.  :meth:`variant` creates lightweight
    R=1/8/32 views sharing those exact resident tensors, avoiding three copies
    of the 6.25-GiB mapper.
    """

    def __init__(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        mapper_manifest: str | Path,
        replay_tokens: int,
        token_chunk: int = 1024,
    ) -> None:
        mapped_builder = ReverseMappedPrefixCacheBuilder(
            source_model=source_model,
            target_model=target_model,
            mapper_manifest=mapper_manifest,
            token_chunk=token_chunk,
        )
        self._bind_shared_mapper(mapped_builder, replay_tokens=replay_tokens)

    def _bind_shared_mapper(
        self,
        mapped_builder: ReverseMappedPrefixCacheBuilder,
        *,
        replay_tokens: int,
    ) -> None:
        if not isinstance(mapped_builder, ReverseMappedPrefixCacheBuilder):
            raise TypeError(
                "tail refresh requires a verified ReverseMappedPrefixCacheBuilder"
            )
        # Validate R even before any prompt length is known.
        if replay_tokens not in ALLOWED_REPLAY_TOKENS or isinstance(
            replay_tokens, bool
        ):
            raise TailRefreshValidationError(
                f"replay_tokens must be one of {ALLOWED_REPLAY_TOKENS}"
            )
        self._mapped_builder = mapped_builder
        self.replay_tokens = int(replay_tokens)
        self._validated_source_model = mapped_builder.source_model
        self._validated_target_model = mapped_builder.target_model
        self._last_call_audit: dict[str, Any] | None = None

    @classmethod
    def from_shared_mapper(
        cls,
        mapped_builder: ReverseMappedPrefixCacheBuilder,
        *,
        replay_tokens: int,
    ) -> "TailRefreshMappedPrefixCacheBuilder":
        """Create an R view without reloading or copying mapper parameters."""

        instance = cls.__new__(cls)
        instance._bind_shared_mapper(mapped_builder, replay_tokens=replay_tokens)
        return instance

    def variant(self, replay_tokens: int) -> "TailRefreshMappedPrefixCacheBuilder":
        return type(self).from_shared_mapper(
            self._mapped_builder, replay_tokens=replay_tokens
        )

    @property
    def source_model(self) -> torch.nn.Module:
        return self._validated_source_model

    @property
    def target_model(self) -> torch.nn.Module:
        return self._validated_target_model

    @property
    def provenance(self) -> dict[str, Any]:
        baseline = self._mapped_builder.provenance
        if not isinstance(baseline, Mapping):
            raise TailRefreshValidationError("baseline mapper provenance is malformed")
        return {
            **dict(baseline),
            "method": METHOD_ID,
            "replay_tokens_R": self.replay_tokens,
            "builder_native_tail_tokens": self.replay_tokens - 1,
            "engine_boundary_tokens": 1,
            "boundary_owner": "TransferEvaluationEngine",
            "split_formula": (
                "builder receives C[:-1], maps first P-(R-1), natively appends "
                "last R-1, engine separately replays C[-1]"
            ),
            "r1_zero_op_passthrough": self.replay_tokens == 1,
            "resident_mapper_shared_across_variants": True,
        }

    @property
    def last_call_audit(self) -> dict[str, Any] | None:
        return (
            None
            if self._last_call_audit is None
            else dict(self._last_call_audit)
        )

    def _assert_model_instance_bindings(self) -> None:
        if self._mapped_builder.source_model is not self._validated_source_model:
            raise TailRefreshValidationError(
                "source model instance changed after mapper identity validation"
            )
        if self._mapped_builder.target_model is not self._validated_target_model:
            raise TailRefreshValidationError(
                "target model instance changed after mapper identity validation"
            )

    @torch.inference_mode()
    def __call__(self, prefix_token_ids: Sequence[int]) -> Any:
        self._assert_model_instance_bindings()
        target_vocab = getattr(self._validated_target_model.config, "vocab_size", None)
        prefix = _validate_token_ids(
            prefix_token_ids,
            vocabulary_size=(int(target_vocab) if target_vocab is not None else None),
        )
        plan = plan_tail_refresh(len(prefix), self.replay_tokens)
        mapped_ids = prefix[: plan.mapped_prefix_tokens]
        cache = self._mapped_builder(mapped_ids)
        if cache_sequence_length(cache) != plan.mapped_prefix_tokens:
            raise TailRefreshValidationError(
                "k20 cache length differs from the planned mapped prefix"
            )
        if cache_batch_size(cache) != 1:
            raise TailRefreshValidationError("tail refresh supports batch size one only")

        # R=1 must remain byte/tensor/object-identical to the k20 builder.  Do
        # not even construct target tensors on this path.
        if plan.builder_native_tail_tokens == 0:
            self._last_call_audit = {
                **plan.to_dict(),
                "target_forward_calls": 0,
                "target_forward_query_tokens": 0,
                "cache_length_before_refresh": plan.mapped_prefix_tokens,
                "cache_length_after_refresh": plan.mapped_prefix_tokens,
                "r1_returned_underlying_cache_object_directly": True,
            }
            return cache

        tail_ids = prefix[plan.mapped_prefix_tokens :]
        if len(tail_ids) != plan.builder_native_tail_tokens:
            raise AssertionError("native tail slice length differs from its plan")
        device = _input_device(self._validated_target_model)
        tail = torch.tensor([tail_ids], dtype=torch.long, device=device)
        attention_mask = full_attention_mask(
            plan.mapped_prefix_tokens,
            plan.builder_native_tail_tokens,
            batch_size=1,
            device=device,
        )
        position_ids = torch.tensor(
            [plan.native_tail_position_ids], dtype=torch.long, device=device
        )
        derived_position_ids = replay_position_ids(
            attention_mask, plan.builder_native_tail_tokens
        )
        if not torch.equal(position_ids, derived_position_ids):
            raise TailRefreshValidationError(
                "explicit RoPE positions differ from attention-mask-derived positions"
            )
        cache_position = replay_cache_position(
            plan.mapped_prefix_tokens,
            plan.builder_native_tail_tokens,
            device=device,
        )
        if not torch.equal(cache_position, position_ids.squeeze(0)):
            raise TailRefreshValidationError(
                "physical cache positions differ from logical RoPE positions"
            )

        target_base = getattr(
            self._validated_target_model, "model", self._validated_target_model
        )
        output = target_base(
            input_ids=tail,
            past_key_values=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        refreshed = getattr(output, "past_key_values", None)
        if refreshed is None:
            raise TailRefreshValidationError("target tail forward returned no cache")
        observed_length = cache_sequence_length(refreshed)
        if observed_length != plan.builder_prefix_tokens:
            raise TailRefreshValidationError(
                f"refreshed cache length {observed_length} != builder prefix "
                f"length {plan.builder_prefix_tokens}"
            )
        self._last_call_audit = {
            **plan.to_dict(),
            "target_forward_calls": 1,
            "target_forward_query_tokens": plan.builder_native_tail_tokens,
            "attention_mask_shape": list(attention_mask.shape),
            "position_ids": position_ids.squeeze(0).tolist(),
            "cache_position": cache_position.tolist(),
            "cache_length_before_refresh": plan.mapped_prefix_tokens,
            "cache_length_after_refresh": observed_length,
            "r1_returned_underlying_cache_object_directly": False,
        }
        return refreshed


__all__ = [
    "ALLOWED_REPLAY_TOKENS",
    "METHOD_ID",
    "TailRefreshMappedPrefixCacheBuilder",
    "TailRefreshPlan",
    "TailRefreshValidationError",
    "plan_tail_refresh",
]
