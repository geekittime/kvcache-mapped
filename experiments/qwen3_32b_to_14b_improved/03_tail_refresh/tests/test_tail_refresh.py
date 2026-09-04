from __future__ import annotations

from typing import Any, Sequence

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.tail_refresh import (
    TailRefreshMappedPrefixCacheBuilder,
    TailRefreshValidationError,
    plan_tail_refresh,
)

# The test double remains a subclass of the production verified builder.  It
# replaces only source->target mapping with an exact native-target prefix so
# the CPU oracle can isolate tail splitting, positions, RoPE, and cache growth.
from reverse_kv.runtime import ReverseMappedPrefixCacheBuilder
from kvcache_transfer.cache_adapter import (
    boundary_replay_inputs,
    cache_sequence_length,
)


def _tiny_qwen3() -> Qwen3ForCausalLM:
    torch.manual_seed(20260830)
    config = Qwen3Config(
        vocab_size=127,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        attention_dropout=0.0,
        use_cache=True,
    )
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


class TargetOwnPrefixBuilder(ReverseMappedPrefixCacheBuilder):
    """Exact-cache mapper test double; intentionally bypasses formal init."""

    def __init__(self, model: Qwen3ForCausalLM) -> None:
        self.source_model = model
        self.target_model = model
        self.calls: list[tuple[int, ...]] = []

    @property
    def provenance(self) -> dict[str, Any]:
        return {"test_double": "native_target_prefix"}

    @torch.inference_mode()
    def __call__(self, prefix_token_ids: Sequence[int]) -> Any:
        values = tuple(int(value) for value in prefix_token_ids)
        self.calls.append(values)
        ids = torch.tensor([values], dtype=torch.long)
        output = self.target_model.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(len(values)).unsqueeze(0),
            cache_position=torch.arange(len(values)),
            use_cache=True,
            return_dict=True,
        )
        return output.past_key_values


class FixedPrefixBuilder(ReverseMappedPrefixCacheBuilder):
    """Return one exact object repeatedly for the R=1 identity assertion."""

    def __init__(self, model: Qwen3ForCausalLM, cache: Any) -> None:
        self.source_model = model
        self.target_model = model
        self.cache = cache
        self.calls = 0

    @property
    def provenance(self) -> dict[str, Any]:
        return {"test_double": "fixed_cache"}

    def __call__(self, prefix_token_ids: Sequence[int]) -> Any:
        self.calls += 1
        return self.cache


@pytest.mark.parametrize(
    ("builder_tokens", "replay_tokens", "mapped", "tail", "positions"),
    [
        (63, 1, 63, 0, ()),
        (63, 8, 56, 7, tuple(range(56, 63))),
        (63, 32, 32, 31, tuple(range(32, 63))),
    ],
)
def test_plan_excludes_the_engine_owned_boundary(
    builder_tokens: int,
    replay_tokens: int,
    mapped: int,
    tail: int,
    positions: tuple[int, ...],
) -> None:
    plan = plan_tail_refresh(builder_tokens, replay_tokens)
    assert plan.full_prompt_tokens == builder_tokens + 1
    assert plan.mapped_prefix_tokens == mapped
    assert plan.builder_native_tail_tokens == tail
    assert plan.engine_boundary_tokens == 1
    assert plan.builder_native_tail_tokens + 1 == replay_tokens
    assert plan.native_tail_position_ids == positions
    assert plan.engine_boundary_position_id == builder_tokens


@pytest.mark.parametrize("bad_r", [0, 2, 7, 9, 31, 33, True])
def test_plan_rejects_unregistered_windows(bad_r: int) -> None:
    with pytest.raises((TypeError, TailRefreshValidationError)):
        plan_tail_refresh(64, bad_r)


def test_plan_fails_closed_instead_of_silently_becoming_all_native() -> None:
    with pytest.raises(TailRefreshValidationError, match="one token remains mapped"):
        plan_tail_refresh(31, 32)


def test_r1_is_exact_cache_object_passthrough_and_never_calls_target() -> None:
    model = _tiny_qwen3()
    ids = torch.arange(1, 17).remainder(model.config.vocab_size).unsqueeze(0)
    with torch.inference_mode():
        fixed_cache = model.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
            cache_position=torch.arange(ids.shape[1]),
            use_cache=True,
            return_dict=True,
        ).past_key_values
    raw = FixedPrefixBuilder(model, fixed_cache)
    wrapper = TailRefreshMappedPrefixCacheBuilder.from_shared_mapper(
        raw, replay_tokens=1
    )
    target_calls = 0
    original_forward = model.model.forward

    def counted_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal target_calls
        target_calls += 1
        return original_forward(*args, **kwargs)

    model.model.forward = counted_forward
    try:
        observed = wrapper(ids.squeeze(0).tolist())
    finally:
        model.model.forward = original_forward
    assert observed is fixed_cache
    assert raw.calls == 1
    assert target_calls == 0
    assert wrapper.last_call_audit == {
        **plan_tail_refresh(16, 1).to_dict(),
        "target_forward_calls": 0,
        "target_forward_query_tokens": 0,
        "cache_length_before_refresh": 16,
        "cache_length_after_refresh": 16,
        "r1_returned_underlying_cache_object_directly": True,
    }


@pytest.mark.parametrize("replay_tokens", [1, 8, 32])
def test_tiny_qwen3_tail_then_engine_boundary_matches_full_forward(
    replay_tokens: int,
) -> None:
    model = _tiny_qwen3()
    context = [((index * 17) + 3) % model.config.vocab_size for index in range(48)]
    underlying = TargetOwnPrefixBuilder(model)
    builder = TailRefreshMappedPrefixCacheBuilder.from_shared_mapper(
        underlying, replay_tokens=replay_tokens
    )

    # This is exactly the contract used by TransferEvaluationEngine:
    # builder(C[:-1]), followed by a separate replay of C[-1].
    prefix_cache = builder(context[:-1])
    assert cache_sequence_length(prefix_cache) == len(context) - 1
    boundary = torch.tensor([[context[-1]]], dtype=torch.long)
    with torch.inference_mode():
        replay = model(
            **boundary_replay_inputs(boundary, prefix_cache, clone_cache=True),
            return_dict=True,
        )
        full_ids = torch.tensor([context], dtype=torch.long)
        full = model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            position_ids=torch.arange(len(context)).unsqueeze(0),
            cache_position=torch.arange(len(context)),
            use_cache=True,
            return_dict=True,
        )
    torch.testing.assert_close(
        replay.logits[:, -1], full.logits[:, -1], atol=3e-5, rtol=3e-5
    )
    assert cache_sequence_length(replay.past_key_values) == len(context)

    audit = builder.last_call_audit
    assert audit is not None
    expected_tail = replay_tokens - 1
    expected_mapped = len(context) - replay_tokens
    assert audit["mapped_prefix_tokens"] == expected_mapped
    assert audit["builder_native_tail_tokens"] == expected_tail
    assert audit["engine_boundary_tokens"] == 1
    assert audit["target_forward_query_tokens"] == expected_tail
    assert audit.get("position_ids", []) == list(
        range(expected_mapped, len(context) - 1)
    )
    assert audit.get("cache_position", []) == list(
        range(expected_mapped, len(context) - 1)
    )
    assert underlying.calls == [tuple(context[:expected_mapped])]


def test_model_instance_swap_after_validation_is_rejected() -> None:
    model = _tiny_qwen3()
    underlying = TargetOwnPrefixBuilder(model)
    builder = TailRefreshMappedPrefixCacheBuilder.from_shared_mapper(
        underlying, replay_tokens=8
    )
    underlying.target_model = _tiny_qwen3()
    with pytest.raises(TailRefreshValidationError, match="target model instance changed"):
        builder(list(range(40)))


def test_variants_share_one_resident_mapper_object() -> None:
    model = _tiny_qwen3()
    underlying = TargetOwnPrefixBuilder(model)
    r1 = TailRefreshMappedPrefixCacheBuilder.from_shared_mapper(
        underlying, replay_tokens=1
    )
    r8 = r1.variant(8)
    r32 = r1.variant(32)
    assert r1._mapped_builder is r8._mapped_builder is r32._mapped_builder
    assert [r1.replay_tokens, r8.replay_tokens, r32.replay_tokens] == [1, 8, 32]
    assert r32.provenance["resident_mapper_shared_across_variants"] is True
