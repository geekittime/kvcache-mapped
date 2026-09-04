from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.utils import is_flash_attn_2_available

from kvcache_transfer.cache_adapter import (
    boundary_replay_inputs,
    build_dynamic_cache,
    cache_sequence_length,
    clone_dynamic_cache,
    continuation_scoring_inputs,
    full_attention_mask,
    qwen3_layer_devices,
    replay_cache_position,
    replay_position_ids,
    slice_dynamic_cache,
)


def tiny_qwen3(
    *,
    attn_implementation: str = "eager",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Qwen3ForCausalLM:
    torch.manual_seed(23)
    config = Qwen3Config(
        vocab_size=101,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        use_sliding_window=False,
        sliding_window=None,
        _attn_implementation=attn_implementation,
    )
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def test_build_dynamic_cache_accepts_stacked_tensors_and_does_not_alias() -> None:
    generator = torch.Generator().manual_seed(3)
    keys = torch.randn(3, 2, 4, 7, 8, generator=generator)
    values = torch.randn(3, 2, 4, 7, 8, generator=generator)
    cache = build_dynamic_cache(
        keys,
        values,
        layer_devices=["cpu", "cpu", "cpu"],
        dtype=torch.bfloat16,
    )

    assert len(cache) == 3
    assert cache.get_seq_length() == 7
    for layer_idx in range(3):
        assert cache.key_cache[layer_idx].shape == (2, 4, 7, 8)
        assert cache.key_cache[layer_idx].device.type == "cpu"
        assert cache.key_cache[layer_idx].dtype == torch.bfloat16
    before = cache.key_cache[0].clone()
    keys[0].zero_()
    torch.testing.assert_close(cache.key_cache[0], before)


def test_model_layer_device_placement_and_branch_clone_are_independent() -> None:
    model = tiny_qwen3()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        original = model(input_ids=input_ids, use_cache=True).past_key_values
    assert qwen3_layer_devices(model) == (torch.device("cpu"), torch.device("cpu"))

    branch = clone_dynamic_cache(original, model=model)
    assert branch.get_seq_length() == original.get_seq_length() == 4
    assert branch.key_cache[0].data_ptr() != original.key_cache[0].data_ptr()
    branch.key_cache[0].zero_()
    assert torch.count_nonzero(original.key_cache[0]) > 0


def test_slice_dynamic_cache_is_nonmutating() -> None:
    keys = torch.randn(2, 1, 2, 6, 8)
    values = torch.randn_like(keys)
    full = build_dynamic_cache(keys, values)
    prefix = slice_dynamic_cache(full, -1)
    assert full.get_seq_length() == 6
    assert prefix.get_seq_length() == 5
    assert prefix.key_cache[0].data_ptr() != full.key_cache[0].data_ptr()
    torch.testing.assert_close(prefix.key_cache[0], full.key_cache[0][..., :5, :])


def test_mask_position_and_cache_position_helpers() -> None:
    prefix_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    mask = full_attention_mask(prefix_mask, 2)
    expected_mask = torch.tensor([[0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]])
    torch.testing.assert_close(mask, expected_mask)
    expected_position_ids = torch.tensor([[2, 3], [3, 4]])
    torch.testing.assert_close(replay_position_ids(mask, 2), expected_position_ids)
    torch.testing.assert_close(replay_cache_position(4, 2, device="cpu"), torch.tensor([4, 5]))


def test_target_own_prefix_oracle_boundary_replay_matches_full_forward() -> None:
    model = tiny_qwen3()
    context = torch.tensor([[7, 5, 19, 2, 31, 11, 3]])
    with torch.inference_mode():
        full_logits = model(input_ids=context, use_cache=False).logits
        prefix_cache = model(input_ids=context[:, :-1], use_cache=True).past_key_values

    inputs = boundary_replay_inputs(context[:, -1:], prefix_cache)
    branch = inputs["past_key_values"]
    assert branch is not prefix_cache
    with torch.inference_mode():
        replay = model(**inputs)

    torch.testing.assert_close(replay.logits[:, -1], full_logits[:, -1], atol=2e-5, rtol=2e-5)
    # HF appended the boundary only to the branch supplied to the model.
    assert cache_sequence_length(prefix_cache) == context.shape[1] - 1
    assert cache_sequence_length(replay.past_key_values) == context.shape[1]


def test_continuation_scoring_matches_full_forward_and_preserves_branches() -> None:
    model = tiny_qwen3()
    context = torch.tensor([[7, 5, 19, 2, 31]])
    candidates = (torch.tensor([[11, 3, 17]]), torch.tensor([[29, 13]]))
    with torch.inference_mode():
        prefix_cache = model(input_ids=context[:, :-1], use_cache=True).past_key_values
    prefix_length = cache_sequence_length(prefix_cache)

    for continuation in candidates:
        whole_sequence = torch.cat((context, continuation), dim=-1)
        with torch.inference_mode():
            full_logits = model(input_ids=whole_sequence, use_cache=False).logits
            inputs, labels = continuation_scoring_inputs(
                context[:, -1:],
                continuation,
                prefix_cache,
            )
            replay = model(**inputs)

        expected_logits = full_logits[
            :,
            context.shape[1] - 1 : context.shape[1] - 1 + continuation.shape[1],
        ]
        torch.testing.assert_close(replay.logits, expected_logits, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(labels, continuation)
        expected_score = expected_logits.log_softmax(-1).gather(-1, continuation.unsqueeze(-1)).sum()
        replay_score = replay.logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).sum()
        torch.testing.assert_close(replay_score, expected_score, atol=3e-5, rtol=3e-5)
        assert cache_sequence_length(prefix_cache) == prefix_length


def test_boundary_replay_validates_prefix_mask_length() -> None:
    cache = build_dynamic_cache(torch.randn(2, 1, 2, 4, 8), torch.randn(2, 1, 2, 4, 8))
    with pytest.raises(ValueError, match="prefix_attention_mask"):
        boundary_replay_inputs(
            torch.tensor([[2]]),
            cache,
            prefix_attention_mask=torch.ones(1, 3, dtype=torch.long),
        )


@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or not is_flash_attn_2_available(),
    reason="CUDA FlashAttention-2 is unavailable",
)
def test_cuda_bf16_flash_attention_boundary_oracle() -> None:
    device = torch.device("cuda:0")
    model = tiny_qwen3(
        attn_implementation="flash_attention_2",
        device=device,
        dtype=torch.bfloat16,
    )
    context = torch.tensor([[7, 5, 19, 2, 31, 11, 3]], device=device)
    with torch.inference_mode():
        full_logits = model(input_ids=context, use_cache=False).logits
        prefix_cache = model(input_ids=context[:, :-1], use_cache=True).past_key_values
        replay = model(**boundary_replay_inputs(context[:, -1:], prefix_cache))
    assert replay.past_key_values.key_cache[0].dtype == torch.bfloat16
    torch.testing.assert_close(replay.logits[:, -1], full_logits[:, -1], atol=2e-2, rtol=2e-2)
