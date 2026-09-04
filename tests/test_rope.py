from __future__ import annotations

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvcache_transfer.rope import (
    apply_rope,
    qwen3_rope_cos_sin,
    qwen3_rope_cos_sin_from_config,
    remove_rope,
    rotate_half,
)


def tiny_qwen3() -> Qwen3ForCausalLM:
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=97,
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
    )
    return Qwen3ForCausalLM(config).eval()


def test_rotate_half_is_hf_split_half_not_interleaved() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    expected = torch.tensor([[-4.0, -5.0, -6.0, 1.0, 2.0, 3.0]])
    torch.testing.assert_close(rotate_half(x), expected)
    with pytest.raises(ValueError, match="even"):
        rotate_half(torch.zeros(1, 7))


@pytest.mark.parametrize("batched_positions", [False, True])
def test_default_frequency_order_matches_hf_qwen3(batched_positions: bool) -> None:
    model = tiny_qwen3()
    positions = torch.arange(11)
    position_ids = positions.unsqueeze(0).expand(2, -1) if batched_positions else positions
    ours = qwen3_rope_cos_sin_from_config(position_ids, model.config)

    hf_positions = position_ids if position_ids.ndim == 2 else position_ids.unsqueeze(0)
    hidden = torch.zeros(hf_positions.shape[0], hf_positions.shape[1], model.config.hidden_size)
    hf = model.model.rotary_emb(hidden, hf_positions)
    if not batched_positions:
        hf = tuple(item.squeeze(0) for item in hf)
    torch.testing.assert_close(ours[0], hf[0], atol=0, rtol=0)
    torch.testing.assert_close(ours[1], hf[1], atol=0, rtol=0)


def test_rope_forward_inverse_round_trip_uses_fp32() -> None:
    generator = torch.Generator().manual_seed(11)
    content = torch.randn(2, 3, 19, 16, generator=generator, dtype=torch.bfloat16)
    positions = torch.arange(19).unsqueeze(0).expand(2, -1)
    cos, sin = qwen3_rope_cos_sin(
        positions,
        head_dim=16,
        rope_theta=1_000_000.0,
        dtype=torch.bfloat16,
    )

    rotated = apply_rope(content, cos, sin)
    restored = remove_rope(rotated, cos, sin)
    assert rotated.dtype == torch.float32
    assert restored.dtype == torch.float32
    torch.testing.assert_close(restored, content.float(), atol=2e-6, rtol=2e-6)


def test_apply_rope_matches_hf_and_real_cache_is_k_norm_then_rope() -> None:
    model = tiny_qwen3()
    input_ids = torch.tensor([[4, 9, 2, 13, 8, 17, 1]])
    captured: dict[str, torch.Tensor] = {}

    def capture_attention_input(_module, args, kwargs) -> None:
        hidden_states = args[0] if args else kwargs["hidden_states"]
        captured["hidden_states"] = hidden_states.detach()

    handle = model.model.layers[0].self_attn.register_forward_pre_hook(
        capture_attention_input,
        with_kwargs=True,
    )
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=True)
    finally:
        handle.remove()

    attention = model.model.layers[0].self_attn
    hidden = captured["hidden_states"]
    shape = (*hidden.shape[:-1], -1, attention.head_dim)
    with torch.inference_mode():
        content_key = attention.k_norm(attention.k_proj(hidden).view(shape)).transpose(1, 2)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        cos, sin = model.model.rotary_emb(hidden, position_ids)

    expected = apply_rope(content_key, cos, sin)
    cached_key = output.past_key_values.key_cache[0]
    torch.testing.assert_close(expected, cached_key.float(), atol=2e-6, rtol=2e-6)
    restored = remove_rope(cached_key, cos, sin)
    torch.testing.assert_close(restored, content_key.float(), atol=2e-6, rtol=2e-6)


def test_remove_rope_rejects_noninvertible_frequencies() -> None:
    x = torch.randn(1, 2, 3, 8)
    cos = torch.zeros(1, 3, 8)
    sin = torch.zeros_like(cos)
    with pytest.raises(ValueError, match="not invertible"):
        remove_rope(x, cos, sin, validate=True)
