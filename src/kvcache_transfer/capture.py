"""Capture sampled, RoPE-stripped Qwen3 KV activations.

The paper fits on the values that are actually stored in the model cache.  For
Qwen3 this means K after ``k_norm`` and RoPE, and V after ``v_proj``.  This
module deliberately consumes ``past_key_values`` instead of a pre-projection
hook so calibration and serving traverse the same numerical path.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rope import qwen3_rope_cos_sin_from_config, remove_rope


@dataclass(frozen=True)
class SampledLayerKV:
    k_content: torch.Tensor
    value: torch.Tensor


def sampled_positions(
    sequence_length: int, stride: int = 4, start: int = 0, device: torch.device | str = "cpu"
) -> torch.Tensor:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not 0 <= start < sequence_length:
        raise ValueError("start must be inside the sequence")
    return torch.arange(start, sequence_length, stride, device=device)


def cache_layers(past_key_values: object) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return cache layers through the public Cache indexing protocol."""
    if not hasattr(past_key_values, "__len__") or not hasattr(past_key_values, "__getitem__"):
        raise TypeError("past_key_values must implement the transformers Cache protocol")
    return [past_key_values[index] for index in range(len(past_key_values))]


@torch.inference_mode()
def capture_sampled_content_kv(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    stride: int = 4,
    stride_start: int = 0,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
) -> list[SampledLayerKV]:
    """Run a BF16 prefill and return sampled content K and unmodified V.

    The first implementation path intentionally supports batch size one and no
    padding.  This makes positions unambiguous and matches the paper's
    token-level observations.  Batched padding must not be added without an
    explicit position/mask equivalence test.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("calibration capture currently requires batch size 1")
    sequence_length = input_ids.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if not bool(torch.all(attention_mask == 1)):
        raise ValueError("calibration capture does not permit padded sequences")
    if position_ids is None:
        position_ids = torch.arange(
            sequence_length, device=input_ids.device, dtype=torch.long
        ).unsqueeze(0)

    base_model = getattr(model, "model", model)
    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    layers = cache_layers(outputs.past_key_values)
    indices = sampled_positions(
        sequence_length, stride=stride, start=stride_start, device=input_ids.device
    )

    sampled: list[SampledLayerKV] = []
    for layer_index, (key, value) in enumerate(layers):
        layer_indices = indices.to(key.device)
        layer_positions = position_ids.to(key.device).index_select(1, layer_indices)
        layer_cos, layer_sin = qwen3_rope_cos_sin_from_config(
            layer_positions,
            model.config,
            device=key.device,
            dtype=key.dtype,
        )
        sampled_key = key.index_select(2, layer_indices)
        sampled_value = value.index_select(2, layer_indices)
        content = remove_rope(sampled_key.float(), layer_cos.float(), layer_sin.float())
        sampled.append(
            SampledLayerKV(
                k_content=content.transpose(1, 2).contiguous(),
                value=sampled_value.transpose(1, 2).contiguous(),
            )
        )
    return sampled
