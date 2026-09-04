"""Apply a fitted content-space KV mapper to a Qwen3 source cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .rope import apply_rope, remove_rope


@dataclass(frozen=True)
class LayerMapper:
    source_layers: tuple[int, ...]
    key_weight: torch.Tensor
    key_bias: torch.Tensor
    value_weight: torch.Tensor
    value_bias: torch.Tensor

    def validate(self, *, source_heads: int = 8, head_dim: int = 128) -> None:
        input_features = len(self.source_layers) * source_heads * head_dim
        output_features = source_heads * head_dim
        expected_weight = (input_features, output_features)
        expected_bias = (output_features,)
        for name, value in (
            ("key_weight", self.key_weight),
            ("value_weight", self.value_weight),
        ):
            if tuple(value.shape) != expected_weight:
                raise ValueError(f"{name}: expected {expected_weight}, got {tuple(value.shape)}")
        for name, value in (("key_bias", self.key_bias), ("value_bias", self.value_bias)):
            if tuple(value.shape) != expected_bias:
                raise ValueError(f"{name}: expected {expected_bias}, got {tuple(value.shape)}")


def _flatten_source_layer_chunk(
    layers: Sequence[torch.Tensor],
    selected: Sequence[int],
    begin: int,
    end: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Gather one token chunk in persisted ``[layer,head,dim]`` order.

    Accelerate leaves each source cache tensor on its decoder layer's device.
    Moving each selected piece before concatenation is therefore required for
    a layer-sharded Qwen3 model.  Chunking also avoids materializing the full
    8,192-wide FP32 feature matrix for long serving contexts.
    """

    pieces = [
        layers[index][:, :, begin:end, :]
        .permute(0, 2, 1, 3)
        .flatten(2)
        .to(device=device, dtype=torch.float32)
        for index in selected
    ]
    return torch.cat(pieces, dim=-1)


def apply_layer_mapper(
    source_content_keys: Sequence[torch.Tensor],
    source_values: Sequence[torch.Tensor],
    mapper: LayerMapper,
    target_cos: torch.Tensor,
    target_sin: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    token_chunk: int = 1024,
    source_heads: int = 8,
    head_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map one target layer in token chunks, returning ``[B,H,T,D]``.

    The defaults are the paper's 8 KV heads and 128-wide heads.  Explicit
    dimensions keep tiny Qwen3 integration tests on this exact runtime path
    without allocating a production 8192x1024 mapper.
    """

    mapper.validate(source_heads=source_heads, head_dim=head_dim)
    if token_chunk <= 0:
        raise ValueError("token_chunk must be positive")
    if not mapper.source_layers:
        raise ValueError("mapper must select at least one source layer")
    maximum_layer = max(mapper.source_layers)
    if maximum_layer >= len(source_content_keys) or maximum_layer >= len(source_values):
        raise ValueError("mapper selects a source layer absent from the supplied cache")
    reference_shape = source_content_keys[mapper.source_layers[0]].shape
    if len(reference_shape) != 4:
        raise ValueError("source cache tensors must have shape [batch,heads,sequence,head_dim]")
    for layer_index in mapper.source_layers:
        key_shape = source_content_keys[layer_index].shape
        value_shape = source_values[layer_index].shape
        if key_shape != reference_shape or value_shape != reference_shape:
            raise ValueError(
                "selected source K/V tensors must share one shape; "
                f"reference={tuple(reference_shape)}, layer {layer_index} "
                f"K={tuple(key_shape)} V={tuple(value_shape)}"
            )
    if reference_shape[1] != source_heads or reference_shape[3] != head_dim:
        raise ValueError(
            "source cache KV dimensions do not match mapper layout: "
            f"cache heads/dim=({reference_shape[1]},{reference_shape[3]}), "
            f"mapper=({source_heads},{head_dim})"
        )
    device = mapper.key_weight.device
    parameter_devices = {
        mapper.key_weight.device,
        mapper.key_bias.device,
        mapper.value_weight.device,
        mapper.value_bias.device,
    }
    if parameter_devices != {device}:
        raise ValueError(f"all mapper tensors must share one device, got {parameter_devices}")
    key_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    sequence_length = reference_shape[2]
    for begin in range(0, sequence_length, token_chunk):
        end = min(begin + token_chunk, sequence_length)
        x_key = _flatten_source_layer_chunk(
            source_content_keys, mapper.source_layers, begin, end, device=device
        )
        x_value = _flatten_source_layer_chunk(
            source_values, mapper.source_layers, begin, end, device=device
        )
        key_parts.append(x_key @ mapper.key_weight.float() + mapper.key_bias.float())
        value_parts.append(x_value @ mapper.value_weight.float() + mapper.value_bias.float())
    content_key = (
        torch.cat(key_parts, dim=1)
        .unflatten(-1, (source_heads, head_dim))
        .transpose(1, 2)
    )
    value = (
        torch.cat(value_parts, dim=1)
        .unflatten(-1, (source_heads, head_dim))
        .transpose(1, 2)
    )
    rotated_key = apply_rope(
        content_key,
        target_cos.to(device=device, dtype=torch.float32),
        target_sin.to(device=device, dtype=torch.float32),
    )
    return rotated_key.to(output_dtype), value.to(output_dtype)


def source_cache_to_content(
    source_cache: Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_cos: torch.Tensor,
    source_sin: torch.Tensor,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Strip source RoPE from all cache keys without repeating KV heads."""
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for key, value in source_cache:
        keys.append(
            remove_rope(
                key.float(),
                source_cos.to(device=key.device, dtype=torch.float32),
                source_sin.to(device=key.device, dtype=torch.float32),
            )
        )
        values.append(value)
    return keys, values
