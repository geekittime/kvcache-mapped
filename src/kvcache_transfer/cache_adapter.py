"""Adapters for injecting transferred Qwen3 KV states into Transformers 4.51.

``DynamicCache.update`` mutates its argument.  All public branch/replay helpers
in this module therefore clone their input cache by default.  This is crucial
for multiple-choice evaluation, where every continuation must start from the
same mapped prefix rather than from the cache left behind by the prior choice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]
LayerSpec = (
    torch.device
    | str
    | torch.dtype
    | Sequence[torch.device | str | torch.dtype | None]
    | Mapping[int, torch.device | str | torch.dtype | None]
    | Callable[[int], torch.device | str | torch.dtype | None]
    | None
)


def _layer_value(spec: LayerSpec, layer_idx: int) -> Any:
    if spec is None:
        return None
    if callable(spec):
        return spec(layer_idx)
    if isinstance(spec, Mapping):
        if layer_idx not in spec:
            raise ValueError(f"missing specification for layer {layer_idx}")
        return spec[layer_idx]
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        if layer_idx >= len(spec):
            raise ValueError(f"only {len(spec)} layer specifications for layer {layer_idx}")
        return spec[layer_idx]
    return spec


def _as_legacy_pairs(
    key_cache: Cache | Sequence[Any] | torch.Tensor,
    value_cache: Sequence[torch.Tensor] | torch.Tensor | None,
) -> LegacyCache:
    if isinstance(key_cache, Cache):
        if value_cache is not None:
            raise ValueError("value_cache must be omitted when key_cache is a Cache")
        pairs = tuple((key, value) for key, value in key_cache)
    elif value_cache is None:
        if not isinstance(key_cache, Sequence):
            raise TypeError("expected a Cache, legacy sequence, or separate key/value tensors")
        pairs_list: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, pair in enumerate(key_cache):
            if not isinstance(pair, Sequence) or len(pair) != 2:
                raise TypeError(f"legacy cache layer {layer_idx} is not a (key, value) pair")
            pairs_list.append((pair[0], pair[1]))
        pairs = tuple(pairs_list)
    else:
        if isinstance(key_cache, torch.Tensor):
            if not isinstance(value_cache, torch.Tensor):
                raise TypeError("key_cache and value_cache must both be tensors or both be sequences")
            if key_cache.ndim != 5 or value_cache.ndim != 5:
                raise ValueError(
                    "stacked key/value tensors must be [layers,batch,heads,sequence,head_dim], "
                    f"got {tuple(key_cache.shape)} and {tuple(value_cache.shape)}"
                )
            if key_cache.shape[0] != value_cache.shape[0]:
                raise ValueError("stacked key/value tensors contain different layer counts")
            pairs = tuple((key_cache[i], value_cache[i]) for i in range(key_cache.shape[0]))
        else:
            if not isinstance(key_cache, Sequence) or not isinstance(value_cache, Sequence):
                raise TypeError("key_cache and value_cache must both be tensors or both be sequences")
            if len(key_cache) != len(value_cache):
                raise ValueError(f"key/value layer counts differ: {len(key_cache)} != {len(value_cache)}")
            pairs = tuple(zip(key_cache, value_cache, strict=True))

    if not pairs:
        raise ValueError("cache must contain at least one layer")
    for layer_idx, (key, value) in enumerate(pairs):
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"cache layer {layer_idx} does not contain torch tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                "DynamicCache tensors must be [batch,heads,sequence,head_dim], "
                f"got layer {layer_idx}: {tuple(key.shape)}, {tuple(value.shape)}"
            )
        if key.shape != value.shape:
            raise ValueError(
                f"key/value shapes differ at layer {layer_idx}: {tuple(key.shape)} != {tuple(value.shape)}"
            )
        if not key.is_floating_point() or not value.is_floating_point():
            raise TypeError(f"cache layer {layer_idx} must use floating-point tensors")
        if key.shape[-2] <= 0:
            raise ValueError(f"cache layer {layer_idx} has an empty sequence")

    reference_batch, reference_sequence = pairs[0][0].shape[0], pairs[0][0].shape[-2]
    for layer_idx, (key, _) in enumerate(pairs[1:], start=1):
        if key.shape[0] != reference_batch or key.shape[-2] != reference_sequence:
            raise ValueError(
                "all DynamicCache layers must have one batch/sequence shape; "
                f"layer 0 is ({reference_batch},{reference_sequence}), layer {layer_idx} is "
                f"({key.shape[0]},{key.shape[-2]})"
            )
    return pairs


def cache_to_legacy(
    cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    clone: bool = False,
    detach: bool = False,
) -> LegacyCache:
    """Return validated per-layer ``(key, value)`` tensors.

    The default is a zero-copy view of the cache.  Set ``clone=True`` before
    handing tensors to code that may write in-place.
    """

    pairs = _as_legacy_pairs(cache, None)
    if not clone and not detach:
        return pairs
    result = []
    for key, value in pairs:
        if detach:
            key, value = key.detach(), value.detach()
        if clone:
            key, value = key.clone(), value.clone()
        result.append((key, value))
    return tuple(result)


def qwen3_layer_devices(model: torch.nn.Module) -> tuple[torch.device, ...]:
    """Resolve the actual device of every Qwen3 decoder layer."""

    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise TypeError("model does not expose Qwen3 decoder layers as model.layers or layers")
    devices: list[torch.device] = []
    for layer_idx, layer in enumerate(layers):
        try:
            parameter = next(layer.parameters())
        except StopIteration as exc:
            raise ValueError(f"decoder layer {layer_idx} has no parameters from which to infer a device") from exc
        if parameter.device.type == "meta":
            raise ValueError(f"decoder layer {layer_idx} is still on the meta device")
        devices.append(parameter.device)
    return tuple(devices)


def build_dynamic_cache(
    key_cache: Cache | Sequence[Any] | torch.Tensor,
    value_cache: Sequence[torch.Tensor] | torch.Tensor | None = None,
    *,
    model: torch.nn.Module | None = None,
    layer_devices: LayerSpec = None,
    dtype: LayerSpec = None,
    clone: bool = True,
    detach: bool = True,
    non_blocking: bool = False,
) -> DynamicCache:
    """Build a fresh Transformers-4.51 ``DynamicCache``.

    Inputs may be a cache/legacy tuple, two per-layer sequences, or stacked
    ``[L,B,H,T,D]`` tensors.  ``model`` places each layer on the corresponding
    Qwen3 decoder device; alternatively use ``layer_devices`` explicitly.
    """

    if model is not None and layer_devices is not None:
        raise ValueError("pass model or layer_devices, not both")
    pairs = _as_legacy_pairs(key_cache, value_cache)
    if model is not None:
        layer_devices = qwen3_layer_devices(model)
        if len(layer_devices) != len(pairs):
            raise ValueError(
                f"model has {len(layer_devices)} decoder layers but cache has {len(pairs)} layers"
            )

    result = DynamicCache()
    for layer_idx, (key, value) in enumerate(pairs):
        device_value = _layer_value(layer_devices, layer_idx)
        dtype_value = _layer_value(dtype, layer_idx)
        if dtype_value is not None and not isinstance(dtype_value, torch.dtype):
            raise TypeError(f"dtype for layer {layer_idx} is not torch.dtype: {dtype_value!r}")
        if detach:
            key, value = key.detach(), value.detach()
        key = key.to(device=device_value, dtype=dtype_value, non_blocking=non_blocking)
        value = value.to(device=device_value, dtype=dtype_value, non_blocking=non_blocking)
        # .to can return the original storage.  A branch cache must not alias
        # either the mapper's output buffers or an oracle cache.
        if clone:
            key, value = key.clone(), value.clone()
        result.update(key, value, layer_idx)
    return result


def clone_dynamic_cache(
    cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    model: torch.nn.Module | None = None,
    layer_devices: LayerSpec = None,
    dtype: LayerSpec = None,
    detach: bool = True,
    non_blocking: bool = False,
) -> DynamicCache:
    """Deep-copy cache tensors so a decode branch cannot mutate its parent."""

    return build_dynamic_cache(
        cache,
        model=model,
        layer_devices=layer_devices,
        dtype=dtype,
        clone=True,
        detach=detach,
        non_blocking=non_blocking,
    )


def slice_dynamic_cache(
    cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    end: int,
    *,
    clone: bool = True,
) -> DynamicCache:
    """Create a prefix cache without calling mutating ``DynamicCache.crop``."""

    pairs = cache_to_legacy(cache)
    sequence_length = pairs[0][0].shape[-2]
    normalized_end = end if end >= 0 else sequence_length + end
    if not 0 < normalized_end <= sequence_length:
        raise ValueError(f"slice end must select a non-empty prefix of {sequence_length}, got {end}")
    sliced = tuple(
        (key[..., :normalized_end, :], value[..., :normalized_end, :]) for key, value in pairs
    )
    return build_dynamic_cache(sliced, clone=clone, detach=True)


def cache_sequence_length(cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Return a validated cache sequence length."""

    return int(cache_to_legacy(cache)[0][0].shape[-2])


def cache_batch_size(cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Return a validated cache batch size."""

    return int(cache_to_legacy(cache)[0][0].shape[0])


def full_attention_mask(
    prefix: int | torch.Tensor,
    query_length: int,
    *,
    batch_size: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """Append visible query positions to a cached-prefix attention mask."""

    if query_length <= 0:
        raise ValueError(f"query_length must be positive, got {query_length}")
    if isinstance(prefix, int):
        if prefix <= 0:
            raise ValueError(f"prefix length must be positive, got {prefix}")
        if batch_size is None or batch_size <= 0:
            raise ValueError("batch_size must be positive when prefix is a length")
        return torch.ones((batch_size, prefix + query_length), device=device, dtype=dtype)
    if not isinstance(prefix, torch.Tensor) or prefix.ndim != 2:
        raise ValueError("prefix attention mask must have shape [batch, prefix_length]")
    if batch_size is not None and prefix.shape[0] != batch_size:
        raise ValueError(f"prefix mask batch {prefix.shape[0]} != requested batch {batch_size}")
    resolved_device = torch.device(device) if device is not None else prefix.device
    prefix = prefix.to(device=resolved_device, dtype=dtype)
    suffix = torch.ones((prefix.shape[0], query_length), device=resolved_device, dtype=dtype)
    return torch.cat((prefix, suffix), dim=-1)


def replay_cache_position(
    prefix_length: int,
    query_length: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return physical cache slots for boundary replay/continuation tokens."""

    if prefix_length <= 0 or query_length <= 0:
        raise ValueError("prefix_length and query_length must both be positive")
    return torch.arange(prefix_length, prefix_length + query_length, device=device, dtype=torch.long)


def replay_position_ids(attention_mask: torch.Tensor, query_length: int) -> torch.Tensor:
    """Derive HF-generation position IDs, including correct left-padding behavior."""

    if attention_mask.ndim != 2 or query_length <= 0 or query_length > attention_mask.shape[-1]:
        raise ValueError("attention_mask must be [batch,total] and contain the complete non-empty query")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids[:, -query_length:]


def _validate_query_ids(input_ids: torch.Tensor, *, name: str, expected_batch: int) -> None:
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, sequence]")
    if input_ids.shape[0] != expected_batch or input_ids.shape[1] <= 0:
        raise ValueError(
            f"{name} has shape {tuple(input_ids.shape)}, expected batch {expected_batch} and non-empty sequence"
        )
    if input_ids.dtype not in (torch.int64, torch.int32):
        raise TypeError(f"{name} must contain integer token IDs, got {input_ids.dtype}")


def boundary_replay_inputs(
    boundary_input_ids: torch.Tensor,
    prefix_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    prefix_attention_mask: torch.Tensor | None = None,
    clone_cache: bool = True,
) -> dict[str, torch.Tensor | DynamicCache | bool]:
    """Prepare a one-token target replay after an injected prefix cache.

    The returned dictionary can be passed directly to Qwen3ForCausalLM.  It
    includes the *full* prefix-plus-query attention mask required by HF cache
    forwards, and an independent cache branch by default.
    """

    prefix_length = cache_sequence_length(prefix_cache)
    batch_size = cache_batch_size(prefix_cache)
    _validate_query_ids(boundary_input_ids, name="boundary_input_ids", expected_batch=batch_size)
    if boundary_input_ids.shape[1] != 1:
        raise ValueError("boundary_input_ids must contain exactly one token per sequence")
    if prefix_attention_mask is None:
        prefix: int | torch.Tensor = prefix_length
    else:
        if prefix_attention_mask.shape != (batch_size, prefix_length):
            raise ValueError(
                f"prefix_attention_mask must be {(batch_size, prefix_length)}, "
                f"got {tuple(prefix_attention_mask.shape)}"
            )
        prefix = prefix_attention_mask
    attention_mask = full_attention_mask(
        prefix,
        1,
        batch_size=batch_size,
        device=boundary_input_ids.device,
    )
    branch = clone_dynamic_cache(prefix_cache) if clone_cache else prefix_cache
    if not isinstance(branch, Cache):
        branch = build_dynamic_cache(branch)
    return {
        "input_ids": boundary_input_ids,
        "past_key_values": branch,
        "attention_mask": attention_mask,
        "position_ids": replay_position_ids(attention_mask, 1),
        "cache_position": replay_cache_position(prefix_length, 1, device=boundary_input_ids.device),
        "use_cache": True,
    }


def continuation_scoring_inputs(
    boundary_input_ids: torch.Tensor,
    continuation_input_ids: torch.Tensor,
    prefix_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    prefix_attention_mask: torch.Tensor | None = None,
    clone_cache: bool = True,
) -> tuple[dict[str, torch.Tensor | DynamicCache | bool], torch.Tensor]:
    """Prepare exact causal-LM logits for a continuation after a cached context.

    If context ``C`` has cached ``C[:-1]`` and candidate tokens are ``Y``, the
    target query is ``C[-1] + Y[:-1]``.  Its logits line up one-to-one with the
    separately returned labels ``Y``.  A fresh cache branch is used for every
    call, preventing multiple-choice candidates from contaminating each other.
    """

    prefix_length = cache_sequence_length(prefix_cache)
    batch_size = cache_batch_size(prefix_cache)
    _validate_query_ids(boundary_input_ids, name="boundary_input_ids", expected_batch=batch_size)
    if boundary_input_ids.shape[1] != 1:
        raise ValueError("boundary_input_ids must contain exactly one token per sequence")
    _validate_query_ids(
        continuation_input_ids,
        name="continuation_input_ids",
        expected_batch=batch_size,
    )
    continuation_input_ids = continuation_input_ids.to(boundary_input_ids.device)
    query = torch.cat((boundary_input_ids, continuation_input_ids[:, :-1]), dim=-1)
    query_length = query.shape[1]
    if prefix_attention_mask is None:
        prefix: int | torch.Tensor = prefix_length
    else:
        if prefix_attention_mask.shape != (batch_size, prefix_length):
            raise ValueError(
                f"prefix_attention_mask must be {(batch_size, prefix_length)}, "
                f"got {tuple(prefix_attention_mask.shape)}"
            )
        prefix = prefix_attention_mask
    attention_mask = full_attention_mask(
        prefix,
        query_length,
        batch_size=batch_size,
        device=query.device,
    )
    branch = clone_dynamic_cache(prefix_cache) if clone_cache else prefix_cache
    if not isinstance(branch, Cache):
        branch = build_dynamic_cache(branch)
    model_inputs: dict[str, torch.Tensor | DynamicCache | bool] = {
        "input_ids": query,
        "past_key_values": branch,
        "attention_mask": attention_mask,
        "position_ids": replay_position_ids(attention_mask, query_length),
        "cache_position": replay_cache_position(prefix_length, query_length, device=query.device),
        "use_cache": True,
    }
    return model_inputs, continuation_input_ids


# Short aliases retained for convenient use in evaluation scripts.
dynamic_cache_from_legacy = build_dynamic_cache
prepare_boundary_replay = boundary_replay_inputs
prepare_continuation_scoring = continuation_scoring_inputs


__all__ = [
    "boundary_replay_inputs",
    "build_dynamic_cache",
    "cache_batch_size",
    "cache_sequence_length",
    "cache_to_legacy",
    "clone_dynamic_cache",
    "continuation_scoring_inputs",
    "dynamic_cache_from_legacy",
    "full_attention_mask",
    "prepare_boundary_replay",
    "prepare_continuation_scoring",
    "qwen3_layer_devices",
    "replay_cache_position",
    "replay_position_ids",
    "slice_dynamic_cache",
]
