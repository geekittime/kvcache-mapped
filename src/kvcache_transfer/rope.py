"""Qwen3 rotary-position helpers used by cache extraction and injection.

Qwen3 in Transformers 4.51 uses the *split-half* rotary convention: the
first half of a head is paired with the second half.  It does not use the
even/odd (interleaved) convention found in some other implementations.

The paper fits keys in position-independent content space.  Consequently,
the numerically sensitive operations in this module deliberately run in
FP32, even when the model cache itself is BF16.
"""

from __future__ import annotations

from typing import Any

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Apply Hugging Face Qwen3's split-half quarter turn.

    For ``[..., d]`` with even ``d``, this returns
    ``cat(-x[..., d/2:], x[..., :d/2])``.  In particular this is not an
    even/odd interleave.
    """

    if x.ndim == 0 or x.shape[-1] % 2:
        raise ValueError(f"RoPE head dimension must be even, got shape {tuple(x.shape)}")
    midpoint = x.shape[-1] // 2
    return torch.cat((-x[..., midpoint:], x[..., :midpoint]), dim=-1)


def _broadcast_frequencies(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make HF-style ``[B,T,D]`` frequencies broadcast over cache heads."""

    if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
        raise TypeError("cos and sin must be torch tensors")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shapes differ: {tuple(cos.shape)} != {tuple(sin.shape)}")
    if cos.ndim == 0 or cos.shape[-1] != x.shape[-1]:
        raise ValueError(
            "RoPE frequencies must end in the cache head dimension; "
            f"got x={tuple(x.shape)}, cos={tuple(cos.shape)}"
        )
    if cos.ndim > x.ndim:
        raise ValueError(f"cos has more dimensions than x: {cos.ndim} > {x.ndim}")

    # HF Qwen3 produces [B,T,D] and explicitly inserts the KV-head axis for
    # cache tensors [B,H,T,D].  Leading singleton insertion also supports
    # shared [T,D] frequencies.
    while cos.ndim < x.ndim - 1:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    if cos.ndim == x.ndim - 1:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    try:
        torch.broadcast_shapes(x.shape, cos.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"RoPE frequencies {tuple(cos.shape)} do not broadcast to x {tuple(x.shape)}"
        ) from exc
    return cos, sin


def apply_rope(
    content: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """Rotate position-independent keys with FP32 intermediate arithmetic.

    Args:
        content: Usually ``[batch, kv_heads, sequence, head_dim]``.
        cos, sin: HF-style ``[batch, sequence, head_dim]``, shared
            ``[sequence, head_dim]``, or an already broadcastable tensor.
        output_dtype: Result dtype.  The default remains FP32 so fitting never
            silently drops back to BF16.  Pass ``content.dtype`` when building
            the final model cache.
    """

    if not isinstance(content, torch.Tensor) or not content.is_floating_point():
        raise TypeError("content must be a floating-point torch tensor")
    cos, sin = _broadcast_frequencies(content, cos, sin)
    content_fp32 = content.float()
    cos_fp32 = cos.to(device=content.device, dtype=torch.float32)
    sin_fp32 = sin.to(device=content.device, dtype=torch.float32)
    rotated = content_fp32 * cos_fp32 + rotate_half(content_fp32) * sin_fp32
    return rotated if output_dtype is None else rotated.to(output_dtype)


def remove_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = torch.float32,
    validate: bool = False,
) -> torch.Tensor:
    """Undo Qwen3 RoPE using FP32 arithmetic.

    Dividing by ``cos**2 + sin**2`` makes this a true inverse even for a
    rotary implementation with a non-unit attention scaling factor.  The
    Qwen3-14B and Qwen3-32B checkpoints used here both have unit scaling.
    ``validate=True`` diagnoses singular hand-written frequencies, but is
    intentionally off in calibration loops because it forces a CUDA sync.
    """

    if not isinstance(rotated, torch.Tensor) or not rotated.is_floating_point():
        raise TypeError("rotated must be a floating-point torch tensor")
    cos, sin = _broadcast_frequencies(rotated, cos, sin)
    rotated_fp32 = rotated.float()
    cos_fp32 = cos.to(device=rotated.device, dtype=torch.float32)
    sin_fp32 = sin.to(device=rotated.device, dtype=torch.float32)
    denominator = cos_fp32.square() + sin_fp32.square()
    # A device-wide reduction here would synchronize CUDA once per layer for
    # every calibration sequence.  Valid model-produced sin/cos cannot be
    # singular, so expensive defensive checking is opt-in for diagnostics.
    if validate and torch.any(denominator == 0):
        raise ValueError("cos**2 + sin**2 contains zero; RoPE is not invertible")
    content = (rotated_fp32 * cos_fp32 - rotate_half(rotated_fp32) * sin_fp32) / denominator
    return content if output_dtype is None else content.to(output_dtype)


def qwen3_rope_cos_sin(
    position_ids: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct default Qwen3 RoPE frequencies exactly in HF's dimension order.

    ``position_ids`` can be ``[T]`` or ``[B,T]``.  The returned leading shape
    matches it and the last dimension is ``head_dim``.  Trigonometry is always
    evaluated in FP32 and only the returned values are optionally cast.
    """

    if not isinstance(position_ids, torch.Tensor):
        raise TypeError("position_ids must be a torch tensor")
    if position_ids.ndim not in (1, 2):
        raise ValueError(f"position_ids must have shape [T] or [B,T], got {tuple(position_ids.shape)}")
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even integer, got {head_dim}")
    if rope_theta <= 0:
        raise ValueError(f"rope_theta must be positive, got {rope_theta}")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError(f"RoPE output dtype must be floating point, got {dtype}")

    resolved_device = torch.device(device) if device is not None else position_ids.device
    positions_fp32 = position_ids.to(device=resolved_device, dtype=torch.float32)
    dimension_ids = torch.arange(0, head_dim, 2, device=resolved_device, dtype=torch.float32)
    inv_freq = 1.0 / (float(rope_theta) ** (dimension_ids / head_dim))
    freqs = positions_fp32.unsqueeze(-1) * inv_freq
    # HF Qwen3 uses cat(freqs, freqs), not repeat_interleave(freqs, 2).
    angles = torch.cat((freqs, freqs), dim=-1)
    with torch.autocast(device_type=resolved_device.type if resolved_device.type != "mps" else "cpu", enabled=False):
        cos = angles.cos()
        sin = angles.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def qwen3_rope_cos_sin_from_config(
    position_ids: torch.Tensor,
    config: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build frequencies for the default-RoPE Qwen3 checkpoints in this run."""

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling not in (None, {}):
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        if rope_type != "default":
            raise NotImplementedError(
                "Standalone frequency construction only supports default Qwen3 RoPE; "
                "for scaled/dynamic RoPE pass cos/sin produced by model.model.rotary_emb"
            )
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    return qwen3_rope_cos_sin(
        position_ids,
        head_dim=int(head_dim),
        rope_theta=float(config.rope_theta),
        device=device,
        dtype=dtype,
    )


# Explicit aliases make call sites read naturally during mapper application.
apply_qwen3_rope = apply_rope
invert_qwen3_rope = remove_rope


__all__ = [
    "apply_qwen3_rope",
    "apply_rope",
    "invert_qwen3_rope",
    "qwen3_rope_cos_sin",
    "qwen3_rope_cos_sin_from_config",
    "remove_rope",
    "rotate_half",
]
