from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _validate_finite(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask_float = mask.to(dtype=values.dtype)
    while mask_float.ndim < values.ndim:
        mask_float = mask_float.unsqueeze(-1)
    denominator = mask_float.expand_as(values).sum()
    if not bool(denominator > 0):
        raise ValueError("mask selects no elements")
    return (values * mask_float).sum() / denominator


@dataclass(frozen=True)
class AttentionLoss:
    centered_logit_huber: Tensor
    output_cosine: Tensor
    output_nmse: Tensor
    native_output: Tensor
    candidate_output: Tensor


def fixed_query_attention_loss(
    *,
    native_query: Tensor,
    native_key: Tensor,
    native_value: Tensor,
    candidate_key: Tensor,
    candidate_value: Tensor,
    causal_mask: Tensor,
    smooth_l1_beta: float = 0.1,
) -> AttentionLoss:
    """Native-fixed-query GQA loss without an output projection.

    Q is [B,Hq,Q,D], K/V are [B,Hkv,T,D], and causal_mask is [B,Q,T]
    or [B,1,Q,T]. Every query head mapped to a KV head is included.
    """

    tensors = {
        "native_query": native_query,
        "native_key": native_key,
        "native_value": native_value,
        "candidate_key": candidate_key,
        "candidate_value": candidate_value,
    }
    for name, tensor in tensors.items():
        _validate_finite(name, tensor)
        if tensor.ndim != 4:
            raise ValueError(f"{name} must be rank 4")
    if native_key.shape != candidate_key.shape or native_value.shape != candidate_value.shape:
        raise ValueError("native and candidate cache shapes must match")
    if native_key.shape != native_value.shape:
        raise ValueError("K and V shapes must match")
    batch, query_heads, query_rows, head_dim = native_query.shape
    key_batch, kv_heads, key_rows, key_dim = native_key.shape
    if batch != key_batch or head_dim != key_dim or query_heads % kv_heads:
        raise ValueError("incompatible GQA dimensions")
    if causal_mask.dtype != torch.bool:
        raise ValueError("causal_mask must be boolean")
    if causal_mask.ndim == 3:
        causal_mask = causal_mask.unsqueeze(1)
    if tuple(causal_mask.shape) not in ((batch, 1, query_rows, key_rows), (batch, query_heads, query_rows, key_rows)):
        raise ValueError("causal_mask has the wrong shape")
    mask = causal_mask.expand(batch, query_heads, query_rows, key_rows)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("every query row must have at least one causal key")

    repeat = query_heads // kv_heads
    native_k = native_key.repeat_interleave(repeat, dim=1)
    native_v = native_value.repeat_interleave(repeat, dim=1)
    candidate_k = candidate_key.repeat_interleave(repeat, dim=1)
    candidate_v = candidate_value.repeat_interleave(repeat, dim=1)
    scale = 1.0 / math.sqrt(head_dim)
    native_logits = torch.einsum("bhqd,bhkd->bhqk", native_query, native_k) * scale
    candidate_logits = torch.einsum("bhqd,bhkd->bhqk", native_query, candidate_k) * scale

    mask_float = mask.to(dtype=native_logits.dtype)
    count = mask_float.sum(dim=-1, keepdim=True)
    native_mean = (native_logits * mask_float).sum(dim=-1, keepdim=True) / count
    candidate_mean = (candidate_logits * mask_float).sum(dim=-1, keepdim=True) / count
    centered_diff = (candidate_logits - candidate_mean) - (native_logits - native_mean)
    logit_huber = _masked_mean(
        F.smooth_l1_loss(centered_diff, torch.zeros_like(centered_diff), beta=smooth_l1_beta, reduction="none"),
        mask,
    )

    negative_infinity = torch.finfo(native_logits.dtype).min
    native_probability = torch.softmax(native_logits.masked_fill(~mask, negative_infinity), dim=-1)
    candidate_probability = torch.softmax(candidate_logits.masked_fill(~mask, negative_infinity), dim=-1)
    native_output = torch.einsum("bhqk,bhkd->bhqd", native_probability, native_v)
    candidate_output = torch.einsum("bhqk,bhkd->bhqd", candidate_probability, candidate_v)
    cosine = 1.0 - F.cosine_similarity(native_output, candidate_output, dim=-1, eps=1e-8)
    output_cosine = cosine.mean()
    error_energy = (candidate_output - native_output).square().sum(dim=-1)
    native_energy = native_output.square().sum(dim=-1).clamp_min(1e-8)
    output_nmse = (error_energy / native_energy).mean()
    return AttentionLoss(logit_huber, output_cosine, output_nmse, native_output, candidate_output)


def suffix_kl_native_to_candidate(
    *,
    native_logits: Tensor,
    candidate_logits: Tensor,
    suffix_mask: Tensor,
    temperatures: Sequence[float] = (1.0, 2.0),
) -> Tensor:
    if native_logits.shape != candidate_logits.shape or native_logits.ndim != 3:
        raise ValueError("suffix logits must share shape [B,T,V]")
    if suffix_mask.dtype != torch.bool or tuple(suffix_mask.shape) != tuple(native_logits.shape[:2]):
        raise ValueError("suffix_mask must be boolean [B,T]")
    _validate_finite("native_logits", native_logits)
    _validate_finite("candidate_logits", candidate_logits)
    if not temperatures:
        raise ValueError("at least one temperature is required")
    losses: list[Tensor] = []
    for temperature in temperatures:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperatures must be real")
        tau = float(temperature)
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("temperatures must be finite and positive")
        teacher_log_probability = F.log_softmax(native_logits.detach() / tau, dim=-1)
        candidate_log_probability = F.log_softmax(candidate_logits / tau, dim=-1)
        teacher_probability = teacher_log_probability.exp()
        per_position = (
            teacher_probability * (teacher_log_probability - candidate_log_probability)
        ).sum(dim=-1) * (tau * tau)
        # Tiny negative values can arise from finite-precision subtraction even
        # though KL is mathematically nonnegative; clamp only that numerical
        # noise, never the logits or the teacher distribution.
        losses.append(_masked_mean(per_position.clamp_min(0.0), suffix_mask))
    return torch.stack(losses).mean()


def eos_soft_bce(
    *,
    native_logits: Tensor,
    candidate_logits: Tensor,
    suffix_mask: Tensor,
    eos_token_ids: Sequence[int],
) -> Tensor:
    if native_logits.shape != candidate_logits.shape or native_logits.ndim != 3:
        raise ValueError("suffix logits must share shape [B,T,V]")
    if suffix_mask.dtype != torch.bool or tuple(suffix_mask.shape) != tuple(native_logits.shape[:2]):
        raise ValueError("suffix_mask must be boolean [B,T]")
    vocabulary = native_logits.shape[-1]
    ids = tuple(eos_token_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("eos_token_ids must be nonempty and unique")
    if any(isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocabulary for token in ids):
        raise ValueError("EOS token outside vocabulary")
    _validate_finite("native_logits", native_logits)
    _validate_finite("candidate_logits", candidate_logits)
    teacher_probability = F.softmax(native_logits.detach(), dim=-1)[..., list(ids)].sum(dim=-1)
    candidate_probability = F.softmax(candidate_logits, dim=-1)[..., list(ids)].sum(dim=-1)
    eps = torch.finfo(candidate_probability.dtype).eps
    candidate_probability = candidate_probability.clamp(eps, 1.0 - eps)
    per_position = -teacher_probability * candidate_probability.log() - (
        1.0 - teacher_probability
    ) * torch.log1p(-candidate_probability)
    return _masked_mean(per_position, suffix_mask)


def variance_normalized_huber(
    *, candidate: Tensor, native: Tensor, variance: Tensor, delta: float = 1.0
) -> Tensor:
    if candidate.shape != native.shape:
        raise ValueError("candidate and native KV shapes must match")
    if variance.shape != native.shape[-2:]:
        raise ValueError("variance must have [kv_heads,head_dim] shape")
    if not math.isfinite(float(delta)) or delta <= 0:
        raise ValueError("delta must be finite and positive")
    _validate_finite("candidate", candidate)
    _validate_finite("native", native)
    _validate_finite("variance", variance)
    if not bool((variance > 0).all()):
        raise ValueError("variance must be positive")
    standardized_error = (candidate - native) / variance.sqrt().clamp_min(1e-8)
    return F.huber_loss(standardized_error, torch.zeros_like(standardized_error), delta=delta)


def global_r2(*, candidate: Tensor, native: Tensor) -> Tensor:
    if candidate.shape != native.shape or candidate.numel() < 2:
        raise ValueError("R2 tensors must share a nontrivial shape")
    _validate_finite("candidate", candidate)
    _validate_finite("native", native)
    residual = (native - candidate).double().square().sum()
    centered = native.double() - native.double().mean()
    total = centered.square().sum()
    if not bool(total > 0):
        raise ValueError("R2 undefined for constant native target")
    return (1.0 - residual / total).to(dtype=candidate.dtype)
