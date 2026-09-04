"""Auditable mapped-vs-target attention-output cosine diagnostics.

The paper defines attention-output cosine only at a high level: compute the
cosine between attention outputs obtained with mapped and ground-truth KV,
then average over layers and heads.  It does not publish the exact query-token
sample, attention kernel/precision, or whether queries follow the mapped or
ground-truth hidden-state trajectory.

This independent diagnostic therefore freezes the following local convention:

* queries come from the target model's *ground-truth* prefill trajectory;
* the same post-RoPE query is applied to ground-truth and mapped post-RoPE KV;
* causal attention is evaluated explicitly in FP32, before ``o_proj``;
* cosine is computed in FP64 for every (sample, query token, query head);
* the headline is the explicit arithmetic mean of target-layer means; strict
  equal-count invariants within and across layers then give every query head,
  sample, and selected query token equal weight as well.

Holding Q fixed isolates the cache mapper itself instead of mixing cache error
with downstream hidden-state drift.  The implementation is deliberately kept
outside the formal evaluation/runtime code identity so adding this post-hoc
diagnostic cannot alter an accuracy/PPL run fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch


class AttentionDiagnosticError(ValueError):
    """Raised when a diagnostic input or model hook violates its contract."""


def _require_float_tensor(name: str, value: torch.Tensor, *, ndim: int) -> None:
    if not isinstance(value, torch.Tensor):
        raise AttentionDiagnosticError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise AttentionDiagnosticError(
            f"{name} must have {ndim} dimensions, got shape {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        raise AttentionDiagnosticError(f"{name} must be floating point")


def _validated_query_positions(
    query_positions: Sequence[int] | torch.Tensor,
    *,
    query_length: int,
    key_length: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.as_tensor(query_positions, dtype=torch.long, device=device)
    if positions.ndim != 1 or positions.numel() == 0:
        raise AttentionDiagnosticError("query_positions must be a non-empty 1-D sequence")
    if int(torch.unique(positions).numel()) != int(positions.numel()):
        raise AttentionDiagnosticError("query_positions must not contain duplicates")
    if bool(torch.any(positions < 0)) or bool(torch.any(positions >= query_length)):
        raise AttentionDiagnosticError(
            f"query_positions must lie in [0,{query_length}), got {positions.tolist()}"
        )
    if bool(torch.any(positions >= key_length)):
        raise AttentionDiagnosticError(
            "a selected query has no matching causal key position in the supplied cache"
        )
    return positions


def _validate_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[int, int, int, int, int, int]:
    _require_float_tensor("query", query, ndim=4)
    _require_float_tensor("key", key, ndim=4)
    _require_float_tensor("value", value, ndim=4)
    if key.shape != value.shape:
        raise AttentionDiagnosticError(
            f"key/value shapes differ: {tuple(key.shape)} != {tuple(value.shape)}"
        )
    batch, query_heads, query_length, head_dim = query.shape
    key_batch, kv_heads, key_length, key_head_dim = key.shape
    if batch != key_batch or head_dim != key_head_dim:
        raise AttentionDiagnosticError(
            "query and KV batch/head dimensions disagree: "
            f"Q={tuple(query.shape)}, K={tuple(key.shape)}"
        )
    if query_heads <= 0 or kv_heads <= 0 or query_heads % kv_heads:
        raise AttentionDiagnosticError(
            f"query heads ({query_heads}) must be divisible by KV heads ({kv_heads})"
        )
    if query_length <= 0 or key_length <= 0 or head_dim <= 0:
        raise AttentionDiagnosticError("Q/K/V dimensions must all be positive")
    return batch, query_heads, kv_heads, query_length, key_length, head_dim


@torch.inference_mode()
def causal_attention_output_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_positions: Sequence[int] | torch.Tensor,
    key_valid_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    query_chunk: int = 32,
) -> torch.Tensor:
    """Return explicit causal GQA output before ``o_proj`` in FP32.

    Args:
        query: Post-RoPE target queries ``[B,Hq,Tq,D]``.
        key/value: Post-RoPE key and value tensors ``[B,Hkv,Tk,D]``.
        query_positions: Absolute positions to select along ``Tq``.  Key
            positions are the contiguous range ``[0,Tk)``.
        key_valid_mask: Optional Boolean/0-1 padding mask ``[B,Tk]``.
        scaling: Attention scaling.  Defaults to ``1/sqrt(D)``.
        query_chunk: Maximum selected query positions per explicit softmax.

    The GQA grouping follows Transformers' ``repeat_kv`` ordering without
    materializing repeated K/V.  A query at position ``p`` can attend exactly
    to valid keys with position ``<=p``.
    """

    batch, query_heads, kv_heads, query_length, key_length, head_dim = _validate_qkv(
        query, key, value
    )
    if query.device != key.device or query.device != value.device:
        raise AttentionDiagnosticError(
            f"Q/K/V must share one device, got {query.device}, {key.device}, {value.device}"
        )
    if query_chunk <= 0:
        raise AttentionDiagnosticError("query_chunk must be positive")
    resolved_scaling = 1.0 / math.sqrt(head_dim) if scaling is None else float(scaling)
    if not math.isfinite(resolved_scaling) or resolved_scaling <= 0:
        raise AttentionDiagnosticError("scaling must be finite and positive")
    positions = _validated_query_positions(
        query_positions,
        query_length=query_length,
        key_length=key_length,
        device=query.device,
    )
    if key_valid_mask is None:
        valid = torch.ones((batch, key_length), dtype=torch.bool, device=query.device)
    else:
        if not isinstance(key_valid_mask, torch.Tensor) or key_valid_mask.shape != (
            batch,
            key_length,
        ):
            raise AttentionDiagnosticError(
                f"key_valid_mask must have shape {(batch, key_length)}"
            )
        if key_valid_mask.dtype == torch.bool:
            valid = key_valid_mask.to(device=query.device)
        else:
            if not bool(torch.all((key_valid_mask == 0) | (key_valid_mask == 1))):
                raise AttentionDiagnosticError("key_valid_mask must be Boolean or binary")
            valid = key_valid_mask.to(device=query.device, dtype=torch.bool)

    query_fp32 = query.index_select(2, positions).to(torch.float32)
    key_fp32 = key.to(torch.float32)
    value_fp32 = value.to(torch.float32)
    groups = query_heads // kv_heads
    # Query-head ordering is [kv_head, repeat_index], exactly as repeat_kv.
    query_grouped = query_fp32.reshape(
        batch, kv_heads, groups, positions.numel(), head_dim
    )
    outputs: list[torch.Tensor] = []
    key_positions = torch.arange(key_length, device=query.device)
    for begin in range(0, int(positions.numel()), query_chunk):
        end = min(begin + query_chunk, int(positions.numel()))
        query_part = query_grouped[:, :, :, begin:end, :]
        scores = torch.einsum("bkgqd,bkmd->bkgqm", query_part, key_fp32)
        scores.mul_(resolved_scaling)
        causal = key_positions.unsqueeze(0) <= positions[begin:end].unsqueeze(1)
        allowed = valid[:, None, None, None, :] & causal[None, None, None, :, :]
        if bool(torch.any(~allowed.any(dim=-1))):
            raise AttentionDiagnosticError("at least one selected query has no valid causal key")
        scores.masked_fill_(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        outputs.append(torch.einsum("bkgqm,bkmd->bkgqd", weights, value_fp32))
    output = torch.cat(outputs, dim=3).reshape(
        batch, query_heads, int(positions.numel()), head_dim
    )
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError("explicit attention output contains NaN or Inf")
    return output


@torch.inference_mode()
def rowwise_cosine_fp64(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    """Compute one FP64 cosine for each row with deterministic zero rules.

    Both-zero vectors have cosine 1; exactly one zero vector has cosine 0.
    The output lives on CPU so aggregation is deterministic and does not keep
    diagnostic state on model GPUs.
    """

    _require_float_tensor("reference", reference, ndim=4)
    _require_float_tensor("candidate", candidate, ndim=4)
    if reference.shape != candidate.shape:
        raise AttentionDiagnosticError(
            f"attention-output shapes differ: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    reference_fp64 = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate_fp64 = candidate.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(reference_fp64).all()) or not bool(
        torch.isfinite(candidate_fp64).all()
    ):
        raise FloatingPointError("attention outputs contain NaN or Inf")
    numerator = (reference_fp64 * candidate_fp64).sum(dim=-1)
    reference_norm = torch.linalg.vector_norm(reference_fp64, dim=-1)
    candidate_norm = torch.linalg.vector_norm(candidate_fp64, dim=-1)
    denominator = reference_norm * candidate_norm
    cosine = torch.zeros_like(numerator)
    nonzero = denominator > 0
    cosine[nonzero] = numerator[nonzero] / denominator[nonzero]
    cosine[(reference_norm == 0) & (candidate_norm == 0)] = 1.0
    # Numerical roundoff may exceed the mathematical range by a few ulps.
    return cosine.clamp_(-1.0, 1.0)


@dataclass
class LayerCosineAccumulator:
    """Streaming FP64 sufficient statistics for one target layer."""

    query_heads: int

    def __post_init__(self) -> None:
        if self.query_heads <= 0:
            raise AttentionDiagnosticError("query_heads must be positive")
        self.per_head_sum = torch.zeros(self.query_heads, dtype=torch.float64)
        self.per_head_count = torch.zeros(self.query_heads, dtype=torch.int64)
        self.cosine_min = float("inf")
        self.cosine_max = float("-inf")
        self.reference_norm_sum = 0.0
        self.candidate_norm_sum = 0.0
        self.observations = 0

    def update(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> None:
        cosine = rowwise_cosine_fp64(reference, candidate)
        # [batch, head, selected_query] -> per-head sums and equal row counts.
        if cosine.shape[1] != self.query_heads:
            raise AttentionDiagnosticError(
                f"observed {cosine.shape[1]} query heads, expected {self.query_heads}"
            )
        self.per_head_sum += cosine.sum(dim=(0, 2))
        count_per_head = int(cosine.shape[0] * cosine.shape[2])
        self.per_head_count += count_per_head
        self.cosine_min = min(self.cosine_min, float(cosine.min().item()))
        self.cosine_max = max(self.cosine_max, float(cosine.max().item()))
        reference_fp64 = reference.detach().to(device="cpu", dtype=torch.float64)
        candidate_fp64 = candidate.detach().to(device="cpu", dtype=torch.float64)
        self.reference_norm_sum += float(
            torch.linalg.vector_norm(reference_fp64, dim=-1).sum().item()
        )
        self.candidate_norm_sum += float(
            torch.linalg.vector_norm(candidate_fp64, dim=-1).sum().item()
        )
        self.observations += int(cosine.numel())

    def merge(self, other: "LayerCosineAccumulator") -> None:
        if other.query_heads != self.query_heads:
            raise AttentionDiagnosticError("cannot merge different query-head counts")
        self.per_head_sum += other.per_head_sum
        self.per_head_count += other.per_head_count
        self.cosine_min = min(self.cosine_min, other.cosine_min)
        self.cosine_max = max(self.cosine_max, other.cosine_max)
        self.reference_norm_sum += other.reference_norm_sum
        self.candidate_norm_sum += other.candidate_norm_sum
        self.observations += other.observations

    def result(self, *, layer_index: int) -> dict[str, Any]:
        if self.observations <= 0 or bool(torch.any(self.per_head_count <= 0)):
            raise AttentionDiagnosticError(f"target layer {layer_index} has no observations")
        if self.observations != int(self.per_head_count.sum().item()):
            raise AttentionDiagnosticError(
                f"target layer {layer_index} row accounting is inconsistent"
            )
        if not bool(torch.all(self.per_head_count == self.per_head_count[0])):
            raise AttentionDiagnosticError(
                f"target layer {layer_index} has unequal per-head observation counts: "
                f"{self.per_head_count.tolist()}"
            )
        per_head = self.per_head_sum / self.per_head_count.to(torch.float64)
        mean = float(self.per_head_sum.sum().item() / self.per_head_count.sum().item())
        result = {
            "target_layer": int(layer_index),
            "mean_cosine_fp64": mean,
            "min_row_cosine_fp64": self.cosine_min,
            "max_row_cosine_fp64": self.cosine_max,
            "per_query_head_mean_cosine_fp64": [float(value) for value in per_head],
            "query_heads": self.query_heads,
            "row_observations_per_query_head": int(self.per_head_count[0].item()),
            "row_observations": self.observations,
            "mean_ground_truth_attention_output_l2_fp64": (
                self.reference_norm_sum / self.observations
            ),
            "mean_mapped_attention_output_l2_fp64": (
                self.candidate_norm_sum / self.observations
            ),
            "nan_or_inf_count": 0,
        }
        for value in (
            result["mean_cosine_fp64"],
            result["min_row_cosine_fp64"],
            result["max_row_cosine_fp64"],
            result["mean_ground_truth_attention_output_l2_fp64"],
            result["mean_mapped_attention_output_l2_fp64"],
        ):
            if not math.isfinite(value):
                raise FloatingPointError("layer diagnostic contains NaN or Inf")
        return result


def aggregate_layer_accumulators(
    layers: Sequence[LayerCosineAccumulator],
) -> dict[str, Any]:
    """Finalize equal-weight layer/head/query/sample aggregation."""

    if not layers:
        raise AttentionDiagnosticError("at least one target layer is required")
    query_heads = layers[0].query_heads
    if any(layer.query_heads != query_heads for layer in layers):
        raise AttentionDiagnosticError("all target layers must use one query-head count")
    results = [layer.result(layer_index=index) for index, layer in enumerate(layers)]
    reference_counts = layers[0].per_head_count
    unequal_layers = [
        index
        for index, layer in enumerate(layers)
        if not torch.equal(layer.per_head_count, reference_counts)
    ]
    if unequal_layers:
        raise AttentionDiagnosticError(
            "target layers have unequal per-head observation counts; unequal layers="
            f"{unequal_layers[:16]}"
        )
    layer_means = torch.tensor(
        [result["mean_cosine_fp64"] for result in results], dtype=torch.float64
    )
    per_layer_per_head = torch.stack(
        [
            layer.per_head_sum / layer.per_head_count.to(torch.float64)
            for layer in layers
        ]
    )
    observations = sum(layer.observations for layer in layers)
    if observations <= 0:
        raise AttentionDiagnosticError("attention diagnostic has no observations")
    return {
        "definition": (
            "cosine(mapped-KV attention output, ground-truth-KV attention output) "
            "using fixed target ground-truth queries; headline is the arithmetic mean "
            "of target-layer means, with every target layer explicitly equal-weighted; "
            "equal per-layer/per-head row counts make sample, selected query token, and "
            "query head equal-weighted within every layer"
        ),
        "headline_aggregation": "equal_arithmetic_mean_of_target_layer_means",
        "mean_cosine_fp64": float(layer_means.mean().item()),
        "min_row_cosine_fp64": min(layer.cosine_min for layer in layers),
        "max_row_cosine_fp64": max(layer.cosine_max for layer in layers),
        "per_target_layer_mean_cosine_fp64": [float(value) for value in layer_means],
        "per_query_head_mean_cosine_fp64": [
            float(value) for value in per_layer_per_head.mean(dim=0)
        ],
        "target_layers": len(layers),
        "query_heads": query_heads,
        "row_observations_per_layer": layers[0].observations,
        "row_observations_per_layer_per_query_head": int(reference_counts[0].item()),
        "row_observations": observations,
        "aggregation_invariants": {
            "all_target_layers_equal_weight": True,
            "all_target_layers_have_identical_per_head_counts": True,
            "all_query_heads_have_identical_counts_within_each_layer": True,
        },
        "nan_or_inf_count": 0,
        "layers": results,
    }


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    value: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    if cos.ndim != 3 or sin.shape != cos.shape:
        raise AttentionDiagnosticError(
            f"position embeddings must both be [B,T,D], got {tuple(cos.shape)}, {tuple(sin.shape)}"
        )
    if value.shape[0] != cos.shape[0] or value.shape[2:] != cos.shape[1:]:
        raise AttentionDiagnosticError(
            f"position embeddings {tuple(cos.shape)} do not broadcast to {tuple(value.shape)}"
        )
    cos = cos.unsqueeze(1).to(device=value.device, dtype=value.dtype)
    sin = sin.unsqueeze(1).to(device=value.device, dtype=value.dtype)
    return value * cos + _rotate_half(value) * sin


def _kwarg_or_positional(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    name: str,
    position: int,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    if position < len(args):
        return args[position]
    raise AttentionDiagnosticError(f"attention hook did not receive {name}")


class TargetTrajectoryAttentionCosineProbe:
    """Context manager that probes every target Qwen3 attention layer.

    The target model itself is not modified.  Forward-pre-hooks reconstruct
    the exact post-q_norm/k_norm/post-RoPE Q/K and V from the current target
    hidden state, compare ground-truth and supplied mapped KV with the explicit
    kernel above, then let the original target forward continue unchanged.
    """

    def __init__(
        self,
        target_model: torch.nn.Module,
        mapped_cache: Sequence[tuple[torch.Tensor, torch.Tensor]] | Any,
        *,
        query_positions: Sequence[int],
        key_valid_mask: torch.Tensor | None = None,
        query_chunk: int = 32,
    ) -> None:
        base = getattr(target_model, "model", target_model)
        decoder_layers = getattr(base, "layers", None)
        if decoder_layers is None:
            raise AttentionDiagnosticError("target model does not expose model.layers")
        try:
            cache_pairs = tuple((key, value) for key, value in mapped_cache)
        except (TypeError, ValueError) as exc:
            raise AttentionDiagnosticError("mapped_cache is not iterable as K/V layer pairs") from exc
        if len(cache_pairs) != len(decoder_layers):
            raise AttentionDiagnosticError(
                f"mapped cache has {len(cache_pairs)} layers, target has {len(decoder_layers)}"
            )
        self.target_model = target_model
        self.decoder_layers = tuple(decoder_layers)
        self.mapped_cache = cache_pairs
        self.query_positions = tuple(int(position) for position in query_positions)
        self.key_valid_mask = key_valid_mask
        self.query_chunk = int(query_chunk)
        self.handles: list[Any] = []
        self.forward_calls = [0] * len(self.decoder_layers)
        self.accumulators: list[LayerCosineAccumulator] = []
        for index, layer in enumerate(self.decoder_layers):
            attention = getattr(layer, "self_attn", None)
            if attention is None or not all(
                hasattr(attention, name)
                for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "head_dim")
            ):
                raise AttentionDiagnosticError(
                    f"target decoder layer {index} is not a supported Qwen3 attention module"
                )
            query_heads = int(getattr(attention, "num_heads", 0))
            if query_heads <= 0:
                # Transformers 4.51 stores the value on config, not always the module.
                query_heads = int(getattr(attention.config, "num_attention_heads"))
            self.accumulators.append(LayerCosineAccumulator(query_heads=query_heads))

    def _hook(self, layer_index: int):
        def run(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden_states = _kwarg_or_positional(args, kwargs, "hidden_states", 0)
            position_embeddings = _kwarg_or_positional(args, kwargs, "position_embeddings", 1)
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                raise AttentionDiagnosticError("target hidden_states must be [B,T,hidden]")
            if not isinstance(position_embeddings, (tuple, list)) or len(position_embeddings) != 2:
                raise AttentionDiagnosticError("target position_embeddings must be (cos,sin)")
            cos, sin = position_embeddings
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, int(module.head_dim))
            query = module.q_norm(module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            true_key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            true_value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query = _apply_rotary(query, cos, sin)
            true_key = _apply_rotary(true_key, cos, sin)
            mapped_key, mapped_value = self.mapped_cache[layer_index]
            if mapped_key.device != hidden_states.device or mapped_value.device != hidden_states.device:
                raise AttentionDiagnosticError(
                    f"mapped cache layer {layer_index} is on {mapped_key.device}/{mapped_value.device}, "
                    f"target layer is on {hidden_states.device}"
                )
            expected_kv_shape = true_key.shape
            if mapped_key.shape != expected_kv_shape or mapped_value.shape != true_value.shape:
                raise AttentionDiagnosticError(
                    f"mapped cache layer {layer_index} shape mismatch: mapped "
                    f"K/V={tuple(mapped_key.shape)}/{tuple(mapped_value.shape)}, true "
                    f"K/V={tuple(true_key.shape)}/{tuple(true_value.shape)}"
                )
            scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(module.head_dim)))
            reference_output = causal_attention_output_fp32(
                query,
                true_key,
                true_value,
                query_positions=self.query_positions,
                key_valid_mask=self.key_valid_mask,
                scaling=scaling,
                query_chunk=self.query_chunk,
            )
            mapped_output = causal_attention_output_fp32(
                query,
                mapped_key,
                mapped_value,
                query_positions=self.query_positions,
                key_valid_mask=self.key_valid_mask,
                scaling=scaling,
                query_chunk=self.query_chunk,
            )
            self.accumulators[layer_index].update(reference_output, mapped_output)
            self.forward_calls[layer_index] += 1

        return run

    def __enter__(self) -> "TargetTrajectoryAttentionCosineProbe":
        if self.handles:
            raise RuntimeError("attention probe is already installed")
        for layer_index, layer in enumerate(self.decoder_layers):
            handle = layer.self_attn.register_forward_pre_hook(
                self._hook(layer_index), with_kwargs=True
            )
            self.handles.append(handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def validate_forward(self, *, expected_calls_per_layer: int = 1) -> None:
        unexpected = [
            (index, calls)
            for index, calls in enumerate(self.forward_calls)
            if calls != expected_calls_per_layer
        ]
        if unexpected:
            raise AttentionDiagnosticError(
                "target attention hook coverage mismatch: "
                + ", ".join(f"layer {index} calls={calls}" for index, calls in unexpected[:8])
            )


__all__ = [
    "AttentionDiagnosticError",
    "LayerCosineAccumulator",
    "TargetTrajectoryAttentionCosineProbe",
    "aggregate_layer_accumulators",
    "causal_attention_output_fp32",
    "rowwise_cosine_fp64",
]
