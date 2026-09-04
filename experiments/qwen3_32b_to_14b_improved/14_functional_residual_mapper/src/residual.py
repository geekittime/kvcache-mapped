from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ResidualDiagnostics:
    ood_gate: Tensor
    cap_scale: Tensor
    residual_to_anchor_rms: Tensor
    head_gate: Tensor
    fallback_mask: Tensor


def parameter_count_per_active_layer(
    *, input_dim: int, rank: int, kv_heads: int, head_dim: int
) -> int:
    """K and V combined; includes encoder, decoder, bias and head gates."""

    values = (input_dim, rank, kv_heads, head_dim)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("dimensions must be positive integers")
    per_component = input_dim * rank + kv_heads * rank * head_dim + kv_heads * head_dim + kv_heads
    return 2 * per_component


def estimated_state_bytes(parameter_count: int, bytes_per_parameter: int) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (parameter_count, bytes_per_parameter)
    ):
        raise ValueError("counts must be non-negative integers")
    return parameter_count * bytes_per_parameter


class HeadStructuredResidual(nn.Module):
    """One target layer and one cache component (K or V).

    The input encoder is shared over target KV heads; each target KV head has
    an independent decoder. The decoder and delta bias are zero-initialized,
    making step zero exactly equal to the supplied frozen anchor.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        rank: int,
        kv_heads: int,
        head_dim: int,
        feature_mean: Tensor,
        feature_inv_std: Tensor,
        rms_q99: float,
        absmax_q99: float,
        ood_close_ratio: float = 1.25,
        residual_to_anchor_rms_cap: float = 0.25,
        layer_gain: float = 1.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "input_dim": input_dim,
            "rank": rank,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in {
            "rms_q99": rms_q99,
            "absmax_q99": absmax_q99,
            "ood_close_ratio": ood_close_ratio,
            "residual_to_anchor_rms_cap": residual_to_anchor_rms_cap,
            "layer_gain": layer_gain,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if rms_q99 <= 0 or absmax_q99 <= 0:
            raise ValueError("OOD quantiles must be positive")
        if ood_close_ratio <= 1:
            raise ValueError("ood_close_ratio must be > 1")
        if not 0 < residual_to_anchor_rms_cap <= 1:
            raise ValueError("residual cap must be in (0, 1]")
        if not 0 < layer_gain <= 2:
            raise ValueError("layer_gain must be in (0, 2]")
        if tuple(feature_mean.shape) != (input_dim,) or tuple(feature_inv_std.shape) != (input_dim,):
            raise ValueError("feature statistics must have shape [input_dim]")
        if not bool(torch.isfinite(feature_mean).all()) or not bool(torch.isfinite(feature_inv_std).all()):
            raise ValueError("feature statistics must be finite")
        if not bool((feature_inv_std > 0).all()):
            raise ValueError("feature_inv_std must be strictly positive")

        self.input_dim = input_dim
        self.rank = rank
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.ood_close_ratio = float(ood_close_ratio)
        self.residual_to_anchor_rms_cap = float(residual_to_anchor_rms_cap)

        self.register_buffer("feature_mean", feature_mean.detach().clone().to(dtype=torch.float32))
        self.register_buffer("feature_inv_std", feature_inv_std.detach().clone().to(dtype=torch.float32))
        self.register_buffer("rms_q99", torch.tensor(float(rms_q99), dtype=torch.float32))
        self.register_buffer("absmax_q99", torch.tensor(float(absmax_q99), dtype=torch.float32))
        self.register_buffer("layer_gain", torch.tensor(float(layer_gain), dtype=torch.float32))

        self.encoder = nn.Parameter(torch.empty(input_dim, rank, dtype=torch.float32))
        self.decoder = nn.Parameter(torch.zeros(kv_heads, rank, head_dim, dtype=torch.float32))
        self.delta_bias = nn.Parameter(torch.zeros(kv_heads, head_dim, dtype=torch.float32))
        self.head_gate_logits = nn.Parameter(torch.zeros(kv_heads, dtype=torch.float32))
        nn.init.normal_(self.encoder, mean=0.0, std=1.0 / math.sqrt(input_dim))

    def _ood_gate(self, standardized: Tensor, finite_input: Tensor) -> Tensor:
        rms = standardized.square().mean(dim=-1).sqrt()
        absmax = standardized.abs().amax(dim=-1)
        distance = torch.maximum(rms / self.rms_q99, absmax / self.absmax_q99)
        t = ((distance - 1.0) / (self.ood_close_ratio - 1.0)).clamp(0.0, 1.0)
        smoothstep = t.square() * (3.0 - 2.0 * t)
        return torch.where(finite_input, 1.0 - smoothstep, torch.zeros_like(smoothstep))

    def forward(self, source_features: Tensor, anchor: Tensor) -> tuple[Tensor, ResidualDiagnostics]:
        if source_features.shape[-1] != self.input_dim:
            raise ValueError(f"source feature dimension must be {self.input_dim}")
        if tuple(anchor.shape[-2:]) != (self.kv_heads, self.head_dim):
            raise ValueError(f"anchor tail shape must be [{self.kv_heads}, {self.head_dim}]")
        if source_features.shape[:-1] != anchor.shape[:-2]:
            raise ValueError("source and anchor prefix dimensions must match")
        if not bool(torch.isfinite(anchor).all()):
            raise FloatingPointError("anchor contains NaN or Inf")

        compute_dtype = self.encoder.dtype
        x = source_features.to(dtype=compute_dtype)
        anchor_compute = anchor.to(dtype=compute_dtype)
        finite_input = torch.isfinite(x).all(dim=-1)
        safe_x = torch.where(torch.isfinite(x), x, self.feature_mean)
        standardized = (safe_x - self.feature_mean) * self.feature_inv_std
        ood_gate = self._ood_gate(standardized, finite_input)

        hidden = standardized.matmul(self.encoder)
        residual = torch.einsum("...r,hrd->...hd", hidden, self.decoder) + self.delta_bias
        head_gate = torch.sigmoid(self.head_gate_logits)
        residual = residual * head_gate.view(*([1] * (residual.ndim - 2)), self.kv_heads, 1)

        # The decoder starts at exact zero. Adding epsilon *inside* sqrt is
        # essential: sqrt(x).backward() at x=0 would otherwise create 0/0 and
        # poison the first functional-loss gradient even though the forward
        # value is finite.
        residual_rms = (residual.square().mean(dim=-1) + 1e-12).sqrt()
        anchor_rms = (anchor_compute.square().mean(dim=-1) + 1e-12).sqrt()
        allowed = self.residual_to_anchor_rms_cap * anchor_rms.clamp_min(torch.finfo(compute_dtype).eps)
        cap_scale = torch.minimum(torch.ones_like(residual_rms), allowed / residual_rms.clamp_min(1e-12))
        scaled = residual * cap_scale.unsqueeze(-1)
        scaled = scaled * ood_gate.unsqueeze(-1).unsqueeze(-1) * self.layer_gain
        corrected = anchor_compute + scaled
        if not bool(torch.isfinite(corrected).all()):
            raise FloatingPointError("corrected cache contains NaN or Inf")

        ratio = residual_rms / anchor_rms.clamp_min(1e-12)
        diagnostics = ResidualDiagnostics(
            ood_gate=ood_gate,
            cap_scale=cap_scale,
            residual_to_anchor_rms=ratio,
            head_gate=head_gate,
            fallback_mask=ood_gate == 0,
        )
        return corrected.to(dtype=anchor.dtype), diagnostics


class FunctionalResidualMapper(nn.Module):
    """K/V residual modules for a frozen subset of target layers."""

    def __init__(
        self,
        *,
        active_layers: Sequence[int],
        layer_loss_weights: Mapping[int, float],
        input_dim: int,
        rank: int,
        kv_heads: int,
        head_dim: int,
        statistics: Mapping[tuple[int, str], Mapping[str, object]],
        target_layers: int = 40,
        ood_close_ratio: float = 1.25,
        residual_to_anchor_rms_cap: float = 0.25,
    ) -> None:
        super().__init__()
        active = tuple(active_layers)
        if not active or len(set(active)) != len(active) or tuple(sorted(active)) != active:
            raise ValueError("active_layers must be a nonempty, unique, sorted sequence")
        if any(isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < target_layers for layer in active):
            raise ValueError("active layer outside target range")
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in layer_loss_weights):
            raise ValueError("layer_loss_weights keys must be integer layer IDs")
        if set(layer_loss_weights) != set(active):
            raise ValueError("layer_loss_weights keys must exactly match active_layers")
        for layer, weight in layer_loss_weights.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(float(weight)):
                raise ValueError(f"invalid layer weight for {layer}")
            if not 0.25 <= float(weight) <= 2.0:
                raise ValueError("layer loss weight must be in [0.25, 2.0]")

        self.active_layers = active
        self.layer_loss_weights = {int(k): float(v) for k, v in layer_loss_weights.items()}
        self.input_dim = input_dim
        self.rank = rank
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.target_layers = target_layers
        modules: dict[str, HeadStructuredResidual] = {}
        for layer in active:
            for component in ("key", "value"):
                key = (layer, component)
                if key not in statistics:
                    raise ValueError(f"missing mapper-fit statistics for {key}")
                item = statistics[key]
                modules[f"layer_{layer:02d}_{component}"] = HeadStructuredResidual(
                    input_dim=input_dim,
                    rank=rank,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    feature_mean=item["feature_mean"],  # type: ignore[arg-type]
                    feature_inv_std=item["feature_inv_std"],  # type: ignore[arg-type]
                    rms_q99=float(item["rms_q99"]),
                    absmax_q99=float(item["absmax_q99"]),
                    ood_close_ratio=ood_close_ratio,
                    residual_to_anchor_rms_cap=residual_to_anchor_rms_cap,
                    layer_gain=1.0,
                )
        self.modules_by_key = nn.ModuleDict(modules)

    def module_for(self, layer: int, component: str) -> HeadStructuredResidual:
        if component not in ("key", "value"):
            raise ValueError("component must be key or value")
        if layer not in self.active_layers:
            raise KeyError(f"layer {layer} is not active")
        return self.modules_by_key[f"layer_{layer:02d}_{component}"]

    def correct(
        self, layer: int, component: str, source_features: Tensor, anchor: Tensor
    ) -> tuple[Tensor, ResidualDiagnostics]:
        return self.module_for(layer, component)(source_features, anchor)

    def expected_parameter_count(self) -> int:
        return len(self.active_layers) * parameter_count_per_active_layer(
            input_dim=self.input_dim,
            rank=self.rank,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
