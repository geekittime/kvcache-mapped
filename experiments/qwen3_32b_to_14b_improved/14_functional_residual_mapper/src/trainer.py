from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .losses import (
    AttentionLoss,
    eos_soft_bce,
    fixed_query_attention_loss,
    suffix_kl_native_to_candidate,
    variance_normalized_huber,
)
from .residual import FunctionalResidualMapper, ResidualDiagnostics


@dataclass(frozen=True)
class LayerBatch:
    source_key: Tensor
    source_value: Tensor
    anchor_key: Tensor
    anchor_value: Tensor
    native_key: Tensor
    native_value: Tensor
    native_query: Tensor
    causal_mask: Tensor
    key_variance: Tensor
    value_variance: Tensor
    # Residual modules use [B,T,Hkv,D] because source features are token-major.
    # The attention loss consumes [B,Hkv,T,D]. A future capture adapter may
    # provide the latter directly by setting this field explicitly.
    cache_layout: str = "tokens_heads"


@dataclass(frozen=True)
class FunctionalBatch:
    layers: Mapping[int, LayerBatch]
    native_suffix_logits: Tensor
    suffix_mask: Tensor
    eos_token_ids: Sequence[int]


@dataclass(frozen=True)
class CurriculumState:
    name: str
    suffix_horizon: int
    suffix_kl_scale: float
    eos_scale: float
    kv_scale: float


@dataclass(frozen=True)
class BatchResult:
    total_loss: Tensor
    terms: Mapping[str, Tensor]
    mapped_layers: Mapping[int, Mapping[str, Tensor]]
    diagnostics: Mapping[tuple[int, str], ResidualDiagnostics]


@dataclass(frozen=True)
class TrainRecord:
    step: int
    epoch: int
    loss: float
    validation_loss: float | None
    curriculum: str


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    fraction_start: float
    fraction_end: float
    suffix_horizon: int
    suffix_kl_scale: float
    eos_scale: float
    kv_scale: float


class Curriculum:
    def __init__(self, phases: Sequence[CurriculumPhase]) -> None:
        if not phases:
            raise ValueError("curriculum must contain at least one phase")
        ordered = tuple(phases)
        if ordered[0].fraction_start != 0.0 or ordered[-1].fraction_end != 1.0:
            raise ValueError("curriculum must cover [0,1]")
        previous = 0.0
        for phase in ordered:
            if phase.fraction_start != previous or not phase.fraction_start <= phase.fraction_end:
                raise ValueError("curriculum phases must be contiguous and ordered")
            if phase.suffix_horizon not in (0, 8, 32):
                raise ValueError("suffix horizon must be 0, 8, or 32")
            for name, value in {
                "fraction_start": phase.fraction_start,
                "fraction_end": phase.fraction_end,
                "suffix_kl_scale": phase.suffix_kl_scale,
                "eos_scale": phase.eos_scale,
                "kv_scale": phase.kv_scale,
            }.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"invalid curriculum {name}")
            if min(phase.suffix_kl_scale, phase.eos_scale, phase.kv_scale) < 0:
                raise ValueError("curriculum scales must be nonnegative")
            previous = phase.fraction_end
        self.phases = ordered

    def at(self, progress: float) -> CurriculumState:
        if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not math.isfinite(float(progress)):
            raise ValueError("progress must be finite")
        value = min(1.0, max(0.0, float(progress)))
        selected = self.phases[-1]
        for phase in self.phases:
            if value < phase.fraction_end or phase is self.phases[-1]:
                selected = phase
                break
        return CurriculumState(
            name=selected.name,
            suffix_horizon=selected.suffix_horizon,
            suffix_kl_scale=float(selected.suffix_kl_scale),
            eos_scale=float(selected.eos_scale),
            kv_scale=float(selected.kv_scale),
        )


def default_curriculum() -> Curriculum:
    return Curriculum(
        (
            CurriculumPhase("attention_alignment", 0.0, 0.15, 0, 0.0, 0.0, 1.0),
            CurriculumPhase("suffix8", 0.15, 0.4, 8, 1.0, 1.0, 0.5),
            CurriculumPhase("suffix32", 0.4, 1.0, 32, 1.0, 1.0, 0.25),
        )
    )


ReceiverForward = Callable[[Mapping[int, Mapping[str, Tensor]], FunctionalBatch], Tensor]


def _attention_cache_layout(cache: Tensor, layout: str, kv_heads: int) -> Tensor:
    if cache.ndim != 4:
        raise ValueError("cache must be rank 4")
    if layout == "tokens_heads":
        if cache.shape[-2] != kv_heads:
            raise ValueError("token-major cache has wrong KV-head dimension")
        return cache.transpose(1, 2)
    if layout == "heads_tokens":
        if cache.shape[1] != kv_heads:
            raise ValueError("head-major cache has wrong KV-head dimension")
        return cache
    raise ValueError("cache_layout must be tokens_heads or heads_tokens")


class FunctionalTrainer:
    """Train only residual parameters around frozen cache and receiver targets.

    `receiver_forward` must keep the native-14B receiver weights frozen and
    return candidate suffix logits. The optimizer is constructed exclusively
    from `mapper.parameters()`, so this class never updates receiver weights.
    """

    def __init__(
        self,
        *,
        mapper: FunctionalResidualMapper,
        receiver_forward: ReceiverForward,
        optimizer: torch.optim.Optimizer,
        curriculum: Curriculum | None = None,
        attention_logit_weight: float = 0.25,
        attention_output_cosine_weight: float = 1.0,
        attention_output_nmse_weight: float = 0.25,
        suffix_kl_weight: float = 1.0,
        eos_weight: float = 0.1,
        kv_weight: float = 0.02,
        head_gate_l1_weight: float = 0.0001,
        temperatures: Sequence[float] = (1.0, 2.0),
        gradient_clip_norm: float = 1.0,
    ) -> None:
        self.mapper = mapper
        self.receiver_forward = receiver_forward
        self.optimizer = optimizer
        mapper_parameter_ids = {id(parameter) for parameter in mapper.parameters()}
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group.get("params", ())
        }
        if optimizer_parameter_ids != mapper_parameter_ids:
            raise ValueError("optimizer must contain exactly the residual mapper parameters")
        self.curriculum = curriculum or default_curriculum()
        self.weights = {
            "attention_logit": float(attention_logit_weight),
            "attention_output_cosine": float(attention_output_cosine_weight),
            "attention_output_nmse": float(attention_output_nmse_weight),
            "suffix_kl": float(suffix_kl_weight),
            "eos": float(eos_weight),
            "kv": float(kv_weight),
            "head_gate_l1": float(head_gate_l1_weight),
        }
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("loss weights must be finite and nonnegative")
        if not temperatures or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0
            for value in temperatures
        ):
            raise ValueError("temperatures must be finite and positive")
        if not math.isfinite(float(gradient_clip_norm)) or gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be finite and positive")
        self.temperatures = tuple(float(value) for value in temperatures)
        self.gradient_clip_norm = float(gradient_clip_norm)

    def _validate_batch(self, batch: FunctionalBatch) -> None:
        if set(batch.layers) != set(self.mapper.active_layers):
            raise ValueError("batch layers must exactly match mapper active_layers")
        if batch.suffix_mask.dtype != torch.bool or batch.suffix_mask.ndim != 2:
            raise ValueError("suffix_mask must be boolean [batch,time]")
        if batch.native_suffix_logits.ndim != 3 or tuple(batch.native_suffix_logits.shape[:2]) != tuple(batch.suffix_mask.shape):
            raise ValueError("native suffix logits and mask shape mismatch")
        if not batch.eos_token_ids:
            raise ValueError("eos_token_ids cannot be empty")

    def compute_batch(self, batch: FunctionalBatch, *, progress: float) -> BatchResult:
        self._validate_batch(batch)
        state = self.curriculum.at(progress)
        mapped: dict[int, dict[str, Tensor]] = {}
        diagnostics: dict[tuple[int, str], ResidualDiagnostics] = {}
        attention_terms: list[Tensor] = []
        kv_terms: list[Tensor] = []
        weighted_layers: list[float] = []
        for layer in self.mapper.active_layers:
            item = batch.layers[layer]
            mapped_key, key_diag = self.mapper.correct(layer, "key", item.source_key, item.anchor_key)
            mapped_value, value_diag = self.mapper.correct(layer, "value", item.source_value, item.anchor_value)
            mapped[layer] = {"key": mapped_key, "value": mapped_value}
            diagnostics[(layer, "key")] = key_diag
            diagnostics[(layer, "value")] = value_diag
            attention = fixed_query_attention_loss(
                native_query=item.native_query,
                native_key=_attention_cache_layout(item.native_key, item.cache_layout, self.mapper.kv_heads),
                native_value=_attention_cache_layout(item.native_value, item.cache_layout, self.mapper.kv_heads),
                candidate_key=_attention_cache_layout(mapped_key, item.cache_layout, self.mapper.kv_heads),
                candidate_value=_attention_cache_layout(mapped_value, item.cache_layout, self.mapper.kv_heads),
                causal_mask=item.causal_mask,
            )
            layer_weight = self.mapper.layer_loss_weights[layer]
            attention_terms.append(
                torch.stack(
                    [
                        attention.centered_logit_huber * self.weights["attention_logit"],
                        attention.output_cosine * self.weights["attention_output_cosine"],
                        attention.output_nmse * self.weights["attention_output_nmse"],
                    ]
                ).sum()
                * layer_weight
            )
            kv_terms.append(
                (
                    variance_normalized_huber(
                        candidate=mapped_key,
                        native=item.native_key,
                        variance=item.key_variance,
                    )
                    + variance_normalized_huber(
                        candidate=mapped_value,
                        native=item.native_value,
                        variance=item.value_variance,
                    )
                )
                * 0.5
                * layer_weight
            )
            weighted_layers.append(layer_weight)
        layer_normalizer = sum(weighted_layers)
        attention_loss = torch.stack(attention_terms).sum() / layer_normalizer
        kv_loss = torch.stack(kv_terms).sum() / layer_normalizer

        candidate_suffix_logits = self.receiver_forward(mapped, batch)
        if not isinstance(candidate_suffix_logits, Tensor) or not bool(torch.isfinite(candidate_suffix_logits).all()):
            raise FloatingPointError("receiver_forward returned non-finite suffix logits")
        if candidate_suffix_logits.shape != batch.native_suffix_logits.shape:
            raise ValueError("receiver_forward returned unexpected suffix-logit shape")
        if state.suffix_horizon:
            horizon_mask = batch.suffix_mask.clone()
            horizon_mask[:, state.suffix_horizon :] = False
            suffix_loss = suffix_kl_native_to_candidate(
                native_logits=batch.native_suffix_logits,
                candidate_logits=candidate_suffix_logits,
                suffix_mask=horizon_mask,
                temperatures=self.temperatures,
            )
            eos_loss = eos_soft_bce(
                native_logits=batch.native_suffix_logits,
                candidate_logits=candidate_suffix_logits,
                suffix_mask=horizon_mask,
                eos_token_ids=batch.eos_token_ids,
            )
        else:
            suffix_loss = candidate_suffix_logits.sum() * 0.0
            eos_loss = candidate_suffix_logits.sum() * 0.0

        gate_values = [
            torch.sigmoid(module.head_gate_logits)
            for module in self.mapper.modules_by_key.values()
        ]
        gate_l1 = torch.cat(gate_values).mean() if gate_values else candidate_suffix_logits.sum() * 0.0
        total = (
            attention_loss
            + self.weights["kv"] * state.kv_scale * kv_loss
            + self.weights["suffix_kl"] * state.suffix_kl_scale * suffix_loss
            + self.weights["eos"] * state.eos_scale * eos_loss
            + self.weights["head_gate_l1"] * gate_l1
        )
        if not bool(torch.isfinite(total)):
            raise FloatingPointError("total functional loss is non-finite")
        terms = {
            "attention": attention_loss,
            "kv_guardrail": kv_loss,
            "suffix_kl": suffix_loss,
            "eos_bce": eos_loss,
            "head_gate_l1": gate_l1,
            "total": total,
        }
        return BatchResult(total, terms, mapped, diagnostics)

    def train_step(self, batch: FunctionalBatch, *, progress: float) -> BatchResult:
        self.mapper.train()
        self.optimizer.zero_grad(set_to_none=True)
        result = self.compute_batch(batch, progress=progress)
        result.total_loss.backward()
        for parameter in self.mapper.parameters():
            if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError("non-finite residual gradient")
        torch.nn.utils.clip_grad_norm_(self.mapper.parameters(), self.gradient_clip_norm, error_if_nonfinite=True)
        self.optimizer.step()
        for parameter in self.mapper.parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError("non-finite residual parameter after optimizer step")
        return result

    @torch.no_grad()
    def evaluate_batch(self, batch: FunctionalBatch, *, progress: float = 1.0) -> BatchResult:
        self.mapper.eval()
        return self.compute_batch(batch, progress=progress)

    def fit(
        self,
        train_batches: Sequence[FunctionalBatch],
        validation_batches: Sequence[FunctionalBatch],
        *,
        epochs: int,
        early_stopping_patience: int | None = None,
    ) -> list[TrainRecord]:
        if not train_batches or not validation_batches:
            raise ValueError("train and validation batches must be nonempty")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if early_stopping_patience is not None and (
            isinstance(early_stopping_patience, bool)
            or not isinstance(early_stopping_patience, int)
            or early_stopping_patience <= 0
        ):
            raise ValueError("early_stopping_patience must be positive")
        records: list[TrainRecord] = []
        best = math.inf
        stale = 0
        total_steps = epochs * len(train_batches)
        step = 0
        for epoch in range(epochs):
            for batch in train_batches:
                progress = step / max(total_steps - 1, 1)
                result = self.train_step(batch, progress=progress)
                validation = [self.evaluate_batch(item).total_loss.detach() for item in validation_batches]
                validation_loss = float(torch.stack(validation).mean().cpu())
                loss = float(result.total_loss.detach().cpu())
                records.append(
                    TrainRecord(
                        step=step,
                        epoch=epoch,
                        loss=loss,
                        validation_loss=validation_loss,
                        curriculum=self.curriculum.at(progress).name,
                    )
                )
                if validation_loss < best - 1e-12:
                    best = validation_loss
                    stale = 0
                else:
                    stale += 1
                step += 1
                if early_stopping_patience is not None and stale >= early_stopping_patience:
                    return records
        return records
