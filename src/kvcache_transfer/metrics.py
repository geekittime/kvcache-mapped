"""FP64 reconstruction metrics for KV-cache mapper analysis.

The layer-selection score in the paper treats every target channel equally:
an independent :math:`R^2` is calculated for each channel and those values are
uniformly averaged.  This differs from pooling all residuals, which weights
high-variance channels more heavily.  :func:`r2_diagnostics` returns both the
paper-aligned score and pooled diagnostics so the two cannot be confused.

The streaming accumulator uses parallel Welford updates for target variance.
It therefore avoids the cancellation-prone ``sum(y**2) - sum(y)**2 / n`` raw
moment formula while keeping all analysis arithmetic in FP64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ConstantPolicy = Literal["finite", "nan"]


class MetricInputError(ValueError):
    """Raised when reconstruction metric inputs have incompatible layouts."""


def _channel_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 0:
        raise MetricInputError(f"{name} must contain a sample dimension")
    if value.numel() == 0 or value.shape[0] == 0:
        raise MetricInputError(f"{name} must be non-empty")
    if value.ndim == 1:
        return value.reshape(-1, 1)
    return value.reshape(value.shape[0], -1)


def _validate_pair(y_true: torch.Tensor, y_pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    true = _channel_matrix(y_true, "y_true")
    pred = _channel_matrix(y_pred, "y_pred")
    if true.shape != pred.shape:
        raise MetricInputError(
            f"y_true and y_pred must have identical shape, got {tuple(true.shape)} "
            f"and {tuple(pred.shape)}"
        )
    return true, pred


def _r2_from_sums(
    sse: torch.Tensor, sst: torch.Tensor, *, constant_policy: ConstantPolicy
) -> torch.Tensor:
    if constant_policy not in ("finite", "nan"):
        raise MetricInputError(f"unknown constant-channel policy: {constant_policy!r}")
    nonconstant = sst != 0
    result = torch.empty_like(sst, dtype=torch.float64)
    result[nonconstant] = 1.0 - sse[nonconstant] / sst[nonconstant]
    if constant_policy == "nan":
        result[~nonconstant] = torch.nan
    else:
        # This is sklearn's force_finite behavior: an exact constant-channel
        # prediction is perfect; any nonzero error receives zero rather than
        # an undefined or -infinite score.
        result[~nonconstant] = torch.where(
            sse[~nonconstant] == 0,
            torch.ones_like(sse[~nonconstant]),
            torch.zeros_like(sse[~nonconstant]),
        )
    return result


@dataclass(frozen=True)
class R2Diagnostics:
    """Paper-aligned per-channel scores plus pooled reconstruction diagnostics.

    Attributes:
        per_channel: One :math:`R^2` value per flattened target channel.
        uniform_average: Arithmetic mean of ``per_channel``.  This is the
            score that must be used for source-layer ranking.
        global_r2: ``1 - sum(channel SSE) / sum(channel-centered SST)``.  This
            is a variance-weighted diagnostic and must not select layers.
        flattened_global_r2: Diagnostic obtained by flattening every target
            value and centering around one scalar mean.
        sse_per_channel: FP64 residual sum of squares for each channel.
        sst_per_channel: FP64 target total sum of squares around each channel's
            own mean.
        global_sse: Sum of all elements of ``sse_per_channel``.
        global_sst: Sum of all elements of ``sst_per_channel``.
        flattened_global_sst: Target SST around the single flattened mean.
    """

    n_samples: int
    n_channels: int
    per_channel: torch.Tensor
    uniform_average: float
    global_r2: float
    flattened_global_r2: float
    sse_per_channel: torch.Tensor
    sst_per_channel: torch.Tensor
    global_sse: float
    global_sst: float
    flattened_global_sst: float

    @property
    def mse(self) -> float:
        """Mean squared error over every sample and target channel."""

        return self.global_sse / (self.n_samples * self.n_channels)


class R2Accumulator:
    """Streaming FP64 accumulator for uniform and pooled :math:`R^2` metrics."""

    def __init__(
        self,
        *,
        analysis_device: torch.device | str = "cpu",
        check_finite: bool = True,
    ) -> None:
        self.analysis_device = torch.device(analysis_device)
        self.check_finite = check_finite
        self._n_samples = 0
        self._n_elements = 0
        self._mean: torch.Tensor | None = None
        self._m2: torch.Tensor | None = None
        self._sse: torch.Tensor | None = None
        self._flat_mean = torch.zeros((), dtype=torch.float64, device=self.analysis_device)
        self._flat_m2 = torch.zeros((), dtype=torch.float64, device=self.analysis_device)

    @torch.no_grad()
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """Merge a same-shaped, non-empty prediction chunk."""

        true, pred = _validate_pair(y_true, y_pred)
        true64 = true.to(device=self.analysis_device, dtype=torch.float64)
        pred64 = pred.to(device=self.analysis_device, dtype=torch.float64)
        if self.check_finite and (not torch.isfinite(true64).all() or not torch.isfinite(pred64).all()):
            raise MetricInputError("R2 inputs contain NaN or Inf")

        if self._mean is None:
            channels = true64.shape[1]
            self._mean = torch.zeros(channels, dtype=torch.float64, device=self.analysis_device)
            self._m2 = torch.zeros(channels, dtype=torch.float64, device=self.analysis_device)
            self._sse = torch.zeros(channels, dtype=torch.float64, device=self.analysis_device)
        elif true64.shape[1] != self._mean.numel():
            raise MetricInputError(
                f"channel count changed between chunks: expected {self._mean.numel()}, "
                f"got {true64.shape[1]}"
            )

        assert self._m2 is not None and self._sse is not None
        chunk_n = true64.shape[0]
        total_n = self._n_samples + chunk_n
        chunk_mean = true64.mean(dim=0)
        chunk_delta = true64 - chunk_mean
        chunk_m2 = torch.sum(chunk_delta * chunk_delta, dim=0)
        mean_delta = chunk_mean - self._mean
        merge_scale = float(self._n_samples * chunk_n) / float(total_n)
        self._m2.add_(chunk_m2 + mean_delta.square() * merge_scale)
        self._mean.add_(mean_delta * (float(chunk_n) / float(total_n)))
        self._sse.add_(torch.sum((true64 - pred64).square(), dim=0))
        self._n_samples = total_n

        # Maintain a second Welford state in which all channels are flattened
        # around one scalar mean.  This is a diagnostic, not the paper score.
        flat = true64.reshape(-1)
        chunk_elements = flat.numel()
        total_elements = self._n_elements + chunk_elements
        flat_chunk_mean = flat.mean()
        flat_chunk_m2 = torch.sum((flat - flat_chunk_mean).square())
        flat_delta = flat_chunk_mean - self._flat_mean
        flat_scale = float(self._n_elements * chunk_elements) / float(total_elements)
        self._flat_m2.add_(flat_chunk_m2 + flat_delta.square() * flat_scale)
        self._flat_mean.add_(flat_delta * (float(chunk_elements) / float(total_elements)))
        self._n_elements = total_elements

    def finalize(self, *, constant_policy: ConstantPolicy = "finite") -> R2Diagnostics:
        """Return immutable diagnostics after at least one update."""

        if self._mean is None or self._m2 is None or self._sse is None:
            raise MetricInputError("cannot finalize R2 metrics without observations")
        per_channel = _r2_from_sums(
            self._sse, self._m2, constant_policy=constant_policy
        )
        if constant_policy == "nan":
            uniform_tensor = torch.mean(per_channel)
        else:
            uniform_tensor = per_channel.mean()

        global_sse_tensor = self._sse.sum()
        global_sst_tensor = self._m2.sum()
        global_r2_tensor = _r2_from_sums(
            global_sse_tensor.reshape(1),
            global_sst_tensor.reshape(1),
            constant_policy=constant_policy,
        )[0]
        flattened_r2_tensor = _r2_from_sums(
            global_sse_tensor.reshape(1),
            self._flat_m2.reshape(1),
            constant_policy=constant_policy,
        )[0]
        return R2Diagnostics(
            n_samples=self._n_samples,
            n_channels=self._mean.numel(),
            per_channel=per_channel.clone(),
            uniform_average=float(uniform_tensor.item()),
            global_r2=float(global_r2_tensor.item()),
            flattened_global_r2=float(flattened_r2_tensor.item()),
            sse_per_channel=self._sse.clone(),
            sst_per_channel=self._m2.clone(),
            global_sse=float(global_sse_tensor.item()),
            global_sst=float(global_sst_tensor.item()),
            flattened_global_sst=float(self._flat_m2.item()),
        )


def r2_diagnostics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    constant_policy: ConstantPolicy = "finite",
    analysis_device: torch.device | str = "cpu",
) -> R2Diagnostics:
    """Compute all paper-aligned and pooled :math:`R^2` diagnostics in FP64."""

    accumulator = R2Accumulator(analysis_device=analysis_device)
    accumulator.update(y_true, y_pred)
    return accumulator.finalize(constant_policy=constant_policy)


def uniform_average_r2(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    constant_policy: ConstantPolicy = "finite",
    analysis_device: torch.device | str = "cpu",
) -> float:
    """Return the paper's equal-per-channel :math:`R^2` layer-selection score."""

    return r2_diagnostics(
        y_true,
        y_pred,
        constant_policy=constant_policy,
        analysis_device=analysis_device,
    ).uniform_average


def r2_uniform_average(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    constant_policy: ConstantPolicy = "finite",
    analysis_device: torch.device | str = "cpu",
) -> float:
    """Compatibility spelling for the paper's uniform-average channel score."""

    return uniform_average_r2(
        target,
        prediction,
        constant_policy=constant_policy,
        analysis_device=analysis_device,
    )


def global_r2(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    constant_policy: ConstantPolicy = "finite",
    analysis_device: torch.device | str = "cpu",
) -> float:
    """Return pooled SSE/SST :math:`R^2` for diagnostics, never layer ranking."""

    return r2_diagnostics(
        y_true,
        y_pred,
        constant_policy=constant_policy,
        analysis_device=analysis_device,
    ).global_r2


def mean_squared_error(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    analysis_device: torch.device | str = "cpu",
) -> float:
    """Compute elementwise MSE in FP64."""

    true, pred = _validate_pair(y_true, y_pred)
    true64 = true.to(device=analysis_device, dtype=torch.float64)
    pred64 = pred.to(device=analysis_device, dtype=torch.float64)
    if not torch.isfinite(true64).all() or not torch.isfinite(pred64).all():
        raise MetricInputError("MSE inputs contain NaN or Inf")
    return float(torch.mean((true64 - pred64).square()).item())


def mean_row_cosine_similarity(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    analysis_device: torch.device | str = "cpu",
) -> float:
    """Compute mean sample-wise cosine similarity in FP64.

    Target dimensions after axis zero are flattened into a feature vector.
    Two all-zero rows are treated as identical (cosine 1); a single zero row
    receives cosine 0.  Real KV activations are nonzero, but the explicit rule
    makes synthetic and failure-case diagnostics deterministic.
    """

    true, pred = _validate_pair(y_true, y_pred)
    true64 = true.to(device=analysis_device, dtype=torch.float64)
    pred64 = pred.to(device=analysis_device, dtype=torch.float64)
    if not torch.isfinite(true64).all() or not torch.isfinite(pred64).all():
        raise MetricInputError("cosine inputs contain NaN or Inf")
    numerator = torch.sum(true64 * pred64, dim=1)
    true_norm = torch.linalg.vector_norm(true64, dim=1)
    pred_norm = torch.linalg.vector_norm(pred64, dim=1)
    denominator = true_norm * pred_norm
    cosine = torch.zeros_like(numerator)
    nonzero = denominator != 0
    cosine[nonzero] = numerator[nonzero] / denominator[nonzero]
    both_zero = (true_norm == 0) & (pred_norm == 0)
    cosine[both_zero] = 1.0
    return float(cosine.mean().item())


__all__ = [
    "ConstantPolicy",
    "MetricInputError",
    "R2Accumulator",
    "R2Diagnostics",
    "global_r2",
    "mean_row_cosine_similarity",
    "mean_squared_error",
    "r2_uniform_average",
    "r2_diagnostics",
    "uniform_average_r2",
]
