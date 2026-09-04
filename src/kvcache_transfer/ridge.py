r"""Closed-form, centered ridge regression used by the KV-cache mapper.

The paper fits an affine mapper after centering both the source and target
activations.  For source features ``X`` and target features ``Y`` this module
implements exactly

.. math::

   W = (X_c^T X_c + \lambda I)^{-1} X_c^T Y_c,
   \qquad b = \bar{Y} - \bar{X}W.

Importantly, ``X_c.T @ X_c`` is a *sum*, not a covariance divided by the
number of observations.  Consequently the configured ``ridge_lambda`` is
added directly and the intercept is never regularized.

Production-sized feature matrices do not fit in memory.  The public
accumulators therefore support the paper's numerically preferable two-pass
algorithm: first estimate means, then stream the data again to accumulate
centered sufficient statistics.  Inputs may be BF16, but all fitting
arithmetic is explicitly performed in FP32 by default.  Expensive diagnostic
analysis can be requested separately in FP64.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Callable, Iterable, Iterator, Literal, Sequence

import torch


TensorPair = tuple[torch.Tensor, torch.Tensor]
ChunkFactory = Callable[[], Iterable[TensorPair]]
Solver = Literal["cholesky", "lstsq"]


class RidgeInputError(ValueError):
    """Raised when streamed source/target activation chunks are inconsistent."""


class RidgeSolveError(RuntimeError):
    """Raised when the requested ridge solver cannot solve the normal equations."""

    def __init__(self, message: str, *, cholesky_info: int | None = None) -> None:
        super().__init__(message)
        self.cholesky_info = cholesky_info


def _as_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim != 2:
        raise RidgeInputError(f"{name} must be rank 2 [samples, features], got {tuple(value.shape)}")
    if value.shape[0] == 0:
        raise RidgeInputError(f"{name} contains an empty sample chunk")
    if value.shape[1] == 0:
        raise RidgeInputError(f"{name} contains zero features")
    return value


def _validate_pair(x: torch.Tensor, y: torch.Tensor) -> TensorPair:
    x = _as_matrix(x, "x")
    y = _as_matrix(y, "y")
    if x.shape[0] != y.shape[0]:
        raise RidgeInputError(
            f"x and y must have the same sample count, got {x.shape[0]} and {y.shape[0]}"
        )
    return x, y


@contextmanager
def _cuda_tf32_mode(enabled: bool, device: torch.device) -> Iterator[None]:
    """Temporarily select TF32 behavior without leaking process-global state."""

    if device.type != "cuda":
        yield
        return

    old_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_value


@dataclass(frozen=True)
class FeatureMeans:
    """FP32 feature means produced by the first streaming pass.

    Attributes:
        n_samples: Number of observations contributing to the means.
        x: Source-feature mean with shape ``[source_features]``.
        y: Target-feature mean with shape ``[target_features]``.
    """

    n_samples: int
    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self) -> None:
        if self.n_samples <= 0:
            raise RidgeInputError("FeatureMeans.n_samples must be positive")
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise RidgeInputError("FeatureMeans tensors must both be rank 1")
        if self.x.dtype != torch.float32 or self.y.dtype != torch.float32:
            raise RidgeInputError("FeatureMeans must be stored in torch.float32")
        if self.x.device != self.y.device:
            raise RidgeInputError("FeatureMeans x and y must reside on the same device")

    @property
    def device(self) -> torch.device:
        """Device on which the fitting statistics are accumulated."""

        return self.x.device


class StreamingFeatureMeans:
    """Numerically stable first-pass accumulator for source and target means.

    Chunk means are merged using the parallel/online mean formula rather than
    accumulating one very large raw sum.  All chunks are moved to
    ``compute_device`` and converted to FP32 before reduction.
    """

    def __init__(self, compute_device: torch.device | str = "cpu") -> None:
        self.compute_device = torch.device(compute_device)
        self._n_samples = 0
        self._mean_x: torch.Tensor | None = None
        self._mean_y: torch.Tensor | None = None

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Merge one non-empty pair of ``[samples, features]`` chunks."""

        x, y = _validate_pair(x, y)
        x32 = x.to(device=self.compute_device, dtype=torch.float32)
        y32 = y.to(device=self.compute_device, dtype=torch.float32)

        if self._mean_x is None:
            self._mean_x = torch.zeros(x32.shape[1], dtype=torch.float32, device=self.compute_device)
            self._mean_y = torch.zeros(y32.shape[1], dtype=torch.float32, device=self.compute_device)
        elif x32.shape[1] != self._mean_x.numel() or y32.shape[1] != self._mean_y.numel():
            raise RidgeInputError(
                "feature dimensions changed between chunks: "
                f"expected ({self._mean_x.numel()}, {self._mean_y.numel()}), "
                f"got ({x32.shape[1]}, {y32.shape[1]})"
            )

        chunk_n = x32.shape[0]
        total_n = self._n_samples + chunk_n
        ratio = float(chunk_n) / float(total_n)
        self._mean_x.add_((x32.mean(dim=0) - self._mean_x) * ratio)
        self._mean_y.add_((y32.mean(dim=0) - self._mean_y) * ratio)
        self._n_samples = total_n

    def finalize(self) -> FeatureMeans:
        """Return immutable means, failing if no samples were observed."""

        if self._mean_x is None or self._mean_y is None or self._n_samples == 0:
            raise RidgeInputError("cannot finalize feature means without observations")
        return FeatureMeans(
            n_samples=self._n_samples,
            x=self._mean_x.clone(),
            y=self._mean_y.clone(),
        )


@dataclass(frozen=True)
class CenteredRidgeStatistics:
    """Centered sufficient statistics used by the normal equations.

    ``xtx`` is ``sum(Xc.T @ Xc)`` and ``xty`` is ``sum(Xc.T @ Yc)``.
    Neither tensor is normalized by ``n_samples``.
    """

    n_samples: int
    mean_x: torch.Tensor
    mean_y: torch.Tensor
    xtx: torch.Tensor
    xty: torch.Tensor

    def __post_init__(self) -> None:
        source_features = self.mean_x.numel()
        target_features = self.mean_y.numel()
        if self.n_samples <= 0:
            raise RidgeInputError("statistics must contain at least one sample")
        if self.mean_x.ndim != 1 or self.mean_y.ndim != 1:
            raise RidgeInputError("mean_x and mean_y must be rank 1")
        if self.xtx.shape != (source_features, source_features):
            raise RidgeInputError(
                f"xtx has shape {tuple(self.xtx.shape)}, expected "
                f"({source_features}, {source_features})"
            )
        if self.xty.shape != (source_features, target_features):
            raise RidgeInputError(
                f"xty has shape {tuple(self.xty.shape)}, expected "
                f"({source_features}, {target_features})"
            )
        tensors = (self.mean_x, self.mean_y, self.xtx, self.xty)
        if any(t.dtype != torch.float32 for t in tensors):
            raise RidgeInputError("production fitting statistics must all be torch.float32")
        if len({t.device for t in tensors}) != 1:
            raise RidgeInputError("all statistics must reside on the same device")

    @property
    def device(self) -> torch.device:
        """Device on which the sufficient statistics reside."""

        return self.xtx.device

    @property
    def source_features(self) -> int:
        return self.mean_x.numel()

    @property
    def target_features(self) -> int:
        return self.mean_y.numel()


class CenteredRidgeAccumulator:
    """Second-pass FP32 accumulator for centered normal-equation statistics."""

    def __init__(self, means: FeatureMeans, *, allow_tf32: bool = False) -> None:
        self.means = means
        self.allow_tf32 = allow_tf32
        source_features = means.x.numel()
        target_features = means.y.numel()
        self._xtx = torch.zeros(
            (source_features, source_features), dtype=torch.float32, device=means.device
        )
        self._xty = torch.zeros(
            (source_features, target_features), dtype=torch.float32, device=means.device
        )
        self._n_samples = 0

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Accumulate one source/target chunk after subtracting fixed means."""

        x, y = _validate_pair(x, y)
        if x.shape[1] != self.means.x.numel() or y.shape[1] != self.means.y.numel():
            raise RidgeInputError(
                "chunk feature dimensions do not match first-pass means: "
                f"expected ({self.means.x.numel()}, {self.means.y.numel()}), "
                f"got ({x.shape[1]}, {y.shape[1]})"
            )

        x_centered = x.to(device=self.means.device, dtype=torch.float32) - self.means.x
        y_centered = y.to(device=self.means.device, dtype=torch.float32) - self.means.y
        with _cuda_tf32_mode(self.allow_tf32, self.means.device):
            self._xtx.addmm_(x_centered.T, x_centered)
            self._xty.addmm_(x_centered.T, y_centered)
        self._n_samples += x.shape[0]

    def finalize(self) -> CenteredRidgeStatistics:
        """Return statistics and verify both passes saw the same sample count."""

        if self._n_samples != self.means.n_samples:
            raise RidgeInputError(
                "first and second passes observed different sample counts: "
                f"{self.means.n_samples} versus {self._n_samples}"
            )
        return CenteredRidgeStatistics(
            n_samples=self._n_samples,
            mean_x=self.means.x.clone(),
            mean_y=self.means.y.clone(),
            xtx=self._xtx,
            xty=self._xty,
        )


@dataclass(frozen=True)
class RidgeSolution:
    """Affine ridge mapper with weight layout ``[source, target]``."""

    weight: torch.Tensor
    bias: torch.Tensor
    ridge_lambda: float
    n_samples: int
    solver: Solver

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.bias.ndim != 1:
            raise RidgeInputError("weight must be rank 2 and bias must be rank 1")
        if self.weight.shape[1] != self.bias.numel():
            raise RidgeInputError("weight output dimension and bias length do not match")
        if self.weight.dtype != torch.float32 or self.bias.dtype != torch.float32:
            raise RidgeInputError("ridge solutions must be stored in FP32")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the affine mapper in FP32.

        Callers serving a BF16 cache should cast the returned tensor only after
        this FP32 matmul, matching the primary reproduction protocol.
        """

        x = _as_matrix(x, "x")
        if x.shape[1] != self.weight.shape[0]:
            raise RidgeInputError(
                f"x has {x.shape[1]} features, mapper expects {self.weight.shape[0]}"
            )
        x32 = x.to(device=self.weight.device, dtype=torch.float32)
        return torch.addmm(self.bias, x32, self.weight)


def estimate_feature_means(
    chunks: Iterable[TensorPair], *, compute_device: torch.device | str = "cpu"
) -> FeatureMeans:
    """Run the first pass over a stream of paired activation chunks."""

    accumulator = StreamingFeatureMeans(compute_device=compute_device)
    for x, y in chunks:
        accumulator.update(x, y)
    return accumulator.finalize()


def accumulate_centered_statistics(
    chunks: Iterable[TensorPair],
    means: FeatureMeans,
    *,
    allow_tf32: bool = False,
) -> CenteredRidgeStatistics:
    """Run the second pass and return unnormalized centered statistics."""

    accumulator = CenteredRidgeAccumulator(means, allow_tf32=allow_tf32)
    for x, y in chunks:
        accumulator.update(x, y)
    return accumulator.finalize()


@torch.no_grad()
def solve_centered_ridge(
    statistics: CenteredRidgeStatistics,
    ridge_lambda: float = 0.01,
    *,
    solver: Solver = "cholesky",
) -> RidgeSolution:
    """Solve centered ridge normal equations without regularizing the bias.

    Args:
        statistics: Unnormalized, centered FP32 sufficient statistics.
        ridge_lambda: Non-negative value added directly to the Gram diagonal.
            It is deliberately **not** divided by ``n_samples``.
        solver: ``"cholesky"`` is this reproduction's production solver for
            the paper's closed-form system; the paper does not publish its
            numerical solver. ``"lstsq"`` is an explicit diagnostic fallback
            and is never chosen silently after a Cholesky failure.

    Raises:
        RidgeSolveError: If Cholesky reports that the regularized Gram matrix
            is not positive definite.  The exception exposes ``cholesky_info``
            so the caller can record the failure before explicitly retrying a
            more stable solver.
    """

    if not math.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise RidgeInputError("ridge_lambda must be finite and non-negative")
    if solver not in ("cholesky", "lstsq"):
        raise RidgeInputError(f"unknown ridge solver: {solver!r}")

    system = statistics.xtx.clone()
    system.diagonal().add_(float(ridge_lambda))

    if solver == "cholesky":
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        info_value = int(info.max().item())
        if info_value != 0:
            raise RidgeSolveError(
                "Cholesky failed for the centered ridge system; record matrix "
                "diagnostics before explicitly selecting solver='lstsq'",
                cholesky_info=info_value,
            )
        weight = torch.cholesky_solve(statistics.xty, factor)
    else:
        weight = torch.linalg.lstsq(system, statistics.xty).solution

    weight = weight.to(dtype=torch.float32)
    bias = statistics.mean_y - statistics.mean_x @ weight
    return RidgeSolution(
        weight=weight,
        bias=bias.to(dtype=torch.float32),
        ridge_lambda=float(ridge_lambda),
        n_samples=statistics.n_samples,
        solver=solver,
    )


def fit_centered_ridge_streaming(
    chunk_factory: ChunkFactory,
    ridge_lambda: float = 0.01,
    *,
    compute_device: torch.device | str = "cpu",
    allow_tf32: bool = False,
    solver: Solver = "cholesky",
) -> tuple[RidgeSolution, CenteredRidgeStatistics]:
    """Fit an affine mapper from a replayable chunk factory in two passes.

    ``chunk_factory`` is called exactly twice and must yield the same samples
    in the same feature layout on each call.  A factory is required instead of
    a one-shot iterator so production mmap data can be scanned twice without
    materializing it.
    """

    if not callable(chunk_factory):
        raise RidgeInputError("chunk_factory must be callable and replayable")
    means = estimate_feature_means(chunk_factory(), compute_device=compute_device)
    statistics = accumulate_centered_statistics(
        chunk_factory(), means, allow_tf32=allow_tf32
    )
    solution = solve_centered_ridge(statistics, ridge_lambda, solver=solver)
    return solution, statistics


def fit_centered_ridge(
    x: torch.Tensor,
    y: torch.Tensor,
    ridge_lambda: float = 0.01,
    *,
    compute_device: torch.device | str | None = None,
    allow_tf32: bool = False,
    solver: Solver = "cholesky",
) -> RidgeSolution:
    """Convenience wrapper for fitting a materialized source/target matrix."""

    x, y = _validate_pair(x, y)
    device = x.device if compute_device is None else torch.device(compute_device)

    def chunks() -> Sequence[TensorPair]:
        return ((x, y),)

    solution, _ = fit_centered_ridge_streaming(
        chunks,
        ridge_lambda,
        compute_device=device,
        allow_tf32=allow_tf32,
        solver=solver,
    )
    return solution


@torch.no_grad()
def normal_equation_relative_residual(
    statistics: CenteredRidgeStatistics,
    solution: RidgeSolution,
    *,
    analysis_device: torch.device | str = "cpu",
) -> float:
    """Evaluate the ridge normal-equation residual in FP64.

    This intentionally lives outside the fit path: converting an ``8192²``
    Gram matrix to FP64 is expensive on RTX 4090 hardware and should only be
    done when a run explicitly requests the diagnostic.
    """

    if solution.weight.shape != statistics.xty.shape:
        raise RidgeInputError("solution and statistics dimensions do not match")
    device = torch.device(analysis_device)
    system64 = statistics.xtx.to(device=device, dtype=torch.float64)
    system64.diagonal().add_(solution.ridge_lambda)
    weight64 = solution.weight.to(device=device, dtype=torch.float64)
    rhs64 = statistics.xty.to(device=device, dtype=torch.float64)
    residual = torch.linalg.vector_norm(system64 @ weight64 - rhs64)
    denominator = torch.linalg.vector_norm(rhs64)
    if denominator == 0:
        return 0.0 if residual == 0 else math.inf
    return float((residual / denominator).item())


__all__ = [
    "CenteredRidgeAccumulator",
    "CenteredRidgeStatistics",
    "ChunkFactory",
    "FeatureMeans",
    "RidgeInputError",
    "RidgeSolution",
    "RidgeSolveError",
    "StreamingFeatureMeans",
    "accumulate_centered_statistics",
    "estimate_feature_means",
    "fit_centered_ridge",
    "fit_centered_ridge_streaming",
    "normal_equation_relative_residual",
    "solve_centered_ridge",
]
