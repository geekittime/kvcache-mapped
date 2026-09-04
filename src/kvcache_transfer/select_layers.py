"""Paper-conformant single-source OLS layer selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .metrics import r2_uniform_average
from .ridge import fit_centered_ridge


@dataclass(frozen=True)
class LayerSelection:
    selected_layers: tuple[tuple[int, ...], ...]
    combined_scores: np.ndarray
    key_scores: np.ndarray
    value_scores: np.ndarray


def rank_top_k(combined_scores: np.ndarray, k: int) -> tuple[tuple[int, ...], ...]:
    """Rank source layers per target layer with deterministic tie breaking.

    ``combined_scores`` has shape ``[target_layer, source_layer]``.  Higher R²
    wins; equal values prefer the lower source-layer index.  Returned layer
    order is prediction rank order and therefore also fixes mapper flattening.
    """
    if combined_scores.ndim != 2:
        raise ValueError("combined_scores must have [target, source] shape")
    if not np.isfinite(combined_scores).all():
        raise ValueError("combined_scores contains NaN or Inf")
    if not 1 <= k <= combined_scores.shape[1]:
        raise ValueError("k must be between one and the number of source layers")
    source_indices = np.arange(combined_scores.shape[1])
    result = []
    for row in combined_scores:
        order = np.lexsort((source_indices, -row))[:k]
        result.append(tuple(int(value) for value in order))
    return tuple(result)


def select_from_head_scores(
    key_head_scores: np.ndarray, value_head_scores: np.ndarray, k: int = 8
) -> LayerSelection:
    """Average K/V and target heads equally, exactly as production selection."""
    if key_head_scores.shape != value_head_scores.shape:
        raise ValueError("K and V score tensors must have identical shape")
    if key_head_scores.ndim != 3:
        raise ValueError("scores must have [target, source, target_head] shape")
    if not np.isfinite(key_head_scores).all() or not np.isfinite(value_head_scores).all():
        raise ValueError("K/V head scores contain NaN or Inf")
    key_scores = key_head_scores.astype(np.float64).mean(axis=-1)
    value_scores = value_head_scores.astype(np.float64).mean(axis=-1)
    combined = (key_scores + value_scores) / 2.0
    return LayerSelection(
        selected_layers=rank_top_k(combined, k),
        combined_scores=combined,
        key_scores=key_scores,
        value_scores=value_scores,
    )


def single_source_ols_r2(
    source: torch.Tensor, target: torch.Tensor
) -> float:
    """Fit the paper's 128→128 OLS+bias probe and return channel-uniform R²."""
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be rank-two token matrices")
    if source.shape[0] != target.shape[0]:
        raise ValueError("source and target must have the same token count")
    fit = fit_centered_ridge(source, target, ridge_lambda=0.0)
    prediction = source.float() @ fit.weight + fit.bias
    return float(r2_uniform_average(target, prediction))
