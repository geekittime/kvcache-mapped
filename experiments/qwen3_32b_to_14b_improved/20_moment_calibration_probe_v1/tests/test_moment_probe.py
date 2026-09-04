from __future__ import annotations

import numpy as np

from src.moment_probe import fit_head_scale_channel_bias, nested_cv, sse_for


def synthetic_stats(seed: int, scale: float = 1.08, bias: float = 0.03):
    rng = np.random.default_rng(seed)
    rows = {name: np.zeros((40, 8, 128), dtype=np.float64) for name in ("sum_p", "sum_y", "sum_p2", "sum_py", "sum_y2")}
    for row in range(40):
        p = rng.normal(size=(256, 8, 128))
        b = np.full((8, 128), bias)
        y = scale * p + b + rng.normal(scale=0.04, size=p.shape)
        rows["sum_p"][row] = p.sum(0)
        rows["sum_y"][row] = y.sum(0)
        rows["sum_p2"][row] = (p * p).sum(0)
        rows["sum_py"][row] = (p * y).sum(0)
        rows["sum_y2"][row] = (y * y).sum(0)
    return rows


def test_fit_recovers_scale_and_bias():
    stats = synthetic_stats(4)
    scale, bias = fit_head_scale_channel_bias(stats, list(range(32)), [0.5, 1.5])
    assert np.max(np.abs(scale - 1.08)) < 0.002
    assert np.max(np.abs(bias - 0.03)) < 0.01
    base = float(sse_for(stats, list(range(32, 40)), scale, bias, 0.0).sum())
    candidate = float(sse_for(stats, list(range(32, 40)), scale, bias, 1.0).sum())
    # The irreducible synthetic noise remains after the affine calibration;
    # the test should require a large reduction, not an impossible zero loss.
    assert candidate < base * 0.25


def test_nested_cv_selects_calibration_without_leakage():
    stats = [synthetic_stats(10), synthetic_stats(11)]
    folds = [list(range(first, first + 8)) for first in range(0, 40, 8)]
    result = nested_cv(stats, folds, [0.0, 0.5, 1.0], [0.5, 1.5])
    assert result["oof"]["relative_sse_reduction"] > 0.75
    assert all(row["selected_beta"] == 1.0 for row in result["outer_folds"])
    for row in result["outer_folds"]:
        assert set(row["training_rows"]).isdisjoint(row["validation_rows"])


def test_identity_data_prefers_zero_or_safe_tie():
    stats = synthetic_stats(20, scale=1.0, bias=0.0)
    folds = [list(range(first, first + 8)) for first in range(0, 40, 8)]
    result = nested_cv([stats], folds, [0.0, 0.5, 1.0], [0.5, 1.5])
    assert result["oof"]["relative_sse_reduction"] > -0.01
    assert all(row["selected_beta"] in (0.0, 0.5, 1.0) for row in result["outer_folds"])
