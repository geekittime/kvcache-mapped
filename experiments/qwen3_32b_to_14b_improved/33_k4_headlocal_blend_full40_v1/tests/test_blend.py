from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.blend import (
    ProtocolError,
    canonical_sha256,
    dense_from_head_local,
    deterministic_train_calibration_split,
    embed_k4_dense,
    exact_prediction_equivalence,
    fit_convex_alpha,
    fit_fold_alpha,
    fuse_dense_affines,
    source_batch_to_position_layer_major,
    validate_cv_folds,
    validate_promotion_rows,
    validate_safety_gate,
    validate_selection_subset,
)


def _selection():
    # Deliberately non-numeric order; embedding must preserve rank, not sort.
    s20 = [10, 2, 55, 7, 9, 31, 4, 60, 1, 18, 22, 33, 44, 5, 6, 12, 13, 14, 15, 16]
    s4 = [55, 7, 10, 31]
    return s20, s4


def test_k4_subset_and_order_validation():
    s20, s4 = _selection()
    assert validate_selection_subset(s20, s4) == (tuple(s20), tuple(s4))
    with pytest.raises(ProtocolError):
        validate_selection_subset(s20, [55, 7, 10, 63])
    with pytest.raises(ProtocolError):
        validate_selection_subset(s20, [55, 7, 10, True])


def test_dense_embedding_preserves_rank_and_cross_head_structure():
    s20, s4 = _selection()
    w4 = np.zeros((4096, 1024), dtype=np.float32)
    b4 = np.arange(1024, dtype=np.float32)
    # One sentinel per k4 rank/head/channel block.
    for r in range(4):
        for h in range(8):
            w4[(r * 8 + h) * 128 : (r * 8 + h + 1) * 128, h * 128 : (h + 1) * 128] = (1000 * r + 100 * h + np.arange(128, dtype=np.float32))[:, None]
    dense, bias = embed_k4_dense(w4, b4, s20, s4)
    assert dense.shape == (20480, 1024)
    assert np.array_equal(bias, b4)
    rank20 = {layer: i for i, layer in enumerate(s20)}
    for r4, layer in enumerate(s4):
        r20 = rank20[layer]
        for h in range(8):
            got = dense[(r20 * 8 + h) * 128 : (r20 * 8 + h + 1) * 128, h * 128 : (h + 1) * 128]
            want = w4[(r4 * 8 + h) * 128 : (r4 * 8 + h + 1) * 128, h * 128 : (h + 1) * 128]
            assert np.array_equal(got, want)
    # A non-selected canonical row stays zero.
    assert np.count_nonzero(dense) == np.count_nonzero(w4)


def test_head_local_embedding_and_dense_fusion_equivalence():
    rng = np.random.default_rng(7)
    s20, s4 = _selection()
    w4 = rng.normal(size=(4096, 1024)).astype(np.float32)
    b4 = rng.normal(size=(1024,)).astype(np.float32)
    wc = rng.normal(size=(8, 2560, 128)).astype(np.float32)
    bc = rng.normal(size=(8, 128)).astype(np.float32)
    x = rng.normal(size=(3, 20480)).astype(np.float64)
    assert exact_prediction_equivalence(x, w4, b4, wc, bc, s20, s4, 0.37)
    k4w, k4b = embed_k4_dense(w4, b4, s20, s4)
    cw, cb = dense_from_head_local(wc, bc)
    for h in range(8):
        rows = [((r * 8 + h) * 128 + ch) for r in range(20) for ch in range(128)]
        other_cols = [c for c in range(1024) if not (h * 128 <= c < (h + 1) * 128)]
        assert np.count_nonzero(cw[np.ix_(rows, other_cols)]) == 0
    fw, fb = fuse_dense_affines(k4w, k4b, cw, cb, 0.37)
    assert fw.dtype == np.float32 and fb.dtype == np.float32
    assert fw.shape == (20480, 1024) and fb.shape == (1024,)


def test_alpha_formula_sign_clip_and_degenerate_case():
    y = np.array([[3.0], [5.0]])
    p4 = np.array([[1.0], [1.0]])
    pc = np.array([[5.0], [5.0]])
    assert fit_convex_alpha(y, p4, pc) == pytest.approx(0.75)
    assert fit_convex_alpha(y, pc, p4) == pytest.approx(0.25)
    assert fit_convex_alpha(y, p4, p4) == 0.0
    with pytest.raises(ProtocolError):
        fit_convex_alpha(y, p4, np.array([[np.nan], [1.0]]))


def test_postfit_alpha_requires_explicit_sequence_namespace():
    y = np.ones((2, 1))
    p4 = np.zeros((2, 1))
    pc = np.full((2, 1), 2.0)
    predictions = {4: (y[:1], p4[:1], pc[:1]), 5: (y[1:], p4[1:], pc[1:])}
    assert fit_fold_alpha(predictions, [4, 5]) == pytest.approx(0.5)
    with pytest.raises(ProtocolError):
        fit_fold_alpha(predictions, [4, 4])
    with pytest.raises(ProtocolError):
        fit_fold_alpha(predictions, [4, 6])


def test_train_split_is_deterministic_and_disjoint():
    a = deterministic_train_calibration_split(seed=1777)
    b = deterministic_train_calibration_split(seed=1777)
    assert a == b and len(a[0]) == 400 and len(a[1]) == 100
    assert set(a[0]).isdisjoint(a[1]) and set(a[0]) | set(a[1]) == set(range(500))
    assert a != deterministic_train_calibration_split(seed=1778)


def test_cv_fold_schema_is_exact_and_covers_train500():
    folds = [{"fold": i, "validation_ids": list(range(i * 100, (i + 1) * 100))} for i in range(5)]
    assert len(validate_cv_folds(folds)) == 5
    with pytest.raises(ProtocolError):
        validate_cv_folds([{**folds[0], "validation_ids": [0, 0]}] + folds[1:])
    with pytest.raises(ProtocolError):
        validate_cv_folds([{**folds[0], "extra": 1}] + folds[1:])


def _rows(candidate_ratio: float = 0.98):
    rows = []
    for layer in range(40):
        for component in ("key", "value"):
            c = 100.0 * candidate_ratio
            n = 100.0
            k4 = 110.0
            s = 120.0
            rows.append({
                "target_layer": layer,
                "component": component,
                "eval_split": "heldout_tuning",
                "observations": 40 * 256,
                "candidate_sse": c,
                "nvidia_k20_sse": n,
                "k4_sse": k4,
                "shrinkage_sse": s,
                "strictly_below_k20": bool(c < n),
                "strictly_below_k20_and_k4": bool(c < n and c < k4),
            })
    return rows


def test_promotion_safety_gate_is_pre_registered_and_typed():
    rows = _rows(0.98)
    assert validate_promotion_rows(rows) is True
    assert validate_safety_gate(rows, aggregate_ratio_limit=1.0, row_ratio_limit=1.05) is True
    rows_bad = _rows(1.06)
    assert validate_promotion_rows(rows_bad) is False
    assert validate_safety_gate(rows_bad, aggregate_ratio_limit=1.0, row_ratio_limit=1.05) is False
    rows_bad[0]["candidate_sse"] = True
    with pytest.raises(ProtocolError):
        validate_safety_gate(rows_bad)


def test_axis_helper_handles_batch_one_and_multiple_batches():
    torch = pytest.importorskip("torch")
    # Value encoding gives [B,P,L] expected rows after the conversion.
    for b in (1, 2, 3):
        x = torch.empty((b, 2, 3, 1, 1), dtype=torch.int64)
        for bi in range(b):
            for li in range(2):
                for pi in range(3):
                    x[bi, li, pi, 0, 0] = bi * 100 + pi * 10 + li
        got = source_batch_to_position_layer_major(x)
        assert got.shape == (b, 3, 2, 1, 1)
        for bi in range(b):
            for pi in range(3):
                assert got[bi, pi, :, 0, 0].tolist() == [bi * 100 + pi * 10 + 0, bi * 100 + pi * 10 + 1]


def test_canonical_digest_is_stable():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})
