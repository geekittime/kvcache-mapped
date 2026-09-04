"""Pure, deterministic pieces of the k4 + head-local blend protocol.

The module deliberately contains no model loading and no CUDA side effects.  A
future runner may use these functions after the CPU input gate has passed.
Every layout conversion is explicit because a silent layer/position transpose
was the cause of the superseded full40 attempt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ProtocolError(RuntimeError):
    """Raised when a frozen experiment contract is violated."""


K20 = 20
K4 = 4
KV_HEADS = 8
HEAD_DIM = 128
SOURCE_WIDTH_20 = K20 * KV_HEADS * HEAD_DIM
SOURCE_WIDTH_4 = K4 * KV_HEADS * HEAD_DIM
HEAD_LOCAL_WIDTH = K20 * HEAD_DIM
TARGET_WIDTH = KV_HEADS * HEAD_DIM


def _strict_int(value: Any, label: str, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ProtocolError(f"{label}: expected a JSON integer (bool is not accepted)")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{label}: below minimum")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{label}: above maximum")
    return value


def _strict_float(value: Any, label: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProtocolError(f"{label}: expected finite JSON float")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{label}: below minimum")
    if maximum is not None and value > maximum:
        raise ProtocolError(f"{label}: above maximum")
    return value


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used for machine-readable manifest digests."""

    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path | str, *, reject_symlink: bool = True) -> dict[str, Any]:
    """Return a content reference, rejecting links and non-regular files."""

    p = Path(path)
    try:
        info = p.lstat()
    except FileNotFoundError as exc:
        raise ProtocolError(f"missing file: {p}") from exc
    if reject_symlink and stat.S_ISLNK(info.st_mode):
        raise ProtocolError(f"symlink is not an immutable file: {p}")
    if not stat.S_ISREG(info.st_mode):
        raise ProtocolError(f"non-regular file: {p}")
    return {"path": str(p.resolve()), "nbytes": int(info.st_size), "sha256": sha256_file(p)}


def _as_array(value: Any, *, name: str, shape: tuple[int, ...]):
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - dependency is in the pinned venv
        raise ProtocolError("numpy is required for affine layout operations") from exc
    array = np.asarray(value)
    if tuple(array.shape) != shape:
        raise ProtocolError(f"{name}: expected shape {shape}, got {tuple(array.shape)}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ProtocolError(f"{name}: non-finite or non-numeric values")
    return array


def validate_selection_subset(
    selection20: Sequence[int],
    selection4: Sequence[int],
    *,
    source_layers: int = 64,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate and return the frozen rank orders, proving k4 ⊆ k20.

    The order is meaningful: k4 rank ``r`` is looked up in the corresponding
    k20 rank, rather than sorted by numeric layer ID.  This prevents a common
    but silent permutation bug when k4 was selected by a different score pass.
    """

    if not isinstance(selection20, Sequence) or isinstance(selection20, (str, bytes)):
        raise ProtocolError("k20 selection must be a sequence")
    if not isinstance(selection4, Sequence) or isinstance(selection4, (str, bytes)):
        raise ProtocolError("k4 selection must be a sequence")
    if len(selection20) != K20 or len(selection4) != K4:
        raise ProtocolError("selection cardinality must be exactly 20 and 4")
    s20 = tuple(_strict_int(v, "k20 source layer", 0, source_layers - 1) for v in selection20)
    s4 = tuple(_strict_int(v, "k4 source layer", 0, source_layers - 1) for v in selection4)
    if len(set(s20)) != K20 or len(set(s4)) != K4:
        raise ProtocolError("source selection contains duplicates")
    if not set(s4).issubset(set(s20)):
        raise ProtocolError("k4 source selection is not a subset of k20 selection")
    return s20, s4


def _source_row(rank: int, head: int, channel: int) -> int:
    return (rank * KV_HEADS + head) * HEAD_DIM + channel


def _target_col(head: int, channel: int) -> int:
    return head * HEAD_DIM + channel


def dense_from_head_local(head_weights: Any, head_bias: Any) -> tuple[Any, Any]:
    """Embed C's eight independent maps into a [20480,1024] affine map."""

    import numpy as np

    w = _as_array(head_weights, name="head-local weight", shape=(KV_HEADS, HEAD_LOCAL_WIDTH, HEAD_DIM))
    b = _as_array(head_bias, name="head-local bias", shape=(KV_HEADS, HEAD_DIM))
    dense = np.zeros((SOURCE_WIDTH_20, TARGET_WIDTH), dtype=w.dtype)
    # C[h] feature order is (20-layer rank, channel), and its output is the
    # same target head.  Cross-head blocks intentionally remain exactly zero.
    for head in range(KV_HEADS):
        cols = slice(_target_col(head, 0), _target_col(head, HEAD_DIM - 1) + 1)
        for rank in range(K20):
            rows = slice(_source_row(rank, head, 0), _source_row(rank, head, HEAD_DIM - 1) + 1)
            dense[rows, cols] = w[head, rank * HEAD_DIM : (rank + 1) * HEAD_DIM, :]
    return dense, b.reshape(TARGET_WIDTH).copy()


def embed_k4_dense(k4_weight: Any, k4_bias: Any, selection20: Sequence[int], selection4: Sequence[int]) -> tuple[Any, Any]:
    """Embed a k4 mapper in the canonical k20 feature coordinate system."""

    import numpy as np

    s20, s4 = validate_selection_subset(selection20, selection4)
    w4 = _as_array(k4_weight, name="k4 weight", shape=(SOURCE_WIDTH_4, TARGET_WIDTH))
    b4 = _as_array(k4_bias, name="k4 bias", shape=(TARGET_WIDTH,))
    dense = np.zeros((SOURCE_WIDTH_20, TARGET_WIDTH), dtype=w4.dtype)
    rank20 = {layer: rank for rank, layer in enumerate(s20)}
    for r4, layer in enumerate(s4):
        r20 = rank20[layer]
        for head in range(KV_HEADS):
            rows20 = slice(_source_row(r20, head, 0), _source_row(r20, head, HEAD_DIM - 1) + 1)
            rows4 = slice(_source_row(r4, head, 0), _source_row(r4, head, HEAD_DIM - 1) + 1)
            dense[rows20, :] = w4[rows4, :]
    return dense, b4.copy()


def fuse_dense_affines(
    k4_dense_weight: Any,
    k4_dense_bias: Any,
    head_local_dense_weight: Any,
    head_local_dense_bias: Any,
    alpha: Any,
) -> tuple[Any, Any]:
    """Fuse two compatible affine maps without changing the downstream API."""

    import numpy as np

    w4 = _as_array(k4_dense_weight, name="embedded k4 weight", shape=(SOURCE_WIDTH_20, TARGET_WIDTH))
    b4 = _as_array(k4_dense_bias, name="embedded k4 bias", shape=(TARGET_WIDTH,))
    wc = _as_array(head_local_dense_weight, name="embedded head-local weight", shape=(SOURCE_WIDTH_20, TARGET_WIDTH))
    bc = _as_array(head_local_dense_bias, name="embedded head-local bias", shape=(TARGET_WIDTH,))
    a = np.asarray(alpha)
    if a.ndim == 0:
        scalar = a.item()
        if type(scalar) not in (float, int) or isinstance(scalar, bool):
            raise ProtocolError("blend alpha must be a finite scalar")
        av = float(scalar)
        if not math.isfinite(av) or not 0.0 <= av <= 1.0:
            raise ProtocolError("blend alpha outside [0,1]")
        out_w = (1.0 - av) * w4 + av * wc
        out_b = (1.0 - av) * b4 + av * bc
    elif tuple(a.shape) == (TARGET_WIDTH,):
        if not np.issubdtype(a.dtype, np.number) or not np.isfinite(a).all() or np.any(a < 0.0) or np.any(a > 1.0):
            raise ProtocolError("per-output blend alpha outside [0,1]")
        out_w = w4 * (1.0 - a.reshape(1, -1)) + wc * a.reshape(1, -1)
        out_b = b4 * (1.0 - a) + bc * a
    else:
        raise ProtocolError("alpha must be scalar or one value per target channel")
    if not np.isfinite(out_w).all() or not np.isfinite(out_b).all():
        raise ProtocolError("fused affine is non-finite")
    return out_w.astype(np.float32, copy=False), out_b.astype(np.float32, copy=False)


def fit_convex_alpha(target: Any, p4: Any, p_c: Any) -> float:
    """Fit scalar α on a *calibration* namespace only.

    With ``e = target - p4`` and ``d = p_c - p4``, minimizing
    ``||target - (p4 + αd)||²`` gives ``α = sum(e*d)/sum(d²)``.  The sign is
    intentionally documented here to avoid the residual-convention mistake.
    """

    import numpy as np

    y = np.asarray(target, dtype=np.float64)
    a = np.asarray(p4, dtype=np.float64)
    c = np.asarray(p_c, dtype=np.float64)
    if y.shape != a.shape or y.shape != c.shape or y.size == 0:
        raise ProtocolError("alpha arrays must have equal non-empty shape")
    if not np.isfinite(y).all() or not np.isfinite(a).all() or not np.isfinite(c).all():
        raise ProtocolError("alpha arrays contain non-finite values")
    d = c - a
    den = float(np.sum(d * d, dtype=np.float64))
    num = float(np.sum((y - a) * d, dtype=np.float64))
    raw = num / den if den > 0.0 else 0.0
    return float(min(1.0, max(0.0, raw)))


def fit_fold_alpha(
    calibration_predictions: Mapping[int, tuple[Any, Any, Any]],
    calibration_ids: Iterable[int],
) -> float:
    """Fit α from sequence-keyed calibration predictions, with no implicit IDs."""

    import numpy as np

    ids = tuple(calibration_ids)
    if not ids or len(set(ids)) != len(ids) or any(type(v) is not int or v < 0 for v in ids):
        raise ProtocolError("calibration IDs must be unique nonnegative integers")
    missing = [sid for sid in ids if sid not in calibration_predictions]
    if missing:
        raise ProtocolError(f"missing calibration predictions for sequence {missing[0]}")
    ys, p4s, pcs = [], [], []
    for sid in ids:
        y, p4, pc = calibration_predictions[sid]
        ys.append(np.asarray(y))
        p4s.append(np.asarray(p4))
        pcs.append(np.asarray(pc))
    return fit_convex_alpha(np.concatenate(ys, axis=0), np.concatenate(p4s, axis=0), np.concatenate(pcs, axis=0))


def deterministic_train_calibration_split(
    *,
    sequence_count: int = 500,
    fit_count: int = 400,
    seed: int = 1777,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Make a reproducible sequence-level train-fit/calibration split.

    IDs are hashed before ordering rather than relying on process-randomized
    Python ``hash``.  Both sets remain inside the public train500 namespace;
    heldout_tuning and locked IDs are never accepted by this helper.
    """

    n = _strict_int(sequence_count, "sequence_count", 2)
    k = _strict_int(fit_count, "fit_count", 1, n - 1)
    s = _strict_int(seed, "split seed", 0)
    ranked = sorted(range(n), key=lambda i: hashlib.sha256(f"{s}:{i}".encode("ascii")).hexdigest())
    fit = tuple(sorted(ranked[:k]))
    calibration = tuple(sorted(ranked[k:]))
    assert_train_calibration_disjoint(fit, calibration, sequence_count=n)
    return fit, calibration


def assert_train_calibration_disjoint(
    fit_ids: Sequence[int],
    calibration_ids: Sequence[int],
    *,
    sequence_count: int = 500,
) -> None:
    n = _strict_int(sequence_count, "sequence_count", 2)
    for label, ids in (("fit", fit_ids), ("calibration", calibration_ids)):
        if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
            raise ProtocolError(f"{label} IDs must be a sequence")
        if any(type(v) is not int or not 0 <= v < n for v in ids):
            raise ProtocolError(f"{label} ID outside train namespace")
        if len(set(ids)) != len(ids):
            raise ProtocolError(f"duplicate {label} ID")
    if set(fit_ids).intersection(calibration_ids):
        raise ProtocolError("fit and alpha-calibration IDs overlap")
    if set(fit_ids).union(calibration_ids) != set(range(n)):
        raise ProtocolError("fit/calibration split does not cover train namespace")


def validate_cv_folds(
    folds: Sequence[Mapping[str, Any]],
    *,
    sequence_count: int = 500,
) -> tuple[tuple[int, ...], ...]:
    """Validate sequence-level folds used only for alpha/complexity selection."""

    if not isinstance(folds, Sequence) or len(folds) < 2:
        raise ProtocolError("at least two train-only CV folds are required")
    n = _strict_int(sequence_count, "sequence_count", 2)
    seen: set[int] = set()
    out: list[tuple[int, ...]] = []
    for index, row in enumerate(folds):
        if not isinstance(row, Mapping) or set(row) != {"fold", "validation_ids"}:
            raise ProtocolError(f"fold {index}: exact schema mismatch")
        if _strict_int(row["fold"], f"fold {index} number", 0) != index:
            raise ProtocolError("fold order is not deterministic")
        ids = tuple(_strict_int(v, f"fold {index} id", 0, n - 1) for v in row["validation_ids"])
        if not ids or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ProtocolError("CV folds overlap or are empty")
        seen.update(ids)
        out.append(ids)
    if seen != set(range(n)):
        raise ProtocolError("CV folds do not cover exactly train500")
    return tuple(out)


def validate_promotion_rows(rows: Sequence[Mapping[str, Any]], *, row_count: int = 80) -> bool:
    """Strict clean gate against NVIDIA k20; no shrinkage tuning is hidden."""

    if not isinstance(rows, Sequence) or len(rows) != row_count:
        raise ProtocolError(f"promotion grid must contain exactly {row_count} rows")
    seen: set[tuple[int, str]] = set()
    passed = True
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "target_layer", "component", "eval_split", "observations",
            "candidate_sse", "nvidia_k20_sse", "k4_sse", "shrinkage_sse",
            "strictly_below_k20", "strictly_below_k20_and_k4",
        }:
            raise ProtocolError("promotion row schema changed")
        layer = _strict_int(row["target_layer"], "target layer", 0, 39)
        component = row["component"]
        if component not in ("key", "value") or (layer, component) in seen:
            raise ProtocolError("duplicate or invalid promotion row")
        seen.add((layer, component))
        if row["eval_split"] != "heldout_tuning" or _strict_int(row["observations"], "observations", 1) != 40 * 256:
            raise ProtocolError("promotion used a non-heldout split")
        vals = [_strict_float(row[k], k, 0.0) for k in ("candidate_sse", "nvidia_k20_sse", "k4_sse", "shrinkage_sse")]
        for key in ("strictly_below_k20", "strictly_below_k20_and_k4"):
            if type(row[key]) is not bool:
                raise ProtocolError(f"{key} must be a bool")
        below_k20 = vals[0] < vals[1]
        below_k20_k4 = below_k20 and vals[0] < vals[2]
        if row["strictly_below_k20"] is not below_k20 or row["strictly_below_k20_and_k4"] is not below_k20_k4:
            raise ProtocolError("promotion flags do not match SSEs")
        passed = passed and below_k20
    expected = {(layer, comp) for layer in range(40) for comp in ("key", "value")}
    if seen != expected:
        raise ProtocolError("promotion grid is incomplete")
    return passed


def validate_safety_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    aggregate_ratio_limit: float = 1.0,
    row_ratio_limit: float = 1.05,
) -> bool:
    """Evaluate the pre-registered exploratory safety gate.

    Unlike a superiority gate, this permits small individual regressions while
    requiring a strict aggregate improvement and a fixed worst-row bound.  The
    limits are protocol constants chosen *before* the heldout diagnostic and
    are not inferred from its result.
    """

    import numpy as np

    if type(aggregate_ratio_limit) not in (float, int) or isinstance(aggregate_ratio_limit, bool) or not math.isfinite(float(aggregate_ratio_limit)) or float(aggregate_ratio_limit) <= 0.0:
        raise ProtocolError("invalid aggregate safety limit")
    if type(row_ratio_limit) not in (float, int) or isinstance(row_ratio_limit, bool) or not math.isfinite(float(row_ratio_limit)) or float(row_ratio_limit) <= 0.0:
        raise ProtocolError("invalid row safety limit")
    # Reuse the strict identity/type checks, but do not interpret its boolean
    # return as an all-rows superiority claim.
    validate_promotion_rows(rows, row_count=80)
    candidates = np.asarray([float(row["candidate_sse"]) for row in rows], dtype=np.float64)
    baselines = np.asarray([float(row["nvidia_k20_sse"]) for row in rows], dtype=np.float64)
    if np.any(baselines <= 0.0) or not np.isfinite(candidates).all() or not np.isfinite(baselines).all():
        raise ProtocolError("invalid safety-gate SSE")
    return bool(float(candidates.sum()) / float(baselines.sum()) < float(aggregate_ratio_limit) and float(np.max(candidates / baselines)) <= float(row_ratio_limit))


def exact_prediction_equivalence(
    x20: Any,
    k4_weight: Any,
    k4_bias: Any,
    c_weight: Any,
    c_bias: Any,
    selection20: Sequence[int],
    selection4: Sequence[int],
    alpha: float,
    *,
    atol: float = 1e-5,
) -> bool:
    """Numerically verify dense fusion equals direct k4/C prediction blend."""

    import numpy as np

    x = np.asarray(x20, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != SOURCE_WIDTH_20:
        raise ProtocolError("x20 must be [N,20480]")
    s20, s4 = validate_selection_subset(selection20, selection4)
    w4, b4 = embed_k4_dense(k4_weight, k4_bias, s20, s4)
    wc, bc = dense_from_head_local(c_weight, c_bias)
    wf, bf = fuse_dense_affines(w4, b4, wc, bc, float(alpha))
    rank20 = {layer: rank for rank, layer in enumerate(s20)}
    idx4 = [_source_row(rank20[layer], h, ch) for layer in s4 for h in range(KV_HEADS) for ch in range(HEAD_DIM)]
    x4 = x[:, idx4]
    direct4 = x4 @ _as_array(k4_weight, name="k4 weight", shape=(SOURCE_WIDTH_4, TARGET_WIDTH)) + _as_array(k4_bias, name="k4 bias", shape=(TARGET_WIDTH,))
    # Convert canonical x to C's eight [rank,channel] blocks and preserve
    # target head order.
    direct_c = np.zeros((x.shape[0], TARGET_WIDTH), dtype=np.float64)
    cw = _as_array(c_weight, name="head-local weight", shape=(KV_HEADS, HEAD_LOCAL_WIDTH, HEAD_DIM))
    cb = _as_array(c_bias, name="head-local bias", shape=(KV_HEADS, HEAD_DIM))
    for h in range(KV_HEADS):
        rows = [_source_row(r, h, ch) for r in range(K20) for ch in range(HEAD_DIM)]
        direct_c[:, h * HEAD_DIM : (h + 1) * HEAD_DIM] = x[:, rows] @ cw[h] + cb[h]
    direct = (1.0 - alpha) * direct4 + alpha * direct_c
    dense = x @ wf + bf
    return bool(np.allclose(dense, direct, rtol=0.0, atol=float(atol)))


def source_batch_to_position_layer_major(stacked: Any) -> Any:
    """Convert ``[B,L,P,H,D]`` activations to ``[B,P,L,H,D]``.

    Both the candidate fit and the k4 branch use this single helper.  Keeping
    it here makes the axis contract independently testable before any model or
    CUDA import.
    """

    shape = getattr(stacked, "shape", None)
    if shape is None or len(shape) != 5:
        raise ProtocolError("stacked activation batch must have five axes")
    return stacked.permute(0, 2, 1, 3, 4)


def current_python_identity() -> dict[str, Any]:
    """Small runtime identity record for a future formal runner."""

    executable = os.environ.get("PYTHON", os.sys.executable)
    return {
        # Keep the lexical venv path in the receipt; recording only the target
        # of its symlink (e.g. /usr/bin/python3.12) loses the interpreter
        # binding required by the protocol.
        "executable": executable,
        "resolved_executable": os.path.realpath(executable),
        "python_major_minor": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dont_write_bytecode": bool(os.sys.dont_write_bytecode),
        "pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX"),
    }
