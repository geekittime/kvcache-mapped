"""Derive the Figure-4 ``k=4`` rows from the frozen full score tensor.

No OLS probe is re-run and no top-20 row is blindly truncated.  The full
40x64 score matrix is reconstructed from the same-head K/V score tensors,
then every target row is ranked again over all 64 source layers using the
paper's deterministic score reduction and lower-layer tie break.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .k4_protocol import K20_SELECTION, K4_SELECTION, K4_TOP_K
from .protocol import (
    DIRECTION,
    EXPERIMENT_ROOT,
    atomic_write_json,
    read_json,
    sha256_file,
)
from .selection_adapter import (
    FEATURE_ORDER,
    METHOD,
    SCORE_SHAPES_DTYPES,
    SELECTION_RULE,
    validate_reverse_selection_artifact,
)


PAPER_CURVE = {
    "paper": "arXiv:2608.03893v1",
    "section": "4.6",
    "figure": 4,
    "direction": "large_to_small_qwen3_32b_to_14b",
    "curve": "dotted_retuned",
    "selected_source_layers": K4_TOP_K,
    "figure_legend_evidence": "L→S: 32B→14B handoff (k=4, re-tuned)",
    "paper_disclosed": (
        "100 CoQA conversations of approximately 15 turns across five domains; "
        "word-level F1 across handoff turns; dotted large-to-small curve uses k=4"
    ),
    "paper_undisclosed": (
        "multi-turn k-sweep objective and tie break; conversation IDs; prompt; "
        "history construction; decoding settings; exact answer scorer implementation"
    ),
    "paper_bit_exact_claimed": False,
}


def _inside_experiment(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the isolated reverse experiment") from exc
    return resolved


def _score_path(parent_path: Path, parent: Mapping[str, Any]) -> Path:
    artifacts = parent.get("score_artifacts")
    reference = artifacts.get("npz") if isinstance(artifacts, Mapping) else None
    if not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}:
        raise ValueError("top-20 selection lacks an exact score-NPZ reference")
    recorded = Path(str(reference["path"]))
    path = recorded.resolve() if recorded.is_absolute() else (parent_path.parent / recorded).resolve()
    _inside_experiment(path, label="selection score NPZ")
    if not path.is_file() or sha256_file(path) != reference["sha256"]:
        raise ValueError("top-20 selection score NPZ is absent or changed")
    return path


def recompute_top4_rows(
    parent_path: str | Path = K20_SELECTION,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Return exact top-4 rows and score provenance after full parent audit."""

    path = _inside_experiment(parent_path, label="top-20 parent selection")
    parent = validate_reverse_selection_artifact(path)
    score_path = _score_path(path, parent)
    with np.load(score_path, allow_pickle=False) as archive:
        if set(archive.files) != set(SCORE_SHAPES_DTYPES):
            raise ValueError("selection score tensor inventory changed")
        arrays = {name: archive[name] for name in archive.files}
    for name, (shape, dtype) in SCORE_SHAPES_DTYPES.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"selection score {name} shape/dtype changed")
        if name != "solver_code" and not np.isfinite(value).all():
            raise ValueError(f"selection score {name} contains NaN/Inf")
    key = arrays["key_head_uniform_r2"].astype(np.float64).mean(axis=-1)
    value = arrays["value_head_uniform_r2"].astype(np.float64).mean(axis=-1)
    combined = (key + value) / 2.0
    if not np.array_equal(key, arrays["key_head_averaged_uniform_r2"]):
        raise ValueError("K score reduction differs from the frozen full tensor")
    if not np.array_equal(value, arrays["value_head_averaged_uniform_r2"]):
        raise ValueError("V score reduction differs from the frozen full tensor")
    if not np.array_equal(combined, arrays["combined_equal_kv_uniform_r2"]):
        raise ValueError("combined score reduction differs from the frozen full tensor")
    indices = np.arange(64)
    rows = [
        [int(item) for item in np.lexsort((indices, -row))[:K4_TOP_K]]
        for row in combined
    ]
    return rows, {
        "path": str(score_path),
        "sha256": sha256_file(score_path),
        "ranking_scope": "all_64_source_layers_per_target_not_top20_row_truncation",
        "score_reduction": "equal_KV_mean_of_uniform_head_and_channel_R2",
        "tie_break": "lower_source_layer_index",
    }


def expected_k4_selection(parent_path: str | Path = K20_SELECTION) -> dict[str, Any]:
    path = _inside_experiment(parent_path, label="top-20 parent selection")
    parent = validate_reverse_selection_artifact(path)
    rows, score = recompute_top4_rows(path)
    result = dict(parent)
    result["k"] = K4_TOP_K
    result["selected_source_layers_rank_order"] = rows
    result["paper_curve_variant"] = PAPER_CURVE
    result["k_hyperparameter_provenance"] = {
        "value": K4_TOP_K,
        "origin": "Figure 4 dotted large-to-small legend",
        "local_multi_turn_sweep_executed": False,
        "local_CoQA_gold_used_to_choose_k": False,
        "reason": "paper reports k=4 but does not disclose the retuning sweep criterion",
    }
    result["derivation"] = {
        "parent_top20_selection": {
            "path": str(path),
            "sha256": sha256_file(path),
        },
        "selection_scores_npz": score,
        "new_single_source_OLS_probes_run": False,
        "rows_recomputed_from_full_score_domain": True,
        "rows_obtained_by_slicing_k20_mapper_weights": False,
    }
    adapter = Path(__file__).resolve()
    result["implementation"] = {
        "k4_derivation_adapter": str(adapter),
        "k4_derivation_adapter_sha256": sha256_file(adapter),
        "parent_top20_selection": str(path),
        "parent_top20_selection_sha256": sha256_file(path),
    }
    return result


def validate_k4_selection_artifact(path: str | Path = K4_SELECTION) -> dict[str, Any]:
    selection_path = _inside_experiment(path, label="k4 selection")
    raw = read_json(selection_path)
    derivation = raw.get("derivation")
    parent_reference = (
        derivation.get("parent_top20_selection")
        if isinstance(derivation, Mapping)
        else None
    )
    if not isinstance(parent_reference, Mapping) or set(parent_reference) != {"path", "sha256"}:
        raise ValueError("k4 selection parent reference is malformed")
    parent_path = _inside_experiment(
        str(parent_reference["path"]), label="k4 parent selection"
    )
    if parent_path != K20_SELECTION.resolve():
        raise ValueError("k4 selection derives from an unexpected parent")
    if sha256_file(parent_path) != parent_reference["sha256"]:
        raise ValueError("k4 parent selection SHA256 changed")
    expected = expected_k4_selection(parent_path)
    if raw != expected:
        raise ValueError("k4 selection differs from exact full-score recomputation")
    fixed = {
        "direction": DIRECTION,
        "formal_protocol": True,
        "method": METHOD,
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "k": K4_TOP_K,
        "n_samples": 128_000,
        "feature_order": FEATURE_ORDER,
        "selection_rule": SELECTION_RULE,
    }
    for field, wanted in fixed.items():
        if raw.get(field) != wanted:
            raise ValueError(f"k4 selection {field} changed")
    return raw


def publish_k4_selection(
    destination: str | Path = K4_SELECTION,
    *,
    parent_path: str | Path = K20_SELECTION,
) -> dict[str, Any]:
    output = _inside_experiment(destination, label="k4 selection output")
    if output != K4_SELECTION.resolve() or Path(parent_path).resolve() != K20_SELECTION.resolve():
        raise ValueError("k4 selection paths are immutable")
    value = expected_k4_selection(parent_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if read_json(output) != value:
            raise FileExistsError("existing immutable k4 selection differs")
    else:
        atomic_write_json(output, value)
    sidecar = output.with_suffix(".sha256.json")
    marker = {"path": str(output), "sha256": sha256_file(output)}
    if sidecar.exists() and read_json(sidecar) != marker:
        raise FileExistsError("existing immutable k4 selection checksum differs")
    atomic_write_json(sidecar, marker)
    validate_k4_selection_artifact(output)
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    parse_args(argv)
    value = publish_k4_selection()
    print(
        f"published {len(value['selected_source_layers_rank_order'])} exact top-4 rows "
        f"to {K4_SELECTION}"
    )
    return 0


__all__ = [
    "PAPER_CURVE",
    "expected_k4_selection",
    "main",
    "publish_k4_selection",
    "recompute_top4_rows",
    "validate_k4_selection_artifact",
]
