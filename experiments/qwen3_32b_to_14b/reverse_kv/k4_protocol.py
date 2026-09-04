"""Immutable protocol constants for the Figure-4 retuned ``k=4`` curve.

This is deliberately separate from :mod:`reverse_kv.protocol`, whose
``TOP_K=20`` constants define the production 32B -> 14B mapper used by the
single-turn and latency experiments.  Figure 4 additionally labels a dotted
large-to-small curve as ``k=4, re-tuned``.  The paper does not disclose the
multi-turn sweep objective or the exact CoQA examples, prompt, history, or
decoding settings; those remain explicitly local deterministic choices.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .protocol import (
    DIRECTION,
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    SOURCE,
    TARGET,
    load_config,
    sha256_file,
)


K4_TOP_K = 4
K4_SOURCE_FEATURES = K4_TOP_K * 8 * 128
K4_TARGET_FEATURES = 8 * 128
K4_WEIGHT_COUNT = 2 * 40 * K4_SOURCE_FEATURES * K4_TARGET_FEATURES
K4_BIAS_COUNT = 2 * 40 * K4_TARGET_FEATURES
K4_TRAIN_OBSERVATIONS = 128_000

DEFAULT_K4_MAPPER_CONFIG = EXPERIMENT_ROOT / "config/mapper_k4_coqa.yaml"
K20_SELECTION = (
    EXPERIMENT_ROOT
    / "artifacts/selection/qwen3_32b_to_14b/selected_layers.json"
)
K4_SELECTION = (
    EXPERIMENT_ROOT
    / "artifacts/selection/qwen3_32b_to_14b/k4_retuned/selected_layers.json"
)
K4_MAPPER_ROOT = (
    EXPERIMENT_ROOT
    / "artifacts/mappers/qwen3_32b_to_14b/k4_retuned_lambda0.01"
)


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _inside_experiment(path: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the isolated reverse experiment: {resolved}") from exc
    return resolved


def validate_k4_mapper_config(
    path: str | Path = DEFAULT_K4_MAPPER_CONFIG,
) -> dict[str, Any]:
    """Validate every scientific and filesystem input of the independent fit."""

    config_path = Path(path).resolve()
    if config_path != DEFAULT_K4_MAPPER_CONFIG.resolve():
        raise ValueError("Figure-4 k=4 mapper config is immutable")
    raw = load_config(config_path)
    path_expectations = {
        "project_root": EXPERIMENT_ROOT,
        "shared_repository": REPOSITORY_ROOT,
        "source_model": SOURCE.model_path,
        "target_model": TARGET.model_path,
    }
    for field, expected in path_expectations.items():
        if Path(str(raw.get(field, ""))).resolve() != expected.resolve():
            raise ValueError(f"k4 config {field} changed")
    if raw.get("direction") != DIRECTION:
        raise ValueError("k4 config direction changed")
    if raw.get("curve_variant") != "figure4_large_to_small_retuned_k4":
        raise ValueError("k4 curve identity changed")

    paper = raw.get("paper")
    if not isinstance(paper, Mapping):
        raise ValueError("k4 config lacks paper settings")
    paper_expected = {
        "calibration_sequences": 500,
        "sequence_length": 1024,
        "position_stride": 4,
        "selected_source_layers": K4_TOP_K,
        "ridge_lambda": 0.01,
        "forward_dtype": "bfloat16",
        "covariance_dtype": "float32",
        "analysis_dtype": "float64",
    }
    for field, expected in paper_expected.items():
        if paper.get(field) != expected:
            raise ValueError(f"k4 paper.{field} changed")

    architecture = raw.get("architecture")
    architecture_expected = {
        "source_layers": 64,
        "target_layers": 40,
        "source_kv_heads": 8,
        "target_kv_heads": 8,
        "source_head_dim": 128,
        "target_head_dim": 128,
        "rope_theta": 1_000_000,
    }
    if not isinstance(architecture, Mapping):
        raise ValueError("k4 config lacks architecture")
    for field, expected in architecture_expected.items():
        if architecture.get(field) != expected:
            raise ValueError(f"k4 architecture.{field} changed")

    captures = raw.get("captures")
    capture_expected = {
        "source_train": SOURCE.train_root,
        "target_train": TARGET.train_root,
        "source_native_role": SOURCE.capture_native_role,
        "target_native_role": TARGET.capture_native_role,
    }
    if not isinstance(captures, Mapping):
        raise ValueError("k4 config lacks capture bindings")
    for field, expected in capture_expected.items():
        observed = captures.get(field)
        if isinstance(expected, Path):
            if Path(str(observed)).resolve() != expected.resolve():
                raise ValueError(f"k4 captures.{field} changed")
        elif observed != expected:
            raise ValueError(f"k4 captures.{field} changed")

    runtime = raw.get("runtime")
    runtime_expected = {
        "disable_tf32_for_master_fit": True,
        "statistics_chunk_tokens": 1024,
    }
    if not isinstance(runtime, Mapping):
        raise ValueError("k4 config lacks runtime settings")
    for field, expected in runtime_expected.items():
        if runtime.get(field) != expected:
            raise ValueError(f"k4 runtime.{field} changed")

    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"selection", "mapper"}:
        raise ValueError("k4 output inventory changed")
    resolved_outputs = {
        field: _inside_experiment(_resolve(EXPERIMENT_ROOT, value), label=f"k4 outputs.{field}")
        for field, value in outputs.items()
    }
    if resolved_outputs != {
        "selection": K4_SELECTION.resolve(),
        "mapper": K4_MAPPER_ROOT.resolve(),
    }:
        raise ValueError("k4 output paths changed")
    return {
        "path": config_path,
        "sha256": sha256_file(config_path),
        "raw": raw,
        "outputs": resolved_outputs,
    }


__all__ = [
    "DEFAULT_K4_MAPPER_CONFIG",
    "K20_SELECTION",
    "K4_BIAS_COUNT",
    "K4_MAPPER_ROOT",
    "K4_SELECTION",
    "K4_SOURCE_FEATURES",
    "K4_TARGET_FEATURES",
    "K4_TOP_K",
    "K4_TRAIN_OBSERVATIONS",
    "K4_WEIGHT_COUNT",
    "validate_k4_mapper_config",
]
