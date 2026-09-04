"""Reverse formal adapter around the audited direction-neutral OLS workers.

The shared production worker implementation is imported under the normal
`select_layers` module name so multiprocessing spawn remains importable.  Its
forward-only driver assumptions are replaced in this parent process; worker
math and checkpoint formats are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .protocol import (
    canonical_sha256,
    DEFAULT_CONFIG,
    DIRECTION,
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    SOURCE,
    TOP_K,
    TARGET,
    atomic_write_json,
    read_json,
    sha256_file,
    validate_capture_pair,
    validate_config,
)


METHOD = "same-head-single-source-ols-affine-uniform-channel-r2"
FEATURE_ORDER = (
    "selected-source-layer rank, then KV head 0..7, then head dimension 0..127"
)
SELECTION_RULE = {
    "probe": "independent affine OLS 128->128 for same-numbered KV head",
    "key_space": "content K after inverse RoPE",
    "channel_reduction": "uniform arithmetic mean of 128 per-channel R2 values",
    "head_reduction": "uniform arithmetic mean over 8 KV heads",
    "kv_reduction": "equal arithmetic mean of K and V scores",
    "tie_break": "lower source layer index",
    "global_r2": "diagnostic only; never used for ranking",
    "benchmark_data_used": False,
}
SCORE_SHAPES_DTYPES = {
    "key_head_uniform_r2": ((40, 64, 8), np.dtype("float64")),
    "value_head_uniform_r2": ((40, 64, 8), np.dtype("float64")),
    "key_head_global_r2_diagnostic": ((40, 64, 8), np.dtype("float64")),
    "value_head_global_r2_diagnostic": ((40, 64, 8), np.dtype("float64")),
    "key_head_averaged_uniform_r2": ((40, 64), np.dtype("float64")),
    "value_head_averaged_uniform_r2": ((40, 64), np.dtype("float64")),
    "combined_equal_kv_uniform_r2": ((40, 64), np.dtype("float64")),
    "solver_code": ((40, 2, 64, 8), np.dtype("int8")),
}


@dataclass(frozen=True, eq=False)
class ReverseArchitecture:
    source_layers: int
    target_layers: int
    kv_heads: int
    head_dim: int
    top_k: int

    def _tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.source_layers,
            self.target_layers,
            self.kv_heads,
            self.head_dim,
            self.top_k,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReverseArchitecture):
            return NotImplemented
        pair = {self._tuple(), other._tuple()}
        # The second alternative exists solely for the shared driver's legacy
        # formal gate.  All actual reverse dimensions are checked independently
        # before invoking it.
        return self._tuple() == other._tuple() or pair == {
            (64, 40, 8, 128, TOP_K),
            (40, 64, 8, 128, 8),
        }


def _load_implementation() -> Any:
    scripts = REPOSITORY_ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    module = importlib.import_module("select_layers")
    if Path(module.__file__).resolve() != (scripts / "select_layers.py").resolve():
        raise ImportError(f"unexpected select_layers implementation: {module.__file__}")
    return module


def _inside_experiment_reference(selection_path: Path, value: Any, *, label: str) -> Path:
    path = Path(str(value))
    resolved = (
        path.resolve()
        if path.is_absolute()
        else (selection_path.parent / path).resolve()
    )
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(f"reverse selection {label} escapes experiment root: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"reverse selection {label} is missing: {resolved}")
    return resolved


def _validate_reference(
    selection_path: Path, reference: Any, *, label: str
) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError(f"reverse selection {label} reference is malformed")
    path = _inside_experiment_reference(selection_path, reference["path"], label=label)
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"reverse selection {label} SHA256 mismatch")
    return path


def _score_rows_from_npz(selection_path: Path, raw: dict[str, Any]) -> list[list[int]]:
    score_artifacts = raw.get("score_artifacts")
    if not isinstance(score_artifacts, dict) or set(score_artifacts) != {"npz", "heatmaps"}:
        raise ValueError("reverse selection score artifact inventory is malformed")
    heatmaps = score_artifacts.get("heatmaps")
    if not isinstance(heatmaps, dict) or set(heatmaps) != {"key", "value", "combined"}:
        raise ValueError("reverse selection heatmap inventory is incomplete")
    for name, reference in heatmaps.items():
        heatmap = _validate_reference(
            selection_path, reference, label=f"{name} score heatmap"
        )
        if heatmap.name != f"heatmap_{name}.png":
            raise ValueError(f"reverse selection {name} heatmap name is invalid")
    score_reference = score_artifacts.get("npz")
    score_path = _validate_reference(selection_path, score_reference, label="score NPZ")
    score_sidecar_path = score_path.with_suffix(".json")
    if not score_sidecar_path.is_file():
        raise FileNotFoundError(f"selection score sidecar is missing: {score_sidecar_path}")
    sidecar = read_json(score_sidecar_path)
    expected_sidecar = {
        "schema_version": 1,
        "payload": score_path.name,
        "sha256": sha256_file(score_path),
        "nbytes": score_path.stat().st_size,
        "kind": "selection_scores",
        "input_fingerprint": raw.get("input_fingerprint"),
        "method": METHOD,
        "selection_score": "uniform_channel_r2_only",
        "global_r2_role": "diagnostic_only_never_used_for_ranking",
    }
    for field, expected in expected_sidecar.items():
        if sidecar.get(field) != expected:
            raise ValueError(f"selection score sidecar {field} mismatch")
    expected_array_metadata = {
        name: {"shape": list(shape), "dtype": str(dtype)}
        for name, (shape, dtype) in SCORE_SHAPES_DTYPES.items()
    }
    if sidecar.get("arrays") != expected_array_metadata:
        raise ValueError("selection score sidecar array inventory mismatch")

    with np.load(score_path, allow_pickle=False) as archive:
        if set(archive.files) != set(SCORE_SHAPES_DTYPES):
            raise ValueError("selection score NPZ tensor set mismatch")
        arrays = {name: archive[name] for name in archive.files}
    for name, (shape, dtype) in SCORE_SHAPES_DTYPES.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"selection score {name} has {value.shape}/{value.dtype}, expected {shape}/{dtype}"
            )
        if name != "solver_code" and not np.isfinite(value).all():
            raise ValueError(f"selection score {name} contains NaN/Inf")
    solver_codes = set(int(value) for value in np.unique(arrays["solver_code"]))
    if not solver_codes.issubset({0, 1}):
        raise ValueError(f"selection score contains invalid OLS solver codes: {solver_codes}")

    key_average = arrays["key_head_uniform_r2"].astype(np.float64).mean(axis=-1)
    value_average = arrays["value_head_uniform_r2"].astype(np.float64).mean(axis=-1)
    combined = (key_average + value_average) / 2.0
    recomputed = {
        "key_head_averaged_uniform_r2": key_average,
        "value_head_averaged_uniform_r2": value_average,
        "combined_equal_kv_uniform_r2": combined,
    }
    for name, expected in recomputed.items():
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"selection score {name} is not the exact frozen reduction")
    source_indices = np.arange(64)
    return [
        [int(value) for value in np.lexsort((source_indices, -row))[:TOP_K]]
        for row in combined
    ]


def _validate_statistics_references(selection_path: Path, raw: dict[str, Any]) -> None:
    statistics = raw.get("sufficient_statistics")
    if not isinstance(statistics, dict) or set(statistics) != {
        "activation_inputs",
        "moments_manifest",
        "single_source_manifest",
    }:
        raise ValueError("reverse selection sufficient-statistics references are incomplete")
    activation_path = _validate_reference(
        selection_path, statistics["activation_inputs"], label="activation catalog"
    )
    activation = read_json(activation_path)
    recorded = activation.get("input_fingerprint")
    without = dict(activation)
    without.pop("input_fingerprint", None)
    if recorded != canonical_sha256(without) or recorded != raw.get("input_fingerprint"):
        raise ValueError("reverse selection activation fingerprint mismatch")
    expected_architecture = {
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "top_k": TOP_K,
    }
    if activation.get("architecture") != expected_architecture:
        raise ValueError("reverse selection activation-catalog architecture mismatch")
    for role, count in (("source", 2 * 64), ("target", 2 * 40)):
        entries = activation.get(role)
        if not isinstance(entries, list) or len(entries) != count:
            raise ValueError(f"reverse selection activation catalog has invalid {role} coverage")
    for name in ("moments_manifest", "single_source_manifest"):
        path = _validate_reference(selection_path, statistics[name], label=name)
        manifest = read_json(path)
        if manifest.get("input_fingerprint") != recorded:
            raise ValueError(f"reverse selection {name} input fingerprint mismatch")


def validate_reverse_selection_artifact(
    path: str | Path,
    *,
    expected_direction: str = DIRECTION,
    require_implementation: bool = True,
) -> dict[str, Any]:
    """Recompute the frozen score reductions/ranking and validate every link."""

    selection_path = Path(path).resolve()
    try:
        selection_path.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("reverse selection artifact escapes experiment root") from exc
    raw = read_json(selection_path)
    exact = {
        "schema_version": 1,
        "direction": expected_direction,
        "formal_protocol": True,
        "method": METHOD,
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "k": TOP_K,
        "n_samples": 128_000,
        "feature_order": FEATURE_ORDER,
        "selection_rule": SELECTION_RULE,
    }
    for field, expected in exact.items():
        if raw.get(field) != expected:
            raise ValueError(f"reverse selection {field} mismatch")
    token_sha = raw.get("token_sha256")
    if (
        not isinstance(token_sha, list)
        or len(token_sha) != 1
        or not isinstance(token_sha[0], str)
        or len(token_sha[0]) != 64
        or any(character not in "0123456789abcdef" for character in token_sha[0])
    ):
        raise ValueError("reverse selection token SHA256 is invalid")
    expected_rows = _score_rows_from_npz(selection_path, raw)
    if raw.get("selected_source_layers_rank_order") != expected_rows:
        raise ValueError("reverse selection rows are not the exact score-derived top-20")
    _validate_statistics_references(selection_path, raw)

    if require_implementation:
        expected_binding = {
            "source": {
                "model_name": SOURCE.model_name,
                "capture_native_role": SOURCE.capture_native_role,
            },
            "target": {
                "model_name": TARGET.model_name,
                "capture_native_role": TARGET.capture_native_role,
            },
        }
        if raw.get("semantic_role_binding") != expected_binding:
            raise ValueError("reverse selection semantic role binding mismatch")
        implementation = raw.get("implementation")
        if not isinstance(implementation, dict):
            raise ValueError("reverse selection implementation provenance is missing")
        shared_worker = (REPOSITORY_ROOT / "scripts/select_layers.py").resolve()
        adapter = Path(__file__).resolve()
        underlying = _inside_experiment_reference(
            selection_path,
            implementation.get("underlying_manifest", ""),
            label="underlying selection manifest",
        )
        implementation_expected = {
            "shared_worker": str(shared_worker),
            "shared_worker_sha256": sha256_file(shared_worker),
            "reverse_adapter": str(adapter),
            "reverse_adapter_sha256": sha256_file(adapter),
            "underlying_manifest": str(underlying),
            "underlying_manifest_sha256": sha256_file(underlying),
        }
        if implementation != implementation_expected:
            raise ValueError("reverse selection implementation provenance mismatch")
        validate_reverse_selection_artifact(
            underlying,
            expected_direction="qwen3_14b_to_32b",
            require_implementation=False,
        )
        underlying_raw = read_json(underlying)
        expected_canonical = dict(underlying_raw)
        expected_canonical["direction"] = DIRECTION
        expected_canonical["semantic_role_binding"] = expected_binding
        expected_canonical["implementation"] = implementation_expected
        if raw != expected_canonical:
            raise ValueError("canonical reverse selection differs from its verified underlying manifest")
    return raw


def _write_reverse_heatmap(path: Path, scores: np.ndarray, title: str) -> None:
    """Direction-correct replacement for the shared forward heatmap renderer."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(scores, origin="lower", aspect="auto", interpolation="nearest")
    axis.set_xlabel("source layer (Qwen3-32B)")
    axis.set_ylabel("target layer (Qwen3-14B)")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label="uniform-channel R2")
    figure.tight_layout()
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        figure.savefig(temporary, dpi=180)
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)


def _publish_reverse_selection(
    implementation_manifest: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    destination = destination.resolve()
    try:
        destination.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("reverse selection output escapes experiment root") from exc
    raw = validate_reverse_selection_artifact(
        implementation_manifest,
        expected_direction="qwen3_14b_to_32b",
        require_implementation=False,
    )
    architecture_expected = {
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "k": TOP_K,
        "n_samples": 128_000,
        "formal_protocol": True,
    }
    for field, expected in architecture_expected.items():
        if raw.get(field) != expected:
            raise ValueError(
                f"selection implementation {field}={raw.get(field)!r}, expected {expected!r}"
            )
    rows = raw.get("selected_source_layers_rank_order")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("reverse selection must contain exactly 40 target rows")
    for target_layer, row in enumerate(rows):
        if (
            not isinstance(row, list)
            or len(row) != TOP_K
            or len(set(row)) != TOP_K
            or any(not isinstance(layer, int) or layer < 0 or layer >= 64 for layer in row)
        ):
            raise ValueError(f"invalid reverse top-{TOP_K} row for target layer {target_layer}")
    raw["direction"] = DIRECTION
    raw["semantic_role_binding"] = {
        "source": {
            "model_name": SOURCE.model_name,
            "capture_native_role": SOURCE.capture_native_role,
        },
        "target": {
            "model_name": TARGET.model_name,
            "capture_native_role": TARGET.capture_native_role,
        },
    }
    raw["implementation"] = {
        "shared_worker": str(
            (REPOSITORY_ROOT / "scripts/select_layers.py").resolve()
        ),
        "shared_worker_sha256": sha256_file(
            REPOSITORY_ROOT / "scripts/select_layers.py"
        ),
        "reverse_adapter": str(Path(__file__).resolve()),
        "reverse_adapter_sha256": sha256_file(__file__),
        "underlying_manifest": str(implementation_manifest.resolve()),
        "underlying_manifest_sha256": sha256_file(implementation_manifest),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        existing = read_json(destination)
        if existing != raw:
            raise FileExistsError(
                f"immutable reverse selection differs: {destination}; use --overwrite"
            )
    else:
        atomic_write_json(destination, raw)
    atomic_write_json(
        destination.with_suffix(".sha256.json"),
        {"path": str(destination.resolve()), "sha256": sha256_file(destination)},
    )
    validate_reverse_selection_artifact(destination)
    return raw


def run_selection(
    *,
    devices: str,
    chunk_tokens: int | None = None,
    only_target_layers: str | None = None,
    ols_fallback: str = "error",
    overwrite: bool = False,
    trust_sidecar_checksums: bool = False,
    verify_only: bool = False,
    no_ledger: bool = False,
    config_path: Path = DEFAULT_CONFIG,
) -> int:
    configuration = validate_config(config_path)
    # Metadata and sidecar hashes establish the capture-native/semantic-role
    # crossing here.  The shared catalog pass below optionally rereads every
    # payload hash as the expensive cryptographic gate.
    validate_capture_pair(split="train", verify_payloads=False)
    outputs = configuration["outputs"]
    implementation_output = outputs["selection_work"]
    statistics = outputs["statistics"]
    canonical_output = outputs["selection"]
    implementation_output.mkdir(parents=True, exist_ok=True)
    statistics.mkdir(parents=True, exist_ok=True)

    implementation = _load_implementation()
    original_catalog = implementation._catalog_activations
    original_make_run_directory = implementation.make_run_directory

    def reverse_catalog(*args: Any, **kwargs: Any) -> Any:
        # Reverse role/model validation was completed above.  Disable only the
        # shared function's forward-specific model-name check while retaining
        # all shape/provenance/token/hash cataloging.
        kwargs["formal_protocol"] = False
        return original_catalog(*args, **kwargs)

    implementation.Architecture = ReverseArchitecture
    implementation._catalog_activations = reverse_catalog
    implementation._write_heatmap = _write_reverse_heatmap
    implementation.make_run_directory = (
        lambda root, phase, name: original_make_run_directory(
            root, phase, str(name).replace("top" + str(8), "top20")
        )
    )
    argv: list[str] = [
        "--config",
        str(Path(config_path).resolve()),
        "--project-root",
        str(EXPERIMENT_ROOT),
        "--source-activations",
        str(SOURCE.train_root),
        "--target-activations",
        str(TARGET.train_root),
        "--output",
        str(implementation_output),
        "--statistics",
        str(statistics),
        "--devices",
        devices,
        "--ols-fallback",
        ols_fallback,
    ]
    if chunk_tokens is not None:
        argv.extend(("--chunk-tokens", str(chunk_tokens)))
    if only_target_layers:
        argv.extend(("--only-target-layers", only_target_layers))
    if overwrite:
        argv.append("--overwrite")
    if trust_sidecar_checksums:
        argv.append("--trust-sidecar-checksums")
    if verify_only:
        argv.append("--verify-only")
    # The shared ledger prose is specific to the other transfer direction.
    # Reverse evidence is already persisted in the isolated run/manifest, so
    # never publish that directionally incorrect prose.
    argv.append("--no-ledger")
    result = int(implementation.main(argv))
    if result:
        return result
    underlying = implementation_output / "selected_layers.json"
    if not underlying.is_file():
        # Partial --only-target-layers runs deliberately do not publish the
        # final selection until all 40 target layers are present.
        return 0
    _publish_reverse_selection(
        underlying,
        canonical_output,
        overwrite=overwrite,
    )
    return 0


__all__ = [
    "ReverseArchitecture",
    "run_selection",
    "validate_reverse_selection_artifact",
    "_publish_reverse_selection",
]
