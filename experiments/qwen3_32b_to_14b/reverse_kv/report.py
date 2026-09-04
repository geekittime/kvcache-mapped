"""Fail-closed, CPU-only final report for the Qwen3 32B -> 14B run.

The report is deliberately the last formal pipeline stage.  It re-runs the
scientific validators for every upstream result and derives all displayed
numbers from per-layer, per-example, or raw-timing records.  A copied summary
is never accepted as the source of a headline number.

Publication is an atomic directory rename.  Existing output is immutable: a
valid directory is adopted and an incomplete or changed directory is fatal.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from . import coqa_eval
from .coqa_k4_eval import (
    RECEIPT_NAME as K4_COQA_RECEIPT_NAME,
    validate_k4_receipt,
)
from .eval_common import (
    EVALUATION_SCHEMA_VERSION,
    canonical_jsonl_bytes,
    code_identity,
    mapper_manifest_identity,
    verify_hashed_json,
)
from .fit_adapter import _validate_complete_payloads as validate_k20_payloads
from .hellaswag_eval import (
    ALLOWED_MODES as HELLASWAG_MODES,
    FORMAL_RECEIPT as HELLASWAG_RECEIPT_NAME,
    TASK as HELLASWAG_TASK,
    _validate_existing_output as validate_hellaswag_output,
    validate_hellaswag_result,
    validate_run_definition as validate_hellaswag_definition,
)
from .heldout import validate_heldout_artifact
from .k4_fit import _validate_complete_payloads as validate_k4_payloads
from .k4_heldout import K4_HELDOUT, validate_k4_heldout_artifact
from .k4_protocol import (
    K4_BIAS_COUNT,
    K4_MAPPER_ROOT,
    K4_SELECTION,
    K4_SOURCE_FEATURES,
    K4_TARGET_FEATURES,
    K4_TOP_K,
    K4_TRAIN_OBSERVATIONS,
    K4_WEIGHT_COUNT,
)
from .k4_runtime import K4ReverseMapperBundle
from .k4_selection import validate_k4_selection_artifact
from .mechanism import (
    RECEIPT_NAME as MECHANISM_RECEIPT_NAME,
    RESULT_NAME as MECHANISM_RESULT_NAME,
    load_mechanism_config,
    validate_completed_output as validate_mechanism_output,
)
from .protocol import (
    BIAS_COUNT,
    DIRECTION,
    EXPERIMENT_ROOT,
    SOURCE_FEATURES,
    TARGET_FEATURES,
    TOP_K,
    TRAIN_OBSERVATIONS,
    WEIGHT_COUNT,
    canonical_sha256,
    read_json,
    sha256_file,
)
from .reverse_latency import load_latency_config, verify_latency_completion
from .runtime import ReverseMapperBundle
from .selection_adapter import validate_reverse_selection_artifact


SCHEMA_VERSION = 1
REPORT_KIND = "qwen3_32b_to_14b_formal_reproduction_report"
REPORT_STAGE_NAME = "16_final_report"
PIPELINE_ROOT = EXPERIMENT_ROOT / "runs/formal_pipeline_v1"
REPORT_ROOT = PIPELINE_ROOT / "final_report"

SELECTION = (
    EXPERIMENT_ROOT
    / "artifacts/selection/qwen3_32b_to_14b/selected_layers.json"
)
MAPPER_ROOT = (
    EXPERIMENT_ROOT / "artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01"
)
HELDOUT = (
    EXPERIMENT_ROOT
    / "artifacts/validation/qwen3_32b_to_14b/heldout_activation_reconstruction.json"
)
HELLASWAG_ROOT = PIPELINE_ROOT / "hellaswag_full_k20"
MECHANISM_ROOT = PIPELINE_ROOT / "mechanism_hellaswag_k20"
COQA_ROOT = PIPELINE_ROOT / "coqa_reverse_k20"
K4_COQA_ROOT = PIPELINE_ROOT / "coqa_reverse_k4_retuned"
LATENCY_ROOT = PIPELINE_ROOT / "reverse_latency_k20"
STAGE_RECEIPT_ROOT = PIPELINE_ROOT / "stage_receipts"

CSV_FILENAMES = (
    "selection.csv",
    "mapper.csv",
    "heldout.csv",
    "hellaswag.csv",
    "mechanism.csv",
    "coqa.csv",
    "latency.csv",
)
PAYLOAD_FILENAMES = ("report.json", "report.md", *CSV_FILENAMES)
MANIFEST_FILENAME = "sha256_manifest.json"
RECEIPT_FILENAME = "formal_report_receipt.json"
RECEIPT_SIDECAR_FILENAME = f"{RECEIPT_FILENAME}.sha256.json"
COMPLETE_FILENAMES = (
    *PAYLOAD_FILENAMES,
    MANIFEST_FILENAME,
    RECEIPT_FILENAME,
    RECEIPT_SIDECAR_FILENAME,
)
REPORT_PIPELINE_INTEGRATION = {
    "stage": REPORT_STAGE_NAME,
    "output": str(REPORT_ROOT),
    "cpu_only": True,
    "gpu_lock_required": False,
    "append_after": "15_k4_coqa",
    "append_before_pipeline_completion_receipt": True,
    "completion_validation_must_rerun_upstream_validators": True,
    "required_completion_artifact": str(REPORT_ROOT / RECEIPT_FILENAME),
}

# Each final scientific object must already have a formal stage receipt.  The
# report also re-runs the deeper object-specific validator below; this table is
# the pipeline receipt -> result binding, not a substitute for it.
REQUIRED_STAGE_BINDINGS: dict[str, tuple[Path, ...]] = {
    "03_selection_full": (SELECTION,),
    "05_fit_full": (MAPPER_ROOT / "manifest.json",),
    "06_heldout": (HELDOUT,),
    "07_hellaswag": (HELLASWAG_ROOT / HELLASWAG_RECEIPT_NAME,),
    "08_mechanism": (MECHANISM_ROOT / MECHANISM_RECEIPT_NAME,),
    "09_coqa": (COQA_ROOT / "manifest.json",),
    "10_latency": (LATENCY_ROOT / "formal_latency_receipt.json",),
    "11_k4_selection": (K4_SELECTION,),
    "13_k4_fit_full": (K4_MAPPER_ROOT / "manifest.json",),
    "14_k4_heldout": (K4_HELDOUT,),
    "15_k4_coqa": (K4_COQA_ROOT / K4_COQA_RECEIPT_NAME,),
}

PAPER_HELLASWAG_PUBLIC = {
    "qwen3_14b_standalone_acc_norm_percent": 78.87,
    "qwen3_32b_standalone_acc_norm_percent": 82.65,
    "qwen3_32b_to_14b_mapped_acc_norm_percent": None,
    "boundary": (
        "the paper publishes endpoint-model HellaSwag baselines but does not "
        "publish a numeric Qwen3 32B->14B mapped HellaSwag value"
    ),
}
PAPER_COQA_PUBLIC = {
    "large_to_small_drift_growth_pp_per_turn_approx": 0.33,
    "multi_turn_retuning_max_change_pp_approx": 2.0,
    "exact_qwen3_32b_to_14b_k20_curve_points": None,
    "exact_qwen3_32b_to_14b_k4_curve_points": None,
    "boundary": (
        "Figure 4 discloses task/scale/depths and labels k=20 plus retuned k=4, "
        "but exact selected conversations and the complete scoring protocol are unavailable"
    ),
}


class ReportValidationError(ValueError):
    """An upstream or published-report invariant was violated."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportValidationError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportValidationError(f"{label} is not finite")
    return result


def _artifact(path: str | Path, *, base: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        relative = str(resolved.relative_to(base.resolve()))
    except ValueError:
        relative = str(resolved)
    return {
        "path": str(resolved),
        "relative_path": relative,
        "sha256": sha256_file(resolved),
        "nbytes": resolved.stat().st_size,
    }


def _verify_artifact_row(
    row: Mapping[str, Any], *, required_root: Path = EXPERIMENT_ROOT
) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    try:
        path.relative_to(required_root.resolve())
    except ValueError as exc:
        raise ReportValidationError(f"artifact escaped the reverse experiment: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    if row.get("nbytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
        raise ReportValidationError(f"artifact changed after receipt: {path}")
    return path


def _hashed_sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256.json")


def _identity_pair(path: Path) -> dict[str, Any]:
    return {
        "artifact": _artifact(path),
        "checksum": _artifact(_hashed_sidecar(path)),
    }


def _validate_stage_receipt_set(
    receipt_root: Path = STAGE_RECEIPT_ROOT,
    bindings: Mapping[str, Sequence[Path]] = REQUIRED_STAGE_BINDINGS,
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    definition_digests: set[str] = set()
    for stage, required_paths in bindings.items():
        path = receipt_root / f"{stage}.json"
        verify_hashed_json(path)
        value = read_json(path)
        if (
            value.get("schema_version") != 1
            or value.get("status") != "complete"
            or value.get("stage") != stage
        ):
            raise ReportValidationError(f"formal stage receipt changed: {stage}")
        digest = value.get("definition_digest")
        if not _is_sha256(digest):
            raise ReportValidationError(f"formal stage receipt lacks definition digest: {stage}")
        definition_digests.add(str(digest))
        rows = value.get("artifacts")
        if not isinstance(rows, list) or not rows:
            raise ReportValidationError(f"formal stage receipt lacks artifacts: {stage}")
        if value.get("artifacts_sha256") != canonical_sha256(rows):
            raise ReportValidationError(f"formal stage artifact digest changed: {stage}")
        seen: set[Path] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise ReportValidationError(f"formal stage has malformed artifact: {stage}")
            artifact = _verify_artifact_row(row, required_root=experiment_root)
            if artifact in seen:
                raise ReportValidationError(f"formal stage repeats an artifact: {stage}")
            seen.add(artifact)
        for required in required_paths:
            if required.resolve() not in seen:
                raise ReportValidationError(
                    f"formal stage {stage} does not bind {required.resolve()}"
                )
        receipts[stage] = _identity_pair(path)
    if len(definition_digests) != 1:
        raise ReportValidationError(
            "formal result stages do not share one pipeline definition digest"
        )
    return {
        "pipeline_definition_digest": next(iter(definition_digests)),
        "receipts": receipts,
        "receipt_inventory_sha256": canonical_sha256(receipts),
    }


def _selection_report(
    selection: Mapping[str, Any], *, variant: str, expected_k: int
) -> dict[str, Any]:
    rows = selection.get("selected_source_layers_rank_order")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ReportValidationError(f"{variant} selection does not cover 40 target layers")
    normalized: list[list[int]] = []
    counts: Counter[int] = Counter()
    for target_layer, raw in enumerate(rows):
        if (
            not isinstance(raw, list)
            or len(raw) != expected_k
            or len(set(raw)) != expected_k
            or any(type(value) is not int or not 0 <= value < 64 for value in raw)
        ):
            raise ReportValidationError(
                f"{variant} target layer {target_layer} has an invalid top-k row"
            )
        normalized.append(list(raw))
        counts.update(raw)
    return {
        "variant": variant,
        "k": expected_k,
        "source_layers": 64,
        "target_layers": 40,
        "rows": normalized,
        "rows_sha256": canonical_sha256(normalized),
        "source_layer_frequency": {
            str(layer): counts[layer] for layer in sorted(counts)
        },
        "ranking_scope": "all_64_source_layers_per_target",
        "gold_used_for_selection": False,
    }


def _mapper_diagnostics(
    *,
    variant: str,
    root: Path,
    manifest: Mapping[str, Any],
    expected_k: int,
    source_features: int,
    target_features: int,
    weight_count: int,
    bias_count: int,
) -> dict[str, Any]:
    raw_layers = manifest.get("layers")
    if not isinstance(raw_layers, list) or len(raw_layers) != 40:
        raise ReportValidationError(f"{variant} mapper does not contain 40 layers")
    ordered: list[Mapping[str, Any] | None] = [None] * 40
    for entry in raw_layers:
        if not isinstance(entry, Mapping):
            raise ReportValidationError(f"{variant} mapper has a malformed layer")
        layer = entry.get("target_layer")
        if type(layer) is not int or not 0 <= layer < 40 or ordered[layer] is not None:
            raise ReportValidationError(f"{variant} mapper layer index is invalid")
        ordered[layer] = entry
    if any(value is None for value in ordered):
        raise ReportValidationError(f"{variant} mapper target-layer coverage is incomplete")

    component_values: dict[str, list[float]] = {"key": [], "value": []}
    global_values: dict[str, list[float]] = {"key": [], "value": []}
    residual_values: dict[str, list[float]] = {"key": [], "value": []}
    for layer, entry_or_none in enumerate(ordered):
        assert entry_or_none is not None
        entry = entry_or_none
        sidecar_path = root / "layers" / f"layer_{layer:03d}.json"
        sidecar = read_json(sidecar_path)
        metadata = sidecar.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ReportValidationError(f"{variant} mapper layer {layer} lacks metadata")
        diagnostics = metadata.get("diagnostics")
        if not isinstance(diagnostics, Mapping) or entry.get("diagnostics") != diagnostics:
            raise ReportValidationError(
                f"{variant} manifest diagnostics differ from the hashed layer sidecar"
            )
        for component in ("key", "value"):
            part = diagnostics.get(component)
            if not isinstance(part, Mapping):
                raise ReportValidationError(
                    f"{variant} mapper layer {layer} lacks {component} diagnostics"
                )
            exact = {
                "n_samples": 128_000,
                "source_features": source_features,
                "target_features": target_features,
                "ridge_lambda": 0.01,
                "solver": "cholesky",
                "solution_dtype": "float32",
                "statistics_dtype": "float32",
                "analysis_dtype": "float64",
                "feature_standardization": False,
                "bias_regularized": False,
                "gram_normalized_by_n": False,
                "allow_tf32": False,
                "cholesky_info": 0,
            }
            for field, wanted in exact.items():
                if part.get(field) != wanted:
                    raise ReportValidationError(
                        f"{variant} mapper layer {layer}/{component} changed {field}"
                    )
            component_values[component].append(
                _finite(
                    part.get("train_r2_uniform_channel_average_fp64"),
                    f"{variant} layer {layer}/{component} train uniform R2",
                )
            )
            global_values[component].append(
                _finite(
                    part.get("train_r2_global_fp64"),
                    f"{variant} layer {layer}/{component} train global R2",
                )
            )
            residual_values[component].append(
                _finite(
                    part.get("normal_equation_relative_residual_fp64"),
                    f"{variant} layer {layer}/{component} normal-equation residual",
                )
            )
    total_parameters = weight_count + bias_count
    storage_bytes = total_parameters * 4
    return {
        "variant": variant,
        "k": expected_k,
        "geometry": f"{source_features}_to_{target_features}",
        "source_features": source_features,
        "target_features": target_features,
        "target_layers": 40,
        "components_per_layer": 2,
        "ridge_lambda": 0.01,
        "fit": "centered_unnormalized_ridge_bias_unregularized",
        "forward_activation_dtype": "bfloat16",
        "statistics_and_mapper_dtype": "float32",
        "analysis_dtype": "float64",
        "weight_count": weight_count,
        "bias_count": bias_count,
        "parameter_count": total_parameters,
        "fp32_parameter_bytes": storage_bytes,
        "fp32_parameter_gib": storage_bytes / (1024**3),
        "fit_fingerprint": manifest.get("fit_fingerprint"),
        "train_key_uniform_channel_r2_mean_across_layers": sum(
            component_values["key"]
        )
        / 40,
        "train_value_uniform_channel_r2_mean_across_layers": sum(
            component_values["value"]
        )
        / 40,
        "train_key_global_r2_mean_across_layers": sum(global_values["key"]) / 40,
        "train_value_global_r2_mean_across_layers": sum(global_values["value"])
        / 40,
        "normal_equation_relative_residual_max": max(
            (*residual_values["key"], *residual_values["value"])
        ),
    }


@dataclass(frozen=True)
class _FitArchitectureView:
    """Training-validator view of a verified runtime architecture.

    Runtime bundles use the direction-neutral ``ArchitectureSpec`` field
    names, while the fit payload auditor consumes the training driver's
    compact architecture interface.  The view derives every dimension from
    the already verified runtime bundle; only the frozen calibration count is
    supplied by the corresponding k20/k4 protocol.
    """

    source_layers: int
    target_layers: int
    kv_heads: int
    head_dim: int
    selected_source_layers: int
    n_samples: int

    @property
    def source_features(self) -> int:
        return self.selected_source_layers * self.kv_heads * self.head_dim

    @property
    def target_features(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def expected_weight_count(self) -> int:
        return 2 * self.target_layers * self.source_features * self.target_features

    @property
    def expected_bias_count(self) -> int:
        return 2 * self.target_layers * self.target_features


def _fit_architecture_view(
    architecture: Any,
    *,
    n_samples: int,
) -> _FitArchitectureView:
    source_heads = int(architecture.source_kv_heads)
    target_heads = int(architecture.target_kv_heads)
    source_head_dim = int(architecture.source_head_dim)
    target_head_dim = int(architecture.target_head_dim)
    if source_heads != target_heads:
        raise ReportValidationError(
            "mapper runtime architecture has unequal source/target KV head counts"
        )
    if source_head_dim != target_head_dim:
        raise ReportValidationError(
            "mapper runtime architecture has unequal source/target head dimensions"
        )
    return _FitArchitectureView(
        source_layers=int(architecture.source_layers),
        target_layers=int(architecture.target_layers),
        kv_heads=source_heads,
        head_dim=source_head_dim,
        selected_source_layers=int(architecture.selected_source_layers),
        n_samples=int(n_samples),
    )


def _validate_mapper_sources(
    k20_selection: Mapping[str, Any], k4_selection: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    k20_bundle = ReverseMapperBundle.open(MAPPER_ROOT)
    k20_manifest = read_json(MAPPER_ROOT / "manifest.json")
    k20_rows = k20_selection["selected_source_layers_rank_order"]
    k20_snapshot = MAPPER_ROOT / "inputs/selected_layers.json"
    complete_k20 = validate_k20_payloads(
        MAPPER_ROOT,
        architecture=_fit_architecture_view(
            k20_bundle.architecture,
            n_samples=TRAIN_OBSERVATIONS,
        ),
        fit_fingerprint=k20_bundle.fit_fingerprint,
        selection_sha256=sha256_file(k20_snapshot),
        selected_layers=k20_rows,
    )
    if k20_manifest.get("payload_validation") != complete_k20:
        raise ReportValidationError(
            "k20 mapper manifest payload-validation receipt differs from a fresh tensor audit"
        )
    k20 = _mapper_diagnostics(
        variant="production_k20",
        root=MAPPER_ROOT,
        manifest=k20_manifest,
        expected_k=TOP_K,
        source_features=SOURCE_FEATURES,
        target_features=TARGET_FEATURES,
        weight_count=WEIGHT_COUNT,
        bias_count=BIAS_COUNT,
    )

    k4_bundle = K4ReverseMapperBundle.open(K4_MAPPER_ROOT)
    k4_manifest = read_json(K4_MAPPER_ROOT / "manifest.json")
    k4_rows = k4_selection["selected_source_layers_rank_order"]
    k4_snapshot = K4_MAPPER_ROOT / "inputs/selected_layers.json"
    complete_k4 = validate_k4_payloads(
        K4_MAPPER_ROOT,
        architecture=_fit_architecture_view(
            k4_bundle.architecture,
            n_samples=K4_TRAIN_OBSERVATIONS,
        ),
        fit_fingerprint=k4_bundle.fit_fingerprint,
        selection_sha256=sha256_file(k4_snapshot),
        selected_layers=k4_rows,
    )
    if k4_manifest.get("payload_validation") != complete_k4:
        raise ReportValidationError(
            "k4 mapper manifest payload-validation receipt differs from a fresh tensor audit"
        )
    k4 = _mapper_diagnostics(
        variant="figure4_retuned_k4",
        root=K4_MAPPER_ROOT,
        manifest=k4_manifest,
        expected_k=K4_TOP_K,
        source_features=K4_SOURCE_FEATURES,
        target_features=K4_TARGET_FEATURES,
        weight_count=K4_WEIGHT_COUNT,
        bias_count=K4_BIAS_COUNT,
    )
    if k4_manifest.get("independent_refit_evidence", {}).get(
        "k20_weights_sliced_or_truncated"
    ) is not False:
        raise ReportValidationError("k4 mapper is not an independently fitted mapper")
    k4["independently_refit_from_train_activations"] = True
    k4["k20_weights_sliced_or_truncated"] = False
    return k20, k4


_HELDOUT_METRIC_FIELDS = (
    "uniform_channel_r2_fp64",
    "global_r2_fp64",
    "flattened_global_r2_fp64",
    "mse_fp64",
    "global_sse_fp64",
    "global_sst_fp64",
)


def _heldout_report_from_layers(
    value: Mapping[str, Any], *, variant: str, expected_k: int
) -> dict[str, Any]:
    layers = value.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ReportValidationError(f"{variant} held-out layers are incomplete")
    key: list[float] = []
    val: list[float] = []
    for target_layer, row in enumerate(layers):
        if not isinstance(row, Mapping) or row.get("target_layer") != target_layer:
            raise ReportValidationError(f"{variant} held-out layer order changed")
        selected = row.get("selected_source_layers_rank_order")
        if not isinstance(selected, list) or len(selected) != expected_k:
            raise ReportValidationError(f"{variant} held-out selection width changed")
        for component in ("key", "value"):
            metrics = row.get(component)
            if not isinstance(metrics, Mapping):
                raise ReportValidationError(
                    f"{variant} held-out layer {target_layer} lacks {component}"
                )
            if metrics.get("observations") != 12_800 or metrics.get("channels") != 1_024:
                raise ReportValidationError(f"{variant} held-out dimensions changed")
            for field in _HELDOUT_METRIC_FIELDS:
                _finite(
                    metrics.get(field),
                    f"{variant} held-out layer {target_layer}/{component}/{field}",
                )
        key.append(float(row["key"]["uniform_channel_r2_fp64"]))
        val.append(float(row["value"]["uniform_channel_r2_fp64"]))
    expected_summary = {
        "target_layers": 40,
        "key_uniform_channel_r2_fp64_mean_across_layers": sum(key) / 40,
        "value_uniform_channel_r2_fp64_mean_across_layers": sum(val) / 40,
        "key_uniform_channel_r2_fp64_min": min(key),
        "value_uniform_channel_r2_fp64_min": min(val),
    }
    if value.get("summary") != expected_summary:
        raise ReportValidationError(
            f"{variant} held-out summary differs from raw layer metrics"
        )
    return {
        "variant": variant,
        "scope": "local_independent_fineweb_heldout_not_disclosed_by_paper",
        "sequences": 50,
        "sequence_length": 1024,
        "stride": 4,
        "observations": 12_800,
        **expected_summary,
        "paper_reference_value": None,
        "paper_comparison": "not_available_for_this_local_heldout_split",
    }


def _validate_k20_heldout() -> tuple[dict[str, Any], dict[str, Any]]:
    # This validator reconstructs the complete preflight with full activation
    # payload checksums, reopens the current mapper, and re-derives every metric
    # algebra/summary/acceptance field.  The report intentionally pays that
    # CPU I/O cost instead of trusting the held-out JSON or its checksum marker.
    value = validate_heldout_artifact(HELDOUT)
    return value, _heldout_report_from_layers(
        value, variant="production_k20", expected_k=TOP_K
    )


def _validate_hellaswag() -> tuple[dict[str, Any], dict[str, Any]]:
    validate_hellaswag_output(HELLASWAG_ROOT)
    definition_path = HELLASWAG_ROOT / "run_definition.json"
    fingerprint = validate_hellaswag_definition(read_json(definition_path))
    rows = [
        validate_hellaswag_result(
            HELLASWAG_ROOT / "results" / mode / f"{HELLASWAG_TASK}.json",
            fingerprint=fingerprint,
            expected_mode=mode,
        )
        for mode in HELLASWAG_MODES
    ]
    receipt_path = HELLASWAG_ROOT / HELLASWAG_RECEIPT_NAME
    verify_hashed_json(receipt_path)
    receipt = read_json(receipt_path)
    if receipt.get("results") != rows:
        raise ReportValidationError(
            "HellaSwag receipt metrics differ from raw-sample recomputation"
        )
    by_mode = {row["mode"]: row for row in rows}
    target = by_mode["target_standalone"]
    mapped = by_mode["mapped"]
    return receipt, {
        "task": "HellaSwag",
        "protocol": "full_validation_10042_zero_shot_acc_norm_no_chat",
        "sample_count_per_mode": target["samples"],
        "target_standalone": {
            "correct": target["correct"],
            "acc_norm": target["acc_norm"],
            "acc_norm_percent": target["acc_norm"] * 100.0,
        },
        "mapped": {
            "correct": mapped["correct"],
            "acc_norm": mapped["acc_norm"],
            "acc_norm_percent": mapped["acc_norm"] * 100.0,
        },
        "mapped_over_target_retention": (
            mapped["acc_norm"] / target["acc_norm"]
            if target["acc_norm"]
            else None
        ),
        "mapped_over_target_retention_percent": (
            mapped["acc_norm"] / target["acc_norm"] * 100.0
            if target["acc_norm"]
            else None
        ),
        "paper_public_values": dict(PAPER_HELLASWAG_PUBLIC),
        "local_protocol_is_byte_identical_to_hidden_paper_inputs": False,
    }


def _validate_mechanism() -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = validate_mechanism_output(MECHANISM_ROOT)
    if receipt is None:
        raise ReportValidationError("formal mechanism receipt is missing")
    result_path = MECHANISM_ROOT / MECHANISM_RESULT_NAME
    verify_hashed_json(result_path)
    result = read_json(result_path)
    headline = result.get("headline")
    if not isinstance(headline, Mapping):
        raise ReportValidationError("mechanism result lacks recomputed headline")
    configuration = load_mechanism_config()
    paper = configuration["paper_reference"]
    return receipt, {
        "scope": result.get("metric_scope"),
        "documents": result.get("documents"),
        "scoring_sequences": result.get("scoring_sequences"),
        "scoring_input_tokens": result.get("scoring_input_tokens"),
        **{field: _finite(value, f"mechanism {field}") for field, value in headline.items()},
        "paper_public_values": {
            "attention_cosine_retention_pearson_r_cross_12_pairs": paper[
                "published_attention_cosine_retention_pearson_r"
            ],
            "calibration_r2k_retention_pearson_r_cross_12_pairs": paper[
                "published_calibration_r2k_retention_pearson_r"
            ],
            "qwen3_32b_to_14b_single_pair_mechanism_values": None,
        },
        "paper_comparison": (
            "cross-pair Pearson correlations are context only; a one-pair "
            "correlation and bit-exact scalar match are not defined"
        ),
    }


def _verify_control_link(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    field: str,
    filename: str,
) -> None:
    path = output / filename
    sidecar = verify_hashed_json(path)
    if manifest.get(field) != {"sha256": sidecar["sha256"], "nbytes": sidecar["nbytes"]}:
        raise ReportValidationError(f"CoQA manifest {field} binding changed")


def _read_verified_jsonl(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    field: str,
    filename: str,
    expected_rows: int,
) -> list[dict[str, Any]]:
    path = output / filename
    sidecar_path = _hashed_sidecar(path)
    sidecar = read_json(sidecar_path)
    expected = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "rows": expected_rows,
    }
    if sidecar != expected or manifest.get(field) != expected:
        raise ReportValidationError(f"CoQA {field} JSONL binding changed")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReportValidationError(
                    f"CoQA {field} line {line_number} is invalid"
                ) from exc
            if not isinstance(row, dict):
                raise ReportValidationError(f"CoQA {field} line is not an object")
            rows.append(row)
    if len(rows) != expected_rows:
        raise ReportValidationError(f"CoQA {field} row coverage changed")
    return rows


def _validate_coqa_runtime_setup(
    setup: Mapping[str, Any], *, fingerprint: str, mapper: Mapping[str, Any]
) -> None:
    if (
        setup.get("schema_version") != 1
        or setup.get("status") != "ready"
        or setup.get("request_fingerprint") != fingerprint
        or setup.get("semantic_role_binding")
        != {
            "source": {"model": "Qwen3-32B", "capture_native_forward_role": "target"},
            "target": {"model": "Qwen3-14B", "capture_native_forward_role": "source"},
        }
    ):
        raise ReportValidationError("k20 CoQA runtime identity changed")
    runtime_mapper = setup.get("mapper")
    if not isinstance(runtime_mapper, Mapping):
        raise ReportValidationError("k20 CoQA runtime mapper identity is missing")
    expected_mapper = {
        "direction": DIRECTION,
        "manifest": str((MAPPER_ROOT / "manifest.json").resolve()),
        "manifest_sha256": mapper["manifest_sha256"],
        "fit_fingerprint": mapper["fit_fingerprint"],
        "layers": 40,
        "selected_source_layers": 20,
        "token_chunk": 1024,
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
        "target_layer_devices": [
            *(["cuda:5"] * 13),
            *(["cuda:6"] * 14),
            *(["cuda:7"] * 13),
        ],
        "resident_parameter_bytes": (WEIGHT_COUNT + BIAS_COUNT) * 4,
    }
    for field, wanted in expected_mapper.items():
        if runtime_mapper.get(field) != wanted:
            raise ReportValidationError(f"k20 CoQA runtime mapper changed {field}")
    for role, devices, counts, native_role, layers in (
        ("source", [0, 1, 2, 3, 4], [12, 13, 14, 13, 12], "target", 64),
        ("target", [5, 6, 7], [13, 14, 13], "source", 40),
    ):
        sharding = setup.get(f"{role}_model_sharding")
        expected_layer_devices = [
            device
            for device, count in zip(devices, counts, strict=True)
            for _ in range(count)
        ]
        if (
            not isinstance(sharding, Mapping)
            or sharding.get("status") != "passed"
            or sharding.get("role") != native_role
            or sharding.get("mode") != "explicit_exact_counts"
            or sharding.get("devices") != devices
            or sharding.get("declared_layer_counts") != counts
            or sharding.get("expected_layer_counts") != counts
            or sharding.get("actual_layer_counts") != counts
            or sharding.get("total_layers") != layers
            or sharding.get("layer_devices") != expected_layer_devices
            or sharding.get("actual_parameters_inspected_for_every_layer") is not True
        ):
            raise ReportValidationError(f"k20 CoQA {role} sharding proof changed")


def _validate_selection_manifest(value: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = value.get("rows")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "complete"
        or value.get("paper_bit_exact_claimed") is not False
        or value.get("selection_uses_answers_or_gold") is not False
        or not isinstance(rows, list)
        or len(rows) != 100
        or value.get("rows_sha256") != canonical_sha256(rows)
    ):
        raise ReportValidationError("CoQA selection manifest changed")
    by_index: dict[int, Mapping[str, Any]] = {}
    domains: Counter[str] = Counter()
    ranks: dict[str, set[int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ReportValidationError("CoQA selection row is malformed")
        index = row.get("selection_index")
        domain = row.get("domain")
        rank = row.get("rank_within_domain")
        if (
            type(index) is not int
            or index in by_index
            or not isinstance(domain, str)
            or type(rank) is not int
        ):
            raise ReportValidationError("CoQA selection key is malformed")
        by_index[index] = row
        domains[domain] += 1
        ranks.setdefault(domain, set()).add(rank)
    if set(by_index) != set(range(100)):
        raise ReportValidationError("CoQA selection indices are not 0..99")
    if set(domains) != set(coqa_eval.PAPER_DOMAINS) or any(
        domains[domain] != 20 or ranks[domain] != set(range(1, 21))
        for domain in coqa_eval.PAPER_DOMAINS
    ):
        raise ReportValidationError("CoQA selection is not 20 rows per paper domain")
    return by_index


def _validate_k20_coqa() -> tuple[dict[str, Any], dict[str, Any]]:
    output = COQA_ROOT
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    definition_path = output / "run_definition.json"
    for path in (manifest_path, summary_path, definition_path):
        verify_hashed_json(path)
    manifest = read_json(manifest_path)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "direction": DIRECTION,
        "task": "coqa_multi_turn_handoff",
        "paper_bit_exact_claimed": False,
        "headline_handoff_depths": list(coqa_eval.HEADLINE_DEPTHS),
        "records": 1000,
        "expected_records": 1000,
    }
    for field, wanted in fixed.items():
        if manifest.get(field) != wanted:
            raise ReportValidationError(f"k20 CoQA manifest changed {field}")
    definition = read_json(definition_path)
    configuration = coqa_eval.load_coqa_config()
    variant_fields = coqa_eval._variant_run_fields(coqa_eval.DEFAULT_RUN_VARIANT)
    for field, wanted in {
        "schema_version": 1,
        "kind": variant_fields["kind"],
        "direction": DIRECTION,
        "paper_reference": variant_fields["paper_reference"],
        "modes": list(coqa_eval.MODES),
        "prompt": configuration["prompt"],
        "generation": configuration["generation"],
        "metric": configuration["metric"],
        "runtime": configuration["runtime"],
    }.items():
        if definition.get(field) != wanted:
            raise ReportValidationError(f"k20 CoQA run definition changed {field}")
    fingerprint = definition.get("request_fingerprint")
    unsigned = dict(definition)
    unsigned.pop("request_fingerprint", None)
    if not _is_sha256(fingerprint) or canonical_sha256(unsigned) != fingerprint:
        raise ReportValidationError("k20 CoQA request fingerprint changed")
    if manifest.get("request_fingerprint") != fingerprint:
        raise ReportValidationError("k20 CoQA manifest fingerprint changed")
    expected_code = {
        "reverse_experiment": code_identity(coqa_eval.CODE_FILES),
        "shared_read_only_runtime": code_identity(
            coqa_eval.SHARED_CODE_FILES, root=coqa_eval.REPOSITORY_ROOT
        ),
    }
    if definition.get("code") != expected_code:
        raise ReportValidationError("k20 CoQA code identity changed")
    mapper = mapper_manifest_identity(MAPPER_ROOT / "manifest.json")
    if definition.get("artifacts", {}).get("mapper") != mapper:
        raise ReportValidationError("k20 CoQA mapper identity changed")
    for field, filename in (
        ("run_definition", "run_definition.json"),
        ("dataset_receipt", "dataset_receipt.json"),
        ("selection_manifest", "selection_manifest.json"),
        ("runtime_setup", "runtime_setup.json"),
    ):
        _verify_control_link(output, manifest, field=field, filename=filename)
    dataset_receipt = read_json(output / "dataset_receipt.json")
    if definition.get("dataset_receipt") != dataset_receipt:
        raise ReportValidationError("k20 CoQA dataset receipt changed")
    if dataset_receipt.get("dataset") != coqa_eval.verify_dataset_file(configuration):
        raise ReportValidationError("k20 CoQA dataset payload changed")
    selection_manifest = read_json(output / "selection_manifest.json")
    selected = _validate_selection_manifest(selection_manifest)
    if (
        definition.get("selection_manifest_sha256")
        != sha256_file(output / "selection_manifest.json")
        or definition.get("selection_rows_sha256")
        != selection_manifest.get("rows_sha256")
    ):
        raise ReportValidationError("k20 CoQA selection binding changed")
    runtime_setup = read_json(output / "runtime_setup.json")
    _validate_coqa_runtime_setup(runtime_setup, fingerprint=str(fingerprint), mapper=mapper)

    token_plans = _read_verified_jsonl(
        output,
        manifest,
        field="token_plans",
        filename="token_plans.jsonl",
        expected_rows=1000,
    )
    # The k4 validator's helper is mapper-agnostic and validates all token-plan
    # keys, hashes, lengths, deltas, and selection bindings.
    from .coqa_k4_eval import _validate_token_plans

    _validate_token_plans(token_plans, selection_manifest)
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1000:
        raise ReportValidationError("k20 CoQA checkpoint inventory is incomplete")
    if manifest.get("checkpoint_manifest_sha256") != canonical_sha256(checkpoints):
        raise ReportValidationError("k20 CoQA checkpoint inventory digest changed")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    mode_counts = {"target_standalone": 0, "mapped": 0}
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise ReportValidationError("k20 CoQA checkpoint row is malformed")
        relative = str(checkpoint.get("path", ""))
        path = (output / relative).resolve()
        if output.resolve() not in path.parents or relative in seen:
            raise ReportValidationError("k20 CoQA checkpoint path escaped or repeated")
        seen.add(relative)
        if (
            checkpoint.get("nbytes") != path.stat().st_size
            or checkpoint.get("sha256") != sha256_file(path)
        ):
            raise ReportValidationError(f"k20 CoQA checkpoint changed: {path}")
        verify_hashed_json(path)
        record = read_json(path)
        mode = str(record.get("mode", ""))
        index = record.get("selection_index")
        depth = record.get("handoff_depth")
        if (
            record.get("schema_version") != 1
            or record.get("status") != "complete"
            or record.get("request_fingerprint") != fingerprint
            or mode not in mode_counts
            or type(index) is not int
            or index not in selected
            or type(depth) is not int
            or depth not in coqa_eval.HEADLINE_DEPTHS
        ):
            raise ReportValidationError("k20 CoQA checkpoint scientific identity changed")
        selection_row = selected[index]
        import hashlib

        expected_relative = (
            f"records/{mode}/{index:03d}_"
            f"{hashlib.sha256(str(selection_row['conversation_id']).encode('utf-8')).hexdigest()[:16]}_"
            f"depth_{depth:02d}.json"
        )
        if relative != expected_relative:
            raise ReportValidationError("k20 CoQA checkpoint path/key changed")
        for field in ("conversation_id", "domain", "rank_within_domain", "rank_sha256"):
            if record.get(field) != selection_row.get(field):
                raise ReportValidationError(
                    f"k20 CoQA checkpoint selection field changed: {field}"
                )
        if record.get("handoff_executed") is not (mode == "mapped"):
            raise ReportValidationError("k20 CoQA handoff evidence changed")
        mode_counts[mode] += 1
        records.append(record)
    if mode_counts != {"target_standalone": 500, "mapped": 500}:
        raise ReportValidationError("k20 CoQA mode coverage changed")
    recomputed = coqa_eval.summarize_records(records)
    if read_json(summary_path) != recomputed:
        raise ReportValidationError("k20 CoQA summary differs from raw checkpoints")
    raw_records = _read_verified_jsonl(
        output,
        manifest,
        field="raw_records",
        filename="raw_records.jsonl",
        expected_rows=1000,
    )
    if raw_records != records or (output / "raw_records.jsonl").read_bytes() != canonical_jsonl_bytes(records):
        raise ReportValidationError("k20 CoQA raw JSONL differs from checkpoints")
    if manifest.get("summary") != verify_hashed_json(summary_path):
        raise ReportValidationError("k20 CoQA summary manifest binding changed")
    return manifest, recomputed


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ReportValidationError("linear slope needs paired values")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denominator = sum((value - mean_x) ** 2 for value in xs)
    if denominator == 0:
        raise ReportValidationError("linear slope has zero x variance")
    return sum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    ) / denominator


def _coqa_report_from_summary(
    summary: Mapping[str, Any], *, variant: str, k: int
) -> dict[str, Any]:
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "complete"
        or summary.get("primary_metric") != "official_normalized_word_f1"
    ):
        raise ReportValidationError(f"{variant} CoQA summary identity changed")
    modes = summary.get("modes")
    drift = summary.get("drift_by_handoff_depth")
    if not isinstance(modes, Mapping) or not isinstance(drift, Mapping):
        raise ReportValidationError(f"{variant} CoQA summary is malformed")
    by_depth: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []
    for depth in coqa_eval.HEADLINE_DEPTHS:
        target_row = modes["target_standalone"]["by_handoff_depth"][str(depth)]
        mapped_row = modes["mapped"]["by_handoff_depth"][str(depth)]
        drift_row = drift[str(depth)]
        target = _finite(target_row["official_word_f1"], "CoQA target F1")
        mapped = _finite(mapped_row["official_word_f1"], "CoQA mapped F1")
        expected_pp = (target - mapped) * 100.0
        if not math.isclose(
            _finite(drift_row["drift_percentage_points"], "CoQA drift"),
            expected_pp,
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ReportValidationError(f"{variant} CoQA depth drift changed")
        by_depth.append(
            {
                "handoff_depth": depth,
                "target_official_word_f1": target,
                "target_official_word_f1_percent": target * 100.0,
                "mapped_official_word_f1": mapped,
                "mapped_official_word_f1_percent": mapped * 100.0,
                "drift_target_minus_mapped_percentage_points": expected_pp,
                "mapped_over_target_retention": mapped / target if target else None,
                "target_cap_hits": target_row["cap_hits"],
                "mapped_cap_hits": mapped_row["cap_hits"],
            }
        )
        xs.append(float(depth))
        ys.append(expected_pp)
    target_overall = _finite(
        modes["target_standalone"]["official_word_f1"], "CoQA target overall"
    )
    mapped_overall = _finite(
        modes["mapped"]["official_word_f1"], "CoQA mapped overall"
    )
    return {
        "variant": variant,
        "k": k,
        "protocol": "local_deterministic_completion_of_undisclosed_paper_details",
        "paper_bit_exact_claimed": False,
        "records": 1000,
        "conversations": 100,
        "handoff_depths": list(coqa_eval.HEADLINE_DEPTHS),
        "target_official_word_f1": target_overall,
        "target_official_word_f1_percent": target_overall * 100.0,
        "mapped_official_word_f1": mapped_overall,
        "mapped_official_word_f1_percent": mapped_overall * 100.0,
        "overall_drift_target_minus_mapped_percentage_points": (
            target_overall - mapped_overall
        )
        * 100.0,
        "overall_mapped_over_target_retention": (
            mapped_overall / target_overall if target_overall else None
        ),
        "drift_linear_slope_pp_per_turn_over_depths_1_3_5_7_10": _linear_slope(xs, ys),
        "by_handoff_depth": by_depth,
    }


def _validate_coqa() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    k20_manifest, k20_summary = _validate_k20_coqa()
    k4_receipt = validate_k4_receipt(K4_COQA_ROOT)
    k4_summary = read_json(K4_COQA_ROOT / "summary.json")
    # validate_k4_receipt recomputes the summary from all 1,000 checkpoints.
    k20_selection = read_json(COQA_ROOT / "selection_manifest.json")
    k4_selection = read_json(K4_COQA_ROOT / "selection_manifest.json")
    if (
        k20_selection.get("rows_sha256") != k4_selection.get("rows_sha256")
        or k20_selection.get("rows") != k4_selection.get("rows")
    ):
        raise ReportValidationError("k20 and k4 CoQA did not use the same conversations")
    return (
        k20_manifest,
        k4_receipt,
        {
            "variants": [
                _coqa_report_from_summary(
                    k20_summary, variant="production_k20", k=TOP_K
                ),
                _coqa_report_from_summary(
                    k4_summary, variant="figure4_retuned_k4", k=K4_TOP_K
                ),
            ],
            "paper_public_values": dict(PAPER_COQA_PUBLIC),
            "local_protocol_boundary": (
                "same pinned 100 conversations, prompt, gold-history policy, decoder, "
                "and official word-F1 scorer for k20/k4; only mapper family changes"
            ),
        },
    )


def _validate_latency() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = verify_latency_completion(LATENCY_ROOT)
    if manifest is None:
        raise ReportValidationError("formal staged latency receipt is missing")
    summary = read_json(LATENCY_ROOT / "summary.json")
    hardware = read_json(LATENCY_ROOT / "hardware.json")
    configuration = load_latency_config()
    paper = configuration["paper_h100_reference_only"]
    rows = []
    for raw in summary["rows"]:
        length = int(raw["prefix_length"])
        target = _finite(raw["target_reprefill_median_ms"], "latency target median")
        mapper = _finite(raw["mapper_median_ms"], "latency mapper median")
        speedup = target / mapper
        if not math.isclose(
            _finite(raw["speedup_reprefill_over_mapper"], "latency speedup"),
            speedup,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ReportValidationError("latency speedup differs from raw medians")
        anchor = paper["anchors"].get(str(length))
        paper_row = None
        if anchor is not None:
            paper_row = {
                **anchor,
                "speedup_reprefill_over_mapper": (
                    float(anchor["target_reprefill_ms"]) / float(anchor["mapper_ms"])
                ),
            }
        rows.append(
            {
                "prefix_length": length,
                "local_target_reprefill_median_ms": target,
                "local_mapper_median_ms": mapper,
                "local_speedup_reprefill_over_mapper": speedup,
                "paper_h100_reference": paper_row,
            }
        )
    devices = hardware.get("devices")
    if not isinstance(devices, list) or len(devices) != 8:
        raise ReportValidationError("latency hardware inventory changed")
    device_names = [str(row.get("name")) for row in devices]
    return manifest, {
        "protocol": {
            "lengths": summary["lengths"],
            "warmup": summary["warmup"],
            "timed_repeats": summary["timed_repeats"],
            "source_prefill_timed": False,
            "cpu_staging_timed": False,
            "gpu_restage_timed": False,
            "target_scope": "14B_transformer_body_with_KV_excluding_lm_head",
            "mapper_scope": (
                "resident_32B_KV_handoff_inverse_rope_FP32_affine_target_rope_BF16_cache_build"
            ),
        },
        "local_hardware": {
            "device_names": device_names,
            "nvidia_driver_version": hardware.get("nvidia_driver_version"),
            "nvlink_present": hardware.get("nvlink_present"),
            "all_cross_device_peer_access": hardware.get(
                "all_cross_device_peer_access"
            ),
            "paper_h100_nvlink_hardware_match": hardware.get(
                "paper_h100_nvlink_hardware_match"
            ),
        },
        "paper_h100_reference": {
            "hardware": paper["hardware"],
            "anchors": paper["anchors"],
            "reported_speedup_range": paper["reported_speedup_range"],
        },
        "absolute_latency_bit_exact_comparison_claimed": False,
        "hardware_limitation": (
            "local RTX 4090 PCIe results are not absolute-latency replicas of the "
            "paper's 8xH100 NVLink measurements; local relative speedup remains reportable"
        ),
        "rows": rows,
    }


def collect_formal_results() -> dict[str, Any]:
    """Revalidate every upstream formal artifact and derive one report payload."""

    stage_receipts = _validate_stage_receipt_set()
    k20_selection_raw = validate_reverse_selection_artifact(SELECTION)
    k4_selection_raw = validate_k4_selection_artifact(K4_SELECTION)
    selections = [
        _selection_report(
            k20_selection_raw, variant="production_k20", expected_k=TOP_K
        ),
        _selection_report(
            k4_selection_raw, variant="figure4_retuned_k4", expected_k=K4_TOP_K
        ),
    ]
    mapper_k20, mapper_k4 = _validate_mapper_sources(
        k20_selection_raw, k4_selection_raw
    )
    k20_heldout_raw, k20_heldout = _validate_k20_heldout()
    k4_heldout_raw = validate_k4_heldout_artifact(K4_HELDOUT)
    k4_heldout = _heldout_report_from_layers(
        k4_heldout_raw, variant="figure4_retuned_k4", expected_k=K4_TOP_K
    )
    hellaswag_receipt, hellaswag = _validate_hellaswag()
    mechanism_receipt, mechanism = _validate_mechanism()
    coqa_k20_manifest, coqa_k4_receipt, coqa = _validate_coqa()
    latency_manifest, latency = _validate_latency()

    source_inventory = {
        "stage_receipts": stage_receipts,
        "selection": {
            "k20": {
                "artifact": _artifact(SELECTION),
                "checksum": _artifact(SELECTION.with_suffix(".sha256.json")),
            },
            "k4": {
                "artifact": _artifact(K4_SELECTION),
                "checksum": _artifact(K4_SELECTION.with_suffix(".sha256.json")),
            },
        },
        "mappers": {
            "k20": {
                "manifest": _artifact(MAPPER_ROOT / "manifest.json"),
                "checksum": _artifact(MAPPER_ROOT / "manifest.sha256.json"),
            },
            "k4": {
                "manifest": _artifact(K4_MAPPER_ROOT / "manifest.json"),
                "checksum": _artifact(K4_MAPPER_ROOT / "manifest.sha256.json"),
            },
        },
        "heldout": {
            "k20": {
                "artifact": _artifact(HELDOUT),
                "checksum": _artifact(HELDOUT.with_suffix(".sha256.json")),
                "raw_layers_sha256": canonical_sha256(k20_heldout_raw["layers"]),
            },
            "k4": {
                "artifact": _artifact(K4_HELDOUT),
                "checksum": _artifact(K4_HELDOUT.with_suffix(".sha256.json")),
                "raw_layers_sha256": canonical_sha256(k4_heldout_raw["layers"]),
            },
        },
        "hellaswag": {
            "receipt": _identity_pair(HELLASWAG_ROOT / HELLASWAG_RECEIPT_NAME),
            "receipt_payload_sha256": canonical_sha256(hellaswag_receipt),
            "target_result": _artifact(
                HELLASWAG_ROOT / "results/target_standalone/hellaswag.json"
            ),
            "mapped_result": _artifact(
                HELLASWAG_ROOT / "results/mapped/hellaswag.json"
            ),
        },
        "mechanism": {
            "receipt": _identity_pair(MECHANISM_ROOT / MECHANISM_RECEIPT_NAME),
            "receipt_payload_sha256": canonical_sha256(mechanism_receipt),
            "result": _identity_pair(MECHANISM_ROOT / MECHANISM_RESULT_NAME),
        },
        "coqa": {
            "k20_manifest": _identity_pair(COQA_ROOT / "manifest.json"),
            "k20_summary": _identity_pair(COQA_ROOT / "summary.json"),
            "k20_raw_records": _identity_pair(COQA_ROOT / "raw_records.jsonl"),
            "k20_manifest_payload_sha256": canonical_sha256(coqa_k20_manifest),
            "k4_receipt": _identity_pair(K4_COQA_ROOT / K4_COQA_RECEIPT_NAME),
            "k4_summary": _identity_pair(K4_COQA_ROOT / "summary.json"),
            "k4_raw_records": _identity_pair(K4_COQA_ROOT / "raw_records.jsonl"),
            "k4_receipt_payload_sha256": canonical_sha256(coqa_k4_receipt),
        },
        "latency": {
            "receipt": _identity_pair(LATENCY_ROOT / "formal_latency_receipt.json"),
            "manifest": _identity_pair(LATENCY_ROOT / "manifest.json"),
            "summary": _identity_pair(LATENCY_ROOT / "summary.json"),
            "hardware": _identity_pair(LATENCY_ROOT / "hardware.json"),
            "manifest_payload_sha256": canonical_sha256(latency_manifest),
        },
        "configs": {
            name: _artifact(EXPERIMENT_ROOT / f"config/{name}.yaml")
            for name in (
                "mapper",
                "mapper_k4_coqa",
                "eval",
                "mechanism",
                "coqa",
                "coqa_k4",
                "latency",
            )
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": REPORT_KIND,
        "direction": DIRECTION,
        "endpoints": {
            "semantic_source": "Qwen3-32B",
            "semantic_target": "Qwen3-14B",
        },
        "paper": {
            "identity": "arXiv:2608.03893v1",
            "reverse_five_task_table_published": False,
            "reverse_mapped_hellaswag_number_published": False,
            "reverse_single_pair_mechanism_scalars_published": False,
            "exact_coqa_curve_points_and_hidden_protocol_published": False,
            "h100_latency_anchors_published": True,
        },
        "claim_boundary": {
            "paper_bit_exact_reproduction_claimed": False,
            "mapper_equations_and_disclosed_k_values_reproduced": True,
            "local_deterministic_completions": [
                "HellaSwag harness/dataset revision",
                "mechanism sample IDs and aggregation order",
                "CoQA conversation IDs, prompt, history, decoding and scorer details",
            ],
            "heldout_activation_split": "local_validation_not_paper_disclosed",
            "latency_hardware": "8xRTX4090_PCIe_not_8xH100_NVLink",
        },
        "source_inventory": source_inventory,
        "source_inventory_sha256": canonical_sha256(source_inventory),
        "selection": selections,
        "mappers": [mapper_k20, mapper_k4],
        "heldout_activation_reconstruction": [k20_heldout, k4_heldout],
        "hellaswag": hellaswag,
        "mechanism": mechanism,
        "coqa": coqa,
        "latency": latency,
        "aggregation": {
            "headlines_recomputed_from_raw_artifacts": True,
            "upstream_summary_values_trusted_without_recomputation": False,
            "mapper_safetensors_freshly_loaded_and_finite_checked": True,
            "formal_stage_receipts_reverified": sorted(REQUIRED_STAGE_BINDINGS),
        },
    }


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _render_csvs(report: Mapping[str, Any]) -> dict[str, bytes]:
    selection_by_variant = {row["variant"]: row for row in report["selection"]}
    selection_rows = []
    for target_layer in range(40):
        selection_rows.append(
            (
                target_layer,
                " ".join(
                    map(
                        str,
                        selection_by_variant["production_k20"]["rows"][target_layer],
                    )
                ),
                " ".join(
                    map(
                        str,
                        selection_by_variant["figure4_retuned_k4"]["rows"][target_layer],
                    )
                ),
            )
        )
    mapper_rows = [
        (
            row["variant"],
            row["k"],
            row["source_features"],
            row["target_features"],
            row["weight_count"],
            row["bias_count"],
            repr(row["fp32_parameter_gib"]),
            repr(row["train_key_uniform_channel_r2_mean_across_layers"]),
            repr(row["train_value_uniform_channel_r2_mean_across_layers"]),
            repr(row["normal_equation_relative_residual_max"]),
        )
        for row in report["mappers"]
    ]
    heldout_rows = [
        (
            row["variant"],
            repr(row["key_uniform_channel_r2_fp64_mean_across_layers"]),
            repr(row["value_uniform_channel_r2_fp64_mean_across_layers"]),
            repr(row["key_uniform_channel_r2_fp64_min"]),
            repr(row["value_uniform_channel_r2_fp64_min"]),
            row["scope"],
        )
        for row in report["heldout_activation_reconstruction"]
    ]
    hella = report["hellaswag"]
    hellaswag_rows = [
        (
            mode,
            hella[mode]["correct"],
            hella["sample_count_per_mode"],
            repr(hella[mode]["acc_norm"]),
            repr(hella[mode]["acc_norm_percent"]),
        )
        for mode in ("target_standalone", "mapped")
    ]
    mechanism_rows = [
        (field, repr(report["mechanism"][field]))
        for field in (
            "hellaswag_eval_domain_r2_k_content",
            "hellaswag_eval_domain_r2_k_rope_cache",
            "hellaswag_eval_domain_r2_v",
            "key_error_concentration",
            "value_error_concentration",
            "attention_output_cosine",
        )
    ]
    coqa_rows = []
    for variant in report["coqa"]["variants"]:
        for row in variant["by_handoff_depth"]:
            coqa_rows.append(
                (
                    variant["variant"],
                    variant["k"],
                    row["handoff_depth"],
                    repr(row["target_official_word_f1_percent"]),
                    repr(row["mapped_official_word_f1_percent"]),
                    repr(row["drift_target_minus_mapped_percentage_points"]),
                    repr(row["mapped_over_target_retention"]),
                    repr(
                        variant[
                            "drift_linear_slope_pp_per_turn_over_depths_1_3_5_7_10"
                        ]
                    ),
                )
            )
    latency_rows = []
    for row in report["latency"]["rows"]:
        anchor = row["paper_h100_reference"] or {}
        latency_rows.append(
            (
                row["prefix_length"],
                repr(row["local_target_reprefill_median_ms"]),
                repr(row["local_mapper_median_ms"]),
                repr(row["local_speedup_reprefill_over_mapper"]),
                anchor.get("target_reprefill_ms", ""),
                anchor.get("mapper_ms", ""),
                anchor.get("speedup_reprefill_over_mapper", ""),
                "RTX4090_PCIe_local_vs_H100_NVLink_paper_not_absolute_bit_exact",
            )
        )
    return {
        "selection.csv": _csv_bytes(
            ("target_layer", "production_k20_source_layers", "retuned_k4_source_layers"),
            selection_rows,
        ),
        "mapper.csv": _csv_bytes(
            (
                "variant",
                "k",
                "source_features",
                "target_features",
                "weight_count",
                "bias_count",
                "fp32_parameter_gib",
                "train_key_uniform_channel_r2_mean",
                "train_value_uniform_channel_r2_mean",
                "normal_equation_relative_residual_max",
            ),
            mapper_rows,
        ),
        "heldout.csv": _csv_bytes(
            (
                "variant",
                "key_uniform_channel_r2_mean",
                "value_uniform_channel_r2_mean",
                "key_uniform_channel_r2_min",
                "value_uniform_channel_r2_min",
                "scope",
            ),
            heldout_rows,
        ),
        "hellaswag.csv": _csv_bytes(
            ("mode", "correct", "samples", "acc_norm", "acc_norm_percent"),
            hellaswag_rows,
        ),
        "mechanism.csv": _csv_bytes(("metric", "local_value"), mechanism_rows),
        "coqa.csv": _csv_bytes(
            (
                "variant",
                "k",
                "handoff_depth",
                "target_f1_percent",
                "mapped_f1_percent",
                "drift_pp",
                "mapped_over_target",
                "drift_slope_pp_per_turn",
            ),
            coqa_rows,
        ),
        "latency.csv": _csv_bytes(
            (
                "prefix_length",
                "local_target_reprefill_median_ms",
                "local_mapper_median_ms",
                "local_speedup",
                "paper_h100_target_reprefill_ms",
                "paper_h100_mapper_ms",
                "paper_h100_speedup",
                "comparability",
            ),
            latency_rows,
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3-32B → Qwen3-14B KV 映射正式复现报告",
        "",
        "本报告由正式产物自动重验生成。所有 headline 均从逐样本、逐层或原始计时样本重新计算，不直接信任已有 summary。",
        "",
        "## 结论边界",
        "",
        "论文公开了映射公式、production k=20、Figure 4 retuned k=4、CoQA 高层协议、跨 12 个方向的机制相关性和 H100 延迟锚点；但没有公开该反向映射的五任务主表、mapped HellaSwag 数值、单对机制标量或完整 CoQA 隐藏协议。因此这里不填造这些未公开数字，也不声称 bit-exact。",
        "",
        "本机延迟硬件为 8×RTX 4090（PCIe，非论文 8×H100 NVLink）。绝对毫秒值不可作 bit-exact 比较；本机相同计时边界下的相对加速可报告。",
        "",
        "## Mapper 与本地 held-out 重建",
        "",
        "| variant | k | geometry | FP32 GiB | train K R² | train V R² | held-out K R² | held-out V R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    heldout = {
        row["variant"]: row for row in report["heldout_activation_reconstruction"]
    }
    for mapper in report["mappers"]:
        validation = heldout[mapper["variant"]]
        lines.append(
            "| {variant} | {k} | {source_features}→{target_features} | {gib:.4f} | "
            "{train_k:.6f} | {train_v:.6f} | {held_k:.6f} | {held_v:.6f} |".format(
                variant=mapper["variant"],
                k=mapper["k"],
                source_features=mapper["source_features"],
                target_features=mapper["target_features"],
                gib=mapper["fp32_parameter_gib"],
                train_k=mapper["train_key_uniform_channel_r2_mean_across_layers"],
                train_v=mapper["train_value_uniform_channel_r2_mean_across_layers"],
                held_k=validation[
                    "key_uniform_channel_r2_fp64_mean_across_layers"
                ],
                held_v=validation[
                    "value_uniform_channel_r2_fp64_mean_across_layers"
                ],
            )
        )
    hella = report["hellaswag"]
    lines.extend(
        [
            "",
            "held-out 是本地独立 FineWeb 验证集，不是论文公开的 calibration scalar。k=4 权重从训练激活独立重拟合，并非截断 k=20 权重。",
            "",
            "## HellaSwag",
            "",
            "| mode | correct / 10042 | acc_norm |",
            "|---|---:|---:|",
            f"| 14B target standalone | {hella['target_standalone']['correct']} / 10042 | {hella['target_standalone']['acc_norm_percent']:.4f}% |",
            f"| 32B→14B mapped | {hella['mapped']['correct']} / 10042 | {hella['mapped']['acc_norm_percent']:.4f}% |",
            "",
            f"Mapped/Target retention：{hella['mapped_over_target_retention_percent']:.4f}%。论文公开 14B standalone 参考为 78.87%，但未公开 32B→14B mapped 数值。",
            "",
            "## HellaSwag-domain 机制诊断",
            "",
            "| metric | local value |",
            "|---|---:|",
        ]
    )
    for field, label in (
        ("hellaswag_eval_domain_r2_k_content", "content-key R²"),
        ("hellaswag_eval_domain_r2_k_rope_cache", "RoPE-cache key R²"),
        ("hellaswag_eval_domain_r2_v", "value R²"),
        ("key_error_concentration", "key error concentration"),
        ("value_error_concentration", "value error concentration"),
        ("attention_output_cosine", "attention-output cosine"),
    ):
        lines.append(f"| {label} | {report['mechanism'][field]:.6f} |")
    lines.extend(
        [
            "",
            "论文的 attention cosine r=+0.57 与 calibration R²K r=-0.20 是跨 12 个 pair-direction 的 Pearson 相关性，不是本单对的参考标量。",
            "",
            "## CoQA 多轮 handoff",
            "",
            "| variant | depth | target F1 | mapped F1 | drift (pp) | retention |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in report["coqa"]["variants"]:
        for row in variant["by_handoff_depth"]:
            lines.append(
                f"| {variant['variant']} | {row['handoff_depth']} | "
                f"{row['target_official_word_f1_percent']:.4f}% | "
                f"{row['mapped_official_word_f1_percent']:.4f}% | "
                f"{row['drift_target_minus_mapped_percentage_points']:.4f} | "
                f"{row['mapped_over_target_retention'] * 100.0:.4f}% |"
            )
        lines.append(
            f"\n{variant['variant']} 在 depths 1/3/5/7/10 上的 drift 线性斜率："
            f"{variant['drift_linear_slope_pp_per_turn_over_depths_1_3_5_7_10']:.6f} pp/turn。"
        )
    lines.extend(
        [
            "",
            "论文约报告 large-to-small drift 随轮次增长 0.33 pp/turn、retuning 相对 production k 最多改变约 2.0 pp；由于精确 conversation IDs、prompt、history、decoder 与 scorer 未公开，本地曲线不作 bit-exact 声明。",
            "",
            "## Prefill latency",
            "",
            "| tokens | local 14B re-prefill ms | local mapper ms | local speedup | paper H100 re-prefill / mapper ms |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["latency"]["rows"]:
        anchor = row["paper_h100_reference"]
        anchor_text = (
            f"{anchor['target_reprefill_ms']:.1f} / {anchor['mapper_ms']:.1f}"
            if anchor
            else "—"
        )
        lines.append(
            f"| {row['prefix_length']} | {row['local_target_reprefill_median_ms']:.3f} | "
            f"{row['local_mapper_median_ms']:.3f} | "
            f"{row['local_speedup_reprefill_over_mapper']:.4f}× | {anchor_text} |"
        )
    lines.extend(
        [
            "",
            "计时均为 50 warmups + 30 repeats。source prefill、CPU staging 与 GPU restaging 均在 timed region 外；mapper 从已驻 GPU 的完整 32B KV 开始。",
            "",
            "## 完整性",
            "",
            f"上游 source inventory SHA256：`{report['source_inventory_sha256']}`。报告目录内所有 JSON/Markdown/CSV 由 `sha256_manifest.json` 覆盖，正式 receipt 另有独立 SHA256 sidecar。",
            "",
        ]
    )
    return "\n".join(lines)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _report_payloads(report: Mapping[str, Any]) -> dict[str, bytes]:
    payloads = {
        "report.json": _json_bytes(report),
        "report.md": _render_markdown(report).encode("utf-8"),
        **_render_csvs(report),
    }
    if set(payloads) != set(PAYLOAD_FILENAMES):
        raise AssertionError("report renderer did not produce the sealed file inventory")
    return payloads


def _sha256_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def _write_file_atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _build_report_directory(directory: Path, report: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    payloads = _report_payloads(report)
    for name, payload in payloads.items():
        _write_file_atomic(directory / name, payload)
    inventory = [
        {
            "path": name,
            "sha256": _sha256_bytes(payloads[name]),
            "nbytes": len(payloads[name]),
        }
        for name in PAYLOAD_FILENAMES
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_report_payload_manifest",
        "files": inventory,
        "files_sha256": canonical_sha256(inventory),
        "source_inventory_sha256": report["source_inventory_sha256"],
    }
    _write_file_atomic(directory / MANIFEST_FILENAME, _json_bytes(manifest))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_formal_report_receipt",
        "direction": DIRECTION,
        "report_kind": REPORT_KIND,
        "source_inventory_sha256": report["source_inventory_sha256"],
        "report_payload_sha256": canonical_sha256(report),
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": sha256_file(directory / MANIFEST_FILENAME),
            "nbytes": (directory / MANIFEST_FILENAME).stat().st_size,
        },
        "files_sha256": manifest["files_sha256"],
        "atomic_publication": "temporary_directory_then_single_rename",
    }
    receipt_payload = _json_bytes(receipt)
    _write_file_atomic(directory / RECEIPT_FILENAME, receipt_payload)
    receipt_sidecar = {
        "schema_version": SCHEMA_VERSION,
        "artifact": RECEIPT_FILENAME,
        "sha256": _sha256_bytes(receipt_payload),
        "nbytes": len(receipt_payload),
    }
    _write_file_atomic(
        directory / RECEIPT_SIDECAR_FILENAME, _json_bytes(receipt_sidecar)
    )
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_report_bundle(
    output: Path, expected_report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if not output.is_dir():
        raise FileNotFoundError(output)
    actual_names = {path.name for path in output.iterdir() if path.is_file()}
    if actual_names != set(COMPLETE_FILENAMES) or any(path.is_dir() for path in output.iterdir()):
        raise ReportValidationError(
            f"formal report file inventory changed: {sorted(actual_names)}"
        )
    payloads = _report_payloads(expected_report)
    for name, wanted in payloads.items():
        if (output / name).read_bytes() != wanted:
            raise ReportValidationError(f"formal report derived artifact changed: {name}")
    manifest = read_json(output / MANIFEST_FILENAME)
    inventory = [
        {
            "path": name,
            "sha256": sha256_file(output / name),
            "nbytes": (output / name).stat().st_size,
        }
        for name in PAYLOAD_FILENAMES
    ]
    expected_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_report_payload_manifest",
        "files": inventory,
        "files_sha256": canonical_sha256(inventory),
        "source_inventory_sha256": expected_report["source_inventory_sha256"],
    }
    if manifest != expected_manifest:
        raise ReportValidationError("formal report SHA256 manifest changed")
    verify_hashed_json(output / RECEIPT_FILENAME)
    receipt = read_json(output / RECEIPT_FILENAME)
    expected_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_formal_report_receipt",
        "direction": DIRECTION,
        "report_kind": REPORT_KIND,
        "source_inventory_sha256": expected_report["source_inventory_sha256"],
        "report_payload_sha256": canonical_sha256(expected_report),
        "manifest": {
            "path": MANIFEST_FILENAME,
            "sha256": sha256_file(output / MANIFEST_FILENAME),
            "nbytes": (output / MANIFEST_FILENAME).stat().st_size,
        },
        "files_sha256": expected_manifest["files_sha256"],
        "atomic_publication": "temporary_directory_then_single_rename",
    }
    if receipt != expected_receipt:
        raise ReportValidationError("formal report receipt changed")
    return [_artifact(output / name) for name in COMPLETE_FILENAMES]


def publish_report_bundle(
    output: Path, report: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Publish a previously validated report by atomic directory rename."""

    destination = output.resolve()
    if destination.exists():
        return _validate_report_bundle(destination, report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    # mkdtemp creates the directory; the builder owns a child so its
    # create-without-overwrite invariant stays explicit.
    staging = temporary / "payload"
    try:
        _build_report_directory(staging, report)
        os.replace(staging, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return _validate_report_bundle(destination, report)


def build_formal_report() -> list[dict[str, Any]]:
    report = collect_formal_results()
    return publish_report_bundle(REPORT_ROOT, report)


def validate_report_output(output: Path = REPORT_ROOT) -> list[dict[str, Any]]:
    """Re-run every upstream validator, then compare every report byte."""

    report = collect_formal_results()
    return _validate_report_bundle(output.resolve(), report)


def report_stage_command(python: str | Path) -> list[str]:
    return [
        os.path.abspath(os.fspath(python)),
        str((EXPERIMENT_ROOT / "scripts/build_report.py").resolve()),
    ]


def validate_report_stage() -> list[dict[str, Any]] | None:
    receipt = REPORT_ROOT / RECEIPT_FILENAME
    if not REPORT_ROOT.exists():
        return None
    if not receipt.is_file():
        raise ReportValidationError("formal report directory is only partially committed")
    return validate_report_output(REPORT_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv or ())
    if arguments not in ([], ["--validate-only"]):
        raise ReportValidationError("formal report accepts only --validate-only")
    artifacts = (
        validate_report_output(REPORT_ROOT)
        if arguments == ["--validate-only"]
        else build_formal_report()
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "report_root": str(REPORT_ROOT),
                "artifacts": artifacts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


__all__ = [
    "COMPLETE_FILENAMES",
    "PAYLOAD_FILENAMES",
    "REPORT_ROOT",
    "REPORT_PIPELINE_INTEGRATION",
    "REPORT_STAGE_NAME",
    "REQUIRED_STAGE_BINDINGS",
    "ReportValidationError",
    "build_formal_report",
    "collect_formal_results",
    "main",
    "publish_report_bundle",
    "report_stage_command",
    "validate_report_output",
    "validate_report_stage",
    "_coqa_report_from_summary",
    "_heldout_report_from_layers",
    "_validate_report_bundle",
    "_validate_stage_receipt_set",
]
