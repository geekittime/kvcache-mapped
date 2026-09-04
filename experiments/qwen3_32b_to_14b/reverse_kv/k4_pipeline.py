"""Stage commands and validators for appending k=4 to the formal pipeline.

The primary pipeline owns locking, idle sampling, process monitoring, state,
and stage receipts.  This module keeps all k=4-specific paths and scientific
validators independent so integration cannot accidentally reuse k=20 paths.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

from .coqa_k4_eval import validate_k4_receipt
from .eval_common import verify_hashed_json
from .fit_adapter import _validate_layer_payload
from .k4_heldout import K4_HELDOUT, validate_k4_heldout_artifact
from .k4_protocol import (
    K4_BIAS_COUNT,
    K4_MAPPER_ROOT,
    K4_SELECTION,
    K4_SOURCE_FEATURES,
    K4_TARGET_FEATURES,
    K4_WEIGHT_COUNT,
)
from .k4_runtime import K4ReverseMapperBundle
from .k4_selection import validate_k4_selection_artifact
from .protocol import EXPERIMENT_ROOT, canonical_sha256, read_json, sha256_file


PIPELINE_ROOT = EXPERIMENT_ROOT / "runs/formal_pipeline_v1"
K4_COQA_ROOT = PIPELINE_ROOT / "coqa_reverse_k4_retuned"
K4_FIT_SMOKE_EVIDENCE = PIPELINE_ROOT / "evidence/fit_k4_target0_gpu_monitor.json"
K4_TRAINING_IDLE_GATE = PIPELINE_ROOT / "gates/k4_training_segment_idle.json"
K4_COQA_IDLE_GATE = PIPELINE_ROOT / "gates/coqa_k4_segment_idle.json"

K4_STAGE_ORDER = (
    "11_k4_selection",
    "12_k4_fit_target0",
    "13_k4_fit_full",
    "14_k4_heldout",
    "15_k4_coqa",
)

K4_PIPELINE_INTEGRATION = {
    "append_before_pipeline_completion_receipt": True,
    "production_k20_stage_commands_changed": False,
    "completion_receipt_validation_must_rerun_k4_validators": True,
    "lock_scope": {
        "selection_fit_heldout": str(K4_TRAINING_IDLE_GATE),
        "before_coqa": "same_lock_after_distinct_12_sample_coqa_k4_idle_gate",
    },
    "fit_smoke_monitor": {
        "allowed_gpu": 0,
        "peak_memory_limit_mib": 18 * 1024,
        "distinct_evidence_path": str(K4_FIT_SMOKE_EVIDENCE),
        "must_not_reuse_k20_evidence": True,
    },
    "required_completion_artifacts": (
        str(K4_SELECTION),
        str(K4_MAPPER_ROOT / "manifest.json"),
        str(K4_HELDOUT),
        str(K4_COQA_ROOT / "formal_k4_coqa_receipt.json"),
        str(K4_TRAINING_IDLE_GATE),
        str(K4_COQA_IDLE_GATE),
    ),
}


def _artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "relative_path": str(resolved.relative_to(EXPERIMENT_ROOT.resolve())),
        "sha256": sha256_file(resolved),
        "nbytes": resolved.stat().st_size,
    }


def _hashed_sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256.json")


def _validate_idle_gate(path: Path, *, name: str) -> list[dict[str, Any]]:
    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "schema_version": 1,
        "status": "passed",
        "name": name,
        "required_consecutive_samples": 12,
        "memory_limit_mib": 64,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"k4 idle gate {name} field {field} changed")
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) < 12:
        raise ValueError(f"k4 idle gate {name} lacks 12 observations")
    for observation in observations[-12:]:
        if not isinstance(observation, Mapping) or observation.get("passed") is not True:
            raise ValueError(f"k4 idle gate {name} lacks 12 consecutive passed samples")
        memory = observation.get("memory")
        processes = observation.get("processes")
        if (
            not isinstance(memory, list)
            or len(memory) != 8
            or {row.get("index") for row in memory if isinstance(row, Mapping)}
            != set(range(8))
            or any(
                not isinstance(row, Mapping)
                or int(row.get("memory_used_mib", 65)) > 64
                for row in memory
            )
            or processes != []
        ):
            raise ValueError(f"k4 idle gate {name} contains a non-idle sample")
    return [_artifact(path), _artifact(_hashed_sidecar(path))]


def validate_k4_training_idle_gate() -> list[dict[str, Any]]:
    return _validate_idle_gate(
        K4_TRAINING_IDLE_GATE, name="k4_training_segment_idle"
    )


def validate_k4_coqa_idle_gate() -> list[dict[str, Any]]:
    return _validate_idle_gate(K4_COQA_IDLE_GATE, name="coqa_k4_segment_idle")


def k4_stage_commands(python: str | Path) -> dict[str, list[str]]:
    py = os.path.abspath(os.fspath(python))
    scripts = EXPERIMENT_ROOT / "scripts"
    return {
        "11_k4_selection": [py, str(scripts / "select_k4_coqa.py")],
        "12_k4_fit_target0": [
            py,
            str(scripts / "fit_mapper_k4_coqa.py"),
            "--devices",
            "0",
            "--target-layers",
            "0",
            "--chunk-tokens",
            "1024",
            "--solver",
            "cholesky",
            "--analysis-device",
            "cpu",
            "--cpu-threads-per-worker",
            "2",
        ],
        "13_k4_fit_full": [
            py,
            str(scripts / "fit_mapper_k4_coqa.py"),
            "--devices",
            "0,1,2,3,4,5,6,7",
            "--chunk-tokens",
            "1024",
            "--solver",
            "cholesky",
            "--analysis-device",
            "cpu",
            "--cpu-threads-per-worker",
            "2",
        ],
        "14_k4_heldout": [
            py,
            str(scripts / "validate_heldout_k4_coqa.py"),
            "--compute-device",
            "cuda:0",
            "--analysis-device",
            "cpu",
            "--chunk-samples",
            "1024",
        ],
        "15_k4_coqa": [
            py,
            str(scripts / "evaluate_coqa_k4.py"),
            "--output-dir",
            str(K4_COQA_ROOT),
        ],
    }


def validate_k4_selection_stage() -> list[dict[str, Any]] | None:
    if not K4_SELECTION.exists():
        return None
    validate_k4_selection_artifact(K4_SELECTION)
    sidecar = K4_SELECTION.with_suffix(".sha256.json")
    marker = read_json(sidecar)
    if marker != {"path": str(K4_SELECTION), "sha256": sha256_file(K4_SELECTION)}:
        raise ValueError("k4 selection checksum marker changed")
    return [_artifact(K4_SELECTION), _artifact(sidecar)]


def validate_k4_fit_smoke_stage() -> list[dict[str, Any]] | None:
    layer = K4_MAPPER_ROOT / "layers/layer_000.safetensors"
    if not layer.exists() and not K4_FIT_SMOKE_EVIDENCE.exists():
        return None
    if not layer.is_file() or not K4_FIT_SMOKE_EVIDENCE.is_file():
        raise ValueError("k4 fit smoke is only partially committed")
    verify_hashed_json(K4_FIT_SMOKE_EVIDENCE)
    monitor = read_json(K4_FIT_SMOKE_EVIDENCE)
    expected_monitor = {
        "schema_version": 1,
        "status": "passed",
        "stage": "12_k4_fit_target0",
        "single_compute_owner": True,
        "compute_owner_observed": True,
        "allowed_gpu": 0,
        "maximum_allowed_peak_mib": 18 * 1024,
    }
    for field, expected in expected_monitor.items():
        if monitor.get(field) != expected:
            raise ValueError(f"k4 fit smoke monitor {field} changed")
    if int(monitor.get("peak_memory_used_mib", 18 * 1024 + 1)) > 18 * 1024:
        raise ValueError("k4 fit smoke exceeded its GPU-memory ceiling")
    samples = monitor.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("k4 fit smoke has no physical GPU samples")
    owners: set[tuple[str, int]] = set()
    recomputed_peak = 0
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("k4 fit smoke contains a malformed sample")
        memory = sample.get("memory")
        processes = sample.get("processes")
        if (
            not isinstance(memory, list)
            or len(memory) != 8
            or {row.get("index") for row in memory if isinstance(row, Mapping)}
            != set(range(8))
            or not isinstance(processes, list)
            or len(processes) > 1
        ):
            raise ValueError("k4 fit smoke GPU inventory changed")
        if any(
            not isinstance(row, Mapping)
            or (row.get("index") != 0 and int(row.get("memory_used_mib", 65)) > 64)
            for row in memory
        ):
            raise ValueError("k4 fit smoke observed memory on GPUs 1-7")
        gpu0 = next(row for row in memory if row.get("index") == 0)
        recomputed_peak = max(recomputed_peak, int(gpu0.get("memory_used_mib", -1)))
        for process in processes:
            if not isinstance(process, Mapping) or process.get("index") != 0:
                raise ValueError("k4 fit smoke observed a non-GPU0 owner")
            identity = (str(process.get("uuid", "")), int(process.get("pid", -1)))
            if not identity[0] or identity[1] < 1:
                raise ValueError("k4 fit smoke owner identity is malformed")
            owners.add(identity)
    if (
        len(owners) != 1
        or monitor.get("compute_owner_identities")
        != [list(identity) for identity in sorted(owners)]
        or monitor.get("peak_memory_used_mib") != recomputed_peak
    ):
        raise ValueError("k4 fit smoke aggregate evidence changed")
    selection = validate_k4_selection_artifact(K4_SELECTION)
    sidecar = read_json(layer.with_suffix(".json"))
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("k4 fit smoke lacks layer metadata")
    _validate_layer_payload(
        K4_MAPPER_ROOT,
        target_layer=0,
        selected_layers=selection["selected_source_layers_rank_order"][0],
        source_features=K4_SOURCE_FEATURES,
        target_features=K4_TARGET_FEATURES,
        kv_heads=8,
        head_dim=128,
        fit_fingerprint=str(metadata.get("fit_fingerprint")),
        selection_sha256=sha256_file(K4_SELECTION),
    )
    artifacts = [
        _artifact(layer),
        _artifact(layer.with_suffix(".json")),
        _artifact(K4_FIT_SMOKE_EVIDENCE),
        _artifact(_hashed_sidecar(K4_FIT_SMOKE_EVIDENCE)),
    ]
    for cache_type in ("k_content", "v"):
        component = (
            K4_MAPPER_ROOT
            / "components"
            / cache_type
            / "layer_000.safetensors"
        )
        artifacts.extend((_artifact(component), _artifact(component.with_suffix(".json"))))
    return artifacts


def validate_k4_fit_full_stage() -> list[dict[str, Any]] | None:
    manifest = K4_MAPPER_ROOT / "manifest.json"
    if not manifest.exists():
        return None
    bundle = K4ReverseMapperBundle.open(K4_MAPPER_ROOT)
    value = read_json(manifest)
    if (
        len(bundle.layers) != 40
        or value.get("weight_count") != K4_WEIGHT_COUNT
        or value.get("bias_count") != K4_BIAS_COUNT
        or value.get("independent_refit_evidence", {}).get(
            "k20_weights_sliced_or_truncated"
        )
        is not False
    ):
        raise ValueError("complete k4 mapper aggregate or refit evidence changed")
    artifacts = [
        _artifact(manifest),
        _artifact(K4_MAPPER_ROOT / "manifest.sha256.json"),
    ]
    for layer in bundle.layers:
        artifacts.extend((_artifact(layer.path), _artifact(layer.sidecar_path)))
    return artifacts


def validate_k4_heldout_stage() -> list[dict[str, Any]] | None:
    if not K4_HELDOUT.exists():
        return None
    validate_k4_heldout_artifact(K4_HELDOUT)
    return [_artifact(K4_HELDOUT), _artifact(K4_HELDOUT.with_suffix(".sha256.json"))]


def validate_k4_coqa_stage() -> list[dict[str, Any]] | None:
    receipt = K4_COQA_ROOT / "formal_k4_coqa_receipt.json"
    if not receipt.exists():
        return None
    value = validate_k4_receipt(K4_COQA_ROOT)
    validation = value.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("formal k4 CoQA receipt validation identity changed")
    validate_k4_coqa_idle_gate()
    controls = [
        _artifact(receipt),
        _artifact(_hashed_sidecar(receipt)),
    ]
    for name in (
        "manifest.json",
        "summary.json",
        "run_definition.json",
        "dataset_receipt.json",
        "selection_manifest.json",
        "runtime_setup.json",
        "token_plans.jsonl",
        "raw_records.jsonl",
    ):
        path = K4_COQA_ROOT / name
        controls.extend((_artifact(path), _artifact(_hashed_sidecar(path))))
    return [
        *controls,
        _artifact(K4_HELDOUT),
        _artifact(K4_HELDOUT.with_suffix(".sha256.json")),
        _artifact(K4_MAPPER_ROOT / "manifest.json"),
        _artifact(K4_MAPPER_ROOT / "manifest.sha256.json"),
        _artifact(K4_COQA_IDLE_GATE),
        _artifact(_hashed_sidecar(K4_COQA_IDLE_GATE)),
    ]


def k4_validator_map() -> dict[str, Any]:
    return {
        "11_k4_selection": validate_k4_selection_stage,
        "12_k4_fit_target0": validate_k4_fit_smoke_stage,
        "13_k4_fit_full": validate_k4_fit_full_stage,
        "14_k4_heldout": validate_k4_heldout_stage,
        "15_k4_coqa": validate_k4_coqa_stage,
    }


__all__ = [
    "K4_COQA_IDLE_GATE",
    "K4_COQA_ROOT",
    "K4_FIT_SMOKE_EVIDENCE",
    "K4_PIPELINE_INTEGRATION",
    "K4_STAGE_ORDER",
    "K4_TRAINING_IDLE_GATE",
    "k4_stage_commands",
    "k4_validator_map",
    "validate_k4_coqa_stage",
    "validate_k4_coqa_idle_gate",
    "validate_k4_fit_full_stage",
    "validate_k4_fit_smoke_stage",
    "validate_k4_heldout_stage",
    "validate_k4_selection_stage",
    "validate_k4_training_idle_gate",
]
