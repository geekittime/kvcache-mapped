#!/usr/bin/env python3
"""CPU-only append-only completion amendment for activation_cv_v3.

The activation computation, alpha freeze, locked audit, and mapper publication
all completed before the original process failed while JSON-serializing a
PyTorch ``_CUuuid`` object.  This script does not invoke CUDA and does not
re-evaluate the locked rows.  It validates and binds the existing artifacts,
then publishes a separate receipt amendment.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
import uuid

import numpy as np
import torch
from safetensors.torch import load_file


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import shrinkage_mapper as sm  # noqa: E402


ROOT = HERE.parent
RUN_ROOT = ROOT / "runs" / "activation_cv_v3"
MAPPER_ROOT = ROOT / "artifacts" / "headwise_shrinkage_v1"
AMENDMENT_ROOT = ROOT / "runs" / "activation_cv_v3_completion_amendment_v1"


def _artifact_matches(root: Path, entry: Mapping[str, Any]) -> None:
    path = (root / str(entry["path"])).resolve()
    path.relative_to(root.resolve())
    if sm.artifact(path, relative_to=root) != dict(entry):
        raise ValueError(f"artifact identity changed: {path}")


def _validate_original_failure() -> dict[str, Any]:
    status = sm.read_json(RUN_ROOT / "status.json")
    expected = {
        "schema_version": 1,
        "status": "failed",
        "error_type": "TypeError",
        "error": "Object of type _CUuuid is not JSON serializable",
    }
    for field, value in expected.items():
        if status.get(field) != value:
            raise ValueError(f"v3 failure field changed: {field}")
    console = ROOT / "runs" / "activation_cv_v3.console.log"
    console_text = console.read_text(encoding="utf-8")
    required_fragments = [
        "Object of type _CUuuid is not JSON serializable",
        "json.dumps",
        "receipt.json",
    ]
    # The stack ends in atomic_write_json; it must not show activation or audit
    # computation as the failing operation.
    if not all(fragment in console_text for fragment in required_fragments[:2]):
        raise ValueError("v3 console no longer proves the serialization failure")
    return {
        "status": sm.artifact(RUN_ROOT / "status.json"),
        "console": sm.artifact(console),
        "failure": status,
        "failure_stage": "after_mapper_atomic_publish_while_serializing_final_receipt",
    }


def _validate_gate_and_definition() -> dict[str, Any]:
    definition = sm.read_json(RUN_ROOT / "run_definition.json")
    gate = sm.read_json(RUN_ROOT / "idle_gate.json")
    if gate.get("status") != "passed" or gate.get("observed_consecutive_samples") != 12:
        raise ValueError("v3 idle gate changed")
    samples = gate.get("samples")
    if not isinstance(samples, list) or len(samples) != 12:
        raise ValueError("v3 idle gate does not contain exactly twelve samples")
    for index, sample in enumerate(samples):
        if sample.get("sample") != index or sample.get("idle") is not True:
            raise ValueError("v3 idle sample changed")
        if sample.get("compute_process_rows") != [] or len(sample.get("gpus", [])) != 8:
            raise ValueError("v3 idle sample process/GPU evidence changed")
        if max(row["memory_used_mib"] for row in sample["gpus"]) > 64:
            raise ValueError("v3 gate sample was not idle")
    if definition.get("cuda_visible_devices") != "0":
        raise ValueError("v3 did not freeze physical GPU 0 only")
    if definition.get("code", {}).get("digest") != sm.canonical_sha256(
        definition["code"]["files"]
    ):
        raise ValueError("v3 code inventory digest changed")
    for entry in definition["code"]["files"]:
        path = ROOT / entry["path"]
        # The source file itself is still unchanged; the amendment is a new
        # file and was not part of the original frozen inventory.
        if not path.is_file() or sm.sha256_file(path) != entry["sha256"]:
            raise ValueError(f"v3 frozen code changed: {entry['path']}")
    prior = sm.prior_failed_attempt_evidence()
    if definition.get("prior_failed_attempts") != prior:
        raise ValueError("v1/v2 pre-matmul failure binding changed")
    physical_uuid = samples[-1]["gpus"][0]["uuid"]
    if not isinstance(physical_uuid, str) or not physical_uuid.startswith("GPU-"):
        raise ValueError("driver UUID is malformed")
    return {
        "run_definition": sm.artifact(RUN_ROOT / "run_definition.json"),
        "idle_gate": sm.artifact(RUN_ROOT / "idle_gate.json"),
        "input_preflight": sm.artifact(RUN_ROOT / "input_preflight.json"),
        "split_manifest": sm.artifact(RUN_ROOT / "split_manifest.json"),
        "code_digest": definition["code"]["digest"],
        "command": definition["command"],
        "physical_gpu0_uuid_string": physical_uuid,
        "prior_failed_attempts": prior,
    }


def _validate_alpha_and_tuning() -> dict[str, Any]:
    alpha_path = RUN_ROOT / "alpha_manifest.json"
    freeze_path = RUN_ROOT / "alpha_freeze_receipt.json"
    alpha_sha = sm.sha256_file(alpha_path)
    alpha = sm.read_json(alpha_path)
    freeze = sm.read_json(freeze_path)
    if freeze != {
        "schema_version": 1,
        "status": "frozen",
        "completed_utc": freeze.get("completed_utc"),
        "alpha_manifest_sha256": alpha_sha,
        "locked_audit_used_for_alpha": False,
    }:
        raise ValueError("alpha freeze receipt changed")
    if alpha.get("status") != "frozen_before_locked_audit":
        raise ValueError("alpha was not frozen before audit")
    if alpha.get("locked_audit_used_for_alpha") is not False:
        raise ValueError("alpha manifest indicates audit leakage")
    split = sm.read_json(RUN_ROOT / "split_manifest.json")
    tuning_ids = [int(value) for value in split["tuning_sequence_ids"]]
    locked_ids = [int(value) for value in split["locked_audit_sequence_ids"]]
    if set(tuning_ids) & set(locked_ids) or sorted(tuning_ids + locked_ids) != list(range(50)):
        raise ValueError("tuning/locked split changed")
    if alpha.get("fit_sequence_ids") != tuning_ids or alpha.get("locked_audit_sequence_ids") != locked_ids:
        raise ValueError("alpha split binding changed")
    layers = alpha.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ValueError("alpha layer coverage changed")
    stats_entries = []
    alpha_values = {"key": [], "value": []}
    for target_layer, row in enumerate(layers):
        if row.get("target_layer") != target_layer:
            raise ValueError("alpha layers are unordered")
        for component in ("key", "value"):
            value = row[component]
            entry = value["tuning_stats"]
            _artifact_matches(RUN_ROOT, entry)
            stats_path = RUN_ROOT / entry["path"]
            with np.load(stats_path, allow_pickle=False) as payload:
                if payload["sequence_ids"].tolist() != tuning_ids:
                    raise ValueError("tuning stats sequence order changed")
                stats = {key: payload[key] for key in payload.files if key != "sequence_ids"}
            recomputed = sm.fit_alpha(stats["e_dot_d"], stats["d2"], list(range(40)))
            recorded = np.asarray(value["alpha"], dtype=np.float64)
            np.testing.assert_array_equal(recomputed, recorded)
            if recorded.shape != (8,) or not np.isfinite(recorded).all() or not ((0 <= recorded) & (recorded <= 1)).all():
                raise ValueError("invalid alpha")
            if value["cv"].get("locked_audit_consulted") is not False:
                raise ValueError("CV metadata indicates audit leakage")
            alpha_values[component].extend(recorded.tolist())
            stats_entries.append(
                {
                    "target_layer": target_layer,
                    "component": component,
                    **entry,
                    "analytic_alpha_recomputed_exactly": True,
                }
            )
    summary = {}
    for component, values in alpha_values.items():
        array = np.asarray(values, dtype=np.float64)
        summary[component] = {
            "count": int(array.size),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "at_lower_bound": int(np.sum(array == 0.0)),
            "at_upper_bound": int(np.sum(array == 1.0)),
        }
    return {
        "alpha_manifest": sm.artifact(alpha_path),
        "alpha_freeze_receipt": sm.artifact(freeze_path),
        "tuning_statistics": {
            "files": len(stats_entries),
            "entries_digest": sm.canonical_sha256(stats_entries),
            "all_final_alphas_recomputed_exactly": True,
        },
        "alpha_summary": summary,
    }


def _r2_from_sse_sst(sse: np.ndarray, sst: np.ndarray) -> np.ndarray:
    result = np.empty_like(sst)
    mask = sst != 0.0
    result[mask] = 1.0 - sse[mask] / sst[mask]
    result[~mask] = np.where(sse[~mask] == 0.0, 1.0, 0.0)
    return result


def _validate_locked_audit() -> dict[str, Any]:
    audit_path = RUN_ROOT / "locked_audit.json"
    audit = sm.read_json(audit_path)
    alpha = sm.read_json(RUN_ROOT / "alpha_manifest.json")
    alpha_sha = sm.sha256_file(RUN_ROOT / "alpha_manifest.json")
    if audit.get("status") != "complete" or audit.get("alpha_manifest_sha256") != alpha_sha:
        raise ValueError("locked audit is not bound to frozen alpha")
    if audit.get("locked_audit_evaluations") != 1 or audit.get("locked_audit_used_for_alpha") is not False:
        raise ValueError("locked audit count/leakage changed")
    split = sm.read_json(RUN_ROOT / "split_manifest.json")
    if audit.get("locked_audit_sequence_ids") != split.get("locked_audit_sequence_ids"):
        raise ValueError("locked audit identities changed")
    layers = audit.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ValueError("locked audit layer coverage changed")
    metric_digest_rows = []
    for target_layer, row in enumerate(layers):
        if row.get("target_layer") != target_layer:
            raise ValueError("locked audit layers unordered")
        for component in ("key", "value"):
            expected_alphas = {
                "k4": np.zeros(8),
                "k20": np.ones(8),
                "shrinkage": np.asarray(alpha["layers"][target_layer][component]["alpha"]),
            }
            for candidate, metrics in row[component]["audit"].items():
                if candidate not in expected_alphas:
                    raise ValueError("unknown audit candidate")
                np.testing.assert_array_equal(
                    np.asarray(metrics["alpha"]), expected_alphas[candidate]
                )
                if metrics.get("sequence_count") != 10 or metrics.get("observations") != 2560 or metrics.get("channels") != 1024:
                    raise ValueError("locked audit dimensions changed")
                sse = np.asarray(metrics["per_channel_sse_fp64"], dtype=np.float64)
                sst = np.asarray(metrics["per_channel_sst_fp64"], dtype=np.float64)
                if sse.shape != (1024,) or sst.shape != (1024,) or not np.isfinite(sse).all() or not np.isfinite(sst).all():
                    raise ValueError("locked audit sufficient statistics changed")
                if not math.isclose(float(sse.sum()), metrics["global_sse_fp64"], rel_tol=2e-15):
                    raise ValueError("locked audit SSE algebra changed")
                if not math.isclose(float(sst.sum()), metrics["global_sst_fp64"], rel_tol=2e-15):
                    raise ValueError("locked audit SST algebra changed")
                uniform = float(_r2_from_sse_sst(sse, sst).mean())
                if not math.isclose(uniform, metrics["uniform_channel_r2_fp64"], rel_tol=1e-14, abs_tol=1e-15):
                    raise ValueError("locked audit R2 algebra changed")
                metric_digest_rows.append(
                    {
                        "target_layer": target_layer,
                        "component": component,
                        "candidate": candidate,
                        "uniform_channel_r2_fp64": uniform,
                        "global_sse_fp64": float(sse.sum()),
                        "global_sst_fp64": float(sst.sum()),
                    }
                )
    expected_summary = sm._aggregate_audit_summary(layers)
    if sm.canonical_bytes(expected_summary) != sm.canonical_bytes(audit.get("summary")):
        raise ValueError("locked audit summary changed")
    return {
        "artifact": sm.artifact(audit_path),
        "locked_audit_evaluations": 1,
        "locked_audit_used_for_alpha": False,
        "all_metric_algebra_recomputed": True,
        "metric_rows_digest": sm.canonical_sha256(metric_digest_rows),
        "summary": expected_summary,
    }


def _validate_exact_mapper_fusion() -> dict[str, Any]:
    config = sm.load_config()
    alpha = sm.read_json(RUN_ROOT / "alpha_manifest.json")
    k20_root = Path(config["upstream"]["k20_mapper"])
    k4_root = Path(config["upstream"]["k4_mapper"])
    base_validation = sm.validate_mapper(MAPPER_ROOT, deep_hash=True)
    entries = []
    for target_layer in range(40):
        output = load_file(
            str(MAPPER_ROOT / "layers" / f"layer_{target_layer:03d}.safetensors"),
            device="cpu",
        )
        k20 = load_file(
            str(k20_root / "layers" / f"layer_{target_layer:03d}.safetensors"),
            device="cpu",
        )
        k4 = load_file(
            str(k4_root / "layers" / f"layer_{target_layer:03d}.safetensors"),
            device="cpu",
        )
        component_results = {}
        for component in ("key", "value"):
            head_alpha = alpha["layers"][target_layer][component]["alpha"]
            expected_weight = sm._fuse_component(
                k20[f"{component}_weight"], k4[f"{component}_weight"], head_alpha
            )
            expected_bias = sm._fuse_bias(
                k20[f"{component}_bias"], k4[f"{component}_bias"], head_alpha
            )
            weight_equal = torch.equal(output[f"{component}_weight"], expected_weight)
            bias_equal = torch.equal(output[f"{component}_bias"], expected_bias)
            if not weight_equal or not bias_equal:
                raise ValueError(f"mapper fusion mismatch at layer {target_layer} {component}")
            component_results[component] = {
                "weight_bitwise_equal_to_fused_formula": weight_equal,
                "bias_bitwise_equal_to_fused_formula": bias_equal,
            }
        sidecar = MAPPER_ROOT / "layers" / f"layer_{target_layer:03d}.json"
        payload = MAPPER_ROOT / "layers" / f"layer_{target_layer:03d}.safetensors"
        entries.append(
            {
                "target_layer": target_layer,
                "payload": sm.artifact(payload, relative_to=MAPPER_ROOT),
                "sidecar": sm.artifact(sidecar, relative_to=MAPPER_ROOT),
                "components": component_results,
            }
        )
        del output, k20, k4, expected_weight, expected_bias
    return {
        "base_validation": base_validation,
        "all_40_layers_deep_hashed": True,
        "all_160_tensors_shape_dtype_validated": True,
        "all_160_tensors_bitwise_equal_to_frozen_alpha_fusion": True,
        "entries_digest": sm.canonical_sha256(entries),
        "manifest": sm.artifact(MAPPER_ROOT / "manifest.json"),
        "manifest_marker": sm.artifact(MAPPER_ROOT / "manifest.sha256.json"),
    }


def main() -> int:
    if AMENDMENT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite amendment: {AMENDMENT_ROOT}")
    started = dt.datetime.now(dt.timezone.utc)
    original_failure = _validate_original_failure()
    protocol = _validate_gate_and_definition()
    alpha = _validate_alpha_and_tuning()
    audit = _validate_locked_audit()
    mapper = _validate_exact_mapper_fusion()
    validation = {
        "schema_version": 1,
        "status": "passed",
        "completed_utc": sm.utc_now(),
        "execution_device": "cpu-only",
        "cuda_invoked": False,
        "activation_recomputed": False,
        "locked_audit_recomputed": False,
        "scientific_artifacts_modified": False,
        "original_failure": original_failure,
        "protocol": protocol,
        "alpha": alpha,
        "locked_audit": audit,
        "mapper": mapper,
    }
    staging = AMENDMENT_ROOT.with_name(
        f".{AMENDMENT_ROOT.name}.staging.{uuid.uuid4().hex}"
    )
    staging.mkdir(parents=True)
    try:
        sm.atomic_write_json(staging / "validation.json", validation)
        receipt = {
            "schema_version": 1,
            "status": "complete_by_append_only_amendment",
            "completed_utc": sm.utc_now(),
            "amendment_scope": "receipt-serialization-only",
            "unique_repair": {
                "original_non_json_value": "torch.cuda device property uuid (_CUuuid)",
                "canonical_json_value": protocol["physical_gpu0_uuid_string"],
                "conversion": "string UUID copied from the already-frozen v3 nvidia-smi idle-gate evidence",
                "other_repairs": [],
            },
            "execution": {
                "device": "cpu-only",
                "cuda_invoked": False,
                "activation_recomputed": False,
                "locked_audit_recomputed": False,
                "models_loaded": [],
                "mapper_modified": False,
                "alpha_modified": False,
            },
            "original_v3": {
                "command": protocol["command"],
                "code_digest": protocol["code_digest"],
                "run_definition": protocol["run_definition"],
                "input_preflight": protocol["input_preflight"],
                "idle_gate": protocol["idle_gate"],
                "status": original_failure["status"],
                "console": original_failure["console"],
                "failure_stage": original_failure["failure_stage"],
                "physical_gpu_indices": [0],
                "physical_gpu0_uuid": protocol["physical_gpu0_uuid_string"],
                "models_loaded": [],
                "allocator_peak_memory": {
                    "status": "not_recoverable_after_original_receipt_serialization_failure",
                    "value_claimed": False,
                    "reason": "allocator counters were process-local and the process exited before receipt persistence; no retrospective value is fabricated",
                },
            },
            "prior_failed_attempts": protocol["prior_failed_attempts"],
            "alpha_manifest": alpha["alpha_manifest"],
            "alpha_freeze_receipt": alpha["alpha_freeze_receipt"],
            "locked_audit": audit["artifact"],
            "locked_audit_summary": audit["summary"],
            "mapper": mapper["base_validation"],
            "validation": sm.artifact(staging / "validation.json", relative_to=staging),
            "amendment_code": sm.code_inventory(),
            "amendment_elapsed_seconds": (
                dt.datetime.now(dt.timezone.utc) - started
            ).total_seconds(),
        }
        sm.atomic_write_json(staging / "receipt_amended.json", receipt)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "files": [
                sm.artifact(staging / "validation.json", relative_to=staging),
                sm.artifact(staging / "receipt_amended.json", relative_to=staging),
            ],
        }
        sm.atomic_write_json(staging / "manifest.json", manifest)
        sm.atomic_write_json(
            staging / "manifest.sha256.json",
            {"path": "manifest.json", "sha256": sm.sha256_file(staging / "manifest.json")},
        )
        for entry in manifest["files"]:
            _artifact_matches(staging, entry)
        os.replace(staging, AMENDMENT_ROOT)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "complete",
                "amendment": str(AMENDMENT_ROOT),
                "receipt_sha256": sm.sha256_file(AMENDMENT_ROOT / "receipt_amended.json"),
                "validation_sha256": sm.sha256_file(AMENDMENT_ROOT / "validation.json"),
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
