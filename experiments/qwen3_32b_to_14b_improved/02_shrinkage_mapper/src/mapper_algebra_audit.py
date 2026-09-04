#!/usr/bin/env python3
"""CPU-only append-only, per-layer algebra audit of the published mapper.

This audit never parses locked-audit metrics.  It verifies their already
frozen file identity only, then proves that every published tensor is exactly
the FP32 k4/k20 function audited by that frozen artifact.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
import uuid

import torch
from safetensors.torch import load_file


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import shrinkage_mapper as sm  # noqa: E402


ROOT = HERE.parent
MAPPER_ROOT = ROOT / "artifacts" / "headwise_shrinkage_v1"
ALPHA_PATH = ROOT / "runs" / "activation_cv_v3" / "alpha_manifest.json"
LOCKED_AUDIT_PATH = ROOT / "runs" / "activation_cv_v3" / "locked_audit.json"
LOCKED_AUDIT_SHA256 = "99da4cb55e1c52a1fb15a454443ab26f79914f8b71cc3689fa5d9f18f75b4369"
AMENDMENT_V1_RECEIPT = (
    ROOT
    / "runs"
    / "activation_cv_v3_completion_amendment_v1"
    / "receipt_amended.json"
)
AMENDMENT_V1_RECEIPT_SHA256 = (
    "49450ace00fee697fd2ebd07c6b2def9efe9241721e7cebe82bea8518699dd14"
)
OUTPUT_ROOT = ROOT / "runs" / "activation_cv_v3_mapper_algebra_audit_v2"


def require_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    """Fail closed on a single changed bit and report a finite diagnostic."""
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError(
            f"{label}: dtype/shape mismatch {actual.dtype}/{tuple(actual.shape)} "
            f"!= {expected.dtype}/{tuple(expected.shape)}"
        )
    if torch.equal(actual, expected):
        return
    difference = torch.abs(actual.to(torch.float64) - expected.to(torch.float64))
    raise ValueError(
        f"{label}: tensors are not bitwise equal; "
        f"changed_elements={int(torch.count_nonzero(actual != expected))}, "
        f"max_abs={float(difference.max())}"
    )


def _assert_artifact(path: Path, entry: dict[str, Any]) -> None:
    if path.stat().st_size != entry.get("nbytes") or sm.sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"artifact identity mismatch: {path}")


def run_audit() -> dict[str, Any]:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite audit: {OUTPUT_ROOT}")
    if sm.sha256_file(LOCKED_AUDIT_PATH) != LOCKED_AUDIT_SHA256:
        raise ValueError("locked audit byte identity changed")
    if sm.sha256_file(AMENDMENT_V1_RECEIPT) != AMENDMENT_V1_RECEIPT_SHA256:
        raise ValueError("amendment-v1 receipt identity changed")
    mapper_manifest_path = MAPPER_ROOT / "manifest.json"
    mapper_manifest = sm.read_json(mapper_manifest_path)
    mapper_marker = sm.read_json(MAPPER_ROOT / "manifest.sha256.json")
    mapper_manifest_sha = sm.sha256_file(mapper_manifest_path)
    if mapper_marker != {"path": "manifest.json", "sha256": mapper_manifest_sha}:
        raise ValueError("candidate mapper manifest marker changed")
    upstream = mapper_manifest.get("upstream")
    if not isinstance(upstream, dict):
        raise ValueError("candidate upstream binding missing")
    k20_manifest_path = Path(upstream["k20_manifest"]["path"])
    k4_manifest_path = Path(upstream["k4_manifest"]["path"])
    _assert_artifact(k20_manifest_path, upstream["k20_manifest"])
    _assert_artifact(k4_manifest_path, upstream["k4_manifest"])
    k20_manifest = sm.read_json(k20_manifest_path)
    k4_manifest = sm.read_json(k4_manifest_path)
    k20_root = k20_manifest_path.parent
    k4_root = k4_manifest_path.parent
    alpha = sm.read_json(ALPHA_PATH)
    if alpha.get("locked_audit_used_for_alpha") is not False:
        raise ValueError("alpha indicates locked-audit leakage")
    if len(alpha.get("layers", [])) != 40:
        raise ValueError("alpha layer coverage changed")
    rows = []
    tensor_count = 0
    for target_layer in range(40):
        candidate_entry = mapper_manifest["layers"][target_layer]
        k20_entry = k20_manifest["layers"][target_layer]
        k4_entry = k4_manifest["layers"][target_layer]
        candidate_path = MAPPER_ROOT / candidate_entry["path"]
        candidate_sidecar_path = MAPPER_ROOT / candidate_entry["sidecar_path"]
        k20_path = k20_root / "layers" / f"layer_{target_layer:03d}.safetensors"
        k4_path = k4_root / "layers" / f"layer_{target_layer:03d}.safetensors"
        identities = {
            "candidate_payload_sha256": sm.sha256_file(candidate_path),
            "candidate_sidecar_sha256": sm.sha256_file(candidate_sidecar_path),
            "upstream_k20_payload_sha256": sm.sha256_file(k20_path),
            "upstream_k4_payload_sha256": sm.sha256_file(k4_path),
        }
        expected_identities = {
            "candidate_payload_sha256": candidate_entry["sha256"],
            "candidate_sidecar_sha256": candidate_entry["sidecar_sha256"],
            "upstream_k20_payload_sha256": k20_entry["sha256"],
            "upstream_k4_payload_sha256": k4_entry["sha256"],
        }
        if identities != expected_identities:
            raise ValueError(f"layer identity drift at {target_layer}")
        candidate_sidecar = sm.read_json(candidate_sidecar_path)
        metadata = candidate_sidecar.get("metadata", {})
        if (
            metadata.get("upstream_k20_layer_sha256")
            != identities["upstream_k20_payload_sha256"]
            or metadata.get("upstream_k4_layer_sha256")
            != identities["upstream_k4_payload_sha256"]
        ):
            raise ValueError(f"candidate sidecar upstream binding drift at {target_layer}")
        candidate = load_file(str(candidate_path), device="cpu")
        k20 = load_file(str(k20_path), device="cpu")
        k4 = load_file(str(k4_path), device="cpu")
        component_rows = {}
        for component in ("key", "value"):
            head_alpha = alpha["layers"][target_layer][component]["alpha"]
            expected_weight = sm._fuse_component(
                k20[f"{component}_weight"], k4[f"{component}_weight"], head_alpha
            )
            expected_bias = sm._fuse_bias(
                k20[f"{component}_bias"], k4[f"{component}_bias"], head_alpha
            )
            require_bitwise_equal(
                candidate[f"{component}_weight"],
                expected_weight,
                f"layer {target_layer} {component}_weight",
            )
            require_bitwise_equal(
                candidate[f"{component}_bias"],
                expected_bias,
                f"layer {target_layer} {component}_bias",
            )
            tensor_count += 2
            component_rows[component] = {
                "alpha": head_alpha,
                "weight_bitwise_equal": True,
                "bias_bitwise_equal": True,
                "weight_shape": list(expected_weight.shape),
                "bias_shape": list(expected_bias.shape),
                "dtype": str(expected_weight.dtype).removeprefix("torch."),
            }
        rows.append(
            {
                "target_layer": target_layer,
                "identities": identities,
                "sidecar_upstream_binding": "passed",
                "components": component_rows,
            }
        )
        del candidate, k20, k4, expected_weight, expected_bias
    if tensor_count != 160:
        raise AssertionError("did not check exactly 160 tensors")
    validation = {
        "schema_version": 1,
        "status": "passed",
        "completed_utc": sm.utc_now(),
        "execution_device": "cpu-only",
        "cuda_invoked": False,
        "activation_recomputed": False,
        "locked_audit_content_parsed": False,
        "locked_audit_recomputed": False,
        "locked_audit_sha256": LOCKED_AUDIT_SHA256,
        "alpha_manifest": sm.artifact(ALPHA_PATH),
        "candidate_mapper_manifest": sm.artifact(mapper_manifest_path),
        "upstream_k20_manifest": sm.artifact(k20_manifest_path),
        "upstream_k4_manifest": sm.artifact(k4_manifest_path),
        "layers_checked": 40,
        "tensors_checked": tensor_count,
        "all_tensor_comparisons": "torch.equal",
        "all_tensors_bitwise_equal": True,
        "layer_rows": rows,
        "layer_rows_digest": sm.canonical_sha256(rows),
    }
    staging = OUTPUT_ROOT.with_name(f".{OUTPUT_ROOT.name}.staging.{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        sm.atomic_write_json(staging / "algebra_validation.json", validation)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "completed_utc": sm.utc_now(),
            "scope": "append-only CPU algebra audit; no activation or locked-audit recomputation",
            "supersedes_no_scientific_artifact": True,
            "amendment_v1_receipt": {
                "path": str(AMENDMENT_V1_RECEIPT),
                "sha256": AMENDMENT_V1_RECEIPT_SHA256,
            },
            "unique_addition": "explicit per-layer upstream SHA binding and torch.equal proof for all 160 published tensors",
            "locked_audit_content_parsed": False,
            "locked_audit_recomputed": False,
            "activation_recomputed": False,
            "cuda_invoked": False,
            "algebra_validation": sm.artifact(
                staging / "algebra_validation.json", relative_to=staging
            ),
            "code": sm.code_inventory(),
        }
        sm.atomic_write_json(staging / "receipt.json", receipt)
        manifest = {
            "schema_version": 1,
            "status": "complete",
            "files": [
                sm.artifact(staging / "algebra_validation.json", relative_to=staging),
                sm.artifact(staging / "receipt.json", relative_to=staging),
            ],
        }
        sm.atomic_write_json(staging / "manifest.json", manifest)
        sm.atomic_write_json(
            staging / "manifest.sha256.json",
            {"path": "manifest.json", "sha256": sm.sha256_file(staging / "manifest.json")},
        )
        os.replace(staging, OUTPUT_ROOT)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "root": str(OUTPUT_ROOT),
        "receipt_sha256": sm.sha256_file(OUTPUT_ROOT / "receipt.json"),
        "validation_sha256": sm.sha256_file(OUTPUT_ROOT / "algebra_validation.json"),
        "layers_checked": 40,
        "tensors_checked": 160,
    }


def main() -> int:
    print(json.dumps(run_audit(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

