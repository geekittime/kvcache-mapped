"""Independent validation of the repaired quota/allocation payloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .allocation import validate_allocation_report, validate_selection
from .common import ProtocolError, file_spec, json_exact, strict_json_load, strict_json_loads, validate_publication
from .inputs import InputBundle
from .quota import Target, targets_from_contract, validate_quota_proof, validate_quota_vector


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise ProtocolError("deep allocation JSONL row is not an object")
            rows.append(value)
    return rows


def validate_ready_stages(bundle: InputBundle, config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Revalidate A--D outputs without opening any answer/outcome artifact."""

    quota_dir = root / config["outputs"]["quota_search"]
    allocation_dir = root / config["outputs"]["allocation"]
    if not quota_dir.is_dir() or not allocation_dir.is_dir():
        raise ProtocolError("quota/allocation stage directories are missing")
    validate_publication(
        quota_dir,
        ["quota.json", "quota_report.json", "receipt.json"],
        identity={"protocol_id": "quota_capacity_repair_v1", "artifact_kind": "quota_search"},
        status="published_complete_quota_search",
    )
    validate_publication(
        allocation_dir,
        ["allocation.jsonl", "allocation_report.json", "receipt.json"],
        identity={"protocol_id": "quota_capacity_repair_v1", "artifact_kind": "repaired_allocation"},
        status="published_complete_repaired_allocation",
    )
    quota_vector = strict_json_load(quota_dir / "quota.json")
    quota_report = strict_json_load(quota_dir / "quota_report.json")
    allocation_report = strict_json_load(allocation_dir / "allocation_report.json")
    if not isinstance(quota_vector, dict) or not isinstance(quota_report, dict) or not isinstance(allocation_report, dict):
        raise ProtocolError("deep stage payload is not an object")
    targets = targets_from_contract(bundle.contract, config["upper_quotas"])
    quota_check = validate_quota_vector(quota_vector, targets)
    proof = quota_report.get("proof")
    if not isinstance(proof, dict) or proof.get("status") != "quota_search_proven_optimal":
        raise ProtocolError("deep quota proof is not proven optimal")
    validate_quota_proof(proof, targets)
    if quota_report.get("replica_quota_bytes_identical") is not True or not isinstance(quota_report.get("replicas"), list) or len(quota_report["replicas"]) != 2:
        raise ProtocolError("deep quota replica identity is missing")
    rows = _load_jsonl(allocation_dir / "allocation.jsonl")
    allocation_check = validate_selection(
        rows,
        bundle.records,
        bundle.hyperedges,
        targets,
        quota_vector,
        config["solver"]["hash_cost_namespace"],
    )
    if allocation_report.get("selection") != file_spec(allocation_dir / "allocation.jsonl"):
        raise ProtocolError("deep allocation report selection binding mismatch")
    for replica in allocation_report.get("replicas", []):
        if not isinstance(replica, dict) or not isinstance(replica.get("worker"), dict):
            raise ProtocolError("deep allocation replica worker binding is missing")
        validate_allocation_report(
            replica["worker"].get("solver"),
            config,
            expected_seed=replica.get("seed"),
        )
    if allocation_report.get("replica_selection_byte_identical") is not True:
        raise ProtocolError("deep allocation replica identity missing")
    return {
        "status": "pass_repaired_quota_and_allocation",
        "quota_check": quota_check,
        "allocation_check": allocation_check,
        "quota_search_PUBLISHED": file_spec(quota_dir / "PUBLISHED.json"),
        "allocation_PUBLISHED": file_spec(allocation_dir / "PUBLISHED.json"),
        "source_allocation_v3_deep_receipt": bundle.allocation_deep_binding,
        "process_id": os.getpid(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "cpu_only": True,
        "answer_values_read": False,
        "model_outcomes_read": False,
    }
