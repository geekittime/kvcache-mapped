"""Independent validation of the repaired quota/allocation payloads."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .allocation import validate_allocation_report, verify_selection
from .common import (
    ProtocolError,
    file_spec,
    formal_config_path,
    json_exact,
    strict_json_load,
    strict_json_loads,
    validate_publication,
    verify_config_file_binding,
)
from .inputs import InputBundle
from .quota import Target, targets_from_contract, validate_quota_proof, validate_quota_vector


def _validate_child_runtime_identity(identity: Any, config: Mapping[str, Any], *, name: str) -> None:
    """Check a persisted worker envelope against the pinned venv."""

    expected_keys = {
        "python_executable",
        "python_prefix",
        "python_real_executable",
        "python_real_executable_sha256",
        "python_major_minor",
        "python_version",
    }
    if not isinstance(identity, Mapping) or set(identity) != expected_keys or any(not isinstance(value, str) or not value for value in identity.values()):
        raise ProtocolError(f"{name} runtime identity schema mismatch")
    binding = config.get("runtime_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("runtime binding is missing during deep validation")
    for key in ("python_executable", "python_prefix", "python_real_executable", "python_real_executable_sha256", "python_major_minor"):
        if identity.get(key) != binding.get(key):
            raise ProtocolError(f"{name} runtime identity does not match pinned venv")


def _validate_child_config_binding(binding: Any, expected: Mapping[str, Any], *, name: str) -> None:
    if not json_exact(binding, dict(expected)):
        raise ProtocolError(f"{name} formal config binding mismatch")


def _strict_pid(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ProtocolError(f"{name} must be a positive strict integer PID")
    return value


def _validate_quota_replicas(
    replicas: Any,
    config: Mapping[str, Any],
    config_binding: Mapping[str, Any],
) -> None:
    """Require the complete, ordered two-replica quota identity."""

    if type(replicas) is not list or len(replicas) != 2:
        raise ProtocolError("deep quota replicas must be an exact list of two")
    expected_seeds = [1, 777]
    pids: set[int] = set()
    for index, (replica, expected_seed) in enumerate(zip(replicas, expected_seeds, strict=True)):
        if not isinstance(replica, Mapping) or set(replica) != {
            "seed", "python_hash_seed", "process_id", "result", "config_binding", "runtime_identity", "preformal_stop_binding"
        }:
            raise ProtocolError(f"deep quota replica {index} schema mismatch")
        if type(replica.get("seed")) is not int or replica["seed"] != expected_seed or replica.get("python_hash_seed") != str(expected_seed):
            raise ProtocolError(f"deep quota replica {index} seed mismatch")
        pid = _strict_pid(replica.get("process_id"), name=f"deep quota replica {index} PID")
        if pid in pids:
            raise ProtocolError("deep quota replica PIDs must be distinct")
        pids.add(pid)
        if not isinstance(replica.get("result"), Mapping) or set(replica["result"]) != {"quota_vector", "proof"}:
            raise ProtocolError(f"deep quota replica {index} result schema mismatch")
        _validate_child_config_binding(replica.get("config_binding"), config_binding, name=f"quota replica {index}")
        if not json_exact(replica.get("preformal_stop_binding"), dict(config["preformal_stop_binding"])):
            raise ProtocolError(f"quota replica {index} PRE-FORMAL binding mismatch")
        _validate_child_runtime_identity(replica.get("runtime_identity"), config, name=f"quota replica {index}")


def _validate_allocation_replicas(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    config_binding: Mapping[str, Any],
    selection_spec: Mapping[str, Any],
) -> None:
    """Validate both persisted allocation workers, including identity."""

    replicas = report.get("replicas")
    if type(replicas) is not list or len(replicas) != 2:
        raise ProtocolError("deep allocation replicas must be an exact list of two")
    pids: set[int] = set()
    seen_seeds: list[int] = []
    expected_seeds = [1, 777]
    for index, replica in enumerate(replicas):
        if not isinstance(replica, Mapping) or set(replica) != {
            "seed", "python_hash_seed", "process_id", "worker", "config_binding", "preformal_stop_binding"
        }:
            raise ProtocolError(f"deep allocation replica {index} schema mismatch")
        seed = replica.get("seed")
        if type(seed) is not int or seed not in expected_seeds or seed in seen_seeds:
            raise ProtocolError("deep allocation replica seeds must be exactly {1,777} once each")
        seen_seeds.append(seed)
        if replica.get("python_hash_seed") != str(seed):
            raise ProtocolError(f"deep allocation replica {seed} hash seed mismatch")
        pid = _strict_pid(replica.get("process_id"), name=f"deep allocation replica {seed} PID")
        if pid in pids:
            raise ProtocolError("deep allocation replica PIDs must be distinct")
        pids.add(pid)
        worker = replica.get("worker")
        if not isinstance(worker, Mapping) or set(worker) != {
            "process_id", "python_hash_seed", "seed", "selection", "solver", "config_binding", "runtime_identity", "preformal_stop_binding"
        }:
            raise ProtocolError(f"deep allocation worker {seed} schema mismatch")
        if worker.get("process_id") != pid or worker.get("python_hash_seed") != str(seed) or type(worker.get("seed")) is not int or worker["seed"] != seed:
            raise ProtocolError(f"deep allocation worker {seed} identity mismatch")
        _validate_child_config_binding(replica.get("config_binding"), config_binding, name=f"allocation replica {seed}")
        if not json_exact(replica.get("preformal_stop_binding"), dict(config["preformal_stop_binding"])):
            raise ProtocolError(f"allocation replica {seed} PRE-FORMAL binding mismatch")
        _validate_child_config_binding(worker.get("config_binding"), config_binding, name=f"allocation worker {seed}")
        if not json_exact(worker.get("preformal_stop_binding"), dict(config["preformal_stop_binding"])):
            raise ProtocolError(f"allocation worker {seed} PRE-FORMAL binding mismatch")
        _validate_child_runtime_identity(worker.get("runtime_identity"), config, name=f"allocation replica {seed}")
        worker_selection = worker.get("selection")
        if not isinstance(worker_selection, Mapping) or set(worker_selection) != {"nbytes", "sha256"}:
            raise ProtocolError(f"allocation worker {seed} selection binding schema mismatch")
        if worker_selection.get("nbytes") != selection_spec.get("nbytes") or worker_selection.get("sha256") != selection_spec.get("sha256"):
            raise ProtocolError(f"allocation worker {seed} selection bytes mismatch")
        solver_report = worker.get("solver")
        if not isinstance(solver_report, Mapping):
            raise ProtocolError(f"allocation worker {seed} solver report is missing")
        execution = solver_report.get("execution")
        if not isinstance(execution, Mapping) or not json_exact(execution.get("formal_config"), dict(config_binding)):
            raise ProtocolError(f"deep allocation solver {seed} config binding mismatch")
        validate_allocation_report(solver_report, config, expected_seed=seed)
    if seen_seeds != expected_seeds:
        raise ProtocolError("deep allocation replica order must be [1,777]")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for line in handle:
            value = strict_json_loads(line)
            if not isinstance(value, dict):
                raise ProtocolError("deep allocation JSONL row is not an object")
            rows.append(value)
    return rows


def validate_ready_stages(
    bundle: InputBundle,
    config: Mapping[str, Any],
    root: Path,
    *,
    config_path: Path,
    config_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate A--D outputs without opening any answer/outcome artifact."""

    formal_config_path(root, config_path)
    verify_config_file_binding(config_path, config_binding)
    quota_dir = root / config["outputs"]["quota_search"]
    allocation_dir = root / config["outputs"]["allocation"]
    if not quota_dir.is_dir() or not allocation_dir.is_dir():
        raise ProtocolError("quota/allocation stage directories are missing")
    validate_publication(
        quota_dir,
        ["quota.json", "quota_report.json", "receipt.json"],
        identity={"protocol_id": "quota_capacity_repair_v2", "artifact_kind": "quota_search"},
        status="published_complete_quota_search",
    )
    validate_publication(
        allocation_dir,
        ["allocation.jsonl", "allocation_report.json", "receipt.json"],
        identity={"protocol_id": "quota_capacity_repair_v2", "artifact_kind": "repaired_allocation"},
        status="published_complete_repaired_allocation",
    )
    quota_vector = strict_json_load(quota_dir / "quota.json")
    quota_report = strict_json_load(quota_dir / "quota_report.json")
    allocation_report = strict_json_load(allocation_dir / "allocation_report.json")
    if not isinstance(quota_vector, dict) or not isinstance(quota_report, dict) or not isinstance(allocation_report, dict):
        raise ProtocolError("deep stage payload is not an object")
    if not json_exact(quota_report.get("formal_config"), dict(config_binding)):
        raise ProtocolError("deep quota report config binding mismatch")
    if not json_exact(allocation_report.get("formal_config"), dict(config_binding)):
        raise ProtocolError("deep allocation report config binding mismatch")
    quota_receipt = strict_json_load(quota_dir / "receipt.json")
    allocation_receipt = strict_json_load(allocation_dir / "receipt.json")
    if not isinstance(quota_receipt, dict) or not json_exact(quota_receipt.get("formal_config"), dict(config_binding)):
        raise ProtocolError("deep quota receipt config binding mismatch")
    if not isinstance(allocation_receipt, dict) or not json_exact(allocation_receipt.get("formal_config"), dict(config_binding)):
        raise ProtocolError("deep allocation receipt config binding mismatch")
    targets = targets_from_contract(bundle.contract, config["upper_quotas"])
    if set(quota_report) != {
        "schema_version", "protocol_id", "artifact_kind", "status", "quota_vector", "proof",
        "replicas", "replica_quota_bytes_identical", "formal_config", "answer_values_read", "model_outcomes_read"
    }:
        raise ProtocolError("deep quota report top-level schema mismatch")
    if quota_report.get("schema_version") != 1 or quota_report.get("protocol_id") != "quota_capacity_repair_v2" or quota_report.get("artifact_kind") != "quota_search" or quota_report.get("status") != "quota_search_proven_optimal":
        raise ProtocolError("deep quota report status/identity mismatch")
    if quota_report.get("answer_values_read") is not False or quota_report.get("model_outcomes_read") is not False:
        raise ProtocolError("deep quota report privacy flags are not false")
    quota_check = validate_quota_vector(quota_vector, targets)
    if not json_exact(quota_report.get("quota_vector"), quota_vector):
        raise ProtocolError("deep quota report vector binding mismatch")
    proof = quota_report.get("proof")
    if not isinstance(proof, dict) or proof.get("status") != "quota_search_proven_optimal":
        raise ProtocolError("deep quota proof is not proven optimal")
    validate_quota_proof(proof, targets)
    if quota_report.get("replica_quota_bytes_identical") is not True:
        raise ProtocolError("deep quota replica identity is missing")
    _validate_quota_replicas(quota_report.get("replicas"), config, config_binding)
    rows = _load_jsonl(allocation_dir / "allocation.jsonl")
    allocation_check = verify_selection(
        rows,
        bundle.records,
        bundle.hyperedges,
        targets,
        quota_vector,
        config["solver"]["hash_cost_namespace"],
    )
    if allocation_report.get("selection") != file_spec(allocation_dir / "allocation.jsonl"):
        raise ProtocolError("deep allocation report selection binding mismatch")
    if not isinstance(allocation_report, Mapping) or set(allocation_report) != {
        "schema_version", "protocol_id", "artifact_kind", "status", "quota", "replicas",
        "replica_selection_byte_identical", "selection", "formal_config", "answer_values_read", "model_outcomes_read"
    }:
        raise ProtocolError("deep allocation report top-level schema mismatch")
    if allocation_report.get("schema_version") != 1 or allocation_report.get("protocol_id") != "quota_capacity_repair_v2" or allocation_report.get("artifact_kind") != "repaired_allocation" or allocation_report.get("status") != "allocation_proven_optimal":
        raise ProtocolError("deep allocation report status/identity mismatch")
    if allocation_report.get("quota") != file_spec(quota_dir / "quota.json"):
        raise ProtocolError("deep allocation quota binding mismatch")
    if allocation_report.get("answer_values_read") is not False or allocation_report.get("model_outcomes_read") is not False:
        raise ProtocolError("deep allocation report privacy flags are not false")
    _validate_allocation_replicas(allocation_report, config, config_binding, file_spec(allocation_dir / "allocation.jsonl", include_path=False))
    if allocation_report.get("replica_selection_byte_identical") is not True:
        raise ProtocolError("deep allocation replica identity missing")
    verify_config_file_binding(config_path, config_binding)
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
        "formal_config": dict(config_binding),
    }
