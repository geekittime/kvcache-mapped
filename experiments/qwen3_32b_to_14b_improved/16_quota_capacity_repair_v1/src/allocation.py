"""Stage D: deterministic SHA-cost allocation under a frozen repaired quota."""

from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .common import ProtocolError, json_exact, require_fresh_python_bootstrap
from .quota import Target, _BINARY_TOLERANCE, _environment_gate, targets_from_contract


SELECTION_KEYS = {
    "target_index",
    "target_selection_rank",
    "partition",
    "dataset_id",
    "stratum",
    "record_index",
    "source_projection_sha256",
    "objective_cost_sha256",
}


def objective_digest(namespace: str, target_index: int, projection_sha256: str) -> str:
    return hashlib.sha256(f"{namespace}\0{target_index}\0{projection_sha256}".encode("utf-8")).hexdigest()


def _build_layout(records: Sequence[Mapping[str, Any]], targets: Sequence[Target]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], dict[int, list[int]]]:
    by_cell: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_cell[f"{record['dataset_id']}|{record['stratum']}"].append(record["record_index"])
    variables: list[tuple[int, int]] = []
    target_columns: dict[int, list[int]] = defaultdict(list)
    for target in targets:
        for record_index in by_cell.get(target.cell, []):
            target_columns[target.index].append(len(variables))
            variables.append((target.index, record_index))
    return variables, {pair: i for i, pair in enumerate(variables)}, target_columns


def _build_constraints(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    quota_vector: Mapping[str, Any],
) -> tuple[LinearConstraint, list[tuple[int, int]], dict[tuple[int, int], int], dict[int, list[int]]]:
    variables, variable_index, target_columns = _build_layout(records, targets)
    if not variables:
        raise ProtocolError("allocation has no eligible variables")
    by_cell: dict[str, dict[str, int]] = defaultdict(dict)
    for target in targets:
        by_cell[target.cell][target.partition] = target.index
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def append(columns: Sequence[int], lo: float, hi: float) -> None:
        row = len(lower)
        row_indices.extend([row] * len(columns))
        column_indices.extend(columns)
        values.extend([1.0] * len(columns))
        lower.append(float(lo))
        upper.append(float(hi))

    record_columns: dict[int, list[int]] = defaultdict(list)
    for column, (_, record_index) in enumerate(variables):
        record_columns[record_index].append(column)
    for record_index in range(len(records)):
        append(record_columns.get(record_index, ()), -np.inf, 1.0)
    for edge in hyperedges:
        columns: list[int] = []
        for record_index in edge["member_record_indices"]:
            cell = f"{records[record_index]['dataset_id']}|{records[record_index]['stratum']}"
            for target_index in by_cell[cell].values():
                columns.append(variable_index[(target_index, record_index)])
        append(columns, -np.inf, 1.0)
    for target in targets:
        quota = quota_vector.get(str(target.index))
        if type(quota) is not int or quota < 0 or quota > target.upper_quota:
            raise ProtocolError(f"allocation quota outside frozen upper bound: {target.index}")
        append(target_columns.get(target.index, ()), quota, quota)
    matrix = coo_matrix(
        (np.asarray(values, dtype=float), (np.asarray(row_indices), np.asarray(column_indices))),
        shape=(len(lower), len(variables)),
    ).tocsr()
    return LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)), variables, variable_index, target_columns


def _validated_binary(result: Any, count: int, constraints: LinearConstraint, phase: str) -> np.ndarray:
    values = np.asarray(getattr(result, "x", None), dtype=float)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ProtocolError(f"{phase} incumbent missing/wrong-length/nonfinite")
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > _BINARY_TOLERANCE) or np.any((rounded != 0.0) & (rounded != 1.0)):
        raise ProtocolError(f"{phase} incumbent is not binary")
    activity = np.asarray(constraints.A @ rounded, dtype=float)
    if np.any(activity < constraints.lb - _BINARY_TOLERANCE) or np.any(activity > constraints.ub + _BINARY_TOLERANCE):
        raise ProtocolError(f"{phase} incumbent violates allocation constraints")
    return rounded.astype(np.uint8)


def _solver_result(result: Any, phase: str) -> tuple[int, bool, str]:
    status = getattr(result, "status", None)
    success = getattr(result, "success", None)
    message = getattr(result, "message", None)
    if type(status) is not int or type(success) is not bool or not isinstance(message, str):
        raise ProtocolError(f"{phase} result status/success/message type mismatch")
    if status != 0 or success is not True or not message.startswith("Optimization terminated successfully.") or "HiGHS Status 7: Optimal" not in message:
        raise ProtocolError(f"{phase} did not prove optimality: status={status} message={message}")
    return status, success, message


def verify_selection(
    selection: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    targets: Sequence[Target],
    quota_vector: Mapping[str, Any],
    namespace: str,
) -> dict[str, Any]:
    target_by_index = {target.index: target for target in targets}
    if not isinstance(quota_vector, Mapping) or set(quota_vector) != {str(target.index) for target in targets}:
        raise ProtocolError("allocation quota vector keys mismatch")
    for target in targets:
        quota = quota_vector[str(target.index)]
        if type(quota) is not int or quota < 0 or quota > target.upper_quota:
            raise ProtocolError("allocation quota vector contains non-integer/out-of-bound value")
    counts: Counter[int] = Counter()
    assigned: Counter[int] = Counter()
    selected: set[int] = set()
    previous_target = -1
    expected_rank = 0
    previous_cost_record: tuple[str, int] = ("", -1)
    for row in selection:
        if set(row) != SELECTION_KEYS:
            raise ProtocolError("allocation row keys mismatch")
        if type(row.get("target_index")) is not int or row["target_index"] not in target_by_index:
            raise ProtocolError("allocation target index mismatch")
        if type(row.get("target_selection_rank")) is not int or row["target_selection_rank"] < 0:
            raise ProtocolError("allocation target rank type mismatch")
        if type(row.get("record_index")) is not int or not 0 <= row["record_index"] < len(records):
            raise ProtocolError("allocation record index mismatch")
        if not all(isinstance(row.get(key), str) for key in ("partition", "dataset_id", "stratum", "source_projection_sha256", "objective_cost_sha256")):
            raise ProtocolError("allocation string field type mismatch")
        if len(row["source_projection_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["source_projection_sha256"]):
            raise ProtocolError("allocation source hash is not lowercase SHA-256")
        if len(row["objective_cost_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in row["objective_cost_sha256"]):
            raise ProtocolError("allocation objective hash is not lowercase SHA-256")
        target = target_by_index[row["target_index"]]
        record = records[row["record_index"]]
        if (
            row["partition"] != target.partition
            or row["dataset_id"] != target.dataset_id
            or row["stratum"] != target.stratum
            or row["source_projection_sha256"] != record["source_projection_sha256"]
            or row["objective_cost_sha256"] != objective_digest(namespace, target.index, record["source_projection_sha256"])
        ):
            raise ProtocolError("allocation target/record/cost binding mismatch")
        if row["target_index"] != previous_target:
            if row["target_index"] <= previous_target:
                raise ProtocolError("allocation targets are not in public order")
            previous_target = row["target_index"]
            expected_rank = 0
            previous_cost_record = ("", -1)
        if row["target_selection_rank"] != expected_rank:
            raise ProtocolError("allocation ranks are not contiguous")
        key = (row["objective_cost_sha256"], row["record_index"])
        if key <= previous_cost_record:
            raise ProtocolError("allocation rows are not SHA-cost ordered")
        previous_cost_record = key
        expected_rank += 1
        counts[row["target_index"]] += 1
        assigned[row["record_index"]] += 1
        selected.add(row["record_index"])
    if any(value > 1 for value in assigned.values()):
        raise ProtocolError("record assigned to more than one target")
    violating = sum(sum(index in selected for index in edge["member_record_indices"]) > 1 for edge in hyperedges)
    if violating:
        raise ProtocolError("direct collision hyperedge violated")
    observed = {str(target.index): counts[target.index] for target in targets}
    expected = {str(target.index): quota_vector[str(target.index)] for target in targets}
    if observed != expected:
        raise ProtocolError("allocation does not satisfy repaired quotas")
    return {
        "selected_records": len(selection),
        "selected_record_indices_unique": len(selected) == len(selection),
        "target_counts": observed,
        "violating_hyperedges": violating,
    }


def solve_allocation_rows(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    quota_vector: Mapping[str, Any],
    *,
    seed: int = 1,
    require_environment: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Keep direct library use fail-closed with the script workers: no solver
    # may run while an adjacent interpreter cache could be reused.
    require_fresh_python_bootstrap()
    if type(seed) is not int or seed <= 0:
        raise ProtocolError("allocation seed must be a positive integer")
    if require_environment:
        _environment_gate(config["solver"]["required_environment"])
    if scipy.__version__ != config["solver"]["scipy_version"]:
        raise ProtocolError("SciPy version mismatch")
    targets = targets_from_contract(contract, config.get("upper_quotas"))
    constraints, variables, variable_index, target_columns = _build_constraints(records, hyperedges, targets, quota_vector)
    n = len(variables)
    integrality = np.ones(n, dtype=np.uint8)
    options = {"disp": False, **dict(config["solver"]["highs_options"]), "random_seed": seed}
    execution = {
        "process_id": os.getpid(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "highs_random_seed": seed,
        "thread_environment": {name: os.environ.get(name) for name in config["solver"]["required_environment"]},
        "highs_options": dict(options),
        "runtime_identity": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "scipy_version": scipy.__version__,
        },
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*", category=RuntimeWarning)
        phase1 = milp(np.zeros(n), integrality=integrality, bounds=Bounds(np.zeros(n), np.ones(n)), constraints=constraints, options=options)
    status1, success1, message1 = _solver_result(phase1, "allocation_phase_1")
    _validated_binary(phase1, n, constraints, "allocation_phase_1")
    namespace = config["solver"]["hash_cost_namespace"]
    digests = [objective_digest(namespace, target_index, records[record_index]["source_projection_sha256"]) for target_index, record_index in variables]
    numerators = [int(digest[:13], 16) for digest in digests]
    denominator = int(config["solver"]["objective_denominator"])
    if denominator != 16 ** 13:
        raise ProtocolError("objective denominator is not the frozen 52-bit denominator")
    costs = np.asarray([numerator / float(denominator) for numerator in numerators], dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected:.*", category=RuntimeWarning)
        phase2 = milp(costs, integrality=integrality, bounds=Bounds(np.zeros(n), np.ones(n)), constraints=constraints, options=options)
    status2, success2, message2 = _solver_result(phase2, "allocation_phase_2")
    rounded = _validated_binary(phase2, n, constraints, "allocation_phase_2")
    objective = getattr(phase2, "fun", None)
    if isinstance(objective, bool) or not isinstance(objective, (float, np.floating)) or not math.isfinite(float(objective)):
        raise ProtocolError("allocation phase-2 objective missing/nonfinite")
    chosen = [index for index, value in enumerate(rounded) if value == 1]
    selected_numerator = sum(numerators[index] for index in chosen)
    exact_objective = selected_numerator / float(denominator)
    raw_objective = float(np.dot(costs, np.asarray(getattr(phase2, "x"), dtype=float)))
    if not math.isfinite(raw_objective) or abs(float(objective) - raw_objective) > 1e-10 or abs(float(objective) - exact_objective) > 1e-10:
        raise ProtocolError("allocation phase-2 objective does not match binary incumbent")
    by_target: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for index in chosen:
        target_index, record_index = variables[index]
        by_target[target_index].append((digests[index], record_index))
    target_by_index = {target.index: target for target in targets}
    selection: list[dict[str, Any]] = []
    for target_index in sorted(by_target):
        target = target_by_index[target_index]
        for rank, (digest, record_index) in enumerate(sorted(by_target[target_index])):
            record = records[record_index]
            selection.append({
                "target_index": target_index,
                "target_selection_rank": rank,
                "partition": target.partition,
                "dataset_id": target.dataset_id,
                "stratum": target.stratum,
                "record_index": record_index,
                "source_projection_sha256": record["source_projection_sha256"],
                "objective_cost_sha256": digest,
            })
    verification = verify_selection(selection, records, hyperedges, targets, quota_vector, namespace)
    report = {
        "schema_version": 1,
        "status": "allocation_proven_optimal",
        "seed": seed,
        "process_id": os.getpid(),
        "phase_1": {"status": status1, "success": success1, "message": message1, "classification": "optimal_feasible"},
        "phase_2": {
            "status": status2,
            "success": success2,
            "message": message2,
            "classification": "proven_optimal",
            "objective": float(objective),
            "selected_objective_numerator": selected_numerator,
            "objective_denominator": denominator,
            "binary_tolerance": float(_BINARY_TOLERANCE),
        },
        "variable_count": n,
        "constraint_count": int(constraints.A.shape[0]),
        "execution": execution,
        "verification": verification,
        "privacy": {"answer_values_read": False, "model_outcomes_read": False, "raw_text_persisted": False},
    }
    return selection, report


def validate_selection_jsonl(rows: Sequence[Mapping[str, Any]]) -> None:
    """Check the structural order before a deep consumer opens the rows."""

    previous_target = -1
    expected_rank = 0
    previous: tuple[str, int] = ("", -1)
    for row in rows:
        if set(row) != SELECTION_KEYS:
            raise ProtocolError("allocation selection row typed keys mismatch")
        target = row["target_index"]
        if type(target) is not int or target < 0:
            raise ProtocolError("allocation selection target type mismatch")
        if target != previous_target:
            if target <= previous_target:
                raise ProtocolError("allocation selection target order mismatch")
            previous_target = target
            expected_rank = 0
            previous = ("", -1)
        if type(row.get("target_selection_rank")) is not int or row["target_selection_rank"] != expected_rank:
            raise ProtocolError("allocation selection rank mismatch")
        if type(row.get("record_index")) is not int or row["record_index"] < 0:
            raise ProtocolError("allocation selection record index type mismatch")
        for key in ("partition", "dataset_id", "stratum"):
            if not isinstance(row.get(key), str):
                raise ProtocolError("allocation selection identity field type mismatch")
        for key in ("source_projection_sha256", "objective_cost_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ProtocolError("allocation selection hash type mismatch")
        key = (row["objective_cost_sha256"], row["record_index"])
        if key <= previous:
            raise ProtocolError("allocation selection cost order mismatch")
        previous = key
        expected_rank += 1


def validate_allocation_report(report: Mapping[str, Any], config: Mapping[str, Any], *, expected_seed: int | None = None) -> dict[str, Any]:
    """Validate one Stage-D solver report independently of its selection."""

    if not isinstance(report, Mapping):
        raise ProtocolError("allocation report must be an object")
    expected_keys = {
        "schema_version", "status", "seed", "process_id", "phase_1", "phase_2",
        "variable_count", "constraint_count", "execution", "verification", "privacy",
    }
    if set(report) != expected_keys or report.get("schema_version") != 1 or report.get("status") != "allocation_proven_optimal":
        raise ProtocolError("allocation report top-level schema/status mismatch")
    if type(report.get("seed")) is not int or report["seed"] <= 0 or (expected_seed is not None and report["seed"] != expected_seed):
        raise ProtocolError("allocation report seed mismatch")
    if type(report.get("process_id")) is not int or report["process_id"] <= 0 or type(report.get("variable_count")) is not int or report["variable_count"] <= 0 or type(report.get("constraint_count")) is not int or report["constraint_count"] <= 0:
        raise ProtocolError("allocation report integer field mismatch")
    execution = report.get("execution")
    expected_execution = {"process_id", "python_hash_seed", "highs_random_seed", "thread_environment", "highs_options", "runtime_identity"}
    if not isinstance(execution, Mapping) or set(execution) != expected_execution or execution.get("process_id") != report["process_id"] or type(execution.get("highs_random_seed")) is not int or execution["highs_random_seed"] != report["seed"] or not isinstance(execution.get("python_hash_seed"), (str, type(None))):
        raise ProtocolError("allocation report execution binding mismatch")
    if execution.get("thread_environment") != config["solver"]["required_environment"]:
        raise ProtocolError("allocation report thread environment mismatch")
    options = execution.get("highs_options")
    expected_options = {"disp": False, **dict(config["solver"]["highs_options"]), "random_seed": report["seed"]}
    if not json_exact(options, expected_options):
        raise ProtocolError("allocation report HiGHS options mismatch")
    phase_1 = report.get("phase_1")
    phase_2 = report.get("phase_2")
    if not isinstance(phase_1, Mapping) or set(phase_1) != {"status", "success", "message", "classification"}:
        raise ProtocolError("phase_1 report keys mismatch")
    if not isinstance(phase_2, Mapping) or set(phase_2) != {"status", "success", "message", "classification", "objective", "selected_objective_numerator", "objective_denominator", "binary_tolerance"}:
        raise ProtocolError("phase_2 report keys mismatch")
    for phase_name in ("phase_1", "phase_2"):
        phase = report.get(phase_name)
        if type(phase.get("status")) is not int or phase["status"] != 0 or type(phase.get("success")) is not bool or phase["success"] is not True or phase.get("classification") not in ({"optimal_feasible"} if phase_name == "phase_1" else {"proven_optimal"}) or not isinstance(phase.get("message"), str) or not phase["message"].startswith("Optimization terminated successfully.") or "HiGHS Status 7: Optimal" not in phase["message"]:
            raise ProtocolError(f"{phase_name} status is not strict Optimal")
    phase2 = report["phase_2"]
    if isinstance(phase2.get("objective"), bool) or not isinstance(phase2.get("objective"), float) or not math.isfinite(phase2["objective"]):
        raise ProtocolError("allocation objective is nonfinite/not float")
    if type(phase2.get("selected_objective_numerator")) is not int or phase2["selected_objective_numerator"] < 0 or type(phase2.get("objective_denominator")) is not int or phase2["objective_denominator"] != 16 ** 13 or not isinstance(phase2.get("binary_tolerance"), float) or not math.isfinite(phase2["binary_tolerance"]) or phase2["binary_tolerance"] != _BINARY_TOLERANCE:
        raise ProtocolError("allocation exact objective fields mismatch")
    verification = report.get("verification")
    if not isinstance(verification, Mapping) or set(verification) != {"selected_records", "selected_record_indices_unique", "target_counts", "violating_hyperedges"}:
        raise ProtocolError("allocation verification keys mismatch")
    if type(verification.get("selected_records")) is not int or verification["selected_records"] < 0 or type(verification.get("selected_record_indices_unique")) is not bool or type(verification.get("violating_hyperedges")) is not int or verification["violating_hyperedges"] < 0:
        raise ProtocolError("allocation verification integer/bool fields mismatch")
    target_counts = verification.get("target_counts")
    if not isinstance(target_counts, Mapping) or any(type(value) is not int or value < 0 for value in target_counts.values()):
        raise ProtocolError("allocation verification target counts are not strict integers")
    if set(target_counts) != {str(index) for index in range(18)}:
        raise ProtocolError("allocation verification target count keys mismatch")
    runtime_identity = execution.get("runtime_identity")
    if not isinstance(runtime_identity, Mapping) or set(runtime_identity) != {"python_executable", "python_version", "scipy_version"} or any(not isinstance(value, str) for value in runtime_identity.values()) or runtime_identity["scipy_version"] != config["solver"]["scipy_version"]:
        raise ProtocolError("allocation runtime identity mismatch")
    if report.get("privacy") != {"answer_values_read": False, "model_outcomes_read": False, "raw_text_persisted": False}:
        raise ProtocolError("allocation privacy flags are not strict false")
    return {"seed": report["seed"], "process_id": report["process_id"], "objective": phase2["objective"]}
