from __future__ import annotations

import hashlib
import json
import math
import os
import re
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

from .common import (
    ProtocolError,
    atomic_json,
    atomic_jsonl,
    atomic_write,
    claim_artifact,
    file_spec,
    finalize_publication,
    json_exact,
    runtime_identity,
    safe_relative_output,
    strict_json_loads,
    validate_publication,
    verify_runtime_identity,
)
from .graph import _implementation
from .validate_graph import validate_graph


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = {
    "record_index", "dataset_id", "stratum", "source_index", "source_projection_sha256",
}
_HYPEREDGE_KEYS = {
    "hyperedge_id", "member_record_indices", "member_count",
    "relation_kinds", "derivation_count_by_kind",
}
_TARGET_KEYS = {
    "target_index", "partition", "cell", "dataset_id", "stratum", "exact_quota",
}
_SELECTION_KEYS = {
    "target_index", "target_selection_rank", "partition", "dataset_id", "stratum",
    "record_index", "source_projection_sha256", "objective_cost_sha256",
}
_BINARY_TOLERANCE = 1e-7


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [strict_json_loads(line) for line in path.open(encoding="utf-8")]


def _validate_solver_inputs(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> None:
    cells: set[str] = set()
    previous_projection = ""
    for expected_index, record in enumerate(records):
        if (
            set(record) != _RECORD_KEYS
            or type(record.get("record_index")) is not int
            or record["record_index"] != expected_index
            or type(record.get("source_index")) is not int
            or record["source_index"] < 0
            or not isinstance(record.get("dataset_id"), str)
            or not isinstance(record.get("stratum"), str)
            or not isinstance(record.get("source_projection_sha256"), str)
            or not _HEX64.fullmatch(record["source_projection_sha256"])
            or record["source_projection_sha256"] <= previous_projection
        ):
            raise ProtocolError("solver record typed schema mismatch")
        previous_projection = record["source_projection_sha256"]
        cells.add(f"{record['dataset_id']}|{record['stratum']}")
    if not records:
        raise ProtocolError("solver records cannot be empty")
    previous_edge_id = ""
    for edge in hyperedges:
        members = edge.get("member_record_indices")
        kinds = edge.get("relation_kinds")
        derivations = edge.get("derivation_count_by_kind")
        if (
            set(edge) != _HYPEREDGE_KEYS
            or not isinstance(edge.get("hyperedge_id"), str)
            or not _HEX64.fullmatch(edge["hyperedge_id"])
            or edge["hyperedge_id"] <= previous_edge_id
            or not isinstance(members, list)
            or any(type(value) is not int for value in members)
            or len(members) < 2
            or members != sorted(set(members))
            or members[0] < 0
            or members[-1] >= len(records)
            or type(edge.get("member_count")) is not int
            or edge["member_count"] != len(members)
            or not isinstance(kinds, list)
            or kinds != sorted(set(kinds))
            or any(not isinstance(kind, str) for kind in kinds)
            or not isinstance(derivations, dict)
            or set(derivations) != set(kinds)
            or any(type(value) is not int or value <= 0 for value in derivations.values())
        ):
            raise ProtocolError("solver hyperedge typed schema mismatch")
        previous_edge_id = edge["hyperedge_id"]
    targets = contract.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ProtocolError("solver target list missing")
    seen_target_cells: set[tuple[str, str]] = set()
    for expected_index, target in enumerate(targets):
        if (
            set(target) != _TARGET_KEYS
            or type(target.get("target_index")) is not int
            or target["target_index"] != expected_index
            or target.get("partition") not in {"screen", "confirm"}
            or not isinstance(target.get("cell"), str)
            or not isinstance(target.get("dataset_id"), str)
            or not isinstance(target.get("stratum"), str)
            or target["cell"] != f"{target['dataset_id']}|{target['stratum']}"
            or type(target.get("exact_quota")) is not int
            or target["exact_quota"] < 0
            or (target["cell"] not in cells and target["exact_quota"] != 0)
            or (target["partition"], target["cell"]) in seen_target_cells
        ):
            raise ProtocolError("solver target typed schema mismatch")
        seen_target_cells.add((target["partition"], target["cell"]))
    if any(
        {(partition, cell) for partition in ("screen", "confirm")} - seen_target_cells
        for cell in cells
    ):
        raise ProtocolError("every record cell must have exactly two partition targets")


def load_selection_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    previous_target = -1
    expected_rank = 0
    previous_cost_record = ("", -1)
    for row in rows:
        if (
            set(row) != _SELECTION_KEYS
            or type(row.get("target_index")) is not int
            or type(row.get("target_selection_rank")) is not int
            or row["target_selection_rank"] < 0
            or type(row.get("record_index")) is not int
            or row["record_index"] < 0
            or not isinstance(row.get("partition"), str)
            or not isinstance(row.get("dataset_id"), str)
            or not isinstance(row.get("stratum"), str)
            or not isinstance(row.get("objective_cost_sha256"), str)
            or not _HEX64.fullmatch(row["objective_cost_sha256"])
            or not isinstance(row.get("source_projection_sha256"), str)
            or not _HEX64.fullmatch(row["source_projection_sha256"])
        ):
            raise ProtocolError("allocation row typed schema mismatch")
        target = row["target_index"]
        if target != previous_target:
            if target <= previous_target:
                raise ProtocolError("allocation targets are not ascending")
            previous_target = target
            expected_rank = 0
            previous_cost_record = ("", -1)
        if row["target_selection_rank"] != expected_rank:
            raise ProtocolError("allocation target ranks are not contiguous")
        cost_record = (row["objective_cost_sha256"], row["record_index"])
        if cost_record <= previous_cost_record:
            raise ProtocolError("allocation target rows are not hash-cost ordered")
        expected_rank += 1
        previous_cost_record = cost_record
    return rows


def verify_deep_receipt(config: Mapping[str, Any], root: Path, graph_dir: Path) -> dict[str, Any]:
    receipt_dir = safe_relative_output(root, config["outputs"]["graph_deep_receipt"])
    validate_publication(
        receipt_dir,
        ["receipt.json"],
        identity={"protocol_id": config["protocol_id"], "artifact_kind": "graph_deep_validation"},
        status="published_complete_graph_deep_validation",
    )
    receipt = strict_json_loads((receipt_dir / "receipt.json").read_text())
    expected_keys = {
        "schema_version", "protocol_id", "artifact_kind", "status", "cpu_only",
        "runtime_identity", "graph_build_process", "deep_validation_process",
        "separate_process_assertion", "graph_PUBLISHED", "graph_manifest",
        "deep_recomputed_from_pinned_raw_sources", "records_byte_identical",
        "hyperedges_byte_identical", "graph_report_byte_identical",
        "solver_contract_byte_identical", "replica_files",
        "required_allocation_command", "claim_boundary",
    }
    graph_manifest = strict_json_loads((graph_dir / "manifest.json").read_text())
    graph_process = receipt.get("graph_build_process", {})
    deep_process = receipt.get("deep_validation_process", {})
    if (
        set(receipt) != expected_keys
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != config["protocol_id"]
        or receipt.get("artifact_kind") != "graph_deep_validation"
        or receipt.get("status") != "pass_pinned_source_graph_byte_identical"
        or receipt.get("cpu_only") is not True
        or not json_exact(receipt.get("runtime_identity"), config["runtime_identity"])
        or not json_exact(graph_process, {
            "process_id": graph_manifest["runtime"]["process_id"],
            "python_hash_seed": config["deterministic_processes"]["graph_build_python_hash_seed"],
        })
        or type(deep_process.get("process_id")) is not int
        or deep_process["process_id"] <= 0
        or deep_process["process_id"] == graph_process.get("process_id")
        or deep_process.get("python_hash_seed") != config["deterministic_processes"]["graph_deep_python_hash_seed"]
        or receipt.get("separate_process_assertion") is not True
        or not json_exact(receipt.get("graph_PUBLISHED"), file_spec(graph_dir / "PUBLISHED.json"))
        or not json_exact(receipt.get("graph_manifest"), file_spec(graph_dir / "manifest.json"))
        or receipt.get("deep_recomputed_from_pinned_raw_sources") is not True
        or not all(receipt.get(key) is True for key in (
            "records_byte_identical", "hyperedges_byte_identical",
            "graph_report_byte_identical", "solver_contract_byte_identical",
        ))
        or receipt.get("required_allocation_command") != config["commands"]["allocation"]
        or receipt.get("claim_boundary") != "deep validation only; no MILP solved and no allocation selected"
    ):
        raise ProtocolError("graph deep validation receipt semantics mismatch")
    expected_replica_files = {
        name: file_spec(graph_dir / name, include_path=False)
        for name in (
            "records.jsonl", "hyperedges.jsonl", "graph_report.json", "solver_contract.json"
        )
    }
    if not json_exact(receipt.get("replica_files"), expected_replica_files):
        raise ProtocolError("graph deep receipt replica bindings mismatch")
    return {
        "PUBLISHED": file_spec(receipt_dir / "PUBLISHED.json"),
        "receipt": file_spec(receipt_dir / "receipt.json"),
    }


def _eligible_variables(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int], dict[int, list[int]]]:
    records_by_cell: dict[str, list[int]] = defaultdict(list)
    for record in records:
        records_by_cell[f"{record['dataset_id']}|{record['stratum']}"].append(record["record_index"])
    variables: list[tuple[int, int]] = []
    target_variables: dict[int, list[int]] = defaultdict(list)
    for target in contract["targets"]:
        target_index = target["target_index"]
        for record_index in records_by_cell[target["cell"]]:
            target_variables[target_index].append(len(variables))
            variables.append((target_index, record_index))
    return variables, {pair: index for index, pair in enumerate(variables)}, dict(target_variables)


def _build_constraints(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    variables: Sequence[tuple[int, int]],
    variable_index: Mapping[tuple[int, int], int],
    target_variables: Mapping[int, Sequence[int]],
) -> LinearConstraint:
    targets_by_cell: dict[str, list[int]] = defaultdict(list)
    for target in contract["targets"]:
        targets_by_cell[target["cell"]].append(target["target_index"])
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    def append_constraint(columns: Sequence[int], minimum: float, maximum: float) -> None:
        row = len(lower)
        row_indices.extend([row] * len(columns))
        column_indices.extend(columns)
        values.extend([1.0] * len(columns))
        lower.append(minimum)
        upper.append(maximum)

    record_variables: dict[int, list[int]] = defaultdict(list)
    for column, (_, record_index) in enumerate(variables):
        record_variables[record_index].append(column)
    for record_index in range(len(records)):
        append_constraint(record_variables[record_index], -np.inf, 1.0)
    for edge in hyperedges:
        columns = []
        for record_index in edge["member_record_indices"]:
            cell = f"{records[record_index]['dataset_id']}|{records[record_index]['stratum']}"
            columns.extend(
                variable_index[(target_index, record_index)]
                for target_index in targets_by_cell[cell]
            )
        append_constraint(columns, -np.inf, 1.0)
    for target in contract["targets"]:
        quota = float(target["exact_quota"])
        append_constraint(target_variables.get(target["target_index"], ()), quota, quota)
    matrix = coo_matrix(
        (np.asarray(values), (np.asarray(row_indices), np.asarray(column_indices))),
        shape=(len(lower), len(variables)),
    ).tocsr()
    return LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))


def objective_digest(namespace: str, target_index: int, projection_sha256: str) -> str:
    return hashlib.sha256(f"{namespace}\0{target_index}\0{projection_sha256}".encode()).hexdigest()


def _environment_gate(required: Mapping[str, str]) -> None:
    observed = {name: os.environ.get(name) for name in required}
    if observed != dict(required):
        raise ProtocolError(f"solver thread environment mismatch: {observed}")


def _result_header(result: Any, phase: str) -> tuple[int, bool, str]:
    status = getattr(result, "status", None)
    success = getattr(result, "success", None)
    message = getattr(result, "message", None)
    if type(status) is not int or type(success) is not bool or not isinstance(message, str):
        raise ProtocolError(f"{phase} solver result typed schema mismatch")
    return status, success, message


def _validated_binary_incumbent(
    result: Any,
    variable_count: int,
    constraints: LinearConstraint,
    phase: str,
) -> np.ndarray:
    values = np.asarray(getattr(result, "x", None), dtype=float)
    if values.shape != (variable_count,) or not np.all(np.isfinite(values)):
        raise ProtocolError(f"{phase} incumbent missing, wrong-length, or nonfinite")
    rounded = np.rint(values)
    if np.any(np.abs(values - rounded) > _BINARY_TOLERANCE) or np.any(
        (rounded != 0.0) & (rounded != 1.0)
    ):
        raise ProtocolError(f"{phase} incumbent is not binary within frozen tolerance")
    activity = np.asarray(constraints.A @ rounded, dtype=float)
    lower = np.asarray(constraints.lb, dtype=float)
    upper = np.asarray(constraints.ub, dtype=float)
    if np.any(activity < lower - _BINARY_TOLERANCE) or np.any(
        activity > upper + _BINARY_TOLERANCE
    ):
        raise ProtocolError(f"{phase} incumbent violates published MILP constraints")
    return rounded.astype(np.uint8)


def solver_reports_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    def normalized(value: Mapping[str, Any]) -> dict[str, Any]:
        result = json.loads(json.dumps(value, allow_nan=False))
        execution = result.get("execution", {})
        for key in ("process_id", "python_hash_seed", "highs_random_seed"):
            if key in execution:
                execution[key] = "<replica-specific>"
        if "random_seed" in execution.get("highs_options", {}):
            execution["highs_options"]["random_seed"] = "<replica-specific>"
        return result

    return json_exact(normalized(left), normalized(right))


def verify_selection(
    selection: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    require_exact_quotas: bool,
) -> dict[str, Any]:
    record_assignments: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    selected_records: set[int] = set()
    targets = {row["target_index"]: row for row in contract["targets"]}
    for row in selection:
        if (
            set(row) != _SELECTION_KEYS
            or type(row.get("target_index")) is not int
            or row["target_index"] not in targets
            or type(row.get("target_selection_rank")) is not int
            or row["target_selection_rank"] < 0
            or type(row.get("record_index")) is not int
            or not 0 <= row["record_index"] < len(records)
        ):
            raise ProtocolError("selection row typed/index schema mismatch")
        target = targets[row["target_index"]]
        record = records[row["record_index"]]
        if (
            row["partition"] != target["partition"]
            or row["dataset_id"] != target["dataset_id"]
            or row["stratum"] != target["stratum"]
            or row["source_projection_sha256"] != record["source_projection_sha256"]
            or f"{record['dataset_id']}|{record['stratum']}" != target["cell"]
            or row["objective_cost_sha256"] != objective_digest(
                contract["solver"]["hash_cost_namespace"],
                target["target_index"],
                record["source_projection_sha256"],
            )
        ):
            raise ProtocolError("selection row target/record binding mismatch")
        record_assignments[row["record_index"]] += 1
        target_counts[row["target_index"]] += 1
        selected_records.add(row["record_index"])
    if any(value > 1 for value in record_assignments.values()):
        raise ProtocolError("record assigned to more than one target")
    violating_edges = sum(
        sum(index in selected_records for index in edge["member_record_indices"]) > 1
        for edge in hyperedges
    )
    if violating_edges:
        raise ProtocolError("direct collision hyperedge violated")
    expected_counts = {
        target["target_index"]: target["exact_quota"] for target in contract["targets"]
    }
    observed_counts = {key: target_counts[key] for key in sorted(expected_counts)}
    if require_exact_quotas and observed_counts != expected_counts:
        raise ProtocolError("target exact quotas not satisfied")
    return {
        "selected_records": len(selection),
        "selected_record_indices_unique": len(selected_records) == len(selection),
        "target_counts": {str(key): observed_counts[key] for key in sorted(expected_counts)},
        "violating_hyperedges": violating_edges,
    }


def solve_allocation_rows(
    records: Sequence[Mapping[str, Any]],
    hyperedges: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    require_environment: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if scipy.__version__ != contract["solver"]["scipy_version"]:
        raise ProtocolError("SciPy version mismatch")
    if require_environment:
        _environment_gate(contract["solver"]["required_environment"])
    _validate_solver_inputs(records, hyperedges, contract)
    variables, variable_index, target_variables = _eligible_variables(records, contract)
    constraints = _build_constraints(
        records, hyperedges, contract, variables, variable_index, target_variables
    )
    integrality = np.ones(len(variables), dtype=np.uint8)
    bounds = Bounds(np.zeros(len(variables)), np.ones(len(variables)))
    highs_random_seed = int(os.environ.get("HYPERGRAPH_HIGHS_RANDOM_SEED", "0"))
    options = {
        "disp": False,
        **dict(contract["solver"].get("highs_options", {
            "threads": 1, "presolve": True, "mip_rel_gap": 0.0,
        })),
        "random_seed": highs_random_seed,
    }
    execution = {
        "process_id": os.getpid(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "highs_random_seed": highs_random_seed,
        "thread_environment": {
            name: os.environ.get(name)
            for name in contract["solver"]["required_environment"]
        },
        "highs_options": dict(options),
        "runtime_identity": runtime_identity(),
    }
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        # SciPy 1.12 deliberately forwards these two options verbatim to
        # HiGHS even though its Python-side allowlist emits a warning.
        phase_1 = milp(
            np.zeros(len(variables)),
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=dict(options),
        )
    phase_1_status, phase_1_success, phase_1_message = _result_header(phase_1, "phase_1")
    phase_1_report = {
        "status": phase_1_status,
        "success": phase_1_success,
        "message": phase_1_message,
        "classification": "optimal_feasible" if phase_1_status == 0 else "explicit_infeasible" if phase_1_status == 2 else "nonpublishable_solver_failure",
    }
    if phase_1_status == 2:
        if (
            phase_1_success is not False
            or getattr(phase_1, "x", None) is not None
            or not phase_1_message.startswith("The problem is infeasible.")
            or "model_status is Infeasible" not in phase_1_message
        ):
            raise ProtocolError("phase_1 status 2 lacks pinned explicit infeasibility semantics")
        return [], {
            "original_quota_feasible": False,
            "explicit_infeasibility_certificate": True,
            "phase_1": phase_1_report,
            "phase_2_run": False,
            "variable_count": len(variables),
            "constraint_count": constraints.A.shape[0],
            "execution": execution,
        }
    if (
        phase_1_status != 0
        or phase_1_success is not True
        or not phase_1_message.startswith("Optimization terminated successfully.")
        or "HiGHS Status 7: Optimal" not in phase_1_message
    ):
        raise ProtocolError(
            f"phase_1 did not prove feasibility or explicit infeasibility: status={phase_1_status} message={phase_1_message}"
        )
    _validated_binary_incumbent(phase_1, len(variables), constraints, "phase_1")
    namespace = contract["solver"]["hash_cost_namespace"]
    digests = [
        objective_digest(namespace, target_index, records[record_index]["source_projection_sha256"])
        for target_index, record_index in variables
    ]
    cost_numerators = [int(value[:13], 16) for value in digests]
    objective_denominator = 16 ** 13
    costs = np.asarray([value / float(objective_denominator) for value in cost_numerators])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        phase_2 = milp(
            costs,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=dict(options),
        )
    phase_2_status, phase_2_success, phase_2_message = _result_header(phase_2, "phase_2")
    if (
        phase_2_status != 0
        or phase_2_success is not True
        or not phase_2_message.startswith("Optimization terminated successfully.")
        or "HiGHS Status 7: Optimal" not in phase_2_message
    ):
        raise ProtocolError(
            f"hash-cost MILP did not prove optimality: status={phase_2_status} message={phase_2_message}"
        )
    rounded = _validated_binary_incumbent(phase_2, len(variables), constraints, "phase_2")
    phase_2_fun = getattr(phase_2, "fun", None)
    if isinstance(phase_2_fun, bool) or not isinstance(phase_2_fun, (float, np.floating)) or not math.isfinite(float(phase_2_fun)):
        raise ProtocolError("phase_2 objective is missing, non-float, or nonfinite")
    chosen = [index for index, value in enumerate(rounded) if value == 1]
    selected_objective_numerator = sum(cost_numerators[index] for index in chosen)
    exact_selected_objective = selected_objective_numerator / float(objective_denominator)
    raw_incumbent_objective = float(np.dot(costs, np.asarray(phase_2.x, dtype=float)))
    if (
        not math.isfinite(raw_incumbent_objective)
        or abs(float(phase_2_fun) - raw_incumbent_objective) > 1e-10
        or abs(float(phase_2_fun) - exact_selected_objective) > 1e-10
    ):
        raise ProtocolError("phase_2 objective does not match the selected binary incumbent")
    by_target: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for variable in chosen:
        target_index, record_index = variables[variable]
        by_target[target_index].append((digests[variable], record_index))
    targets = {row["target_index"]: row for row in contract["targets"]}
    selection = []
    for target_index in sorted(by_target):
        target = targets[target_index]
        for rank, (cost_digest, record_index) in enumerate(sorted(by_target[target_index])):
            record = records[record_index]
            selection.append({
                "target_index": target_index,
                "target_selection_rank": rank,
                "partition": target["partition"],
                "dataset_id": target["dataset_id"],
                "stratum": target["stratum"],
                "record_index": record_index,
                "source_projection_sha256": record["source_projection_sha256"],
                "objective_cost_sha256": cost_digest,
            })
    verification = verify_selection(
        selection, records, hyperedges, contract, require_exact_quotas=True
    )
    return selection, {
        "original_quota_feasible": True,
        "explicit_infeasibility_certificate": False,
        "phase_1": phase_1_report,
        "phase_2_run": True,
        "phase_2": {
            "status": phase_2_status,
            "success": phase_2_success,
            "message": phase_2_message,
            "classification": "proven_optimal",
            "objective": float(phase_2_fun),
            "selected_objective_numerator": selected_objective_numerator,
            "objective_denominator": objective_denominator,
            "binary_tolerance": _BINARY_TOLERANCE,
        },
        "variable_count": len(variables),
        "constraint_count": constraints.A.shape[0],
        "verification": verification,
        "execution": execution,
    }


def run_worker(graph_dir: Path, output_path: Path) -> dict[str, Any]:
    records = load_jsonl(graph_dir / "records.jsonl")
    hyperedges = load_jsonl(graph_dir / "hyperedges.jsonl")
    contract = strict_json_loads((graph_dir / "solver_contract.json").read_text())
    selection, report = solve_allocation_rows(records, hyperedges, contract)
    atomic_jsonl(output_path, selection)
    return {
        "process_id": os.getpid(),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "highs_random_seed": int(os.environ.get("HYPERGRAPH_HIGHS_RANDOM_SEED", "0")),
        "selection": file_spec(output_path),
        "solver": report,
    }


def _run_replica(
    root: Path,
    graph_dir: Path,
    output_path: Path,
    seed: str,
    required_environment: Mapping[str, str],
) -> dict[str, Any]:
    environment = {
        **os.environ,
        **dict(required_environment),
        "PYTHONHASHSEED": seed,
        "HYPERGRAPH_HIGHS_RANDOM_SEED": seed,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "allocation_worker.py"),
            "--graph-dir", str(graph_dir),
            "--output", str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=environment,
    )
    if result.returncode != 0:
        raise ProtocolError(
            f"allocation replica seed {seed} failed: stdout={result.stdout!r} stderr={result.stderr!r}"
        )
    if result.stderr != "":
        raise ProtocolError(f"allocation replica seed {seed} emitted stderr: {result.stderr!r}")
    try:
        worker = strict_json_loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ProtocolError("allocation worker stdout contract mismatch") from error
    solver = worker.get("solver") if isinstance(worker, dict) else None
    execution = solver.get("execution") if isinstance(solver, dict) else None
    if (
        not isinstance(worker, dict)
        or set(worker) != {
            "process_id", "python_hash_seed", "highs_random_seed", "selection", "solver",
        }
        or type(worker.get("process_id")) is not int
        or worker["process_id"] <= 0
        or worker.get("python_hash_seed") != seed
        or type(worker.get("highs_random_seed")) is not int
        or worker["highs_random_seed"] != int(seed)
        or not json_exact(worker.get("selection"), file_spec(output_path))
        or not isinstance(execution, dict)
        or execution.get("process_id") != worker["process_id"]
        or execution.get("python_hash_seed") != seed
        or execution.get("highs_random_seed") != int(seed)
    ):
        raise ProtocolError("allocation worker selection binding mismatch")
    return {
        "python_hash_seed": seed,
        "highs_random_seed": int(seed),
        "worker": worker,
        "stderr": result.stderr,
    }


def run_allocation(config_path: Path, root: Path) -> dict[str, Any]:
    config = strict_json_loads(config_path.read_text())
    required_parent_seed = config["deterministic_processes"]["graph_build_python_hash_seed"]
    empty_prefix = Path(config["deterministic_processes"]["empty_python_pycache_prefix"])
    if os.environ.get("PYTHONHASHSEED") != required_parent_seed:
        raise ProtocolError("allocation parent PYTHONHASHSEED mismatch")
    if os.environ.get("PYTHONPYCACHEPREFIX") != str(empty_prefix) or empty_prefix.exists():
        raise ProtocolError("allocation requires the frozen absent pycache prefix")
    observed_runtime = verify_runtime_identity(config["runtime_identity"])
    implementation_start = _implementation(root, config_path)
    graph_validation = validate_graph(config_path, root)
    graph_dir = safe_relative_output(root, config["outputs"]["graph_artifact_dir"])
    deep_binding = verify_deep_receipt(config, root, graph_dir)
    output_dir = safe_relative_output(root, config["outputs"]["allocation_artifact_dir"])
    identity = {"protocol_id": config["protocol_id"], "artifact_kind": "milp_allocation"}
    claim_artifact(output_dir, identity)
    seeds = config["solver"]["replica_python_hash_seeds"]
    if [int(seed) for seed in seeds] != config["solver"]["replica_highs_random_seeds"]:
        raise ProtocolError("Python and HiGHS replica seed pairing mismatch")
    replica_paths = [output_dir / f"selection_seed_{seed}.jsonl" for seed in seeds]
    replicas = [
        _run_replica(
            root,
            graph_dir,
            path,
            seed,
            config["solver"]["required_environment"],
        )
        for seed, path in zip(seeds, replica_paths, strict=True)
    ]
    worker_process_ids = [replica["worker"]["process_id"] for replica in replicas]
    if len(set(worker_process_ids)) != len(worker_process_ids) or os.getpid() in worker_process_ids:
        raise ProtocolError("allocation replicas must be distinct child processes")
    for path in replica_paths:
        load_selection_jsonl(path)
    replica_payloads = [path.read_bytes() for path in replica_paths]
    if len(set(replica_payloads)) != 1:
        raise ProtocolError("allocation selections are not byte-identical across processes/seeds")
    solver_reports = [replica["worker"]["solver"] for replica in replicas]
    if not solver_reports_equivalent(solver_reports[0], solver_reports[1]):
        raise ProtocolError("allocation solver reports differ across replicas")
    allocation_path = output_dir / "allocation.jsonl"
    atomic_write(allocation_path, replica_payloads[0])
    records = load_jsonl(graph_dir / "records.jsonl")
    hyperedges = load_jsonl(graph_dir / "hyperedges.jsonl")
    contract = strict_json_loads((graph_dir / "solver_contract.json").read_text())
    selection = load_selection_jsonl(allocation_path)
    feasible = solver_reports[0]["original_quota_feasible"]
    if type(feasible) is not bool:
        raise ProtocolError("solver feasibility classification must be a JSON boolean")
    verification = verify_selection(
        selection, records, hyperedges, contract, require_exact_quotas=feasible
    )
    if not feasible and selection:
        raise ProtocolError("infeasible original quotas cannot have a selection")
    if feasible is False and (
        solver_reports[0].get("explicit_infeasibility_certificate") is not True
        or solver_reports[0].get("phase_1", {}).get("status") != 2
        or solver_reports[0].get("phase_1", {}).get("classification") != "explicit_infeasible"
    ):
        raise ProtocolError("infeasible publication lacks explicit status-2 certificate")
    graph_validation_end = validate_graph(config_path, root)
    deep_binding_end = verify_deep_receipt(config, root, graph_dir)
    if not json_exact(graph_validation_end, graph_validation) or not json_exact(
        deep_binding_end, deep_binding
    ):
        raise ProtocolError("graph/deep-validation bindings changed during allocation")
    implementation = _implementation(root, config_path)
    if not json_exact(implementation, implementation_start):
        raise ProtocolError("implementation changed during allocation replicas")
    report = {
        "schema_version": 1,
        **identity,
        "status": "original_quota_feasible_allocation_selected" if feasible else "original_quota_infeasible_no_allocation",
        "original_quota_feasible": feasible,
        "graph_validation": graph_validation,
        "graph_PUBLISHED": file_spec(graph_dir / "PUBLISHED.json"),
        "graph_deep_validation": deep_binding,
        "solver_contract": file_spec(graph_dir / "solver_contract.json"),
        "runtime": {
            **observed_runtime,
            "backend": "HiGHS",
            "parent_process_id": os.getpid(),
            "parent_python_hash_seed": required_parent_seed,
            "python_pycache_prefix_absent": str(empty_prefix),
        },
        "replicas": replicas,
        "replica_selection_byte_identical": True,
        "selection": file_spec(allocation_path),
        "solver": solver_reports[0],
        "verification": verification,
        "privacy": {
            "answer_values_available_to_solver_or_allocation": False,
            "model_outcomes_read": False,
            "raw_ids_questions_contexts_answers_or_outcomes_persisted": False,
        },
    }
    atomic_json(output_dir / "allocation_report.json", report)
    manifest = {
        "schema_version": 1,
        **identity,
        "status": "prepared_deterministic_milp_result",
        "implementation": implementation,
        "config": file_spec(config_path),
        "graph_PUBLISHED": file_spec(graph_dir / "PUBLISHED.json"),
        "graph_deep_PUBLISHED": deep_binding["PUBLISHED"],
        "files": {
            "allocation.jsonl": {**file_spec(allocation_path), "rows": len(selection)},
            **{
                path.name: {**file_spec(path), "rows": len(selection)} for path in replica_paths
            },
            "allocation_report.json": file_spec(output_dir / "allocation_report.json"),
        },
        "original_quota_feasible": feasible,
        "completion_marker_required": "PUBLISHED.json",
    }
    atomic_json(output_dir / "manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        **identity,
        "status": "pass_allocation_selected" if feasible else "pass_infeasibility_proven_no_allocation",
        "cpu_only": True,
        "original_quota_feasible": feasible,
        "selected_records": len(selection),
        "replica_selection_byte_identical": True,
        "manifest": file_spec(output_dir / "manifest.json"),
        "required_deep_validation_command": config["commands"]["allocation_deep_validation"],
        "claim_boundary": "direct-hypergraph original-quota MILP result; no model outcomes or locked metrics evaluated",
    }
    atomic_json(output_dir / "receipt.json", receipt)
    if not json_exact(_implementation(root, config_path), implementation):
        raise ProtocolError("implementation changed before allocation completion commit")
    graph_validation_commit = validate_graph(config_path, root)
    deep_binding_commit = verify_deep_receipt(config, root, graph_dir)
    if not json_exact(graph_validation_commit, graph_validation) or not json_exact(
        deep_binding_commit, deep_binding
    ):
        raise ProtocolError("graph/deep bindings changed before allocation completion commit")
    payloads = [
        "allocation.jsonl", *(path.name for path in replica_paths),
        "allocation_report.json", "manifest.json", "receipt.json",
    ]
    marker = finalize_publication(
        output_dir,
        payloads,
        identity=identity,
        status="published_complete_milp_allocation_result",
    )
    return {
        "status": receipt["status"],
        "artifact": str(output_dir),
        "original_quota_feasible": feasible,
        "selected_records": len(selection),
        "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
        "ordered_inventory_sha256": marker["ordered_inventory_sha256"],
    }
