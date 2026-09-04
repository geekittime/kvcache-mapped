from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

from .allocation import (
    _run_replica,
    load_jsonl,
    load_selection_jsonl,
    solver_reports_equivalent,
    verify_deep_receipt,
    verify_selection,
)
from .common import (
    ProtocolError,
    atomic_json,
    claim_artifact,
    file_spec,
    finalize_publication,
    json_exact,
    safe_relative_output,
    strict_json_loads,
    validate_publication,
    verify_runtime_identity,
)
from .validate_graph import validate_graph


_REPORT_KEYS = {
    "schema_version", "protocol_id", "artifact_kind", "status",
    "original_quota_feasible", "graph_validation", "graph_PUBLISHED",
    "graph_deep_validation", "solver_contract", "runtime", "replicas",
    "replica_selection_byte_identical", "selection", "solver", "verification", "privacy",
}
_MANIFEST_KEYS = {
    "schema_version", "protocol_id", "artifact_kind", "status", "implementation",
    "config", "graph_PUBLISHED", "graph_deep_PUBLISHED", "files",
    "original_quota_feasible", "completion_marker_required",
}
_RECEIPT_KEYS = {
    "schema_version", "protocol_id", "artifact_kind", "status", "cpu_only",
    "original_quota_feasible", "selected_records", "replica_selection_byte_identical",
    "manifest", "required_deep_validation_command", "claim_boundary",
}


def _validate_solver_report(
    report: dict[str, Any], config: dict[str, Any], *, feasible: bool, seed: str
) -> None:
    execution = report.get("execution", {})
    expected_execution_keys = {
        "process_id", "python_hash_seed", "highs_random_seed", "thread_environment",
        "highs_options", "runtime_identity",
    }
    expected_options = {
        "disp": False,
        **config["solver"]["highs_options"],
        "random_seed": int(seed),
    }
    if (
        type(report.get("original_quota_feasible")) is not bool
        or report["original_quota_feasible"] is not feasible
        or type(report.get("explicit_infeasibility_certificate")) is not bool
        or type(report.get("phase_2_run")) is not bool
        or type(report.get("variable_count")) is not int
        or report["variable_count"] <= 0
        or type(report.get("constraint_count")) is not int
        or report["constraint_count"] <= 0
        or set(execution) != expected_execution_keys
        or type(execution.get("process_id")) is not int
        or execution["process_id"] <= 0
        or execution.get("python_hash_seed") != seed
        or type(execution.get("highs_random_seed")) is not int
        or execution["highs_random_seed"] != int(seed)
        or not json_exact(execution.get("thread_environment"), config["solver"]["required_environment"])
        or not json_exact(execution.get("highs_options"), expected_options)
        or not json_exact(execution.get("runtime_identity"), config["runtime_identity"])
    ):
        raise ProtocolError("solver report runtime/typed schema mismatch")
    phase_1 = report.get("phase_1", {})
    if (
        set(phase_1) != {"status", "success", "message", "classification"}
        or type(phase_1.get("status")) is not int
        or type(phase_1.get("success")) is not bool
        or not isinstance(phase_1.get("message"), str)
    ):
        raise ProtocolError("phase_1 report typed schema mismatch")
    if feasible:
        phase_2 = report.get("phase_2", {})
        if (
            report.get("explicit_infeasibility_certificate") is not False
            or report.get("phase_2_run") is not True
            or phase_1.get("status") != 0
            or phase_1.get("success") is not True
            or phase_1.get("classification") != "optimal_feasible"
            or set(report) != {
                "original_quota_feasible", "explicit_infeasibility_certificate", "phase_1",
                "phase_2_run", "phase_2", "variable_count", "constraint_count",
                "verification", "execution",
            }
            or set(phase_2) != {
                "status", "success", "message", "classification", "objective",
                "selected_objective_numerator", "objective_denominator", "binary_tolerance",
            }
            or type(phase_2.get("status")) is not int
            or phase_2["status"] != 0
            or phase_2.get("success") is not True
            or phase_2.get("classification") != "proven_optimal"
            or isinstance(phase_2.get("objective"), bool)
            or not isinstance(phase_2.get("objective"), float)
            or not math.isfinite(phase_2["objective"])
            or type(phase_2.get("selected_objective_numerator")) is not int
            or phase_2["selected_objective_numerator"] < 0
            or type(phase_2.get("objective_denominator")) is not int
            or phase_2["objective_denominator"] != 16 ** 13
            or not isinstance(phase_2.get("binary_tolerance"), float)
            or phase_2["binary_tolerance"] != 1e-7
        ):
            raise ProtocolError("feasible solver proof schema mismatch")
    elif (
        set(report) != {
            "original_quota_feasible", "explicit_infeasibility_certificate", "phase_1",
            "phase_2_run", "variable_count", "constraint_count", "execution",
        }
        or report.get("explicit_infeasibility_certificate") is not True
        or report.get("phase_2_run") is not False
        or phase_1.get("status") != 2
        or phase_1.get("success") is not False
        or phase_1.get("classification") != "explicit_infeasible"
        or not phase_1["message"].startswith("The problem is infeasible.")
        or "model_status is Infeasible" not in phase_1["message"]
    ):
        raise ProtocolError("infeasible solver proof is not an explicit status-2 certificate")


def _write_allocation_deep_receipt(
    config: dict[str, Any],
    root: Path,
    allocation_dir: Path,
    graph_deep_result: dict[str, Any],
    published_solver: dict[str, Any],
    published_worker_pids: list[int],
    deep_replicas: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir = safe_relative_output(root, config["outputs"]["allocation_deep_receipt"])
    identity = {"protocol_id": config["protocol_id"], "artifact_kind": "allocation_deep_validation"}
    claim_artifact(output_dir, identity)
    replica_evidence = []
    for value in deep_replicas:
        worker = value["worker"]
        replica_evidence.append({
            "python_hash_seed": value["python_hash_seed"],
            "highs_random_seed": value["highs_random_seed"],
            "process_id": worker["process_id"],
            "selection": {
                "nbytes": worker["selection"]["nbytes"],
                "sha256": worker["selection"]["sha256"],
            },
            "solver_report": worker["solver"],
        })
    receipt = {
        "schema_version": 1,
        **identity,
        "status": "pass_raw_graph_rebuild_and_two_new_solver_replicas",
        "cpu_only": True,
        "runtime_identity": config["runtime_identity"],
        "deep_validation_process": {
            "process_id": os.getpid(),
            "python_hash_seed": config["deterministic_processes"]["graph_deep_python_hash_seed"],
        },
        "allocation_PUBLISHED": file_spec(allocation_dir / "PUBLISHED.json"),
        "allocation_report": file_spec(allocation_dir / "allocation_report.json"),
        "graph_recomputed_from_pinned_raw_sources": True,
        "graph_rebuild_validation": graph_deep_result,
        "published_worker_process_ids": published_worker_pids,
        "deep_replica_evidence": replica_evidence,
        "deep_worker_processes_distinct_from_publication_and_validator": True,
        "selection_byte_identical_to_publication": True,
        "solver_reports_semantically_identical_to_published_except_declared_replica_runtime": True,
        "published_solver_report": published_solver,
        "claim_boundary": "deep graph/allocation validation only; no model outcomes or locked metrics evaluated",
    }
    atomic_json(output_dir / "receipt.json", receipt)
    marker = finalize_publication(
        output_dir,
        ["receipt.json"],
        identity=identity,
        status="published_complete_allocation_deep_validation",
    )
    return {
        "artifact": str(output_dir),
        "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
        "ordered_inventory_sha256": marker["ordered_inventory_sha256"],
    }


def validate_allocation(
    config_path: Path,
    root: Path,
    *,
    deep: bool = False,
) -> dict[str, Any]:
    config = strict_json_loads(config_path.read_text())
    verify_runtime_identity(config["runtime_identity"])
    if deep:
        required_seed = config["deterministic_processes"]["graph_deep_python_hash_seed"]
        empty_prefix = Path(config["deterministic_processes"]["empty_python_pycache_prefix"])
        if os.environ.get("PYTHONHASHSEED") != required_seed:
            raise ProtocolError("deep allocation validation PYTHONHASHSEED mismatch")
        if os.environ.get("PYTHONPYCACHEPREFIX") != str(empty_prefix) or empty_prefix.exists():
            raise ProtocolError("deep allocation validation requires the frozen absent pycache prefix")
    graph_result = validate_graph(config_path, root)
    graph_dir = safe_relative_output(root, config["outputs"]["graph_artifact_dir"])
    deep_binding = verify_deep_receipt(config, root, graph_dir)
    output_dir = safe_relative_output(root, config["outputs"]["allocation_artifact_dir"])
    seeds = config["solver"]["replica_python_hash_seeds"]
    replica_names = [f"selection_seed_{seed}.jsonl" for seed in seeds]
    payloads = [
        "allocation.jsonl", *replica_names,
        "allocation_report.json", "manifest.json", "receipt.json",
    ]
    validate_publication(
        output_dir,
        payloads,
        identity={"protocol_id": config["protocol_id"], "artifact_kind": "milp_allocation"},
        status="published_complete_milp_allocation_result",
    )
    allocation_bindings_start = {
        "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
        "allocation": file_spec(output_dir / "allocation.jsonl"),
        "report": file_spec(output_dir / "allocation_report.json"),
        "manifest": file_spec(output_dir / "manifest.json"),
        "receipt": file_spec(output_dir / "receipt.json"),
    }
    allocation_path = output_dir / "allocation.jsonl"
    allocation_bytes = allocation_path.read_bytes()
    if any((output_dir / name).read_bytes() != allocation_bytes for name in replica_names):
        raise ProtocolError("published allocation replicas are not byte-identical")
    selection = load_selection_jsonl(allocation_path)
    records = load_jsonl(graph_dir / "records.jsonl")
    hyperedges = load_jsonl(graph_dir / "hyperedges.jsonl")
    contract = strict_json_loads((graph_dir / "solver_contract.json").read_text())
    report = strict_json_loads((output_dir / "allocation_report.json").read_text())
    manifest = strict_json_loads((output_dir / "manifest.json").read_text())
    receipt = strict_json_loads((output_dir / "receipt.json").read_text())
    feasible = report.get("original_quota_feasible")
    if type(feasible) is not bool:
        raise ProtocolError("allocation feasibility result missing")
    verification = verify_selection(
        selection, records, hyperedges, contract, require_exact_quotas=feasible
    )
    if not feasible and selection:
        raise ProtocolError("infeasible allocation must be empty")
    replicas = report.get("replicas")
    seeds = config["solver"]["replica_python_hash_seeds"]
    if not isinstance(replicas, list) or len(replicas) != len(seeds):
        raise ProtocolError("allocation replica evidence count mismatch")
    solver_reports = []
    published_worker_pids = []
    for replica, seed, name in zip(replicas, seeds, replica_names, strict=True):
        worker = replica.get("worker") if isinstance(replica, dict) else None
        if (
            not isinstance(replica, dict)
            or set(replica) != {"python_hash_seed", "highs_random_seed", "worker", "stderr"}
            or replica.get("python_hash_seed") != seed
            or type(replica.get("highs_random_seed")) is not int
            or replica["highs_random_seed"] != int(seed)
            or replica.get("stderr") != ""
            or not isinstance(worker, dict)
            or set(worker) != {
                "process_id", "python_hash_seed", "highs_random_seed", "selection", "solver",
            }
            or type(worker.get("process_id")) is not int
            or worker["process_id"] <= 0
            or worker.get("python_hash_seed") != seed
            or type(worker.get("highs_random_seed")) is not int
            or worker["highs_random_seed"] != int(seed)
            or not json_exact(worker.get("selection"), file_spec(output_dir / name))
            or not isinstance(worker.get("solver"), dict)
        ):
            raise ProtocolError("allocation replica evidence semantics mismatch")
        _validate_solver_report(worker["solver"], config, feasible=feasible, seed=seed)
        solver_reports.append(worker["solver"])
        published_worker_pids.append(worker["process_id"])
    if (
        len(set(published_worker_pids)) != len(published_worker_pids)
        or not solver_reports_equivalent(solver_reports[0], solver_reports[1])
        or not json_exact(report.get("solver"), solver_reports[0])
    ):
        raise ProtocolError("allocation replica solver reports mismatch")
    if feasible:
        objective_numerator = sum(
            int(row["objective_cost_sha256"][:13], 16) for row in selection
        )
        expected_objective = objective_numerator / float(16 ** 13)
        for solver_report in solver_reports:
            phase_2 = solver_report["phase_2"]
            if (
                phase_2["selected_objective_numerator"] != objective_numerator
                or abs(phase_2["objective"] - expected_objective) > 1e-10
                or not json_exact(solver_report.get("verification"), verification)
            ):
                raise ProtocolError("published phase_2 objective/incumbent does not recompute")
    if (
        set(report) != _REPORT_KEYS
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("protocol_id") != config["protocol_id"]
        or report.get("artifact_kind") != "milp_allocation"
        or report.get("status") != (
            "original_quota_feasible_allocation_selected" if feasible
            else "original_quota_infeasible_no_allocation"
        )
        or not json_exact(report.get("graph_validation"), graph_result)
        or not json_exact(report.get("graph_PUBLISHED"), file_spec(graph_dir / "PUBLISHED.json"))
        or not json_exact(report.get("graph_deep_validation"), deep_binding)
        or not json_exact(report.get("solver_contract"), file_spec(graph_dir / "solver_contract.json"))
        or report.get("replica_selection_byte_identical") is not True
        or not json_exact(report.get("selection"), file_spec(allocation_path))
        or not json_exact(report.get("verification"), verification)
        or not json_exact(report.get("privacy"), {
            "answer_values_available_to_solver_or_allocation": False,
            "model_outcomes_read": False,
            "raw_ids_questions_contexts_answers_or_outcomes_persisted": False,
        })
    ):
        raise ProtocolError("allocation report semantics mismatch")
    runtime = report.get("runtime", {})
    if (
        set(runtime) != {
            *config["runtime_identity"], "backend", "parent_process_id",
            "parent_python_hash_seed", "python_pycache_prefix_absent",
        }
        or not json_exact(
            {key: runtime[key] for key in config["runtime_identity"]},
            config["runtime_identity"],
        )
        or runtime.get("backend") != "HiGHS"
        or type(runtime.get("parent_process_id")) is not int
        or runtime["parent_process_id"] <= 0
        or runtime.get("parent_python_hash_seed") != config["deterministic_processes"]["graph_build_python_hash_seed"]
        or runtime.get("python_pycache_prefix_absent") != config["deterministic_processes"]["empty_python_pycache_prefix"]
        or runtime["parent_process_id"] in published_worker_pids
    ):
        raise ProtocolError("allocation runtime/process semantics mismatch")
    graph_manifest = strict_json_loads((graph_dir / "manifest.json").read_text())
    if (
        set(manifest) != _MANIFEST_KEYS
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != config["protocol_id"]
        or manifest.get("artifact_kind") != "milp_allocation"
        or manifest.get("status") != "prepared_deterministic_milp_result"
        or not json_exact(manifest.get("implementation"), graph_manifest["implementation"])
        or not json_exact(manifest.get("config"), file_spec(config_path))
        or not json_exact(manifest.get("graph_PUBLISHED"), file_spec(graph_dir / "PUBLISHED.json"))
        or not json_exact(manifest.get("graph_deep_PUBLISHED"), deep_binding["PUBLISHED"])
        or manifest.get("original_quota_feasible") is not feasible
        or manifest.get("completion_marker_required") != "PUBLISHED.json"
    ):
        raise ProtocolError("allocation manifest semantics mismatch")
    expected_files = {
        "allocation.jsonl": {**file_spec(allocation_path), "rows": len(selection)},
        **{
            name: {**file_spec(output_dir / name), "rows": len(selection)} for name in replica_names
        },
        "allocation_report.json": file_spec(output_dir / "allocation_report.json"),
    }
    if not json_exact(manifest.get("files"), expected_files):
        raise ProtocolError("allocation manifest file bindings mismatch")
    if (
        set(receipt) != _RECEIPT_KEYS
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != config["protocol_id"]
        or receipt.get("artifact_kind") != "milp_allocation"
        or receipt.get("status") != (
            "pass_allocation_selected" if feasible else "pass_infeasibility_proven_no_allocation"
        )
        or receipt.get("cpu_only") is not True
        or receipt.get("original_quota_feasible") is not feasible
        or type(receipt.get("selected_records")) is not int
        or receipt.get("selected_records") != len(selection)
        or receipt.get("replica_selection_byte_identical") is not True
        or not json_exact(receipt.get("manifest"), file_spec(output_dir / "manifest.json"))
        or receipt.get("required_deep_validation_command") != config["commands"]["allocation_deep_validation"]
        or receipt.get("claim_boundary") != "direct-hypergraph original-quota MILP result; no model outcomes or locked metrics evaluated"
    ):
        raise ProtocolError("allocation receipt semantics mismatch")
    deep_result = {"performed": False}
    if deep:
        # This is deliberately a new pinned-raw-source graph rebuild, not just
        # a solver replay over trusted published graph rows.
        graph_deep_result = validate_graph(
            config_path, root, deep=True, write_receipt=False
        )
        with tempfile.TemporaryDirectory(prefix="hypergraph_v2_allocation_deep_") as temporary:
            temporary_root = Path(temporary)
            deep_replicas = []
            for seed in seeds:
                output = temporary_root / f"selection_seed_{seed}.jsonl"
                replica = _run_replica(
                    root,
                    graph_dir,
                    output,
                    seed,
                    config["solver"]["required_environment"],
                )
                load_selection_jsonl(output)
                if output.read_bytes() != allocation_bytes:
                    raise ProtocolError("deep allocation replica differs from publication")
                deep_replicas.append(replica)
            deep_solver_reports = [value["worker"]["solver"] for value in deep_replicas]
            deep_worker_pids = [value["worker"]["process_id"] for value in deep_replicas]
            forbidden_pids = {
                os.getpid(), report["runtime"]["parent_process_id"], *published_worker_pids,
            }
            if (
                len(set(deep_worker_pids)) != len(deep_worker_pids)
                or set(deep_worker_pids) & forbidden_pids
            ):
                raise ProtocolError("deep solver workers are not new distinct processes")
            for solver_report, seed in zip(deep_solver_reports, seeds, strict=True):
                _validate_solver_report(solver_report, config, feasible=feasible, seed=seed)
                if not solver_reports_equivalent(solver_report, report["solver"]):
                    raise ProtocolError("deep solver report differs from published solver proof")
            if not solver_reports_equivalent(deep_solver_reports[0], deep_solver_reports[1]):
                raise ProtocolError("deep allocation solver reports differ across processes")
            # Close the graph/allocation race window after both new workers.
            graph_end = validate_graph(config_path, root)
            deep_binding_end = verify_deep_receipt(config, root, graph_dir)
            validate_publication(
                output_dir,
                payloads,
                identity={"protocol_id": config["protocol_id"], "artifact_kind": "milp_allocation"},
                status="published_complete_milp_allocation_result",
            )
            allocation_bindings_end = {
                "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
                "allocation": file_spec(output_dir / "allocation.jsonl"),
                "report": file_spec(output_dir / "allocation_report.json"),
                "manifest": file_spec(output_dir / "manifest.json"),
                "receipt": file_spec(output_dir / "receipt.json"),
            }
            if not json_exact(graph_end, graph_result) or not json_exact(
                deep_binding_end, deep_binding
            ) or not json_exact(allocation_bindings_end, allocation_bindings_start):
                raise ProtocolError("graph/deep receipt changed during deep allocation validation")
            written_receipt = _write_allocation_deep_receipt(
                config,
                root,
                output_dir,
                graph_deep_result,
                report["solver"],
                published_worker_pids,
                deep_replicas,
            )
        deep_result = {
            "performed": True,
            "pinned_raw_graph_rebuilt": True,
            "graph_payloads_byte_identical": True,
            "two_new_solver_processes": True,
            "selection_byte_identical_to_publication": True,
            "replica_solver_reports": deep_solver_reports,
            "deep_worker_process_ids": deep_worker_pids,
            "receipt": written_receipt,
        }
    return {
        "status": "pass",
        "artifact": str(output_dir),
        "original_quota_feasible": feasible,
        "selected_records": len(selection),
        "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
        "graph_validation": graph_result,
        "deep_allocation_validation": deep_result,
    }
