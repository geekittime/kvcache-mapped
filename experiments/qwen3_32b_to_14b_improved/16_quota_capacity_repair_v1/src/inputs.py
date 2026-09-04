"""Read-only binding and privacy checks for the immutable pairwise-v3 inputs."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .common import (
    ProtocolError,
    canonical_bytes,
    file_spec,
    json_exact,
    require_directory,
    require_exact_keys,
    require_regular_file,
    safe_child,
    sha256_file,
    strict_json_load,
    strict_json_loads,
    validate_publication,
    verify_file_spec,
)


RECORD_KEYS = {
    "record_index",
    "dataset_id",
    "stratum",
    "source_index",
    "source_projection_sha256",
}
HYPEREDGE_KEYS = {
    "hyperedge_id",
    "member_record_indices",
    "member_count",
    "relation_kinds",
    "derivation_count_by_kind",
}
HEX64 = set("0123456789abcdef")


@dataclass(frozen=True)
class InputBundle:
    """Validated graph rows and immutable source receipts.

    The object intentionally contains no answer, support, model-output, or
    metric fields.  It is the only input accepted by quota/allocation code.
    """

    config: Mapping[str, Any]
    graph_dir: Path
    records: tuple[dict[str, Any], ...]
    hyperedges: tuple[dict[str, Any], ...]
    contract: Mapping[str, Any]
    graph_report: Mapping[str, Any]
    bindings: Mapping[str, Any]
    allocation_deep_ready: bool
    allocation_deep_binding: Mapping[str, Any]


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    require_regular_file(path)
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.endswith(b"\n"):
                raise ProtocolError(f"JSONL line {line_number} lacks terminal newline: {path}")
            value = strict_json_loads(raw)
            if not isinstance(value, dict):
                raise ProtocolError(f"JSONL line {line_number} is not an object: {path}")
            yield value


def _validate_external_marker(
    directory: Path,
    *,
    protocol_id: str,
    artifact_kind: str,
    status: str,
) -> dict[str, Any]:
    """Validate a v3 PUBLISHED marker without writing to its directory."""

    require_directory(directory)
    marker_path = directory / "PUBLISHED.json"
    require_regular_file(marker_path)
    marker = strict_json_load(marker_path)
    if not isinstance(marker, dict):
        raise ProtocolError(f"external marker is not an object: {marker_path}")
    expected = {
        "schema_version",
        "artifact_kind",
        "atomic_last_completion_marker",
        "ordered_inventory_sha256",
        "protocol_id",
        "published_files",
        "receipt",
        "receipt_sidecar",
        "status",
    }
    if (
        set(marker) != expected
        or type(marker.get("schema_version")) is not int
        or marker.get("schema_version") != 1
        or marker.get("protocol_id") != protocol_id
        or marker.get("artifact_kind") != artifact_kind
        or marker.get("status") != status
        or marker.get("atomic_last_completion_marker") is not True
        or not isinstance(marker.get("published_files"), dict)
    ):
        raise ProtocolError(f"external PUBLISHED marker semantics mismatch: {marker_path}")
    published = marker["published_files"]
    names = sorted(published)
    if "receipt.json" not in published or "receipt.json.sha256.json" not in published:
        raise ProtocolError(f"external marker has no receipt binding: {marker_path}")
    expected_names = sorted([*published])
    observed_names = sorted(p.name for p in directory.iterdir() if p.name != "PUBLISHED.json")
    if observed_names != expected_names:
        raise ProtocolError(f"external marker inventory mismatch: {directory}")
    observed: dict[str, dict[str, Any]] = {}
    for name in names:
        if Path(name).name != name:
            raise ProtocolError(f"external marker contains non-basename: {name}")
        path = directory / name
        require_regular_file(path)
        spec = published[name]
        if not isinstance(spec, dict) or set(spec) != {"nbytes", "sha256"}:
            raise ProtocolError(f"external marker file spec mismatch: {name}")
        observed[name] = file_spec(path, include_path=False)
        if not json_exact(observed[name], spec):
            raise ProtocolError(f"external marker file digest mismatch: {name}")
    ordered = [{"name": name, **observed[name]} for name in names]
    expected_inventory = hashlib.sha256(canonical_bytes(ordered)).hexdigest()
    if marker.get("ordered_inventory_sha256") != expected_inventory:
        raise ProtocolError(f"external marker ordered inventory mismatch: {directory}")
    if not json_exact(marker.get("receipt"), observed["receipt.json"]):
        raise ProtocolError("external receipt binding mismatch")
    if not json_exact(marker.get("receipt_sidecar"), observed["receipt.json.sha256.json"]):
        raise ProtocolError("external receipt sidecar binding mismatch")
    for name in names:
        if not name.endswith(".sha256.json"):
            continue
        sidecar = strict_json_load(directory / name)
        payload_name = name[: -len(".sha256.json")]
        if payload_name not in published:
            raise ProtocolError(f"orphan external sidecar: {name}")
        expected_sidecar = {
            "path": str(directory / payload_name),
            **observed[payload_name],
        }
        if not json_exact(sidecar, expected_sidecar):
            raise ProtocolError(f"external sidecar semantic mismatch: {name}")
    return marker


def _validate_record_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    previous_projection = ""
    for expected_index, row in enumerate(_iter_jsonl(path)):
        if (
            set(row) != RECORD_KEYS
            or type(row.get("record_index")) is not int
            or row["record_index"] != expected_index
            or not isinstance(row.get("dataset_id"), str)
            or not isinstance(row.get("stratum"), str)
            or type(row.get("source_index")) is not int
            or row["source_index"] < 0
            or not _is_hex64(row.get("source_projection_sha256"))
            or row["source_projection_sha256"] <= previous_projection
        ):
            raise ProtocolError(f"records typed/order schema mismatch at row {expected_index}")
        previous_projection = row["source_projection_sha256"]
        rows.append(row)
    if not rows:
        raise ProtocolError("records artifact is empty")
    return tuple(rows)


def _validate_hyperedge_rows(path: Path, record_count: int) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    previous_id = ""
    for row_number, edge in enumerate(_iter_jsonl(path), 1):
        members = edge.get("member_record_indices")
        kinds = edge.get("relation_kinds")
        derivations = edge.get("derivation_count_by_kind")
        if (
            set(edge) != HYPEREDGE_KEYS
            or not _is_hex64(edge.get("hyperedge_id"))
            or edge["hyperedge_id"] <= previous_id
            or not isinstance(members, list)
            or len(members) < 2
            or any(type(i) is not int for i in members)
            or members != sorted(set(members))
            or members[0] < 0
            or members[-1] >= record_count
            or type(edge.get("member_count")) is not int
            or edge["member_count"] != len(members)
            or not isinstance(kinds, list)
            or kinds != sorted(set(kinds))
            or any(not isinstance(k, str) for k in kinds)
            or not isinstance(derivations, dict)
            or set(derivations) != set(kinds)
            or any(type(value) is not int or value <= 0 for value in derivations.values())
        ):
            raise ProtocolError(f"hyperedges typed/order schema mismatch at row {row_number}")
        previous_id = edge["hyperedge_id"]
        rows.append(edge)
    if not rows:
        raise ProtocolError("hyperedges artifact is empty")
    return tuple(rows)


def _validate_no_forbidden_fields(value: Any, forbidden: set[str], location: str = "$", *, allow_text_values: bool = True) -> None:
    """Reject forbidden *keys* recursively; values are never inspected as data."""

    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                raise ProtocolError(f"forbidden data field {key!r} at {location}")
            _validate_no_forbidden_fields(child, forbidden, f"{location}.{key}", allow_text_values=allow_text_values)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_forbidden_fields(child, forbidden, f"{location}[{index}]", allow_text_values=allow_text_values)


def _validate_allocation_infeasible(
    allocation_dir: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    protocol = "pairwise_direct_collision_hypergraph_feasibility_v3"
    marker = _validate_external_marker(
        allocation_dir,
        protocol_id=protocol,
        artifact_kind="milp_allocation",
        status="published_complete_milp_allocation_result",
    )
    receipt = strict_json_load(allocation_dir / "receipt.json")
    if not isinstance(receipt, dict):
        raise ProtocolError("allocation receipt is not an object")
    expected_receipt_keys = {
        "artifact_kind", "claim_boundary", "cpu_only", "manifest",
        "original_quota_feasible", "protocol_id", "replica_selection_byte_identical",
        "required_deep_validation_command", "schema_version", "selected_records", "status",
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("artifact_kind") != "milp_allocation"
        or receipt.get("protocol_id") != protocol
        or receipt.get("status") != "pass_infeasibility_proven_no_allocation"
        or receipt.get("cpu_only") is not True
        or receipt.get("original_quota_feasible") is not False
        or type(receipt.get("selected_records")) is not int
        or receipt.get("selected_records") != 0
        or receipt.get("replica_selection_byte_identical") is not True
        or not isinstance(receipt.get("required_deep_validation_command"), str)
    ):
        raise ProtocolError("allocation infeasibility receipt typed/status mismatch")
    _validate_no_forbidden_fields(receipt, set(config["forbidden_data"]["forbidden_field_names"]))
    report = strict_json_load(allocation_dir / "allocation_report.json")
    if not isinstance(report, dict):
        raise ProtocolError("allocation report is not an object")
    selection_ref = report.get("selection")
    if (
        report.get("protocol_id") != protocol
        or report.get("artifact_kind") != "milp_allocation"
        or report.get("original_quota_feasible") is not False
        or report.get("status") != "original_quota_infeasible_no_allocation"
        or report.get("replica_selection_byte_identical") is not True
        or not isinstance(selection_ref, dict)
        or set(selection_ref) != {"nbytes", "path", "sha256"}
        or type(selection_ref.get("nbytes")) is not int
        or selection_ref["nbytes"] != 0
        or selection_ref.get("path") != str(allocation_dir / "allocation.jsonl")
        or selection_ref.get("sha256") != hashlib.sha256(b"").hexdigest()
    ):
        raise ProtocolError("allocation report infeasibility binding mismatch")
    replicas = report.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 2:
        raise ProtocolError("allocation report must contain exactly two replicas")
    pids: list[int] = []
    seeds: list[int] = []
    hash_seeds: list[str] = []
    for replica in replicas:
        if not isinstance(replica, dict) or set(replica) != {"highs_random_seed", "python_hash_seed", "worker", "stderr"}:
            raise ProtocolError("allocation replica typed schema mismatch")
        if replica.get("stderr") != "":
            raise ProtocolError("allocation replica emitted stderr")
        hs = replica.get("highs_random_seed")
        ps = replica.get("python_hash_seed")
        worker = replica.get("worker")
        if type(hs) is not int or not isinstance(ps, str) or not isinstance(worker, dict):
            raise ProtocolError("allocation replica seed type mismatch")
        if ps not in {"1", "777"} or hs != int(ps):
            raise ProtocolError("allocation replica seed pairing mismatch")
        pids.append(worker.get("process_id", -1))
        seeds.append(hs)
        hash_seeds.append(ps)
        if type(worker.get("process_id")) is not int or worker["process_id"] <= 0:
            raise ProtocolError("allocation worker PID is invalid")
        if set(worker) != {"process_id", "python_hash_seed", "highs_random_seed", "selection", "solver"} or worker.get("python_hash_seed") != ps or type(worker.get("highs_random_seed")) is not int or worker["highs_random_seed"] != hs:
            raise ProtocolError("allocation worker runtime/seed schema mismatch")
        worker_selection = worker.get("selection")
        if not isinstance(worker_selection, dict) or set(worker_selection) != {"nbytes", "path", "sha256"} or type(worker_selection.get("nbytes")) is not int or worker_selection["nbytes"] != 0 or worker_selection.get("path") != str(allocation_dir / f"selection_seed_{ps}.jsonl") or worker_selection.get("sha256") != hashlib.sha256(b"").hexdigest():
            raise ProtocolError("allocation worker empty-selection binding mismatch")
        solver = worker.get("solver")
        if not isinstance(solver, dict) or set(solver) != {
            "constraint_count", "execution", "explicit_infeasibility_certificate",
            "original_quota_feasible", "phase_1", "phase_2_run", "variable_count",
        }:
            raise ProtocolError("allocation worker solver report missing")
        if type(solver.get("constraint_count")) is not int or solver["constraint_count"] <= 0 or type(solver.get("variable_count")) is not int or solver["variable_count"] <= 0 or type(solver.get("explicit_infeasibility_certificate")) is not bool or solver["explicit_infeasibility_certificate"] is not True or type(solver.get("phase_2_run")) is not bool or solver["phase_2_run"] is not False or type(solver.get("original_quota_feasible")) is not bool or solver["original_quota_feasible"] is not False:
            raise ProtocolError("allocation worker solver integer/boolean fields mismatch")
        execution = solver.get("execution")
        if not isinstance(execution, dict) or set(execution) != {"process_id", "python_hash_seed", "highs_random_seed", "thread_environment", "highs_options", "runtime_identity"} or type(execution.get("process_id")) is not int or execution["process_id"] != worker["process_id"] or execution.get("python_hash_seed") != ps or type(execution.get("highs_random_seed")) is not int or execution["highs_random_seed"] != hs or execution.get("thread_environment") != config["solver"]["required_environment"]:
            raise ProtocolError("allocation worker execution binding mismatch")
        options = execution.get("highs_options")
        expected_options = {"disp": False, **dict(config["solver"]["highs_options"]), "random_seed": hs}
        if not json_exact(options, expected_options):
            raise ProtocolError("allocation worker HiGHS options mismatch")
        runtime_identity = execution.get("runtime_identity")
        if not isinstance(runtime_identity, dict) or not isinstance(runtime_identity.get("python_executable"), str) or not isinstance(runtime_identity.get("python_version"), str) or not isinstance(runtime_identity.get("scipy_version"), str) or runtime_identity["scipy_version"] != config["solver"]["scipy_version"]:
            raise ProtocolError("allocation worker runtime identity mismatch")
        phase = solver.get("phase_1")
        if not isinstance(phase, dict) or set(phase) != {"status", "success", "message", "classification"} or (
            type(phase.get("status")) is not int
            or type(phase.get("success")) is not bool
            or not isinstance(phase.get("message"), str)
            or phase.get("status") != 2
            or phase.get("success") is not False
            or not phase["message"].startswith("The problem is infeasible.")
        ):
            raise ProtocolError("allocation phase-1 status is not strict explicit infeasible")
        if solver.get("phase_2_run") is not False or solver.get("original_quota_feasible") is not False:
            raise ProtocolError("allocation solver report status mismatch")
        if not json_exact(execution.get("process_id"), worker["process_id"]):
            raise ProtocolError("allocation execution process binding mismatch")
    if len(set(pids)) != 2 or sorted(seeds) != [1, 777] or sorted(hash_seeds) != ["1", "777"]:
        raise ProtocolError("allocation replicas are not two distinct required seeds")
    if not json_exact(report.get("solver"), report.get("replicas")[0]["worker"]["solver"]):
        raise ProtocolError("allocation top-level solver report does not bind replica")
    for name in ("allocation.jsonl", "selection_seed_1.jsonl", "selection_seed_777.jsonl"):
        require_regular_file(allocation_dir / name)
        if (allocation_dir / name).read_bytes() != b"":
            raise ProtocolError(f"infeasible allocation unexpectedly contains rows: {name}")
    _validate_no_forbidden_fields(report, set(config["forbidden_data"]["forbidden_field_names"]))
    return {
        "PUBLISHED": file_spec(allocation_dir / "PUBLISHED.json"),
        "receipt": file_spec(allocation_dir / "receipt.json"),
        "report": file_spec(allocation_dir / "allocation_report.json"),
        "marker": marker,
        "replica_process_ids": pids,
        "replica_seeds": seeds,
    }


def _validate_deep_allocation_placeholder(config: Mapping[str, Any], source_root: Path) -> tuple[bool, dict[str, Any]]:
    placeholder = config["source_bindings"]["allocation_deep_receipt_placeholder"]
    if not isinstance(placeholder, dict):
        raise ProtocolError("allocation deep placeholder binding missing")
    path = Path(placeholder.get("path", ""))
    if not path.is_absolute() or not path.is_relative_to(source_root):
        raise ProtocolError("allocation deep receipt path is not under pinned source root")
    # Check every existing component without resolving a symlink.  The final
    # directory is intentionally allowed to be absent while the v3 deep worker
    # is running; this is the sole placeholder state accepted pre-formal.
    relative = path.relative_to(source_root)
    safe_child(source_root, str(relative), allow_missing=True)
    # Resolve only after checking components; an absent leaf is the expected
    # pre-formal state and must not be created by this consumer.
    if not path.exists():
        return False, {
            "ready": False,
            "placeholder": True,
            "status": "PLACEHOLDER_MISSING",
            "path": str(path),
            "reason": "allocation_v3_deep_receipt_not_yet_published",
        }
    require_directory(path)
    marker_path = path / "PUBLISHED.json"
    if not marker_path.exists():
        return False, {
            "ready": False,
            "placeholder": True,
            "status": "PLACEHOLDER_IN_PROGRESS",
            "path": str(path),
            "reason": "allocation_v3_deep_receipt_directory_exists_without_PUBLISHED",
        }
    marker = _validate_external_marker(
        path,
        protocol_id="pairwise_direct_collision_hypergraph_feasibility_v3",
        artifact_kind="allocation_deep_validation",
        status="published_complete_allocation_deep_validation",
    )
    # A ready-gate snapshot may pin all three deep payload bytes explicitly.
    # Older placeholder configs intentionally omit these optional specs; when
    # present, every one is checked before the gate can be considered ready.
    expected_published = placeholder.get("published")
    expected_receipt = placeholder.get("receipt")
    expected_sidecar = placeholder.get("receipt_sidecar")
    if any(value is not None for value in (expected_published, expected_receipt, expected_sidecar)):
        if not all(isinstance(value, dict) for value in (expected_published, expected_receipt, expected_sidecar)):
            raise ProtocolError("allocation deep ready snapshot requires all three file specs")
        verify_file_spec(marker_path, expected_published, name="allocation_deep_published")
        verify_file_spec(path / "receipt.json", expected_receipt, name="allocation_deep_receipt")
        verify_file_spec(path / "receipt.json.sha256.json", expected_sidecar, name="allocation_deep_receipt_sidecar")
    receipt = strict_json_load(path / "receipt.json")
    if not isinstance(receipt, dict) or receipt.get("status") != "pass_raw_graph_rebuild_and_two_new_solver_replicas" or receipt.get("cpu_only") is not True:
        raise ProtocolError("allocation_v3 deep receipt status/cpu gate mismatch")
    if receipt.get("selection_byte_identical_to_publication") is not True or receipt.get("solver_reports_semantically_identical_to_published_except_declared_replica_runtime") is not True:
        raise ProtocolError("allocation_v3 deep receipt identity gate mismatch")
    _validate_no_forbidden_fields(receipt, set(config["forbidden_data"]["forbidden_field_names"]))
    return True, {
        "ready": True,
        "placeholder": False,
        "status": "READY",
        "path": str(path),
        "PUBLISHED": file_spec(marker_path),
        "receipt": file_spec(path / "receipt.json"),
        "receipt_sidecar": file_spec(path / "receipt.json.sha256.json"),
        "marker": marker,
    }


def load_input_bundle(config: Mapping[str, Any], *, require_deep_allocation: bool = False) -> InputBundle:
    """Validate and load all answer-free v3 input artifacts, read-only."""

    if config.get("protocol_id") != "quota_capacity_repair_v1":
        raise ProtocolError("quota-repair config protocol_id mismatch")
    source_root = Path(config.get("source_root", ""))
    require_directory(source_root)
    if source_root != source_root.resolve():
        raise ProtocolError("source root must be canonical")
    bindings = config.get("source_bindings")
    if not isinstance(bindings, dict):
        raise ProtocolError("source bindings missing")
    required_files = (
        "graph_published", "graph_manifest", "records", "hyperedges", "graph_report",
        "solver_contract", "graph_deep_published", "graph_deep_receipt",
        "allocation_published", "allocation_receipt", "allocation_report",
    )
    for name in required_files:
        spec = bindings.get(name)
        if not isinstance(spec, dict):
            raise ProtocolError(f"missing source binding: {name}")
        path = Path(spec.get("path", ""))
        if not path.is_absolute() or not path.is_relative_to(source_root):
            raise ProtocolError(f"source binding escapes pinned source root: {name}")
        verify_file_spec(path, spec, name=name)
    graph_dir = Path(bindings["records"]["path"]).parent
    if graph_dir != Path(bindings["hyperedges"]["path"]).parent or graph_dir != Path(bindings["graph_report"]["path"]).parent:
        raise ProtocolError("graph payload directories differ")
    _validate_external_marker(
        graph_dir,
        protocol_id="pairwise_direct_collision_hypergraph_feasibility_v3",
        artifact_kind="direct_hypergraph",
        status="published_complete_direct_hypergraph",
    )
    graph_deep_dir = Path(bindings["graph_deep_receipt"]["path"]).parent
    _validate_external_marker(
        graph_deep_dir,
        protocol_id="pairwise_direct_collision_hypergraph_feasibility_v3",
        artifact_kind="graph_deep_validation",
        status="published_complete_graph_deep_validation",
    )
    allocation_dir = Path(bindings["allocation_receipt"]["path"]).parent
    allocation_binding = _validate_allocation_infeasible(allocation_dir, config)
    records = _validate_record_rows(Path(bindings["records"]["path"]))
    hyperedges = _validate_hyperedge_rows(Path(bindings["hyperedges"]["path"]), len(records))
    contract = strict_json_load(Path(bindings["solver_contract"]["path"]))
    graph_report = strict_json_load(Path(bindings["graph_report"]["path"]))
    if not isinstance(contract, dict) or not isinstance(graph_report, dict):
        raise ProtocolError("graph contract/report must be objects")
    if contract.get("protocol_id") != "pairwise_direct_collision_hypergraph_feasibility_v3" or graph_report.get("protocol_id") != contract.get("protocol_id"):
        raise ProtocolError("v3 protocol binding mismatch")
    forbidden = set(config["forbidden_data"]["forbidden_field_names"])
    _validate_no_forbidden_fields(records, forbidden)
    _validate_no_forbidden_fields(hyperedges, forbidden)
    _validate_no_forbidden_fields(contract, forbidden)
    # Thresholds and direct-collision policy are copied, never reinterpreted.
    if not json_exact(config.get("thresholds"), graph_report.get("thresholds")):
        raise ProtocolError("collision thresholds differ from v3 graph report")
    threshold_digest = hashlib.sha256(canonical_bytes(graph_report["thresholds"])).hexdigest()
    if config.get("thresholds_canonical_sha256") != threshold_digest:
        raise ProtocolError("threshold canonical digest mismatch")
    if not json_exact(config.get("collision_policy", {}).get("no_transitive_closure"), contract.get("constraints", {}).get("no_transitive_closure") is True):
        raise ProtocolError("collision policy no-transitive-closure binding mismatch")
    policy = config.get("collision_policy")
    contract_constraints = contract.get("constraints")
    report_policy = graph_report.get("direct_hyperedge_policy")
    if (
        not isinstance(policy, dict)
        or policy.get("selected_records_per_hyperedge_maximum") != 1
        or policy.get("record_target_assignments_per_record_maximum") != 1
        or not isinstance(contract_constraints, dict)
        or contract_constraints.get("direct_hyperedge") != "for every published hyperedge E, sum over records in E and both partition targets <= 1"
        or contract_constraints.get("record_assignment") != "for every record, sum over its two partition targets <= 1"
        or not isinstance(report_policy, dict)
        or report_policy.get("no_transitive_closure") is not True
        or report_policy.get("selected_records_per_hyperedge_maximum") != 1
        or report_policy.get("record_target_assignments_per_record_maximum") != 1
    ):
        raise ProtocolError("direct-collision policy is not the frozen v3 policy")
    targets = contract.get("targets")
    if not isinstance(targets, list) or len(targets) != 18:
        raise ProtocolError("v3 solver contract must expose exactly 18 targets")
    if config.get("target_order") != [target.get("target_index") for target in targets]:
        raise ProtocolError("public target order differs from v3 contract")
    upper = config.get("upper_quotas")
    if not isinstance(upper, dict):
        raise ProtocolError("upper quota map missing")
    for target in targets:
        key = str(target.get("target_index"))
        if key not in upper or type(upper[key]) is not int or upper[key] != target.get("exact_quota"):
            raise ProtocolError(f"upper quota is not exactly the v3 quota for target {key}")
    deep_ready, deep_binding = _validate_deep_allocation_placeholder(config, source_root)
    if require_deep_allocation and not deep_ready:
        raise ProtocolError("allocation_v3 deep receipt placeholder gate is not ready")
    return InputBundle(
        config=config,
        graph_dir=graph_dir,
        records=records,
        hyperedges=hyperedges,
        contract=contract,
        graph_report=graph_report,
        bindings={**bindings, "allocation_infeasible": allocation_binding},
        allocation_deep_ready=deep_ready,
        allocation_deep_binding=deep_binding,
    )


def load_config(path: Path) -> dict[str, Any]:
    value = strict_json_load(path)
    if not isinstance(value, dict):
        raise ProtocolError("config must be a JSON object")
    return value
