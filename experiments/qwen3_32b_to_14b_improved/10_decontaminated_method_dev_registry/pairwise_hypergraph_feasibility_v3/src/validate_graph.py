from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from .common import (
    ProtocolError,
    atomic_json,
    canonical_bytes,
    claim_artifact,
    file_spec,
    finalize_publication,
    json_exact,
    safe_relative_output,
    strict_json_loads,
    validate_publication,
    verify_runtime_identity,
)
from .graph import _implementation, build_graph_payloads, build_solver_contract, verify_bindings


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = {
    "record_index", "dataset_id", "stratum", "source_index", "source_projection_sha256"
}
_HYPEREDGE_KEYS = {
    "hyperedge_id", "member_record_indices", "member_count",
    "relation_kinds", "derivation_count_by_kind",
}
_RELATION_KINDS = {
    "raw_sample_id", "raw_question", "normalized_question",
    "normalized_question_context", "normalized_question_unit_owners",
    "normalized_passage_unit_owners",
    "normalized_question_substring_owner_union", "near_question_owner_union",
    "normalized_passage_substring_owner_union", "near_passage_owner_union",
}
_GRAPH_PAYLOADS = [
    "records.jsonl", "hyperedges.jsonl", "graph_report.json",
    "solver_contract.json", "manifest.json", "receipt.json",
]
_REPORT_KEYS = {
    "schema_version", "protocol_id", "status", "thresholds", "direct_hyperedge_policy",
    "record_count", "hyperedge_count", "hyperedge_member_occurrences",
    "hyperedge_member_count_distribution", "exact_group_statistics",
    "question_unit_statistics", "passage_unit_statistics",
    "question_similarity_statistics", "passage_similarity_statistics",
    "eligible_question_collision_residual_assertion", "observed_complete_scan",
    "decontamination_aggregate", "musique_4hop_record_buffer", "privacy", "claim_boundary",
}
_MANIFEST_KEYS = {
    "schema_version", "protocol_id", "artifact_kind", "status", "bindings",
    "implementation", "runtime", "files", "deep_validation_required_before_allocation",
    "deep_validation_receipt_path", "completion_marker_required",
}
_RECEIPT_KEYS = {
    "schema_version", "protocol_id", "artifact_kind", "status", "cpu_only",
    "record_count", "hyperedge_count", "manifest", "solver_contract",
    "component_v1_failure", "component_v1_checkpoint_erratum",
    "pairwise_v2_terminal_failure", "required_commands", "claim_boundary",
}
_SIMILARITY_STATISTIC_KEYS = {
    "unique_units", "candidates_verified", "direct_relation_pairs",
    "relation_pairs_by_kind", "nontrivial_owner_union_derivations",
    "terminal_completion_event",
}
_TERMINAL_EVENT_KEYS = {
    "phase", "completed_left_count", "unique_units_total",
    "candidates_verified", "direct_relation_pairs",
}


def _validate_passage_similarity_statistics(
    passage_statistics: Any, expected_scan: Mapping[str, Any]
) -> None:
    if not isinstance(passage_statistics, dict):
        raise ProtocolError("passage similarity statistics must be an object")
    terminal_event = passage_statistics.get("terminal_completion_event", {})
    expected_terminal_event = {
        "phase": "passage_similarity_complete",
        "completed_left_count": expected_scan["unique_normalized_passage_units"],
        "unique_units_total": expected_scan["unique_normalized_passage_units"],
        "candidates_verified": expected_scan["passage_candidates_verified"],
        "direct_relation_pairs": expected_scan["passage_direct_relation_pairs"],
    }
    relation_counts = passage_statistics.get("relation_pairs_by_kind", {})
    if (
        set(passage_statistics) != _SIMILARITY_STATISTIC_KEYS
        or type(passage_statistics.get("unique_units")) is not int
        or passage_statistics["unique_units"] != expected_scan["unique_normalized_passage_units"]
        or type(passage_statistics.get("candidates_verified")) is not int
        or passage_statistics["candidates_verified"] != expected_scan["passage_candidates_verified"]
        or type(passage_statistics.get("direct_relation_pairs")) is not int
        or passage_statistics["direct_relation_pairs"] != expected_scan["passage_direct_relation_pairs"]
        or type(passage_statistics.get("nontrivial_owner_union_derivations")) is not int
        or passage_statistics["nontrivial_owner_union_derivations"] < 0
        or not isinstance(relation_counts, dict)
        or not set(relation_counts) <= {
            "normalized_passage_substring_owner_union", "near_passage_owner_union"
        }
        or any(type(value) is not int or value <= 0 for value in relation_counts.values())
        or sum(relation_counts.values()) != passage_statistics["direct_relation_pairs"]
        or not isinstance(terminal_event, dict)
        or set(terminal_event) != _TERMINAL_EVENT_KEYS
        or not json_exact(terminal_event, expected_terminal_event)
        or "unique_units_processed" in terminal_event
    ):
        raise ProtocolError("passage similarity terminal-completion semantics mismatch")


def _load_records(path: Path, cells_order: list[str]) -> list[dict[str, Any]]:
    rows = []
    previous_projection = ""
    valid_cells = set(cells_order)
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        row = strict_json_loads(line)
        if (
            set(row) != _RECORD_KEYS
            or type(row["record_index"]) is not int
            or row["record_index"] != len(rows)
        ):
            raise ProtocolError(f"record row/index mismatch at line {line_number}")
        projection = row["source_projection_sha256"]
        if not isinstance(projection, str) or not _HEX64.fullmatch(projection) or projection <= previous_projection:
            raise ProtocolError("records must have unique ascending projection hashes")
        previous_projection = projection
        if (
            not isinstance(row["dataset_id"], str)
            or not isinstance(row["stratum"], str)
            or f"{row['dataset_id']}|{row['stratum']}" not in valid_cells
        ):
            raise ProtocolError("record cell is not frozen")
        if type(row["source_index"]) is not int or row["source_index"] < 0:
            raise ProtocolError("record source index invalid")
        rows.append(row)
    if not rows:
        raise ProtocolError("records graph cannot be empty")
    return rows


def _load_hyperedges(path: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    previous_id = ""
    member_sets: set[tuple[int, ...]] = set()
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        row = strict_json_loads(line)
        if set(row) != _HYPEREDGE_KEYS:
            raise ProtocolError(f"hyperedge keys mismatch at line {line_number}")
        edge_id = row["hyperedge_id"]
        if not isinstance(edge_id, str) or not _HEX64.fullmatch(edge_id) or edge_id <= previous_id:
            raise ProtocolError("hyperedges must have unique ascending IDs")
        previous_id = edge_id
        members = row["member_record_indices"]
        if (
            not isinstance(members, list)
            or any(type(value) is not int for value in members)
            or len(members) < 2
            or members != sorted(set(members))
            or members[-1] >= len(records)
            or members[0] < 0
            or type(row["member_count"]) is not int
            or row["member_count"] != len(members)
        ):
            raise ProtocolError("invalid hyperedge members")
        member_tuple = tuple(members)
        if member_tuple in member_sets:
            raise ProtocolError("duplicate hyperedge member set")
        member_sets.add(member_tuple)
        expected_id = hashlib.sha256(canonical_bytes([
            records[index]["source_projection_sha256"] for index in members
        ])).hexdigest()
        if edge_id != expected_id:
            raise ProtocolError("hyperedge content ID mismatch")
        kinds = row["relation_kinds"]
        derivations = row["derivation_count_by_kind"]
        if (
            not isinstance(kinds, list)
            or kinds != sorted(set(kinds))
            or not set(kinds) <= _RELATION_KINDS
            or not isinstance(derivations, dict)
            or set(derivations) != set(kinds)
            or any(type(value) is not int or value <= 0 for value in derivations.values())
        ):
            raise ProtocolError("invalid hyperedge derivation semantics")
        rows.append(row)
    return rows


def _verify_implementation(root: Path, config_path: Path, observed: Mapping[str, Any]) -> None:
    if not json_exact(observed, _implementation(root, config_path)):
        raise ProtocolError("graph implementation binding mismatch")


def _byte_spec_jsonl(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    nbytes = 0
    for row in rows:
        payload = canonical_bytes(row)
        digest.update(payload)
        nbytes += len(payload)
    return {"nbytes": nbytes, "sha256": digest.hexdigest()}


def _write_deep_receipt(
    config: Mapping[str, Any],
    root: Path,
    graph_dir: Path,
    deep: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = safe_relative_output(root, config["outputs"]["graph_deep_receipt"])
    identity = {"protocol_id": config["protocol_id"], "artifact_kind": "graph_deep_validation"}
    graph_manifest = strict_json_loads((graph_dir / "manifest.json").read_text())
    graph_process_id = graph_manifest["runtime"]["process_id"]
    if type(graph_process_id) is not int or graph_process_id <= 0 or graph_process_id == os.getpid():
        raise ProtocolError("deep graph validation must run in a process distinct from graph build")
    claim_artifact(output_dir, identity)
    receipt = {
        "schema_version": 1,
        **identity,
        "status": "pass_pinned_source_graph_byte_identical",
        "cpu_only": True,
        "runtime_identity": config["runtime_identity"],
        "graph_build_process": {
            "process_id": graph_process_id,
            "python_hash_seed": config["deterministic_processes"]["graph_build_python_hash_seed"],
        },
        "deep_validation_process": {
            "process_id": os.getpid(),
            "python_hash_seed": config["deterministic_processes"]["graph_deep_python_hash_seed"],
        },
        "separate_process_assertion": True,
        "graph_PUBLISHED": file_spec(graph_dir / "PUBLISHED.json"),
        "graph_manifest": file_spec(graph_dir / "manifest.json"),
        "deep_recomputed_from_pinned_raw_sources": True,
        "records_byte_identical": True,
        "hyperedges_byte_identical": True,
        "graph_report_byte_identical": True,
        "solver_contract_byte_identical": True,
        "replica_files": dict(deep),
        "required_allocation_command": config["commands"]["allocation"],
        "claim_boundary": "deep validation only; no MILP solved and no allocation selected",
    }
    atomic_json(output_dir / "receipt.json", receipt)
    marker = finalize_publication(
        output_dir,
        ["receipt.json"],
        identity=identity,
        status="published_complete_graph_deep_validation",
    )
    return {"artifact": str(output_dir), "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"), "ordered_inventory_sha256": marker["ordered_inventory_sha256"]}


def validate_graph(
    config_path: Path,
    root: Path,
    *,
    deep: bool = False,
    write_receipt: bool = False,
) -> dict[str, Any]:
    if write_receipt and not deep:
        raise ProtocolError("a deep receipt requires deep source recomputation")
    config = strict_json_loads(config_path.read_text())
    verify_runtime_identity(config["runtime_identity"])
    if deep:
        required_seed = config["deterministic_processes"]["graph_deep_python_hash_seed"]
        empty_prefix = Path(config["deterministic_processes"]["empty_python_pycache_prefix"])
        if os.environ.get("PYTHONHASHSEED") != required_seed:
            raise ProtocolError("deep graph validation PYTHONHASHSEED mismatch")
        if os.environ.get("PYTHONPYCACHEPREFIX") != str(empty_prefix) or empty_prefix.exists():
            raise ProtocolError("deep graph validation requires the frozen absent pycache prefix")
    _, _, registry_config, _ = verify_bindings(config)
    graph_dir = safe_relative_output(root, config["outputs"]["graph_artifact_dir"])
    validate_publication(
        graph_dir,
        _GRAPH_PAYLOADS,
        identity={"protocol_id": config["protocol_id"], "artifact_kind": "direct_hypergraph"},
        status="published_complete_direct_hypergraph",
    )
    records = _load_records(graph_dir / "records.jsonl", config["cells_order"])
    hyperedges = _load_hyperedges(graph_dir / "hyperedges.jsonl", records)
    report = strict_json_loads((graph_dir / "graph_report.json").read_text())
    contract = strict_json_loads((graph_dir / "solver_contract.json").read_text())
    manifest = strict_json_loads((graph_dir / "manifest.json").read_text())
    receipt = strict_json_loads((graph_dir / "receipt.json").read_text())
    if not json_exact(contract, build_solver_contract(config)):
        raise ProtocolError("solver contract does not recompute")
    if (
        set(report) != _REPORT_KEYS
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("protocol_id") != config["protocol_id"]
        or report.get("status") != "direct_hypergraph_built_not_allocation"
        or type(report.get("record_count")) is not int
        or report.get("record_count") != len(records)
        or type(report.get("hyperedge_count")) is not int
        or report.get("hyperedge_count") != len(hyperedges)
        or not json_exact(report.get("direct_hyperedge_policy"), config["direct_hyperedge_policy"])
        or not json_exact(report.get("observed_complete_scan"), config["expected_complete_scan"])
        or not json_exact(report.get("thresholds"), registry_config["decontamination"])
        or report.get("claim_boundary") != "privacy-safe direct hypergraph only; no MILP solved and no allocation selected"
    ):
        raise ProtocolError("graph report structural semantics mismatch")
    _validate_passage_similarity_statistics(
        report.get("passage_similarity_statistics"), config["expected_complete_scan"]
    )
    member_counts = [row["member_count"] for row in hyperedges]
    if type(report.get("hyperedge_member_occurrences")) is not int or report.get("hyperedge_member_occurrences") != sum(member_counts) or not json_exact(report.get("hyperedge_member_count_distribution"), {
        "minimum": min(member_counts) if member_counts else 0,
        "maximum": max(member_counts) if member_counts else 0,
        "size_2": sum(value == 2 for value in member_counts),
        "size_gt_2": sum(value > 2 for value in member_counts),
    }):
        raise ProtocolError("hyperedge distribution does not recompute")
    residual = report.get("eligible_question_collision_residual_assertion", {})
    expected_residual_keys = {
        "pass", "raw_question", "normalized_question", "normalized_question_context",
        "normalized_question_unit_owner_groups", "question_direct_relation_pairs",
    }
    if (
        set(residual) != expected_residual_keys
        or residual.get("pass") is not True
        or any(type(value) is not int or value != 0 for key, value in residual.items() if key != "pass")
    ):
        raise ProtocolError("question residual assertion mismatch")
    privacy = report.get("privacy")
    if privacy != {
        "raw_ids_questions_contexts_answers_or_outcomes_persisted": False,
        "answer_values_available_to_graph_logic": False,
        "model_outcomes_read": False,
    }:
        raise ProtocolError("graph privacy semantics mismatch")
    clean_4hop = sum(
        row["dataset_id"] == "musique_train" and row["stratum"] == "4hop"
        for row in records
    )
    selected_4hop = sum(
        config["original_quotas"][partition]["musique_train|4hop"]
        for partition in config["partitions_order"]
    )
    expected_buffer = {
        **config["musique_4hop_record_buffer"],
        "observed_clean_records": clean_4hop,
        "original_quota_selected_records": selected_4hop,
        "observed_unselected_record_buffer": clean_4hop - selected_4hop,
    }
    buffer_contract = config["musique_4hop_record_buffer"]
    if (
        not json_exact(report.get("musique_4hop_record_buffer"), expected_buffer)
        or clean_4hop != buffer_contract["strict_clean_records"]
        or selected_4hop > buffer_contract["maximum_selected_records"]
        or clean_4hop - selected_4hop < buffer_contract["minimum_unselected_records"]
    ):
        raise ProtocolError("MuSiQue 4hop record buffer report mismatch")
    if (
        set(manifest) != _MANIFEST_KEYS
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != config["protocol_id"]
        or manifest.get("artifact_kind") != "direct_hypergraph"
        or manifest.get("status") != "prepared_privacy_safe_reusable_hypergraph"
        or not json_exact(manifest.get("bindings"), config["bindings"])
        or manifest.get("deep_validation_required_before_allocation") is not True
        or manifest.get("deep_validation_receipt_path") != str(
            safe_relative_output(root, config["outputs"]["graph_deep_receipt"])
        )
        or manifest.get("completion_marker_required") != "PUBLISHED.json"
    ):
        raise ProtocolError("graph manifest semantics mismatch")
    runtime = manifest.get("runtime", {})
    if (
        set(runtime) != {*config["runtime_identity"], "python_hash_seed", "python_pycache_prefix_absent", "process_id"}
        or not json_exact({key: runtime[key] for key in config["runtime_identity"]}, config["runtime_identity"])
        or runtime.get("python_hash_seed") != config["deterministic_processes"]["graph_build_python_hash_seed"]
        or runtime.get("python_pycache_prefix_absent") != config["deterministic_processes"]["empty_python_pycache_prefix"]
        or type(runtime.get("process_id")) is not int
        or runtime["process_id"] <= 0
    ):
        raise ProtocolError("graph runtime identity/process semantics mismatch")
    _verify_implementation(root, config_path, manifest["implementation"])
    expected_files = {
        "records.jsonl": {**file_spec(graph_dir / "records.jsonl"), "rows": len(records)},
        "hyperedges.jsonl": {**file_spec(graph_dir / "hyperedges.jsonl"), "rows": len(hyperedges)},
        "graph_report.json": file_spec(graph_dir / "graph_report.json"),
        "solver_contract.json": file_spec(graph_dir / "solver_contract.json"),
    }
    if not json_exact(manifest.get("files"), expected_files):
        raise ProtocolError("graph manifest file bindings mismatch")
    if (
        set(receipt) != _RECEIPT_KEYS
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != config["protocol_id"]
        or receipt.get("artifact_kind") != "direct_hypergraph"
        or receipt.get("status") != "pass_graph_built_no_allocation"
        or receipt.get("cpu_only") is not True
        or type(receipt.get("record_count")) is not int
        or receipt.get("record_count") != len(records)
        or type(receipt.get("hyperedge_count")) is not int
        or receipt.get("hyperedge_count") != len(hyperedges)
        or not json_exact(receipt.get("manifest"), file_spec(graph_dir / "manifest.json"))
        or not json_exact(receipt.get("solver_contract"), file_spec(graph_dir / "solver_contract.json"))
        or not json_exact(receipt.get("component_v1_failure"), config["bindings"]["component_v1_failure"])
        or not json_exact(
            receipt.get("component_v1_checkpoint_erratum"),
            config["bindings"]["component_v1_checkpoint_erratum"],
        )
        or not json_exact(
            receipt.get("pairwise_v2_terminal_failure"),
            config["bindings"]["pairwise_v2_terminal_failure"],
        )
        or not json_exact(receipt.get("required_commands"), config["commands"])
        or receipt.get("claim_boundary") != "graph and reusable solver contract only; no feasibility or allocation result"
    ):
        raise ProtocolError("graph receipt semantics mismatch")

    deep_result: dict[str, Any] = {"performed": False}
    written_receipt = None
    if deep:
        implementation_start = _implementation(root, config_path)
        def progress(value: Mapping[str, Any]) -> None:
            print(json.dumps({"deep_graph_validation": dict(value)}, sort_keys=True), file=sys.stderr, flush=True)

        replica_records, replica_edges, replica_report, replica_contract = build_graph_payloads(
            config, progress=progress
        )
        implementation_end = _implementation(root, config_path)
        if not json_exact(implementation_start, implementation_end) or not json_exact(
            implementation_end, manifest["implementation"]
        ):
            raise ProtocolError("implementation changed during deep graph rebuild")
        replica_specs = {
            "records.jsonl": _byte_spec_jsonl(replica_records),
            "hyperedges.jsonl": _byte_spec_jsonl(replica_edges),
            "graph_report.json": {
                "nbytes": len(canonical_bytes(replica_report)),
                "sha256": hashlib.sha256(canonical_bytes(replica_report)).hexdigest(),
            },
            "solver_contract.json": {
                "nbytes": len(canonical_bytes(replica_contract)),
                "sha256": hashlib.sha256(canonical_bytes(replica_contract)).hexdigest(),
            },
        }
        observed_specs = {
            name: file_spec(graph_dir / name, include_path=False)
            for name in replica_specs
        }
        if replica_specs != observed_specs or replica_records != records or replica_edges != hyperedges or canonical_bytes(replica_report) != (graph_dir / "graph_report.json").read_bytes() or canonical_bytes(replica_contract) != (graph_dir / "solver_contract.json").read_bytes():
            raise ProtocolError("pinned-source deep graph replica is not byte-identical")
        deep_result = {
            "performed": True,
            "recomputed_from_pinned_raw_sources": True,
            "all_four_payloads_byte_identical": True,
            "replica_files": replica_specs,
            "deep_process_id": os.getpid(),
            "deep_python_hash_seed": required_seed,
        }
        if write_receipt:
            written_receipt = _write_deep_receipt(config, root, graph_dir, replica_specs)
    return {
        "status": "pass",
        "artifact": str(graph_dir),
        "records": len(records),
        "hyperedges": len(hyperedges),
        "PUBLISHED": file_spec(graph_dir / "PUBLISHED.json"),
        "deep_validation": deep_result,
        "deep_receipt": written_receipt,
    }
