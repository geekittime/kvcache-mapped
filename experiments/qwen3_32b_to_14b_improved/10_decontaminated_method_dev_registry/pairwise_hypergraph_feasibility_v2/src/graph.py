from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .common import (
    ProtocolError,
    atomic_json,
    atomic_jsonl,
    canonical_bytes,
    claim_artifact,
    file_spec,
    finalize_publication,
    json_exact,
    load_module,
    read_pinned_json,
    require_regular_file,
    safe_relative_output,
    strict_json_loads,
    verify_file,
    verify_runtime_identity,
    verify_sidecar,
)


Progress = Callable[[Mapping[str, Any]], None] | None


def _progress(callback: Progress, **value: Any) -> None:
    if callback is not None:
        callback(value)


def verify_component_v1_failure(path: Path, audit_root: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_text())
    expected = {
        "schema_version": 1,
        "attempt_id": "component_feasibility_buffer_prerequisite_fail_v1",
        "status": "failed_closed_preserved_append_only",
        "error": "fewer buffer-supporting components than required reserve",
        "formal_feasibility_payload_files_written": 0,
        "completion_marker_present": False,
        "published": False,
        "gpu_used": False,
        "model_outcomes_read": False,
        "thresholds_changed": False,
        "answer_values_available_to_component_or_flow_logic": False,
        "observed_complete_scan": {
            "clean_eligible_records": 31260,
            "passage_candidates_verified": 39817022,
            "passage_direct_relation_pairs": 7797,
            "unique_normalized_passage_units": 244038,
        },
        "prerequisite": {
            "minimum_completely_unselected_4hop_supporting_components": 2,
            "observed_4hop_supporting_connected_components": "strictly_less_than_2",
            "pass": False,
        },
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ProtocolError("component v1 failure evidence semantics mismatch")
    artifact = audit_root / value["artifact_dir"]
    if not artifact.is_dir() or sorted(child.name for child in artifact.iterdir()) != ["IN_PROGRESS.json"]:
        raise ProtocolError("component v1 failed artifact inventory mismatch")
    verify_file(artifact / "IN_PROGRESS.json", value["in_progress_marker"])
    verify_file(audit_root / value["log"]["path"], value["log"])
    verify_file(audit_root / value["config"]["path"], value["config"])
    for spec in value["implementation"].values():
        if "path" in spec:
            verify_file(audit_root / spec["path"], spec)
        else:
            for nested in spec.values():
                verify_file(audit_root / nested["path"], nested)
    return value


def verify_bindings(config: Mapping[str, Any]) -> tuple[Any, Any, Mapping[str, Any], Path]:
    verify_runtime_identity(config["runtime_identity"])
    bindings = config["bindings"]
    for spec in bindings.values():
        verify_file(Path(spec["path"]), spec)
    verify_sidecar(
        Path(bindings["component_v1_failure_sidecar"]["path"]),
        Path(bindings["component_v1_failure"]["path"]),
        bindings["component_v1_failure"],
    )
    audit_root = Path(bindings["registry_config"]["path"]).parent
    verify_component_v1_failure(Path(bindings["component_v1_failure"]["path"]), audit_root)
    core = load_module(
        "hypergraph_v2_frozen_registry_core",
        Path(bindings["registry_source"]["path"]),
        bindings["registry_source"],
    )
    component_v1 = load_module(
        "hypergraph_v2_frozen_component_v1",
        Path(bindings["component_v1_source"]["path"]),
        bindings["component_v1_source"],
    )
    registry_config = read_pinned_json(
        Path(bindings["registry_config"]["path"]), bindings["registry_config"]
    )
    component_v1.verify_attempt3_failure(Path(bindings["attempt3_failure"]["path"]), audit_root)
    return core, component_v1, registry_config, audit_root


def verify_registry_inputs_unchanged(
    registry_config: Mapping[str, Any], audit_root: Path
) -> None:
    for spec in registry_config["inputs"].values():
        verify_file(Path(spec["path"]), spec)
    for name, spec in registry_config["failure_evidence"].items():
        if not isinstance(spec, Mapping) or "path" not in spec:
            continue
        path = Path(spec["path"])
        verify_file(path if path.is_absolute() else audit_root / path, spec)


def load_clean_eligible(
    config: Mapping[str, Any],
    *,
    progress: Progress = None,
) -> tuple[Any, Mapping[str, Any], Path, list[Any], Mapping[str, Any]]:
    core, component_v1, registry_config, audit_root = verify_bindings(config)
    eligible, quarantine = component_v1.prepare_eligible(
        core, registry_config, audit_root, progress
    )
    expected = config["expected_complete_scan"]["clean_eligible_records"]
    if len(eligible) != expected:
        raise ProtocolError(f"clean eligible record count changed: {len(eligible)} != {expected}")
    return core, registry_config, audit_root, eligible, quarantine


def make_record_rows(eligible: Sequence[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    ordered = sorted(
        eligible,
        key=lambda row: (row.projection_sha256, row.dataset_id, row.stratum, row.source_index),
    )
    projection_hashes = [row.projection_sha256 for row in ordered]
    if len(projection_hashes) != len(set(projection_hashes)):
        raise ProtocolError("source projection hashes must uniquely identify clean records")
    rows = [{
        "record_index": index,
        "dataset_id": row.dataset_id,
        "stratum": row.stratum,
        "source_index": row.source_index,
        "source_projection_sha256": row.projection_sha256,
    } for index, row in enumerate(ordered)]
    return ordered, rows


def _add_hyperedge(
    edges: dict[tuple[int, ...], Counter[str]],
    owners: Iterable[int],
    kind: str,
) -> bool:
    members = tuple(sorted(set(owners)))
    if len(members) < 2:
        return False
    edges.setdefault(members, Counter())[kind] += 1
    return True


def _add_exact_groups(
    records: Sequence[Any],
    core: Any,
    edges: dict[tuple[int, ...], Counter[str]],
) -> dict[str, Any]:
    key_functions = (
        ("raw_sample_id", lambda row: row.sample_id),
        ("raw_question", lambda row: core.digest(row.question)),
        ("normalized_question", lambda row: core.digest(core.normalize(row.question))),
        (
            "normalized_question_context",
            lambda row: core.digest(
                core.normalize(row.question)
                + "\u241f"
                + core.normalize("\n\n".join(row.passages))
            ),
        ),
    )
    result: dict[str, Any] = {}
    for kind, key_function in key_functions:
        groups: dict[str, list[int]] = defaultdict(list)
        for record_index, record in enumerate(records):
            groups[key_function(record)].append(record_index)
        derivations = 0
        records_in_groups = 0
        for owners in groups.values():
            if len(owners) >= 2:
                derivations += 1
                records_in_groups += len(owners)
                _add_hyperedge(edges, owners, kind)
        result[kind] = {
            "groups_with_multiple_owners": derivations,
            "owner_occurrences_in_groups": records_in_groups,
        }
    return result


def _collapse_units(
    records: Sequence[Any],
    core: Any,
    *,
    passages: bool,
    edges: dict[tuple[int, ...], Counter[str]],
    progress: Progress,
) -> tuple[list[str], list[tuple[int, ...]], dict[str, Any]]:
    index_by_hash: dict[str, int] = {}
    texts: list[str] = []
    mutable_owners: list[set[int]] = []
    occurrences = 0
    for record_index, record in enumerate(records):
        values = record.passages if passages else (record.question,)
        seen_in_record: set[str] = set()
        for raw in values:
            occurrences += 1
            normalized = core.normalize(raw)
            normalized_hash = core.digest(normalized)
            if normalized_hash in seen_in_record:
                continue
            seen_in_record.add(normalized_hash)
            unit_index = index_by_hash.get(normalized_hash)
            if unit_index is None:
                unit_index = len(texts)
                index_by_hash[normalized_hash] = unit_index
                texts.append(normalized)
                mutable_owners.append(set())
            elif texts[unit_index] != normalized:
                raise ProtocolError("normalized SHA-256 collision")
            mutable_owners[unit_index].add(record_index)
        if record_index and record_index % 10000 == 0:
            _progress(
                progress,
                phase="collapse_passage_units" if passages else "collapse_question_units",
                records_processed=record_index,
                unique_units=len(texts),
            )
    owners = [tuple(sorted(value)) for value in mutable_owners]
    kind = "normalized_passage_unit_owners" if passages else "normalized_question_unit_owners"
    exact_owner_groups = sum(_add_hyperedge(edges, value, kind) for value in owners)
    return texts, owners, {
        "occurrences": occurrences,
        "unique_normalized_units": len(texts),
        "units_with_multiple_owners": exact_owner_groups,
        "owner_occurrences": sum(len(value) for value in owners),
    }


def _add_similarity_hyperedges(
    texts: Sequence[str],
    owners: Sequence[Sequence[int]],
    core: Any,
    thresholds: Mapping[str, Any],
    edges: dict[tuple[int, ...], Counter[str]],
    *,
    question: bool,
    progress: Progress,
) -> dict[str, Any]:
    width = thresholds["question_shingle_words"] if question else thresholds["passage_shingle_words"]
    minimum_intersection = thresholds["question_min_intersection"] if question else thresholds["passage_min_intersection"]
    minimum_containment = thresholds["question_min_containment"] if question else thresholds["passage_min_containment"]
    minimum_jaccard = thresholds["question_min_jaccard"] if question else thresholds["passage_min_jaccard"]
    substring_minimum = thresholds["exact_normalized_unit_substring_min_characters"]
    index = core.build_unit_index(texts, width, substring_minimum)
    candidates_verified = 0
    relation_counts: Counter[str] = Counter()
    hyperedge_derivations = 0
    for left in range(len(texts)):
        left_set = index.sets[left]
        counts: Counter[int] = Counter()
        for token in sorted(left_set):
            for right in index.inverted.get(token, ()):
                if right > left:
                    counts[right] += 1
        candidates = set(counts)
        for fingerprint in core._substring_fingerprints(index.texts[left], substring_minimum):
            candidates.update(
                right
                for right in index.substring_fingerprint_inverted.get(fingerprint, ())
                if right > left
            )
        for right in sorted(candidates):
            candidates_verified += 1
            kind = None
            shorter, longer = (
                (index.texts[left], index.texts[right])
                if len(index.texts[left]) <= len(index.texts[right])
                else (index.texts[right], index.texts[left])
            )
            if len(shorter) >= substring_minimum and shorter in longer:
                kind = "normalized_question_substring_owner_union" if question else "normalized_passage_substring_owner_union"
            elif right in counts:
                intersection = counts[right]
                denominator = min(len(left_set), len(index.sets[right]))
                containment = intersection / denominator if denominator else 0.0
                union_size = len(left_set) + len(index.sets[right]) - intersection
                jaccard = intersection / union_size if union_size else 0.0
                if intersection >= minimum_intersection and (
                    containment >= minimum_containment or jaccard >= minimum_jaccard
                ):
                    kind = "near_question_owner_union" if question else "near_passage_owner_union"
            if kind is not None:
                relation_counts[kind] += 1
                hyperedge_derivations += int(
                    _add_hyperedge(edges, [*owners[left], *owners[right]], kind)
                )
        if left and left % 10000 == 0:
            _progress(
                progress,
                phase="question_similarity_hyperedges" if question else "passage_similarity_hyperedges",
                unique_units_processed=left,
                unique_units_total=len(texts),
                candidates_verified=candidates_verified,
                direct_relation_pairs=sum(relation_counts.values()),
            )
    return {
        "unique_units": len(texts),
        "candidates_verified": candidates_verified,
        "direct_relation_pairs": sum(relation_counts.values()),
        "relation_pairs_by_kind": dict(sorted(relation_counts.items())),
        "nontrivial_owner_union_derivations": hyperedge_derivations,
    }


def _hyperedge_rows(
    edges: Mapping[tuple[int, ...], Counter[str]],
    record_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    seen_ids: set[str] = set()
    for members, kinds in edges.items():
        projections = [record_rows[index]["source_projection_sha256"] for index in members]
        hyperedge_id = hashlib.sha256(canonical_bytes(projections)).hexdigest()
        if hyperedge_id in seen_ids:
            raise ProtocolError("hyperedge identifier collision")
        seen_ids.add(hyperedge_id)
        result.append({
            "hyperedge_id": hyperedge_id,
            "member_record_indices": list(members),
            "member_count": len(members),
            "relation_kinds": sorted(kinds),
            "derivation_count_by_kind": dict(sorted(kinds.items())),
        })
    result.sort(key=lambda row: row["hyperedge_id"])
    return result


def build_solver_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    cells = config["cells_order"]
    partitions = config["partitions_order"]
    if (
        not isinstance(cells, list)
        or len(cells) != 9
        or len(set(cells)) != len(cells)
        or any(not isinstance(cell, str) or cell.count("|") != 1 for cell in cells)
        or not json_exact(partitions, ["screen", "confirm"])
        or set(config["original_quotas"]) != set(partitions)
    ):
        raise ProtocolError("target cell/partition schema mismatch")
    for partition in partitions:
        quotas = config["original_quotas"][partition]
        if set(quotas) != set(cells) or any(
            type(value) is not int or value < 0 for value in quotas.values()
        ):
            raise ProtocolError("target quotas must be exact nonnegative integers")
    solver = config["solver"]
    if (
        solver.get("scipy_version") != config["runtime_identity"]["scipy_version"]
        or not json_exact(solver.get("required_environment"), {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        or not json_exact(solver.get("highs_options"), {
            "threads": 1, "presolve": True, "mip_rel_gap": 0.0,
        })
        or not json_exact(solver.get("replica_python_hash_seeds"), ["1", "777"])
        or not json_exact(solver.get("replica_highs_random_seeds"), [1, 777])
    ):
        raise ProtocolError("frozen solver runtime/options schema mismatch")
    buffer = config["musique_4hop_record_buffer"]
    if any(type(buffer.get(key)) is not int or buffer[key] < 0 for key in (
        "strict_clean_records", "maximum_selected_records", "minimum_unselected_records"
    )) or buffer.get("does_not_require_distinct_or_unselected_connected_components") is not True:
        raise ProtocolError("record-buffer typed schema mismatch")
    targets = []
    for partition in partitions:
        for cell in cells:
            dataset_id, stratum = cell.split("|", 1)
            targets.append({
                "target_index": len(targets),
                "partition": partition,
                "cell": cell,
                "dataset_id": dataset_id,
                "stratum": stratum,
                "exact_quota": config["original_quotas"][partition][cell],
            })
    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "reusable_solver_contract_not_allocation",
        "runtime_identity": config["runtime_identity"],
        "solver": solver,
        "commands": config["commands"],
        "targets": targets,
        "variable_domain": "binary y[target_index,record_index] only when record cell equals target cell",
        "constraints": {
            "record_assignment": "for every record, sum over its two partition targets <= 1",
            "direct_hyperedge": "for every published hyperedge E, sum over records in E and both partition targets <= 1",
            "target_quota": "for every target, sum eligible record variables == exact_quota",
            "no_transitive_closure": True,
        },
        "objective_cost": {
            "formula": "first 13 hex digits of sha256(namespace + NUL + target_index + NUL + source_projection_sha256), interpreted as an integer divided by 16^13",
            "direction": "minimize",
            "mathematical_unique_optimum_claimed": False,
            "two_process_byte_identity_required": True,
        },
        "musique_4hop_record_buffer": config["musique_4hop_record_buffer"],
        "selection_input_policy": {
            "inputs": config["privacy"]["solver_selection_inputs"],
            "projection_excludes_answers": config["privacy"]["solver_selection_input_projection_excludes_answers"],
            "answer_values_available": config["privacy"]["answer_values_available_to_graph_solver_or_allocation_logic"],
            "model_outcomes_read": config["privacy"]["model_outcomes_read"],
        },
        "claim_boundary": "contract only; no MILP solved and no records allocated",
    }


def build_graph_payloads(
    config: Mapping[str, Any],
    *,
    progress: Progress = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    core, registry_config, _, eligible, quarantine = load_clean_eligible(config, progress=progress)
    records, record_rows = make_record_rows(eligible)
    edges: dict[tuple[int, ...], Counter[str]] = {}
    exact_stats = _add_exact_groups(records, core, edges)
    question_texts, question_owners, question_units = _collapse_units(
        records, core, passages=False, edges=edges, progress=progress
    )
    passage_texts, passage_owners, passage_units = _collapse_units(
        records, core, passages=True, edges=edges, progress=progress
    )
    thresholds = registry_config["decontamination"]
    question_similarity = _add_similarity_hyperedges(
        question_texts, question_owners, core, thresholds, edges,
        question=True, progress=progress,
    )
    passage_similarity = _add_similarity_hyperedges(
        passage_texts, passage_owners, core, thresholds, edges,
        question=False, progress=progress,
    )
    expected = config["expected_complete_scan"]
    observed_scan = {
        "clean_eligible_records": len(record_rows),
        "unique_normalized_passage_units": passage_units["unique_normalized_units"],
        "passage_candidates_verified": passage_similarity["candidates_verified"],
        "passage_direct_relation_pairs": passage_similarity["direct_relation_pairs"],
    }
    if observed_scan != expected:
        raise ProtocolError(f"complete direct relation scan changed: {observed_scan}")
    residual_question = {
        **{key: value["groups_with_multiple_owners"] for key, value in exact_stats.items() if key != "raw_sample_id"},
        "normalized_question_unit_owner_groups": question_units["units_with_multiple_owners"],
        "question_direct_relation_pairs": question_similarity["direct_relation_pairs"],
    }
    if any(residual_question.values()):
        raise ProtocolError(f"clean pool retained a question collision: {residual_question}")
    hyperedges = _hyperedge_rows(edges, record_rows)
    member_counts = [row["member_count"] for row in hyperedges]
    clean_4hop = sum(
        row["dataset_id"] == "musique_train" and row["stratum"] == "4hop"
        for row in record_rows
    )
    selected_4hop = sum(
        config["original_quotas"][partition]["musique_train|4hop"]
        for partition in config["partitions_order"]
    )
    buffer = config["musique_4hop_record_buffer"]
    if clean_4hop != buffer["strict_clean_records"] or selected_4hop > buffer["maximum_selected_records"] or clean_4hop - selected_4hop < buffer["minimum_unselected_records"]:
        raise ProtocolError("MuSiQue 4hop record-buffer semantics mismatch")
    graph_report = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "direct_hypergraph_built_not_allocation",
        "thresholds": thresholds,
        "direct_hyperedge_policy": config["direct_hyperedge_policy"],
        "record_count": len(record_rows),
        "hyperedge_count": len(hyperedges),
        "hyperedge_member_occurrences": sum(member_counts),
        "hyperedge_member_count_distribution": {
            "minimum": min(member_counts) if member_counts else 0,
            "maximum": max(member_counts) if member_counts else 0,
            "size_2": sum(value == 2 for value in member_counts),
            "size_gt_2": sum(value > 2 for value in member_counts),
        },
        "exact_group_statistics": exact_stats,
        "question_unit_statistics": question_units,
        "passage_unit_statistics": passage_units,
        "question_similarity_statistics": question_similarity,
        "passage_similarity_statistics": passage_similarity,
        "eligible_question_collision_residual_assertion": {"pass": True, **residual_question},
        "observed_complete_scan": observed_scan,
        "decontamination_aggregate": quarantine,
        "musique_4hop_record_buffer": {
            **buffer,
            "observed_clean_records": clean_4hop,
            "original_quota_selected_records": selected_4hop,
            "observed_unselected_record_buffer": clean_4hop - selected_4hop,
        },
        "privacy": {
            "raw_ids_questions_contexts_answers_or_outcomes_persisted": False,
            "answer_values_available_to_graph_logic": False,
            "model_outcomes_read": False,
        },
        "claim_boundary": "privacy-safe direct hypergraph only; no MILP solved and no allocation selected",
    }
    # Close the long-scan TOCTOU window before any caller may publish.
    verify_bindings(config)
    verify_registry_inputs_unchanged(registry_config, Path(config["bindings"]["registry_config"]["path"]).parent)
    return record_rows, hyperedges, graph_report, build_solver_contract(config)


def _implementation(root: Path, config_path: Path) -> dict[str, Any]:
    bytecode = sorted([
        *root.rglob("*.pyc"),
        *(path for path in root.rglob("__pycache__") if path.is_dir()),
    ])
    if bytecode:
        raise ProtocolError(f"bytecode/cache files forbidden from frozen execution tree: {bytecode}")
    for directory_name in ("src", "scripts", "tests"):
        directory = root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise ProtocolError(f"execution directory must be a regular directory: {directory}")
        unexpected = sorted(
            child for child in directory.iterdir()
            if child.is_symlink() or child.is_dir() or child.suffix != ".py"
        )
        if unexpected:
            raise ProtocolError(f"unexpected execution-tree member: {unexpected}")

    def bound(path: Path) -> dict[str, Any]:
        require_regular_file(path)
        return file_spec(path)

    return {
        "config": bound(config_path),
        "README": bound(root / "README.md"),
        "pytest_ini": bound(root / "pytest.ini"),
        "sources": {
            path.name: bound(path) for path in sorted((root / "src").glob("*.py"))
        },
        "scripts": {
            path.name: bound(path) for path in sorted((root / "scripts").glob("*.py"))
        },
        "tests": {
            path.name: bound(path) for path in sorted((root / "tests").glob("*.py"))
        },
    }


def run_graph(config_path: Path, root: Path) -> dict[str, Any]:
    config = strict_json_loads(config_path.read_text())
    required_hash_seed = config["deterministic_processes"]["graph_build_python_hash_seed"]
    if os.environ.get("PYTHONHASHSEED") != required_hash_seed:
        raise ProtocolError("formal graph build PYTHONHASHSEED mismatch")
    empty_prefix = Path(config["deterministic_processes"]["empty_python_pycache_prefix"])
    if os.environ.get("PYTHONPYCACHEPREFIX") != str(empty_prefix) or empty_prefix.exists():
        raise ProtocolError("formal graph build requires the frozen absent pycache prefix")
    observed_runtime = verify_runtime_identity(config["runtime_identity"])
    implementation_start = _implementation(root, config_path)
    output_dir = safe_relative_output(root, config["outputs"]["graph_artifact_dir"])
    claim_artifact(output_dir, {"protocol_id": config["protocol_id"], "artifact_kind": "direct_hypergraph"})

    def progress(value: Mapping[str, Any]) -> None:
        print(json.dumps(dict(value), sort_keys=True), flush=True)

    records, hyperedges, report, contract = build_graph_payloads(config, progress=progress)
    atomic_jsonl(output_dir / "records.jsonl", records)
    atomic_jsonl(output_dir / "hyperedges.jsonl", hyperedges)
    atomic_json(output_dir / "graph_report.json", report)
    atomic_json(output_dir / "solver_contract.json", contract)
    implementation = _implementation(root, config_path)
    if not json_exact(implementation, implementation_start):
        raise ProtocolError("implementation changed during graph build")
    manifest = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "artifact_kind": "direct_hypergraph",
        "status": "prepared_privacy_safe_reusable_hypergraph",
        "bindings": config["bindings"],
        "implementation": implementation,
        "runtime": {
            **observed_runtime,
            "python_hash_seed": required_hash_seed,
            "python_pycache_prefix_absent": str(empty_prefix),
            "process_id": os.getpid(),
        },
        "files": {
            "records.jsonl": {**file_spec(output_dir / "records.jsonl"), "rows": len(records)},
            "hyperedges.jsonl": {**file_spec(output_dir / "hyperedges.jsonl"), "rows": len(hyperedges)},
            "graph_report.json": file_spec(output_dir / "graph_report.json"),
            "solver_contract.json": file_spec(output_dir / "solver_contract.json"),
        },
        "deep_validation_required_before_allocation": True,
        "deep_validation_receipt_path": str(safe_relative_output(root, config["outputs"]["graph_deep_receipt"])),
        "completion_marker_required": "PUBLISHED.json",
    }
    atomic_json(output_dir / "manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "artifact_kind": "direct_hypergraph",
        "status": "pass_graph_built_no_allocation",
        "cpu_only": True,
        "record_count": len(records),
        "hyperedge_count": len(hyperedges),
        "manifest": file_spec(output_dir / "manifest.json"),
        "solver_contract": file_spec(output_dir / "solver_contract.json"),
        "component_v1_failure": config["bindings"]["component_v1_failure"],
        "required_commands": config["commands"],
        "claim_boundary": "graph and reusable solver contract only; no feasibility or allocation result",
    }
    atomic_json(output_dir / "receipt.json", receipt)
    if not json_exact(_implementation(root, config_path), implementation):
        raise ProtocolError("implementation changed before graph completion commit")
    payloads = [
        "records.jsonl", "hyperedges.jsonl", "graph_report.json",
        "solver_contract.json", "manifest.json", "receipt.json",
    ]
    marker = finalize_publication(
        output_dir,
        payloads,
        identity={"protocol_id": config["protocol_id"], "artifact_kind": "direct_hypergraph"},
        status="published_complete_direct_hypergraph",
    )
    return {
        "status": receipt["status"],
        "artifact": str(output_dir),
        "records": len(records),
        "hyperedges": len(hyperedges),
        "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
        "ordered_inventory_sha256": marker["ordered_inventory_sha256"],
    }
