from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx


class FeasibilityError(RuntimeError):
    pass


def load_registry_core(path: Path):
    spec = importlib.util.spec_from_file_location("registry_v2_frozen_core", path)
    if spec is None or spec.loader is None:
        raise FeasibilityError(f"cannot load frozen registry core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(core: Any, path: Path, value: Any) -> None:
    core.atomic_json(path, value)


def atomic_jsonl(core: Any, path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    core.atomic_jsonl(path, rows)


def file_spec(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"nbytes": path.stat().st_size, "sha256": sha256_file(path)}
    if include_path:
        result = {"path": str(path), **result}
    return result


def verify_file(path: Path, spec: Mapping[str, Any]) -> None:
    observed = file_spec(path, include_path=False) if path.is_file() else None
    expected = {"nbytes": spec["nbytes"], "sha256": spec["sha256"]}
    if observed != expected:
        raise FeasibilityError(f"pinned file mismatch: {path}")


def verify_sidecar(sidecar_path: Path, payload_path: Path, payload_spec: Mapping[str, Any]) -> None:
    value = json.loads(sidecar_path.read_text())
    expected = {
        "path": str(payload_path),
        "nbytes": int(payload_spec["nbytes"]),
        "sha256": str(payload_spec["sha256"]),
    }
    if value != expected:
        raise FeasibilityError(f"pinned sidecar semantic mismatch: {sidecar_path}")


@dataclass
class UnionFind:
    parent: list[int]
    size: list[int]

    @classmethod
    def create(cls, count: int) -> "UnionFind":
        return cls(list(range(count)), [1] * count)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if (self.size[left_root], -left_root) < (self.size[right_root], -right_root):
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]
        return True


def _progress(callback: Callable[[Mapping[str, Any]], None] | None, **value: Any) -> None:
    if callback is not None:
        callback(value)


def _union_first_by_key(
    rows: Sequence[Any],
    uf: UnionFind,
    key: Callable[[Any], str],
) -> tuple[int, int]:
    first: dict[str, int] = {}
    duplicate_records = 0
    union_operations = 0
    for index, row in enumerate(rows):
        value = key(row)
        if value in first:
            duplicate_records += 1
            union_operations += int(uf.union(first[value], index))
        else:
            first[value] = index
    return duplicate_records, union_operations


def _unique_normalized_units(
    rows: Sequence[Any],
    uf: UnionFind,
    core: Any,
    *,
    passages: bool,
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[list[str], list[int], dict[str, int]]:
    first_by_hash: dict[str, int] = {}
    texts: list[str] = []
    owners: list[int] = []
    occurrences = 0
    exact_duplicates = 0
    exact_union_operations = 0
    for owner, row in enumerate(rows):
        values = row.passages if passages else (row.question,)
        seen_in_record: set[str] = set()
        for raw in values:
            occurrences += 1
            normalized = core.normalize(raw)
            normalized_hash = core.digest(normalized)
            if normalized_hash in seen_in_record:
                continue
            seen_in_record.add(normalized_hash)
            existing = first_by_hash.get(normalized_hash)
            if existing is None:
                first_by_hash[normalized_hash] = len(texts)
                texts.append(normalized)
                owners.append(owner)
            else:
                exact_duplicates += 1
                exact_union_operations += int(uf.union(owners[existing], owner))
        if owner and owner % 10000 == 0:
            _progress(
                progress,
                phase="collapse_passages" if passages else "collapse_questions",
                records_processed=owner,
                unique_units=len(texts),
            )
    return texts, owners, {
        "occurrences": occurrences,
        "unique_normalized_units": len(texts),
        "exact_duplicate_occurrences": exact_duplicates,
        "exact_union_operations": exact_union_operations,
    }


def _union_similarity_units(
    texts: Sequence[str],
    owners: Sequence[int],
    uf: UnionFind,
    core: Any,
    cfg: Mapping[str, Any],
    *,
    question: bool,
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> dict[str, int]:
    width = cfg["question_shingle_words"] if question else cfg["passage_shingle_words"]
    minimum_intersection = cfg["question_min_intersection"] if question else cfg["passage_min_intersection"]
    minimum_containment = cfg["question_min_containment"] if question else cfg["passage_min_containment"]
    minimum_jaccard = cfg["question_min_jaccard"] if question else cfg["passage_min_jaccard"]
    substring_minimum = cfg["exact_normalized_unit_substring_min_characters"]
    index = core.build_unit_index(texts, width, substring_minimum)
    relation_counts: Counter[str] = Counter()
    union_operations = 0
    candidates_verified = 0
    for left in range(len(texts)):
        candidate_set = index.sets[left]
        counts: Counter[int] = Counter()
        for token in sorted(candidate_set):
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
                kind = "normalized_substring"
            elif right in counts:
                intersection = counts[right]
                denominator = min(len(candidate_set), len(index.sets[right]))
                containment = intersection / denominator if denominator else 0.0
                union = len(candidate_set) + len(index.sets[right]) - intersection
                jaccard = intersection / union if union else 0.0
                if intersection >= minimum_intersection and (
                    containment >= minimum_containment or jaccard >= minimum_jaccard
                ):
                    kind = "near_shingle"
            if kind is not None:
                relation_counts[kind] += 1
                union_operations += int(uf.union(owners[left], owners[right]))
        if left and left % 10000 == 0:
            _progress(
                progress,
                phase="question_similarity" if question else "passage_similarity",
                unique_units_processed=left,
                unique_units_total=len(texts),
                candidates_verified=candidates_verified,
                relation_pairs=sum(relation_counts.values()),
            )
    return {
        "unique_units": len(texts),
        "candidates_verified": candidates_verified,
        "normalized_substring_pairs": relation_counts["normalized_substring"],
        "near_shingle_pairs": relation_counts["near_shingle"],
        "union_operations": union_operations,
    }


def build_components(
    rows: Sequence[Any],
    cfg: Mapping[str, Any],
    core: Any,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[UnionFind, dict[str, Any]]:
    uf = UnionFind.create(len(rows))
    exact: dict[str, Any] = {}
    for label, key in (
        ("raw_id", lambda row: row.sample_id),
        ("raw_question", lambda row: core.digest(row.question)),
        ("normalized_question", lambda row: core.digest(core.normalize(row.question))),
        (
            "normalized_question_context",
            lambda row: core.digest(
                core.normalize(row.question) + "\u241f" + core.normalize("\n\n".join(row.passages))
            ),
        ),
    ):
        duplicate_records, operations = _union_first_by_key(rows, uf, key)
        exact[label] = {"duplicate_records": duplicate_records, "union_operations": operations}
    question_texts, question_owners, question_exact = _unique_normalized_units(
        rows, uf, core, passages=False, progress=progress
    )
    passage_texts, passage_owners, passage_exact = _unique_normalized_units(
        rows, uf, core, passages=True, progress=progress
    )
    question_similarity = _union_similarity_units(
        question_texts, question_owners, uf, core, cfg, question=True, progress=progress
    )
    passage_similarity = _union_similarity_units(
        passage_texts, passage_owners, uf, core, cfg, question=False, progress=progress
    )
    return uf, {
        "exact_record_relations": exact,
        "normalized_question_units": question_exact,
        "normalized_passage_units": passage_exact,
        "question_similarity": question_similarity,
        "passage_similarity": passage_similarity,
    }


def assert_no_residual_question_collisions(relation_stats: Mapping[str, Any]) -> dict[str, Any]:
    exact = relation_stats["exact_record_relations"]
    values = {
        "raw_question_duplicate_records": exact["raw_question"]["duplicate_records"],
        "normalized_question_duplicate_records": exact["normalized_question"]["duplicate_records"],
        "normalized_question_context_duplicate_records": exact["normalized_question_context"]["duplicate_records"],
        "normalized_question_unit_duplicate_occurrences": relation_stats["normalized_question_units"]["exact_duplicate_occurrences"],
        "normalized_question_substring_pairs": relation_stats["question_similarity"]["normalized_substring_pairs"],
        "near_question_pairs": relation_stats["question_similarity"]["near_shingle_pairs"],
    }
    if any(values.values()):
        raise FeasibilityError(
            "eligible pool retained a question collision that train-union quarantine should remove"
        )
    return {"pass": True, **values}


def component_inventory(
    rows: Sequence[Any],
    uf: UnionFind,
    cells_order: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[int, int], dict[str, Any]]:
    cell_index = {cell: index for index, cell in enumerate(cells_order)}
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        members[uf.find(index)].append(index)
    inventory = []
    for member_indices in members.values():
        projection_hashes = sorted(rows[index].projection_sha256 for index in member_indices)
        component_id = hashlib.sha256(canonical_bytes(projection_hashes)).hexdigest()
        representatives: dict[str, str] = {}
        member_count_by_cell: Counter[str] = Counter()
        for index in member_indices:
            row = rows[index]
            cell = f"{row.dataset_id}|{row.stratum}"
            if cell not in cell_index:
                raise FeasibilityError(f"unexpected eligible cell: {cell}")
            member_count_by_cell[cell] += 1
            representatives[cell] = min(representatives.get(cell, row.projection_sha256), row.projection_sha256)
        eligible_cells = [cell for cell in cells_order if cell in representatives]
        mask = sum(1 << cell_index[cell] for cell in eligible_cells)
        inventory.append({
            "component_id": component_id,
            "eligible_cell_mask": mask,
            "eligible_cells": eligible_cells,
            "member_record_count": len(member_indices),
            "member_record_count_by_cell": {
                cell: member_count_by_cell[cell] for cell in eligible_cells
            },
            "representative_projection_sha256_by_cell": {
                cell: representatives[cell] for cell in eligible_cells
            },
        })
    inventory.sort(key=lambda value: value["component_id"])
    mask_counts, stats = statistics_from_inventory(inventory, cells_order)
    return inventory, mask_counts, stats


def statistics_from_inventory(
    inventory: Sequence[Mapping[str, Any]],
    cells_order: Sequence[str],
) -> tuple[dict[int, int], dict[str, Any]]:
    mask_counts: Counter[int] = Counter()
    supporting_global_sizes: dict[str, list[int]] = defaultdict(list)
    within_cell_sizes: dict[str, list[int]] = defaultdict(list)
    for row in inventory:
        mask = int(row["eligible_cell_mask"])
        size = int(row["member_record_count"])
        mask_counts[mask] += 1
        for cell in row["eligible_cells"]:
            supporting_global_sizes[cell].append(size)
            within_cell_sizes[cell].append(int(row["member_record_count_by_cell"][cell]))

    def distribution(values: Sequence[int]) -> dict[str, Any]:
        ordered = sorted(values)
        if not ordered:
            return {"components": 0, "max": 0, "p50": 0, "p90": 0, "p99": 0}
        percentile = lambda fraction: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]
        return {
            "components": len(ordered),
            "max": ordered[-1],
            "p50": percentile(.50),
            "p90": percentile(.90),
            "p99": percentile(.99),
        }

    global_sizes = [row["member_record_count"] for row in inventory]
    stats = {
        "eligible_records": sum(global_sizes),
        "connected_components": len(inventory),
        "multi_record_components": sum(value > 1 for value in global_sizes),
        "global_size_distribution": distribution(global_sizes),
        "supporting_component_global_size_distribution_by_cell": {
            cell: distribution(supporting_global_sizes[cell]) for cell in cells_order
        },
        "within_cell_member_count_distribution_by_cell": {
            cell: distribution(within_cell_sizes[cell]) for cell in cells_order
        },
        "supporting_component_count_by_cell": {
            cell: len(supporting_global_sizes[cell]) for cell in cells_order
        },
        "component_mask_counts": {
            str(mask): count for mask, count in sorted(mask_counts.items())
        },
    }
    return dict(mask_counts), stats


def max_flow_feasibility(
    mask_counts: Mapping[int, int],
    cells_order: Sequence[str],
    quotas: Mapping[str, Mapping[str, int]],
    *,
    buffer_cell: str,
    minimum_unselected_buffer_components: int,
) -> dict[str, Any]:
    source, sink = "source", "sink"
    graph = nx.DiGraph()
    if buffer_cell not in cells_order:
        raise FeasibilityError(f"buffer cell is not registered: {buffer_cell}")
    if minimum_unselected_buffer_components < 0:
        raise FeasibilityError("component buffer cannot be negative")
    buffer_bit = 1 << cells_order.index(buffer_cell)
    buffer_supporting_components = sum(
        int(count) for mask, count in mask_counts.items() if mask & buffer_bit
    )
    if buffer_supporting_components < minimum_unselected_buffer_components:
        raise FeasibilityError("fewer buffer-supporting components than required reserve")
    buffer_gate = f"buffer_gate|{buffer_cell}"
    graph.add_edge(
        source,
        buffer_gate,
        capacity=buffer_supporting_components - minimum_unselected_buffer_components,
    )
    target_quotas: dict[str, int] = {}
    for partition in ("screen", "confirm"):
        for cell in cells_order:
            target = f"target|{partition}|{cell}"
            quota = int(quotas[partition].get(cell, 0))
            target_quotas[target] = quota
            graph.add_edge(target, sink, capacity=quota)
    for mask, count in sorted(mask_counts.items()):
        node = f"mask|{mask}"
        if mask & buffer_bit:
            # Every use of a component that can supply the buffer cell passes
            # through one shared gate, even if flow assigns it to another cell.
            # Thus the reserved components are completely unused, rather than
            # merely absent from the buffer cell's own target.
            graph.add_edge(buffer_gate, node, capacity=int(count))
        else:
            graph.add_edge(source, node, capacity=int(count))
        for cell_index, cell in enumerate(cells_order):
            if mask & (1 << cell_index):
                for partition in ("screen", "confirm"):
                    graph.add_edge(node, f"target|{partition}|{cell}", capacity=int(count))
    value, flow = nx.maximum_flow(
        graph, source, sink, flow_func=nx.algorithms.flow.edmonds_karp
    )
    requested = sum(target_quotas.values())
    buffer_components_used = int(flow.get(source, {}).get(buffer_gate, 0))
    fulfilled = {
        target.removeprefix("target|"): int(flow.get(target, {}).get(sink, 0))
        for target in sorted(target_quotas)
    }
    deficits = {
        target.removeprefix("target|"): target_quotas[target] - fulfilled[target.removeprefix("target|")]
        for target in sorted(target_quotas)
        if target_quotas[target] != fulfilled[target.removeprefix("target|")]
    }
    mask_to_targets = {
        node.removeprefix("mask|"): {
            target.removeprefix("target|"): int(amount)
            for target, amount in sorted(edges.items())
            if target.startswith("target|") and amount
        }
        for node, edges in sorted(flow.items())
        if node.startswith("mask|")
    }
    return {
        "pass": value == requested,
        "requested_records": requested,
        "maximum_flow_records": int(value),
        "fulfilled_by_target": fulfilled,
        "deficits_by_target": deficits,
        "flow_from_component_masks": mask_to_targets,
        "component_buffer": {
            "cell": buffer_cell,
            "supporting_components": buffer_supporting_components,
            "minimum_completely_unselected": minimum_unselected_buffer_components,
            "maximum_allowed_used": buffer_supporting_components - minimum_unselected_buffer_components,
            "used_by_flow": buffer_components_used,
            "completely_unselected_under_flow": buffer_supporting_components - buffer_components_used,
            "applies_when_component_assigned_to_any_target": True,
        },
    }


def _max_min_waterfill(
    total: int,
    cells: Sequence[str],
    capacities: Mapping[str, int],
) -> dict[str, int] | None:
    allocation = {cell: 0 for cell in cells}
    for _ in range(total):
        available = [cell for cell in cells if allocation[cell] < capacities[cell]]
        if not available:
            return None
        chosen = min(available, key=lambda cell: (allocation[cell], cells.index(cell)))
        allocation[chosen] += 1
    return allocation


def _split_one_to_two(
    combined: Mapping[str, int], cells: Sequence[str]
) -> tuple[dict[str, int], dict[str, int]]:
    total = sum(combined.values())
    if total % 3:
        raise FeasibilityError("combined balanced quota must be divisible by three")
    screen_total = total // 3
    screen = {cell: combined[cell] // 3 for cell in cells}
    additions = screen_total - sum(screen.values())
    candidates = sorted(
        (cell for cell in cells if combined[cell] % 3),
        key=lambda cell: (-(combined[cell] % 3), cells.index(cell)),
    )
    if additions > len(candidates):
        raise FeasibilityError("cannot split combined quota into exact 1:2 partitions")
    for cell in candidates[:additions]:
        screen[cell] += 1
    confirm = {cell: combined[cell] - screen[cell] for cell in cells}
    return screen, confirm


def balanced_common_scale_quotas(
    h_value: int,
    cells_order: Sequence[str],
    component_capacities: Mapping[str, int],
    *,
    buffer_cell: str | None = None,
    minimum_unselected_buffer_components: int = 2,
) -> dict[str, dict[str, int]] | None:
    hotpot_cells = list(cells_order[:6])
    musique_cells = list(cells_order[6:])
    hotpot_combined = _max_min_waterfill(3 * h_value, hotpot_cells, component_capacities)
    musique_capacities = {cell: component_capacities[cell] for cell in musique_cells}
    effective_buffer_cell = buffer_cell or musique_cells[-1]
    if effective_buffer_cell not in musique_capacities:
        raise FeasibilityError("balanced-family buffer cell must be a MuSiQue cell")
    musique_capacities[effective_buffer_cell] = max(
        0,
        musique_capacities[effective_buffer_cell] - minimum_unselected_buffer_components,
    )
    musique_combined = _max_min_waterfill(6 * h_value, musique_cells, musique_capacities)
    if hotpot_combined is None or musique_combined is None:
        return None
    hotpot_screen, hotpot_confirm = _split_one_to_two(hotpot_combined, hotpot_cells)
    musique_screen, musique_confirm = _split_one_to_two(musique_combined, musique_cells)
    return {
        "screen": {**hotpot_screen, **musique_screen},
        "confirm": {**hotpot_confirm, **musique_confirm},
    }


def search_maximum_balanced(
    mask_counts: Mapping[int, int],
    cells_order: Sequence[str],
    component_capacities: Mapping[str, int],
    *,
    buffer_cell: str,
    minimum_unselected_buffer_components: int,
) -> dict[str, Any]:
    for h_value in range(128, -1, -1):
        quotas = balanced_common_scale_quotas(
            h_value,
            cells_order,
            component_capacities,
            buffer_cell=buffer_cell,
            minimum_unselected_buffer_components=minimum_unselected_buffer_components,
        )
        if quotas is None:
            continue
        flow = max_flow_feasibility(
            mask_counts,
            cells_order,
            quotas,
            buffer_cell=buffer_cell,
            minimum_unselected_buffer_components=minimum_unselected_buffer_components,
        )
        if flow["pass"]:
            return {
                "H": h_value,
                "hotpot_screen_total": h_value,
                "hotpot_confirm_total": 2 * h_value,
                "musique_screen_total": 2 * h_value,
                "musique_confirm_total": 4 * h_value,
                "quotas": quotas,
                "flow": flow,
                "minimum_completely_unselected_buffer_components": minimum_unselected_buffer_components,
            }
    raise FeasibilityError("zero-size balanced max-flow unexpectedly infeasible")


def make_report(
    config: Mapping[str, Any],
    registry_config: Mapping[str, Any],
    relation_stats: Mapping[str, Any],
    component_stats: Mapping[str, Any],
    original_flow: Mapping[str, Any],
    maximum_balanced: Mapping[str, Any],
    quarantine: Mapping[str, Any],
) -> dict[str, Any]:
    residual_question_assertion = assert_no_residual_question_collisions(relation_stats)
    return {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "feasibility_computed_not_registry",
        "component_policy": config["component_policy"],
        "component_buffer": config["component_buffer"],
        "thresholds": registry_config["decontamination"],
        "relation_statistics": relation_stats,
        "eligible_question_collision_residual_assertion": residual_question_assertion,
        "component_statistics": component_stats,
        "original_quotas": config["original_quotas"],
        "original_quota_flow": original_flow,
        "maximum_balanced_common_scale": maximum_balanced,
        "decontamination_aggregate": quarantine,
        "privacy": {
            "component_inventory_contains_raw_id_question_context_answer_or_outcome": False,
            "answer_values_available_to_component_or_flow_logic": False,
            "model_outcomes_read": False,
        },
        "claim_boundary": "CPU-only feasibility; no sample registry selected or published",
    }


def verify_attempt3_failure(path: Path, audit_root: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    expected = {
        "attempt_id": "screen_confirm_sequence_capacity_fail_attempt3",
        "status": "failed_closed_preserved_append_only",
        "error": "fail closed: insufficient eligible pool for confirm/3hop: 40 < 217",
        "remaining_eligible_in_failed_stratum": 40,
        "requested_confirm_quota": 217,
        "completion_marker_present": False,
        "published": False,
        "thresholds_changed": False,
        "gpu_used": False,
        "model_outcomes_read": False,
        "answer_values_available_to_partition_or_quota_logic": False,
        "raw_answer_values_transiently_parsed_into_decoder_object_pairs": True,
        "formal_registry_files_written": 0,
        "partition": "confirm",
        "stratum": "3hop",
        "capacity_assertion": {
            "observed_clean_4hop_capacity": 119,
            "passed": True,
            "reserved_clean_4hop_buffer": 2,
        },
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise FeasibilityError("attempt3 failure evidence semantic mismatch")
    log_path = audit_root / value["log"]["path"]
    verify_file(log_path, value["log"])
    if value.get("config") != {
        "nbytes": 8324,
        "sha256": "56c62715793041df3fb532a2a14cf4fd5748be6c086eb7fbe87059beaa6d0d09",
    } or value.get("source") != {
        "nbytes": 54002,
        "sha256": "c6f2e75388276c1fb22b976610949a8cbd42c16f9cb82ddcb903d598f3de78de",
    }:
        raise FeasibilityError("attempt3 frozen implementation binding mismatch")
    failed_artifact = audit_root / value["artifact_dir"]
    if not failed_artifact.is_dir() or sorted(child.name for child in failed_artifact.iterdir()) != ["IN_PROGRESS.json"]:
        raise FeasibilityError("attempt3 failed artifact inventory mismatch")
    verify_file(failed_artifact / "IN_PROGRESS.json", value["in_progress_marker"])
    return value


def prepare_eligible(
    core: Any,
    registry_config: Mapping[str, Any],
    audit_root: Path,
    progress: Callable[[Mapping[str, Any]], None] | None,
) -> tuple[list[Any], dict[str, Any]]:
    core.verify_threshold_binding(registry_config)
    for spec in registry_config["inputs"].values():
        core.verify_file(Path(spec["path"]), spec)
    failure_09 = registry_config["failure_evidence"]["complete_g0_v1_receipt"]
    core.verify_file(Path(failure_09["path"]), failure_09)
    core.verify_failure_evidence_semantics(Path(failure_09["path"]))
    attempt2 = registry_config["failure_evidence"]["local_strict_substring_capacity_attempt"]
    attempt2_evidence_path = audit_root / attempt2["path"]
    core.verify_file(attempt2_evidence_path, attempt2)
    core.verify_capacity_failure_semantics(attempt2_evidence_path, audit_root)

    inputs = registry_config["inputs"]
    _progress(progress, phase="load_mapper")
    mapper_texts = core.load_mapper_texts(registry_config)
    _progress(progress, phase="load_sources")
    hotpot_train = core.load_hotpot(Path(inputs["hotpot_train"]["path"]), inputs["hotpot_train"]["expected_records"], locked=False)
    hotpot_locked = core.load_hotpot(Path(inputs["hotpot_locked_dev"]["path"]), inputs["hotpot_locked_dev"]["expected_records"], locked=True)
    musique_train = core.load_musique(Path(inputs["musique_train"]["path"]), inputs["musique_train"]["expected_records"], locked=False)
    musique_locked = core.load_musique(Path(inputs["musique_locked_dev"]["path"]), inputs["musique_locked_dev"]["expected_records"], locked=True)
    cfg = registry_config["decontamination"]
    _progress(progress, phase="build_locked_mapper_indexes")
    locked_indexes = {
        "hotpotqa_locked_dev": core.build_reference_index(hotpot_locked, cfg),
        "musique_locked_dev": core.build_reference_index(musique_locked, cfg),
    }
    locked_counts = {"hotpotqa_locked_dev": len(hotpot_locked), "musique_locked_dev": len(musique_locked)}
    mapper_question = core.build_mapper_index(mapper_texts, cfg, question=True)
    mapper_passage = core.build_mapper_index(mapper_texts, cfg, question=False)
    _progress(progress, phase="train_union_question_collisions")
    collisions, collision_pairs = core.train_union_question_collisions([*hotpot_train, *musique_train], cfg)
    _progress(progress, phase="decontaminate_hotpot")
    hotpot_eligible, hotpot_quarantine = core.decontaminate(
        hotpot_train, locked_indexes, locked_counts, mapper_question, mapper_passage, cfg, collisions
    )
    _progress(progress, phase="decontaminate_musique")
    musique_eligible, musique_quarantine = core.decontaminate(
        musique_train, locked_indexes, locked_counts, mapper_question, mapper_passage, cfg, collisions
    )
    observed_4hop = musique_quarantine["eligible_by_stratum"].get("4hop", 0)
    if observed_4hop != 119:
        raise FeasibilityError(f"strict clean 4hop capacity changed: {observed_4hop} != 119")
    return [*hotpot_eligible, *musique_eligible], {
        "hotpotqa": hotpot_quarantine,
        "musique": musique_quarantine,
        "train_union_question_collision_pairs": collision_pairs,
        "train_union_question_collision_records": len(collisions),
    }


def run(config_path: Path, root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    bindings = config["bindings"]
    for spec in bindings.values():
        verify_file(Path(spec["path"]), spec)
    verify_sidecar(
        Path(bindings["attempt2_capacity_failure_sidecar"]["path"]),
        Path(bindings["attempt2_capacity_failure"]["path"]),
        bindings["attempt2_capacity_failure"],
    )
    verify_sidecar(
        Path(bindings["attempt3_sequence_failure_sidecar"]["path"]),
        Path(bindings["attempt3_sequence_failure"]["path"]),
        bindings["attempt3_sequence_failure"],
    )
    core = load_registry_core(Path(bindings["registry_source"]["path"]))
    registry_config = json.loads(Path(bindings["registry_config"]["path"]).read_text())
    audit_root = Path(bindings["registry_config"]["path"]).parent
    verify_attempt3_failure(Path(bindings["attempt3_sequence_failure"]["path"]), audit_root)

    relative_output = Path(config["outputs"]["artifact_dir"])
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise FeasibilityError("artifact directory must stay below the feasibility root")
    output_dir = root / relative_output
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.single_writer.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise FeasibilityError("another feasibility writer is active") from error
    try:
        if output_dir.exists():
            raise FeasibilityError(f"refusing to reuse existing artifact directory: {output_dir}")
        output_dir.mkdir(parents=True)
        atomic_json(core, output_dir / "IN_PROGRESS.json", {
            "schema_version": 1,
            "audit_id": config["audit_id"],
            "status": "in_progress_not_published",
            "completion_marker_required": "PUBLISHED.json",
        })
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()

    def progress(value: Mapping[str, Any]) -> None:
        print(json.dumps(dict(value), sort_keys=True), flush=True)

    eligible, quarantine = prepare_eligible(core, registry_config, audit_root, progress)
    progress({"phase": "build_components", "eligible_records": len(eligible)})
    uf, relation_stats = build_components(
        eligible, registry_config["decontamination"], core, progress=progress
    )
    inventory, mask_counts, component_stats = component_inventory(
        eligible, uf, config["cells_order"]
    )
    buffer_config = config["component_buffer"]
    buffer_cell = buffer_config["cell"]
    buffer_reserve = int(buffer_config["minimum_completely_unselected_supporting_components"])
    original_flow = max_flow_feasibility(
        mask_counts,
        config["cells_order"],
        config["original_quotas"],
        buffer_cell=buffer_cell,
        minimum_unselected_buffer_components=buffer_reserve,
    )
    maximum_balanced = search_maximum_balanced(
        mask_counts,
        config["cells_order"],
        component_stats["supporting_component_count_by_cell"],
        buffer_cell=buffer_cell,
        minimum_unselected_buffer_components=buffer_reserve,
    )
    report = make_report(
        config,
        registry_config,
        relation_stats,
        component_stats,
        original_flow,
        maximum_balanced,
        quarantine,
    )
    atomic_jsonl(core, output_dir / "component_inventory.jsonl", inventory)
    atomic_json(core, output_dir / "feasibility_report.json", report)
    implementation = {
        "config": file_spec(config_path),
        "source": file_spec(root / "src" / "feasibility.py"),
        "validation_source": file_spec(root / "src" / "validate.py"),
        "entrypoint": file_spec(root / "scripts" / "run.py"),
        "validator": file_spec(root / "scripts" / "validate.py"),
        "tests": {
            path.name: file_spec(path) for path in sorted((root / "tests").glob("*.py"))
        },
    }
    manifest = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "prepared_feasibility_only_not_registry",
        "bindings": bindings,
        "implementation": implementation,
        "files": {
            "component_inventory.jsonl": {**file_spec(output_dir / "component_inventory.jsonl"), "rows": len(inventory)},
            "feasibility_report.json": file_spec(output_dir / "feasibility_report.json"),
        },
        "original_quota_feasible": original_flow["pass"],
        "maximum_balanced_H": maximum_balanced["H"],
        "completion_marker_required": "PUBLISHED.json",
    }
    atomic_json(core, output_dir / "manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "pass_feasibility_computed",
        "cpu_only": True,
        "not_a_sample_registry": True,
        "component_count": len(inventory),
        "original_quota_feasible": original_flow["pass"],
        "original_quota_maximum_flow_records": original_flow["maximum_flow_records"],
        "original_quota_requested_records": original_flow["requested_records"],
        "maximum_balanced_common_scale": maximum_balanced,
        "manifest": file_spec(output_dir / "manifest.json"),
        "required_post_publication_validation": {
            "structural_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/validate.py",
            "deep_recompute_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/validate.py --deep",
            "deep_recomputes_clean_pool_components_inventory_and_report_from_pinned_sources": True,
        },
        "completion_marker_required": "PUBLISHED.json",
        "claim_boundary": "feasibility only; a later registry needs a separately frozen allocation algorithm and full validation",
    }
    atomic_json(core, output_dir / "receipt.json", receipt)
    payloads = ["component_inventory.jsonl", "feasibility_report.json", "manifest.json", "receipt.json"]
    for name in payloads:
        atomic_json(core, output_dir / f"{name}.sha256.json", file_spec(output_dir / name))
    (output_dir / "IN_PROGRESS.json").unlink()
    expected = sorted([*payloads, *(f"{name}.sha256.json" for name in payloads)])
    actual = sorted(path.name for path in output_dir.iterdir())
    if actual != expected:
        raise FeasibilityError("pre-publication inventory mismatch")
    published_files = {
        name: file_spec(output_dir / name, include_path=False) for name in expected
    }
    ordered = [{"name": name, **published_files[name]} for name in expected]
    publication = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "published_complete_feasibility_only",
        "atomic_last_completion_marker": True,
        "published_files": published_files,
        "ordered_inventory_sha256": hashlib.sha256(canonical_bytes(ordered)).hexdigest(),
        "receipt": published_files["receipt.json"],
        "receipt_sidecar": published_files["receipt.json.sha256.json"],
    }
    atomic_json(core, output_dir / "PUBLISHED.json", publication)
    return {
        "status": receipt["status"],
        "artifact": str(output_dir),
        "component_count": len(inventory),
        "original_quota_feasible": original_flow["pass"],
        "maximum_balanced_H": maximum_balanced["H"],
        "publication": file_spec(output_dir / "PUBLISHED.json"),
    }
