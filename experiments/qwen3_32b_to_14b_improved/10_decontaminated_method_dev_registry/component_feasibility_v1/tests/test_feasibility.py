from __future__ import annotations

import json
import hashlib
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.feasibility import (
    UnionFind,
    _split_one_to_two,
    balanced_common_scale_quotas,
    build_components,
    canonical_bytes,
    component_inventory,
    load_registry_core,
    make_report,
    max_flow_feasibility,
    search_maximum_balanced,
    statistics_from_inventory,
    verify_attempt3_failure,
    verify_sidecar,
)
from src.validate import validate


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT.parent
CONFIG = json.loads((ROOT / "config.json").read_text())
REGISTRY_CONFIG = json.loads(Path(CONFIG["bindings"]["registry_config"]["path"]).read_text())
CELLS = CONFIG["cells_order"]


ZERO_RELATION_STATS = {
    "exact_record_relations": {
        "raw_id": {"duplicate_records": 0, "union_operations": 0},
        "raw_question": {"duplicate_records": 0, "union_operations": 0},
        "normalized_question": {"duplicate_records": 0, "union_operations": 0},
        "normalized_question_context": {"duplicate_records": 0, "union_operations": 0},
    },
    "normalized_question_units": {
        "occurrences": 0,
        "unique_normalized_units": 0,
        "exact_duplicate_occurrences": 0,
        "exact_union_operations": 0,
    },
    "normalized_passage_units": {
        "occurrences": 0,
        "unique_normalized_units": 0,
        "exact_duplicate_occurrences": 0,
        "exact_union_operations": 0,
    },
    "question_similarity": {
        "unique_units": 0,
        "candidates_verified": 0,
        "normalized_substring_pairs": 0,
        "near_shingle_pairs": 0,
        "union_operations": 0,
    },
    "passage_similarity": {
        "unique_units": 0,
        "candidates_verified": 0,
        "normalized_substring_pairs": 0,
        "near_shingle_pairs": 0,
        "union_operations": 0,
    },
}


@pytest.fixture(scope="session")
def core():
    return load_registry_core(Path(CONFIG["bindings"]["registry_source"]["path"]))


def projection(core, index: int, question: str, passages: tuple[str, ...], *, dataset: str = "hotpotqa_train", stratum: str = "bridge/easy", sample_id: str | None = None):
    identity = sample_id or f"id-{index}"
    return core.Projection(
        dataset,
        identity,
        index,
        stratum,
        question,
        passages,
        core.digest(f"projection-{index}-{dataset}-{stratum}-{identity}"),
    )


def test_union_find_transitive_and_deterministic_root():
    uf = UnionFind.create(5)
    assert uf.union(3, 4)
    assert uf.union(1, 2)
    assert uf.union(2, 3)
    assert uf.find(1) == uf.find(4)
    assert not uf.union(1, 4)
    assert uf.size[uf.find(1)] == 4


def test_component_relation_covers_short_exact_substring_width3_width5_near_and_id(core):
    question_ref = " ".join(character * 42 for character in ("a", "b", "c"))
    question_embedded = "prefix" + question_ref + " suffix"
    passage_ref = " ".join(character * 30 for character in ("d", "e", "f", "g", "h"))
    passage_embedded = "prefix" + passage_ref + " suffix"
    near_question = "one two three four five six seven eight nine ten"
    near_passage = " ".join(chr(ord("a") + (index % 20)) for index in range(25))
    rows = [
        projection(core, 0, question_ref, ("unrelated zero",)),
        projection(core, 1, question_embedded, ("unrelated one",)),
        projection(core, 2, "passage left", (passage_ref,)),
        projection(core, 3, "passage right", (passage_embedded,)),
        projection(core, 4, near_question, ("near question left",)),
        projection(core, 5, near_question + " eleven", ("near question right",)),
        projection(core, 6, "near passage left", (near_passage,)),
        projection(core, 7, "near passage right", (near_passage + " z",)),
        projection(core, 8, "Short, EXACT!", ("short exact passage",)),
        projection(core, 9, "short exact", ("different",)),
        projection(core, 10, "id left", ("id passage left",), sample_id="cross-id"),
        projection(core, 11, "id right", ("id passage right",), dataset="musique_train", stratum="2hop", sample_id="cross-id"),
        projection(core, 12, "fully unrelated", ("fully unrelated passage",)),
    ]
    uf, stats = build_components(rows, REGISTRY_CONFIG["decontamination"], core)
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)):
        assert uf.find(left) == uf.find(right)
    assert uf.find(12) not in {uf.find(index) for index in range(12)}
    assert stats["question_similarity"]["normalized_substring_pairs"] >= 1
    assert stats["passage_similarity"]["normalized_substring_pairs"] >= 1
    assert stats["question_similarity"]["near_shingle_pairs"] >= 1
    assert stats["passage_similarity"]["near_shingle_pairs"] >= 1


def test_component_inventory_is_ordered_hashed_and_text_free(core):
    rows = [
        projection(core, 0, "same normalized", ("p0",)),
        projection(core, 1, "SAME---NORMALIZED", ("p1",)),
        projection(core, 2, "other", ("p2",), dataset="musique_train", stratum="2hop"),
    ]
    uf, _ = build_components(rows, REGISTRY_CONFIG["decontamination"], core)
    inventory, masks, stats = component_inventory(rows, uf, CELLS)
    assert [row["component_id"] for row in inventory] == sorted(row["component_id"] for row in inventory)
    assert sum(row["member_record_count"] for row in inventory) == 3
    assert stats["eligible_records"] == 3
    assert stats["connected_components"] == 2
    assert sum(masks.values()) == 2
    serialized = json.dumps(inventory, sort_keys=True)
    for forbidden in ("sample_id", "question", "context", "answer", "outcome", "same normalized"):
        assert forbidden not in serialized


def test_flow_enforces_one_target_per_component():
    cells = ["a", "b"]
    quotas = {"screen": {"a": 1, "b": 0}, "confirm": {"a": 0, "b": 1}}
    result = max_flow_feasibility(
        {3: 1}, cells, quotas, buffer_cell="b", minimum_unselected_buffer_components=0
    )
    assert not result["pass"]
    assert result["requested_records"] == 2
    assert result["maximum_flow_records"] == 1


def test_buffer_gate_reserves_components_even_when_assigned_to_other_cell():
    cells = ["a", "b"]
    quotas = {"screen": {"a": 2, "b": 0}, "confirm": {"a": 0, "b": 1}}
    without_reserve = max_flow_feasibility(
        {1: 1, 3: 2}, cells, quotas, buffer_cell="b", minimum_unselected_buffer_components=0
    )
    with_reserve = max_flow_feasibility(
        {1: 1, 3: 2}, cells, quotas, buffer_cell="b", minimum_unselected_buffer_components=1
    )
    assert without_reserve["pass"]
    assert not with_reserve["pass"]
    assert with_reserve["maximum_flow_records"] == 2
    assert with_reserve["component_buffer"]["completely_unselected_under_flow"] >= 1


def test_one_to_two_split_prefers_remainder_two_then_frozen_order():
    cells = ["a", "b", "c"]
    screen, confirm = _split_one_to_two({"a": 5, "b": 4, "c": 3}, cells)
    assert screen == {"a": 2, "b": 1, "c": 1}
    assert confirm == {"a": 3, "b": 3, "c": 2}
    assert sum(confirm.values()) == 2 * sum(screen.values())


def test_maximum_balanced_known_capacity_reproduces_H128_family():
    counts = {
        **{1 << index: 64 for index in range(6)},
        1 << 6: 326,
        1 << 7: 325,
        1 << 8: 119,
    }
    capacities = {cell: counts[1 << index] for index, cell in enumerate(CELLS)}
    result = search_maximum_balanced(
        counts,
        CELLS,
        capacities,
        buffer_cell=CELLS[-1],
        minimum_unselected_buffer_components=2,
    )
    assert result["H"] == 128
    assert result["flow"]["pass"]
    assert sum(result["quotas"]["screen"][cell] for cell in CELLS[:6]) == 128
    assert sum(result["quotas"]["confirm"][cell] for cell in CELLS[:6]) == 256
    assert [
        result["quotas"]["screen"][cell] + result["quotas"]["confirm"][cell]
        for cell in CELLS[6:]
    ] == [326, 325, 117]
    assert result["flow"]["component_buffer"]["completely_unselected_under_flow"] == 2


def test_balanced_quota_capacity_fail_returns_none():
    capacities = {cell: 0 for cell in CELLS}
    assert balanced_common_scale_quotas(1, CELLS, capacities) is None


def test_attempt2_attempt3_sidecars_and_attempt3_failure_semantics():
    bindings = CONFIG["bindings"]
    for prefix in ("attempt2_capacity_failure", "attempt3_sequence_failure"):
        verify_sidecar(
            Path(bindings[f"{prefix}_sidecar"]["path"]),
            Path(bindings[prefix]["path"]),
            bindings[prefix],
        )
    value = verify_attempt3_failure(
        Path(bindings["attempt3_sequence_failure"]["path"]), AUDIT_ROOT
    )
    assert value["remaining_eligible_in_failed_stratum"] == 40
    assert value["capacity_assertion"]["observed_clean_4hop_capacity"] == 119


def test_max_flow_hashseed_determinism():
    code = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
from src.feasibility import build_components, canonical_bytes, component_inventory, load_registry_core, max_flow_feasibility
r=max_flow_feasibility({{1:7,2:6,3:5}}, ['a','b'], {{'screen':{{'a':4,'b':3}},'confirm':{{'a':4,'b':3}}}}, buffer_cell='b', minimum_unselected_buffer_components=2)
core=load_registry_core(Path({CONFIG['bindings']['registry_source']['path']!r}))
cfg=json.loads(Path({CONFIG['bindings']['registry_config']['path']!r}).read_text())['decontamination']
rows=[
 core.Projection('hotpotqa_train','a',0,'bridge/easy','short exact',('passage a',),core.digest('p0')),
 core.Projection('hotpotqa_train','b',1,'bridge/easy','SHORT---EXACT',('passage b',),core.digest('p1')),
 core.Projection('musique_train','c',2,'2hop','other',(' '.join(x*30 for x in ('d','e','f','g','h')),),core.digest('p2')),
 core.Projection('musique_train','d',3,'2hop','different',('prefix'+(' '.join(x*30 for x in ('d','e','f','g','h')))+' suffix',),core.digest('p3')),
]
uf,stats=build_components(rows,cfg,core)
inventory,masks,component_stats=component_inventory(rows,uf,['hotpotqa_train|bridge/easy','musique_train|2hop'])
print(canonical_bytes({{'flow':r,'inventory':inventory,'masks':masks,'component_stats':component_stats,'relation_stats':stats}}).decode().strip())
"""
    outputs = []
    for seed in ("0", "1", "7", "123456"):
        environment = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, "-c", code], check=True, text=True, capture_output=True, env=environment
        )
        outputs.append(result.stdout)
    assert len(set(outputs)) == 1


def test_structural_and_deep_validator_accept_exact_synthetic_publication(tmp_path, core, monkeypatch):
    replica_root = tmp_path / "component_feasibility_v1"
    for directory in ("src", "scripts", "tests"):
        (replica_root / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "config.json", replica_root / "config.json")
    for relative in (
        "src/__init__.py", "src/feasibility.py", "src/validate.py",
        "scripts/run.py", "scripts/validate.py",
        "tests/conftest.py", "tests/test_feasibility.py",
    ):
        shutil.copy2(ROOT / relative, replica_root / relative)
    config = json.loads((replica_root / "config.json").read_text())
    counts = [65, 65, 64, 64, 64, 64, 326, 325, 119]
    inventory = []
    sequence = 0
    for cell_index, (cell, count) in enumerate(zip(CELLS, counts, strict=True)):
        for _ in range(count):
            projection_hash = hashlib.sha256(f"projection-{sequence}".encode()).hexdigest()
            component_id = hashlib.sha256(canonical_bytes([projection_hash])).hexdigest()
            inventory.append({
                "component_id": component_id,
                "eligible_cell_mask": 1 << cell_index,
                "eligible_cells": [cell],
                "member_record_count": 1,
                "member_record_count_by_cell": {cell: 1},
                "representative_projection_sha256_by_cell": {cell: projection_hash},
            })
            sequence += 1
    inventory.sort(key=lambda row: row["component_id"])
    masks, stats = statistics_from_inventory(inventory, CELLS)
    buffer = config["component_buffer"]
    original_flow = max_flow_feasibility(
        masks,
        CELLS,
        config["original_quotas"],
        buffer_cell=buffer["cell"],
        minimum_unselected_buffer_components=buffer["minimum_completely_unselected_supporting_components"],
    )
    maximum = search_maximum_balanced(
        masks,
        CELLS,
        stats["supporting_component_count_by_cell"],
        buffer_cell=buffer["cell"],
        minimum_unselected_buffer_components=buffer["minimum_completely_unselected_supporting_components"],
    )
    hotpot_records = sum(counts[:6])
    musique_records = sum(counts[6:])
    quarantine = {
        "hotpotqa": {"eligible_records": hotpot_records},
        "musique": {"eligible_records": musique_records},
        "train_union_question_collision_pairs": {},
        "train_union_question_collision_records": 0,
    }
    registry_config = json.loads(Path(config["bindings"]["registry_config"]["path"]).read_text())
    report = make_report(config, registry_config, ZERO_RELATION_STATS, stats, original_flow, maximum, quarantine)
    output_dir = replica_root / config["outputs"]["artifact_dir"]
    output_dir.mkdir(parents=True)
    core.atomic_jsonl(output_dir / "component_inventory.jsonl", inventory)
    core.atomic_json(output_dir / "feasibility_report.json", report)
    implementation = {
        "config": {"path": str(replica_root / "config.json"), **{
            "nbytes": (replica_root / "config.json").stat().st_size,
            "sha256": core.sha256_file(replica_root / "config.json"),
        }},
        "source": {"path": str(replica_root / "src" / "feasibility.py"), **{
            "nbytes": (replica_root / "src" / "feasibility.py").stat().st_size,
            "sha256": core.sha256_file(replica_root / "src" / "feasibility.py"),
        }},
        "validation_source": {"path": str(replica_root / "src" / "validate.py"), **{
            "nbytes": (replica_root / "src" / "validate.py").stat().st_size,
            "sha256": core.sha256_file(replica_root / "src" / "validate.py"),
        }},
        "entrypoint": {"path": str(replica_root / "scripts" / "run.py"), **{
            "nbytes": (replica_root / "scripts" / "run.py").stat().st_size,
            "sha256": core.sha256_file(replica_root / "scripts" / "run.py"),
        }},
        "validator": {"path": str(replica_root / "scripts" / "validate.py"), **{
            "nbytes": (replica_root / "scripts" / "validate.py").stat().st_size,
            "sha256": core.sha256_file(replica_root / "scripts" / "validate.py"),
        }},
        "tests": {
            path.name: {"path": str(path), "nbytes": path.stat().st_size, "sha256": core.sha256_file(path)}
            for path in sorted((replica_root / "tests").glob("*.py"))
        },
    }
    manifest = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "prepared_feasibility_only_not_registry",
        "bindings": config["bindings"],
        "implementation": implementation,
        "files": {
            "component_inventory.jsonl": {
                "path": str(output_dir / "component_inventory.jsonl"),
                "nbytes": (output_dir / "component_inventory.jsonl").stat().st_size,
                "sha256": core.sha256_file(output_dir / "component_inventory.jsonl"),
                "rows": len(inventory),
            },
            "feasibility_report.json": {
                "path": str(output_dir / "feasibility_report.json"),
                "nbytes": (output_dir / "feasibility_report.json").stat().st_size,
                "sha256": core.sha256_file(output_dir / "feasibility_report.json"),
            },
        },
        "original_quota_feasible": original_flow["pass"],
        "maximum_balanced_H": maximum["H"],
        "completion_marker_required": "PUBLISHED.json",
    }
    core.atomic_json(output_dir / "manifest.json", manifest)
    required_validation = {
        "structural_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/validate.py",
        "deep_recompute_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/validate.py --deep",
        "deep_recomputes_clean_pool_components_inventory_and_report_from_pinned_sources": True,
    }
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
        "maximum_balanced_common_scale": maximum,
        "manifest": {
            "path": str(output_dir / "manifest.json"),
            "nbytes": (output_dir / "manifest.json").stat().st_size,
            "sha256": core.sha256_file(output_dir / "manifest.json"),
        },
        "required_post_publication_validation": required_validation,
        "completion_marker_required": "PUBLISHED.json",
        "claim_boundary": "feasibility only; a later registry needs a separately frozen allocation algorithm and full validation",
    }
    core.atomic_json(output_dir / "receipt.json", receipt)
    payloads = ["component_inventory.jsonl", "feasibility_report.json", "manifest.json", "receipt.json"]
    for name in payloads:
        path = output_dir / name
        core.atomic_json(output_dir / f"{name}.sha256.json", {
            "path": str(path), "nbytes": path.stat().st_size, "sha256": core.sha256_file(path)
        })
    names = sorted([*payloads, *(f"{name}.sha256.json" for name in payloads)])
    published_files = {
        name: {"nbytes": (output_dir / name).stat().st_size, "sha256": core.sha256_file(output_dir / name)}
        for name in names
    }
    ordered = [{"name": name, **published_files[name]} for name in names]
    core.atomic_json(output_dir / "PUBLISHED.json", {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "published_complete_feasibility_only",
        "atomic_last_completion_marker": True,
        "published_files": published_files,
        "ordered_inventory_sha256": hashlib.sha256(canonical_bytes(ordered)).hexdigest(),
        "receipt": published_files["receipt.json"],
        "receipt_sidecar": published_files["receipt.json.sha256.json"],
    })
    result = validate(replica_root / "config.json", replica_root)
    assert result["status"] == "pass"
    assert result["deep_validation"] == {"performed": False}

    rows = []
    for source_index, item in enumerate(inventory):
        cell = item["eligible_cells"][0]
        dataset, stratum = cell.split("|", 1)
        rows.append(core.Projection(
            dataset,
            f"synthetic-{source_index}",
            source_index,
            stratum,
            f"synthetic question {source_index}",
            (f"synthetic passage {source_index}",),
            item["representative_projection_sha256_by_cell"][cell],
        ))
    validate_module = importlib.import_module("src.validate")
    monkeypatch.setattr(validate_module, "load_registry_core", lambda path: core)
    monkeypatch.setattr(
        validate_module,
        "prepare_eligible",
        lambda loaded_core, loaded_config, audit_root, progress: (rows, quarantine),
    )
    monkeypatch.setattr(
        validate_module,
        "build_components",
        lambda eligible, cfg, loaded_core, progress: (UnionFind.create(len(eligible)), ZERO_RELATION_STATS),
    )
    deep_result = validate(replica_root / "config.json", replica_root, deep=True)
    assert deep_result["deep_validation"]["performed"] is True
    assert deep_result["deep_validation"]["inventory_byte_identical"] is True
    assert deep_result["deep_validation"]["report_byte_identical"] is True


@pytest.mark.actual_artifact
def test_actual_published_artifact():
    if not (ROOT / CONFIG["outputs"]["artifact_dir"] / "PUBLISHED.json").is_file():
        pytest.skip("formal feasibility artifact has not been published yet")
    result = validate(ROOT / "config.json", ROOT)
    assert result["status"] == "pass"
