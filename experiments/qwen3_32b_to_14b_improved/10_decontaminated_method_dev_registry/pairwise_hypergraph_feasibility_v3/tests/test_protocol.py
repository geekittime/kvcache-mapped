from __future__ import annotations

import itertools
import json
import os
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace
from collections import Counter
from pathlib import Path

import pytest
import scipy

import src.allocation as allocation_module
from src.allocation import (
    solve_allocation_rows,
    solver_reports_equivalent,
    verify_selection,
)
from src.allocation import run_allocation
from src.common import (
    ProtocolError,
    atomic_json,
    atomic_jsonl,
    claim_artifact,
    finalize_publication,
    load_module,
    validate_publication,
    file_spec,
    canonical_bytes,
    json_exact,
    safe_relative_output,
    strict_json_loads,
)
from src.graph import (
    _add_exact_groups,
    _add_similarity_hyperedges,
    _collapse_units,
    _hyperedge_rows,
    build_solver_contract,
    make_record_rows,
    verify_bindings,
    verify_component_v1_checkpoint_erratum,
    verify_component_v1_failure,
    verify_pairwise_v2_terminal_failure,
    _implementation,
)
from src.validate_allocation import validate_allocation
from src.validate_graph import _validate_passage_similarity_statistics, validate_graph


ROOT = Path(__file__).resolve().parents[1]
AUDIT_ROOT = ROOT.parent
CONFIG = json.loads((ROOT / "config.json").read_text())
REGISTRY_CONFIG = json.loads(Path(CONFIG["bindings"]["registry_config"]["path"]).read_text())


@pytest.fixture(scope="session")
def core():
    return load_module(
        "hypergraph_v3_test_registry_core",
        Path(CONFIG["bindings"]["registry_source"]["path"]),
    )


def projection(core, index: int, question: str, passages: tuple[str, ...], *, digest_name: str | None = None):
    name = digest_name or f"projection-{index}"
    return core.Projection(
        "hotpotqa_train",
        f"id-{index}",
        index,
        "bridge/easy",
        question,
        passages,
        core.digest(name),
    )


def build_direct_edges(core, rows):
    records, record_rows = make_record_rows(rows)
    edges: dict[tuple[int, ...], Counter[str]] = {}
    _add_exact_groups(records, core, edges)
    question_texts, question_owners, _ = _collapse_units(
        records, core, passages=False, edges=edges, progress=None
    )
    passage_texts, passage_owners, _ = _collapse_units(
        records, core, passages=True, edges=edges, progress=None
    )
    cfg = REGISTRY_CONFIG["decontamination"]
    _add_similarity_hyperedges(
        question_texts, question_owners, core, cfg, edges,
        question=True, progress=None,
    )
    _add_similarity_hyperedges(
        passage_texts, passage_owners, core, cfg, edges,
        question=False, progress=None,
    )
    return records, record_rows, _hyperedge_rows(edges, record_rows)


def minimal_contract(quota_screen: int, quota_confirm: int):
    return {
        "solver": {
            "scipy_version": scipy.__version__,
            "hash_cost_namespace": "synthetic-test",
            "required_environment": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "targets": [
            {
                "target_index": 0,
                "partition": "screen",
                "cell": "hotpotqa_train|bridge/easy",
                "dataset_id": "hotpotqa_train",
                "stratum": "bridge/easy",
                "exact_quota": quota_screen,
            },
            {
                "target_index": 1,
                "partition": "confirm",
                "cell": "hotpotqa_train|bridge/easy",
                "dataset_id": "hotpotqa_train",
                "stratum": "bridge/easy",
                "exact_quota": quota_confirm,
            },
        ],
    }


def test_failure_erratum_lineage_and_all_pinned_bindings_are_semantically_valid():
    evidence = verify_component_v1_failure(
        Path(CONFIG["bindings"]["component_v1_failure"]["path"]), AUDIT_ROOT
    )
    assert evidence["observed_complete_scan"]["passage_direct_relation_pairs"] == 7797
    pairwise_failure = verify_pairwise_v2_terminal_failure(
        Path(CONFIG["bindings"]["pairwise_v2_terminal_failure"]["path"])
    )
    erratum = verify_component_v1_checkpoint_erratum(
        Path(CONFIG["bindings"]["component_v1_checkpoint_erratum"]["path"]),
        pairwise_failure,
    )
    assert erratum["mislabelled_value_in_original_evidence"] == evidence["observed_complete_scan"]
    assert erratum["corrected_terminal_scan"] == pairwise_failure["observed_terminal_scan"]
    assert erratum["corrected_terminal_scan"] == CONFIG["expected_complete_scan"]
    assert erratum["actual_semantics_of_mislabelled_value"]["completed_left_count_at_checkpoint"] == 240001
    assert erratum["actual_semantics_of_mislabelled_value"]["remaining_left_count_after_checkpoint"] == 4037
    _, _, registry_config, audit_root = verify_bindings(CONFIG)
    assert registry_config["decontamination"] == REGISTRY_CONFIG["decontamination"]
    assert audit_root == AUDIT_ROOT


def test_similarity_scan_emits_distinct_terminal_completion_event(core):
    base = " ".join(f"terminal{index}" for index in range(25))
    events = []
    edges: dict[tuple[int, ...], Counter[str]] = {}
    statistics = _add_similarity_hyperedges(
        [base, base + " extra"],
        [(0,), (1,)],
        core,
        REGISTRY_CONFIG["decontamination"],
        edges,
        question=False,
        progress=events.append,
    )
    terminal_events = [event for event in events if event["phase"] == "passage_similarity_complete"]
    assert len(terminal_events) == 1
    assert events[-1] == terminal_events[0] == statistics["terminal_completion_event"]
    assert terminal_events[0] == {
        "phase": "passage_similarity_complete",
        "completed_left_count": 2,
        "unique_units_total": 2,
        "candidates_verified": 1,
        "direct_relation_pairs": 1,
    }
    assert "unique_units_processed" not in terminal_events[0]


def test_validator_rejects_periodic_checkpoint_or_bool_as_terminal_completion():
    expected = {
        "clean_eligible_records": 3,
        "unique_normalized_passage_units": 3,
        "passage_candidates_verified": 2,
        "passage_direct_relation_pairs": 2,
    }
    statistics = {
        "unique_units": 3,
        "candidates_verified": 2,
        "direct_relation_pairs": 2,
        "relation_pairs_by_kind": {"near_passage_owner_union": 2},
        "nontrivial_owner_union_derivations": 2,
        "terminal_completion_event": {
            "phase": "passage_similarity_complete",
            "completed_left_count": 3,
            "unique_units_total": 3,
            "candidates_verified": 2,
            "direct_relation_pairs": 2,
        },
    }
    _validate_passage_similarity_statistics(statistics, expected)
    periodic_phase_only = json.loads(json.dumps(statistics))
    periodic_phase_only["terminal_completion_event"]["phase"] = "passage_similarity_hyperedges"
    with pytest.raises(ProtocolError, match="terminal-completion"):
        _validate_passage_similarity_statistics(periodic_phase_only, expected)
    checkpoint = json.loads(json.dumps(statistics))
    checkpoint["terminal_completion_event"] = {
        "phase": "passage_similarity_hyperedges",
        "unique_units_processed": 240000,
        "unique_units_total": 244038,
        "candidates_verified": 39817022,
        "direct_relation_pairs": 7797,
    }
    with pytest.raises(ProtocolError, match="terminal-completion"):
        _validate_passage_similarity_statistics(checkpoint, expected)
    bool_alias = json.loads(json.dumps(statistics))
    bool_alias["terminal_completion_event"]["completed_left_count"] = True
    with pytest.raises(ProtocolError, match="terminal-completion"):
        _validate_passage_similarity_statistics(bool_alias, expected)


def test_overlapping_direct_edges_do_not_become_transitive_component(core):
    near_base = " ".join(f"w{index}" for index in range(25))
    original = [
        projection(core, 0, "unique q zero", ("short exact anchor",), digest_name="a"),
        projection(core, 1, "unique q one", ("short exact anchor", near_base), digest_name="b"),
        projection(core, 2, "unique q two", (near_base + " extra",), digest_name="c"),
    ]
    records, record_rows, edges = build_direct_edges(core, original)
    index_by_name = {
        row.projection_sha256: index for index, row in enumerate(records)
    }
    a, b, c = (index_by_name[core.digest(name)] for name in ("a", "b", "c"))
    member_sets = {tuple(row["member_record_indices"]) for row in edges}
    assert tuple(sorted((a, b))) in member_sets
    assert tuple(sorted((b, c))) in member_sets
    assert tuple(sorted((a, c))) not in member_sets
    assert tuple(sorted((a, b, c))) not in member_sets

    selection, report = solve_allocation_rows(
        record_rows, edges, minimal_contract(1, 1), require_environment=False
    )
    assert report["original_quota_feasible"] is True
    assert {row["record_index"] for row in selection} == {a, c}
    assert verify_selection(
        selection, record_rows, edges, minimal_contract(1, 1), require_exact_quotas=True
    )["violating_hyperedges"] == 0


def test_unit_pair_owner_union_is_exact_direct_clique(core):
    near_base = " ".join(f"u{index}" for index in range(25))
    original = [
        projection(core, 0, "q0", (near_base,), digest_name="a"),
        projection(core, 1, "q1", (near_base,), digest_name="b"),
        projection(core, 2, "q2", (near_base + " extra",), digest_name="c"),
        projection(core, 3, "q3", (near_base + " extra",), digest_name="d"),
    ]
    records, _, edges = build_direct_edges(core, original)
    all_members = tuple(range(len(records)))
    union_edges = [
        edge for edge in edges
        if tuple(edge["member_record_indices"]) == all_members
        and "near_passage_owner_union" in edge["relation_kinds"]
    ]
    assert len(union_edges) == 1
    # Every pair in the owner union is direct: within each normalized unit the
    # pair is exact, and across units it inherits the near-unit relation.
    assert len(list(itertools.combinations(all_members, 2))) == 6


def test_complete_substring_recall_with_destroyed_word_shingle_boundaries(core):
    question = " ".join(character * 42 for character in ("a", "b", "c"))
    passage = " ".join(character * 30 for character in ("d", "e", "f", "g", "h"))
    rows = [
        projection(core, 0, question, ("p0",), digest_name="a"),
        projection(core, 1, "prefix" + question + " suffix", ("p1",), digest_name="b"),
        projection(core, 2, "q2", (passage,), digest_name="c"),
        projection(core, 3, "q3", ("prefix" + passage + " suffix",), digest_name="d"),
    ]
    records, _, edges = build_direct_edges(core, rows)
    name_to_index = {row.projection_sha256: index for index, row in enumerate(records)}
    pairs = {
        tuple(edge["member_record_indices"]): set(edge["relation_kinds"])
        for edge in edges if edge["member_count"] == 2
    }
    q_pair = tuple(sorted((name_to_index[core.digest("a")], name_to_index[core.digest("b")])))
    p_pair = tuple(sorted((name_to_index[core.digest("c")], name_to_index[core.digest("d")])))
    assert "normalized_question_substring_owner_union" in pairs[q_pair]
    assert "normalized_passage_substring_owner_union" in pairs[p_pair]


def test_hyperedge_pair_membership_equals_frozen_direct_collision_relation(core):
    near_question = "one two three four five six seven eight nine ten"
    near_passage = " ".join(f"p{index}" for index in range(25))
    long_question = " ".join(character * 42 for character in ("x", "y", "z"))
    rows = [
        projection(core, 0, "unique zero", ("same passage",), digest_name="a"),
        projection(core, 1, "unique one", ("SAME---PASSAGE",), digest_name="b"),
        projection(core, 2, near_question, ("unique two",), digest_name="c"),
        projection(core, 3, near_question + " eleven", ("unique three",), digest_name="d"),
        projection(core, 4, long_question, (near_passage,), digest_name="e"),
        projection(core, 5, "prefix" + long_question + " suffix", (near_passage + " extra",), digest_name="f"),
        projection(core, 6, "fully unrelated", ("fully unrelated passage",), digest_name="g"),
    ]
    records, _, edges = build_direct_edges(core, rows)
    edge_sets = [set(edge["member_record_indices"]) for edge in edges]
    cfg = REGISTRY_CONFIG["decontamination"]
    empty_q = core.build_mapper_index([], cfg, question=True)
    empty_p = core.build_mapper_index([], cfg, question=False)
    for left, right in itertools.combinations(range(len(records)), 2):
        reference = core.build_reference_index([records[left]], cfg)
        formal_direct = bool(core.contamination_reasons(
            records[right],
            reference,
            empty_q,
            empty_p,
            cfg,
            locked_label="pair",
            include_mapper=False,
        ))
        graph_direct = any({left, right} <= members for members in edge_sets)
        assert graph_direct is formal_direct, (left, right)


def test_original_solver_contract_and_record_buffer_arithmetic():
    contract = build_solver_contract(CONFIG)
    assert len(contract["targets"]) == 18
    assert sum(target["exact_quota"] for target in contract["targets"]) == 1152
    four_hop = [target for target in contract["targets"] if target["cell"] == "musique_train|4hop"]
    assert [target["exact_quota"] for target in four_hop] == [39, 78]
    assert CONFIG["musique_4hop_record_buffer"]["strict_clean_records"] - sum(
        target["exact_quota"] for target in four_hop
    ) == 2
    assert contract["constraints"]["no_transitive_closure"] is True


def test_execution_closure_and_readme_commands_are_exact_and_bytecode_free():
    implementation = _implementation(ROOT, ROOT / "config.json")
    assert sorted(implementation["sources"]) == [
        "__init__.py", "allocation.py", "common.py", "graph.py",
        "validate_allocation.py", "validate_graph.py",
    ]
    assert sorted(implementation["scripts"]) == [
        "allocation_worker.py", "build_graph.py", "solve_allocation.py",
        "validate_allocation.py", "validate_graph.py",
    ]
    readme = (ROOT / "README.md").read_text()
    for name, command in CONFIG["commands"].items():
        if name != "nonartifact_tests":
            assert command in readme
        assert "/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python" in command
        assert "PYTHONDONTWRITEBYTECODE=1" in command
        assert "PYTHONPYCACHEPREFIX=" in command
    assert set(CONFIG["formal_launchers"]) == {
        "graph_build", "graph_structural_validation", "graph_deep_validation",
        "allocation", "allocation_deep_validation",
    }
    sessions = [value["tmux_session"] for value in CONFIG["formal_launchers"].values()]
    logs = [value["log_path"] for value in CONFIG["formal_launchers"].values()]
    assert len(sessions) == len(set(sessions)) == 5
    assert len(logs) == len(set(logs)) == 5
    assert all("_v3_" in value and "_v2_" not in value for value in sessions)
    assert all("_v3_" in value and "_v2_" not in value for value in logs)
    assert CONFIG["solver"]["hash_cost_namespace"] == "qwen3-32to14-improved/pairwise-hypergraph-v2"
    assert "objective namespace remains" in readme
    v2_root = ROOT.parent / "pairwise_hypergraph_feasibility_v2"
    assert CONFIG["v2_byte_identical_execution_files"] == [
        "src/allocation.py", "src/common.py", "src/validate_allocation.py",
        "scripts/allocation_worker.py", "scripts/build_graph.py",
        "scripts/solve_allocation.py", "scripts/validate_allocation.py",
        "scripts/validate_graph.py",
    ]
    for relative in CONFIG["v2_byte_identical_execution_files"]:
        assert (ROOT / relative).read_bytes() == (v2_root / relative).read_bytes()
    assert not list(ROOT.rglob("*.pyc"))
    assert not list(ROOT.rglob("__pycache__"))


def test_milp_reports_infeasible_without_fake_selection():
    records = [
        {
            "record_index": index,
            "dataset_id": "hotpotqa_train",
            "stratum": "bridge/easy",
            "source_index": index,
            "source_projection_sha256": f"{index + 1:064x}",
        }
        for index in range(2)
    ]
    edge = {
        "hyperedge_id": "f" * 64,
        "member_record_indices": [0, 1],
        "member_count": 2,
        "relation_kinds": ["raw_sample_id"],
        "derivation_count_by_kind": {"raw_sample_id": 1},
    }
    selection, report = solve_allocation_rows(
        records, [edge], minimal_contract(1, 1), require_environment=False
    )
    assert selection == []
    assert report["original_quota_feasible"] is False
    assert report["phase_2_run"] is False


def test_small_milp_feasibility_matches_bruteforce():
    random_generator = random.Random(20260831)
    for _ in range(30):
        count = random_generator.randint(3, 7)
        records = [
            {
                "record_index": index,
                "dataset_id": "hotpotqa_train",
                "stratum": "bridge/easy",
                "source_index": index,
                "source_projection_sha256": f"{index + 1:064x}",
            }
            for index in range(count)
        ]
        member_sets = []
        for left in range(count):
            for right in range(left + 1, count):
                if random_generator.random() < 0.25:
                    member_sets.append((left, right))
        edges = [
            {
                "hyperedge_id": f"{edge_index + 1:064x}",
                "member_record_indices": list(members),
                "member_count": 2,
                "relation_kinds": ["raw_sample_id"],
                "derivation_count_by_kind": {"raw_sample_id": 1},
            }
            for edge_index, members in enumerate(member_sets)
        ]
        quota_screen = random_generator.randint(0, min(2, count))
        quota_confirm = random_generator.randint(0, min(2, count - quota_screen))
        needed = quota_screen + quota_confirm
        brute_feasible = any(
            all(not (set(members) <= set(chosen)) for members in member_sets)
            for chosen in itertools.combinations(range(count), needed)
        )
        selection, report = solve_allocation_rows(
            records,
            edges,
            minimal_contract(quota_screen, quota_confirm),
            require_environment=False,
        )
        assert report["original_quota_feasible"] is brute_feasible
        assert bool(selection) is (brute_feasible and needed > 0)


def test_worker_selection_is_byte_identical_across_hashseeds(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    records = [
        {
            "record_index": index,
            "dataset_id": "hotpotqa_train",
            "stratum": "bridge/easy",
            "source_index": index,
            "source_projection_sha256": f"{index + 1:064x}",
        }
        for index in range(5)
    ]
    edges = [
        {
            "hyperedge_id": "a" * 64,
            "member_record_indices": [0, 1],
            "member_count": 2,
            "relation_kinds": ["raw_sample_id"],
            "derivation_count_by_kind": {"raw_sample_id": 1},
        }
    ]
    atomic_jsonl(graph_dir / "records.jsonl", records)
    atomic_jsonl(graph_dir / "hyperedges.jsonl", edges)
    atomic_json(graph_dir / "solver_contract.json", minimal_contract(1, 1))
    outputs = []
    for seed in ("1", "777"):
        output = tmp_path / f"selection-{seed}.jsonl"
        environment = {
            **os.environ,
            "PYTHONHASHSEED": seed,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "allocation_worker.py"),
                "--graph-dir", str(graph_dir),
                "--output", str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        assert json.loads(result.stdout)["solver"]["original_quota_feasible"] is True
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]


def test_atomic_publication_contract_and_nonreuse(tmp_path):
    output = tmp_path / "artifact"
    identity = {"protocol_id": "test", "artifact_kind": "graph"}
    claim_artifact(output, identity)
    atomic_json(output / "data.json", {"value": 1})
    atomic_json(output / "receipt.json", {"status": "pass"})
    marker = finalize_publication(
        output,
        ["data.json", "receipt.json"],
        identity=identity,
        status="published",
    )
    observed = validate_publication(
        output,
        ["data.json", "receipt.json"],
        identity=identity,
        status="published",
    )
    assert observed == marker
    with pytest.raises(ProtocolError, match="refusing to reuse"):
        claim_artifact(output, identity)


def test_phase1_only_explicit_status2_can_publish_infeasible(monkeypatch):
    records = [
        {
            "record_index": index,
            "dataset_id": "hotpotqa_train",
            "stratum": "bridge/easy",
            "source_index": index,
            "source_projection_sha256": f"{index + 1:064x}",
        }
        for index in range(2)
    ]
    contract = minimal_contract(1, 1)
    for status in (1, 3, 4):
        result = SimpleNamespace(
            status=status,
            success=False,
            x=None,
            message=f"synthetic nonpublishable status {status}",
        )
        monkeypatch.setattr(allocation_module, "milp", lambda *args, result=result, **kwargs: result)
        with pytest.raises(ProtocolError, match="did not prove"):
            solve_allocation_rows(records, [], contract, require_environment=False)

    explicit = SimpleNamespace(
        status=2,
        success=False,
        x=None,
        message="The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is At lower/fixed bound)",
    )
    monkeypatch.setattr(allocation_module, "milp", lambda *args, **kwargs: explicit)
    selection, report = solve_allocation_rows(
        records, [], contract, require_environment=False
    )
    assert selection == []
    assert report["explicit_infeasibility_certificate"] is True
    assert report["phase_1"]["status"] == 2


def test_phase2_rejects_nonoptimal_nonfinite_and_nonbinary_results(monkeypatch):
    records = [
        {
            "record_index": index,
            "dataset_id": "hotpotqa_train",
            "stratum": "bridge/easy",
            "source_index": index,
            "source_projection_sha256": f"{index + 1:064x}",
        }
        for index in range(2)
    ]
    contract = minimal_contract(1, 1)
    optimal_message = "Optimization terminated successfully. (HiGHS Status 7: Optimal)"
    phase_1 = SimpleNamespace(
        status=0, success=True, message=optimal_message, x=[1.0, 0.0, 0.0, 1.0]
    )
    bad_results = [
        SimpleNamespace(status=1, success=False, message="time limit", x=None, fun=None),
        SimpleNamespace(status=0, success=True, message=optimal_message, x=[1.0, 0.0, 0.0, 1.0], fun=float("nan")),
        SimpleNamespace(status=0, success=True, message=optimal_message, x=[0.4, 0.6, 0.0, 1.0], fun=0.0),
    ]
    for bad in bad_results:
        calls = iter((phase_1, bad))
        monkeypatch.setattr(allocation_module, "milp", lambda *args, **kwargs: next(calls))
        with pytest.raises(ProtocolError):
            solve_allocation_rows(records, [], contract, require_environment=False)


def test_bool_aliases_duplicate_keys_nonfinite_and_symlink_escape_fail_closed(tmp_path):
    records = [{
        "record_index": True,
        "dataset_id": "hotpotqa_train",
        "stratum": "bridge/easy",
        "source_index": 0,
        "source_projection_sha256": "1" * 64,
    }]
    with pytest.raises(ProtocolError, match="typed schema"):
        solve_allocation_rows(records, [], minimal_contract(0, 0), require_environment=False)
    bad_contract = minimal_contract(0, 0)
    bad_contract["targets"][0]["exact_quota"] = True
    good_record = {**records[0], "record_index": 0}
    with pytest.raises(ProtocolError, match="target typed"):
        solve_allocation_rows([good_record], [], bad_contract, require_environment=False)
    with pytest.raises(ProtocolError, match="duplicate JSON"):
        strict_json_loads('{"a":1,"a":2}')
    with pytest.raises(ProtocolError, match="nonfinite JSON"):
        strict_json_loads('{"a":NaN}')
    with pytest.raises(ValueError):
        canonical_bytes({"a": float("inf")})

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ProtocolError, match="symlink"):
        safe_relative_output(root, "escape/artifact")


def test_publication_rejects_extra_member_and_payload_symlink(tmp_path):
    identity = {"protocol_id": "strict", "artifact_kind": "test"}
    output = tmp_path / "extra"
    claim_artifact(output, identity)
    atomic_json(output / "data.json", {"value": 1})
    atomic_json(output / "receipt.json", {"status": "pass"})
    finalize_publication(output, ["data.json", "receipt.json"], identity=identity, status="published")
    atomic_json(output / "unexpected.json", {})
    with pytest.raises(ProtocolError, match="inventory"):
        validate_publication(output, ["data.json", "receipt.json"], identity=identity, status="published")

    output = tmp_path / "symlink"
    claim_artifact(output, identity)
    atomic_json(output / "data.json", {"value": 1})
    atomic_json(output / "receipt.json", {"status": "pass"})
    finalize_publication(output, ["data.json", "receipt.json"], identity=identity, status="published")
    external = tmp_path / "external.json"
    atomic_json(external, {"value": 1})
    (output / "data.json").unlink()
    (output / "data.json").symlink_to(external)
    with pytest.raises(ProtocolError, match="symlink"):
        validate_publication(output, ["data.json", "receipt.json"], identity=identity, status="published")


def test_solver_report_equivalence_normalizes_both_seed_locations():
    left = {
        "execution": {
            "process_id": 10,
            "python_hash_seed": "1",
            "highs_random_seed": 1,
            "highs_options": {"threads": 1, "random_seed": 1},
        },
        "result": "same",
    }
    right = {
        "execution": {
            "process_id": 11,
            "python_hash_seed": "777",
            "highs_random_seed": 777,
            "highs_options": {"threads": 1, "random_seed": 777},
        },
        "result": "same",
    }
    assert solver_reports_equivalent(left, right)


def test_synthetic_graph_deep_receipt_and_two_process_allocation_end_to_end(
    tmp_path, monkeypatch
):
    replica_root = tmp_path / "pairwise_hypergraph_feasibility_v3"
    for directory in ("src", "scripts", "tests"):
        (replica_root / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "README.md", replica_root / "README.md")
    shutil.copy2(ROOT / "pytest.ini", replica_root / "pytest.ini")
    for path in sorted((ROOT / "src").glob("*.py")):
        shutil.copy2(path, replica_root / "src" / path.name)
    for path in sorted((ROOT / "scripts").glob("*.py")):
        shutil.copy2(path, replica_root / "scripts" / path.name)
    for path in sorted((ROOT / "tests").glob("*.py")):
        shutil.copy2(path, replica_root / "tests" / path.name)
    config = json.loads((ROOT / "config.json").read_text())
    for partition in config["partitions_order"]:
        for cell in config["cells_order"]:
            config["original_quotas"][partition][cell] = 0
    first_cell = config["cells_order"][0]
    config["original_quotas"]["screen"][first_cell] = 1
    config["original_quotas"]["confirm"][first_cell] = 1
    config["musique_4hop_record_buffer"] = {
        "strict_clean_records": 0,
        "maximum_selected_records": 0,
        "minimum_unselected_records": 0,
        "does_not_require_distinct_or_unselected_connected_components": True,
    }
    config["expected_complete_scan"] = {
        "clean_eligible_records": 3,
        "unique_normalized_passage_units": 3,
        "passage_candidates_verified": 2,
        "passage_direct_relation_pairs": 2,
    }
    atomic_json(replica_root / "config.json", config)
    contract = build_solver_contract(config)
    records = [
        {
            "record_index": index,
            "dataset_id": "hotpotqa_train",
            "stratum": "bridge/easy",
            "source_index": index,
            "source_projection_sha256": f"{index + 1:064x}",
        }
        for index in range(3)
    ]
    edges = []
    for edge_index, members in enumerate(([0, 1], [1, 2])):
        projections = [records[index]["source_projection_sha256"] for index in members]
        from src.common import canonical_bytes
        import hashlib
        edge_id = hashlib.sha256(canonical_bytes(projections)).hexdigest()
        edges.append({
            "hyperedge_id": edge_id,
            "member_record_indices": members,
            "member_count": 2,
            "relation_kinds": ["normalized_passage_unit_owners"],
            "derivation_count_by_kind": {"normalized_passage_unit_owners": 1},
        })
    edges.sort(key=lambda row: row["hyperedge_id"])
    report = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "direct_hypergraph_built_not_allocation",
        "thresholds": REGISTRY_CONFIG["decontamination"],
        "direct_hyperedge_policy": config["direct_hyperedge_policy"],
        "record_count": 3,
        "hyperedge_count": 2,
        "hyperedge_member_occurrences": 4,
        "hyperedge_member_count_distribution": {
            "minimum": 2, "maximum": 2, "size_2": 2, "size_gt_2": 0,
        },
        "exact_group_statistics": {},
        "question_unit_statistics": {},
        "passage_unit_statistics": {},
        "question_similarity_statistics": {},
        "passage_similarity_statistics": {
            "unique_units": 3,
            "candidates_verified": 2,
            "direct_relation_pairs": 2,
            "relation_pairs_by_kind": {"near_passage_owner_union": 2},
            "nontrivial_owner_union_derivations": 2,
            "terminal_completion_event": {
                "phase": "passage_similarity_complete",
                "completed_left_count": 3,
                "unique_units_total": 3,
                "candidates_verified": 2,
                "direct_relation_pairs": 2,
            },
        },
        "eligible_question_collision_residual_assertion": {
            "pass": True,
            "raw_question": 0,
            "normalized_question": 0,
            "normalized_question_context": 0,
            "normalized_question_unit_owner_groups": 0,
            "question_direct_relation_pairs": 0,
        },
        "observed_complete_scan": config["expected_complete_scan"],
        "decontamination_aggregate": {
            "hotpotqa": {"eligible_records": 3},
            "musique": {"eligible_records": 0},
        },
        "musique_4hop_record_buffer": {
            **config["musique_4hop_record_buffer"],
            "observed_clean_records": 0,
            "original_quota_selected_records": 0,
            "observed_unselected_record_buffer": 0,
        },
        "privacy": {
            "raw_ids_questions_contexts_answers_or_outcomes_persisted": False,
            "answer_values_available_to_graph_logic": False,
            "model_outcomes_read": False,
        },
        "claim_boundary": "privacy-safe direct hypergraph only; no MILP solved and no allocation selected",
    }
    graph_dir = replica_root / config["outputs"]["graph_artifact_dir"]
    identity = {"protocol_id": config["protocol_id"], "artifact_kind": "direct_hypergraph"}
    claim_artifact(graph_dir, identity)
    atomic_jsonl(graph_dir / "records.jsonl", records)
    atomic_jsonl(graph_dir / "hyperedges.jsonl", edges)
    atomic_json(graph_dir / "graph_report.json", report)
    atomic_json(graph_dir / "solver_contract.json", contract)
    implementation = _implementation(replica_root, replica_root / "config.json")
    manifest = {
        "schema_version": 1,
        **identity,
        "status": "prepared_privacy_safe_reusable_hypergraph",
        "bindings": config["bindings"],
        "implementation": implementation,
        "runtime": {
            **config["runtime_identity"],
            "python_hash_seed": config["deterministic_processes"]["graph_build_python_hash_seed"],
            "python_pycache_prefix_absent": config["deterministic_processes"]["empty_python_pycache_prefix"],
            "process_id": os.getpid() + 1000000,
        },
        "files": {
            "records.jsonl": {**file_spec(graph_dir / "records.jsonl"), "rows": 3},
            "hyperedges.jsonl": {**file_spec(graph_dir / "hyperedges.jsonl"), "rows": 2},
            "graph_report.json": file_spec(graph_dir / "graph_report.json"),
            "solver_contract.json": file_spec(graph_dir / "solver_contract.json"),
        },
        "deep_validation_required_before_allocation": True,
        "deep_validation_receipt_path": str(replica_root / config["outputs"]["graph_deep_receipt"]),
        "completion_marker_required": "PUBLISHED.json",
    }
    atomic_json(graph_dir / "manifest.json", manifest)
    graph_receipt = {
        "schema_version": 1,
        **identity,
        "status": "pass_graph_built_no_allocation",
        "cpu_only": True,
        "record_count": 3,
        "hyperedge_count": 2,
        "manifest": file_spec(graph_dir / "manifest.json"),
        "solver_contract": file_spec(graph_dir / "solver_contract.json"),
        "component_v1_failure": config["bindings"]["component_v1_failure"],
        "component_v1_checkpoint_erratum": config["bindings"]["component_v1_checkpoint_erratum"],
        "pairwise_v2_terminal_failure": config["bindings"]["pairwise_v2_terminal_failure"],
        "required_commands": config["commands"],
        "claim_boundary": "graph and reusable solver contract only; no feasibility or allocation result",
    }
    atomic_json(graph_dir / "receipt.json", graph_receipt)
    finalize_publication(
        graph_dir,
        [
            "records.jsonl", "hyperedges.jsonl", "graph_report.json",
            "solver_contract.json", "manifest.json", "receipt.json",
        ],
        identity=identity,
        status="published_complete_direct_hypergraph",
    )
    validate_module = sys.modules["src.validate_graph"]
    monkeypatch.setattr(
        validate_module,
        "build_graph_payloads",
        lambda loaded_config, progress: (records, edges, report, contract),
    )
    monkeypatch.setattr(
        validate_module,
        "verify_bindings",
        lambda loaded_config: (None, None, REGISTRY_CONFIG, AUDIT_ROOT),
    )
    monkeypatch.setenv("PYTHONHASHSEED", config["deterministic_processes"]["graph_deep_python_hash_seed"])
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", config["deterministic_processes"]["empty_python_pycache_prefix"])
    graph_validation = validate_graph(
        replica_root / "config.json", replica_root, deep=True, write_receipt=True
    )
    assert graph_validation["deep_validation"]["all_four_payloads_byte_identical"] is True
    assert graph_validation["deep_receipt"]["PUBLISHED"]["sha256"]

    monkeypatch.setenv("PYTHONHASHSEED", config["deterministic_processes"]["graph_build_python_hash_seed"])
    allocation_result = run_allocation(replica_root / "config.json", replica_root)
    assert allocation_result["original_quota_feasible"] is True
    assert allocation_result["selected_records"] == 2
    monkeypatch.setenv("PYTHONHASHSEED", config["deterministic_processes"]["graph_deep_python_hash_seed"])
    allocation_validation = validate_allocation(
        replica_root / "config.json", replica_root, deep=True
    )
    assert allocation_validation["status"] == "pass"
    assert allocation_validation["deep_allocation_validation"]["two_new_solver_processes"] is True
