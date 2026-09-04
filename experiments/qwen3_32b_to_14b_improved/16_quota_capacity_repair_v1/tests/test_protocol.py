from __future__ import annotations

import copy
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import scipy

# Keep direct test-module execution fail-closed as well; pytest normally runs
# ``tests/conftest.py`` first, but a reviewer may invoke this file directly.
_test_cache_prefix = os.environ.get("PYTHONPYCACHEPREFIX")
if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
    raise RuntimeError("PYTHONDONTWRITEBYTECODE must be exactly '1'")
if (
    not isinstance(_test_cache_prefix, str)
    or not _test_cache_prefix
    or not os.path.isabs(_test_cache_prefix)
):
    raise RuntimeError("PYTHONPYCACHEPREFIX must be an absolute fresh absent path")
import sys
if (
    sys.dont_write_bytecode is not True
    or type(sys.pycache_prefix) is not str
    or sys.pycache_prefix != _test_cache_prefix
):
    raise RuntimeError("interpreter bytecode flags do not match the requested cache prefix")
if os.path.lexists(_test_cache_prefix):
    raise RuntimeError("PYTHONPYCACHEPREFIX must be an absolute fresh absent path")
for _component in Path(_test_cache_prefix).parents:
    if _component.is_symlink() or not _component.exists() or not _component.is_dir():
        raise RuntimeError("PYTHONPYCACHEPREFIX parent is missing, non-directory, or symlink")
sys.dont_write_bytecode = True
del _test_cache_prefix

from src.allocation import solve_allocation_rows, verify_selection
from src.common import ProtocolError, canonical_bytes, require_fresh_python_bootstrap, strict_json_loads
from src.inputs import load_config, load_input_bundle
from src.preflight import preflight, validate_config_shape
from src.quota import (
    _exact_lambda_certificate,
    build_matrix,
    solve_quota_search,
    targets_from_contract,
    validate_quota_vector,
)
from src.registry_adapter import project_selection_to_native_interface
from src.runner import validate_stage


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_problem():
    # Nine cells, six independent rows per cell, and three disjoint collision
    # pairs per cell.  Upper quotas are 2/4, so each cell's capacity of three
    # gives the exact max-min lambda 1/2.
    cells = [("hotpotqa_train", f"bridge/{level}") for level in ("easy", "medium", "hard")]
    cells += [("hotpotqa_train", f"comparison/{level}") for level in ("easy", "medium", "hard")]
    cells += [("musique_train", hop) for hop in ("2hop", "3hop", "4hop")]
    records = []
    for cell_index, (dataset, stratum) in enumerate(cells):
        for local in range(6):
            record_index = cell_index * 6 + local
            records.append({
                "record_index": record_index,
                "dataset_id": dataset,
                "stratum": stratum,
                "source_index": record_index,
                "source_projection_sha256": f"{record_index + 1:064x}",
            })
    # Hyperedge IDs and record hashes are both strictly ascending.
    edges = []
    edge_number = 1
    for cell_index in range(9):
        for local in range(0, 6, 2):
            left = cell_index * 6 + local
            right = left + 1
            edges.append({
                "hyperedge_id": f"{edge_number:064x}",
                "member_record_indices": [left, right],
                "member_count": 2,
                "relation_kinds": ["near_passage_owner_union"],
                "derivation_count_by_kind": {"near_passage_owner_union": 1},
            })
            edge_number += 1
    targets = []
    target_index = 0
    for dataset, stratum in cells:
        cell = f"{dataset}|{stratum}"
        for partition, quota in (("screen", 2), ("confirm", 4)):
            targets.append({
                "target_index": target_index,
                "partition": partition,
                "cell": cell,
                "dataset_id": dataset,
                "stratum": stratum,
                "exact_quota": quota,
            })
            target_index += 1
    contract = {"targets": targets}
    config = {
        "upper_quotas": {str(index): (2 if index % 2 == 0 else 4) for index in range(18)},
        "solver": {
            "scipy_version": scipy.__version__,
            "required_environment": {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            "highs_options": {"threads": 1, "presolve": True, "mip_rel_gap": 0.0},
            "hash_cost_namespace": "qwen3-32to14-improved/pairwise-hypergraph-v2",
            "objective_denominator": 16 ** 13,
        },
    }
    return records, edges, contract, config


def test_preflight_binds_graph_and_reports_placeholder_gate_without_writes():
    config_path = ROOT / "config.json"
    result = preflight(config_path)
    # The external allocation-v3 deep worker may finish while this package is
    # being audited.  Both states are safe and explicit: a missing/in-progress
    # receipt is a placeholder gate; a published receipt is a ready gate, but
    # neither state starts A--D or formal/GPU work in this test.
    if result["allocation_v3_deep_receipt"].get("ready"):
        assert result["status"] == "PRE_FORMAL_READY_FOR_CPU_REPAIR"
    else:
        assert result["status"] == "PRE_FORMAL_PLACEHOLDER_GATE"
    assert result["cpu_only"] is True
    assert result["gpu_execution_performed"] is False
    assert result["answer_values_read"] is False
    assert result["model_outcomes_read"] is False
    assert result["graph_records"] == 31260
    assert result["collision_hyperedges"] == 39447
    assert result["allocation_v3_deep_receipt"]["placeholder"] is (result["status"] == "PRE_FORMAL_PLACEHOLDER_GATE")


def test_lambda_matrix_has_one_well_formed_row_per_target():
    records, edges, contract, config = _synthetic_problem()
    targets = targets_from_contract(contract, config["upper_quotas"])
    built = build_matrix(records, edges, targets, with_lambda=True)
    assert len(built.row_kinds) == built.constraints.A.shape[0]
    assert built.row_kinds.count("lambda_minimum") == 18
    matrix = built.constraints.A.tocsr()
    lambda_column = built.layout.lambda_index
    assert lambda_column is not None
    lambda_rows = [i for i, kind in enumerate(built.row_kinds) if kind == "lambda_minimum"]
    for target, row_index in zip(targets, lambda_rows, strict=True):
        row = matrix.getrow(row_index)
        assert row[0, lambda_column] == pytest.approx(-target.upper_quota)
        assert np.count_nonzero(row.toarray()) == len(built.layout.target_variables[target.index]) + 1
        assert built.constraints.lb[row_index] == 0
        assert np.isinf(built.constraints.ub[row_index])


def test_capacity_search_uses_exact_breakpoint_certificate_and_ratio():
    records, edges, contract, config = _synthetic_problem()
    quota, proof = solve_quota_search(records, edges, contract, config, seed=1, require_environment=True)
    assert proof["phase_a"]["lambda_numerator"] == 1
    assert proof["phase_a"]["lambda_denominator"] == 2
    assert proof["phase_a"]["exact_lambda_certificate"]["method"] == "finite_breakpoint_binary_search"
    assert proof["phase_a"]["exact_lambda_certificate"]["next_breakpoint_feasible"] is False
    assert proof["phase_b"]["selected_total"] == 27
    assert sum(quota.values()) == 27
    checked = validate_quota_vector(quota, targets_from_contract(contract, config["upper_quotas"]), expected_total=27)
    assert checked["lambda_numerator"] == 1 and checked["lambda_denominator"] == 2
    for index in range(0, 18, 2):
        assert abs(quota[str(index + 1)] - 2 * quota[str(index)]) <= 1


def test_allocation_sha_cost_two_seeds_are_byte_identical_and_privacy_safe():
    records, edges, contract, config = _synthetic_problem()
    quota, _ = solve_quota_search(records, edges, contract, config, seed=1)
    rows_one, report_one = solve_allocation_rows(records, edges, contract, config, quota, seed=1)
    rows_two, report_two = solve_allocation_rows(records, edges, contract, config, quota, seed=777)
    payload_one = b"".join(canonical_bytes(row) for row in rows_one)
    payload_two = b"".join(canonical_bytes(row) for row in rows_two)
    assert payload_one == payload_two
    assert report_one["phase_2"]["objective_denominator"] == 16 ** 13
    assert report_one["privacy"] == {"answer_values_read": False, "model_outcomes_read": False, "raw_text_persisted": False}
    assert report_one["verification"]["violating_hyperedges"] == 0
    assert report_two["phase_1"]["status"] == 0
    assert report_one["process_id"] != report_two["process_id"] or report_one["seed"] != report_two["seed"]
    verify_selection(rows_one, records, edges, targets_from_contract(contract, config["upper_quotas"]), quota, config["solver"]["hash_cost_namespace"])


def test_registry_interface_has_four_splits_and_defers_sample_id():
    records, edges, contract, config = _synthetic_problem()
    quota, _ = solve_quota_search(records, edges, contract, config, seed=1)
    rows, _ = solve_allocation_rows(records, edges, contract, config, quota, seed=1)
    interface = project_selection_to_native_interface(rows, records)
    assert set(interface["splits"]) == {"hotpotqa_screen", "hotpotqa_confirm", "musique_screen", "musique_confirm"}
    for split_rows in interface["splits"].values():
        for row in split_rows:
            assert set(row) == {"dataset_id", "partition", "source_index", "source_projection_sha256", "selection_rank", "stratum"}
            assert "sample_id" not in row
    assert interface["requires_downstream_fields"] == ["sample_id", "selection_rank_sha256"]
    assert all(interface["screen_confirm_disjoint"].values())


def test_tamper_nonfinite_duplicate_and_bool_are_rejected(monkeypatch, tmp_path):
    # Regression for the entrypoint bootstrap audit: a solver-capable process
    # must be launched with an explicitly supplied, initially absent cache
    # prefix and bytecode writes disabled.  An occupied prefix fails closed.
    current_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    assert require_fresh_python_bootstrap() == current_prefix
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path))
    with pytest.raises(ProtocolError, match="(absent|match)"):
        require_fresh_python_bootstrap()
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", current_prefix)
    with pytest.raises(ProtocolError, match="nonfinite"):
        strict_json_loads('{"x":NaN}')
    with pytest.raises(ProtocolError, match="duplicate"):
        strict_json_loads('{"x":1,"x":2}')
    records, edges, contract, config = _synthetic_problem()
    bad = copy.deepcopy(config)
    bad["upper_quotas"]["0"] = True
    with pytest.raises(ProtocolError, match="upper quota"):
        targets_from_contract(contract, bad["upper_quotas"])


def test_tamper_ratio_and_target_order_fail_closed():
    records, edges, contract, config = _synthetic_problem()
    targets = targets_from_contract(contract, config["upper_quotas"])
    quota, proof = solve_quota_search(records, edges, contract, config, seed=1)
    bad_quota = dict(quota)
    bad_quota["1"] = bad_quota["1"] + 2
    with pytest.raises(ProtocolError, match="ratio"):
        validate_quota_vector(bad_quota, targets)
    bad_config = json.loads((ROOT / "config.json").read_text())
    bad_config["target_order"] = list(reversed(bad_config["target_order"]))
    with pytest.raises(ProtocolError, match="target order"):
        validate_config_shape(bad_config)


def test_tamper_external_binding_is_detected_without_mutating_source():
    config = load_config(ROOT / "config.json")
    bad = copy.deepcopy(config)
    bad["thresholds"]["passage_min_jaccard"] = 0.31
    with pytest.raises(ProtocolError, match="threshold"):
        load_input_bundle(bad)


def test_three_placeholder_markers_are_independent_and_not_formal_passes():
    config = load_config(ROOT / "config.json")
    for stage in ("quota_search", "allocation", "deep_validation"):
        marker = validate_stage(stage, ROOT, config)
        assert marker["atomic_last_completion_marker"] is True
        assert "placeholder" in marker["status"]
        receipt = json.loads((ROOT / config["outputs"][stage] / "receipt.json").read_text())
        assert receipt["completed"] is False
        assert receipt["formal_execution_performed"] is False
        assert receipt["gpu_execution_performed"] is False


def test_tamper_stage_extra_file_and_sidecar_fail_closed(tmp_path):
    # Build a tiny independent publication so this test never writes to the
    # immutable experiment artifacts.
    from src.common import atomic_json, claim_artifact, finalize_publication, validate_publication

    out = tmp_path / "stage"
    identity = {"protocol_id": "synthetic", "artifact_kind": "stage"}
    claim_artifact(out, identity)
    atomic_json(out / "receipt.json", {"ok": True})
    finalize_publication(out, ["receipt.json"], identity=identity, status="done")
    (out / "extra.json").write_text("{}")
    with pytest.raises(ProtocolError, match="inventory"):
        validate_publication(out, ["receipt.json"], identity=identity, status="done")


def test_tamper_solver_status_and_nonbinary_incumbent_fail_closed(monkeypatch):
    import src.quota as quota_module

    records, edges, contract, config = _synthetic_problem()
    targets = targets_from_contract(contract, config["upper_quotas"])
    matrix = build_matrix(records, edges, targets, with_lambda=False)
    class BadResult:
        status = 1
        success = False
        message = "time limit"
        x = None
        fun = None
    monkeypatch.setattr(quota_module, "milp", lambda *args, **kwargs: BadResult())
    with pytest.raises(ProtocolError, match="did not prove optimality"):
        quota_module._run_milp(
            np.zeros(matrix.constraints.A.shape[1]),
            matrix.layout,
            matrix,
            seed=1,
            config=config,
            phase="tamper",
        )
    class NonBinary:
        status = 0
        success = True
        message = "Optimization terminated successfully. (HiGHS Status 7: Optimal)"
        x = np.full(matrix.constraints.A.shape[1], 0.5)
        fun = 0.0
    with pytest.raises(ProtocolError, match="not binary"):
        quota_module._validated_incumbent(NonBinary(), matrix, phase="tamper_nonbinary")
