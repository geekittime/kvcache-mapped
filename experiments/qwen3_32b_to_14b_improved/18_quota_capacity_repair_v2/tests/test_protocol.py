from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
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
from src.common import (
    ProtocolError,
    canonical_bytes,
    config_file_binding,
    formal_config_path,
    implementation_inventory,
    implementation_inventory_digest,
    require_fresh_python_bootstrap,
    safe_output_child,
    strict_json_loads,
    validate_runtime_binding,
    verify_config_file_binding,
)
from src.inputs import InputBundle, load_config, load_input_bundle
from src.preflight import preflight, validate_config_shape, validate_formal_config, validate_preformal_stop_binding
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


def test_three_placeholder_markers_are_independent_and_not_formal_passes(tmp_path):
    config = load_config(ROOT / "config.json")
    # The v2 sibling deliberately starts without copied v1 artifacts.  Build
    # isolated placeholders under pytest's temporary root solely to exercise
    # the marker validator; no historical output is imported or overwritten.
    pf = preflight(ROOT / "config.json")
    from src.runner import publish_placeholder_stages
    (tmp_path / "artifacts" / "validations").mkdir(parents=True)
    publish_placeholder_stages(tmp_path, config, pf)
    for stage in ("quota_search", "allocation", "deep_validation"):
        marker = validate_stage(stage, tmp_path, config)
        assert marker["atomic_last_completion_marker"] is True
        assert "placeholder" in marker["status"]
        receipt = json.loads((tmp_path / config["outputs"][stage] / "receipt.json").read_text())
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


def test_formal_config_and_pinned_venv_bindings_are_exact(tmp_path):
    config_path = ROOT / "config.json"
    config = load_config(config_path)
    binding = validate_formal_config(config_path, ROOT, config)
    assert binding == config_file_binding(config_path)
    assert formal_config_path(ROOT, config_path) == config_path
    assert validate_runtime_binding(config["runtime_binding"])["python_executable"] == config["runtime_binding"]["python_executable"]
    with pytest.raises(ProtocolError, match="exactly"):
        formal_config_path(ROOT, ROOT / "config.json.backup")
    alternate = tmp_path / "config.json"
    alternate.write_bytes(config_path.read_bytes())
    with pytest.raises(ProtocolError, match="exactly"):
        formal_config_path(ROOT, alternate)
    symlink = tmp_path / "config-link.json"
    symlink.symlink_to(config_path)
    with pytest.raises(ProtocolError, match="(symlink|regular)"):
        verify_config_file_binding(symlink, {"path": str(symlink), "nbytes": binding["nbytes"], "sha256": binding["sha256"]})


def test_output_publication_rejects_symlink_and_dangling_parent_components(tmp_path):
    from src.common import atomic_json, claim_artifact

    root = tmp_path / "root"
    root.mkdir()
    real = tmp_path / "real"
    real.mkdir()
    (root / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(ProtocolError, match="symlink"):
        safe_output_child(root, "link/payload.json")
    with pytest.raises(ProtocolError, match="symlink"):
        claim_artifact(root / "link" / "stage", {"protocol_id": "x", "artifact_kind": "y"})
    dangling = root / "dangling"
    dangling.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(ProtocolError, match="symlink"):
        safe_output_child(root, "dangling/payload.json")
    with pytest.raises(ProtocolError, match="symlink"):
        atomic_json(root / "link" / "escape.json", {"x": 1})


def test_deep_allocation_replica_gate_rejects_empty_duplicate_and_typed_tamper(monkeypatch):
    import src.deep as deep

    config = load_config(ROOT / "config.json")
    binding = config_file_binding(ROOT / "config.json")
    runtime_identity = validate_runtime_binding(config["runtime_binding"])
    selection_spec = {"nbytes": 0, "sha256": hashlib.sha256(b"").hexdigest()}

    def replica(seed: int, pid: int):
        worker = {
            "process_id": pid,
            "python_hash_seed": str(seed),
            "seed": seed,
            "selection": dict(selection_spec),
            "solver": {"execution": {"formal_config": dict(binding)}},
            "config_binding": dict(binding),
            "runtime_identity": dict(runtime_identity),
            "preformal_stop_binding": dict(config["preformal_stop_binding"]),
        }
        return {
            "seed": seed,
            "python_hash_seed": str(seed),
            "process_id": pid,
            "worker": worker,
            "config_binding": dict(binding),
            "preformal_stop_binding": dict(config["preformal_stop_binding"]),
        }

    report = {"replicas": [replica(1, 101), replica(777, 202)]}
    monkeypatch.setattr(deep, "validate_allocation_report", lambda *args, **kwargs: None)
    deep._validate_allocation_replicas(report, config, binding, selection_spec)
    for bad in (
        {"replicas": []},
        {"replicas": [replica(1, 101), replica(1, 202)]},
        {"replicas": [replica(1, 101), replica(777, 101)]},
        {"replicas": [replica(1, 101), {**replica(777, 202), "process_id": True}]},
    ):
        with pytest.raises(ProtocolError):
            deep._validate_allocation_replicas(bad, config, binding, selection_spec)


def test_preformal_snapshot_binding_is_strict_and_previous_transition_is_explicit():
    config = load_config(ROOT / "config.json")
    checked = validate_preformal_stop_binding(ROOT / "config.json", ROOT, config)
    assert checked["inventory_verified"] is (config["preformal_stop_binding"]["role"] == "active_snapshot")
    bad = copy.deepcopy(config)
    bad["preformal_stop_binding"]["sha256"] = "0" * 64
    with pytest.raises(ProtocolError, match="binding|marker"):
        validate_preformal_stop_binding(ROOT / "config.json", ROOT, bad)
    bad_shape = copy.deepcopy(config)
    bad_shape["preformal_stop_binding"]["extra"] = True
    with pytest.raises(ProtocolError, match="keys"):
        validate_config_shape(bad_shape)


def test_readme_fresh_prefix_fragment_is_executed_not_only_parsed(tmp_path):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert 'PREFLIGHT_CACHE="$PREFLIGHT_PARENT/fresh"' in readme
    py = load_config(ROOT / "config.json")["runtime_binding"]["python_executable"]
    snippet = f'''set -eu
PARENT=$(mktemp -d /tmp/qcr_v2_bootstrap_test.XXXXXX)
trap 'rmdir "$PARENT"' EXIT
PREFIX="$PARENT/fresh"
test ! -e "$PREFIX"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=1 PYTHONPYCACHEPREFIX="$PREFIX" "{py}" -c 'import os,sys; assert sys.pycache_prefix == os.environ["PYTHONPYCACHEPREFIX"]; assert sys.dont_write_bytecode is True; assert os.environ["CUDA_VISIBLE_DEVICES"] == ""'
test ! -e "$PREFIX"
'''
    result = subprocess.run(["bash", "-eu", "-o", "pipefail", "-c", snippet], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_marker_payload_strips_mutable_config_echoes():
    import scripts.publish_ready_snapshot as publisher

    payload = {"formal_config": {"sha256": "old"}, "nested": [{"formal_config": 1, "ok": True}]}
    assert publisher._strip_config_bindings(payload) == {"nested": [{"ok": True}]}


def test_publishers_reject_nonregular_implementation_entries(monkeypatch, tmp_path):
    import scripts.publish_ready_snapshot as ready_publisher
    import scripts.run_preformal as preformal_publisher

    os.mkfifo(tmp_path / "malicious.fifo")
    monkeypatch.setattr(ready_publisher, "ROOT", tmp_path)
    with pytest.raises(ProtocolError, match="nonregular"):
        ready_publisher._inventory()
    monkeypatch.setattr(preformal_publisher, "ROOT", tmp_path)
    with pytest.raises(ProtocolError, match="nonregular"):
        preformal_publisher._inventory()


def test_all_workers_fail_closed_on_tampered_formal_config_binding(tmp_path):
    config_path = ROOT / "config.json"
    config = load_config(config_path)
    binding = config_file_binding(config_path)
    python = config["runtime_binding"]["python_executable"]
    bad_sha = "0" * 64 if binding["sha256"] != "0" * 64 else "1" * 64
    scripts_and_extra = [
        (ROOT / "scripts" / "quota_worker.py", ["--output", str(tmp_path / "quota.json")]),
        (ROOT / "scripts" / "allocation_worker.py", ["--quota", str(tmp_path / "quota.json"), "--output", str(tmp_path / "allocation.jsonl")]),
        (ROOT / "scripts" / "deep_validate.py", []),
    ]
    for script, extra in scripts_and_extra:
        prefix = tmp_path / f"pycache-{script.stem}"
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(prefix),
            "PYTHONHASHSEED": "1",
            "FORMAL_CONFIG_PATH": str(config_path),
            "FORMAL_CONFIG_NBYTES": str(binding["nbytes"]),
            "FORMAL_CONFIG_SHA256": binding["sha256"],
            "FORMAL_PYTHON_EXECUTABLE": python,
            "VIRTUAL_ENV": config["runtime_binding"]["python_prefix"],
            "QUOTA_REPAIR_SEED": "1",
            "HYPERGRAPH_HIGHS_RANDOM_SEED": "1",
        }
        command = [
            python,
            str(script),
            "--config",
            str(config_path),
            "--config-nbytes",
            str(binding["nbytes"]),
            "--config-sha256",
            bad_sha,
            *extra,
        ]
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
        assert result.returncode != 0
        assert "binding" in result.stderr.lower() or "BLOCKED" in result.stderr
        assert not prefix.exists()


def test_runner_has_no_implicit_config_fallback_and_passes_explicit_path():
    from src.runner import _worker_config_args

    config_path = ROOT / "config.json"
    binding = config_file_binding(config_path)
    args = _worker_config_args(config_path, binding)
    assert args == ["--config", str(config_path), "--config-nbytes", str(binding["nbytes"]), "--config-sha256", binding["sha256"]]
    source = (ROOT / "src" / "runner.py").read_text(encoding="utf-8")
    assert 'str(root / "config.json")' not in source
    assert "def _run_ready_repair(" in source and "config_path: Path" in source


def test_readme_commands_match_canonical_v16_config():
    """Keep the operator-facing commands bound to this sibling and marker."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    config = load_config(ROOT / "config.json")
    assert readme.startswith("# 18 — quota capacity repair v2")
    assert "16_quota_capacity_repair_v1/config_ready_v2" not in readme
    assert "config_ready_v2" not in readme
    assert "scripts/preflight.py --config \"$ROOT/config.json\"" in readme
    assert 'scripts/publish_ready_snapshot.py --config "$ROOT/config.json" --version v16' in readme
    assert config["outputs"]["preformal_stop"] == "artifacts/preformal/PRE_FORMAL_STOP_v16.json"
    for script in ("publish_ready_snapshot.py", "run_preformal.py", "publish_preformal.py"):
        source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert 'parser.add_argument("--version", default="v16")' in source


def test_ready_repair_wires_two_quota_and_two_allocation_workers(monkeypatch, tmp_path):
    """Exercise the real orchestration with subprocesses replaced by a strict oracle.

    The four child envelopes are intentionally checked by ``_run_ready_repair``
    itself.  This catches a regression where one stage silently falls back to
    another config path or interpreter without spending the full MILP cost.
    """

    import types
    import src.runner as runner

    records, edges, contract, synthetic = _synthetic_problem()
    base = load_config(ROOT / "config.json")
    config = copy.deepcopy(base)
    config["upper_quotas"] = {str(index): (2 if index % 2 == 0 else 4) for index in range(18)}
    config["outputs"] = {
        "quota_search": "artifacts/quota_search",
        "allocation": "artifacts/allocation",
        "deep_validation": "artifacts/validations/deep",
        "preformal_stop": "artifacts/preformal/PRE_FORMAL_STOP_v12.json",
    }
    config_path = tmp_path / "config.json"
    config_path.write_bytes(canonical_bytes(config))
    config_binding = config_file_binding(config_path)
    (tmp_path / "artifacts" / "validations").mkdir(parents=True)
    bundle = InputBundle(
        config=config,
        graph_dir=tmp_path,
        records=tuple(records),
        hyperedges=tuple(edges),
        contract=contract,
        graph_report={},
        bindings={},
        allocation_deep_ready=True,
        allocation_deep_binding={"ready": True, "placeholder": False},
    )
    quota_vector = {str(index): (2 if index % 2 == 0 else 4) for index in range(18)}
    runtime_identity = validate_runtime_binding(config["runtime_binding"])
    calls = []

    def fake_run(command, *, cwd, env, text, capture_output, check):
        assert cwd == tmp_path
        assert command[0] == config["runtime_binding"]["python_executable"]
        assert command[1].startswith(str(tmp_path / "scripts"))
        assert env["FORMAL_CONFIG_PATH"] == str(config_path)
        assert env["FORMAL_CONFIG_NBYTES"] == str(config_binding["nbytes"])
        assert env["FORMAL_CONFIG_SHA256"] == config_binding["sha256"]
        assert env["FORMAL_PYTHON_EXECUTABLE"] == config["runtime_binding"]["python_executable"]
        assert env["VIRTUAL_ENV"] == config["runtime_binding"]["python_prefix"]
        assert env["CUDA_VISIBLE_DEVICES"] == ""
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
        assert env["PYTHONPYCACHEPREFIX"] == os.environ["PYTHONPYCACHEPREFIX"]
        parsed = {command[index]: command[index + 1] for index in range(2, len(command) - 1, 2) if command[index].startswith("--")}
        output = Path(parsed["--output"])
        if command[1].endswith("quota_worker.py"):
            seed = env["QUOTA_REPAIR_SEED"]
            calls.append(("quota", seed, list(command), dict(env)))
            result = {"quota_vector": quota_vector, "proof": {"status": "quota_search_proven_optimal"}}
            output.write_bytes(canonical_bytes(result))
            envelope = {"process_id": 1000 + int(seed), "python_hash_seed": seed, "seed": int(seed), "result": result, "config_binding": config_binding, "runtime_identity": runtime_identity, "preformal_stop_binding": config["preformal_stop_binding"]}
        else:
            seed = env["HYPERGRAPH_HIGHS_RANDOM_SEED"]
            calls.append(("allocation", seed, list(command), dict(env)))
            output.write_bytes(b"")
            envelope = {
                "process_id": 2000 + int(seed),
                "python_hash_seed": seed,
                "seed": int(seed),
                "selection": {"nbytes": 0, "sha256": __import__("hashlib").sha256(b"").hexdigest()},
                "solver": {"execution": {"formal_config": config_binding}},
                "config_binding": config_binding,
                "runtime_identity": runtime_identity,
                "preformal_stop_binding": config["preformal_stop_binding"],
            }
        return types.SimpleNamespace(returncode=0, stderr="", stdout=json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    # The synthetic orchestration root has no published source snapshot; use
    # an explicit test-only oracle for the production inventory gate.
    monkeypatch.setattr(runner, "_active_implementation_snapshot", lambda *args, **kwargs: ([], "synthetic-inventory"))
    monkeypatch.setattr(runner, "_assert_implementation_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "validate_quota_vector", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "validate_quota_proof", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "validate_allocation_report", lambda *args, **kwargs: None)
    result = runner._run_ready_repair(bundle, config, tmp_path, config_path, config_binding)
    assert result["status"] == "CPU_REPAIR_COMPLETE_A_B_C_D_AWAITING_DEEP"
    assert [(kind, seed) for kind, seed, _, _ in calls] == [("quota", "1"), ("quota", "777"), ("allocation", "1"), ("allocation", "777")]
    for _, _, command, _ in calls:
        assert command[2:8] == ["--config", str(config_path), "--config-nbytes", str(config_binding["nbytes"]), "--config-sha256", config_binding["sha256"]]
    assert (tmp_path / "artifacts" / "quota_search" / "PUBLISHED.json").is_file()
    assert (tmp_path / "artifacts" / "allocation" / "PUBLISHED.json").is_file()

    # Deep validation must inspect every persisted replica envelope, not only
    # the first seed.  Stub the expensive semantic checks while retaining the
    # real publication/config/runtime checks and count each child explicitly.
    import src.deep as deep
    deep_calls = []
    original_config_check = deep._validate_child_config_binding
    original_runtime_check = deep._validate_child_runtime_identity
    monkeypatch.setattr(
        deep,
        "_validate_child_config_binding",
        lambda value, expected, *, name: (deep_calls.append(("config", name)), original_config_check(value, expected, name=name))[1],
    )
    monkeypatch.setattr(
        deep,
        "_validate_child_runtime_identity",
        lambda value, expected, *, name: (deep_calls.append(("runtime", name)), original_runtime_check(value, expected, name=name))[1],
    )
    monkeypatch.setattr(deep, "validate_quota_vector", lambda *args, **kwargs: {})
    monkeypatch.setattr(deep, "validate_quota_proof", lambda *args, **kwargs: None)
    monkeypatch.setattr(deep, "verify_selection", lambda *args, **kwargs: {})
    monkeypatch.setattr(deep, "validate_allocation_report", lambda *args, **kwargs: None)
    deep_result = deep.validate_ready_stages(
        bundle,
        config,
        tmp_path,
        config_path=config_path,
        config_binding=config_binding,
    )
    assert deep_result["formal_config"] == config_binding
    assert [name for kind, name in deep_calls if kind == "config"] == [
        "quota replica 0",
        "quota replica 1",
        "allocation replica 1",
        "allocation worker 1",
        "allocation replica 777",
        "allocation worker 777",
    ]
    assert [name for kind, name in deep_calls if kind == "runtime"] == [
        "quota replica 0",
        "quota replica 1",
        "allocation replica 1",
        "allocation replica 777",
    ]
