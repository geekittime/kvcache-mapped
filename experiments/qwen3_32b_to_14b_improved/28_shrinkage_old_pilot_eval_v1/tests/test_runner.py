from __future__ import annotations

import json
import copy
from pathlib import Path
import subprocess
import sys
import types

import pytest

from src.runner import (
    BASELINE_MODES,
    ProtocolError,
    candidate_result_path,
    canonical_sha256,
    is_sha256,
    load_config,
    old_result_path,
    source_byte_closure,
    token_ids_sha256,
)
import src.runner as runner


def test_config_and_mode_contract_are_frozen():
    config = load_config()
    assert config["protocol_id"] == "qwen3_32b_to_14b_shrinkage_old_pilot_eval_v1"
    assert config["experiment_id"].endswith("attempt28_v1")
    assert all("attempt28" in value for value in config["outputs"].values())
    assert config["protocol"]["candidate_mode"] == "shrinkage_r1"
    assert BASELINE_MODES == {"native14": "target_standalone", "nvidia_k20": "mapped_r1"}
    assert config["runtime"]["source_layer_counts"] == [12, 13, 14, 13, 12]
    assert config["runtime"]["target_layer_counts"] == [13, 14, 13]


def test_sha256_shape_is_strict():
    assert is_sha256("0" * 64)
    assert not is_sha256("0" * 63)
    assert not is_sha256("0" * 63 + "G")
    assert not is_sha256(123)


def test_result_paths_are_independent_from_old03():
    row = {"dataset_id": "hotpotqa_train", "sample_id": "abc"}
    old = old_result_path(row, "mapped_r1")
    new = candidate_result_path(Path("/tmp/dir28"), row)
    assert "03_tail_refresh" in str(old)
    assert "21_shrinkage_old_pilot_eval_v1" not in str(old)
    assert "shrinkage_r1" in str(new)
    assert old != new


def test_token_hash_is_canonical():
    assert token_ids_sha256([1, 2, 3]) == canonical_sha256([1, 2, 3])
    assert token_ids_sha256([1, 2, 3]) != token_ids_sha256([1, 2, 4])


def test_source_closure_has_local_runner_and_shared_engine():
    closure = source_byte_closure()
    paths = {entry["path"] for entry in closure["files"]}
    assert any(path.endswith("28_shrinkage_old_pilot_eval_v1/src/runner.py") for path in paths)
    assert any(path.endswith("src/kvcache_transfer/evaluation.py") for path in paths)
    assert len(closure["digest"]) == 64
    assert all(is_sha256(entry["sha256"]) for entry in closure["files"])


def test_preflight_does_not_import_torch():
    code = """
import sys
sys.path.insert(0, '.')
import src.runner
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[1],
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": "/tmp/dir28_test_subprocess_pycache", **{"PATH": "/usr/bin:/bin"}},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_baseline_rerun_is_recorded_in_protocol():
    config = load_config()
    assert config["outputs"]["run_root"].startswith(str(Path(__file__).parents[1]))
    assert config["upstream"]["baseline_run_root"].startswith("/home/wby/llm-cache")


def test_prior_failure_chain_binds_25_24_and_nested_23():
    config = load_config()
    prior = config["upstream"]["prior_attempt"]
    assert prior["attempt_id"].endswith("attempt26_v1")
    assert prior["request_fingerprint"] == "051376d7df2e05465740559b4c116e08d677390a8b542dc4f99fa1dda03f35ff"
    nested25 = prior["prior25_evidence"]
    assert nested25["attempt_id"].endswith("attempt25_v1")
    assert nested25["request_fingerprint"] == "80c74d0a2167dad62dd65eead0c86bf3c0cdc844699ac18aa835991f6bdb602c"
    assert nested25["candidate_count"] == 96
    assert nested25["candidate_results_sha256"] == "ad764524eee4a86d5c74339ab7c7f688e3d2bf114de6a75b44d42944477cb024"
    nested24 = nested25["prior24_evidence"]
    assert nested24["attempt_id"].endswith("attempt24_v1")
    nested = nested24["prior23_evidence"]
    assert nested["attempt_id"].endswith("attempt23_v1")
    assert nested["candidate_inventory"]["path"].endswith("24_shrinkage_old_pilot_eval_v1/artifacts/prior23_candidate_inventory.json")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_root", "/tmp/not-dir25"),
        ("run_root", "/tmp/not-run-root"),
        ("status_path", "/tmp/not-status.json"),
        ("failure_stage", "after_cuda"),
        ("failure_reason", "changed reason"),
        ("error_type", "ValueError"),
        ("error", "changed error"),
        ("request_fingerprint", "0" * 64),
    ],
)
def test_prior25_mutation_is_rejected(field, value):
    config = load_config()
    prior25 = copy.deepcopy(config["upstream"]["prior_attempt"]["prior25_evidence"])
    prior25[field] = value
    with pytest.raises(ProtocolError):
        runner._validate_prior25_evidence(prior25)


def test_prior24_exact_root_filename_and_numeric_mutations_are_rejected():
    config = load_config()
    prior24 = config["upstream"]["prior_attempt"]["prior25_evidence"]["prior24_evidence"]
    mutations = (
        ("experiment_root", str(Path(prior24["experiment_root"]).parent)),
        ("preflight", {**prior24["preflight"], "path": prior24["preflight"]["path"].replace("attempt24_v1.json", "attempt24_wrong.json")}),
        ("v2_preflight_interrupted_exit_code", True),
        ("candidate_count", True),
    )
    for field, value in mutations:
        candidate = copy.deepcopy(prior24)
        candidate[field] = value
        with pytest.raises(ProtocolError):
            runner._validate_prior24_evidence(candidate)


def test_prior23_exact_root_filename_and_numeric_mutations_are_rejected():
    config = load_config()
    prior23 = config["upstream"]["prior_attempt"]["prior25_evidence"]["prior24_evidence"]["prior23_evidence"]
    mutations = (
        ("experiment_root", str(Path(prior23["experiment_root"]).parent)),
        ("status", {**prior23["status"], "path": prior23["status"]["path"].replace("status.json", "status_wrong.json")}),
        ("candidate_count", True),
    )
    for field, value in mutations:
        candidate = copy.deepcopy(prior23)
        candidate[field] = value
        with pytest.raises(ProtocolError):
            runner._validate_prior23_evidence(candidate)


def test_stop_receipt_binding_and_tamper_guard(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    stop_path = tmp_path / "artifacts" / "STOP.json"
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    config = {
        "config_path": str(config_path),
        "protocol_id": "synthetic-protocol-v1",
        "outputs": {"stop_receipt": str(stop_path)},
        "runtime": {"global_gpu_lock": str(tmp_path / "gpu.lock")},
    }
    definition = {"request_fingerprint": "1" * 64, "experiment_id": "synthetic-attempt28"}
    preflight_ref = {"path": "preflight.json", "nbytes": 1, "sha256": "0" * 64}
    value = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "STOP_WAITING_FOR_OPERATOR_GPU_CONFIRMATION",
        "request_fingerprint": definition["request_fingerprint"],
        "preflight": preflight_ref,
        "gpu_used": False,
        "cuda_api_called": False,
        "execute_authorized": False,
        "唯一启动命令": f"{config_path} --execute",
        "lock": config["runtime"]["global_gpu_lock"],
    }
    runner.atomic_json(stop_path, value)
    runner.write_sidecar(stop_path)
    assert runner._validate_stop_receipt(config, definition, preflight_ref) == stop_path.resolve()
    value["execute_authorized"] = True
    stop_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ProtocolError, match="sidecar mismatch|STOP receipt"):
        runner._validate_stop_receipt(config, definition, preflight_ref)


def _failure_fixture(tmp_path: Path, *, fingerprint: str = "f" * 64):
    output = tmp_path / "run"
    destination = output / "results" / "hotpotqa_train" / "shrinkage_r1" / "abc.json"
    destination.parent.mkdir(parents=True)
    row = {
        "dataset_id": "hotpotqa_train",
        "sample_id": "sample-1",
        "source_record_sha256": "1" * 64,
        "prompt_token_ids_sha256": "2" * 64,
        "prompt_token_count": 3,
    }
    definition = {"request_fingerprint": fingerprint}
    failure = destination.with_suffix(".failure.json")
    runner.atomic_json(
        failure,
        {
            "schema_version": 1,
            "status": "failed",
            "request_fingerprint": fingerprint,
            "dataset_id": row["dataset_id"],
            "sample_id": row["sample_id"],
            "mode": "shrinkage_r1",
            "error_type": "RuntimeError",
            "error": "synthetic",
            "traceback": "trace",
        },
    )
    return output, destination, row, definition, failure


def test_bound_failure_is_atomically_marked_superseded(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output, destination, row, definition, failure = _failure_fixture(tmp_path)
    marker = runner._supersede_failure_marker(destination, row, definition, output)
    assert marker is not None
    assert not failure.exists()
    superseded = destination.with_suffix(".failure.superseded.json")
    assert superseded.is_file()
    assert json.loads(superseded.read_text())["status"] == "superseded"
    runner._assert_no_stale_failure_markers(output)


def test_mismatched_failure_binding_rejects_publication(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output, destination, row, definition, failure = _failure_fixture(tmp_path, fingerprint="a" * 64)
    definition["request_fingerprint"] = "b" * 64
    with pytest.raises(ProtocolError):
        runner._supersede_failure_marker(destination, row, definition, output)
    assert failure.exists()


def test_malformed_superseded_marker_blocks_terminal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "run"
    marker = output / "results" / "hotpotqa_train" / "shrinkage_r1" / "abc.failure.superseded.json"
    marker.parent.mkdir(parents=True)
    runner.atomic_json(marker, {"status": "superseded"})
    with pytest.raises(ProtocolError, match="malformed superseded"):
        runner._assert_no_stale_failure_markers(output)


def test_output_root_rejects_symlink_component(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ProtocolError):
        runner._assert_output_root(link / "run")


def test_verify_ref_rejects_symlinked_parent(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    target = real / "artifact.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    ref = runner.artifact(target)
    ref["path"] = str(link / target.name)
    with pytest.raises(ProtocolError, match="symlink"):
        runner.verify_ref(ref, "synthetic upstream")


def test_terminal_recheck_requires_bound_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    definition = {"request_fingerprint": "c" * 64}
    assert runner._terminal_completion_summary(output, definition) is None
    runner.atomic_json(
        output / "completion_receipt.json",
        {"status": "complete", "request_fingerprint": definition["request_fingerprint"]},
    )
    runner.write_sidecar(output / "completion_receipt.json")
    with pytest.raises(ProtocolError):
        runner._terminal_completion_summary(output, definition)


def test_completion_receipt_and_sidecar_publish_and_tamper(monkeypatch, tmp_path: Path):
    """The terminal pair is explicit, durable, and tamper-evident."""

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "results").mkdir()
    definition = {"request_fingerprint": "d" * 64, "experiment_id": "synthetic-attempt28", "sample_count": 0}
    completion_path, sidecar_path = runner._completion_paths(output)
    assert completion_path == output / "completion_receipt.json"
    assert sidecar_path == output / "completion_receipt.json.sha256.json"
    definition_path = output / "run_definition.json"
    runner.immutable_json(definition_path, {"synthetic": True})
    runner.write_sidecar(definition_path)
    summary_path = output / "paired_summary.json"
    runner.atomic_json(summary_path, {"status": "complete", "request_fingerprint": definition["request_fingerprint"]})
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": definition["experiment_id"],
        "attempt_id": definition["experiment_id"],
        "request_fingerprint": definition["request_fingerprint"],
        "candidate_result_count": 0,
        "expected_candidate_result_count": 0,
        "run_definition": runner.artifact(definition_path),
        "paired_summary": runner.artifact(summary_path),
        "candidate_results": [],
        "superseded_failure_markers": [],
    }
    runner.immutable_json(completion_path, receipt)
    runner.write_sidecar(completion_path)
    assert runner._terminal_completion_summary(output, definition)["status"] == "complete"
    sidecar_path.write_text(json.dumps({"path": completion_path.name, "nbytes": 0, "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ProtocolError, match="sidecar mismatch"):
        runner._terminal_completion_summary(output, definition)


def test_orphan_completion_sidecar_blocks_execution(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    _, sidecar_path = runner._completion_paths(output)
    sidecar_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="orphan completion sidecar"):
        runner._terminal_completion_summary(output, {"request_fingerprint": "e" * 64})


def test_completion_without_sidecar_fails_closed(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    completion, _ = runner._completion_paths(output)
    completion.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProtocolError, match="completion receipt is not a regular|sidecar"):
        runner._terminal_completion_summary(output, {"request_fingerprint": "f" * 64, "experiment_id": "x", "sample_count": 0})


def test_execute_binds_completion_paths_before_finalization():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    execute_pos = source.index("def execute(")
    bind_pos = source.index('completion_path = output / "completion_receipt.json"', execute_pos)
    guard_pos = source.index('_guard_output_component(completion_path, output, label="completion receipt")', bind_pos)
    publish_pos = source.index("immutable_json(completion_path, receipt)", execute_pos)
    sidecar_pos = source.index("verify_sidecar(completion_path)", publish_pos)
    assert execute_pos < bind_pos < guard_pos < publish_pos < sidecar_pos


def test_execute_contains_in_lock_terminal_recheck():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    lock_pos = source.index("lock = acquire_gpu_lock(")
    recheck_pos = source.index("terminal = _terminal_completion_summary(output, definition)", lock_pos)
    assert recheck_pos > lock_pos


def test_real_shrinkage_loader_cpu_regression():
    cls = runner._load_shrinkage_builder()
    assert cls.__name__ == "ShrinkageMappedPrefixCacheBuilder"
    assert "dir28_shrinkage_runtime" in sys.modules


def test_lazy_loader_cleans_sys_modules_on_import_failure(tmp_path: Path):
    bad = tmp_path / "bad_loader.py"
    bad.write_text("raise RuntimeError('synthetic loader failure')\n", encoding="utf-8")
    sys.modules.pop("dir28_synthetic_bad_loader", None)
    with pytest.raises(RuntimeError, match="synthetic loader failure"):
        runner._load_module_from_path("dir28_synthetic_bad_loader", bad)
    assert "dir28_synthetic_bad_loader" not in sys.modules


def test_execute_pre_gpu_smoke_uses_mapping_and_stops_before_model_load(monkeypatch, tmp_path: Path):
    """Exercise the real execute path through the CUDA guard with fake models.

    This catches the dir25 ``dict(PosixPath)`` regression while proving that
    the idle/model imports are reached only after the bound preflight and STOP
    receipts.  The fake torch reports zero devices, so no CUDA API or model
    weights are touched.
    """

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    preflight_path = tmp_path / "artifacts" / "preflight.json"
    stop_path = tmp_path / "artifacts" / "STOP.json"
    lock_path = tmp_path / "gpu.lock"
    output = tmp_path / "run"
    definition = {
        "request_fingerprint": "a" * 64,
        "experiment_id": "synthetic-attempt28",
        "sample_count": 0,
        "baseline_binding_digest": "b" * 64,
    }
    config = {
        "config_path": str(config_path),
        "protocol_id": "synthetic-protocol-v1",
        "outputs": {"run_root": str(output), "preflight_receipt": str(preflight_path), "stop_receipt": str(stop_path)},
        "runtime": {"global_gpu_lock": str(lock_path), "cuda_visible_devices": "0,1,2,3,4,5,6,7"},
    }
    preflight_value = {"schema_version": 1, "protocol_id": config["protocol_id"], "status": "passed_cpu_only_preflight", "gpu_used": False, "cuda_api_called": False, "request_fingerprint": definition["request_fingerprint"]}
    preflight_ref = runner.atomic_json(preflight_path, preflight_value)
    runner.write_sidecar(preflight_path)
    stop_value = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "STOP_WAITING_FOR_OPERATOR_GPU_CONFIRMATION",
        "request_fingerprint": definition["request_fingerprint"],
        "preflight": preflight_ref,
        "gpu_used": False,
        "cuda_api_called": False,
        "execute_authorized": False,
        "唯一启动命令": f"{config_path} --execute",
        "lock": str(lock_path),
    }
    runner.atomic_json(stop_path, stop_value)
    runner.write_sidecar(stop_path)
    monkeypatch.setattr(runner, "load_config", lambda _path=None: config)
    monkeypatch.setattr(runner, "_validate_bytecode_environment", lambda _config: None)
    monkeypatch.setattr(runner, "_preflight_artifacts", lambda _config: ([], {}, {}, {}, {"source_closure": {"digest": "c" * 64}, "old_protocol": {}}, {}))
    monkeypatch.setattr(runner, "build_definition", lambda *args, **kwargs: definition)
    monkeypatch.setattr(runner, "idle_gate", lambda _config, _output: {"status": "pass", "samples": []})
    fake_cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = object
    fake_eval = types.ModuleType("kvcache_transfer.evaluation")
    fake_eval.TransferEvaluationEngine = object
    fake_modeling = types.ModuleType("kvcache_transfer.modeling")
    fake_modeling.load_qwen3 = lambda *args, **kwargs: None
    fake_pkg = types.ModuleType("kvcache_transfer")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "kvcache_transfer", fake_pkg)
    monkeypatch.setitem(sys.modules, "kvcache_transfer.evaluation", fake_eval)
    monkeypatch.setitem(sys.modules, "kvcache_transfer.modeling", fake_modeling)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    with pytest.raises(ProtocolError, match="exactly eight visible CUDA cards"):
        runner.execute(config_path)
    assert (output / "idle_gate.json").is_file()
    assert not (output / "status.json").exists()  # model loading was never entered
