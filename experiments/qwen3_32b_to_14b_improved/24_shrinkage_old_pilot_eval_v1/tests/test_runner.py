from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

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
    new = candidate_result_path(Path("/tmp/dir24"), row)
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
    assert any(path.endswith("24_shrinkage_old_pilot_eval_v1/src/runner.py") for path in paths)
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
        env={"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": "/tmp/dir24_test_subprocess_pycache", **{"PATH": "/usr/bin:/bin"}},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_no_baseline_rerun_is_recorded_in_protocol():
    config = load_config()
    assert config["outputs"]["run_root"].startswith(str(Path(__file__).parents[1]))
    assert config["upstream"]["baseline_run_root"].startswith("/home/wby/llm-cache")


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


def test_output_root_rejects_symlink_component(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ProtocolError):
        runner._assert_output_root(link / "run")


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
    definition = {"request_fingerprint": "d" * 64}
    completion_path, sidecar_path = runner._completion_paths(output)
    assert completion_path == output / "completion_receipt.json"
    assert sidecar_path == output / "completion_receipt.json.sha256.json"
    runner.immutable_json(completion_path, {"status": "complete", "request_fingerprint": definition["request_fingerprint"]})
    runner.write_sidecar(completion_path)
    runner.atomic_json(output / "paired_summary.json", {"status": "complete", "request_fingerprint": definition["request_fingerprint"]})
    assert runner._terminal_completion_summary(output, definition)["status"] == "complete"
    sidecar_path.write_text(json.dumps({"path": completion_path.name, "nbytes": 0, "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ProtocolError, match="sidecar mismatch"):
        runner._terminal_completion_summary(output, definition)


def test_execute_binds_completion_paths_before_finalization():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    execute_pos = source.index("def execute(")
    bind_pos = source.index("completion_path, completion_sidecar_path = _completion_paths(output)", execute_pos)
    publish_pos = source.index("immutable_json(completion_path, receipt)", execute_pos)
    sidecar_pos = source.index("verify_sidecar(completion_path)", publish_pos)
    assert execute_pos < bind_pos < publish_pos < sidecar_pos


def test_execute_contains_in_lock_terminal_recheck():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    lock_pos = source.index("lock = acquire_gpu_lock(")
    recheck_pos = source.index("terminal = _terminal_completion_summary(output, definition)", lock_pos)
    assert recheck_pos > lock_pos


def test_real_shrinkage_loader_cpu_regression():
    cls = runner._load_shrinkage_builder()
    assert cls.__name__ == "ShrinkageMappedPrefixCacheBuilder"
    assert "dir24_shrinkage_runtime" in sys.modules


def test_lazy_loader_cleans_sys_modules_on_import_failure(tmp_path: Path):
    bad = tmp_path / "bad_loader.py"
    bad.write_text("raise RuntimeError('synthetic loader failure')\n", encoding="utf-8")
    sys.modules.pop("dir24_synthetic_bad_loader", None)
    with pytest.raises(RuntimeError, match="synthetic loader failure"):
        runner._load_module_from_path("dir24_synthetic_bad_loader", bad)
    assert "dir24_synthetic_bad_loader" not in sys.modules
