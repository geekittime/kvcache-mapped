from __future__ import annotations

import json
from pathlib import Path

import pytest

from reverse_kv import pipeline
from reverse_kv.protocol import canonical_sha256


def test_modelscope_command_uses_current_dataset_flag() -> None:
    command = pipeline.coqa_download_command()
    assert command[:3] == [
        "/home/wby/.venvs/modelscope/bin/ms-hub", "download", "EleutherAI/coqa"
    ]
    assert "--repo-type" in command
    assert command[command.index("--repo-type") + 1] == "dataset"
    assert "--dataset" not in command
    assert command[command.index("--revision") + 1] == pipeline.COQA_REVISION


def test_query_gpus_parses_empty_process_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = "\n".join(f"{index}, GPU-{index}, {index + 10}" for index in range(8))

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    calls = iter((Result(rows), Result("")))
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *a, **k: next(calls))
    memory, processes = pipeline._query_gpus()
    assert len(memory) == 8
    assert memory[0] == {"index": 0, "uuid": "GPU-0", "memory_used_mib": 10}
    assert processes == []


def test_idle_gate_requires_twelve_consecutive_samples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    samples = 0

    def query():
        nonlocal samples
        samples += 1
        memory = [
            {"index": index, "uuid": f"GPU-{index}", "memory_used_mib": 8}
            for index in range(8)
        ]
        return memory, []

    monkeypatch.setattr(pipeline, "_query_gpus", query)
    monkeypatch.setattr(pipeline, "PIPELINE_ROOT", tmp_path)
    monkeypatch.setattr(pipeline.time, "sleep", lambda _: None)
    artifact = pipeline.wait_for_idle(name="unit", interval_seconds=0.001)
    assert samples == 12
    value = json.loads(Path(artifact["path"]).read_text())
    assert value["status"] == "passed"
    assert len(value["observations"]) == 12


def test_forward_wait_polls_running_then_verifies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "forward"
    root.mkdir()
    fingerprint = pipeline.FORWARD_FINGERPRINT
    (root / "run_definition.json").write_text(json.dumps({"request_fingerprint": fingerprint}))
    (root / "status.json").write_text(json.dumps({"request_fingerprint": fingerprint, "status": "running"}))
    polls = 0

    def sleep(_: float) -> None:
        nonlocal polls
        polls += 1
        (root / "status.json").write_text(json.dumps({"request_fingerprint": fingerprint, "status": "complete"}))

    monkeypatch.setattr(pipeline, "FORWARD_ROOT", root)
    monkeypatch.setattr(pipeline.time, "sleep", sleep)
    monkeypatch.setattr(pipeline, "verify_forward_completion", lambda publish=True: {"status": "complete"})
    assert pipeline.wait_for_forward_completion(timeout_seconds=2, poll_seconds=0.01) == {"status": "complete"}
    assert polls == 1


def test_stage_receipt_skips_only_after_revalidation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pipeline, "PIPELINE_ROOT", tmp_path)
    launcher = pipeline.FormalPipeline(python=Path("/usr/bin/python3"))
    launcher.definition_digest = "a" * 64
    launcher.state_path = tmp_path / "pipeline_state.json"
    launcher.receipt_root = tmp_path / "receipts"
    output = tmp_path / "answer.json"
    calls = 0

    def validator():
        if not output.exists():
            return None
        return [pipeline._artifact(output, base=tmp_path)]

    def execute(
        command,
        *,
        stage,
        monitor_fit_smoke=False,
        fit_smoke_evidence_path=None,
    ):
        nonlocal calls
        calls += 1
        output.write_text('{"ok": true}\n')

    monkeypatch.setattr(pipeline, "run_command", execute)
    first = launcher._stage("s", ["cmd"], validator)
    second = launcher._stage("s", ["cmd"], validator)
    assert first == second
    assert calls == 1
    output.write_text('{"ok": false}\n')
    with pytest.raises(pipeline.PipelineError, match="drifted"):
        launcher._stage("s", ["cmd"], validator)


def test_formal_main_rejects_all_overrides() -> None:
    with pytest.raises(pipeline.PipelineError, match="no mutable CLI overrides"):
        pipeline.main(["--output", "/tmp/not-allowed"])


def test_formal_pipeline_includes_all_k4_append_stages() -> None:
    launcher = pipeline.FormalPipeline(python=Path("/usr/bin/python3"))
    commands = launcher._commands()
    validators = launcher._validators()
    assert tuple(commands) == tuple(validators)
    assert tuple(commands)[10:15] == pipeline.K4_STAGE_ORDER
    assert tuple(commands)[-1] == pipeline.REPORT_STAGE_NAME
    assert commands["12_k4_fit_target0"] != commands["04_fit_target0"]
    assert pipeline.K4_FIT_SMOKE_EVIDENCE.name == "fit_k4_target0_gpu_monitor.json"


def test_formal_pipeline_preserves_virtualenv_launcher_symlink(
    tmp_path: Path,
) -> None:
    launcher_path = tmp_path / "venv/bin/python"
    launcher_path.parent.mkdir(parents=True)
    launcher_path.symlink_to(Path("/usr/bin/python3.12"))
    launcher = pipeline.FormalPipeline(python=launcher_path)
    assert launcher.python == launcher_path.absolute()
    assert launcher.python != launcher_path.resolve()
    commands = launcher._commands()
    assert all(command[0] == str(launcher_path.absolute()) for command in commands.values())


def test_preliminary_preflight_is_upgraded_not_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps({
        "status": "passed", "direction": pipeline.DIRECTION,
        "payload_checksums_verified": False,
    }))
    monkeypatch.setattr(pipeline, "PREFLIGHT", path)
    assert pipeline._validate_preflight() is None


def test_code_freeze_requires_every_exact_group_member(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    experiment = tmp_path / "experiment"
    repository = tmp_path / "repository"
    experiment.mkdir()
    repository.mkdir()
    groups = {
        "local_files": (experiment / "local.py",),
        "read_only_shared_dependencies": (
            repository / "shared.py",
            repository / "shared_second.py",
        ),
        "benchmark_protocol_dependencies": (repository / "benchmark.json",),
        "vendored_harness_dependencies": (repository / "harness.py",),
    }
    for paths in groups.values():
        for path in paths:
            path.write_text(f"{path.name}\n")

    freeze = experiment / "code_freeze.json"
    monkeypatch.setattr(pipeline, "EXPERIMENT_ROOT", experiment)
    monkeypatch.setattr(pipeline, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(pipeline, "CODE_FREEZE", freeze)
    monkeypatch.setattr(pipeline, "expected_freeze_groups", lambda: groups)

    value = {
        "schema_version": 1,
        "status": "frozen",
        "direction": pipeline.DIRECTION,
    }
    for key, paths in groups.items():
        value[key] = [pipeline._artifact(path, base=experiment if key == "local_files" else repository) for path in paths]
    value["inventory_digest"] = canonical_sha256(value)
    pipeline.atomic_write_json(freeze, value)
    pipeline.atomic_write_json(
        freeze.with_suffix(".sha256.json"), {"sha256": pipeline.sha256_file(freeze)}
    )
    assert pipeline.verify_code_freeze()["files"] == 5

    value["read_only_shared_dependencies"] = value[
        "read_only_shared_dependencies"
    ][1:]
    unsigned = dict(value)
    unsigned.pop("inventory_digest")
    value["inventory_digest"] = canonical_sha256(unsigned)
    pipeline.atomic_write_json(freeze, value)
    pipeline.atomic_write_json(
        freeze.with_suffix(".sha256.json"), {"sha256": pipeline.sha256_file(freeze)}
    )
    with pytest.raises(
        pipeline.PipelineError,
        match="read_only_shared_dependencies coverage changed",
    ):
        pipeline.verify_code_freeze()


def test_required_gpu_gates_are_all_directly_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pipeline_root = tmp_path / "pipeline"
    hellaswag_root = pipeline_root / "hellaswag"
    mechanism_root = pipeline_root / "mechanism"
    latency_root = pipeline_root / "latency"
    k4_training_gate = pipeline_root / "gates/k4_training_segment_idle.json"
    k4_coqa_gate = pipeline_root / "gates/coqa_k4_segment_idle.json"
    monkeypatch.setattr(pipeline, "PIPELINE_ROOT", pipeline_root)
    monkeypatch.setattr(pipeline, "HELLASWAG_ROOT", hellaswag_root)
    monkeypatch.setattr(pipeline, "MECHANISM_ROOT", mechanism_root)
    monkeypatch.setattr(pipeline, "LATENCY_ROOT", latency_root)
    monkeypatch.setattr(pipeline, "K4_TRAINING_IDLE_GATE", k4_training_gate)
    monkeypatch.setattr(pipeline, "K4_COQA_IDLE_GATE", k4_coqa_gate)
    idle_memory = [
        {"index": index, "uuid": f"GPU-{index}", "memory_used_mib": 8}
        for index in range(8)
    ]
    pipeline_gate = lambda name: {
        "schema_version": 1,
        "status": "passed",
        "name": name,
        "required_consecutive_samples": 12,
        "memory_limit_mib": 64,
        "sample_interval_seconds": 5.0,
        "timeout_seconds": 100,
        "observations": [
            {
                "utc": f"sample-{index}",
                "passed": True,
                "memory": idle_memory,
                "processes": [],
            }
            for index in range(12)
        ],
    }
    paths = (
        pipeline_root / "gates/training_segment_idle.json",
        pipeline_root / "gates/hellaswag_segment_idle.json",
        hellaswag_root / "gpu_idle_gate.json",
        mechanism_root / "gpu_idle_gate.json",
        pipeline_root / "gates/coqa_segment_idle.json",
        latency_root / "gpu_idle_gate.json",
        k4_training_gate,
        k4_coqa_gate,
    )
    for path in paths:
        if path.parent == pipeline_root / "gates" and path not in {
            k4_training_gate,
            k4_coqa_gate,
        }:
            pipeline.hashed_json(path, pipeline_gate(path.stem))
        else:
            pipeline.hashed_json(path, {"status": "passed", "name": path.stem})
    artifacts = pipeline.required_gpu_gate_artifacts()
    assert len(artifacts) == 16
    assert {Path(row["path"]) for row in artifacts} == {
        candidate
        for path in paths
        for candidate in (path.resolve(), path.with_name(f"{path.name}.sha256.json").resolve())
    }
    paths[3].write_text('{"status":"passed","name":"tampered"}\n')
    with pytest.raises(ValueError, match="SHA sidecar mismatch"):
        pipeline.required_gpu_gate_artifacts()


def test_pipeline_idle_gate_rejects_coordinated_minimal_passed_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "training_segment_idle.json"
    pipeline.hashed_json(
        path,
        {"schema_version": 1, "status": "passed", "name": "training_segment_idle"},
    )
    with pytest.raises(pipeline.PipelineError, match="field|required"):
        pipeline._validate_pipeline_idle_gate(
            path, expected_name="training_segment_idle"
        )


def test_fit_monitor_recomputes_owner_and_peak(tmp_path: Path) -> None:
    path = tmp_path / "fit.json"
    memory = [
        {"index": index, "uuid": f"GPU-{index}", "memory_used_mib": 8}
        for index in range(8)
    ]
    memory[0]["memory_used_mib"] = 1234
    owner = {"uuid": "GPU-0", "pid": 4321, "used_memory_mib": 1200, "index": 0}
    value = {
        "schema_version": 1,
        "status": "passed",
        "stage": "04_fit_target0",
        "single_compute_owner": True,
        "compute_owner_observed": True,
        "compute_owner_identities": [["GPU-0", 4321]],
        "allowed_gpu": 0,
        "peak_memory_used_mib": 1234,
        "maximum_allowed_peak_mib": pipeline.SMOKE_LIMIT_MIB,
        "samples": [{"utc": "now", "memory": memory, "processes": [owner]}],
    }
    pipeline.hashed_json(path, value)
    pipeline._validate_fit_monitor(path, expected_stage="04_fit_target0")

    value["peak_memory_used_mib"] = 1
    pipeline.hashed_json(path, value)
    with pytest.raises(pipeline.PipelineError, match="aggregate evidence"):
        pipeline._validate_fit_monitor(path, expected_stage="04_fit_target0")
