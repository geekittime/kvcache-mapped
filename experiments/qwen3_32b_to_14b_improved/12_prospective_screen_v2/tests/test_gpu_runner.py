from contextlib import contextmanager
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from src.contracts import (
    ContractError,
    artifact_ref,
    atomic_json,
    canonical_sha256,
    immutable_json,
    read_json,
    token_ids_sha256,
)
from src.checkpoints import expected_r32_builder_audit
from src.gpu_guard import wait_for_idle_gpus
from src.gpu_runner import execute_raw_grid
import src.gpu_runner as gpu_runner
from src.gpu_runtime import RuntimeComponents
import src.gpu_runtime as gpu_runtime
from src.run_state import run_state_lock, write_live_status, write_terminal_status
import src.runtime_binding as runtime_binding
from src.runtime_binding import validate_runtime_provenance
from src.runtime_binding import verify_python_bytecode_policy


class _Tokenizer:
    _prospective_frozen_eos_token_ids = (2,)

    def decode(self, ids, skip_special_tokens=True):
        return "answer" if list(ids) == [17] else ""


class _Engine:
    def __init__(self):
        self.calls = 0
        self.clears = 0

    def generate_token_ids(self, prompt, kwargs, *, truncation):
        self.calls += 1
        return SimpleNamespace(
            text="answer",
            generated_token_ids=(17,),
            stop_reason="eos",
            terminal_token_id=2,
            cap_hit=False,
            requested_max_tokens=256,
            original_context_tokens=len(prompt),
            used_context_tokens=len(prompt),
            truncation=truncation,
        )

    def clear_prefix_cache(self):
        self.clears += 1


class _Cuda:
    def device_count(self):
        return 8

    def synchronize(self, _index):
        return None

    def reset_peak_memory_stats(self, _index):
        return None

    def max_memory_allocated(self, index):
        return 1000 + index


def _row():
    body = {
        "dataset_id": "hotpotqa_train",
        "sample_id": "sample-1",
        "source_projection_sha256": "a" * 64,
        "prompt_token_ids": list(range(40)),
        "prompt_token_count": 40,
        "prompt_token_ids_sha256": token_ids_sha256(list(range(40))),
        "references": ["answer"],
        "truncation": {"truncated": False},
        "generation_cap": 256,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
    }
    body["sample_binding_sha256"] = canonical_sha256(body)
    return body


def _components():
    engines = {
        mode: _Engine()
        for mode in ("native14", "nvidia_k20", "shrinkage", "tail_refresh_r32")
    }
    tail = SimpleNamespace(
        last_call_audit=expected_r32_builder_audit(40)
    )
    return RuntimeComponents(
        tokenizer=_Tokenizer(),
        source_model=None,
        target_model=None,
        engines=engines,
        builders={
            "nvidia_k20": SimpleNamespace(),
            "shrinkage": SimpleNamespace(),
            "tail_refresh_r32": tail,
        },
        method_provenance={},
        model_sharding={},
    )


def test_real_grid_core_writes_four_modes_repairs_sidecar_and_resumes(tmp_path):
    row = _row()
    definition = {
        "profile": "smoke",
        "request_fingerprint": "f" * 64,
        "method_binding_sha256": {
            mode: canonical_sha256(mode)
            for mode in ("native14", "nvidia_k20", "shrinkage", "tail_refresh_r32")
        },
    }
    components = _components()
    torch = SimpleNamespace(cuda=_Cuda())
    counts = execute_raw_grid(
        output=tmp_path,
        profile="smoke",
        rows=[row],
        definition=definition,
        components=components,
        torch_module=torch,
        attempt_id="attempt_0001",
    )
    assert counts == {
        "completed_raw_results": 4,
        "generated_now": 4,
        "resumed_existing": 0,
    }
    result_files = sorted((tmp_path / "results" / "raw").rglob("*.json"))
    result_files = [path for path in result_files if not path.name.endswith(".sha256.json")]
    assert len(result_files) == 4
    assert all(engine.calls == 1 and engine.clears == 1 for engine in components.engines.values())

    # Simulate death after the atomic result rename but before sidecar rename.
    missing_sidecar = result_files[0].with_name(result_files[0].name + ".sha256.json")
    missing_sidecar.unlink()
    # Simulate the adjacent kill window: result+sidecar are complete, but the
    # previous failure marker has not yet been removed.
    stale_failure = result_files[1].with_suffix(".failure.json")
    atomic_json(stale_failure, {"status": "stale_after_success"})
    resumed = execute_raw_grid(
        output=tmp_path,
        profile="smoke",
        rows=[row],
        definition=definition,
        components=components,
        torch_module=torch,
        attempt_id="attempt_0002",
    )
    assert resumed == {
        "completed_raw_results": 4,
        "generated_now": 0,
        "resumed_existing": 4,
    }
    assert missing_sidecar.is_file()
    assert not stale_failure.exists()
    assert {
        read_json(path)["execution_attempt_id"] for path in result_files
    } == {"attempt_0001"}
    assert all(engine.calls == 1 for engine in components.engines.values())


def test_idle_gate_resets_consecutive_counter_and_uses_no_cuda():
    busy = {
        "memory_used_mib": {str(index): 65 for index in range(8)},
        "compute_process_count": 1,
    }
    idle = {
        "memory_used_mib": {str(index): 0 for index in range(8)},
        "compute_process_count": 0,
    }
    values = iter([idle, busy, idle, idle])
    clock = iter(float(index) for index in range(20))
    result = wait_for_idle_gpus(
        {
            "memory_used_limit_mib": 64,
            "consecutive_samples": 2,
            "sample_interval_seconds": 1,
            "timeout_seconds": 100,
        },
        probe=lambda: next(values),
        sleeper=lambda _seconds: None,
        monotonic=lambda: next(clock),
    )
    assert result["gpu_used_by_gate"] is False
    assert [item["consecutive_idle"] for item in result["samples"]] == [1, 0, 1, 2]


def _file_ref(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return artifact_ref(path)


def _model_identity(root: Path):
    metadata = _file_ref(root / "config.json", b"{}\n")
    shard = _file_ref(root / "model-00001-of-00001.safetensors", b"weights\n")
    shard_row = {
        "path": Path(shard.path).name,
        "nbytes": shard.nbytes,
        "sha256": shard.sha256,
    }
    return {
        "path": str(root),
        "content_hashed_files": {
            "config.json": {"nbytes": metadata.nbytes, "sha256": metadata.sha256}
        },
        "weight_shards": [shard_row],
        "weight_shard_count": 1,
        "weight_shards_sha256": canonical_sha256([shard_row]),
        "weight_shard_total_nbytes": shard.nbytes,
    }


def test_full_weight_verification_strength_does_not_change_identity_digest(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repository"
    baseline_root = repository / "experiments" / "qwen3_32b_to_14b"
    monkeypatch.setattr(runtime_binding, "REPOSITORY_ROOT", repository)
    monkeypatch.setattr(runtime_binding, "BASELINE_EXPERIMENT_ROOT", baseline_root)
    core_files = {}
    for relative in sorted(runtime_binding.REQUIRED_CORE_RUNTIME_FILES):
        reference = _file_ref(repository / relative, f"{relative}\n".encode())
        core_files[relative] = {
            "nbytes": reference.nbytes,
            "sha256": reference.sha256,
        }
    reverse_files = {}
    for relative in sorted(runtime_binding.REQUIRED_REVERSE_RUNTIME_FILES):
        reference = _file_ref(baseline_root / relative, f"{relative}\n".encode())
        reverse_files[relative] = {
            "nbytes": reference.nbytes,
            "sha256": reference.sha256,
        }
    extension_freeze = _file_ref(tmp_path / "extension_freeze.json", b"{}\n")
    reverse_freeze = _file_ref(tmp_path / "reverse_freeze.json", b"{}\n")
    source = _model_identity(tmp_path / "Qwen3-32B")
    target = _model_identity(tmp_path / "Qwen3-14B")
    mapper = _file_ref(tmp_path / "mapper.json", b"{}\n")
    receipt_body = {
        "schema_version": 1,
        "direction": "qwen3_32b_to_14b",
        "max_length": 16384,
        "mapper_token_chunk": 1024,
        "attention_implementation": "flash_attention_2",
        "formal_topology": {
            "source": {
                "model": source["path"],
                "devices": [0, 1, 2, 3, 4],
                "layer_counts": [12, 13, 14, 13, 12],
            },
            "target": {
                "model": target["path"],
                "devices": [5, 6, 7],
                "layer_counts": [13, 14, 13],
            },
        },
        "decoding": {"batch_size": 1, "do_sample": False, "temperature": 0.0},
        "code": {
            "files": core_files,
            "files_sha256": canonical_sha256(core_files),
            "reverse_runtime_files": reverse_files,
            "long_context_extension_code_freeze": extension_freeze.to_dict(),
            "reverse_base_code_freeze": reverse_freeze.to_dict(),
        },
        "runtime": {"python": "fixture", "distributions": {}},
        "mapper": {
            "manifest": mapper.path,
            "manifest_sha256": mapper.sha256,
            "target_layer_count": 40,
        },
        "source_model": source,
        "target_model": target,
    }
    receipt = {
        **receipt_body,
        "request_fingerprint": canonical_sha256(receipt_body),
    }
    receipt_ref = atomic_json(tmp_path / "runtime_receipt.json", receipt)
    package_files = {
        relative: core_files[relative]
        for relative in sorted(runtime_binding.REQUIRED_PACKAGE_RUNTIME_FILES)
    }
    monkeypatch.setattr(runtime_binding, "RUNTIME_PACKAGE_CLOSURE_FILES", package_files)
    config = {
        "runtime": {
            "source_model": source["path"],
            "target_model": target["path"],
            "source_devices": [0, 1, 2, 3, 4],
            "source_layer_counts": [12, 13, 14, 13, 12],
            "target_devices": [5, 6, 7],
            "target_layer_counts": [13, 14, 13],
        },
        "methods": {"nvidia_k20": {"mapper_artifact": mapper.to_dict()}},
        "runtime_provenance_receipt": receipt_ref.to_dict(),
        "runtime_package_closure": {
            "root": str(repository),
            "files": package_files,
            "files_sha256": canonical_sha256(package_files),
        },
    }
    shallow = validate_runtime_provenance(config, deep_model_hash=False)
    deep = validate_runtime_provenance(config, deep_model_hash=True)
    assert shallow["deep_model_hash_verified"] is False
    assert deep["deep_model_hash_verified"] is True
    assert shallow["binding_digest"] == deep["binding_digest"]
    assert shallow["runtime_code"]["package_closure"] == config["runtime_package_closure"]

    missing_closure = dict(config)
    missing_closure.pop("runtime_package_closure")
    with pytest.raises(ContractError, match="runtime package closure"):
        validate_runtime_provenance(missing_closure, deep_model_hash=False)

    # The package files are not covered by the immutable historical receipt;
    # their independent closure binding must reject either file's drift.
    package_changed = repository / "src/kvcache_transfer/io.py"
    package_original = package_changed.read_bytes()
    package_changed.write_text("tampered package helper\n", encoding="utf-8")
    with pytest.raises(ContractError, match="SHA-256 drifted|byte size drifted"):
        validate_runtime_provenance(config, deep_model_hash=False)
    package_changed.write_bytes(package_original)

    changed = repository / sorted(runtime_binding.REQUIRED_CORE_RUNTIME_FILES)[0]
    core_original = changed.read_bytes()
    changed.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ContractError, match="SHA-256 drifted|byte size drifted"):
        validate_runtime_provenance(config, deep_model_hash=False)
    changed.write_bytes(core_original)


def test_python_bytecode_policy_is_exact_and_rejects_aliases_or_existing_prefix(
    tmp_path, monkeypatch
):
    prefix = tmp_path / "absent-pycache"
    runtime = {
        "required_python_dont_write_bytecode": True,
        "required_python_pycache_prefix": str(prefix),
    }
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(prefix))
    monkeypatch.setattr(sys, "pycache_prefix", str(prefix))
    assert verify_python_bytecode_policy(runtime) == {
        "python_dont_write_bytecode": True,
        "python_pycache_prefix": str(prefix),
        "python_pycache_prefix_absent": True,
    }

    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "0")
    with pytest.raises(ContractError, match="PYTHONDONTWRITEBYTECODE"):
        verify_python_bytecode_policy(runtime)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")

    prefix.mkdir()
    with pytest.raises(ContractError, match="must be absent"):
        verify_python_bytecode_policy(runtime)

    prefix.rmdir()
    with pytest.raises(ContractError, match="exact true"):
        verify_python_bytecode_policy(
            {
                "required_python_dont_write_bytecode": 1,
                "required_python_pycache_prefix": str(prefix),
            }
        )
    with pytest.raises(ContractError, match="must be absolute"):
        verify_python_bytecode_policy(
            {
                "required_python_dont_write_bytecode": True,
                "required_python_pycache_prefix": "relative-prefix",
            }
        )

    monkeypatch.setattr(sys, "pycache_prefix", str(prefix) + ".wrong")
    with pytest.raises(ContractError, match="differs from frozen policy"):
        verify_python_bytecode_policy(runtime)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    chained_prefix = symlink_parent / "leaf"
    chained_runtime = {
        "required_python_dont_write_bytecode": True,
        "required_python_pycache_prefix": str(chained_prefix),
    }
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(chained_prefix))
    monkeypatch.setattr(sys, "pycache_prefix", str(chained_prefix))
    with pytest.raises(ContractError, match="symlink component"):
        verify_python_bytecode_policy(chained_runtime)


def test_verified_interface_executes_hashed_source_not_timestamp_pyc(tmp_path):
    import os
    import py_compile

    source_path = tmp_path / "interface.py"
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    good = b"VALUE = 'source-bytes'\n"
    bad = b"VALUE = 'pyc-import!!'\n"
    assert len(good) == len(bad)
    source_path.write_bytes(bad)
    timestamp = 1_700_000_123
    os.utime(source_path, (timestamp, timestamp))
    pyc_path = pycache / f"interface.cpython-{sys.version_info[0]}{sys.version_info[1]}.pyc"
    py_compile.compile(str(source_path), cfile=str(pyc_path), doraise=True)
    source_path.write_bytes(good)
    os.utime(source_path, (timestamp, timestamp))

    reference = artifact_ref(source_path)
    module_name = "_prospective_timestamp_pyc_probe"
    sys.modules.pop(module_name, None)
    module = gpu_runtime._load_verified_module(reference, module_name)
    try:
        assert module.VALUE == "source-bytes"
        assert module.__file__ == reference.path
    finally:
        sys.modules.pop(module_name, None)


def test_verified_interface_execution_failure_does_not_leave_module(tmp_path):
    source_path = tmp_path / "broken_interface.py"
    source_path.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    reference = artifact_ref(source_path)
    module_name = "_prospective_broken_interface_probe"
    sys.modules.pop(module_name, None)
    with pytest.raises(ContractError, match="execution failed"):
        gpu_runtime._load_verified_module(reference, module_name)
    assert module_name not in sys.modules


def test_grid_rejects_runtime_metadata_drift_before_writing(tmp_path):
    row = _row()
    definition = {
        "profile": "smoke",
        "request_fingerprint": "f" * 64,
        "method_binding_sha256": {
            mode: canonical_sha256(mode)
            for mode in ("native14", "nvidia_k20", "shrinkage", "tail_refresh_r32")
        },
    }
    components = _components()
    components.engines["native14"].generate_token_ids = lambda *_args, **_kwargs: SimpleNamespace(
        text="answer",
        generated_token_ids=(17,),
        stop_reason="eos",
        terminal_token_id=2,
        cap_hit=False,
        requested_max_tokens=256,
        original_context_tokens=39,
        used_context_tokens=39,
        truncation="error",
    )
    with pytest.raises(ContractError, match="original_context_tokens changed"):
        execute_raw_grid(
            output=tmp_path,
            profile="smoke",
            rows=[row],
            definition=definition,
            components=components,
            torch_module=SimpleNamespace(cuda=_Cuda()),
            attempt_id="attempt_0001",
        )


def test_grid_rejects_runtime_result_type_coercion(tmp_path):
    row = _row()
    definition = {
        "profile": "smoke",
        "request_fingerprint": "f" * 64,
        "method_binding_sha256": {
            mode: canonical_sha256(mode)
            for mode in ("native14", "nvidia_k20", "shrinkage", "tail_refresh_r32")
        },
    }
    components = _components()
    components.engines["native14"].generate_token_ids = lambda *_args, **_kwargs: SimpleNamespace(
        text="answer",
        generated_token_ids=(17,),
        stop_reason="eos",
        terminal_token_id=2,
        cap_hit=0,
        requested_max_tokens=256,
        original_context_tokens=40,
        used_context_tokens=40,
        truncation="error",
    )
    with pytest.raises(ContractError, match="result types changed"):
        execute_raw_grid(
            output=tmp_path,
            profile="smoke",
            rows=[row],
            definition=definition,
            components=components,
            torch_module=SimpleNamespace(cuda=_Cuda()),
            attempt_id="attempt_0001",
        )
    assert read_json(tmp_path / "status.json")["status"] == "gpu_failed"


def test_content_hashed_target_eos_set_and_local_engine_capture_actual_terminal(
    tmp_path,
):
    (tmp_path / "config.json").write_text(
        '{"eos_token_id":2}\n', encoding="utf-8"
    )
    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id":[2,4]}\n', encoding="utf-8"
    )
    assert gpu_runtime.frozen_target_eos_token_ids(tmp_path) == (2, 4)

    torch = pytest.importorskip("torch")

    class _BaseEngine:
        def __init__(self, *, target_model, tokenizer, mode, max_length, **_kwargs):
            self.target_model = target_model
            self.tokenizer = tokenizer
            self.mode = mode
            self.max_length = max_length
            self.device = "cpu"
            self.prefix_token_id = 0

        def _generation_kwargs(self, kwargs):
            return int(kwargs["max_gen_toks"]), list(kwargs.get("until", []))

    class _OneStepModel:
        def __init__(self, next_token):
            self.next_token = next_token

        def __call__(self, *, input_ids, **_kwargs):
            logits = torch.zeros((1, input_ids.shape[1], 20), dtype=torch.float32)
            logits[:, -1, self.next_token] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=None)

    eval_module = SimpleNamespace(
        torch=torch,
        STANDALONE_MODES=("target_standalone",),
        _tokenizer_decode=lambda tokenizer, ids: tokenizer.decode(ids),
    )
    audited_class = gpu_runtime._audited_engine_class(_BaseEngine, eval_module)
    tokenizer = _Tokenizer()
    tokenizer._prospective_frozen_eos_token_ids = (2, 4)
    engine = audited_class(
        target_model=_OneStepModel(4),
        tokenizer=tokenizer,
        mode="target_standalone",
        max_length=32,
    )
    result = engine.generate_token_ids(
        [1], {"max_gen_toks": 4, "until": []}, truncation="error"
    )
    assert result.stop_reason == "eos"
    assert result.terminal_token_id == 4
    assert result.generated_token_ids == ()

    engine.target_model = _OneStepModel(3)
    max_result = engine.generate_token_ids(
        [1], {"max_gen_toks": 1, "until": []}, truncation="error"
    )
    assert max_result.stop_reason == "max_tokens"
    assert max_result.terminal_token_id is None
    assert max_result.generated_token_ids == (3,)

    engine.target_model = _OneStepModel(17)
    stop_result = engine.generate_token_ids(
        [1], {"max_gen_toks": 4, "until": ["answer"]}, truncation="error"
    )
    assert stop_result.stop_reason == "stop_sequence"
    assert stop_result.terminal_token_id is None
    assert stop_result.generated_token_ids == (17,)


def test_waiting_second_gpu_rechecks_raw_terminal_before_any_model_load(
    tmp_path, monkeypatch
):
    output = tmp_path / "runs" / "smoke_v2"
    output.mkdir(parents=True)
    config = {
        "runtime": {
            "required_cuda_visible_devices": "test-visible",
            "python_executable": sys.executable,
            "required_python_dont_write_bytecode": True,
            "required_python_pycache_prefix": "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/12_prospective_screen_v2/.formal_empty_pycache",
            "global_gpu_lock": str(tmp_path / "global.lock"),
        }
    }
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "test-visible")
    monkeypatch.setattr(gpu_runner, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        gpu_runner, "canonical_run_output", lambda *_args, **_kwargs: output
    )
    definition = {"profile": "smoke", "request_fingerprint": "f" * 64}
    monkeypatch.setattr(
        gpu_runner,
        "reproduce_run_definition",
        lambda *_args, **_kwargs: (definition, []),
    )
    monkeypatch.setattr(
        gpu_runner, "load_frozen_decode_tokenizer", lambda _config: object()
    )
    validated = []

    def recover(_config, _definition, _rows, recovered_output, **kwargs):
        assert kwargs["repair_missing_sidecar"] is True
        validated.append(True)
        receipt = artifact_ref(recovered_output / "raw_completion_receipt.json")
        from src.contracts import write_sidecar

        write_sidecar(receipt)
        return {}

    monkeypatch.setattr(gpu_runner, "validate_raw_completion_receipt", recover)
    model_loads = []
    monkeypatch.setattr(
        gpu_runner,
        "validate_method_bindings",
        lambda *_args, **_kwargs: model_loads.append("deep_validation"),
    )

    shared = threading.Lock()
    first_entered = threading.Event()
    second_waiting = threading.Event()

    @contextmanager
    def fake_global_lock(_path):
        if threading.current_thread().name == "gpu-b":
            second_waiting.set()
        with shared:
            yield None

    @contextmanager
    def fake_run_lock(_output):
        yield None

    monkeypatch.setattr(gpu_runner, "global_gpu_lock", fake_global_lock)
    monkeypatch.setattr(gpu_runner, "run_state_lock", fake_run_lock)
    errors = []

    def first_gpu_finishes():
        with fake_global_lock(config["runtime"]["global_gpu_lock"]), fake_run_lock(output):
            first_entered.set()
            assert second_waiting.wait(timeout=5)
            immutable_json(output / "raw_completion_receipt.json", {"status": "fixture"})

    def second_gpu_starts_while_first_holds_lock():
        try:
            gpu_runner.execute_gpu("fixture.json", profile="smoke", output_dir=output)
        except ContractError as exc:
            errors.append(str(exc))

    first = threading.Thread(target=first_gpu_finishes, name="gpu-a")
    second = threading.Thread(target=second_gpu_starts_while_first_holds_lock, name="gpu-b")
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()
    assert errors == ["raw grid is already complete; run seal/validate instead"]
    assert validated == [True]
    assert model_loads == []
    assert not (output / "gpu_attempts").exists()
    assert not (output / "status.json").exists()


def test_cpu_terminal_seal_cannot_be_downgraded_by_waiting_gpu_writer(tmp_path):
    output = tmp_path / "runs" / "smoke_v2"
    status_path = output / "status.json"
    seal_entered = threading.Event()
    gpu_waiting = threading.Event()
    errors = []
    terminal = {
        "schema_version": 2,
        "status": "complete",
        "profile": "smoke",
        "request_fingerprint": "f" * 64,
    }

    def cpu_seal():
        with run_state_lock(output):
            seal_entered.set()
            assert gpu_waiting.wait(timeout=5)
            write_terminal_status(status_path, terminal)

    def waiting_gpu_writer():
        assert seal_entered.wait(timeout=5)
        gpu_waiting.set()
        with run_state_lock(output):
            try:
                write_live_status(
                    status_path,
                    {"schema_version": 2, "status": "gpu_failed"},
                )
            except ContractError as exc:
                errors.append(str(exc))

    seal = threading.Thread(target=cpu_seal)
    gpu = threading.Thread(target=waiting_gpu_writer)
    seal.start()
    gpu.start()
    seal.join(timeout=5)
    gpu.join(timeout=5)
    assert not seal.is_alive() and not gpu.is_alive()
    assert errors == ["terminal run status cannot be downgraded or overwritten"]
    assert read_json(status_path) == terminal


def test_terminal_status_sidecar_kill_window_recovers_only_identical_state(tmp_path):
    status_path = tmp_path / "runs" / "smoke_v2" / "status.json"
    terminal = {
        "schema_version": 2,
        "status": "complete",
        "profile": "smoke",
        "request_fingerprint": "f" * 64,
    }
    atomic_json(status_path, terminal)
    assert not status_path.with_name(status_path.name + ".sha256.json").exists()
    write_terminal_status(status_path, terminal)
    assert status_path.with_name(status_path.name + ".sha256.json").is_file()
