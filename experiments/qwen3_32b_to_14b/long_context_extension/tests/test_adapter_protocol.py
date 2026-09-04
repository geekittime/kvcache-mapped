from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[2]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from long_context_extension import adapter, freeze, run, validator  # noqa: E402
from long_context_extension.protocol import (  # noqa: E402
    CONFIG_PATH,
    EXPECTED_CAPS,
    EXPECTED_MAPPER_FIT_FINGERPRINT,
    EXPECTED_MAPPER_MANIFEST_SHA256,
    FORMAL_TOPOLOGY,
    MAPPER_MANIFEST,
    RUNS_ROOT,
    SMOKE_ENDPOINTS,
    SMOKE_SAMPLE_INDICES,
    TASKS,
    output_within_experiment,
)
from kvcache_transfer.config import load_yaml  # noqa: E402
from reverse_kv.eval_common import immutable_hashed_json  # noqa: E402
from reverse_kv.protocol import atomic_write_json, canonical_sha256, sha256_file  # noqa: E402


def _args() -> SimpleNamespace:
    return SimpleNamespace(tasks=TASKS, limit=None, sample_indices=None)


def test_frozen_reverse_config_and_1024_caps_pass_contract_validation():
    raw = load_yaml(CONFIG_PATH)
    adapter._validate_raw_extension_contract(raw, _args())
    assert {
        task: int(raw["tasks"][task]["max_gen_tokens"]) for task in TASKS
    } == EXPECTED_CAPS
    assert tuple(raw["runtime"]["source_devices"]) == FORMAL_TOPOLOGY["source"][
        "devices"
    ]
    assert tuple(raw["runtime"]["target_devices"]) == FORMAL_TOPOLOGY["target"][
        "devices"
    ]


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    (
        ("runtime", "source_devices", [0, 1, 2, 3], "source devices changed"),
        ("data", "lmsys_sample_count", 499, "lmsys_sample_count changed"),
        ("tasks", "hotpotqa", {"max_gen_tokens": 512, "prompt_mode": "qwen_chat_no_think"}, "hotpotqa cap changed"),
    ),
)
def test_contract_rejects_scientific_drift(section, field, value, message):
    raw = deepcopy(load_yaml(CONFIG_PATH))
    raw[section][field] = value
    with pytest.raises(ValueError, match=message):
        adapter._validate_raw_extension_contract(raw, _args())


def test_shared_validator_compatibility_shim_does_not_change_raw_config():
    raw = load_yaml(CONFIG_PATH)
    compatible = adapter._compatible_forward_validator_raw(raw)
    assert tuple(compatible["runtime"]["source_layer_counts"]) == (20, 20)
    assert tuple(compatible["runtime"]["target_layer_counts"]) == (
        10,
        11,
        11,
        11,
        11,
        10,
    )
    assert tuple(raw["runtime"]["source_layer_counts"]) == FORMAL_TOPOLOGY[
        "source"
    ]["layer_counts"]
    assert tuple(raw["runtime"]["target_layer_counts"]) == FORMAL_TOPOLOGY[
        "target"
    ]["layer_counts"]


def test_reverse_mapper_identity_is_direction_and_content_bound():
    identity = adapter._strict_mapper_identity(MAPPER_MANIFEST)
    assert identity["manifest_sha256"] == EXPECTED_MAPPER_MANIFEST_SHA256
    assert identity["fit_fingerprint"] == EXPECTED_MAPPER_FIT_FINGERPRINT
    assert identity["target_layer_count"] == 40


def test_output_containment_rejects_repository_sibling():
    accepted = output_within_experiment(EXPERIMENT_ROOT / "runs/test-only")
    assert accepted == (EXPERIMENT_ROOT / "runs/test-only").resolve()
    with pytest.raises(ValueError, match="below the reverse runs"):
        output_within_experiment(EXPERIMENT_ROOT.parent / "escaped")


def test_output_containment_rejects_runs_root_mapper_and_symlink(tmp_path):
    with pytest.raises(ValueError, match="root itself"):
        output_within_experiment(RUNS_ROOT)
    with pytest.raises(ValueError, match="below the reverse runs"):
        output_within_experiment(EXPERIMENT_ROOT / "artifacts/mappers")
    link = tmp_path / "linked-output"
    link.symlink_to(RUNS_ROOT / "test-only", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        output_within_experiment(link)


class _FakeTokenizer:
    @staticmethod
    def decode(token_ids, **_kwargs):
        return "|".join(str(token) for token in token_ids)


def _generation_payload(token_ids, *, cap, stop_reason):
    ids = list(token_ids)
    return {
        "prediction": _FakeTokenizer.decode(ids),
        "generation": {
            "requested_max_tokens": cap,
            "generated_tokens": len(ids),
            "stop_reason": stop_reason,
            "cap_hit": stop_reason == "max_tokens",
            "generated_token_ids": ids,
            "generated_token_ids_sha256": canonical_sha256(ids),
        },
    }


@pytest.mark.parametrize(
    ("token_ids", "cap", "stop_reason"),
    (([7, 8], 3, "eos"), ([7, 8, 9], 3, "max_tokens"), ([151_935], 3, "eos")),
)
def test_exact_generated_token_evidence_accepts_only_valid_stops(
    token_ids, cap, stop_reason
):
    payload = _generation_payload(token_ids, cap=cap, stop_reason=stop_reason)
    adapter.validate_generated_token_evidence(
        payload,
        tokenizer=_FakeTokenizer(),
        expected_cap=cap,
        token_ids_sha256=canonical_sha256,
    )
    assert validator._validate_generated_tokens_independent(
        payload, tokenizer=_FakeTokenizer(), expected_cap=cap
    ) == token_ids


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value["generation"].update(generated_tokens=1), "count"),
        (lambda value: value["generation"].update(generated_token_ids_sha256="0" * 64), "hash|SHA256"),
        (lambda value: value.update(prediction="changed"), "decode"),
        (lambda value: value["generation"].update(stop_reason="stop_sequence"), "only eos or max_tokens"),
        (lambda value: value["generation"].update(stop_reason="max_tokens", cap_hit=True), "full-cap|exactly cap"),
        (lambda value: value["generation"].update(generated_token_ids=[True, 8]), "integer"),
        (lambda value: value["generation"].update(generated_token_ids=[151_936]), "vocab"),
    ),
)
def test_generated_token_evidence_rejects_tampering(mutation, message):
    payload = _generation_payload([7, 8], cap=3, stop_reason="eos")
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        adapter.validate_generated_token_evidence(
            payload,
            tokenizer=_FakeTokenizer(),
            expected_cap=3,
            token_ids_sha256=canonical_sha256,
        )
    with pytest.raises(ValueError):
        validator._validate_generated_tokens_independent(
            payload, tokenizer=_FakeTokenizer(), expected_cap=3
        )


def test_adapter_persists_tokens_and_resume_rejects_token_tamper(
    monkeypatch, tmp_path
):
    base = adapter._load_base_runner()

    def fake_evaluate_one(**kwargs):
        generated = kwargs["engine"].generate_token_ids()
        ids = list(generated.generated_token_ids)
        return {
            "prediction": kwargs["tokenizer"].decode(ids),
            "generation": {
                "requested_max_tokens": 1024,
                "generated_tokens": len(ids),
                "stop_reason": "eos",
                "cap_hit": False,
                "generated_token_ids_sha256": base.token_ids_sha256(ids),
            },
        }

    base.evaluate_one = fake_evaluate_one
    base.checkpoint_complete = lambda *_args, **_kwargs: True
    monkeypatch.setattr(adapter, "_load_base_runner", lambda: base)
    configured = adapter.configured_base_runner()

    class Engine:
        @staticmethod
        def generate_token_ids():
            return SimpleNamespace(generated_token_ids=(7, 8))

    value = configured.evaluate_one(
        engine=Engine(),
        tokenizer=_FakeTokenizer(),
        task_config={"max_gen_tokens": 1024},
    )
    assert value["generation"]["generated_token_ids"] == [7, 8]
    checkpoint = tmp_path / "checkpoint.json"
    atomic_write_json(checkpoint, value)
    kwargs = {
        "fingerprint": "unused",
        "task": "hotpotqa",
        "mode": "target_standalone",
        "sample": SimpleNamespace(),
        "tokenizer": _FakeTokenizer(),
        "max_length": 16_384,
        "reference_score_cap": 1_024,
        "target_checkpoint": None,
    }
    assert configured.checkpoint_complete(checkpoint, **kwargs) is True
    changed = deepcopy(value)
    changed["generation"]["generated_token_ids"][0] = 9
    atomic_write_json(checkpoint, changed)
    assert configured.checkpoint_complete(checkpoint, **kwargs) is False


def test_smoke_argv_scope_and_endpoint_identity_are_exact(tmp_path):
    argv = run._sealed_argv("smoke", tmp_path)
    assert "--limit" not in argv
    explicit = argv[argv.index("--sample-indices") + 1]
    assert explicit == ",".join(
        f"{task}:{index}"
        for task in TASKS
        for index in SMOKE_SAMPLE_INDICES[task]
    )
    for task, endpoints in SMOKE_ENDPOINTS.items():
        assert tuple(row["evaluation_index"] for row in endpoints) == (
            SMOKE_SAMPLE_INDICES[task]
        )
        assert all(len(row["record_sha256"]) == 64 for row in endpoints)
        assert all(row["coverage_intent"] for row in endpoints)


def test_loader_endpoint_identity_check_fails_before_gpu():
    for task, endpoints in SMOKE_ENDPOINTS.items():
        samples = [
            SimpleNamespace(
                index=endpoint["evaluation_index"],
                sample_id=endpoint["sample_id"],
                record_sha256=endpoint["record_sha256"],
            )
            for endpoint in endpoints
        ]
        manifest = (
            {
                "selected": [
                    {
                        "evaluation_index": endpoint["evaluation_index"],
                        "source_order": endpoint["source_index"],
                    }
                    for endpoint in endpoints
                ]
            }
            if task in ("lmsys", "sharegpt4")
            else None
        )
        adapter._validate_smoke_endpoint_identities(task, samples, manifest)
        samples[0].record_sha256 = "0" * 64
        with pytest.raises(ValueError, match="identity changed"):
            adapter._validate_smoke_endpoint_identities(task, samples, manifest)


def _valid_gpu_gate():
    samples = []
    for sample_index in range(12):
        samples.append(
            {
                "sample_index": sample_index,
                "utc": f"2026-08-27T00:00:{sample_index:02d}+00:00",
                "passed": True,
                "memory": [
                    {
                        "physical_index": index,
                        "memory_used_mib": 0,
                        "memory_total_mib": 24_000,
                        "uuid": f"GPU-{index}",
                    }
                    for index in range(8)
                ],
                "compute_processes": [],
            }
        )
    return {
        "status": "passed",
        "required_consecutive_samples": 12,
        "memory_limit_mib": 64,
        "final_consecutive_samples": 12,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "samples": samples,
    }


def test_gpu_gate_resume_ignores_valid_sidecars_but_rejects_extras(tmp_path):
    directory = tmp_path / "long_context_gpu_gates"
    immutable_hashed_json(directory / "gate_0001.json", _valid_gpu_gate())
    assert run._next_gpu_gate(tmp_path).name == "gate_0002.json"
    (directory / "unexpected").write_text("attack", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        run._next_gpu_gate(tmp_path)


def test_result_inventory_rejects_dangling_and_directory_symlinks(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "dangling.json").symlink_to(tmp_path / "absent.json")
    with pytest.raises(validator.ExtensionValidationError, match="symlink"):
        validator._result_inventory_paths(tmp_path, [])
    (results / "dangling.json").unlink()
    real = tmp_path / "real"
    real.mkdir()
    (results / "linked-directory").symlink_to(real, target_is_directory=True)
    with pytest.raises(validator.ExtensionValidationError, match="symlink"):
        validator._result_inventory_paths(tmp_path, [])


def test_prepare_holds_writer_lock_across_scope_and_shared_runner(monkeypatch, tmp_path):
    events = []

    class Lock:
        def close(self):
            events.append("close")

    monkeypatch.setattr(run, "validate_code_freeze", lambda: events.append("freeze"))
    monkeypatch.setattr(run, "output_within_experiment", lambda _path: tmp_path)
    monkeypatch.setattr(run, "_reject_foreign_receipt", lambda *_args: None)
    monkeypatch.setattr(run, "_validate_committed_receipt", lambda *_args: False)
    monkeypatch.setattr(
        run, "acquire_gpu_lock", lambda **_kwargs: (events.append("lock") or Lock())
    )
    monkeypatch.setattr(
        run, "_recover_completed_under_lock", lambda *_args: (events.append("recover") or False)
    )
    monkeypatch.setattr(run, "_require_prepare_safe_status", lambda *_args: events.append("safe"))
    monkeypatch.setattr(run, "_record_scope", lambda *_args: events.append("scope"))
    monkeypatch.setattr(
        run, "run_extension_evaluation", lambda *_args: (events.append("runner") or 0)
    )
    assert run.main(["--profile", "prepare", "--output-dir", str(tmp_path)]) == 0
    assert events == ["freeze", "lock", "recover", "safe", "scope", "runner", "close"]


def test_gpu_waiter_rechecks_receipt_after_lock_before_idle(monkeypatch, tmp_path):
    events = []

    class Lock:
        def close(self):
            events.append("close")

    monkeypatch.setattr(run, "validate_code_freeze", lambda: None)
    monkeypatch.setattr(run, "output_within_experiment", lambda _path: tmp_path)
    monkeypatch.setattr(run, "_reject_foreign_receipt", lambda *_args: None)
    monkeypatch.setattr(run, "_validate_committed_receipt", lambda *_args: False)
    monkeypatch.setattr(run, "acquire_gpu_lock", lambda **_kwargs: Lock())
    monkeypatch.setattr(
        run, "_recover_completed_under_lock", lambda *_args: (events.append("receipt") or True)
    )
    monkeypatch.setattr(
        run,
        "wait_for_idle_gpus",
        lambda *_args: pytest.fail("idle gate must not run after lock-time completion"),
    )
    assert run.main(["--profile", "smoke", "--output-dir", str(tmp_path)]) == 0
    assert events == ["receipt", "close"]


def test_raw_complete_status_finalizes_without_gpu(monkeypatch, tmp_path):
    atomic_write_json(tmp_path / "status.json", {"status": "complete", "failures": 0})
    finalized = []
    monkeypatch.setattr(run, "finalize_full_output", lambda path: finalized.append(path))
    assert run._recover_completed_under_lock(tmp_path, "full") is True
    assert finalized == [tmp_path]


def test_mapper_latency_summary_is_exactly_recomputed():
    path = EXPERIMENT_ROOT / "runs/five_task_extension_smoke_k20_v1/mapper.json"
    value = json.loads(path.read_text(encoding="utf-8"))["materialization"]
    samples = [row["host_load_and_hash_ms"] for row in value["per_layer"]]
    validator._validate_latency_summary(
        value["host_load_and_hash"],
        label="test mapper load",
        expected_samples=samples,
    )
    changed = deepcopy(value["host_load_and_hash"])
    changed["mean"] += 1.0
    with pytest.raises(validator.ExtensionValidationError, match="recomputed"):
        validator._validate_latency_summary(
            changed, label="test mapper load", expected_samples=samples
        )


def _sharding_row(semantic_role):
    contract = FORMAL_TOPOLOGY[semantic_role]
    devices = list(contract["devices"])
    counts = list(contract["layer_counts"])
    layer_devices = [
        device
        for device, count in zip(devices, counts, strict=True)
        for _ in range(count)
    ]
    ranges = []
    start = 0
    for device, count in zip(devices, counts, strict=True):
        ranges.append(
            {
                "device": device,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "count": count,
            }
        )
        start += count
    auxiliary_devices = {
        "model.embed_tokens": devices[0],
        "model.norm": devices[-1],
        "lm_head": devices[-1],
    }
    return {
        "status": "passed",
        "role": {"source": "target", "target": "source"}[semantic_role],
        "mode": "explicit_exact_counts",
        "devices": devices,
        "declared_layer_counts": counts,
        "expected_layer_counts": counts,
        "actual_layer_counts": counts,
        "total_layers": len(layer_devices),
        "layer_devices": layer_devices,
        "contiguous_ranges": ranges,
        "actual_parameters_inspected_for_every_layer": True,
        "parameter_tensors_inspected": len(layer_devices) * 11,
        "auxiliary_modules": {
            name: {
                "expected_device": device,
                "actual_device": device,
                "parameter_tensors_inspected": 1,
            }
            for name, device in auxiliary_devices.items()
        },
        "auxiliary_parameter_tensors_inspected": 3,
        "auxiliary_parameters_inspected_for_every_module": True,
    }


def test_model_sharding_validator_checks_full_5_plus_3_proof(tmp_path):
    path = tmp_path / "model_sharding.json"
    value = {role: _sharding_row(role) for role in ("source", "target")}
    atomic_write_json(path, value)
    validator._validate_model_sharding(tmp_path)
    value["source"]["contiguous_ranges"][0]["count"] -= 1
    atomic_write_json(path, value)
    with pytest.raises(validator.ExtensionValidationError, match="contiguous_ranges"):
        validator._validate_model_sharding(tmp_path)


def test_code_freeze_is_exact_cycle_free_and_rejects_recommitted_tamper(
    tmp_path, monkeypatch
):
    observed = freeze.validate_code_freeze()
    assert observed == freeze.build_code_freeze()
    inventory_paths = {
        row["relative_path"]
        for group in ("extension_files", "reverse_runtime_files", "shared_runtime_files")
        for row in observed[group]
    }
    assert "long_context_extension/code_freeze.json" not in inventory_paths
    assert "reverse_kv/__init__.py" in inventory_paths
    assert "src/kvcache_transfer/__init__.py" in inventory_paths
    provenance = adapter.configured_base_runner().code_provenance()
    assert provenance["reverse_runtime_files"]["reverse_kv/__init__.py"] == {
        "sha256": sha256_file(EXPERIMENT_ROOT / "reverse_kv/__init__.py"),
        "nbytes": (EXPERIMENT_ROOT / "reverse_kv/__init__.py").stat().st_size,
    }
    assert (
        provenance["long_context_extension_code_freeze"]["inventory_digest"]
        == observed["inventory_digest"]
    )
    copied = tmp_path / "code_freeze.json"
    copied_sidecar = tmp_path / "code_freeze.json.sha256.json"
    tampered = deepcopy(observed)
    tampered["status"] = "tampered"
    atomic_write_json(copied, tampered)
    atomic_write_json(
        copied_sidecar,
        {
            "schema_version": 1,
            "artifact": copied.name,
            "sha256": sha256_file(copied),
            "nbytes": copied.stat().st_size,
        },
    )
    monkeypatch.setattr(freeze, "CODE_FREEZE", copied)
    monkeypatch.setattr(freeze, "CODE_FREEZE_SIDECAR", copied_sidecar)
    with pytest.raises(freeze.CodeFreezeValidationError, match="inventory"):
        freeze.validate_code_freeze()
