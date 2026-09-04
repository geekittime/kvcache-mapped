from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcache_transfer.io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kvcache_transfer_evaluate_script", ROOT / "scripts/evaluate.py"
)
assert SPEC is not None and SPEC.loader is not None
EVALUATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATE)


BENCHMARKS = {
    "arc_challenge": {"metric": "acc_norm"},
    "hellaswag": {"metric": "acc_norm"},
    "winogrande": {"metric": "acc"},
    "mmlu": {"metric": "acc"},
    "gsm8k_cot": {"metric": "exact_match"},
    "wikitext2_prefix_ppl": {"metric": "perplexity"},
}


def _fixture_doc_to_text(doc):
    return str(doc)


def test_harness_metadata_normalization_persists_callable_identity_stably():
    raw = {
        "configs": {
            "hellaswag": {
                "doc_to_text": _fixture_doc_to_text,
                "nested": (Path("data/local.parquet"), {"b", "a"}),
            }
        },
        "results": {"hellaswag": {"acc_norm,none": 0.8}},
        "samples": [{"acc_norm": 1.0}],
    }
    first = EVALUATE._json_safe_harness_value(raw)
    second = EVALUATE._json_safe_harness_value(raw)
    assert first == second
    assert first["configs"]["hellaswag"]["doc_to_text"] == {
        "__python_callable__": {
            "module": __name__,
            "qualname": "_fixture_doc_to_text",
        }
    }
    assert first["configs"]["hellaswag"]["nested"] == [
        "data/local.parquet",
        ["a", "b"],
    ]
    assert EVALUATE._canonical_digest(first) == EVALUATE._canonical_digest(second)
    assert "0x" not in json.dumps(first, sort_keys=True)


def test_harness_metadata_normalization_rejects_unknown_objects_and_nonfinite():
    with pytest.raises(TypeError, match="unsupported lm-eval metadata object"):
        EVALUATE._json_safe_harness_value(object())
    with pytest.raises(ValueError, match="non-finite"):
        EVALUATE._json_safe_harness_value({"value": math.nan})


def _checkpoint_payload(
    *, fingerprint: str, mode: str, task: str, metric: str, value: float
) -> dict[str, object]:
    return {
        "schema_version": EVALUATE.SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "mode": mode,
        "task": task,
        "primary_metric": metric,
        "primary_metric_value": value,
    }


def test_checkpoint_complete_strictly_validates_identity_and_metric(tmp_path):
    path = tmp_path / "result.json"
    payload = _checkpoint_payload(
        fingerprint="fingerprint",
        mode="mapped",
        task="hellaswag",
        metric="acc_norm",
        value=0.8,
    )
    atomic_write_json(path, payload)
    kwargs = {
        "expected_mode": "mapped",
        "expected_task": "hellaswag",
        "expected_primary_metric": "acc_norm",
    }
    assert EVALUATE._checkpoint_complete(path, "fingerprint", **kwargs)
    assert not EVALUATE._checkpoint_complete(
        path, "other-fingerprint", **kwargs
    )
    assert not EVALUATE._checkpoint_complete(
        path, "fingerprint", **{**kwargs, "expected_mode": "target_standalone"}
    )
    assert not EVALUATE._checkpoint_complete(
        path, "fingerprint", **{**kwargs, "expected_task": "arc_challenge"}
    )
    assert not EVALUATE._checkpoint_complete(
        path, "fingerprint", **{**kwargs, "expected_primary_metric": "acc"}
    )

    payload["primary_metric_value"] = math.nan
    atomic_write_json(path, payload)
    assert not EVALUATE._checkpoint_complete(path, "fingerprint", **kwargs)


def test_evaluation_order_is_task_outer_and_resume_skips_only_valid_pairs(tmp_path):
    tasks = ("arc_challenge", "hellaswag")
    modes = ("target_standalone", "mapped")
    order = EVALUATE._evaluation_order(tasks, modes)
    assert order == (
        ("arc_challenge", "target_standalone"),
        ("arc_challenge", "mapped"),
        ("hellaswag", "target_standalone"),
        ("hellaswag", "mapped"),
    )

    atomic_write_json(
        EVALUATE._result_path(tmp_path, "target_standalone", "arc_challenge"),
        _checkpoint_payload(
            fingerprint="formal",
            mode="target_standalone",
            task="arc_challenge",
            metric="acc_norm",
            value=0.61,
        ),
    )
    pending = []
    for task, mode in order:
        if not EVALUATE._checkpoint_complete(
            EVALUATE._result_path(tmp_path, mode, task),
            "formal",
            expected_mode=mode,
            expected_task=task,
            expected_primary_metric="acc_norm",
        ):
            pending.append((task, mode))
    assert pending == list(order[1:])


def test_request_cache_paths_isolate_fingerprint_mode_and_task(tmp_path):
    base = EVALUATE._request_cache_base(
        tmp_path, "a" * 64, mode="target_standalone", task="hellaswag"
    )
    assert base != EVALUATE._request_cache_base(
        tmp_path, "a" * 64, mode="mapped", task="hellaswag"
    )
    assert base != EVALUATE._request_cache_base(
        tmp_path, "a" * 64, mode="target_standalone", task="arc_challenge"
    )
    assert base != EVALUATE._request_cache_base(
        tmp_path, "b" * 64, mode="target_standalone", task="hellaswag"
    )
    record = EVALUATE._request_cache_record(base)
    assert record["enabled"] is True
    assert record["isolation"] == "request_fingerprint/mode/task"
    assert record["rank0_database"].endswith("responses_rank0.db")


def test_lm_eval_sqlite_cache_survives_partial_custom_adapter_failure(tmp_path):
    """v0.4.12 CachingLM persists our adapter hook one request at a time."""

    from lm_eval.api.model import CacheHook, CachingLM

    class InterruptingAdapter:
        def __init__(self, value, *, fail_after=None):
            self.value = value
            self.fail_after = fail_after
            self.calls = []
            self.cache_hook = CacheHook(None)

        def set_cache_hook(self, hook):
            self.cache_hook = hook

        def loglikelihood(self, requests):
            results = []
            for request in requests:
                self.calls.append(request.args)
                result = (self.value, True)
                self.cache_hook.add_partial("loglikelihood", request.args, result)
                results.append(result)
                if self.fail_after is not None and len(self.calls) == self.fail_after:
                    raise RuntimeError("simulated interruption")
            return results

    requests = [
        SimpleNamespace(args=("context-1", " continuation-1")),
        SimpleNamespace(args=("context-2", " continuation-2")),
    ]
    cache_path = str(tmp_path / "responses_rank0.db")
    first_adapter = InterruptingAdapter(1.0, fail_after=1)
    first = CachingLM(first_adapter, cache_path)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        first.loglikelihood(requests)
    first.dbdict.close()

    resumed_adapter = InterruptingAdapter(2.0)
    resumed = CachingLM(resumed_adapter, cache_path)
    assert resumed.loglikelihood(requests) == [(1.0, True), (2.0, True)]
    assert resumed_adapter.calls == [requests[1].args]
    resumed.dbdict.close()


def test_harness_task_wires_the_isolated_cache_base(tmp_path):
    calls = []

    class FakeLMEval:
        @staticmethod
        def simple_evaluate(**kwargs):
            calls.append(kwargs)
            return {"results": {"arc_challenge": {"acc_norm,none": 0.5}}}

    base = tmp_path / "isolated" / "responses"
    result = EVALUATE._run_harness_task(
        FakeLMEval(),
        adapter=object(),
        task="arc_challenge",
        benchmark={"num_fewshot": 0, "metric": "acc_norm"},
        limit=200.0,
        log_samples=True,
        task_manager=object(),
        request_cache_base=base,
    )
    assert result["results"]["arc_challenge"]["acc_norm,none"] == 0.5
    assert calls[0]["use_cache"] == str(base.resolve())
    assert calls[0]["apply_chat_template"] is False
    assert calls[0]["fewshot_as_multiturn"] is False


def test_run_definition_fingerprint_changes_with_log_samples(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    base_args = {
        "config": config_path,
        "limit": 1.0,
        "wikitext_max_chunks": 1,
    }
    configuration = {
        "modes": ("source_standalone", "target_standalone"),
        "tasks": ("arc_challenge",),
        "source_model": Path("/models/source"),
        "target_model": Path("/models/target"),
        "mapper_manifest": Path("/mapper/manifest.json"),
        "source_devices": (4, 5),
        "target_devices": (0, 1, 2, 3),
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "max_gen_tokens": 256,
        "benchmarks": {"arc_challenge": BENCHMARKS["arc_challenge"]},
    }
    without_samples = EVALUATE._build_run_definition(
        argparse.Namespace(**base_args, log_samples=False),
        configuration,
        config_sha256="config",
        benchmark_data={"datasets": {}},
        provenance={"commit": "harness"},
        runtime_artifacts={"source_model": {"config_sha256": "source"}},
        project_code={"files_digest": "code"},
    )
    with_samples = EVALUATE._build_run_definition(
        argparse.Namespace(**base_args, log_samples=True),
        configuration,
        config_sha256="config",
        benchmark_data={"datasets": {}},
        provenance={"commit": "harness"},
        runtime_artifacts={"source_model": {"config_sha256": "source"}},
        project_code={"files_digest": "code"},
    )
    assert without_samples["log_samples"] is False
    assert with_samples["log_samples"] is True
    assert EVALUATE._canonical_digest(without_samples) != EVALUATE._canonical_digest(
        with_samples
    )


def test_run_definition_fingerprint_binds_mapper_model_and_project_code(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("schema_version: 1\n", encoding="utf-8")
    args = argparse.Namespace(
        config=config_path,
        limit=1.0,
        wikitext_max_chunks=1,
        log_samples=False,
    )
    configuration = {
        "modes": ("mapped",),
        "tasks": ("arc_challenge",),
        "source_model": Path("/models/source"),
        "target_model": Path("/models/target"),
        "mapper_manifest": Path("/mapper/manifest.json"),
        "source_devices": (4, 5),
        "target_devices": (0, 1, 2, 3),
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "max_gen_tokens": 256,
        "benchmarks": {"arc_challenge": BENCHMARKS["arc_challenge"]},
    }
    runtime_artifacts = {
        "source_model": {"config": "source-a", "shards": [["s", 10]]},
        "target_model": {"config": "target-a", "shards": [["t", 20]]},
        "mapper": {"manifest_sha256": "mapper-a", "checksum_sha256": "commit-a"},
    }
    project_code = {"files_digest": "code-a"}

    def fingerprint(artifacts=runtime_artifacts, code=project_code):
        definition = EVALUATE._build_run_definition(
            args,
            configuration,
            config_sha256="config",
            benchmark_data={"datasets": {}},
            provenance={"commit": "harness"},
            runtime_artifacts=artifacts,
            project_code=code,
        )
        return EVALUATE._canonical_digest(definition)

    baseline = fingerprint()
    for section, field in (
        ("source_model", "config"),
        ("target_model", "config"),
        ("mapper", "manifest_sha256"),
    ):
        changed = json.loads(json.dumps(runtime_artifacts))
        changed[section][field] += "-changed"
        assert fingerprint(changed) != baseline
    assert fingerprint(code={"files_digest": "code-b"}) != baseline


def test_model_and_mapper_identity_contracts_detect_changes(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"layers":2}\n', encoding="utf-8")
    (model / "tokenizer.json").write_text('{"vocab":{}}\n', encoding="utf-8")
    (model / "model-1.safetensors").write_bytes(b"weights")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-1.safetensors"}}), encoding="utf-8"
    )
    first = EVALUATE.model_artifact_identity(model)
    assert first["weight_shards"] == [{"path": "model-1.safetensors", "nbytes": 7}]
    assert first["weight_shard_contents_hashed"] is False
    (model / "model-1.safetensors").write_bytes(b"weights-more")
    second = EVALUATE.model_artifact_identity(model)
    assert EVALUATE._canonical_digest(first) != EVALUATE._canonical_digest(second)

    mapper = tmp_path / "mapper"
    mapper.mkdir()
    manifest = mapper / "manifest.json"
    manifest.write_text(
        json.dumps({"status": "complete", "fit_fingerprint": "a" * 64, "layers": [{}, {}]}),
        encoding="utf-8",
    )
    atomic_write_json(
        mapper / "manifest.sha256.json",
        {"sha256": EVALUATE.sha256_file(manifest)},
    )
    identity = EVALUATE.mapper_artifact_identity(manifest)
    assert identity["manifest_sha256"] == identity["checksum_declared_manifest_sha256"]
    manifest.write_text('{"status":"complete","layers":[]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees"):
        EVALUATE.mapper_artifact_identity(manifest)


def test_summary_uses_only_accuracy_for_retention_and_separates_ppl(tmp_path):
    fingerprint = "formal-request"
    tasks = tuple(BENCHMARKS)
    modes = (
        "source_standalone",
        "target_standalone",
        "target_oracle",
        "mapped",
    )
    values = {
        "source_standalone": dict.fromkeys(tasks, 0.5),
        "target_standalone": {
            "arc_challenge": 0.60,
            "hellaswag": 0.80,
            "winogrande": 0.70,
            "mmlu": 0.80,
            "gsm8k_cot": 0.90,
            "wikitext2_prefix_ppl": 6.0,
        },
        "target_oracle": {
            "arc_challenge": 0.601,
            "hellaswag": 0.799,
            "winogrande": 0.70,
            "mmlu": 0.802,
            "gsm8k_cot": 0.899,
            "wikitext2_prefix_ppl": 6.01,
        },
        "mapped": {
            "arc_challenge": 0.58,
            "hellaswag": 0.76,
            "winogrande": 0.68,
            "mmlu": 0.72,
            "gsm8k_cot": 0.81,
            "wikitext2_prefix_ppl": 6.6,
        },
    }
    for mode in modes:
        for task in tasks:
            destination = EVALUATE._result_path(tmp_path, mode, task)
            atomic_write_json(
                destination,
                _checkpoint_payload(
                    fingerprint=fingerprint,
                    mode=mode,
                    task=task,
                    metric=BENCHMARKS[task]["metric"],
                    value=float(values[mode][task]),
                ),
            )

    configuration = {
        "modes": modes,
        "tasks": tasks,
        "benchmarks": BENCHMARKS,
    }
    summary = EVALUATE._summarize(tmp_path, configuration, fingerprint)
    retention = summary["mapped_retention_percent"]
    assert set(retention) == set(EVALUATE.ACCURACY_TASKS)
    assert "wikitext2_prefix_ppl" not in retention
    assert retention["hellaswag"] == pytest.approx(95.0)

    expected_retention_mean = sum(
        values["mapped"][task] / values["target_standalone"][task] * 100.0
        for task in EVALUATE.ACCURACY_TASKS
    ) / len(EVALUATE.ACCURACY_TASKS)
    comparison = summary["accuracy_comparison"]
    assert comparison["five_task_complete"] is True
    assert comparison["five_task_retention_arithmetic_mean_percent"] == pytest.approx(
        expected_retention_mean
    )
    expected_floor_mean = sum(
        (
            values["mapped"][task] - EVALUATE.CHANCE_FLOORS[task]
        )
        / (
            values["target_standalone"][task] - EVALUATE.CHANCE_FLOORS[task]
        )
        * 100.0
        for task in EVALUATE.ACCURACY_TASKS
    ) / len(EVALUATE.ACCURACY_TASKS)
    assert comparison[
        "five_task_floor_normalized_arithmetic_mean_percent"
    ] == pytest.approx(expected_floor_mean)

    oracle = summary["oracle_vs_target"]
    assert oracle["arc_challenge"]["absolute_delta"] == pytest.approx(0.001)
    assert oracle["arc_challenge"]["absolute_delta_percentage_points"] == pytest.approx(
        0.1
    )
    assert oracle["wikitext2_prefix_ppl"]["lower_is_better"] is True

    ppl = summary["wikitext2_prefix_ppl_comparison"]
    assert ppl["lower_is_better"] is True
    assert ppl["mapped_minus_target"] == pytest.approx(0.6)
    assert ppl["mapped_over_target_ratio"] == pytest.approx(1.1)

    persisted = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert persisted["mapped_retention_percent"] == retention


def test_partial_accuracy_summary_does_not_claim_five_task_mean(tmp_path):
    fingerprint = "pilot"
    task = "hellaswag"
    for mode, value in (("target_standalone", 0.8), ("mapped", 0.76)):
        atomic_write_json(
            EVALUATE._result_path(tmp_path, mode, task),
            _checkpoint_payload(
                fingerprint=fingerprint,
                mode=mode,
                task=task,
                metric="acc_norm",
                value=value,
            ),
        )
    summary = EVALUATE._summarize(
        tmp_path,
        {
            "modes": ("target_standalone", "mapped"),
            "tasks": (task,),
            "benchmarks": {task: BENCHMARKS[task]},
        },
        fingerprint,
    )
    comparison = summary["accuracy_comparison"]
    assert comparison["available_task_count"] == 1
    assert comparison["five_task_complete"] is False
    assert comparison["five_task_retention_arithmetic_mean_percent"] is None
    assert comparison["five_task_floor_normalized_arithmetic_mean_percent"] is None
