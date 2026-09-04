from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from kvcache_transfer.evaluation_aggregate import (
    ACCURACY_TASKS,
    AggregationError,
    CHANCE_FLOORS,
    EXPECTED_METRICS,
    PAPER_REFERENCE,
    PPL_TASK,
    REQUIRED_MODES,
    REQUIRED_TASKS,
    aggregate_runs,
    canonical_digest,
    write_aggregate_outputs,
)
from kvcache_transfer.io import atomic_write_json, sha256_file


TASK_DIRS = {
    "arc_challenge": "arc",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "mmlu": "mmlu",
    "gsm8k_cot": "gsm8k",
}


def _benchmarks():
    return {
        "arc_challenge": {"metric": "acc_norm", "num_fewshot": 0},
        "hellaswag": {"metric": "acc_norm", "num_fewshot": 0},
        "winogrande": {"metric": "acc", "num_fewshot": 0},
        "mmlu": {"metric": "acc", "num_fewshot": 5},
        "gsm8k_cot": {
            "metric": "exact_match",
            "metric_filter": "strict-match",
            "num_fewshot": 8,
        },
        PPL_TASK: {
            "metric": "perplexity",
            "chunk_length": 2048,
            "prefix_length": 1024,
            "scored_length": 1024,
            "dataset_path": "EleutherAI/wikitext_document_level",
            "dataset_name": "wikitext-2-raw-v1",
            "split": "test",
        },
    }


def _default_value(mode: str, task: str) -> float:
    return PAPER_REFERENCE[mode][task]


def _make_run(
    root: Path,
    *,
    modes: tuple[str, ...],
    tasks: tuple[str, ...],
    values: dict[tuple[str, str], float] | None = None,
    limit: float | None = None,
    wikitext_max_chunks: int | None = None,
    log_samples: bool = False,
    definition_mutator=None,
    resolved_tag: str | None = None,
) -> Path:
    harness_files = {
        f"lm_eval/tasks/{TASK_DIRS[task]}/{task}.yaml": {
            "sha256": f"task-file-{task}",
            "nbytes": 10 + len(task),
        }
        for task in tasks
        if task != PPL_TASK
    }
    code_files = {
        "scripts/evaluate.py": {"sha256": "formal-code", "nbytes": 1234},
        "src/kvcache_transfer/evaluation.py": {
            "sha256": "formal-evaluation-code",
            "nbytes": 5678,
        },
    }
    runtime = {}
    if set(modes) & {"source_standalone", "mapped"}:
        runtime["source_model"] = {"identity": "qwen3-14b"}
    if set(modes) & {"target_standalone", "target_oracle", "mapped"}:
        runtime["target_model"] = {"identity": "qwen3-32b"}
    if "mapped" in modes:
        runtime["mapper"] = {"manifest_sha256": "mapper-manifest"}
    definition = {
        "schema_version": 1,
        "config_path": "/project/configs/eval.yaml",
        "config_sha256": "config-sha",
        "modes": list(modes),
        "tasks": list(tasks),
        "limit": limit,
        "wikitext_max_chunks": wikitext_max_chunks,
        "log_samples": log_samples,
        "completion_mode": True,
        "apply_chat_template": False,
        "batch_size": 1,
        "boundary_replay": True,
        "execution_order": "task_outer_mode_inner",
        "request_response_cache": {"isolation": "request_fingerprint/mode/task"},
        "source_model": "/models/Qwen3-14B",
        "target_model": "/models/Qwen3-32B",
        "mapper_manifest": "/artifacts/mapper/manifest.json",
        "source_devices": [4, 5],
        "target_devices": [0, 1, 2, 3],
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "max_gen_tokens": 256,
        "benchmarks": _benchmarks(),
        "benchmark_data": {"manifest_sha256": "benchmark-manifest"},
        "harness": {
            "path": "/vendor/lm-evaluation-harness",
            "commit": "harness-commit",
            "dirty": False,
            "dirty_status": [],
            "package_version": "0.4.12",
            "task_files": harness_files,
            "task_files_digest": canonical_digest(harness_files),
        },
        "runtime_artifacts": runtime,
        "project_code": {
            "files": code_files,
            "files_digest": canonical_digest(code_files),
        },
    }
    if definition_mutator is not None:
        definition_mutator(definition)
    fingerprint = canonical_digest(definition)
    definition["request_fingerprint"] = fingerprint
    atomic_write_json(root / "run_definition.json", definition)

    summary_metrics = {mode: {} for mode in modes}
    for mode in modes:
        for task in tasks:
            value = (
                values[(mode, task)]
                if values is not None and (mode, task) in values
                else _default_value(mode, task)
            )
            metric = EXPECTED_METRICS[task]
            if task == PPL_TASK:
                chunks = 2 if wikitext_max_chunks is None else wikitext_max_chunks
                raw_result = {
                    "loglikelihood": -math.log(value) * chunks * 1024,
                    "scored_tokens": chunks * 1024,
                    "chunks": chunks,
                    "perplexity": value,
                }
                resolved = copy.deepcopy(definition["benchmarks"][task])
            else:
                task_config = {
                    task: {
                        "task": task,
                        "num_fewshot": definition["benchmarks"][task]["num_fewshot"],
                        "resolved_tag": resolved_tag,
                    }
                }
                metric_filter = "strict-match" if task == "gsm8k_cot" else "none"
                effective = 1 if limit is not None else 100
                raw_result = {
                    "config": {
                        "evaluation_mode": mode,
                        "model": (
                            definition["source_model"]
                            if mode == "source_standalone"
                            else definition["target_model"]
                        ),
                        "limit": limit,
                        "apply_chat_template": False,
                        "batch_size": 1,
                        "boundary_replay": True,
                        "completion_mode": True,
                    },
                    "configs": task_config,
                    "n-shot": {task: definition["benchmarks"][task]["num_fewshot"]},
                    "versions": {task: 1.0},
                    "n-samples": {
                        task: {"effective": effective, "original": 100}
                    },
                    "results": {task: {f"{metric},{metric_filter}": value}},
                }
                resolved = {
                    "configs": task_config,
                    "canonical_configs_sha256": canonical_digest(task_config),
                    "n_shot": raw_result["n-shot"],
                    "versions": raw_result["versions"],
                }
            checkpoint = {
                "schema_version": 1,
                "status": "complete",
                "request_fingerprint": fingerprint,
                "mode": mode,
                "task": task,
                "started_utc": "2026-08-13T00:00:00+00:00",
                "finished_utc": "2026-08-13T00:00:01+00:00",
                "primary_metric": metric,
                "primary_metric_value": value,
                "resolved_task_config": resolved,
                "result": raw_result,
            }
            atomic_write_json(root / "results" / mode / f"{task}.json", checkpoint)
            summary_metrics[mode][task] = value
    summary = {
        "schema_version": 1,
        "request_fingerprint": fingerprint,
        "updated_utc": "2026-08-13T00:00:02+00:00",
        "metrics": summary_metrics,
    }
    atomic_write_json(root / "summary.json", summary)
    atomic_write_json(
        root / "status.json",
        {
            "status": "complete",
            "failures": 0,
            "finished_utc": "2026-08-13T00:00:03+00:00",
            "summary": summary,
        },
    )
    return root


def _split_formal_runs(
    tmp_path: Path,
    *,
    omit: tuple[str, str] | None = None,
    values: dict[tuple[str, str], float] | None = None,
):
    runs = []
    for mode in REQUIRED_MODES:
        for task in REQUIRED_TASKS:
            if (mode, task) == omit:
                continue
            runs.append(
                _make_run(
                    tmp_path / f"{mode}-{task}",
                    modes=(mode,),
                    tasks=(task,),
                    values=values,
                )
            )
    return runs


def test_split_mode_task_runs_aggregate_and_write_hashed_paper_table(tmp_path):
    runs = _split_formal_runs(tmp_path)
    # log_samples is an execution/output choice and may differ across runs.
    run_definition = json.loads((runs[-1] / "run_definition.json").read_text())
    assert run_definition["log_samples"] is False

    result = aggregate_runs(runs)
    assert result["formal_full_dataset"] is True
    assert result["paper_comparison_valid"] is True
    assert set(result["metrics"]) == set(REQUIRED_MODES)
    expected_retention = sum(
        PAPER_REFERENCE["mapped"][task]
        / PAPER_REFERENCE["target_standalone"][task]
        * 100.0
        for task in ACCURACY_TASKS
    ) / len(ACCURACY_TASKS)
    assert result["five_accuracy_task_retention"][
        "reproduced_arithmetic_mean_percent"
    ] == pytest.approx(expected_retention)
    assert PPL_TASK not in result["five_accuracy_task_retention"]["tasks"]
    expected_floor_normalized = {
        task: (
            PAPER_REFERENCE["mapped"][task] - CHANCE_FLOORS[task]
        )
        / (
            PAPER_REFERENCE["target_standalone"][task] - CHANCE_FLOORS[task]
        )
        * 100.0
        for task in ACCURACY_TASKS
    }
    floor_summary = result["five_accuracy_task_floor_normalized_retention"]
    assert floor_summary["chance_floors"] == CHANCE_FLOORS
    assert floor_summary["tasks"] == pytest.approx(expected_floor_normalized)
    assert floor_summary["paper_tasks"] == pytest.approx(expected_floor_normalized)
    assert floor_summary["reproduced_arithmetic_mean_percent"] == pytest.approx(
        sum(expected_floor_normalized.values()) / len(ACCURACY_TASKS)
    )
    assert floor_summary["paper_arithmetic_mean_percent"] == pytest.approx(
        96.31787260466626
    )
    assert floor_summary["paper_reported_rounded_percent"] == 96.3
    assert PPL_TASK not in floor_summary["tasks"]
    rows = {row["task"]: row for row in result["table"]}
    assert rows[PPL_TASK]["chance_floor_percent"] is None
    assert rows[PPL_TASK]["mapped_floor_normalized_retention_percent"] is None
    for task in ACCURACY_TASKS:
        assert rows[task]["chance_floor_percent"] == CHANCE_FLOORS[task] * 100.0
        assert rows[task]["mapped_retention_percent"] == pytest.approx(
            PAPER_REFERENCE["mapped"][task]
            / PAPER_REFERENCE["target_standalone"][task]
            * 100.0
        )
        assert rows[task]["mapped_floor_normalized_retention_percent"] == pytest.approx(
            expected_floor_normalized[task]
        )

    output = tmp_path / "aggregate"
    manifest = write_aggregate_outputs(result, output)
    assert set(manifest["files"]) == {
        "aggregate.json",
        "paper_table.csv",
        "paper_table.md",
    }
    for name, identity in manifest["files"].items():
        assert identity["sha256"] == sha256_file(output / name)
        assert identity["nbytes"] == (output / name).stat().st_size
    markdown = (output / "paper_table.md").read_text()
    assert "WikiText-2 perplexity is excluded" in markdown
    assert "97.6%" in markdown
    assert "chance-floor-normalized retention" in markdown
    assert "96.3%" in markdown
    csv_header = (output / "paper_table.csv").read_text().splitlines()[0]
    assert "mapped_retention_percent" in csv_header
    assert "mapped_floor_normalized_retention_percent" in csv_header


def test_floor_normalized_retention_uses_fixed_floors_and_differs_from_raw(tmp_path):
    target_values = {
        "arc_challenge": 0.60,
        "hellaswag": 0.80,
        "winogrande": 0.70,
        "mmlu": 0.80,
        "gsm8k_cot": 0.90,
    }
    mapped_values = {
        "arc_challenge": 0.58,
        "hellaswag": 0.76,
        "winogrande": 0.68,
        "mmlu": 0.72,
        "gsm8k_cot": 0.81,
    }
    values = {
        **{
            ("target_standalone", task): value
            for task, value in target_values.items()
        },
        **{("mapped", task): value for task, value in mapped_values.items()},
    }
    result = aggregate_runs(_split_formal_runs(tmp_path, values=values))
    raw = result["five_accuracy_task_retention"]
    normalized = result["five_accuracy_task_floor_normalized_retention"]
    expected_raw = {
        task: mapped_values[task] / target_values[task] * 100.0
        for task in ACCURACY_TASKS
    }
    expected_normalized = {
        task: (mapped_values[task] - CHANCE_FLOORS[task])
        / (target_values[task] - CHANCE_FLOORS[task])
        * 100.0
        for task in ACCURACY_TASKS
    }
    assert raw["tasks"] == pytest.approx(expected_raw)
    assert normalized["tasks"] == pytest.approx(expected_normalized)
    assert raw["reproduced_arithmetic_mean_percent"] == pytest.approx(
        sum(expected_raw.values()) / len(ACCURACY_TASKS)
    )
    assert normalized["reproduced_arithmetic_mean_percent"] == pytest.approx(
        sum(expected_normalized.values()) / len(ACCURACY_TASKS)
    )
    assert normalized["tasks"]["gsm8k_cot"] == pytest.approx(
        raw["tasks"]["gsm8k_cot"]
    )
    assert normalized["tasks"]["winogrande"] != pytest.approx(
        raw["tasks"]["winogrande"]
    )


def test_rejects_floor_normalization_when_target_equals_chance_floor(tmp_path):
    values = {
        ("target_standalone", "arc_challenge"): CHANCE_FLOORS["arc_challenge"],
        ("mapped", "arc_challenge"): 0.30,
    }
    with pytest.raises(AggregationError, match="target score equals chance floor"):
        aggregate_runs(_split_formal_runs(tmp_path, values=values))


def test_valid_cross_fingerprint_partitioning_and_complete_optional_oracle(tmp_path):
    runs = _split_formal_runs(tmp_path)
    # Each execution partition has a different immutable request fingerprint.
    fingerprints = {
        json.loads((path / "run_definition.json").read_text())["request_fingerprint"]
        for path in runs
    }
    assert len(fingerprints) == len(runs)
    for task in REQUIRED_TASKS:
        runs.append(
            _make_run(
                tmp_path / f"target_oracle-{task}",
                modes=("target_oracle",),
                tasks=(task,),
                values={("target_oracle", task): PAPER_REFERENCE["target_standalone"][task]},
                log_samples=True,
            )
        )
    result = aggregate_runs(runs, require_oracle=True)
    assert set(result["metrics"]) == {*REQUIRED_MODES, "target_oracle"}
    assert all(row["target_oracle_reproduced"] is not None for row in result["table"])


def test_rejects_run_definition_tamper_without_matching_fingerprint(tmp_path):
    runs = _split_formal_runs(tmp_path)
    definition_path = runs[0] / "run_definition.json"
    definition = json.loads(definition_path.read_text())
    definition["boundary_replay"] = False
    atomic_write_json(definition_path, definition)
    with pytest.raises(AggregationError, match="fingerprint mismatch"):
        aggregate_runs(runs)


def test_rejects_valid_but_incompatible_project_code_identity(tmp_path):
    runs = _split_formal_runs(tmp_path)

    def change_code(definition):
        definition["project_code"]["files"]["scripts/evaluate.py"]["sha256"] = "changed"
        definition["project_code"]["files_digest"] = canonical_digest(
            definition["project_code"]["files"]
        )

    replacement = _make_run(
        tmp_path / "incompatible",
        modes=("mapped",),
        tasks=(PPL_TASK,),
        definition_mutator=change_code,
    )
    runs[-1] = replacement
    with pytest.raises(AggregationError, match="scientific run identity is incompatible"):
        aggregate_runs(runs)


def test_rejects_conflicting_duplicate_mode_task_value(tmp_path):
    runs = _split_formal_runs(tmp_path)
    duplicate = _make_run(
        tmp_path / "duplicate",
        modes=("target_standalone",),
        tasks=("arc_challenge",),
        values={("target_standalone", "arc_challenge"): 0.5},
    )
    with pytest.raises(AggregationError, match="duplicate target_standalone/arc_challenge"):
        aggregate_runs([*runs, duplicate])


def test_rejects_conflicting_resolved_config_even_when_metric_matches(tmp_path):
    runs = _split_formal_runs(tmp_path)
    duplicate = _make_run(
        tmp_path / "resolved-conflict",
        modes=("mapped",),
        tasks=("hellaswag",),
        resolved_tag="different",
    )
    with pytest.raises(AggregationError, match="resolved task config conflicts"):
        aggregate_runs([*runs, duplicate])


def test_rejects_missing_formal_task_pair(tmp_path):
    runs = _split_formal_runs(tmp_path, omit=("mapped", "gsm8k_cot"))
    with pytest.raises(AggregationError, match="missing mode/task checkpoints"):
        aggregate_runs(runs)


def test_rejects_nonfinite_checkpoint_json(tmp_path):
    runs = _split_formal_runs(tmp_path)
    path = runs[0] / "results" / "source_standalone" / "arc_challenge.json"
    checkpoint = json.loads(path.read_text())
    checkpoint["primary_metric_value"] = math.nan
    # atomic_write_json intentionally mirrors the experiment writer, whose
    # default encoder can expose an accidental NaN for this adversarial test.
    atomic_write_json(path, checkpoint)
    with pytest.raises(AggregationError, match="non-finite JSON constant"):
        aggregate_runs(runs)


def test_rejects_full_pilot_mix_for_same_task_even_when_sampled_allowed(tmp_path):
    runs = _split_formal_runs(tmp_path)
    pilot = _make_run(
        tmp_path / "pilot-arc",
        modes=("mapped",),
        tasks=("arc_challenge",),
        limit=1.0,
    )
    with pytest.raises(AggregationError, match="full/pilot or sample definition conflict"):
        aggregate_runs([*runs, pilot], allow_sampled=True)


def test_sample_limited_complete_table_requires_explicit_nonformal_opt_in(tmp_path):
    runs = []
    for mode in REQUIRED_MODES:
        runs.append(
            _make_run(
                tmp_path / f"pilot-{mode}",
                modes=(mode,),
                tasks=REQUIRED_TASKS,
                limit=1.0,
                wikitext_max_chunks=1,
            )
        )
    with pytest.raises(AggregationError, match="sample-limited pilot"):
        aggregate_runs(runs)
    result = aggregate_runs(runs, allow_sampled=True)
    assert result["formal_full_dataset"] is False
    assert result["paper_comparison_valid"] is False


def test_rejects_incomplete_oracle_column(tmp_path):
    runs = _split_formal_runs(tmp_path)
    runs.append(
        _make_run(
            tmp_path / "one-oracle",
            modes=("target_oracle",),
            tasks=("arc_challenge",),
            values={("target_oracle", "arc_challenge"): 0.61},
        )
    )
    with pytest.raises(AggregationError, match="partial oracle column"):
        aggregate_runs(runs)
