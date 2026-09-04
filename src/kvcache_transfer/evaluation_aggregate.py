"""Strict cross-run aggregation for the formal KV-cache evaluation.

Formal evaluation is intentionally allowed to run one mode and one task at a
time.  This module joins those atomic runs without weakening their provenance:
it revalidates every run fingerprint and checkpoint, separates scientific
identity from execution partitioning, and refuses incomplete or mixed-sample
tables.
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .io import sha256_file


SCHEMA_VERSION = 1
MODES = (
    "source_standalone",
    "target_standalone",
    "target_oracle",
    "mapped",
)
REQUIRED_MODES = ("source_standalone", "target_standalone", "mapped")
ACCURACY_TASKS = (
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu",
    "gsm8k_cot",
)
# Chance floors fixed in REPRODUCTION.md for the paper comparison.  They are
# fractions, matching the internally stored accuracies (not display percent).
CHANCE_FLOORS = {
    "arc_challenge": 0.25,
    "hellaswag": 0.25,
    "winogrande": 0.50,
    "mmlu": 0.25,
    "gsm8k_cot": 0.0,
}
PPL_TASK = "wikitext2_prefix_ppl"
REQUIRED_TASKS = (*ACCURACY_TASKS, PPL_TASK)
EXPECTED_METRICS = {
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "mmlu": "acc",
    "gsm8k_cot": "exact_match",
    PPL_TASK: "perplexity",
}
METRIC_FILTERS = {"gsm8k_cot": "strict-match"}
HARNESS_TASK_DIRECTORIES = {
    "arc_challenge": "arc",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "mmlu": "mmlu",
    "gsm8k_cot": "gsm8k",
}

# Table 1 / Section 4 values for Qwen3-14B -> Qwen3-32B.  Accuracy values are
# stored as fractions internally; WikiText-2 is raw perplexity.
PAPER_REFERENCE = {
    "source_standalone": {
        "arc_challenge": 0.6049,
        "hellaswag": 0.7887,
        "winogrande": 0.6930,
        "mmlu": 0.7886,
        "gsm8k_cot": 0.9500,
        PPL_TASK: 7.63,
    },
    "target_standalone": {
        "arc_challenge": 0.6101,
        "hellaswag": 0.8265,
        "winogrande": 0.7001,
        "mmlu": 0.8217,
        "gsm8k_cot": 0.9515,
        PPL_TASK: 6.79,
    },
    "mapped": {
        "arc_challenge": 0.6160,
        "hellaswag": 0.8070,
        "winogrande": 0.6898,
        "mmlu": 0.7809,
        "gsm8k_cot": 0.9098,
        PPL_TASK: 7.33,
    },
}


class AggregationError(ValueError):
    """Raised when inputs cannot form one defensible paper comparison."""


@dataclass(frozen=True)
class CheckpointRecord:
    run_dir: Path
    path: Path
    mode: str
    task: str
    metric: str
    value: float
    resolved_task_config: Any
    sampling: Any
    sha256: str


@dataclass(frozen=True)
class RunRecord:
    run_dir: Path
    fingerprint: str
    definition: Mapping[str, Any]
    compatibility_identity: Mapping[str, Any]
    runtime_artifacts: Mapping[str, Any]
    harness_task_files: Mapping[str, Any]
    checkpoints: tuple[CheckpointRecord, ...]
    provenance: Mapping[str, Any]


def canonical_digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AggregationError(f"value is not canonical finite JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise AggregationError(f"non-finite JSON constant is forbidden: {value}")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise AggregationError(f"required JSON file is missing: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_reject_json_constant)
    except AggregationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise AggregationError(f"cannot read valid JSON from {path}: {exc}") from exc


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"{label} must be a number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise AggregationError(f"{label} must be finite, got {value!r}")
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AggregationError(f"{label} must be a JSON object")
    return value


def _require_unique_strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise AggregationError(f"{label} must be a non-empty list of strings")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise AggregationError(f"{label} contains duplicates: {result}")
    return result


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: types differ ({type(left).__name__} != {type(right).__name__})"
    if isinstance(left, Mapping):
        left_keys, right_keys = set(left), set(right)
        if left_keys != right_keys:
            return f"{path}: keys differ ({sorted(left_keys ^ right_keys)})"
        for key in sorted(left):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths differ ({len(left)} != {len(right)})"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(left_item, right_item, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if left != right:
        return f"{path}: values differ ({left!r} != {right!r})"
    return None


def _validate_digest_container(
    value: Mapping[str, Any], *, items_key: str, digest_key: str, label: str
) -> None:
    items = _require_mapping(value.get(items_key), f"{label}.{items_key}")
    expected = canonical_digest(items)
    if value.get(digest_key) != expected:
        raise AggregationError(
            f"{label}.{digest_key} does not hash {items_key}: "
            f"{value.get(digest_key)!r} != {expected!r}"
        )


def _compatibility_identity(definition: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only execution partitioning and separately merged identities."""

    identity = copy.deepcopy(dict(definition))
    for key in (
        "request_fingerprint",
        "modes",
        "tasks",
        "limit",
        "wikitext_max_chunks",
        "log_samples",
        "created_utc",
        "started_utc",
        "finished_utc",
        "updated_utc",
        "output_dir",
        "output_path",
    ):
        identity.pop(key, None)
    # Mode subsets legitimately capture only the runtime artifacts they use;
    # those are union-merged with conflict checks below.
    identity.pop("runtime_artifacts", None)
    harness = _require_mapping(identity.get("harness"), "run_definition.harness")
    harness = copy.deepcopy(dict(harness))
    harness.pop("task_files", None)
    harness.pop("task_files_digest", None)
    identity["harness"] = harness
    return identity


def _task_harness_files(
    all_files: Mapping[str, Any], task: str, *, run_dir: Path
) -> dict[str, Any]:
    if task == PPL_TASK:
        return {}
    directory = HARNESS_TASK_DIRECTORIES[task]
    marker = f"/tasks/{directory}/"
    selected = {key: value for key, value in all_files.items() if marker in f"/{key}"}
    if not selected:
        raise AggregationError(
            f"{run_dir}: harness provenance has no task files for {task} ({directory})"
        )
    return selected


def _validate_runtime_artifacts(
    runtime: Mapping[str, Any], modes: Sequence[str], *, run_dir: Path
) -> None:
    required: set[str] = set()
    if set(modes) & {"source_standalone", "mapped"}:
        required.add("source_model")
    if set(modes) & {"target_standalone", "target_oracle", "mapped"}:
        required.add("target_model")
    if "mapped" in modes:
        required.add("mapper")
    missing = sorted(required - set(runtime))
    if missing:
        raise AggregationError(f"{run_dir}: runtime_artifacts lacks {missing}")


def _derive_harness_metric(
    raw_result: Mapping[str, Any], task: str, metric: str
) -> float:
    task_results = _require_mapping(raw_result.get("results", {}), "result.results")
    group_results = _require_mapping(raw_result.get("groups", {}), "result.groups")
    if task in group_results:
        values = _require_mapping(group_results[task], f"result.groups.{task}")
    elif task in task_results:
        values = _require_mapping(task_results[task], f"result.results.{task}")
    else:
        raise AggregationError(f"raw lm-eval result has no aggregate entry for {task}")
    wanted_filter = METRIC_FILTERS.get(task)
    candidates: list[tuple[str, float]] = []
    for key, value in values.items():
        base, _, filter_name = key.partition(",")
        if base == metric and (wanted_filter is None or filter_name == wanted_filter):
            candidates.append((key, _finite_number(value, f"raw metric {task}.{key}")))
    if len(candidates) != 1:
        raise AggregationError(
            f"expected exactly one raw {metric!r} value for {task}, "
            f"found {[key for key, _ in candidates]}"
        )
    return candidates[0][1]


def _validate_sample_counts(n_samples: Any, *, task: str, full: bool) -> Any:
    values = _require_mapping(n_samples, f"{task} result.n-samples")
    if not values:
        raise AggregationError(f"{task} result.n-samples is empty")
    for name, item in values.items():
        item = _require_mapping(item, f"{task} result.n-samples.{name}")
        effective = item.get("effective")
        original = item.get("original")
        if (
            isinstance(effective, bool)
            or not isinstance(effective, int)
            or effective <= 0
            or isinstance(original, bool)
            or not isinstance(original, int)
            or original <= 0
            or effective > original
        ):
            raise AggregationError(f"invalid sample counts for {task}/{name}: {item}")
        if full and effective != original:
            raise AggregationError(
                f"full {task} run scored {effective}/{original} samples for {name}"
            )
    return copy.deepcopy(values)


def _validate_checkpoint(
    path: Path,
    *,
    run_dir: Path,
    fingerprint: str,
    definition: Mapping[str, Any],
    mode: str,
    task: str,
) -> CheckpointRecord:
    payload = _require_mapping(_load_json(path), str(path))
    expected_metric = EXPECTED_METRICS[task]
    checks = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "mode": mode,
        "task": task,
        "primary_metric": expected_metric,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise AggregationError(
                f"{path}: {key} mismatch ({payload.get(key)!r} != {expected!r})"
            )
    benchmarks = _require_mapping(definition.get("benchmarks"), "run_definition.benchmarks")
    benchmark = _require_mapping(benchmarks.get(task), f"benchmarks.{task}")
    if benchmark.get("metric") != expected_metric:
        raise AggregationError(
            f"{run_dir}: benchmark metric for {task} is {benchmark.get('metric')!r}, "
            f"expected {expected_metric!r}"
        )
    value = _finite_number(payload.get("primary_metric_value"), f"{path} primary metric")
    if task in ACCURACY_TASKS and not 0.0 <= value <= 1.0:
        raise AggregationError(f"{path}: accuracy metric lies outside [0,1]: {value}")
    if task == PPL_TASK and value <= 0.0:
        raise AggregationError(f"{path}: perplexity must be positive: {value}")

    resolved = payload.get("resolved_task_config")
    raw = _require_mapping(payload.get("result"), f"{path}.result")
    if task == PPL_TASK:
        difference = _first_difference(resolved, benchmark)
        if difference is not None:
            raise AggregationError(f"{path}: WikiText resolved config mismatch: {difference}")
        raw_value = _finite_number(raw.get("perplexity"), f"{path} raw perplexity")
        if raw_value != value:
            raise AggregationError(f"{path}: primary/raw perplexity differ ({value} != {raw_value})")
        chunks = raw.get("chunks")
        scored_tokens = raw.get("scored_tokens")
        if isinstance(chunks, bool) or not isinstance(chunks, int) or chunks <= 0:
            raise AggregationError(f"{path}: invalid WikiText chunk count {chunks!r}")
        if (
            isinstance(scored_tokens, bool)
            or not isinstance(scored_tokens, int)
            or scored_tokens <= 0
        ):
            raise AggregationError(f"{path}: invalid WikiText scored token count {scored_tokens!r}")
        expected_tokens = chunks * int(benchmark["scored_length"])
        if scored_tokens != expected_tokens:
            raise AggregationError(
                f"{path}: WikiText scored {scored_tokens} tokens, expected {expected_tokens}"
            )
        _finite_number(raw.get("loglikelihood"), f"{path} raw loglikelihood")
        sampling = {
            "kind": "wikitext_nonoverlap_chunks",
            "requested_max_chunks": definition.get("wikitext_max_chunks"),
            "actual_chunks": chunks,
            "scored_tokens": scored_tokens,
        }
    else:
        resolved_map = _require_mapping(resolved, f"{path}.resolved_task_config")
        raw_configs = _require_mapping(raw.get("configs"), f"{path}.result.configs")
        if resolved_map.get("configs") != raw_configs:
            raise AggregationError(f"{path}: resolved configs do not equal raw lm-eval configs")
        if resolved_map.get("canonical_configs_sha256") != canonical_digest(raw_configs):
            raise AggregationError(f"{path}: canonical resolved task-config digest is invalid")
        if resolved_map.get("n_shot") != raw.get("n-shot", {}):
            raise AggregationError(f"{path}: resolved n_shot does not equal raw n-shot")
        if resolved_map.get("versions") != raw.get("versions", {}):
            raise AggregationError(f"{path}: resolved versions do not equal raw versions")
        raw_value = _derive_harness_metric(raw, task, expected_metric)
        if raw_value != value:
            raise AggregationError(f"{path}: primary/raw metric differ ({value} != {raw_value})")
        raw_config = _require_mapping(raw.get("config"), f"{path}.result.config")
        expected_model = (
            definition.get("source_model")
            if mode == "source_standalone"
            else definition.get("target_model")
        )
        config_checks = {
            "evaluation_mode": mode,
            "model": expected_model,
            "limit": definition.get("limit"),
            "apply_chat_template": definition.get("apply_chat_template"),
            "batch_size": definition.get("batch_size"),
            "boundary_replay": definition.get("boundary_replay"),
            "completion_mode": definition.get("completion_mode"),
        }
        for key, expected in config_checks.items():
            if raw_config.get(key) != expected:
                raise AggregationError(
                    f"{path}: raw config {key} mismatch "
                    f"({raw_config.get(key)!r} != {expected!r})"
                )
        sample_counts = _validate_sample_counts(
            raw.get("n-samples"), task=task, full=definition.get("limit") is None
        )
        sampling = {
            "kind": "lm_eval_examples",
            "requested_limit": definition.get("limit"),
            "n_samples": sample_counts,
        }
    return CheckpointRecord(
        run_dir=run_dir,
        path=path,
        mode=mode,
        task=task,
        metric=expected_metric,
        value=value,
        resolved_task_config=copy.deepcopy(resolved),
        sampling=sampling,
        sha256=sha256_file(path),
    )


def validate_run(run_dir: str | Path) -> RunRecord:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise AggregationError(f"evaluation run directory does not exist: {root}")
    definition_path = root / "run_definition.json"
    definition = _require_mapping(_load_json(definition_path), str(definition_path))
    if definition.get("schema_version") != SCHEMA_VERSION:
        raise AggregationError(f"{root}: unsupported run_definition schema")
    fingerprint = definition.get("request_fingerprint")
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise AggregationError(f"{root}: invalid request_fingerprint")
    unhashed = dict(definition)
    unhashed.pop("request_fingerprint", None)
    recomputed = canonical_digest(unhashed)
    if recomputed != fingerprint:
        raise AggregationError(
            f"{root}: run_definition fingerprint mismatch ({fingerprint} != {recomputed})"
        )
    modes = _require_unique_strings(definition.get("modes"), f"{root} modes")
    tasks = _require_unique_strings(definition.get("tasks"), f"{root} tasks")
    unknown_modes = sorted(set(modes) - set(MODES))
    unknown_tasks = sorted(set(tasks) - set(REQUIRED_TASKS))
    if unknown_modes or unknown_tasks:
        raise AggregationError(
            f"{root}: unsupported modes/tasks: modes={unknown_modes}, tasks={unknown_tasks}"
        )

    project_code = _require_mapping(definition.get("project_code"), "project_code")
    _validate_digest_container(
        project_code, items_key="files", digest_key="files_digest", label="project_code"
    )
    harness = _require_mapping(definition.get("harness"), "harness")
    _validate_digest_container(
        harness,
        items_key="task_files",
        digest_key="task_files_digest",
        label="harness",
    )
    harness_files = _require_mapping(harness.get("task_files"), "harness.task_files")
    for task in tasks:
        _task_harness_files(harness_files, task, run_dir=root)
    runtime = _require_mapping(definition.get("runtime_artifacts"), "runtime_artifacts")
    _validate_runtime_artifacts(runtime, modes, run_dir=root)

    status = _require_mapping(_load_json(root / "status.json"), f"{root}/status.json")
    failures = status.get("failures")
    if (
        status.get("status") != "complete"
        or isinstance(failures, bool)
        or failures != 0
    ):
        raise AggregationError(f"{root}: run status is not cleanly complete: {status.get('status')}")
    summary = _require_mapping(_load_json(root / "summary.json"), f"{root}/summary.json")
    if summary.get("schema_version") != SCHEMA_VERSION:
        raise AggregationError(f"{root}: unsupported summary schema")
    if summary.get("request_fingerprint") != fingerprint:
        raise AggregationError(f"{root}: summary fingerprint mismatch")
    if status.get("summary") != summary:
        raise AggregationError(f"{root}: embedded status summary differs from summary.json")

    expected_paths = {
        (root / "results" / mode / f"{task}.json").resolve()
        for mode in modes
        for task in tasks
    }
    result_root = root / "results"
    actual_paths = {path.resolve() for path in result_root.glob("*/*.json")} if result_root.is_dir() else set()
    if actual_paths != expected_paths:
        missing = sorted(str(x) for x in expected_paths - actual_paths)
        unexpected = sorted(str(x) for x in actual_paths - expected_paths)
        raise AggregationError(
            f"{root}: checkpoint set mismatch; missing={missing}, unexpected={unexpected}"
        )

    checkpoints = tuple(
        _validate_checkpoint(
            root / "results" / mode / f"{task}.json",
            run_dir=root,
            fingerprint=fingerprint,
            definition=definition,
            mode=mode,
            task=task,
        )
        for mode in modes
        for task in tasks
    )
    summary_metrics = _require_mapping(summary.get("metrics"), f"{root} summary.metrics")
    if set(summary_metrics) != set(modes):
        raise AggregationError(
            f"{root}: summary mode set differs from run_definition "
            f"({sorted(summary_metrics)} != {sorted(modes)})"
        )
    for mode in modes:
        mode_values = _require_mapping(summary_metrics[mode], f"summary.metrics.{mode}")
        if set(mode_values) != set(tasks):
            raise AggregationError(
                f"{root}: summary task set for {mode} differs from run_definition "
                f"({sorted(mode_values)} != {sorted(tasks)})"
            )
    for checkpoint in checkpoints:
        mode_values = _require_mapping(
            summary_metrics.get(checkpoint.mode), f"summary.metrics.{checkpoint.mode}"
        )
        summarized = _finite_number(
            mode_values.get(checkpoint.task),
            f"summary.metrics.{checkpoint.mode}.{checkpoint.task}",
        )
        if summarized != checkpoint.value:
            raise AggregationError(
                f"{root}: summary/checkpoint metric mismatch for "
                f"{checkpoint.mode}/{checkpoint.task}"
            )

    provenance = {
        "run_dir": str(root),
        "request_fingerprint": fingerprint,
        "run_definition_sha256": sha256_file(definition_path),
        "status_sha256": sha256_file(root / "status.json"),
        "summary_sha256": sha256_file(root / "summary.json"),
        "modes": list(modes),
        "tasks": list(tasks),
        "limit": definition.get("limit"),
        "wikitext_max_chunks": definition.get("wikitext_max_chunks"),
        "log_samples": definition.get("log_samples"),
    }
    return RunRecord(
        run_dir=root,
        fingerprint=fingerprint,
        definition=copy.deepcopy(definition),
        compatibility_identity=_compatibility_identity(definition),
        runtime_artifacts=copy.deepcopy(runtime),
        harness_task_files=copy.deepcopy(harness_files),
        checkpoints=checkpoints,
        provenance=provenance,
    )


def _merge_named_identity(
    destination: dict[str, Any], incoming: Mapping[str, Any], *, label: str, source: Path
) -> None:
    for key, value in incoming.items():
        if key not in destination:
            destination[key] = copy.deepcopy(value)
            continue
        difference = _first_difference(destination[key], value, f"$.{label}.{key}")
        if difference is not None:
            raise AggregationError(f"{source}: conflicting {label} identity: {difference}")


def _paper_retention(task: str) -> float:
    return (
        PAPER_REFERENCE["mapped"][task]
        / PAPER_REFERENCE["target_standalone"][task]
        * 100.0
    )


def _floor_normalized_retention(
    *, mapped: float, target: float, floor: float, task: str, label: str
) -> float:
    denominator = target - floor
    if denominator == 0.0:
        raise AggregationError(
            f"cannot compute {label} floor-normalized retention for {task}: "
            f"target score equals chance floor ({target})"
        )
    return (mapped - floor) / denominator * 100.0


def _build_rows(
    metrics: Mapping[str, Mapping[str, float]], *, include_oracle: bool
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    retention: dict[str, float] = {}
    paper_retention: dict[str, float] = {}
    floor_normalized: dict[str, float] = {}
    paper_floor_normalized: dict[str, float] = {}
    for task in REQUIRED_TASKS:
        accuracy = task in ACCURACY_TASKS
        scale = 100.0 if accuracy else 1.0
        row: dict[str, Any] = {
            "task": task,
            "metric": EXPECTED_METRICS[task],
            "unit": "percent" if accuracy else "perplexity",
        }
        for mode in REQUIRED_MODES:
            reproduced = metrics[mode][task]
            paper = PAPER_REFERENCE[mode][task]
            row[f"{mode}_paper"] = paper * scale
            row[f"{mode}_reproduced"] = reproduced * scale
            row[f"{mode}_delta"] = (reproduced - paper) * scale
        row["target_oracle_reproduced"] = (
            metrics["target_oracle"][task] * scale if include_oracle else None
        )
        if accuracy:
            target = metrics["target_standalone"][task]
            mapped = metrics["mapped"][task]
            floor = CHANCE_FLOORS[task]
            if target == 0.0:
                raise AggregationError(f"cannot compute mapped retention for zero target score: {task}")
            retention[task] = mapped / target * 100.0
            paper_retention[task] = _paper_retention(task)
            floor_normalized[task] = _floor_normalized_retention(
                mapped=mapped,
                target=target,
                floor=floor,
                task=task,
                label="reproduced",
            )
            paper_floor_normalized[task] = _floor_normalized_retention(
                mapped=PAPER_REFERENCE["mapped"][task],
                target=PAPER_REFERENCE["target_standalone"][task],
                floor=floor,
                task=task,
                label="paper",
            )
            row["chance_floor_percent"] = floor * 100.0
            row["mapped_retention_percent"] = retention[task]
            row["paper_retention_percent"] = paper_retention[task]
            row["retention_delta_percentage_points"] = (
                retention[task] - paper_retention[task]
            )
            row["mapped_floor_normalized_retention_percent"] = floor_normalized[task]
            row["paper_floor_normalized_retention_percent"] = paper_floor_normalized[task]
            row["floor_normalized_retention_delta_percentage_points"] = (
                floor_normalized[task] - paper_floor_normalized[task]
            )
        else:
            row["chance_floor_percent"] = None
            row["mapped_retention_percent"] = None
            row["paper_retention_percent"] = None
            row["retention_delta_percentage_points"] = None
            row["mapped_floor_normalized_retention_percent"] = None
            row["paper_floor_normalized_retention_percent"] = None
            row["floor_normalized_retention_delta_percentage_points"] = None
        rows.append(row)
    reproduced_mean = sum(retention.values()) / len(ACCURACY_TASKS)
    paper_mean = sum(paper_retention.values()) / len(ACCURACY_TASKS)
    reproduced_floor_mean = sum(floor_normalized.values()) / len(ACCURACY_TASKS)
    paper_floor_mean = sum(paper_floor_normalized.values()) / len(ACCURACY_TASKS)
    return (
        rows,
        {
            "definition": "unweighted arithmetic mean of raw mapped/target retention over five accuracy tasks; WikiText-2 perplexity excluded",
            "tasks": retention,
            "paper_tasks": paper_retention,
            "reproduced_arithmetic_mean_percent": reproduced_mean,
            "paper_arithmetic_mean_percent": paper_mean,
            "paper_reported_rounded_percent": 97.6,
            "delta_from_paper_unrounded_percentage_points": reproduced_mean - paper_mean,
            "delta_from_paper_reported_percentage_points": reproduced_mean - 97.6,
        },
        {
            "definition": "unweighted arithmetic mean of (mapped - chance_floor) / (target - chance_floor) retention over five accuracy tasks; WikiText-2 perplexity excluded",
            "chance_floors": dict(CHANCE_FLOORS),
            "tasks": floor_normalized,
            "paper_tasks": paper_floor_normalized,
            "reproduced_arithmetic_mean_percent": reproduced_floor_mean,
            "paper_arithmetic_mean_percent": paper_floor_mean,
            "paper_reported_rounded_percent": 96.3,
            "delta_from_paper_unrounded_percentage_points": (
                reproduced_floor_mean - paper_floor_mean
            ),
            "delta_from_paper_reported_percentage_points": (
                reproduced_floor_mean - 96.3
            ),
        },
    )


def aggregate_runs(
    run_dirs: Iterable[str | Path],
    *,
    allow_sampled: bool = False,
    require_oracle: bool = False,
) -> dict[str, Any]:
    roots: list[Path] = []
    seen_roots: set[Path] = set()
    for value in run_dirs:
        root = Path(value).resolve()
        if root not in seen_roots:
            roots.append(root)
            seen_roots.add(root)
    if not roots:
        raise AggregationError("at least one evaluation run directory is required")
    records = [validate_run(root) for root in roots]

    reference_identity = records[0].compatibility_identity
    for record in records[1:]:
        difference = _first_difference(reference_identity, record.compatibility_identity)
        if difference is not None:
            raise AggregationError(
                f"{record.run_dir}: scientific run identity is incompatible: {difference}"
            )

    runtime_artifacts: dict[str, Any] = {}
    harness_files: dict[str, Any] = {}
    task_harness_identities: dict[str, Any] = {}
    task_resolved_configs: dict[str, Any] = {}
    task_sampling: dict[str, Any] = {}
    pairs: dict[tuple[str, str], CheckpointRecord] = {}
    for record in records:
        _merge_named_identity(
            runtime_artifacts,
            record.runtime_artifacts,
            label="runtime_artifacts",
            source=record.run_dir,
        )
        _merge_named_identity(
            harness_files,
            record.harness_task_files,
            label="harness.task_files",
            source=record.run_dir,
        )
        for checkpoint in record.checkpoints:
            task_files = _task_harness_files(
                record.harness_task_files, checkpoint.task, run_dir=record.run_dir
            )
            if checkpoint.task in task_harness_identities:
                difference = _first_difference(
                    task_harness_identities[checkpoint.task], task_files
                )
                if difference is not None:
                    raise AggregationError(
                        f"{record.run_dir}: harness files conflict for {checkpoint.task}: {difference}"
                    )
            else:
                task_harness_identities[checkpoint.task] = task_files
            if checkpoint.task in task_resolved_configs:
                difference = _first_difference(
                    task_resolved_configs[checkpoint.task], checkpoint.resolved_task_config
                )
                if difference is not None:
                    raise AggregationError(
                        f"{record.run_dir}: resolved task config conflicts for "
                        f"{checkpoint.task}: {difference}"
                    )
            else:
                task_resolved_configs[checkpoint.task] = checkpoint.resolved_task_config
            if checkpoint.task in task_sampling:
                difference = _first_difference(task_sampling[checkpoint.task], checkpoint.sampling)
                if difference is not None:
                    raise AggregationError(
                        f"{record.run_dir}: full/pilot or sample definition conflict for "
                        f"{checkpoint.task}: {difference}"
                    )
            else:
                task_sampling[checkpoint.task] = checkpoint.sampling
            pair = (checkpoint.mode, checkpoint.task)
            if pair in pairs:
                previous = pairs[pair]
                differences = [
                    _first_difference(previous.metric, checkpoint.metric, "$.metric"),
                    _first_difference(previous.value, checkpoint.value, "$.value"),
                    _first_difference(
                        previous.resolved_task_config,
                        checkpoint.resolved_task_config,
                        "$.resolved_task_config",
                    ),
                    _first_difference(previous.sampling, checkpoint.sampling, "$.sampling"),
                ]
                conflict = next((item for item in differences if item is not None), None)
                if conflict is not None:
                    raise AggregationError(
                        f"duplicate {pair[0]}/{pair[1]} checkpoint conflicts between "
                        f"{previous.path} and {checkpoint.path}: {conflict}"
                    )
            else:
                pairs[pair] = checkpoint

    missing_artifacts = sorted({"source_model", "target_model", "mapper"} - set(runtime_artifacts))
    if missing_artifacts:
        raise AggregationError(f"aggregate runtime artifact identity lacks {missing_artifacts}")
    required_pairs = {
        (mode, task) for mode in REQUIRED_MODES for task in REQUIRED_TASKS
    }
    missing_pairs = sorted(required_pairs - set(pairs))
    if missing_pairs:
        raise AggregationError(f"formal paper table is missing mode/task checkpoints: {missing_pairs}")
    oracle_pairs = {pair for pair in pairs if pair[0] == "target_oracle"}
    expected_oracle = {("target_oracle", task) for task in REQUIRED_TASKS}
    if require_oracle and oracle_pairs != expected_oracle:
        raise AggregationError(
            f"oracle was required but is incomplete; missing={sorted(expected_oracle - oracle_pairs)}"
        )
    if oracle_pairs and oracle_pairs != expected_oracle:
        raise AggregationError(
            f"partial oracle column is forbidden; missing={sorted(expected_oracle - oracle_pairs)}"
        )
    unexpected_pairs = sorted(
        set(pairs) - required_pairs - expected_oracle
    )
    if unexpected_pairs:
        raise AggregationError(f"unexpected aggregate checkpoints: {unexpected_pairs}")

    formal_full_dataset = all(
        (
            descriptor["requested_max_chunks"] is None
            if task == PPL_TASK
            else descriptor["requested_limit"] is None
        )
        for task, descriptor in task_sampling.items()
    )
    if not allow_sampled and not formal_full_dataset:
        raise AggregationError(
            "sample-limited pilot results cannot produce the formal paper table; "
            "pass allow_sampled=True only for an explicitly non-formal aggregate"
        )

    metrics = {
        mode: {task: pairs[(mode, task)].value for task in REQUIRED_TASKS}
        for mode in REQUIRED_MODES
    }
    if oracle_pairs:
        metrics["target_oracle"] = {
            task: pairs[("target_oracle", task)].value for task in REQUIRED_TASKS
        }
    rows, retention, floor_normalized_retention = _build_rows(
        metrics, include_oracle=bool(oracle_pairs)
    )
    identity_bundle = {
        "scientific_run_identity": reference_identity,
        "runtime_artifacts": runtime_artifacts,
        "harness_task_files": harness_files,
        "resolved_task_configs": task_resolved_configs,
        "sampling": task_sampling,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "formal_full_dataset": formal_full_dataset,
        "paper_comparison_valid": formal_full_dataset,
        "input_runs": [record.provenance for record in records],
        "compatibility_identity_sha256": canonical_digest(identity_bundle),
        "merged_identity": {
            "base_scientific_identity_sha256": canonical_digest(reference_identity),
            "runtime_artifacts": runtime_artifacts,
            "runtime_artifacts_sha256": canonical_digest(runtime_artifacts),
            "harness_task_files_sha256": canonical_digest(harness_files),
            "resolved_task_configs_sha256": canonical_digest(task_resolved_configs),
            "sampling": task_sampling,
            "sampling_sha256": canonical_digest(task_sampling),
            "project_code_files_digest": reference_identity["project_code"]["files_digest"],
        },
        "checkpoints": [
            {
                "mode": mode,
                "task": task,
                "path": str(checkpoint.path),
                "sha256": checkpoint.sha256,
            }
            for (mode, task), checkpoint in sorted(pairs.items())
        ],
        "metrics": metrics,
        "paper_reference": PAPER_REFERENCE,
        "table": rows,
        "five_accuracy_task_retention": retention,
        "five_accuracy_task_floor_normalized_retention": floor_normalized_retention,
        "wikitext2_prefix_perplexity": {
            mode: metrics[mode][PPL_TASK] for mode in metrics
        },
    }
    result["aggregate_payload_sha256"] = canonical_digest(result)
    return result


CSV_COLUMNS = (
    "task",
    "metric",
    "unit",
    "chance_floor_percent",
    "source_standalone_paper",
    "source_standalone_reproduced",
    "source_standalone_delta",
    "target_standalone_paper",
    "target_standalone_reproduced",
    "target_standalone_delta",
    "mapped_paper",
    "mapped_reproduced",
    "mapped_delta",
    "target_oracle_reproduced",
    "mapped_retention_percent",
    "paper_retention_percent",
    "retention_delta_percentage_points",
    "mapped_floor_normalized_retention_percent",
    "paper_floor_normalized_retention_percent",
    "floor_normalized_retention_delta_percentage_points",
)


def render_csv(result: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in result["table"]:
        writer.writerow({key: row.get(key) for key in CSV_COLUMNS})
    return buffer.getvalue()


def _display(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def render_markdown(result: Mapping[str, Any]) -> str:
    oracle = "target_oracle" in result["metrics"]
    formal = bool(result["formal_full_dataset"])
    header = (
        "| Task | Metric | Source paper / ours / Δ | Target paper / ours / Δ | "
        "Mapped paper / ours / Δ |"
    )
    separator = "|---|---|---:|---:|---:|"
    if oracle:
        header += " Oracle ours |"
        separator += "---:|"
    header += (
        " Raw mapped retention paper / ours / Δpp | Chance floor | "
        "Floor-normalized retention paper / ours / Δpp |"
    )
    separator += "---:|---:|---:|"
    lines = [
        (
            "# Qwen3-14B → Qwen3-32B formal evaluation"
            if formal
            else "# Qwen3-14B → Qwen3-32B sampled evaluation (non-formal)"
        ),
        "",
        f"- Full-dataset protocol: `{str(formal).lower()}`",
        f"- Valid paper-result comparison: `{str(formal).lower()}`",
        f"- Compatibility identity: `{result['compatibility_identity_sha256']}`",
        "- Accuracy is shown in percent; Δ is percentage points. WikiText-2 is raw perplexity.",
        "",
        header,
        separator,
    ]
    for row in result["table"]:
        cells = [
            row["task"],
            row["metric"],
            " / ".join(
                _display(row[key])
                for key in (
                    "source_standalone_paper",
                    "source_standalone_reproduced",
                    "source_standalone_delta",
                )
            ),
            " / ".join(
                _display(row[key])
                for key in (
                    "target_standalone_paper",
                    "target_standalone_reproduced",
                    "target_standalone_delta",
                )
            ),
            " / ".join(
                _display(row[key])
                for key in ("mapped_paper", "mapped_reproduced", "mapped_delta")
            ),
        ]
        if oracle:
            cells.append(_display(row["target_oracle_reproduced"]))
        cells.append(
            " / ".join(
                _display(row[key])
                for key in (
                    "paper_retention_percent",
                    "mapped_retention_percent",
                    "retention_delta_percentage_points",
                )
            )
        )
        cells.append(_display(row["chance_floor_percent"]))
        cells.append(
            " / ".join(
                _display(row[key])
                for key in (
                    "paper_floor_normalized_retention_percent",
                    "mapped_floor_normalized_retention_percent",
                    "floor_normalized_retention_delta_percentage_points",
                )
            )
        )
        lines.append("| " + " | ".join(cells) + " |")
    retention = result["five_accuracy_task_retention"]
    floor_normalized = result["five_accuracy_task_floor_normalized_retention"]
    lines.extend(
        [
            "",
            "Five-task mapped retention is the unweighted arithmetic mean over ARC-Challenge, "
            "HellaSwag, WinoGrande, MMLU, and GSM8K; WikiText-2 perplexity is excluded.",
            "",
            f"Paper: **{retention['paper_arithmetic_mean_percent']:.4f}%** "
            f"(reported rounded: 97.6%); reproduced: "
            f"**{retention['reproduced_arithmetic_mean_percent']:.4f}%**; "
            f"difference: **{retention['delta_from_paper_unrounded_percentage_points']:+.4f} pp**.",
            "",
            "Five-task chance-floor-normalized retention uses fixed floors of 25% for "
            "ARC-Challenge, HellaSwag, and MMLU; 50% for WinoGrande; and 0% for "
            "GSM8K. WikiText-2 perplexity is excluded.",
            "",
            f"Paper: **{floor_normalized['paper_arithmetic_mean_percent']:.4f}%** "
            f"(reported rounded: 96.3%); reproduced: "
            f"**{floor_normalized['reproduced_arithmetic_mean_percent']:.4f}%**; "
            f"difference: "
            f"**{floor_normalized['delta_from_paper_unrounded_percentage_points']:+.4f} pp**.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def write_aggregate_outputs(result: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations = {
        "aggregate.json": _strict_json_text(result),
        "paper_table.csv": render_csv(result),
        "paper_table.md": render_markdown(result),
    }
    for name, content in destinations.items():
        _atomic_write_text(output / name, content)
    generated = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": generated,
        "algorithm": "sha256",
        "files": {
            name: {
                "sha256": sha256_file(output / name),
                "nbytes": (output / name).stat().st_size,
            }
            for name in sorted(destinations)
        },
    }
    _atomic_write_text(output / "sha256_manifest.json", _strict_json_text(manifest))
    return manifest


__all__ = [
    "ACCURACY_TASKS",
    "AggregationError",
    "CHANCE_FLOORS",
    "EXPECTED_METRICS",
    "PAPER_REFERENCE",
    "PPL_TASK",
    "REQUIRED_MODES",
    "REQUIRED_TASKS",
    "aggregate_runs",
    "canonical_digest",
    "render_csv",
    "render_markdown",
    "validate_run",
    "write_aggregate_outputs",
]
