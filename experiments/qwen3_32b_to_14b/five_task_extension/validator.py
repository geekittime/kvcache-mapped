"""Fail-closed validation and final receipt for the five-task extension."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import yaml

from .protocol import (
    ACCURACY_TASKS,
    BASE_EVAL_CONFIG,
    CODE_IDENTITY_FILES,
    EXECUTED_TASKS,
    EXPECTED_FULL_SAMPLES,
    EXPERIMENT_ROOT,
    EXTENSION_CONFIG,
    EXTENSION_CODE_FREEZE,
    EXTENSION_ROOT,
    EXTENSION_VERSION,
    FORMAL_TOPOLOGY,
    GPU_GATES_DIRECTORY,
    HELLASWAG_ROOT,
    MODES,
    PAPER_BIT_EXACT,
    PAPER_SCOPE_NOTE,
    RECEIPT_NAME,
    REPOSITORY_ROOT,
    SMOKE_LIMIT,
    SMOKE_RECEIPT_NAME,
    SMOKE_WIKITEXT_CHUNKS,
    SUMMARY_MARKDOWN_NAME,
    SUMMARY_NAME,
    TASK_METRICS,
    output_within_experiment,
)

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reverse_kv.eval_common import (  # noqa: E402
    code_identity,
    hashed_json,
    verify_hashed_json,
    model_artifact_identity,
)
from reverse_kv.eval_adapter import _load_implementation  # noqa: E402
from five_task_extension.adapter import extension_project_code_identity  # noqa: E402
from reverse_kv.hellaswag_eval import (  # noqa: E402
    FORMAL_RECEIPT as HELLASWAG_RECEIPT,
    validate_hellaswag_result,
    validate_run_definition as validate_hellaswag_definition,
)
from reverse_kv.protocol import canonical_sha256, read_json, sha256_file  # noqa: E402
from kvcache_transfer.gsm8k_numeric_diagnostic import (  # noqa: E402
    score_checkpoint,
    verify_report_directory,
    write_report,
)


class ExtensionValidationError(ValueError):
    """The extension output is incomplete, ambiguous, or scientifically changed."""


_HEX64 = re.compile(r"[0-9a-f]{64}")
_SAMPLE_IDENTITY_FIELDS = (
    "doc_id",
    "doc_hash",
    "prompt_hash",
    "target_hash",
    "doc",
    "target",
    "arguments",
)
_REVERSE_SCIENTIFIC_FILES = (
    "reverse_kv/eval_adapter.py",
    "reverse_kv/eval_common.py",
    "reverse_kv/runtime.py",
    "reverse_kv/protocol.py",
    "reverse_kv/provenance.py",
    "reverse_kv/hellaswag_eval.py",
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtensionValidationError(f"{label} must be an object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExtensionValidationError(f"{label} must be a list")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtensionValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ExtensionValidationError(f"{label} must be finite")
    return result


def _artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExtensionValidationError(f"required artifact is missing/symlink: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
    }


def validate_extension_config() -> dict[str, Any]:
    value = yaml.safe_load(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ExtensionValidationError("extension config must be an object")
    fixed = {
        "schema_version": 1,
        "extension_version": EXTENSION_VERSION,
        "direction": "qwen3_32b_to_14b",
        "paper_bit_exact": False,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ExtensionValidationError(f"extension config changed {field}")
    protocol = _mapping(value.get("protocol"), "extension protocol")
    if (
        protocol.get("completion_mode") is not True
        or protocol.get("apply_chat_template") is not False
        or protocol.get("batch_size") != 1
        or protocol.get("max_gen_tokens") != 1024
        or protocol.get("mapper_k") != 20
        or float(protocol.get("ridge_lambda", -1)) != 0.01
        or tuple(protocol.get("modes", ())) != MODES
        or tuple(protocol.get("freshly_executed_tasks", ())) != EXECUTED_TASKS
        or protocol.get("bridged_task") != "hellaswag"
    ):
        raise ExtensionValidationError("extension protocol config changed")
    return dict(value)


def validate_extension_code_freeze(
    freeze_path: str | Path = EXTENSION_CODE_FREEZE,
    *,
    deep_models: bool = False,
    deep_mapper: bool = True,
) -> dict[str, Any]:
    """Verify the independent overlay, including current mapper/model identity.

    ``deep_models`` rereads every 32B and 14B shard byte. Formal preparation
    and GPU entry enable it; unit tests may rely on the already locked full
    shard hashes and the evaluator repeats the same deep hash at run start.
    """

    from five_task_extension.freeze import (  # local import avoids a module cycle
        _base_amendment,
        _base_shared_paths,
        _benchmark_paths,
        _extension_paths,
        _harness_paths,
        _reverse_paths,
    )

    path = Path(freeze_path).resolve()
    sidecar = path.with_name(f"{path.name}.sha256.json")
    sidecar_value = read_json(sidecar)
    if sidecar_value != {
        "schema_version": 1,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
    }:
        raise ExtensionValidationError("extension freeze SHA sidecar changed")
    value = read_json(path)
    unsigned = dict(value)
    digest = unsigned.pop("inventory_digest", None)
    if (
        value.get("schema_version") != 1
        or value.get("status") != "frozen"
        or value.get("kind") != EXTENSION_VERSION
        or value.get("paper_bit_exact") is not False
        or digest != canonical_sha256(unsigned)
    ):
        raise ExtensionValidationError("extension freeze header/inventory digest changed")

    base_path = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
    base = read_json(base_path)
    expected_paths = {
        "extension_files": _extension_paths(),
        "current_reverse_dependencies": _reverse_paths(),
        "current_shared_dependencies": _base_shared_paths(base),
        "current_vendored_harness_dependencies": _harness_paths(),
        "benchmark_protocol_dependencies": _benchmark_paths(),
    }
    base_for_group = {
        "extension_files": EXPERIMENT_ROOT,
        "current_reverse_dependencies": EXPERIMENT_ROOT,
        "current_shared_dependencies": REPOSITORY_ROOT,
        "current_vendored_harness_dependencies": REPOSITORY_ROOT,
        "benchmark_protocol_dependencies": REPOSITORY_ROOT,
    }
    for group, paths in expected_paths.items():
        rows = _list(value.get(group), f"freeze {group}")
        if {str(Path(str(row.get("path", ""))).resolve()) for row in rows if isinstance(row, Mapping)} != {
            str(item.resolve()) for item in paths
        }:
            raise ExtensionValidationError(f"extension freeze inventory changed: {group}")
        for row_raw in rows:
            row = _mapping(row_raw, f"freeze {group} row")
            candidate = Path(str(row.get("path", ""))).resolve()
            try:
                relative = str(candidate.relative_to(base_for_group[group].resolve()))
            except ValueError as exc:
                raise ExtensionValidationError(f"freeze path escaped: {candidate}") from exc
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or row.get("relative_path") != relative
                or row.get("nbytes") != candidate.stat().st_size
                or row.get("sha256") != sha256_file(candidate)
            ):
                raise ExtensionValidationError(f"frozen dependency drifted: {candidate}")

    expected_amendment = _base_amendment(base_path, base)
    if value.get("base_freeze_amendment") != expected_amendment:
        raise ExtensionValidationError("base-freeze amendment/drift ledger changed")

    bridge_definition = read_json(HELLASWAG_ROOT / "run_definition.json")
    expected_models = {
        "identity_scope": "every model weight shard content SHA256",
        "source_32b": bridge_definition["runtime_artifacts"]["source_model"],
        "target_14b": bridge_definition["runtime_artifacts"]["target_model"],
    }
    if value.get("models") != expected_models:
        raise ExtensionValidationError("extension freeze model identities changed")
    for role in ("source_32b", "target_14b"):
        identity = _mapping(expected_models[role], f"model {role}")
        shards = _list(identity.get("weight_shards"), f"model {role} shards")
        if (
            identity.get("weight_shards_sha256") != canonical_sha256(shards)
            or identity.get("weight_shard_contents_hashed") is not True
            or len(shards) != identity.get("weight_shard_count")
            or any(not _HEX64.fullmatch(str(row.get("sha256", ""))) for row in shards if isinstance(row, Mapping))
        ):
            raise ExtensionValidationError(f"extension freeze lacks full {role} shard hashes")
        if deep_models and model_artifact_identity(identity["path"]) != identity:
            raise ExtensionValidationError(f"current model bytes differ from frozen {role}")

    mapper = _mapping(value.get("mapper_payload"), "mapper payload freeze")
    mapper_rows = _list(mapper.get("files"), "mapper payload files")
    if (
        mapper.get("every_mapper_payload_file_content_hashed") is not True
        or mapper.get("files_digest") != canonical_sha256(mapper_rows)
        or mapper.get("file_count") != len(mapper_rows)
        or mapper.get("total_nbytes") != sum(int(row.get("nbytes", -1)) for row in mapper_rows if isinstance(row, Mapping))
    ):
        raise ExtensionValidationError("mapper payload freeze summary changed")
    mapper_root = Path(str(mapper.get("root", ""))).resolve()
    mapper_manifest = read_json(mapper_root / "manifest.json")
    expected_mapper_paths = {mapper_root / "manifest.json", mapper_root / "manifest.sha256.json"}
    for layer in mapper_manifest["layers"]:
        expected_mapper_paths.update((Path(str(layer["path"])), Path(str(layer["sidecar_path"]))))
    expected_mapper_paths.update(path for path in (mapper_root / "inputs").rglob("*") if path.is_file())
    if {Path(str(row.get("path", ""))).resolve() for row in mapper_rows if isinstance(row, Mapping)} != {
        item.resolve() for item in expected_mapper_paths
    }:
        raise ExtensionValidationError("mapper payload file inventory changed")
    for row_raw in mapper_rows:
        row = _mapping(row_raw, "mapper payload row")
        candidate = Path(str(row.get("path", ""))).resolve()
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or row.get("nbytes") != candidate.stat().st_size
            or (deep_mapper and row.get("sha256") != sha256_file(candidate))
        ):
            raise ExtensionValidationError(f"mapper payload bytes changed: {candidate}")
    return {
        "artifact": _artifact(path),
        "sha256_sidecar": _artifact(sidecar),
        "inventory_digest": digest,
        "deep_models_verified": deep_models,
        "deep_mapper_verified": deep_mapper,
        "model_weight_shards_content_hashed": True,
        "mapper_payload_file_count": len(mapper_rows),
    }


def validate_gpu_idle_gate(path: Path) -> dict[str, Any]:
    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "status": "passed",
        "required_consecutive_samples": 12,
        "memory_limit_mib": 64,
        "final_consecutive_samples": 12,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ExtensionValidationError(f"GPU idle gate changed {field}")
    samples = _list(value.get("samples"), "GPU idle samples")
    if len(samples) < 12:
        raise ExtensionValidationError("GPU idle gate lacks twelve samples")
    for sample_raw in samples[-12:]:
        sample = _mapping(sample_raw, "GPU idle sample")
        memory = _list(sample.get("memory"), "GPU idle memory")
        if (
            sample.get("passed") is not True
            or sample.get("compute_processes") != []
            or len(memory) != 8
            or {int(_mapping(row, "GPU memory row").get("physical_index", -1)) for row in memory}
            != set(range(8))
            or any(int(_mapping(row, "GPU memory row").get("memory_used_mib", 65)) > 64 for row in memory)
        ):
            raise ExtensionValidationError("GPU idle gate final samples are invalid")
    sidecar = path.with_name(f"{path.name}.sha256.json")
    return {"artifact": _artifact(path), "sha256_sidecar": _artifact(sidecar)}


def validate_gpu_idle_gates(output: Path) -> list[dict[str, Any]]:
    directory = output / GPU_GATES_DIRECTORY
    if not directory.is_dir() or directory.is_symlink():
        raise ExtensionValidationError("extension segment-gate directory is missing")
    paths = sorted(directory.glob("segment_*_gpu_idle_gate.json"))
    expected_names = [f"segment_{index:04d}_gpu_idle_gate.json" for index in range(1, len(paths) + 1)]
    if not paths or [path.name for path in paths] != expected_names:
        raise ExtensionValidationError("extension segment gates are incomplete/reordered")
    actual_names = {path.name for path in directory.iterdir()}
    expected_all = set(expected_names) | {f"{name}.sha256.json" for name in expected_names}
    if actual_names != expected_all:
        raise ExtensionValidationError("extension segment-gate directory has unexpected files")
    return [
        {"segment_index": index, **validate_gpu_idle_gate(path)}
        for index, path in enumerate(paths, start=1)
    ]


def _fingerprint(definition: Mapping[str, Any]) -> str:
    declared = definition.get("request_fingerprint")
    unsigned = dict(definition)
    unsigned.pop("request_fingerprint", None)
    if not isinstance(declared, str) or not _HEX64.fullmatch(declared):
        raise ExtensionValidationError("request fingerprint is malformed")
    if canonical_sha256(unsigned) != declared:
        raise ExtensionValidationError("request fingerprint disagrees with canonical definition")
    return declared


def current_shared_project_code() -> dict[str, Any]:
    """Reproduce ``eval_adapter.run_evaluation`` project-code provenance."""

    implementation = _load_implementation()
    evidence = dict(implementation.project_code_provenance(REPOSITORY_ROOT))
    reverse_files: dict[str, str] = {}
    for path in sorted((EXPERIMENT_ROOT / "reverse_kv").glob("*.py")):
        reverse_files[str(path.relative_to(EXPERIMENT_ROOT))] = sha256_file(path)
    for path in sorted((EXPERIMENT_ROOT / "scripts").glob("*.py")):
        reverse_files[str(path.relative_to(EXPERIMENT_ROOT))] = sha256_file(path)
    evidence["reverse_direction_code"] = reverse_files
    evidence["reverse_direction"] = "qwen3_32b_to_14b"
    freeze = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
    evidence["reverse_code_freeze"] = {
        "path": str(freeze.resolve()),
        "sha256": sha256_file(freeze),
        "nbytes": freeze.stat().st_size,
    }
    return evidence


def _validate_current_benchmark_payload(definition: Mapping[str, Any]) -> None:
    """Bind every recorded task YAML and dataset parquet to current bytes."""

    benchmark = _mapping(definition.get("benchmark_data"), "benchmark_data")
    manifest_path = Path(str(benchmark.get("manifest_path", ""))).resolve()
    if manifest_path != (REPOSITORY_ROOT / "data/benchmarks/manifest.json").resolve():
        raise ExtensionValidationError("benchmark manifest path changed")
    manifest = read_json(manifest_path)
    manifest_datasets = _mapping(manifest.get("datasets"), "benchmark manifest datasets")
    datasets = _mapping(benchmark.get("datasets"), "benchmark_data.datasets")
    required = set(EXECUTED_TASKS) | {"hellaswag"}
    if not required <= set(datasets):
        raise ExtensionValidationError("benchmark definition lacks extension datasets")
    for task in sorted(required):
        declared_files = _list(_mapping(datasets[task], task).get("files"), f"{task}.files")
        live_rows = _list(
            _mapping(manifest_datasets.get(task), f"manifest.{task}").get("files"),
            f"manifest.{task}.files",
        )
        live = {str(row.get("relative_path")): row for row in live_rows if isinstance(row, Mapping)}
        for row_raw in declared_files:
            row = _mapping(row_raw, f"{task} dataset row")
            relative = str(row.get("relative_path", ""))
            observed = _mapping(live.get(relative), f"live {task}/{relative}")
            if observed.get("sha256") != row.get("sha256") or observed.get("rows") != row.get("rows"):
                raise ExtensionValidationError(f"dataset manifest identity changed: {task}/{relative}")
            path = Path(str(observed.get("path", "")))
            if (
                not path.is_file()
                or path.stat().st_size != observed.get("nbytes")
                or sha256_file(path) != row.get("sha256")
            ):
                raise ExtensionValidationError(f"dataset bytes changed: {task}/{relative}")
    local_tasks = _mapping(benchmark.get("local_task_files"), "local_task_files")
    for relative, identity_raw in local_tasks.items():
        identity = _mapping(identity_raw, f"task file {relative}")
        path = REPOSITORY_ROOT / "data/benchmarks/lm_eval_tasks" / str(relative)
        if not path.is_file() or sha256_file(path) != identity.get("sha256"):
            raise ExtensionValidationError(f"local task YAML changed: {relative}")


def validate_hellaswag_bridge() -> dict[str, Any]:
    """Revalidate the exact reverse formal HellaSwag run before bridging it."""

    definition_path = HELLASWAG_ROOT / "run_definition.json"
    definition = read_json(definition_path)
    fingerprint = validate_hellaswag_definition(definition)
    rows = [
        validate_hellaswag_result(
            HELLASWAG_ROOT / "results" / mode / "hellaswag.json",
            fingerprint=fingerprint,
            expected_mode=mode,
        )
        for mode in MODES
    ]
    for row in rows:
        checkpoint = read_json(Path(row["path"]))
        recomputed = _recompute_multiple_choice_rows(
            _sample_rows(_mapping(checkpoint.get("result"), "Hella result"), "hellaswag"),
            task="hellaswag",
        )
        if (
            recomputed["samples"] != row["samples"]
            or recomputed["acc_norm_correct"] != row["correct"]
            or not math.isclose(
                recomputed["acc_norm"], row["acc_norm"], rel_tol=0.0, abs_tol=1e-15
            )
        ):
            raise ExtensionValidationError("Hella bridge raw responses do not reproduce acc_norm")
    receipt_path = HELLASWAG_ROOT / HELLASWAG_RECEIPT
    verify_hashed_json(receipt_path)
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != "complete"
        or receipt.get("request_fingerprint") != fingerprint
        or receipt.get("sample_count_per_mode") != EXPECTED_FULL_SAMPLES["hellaswag"]
    ):
        raise ExtensionValidationError("formal HellaSwag receipt changed")
    recorded = {str(row.get("mode")): row for row in _list(receipt.get("results"), "Hella receipt results")}
    for row in rows:
        prior = _mapping(recorded.get(row["mode"]), f"Hella receipt {row['mode']}")
        for field in ("sha256", "nbytes", "correct", "samples", "acc_norm"):
            if prior.get(field) != row.get(field):
                raise ExtensionValidationError(f"Hella bridge disagrees on {field}")
    # Bridge compatibility is stricter than merely recording current code in
    # the extension overlay. Every shared evaluator file and every Hella task
    # source byte used in the old run must remain identical. Only unrelated
    # report/amendment modules are allowed to appear in the current overlay.
    current_code = current_shared_project_code()
    prior_code = _mapping(definition.get("project_code"), "Hella project code")
    if current_code.get("files") != prior_code.get("files"):
        raise ExtensionValidationError("shared scientific evaluator code changed since Hella")
    prior_reverse = _mapping(prior_code.get("reverse_direction_code"), "Hella reverse code")
    current_reverse = _mapping(current_code.get("reverse_direction_code"), "current reverse code")
    for relative in _REVERSE_SCIENTIFIC_FILES:
        if current_reverse.get(relative) != prior_reverse.get(relative):
            raise ExtensionValidationError(f"reverse scientific code changed since Hella: {relative}")
    harness = _mapping(definition.get("harness"), "Hella harness")
    harness_root = Path(str(harness.get("path", ""))).resolve()
    task_files = _mapping(harness.get("task_files"), "Hella task files")
    for relative, identity_raw in task_files.items():
        identity = _mapping(identity_raw, f"Hella task file {relative}")
        candidate = harness_root / str(relative)
        if (
            not candidate.is_file()
            or candidate.stat().st_size != identity.get("nbytes")
            or sha256_file(candidate) != identity.get("sha256")
        ):
            raise ExtensionValidationError(f"Hella harness task byte changed: {relative}")
    return {
        "reuse_decision": "strict_bridge_no_rerun",
        "scientific_reason": (
            "The completed reverse run has the identical model/mapper/topology, "
            "completion/no-chat/batch-1 protocol, zero-shot acc_norm task, full "
            "10,042-row validation payload, and persisted raw samples."
        ),
        "request_fingerprint": fingerprint,
        "definition": _artifact(definition_path),
        "receipt": _artifact(receipt_path),
        "receipt_sha_sidecar": _artifact(receipt_path.with_name(f"{receipt_path.name}.sha256.json")),
        "results": rows,
        "runtime_artifacts": definition.get("runtime_artifacts"),
        "project_code": definition.get("project_code"),
        "benchmark_data": definition.get("benchmark_data"),
    }


def validate_run_definition(output: str | Path, *, profile: str = "full") -> tuple[dict[str, Any], str]:
    validate_extension_config()
    validate_extension_code_freeze(deep_models=False, deep_mapper=True)
    root = output_within_experiment(output)
    definition = read_json(root / "run_definition.json")
    fingerprint = _fingerprint(definition)
    limit = None if profile == "full" else float(SMOKE_LIMIT)
    wiki_limit = None if profile == "full" else SMOKE_WIKITEXT_CHUNKS
    fixed = {
        "schema_version": 1,
        "config_path": str(BASE_EVAL_CONFIG.resolve()),
        "config_sha256": sha256_file(BASE_EVAL_CONFIG),
        "modes": list(MODES),
        "tasks": list(EXECUTED_TASKS),
        "limit": limit,
        "wikitext_max_chunks": wiki_limit,
        "log_samples": True,
        "completion_mode": True,
        "apply_chat_template": False,
        "batch_size": 1,
        "boundary_replay": True,
        "source_model": FORMAL_TOPOLOGY["source"]["model"],
        "target_model": FORMAL_TOPOLOGY["target"]["model"],
        "source_devices": FORMAL_TOPOLOGY["source"]["devices"],
        "target_devices": FORMAL_TOPOLOGY["target"]["devices"],
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "max_gen_tokens": 1024,
    }
    for field, expected in fixed.items():
        if definition.get(field) != expected:
            raise ExtensionValidationError(f"run definition changed {field}")
    benchmarks = _mapping(definition.get("benchmarks"), "benchmarks")
    expected_benchmarks = {
        "arc_challenge": (0, "acc_norm"),
        "hellaswag": (0, "acc_norm"),
        "winogrande": (0, "acc"),
        "mmlu": (5, "acc"),
        "gsm8k_cot": (8, "exact_match"),
    }
    for task, (shots, metric) in expected_benchmarks.items():
        row = _mapping(benchmarks.get(task), f"benchmarks.{task}")
        if row.get("num_fewshot") != shots or row.get("metric") != metric:
            raise ExtensionValidationError(f"benchmark contract changed: {task}")
    gsm = _mapping(benchmarks["gsm8k_cot"], "GSM benchmark")
    if gsm.get("metric_filter") != "strict-match" or gsm.get("mode") != "greedy_generation_cot":
        raise ExtensionValidationError("GSM8K generation/scoring contract changed")
    wiki = _mapping(benchmarks.get("wikitext2_prefix_ppl"), "WikiText benchmark")
    if [wiki.get(k) for k in ("chunk_length", "prefix_length", "scored_length")] != [2048, 1024, 1024]:
        raise ExtensionValidationError("WikiText 1024-prefix/1024-score contract changed")

    bridge = validate_hellaswag_bridge()
    for field in ("runtime_artifacts", "benchmark_data"):
        if definition.get(field) != bridge[field]:
            raise ExtensionValidationError(
                f"extension {field} differs from the completed reverse formal identity"
            )
    current_code = extension_project_code_identity()
    if definition.get("project_code") != current_code:
        raise ExtensionValidationError("extension project-code identity differs from current evaluator bytes")
    # The adapter intentionally records every reverse module, including report
    # builders irrelevant to this evaluation. Those may have completed later
    # in the paper pipeline. The scientific runtime/validator subset must be
    # byte-identical to the already executed Hella segment before bridging it.
    prior_reverse = _mapping(bridge["project_code"].get("reverse_direction_code"), "bridge reverse code")
    current_reverse = _mapping(current_code.get("reverse_direction_code"), "current reverse code")
    for relative in _REVERSE_SCIENTIFIC_FILES:
        if current_reverse.get(relative) != prior_reverse.get(relative):
            raise ExtensionValidationError(f"reverse scientific code changed since Hella: {relative}")
    _validate_current_benchmark_payload(definition)
    return dict(definition), fingerprint


def _validate_request_cache(
    checkpoint: Mapping[str, Any], output: Path, *, fingerprint: str, mode: str, task: str
) -> None:
    cache = _mapping(checkpoint.get("request_cache"), "request_cache")
    if task == "wikitext2_prefix_ppl":
        if cache != {
            "enabled": False,
            "reason": "custom_wikitext_prefix_protocol_not_dispatched_through_lm_eval",
        }:
            raise ExtensionValidationError("WikiText request-cache contract changed")
        return
    if (
        cache.get("enabled") is not True
        or cache.get("isolation") != "request_fingerprint/mode/task"
        or cache.get("generation_requirement") != "deterministic_do_sample_false"
    ):
        raise ExtensionValidationError(f"request cache contract changed: {mode}/{task}")
    database = Path(str(cache.get("rank0_database", ""))).resolve()
    expected_parent = output / "request_cache" / fingerprint / mode / task
    if expected_parent.resolve() not in database.parents or not database.is_file() or database.stat().st_size <= 0:
        raise ExtensionValidationError(f"request cache database is missing/outside run: {mode}/{task}")


def _sample_rows(result: Mapping[str, Any], task: str) -> list[tuple[str, Mapping[str, Any]]]:
    samples = _mapping(result.get("samples"), "samples")
    if task != "mmlu":
        return [(task, _mapping(row, f"{task} sample")) for row in _list(samples.get(task), task)]
    rows: list[tuple[str, Mapping[str, Any]]] = []
    keys = sorted(key for key in samples if key.startswith("mmlu_") and key != "mmlu")
    if len(keys) != 57:
        raise ExtensionValidationError("MMLU must contain exactly 57 subject sample groups")
    for key in keys:
        rows.extend((key, _mapping(row, f"{key} sample")) for row in _list(samples[key], key))
    return rows


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordered_doc(task: str, doc: Mapping[str, Any]) -> dict[str, Any]:
    if task == "arc_challenge":
        choices = _mapping(doc.get("choices"), "ARC choices")
        return {
            "id": doc["id"],
            "question": doc["question"],
            "choices": {"text": choices["text"], "label": choices["label"]},
            "answerKey": doc["answerKey"],
        }
    if task == "winogrande":
        return {key: doc[key] for key in ("sentence", "option1", "option2", "answer")}
    if task == "mmlu":
        return {key: doc[key] for key in ("question", "subject", "choices", "answer")}
    if task == "hellaswag":
        return {
            key: doc[key]
            for key in (
                "ind",
                "activity_label",
                "ctx_a",
                "ctx_b",
                "ctx",
                "endings",
                "source_id",
                "split",
                "split_type",
                "label",
                "query",
                "choices",
                "gold",
            )
        }
    raise ExtensionValidationError(f"unsupported multiple-choice task: {task}")


def _recompute_multiple_choice_rows(
    rows: Sequence[tuple[str, Mapping[str, Any]]], *, task: str
) -> dict[str, Any]:
    acc_correct = 0
    acc_norm_correct = 0
    for index, (_, sample) in enumerate(rows):
        doc = _mapping(sample.get("doc"), f"{task} sample doc")
        arguments = _list(sample.get("arguments"), f"{task} sample arguments")
        responses = _list(sample.get("resps"), f"{task} sample resps")
        filtered = _list(sample.get("filtered_resps"), f"{task} sample filtered resps")
        if not arguments or len(arguments) != len(responses) or len(filtered) != len(arguments):
            raise ExtensionValidationError(f"{task} option/response cardinality changed")
        loglikelihoods: list[float] = []
        continuations: list[str] = []
        for option, (argument_raw, response_raw, filtered_raw) in enumerate(
            zip(arguments, responses, filtered, strict=True)
        ):
            argument = _list(argument_raw, f"{task} argument {option}")
            repeats = _list(response_raw, f"{task} response {option}")
            selected = _list(filtered_raw, f"{task} filtered response {option}")
            if (
                len(argument) != 2
                or not all(isinstance(item, str) for item in argument)
                or len(repeats) != 1
                or _list(repeats[0], f"{task} response repeat") != selected
                or len(selected) != 2
                or not isinstance(selected[1], bool)
            ):
                raise ExtensionValidationError(f"{task} raw option response changed")
            loglikelihoods.append(_finite(selected[0], f"{task} loglikelihood"))
            continuations.append(str(argument[1]))
        predicted = max(range(len(loglikelihoods)), key=loglikelihoods.__getitem__)
        if task == "arc_challenge":
            labels = list(_mapping(doc["choices"], "ARC choices")["label"])
            gold = labels.index(str(doc["answerKey"]))
            normalization_choices = list(_mapping(doc["choices"], "ARC choices")["text"])
            if sample.get("target") != gold:
                raise ExtensionValidationError("ARC target disagrees with answerKey")
        elif task == "winogrande":
            gold = int(doc["answer"]) - 1
            normalization_choices = continuations
            if sample.get("target") != continuations[0].lstrip() or len(set(continuations)) != 1:
                raise ExtensionValidationError("WinoGrande target/continuations changed")
        elif task == "mmlu":
            gold = int(doc["answer"])
            normalization_choices = list(doc["choices"])
            if sample.get("target") != gold:
                raise ExtensionValidationError("MMLU target disagrees with answer")
        else:
            gold = int(doc["gold"])
            normalization_choices = list(doc["choices"])
            if sample.get("target") != gold:
                raise ExtensionValidationError("HellaSwag target disagrees with gold")
        if not (0 <= gold < len(loglikelihoods)) or len(normalization_choices) != len(loglikelihoods):
            raise ExtensionValidationError(f"{task} gold/choice cardinality changed")
        lengths = [len(str(choice)) for choice in normalization_choices]
        if any(length <= 0 for length in lengths):
            raise ExtensionValidationError(f"{task} has an empty normalization choice")
        predicted_norm = max(
            range(len(loglikelihoods)),
            key=lambda choice: loglikelihoods[choice] / lengths[choice],
        )
        acc = int(predicted == gold)
        acc_norm = int(predicted_norm == gold)
        if "acc" in sample and _finite(sample.get("acc"), f"{task} persisted acc") != acc:
            raise ExtensionValidationError(f"{task} persisted acc differs from raw responses")
        if "acc_norm" in sample and _finite(
            sample.get("acc_norm"), f"{task} persisted acc_norm"
        ) != acc_norm:
            raise ExtensionValidationError(f"{task} persisted acc_norm differs from raw responses")
        first_prompt = str(_list(arguments[0], "first argument")[0])
        if sample.get("prompt_hash") != _sha256_text(first_prompt):
            raise ExtensionValidationError(f"{task} prompt hash differs from arguments")
        if sample.get("target_hash") != _sha256_text(str(sample.get("target"))):
            raise ExtensionValidationError(f"{task} target hash differs from target")
        ordered = _ordered_doc(task, doc)
        doc_payload = json.dumps(ordered, indent=2, ensure_ascii=False)
        if sample.get("doc_hash") != _sha256_text(doc_payload):
            raise ExtensionValidationError(f"{task} doc hash differs from dataset-order document")
        acc_correct += acc
        acc_norm_correct += acc_norm
    total = len(rows)
    if total <= 0:
        raise ExtensionValidationError(f"{task} has no multiple-choice rows")
    return {
        "samples": total,
        "acc_correct": acc_correct,
        "acc": acc_correct / total,
        "acc_norm_correct": acc_norm_correct,
        "acc_norm": acc_norm_correct / total,
    }


def _expected_task_samples(task: str, profile: str) -> int:
    if profile == "full":
        return EXPECTED_FULL_SAMPLES[task]
    if task == "mmlu":
        return 57 * SMOKE_LIMIT
    return SMOKE_LIMIT


def _validate_accuracy_checkpoint(
    path: Path, *, definition: Mapping[str, Any], fingerprint: str, mode: str, task: str, profile: str
) -> dict[str, Any]:
    value = read_json(path)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "mode": mode,
        "task": task,
        "primary_metric": TASK_METRICS[task],
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ExtensionValidationError(f"checkpoint changed {field}: {mode}/{task}")
    _validate_request_cache(value, path.parents[2], fingerprint=fingerprint, mode=mode, task=task)
    result = _mapping(value.get("result"), "result")
    resolved = _mapping(value.get("resolved_task_config"), "resolved_task_config")
    configs = _mapping(resolved.get("configs"), "resolved task configs")
    if (
        resolved.get("canonical_configs_sha256") != canonical_sha256(configs)
        or configs != _mapping(result.get("configs"), "result configs")
        or resolved.get("n_shot") != result.get("n-shot")
        or resolved.get("versions") != result.get("versions")
    ):
        raise ExtensionValidationError(f"resolved task configuration changed: {mode}/{task}")
    invocation = _mapping(result.get("config"), "lm-eval invocation")
    expected_limit = None if profile == "full" else float(SMOKE_LIMIT)
    if (
        invocation.get("limit") != expected_limit
        or invocation.get("batch_size") != 1
        or invocation.get("apply_chat_template") is not False
    ):
        raise ExtensionValidationError(f"lm-eval invocation changed: {mode}/{task}")
    n_shot = _mapping(result.get("n-shot"), "n-shot")
    expected_shots = {"arc_challenge": 0, "winogrande": 0, "mmlu": 5, "gsm8k_cot": 8}[task]
    if task == "mmlu":
        # lm-eval also persists aggregate MMLU group/category keys. The 57
        # leaf keys are cross-checked against the sample groups below; every
        # aggregate and leaf declaration must still say five-shot.
        if len(n_shot) < 57 or set(n_shot.values()) != {expected_shots}:
            raise ExtensionValidationError("MMLU n-shot contract changed")
    elif n_shot.get(task) != expected_shots:
        raise ExtensionValidationError(f"{task} n-shot contract changed")
    rows = _sample_rows(result, task)
    if task == "mmlu":
        leaf_keys = set(_mapping(result.get("samples"), "MMLU samples"))
        if len(leaf_keys) != 57 or not leaf_keys <= set(n_shot):
            raise ExtensionValidationError("MMLU leaf n-shot inventory changed")
    expected = _expected_task_samples(task, profile)
    metric = TASK_METRICS[task]

    if task == "gsm8k_cot":
        if len(rows) != 2 * expected:
            raise ExtensionValidationError("GSM raw row count is not twice the sample count")
        by_id: dict[int, dict[str, Mapping[str, Any]]] = {}
        for _, row in rows:
            doc_id = row.get("doc_id")
            filt = row.get("filter")
            if not isinstance(doc_id, int) or filt not in {"strict-match", "flexible-extract"}:
                raise ExtensionValidationError("GSM sample identity/filter is invalid")
            bucket = by_id.setdefault(doc_id, {})
            if str(filt) in bucket:
                raise ExtensionValidationError("GSM contains a duplicate filter row")
            bucket[str(filt)] = row
        if sorted(by_id) != list(range(expected)) or any(
            set(bucket) != {"strict-match", "flexible-extract"} for bucket in by_id.values()
        ):
            raise ExtensionValidationError("GSM must contain paired strict/flexible rows for every ID")
        strict_bits = [_finite(by_id[i]["strict-match"].get("exact_match"), "GSM strict bit") for i in range(expected)]
        if any(bit not in (0.0, 1.0) for bit in strict_bits):
            raise ExtensionValidationError("GSM strict metric is not binary")
        recomputed = sum(strict_bits) / expected
        row_count = len(rows)
    else:
        if len(rows) != expected:
            raise ExtensionValidationError(f"{task} sample count {len(rows)} != {expected}")
        grouped: dict[str, list[int]] = {}
        bits: list[float] = []
        for group, row in rows:
            doc_id = row.get("doc_id")
            if not isinstance(doc_id, int) or isinstance(doc_id, bool):
                raise ExtensionValidationError(f"{task} has invalid doc_id")
            grouped.setdefault(group, []).append(doc_id)
            if row.get("filter") != "none" or metric not in _list(row.get("metrics"), "sample metrics"):
                raise ExtensionValidationError(f"{task} sample metric/filter changed")
            bit = _finite(row.get(metric), f"{task} sample metric")
            if bit not in (0.0, 1.0):
                raise ExtensionValidationError(f"{task} sample metric is not binary")
            bits.append(bit)
            for field in ("doc_hash", "prompt_hash", "target_hash"):
                if not isinstance(row.get(field), str) or not _HEX64.fullmatch(str(row[field])):
                    raise ExtensionValidationError(f"{task} sample {field} is invalid")
        if any(ids != list(range(len(ids))) for ids in grouped.values()):
            raise ExtensionValidationError(f"{task} doc IDs are incomplete/duplicated/reordered")
        raw_recomputed = _recompute_multiple_choice_rows(rows, task=task)
        recomputed = float(raw_recomputed[metric])
        if not math.isclose(sum(bits) / expected, recomputed, rel_tol=0.0, abs_tol=0.0):
            raise ExtensionValidationError(f"{task} persisted bits differ from raw responses")
        row_count = len(rows)

    aggregates = _mapping(result.get("results"), "aggregate results")
    aggregate = _mapping(aggregates.get(task), f"aggregate {task}")
    embedded = _finite(aggregate.get(f"{metric},none" if task != "gsm8k_cot" else "exact_match,strict-match"), "aggregate metric")
    primary = _finite(value.get("primary_metric_value"), "primary metric")
    for label, observed in (("aggregate", embedded), ("primary", primary)):
        if not math.isclose(observed, recomputed, rel_tol=0.0, abs_tol=1e-15):
            raise ExtensionValidationError(f"{task} {label} differs from raw sample recomputation")
    counts = _mapping(result.get("n-samples"), "n-samples")
    if task == "mmlu":
        if len(counts) != 57:
            raise ExtensionValidationError("MMLU n-samples lacks 57 subjects")
        effective_total = 0
        original_total = 0
        for count_raw in counts.values():
            count = _mapping(count_raw, "MMLU subject count")
            effective_total += int(count.get("effective", -1))
            original_total += int(count.get("original", -1))
        if effective_total != expected or original_total != EXPECTED_FULL_SAMPLES["mmlu"]:
            raise ExtensionValidationError("MMLU effective/original counts changed")
    else:
        count = _mapping(counts.get(task), f"{task} n-samples")
        if count.get("effective") != expected or count.get("original") != EXPECTED_FULL_SAMPLES[task]:
            raise ExtensionValidationError(f"{task} effective/original counts changed")
    return {
        "mode": mode,
        "task": task,
        "samples": expected,
        "raw_rows": row_count,
        "correct": int(round(recomputed * expected)),
        "metric": metric,
        "value": recomputed,
        **_artifact(path),
    }


def _validate_wikitext_checkpoint(
    path: Path, *, fingerprint: str, mode: str, profile: str
) -> dict[str, Any]:
    value = read_json(path)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "mode": mode,
        "task": "wikitext2_prefix_ppl",
        "primary_metric": "perplexity",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ExtensionValidationError(f"WikiText checkpoint changed {field}")
    _validate_request_cache(
        value, path.parents[2], fingerprint=fingerprint, mode=mode, task="wikitext2_prefix_ppl"
    )
    result = _mapping(value.get("result"), "WikiText result")
    chunks = result.get("chunks")
    scored = result.get("scored_tokens")
    if not isinstance(chunks, int) or chunks <= 0 or scored != chunks * 1024:
        raise ExtensionValidationError("WikiText chunk/token count is invalid")
    expected_chunks = SMOKE_WIKITEXT_CHUNKS if profile == "smoke" else None
    dataset_path = path.parents[2] / "wikitext_dataset.json"
    dataset = read_json(dataset_path)
    if profile == "full":
        token_count = dataset.get("token_count")
        if not isinstance(token_count, int) or token_count <= 0:
            raise ExtensionValidationError("WikiText token count is invalid")
        expected_chunks = token_count // 2048
    if chunks != expected_chunks:
        raise ExtensionValidationError(f"WikiText chunks {chunks} != {expected_chunks}")
    loglikelihood = _finite(result.get("loglikelihood"), "WikiText loglikelihood")
    ppl = _finite(result.get("perplexity"), "WikiText perplexity")
    if loglikelihood >= 0 or not math.isclose(
        ppl, math.exp(-loglikelihood / scored), rel_tol=1e-14, abs_tol=1e-14
    ):
        raise ExtensionValidationError("WikiText perplexity differs from NLL/token recomputation")
    if not math.isclose(_finite(value.get("primary_metric_value"), "Wiki primary"), ppl, rel_tol=0, abs_tol=0):
        raise ExtensionValidationError("WikiText primary metric changed")
    return {
        "mode": mode,
        "task": "wikitext2_prefix_ppl",
        "chunks": chunks,
        "scored_tokens": scored,
        "loglikelihood": loglikelihood,
        "metric": "perplexity",
        "value": ppl,
        **_artifact(path),
    }


def _validate_cross_mode_identity(output: Path, task: str) -> None:
    target = read_json(output / "results/target_standalone" / f"{task}.json")
    mapped = read_json(output / "results/mapped" / f"{task}.json")
    target_rows = _sample_rows(_mapping(target.get("result"), "target result"), task)
    mapped_rows = _sample_rows(_mapping(mapped.get("result"), "mapped result"), task)
    if len(target_rows) != len(mapped_rows):
        raise ExtensionValidationError(f"cross-mode row count changed: {task}")
    for index, ((target_group, a), (mapped_group, b)) in enumerate(zip(target_rows, mapped_rows)):
        if target_group != mapped_group:
            raise ExtensionValidationError(f"cross-mode MMLU group changed at row {index}")
        for field in _SAMPLE_IDENTITY_FIELDS:
            if a.get(field) != b.get(field):
                raise ExtensionValidationError(f"cross-mode {task} identity changed {field} at row {index}")


def validate_raw_output(output: str | Path, *, profile: str = "full") -> dict[str, Any]:
    root = output_within_experiment(output)
    definition, fingerprint = validate_run_definition(root, profile=profile)
    status = read_json(root / "status.json")
    if status.get("status") != "complete" or status.get("failures") != 0:
        raise ExtensionValidationError("shared evaluator status is not complete/clean")
    rows: list[dict[str, Any]] = []
    for task in EXECUTED_TASKS:
        for mode in MODES:
            path = root / "results" / mode / f"{task}.json"
            if task == "wikitext2_prefix_ppl":
                rows.append(_validate_wikitext_checkpoint(path, fingerprint=fingerprint, mode=mode, profile=profile))
            else:
                rows.append(
                    _validate_accuracy_checkpoint(
                        path,
                        definition=definition,
                        fingerprint=fingerprint,
                        mode=mode,
                        task=task,
                        profile=profile,
                    )
                )
        if task != "wikitext2_prefix_ppl":
            _validate_cross_mode_identity(root, task)
    shared_summary_value = read_json(root / "summary.json")
    if (
        shared_summary_value.get("request_fingerprint") != fingerprint
        or status.get("summary") != shared_summary_value
    ):
        raise ExtensionValidationError("shared status/summary/fingerprint binding changed")
    shared_metrics = _mapping(shared_summary_value.get("metrics"), "shared summary metrics")
    for row in rows:
        mode_metrics = _mapping(shared_metrics.get(row["mode"]), f"shared metrics {row['mode']}")
        if mode_metrics.get(row["task"]) != row["value"]:
            raise ExtensionValidationError(
                f"shared summary differs from validated raw metric: {row['mode']}/{row['task']}"
            )
    tokenizer_path = root / "tokenizer.json"
    tokenizer = read_json(tokenizer_path)
    bridge_tokenizer = read_json(HELLASWAG_ROOT / "tokenizer.json")
    if tokenizer != bridge_tokenizer or tokenizer.get("vocabulary_size") != 151_669:
        raise ExtensionValidationError("extension tokenizer identity differs from reverse Hella")
    if not _HEX64.fullmatch(str(tokenizer.get("vocabulary_sha256", ""))):
        raise ExtensionValidationError("extension tokenizer vocabulary SHA is invalid")
    mapper_path = root / "mapper.json"
    mapper = read_json(mapper_path)
    mapper_fixed = {
        "direction": "qwen3_32b_to_14b",
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
        "fit_fingerprint": "0aa81cb5816b56fe19c0349db7c54496751be6aa1918b8aff895f1e824ac2443",
        "layers": 40,
        "manifest": str(
            (EXPERIMENT_ROOT / "artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01/manifest.json").resolve()
        ),
        "manifest_sha256": "382a1deb6f4fd33c824fe782d956f46bdfb2b876081cb36487616e4de92a8ee2",
    }
    for field, expected in mapper_fixed.items():
        if mapper.get(field) != expected:
            raise ExtensionValidationError(f"runtime mapper provenance changed {field}")
    materialization = _mapping(mapper.get("materialization"), "mapper materialization")
    per_layer = _list(materialization.get("per_layer"), "mapper materialization layers")
    expected_devices = ["cuda:5"] * 13 + ["cuda:6"] * 14 + ["cuda:7"] * 13
    if len(per_layer) != 40:
        raise ExtensionValidationError("runtime mapper did not materialize 40 layers")
    for layer, (row_raw, device) in enumerate(zip(per_layer, expected_devices, strict=True)):
        row = _mapping(row_raw, "mapper materialization layer")
        if row.get("target_layer") != layer or row.get("device") != device:
            raise ExtensionValidationError("runtime mapper layer/device placement changed")
        for timing in ("host_load_and_hash_ms", "weight_host_to_device_ms"):
            if _finite(row.get(timing), f"mapper {timing}") <= 0:
                raise ExtensionValidationError("runtime mapper materialization timing is invalid")
    return {
        "definition": definition,
        "fingerprint": fingerprint,
        "rows": rows,
        "status": _artifact(root / "status.json"),
        "summary": _artifact(root / "summary.json"),
        "wikitext_dataset": _artifact(root / "wikitext_dataset.json"),
        "run_definition": _artifact(root / "run_definition.json"),
        "tokenizer": _artifact(tokenizer_path),
        "mapper_runtime": _artifact(mapper_path),
    }


def _numeric_reports(output: Path, *, create: bool) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for mode in MODES:
        checkpoint = output / "results" / mode / "gsm8k_cot.json"
        report = score_checkpoint(checkpoint)
        destination = output / "gsm8k_numeric_final" / mode
        if create:
            write_report(report, destination)
        manifest = verify_report_directory(destination, expected_report=report)
        reports[mode] = {
            "strict": report["preserved_lm_eval_metrics"]["strict_match"],
            "flexible": report["preserved_lm_eval_metrics"]["flexible_extract"],
            "numeric_final": report["numeric_final_diagnostic"],
            "paper_disclosure": report["paper_disclosure"],
            "manifest": _artifact(destination / "sha256_manifest.json"),
            "manifest_value": manifest,
        }
    return reports


def _build_summary(raw: Mapping[str, Any], bridge: Mapping[str, Any], numeric: Mapping[str, Any]) -> dict[str, Any]:
    row_map = {(row["mode"], row["task"]): row for row in raw["rows"]}
    hella = {(row["mode"], "hellaswag"): row for row in bridge["results"]}
    comparison: dict[str, Any] = {}
    for task in ACCURACY_TASKS:
        if task == "hellaswag":
            target = float(hella[(MODES[0], task)]["acc_norm"])
            mapped = float(hella[(MODES[1], task)]["acc_norm"])
            metric = "acc_norm"
        elif task == "gsm8k_cot":
            target = float(numeric[MODES[0]]["numeric_final"]["accuracy"])
            mapped = float(numeric[MODES[1]]["numeric_final"]["accuracy"])
            metric = "numeric-final-result-level-diagnostic-v1"
        else:
            target = float(row_map[(MODES[0], task)]["value"])
            mapped = float(row_map[(MODES[1], task)]["value"])
            metric = TASK_METRICS[task]
        comparison[task] = {
            "reported_metric": metric,
            "target_14b": target,
            "mapped_32b_to_14b": mapped,
            "mapped_over_target_percent": mapped / target * 100.0 if target else None,
        }
        if task == "gsm8k_cot":
            comparison[task]["canonical_strict_match"] = {
                mode: numeric[mode]["strict"] for mode in MODES
            }
    wiki_target = float(row_map[(MODES[0], "wikitext2_prefix_ppl")]["value"])
    wiki_mapped = float(row_map[(MODES[1], "wikitext2_prefix_ppl")]["value"])
    return {
        "schema_version": 1,
        "status": "complete",
        "extension_version": EXTENSION_VERSION,
        "direction": "qwen3_32b_to_14b",
        "mapper": "k20_lambda0.01",
        "paper_bit_exact": PAPER_BIT_EXACT,
        "paper_scope_note": PAPER_SCOPE_NOTE,
        "protocol": {
            "completion_mode": True,
            "apply_chat_template": False,
            "batch_size": 1,
            "max_gen_tokens": 1024,
            "fresh_tasks": list(EXECUTED_TASKS),
            "hellaswag": "strictly bridged from completed reverse formal run",
            "gsm8k_headline": "gold-independent numeric-final diagnostic; strict preserved",
        },
        "request_fingerprint": raw["fingerprint"],
        "accuracy": comparison,
        "wikitext2_prefix_ppl": {
            "target_14b": wiki_target,
            "mapped_32b_to_14b": wiki_mapped,
            "mapped_minus_target": wiki_mapped - wiki_target,
            "mapped_over_target_percent": wiki_mapped / wiki_target * 100.0,
            "lower_is_better": True,
        },
    }


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Qwen3 32B → 14B five-task extension",
        "",
        f"> paper_bit_exact=false. {summary['paper_scope_note']}",
        "",
        "| Task | Metric | 14B target | 32B→14B mapped | retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for task in ACCURACY_TASKS:
        row = summary["accuracy"][task]
        lines.append(
            f"| {task} | {row['reported_metric']} | {100*row['target_14b']:.4f}% | "
            f"{100*row['mapped_32b_to_14b']:.4f}% | {row['mapped_over_target_percent']:.4f}% |"
        )
    wiki = summary["wikitext2_prefix_ppl"]
    lines.extend(
        [
            "",
            f"WikiText-2 prefix PPL (lower is better): target={wiki['target_14b']:.6f}, "
            f"mapped={wiki['mapped_32b_to_14b']:.6f}.",
            "",
            "GSM8K's table row uses the frozen gold-independent numeric-final diagnostic. "
            "The canonical lm-eval strict-match counts remain embedded in the JSON summary.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    # The receipt is the commit marker. A deterministic summary rewrite before
    # that marker is harmless and makes an interrupted finalization resumable.
    path.write_text(text, encoding="utf-8")


def finalize_full_output(output: str | Path) -> dict[str, Any]:
    root = output_within_experiment(output)
    receipt_path = root / RECEIPT_NAME
    if receipt_path.exists():
        return validate_completed_output(root)
    raw = validate_raw_output(root, profile="full")
    bridge = validate_hellaswag_bridge()
    numeric = _numeric_reports(root, create=True)
    summary = _build_summary(raw, bridge, numeric)
    hashed_json(root / SUMMARY_NAME, summary)
    _write_text(root / SUMMARY_MARKDOWN_NAME, _summary_markdown(summary))
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "kind": EXTENSION_VERSION,
        "paper_bit_exact": PAPER_BIT_EXACT,
        "paper_scope_note": PAPER_SCOPE_NOTE,
        "request_fingerprint": raw["fingerprint"],
        "runtime_identity_source": "identical full-content-hashed reverse HellaSwag definition",
        "fresh_result_rows": raw["rows"],
        "hellaswag_bridge": bridge,
        "gsm8k_numeric_reports": numeric,
        "code_identity": code_identity(CODE_IDENTITY_FILES, root=EXPERIMENT_ROOT),
        "extension_config": _artifact(EXTENSION_CONFIG),
        "extension_code_freeze": validate_extension_code_freeze(deep_models=True, deep_mapper=True),
        "gpu_idle_gates": validate_gpu_idle_gates(root),
        "summary": _artifact(root / SUMMARY_NAME),
        "summary_sha_sidecar": _artifact(root / f"{SUMMARY_NAME}.sha256.json"),
        "summary_markdown": _artifact(root / SUMMARY_MARKDOWN_NAME),
        "shared_status": raw["status"],
        "shared_summary": raw["summary"],
        "wikitext_dataset": raw["wikitext_dataset"],
        "run_definition": raw["run_definition"],
        "tokenizer": raw["tokenizer"],
        "mapper_runtime": raw["mapper_runtime"],
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    hashed_json(receipt_path, receipt)
    return validate_completed_output(root)


def validate_completed_output(output: str | Path) -> dict[str, Any]:
    root = output_within_experiment(output)
    verify_hashed_json(root / RECEIPT_NAME)
    receipt = read_json(root / RECEIPT_NAME)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "complete"
        or receipt.get("kind") != EXTENSION_VERSION
        or receipt.get("paper_bit_exact") is not False
    ):
        raise ExtensionValidationError("extension receipt header changed")
    raw = validate_raw_output(root, profile="full")
    bridge = validate_hellaswag_bridge()
    numeric = _numeric_reports(root, create=False)
    summary = _build_summary(raw, bridge, numeric)
    verify_hashed_json(root / SUMMARY_NAME)
    if read_json(root / SUMMARY_NAME) != summary:
        raise ExtensionValidationError("extension summary differs from full raw recomputation")
    if (root / SUMMARY_MARKDOWN_NAME).read_text(encoding="utf-8") != _summary_markdown(summary):
        raise ExtensionValidationError("extension markdown differs from summary")
    expected_code = code_identity(CODE_IDENTITY_FILES, root=EXPERIMENT_ROOT)
    checks = {
        "request_fingerprint": raw["fingerprint"],
        "fresh_result_rows": raw["rows"],
        "hellaswag_bridge": bridge,
        "gsm8k_numeric_reports": numeric,
        "code_identity": expected_code,
        "extension_config": _artifact(EXTENSION_CONFIG),
        "extension_code_freeze": validate_extension_code_freeze(deep_models=True, deep_mapper=True),
        "gpu_idle_gates": validate_gpu_idle_gates(root),
        "summary": _artifact(root / SUMMARY_NAME),
        "summary_sha_sidecar": _artifact(root / f"{SUMMARY_NAME}.sha256.json"),
        "summary_markdown": _artifact(root / SUMMARY_MARKDOWN_NAME),
        "shared_status": raw["status"],
        "shared_summary": raw["summary"],
        "wikitext_dataset": raw["wikitext_dataset"],
        "run_definition": raw["run_definition"],
        "tokenizer": raw["tokenizer"],
        "mapper_runtime": raw["mapper_runtime"],
    }
    for field, expected in checks.items():
        if receipt.get(field) != expected:
            raise ExtensionValidationError(f"extension receipt changed {field}")
    return dict(receipt)


def finalize_smoke_output(output: str | Path) -> dict[str, Any]:
    root = output_within_experiment(output)
    raw = validate_raw_output(root, profile="smoke")
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "kind": f"{EXTENSION_VERSION}-smoke",
        "paper_bit_exact": False,
        "not_a_formal_result": True,
        "limit_per_lm_eval_leaf_task": SMOKE_LIMIT,
        "wikitext_max_chunks": SMOKE_WIKITEXT_CHUNKS,
        "request_fingerprint": raw["fingerprint"],
        "rows": raw["rows"],
        "code_identity": code_identity(CODE_IDENTITY_FILES, root=EXPERIMENT_ROOT),
        "extension_code_freeze": validate_extension_code_freeze(deep_models=True, deep_mapper=True),
        "gpu_idle_gates": validate_gpu_idle_gates(root),
    }
    hashed_json(root / SMOKE_RECEIPT_NAME, receipt)
    return receipt


__all__ = [
    "ExtensionValidationError",
    "finalize_full_output",
    "finalize_smoke_output",
    "validate_completed_output",
    "validate_extension_config",
    "validate_extension_code_freeze",
    "validate_gpu_idle_gate",
    "validate_gpu_idle_gates",
    "validate_hellaswag_bridge",
    "validate_raw_output",
    "validate_run_definition",
]
