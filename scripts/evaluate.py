#!/usr/bin/env python3
"""Run the paper evaluation for source/target full, oracle, and mapped paths.

Every ``(mode, task)`` result is an atomic checkpoint.  Re-running with the
same output directory skips validated completed checkpoints and retries only
missing/failed work.  The script never applies a chat template and fixes
batch size to one.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.config import load_yaml  # noqa: E402
from kvcache_transfer.eval_data import (  # noqa: E402
    local_wikitext_file,
    manifest_digest_payload,
    verify_ready_manifest,
)
from kvcache_transfer.evaluation import (  # noqa: E402
    EVALUATION_MODES,
    MappedPrefixCacheBuilder,
    TransferEvaluationEngine,
    concatenate_document_tokens,
    evaluate_prefix_perplexity,
    make_lm_eval_adapter,
)
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.modeling import load_qwen3  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/eval_qwen3_14b_to_32b.yaml"
DEFAULT_BENCHMARK_MANIFEST = ROOT / "data/benchmarks/manifest.json"
DEFAULT_LOCAL_TASK_DIR = ROOT / "data/benchmarks/lm_eval_tasks"
SCHEMA_VERSION = 1

# lm-eval's response cache is safe for this reproduction only because every
# database is scoped below the immutable run fingerprint *and* the evaluation
# mode.  Upstream hashes only ``(request_type, request.args)``; sharing one DB
# between target/mapped/source would therefore silently return another mode's
# answers for identical prompts.
REQUEST_CACHE_POLICY = {
    "enabled_for_lm_eval_tasks": True,
    "backend": "lm_eval.CachingLM_sqlitedict",
    "upstream_cache_key": "sha256(json([request_type, *request.args]))",
    "isolation": "request_fingerprint/mode/task",
    "partial_durability": "adapter_cache_hook_autocommit_after_each_completed_request",
    "generation_requirement": "deterministic_do_sample_false",
    "wikitext_supported": False,
}

ACCURACY_TASKS = (
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu",
    "gsm8k_cot",
)
CHANCE_FLOORS = {
    "arc_challenge": 0.25,
    "hellaswag": 0.25,
    "winogrande": 0.50,
    "mmlu": 0.25,
    "gsm8k_cot": 0.0,
}
PPL_TASK = "wikitext2_prefix_ppl"

TASK_FILES = {
    "arc_challenge": "arc",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "mmlu": "mmlu",
    "gsm8k_cot": "gsm8k",
}

FORMAL_EVALUATION_CODE_FILES = (
    "scripts/evaluate.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/transfer.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/eval_data.py",
    # Mapped evaluation reuses its FP32 resident-mapper implementation.
    "src/kvcache_transfer/latency.py",
)


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_safe_harness_value(value: Any) -> Any:
    """Convert lm-eval metadata to deterministic, auditable JSON values.

    Some task configurations (notably HellaSwag in the pinned harness) expose
    resolved ``doc_to_*`` or ``process_results`` Python functions.  The model
    responses and metrics are already JSON-native, but persisting the complete
    resolved configuration would otherwise fail after all requests finish.
    Callable identity is represented by module plus qualified name; the exact
    task source files are independently content-hashed in the run definition.
    Unknown object types are rejected instead of falling back to a potentially
    unstable ``repr`` containing a process-specific memory address.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("lm-eval metadata contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)):
                raise TypeError(
                    "lm-eval metadata has a non-scalar mapping key: "
                    f"{type(key).__module__}.{type(key).__qualname__}"
                )
            result[str(key)] = _json_safe_harness_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe_harness_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_json_safe_harness_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if not isinstance(module, str) or not module or not isinstance(qualname, str) or not qualname:
            raise TypeError(
                "lm-eval callable lacks a stable module/qualified name: "
                f"{type(value).__module__}.{type(value).__qualname__}"
            )
        return {"__python_callable__": {"module": module, "qualname": qualname}}
    raise TypeError(
        "unsupported lm-eval metadata object: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def model_artifact_identity(model_root: Path) -> dict[str, Any]:
    """Identify a local sharded model without hashing tens of GB of weights.

    Configuration, tokenizer, and the safetensors index are content-hashed.
    Every shard named by the index is then bound by its normalized relative
    filename and byte size.  This intentionally does not claim a content hash
    of the weight shards; the scope is explicit in the persisted provenance.
    """

    root = model_root.resolve()
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json")
    files: dict[str, dict[str, Any]] = {}
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required model identity file is missing: {path}")
        files[name] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    for optional in ("tokenizer_config.json", "special_tokens_map.json"):
        path = root / optional
        if path.is_file():
            files[optional] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}

    index_path = root / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"invalid model safetensors index: {index_path}") from exc
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"model safetensors index has no weight_map: {index_path}")
    shard_names = sorted({str(value) for value in weight_map.values()})
    shards: list[dict[str, Any]] = []
    for name in shard_names:
        shard = (root / name).resolve()
        try:
            relative = str(shard.relative_to(root))
        except ValueError as exc:
            raise ValueError(f"model index shard escapes model directory: {name}") from exc
        if not shard.is_file():
            raise FileNotFoundError(f"model shard named by index is missing: {shard}")
        shards.append({"path": relative, "nbytes": shard.stat().st_size})
    return {
        "path": str(root),
        "content_hashed_files": files,
        "weight_shards": shards,
        "weight_shard_count": len(shards),
        "weight_shard_total_nbytes": sum(item["nbytes"] for item in shards),
        "weight_shard_identity_scope": "index_sha256_plus_relative_filename_and_nbytes",
        "weight_shard_contents_hashed": False,
    }


def mapper_artifact_identity(manifest_path: Path) -> dict[str, Any]:
    """Bind a mapped run to the complete aggregate manifest commit marker."""

    manifest = manifest_path.resolve()
    checksum = manifest.parent / "manifest.sha256.json"
    if not manifest.is_file() or not checksum.is_file():
        raise FileNotFoundError(
            f"mapper manifest and checksum commit marker are required: {manifest}, {checksum}"
        )
    manifest_sha = sha256_file(manifest)
    try:
        checksum_value = json.loads(checksum.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid mapper aggregate checksum: {checksum}") from exc
    if checksum_value.get("sha256") != manifest_sha:
        raise ValueError("mapper aggregate manifest SHA256 disagrees with commit marker")
    try:
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid mapper aggregate manifest: {manifest}") from exc
    if manifest_value.get("status") != "complete":
        raise ValueError("mapped evaluation requires a complete mapper aggregate manifest")
    return {
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "checksum": str(checksum.resolve()),
        "checksum_sha256": sha256_file(checksum),
        "checksum_declared_manifest_sha256": checksum_value["sha256"],
        "fit_fingerprint": manifest_value.get("fit_fingerprint"),
        "layer_count": len(manifest_value.get("layers", [])),
    }


def runtime_artifact_provenance(configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Return mode-scoped model and mapper identities used by the run digest."""

    modes = set(configuration["modes"])
    result: dict[str, Any] = {}
    if modes & {"source_standalone", "mapped"}:
        result["source_model"] = model_artifact_identity(configuration["source_model"])
    if modes & {"target_standalone", "target_oracle", "mapped"}:
        result["target_model"] = model_artifact_identity(configuration["target_model"])
    if "mapped" in modes:
        result["mapper"] = mapper_artifact_identity(configuration["mapper_manifest"])
    return result


def project_code_provenance(project_root: Path = ROOT) -> dict[str, Any]:
    """Content-hash every local code file on the formal evaluation path."""

    files: dict[str, dict[str, Any]] = {}
    for relative in FORMAL_EVALUATION_CODE_FILES:
        path = project_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"formal evaluation code file is missing: {path}")
        files[relative] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    return {
        "files": files,
        "files_digest": _canonical_digest(files),
    }


def _git_value(directory: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def harness_provenance(harness_root: Path, tasks: Sequence[str]) -> dict[str, Any]:
    task_root = harness_root / "lm_eval" / "tasks"
    files: dict[str, dict[str, Any]] = {}
    for task in tasks:
        directory_name = TASK_FILES[task]
        directory = task_root / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"lm-eval task directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".yaml", ".yml", ".py", ".json"}:
                relative = str(path.relative_to(harness_root))
                files[relative] = {
                    "sha256": sha256_file(path),
                    "nbytes": path.stat().st_size,
                }
    commit = _git_value(harness_root, "rev-parse", "HEAD")
    status = _git_value(harness_root, "status", "--porcelain")
    return {
        "path": str(harness_root.resolve()),
        "commit": commit,
        "dirty": bool(status),
        "dirty_status": status.splitlines() if status else [],
        "package_version": _installed_harness_version(),
        "task_files": files,
        "task_files_digest": _canonical_digest(files),
    }


def _installed_harness_version() -> str | None:
    try:
        return importlib.metadata.version("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return None


def require_harness(harness_root: Path) -> Any:
    """Import the pinned harness, with an actionable error for an unbootstrapped venv."""

    version = _installed_harness_version()
    if version is None:
        raise RuntimeError(
            "the vendored lm-evaluation-harness is not installed in this environment. "
            f"Install {harness_root} (including its task dependencies) into .venv before evaluation"
        )
    try:
        import lm_eval
    except ImportError as exc:
        raise RuntimeError(
            "lm-evaluation-harness task dependencies are incomplete; run the project environment bootstrap"
        ) from exc
    imported_path = Path(lm_eval.__file__).resolve()
    try:
        imported_path.relative_to(harness_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"installed lm_eval resolves to {imported_path}, not the pinned checkout {harness_root}"
        ) from exc
    return lm_eval


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not result:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--modes",
        type=_parse_csv,
        default=EVALUATION_MODES,
        help="comma-separated source_standalone,target_standalone,target_oracle,mapped",
    )
    parser.add_argument(
        "--tasks",
        type=_parse_csv,
        default=("arc_challenge", "hellaswag", "winogrande", "mmlu", "gsm8k_cot", "wikitext2_prefix_ppl"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit",
        type=float,
        help="lm-eval sample limit (positive integer count or fraction in (0,1))",
    )
    parser.add_argument("--wikitext-max-chunks", type=int)
    parser.add_argument("--attention-implementation")
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_configuration(raw: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    root = Path(raw.get("project_root", ROOT)).resolve()
    if not bool(raw.get("completion_mode")):
        raise ValueError("formal evaluation requires completion_mode=true")
    if bool(raw.get("apply_chat_template")):
        raise ValueError("formal evaluation forbids chat templates")
    if not bool(raw.get("boundary_replay")):
        raise ValueError("formal evaluation requires boundary_replay=true")
    if int(raw.get("batch_size", -1)) != 1:
        raise ValueError("formal evaluation requires batch_size=1 without padding")

    modes = tuple(args.modes)
    invalid_modes = set(modes) - set(EVALUATION_MODES)
    if invalid_modes or len(set(modes)) != len(modes):
        raise ValueError(f"invalid or duplicate modes: {modes}")
    benchmark_config = raw.get("benchmarks")
    if not isinstance(benchmark_config, Mapping):
        raise ValueError("evaluation config lacks benchmarks")
    tasks = tuple(args.tasks)
    invalid_tasks = set(tasks) - set(benchmark_config)
    if invalid_tasks or len(set(tasks)) != len(tasks):
        raise ValueError(f"unknown or duplicate tasks: {tasks}")
    if args.limit is not None and not (args.limit > 0 and (args.limit < 1 or args.limit.is_integer())):
        raise ValueError("--limit must be a positive integer count or a fraction in (0,1)")
    if args.wikitext_max_chunks is not None and args.wikitext_max_chunks <= 0:
        raise ValueError("--wikitext-max-chunks must be positive")

    exact = {
        "arc_challenge": (0, "acc_norm"),
        "hellaswag": (0, "acc_norm"),
        "winogrande": (0, "acc"),
        "mmlu": (5, "acc"),
        "gsm8k_cot": (8, "exact_match"),
    }
    for task, (shots, metric) in exact.items():
        value = benchmark_config.get(task)
        if not isinstance(value, Mapping):
            raise ValueError(f"evaluation config lacks paper task {task}")
        if int(value.get("num_fewshot", -1)) != shots or value.get("metric") != metric:
            raise ValueError(
                f"{task} must use num_fewshot={shots}, metric={metric}; got {dict(value)}"
            )
    wiki = benchmark_config.get("wikitext2_prefix_ppl")
    if not isinstance(wiki, Mapping):
        raise ValueError("evaluation config lacks wikitext2_prefix_ppl")
    if tuple(int(wiki.get(key, -1)) for key in ("chunk_length", "prefix_length", "scored_length")) != (
        2048,
        1024,
        1024,
    ):
        raise ValueError("WikiText-2 must use non-overlap 2048 = 1024 prefix + 1024 scored")

    paths = raw.get("paths", {})
    runtime = raw.get("runtime", {})
    required_paths = ("source_model", "target_model", "mapper_manifest", "harness_root")
    missing = [name for name in required_paths if name not in paths]
    if missing:
        raise ValueError(f"evaluation config lacks paths: {missing}")
    return {
        "root": root,
        "raw": dict(raw),
        "modes": modes,
        "tasks": tasks,
        "benchmarks": benchmark_config,
        "source_model": _resolve(root, paths["source_model"]),
        "target_model": _resolve(root, paths["target_model"]),
        "mapper_manifest": _resolve(root, paths["mapper_manifest"]),
        "harness_root": _resolve(root, paths["harness_root"]),
        "source_devices": tuple(int(value) for value in runtime.get("source_devices", [4, 5])),
        "target_devices": tuple(int(value) for value in runtime.get("target_devices", [0, 1, 2, 3])),
        "attention_implementation": args.attention_implementation
        or str(runtime.get("attention_implementation", "flash_attention_2")),
        "mapper_token_chunk": int(runtime.get("mapper_token_chunk", 1024)),
        "max_gen_tokens": int(runtime.get("max_gen_tokens", 256)),
        "benchmark_manifest": _resolve(
            root, raw.get("benchmark_manifest", DEFAULT_BENCHMARK_MANIFEST)
        ),
        "local_task_dir": _resolve(
            root, raw.get("local_task_dir", DEFAULT_LOCAL_TASK_DIR)
        ),
    }


def _preflight_paths(configuration: Mapping[str, Any]) -> None:
    modes = set(configuration["modes"])
    required_directories = ["harness_root"]
    if modes & {"source_standalone", "mapped"}:
        required_directories.append("source_model")
    if modes & {"target_standalone", "target_oracle", "mapped"}:
        required_directories.append("target_model")
    for name in required_directories:
        path = configuration[name]
        if not path.is_dir():
            raise FileNotFoundError(f"{name} directory is missing: {path}")
    if "mapped" in configuration["modes"] and not configuration["mapper_manifest"].is_file():
        raise FileNotFoundError(
            f"mapped mode requires the complete mapper manifest: {configuration['mapper_manifest']}"
        )
    if not configuration["benchmark_manifest"].is_file():
        raise FileNotFoundError(
            "verified local benchmark manifest is missing: "
            f"{configuration['benchmark_manifest']}; run scripts/prepare_benchmarks.py"
        )
    if not configuration["local_task_dir"].is_dir():
        raise FileNotFoundError(
            f"local canonical task overrides are missing: {configuration['local_task_dir']}"
        )
    requested: set[int] = set()
    if modes & {"target_standalone", "target_oracle", "mapped"}:
        requested.update(configuration["target_devices"])
    if modes & {"source_standalone", "mapped"}:
        requested.update(configuration["source_devices"])
    if requested and max(requested) >= torch.cuda.device_count():
        raise ValueError(
            f"requested CUDA devices {sorted(requested)}, but torch reports {torch.cuda.device_count()} devices"
        )


def _tokenizer_identity(source: Any, target: Any) -> dict[str, Any]:
    source_vocab = source.get_vocab()
    target_vocab = target.get_vocab()
    if source_vocab != target_vocab:
        raise ValueError("source and target tokenizer vocabularies differ; cache transfer tokens are unsafe")
    source_special = {
        name: getattr(source, name, None)
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    }
    target_special = {
        name: getattr(target, name, None)
        for name in source_special
    }
    if source_special != target_special:
        raise ValueError("source and target tokenizer special token IDs differ")
    vocabulary_digest = _canonical_digest(sorted(source_vocab.items()))
    return {
        "vocabulary_size": len(source_vocab),
        "vocabulary_sha256": vocabulary_digest,
        "special_token_ids": source_special,
        "source_name": str(getattr(source, "name_or_path", "")),
        "target_name": str(getattr(target, "name_or_path", "")),
    }


def _make_output_directory(root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        output = supplied.resolve()
        output.mkdir(parents=True, exist_ok=True)
        return output
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = root / "runs" / f"{stamp}_evaluation_qwen3_14b_to_32b"
    output.mkdir(parents=True, exist_ok=False)
    return output


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if _canonical_digest(existing) != _canonical_digest(value):
            raise ValueError(f"existing immutable run input disagrees: {path}")
        return
    atomic_write_json(path, value)


def _result_path(output: Path, mode: str, task: str) -> Path:
    return output / "results" / mode / f"{task}.json"


def _evaluation_order(
    tasks: Sequence[str], modes: Sequence[str]
) -> tuple[tuple[str, str], ...]:
    """Return the durable execution order: finish every mode per task first.

    Keeping task outermost makes the target/mapped comparison available as
    soon as one benchmark completes.  It also avoids waiting for every large
    target task before the first mapped checkpoint is produced.
    """

    return tuple((task, mode) for task in tasks for mode in modes)


def _request_cache_base(
    output: Path,
    fingerprint: str,
    *,
    mode: str,
    task: str,
) -> Path:
    """Return an lm-eval cache base with strict run/mode/task isolation.

    ``simple_evaluate`` appends ``_rank0.db`` for this single-process run.  A
    full fingerprint in the path protects against accidental reuse after any
    model, mapper, code, dataset, prompt, limit, or seed-affecting run input
    changes.  Mode isolation is mandatory because lm-eval's internal key does
    not include model identity.
    """

    return output / "request_cache" / fingerprint / mode / task / "responses"


def _request_cache_record(base: Path | None) -> dict[str, Any]:
    if base is None:
        return {
            "enabled": False,
            "reason": "custom_wikitext_prefix_protocol_not_dispatched_through_lm_eval",
        }
    return {
        "enabled": True,
        "base_path": str(base.resolve()),
        "rank0_database": str(Path(f"{base.resolve()}_rank0.db")),
        **REQUEST_CACHE_POLICY,
    }


def _checkpoint_complete(
    path: Path,
    fingerprint: str,
    *,
    expected_mode: str,
    expected_task: str,
    expected_primary_metric: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metric_value = value.get("primary_metric_value")
    valid_metric_value = (
        isinstance(metric_value, (float, int))
        and not isinstance(metric_value, bool)
        and math.isfinite(float(metric_value))
    )
    return (
        value.get("schema_version") == SCHEMA_VERSION
        and value.get("status") == "complete"
        and value.get("request_fingerprint") == fingerprint
        and value.get("mode") == expected_mode
        and value.get("task") == expected_task
        and value.get("primary_metric") == expected_primary_metric
        and valid_metric_value
    )


def _metric_value(result: Mapping[str, Any], task: str, benchmark: Mapping[str, Any]) -> float:
    task_results = result.get("results", {})
    aggregate_results = result.get("groups", {})
    if task in aggregate_results:
        task_result = aggregate_results[task]
    elif task in task_results:
        task_result = task_results[task]
    else:
        raise ValueError(f"lm-eval result lacks aggregate task/group {task!r}")
    metric = str(benchmark["metric"])
    wanted_filter = benchmark.get("metric_filter")
    candidates = []
    for key, value in task_result.items():
        base, _, filter_name = key.partition(",")
        if base == metric and (wanted_filter is None or filter_name == wanted_filter):
            if isinstance(value, (float, int)):
                candidates.append((key, float(value)))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one {metric!r} metric for {task}, found {[key for key, _ in candidates]}"
        )
    return candidates[0][1]


def _run_harness_task(
    lm_eval: Any,
    *,
    adapter: Any,
    task: str,
    benchmark: Mapping[str, Any],
    limit: float | None,
    log_samples: bool,
    task_manager: Any,
    request_cache_base: Path,
) -> dict[str, Any]:
    result = lm_eval.simple_evaluate(
        model=adapter,
        tasks=[task],
        num_fewshot=int(benchmark["num_fewshot"]),
        batch_size=1,
        limit=limit,
        bootstrap_iters=0,
        apply_chat_template=False,
        fewshot_as_multiturn=False,
        log_samples=log_samples,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
        task_manager=task_manager,
        # CachingLM wraps this custom LM adapter.  The adapter's per-request
        # cache_hook writes with sqlitedict autocommit, so completed requests
        # survive SIGINT/OOM even before the task-level JSON checkpoint exists.
        use_cache=str(request_cache_base.resolve()),
    )
    if not isinstance(result, dict):
        raise RuntimeError("lm-eval returned no rank-zero result")
    _metric_value(result, task, benchmark)
    return result


def _load_wikitext(
    tokenizer: Any, benchmark: Mapping[str, Any], *, local_file: Path | None = None
) -> tuple[list[int], dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("datasets is required for WikiText-2 evaluation") from exc
    revision = benchmark.get("revision")
    if local_file is not None:
        split = str(benchmark.get("split", "test"))
        kwargs: dict[str, Any] = {
            "path": "parquet",
            "data_files": {split: str(local_file.resolve())},
            "split": split,
        }
    else:
        kwargs = {
            "path": str(benchmark.get("dataset_path", "EleutherAI/wikitext_document_level")),
            "name": str(benchmark.get("dataset_name", "wikitext-2-raw-v1")),
            "split": str(benchmark.get("split", "test")),
        }
        if revision:
            kwargs["revision"] = str(revision)
    dataset = load_dataset(**kwargs)
    field = str(benchmark.get("text_field", "page"))
    if field not in dataset.column_names:
        raise ValueError(f"WikiText dataset has no field {field!r}: {dataset.column_names}")
    separator = str(benchmark.get("document_separator", "\n\n"))
    tokens = concatenate_document_tokens(tokenizer, dataset[field], separator=separator)
    provenance = {
        "dataset_path": kwargs["path"],
        "dataset_name": kwargs.get("name"),
        "split": kwargs["split"],
        "requested_revision": revision,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "rows": len(dataset),
        "text_field": field,
        "document_separator": separator,
        "token_count": len(tokens),
        "local_file": str(local_file.resolve()) if local_file is not None else None,
    }
    return tokens, provenance


def _summarize(output: Path, configuration: Mapping[str, Any], fingerprint: str) -> dict[str, Any]:
    metrics: dict[str, dict[str, float]] = {}
    for mode in configuration["modes"]:
        metrics[mode] = {}
        for task in configuration["tasks"]:
            path = _result_path(output, mode, task)
            primary_metric = str(configuration["benchmarks"][task]["metric"])
            if not _checkpoint_complete(
                path,
                fingerprint,
                expected_mode=mode,
                expected_task=task,
                expected_primary_metric=primary_metric,
            ):
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            metrics[mode][task] = float(value["primary_metric_value"])

    retention: dict[str, float] = {}
    floor_normalized: dict[str, float] = {}
    accuracy_comparison: dict[str, dict[str, float | None]] = {}
    if "target_standalone" in metrics and "mapped" in metrics:
        common = set(metrics["target_standalone"]) & set(metrics["mapped"])
        for task in ACCURACY_TASKS:
            if task not in common:
                continue
            target = metrics["target_standalone"][task]
            mapped = metrics["mapped"][task]
            floor = CHANCE_FLOORS[task]
            if target != 0:
                retention[task] = mapped / target * 100.0
            denominator = target - floor
            normalized = None
            if denominator != 0:
                normalized = (mapped - floor) / denominator * 100.0
                floor_normalized[task] = normalized
            accuracy_comparison[task] = {
                "target_standalone": target,
                "mapped": mapped,
                "chance_floor": floor,
                "retention_percent": retention.get(task),
                "floor_normalized_retention_percent": normalized,
            }

    five_task_complete = all(task in accuracy_comparison for task in ACCURACY_TASKS)
    retention_mean = (
        sum(retention[task] for task in ACCURACY_TASKS) / len(ACCURACY_TASKS)
        if five_task_complete and all(task in retention for task in ACCURACY_TASKS)
        else None
    )
    floor_normalized_mean = (
        sum(floor_normalized[task] for task in ACCURACY_TASKS) / len(ACCURACY_TASKS)
        if five_task_complete and all(task in floor_normalized for task in ACCURACY_TASKS)
        else None
    )

    oracle_comparison: dict[str, dict[str, float | bool]] = {}
    oracle_absolute_delta: dict[str, float] = {}
    if "target_standalone" in metrics and "target_oracle" in metrics:
        common = set(metrics["target_standalone"]) & set(metrics["target_oracle"])
        for task in configuration["tasks"]:
            if task not in common:
                continue
            target = metrics["target_standalone"][task]
            oracle = metrics["target_oracle"][task]
            signed_delta = oracle - target
            absolute_delta = abs(signed_delta)
            oracle_absolute_delta[task] = absolute_delta
            item: dict[str, float | bool] = {
                "target_standalone": target,
                "target_oracle": oracle,
                "signed_delta": signed_delta,
                "absolute_delta": absolute_delta,
                "lower_is_better": task == PPL_TASK,
            }
            if task in ACCURACY_TASKS:
                item["signed_delta_percentage_points"] = signed_delta * 100.0
                item["absolute_delta_percentage_points"] = absolute_delta * 100.0
            oracle_comparison[task] = item

    ppl_comparison: dict[str, float | bool | None] = {
        "lower_is_better": True,
        "target_standalone": metrics.get("target_standalone", {}).get(PPL_TASK),
        "target_oracle": metrics.get("target_oracle", {}).get(PPL_TASK),
        "mapped": metrics.get("mapped", {}).get(PPL_TASK),
        "mapped_minus_target": None,
        "mapped_over_target_ratio": None,
        "mapped_over_target_percent": None,
    }
    target_ppl = ppl_comparison["target_standalone"]
    mapped_ppl = ppl_comparison["mapped"]
    if isinstance(target_ppl, (float, int)) and isinstance(mapped_ppl, (float, int)):
        ppl_comparison["mapped_minus_target"] = float(mapped_ppl) - float(target_ppl)
        if float(target_ppl) != 0:
            ratio = float(mapped_ppl) / float(target_ppl)
            ppl_comparison["mapped_over_target_ratio"] = ratio
            ppl_comparison["mapped_over_target_percent"] = ratio * 100.0

    summary = {
        "schema_version": SCHEMA_VERSION,
        "request_fingerprint": fingerprint,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metrics": metrics,
        "mapped_retention_percent": retention,
        "mapped_floor_normalized_retention_percent": floor_normalized,
        "accuracy_comparison": {
            "tasks": accuracy_comparison,
            "required_tasks": list(ACCURACY_TASKS),
            "available_task_count": len(accuracy_comparison),
            "five_task_complete": five_task_complete,
            "five_task_retention_arithmetic_mean_percent": retention_mean,
            "five_task_floor_normalized_arithmetic_mean_percent": floor_normalized_mean,
        },
        "oracle_vs_target": oracle_comparison,
        "oracle_vs_target_absolute_delta": oracle_absolute_delta,
        "wikitext2_prefix_ppl_comparison": ppl_comparison,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def _build_run_definition(
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    *,
    config_sha256: str,
    benchmark_data: Mapping[str, Any],
    provenance: Mapping[str, Any],
    runtime_artifacts: Mapping[str, Any],
    project_code: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable request whose digest governs checkpoint reuse."""

    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(args.config.resolve()),
        "config_sha256": config_sha256,
        "modes": list(configuration["modes"]),
        "tasks": list(configuration["tasks"]),
        "limit": args.limit,
        "wikitext_max_chunks": args.wikitext_max_chunks,
        "log_samples": bool(args.log_samples),
        "completion_mode": True,
        "apply_chat_template": False,
        "batch_size": 1,
        "boundary_replay": True,
        "execution_order": "task_outer_mode_inner",
        "request_response_cache": dict(REQUEST_CACHE_POLICY),
        "source_model": str(configuration["source_model"]),
        "target_model": str(configuration["target_model"]),
        "mapper_manifest": str(configuration["mapper_manifest"]),
        "source_devices": list(configuration["source_devices"]),
        "target_devices": list(configuration["target_devices"]),
        "attention_implementation": configuration["attention_implementation"],
        "mapper_token_chunk": configuration["mapper_token_chunk"],
        "max_gen_tokens": configuration["max_gen_tokens"],
        "benchmarks": configuration["benchmarks"],
        "benchmark_data": dict(benchmark_data),
        "harness": dict(provenance),
        "runtime_artifacts": dict(runtime_artifacts),
        "project_code": dict(project_code),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load_yaml(args.config)
    configuration = _validate_configuration(raw, args)
    _preflight_paths(configuration)
    benchmark_manifest = verify_ready_manifest(configuration["benchmark_manifest"])
    harness_tasks = tuple(task for task in configuration["tasks"] if task in TASK_FILES)
    provenance = harness_provenance(configuration["harness_root"], harness_tasks)
    runtime_artifacts = runtime_artifact_provenance(configuration)
    project_code = project_code_provenance(ROOT)

    run_definition = _build_run_definition(
        args,
        configuration,
        config_sha256=sha256_file(args.config),
        benchmark_data=manifest_digest_payload(benchmark_manifest),
        provenance=provenance,
        runtime_artifacts=runtime_artifacts,
        project_code=project_code,
    )
    fingerprint = _canonical_digest(run_definition)
    if args.dry_run:
        print(json.dumps({**run_definition, "request_fingerprint": fingerprint}, indent=2))
        return 0

    lm_eval = require_harness(configuration["harness_root"])
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=configuration["local_task_dir"])
    output = _make_output_directory(configuration["root"], args.output_dir)
    _write_immutable(
        output / "run_definition.json",
        {**run_definition, "request_fingerprint": fingerprint},
    )

    modes = set(configuration["modes"])
    target_required = bool(modes & {"target_standalone", "target_oracle", "mapped"})
    source_required = bool(modes & {"source_standalone", "mapped"})
    target_tokenizer = None
    source_tokenizer = None
    if target_required:
        target_tokenizer = AutoTokenizer.from_pretrained(
            configuration["target_model"], local_files_only=True, use_fast=True
        )
    if source_required:
        source_tokenizer = AutoTokenizer.from_pretrained(
            configuration["source_model"], local_files_only=True, use_fast=True
        )
    if source_tokenizer is not None and target_tokenizer is not None:
        tokenizer_provenance = _tokenizer_identity(source_tokenizer, target_tokenizer)
    elif source_tokenizer is not None:
        tokenizer_provenance = {
            "source_name": str(getattr(source_tokenizer, "name_or_path", ""))
        }
    elif target_tokenizer is not None:
        tokenizer_provenance = {
            "target_name": str(getattr(target_tokenizer, "name_or_path", ""))
        }
    else:  # configuration validation guarantees at least one valid mode
        raise AssertionError("evaluation requested neither source nor target model")
    atomic_write_json(output / "tokenizer.json", tokenizer_provenance)

    target = None
    source = None
    if target_required:
        target = load_qwen3(
            configuration["target_model"],
            devices=configuration["target_devices"],
            role="target",
            attention_implementation=configuration["attention_implementation"],
        )
    if source_required:
        source = load_qwen3(
            configuration["source_model"],
            devices=configuration["source_devices"],
            role="source",
            attention_implementation=configuration["attention_implementation"],
        )
    mapped_builder = None
    if "mapped" in configuration["modes"]:
        assert source is not None and target is not None
        mapped_builder = MappedPrefixCacheBuilder(
            source_model=source,
            target_model=target,
            mapper_manifest=configuration["mapper_manifest"],
            token_chunk=configuration["mapper_token_chunk"],
        )
        atomic_write_json(output / "mapper.json", mapped_builder.provenance)

    engines: dict[str, TransferEvaluationEngine] = {}
    for mode in configuration["modes"]:
        if mode == "source_standalone":
            assert source is not None and source_tokenizer is not None
            model, tokenizer = source, source_tokenizer
        else:
            assert target is not None and target_tokenizer is not None
            model, tokenizer = target, target_tokenizer
        engines[mode] = TransferEvaluationEngine(
            target_model=model,
            tokenizer=tokenizer,
            mode=mode,
            mapped_prefix_builder=mapped_builder if mode == "mapped" else None,
            max_gen_tokens=configuration["max_gen_tokens"],
        )
    wiki_tokens: list[int] | None = None
    wiki_provenance: dict[str, Any] | None = None
    adapters = {mode: make_lm_eval_adapter(engine) for mode, engine in engines.items()}
    failures = 0
    for task, mode in _evaluation_order(configuration["tasks"], configuration["modes"]):
        engine = engines[mode]
        adapter = adapters[mode]
        destination = _result_path(output, mode, task)
        primary_metric = str(configuration["benchmarks"][task]["metric"])
        if _checkpoint_complete(
            destination,
            fingerprint,
            expected_mode=mode,
            expected_task=task,
            expected_primary_metric=primary_metric,
        ):
            print(f"[resume] {mode}/{task}")
            continue
        started = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            request_cache_base: Path | None = None
            if task == "wikitext2_prefix_ppl":
                benchmark = configuration["benchmarks"][task]
                if wiki_tokens is None:
                    wiki_tokens, wiki_provenance = _load_wikitext(
                        engine.tokenizer,
                        benchmark,
                        local_file=local_wikitext_file(benchmark_manifest),
                    )
                    atomic_write_json(output / "wikitext_dataset.json", wiki_provenance)
                result = evaluate_prefix_perplexity(
                    engine,
                    wiki_tokens,
                    chunk_length=int(benchmark["chunk_length"]),
                    prefix_length=int(benchmark["prefix_length"]),
                    scored_length=int(benchmark["scored_length"]),
                    max_chunks=args.wikitext_max_chunks,
                )
                raw_result: dict[str, Any] = result.to_dict()
                primary_metric_value = result.perplexity
                task_config = dict(benchmark)
            else:
                benchmark = configuration["benchmarks"][task]
                request_cache_base = _request_cache_base(
                    output,
                    fingerprint,
                    mode=mode,
                    task=task,
                )
                raw_result = _run_harness_task(
                    lm_eval,
                    adapter=adapter,
                    task=task,
                    benchmark=benchmark,
                    limit=args.limit,
                    log_samples=args.log_samples,
                    task_manager=task_manager,
                    request_cache_base=request_cache_base,
                )
                primary_metric_value = _metric_value(raw_result, task, benchmark)
                # The pinned harness can place task functions in resolved
                # configs (HellaSwag does this).  Normalize the complete result
                # deterministically before hashing or atomic JSON persistence.
                raw_result = _json_safe_harness_value(raw_result)
                if _metric_value(raw_result, task, benchmark) != primary_metric_value:
                    raise AssertionError("lm-eval JSON normalization changed the primary metric")
                # This records the resolved harness task config in addition
                # to the exact source-file hashes captured at run start.
                task_configs = raw_result.get("configs", {})
                task_config = {
                    "configs": task_configs,
                    "canonical_configs_sha256": _canonical_digest(task_configs),
                    "n_shot": raw_result.get("n-shot", {}),
                    "versions": raw_result.get("versions", {}),
                }
            payload = {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "request_fingerprint": fingerprint,
                "mode": mode,
                "task": task,
                "started_utc": started,
                "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "primary_metric": configuration["benchmarks"][task]["metric"],
                "primary_metric_value": primary_metric_value,
                "request_cache": _request_cache_record(request_cache_base),
                "resolved_task_config": task_config,
                "result": raw_result,
            }
            atomic_write_json(destination, payload)
            (destination.with_suffix(".failure.json")).unlink(missing_ok=True)
            engine.clear_prefix_cache()
            print(f"[complete] {mode}/{task}: {primary_metric_value}")
        except BaseException as exc:
            failures += 1
            atomic_write_json(
                destination.with_suffix(".failure.json"),
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed",
                    "request_fingerprint": fingerprint,
                    "mode": mode,
                    "task": task,
                    "started_utc": started,
                    "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            engine.clear_prefix_cache()
            print(f"[failed] {mode}/{task}: {exc}", file=sys.stderr)
            break

    summary = _summarize(output, configuration, fingerprint)
    atomic_write_json(
        output / "status.json",
        {
            "status": "failed" if failures else "complete",
            "failures": failures,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "summary": summary,
        },
    )
    print(f"results: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
