#!/usr/bin/env python3
"""Finalize the audited Qwen3-14B -> 32B GSM8K and five-task result.

This is a deliberately task-scoped bridge.  It accepts the two fresh GSM8K
standalone runs and the pinned, already-complete mapped GSM8K run, proves that
their frozen task protocol and sample identities agree, and appends the
canonical strict-match GSM8K row to the previously audited four-task result.

The script is fail-closed: every run is revalidated by the formal aggregation
validator and the result-level GSM8K scorer, the historical/current code
transition is an exact two-file SHA256 allowlist, and publication targets must
not already exist.  The output manifest is written last in a staging directory
and the complete directory is then atomically renamed into place.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.evaluation_aggregate import (  # noqa: E402
    ACCURACY_TASKS,
    AggregationError,
    CHANCE_FLOORS,
    PAPER_REFERENCE,
    RunRecord,
    validate_run,
)
from kvcache_transfer.gsm8k_numeric_diagnostic import (  # noqa: E402
    FLEXIBLE_FILTER,
    GSM8KNumericDiagnosticError,
    METRIC_NAME as NUMERIC_METRIC_NAME,
    STRICT_FILTER,
    score_checkpoint,
)
from kvcache_transfer.io import sha256_file  # noqa: E402


SCHEMA_VERSION = 1
TASK = "gsm8k_cot"
SAMPLE_COUNT = 1_319
FILTER_ROW_COUNT = 2_638
MAX_GEN_TOKENS = 1_024
MODES = ("source_standalone", "target_standalone", "mapped")
MODE_LABELS = {
    "source_standalone": "source",
    "target_standalone": "target",
    "mapped": "mapped",
}
FIVE_TASKS = (
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu",
    TASK,
)
BASE_FOUR_TASKS = FIVE_TASKS[:-1]
EXPECTED_BASE_SHA256 = (
    "1af5ace3fd851405d9c78f1a57687287a36d2d27212166d1362279038ea4eb73"
)
EXPECTED_RUN_FINGERPRINTS = {
    "source_standalone": (
        "ab29eddde52b99d71d1178106413d2e1b894a60b24bf70e131bf446804c2c57d"
    ),
    "target_standalone": (
        "ee44d17c448e0ef4046c8de59e2b18498114c4b60d1d7e0bc66de16610ee5d21"
    ),
    "mapped": (
        "34acf472afbe0d65c1a0e382a1796641360b4d1496fea574702fc82501f06e11"
    ),
}
EXPECTED_MAPPED_CHECKPOINT_SHA256 = (
    "2628b3c0668cc7b51ddc5f984d3b742b0a993424f38f3d26f848ca7752cb84fb"
)
EXPECTED_SCORER_FILES_DIGEST = (
    "cb71a3efa12f44e8797517fe6da98e1a521890bfff8776e1ff815152ec2dfe7c"
)
EXPECTED_PROJECT_FILES = {
    "scripts/evaluate.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/eval_data.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/transfer.py",
}
REVIEWED_PROJECT_CODE_TRANSITION = {
    "src/kvcache_transfer/latency.py": {
        "historical": {
            "nbytes": 35_307,
            "sha256": (
                "9e1180e0e92d0cf79f944812e94cf9cadd0348cfb261563ebf8cfa4d25809cf7"
            ),
        },
        "current": {
            "nbytes": 37_705,
            "sha256": (
                "1566894a35cdf52d947d78e9499efcd13d4d0fd5ea530b5c5e73af858b57e582"
            ),
        },
        "scope": "reviewed evaluation-unrelated latency implementation change",
    },
    "src/kvcache_transfer/modeling.py": {
        "historical": {
            "nbytes": 1_873,
            "sha256": (
                "e8c1dc342587780c039fdcd04fde6c077f407edc974abad1a3a87b4bba959757"
            ),
        },
        "current": {
            "nbytes": 10_938,
            "sha256": (
                "303ff79e388acbe30ca92adceca47a1013bc5ecfe6268e847ce46374120d7364"
            ),
        },
        "scope": (
            "reviewed model-placement implementation change; persisted model, "
            "tokenizer, protocol, and sample identities are independently verified"
        ),
    },
}
BASE_EXPECTED_N = {
    "arc_challenge": 1_172,
    "hellaswag": 10_042,
    "winogrande": 1_267,
    "mmlu": 14_042,
}
BASE_EXPECTED_METRICS = {
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "mmlu": "acc",
}
IDENTITY_FIELDS = (
    "doc",
    "doc_hash",
    "prompt_hash",
    "target",
    "target_hash",
    "arguments",
)
OUTPUT_NAMES = {
    "compatibility_receipt.json",
    "cross_mode_sample_identity.jsonl",
    "five_task_strict_retention.csv",
    "five_task_strict_retention.md",
    "report.json",
    "score_layers.json",
}


class FinalizationError(ValueError):
    """Raised when inputs cannot support the formal five-task result."""


def _reject_json_constant(value: str) -> None:
    raise FinalizationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FinalizationError(f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path, label: str | None = None) -> Any:
    candidate = Path(path).resolve()
    if candidate.is_symlink() or not candidate.is_file():
        raise FinalizationError(f"{label or 'JSON input'} is missing or a symlink: {candidate}")
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_pairs,
            )
    except FinalizationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FinalizationError(f"cannot read strict JSON from {candidate}: {exc}") from exc


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FinalizationError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise FinalizationError(f"{label} must be a JSON array")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizationError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FinalizationError(f"{label} must be a finite number")
    return result


def _json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    try:
        if pretty:
            text = json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        else:
            text = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
    except (TypeError, ValueError) as exc:
        raise FinalizationError(f"value is not strict canonical JSON: {exc}") from exc
    return (text + ("\n" if pretty else "")).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _equal(left: Any, right: Any) -> bool:
    """JSON-type-sensitive structural equality independent of object key order."""

    return _json_bytes(left) == _json_bytes(right)


def _close(left: float, right: float, label: str, *, tolerance: float = 1e-12) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise FinalizationError(f"{label} mismatch: {left!r} != {right!r}")


def _sha_metadata(data: bytes) -> dict[str, Any]:
    return {"sha256": _sha256_bytes(data), "size_bytes": len(data)}


def _paper_values(task: str) -> dict[str, float]:
    return {
        "source": float(PAPER_REFERENCE["source_standalone"][task]),
        "target": float(PAPER_REFERENCE["target_standalone"][task]),
        "mapped": float(PAPER_REFERENCE["mapped"][task]),
    }


def _floor_retention(mapped: float, target: float, floor: float, label: str) -> float:
    denominator = target - floor
    if denominator == 0.0:
        raise FinalizationError(f"cannot compute floor-normalized retention for {label}")
    return (mapped - floor) / denominator * 100.0


def _validate_base_report(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    base_path = path.resolve()
    actual_sha = sha256_file(base_path)
    if actual_sha != EXPECTED_BASE_SHA256:
        raise FinalizationError(
            f"base audited report SHA256 mismatch: {actual_sha} != {EXPECTED_BASE_SHA256}"
        )

    sidecar_path = base_path.with_name(f"{base_path.stem}.sha256.json")
    markdown_path = base_path.with_suffix(".md")
    sidecar = _mapping(_load_json(sidecar_path, "base SHA256 sidecar"), "base sidecar")
    if sidecar.get("schema_version") != 1:
        raise FinalizationError("base SHA256 sidecar schema_version must be 1")
    files = _mapping(sidecar.get("files"), "base sidecar.files")
    expected_names = {base_path.name, markdown_path.name}
    if set(files) != expected_names:
        raise FinalizationError(
            f"base sidecar file set mismatch: {sorted(files)} != {sorted(expected_names)}"
        )
    for candidate in (base_path, markdown_path):
        declared = files.get(candidate.name)
        if not isinstance(declared, str) or declared != sha256_file(candidate):
            raise FinalizationError(f"base sidecar checksum mismatch for {candidate.name}")

    base = dict(_mapping(_load_json(base_path, "base audited report"), "base report"))
    required_header = {
        "schema_version": 1,
        "status": "audited_partial_five_task_table",
        "direction": "Qwen3-14B_to_Qwen3-32B",
        "five_accuracy_task_table_complete": False,
        "mapper_manifest_sha256": (
            "785e2d57c8953a0c630f3044c4c06381f81a88799ca651960586f584f569ec08"
        ),
        "mapper_fit_fingerprint": (
            "860adcba99ae9c61bdf39c084145bc8d65cedb6ebac0454d07ff939b7b85f0f9"
        ),
    }
    for field, expected in required_header.items():
        if base.get(field) != expected:
            raise FinalizationError(
                f"base report {field} mismatch: {base.get(field)!r} != {expected!r}"
            )
    if base.get("comparable_accuracy_tasks") != list(BASE_FOUR_TASKS):
        raise FinalizationError("base report comparable task ordering mismatch")

    tasks = _mapping(base.get("tasks"), "base report.tasks")
    if set(tasks) != set(FIVE_TASKS) | {"wikitext2_prefix_ppl"}:
        raise FinalizationError("base report task set mismatch")
    raw_retentions: list[float] = []
    floor_retentions: list[float] = []
    paper_raw_retentions: list[float] = []
    paper_floor_retentions: list[float] = []
    for task in BASE_FOUR_TASKS:
        row = _mapping(tasks.get(task), f"base tasks.{task}")
        if row.get("metric") != BASE_EXPECTED_METRICS[task]:
            raise FinalizationError(f"base {task} metric mismatch")
        if row.get("n") != BASE_EXPECTED_N[task]:
            raise FinalizationError(f"base {task} sample count mismatch")
        floor = _finite(row.get("chance_floor"), f"base {task}.chance_floor")
        if floor != CHANCE_FLOORS[task]:
            raise FinalizationError(f"base {task} chance floor mismatch")
        source = _finite(row.get("source"), f"base {task}.source")
        target = _finite(row.get("target"), f"base {task}.target")
        mapped = _finite(row.get("mapped"), f"base {task}.mapped")
        for label, value in (("source", source), ("target", target), ("mapped", mapped)):
            if not 0.0 <= value <= 1.0:
                raise FinalizationError(f"base {task}.{label} is outside [0, 1]")
        raw = mapped / target * 100.0
        floor_value = _floor_retention(mapped, target, floor, task)
        _close(raw, _finite(row.get("raw_retention_percent"), f"base {task}.raw retention"), f"base {task} raw retention")
        _close(
            floor_value,
            _finite(
                row.get("floor_normalized_retention_percent"),
                f"base {task}.floor retention",
            ),
            f"base {task} floor retention",
        )
        paper = _mapping(row.get("paper"), f"base {task}.paper")
        expected_paper = _paper_values(task)
        if not _equal(paper, expected_paper):
            raise FinalizationError(f"base {task} paper reference mismatch")
        paper_raw = expected_paper["mapped"] / expected_paper["target"] * 100.0
        paper_floor = _floor_retention(
            expected_paper["mapped"], expected_paper["target"], floor, f"paper {task}"
        )
        raw_retentions.append(raw)
        floor_retentions.append(floor_value)
        paper_raw_retentions.append(paper_raw)
        paper_floor_retentions.append(paper_floor)

    summary = _mapping(base.get("four_task_summary"), "base four_task_summary")
    expected_summary = {
        "local_raw_retention_arithmetic_mean_percent": sum(raw_retentions) / 4,
        "paper_same_four_task_raw_retention_arithmetic_mean_percent": (
            sum(paper_raw_retentions) / 4
        ),
        "local_floor_normalized_arithmetic_mean_percent": sum(floor_retentions) / 4,
        "paper_same_four_task_floor_normalized_arithmetic_mean_percent": (
            sum(paper_floor_retentions) / 4
        ),
    }
    for field, expected in expected_summary.items():
        _close(_finite(summary.get(field), f"base summary.{field}"), expected, field)
    if summary.get("must_not_be_reported_as_paper_five_task_retention") is not True:
        raise FinalizationError("base four-task caveat is missing")

    wikitext = _mapping(tasks.get("wikitext2_prefix_ppl"), "base WikiText row")
    if wikitext.get("included_in_accuracy_retention") is not False:
        raise FinalizationError("base WikiText row must be excluded from accuracy retention")
    gsm = _mapping(tasks.get(TASK), "base GSM8K row")
    if gsm.get("n") != SAMPLE_COUNT or gsm.get("mapped_max1024_correct") != 875:
        raise FinalizationError("base mapped GSM8K sample/correct count mismatch")
    if gsm.get("mapped_max1024_flexible_correct") != 1_023:
        raise FinalizationError("base mapped GSM8K flexible count mismatch")
    if gsm.get("mapped_max1024_cap_hits") != 269 or gsm.get("mapped_max1024_cap") != 1_024:
        raise FinalizationError("base mapped GSM8K cap-proxy anchor mismatch")
    _close(
        _finite(gsm.get("mapped_max1024"), "base mapped GSM strict"),
        875 / SAMPLE_COUNT,
        "base mapped GSM strict",
    )
    _close(
        _finite(gsm.get("mapped_max1024_flexible"), "base mapped GSM flexible"),
        1_023 / SAMPLE_COUNT,
        "base mapped GSM flexible",
    )

    result_hashes = _mapping(base.get("result_sha256"), "base result_sha256")
    for name, value in result_hashes.items():
        if (
            not isinstance(name, str)
            or not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise FinalizationError("base result_sha256 contains an invalid entry")
    if result_hashes.get("gsm_mapped_max1024") != EXPECTED_MAPPED_CHECKPOINT_SHA256:
        raise FinalizationError("base mapped GSM8K checkpoint anchor mismatch")

    provenance = {
        "path": str(base_path),
        "sha256": actual_sha,
        "size_bytes": base_path.stat().st_size,
        "sidecar": {
            "path": str(sidecar_path),
            "sha256": sha256_file(sidecar_path),
            "verified_files": dict(files),
        },
        "markdown": {
            "path": str(markdown_path),
            "sha256": sha256_file(markdown_path),
        },
        "internal_four_task_arithmetic_recomputed": True,
        "pinned_expected_sha256": EXPECTED_BASE_SHA256,
    }
    return base, provenance


def _project_files(record: RunRecord) -> Mapping[str, Any]:
    project = _mapping(record.definition.get("project_code"), "run_definition.project_code")
    files = _mapping(project.get("files"), "run_definition.project_code.files")
    if set(files) != EXPECTED_PROJECT_FILES:
        raise FinalizationError(
            f"project-code file set mismatch for {record.run_dir}: {sorted(files)}"
        )
    return files


def _shared_definition(definition: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(definition))
    # request_fingerprint is derived from the three explicitly partitioned
    # fields and therefore cannot itself be shared across mode-specific runs.
    for field in ("request_fingerprint", "modes", "runtime_artifacts", "project_code"):
        result.pop(field, None)
    return result


def _validate_project_code(records: Mapping[str, RunRecord]) -> dict[str, Any]:
    source = _project_files(records["source_standalone"])
    target = _project_files(records["target_standalone"])
    historical = _project_files(records["mapped"])
    if not _equal(source, target):
        raise FinalizationError("fresh source and target project-code identities differ")

    changed: dict[str, Any] = {}
    unchanged: dict[str, Any] = {}
    for name in sorted(EXPECTED_PROJECT_FILES):
        old = historical[name]
        current = source[name]
        if name in REVIEWED_PROJECT_CODE_TRANSITION:
            expected = REVIEWED_PROJECT_CODE_TRANSITION[name]
            if not _equal(old, expected["historical"]):
                raise FinalizationError(f"historical project-code identity mismatch for {name}")
            if not _equal(current, expected["current"]):
                raise FinalizationError(f"current project-code identity mismatch for {name}")
            changed[name] = copy.deepcopy(expected)
        else:
            if not _equal(old, current):
                raise FinalizationError(f"unreviewed project-code change is forbidden: {name}")
            unchanged[name] = copy.deepcopy(old)
    if set(changed) != set(REVIEWED_PROJECT_CODE_TRANSITION) or len(unchanged) != 9:
        raise FinalizationError("project-code transition is not exactly two changed plus nine unchanged files")
    return {
        "source_target_project_code_exact_match": True,
        "historical_to_current_policy": "exact two-file reviewed SHA256 allowlist",
        "changed_files": changed,
        "unchanged_files": unchanged,
        "changed_file_count": 2,
        "unchanged_file_count": 9,
        "historical_files_digest": records["mapped"].definition["project_code"]["files_digest"],
        "current_files_digest": records["source_standalone"].definition["project_code"]["files_digest"],
    }


def _verify_live_model_identity(identity: Mapping[str, Any], label: str) -> dict[str, Any]:
    model_path_value = identity.get("path")
    if not isinstance(model_path_value, str):
        raise FinalizationError(f"{label} model path is invalid")
    model_path = Path(model_path_value).resolve()
    if not model_path.is_dir():
        raise FinalizationError(f"{label} model directory is missing: {model_path}")

    content_files = _mapping(identity.get("content_hashed_files"), f"{label}.content_hashed_files")
    for relative, raw_metadata in content_files.items():
        if not isinstance(relative, str):
            raise FinalizationError(f"{label} content filename is invalid")
        metadata = _mapping(raw_metadata, f"{label} content metadata")
        candidate = model_path / relative
        if not candidate.is_file():
            raise FinalizationError(f"{label} content file is missing: {candidate}")
        if candidate.stat().st_size != metadata.get("nbytes"):
            raise FinalizationError(f"{label} content file size mismatch: {relative}")
        if sha256_file(candidate) != metadata.get("sha256"):
            raise FinalizationError(f"{label} content SHA256 mismatch: {relative}")

    shards = _sequence(identity.get("weight_shards"), f"{label}.weight_shards")
    if identity.get("weight_shard_count") != len(shards):
        raise FinalizationError(f"{label} weight shard count mismatch")
    seen: set[str] = set()
    total = 0
    for raw_shard in shards:
        shard = _mapping(raw_shard, f"{label} weight shard")
        relative = shard.get("path")
        nbytes = shard.get("nbytes")
        if not isinstance(relative, str) or relative in seen:
            raise FinalizationError(f"{label} weight shard path is invalid or duplicated")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes <= 0:
            raise FinalizationError(f"{label} weight shard size is invalid")
        candidate = model_path / relative
        if not candidate.is_file() or candidate.stat().st_size != nbytes:
            raise FinalizationError(f"{label} weight shard identity mismatch: {relative}")
        seen.add(relative)
        total += nbytes
    if total != identity.get("weight_shard_total_nbytes"):
        raise FinalizationError(f"{label} total weight shard bytes mismatch")
    return {
        "path": str(model_path),
        "captured_identity_sha256": _digest(identity),
        "content_hashed_files_reverified": sorted(content_files),
        "weight_shards_stat_reverified": len(shards),
        "weight_shard_bytes_reverified": total,
        "weight_shard_hash_scope": identity.get("weight_shard_identity_scope"),
    }


def _validate_runtime_identity(
    records: Mapping[str, RunRecord], base: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    source_runtime = _mapping(records["source_standalone"].runtime_artifacts, "source runtime")
    target_runtime = _mapping(records["target_standalone"].runtime_artifacts, "target runtime")
    mapped_runtime = _mapping(records["mapped"].runtime_artifacts, "mapped runtime")
    if set(source_runtime) != {"source_model"}:
        raise FinalizationError("source runtime artifact set mismatch")
    if set(target_runtime) != {"target_model"}:
        raise FinalizationError("target runtime artifact set mismatch")
    if set(mapped_runtime) != {"source_model", "target_model", "mapper"}:
        raise FinalizationError("mapped runtime artifact set mismatch")
    source_model = _mapping(source_runtime["source_model"], "source model identity")
    target_model = _mapping(target_runtime["target_model"], "target model identity")
    if not _equal(source_model, mapped_runtime["source_model"]):
        raise FinalizationError("source model identity differs between fresh and mapped runs")
    if not _equal(target_model, mapped_runtime["target_model"]):
        raise FinalizationError("target model identity differs between fresh and mapped runs")
    for record in records.values():
        if record.definition.get("source_model") != source_model.get("path"):
            raise FinalizationError(f"{record.run_dir}: source model path/identity mismatch")
        if record.definition.get("target_model") != target_model.get("path"):
            raise FinalizationError(f"{record.run_dir}: target model path/identity mismatch")

    source_content = _mapping(source_model.get("content_hashed_files"), "source content identity")
    target_content = _mapping(target_model.get("content_hashed_files"), "target content identity")
    tokenizer_names = ("tokenizer.json", "tokenizer_config.json")
    for name in tokenizer_names:
        if name not in source_content or not _equal(source_content[name], target_content.get(name)):
            raise FinalizationError(f"source/target tokenizer identity mismatch for {name}")

    mapper = _mapping(mapped_runtime["mapper"], "mapper runtime identity")
    if mapper.get("manifest_sha256") != base.get("mapper_manifest_sha256"):
        raise FinalizationError("mapper manifest identity differs from the pinned base report")
    if mapper.get("fit_fingerprint") != base.get("mapper_fit_fingerprint"):
        raise FinalizationError("mapper fit fingerprint differs from the pinned base report")
    if mapper.get("checksum_declared_manifest_sha256") != mapper.get("manifest_sha256"):
        raise FinalizationError("mapper checksum declaration does not anchor its manifest")
    if mapper.get("layer_count") != 64:
        raise FinalizationError("mapper layer count must be 64")

    mapper_manifest = Path(str(mapper.get("manifest"))).resolve()
    mapper_checksum = Path(str(mapper.get("checksum"))).resolve()
    if sha256_file(mapper_manifest) != mapper.get("manifest_sha256"):
        raise FinalizationError("live mapper manifest SHA256 differs from captured identity")
    if sha256_file(mapper_checksum) != mapper.get("checksum_sha256"):
        raise FinalizationError("live mapper checksum sidecar differs from captured identity")
    checksum_payload = _mapping(_load_json(mapper_checksum, "mapper checksum"), "mapper checksum")
    if checksum_payload.get("sha256") != mapper.get("manifest_sha256"):
        raise FinalizationError("mapper checksum sidecar has the wrong manifest SHA256")
    manifest_payload = _mapping(_load_json(mapper_manifest, "mapper manifest"), "mapper manifest")
    if (
        manifest_payload.get("status") != "complete"
        or manifest_payload.get("fit_fingerprint") != mapper.get("fit_fingerprint")
        or len(_sequence(manifest_payload.get("layers"), "mapper manifest.layers")) != 64
    ):
        raise FinalizationError("mapper manifest completion/fingerprint/layer contract mismatch")

    verification = {
        "source_model": _verify_live_model_identity(source_model, "source"),
        "target_model": _verify_live_model_identity(target_model, "target"),
        "source_model_cross_run_exact_match": True,
        "target_model_cross_run_exact_match": True,
        "source_target_tokenizer_exact_match": True,
        "tokenizer_files": {
            name: copy.deepcopy(source_content[name]) for name in tokenizer_names
        },
        "mapper": {
            "captured_identity": copy.deepcopy(mapper),
            "captured_identity_sha256": _digest(mapper),
            "live_manifest_sha256_reverified": True,
            "live_checksum_sha256_reverified": True,
            "fit_fingerprint_reverified": True,
            "layer_count_reverified": 64,
        },
    }
    return verification, str(target_model["path"])


def _checkpoint_path(record: RunRecord, mode: str) -> Path:
    matches = [item for item in record.checkpoints if item.mode == mode and item.task == TASK]
    if len(matches) != 1 or len(record.checkpoints) != 1:
        raise FinalizationError(f"{record.run_dir}: expected exactly one {mode}/{TASK} checkpoint")
    return matches[0].path


def _validate_runs(run_dirs: Mapping[str, Path]) -> tuple[dict[str, RunRecord], dict[str, Any]]:
    records: dict[str, RunRecord] = {}
    reports: dict[str, Any] = {}
    for mode in MODES:
        try:
            record = validate_run(run_dirs[mode])
        except AggregationError as exc:
            raise FinalizationError(f"formal run validation failed for {mode}: {exc}") from exc
        if record.fingerprint != EXPECTED_RUN_FINGERPRINTS[mode]:
            raise FinalizationError(f"{mode} request fingerprint is not the pinned protocol fingerprint")
        if record.definition.get("modes") != [mode] or record.definition.get("tasks") != [TASK]:
            raise FinalizationError(f"{mode} run must contain exactly one mode and GSM8K task")
        status = _mapping(_load_json(record.run_dir / "status.json", f"{mode} status"), f"{mode} status")
        if status.get("status") != "complete" or status.get("failures") != 0:
            raise FinalizationError(f"{mode} must be complete with zero failures")
        checkpoint_path = _checkpoint_path(record, mode)
        try:
            report = score_checkpoint(checkpoint_path)
        except GSM8KNumericDiagnosticError as exc:
            raise FinalizationError(f"deep GSM8K scoring validation failed for {mode}: {exc}") from exc
        if report.get("status") != "complete" or report.get("task") != TASK:
            raise FinalizationError(f"{mode} scorer report completion/task mismatch")
        if report.get("metric_name") != NUMERIC_METRIC_NAME:
            raise FinalizationError(f"{mode} numeric diagnostic version mismatch")
        implementation = _mapping(report.get("implementation"), f"{mode} scorer implementation")
        if implementation.get("files_digest") != EXPECTED_SCORER_FILES_DIGEST:
            raise FinalizationError(f"{mode} scorer implementation digest is not pinned")
        counts = _mapping(report.get("counts"), f"{mode} scorer counts")
        if counts.get("samples") != SAMPLE_COUNT or counts.get("raw_filter_rows") != FILTER_ROW_COUNT:
            raise FinalizationError(f"{mode} scorer did not validate 1319 samples / 2638 rows")
        checkpoint_sha = report["input"]["checkpoint_sha256"]
        if checkpoint_sha != sha256_file(checkpoint_path):
            raise FinalizationError(f"{mode} scorer checkpoint checksum mismatch")
        if mode == "mapped" and checkpoint_sha != EXPECTED_MAPPED_CHECKPOINT_SHA256:
            raise FinalizationError("historical mapped checkpoint SHA256 is not pinned")
        records[mode] = record
        reports[mode] = report

    shared = _shared_definition(records[MODES[0]].definition)
    for mode in MODES[1:]:
        candidate = _shared_definition(records[mode].definition)
        if not _equal(shared, candidate):
            raise FinalizationError(
                f"shared run_definition differs for {mode}; only modes/runtime_artifacts/"
                "project_code and derived request_fingerprint may differ"
            )
    first_resolved = records[MODES[0]].checkpoints[0].resolved_task_config
    for mode in MODES[1:]:
        if not _equal(first_resolved, records[mode].checkpoints[0].resolved_task_config):
            raise FinalizationError(f"resolved GSM8K task config differs for {mode}")
    return records, reports


def _strict_samples(path: Path, mode: str) -> list[Mapping[str, Any]]:
    checkpoint = _mapping(_load_json(path, f"{mode} checkpoint"), f"{mode} checkpoint")
    result = _mapping(checkpoint.get("result"), f"{mode} checkpoint.result")
    samples_by_task = _mapping(result.get("samples"), f"{mode} result.samples")
    samples = _sequence(samples_by_task.get(TASK), f"{mode} GSM8K samples")
    if len(samples) != FILTER_ROW_COUNT:
        raise FinalizationError(f"{mode} raw filter row count must be {FILTER_ROW_COUNT}")
    paired: dict[int, dict[str, Mapping[str, Any]]] = {}
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"{mode} samples[{index}]")
        doc_id = sample.get("doc_id")
        filter_name = sample.get("filter")
        if isinstance(doc_id, bool) or not isinstance(doc_id, int) or doc_id < 0:
            raise FinalizationError(f"{mode} samples[{index}] has invalid doc_id")
        if filter_name not in {STRICT_FILTER, FLEXIBLE_FILTER}:
            raise FinalizationError(f"{mode} doc_id={doc_id} has invalid filter")
        bucket = paired.setdefault(doc_id, {})
        if filter_name in bucket:
            raise FinalizationError(f"{mode} doc_id={doc_id} has duplicate filter row")
        bucket[str(filter_name)] = sample
    if sorted(paired) != list(range(SAMPLE_COUNT)):
        raise FinalizationError(f"{mode} doc_id coverage must be contiguous 0..1318")
    strict: list[Mapping[str, Any]] = []
    for doc_id in range(SAMPLE_COUNT):
        bucket = paired[doc_id]
        if set(bucket) != {STRICT_FILTER, FLEXIBLE_FILTER}:
            raise FinalizationError(f"{mode} doc_id={doc_id} must have two canonical filters")
        for field in (*IDENTITY_FIELDS, "resps"):
            if not _equal(bucket[STRICT_FILTER].get(field), bucket[FLEXIBLE_FILTER].get(field)):
                raise FinalizationError(f"{mode} doc_id={doc_id} filter rows differ in {field}")
        strict.append(bucket[STRICT_FILTER])
    return strict


def _response(sample: Mapping[str, Any], label: str) -> str:
    outer = _sequence(sample.get("resps"), f"{label}.resps")
    if len(outer) != 1:
        raise FinalizationError(f"{label}.resps must have one repeat")
    inner = _sequence(outer[0], f"{label}.resps[0]")
    if len(inner) != 1 or not isinstance(inner[0], str):
        raise FinalizationError(f"{label}.resps[0] must contain one string")
    return inner[0]


def _cross_mode_identity(
    samples: Mapping[str, list[Mapping[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for doc_id in range(SAMPLE_COUNT):
        reference = samples["source_standalone"][doc_id]
        field_digests: dict[str, str] = {}
        for field in IDENTITY_FIELDS:
            reference_value = reference.get(field)
            for mode in MODES[1:]:
                if not _equal(reference_value, samples[mode][doc_id].get(field)):
                    raise FinalizationError(
                        f"cross-mode sample identity mismatch: doc_id={doc_id}, field={field}, mode={mode}"
                    )
            field_digests[f"{field}_canonical_sha256"] = _digest(reference_value)
        identity = {field: reference.get(field) for field in IDENTITY_FIELDS}
        rows.append(
            {
                "doc_id": doc_id,
                "identity_sha256": _digest(identity),
                "doc_hash": reference.get("doc_hash"),
                "prompt_hash": reference.get("prompt_hash"),
                "target_hash": reference.get("target_hash"),
                "field_canonical_sha256": field_digests,
                "verified_equal_modes": list(MODES),
            }
        )
    assurance = {
        "sample_count": len(rows),
        "fields_compared_exactly": list(IDENTITY_FIELDS),
        "all_fields_equal_across_modes": True,
        "per_sample_identity_vector_sha256": _digest(
            [row["identity_sha256"] for row in rows]
        ),
    }
    return rows, assurance


def _cap_proxies(
    samples: Mapping[str, list[Mapping[str, Any]]], tokenizer_path: str
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise FinalizationError("transformers is required for the token-cap proxy audit") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:  # pragma: no cover - environment-specific loader errors
        raise FinalizationError(f"cannot load the pinned local tokenizer: {exc}") from exc

    result: dict[str, Any] = {}
    for mode in MODES:
        lengths = [
            len(
                tokenizer.encode(
                    _response(sample, f"{mode} doc_id={doc_id}"),
                    add_special_tokens=False,
                )
            )
            for doc_id, sample in enumerate(samples[mode])
        ]
        at_cap = sum(length == MAX_GEN_TOKENS for length in lengths)
        at_or_above = sum(length >= MAX_GEN_TOKENS for length in lengths)
        result[mode] = {
            "decoded_completion_reencoded_length_equal_cap": at_cap,
            "decoded_completion_reencoded_length_at_or_above_cap": at_or_above,
            "fraction_at_or_above_cap": at_or_above / SAMPLE_COUNT,
            "minimum_reencoded_tokens": min(lengths),
            "maximum_reencoded_tokens": max(lengths),
            "cap": MAX_GEN_TOKENS,
        }
    return {
        "role": "tokenizer_reencode_proxy_only_not_direct_generation_token_counter",
        "definition": (
            "Raw decoded completion is re-encoded with the pinned shared tokenizer, "
            "add_special_tokens=False; length >= max_gen_tokens is counted as a cap-hit proxy."
        ),
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "modes": result,
    }


def _score_layers(reports: Mapping[str, Any], cap_proxy: Mapping[str, Any]) -> dict[str, Any]:
    layers = {
        "canonical_strict_match": {
            "role": "formal_primary_and_only_five_task_headline_scoring",
            "metric": "exact_match,strict-match",
        },
        "flexible_extract": {
            "role": "lm_eval_sensitivity_diagnostic_only",
            "metric": "exact_match,flexible-extract",
        },
        "numeric_final_v1": {
            "role": "result_level_alignment_diagnostic_only",
            "metric": NUMERIC_METRIC_NAME,
        },
    }
    report_keys = {
        "canonical_strict_match": ("preserved_lm_eval_metrics", "strict_match"),
        "flexible_extract": ("preserved_lm_eval_metrics", "flexible_extract"),
        "numeric_final_v1": ("numeric_final_diagnostic",),
    }
    for layer, path in report_keys.items():
        by_mode: dict[str, Any] = {}
        for mode in MODES:
            value: Any = reports[mode]
            for key in path:
                value = value[key]
            metric = _mapping(value, f"{mode} {layer}")
            correct = metric.get("correct")
            total = metric.get("total")
            accuracy = _finite(metric.get("accuracy"), f"{mode} {layer}.accuracy")
            if correct != round(accuracy * SAMPLE_COUNT) or total != SAMPLE_COUNT:
                raise FinalizationError(f"{mode} {layer} count/accuracy mismatch")
            by_mode[MODE_LABELS[mode]] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
                "accuracy_percent": accuracy * 100.0,
            }
        target = by_mode["target"]["accuracy"]
        mapped = by_mode["mapped"]["accuracy"]
        if target == 0.0:
            raise FinalizationError(f"cannot compute {layer} GSM8K retention with zero target")
        gsm_retention = mapped / target * 100.0
        base_raw = []
        base_floor = []
        # These fields are populated by _add_sensitivity_means after the base
        # report has been validated; placeholders here keep layer construction
        # independent from the historical four-task input.
        layers[layer]["modes"] = by_mode
        layers[layer]["gsm8k_mapped_over_target_retention_percent"] = gsm_retention
        layers[layer]["gsm8k_floor_normalized_retention_percent"] = gsm_retention
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "canonical_headline_layer": "canonical_strict_match",
        "diagnostic_layers_are_not_formal_paper_replacements": True,
        "layers": layers,
        "max_generation_token_cap_proxy": copy.deepcopy(cap_proxy),
    }


def _build_formal_rows(
    base: Mapping[str, Any], score_layers: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    base_tasks = _mapping(base.get("tasks"), "base tasks")
    strict_modes = score_layers["layers"]["canonical_strict_match"]["modes"]
    rows: list[dict[str, Any]] = []
    raw_values: list[float] = []
    floor_values: list[float] = []
    paper_raw_values: list[float] = []
    paper_floor_values: list[float] = []
    for task in FIVE_TASKS:
        if task == TASK:
            n = SAMPLE_COUNT
            metric = "exact_match,strict-match"
            values = {
                label: _finite(strict_modes[label]["accuracy"], f"GSM8K {label}")
                for label in ("source", "target", "mapped")
            }
        else:
            base_row = _mapping(base_tasks.get(task), f"base tasks.{task}")
            n = int(base_row["n"])
            metric = str(base_row["metric"])
            values = {
                label: _finite(base_row[label], f"base {task}.{label}")
                for label in ("source", "target", "mapped")
            }
        floor = float(CHANCE_FLOORS[task])
        paper = _paper_values(task)
        raw = values["mapped"] / values["target"] * 100.0
        floor_value = _floor_retention(values["mapped"], values["target"], floor, task)
        paper_raw = paper["mapped"] / paper["target"] * 100.0
        paper_floor = _floor_retention(
            paper["mapped"], paper["target"], floor, f"paper {task}"
        )
        row = {
            "task": task,
            "n": n,
            "metric": metric,
            "chance_floor": floor,
            "source": values["source"],
            "target": values["target"],
            "mapped": values["mapped"],
            "paper": paper,
            "source_delta_from_paper_percentage_points": (
                values["source"] - paper["source"]
            )
            * 100.0,
            "target_delta_from_paper_percentage_points": (
                values["target"] - paper["target"]
            )
            * 100.0,
            "mapped_delta_from_paper_percentage_points": (
                values["mapped"] - paper["mapped"]
            )
            * 100.0,
            "raw_retention_percent": raw,
            "paper_raw_retention_percent": paper_raw,
            "raw_retention_delta_percentage_points": raw - paper_raw,
            "floor_normalized_retention_percent": floor_value,
            "paper_floor_normalized_retention_percent": paper_floor,
            "floor_normalized_retention_delta_percentage_points": (
                floor_value - paper_floor
            ),
        }
        rows.append(row)
        raw_values.append(raw)
        floor_values.append(floor_value)
        paper_raw_values.append(paper_raw)
        paper_floor_values.append(paper_floor)

    raw_summary = {
        "definition": "unweighted arithmetic mean of mapped/target retention over the five accuracy tasks",
        "task_retention_percent": {
            row["task"]: row["raw_retention_percent"] for row in rows
        },
        "reproduced_arithmetic_mean_percent": sum(raw_values) / len(raw_values),
        "paper_task_retention_percent": {
            row["task"]: row["paper_raw_retention_percent"] for row in rows
        },
        "paper_arithmetic_mean_percent": sum(paper_raw_values) / len(paper_raw_values),
        "paper_reported_rounded_percent": 97.6,
    }
    raw_summary["delta_from_paper_unrounded_percentage_points"] = (
        raw_summary["reproduced_arithmetic_mean_percent"]
        - raw_summary["paper_arithmetic_mean_percent"]
    )
    raw_summary["delta_from_paper_reported_percentage_points"] = (
        raw_summary["reproduced_arithmetic_mean_percent"] - 97.6
    )
    floor_summary = {
        "definition": (
            "unweighted arithmetic mean of (mapped-chance)/(target-chance) "
            "retention over the five accuracy tasks"
        ),
        "chance_floors": {task: CHANCE_FLOORS[task] for task in FIVE_TASKS},
        "task_retention_percent": {
            row["task"]: row["floor_normalized_retention_percent"] for row in rows
        },
        "reproduced_arithmetic_mean_percent": sum(floor_values) / len(floor_values),
        "paper_task_retention_percent": {
            row["task"]: row["paper_floor_normalized_retention_percent"] for row in rows
        },
        "paper_arithmetic_mean_percent": sum(paper_floor_values) / len(paper_floor_values),
        "paper_reported_rounded_percent": 96.3,
    }
    floor_summary["delta_from_paper_unrounded_percentage_points"] = (
        floor_summary["reproduced_arithmetic_mean_percent"]
        - floor_summary["paper_arithmetic_mean_percent"]
    )
    floor_summary["delta_from_paper_reported_percentage_points"] = (
        floor_summary["reproduced_arithmetic_mean_percent"] - 96.3
    )
    return rows, raw_summary, floor_summary


def _add_sensitivity_means(
    score_layers: dict[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    four_raw = [float(row["raw_retention_percent"]) for row in rows if row["task"] != TASK]
    four_floor = [
        float(row["floor_normalized_retention_percent"])
        for row in rows
        if row["task"] != TASK
    ]
    for name, layer in score_layers["layers"].items():
        gsm = float(layer["gsm8k_mapped_over_target_retention_percent"])
        layer["five_task_raw_retention_sensitivity_mean_percent"] = (
            sum(four_raw) + gsm
        ) / 5
        layer["five_task_floor_normalized_sensitivity_mean_percent"] = (
            sum(four_floor) + gsm
        ) / 5
        layer["formal_five_task_result"] = name == "canonical_strict_match"


def _csv_bytes(rows: Sequence[Mapping[str, Any]], raw: Mapping[str, Any], floor: Mapping[str, Any]) -> bytes:
    columns = (
        "task",
        "n",
        "metric",
        "chance_floor_percent",
        "source_reproduced_percent",
        "target_reproduced_percent",
        "mapped_reproduced_percent",
        "source_paper_percent",
        "target_paper_percent",
        "mapped_paper_percent",
        "source_delta_from_paper_percentage_points",
        "target_delta_from_paper_percentage_points",
        "mapped_delta_from_paper_percentage_points",
        "raw_retention_percent",
        "paper_raw_retention_percent",
        "raw_retention_delta_percentage_points",
        "floor_normalized_retention_percent",
        "paper_floor_normalized_retention_percent",
        "floor_normalized_retention_delta_percentage_points",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        paper = row["paper"]
        writer.writerow(
            {
                "task": row["task"],
                "n": row["n"],
                "metric": row["metric"],
                "chance_floor_percent": row["chance_floor"] * 100.0,
                "source_reproduced_percent": row["source"] * 100.0,
                "target_reproduced_percent": row["target"] * 100.0,
                "mapped_reproduced_percent": row["mapped"] * 100.0,
                "source_paper_percent": paper["source"] * 100.0,
                "target_paper_percent": paper["target"] * 100.0,
                "mapped_paper_percent": paper["mapped"] * 100.0,
                "source_delta_from_paper_percentage_points": row[
                    "source_delta_from_paper_percentage_points"
                ],
                "target_delta_from_paper_percentage_points": row[
                    "target_delta_from_paper_percentage_points"
                ],
                "mapped_delta_from_paper_percentage_points": row[
                    "mapped_delta_from_paper_percentage_points"
                ],
                "raw_retention_percent": row["raw_retention_percent"],
                "paper_raw_retention_percent": row["paper_raw_retention_percent"],
                "raw_retention_delta_percentage_points": row[
                    "raw_retention_delta_percentage_points"
                ],
                "floor_normalized_retention_percent": row[
                    "floor_normalized_retention_percent"
                ],
                "paper_floor_normalized_retention_percent": row[
                    "paper_floor_normalized_retention_percent"
                ],
                "floor_normalized_retention_delta_percentage_points": row[
                    "floor_normalized_retention_delta_percentage_points"
                ],
            }
        )
    writer.writerow(
        {
            "task": "five_task_arithmetic_mean",
            "metric": "canonical_strict_retention_headline",
            "raw_retention_percent": raw["reproduced_arithmetic_mean_percent"],
            "paper_raw_retention_percent": raw["paper_arithmetic_mean_percent"],
            "raw_retention_delta_percentage_points": raw[
                "delta_from_paper_unrounded_percentage_points"
            ],
            "floor_normalized_retention_percent": floor[
                "reproduced_arithmetic_mean_percent"
            ],
            "paper_floor_normalized_retention_percent": floor[
                "paper_arithmetic_mean_percent"
            ],
            "floor_normalized_retention_delta_percentage_points": floor[
                "delta_from_paper_unrounded_percentage_points"
            ],
        }
    )
    return stream.getvalue().encode("utf-8")


def _markdown_bytes(
    rows: Sequence[Mapping[str, Any]],
    raw: Mapping[str, Any],
    floor: Mapping[str, Any],
    score_layers: Mapping[str, Any],
) -> bytes:
    lines = [
        "# Qwen3-14B → Qwen3-32B formal five-task retention",
        "",
        (
            "Canonical headline: **"
            f"{raw['reproduced_arithmetic_mean_percent']:.6f}% raw retention** and **"
            f"{floor['reproduced_arithmetic_mean_percent']:.6f}% chance-floor-normalized "
            "retention**. GSM8K uses the frozen lm-eval `strict-match` filter."
        ),
        "",
        "| Task | Source | Target | Mapped | Raw retention | Floor-normalized | Paper raw retention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {100 * row['source']:.4f}% | "
            f"{100 * row['target']:.4f}% | {100 * row['mapped']:.4f}% | "
            f"{row['raw_retention_percent']:.4f}% | "
            f"{row['floor_normalized_retention_percent']:.4f}% | "
            f"{row['paper_raw_retention_percent']:.4f}% |"
        )
    lines.extend(
        [
            "",
            (
                f"The reproduced raw mean differs from the paper's unrounded arithmetic "
                f"mean ({raw['paper_arithmetic_mean_percent']:.6f}%) by "
                f"{raw['delta_from_paper_unrounded_percentage_points']:+.6f} percentage "
                f"points, and from the paper-reported rounded 97.6% by "
                f"{raw['delta_from_paper_reported_percentage_points']:+.6f} points."
            ),
            (
                f"The reproduced floor-normalized mean differs from the paper's unrounded "
                f"mean ({floor['paper_arithmetic_mean_percent']:.6f}%) by "
                f"{floor['delta_from_paper_unrounded_percentage_points']:+.6f} percentage "
                f"points, and from the paper-reported rounded 96.3% by "
                f"{floor['delta_from_paper_reported_percentage_points']:+.6f} points."
            ),
            "",
            "## GSM8K scoring layers",
            "",
            "| Layer | Role | Source | Target | Mapped | GSM mapped/target | Five-task sensitivity mean |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("canonical_strict_match", "flexible_extract", "numeric_final_v1"):
        layer = score_layers["layers"][name]
        modes = layer["modes"]
        lines.append(
            f"| {name} | {layer['role']} | {modes['source']['accuracy_percent']:.4f}% | "
            f"{modes['target']['accuracy_percent']:.4f}% | "
            f"{modes['mapped']['accuracy_percent']:.4f}% | "
            f"{layer['gsm8k_mapped_over_target_retention_percent']:.4f}% | "
            f"{layer['five_task_raw_retention_sensitivity_mean_percent']:.4f}% |"
        )
    cap = score_layers["max_generation_token_cap_proxy"]
    cap_text = ", ".join(
        f"{MODE_LABELS[mode]}={cap['modes'][mode]['decoded_completion_reencoded_length_at_or_above_cap']}/{SAMPLE_COUNT}"
        for mode in MODES
    )
    lines.extend(
        [
            "",
            "Flexible-extract and numeric-final-v1 are diagnostics only; neither replaces the canonical strict-match headline or claims to reconstruct the paper's undisclosed GSM8K answer filter.",
            "",
            f"Max-generation-token cap-hit proxy: {cap_text}. This is only a tokenizer re-encode proxy over decoded completions, not a direct generation-time token counter.",
            "",
            "The compatibility receipt and the 1,319-line cross-mode sample-identity ledger are committed beside this report.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _write_file_atomic(path: Path, data: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _publish(output_dir: Path, files: Mapping[str, bytes], generated_utc: str) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise FinalizationError(
            f"output path already exists; fail-closed publication requires a fresh path: {output}"
        )
    if set(files) != OUTPUT_NAMES:
        raise FinalizationError("internal output file set mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent)
    ).resolve()
    try:
        for name in sorted(files):
            _write_file_atomic(staging / name, files[name])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "generated_utc": generated_utc,
            "commit_semantics": "manifest_written_last_then_directory_atomically_renamed",
            "files": {
                name: _sha_metadata(files[name]) for name in sorted(files)
            },
        }
        manifest_bytes = _json_bytes(manifest, pretty=True)
        _write_file_atomic(staging / "sha256_manifest.json", manifest_bytes)
        for name, metadata in manifest["files"].items():
            candidate = staging / name
            if candidate.stat().st_size != metadata["size_bytes"] or sha256_file(candidate) != metadata["sha256"]:
                raise FinalizationError(f"staged output verification failed for {name}")
        os.replace(staging, output)
        return {
            **manifest,
            "path": str(output / "sha256_manifest.json"),
            "sha256": sha256_file(output / "sha256_manifest.json"),
        }
    except BaseException:
        if staging.exists() and staging.parent == output.parent and staging.name.startswith(
            f".{output.name}.staging."
        ):
            shutil.rmtree(staging)
        raise


def finalize(
    *,
    source_run: Path,
    target_run: Path,
    mapped_run: Path,
    base_report_path: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if output_dir.resolve().exists():
        raise FinalizationError(
            f"output path already exists; choose a fresh output directory: {output_dir.resolve()}"
        )
    generated_utc = datetime.now(timezone.utc).isoformat()
    base, base_provenance = _validate_base_report(base_report_path)
    run_dirs = {
        "source_standalone": source_run,
        "target_standalone": target_run,
        "mapped": mapped_run,
    }
    records, scorer_reports = _validate_runs(run_dirs)
    project_transition = _validate_project_code(records)
    runtime_identity, tokenizer_path = _validate_runtime_identity(records, base)

    sample_sets = {
        mode: _strict_samples(_checkpoint_path(records[mode], mode), mode) for mode in MODES
    }
    identity_rows, identity_assurance = _cross_mode_identity(sample_sets)
    identity_jsonl = b"".join(_json_bytes(row) + b"\n" for row in identity_rows)
    if identity_jsonl.count(b"\n") != SAMPLE_COUNT:
        raise FinalizationError("cross-mode identity ledger must contain exactly 1319 lines")
    cap_proxy = _cap_proxies(sample_sets, tokenizer_path)
    if (
        cap_proxy["modes"]["mapped"][
            "decoded_completion_reencoded_length_at_or_above_cap"
        ]
        != 269
    ):
        raise FinalizationError("mapped cap-hit proxy no longer matches the pinned base audit")

    score_layers = _score_layers(scorer_reports, cap_proxy)
    base_gsm = _mapping(base["tasks"][TASK], "base GSM8K")
    mapped_strict = score_layers["layers"]["canonical_strict_match"]["modes"]["mapped"]
    mapped_flexible = score_layers["layers"]["flexible_extract"]["modes"]["mapped"]
    _close(mapped_strict["accuracy"], float(base_gsm["mapped_max1024"]), "mapped strict/base")
    _close(
        mapped_flexible["accuracy"],
        float(base_gsm["mapped_max1024_flexible"]),
        "mapped flexible/base",
    )

    rows, raw_summary, floor_summary = _build_formal_rows(base, score_layers)
    _add_sensitivity_means(score_layers, rows)
    score_layers.update(
        {
            "generated_utc": generated_utc,
            "scorer_implementation_files_digest": EXPECTED_SCORER_FILES_DIGEST,
            "paper_answer_filter_disclosure": (
                "The paper reports GSM8K accuracy but does not disclose the exact answer "
                "filter; only local strict-match is formal here."
            ),
        }
    )

    shared_definition = _shared_definition(records[MODES[0]].definition)
    run_provenance = {}
    for mode in MODES:
        record = records[mode]
        checkpoint = record.checkpoints[0]
        status_path = record.run_dir / "status.json"
        summary_path = record.run_dir / "summary.json"
        definition_path = record.run_dir / "run_definition.json"
        run_provenance[mode] = {
            "run_dir": str(record.run_dir),
            "request_fingerprint": record.fingerprint,
            "run_definition_sha256": sha256_file(definition_path),
            "status_sha256": sha256_file(status_path),
            "summary_sha256": sha256_file(summary_path),
            "checkpoint_path": str(checkpoint.path),
            "checkpoint_sha256": checkpoint.sha256,
            "status_complete": True,
            "failures": 0,
            "sample_count": SAMPLE_COUNT,
            "filter_row_count": FILTER_ROW_COUNT,
            "strict_filter_rows": SAMPLE_COUNT,
            "flexible_filter_rows": SAMPLE_COUNT,
            "validate_run_passed": True,
            "score_checkpoint_passed": True,
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_utc": generated_utc,
        "direction": "Qwen3-14B_to_Qwen3-32B",
        "formal_full_dataset": True,
        "canonical_primary_scoring": "GSM8K exact_match,strict-match",
        "headline": {
            "metric": "five_task_raw_mapped_over_target_retention_arithmetic_mean_percent",
            "value": raw_summary["reproduced_arithmetic_mean_percent"],
            "paper_reported_rounded_percent": 97.6,
            "delta_from_paper_reported_percentage_points": raw_summary[
                "delta_from_paper_reported_percentage_points"
            ],
        },
        "five_task_rows": rows,
        "five_task_raw_retention": raw_summary,
        "five_task_floor_normalized_retention": floor_summary,
        "diagnostics": {
            "file": "score_layers.json",
            "flexible_extract_and_numeric_final_v1_are_not_formal_headlines": True,
            "cap_hit_counts_are_tokenizer_reencode_proxies_only": True,
        },
        "compatibility_receipt": "compatibility_receipt.json",
        "cross_mode_sample_identity": "cross_mode_sample_identity.jsonl",
    }
    score_bytes = _json_bytes(score_layers, pretty=True)
    csv_bytes = _csv_bytes(rows, raw_summary, floor_summary)
    markdown_bytes = _markdown_bytes(rows, raw_summary, floor_summary, score_layers)
    report_bytes = _json_bytes(report, pretty=True)

    compatibility_receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_utc": generated_utc,
        "scope": "task-scoped GSM8K max_gen_tokens=1024 compatibility bridge",
        "decision": "compatible_for_formal_local_strict_match_five_task_aggregation",
        "base_audited_report": base_provenance,
        "runs": run_provenance,
        "run_definition": {
            "shared_definition_exact_across_all_three_modes": True,
            "shared_definition_sha256": _digest(shared_definition),
            "explicit_partition_fields": ["modes", "runtime_artifacts", "project_code"],
            "derived_partition_field": "request_fingerprint",
            "resolved_task_config_exact_across_all_three_modes": True,
            "max_gen_tokens": MAX_GEN_TOKENS,
            "log_samples": True,
            "limit": None,
        },
        "project_code_transition": project_transition,
        "runtime_identity": runtime_identity,
        "sample_identity": {
            **identity_assurance,
            "ledger_file": "cross_mode_sample_identity.jsonl",
            "ledger_sha256": _sha256_bytes(identity_jsonl),
            "ledger_size_bytes": len(identity_jsonl),
            "doc_doc_hash_prompt_hash_target_target_hash_arguments_compared_individually": True,
        },
        "scoring": {
            "validate_run_called_for_all_modes": True,
            "score_checkpoint_called_for_all_modes": True,
            "scorer_implementation_files_digest": EXPECTED_SCORER_FILES_DIGEST,
            "canonical_formal_layer": "exact_match,strict-match",
            "diagnostic_only_layers": [
                "exact_match,flexible-extract",
                NUMERIC_METRIC_NAME,
            ],
            "score_layers_file_sha256": _sha256_bytes(score_bytes),
        },
        "derived_outputs": {
            "report.json": _sha_metadata(report_bytes),
            "five_task_strict_retention.csv": _sha_metadata(csv_bytes),
            "five_task_strict_retention.md": _sha_metadata(markdown_bytes),
            "score_layers.json": _sha_metadata(score_bytes),
        },
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    receipt_bytes = _json_bytes(compatibility_receipt, pretty=True)
    files = {
        "compatibility_receipt.json": receipt_bytes,
        "cross_mode_sample_identity.jsonl": identity_jsonl,
        "five_task_strict_retention.csv": csv_bytes,
        "five_task_strict_retention.md": markdown_bytes,
        "report.json": report_bytes,
        "score_layers.json": score_bytes,
    }
    manifest = _publish(output_dir, files, generated_utc)
    return report, manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--target-run", type=Path, required=True)
    parser.add_argument(
        "--mapped-run",
        type=Path,
        default=ROOT / "runs/20260814_062228_p7_mapped_gsm_max1024",
        help="pinned historical complete mapped max1024 run",
    )
    parser.add_argument(
        "--base-report",
        type=Path,
        default=(
            ROOT
            / "artifacts/evaluation/qwen3_14b_to_32b/audited_available_results.json"
        ),
        help="pinned audited four-task JSON (its sibling SHA256 sidecar is required)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="fresh, nonexistent directory for the atomic committed result",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report, manifest = finalize(
            source_run=args.source_run,
            target_run=args.target_run,
            mapped_run=args.mapped_run,
            base_report_path=args.base_report,
            output_dir=args.output_dir,
        )
    except (FinalizationError, AggregationError, GSM8KNumericDiagnosticError) as exc:
        print(f"five-task finalization refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.resolve()),
                "canonical_gsm8k_scoring": report["canonical_primary_scoring"],
                "five_task_raw_retention_percent": report["five_task_raw_retention"][
                    "reproduced_arithmetic_mean_percent"
                ],
                "five_task_floor_normalized_retention_percent": report[
                    "five_task_floor_normalized_retention"
                ]["reproduced_arithmetic_mean_percent"],
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
