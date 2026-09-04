"""Independent CPU finalizer for reverse long-context smoke and full runs."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
import re
import statistics
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from transformers import AutoTokenizer

from . import metrics as extension_metrics
from .adapter import configured_base_runner
from .freeze import CODE_FREEZE, CODE_FREEZE_SIDECAR, validate_code_freeze
from .protocol import (
    CONFIG_PATH,
    DIALOG_REFERENCE_SCORE_CAP_TOKENS,
    DIRECTION,
    EXPECTED_CAPS,
    EXPECTED_MAPPER_FIT_FINGERPRINT,
    EXPECTED_MAPPER_MANIFEST_SHA256,
    EXPECTED_RESULT_COUNT,
    EXPECTED_SAMPLE_COUNTS,
    EXPECTED_SMOKE_RESULT_COUNT,
    EXPECTED_TARGET_VOCAB_SIZE,
    EXTENSION_VERSION,
    FORMAL_TOPOLOGY,
    MAPPER_MANIFEST,
    MAX_LENGTH,
    MODES,
    PAPER_BIT_EXACT,
    PROTOCOL_NAME,
    SMOKE_ENDPOINTS,
    SMOKE_SAMPLE_INDICES,
    SOURCE_MODEL,
    TARGET_MODEL,
    TASKS,
    output_within_experiment,
)

from reverse_kv.eval_common import hashed_json, verify_hashed_json
from reverse_kv.protocol import canonical_sha256, read_json, sha256_file
from reverse_kv.runtime import ReverseMapperBundle


FULL_RECEIPT_NAME = "long_context_extension_receipt.json"
SMOKE_RECEIPT_NAME = "long_context_extension_smoke_receipt.json"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_GATE_NAME = re.compile(r"gate_([0-9]{4})\.json")


class ExtensionValidationError(ValueError):
    """A formal long-context artifact violates the frozen protocol."""


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExtensionValidationError(f"{label} must be a mapping")
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


def _artifact(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    if path.is_symlink():
        raise ExtensionValidationError(f"artifact is symlinked: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ExtensionValidationError(f"artifact is missing or symlinked: {resolved}")
    return {
        "path": (
            str(resolved.relative_to(base.resolve())) if base is not None else str(resolved)
        ),
        "sha256": sha256_file(resolved),
        "nbytes": resolved.stat().st_size,
    }


def _profile_args(profile: str, output: Path) -> SimpleNamespace:
    if profile not in {"full", "smoke"}:
        raise ExtensionValidationError(f"unsupported validation profile: {profile}")
    return SimpleNamespace(
        config=CONFIG_PATH,
        tasks=TASKS,
        output_dir=output,
        limit=None,
        sample_indices=(dict(SMOKE_SAMPLE_INDICES) if profile == "smoke" else None),
        prepare_only=False,
    )


def _expected_scope(profile: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": (
            "frozen_explicit_two_indices_per_task" if profile == "smoke" else "formal_full"
        ),
        "tasks": list(TASKS),
        "smoke_sample_indices": (
            {task: list(indices) for task, indices in SMOKE_SAMPLE_INDICES.items()}
            if profile == "smoke"
            else None
        ),
        "smoke_endpoints": (
            {
                task: [dict(endpoint) for endpoint in endpoints]
                for task, endpoints in SMOKE_ENDPOINTS.items()
            }
            if profile == "smoke"
            else None
        ),
        "expected_atomic_result_count": (
            EXPECTED_SMOKE_RESULT_COUNT
            if profile == "smoke"
            else EXPECTED_RESULT_COUNT
        ),
    }


def _utc(value: Any, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ExtensionValidationError(f"{label} timestamp is malformed") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ExtensionValidationError(f"{label} timestamp is not UTC-aware")
    return parsed


def _validate_generated_tokens_independent(
    payload: Mapping[str, Any], *, tokenizer: Any, expected_cap: int
) -> list[int]:
    """Revalidate persisted generation tokens without the adapter/base helper."""

    generation = _mapping(payload.get("generation"), "generation evidence")
    expected_schema = {
        "requested_max_tokens",
        "generated_tokens",
        "stop_reason",
        "cap_hit",
        "generated_token_ids",
        "generated_token_ids_sha256",
    }
    if set(generation) != expected_schema:
        raise ExtensionValidationError("generation evidence schema changed")
    token_ids = _list(generation.get("generated_token_ids"), "generated token IDs")
    if any(type(token) is not int for token in token_ids):
        raise ExtensionValidationError("generated token IDs are not exact integers")
    if any(token < 0 or token >= EXPECTED_TARGET_VOCAB_SIZE for token in token_ids):
        raise ExtensionValidationError("generated token ID exceeds target vocab_size")
    if generation.get("requested_max_tokens") != expected_cap:
        raise ExtensionValidationError("requested generation cap changed")
    if generation.get("generated_tokens") != len(token_ids):
        raise ExtensionValidationError("generated token count/list length changed")
    if generation.get("generated_token_ids_sha256") != canonical_sha256(token_ids):
        raise ExtensionValidationError("generated token ID hash changed")
    stop_reason = generation.get("stop_reason")
    cap_hit = generation.get("cap_hit")
    if stop_reason == "max_tokens":
        if len(token_ids) != expected_cap or cap_hit is not True:
            raise ExtensionValidationError(
                "max_tokens requires exactly cap tokens and cap_hit=true"
            )
    elif stop_reason == "eos":
        if len(token_ids) >= expected_cap or cap_hit is not False:
            raise ExtensionValidationError(
                "eos requires fewer than cap tokens and cap_hit=false"
            )
    else:
        raise ExtensionValidationError("until=[] permits only eos or max_tokens")
    try:
        prediction = str(tokenizer.decode(token_ids, skip_special_tokens=True))
    except TypeError:
        prediction = str(tokenizer.decode(token_ids))
    if payload.get("prediction") != prediction:
        raise ExtensionValidationError(
            "prediction is not the exact decode of generated token IDs"
        )
    return [int(token) for token in token_ids]


def _metric_dict_matches(
    observed: Any, expected: Mapping[str, Any], *, label: str
) -> None:
    value = _mapping(observed, label)
    if set(value) != set(expected):
        raise ExtensionValidationError(f"{label} schema changed")
    for name, expected_value in expected.items():
        observed_value = value.get(name)
        if isinstance(expected_value, (int, float)) and not isinstance(
            expected_value, bool
        ):
            if _finite(observed_value, f"{label}.{name}") != _finite(
                expected_value, f"recomputed {label}.{name}"
            ):
                raise ExtensionValidationError(f"{label}.{name} was not recomputed")
        elif observed_value != expected_value:
            raise ExtensionValidationError(f"{label}.{name} was not recomputed")


def _fingerprint(definition: Mapping[str, Any]) -> str:
    declared = definition.get("request_fingerprint")
    unsigned = dict(definition)
    unsigned.pop("request_fingerprint", None)
    if not isinstance(declared, str) or _HEX64.fullmatch(declared) is None:
        raise ExtensionValidationError("request fingerprint is malformed")
    if canonical_sha256(unsigned) != declared:
        raise ExtensionValidationError("request fingerprint disagrees with definition")
    return declared


def validate_gpu_idle_gate(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ExtensionValidationError("GPU idle gate is symlinked")
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
    if set(value) != {*fixed, "samples"}:
        raise ExtensionValidationError("GPU idle gate schema changed")
    samples = _list(value.get("samples"), "GPU idle samples")
    if len(samples) < 12:
        raise ExtensionValidationError("GPU idle gate lacks twelve samples")
    trailing = 0
    previous_utc: dt.datetime | None = None
    for index, raw in enumerate(samples):
        sample = _mapping(raw, "GPU idle sample")
        if set(sample) != {
            "sample_index",
            "utc",
            "passed",
            "memory",
            "compute_processes",
        }:
            raise ExtensionValidationError("GPU idle sample schema changed")
        sample_utc = _utc(sample.get("utc"), "GPU idle sample")
        if previous_utc is not None and sample_utc < previous_utc:
            raise ExtensionValidationError("GPU idle timestamps are reordered")
        previous_utc = sample_utc
        memory = _list(sample.get("memory"), "GPU idle memory")
        processes = sample.get("compute_processes")
        passed = bool(
            len(memory) == 8
            and [
                int(_mapping(row, "GPU memory row").get("physical_index", -1))
                for row in memory
            ]
            == list(range(8))
            and all(
                set(_mapping(row, "GPU memory row"))
                == {"physical_index", "memory_used_mib", "memory_total_mib", "uuid"}
                and type(row.get("memory_used_mib")) is int
                and 0 <= int(row["memory_used_mib"]) <= 64
                and type(row.get("memory_total_mib")) is int
                and int(row["memory_total_mib"]) > 0
                and isinstance(row.get("uuid"), str)
                and bool(row["uuid"])
                for row in memory
            )
            and processes == []
        )
        if sample.get("sample_index") != index or sample.get("passed") is not passed:
            raise ExtensionValidationError("GPU idle sample verdict/order changed")
        trailing = trailing + 1 if passed else 0
    if trailing != 12:
        raise ExtensionValidationError("GPU idle gate trailing pass count changed")
    sidecar = path.with_name(f"{path.name}.sha256.json")
    return {
        "artifact": _artifact(path),
        "sidecar": _artifact(sidecar),
    }


def validate_gpu_idle_gates(output: Path) -> list[dict[str, Any]]:
    directory = output / "long_context_gpu_gates"
    if not directory.is_dir() or directory.is_symlink():
        raise ExtensionValidationError("GPU gate directory is missing or symlinked")
    gates = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and _GATE_NAME.fullmatch(path.name)
    )
    expected_names = [f"gate_{index:04d}.json" for index in range(1, len(gates) + 1)]
    if not gates or [path.name for path in gates] != expected_names:
        raise ExtensionValidationError("GPU gates are absent, incomplete, or reordered")
    expected_files = set(expected_names) | {
        f"{name}.sha256.json" for name in expected_names
    }
    if {path.name for path in directory.iterdir()} != expected_files:
        raise ExtensionValidationError("GPU gate directory contains unexpected files")
    return [
        {"gate_index": index, **validate_gpu_idle_gate(path)}
        for index, path in enumerate(gates, start=1)
    ]


def _validate_model_sharding(output: Path) -> dict[str, Any]:
    path = output / "model_sharding.json"
    value = read_json(path)
    if set(value) != {"source", "target"}:
        raise ExtensionValidationError("model sharding roles changed")
    expected_schema = {
        "status",
        "role",
        "mode",
        "devices",
        "declared_layer_counts",
        "expected_layer_counts",
        "actual_layer_counts",
        "total_layers",
        "layer_devices",
        "contiguous_ranges",
        "actual_parameters_inspected_for_every_layer",
        "parameter_tensors_inspected",
        "auxiliary_modules",
        "auxiliary_parameter_tensors_inspected",
        "auxiliary_parameters_inspected_for_every_module",
    }
    for semantic_role, contract in FORMAL_TOPOLOGY.items():
        row = _mapping(value.get(semantic_role), f"{semantic_role} sharding")
        if set(row) != expected_schema:
            raise ExtensionValidationError(
                f"{semantic_role} sharding evidence schema changed"
            )
        expected_native_role = {"source": "target", "target": "source"}[
            semantic_role
        ]
        counts = list(contract["layer_counts"])
        devices = list(contract["devices"])
        expanded = [
            device
            for device, count in zip(devices, counts, strict=True)
            for _ in range(count)
        ]
        ranges: list[dict[str, int]] = []
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
        expected_layers = 64 if semantic_role == "source" else 40
        checks = {
            "status": "passed",
            "role": expected_native_role,
            "mode": "explicit_exact_counts",
            "devices": devices,
            "declared_layer_counts": counts,
            "expected_layer_counts": counts,
            "actual_layer_counts": counts,
            "total_layers": expected_layers,
            "layer_devices": expanded,
            "contiguous_ranges": ranges,
            "actual_parameters_inspected_for_every_layer": True,
            "auxiliary_parameters_inspected_for_every_module": True,
        }
        for field, expected in checks.items():
            if row.get(field) != expected:
                raise ExtensionValidationError(
                    f"{semantic_role} sharding changed {field}"
                )
        inspected = row.get("parameter_tensors_inspected")
        auxiliary_inspected = row.get("auxiliary_parameter_tensors_inspected")
        if inspected != expected_layers * 11:
            raise ExtensionValidationError(
                f"{semantic_role} layer parameter inspection count changed"
            )
        if auxiliary_inspected != 3:
            raise ExtensionValidationError(
                f"{semantic_role} auxiliary inspection count changed"
            )
        expected_auxiliary = {
            "model.embed_tokens": devices[0],
            "model.norm": devices[-1],
            "lm_head": devices[-1],
        }
        auxiliary = _mapping(
            row.get("auxiliary_modules"), f"{semantic_role} auxiliary modules"
        )
        if set(auxiliary) != set(expected_auxiliary):
            raise ExtensionValidationError(
                f"{semantic_role} auxiliary module coverage changed"
            )
        recomputed_auxiliary = 0
        for module_name, expected_device in expected_auxiliary.items():
            module = _mapping(
                auxiliary[module_name],
                f"{semantic_role} auxiliary module {module_name}",
            )
            if set(module) != {
                "expected_device",
                "actual_device",
                "parameter_tensors_inspected",
            }:
                raise ExtensionValidationError(
                    f"{semantic_role} auxiliary module schema changed"
                )
            count = module.get("parameter_tensors_inspected")
            if (
                module.get("expected_device") != expected_device
                or module.get("actual_device") != expected_device
                or count != 1
            ):
                raise ExtensionValidationError(
                    f"{semantic_role} auxiliary placement proof changed"
                )
            recomputed_auxiliary += count
        if recomputed_auxiliary != auxiliary_inspected:
            raise ExtensionValidationError(
                f"{semantic_role} auxiliary inspection total changed"
            )
    return _artifact(path)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_latency_summary(
    value: Any,
    *,
    label: str,
    expected_samples: Sequence[float],
    timing_backends: Sequence[str] | None = None,
) -> None:
    summary = _mapping(value, label)
    expected_schema = {
        "unit",
        "count",
        "mean",
        "median",
        "p10",
        "p50",
        "p90",
        "std_population",
        "min",
        "max",
        "samples",
    }
    if timing_backends is not None:
        expected_schema.add("timing_backends")
    if set(summary) != expected_schema:
        raise ExtensionValidationError(f"{label} timing schema changed")
    samples = _list(summary.get("samples"), f"{label} timing samples")
    observed = [_finite(value, f"{label} timing sample") for value in samples]
    expected = [float(value) for value in expected_samples]
    if len(observed) != 40 or observed != expected or any(value < 0 for value in observed):
        raise ExtensionValidationError(f"{label} timing samples changed")
    recomputed = {
        "unit": "ms",
        "count": len(observed),
        "mean": float(statistics.fmean(observed)),
        "median": float(statistics.median(observed)),
        "p10": float(_percentile(observed, 0.10)),
        "p50": float(_percentile(observed, 0.50)),
        "p90": float(_percentile(observed, 0.90)),
        "std_population": float(statistics.pstdev(observed)),
        "min": float(min(observed)),
        "max": float(max(observed)),
        "samples": observed,
    }
    if timing_backends is not None:
        recomputed["timing_backends"] = list(timing_backends)
    if dict(summary) != recomputed:
        raise ExtensionValidationError(f"{label} timing summary was not recomputed")


def _runtime_model_artifact(
    identity: Mapping[str, Any], *, expected_path: Path
) -> dict[str, str]:
    files = _mapping(identity.get("content_hashed_files"), "model content identity")
    return {
        "model_path": str(expected_path.resolve()),
        "model_config_sha256": str(
            _mapping(files.get("config.json"), "model config identity").get("sha256")
        ),
        "model_weight_index_sha256": str(
            _mapping(
                files.get("model.safetensors.index.json"), "model index identity"
            ).get("sha256")
        ),
        "model_tokenizer_sha256": str(
            _mapping(files.get("tokenizer.json"), "model tokenizer identity").get(
                "sha256"
            )
        ),
    }


def _validate_mapper_runtime(
    output: Path, *, expected_definition: Mapping[str, Any]
) -> dict[str, Any]:
    path = output / "mapper.json"
    value = read_json(path)
    expected_schema = {
        "direction",
        "manifest",
        "manifest_sha256",
        "fit_fingerprint",
        "layers",
        "selected_source_layers",
        "token_chunk",
        "execution",
        "target_layer_devices",
        "resident_parameter_bytes",
        "materialization",
        "source_model_artifact",
        "target_model_artifact",
        "source_capture_model_identity",
        "target_capture_model_identity",
    }
    if set(value) != expected_schema:
        raise ExtensionValidationError("mapper runtime schema changed")
    fixed = {
        "direction": DIRECTION,
        "manifest": str(MAPPER_MANIFEST.resolve()),
        "manifest_sha256": EXPECTED_MAPPER_MANIFEST_SHA256,
        "fit_fingerprint": EXPECTED_MAPPER_FIT_FINGERPRINT,
        "layers": 40,
        "selected_source_layers": 20,
        "token_chunk": 1024,
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ExtensionValidationError(f"mapper runtime changed {field}")
    expected_devices = [
        f"cuda:{device}"
        for device, count in zip(
            FORMAL_TOPOLOGY["target"]["devices"],
            FORMAL_TOPOLOGY["target"]["layer_counts"],
            strict=True,
        )
        for _ in range(count)
    ]
    if value.get("target_layer_devices") != expected_devices:
        raise ExtensionValidationError("mapper target layer devices changed")
    manifest = read_json(MAPPER_MANIFEST)
    expected_bytes = 4 * (
        int(manifest.get("weight_count", -1)) + int(manifest.get("bias_count", -1))
    )
    if value.get("resident_parameter_bytes") != expected_bytes:
        raise ExtensionValidationError("resident mapper byte count changed")
    if type(value.get("resident_parameter_bytes")) is not int:
        raise ExtensionValidationError("resident mapper byte count is not an integer")
    materialization = _mapping(value.get("materialization"), "mapper materialization")
    if set(materialization) != {
        "scope",
        "host_load_and_hash",
        "weight_host_to_device",
        "per_layer",
    }:
        raise ExtensionValidationError("mapper materialization schema changed")
    if materialization.get("scope") != "one_time_setup_excluded_from_per_request_latency":
        raise ExtensionValidationError("mapper materialization timing scope changed")
    per_layer = _list(materialization.get("per_layer"), "mapper per-layer materialization")
    if len(per_layer) != 40:
        raise ExtensionValidationError("mapper materialization coverage changed")
    host_samples: list[float] = []
    transfer_samples: list[float] = []
    for layer, row_raw in enumerate(per_layer):
        row = _mapping(row_raw, "mapper materialization row")
        if set(row) != {
            "target_layer",
            "device",
            "host_load_and_hash_ms",
            "weight_host_to_device_ms",
        }:
            raise ExtensionValidationError("mapper materialization row schema changed")
        if row.get("target_layer") != layer or row.get("device") != expected_devices[layer]:
            raise ExtensionValidationError("mapper materialization layer/device changed")
        host = _finite(row.get("host_load_and_hash_ms"), "mapper host load timing")
        transfer = _finite(
            row.get("weight_host_to_device_ms"), "mapper device transfer timing"
        )
        if host < 0 or transfer < 0:
            raise ExtensionValidationError("mapper materialization timing is negative")
        host_samples.append(host)
        transfer_samples.append(transfer)
    _validate_latency_summary(
        materialization.get("host_load_and_hash"),
        label="mapper host load/hash",
        expected_samples=host_samples,
    )
    _validate_latency_summary(
        materialization.get("weight_host_to_device"),
        label="mapper host-to-device",
        expected_samples=transfer_samples,
        timing_backends=("host_perf_counter_with_cuda_synchronize",),
    )
    # Opening the strict bundle rereads and verifies every committed layer
    # payload/sidecar, selection snapshot, fit fingerprint, and model binding.
    bundle = ReverseMapperBundle.open(MAPPER_MANIFEST.parent)
    if bundle.manifest_sha256 != EXPECTED_MAPPER_MANIFEST_SHA256:
        raise ExtensionValidationError("deep mapper validation changed manifest")
    source_capture = dict(bundle.source_model_identity)
    target_capture = dict(bundle.target_model_identity)
    if value.get("source_capture_model_identity") != source_capture:
        raise ExtensionValidationError("mapper source capture identity changed")
    if value.get("target_capture_model_identity") != target_capture:
        raise ExtensionValidationError("mapper target capture identity changed")
    capture_fixed = (
        (source_capture, SOURCE_MODEL, "qwen3_32b", "target"),
        (target_capture, TARGET_MODEL, "qwen3_14b", "source"),
    )
    for capture, model_path, model_name, native_role in capture_fixed:
        if (
            capture.get("model_path") != str(model_path.resolve())
            or capture.get("model_name") != model_name
            or capture.get("model_role") != native_role
        ):
            raise ExtensionValidationError("mapper semantic/capture model role changed")
    expected_source_artifact = _runtime_model_artifact(
        _mapping(expected_definition.get("source_model"), "source model identity"),
        expected_path=SOURCE_MODEL,
    )
    expected_target_artifact = _runtime_model_artifact(
        _mapping(expected_definition.get("target_model"), "target model identity"),
        expected_path=TARGET_MODEL,
    )
    if value.get("source_model_artifact") != expected_source_artifact:
        raise ExtensionValidationError("runtime source model identity changed")
    if value.get("target_model_artifact") != expected_target_artifact:
        raise ExtensionValidationError("runtime target model identity changed")
    return {
        "artifact": _artifact(path),
        "deep_payload_validation": True,
        "target_layers": len(bundle.layers),
        "fit_fingerprint": bundle.fit_fingerprint,
    }


def _result_inventory_paths(
    output: Path, expected_relative_paths: Sequence[str]
) -> None:
    lexical_root = output / "results"
    if lexical_root.is_symlink():
        raise ExtensionValidationError("result directory is symlinked")
    root = lexical_root.resolve()
    if not root.is_dir():
        raise ExtensionValidationError("result directory is missing or symlinked")
    expected = set(expected_relative_paths)
    if len(expected) != len(expected_relative_paths):
        raise ExtensionValidationError("expected result inventory contains duplicates")
    for relative in expected:
        candidate = (output / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ExtensionValidationError("result inventory path escaped results") from exc
    expected_directories: set[str] = set()
    for relative in expected:
        parent = Path(relative).parent
        while str(parent) not in {".", "results"}:
            expected_directories.add(str(parent))
            parent = parent.parent
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in lexical_root.rglob("*"):
        # ``exists`` is false for dangling links, so identity must be checked
        # first.  Directory links are rejected even though pathlib does not
        # recurse through them by default.
        if path.is_symlink():
            raise ExtensionValidationError(f"result tree contains a symlink: {path}")
        if path.is_file():
            actual.add(str(path.resolve().relative_to(output.resolve())))
        elif path.is_dir():
            actual_directories.add(
                str(path.resolve().relative_to(output.resolve()))
            )
        else:
            raise ExtensionValidationError(
                f"result tree contains a non-regular entry: {path}"
            )
    if actual != expected:
        raise ExtensionValidationError("result directory has missing or unexpected files")
    if actual_directories != expected_directories:
        raise ExtensionValidationError(
            "result directory has missing or unexpected subdirectories"
        )


_CHECKPOINT_SCHEMA = {
    "schema_version",
    "status",
    "request_fingerprint",
    "task",
    "mode",
    "sample_index",
    "sample_id",
    "record_sha256",
    "source_length",
    "primary_metric",
    "started_utc",
    "finished_utc",
    "prediction",
    "references",
    "metrics",
    "generation",
    "prompt",
    "reference_scoring",
    "target_checkpoint_binding",
}


def _validate_checkpoint_envelope(payload: Mapping[str, Any], *, label: str) -> None:
    if set(payload) != _CHECKPOINT_SCHEMA:
        raise ExtensionValidationError(f"{label} checkpoint schema changed")
    started = _utc(payload.get("started_utc"), f"{label} started")
    finished = _utc(payload.get("finished_utc"), f"{label} finished")
    if finished < started:
        raise ExtensionValidationError(f"{label} finished before it started")


def _validate_results(
    *,
    base: Any,
    output: Path,
    configuration: Mapping[str, Any],
    samples: Mapping[str, Sequence[Any]],
    fingerprint: str,
    tokenizer: Any,
) -> dict[str, Any]:
    tasks_summary: dict[str, Any] = {}
    result_files: list[dict[str, Any]] = []
    expected_paths: list[str] = []
    for task in TASKS:
        task_summary: dict[str, Any] = {
            "expected_samples": len(samples[task]),
            "modes": {},
        }
        for mode in MODES:
            metric_values: dict[str, list[float]] = {}
            cap_hits = 0
            total_ref_ll = 0.0
            total_ref_tokens = 0
            for sample in samples[task]:
                path = base.result_path(output, task, mode, sample.index)
                relative = str(path.resolve().relative_to(output.resolve()))
                expected_paths.append(relative)
                target_payload: Mapping[str, Any] | None = None
                if mode == "mapped":
                    target_path = base.result_path(
                        output, task, "target_standalone", sample.index
                    )
                    if not target_path.is_file():
                        raise ExtensionValidationError(
                            f"mapped result lacks target checkpoint: {task}/{sample.index}"
                        )
                    target_payload = _mapping(
                        read_json(target_path), "target checkpoint"
                    )
                if not base.checkpoint_complete(
                    path,
                    fingerprint=fingerprint,
                    task=task,
                    mode=mode,
                    sample=sample,
                    tokenizer=tokenizer,
                    max_length=MAX_LENGTH,
                    reference_score_cap=DIALOG_REFERENCE_SCORE_CAP_TOKENS,
                    target_checkpoint=target_payload,
                ):
                    raise ExtensionValidationError(
                        f"checkpoint failed full recomputation: {task}/{sample.index}/{mode}"
                    )
                failure = path.with_suffix(".failure.json")
                if failure.exists():
                    raise ExtensionValidationError(f"completed checkpoint retains {failure}")
                payload = read_json(path)
                label = f"{task}/{sample.index}/{mode}"
                _validate_checkpoint_envelope(payload, label=label)
                if mode == "mapped" and target_payload is not None:
                    if _utc(
                        payload.get("started_utc"), f"{label} started"
                    ) < _utc(
                        target_payload.get("finished_utc"),
                        f"{task}/{sample.index}/target finished",
                    ):
                        raise ExtensionValidationError(
                            "mapped checkpoint started before its target checkpoint finished"
                        )
                _validate_generated_tokens_independent(
                    payload,
                    tokenizer=tokenizer,
                    expected_cap=EXPECTED_CAPS[task],
                )
                result_files.append(
                    {
                        "path": relative,
                        "sha256": sha256_file(path),
                        "nbytes": path.stat().st_size,
                    }
                )
                cap_hits += int(bool(payload["generation"]["cap_hit"]))
                prediction = payload.get("prediction")
                references = _list(payload.get("references"), f"{label} references")
                if not isinstance(prediction, str) or references != list(sample.references):
                    raise ExtensionValidationError(
                        f"{label} raw prediction/reference identity changed"
                    )
                recomputed_metrics = extension_metrics.sample_metrics(
                    base, task, prediction, references
                )
                reference = payload.get("reference_scoring")
                if task in ("lmsys", "sharegpt4"):
                    reference = _mapping(reference, f"{label} reference scoring")
                    loglikelihood = _finite(
                        reference.get("reference_loglikelihood"),
                        f"{label} reference loglikelihood",
                    )
                    scored = reference.get("reference_scored_tokens")
                    if isinstance(scored, bool) or not isinstance(scored, int) or scored <= 0:
                        raise ExtensionValidationError("reference token count changed")
                    nll = -loglikelihood
                    ppl = math.exp(nll / scored)
                    if not math.isfinite(ppl):
                        raise ExtensionValidationError("reference perplexity is not finite")
                    if _finite(
                        reference.get("reference_nll"), f"{label} reference NLL"
                    ) != nll:
                        raise ExtensionValidationError("reference NLL algebra changed")
                    if _finite(
                        reference.get("reference_ppl"), f"{label} reference PPL"
                    ) != ppl:
                        raise ExtensionValidationError("reference PPL algebra changed")
                    recomputed_metrics.update(
                        {"reference_nll": nll, "reference_ppl": ppl}
                    )
                    if mode == "mapped":
                        if target_payload is None:
                            raise ExtensionValidationError(
                                "mapped dialogue lacks target payload"
                            )
                        target_prediction = target_payload.get("prediction")
                        if not isinstance(target_prediction, str):
                            raise ExtensionValidationError(
                                "mapped dialogue target prediction changed"
                            )
                        recomputed_metrics.update(
                            {
                                "output_retention_rouge_l_f1": base.rouge_l_f1(
                                    prediction, [target_prediction]
                                ),
                                "output_retention_chrf": base.chrf_score(
                                    prediction, [target_prediction]
                                ),
                                "output_exact_match": float(
                                    prediction == target_prediction
                                ),
                            }
                        )
                    total_ref_ll += loglikelihood
                    total_ref_tokens += scored
                elif reference is not None:
                    raise ExtensionValidationError(
                        f"{label} non-dialogue reference scoring is not null"
                    )
                _metric_dict_matches(
                    payload.get("metrics"),
                    recomputed_metrics,
                    label=f"{label} metrics",
                )
                for name, value in recomputed_metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metric_values.setdefault(name, []).append(
                            _finite(value, f"{label}/{name}")
                        )
            means = {
                name: sum(values) / len(values)
                for name, values in sorted(metric_values.items())
            }
            if total_ref_tokens:
                means["reference_corpus_ppl"] = math.exp(
                    -total_ref_ll / total_ref_tokens
                )
            primary = base.SUMMARY_PRIMARY_METRICS[task]
            task_summary["modes"][mode] = {
                "completed_samples": len(samples[task]),
                "complete": primary in means and math.isfinite(float(means[primary])),
                "mean_metrics": means,
                "generation_cap_hits": cap_hits,
                "generation_cap_hit_rate": cap_hits / len(samples[task]),
                "reference_scored_tokens": total_ref_tokens,
            }
        target_metrics = task_summary["modes"]["target_standalone"]["mean_metrics"]
        mapped_metrics = task_summary["modes"]["mapped"]["mean_metrics"]
        comparisons: dict[str, Any] = {}
        for metric in sorted(set(target_metrics) & set(mapped_metrics)):
            target = target_metrics[metric]
            mapped = mapped_metrics[metric]
            comparisons[metric] = {
                "higher_is_better": metric
                not in {"reference_nll", "reference_ppl", "reference_corpus_ppl"},
                "target_standalone": target,
                "mapped": mapped,
                "mapped_minus_target": mapped - target,
                "mapped_over_target_percent": (
                    mapped / target * 100.0 if target != 0 else None
                ),
            }
        task_summary["metric_comparisons"] = comparisons
        primary = base.SUMMARY_PRIMARY_METRICS[task]
        comparison: dict[str, Any] = {"primary_metric": primary}
        if primary in comparisons:
            comparison.update(comparisons[primary])
        task_summary["comparison"] = comparison
        tasks_summary[task] = task_summary

    _result_inventory_paths(output, expected_paths)
    ordered = sorted(result_files, key=lambda row: row["path"])
    manifest_expected = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "expected_result_count": len(expected_paths),
        "files": ordered,
        "files_sha256": base.canonical_digest(ordered),
    }
    manifest_path = output / "results.manifest.json"
    if read_json(manifest_path) != manifest_expected:
        raise ExtensionValidationError("results manifest differs from raw inventory")
    summary_path = output / "summary.json"
    summary = read_json(summary_path)
    expected_summary_fixed = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "results_manifest": {
            "path": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "nbytes": manifest_path.stat().st_size,
        },
        "tasks": tasks_summary,
    }
    for field, expected in expected_summary_fixed.items():
        if summary.get(field) != expected:
            raise ExtensionValidationError(f"summary changed {field}")
    _utc(summary.get("updated_utc"), "summary updated")
    if set(summary) != {*expected_summary_fixed, "updated_utc"}:
        raise ExtensionValidationError("summary contains unexpected fields")
    return {
        "result_count": len(ordered),
        "result_inventory_sha256": canonical_sha256(ordered),
        "results_manifest": _artifact(manifest_path),
        "summary": _artifact(summary_path),
    }


def _validate_control_files(
    output: Path,
    *,
    profile: str,
    expected_definition: Mapping[str, Any],
    expected_dataset_manifest: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    verify_hashed_json(output / "extension_scope.json")
    if read_json(output / "extension_scope.json") != _expected_scope(profile):
        raise ExtensionValidationError("extension scope changed")
    definition_path = output / "run_definition.json"
    observed_definition = read_json(definition_path)
    fingerprint = _fingerprint(observed_definition)
    expected_with_fingerprint = {
        **dict(expected_definition),
        "request_fingerprint": canonical_sha256(expected_definition),
    }
    if observed_definition != expected_with_fingerprint:
        raise ExtensionValidationError("run definition differs from full reconstruction")
    if read_json(output / "dataset_manifest.json") != dict(expected_dataset_manifest):
        raise ExtensionValidationError("dataset manifest differs from source reconstruction")
    status_path = output / "status.json"
    status = read_json(status_path)
    fixed = {
        "status": "complete",
        "failures": 0,
        "request_fingerprint": fingerprint,
        "summary_sha256": sha256_file(output / "summary.json"),
        "summary_nbytes": (output / "summary.json").stat().st_size,
        "results_manifest": {
            "path": "results.manifest.json",
            "sha256": sha256_file(output / "results.manifest.json"),
            "nbytes": (output / "results.manifest.json").stat().st_size,
        },
    }
    for field, expected in fixed.items():
        if status.get(field) != expected:
            raise ExtensionValidationError(f"status changed {field}")
    status_started = _utc(status.get("started_utc"), "status started")
    status_finished = _utc(status.get("finished_utc"), "status finished")
    if status_finished < status_started:
        raise ExtensionValidationError("status finished before it started")
    if set(status) != {*fixed, "started_utc", "finished_utc"}:
        raise ExtensionValidationError("status contains unexpected fields")
    return fingerprint, {
        "extension_scope": _artifact(output / "extension_scope.json"),
        "extension_scope_sidecar": _artifact(
            output / "extension_scope.json.sha256.json"
        ),
        "run_definition": _artifact(definition_path),
        "dataset_manifest": _artifact(output / "dataset_manifest.json"),
        "status": _artifact(status_path),
    }


def _receipt_payload(output: Path, *, profile: str) -> dict[str, Any]:
    validate_code_freeze()
    base = configured_base_runner()
    args = _profile_args(profile, output)
    raw = base.load_yaml(CONFIG_PATH)
    configuration = base.validate_configuration(raw, args)
    all_samples, dataset_manifest = base.load_all_samples(configuration, TASKS)
    samples = base._apply_sample_scope(
        all_samples,
        limit=None,
        sample_indices=args.sample_indices,
    )
    expected_counts = (
        {task: len(indices) for task, indices in SMOKE_SAMPLE_INDICES.items()}
        if profile == "smoke"
        else EXPECTED_SAMPLE_COUNTS
    )
    if {task: len(rows) for task, rows in samples.items()} != expected_counts:
        raise ExtensionValidationError("reconstructed sample coverage changed")
    if profile == "smoke":
        dialogue_source_indices: dict[str, dict[int, int]] = {}
        for task in ("lmsys", "sharegpt4"):
            manifest = _mapping(dataset_manifest.get(task), f"{task} manifest")
            selected = _list(manifest.get("selected"), f"{task} selected records")
            dialogue_source_indices[task] = {
                int(_mapping(row, f"{task} selected record")["evaluation_index"]): int(
                    row["source_order"]
                )
                for row in selected
            }
        observed_endpoints = {
            task: [
                {
                    "evaluation_index": sample.index,
                    "source_index": (
                        dialogue_source_indices[task][sample.index]
                        if task in dialogue_source_indices
                        else sample.index
                    ),
                    "sample_id": sample.sample_id,
                    "record_sha256": sample.record_sha256,
                }
                for sample in samples[task]
            ]
            for task in TASKS
        }
        expected_endpoints = {
            task: [
                {
                    "evaluation_index": endpoint["evaluation_index"],
                    "source_index": endpoint["source_index"],
                    "sample_id": endpoint["sample_id"],
                    "record_sha256": endpoint["record_sha256"],
                }
                for endpoint in SMOKE_ENDPOINTS[task]
            ]
            for task in TASKS
        }
        if observed_endpoints != expected_endpoints:
            raise ExtensionValidationError("reconstructed smoke endpoint identity changed")
    expected_definition = base.build_run_definition(
        args=args,
        configuration=configuration,
        dataset_manifest=dataset_manifest,
    )
    fingerprint, controls = _validate_control_files(
        output,
        profile=profile,
        expected_definition=expected_definition,
        expected_dataset_manifest=dataset_manifest,
    )
    source_tokenizer = AutoTokenizer.from_pretrained(
        SOURCE_MODEL, local_files_only=True, use_fast=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        TARGET_MODEL, local_files_only=True, use_fast=True
    )
    target_config = read_json(TARGET_MODEL / "config.json")
    if target_config.get("vocab_size") != EXPECTED_TARGET_VOCAB_SIZE:
        raise ExtensionValidationError("target model vocab_size changed")
    tokenizer_identity = base._tokenizer_identity(source_tokenizer, target_tokenizer)
    if read_json(output / "tokenizer.json") != tokenizer_identity:
        raise ExtensionValidationError("tokenizer identity changed")
    results = _validate_results(
        base=base,
        output=output,
        configuration=configuration,
        samples=samples,
        fingerprint=fingerprint,
        tokenizer=target_tokenizer,
    )
    expected_count = (
        EXPECTED_SMOKE_RESULT_COUNT if profile == "smoke" else EXPECTED_RESULT_COUNT
    )
    if results["result_count"] != expected_count:
        raise ExtensionValidationError("atomic result coverage changed")
    model_sharding = _validate_model_sharding(output)
    mapper_runtime = _validate_mapper_runtime(
        output, expected_definition=expected_definition
    )
    gates = validate_gpu_idle_gates(output)
    freeze = validate_code_freeze()
    source_identity = expected_definition["source_model"]
    target_identity = expected_definition["target_model"]
    models = {
        "source_32b": {
            "path": source_identity["path"],
            "weight_shard_count": source_identity["weight_shard_count"],
            "weight_shard_total_nbytes": source_identity["weight_shard_total_nbytes"],
            "weight_shards_sha256": source_identity["weight_shards_sha256"],
        },
        "target_14b": {
            "path": target_identity["path"],
            "weight_shard_count": target_identity["weight_shard_count"],
            "weight_shard_total_nbytes": target_identity["weight_shard_total_nbytes"],
            "weight_shards_sha256": target_identity["weight_shards_sha256"],
        },
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "kind": (
            EXTENSION_VERSION if profile == "full" else f"{EXTENSION_VERSION}-smoke"
        ),
        "profile": profile,
        "direction": DIRECTION,
        "protocol": PROTOCOL_NAME,
        "paper_bit_exact": PAPER_BIT_EXACT,
        "not_a_formal_result": profile == "smoke",
        "request_fingerprint": fingerprint,
        "expected_sample_counts": dict(expected_counts),
        "expected_atomic_result_count": expected_count,
        "frozen_smoke_sample_indices": {
            task: list(indices) for task, indices in SMOKE_SAMPLE_INDICES.items()
        },
        "frozen_smoke_endpoints": {
            task: [dict(endpoint) for endpoint in endpoints]
            for task, endpoints in SMOKE_ENDPOINTS.items()
        },
        "controls": controls,
        "tokenizer": _artifact(output / "tokenizer.json"),
        "model_sharding": model_sharding,
        "models_rehashed": models,
        "mapper_runtime": mapper_runtime,
        "code_freeze": {
            "artifact": _artifact(CODE_FREEZE),
            "sidecar": _artifact(CODE_FREEZE_SIDECAR),
            "inventory_digest": freeze["inventory_digest"],
        },
        "gpu_idle_gates": gates,
        "results": results,
    }


def _receipt_path(output: Path, profile: str) -> Path:
    return output / (FULL_RECEIPT_NAME if profile == "full" else SMOKE_RECEIPT_NAME)


def finalize_output(output: str | Path, *, profile: str) -> dict[str, Any]:
    root = output_within_experiment(output)
    receipt_path = _receipt_path(root, profile)
    if receipt_path.exists():
        return validate_completed_output(root, profile=profile)
    receipt = _receipt_payload(root, profile=profile)
    hashed_json(receipt_path, receipt)
    return validate_completed_output(root, profile=profile)


def finalize_full_output(output: str | Path) -> dict[str, Any]:
    return finalize_output(output, profile="full")


def finalize_smoke_output(output: str | Path) -> dict[str, Any]:
    return finalize_output(output, profile="smoke")


def validate_completed_output(
    output: str | Path, *, profile: str
) -> dict[str, Any]:
    root = output_within_experiment(output)
    receipt_path = _receipt_path(root, profile)
    receipt_sidecar = receipt_path.with_name(f"{receipt_path.name}.sha256.json")
    if receipt_path.is_symlink() or receipt_sidecar.is_symlink():
        raise ExtensionValidationError("completion receipt is symlinked")
    verify_hashed_json(receipt_path)
    observed = read_json(receipt_path)
    expected = _receipt_payload(root, profile=profile)
    if observed != expected:
        raise ExtensionValidationError("completion receipt differs from full recomputation")
    return dict(observed)


__all__ = [
    "ExtensionValidationError",
    "FULL_RECEIPT_NAME",
    "SMOKE_RECEIPT_NAME",
    "finalize_full_output",
    "finalize_smoke_output",
    "validate_completed_output",
    "validate_gpu_idle_gate",
    "validate_gpu_idle_gates",
]
