"""Real, resumable eight-GPU execution path for the four frozen raw modes."""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

from .checkpoints import result_path, validate_raw_checkpoint
from .configuration import load_config
from .contracts import (
    ArtifactRef,
    ContractError,
    RAW_MODES,
    artifact_ref,
    atomic_json,
    canonical_sha256,
    exact_json_equal,
    finite_float,
    immutable_json,
    read_json,
    require_child,
    sha256_file,
    strict_int,
    token_ids_sha256,
    verify_sidecar,
    write_sidecar,
)
from .gpu_guard import global_gpu_lock, wait_for_idle_gpus
from .gpu_runtime import (
    RuntimeComponents,
    build_runtime_components,
    frozen_target_eos_token_ids,
    validate_method_bindings,
)
from .raw_writer import write_raw_checkpoint
from .run_protocol import canonical_run_output, reproduce_run_definition
from .run_state import run_state_lock, write_live_status
from .runtime_binding import verify_python_bytecode_policy


RAW_RECEIPT_FIELDS = {
    "schema_version",
    "status",
    "profile",
    "request_fingerprint",
    "attempt_id",
    "runner",
    "cuda_execution_started",
    "required_cuda_visible_devices",
    "run_definition",
    "gpu_idle_gate",
    "deep_runtime_validation",
    "cuda_runtime_validation",
    "method_provenance",
    "model_sharding",
    "runtime_binding_digest",
    "validated_method_bindings_digest",
    "raw_result_count",
    "expected_raw_result_count",
    "raw_inventory",
    "raw_inventory_sha256",
    "contributing_attempts",
    "execution_counts",
    "official_dev_or_test_opened",
}
ATTEMPT_EVIDENCE_FILENAMES = {
    "gpu_idle_gate": "gpu_idle_gate.json",
    "deep_runtime_validation": "deep_runtime_validation.json",
    "cuda_runtime_validation": "cuda_runtime_validation.json",
    "method_provenance": "method_provenance.json",
    "model_sharding": "model_sharding.json",
}


def _strict_int_mismatch(
    value: Any, expected: int, *, minimum: int | None = None
) -> bool:
    try:
        observed = strict_int(value, label="integer evidence", minimum=minimum)
    except ContractError:
        return True
    return observed != expected


def _decode_exact(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(list(ids), skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(list(ids)))


def _sync_cuda(torch_module: Any) -> None:
    for index in range(torch_module.cuda.device_count()):
        torch_module.cuda.synchronize(index)


def _reset_peaks(torch_module: Any) -> None:
    for index in range(torch_module.cuda.device_count()):
        torch_module.cuda.reset_peak_memory_stats(index)


def _peak_allocated(torch_module: Any) -> dict[str, int]:
    return {
        str(index): int(torch_module.cuda.max_memory_allocated(index))
        for index in range(torch_module.cuda.device_count())
    }


def _validate_existing_raw(
    path: Path,
    *,
    row: Mapping[str, Any],
    mode: str,
    definition: Mapping[str, Any],
    tokenizer: Any,
    native_sha: str | None,
    baseline_sha: str | None,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    sidecar = path.with_name(path.name + ".sha256.json")
    if sidecar.exists():
        verify_sidecar(path)
    value = validate_raw_checkpoint(
        path,
        row=row,
        mode=mode,
        request_fingerprint=definition["request_fingerprint"],
        method_binding_sha256=definition["method_binding_sha256"][mode],
        tokenizer=tokenizer,
        expected_native_file_sha256=native_sha if mode != "native14" else None,
        expected_baseline_file_sha256=(
            baseline_sha if mode in {"shrinkage", "tail_refresh_r32"} else None
        ),
    )
    # The result JSON itself is one atomic rename, while its digest sidecar is
    # the next atomic rename.  A process death between those two operations is
    # safely recoverable only after full semantic validation of the JSON.
    if not sidecar.exists():
        write_sidecar(artifact_ref(path))
    return value


def _remove_stale_failure(path: Path) -> None:
    """Remove only the failure marker paired with a fully validated result."""

    failure_path = path.with_suffix(".failure.json")
    if not failure_path.exists() and not failure_path.is_symlink():
        return
    if failure_path.is_symlink() or not failure_path.is_file():
        raise ContractError(f"raw failure marker is not a regular file: {failure_path}")
    failure_path.unlink()


def _assert_no_raw_failure_markers(output: Path) -> None:
    root = output / "results" / "raw"
    if root.is_symlink():
        raise ContractError("raw results root cannot be a symlink")
    failures = sorted(path for path in root.rglob("*.failure.json")) if root.exists() else []
    if failures:
        raise ContractError(f"raw completion cannot retain failure markers: {failures[:4]}")


def execute_raw_grid(
    *,
    output: Path,
    profile: str,
    rows: Sequence[Mapping[str, Any]],
    definition: Mapping[str, Any],
    components: RuntimeComponents,
    torch_module: Any,
    attempt_id: str | None = None,
    status_writer: Any = write_live_status,
) -> dict[str, Any]:
    """Run/resume all raw cells using already-loaded *real or mock* components."""

    if definition.get("profile") != profile:
        raise ContractError("GPU raw-grid profile differs from frozen definition")
    if not isinstance(attempt_id, str) or re.fullmatch(r"attempt_[0-9]{4,}", attempt_id) is None:
        raise ContractError("GPU raw-grid requires a canonical execution attempt ID")
    if set(components.engines) != set(RAW_MODES):
        raise ContractError("GPU runtime engine modes changed")
    if set(components.builders) != {"nvidia_k20", "shrinkage", "tail_refresh_r32"}:
        raise ContractError("GPU runtime builder modes changed")
    completed = 0
    generated_now = 0
    resumed = 0
    for row_index, row in enumerate(rows):
        native_sha: str | None = None
        baseline_sha: str | None = None
        for mode in RAW_MODES:
            path = result_path(output, row, mode)
            existing = _validate_existing_raw(
                path,
                row=row,
                mode=mode,
                definition=definition,
                tokenizer=components.tokenizer,
                native_sha=native_sha,
                baseline_sha=baseline_sha,
            )
            if existing is not None:
                # A process can die after atomically publishing a valid result
                # and sidecar but before deleting the preceding failure
                # marker.  Full semantic validation above makes this cleanup
                # safe and prevents an otherwise unrecoverable sealed run.
                _remove_stale_failure(path)
                if mode == "native14":
                    native_sha = sha256_file(path)
                elif mode == "nvidia_k20":
                    baseline_sha = sha256_file(path)
                completed += 1
                resumed += 1
                continue
            if mode != "native14" and native_sha is None:
                raise ContractError("candidate raw cell cannot precede native14")
            if mode in {"shrinkage", "tail_refresh_r32"} and baseline_sha is None:
                raise ContractError("improved raw cell cannot precede NVIDIA k20")
            engine = components.engines[mode]
            _reset_peaks(torch_module)
            _sync_cuda(torch_module)
            began = time.perf_counter()
            try:
                generated = engine.generate_token_ids(
                    row["prompt_token_ids"],
                    {
                        "max_gen_toks": row["generation_cap"],
                        "do_sample": False,
                        "temperature": 0.0,
                        "until": row["stop_sequences"],
                    },
                    truncation="error",
                )
                _sync_cuda(torch_module)
                elapsed = time.perf_counter() - began
                expected_generation_metadata = {
                    "requested_max_tokens": row["generation_cap"],
                    "original_context_tokens": row["prompt_token_count"],
                    "used_context_tokens": row["prompt_token_count"],
                    "truncation": "error",
                }
                for field, expected in expected_generation_metadata.items():
                    observed = getattr(generated, field, None)
                    if type(expected) is int:
                        try:
                            observed = strict_int(
                                observed,
                                label=f"runtime generation {field}",
                                minimum=0,
                            )
                        except ContractError as exc:
                            raise ContractError(
                                f"runtime generation {field} changed"
                            ) from exc
                    if not exact_json_equal(observed, expected):
                        raise ContractError(f"runtime generation {field} changed")
                missing_terminal = object()
                terminal_token_id = getattr(
                    generated, "terminal_token_id", missing_terminal
                )
                raw_generated_ids = getattr(generated, "generated_token_ids", None)
                if isinstance(raw_generated_ids, tuple):
                    try:
                        for item in raw_generated_ids:
                            strict_int(
                                item, label="runtime generated token ID", minimum=0
                            )
                        if terminal_token_id is not None and terminal_token_id is not missing_terminal:
                            strict_int(
                                terminal_token_id,
                                label="runtime terminal token ID",
                                minimum=0,
                            )
                    except ContractError as exc:
                        raise ContractError(
                            "runtime generation result types changed"
                        ) from exc
                if (
                    type(getattr(generated, "text", None)) is not str
                    or not isinstance(raw_generated_ids, tuple)
                    or type(getattr(generated, "stop_reason", None)) is not str
                    or type(getattr(generated, "cap_hit", None)) is not bool
                    or terminal_token_id is missing_terminal
                ):
                    raise ContractError("runtime generation result types changed")
                generated_ids = list(generated.generated_token_ids)
                decoded = _decode_exact(components.tokenizer, generated_ids)
                generation = {
                    "requested_max_tokens": row["generation_cap"],
                    "generated_tokens": len(generated_ids),
                    "generated_token_ids": generated_ids,
                    "generated_token_ids_sha256": token_ids_sha256(generated_ids),
                    "decoded_generated_token_ids": decoded,
                    "stop_reason": generated.stop_reason,
                    "terminal_token_id": terminal_token_id,
                    "frozen_eos_token_ids": list(
                        components.tokenizer._prospective_frozen_eos_token_ids
                    ),
                    "stop_sequences": list(row["stop_sequences"]),
                    "cap_hit": generated.cap_hit,
                }
                audit = (
                    dict(components.builders[mode].last_call_audit)
                    if mode == "tail_refresh_r32"
                    else None
                )
                write_raw_checkpoint(
                    output_dir=output,
                    row=row,
                    mode=mode,
                    request_fingerprint=definition["request_fingerprint"],
                    method_binding_sha256=definition["method_binding_sha256"][mode],
                    execution_attempt_id=attempt_id,
                    prediction=generated.text,
                    generation=generation,
                    runtime_cost={
                        "wall_seconds": elapsed,
                        "phase_timing": {
                            "available": False,
                            "prefill_seconds": None,
                            "decode_seconds": None,
                            "scope": "monolithic_engine_generate_token_ids",
                        },
                        "peak_gpu_allocated_bytes": _peak_allocated(torch_module),
                    },
                    builder_audit=audit,
                    native14_raw_file_sha256=(native_sha if mode != "native14" else None),
                    nvidia_k20_raw_file_sha256=(
                        baseline_sha if mode in {"shrinkage", "tail_refresh_r32"} else None
                    ),
                    tokenizer=components.tokenizer,
                )
                _remove_stale_failure(path)
                if mode == "native14":
                    native_sha = sha256_file(path)
                elif mode == "nvidia_k20":
                    baseline_sha = sha256_file(path)
                completed += 1
                generated_now += 1
            except BaseException as exc:
                atomic_json(
                    path.with_suffix(".failure.json"),
                    {
                        "schema_version": 2,
                        "status": "failed",
                        "request_fingerprint": definition["request_fingerprint"],
                        "dataset_id": row["dataset_id"],
                        "sample_id": row["sample_id"],
                        "mode": mode,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                status_writer(
                    output / "status.json",
                    {
                        "schema_version": 2,
                        "status": "gpu_failed",
                        "request_fingerprint": definition["request_fingerprint"],
                        "completed_raw_results": completed,
                        **({"attempt_id": attempt_id} if attempt_id is not None else {}),
                        "failed_cell": {
                            "dataset_id": row["dataset_id"],
                            "sample_id": row["sample_id"],
                            "mode": mode,
                        },
                    },
                )
                raise
            finally:
                engine.clear_prefix_cache()
        status_writer(
            output / "status.json",
            {
                "schema_version": 2,
                "status": "gpu_running",
                "request_fingerprint": definition["request_fingerprint"],
                "completed_samples": row_index + 1,
                "total_samples": len(rows),
                "completed_raw_results": completed,
                **({"attempt_id": attempt_id} if attempt_id is not None else {}),
            },
        )
    return {
        "completed_raw_results": completed,
        "generated_now": generated_now,
        "resumed_existing": resumed,
    }


def _next_attempt_dir(output: Path) -> tuple[str, Path]:
    root = output / "gpu_attempts"
    if root.is_symlink():
        raise ContractError("GPU attempt root cannot be a symlink")
    existing = []
    if root.exists():
        for path in root.iterdir():
            if path.is_symlink():
                raise ContractError("GPU attempt inventory cannot contain symlinks")
            if path.is_dir() and path.name.startswith("attempt_") and path.name[8:].isdigit():
                existing.append(int(path.name[8:]))
    index = (max(existing) + 1) if existing else 1
    name = f"attempt_{index:04d}"
    return name, root / name


def raw_inventory(rows: Sequence[Mapping[str, Any]], output: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        for mode in RAW_MODES:
            path = result_path(output, row, mode)
            verify_sidecar(path)
            result.append(artifact_ref(path).to_dict())
    return result


def validate_raw_grid_for_receipt(
    *,
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Deeply validate every raw cell and the exact raw-result JSON tree."""

    if tokenizer is None:
        raise ContractError("raw completion validation requires the frozen tokenizer")
    inventory: list[dict[str, Any]] = []
    expected_files: set[Path] = set()
    for row in rows:
        native_path = result_path(output, row, "native14")
        baseline_path = result_path(output, row, "nvidia_k20")
        if not native_path.is_file() or not baseline_path.is_file():
            raise ContractError("raw completion grid lacks native14/nvidia_k20")
        native_sha = sha256_file(native_path)
        baseline_sha = sha256_file(baseline_path)
        for mode in RAW_MODES:
            path = result_path(output, row, mode)
            sidecar = path.with_name(path.name + ".sha256.json")
            verify_sidecar(path)
            validate_raw_checkpoint(
                path,
                row=row,
                mode=mode,
                request_fingerprint=definition["request_fingerprint"],
                method_binding_sha256=definition["method_binding_sha256"][mode],
                tokenizer=tokenizer,
                expected_native_file_sha256=(native_sha if mode != "native14" else None),
                expected_baseline_file_sha256=(
                    baseline_sha
                    if mode in {"shrinkage", "tail_refresh_r32"}
                    else None
                ),
            )
            expected_files.update({path.resolve(), sidecar.resolve()})
            inventory.append(artifact_ref(path).to_dict())
    root = output / "results" / "raw"
    observed: set[Path] = set()
    if root.is_symlink() or not root.is_dir():
        raise ContractError("raw completion results root is missing/not regular")
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"raw result tree contains a symlink: {path}")
        if path.name.endswith(".json"):
            if not path.is_file():
                raise ContractError(f"raw result tree contains a non-regular JSON: {path}")
            observed.add(path.resolve())
    if observed != expected_files:
        raise ContractError("raw result tree contains missing/unexpected/failure JSON")
    return inventory


def contributing_attempts(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    attempt_ids: set[str] = set()
    for row in rows:
        for mode in RAW_MODES:
            path = result_path(output, row, mode)
            verify_sidecar(path)
            value = read_json(path)
            attempt_id = value.get("execution_attempt_id") if isinstance(value, Mapping) else None
            if (
                not isinstance(attempt_id, str)
                or re.fullmatch(r"attempt_[0-9]{4,}", attempt_id) is None
            ):
                raise ContractError("raw grid contains a malformed execution attempt ID")
            attempt_ids.add(attempt_id)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for attempt_id in sorted(attempt_ids):
        root = output / "gpu_attempts" / attempt_id
        evidence: dict[str, dict[str, Any]] = {}
        for field, filename in ATTEMPT_EVIDENCE_FILENAMES.items():
            path = require_child(root, root / filename, label=f"{attempt_id}.{field}")
            verify_sidecar(path)
            evidence[field] = artifact_ref(path).to_dict()
        result[attempt_id] = evidence
    return result


def raw_attempt_cell_counts(
    rows: Sequence[Mapping[str, Any]], output: Path
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for mode in RAW_MODES:
            path = result_path(output, row, mode)
            verify_sidecar(path)
            value = read_json(path)
            attempt_id = value.get("execution_attempt_id") if isinstance(value, Mapping) else None
            if (
                not isinstance(attempt_id, str)
                or re.fullmatch(r"attempt_[0-9]{4,}", attempt_id) is None
            ):
                raise ContractError("raw grid contains a malformed execution attempt ID")
            counts[attempt_id] += 1
    return dict(sorted(counts.items()))


def seal_raw_completion_receipt(
    *,
    config: Mapping[str, Any],
    output: Path,
    profile: str,
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    attempt_id: str,
    gpu_idle_gate_ref: ArtifactRef,
    deep_runtime_validation_ref: ArtifactRef,
    cuda_runtime_validation_ref: ArtifactRef,
    method_provenance_ref: ArtifactRef,
    model_sharding_ref: ArtifactRef,
    execution_counts: Mapping[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    if profile != definition.get("profile"):
        raise ContractError("raw completion profile differs from run definition")
    expected_count = len(rows) * len(RAW_MODES)
    inventory = validate_raw_grid_for_receipt(
        definition=definition, rows=rows, output=output, tokenizer=tokenizer
    )
    attempt_counts = raw_attempt_cell_counts(rows, output)
    _validate_execution_counts(
        execution_counts,
        expected_count,
        expected_generated_now=attempt_counts.get(attempt_id, 0),
    )
    _assert_no_raw_failure_markers(output)
    contributions = contributing_attempts(rows, output)
    current_evidence = {
        "gpu_idle_gate": gpu_idle_gate_ref.to_dict(),
        "deep_runtime_validation": deep_runtime_validation_ref.to_dict(),
        "cuda_runtime_validation": cuda_runtime_validation_ref.to_dict(),
        "method_provenance": method_provenance_ref.to_dict(),
        "model_sharding": model_sharding_ref.to_dict(),
    }
    if attempt_id in contributions and contributions[attempt_id] != current_evidence:
        raise ContractError("final GPU attempt evidence differs from raw-cell attempt bindings")
    if execution_counts["generated_now"] > 0 and attempt_id not in contributions:
        raise ContractError("new raw cells are not bound to the final GPU attempt")
    receipt = {
        "schema_version": 2,
        "status": "raw_gpu_complete",
        "profile": profile,
        "request_fingerprint": definition["request_fingerprint"],
        "attempt_id": attempt_id,
        "runner": "real_qwen3_32b_14b_5plus3_four_mode_runner_v2",
        "cuda_execution_started": True,
        "required_cuda_visible_devices": config["runtime"]["required_cuda_visible_devices"],
        "run_definition": artifact_ref(output / "run_definition.json").to_dict(),
        "gpu_idle_gate": gpu_idle_gate_ref.to_dict(),
        "deep_runtime_validation": deep_runtime_validation_ref.to_dict(),
        "cuda_runtime_validation": cuda_runtime_validation_ref.to_dict(),
        "method_provenance": method_provenance_ref.to_dict(),
        "model_sharding": model_sharding_ref.to_dict(),
        "runtime_binding_digest": definition["runtime_binding_digest"],
        "validated_method_bindings_digest": definition["validated_method_bindings_digest"],
        "raw_result_count": len(inventory),
        "expected_raw_result_count": len(rows) * len(RAW_MODES),
        "raw_inventory": inventory,
        "raw_inventory_sha256": canonical_sha256(inventory),
        "contributing_attempts": contributions,
        "execution_counts": dict(execution_counts),
        "official_dev_or_test_opened": False,
    }
    # No terminal raw marker may be published before every cell, evidence
    # artifact, attempt count, and result-tree invariant passes.  Otherwise a
    # bad terminal JSON would block the only safe resume path.
    _validate_raw_completion_payload(
        config,
        definition,
        rows,
        output,
        receipt,
        tokenizer=tokenizer,
    )
    reference = immutable_json(output / "raw_completion_receipt.json", receipt)
    write_sidecar(reference)
    validate_raw_completion_receipt(
        config, definition, rows, output, tokenizer=tokenizer
    )
    return {**receipt, "raw_completion_receipt_sha256": reference.sha256}


def _validate_execution_counts(
    value: Any, expected: int, *, expected_generated_now: int
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "completed_raw_results",
        "generated_now",
        "resumed_existing",
    }:
        raise ContractError("raw completion execution-count schema changed")
    try:
        for field in value:
            strict_int(
                value[field],
                label=f"raw completion execution_counts.{field}",
                minimum=0,
            )
        strict_int(expected, label="expected raw result count", minimum=0)
        strict_int(
            expected_generated_now,
            label="expected generated-now count",
            minimum=0,
        )
    except ContractError as exc:
        raise ContractError("raw completion execution counts are malformed") from exc
    if (
        value["completed_raw_results"] != expected
        or value["generated_now"] + value["resumed_existing"] != expected
    ):
        raise ContractError("raw completion execution counts do not close the grid")
    if (
        value["generated_now"] != expected_generated_now
        or value["resumed_existing"] != expected - expected_generated_now
    ):
        raise ContractError("raw completion execution counts differ from raw-cell attempt ledger")


def _validate_idle_gate(value: Any, contract: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "status",
        "probe",
        "gpu_used_by_gate",
        "physical_gpu_indices",
        "contract",
        "samples",
        "final_consecutive_idle_samples",
        "elapsed_seconds",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ContractError("raw completion GPU idle-gate schema changed")
    fixed = {
        "schema_version": 2,
        "status": "passed",
        "probe": "nvidia-smi_without_CUDA_runtime",
        "gpu_used_by_gate": False,
        "physical_gpu_indices": list(range(8)),
        "contract": dict(contract),
    }
    for field, expected in fixed.items():
        if not exact_json_equal(value.get(field), expected):
            raise ContractError(f"raw completion GPU idle-gate {field} changed")
    samples = value.get("samples")
    required = strict_int(
        contract["consecutive_samples"],
        label="idle-gate consecutive sample requirement",
        minimum=1,
    )
    limit = strict_int(
        contract["memory_used_limit_mib"],
        label="idle-gate memory limit",
        minimum=0,
    )
    if not isinstance(samples, list) or len(samples) < required:
        raise ContractError("raw completion GPU idle gate has too few samples")
    consecutive = 0
    previous_elapsed = -1.0
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) != {
            "sample_index",
            "elapsed_seconds",
            "memory_used_mib",
            "compute_process_count",
            "idle",
            "consecutive_idle",
        }:
            raise ContractError("raw completion GPU idle sample schema changed")
        elapsed = finite_float(sample.get("elapsed_seconds"), label="GPU idle sample elapsed")
        memory = sample.get("memory_used_mib")
        processes = sample.get("compute_process_count")
        if (
            _strict_int_mismatch(sample.get("sample_index"), index)
            or elapsed < previous_elapsed
            or not isinstance(memory, Mapping)
            or set(memory) != {str(device) for device in range(8)}
            or any(_strict_int_mismatch(item, item, minimum=0) for item in memory.values())
            or _strict_int_mismatch(processes, processes, minimum=0)
            or type(sample.get("idle")) is not bool
        ):
            raise ContractError("raw completion GPU idle sample is malformed")
        expected_idle = processes == 0 and all(item <= limit for item in memory.values())
        consecutive = consecutive + 1 if expected_idle else 0
        if (
            sample["idle"] is not expected_idle
            or _strict_int_mismatch(sample.get("consecutive_idle"), consecutive)
        ):
            raise ContractError("raw completion GPU idle sample does not recompute")
        previous_elapsed = elapsed
    if (
        _strict_int_mismatch(value.get("final_consecutive_idle_samples"), required)
        or consecutive != required
    ):
        raise ContractError("raw completion GPU idle gate consecutive count changed")
    if finite_float(value.get("elapsed_seconds"), label="GPU idle elapsed") < previous_elapsed:
        raise ContractError("raw completion GPU idle elapsed time regressed")


def _same_path(left: Any, right: Any) -> bool:
    try:
        return Path(str(left)).resolve() == Path(str(right)).resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _validate_method_provenance(value: Any, config: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(RAW_MODES):
        raise ContractError("raw completion method provenance modes changed")
    native = value["native14"]
    if not exact_json_equal(
        native, {"method_id": config["methods"]["native14"]["method_id"]}
    ):
        raise ContractError("native14 runtime provenance changed")
    baseline_ref = ArtifactRef.parse(
        config["methods"]["nvidia_k20"]["mapper_artifact"], label="configured k20 mapper"
    )
    baseline = value["nvidia_k20"]
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("direction") != config["direction"]
        or not _same_path(baseline.get("manifest"), baseline_ref.path)
        or baseline.get("manifest_sha256") != baseline_ref.sha256
        or _strict_int_mismatch(baseline.get("layers"), 40)
        or _strict_int_mismatch(baseline.get("selected_source_layers"), 20)
        or not exact_json_equal(
            baseline.get("token_chunk"), config["runtime"]["mapper_token_chunk"]
        )
    ):
        raise ContractError("NVIDIA k20 runtime provenance changed")
    shrink_ref = ArtifactRef.parse(
        config["methods"]["shrinkage"]["mapper_artifact"], label="configured shrinkage mapper"
    )
    shrinkage = value["shrinkage"]
    shrinkage_manifest = read_json(shrink_ref.path)
    if (
        not isinstance(shrinkage, Mapping)
        or shrinkage.get("direction") != config["direction"]
        or shrinkage.get("method") != shrinkage_manifest.get("method")
        or not _same_path(shrinkage.get("manifest"), shrink_ref.path)
        or shrinkage.get("manifest_sha256") != shrink_ref.sha256
        or _strict_int_mismatch(shrinkage.get("layers"), 40)
        or _strict_int_mismatch(shrinkage.get("selected_source_layers"), 20)
        or not exact_json_equal(
            shrinkage.get("token_chunk"), config["runtime"]["mapper_token_chunk"]
        )
    ):
        raise ContractError("shrinkage runtime provenance changed")
    tail = value["tail_refresh_r32"]
    if (
        not isinstance(tail, Mapping)
        or tail.get("method") != config["methods"]["tail_refresh_r32"]["method_id"]
        or _strict_int_mismatch(tail.get("replay_tokens_R"), 32)
        or _strict_int_mismatch(tail.get("builder_native_tail_tokens"), 31)
        or _strict_int_mismatch(tail.get("engine_boundary_tokens"), 1)
        or tail.get("resident_mapper_shared_across_variants") is not True
        or not _same_path(tail.get("manifest"), baseline_ref.path)
        or tail.get("manifest_sha256") != baseline_ref.sha256
    ):
        raise ContractError("R32 runtime provenance changed")


def _validate_deep_runtime_validation(
    value: Any, config: Mapping[str, Any], definition: Mapping[str, Any]
) -> None:
    current = validate_method_bindings(config, deep_model_hash=False)
    frozen_runtime = read_json(current.runtime["receipt"]["path"])["runtime"]
    expected_mapper_payloads = {
        mode: {**dict(evidence), "deep_hash_verified": True}
        for mode, evidence in current.mapper_payload_validation.items()
    }
    expected = {
        "schema_version": 2,
        "status": "passed_before_cuda_import",
        "deep_model_hash_verified": True,
        "runtime_environment_verified": True,
        "runtime_binding_digest": current.runtime["binding_digest"],
        "validated_method_bindings_digest": current.digest,
        "validated_mode_digests": dict(current.mode_digests),
        "runtime_provenance_receipt": current.runtime["receipt"],
        "source_model_validation": current.runtime["source_model"],
        "target_model_validation": current.runtime["target_model"],
        "mapper_manifest": current.runtime["mapper_manifest"],
        "runtime_code": current.runtime["runtime_code"],
        "runtime_environment": {
            "python": frozen_runtime["python"],
            "distributions": frozen_runtime["distributions"],
            "python_dont_write_bytecode": True,
            "python_pycache_prefix": config["runtime"][
                "required_python_pycache_prefix"
            ],
            "python_pycache_prefix_absent": True,
        },
        "mapper_payload_validation": expected_mapper_payloads,
    }
    if not exact_json_equal(value, expected):
        raise ContractError("deep pre-CUDA runtime validation evidence changed")
    if (
        expected["runtime_binding_digest"] != definition["runtime_binding_digest"]
        or expected["validated_method_bindings_digest"]
        != definition["validated_method_bindings_digest"]
        or not exact_json_equal(
            expected["validated_mode_digests"], definition["validated_mode_digests"]
        )
    ):
        raise ContractError("deep runtime validation is not bound to run definition")


def _cuda_runtime_evidence(
    torch_module: Any, config: Mapping[str, Any], frozen_runtime: Mapping[str, Any]
) -> dict[str, Any]:
    cuda_available = torch_module.cuda.is_available()
    if type(cuda_available) is not bool:
        raise ContractError("torch CUDA availability evidence is not boolean")
    visible_device_count = strict_int(
        torch_module.cuda.device_count(),
        label="visible CUDA device count",
        minimum=0,
    )
    value = {
        "schema_version": 2,
        "status": "passed_before_model_load",
        "torch_version": str(getattr(torch_module, "__version__", "")),
        "torch_cuda_version": str(getattr(getattr(torch_module, "version", None), "cuda", "")),
        "cuda_available": cuda_available,
        "visible_cuda_device_count": visible_device_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "validated_before_model_load": True,
    }
    expected = {
        **value,
        "torch_version": frozen_runtime["torch"],
        "torch_cuda_version": frozen_runtime["torch_cuda"],
        "cuda_available": True,
        "visible_cuda_device_count": 8,
        "cuda_visible_devices": config["runtime"]["required_cuda_visible_devices"],
    }
    if not exact_json_equal(value, expected):
        raise ContractError("loaded torch/CUDA runtime differs from frozen provenance")
    return value


def _validate_cuda_runtime_validation(value: Any, config: Mapping[str, Any]) -> None:
    provenance_ref = ArtifactRef.parse(
        config["runtime_provenance_receipt"], label="runtime provenance receipt"
    )
    frozen_runtime = read_json(provenance_ref.path)["runtime"]
    expected = {
        "schema_version": 2,
        "status": "passed_before_model_load",
        "torch_version": frozen_runtime["torch"],
        "torch_cuda_version": frozen_runtime["torch_cuda"],
        "cuda_available": True,
        "visible_cuda_device_count": 8,
        "cuda_visible_devices": config["runtime"]["required_cuda_visible_devices"],
        "validated_before_model_load": True,
    }
    if not exact_json_equal(value, expected):
        raise ContractError("CUDA runtime validation evidence changed")


def _validate_sharding_entry(
    value: Any,
    *,
    devices: Sequence[int],
    layer_counts: Sequence[int],
    total_layers: int,
    capture_role: str,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} sharding evidence is malformed")
    fixed = {
        "status": "passed",
        "role": capture_role,
        "mode": "explicit_exact_counts",
        "devices": list(devices),
        "declared_layer_counts": list(layer_counts),
        "expected_layer_counts": list(layer_counts),
        "actual_layer_counts": list(layer_counts),
        "total_layers": total_layers,
        "layer_devices": [
            device
            for device, count in zip(devices, layer_counts, strict=True)
            for _ in range(count)
        ],
        "actual_parameters_inspected_for_every_layer": True,
        "auxiliary_parameters_inspected_for_every_module": True,
    }
    for field, expected in fixed.items():
        if not exact_json_equal(value.get(field), expected):
            raise ContractError(f"{label} model sharding {field} changed")
    ranges = []
    start = 0
    for device, count in zip(devices, layer_counts, strict=True):
        ranges.append(
            {
                "device": device,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "count": count,
            }
        )
        start += count
    if not exact_json_equal(value.get("contiguous_ranges"), ranges):
        raise ContractError(f"{label} contiguous layer ranges changed")
    try:
        strict_int(
            value.get("parameter_tensors_inspected"),
            label=f"{label} inspected layer parameter count",
            minimum=1,
        )
        strict_int(
            value.get("auxiliary_parameter_tensors_inspected"),
            label=f"{label} inspected auxiliary parameter count",
            minimum=1,
        )
    except ContractError as exc:
        raise ContractError(f"{label} sharding inspection counts are malformed") from exc


def _validate_model_sharding(value: Any, config: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {"source_32b", "target_14b"}:
        raise ContractError("raw completion model sharding changed")
    runtime = config["runtime"]
    _validate_sharding_entry(
        value["source_32b"],
        devices=runtime["source_devices"],
        layer_counts=runtime["source_layer_counts"],
        total_layers=64,
        capture_role="target",
        label="Qwen3-32B",
    )
    _validate_sharding_entry(
        value["target_14b"],
        devices=runtime["target_devices"],
        layer_counts=runtime["target_layer_counts"],
        total_layers=40,
        capture_role="source",
        label="Qwen3-14B",
    )


def _validate_raw_completion_payload(
    config: Mapping[str, Any],
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    value: Any,
    *,
    tokenizer: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("raw completion receipt root is malformed")
    if set(value) != RAW_RECEIPT_FIELDS:
        raise ContractError("raw completion receipt field set changed")
    fixed = {
        "schema_version": 2,
        "status": "raw_gpu_complete",
        "profile": definition["profile"],
        "request_fingerprint": definition["request_fingerprint"],
        "runner": "real_qwen3_32b_14b_5plus3_four_mode_runner_v2",
        "cuda_execution_started": True,
        "required_cuda_visible_devices": config["runtime"]["required_cuda_visible_devices"],
        "run_definition": artifact_ref(output / "run_definition.json").to_dict(),
        "runtime_binding_digest": definition["runtime_binding_digest"],
        "validated_method_bindings_digest": definition["validated_method_bindings_digest"],
        "raw_result_count": len(rows) * len(RAW_MODES),
        "expected_raw_result_count": len(rows) * len(RAW_MODES),
        "official_dev_or_test_opened": False,
    }
    for field, expected in fixed.items():
        if not exact_json_equal(value.get(field), expected):
            raise ContractError(f"raw completion receipt {field} changed")
    inventory = validate_raw_grid_for_receipt(
        definition=definition, rows=rows, output=output, tokenizer=tokenizer
    )
    if (
        not exact_json_equal(value.get("raw_inventory"), inventory)
        or value.get("raw_inventory_sha256") != canonical_sha256(inventory)
    ):
        raise ContractError("raw completion inventory changed")
    attempt = value.get("attempt_id")
    if not isinstance(attempt, str) or re.fullmatch(r"attempt_[0-9]{4,}", attempt) is None:
        raise ContractError("raw completion attempt ID is malformed")
    contributions = contributing_attempts(rows, output)
    if not exact_json_equal(value.get("contributing_attempts"), contributions):
        raise ContractError("raw completion contributing-attempt ledger changed")
    final_evidence = {
        field: value.get(field) for field in ATTEMPT_EVIDENCE_FILENAMES
    }
    if attempt in contributions and not exact_json_equal(
        final_evidence, contributions[attempt]
    ):
        raise ContractError("raw completion final-attempt evidence differs from its raw cells")
    attempt_counts = raw_attempt_cell_counts(rows, output)
    _validate_execution_counts(
        value.get("execution_counts"),
        len(rows) * len(RAW_MODES),
        expected_generated_now=attempt_counts.get(attempt, 0),
    )
    if value["execution_counts"]["generated_now"] > 0 and attempt not in contributions:
        raise ContractError("raw completion new cells lack a final-attempt binding")
    evidence_sets = dict(contributions)
    evidence_sets[attempt] = final_evidence
    for contribution_id, binding in evidence_sets.items():
        if set(binding) != set(ATTEMPT_EVIDENCE_FILENAMES):
            raise ContractError(
                f"raw completion {contribution_id} evidence field set changed"
            )
        for field in ATTEMPT_EVIDENCE_FILENAMES:
            if not isinstance(binding.get(field), Mapping):
                raise ContractError(
                    f"raw completion {contribution_id}.{field} reference is malformed"
                )
    for contribution_id, binding in evidence_sets.items():
        evidence: dict[str, Path] = {}
        attempt_root = output / "gpu_attempts" / contribution_id
        for field, expected_name in ATTEMPT_EVIDENCE_FILENAMES.items():
            reference = ArtifactRef.parse(
                binding.get(field), label=f"raw completion {contribution_id}.{field}"
            )
            evidence_path = require_child(
                attempt_root, reference.path, label=f"{contribution_id}.{field}"
            )
            expected_path = require_child(
                attempt_root,
                attempt_root / expected_name,
                label=f"{contribution_id}.{field}.expected",
            )
            if evidence_path != expected_path:
                raise ContractError(f"raw completion {contribution_id}.{field} path changed")
            reference.verify(label=f"raw completion {contribution_id}.{field}")
            verify_sidecar(evidence_path)
            evidence[field] = evidence_path
        _validate_idle_gate(
            read_json(evidence["gpu_idle_gate"]), config["runtime"]["idle_gate"]
        )
        _validate_deep_runtime_validation(
            read_json(evidence["deep_runtime_validation"]), config, definition
        )
        _validate_cuda_runtime_validation(
            read_json(evidence["cuda_runtime_validation"]), config
        )
        _validate_method_provenance(read_json(evidence["method_provenance"]), config)
        _validate_model_sharding(read_json(evidence["model_sharding"]), config)
    _assert_no_raw_failure_markers(output)
    return dict(value)


def validate_raw_completion_receipt(
    config: Mapping[str, Any],
    definition: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    repair_missing_sidecar: bool = False,
    tokenizer: Any = None,
) -> dict[str, Any]:
    path = output / "raw_completion_receipt.json"
    sidecar = path.with_name(path.name + ".sha256.json")
    sidecar_missing = not sidecar.exists() and not sidecar.is_symlink()
    if sidecar_missing:
        if not repair_missing_sidecar:
            raise ContractError(f"missing SHA sidecar: {sidecar}")
        # Establish regular-file/hash identity now; the sidecar is published
        # only after every semantic check below succeeds.
        artifact_ref(path)
    else:
        verify_sidecar(path)
    value = _validate_raw_completion_payload(
        config,
        definition,
        rows,
        output,
        read_json(path),
        tokenizer=tokenizer,
    )
    if sidecar_missing:
        write_sidecar(artifact_ref(path))
    return value


def load_frozen_decode_tokenizer(config: Mapping[str, Any]) -> Any:
    """Load only the hashed Rust tokenizer backend; never import torch."""

    from tokenizers import Tokenizer

    model_root = Path(config["runtime"]["target_model"])
    tokenizer_config = read_json(model_root / "tokenizer_config.json")
    if (
        not isinstance(tokenizer_config, Mapping)
        or tokenizer_config.get("clean_up_tokenization_spaces") is not False
    ):
        raise ContractError("frozen tokenizer decode cleanup contract changed")
    tokenizer_path = model_root / "tokenizer.json"
    backend = Tokenizer.from_file(str(tokenizer_path))

    class _DecodeAdapter:
        _prospective_frozen_eos_token_ids = frozen_target_eos_token_ids(model_root)

        @staticmethod
        def decode(ids: Sequence[int], skip_special_tokens: bool = True) -> str:
            return str(
                backend.decode(
                    [int(item) for item in ids],
                    skip_special_tokens=skip_special_tokens,
                )
            )

    return _DecodeAdapter()


def execute_gpu(config_path: str | Path, *, profile: str, output_dir: str | Path) -> int:
    """Acquire the global lock, prove idle GPUs, load real models, and run raw grid."""

    config = load_config(config_path, verify_inputs=True)
    # The script bootstrap performs this check before importing ``src``.  Keep
    # the same fail-closed check here for callers invoking the Python API
    # directly, before any external runtime modules or CUDA are loaded.
    verify_python_bytecode_policy(config["runtime"])
    output = canonical_run_output(config, profile, output_dir)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    required_visible = config["runtime"]["required_cuda_visible_devices"]
    if visible != required_visible:
        raise ContractError(f"GPU runner requires CUDA_VISIBLE_DEVICES={required_visible}")
    if Path(sys.executable).resolve() != Path(config["runtime"]["python_executable"]).resolve():
        raise ContractError(
            f"GPU runner requires interpreter {config['runtime']['python_executable']}"
        )
    with global_gpu_lock(config["runtime"]["global_gpu_lock"]), run_state_lock(output):
        # This in-lock check is authoritative.  It closes the race in which a
        # second GPU process passes an optimistic outer check, waits for the
        # first process, and would otherwise load both models after completion.
        completion_path = output / "completion_receipt.json"
        if completion_path.exists() or completion_path.is_symlink():
            raise ContractError("sealed run cannot be re-entered by GPU runner")
        definition, rows = reproduce_run_definition(config, profile, output)
        raw_receipt_path = output / "raw_completion_receipt.json"
        raw_sidecar_path = raw_receipt_path.with_name(
            raw_receipt_path.name + ".sha256.json"
        )
        if raw_receipt_path.exists() or raw_receipt_path.is_symlink():
            validate_raw_completion_receipt(
                config,
                definition,
                rows,
                output,
                repair_missing_sidecar=True,
                tokenizer=load_frozen_decode_tokenizer(config),
            )
            raise ContractError("raw grid is already complete; run seal/validate instead")
        if raw_sidecar_path.exists() or raw_sidecar_path.is_symlink():
            raise ContractError("orphan raw-completion sidecar blocks GPU entry")
        attempt_id, attempt = _next_attempt_dir(output)
        write_live_status(
            output / "status.json",
            {
                "schema_version": 2,
                "status": "verifying_full_model_hashes_before_cuda",
                "request_fingerprint": definition["request_fingerprint"],
                "attempt_id": attempt_id,
            },
        )
        try:
            # Full weight hashing occurs before importing torch or calling any
            # CUDA API.  Its identity digest must equal the shallow frozen one.
            bindings = validate_method_bindings(
                config, deep_model_hash=True, verify_environment=True
            )
            if (
                bindings.runtime.get("deep_model_hash_verified") is not True
                or bindings.runtime.get("runtime_environment_verified") is not True
                or bindings.deep_mapper_hash_verified is not True
                or definition["runtime_binding_digest"] != bindings.runtime["binding_digest"]
                or definition["validated_method_bindings_digest"] != bindings.digest
                or definition["validated_mode_digests"] != dict(bindings.mode_digests)
            ):
                raise ContractError(
                    "deep-validated runtime/method binding differs from run definition"
                )
            deep_validation = {
                "schema_version": 2,
                "status": "passed_before_cuda_import",
                "deep_model_hash_verified": True,
                "runtime_environment_verified": True,
                "runtime_binding_digest": bindings.runtime["binding_digest"],
                "validated_method_bindings_digest": bindings.digest,
                "validated_mode_digests": dict(bindings.mode_digests),
                "runtime_provenance_receipt": bindings.runtime["receipt"],
                "source_model_validation": bindings.runtime["source_model"],
                "target_model_validation": bindings.runtime["target_model"],
                "mapper_manifest": bindings.runtime["mapper_manifest"],
                "runtime_code": bindings.runtime["runtime_code"],
                "runtime_environment": bindings.runtime["runtime_environment"],
                "mapper_payload_validation": bindings.mapper_payload_validation,
            }
            deep_ref = immutable_json(
                attempt / "deep_runtime_validation.json", deep_validation
            )
            write_sidecar(deep_ref)
            write_live_status(
                output / "status.json",
                {
                    "schema_version": 2,
                    "status": "waiting_for_idle_gpus",
                    "request_fingerprint": definition["request_fingerprint"],
                    "attempt_id": attempt_id,
                },
            )
            gate = wait_for_idle_gpus(config["runtime"]["idle_gate"])
            gate_ref = immutable_json(attempt / "gpu_idle_gate.json", gate)
            write_sidecar(gate_ref)

            # This is deliberately the first torch import in the entrypoint.
            import torch

            frozen_runtime = read_json(bindings.runtime["receipt"]["path"])["runtime"]
            cuda_validation = _cuda_runtime_evidence(torch, config, frozen_runtime)
            cuda_ref = immutable_json(
                attempt / "cuda_runtime_validation.json", cuda_validation
            )
            write_sidecar(cuda_ref)
            write_live_status(
                output / "status.json",
                {
                    "schema_version": 2,
                    "status": "loading_real_qwen3_models",
                    "request_fingerprint": definition["request_fingerprint"],
                    "attempt_id": attempt_id,
                },
            )
            components = build_runtime_components(config, bindings)
            _validate_method_provenance(components.method_provenance, config)
            _validate_model_sharding(components.model_sharding, config)
            provenance_ref = immutable_json(
                attempt / "method_provenance.json", components.method_provenance
            )
            write_sidecar(provenance_ref)
            sharding_ref = immutable_json(
                attempt / "model_sharding.json", components.model_sharding
            )
            write_sidecar(sharding_ref)
            write_live_status(
                output / "status.json",
                {
                    "schema_version": 2,
                    "status": "gpu_running",
                    "request_fingerprint": definition["request_fingerprint"],
                    "attempt_id": attempt_id,
                },
            )
            counts = execute_raw_grid(
                output=output,
                profile=profile,
                rows=rows,
                definition=definition,
                components=components,
                torch_module=torch,
                attempt_id=attempt_id,
            )
            receipt = seal_raw_completion_receipt(
                config=config,
                output=output,
                profile=profile,
                definition=definition,
                rows=rows,
                attempt_id=attempt_id,
                gpu_idle_gate_ref=gate_ref,
                deep_runtime_validation_ref=deep_ref,
                cuda_runtime_validation_ref=cuda_ref,
                method_provenance_ref=provenance_ref,
                model_sharding_ref=sharding_ref,
                execution_counts=counts,
                tokenizer=components.tokenizer,
            )
            write_live_status(
                output / "status.json",
                {
                    "schema_version": 2,
                    "status": "raw_gpu_complete",
                    "request_fingerprint": definition["request_fingerprint"],
                    "attempt_id": attempt_id,
                    "completed_raw_results": receipt["raw_result_count"],
                    "raw_completion_receipt_sha256": receipt[
                        "raw_completion_receipt_sha256"
                    ],
                },
            )
            return 0
        except BaseException as exc:
            failure = {
                "schema_version": 2,
                "status": "gpu_attempt_failed",
                "request_fingerprint": definition["request_fingerprint"],
                "attempt_id": attempt_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            failure_ref = immutable_json(attempt / "failure.json", failure)
            write_sidecar(failure_ref)
            write_live_status(
                output / "status.json",
                {
                    "schema_version": 2,
                    "status": "gpu_failed",
                    "request_fingerprint": definition["request_fingerprint"],
                    "attempt_id": attempt_id,
                    "failure": failure_ref.to_dict(),
                },
            )
            raise


__all__ = [
    "execute_gpu",
    "execute_raw_grid",
    "load_frozen_decode_tokenizer",
    "raw_inventory",
    "seal_raw_completion_receipt",
    "validate_raw_completion_receipt",
]
