"""Guarded five-mode paired runner with durable per-sample checkpoints."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, BinaryIO, Mapping, Sequence

from .contracts import (
    ArtifactRef,
    ContractError,
    DATASETS,
    DEFAULT_CONFIG,
    MODES,
    PREPARED_ROOT,
    PROFILES,
    ROOT,
    atomic_json,
    canonical_sha256,
    code_inventory,
    immutable_json,
    load_config,
    read_json,
    read_jsonl,
    require_child,
    sha256_file,
    token_ids_sha256,
    write_sidecar,
)
from .metrics import score_answer
from .prepare_data import validate_prepared_row
from .runtime_adapter import (
    MethodBindings,
    build_runtime_components,
    validate_method_bindings,
)
from .statistics import aggregate_paired_results


def load_prepared(
    config: Mapping[str, Any], profile: str
) -> tuple[list[dict[str, Any]], dict[str, Any], ArtifactRef]:
    if profile not in PROFILES:
        raise ContractError(f"unsupported profile: {profile}")
    prepared = Path(str(config["outputs"]["prepared"])).resolve()
    if prepared != PREPARED_ROOT.resolve():
        raise ContractError("prepared root changed")
    manifest_path = prepared / "manifest.json"
    manifest_ref = ArtifactRef(
        path=str(manifest_path),
        nbytes=manifest_path.stat().st_size if manifest_path.is_file() else -1,
        sha256=sha256_file(manifest_path) if manifest_path.is_file() else "",
    )
    if not manifest_path.is_file():
        raise ContractError("prepared manifest is missing; run scripts/prepare.py --prepare")
    manifest = read_json(manifest_path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "prepared"
        or manifest.get("protocol_id") != config["protocol_id"]
        or manifest.get("config", {}).get("sha256") != config["config_sha256"]
    ):
        raise ContractError("prepared manifest/config binding changed")
    if manifest.get("code") != code_inventory():
        raise ContractError("code changed after data preparation; re-prepare before running")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "samples.jsonl",
        "data_registry.json",
        "preflight.json",
    }:
        raise ContractError("prepared file inventory changed")
    verified: dict[str, Path] = {}
    for name, raw in files.items():
        reference = ArtifactRef.parse(raw, label=f"prepared {name}")
        expected = (prepared / name).resolve()
        if Path(reference.path).resolve() != expected:
            raise ContractError(f"prepared {name} path changed")
        verified[name] = reference.verify(label=f"prepared {name}")
    preflight = read_json(verified["preflight.json"])
    if (
        preflight.get("status") != "passed_cpu_prepare"
        or preflight.get("gpu_used") is not False
        or preflight.get("cuda_api_called") is not False
    ):
        raise ContractError("prepared CPU preflight did not pass")
    registry = read_json(verified["data_registry.json"])
    if (
        registry.get("tier") != "method-dev"
        or registry.get("official_dev_raw_path_opened") is not False
        or registry.get("locked_test_metrics_observed") is not False
        or registry.get("musique_registry_audit", {}).get("official_dev_state")
        != "sealed_until_candidate_freeze"
    ):
        raise ContractError("prepared data registry violates method-dev isolation")
    rows = read_jsonl(verified["samples.jsonl"])
    max_length = int(config["generation"]["max_length"])
    cap = int(config["generation"]["max_new_tokens"])
    for row in rows:
        validate_prepared_row(row, max_length=max_length, cap=cap)
    selected = [row for row in rows if profile in row["profile_membership"]]
    expected_counts = config["profiles"][profile]
    observed_counts = {
        dataset: sum(row["dataset_id"] == dataset for row in selected)
        for dataset in DATASETS
    }
    if observed_counts != expected_counts:
        raise ContractError(
            f"prepared {profile} counts changed: {observed_counts} != {expected_counts}"
        )
    if canonical_sha256([row["sample_binding_sha256"] for row in rows]) != manifest.get("sample_bindings_sha256"):
        raise ContractError("prepared sample binding inventory changed")
    return selected, manifest, manifest_ref


def mode_binding_sha256(config: Mapping[str, Any], bindings: MethodBindings, mode: str) -> str:
    if mode == "native14":
        value: Any = {
            "method": config["methods"]["native14"],
            "target_model_identity": bindings.runtime_receipt["target_model"],
        }
    elif mode == "nvidia_k20":
        value = {
            "method": config["methods"]["nvidia_k20"],
            "upstream_runtime_receipt": bindings.runtime_receipt_ref.to_dict(),
        }
    elif mode == "shrinkage":
        value = {
            "method": config["methods"]["shrinkage"],
            "validated_binding_digest": bindings.shrinkage["binding_digest"],
        }
    elif mode in {"tail_refresh_r8", "tail_refresh_r32"}:
        replay = 8 if mode.endswith("r8") else 32
        value = {
            "method": config["methods"]["tail_refresh"],
            "validated_binding_digest": bindings.tail_refresh["binding_digest"],
            "replay_tokens": replay,
            "underlying_baseline_manifest": bindings.baseline_manifest_ref.to_dict(),
        }
    else:
        raise ContractError(f"unknown mode: {mode}")
    return canonical_sha256(value)


def result_path(output: Path, row: Mapping[str, Any], mode: str) -> Path:
    identity = hashlib.sha256(
        f"{row['dataset_id']}\0{row['sample_id']}".encode("utf-8")
    ).hexdigest()[:24]
    return output / "results" / str(row["dataset_id"]) / mode / f"{identity}.json"


def _decode_exact(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def validate_generation(
    generation: Any,
    *,
    tokenizer: Any | None,
    expected_cap: int,
    stop_sequences: Sequence[str],
    prediction: str,
) -> None:
    if not isinstance(generation, Mapping):
        raise ContractError("checkpoint lacks generation evidence")
    required = {
        "requested_max_tokens",
        "generated_tokens",
        "generated_token_ids",
        "generated_token_ids_sha256",
        "decoded_generated_token_ids",
        "stop_reason",
        "stop_sequences",
        "cap_hit",
    }
    if set(generation) != required:
        raise ContractError("generation evidence schema changed")
    ids = generation.get("generated_token_ids")
    if not isinstance(ids, list) or any(type(value) is not int or value < 0 for value in ids):
        raise ContractError("generated_token_ids must be exact non-negative integers")
    if generation.get("generated_tokens") != len(ids):
        raise ContractError("generated token count/list length mismatch")
    if generation.get("generated_token_ids_sha256") != token_ids_sha256(ids):
        raise ContractError("generated token IDs changed")
    if generation.get("requested_max_tokens") != expected_cap:
        raise ContractError("generation cap changed")
    if generation.get("stop_sequences") != list(stop_sequences):
        raise ContractError("generation stop sequences changed")
    reason = generation.get("stop_reason")
    cap_hit = generation.get("cap_hit")
    if reason not in {"eos", "stop_sequence", "max_tokens"} or not isinstance(cap_hit, bool):
        raise ContractError("generation stop evidence is invalid")
    if reason == "max_tokens":
        if len(ids) != expected_cap or cap_hit is not True:
            raise ContractError("max_tokens stop is inconsistent")
    elif len(ids) > expected_cap or cap_hit is not False:
        raise ContractError("non-cap stop is inconsistent")
    decoded = generation.get("decoded_generated_token_ids")
    if not isinstance(decoded, str):
        raise ContractError("exact decoded continuation is missing")
    if tokenizer is not None and _decode_exact(tokenizer, ids) != decoded:
        raise ContractError("decoded continuation does not match generated token IDs")
    if reason == "stop_sequence":
        stop_offsets = [decoded.find(stop) for stop in stop_sequences]
        stop_offsets = [offset for offset in stop_offsets if offset >= 0]
        if not stop_offsets:
            raise ContractError("stop_sequence reason lacks the registered stop marker")
        stop_offset = min(stop_offsets)
        suffix = decoded[stop_offset:]
        if prediction != decoded[:stop_offset] or suffix not in stop_sequences:
            raise ContractError("stop-trimmed prediction is not the exact first registered stop boundary")
    elif prediction != decoded:
        raise ContractError("prediction is not the exact decoded continuation")


def validate_checkpoint(
    path: Path,
    *,
    row: Mapping[str, Any],
    mode: str,
    request_fingerprint: str,
    method_binding: str,
    tokenizer: Any | None,
    expected_target_sha256: str | None,
    expected_baseline_sha256: str | None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"checkpoint is not a regular file: {path}")
    value = read_json(path)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": request_fingerprint,
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "sample_binding_sha256": row["sample_binding_sha256"],
        "source_record_sha256": row["source_record_sha256"],
        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        "mode": mode,
        "method_binding_sha256": method_binding,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ContractError(f"checkpoint {field} mismatch: {path}")
    if value.get("references") != row["references"]:
        raise ContractError("checkpoint references changed")
    if value.get("target_result_sha256") != expected_target_sha256:
        raise ContractError("checkpoint target binding changed")
    if value.get("baseline_result_sha256") != expected_baseline_sha256:
        raise ContractError("checkpoint baseline binding changed")
    prediction = value.get("prediction")
    if not isinstance(prediction, str):
        raise ContractError("checkpoint prediction is not text")
    validate_generation(
        value.get("generation"),
        tokenizer=tokenizer,
        expected_cap=int(row["generation_cap"]),
        stop_sequences=row["stop_sequences"],
        prediction=prediction,
    )
    if value.get("metrics") != score_answer(row["dataset_id"], prediction, row["references"]):
        raise ContractError("checkpoint metrics do not recompute")
    if value.get("truncation") != row["truncation"]:
        raise ContractError("checkpoint prompt truncation evidence changed")
    audit = value.get("builder_audit")
    if mode.startswith("tail_refresh_r"):
        replay = 8 if mode.endswith("r8") else 32
        if not isinstance(audit, Mapping):
            raise ContractError("tail-refresh checkpoint lacks causal audit")
        expected_tail = replay - 1
        if (
            audit.get("requested_total_native_prompt_tokens") != replay
            or audit.get("builder_native_tail_tokens") != expected_tail
            or audit.get("engine_boundary_tokens") != 1
            or audit.get("mapped_prefix_tokens") + expected_tail + 1
            != row["prompt_token_count"]
        ):
            raise ContractError("tail-refresh checkpoint causal partition changed")
    elif audit is not None:
        raise ContractError("non-tail checkpoint unexpectedly contains tail audit")
    return value


def checkpoint_if_complete(
    path: Path,
    **kwargs: Any,
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return validate_checkpoint(path, **kwargs)


def acquire_gpu_lock(path: str | Path) -> BinaryIO:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _sync_cuda(torch: Any) -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def _gpu_peaks(torch: Any) -> dict[str, Any]:
    return {
        str(index): {
            "allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
            "reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
        }
        for index in range(torch.cuda.device_count())
    }


def build_run_definition(
    config: Mapping[str, Any],
    profile: str,
    rows: Sequence[Mapping[str, Any]],
    prepared_manifest_ref: ArtifactRef,
    bindings: MethodBindings,
    confirmation_prerequisite: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    definition = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "direction": config["direction"],
        "tier": "method-dev",
        "profile": profile,
        "modes": list(MODES),
        "sample_count": len(rows),
        "expected_atomic_results": len(rows) * len(MODES),
        "sample_bindings_sha256": canonical_sha256(
            [row["sample_binding_sha256"] for row in rows]
        ),
        "prepared_manifest": prepared_manifest_ref.to_dict(),
        "config": {
            "path": config["config_path"],
            "nbytes": Path(config["config_path"]).stat().st_size,
            "sha256": config["config_sha256"],
        },
        "code": code_inventory(),
        "runtime_binding_digest": bindings.digest,
        "method_binding_sha256": {
            mode: mode_binding_sha256(config, bindings, mode) for mode in MODES
        },
        "generation": config["generation"],
        "prompts": config["prompts"],
        "metrics": config["metrics"],
        "models_and_topology": config["runtime"],
        "method_artifacts": config["methods"],
        "provenance": config["provenance"],
        "confirmation_gate": config["confirmation_gate"],
        "causal_boundary": {
            "engine_replays_final_prompt_token": True,
            "nvidia_k20_total_native_prompt_tokens": 1,
            "tail_r8_builder_native_plus_boundary": "7+1",
            "tail_r32_builder_native_plus_boundary": "31+1",
        },
        "official_dev_opened": False,
        "locked_test_metrics_observed": False,
        "claim_boundary": config["claim_boundary"],
    }
    if confirmation_prerequisite is not None:
        definition["confirmation_prerequisite"] = dict(confirmation_prerequisite)
    return {**definition, "request_fingerprint": canonical_sha256(definition)}


def contract_preflight(
    config_path: str | Path,
    *,
    profile: str,
    deep_model_hash: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], MethodBindings]:
    config = load_config(config_path, verify_inputs=True)
    rows, _prepared, prepared_ref = load_prepared(config, profile)
    bindings = validate_method_bindings(config, deep_model_hash=deep_model_hash)
    definition = build_run_definition(config, profile, rows, prepared_ref, bindings)
    result = {
        "schema_version": 1,
        "status": "passed_cpu_contract_preflight",
        "gpu_used": False,
        "cuda_api_called": False,
        "profile": profile,
        "sample_count": len(rows),
        "expected_atomic_results": len(rows) * len(MODES),
        "modes": list(MODES),
        "generation_cap": config["generation"]["max_new_tokens"],
        "stop_sequences": config["generation"]["stop_sequences"],
        "runtime_binding_digest": bindings.digest,
        "request_fingerprint": definition["request_fingerprint"],
        "official_dev_opened": False,
    }
    return result, definition, rows, bindings


def _wait_for_idle_gpus(config: Mapping[str, Any]) -> Mapping[str, Any]:
    for path in (REPOSITORY_ROOT := Path("/home/wby/llm-cache/nvidia-kvcache"), REPOSITORY_ROOT / "experiments/qwen3_32b_to_14b"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from reverse_kv.reverse_latency import wait_for_idle_gpus

    return wait_for_idle_gpus(
        {
            "runtime": {
                "require_cuda_visible_devices": config["runtime"]["required_cuda_visible_devices"],
                "idle_memory_limit_mib": 64,
                "idle_consecutive_samples": 12,
                "idle_sample_interval_seconds": 5,
                "idle_timeout_seconds": 86400,
            }
        }
    )


def _method_provenance_for_result(components: Any, mode: str) -> Any:
    if mode == "native14":
        return components.method_provenance["native14"]
    return components.method_provenance[mode]


def _confirmation_prerequisite(
    config_path: str | Path, config: Mapping[str, Any], profile: str, screen_output_dir: str | Path | None
) -> Mapping[str, Any] | None:
    if profile != "confirm":
        return None
    if screen_output_dir is None:
        raise ContractError("confirm execution requires --screen-output-dir for the pre-registered gate")
    from .validator import validate_completed_run

    validated = validate_completed_run(config_path, profile="screen", output_dir=screen_output_dir)
    if validated.get("status") != config["confirmation_gate"]["required_screen_validation_status"]:
        raise ContractError("screen output did not satisfy the confirmation validation gate")
    receipt_path = require_child(
        config["outputs"]["runs_root"], Path(screen_output_dir) / "completion_receipt.json", label="screen completion receipt"
    )
    receipt = ArtifactRef(
        path=str(receipt_path), nbytes=receipt_path.stat().st_size, sha256=sha256_file(receipt_path)
    )
    receipt.verify(label="screen completion receipt")
    return {
        "rule_id": config["confirmation_gate"]["rule_id"],
        "validated_screen": validated,
        "screen_completion_receipt": receipt.to_dict(),
    }


def execute(
    config_path: str | Path,
    *,
    profile: str,
    output_dir: str | Path,
    screen_output_dir: str | Path | None = None,
) -> int:
    config = load_config(config_path, verify_inputs=True)
    output = require_child(config["outputs"]["runs_root"], output_dir, label="run output")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != config["runtime"]["required_cuda_visible_devices"]:
        raise ContractError(
            "GPU execution requires CUDA_VISIBLE_DEVICES="
            + config["runtime"]["required_cuda_visible_devices"]
        )
    confirmation_prerequisite = _confirmation_prerequisite(
        config_path, config, profile, screen_output_dir
    )
    lock = acquire_gpu_lock(config["runtime"]["global_gpu_lock"])
    try:
        # Deep model hashing and all candidate receipts are verified before the
        # first CUDA API call.  The lock prevents a second eight-GPU job from
        # racing between validation and model load.
        preflight, definition, rows, bindings = contract_preflight(
            config_path, profile=profile, deep_model_hash=True
        )
        if confirmation_prerequisite is not None:
            _selected, _manifest, prepared_ref = load_prepared(config, profile)
            definition = build_run_definition(
                config,
                profile,
                rows,
                prepared_ref,
                bindings,
                confirmation_prerequisite=confirmation_prerequisite,
            )
        output.mkdir(parents=True, exist_ok=True)
        definition_ref = immutable_json(output / "run_definition.json", definition)
        write_sidecar(definition_ref)
        fingerprint = definition["request_fingerprint"]
        atomic_json(
            output / "status.json",
            {"status": "waiting_for_idle_gpus", "request_fingerprint": fingerprint},
        )
        idle = _wait_for_idle_gpus(config)
        gate_ref = immutable_json(output / "gpu_idle_gate.json", idle)
        write_sidecar(gate_ref)

        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            raise ContractError(
                f"exactly eight visible CUDA GPUs are required, got {torch.cuda.device_count()}"
            )
        components = build_runtime_components(config, bindings)
        atomic_json(output / "method_provenance.json", components.method_provenance)
        atomic_json(
            output / "model_sharding.json",
            {
                "source_32b": getattr(
                    components.source_model, "_kvcache_transfer_layer_sharding_evidence"
                ),
                "target_14b": getattr(
                    components.target_model, "_kvcache_transfer_layer_sharding_evidence"
                ),
            },
        )
        atomic_json(
            output / "status.json",
            {"status": "running", "request_fingerprint": fingerprint},
        )
        completed_cells = 0
        for row_index, row in enumerate(rows):
            target_sha: str | None = None
            baseline_sha: str | None = None
            for mode in MODES:
                destination = result_path(output, row, mode)
                method_binding = definition["method_binding_sha256"][mode]
                existing = checkpoint_if_complete(
                    destination,
                    row=row,
                    mode=mode,
                    request_fingerprint=fingerprint,
                    method_binding=method_binding,
                    tokenizer=components.tokenizer,
                    expected_target_sha256=target_sha if mode != "native14" else None,
                    expected_baseline_sha256=(
                        baseline_sha
                        if mode in {"shrinkage", "tail_refresh_r8", "tail_refresh_r32"}
                        else None
                    ),
                )
                if existing is not None:
                    if mode == "native14":
                        target_sha = sha256_file(destination)
                    elif mode == "nvidia_k20":
                        baseline_sha = sha256_file(destination)
                    completed_cells += 1
                    continue
                if mode != "native14" and target_sha is None:
                    raise ContractError("candidate result cannot precede target checkpoint")
                if mode in {"shrinkage", "tail_refresh_r8", "tail_refresh_r32"} and baseline_sha is None:
                    raise ContractError("candidate result cannot precede baseline checkpoint")
                engine = components.engines[mode]
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
                _sync_cuda(torch)
                started = time.perf_counter()
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
                    _sync_cuda(torch)
                    elapsed = time.perf_counter() - started
                    generated_ids = [int(value) for value in generated.generated_token_ids]
                    decoded = _decode_exact(components.tokenizer, generated_ids)
                    audit = (
                        dict(components.builders[mode].last_call_audit)
                        if mode.startswith("tail_refresh_r")
                        else None
                    )
                    payload = {
                        "schema_version": 1,
                        "status": "complete",
                        "request_fingerprint": fingerprint,
                        "profile": profile,
                        "dataset_id": row["dataset_id"],
                        "sample_id": row["sample_id"],
                        "sample_binding_sha256": row["sample_binding_sha256"],
                        "source_record_sha256": row["source_record_sha256"],
                        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
                        "prompt_token_count": row["prompt_token_count"],
                        "truncation": row["truncation"],
                        "mode": mode,
                        "method_binding_sha256": method_binding,
                        "method_provenance": _method_provenance_for_result(components, mode),
                        "target_result_sha256": target_sha if mode != "native14" else None,
                        "baseline_result_sha256": (
                            baseline_sha
                            if mode in {"shrinkage", "tail_refresh_r8", "tail_refresh_r32"}
                            else None
                        ),
                        "prediction": generated.text,
                        "references": row["references"],
                        "metrics": score_answer(
                            row["dataset_id"], generated.text, row["references"]
                        ),
                        "generation": {
                            "requested_max_tokens": row["generation_cap"],
                            "generated_tokens": len(generated_ids),
                            "generated_token_ids": generated_ids,
                            "generated_token_ids_sha256": token_ids_sha256(generated_ids),
                            "decoded_generated_token_ids": decoded,
                            "stop_reason": generated.stop_reason,
                            "stop_sequences": row["stop_sequences"],
                            "cap_hit": generated.cap_hit,
                        },
                        "builder_audit": audit,
                        "systems": {
                            "wall_seconds": elapsed,
                            "gpu_peak": _gpu_peaks(torch),
                        },
                    }
                    atomic_json(destination, payload)
                    validate_checkpoint(
                        destination,
                        row=row,
                        mode=mode,
                        request_fingerprint=fingerprint,
                        method_binding=method_binding,
                        tokenizer=components.tokenizer,
                        expected_target_sha256=target_sha if mode != "native14" else None,
                        expected_baseline_sha256=(
                            baseline_sha
                            if mode in {"shrinkage", "tail_refresh_r8", "tail_refresh_r32"}
                            else None
                        ),
                    )
                    destination.with_suffix(".failure.json").unlink(missing_ok=True)
                    if mode == "native14":
                        target_sha = sha256_file(destination)
                    elif mode == "nvidia_k20":
                        baseline_sha = sha256_file(destination)
                    completed_cells += 1
                except BaseException as exc:
                    atomic_json(
                        destination.with_suffix(".failure.json"),
                        {
                            "status": "failed",
                            "request_fingerprint": fingerprint,
                            "dataset_id": row["dataset_id"],
                            "sample_id": row["sample_id"],
                            "mode": mode,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    atomic_json(
                        output / "status.json",
                        {
                            "status": "failed",
                            "request_fingerprint": fingerprint,
                            "completed_atomic_results": completed_cells,
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
            atomic_json(
                output / "status.json",
                {
                    "status": "running",
                    "request_fingerprint": fingerprint,
                    "completed_samples": row_index + 1,
                    "total_samples": len(rows),
                    "completed_atomic_results": completed_cells,
                },
            )

        result_records: list[dict[str, Any]] = []
        inventory: list[dict[str, Any]] = []
        for row in rows:
            target_path = result_path(output, row, "native14")
            baseline_path = result_path(output, row, "nvidia_k20")
            target_hash = sha256_file(target_path)
            baseline_hash = sha256_file(baseline_path)
            for mode in MODES:
                path = result_path(output, row, mode)
                value = validate_checkpoint(
                    path,
                    row=row,
                    mode=mode,
                    request_fingerprint=fingerprint,
                    method_binding=definition["method_binding_sha256"][mode],
                    tokenizer=components.tokenizer,
                    expected_target_sha256=target_hash if mode != "native14" else None,
                    expected_baseline_sha256=(
                        baseline_hash
                        if mode in {"shrinkage", "tail_refresh_r8", "tail_refresh_r32"}
                        else None
                    ),
                )
                result_records.append(value)
                reference = ArtifactRef(
                    path=str(path), nbytes=path.stat().st_size, sha256=sha256_file(path)
                )
                inventory.append(reference.to_dict())
        summary = aggregate_paired_results(
            result_records,
            profile=profile,
            request_fingerprint=fingerprint,
            bootstrap_seed=int(config["metrics"]["bootstrap_seed"]),
            bootstrap_replicates=int(config["metrics"]["bootstrap_replicates"]),
            confidence_level=float(config["metrics"]["confidence_level"]),
        )
        summary_ref = atomic_json(output / "summary.json", summary)
        write_sidecar(summary_ref)
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "request_fingerprint": fingerprint,
            "profile": profile,
            "run_definition": definition_ref.to_dict(),
            "gpu_idle_gate": gate_ref.to_dict(),
            "summary": summary_ref.to_dict(),
            "result_count": len(inventory),
            "expected_result_count": len(rows) * len(MODES),
            "result_inventory": inventory,
            "result_inventory_sha256": canonical_sha256(inventory),
            "runtime_binding_digest": bindings.digest,
            "paired_complete": True,
            "official_dev_opened": False,
            "claim_boundary": config["claim_boundary"],
        }
        receipt_ref = immutable_json(output / "completion_receipt.json", receipt)
        write_sidecar(receipt_ref)
        atomic_json(
            output / "status.json",
            {
                "status": "complete",
                "request_fingerprint": fingerprint,
                "completion_receipt_sha256": receipt_ref.sha256,
                "completed_atomic_results": len(inventory),
            },
        )
        return 0
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


__all__ = [
    "build_run_definition",
    "checkpoint_if_complete",
    "contract_preflight",
    "execute",
    "load_prepared",
    "mode_binding_sha256",
    "result_path",
    "validate_checkpoint",
    "validate_generation",
]
