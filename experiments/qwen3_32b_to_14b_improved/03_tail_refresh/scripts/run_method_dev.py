#!/usr/bin/env python3
"""Run the isolated R=1/8/32 method-dev QA pilot on eight GPUs.

The command refuses to touch CUDA unless ``--execute`` is explicit.  A normal
preflight validates frozen samples, hashes, phase membership, output
containment, and the exact mode set.  GPU artifacts are written only below
``03_tail_refresh/runs``; the repository-wide lock is used solely for mutual
exclusion with other eight-GPU experiments.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, BinaryIO, Mapping, Sequence


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CANDIDATE_ROOT.parents[2]
BASELINE_EXPERIMENT_ROOT = CANDIDATE_ROOT.parent.parent / "qwen3_32b_to_14b"
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
for _path in (CANDIDATE_ROOT, BASELINE_EXPERIMENT_ROOT, REPOSITORY_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.method_dev import (  # noqa: E402
    DEFAULT_CONFIG,
    PHASES,
    canonical_json_bytes,
    canonical_sha256,
    ensure_output_within,
    load_and_validate_config,
    read_json,
    sha256_file,
    token_ids_sha256,
)
from src.qa_metrics import qa_metrics  # noqa: E402
from src.tail_refresh import TailRefreshMappedPrefixCacheBuilder  # noqa: E402


MODES = ("target_standalone", "mapped_r1", "tail_refresh_r8", "tail_refresh_r32")
GLOBAL_GPU_LOCK = REPOSITORY_ROOT / "runs/gpu8_experiment.lock"


def _atomic_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable run artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"prepared JSONL line {line_number} is not an object")
            rows.append(value)
    return rows


def validate_prepared(config: Mapping[str, Any], phase: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared = Path(str(config["outputs"]["prepared"])).resolve()
    manifest = read_json(prepared / "manifest.json")
    if not isinstance(manifest, Mapping) or manifest.get("status") != "prepared":
        raise ValueError("prepared method-dev manifest is missing/incomplete")
    if manifest.get("protocol_id") != config["protocol_id"]:
        raise ValueError("prepared protocol ID differs from config")
    if manifest.get("config", {}).get("sha256") != config["config_sha256"]:
        raise ValueError("prepared inputs were produced by a different config")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("prepared manifest file inventory is malformed")
    for name in (
        "selected_samples.jsonl",
        "data_registry.json",
        "preflight.json",
        "interface_receipt.json",
    ):
        entry = files.get(name)
        path = prepared / name
        if not isinstance(entry, Mapping) or not path.is_file():
            raise ValueError(f"prepared file is missing: {name}")
        if sha256_file(path) != entry.get("sha256") or path.stat().st_size != entry.get("nbytes"):
            raise ValueError(f"prepared file identity changed: {name}")
    preflight = read_json(prepared / "preflight.json")
    if preflight.get("status") != "passed_method_dev_local_preflight":
        raise ValueError("local method-dev preflight did not pass")
    interface = read_json(prepared / "interface_receipt.json")
    implementation = interface.get("implementation")
    if (
        interface.get("status") != "frozen_method_dev_interface"
        or not isinstance(implementation, Mapping)
        or implementation.get("sha256")
        != sha256_file(CANDIDATE_ROOT / "src/tail_refresh.py")
        or implementation.get("nbytes")
        != (CANDIDATE_ROOT / "src/tail_refresh.py").stat().st_size
    ):
        raise ValueError("prepared tail-refresh interface/code binding changed")
    rows = [row for row in _read_jsonl(prepared / "selected_samples.jsonl") if phase in row["phase_membership"]]
    if not rows:
        raise ValueError(f"prepared phase {phase} is empty")
    for row in rows:
        ids = row.get("prompt_token_ids")
        if not isinstance(ids, list) or any(type(value) is not int for value in ids):
            raise ValueError("prepared prompt token IDs are malformed")
        if token_ids_sha256(ids) != row.get("prompt_token_ids_sha256"):
            raise ValueError("prepared prompt token IDs changed")
        if len(ids) != row.get("prompt_token_count") or len(ids) - 1 < 32:
            raise ValueError("prepared prompt length cannot implement registered R=32")
        if row.get("prompt_truncated") is not False:
            raise ValueError("method-dev prompt was truncated")
    expected_counts = manifest.get("phase_counts", {}).get(phase)
    observed_counts = {
        dataset: sum(row["dataset_id"] == dataset for row in rows)
        for dataset in ("hotpotqa_train", "musique_train")
    }
    if observed_counts != expected_counts:
        raise ValueError(f"phase counts changed: {observed_counts} != {expected_counts}")
    return rows, dict(manifest)


def result_path(output: Path, row: Mapping[str, Any], mode: str) -> Path:
    identity = hashlib.sha256(
        f"{row['dataset_id']}\0{row['sample_id']}".encode("utf-8")
    ).hexdigest()[:20]
    return output / "results" / str(row["dataset_id"]) / mode / f"{identity}.json"


def checkpoint_complete(path: Path, *, fingerprint: str, row: Mapping[str, Any], mode: str) -> bool:
    if not path.is_file():
        return False
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("status") == "complete"
        and value.get("request_fingerprint") == fingerprint
        and value.get("dataset_id") == row["dataset_id"]
        and value.get("sample_id") == row["sample_id"]
        and value.get("source_record_sha256") == row["source_record_sha256"]
        and value.get("prompt_token_ids_sha256") == row["prompt_token_ids_sha256"]
        and value.get("mode") == mode
        and isinstance(value.get("metrics"), Mapping)
        and isinstance(value.get("generation"), Mapping)
    )


def acquire_gpu_lock() -> BinaryIO:
    GLOBAL_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = GLOBAL_GPU_LOCK.open("a+b")
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


def _summarize(output: Path, rows: Sequence[Mapping[str, Any]], fingerprint: str) -> dict[str, Any]:
    raw: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    missing: list[dict[str, str]] = []
    for row in rows:
        for mode in MODES:
            path = result_path(output, row, mode)
            if not checkpoint_complete(path, fingerprint=fingerprint, row=row, mode=mode):
                missing.append({"dataset_id": row["dataset_id"], "sample_id": row["sample_id"], "mode": mode})
                continue
            value = read_json(path)
            raw.setdefault((str(row["dataset_id"]), mode), []).append(value)
    aggregates: dict[str, Any] = {}
    for (dataset, mode), values in sorted(raw.items()):
        metric_names = (
            "relaxed_qa_f1",
            "relaxed_qa_exact_match",
            "normalized_gold_containment",
            "musique_official_answer_f1",
            "musique_official_answer_em",
        )
        aggregates.setdefault(dataset, {})[mode] = {
            "count": len(values),
            **{
                metric: sum(float(value["metrics"][metric]) for value in values) / len(values)
                for metric in metric_names
            },
            "mean_generated_tokens": sum(int(value["generation"]["generated_tokens"]) for value in values) / len(values),
            "eos_rate": sum(value["generation"]["stop_reason"] == "eos" for value in values) / len(values),
            "stop_sequence_rate": sum(value["generation"]["stop_reason"] == "stop_sequence" for value in values) / len(values),
            "cap_rate": sum(bool(value["generation"]["cap_hit"]) for value in values) / len(values),
            "mean_wall_seconds": sum(float(value["systems"]["wall_seconds"]) for value in values) / len(values),
        }
    paired_fidelity: dict[str, Any] = {}
    if not missing:
        for dataset in ("hotpotqa_train", "musique_train"):
            by_mode = aggregates[dataset]
            target = float(by_mode["target_standalone"]["relaxed_qa_f1"])
            baseline = float(by_mode["mapped_r1"]["relaxed_qa_f1"])
            denominator = target - baseline
            paired_fidelity[dataset] = {}
            for mode in MODES[1:]:
                score = float(by_mode[mode]["relaxed_qa_f1"])
                paired_fidelity[dataset][mode] = {
                    "delta_vs_mapped_r1": score - baseline,
                    "delta_vs_native_target": score - target,
                    "gap_closure_vs_mapped_r1": (
                        (score - baseline) / denominator if denominator > 0 else None
                    ),
                }
    summary = {
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete",
        "request_fingerprint": fingerprint,
        "expected_atomic_results": len(rows) * len(MODES),
        "complete_atomic_results": len(rows) * len(MODES) - len(missing),
        "missing": missing,
        "aggregates": aggregates,
        "fidelity": paired_fidelity,
        "claim_boundary": "method-dev only; not a locked-test or paper superiority claim",
    }
    _atomic_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    execution = parser.add_mutually_exclusive_group(required=True)
    execution.add_argument("--preflight-only", action="store_true")
    execution.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_and_validate_config(args.config)
    rows, prepared_manifest = validate_prepared(config, args.phase)
    output = ensure_output_within(config["outputs"]["runs_root"], args.output_dir)
    definition = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "phase": args.phase,
        "modes": list(MODES),
        "sample_count": len(rows),
        "expected_atomic_results": len(rows) * len(MODES),
        "sample_identity_digest": canonical_sha256(
            [
                [row["dataset_id"], row["sample_id"], row["source_record_sha256"], row["prompt_token_ids_sha256"]]
                for row in rows
            ]
        ),
        "prepared_manifest_sha256": sha256_file(Path(config["outputs"]["prepared"]) / "manifest.json"),
        "config_sha256": config["config_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "tail_refresh_code_sha256": sha256_file(CANDIDATE_ROOT / "src/tail_refresh.py"),
        "metrics_code_sha256": sha256_file(CANDIDATE_ROOT / "src/qa_metrics.py"),
        "models": config["models"],
        "mapper": config["mapper"],
        "runtime": config["runtime"],
        "prompt": config["prompt"],
        "musique_protocol_binding": config["musique_protocol_binding"],
        "data_tier": "method-dev",
        "locked_test_access": False,
        "boundary_contract": {
            "engine_replays_final_boundary": True,
            "builder_native_tail_by_R": {"1": 0, "8": 7, "32": 31},
        },
    }
    fingerprint = canonical_sha256(definition)
    preflight_result = {
        "status": "passed",
        "gpu_used": False,
        "phase": args.phase,
        "sample_count": len(rows),
        "expected_atomic_results": len(rows) * len(MODES),
        "output": str(output),
        "request_fingerprint": fingerprint,
        "prepared_status": prepared_manifest["status"],
        "registered_modes": list(MODES),
        "generation_cap": config["prompt"]["max_generation_tokens"],
        "stop_sequences": config["prompt"]["stop_sequences"],
        "musique_config_sha256": config["musique_protocol_binding"]["config"]["sha256"],
        "musique_registry_manifest_sha256": config["musique_protocol_binding"]["registry_manifest"]["sha256"],
    }
    if args.preflight_only:
        print(json.dumps(preflight_result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0,1,2,3,4,5,6,7":
        raise RuntimeError(
            "GPU execution requires CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable_json(output / "run_definition.json", {**definition, "request_fingerprint": fingerprint})
    lock = acquire_gpu_lock()
    try:
        import torch
        from transformers import AutoTokenizer
        from kvcache_transfer.evaluation import TransferEvaluationEngine
        from kvcache_transfer.modeling import load_qwen3

        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            raise RuntimeError(f"exactly eight visible CUDA GPUs are required, got {torch.cuda.device_count()}")
        _atomic_json(output / "status.json", {"status": "loading_models", "request_fingerprint": fingerprint})
        tokenizer = AutoTokenizer.from_pretrained(
            config["models"]["target_14b"]["path"], local_files_only=True, use_fast=True
        )
        runtime = config["runtime"]
        # Modeling roles retain the repository's historical forward naming:
        # reverse semantic source 32B is native role=target; reverse semantic
        # target 14B is native role=source.
        target = load_qwen3(
            config["models"]["target_14b"]["path"],
            devices=runtime["target_devices"],
            layer_counts=runtime["target_layer_counts"],
            role="source",
            attention_implementation=runtime["attention_implementation"],
        )
        source = load_qwen3(
            config["models"]["source_32b"]["path"],
            devices=runtime["source_devices"],
            layer_counts=runtime["source_layer_counts"],
            role="target",
            attention_implementation=runtime["attention_implementation"],
        )
        _atomic_json(
            output / "model_sharding.json",
            {
                "source_32b": getattr(source, "_kvcache_transfer_layer_sharding_evidence"),
                "target_14b": getattr(target, "_kvcache_transfer_layer_sharding_evidence"),
            },
        )
        r1_builder = TailRefreshMappedPrefixCacheBuilder(
            source_model=source,
            target_model=target,
            mapper_manifest=config["mapper"]["manifest"],
            replay_tokens=1,
            token_chunk=int(runtime["mapper_token_chunk"]),
        )
        builders = {
            "mapped_r1": r1_builder,
            "tail_refresh_r8": r1_builder.variant(8),
            "tail_refresh_r32": r1_builder.variant(32),
        }
        _atomic_json(output / "method_provenance.json", {mode: builder.provenance for mode, builder in builders.items()})
        max_length = int(config["prompt"]["max_length"])
        cap = int(config["prompt"]["max_generation_tokens"])
        engines = {
            "target_standalone": TransferEvaluationEngine(
                target_model=target,
                tokenizer=tokenizer,
                mode="target_standalone",
                max_length=max_length,
                max_gen_tokens=cap,
            ),
            **{
                mode: TransferEvaluationEngine(
                    target_model=target,
                    tokenizer=tokenizer,
                    mode="mapped",
                    mapped_prefix_builder=builder,
                    max_length=max_length,
                    max_gen_tokens=cap,
                )
                for mode, builder in builders.items()
            },
        }
        _atomic_json(output / "status.json", {"status": "running", "request_fingerprint": fingerprint})
        failures = 0
        for row_index, row in enumerate(rows):
            target_binding: str | None = None
            for mode in MODES:
                destination = result_path(output, row, mode)
                if checkpoint_complete(destination, fingerprint=fingerprint, row=row, mode=mode):
                    existing = read_json(destination)
                    if mode == "target_standalone":
                        target_binding = canonical_sha256(existing)
                    print(f"[resume] {row['dataset_id']}/{row['sample_id']}/{mode}")
                    continue
                engine = engines[mode]
                for device in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device)
                _sync_cuda(torch)
                started = time.perf_counter()
                try:
                    generation = engine.generate_token_ids(
                        row["prompt_token_ids"],
                        {
                            "max_gen_toks": cap,
                            "do_sample": False,
                            "temperature": 0.0,
                            "until": config["prompt"]["stop_sequences"],
                        },
                        truncation="error",
                    )
                    _sync_cuda(torch)
                    elapsed = time.perf_counter() - started
                    metrics = qa_metrics(generation.text, row["references"])
                    generation_ids = [int(value) for value in generation.generated_token_ids]
                    tail_audit = None if mode == "target_standalone" else builders[mode].last_call_audit
                    payload = {
                        "schema_version": 1,
                        "status": "complete",
                        "request_fingerprint": fingerprint,
                        "phase": args.phase,
                        "dataset_id": row["dataset_id"],
                        "sample_id": row["sample_id"],
                        "stratum": row["stratum"],
                        "source_record_sha256": row["source_record_sha256"],
                        "mode": mode,
                        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
                        "prompt_token_count": row["prompt_token_count"],
                        "prompt_truncated": False,
                        "prediction": generation.text,
                        "references": row["references"],
                        "metrics": metrics,
                        "generation": {
                            "requested_max_tokens": cap,
                            "generated_tokens": generation.generated_token_count,
                            "generated_token_ids": generation_ids,
                            "generated_token_ids_sha256": token_ids_sha256(generation_ids),
                            "stop_reason": generation.stop_reason,
                            "cap_hit": generation.cap_hit,
                        },
                        "tail_refresh_audit": tail_audit,
                        "target_checkpoint_binding": target_binding if mode != "target_standalone" else None,
                        "systems": {"wall_seconds": elapsed, "gpu_peak": _gpu_peaks(torch)},
                    }
                    _atomic_json(destination, payload)
                    destination.with_suffix(".failure.json").unlink(missing_ok=True)
                    if mode == "target_standalone":
                        target_binding = canonical_sha256(payload)
                    print(
                        f"[complete] {row_index + 1}/{len(rows)} {row['dataset_id']}/"
                        f"{row['sample_id']}/{mode} f1={metrics['relaxed_qa_f1']:.6f}"
                    )
                except BaseException as exc:
                    failures += 1
                    _atomic_json(
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
                    print(f"[failed] {row['dataset_id']}/{row['sample_id']}/{mode}: {exc}", file=sys.stderr)
                finally:
                    engine.clear_prefix_cache()
                if failures:
                    break
            if failures:
                break
            _summarize(output, rows, fingerprint)
        summary = _summarize(output, rows, fingerprint)
        final_status = "failed" if failures else summary["status"]
        _atomic_json(
            output / "status.json",
            {
                "status": final_status,
                "failures": failures,
                "request_fingerprint": fingerprint,
                "summary_sha256": sha256_file(output / "summary.json"),
            },
        )
        return 0 if final_status == "complete" else 1
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
