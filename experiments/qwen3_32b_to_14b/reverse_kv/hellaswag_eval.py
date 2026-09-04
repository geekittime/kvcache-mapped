"""Sealed paper-scope HellaSwag entry for Qwen3 32B -> 14B."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .eval_common import hashed_json, verify_hashed_json
from .eval_adapter import FORMAL_HELLASWAG_TOPOLOGY, run_evaluation
from .protocol import (
    EXPERIMENT_ROOT,
    canonical_sha256,
    read_json,
    sha256_file,
)
from .reverse_latency import (
    GLOBAL_GPU_LOCK,
    acquire_gpu_lock,
    wait_for_idle_gpus,
)


ALLOWED_MODES = ("target_standalone", "mapped")
TASK = "hellaswag"
SPLIT_SCOPE = "full_validation_10042_no_limit"
EXPECTED_SAMPLES = 10_042
EXPECTED_VALIDATION_SHA256 = (
    "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd"
)
FORMAL_RECEIPT = "formal_hellaswag_receipt.json"
GPU_IDLE_GATE = "gpu_idle_gate.json"
GPU_IDLE_RUNTIME = {
    "require_cuda_visible_devices": "0,1,2,3,4,5,6,7",
    "idle_memory_limit_mib": 64,
    "idle_consecutive_samples": 12,
    "idle_sample_interval_seconds": 5,
    "idle_timeout_seconds": 86_400,
}
FORMAL_RUNTIME_TOPOLOGY = {
    role: {
        "devices": list(contract["devices"]),
        "layer_counts": list(contract["layer_counts"]),
        "model": str(contract["model"]),
        "explicit_exact_counts": True,
    }
    for role, contract in FORMAL_HELLASWAG_TOPOLOGY.items()
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only full HellaSwag for the 14B target standalone and the "
            "32B->14B mapped cache. Modes, task, split, shots, and metric are sealed."
        )
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def sealed_shared_argv(args: argparse.Namespace) -> list[str]:
    result = [
        "--modes",
        ",".join(ALLOWED_MODES),
        "--tasks",
        TASK,
        # Formal HellaSwag always retains every sample.  The independent
        # validator below recomputes acc_norm from these rows, so an aggregate
        # metric or sample-count-only checkpoint can never be resumed.
        "--log-samples",
    ]
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        try:
            output.relative_to(EXPERIMENT_ROOT)
        except ValueError as exc:
            raise ValueError("HellaSwag output must remain inside the reverse experiment") from exc
        result.extend(("--output-dir", str(output)))
    if args.dry_run:
        result.append("--dry-run")
    # Deliberately no --limit and no user-controlled --modes/--tasks.
    return result


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"formal HellaSwag result lacks {label}")
    return value


def _require_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"formal HellaSwag {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"formal HellaSwag {label} is not finite")
    return result


def validate_run_definition(value: Mapping[str, Any]) -> str:
    """Validate the immutable shared-evaluator request before trusting resume."""

    fingerprint = str(value.get("request_fingerprint", ""))
    unsigned = dict(value)
    unsigned.pop("request_fingerprint", None)
    if len(fingerprint) != 64 or canonical_sha256(unsigned) != fingerprint:
        raise ValueError("formal HellaSwag request fingerprint is invalid")
    fixed = {
        "modes": list(ALLOWED_MODES),
        "tasks": [TASK],
        "limit": None,
        "wikitext_max_chunks": None,
        "log_samples": True,
        "completion_mode": True,
        "apply_chat_template": False,
        "batch_size": 1,
        "boundary_replay": True,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"formal HellaSwag request changed {field}")
    config_path = EXPERIMENT_ROOT / "config/eval.yaml"
    endpoint_fixed = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_model": FORMAL_RUNTIME_TOPOLOGY["source"]["model"],
        "target_model": FORMAL_RUNTIME_TOPOLOGY["target"]["model"],
        "source_devices": FORMAL_RUNTIME_TOPOLOGY["source"]["devices"],
        "target_devices": FORMAL_RUNTIME_TOPOLOGY["target"]["devices"],
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "max_gen_tokens": 1024,
    }
    for field, expected in endpoint_fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"formal HellaSwag endpoint contract changed {field}")
    benchmark = _require_mapping(value.get("benchmarks"), "benchmark config").get(TASK)
    benchmark = _require_mapping(benchmark, "HellaSwag benchmark config")
    if int(benchmark.get("num_fewshot", -1)) != 0 or benchmark.get("metric") != "acc_norm":
        raise ValueError("formal HellaSwag must be zero-shot acc_norm")
    datasets = _require_mapping(
        _require_mapping(value.get("benchmark_data"), "benchmark identity").get("datasets"),
        "benchmark datasets",
    )
    files = _require_mapping(datasets.get(TASK), "HellaSwag dataset identity").get("files")
    if not isinstance(files, list):
        raise ValueError("formal HellaSwag dataset identity lacks files")
    validation = [
        row
        for row in files
        if isinstance(row, Mapping)
        and str(row.get("relative_path", "")).endswith("validation-00000-of-00001.parquet")
    ]
    if len(validation) != 1 or validation[0].get("rows") != EXPECTED_SAMPLES:
        raise ValueError("formal HellaSwag validation split is not exactly 10,042 rows")
    if validation[0].get("sha256") != EXPECTED_VALIDATION_SHA256:
        raise ValueError("formal HellaSwag validation payload changed")
    return fingerprint


def validate_hellaswag_result(
    path: str | Path, *, fingerprint: str, expected_mode: str
) -> dict[str, Any]:
    """Fail closed on every formal output invariant and recompute acc_norm."""

    if expected_mode not in ALLOWED_MODES:
        raise ValueError(f"unexpected formal HellaSwag mode: {expected_mode}")
    destination = Path(path)
    value = read_json(destination)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "mode": expected_mode,
        "task": TASK,
        "primary_metric": "acc_norm",
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(
                f"formal HellaSwag {expected_mode} checkpoint changed {field}"
            )

    result = _require_mapping(value.get("result"), "lm-eval payload")
    sample_counts = _require_mapping(result.get("n-samples"), "n-samples")
    task_counts = _require_mapping(sample_counts.get(TASK), "HellaSwag n-samples")
    if task_counts.get("effective") != EXPECTED_SAMPLES or task_counts.get("original") != EXPECTED_SAMPLES:
        raise ValueError("formal HellaSwag did not score the full 10,042-row split")
    if _require_mapping(result.get("n-shot"), "n-shot").get(TASK) != 0:
        raise ValueError("formal HellaSwag result is not zero-shot")
    invocation = _require_mapping(result.get("config"), "lm-eval invocation config")
    if invocation.get("limit") is not None or invocation.get("batch_size") != 1:
        raise ValueError("formal HellaSwag result used a limit or non-unit batch")
    if invocation.get("apply_chat_template") is not False:
        raise ValueError("formal HellaSwag result used a chat template")

    resolved = _require_mapping(result.get("configs"), "resolved task configs")
    task_config = _require_mapping(resolved.get(TASK), "resolved HellaSwag config")
    if (
        task_config.get("validation_split") != "validation"
        or task_config.get("num_fewshot") != 0
        or task_config.get("output_type") != "multiple_choice"
    ):
        raise ValueError("resolved HellaSwag task protocol changed")
    metric_list = task_config.get("metric_list")
    if not isinstance(metric_list, list) or not any(
        isinstance(row, Mapping)
        and row.get("metric") == "acc_norm"
        and row.get("aggregation") == "mean"
        and row.get("higher_is_better") is True
        for row in metric_list
    ):
        raise ValueError("resolved HellaSwag acc_norm metric changed")

    aggregates = _require_mapping(result.get("results"), "aggregate results")
    aggregate = _require_mapping(aggregates.get(TASK), "HellaSwag aggregate")
    if aggregate.get("sample_len") != EXPECTED_SAMPLES:
        raise ValueError("formal HellaSwag aggregate sample_len is not 10,042")
    aggregate_accuracy = _require_finite(aggregate.get("acc_norm,none"), "acc_norm")
    primary = _require_finite(value.get("primary_metric_value"), "primary metric")

    sample_groups = _require_mapping(result.get("samples"), "logged samples")
    samples = sample_groups.get(TASK)
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        raise ValueError("formal HellaSwag must persist all 10,042 sample rows")
    doc_ids: list[int] = []
    correct = 0.0
    for index, sample_raw in enumerate(samples):
        sample = _require_mapping(sample_raw, f"sample {index}")
        doc_id = sample.get("doc_id")
        if isinstance(doc_id, bool) or not isinstance(doc_id, int):
            raise ValueError(f"formal HellaSwag sample {index} has invalid doc_id")
        doc_ids.append(doc_id)
        score = _require_finite(sample.get("acc_norm"), f"sample {index} acc_norm")
        if score not in (0.0, 1.0):
            raise ValueError(f"formal HellaSwag sample {index} acc_norm is not binary")
        if sample.get("filter") != "none" or "acc_norm" not in sample.get("metrics", ()):
            raise ValueError(f"formal HellaSwag sample {index} metric/filter changed")
        correct += score
    if doc_ids != list(range(EXPECTED_SAMPLES)):
        raise ValueError("formal HellaSwag sample IDs are incomplete, duplicated, or reordered")
    recomputed = correct / EXPECTED_SAMPLES
    for label, observed in (("aggregate", aggregate_accuracy), ("primary", primary)):
        if not math.isclose(observed, recomputed, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"formal HellaSwag {label} acc_norm disagrees with raw samples")
    return {
        "mode": expected_mode,
        "samples": EXPECTED_SAMPLES,
        "correct": int(correct),
        "acc_norm": recomputed,
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "nbytes": destination.stat().st_size,
    }


def _validate_existing_output(output: Path) -> None:
    """Reject invalid completed checkpoints before the shared resume check sees them."""

    result_paths = {
        mode: output / "results" / mode / f"{TASK}.json" for mode in ALLOWED_MODES
    }
    if not any(path.exists() for path in result_paths.values()):
        return
    definition_path = output / "run_definition.json"
    if not definition_path.is_file():
        raise ValueError("formal HellaSwag checkpoint exists without run_definition.json")
    fingerprint = validate_run_definition(read_json(definition_path))
    for mode, path in result_paths.items():
        if path.exists():
            validate_hellaswag_result(path, fingerprint=fingerprint, expected_mode=mode)
    receipt_path = output / FORMAL_RECEIPT
    if receipt_path.exists():
        verify_hashed_json(receipt_path)
        receipt = read_json(receipt_path)
        if (
            receipt.get("status") != "complete"
            or receipt.get("kind") != "qwen3_32b_to_14b_full_hellaswag"
            or receipt.get("request_fingerprint") != fingerprint
            or receipt.get("runtime_topology") != FORMAL_RUNTIME_TOPOLOGY
            or receipt.get("runtime_topology_sha256")
            != canonical_sha256(FORMAL_RUNTIME_TOPOLOGY)
        ):
            raise ValueError("formal HellaSwag receipt fingerprint changed")
        recorded = receipt.get("results")
        if not isinstance(recorded, list) or len(recorded) != len(ALLOWED_MODES):
            raise ValueError("formal HellaSwag receipt result inventory changed")
        by_mode = {
            str(row.get("mode")): row for row in recorded if isinstance(row, Mapping)
        }
        if set(by_mode) != set(ALLOWED_MODES):
            raise ValueError("formal HellaSwag receipt mode inventory changed")
        for mode, path in result_paths.items():
            if not path.is_file():
                raise ValueError("complete formal HellaSwag receipt has a missing result")
            row = by_mode[mode]
            if row.get("sha256") != sha256_file(path) or row.get("nbytes") != path.stat().st_size:
                raise ValueError(f"formal HellaSwag receipt/result mismatch for {mode}")
        idle_identity = receipt.get("gpu_idle_gate")
        gate = _validate_gpu_idle_gate(output)
        if idle_identity != gate:
            raise ValueError("formal HellaSwag receipt/GPU-idle-gate binding changed")


def _validate_gpu_idle_gate(output: Path) -> dict[str, Any]:
    path = output / GPU_IDLE_GATE
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
            raise ValueError(f"formal HellaSwag GPU idle gate changed {field}")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) < 12:
        raise ValueError("formal HellaSwag GPU idle gate lacks twelve samples")
    for sample in samples[-12:]:
        if not isinstance(sample, Mapping) or sample.get("passed") is not True:
            raise ValueError("formal HellaSwag GPU idle gate has a failed final sample")
        if sample.get("compute_processes") != []:
            raise ValueError("formal HellaSwag GPU idle gate observed a compute process")
        memory = sample.get("memory")
        if not isinstance(memory, list) or len(memory) != 8:
            raise ValueError("formal HellaSwag GPU idle gate has incomplete memory evidence")
        if any(not isinstance(row, Mapping) for row in memory):
            raise ValueError("formal HellaSwag GPU idle gate has malformed memory evidence")
        if {int(row.get("physical_index", -1)) for row in memory} != set(range(8)):
            raise ValueError("formal HellaSwag GPU idle gate has the wrong GPU inventory")
        if any(int(row.get("memory_used_mib", 65)) > 64 for row in memory):
            raise ValueError("formal HellaSwag GPU idle gate exceeded the memory limit")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "sha256_sidecar": {
            "path": str(path.with_name(f"{path.name}.sha256.json").resolve()),
            "sha256": sha256_file(path.with_name(f"{path.name}.sha256.json")),
            "nbytes": path.with_name(f"{path.name}.sha256.json").stat().st_size,
        },
    }


def finalize_formal_output(output: Path) -> dict[str, Any]:
    definition_path = output / "run_definition.json"
    fingerprint = validate_run_definition(read_json(definition_path))
    status_path = output / "status.json"
    status = read_json(status_path)
    if status.get("status") != "complete" or status.get("failures") != 0:
        raise ValueError("formal HellaSwag shared evaluator did not complete cleanly")
    rows = [
        validate_hellaswag_result(
            output / "results" / mode / f"{TASK}.json",
            fingerprint=fingerprint,
            expected_mode=mode,
        )
        for mode in ALLOWED_MODES
    ]
    idle_gate = _validate_gpu_idle_gate(output)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_full_hellaswag",
        "request_fingerprint": fingerprint,
        "modes": list(ALLOWED_MODES),
        "task": TASK,
        "split_scope": SPLIT_SCOPE,
        "num_fewshot": 0,
        "metric": "acc_norm",
        "sample_count_per_mode": EXPECTED_SAMPLES,
        "validation_payload_sha256": EXPECTED_VALIDATION_SHA256,
        "runtime_topology": FORMAL_RUNTIME_TOPOLOGY,
        "runtime_topology_sha256": canonical_sha256(FORMAL_RUNTIME_TOPOLOGY),
        "gpu_lock": {
            "path": str(GLOBAL_GPU_LOCK),
            "scope": "repository_wide_forward_and_reverse_eight_gpu_experiments",
            "held_for_shared_evaluation_and_receipt": True,
            "idle_gate_rerun_after_lock_acquisition": True,
        },
        "gpu_idle_gate": idle_gate,
        "results": rows,
        "run_definition": {
            "path": str(definition_path.resolve()),
            "sha256": sha256_file(definition_path),
            "nbytes": definition_path.stat().st_size,
        },
        "shared_status": {
            "path": str(status_path.resolve()),
            "sha256": sha256_file(status_path),
            "nbytes": status_path.stat().st_size,
        },
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    hashed_json(output / FORMAL_RECEIPT, receipt)
    return receipt


def _default_output() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return EXPERIMENT_ROOT / "runs" / f"{stamp}_hellaswag_full_k20"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run and args.output_dir is None:
        args.output_dir = _default_output()
    if not args.dry_run:
        assert args.output_dir is not None
        _validate_existing_output(args.output_dir.resolve())
    # Wait for any repository-managed predecessor, then repeat the physical
    # idle proof while holding the lock before loading either model.
    lock = None if args.dry_run else acquire_gpu_lock(blocking=True)
    try:
        if not args.dry_run:
            assert args.output_dir is not None
            idle_gate = wait_for_idle_gpus({"runtime": GPU_IDLE_RUNTIME})
            hashed_json(args.output_dir.resolve() / GPU_IDLE_GATE, idle_gate)
        result = run_evaluation(sealed_shared_argv(args))
        if result == 0 and not args.dry_run:
            assert args.output_dir is not None
            receipt = finalize_formal_output(args.output_dir.resolve())
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return result
    finally:
        if lock is not None:
            lock.close()


__all__ = [
    "ALLOWED_MODES",
    "EXPECTED_SAMPLES",
    "FORMAL_RECEIPT",
    "GPU_IDLE_GATE",
    "GPU_IDLE_RUNTIME",
    "FORMAL_RUNTIME_TOPOLOGY",
    "SPLIT_SCOPE",
    "TASK",
    "finalize_formal_output",
    "main",
    "parse_args",
    "sealed_shared_argv",
    "validate_hellaswag_result",
    "validate_run_definition",
]
