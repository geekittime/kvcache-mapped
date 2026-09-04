#!/usr/bin/env python3
"""Run the isolated reverse five-task smoke or full extension.

The shared evaluator owns request execution and request-level durable caches.
This wrapper seals the scientific CLI, validates every reusable task
checkpoint before resume, and publishes an independently recomputed receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

EXTENSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXTENSION_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from five_task_extension.protocol import (  # noqa: E402
    DEFAULT_FULL_OUTPUT,
    DEFAULT_SMOKE_OUTPUT,
    EXECUTED_TASKS,
    GPU_GATES_DIRECTORY,
    MODES,
    RECEIPT_NAME,
    SMOKE_LIMIT,
    SMOKE_WIKITEXT_CHUNKS,
    output_within_experiment,
)
from five_task_extension.validator import (  # noqa: E402
    _validate_accuracy_checkpoint,
    _validate_cross_mode_identity,
    _validate_wikitext_checkpoint,
    finalize_full_output,
    finalize_smoke_output,
    validate_completed_output,
    validate_extension_config,
    validate_extension_code_freeze,
    validate_gpu_idle_gate,
    validate_hellaswag_bridge,
    validate_run_definition,
)
from five_task_extension.adapter import run_extension_evaluation  # noqa: E402
from reverse_kv.eval_common import hashed_json  # noqa: E402
from reverse_kv.protocol import read_json  # noqa: E402
from reverse_kv.reverse_latency import (  # noqa: E402
    acquire_gpu_lock,
    wait_for_idle_gpus,
)


GPU_IDLE_RUNTIME = {
    "require_cuda_visible_devices": "0,1,2,3,4,5,6,7",
    "idle_memory_limit_mib": 64,
    "idle_consecutive_samples": 12,
    "idle_sample_interval_seconds": 5,
    "idle_timeout_seconds": 86_400,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _default_output(profile: str) -> Path:
    return DEFAULT_FULL_OUTPUT if profile == "full" else DEFAULT_SMOKE_OUTPUT


def _sealed_argv(profile: str, output: Path) -> list[str]:
    result = [
        "--modes",
        ",".join(MODES),
        "--tasks",
        ",".join(EXECUTED_TASKS),
        "--log-samples",
        "--output-dir",
        str(output),
    ]
    if profile == "smoke":
        result.extend(("--limit", str(SMOKE_LIMIT), "--wikitext-max-chunks", str(SMOKE_WIKITEXT_CHUNKS)))
    return result


def _validate_existing_partial(output: Path, profile: str) -> None:
    definition_path = output / "run_definition.json"
    if not definition_path.exists():
        # No result is allowed without its immutable request definition.
        if (output / "results").exists():
            raise ValueError("extension results exist without run_definition.json")
        return
    definition, fingerprint = validate_run_definition(output, profile=profile)
    for task in EXECUTED_TASKS:
        present_modes: list[str] = []
        for mode in MODES:
            path = output / "results" / mode / f"{task}.json"
            if not path.exists():
                continue
            present_modes.append(mode)
            if task == "wikitext2_prefix_ppl":
                _validate_wikitext_checkpoint(path, fingerprint=fingerprint, mode=mode, profile=profile)
            else:
                _validate_accuracy_checkpoint(
                    path,
                    definition=definition,
                    fingerprint=fingerprint,
                    mode=mode,
                    task=task,
                    profile=profile,
                )
        if len(present_modes) == len(MODES) and task != "wikitext2_prefix_ppl":
            _validate_cross_mode_identity(output, task)


def _next_segment_gate(output: Path) -> Path:
    directory = output / GPU_GATES_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(directory.glob("segment_*_gpu_idle_gate.json"))
    expected = [f"segment_{index:04d}_gpu_idle_gate.json" for index in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected:
        raise ValueError("existing extension segment gates are incomplete/reordered")
    for path in paths:
        validate_gpu_idle_gate(path)
    return directory / f"segment_{len(paths) + 1:04d}_gpu_idle_gate.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = output_within_experiment(args.output_dir or _default_output(args.profile))
    validate_extension_config()
    validate_extension_code_freeze(deep_models=True, deep_mapper=True)
    output.mkdir(parents=True, exist_ok=True)
    if args.profile == "full" and (output / RECEIPT_NAME).exists():
        print(json.dumps(validate_completed_output(output), ensure_ascii=False, indent=2))
        return 0
    validate_hellaswag_bridge()
    _validate_existing_partial(output, args.profile)

    # A shared-evaluator run that completed but was interrupted before the
    # extension receipt needs only CPU finalization, not another model load.
    status_path = output / "status.json"
    if status_path.is_file():
        status = read_json(status_path)
        if status.get("status") == "complete" and status.get("failures") == 0:
            receipt = (
                finalize_full_output(output)
                if args.profile == "full"
                else finalize_smoke_output(output)
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2))
            return 0

    lock = acquire_gpu_lock(blocking=True)
    try:
        idle = wait_for_idle_gpus({"runtime": GPU_IDLE_RUNTIME})
        hashed_json(_next_segment_gate(output), idle)
        result = run_extension_evaluation(_sealed_argv(args.profile, output))
        if result != 0:
            return int(result)
        receipt = (
            finalize_full_output(output)
            if args.profile == "full"
            else finalize_smoke_output(output)
        )
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
