#!/usr/bin/env python3
"""Prepare, smoke, or run the sealed reverse long-context extension."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Sequence


EXTENSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXTENSION_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from long_context_extension.adapter import run_extension_evaluation  # noqa: E402
from long_context_extension.protocol import (  # noqa: E402
    DEFAULT_FULL_OUTPUT,
    DEFAULT_SMOKE_OUTPUT,
    EXPECTED_RESULT_COUNT,
    EXPECTED_SMOKE_RESULT_COUNT,
    GPU_IDLE_RUNTIME,
    SMOKE_ENDPOINTS,
    SMOKE_SAMPLE_INDICES,
    TASKS,
    output_within_experiment,
)
from long_context_extension.freeze import validate_code_freeze  # noqa: E402
from long_context_extension.validator import (  # noqa: E402
    FULL_RECEIPT_NAME,
    SMOKE_RECEIPT_NAME,
    finalize_full_output,
    finalize_smoke_output,
    validate_completed_output,
    validate_gpu_idle_gate,
)
from reverse_kv.eval_common import immutable_hashed_json  # noqa: E402
from reverse_kv.protocol import read_json  # noqa: E402
from reverse_kv.reverse_latency import (  # noqa: E402
    acquire_gpu_lock,
    wait_for_idle_gpus,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=("prepare", "smoke", "full"), required=True
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args(argv)


def _default_output(profile: str) -> Path:
    return DEFAULT_SMOKE_OUTPUT if profile == "smoke" else DEFAULT_FULL_OUTPUT


def _sealed_argv(profile: str, output: Path) -> list[str]:
    values = [
        "--tasks",
        ",".join(TASKS),
        "--output-dir",
        str(output),
    ]
    if profile == "prepare":
        values.append("--prepare-only")
    elif profile == "smoke":
        explicit = ",".join(
            f"{task}:{index}"
            for task in TASKS
            for index in SMOKE_SAMPLE_INDICES[task]
        )
        values.extend(("--sample-indices", explicit))
    return values


def _record_scope(output: Path, profile: str) -> None:
    # Prepare and full intentionally share one immutable request fingerprint;
    # prepare performs the CPU-only checks for the later full run.
    scope = "frozen_explicit_two_indices_per_task" if profile == "smoke" else "formal_full"
    immutable_hashed_json(
        output / "extension_scope.json",
        {
            "schema_version": 1,
            "scope": scope,
            "tasks": list(TASKS),
            "smoke_sample_indices": (
                {
                    task: list(indices)
                    for task, indices in SMOKE_SAMPLE_INDICES.items()
                }
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
        },
    )


def _next_gpu_gate(output: Path) -> Path:
    directory = output / "long_context_gpu_gates"
    if directory.is_symlink():
        raise ValueError("long-context GPU gate directory is symlinked")
    directory.mkdir(parents=True, exist_ok=True)
    gate_pattern = re.compile(r"gate_[0-9]{4}\.json")
    existing = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and gate_pattern.fullmatch(path.name)
    )
    expected = [f"gate_{index:04d}.json" for index in range(1, len(existing) + 1)]
    if [path.name for path in existing] != expected:
        raise ValueError("long-context GPU gate sequence is incomplete or reordered")
    expected_files = set(expected) | {
        f"{name}.sha256.json" for name in expected
    }
    if {path.name for path in directory.iterdir()} != expected_files:
        raise ValueError("long-context GPU gate directory contains unexpected files")
    for path in existing:
        validate_gpu_idle_gate(path)
    return directory / f"gate_{len(existing) + 1:04d}.json"


def _receipt_artifacts(output: Path, profile: str) -> tuple[Path, Path]:
    name = FULL_RECEIPT_NAME if profile == "full" else SMOKE_RECEIPT_NAME
    receipt = output / name
    return receipt, receipt.with_name(f"{receipt.name}.sha256.json")


def _shared_status_is_complete(output: Path) -> bool:
    path = output / "status.json"
    if not path.exists():
        return False
    if path.is_symlink() or not path.is_file():
        raise ValueError("shared long-context status is not a regular file")
    value = read_json(path)
    if value.get("status") != "complete":
        return False
    if value.get("failures") != 0:
        raise ValueError("shared complete status has a nonzero failure count")
    return True


def _require_prepare_safe_status(output: Path) -> None:
    path = output / "status.json"
    if not path.exists():
        return
    if path.is_symlink() or not path.is_file():
        raise ValueError("prepare status is not a regular file")
    status = read_json(path).get("status")
    if status != "prepared":
        raise ValueError(
            f"prepare refuses to overwrite existing shared status {status!r}"
        )


def _reject_foreign_receipt(output: Path, profile: str) -> None:
    foreign = "smoke" if profile == "full" else "full"
    if any(path.exists() or path.is_symlink() for path in _receipt_artifacts(output, foreign)):
        raise ValueError(f"output contains a conflicting {foreign} completion receipt")


def _validate_committed_receipt(output: Path, profile: str) -> bool:
    receipt, sidecar = _receipt_artifacts(output, profile)
    if receipt.is_file() and sidecar.is_file():
        validate_completed_output(output, profile=profile)
        return True
    return False


def _recover_completed_under_lock(output: Path, profile: str) -> bool:
    """Finalize or validate completed raw artifacts without loading a model."""

    receipt, sidecar = _receipt_artifacts(output, profile)
    if any(path.exists() or path.is_symlink() for path in (receipt, sidecar)):
        # A crash between the receipt and sidecar writes is deliberately
        # fail-closed: the strict validator reports the incomplete commit.
        validate_completed_output(output, profile=profile)
        return True
    if not _shared_status_is_complete(output):
        return False
    if profile == "full":
        finalize_full_output(output)
    else:
        finalize_smoke_output(output)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_code_freeze()
    output = output_within_experiment(args.output_dir or _default_output(args.profile))
    output.mkdir(parents=True, exist_ok=True)
    sealed = _sealed_argv(args.profile, output)
    receipt_profile = "full" if args.profile == "prepare" else args.profile
    _reject_foreign_receipt(output, receipt_profile)
    if _validate_committed_receipt(output, receipt_profile):
        return 0

    if args.profile == "prepare":
        # Serialize the entire writer section with full/smoke GPU jobs.  This
        # CPU-only profile does not consume CUDA, but without the shared lock it
        # could overwrite status/control files while a full process is active.
        lock = acquire_gpu_lock(blocking=True)
        try:
            _reject_foreign_receipt(output, "full")
            if _recover_completed_under_lock(output, "full"):
                return 0
            _require_prepare_safe_status(output)
            _record_scope(output, args.profile)
            # The shared --prepare-only path reads and hashes data, code,
            # models, and mapper metadata but returns before CUDA checks.
            return int(run_extension_evaluation(sealed))
        finally:
            lock.close()

    lock = acquire_gpu_lock(blocking=True)
    try:
        # Two callers can both observe an incomplete directory before either
        # owns the global lock.  Recheck completion after acquiring it so the
        # second waiter never performs an idle gate or loads both models.
        _reject_foreign_receipt(output, args.profile)
        if _recover_completed_under_lock(output, args.profile):
            return 0
        _record_scope(output, args.profile)
        idle_evidence = wait_for_idle_gpus({"runtime": GPU_IDLE_RUNTIME})
        immutable_hashed_json(_next_gpu_gate(output), idle_evidence)
        result = int(run_extension_evaluation(sealed))
        if result != 0:
            return result
        if args.profile == "full":
            finalize_full_output(output)
        else:
            finalize_smoke_output(output)
        return 0
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
