"""Global eight-GPU lock and CUDA-free idle gate."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
import subprocess
import time
from typing import Any, BinaryIO, Callable, Iterator, Mapping

from .contracts import ContractError


@contextmanager
def global_gpu_lock(path: str | Path) -> Iterator[BinaryIO]:
    lock_path = Path(path)
    if lock_path.is_symlink():
        raise ContractError("global GPU lock cannot be a symlink")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def nvidia_smi_probe() -> dict[str, Any]:
    memory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: dict[str, int] = {}
    for line in memory.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            raise ContractError(f"unexpected nvidia-smi memory row: {line}")
        rows[parts[0]] = int(parts[1])
    if set(rows) != {str(index) for index in range(8)}:
        raise ContractError(f"idle gate requires physical GPU indices 0..7, got {sorted(rows)}")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    process_rows = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
    return {
        "memory_used_mib": rows,
        "compute_process_rows": process_rows,
        "compute_process_count": len(process_rows),
    }


def wait_for_idle_gpus(
    contract: Mapping[str, Any],
    *,
    probe: Callable[[], Mapping[str, Any]] = nvidia_smi_probe,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    limit = int(contract["memory_used_limit_mib"])
    required = int(contract["consecutive_samples"])
    interval = float(contract["sample_interval_seconds"])
    timeout = float(contract["timeout_seconds"])
    if limit < 0 or required <= 0 or interval <= 0 or timeout <= 0:
        raise ContractError("GPU idle contract is malformed")
    began = monotonic()
    consecutive = 0
    samples: list[dict[str, Any]] = []
    while True:
        observed = dict(probe())
        memory = observed.get("memory_used_mib")
        process_count = observed.get("compute_process_count")
        if not isinstance(memory, Mapping) or set(memory) != {str(index) for index in range(8)}:
            raise ContractError("GPU idle probe returned wrong device set")
        if any(type(value) is not int or value < 0 for value in memory.values()):
            raise ContractError("GPU idle probe returned malformed memory")
        if type(process_count) is not int or process_count < 0:
            raise ContractError("GPU idle probe returned malformed process count")
        idle = process_count == 0 and all(value <= limit for value in memory.values())
        consecutive = consecutive + 1 if idle else 0
        sample = {
            "sample_index": len(samples),
            "elapsed_seconds": monotonic() - began,
            "memory_used_mib": dict(memory),
            "compute_process_count": process_count,
            "idle": idle,
            "consecutive_idle": consecutive,
        }
        samples.append(sample)
        if consecutive >= required:
            return {
                "schema_version": 2,
                "status": "passed",
                "probe": "nvidia-smi_without_CUDA_runtime",
                "gpu_used_by_gate": False,
                "physical_gpu_indices": list(range(8)),
                "contract": dict(contract),
                "samples": samples,
                "final_consecutive_idle_samples": consecutive,
                "elapsed_seconds": monotonic() - began,
            }
        if monotonic() - began >= timeout:
            raise ContractError("timed out waiting for eight idle GPUs")
        sleeper(interval)


__all__ = ["global_gpu_lock", "nvidia_smi_probe", "wait_for_idle_gpus"]
