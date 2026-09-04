"""Per-run serialization and monotonic live/terminal status publication."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping

from .contracts import (
    ContractError,
    atomic_json,
    artifact_ref,
    read_json,
    verify_sidecar,
    write_sidecar,
)


@contextmanager
def run_state_lock(output: str | Path) -> Iterator[BinaryIO]:
    """Serialize GPU execution and CPU sealing for one canonical run."""

    output_path = Path(output).resolve()
    lock_root = output_path.parent / ".run_state_locks"
    if lock_root.is_symlink():
        raise ContractError("run-state lock root cannot be a symlink")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{output_path.name}.lock"
    if lock_path.is_symlink():
        raise ContractError("run-state lock cannot be a symlink")
    handle = lock_path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def write_live_status(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write a mutable nonterminal state without ever overwriting terminal."""

    destination = Path(path)
    if destination.is_symlink():
        raise ContractError("run status cannot be a symlink")
    if value.get("status") == "complete":
        raise ContractError("live status writer cannot publish terminal complete")
    sidecar = destination.with_name(destination.name + ".sha256.json")
    if sidecar.exists() or sidecar.is_symlink():
        verify_sidecar(destination)
        frozen = read_json(destination)
        if isinstance(frozen, Mapping) and frozen.get("status") == "complete":
            raise ContractError("terminal run status cannot be downgraded or overwritten")
        raise ContractError("content-addressed nonterminal status is invalid")
    if destination.exists() or destination.is_symlink():
        frozen = read_json(destination)
        if isinstance(frozen, Mapping) and frozen.get("status") == "complete":
            raise ContractError("unsealed terminal run status cannot be overwritten")
    atomic_json(destination, dict(value))


def write_terminal_status(path: str | Path, value: Mapping[str, Any]) -> None:
    """Publish/recover the sole content-addressed terminal status transition."""

    destination = Path(path)
    if destination.is_symlink():
        raise ContractError("run status cannot be a symlink")
    if value.get("status") != "complete":
        raise ContractError("terminal status writer requires status=complete")
    sidecar = destination.with_name(destination.name + ".sha256.json")
    if sidecar.exists() or sidecar.is_symlink():
        verify_sidecar(destination)
        if read_json(destination) != dict(value):
            raise ContractError("terminal run status already exists with different bytes")
        return
    # A previous process may have died after atomically renaming the terminal
    # JSON but before its sidecar.  Rewriting the same canonical bytes and then
    # publishing the sidecar is safe and recoverable.
    if destination.exists() or destination.is_symlink():
        existing = read_json(destination)
        if (
            isinstance(existing, Mapping)
            and existing.get("status") == "complete"
            and existing != dict(value)
        ):
            raise ContractError("unsealed terminal run status differs from requested transition")
    atomic_json(destination, dict(value))
    write_sidecar(artifact_ref(destination))


__all__ = ["run_state_lock", "write_live_status", "write_terminal_status"]
