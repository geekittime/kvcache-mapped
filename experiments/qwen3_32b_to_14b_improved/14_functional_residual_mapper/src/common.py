from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class ContractError(ValueError):
    """Raised when a fail-closed machine contract is violated."""


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f"duplicate JSON key: {key!r}")
        out[key] = value
    return out


def load_json_strict(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ContractError(f"non-finite JSON constant: {value}")
            ),
        )
    except ContractError:
        raise
    except Exception as exc:  # pragma: no cover - exact parser exception is irrelevant
        raise ContractError(f"cannot parse strict JSON {path}: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not is_int(value):
        raise ContractError(f"{name} must be an integer, not {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} must be >= {minimum}")
    return value


def require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be finite")
    return result


def require_exact_keys(value: Mapping[str, Any], expected: Iterable[str], name: str) -> None:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        missing = sorted(wanted - actual)
        extra = sorted(actual - wanted)
        raise ContractError(f"{name} key mismatch; missing={missing}, extra={extra}")


def ensure_no_symlink_components(path: Path, *, require_leaf: bool = True) -> Path:
    """Reject live or dangling symlinks before resolving an absolute path."""

    if not path.is_absolute():
        raise ContractError(f"path must be absolute: {path}")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if is_leaf and not require_leaf:
                return current
            raise ContractError(f"missing path component: {current}") from None
        if stat.S_ISLNK(mode):
            raise ContractError(f"symlink component forbidden: {current}")
    return path.resolve(strict=require_leaf)


def validate_artifact_ref(value: Mapping[str, Any], name: str) -> Path:
    require_exact_keys(value, ("path", "nbytes", "sha256"), name)
    raw_path = value["path"]
    if not isinstance(raw_path, str) or not raw_path.startswith("/"):
        raise ContractError(f"{name}.path must be an absolute string")
    nbytes = require_int(value["nbytes"], f"{name}.nbytes", minimum=0)
    digest = value["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ContractError(f"{name}.sha256 must be 64 lowercase hex characters")
    path = ensure_no_symlink_components(Path(raw_path))
    if not path.is_file():
        raise ContractError(f"{name}.path is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != nbytes:
        raise ContractError(f"{name} reference mismatch (nbytes): expected {nbytes}, got {actual_size}")
    actual_digest = sha256_file(path)
    if actual_digest != digest:
        raise ContractError(f"{name} reference mismatch (sha256): expected {digest}, got {actual_digest}")
    return path


def artifact_ref(path: Path) -> dict[str, Any]:
    resolved = ensure_no_symlink_components(path)
    if not resolved.is_file():
        raise ContractError(f"artifact must be a regular file: {resolved}")
    return {"path": str(resolved), "nbytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def ensure_output_under(path: Path, root: Path) -> Path:
    if not path.is_absolute() or not root.is_absolute():
        raise ContractError("output path and root must be absolute")
    resolved_root = ensure_no_symlink_components(root)
    if not resolved_root.is_dir():
        raise ContractError(f"output root is not a directory: {resolved_root}")
    parent = path.parent
    resolved_parent = ensure_no_symlink_components(parent)
    resolved = resolved_parent / path.name
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        raise ContractError(f"output escapes experiment root: {resolved}") from None
    return resolved


def placeholders(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and value.startswith("__") and value.endswith("__"):
        found.append(prefix)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(placeholders(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(placeholders(child, f"{prefix}[{index}]"))
    return found


def atomic_write_json(path: Path, value: Any, *, mode: int = 0o444) -> None:
    """Create a canonical JSON file atomically and refuse overwrite."""

    if path.exists() or path.is_symlink():
        raise ContractError(f"refusing to overwrite existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        finally:
            raise
