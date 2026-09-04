"""Small, strict, dependency-light publication helpers.

The quota repair experiment is deliberately a *consumer* of the immutable
pairwise-v3 artifacts.  This module therefore has no code which can write to a
source artifact.  All writes are confined to the new experiment root and are
single-writer, atomic, and refuse overwrite.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class ProtocolError(RuntimeError):
    """Raised for any contract violation (the protocol fails closed)."""


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used in all digests and JSON payloads."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"value is not finite/canonical JSON: {exc}") from exc
    return (text + "\n").encode("utf-8")


def _reject_constant(value: str) -> Any:
    raise ProtocolError(f"nonfinite JSON number is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str | bytes) -> Any:
    try:
        return json.loads(
            payload,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except ProtocolError:
        raise
    except Exception as exc:
        raise ProtocolError(f"invalid JSON: {exc}") from exc


def strict_json_load(path: Path) -> Any:
    require_regular_file(path)
    return strict_json_loads(path.read_bytes())


def json_exact(left: Any, right: Any) -> bool:
    """Semantic equality which does not alias bool and int values."""

    try:
        return canonical_bytes(left) == canonical_bytes(right)
    except ProtocolError:
        return False


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_no_symlink(path: Path, *, allow_missing_leaf: bool = False) -> None:
    """Reject symlink components, including dangling links."""

    if not path.is_absolute():
        raise ProtocolError(f"path must be absolute: {path}")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if is_leaf and allow_missing_leaf:
                return
            raise ProtocolError(f"missing path component: {current}") from None
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"symlink path component forbidden: {current}")


def require_regular_file(path: Path) -> None:
    _walk_no_symlink(path)
    mode = os.lstat(path).st_mode
    if not stat.S_ISREG(mode):
        raise ProtocolError(f"not a regular file: {path}")


def require_directory(path: Path) -> None:
    _walk_no_symlink(path)
    mode = os.lstat(path).st_mode
    if not stat.S_ISDIR(mode):
        raise ProtocolError(f"not a directory: {path}")


def file_spec(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    require_regular_file(path)
    value = {"nbytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"path": str(path), **value} if include_path else value


def verify_file_spec(path: Path, spec: Mapping[str, Any], *, name: str = "file") -> None:
    if not isinstance(spec, Mapping) or set(spec) != {"path", "nbytes", "sha256"}:
        raise ProtocolError(f"{name} file specification keys mismatch")
    if spec.get("path") != str(path):
        raise ProtocolError(f"{name} path binding mismatch")
    if type(spec.get("nbytes")) is not int or spec["nbytes"] < 0:
        raise ProtocolError(f"{name} nbytes must be a nonnegative integer")
    digest = spec.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ProtocolError(f"{name} sha256 must be lowercase hexadecimal")
    observed = file_spec(path)
    if not json_exact(observed, dict(spec)):
        raise ProtocolError(f"{name} file binding mismatch")


def safe_child(root: Path, relative: str, *, allow_missing: bool = False) -> Path:
    """Resolve a relative path under root without following symlinks."""

    require_directory(root)
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ProtocolError(f"path escapes experiment root: {relative}")
    candidate = root / rel
    _walk_no_symlink(candidate, allow_missing_leaf=allow_missing)
    try:
        candidate.resolve(strict=not allow_missing).relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"path escapes experiment root: {relative}") from exc
    return candidate


def atomic_write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    """Atomically create a file and refuse overwrite."""

    if path.exists() or path.is_symlink():
        raise ProtocolError(f"refusing to overwrite existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def claim_artifact(output_dir: Path, identity: Mapping[str, Any]) -> None:
    """Create an output directory once, with an in-progress marker."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.single_writer.lock"
    lock = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError(f"another writer owns {output_dir}") from exc
        if output_dir.exists() or output_dir.is_symlink():
            raise ProtocolError(f"refusing to reuse artifact directory: {output_dir}")
        output_dir.mkdir(parents=False, exist_ok=False)
        atomic_json(
            output_dir / "IN_PROGRESS.json",
            {
                "schema_version": 1,
                **dict(identity),
                "status": "in_progress_not_published",
                "completion_marker_required": "PUBLISHED.json",
            },
        )
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()


def finalize_publication(
    output_dir: Path,
    payload_names: list[str],
    *,
    identity: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    """Write sidecars and atomically-last PUBLISHED marker."""

    require_directory(output_dir)
    if len(payload_names) != len(set(payload_names)) or any(Path(n).name != n for n in payload_names):
        raise ProtocolError("publication payload names must be unique basenames")
    for name in payload_names:
        require_regular_file(output_dir / name)
        atomic_json(output_dir / f"{name}.sha256.json", file_spec(output_dir / name))
    names = sorted([*payload_names, *(f"{name}.sha256.json" for name in payload_names)])
    expected = sorted([*names, "IN_PROGRESS.json"])
    if sorted(p.name for p in output_dir.iterdir()) != expected:
        raise ProtocolError("publication inventory mismatch before commit")
    for name in expected:
        require_regular_file(output_dir / name)
    (output_dir / "IN_PROGRESS.json").unlink()
    if sorted(p.name for p in output_dir.iterdir()) != names:
        raise ProtocolError("publication inventory changed during commit")
    published_files = {name: file_spec(output_dir / name, include_path=False) for name in names}
    ordered = [{"name": name, **published_files[name]} for name in names]
    marker = {
        "schema_version": 1,
        **dict(identity),
        "status": status,
        "atomic_last_completion_marker": True,
        "published_files": published_files,
        "ordered_inventory_sha256": sha256_bytes(canonical_bytes(ordered)),
        "receipt": published_files.get("receipt.json"),
        "receipt_sidecar": published_files.get("receipt.json.sha256.json"),
    }
    # Every stage in this experiment includes a receipt payload.  Refuse an
    # accidental marker with absent receipt bindings rather than weakening the
    # publication contract.
    if marker["receipt"] is None or marker["receipt_sidecar"] is None:
        raise ProtocolError("publication requires receipt.json payload")
    atomic_json(output_dir / "PUBLISHED.json", marker)
    return marker


def validate_publication(
    output_dir: Path,
    payload_names: list[str],
    *,
    identity: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    require_directory(output_dir)
    marker_path = output_dir / "PUBLISHED.json"
    require_regular_file(marker_path)
    names = sorted([*payload_names, *(f"{name}.sha256.json" for name in payload_names)])
    if sorted(p.name for p in output_dir.iterdir()) != sorted([*names, "PUBLISHED.json"]):
        raise ProtocolError("published artifact inventory mismatch")
    marker = strict_json_load(marker_path)
    expected_keys = {
        "schema_version", *identity, "status", "atomic_last_completion_marker",
        "published_files", "ordered_inventory_sha256", "receipt", "receipt_sidecar",
    }
    if (
        not isinstance(marker, Mapping)
        or set(marker) != expected_keys
        or type(marker.get("schema_version")) is not int
        or marker.get("schema_version") != 1
        or any(marker.get(k) != v for k, v in identity.items())
        or marker.get("status") != status
        or marker.get("atomic_last_completion_marker") is not True
    ):
        raise ProtocolError("PUBLISHED marker semantics mismatch")
    observed = {name: file_spec(output_dir / name, include_path=False) for name in names}
    if not json_exact(marker.get("published_files"), observed):
        raise ProtocolError("PUBLISHED file bindings mismatch")
    if list(marker["published_files"]) != names:
        raise ProtocolError("PUBLISHED file ordering mismatch")
    ordered = [{"name": name, **observed[name]} for name in names]
    if marker.get("ordered_inventory_sha256") != sha256_bytes(canonical_bytes(ordered)):
        raise ProtocolError("PUBLISHED inventory digest mismatch")
    if not json_exact(marker.get("receipt"), observed.get("receipt.json")) or not json_exact(
        marker.get("receipt_sidecar"), observed.get("receipt.json.sha256.json")
    ):
        raise ProtocolError("PUBLISHED receipt binding mismatch")
    for name in payload_names:
        sidecar = strict_json_load(output_dir / f"{name}.sha256.json")
        if not json_exact(sidecar, file_spec(output_dir / name)):
            raise ProtocolError(f"payload sidecar mismatch: {name}")
    return dict(marker)


def require_exact_keys(value: Mapping[str, Any], expected: Iterable[str], name: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ProtocolError(f"{name} keys mismatch")


def require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ProtocolError(f"{name} must be an integer")
    return value


def require_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ProtocolError(f"{name} must be finite")
    return result


def require_fresh_python_bootstrap() -> str:
    """Require the process-level no-bytecode/fresh-cache bootstrap.

    Entry-point scripts call an equivalent tiny guard *before importing local
    ``src`` modules*.  Library callers use this function immediately before
    starting a solver.  A pre-existing cache prefix is rejected rather than
    reused: this prevents an adjacent run's pyc from becoming an implicit
    input to a PRE-FORMAL result.
    """

    import sys

    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ProtocolError("PYTHONDONTWRITEBYTECODE must be exactly '1'")
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not isinstance(prefix, str) or not prefix or not os.path.isabs(prefix):
        raise ProtocolError("PYTHONPYCACHEPREFIX must be an absolute fresh absent path")
    if sys.dont_write_bytecode is not True:
        raise ProtocolError("sys.dont_write_bytecode must already be True")
    if type(sys.pycache_prefix) is not str or sys.pycache_prefix != prefix:
        raise ProtocolError("sys.pycache_prefix must exactly match PYTHONPYCACHEPREFIX")
    if os.path.lexists(prefix):
        raise ProtocolError("PYTHONPYCACHEPREFIX must be absent at process start")
    prefix_path = Path(prefix)
    # A symlink anywhere in the prefix's existing parent chain could redirect
    # imports or writes outside the audited run.  Parent components must be
    # real, existing directories; only the final prefix leaf is intentionally
    # absent and will remain absent because bytecode writes are disabled.
    for component in prefix_path.parents:
        if component.is_symlink():
            raise ProtocolError("PYTHONPYCACHEPREFIX parent contains a symlink")
        if not component.exists() or not component.is_dir():
            raise ProtocolError("PYTHONPYCACHEPREFIX parent is missing or not a directory")

    # This also protects library callers that entered through an embedding API
    # instead of a script-level bootstrap.
    sys.dont_write_bytecode = True
    return prefix
