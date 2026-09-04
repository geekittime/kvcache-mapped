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
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class ProtocolError(RuntimeError):
    """Raised for any contract violation (the protocol fails closed)."""


_FILE_SPEC_KEYS = {"path", "nbytes", "sha256"}
_RUNTIME_BINDING_KEYS = {
    "python_executable",
    "python_prefix",
    "python_real_executable",
    "python_real_executable_sha256",
    "python_major_minor",
    "venv_required",
}


def formal_config_path(root: Path, config_path: Path) -> Path:
    """Return the one canonical formal-config path for this experiment.

    A run may not silently substitute a second JSON file (or a symlink) for
    the frozen configuration.  Every parent and the leaf are checked with
    ``lstat`` before any bytes are consumed.
    """

    root = Path(root)
    config_path = Path(config_path)
    require_directory(root)
    if not config_path.is_absolute():
        raise ProtocolError("formal config path must be absolute")
    expected = root / "config.json"
    if config_path != expected:
        raise ProtocolError(f"formal config path must be exactly {expected}")
    require_regular_file(config_path)
    return config_path


def config_file_binding(path: Path) -> dict[str, Any]:
    """Return a content binding for a regular, non-symlink config file."""

    require_regular_file(Path(path))
    return file_spec(Path(path))


def verify_config_file_binding(path: Path, binding: Mapping[str, Any], *, name: str = "formal config") -> dict[str, Any]:
    """Rehash and strictly verify a worker's config binding."""

    if not isinstance(binding, Mapping) or set(binding) != _FILE_SPEC_KEYS:
        raise ProtocolError(f"{name} binding keys mismatch")
    if type(binding.get("nbytes")) is not int or binding["nbytes"] < 0:
        raise ProtocolError(f"{name} nbytes must be a strict nonnegative integer")
    digest = binding.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ProtocolError(f"{name} sha256 must be lowercase hexadecimal")
    path = Path(path)
    if binding.get("path") != str(path):
        raise ProtocolError(f"{name} path binding mismatch")
    observed = config_file_binding(path)
    if not json_exact(observed, dict(binding)):
        raise ProtocolError(f"{name} content binding mismatch")
    return observed


def runtime_binding_shape(binding: Mapping[str, Any]) -> None:
    """Validate the immutable interpreter/venv binding schema."""

    if not isinstance(binding, Mapping) or set(binding) != _RUNTIME_BINDING_KEYS:
        raise ProtocolError("runtime binding keys mismatch")
    for key in ("python_executable", "python_prefix", "python_real_executable", "python_real_executable_sha256", "python_major_minor"):
        if not isinstance(binding.get(key), str) or not binding[key] or "\x00" in binding[key]:
            raise ProtocolError(f"runtime binding {key} must be a nonempty string")
    digest = binding["python_real_executable_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ProtocolError("runtime executable digest must be lowercase hexadecimal")
    if type(binding.get("venv_required")) is not bool or binding["venv_required"] is not True:
        raise ProtocolError("runtime binding requires the pinned virtual environment")
    if not Path(binding["python_executable"]).is_absolute() or not Path(binding["python_prefix"]).is_absolute() or not Path(binding["python_real_executable"]).is_absolute():
        raise ProtocolError("runtime binding paths must be absolute")


def validate_runtime_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Require that the current process is the pinned venv interpreter."""

    runtime_binding_shape(binding)
    expected_executable = binding["python_executable"]
    expected_prefix = binding["python_prefix"]
    expected_real = binding["python_real_executable"]
    if sys.executable != expected_executable:
        raise ProtocolError("sys.executable does not match the frozen venv executable")
    if sys.prefix != expected_prefix or sys.base_prefix == sys.prefix:
        raise ProtocolError("Python process is not running inside the frozen virtual environment")
    observed_real = os.path.realpath(sys.executable)
    if observed_real != expected_real:
        raise ProtocolError("resolved Python executable does not match the frozen real executable")
    real_path = Path(expected_real)
    if not real_path.exists() or not real_path.is_file() or real_path.is_symlink():
        raise ProtocolError("frozen real Python executable is missing/non-regular")
    if sha256_file(real_path) != binding["python_real_executable_sha256"]:
        raise ProtocolError("frozen real Python executable digest mismatch")
    major_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if major_minor != binding["python_major_minor"]:
        raise ProtocolError("Python major/minor does not match runtime binding")
    return {
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "python_real_executable": observed_real,
        "python_real_executable_sha256": binding["python_real_executable_sha256"],
        "python_major_minor": major_minor,
        "python_version": sys.version,
    }


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


def implementation_inventory(root: Path) -> list[dict[str, Any]]:
    """Return a deterministic, link-free inventory of executable source.

    Configuration and publication output are intentionally excluded from this
    *implementation* inventory.  The formal config is content-bound
    separately, which avoids an impossible self-reference when a config binds
    the hash of the PRE-FORMAL marker that records this inventory.
    """

    root = Path(root)
    require_directory(root)
    excluded = {
        "artifacts",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "config.json",
    }
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root)
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"implementation inventory forbids symlink: {path}")
        if not stat.S_ISREG(mode) and not stat.S_ISDIR(mode):
            raise ProtocolError(f"implementation inventory forbids nonregular path: {path}")
        if any(part in excluded for part in rel.parts):
            continue
        if stat.S_ISDIR(mode):
            continue
        rows.append({
            "path": rel.as_posix(),
            "nbytes": os.lstat(path).st_size,
            "sha256": sha256_file(path),
        })
    return rows


def implementation_inventory_digest(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash the canonical implementation inventory list."""

    canonical_rows = [dict(row) for row in rows]
    return sha256_bytes(canonical_bytes(canonical_rows))


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


def _ensure_real_directory_chain(path: Path) -> None:
    """Create *path* component-by-component without following links.

    ``Path.mkdir(parents=True)`` is deliberately not used by publication
    code: it follows a malicious symlink in an existing parent.  Every
    component is lstat'ed immediately before and after a possible mkdir.
    """

    path = Path(path)
    if not path.is_absolute():
        raise ProtocolError(f"directory path must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            try:
                os.mkdir(current)
            except FileExistsError:
                # A concurrent creator is acceptable only if it created a
                # real directory; the lstat below closes that race.
                pass
            try:
                mode = os.lstat(current).st_mode
            except FileNotFoundError as exc:
                raise ProtocolError(f"directory component disappeared: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"symlink directory component forbidden: {current}")
        if not stat.S_ISDIR(mode):
            raise ProtocolError(f"directory component is not a directory: {current}")


def safe_output_child(root: Path, relative: str, *, allow_missing: bool = True) -> Path:
    """Resolve/create an output child while rejecting symlink parent races.

    Unlike :func:`safe_child`, this helper is intended for writers and may
    create missing intermediate directories.  The leaf is never created by
    this function.  All components are checked with ``lstat`` and the final
    real-path containment check is repeated before returning.
    """

    root = Path(root)
    require_directory(root)
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(part in {"", ".", ".."} for part in rel.parts):
        raise ProtocolError(f"invalid output relative path: {relative}")
    current = root
    parts = rel.parts
    for index, part in enumerate(parts):
        current = current / part
        is_leaf = index == len(parts) - 1
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if is_leaf and allow_missing:
                break
            if is_leaf:
                raise ProtocolError(f"missing output leaf: {current}")
            try:
                os.mkdir(current)
            except FileExistsError:
                pass
            try:
                mode = os.lstat(current).st_mode
            except FileNotFoundError as exc:
                raise ProtocolError(f"output parent disappeared: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"symlink output component forbidden: {current}")
        if not is_leaf and not stat.S_ISDIR(mode):
            raise ProtocolError(f"output parent is not a directory: {current}")
    try:
        current.resolve(strict=False).relative_to(root.resolve(strict=True))
    except (ValueError, OSError) as exc:
        raise ProtocolError(f"output path escapes experiment root: {relative}") from exc
    # Re-check every existing component immediately before a caller opens it.
    _walk_no_symlink(current.parent)
    return current


def _assert_atomic_parent(path: Path) -> None:
    """Validate an atomic target's parent and absent leaf at one instant."""

    path = Path(path)
    if not path.is_absolute():
        raise ProtocolError(f"atomic output path must be absolute: {path}")
    _walk_no_symlink(path.parent)
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISLNK(mode):
        raise ProtocolError(f"atomic output symlink forbidden: {path}")
    raise ProtocolError(f"refusing to overwrite existing path: {path}")


def atomic_write(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    """Atomically create a file and refuse overwrite."""

    path = Path(path)
    _assert_atomic_parent(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        # A parent can be replaced by a link after mkstemp.  Refuse that race
        # before bytes are committed and once more immediately before rename.
        _walk_no_symlink(path.parent)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, mode)
        _assert_atomic_parent(path)
        os.replace(temporary_path, path)
        _walk_no_symlink(path.parent)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
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

    output_dir = Path(output_dir)
    _ensure_real_directory_chain(output_dir.parent)
    _walk_no_symlink(output_dir.parent)
    lock_path = output_dir.parent / f".{output_dir.name}.single_writer.lock"
    # Opening with append is safe only after the complete parent chain has
    # been lstat-checked; reject an existing lock symlink instead of following
    # it.  A lock file is intentionally non-artifact metadata.
    try:
        lock_mode = os.lstat(lock_path).st_mode
    except FileNotFoundError:
        lock_mode = None
    if lock_mode is not None and (stat.S_ISLNK(lock_mode) or not stat.S_ISREG(lock_mode)):
        raise ProtocolError(f"artifact lock path is not a regular file: {lock_path}")
    try:
        lock_fd = os.open(
            lock_path,
            os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except OSError as exc:
        raise ProtocolError(f"artifact lock path cannot be opened safely: {lock_path}") from exc
    lock = os.fdopen(lock_fd, "a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ProtocolError(f"another writer owns {output_dir}") from exc
        _walk_no_symlink(output_dir.parent)
        try:
            existing_mode = os.lstat(output_dir).st_mode
        except FileNotFoundError:
            existing_mode = None
        if existing_mode is not None:
            raise ProtocolError(f"refusing to reuse artifact directory: {output_dir}")
        try:
            os.mkdir(output_dir)
        except FileExistsError as exc:
            raise ProtocolError(f"refusing to reuse artifact directory: {output_dir}") from exc
        require_directory(output_dir)
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

    output_dir = Path(output_dir)
    require_directory(output_dir)
    _walk_no_symlink(output_dir)
    if len(payload_names) != len(set(payload_names)) or any(Path(n).name != n for n in payload_names):
        raise ProtocolError("publication payload names must be unique basenames")
    for name in payload_names:
        safe_output_child(output_dir, name, allow_missing=False)
        require_regular_file(output_dir / name)
        atomic_json(output_dir / f"{name}.sha256.json", file_spec(output_dir / name))
    names = sorted([*payload_names, *(f"{name}.sha256.json" for name in payload_names)])
    expected = sorted([*names, "IN_PROGRESS.json"])
    if sorted(p.name for p in output_dir.iterdir()) != expected:
        raise ProtocolError("publication inventory mismatch before commit")
    for name in expected:
        require_regular_file(output_dir / name)
    in_progress = output_dir / "IN_PROGRESS.json"
    require_regular_file(in_progress)
    _walk_no_symlink(output_dir)
    in_progress.unlink()
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
    safe_output_child(output_dir, "PUBLISHED.json", allow_missing=True)
    atomic_json(output_dir / "PUBLISHED.json", marker)
    return marker


def validate_publication(
    output_dir: Path,
    payload_names: list[str],
    *,
    identity: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    require_directory(output_dir)
    _walk_no_symlink(output_dir)
    marker_path = output_dir / "PUBLISHED.json"
    safe_output_child(output_dir, "PUBLISHED.json", allow_missing=False)
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
