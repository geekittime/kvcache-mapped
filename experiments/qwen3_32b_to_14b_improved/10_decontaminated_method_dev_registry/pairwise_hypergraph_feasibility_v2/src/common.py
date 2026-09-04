from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import sys
import tempfile
import types
from pathlib import Path
from typing import Any, Iterable, Mapping


class ProtocolError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n").encode()


def strict_json_loads(payload: str | bytes) -> Any:
    def reject_constant(value: str) -> Any:
        raise ProtocolError(f"nonfinite JSON number is forbidden: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError(f"duplicate JSON object key is forbidden: {key}")
            result[key] = value
        return result

    return json.loads(
        payload, parse_constant=reject_constant, object_pairs_hook=unique_object
    )


def json_exact(left: Any, right: Any) -> bool:
    """JSON-semantic equality that does not alias True/False with 1/0."""
    return canonical_bytes(left) == canonical_bytes(right)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_spec(path: Path, *, include_path: bool = True) -> dict[str, Any]:
    value: dict[str, Any] = {"nbytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"path": str(path), **value} if include_path else value


def require_regular_file(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ProtocolError(f"missing regular file path component: {current}") from error
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"regular file path traverses a symlink: {current}")
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise ProtocolError(f"missing regular file: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise ProtocolError(f"artifact member is not a regular non-symlink file: {path}")


def verify_file(path: Path, spec: Mapping[str, Any]) -> None:
    if (
        type(spec.get("nbytes")) is not int
        or spec["nbytes"] < 0
        or not isinstance(spec.get("sha256"), str)
        or len(spec["sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in spec["sha256"])
    ):
        raise ProtocolError(f"invalid pinned file specification: {path}")
    require_regular_file(path)
    if file_spec(path, include_path=False) != {
        "nbytes": spec["nbytes"], "sha256": spec["sha256"]
    }:
        raise ProtocolError(f"pinned file mismatch: {path}")


def verify_sidecar(sidecar_path: Path, payload_path: Path, payload_spec: Mapping[str, Any]) -> None:
    verify_file(sidecar_path, {
        "nbytes": sidecar_path.stat().st_size,
        "sha256": sha256_file(sidecar_path),
    })
    observed = strict_json_loads(sidecar_path.read_text())
    expected = {
        "path": str(payload_path),
        "nbytes": payload_spec["nbytes"],
        "sha256": payload_spec["sha256"],
    }
    if not json_exact(observed, expected):
        raise ProtocolError(f"sidecar semantic mismatch: {sidecar_path}")


def load_module(name: str, path: Path, expected: Mapping[str, Any] | None = None):
    # Execute the exact bytes that callers content-addressed. SourceFileLoader
    # may otherwise accept an adjacent timestamp pyc, which weakens the source
    # digest claim even when bytecode writing is disabled.
    require_regular_file(path)
    source = path.read_bytes()
    if expected is not None and {
        "nbytes": len(source), "sha256": hashlib.sha256(source).hexdigest()
    } != {"nbytes": expected.get("nbytes"), "sha256": expected.get("sha256")}:
        raise ProtocolError(f"module bytes do not match pinned source: {path}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def read_pinned_json(path: Path, expected: Mapping[str, Any]) -> Any:
    require_regular_file(path)
    payload = path.read_bytes()
    if {
        "nbytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()
    } != {"nbytes": expected.get("nbytes"), "sha256": expected.get("sha256")}:
        raise ProtocolError(f"JSON bytes do not match pinned source: {path}")
    return strict_json_loads(payload)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def claim_artifact(output_dir: Path, identity: Mapping[str, Any]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir.parent / f".{output_dir.name}.single_writer.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise ProtocolError(f"another writer owns {output_dir}") from error
    try:
        if output_dir.exists():
            raise ProtocolError(f"refusing to reuse artifact directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        atomic_json(output_dir / "IN_PROGRESS.json", {
            "schema_version": 1,
            **dict(identity),
            "status": "in_progress_not_published",
            "completion_marker_required": "PUBLISHED.json",
        })
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def finalize_publication(
    output_dir: Path,
    payload_names: list[str],
    *,
    identity: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ProtocolError("publication target must be a non-symlink directory")
    if len(payload_names) != len(set(payload_names)) or any(
        Path(name).name != name for name in payload_names
    ):
        raise ProtocolError("publication payload names must be unique basenames")
    for name in payload_names:
        require_regular_file(output_dir / name)
        atomic_json(output_dir / f"{name}.sha256.json", file_spec(output_dir / name))
    names = sorted([*payload_names, *(f"{name}.sha256.json" for name in payload_names)])
    expected_before_commit = sorted([*names, "IN_PROGRESS.json"])
    if sorted(path.name for path in output_dir.iterdir()) != expected_before_commit:
        raise ProtocolError("pre-publication exact inventory mismatch")
    for name in expected_before_commit:
        require_regular_file(output_dir / name)
    (output_dir / "IN_PROGRESS.json").unlink()
    if sorted(path.name for path in output_dir.iterdir()) != names:
        raise ProtocolError("payload inventory changed during final commit")
    published_files = {name: file_spec(output_dir / name, include_path=False) for name in names}
    ordered = [{"name": name, **published_files[name]} for name in names]
    marker = {
        "schema_version": 1,
        **dict(identity),
        "status": status,
        "atomic_last_completion_marker": True,
        "published_files": published_files,
        "ordered_inventory_sha256": hashlib.sha256(canonical_bytes(ordered)).hexdigest(),
        "receipt": published_files["receipt.json"],
        "receipt_sidecar": published_files["receipt.json.sha256.json"],
    }
    atomic_json(output_dir / "PUBLISHED.json", marker)
    return marker


def validate_publication(
    output_dir: Path,
    payload_names: list[str],
    *,
    identity: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ProtocolError("published artifact directory must be a non-symlink directory")
    marker_path = output_dir / "PUBLISHED.json"
    require_regular_file(marker_path)
    names = sorted([*payload_names, *(f"{name}.sha256.json" for name in payload_names)])
    if sorted(path.name for path in output_dir.iterdir()) != sorted([*names, "PUBLISHED.json"]):
        raise ProtocolError("published exact inventory mismatch")
    for name in names:
        require_regular_file(output_dir / name)
    marker = strict_json_loads(marker_path.read_text())
    expected_keys = {
        "schema_version", *identity, "status", "atomic_last_completion_marker",
        "published_files", "ordered_inventory_sha256", "receipt", "receipt_sidecar",
    }
    if set(marker) != expected_keys or type(marker["schema_version"]) is not int or marker["schema_version"] != 1 or any(
        marker.get(key) != value for key, value in identity.items()
    ) or marker["status"] != status or marker["atomic_last_completion_marker"] is not True:
        raise ProtocolError("PUBLISHED marker semantics mismatch")
    observed = {name: file_spec(output_dir / name, include_path=False) for name in names}
    if not json_exact(marker["published_files"], observed) or list(marker["published_files"]) != names:
        raise ProtocolError("PUBLISHED file bindings mismatch")
    ordered = [{"name": name, **observed[name]} for name in names]
    if marker["ordered_inventory_sha256"] != hashlib.sha256(canonical_bytes(ordered)).hexdigest():
        raise ProtocolError("PUBLISHED ordered digest mismatch")
    if not json_exact(marker["receipt"], observed["receipt.json"]) or not json_exact(marker["receipt_sidecar"], observed["receipt.json.sha256.json"]):
        raise ProtocolError("PUBLISHED receipt binding mismatch")
    for name in payload_names:
        if not json_exact(
            strict_json_loads((output_dir / f"{name}.sha256.json").read_text()),
            file_spec(output_dir / name),
        ):
            raise ProtocolError(f"sidecar mismatch: {name}")
    return marker


def safe_relative_output(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ProtocolError("output must be relative below protocol root")
    if root.is_symlink() or not root.is_dir() or root.absolute() != root.resolve():
        raise ProtocolError("protocol root must be an existing canonical non-symlink directory")
    candidate = root / path
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ProtocolError(f"output path traverses a symlink: {current}")
    if not candidate.resolve(strict=False).is_relative_to(root.resolve()):
        raise ProtocolError("resolved output escapes protocol root")
    return candidate


def runtime_identity() -> dict[str, Any]:
    import networkx
    import numpy
    import scipy
    import scipy.optimize._highs._highs_wrapper as highs_wrapper

    wrapper_path = Path(highs_wrapper.__file__)
    require_regular_file(wrapper_path)
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "networkx_version": networkx.__version__,
        "highs_wrapper_binary": file_spec(wrapper_path),
    }


def verify_runtime_identity(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = runtime_identity()
    if not json_exact(observed, expected):
        raise ProtocolError(f"runtime identity mismatch: {observed}")
    return observed
