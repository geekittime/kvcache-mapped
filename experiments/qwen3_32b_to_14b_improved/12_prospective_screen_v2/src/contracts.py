"""Fail-closed, content-addressed contracts for prospective evaluation v2.

This module deliberately has no dependency on experiment 07.  Experiment 07
is a read-only design reference whose hashes are recorded in
``reference_interfaces.json``; importing it would silently couple the new
protocol to mutable legacy code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
# The historical baseline provenance receipt predates enumeration of these
# two package files.  Experiment 12 binds them independently so the actual
# import closure remains content-addressed without rewriting that receipt.
RUNTIME_PACKAGE_CLOSURE_ROOT = "/home/wby/llm-cache/nvidia-kvcache"
RUNTIME_PACKAGE_CLOSURE_FILES = {
    "src/kvcache_transfer/__init__.py": {
        "nbytes": 98,
        "sha256": "9e433f87c7bb0b7c5b70a095e0a71f306b4653717146144043b56dd5d6c7b794",
    },
    "src/kvcache_transfer/io.py": {
        "nbytes": 1251,
        "sha256": "6422adcef32f7fdc153c842ff2c734b9b412dd7b3fb93900cc853416d4c67766",
    },
}
RAW_MODES = ("native14", "nvidia_k20", "shrinkage", "tail_refresh_r32")
DERIVED_MODES = ("two_pass_r1_then_r32",)
ALL_MODES = RAW_MODES + DERIVED_MODES
DATASETS = ("hotpotqa_train", "musique_train")
PARTITIONS = ("screen", "confirm")
PROFILES = ("smoke", "screen", "confirm")
SPLIT_KEYS = (
    "hotpotqa_screen",
    "hotpotqa_confirm",
    "musique_screen",
    "musique_confirm",
)
SMOKE_COUNTS = {"hotpotqa_train": 2, "musique_train": 2}
PROFILE_MAX_COUNTS = {
    "smoke": {"hotpotqa_train": 2, "musique_train": 2},
    "screen": {"hotpotqa_train": 128, "musique_train": 256},
    "confirm": {"hotpotqa_train": 256, "musique_train": 512},
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER_MARKERS = ("__PENDING_", "__PLACEHOLDER_", "TEMPLATE_ONLY")


class ContractError(RuntimeError):
    """A frozen input, result, gate, or receipt violated its contract."""


def strict_int(
    value: Any,
    *,
    label: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return an exact JSON integer, explicitly rejecting Python booleans."""

    if type(value) is not int:
        raise ContractError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ContractError(f"{label} must be <= {maximum}")
    return value


def exact_json_equal(left: Any, right: Any) -> bool:
    """Recursively compare canonical JSON values without bool/int aliasing."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            exact_json_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            exact_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc
    return payload + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    for value in values:
        strict_int(value, label="token ID", minimum=0)
    return canonical_sha256(list(values))


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON at {path}: {exc}") from exc


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ContractError(f"blank JSONL row at {path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(row)
    return rows


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    encoded = [canonical_bytes(dict(row)) for row in rows]
    return (b"\n".join(encoded) + b"\n") if encoded else b""


def _reject_symlinks(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ContractError(f"{label} contains a symlink: {component}")


def require_child(root: str | Path, value: str | Path, *, label: str) -> Path:
    root_path = Path(root)
    value_path = Path(value)
    _reject_symlinks(root_path, f"{label} root")
    _reject_symlinks(value_path, label)
    root_resolved = root_path.resolve()
    value_resolved = value_path.resolve()
    try:
        relative = value_resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ContractError(f"{label} escapes {root_resolved}: {value_resolved}") from exc
    if not relative.parts:
        raise ContractError(f"{label} cannot equal its root")
    return value_resolved


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    nbytes: int
    sha256: str

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "ArtifactRef":
        if not isinstance(value, Mapping) or set(value) != {"path", "nbytes", "sha256"}:
            raise ContractError(f"{label} must contain exactly path/nbytes/sha256")
        path = value.get("path")
        nbytes = value.get("nbytes")
        digest = value.get("sha256")
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ContractError(f"{label}.path must be absolute")
        if type(nbytes) is not int or nbytes < 0:
            raise ContractError(f"{label}.nbytes must be a non-negative integer")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ContractError(f"{label}.sha256 is malformed")
        return cls(path, nbytes, digest)

    def verify(self, *, label: str, deep_hash: bool = True) -> Path:
        lexical = Path(self.path)
        _reject_symlinks(lexical, label)
        path = lexical.resolve()
        if not path.is_file():
            raise ContractError(f"{label} is not a regular file: {path}")
        if path.stat().st_size != self.nbytes:
            raise ContractError(f"{label} byte size drifted")
        if deep_hash and sha256_file(path) != self.sha256:
            raise ContractError(f"{label} SHA-256 drifted")
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifact_ref(path: str | Path) -> ArtifactRef:
    lexical = Path(path)
    _reject_symlinks(lexical, "artifact")
    resolved = lexical.resolve()
    if not resolved.is_file():
        raise ContractError(f"artifact is not a regular file: {resolved}")
    return ArtifactRef(str(resolved), resolved.stat().st_size, sha256_file(resolved))


def _atomic_write(path: Path, payload: bytes) -> None:
    _reject_symlinks(path, "atomic destination")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinks(path, "atomic destination")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def immutable_json(path: str | Path, value: Any) -> ArtifactRef:
    destination = Path(path)
    payload = canonical_bytes(value, newline=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ContractError(f"immutable path is not a regular file: {destination}")
        if destination.read_bytes() != payload:
            raise ContractError(f"immutable artifact differs: {destination}")
    else:
        _atomic_write(destination, payload)
    return artifact_ref(destination)


def atomic_json(path: str | Path, value: Any) -> ArtifactRef:
    destination = Path(path)
    _atomic_write(destination, canonical_bytes(value, newline=True))
    return artifact_ref(destination)


def immutable_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> ArtifactRef:
    destination = Path(path)
    payload = jsonl_bytes(rows)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ContractError(f"immutable path is not a regular file: {destination}")
        if destination.read_bytes() != payload:
            raise ContractError(f"immutable artifact differs: {destination}")
    else:
        _atomic_write(destination, payload)
    return artifact_ref(destination)


def write_sidecar(reference: ArtifactRef) -> ArtifactRef:
    path = Path(reference.path)
    return immutable_json(
        path.with_name(path.name + ".sha256.json"),
        {
            "schema_version": 1,
            "path": path.name,
            "nbytes": reference.nbytes,
            "sha256": reference.sha256,
        },
    )


def verify_sidecar(path: str | Path) -> None:
    artifact = artifact_ref(path)
    sidecar = Path(path).with_name(Path(path).name + ".sha256.json")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ContractError(f"missing SHA sidecar: {sidecar}")
    expected = {
        "schema_version": 1,
        "path": Path(path).name,
        "nbytes": artifact.nbytes,
        "sha256": artifact.sha256,
    }
    if not exact_json_equal(read_json(sidecar), expected):
        raise ContractError(f"SHA sidecar mismatch: {sidecar}")


def finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def _placeholder_paths(value: Any, prefix: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, str) and any(marker in value for marker in PLACEHOLDER_MARKERS):
        result.append(prefix)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            result.extend(_placeholder_paths(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_placeholder_paths(child, f"{prefix}[{index}]"))
    return result


def assert_no_placeholders(value: Any) -> None:
    paths = _placeholder_paths(value)
    if paths:
        raise ContractError(f"configuration contains unresolved placeholders: {paths[:8]}")


def code_inventory() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(ROOT.rglob("*")):
        # ``Path.is_file``/``stat`` follow symlinks.  Following an attacker
        # supplied link would make the supposedly content-addressed code
        # inventory describe bytes outside the experiment tree (or silently
        # omit a non-regular executable).  Inspect every entry with lstat and
        # fail closed before applying ignored-directory filters.
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise ContractError(f"code inventory entry disappeared: {path}") from exc
        if stat.S_ISLNK(mode):
            raise ContractError(f"code inventory refuses symlink: {path}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ContractError(f"code inventory refuses non-regular file: {path}")
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"artifacts", "runs", "logs"}:
            continue
        if any(
            part in {
                "__pycache__",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                ".git",
            }
            for part in relative.parts
        ) or path.suffix in {".pyc", ".pyo"}:
            continue
        files[str(relative)] = {
            "nbytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"files": files, "digest": canonical_sha256(files)}


__all__ = [
    "ALL_MODES",
    "ArtifactRef",
    "ContractError",
    "DATASETS",
    "DERIVED_MODES",
    "PARTITIONS",
    "PROFILE_MAX_COUNTS",
    "PROFILES",
    "RAW_MODES",
    "RUNTIME_PACKAGE_CLOSURE_FILES",
    "RUNTIME_PACKAGE_CLOSURE_ROOT",
    "ROOT",
    "SPLIT_KEYS",
    "SMOKE_COUNTS",
    "artifact_ref",
    "atomic_json",
    "assert_no_placeholders",
    "canonical_bytes",
    "canonical_sha256",
    "code_inventory",
    "exact_json_equal",
    "finite_float",
    "immutable_json",
    "immutable_jsonl",
    "read_json",
    "read_jsonl",
    "require_child",
    "sha256_file",
    "strict_int",
    "token_ids_sha256",
    "verify_sidecar",
    "write_sidecar",
]
