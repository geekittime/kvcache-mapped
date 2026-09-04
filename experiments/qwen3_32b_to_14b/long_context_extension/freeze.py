#!/usr/bin/env python3
"""Build and verify the exact code freeze for the long-context extension.

The freeze never inventories itself or its SHA sidecar, which avoids a hash
cycle.  It does inventory this generator, the validator, every runtime module,
the frozen configuration, and the extension tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


EXTENSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXTENSION_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from long_context_extension.protocol import (  # noqa: E402
    CODE_IDENTITY_FILES,
    DIRECTION,
    EXTENSION_VERSION,
    MAPPER_MANIFEST,
    PAPER_BIT_EXACT,
    PROTOCOL_NAME,
    REPOSITORY_ROOT,
    REVERSE_RUNTIME_CODE_FILES,
)
from reverse_kv.protocol import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    read_json,
    sha256_file,
)


CODE_FREEZE = EXTENSION_ROOT / "code_freeze.json"
CODE_FREEZE_SIDECAR = EXTENSION_ROOT / "code_freeze.json.sha256.json"

SHARED_RUNTIME_FILES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/evaluate_long_context.py",
    "vendor/lm-evaluation-harness/lm_eval/tasks/longbench/metrics.py",
    "src/kvcache_transfer/__init__.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/transfer.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/io.py",
)


class CodeFreezeValidationError(ValueError):
    """The exact extension code inventory is missing or has drifted."""


def _unique(paths: Iterable[Path]) -> tuple[Path, ...]:
    values: set[Path] = set()
    for path in paths:
        lexical = path if path.is_absolute() else Path.cwd() / path
        if lexical.is_symlink():
            raise CodeFreezeValidationError(f"freeze input is symlinked: {lexical}")
        values.add(lexical.absolute())
    return tuple(sorted(values))


def _entry(path: Path, *, base: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise CodeFreezeValidationError(f"freeze input is symlinked: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        relative = str(resolved.relative_to(base.resolve()))
    except ValueError as exc:
        raise CodeFreezeValidationError(f"freeze path escaped base: {resolved}") from exc
    return {
        "path": str(resolved),
        "relative_path": relative,
        "nbytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _extension_paths() -> tuple[Path, ...]:
    paths = [EXPERIMENT_ROOT / relative for relative in CODE_IDENTITY_FILES]
    paths.extend((EXTENSION_ROOT / "tests").glob("*.py"))
    return _unique(paths)


def _reverse_paths() -> tuple[Path, ...]:
    return _unique(EXPERIMENT_ROOT / relative for relative in REVERSE_RUNTIME_CODE_FILES)


def _shared_paths() -> tuple[Path, ...]:
    return _unique(REPOSITORY_ROOT / relative for relative in SHARED_RUNTIME_FILES)


def _group(paths: Iterable[Path], *, base: Path) -> list[dict[str, Any]]:
    return [_entry(path, base=base) for path in _unique(paths)]


def build_code_freeze() -> dict[str, Any]:
    """Construct the deterministic unsigned inventory and its canonical digest."""

    base_freeze = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
    if not base_freeze.is_file():
        raise FileNotFoundError(base_freeze)
    mapper_sidecar = MAPPER_MANIFEST.with_name("manifest.sha256.json")
    if (
        MAPPER_MANIFEST.is_symlink()
        or mapper_sidecar.is_symlink()
        or not MAPPER_MANIFEST.is_file()
        or not mapper_sidecar.is_file()
    ):
        raise FileNotFoundError("reverse mapper manifest commit is incomplete")
    mapper_declared = read_json(mapper_sidecar)
    mapper_sha = sha256_file(MAPPER_MANIFEST)
    if mapper_declared.get("sha256") != mapper_sha:
        raise CodeFreezeValidationError("reverse mapper manifest commit changed")
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen",
        "kind": EXTENSION_VERSION,
        "direction": DIRECTION,
        "protocol": PROTOCOL_NAME,
        "paper_bit_exact": PAPER_BIT_EXACT,
        "extension_files": _group(_extension_paths(), base=EXPERIMENT_ROOT),
        "reverse_runtime_files": _group(_reverse_paths(), base=EXPERIMENT_ROOT),
        "shared_runtime_files": _group(_shared_paths(), base=REPOSITORY_ROOT),
        "base_reverse_code_freeze": {
            "path": str(base_freeze.resolve()),
            "sha256": sha256_file(base_freeze),
            "nbytes": base_freeze.stat().st_size,
        },
        "mapper_manifest_commit": {
            "manifest": str(MAPPER_MANIFEST.resolve()),
            "manifest_sha256": mapper_sha,
            "sidecar": str(mapper_sidecar.resolve()),
            "sidecar_sha256": sha256_file(mapper_sidecar),
            "sidecar_nbytes": mapper_sidecar.stat().st_size,
        },
        "cycle_policy": (
            "code_freeze.json and its sidecar are excluded from their own inventory"
        ),
    }
    return {**unsigned, "inventory_digest": canonical_sha256(unsigned)}


def write_code_freeze() -> dict[str, Any]:
    value = build_code_freeze()
    atomic_write_json(CODE_FREEZE, value)
    atomic_write_json(
        CODE_FREEZE_SIDECAR,
        {
            "schema_version": 1,
            "artifact": CODE_FREEZE.name,
            "sha256": sha256_file(CODE_FREEZE),
            "nbytes": CODE_FREEZE.stat().st_size,
        },
    )
    return validate_code_freeze()


def validate_code_freeze() -> dict[str, Any]:
    if (
        CODE_FREEZE.is_symlink()
        or CODE_FREEZE_SIDECAR.is_symlink()
        or not CODE_FREEZE.is_file()
        or not CODE_FREEZE_SIDECAR.is_file()
    ):
        raise CodeFreezeValidationError("mandatory extension code freeze is missing")
    sidecar = read_json(CODE_FREEZE_SIDECAR)
    expected_sidecar = {
        "schema_version": 1,
        "artifact": CODE_FREEZE.name,
        "sha256": sha256_file(CODE_FREEZE),
        "nbytes": CODE_FREEZE.stat().st_size,
    }
    if sidecar != expected_sidecar:
        raise CodeFreezeValidationError("extension code-freeze sidecar changed")
    value = read_json(CODE_FREEZE)
    expected = build_code_freeze()
    if value != expected:
        raise CodeFreezeValidationError("extension code inventory changed")
    unsigned = dict(value)
    digest = unsigned.pop("inventory_digest", None)
    if digest != canonical_sha256(unsigned):
        raise CodeFreezeValidationError("extension inventory digest changed")
    return dict(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    value = write_code_freeze() if args.write else validate_code_freeze()
    print(value["inventory_digest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CODE_FREEZE",
    "CODE_FREEZE_SIDECAR",
    "CodeFreezeValidationError",
    "build_code_freeze",
    "validate_code_freeze",
    "write_code_freeze",
]
