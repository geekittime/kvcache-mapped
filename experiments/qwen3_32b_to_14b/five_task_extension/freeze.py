#!/usr/bin/env python3
"""Publish the independent code/model/mapper freeze for this extension."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

EXTENSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXTENSION_ROOT.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from five_task_extension.protocol import (  # noqa: E402
    CODE_IDENTITY_FILES,
    EXTENSION_CODE_FREEZE,
    EXTENSION_VERSION,
    HELLASWAG_ROOT,
    MAPPER_MANIFEST,
)
from reverse_kv.protocol import atomic_write_json, canonical_sha256, read_json, sha256_file  # noqa: E402


SCORER_SHARED_FILES = (
    "src/kvcache_transfer/gsm8k_numeric_diagnostic.py",
    "scripts/rescore_gsm8k_numeric.py",
    "vendor/lm-evaluation-harness/lm_eval/filters/extraction.py",
    "vendor/lm-evaluation-harness/lm_eval/api/metrics.py",
    "vendor/lm-evaluation-harness/lm_eval/evaluator.py",
    "vendor/lm-evaluation-harness/lm_eval/utils.py",
)


def _entry(path: Path, base: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": str(resolved),
        "relative_path": str(resolved.relative_to(base.resolve())),
        "nbytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _unique(paths: Iterable[Path]) -> tuple[Path, ...]:
    return tuple(sorted({path.resolve() for path in paths}))


def _extension_paths() -> tuple[Path, ...]:
    paths = [EXPERIMENT_ROOT / relative for relative in CODE_IDENTITY_FILES]
    paths.extend((EXTENSION_ROOT / "tests").glob("*.py"))
    return _unique(paths)


def _reverse_paths() -> tuple[Path, ...]:
    return _unique(
        [*(EXPERIMENT_ROOT / "reverse_kv").glob("*.py"), *(EXPERIMENT_ROOT / "scripts").glob("*.py")]
    )


def _base_shared_paths(base: Mapping[str, Any]) -> tuple[Path, ...]:
    paths = [Path(str(row["path"])) for row in base["read_only_shared_dependencies"]]
    paths.extend(REPOSITORY_ROOT / relative for relative in SCORER_SHARED_FILES)
    return _unique(paths)


def _harness_paths() -> tuple[Path, ...]:
    root = REPOSITORY_ROOT / "vendor/lm-evaluation-harness/lm_eval"
    paths = [path for path in root.rglob("*.py") if "tasks" not in path.relative_to(root).parts]
    paths.extend((root / "tasks").glob("*.py"))
    for task in ("arc", "hellaswag", "winogrande", "mmlu", "gsm8k"):
        directory = root / "tasks" / task
        for pattern in ("*.py", "*.yaml", "*.yml", "*.json"):
            paths.extend(directory.rglob(pattern))
    return _unique(paths)


def _benchmark_paths() -> tuple[Path, ...]:
    root = REPOSITORY_ROOT / "data/benchmarks"
    paths = [root / "manifest.json"]
    local = root / "lm_eval_tasks"
    paths.extend(local.glob("arc_challenge.yaml"))
    paths.extend(local.glob("hellaswag.yaml"))
    paths.extend(local.glob("winogrande.yaml"))
    paths.extend(local.glob("gsm8k_cot.yaml"))
    paths.extend(local.glob("mmlu*.yaml"))
    return _unique(paths)


def _mapper_payload() -> dict[str, Any]:
    root = MAPPER_MANIFEST.parent.resolve()
    manifest = read_json(MAPPER_MANIFEST)
    paths = [MAPPER_MANIFEST, root / "manifest.sha256.json"]
    for row in manifest["layers"]:
        paths.extend((Path(str(row["path"])), Path(str(row["sidecar_path"]))))
    # The self-contained selection and activation-inventory snapshots are part
    # of the mapper's scientific identity even though they are not loaded for
    # each inference request.
    paths.extend(path for path in (root / "inputs").rglob("*") if path.is_file())
    entries = [_entry(path, root) for path in _unique(paths)]
    return {
        "root": str(root),
        "fit_fingerprint": manifest["fit_fingerprint"],
        "manifest_sha256": sha256_file(MAPPER_MANIFEST),
        "file_count": len(entries),
        "total_nbytes": sum(int(row["nbytes"]) for row in entries),
        "files": entries,
        "files_digest": canonical_sha256(entries),
        "every_mapper_payload_file_content_hashed": True,
    }


def _base_amendment(base_path: Path, base: Mapping[str, Any]) -> dict[str, Any]:
    drift: list[dict[str, Any]] = []
    for group in (
        "local_files",
        "read_only_shared_dependencies",
        "benchmark_protocol_dependencies",
        "vendored_harness_dependencies",
    ):
        for row in base[group]:
            path = Path(str(row["path"]))
            current_sha = sha256_file(path) if path.is_file() else None
            current_nbytes = path.stat().st_size if path.is_file() else None
            if current_sha != row.get("sha256") or current_nbytes != row.get("nbytes"):
                drift.append(
                    {
                        "group": group,
                        "path": str(path),
                        "base_sha256": row.get("sha256"),
                        "base_nbytes": row.get("nbytes"),
                        "current_sha256": current_sha,
                        "current_nbytes": current_nbytes,
                    }
                )
    return {
        "base_freeze": {
            "path": str(base_path.resolve()),
            "sha256": sha256_file(base_path),
            "inventory_digest": base["inventory_digest"],
        },
        "policy": "base_freeze_plus_explicit_current_extension_overlay",
        "base_is_not_claimed_as_current_tree": True,
        "known_drift": drift,
        "known_drift_digest": canonical_sha256(drift),
    }


def build_freeze() -> dict[str, Any]:
    base_path = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
    base = read_json(base_path)
    hella_definition = read_json(HELLASWAG_ROOT / "run_definition.json")
    groups = {
        "extension_files": [_entry(path, EXPERIMENT_ROOT) for path in _extension_paths()],
        "current_reverse_dependencies": [_entry(path, EXPERIMENT_ROOT) for path in _reverse_paths()],
        "current_shared_dependencies": [_entry(path, REPOSITORY_ROOT) for path in _base_shared_paths(base)],
        "current_vendored_harness_dependencies": [
            _entry(path, REPOSITORY_ROOT) for path in _harness_paths()
        ],
        "benchmark_protocol_dependencies": [
            _entry(path, REPOSITORY_ROOT) for path in _benchmark_paths()
        ],
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "frozen",
        "kind": EXTENSION_VERSION,
        "direction": "qwen3_32b_to_14b",
        "paper_bit_exact": False,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_freeze_amendment": _base_amendment(base_path, base),
        **groups,
        "models": {
            "identity_scope": "every model weight shard content SHA256",
            "source_32b": hella_definition["runtime_artifacts"]["source_model"],
            "target_14b": hella_definition["runtime_artifacts"]["target_model"],
        },
        "mapper_payload": _mapper_payload(),
        "dependency_policy": (
            "extension overlay content-hashes current extension, reverse, shared, "
            "vendored harness/task and benchmark protocol files"
        ),
    }
    unsigned = dict(value)
    value["inventory_digest"] = canonical_sha256(unsigned)
    return value


def main() -> int:
    value = build_freeze()
    atomic_write_json(EXTENSION_CODE_FREEZE, value)
    atomic_write_json(
        EXTENSION_CODE_FREEZE.with_name(f"{EXTENSION_CODE_FREEZE.name}.sha256.json"),
        {
            "schema_version": 1,
            "artifact": EXTENSION_CODE_FREEZE.name,
            "sha256": sha256_file(EXTENSION_CODE_FREEZE),
            "nbytes": EXTENSION_CODE_FREEZE.stat().st_size,
        },
    )
    print(f"{EXTENSION_CODE_FREEZE} {sha256_file(EXTENSION_CODE_FREEZE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
