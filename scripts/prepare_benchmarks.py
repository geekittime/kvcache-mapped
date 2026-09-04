#!/usr/bin/env python3
"""Download, verify, and wire the paper benchmarks for fully offline lm-eval."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.eval_data import (  # noqa: E402
    MMLU_SUBJECTS,
    attach_task_overrides,
    build_benchmark_manifest,
    download_missing_from_modelscope,
    generate_local_task_overrides,
    load_benchmark_config,
    missing_dataset_files,
    verify_manifest_modelscope_origins,
    verify_ready_manifest,
    write_benchmark_manifest,
)


DEFAULT_CONFIG = ROOT / "configs" / "benchmarks_modelscope.yaml"
DEFAULT_HARNESS = ROOT / "vendor" / "lm-evaluation-harness"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--harness-root", type=Path, default=DEFAULT_HARNESS)
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="download only absent required parquet files from pinned ModelScope commits",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload every required parquet (implies --download-missing)",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="reuse canonical hashes from an existing verified manifest instead of querying HF",
    )
    parser.add_argument(
        "--skip-modelscope-origin-check",
        action="store_true",
        help="reuse ModelScope hashes from an existing verified manifest",
    )
    parser.add_argument(
        "--skip-task-loader-check",
        action="store_true",
        help="skip instantiating all five canonical lm-eval tasks offline",
    )
    return parser.parse_args(argv)


def _validate_task_loading(task_dir: Path) -> dict[str, object]:
    # This is deliberately offline. A passing check proves the generated task
    # YAMLs do not silently fall back to a hub dataset.
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=task_dir)
    loaded = manager.load(["hellaswag", "arc_challenge", "winogrande", "mmlu", "gsm8k_cot"])
    tasks = loaded["tasks"]
    expected = {"hellaswag", "arc_challenge", "winogrande", "gsm8k_cot"}
    expected.update(f"mmlu_{name}" for name in MMLU_SUBJECTS)
    missing = expected - set(tasks)
    if missing:
        raise RuntimeError(f"local TaskManager did not resolve canonical tasks: {sorted(missing)}")
    nonlocal_loaders = {
        name: task.config.dataset_path
        for name, task in tasks.items()
        if name in expected and task.config.dataset_path != "parquet"
    }
    if nonlocal_loaders:
        raise RuntimeError(f"tasks did not select local parquet: {nonlocal_loaders}")
    return {
        "task_count": len(expected),
        "resolved_tasks": sorted(expected),
        "all_dataset_paths": "parquet",
        "offline_environment": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_workers <= 0:
        raise ValueError("--max-workers must be positive")
    configuration = load_benchmark_config(args.config, ROOT)
    before = missing_dataset_files(configuration)
    if args.download_missing or args.force_download:
        download_missing_from_modelscope(
            configuration, force=args.force_download, max_workers=args.max_workers
        )
    after = missing_dataset_files(configuration)
    if after:
        print(json.dumps({"status": "incomplete", "missing": after}, indent=2), file=sys.stderr)
        return 2

    manifest = build_benchmark_manifest(
        configuration, verify_origin_online=not args.offline
    )
    if not args.skip_modelscope_origin_check:
        manifest = verify_manifest_modelscope_origins(configuration, manifest)
    generated = generate_local_task_overrides(configuration, args.harness_root)
    manifest = attach_task_overrides(manifest, generated)
    manifest["manifest_path"] = str(Path(configuration["manifest"]).resolve())
    if not args.skip_task_loader_check:
        manifest["offline_lm_eval_loader_check"] = _validate_task_loading(
            Path(configuration["local_task_dir"])
        )
    write_benchmark_manifest(configuration, manifest)
    verified = verify_ready_manifest(configuration["manifest"])
    summary = {
        "status": verified["status"],
        "manifest": str(configuration["manifest"]),
        "local_task_dir": str(configuration["local_task_dir"]),
        "missing_before": before,
        "missing_after": after,
        "datasets": {
            name: {
                "rows": value["split_rows"],
                "files": len(value["files"]),
                "canonical_byte_identical": value["all_files_byte_identical_to_canonical"],
            }
            for name, value in verified["datasets"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
