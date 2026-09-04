#!/usr/bin/env python3
"""Run final CPU checks and publish an immutable attempt29 STOP marker.

This command is intentionally one-way: it refuses to overwrite an existing
receipt or STOP marker.  Run it only after all source edits are complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.preflight import (  # noqa: E402
    CONFIG_PATH,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    RECEIPT_PATH,
    PreflightError,
    atomic_write,
    canonical_bytes,
    implementation_inventory,
    run_preflight,
    sha256_file,
    strict_regular,
)


STOP_PATH = ROOT / "artifacts" / "STOP_ROUTER_attempt29_v1.json"
PYTHON = Path(sys.executable).resolve()


def _run_cpu_command(args: list[str], label: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"router29_{label}_") as cache:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPYCACHEPREFIX"] = cache
        env["PYTHONHASHSEED"] = "0"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        completed = subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise PreflightError(f"{label} failed ({completed.returncode}): {completed.stdout[-2000:]}")
        return {
            "label": label,
            "returncode": completed.returncode,
            "output_tail": completed.stdout[-1000:],
            "cache_removed": not Path(cache).exists(),
        }


def _artifact_ref(path: Path) -> dict[str, object]:
    strict_regular(path)
    return {"path": str(path), "nbytes": path.stat().st_size, "sha256": sha256_file(path)}


def _closure(inventory: dict[str, object], receipt_ref: dict[str, object], receipt_sidecar_ref: dict[str, object]) -> dict[str, object]:
    files = list(inventory["files"])
    for path in (receipt_ref["path"], receipt_sidecar_ref["path"]):
        absolute = Path(str(path))
        files.append({"path": absolute.relative_to(ROOT).as_posix(), "nbytes": absolute.stat().st_size, "sha256": sha256_file(absolute)})
    files.sort(key=lambda item: str(item["path"]))
    return {
        "algorithm": "sha256(canonical_json([{path,nbytes,sha256}, ...]))",
        "count": len(files),
        "digest": hashlib.sha256(canonical_bytes(files)).hexdigest(),
        "files": files,
    }


def _assert_tree_clean() -> None:
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            raise PreflightError(f"cache remains: {relative}")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise PreflightError(f"symlink remains: {relative}")
        if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
            raise PreflightError(f"special file remains: {relative}")


def _verify_existing_receipt() -> tuple[dict[str, object], dict[str, object]]:
    strict_regular(RECEIPT_PATH)
    sidecar_path = RECEIPT_PATH.with_name(RECEIPT_PATH.name + ".sha256.json")
    strict_regular(sidecar_path)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    expected = _artifact_ref(RECEIPT_PATH)
    if sidecar != expected:
        raise PreflightError("preflight receipt sidecar mismatch")
    if receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("experiment_id") != EXPERIMENT_ID:
        raise PreflightError("preflight receipt identity mismatch")
    return expected, _artifact_ref(sidecar_path)


def publish_stop(test_results: list[dict[str, object]]) -> dict[str, object]:
    if STOP_PATH.exists() or STOP_PATH.is_symlink():
        raise PreflightError("STOP marker already exists")
    receipt_ref, receipt_sidecar_ref = _verify_existing_receipt()
    inventory = implementation_inventory()
    closure = _closure(inventory, receipt_ref, receipt_sidecar_ref)
    stop = {
        "schema_version": 1,
        "status": "STOP_FROZEN",
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "scope": "CPU-only prospective dataset-blind shrinkage router preregistration",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _artifact_ref(CONFIG_PATH),
        "preflight_receipt": receipt_ref,
        "preflight_sidecar": receipt_sidecar_ref,
        "implementation_inventory": inventory,
        "closure": closure,
        "tests": test_results,
        "integrity": {
            "gpu_used": False,
            "model_runtime_imported": False,
            "quality_or_reference_inputs_read": False,
            "fresh_registry_opened": False,
            "pycache_files_at_freeze": 0,
            "symlinks_at_freeze": 0,
        },
        "claim_boundary": "posthoc old03/attempt28 hypothesis only; fresh sealed registry required; no paper-superiority claim",
        "next_allowed_action": "independent registry audit, then a new fresh-registry prospective run",
    }
    payload = canonical_bytes(stop)
    atomic_write(STOP_PATH, payload)
    sidecar_path = STOP_PATH.with_name(STOP_PATH.name + ".sha256.json")
    reference = {"path": str(STOP_PATH), "nbytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    atomic_write(sidecar_path, canonical_bytes(reference))
    return reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true", help="run checks and write the one-way STOP marker")
    args = parser.parse_args()
    preflight = run_preflight()
    test_results = [
        _run_cpu_command([str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], "pytest"),
        _run_cpu_command([str(PYTHON), "-m", "compileall", "-q", "src", "scripts"], "compileall"),
    ]
    _assert_tree_clean()
    if args.publish:
        # Publish the preflight receipt immediately before STOP; no source file
        # is changed after this point.  The receipt is generated from the same
        # checks and is bound by STOP plus its sidecar.
        from scripts.preflight import publish_receipt

        if RECEIPT_PATH.exists() or RECEIPT_PATH.is_symlink():
            raise PreflightError("preflight receipt already exists; use a fresh attempt")
        receipt_ref = publish_receipt(preflight)
        # Sidecar creation is part of publish_receipt; verify it before STOP.
        _verify_existing_receipt()
        reference = publish_stop(test_results)
        print(json.dumps({"status": "STOP_PUBLISHED", "stop": reference}, sort_keys=True))
    else:
        print(json.dumps({"status": "CPU_CHECKS_PASSED", "preflight": preflight, "tests": test_results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

