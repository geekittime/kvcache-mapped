#!/usr/bin/env python3
"""Publish an append-only PRE-FORMAL snapshot after the external deep gate.

This command deliberately does *not* run quota search or allocation.  It
records that the allocation-v3 deep receipt is now complete, reruns the
answer-free CPU protocol tests, and stops at the A/B/C/D boundary.  The old
placeholder stage markers remain immutable historical evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _require_preimport_bootstrap() -> None:
    """Reject ambiguous interpreter caches before importing local ``src``."""

    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise SystemExit("BLOCKED: PYTHONDONTWRITEBYTECODE must be exactly '1'")
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not isinstance(prefix, str) or not prefix or not os.path.isabs(prefix):
        raise SystemExit("BLOCKED: PYTHONPYCACHEPREFIX must be an absolute fresh absent path")
    if (
        sys.dont_write_bytecode is not True
        or type(sys.pycache_prefix) is not str
        or sys.pycache_prefix != prefix
    ):
        raise SystemExit("BLOCKED: interpreter bytecode flags do not match the requested cache prefix")
    if os.path.lexists(prefix):
        raise SystemExit("BLOCKED: PYTHONPYCACHEPREFIX must be an absolute fresh absent path")
    prefix_path = Path(prefix)
    for component in prefix_path.parents:
        if component.is_symlink() or not component.exists() or not component.is_dir():
            raise SystemExit("BLOCKED: PYTHONPYCACHEPREFIX parent is missing, non-directory, or symlink")
    sys.dont_write_bytecode = True


_require_preimport_bootstrap()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import ProtocolError, atomic_json, file_spec, sha256_file, strict_json_load  # noqa: E402
from src.preflight import preflight, validate_config_shape  # noqa: E402
from src.runner import validate_stage  # noqa: E402


def _inventory() -> list[dict[str, object]]:
    """Hash the implementation tree while rejecting links and caches."""

    excluded = {"__pycache__", ".pytest_cache", ".mypy_cache", "artifacts"}
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"exact implementation inventory forbids symlink: {path}")
        if any(part in excluded for part in path.parts):
            continue
        if not stat.S_ISREG(mode):
            continue
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "nbytes": os.lstat(path).st_size,
            "sha256": sha256_file(path),
        })
    return rows


def _inventory_digest(inventory: list[dict[str, object]]) -> str:
    """Digest the canonical inventory list (without a trailing newline)."""

    payload = json.dumps(inventory, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_ready_v2.json")
    parser.add_argument("--version", default="v4")
    args = parser.parse_args()
    stop_path = ROOT / "artifacts" / "preformal" / f"PRE_FORMAL_STOP_{args.version}.json"
    sidecar_path = Path(str(stop_path) + ".sha256.json")
    if stop_path.exists() or sidecar_path.exists():
        print(f"BLOCKED: refusing to overwrite {stop_path}", file=sys.stderr)
        return 2
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        print("BLOCKED: ready-gate snapshot is CPU-only; hide CUDA devices", file=sys.stderr)
        return 2
    try:
        config = strict_json_load(args.config)
        validate_config_shape(config)
        pf = preflight(args.config, require_deep_allocation=True)
        if pf["status"] != "PRE_FORMAL_READY_FOR_CPU_REPAIR" or not pf["allocation_v3_deep_receipt"].get("ready"):
            raise ProtocolError("ready-gate snapshot requires a published allocation_v3 deep receipt")

        # Existing markers were intentionally published under the earlier
        # missing-gate state.  Validate them as immutable placeholders and
        # label them historical; never reinterpret them as completed stages.
        historical_markers = {}
        for stage in ("quota_search", "allocation", "deep_validation"):
            marker = validate_stage(stage, ROOT, config)
            historical_markers[stage] = {
                "PUBLISHED": file_spec(ROOT / config["outputs"][stage] / "PUBLISHED.json"),
                "ordered_inventory_sha256": marker["ordered_inventory_sha256"],
                "status": marker["status"],
                "historical_placeholder_only": True,
            }

        prefix = os.environ.get("PYTHONPYCACHEPREFIX")
        if not isinstance(prefix, str) or not prefix or not os.path.isabs(prefix) or os.path.lexists(prefix):
            raise ProtocolError("fresh PYTHONPYCACHEPREFIX is required for the isolated CPU regression")
        test_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": prefix,
        }
        test_command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
        test_run = subprocess.run(test_command, cwd=ROOT, env=test_env, text=True, capture_output=True, check=False)
        test_output = (test_run.stdout + ("\n" + test_run.stderr if test_run.stderr else "")).strip()
        if test_run.returncode != 0:
            raise ProtocolError(f"CPU tests failed ({test_run.returncode}): {test_output}")

        inventory = _inventory()
        receipt = {
            "schema_version": 1,
            "kind": "quota_capacity_repair_pre_formal_stop_v2",
            "status": "PRE_FORMAL_STOP",
            "stop_mode": "READY_GATE_NO_ABCD_EXECUTION",
            "published_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_root": str(ROOT),
            "config": file_spec(args.config),
            "preflight": pf,
            "external_gate": {
                "name": "allocation_v3_deep_receipt",
                "status": "READY",
                "PUBLISHED": pf["allocation_v3_deep_receipt"]["PUBLISHED"],
                "receipt": pf["allocation_v3_deep_receipt"]["receipt"],
                "receipt_sidecar": pf["allocation_v3_deep_receipt"]["receipt_sidecar"],
            },
            "independent_stage_markers": historical_markers,
            "cpu_tests": {
                "command": test_command,
                "returncode": test_run.returncode,
                "output": test_output,
                "gpu_execution_performed": False,
                "models_loaded": False,
            },
            "source_consumption": {
                "graph_v3_read_only": True,
                "graph_deep_receipt_bound": True,
                "allocation_v3_infeasibility_bound": True,
                "allocation_v3_deep_receipt_placeholder": False,
                "allocation_v3_deep_receipt_ready": True,
                "dir10_v3_modified": False,
                "answers_read": False,
                "model_outcomes_read": False,
                "collision_hyperedges_recomputed_or_changed": False,
                "thresholds_reused_byte_for_byte": True,
            },
            "formal_execution": {
                "started": False,
                "gpu_started": False,
                "claim_permitted": False,
                "stop_reason": "external_deep_gate_ready; A/B/C/D intentionally deferred to a separately audited CPU invocation",
            },
            "method_summary": {
                "phase_A": "exact finite-breakpoint max-min lambda with adjacent infeasibility certificate",
                "phase_B": "fixed rational lambda floor, maximize total selected",
                "phase_C": "lexicographic quota maximization in frozen public target order",
                "phase_D": "existing SHA-cost objective with two seeds/PIDs, separate from quota search",
                "upper_bounds": "exactly the 18 pairwise-v3 quotas",
                "screen_confirm_constraint": "abs(confirm - 2*screen) <= 1 per cell",
            },
            "unfinished_boundaries": [
                "CPU phases A/B/C/D (not started in this snapshot)",
                "independent repaired-allocation deep receipt",
                "downstream native-registry adapter resolving sample_id and selection_rank_sha256",
                "all formal/GPU/model-outcome evaluation",
            ],
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "executable": sys.executable,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            "implementation_inventory": inventory,
            "implementation_inventory_sha256": _inventory_digest(inventory),
            "comparative_claim_permitted": False,
        }
        atomic_json(stop_path, receipt)
        atomic_json(sidecar_path, {"path": str(stop_path), "nbytes": stop_path.stat().st_size, "sha256": sha256_file(stop_path)})
        print(json.dumps({"status": receipt["status"], "path": str(stop_path), "sha256": sha256_file(stop_path)}, sort_keys=True))
        return 0
    except (ProtocolError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
