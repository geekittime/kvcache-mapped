#!/usr/bin/env python3
"""Publish the PRE-FORMAL STOP and independent placeholder markers."""

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
from src.preflight import preflight  # noqa: E402
from src.runner import publish_placeholder_stages  # noqa: E402


def _inventory() -> list[dict[str, object]]:
    excluded = {"__pycache__", ".pytest_cache", ".mypy_cache", "artifacts"}
    rows: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"exact implementation inventory forbids symlink: {path}")
        if any(part in excluded for part in path.parts):
            continue
        # ``Path.is_file`` follows links.  An inventory that silently hashes
        # through a symlink would make the PRE-FORMAL tree non-reproducible
        # and would also accept dangling/foreign files.  Every non-excluded
        # component is therefore checked with lstat before it is considered.
        if not stat.S_ISREG(mode):
            continue
        relative = path.relative_to(ROOT)
        rows.append({"path": str(relative), "nbytes": os.lstat(path).st_size, "sha256": sha256_file(path)})
    return rows


def _inventory_digest(inventory: list[dict[str, object]]) -> str:
    payload = json.dumps(inventory, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--version", default="v2")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        print("BLOCKED: set CUDA_VISIBLE_DEVICES='' for PRE-FORMAL run", file=sys.stderr)
        return 2
    stop_path = ROOT / "artifacts" / "preformal" / f"PRE_FORMAL_STOP_{args.version}.json"
    sidecar_path = Path(str(stop_path) + ".sha256.json")
    if stop_path.exists() or sidecar_path.exists():
        print(f"BLOCKED: refusing to overwrite {stop_path}", file=sys.stderr)
        return 2
    try:
        config = strict_json_load(args.config)
        pf = preflight(args.config)
        if pf["status"] != "PRE_FORMAL_PLACEHOLDER_GATE":
            # This command is intentionally a STOP publisher.  A ready gate
            # requires a separately audited explicit CPU-repair invocation.
            raise ProtocolError("run_preformal only accepts the current placeholder gate")
        stages = publish_placeholder_stages(ROOT, config, pf)
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
            "kind": "quota_capacity_repair_pre_formal_stop_v1",
            "status": "PRE_FORMAL_STOP",
            "published_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_root": str(ROOT),
            "config": file_spec(args.config),
            "preflight": pf,
            "independent_stage_markers": stages,
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
                "allocation_v3_deep_receipt_placeholder": True,
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
                "stop_reason": "allocation_v3_deep_receipt_not_yet_published",
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
                "allocation_v3_deep_receipt",
                "CPU phases A/B/C/D after deep gate",
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
