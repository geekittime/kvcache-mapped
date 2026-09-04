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

from src.common import (  # noqa: E402
    ProtocolError,
    atomic_json,
    file_spec,
    formal_config_path,
    implementation_inventory,
    implementation_inventory_digest,
    sha256_file,
    strict_json_load,
    validate_runtime_binding,
    verify_config_file_binding,
)
from src.preflight import preflight, validate_config_shape, validate_formal_config, validate_preformal_stop_binding  # noqa: E402
from src.runner import validate_stage  # noqa: E402


def _inventory() -> list[dict[str, object]]:
    """Hash the implementation tree while rejecting links and caches."""
    return implementation_inventory(ROOT)


def _inventory_digest(inventory: list[dict[str, object]]) -> str:
    """Digest the canonical inventory list (without a trailing newline)."""
    return implementation_inventory_digest(inventory)


def _strip_config_bindings(value):
    """Remove mutable config-content echoes from a marker payload.

    The active config binds the marker by hash.  Retaining a nested
    ``formal_config`` object inside the marker would create a circular
    self-reference and would become stale when the config is activated after
    the append-only marker is written.  The config itself remains separately
    content-bound and is revalidated by every consumer.
    """

    if isinstance(value, dict):
        return {key: _strip_config_bindings(item) for key, item in value.items() if key != "formal_config"}
    if isinstance(value, list):
        return [_strip_config_bindings(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--version", default="v16")
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
        config_path = formal_config_path(ROOT, args.config)
        config = strict_json_load(config_path)
        validate_config_shape(config)
        config_binding = validate_formal_config(config_path, ROOT, config)
        previous_gate = validate_preformal_stop_binding(config_path, ROOT, config, require_current_inventory=False)
        runtime_identity = validate_runtime_binding(config["runtime_binding"])
        pf = preflight(config_path, require_deep_allocation=True)
        if pf["status"] != "PRE_FORMAL_READY_FOR_CPU_REPAIR" or not pf["allocation_v3_deep_receipt"].get("ready"):
            raise ProtocolError("ready-gate snapshot requires a published allocation_v3 deep receipt")

        # This sibling has no writable copy of v1's historical stage outputs.
        # If local placeholders exist, validate them; otherwise record an
        # explicit empty historical set rather than importing/reusing v1.
        historical_markers = {}
        for stage in ("quota_search", "allocation", "deep_validation"):
            stage_dir = ROOT / config["outputs"][stage]
            if stage_dir.exists():
                marker = validate_stage(stage, ROOT, config)
                historical_markers[stage] = {
                    "PUBLISHED": file_spec(stage_dir / "PUBLISHED.json"),
                    "ordered_inventory_sha256": marker["ordered_inventory_sha256"],
                    "status": marker["status"],
                    "historical_placeholder_only": True,
                }

        prefix = os.environ.get("PYTHONPYCACHEPREFIX")
        if not isinstance(prefix, str) or not prefix or not os.path.isabs(prefix) or os.path.lexists(prefix):
            raise ProtocolError("fresh PYTHONPYCACHEPREFIX is required for the isolated CPU regression")
        inventory_start = _inventory()
        inventory_start_digest = _inventory_digest(inventory_start)
        test_hash_seed = os.environ.get("PYTHONHASHSEED", "1")
        if test_hash_seed not in {"1", "777"}:
            raise ProtocolError("CPU snapshot PYTHONHASHSEED must be 1 or 777")
        test_env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": prefix,
            "PYTHONHASHSEED": test_hash_seed,
            **dict(config["solver"]["required_environment"]),
        }
        test_command = [str(config["runtime_binding"]["python_executable"]), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
        test_run = subprocess.run(test_command, cwd=ROOT, env=test_env, text=True, capture_output=True, check=False)
        test_output = (test_run.stdout + ("\n" + test_run.stderr if test_run.stderr else "")).strip()
        if test_run.returncode != 0:
            raise ProtocolError(f"CPU tests failed ({test_run.returncode}): {test_output}")

        inventory = _inventory()
        inventory_digest = _inventory_digest(inventory)
        if inventory != inventory_start or inventory_digest != inventory_start_digest:
            raise ProtocolError("implementation inventory drifted during CPU snapshot tests")
        receipt = {
            "schema_version": 1,
            "kind": "quota_capacity_repair_pre_formal_stop_v2",
            "status": "PRE_FORMAL_STOP",
            "stop_mode": "READY_GATE_NO_ABCD_EXECUTION",
            "published_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_root": str(ROOT),
            # The config binds this marker by content after publication.  A
            # publication-time config hash would create a self-reference, so
            # the marker records only its canonical path and the consumer
            # verifies the current config binding independently.
            "config_path": str(config_path),
            "preformal_stop_binding_previous": previous_gate["marker"],
            "preflight": _strip_config_bindings(pf),
            "external_gate": {
                "name": "allocation_v3_deep_receipt",
                "status": "READY",
                "PUBLISHED": pf["allocation_v3_deep_receipt"]["PUBLISHED"],
                "receipt": pf["allocation_v3_deep_receipt"]["receipt"],
                "receipt_sidecar": pf["allocation_v3_deep_receipt"]["receipt_sidecar"],
            },
            "independent_stage_markers": _strip_config_bindings(historical_markers),
            "cpu_tests": {
                "command": test_command,
                "returncode": test_run.returncode,
                "output": test_output,
                "gpu_execution_performed": False,
                "models_loaded": False,
                "environment": {
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": prefix,
                    "PYTHONHASHSEED": test_hash_seed,
                    **dict(config["solver"]["required_environment"]),
                },
                "bootstrap": {
                    "prefix_initially_absent": True,
                    "prefix_parent_policy": "all_existing_parents_real_directories_no_symlinks",
                    "sys_dont_write_bytecode": True,
                },
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
                "executable": runtime_identity["python_executable"],
                "prefix": runtime_identity["python_prefix"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
            "implementation_inventory": inventory,
            "implementation_inventory_sha256": inventory_digest,
            "implementation_inventory_start_sha256": inventory_start_digest,
            "implementation_inventory_end_sha256": inventory_digest,
            "implementation_inventory_start_equals_end": True,
            "superseded_upstream": config["superseded_upstream"],
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
