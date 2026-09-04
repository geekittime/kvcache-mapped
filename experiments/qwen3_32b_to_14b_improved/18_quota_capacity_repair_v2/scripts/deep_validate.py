#!/usr/bin/env python3
"""Run the independent deep validator (never starts GPU/formal work)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _require_preimport_bootstrap() -> None:
    """Enforce a fresh no-bytecode interpreter before local imports."""

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
    claim_artifact,
    file_spec,
    finalize_publication,
    formal_config_path,
    validate_runtime_binding,
    verify_config_file_binding,
)
from src.deep import validate_ready_stages  # noqa: E402
from src.inputs import load_config, load_input_bundle  # noqa: E402
from src.preflight import validate_config_shape, validate_formal_config, validate_preformal_stop_binding  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-nbytes", required=True, type=int)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    try:
        config_path = formal_config_path(ROOT, args.config)
        config = load_config(config_path)
        validate_config_shape(config)
        config_binding = verify_config_file_binding(
            config_path,
            {"path": str(config_path), "nbytes": args.config_nbytes, "sha256": args.config_sha256},
        )
        if os.environ.get("FORMAL_CONFIG_PATH") != config_binding["path"] or os.environ.get("FORMAL_CONFIG_NBYTES") != str(config_binding["nbytes"]) or os.environ.get("FORMAL_CONFIG_SHA256") != config_binding["sha256"]:
            raise ProtocolError("deep worker environment/config binding mismatch")
        runtime_identity = validate_runtime_binding(config["runtime_binding"])
        validate_formal_config(config_path, ROOT, config)
        gate = validate_preformal_stop_binding(config_path, ROOT, config, require_current_inventory=True)
        marker_binding = gate["marker"]
        if os.environ.get("PREFORMAL_STOP_PATH") != marker_binding["path"] or os.environ.get("PREFORMAL_STOP_NBYTES") != str(marker_binding["nbytes"]) or os.environ.get("PREFORMAL_STOP_SHA256") != marker_binding["sha256"] or os.environ.get("PREFORMAL_INVENTORY_SHA256") != marker_binding["implementation_inventory_sha256"]:
            raise ProtocolError("deep worker PRE-FORMAL snapshot environment binding mismatch")
        bundle = load_input_bundle(config, require_deep_allocation=True)
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            raise ProtocolError("deep validator refuses visible GPU devices")
        output_dir = ROOT / config["outputs"]["deep_validation"]
        identity = {"protocol_id": "quota_capacity_repair_v2", "artifact_kind": "deep_validation"}
        claim_artifact(output_dir, identity)
        result = validate_ready_stages(bundle, config, ROOT, config_path=config_path, config_binding=config_binding)
        verify_config_file_binding(config_path, config_binding)
        receipt = {
            "schema_version": 1,
            **identity,
            "status": "pass_repaired_quota_and_allocation",
            "cpu_only": True,
            "completed": True,
            "validation": result,
            "formal_config": config_binding,
            "runtime_identity": runtime_identity,
            "claim_boundary": "independent CPU deep validation only; no model outcomes or locked metrics evaluated",
        }
        atomic_json(output_dir / "receipt.json", receipt)
        marker = finalize_publication(output_dir, ["receipt.json"], identity=identity, status="published_complete_deep_validation")
        print({"status": receipt["status"], "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"), "ordered_inventory_sha256": marker["ordered_inventory_sha256"]})
        return 0
    except (ProtocolError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
