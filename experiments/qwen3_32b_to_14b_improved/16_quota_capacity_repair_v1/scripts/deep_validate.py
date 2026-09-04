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

from src.common import ProtocolError, atomic_json, claim_artifact, file_spec, finalize_publication  # noqa: E402
from src.deep import validate_ready_stages  # noqa: E402
from src.inputs import load_config, load_input_bundle  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        bundle = load_input_bundle(config, require_deep_allocation=True)
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            raise ProtocolError("deep validator refuses visible GPU devices")
        output_dir = ROOT / config["outputs"]["deep_validation"]
        identity = {"protocol_id": "quota_capacity_repair_v1", "artifact_kind": "deep_validation"}
        claim_artifact(output_dir, identity)
        result = validate_ready_stages(bundle, config, ROOT)
        receipt = {
            "schema_version": 1,
            **identity,
            "status": "pass_repaired_quota_and_allocation",
            "cpu_only": True,
            "completed": True,
            "validation": result,
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
