#!/usr/bin/env python3
"""One deterministic quota-search replica (called only after the deep gate)."""

from __future__ import annotations

import argparse
import json
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
    atomic_write,
    canonical_bytes,
    formal_config_path,
    validate_runtime_binding,
    verify_config_file_binding,
)
from src.inputs import load_config, load_input_bundle  # noqa: E402
from src.preflight import validate_config_shape, validate_formal_config, validate_preformal_stop_binding  # noqa: E402
from src.quota import solve_quota_search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-nbytes", required=True, type=int)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        seed_text = os.environ.get("QUOTA_REPAIR_SEED")
        if seed_text not in {"1", "777"}:
            raise ProtocolError("QUOTA_REPAIR_SEED must be 1 or 777")
        config_path = formal_config_path(ROOT, args.config)
        config = load_config(config_path)
        validate_config_shape(config)
        config_binding = verify_config_file_binding(
            config_path,
            {"path": str(config_path), "nbytes": args.config_nbytes, "sha256": args.config_sha256},
        )
        if os.environ.get("FORMAL_CONFIG_PATH") != config_binding["path"] or os.environ.get("FORMAL_CONFIG_NBYTES") != str(config_binding["nbytes"]) or os.environ.get("FORMAL_CONFIG_SHA256") != config_binding["sha256"]:
            raise ProtocolError("quota worker environment/config binding mismatch")
        runtime_identity = validate_runtime_binding(config["runtime_binding"])
        validate_formal_config(config_path, ROOT, config)
        gate = validate_preformal_stop_binding(config_path, ROOT, config, require_current_inventory=True)
        marker = gate["marker"]
        if os.environ.get("PREFORMAL_STOP_PATH") != marker["path"] or os.environ.get("PREFORMAL_STOP_NBYTES") != str(marker["nbytes"]) or os.environ.get("PREFORMAL_STOP_SHA256") != marker["sha256"] or os.environ.get("PREFORMAL_INVENTORY_SHA256") != marker["implementation_inventory_sha256"]:
            raise ProtocolError("quota worker PRE-FORMAL snapshot environment binding mismatch")
        bundle = load_input_bundle(config, require_deep_allocation=True)
        quota_vector, proof = solve_quota_search(
            bundle.records,
            bundle.hyperedges,
            bundle.contract,
            config,
            seed=int(seed_text),
            require_environment=True,
        )
        verify_config_file_binding(config_path, config_binding)
        result = {"quota_vector": quota_vector, "proof": proof}
        atomic_write(args.output, canonical_bytes(result))
        envelope = {
            "process_id": os.getpid(),
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "seed": int(seed_text),
            "result": result,
            "config_binding": config_binding,
            "runtime_identity": runtime_identity,
            "preformal_stop_binding": marker,
        }
        sys.stdout.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except (ProtocolError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
