#!/usr/bin/env python3
"""Explicit CPU preflight/STOP publisher for experiment 33.

No GPU or model import is possible in this script.  A future execution runner
is intentionally not included until the preflight snapshot has been audited.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap() -> None:
    required_text = "/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python"
    required = Path(required_text)
    if sys.executable != required_text or not required.is_file() or Path(sys.executable).resolve() != required.resolve():
        raise SystemExit(f"wrong interpreter: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit("preflight requires CUDA_VISIBLE_DEVICES=''")
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise SystemExit("preflight requires PYTHONDONTWRITEBYTECODE=1")
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not prefix or not Path(prefix).is_absolute() or Path(prefix).resolve().is_relative_to(ROOT):
        raise SystemExit("PYTHONPYCACHEPREFIX must be an external absolute path")
    # The leaf is deliberately supplied as an absent path by the documented
    # command.  Python may create it while importing this script; preserve the
    # parent/leaf identity for the receipt rather than silently accepting an
    # in-tree cache.
    sys.path.insert(0, str(ROOT))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--publish-stop", action="store_true")
    parser.add_argument("--tests", type=int, default=0)
    parser.add_argument("--compile-files", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    _bootstrap()
    from src.blend import ProtocolError, current_python_identity, file_ref
    from src.preflight import ROOT as SOURCE_ROOT, atomic_json, make_stop, strict_json, validate_inputs, write_sidecar

    args = _args()
    config_path = args.config.resolve()
    if config_path != SOURCE_ROOT / "config.json":
        raise ProtocolError("config must be the frozen sibling config.json")
    config = strict_json(config_path)
    preflight = validate_inputs(config)
    pre_path = SOURCE_ROOT / config["outputs"]["preflight_receipt"]
    pre_ref = atomic_json(pre_path, preflight)
    write_sidecar(pre_path)
    print(json.dumps({"event": "preflight", "status": "pass", "path": str(pre_path), "nbytes": pre_ref["nbytes"], "sha256": pre_ref["sha256"], "input_inventory_sha256": preflight["input_inventory_sha256"]}, sort_keys=True))
    if args.publish_stop:
        stop_path = SOURCE_ROOT / config["outputs"]["stop_receipt"]
        if stop_path.exists() or stop_path.is_symlink():
            raise ProtocolError(f"refusing to overwrite existing STOP: {stop_path}")
        stop = make_stop(config_path, pre_path, preflight, test_count=args.tests, compile_count=args.compile_files, runtime_identity=current_python_identity())
        stop_ref = atomic_json(stop_path, stop)
        write_sidecar(stop_path)
        print(json.dumps({"event": "stop", "status": "PRE_FORMAL_STOP", "path": str(stop_path), "nbytes": stop_ref["nbytes"], "sha256": stop_ref["sha256"]}, sort_keys=True))
    elif not args.preflight_only:
        raise ProtocolError("only --preflight-only or --publish-stop is permitted")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Bootstrap/provenance failures are intentionally a stable nonzero
        # result; no partial formal artifact is published.
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
