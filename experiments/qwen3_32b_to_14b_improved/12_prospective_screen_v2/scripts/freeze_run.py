#!/usr/bin/env python3
"""Freeze a content-addressed run definition; does not execute models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

_FORMAL_PYCACHE_PREFIX = "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/12_prospective_screen_v2/.formal_empty_pycache"


def _require_formal_bytecode_policy() -> None:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise SystemExit("formal Python bytecode policy violation: set PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONPYCACHEPREFIX") != _FORMAL_PYCACHE_PREFIX:
        raise SystemExit("formal Python bytecode policy violation: set the exact PYTHONPYCACHEPREFIX")
    if (
        sys.dont_write_bytecode is not True
        or type(sys.pycache_prefix) is not str
        or sys.pycache_prefix != _FORMAL_PYCACHE_PREFIX
    ):
        raise SystemExit("formal Python bytecode policy violation: interpreter flags differ")
    prefix_path = Path(_FORMAL_PYCACHE_PREFIX)
    if any(component.is_symlink() for component in (prefix_path, *prefix_path.parents)):
        raise SystemExit("formal Python bytecode policy violation: prefix contains a symlink")
    if os.path.lexists(_FORMAL_PYCACHE_PREFIX):
        raise SystemExit("formal Python bytecode policy violation: pycache prefix must be absent")


_require_formal_bytecode_policy()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.contracts import artifact_ref
from src.run_protocol import freeze_run_definition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True, choices=("smoke", "screen", "confirm"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--screen-validation-receipt")
    args = parser.parse_args()
    screen_ref = (
        artifact_ref(args.screen_validation_receipt).to_dict()
        if args.screen_validation_receipt
        else None
    )
    value = freeze_run_definition(
        args.config,
        profile=args.profile,
        output_dir=args.output,
        screen_validation_receipt=screen_ref,
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
