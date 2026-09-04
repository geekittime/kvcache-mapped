#!/usr/bin/env python3
"""Independently recompute a sealed run and write a validation receipt."""

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

from src.configuration import load_config
from src.validator import write_validation_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", required=True, choices=("smoke", "screen", "confirm"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config, verify_inputs=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["runtime"]["target_model"], local_files_only=True, use_fast=True
    )
    value = write_validation_receipt(
        args.config, profile=args.profile, output_dir=args.output, tokenizer=tokenizer
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
