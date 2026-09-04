#!/usr/bin/env python3
"""CPU preflight and explicitly gated full40 GPU entrypoint for dir32."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_before_local_import() -> None:
    """Reject timestamp pyc/local cache execution before importing ``src``."""

    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise RuntimeError("PYTHONDONTWRITEBYTECODE=1 is required before local import")
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if type(prefix) is not str or not prefix or not os.path.isabs(prefix) or os.path.lexists(prefix):
        raise RuntimeError("PYTHONPYCACHEPREFIX must be an absent absolute leaf before local import")
    if type(sys.pycache_prefix) is not str or sys.pycache_prefix != prefix:
        raise RuntimeError("sys.pycache_prefix does not match PYTHONPYCACHEPREFIX")
    config_path = ROOT / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError("canonical config.json is missing/nonregular")
    try:
        expected_python = json.loads(config_path.read_text(encoding="utf-8"))["runtime"]["required_python"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("cannot read configured Python binding before local import") from exc
    if type(expected_python) is not str or sys.executable != expected_python:
        raise RuntimeError(f"exact configured Python path required before local import: {sys.executable}")
    current = Path(prefix).anchor
    for part in Path(prefix).parts[1:-1]:
        current = str(Path(current) / part)
        mode = os.lstat(current).st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise RuntimeError(f"pycache prefix parent is unsafe: {current}")
    for path in ROOT.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"local tree contains a symlink: {path}")
        if path.name == "__pycache__" or path.suffix == ".pyc":
            raise RuntimeError(f"local bytecode/cache is forbidden: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--stage", choices=("full40",))
    args = parser.parse_args(argv)
    _bootstrap_before_local_import()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.probe import ProtocolError, preflight, run_stage

    try:
        if args.preflight_only:
            if args.stage is not None:
                raise ProtocolError("--stage is forbidden for --preflight-only")
            result = preflight(args.config)
            print(json.dumps({
                "status": "PRE_FORMAL_STOP",
                "preflight": result["preflight_ref"],
                "stop": result["stop_ref"],
                "request_fingerprint": result["definition"]["request_fingerprint"],
            }, ensure_ascii=False, sort_keys=True))
            return 0
        if args.stage is None:
            raise ProtocolError("--execute requires --stage full40")
        result = run_stage(args.config, args.stage, execute=True)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except ProtocolError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
