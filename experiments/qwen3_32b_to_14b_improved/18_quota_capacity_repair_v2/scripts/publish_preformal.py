#!/usr/bin/env python3
"""Stable entrypoint for the CPU-only PRE-FORMAL STOP publisher."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _require_preimport_bootstrap() -> None:
    """Enforce a fresh no-bytecode interpreter before launching any entrypoint."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="publish quota repair PRE-FORMAL STOP")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--version", default="v16")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        print("BLOCKED: CUDA_VISIBLE_DEVICES must be empty", file=sys.stderr)
        return 2
    try:
        raw = json.loads(args.config.read_text(encoding="utf-8"))
        executable = raw["runtime_binding"]["python_executable"]
        if not isinstance(executable, str) or not os.path.isabs(executable):
            raise ValueError("invalid runtime executable")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: cannot resolve pinned runtime executable: {exc}", file=sys.stderr)
        return 3
    command = [executable, str(ROOT / "scripts" / "run_preformal.py"), "--config", str(args.config), "--version", args.version]
    return subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
