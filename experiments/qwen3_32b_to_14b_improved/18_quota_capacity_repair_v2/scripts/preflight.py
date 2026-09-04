#!/usr/bin/env python3
"""Validate the answer-free PRE-FORMAL input gate."""

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

from src.common import ProtocolError  # noqa: E402
from src.preflight import preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--expect-placeholder", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight(args.config)
        if args.expect_placeholder and result["status"] != "PRE_FORMAL_PLACEHOLDER_GATE":
            raise ProtocolError("expected allocation-v3 deep placeholder gate, but gate is ready")
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (ProtocolError, OSError, ValueError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
