#!/usr/bin/env python3
"""Run dir21 CPU preflight or explicitly authorized GPU evaluation."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
