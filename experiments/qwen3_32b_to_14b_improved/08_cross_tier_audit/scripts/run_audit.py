#!/usr/bin/env python3
"""Repository-local entry point for the cross-tier audit."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cross_tier_audit import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
