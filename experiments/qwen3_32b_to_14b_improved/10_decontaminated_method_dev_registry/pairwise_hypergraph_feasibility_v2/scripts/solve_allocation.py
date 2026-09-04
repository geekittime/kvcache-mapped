#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.allocation import run_allocation


if __name__ == "__main__":
    print(json.dumps(run_allocation(ROOT / "config.json", ROOT), sort_keys=True))
