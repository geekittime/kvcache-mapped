#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validate_allocation import validate_allocation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(validate_allocation(
        ROOT / "config.json", ROOT, deep=arguments.deep
    ), sort_keys=True))
