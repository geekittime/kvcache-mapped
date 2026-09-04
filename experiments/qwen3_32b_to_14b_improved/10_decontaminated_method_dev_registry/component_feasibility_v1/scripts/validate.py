#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validate import validate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true", help="rebuild clean pool and components from pinned sources")
    arguments = parser.parse_args()
    print(json.dumps(validate(ROOT / "config.json", ROOT, deep=arguments.deep), sort_keys=True))
