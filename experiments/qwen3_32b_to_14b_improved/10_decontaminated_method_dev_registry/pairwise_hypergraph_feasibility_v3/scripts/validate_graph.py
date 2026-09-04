#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.validate_graph import validate_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--write-receipt", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(validate_graph(
        ROOT / "config.json", ROOT,
        deep=arguments.deep,
        write_receipt=arguments.write_receipt,
    ), sort_keys=True))
