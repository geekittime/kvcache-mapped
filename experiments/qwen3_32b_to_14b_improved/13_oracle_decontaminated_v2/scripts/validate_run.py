#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validator import validate_run


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only oracle receipt validator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=("smoke", "diagnostic"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.config, profile=args.profile, output_dir=args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

