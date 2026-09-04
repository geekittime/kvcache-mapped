#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gpu_runner import execute


def main() -> int:
    parser = argparse.ArgumentParser(description="Concrete decontaminated causal-oracle runner")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", choices=("smoke", "diagnostic"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("GPU execution requires the explicit --execute flag")
    return execute(args.config, profile=args.profile, output_dir=args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())

