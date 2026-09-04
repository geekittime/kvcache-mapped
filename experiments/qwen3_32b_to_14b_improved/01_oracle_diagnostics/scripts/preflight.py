#!/usr/bin/env python3
"""Deep CPU-only validator for prepared A1 inputs and the read-only k20 mapper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_diagnostics import DEFAULT_OUTPUT, deep_preflight  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--manifest-only-mapper",
        action="store_true",
        help="skip 40 layer payload hashes (unit/smoke only; not a final preflight)",
    )
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    receipt = deep_preflight(
        ROOT,
        args.output,
        deep_mapper_hashes=not args.manifest_only_mapper,
        publish_receipt=not args.no_publish,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

