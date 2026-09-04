#!/usr/bin/env python3
"""Prepare the frozen train-only A1 screen; never load a model or use CUDA."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


# Hide accelerators before importing Torch through the A1 contracts.  This
# script only tokenizes with local tokenizer files.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oracle_diagnostics import DEFAULT_OUTPUT, prepare  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = prepare(ROOT, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

