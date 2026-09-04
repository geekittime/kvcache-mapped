#!/usr/bin/env python3
"""Prepare or run the deterministic CoQA reverse multi-turn protocol."""

from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.coqa_eval import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
