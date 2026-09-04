#!/usr/bin/env python3
"""Fit the independent 4096 -> 1024 reverse mapper for Figure 4 k=4."""

from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.k4_fit import run_k4_fit  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(run_k4_fit(sys.argv[1:]))
