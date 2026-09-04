#!/usr/bin/env python3
"""Run the sealed ten-length Qwen3 32B -> 14B latency protocol."""

from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.reverse_latency import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
