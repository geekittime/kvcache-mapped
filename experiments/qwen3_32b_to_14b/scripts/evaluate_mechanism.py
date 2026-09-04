#!/usr/bin/env python3
"""Run the sealed Qwen3 32B -> 14B HellaSwag mechanism diagnostics."""

from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.mechanism import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
