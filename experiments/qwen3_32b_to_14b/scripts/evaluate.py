#!/usr/bin/env python3
"""Evaluate 32B source, 14B target, and top-20 mapped 14B modes."""

from __future__ import annotations

from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.eval_adapter import run_evaluation  # noqa: E402


def main() -> int:
    # The shared evaluator owns the stable CLI.  This isolated entry point
    # fixes the reverse config and forwards every other argument verbatim.
    return run_evaluation(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
