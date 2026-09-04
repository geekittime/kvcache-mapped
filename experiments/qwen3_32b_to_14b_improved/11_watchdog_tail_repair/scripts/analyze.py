#!/usr/bin/env python3
"""Repository-local entry point for the CPU-only counterfactual analyzer."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analyze_counterfactual import main


if __name__ == "__main__":
    raise SystemExit(main())
