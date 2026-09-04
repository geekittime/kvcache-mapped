#!/usr/bin/env python3
"""Prepare the pinned MuSiQue registry without evaluating model outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.musique_benchmark import prepare_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "registry_v1")
    args = parser.parse_args()
    print(prepare_registry(ROOT, args.output_dir))


if __name__ == "__main__":
    main()
