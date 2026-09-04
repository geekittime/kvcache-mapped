#!/usr/bin/env python3
"""Fit the formal top-20 Qwen3-32B -> Qwen3-14B KV mapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.fit_adapter import run_fit  # noqa: E402
from reverse_kv.protocol import DEFAULT_CONFIG  # noqa: E402


def parse_args() -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    known, remaining = parser.parse_known_args()
    return known.config, remaining


def main() -> int:
    config, remaining = parse_args()
    return run_fit(remaining, config_path=config)


if __name__ == "__main__":
    raise SystemExit(main())
