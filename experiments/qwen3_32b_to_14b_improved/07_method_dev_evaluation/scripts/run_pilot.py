#!/usr/bin/env python3
"""Validate or execute the five-mode method-dev smoke/screen/confirm run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import DEFAULT_CONFIG, PROFILES, RUNS_ROOT  # noqa: E402
from src.runner import contract_preflight, execute  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--screen-output-dir",
        type=Path,
        help="completed screen output required only for guarded confirm execution",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--contract-only",
        action="store_true",
        help="CPU-only validation; no output files and no CUDA API calls",
    )
    action.add_argument(
        "--execute",
        action="store_true",
        help="run after content checks, global lock, and GPU idle gate",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.contract_only:
        result, _definition, _rows, _bindings = contract_preflight(
            args.config, profile=args.profile, deep_model_hash=False
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.output_dir is None:
        raise SystemExit("--output-dir is required with --execute")
    return execute(
        args.config,
        profile=args.profile,
        output_dir=args.output_dir,
        screen_output_dir=args.screen_output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
