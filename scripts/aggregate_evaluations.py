#!/usr/bin/env python3
"""Strictly aggregate split formal evaluation runs into the paper table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.evaluation_aggregate import (  # noqa: E402
    AggregationError,
    aggregate_runs,
    write_aggregate_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        help="evaluation run directories containing run_definition/status/summary/results",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-sampled",
        action="store_true",
        help="allow a clearly marked non-formal aggregate of sample-limited runs",
    )
    parser.add_argument(
        "--require-oracle",
        action="store_true",
        help="require a complete target_oracle column in addition to the three paper modes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = aggregate_runs(
            args.run_dirs,
            allow_sampled=args.allow_sampled,
            require_oracle=args.require_oracle,
        )
        manifest = write_aggregate_outputs(result, args.output_dir)
    except AggregationError as exc:
        print(f"aggregation refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.resolve()),
                "formal_full_dataset": result["formal_full_dataset"],
                "compatibility_identity_sha256": result[
                    "compatibility_identity_sha256"
                ],
                "five_task_retention_percent": result[
                    "five_accuracy_task_retention"
                ]["reproduced_arithmetic_mean_percent"],
                "five_task_floor_normalized_retention_percent": result[
                    "five_accuracy_task_floor_normalized_retention"
                ]["reproduced_arithmetic_mean_percent"],
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
