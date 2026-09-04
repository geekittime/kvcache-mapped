#!/usr/bin/env python3
"""Validate endpoint/formal latency runs and write the paper-main report."""

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

from kvcache_transfer.latency_report import (  # noqa: E402
    LatencyReportError,
    build_latency_report,
    write_latency_report_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "endpoint_run_dir",
        type=Path,
        help="complete schema-3 64/32768 endpoint-gate run directory",
    )
    parser.add_argument(
        "formal_run_dir",
        type=Path,
        help="complete schema-3 ten-length paper-main run directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_latency_report(args.endpoint_run_dir, args.formal_run_dir)
        manifest = write_latency_report_outputs(report, args.output_dir)
    except (LatencyReportError, OSError) as exc:
        print(f"latency summary refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(args.output_dir.resolve()),
                "formal_lengths": len(report["rows"]),
                "minimum_peak_reserved_headroom_bytes": report[
                    "minimum_peak_reserved_headroom_bytes_across_formal_lengths"
                ],
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
