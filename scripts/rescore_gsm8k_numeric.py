#!/usr/bin/env python3
"""Create a gold-independent numeric-final diagnostic for a GSM8K checkpoint."""

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

from kvcache_transfer.gsm8k_numeric_diagnostic import (  # noqa: E402
    GSM8KNumericDiagnosticError,
    score_checkpoint,
    write_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="completed results/<mode>/gsm8k_cot.json task checkpoint",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace known report files when existing content differs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = score_checkpoint(args.checkpoint)
        manifest = write_report(report, args.output_dir, overwrite=args.overwrite)
    except GSM8KNumericDiagnosticError as exc:
        print(f"GSM8K numeric diagnostic refused: {exc}", file=sys.stderr)
        return 2
    value = report["numeric_final_diagnostic"]
    print(
        json.dumps(
            {
                "status": "complete",
                "metric_role": report["metric_role"],
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": value["accuracy"],
                "output_dir": str(args.output_dir.resolve()),
                "publication_status": manifest["publication_status"],
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
