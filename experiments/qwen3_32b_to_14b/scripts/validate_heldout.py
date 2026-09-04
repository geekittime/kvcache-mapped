#!/usr/bin/env python3
"""Validate reverse mapper reconstruction on the frozen held-out activations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.heldout import evaluate_heldout, heldout_preflight  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-device", default="cuda:0")
    parser.add_argument("--analysis-device", default="cpu")
    parser.add_argument("--chunk-samples", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--verify-payloads",
        action="store_true",
        help="retained for CLI compatibility; formal validation always verifies every payload",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.preflight_only:
        result = heldout_preflight(verify_payloads=True)
    else:
        result = evaluate_heldout(
            compute_device=args.compute_device,
            analysis_device=args.analysis_device,
            chunk_samples=args.chunk_samples,
            output=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
