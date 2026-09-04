#!/usr/bin/env python3
"""Run or validate the mandatory k4 held-out reconstruction acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.k4_heldout import (  # noqa: E402
    evaluate_k4_heldout,
    k4_heldout_preflight,
    validate_k4_heldout_artifact,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-device", default="cuda:0")
    parser.add_argument("--analysis-device", default="cpu")
    parser.add_argument("--chunk-samples", type=int, default=1024)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.preflight_only and args.validate_only:
        raise ValueError("choose at most one k4 held-out inspection mode")
    if args.preflight_only:
        result = k4_heldout_preflight(verify_payloads=True)
    elif args.validate_only:
        result = validate_k4_heldout_artifact()
    else:
        result = evaluate_k4_heldout(
            compute_device=args.compute_device,
            analysis_device=args.analysis_device,
            chunk_samples=args.chunk_samples,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
