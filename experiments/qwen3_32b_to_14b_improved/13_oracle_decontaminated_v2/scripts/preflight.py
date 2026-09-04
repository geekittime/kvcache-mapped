#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import ContractError
from src.gpu_runner import preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only final-gate preflight")
    parser.add_argument("--config", default=str(ROOT / "config.template.json"))
    parser.add_argument("--profile", choices=("smoke", "diagnostic"), required=True)
    args = parser.parse_args()
    try:
        result = preflight(args.config, profile=args.profile)
    except ContractError as exc:
        print(json.dumps({
            "schema_version": 1,
            "status": "blocked_preformal",
            "gpu_used": False,
            "cuda_api_called": False,
            "reason": str(exc),
        }, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

