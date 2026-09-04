#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from src.common import ContractError  # noqa: E402
from src.preflight import validate_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only functional-residual preflight")
    parser.add_argument("--config", type=Path, default=ROOT / "config.template.json")
    parser.add_argument("--expect-template-blocked", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    try:
        report = validate_config(config_path, expect_template_blocked=args.expect_template_blocked)
    except (ContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
