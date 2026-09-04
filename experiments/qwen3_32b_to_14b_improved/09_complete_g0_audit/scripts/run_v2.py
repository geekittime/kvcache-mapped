#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from g0_audit_v2 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_v2.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "complete_g0_v2")
    args = parser.parse_args()
    receipt = run(args.config.resolve(), args.output_dir.resolve())
    print(json.dumps({
        "pass": receipt["result"]["pass"],
        "total_isolation_flag_count": receipt["result"]["total_isolation_flag_count"],
        "closes_complete_g0": receipt["closes_complete_g0"],
    }, sort_keys=True))
    if not receipt["result"]["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
