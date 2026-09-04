#!/usr/bin/env python3
from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        help="relative, previously nonexistent artifact directory for a deterministic mirror run",
    )
    args = parser.parse_args()
    result = run(ROOT / "config.json", ROOT, artifact_dir_override=args.artifact_dir)
    print(json.dumps({
        "status": result["status"],
        "selection_counts": result["selection_counts"],
        "screen_confirm_disjoint": result["screen_confirm_disjoint"],
        "manifest": result["manifest"],
    }, sort_keys=True))
