#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.moment_probe import run


if __name__ == "__main__":
    result = run(ROOT / "config.json")
    print(result["decision"])

