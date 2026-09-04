from __future__ import annotations

from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
for value in (EXPERIMENT_ROOT, REPOSITORY_ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))
