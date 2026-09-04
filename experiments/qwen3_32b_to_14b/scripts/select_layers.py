#!/usr/bin/env python3
"""Run formal top-20 selection for Qwen3-32B → Qwen3-14B."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.protocol import DEFAULT_CONFIG  # noqa: E402
from reverse_kv.selection_adapter import run_selection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--only-target-layers")
    parser.add_argument(
        "--ols-fallback", choices=("error", "cpu-fp64-lstsq"), default="error"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-sidecar-checksums", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_selection(
        devices=args.devices,
        chunk_tokens=args.chunk_tokens,
        only_target_layers=args.only_target_layers,
        ols_fallback=args.ols_fallback,
        overwrite=args.overwrite,
        trust_sidecar_checksums=args.trust_sidecar_checksums,
        verify_only=args.verify_only,
        no_ledger=args.no_ledger,
        config_path=args.config,
    )


if __name__ == "__main__":
    raise SystemExit(main())
