#!/usr/bin/env python3
"""Verify reverse role binding and publish immutable CPU preflight evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.protocol import (  # noqa: E402
    atomic_write_json,
    protocol_summary,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-payloads",
        action="store_true",
        help="reread and SHA256 every BF16 activation payload (slow, no GPU)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts/inputs/preflight.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = protocol_summary(verify_payloads=args.verify_payloads)
    output = args.output.resolve()
    try:
        output.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("preflight output must remain inside reverse experiment") from exc
    atomic_write_json(output, summary)
    checksum = sha256_file(output)
    atomic_write_json(
        output.with_suffix(".sha256.json"),
        {"path": str(output), "sha256": checksum},
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(output),
                "sha256": checksum,
                "payload_checksums_verified": args.verify_payloads,
                "role_binding": summary["role_binding"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
