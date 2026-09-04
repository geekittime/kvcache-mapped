#!/usr/bin/env python3
"""Publish SHA256 inventory for the frozen reverse implementation and dependencies."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.protocol import atomic_write_json, canonical_sha256, sha256_file  # noqa: E402
from reverse_kv.freeze_spec import expected_groups  # noqa: E402


def _entry(path: Path, *, base: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "relative_path": str(path.resolve().relative_to(base.resolve())),
        "nbytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    groups = expected_groups()
    value = {
        "schema_version": 1,
        "status": "frozen",
        "direction": "qwen3_32b_to_14b",
        "protocol": {
            "source_layers": 64,
            "target_layers": 40,
            "selected_source_layers": 20,
            "source_features": 20_480,
            "target_features": 1_024,
            "weight_count": 1_677_721_600,
            "bias_count": 81_920,
        },
        "curve_variants": {
            "figure4_large_to_small_retuned_k4": {
                "scope": "CoQA Figure 4 dotted retuned curve only",
                "production_k20_unchanged": True,
                "selected_source_layers": 4,
                "source_features": 4_096,
                "target_features": 1_024,
                "weight_count": 335_544_320,
                "bias_count": 81_920,
                "selection": (
                    "re-rank all 64 source layers from the frozen full score tensor"
                ),
                "fit": "independent centered ridge refit; never slice k20 weights",
                "heldout_gate_required_before_coqa": True,
            }
        },
        "local_files": [
            _entry(path, base=EXPERIMENT_ROOT) for path in groups["local_files"]
        ],
        "read_only_shared_dependencies": [
            _entry(path, base=REPOSITORY_ROOT)
            for path in groups["read_only_shared_dependencies"]
        ],
        "benchmark_protocol_dependencies": [
            _entry(path, base=REPOSITORY_ROOT)
            for path in groups["benchmark_protocol_dependencies"]
        ],
        "vendored_harness_dependencies": [
            _entry(path, base=REPOSITORY_ROOT)
            for path in groups["vendored_harness_dependencies"]
        ],
        "shared_dependency_policy": "content-hashed and imported read-only; no shared file is modified",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    digest_payload = dict(value)
    value["inventory_digest"] = canonical_sha256(digest_payload)
    output = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
    atomic_write_json(output, value)
    atomic_write_json(
        output.with_suffix(".sha256.json"),
        {"path": str(output), "sha256": sha256_file(output)},
    )
    print(f"{output} {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
