#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.contracts import artifact_ref, canonical_bytes, canonical_sha256, find_placeholders, read_json


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(value, newline=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish a non-GPU pre-formal code inventory")
    parser.add_argument("--pytest-summary", required=True)
    args = parser.parse_args()
    config = read_json(ROOT / "config.template.json")
    placeholders = find_placeholders(config)
    if not placeholders or config["execution_policy"]["gpu_execution_authorized_by_this_config"] is not False:
        raise RuntimeError("pre-formal publication requires placeholders and explicit no-GPU authorization")
    paths = [
        ROOT / "README.md",
        ROOT / "PROTOCOL.md",
        ROOT / "config.template.json",
        ROOT / "g0_consumer_contract.json",
        ROOT / "native_registry_consumer_contract.json",
    ]
    paths.extend(sorted((ROOT / "src").glob("*.py")))
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    paths.extend(sorted((ROOT / "tests").glob("*.py")))
    inventory = [artifact_ref(path) for path in paths]
    predecessor_path = ROOT / "artifacts" / "preformal" / "PRE_FORMAL_STOP_v2.json"
    if not predecessor_path.is_file():
        raise RuntimeError("hardened v3 stop must preserve and bind the schema-aligned v2 predecessor")
    frozen_schema_paths = {
        "dir12_g0_consumer_contract": ROOT.parent / "12_prospective_screen_v2" / "contracts" / "g0_consumer_contract.json",
        "dir09_g0_v2_producer": ROOT.parent / "09_complete_g0_audit" / "src" / "g0_audit_v2.py",
        "dir09_g0_v2_tests": ROOT.parent / "09_complete_g0_audit" / "tests" / "test_g0_audit_v2.py",
    }
    payload = {
        "schema_version": 1,
        "status": "PRE_FORMAL_STOP",
        "gpu_used": False,
        "model_loaded": False,
        "formal_config_created": False,
        "dir10_intermediate_artifacts_opened": False,
        "checked_in_template_gpu_authorized": False,
        "unresolved_placeholder_count": len(placeholders),
        "pytest_summary": args.pytest_summary,
        "revision": "pre_resolve_symlink_and_exact_answer_policy_hardening_v3",
        "predecessor": artifact_ref(predecessor_path),
        "frozen_external_schema_bindings": {
            name: artifact_ref(path) for name, path in frozen_schema_paths.items()
        },
        "schema_alignment": {
            "candidate_registry_binding": "nested exact4",
            "comparison_directions": [
                "methoddev_hotpot_vs_mapper_fit",
                "methoddev_musique_vs_mapper_fit",
                "methoddev_hotpot_vs_locked_eval",
                "methoddev_musique_vs_locked_eval",
            ],
            "each_comparison_exact_value": {"flags": 0},
            "dir10_receipt_direct_checks_retained": [
                "train_sources",
                "answer_access_policy",
                "projection",
            ],
        },
        "hardening": {
            "pre_resolve_path_gate": "component-wise lstat rejects live and dangling symlink chains",
            "answer_access_policy": {
                "exact_keys": ["policy_id", "allowed", "allowed_uses", "forbidden_uses"],
                "additional_keys_allowed": False,
                "typed_values_exact_between_code_template_and_receipt": True,
                "coordinated_formal_config_and_receipt_drift_rejected": True,
            },
        },
        "implementation_inventory": inventory,
        "implementation_inventory_sha256": canonical_sha256(inventory),
        "remaining_gates": [
            "final native dir10 PUBLISHED/manifest/receipt/four split refs",
            "final complete dir09 G0-v2 receipt",
            "pinned train-source refs and exact registry projection algorithm",
            "final strata order and smoke/diagnostic counts",
            "independent CPU audit of a frozen formal config",
            "GPU smoke authorization",
        ],
    }
    destination = ROOT / "artifacts" / "preformal" / "PRE_FORMAL_STOP_v3.json"
    atomic_json(destination, payload)
    atomic_json(destination.with_suffix(".json.sha256.json"), artifact_ref(destination))
    print(json.dumps(artifact_ref(destination), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
