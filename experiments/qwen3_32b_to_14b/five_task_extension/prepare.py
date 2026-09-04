#!/usr/bin/env python3
"""CPU-only preflight and exact expected-fingerprint derivation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

EXTENSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXTENSION_ROOT.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from five_task_extension.protocol import (  # noqa: E402
    EXECUTED_TASKS,
    EXTENSION_CONFIG,
    EXTENSION_VERSION,
    HELLASWAG_ROOT,
    PAPER_BIT_EXACT,
    PAPER_SCOPE_NOTE,
    SMOKE_LIMIT,
    SMOKE_WIKITEXT_CHUNKS,
)
from five_task_extension.validator import (  # noqa: E402
    _validate_current_benchmark_payload,
    validate_extension_config,
    validate_extension_code_freeze,
    validate_hellaswag_bridge,
)
from five_task_extension.adapter import extension_project_code_identity  # noqa: E402
from reverse_kv.eval_adapter import _load_implementation  # noqa: E402
from reverse_kv.protocol import canonical_sha256, read_json, sha256_file  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    return parser.parse_args(argv)


def expected_definition(profile: str, *, deep_freeze: bool = False) -> dict[str, Any]:
    validate_extension_config()
    validate_extension_code_freeze(deep_models=deep_freeze, deep_mapper=True)
    bridge = validate_hellaswag_bridge()
    base = read_json(HELLASWAG_ROOT / "run_definition.json")
    _validate_current_benchmark_payload(base)
    implementation = _load_implementation()
    value = copy.deepcopy(base)
    value.pop("request_fingerprint", None)
    value["tasks"] = list(EXECUTED_TASKS)
    value["limit"] = None if profile == "full" else float(SMOKE_LIMIT)
    value["wikitext_max_chunks"] = None if profile == "full" else SMOKE_WIKITEXT_CHUNKS
    value["project_code"] = extension_project_code_identity()
    value["harness"] = implementation.harness_provenance(
        Path(value["harness"]["path"]),
        EXECUTED_TASKS[:-1],  # WikiText uses the custom scorer, not lm-eval.
    )
    fingerprint = canonical_sha256(value)
    return {**value, "request_fingerprint": fingerprint}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze_evidence = validate_extension_code_freeze(deep_models=True, deep_mapper=True)
    definition = expected_definition(args.profile, deep_freeze=False)
    output = {
        "schema_version": 1,
        "status": "ready",
        "cpu_only": True,
        "extension_version": EXTENSION_VERSION,
        "profile": args.profile,
        "paper_bit_exact": PAPER_BIT_EXACT,
        "paper_scope_note": PAPER_SCOPE_NOTE,
        "expected_request_fingerprint": definition["request_fingerprint"],
        "base_eval_config_sha256": definition["config_sha256"],
        "extension_config_sha256": sha256_file(EXTENSION_CONFIG),
        "base_code_freeze_referenced_by_shared_adapter": definition["project_code"]["reverse_code_freeze"],
        "extension_code_freeze": freeze_evidence,
        "runtime_artifacts": definition["runtime_artifacts"],
        "hellaswag_bridge": validate_hellaswag_bridge(),
        "old_forward_14b_bridge": {
            "decision": "rejected_for_formal_receipt",
            "reasons": [
                "old forward provenance did not content-hash every 14B weight shard",
                "semantic role was forward source rather than reverse target",
                "physical topology was two GPUs rather than the formal reverse three-GPU target",
                "the combined old run used max_gen_tokens=256; GSM used a separate max1024 run",
            ],
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
