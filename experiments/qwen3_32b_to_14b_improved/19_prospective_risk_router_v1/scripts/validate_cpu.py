#!/usr/bin/env python3
"""Run deterministic CPU-only contract checks without touching model state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk_router import (  # noqa: E402
    DEFAULT_CONFIG,
    Budget,
    Observation,
    assess,
    route_batch,
)


def load_config() -> dict[str, object]:
    with (ROOT / "configs/router.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run() -> dict[str, object]:
    config = load_config()
    signals = config["signals"]
    assert signals["prompt_tokens"]["enter"] == DEFAULT_CONFIG.prompt_enter_tokens
    assert signals["prompt_tokens"]["exit"] == DEFAULT_CONFIG.prompt_exit_tokens
    assert signals["context_tokens"]["enter"] == DEFAULT_CONFIG.context_enter_tokens
    assert signals["context_tokens"]["exit"] == DEFAULT_CONFIG.context_exit_tokens
    assert signals["r1_generated_tokens"]["enter"] == DEFAULT_CONFIG.r1_generated_enter_tokens
    assert signals["r1_generated_tokens"]["exit"] == DEFAULT_CONFIG.r1_generated_exit_tokens
    budget = Budget.for_batch(96)
    assert budget.limit == 28
    rows = (
        Observation(prompt_tokens=1000, r1_generated_tokens=4),
        Observation(prompt_tokens=2800, r1_generated_tokens=4),
        Observation(prompt_tokens=1000, r1_generated_tokens=16),
        Observation(prompt_tokens=2800, r1_generated_tokens=16),
        Observation(prompt_tokens=2600, r1_generated_tokens=12, context_tokens=2600),
    )
    decisions = route_batch(rows, budget=Budget(limit=2))
    assert [decision.action for decision in decisions] == ["r1", "r1", "r32", "r32", "r1"]
    assert assess(Observation(prompt_tokens=1000, r1_generated_tokens=4)).valid
    invalid = assess(Observation(prompt_tokens=1000, r1_generated_tokens=4, context_tokens=1001))
    assert invalid.valid is False and invalid.no_go == "context_exceeds_prompt"
    return {
        "status": "passed_cpu_contract",
        "gpu_used": False,
        "batch_budget_96": budget.limit,
        "decision_vector": [decision.action for decision in decisions],
        "invalid_reason": invalid.no_go,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit a JSON summary")
    args = parser.parse_args()
    result = run()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
