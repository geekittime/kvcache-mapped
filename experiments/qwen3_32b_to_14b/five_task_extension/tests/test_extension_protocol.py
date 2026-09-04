from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path
import copy

import pytest
import yaml

from five_task_extension.prepare import expected_definition
from five_task_extension.run import _next_segment_gate
from five_task_extension.protocol import (
    ACCURACY_TASKS,
    EXECUTED_TASKS,
    EXTENSION_CONFIG,
    EXTENSION_CODE_FREEZE,
    PAPER_BIT_EXACT,
    output_within_experiment,
)
from five_task_extension.validator import (
    ExtensionValidationError,
    _build_summary,
    _recompute_multiple_choice_rows,
    validate_extension_code_freeze,
    validate_hellaswag_bridge,
)
from reverse_kv.protocol import canonical_sha256
from reverse_kv.eval_common import hashed_json


def test_declared_scope_is_explicitly_not_paper_bit_exact() -> None:
    config = yaml.safe_load(EXTENSION_CONFIG.read_text(encoding="utf-8"))
    assert PAPER_BIT_EXACT is False
    assert config["paper_bit_exact"] is False
    assert tuple(config["protocol"]["freshly_executed_tasks"]) == EXECUTED_TASKS
    assert tuple(ACCURACY_TASKS) == (
        "arc_challenge",
        "hellaswag",
        "winogrande",
        "mmlu",
        "gsm8k_cot",
    )
    assert config["reuse_decision"]["old_forward_14b_source"] == "rejected_for_formal_bridge"


def test_output_cannot_escape_reverse_experiment(tmp_path) -> None:
    with pytest.raises(ValueError, match="must remain"):
        output_within_experiment(tmp_path)


def test_existing_hellaswag_bridge_is_full_and_two_mode() -> None:
    bridge = validate_hellaswag_bridge()
    assert bridge["reuse_decision"] == "strict_bridge_no_rerun"
    assert {row["mode"] for row in bridge["results"]} == {"target_standalone", "mapped"}
    assert all(row["samples"] == 10_042 for row in bridge["results"])


def test_cpu_preparation_derives_exact_full_fingerprint() -> None:
    definition = expected_definition("full")
    assert definition["tasks"] == list(EXECUTED_TASKS)
    assert definition["modes"] == ["target_standalone", "mapped"]
    assert definition["limit"] is None
    assert definition["wikitext_max_chunks"] is None
    assert definition["max_gen_tokens"] == 1024
    assert definition["project_code"]["five_task_extension_overlay"]["paper_bit_exact"] is False
    assert definition["project_code"]["five_task_extension_overlay"]["sha256"] == hashlib.sha256(
        EXTENSION_CODE_FREEZE.read_bytes()
    ).hexdigest()
    assert re.fullmatch(r"[0-9a-f]{64}", definition["request_fingerprint"])


def test_summary_uses_numeric_gsm_and_preserves_strict() -> None:
    tasks = ("arc_challenge", "winogrande", "mmlu")
    rows = []
    for task in tasks:
        rows.extend(
            [
                {"mode": "target_standalone", "task": task, "value": 0.8},
                {"mode": "mapped", "task": task, "value": 0.7},
            ]
        )
    rows.extend(
        [
            {"mode": "target_standalone", "task": "wikitext2_prefix_ppl", "value": 6.0},
            {"mode": "mapped", "task": "wikitext2_prefix_ppl", "value": 7.0},
        ]
    )
    raw = {"fingerprint": "a" * 64, "rows": rows}
    bridge = {
        "results": [
            {"mode": "target_standalone", "acc_norm": 0.79},
            {"mode": "mapped", "acc_norm": 0.77},
        ]
    }
    numeric = {
        "target_standalone": {
            "strict": {"correct": 1000, "total": 1319, "accuracy": 1000 / 1319},
            "numeric_final": {"correct": 1200, "total": 1319, "accuracy": 1200 / 1319},
        },
        "mapped": {
            "strict": {"correct": 900, "total": 1319, "accuracy": 900 / 1319},
            "numeric_final": {"correct": 1100, "total": 1319, "accuracy": 1100 / 1319},
        },
    }
    summary = _build_summary(raw, bridge, numeric)
    gsm = summary["accuracy"]["gsm8k_cot"]
    assert gsm["target_14b"] == 1200 / 1319
    assert gsm["mapped_32b_to_14b"] == 1100 / 1319
    assert gsm["canonical_strict_match"]["target_standalone"]["correct"] == 1000
    assert summary["paper_bit_exact"] is False


def _commit_mutated_freeze(tmp_path: Path, mutate) -> Path:
    value = json.loads(EXTENSION_CODE_FREEZE.read_text(encoding="utf-8"))
    mutate(value)
    unsigned = dict(value)
    unsigned.pop("inventory_digest", None)
    value["inventory_digest"] = canonical_sha256(unsigned)
    destination = tmp_path / "code_freeze.json"
    destination.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    payload = destination.read_bytes()
    sidecar = {
        "schema_version": 1,
        "artifact": destination.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "nbytes": len(payload),
    }
    (tmp_path / "code_freeze.json.sha256.json").write_text(
        json.dumps(sidecar, sort_keys=True), encoding="utf-8"
    )
    return destination


def test_coordinated_extension_code_rehash_cannot_hide_drift(tmp_path) -> None:
    def mutate(value):
        value["extension_files"][0]["sha256"] = "0" * 64

    path = _commit_mutated_freeze(tmp_path, mutate)
    with pytest.raises(ExtensionValidationError, match="frozen dependency drifted"):
        validate_extension_code_freeze(path, deep_models=False, deep_mapper=False)


def test_coordinated_mapper_rehash_cannot_hide_payload_tamper(tmp_path) -> None:
    def mutate(value):
        value["mapper_payload"]["files"][0]["sha256"] = "0" * 64
        value["mapper_payload"]["files_digest"] = canonical_sha256(
            value["mapper_payload"]["files"]
        )

    path = _commit_mutated_freeze(tmp_path, mutate)
    with pytest.raises(ExtensionValidationError, match="mapper payload bytes changed"):
        validate_extension_code_freeze(path, deep_models=False, deep_mapper=True)


def test_coordinated_model_inventory_rehash_cannot_remove_a_shard(tmp_path) -> None:
    def mutate(value):
        model = value["models"]["target_14b"]
        model["weight_shards"].pop()
        model["weight_shard_count"] -= 1
        model["weight_shards_sha256"] = canonical_sha256(model["weight_shards"])

    path = _commit_mutated_freeze(tmp_path, mutate)
    with pytest.raises(ExtensionValidationError, match="model identities changed"):
        validate_extension_code_freeze(path, deep_models=False, deep_mapper=False)


def test_raw_multiple_choice_response_tamper_changes_recomputed_bit() -> None:
    checkpoint = json.loads(
        Path(
            "/home/wby/llm-cache/nvidia-kvcache/runs/20260813_185000_p7_full_target/"
            "results/target_standalone/arc_challenge.json"
        ).read_text(encoding="utf-8")
    )
    sample = copy.deepcopy(checkpoint["result"]["samples"]["arc_challenge"][0])
    gold = int(sample["target"])
    wrong = 0 if gold != 0 else 1
    sample["resps"][wrong][0][0] = 1_000.0
    sample["filtered_resps"][wrong][0] = 1_000.0
    with pytest.raises(ExtensionValidationError, match="persisted acc"):
        _recompute_multiple_choice_rows([("arc_challenge", sample)], task="arc_challenge")


def test_resume_segment_gate_names_are_append_only(tmp_path) -> None:
    first = _next_segment_gate(tmp_path)
    assert first.name == "segment_0001_gpu_idle_gate.json"
    gate = json.loads(
        Path(
            "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b/runs/"
            "formal_pipeline_v1/hellaswag_full_k20/gpu_idle_gate.json"
        ).read_text(encoding="utf-8")
    )
    hashed_json(first, gate)
    second = _next_segment_gate(tmp_path)
    assert second.name == "segment_0002_gpu_idle_gate.json"
    assert json.loads(first.read_text(encoding="utf-8"))["status"] == "passed"
