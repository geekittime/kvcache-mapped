from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.common import ContractError, artifact_ref
from src.tier_gate import (
    validate_answer_free_oracle_receipt,
    validate_capture_manifest,
    validate_registry_document,
)


def write_blob(path: Path, payload: bytes = b"synthetic") -> dict:
    path.write_bytes(payload)
    return artifact_ref(path)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def registry():
    return {
        "schema_version": 1,
        "kind": "three_tier_sequence_registry_v1",
        "records": [
            {
                "sequence_id": "fit-1",
                "group_id": "fit-group",
                "tier": "mapper-fit",
                "split": "train",
                "dataset": "general",
                "normalized_text_sha256": digest("fit-1"),
                "source_projection_sha256": digest("fit-proj-1"),
                "partition": "fit",
            },
            {
                "sequence_id": "fit-2",
                "group_id": "fit-group",
                "tier": "mapper-fit",
                "split": "train",
                "dataset": "general",
                "normalized_text_sha256": digest("fit-2"),
                "source_projection_sha256": digest("fit-proj-2"),
                "partition": "fit",
            },
            {
                "sequence_id": "dev-1",
                "group_id": "dev-group",
                "tier": "method-dev",
                "split": "train",
                "dataset": "hotpotqa",
                "normalized_text_sha256": digest("dev-1"),
                "source_projection_sha256": digest("dev-proj-1"),
                "partition": "screen",
            },
            {
                "sequence_id": "lock-1",
                "group_id": "lock-group",
                "tier": "locked-test",
                "split": "test",
                "dataset": "hotpotqa",
                "normalized_text_sha256": digest("lock-1"),
                "source_projection_sha256": digest("lock-proj-1"),
                "partition": "locked",
            },
        ],
    }


def test_registry_allows_multiple_records_per_same_group_but_no_cross_tier_collision():
    summary = validate_registry_document(registry())
    assert summary["record_count"] == 4
    assert summary["cross_tier_exact_collisions"] == 0
    bad = registry()
    bad["records"][2]["group_id"] = "fit-group"
    with pytest.raises(ContractError, match="cross-tier group"):
        validate_registry_document(bad)


def test_registry_rejects_nested_answer_or_outcome_and_hash_collision():
    bad = registry()
    bad["records"][0]["answer"] = "secret"
    with pytest.raises(ContractError, match="key mismatch|forbidden"):
        validate_registry_document(bad)
    bad = registry()
    bad["records"][2]["normalized_text_sha256"] = bad["records"][0]["normalized_text_sha256"]
    with pytest.raises(ContractError, match="normalized text collision"):
        validate_registry_document(bad)


def test_capture_manifest_checks_every_payload_ref_and_teacher_policy(tmp_path: Path):
    blobs = {}
    for name in ("source", "native", "anchor", "query", "mask", "logits", "tokens", "positions", "registry"):
        blobs[name] = write_blob(tmp_path / f"{name}.bin", name.encode())
    manifest = {
        "schema_version": 1,
        "kind": "functional_capture_manifest_v1",
        "status": "PUBLISHED",
        "tier": "mapper-fit",
        "teacher": "native_qwen3_14b",
        "source_logits_as_teacher_allowed": False,
        "sequence_group_ids_sha256": digest("group-1"),
        "records": [
            {
                "sequence_id": "s1",
                "group_id": "g1",
                "source_features": blobs["source"],
                "native_cache": blobs["native"],
                "anchor_cache": blobs["anchor"],
                "functional_signals": {
                    "native_fixed_queries": blobs["query"],
                    "attention_row_masks": blobs["mask"],
                    "native_suffix_logits": blobs["logits"],
                    "suffix_token_ids": blobs["tokens"],
                    "eos_token_ids": [2],
                    "positions": blobs["positions"],
                },
            }
        ],
    }
    assert validate_capture_manifest(manifest, expected_tier="mapper-fit")["record_count"] == 1
    bad = json.loads(json.dumps(manifest))
    bad["source_logits_as_teacher_allowed"] = True
    with pytest.raises(ContractError):
        validate_capture_manifest(bad, expected_tier="mapper-fit")
    bad = json.loads(json.dumps(manifest))
    bad["records"][0]["functional_signals"]["native_suffix_logits"]["nbytes"] += 1
    with pytest.raises(ContractError):
        validate_capture_manifest(bad, expected_tier="mapper-fit")


def test_answer_free_oracle_receipt_requires_exact_candidate_and_mean_one_weight(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}", encoding="utf-8")
    source_ref = artifact_ref(registry_path)
    receipt = {
        "schema_version": 1,
        "kind": "answer_free_layer_oracle_receipt_v1",
        "status": "PUBLISHED",
        "tier": "method-dev",
        "answers_opened_for_selection": False,
        "model_outcomes_opened_for_selection": False,
        "candidate_layers": [7, 8, 9, 10, 34, 35],
        "active_layers": [7, 34],
        "layer_loss_weights": {"7": 0.8, "34": 1.2},
        "selection_metrics": [
            "native_repair_first_token_kl_reduction",
            "native_fixed_query_attention_output_cosine_gain",
        ],
        "selection_rule": "positive effect and paired bootstrap LCB > 0",
        "bootstrap_seed": 20260830,
        "bootstrap_replicates": 10000,
        "source_registry_ref": source_ref,
    }
    assert validate_answer_free_oracle_receipt(receipt, expected_candidate_layers=[7, 8, 9, 10, 34, 35])["active_layers"] == [7, 34]
    bad = json.loads(json.dumps(receipt))
    bad["layer_loss_weights"]["7"] = True
    with pytest.raises(ContractError):
        validate_answer_free_oracle_receipt(bad, expected_candidate_layers=[7, 8, 9, 10, 34, 35])

