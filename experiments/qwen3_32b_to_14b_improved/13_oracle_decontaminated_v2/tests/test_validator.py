from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.contracts import ContractError, canonical_sha256, intervention_catalog
from src.validator import validate_atomic_result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def mock_binding() -> dict:
    h = "1" * 64
    return {
        "dataset": "hotpotqa",
        "sample_id": "sample000000000000000001",
        "source_index": 0,
        "selection_rank": 0,
        "stratum": "all",
        "source_projection_sha256": h,
        "input_commitment_sha256": h,
        "prompt_token_count": 10,
        "serialized_prompt_sha256": "2" * 64,
        "prompt_token_ids_sha256": "3" * 64,
        "prefix_token_ids_sha256": "4" * 64,
        "references_sha256": "5" * 64,
        "gold_first_token_ids_sha256": "6" * 64,
    }


def mock_result(binding: dict, fingerprint: str) -> dict:
    ids = [item["id"] for item in intervention_catalog()]
    layers = [
        {
            "layer": index,
            "attention_output_cosine": 1.0,
            "attention_output_normalized_mse": 0.0,
            "attention_probability_js": 0.0,
        }
        for index in range(40)
    ]
    paired = [
        {
            "intervention_id": intervention_id,
            "context_sha256": "8" * 64,
            "first_token_kl_native_to_candidate": 0.0,
            "native_answer_margin": 0.1,
            "candidate_answer_margin": 0.1,
            "answer_margin_delta_candidate_minus_native": 0.0,
            "native_answer_em": 1.0,
            "candidate_answer_em": 1.0,
            "answer_em_delta_candidate_minus_native": 0.0,
            "native_answer_f1": 1.0,
            "candidate_answer_f1": 1.0,
            "answer_f1_delta_candidate_minus_native": 0.0,
            "attention_output_cosine_uniform_layer_mean": 1.0,
            "attention_layers": copy.deepcopy(layers),
        }
        for intervention_id in ids
    ]
    traces = [
        {
            "intervention_id": intervention_id,
            "generated_token_ids": [],
            "generated_token_ids_sha256": canonical_sha256([]),
            "generated_token_count": 0,
            "stop_reason": "eos",
            "cap": 256,
            "decoded_text_sha256": "7" * 64,
        }
        for intervention_id in ids
    ]
    compositions = []
    for spec in intervention_catalog():
        native_layers = set(spec["native_kv_layers"])
        compositions.append({
            "intervention_id": spec["id"],
            "cloned": True,
            "k_sources": ["native" if index in native_layers else spec["default_k"] for index in range(40)],
            "v_sources": ["native" if index in native_layers else spec["default_v"] for index in range(40)],
            "context_sha256": "8" * 64,
            "mapper_manifest_sha256": "a" * 64,
        })
    return {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        **binding,
        "causal_boundary": {
            "prefix": "C[:-1]",
            "replay": "C[-1]",
            "full_prompt_identity_passed": True,
            "native_boundary_qkv_identity_passed": True,
        },
        "exact_cache_envelope": {
            "layers": 40,
            "batch": 1,
            "kv_heads": 8,
            "head_dim": 128,
            "prefix_length": 9,
            "positions": "contiguous_zero_based_C_prefix",
            "key_space": "post_target_qwen3_rope_cache",
            "value_space": "target_cache_value_space",
            "finite_checked": True,
            "native_mapped_shape_dtype_device_equal": True,
            "context_sha256": "8" * 64,
        },
        "mapped_cache_materializations": 1,
        "paired": paired,
        "intervention_generation": traces,
        "compositions": compositions,
        "systems": {"gpu_peak": {}},
    }


def test_complete_atomic_mock_validates(tmp_path: Path) -> None:
    binding, fingerprint = mock_binding(), "9" * 64
    path = tmp_path / "result.json"
    write_json(path, mock_result(binding, fingerprint))
    assert validate_atomic_result(path, request_fingerprint=fingerprint, plan_binding=binding)["status"] == "complete"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(mapped_cache_materializations=2), "exactly once"),
        (lambda value: value["paired"][0]["attention_layers"].pop(), "40 ordered"),
        (lambda value: value["intervention_generation"][0].update(generated_token_ids_sha256="0" * 64), "token commitment"),
        (lambda value: value["causal_boundary"].update(full_prompt_identity_passed=False), "causal boundary"),
        (lambda value: value["paired"].reverse(), "intervention order"),
        (lambda value: value["compositions"][4]["k_sources"].__setitem__(0, "mapped"), "K/V sources changed"),
        (lambda value: value.update(references_sha256="0" * 64), "binding changed for references_sha256"),
    ],
)
def test_atomic_tampering_fails_closed(tmp_path: Path, mutation, match: str) -> None:
    binding, fingerprint = mock_binding(), "9" * 64
    value = mock_result(binding, fingerprint)
    mutation(value)
    path = tmp_path / "tampered.json"
    write_json(path, value)
    with pytest.raises(ContractError, match=match):
        validate_atomic_result(path, request_fingerprint=fingerprint, plan_binding=binding)



def test_nonfinite_metric_fails_closed(tmp_path: Path) -> None:
    binding, fingerprint = mock_binding(), "9" * 64
    value = mock_result(binding, fingerprint)
    value["paired"][2]["first_token_kl_native_to_candidate"] = float("inf")
    path = tmp_path / "nonfinite.json"
    write_json(path, value)
    with pytest.raises(ContractError, match="non-finite"):
        validate_atomic_result(path, request_fingerprint=fingerprint, plan_binding=binding)
