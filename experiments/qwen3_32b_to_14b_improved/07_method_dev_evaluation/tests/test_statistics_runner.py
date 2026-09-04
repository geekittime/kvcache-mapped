from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.contracts import ContractError, MODES, canonical_sha256, token_ids_sha256
from src.metrics import score_answer
from src.runner import _confirmation_prerequisite, validate_checkpoint, validate_generation
from src.statistics import aggregate_paired_results, paired_bootstrap_mean_delta


class FakeTokenizer:
    def decode(self, values, skip_special_tokens=True):
        return "".join(chr(value) for value in values)


def _record(dataset: str, sample: str, mode: str, f1: float, em: float) -> dict:
    return {
        "request_fingerprint": "f" * 64,
        "dataset_id": dataset,
        "sample_id": sample,
        "mode": mode,
        "metrics": {"official_answer_f1": f1, "official_answer_em": em},
        "generation": {
            "generated_tokens": 1,
            "stop_reason": "eos",
            "cap_hit": False,
        },
        "truncation": {"truncated": False},
    }


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    first = paired_bootstrap_mean_delta([1.0, 0.0, 0.5], [0.0, 0.0, 0.0], seed=7, replicates=500)
    second = paired_bootstrap_mean_delta([1.0, 0.0, 0.5], [0.0, 0.0, 0.0], seed=7, replicates=500)
    assert first == second
    assert first["mean_delta"] == 0.5
    assert first["ci_low"] <= first["mean_delta"] <= first["ci_high"]


def test_aggregate_reports_delta_retention_gap_and_flips() -> None:
    records = []
    for dataset in ("hotpotqa_train", "musique_train"):
        for sample, target, baseline, shrinkage, r8, r32 in (
            ("a", 1.0, 0.0, 1.0, 1.0, 1.0),
            ("b", 1.0, 1.0, 1.0, 0.0, 1.0),
        ):
            values = {
                "native14": target,
                "nvidia_k20": baseline,
                "shrinkage": shrinkage,
                "tail_refresh_r8": r8,
                "tail_refresh_r32": r32,
            }
            for mode, value in values.items():
                records.append(_record(dataset, sample, mode, value, value))
    summary = aggregate_paired_results(
        records,
        profile="smoke",
        request_fingerprint="f" * 64,
        bootstrap_replicates=200,
    )
    comparison = summary["paired_comparisons"]["hotpotqa_train"]["shrinkage"]
    assert comparison["official_answer_f1"]["mean_delta"] == 0.5
    assert comparison["official_answer_f1"]["raw_score_retention_percent"] == 100.0
    assert comparison["official_answer_f1"]["gap_closure_vs_k20"] == 1.0
    assert comparison["exact_match_fidelity"]["target_correct_retention"] == 1.0
    assert comparison["exact_match_fidelity"]["candidate_vs_k20_2x2"]["candidate_only"] == 1


def test_aggregate_rejects_missing_paired_cell() -> None:
    records = [_record("hotpotqa_train", "a", mode, 1.0, 1.0) for mode in MODES[:-1]]
    with pytest.raises(ContractError, match="missing"):
        aggregate_paired_results(records, profile="smoke", request_fingerprint="f" * 64)


def test_generation_contract_distinguishes_stop_sequence_from_exact_decode() -> None:
    ids = [ord(value) for value in "Paris\nQuestion:"]
    generation = {
        "requested_max_tokens": 256,
        "generated_tokens": len(ids),
        "generated_token_ids": ids,
        "generated_token_ids_sha256": token_ids_sha256(ids),
        "decoded_generated_token_ids": "Paris\nQuestion:",
        "stop_reason": "stop_sequence",
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "cap_hit": False,
    }
    validate_generation(
        generation,
        tokenizer=FakeTokenizer(),
        expected_cap=256,
        stop_sequences=["\nQuestion:", "\nContext:"],
        prediction="Paris",
    )
    broken = copy.deepcopy(generation)
    broken["cap_hit"] = True
    with pytest.raises(ContractError, match="non-cap"):
        validate_generation(
            broken,
            tokenizer=FakeTokenizer(),
            expected_cap=256,
            stop_sequences=["\nQuestion:", "\nContext:"],
            prediction="Paris",
        )
    trailing = copy.deepcopy(generation)
    trailing["decoded_generated_token_ids"] = "Paris\nQuestion: trailing"
    with pytest.raises(ContractError, match="exact first registered stop boundary"):
        validate_generation(
            trailing,
            tokenizer=None,
            expected_cap=256,
            stop_sequences=["\nQuestion:", "\nContext:"],
            prediction="Paris",
        )


def test_confirm_requires_explicit_screen_output_before_gpu_gate() -> None:
    from src.contracts import load_config

    with pytest.raises(ContractError, match="screen-output-dir"):
        _confirmation_prerequisite("config.json", load_config(verify_inputs=False), "confirm", None)


def test_checkpoint_recomputes_metric_and_bindings(tmp_path: Path) -> None:
    row = {
        "dataset_id": "hotpotqa_train",
        "sample_id": "sample",
        "sample_binding_sha256": "a" * 64,
        "source_record_sha256": "b" * 64,
        "prompt_token_ids_sha256": "c" * 64,
        "prompt_token_count": 40,
        "generation_cap": 256,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "references": ["Paris"],
        "truncation": {"truncated": False},
    }
    ids = [ord(value) for value in "Paris"]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": "f" * 64,
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "sample_binding_sha256": row["sample_binding_sha256"],
        "source_record_sha256": row["source_record_sha256"],
        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        "mode": "nvidia_k20",
        "method_binding_sha256": "d" * 64,
        "target_result_sha256": "e" * 64,
        "baseline_result_sha256": None,
        "prediction": "Paris",
        "references": ["Paris"],
        "metrics": score_answer("hotpotqa_train", "Paris", ["Paris"]),
        "generation": {
            "requested_max_tokens": 256,
            "generated_tokens": len(ids),
            "generated_token_ids": ids,
            "generated_token_ids_sha256": token_ids_sha256(ids),
            "decoded_generated_token_ids": "Paris",
            "stop_reason": "eos",
            "stop_sequences": ["\nQuestion:", "\nContext:"],
            "cap_hit": False,
        },
        "truncation": {"truncated": False},
        "builder_audit": None,
    }
    path = tmp_path / "result.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    validated = validate_checkpoint(
        path,
        row=row,
        mode="nvidia_k20",
        request_fingerprint="f" * 64,
        method_binding="d" * 64,
        tokenizer=FakeTokenizer(),
        expected_target_sha256="e" * 64,
        expected_baseline_sha256=None,
    )
    assert validated["metrics"]["official_answer_em"] == 1.0
