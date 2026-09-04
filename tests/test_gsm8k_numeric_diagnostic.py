from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import kvcache_transfer.gsm8k_numeric_diagnostic as gsm
from kvcache_transfer.gsm8k_numeric_diagnostic import (
    GSM8KNumericDiagnosticError,
    canonical_lm_eval_exact_match,
    canonical_lm_eval_filter,
    extract_final_numeric_answer,
    parse_numeric_value,
    score_checkpoint,
    verify_report_directory,
    write_report,
)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("$70,000", "70000"),
        ("18.00", "18"),
        ("-2", "-2"),
        ("0.125", "1/8"),
    ],
)
def test_parse_numeric_value_uses_exact_decimal_equality(raw, canonical):
    assert parse_numeric_value(raw).canonical == canonical


def test_extract_final_answer_is_gold_independent_and_prefers_last_cue():
    response = (
        "Work gives 17. The answer is 17. I corrected the arithmetic. "
        "Final answer: **$18.00**."
    )
    result = extract_final_numeric_answer(response)
    assert result.value is not None
    assert result.value.canonical == "18"
    assert result.strategy == "last_explicit_answer_marker"


def test_extract_box_then_fallback_and_missing():
    boxed = extract_final_numeric_answer(r"Thus \boxed{18}.")
    assert boxed.value is not None
    assert boxed.value.canonical == "18"
    assert boxed.strategy == "fallback_last_numeric_match"
    fallback = extract_final_numeric_answer("We obtain 9, then 18.")
    assert fallback.value is not None
    assert fallback.value.canonical == "18"
    assert fallback.strategy == "fallback_last_numeric_match"
    assert extract_final_numeric_answer("No numeric answer was produced.").status == "missing"


def test_pinned_lm_eval_filters_and_metric_are_recomputed_from_raw_response():
    response = "The answer is 7. Correction: The answer is $18."
    # Strict is RegexFilter default group_select=0: first strict-compatible match.
    assert canonical_lm_eval_filter(response, "strict-match") == "7."
    # Flexible has group_select=-1: final regex match, including currency/period.
    flexible = canonical_lm_eval_filter(response, "flexible-extract")
    assert flexible == "$18."
    assert canonical_lm_eval_exact_match(flexible, "18") == 1
    assert canonical_lm_eval_exact_match("[invalid]", "18") == 0


def _digest(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sample(doc_id: int, target: str, response: str, *, filter_name: str) -> dict:
    question = f"q{doc_id}"
    doc = {"question": question, "answer": f"work\n#### {target}"}
    prompt = f"eight-shot-prefix\n\nQ: {question}\nA:"
    filtered = canonical_lm_eval_filter(response, filter_name)
    bit = canonical_lm_eval_exact_match(filtered, target)
    return {
        "arguments": [[prompt, copy.deepcopy(gsm.EXPECTED_GENERATION_KWARGS)]],
        "doc": doc,
        "doc_hash": hashlib.sha256(
            json.dumps(doc, indent=2, ensure_ascii=False).encode()
        ).hexdigest(),
        "doc_id": doc_id,
        "exact_match": float(bit),
        "filter": filter_name,
        "filtered_resps": [filtered],
        "metrics": ["exact_match"],
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "resps": [[response]],
        "target": target,
        "target_hash": hashlib.sha256(target.encode()).hexdigest(),
    }


def _checkpoint() -> dict:
    records = [
        (0, "18", "The answer is $18."),
        (1, "3", "The answer is 4."),
    ]
    strict = [
        _sample(doc_id, target, response, filter_name="strict-match")
        for doc_id, target, response in records
    ]
    flexible = [
        _sample(doc_id, target, response, filter_name="flexible-extract")
        for doc_id, target, response in records
    ]
    task_config = {
        "gsm8k_cot": {
            "num_fewshot": 8,
            "output_type": "generate_until",
            "repeats": 1,
            "test_split": "test",
            "generation_kwargs": copy.deepcopy(gsm.EXPECTED_GENERATION_KWARGS),
            "filter_list": [
                copy.deepcopy(gsm.EXPECTED_STRICT_FILTER),
                copy.deepcopy(gsm.EXPECTED_FLEXIBLE_FILTER),
            ],
            "metric_list": [copy.deepcopy(gsm.EXPECTED_METRIC)],
        }
    }
    return {
        "schema_version": 1,
        "status": "complete",
        "task": "gsm8k_cot",
        "mode": "mapped",
        "primary_metric": "exact_match",
        "primary_metric_value": 0.0,
        "request_fingerprint": None,
        "finished_utc": "2026-01-01T00:00:00+00:00",
        "resolved_task_config": {
            "configs": task_config,
            "canonical_configs_sha256": _digest(task_config),
            "n_shot": {"gsm8k_cot": 8},
            "versions": {"gsm8k_cot": 3.0},
        },
        "result": {
            "configs": task_config,
            "n-shot": {"gsm8k_cot": 8},
            "versions": {"gsm8k_cot": 3.0},
            "results": {
                "gsm8k_cot": {
                    "exact_match,strict-match": 0.0,
                    "exact_match,flexible-extract": 0.5,
                    "sample_len": 2,
                }
            },
            "n-samples": {"gsm8k_cot": {"effective": 2, "original": 2}},
            "samples": {"gsm8k_cot": strict + flexible},
        },
    }


def _publish_fixture(tmp_path: Path, checkpoint: dict | None = None) -> Path:
    run = tmp_path / "run"
    destination = run / "results" / "mapped" / "gsm8k_cot.json"
    destination.parent.mkdir(parents=True)
    config = tmp_path / "max1024.yaml"
    config.write_text("max_gen_tokens: 1024\n", encoding="utf-8")
    run_definition = {
        "schema_version": 1,
        "config_path": str(config.resolve()),
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "modes": ["mapped"],
        "tasks": ["gsm8k_cot"],
        "limit": None,
        "log_samples": True,
        "completion_mode": True,
        "apply_chat_template": False,
        "batch_size": 1,
        "max_gen_tokens": 1024,
        "benchmarks": {
            "gsm8k_cot": {
                "metric": "exact_match",
                "metric_filter": "strict-match",
                "mode": "greedy_generation_cot",
                "num_fewshot": 8,
            }
        },
    }
    fingerprint = _digest(run_definition)
    run_definition["request_fingerprint"] = fingerprint
    value = copy.deepcopy(checkpoint if checkpoint is not None else _checkpoint())
    value["request_fingerprint"] = fingerprint
    summary = {
        "schema_version": 1,
        "request_fingerprint": fingerprint,
        "metrics": {"mapped": {"gsm8k_cot": value["primary_metric_value"]}},
    }
    (run / "run_definition.json").write_text(json.dumps(run_definition), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "status.json").write_text(
        json.dumps({"status": "complete", "failures": 0, "summary": summary}),
        encoding="utf-8",
    )
    destination.write_text(json.dumps(value), encoding="utf-8")
    return destination


@pytest.fixture(autouse=True)
def _tiny_full_dataset_contract(monkeypatch):
    monkeypatch.setattr(gsm, "EXPECTED_SAMPLE_COUNT", 2)


def test_score_checkpoint_recomputes_lm_eval_and_sample_hashes(tmp_path: Path):
    checkpoint = _publish_fixture(tmp_path)
    report = score_checkpoint(checkpoint)
    assert report["preserved_lm_eval_metrics"]["strict_match"]["correct"] == 0
    assert report["preserved_lm_eval_metrics"]["flexible_extract"]["correct"] == 1
    assert report["numeric_final_diagnostic"] == {
        "correct": 1,
        "total": 2,
        "accuracy": 0.5,
    }
    assert report["assurance"]["raw_response_filters_recomputed"] is True
    assert report["assurance"]["sample_identity_hashes_recomputed"] is True
    assert report["input"]["run_provenance"]["request_fingerprint_verified"] is True
    assert report["samples"][0]["lm_eval"]["raw_response_filter_recomputed"] is True


def test_refuses_coordinated_persisted_bit_and_aggregate_tampering(tmp_path: Path):
    value = _checkpoint()
    strict0 = value["result"]["samples"]["gsm8k_cot"][0]
    strict0["exact_match"] = 1.0
    value["result"]["results"]["gsm8k_cot"]["exact_match,strict-match"] = 0.5
    value["primary_metric_value"] = 0.5
    checkpoint = _publish_fixture(tmp_path, value)
    with pytest.raises(GSM8KNumericDiagnosticError, match="strict exact_match bit"):
        score_checkpoint(checkpoint)


@pytest.mark.parametrize("hash_field", ["prompt_hash", "target_hash", "doc_hash"])
def test_refuses_recomputed_sample_hash_tampering(tmp_path: Path, hash_field: str):
    value = _checkpoint()
    for sample in value["result"]["samples"]["gsm8k_cot"]:
        if sample["doc_id"] == 0:
            sample[hash_field] = "0" * 64
    checkpoint = _publish_fixture(tmp_path, value)
    with pytest.raises(GSM8KNumericDiagnosticError, match=rf"{hash_field} mismatch"):
        score_checkpoint(checkpoint)


def test_refuses_raw_response_filter_tampering(tmp_path: Path):
    value = _checkpoint()
    for sample in value["result"]["samples"]["gsm8k_cot"]:
        if sample["doc_id"] == 0:
            sample["resps"] = [["The answer is 19."]]
    checkpoint = _publish_fixture(tmp_path / "raw", value)
    with pytest.raises(GSM8KNumericDiagnosticError, match="filtered response"):
        score_checkpoint(checkpoint)


def test_refuses_run_fingerprint_and_protocol_tampering(tmp_path: Path):
    checkpoint = _publish_fixture(tmp_path)
    run_definition_path = checkpoint.parents[2] / "run_definition.json"
    run_definition = json.loads(run_definition_path.read_text())
    run_definition["max_gen_tokens"] = 2048
    run_definition_path.write_text(json.dumps(run_definition), encoding="utf-8")
    with pytest.raises(GSM8KNumericDiagnosticError, match="request_fingerprint"):
        score_checkpoint(checkpoint)


def test_report_strict_reuse_conflict_overwrite_and_manifest_verify(tmp_path: Path):
    report = score_checkpoint(_publish_fixture(tmp_path / "input"))
    output = tmp_path / "artifact"
    created = write_report(report, output)
    assert created["publication_status"] == "created"
    manifest_bytes = (output / "sha256_manifest.json").read_bytes()
    reused = write_report(report, output)
    assert reused["publication_status"] == "reused_complete"
    assert (output / "sha256_manifest.json").read_bytes() == manifest_bytes
    assert verify_report_directory(output, expected_report=report)["status"] == "complete"

    changed = copy.deepcopy(report)
    changed["paper_disclosure"] += " changed"
    with pytest.raises(GSM8KNumericDiagnosticError, match="pass --overwrite"):
        write_report(changed, output)
    overwritten = write_report(changed, output, overwrite=True)
    assert overwritten["publication_status"] == "overwritten"
    verify_report_directory(output, expected_report=changed)

    with (output / "summary.json").open("a", encoding="utf-8") as handle:
        handle.write("tamper")
    with pytest.raises(GSM8KNumericDiagnosticError, match="checksum/size mismatch"):
        verify_report_directory(output)


def test_checkpoint_must_be_complete_and_gsm8k(tmp_path: Path):
    value = _checkpoint()
    value["status"] = "failed"
    checkpoint = _publish_fixture(tmp_path, value)
    with pytest.raises(GSM8KNumericDiagnosticError, match="status must be complete"):
        score_checkpoint(checkpoint)
