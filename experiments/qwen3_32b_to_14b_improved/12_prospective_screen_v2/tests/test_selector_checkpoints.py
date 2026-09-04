import json

import pytest

from src.checkpoints import (
    expected_r32_builder_audit,
    validate_generation,
    validate_raw_checkpoint,
)
from src.contracts import ContractError, canonical_sha256, token_ids_sha256
from src.metrics import score_answer
from src.two_pass_selector import derive_two_pass_result, should_choose_r32, validate_derived_result


class _Tokenizer:
    _prospective_frozen_eos_token_ids = (999,)

    def __init__(self, decoded):
        self.decoded = decoded

    def decode(self, _ids, skip_special_tokens=True):
        return self.decoded


def _generation(count=1, cap=False, prediction="answer"):
    ids = list(range(count))
    reason = "max_tokens" if cap else "eos"
    return {
        "requested_max_tokens": 256,
        "generated_tokens": count,
        "generated_token_ids": ids,
        "generated_token_ids_sha256": token_ids_sha256(ids),
        "decoded_generated_token_ids": prediction,
        "stop_reason": reason,
        "terminal_token_id": None if cap else 999,
        "frozen_eos_token_ids": [999],
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "cap_hit": cap,
    }


def _raw(mode, *, count=1, prediction="answer", native_hash=None, baseline_hash=None):
    result = {
        "schema_version": 2,
        "status": "complete",
        "result_kind": "raw_model_run",
        "request_fingerprint": "f" * 64,
        "dataset_id": "hotpotqa_train",
        "sample_id": "sample",
        "sample_binding_sha256": "a" * 64,
        "source_projection_sha256": "b" * 64,
        "prompt_token_ids_sha256": "c" * 64,
        "mode": mode,
        "method_binding_sha256": "d" * 64,
        "execution_attempt_id": "attempt_0001",
        "references": ["answer"],
        "truncation": {"truncated": False},
        "prediction": prediction,
        "generation": _generation(count=count, prediction=prediction),
        "metrics": score_answer("hotpotqa_train", prediction, ["answer"]),
        "runtime_cost": {
            "wall_seconds": 1.0,
            "phase_timing": {
                "available": False,
                "prefill_seconds": None,
                "decode_seconds": None,
                "scope": "monolithic_engine_generate_token_ids",
            },
            "peak_gpu_allocated_bytes": {
                str(index): 10 + index for index in range(8)
            },
        },
        "builder_audit": None,
        "comparison_bindings": {
            "native14_raw_file_sha256": native_hash,
            "nvidia_k20_raw_file_sha256": baseline_hash,
        },
    }
    if mode == "tail_refresh_r32":
        result["builder_audit"] = expected_r32_builder_audit(40)
    return result


def test_selector_boundary_is_strictly_greater_than_32_and_cap_hit():
    assert should_choose_r32(_generation(count=32)) is False
    assert should_choose_r32(_generation(count=33)) is True
    cap = _generation(count=256, cap=True)
    assert should_choose_r32(cap) is True


def test_selector_rejects_incomplete_r1():
    value = _generation(count=32)
    value["stop_reason"] = None
    with pytest.raises(ContractError, match="not a complete"):
        should_choose_r32(value)


def test_stop_sequence_allows_same_token_suffix_but_trims_first_boundary():
    decoded = "answer\nQuestion: trailing-piece-suffix"
    generation = {
        "requested_max_tokens": 256,
        "generated_tokens": 1,
        "generated_token_ids": [17],
        "generated_token_ids_sha256": token_ids_sha256([17]),
        "decoded_generated_token_ids": decoded,
        "stop_reason": "stop_sequence",
        "terminal_token_id": None,
        "frozen_eos_token_ids": [999],
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "cap_hit": False,
    }
    assert validate_generation(
        generation,
        expected_cap=256,
        stop_sequences=["\nQuestion:", "\nContext:"],
        prediction="answer",
        tokenizer=_Tokenizer(decoded),
    )["stop_reason"] == "stop_sequence"
    with pytest.raises(ContractError, match="trimmed at exact first stop boundary"):
        validate_generation(
            generation,
            expected_cap=256,
            stop_sequences=["\nQuestion:", "\nContext:"],
            prediction="answer\nQuestion:",
            tokenizer=_Tokenizer(decoded),
        )


@pytest.mark.parametrize("reason,cap_hit,count", (("eos", False, 1), ("max_tokens", True, 256)))
def test_non_stop_termination_rejects_registered_marker(reason, cap_hit, count):
    ids = list(range(count))
    generation = {
        "requested_max_tokens": 256,
        "generated_tokens": count,
        "generated_token_ids": ids,
        "generated_token_ids_sha256": token_ids_sha256(ids),
        "decoded_generated_token_ids": "answer\nQuestion:",
        "stop_reason": reason,
        "terminal_token_id": 999 if reason == "eos" else None,
        "frozen_eos_token_ids": [999],
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "cap_hit": cap_hit,
    }
    with pytest.raises(ContractError, match="contains a registered marker"):
        validate_generation(
            generation,
            expected_cap=256,
            stop_sequences=["\nQuestion:", "\nContext:"],
            prediction="answer\nQuestion:",
            tokenizer=_Tokenizer("answer\nQuestion:"),
        )


def test_eos_requires_room_for_excluded_terminal_token():
    ids = list(range(256))
    generation = {
        "requested_max_tokens": 256,
        "generated_tokens": 256,
        "generated_token_ids": ids,
        "generated_token_ids_sha256": token_ids_sha256(ids),
        "decoded_generated_token_ids": "answer",
        "stop_reason": "eos",
        "terminal_token_id": 999,
        "frozen_eos_token_ids": [999],
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "cap_hit": False,
    }
    with pytest.raises(ContractError, match="EOS terminal-token evidence"):
        validate_generation(
            generation,
            expected_cap=256,
            stop_sequences=["\nQuestion:", "\nContext:"],
            prediction="answer",
            tokenizer=_Tokenizer("answer"),
        )


def test_generated_token_count_rejects_python_bool_integer_alias():
    generation = _generation(count=1)
    generation["generated_tokens"] = True
    with pytest.raises(ContractError, match="token/count evidence is malformed"):
        validate_generation(
            generation,
            expected_cap=256,
            stop_sequences=["\nQuestion:", "\nContext:"],
            prediction="answer",
            tokenizer=_Tokenizer("answer"),
        )


@pytest.mark.parametrize(
    "case,reason,cap",
    (
        ("missing_eos_terminal", "eos", False),
        ("foreign_eos_terminal", "eos", False),
        ("changed_frozen_set", "eos", False),
        ("max_has_terminal", "max_tokens", True),
        ("stop_has_terminal", "stop_sequence", False),
    ),
)
def test_terminal_token_and_frozen_eos_evidence_tamper_is_rejected(case, reason, cap):
    if reason == "stop_sequence":
        generation = {
            "requested_max_tokens": 256,
            "generated_tokens": 1,
            "generated_token_ids": [17],
            "generated_token_ids_sha256": token_ids_sha256([17]),
            "decoded_generated_token_ids": "answer\nQuestion:",
            "stop_reason": reason,
            "terminal_token_id": None,
            "frozen_eos_token_ids": [999],
            "stop_sequences": ["\nQuestion:", "\nContext:"],
            "cap_hit": False,
        }
        decoded = "answer\nQuestion:"
        prediction = "answer"
    else:
        generation = _generation(count=256 if cap else 1, cap=cap)
        decoded = "answer"
        prediction = "answer"
    if case == "missing_eos_terminal":
        generation["terminal_token_id"] = None
    elif case == "foreign_eos_terminal":
        generation["terminal_token_id"] = 998
    elif case == "changed_frozen_set":
        generation["frozen_eos_token_ids"] = [998, 999]
    else:
        generation["terminal_token_id"] = 999
    with pytest.raises(
        ContractError, match="EOS|terminal|max-token|frozen target metadata"
    ):
        validate_generation(
            generation,
            expected_cap=256,
            stop_sequences=["\nQuestion:", "\nContext:"],
            prediction=prediction,
            tokenizer=_Tokenizer(decoded),
        )


def test_two_pass_returns_exact_r1_without_trigger_and_exact_r32_on_trigger():
    r1 = _raw("nvidia_k20", count=2, prediction="answer", native_hash="1" * 64)
    r32 = _raw("tail_refresh_r32", count=1, prediction="better", native_hash="1" * 64, baseline_hash="2" * 64)
    derived = derive_two_pass_result(r1, r32, r1_file_sha256="2" * 64, r32_file_sha256="3" * 64)
    assert derived["prediction"] == r1["prediction"]
    assert derived["selector_audit"]["selected_mode"] == "nvidia_k20"
    assert derived["selector_audit"]["online_watchdog"] is False
    r1_long = _raw("nvidia_k20", count=33, prediction="wrong", native_hash="1" * 64)
    selected = derive_two_pass_result(r1_long, r32, r1_file_sha256="4" * 64, r32_file_sha256="3" * 64)
    assert selected["prediction"] == r32["prediction"]
    assert selected["metrics"] == r32["metrics"]
    assert selected["selector_audit"]["selection_inputs"] == {
        "r1_generated_tokens": 33,
        "r1_cap_hit": False,
    }


def test_derived_validation_detects_selected_metric_tamper():
    r1 = _raw("nvidia_k20", count=33, prediction="wrong", native_hash="1" * 64)
    r32 = _raw("tail_refresh_r32", prediction="answer", native_hash="1" * 64, baseline_hash="2" * 64)
    value = derive_two_pass_result(r1, r32, r1_file_sha256="2" * 64, r32_file_sha256="3" * 64)
    value["metrics"]["official_answer_f1"] = 0.5
    with pytest.raises(ContractError, match="does not reproduce"):
        validate_derived_result(value, r1, r32, r1_file_sha256="2" * 64, r32_file_sha256="3" * 64)


def test_raw_checkpoint_recomputes_metrics_tokens_and_pair_bindings(tmp_path):
    row = {
        "dataset_id": "hotpotqa_train",
        "sample_id": "sample",
        "sample_binding_sha256": "a" * 64,
        "source_projection_sha256": "b" * 64,
        "prompt_token_ids_sha256": "c" * 64,
        "references": ["answer"],
        "truncation": {"truncated": False},
        "generation_cap": 256,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "prompt_token_count": 40,
    }
    value = _raw("tail_refresh_r32", native_hash="1" * 64, baseline_hash="2" * 64)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert validate_raw_checkpoint(
        path,
        row=row,
        mode="tail_refresh_r32",
        request_fingerprint="f" * 64,
        method_binding_sha256="d" * 64,
        expected_native_file_sha256="1" * 64,
        expected_baseline_file_sha256="2" * 64,
        tokenizer=_Tokenizer("answer"),
    )["status"] == "complete"
    value["comparison_bindings"]["nvidia_k20_raw_file_sha256"] = "9" * 64
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="comparison bindings"):
        validate_raw_checkpoint(
            path,
            row=row,
            mode="tail_refresh_r32",
            request_fingerprint="f" * 64,
            method_binding_sha256="d" * 64,
            expected_native_file_sha256="1" * 64,
            expected_baseline_file_sha256="2" * 64,
            tokenizer=_Tokenizer("answer"),
        )


@pytest.mark.parametrize(
    "case", ("missing_gpu_7", "extra_gpu_8", "available_phase", "bool_phase_alias")
)
def test_runtime_cost_exact_gpu_and_phase_schema_rejects_tamper(tmp_path, case):
    row = {
        "dataset_id": "hotpotqa_train",
        "sample_id": "sample",
        "sample_binding_sha256": "a" * 64,
        "source_projection_sha256": "b" * 64,
        "prompt_token_ids_sha256": "c" * 64,
        "references": ["answer"],
        "truncation": {"truncated": False},
        "generation_cap": 256,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "prompt_token_count": 40,
    }
    value = _raw("native14")
    if case == "missing_gpu_7":
        value["runtime_cost"]["peak_gpu_allocated_bytes"].pop("7")
    elif case == "extra_gpu_8":
        value["runtime_cost"]["peak_gpu_allocated_bytes"]["8"] = 18
    elif case == "available_phase":
        value["runtime_cost"]["phase_timing"] = {
            "available": True,
            "prefill_seconds": 0.5,
            "decode_seconds": 0.5,
            "scope": "another_scope",
        }
    else:
        value["runtime_cost"]["phase_timing"]["available"] = 0
    path = tmp_path / f"cost-{case}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="peak VRAM|phase timing"):
        validate_raw_checkpoint(
            path,
            row=row,
            mode="native14",
            request_fingerprint="f" * 64,
            method_binding_sha256="d" * 64,
            tokenizer=_Tokenizer("answer"),
        )


@pytest.mark.parametrize(
    "field,bad_value",
    (
        ("position_ids", list(range(9, 40))),
        ("cache_position", list(range(9, 40))),
        ("target_forward_calls", 0),
        ("target_forward_calls", True),
        ("engine_boundary_tokens", True),
        ("attention_mask_shape", [1, 38]),
    ),
)
def test_r32_exact_causal_audit_rejects_position_cache_and_forward_tamper(
    tmp_path, field, bad_value
):
    row = {
        "dataset_id": "hotpotqa_train",
        "sample_id": "sample",
        "sample_binding_sha256": "a" * 64,
        "source_projection_sha256": "b" * 64,
        "prompt_token_ids_sha256": "c" * 64,
        "references": ["answer"],
        "truncation": {"truncated": False},
        "generation_cap": 256,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "prompt_token_count": 40,
    }
    value = _raw(
        "tail_refresh_r32", native_hash="1" * 64, baseline_hash="2" * 64
    )
    value["builder_audit"][field] = bad_value
    path = tmp_path / f"r32-{field}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ContractError, match="R32 exact causal replay audit changed"):
        validate_raw_checkpoint(
            path,
            row=row,
            mode="tail_refresh_r32",
            request_fingerprint="f" * 64,
            method_binding_sha256="d" * 64,
            expected_native_file_sha256="1" * 64,
            expected_baseline_file_sha256="2" * 64,
            tokenizer=_Tokenizer("answer"),
        )
