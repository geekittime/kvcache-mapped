"""Atomic raw-result and derived two-pass-result validation."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .contracts import (
    ContractError,
    RAW_MODES,
    canonical_sha256,
    exact_json_equal,
    finite_float,
    read_json,
    strict_int,
    token_ids_sha256,
)
from .metrics import score_answer


GENERATION_FIELDS = {
    "requested_max_tokens",
    "generated_tokens",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "decoded_generated_token_ids",
    "stop_reason",
    "terminal_token_id",
    "frozen_eos_token_ids",
    "stop_sequences",
    "cap_hit",
}
R32_AUDIT_FIELDS = {
    "builder_prefix_tokens",
    "full_prompt_tokens",
    "requested_total_native_prompt_tokens",
    "mapped_prefix_tokens",
    "builder_native_tail_tokens",
    "engine_boundary_tokens",
    "mapped_token_slice",
    "native_tail_token_slice",
    "native_tail_position_ids",
    "engine_boundary_position_id",
    "target_forward_calls",
    "target_forward_query_tokens",
    "attention_mask_shape",
    "position_ids",
    "cache_position",
    "cache_length_before_refresh",
    "cache_length_after_refresh",
    "r1_returned_underlying_cache_object_directly",
}


def _strict_int_invalid(value: Any, *, minimum: int | None = None) -> bool:
    try:
        strict_int(value, label="integer evidence", minimum=minimum)
    except ContractError:
        return True
    return False


def expected_r32_builder_audit(prompt_token_count: int) -> dict[str, Any]:
    """Recompute the exact frozen R32 builder audit for one prompt.

    ``TransferEvaluationEngine`` owns the final causal boundary token, so the
    builder receives ``C[:-1]``.  It maps ``P-32`` tokens and natively appends
    positions ``P-32 .. P-2`` before the engine replays position ``P-1``.
    """

    try:
        strict_int(prompt_token_count, label="R32 prompt token count", minimum=33)
    except ContractError as exc:
        raise ContractError("R32 prompt must contain more than 32 tokens") from exc
    mapped = prompt_token_count - 32
    builder_prefix = prompt_token_count - 1
    native_positions = list(range(mapped, builder_prefix))
    value = {
        "builder_prefix_tokens": builder_prefix,
        "full_prompt_tokens": prompt_token_count,
        "requested_total_native_prompt_tokens": 32,
        "mapped_prefix_tokens": mapped,
        "builder_native_tail_tokens": 31,
        "engine_boundary_tokens": 1,
        "mapped_token_slice": [0, mapped],
        "native_tail_token_slice": [mapped, builder_prefix],
        "native_tail_position_ids": native_positions,
        "engine_boundary_position_id": builder_prefix,
        "target_forward_calls": 1,
        "target_forward_query_tokens": 31,
        "attention_mask_shape": [1, builder_prefix],
        "position_ids": native_positions,
        "cache_position": native_positions,
        "cache_length_before_refresh": mapped,
        "cache_length_after_refresh": builder_prefix,
        "r1_returned_underlying_cache_object_directly": False,
    }
    if set(value) != R32_AUDIT_FIELDS:  # pragma: no cover - developer invariant
        raise AssertionError("R32 audit implementation/schema diverged")
    return value


def result_path(output: str | Path, row: Mapping[str, Any], mode: str, *, derived: bool = False) -> Path:
    identity = canonical_sha256([row["dataset_id"], row["sample_id"]])[:24]
    group = "derived" if derived else "raw"
    return Path(output) / "results" / group / str(row["dataset_id"]) / mode / f"{identity}.json"


def validate_generation(
    generation: Any,
    *,
    expected_cap: int,
    stop_sequences: Sequence[str],
    prediction: str,
    tokenizer: Any,
) -> dict[str, Any]:
    if not isinstance(generation, Mapping) or set(generation) != GENERATION_FIELDS:
        raise ContractError("generation evidence schema changed")
    ids = generation.get("generated_token_ids")
    if not isinstance(ids, list):
        raise ContractError("generated token IDs are malformed")
    try:
        for item in ids:
            strict_int(item, label="generated token ID", minimum=0)
        generated_tokens = strict_int(
            generation.get("generated_tokens"),
            label="generated token count",
            minimum=0,
        )
        strict_int(expected_cap, label="expected generation cap", minimum=1)
        requested_max_tokens = strict_int(
            generation.get("requested_max_tokens"),
            label="requested generation cap",
            minimum=1,
        )
    except ContractError as exc:
        raise ContractError("generated token/count evidence is malformed") from exc
    if generated_tokens != len(ids):
        raise ContractError("generated token count/list mismatch")
    if generation.get("generated_token_ids_sha256") != token_ids_sha256(ids):
        raise ContractError("generated token hash changed")
    if requested_max_tokens != expected_cap:
        raise ContractError("generation cap changed")
    if not exact_json_equal(generation.get("stop_sequences"), list(stop_sequences)):
        raise ContractError("stop sequences changed")
    reason = generation.get("stop_reason")
    terminal_token_id = generation.get("terminal_token_id")
    cap_hit = generation.get("cap_hit")
    if reason not in {"eos", "stop_sequence", "max_tokens"} or type(cap_hit) is not bool:
        raise ContractError("stop evidence is malformed")
    raw_eos_ids = getattr(tokenizer, "_prospective_frozen_eos_token_ids", None)
    if (
        not isinstance(raw_eos_ids, tuple)
        or not raw_eos_ids
        or any(
            _strict_int_invalid(item, minimum=0)
            for item in raw_eos_ids
        )
        or tuple(sorted(set(raw_eos_ids))) != raw_eos_ids
    ):
        raise ContractError("frozen EOS token IDs are unavailable or malformed")
    eos_ids = set(raw_eos_ids)
    if not exact_json_equal(
        generation.get("frozen_eos_token_ids"), list(raw_eos_ids)
    ):
        raise ContractError("generation EOS set differs from frozen target metadata")
    if any(item in eos_ids for item in ids):
        raise ContractError("persisted generated IDs contain a frozen EOS token")
    if reason == "max_tokens":
        if (
            len(ids) != expected_cap
            or cap_hit is not True
            or terminal_token_id is not None
        ):
            raise ContractError("max-token stop is inconsistent")
    elif reason == "eos":
        if (
            len(ids) >= expected_cap
            or cap_hit is not False
            or _strict_int_invalid(terminal_token_id, minimum=0)
            or terminal_token_id not in eos_ids
        ):
            # The locally frozen engine records the actual greedy argmax EOS
            # token and excludes it from generated_token_ids.  This closes
            # the evidence gap where an arbitrary short sequence could merely
            # claim an EOS reason.
            raise ContractError("EOS terminal-token evidence is inconsistent")
    elif (
        len(ids) > expected_cap
        or cap_hit is not False
        or terminal_token_id is not None
    ):
        # A stop-string token is already included in generated_token_ids; no
        # separate/excluded terminal token exists for this reason.
        raise ContractError("stop-sequence terminal evidence is inconsistent")
    decoded = generation.get("decoded_generated_token_ids")
    if not isinstance(decoded, str):
        raise ContractError("decoded continuation is missing")
    try:
        observed = str(tokenizer.decode(ids, skip_special_tokens=True))
    except TypeError:
        observed = str(tokenizer.decode(ids))
    if observed != decoded:
        raise ContractError("decoded continuation does not match token IDs")
    marker_offsets = [
        (decoded.find(stop), stop)
        for stop in stop_sequences
        if decoded.find(stop) >= 0
    ]
    if reason == "stop_sequence":
        if not marker_offsets:
            raise ContractError("stop_sequence reason lacks registered marker")
        offset, marker = min(marker_offsets, key=lambda item: item[0])
        # The engine stops as soon as decoded text contains a registered
        # marker.  A single tokenizer piece can contain the marker plus a
        # suffix, so token-exact evidence may legitimately decode past the
        # character boundary even though no later generation step ran.
        if prediction != decoded[:offset] or not decoded[offset:].startswith(marker):
            raise ContractError("prediction is not trimmed at exact first stop boundary")
    else:
        if marker_offsets:
            raise ContractError("non-stop-sequence termination contains a registered marker")
        if prediction != decoded:
            raise ContractError("prediction differs from decoded continuation")
    return dict(generation)


def _validate_cost(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "wall_seconds",
        "phase_timing",
        "peak_gpu_allocated_bytes",
    }:
        raise ContractError("runtime cost evidence schema changed")
    if finite_float(value.get("wall_seconds"), label="runtime_cost.wall_seconds") < 0:
        raise ContractError("runtime time cannot be negative")
    phase = value.get("phase_timing")
    if not isinstance(phase, Mapping) or set(phase) != {
        "available",
        "prefill_seconds",
        "decode_seconds",
        "scope",
    }:
        raise ContractError("runtime phase timing schema changed")
    if not exact_json_equal(dict(phase), {
        "available": False,
        "prefill_seconds": None,
        "decode_seconds": None,
        "scope": "monolithic_engine_generate_token_ids",
    }):
        raise ContractError("runtime phase timing evidence changed")
    peaks = value.get("peak_gpu_allocated_bytes")
    if not isinstance(peaks, Mapping) or set(peaks) != {str(index) for index in range(8)}:
        raise ContractError("runtime peak VRAM must cover exact GPU keys 0..7")
    try:
        for device, item in peaks.items():
            strict_int(item, label=f"runtime peak VRAM GPU {device}", minimum=0)
    except ContractError as exc:
        raise ContractError("runtime peak VRAM evidence is malformed") from exc
    return dict(value)


def validate_raw_checkpoint(
    path: str | Path,
    *,
    row: Mapping[str, Any],
    mode: str,
    request_fingerprint: str,
    method_binding_sha256: str,
    expected_execution_attempt_id: str | None = None,
    tokenizer: Any = None,
    expected_native_file_sha256: str | None = None,
    expected_baseline_file_sha256: str | None = None,
) -> dict[str, Any]:
    if mode not in RAW_MODES:
        raise ContractError(f"not a raw mode: {mode}")
    result_path_value = Path(path)
    if result_path_value.is_symlink() or not result_path_value.is_file():
        raise ContractError(f"raw checkpoint is missing/not regular: {path}")
    result = read_json(result_path_value)
    if not isinstance(result, dict):
        raise ContractError("raw checkpoint root is malformed")
    expected_fields = {
        "schema_version",
        "status",
        "result_kind",
        "request_fingerprint",
        "dataset_id",
        "sample_id",
        "sample_binding_sha256",
        "source_projection_sha256",
        "prompt_token_ids_sha256",
        "mode",
        "method_binding_sha256",
        "execution_attempt_id",
        "references",
        "truncation",
        "prediction",
        "generation",
        "metrics",
        "runtime_cost",
        "builder_audit",
        "comparison_bindings",
    }
    if set(result) != expected_fields:
        raise ContractError("raw checkpoint field set changed")
    fixed = {
        "schema_version": 2,
        "status": "complete",
        "result_kind": "raw_model_run",
        "request_fingerprint": request_fingerprint,
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "sample_binding_sha256": row["sample_binding_sha256"],
        "source_projection_sha256": row["source_projection_sha256"],
        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        "mode": mode,
        "method_binding_sha256": method_binding_sha256,
        "references": row["references"],
        "truncation": row["truncation"],
    }
    for field, expected in fixed.items():
        if not exact_json_equal(result.get(field), expected):
            raise ContractError(f"raw checkpoint {field} changed: {path}")
    attempt_id = result.get("execution_attempt_id")
    if (
        not isinstance(attempt_id, str)
        or re.fullmatch(r"attempt_[0-9]{4,}", attempt_id) is None
        or (
            expected_execution_attempt_id is not None
            and attempt_id != expected_execution_attempt_id
        )
    ):
        raise ContractError("raw checkpoint execution attempt binding changed")
    prediction = result.get("prediction")
    if not isinstance(prediction, str):
        raise ContractError("raw prediction is not text")
    if tokenizer is None:
        raise ContractError("raw checkpoint validation requires the frozen tokenizer")
    validate_generation(
        result.get("generation"),
        expected_cap=row["generation_cap"],
        stop_sequences=row["stop_sequences"],
        prediction=prediction,
        tokenizer=tokenizer,
    )
    if not exact_json_equal(
        result.get("metrics"),
        score_answer(row["dataset_id"], prediction, row["references"]),
    ):
        raise ContractError("raw metrics do not recompute")
    _validate_cost(result.get("runtime_cost"))
    expected_comparison = {
        "native14_raw_file_sha256": None if mode == "native14" else expected_native_file_sha256,
        "nvidia_k20_raw_file_sha256": (
            expected_baseline_file_sha256 if mode in {"shrinkage", "tail_refresh_r32"} else None
        ),
    }
    if not exact_json_equal(result.get("comparison_bindings"), expected_comparison):
        raise ContractError("raw checkpoint comparison bindings changed")
    audit = result.get("builder_audit")
    if mode == "tail_refresh_r32":
        if not isinstance(audit, Mapping):
            raise ContractError("R32 checkpoint lacks causal builder audit")
        expected_audit = expected_r32_builder_audit(row["prompt_token_count"])
        if not exact_json_equal(dict(audit), expected_audit):
            raise ContractError("R32 exact causal replay audit changed")
    elif audit is not None:
        raise ContractError("non-R32 raw result unexpectedly has builder audit")
    return result


def raw_result_sha256(result: Mapping[str, Any]) -> str:
    return canonical_sha256(result)


__all__ = [
    "GENERATION_FIELDS",
    "R32_AUDIT_FIELDS",
    "expected_r32_builder_audit",
    "raw_result_sha256",
    "result_path",
    "validate_generation",
    "validate_raw_checkpoint",
]
