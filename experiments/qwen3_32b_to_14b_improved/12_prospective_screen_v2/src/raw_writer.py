"""Narrow CPU-side writer API used by the real GPU runner.

The runner owns model execution.  It must hand exact generation token IDs,
decoded continuation, timing, peak VRAM, and (for R32) causal builder audit to
this module; the writer computes metrics and freezes one atomic checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .checkpoints import result_path, validate_generation, validate_raw_checkpoint
from .contracts import ContractError, RAW_MODES, ArtifactRef, immutable_json, write_sidecar
from .metrics import score_answer


def build_raw_checkpoint(
    *,
    row: Mapping[str, Any],
    mode: str,
    request_fingerprint: str,
    method_binding_sha256: str,
    execution_attempt_id: str,
    prediction: str,
    generation: Mapping[str, Any],
    runtime_cost: Mapping[str, Any],
    builder_audit: Mapping[str, Any] | None,
    native14_raw_file_sha256: str | None,
    nvidia_k20_raw_file_sha256: str | None,
    tokenizer: Any,
) -> dict[str, Any]:
    if mode not in RAW_MODES:
        raise ContractError(f"not a registered raw mode: {mode}")
    validate_generation(
        generation,
        expected_cap=row["generation_cap"],
        stop_sequences=row["stop_sequences"],
        prediction=prediction,
        tokenizer=tokenizer,
    )
    return {
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
        "execution_attempt_id": execution_attempt_id,
        "references": row["references"],
        "truncation": row["truncation"],
        "prediction": prediction,
        "generation": dict(generation),
        "metrics": score_answer(row["dataset_id"], prediction, row["references"]),
        "runtime_cost": dict(runtime_cost),
        "builder_audit": dict(builder_audit) if builder_audit is not None else None,
        "comparison_bindings": {
            "native14_raw_file_sha256": native14_raw_file_sha256,
            "nvidia_k20_raw_file_sha256": nvidia_k20_raw_file_sha256,
        },
    }


def write_raw_checkpoint(
    *,
    output_dir: str | Path,
    row: Mapping[str, Any],
    mode: str,
    request_fingerprint: str,
    method_binding_sha256: str,
    execution_attempt_id: str,
    prediction: str,
    generation: Mapping[str, Any],
    runtime_cost: Mapping[str, Any],
    builder_audit: Mapping[str, Any] | None,
    native14_raw_file_sha256: str | None,
    nvidia_k20_raw_file_sha256: str | None,
    tokenizer: Any,
) -> ArtifactRef:
    value = build_raw_checkpoint(
        row=row,
        mode=mode,
        request_fingerprint=request_fingerprint,
        method_binding_sha256=method_binding_sha256,
        execution_attempt_id=execution_attempt_id,
        prediction=prediction,
        generation=generation,
        runtime_cost=runtime_cost,
        builder_audit=builder_audit,
        native14_raw_file_sha256=native14_raw_file_sha256,
        nvidia_k20_raw_file_sha256=nvidia_k20_raw_file_sha256,
        tokenizer=tokenizer,
    )
    path = result_path(output_dir, row, mode)
    reference = immutable_json(path, value)
    write_sidecar(reference)
    validate_raw_checkpoint(
        path,
        row=row,
        mode=mode,
        request_fingerprint=request_fingerprint,
        method_binding_sha256=method_binding_sha256,
        expected_execution_attempt_id=execution_attempt_id,
        tokenizer=tokenizer,
        expected_native_file_sha256=native14_raw_file_sha256,
        expected_baseline_file_sha256=nvidia_k20_raw_file_sha256,
    )
    return reference


__all__ = ["build_raw_checkpoint", "write_raw_checkpoint"]
