from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import (
    ContractError,
    ensure_no_symlink_components,
    is_int,
    load_json_strict,
    require_exact_keys,
    require_finite_number,
    require_int,
    validate_artifact_ref,
)


FORBIDDEN_RECORD_KEYS = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "gold_answer",
        "prediction",
        "predictions",
        "metric",
        "metrics",
        "score",
        "scores",
        "outcome",
        "outcomes",
        "qa_f1",
        "qa_em",
        "accuracy",
        "rouge",
        "chrf",
        "normalized_token_f1",
        "locked_test_outcome",
    }
)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def _hex_digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def validate_registry_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate metadata-only three-tier registry without opening row payloads."""

    require_exact_keys(document, ("schema_version", "kind", "records"), "tier registry")
    if document["schema_version"] != 1 or document["kind"] != "three_tier_sequence_registry_v1":
        raise ContractError("unsupported tier registry schema")
    records = document["records"]
    if not isinstance(records, list) or not records:
        raise ContractError("tier registry records must be a nonempty list")
    required_record_keys = (
        "sequence_id",
        "group_id",
        "tier",
        "split",
        "dataset",
        "normalized_text_sha256",
        "source_projection_sha256",
        "partition",
    )
    optional_record_keys = {"source_index", "selection_rank", "stratum"}
    allowed_tiers = {"mapper-fit", "method-dev", "locked-test"}
    seen_ids: set[str] = set()
    groups_by_tier: dict[str, set[str]] = {tier: set() for tier in allowed_tiers}
    by_tier: dict[str, int] = {tier: 0 for tier in allowed_tiers}
    hashes_by_tier: dict[str, set[str]] = {tier: set() for tier in allowed_tiers}
    projections_by_tier: dict[str, set[str]] = {tier: set() for tier in allowed_tiers}
    for index, record in enumerate(records):
        name = f"tier registry record[{index}]"
        if not isinstance(record, Mapping):
            raise ContractError(f"{name} must be an object")
        if not set(required_record_keys).issubset(record) or not set(record).issubset(set(required_record_keys) | optional_record_keys):
            missing = sorted(set(required_record_keys) - set(record))
            extra = sorted(set(record) - (set(required_record_keys) | optional_record_keys))
            raise ContractError(f"{name} key mismatch; missing={missing}, extra={extra}")
        # The exact schema above is intentionally redundant with this recursive
        # check: future extensions cannot smuggle answer/outcome fields in a
        # nested object without a protocol revision.
        forbidden = sorted(set(_walk_keys(record)) & FORBIDDEN_RECORD_KEYS)
        if forbidden:
            raise ContractError(f"{name} contains forbidden answer/outcome keys: {forbidden}")
        for key in ("sequence_id", "group_id", "partition", "dataset", "split", "tier"):
            if not isinstance(record[key], str) or not record[key]:
                raise ContractError(f"{name}.{key} must be a nonempty string")
        sequence_id = record["sequence_id"]
        group_id = record["group_id"]
        if sequence_id in seen_ids:
            raise ContractError(f"duplicate sequence_id: {sequence_id}")
        seen_ids.add(sequence_id)
        tier = record["tier"]
        if tier not in allowed_tiers:
            raise ContractError(f"unsupported tier: {tier}")
        if record["split"] != "train" and tier != "locked-test":
            raise ContractError(f"non-train split in non-locked tier: {name}")
        if "source_index" in record and (not is_int(record["source_index"]) or record["source_index"] < 0):
            raise ContractError(f"{name}.source_index must be a nonnegative integer")
        if "selection_rank" in record and (not is_int(record["selection_rank"]) or record["selection_rank"] < 0):
            raise ContractError(f"{name}.selection_rank must be a nonnegative integer")
        if "stratum" in record and (not isinstance(record["stratum"], str) or not record["stratum"]):
            raise ContractError(f"{name}.stratum must be a nonempty string")
        _hex_digest(record["normalized_text_sha256"], f"{name}.normalized_text_sha256")
        _hex_digest(record["source_projection_sha256"], f"{name}.source_projection_sha256")
        by_tier[tier] += 1
        groups_by_tier[tier].add(group_id)
        hashes_by_tier[tier].add(record["normalized_text_sha256"])
        projections_by_tier[tier].add(record["source_projection_sha256"])

    for left in allowed_tiers:
        for right in allowed_tiers:
            if left >= right:
                continue
            overlap = hashes_by_tier[left] & hashes_by_tier[right]
            if overlap:
                raise ContractError(f"cross-tier normalized text collision {left}/{right}: {len(overlap)}")
            overlap = projections_by_tier[left] & projections_by_tier[right]
            if overlap:
                raise ContractError(f"cross-tier projection collision {left}/{right}: {len(overlap)}")
            overlap = groups_by_tier[left] & groups_by_tier[right]
            if overlap:
                raise ContractError(f"cross-tier group collision {left}/{right}: {len(overlap)}")
    return {
        "record_count": len(records),
        "counts_by_tier": by_tier,
        "sequence_group_count": sum(len(groups) for groups in groups_by_tier.values()),
        "cross_tier_exact_collisions": 0,
    }


def validate_capture_manifest(document: Mapping[str, Any], *, expected_tier: str) -> dict[str, Any]:
    """Validate a metadata manifest; payload files are checked by ArtifactRef."""

    require_exact_keys(
        document,
        (
            "schema_version",
            "kind",
            "status",
            "tier",
            "teacher",
            "source_logits_as_teacher_allowed",
            "records",
            "sequence_group_ids_sha256",
        ),
        "functional capture manifest",
    )
    if document["schema_version"] != 1 or document["kind"] != "functional_capture_manifest_v1":
        raise ContractError("unsupported functional capture schema")
    if document["status"] != "PUBLISHED" or document["tier"] != expected_tier:
        raise ContractError("capture is not a published manifest for the expected tier")
    if document["teacher"] != "native_qwen3_14b" or document["source_logits_as_teacher_allowed"] is not False:
        raise ContractError("capture teacher contract is invalid")
    _hex_digest(document["sequence_group_ids_sha256"], "capture.sequence_group_ids_sha256")
    records = document["records"]
    if not isinstance(records, list) or not records:
        raise ContractError("capture records must be nonempty")
    seen: set[str] = set()
    for index, record in enumerate(records):
        name = f"capture record[{index}]"
        if not isinstance(record, Mapping):
            raise ContractError(f"{name} must be an object")
        require_exact_keys(
            record,
            (
                "sequence_id",
                "group_id",
                "source_features",
                "native_cache",
                "anchor_cache",
                "functional_signals",
            ),
            name,
        )
        forbidden = sorted(set(_walk_keys(record)) & FORBIDDEN_RECORD_KEYS)
        if forbidden:
            raise ContractError(f"{name} contains forbidden keys: {forbidden}")
        for field in ("sequence_id", "group_id"):
            if not isinstance(record[field], str) or not record[field]:
                raise ContractError(f"{name}.{field} must be a string")
        if record["group_id"] in seen:
            raise ContractError(f"duplicate capture group: {record['group_id']}")
        seen.add(record["group_id"])
        for field in ("source_features", "native_cache", "anchor_cache"):
            validate_artifact_ref(record[field], f"{name}.{field}")
        signals = record["functional_signals"]
        if not isinstance(signals, Mapping):
            raise ContractError(f"{name}.functional_signals must be an object")
        required_signal_keys = {
            "native_fixed_queries",
            "attention_row_masks",
            "native_suffix_logits",
            "suffix_token_ids",
            "eos_token_ids",
            "positions",
        }
        if set(signals) != required_signal_keys:
            raise ContractError(f"{name}.functional_signals key set mismatch")
        for signal_name in ("native_fixed_queries", "attention_row_masks", "native_suffix_logits", "suffix_token_ids", "positions"):
            validate_artifact_ref(signals[signal_name], f"{name}.functional_signals.{signal_name}")
        eos = signals["eos_token_ids"]
        if not isinstance(eos, list) or not eos or any(not is_int(token) or token < 0 for token in eos):
            raise ContractError(f"{name}.eos_token_ids must be nonempty integer IDs")
    return {"record_count": len(records), "sequence_group_count": len(seen)}


def validate_answer_free_oracle_receipt(
    receipt: Mapping[str, Any], *, expected_candidate_layers: Sequence[int]
) -> dict[str, Any]:
    """Validate the exact layer-selection interface used by this method."""

    require_exact_keys(
        receipt,
        (
            "schema_version",
            "kind",
            "status",
            "tier",
            "answers_opened_for_selection",
            "model_outcomes_opened_for_selection",
            "candidate_layers",
            "active_layers",
            "layer_loss_weights",
            "selection_metrics",
            "selection_rule",
            "bootstrap_seed",
            "bootstrap_replicates",
            "source_registry_ref",
        ),
        "layer oracle receipt",
    )
    if receipt["schema_version"] != 1 or receipt["kind"] != "answer_free_layer_oracle_receipt_v1":
        raise ContractError("unsupported oracle receipt schema")
    if receipt["status"] != "PUBLISHED" or receipt["tier"] != "method-dev":
        raise ContractError("oracle receipt is not published method-dev")
    if receipt["answers_opened_for_selection"] is not False or receipt["model_outcomes_opened_for_selection"] is not False:
        raise ContractError("oracle selection must be answer/outcome free")
    expected = list(expected_candidate_layers)
    if receipt["candidate_layers"] != expected:
        raise ContractError("oracle candidate layer order mismatch")
    active = receipt["active_layers"]
    if not isinstance(active, list) or not active or active != sorted(set(active)):
        raise ContractError("active_layers must be sorted unique nonempty list")
    if any(layer not in expected for layer in active):
        raise ContractError("oracle selected a layer outside candidate catalog")
    weights = receipt["layer_loss_weights"]
    if not isinstance(weights, Mapping) or set(weights) != {str(layer) for layer in active}:
        raise ContractError("layer_loss_weights must use ordered string layer keys")
    numeric = []
    for layer in active:
        numeric.append(require_finite_number(weights[str(layer)], f"layer_loss_weights[{layer}]"))
    if any(not 0.25 <= value <= 2.0 for value in numeric):
        raise ContractError("oracle layer weights outside [0.25,2]")
    if abs(sum(numeric) / len(numeric) - 1.0) > 1e-6:
        raise ContractError("oracle layer weights must have arithmetic mean one")
    metrics = receipt["selection_metrics"]
    allowlist = {
        "native_repair_first_token_kl_reduction",
        "native_fixed_query_attention_output_cosine_gain",
        "native_fixed_query_attention_logit_huber_reduction",
    }
    if not isinstance(metrics, list) or not metrics or any(metric not in allowlist for metric in metrics):
        raise ContractError("oracle metrics are not answer-free allowlisted metrics")
    if not isinstance(receipt["selection_rule"], str) or "bootstrap" not in receipt["selection_rule"]:
        raise ContractError("oracle selection rule is not the frozen bootstrap rule")
    require_int(receipt["bootstrap_seed"], "oracle.bootstrap_seed", minimum=0)
    if receipt["bootstrap_replicates"] != 10000:
        raise ContractError("oracle bootstrap replicate count must be 10000")
    validate_artifact_ref(receipt["source_registry_ref"], "oracle.source_registry_ref")
    return {"active_layers": active, "layer_loss_weights": dict(weights), "metric_count": len(metrics)}


def validate_no_answers_or_outcomes(value: Any, *, name: str = "document") -> None:
    forbidden = sorted(set(_walk_keys(value)) & FORBIDDEN_RECORD_KEYS)
    if forbidden:
        raise ContractError(f"{name} contains forbidden answer/outcome keys: {forbidden}")
