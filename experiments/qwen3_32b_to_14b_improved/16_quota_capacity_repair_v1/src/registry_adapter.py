"""Answer-free bridge from Stage-D rows to the later native-registry adapter.

This module intentionally stops one step before a native registry: pairwise-v3
records contain ``source_index`` and a projection hash but do not expose the
raw sample ID.  A later dir10/12 consumer, after receiving its own pinned
``train_sources`` and exact ``answer_access_policy`` receipt, resolves that ID
and recomputes the projection.  Fabricating an ID here would weaken the
decontamination boundary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .common import ProtocolError


_FORBIDDEN_INPUT_FIELDS = {
    "sample_id", "answer", "answers", "support", "supporting_facts",
    "gold", "prediction", "predictions", "model_output", "model_outputs",
    "outcome", "outcomes", "metric", "metrics", "logits", "loss",
}
_INPUT_FIELDS = {
    "target_index", "target_selection_rank", "partition", "dataset_id",
    "stratum", "record_index", "source_projection_sha256", "objective_cost_sha256",
}
_ROW_FIELDS = {
    "dataset_id", "partition", "source_index", "source_projection_sha256",
    "selection_rank", "stratum",
}
_SPLIT_NAMES = {
    ("hotpotqa_train", "screen"): "hotpotqa_screen",
    ("hotpotqa_train", "confirm"): "hotpotqa_confirm",
    ("musique_train", "screen"): "musique_screen",
    ("musique_train", "confirm"): "musique_confirm",
}
_STRATA_ORDER = {
    "hotpotqa_train": ("bridge/easy", "bridge/medium", "bridge/hard", "comparison/easy", "comparison/medium", "comparison/hard"),
    "musique_train": ("2hop", "3hop", "4hop"),
}


def _hash(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ProtocolError("source projection hash is not lowercase SHA-256")
    return value


def project_selection_to_native_interface(
    selection: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Project selection rows into four split payloads without raw IDs.

    The returned rows intentionally have six fields (the native contract's
    ``sample_id`` and ``selection_rank_sha256`` are downstream-owned).  Ranks
    are zero-based and contiguous within each split/stratum, ordered by the
    Stage-D SHA rank and then source index.
    """

    if not isinstance(selection, Sequence) or not isinstance(records, Sequence):
        raise ProtocolError("selection and records must be sequences")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_records: set[int] = set()
    for row in selection:
        if not isinstance(row, Mapping) or set(row) != _INPUT_FIELDS:
            raise ProtocolError("selection row fields do not match answer-free interface")
        if type(row.get("target_index")) is not int or type(row.get("target_selection_rank")) is not int or row["target_selection_rank"] < 0:
            raise ProtocolError("selection target/rank type mismatch")
        if not isinstance(row.get("objective_cost_sha256"), str) or len(row["objective_cost_sha256"]) != 64 or any(char not in "0123456789abcdef" for char in row["objective_cost_sha256"]):
            raise ProtocolError("selection objective hash type mismatch")
        index = row.get("record_index")
        if type(index) is not int or not 0 <= index < len(records) or index in seen_records:
            raise ProtocolError("selection record index is invalid or duplicated")
        seen_records.add(index)
        if row.get("partition") not in {"screen", "confirm"} or row.get("dataset_id") not in {"hotpotqa_train", "musique_train"} or not isinstance(row.get("stratum"), str):
            raise ProtocolError("selection dataset/partition/stratum mismatch")
        record = records[index]
        if (
            row["dataset_id"] != record.get("dataset_id")
            or row["stratum"] != record.get("stratum")
            or row["source_projection_sha256"] != record.get("source_projection_sha256")
        ):
            raise ProtocolError("selection row does not bind its pairwise-v3 record")
        projection_hash = _hash(row["source_projection_sha256"])
        split = _SPLIT_NAMES.get((row["dataset_id"], row["partition"]))
        if split is None:
            raise ProtocolError("unsupported dataset/partition split")
        source_index = record.get("source_index")
        if type(source_index) is not int or source_index < 0:
            raise ProtocolError("record source_index is not a nonnegative integer")
        groups[split].append({
            "dataset_id": row["dataset_id"],
            "partition": row["partition"],
            "source_index": source_index,
            "source_projection_sha256": projection_hash,
            "selection_rank": row["target_selection_rank"],
            "stratum": row["stratum"],
            "_record_index": index,
        })
    # Every four split key is present even when a future repaired quota is zero.
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in ("hotpotqa_screen", "hotpotqa_confirm", "musique_screen", "musique_confirm"):
        rows = groups.get(split, [])
        dataset_id = rows[0]["dataset_id"] if rows else ("musique_train" if split.startswith("musique") else "hotpotqa_train")
        # Native order is frozen strata order, then rank, then source index.
        stratum_order = {value: i for i, value in enumerate(_STRATA_ORDER[dataset_id])}
        rows.sort(key=lambda row: (stratum_order.get(row["stratum"], 10**9), row["selection_rank"], row["source_index"], row["_record_index"]))
        if any(row["stratum"] not in stratum_order for row in rows):
            raise ProtocolError(f"unknown stratum in {split}")
        # Convert target-global SHA ranks to the native stratum-local contiguous
        # rank while retaining order; this is the exact rank convention expected
        # by dir12/13.
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_stratum[row["stratum"]].append(row)
        output_rows: list[dict[str, Any]] = []
        for stratum in _STRATA_ORDER[dataset_id]:
            for rank, row in enumerate(by_stratum.get(stratum, [])):
                output_rows.append({
                    key: row[key]
                    for key in _ROW_FIELDS
                    if key != "selection_rank"
                } | {"selection_rank": rank})
        # Verify the exact field set and contiguous rank invariant.
        for row in output_rows:
            if set(row) != _ROW_FIELDS:
                raise ProtocolError("intermediate native row field mismatch")
        for stratum in _STRATA_ORDER[dataset_id]:
            ranks = [row["selection_rank"] for row in output_rows if row["stratum"] == stratum]
            if ranks != list(range(len(ranks))):
                raise ProtocolError("intermediate native ranks are not contiguous")
        splits[split] = output_rows
    screen_sources = {
        dataset: {
            partition: {
                "source_indices": {row["source_index"] for row in splits[name]},
                "projection_hashes": {row["source_projection_sha256"] for row in splits[name]},
            }
            for partition, name in (("screen", f"{dataset}_screen"), ("confirm", f"{dataset}_confirm"))
        }
        for dataset in ("hotpotqa", "musique")
    }
    disjoint = {
        dataset: not (
            screen_sources[dataset]["screen"]["source_indices"] & screen_sources[dataset]["confirm"]["source_indices"]
            or screen_sources[dataset]["screen"]["projection_hashes"] & screen_sources[dataset]["confirm"]["projection_hashes"]
        )
        for dataset in screen_sources
    }
    if not all(disjoint.values()):
        raise ProtocolError("screen/confirm source overlap in native projection interface")
    return {
        "schema_version": 1,
        "status": "INTERMEDIATE_NATIVE_REGISTRY_INTERFACE",
        "claim_boundary": "no sample IDs and no answers; downstream adapter must complete native rows",
        "splits": splits,
        "split_row_counts": {name: len(rows) for name, rows in splits.items()},
        "selection_rank_semantics": "zero-based contiguous within each split stratum; Stage-D SHA order preserved",
        "screen_confirm_disjoint": disjoint,
        "requires_downstream_fields": ["sample_id", "selection_rank_sha256"],
        "train_sources_receipt_passthrough": True,
        "answer_access_policy_receipt_passthrough": True,
        "answer_values_read": False,
        "model_outcomes_read": False,
    }


# Short alias used by downstream experiments.
selection_to_registry_interface = project_selection_to_native_interface
