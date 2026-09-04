"""Direct consumer for experiment 10's native append-only publication."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ArtifactRef,
    ContractError,
    DATASETS,
    PARTITIONS,
    SHA256_RE,
    SPLIT_KEYS,
    read_json,
    read_jsonl,
)
from .configuration import validated_profile_counts


NATIVE_ROW_KEYS = {
    "dataset_id",
    "partition",
    "sample_id",
    "source_index",
    "source_projection_sha256",
    "selection_rank",
    "selection_rank_sha256",
    "stratum",
}
PUBLISHED_MARKER_KEYS = {
    "schema_version",
    "registry_id",
    "status",
    "atomic_last_completion_marker",
    "published_files",
    "ordered_inventory_sha256",
    "receipt",
    "receipt_sidecar",
}
CORE_FILENAMES = {
    "hotpotqa_screen.jsonl",
    "hotpotqa_confirm.jsonl",
    "musique_screen.jsonl",
    "musique_confirm.jsonl",
    "quarantine_aggregate.json",
    "registry_manifest.json",
    "receipt.json",
}
PUBLISHED_FILENAMES = CORE_FILENAMES | {
    f"{name}.sha256.json" for name in CORE_FILENAMES
}


def _ref(value: Any, label: str, *, verify: bool = True) -> ArtifactRef:
    reference = ArtifactRef.parse(value, label=label)
    if verify:
        reference.verify(label=label)
    return reference


def _publisher_inventory_digest(published_files: Mapping[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for name in sorted(published_files):
        entry = published_files[name]
        if not isinstance(entry, Mapping) or set(entry) != {"nbytes", "sha256"}:
            raise ContractError(f"PUBLISHED file inventory entry is malformed: {name}")
        nbytes = entry.get("nbytes")
        digest = entry.get("sha256")
        if type(nbytes) is not int or nbytes < 0:
            raise ContractError(f"PUBLISHED file nbytes is malformed: {name}")
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise ContractError(f"PUBLISHED file SHA-256 is malformed: {name}")
        rows.append({"name": name, "nbytes": nbytes, "sha256": digest})
    payload = (
        json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_split_contract(config: Mapping[str, Any], key: str) -> tuple[str, str, int]:
    prefix, partition = key.rsplit("_", 1)
    dataset = f"{prefix}_train"
    return dataset, partition, validated_profile_counts(config["profiles"])[partition][dataset]


def _validate_native_row(
    row: Mapping[str, Any],
    *,
    label: str,
    dataset: str,
    partition: str,
    allowed_strata: list[str],
) -> dict[str, Any]:
    if set(row) != NATIVE_ROW_KEYS:
        raise ContractError(
            f"{label} native key set changed: {sorted(set(row) ^ NATIVE_ROW_KEYS)}"
        )
    if row.get("dataset_id") != dataset or row.get("partition") != partition:
        raise ContractError(f"{label} dataset/partition changed")
    sample_id = row.get("sample_id")
    source_index = row.get("source_index")
    projection = row.get("source_projection_sha256")
    rank = row.get("selection_rank")
    rank_digest = row.get("selection_rank_sha256")
    stratum = row.get("stratum")
    if not isinstance(sample_id, str) or not sample_id:
        raise ContractError(f"{label}.sample_id is invalid")
    if type(source_index) is not int or source_index < 0:
        raise ContractError(f"{label}.source_index is invalid")
    if not isinstance(projection, str) or SHA256_RE.fullmatch(projection) is None:
        raise ContractError(f"{label}.source_projection_sha256 is invalid")
    if type(rank) is not int or rank < 0:
        raise ContractError(f"{label}.selection_rank is invalid")
    if not isinstance(rank_digest, str) or SHA256_RE.fullmatch(rank_digest) is None:
        raise ContractError(f"{label}.selection_rank_sha256 is invalid")
    if stratum not in allowed_strata:
        raise ContractError(f"{label}.stratum is outside frozen order")
    return dict(row)


def _published_native_refs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "published_marker": config["registry"]["published_marker"],
        "registry_manifest": config["registry"]["registry_manifest"],
        "publication_receipt": config["registry"]["publication_receipt"],
        "split_artifacts": config["registry"]["split_artifacts"],
    }


def validate_registry(config: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Open only PUBLISHED.json first, then verify all native payload bytes."""

    registry = config["registry"]
    marker_ref = _ref(registry["published_marker"], "10 PUBLISHED marker")
    marker = read_json(marker_ref.path)
    if not isinstance(marker, Mapping):
        raise ContractError("10 PUBLISHED marker root is malformed")
    if (
        set(marker) != PUBLISHED_MARKER_KEYS
        or marker.get("schema_version") != 1
        or marker.get("status") != "published_complete"
        or marker.get("atomic_last_completion_marker") is not True
        or not isinstance(marker.get("registry_id"), str)
        or not marker.get("registry_id")
    ):
        raise ContractError("10 PUBLISHED marker is not the atomic complete marker")
    published_files = marker.get("published_files")
    if not isinstance(published_files, Mapping) or set(published_files) != PUBLISHED_FILENAMES:
        raise ContractError("10 PUBLISHED marker lacks native file inventory")
    if marker.get("ordered_inventory_sha256") != _publisher_inventory_digest(published_files):
        raise ContractError("10 PUBLISHED ordered inventory digest changed")

    marker_root = Path(marker_ref.path).resolve().parent
    if Path(marker_ref.path).name != "PUBLISHED.json":
        raise ContractError("10 atomic marker filename changed")
    for name in sorted(PUBLISHED_FILENAMES):
        path = marker_root / name
        entry = published_files[name]
        ArtifactRef(str(path), int(entry["nbytes"]), str(entry["sha256"])).verify(
            label=f"10 published inventory {name}"
        )
    for core_name in sorted(CORE_FILENAMES):
        sidecar_name = f"{core_name}.sha256.json"
        sidecar = read_json(marker_root / sidecar_name)
        if sidecar != {
            "path": str((marker_root / core_name).resolve()),
            "nbytes": published_files[core_name]["nbytes"],
            "sha256": published_files[core_name]["sha256"],
        }:
            raise ContractError(f"10 publication sidecar changed: {sidecar_name}")
    if marker.get("receipt") != published_files["receipt.json"]:
        raise ContractError("10 PUBLISHED receipt binding changed")
    if marker.get("receipt_sidecar") != published_files["receipt.json.sha256.json"]:
        raise ContractError("10 PUBLISHED receipt-sidecar binding changed")

    manifest_ref = _ref(registry["registry_manifest"], "10 registry manifest")
    receipt_ref = _ref(registry["publication_receipt"], "10 publication receipt")
    split_refs = {
        key: _ref(registry["split_artifacts"][key], f"10 native split {key}")
        for key in SPLIT_KEYS
    }
    expected_native_names = {
        "registry_manifest": "registry_manifest.json",
        "publication_receipt": "receipt.json",
        **{key: f"{key}.jsonl" for key in SPLIT_KEYS},
    }
    for label, reference in {
        "registry_manifest": manifest_ref,
        "publication_receipt": receipt_ref,
        **split_refs,
    }.items():
        expected_name = expected_native_names[label]
        if Path(reference.path).resolve() != marker_root / expected_name:
            raise ContractError(f"10 native publication path changed: {label}")
        entry = published_files.get(expected_name)
        if entry != {"nbytes": reference.nbytes, "sha256": reference.sha256}:
            raise ContractError(f"10 PUBLISHED marker does not bind exact {label} bytes")

    manifest = read_json(manifest_ref.path)
    if not isinstance(manifest, Mapping) or manifest.get("status") != "prepared_decontaminated_method_dev_only":
        raise ContractError("10 registry manifest status changed")
    receipt = read_json(receipt_ref.path)
    publication = receipt.get("artifact_publication") if isinstance(receipt, Mapping) else None
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("status") != "pass"
        or not isinstance(publication, Mapping)
        or publication.get("completion_marker_required") != "PUBLISHED.json"
        or publication.get("status_pass_receipt_without_completion_marker_is_not_published") is not True
    ):
        raise ContractError("10 publication receipt contract changed")

    result: dict[str, list[dict[str, Any]]] = {}
    all_ids: dict[str, dict[str, set[str]]] = {
        dataset: {partition: set() for partition in PARTITIONS} for dataset in DATASETS
    }
    all_indices: dict[str, dict[str, set[int]]] = {
        dataset: {partition: set() for partition in PARTITIONS} for dataset in DATASETS
    }
    for key in SPLIT_KEYS:
        dataset, partition, expected_count = _expected_split_contract(config, key)
        raw_rows = read_jsonl(split_refs[key].path)
        if len(raw_rows) != expected_count:
            raise ContractError(f"10 native split {key} count {len(raw_rows)} != {expected_count}")
        strata = list(registry["strata_order"][dataset])
        rows = [
            _validate_native_row(
                row,
                label=f"10 native split {key} row {index}",
                dataset=dataset,
                partition=partition,
                allowed_strata=strata,
            )
            for index, row in enumerate(raw_rows)
        ]
        expected_order = sorted(
            rows,
            key=lambda row: (
                strata.index(row["stratum"]),
                row["selection_rank"],
                row["source_index"],
                row["sample_id"],
            ),
        )
        if rows != expected_order:
            raise ContractError(f"10 native split {key} row order changed")
        by_stratum: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            by_stratum[row["stratum"]].append(row["selection_rank"])
        for stratum, ranks in by_stratum.items():
            if ranks != list(range(len(ranks))):
                raise ContractError(f"10 native split {key}/{stratum} local ranks changed")
        ids = [row["sample_id"] for row in rows]
        indices = [row["source_index"] for row in rows]
        if len(set(ids)) != len(ids) or len(set(indices)) != len(indices):
            raise ContractError(f"10 native split {key} has duplicate IDs/source indices")
        all_ids[dataset][partition].update(ids)
        all_indices[dataset][partition].update(indices)
        result[key] = rows
    for dataset in DATASETS:
        if all_ids[dataset]["screen"] & all_ids[dataset]["confirm"]:
            raise ContractError(f"{dataset} native screen/confirm ID overlap")
        if all_indices[dataset]["screen"] & all_indices[dataset]["confirm"]:
            raise ContractError(f"{dataset} native screen/confirm source-index overlap")
    cross = Counter(
        sample_id
        for dataset in DATASETS
        for partition in PARTITIONS
        for sample_id in all_ids[dataset][partition]
    )
    if any(count != 1 for count in cross.values()):
        raise ContractError("sample ID occurs across native datasets/partitions")
    return result


def validate_complete_g0(config: Mapping[str, Any]) -> dict[str, Any]:
    g0_ref = _ref(config["g0"]["completion_receipt"], "complete G0 v2 receipt")
    receipt = read_json(g0_ref.path)
    if not isinstance(receipt, Mapping):
        raise ContractError("complete G0 v2 receipt root is malformed")
    fixed = {
        "status": "complete",
        "pass": True,
        "closes_complete_g0": True,
        "scope": config["g0"]["required_scope"],
        "locked_outcomes_opened": False,
    }
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise ContractError(f"complete G0 v2 receipt {field} is not {expected!r}")
    expected_binding = _published_native_refs(config)
    if receipt.get("candidate_registry_binding") != expected_binding:
        raise ContractError("complete G0 v2 does not bind exact native 10 publication")
    comparisons = receipt.get("comparisons")
    required = {
        "methoddev_hotpot_vs_mapper_fit",
        "methoddev_musique_vs_mapper_fit",
        "methoddev_hotpot_vs_locked_eval",
        "methoddev_musique_vs_locked_eval",
    }
    if not isinstance(comparisons, Mapping) or set(comparisons) != required:
        raise ContractError("complete G0 v2 comparison direction set changed")
    for direction in required:
        entry = comparisons[direction]
        if not isinstance(entry, Mapping) or dict(entry) != {"flags": 0}:
            raise ContractError(
                f"complete G0 v2 direction {direction} must be exact flags:int0"
            )
    return dict(receipt)


def validate_data_gates(config: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    registry = validate_registry(config)
    g0 = validate_complete_g0(config)
    return registry, g0


__all__ = ["NATIVE_ROW_KEYS", "validate_complete_g0", "validate_data_gates", "validate_registry"]
