#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import canonical_bytes, sha256_file


class ValidationError(RuntimeError):
    pass


CORE_COUNTS = {
    "hotpotqa_screen.jsonl": 128,
    "hotpotqa_confirm.jsonl": 256,
    "musique_screen.jsonl": 256,
    "musique_confirm.jsonl": 512,
}
CORE_PAYLOADS = sorted([*CORE_COUNTS, "quarantine_aggregate.json", "registry_manifest.json", "receipt.json"])
ALLOWED_FIELDS = {
    "dataset_id", "partition", "sample_id", "source_index",
    "source_projection_sha256", "selection_rank",
    "selection_rank_sha256", "stratum",
}


def observed(path: Path) -> dict[str, Any]:
    return {"nbytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as error:
        raise ValidationError(f"invalid JSON: {path}") from error


def ordered_registry_sha256(artifact: Path) -> str:
    hasher = hashlib.sha256()
    for name in CORE_COUNTS:
        payload = (artifact / name).read_bytes()
        hasher.update(name.encode())
        hasher.update(b"\0")
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return hasher.hexdigest()


def validate(artifact: Path) -> dict[str, Any]:
    if not artifact.is_dir():
        raise ValidationError(f"missing artifact directory: {artifact}")
    if (artifact / "IN_PROGRESS.json").exists():
        raise ValidationError("in-progress marker remains")
    publication_path = artifact / "PUBLISHED.json"
    if not publication_path.is_file():
        raise ValidationError("missing final atomic PUBLISHED.json marker")
    publication = load_json(publication_path)
    if publication.get("status") != "published_complete" or publication.get("atomic_last_completion_marker") is not True:
        raise ValidationError("publication marker semantics mismatch")
    actual_names = sorted(path.name for path in artifact.iterdir() if path.is_file())
    expected_names = sorted([*CORE_PAYLOADS, *(f"{name}.sha256.json" for name in CORE_PAYLOADS), "PUBLISHED.json"])
    if actual_names != expected_names:
        raise ValidationError(f"artifact missing/extra files: {actual_names}")
    if sorted(publication.get("published_files", {})) != sorted(name for name in actual_names if name != "PUBLISHED.json"):
        raise ValidationError("publication inventory name mismatch")
    for name, expected in publication["published_files"].items():
        if observed(artifact / name) != expected:
            raise ValidationError(f"publication digest mismatch: {name}")
    inventory = [{"name": name, **publication["published_files"][name]} for name in sorted(publication["published_files"])]
    if hashlib.sha256(canonical_bytes(inventory)).hexdigest() != publication.get("ordered_inventory_sha256"):
        raise ValidationError("ordered publication digest mismatch")

    for name in CORE_PAYLOADS:
        expected = load_json(artifact / f"{name}.sha256.json")
        if {key: expected[key] for key in ("nbytes", "sha256")} != observed(artifact / name):
            raise ValidationError(f"payload sidecar mismatch: {name}")

    manifest = load_json(artifact / "registry_manifest.json")
    receipt = load_json(artifact / "receipt.json")
    if manifest.get("status") != "prepared_decontaminated_method_dev_only" or receipt.get("status") != "pass":
        raise ValidationError("manifest/receipt status mismatch")
    for value in (manifest, receipt):
        publication_contract = value.get("artifact_publication", {})
        if publication_contract.get("completion_marker_required") != "PUBLISHED.json":
            raise ValidationError("publication contract missing")
        if publication_contract.get("status_pass_receipt_without_completion_marker_is_not_published") is not True:
            raise ValidationError("fail-closed publication contract mismatch")
    if manifest.get("counts") != CORE_COUNTS or receipt.get("selection_counts") != CORE_COUNTS:
        raise ValidationError("selection count mismatch")
    if any(value for row in receipt["post_selection_four_dataset_pair_flag_counts"].values() for value in row.values()):
        raise ValidationError("locked-dev four-way matrix is nonzero")
    if any(receipt["screen_confirm_sequence_disjoint_four_cell_flag_counts"].values()):
        raise ValidationError("screen-confirm four-way matrix is nonzero")
    if receipt.get("selected_identity_count") != sum(CORE_COUNTS.values()) or receipt.get("selected_identities_globally_unique") is not True:
        raise ValidationError("selected identity uniqueness mismatch")
    if receipt["musique_quota_derivation"] != {
        "clean_pool_frozen_before_sha_ranking": True,
        "capacity_only_no_answer_or_outcome": True,
        "observed_clean_4hop_pool": 119,
        "selected_4hop": 117,
        "reserved_unused_4hop_buffer": 2,
    }:
        raise ValidationError("MuSiQue 4-hop quota/buffer mismatch")

    identities: set[tuple[str, str]] = set()
    partition_ids: dict[str, set[tuple[str, str]]] = {"screen": set(), "confirm": set()}
    for name, expected_count in CORE_COUNTS.items():
        rows = [json.loads(line) for line in (artifact / name).read_text().splitlines()]
        if len(rows) != expected_count:
            raise ValidationError(f"JSONL row count mismatch: {name}")
        expected_dataset = "hotpotqa_train" if name.startswith("hotpotqa_") else "musique_train"
        expected_partition = "screen" if "_screen." in name else "confirm"
        for row in rows:
            if set(row) != ALLOWED_FIELDS:
                raise ValidationError(f"registry allowlist mismatch: {name}")
            if row["dataset_id"] != expected_dataset or row["partition"] != expected_partition:
                raise ValidationError(f"registry identity/partition mismatch: {name}")
            if len(row["source_projection_sha256"]) != 64 or len(row["selection_rank_sha256"]) != 64:
                raise ValidationError(f"registry SHA field mismatch: {name}")
            identity = (row["dataset_id"], row["sample_id"])
            if identity in identities:
                raise ValidationError("duplicate registry identity")
            identities.add(identity)
            partition_ids[expected_partition].add(identity)
    if partition_ids["screen"] & partition_ids["confirm"]:
        raise ValidationError("screen and confirm identities overlap")

    for key, spec in manifest["implementation"].items():
        if key == "tests":
            for test_spec in spec.values():
                if observed(Path(test_spec["path"])) != {field: test_spec[field] for field in ("nbytes", "sha256")}:
                    raise ValidationError("test implementation binding mismatch")
        elif observed(Path(spec["path"])) != {field: spec[field] for field in ("nbytes", "sha256")}:
            raise ValidationError(f"implementation binding mismatch: {key}")
    return {
        "status": "pass",
        "artifact": str(artifact),
        "counts": CORE_COUNTS,
        "selected_identity_count": len(identities),
        "ordered_registry_sha256": ordered_registry_sha256(artifact),
        "ordered_publication_inventory_sha256": publication["ordered_inventory_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir")
    parser.add_argument("--determinism-artifact-dir")
    args = parser.parse_args()
    config = load_json(ROOT / "config.json")
    artifact = ROOT / (args.artifact_dir or config["outputs"]["artifact_dir"])
    result = validate(artifact)
    if args.determinism_artifact_dir:
        mirror = ROOT / args.determinism_artifact_dir
        mirror_result = validate(mirror)
        for name in [*CORE_COUNTS, "quarantine_aggregate.json"]:
            if (artifact / name).read_bytes() != (mirror / name).read_bytes():
                raise ValidationError(f"deterministic mirror differs: {name}")
        result["determinism"] = {
            "status": "pass",
            "mirror_artifact": str(mirror),
            "ordered_registry_sha256": mirror_result["ordered_registry_sha256"],
            "byte_identical_files": [*CORE_COUNTS, "quarantine_aggregate.json"],
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
