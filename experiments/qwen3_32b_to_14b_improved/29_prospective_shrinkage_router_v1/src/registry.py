"""Fresh-registry contract used by the prospective router.

The manifest is an offline artifact and may contain dataset names for balanced
sampling.  ``router_observations`` intentionally projects it to telemetry-only
objects before routing, so those names cannot enter the decision path.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .router import Observation


HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_DATASETS = ("hotpotqa_train", "musique_train")
REQUIRED_SPLITS = ("calibration", "sealed_eval")


class RegistryError(ValueError):
    """Raised when a fresh registry violates the pre-registration contract."""


@dataclass(frozen=True)
class RegistryRecord:
    sample_id: str
    dataset_id: str
    split: str
    source_record_sha256: str


@dataclass(frozen=True)
class ValidatedRegistry:
    registry_id: str
    tokenizer_path: str
    tokenizer_sha256: str
    records: tuple[RegistryRecord, ...]
    manifest_sha256: str

    @property
    def split_records(self) -> dict[str, tuple[RegistryRecord, ...]]:
        return {
            split: tuple(record for record in self.records if record.split == split)
            for split in REQUIRED_SPLITS
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise RegistryError(f"{name} must be a non-empty string")
    return value


def _require_sha(value: object, name: str) -> str:
    text = _require_string(value, name)
    if HEX64.fullmatch(text) is None:
        raise RegistryError(f"{name} must be lowercase 64-hex SHA-256")
    return text


def validate_manifest(
    manifest: Mapping[str, object],
    *,
    expected_tokenizer_sha256: str,
    min_calibration_per_dataset: int = 24,
    min_sealed_per_dataset: int = 48,
) -> ValidatedRegistry:
    """Fail closed on malformed, overlapping, or under-sized registries.

    This validator is intentionally independent of model outputs.  It checks
    only identity, split disjointness, source hashes, and balancing metadata.
    """

    if not isinstance(manifest, Mapping):
        raise RegistryError("manifest must be a mapping")
    if manifest.get("schema_version") != 1:
        raise RegistryError("unsupported schema_version")
    registry_id = _require_string(manifest.get("registry_id"), "registry_id")
    if manifest.get("labels_hidden_from_router") is not True:
        raise RegistryError("labels_hidden_from_router must be true")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise RegistryError("tokenizer block missing")
    tokenizer_path = _require_string(tokenizer.get("path"), "tokenizer.path")
    tokenizer_sha = _require_sha(tokenizer.get("sha256"), "tokenizer.sha256")
    expected = _require_sha(expected_tokenizer_sha256, "expected_tokenizer_sha256")
    if tokenizer_sha != expected:
        raise RegistryError("tokenizer SHA does not match frozen contract")

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(REQUIRED_SPLITS):
        raise RegistryError("splits must contain exactly calibration and sealed_eval")

    records: list[RegistryRecord] = []
    seen: set[str] = set()
    counts: dict[str, Counter[str]] = {split: Counter() for split in REQUIRED_SPLITS}
    for split in REQUIRED_SPLITS:
        entries = splits[split]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise RegistryError(f"{split} must be a sequence")
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise RegistryError(f"{split}[{index}] must be a mapping")
            sample_id = _require_string(entry.get("sample_id"), f"{split}[{index}].sample_id")
            if sample_id in seen:
                raise RegistryError(f"duplicate sample_id across registry: {sample_id}")
            seen.add(sample_id)
            dataset_id = _require_string(entry.get("dataset_id"), f"{split}[{index}].dataset_id")
            if dataset_id not in REQUIRED_DATASETS:
                raise RegistryError(f"unsupported dataset_id: {dataset_id}")
            source_sha = _require_sha(
                entry.get("source_record_sha256"),
                f"{split}[{index}].source_record_sha256",
            )
            records.append(
                RegistryRecord(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    split=split,
                    source_record_sha256=source_sha,
                )
            )
            counts[split][dataset_id] += 1

    for dataset in REQUIRED_DATASETS:
        if counts["calibration"][dataset] < min_calibration_per_dataset:
            raise RegistryError(f"calibration is too small for {dataset}")
        if counts["sealed_eval"][dataset] < min_sealed_per_dataset:
            raise RegistryError(f"sealed_eval is too small for {dataset}")

    declared_counts = manifest.get("dataset_counts")
    if not isinstance(declared_counts, Mapping):
        raise RegistryError("dataset_counts block missing")
    for split in REQUIRED_SPLITS:
        declared = declared_counts.get(split)
        if not isinstance(declared, Mapping):
            raise RegistryError(f"dataset_counts.{split} missing")
        for dataset in REQUIRED_DATASETS:
            value = declared.get(dataset)
            if type(value) is not int or value != counts[split][dataset]:
                raise RegistryError(f"dataset_counts mismatch for {split}/{dataset}")

    # The manifest digest is computed from the parsed mapping exactly as the
    # receipt will record it; callers should keep the source JSON immutable.
    manifest_sha = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    return ValidatedRegistry(
        registry_id=registry_id,
        tokenizer_path=tokenizer_path,
        tokenizer_sha256=tokenizer_sha,
        records=tuple(records),
        manifest_sha256=manifest_sha,
    )


def router_observations(
    telemetry: Sequence[Mapping[str, object]],
) -> tuple[Observation, ...]:
    """Project telemetry rows to router inputs, rejecting forbidden fields.

    Each row must contain exactly the three allowed keys.  This strictness is
    useful at a serving boundary: a caller cannot accidentally pass a dataset
    label or an answer-bearing field alongside otherwise valid telemetry.
    """

    if isinstance(telemetry, (str, bytes)) or not isinstance(telemetry, Sequence):
        raise RegistryError("telemetry must be a sequence")
    allowed = {"prompt_tokens", "r1_generated_tokens", "first_pass_complete"}
    result: list[Observation] = []
    for index, row in enumerate(telemetry):
        if not isinstance(row, Mapping):
            raise RegistryError(f"telemetry[{index}] must be a mapping")
        if set(row) != allowed:
            raise RegistryError(f"telemetry[{index}] has forbidden or missing fields")
        result.append(
            Observation(
                prompt_tokens=row["prompt_tokens"],
                r1_generated_tokens=row["r1_generated_tokens"],
                first_pass_complete=row["first_pass_complete"],
            )
        )
    return tuple(result)

