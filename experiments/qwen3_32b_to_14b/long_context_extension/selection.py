"""Frozen dataset selection for the Qwen3 32B -> 14B long-context extension.

The forward experiment already established the dataset parsing rules.  This
module deliberately delegates parsing to that frozen runner and owns only the
reverse experiment's exact sample-selection contract and fail-closed identity
checks.  Consequently a direction comparison uses identical examples without
copying a second, subtly different dataset parser.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


MULTINEWS_SOURCE_ROWS = 5_622
MULTINEWS_SAMPLE_COUNT = 500
MULTINEWS_SELECTION_SEED = (
    "qwen3-14b-to-32b-multinews-official-test-bottom500-v1"
)
MULTINEWS_SOURCE_SHA256 = (
    "138d3ac2dc899cbcd2e3745aaa94d1c1db55fb7058d9df4ba3ef2dac05a3a186"
)
MULTINEWS_TARGET_SHA256 = (
    "fa97cf91a62ae82a0af6da88f2ddf8e06eb4e3b90f7971d8e0c516436518fae3"
)
MULTINEWS_FULL_IDENTITY_SHA256 = (
    "65a7d20c35aab4051b520a9cf6faeb08c2c7e193edf2a53bfc7965746239fa27"
)
MULTINEWS_SELECTED_SHA256 = (
    "47f460b9bb3c69a60cea1b1d44ea8a9a64fd33623d107f4129d48ca6c5e76076"
)
MULTINEWS_SELECTED_INDICES_SHA256 = (
    "d39b44894d60ad2fbbb3aaad96b619fcbfde833a982f60644305f109821f8510"
)

LMSYS_SAMPLE_COUNT = 500
LMSYS_SELECTION_SEED = "qwen3-14b-to-32b-lmsys-heldout-v1"
LMSYS_ROWS = 273_402
LMSYS_FILES_SHA256 = (
    "5a82b44c3c574bba4e0f62fd937c1e4ed82e1ad93959ae16e5e5b78257d8eefb"
)
LMSYS_SELECTED_SHA256 = (
    "78a996e129f778ac9cd86a5ac9a2391c556d080d48970bd89650e6a0be0cb946"
)

SHAREGPT4_SAMPLE_COUNT = 500
SHAREGPT4_SELECTION_SEED = "qwen3-14b-to-32b-sharegpt4-heldout-v1"
SHAREGPT4_ROWS = 6_206
SHAREGPT4_FILE_SHA256 = (
    "b59e417da0aa7b5f40dfbfc2af77b3d7fd250356e7cd1b0ff002ad0ffd048f43"
)
SHAREGPT4_SELECTED_SHA256 = (
    "1c3d3e541a5c9c960f3371562dfe3c5e4a75e8a1f8d2278472c5174d4c7404ea"
)


def canonical_digest(value: Any) -> str:
    """Return the repository's architecture-independent canonical JSON hash."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _selection_rank(sample: Any) -> str:
    identity = (
        f"{MULTINEWS_SELECTION_SEED}\0{int(sample.index)}\0{sample.sample_id}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _select_multinews_bottom500(
    samples: Sequence[Any], manifest: Mapping[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Select the exact gold-independent forward MultiNews bottom-500."""

    if len(samples) != MULTINEWS_SOURCE_ROWS or manifest.get("rows") != MULTINEWS_SOURCE_ROWS:
        raise ValueError(
            "MultiNews source must contain exactly "
            f"{MULTINEWS_SOURCE_ROWS} rows, got {len(samples)}/{manifest.get('rows')}"
        )
    observed_indices = [int(sample.index) for sample in samples]
    if observed_indices != list(range(MULTINEWS_SOURCE_ROWS)):
        raise ValueError("MultiNews rows are not in canonical source-index order")
    if len({str(sample.sample_id) for sample in samples}) != MULTINEWS_SOURCE_ROWS:
        raise ValueError("MultiNews sample IDs are not unique")

    source = _mapping(manifest.get("source"), "MultiNews source identity")
    target = _mapping(manifest.get("target"), "MultiNews target identity")
    if manifest.get("split") != "official_test":
        raise ValueError("MultiNews selection requires the official test split")
    if source.get("sha256") != MULTINEWS_SOURCE_SHA256:
        raise ValueError("MultiNews source file does not match the frozen SHA256")
    if target.get("sha256") != MULTINEWS_TARGET_SHA256:
        raise ValueError("MultiNews gold-summary file does not match the frozen SHA256")
    if manifest.get("sample_identity_sha256") != MULTINEWS_FULL_IDENTITY_SHA256:
        raise ValueError("MultiNews full sample identity does not match the frozen SHA256")

    ranked = [
        (_selection_rank(sample), int(sample.index), sample) for sample in samples
    ]
    chosen_by_rank = sorted(ranked, key=lambda row: (row[0], row[1]))[
        :MULTINEWS_SAMPLE_COUNT
    ]
    chosen = sorted(chosen_by_rank, key=lambda row: row[1])
    selected_records = [
        {
            "execution_position": execution_position,
            "source_index": source_index,
            "sample_id": str(sample.sample_id),
            "record_sha256": str(sample.record_sha256),
            "selection_rank_sha256": rank,
        }
        for execution_position, (rank, source_index, sample) in enumerate(chosen)
    ]
    selected_sha256 = canonical_digest(selected_records)
    selected_indices_sha256 = canonical_digest(
        [record["source_index"] for record in selected_records]
    )
    if selected_sha256 != MULTINEWS_SELECTED_SHA256:
        raise ValueError("MultiNews bottom-500 records do not match the frozen SHA256")
    if selected_indices_sha256 != MULTINEWS_SELECTED_INDICES_SHA256:
        raise ValueError("MultiNews bottom-500 indices do not match the frozen SHA256")

    updated_manifest = dict(manifest)
    updated_manifest["formal_selection"] = {
        "name": "sha256_bottom_k",
        "seed": MULTINEWS_SELECTION_SEED,
        "source_split": "official_test",
        "source_rows": MULTINEWS_SOURCE_ROWS,
        "selected_rows": MULTINEWS_SAMPLE_COUNT,
        "rank_identity": "seed\\0source_index\\0sample_id (gold-independent)",
        "execution_order": "ascending_source_index",
        "selected": selected_records,
        "selected_sha256": selected_sha256,
        "selected_indices_sha256": selected_indices_sha256,
    }
    return [sample for _, _, sample in chosen], updated_manifest


def load_multi_news_bottom500(
    base: Any, source_path: str, target_path: str
) -> tuple[list[Any], dict[str, Any]]:
    """Load and select the exact forward-compatible MultiNews test rows."""

    samples, manifest = base.load_multi_news_samples(source_path, target_path)
    return _select_multinews_bottom500(samples, manifest)


def _validate_dialogue_request(
    *,
    task: str,
    sample_count: int,
    selection_seed: str,
    expected_count: int,
    expected_seed: str,
) -> None:
    if int(sample_count) != expected_count:
        raise ValueError(
            f"{task} formal selection requires exactly {expected_count} samples"
        )
    if str(selection_seed) != expected_seed:
        raise ValueError(f"{task} selection seed does not match the frozen protocol")


def _validate_dialogue_manifest(
    samples: Sequence[Any],
    manifest: Mapping[str, Any],
    *,
    task: str,
    expected_rows: int,
    expected_count: int,
    expected_seed: str,
    expected_selected_sha256: str,
) -> None:
    if manifest.get("split") != "train":
        raise ValueError(f"{task} must disclose its train-only source split")
    if manifest.get("rows") != expected_rows:
        raise ValueError(f"{task} source row count does not match the frozen dataset")
    if len(samples) != expected_count or manifest.get("selected_rows") != expected_count:
        raise ValueError(f"{task} did not produce exactly {expected_count} samples")
    selected = manifest.get("selected")
    if not isinstance(selected, list) or len(selected) != expected_count:
        raise ValueError(f"{task} selected-record manifest is incomplete")
    if [int(sample.index) for sample in samples] != list(range(expected_count)):
        raise ValueError(f"{task} selected samples are not canonically reindexed")
    selection = _mapping(manifest.get("selection"), f"{task} selection protocol")
    if selection.get("seed") != expected_seed:
        raise ValueError(f"{task} manifest contains the wrong selection seed")
    if selection.get("official_test_split") is not False:
        raise ValueError(f"{task} must not be described as an official test split")
    selected_sha256 = canonical_digest(selected)
    if manifest.get("selected_sha256") != selected_sha256:
        raise ValueError(f"{task} manifest selected SHA256 is internally inconsistent")
    if selected_sha256 != expected_selected_sha256:
        raise ValueError(f"{task} bottom-500 does not match the frozen SHA256")
    sample_identity = [
        (str(sample.sample_id), str(sample.record_sha256)) for sample in samples
    ]
    manifest_identity = [
        (str(record.get("sample_id")), str(record.get("record_sha256")))
        for record in selected
    ]
    if sample_identity != manifest_identity:
        raise ValueError(f"{task} samples do not match their selected-record manifest")


def load_lmsys_bottom500(
    base: Any,
    parquet_dir: str,
    *,
    sample_count: int,
    selection_seed: str,
) -> tuple[list[Any], dict[str, Any]]:
    """Load the frozen train-derived LMSYS held-out bottom-500."""

    _validate_dialogue_request(
        task="lmsys",
        sample_count=sample_count,
        selection_seed=selection_seed,
        expected_count=LMSYS_SAMPLE_COUNT,
        expected_seed=LMSYS_SELECTION_SEED,
    )
    samples, manifest = base.load_lmsys_samples(
        parquet_dir,
        sample_count=LMSYS_SAMPLE_COUNT,
        selection_seed=LMSYS_SELECTION_SEED,
    )
    if manifest.get("files_sha256") != LMSYS_FILES_SHA256:
        raise ValueError("LMSYS source shards do not match the frozen SHA256")
    _validate_dialogue_manifest(
        samples,
        manifest,
        task="lmsys",
        expected_rows=LMSYS_ROWS,
        expected_count=LMSYS_SAMPLE_COUNT,
        expected_seed=LMSYS_SELECTION_SEED,
        expected_selected_sha256=LMSYS_SELECTED_SHA256,
    )
    return samples, dict(manifest)


def load_sharegpt_bottom500(
    base: Any,
    jsonl_path: str,
    *,
    sample_count: int,
    selection_seed: str,
) -> tuple[list[Any], dict[str, Any]]:
    """Load the frozen train-derived ShareGPT4 held-out bottom-500."""

    _validate_dialogue_request(
        task="sharegpt4",
        sample_count=sample_count,
        selection_seed=selection_seed,
        expected_count=SHAREGPT4_SAMPLE_COUNT,
        expected_seed=SHAREGPT4_SELECTION_SEED,
    )
    samples, manifest = base.load_sharegpt_samples(
        jsonl_path,
        sample_count=SHAREGPT4_SAMPLE_COUNT,
        selection_seed=SHAREGPT4_SELECTION_SEED,
    )
    if manifest.get("sha256") != SHAREGPT4_FILE_SHA256:
        raise ValueError("ShareGPT4 source file does not match the frozen SHA256")
    _validate_dialogue_manifest(
        samples,
        manifest,
        task="sharegpt4",
        expected_rows=SHAREGPT4_ROWS,
        expected_count=SHAREGPT4_SAMPLE_COUNT,
        expected_seed=SHAREGPT4_SELECTION_SEED,
        expected_selected_sha256=SHAREGPT4_SELECTED_SHA256,
    )
    return samples, dict(manifest)


def nested_bottom_k_records(
    selected_records: Sequence[Mapping[str, Any]], *, sample_count: int = 500
) -> list[dict[str, Any]]:
    """Reconstruct a nested bottom-k from a larger frozen bottom-k manifest.

    This audit helper makes the bottom-500 golden test inexpensive: the
    completed forward bottom-1,000 manifest is already content-bound to the
    real LMSYS/ShareGPT4 files.  It is not used by formal inference.
    """

    if sample_count <= 0 or sample_count > len(selected_records):
        raise ValueError("invalid nested bottom-k sample count")
    required = {
        "evaluation_index",
        "source_order",
        "sample_id",
        "record_sha256",
        "conversation_sha256",
        "first_prompt_sha256",
        "selection_rank_sha256",
    }
    rows: list[dict[str, Any]] = []
    for record in selected_records:
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("nested bottom-k record has an unexpected schema")
        rows.append(dict(record))
    chosen = sorted(
        rows,
        key=lambda record: (
            str(record["selection_rank_sha256"]),
            int(record["source_order"]),
        ),
    )[:sample_count]
    chosen.sort(key=lambda record: int(record["source_order"]))
    for evaluation_index, record in enumerate(chosen):
        record["evaluation_index"] = evaluation_index
    return chosen

