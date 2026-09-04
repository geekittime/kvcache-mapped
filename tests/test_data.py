from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.data import (  # noqa: E402
    ArtifactVerificationError,
    SourceRecord,
    document_identity,
    iter_source_records,
    prepare_calibration_dataset,
    sha256_file,
    stable_selection_hash,
    stable_window_offset,
    token_ids_sha256,
    verify_calibration_output,
)


class IntegerTokenizer:
    """Minimal tokenizer whose documents are whitespace-separated integers."""

    name_or_path = "integer-tokenizer-test-double"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        self.calls.append(dict(kwargs))
        return {"input_ids": [int(piece) for piece in text.split()]}


def make_records(count: int, token_count: int = 12) -> list[dict[str, str]]:
    return [
        {
            "id": f"doc-{index:03d}",
            "url": f"https://example.invalid/{index}",
            "text": " ".join(
                str(index * 1000 + token_index) for token_index in range(token_count)
            ),
        }
        for index in range(count)
    ]


def prepare_small(
    records: object,
    tokenizer: IntegerTokenizer,
    output: Path,
    *,
    checkpoint_every: int = 2,
) -> dict[str, object]:
    return prepare_calibration_dataset(
        records,  # type: ignore[arg-type]
        tokenizer,
        output,
        train_count=3,
        heldout_count=2,
        sequence_length=4,
        stride=2,
        seed="unit-test-seed",
        text_field="text",
        identity_fields=("id", "url"),
        source_provenance={"type": "unit-test", "version": 1},
        tokenizer_provenance={"aggregate_sha256": "fake-tokenizer-v1"},
        local_conventions=("unit-test convention",),
        checkpoint_every=checkpoint_every,
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_bottom_k_is_order_independent_and_windows_are_within_documents(
    tmp_path: Path,
) -> None:
    records = make_records(13)
    tokenizer_a = IntegerTokenizer()
    tokenizer_b = IntegerTokenizer()
    manifest_a = prepare_small(records, tokenizer_a, tmp_path / "forward")
    manifest_b = prepare_small(reversed(records), tokenizer_b, tmp_path / "reverse")

    train_a = np.load(tmp_path / "forward/train_tokens.npy")
    heldout_a = np.load(tmp_path / "forward/heldout_tokens.npy")
    train_b = np.load(tmp_path / "reverse/train_tokens.npy")
    heldout_b = np.load(tmp_path / "reverse/heldout_tokens.npy")
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(heldout_a, heldout_b)
    assert train_a.shape == (3, 4)
    assert heldout_a.shape == (2, 4)
    assert manifest_a["dataset_fingerprint"] == manifest_b["dataset_fingerprint"]

    expected = []
    for record in records:
        uid, text_hash, _ = document_identity(record, record["text"], ("id", "url"))
        expected.append((stable_selection_hash("unit-test-seed", uid), uid, text_hash, record))
    expected.sort()
    expected = expected[:5]

    rows = read_jsonl(tmp_path / "forward/samples.jsonl")
    assert [row["record_uid"] for row in rows] == [item[1] for item in expected]
    for row, (_, uid, text_hash, record) in zip(rows, expected, strict=True):
        document_tokens = [int(piece) for piece in record["text"].split()]
        offset = stable_window_offset("unit-test-seed", uid, len(document_tokens), 4)
        window = document_tokens[offset : offset + 4]
        assert row["raw_text_sha256"] == text_hash
        assert row["token_offset"] == offset
        assert row["token_ids_sha256"] == token_ids_sha256(window)
        # Every token comes from one contiguous region of exactly one document.
        actual = train_a[row["split_index"]] if row["split"] == "train" else heldout_a[row["split_index"]]
        np.testing.assert_array_equal(actual, window)

    assert tokenizer_a.calls
    assert all(call["add_special_tokens"] is False for call in tokenizer_a.calls)
    assert all(call["truncation"] is False for call in tokenizer_a.calls)


def test_bottom_k_prunes_tokenization_without_changing_exact_result(
    tmp_path: Path,
) -> None:
    """Hash pruning is an exact optimization, not a different sampler."""

    records = make_records(200)
    tokenizer = IntegerTokenizer()
    manifest = prepare_small(records, tokenizer, tmp_path / "optimized")

    expected = []
    for record in records:
        uid, _, _ = document_identity(record, record["text"], ("id", "url"))
        expected.append((stable_selection_hash("unit-test-seed", uid), uid))
    expected.sort()
    selected_uids = [row["record_uid"] for row in read_jsonl(tmp_path / "optimized/samples.jsonl")]
    assert selected_uids == [uid for _, uid in expected[:5]]
    # The first five records fill the reservoir; after that only progressively
    # better hashes can matter, so far fewer than all 200 documents need the
    # tokenizer.  This is what makes a 10BT shard scan practical.
    assert len(tokenizer.calls) < len(records) // 2
    assert manifest["stats"]["selection_pruned_without_tokenization"] > 0
    assert manifest["stats"]["tokenized_for_eligibility"] == len(tokenizer.calls)


def test_resume_from_checkpoint_then_completed_output_is_reused(tmp_path: Path) -> None:
    records = make_records(11)

    def interrupted():
        for index, record in enumerate(records):
            if index == 6:
                raise RuntimeError("simulated source interruption")
            yield SourceRecord(record, "memory://fixture", index)

    output = tmp_path / "resumed"
    with pytest.raises(RuntimeError, match="simulated"):
        prepare_small(interrupted(), IntegerTokenizer(), output, checkpoint_every=2)
    assert not output.exists()
    checkpoint = tmp_path / ".resumed.inprogress/checkpoint.json"
    assert checkpoint.is_file()
    assert json.loads(checkpoint.read_text())["records_seen"] == 6

    resumed = prepare_small(
        (
            SourceRecord(record, "memory://fixture", index)
            for index, record in enumerate(records)
        ),
        IntegerTokenizer(),
        output,
        checkpoint_every=2,
    )
    assert output.is_dir()
    assert not (tmp_path / ".resumed.inprogress").exists()
    assert resumed["stats"]["records_seen"] == len(records)

    fresh = prepare_small(records, IntegerTokenizer(), tmp_path / "fresh")
    assert resumed["dataset_fingerprint"] == fresh["dataset_fingerprint"]
    np.testing.assert_array_equal(
        np.load(output / "train_tokens.npy"),
        np.load(tmp_path / "fresh/train_tokens.npy"),
    )

    class MustNotIterate:
        def __iter__(self):
            raise AssertionError("completed-output resume opened the source")

    reused = prepare_small(MustNotIterate(), IntegerTokenizer(), output)
    assert reused["dataset_fingerprint"] == resumed["dataset_fingerprint"]


def test_jsonl_and_parquet_streaming_sources(tmp_path: Path) -> None:
    records = make_records(4)
    jsonl = tmp_path / "records.jsonl"
    jsonl.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    json_source = {
        "type": "jsonl",
        "paths": [str(jsonl)],
        "expected_sha256": {str(jsonl): sha256_file(jsonl)},
    }
    loaded_json = list(iter_source_records(json_source))
    assert [record.data["id"] for record in loaded_json] == [
        record["id"] for record in records
    ]
    assert loaded_json[0].locator == str(jsonl.resolve())

    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = tmp_path / "records.parquet"
    pq.write_table(pa.Table.from_pylist(records), parquet, row_group_size=2)
    loaded_parquet = list(
        iter_source_records(
            {"type": "parquet", "paths": [str(parquet)], "batch_size": 1}
        )
    )
    assert [record.data["id"] for record in loaded_parquet] == [
        record["id"] for record in records
    ]
    assert [record.record_index for record in loaded_parquet] == list(range(4))


def test_formal_hf_local_mirror_keeps_hf_provenance_and_checks_hash(
    tmp_path: Path,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = make_records(3)
    mirror = tmp_path / "fixed-shard.parquet"
    pq.write_table(pa.Table.from_pylist(records), mirror)
    remote_path = "sample/10BT/000_00000.parquet"
    source = {
        "type": "hf_streaming",
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "config_name": "sample-10BT",
        "split": "train",
        "revision": "a" * 40,
        "data_files": [remote_path],
        "local_mirror_path": str(mirror),
        "expected_lfs_sha256": {remote_path: sha256_file(mirror)},
        "expected_sizes": {remote_path: mirror.stat().st_size},
        "verify_remote_files_metadata": False,
    }
    loaded = list(iter_source_records(source, base_dir=tmp_path))
    assert [record.data["id"] for record in loaded] == [
        record["id"] for record in records
    ]
    assert loaded[0].locator.startswith(
        "hf://datasets/HuggingFaceFW/fineweb-edu@" + "a" * 40
    )
    assert "verified-local-mirror=" in loaded[0].locator

    source["expected_lfs_sha256"] = {remote_path: "0" * 64}
    with pytest.raises(ArtifactVerificationError, match="local mirror SHA256"):
        list(iter_source_records(source, base_dir=tmp_path))


def test_manifest_detects_tampered_token_artifact(tmp_path: Path) -> None:
    output = tmp_path / "dataset"
    prepare_small(make_records(8), IntegerTokenizer(), output)
    with (output / "train_tokens.npy").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ArtifactVerificationError, match="SHA256 mismatch"):
        verify_calibration_output(output)


def test_special_tokens_cannot_be_enabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="add_special_tokens=False"):
        prepare_calibration_dataset(
            make_records(5),
            IntegerTokenizer(),
            tmp_path / "invalid",
            train_count=3,
            heldout_count=2,
            sequence_length=4,
            stride=2,
            add_special_tokens=True,
        )


def test_formal_yaml_marks_unpublished_choices_as_local_conventions() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/calibration_qwen3.yaml").read_text(encoding="utf-8")
    )
    assert config["paper_protocol"] == {
        "dataset": "FineWeb-Edu",
        "train_sequences": 500,
        "sequence_length": 1024,
        "activation_token_stride": 4,
    }
    assert config["source"]["config_name"] == "sample-10BT"
    assert len(config["source"]["revision"]) == 40
    assert config["source"]["data_files"] == [
        "sample/10BT/000_00000.parquet"
    ]
    assert config["source"]["local_mirror_path"].endswith("000_00000.parquet")
    assert len(next(iter(config["source"]["expected_lfs_sha256"].values()))) == 64
    assert config["selection"]["algorithm"] == "sha256-bottom-k-v1"
    assert config["selection"]["add_special_tokens"] is False
    conventions = " ".join(config["local_conventions"]).lower()
    for unpublished_choice in (
        "sample-10bt",
        "revision",
        "shard",
        "bottom-k",
        "add_special_tokens=false",
    ):
        assert unpublished_choice in conventions
