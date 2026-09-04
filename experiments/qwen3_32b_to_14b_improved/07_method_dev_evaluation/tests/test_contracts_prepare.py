from __future__ import annotations

import copy
from pathlib import Path

import pytest

from src.contracts import (
    ArtifactRef,
    ContractError,
    canonical_sha256,
    load_config,
    token_ids_sha256,
)
from src.prepare_data import (
    head_tail_token_ids,
    normalized_question_group,
    select_hotpot_records,
    validate_prepared_row,
)


def _hotpot(index: int, *, question: str | None = None) -> dict:
    return {
        "_id": f"id-{index:04d}",
        "question": question or f"Unique question number {index}?",
        "answer": f"answer {index}",
        "type": "bridge" if index % 2 else "comparison",
        "level": "hard",
        "context": [["Title", ["A sentence."]]],
    }


def test_artifact_ref_requires_path_size_and_sha(tmp_path: Path) -> None:
    path = tmp_path / "input.txt"
    path.write_bytes(b"exact bytes")
    reference = ArtifactRef(
        path=str(path.resolve()), nbytes=11, sha256="e38e581aade78b64cc86f7ac9f3555ca78c2dcca747942a7f1d9b3275a834f75"
    )
    assert reference.verify() == path.resolve()
    with pytest.raises(ContractError, match="size drift"):
        ArtifactRef(path=str(path.resolve()), nbytes=10, sha256=reference.sha256).verify()


def test_hotpot_selection_is_deterministic_disjoint_and_answer_independent() -> None:
    records = [_hotpot(index) for index in range(410)]
    # Both rows in a repeated normalized-question group must be quarantined.
    records[400]["question"] = " Repeated   QUESTION "
    records[401]["question"] = "repeated question"
    first, audit = select_hotpot_records(
        records,
        screen_namespace="screen-v1",
        confirm_namespace="confirm-v1",
        screen_count=128,
        confirm_count=256,
    )
    reversed_records = list(reversed(records))
    # source_index is the official file index and therefore intentionally part
    # of the tie-break only; restore original sequence for answer independence.
    changed_answers = copy.deepcopy(records)
    for record in changed_answers:
        record["answer"] = "never read by rank"
    second, _ = select_hotpot_records(
        changed_answers,
        screen_namespace="screen-v1",
        confirm_namespace="confirm-v1",
        screen_count=128,
        confirm_count=256,
    )
    assert [row["sample_id"] for row in first["screen"]] == [
        row["sample_id"] for row in second["screen"]
    ]
    assert [row["sample_id"] for row in first["confirm"]] == [
        row["sample_id"] for row in second["confirm"]
    ]
    assert not (
        {row["sample_id"] for row in first["screen"]}
        & {row["sample_id"] for row in first["confirm"]}
    )
    selected = {row["sample_id"] for rows in first.values() for row in rows}
    assert "id-0400" not in selected and "id-0401" not in selected
    assert audit["duplicate_normalized_question_rows_quarantined"] == 2
    assert normalized_question_group(" Repeated   QUESTION ") == "repeated question"
    assert reversed_records[0]["_id"] == "id-0409"


def test_head_tail_uses_exact_ids_and_assigns_odd_token_to_head() -> None:
    values, audit = head_tail_token_ids(list(range(12)), 7)
    assert values == [0, 1, 2, 3, 9, 10, 11]
    assert audit == {
        "policy": "head_tail",
        "truncated": True,
        "original_tokens": 12,
        "used_tokens": 7,
        "removed_tokens": 5,
        "head_tokens": 4,
        "tail_tokens": 3,
    }


def test_prepared_row_binding_and_exact_token_ids_fail_closed() -> None:
    row = {
        "schema_version": 1,
        "dataset_id": "hotpotqa_train",
        "partition": "screen",
        "references": ["gold"],
        "references_sha256": canonical_sha256(["gold"]),
        "prompt_token_ids": list(range(40)),
        "prompt_token_count": 40,
        "prompt_token_ids_sha256": token_ids_sha256(list(range(40))),
        "generation_cap": 256,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
    }
    row["sample_binding_sha256"] = canonical_sha256(row)
    validate_prepared_row(row, max_length=16384, cap=256)
    broken = copy.deepcopy(row)
    broken["prompt_token_ids"][0] = 99
    with pytest.raises(ContractError, match="binding"):
        validate_prepared_row(broken, max_length=16384, cap=256)


def test_real_config_schema_loads_without_opening_inputs() -> None:
    config = load_config(verify_inputs=False)
    assert config["profiles"]["screen"] == {
        "hotpotqa_train": 128,
        "musique_train": 256,
    }
    assert config["generation"]["max_new_tokens"] == 256
    assert config["methods"]["shrinkage"]["state"] == "ready"
    assert config["methods"]["tail_refresh"]["state"] == "ready"
    assert config["confirmation_gate"]["outcome_threshold"] is None
    assert config["provenance"]["cross_tier_audit_receipt"]["sha256"] == (
        "37f69adcafa583d0898579f30701d09e569551895b1339b693d18ada420da84d"
    )
