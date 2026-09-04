from __future__ import annotations

from pathlib import Path
import importlib.util
import json

import pytest

from src.method_dev import (
    CandidateSample,
    MethodDevValidationError,
    audit_cross_dataset_overlap,
    ensure_output_within,
    phase_counts,
    select_preregistered_samples,
    _format_musique_context,
)
from src.qa_metrics import extract_answer_candidate, qa_metrics


def _candidate(dataset: str, sample_id: str, stratum: str, question: str) -> CandidateSample:
    return CandidateSample(
        dataset_id=dataset,
        sample_id=sample_id,
        stratum=stratum,
        question=question,
        context=f"context for {dataset} {sample_id}",
        references=("gold",),
        source_record_sha256=(sample_id * 64)[:64],
        metadata={},
    )


def test_selection_is_nested_and_deterministic_without_answers() -> None:
    candidates = [
        _candidate("hotpotqa_train", f"id-{index}", "bridge/hard", f"q {index}")
        for index in range(12)
    ]
    quotas = {"smoke": 1, "pilot": 3, "expansion": 5}
    first = select_preregistered_samples(candidates, seed="fixed", per_phase=quotas)
    second = select_preregistered_samples(list(reversed(candidates)), seed="fixed", per_phase=quotas)
    assert first == second
    assert [row["phase_membership"] for row in first] == [
        ["smoke", "pilot", "expansion"],
        ["pilot", "expansion"],
        ["pilot", "expansion"],
        ["expansion"],
        ["expansion"],
    ]
    modified_answers = [
        CandidateSample(**{**candidate.__dict__, "references": ("different",)})
        for candidate in candidates
    ]
    changed = select_preregistered_samples(modified_answers, seed="fixed", per_phase=quotas)
    assert [row["sample_id"] for row in changed] == [row["sample_id"] for row in first]


def test_overlap_audit_fails_on_cross_dataset_near_duplicate_question() -> None:
    rows = [
        {
            "dataset_id": "hotpotqa_train",
            "sample_id": "h1",
            "question": "Which city was the famous scientist born in?",
            "context": "one context",
        },
        {
            "dataset_id": "musique_train",
            "sample_id": "m1",
            "question": "Which city was the famous scientist born in?",
            "context": "another context",
        },
    ]
    audit = audit_cross_dataset_overlap(rows)
    assert audit["status"] == "failed"
    assert audit["near_question_collisions"][0]["word_trigram_jaccard"] == 1.0


def test_output_containment_rejects_escape_and_root(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    assert ensure_output_within(root, root / "pilot") == (root / "pilot").resolve()
    with pytest.raises(MethodDevValidationError):
        ensure_output_within(root, tmp_path / "elsewhere")
    with pytest.raises(MethodDevValidationError):
        ensure_output_within(root, root)


def test_gold_independent_extraction_and_relaxed_metric() -> None:
    prediction = "Some extra text\nFinal answer: 1960"
    assert extract_answer_candidate(prediction) == "1960"
    metrics = qa_metrics(prediction, ["1960"])
    assert metrics["qa_exact_match"] == 0.0
    assert metrics["relaxed_qa_exact_match"] == 1.0
    assert metrics["normalized_gold_containment"] == 1.0


def test_musique_prompt_is_byte_exact_with_bound_06_builder() -> None:
    candidate_root = Path(__file__).resolve().parents[1]
    musique_root = candidate_root.parent / "06_musique_benchmark"
    module_path = musique_root / "src/musique_benchmark.py"
    specification = importlib.util.spec_from_file_location(
        "bound_musique_benchmark_for_test", module_path
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    train_path = musique_root / "upstream_raw/data/musique_ans_v1.0_train.jsonl"
    with train_path.open(encoding="utf-8") as handle:
        record = json.loads(next(handle))
    config = json.loads((candidate_root / "configs/method_dev.json").read_text())
    context = _format_musique_context(record["paragraphs"], "test")
    local = config["prompt"]["completion_template"].format(
        context=context, question=record["question"]
    )
    assert local.encode("utf-8") == module.build_prompt(record).encode("utf-8")
