from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[4]
EXTENSION = ROOT / "experiments/qwen3_32b_to_14b/long_context_extension"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SELECTION = _load("reverse_long_context_selection_tested", EXTENSION / "selection.py")
METRICS = _load("reverse_long_context_metrics_tested", EXTENSION / "metrics.py")
BASE = _load("reverse_long_context_forward_base_tested", ROOT / "scripts/evaluate_long_context.py")


@dataclass(frozen=True)
class _Sample:
    index: int
    sample_id: str
    record_sha256: str


def test_real_multinews_bottom500_matches_every_frozen_identity():
    samples, manifest = SELECTION.load_multi_news_bottom500(
        BASE,
        ROOT / "data/MultiNews/data/test.src.cleaned",
        ROOT / "data/MultiNews/data/test.tgt",
    )
    formal = manifest["formal_selection"]
    assert len(samples) == 500
    assert formal["selected_sha256"] == SELECTION.MULTINEWS_SELECTED_SHA256
    assert (
        formal["selected_indices_sha256"]
        == SELECTION.MULTINEWS_SELECTED_INDICES_SHA256
    )
    assert [sample.index for sample in samples] == sorted(
        sample.index for sample in samples
    )
    # Selection ranks do not read the gold-bearing record hash.
    assert formal["rank_identity"].endswith("(gold-independent)")


def test_real_dialogue_bottom500_golden_is_nested_in_completed_bottom1000():
    run_definition = json.loads(
        (
            ROOT
            / "runs/20260820_qwen3_14b_to_32b_dialogue_bottom1000_full_v1"
            / "run_definition.json"
        ).read_text(encoding="utf-8")
    )
    expected = {
        "lmsys": SELECTION.LMSYS_SELECTED_SHA256,
        "sharegpt4": SELECTION.SHAREGPT4_SELECTED_SHA256,
    }
    for task, expected_sha256 in expected.items():
        dataset = run_definition["dataset_manifest"][task]
        records = SELECTION.nested_bottom_k_records(dataset["selected"])
        assert SELECTION.canonical_digest(records) == expected_sha256
        assert dataset["selection"]["official_test_split"] is False
        assert dataset["split"] == "train"


def _dialogue_fixture(task: str, selected_sha256: str, seed: str, rows: int):
    selected = [
        {
            "evaluation_index": index,
            "source_order": index,
            "sample_id": f"{task}-{index}",
            "record_sha256": f"{index:064x}",
            "conversation_sha256": f"{index + 1:064x}",
            "first_prompt_sha256": f"{index + 2:064x}",
            "selection_rank_sha256": f"{index + 3:064x}",
        }
        for index in range(500)
    ]
    # The wrapper must reject this synthetic identity rather than accepting a
    # caller-provided selected_sha256 string at face value.
    manifest = {
        "split": "train",
        "rows": rows,
        "selected_rows": 500,
        "selected": selected,
        "selected_sha256": selected_sha256,
        "selection": {"seed": seed, "official_test_split": False},
    }
    samples = [
        _Sample(index, record["sample_id"], record["record_sha256"])
        for index, record in enumerate(selected)
    ]
    return samples, manifest


def test_dialogue_wrappers_fail_closed_on_wrong_request_and_spoofed_manifest():
    class FakeBase:
        @staticmethod
        def load_lmsys_samples(*_args, **_kwargs):
            samples, manifest = _dialogue_fixture(
                "lmsys",
                SELECTION.LMSYS_SELECTED_SHA256,
                SELECTION.LMSYS_SELECTION_SEED,
                SELECTION.LMSYS_ROWS,
            )
            manifest["files_sha256"] = SELECTION.LMSYS_FILES_SHA256
            return samples, manifest

        @staticmethod
        def load_sharegpt_samples(*_args, **_kwargs):
            samples, manifest = _dialogue_fixture(
                "sharegpt4",
                SELECTION.SHAREGPT4_SELECTED_SHA256,
                SELECTION.SHAREGPT4_SELECTION_SEED,
                SELECTION.SHAREGPT4_ROWS,
            )
            manifest["sha256"] = SELECTION.SHAREGPT4_FILE_SHA256
            return samples, manifest

    with pytest.raises(ValueError, match="exactly 500"):
        SELECTION.load_lmsys_bottom500(
            FakeBase,
            "unused",
            sample_count=499,
            selection_seed=SELECTION.LMSYS_SELECTION_SEED,
        )
    with pytest.raises(ValueError, match="seed"):
        SELECTION.load_sharegpt_bottom500(
            FakeBase,
            "unused",
            sample_count=500,
            selection_seed="changed",
        )
    with pytest.raises(ValueError, match="internally inconsistent"):
        SELECTION.load_lmsys_bottom500(
            FakeBase,
            "unused",
            sample_count=500,
            selection_seed=SELECTION.LMSYS_SELECTION_SEED,
        )


def test_multinews_rouge_adds_r1_r2_and_preserves_pinned_rl():
    from rouge import Rouge

    prediction = "alpha beta. gamma delta"
    reference = "alpha gamma. delta"
    expected = Rouge().get_scores([prediction], [reference], avg=True)
    observed = METRICS.multinews_rouge(prediction, [reference])
    assert observed == pytest.approx(
        {
            "rouge_1_f1": expected["rouge-1"]["f"],
            "rouge_2_f1": expected["rouge-2"]["f"],
            "rouge_l_f1": expected["rouge-l"]["f"],
        }
    )
    assert METRICS.sample_metrics(BASE, "multi_news", prediction, [reference]) == pytest.approx(
        observed
    )


def test_multinews_rouge_only_converts_empty_valueerror_to_zero(monkeypatch):
    class EmptyRouge:
        def get_scores(self, *_args, **_kwargs):
            raise ValueError("Hypothesis is empty.")

    class BadValueRouge:
        def get_scores(self, *_args, **_kwargs):
            raise ValueError("corrupt scorer state")

    class InterruptedRouge:
        def get_scores(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    class FakeRougeModule:
        Rouge = EmptyRouge

    monkeypatch.setitem(sys.modules, "rouge", FakeRougeModule())
    assert METRICS.multinews_rouge("", ["reference"]) == {
        "rouge_1_f1": 0.0,
        "rouge_2_f1": 0.0,
        "rouge_l_f1": 0.0,
    }
    FakeRougeModule.Rouge = BadValueRouge
    with pytest.raises(ValueError, match="corrupt scorer"):
        METRICS.multinews_rouge("prediction", ["reference"])
    FakeRougeModule.Rouge = InterruptedRouge
    with pytest.raises(KeyboardInterrupt):
        METRICS.multinews_rouge("prediction", ["reference"])


def test_hotpot_and_dialogue_metrics_are_exactly_base_compatible():
    qa_prediction = "<think>old answer</think>\nFinal answer: Paris"
    references = ["Paris"]
    assert METRICS.sample_metrics(
        BASE, "hotpotqa", qa_prediction, references
    ) == BASE.qa_metrics(qa_prediction, references)

    dialog_prediction = "Paris is the capital of France."
    dialog_references = ["The capital of France is Paris."]
    expected = BASE._sample_metrics("lmsys", dialog_prediction, dialog_references)
    assert METRICS.sample_metrics(
        BASE, "lmsys", dialog_prediction, dialog_references
    ) == pytest.approx(expected)
    expected_share = BASE._sample_metrics(
        "sharegpt4", dialog_prediction, dialog_references
    )
    assert METRICS.sample_metrics(
        BASE, "sharegpt4", dialog_prediction, dialog_references
    ) == pytest.approx(expected_share)


def test_metric_provenance_discloses_all_three_rouge_metrics():
    provenance = METRICS.metric_implementation_provenance(BASE)
    settings = provenance["rouge"]["settings"]
    assert settings["metrics"] == ["rouge-1", "rouge-2", "rouge-l"]
    assert settings["exclusive"] is True
    assert "empty" in settings["empty_input_policy"]

