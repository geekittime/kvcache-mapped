from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_long_context_multinews_500.py"
SPEC = importlib.util.spec_from_file_location("evaluate_multinews_500_tested", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MULTI = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MULTI
SPEC.loader.exec_module(MULTI)


@dataclass(frozen=True)
class Sample:
    index: int
    sample_id: str
    record_sha256: str


def _samples() -> list[Sample]:
    return [
        Sample(index, f"multi-news-test-{index:04d}", f"{index:064x}")
        for index in range(MULTI.SOURCE_ROWS)
    ]


def test_bottom500_selection_is_deterministic_unique_and_canonical_order():
    samples = _samples()
    manifest = {"split": "official_test", "rows": MULTI.SOURCE_ROWS}
    first, first_manifest = MULTI.select_multinews_bottom_k(
        samples, manifest, enforce_frozen_identity=False
    )
    second, second_manifest = MULTI.select_multinews_bottom_k(
        samples, manifest, enforce_frozen_identity=False
    )

    assert len(first) == MULTI.SAMPLE_COUNT
    assert [sample.index for sample in first] == sorted(sample.index for sample in first)
    assert len({sample.index for sample in first}) == MULTI.SAMPLE_COUNT
    assert [sample.index for sample in first] == [sample.index for sample in second]
    assert first_manifest == second_manifest

    selection = first_manifest["formal_selection"]
    assert selection["source_rows"] == MULTI.SOURCE_ROWS
    assert selection["selected_rows"] == MULTI.SAMPLE_COUNT
    assert len(selection["selected"]) == MULTI.SAMPLE_COUNT
    assert selection["selected_sha256"] == MULTI._canonical_digest(selection["selected"])
    assert selection["selected_indices_sha256"] == MULTI._canonical_digest(
        [sample.index for sample in first]
    )


def test_selection_rank_is_independent_of_gold_bearing_record_hash():
    samples = _samples()
    altered = [
        Sample(sample.index, sample.sample_id, f"{MULTI.SOURCE_ROWS - sample.index:064x}")
        for sample in samples
    ]
    manifest = {"split": "official_test", "rows": MULTI.SOURCE_ROWS}
    original, original_manifest = MULTI.select_multinews_bottom_k(
        samples, manifest, enforce_frozen_identity=False
    )
    changed, changed_manifest = MULTI.select_multinews_bottom_k(
        altered, manifest, enforce_frozen_identity=False
    )
    assert [sample.index for sample in changed] == [sample.index for sample in original]
    assert (
        changed_manifest["formal_selection"]["selected_indices_sha256"]
        == original_manifest["formal_selection"]["selected_indices_sha256"]
    )
    # Record hashes remain in the post-selection identity manifest, so a gold
    # or source-content mutation is still detected even though it cannot alter
    # which indices were selected.
    assert (
        changed_manifest["formal_selection"]["selected_sha256"]
        != original_manifest["formal_selection"]["selected_sha256"]
    )


def test_bottom500_rejects_noncanonical_source_identity():
    samples = _samples()
    manifest = {"split": "official_test", "rows": MULTI.SOURCE_ROWS}
    with pytest.raises(ValueError, match="canonical"):
        MULTI.select_multinews_bottom_k(
            samples[1:] + samples[:1], manifest, enforce_frozen_identity=False
        )
    with pytest.raises(ValueError, match="exactly"):
        MULTI.select_multinews_bottom_k(
            samples[:-1], manifest, enforce_frozen_identity=False
        )


def test_formal_request_forbids_debug_scope_and_wrong_task():
    class Args:
        tasks = ("multi_news",)
        limit = None
        sample_indices = None

    raw = {
        "data": {
            "multi_news_sample_count": MULTI.SAMPLE_COUNT,
            "multi_news_selection_seed": MULTI.SELECTION_SEED,
        }
    }
    MULTI._validate_extension_request(raw, Args())

    wrong_task = Args()
    wrong_task.tasks = ("hotpotqa",)
    with pytest.raises(ValueError, match="exactly"):
        MULTI._validate_extension_request(raw, wrong_task)

    limited = Args()
    limited.limit = 500
    with pytest.raises(ValueError, match="forbids"):
        MULTI._validate_extension_request(raw, limited)


def test_default_arguments_bind_dedicated_config_and_task():
    values = MULTI._normalized_argv(["--prepare-only"])
    assert values[:4] == [
        "--tasks",
        "multi_news",
        "--config",
        str(MULTI.DEFAULT_CONFIG),
    ]
    assert values[-1] == "--prepare-only"


def test_real_data_selection_matches_all_frozen_digests():
    base = MULTI._load_base_runner()
    source = ROOT / "data/MultiNews/data/test.src.cleaned"
    target = ROOT / "data/MultiNews/data/test.tgt"
    samples, manifest = base.load_multi_news_samples(source, target)
    selected, selected_manifest = MULTI.select_multinews_bottom_k(samples, manifest)
    selection = selected_manifest["formal_selection"]
    assert len(selected) == MULTI.SAMPLE_COUNT
    assert selection["selected_sha256"] == MULTI.EXPECTED_SELECTED_SHA256
    assert (
        selection["selected_indices_sha256"]
        == MULTI.EXPECTED_SELECTED_INDICES_SHA256
    )


def test_rouge_does_not_hide_baseexception(monkeypatch):
    class RaisingRouge:
        def get_scores(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    class FakeRougeModule:
        Rouge = RaisingRouge

    monkeypatch.setitem(sys.modules, "rouge", FakeRougeModule())
    with pytest.raises(KeyboardInterrupt):
        MULTI.rouge_l_f1_without_baseexception("prediction", ["reference"])


def test_rouge_only_converts_empty_input_valueerror_to_zero(monkeypatch):
    class EmptyInputRouge:
        def get_scores(self, *_args, **_kwargs):
            raise ValueError("Hypothesis is empty.")

    class BrokenRouge:
        def get_scores(self, *_args, **_kwargs):
            raise RuntimeError("unexpected scorer failure")

    class FakeRougeModule:
        Rouge = EmptyInputRouge

    monkeypatch.setitem(sys.modules, "rouge", FakeRougeModule())
    assert MULTI.rouge_l_f1_without_baseexception("", ["reference"]) == 0.0

    FakeRougeModule.Rouge = BrokenRouge
    with pytest.raises(RuntimeError, match="unexpected scorer failure"):
        MULTI.rouge_l_f1_without_baseexception("prediction", ["reference"])
