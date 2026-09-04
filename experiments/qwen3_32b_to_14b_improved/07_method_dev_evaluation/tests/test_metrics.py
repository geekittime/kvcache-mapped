from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from src.metrics import (
    extract_answer_candidate,
    hotpot_exact,
    hotpot_f1,
    musique_exact,
    musique_f1,
    score_answer,
)


IMPROVED = Path(__file__).resolve().parents[2]
MUSIQUE_VENDOR = IMPROVED / "06_musique_benchmark" / "vendor" / "musique"


def _load(path: Path, name: str, import_root: Path | None = None):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    if import_root is not None:
        sys.path.insert(0, str(import_root))
    try:
        specification.loader.exec_module(module)
    finally:
        if import_root is not None:
            sys.path.pop(0)
    return module


@pytest.mark.parametrize(
    ("gold", "prediction"),
    [
        ("The Beatles", "Beatles"),
        ("yes", "yes indeed"),
        ("no", "yes"),
        ("1960", "1960."),
        ("", ""),
    ],
)
def test_hotpot_metric_matches_pinned_reference(gold: str, prediction: str) -> None:
    official = _load(
        MUSIQUE_VENDOR / "allennlp_lib" / "training" / "metrics" / "hotpotqa_eval.py",
        "pinned_hotpot_metric",
    )
    assert hotpot_exact(gold, prediction) == official.ans_em(prediction, gold)
    assert hotpot_f1(gold, prediction) == official.ans_f1(prediction, gold)[0]


@pytest.mark.parametrize(
    ("gold", "prediction"),
    [
        ("The Beatles", "Beatles"),
        ("New York City", "York"),
        ("1960", "1960."),
        ("", ""),
    ],
)
def test_musique_metric_matches_official(gold: str, prediction: str) -> None:
    official = _load(
        MUSIQUE_VENDOR / "metrics" / "answer.py",
        "pinned_musique_metric",
        MUSIQUE_VENDOR,
    )
    assert musique_exact(gold, prediction) == official.compute_exact(gold, prediction)
    assert musique_f1(gold, prediction) == official.compute_f1(gold, prediction)


def test_official_score_uses_raw_decode_not_extracted_diagnostic() -> None:
    prediction = "Explanation first\nFinal answer: Paris"
    assert extract_answer_candidate(prediction) == "Paris"
    result = score_answer("hotpotqa_train", prediction, ["Paris"])
    assert result["official_answer_em"] == 0.0
    assert result["diagnostic_extracted_answer_em"] == 1.0


def test_musique_takes_max_over_primary_and_aliases() -> None:
    result = score_answer("musique_train", "NYC", ["New York City", "NYC"])
    assert result["official_answer_f1"] == 1.0
    assert result["official_answer_em"] == 1.0

