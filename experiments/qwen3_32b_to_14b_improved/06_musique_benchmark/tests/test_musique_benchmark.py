from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.musique_benchmark import (  # noqa: E402
    MuSiQueError,
    answer_exact,
    answer_f1,
    build_prompt,
    load_config,
    normalize_answer,
    prepare_registry,
    score_answer,
    sha256_file,
)


def _official_answer_module():
    path = ROOT / "vendor" / "musique" / "metrics" / "answer.py"
    specification = importlib.util.spec_from_file_location("official_musique_answer", path)
    assert specification and specification.loader
    module = importlib.util.module_from_spec(specification)
    sys.path.insert(0, str(ROOT / "vendor" / "musique"))
    try:
        specification.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_pinned_payload_hashes_and_counts() -> None:
    config = load_config(ROOT)
    for key in ("answerable_train", "answerable_dev", "single_hop_leakage_registry"):
        specification = config["source"][key]
        path = ROOT / specification["relative_path"]
        assert path.stat().st_size == specification["nbytes"]
        assert sha256_file(path) == specification["sha256"]


@pytest.mark.parametrize(
    ("gold", "prediction"),
    [
        ("The Beatles", "Beatles"),
        ("Miquette Giraudy", "miquette giraudy"),
        ("1960", "1960."),
        ("A, B", "a b"),
        ("New York City", "York"),
        ("", ""),
    ],
)
def test_metric_matches_official(gold: str, prediction: str) -> None:
    official = _official_answer_module()
    assert normalize_answer(gold) == official.normalize_answer(gold)
    assert answer_exact(gold, prediction) == official.compute_exact(gold, prediction)
    assert answer_f1(gold, prediction) == official.compute_f1(gold, prediction)


def test_alias_maximum() -> None:
    result = score_answer("NYC", "New York City", ["NYC", "New York"])
    assert result == {"answer_em": 1.0, "answer_f1": 1.0}


def test_prompt_uses_all_paragraphs_in_official_order() -> None:
    path = ROOT / "upstream_raw" / "data" / "musique_ans_v1.0_train.jsonl"
    record = json.loads(path.open(encoding="utf-8").readline())
    prompt = build_prompt(record)
    for paragraph in record["paragraphs"]:
        assert f"[{paragraph['idx']}] {paragraph['title']}" in prompt
    positions = [prompt.index(f"[{index}]") for index in range(len(record["paragraphs"]))]
    assert positions == sorted(positions)
    assert prompt.endswith("\nAnswer:")


def test_prepare_registry_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    first = ROOT / "artifacts" / f"pytest_{tmp_path.name}_one"
    second = ROOT / "artifacts" / f"pytest_{tmp_path.name}_two"
    try:
        prepare_registry(ROOT, first)
        prepare_registry(ROOT, second)
        screen_one = json.loads((first / "train_screen.json").read_text())
        screen_two = json.loads((second / "train_screen.json").read_text())
        confirm = json.loads((first / "train_confirm.json").read_text())
        locked = json.loads((first / "official_dev_locked.json").read_text())
        assert screen_one == screen_two
        assert screen_one["count"] == 256
        assert confirm["count"] == 512
        assert not (
            {row["id"] for row in screen_one["records"]}
            & {row["id"] for row in confirm["records"]}
        )
        assert locked["count"] == 2417
        assert locked["records_embedded"] is False
        assert locked["state"] == "sealed_until_candidate_freeze"
    finally:
        shutil.rmtree(first, ignore_errors=True)
        shutil.rmtree(second, ignore_errors=True)


def test_prepare_rejects_output_outside_benchmark() -> None:
    with pytest.raises(MuSiQueError, match="artifacts"):
        prepare_registry(ROOT, ROOT.parent / "bad_musique_output")
