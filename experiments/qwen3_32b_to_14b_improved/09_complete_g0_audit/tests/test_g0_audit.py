from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from g0_audit import Unit, _near_pairs, _verify_inputs, normalize


SIM = {
    "passage_shingle_words": 5,
    "passage_min_intersection": 20,
    "passage_min_containment": 0.5,
    "passage_min_jaccard": 0.3,
    "exact_normalized_unit_substring_min_characters": 120,
    "question_shingle_words": 3,
    "question_min_intersection": 4,
    "question_min_containment": 0.8,
    "question_min_jaccard": 0.7,
}


def unit(identity: str, kind: str, text: str) -> Unit:
    hashed = hashlib.sha256(identity.encode()).hexdigest()
    return Unit("x", "c", "d", hashed, hashed, kind, normalize(text))


def test_unicode_normalization_is_deterministic() -> None:
    assert normalize("Ｆoo—BAR\n baz") == "foo bar baz"


def test_synthetic_positive_passage() -> None:
    text = " ".join(f"unique{i}" for i in range(45))
    flags, _, _ = _near_pairs([unit("a", "passage", text)], [unit("b", "passage", text)], SIM, "test")
    assert len(flags) == 1


def test_synthetic_positive_question() -> None:
    left = "Which city contains the old museum near the river"
    right = "Which city contains the old museum near the river today"
    flags, _, _ = _near_pairs([unit("a", "question", left)], [unit("b", "question", right)], SIM, "test")
    assert len(flags) == 1


def test_synthetic_negative() -> None:
    a = " ".join(f"alpha{i}" for i in range(50))
    b = " ".join(f"omega{i}" for i in range(50))
    flags, _, _ = _near_pairs([unit("a", "passage", a)], [unit("b", "passage", b)], SIM, "test")
    assert flags == []


def test_pinned_input_tamper_fails_closed(tmp_path: Path) -> None:
    p = tmp_path / "x"
    p.write_bytes(b"original")
    cfg = {"inputs": {"x": {"path": str(p), "nbytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}}}
    _verify_inputs(cfg)
    p.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="pinned input mismatch"):
        _verify_inputs(cfg)


def test_actual_receipt_privacy_and_sidecar() -> None:
    receipt_path = ROOT / "artifacts" / "complete_g0_v1" / "complete_g0_receipt.json"
    sidecar_path = ROOT / "artifacts" / "complete_g0_v1" / "complete_g0_receipt.json.sha256.json"
    if not receipt_path.exists():
        pytest.skip("actual audit has not run")
    payload = receipt_path.read_bytes()
    receipt = json.loads(payload)
    sidecar = json.loads(sidecar_path.read_text())
    assert receipt["privacy"]["raw_text_written"] is False
    assert receipt["privacy"]["references_written"] is False
    assert receipt["privacy"]["model_outputs_read_or_written"] is False
    assert sidecar["sha256"] == hashlib.sha256(payload).hexdigest()
    assert sidecar["nbytes"] == len(payload)
    # No content-bearing source schema keys may occur anywhere in the receipt.
    forbidden_json_keys = {"question", "context", "references", "answer", "answer_aliases", "prediction", "model_output"}
    def walk(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden_json_keys)
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(receipt)
