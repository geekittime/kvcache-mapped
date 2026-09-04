from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cross_tier_audit import (  # noqa: E402
    AuditError,
    CalibrationText,
    Passage,
    audit_pairs,
    normalize_text,
    split_explicit_passages,
    verify_file,
    word_shingles,
    write_receipt_and_sidecar,
)


ALGORITHM = {
    "exact_min_normalized_chars": 120,
    "near_min_intersection": 20,
    "near_min_passage_containment": 0.5,
    "near_min_jaccard": 0.3,
    "top_candidates": 25,
}


def calibration(text: str) -> CalibrationText:
    normalized = normalize_text(text)
    return CalibrationText(
        split="train",
        sequence_index=0,
        token_ids_sha256="1" * 64,
        normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        normalized=normalized,
        shingles=word_shingles(normalized),
    )


def passage(text: str) -> Passage:
    normalized = normalize_text(text)
    return Passage(
        sample_id="sample",
        dataset_id="synthetic",
        source_record_sha256="2" * 64,
        passage_index=1,
        normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        normalized=normalized,
        shingles=word_shingles(normalized),
    )


def test_normalization_is_nfkc_casefold_alphanumeric_whitespace() -> None:
    assert normalize_text("  ＡB—Straße\tCAFÉ!  ") == "ab strasse café"


def test_explicit_passage_parser_excludes_title_metadata() -> None:
    context = "[Passage 1]\nTitle: Identity only\nAudited body text.\n\n[Passage 2]\nTitle: Next\nSecond body."
    assert split_explicit_passages(context) == [(1, "Audited body text."), (2, "Second body.")]


def test_synthetic_positive_exact_long_single_token() -> None:
    shared = "x" * 125
    top, summary = audit_pairs([calibration(f"left{shared}right")], [passage(f"up{shared}down")], ALGORITHM)
    assert summary["flagged_pairs"] == 1
    assert summary["flag_counts_by_criterion"]["exact_substring"] == 1
    assert top[0]["metrics"]["verified_exact_common_chars"] >= 120


def test_synthetic_positive_near_duplicate() -> None:
    shared = " ".join(f"word{i}" for i in range(30))
    top, summary = audit_pairs(
        [calibration(f"calibration prefix {shared} calibration suffix")],
        [passage(f"passage prefix {shared} passage suffix")],
        ALGORITHM,
    )
    assert summary["flagged_pairs"] == 1
    assert top[0]["metrics"]["shingle_intersection"] >= 20
    assert "intersection_and_passage_containment" in top[0]["triggered_criteria"]


def test_synthetic_negative() -> None:
    top, summary = audit_pairs(
        [calibration("red orange yellow green blue indigo violet")],
        [passage("one two three four five six seven")],
        ALGORITHM,
    )
    assert summary["flagged_pairs"] == 0
    assert top == []


def test_tampered_locked_input_fails_closed(tmp_path: Path) -> None:
    locked = tmp_path / "locked.bin"
    locked.write_bytes(b"expected")
    expected_sha = hashlib.sha256(locked.read_bytes()).hexdigest()
    locked.write_bytes(b"tampered")
    with pytest.raises(AuditError, match="SHA-256 mismatch"):
        verify_file({"path": str(locked), "sha256": expected_sha}, "synthetic")


def test_atomic_receipt_sidecar_binds_exact_bytes(tmp_path: Path) -> None:
    receipt_path, sidecar_path = write_receipt_and_sidecar(tmp_path, {"status": "pass", "schema_version": 1})
    sidecar = json.loads(sidecar_path.read_text())
    data = receipt_path.read_bytes()
    assert sidecar["nbytes"] == len(data)
    assert sidecar["sha256"] == hashlib.sha256(data).hexdigest()
    assert not list(tmp_path.glob("*.tmp"))
