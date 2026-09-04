"""Pinned HotpotQA and MuSiQue answer-only metrics.

Headline fields use the raw decoded model continuation exactly.  The optional
gold-independent extraction is a diagnostic and is never substituted into an
official score.
"""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Sequence
import unicodedata

from .contracts import ContractError


def normalize_answer(value: str) -> str:
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _tokens(value: str) -> list[str]:
    return normalize_answer(value).split() if value else []


def musique_exact(gold: str, prediction: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(prediction))


def musique_f1(gold: str, prediction: str) -> float:
    gold_tokens = _tokens(gold)
    prediction_tokens = _tokens(prediction)
    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    overlap = sum((Counter(gold_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def hotpot_exact(gold: str, prediction: str) -> float:
    return float(normalize_answer(gold) == normalize_answer(prediction))


def hotpot_f1(gold: str, prediction: str) -> float:
    normalized_gold = normalize_answer(gold)
    normalized_prediction = normalize_answer(prediction)
    special = {"yes", "no", "noanswer"}
    if normalized_prediction in special and normalized_prediction != normalized_gold:
        return 0.0
    if normalized_gold in special and normalized_prediction != normalized_gold:
        return 0.0
    gold_tokens = normalized_gold.split()
    prediction_tokens = normalized_prediction.split()
    overlap = sum((Counter(gold_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER = re.compile(
    r"(?im)(?:^|\n)\s*(?:the\s+)?(?:final\s+answer|answer)\s*"
    r"(?:is\s*)?(?::|=|-)?\s*(.+?)\s*$"
)
_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")


def extract_answer_candidate(prediction: str) -> str:
    """Extract a final line without consulting any reference answer."""

    text = unicodedata.normalize("NFKC", str(prediction)).strip()
    text = _THINK_BLOCK.sub("\n", text).strip()
    marked = [match.group(1).strip() for match in _ANSWER_MARKER.finditer(text)]
    if marked:
        candidate = marked[-1].strip("` _")
        candidate = re.sub(r"^\*{1,2}(.+?)\*{1,2}([.!?]?)$", r"\1\2", candidate)
        nested = _BOXED.search(candidate)
        return nested.group(1).strip() if nested else candidate
    boxed = [match.group(1).strip() for match in _BOXED.finditer(text)]
    if boxed:
        return boxed[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def score_answer(
    dataset_id: str, prediction: str, references: Sequence[str]
) -> dict[str, float | str]:
    if not isinstance(prediction, str):
        raise ContractError("prediction must be a string")
    if not references or any(not isinstance(reference, str) for reference in references):
        raise ContractError("answer references must be non-empty strings")
    if dataset_id == "hotpotqa_train":
        if len(references) != 1:
            raise ContractError("HotpotQA requires exactly one official answer")
        official_f1 = hotpot_f1(references[0], prediction)
        official_em = hotpot_exact(references[0], prediction)
        extracted = extract_answer_candidate(prediction)
        extracted_f1 = hotpot_f1(references[0], extracted)
        extracted_em = hotpot_exact(references[0], extracted)
        metric_source = "HotpotQA_official_answer_semantics"
    elif dataset_id == "musique_train":
        official_f1 = max(musique_f1(reference, prediction) for reference in references)
        official_em = max(musique_exact(reference, prediction) for reference in references)
        extracted = extract_answer_candidate(prediction)
        extracted_f1 = max(musique_f1(reference, extracted) for reference in references)
        extracted_em = max(musique_exact(reference, extracted) for reference in references)
        metric_source = "MuSiQue_official_max_over_primary_and_aliases"
    else:
        raise ContractError(f"unsupported dataset for answer scoring: {dataset_id}")
    return {
        "official_answer_f1": float(official_f1),
        "official_answer_em": float(official_em),
        "metric_source": metric_source,
        "diagnostic_extracted_answer": extracted,
        "diagnostic_extracted_answer_f1": float(extracted_f1),
        "diagnostic_extracted_answer_em": float(extracted_em),
    }


__all__ = [
    "extract_answer_candidate",
    "hotpot_exact",
    "hotpot_f1",
    "musique_exact",
    "musique_f1",
    "normalize_answer",
    "score_answer",
]
