"""Gold-independent extraction plus HotpotQA/MuSiQue answer metrics."""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Sequence
import unicodedata


def normalize_answer(value: str) -> str:
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    same = sum((Counter(predicted) & Counter(gold)).values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER = re.compile(
    r"(?im)(?:^|\n)\s*(?:the\s+)?(?:final\s+answer|answer)\s*"
    r"(?:is\s*)?(?::|=|-)?\s*(.+?)\s*$"
)
_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")


def extract_answer_candidate(prediction: str) -> str:
    """Extract without ever consulting a gold answer."""

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


def qa_metrics(prediction: str, references: Sequence[str]) -> dict[str, float | str]:
    if not references:
        raise ValueError("QA references must be non-empty")
    extracted = extract_answer_candidate(prediction)
    raw_f1 = max(token_f1(prediction, reference) for reference in references)
    raw_em = max(exact_match(prediction, reference) for reference in references)
    extracted_f1 = max(token_f1(extracted, reference) for reference in references)
    extracted_em = max(exact_match(extracted, reference) for reference in references)
    normalized_prediction = normalize_answer(prediction)
    containment = float(
        any(
            (gold := normalize_answer(reference))
            and re.search(
                rf"(?:^|\s){re.escape(gold)}(?:$|\s)", normalized_prediction
            )
            for reference in references
        )
    )
    return {
        "qa_f1": raw_f1,
        "qa_exact_match": raw_em,
        "extracted_answer": extracted,
        "extracted_qa_f1": extracted_f1,
        "extracted_qa_exact_match": extracted_em,
        "relaxed_qa_f1": max(raw_f1, extracted_f1),
        "relaxed_qa_exact_match": max(raw_em, extracted_em),
        "normalized_gold_containment": containment,
        # MuSiQue's official AnswerMetric is max-over-alias normalized F1/EM;
        # these raw fields match that metric before its presentation rounding.
        "musique_official_answer_f1": raw_f1,
        "musique_official_answer_em": raw_em,
    }


__all__ = [
    "exact_match",
    "extract_answer_candidate",
    "normalize_answer",
    "qa_metrics",
    "token_f1",
]
