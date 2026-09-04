"""Pinned normalized-token F1 and exact-match semantics for train evaluation."""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Sequence

from .contracts import ContractError


def normalize_answer(value: str) -> str:
    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _f1(gold: str, prediction: str, *, hotpot_special: bool) -> float:
    normalized_gold = normalize_answer(gold)
    normalized_prediction = normalize_answer(prediction)
    special = {"yes", "no", "noanswer"}
    if hotpot_special and (
        (normalized_prediction in special and normalized_prediction != normalized_gold)
        or (normalized_gold in special and normalized_prediction != normalized_gold)
    ):
        return 0.0
    gold_tokens = normalized_gold.split()
    prediction_tokens = normalized_prediction.split()
    if not gold_tokens or not prediction_tokens:
        # HotpotQA's pinned ans_f1 returns ZERO_METRIC when the token
        # intersection is empty, including two normalized-empty strings.
        # MuSiQue inherits SQuAD's explicit empty==empty special case.
        return 0.0 if hotpot_special else float(gold_tokens == prediction_tokens)
    overlap = sum((Counter(gold_tokens) & Counter(prediction_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2.0 * precision * recall / (precision + recall)


def score_answer(dataset_id: str, prediction: str, references: Sequence[str]) -> dict[str, float | str]:
    if not isinstance(prediction, str):
        raise ContractError("prediction must be text")
    if not references or any(not isinstance(item, str) or not item.strip() for item in references):
        raise ContractError("references must be non-empty strings")
    if dataset_id == "hotpotqa_train":
        if len(references) != 1:
            raise ContractError("HotpotQA requires exactly one official answer")
        f1 = _f1(references[0], prediction, hotpot_special=True)
        em = float(normalize_answer(references[0]) == normalize_answer(prediction))
        source = "HotpotQA_official_normalized_token_answer"
    elif dataset_id == "musique_train":
        f1 = max(_f1(reference, prediction, hotpot_special=False) for reference in references)
        em = max(
            float(normalize_answer(reference) == normalize_answer(prediction))
            for reference in references
        )
        source = "MuSiQue_official_max_over_primary_and_aliases"
    else:
        raise ContractError(f"unsupported dataset: {dataset_id}")
    return {
        "official_answer_f1": float(f1),
        "official_answer_em": float(em),
        "metric_source": source,
    }


__all__ = ["normalize_answer", "score_answer"]
