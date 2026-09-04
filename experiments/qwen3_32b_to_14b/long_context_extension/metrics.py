"""Frozen quality metrics for the 32B -> 14B long-context extension."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


def multinews_rouge(
    prediction: str, references: Sequence[str]
) -> dict[str, float]:
    """Return per-sample ROUGE-1/2/L F1 under the pinned forward scorer.

    The only converted failure is the scorer's documented empty-input
    ``ValueError``.  Interrupts, runtime failures, and unrelated ValueErrors
    propagate so a broken formal scorer cannot silently become a quality zero.
    """

    if not references:
        raise ValueError("ROUGE sample has no references")
    try:
        from rouge import Rouge
    except ImportError as exc:  # pragma: no cover - pinned formal dependency
        raise RuntimeError("rouge==1.0.1 is required for this evaluation") from exc

    scorer = Rouge()
    values: dict[str, list[float]] = {
        "rouge_1_f1": [],
        "rouge_2_f1": [],
        "rouge_l_f1": [],
    }
    for reference in references:
        try:
            score = scorer.get_scores([str(prediction)], [str(reference)], avg=True)
        except ValueError as exc:
            # rouge==1.0.1 raises precisely these errors after its sentence
            # normalization reduces a hypothesis/reference to no tokens.
            message = str(exc).casefold()
            if "empty" not in message:
                raise
            for metric_values in values.values():
                metric_values.append(0.0)
            continue
        values["rouge_1_f1"].append(float(score["rouge-1"]["f"]))
        values["rouge_2_f1"].append(float(score["rouge-2"]["f"]))
        values["rouge_l_f1"].append(float(score["rouge-l"]["f"]))
    return {name: max(metric_values) for name, metric_values in values.items()}


def sample_metrics(
    base: Any,
    task: str,
    prediction: str,
    references: Sequence[str],
) -> dict[str, Any]:
    """Score one sample while preserving the frozen forward task semantics."""

    if task == "hotpotqa":
        return dict(base.qa_metrics(prediction, references))
    if task == "multi_news":
        return multinews_rouge(prediction, references)
    if task in tuple(base.DIALOG_TASKS):
        if not references:
            raise ValueError(f"{task} sample has no references")
        return {
            "reference_rouge_l_f1": base.rouge_l_f1(prediction, references),
            "reference_chrf": base.chrf_score(prediction, references),
            "reference_normalized_token_f1": max(
                base._token_f1_normalized(prediction, reference)
                for reference in references
            ),
        }
    raise ValueError(f"unsupported task: {task}")


def metric_implementation_provenance(base: Any) -> dict[str, Any]:
    """Extend the base byte-level metric provenance with ROUGE-1/2 settings."""

    provenance = copy.deepcopy(base.metric_implementation_provenance())
    rouge = provenance.get("rouge")
    if not isinstance(rouge, Mapping):
        raise ValueError("base metric provenance is missing ROUGE")
    rouge = dict(rouge)
    settings = dict(rouge.get("settings", {}))
    settings.pop("metric", None)
    settings.update(
        {
            "metrics": ["rouge-1", "rouge-2", "rouge-l"],
            "stat": "f",
            "avg": True,
            "exclusive": True,
            "stemming": False,
            "aggregation": "per-sample then macro mean",
            "empty_input_policy": (
                "only scorer ValueError containing 'empty' becomes zero"
            ),
        }
    )
    rouge["settings"] = settings
    provenance["rouge"] = rouge
    return provenance

