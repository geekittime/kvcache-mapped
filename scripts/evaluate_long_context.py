#!/usr/bin/env python3
"""Evaluate Qwen3-32B full prefill versus mapped Qwen3-14B KV caches.

This is deliberately separate from ``scripts/evaluate.py``: the latter is the
frozen paper-table protocol.  This runner covers the complete local MultiNews
test and HotpotQA dev-distractor splits plus deterministic held-out slices from
the train-only LMSYS and ShareGPT4 conversation datasets.

Every sample/mode pair is an atomic checkpoint.  An immutable run definition
binds models, mapper, data bytes, prompts, metrics, caps, truncation, and code;
rerunning the same output directory therefore resumes only validated work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import string
import sys
import traceback
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.config import load_yaml  # noqa: E402
from kvcache_transfer.evaluation import (  # noqa: E402
    MappedPrefixCacheBuilder,
    TransferEvaluationEngine,
    head_tail_truncate_token_ids,
)
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.modeling import load_qwen3  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_CONFIG = ROOT / "configs/eval_qwen3_14b_to_32b_long_context.yaml"
MODES = ("target_standalone", "mapped")
TASKS = ("multi_news", "hotpotqa", "lmsys", "sharegpt4")
DIALOG_TASKS = ("lmsys", "sharegpt4")
EXPECTED_CAPS = {
    "multi_news": 1024,
    "hotpotqa": 512,
    "lmsys": 1024,
    "sharegpt4": 1024,
}
PRIMARY_METRICS = {
    "multi_news": "rouge_l_f1",
    "hotpotqa": "relaxed_qa_f1",
    "lmsys": "reference_ppl",
    "sharegpt4": "reference_ppl",
}
SUMMARY_PRIMARY_METRICS = {
    "multi_news": "rouge_l_f1",
    "hotpotqa": "relaxed_qa_f1",
    "lmsys": "reference_corpus_ppl",
    "sharegpt4": "reference_corpus_ppl",
}
PROMPT_TEMPLATES = {
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give me "
        "the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of all news. "
        "\n\nNews:\n{context}\n\nNow, write a one-page summary of all the news."
        "\n\nSummary:"
    ),
}
FORMAL_CODE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "scripts/evaluate_long_context.py",
    "vendor/lm-evaluation-harness/lm_eval/tasks/longbench/metrics.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/transfer.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/latency.py",
)


@dataclass(frozen=True)
class EvaluationSample:
    task: str
    index: int
    sample_id: str
    record_sha256: str
    references: tuple[str, ...]
    input_text: str = ""
    context: str = ""
    messages: tuple[tuple[str, str], ...] = ()
    source_length: int | None = None


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    # Decimal JSON is architecture-independent and sufficient for provenance.
    return canonical_digest([int(token) for token in token_ids])


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def normalize_answer(value: str) -> str:
    """Exact LongBench/SQuAD English normalization for QA metrics."""

    text = str(value).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


_NON_VISIBLE_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Cs"})


def _visible_character_count(value: str) -> int:
    """Count semantic characters, excluding whitespace and invisible controls."""

    return sum(
        1
        for character in unicodedata.normalize("NFKC", str(value))
        if not character.isspace()
        and unicodedata.category(character) not in _NON_VISIBLE_UNICODE_CATEGORIES
    )


def _has_visible_text(value: str) -> bool:
    return _visible_character_count(value) > 0


def _token_f1_normalized(prediction: str, reference: str) -> float:
    from collections import Counter

    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted or not gold:
        # LongBench's vendored f1_score returns zero when the multiset
        # intersection is empty, including two empty token sequences.
        return 0.0
    common = Counter(predicted) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def _normalized_exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_ANSWER_MARKER = re.compile(
    r"(?im)(?:^|\n)\s*(?:the\s+)?(?:final\s+answer|answer)\s*"
    r"(?:is\s*)?(?::|=|-)?\s*(.+?)\s*$"
)
_BOXED = re.compile(r"\\boxed\s*\{([^{}]+)\}")


def extract_answer_candidate(prediction: str) -> str:
    """Extract one answer without consulting any gold reference.

    It removes completed Qwen thinking blocks, then prefers an explicit
    final-answer marker or ``\\boxed{}``; otherwise it takes the last non-empty
    line.  Crucially, extraction never sees a gold answer.
    """

    text = unicodedata.normalize("NFKC", str(prediction)).strip()
    text = _THINK_BLOCK.sub("\n", text).strip()
    marked = [match.group(1).strip() for match in _ANSWER_MARKER.finditer(text)]
    if marked:
        candidate = marked[-1].strip("` _")
        candidate = re.sub(r"^\*{1,2}(.+?)\*{1,2}([.!?]?)$", r"\1\2", candidate)
        nested_box = _BOXED.search(candidate)
        return nested_box.group(1).strip() if nested_box else candidate
    boxed = [match.group(1).strip() for match in _BOXED.finditer(text)]
    if boxed:
        return boxed[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text


def qa_metrics(prediction: str, references: Sequence[str]) -> dict[str, float | str]:
    if not references:
        raise ValueError("QA sample has no references")
    extracted = extract_answer_candidate(prediction)
    f1 = max(_token_f1_normalized(prediction, reference) for reference in references)
    em = max(_normalized_exact_match(prediction, reference) for reference in references)
    extracted_f1 = max(_token_f1_normalized(extracted, reference) for reference in references)
    extracted_em = max(_normalized_exact_match(extracted, reference) for reference in references)
    normalized_prediction = normalize_answer(prediction)
    containment = 0.0
    for reference in references:
        gold = normalize_answer(reference)
        if gold and re.search(rf"(?:^|\s){re.escape(gold)}(?:$|\s)", normalized_prediction):
            containment = 1.0
            break
    return {
        "qa_f1": f1,
        "qa_exact_match": em,
        "extracted_answer": extracted,
        "extracted_qa_f1": extracted_f1,
        "extracted_qa_exact_match": extracted_em,
        # This is the requested relaxed-but-reasonable headline.  It can only
        # improve formatting failures; there is no gold-dependent span search.
        "relaxed_qa_f1": max(f1, extracted_f1),
        "relaxed_qa_exact_match": max(em, extracted_em),
        "normalized_gold_containment": containment,
    }


def rouge_l_f1(prediction: str, references: Sequence[str]) -> float:
    """Pinned ``rouge==1.0.1`` ROUGE-L summary-level F1."""

    if not references:
        raise ValueError("ROUGE sample has no references")
    try:
        from rouge import Rouge
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("rouge==1.0.1 is required for this evaluation") from exc
    scorer = Rouge()
    scores: list[float] = []
    for reference in references:
        # The vendored LongBench metric deliberately turns malformed/empty
        # ROUGE inputs into zero instead of failing an entire run.
        try:
            score = scorer.get_scores([prediction], [reference], avg=True)
            scores.append(float(score["rouge-l"]["f"]))
        except BaseException:
            scores.append(0.0)
    return max(scores)


def chrf_score(prediction: str, references: Sequence[str]) -> float:
    if not references:
        raise ValueError("chrF sample has no references")
    try:
        from sacrebleu.metrics import CHRF
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("sacrebleu is required for dialogue chrF") from exc
    scorer = CHRF()
    return max(scorer.sentence_score(prediction, [reference]).score / 100.0 for reference in references)


def _canonical_record_sha(record: Mapping[str, Any]) -> str:
    return canonical_digest(record)


def _require_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{label} must be a {'possibly empty ' if allow_empty else ''}string")
    return value


def load_multi_news_samples(
    source_path: str | Path, target_path: str | Path
) -> tuple[list[EvaluationSample], dict[str, Any]]:
    """Load all 5,622 paired official MultiNews test lines."""

    source = Path(source_path).resolve()
    target = Path(target_path).resolve()
    if not source.is_file() or not target.is_file():
        raise FileNotFoundError("MultiNews test.src.cleaned/test.tgt is missing")
    source_lines = source.read_text(encoding="utf-8").splitlines()
    target_lines = target.read_text(encoding="utf-8").splitlines()
    if len(source_lines) != 5622 or len(target_lines) != 5622:
        raise ValueError(
            f"MultiNews official test requires 5622 aligned lines, got "
            f"{len(source_lines)}/{len(target_lines)}"
        )
    samples = []
    identities = []
    for index, (raw_document, raw_summary) in enumerate(
        zip(source_lines, target_lines, strict=True)
    ):
        # Exact transformation in the local official dataset loader.
        document = raw_document.strip().replace("NEWLINE_CHAR", "\n")
        summary = raw_summary.strip()
        if not summary:
            raise ValueError(f"MultiNews test target {index} is empty")
        record = {"document": document, "summary": summary}
        record_sha = _canonical_record_sha(record)
        sample_id = f"multi-news-test-{index:04d}"
        samples.append(
            EvaluationSample(
                task="multi_news",
                index=index,
                sample_id=sample_id,
                record_sha256=record_sha,
                references=(summary,),
                context=document,
                source_length=len(document),
            )
        )
        identities.append([sample_id, record_sha])
    manifest = {
        "split": "official_test",
        "rows": len(samples),
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "nbytes": source.stat().st_size,
        },
        "target": {
            "path": str(target),
            "sha256": sha256_file(target),
            "nbytes": target.stat().st_size,
        },
        "transform": "strip each line; replace literal NEWLINE_CHAR with newline in source",
        "sample_identity_sha256": canonical_digest(identities),
    }
    return samples, manifest


def _format_hotpot_context(value: Any, *, label: str) -> str:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}.context must be a non-empty list")
    passages = []
    for passage_index, passage in enumerate(value, start=1):
        if not isinstance(passage, list) or len(passage) != 2:
            raise ValueError(f"{label}.context[{passage_index}] is malformed")
        title = _require_string(passage[0], f"{label}.context.title")
        sentences = passage[1]
        if not isinstance(sentences, list) or not sentences:
            raise ValueError(f"{label}.context.sentences must be non-empty")
        text = "".join(
            _require_string(
                sentence,
                f"{label}.context.sentence",
                allow_empty=True,
            )
            for sentence in sentences
        ).strip()
        passages.append(f"[Passage {passage_index}]\nTitle: {title}\n{text}")
    return "\n\n".join(passages)


def load_hotpotqa_samples(
    json_path: str | Path,
) -> tuple[list[EvaluationSample], dict[str, Any]]:
    """Load the complete 7,405-row distractor development split with gold answers."""

    path = Path(json_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"HotpotQA dev-distractor file is missing: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != 7405:
        raise ValueError(f"HotpotQA dev-distractor must contain 7405 rows")
    samples = []
    identities = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"HotpotQA row {index} is not an object")
        required = {"_id", "answer", "question", "context"}
        if not required.issubset(record):
            raise ValueError(f"HotpotQA row {index} lacks required fields")
        sample_id = _require_string(record["_id"], f"hotpotqa[{index}]._id")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate HotpotQA id: {sample_id}")
        seen_ids.add(sample_id)
        answer = _require_string(record["answer"], f"hotpotqa[{index}].answer")
        question = _require_string(record["question"], f"hotpotqa[{index}].question")
        context = _format_hotpot_context(record["context"], label=f"hotpotqa[{index}]")
        record_sha = _canonical_record_sha(record)
        samples.append(
            EvaluationSample(
                task="hotpotqa",
                index=index,
                sample_id=sample_id,
                record_sha256=record_sha,
                references=(answer,),
                input_text=question,
                context=context,
                source_length=len(context),
            )
        )
        identities.append([sample_id, record_sha])
    manifest = {
        "path": str(path),
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "split": "dev_distractor",
        "rows": len(samples),
        "gold_answers_available": True,
        "test_split_excluded": "HotpotQA test has no public gold answers",
        "context_format": "numbered passages with title and concatenated source sentences",
        "sample_identity_sha256": canonical_digest(identities),
    }
    return samples, manifest


def _conversation_candidate(
    *,
    task: str,
    source_locator: str,
    source_id: str,
    conversations: Any,
    minimum_pairs: int,
    minimum_reference_visible_characters: int,
) -> tuple[EvaluationSample, str, str] | None:
    if minimum_pairs <= 0:
        raise ValueError("minimum_pairs must be positive")
    if minimum_reference_visible_characters <= 0:
        raise ValueError("minimum_reference_visible_characters must be positive")
    if not isinstance(conversations, list) or len(conversations) < 2 * minimum_pairs:
        return None
    normalized: list[tuple[str, str]] = []
    for turn_index, turn in enumerate(conversations):
        if not isinstance(turn, Mapping):
            return None
        role = turn.get("from")
        expected = "human" if turn_index % 2 == 0 else "gpt"
        value = turn.get("value")
        if role != expected or not isinstance(value, str) or not _has_visible_text(value):
            return None
        normalized.append(("user" if role == "human" else "assistant", value))
    if normalized[-1][0] != "assistant":
        return None
    if _visible_character_count(normalized[-1][1]) < minimum_reference_visible_characters:
        return None

    def dedup_text(value: str) -> str:
        # This normalization is used only to form stable de-duplication groups;
        # model inputs and references retain their original text exactly.
        normalized_value = unicodedata.normalize("NFKC", value)
        without_invisible_controls = "".join(
            character
            for character in normalized_value
            if unicodedata.category(character) not in _NON_VISIBLE_UNICODE_CATEGORIES
        )
        return " ".join(without_invisible_controls.casefold().split())

    dedup_conversation = [(role, dedup_text(value)) for role, value in normalized]
    conversation_sha = canonical_digest(dedup_conversation)
    first_prompt_sha = canonical_digest(dedup_conversation[0][1])
    record_sha = canonical_digest(
        {"source_id": source_id, "conversations": normalized}
    )
    return (
        EvaluationSample(
            task=task,
            index=-1,
            sample_id=f"{task}-{source_locator}-{conversation_sha[:12]}",
            record_sha256=record_sha,
            references=(normalized[-1][1],),
            messages=tuple(normalized[:-1]),
            source_length=len(normalized),
        ),
        conversation_sha,
        first_prompt_sha,
    )


def _select_conversation_heldout(
    records: Iterable[tuple[int, str, str, Any]],
    *,
    task: str,
    sample_count: int,
    selection_seed: str,
    minimum_pairs: int,
    minimum_reference_visible_characters: int = 1,
    evaluation_scope: str,
) -> tuple[list[EvaluationSample], dict[str, Any]]:
    """Deduplicate conversations and retain deterministic SHA256 bottom-k."""

    import heapq

    if sample_count <= 0:
        raise ValueError("conversation sample_count must be positive")
    seed = str(selection_seed)
    seen_conversations: set[str] = set()
    seen_first_prompts: set[str] = set()
    heap: list[tuple[int, str, str, int, str, EvaluationSample]] = []
    total_rows = 0
    eligible_rows = 0
    for source_order, source_locator, source_id, conversations in records:
        total_rows += 1
        value = _conversation_candidate(
            task=task,
            source_locator=source_locator,
            source_id=source_id,
            conversations=conversations,
            minimum_pairs=minimum_pairs,
            minimum_reference_visible_characters=minimum_reference_visible_characters,
        )
        if value is None:
            continue
        eligible_rows += 1
        candidate, conversation_sha, first_prompt_sha = value
        if conversation_sha in seen_conversations:
            continue
        seen_conversations.add(conversation_sha)
        # Keep the first source-row representative for a normalized opening
        # prompt.  Source order and every source-file digest are pinned in the
        # run manifest, so this tie-break is deterministic and auditable.
        if first_prompt_sha in seen_first_prompts:
            continue
        seen_first_prompts.add(first_prompt_sha)
        rank = hashlib.sha256(
            f"{seed}\0{conversation_sha}".encode("utf-8")
        ).hexdigest()
        rank_int = int(rank, 16)
        entry = (
            -rank_int,
            conversation_sha,
            first_prompt_sha,
            source_order,
            rank,
            candidate,
        )
        if len(heap) < sample_count:
            heapq.heappush(heap, entry)
        elif rank_int < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    if len(heap) != sample_count:
        raise ValueError(
            f"{task} has only {len(seen_first_prompts)} eligible normalized first-prompt groups"
        )
    chosen = sorted(heap, key=lambda item: item[3])
    samples = [replace(item[5], index=index) for index, item in enumerate(chosen)]
    selected_records = [
        {
            "evaluation_index": index,
            "source_order": source_order,
            "sample_id": sample.sample_id,
            "record_sha256": sample.record_sha256,
            "conversation_sha256": conversation_sha,
            "first_prompt_sha256": first_prompt_sha,
            "selection_rank_sha256": rank,
        }
        for index, (
            _,
            conversation_sha,
            first_prompt_sha,
            source_order,
            rank,
            sample,
        ) in enumerate(chosen)
    ]
    return samples, {
        "rows": total_rows,
        "eligible_rows": eligible_rows,
        "unique_eligible_conversations": len(seen_conversations),
        "duplicate_eligible_rows_removed": eligible_rows - len(seen_conversations),
        "normalized_first_prompt_groups": len(seen_first_prompts),
        "first_prompt_group_duplicates_removed": (
            len(seen_conversations) - len(seen_first_prompts)
        ),
        "selected_rows": len(samples),
        "selection": {
            "name": "sha256_bottom_k_over_normalized_first_prompt_groups",
            "seed": seed,
            "minimum_complete_human_gpt_pairs": minimum_pairs,
            "minimum_reference_visible_characters": minimum_reference_visible_characters,
            "evaluation_scope": evaluation_scope,
            "eligibility": (
                f"at least {minimum_pairs} visibly non-empty strictly alternating human/gpt "
                f"pair(s), final reference has at least "
                f"{minimum_reference_visible_characters} visible Unicode characters, "
                "starts human, "
                "ends gpt; final gpt turn held out as reference; NFKC+casefold+collapsed-"
                "whitespace conversation content is deduplicated, then the first source-row "
                "representative of each equivalently normalized opening prompt is retained"
            ),
            "official_test_split": False,
            "disclosure": "train-only data; deterministic local held-out protocol",
        },
        "selected": selected_records,
        "selected_sha256": canonical_digest(selected_records),
    }


def load_sharegpt_samples(
    jsonl_path: str | Path, *, sample_count: int, selection_seed: str
) -> tuple[list[EvaluationSample], dict[str, Any]]:
    path = Path(jsonl_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ShareGPT4 JSONL is missing: {path}")

    def records() -> Iterable[tuple[int, str, str, Any]]:
        with path.open("rb") as handle:
            for line_index, raw_line in enumerate(handle):
                stripped = raw_line.rstrip(b"\r\n")
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid ShareGPT JSON on line {line_index + 1}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ValueError(f"ShareGPT line {line_index + 1} is not an object")
                yield (
                    line_index,
                    f"line-{line_index:06d}",
                    str(record.get("id", f"line-{line_index:06d}")),
                    record.get("conversations"),
                )

    samples, selection = _select_conversation_heldout(
        records(),
        task="sharegpt4",
        sample_count=sample_count,
        selection_seed=selection_seed,
        minimum_pairs=2,
        minimum_reference_visible_characters=16,
        evaluation_scope="multi_turn_conversation_continuation",
    )
    return samples, {
        "path": str(path),
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "split": "train",
        **selection,
    }


def load_lmsys_samples(
    parquet_dir: str | Path, *, sample_count: int, selection_seed: str
) -> tuple[list[EvaluationSample], dict[str, Any]]:
    import pyarrow.parquet as pq

    root = Path(parquet_dir).resolve()
    paths = sorted(root.glob("train-*-of-00004.parquet"))
    if len(paths) != 4:
        raise FileNotFoundError(f"LMSYS requires exactly four train parquet shards: {root}")
    files = []
    source_offsets = []
    total = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        source_offsets.append(total)
        total += rows
        files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "nbytes": path.stat().st_size,
                "rows": rows,
            }
        )
    if total != 273402:
        raise ValueError(f"LMSYS four-shard train split must contain 273402 rows, got {total}")

    def records() -> Iterable[tuple[int, str, str, Any]]:
        for shard_index, (path, offset) in enumerate(zip(paths, source_offsets, strict=True)):
            local_index = 0
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(
                batch_size=1024, columns=["id", "conversations"]
            ):
                for record in batch.to_pylist():
                    yield (
                        offset + local_index,
                        f"shard-{shard_index}-row-{local_index:06d}",
                        str(record.get("id", "")),
                        record.get("conversations"),
                    )
                    local_index += 1

    samples, selection = _select_conversation_heldout(
        records(),
        task="lmsys",
        sample_count=sample_count,
        selection_seed=selection_seed,
        minimum_pairs=1,
        minimum_reference_visible_characters=1,
        evaluation_scope="single_turn_assistant_response_quality",
    )
    return samples, {
        "directory": str(root),
        "files": files,
        "files_sha256": canonical_digest(files),
        "split": "train",
        **selection,
    }


def load_all_samples(
    configuration: Mapping[str, Any], tasks: Sequence[str]
) -> tuple[dict[str, list[EvaluationSample]], dict[str, Any]]:
    samples: dict[str, list[EvaluationSample]] = {}
    manifest: dict[str, Any] = {}
    if "multi_news" in tasks:
        samples["multi_news"], manifest["multi_news"] = load_multi_news_samples(
            configuration["multi_news_source"], configuration["multi_news_target"]
        )
    if "hotpotqa" in tasks:
        samples["hotpotqa"], manifest["hotpotqa"] = load_hotpotqa_samples(
            configuration["hotpotqa_dev_distractor"]
        )
    if "lmsys" in tasks:
        samples["lmsys"], manifest["lmsys"] = load_lmsys_samples(
            configuration["lmsys_parquet_dir"],
            sample_count=configuration["lmsys_sample_count"],
            selection_seed=configuration["lmsys_selection_seed"],
        )
    if "sharegpt4" in tasks:
        samples["sharegpt4"], manifest["sharegpt4"] = load_sharegpt_samples(
            configuration["sharegpt_jsonl"],
            sample_count=configuration["sharegpt_sample_count"],
            selection_seed=configuration["sharegpt_selection_seed"],
        )
    for task in tasks:
        if task not in samples:
            raise ValueError(f"no samples loaded for {task}")
    return samples, manifest


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("tokenizer unexpectedly returned a batch")
        value = value[0]
    if not isinstance(value, Sequence):
        raise TypeError("tokenizer did not return token IDs")
    return [int(token) for token in value]


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    return _as_token_ids(tokenizer.encode(text, add_special_tokens=False))


def _decode_text(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(
            tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def _apply_qwen_chat_template(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    value = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _as_token_ids(value)


def build_prompt_token_ids(
    tokenizer: Any,
    sample: EvaluationSample,
    *,
    prompt_mode: str,
) -> tuple[list[int], dict[str, Any]]:
    if prompt_mode != "qwen_chat_no_think":
        raise ValueError(
            f"{sample.task} must use qwen_chat_no_think, got {prompt_mode!r}"
        )
    if sample.task in DIALOG_TASKS:
        if prompt_mode != "qwen_chat_no_think":
            raise ValueError(f"{sample.task} requires qwen_chat_no_think")
        messages = [
            {"role": role, "content": content} for role, content in sample.messages
        ]
        token_ids = _apply_qwen_chat_template(tokenizer, messages)
        descriptor = {
            "prompt_mode": prompt_mode,
            "message_count": len(messages),
            "template_arguments": {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        }
    else:
        template = PROMPT_TEMPLATES[sample.task]
        prompt = template.format(context=sample.context, input=sample.input_text)
        token_ids = _apply_qwen_chat_template(
            tokenizer, [{"role": "user", "content": prompt}]
        )
        descriptor = {
            "prompt_mode": prompt_mode,
            "template_sha256": bytes_sha256(template.encode("utf-8")),
            "rendered_prompt_sha256": bytes_sha256(prompt.encode("utf-8")),
            "template_arguments": {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        }
    if not token_ids:
        raise ValueError(f"{sample.task}/{sample.sample_id} produced an empty prompt")
    descriptor["original_prompt_tokens"] = len(token_ids)
    descriptor["original_prompt_token_ids_sha256"] = token_ids_sha256(token_ids)
    return token_ids, descriptor


def _dialog_generation_prompt(
    tokenizer: Any,
    sample: EvaluationSample,
    *,
    budget: int,
    original_prompt_ids: Sequence[int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Truncate a dialogue without cutting through chat-template structure.

    Oldest complete user/assistant pairs are removed first.  If the final user
    message alone still does not fit, only that message's content tokens are
    head-tail truncated, decoded, and passed through the chat template again.
    Thus the exact model input always remains a syntactically valid Qwen chat.
    """

    if budget <= 0:
        raise ValueError("dialog generation prompt budget must be positive")
    messages = [
        {"role": role, "content": content} for role, content in sample.messages
    ]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("dialog held-out prompt must end in a user turn")
    original_ids = (
        [int(token) for token in original_prompt_ids]
        if original_prompt_ids is not None
        else _apply_qwen_chat_template(tokenizer, messages)
    )
    base = {
        "policy": "drop_oldest_complete_pairs_then_head_tail_final_user_content",
        "original_tokens": len(original_ids),
        "original_messages": len(messages),
    }
    if len(original_ids) <= budget:
        return original_ids, {
            **base,
            "truncated": False,
            "used_tokens": len(original_ids),
            "removed_tokens": 0,
            "dropped_message_pairs": 0,
            "dropped_messages": 0,
            "retained_messages": len(messages),
            "final_user_content_truncation": None,
        }

    pair_count = (len(messages) - 1) // 2
    # Prompt length is monotone as complete leading pairs are removed.  Binary
    # search finds the smallest removal that fits without repeatedly tokenizing
    # every giant suffix of long dialogue transcripts.
    low = 1
    high = pair_count
    fitting: tuple[int, list[dict[str, str]], list[int]] | None = None
    while low <= high:
        dropped_pairs = (low + high) // 2
        retained = messages[2 * dropped_pairs :]
        candidate_ids = _apply_qwen_chat_template(tokenizer, retained)
        if len(candidate_ids) <= budget:
            fitting = (dropped_pairs, retained, candidate_ids)
            high = dropped_pairs - 1
        else:
            low = dropped_pairs + 1
    if fitting is not None:
        dropped_pairs, retained, candidate_ids = fitting
        return candidate_ids, {
            **base,
            "truncated": True,
            "used_tokens": len(candidate_ids),
            "removed_tokens": len(original_ids) - len(candidate_ids),
            "dropped_message_pairs": dropped_pairs,
            "dropped_messages": 2 * dropped_pairs,
            "retained_messages": len(retained),
            "final_user_content_truncation": None,
        }

    # Only the final user message remains.  Derive a conservative content
    # budget from the empty-message template, then tighten until the fully
    # re-rendered chat fits.  This avoids assuming any Qwen template literals.
    content = str(messages[-1]["content"])
    content_ids = _encode_text(tokenizer, content)
    empty_prompt_ids = _apply_qwen_chat_template(
        tokenizer, [{"role": "user", "content": ""}]
    )
    if len(empty_prompt_ids) > budget:
        raise ValueError(
            "generation budget is smaller than an empty Qwen user chat prompt"
        )
    kept_budget = min(len(content_ids), max(0, budget - len(empty_prompt_ids)))
    while True:
        if kept_budget == 0:
            kept_content_ids: list[int] = []
            content_truncation: dict[str, Any] = {
                "policy": "head_tail",
                "truncated": bool(content_ids),
                "original_tokens": len(content_ids),
                "used_tokens": 0,
                "removed_tokens": len(content_ids),
                "head_tokens": 0,
                "tail_tokens": 0,
            }
        else:
            kept_content_ids, content_truncation = head_tail_truncate_token_ids(
                content_ids, kept_budget
            )
        rendered_content = _decode_text(tokenizer, kept_content_ids)
        candidate_ids = _apply_qwen_chat_template(
            tokenizer, [{"role": "user", "content": rendered_content}]
        )
        if len(candidate_ids) <= budget:
            break
        overflow = len(candidate_ids) - budget
        next_budget = max(0, kept_budget - max(1, overflow))
        if next_budget == kept_budget:
            raise AssertionError("dialog content truncation made no progress")
        kept_budget = next_budget
    return candidate_ids, {
        **base,
        "truncated": True,
        "used_tokens": len(candidate_ids),
        "removed_tokens": len(original_ids) - len(candidate_ids),
        "dropped_message_pairs": pair_count,
        "dropped_messages": 2 * pair_count,
        "retained_messages": 1,
        "final_user_content_original_sha256": bytes_sha256(content.encode("utf-8")),
        "final_user_content_rendered_sha256": bytes_sha256(
            rendered_content.encode("utf-8")
        ),
        "final_user_content_truncation": content_truncation,
    }


def generation_prompt_token_ids(
    tokenizer: Any,
    sample: EvaluationSample,
    *,
    prompt_mode: str,
    budget: int,
) -> tuple[list[int], dict[str, Any], dict[str, Any], list[int]]:
    """Return original descriptor plus task-appropriate bounded prompt IDs."""

    prompt_ids, descriptor = build_prompt_token_ids(
        tokenizer, sample, prompt_mode=prompt_mode
    )
    if sample.task in DIALOG_TASKS:
        used_ids, truncation = _dialog_generation_prompt(
            tokenizer, sample, budget=budget, original_prompt_ids=prompt_ids
        )
    else:
        used_ids, truncation = head_tail_truncate_token_ids(prompt_ids, budget)
    if not used_ids or len(used_ids) > budget:
        raise AssertionError("bounded generation prompt violates its token budget")
    return used_ids, descriptor, truncation, prompt_ids


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not result:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return result


def _parse_sample_indices(value: str) -> dict[str, tuple[int, ...]]:
    """Parse ``task:index,task:index`` for reproducible GPU smoke runs."""

    grouped: dict[str, list[int]] = {}
    for piece in (item.strip() for item in value.split(",")):
        if not piece or ":" not in piece:
            raise argparse.ArgumentTypeError(
                "sample indices must use task:index,task:index syntax"
            )
        task, raw_index = piece.rsplit(":", 1)
        task = task.strip()
        if task not in TASKS:
            raise argparse.ArgumentTypeError(f"unknown sample-index task: {task}")
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid sample index: {piece}") from exc
        if index < 0:
            raise argparse.ArgumentTypeError("sample indices must be non-negative")
        grouped.setdefault(task, []).append(index)
    if not grouped:
        raise argparse.ArgumentTypeError("sample indices cannot be empty")
    for task, indices in grouped.items():
        if len(indices) != len(set(indices)):
            raise argparse.ArgumentTypeError(f"duplicate sample index for {task}")
    return {task: tuple(indices) for task, indices in grouped.items()}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=_parse_csv, default=TASKS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        help="debug-only prefix count per task; omitted means every frozen sample",
    )
    parser.add_argument(
        "--sample-indices",
        type=_parse_sample_indices,
        help=(
            "debug-only exact rows, e.g. hotpotqa:17,multi_news:83; every "
            "--tasks entry must be represented and this is mutually exclusive with --limit"
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate data and write immutable manifests without loading models or using CUDA",
    )
    return parser.parse_args(argv)


def validate_configuration(raw: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    root = Path(raw.get("project_root", ROOT)).resolve()
    tasks = tuple(args.tasks)
    if len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError(f"unknown or duplicate tasks: {tasks}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.limit is not None and args.sample_indices is not None:
        raise ValueError("--limit and --sample-indices are mutually exclusive")
    if args.sample_indices is not None and set(args.sample_indices) != set(tasks):
        raise ValueError(
            "--sample-indices must contain exactly the tasks selected by --tasks"
        )
    paths = raw.get("paths")
    data = raw.get("data")
    runtime = raw.get("runtime")
    configured_tasks = raw.get("tasks")
    if not all(isinstance(value, Mapping) for value in (paths, data, runtime, configured_tasks)):
        raise ValueError("configuration requires paths, data, runtime, and tasks mappings")
    missing_paths = {
        "source_model", "target_model", "mapper_manifest"
    } - set(paths)
    if missing_paths:
        raise ValueError(f"configuration lacks paths: {sorted(missing_paths)}")
    task_configs: dict[str, dict[str, Any]] = {}
    for task in tasks:
        value = configured_tasks.get(task)
        if not isinstance(value, Mapping):
            raise ValueError(f"configuration lacks task {task}")
        cap = int(value.get("max_gen_tokens", -1))
        if cap != EXPECTED_CAPS[task]:
            raise ValueError(f"{task} max_gen_tokens must be {EXPECTED_CAPS[task]}, got {cap}")
        prompt_mode = str(value.get("prompt_mode", ""))
        if prompt_mode != "qwen_chat_no_think":
            raise ValueError(f"{task} must use qwen_chat_no_think")
        task_configs[task] = {
            "max_gen_tokens": cap,
            "prompt_mode": prompt_mode,
            "checkpoint_primary_metric": PRIMARY_METRICS[task],
            "summary_primary_metric": SUMMARY_PRIMARY_METRICS[task],
        }
    max_length = int(runtime.get("max_length", -1))
    if max_length != 16384:
        raise ValueError("the frozen 8x24-GiB protocol requires max_length=16384")
    reference_cap = int(runtime.get("dialog_reference_score_cap_tokens", -1))
    if reference_cap <= 0 or reference_cap >= max_length:
        raise ValueError("invalid dialogue reference score cap")
    source_devices = tuple(int(value) for value in runtime.get("source_devices", ()))
    target_devices = tuple(int(value) for value in runtime.get("target_devices", ()))
    if not source_devices or not target_devices or set(source_devices) & set(target_devices):
        raise ValueError("source_devices and target_devices must be non-empty and disjoint")
    if len(set(source_devices)) != len(source_devices) or len(set(target_devices)) != len(target_devices):
        raise ValueError("device lists contain duplicates")
    source_layer_counts = tuple(
        int(value) for value in runtime.get("source_layer_counts", ())
    )
    target_layer_counts = tuple(
        int(value) for value in runtime.get("target_layer_counts", ())
    )
    if source_layer_counts != (20, 20):
        raise ValueError("source_layer_counts must be [20, 20]")
    if target_layer_counts != (10, 11, 11, 11, 11, 10):
        raise ValueError(
            "target_layer_counts must be [10, 11, 11, 11, 11, 10]"
        )
    if len(source_layer_counts) != len(source_devices):
        raise ValueError("source layer-count/device lengths differ")
    if len(target_layer_counts) != len(target_devices):
        raise ValueError("target layer-count/device lengths differ")
    lmsys_count = int(data.get("lmsys_sample_count", -1))
    share_count = int(data.get("sharegpt_sample_count", -1))
    if lmsys_count != 200 or share_count != 200:
        raise ValueError("LMSYS and ShareGPT held-out protocols require 200 samples each")
    return {
        "root": root,
        "tasks": tasks,
        "task_configs": task_configs,
        "source_model": _resolve(root, paths["source_model"]),
        "target_model": _resolve(root, paths["target_model"]),
        "mapper_manifest": _resolve(root, paths["mapper_manifest"]),
        "multi_news_source": _resolve(root, data["multi_news_source"]),
        "multi_news_target": _resolve(root, data["multi_news_target"]),
        "hotpotqa_dev_distractor": _resolve(root, data["hotpotqa_dev_distractor"]),
        "lmsys_parquet_dir": _resolve(root, data["lmsys_parquet_dir"]),
        "lmsys_sample_count": lmsys_count,
        "lmsys_selection_seed": str(data.get("lmsys_selection_seed", "")),
        "sharegpt_jsonl": _resolve(root, data["sharegpt_jsonl"]),
        "sharegpt_sample_count": share_count,
        "sharegpt_selection_seed": str(data.get("sharegpt_selection_seed", "")),
        "source_devices": source_devices,
        "target_devices": target_devices,
        "source_layer_counts": source_layer_counts,
        "target_layer_counts": target_layer_counts,
        "attention_implementation": str(
            runtime.get("attention_implementation", "flash_attention_2")
        ),
        "mapper_token_chunk": int(runtime.get("mapper_token_chunk", 1024)),
        "max_length": max_length,
        "dialog_reference_score_cap_tokens": reference_cap,
    }


def model_artifact_identity(model_root: Path) -> dict[str, Any]:
    root = model_root.resolve()
    files: dict[str, Any] = {}
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
    ):
        path = root / name
        if path.is_file():
            files[name] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    for required in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        if required not in files:
            raise FileNotFoundError(f"required model identity file is missing: {root / required}")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"invalid safetensors index: {root}")
    shards = []
    for name in sorted({str(value) for value in weight_map.values()}):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"model shard is missing: {path}")
        shards.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "nbytes": path.stat().st_size,
            }
        )
    return {
        "path": str(root),
        "content_hashed_files": files,
        "weight_shards": shards,
        "weight_shard_identity_scope": "full_sha256_plus_filename_and_nbytes",
    }


def mapper_artifact_identity(manifest_path: Path) -> dict[str, Any]:
    path = manifest_path.resolve()
    checksum = path.parent / "manifest.sha256.json"
    if not path.is_file() or not checksum.is_file():
        raise FileNotFoundError("mapper manifest/checksum commit marker is missing")
    manifest_sha = sha256_file(path)
    declared = json.loads(checksum.read_text(encoding="utf-8"))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if declared.get("sha256") != manifest_sha or manifest.get("status") != "complete":
        raise ValueError("mapper manifest is not a valid complete commit")
    return {
        "manifest": str(path),
        "manifest_sha256": manifest_sha,
        "checksum_sha256": sha256_file(checksum),
        "fit_fingerprint": manifest.get("fit_fingerprint"),
        "layers": len(manifest.get("layers", [])),
    }


def code_provenance() -> dict[str, Any]:
    files = {}
    for relative in FORMAL_CODE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"formal code file is missing: {path}")
        files[relative] = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    return {"files": files, "files_sha256": canonical_digest(files)}


def metric_implementation_provenance() -> dict[str, Any]:
    """Bind third-party metric bytes, not merely dependency names."""

    import rouge
    import rouge.rouge as rouge_api
    import rouge.rouge_score as rouge_algorithm
    import sacrebleu
    import sacrebleu.metrics.chrf as chrf_algorithm

    if getattr(rouge, "__version__", None) != "1.0.1":
        raise RuntimeError("formal LongBench scoring requires rouge==1.0.1")

    def module_identity(module: Any) -> dict[str, Any]:
        path = Path(module.__file__).resolve()
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "nbytes": path.stat().st_size,
        }

    return {
        "rouge": {
            "distribution": "rouge",
            "version": rouge.__version__,
            "api": module_identity(rouge_api),
            "algorithm": module_identity(rouge_algorithm),
            "settings": {
                "class": "rouge.Rouge",
                "metric": "rouge-l",
                "stat": "f",
                "avg": True,
                "exclusive": True,
                "stemming": False,
            },
        },
        "chrf": {
            "distribution": "sacrebleu",
            "version": sacrebleu.__version__,
            "algorithm": module_identity(chrf_algorithm),
            "settings": {"class": "sacrebleu.metrics.CHRF", "score_scale": "0_to_1"},
        },
    }


def runtime_provenance() -> dict[str, Any]:
    from importlib.metadata import PackageNotFoundError, version

    distributions = {}
    for name in (
        "accelerate",
        "flash-attn",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    ):
        try:
            distributions[name] = version(name)
        except PackageNotFoundError:
            distributions[name] = None
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "distributions": distributions,
    }


def build_run_definition(
    *,
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": "qwen3_14b_to_32b_official_and_dialog_target_vs_mapped_v2",
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "modes": list(MODES),
        "tasks": list(configuration["tasks"]),
        "limit": args.limit,
        "sample_indices": (
            {task: list(indices) for task, indices in args.sample_indices.items()}
            if args.sample_indices is not None
            else None
        ),
        "sample_scope": (
            "full"
            if args.limit is None and args.sample_indices is None
            else (
                "debug_explicit_indices_not_formal"
                if args.sample_indices is not None
                else "debug_prefix_not_formal"
            )
        ),
        "task_configs": configuration["task_configs"],
        "generation_cap_disclosure": {
            "configured": {
                task: configuration["task_configs"][task]["max_gen_tokens"]
                for task in configuration["tasks"]
            },
            "rationale": "large deterministic caps requested for this evaluation; cap hits are reported per sample",
        },
        "content_prompt_templates": {
            task: {
                "text": PROMPT_TEMPLATES[task],
                "sha256": bytes_sha256(PROMPT_TEMPLATES[task].encode("utf-8")),
            }
            for task in configuration["tasks"]
            if task not in DIALOG_TASKS
        },
        "truncation": {
            "content_chat_policy": (
                "single user message using the frozen task content template, followed by "
                "the Qwen chat generation prompt with enable_thinking=false; exact-token "
                "head-tail middle removal when over budget"
            ),
            "dialog_policy": (
                "drop_oldest_complete_user_assistant_pairs_then_reapply_chat_template; "
                "if final user alone exceeds budget, head-tail its content tokens and "
                "reapply the chat template"
            ),
            "budget": "max_length-minus-task-generation-cap",
        },
        "decoding": {"do_sample": False, "temperature": 0.0, "batch_size": 1},
        "source_model": model_artifact_identity(configuration["source_model"]),
        "target_model": model_artifact_identity(configuration["target_model"]),
        "mapper": mapper_artifact_identity(configuration["mapper_manifest"]),
        "source_devices": list(configuration["source_devices"]),
        "target_devices": list(configuration["target_devices"]),
        "source_layer_counts": list(configuration["source_layer_counts"]),
        "target_layer_counts": list(configuration["target_layer_counts"]),
        "attention_implementation": configuration["attention_implementation"],
        "mapper_token_chunk": configuration["mapper_token_chunk"],
        "max_length": configuration["max_length"],
        "dialog_reference_score_cap_tokens": configuration[
            "dialog_reference_score_cap_tokens"
        ],
        "metrics": {
            "qa_official": "LongBench/SQuAD normalized full-response max-over-golds token F1 and EM",
            "qa_headline_relaxed": "max(official full-response, frozen gold-independent extracted-answer) F1/EM; no gold-dependent span search",
            "qa_diagnostics": "gold-independent extracted-answer F1/EM plus normalized-gold containment audit; containment is never a headline score",
            "summarization": "rouge==1.0.1 summary-level Rouge-L F1 without stemming, max over references",
            "dialog": "headline reference corpus PPL plus reference Rouge-L/chrF and mapped-vs-target output retention",
        },
        "metric_implementations": metric_implementation_provenance(),
        "runtime": runtime_provenance(),
        "dataset_manifest": dataset_manifest,
        "code": code_provenance(),
    }


def _make_output_directory(root: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        output = supplied.resolve()
        output.mkdir(parents=True, exist_ok=True)
        return output
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = root / "runs" / f"{stamp}_long_context_qwen3_14b_to_32b"
    output.mkdir(parents=True, exist_ok=False)
    return output


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_digest(existing) != canonical_digest(value):
            raise ValueError(f"existing immutable run input disagrees: {path}")
        return
    atomic_write_json(path, value)


def result_path(output: Path, task: str, mode: str, sample_index: int) -> Path:
    return output / "results" / task / mode / f"{sample_index:06d}.json"


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    )


def _numeric_close(observed: Any, expected: Any) -> bool:
    try:
        left = _finite_float(observed, "observed")
        right = _finite_float(expected, "expected")
    except (TypeError, ValueError):
        return False
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _metrics_match(observed: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        return False
    for name, expected_value in expected.items():
        observed_value = observed.get(name)
        if isinstance(expected_value, (int, float)) and not isinstance(
            expected_value, bool
        ):
            if not _numeric_close(observed_value, expected_value):
                return False
        elif observed_value != expected_value:
            return False
    return True


def _expected_prompt_provenance(
    tokenizer: Any,
    sample: EvaluationSample,
    *,
    max_length: int,
) -> tuple[dict[str, Any], list[int]]:
    cap = EXPECTED_CAPS[sample.task]
    used_ids, descriptor, truncation, original_ids = generation_prompt_token_ids(
        tokenizer,
        sample,
        prompt_mode="qwen_chat_no_think",
        budget=max_length - cap,
    )
    return (
        {
            **descriptor,
            "generation_context_truncation": truncation,
            "generation_context_token_ids_sha256": token_ids_sha256(used_ids),
        },
        original_ids,
    )


def _validated_reference_metrics(
    payload: Mapping[str, Any],
    *,
    tokenizer: Any,
    sample: EvaluationSample,
    original_prompt_ids: Sequence[int],
    max_length: int,
    reference_score_cap: int,
) -> dict[str, float] | None:
    reference = payload.get("reference_scoring")
    if sample.task not in DIALOG_TASKS:
        return {} if reference is None else None
    if not isinstance(reference, Mapping):
        return None
    expected_keys = {
        "reference_loglikelihood",
        "reference_nll",
        "reference_ppl",
        "reference_scored_tokens",
        "reference_original_tokens",
        "reference_cap_tokens",
        "reference_cap_hit",
        "reference_context_truncation",
        "reference_context_token_ids_sha256",
    }
    if set(reference) != expected_keys:
        return None
    reference_ids = _encode_text(tokenizer, sample.references[0])
    if not reference_ids:
        return None
    scored_ids = reference_ids[:reference_score_cap]
    context_budget = max_length - len(scored_ids) + 1
    score_context, expected_truncation = _dialog_generation_prompt(
        tokenizer,
        sample,
        budget=context_budget,
        original_prompt_ids=original_prompt_ids,
    )
    scored_tokens = reference.get("reference_scored_tokens")
    original_tokens = reference.get("reference_original_tokens")
    if (
        isinstance(scored_tokens, bool)
        or not isinstance(scored_tokens, int)
        or scored_tokens != len(scored_ids)
        or isinstance(original_tokens, bool)
        or not isinstance(original_tokens, int)
        or original_tokens != len(reference_ids)
        or reference.get("reference_cap_tokens") != reference_score_cap
        or reference.get("reference_cap_hit")
        is not (len(reference_ids) > len(scored_ids))
        or reference.get("reference_context_truncation") != expected_truncation
        or reference.get("reference_context_token_ids_sha256")
        != token_ids_sha256(score_context)
    ):
        return None
    try:
        loglikelihood = _finite_float(
            reference.get("reference_loglikelihood"), "reference_loglikelihood"
        )
        nll = _finite_float(reference.get("reference_nll"), "reference_nll")
        ppl = _finite_float(reference.get("reference_ppl"), "reference_ppl")
    except (TypeError, ValueError):
        return None
    if (
        not _numeric_close(nll, -loglikelihood)
        or not _numeric_close(ppl, math.exp(nll / scored_tokens))
    ):
        return None
    return {"reference_nll": nll, "reference_ppl": ppl}


def _checkpoint_common_valid(
    payload: Mapping[str, Any],
    *,
    fingerprint: str,
    task: str,
    mode: str,
    sample: EvaluationSample,
    expected_prompt: Mapping[str, Any],
) -> bool:
    generation = payload.get("generation")
    return bool(
        payload.get("schema_version") == SCHEMA_VERSION
        and payload.get("status") == "complete"
        and payload.get("request_fingerprint") == fingerprint
        and payload.get("task") == task
        and payload.get("mode") == mode
        and payload.get("sample_index") == sample.index
        and payload.get("sample_id") == sample.sample_id
        and payload.get("record_sha256") == sample.record_sha256
        and payload.get("source_length") == sample.source_length
        and payload.get("primary_metric") == PRIMARY_METRICS[task]
        and isinstance(payload.get("prediction"), str)
        and payload.get("references") == list(sample.references)
        and isinstance(generation, Mapping)
        and generation.get("requested_max_tokens") == EXPECTED_CAPS[task]
        and isinstance(generation.get("generated_tokens"), int)
        and not isinstance(generation.get("generated_tokens"), bool)
        and 0 <= generation.get("generated_tokens") <= EXPECTED_CAPS[task]
        and generation.get("stop_reason") in {"eos", "stop_sequence", "max_tokens"}
        and isinstance(generation.get("cap_hit"), bool)
        and generation.get("cap_hit")
        is (generation.get("stop_reason") == "max_tokens")
        and _is_sha256(generation.get("generated_token_ids_sha256"))
        and payload.get("prompt") == expected_prompt
    )


def checkpoint_complete(
    path: Path,
    *,
    fingerprint: str,
    task: str,
    mode: str,
    sample: EvaluationSample,
    tokenizer: Any,
    max_length: int,
    reference_score_cap: int,
    target_checkpoint: Mapping[str, Any] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        expected_prompt, original_prompt_ids = _expected_prompt_provenance(
            tokenizer, sample, max_length=max_length
        )
    except (TypeError, ValueError, AssertionError):
        return False
    if not _checkpoint_common_valid(
        payload,
        fingerprint=fingerprint,
        task=task,
        mode=mode,
        sample=sample,
        expected_prompt=expected_prompt,
    ):
        return False
    reference_metrics = _validated_reference_metrics(
        payload,
        tokenizer=tokenizer,
        sample=sample,
        original_prompt_ids=original_prompt_ids,
        max_length=max_length,
        reference_score_cap=reference_score_cap,
    )
    if reference_metrics is None:
        return False
    expected_metrics = _sample_metrics(
        task, str(payload["prediction"]), sample.references
    )
    expected_metrics.update(reference_metrics)
    if mode != "mapped":
        return bool(
            payload.get("target_checkpoint_binding") is None
            and _metrics_match(payload.get("metrics"), expected_metrics)
        )
    if target_checkpoint is None or not isinstance(target_checkpoint, Mapping):
        return False
    if not _checkpoint_common_valid(
        target_checkpoint,
        fingerprint=fingerprint,
        task=task,
        mode="target_standalone",
        sample=sample,
        expected_prompt=expected_prompt,
    ):
        return False
    target_reference_metrics = _validated_reference_metrics(
        target_checkpoint,
        tokenizer=tokenizer,
        sample=sample,
        original_prompt_ids=original_prompt_ids,
        max_length=max_length,
        reference_score_cap=reference_score_cap,
    )
    if target_reference_metrics is None:
        return False
    target_expected_metrics = _sample_metrics(
        task, str(target_checkpoint["prediction"]), sample.references
    )
    target_expected_metrics.update(target_reference_metrics)
    if not (
        target_checkpoint.get("target_checkpoint_binding") is None
        and _metrics_match(target_checkpoint.get("metrics"), target_expected_metrics)
    ):
        return False
    if task in DIALOG_TASKS:
        expected_metrics.update(
            {
                "output_retention_rouge_l_f1": rouge_l_f1(
                    str(payload["prediction"]), [str(target_checkpoint["prediction"])]
                ),
                "output_retention_chrf": chrf_score(
                    str(payload["prediction"]), [str(target_checkpoint["prediction"])]
                ),
                "output_exact_match": float(
                    payload["prediction"] == target_checkpoint["prediction"]
                ),
            }
        )
    if not _metrics_match(payload.get("metrics"), expected_metrics):
        return False
    binding = payload.get("target_checkpoint_binding")
    verified = binding.get("verified_fields") if isinstance(binding, Mapping) else None
    expected_top = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task": sample.task,
        "mode": "target_standalone",
        "sample_index": sample.index,
        "sample_id": sample.sample_id,
        "record_sha256": sample.record_sha256,
        "references": list(sample.references),
    }
    return bool(
        isinstance(binding, Mapping)
        and binding.get("target_checkpoint_canonical_sha256")
        == canonical_digest(target_checkpoint)
        and isinstance(verified, Mapping)
        and set(verified)
        == {"sample_and_references", "prompt_and_truncation", "requested_max_tokens"}
        and verified.get("sample_and_references") == canonical_digest(expected_top)
        and verified.get("prompt_and_truncation") == canonical_digest(expected_prompt)
        and verified.get("requested_max_tokens") == EXPECTED_CAPS[task]
    )


def _tokenizer_identity(source: Any, target: Any) -> dict[str, Any]:
    source_vocab = source.get_vocab()
    target_vocab = target.get_vocab()
    if source_vocab != target_vocab:
        raise ValueError("source and target tokenizer vocabularies differ")
    fields = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    source_special = {name: getattr(source, name, None) for name in fields}
    target_special = {name: getattr(target, name, None) for name in fields}
    if source_special != target_special:
        raise ValueError("source and target tokenizer special-token IDs differ")
    source_template = getattr(source, "chat_template", None)
    target_template = getattr(target, "chat_template", None)
    if source_template != target_template:
        raise ValueError("source and target tokenizer chat templates differ")
    return {
        "vocabulary_size": len(source_vocab),
        "vocabulary_sha256": canonical_digest(sorted(source_vocab.items())),
        "special_token_ids": source_special,
        "chat_template_sha256": (
            bytes_sha256(source_template.encode("utf-8"))
            if isinstance(source_template, str)
            else None
        ),
    }


def _sample_metrics(task: str, prediction: str, references: Sequence[str]) -> dict[str, Any]:
    if task == "hotpotqa":
        return qa_metrics(prediction, references)
    if task == "multi_news":
        return {"rouge_l_f1": rouge_l_f1(prediction, references)}
    if task in DIALOG_TASKS:
        return {
            "reference_rouge_l_f1": rouge_l_f1(prediction, references),
            "reference_chrf": chrf_score(prediction, references),
            "reference_normalized_token_f1": max(
                _token_f1_normalized(prediction, reference) for reference in references
            ),
        }
    raise ValueError(f"unsupported task: {task}")


def _score_dialog_reference(
    engine: TransferEvaluationEngine,
    tokenizer: Any,
    sample: EvaluationSample,
    original_prompt_ids: Sequence[int],
    reference: str,
    *,
    max_reference_tokens: int,
) -> dict[str, Any]:
    reference_ids = _encode_text(tokenizer, reference)
    original_reference_tokens = len(reference_ids)
    if not reference_ids:
        raise ValueError("dialog reference tokenized to zero tokens")
    reference_ids = reference_ids[:max_reference_tokens]
    context_budget = engine.max_length - len(reference_ids) + 1
    score_context, truncation = _dialog_generation_prompt(
        tokenizer,
        sample,
        budget=context_budget,
        original_prompt_ids=original_prompt_ids,
    )
    score = engine.score_token_ids(score_context, reference_ids)
    nll = -score.loglikelihood
    return {
        "reference_loglikelihood": score.loglikelihood,
        "reference_nll": nll,
        "reference_ppl": math.exp(nll / score.token_count),
        "reference_scored_tokens": score.token_count,
        "reference_original_tokens": original_reference_tokens,
        "reference_cap_tokens": max_reference_tokens,
        "reference_cap_hit": original_reference_tokens > len(reference_ids),
        "reference_context_truncation": truncation,
        "reference_context_token_ids_sha256": token_ids_sha256(score_context),
    }


def _validate_and_bind_target_checkpoint(
    target_checkpoint: Mapping[str, Any] | None,
    *,
    sample: EvaluationSample,
    cap: int,
    prompt_descriptor: Mapping[str, Any],
    truncation: Mapping[str, Any],
    used_prompt_sha256: str,
) -> dict[str, Any]:
    """Require mapped evaluation to consume the exact target request identity."""

    if target_checkpoint is None:
        raise ValueError("mapped evaluation requires its preceding target checkpoint")
    expected_top = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task": sample.task,
        "mode": "target_standalone",
        "sample_index": sample.index,
        "sample_id": sample.sample_id,
        "record_sha256": sample.record_sha256,
        "references": list(sample.references),
    }
    observed_top = {name: target_checkpoint.get(name) for name in expected_top}
    if observed_top != expected_top:
        raise ValueError("mapped/target sample or reference identity differs")
    target_generation = target_checkpoint.get("generation")
    target_prompt = target_checkpoint.get("prompt")
    if not isinstance(target_generation, Mapping) or not isinstance(target_prompt, Mapping):
        raise ValueError("target checkpoint lacks generation/prompt provenance")
    expected_prompt = {
        **dict(prompt_descriptor),
        "generation_context_truncation": dict(truncation),
        "generation_context_token_ids_sha256": used_prompt_sha256,
    }
    if dict(target_prompt) != expected_prompt:
        raise ValueError("mapped/target prompt token IDs or truncation differ")
    if target_generation.get("requested_max_tokens") != cap:
        raise ValueError("mapped/target generation caps differ")
    return {
        "target_checkpoint_canonical_sha256": canonical_digest(target_checkpoint),
        "verified_fields": {
            "sample_and_references": canonical_digest(expected_top),
            "prompt_and_truncation": canonical_digest(expected_prompt),
            "requested_max_tokens": cap,
        },
    }


def evaluate_one(
    *,
    engine: TransferEvaluationEngine,
    tokenizer: Any,
    sample: EvaluationSample,
    mode: str,
    task_config: Mapping[str, Any],
    reference_score_cap: int,
    target_checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    cap = int(task_config["max_gen_tokens"])
    generation_budget = engine.max_length - cap
    (
        used_prompt_ids,
        prompt_descriptor,
        truncation,
        original_prompt_ids,
    ) = generation_prompt_token_ids(
        tokenizer,
        sample,
        prompt_mode=str(task_config["prompt_mode"]),
        budget=generation_budget,
    )
    used_prompt_sha256 = token_ids_sha256(used_prompt_ids)
    target_binding = None
    if mode == "mapped":
        target_binding = _validate_and_bind_target_checkpoint(
            target_checkpoint,
            sample=sample,
            cap=cap,
            prompt_descriptor=prompt_descriptor,
            truncation=truncation,
            used_prompt_sha256=used_prompt_sha256,
        )
    generation = engine.generate_token_ids(
        used_prompt_ids,
        {
            "max_gen_toks": cap,
            "do_sample": False,
            "temperature": 0.0,
            "until": [],
        },
        truncation="error",
    )
    metrics = _sample_metrics(sample.task, generation.text, sample.references)
    reference_scoring = None
    if sample.task in DIALOG_TASKS:
        reference_scoring = _score_dialog_reference(
            engine,
            tokenizer,
            sample,
            original_prompt_ids,
            sample.references[0],
            max_reference_tokens=reference_score_cap,
        )
        metrics.update(
            {
                "reference_nll": reference_scoring["reference_nll"],
                "reference_ppl": reference_scoring["reference_ppl"],
            }
        )
        if mode == "mapped":
            if target_checkpoint is None:  # guarded for every mapped task above
                raise AssertionError("mapped dialogue target binding disappeared")
            target_prediction = target_checkpoint.get("prediction")
            if not isinstance(target_prediction, str):
                raise ValueError("target checkpoint lacks prediction")
            metrics.update(
                {
                    "output_retention_rouge_l_f1": rouge_l_f1(
                        generation.text, [target_prediction]
                    ),
                    "output_retention_chrf": chrf_score(
                        generation.text, [target_prediction]
                    ),
                    "output_exact_match": float(generation.text == target_prediction),
                }
            )
    return {
        "prediction": generation.text,
        "references": list(sample.references),
        "metrics": metrics,
        "generation": {
            "requested_max_tokens": cap,
            "generated_tokens": generation.generated_token_count,
            "stop_reason": generation.stop_reason,
            "cap_hit": generation.cap_hit,
            "generated_token_ids_sha256": token_ids_sha256(
                generation.generated_token_ids
            ),
        },
        "prompt": {
            **prompt_descriptor,
            "generation_context_truncation": truncation,
            "generation_context_token_ids_sha256": used_prompt_sha256,
        },
        "reference_scoring": reference_scoring,
        "target_checkpoint_binding": target_binding,
    }


def summarize(
    output: Path,
    *,
    configuration: Mapping[str, Any],
    samples: Mapping[str, Sequence[EvaluationSample]],
    fingerprint: str,
    tokenizer: Any,
) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    result_files: list[dict[str, Any]] = []
    for task in configuration["tasks"]:
        task_summary: dict[str, Any] = {"expected_samples": len(samples[task]), "modes": {}}
        for mode in MODES:
            metric_values: dict[str, list[float]] = {}
            cap_hits = 0
            completed = 0
            total_ref_ll = 0.0
            total_ref_tokens = 0
            for sample in samples[task]:
                path = result_path(output, task, mode, sample.index)
                target_checkpoint = None
                if mode == "mapped":
                    target_path = result_path(
                        output, task, "target_standalone", sample.index
                    )
                    if target_path.is_file():
                        try:
                            candidate = json.loads(target_path.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError):
                            candidate = None
                        if isinstance(candidate, Mapping) and checkpoint_complete(
                            target_path,
                            fingerprint=fingerprint,
                            task=task,
                            mode="target_standalone",
                            sample=sample,
                            tokenizer=tokenizer,
                            max_length=configuration["max_length"],
                            reference_score_cap=configuration[
                                "dialog_reference_score_cap_tokens"
                            ],
                        ):
                            target_checkpoint = candidate
                if not checkpoint_complete(
                    path,
                    fingerprint=fingerprint,
                    task=task,
                    mode=mode,
                    sample=sample,
                    tokenizer=tokenizer,
                    max_length=configuration["max_length"],
                    reference_score_cap=configuration[
                        "dialog_reference_score_cap_tokens"
                    ],
                    target_checkpoint=target_checkpoint,
                ):
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                result_files.append(
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": sha256_file(path),
                        "nbytes": path.stat().st_size,
                    }
                )
                completed += 1
                cap_hits += int(bool(payload["generation"]["cap_hit"]))
                for name, value in payload["metrics"].items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric = _finite_float(value, f"{task}/{mode}/{name}")
                        metric_values.setdefault(name, []).append(numeric)
                reference = payload.get("reference_scoring")
                if isinstance(reference, Mapping):
                    total_ref_ll += _finite_float(
                        reference["reference_loglikelihood"], "reference_loglikelihood"
                    )
                    total_ref_tokens += int(reference["reference_scored_tokens"])
            means = {
                name: sum(values) / len(values) for name, values in sorted(metric_values.items())
            }
            if total_ref_tokens:
                means["reference_corpus_ppl"] = math.exp(-total_ref_ll / total_ref_tokens)
            summary_primary = SUMMARY_PRIMARY_METRICS[task]
            mode_complete = bool(
                completed == len(samples[task])
                and summary_primary in means
                and math.isfinite(float(means[summary_primary]))
            )
            task_summary["modes"][mode] = {
                "completed_samples": completed,
                "complete": mode_complete,
                "mean_metrics": means,
                "generation_cap_hits": cap_hits,
                "generation_cap_hit_rate": cap_hits / completed if completed else None,
                "reference_scored_tokens": total_ref_tokens,
            }
        target_metrics = task_summary["modes"]["target_standalone"]["mean_metrics"]
        mapped_metrics = task_summary["modes"]["mapped"]["mean_metrics"]
        metric_comparisons = {}
        for metric in sorted(set(target_metrics) & set(mapped_metrics)):
            target_value = target_metrics[metric]
            mapped_value = mapped_metrics[metric]
            metric_comparisons[metric] = {
                "higher_is_better": metric
                not in {"reference_nll", "reference_ppl", "reference_corpus_ppl"},
                "target_standalone": target_value,
                "mapped": mapped_value,
                "mapped_minus_target": mapped_value - target_value,
                "mapped_over_target_percent": (
                    mapped_value / target_value * 100.0 if target_value != 0 else None
                ),
            }
        task_summary["metric_comparisons"] = metric_comparisons
        primary = SUMMARY_PRIMARY_METRICS[task]
        comparison: dict[str, Any] = {"primary_metric": primary}
        if primary in metric_comparisons:
            comparison.update(metric_comparisons[primary])
        task_summary["comparison"] = comparison
        tasks[task] = task_summary
    complete = all(
        tasks[task]["modes"][mode]["complete"]
        for task in configuration["tasks"]
        for mode in MODES
    )
    results_manifest_identity = None
    if complete:
        expected_result_count = sum(len(samples[task]) for task in configuration["tasks"]) * len(MODES)
        if len(result_files) != expected_result_count:
            raise AssertionError("complete summary has an incorrect result inventory size")
        ordered_files = sorted(result_files, key=lambda item: item["path"])
        if len({item["path"] for item in ordered_files}) != expected_result_count:
            raise AssertionError("complete summary has duplicate result paths")
        manifest_path = output / "results.manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "request_fingerprint": fingerprint,
                "expected_result_count": expected_result_count,
                "files": ordered_files,
                "files_sha256": canonical_digest(ordered_files),
            },
        )
        results_manifest_identity = {
            "path": manifest_path.name,
            "sha256": sha256_file(manifest_path),
            "nbytes": manifest_path.stat().st_size,
        }
    elif (output / "results.manifest.json").exists():
        raise ValueError("an incomplete run cannot retain a complete results manifest")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if complete else "incomplete",
        "request_fingerprint": fingerprint,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "results_manifest": results_manifest_identity,
        "tasks": tasks,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def _apply_limit(
    samples: Mapping[str, Sequence[EvaluationSample]], limit: int | None
) -> dict[str, list[EvaluationSample]]:
    if limit is None:
        return {task: list(values) for task, values in samples.items()}
    return {task: list(values[:limit]) for task, values in samples.items()}


def _apply_sample_scope(
    samples: Mapping[str, Sequence[EvaluationSample]],
    *,
    limit: int | None,
    sample_indices: Mapping[str, Sequence[int]] | None,
) -> dict[str, list[EvaluationSample]]:
    if sample_indices is None:
        return _apply_limit(samples, limit)
    selected: dict[str, list[EvaluationSample]] = {}
    for task, rows in samples.items():
        by_index = {sample.index: sample for sample in rows}
        requested = sample_indices[task]
        missing = [index for index in requested if index not in by_index]
        if missing:
            raise ValueError(f"sample indices out of range for {task}: {missing}")
        selected[task] = [by_index[index] for index in requested]
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw = load_yaml(args.config)
    configuration = validate_configuration(raw, args)
    samples, dataset_manifest = load_all_samples(configuration, configuration["tasks"])
    samples = _apply_sample_scope(
        samples, limit=args.limit, sample_indices=args.sample_indices
    )
    output = _make_output_directory(configuration["root"], args.output_dir)
    write_immutable(output / "dataset_manifest.json", dataset_manifest)
    run_definition = build_run_definition(
        args=args, configuration=configuration, dataset_manifest=dataset_manifest
    )
    fingerprint = canonical_digest(run_definition)
    write_immutable(
        output / "run_definition.json",
        {**run_definition, "request_fingerprint": fingerprint},
    )
    if args.prepare_only:
        atomic_write_json(
            output / "status.json",
            {
                "status": "prepared",
                "request_fingerprint": fingerprint,
                "gpu_used": False,
                "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        print(f"prepared: {output}")
        return 0

    required_devices = set(configuration["source_devices"]) | set(
        configuration["target_devices"]
    )
    if not torch.cuda.is_available() or max(required_devices) >= torch.cuda.device_count():
        raise RuntimeError(
            f"requested CUDA devices {sorted(required_devices)}, available={torch.cuda.device_count()}"
        )
    run_started = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_write_json(
        output / "status.json",
        {
            "status": "loading_models",
            "started_utc": run_started,
            "request_fingerprint": fingerprint,
        },
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        configuration["target_model"], local_files_only=True, use_fast=True
    )
    source_tokenizer = AutoTokenizer.from_pretrained(
        configuration["source_model"], local_files_only=True, use_fast=True
    )
    atomic_write_json(
        output / "tokenizer.json", _tokenizer_identity(source_tokenizer, target_tokenizer)
    )
    target = load_qwen3(
        configuration["target_model"],
        devices=configuration["target_devices"],
        layer_counts=configuration["target_layer_counts"],
        role="target",
        attention_implementation=configuration["attention_implementation"],
    )
    source = load_qwen3(
        configuration["source_model"],
        devices=configuration["source_devices"],
        layer_counts=configuration["source_layer_counts"],
        role="source",
        attention_implementation=configuration["attention_implementation"],
    )
    atomic_write_json(
        output / "model_sharding.json",
        {
            "target": getattr(
                target, "_kvcache_transfer_layer_sharding_evidence"
            ),
            "source": getattr(
                source, "_kvcache_transfer_layer_sharding_evidence"
            ),
        },
    )
    mapped_builder = MappedPrefixCacheBuilder(
        source_model=source,
        target_model=target,
        mapper_manifest=configuration["mapper_manifest"],
        token_chunk=configuration["mapper_token_chunk"],
    )
    atomic_write_json(output / "mapper.json", mapped_builder.provenance)
    engines = {
        "target_standalone": TransferEvaluationEngine(
            target_model=target,
            tokenizer=target_tokenizer,
            mode="target_standalone",
            max_length=configuration["max_length"],
            max_gen_tokens=max(EXPECTED_CAPS.values()),
        ),
        "mapped": TransferEvaluationEngine(
            target_model=target,
            tokenizer=target_tokenizer,
            mode="mapped",
            mapped_prefix_builder=mapped_builder,
            max_length=configuration["max_length"],
            max_gen_tokens=max(EXPECTED_CAPS.values()),
        ),
    }
    atomic_write_json(
        output / "status.json",
        {
            "status": "running",
            "started_utc": run_started,
            "request_fingerprint": fingerprint,
        },
    )
    failures = 0
    for task in configuration["tasks"]:
        for sample in samples[task]:
            target_payload: Mapping[str, Any] | None = None
            for mode in MODES:
                destination = result_path(output, task, mode, sample.index)
                if checkpoint_complete(
                    destination,
                    fingerprint=fingerprint,
                    task=task,
                    mode=mode,
                    sample=sample,
                    tokenizer=target_tokenizer,
                    max_length=configuration["max_length"],
                    reference_score_cap=configuration[
                        "dialog_reference_score_cap_tokens"
                    ],
                    target_checkpoint=target_payload,
                ):
                    print(f"[resume] {task}/{sample.index:06d}/{mode}")
                    if mode == "target_standalone":
                        target_payload = json.loads(destination.read_text(encoding="utf-8"))
                    continue
                started = dt.datetime.now(dt.timezone.utc).isoformat()
                try:
                    value = evaluate_one(
                        engine=engines[mode],
                        tokenizer=target_tokenizer,
                        sample=sample,
                        mode=mode,
                        task_config=configuration["task_configs"][task],
                        reference_score_cap=configuration[
                            "dialog_reference_score_cap_tokens"
                        ],
                        target_checkpoint=target_payload,
                    )
                    payload = {
                        "schema_version": SCHEMA_VERSION,
                        "status": "complete",
                        "request_fingerprint": fingerprint,
                        "task": task,
                        "mode": mode,
                        "sample_index": sample.index,
                        "sample_id": sample.sample_id,
                        "record_sha256": sample.record_sha256,
                        "source_length": sample.source_length,
                        "primary_metric": PRIMARY_METRICS[task],
                        "started_utc": started,
                        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                        **value,
                    }
                    atomic_write_json(destination, payload)
                    destination.with_suffix(".failure.json").unlink(missing_ok=True)
                    if mode == "target_standalone":
                        target_payload = payload
                    print(
                        f"[complete] {task}/{sample.index:06d}/{mode}: "
                        f"{payload['metrics'][PRIMARY_METRICS[task]]:.6f}"
                    )
                except BaseException as exc:
                    failures += 1
                    atomic_write_json(
                        destination.with_suffix(".failure.json"),
                        {
                            "schema_version": SCHEMA_VERSION,
                            "status": "failed",
                            "request_fingerprint": fingerprint,
                            "task": task,
                            "mode": mode,
                            "sample_index": sample.index,
                            "sample_id": sample.sample_id,
                            "record_sha256": sample.record_sha256,
                            "started_utc": started,
                            "failed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(
                        f"[failed] {task}/{sample.index:06d}/{mode}: {exc}", file=sys.stderr
                    )
                finally:
                    engines[mode].clear_prefix_cache()
                if failures:
                    break
            if failures:
                break
        summarize(
            output,
            configuration=configuration,
            samples=samples,
            fingerprint=fingerprint,
            tokenizer=target_tokenizer,
        )
        if failures:
            break
    summary = summarize(
        output,
        configuration=configuration,
        samples=samples,
        fingerprint=fingerprint,
        tokenizer=target_tokenizer,
    )
    status_value = "failed" if failures else summary["status"]
    if not failures and status_value != "complete":
        status_value = "incomplete"
    summary_path = output / "summary.json"
    manifest_identity = summary.get("results_manifest")
    atomic_write_json(
        output / "status.json",
        {
            "status": status_value,
            "failures": failures,
            "started_utc": run_started,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "request_fingerprint": fingerprint,
            "summary_sha256": sha256_file(summary_path),
            "summary_nbytes": summary_path.stat().st_size,
            "results_manifest": manifest_identity,
        },
    )
    print(f"results: {output}")
    return 0 if not failures and summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
