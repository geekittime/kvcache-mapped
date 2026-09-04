"""Deterministic, auditable CoQA multi-turn handoff evaluation.

The paper discloses the high-level CoQA protocol but not the 100 conversation
IDs, prompt, history construction, or decoding settings.  This implementation
therefore labels its deterministic completion explicitly: twenty conversations
from each of the five validation domains are selected by a gold-independent
SHA rank, prior answers are the primary gold answers, and identical cumulative
token histories are advanced in persistent 14B/32B parent caches.  The five
official-figure depths each map the then-current 32B parent exactly once.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import datetime as dt
import hashlib
import json
import math
import re
import string
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from .eval_common import (
    EVALUATION_SCHEMA_VERSION,
    assert_evaluation_output,
    canonical_jsonl_bytes,
    code_identity,
    hashed_json,
    hashed_jsonl,
    immutable_hashed_json,
    mapper_manifest_identity,
    model_artifact_identity,
    token_ids_sha256,
    verify_hashed_json,
)
from .protocol import (
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    canonical_sha256,
    read_json,
    sha256_file,
)


DEFAULT_CONFIG = EXPERIMENT_ROOT / "config/coqa.yaml"
HEADLINE_DEPTHS = (1, 3, 5, 7, 10)
PAPER_DOMAINS = (
    "children_stories",
    "literature",
    "mid-high_school",
    "news",
    "wikipedia",
)
OFFICIAL_DOMAIN_MAPPING = {
    "mctest": "children_stories",
    "gutenberg": "literature",
    "race": "mid-high_school",
    "cnn": "news",
    "wikipedia": "wikipedia",
    # Accept already-normalized mirrors without changing their meaning.
    **{value: value for value in PAPER_DOMAINS},
}
MODES = ("target_standalone", "mapped")
EXPECTED_CONVERSATIONS = 100
EXPECTED_RECORDS_PER_MODE = EXPECTED_CONVERSATIONS * len(HEADLINE_DEPTHS)

CODE_FILES = (
    "reverse_kv/__init__.py",
    "reverse_kv/coqa_eval.py",
    "reverse_kv/eval_common.py",
    "reverse_kv/provenance.py",
    "reverse_kv/runtime.py",
    "reverse_kv/protocol.py",
    "scripts/evaluate_coqa.py",
    "config/coqa.yaml",
    "artifacts/inputs/code_freeze.json",
)
SHARED_CODE_FILES = (
    "src/kvcache_transfer/__init__.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/transfer.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/io.py",
)


@dataclass(frozen=True)
class CoQAConversation:
    row_index: int
    conversation_id: str
    domain: str
    story: str
    questions: tuple[str, ...]
    primary_answers: tuple[str, ...]
    references: tuple[tuple[str, ...], ...]
    raw_domain: str | None = None

    @property
    def turn_count(self) -> int:
        return len(self.questions)


@dataclass(frozen=True)
class SelectedConversation:
    selection_index: int
    rank_within_domain: int
    rank_sha256: str
    conversation: CoQAConversation


@dataclass(frozen=True)
class CoQATurnTokenPlan:
    """One cache-stable turn in the cumulative gold-history transcript.

    ``parent_token_ids`` is the exact cache state immediately before replaying
    ``boundary_token_id``.  ``append_after_turn`` advances that immutable
    logical parent to the next turn's parent without replaying the full story.
    """

    turn: int
    prompt: str
    prompt_token_ids: tuple[int, ...]
    parent_token_ids: tuple[int, ...]
    boundary_token_id: int
    append_after_turn: tuple[int, ...]


@dataclass(frozen=True)
class PersistentGenerationResult:
    text: str
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    cap_hit: bool
    requested_max_tokens: int
    context_tokens: int

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_token_ids)


@dataclass(frozen=True)
class CoQARunVariant:
    """A sealed mapper-family variation of the otherwise identical protocol.

    ``None`` curve/prerequisite values define the production k=20 path.  The
    only accepted non-production variant is Figure 4's explicitly labelled
    retuned k=4 curve, which must supply its dedicated runtime builder and a
    completed held-out mapper-acceptance receipt.
    """

    kind: str
    selected_source_layers: int
    paper_disclosed: str
    figure_evidence: str
    code_files: tuple[str, ...]
    curve_variant: str | None = None
    paper_undisclosed: str | None = None
    prerequisite_acceptance: Mapping[str, Any] | None = None


DEFAULT_RUN_VARIANT = CoQARunVariant(
    kind="qwen3_32b_to_14b_coqa_multi_turn_handoff",
    selected_source_layers=20,
    paper_disclosed=(
        "100 conversations, five domains, word F1 at handoff depths "
        "1,3,5,7,10, k=20"
    ),
    figure_evidence=(
        "official arXiv source figures/retention_vs_turns.png; "
        "x-axis Turn depth at handoff"
    ),
    code_files=CODE_FILES,
)


def _variant_run_fields(variant: CoQARunVariant) -> dict[str, Any]:
    """Return output fields while preserving the legacy k20 schema exactly."""

    paper_reference = {
        "paper": "arXiv:2608.03893v1",
        "section": "4.6",
        "paper_disclosed": variant.paper_disclosed,
        "figure_evidence": variant.figure_evidence,
        "local_completion_due_to_undisclosed_details": True,
        "paper_bit_exact_claimed": False,
    }
    result: dict[str, Any] = {
        "kind": variant.kind,
        "paper_reference": paper_reference,
    }
    if variant.curve_variant is not None:
        paper_reference["paper_undisclosed"] = variant.paper_undisclosed
        result.update(
            {
                "curve_variant": variant.curve_variant,
                "selected_source_layers": variant.selected_source_layers,
                "prerequisite_acceptance": dict(
                    variant.prerequisite_acceptance or {}
                ),
            }
        )
    return result


def _validate_run_variant(
    variant: CoQARunVariant,
    mapper_builder_class: Any | None,
) -> None:
    if variant.curve_variant is None:
        if variant != DEFAULT_RUN_VARIANT:
            raise ValueError("production CoQA variant fields changed")
        if mapper_builder_class is not None:
            raise ValueError("production CoQA cannot inject a non-production mapper builder")
        return
    expected_curve = "figure4_large_to_small_retuned_k4"
    if (
        variant.curve_variant != expected_curve
        or variant.selected_source_layers != 4
        or variant.kind != "qwen3_32b_to_14b_coqa_multi_turn_handoff_k4_retuned"
        or not variant.paper_undisclosed
    ):
        raise ValueError("unsupported CoQA mapper curve variant")
    acceptance = variant.prerequisite_acceptance
    if not isinstance(acceptance, Mapping) or acceptance.get("status") != "passed":
        raise ValueError("k4 CoQA requires a passed held-out mapper acceptance gate")
    if acceptance.get("kind") != "mandatory_pre_CoQA_k4_mapper_acceptance_gate":
        raise ValueError("k4 CoQA prerequisite has the wrong scientific identity")
    if mapper_builder_class is None or (
        getattr(mapper_builder_class, "__module__", "")
        != "reverse_kv.k4_runtime"
        or getattr(mapper_builder_class, "__name__", "")
        != "K4ReverseMappedPrefixCacheBuilder"
    ):
        raise ValueError("k4 CoQA requires the dedicated fail-closed k4 builder")


def _validate_mapper_for_variant(
    mapper_manifest: Mapping[str, Any],
    mapper_identity: Mapping[str, Any],
    variant: CoQARunVariant,
) -> None:
    architecture = mapper_manifest.get("architecture")
    if not isinstance(architecture, Mapping) or int(
        architecture.get("selected_source_layers", -1)
    ) != variant.selected_source_layers:
        raise ValueError("CoQA mapper top-k does not match the sealed run variant")
    if variant.curve_variant is None:
        if mapper_manifest.get("curve_variant") is not None:
            raise ValueError("production k20 CoQA received a curve-variant mapper")
        return
    if mapper_manifest.get("curve_variant") != variant.curve_variant:
        raise ValueError("retuned CoQA curve and mapper variant disagree")
    acceptance = variant.prerequisite_acceptance or {}
    if (
        acceptance.get("mapper_manifest_sha256")
        != mapper_identity.get("manifest_sha256")
        or acceptance.get("fit_fingerprint")
        != mapper_identity.get("fit_fingerprint")
    ):
        raise ValueError("k4 held-out gate is bound to another mapper")


def _resolve_config_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_coqa_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("CoQA config must be a mapping")
    if raw.get("schema_version") != 1 or raw.get("direction") != "qwen3_32b_to_14b":
        raise ValueError("unsupported CoQA reverse protocol")
    dataset = raw.get("dataset")
    paper_reference = raw.get("paper_reference")
    selection = raw.get("selection")
    prompt = raw.get("prompt")
    generation = raw.get("generation")
    metric = raw.get("metric")
    paths = raw.get("paths")
    runtime = raw.get("runtime")
    if not all(isinstance(value, Mapping) for value in (
        dataset, paper_reference, selection, prompt, generation, metric, paths, runtime
    )):
        raise ValueError("CoQA config lacks a required mapping")
    if dict(paper_reference) != {
        "paper": "arXiv:2608.03893v1",
        "section": "4.6",
        "disclosed": (
            "100 conversations, five domains, word F1 at handoff depths "
            "1,3,5,7,10, k=20"
        ),
        "figure_evidence": (
            "official arXiv source figures/retention_vs_turns.png; "
            "x-axis Turn depth at handoff"
        ),
        "undisclosed": (
            "conversation IDs, selection seed, prompt, history policy, decoding settings"
        ),
    }:
        raise ValueError("CoQA paper disclosure boundary changed")
    expected_dataset = {
        "provider": "ModelScope",
        "repository": "EleutherAI/coqa",
        "revision": "5153eac35fe087e44d9ce99ebb2d94c3dbc81583",
        "split": "validation",
        "expected_sha256": "5ab4d352067c30ef4925efb0dfb2abeeed4cdfb236304aa7ee9df62ea781281f",
        "expected_nbytes": 2_221_232,
        "file": (
            "/home/wby/llm-cache/nvidia-kvcache/data/coqa/coqa_validation.parquet"
        ),
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            raise ValueError(f"CoQA dataset lock {key} is not frozen")
    if tuple(selection.get("domains", ())) != PAPER_DOMAINS:
        raise ValueError("CoQA domains must be the five frozen validation domains")
    if int(selection.get("conversations_per_domain", -1)) != 20:
        raise ValueError("CoQA selection must use twenty conversations per domain")
    if int(selection.get("minimum_turns", -1)) != 10:
        raise ValueError("CoQA conversations must have at least ten turns")
    if tuple(
        int(value) for value in selection.get("headline_handoff_depths", ())
    ) != HEADLINE_DEPTHS:
        raise ValueError("CoQA headline handoff depths must be 1,3,5,7,10")
    if selection.get("gold_independent") is not True:
        raise ValueError("CoQA conversation selection must be gold-independent")
    if selection.get("namespace") != "qwen3_32b_to_14b_coqa_v1" or selection.get(
        "ranking"
    ) != "ascending sha256(namespace + NUL + conversation_id)":
        raise ValueError("CoQA deterministic selection rule changed")
    prompt_expected = {
        "template_version": "eleutherai_coqa_completion_gold_history_v1",
        "completion_mode": True,
        "apply_chat_template": False,
        "thinking_disabled_by_no_chat_template": True,
        "prior_answer_history": "gold_primary_answer",
        "answer_instruction": "none",
        "cache_history_execution": (
            "persistent_incremental_token_deltas_full_history_prefill_once"
        ),
        "incremental_tokenization": (
            "encode_segments_and_require_exact_full_prompt_token_equality"
        ),
    }
    for key, expected in prompt_expected.items():
        if prompt.get(key) != expected:
            raise ValueError(f"CoQA prompt invariant {key} changed")
    generation_expected = {
        "do_sample": False,
        "temperature": 0,
        "max_new_tokens": 128,
        "stop_sequences": ["\nQ:"],
        "context_overflow": "error",
    }
    for key, expected in generation_expected.items():
        if generation.get(key) != expected:
            raise ValueError(f"CoQA generation invariant {key} changed")
    if metric.get("primary") != "official_normalized_word_f1":
        raise ValueError("CoQA primary metric must be official normalized word F1")
    official_metric_expected = {
        "primary_evaluator": "stanford_coqa_evaluate_v1.0",
        "official_evaluator_url": (
            "https://nlp.stanford.edu/data/coqa/evaluate-v1.0.py"
        ),
        "official_evaluator_sha256": (
            "d42f70d686abc724a9169947a487f4509087ee69804277cc73950682cb700b4d"
        ),
        "reference_set": "primary_plus_three_additional_human_annotations",
        "reference_deduplication": (
            "none_preserve_official_annotator_multiplicity"
        ),
        "response_postprocess": "lm_eval_first_line_only",
        "lm_eval_reference_difference": (
            "vendored_doc_to_target_case_insensitive_dedup_is_not_used"
        ),
    }
    for key, expected in official_metric_expected.items():
        if metric.get(key) != expected:
            raise ValueError(f"CoQA official metric invariant {key} changed")
    if metric.get("stored_primary_scale") != (
        "fraction_0_to_1_with_percent_companion"
    ):
        raise ValueError("CoQA stored metric scale changed")
    if metric.get("aggregation") != "macro_over_conversation_handoff_depth_records":
        raise ValueError("CoQA aggregation must be macro over headline handoff records")
    if metric.get("multiple_references") != (
        "leave_one_out_average_of_max_over_remaining_references"
    ):
        raise ValueError("CoQA multiple-reference reduction is not the official protocol")
    if metric.get("diagnostic") != (
        "format_relaxed_official_normalized_word_f1"
    ):
        raise ValueError("CoQA diagnostic metric changed")
    if tuple(runtime.get("source_devices", ())) != (0, 1, 2, 3, 4):
        raise ValueError("CoQA source topology changed")
    if tuple(runtime.get("source_layer_counts", ())) != (12, 13, 14, 13, 12):
        raise ValueError("CoQA source layer topology changed")
    if tuple(runtime.get("target_devices", ())) != (5, 6, 7):
        raise ValueError("CoQA target topology changed")
    if tuple(runtime.get("target_layer_counts", ())) != (13, 14, 13):
        raise ValueError("CoQA target layer topology changed")
    runtime_fixed = {
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "max_context_tokens": 32768,
    }
    for key, expected in runtime_fixed.items():
        if runtime.get(key) != expected:
            raise ValueError(f"CoQA runtime invariant {key} changed")
    result = dict(raw)
    result["config_path"] = config_path
    result["dataset"] = dict(dataset)
    result["dataset"]["file"] = _resolve_config_path(
        config_path.parent, str(dataset["file"])
    )
    result["paths"] = dict(paths)
    for key in ("source_model", "target_model", "mapper_manifest"):
        result["paths"][key] = _resolve_config_path(config_path.parent, str(paths[key]))
    result["selection"] = dict(selection)
    result["prompt"] = dict(prompt)
    result["generation"] = dict(generation)
    result["metric"] = dict(metric)
    result["runtime"] = dict(runtime)
    return result


def dataset_download_command(configuration: Mapping[str, Any]) -> list[str]:
    dataset = configuration["dataset"]
    return [
        "/home/wby/.venvs/modelscope/bin/ms-hub",
        "download",
        str(dataset["repository"]),
        "coqa_validation.parquet",
        "--repo-type",
        "dataset",
        "--revision",
        str(dataset["revision"]),
        "--local-dir",
        str(Path(dataset["file"]).parent),
        "--max-workers",
        "8",
    ]


def verify_dataset_file(configuration: Mapping[str, Any]) -> dict[str, Any]:
    dataset = configuration["dataset"]
    path = Path(dataset["file"])
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_size = path.stat().st_size
    observed_sha = sha256_file(path)
    if observed_size != int(dataset["expected_nbytes"]):
        raise ValueError("CoQA parquet byte size differs from the frozen LFS object")
    if observed_sha != dataset["expected_sha256"]:
        raise ValueError("CoQA parquet SHA256 differs from the frozen LFS object")
    return {
        "provider": dataset["provider"],
        "repository": dataset["repository"],
        "revision": dataset["revision"],
        "split": dataset["split"],
        "path": str(path.resolve()),
        "sha256": observed_sha,
        "nbytes": observed_size,
        "identity_scope": "full_parquet_payload_sha256",
    }


def _field_sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, Mapping):
        sequence = value.get(field)
        if isinstance(sequence, Sequence) and not isinstance(sequence, (str, bytes)):
            return list(sequence)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result = []
        for row in value:
            if not isinstance(row, Mapping) or field not in row:
                raise ValueError(f"malformed CoQA sequence field {field}")
            result.append(row[field])
        return result
    raise ValueError(f"CoQA field does not contain {field}")


def _additional_reference_columns(value: Any) -> list[list[str]]:
    """Return one answer-text sequence per additional annotator."""

    if value is None:
        return []
    candidates: list[Any]
    if isinstance(value, Mapping):
        if "input_text" in value:
            candidates = [value]
        else:
            candidates = list(value.values())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        candidates = list(value)
    else:
        raise ValueError("malformed CoQA additional_answers")
    result: list[list[str]] = []
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            values = _field_sequence(candidate, "input_text")
        except ValueError:
            # Some parquet conversions wrap an annotator map one level deeper.
            if isinstance(candidate, Mapping):
                for nested in candidate.values():
                    values = _field_sequence(nested, "input_text")
                    result.append([str(value) for value in values])
                continue
            raise
        result.append([str(value) for value in values])
    return result


def conversation_from_row(row: Mapping[str, Any], row_index: int) -> CoQAConversation:
    conversation_id = row.get("id")
    raw_domain = row.get("source")
    story = row.get("story")
    if not all(
        isinstance(value, str) and value
        for value in (conversation_id, raw_domain, story)
    ):
        raise ValueError(f"CoQA row {row_index} lacks id/source/story")
    domain = OFFICIAL_DOMAIN_MAPPING.get(raw_domain)
    if domain is None:
        raise ValueError(f"CoQA row {row_index} has an unsupported domain {raw_domain!r}")
    questions = tuple(str(value) for value in _field_sequence(row.get("questions"), "input_text"))
    primary = tuple(str(value) for value in _field_sequence(row.get("answers"), "input_text"))
    if not questions or len(questions) != len(primary):
        raise ValueError(f"CoQA row {row_index} has misaligned questions/answers")
    additional = _additional_reference_columns(row.get("additional_answers"))
    if any(len(values) != len(questions) for values in additional):
        raise ValueError(f"CoQA row {row_index} has misaligned additional answers")
    # Preserve annotator multiplicity exactly: official leave-one-out scoring
    # weights every human annotation, even when two texts are identical.
    references = tuple(
        tuple([primary[turn], *(values[turn] for values in additional)])
        for turn in range(len(questions))
    )
    return CoQAConversation(
        row_index=row_index,
        conversation_id=conversation_id,
        domain=domain,
        story=story,
        questions=questions,
        primary_answers=primary,
        references=references,
        raw_domain=raw_domain,
    )


def load_conversations(path: str | Path) -> tuple[list[CoQAConversation], dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read the pinned CoQA parquet") from exc
    table = pq.read_table(path)
    required = {"id", "source", "story", "questions", "answers"}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(f"CoQA parquet lacks columns: {sorted(missing)}")
    rows = table.to_pylist()
    conversations = [conversation_from_row(row, index) for index, row in enumerate(rows)]
    ids = [value.conversation_id for value in conversations]
    if len(ids) != len(set(ids)):
        raise ValueError("CoQA conversation IDs are not unique")
    return conversations, {
        "rows": table.num_rows,
        "columns": table.column_names,
        "schema": str(table.schema),
        "schema_sha256": hashlib.sha256(str(table.schema).encode("utf-8")).hexdigest(),
    }


def select_conversations(
    conversations: Sequence[CoQAConversation], configuration: Mapping[str, Any]
) -> tuple[list[SelectedConversation], dict[str, Any]]:
    selection = configuration["selection"]
    namespace = str(selection["namespace"])
    minimum_turns = int(selection["minimum_turns"])
    per_domain = int(selection["conversations_per_domain"])
    by_domain: dict[str, list[tuple[str, CoQAConversation]]] = defaultdict(list)
    for conversation in conversations:
        if conversation.domain not in PAPER_DOMAINS or conversation.turn_count < minimum_turns:
            continue
        rank = hashlib.sha256(
            namespace.encode("utf-8") + b"\0" + conversation.conversation_id.encode("utf-8")
        ).hexdigest()
        by_domain[conversation.domain].append((rank, conversation))
    selected: list[SelectedConversation] = []
    domain_counts = Counter(value.domain for value in conversations)
    eligible_counts = {domain: len(by_domain[domain]) for domain in PAPER_DOMAINS}
    for domain in PAPER_DOMAINS:
        candidates = sorted(by_domain[domain], key=lambda value: (value[0], value[1].conversation_id))
        if len(candidates) < per_domain:
            raise ValueError(
                f"CoQA domain {domain} has only {len(candidates)} conversations with >=10 turns"
            )
        for rank_within_domain, (rank, conversation) in enumerate(
            candidates[:per_domain], start=1
        ):
            selected.append(
                SelectedConversation(
                    selection_index=len(selected),
                    rank_within_domain=rank_within_domain,
                    rank_sha256=rank,
                    conversation=conversation,
                )
            )
    if len(selected) != EXPECTED_CONVERSATIONS:
        raise AssertionError("CoQA selection did not produce exactly 100 conversations")
    manifest_rows = [
        {
            "selection_index": value.selection_index,
            "domain": value.conversation.domain,
            "raw_source_domain": value.conversation.raw_domain,
            "conversation_id": value.conversation.conversation_id,
            "source_row_index": value.conversation.row_index,
            "turn_count": value.conversation.turn_count,
            "rank_within_domain": value.rank_within_domain,
            "rank_sha256": value.rank_sha256,
            "story_sha256": hashlib.sha256(value.conversation.story.encode("utf-8")).hexdigest(),
            "questions_sha256": canonical_sha256(list(value.conversation.questions)),
        }
        for value in selected
    ]
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "protocol": "local_deterministic_completion_of_undisclosed_paper_details",
        "paper_bit_exact_claimed": False,
        "selection_uses_answers_or_gold": False,
        "namespace": namespace,
        "ranking": selection["ranking"],
        "minimum_turns": minimum_turns,
        "headline_handoff_depths": list(HEADLINE_DEPTHS),
        "domains": list(PAPER_DOMAINS),
        "conversations_per_domain": per_domain,
        "total_conversations": len(selected),
        "dataset_domain_counts": dict(sorted(domain_counts.items())),
        "eligible_domain_counts": eligible_counts,
        "rows": manifest_rows,
        "rows_sha256": canonical_sha256(manifest_rows),
    }
    return selected, manifest


def build_prompt(conversation: CoQAConversation, turn: int) -> str:
    if not 1 <= turn <= 10 or turn > conversation.turn_count:
        raise ValueError("invalid CoQA turn")
    transcript: list[str] = []
    for prior in range(turn - 1):
        transcript.append(f"Q: {conversation.questions[prior]}\n\n")
        transcript.append(f"A: {conversation.primary_answers[prior]}\n\n")
    transcript.append(f"Q: {conversation.questions[turn - 1]}\n\n")
    transcript.append("A:")
    # This is the historical EleutherAI lm-eval CoQA completion template.
    # The paper does not disclose its exact prompt, so the choice is recorded
    # as a local deterministic completion rather than claimed bit-exact.
    return conversation.story + "\n\n" + "".join(transcript)


def _encode_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("CoQA tokenizer unexpectedly returned a batch")
        encoded = encoded[0]
    result = [int(token) for token in encoded]
    if any(token < 0 for token in result):
        raise ValueError("CoQA tokenizer returned a negative token ID")
    return result


def build_incremental_token_plan(
    conversation: CoQAConversation, tokenizer: Any
) -> tuple[CoQATurnTokenPlan, ...]:
    """Tokenize ten turns so every cache update is an exact token delta.

    Segment-wise encoding is used only if it is byte-for-byte equivalent at
    the token-ID level to encoding the complete historical prompt.  This
    fail-closed equality check prevents a BPE boundary change from silently
    turning an incremental cache into a different prompt.
    """

    if conversation.turn_count < 10:
        raise ValueError("incremental CoQA protocol requires at least ten turns")
    story_ids = _encode_without_special_tokens(tokenizer, conversation.story + "\n\n")
    question_segments = tuple(
        tuple(
            _encode_without_special_tokens(
                tokenizer,
                f"Q: {conversation.questions[turn]}\n\nA:",
            )
        )
        for turn in range(10)
    )
    answer_segments = tuple(
        tuple(
            _encode_without_special_tokens(
                tokenizer,
                f" {conversation.primary_answers[turn]}\n\n",
            )
        )
        for turn in range(9)
    )
    if not story_ids or any(not segment for segment in question_segments):
        raise ValueError("CoQA story/question segment tokenized to zero tokens")

    assembled = list(story_ids)
    plans: list[CoQATurnTokenPlan] = []
    for index in range(10):
        assembled.extend(question_segments[index])
        turn = index + 1
        prompt = build_prompt(conversation, turn)
        full_prompt_ids = _encode_without_special_tokens(tokenizer, prompt)
        if assembled != full_prompt_ids:
            raise ValueError(
                "CoQA tokenizer changed IDs across a segment boundary; "
                "incremental cache execution would not equal the canonical prompt"
            )
        if len(assembled) < 2:
            raise ValueError("CoQA prompt must contain a non-empty parent and boundary")
        append_after: tuple[int, ...] = ()
        if turn < 10:
            append_after = tuple(
                [assembled[-1], *answer_segments[index], *question_segments[index + 1][:-1]]
            )
        plans.append(
            CoQATurnTokenPlan(
                turn=turn,
                prompt=prompt,
                prompt_token_ids=tuple(assembled),
                parent_token_ids=tuple(assembled[:-1]),
                boundary_token_id=assembled[-1],
                append_after_turn=append_after,
            )
        )
        if turn < 10:
            assembled.extend(answer_segments[index])

    for current, following in zip(plans[:-1], plans[1:], strict=True):
        if current.parent_token_ids + current.append_after_turn != following.parent_token_ids:
            raise AssertionError("CoQA incremental token delta does not advance to next parent")
    return tuple(plans)


def normalize_coqa_answer(text: str) -> str:
    """Official CoQA/SQuAD-style normalization for word-level F1."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        return "".join(character for character in value if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def pair_word_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_coqa_answer(prediction).split()
    reference_tokens = normalize_coqa_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


def coqa_word_f1(prediction: str, references: Sequence[str]) -> float:
    """Match ``CoQAEvaluator._compute_turn_score`` from the official script.

    With multiple human references, the official evaluator leaves out each
    reference in turn, takes the maximum score over the remaining references,
    then averages those maxima.  This is intentionally not a single max over
    all references.
    """

    if not references:
        raise ValueError("CoQA scoring requires at least one reference")
    values = list(references)
    if len(values) == 1:
        return pair_word_f1(prediction, values[0])
    leave_one_out = []
    for excluded in range(len(values)):
        remaining = values[:excluded] + values[excluded + 1 :]
        leave_one_out.append(
            max(pair_word_f1(prediction, reference) for reference in remaining)
        )
    return sum(leave_one_out) / len(leave_one_out)


_ANSWER_LABEL = re.compile(r"^\s*(?:final\s+answer|answer|a\s*\d*)\s*[:=]\s*", re.I)


def format_relaxed_prediction(text: str) -> str:
    nonempty = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return _ANSWER_LABEL.sub("", nonempty).strip()


def prepare_coqa(
    configuration: Mapping[str, Any], output_directory: str | Path
) -> tuple[list[SelectedConversation] | None, dict[str, Any]]:
    output = assert_evaluation_output(output_directory)
    dataset_path = Path(configuration["dataset"]["file"])
    if not dataset_path.is_file():
        plan = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "awaiting_dataset",
            "paper_reference": configuration["paper_reference"],
            "formal_evaluation": {
                "modes": list(MODES),
                "headline_handoff_depths": list(HEADLINE_DEPTHS),
                "expected_conversations": EXPECTED_CONVERSATIONS,
                "expected_records_per_mode": EXPECTED_RECORDS_PER_MODE,
                "selection": configuration["selection"],
                "prompt": configuration["prompt"],
                "generation": configuration["generation"],
                "metric": configuration["metric"],
                "runtime": configuration["runtime"],
            },
            "dataset": {
                key: configuration["dataset"][key]
                for key in (
                    "provider",
                    "repository",
                    "revision",
                    "split",
                    "expected_sha256",
                    "expected_nbytes",
                )
            },
            "local_file": str(dataset_path),
            "download_command_argv": dataset_download_command(configuration),
            "download_command_shell_display": " ".join(dataset_download_command(configuration)),
            "download_started": False,
            "gpu_started": False,
        }
        hashed_json(output / "dataset_prepare_plan.json", plan)
        return None, plan
    dataset_identity = verify_dataset_file(configuration)
    conversations, schema = load_conversations(dataset_path)
    selected, selection_manifest = select_conversations(conversations, configuration)
    receipt = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "ready",
        "dataset": dataset_identity,
        "parquet": schema,
        "selection_manifest_sha256": canonical_sha256(selection_manifest),
    }
    immutable_hashed_json(output / "dataset_receipt.json", receipt)
    immutable_hashed_json(output / "selection_manifest.json", selection_manifest)
    return selected, receipt


def _record_path(output: Path, mode: str, selected: SelectedConversation, depth: int) -> Path:
    short_id = hashlib.sha256(
        selected.conversation.conversation_id.encode("utf-8")
    ).hexdigest()[:16]
    return output / "records" / mode / (
        f"{selected.selection_index:03d}_{short_id}_depth_{depth:02d}.json"
    )


def _load_validation_tokenizer(model_root: str | Path) -> Any:
    """Load only the frozen tokenizer; completed-output validation is CPU-only."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_root, local_files_only=True, use_fast=True
    )


def _token_plan_row(
    item: SelectedConversation, plan: CoQATurnTokenPlan
) -> dict[str, Any]:
    conversation = item.conversation
    return {
        "selection_index": item.selection_index,
        "conversation_id": conversation.conversation_id,
        "domain": conversation.domain,
        "turn": plan.turn,
        "is_headline_handoff_depth": plan.turn in HEADLINE_DEPTHS,
        "prompt_sha256": hashlib.sha256(plan.prompt.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": token_ids_sha256(plan.prompt_token_ids),
        "prompt_token_count": len(plan.prompt_token_ids),
        "parent_token_ids_sha256": token_ids_sha256(plan.parent_token_ids),
        "parent_token_count": len(plan.parent_token_ids),
        "boundary_token_id": plan.boundary_token_id,
        "append_after_turn_token_ids_sha256": (
            token_ids_sha256(plan.append_after_turn)
            if plan.append_after_turn
            else None
        ),
        "append_after_turn_token_count": len(plan.append_after_turn),
        "canonical_full_prompt_token_equality": True,
    }


def _expected_token_plans(
    selected: Sequence[SelectedConversation],
    tokenizer: Any,
    configuration: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, tuple[CoQATurnTokenPlan, ...]]]:
    max_tokens = int(configuration["generation"]["max_new_tokens"])
    max_context = int(configuration["runtime"]["max_context_tokens"])
    rows: list[dict[str, Any]] = []
    plans_by_index: dict[int, tuple[CoQATurnTokenPlan, ...]] = {}
    for item in selected:
        if item.selection_index in plans_by_index:
            raise ValueError("duplicate CoQA selection index while rebuilding token plans")
        plans = build_incremental_token_plan(item.conversation, tokenizer)
        plans_by_index[item.selection_index] = plans
        for plan in plans:
            if len(plan.prompt_token_ids) > max_context:
                raise ValueError("CoQA rebuilt prompt exceeds the frozen context window")
            if (
                plan.turn in HEADLINE_DEPTHS
                and len(plan.prompt_token_ids) + max_tokens > max_context
            ):
                raise ValueError("CoQA rebuilt prompt cannot reserve the generation cap")
            rows.append(_token_plan_row(item, plan))
    rows.sort(key=lambda row: (int(row["selection_index"]), int(row["turn"])))
    if len(rows) != EXPECTED_CONVERSATIONS * 10 or set(plans_by_index) != set(
        range(EXPECTED_CONVERSATIONS)
    ):
        raise ValueError("CoQA rebuilt token plan is not exactly 100 x 10 turns")
    return rows, plans_by_index


def _expected_record_identity(
    *,
    item: SelectedConversation,
    plan: CoQATurnTokenPlan,
    all_plans: Sequence[CoQATurnTokenPlan],
    mode: str,
    fingerprint: str,
    variant: CoQARunVariant,
) -> dict[str, Any]:
    if mode not in MODES or plan.turn not in HEADLINE_DEPTHS:
        raise ValueError("invalid CoQA checkpoint key")
    if len(all_plans) != 10 or all_plans[plan.turn - 1] != plan:
        raise ValueError("CoQA checkpoint plan is not the canonical turn plan")
    conversation = item.conversation
    initial_parent_tokens = len(all_plans[0].parent_token_ids)
    appended_tokens = sum(
        len(previous.append_after_turn) for previous in all_plans[: plan.turn - 1]
    )
    parent_length = len(plan.parent_token_ids)
    references = list(conversation.references[plan.turn - 1])
    expected: dict[str, Any] = {
        "request_fingerprint": fingerprint,
        "mode": mode,
        "selection_index": item.selection_index,
        "conversation_id": conversation.conversation_id,
        "domain": conversation.domain,
        "handoff_depth": plan.turn,
        "prompt_sha256": hashlib.sha256(plan.prompt.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": token_ids_sha256(plan.prompt_token_ids),
        "references_sha256": canonical_sha256(references),
        "rank_within_domain": item.rank_within_domain,
        "rank_sha256": item.rank_sha256,
        "handoff_executed": mode == "mapped",
        "history_policy": "gold_primary_answer",
        "prompt": plan.prompt,
        "prompt_token_count": len(plan.prompt_token_ids),
        "cache_execution": {
            "persistent_parent_model": (
                "Qwen3-14B" if mode == "target_standalone" else "Qwen3-32B"
            ),
            "initial_full_history_prefill_count": 1,
            "initial_parent_token_count": initial_parent_tokens,
            "incremental_append_calls_before_depth": plan.turn - 1,
            "incremental_appended_tokens_before_depth": appended_tokens,
            "parent_token_count_before_handoff": parent_length,
            "parent_token_ids_sha256": token_ids_sha256(plan.parent_token_ids),
            "full_history_reprefill_at_depth": False,
            "decode_used_deep_cloned_disposable_branch": True,
            "persistent_parent_unchanged_by_decode": True,
            "source_to_target_handoffs_for_record": 1 if mode == "mapped" else 0,
        },
        "mapped_prefix_token_count": parent_length if mode == "mapped" else None,
    }
    if variant.curve_variant is not None:
        expected.update(
            {
                "curve_variant": variant.curve_variant,
                "selected_source_layers": variant.selected_source_layers,
            }
        )
    return expected


def _build_run_definition(
    configuration: Mapping[str, Any],
    output: Path,
    receipt: Mapping[str, Any],
    selection_manifest: Mapping[str, Any],
    variant: CoQARunVariant,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    """Recreate the complete request identity from current immutable inputs."""

    source_identity = model_artifact_identity(configuration["paths"]["source_model"])
    target_identity = model_artifact_identity(configuration["paths"]["target_model"])
    source_tokenizer = source_identity["content_hashed_files"]["tokenizer.json"]["sha256"]
    target_tokenizer = target_identity["content_hashed_files"]["tokenizer.json"]["sha256"]
    if source_tokenizer != target_tokenizer:
        raise ValueError("source and target tokenizer.json hashes differ")
    mapper_path = Path(configuration["paths"]["mapper_manifest"]).resolve()
    mapper_root = mapper_path if mapper_path.is_dir() else mapper_path.parent
    mapper_identity = mapper_manifest_identity(mapper_path)
    mapper_raw = read_json(mapper_root / "manifest.json")
    _validate_mapper_for_variant(mapper_raw, mapper_identity, variant)
    variant_fields = _variant_run_fields(variant)
    unsigned: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": variant_fields["kind"],
        "direction": "qwen3_32b_to_14b",
        "paper_reference": variant_fields["paper_reference"],
        "modes": list(MODES),
        "dataset_receipt": dict(receipt),
        "selection_manifest_sha256": sha256_file(output / "selection_manifest.json"),
        "selection_rows_sha256": selection_manifest["rows_sha256"],
        "prompt": configuration["prompt"],
        "generation": configuration["generation"],
        "metric": configuration["metric"],
        "runtime": configuration["runtime"],
        "handoff_semantics": {
            "headline_depths": list(HEADLINE_DEPTHS),
            "at_handoff_depth_t": (
                "the persistent source 32B cache contains story, questions through t, "
                "and gold primary answers through t-1 except the final boundary token; "
                "that accumulated cache is mapped once and target 14B replays the boundary "
                "on a disposable generation branch"
            ),
            "target_baseline": (
                "a persistent target 14B own cache is advanced with the exact same "
                "gold-history token deltas and answers from a disposable generation branch"
            ),
            "prior_answers": "gold primary answers in both modes",
            "cascading_generated_answer_errors_in_history": False,
            "full_history_prefills_per_conversation_mode": 1,
            "after_initial_prefill": "cache_only_incremental_token_delta_updates",
            "full_history_reprefill_at_each_depth": False,
            "one_real_32b_to_14b_cache_transfer_per_mapped_record": True,
            "generated_branches_persist_into_history": False,
            "repeated_cascade_claimed": False,
        },
        "artifacts": {
            "source_model": source_identity,
            "target_model": target_identity,
            "mapper": mapper_identity,
        },
        "code": {
            "reverse_experiment": code_identity(variant.code_files),
            "shared_read_only_runtime": code_identity(
                SHARED_CODE_FILES, root=REPOSITORY_ROOT
            ),
        },
    }
    for field in ("curve_variant", "selected_source_layers", "prerequisite_acceptance"):
        if field in variant_fields:
            unsigned[field] = variant_fields[field]
    fingerprint = canonical_sha256(unsigned)
    return {**unsigned, "request_fingerprint": fingerprint}, mapper_identity, mapper_raw


def _validated_record_scores(record: Mapping[str, Any]) -> tuple[float, float]:
    """Recompute both score layers from the raw prediction and references.

    Checkpoint hashes detect accidental byte changes, but a checkpoint is only
    safe to resume when its scientific payload is internally self-consistent.
    In particular, stored scalar scores must never be trusted as inputs to an
    aggregate.
    """

    prediction = record.get("prediction")
    relaxed_prediction = record.get("relaxed_prediction")
    references_raw = record.get("references")
    if not isinstance(prediction, str) or not isinstance(relaxed_prediction, str):
        raise ValueError("CoQA record lacks string prediction fields")
    if (
        not isinstance(references_raw, Sequence)
        or isinstance(references_raw, (str, bytes))
        or not references_raw
        or any(not isinstance(value, str) for value in references_raw)
    ):
        raise ValueError("CoQA record references must be a non-empty string sequence")
    references = list(references_raw)
    if canonical_sha256(references) != record.get("references_sha256"):
        raise ValueError("CoQA record references do not match references_sha256")
    expected_relaxed = format_relaxed_prediction(prediction)
    if relaxed_prediction != expected_relaxed:
        raise ValueError("CoQA relaxed prediction was not derived from the raw prediction")

    recomputed = {
        "official_word_f1": coqa_word_f1(prediction, references),
        "format_relaxed_word_f1": coqa_word_f1(relaxed_prediction, references),
    }
    for field, expected in recomputed.items():
        observed = record.get(field)
        percent = record.get(f"{field}_percent")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(f"CoQA stored {field} differs from recomputed raw-response F1")
        if (
            isinstance(percent, bool)
            or not isinstance(percent, (int, float))
            or not math.isfinite(float(percent))
            or not math.isclose(
                float(percent), expected * 100.0, rel_tol=0.0, abs_tol=1e-10
            )
        ):
            raise ValueError(f"CoQA stored {field}_percent is inconsistent")
    return recomputed["official_word_f1"], recomputed["format_relaxed_word_f1"]


def _validate_complete_record(
    record: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    tokenizer: Any,
    stop_strings: Sequence[str],
    max_tokens: int,
) -> None:
    if record.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise ValueError("CoQA checkpoint schema version changed")
    if record.get("status") != "complete":
        raise ValueError("CoQA checkpoint is not complete")
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"CoQA checkpoint identity/evidence mismatch ({key})")

    prompt = record.get("prompt")
    prompt_count = record.get("prompt_token_count")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("CoQA checkpoint lacks its raw prompt")
    if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != record.get("prompt_sha256"):
        raise ValueError("CoQA checkpoint prompt does not match prompt_sha256")
    if isinstance(prompt_count, bool) or not isinstance(prompt_count, int) or prompt_count < 2:
        raise ValueError("CoQA checkpoint prompt token count is invalid")

    _validated_record_scores(record)

    generation = record.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("CoQA checkpoint lacks generation evidence")
    token_ids_raw = generation.get("generated_token_ids")
    if (
        not isinstance(token_ids_raw, Sequence)
        or isinstance(token_ids_raw, (str, bytes))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in token_ids_raw
        )
    ):
        raise ValueError("CoQA generated token IDs are malformed")
    token_ids = [int(value) for value in token_ids_raw]
    if generation.get("generated_token_count") != len(token_ids):
        raise ValueError("CoQA generated token count disagrees with raw token IDs")
    if generation.get("requested_max_tokens") != max_tokens:
        raise ValueError("CoQA generation cap differs from the frozen protocol")
    if generation.get("original_context_tokens") != prompt_count or generation.get(
        "used_context_tokens"
    ) != prompt_count:
        raise ValueError("CoQA generation context length differs from the raw prompt")
    if generation.get("truncation") != "error":
        raise ValueError("CoQA checkpoint silently changed the overflow policy")

    decoded = _decode_tokens(tokenizer, token_ids)
    stop_at = min(
        (decoded.find(stop) for stop in stop_strings if stop in decoded),
        default=-1,
    )
    if stop_at >= 0:
        generated_text = decoded[:stop_at]
        expected_reason = "stop_sequence"
        expected_cap_hit = False
    elif len(token_ids) == max_tokens:
        generated_text = decoded
        expected_reason = "max_tokens"
        expected_cap_hit = True
    else:
        generated_text = decoded
        expected_reason = "eos"
        expected_cap_hit = False
    expected_prediction = generated_text.strip().split("\n")[0]
    if record.get("prediction") != expected_prediction:
        raise ValueError("CoQA prediction cannot be reconstructed from generated token IDs")
    if generation.get("stop_reason") != expected_reason:
        raise ValueError("CoQA stop reason disagrees with generated token IDs")
    if generation.get("cap_hit") is not expected_cap_hit:
        raise ValueError("CoQA cap-hit flag disagrees with generated token IDs")


def _complete_record(
    path: Path,
    expected: Mapping[str, Any],
    *,
    tokenizer: Any,
    stop_strings: Sequence[str],
    max_tokens: int,
) -> dict[str, Any] | None:
    if not path.is_file() or not path.with_name(f"{path.name}.sha256.json").is_file():
        return None
    verify_hashed_json(path)
    record = read_json(path)
    try:
        _validate_complete_record(
            record,
            expected,
            tokenizer=tokenizer,
            stop_strings=stop_strings,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        raise ValueError(f"invalid CoQA checkpoint {path}: {exc}") from exc
    return record


def _mode_summary(records: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    values = [record for record in records if record["mode"] == mode]
    if len(values) != EXPECTED_RECORDS_PER_MODE:
        raise ValueError(f"{mode} has {len(values)} records, expected {EXPECTED_RECORDS_PER_MODE}")
    by_depth: dict[str, Any] = {}
    for depth in HEADLINE_DEPTHS:
        rows = [row for row in values if int(row["handoff_depth"]) == depth]
        if len(rows) != EXPECTED_CONVERSATIONS:
            raise ValueError(f"{mode}/handoff depth {depth} is incomplete")
        scores = [_validated_record_scores(row) for row in rows]
        official = sum(value[0] for value in scores) / len(scores)
        relaxed = sum(value[1] for value in scores) / len(scores)
        by_depth[str(depth)] = {
            "count": len(rows),
            "official_word_f1": official,
            "official_word_f1_percent": official * 100.0,
            "format_relaxed_word_f1": relaxed,
            "format_relaxed_word_f1_percent": relaxed * 100.0,
            "cap_hits": sum(bool(row["generation"]["cap_hit"]) for row in rows),
        }
    by_domain: dict[str, Any] = {}
    for domain in PAPER_DOMAINS:
        rows = [row for row in values if row["domain"] == domain]
        if len(rows) != 20 * len(HEADLINE_DEPTHS):
            raise ValueError(f"{mode}/{domain} is incomplete")
        official = sum(_validated_record_scores(row)[0] for row in rows) / len(rows)
        by_domain[domain] = {
            "count": len(rows),
            "official_word_f1": official,
            "official_word_f1_percent": official * 100.0,
        }
    all_scores = [_validated_record_scores(row) for row in values]
    official_overall = sum(value[0] for value in all_scores) / len(all_scores)
    relaxed_overall = sum(value[1] for value in all_scores) / len(all_scores)
    return {
        "records": len(values),
        "conversations": EXPECTED_CONVERSATIONS,
        "headline_handoff_depths": list(HEADLINE_DEPTHS),
        "official_word_f1": official_overall,
        "official_word_f1_percent": official_overall * 100.0,
        "format_relaxed_word_f1": relaxed_overall,
        "format_relaxed_word_f1_percent": relaxed_overall * 100.0,
        "by_handoff_depth": by_depth,
        "by_domain": by_domain,
        "cap_hits": sum(bool(row["generation"]["cap_hit"]) for row in values),
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_total = len(MODES) * EXPECTED_RECORDS_PER_MODE
    if len(records) != expected_total:
        raise ValueError(f"CoQA run has {len(records)} records, expected {expected_total}")
    paired: dict[str, dict[tuple[int, int], Mapping[str, Any]]] = {}
    for mode in MODES:
        rows: dict[tuple[int, int], Mapping[str, Any]] = {}
        for record in records:
            if record["mode"] != mode:
                continue
            key = (int(record["selection_index"]), int(record["handoff_depth"]))
            if key in rows:
                raise ValueError(f"duplicate CoQA record for {mode}/{key}")
            rows[key] = record
        paired[mode] = rows
    if set(paired["target_standalone"]) != set(paired["mapped"]):
        raise ValueError("target and mapped CoQA records do not have identical paired keys")
    for key, target_record in paired["target_standalone"].items():
        mapped_record = paired["mapped"][key]
        for field in (
            "conversation_id",
            "domain",
            "prompt_sha256",
            "prompt_token_ids_sha256",
            "prompt_token_count",
            "references_sha256",
            "history_policy",
        ):
            if target_record.get(field) != mapped_record.get(field):
                raise ValueError(f"paired CoQA records disagree on {field}: {key}")
        target_cache = target_record.get("cache_execution")
        mapped_cache = mapped_record.get("cache_execution")
        if not isinstance(target_cache, Mapping) or not isinstance(mapped_cache, Mapping):
            raise ValueError(f"paired CoQA records lack cache execution evidence: {key}")
        for field in (
            "initial_full_history_prefill_count",
            "initial_parent_token_count",
            "incremental_append_calls_before_depth",
            "incremental_appended_tokens_before_depth",
            "parent_token_count_before_handoff",
            "parent_token_ids_sha256",
            "full_history_reprefill_at_depth",
        ):
            if target_cache.get(field) != mapped_cache.get(field):
                raise ValueError(
                    f"paired target/mapped CoQA histories disagree on {field}: {key}"
                )
    summaries = {mode: _mode_summary(records, mode) for mode in MODES}
    drift_by_depth: dict[str, Any] = {}
    for depth in HEADLINE_DEPTHS:
        target = summaries["target_standalone"]["by_handoff_depth"][str(depth)][
            "official_word_f1"
        ]
        mapped = summaries["mapped"]["by_handoff_depth"][str(depth)][
            "official_word_f1"
        ]
        drift_by_depth[str(depth)] = {
            "target_standalone_f1": target,
            "mapped_f1": mapped,
            "drift_target_minus_mapped": target - mapped,
            "drift_percentage_points": (target - mapped) * 100.0,
            "retention_mapped_over_target": (mapped / target) if target else None,
        }
    target_overall = summaries["target_standalone"]["official_word_f1"]
    mapped_overall = summaries["mapped"]["official_word_f1"]
    overall_drift = target_overall - mapped_overall
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "primary_metric": "official_normalized_word_f1",
        "modes": summaries,
        "drift_by_handoff_depth": drift_by_depth,
        "overall_drift_target_minus_mapped": overall_drift,
        "overall_drift_percentage_points": overall_drift * 100.0,
        "overall_retention_mapped_over_target": (
            mapped_overall / target_overall if target_overall else None
        ),
    }


def _control_manifest_binding(
    output: Path, manifest: Mapping[str, Any], field: str, filename: str
) -> dict[str, Any]:
    path = output / filename
    sidecar = verify_hashed_json(path)
    expected = {"sha256": sidecar["sha256"], "nbytes": sidecar["nbytes"]}
    if manifest.get(field) != expected:
        raise ValueError(f"CoQA manifest {field} binding changed")
    return expected


def _read_exact_jsonl(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    field: str,
    filename: str,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = output / filename
    sidecar_path = path.with_name(f"{path.name}.sha256.json")
    sidecar = read_json(sidecar_path)
    expected_sidecar = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "rows": expected_rows,
    }
    if sidecar != expected_sidecar or manifest.get(field) != expected_sidecar:
        raise ValueError(f"CoQA {field} JSONL/sidecar/manifest binding changed")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"CoQA {field} line {line_number} is invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"CoQA {field} line {line_number} is not an object")
            rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(f"CoQA {field} row coverage changed")
    return rows, expected_sidecar


def _validate_sharding_proof(
    value: Any,
    *,
    role: str,
    devices: Sequence[int],
    layer_counts: Sequence[int],
    native_role: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"CoQA {role} sharding proof is missing")
    expected_schema = {
        "status",
        "role",
        "mode",
        "devices",
        "declared_layer_counts",
        "expected_layer_counts",
        "actual_layer_counts",
        "total_layers",
        "layer_devices",
        "contiguous_ranges",
        "actual_parameters_inspected_for_every_layer",
        "parameter_tensors_inspected",
        "auxiliary_modules",
        "auxiliary_parameter_tensors_inspected",
        "auxiliary_parameters_inspected_for_every_module",
    }
    if set(value) != expected_schema:
        raise ValueError(f"CoQA {role} sharding proof schema changed")
    expected_devices = [
        int(device)
        for device, count in zip(devices, layer_counts, strict=True)
        for _ in range(int(count))
    ]
    ranges: list[dict[str, int]] = []
    start = 0
    for device, count in zip(devices, layer_counts, strict=True):
        ranges.append(
            {
                "device": int(device),
                "start_layer": start,
                "end_layer_exclusive": start + int(count),
                "count": int(count),
            }
        )
        start += int(count)
    fixed = {
        "status": "passed",
        "role": native_role,
        "mode": "explicit_exact_counts",
        "devices": list(devices),
        "declared_layer_counts": list(layer_counts),
        "expected_layer_counts": list(layer_counts),
        "actual_layer_counts": list(layer_counts),
        "total_layers": sum(layer_counts),
        "layer_devices": expected_devices,
        "contiguous_ranges": ranges,
        "actual_parameters_inspected_for_every_layer": True,
        "auxiliary_parameters_inspected_for_every_module": True,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"CoQA {role} sharding {field} changed")
    parameter_count = value.get("parameter_tensors_inspected")
    auxiliary_count = value.get("auxiliary_parameter_tensors_inspected")
    if (
        isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
        or isinstance(auxiliary_count, bool)
        or not isinstance(auxiliary_count, int)
        or auxiliary_count <= 0
    ):
        raise ValueError(f"CoQA {role} sharding inspection counts are invalid")
    auxiliary = value.get("auxiliary_modules")
    expected_auxiliary = {
        "model.embed_tokens": int(devices[0]),
        "model.norm": int(devices[-1]),
        "lm_head": int(devices[-1]),
    }
    if not isinstance(auxiliary, Mapping) or set(auxiliary) != set(expected_auxiliary):
        raise ValueError(f"CoQA {role} auxiliary sharding evidence changed")
    recomputed_count = 0
    for module_name, device in expected_auxiliary.items():
        row = auxiliary[module_name]
        if not isinstance(row, Mapping) or set(row) != {
            "expected_device",
            "actual_device",
            "parameter_tensors_inspected",
        }:
            raise ValueError(f"CoQA {role} auxiliary module evidence is malformed")
        count = row.get("parameter_tensors_inspected")
        if (
            row.get("expected_device") != device
            or row.get("actual_device") != device
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise ValueError(f"CoQA {role} auxiliary module device proof changed")
        recomputed_count += count
    if recomputed_count != auxiliary_count:
        raise ValueError(f"CoQA {role} auxiliary inspection total changed")


def _validate_runtime_setup(
    value: Mapping[str, Any],
    *,
    fingerprint: str,
    configuration: Mapping[str, Any],
    mapper_identity: Mapping[str, Any],
    mapper_bundle: Any,
    variant: CoQARunVariant,
) -> None:
    expected_keys = {
        "schema_version",
        "status",
        "request_fingerprint",
        "semantic_role_binding",
        "source_model_sharding",
        "target_model_sharding",
        "mapper",
        "cache_execution",
    }
    if set(value) != expected_keys:
        raise ValueError("CoQA runtime setup schema changed")
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "ready",
        "request_fingerprint": fingerprint,
        "semantic_role_binding": {
            "source": {"model": "Qwen3-32B", "capture_native_forward_role": "target"},
            "target": {"model": "Qwen3-14B", "capture_native_forward_role": "source"},
        },
        "cache_execution": {
            "parent_cache_scope": "one_persistent_parent_per_conversation_and_mode",
            "initial_full_history_prefills": 1,
            "subsequent_history_updates": "incremental_token_deltas_only",
            "headline_handoff_depths": list(HEADLINE_DEPTHS),
            "mapped_parent_model": "Qwen3-32B",
            "target_standalone_parent_model": "Qwen3-14B",
            "decode_branch": "deep_clone_then_discard",
        },
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"CoQA runtime setup {field} changed")
    runtime = configuration["runtime"]
    _validate_sharding_proof(
        value.get("source_model_sharding"),
        role="source",
        devices=runtime["source_devices"],
        layer_counts=runtime["source_layer_counts"],
        native_role="target",
    )
    _validate_sharding_proof(
        value.get("target_model_sharding"),
        role="target",
        devices=runtime["target_devices"],
        layer_counts=runtime["target_layer_counts"],
        native_role="source",
    )
    mapper = value.get("mapper")
    if not isinstance(mapper, Mapping):
        raise ValueError("CoQA runtime mapper evidence is missing")
    target_layer_devices = [
        f"cuda:{device}"
        for device, count in zip(
            runtime["target_devices"], runtime["target_layer_counts"], strict=True
        )
        for _ in range(int(count))
    ]
    local_identity_fields = (
        "model_path",
        "model_config_sha256",
        "model_tokenizer_sha256",
        "model_weight_index_sha256",
    )
    expected_source_artifact = {
        field: mapper_bundle.source_model_identity[field]
        for field in local_identity_fields
    }
    expected_target_artifact = {
        field: mapper_bundle.target_model_identity[field]
        for field in local_identity_fields
    }
    source_features = variant.selected_source_layers * 8 * 128
    target_features = 8 * 128
    resident_parameter_bytes = (
        2 * 40 * (source_features * target_features + target_features) * 4
    )
    mapper_fixed = {
        "direction": "qwen3_32b_to_14b",
        "manifest": str(mapper_bundle.manifest_path),
        "manifest_sha256": mapper_identity["manifest_sha256"],
        "fit_fingerprint": mapper_identity["fit_fingerprint"],
        "layers": 40,
        "selected_source_layers": variant.selected_source_layers,
        "token_chunk": int(runtime["mapper_token_chunk"]),
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
        "target_layer_devices": target_layer_devices,
        "resident_parameter_bytes": resident_parameter_bytes,
        "source_model_artifact": expected_source_artifact,
        "target_model_artifact": expected_target_artifact,
        "source_capture_model_identity": dict(mapper_bundle.source_model_identity),
        "target_capture_model_identity": dict(mapper_bundle.target_model_identity),
    }
    if variant.curve_variant is not None:
        mapper_fixed.update(
            {
                "curve_variant": variant.curve_variant,
                "source_features": source_features,
                "target_features": target_features,
                "independently_refit_not_k20_slice": True,
            }
        )
    if set(mapper) != set(mapper_fixed) | {"materialization"}:
        raise ValueError("CoQA runtime mapper evidence schema changed")
    for field, expected in mapper_fixed.items():
        if mapper.get(field) != expected:
            raise ValueError(f"CoQA runtime mapper {field} changed")
    materialization = mapper.get("materialization")
    if not isinstance(materialization, Mapping) or materialization.get("scope") != (
        "one_time_setup_excluded_from_per_request_latency"
    ):
        raise ValueError("CoQA runtime mapper materialization evidence changed")
    if set(materialization) != {
        "scope",
        "host_load_and_hash",
        "weight_host_to_device",
        "per_layer",
    }:
        raise ValueError("CoQA runtime mapper materialization schema changed")
    per_layer = materialization.get("per_layer")
    if not isinstance(per_layer, list) or len(per_layer) != 40:
        raise ValueError("CoQA runtime mapper materialization layer coverage changed")
    for layer, row in enumerate(per_layer):
        if (
            not isinstance(row, Mapping)
            or row.get("target_layer") != layer
            or row.get("device") != target_layer_devices[layer]
        ):
            raise ValueError("CoQA runtime mapper materialization placement changed")
        for field in ("host_load_and_hash_ms", "weight_host_to_device_ms"):
            number = row.get(field)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not math.isfinite(float(number))
                or float(number) < 0.0
            ):
                raise ValueError("CoQA runtime mapper materialization timing is invalid")


def validate_completed_output(
    output_directory: str | Path,
    *,
    configuration: Mapping[str, Any] | None = None,
    variant: CoQARunVariant = DEFAULT_RUN_VARIANT,
    mapper_builder_class: Any | None = None,
    mapper_bundle_class: Any | None = None,
) -> dict[str, Any] | None:
    """Fail-closed validation of a sealed k=20 or retuned-k=4 CoQA run.

    Every one of the 1,000 checkpoint payloads is checked with the same strong
    semantic validator used for resume.  Dataset selection and all 1,000 token
    plans are regenerated from the pinned parquet/tokenizer, while summary,
    raw JSONL, checkpoint inventory, and final manifest are independently
    rebuilt.  A coordinated rewrite of payloads and SHA sidecars is therefore
    not sufficient for adoption.
    """

    output = Path(output_directory).resolve()
    try:
        output.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("CoQA output escaped the reverse experiment") from exc
    manifest_path = output / "manifest.json"
    manifest_sidecar = manifest_path.with_name("manifest.json.sha256.json")
    if not manifest_path.exists():
        if manifest_sidecar.exists():
            raise ValueError("CoQA manifest sidecar exists without its payload")
        return None
    verify_hashed_json(manifest_path)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("CoQA manifest is not an object")
    config = dict(configuration) if configuration is not None else load_coqa_config()
    _validate_run_variant(variant, mapper_builder_class)
    if variant.curve_variant is None:
        if mapper_bundle_class is not None:
            raise ValueError("production CoQA cannot inject a mapper-bundle class")
    elif (
        mapper_bundle_class is None
        or getattr(mapper_bundle_class, "__module__", "")
        != "reverse_kv.k4_runtime"
        or getattr(mapper_bundle_class, "__name__", "")
        != "K4ReverseMapperBundle"
    ):
        raise ValueError("k4 CoQA requires its dedicated fail-closed mapper bundle")

    dataset_identity = verify_dataset_file(config)
    conversations, parquet_schema = load_conversations(config["dataset"]["file"])
    selected, expected_selection = select_conversations(conversations, config)
    expected_receipt = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "ready",
        "dataset": dataset_identity,
        "parquet": parquet_schema,
        "selection_manifest_sha256": canonical_sha256(expected_selection),
    }
    for filename in (
        "dataset_receipt.json",
        "selection_manifest.json",
        "run_definition.json",
        "runtime_setup.json",
        "summary.json",
    ):
        verify_hashed_json(output / filename)
    if read_json(output / "dataset_receipt.json") != expected_receipt:
        raise ValueError("CoQA dataset receipt differs from the pinned parquet")
    if read_json(output / "selection_manifest.json") != expected_selection:
        raise ValueError("CoQA selection manifest differs from deterministic reselection")

    expected_definition, mapper_identity, mapper_raw = _build_run_definition(
        config, output, expected_receipt, expected_selection, variant
    )
    del mapper_raw
    definition = read_json(output / "run_definition.json")
    if definition != expected_definition:
        raise ValueError("CoQA run definition differs from current frozen inputs/code")
    fingerprint = str(expected_definition["request_fingerprint"])

    mapper_path = Path(config["paths"]["mapper_manifest"]).resolve()
    mapper_root = mapper_path if mapper_path.is_dir() else mapper_path.parent
    if mapper_bundle_class is None:
        from .runtime import ReverseMapperBundle

        mapper_bundle_class = ReverseMapperBundle
    mapper_bundle = mapper_bundle_class.open(mapper_root)
    if (
        mapper_bundle.manifest_sha256 != mapper_identity["manifest_sha256"]
        or mapper_bundle.fit_fingerprint != mapper_identity["fit_fingerprint"]
    ):
        raise ValueError("CoQA current mapper bundle differs from the run identity")
    runtime_setup = read_json(output / "runtime_setup.json")
    _validate_runtime_setup(
        runtime_setup,
        fingerprint=fingerprint,
        configuration=config,
        mapper_identity=mapper_identity,
        mapper_bundle=mapper_bundle,
        variant=variant,
    )

    tokenizer = _load_validation_tokenizer(config["paths"]["target_model"])
    expected_plan_rows, plans_by_index = _expected_token_plans(
        selected, tokenizer, config
    )
    token_plan_rows, token_plan_sidecar = _read_exact_jsonl(
        output,
        manifest,
        field="token_plans",
        filename="token_plans.jsonl",
        expected_rows=EXPECTED_CONVERSATIONS * 10,
    )
    if token_plan_rows != expected_plan_rows or (
        output / "token_plans.jsonl"
    ).read_bytes() != canonical_jsonl_bytes(expected_plan_rows):
        raise ValueError("CoQA token plans differ from the pinned data/tokenizer")

    max_tokens = int(config["generation"]["max_new_tokens"])
    stop_strings = list(config["generation"]["stop_sequences"])
    records: list[dict[str, Any]] = []
    expected_checkpoint_rows: list[dict[str, Any]] = []
    expected_payload_paths: set[Path] = set()
    expected_all_record_files: set[Path] = set()
    record_fields = {
        "schema_version",
        "status",
        "request_fingerprint",
        "mode",
        "selection_index",
        "conversation_id",
        "domain",
        "handoff_depth",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "references_sha256",
        "rank_within_domain",
        "rank_sha256",
        "handoff_executed",
        "history_policy",
        "prompt",
        "prompt_token_count",
        "cache_execution",
        "mapped_prefix_token_count",
        "references",
        "prediction",
        "relaxed_prediction",
        "official_word_f1",
        "official_word_f1_percent",
        "format_relaxed_word_f1",
        "format_relaxed_word_f1_percent",
        "generation",
    }
    if variant.curve_variant is not None:
        record_fields.update({"curve_variant", "selected_source_layers"})
    generation_fields = {
        "generated_token_ids",
        "generated_token_count",
        "stop_reason",
        "cap_hit",
        "requested_max_tokens",
        "original_context_tokens",
        "used_context_tokens",
        "truncation",
    }
    for mode in MODES:
        for item in selected:
            plans = plans_by_index[item.selection_index]
            for depth in HEADLINE_DEPTHS:
                plan = plans[depth - 1]
                expected = _expected_record_identity(
                    item=item,
                    plan=plan,
                    all_plans=plans,
                    mode=mode,
                    fingerprint=fingerprint,
                    variant=variant,
                )
                path = _record_path(output, mode, item, depth).resolve()
                if output not in path.parents:
                    raise ValueError("CoQA expected checkpoint escaped its run root")
                expected_payload_paths.add(path)
                expected_all_record_files.update(
                    {path, path.with_name(f"{path.name}.sha256.json")}
                )
                record = _complete_record(
                    path,
                    expected,
                    tokenizer=tokenizer,
                    stop_strings=stop_strings,
                    max_tokens=max_tokens,
                )
                if record is None:
                    raise FileNotFoundError(f"missing complete CoQA checkpoint: {path}")
                if set(record) != record_fields:
                    raise ValueError(f"CoQA checkpoint schema changed: {path}")
                generation = record.get("generation")
                if not isinstance(generation, Mapping) or set(generation) != generation_fields:
                    raise ValueError(f"CoQA generation schema changed: {path}")
                references = list(item.conversation.references[depth - 1])
                if record.get("references") != references:
                    raise ValueError(f"CoQA checkpoint references differ from dataset: {path}")
                records.append(record)
                expected_checkpoint_rows.append(
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": sha256_file(path),
                        "nbytes": path.stat().st_size,
                    }
                )
    records_root = output / "records"
    actual_record_files = (
        {path.resolve() for path in records_root.rglob("*.json")}
        if records_root.is_dir()
        else set()
    )
    if actual_record_files != expected_all_record_files:
        raise ValueError("CoQA checkpoint/sidecar filesystem inventory is not exact")
    if len(expected_payload_paths) != len(MODES) * EXPECTED_RECORDS_PER_MODE:
        raise AssertionError("CoQA validator did not enumerate exactly 1,000 checkpoints")
    if manifest.get("checkpoints") != expected_checkpoint_rows or manifest.get(
        "checkpoint_manifest_sha256"
    ) != canonical_sha256(expected_checkpoint_rows):
        raise ValueError("CoQA checkpoint manifest differs from the 1,000 payloads")

    raw_rows, raw_sidecar = _read_exact_jsonl(
        output,
        manifest,
        field="raw_records",
        filename="raw_records.jsonl",
        expected_rows=len(MODES) * EXPECTED_RECORDS_PER_MODE,
    )
    if raw_rows != records or (output / "raw_records.jsonl").read_bytes() != (
        canonical_jsonl_bytes(records)
    ):
        raise ValueError("CoQA raw-record chain differs from exact checkpoints")
    expected_summary = summarize_records(records)
    if read_json(output / "summary.json") != expected_summary:
        raise ValueError("CoQA summary differs from recomputed checkpoint scores")
    summary_sidecar = verify_hashed_json(output / "summary.json")

    control_bindings = {
        field: _control_manifest_binding(output, manifest, field, filename)
        for field, filename in (
            ("run_definition", "run_definition.json"),
            ("dataset_receipt", "dataset_receipt.json"),
            ("selection_manifest", "selection_manifest.json"),
            ("runtime_setup", "runtime_setup.json"),
        )
    }
    expected_manifest: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "direction": "qwen3_32b_to_14b",
        "task": "coqa_multi_turn_handoff",
        "paper_bit_exact_claimed": False,
        "headline_handoff_depths": list(HEADLINE_DEPTHS),
        "records": len(records),
        "expected_records": len(MODES) * EXPECTED_RECORDS_PER_MODE,
        **control_bindings,
        "token_plans": token_plan_sidecar,
        "raw_records": raw_sidecar,
        "summary": summary_sidecar,
        "checkpoint_manifest_sha256": canonical_sha256(expected_checkpoint_rows),
        "checkpoints": expected_checkpoint_rows,
    }
    if variant.curve_variant is not None:
        expected_manifest.update(
            {
                "curve_variant": variant.curve_variant,
                "selected_source_layers": variant.selected_source_layers,
                "prerequisite_acceptance": dict(
                    variant.prerequisite_acceptance or {}
                ),
            }
        )
    if set(manifest) != set(expected_manifest) | {"finished_utc"}:
        raise ValueError("CoQA final manifest schema changed")
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"CoQA final manifest {field} changed")
    finished = manifest.get("finished_utc")
    if not isinstance(finished, str):
        raise ValueError("CoQA final manifest lacks completion time")
    try:
        timestamp = dt.datetime.fromisoformat(finished)
    except ValueError as exc:
        raise ValueError("CoQA completion time is malformed") from exc
    if timestamp.tzinfo is None:
        raise ValueError("CoQA completion time is not timezone-aware")
    return dict(manifest)


def _default_output(prepare_only: bool) -> Path:
    if prepare_only:
        return EXPERIMENT_ROOT / "artifacts/inputs/coqa_prepare_v1"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    return EXPERIMENT_ROOT / "runs" / f"{stamp}_coqa_reverse_k20"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args(argv)


def _model_input_device(model: torch.nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    try:
        device = next(embeddings.parameters()).device
    except StopIteration as exc:
        raise ValueError("model input embedding has no parameters") from exc
    if device.type == "meta":
        raise ValueError("model input embedding is still on the meta device")
    return device


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(list(token_ids))


@torch.inference_mode()
def _prefill_persistent_parent(
    model: torch.nn.Module, parent_token_ids: Sequence[int]
) -> Any:
    """Create the one permitted full-history cache for a conversation/mode."""

    from kvcache_transfer.cache_adapter import cache_sequence_length

    values = [int(token) for token in parent_token_ids]
    if not values:
        raise ValueError("persistent CoQA parent cache cannot be empty")
    base = getattr(model, "model", model)
    device = _model_input_device(model)
    input_ids = torch.tensor([values], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = base(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=torch.arange(len(values), device=device).unsqueeze(0),
        cache_position=torch.arange(len(values), device=device),
        use_cache=True,
        return_dict=True,
    )
    cache = output.past_key_values
    if cache_sequence_length(cache) != len(values):
        raise ValueError("initial CoQA parent cache has the wrong sequence length")
    return cache


@torch.inference_mode()
def _append_persistent_parent(
    model: torch.nn.Module,
    parent_cache: Any,
    appended_token_ids: Sequence[int],
) -> Any:
    """Mutate/advance a persistent parent using only the next transcript delta."""

    from kvcache_transfer.cache_adapter import (
        cache_sequence_length,
        full_attention_mask,
        replay_cache_position,
        replay_position_ids,
    )

    values = [int(token) for token in appended_token_ids]
    if not values:
        raise ValueError("persistent CoQA cache append cannot be empty")
    prefix_length = cache_sequence_length(parent_cache)
    base = getattr(model, "model", model)
    device = _model_input_device(model)
    input_ids = torch.tensor([values], dtype=torch.long, device=device)
    attention_mask = full_attention_mask(
        prefix_length,
        len(values),
        batch_size=1,
        device=device,
    )
    output = base(
        input_ids=input_ids,
        past_key_values=parent_cache,
        attention_mask=attention_mask,
        position_ids=replay_position_ids(attention_mask, len(values)),
        cache_position=replay_cache_position(prefix_length, len(values), device=device),
        use_cache=True,
        return_dict=True,
    )
    cache = output.past_key_values
    expected = prefix_length + len(values)
    if cache_sequence_length(cache) != expected:
        raise ValueError(
            f"incremental CoQA parent cache length changed incorrectly: expected {expected}"
        )
    return cache


@torch.inference_mode()
def _generate_from_persistent_parent(
    *,
    target_model: torch.nn.Module,
    tokenizer: Any,
    parent_cache: Any,
    boundary_token_id: int,
    context_tokens: int,
    max_tokens: int,
    stop_strings: Sequence[str],
) -> PersistentGenerationResult:
    """Greedily decode from a disposable branch of one accumulated parent."""

    from kvcache_transfer.cache_adapter import (
        boundary_replay_inputs,
        cache_sequence_length,
        full_attention_mask,
        replay_cache_position,
        replay_position_ids,
    )

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if any(not isinstance(stop, str) or not stop for stop in stop_strings):
        raise ValueError("stop strings must be non-empty strings")
    parent_length = cache_sequence_length(parent_cache)
    if context_tokens != parent_length + 1:
        raise ValueError("CoQA context length does not equal parent plus boundary")
    device = _model_input_device(target_model)
    boundary = torch.tensor([[int(boundary_token_id)]], dtype=torch.long, device=device)
    inputs = boundary_replay_inputs(boundary, parent_cache, clone_cache=True)
    output = target_model(**inputs, return_dict=True)

    eos_ids: set[int] = set()
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        eos_ids.add(int(tokenizer_eos))
    generation_eos = getattr(
        getattr(target_model, "generation_config", None), "eos_token_id", None
    )
    if isinstance(generation_eos, int):
        eos_ids.add(generation_eos)
    elif generation_eos is not None:
        eos_ids.update(int(value) for value in generation_eos)

    generated: list[int] = []
    for step in range(max_tokens):
        next_token = int(output.logits[:, -1, :].argmax(-1).item())
        reached_eos = next_token in eos_ids
        if not reached_eos:
            generated.append(next_token)
        text = _decode_tokens(tokenizer, generated)
        stop_at = min(
            (text.find(stop) for stop in stop_strings if stop in text),
            default=-1,
        )
        if stop_at >= 0:
            return PersistentGenerationResult(
                text=text[:stop_at],
                generated_token_ids=tuple(generated),
                stop_reason="stop_sequence",
                cap_hit=False,
                requested_max_tokens=max_tokens,
                context_tokens=context_tokens,
            )
        if reached_eos:
            return PersistentGenerationResult(
                text=text,
                generated_token_ids=tuple(generated),
                stop_reason="eos",
                cap_hit=False,
                requested_max_tokens=max_tokens,
                context_tokens=context_tokens,
            )
        if step + 1 == max_tokens:
            break
        past = output.past_key_values
        prefix_length = cache_sequence_length(past)
        token = torch.tensor([[next_token]], dtype=torch.long, device=device)
        attention_mask = full_attention_mask(
            prefix_length,
            1,
            batch_size=1,
            device=device,
        )
        output = target_model(
            input_ids=token,
            past_key_values=past,
            attention_mask=attention_mask,
            position_ids=replay_position_ids(attention_mask, 1),
            cache_position=replay_cache_position(prefix_length, 1, device=device),
            use_cache=True,
            return_dict=True,
        )
    return PersistentGenerationResult(
        text=_decode_tokens(tokenizer, generated),
        generated_token_ids=tuple(generated),
        stop_reason="max_tokens",
        cap_hit=True,
        requested_max_tokens=max_tokens,
        context_tokens=context_tokens,
    )


def run_evaluation(
    configuration: Mapping[str, Any],
    output: Path,
    *,
    variant: CoQARunVariant = DEFAULT_RUN_VARIANT,
    mapper_builder_class: Any | None = None,
) -> int:
    _validate_run_variant(variant, mapper_builder_class)
    selected, receipt = prepare_coqa(configuration, output)
    if selected is None:
        raise FileNotFoundError(
            f"pinned CoQA file is missing; see {output / 'dataset_prepare_plan.json'}"
        )

    selection_manifest = read_json(output / "selection_manifest.json")
    run_definition, _, _ = _build_run_definition(
        configuration, output, receipt, selection_manifest, variant
    )
    fingerprint = str(run_definition["request_fingerprint"])
    immutable_hashed_json(output / "run_definition.json", run_definition)

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from transformers import AutoTokenizer
    from kvcache_transfer.cache_adapter import cache_sequence_length
    from kvcache_transfer.modeling import load_qwen3
    from kvcache_transfer.transfer import transfer_source_cache
    if mapper_builder_class is None:
        from .runtime import ReverseMappedPrefixCacheBuilder

        mapper_builder_class = ReverseMappedPrefixCacheBuilder

    runtime = configuration["runtime"]
    target = load_qwen3(
        configuration["paths"]["target_model"],
        devices=runtime["target_devices"],
        layer_counts=runtime["target_layer_counts"],
        # Shared loader names are capture-native/forward roles: 14B is source.
        role="source",
        attention_implementation=runtime["attention_implementation"],
    )
    source = load_qwen3(
        configuration["paths"]["source_model"],
        devices=runtime["source_devices"],
        layer_counts=runtime["source_layer_counts"],
        # Shared loader names are capture-native/forward roles: 32B is target.
        role="target",
        attention_implementation=runtime["attention_implementation"],
    )
    tokenizer = AutoTokenizer.from_pretrained(
        configuration["paths"]["target_model"], local_files_only=True, use_fast=True
    )
    builder = mapper_builder_class(
        source_model=source,
        target_model=target,
        mapper_manifest=configuration["paths"]["mapper_manifest"],
        token_chunk=int(runtime["mapper_token_chunk"]),
    )
    hashed_json(
        output / "runtime_setup.json",
        {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "ready",
            "request_fingerprint": fingerprint,
            "semantic_role_binding": {
                "source": {"model": "Qwen3-32B", "capture_native_forward_role": "target"},
                "target": {"model": "Qwen3-14B", "capture_native_forward_role": "source"},
            },
            "source_model_sharding": getattr(
                source, "_kvcache_transfer_layer_sharding_evidence", None
            ),
            "target_model_sharding": getattr(
                target, "_kvcache_transfer_layer_sharding_evidence", None
            ),
            "mapper": builder.provenance,
            "cache_execution": {
                "parent_cache_scope": "one_persistent_parent_per_conversation_and_mode",
                "initial_full_history_prefills": 1,
                "subsequent_history_updates": "incremental_token_deltas_only",
                "headline_handoff_depths": list(HEADLINE_DEPTHS),
                "mapped_parent_model": "Qwen3-32B",
                "target_standalone_parent_model": "Qwen3-14B",
                "decode_branch": "deep_clone_then_discard",
            },
        },
    )

    records: list[dict[str, Any]] = []
    token_plan_rows: list[dict[str, Any]] = []
    max_tokens = int(configuration["generation"]["max_new_tokens"])
    max_context = int(runtime["max_context_tokens"])
    stop_strings = list(configuration["generation"]["stop_sequences"])
    for item in selected:
        conversation = item.conversation
        plans = build_incremental_token_plan(conversation, tokenizer)
        initial_parent_tokens = len(plans[0].parent_token_ids)
        for plan in plans:
            if len(plan.prompt_token_ids) > max_context:
                raise ValueError(
                    f"CoQA {conversation.conversation_id}/turn {plan.turn} has "
                    f"{len(plan.prompt_token_ids)} tokens, exceeding max context {max_context}"
                )
            if (
                plan.turn in HEADLINE_DEPTHS
                and len(plan.prompt_token_ids) + max_tokens > max_context
            ):
                raise ValueError(
                    f"CoQA {conversation.conversation_id}/depth {plan.turn} cannot reserve "
                    f"the frozen {max_tokens}-token generation cap"
                )
            token_plan_rows.append(
                {
                    "selection_index": item.selection_index,
                    "conversation_id": conversation.conversation_id,
                    "domain": conversation.domain,
                    "turn": plan.turn,
                    "is_headline_handoff_depth": plan.turn in HEADLINE_DEPTHS,
                    "prompt_sha256": hashlib.sha256(
                        plan.prompt.encode("utf-8")
                    ).hexdigest(),
                    "prompt_token_ids_sha256": token_ids_sha256(plan.prompt_token_ids),
                    "prompt_token_count": len(plan.prompt_token_ids),
                    "parent_token_ids_sha256": token_ids_sha256(plan.parent_token_ids),
                    "parent_token_count": len(plan.parent_token_ids),
                    "boundary_token_id": plan.boundary_token_id,
                    "append_after_turn_token_ids_sha256": (
                        token_ids_sha256(plan.append_after_turn)
                        if plan.append_after_turn
                        else None
                    ),
                    "append_after_turn_token_count": len(plan.append_after_turn),
                    "canonical_full_prompt_token_equality": True,
                }
            )

        for mode in MODES:
            parent_model = target if mode == "target_standalone" else source
            parent_model_name = "Qwen3-14B" if mode == "target_standalone" else "Qwen3-32B"
            parent_cache = _prefill_persistent_parent(
                parent_model, plans[0].parent_token_ids
            )
            if cache_sequence_length(parent_cache) != initial_parent_tokens:
                raise AssertionError("CoQA initial parent cache evidence is inconsistent")
            append_calls = 0
            appended_tokens = 0
            for plan in plans:
                parent_length = cache_sequence_length(parent_cache)
                if parent_length != len(plan.parent_token_ids):
                    raise ValueError(
                        f"persistent {mode} cache at turn {plan.turn} has {parent_length} "
                        f"tokens, expected {len(plan.parent_token_ids)}"
                    )
                if plan.turn in HEADLINE_DEPTHS:
                    prompt_sha = hashlib.sha256(plan.prompt.encode("utf-8")).hexdigest()
                    prompt_ids_sha = token_ids_sha256(plan.prompt_token_ids)
                    references = list(conversation.references[plan.turn - 1])
                    references_sha = canonical_sha256(references)
                    expected = {
                        "request_fingerprint": fingerprint,
                        "mode": mode,
                        "selection_index": item.selection_index,
                        "conversation_id": conversation.conversation_id,
                        "domain": conversation.domain,
                        "handoff_depth": plan.turn,
                        "prompt_sha256": prompt_sha,
                        "prompt_token_ids_sha256": prompt_ids_sha,
                        "references_sha256": references_sha,
                        "rank_within_domain": item.rank_within_domain,
                        "rank_sha256": item.rank_sha256,
                        "handoff_executed": mode == "mapped",
                        "history_policy": "gold_primary_answer",
                        "prompt": plan.prompt,
                        "prompt_token_count": len(plan.prompt_token_ids),
                        "cache_execution": {
                            "persistent_parent_model": parent_model_name,
                            "initial_full_history_prefill_count": 1,
                            "initial_parent_token_count": initial_parent_tokens,
                            "incremental_append_calls_before_depth": append_calls,
                            "incremental_appended_tokens_before_depth": appended_tokens,
                            "parent_token_count_before_handoff": parent_length,
                            "parent_token_ids_sha256": token_ids_sha256(
                                plan.parent_token_ids
                            ),
                            "full_history_reprefill_at_depth": False,
                            "decode_used_deep_cloned_disposable_branch": True,
                            "persistent_parent_unchanged_by_decode": True,
                            "source_to_target_handoffs_for_record": (
                                1 if mode == "mapped" else 0
                            ),
                        },
                        "mapped_prefix_token_count": (
                            parent_length if mode == "mapped" else None
                        ),
                    }
                    if variant.curve_variant is not None:
                        expected.update(
                            {
                                "curve_variant": variant.curve_variant,
                                "selected_source_layers": variant.selected_source_layers,
                            }
                        )
                    destination = _record_path(output, mode, item, plan.turn)
                    checkpoint = _complete_record(
                        destination,
                        expected,
                        tokenizer=tokenizer,
                        stop_strings=stop_strings,
                        max_tokens=max_tokens,
                    )
                    if checkpoint is not None:
                        records.append(checkpoint)
                    else:
                        generation_parent = parent_cache
                        if mode == "mapped":
                            generation_parent = transfer_source_cache(
                                parent_cache,
                                source_config=source.config,
                                target_model=target,
                                mapper_bundle=builder.mapper_bundle,
                                position_ids=None,
                                output_dtype=None,
                                token_chunk=int(runtime["mapper_token_chunk"]),
                                strict_validation=True,
                            )
                        mapped_length = cache_sequence_length(generation_parent)
                        if mapped_length != parent_length:
                            raise ValueError("CoQA handoff changed the parent sequence length")
                        generated = _generate_from_persistent_parent(
                            target_model=target,
                            tokenizer=tokenizer,
                            parent_cache=generation_parent,
                            boundary_token_id=plan.boundary_token_id,
                            context_tokens=len(plan.prompt_token_ids),
                            max_tokens=max_tokens,
                            stop_strings=stop_strings,
                        )
                        if cache_sequence_length(parent_cache) != parent_length:
                            raise ValueError("CoQA generation contaminated its persistent parent")
                        # Only first-line response postprocessing mirrors the
                        # vendored lm-eval task.  Reference construction and F1
                        # follow Stanford's official evaluator: all four human
                        # annotations retain multiplicity (lm-eval's
                        # doc_to_target case-insensitive dedup is not used).
                        prediction = generated.text.strip().split("\n")[0]
                        relaxed_prediction = format_relaxed_prediction(prediction)
                        official_f1 = coqa_word_f1(prediction, references)
                        relaxed_f1 = coqa_word_f1(relaxed_prediction, references)
                        record = {
                            "schema_version": EVALUATION_SCHEMA_VERSION,
                            "status": "complete",
                            **expected,
                            "references": references,
                            "prediction": prediction,
                            "relaxed_prediction": relaxed_prediction,
                            "official_word_f1": official_f1,
                            "official_word_f1_percent": official_f1 * 100.0,
                            "format_relaxed_word_f1": relaxed_f1,
                            "format_relaxed_word_f1_percent": relaxed_f1 * 100.0,
                            "generation": {
                                "generated_token_ids": list(generated.generated_token_ids),
                                "generated_token_count": generated.generated_token_count,
                                "stop_reason": generated.stop_reason,
                                "cap_hit": generated.cap_hit,
                                "requested_max_tokens": generated.requested_max_tokens,
                                "original_context_tokens": generated.context_tokens,
                                "used_context_tokens": generated.context_tokens,
                                "truncation": "error",
                            },
                        }
                        _validate_complete_record(
                            record,
                            expected,
                            tokenizer=tokenizer,
                            stop_strings=stop_strings,
                            max_tokens=max_tokens,
                        )
                        hashed_json(destination, record)
                        records.append(record)
                        if mode == "mapped":
                            del generation_parent
                if plan.turn < 10:
                    parent_cache = _append_persistent_parent(
                        parent_model,
                        parent_cache,
                        plan.append_after_turn,
                    )
                    append_calls += 1
                    appended_tokens += len(plan.append_after_turn)
                    expected_next_parent = plans[plan.turn].parent_token_ids
                    if cache_sequence_length(parent_cache) != len(expected_next_parent):
                        raise AssertionError("CoQA cache delta did not reach the next turn")
            if append_calls != 9:
                raise AssertionError("CoQA conversation did not receive nine cache deltas")
            del parent_cache

    records.sort(
        key=lambda row: (
            MODES.index(str(row["mode"])),
            int(row["selection_index"]),
            int(row["handoff_depth"]),
        )
    )
    token_plan_rows.sort(
        key=lambda row: (int(row["selection_index"]), int(row["turn"]))
    )
    if len(token_plan_rows) != EXPECTED_CONVERSATIONS * 10:
        raise AssertionError("CoQA token plan does not contain 100 x 10 turns")
    token_plan_sidecar = hashed_jsonl(output / "token_plans.jsonl", token_plan_rows)
    raw_sidecar = hashed_jsonl(output / "raw_records.jsonl", records)
    summary = summarize_records(records)
    summary_sidecar = hashed_json(output / "summary.json", summary)
    checkpoint_rows = []
    for mode in MODES:
        for item in selected:
            for depth in HEADLINE_DEPTHS:
                path = _record_path(output, mode, item, depth)
                verify_hashed_json(path)
                checkpoint_rows.append(
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": sha256_file(path),
                        "nbytes": path.stat().st_size,
                    }
                )
    manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "direction": "qwen3_32b_to_14b",
        "task": "coqa_multi_turn_handoff",
        "paper_bit_exact_claimed": False,
        "headline_handoff_depths": list(HEADLINE_DEPTHS),
        "records": len(records),
        "expected_records": len(MODES) * EXPECTED_RECORDS_PER_MODE,
        "run_definition": {
            "sha256": sha256_file(output / "run_definition.json"),
            "nbytes": (output / "run_definition.json").stat().st_size,
        },
        "dataset_receipt": {
            "sha256": sha256_file(output / "dataset_receipt.json"),
            "nbytes": (output / "dataset_receipt.json").stat().st_size,
        },
        "selection_manifest": {
            "sha256": sha256_file(output / "selection_manifest.json"),
            "nbytes": (output / "selection_manifest.json").stat().st_size,
        },
        "runtime_setup": {
            "sha256": sha256_file(output / "runtime_setup.json"),
            "nbytes": (output / "runtime_setup.json").stat().st_size,
        },
        "token_plans": token_plan_sidecar,
        "raw_records": raw_sidecar,
        "summary": summary_sidecar,
        "checkpoint_manifest_sha256": canonical_sha256(checkpoint_rows),
        "checkpoints": checkpoint_rows,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if variant.curve_variant is not None:
        manifest.update(
            {
                "curve_variant": variant.curve_variant,
                "selected_source_layers": variant.selected_source_layers,
                "prerequisite_acceptance": dict(
                    variant.prerequisite_acceptance or {}
                ),
            }
        )
    hashed_json(output / "manifest.json", manifest)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configuration = load_coqa_config(args.config)
    output = assert_evaluation_output(args.output_dir or _default_output(args.prepare_only))
    if args.prepare_only:
        selected, plan_or_receipt = prepare_coqa(configuration, output)
        print(json.dumps(plan_or_receipt, ensure_ascii=False, indent=2, default=str))
        if selected is not None:
            print(f"prepared {len(selected)} deterministic CoQA conversations")
        return 0
    return run_evaluation(configuration, output)


__all__ = [
    "CoQAConversation",
    "CoQARunVariant",
    "CoQATurnTokenPlan",
    "DEFAULT_CONFIG",
    "DEFAULT_RUN_VARIANT",
    "EXPECTED_CONVERSATIONS",
    "HEADLINE_DEPTHS",
    "MODES",
    "PAPER_DOMAINS",
    "PersistentGenerationResult",
    "SelectedConversation",
    "build_incremental_token_plan",
    "build_prompt",
    "coqa_word_f1",
    "conversation_from_row",
    "dataset_download_command",
    "format_relaxed_prediction",
    "load_coqa_config",
    "main",
    "normalize_coqa_answer",
    "pair_word_f1",
    "prepare_coqa",
    "select_conversations",
    "summarize_records",
    "validate_completed_output",
    "verify_dataset_file",
    "_validate_mapper_for_variant",
    "_validate_run_variant",
    "_variant_run_fields",
]
