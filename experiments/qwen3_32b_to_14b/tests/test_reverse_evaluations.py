from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

import reverse_kv.coqa_eval as coqa_eval
import reverse_kv.eval_adapter as eval_adapter
import reverse_kv.hellaswag_eval as hellaswag_eval
import reverse_kv.reverse_latency as reverse_latency
from reverse_kv.coqa_eval import (
    CoQAConversation,
    EXPECTED_CONVERSATIONS,
    HEADLINE_DEPTHS,
    MODES,
    PAPER_DOMAINS,
    _append_persistent_parent,
    _complete_record,
    _generate_from_persistent_parent,
    _prefill_persistent_parent,
    build_incremental_token_plan,
    build_prompt,
    coqa_word_f1,
    dataset_download_command,
    format_relaxed_prediction,
    load_conversations,
    load_coqa_config,
    normalize_coqa_answer,
    pair_word_f1,
    select_conversations,
    summarize_records,
)
from reverse_kv.eval_common import hashed_json, model_artifact_identity
from reverse_kv.protocol import atomic_write_json, canonical_sha256
from reverse_kv.hellaswag_eval import (
    ALLOWED_MODES,
    FORMAL_RUNTIME_TOPOLOGY,
    SPLIT_SCOPE,
    TASK,
    parse_args as parse_hellaswag_args,
    sealed_shared_argv,
    validate_hellaswag_result,
)
from reverse_kv.reverse_latency import (
    GLOBAL_GPU_LOCK,
    HOST_STAGING_OVERHEAD_BYTES,
    MAPPER_PARAMETER_BYTES,
    PAPER_LENGTHS,
    REPEATS,
    SOURCE_CACHE_DEVICES,
    SOURCE_CACHE_LAYER_COUNTS,
    SOURCE_CACHE_STAGING_POLICY,
    SOURCE_PREFILL_DEVICES,
    SOURCE_PREFILL_LAYER_COUNTS,
    TARGET_DEVICES,
    TARGET_LAYER_COUNTS,
    WARMUP,
    latency_package_versions,
    nvidia_driver_evidence,
    load_latency_config,
    source_cache_exact_bytes,
    staged_runtime_topology,
    summarize_checkpoints,
    verify_length_checkpoint,
    verify_staging_session,
)


def test_hellaswag_entry_is_hard_sealed() -> None:
    assert ALLOWED_MODES == ("target_standalone", "mapped")
    assert TASK == "hellaswag"
    assert SPLIT_SCOPE == "full_validation_10042_no_limit"
    assert FORMAL_RUNTIME_TOPOLOGY == {
        "source": {
            "devices": [0, 1, 2, 3, 4],
            "layer_counts": [12, 13, 14, 13, 12],
            "model": "/home/wby/models/Qwen3-32B",
            "explicit_exact_counts": True,
        },
        "target": {
            "devices": [5, 6, 7],
            "layer_counts": [13, 14, 13],
            "model": "/home/wby/models/Qwen3-14B",
            "explicit_exact_counts": True,
        },
    }
    argv = sealed_shared_argv(parse_hellaswag_args([]))
    assert argv == [
        "--modes",
        "target_standalone,mapped",
        "--tasks",
        "hellaswag",
        "--log-samples",
    ]
    assert "--limit" not in argv
    with pytest.raises(SystemExit):
        parse_hellaswag_args(["--modes", "source_standalone"])
    with pytest.raises(SystemExit):
        parse_hellaswag_args(["--tasks", "mmlu"])
    with pytest.raises(SystemExit):
        parse_hellaswag_args(["--limit", "10"])
    with pytest.raises(SystemExit):
        parse_hellaswag_args(["--attention-implementation", "eager"])
    with pytest.raises(SystemExit):
        parse_hellaswag_args(["--log-samples"])


def test_model_identity_hashes_every_shard_and_rejects_same_size_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "model"
    root.mkdir()
    atomic_write_json(root / "config.json", {"model_type": "qwen3"})
    atomic_write_json(root / "tokenizer.json", {"version": "1"})
    atomic_write_json(
        root / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 8},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
            },
        },
    )
    first_shard = root / "model-00001-of-00002.safetensors"
    second_shard = root / "model-00002-of-00002.safetensors"
    first_shard.write_bytes(b"AAAA")
    second_shard.write_bytes(b"BBBB")
    before = model_artifact_identity(root)
    assert before["weight_shard_contents_hashed"] is True
    assert all(len(row["sha256"]) == 64 for row in before["weight_shards"])
    before_size = first_shard.stat().st_size
    first_shard.write_bytes(b"ZZZZ")
    assert first_shard.stat().st_size == before_size
    after = model_artifact_identity(root)
    assert before["weight_shards"][0]["nbytes"] == after["weight_shards"][0]["nbytes"]
    assert before["weight_shards"][0]["sha256"] != after["weight_shards"][0]["sha256"]
    assert before["weight_shards_sha256"] != after["weight_shards_sha256"]


def test_hellaswag_and_coqa_use_the_strong_reverse_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    class FakeModel:
        pass

    def fake_load(model_path, *, role, **kwargs):
        observed.setdefault("loads", []).append(
            {
                "model_path": str(Path(model_path).resolve()),
                "role": role,
                **kwargs,
            }
        )
        return FakeModel()

    class FakeImplementation:
        load_qwen3 = staticmethod(fake_load)
        project_code_provenance = staticmethod(lambda root: {})

        @staticmethod
        def main(argv):
            observed["argv"] = argv
            observed["identity"] = FakeImplementation.model_artifact_identity
            source = FakeImplementation.load_qwen3(
                "/home/wby/models/Qwen3-32B",
                devices=(0, 1, 2, 3, 4),
                role="source",
            )
            target = FakeImplementation.load_qwen3(
                "/home/wby/models/Qwen3-14B",
                devices=(5, 6, 7),
                role="target",
            )
            observed["source_binding"] = source._reverse_semantic_role_binding
            observed["target_binding"] = target._reverse_semantic_role_binding
            return 0

    monkeypatch.setattr(eval_adapter, "_load_implementation", lambda: FakeImplementation)
    assert eval_adapter.run_evaluation(()) == 0
    assert observed["identity"] is model_artifact_identity
    assert coqa_eval.model_artifact_identity is model_artifact_identity
    assert observed["loads"] == [
        {
            "model_path": "/home/wby/models/Qwen3-32B",
            "role": "target",
            "devices": (0, 1, 2, 3, 4),
            "layer_counts": (12, 13, 14, 13, 12),
        },
        {
            "model_path": "/home/wby/models/Qwen3-14B",
            "role": "source",
            "devices": (5, 6, 7),
            "layer_counts": (13, 14, 13),
        },
    ]
    assert observed["source_binding"] == {
        "semantic_role": "source",
        "capture_native_forward_role": "target",
        "devices": [0, 1, 2, 3, 4],
        "layer_counts": [12, 13, 14, 13, 12],
        "explicit_exact_counts": True,
    }
    assert observed["target_binding"] == {
        "semantic_role": "target",
        "capture_native_forward_role": "source",
        "devices": [5, 6, 7],
        "layer_counts": [13, 14, 13],
        "explicit_exact_counts": True,
    }


def test_hellaswag_minimal_forged_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "hellaswag.json"
    hashed_json(
        path,
        {
            "schema_version": 1,
            "status": "complete",
            "request_fingerprint": "f" * 64,
            "mode": "mapped",
            "task": "hellaswag",
            "primary_metric": "acc_norm",
            "primary_metric_value": 1.0,
        },
    )
    with pytest.raises(ValueError, match="lm-eval payload"):
        validate_hellaswag_result(
            path, fingerprint="f" * 64, expected_mode="mapped"
        )


def test_hellaswag_primary_metric_is_recomputed_from_raw_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(hellaswag_eval, "EXPECTED_SAMPLES", 2)
    samples = [
        {
            "doc_id": 0,
            "acc_norm": 1.0,
            "filter": "none",
            "metrics": ["acc", "acc_norm"],
        },
        {
            "doc_id": 1,
            "acc_norm": 0.0,
            "filter": "none",
            "metrics": ["acc", "acc_norm"],
        },
    ]
    value = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": "f" * 64,
        "mode": "mapped",
        "task": "hellaswag",
        "primary_metric": "acc_norm",
        "primary_metric_value": 0.5,
        "result": {
            "n-samples": {"hellaswag": {"effective": 2, "original": 2}},
            "n-shot": {"hellaswag": 0},
            "config": {"limit": None, "batch_size": 1, "apply_chat_template": False},
            "configs": {
                "hellaswag": {
                    "validation_split": "validation",
                    "num_fewshot": 0,
                    "output_type": "multiple_choice",
                    "metric_list": [
                        {
                            "metric": "acc_norm",
                            "aggregation": "mean",
                            "higher_is_better": True,
                        }
                    ],
                }
            },
            "results": {"hellaswag": {"sample_len": 2, "acc_norm,none": 0.5}},
            "samples": {"hellaswag": samples},
        },
    }
    path = tmp_path / "hellaswag.json"
    hashed_json(path, value)
    validated = validate_hellaswag_result(
        path, fingerprint="f" * 64, expected_mode="mapped"
    )
    assert validated["acc_norm"] == 0.5
    value["primary_metric_value"] = 0.75
    hashed_json(path, value)
    with pytest.raises(ValueError, match="primary acc_norm disagrees"):
        validate_hellaswag_result(
            path, fingerprint="f" * 64, expected_mode="mapped"
        )


def test_hellaswag_idle_gate_is_semantically_verified(tmp_path: Path) -> None:
    memory = [
        {
            "physical_index": index,
            "uuid": f"GPU-{index}",
            "memory_used_mib": 18,
            "memory_total_mib": 24_564,
        }
        for index in range(8)
    ]
    gate = {
        "status": "passed",
        "required_consecutive_samples": 12,
        "memory_limit_mib": 64,
        "final_consecutive_samples": 12,
        "cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "samples": [
            {
                "sample_index": index,
                "passed": True,
                "memory": memory,
                "compute_processes": [],
            }
            for index in range(12)
        ],
    }
    hashed_json(tmp_path / hellaswag_eval.GPU_IDLE_GATE, gate)
    identity = hellaswag_eval._validate_gpu_idle_gate(tmp_path)
    assert identity["sha256"]
    gate["samples"][-1]["compute_processes"] = [{"pid": 42}]
    hashed_json(tmp_path / hellaswag_eval.GPU_IDLE_GATE, gate)
    with pytest.raises(ValueError, match="observed a compute process"):
        hellaswag_eval._validate_gpu_idle_gate(tmp_path)


def test_coqa_config_is_pinned_and_emits_exact_modelscope_command() -> None:
    configuration = load_coqa_config()
    dataset = configuration["dataset"]
    assert dataset["repository"] == "EleutherAI/coqa"
    assert dataset["revision"] == "5153eac35fe087e44d9ce99ebb2d94c3dbc81583"
    assert dataset["expected_sha256"] == (
        "5ab4d352067c30ef4925efb0dfb2abeeed4cdfb236304aa7ee9df62ea781281f"
    )
    assert dataset["expected_nbytes"] == 2_221_232
    assert tuple(configuration["selection"]["headline_handoff_depths"]) == (
        HEADLINE_DEPTHS
    )
    assert configuration["runtime"]["source_devices"] == [0, 1, 2, 3, 4]
    assert configuration["runtime"]["source_layer_counts"] == [12, 13, 14, 13, 12]
    assert configuration["runtime"]["target_devices"] == [5, 6, 7]
    assert configuration["runtime"]["target_layer_counts"] == [13, 14, 13]
    command = dataset_download_command(configuration)
    assert command == [
        "/home/wby/.venvs/modelscope/bin/ms-hub",
        "download",
        "EleutherAI/coqa",
        "coqa_validation.parquet",
        "--repo-type",
        "dataset",
        "--revision",
        dataset["revision"],
        "--local-dir",
        str(dataset["file"].parent),
        "--max-workers",
        "8",
    ]
    metric = configuration["metric"]
    assert metric["primary_evaluator"] == "stanford_coqa_evaluate_v1.0"
    assert metric["reference_deduplication"] == (
        "none_preserve_official_annotator_multiplicity"
    )
    assert metric["lm_eval_reference_difference"] == (
        "vendored_doc_to_target_case_insensitive_dedup_is_not_used"
    )


def test_official_coqa_normalized_word_f1() -> None:
    assert normalize_coqa_answer("The, Cat!") == "cat"
    assert pair_word_f1("quick brown fox", "the quick, brown fox!") == 1.0
    assert pair_word_f1("red red cat", "red cat") == pytest.approx(0.8)
    assert pair_word_f1("", "") == 1.0
    assert pair_word_f1("", "cat") == 0.0
    # Official CoQA uses leave-one-reference-out averaging, not one global max.
    assert coqa_word_f1("NYC", ["Boston", "nyc!"]) == 0.5
    assert coqa_word_f1("NYC", ["Boston", "nyc!", "NYC"]) == 1.0
    assert format_relaxed_prediction("Final Answer: Alice\nBecause...") == "Alice"


def _conversation(domain: str, index: int, answer: str = "gold") -> CoQAConversation:
    questions = tuple(f"question {turn}" for turn in range(12))
    answers = tuple(answer for _ in questions)
    references = tuple((answer, f"{answer}.") for _ in questions)
    return CoQAConversation(
        row_index=index,
        conversation_id=f"{domain}-{index:03d}",
        domain=domain,
        story=f"story {domain} {index}",
        questions=questions,
        primary_answers=answers,
        references=references,
    )


def test_coqa_selection_is_balanced_deterministic_and_gold_independent() -> None:
    configuration = load_coqa_config()
    conversations = [
        _conversation(domain, index)
        for domain in PAPER_DOMAINS
        for index in range(25)
    ]
    selected, manifest = select_conversations(conversations, configuration)
    assert len(selected) == EXPECTED_CONVERSATIONS
    assert manifest["selection_uses_answers_or_gold"] is False
    assert {
        domain: sum(value.conversation.domain == domain for value in selected)
        for domain in PAPER_DOMAINS
    } == {domain: 20 for domain in PAPER_DOMAINS}
    altered = [
        replace(
            value,
            primary_answers=tuple("different gold" for _ in value.primary_answers),
            references=tuple(("different gold",) for _ in value.references),
        )
        for value in conversations
    ]
    selected_altered, manifest_altered = select_conversations(altered, configuration)
    assert [value.conversation.conversation_id for value in selected] == [
        value.conversation.conversation_id for value in selected_altered
    ]
    assert manifest["rows_sha256"] == manifest_altered["rows_sha256"]


def test_coqa_prompt_uses_only_prior_gold_answers() -> None:
    conversation = _conversation("news", 0, answer="KNOWN_GOLD")
    prompt = build_prompt(conversation, 3)
    assert prompt.count("KNOWN_GOLD") == 2
    assert "Q: question 2\n\nA:" in prompt
    assert prompt.endswith("A:")


class _CharacterTokenizer:
    eos_token_id = 255

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    @staticmethod
    def decode(token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token) for token in token_ids)


def test_coqa_incremental_plan_exactly_advances_one_persistent_parent() -> None:
    conversation = _conversation("news", 0, answer="KNOWN_GOLD")
    plans = build_incremental_token_plan(conversation, _CharacterTokenizer())
    assert len(plans) == 10
    assert tuple(plan.turn for plan in plans if plan.turn in HEADLINE_DEPTHS) == (
        HEADLINE_DEPTHS
    )
    for plan in plans:
        assert plan.parent_token_ids + (plan.boundary_token_id,) == plan.prompt_token_ids
        assert plan.prompt == build_prompt(conversation, plan.turn)
    for current, following in zip(plans[:-1], plans[1:], strict=True):
        assert current.parent_token_ids + current.append_after_turn == (
            following.parent_token_ids
        )


def test_coqa_persistent_cache_helpers_run_on_cpu_without_full_reprefill() -> None:
    from kvcache_transfer.cache_adapter import cache_sequence_length
    from transformers import Qwen3Config, Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=256,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=64,
        )
    ).eval()
    parent = _prefill_persistent_parent(model, [1, 2, 3])
    assert cache_sequence_length(parent) == 3
    parent = _append_persistent_parent(model, parent, [4, 5])
    assert cache_sequence_length(parent) == 5
    generated = _generate_from_persistent_parent(
        target_model=model,
        tokenizer=_CharacterTokenizer(),
        parent_cache=parent,
        boundary_token_id=6,
        context_tokens=6,
        max_tokens=2,
        stop_strings=(),
    )
    assert generated.generated_token_count <= 2
    assert cache_sequence_length(parent) == 5


def _valid_coqa_checkpoint() -> tuple[dict[str, object], dict[str, object]]:
    prompt = "Story.\n\nQ: Color?\n\nA:"
    references = ["blue"]
    parent_sha = "a" * 64
    expected: dict[str, object] = {
        "request_fingerprint": "f" * 64,
        "mode": "mapped",
        "selection_index": 0,
        "conversation_id": "conversation-0",
        "domain": "news",
        "handoff_depth": 1,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": "b" * 64,
        "references_sha256": canonical_sha256(references),
        "rank_within_domain": 1,
        "rank_sha256": "c" * 64,
        "handoff_executed": True,
        "history_policy": "gold_primary_answer",
        "prompt": prompt,
        "prompt_token_count": 24,
        "cache_execution": {
            "persistent_parent_model": "Qwen3-32B",
            "initial_full_history_prefill_count": 1,
            "initial_parent_token_count": 23,
            "incremental_append_calls_before_depth": 0,
            "incremental_appended_tokens_before_depth": 0,
            "parent_token_count_before_handoff": 23,
            "parent_token_ids_sha256": parent_sha,
            "full_history_reprefill_at_depth": False,
            "decode_used_deep_cloned_disposable_branch": True,
            "persistent_parent_unchanged_by_decode": True,
            "source_to_target_handoffs_for_record": 1,
        },
        "mapped_prefix_token_count": 23,
    }
    token_ids = [ord(value) for value in "blue"]
    record: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        **copy.deepcopy(expected),
        "references": references,
        "prediction": "blue",
        "relaxed_prediction": "blue",
        "official_word_f1": 1.0,
        "official_word_f1_percent": 100.0,
        "format_relaxed_word_f1": 1.0,
        "format_relaxed_word_f1_percent": 100.0,
        "generation": {
            "generated_token_ids": token_ids,
            "generated_token_count": len(token_ids),
            "stop_reason": "eos",
            "cap_hit": False,
            "requested_max_tokens": 128,
            "original_context_tokens": 24,
            "used_context_tokens": 24,
            "truncation": "error",
        },
    }
    return expected, record


def test_coqa_resume_accepts_a_semantically_valid_checkpoint(tmp_path: Path) -> None:
    expected, record = _valid_coqa_checkpoint()
    path = tmp_path / "record.json"
    hashed_json(path, record)
    assert _complete_record(
        path,
        expected,
        tokenizer=_CharacterTokenizer(),
        stop_strings=("\nQ:",),
        max_tokens=128,
    ) == record


@pytest.mark.parametrize("tamper", ("f1", "generation", "cache"))
def test_coqa_resume_rejects_coordinated_checkpoint_tampering(
    tmp_path: Path, tamper: str
) -> None:
    expected, record = _valid_coqa_checkpoint()
    path = tmp_path / "record.json"
    if tamper == "f1":
        record["official_word_f1"] = 0.0
        record["official_word_f1_percent"] = 0.0
    elif tamper == "generation":
        record["generation"]["generated_token_ids"] = [  # type: ignore[index]
            ord(value) for value in "red"
        ]
        record["generation"]["generated_token_count"] = 3  # type: ignore[index]
    else:
        record["cache_execution"]["parent_token_count_before_handoff"] = 999  # type: ignore[index]
    # Re-commit the sidecar too: the semantic gate must reject more than a
    # stale-checksum accident.
    hashed_json(path, record)
    with pytest.raises(ValueError, match="invalid CoQA checkpoint"):
        _complete_record(
            path,
            expected,
            tokenizer=_CharacterTokenizer(),
            stop_strings=("\nQ:",),
            max_tokens=128,
        )


def test_coqa_official_domain_aliases_are_normalized() -> None:
    from reverse_kv.coqa_eval import conversation_from_row

    row = {
        "id": "story-1",
        "source": "cnn",
        "story": "A story.",
        "questions": {"input_text": ["Who?"], "turn_id": [1]},
        "answers": {
            "input_text": ["Alice"],
            "answer_start": [0],
            "answer_end": [5],
        },
        "additional_answers": {
            "0": {"input_text": ["Alice"], "answer_start": [0], "answer_end": [5]}
        },
    }
    conversation = conversation_from_row(row, 0)
    assert conversation.domain == "news"
    assert conversation.raw_domain == "cnn"
    assert conversation.references == (("Alice", "Alice"),)


def test_coqa_arrow_nested_schema_is_readable(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    questions = [f"Question {index}?" for index in range(10)]
    answers = [f"Answer {index}" for index in range(10)]
    row = {
        "id": "arrow-story",
        "source": "mctest",
        "story": "A nested Arrow story.",
        "questions": {"input_text": questions, "turn_id": list(range(1, 11))},
        "answers": {
            "input_text": answers,
            "answer_start": list(range(10)),
            "answer_end": list(range(1, 11)),
        },
        "additional_answers": {
            "0": {
                "input_text": [f"Alternative {index}" for index in range(10)],
                "answer_start": list(range(10)),
                "answer_end": list(range(1, 11)),
            }
        },
    }
    path = tmp_path / "coqa.parquet"
    pq.write_table(pa.Table.from_pylist([row]), path)
    conversations, schema = load_conversations(path)
    assert schema["rows"] == 1
    assert conversations[0].domain == "children_stories"
    assert conversations[0].turn_count == 10
    assert conversations[0].references[9] == ("Answer 9", "Alternative 9")


def _dummy_coqa_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for mode in MODES:
        for domain_index, domain in enumerate(PAPER_DOMAINS):
            for conversation in range(20):
                for depth in HEADLINE_DEPTHS:
                    score = 0.8 if mode == "target_standalone" else 0.75
                    references = ["red blue green white"]
                    prediction = (
                        "red blue green white black orange"
                        if mode == "target_standalone"
                        else "red blue green black"
                    )
                    rows.append(
                        {
                            "mode": mode,
                            "domain": domain,
                            "conversation_id": f"{domain}-{conversation}",
                            "selection_index": domain_index * 20 + conversation,
                            "handoff_depth": depth,
                            "prompt_sha256": f"prompt-{domain}-{conversation}-{depth}",
                            "prompt_token_ids_sha256": (
                                f"tokens-{domain}-{conversation}-{depth}"
                            ),
                            "prompt_token_count": 100 + depth,
                            "references_sha256": canonical_sha256(references),
                            "history_policy": "gold_primary_answer",
                            "references": references,
                            "prediction": prediction,
                            "relaxed_prediction": prediction,
                            "official_word_f1": score,
                            "official_word_f1_percent": score * 100.0,
                            "format_relaxed_word_f1": score,
                            "format_relaxed_word_f1_percent": score * 100.0,
                            "cache_execution": {
                                "initial_full_history_prefill_count": 1,
                                "initial_parent_token_count": 99,
                                "incremental_append_calls_before_depth": depth - 1,
                                "incremental_appended_tokens_before_depth": depth * 3,
                                "parent_token_count_before_handoff": 99 + depth,
                                "parent_token_ids_sha256": (
                                    f"parent-{domain}-{conversation}-{depth}"
                                ),
                                "full_history_reprefill_at_depth": False,
                            },
                            "generation": {"cap_hit": False},
                        }
                    )
    return rows


def test_coqa_summary_is_paired_by_handoff_depth() -> None:
    summary = summarize_records(_dummy_coqa_records())
    assert summary["status"] == "complete"
    assert summary["overall_drift_target_minus_mapped"] == pytest.approx(0.05)
    assert summary["overall_retention_mapped_over_target"] == pytest.approx(0.9375)
    for depth in HEADLINE_DEPTHS:
        row = summary["drift_by_handoff_depth"][str(depth)]
        assert row["drift_target_minus_mapped"] == pytest.approx(0.05)


def test_coqa_summary_rejects_unpaired_prompt_identity() -> None:
    records = _dummy_coqa_records()
    records[-1]["prompt_token_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="prompt_token_ids_sha256"):
        summarize_records(records)


def test_coqa_summary_rejects_target_mapped_parent_history_mismatch() -> None:
    records = _dummy_coqa_records()
    records[-1]["cache_execution"]["parent_token_ids_sha256"] = "different"  # type: ignore[index]
    with pytest.raises(ValueError, match="histories disagree"):
        summarize_records(records)


def test_coqa_summary_recomputes_f1_instead_of_trusting_stored_scalars() -> None:
    records = _dummy_coqa_records()
    records[0]["official_word_f1"] = 0.0
    records[0]["official_word_f1_percent"] = 0.0
    with pytest.raises(ValueError, match="differs from recomputed raw-response F1"):
        summarize_records(records)


def test_reverse_latency_protocol_is_exact() -> None:
    configuration = load_latency_config()
    assert PAPER_LENGTHS == (
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
    )
    assert WARMUP == 50
    assert REPEATS == 30
    assert configuration["protocol"]["source_prefill_in_timed_region"] is False
    assert configuration["protocol"]["target_reprefill_scope"].endswith(
        "excluding_lm_head"
    )
    assert configuration["protocol"]["mapper_execution"] == "eager"
    assert configuration["protocol"]["endpoint_memory_smoke_length"] == 32768
    assert configuration["protocol"]["minimum_post_operation_headroom_mib"] == 2048
    assert configuration["protocol"]["source_cache_lifecycle"] == (
        SOURCE_CACHE_STAGING_POLICY
    )
    assert configuration["protocol"]["cpu_staging_in_timed_region"] is False
    assert configuration["protocol"]["gpu_restage_in_timed_region"] is False
    assert configuration["protocol"]["mapper_timer_start"] == (
        "preexisting_gpu_resident_source_kv"
    )
    assert configuration["runtime"]["source_prefill_devices"] == list(
        SOURCE_PREFILL_DEVICES
    )
    assert configuration["runtime"]["source_prefill_layer_counts"] == list(
        SOURCE_PREFILL_LAYER_COUNTS
    )
    assert configuration["runtime"]["source_cache_devices"] == list(
        SOURCE_CACHE_DEVICES
    )
    assert configuration["runtime"]["source_cache_layer_counts"] == list(
        SOURCE_CACHE_LAYER_COUNTS
    )
    assert configuration["runtime"]["target_devices"] == list(TARGET_DEVICES)
    assert configuration["runtime"]["target_layer_counts"] == list(
        TARGET_LAYER_COUNTS
    )
    topology = staged_runtime_topology(configuration)
    assert topology["source_model_live_with_target_or_mapper"] is False
    assert topology["timed_cache_target_disjoint"] is True
    assert GLOBAL_GPU_LOCK == reverse_latency.REPOSITORY_ROOT / "runs/gpu8_experiment.lock"
    packages = latency_package_versions()
    assert set(packages) == {
        "transformers",
        "accelerate",
        "flash_attention",
        "safetensors",
    }
    assert all(value["version"] for value in packages.values())


def test_nvidia_driver_version_evidence_is_fail_closed_and_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = "\n".join(
        f"{index}, GPU-{index}, NVIDIA GeForce RTX 4090, 535.183.01"
        for index in range(8)
    )
    monkeypatch.setattr(
        reverse_latency,
        "command_evidence",
        lambda command: {"command": command, "returncode": 0, "stdout": rows, "stderr": ""},
    )
    evidence = nvidia_driver_evidence()
    assert evidence["version"] == "535.183.01"
    assert len(evidence["rows"]) == 8
    bad_rows = rows.replace("7, GPU-7", "6, GPU-7")
    monkeypatch.setattr(
        reverse_latency,
        "command_evidence",
        lambda command: {
            "command": command,
            "returncode": 0,
            "stdout": bad_rows,
            "stderr": "",
        },
    )
    with pytest.raises(RuntimeError, match="physical GPUs 0..7"):
        nvidia_driver_evidence()


def _latency_metric(value: float) -> dict[str, object]:
    return {
        "unit": "ms",
        "count": REPEATS,
        "mean": value,
        "median": value,
        "p10": value,
        "p50": value,
        "p90": value,
        "std_population": 0.0,
        "min": value,
        "max": value,
        "warmup": WARMUP,
        "timed_repeats": REPEATS,
        "samples": [value] * REPEATS,
        "synchronize_each_iteration": True,
        "timing_backends": ["host_perf_counter_with_multi_cuda_synchronize"],
    }


def _latency_checkpoint(length: int, fingerprint: str = "f" * 64) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "prefix_length": length,
        "batch_size": 1,
        "source_prefill_timed": False,
        "source_cache_existed_before_each_timed_metric": True,
        "source_model_reference_passed_to_timed_mapper": False,
        "source_model_live_during_timed_cells": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "timed_mapper_strict_validation": False,
        "untimed_strict_mapper_preflight": {
            "status": "passed",
            "target_cache_layers": 40,
            "target_cache_sequence_length": length,
            "strict_validation": True,
        },
        "target_reprefill": {
            **_latency_metric(4.0),
            "measurement_scope": "target_transformer_reprefill",
            "lm_head_included": False,
            "kv_cache_output": True,
            "input_length": length,
            "batch_size": 1,
            "inference_mode_enabled": True,
            "autograd_graph_disabled": True,
        },
        "mapper_handoff_end_to_end": {
            **_latency_metric(2.0),
            "measurement_scope": "source_cache_to_mapped_target_cache",
            "source_prefill_included": False,
            "mapper_weights_resident_on_target_devices": True,
            "includes_source_cache_device_handoff": True,
            "includes_inverse_and_target_rope": True,
            "includes_cache_build": True,
            "inference_mode_enabled": True,
            "autograd_graph_disabled": True,
        },
    }


def _expanded_devices(devices: tuple[int, ...], counts: tuple[int, ...]) -> list[int]:
    return [
        device
        for device, count in zip(devices, counts, strict=True)
        for _ in range(count)
    ]


def _cache_structure_rows(length: int, devices: list[str]) -> list[dict[str, object]]:
    tensor_bytes = 8 * length * 128 * 2
    return [
        {
            "layer": layer,
            "device": device,
            "key_shape": [1, 8, length, 128],
            "value_shape": [1, 8, length, 128],
            "key_dtype": "torch.bfloat16",
            "value_dtype": "torch.bfloat16",
            "key_nbytes": tensor_bytes,
            "value_nbytes": tensor_bytes,
        }
        for layer, device in enumerate(devices)
    ]


def _source_cache_evidence(length: int, devices: list[str]) -> dict[str, object]:
    layer_bytes = 2 * 8 * length * 128 * 2
    return {
        "status": "passed",
        "source_model": "Qwen3-32B",
        "source_prefill_timed": False,
        "sequence_length": length,
        "layers": 64,
        "total_tensor_bytes": source_cache_exact_bytes(length),
        "per_layer": [
            {
                "layer": layer,
                "key_device": device,
                "value_device": device,
                "shape": [1, 8, length, 128],
                "dtype": "torch.bfloat16",
                "bytes": layer_bytes,
            }
            for layer, device in enumerate(devices)
        ],
    }


def _zero_allocator(devices: tuple[int, ...]) -> dict[str, object]:
    return {
        str(device): {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "driver_free_bytes": 20_000_000_000,
            "driver_total_bytes": 24_000_000_000,
        }
        for device in devices
    }


def _sharding(
    devices: tuple[int, ...],
    counts: tuple[int, ...],
    *,
    role: str,
) -> dict[str, object]:
    ranges = []
    start = 0
    for device, count in zip(devices, counts, strict=True):
        ranges.append(
            {
                "device": device,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "count": count,
            }
        )
        start += count
    return {
        "status": "passed",
        "role": role,
        "mode": "explicit_exact_counts",
        "devices": list(devices),
        "declared_layer_counts": list(counts),
        "expected_layer_counts": list(counts),
        "actual_layer_counts": list(counts),
        "total_layers": sum(counts),
        "layer_devices": _expanded_devices(devices, counts),
        "contiguous_ranges": ranges,
        "actual_parameters_inspected_for_every_layer": True,
        "parameter_tensors_inspected": sum(counts) * 10,
        "auxiliary_parameter_tensors_inspected": 3,
        "auxiliary_parameters_inspected_for_every_module": True,
    }


def _commit_staged_latency_checkpoint(
    output: Path,
    *,
    length: int = 64,
    fingerprint: str = "f" * 64,
) -> tuple[Path, dict[str, object], Path]:
    raw_hash = hashlib.sha256(f"raw-cache-{length}".encode()).hexdigest()
    prefix = {
        "shape": [1, length],
        "dtype": "torch.int64",
        "nbytes": 8 * length,
        "sha256": hashlib.sha256(f"prefix-{length}".encode()).hexdigest(),
    }
    prefill_devices = [
        f"cuda:{device}"
        for device in _expanded_devices(
            SOURCE_PREFILL_DEVICES, SOURCE_PREFILL_LAYER_COUNTS
        )
    ]
    cache_devices = [
        f"cuda:{device}"
        for device in _expanded_devices(SOURCE_CACHE_DEVICES, SOURCE_CACHE_LAYER_COUNTS)
    ]
    preparation = {
        "status": "passed",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "prefix_length": length,
        "deterministic_prefix": prefix,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "copy_semantics": "blocking_detach_to_cpu_bf16_exact_clone",
        "full_raw_bit_sha256_computed": True,
        "source_gpu_raw_bit_sha256": raw_hash,
        "cpu_staged_raw_bit_sha256": raw_hash,
        "gpu_to_cpu_raw_bit_hash_matches": True,
        "source_prefill_layer_devices": prefill_devices,
        "source_gpu_cache_structure": _cache_structure_rows(
            length, prefill_devices
        ),
        "cpu_cache_structure": _cache_structure_rows(length, ["cpu"] * 64),
        "cache_total_bytes": source_cache_exact_bytes(length),
        "source_prefill_untimed_diagnostic_ms": 10.0,
    }
    source_weight_bytes = 61_000_000_000
    target_weight_bytes = 28_000_000_000
    required = (
        source_weight_bytes
        + target_weight_bytes
        + MAPPER_PARAMETER_BYTES
        + source_cache_exact_bytes(length)
        + HOST_STAGING_OVERHEAD_BYTES
    )
    source_release = {
        "status": "passed",
        "source_model_live": False,
        "source_model_weakref_is_none": True,
        "all_source_gpu_cache_references_deleted": True,
        "target_model_loaded_at_proof_time": False,
        "source_prefill_devices": list(SOURCE_PREFILL_DEVICES),
        "all_visible_devices": list(range(8)),
        "allocator_after_release_before_target_load": _zero_allocator(
            tuple(range(8))
        ),
        "allocator_after_release_covers_all_visible_devices": True,
        "all_prefill_allocated_and_reserved_bytes_zero": True,
        "all_visible_allocated_and_reserved_bytes_zero": True,
    }
    session_id = "1" * 32
    session_path = (
        output
        / "staging_sessions"
        / f"source_cache_staging_{session_id}.json"
    )
    session = {
        "schema_version": 1,
        "status": "source_released_before_target_load",
        "request_fingerprint": fingerprint,
        "session_id": session_id,
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "runtime_topology": reverse_latency._frozen_topology(),
        "runtime_topology_sha256": canonical_sha256(
            reverse_latency._frozen_topology()
        ),
        "host_memory_staging_gate": {
            "status": "passed",
            "scope": "before_source_model_load",
            "mem_available_bytes": required + 1,
            "required_available_bytes": required,
            "source_weight_shard_bytes": source_weight_bytes,
            "target_weight_shard_bytes": target_weight_bytes,
            "mapper_parameter_bytes": MAPPER_PARAMETER_BYTES,
            "staged_cache_bytes": source_cache_exact_bytes(length),
            "staged_lengths": [length],
            "conservative_overhead_bytes": HOST_STAGING_OVERHEAD_BYTES,
        },
        "prepared_lengths": [length],
        "source_vocabulary_size": 151_936,
        "source_model_artifact_sha256": "2" * 64,
        "target_model_artifact_sha256": "3" * 64,
        "source_model_instance_identity": {
            "model_path": "/home/wby/models/Qwen3-32B"
        },
        "source_model_prefill_sharding": _sharding(
            SOURCE_PREFILL_DEVICES,
            SOURCE_PREFILL_LAYER_COUNTS,
            role="target",
        ),
        "preparations": {str(length): preparation},
        "cpu_staging_total_bytes": source_cache_exact_bytes(length),
        "source_model_release": source_release,
        "target_model_loaded_when_receipt_written": False,
        "source_model_live_when_receipt_written": False,
    }
    session_sidecar = hashed_json(session_path, session)
    restage_release = {
        "status": "passed",
        "restaged_cache_weakref_is_none": True,
        "source_cache_devices": list(SOURCE_CACHE_DEVICES),
        "allocated_and_reserved_bytes_zero": True,
        "allocator_after_release": _zero_allocator(SOURCE_CACHE_DEVICES),
    }
    lifecycle = {
        "status": "complete",
        "policy": SOURCE_CACHE_STAGING_POLICY,
        "session_receipt": str(session_path.relative_to(output)),
        "session_receipt_sha256": session_sidecar["sha256"],
        "preparation_sha256": canonical_sha256(preparation),
        "source_model_live_during_timed_cells": False,
        "source_prefill_timed": False,
        "cpu_staging_timed": False,
        "gpu_restage_timed": False,
        "mapper_timer_starts_from_preexisting_gpu_source_cache": True,
        "mapper_timer_started": True,
        "deterministic_prefix": prefix,
        "cpu_staged_raw_bit_sha256": raw_hash,
        "cache_total_bytes": source_cache_exact_bytes(length),
        "restage": {
            "status": "passed",
            "policy": SOURCE_CACHE_STAGING_POLICY,
            "prefix_length": length,
            "untimed": True,
            "source_model_live": False,
            "source_cache_gpu_resident_before_mapper_timer": True,
            "copy_semantics": "blocking_cpu_to_cuda_bf16_no_value_transform",
            "raw_bit_sha256": raw_hash,
            "raw_bit_hash_matches_cpu_staging": True,
            "layer_devices": cache_devices,
            "layer_counts": list(SOURCE_CACHE_LAYER_COUNTS),
            "cache_structure": _cache_structure_rows(length, cache_devices),
            "cache_total_bytes": source_cache_exact_bytes(length),
        },
        "restaged_cache_release": restage_release,
    }
    checkpoint = _latency_checkpoint(length, fingerprint)
    checkpoint.update(
        {
            "deterministic_prefix": prefix,
            "source_cache": _source_cache_evidence(length, cache_devices),
            "source_cache_lifecycle": lifecycle,
        }
    )
    path = output / "lengths" / f"length_{length:05d}.json"
    hashed_json(path, checkpoint)
    return path, checkpoint, session_path


def test_latency_checkpoints_keep_raw_samples_and_scope(tmp_path: Path) -> None:
    path, _, _ = _commit_staged_latency_checkpoint(tmp_path)
    verified = verify_length_checkpoint(path, fingerprint="f" * 64, length=64)
    assert verified is not None
    assert len(verified["target_reprefill"]["samples"]) == 30
    checkpoints = [_latency_checkpoint(length) for length in PAPER_LENGTHS]
    summary = summarize_checkpoints(checkpoints)
    assert len(summary["rows"]) == 10
    assert all(row["speedup_reprefill_over_mapper"] == 2.0 for row in summary["rows"])
    assert all(len(row["mapper_raw_samples_ms"]) == 30 for row in summary["rows"])


def test_latency_checkpoint_rejects_nonpositive_raw_sample_even_with_valid_sha(
    tmp_path: Path,
) -> None:
    path, value, _ = _commit_staged_latency_checkpoint(tmp_path)
    value["mapper_handoff_end_to_end"]["samples"][0] = -9.0  # type: ignore[index]
    hashed_json(path, value)
    with pytest.raises(ValueError, match="finite and positive"):
        verify_length_checkpoint(path, fingerprint="f" * 64, length=64)


def test_latency_checkpoint_rejects_median_not_recomputed_from_samples(
    tmp_path: Path,
) -> None:
    path, value, _ = _commit_staged_latency_checkpoint(tmp_path)
    value["target_reprefill"]["median"] = 123_456.0  # type: ignore[index]
    hashed_json(path, value)
    with pytest.raises(ValueError, match="median disagrees with raw samples"):
        verify_length_checkpoint(path, fingerprint="f" * 64, length=64)


def test_staged_latency_exact_cache_bytes_cover_all_paper_lengths() -> None:
    assert source_cache_exact_bytes(32768) == 8 * 1024**3
    assert sum(source_cache_exact_bytes(length) for length in PAPER_LENGTHS) == (
        17_163_091_968
    )


def test_latency_checkpoint_binds_cpu_raw_bits_and_source_release_receipt(
    tmp_path: Path,
) -> None:
    path, _, session_path = _commit_staged_latency_checkpoint(tmp_path)
    verified = verify_staging_session(
        session_path, fingerprint="f" * 64, required_length=64
    )
    assert verified["source_model_release"][
        "all_prefill_allocated_and_reserved_bytes_zero"
    ] is True
    assert set(
        verified["source_model_release"][
            "allocator_after_release_before_target_load"
        ]
    ) == {str(device) for device in range(8)}
    assert verified["source_model_release"][
        "target_model_loaded_at_proof_time"
    ] is False
    session = reverse_latency.read_json(session_path)
    session["preparations"]["64"]["source_gpu_raw_bit_sha256"] = "9" * 64
    session["preparations"]["64"]["cpu_staged_raw_bit_sha256"] = "9" * 64
    hashed_json(session_path, session)
    with pytest.raises(ValueError, match="staging receipt SHA changed"):
        verify_length_checkpoint(path, fingerprint="f" * 64, length=64)


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("source_model_live_during_timed_cells", True),
        ("cpu_staging_timed", True),
        ("gpu_restage_timed", True),
        ("mapper_timer_starts_from_preexisting_gpu_source_cache", False),
    ),
)
def test_latency_checkpoint_rejects_timing_or_lifecycle_scope_drift(
    tmp_path: Path, field: str, changed: bool
) -> None:
    path, value, _ = _commit_staged_latency_checkpoint(tmp_path)
    value[field] = changed
    hashed_json(path, value)
    with pytest.raises(ValueError, match="latency checkpoint invariant changed"):
        verify_length_checkpoint(path, fingerprint="f" * 64, length=64)


def test_latency_checkpoint_rejects_restage_device_drift(tmp_path: Path) -> None:
    path, value, _ = _commit_staged_latency_checkpoint(tmp_path)
    lifecycle = value["source_cache_lifecycle"]  # type: ignore[assignment]
    lifecycle["restage"]["layer_devices"][0] = "cuda:2"  # type: ignore[index]
    hashed_json(path, value)
    with pytest.raises(ValueError, match="restage layer devices changed"):
        verify_length_checkpoint(path, fingerprint="f" * 64, length=64)


def test_latency_staging_inventory_rejects_an_uncommitted_extra_session(
    tmp_path: Path,
) -> None:
    _, _, session_path = _commit_staged_latency_checkpoint(tmp_path)
    extra = session_path.parent / f"source_cache_staging_{'a' * 32}.json"
    atomic_write_json(extra, {"status": "partial"})
    with pytest.raises(ValueError, match="partially committed"):
        reverse_latency._all_staging_session_paths(
            tmp_path,
            referenced=[str(session_path.relative_to(tmp_path))],
        )


def test_reverse_latency_uses_one_repository_wide_nonblocking_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = tmp_path / "gpu8_experiment.lock"
    monkeypatch.setattr(reverse_latency, "GLOBAL_GPU_LOCK", lock_path)
    first = reverse_latency.acquire_gpu_lock()
    try:
        with pytest.raises(RuntimeError, match="repository-managed"):
            reverse_latency.acquire_gpu_lock()
    finally:
        first.close()
