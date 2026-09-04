from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from kvcache_transfer.io import atomic_write_json


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kvcache_transfer_evaluate_long_context_script",
    ROOT / "scripts/evaluate_long_context.py",
)
assert SPEC is not None and SPEC.loader is not None
LONG = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LONG
SPEC.loader.exec_module(LONG)


def _conversation(index: int, *, turns: int = 4) -> list[dict[str, str]]:
    return [
        {
            "from": "human" if turn % 2 == 0 else "gpt",
            "value": f"row {index} substantive turn {turn}",
        }
        for turn in range(turns)
    ]


def test_qa_metrics_keep_official_and_gold_independent_relaxed_layers():
    prediction = "<think>irrelevant 1945</think>\nThe final answer is **Paris**."
    metrics = LONG.qa_metrics(prediction, ["Paris"])
    assert metrics["qa_f1"] < 1.0
    assert metrics["qa_exact_match"] == 0.0
    assert metrics["extracted_answer"] == "Paris."
    assert metrics["extracted_qa_f1"] == 1.0
    assert metrics["extracted_qa_exact_match"] == 1.0
    assert metrics["relaxed_qa_f1"] == 1.0
    assert metrics["relaxed_qa_exact_match"] == 1.0
    assert metrics["normalized_gold_containment"] == 1.0
    assert LONG.extract_answer_candidate("Reasoning\n\\boxed{New York}") == "New York"


def test_multi_news_loads_all_5622_aligned_official_test_rows(tmp_path):
    source = tmp_path / "test.src.cleaned"
    target = tmp_path / "test.tgt"
    source.write_text(
        "".join(
            ("" if index == 7 else f"doc {index} NEWLINE_CHAR next") + "\n"
            for index in range(5622)
        ),
        encoding="utf-8",
    )
    target.write_text(
        "".join(f"summary {index}\n" for index in range(5622)), encoding="utf-8"
    )
    samples, manifest = LONG.load_multi_news_samples(source, target)
    assert len(samples) == 5622
    assert samples[0].context == "doc 0 \n next"
    assert samples[7].context == ""
    assert samples[-1].references == ("summary 5621",)
    assert manifest["split"] == "official_test"
    assert manifest["source"]["sha256"] == LONG.sha256_file(source)

    target.write_text("summary\n", encoding="utf-8")
    with pytest.raises(ValueError, match="5622 aligned"):
        LONG.load_multi_news_samples(source, target)


def test_hotpot_loads_complete_dev_distractor_with_gold_and_formats_passages(tmp_path):
    path = tmp_path / "hotpot_dev_distractor_v1.json"
    records = [
        {
            "_id": f"id-{index}",
            "answer": "yes",
            "question": "Same?",
            "supporting_facts": [["A", 0]],
            "context": [["A", ["First.", " Second."]], ["B", ["Other."]]],
            "type": "comparison",
            "level": "easy",
        }
        for index in range(7405)
    ]
    path.write_text(json.dumps(records), encoding="utf-8")
    samples, manifest = LONG.load_hotpotqa_samples(path)
    assert len(samples) == 7405
    assert samples[0].references == ("yes",)
    assert samples[0].input_text == "Same?"
    assert "[Passage 1]\nTitle: A\nFirst. Second." in samples[0].context
    assert manifest["split"] == "dev_distractor"
    assert manifest["gold_answers_available"] is True
    assert "no public gold" in manifest["test_split_excluded"]


def test_conversation_heldout_is_deduplicated_deterministic_and_train_only():
    rows = [
        (index, f"row-{index}", str(index), _conversation(index))
        for index in range(8)
    ]
    rows.append((8, "duplicate", "duplicate", _conversation(0)))
    rows.append((9, "invalid", "invalid", _conversation(9, turns=1)))
    first, first_manifest = LONG._select_conversation_heldout(
        rows,
        task="lmsys",
        sample_count=3,
        selection_seed="fixed",
        minimum_pairs=2,
        evaluation_scope="multi_turn_test",
    )
    second, second_manifest = LONG._select_conversation_heldout(
        rows,
        task="lmsys",
        sample_count=3,
        selection_seed="fixed",
        minimum_pairs=2,
        evaluation_scope="multi_turn_test",
    )
    assert first == second
    assert first_manifest == second_manifest
    assert first_manifest["rows"] == 10
    assert first_manifest["eligible_rows"] == 9
    assert first_manifest["unique_eligible_conversations"] == 8
    assert first_manifest["duplicate_eligible_rows_removed"] == 1
    assert first_manifest["normalized_first_prompt_groups"] == 8
    assert first_manifest["first_prompt_group_duplicates_removed"] == 0
    assert first_manifest["selection"]["official_test_split"] is False
    assert first_manifest["selection"]["minimum_complete_human_gpt_pairs"] == 2
    assert all(sample.messages[-1][0] == "user" for sample in first)
    assert all(sample.references[0].endswith("turn 3") for sample in first)


def test_conversation_dedup_normalizes_text_and_groups_opening_prompts():
    canonical = _conversation(0)
    normalized_duplicate = [dict(turn) for turn in canonical]
    normalized_duplicate[0]["value"] = "  ROW\u30000 SUBSTANTIVE TURN 0  "
    same_prompt_new_conversation = [dict(turn) for turn in canonical]
    same_prompt_new_conversation[-1]["value"] = "a distinct continuation"
    rows = [
        (0, "canonical", "0", canonical),
        (1, "normalized-duplicate", "1", normalized_duplicate),
        (2, "same-prompt", "2", same_prompt_new_conversation),
        (3, "different", "3", _conversation(3)),
    ]
    samples, manifest = LONG._select_conversation_heldout(
        rows,
        task="sharegpt4",
        sample_count=2,
        selection_seed="fixed",
        minimum_pairs=2,
        evaluation_scope="multi_turn_test",
    )
    assert len(samples) == 2
    assert manifest["eligible_rows"] == 4
    assert manifest["unique_eligible_conversations"] == 3
    assert manifest["duplicate_eligible_rows_removed"] == 1
    assert manifest["normalized_first_prompt_groups"] == 2
    assert manifest["first_prompt_group_duplicates_removed"] == 1


def test_lmsys_single_pair_protocol_is_explicitly_not_multi_turn():
    rows = [
        (index, f"row-{index}", str(index), _conversation(index, turns=2))
        for index in range(3)
    ]
    samples, manifest = LONG._select_conversation_heldout(
        rows,
        task="lmsys",
        sample_count=2,
        selection_seed="fixed",
        minimum_pairs=1,
        evaluation_scope="single_turn_assistant_response_quality",
    )
    assert len(samples) == 2
    assert manifest["selection"]["minimum_complete_human_gpt_pairs"] == 1
    assert manifest["selection"]["evaluation_scope"] == (
        "single_turn_assistant_response_quality"
    )


def test_sharegpt_jsonl_uses_same_deterministic_heldout_protocol(tmp_path):
    path = tmp_path / "share.jsonl"
    records = [{"conversations": _conversation(index)} for index in range(8)]
    records.append(records[0])
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    samples, manifest = LONG.load_sharegpt_samples(
        path, sample_count=3, selection_seed="fixed"
    )
    assert len(samples) == 3
    assert manifest["split"] == "train"
    assert manifest["eligible_rows"] == 9
    assert manifest["unique_eligible_conversations"] == 8
    assert manifest["selection"]["official_test_split"] is False
    assert manifest["selection"]["evaluation_scope"] == (
        "multi_turn_conversation_continuation"
    )
    assert manifest["selection"]["minimum_reference_visible_characters"] == 16


def test_sharegpt_rejects_invisible_or_too_short_final_references(tmp_path):
    path = tmp_path / "share.jsonl"
    valid = [{"conversations": _conversation(index)} for index in range(3)]
    invisible = _conversation(10)
    invisible[-1]["value"] = "\u200b\u200d"
    too_short = _conversation(11)
    too_short[-1]["value"] = "okay"
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [*valid, {"conversations": invisible}, {"conversations": too_short}]
        ),
        encoding="utf-8",
    )
    samples, manifest = LONG.load_sharegpt_samples(
        path, sample_count=2, selection_seed="fixed"
    )
    assert len(samples) == 2
    assert manifest["rows"] == 5
    assert manifest["eligible_rows"] == 3
    assert all(LONG._visible_character_count(row.references[0]) >= 16 for row in samples)


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        rendered = "|".join(f"{item['role']}:{item['content']}" for item in messages)
        return list((rendered + "|assistant:").encode("utf-8"))

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return bytes(token_ids).decode("utf-8")


def test_all_formal_tasks_use_qwen_chat_without_thinking():
    tokenizer = _FakeTokenizer()
    hotpot = LONG.EvaluationSample(
        task="hotpotqa",
        index=0,
        sample_id="h",
        record_sha256="a" * 64,
        references=("answer",),
        input_text="question",
        context="passages",
    )
    ids, descriptor = LONG.build_prompt_token_ids(
        tokenizer, hotpot, prompt_mode="qwen_chat_no_think"
    )
    rendered = LONG.PROMPT_TEMPLATES["hotpotqa"].format(
        context="passages", input="question"
    )
    assert bytes(ids).decode() == f"user:{rendered}|assistant:"
    assert descriptor["prompt_mode"] == "qwen_chat_no_think"
    assert descriptor["template_arguments"]["enable_thinking"] is False
    with pytest.raises(ValueError, match="must use qwen_chat_no_think"):
        LONG.build_prompt_token_ids(tokenizer, hotpot, prompt_mode="completion")

    for task in LONG.DIALOG_TASKS:
        dialog = LONG.EvaluationSample(
            task=task,
            index=0,
            sample_id="s",
            record_sha256="b" * 64,
            references=("held out",),
            messages=(("user", "u1"), ("assistant", "a1"), ("user", "u2")),
        )
        dialog_ids, dialog_descriptor = LONG.build_prompt_token_ids(
            tokenizer, dialog, prompt_mode="qwen_chat_no_think"
        )
        assert bytes(dialog_ids).decode().endswith("user:u2|assistant:")
        assert dialog_descriptor["template_arguments"]["enable_thinking"] is False


def test_dialog_truncation_drops_complete_pairs_and_keeps_valid_chat():
    tokenizer = _FakeTokenizer()
    dialog = LONG.EvaluationSample(
        task="lmsys",
        index=0,
        sample_id="s",
        record_sha256="b" * 64,
        references=("held out",),
        messages=(
            ("user", "old question"),
            ("assistant", "old answer"),
            ("user", "new question"),
        ),
    )
    last_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": "new question"}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    used, descriptor, truncation, original = LONG.generation_prompt_token_ids(
        tokenizer,
        dialog,
        prompt_mode="qwen_chat_no_think",
        budget=len(last_only),
    )
    assert bytes(used).decode() == "user:new question|assistant:"
    assert descriptor["original_prompt_tokens"] > len(used)
    assert len(original) == descriptor["original_prompt_tokens"]
    assert truncation["dropped_message_pairs"] == 1
    assert truncation["final_user_content_truncation"] is None


def test_dialog_oversized_final_user_head_tail_truncates_content_then_retemplates():
    tokenizer = _FakeTokenizer()
    dialog = LONG.EvaluationSample(
        task="sharegpt4",
        index=0,
        sample_id="s",
        record_sha256="b" * 64,
        references=("held out",),
        messages=(("user", "abcdefghijklmnopqrstuvwxyz"),),
    )
    empty = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    used, _, truncation, _ = LONG.generation_prompt_token_ids(
        tokenizer,
        dialog,
        prompt_mode="qwen_chat_no_think",
        budget=len(empty) + 8,
    )
    rendered = bytes(used).decode()
    assert rendered.startswith("user:abcd")
    assert rendered.endswith("wxyz|assistant:")
    assert truncation["final_user_content_truncation"]["truncated"] is True


def test_rouge_l_matches_pinned_reference():
    from rouge import Rouge

    prediction = "alpha beta. gamma delta"
    reference = "alpha gamma. delta"
    expected = Rouge().get_scores([prediction], [reference], avg=True)["rouge-l"]["f"]
    assert LONG.rouge_l_f1(prediction, [reference]) == pytest.approx(expected)
    assert LONG._sample_metrics("multi_news", prediction, [reference])[
        "rouge_l_f1"
    ] == pytest.approx(expected)


def test_explicit_sample_indices_keep_original_indices():
    parsed = LONG._parse_sample_indices("hotpotqa:7404,lmsys:17")
    assert parsed == {"hotpotqa": (7404,), "lmsys": (17,)}
    rows = {
        "hotpotqa": [
            LONG.EvaluationSample("hotpotqa", index, str(index), str(index), ("a",))
            for index in range(7405)
        ],
        "lmsys": [
            LONG.EvaluationSample("lmsys", index, str(index), str(index), ("a",))
            for index in range(200)
        ],
    }
    selected = LONG._apply_sample_scope(rows, limit=None, sample_indices=parsed)
    assert selected["hotpotqa"][0].index == 7404
    assert selected["lmsys"][0].index == 17


def _target_binding_fixture():
    sample = LONG.EvaluationSample(
        task="hotpotqa",
        index=3,
        sample_id="id-3",
        record_sha256="d" * 64,
        references=("gold",),
    )
    truncation = {
        "policy": "head_tail",
        "truncated": False,
        "original_tokens": 10,
        "used_tokens": 10,
        "removed_tokens": 0,
        "head_tokens": 10,
        "tail_tokens": 0,
    }
    descriptor = {
        "prompt_mode": "qwen_chat_no_think",
        "original_prompt_tokens": 10,
        "original_prompt_token_ids_sha256": "prompt-original",
    }
    target = {
        "schema_version": LONG.SCHEMA_VERSION,
        "status": "complete",
        "task": "hotpotqa",
        "mode": "target_standalone",
        "sample_index": 3,
        "sample_id": "id-3",
        "record_sha256": "d" * 64,
        "references": ["gold"],
        "generation": {"requested_max_tokens": 256},
        "prompt": {
            **descriptor,
            "generation_context_truncation": truncation,
            "generation_context_token_ids_sha256": "prompt-used",
        },
    }
    return sample, truncation, descriptor, target


def test_mapped_target_binding_checks_reference_prompt_truncation_and_cap():
    sample, truncation, descriptor, target = _target_binding_fixture()
    binding = LONG._validate_and_bind_target_checkpoint(
        target,
        sample=sample,
        cap=256,
        prompt_descriptor=descriptor,
        truncation=truncation,
        used_prompt_sha256="prompt-used",
    )
    assert binding["target_checkpoint_canonical_sha256"] == LONG.canonical_digest(target)
    changed = json.loads(json.dumps(target))
    changed["prompt"]["generation_context_token_ids_sha256"] = "different"
    with pytest.raises(ValueError, match="prompt token IDs"):
        LONG._validate_and_bind_target_checkpoint(
            changed,
            sample=sample,
            cap=256,
            prompt_descriptor=descriptor,
            truncation=truncation,
            used_prompt_sha256="prompt-used",
        )


def test_evaluate_one_binds_mapped_result_to_preceding_target_payload():
    class FakeEngine:
        max_length = 800

        def generate_token_ids(self, prompt_ids, kwargs, truncation):
            assert prompt_ids
            assert kwargs["max_gen_toks"] == 512
            assert truncation == "error"
            return SimpleNamespace(
                text="Paris",
                generated_token_count=1,
                stop_reason="eos",
                cap_hit=False,
                generated_token_ids=(17,),
            )

    sample = LONG.EvaluationSample(
        task="hotpotqa",
        index=4,
        sample_id="id-4",
        record_sha256="4" * 64,
        references=("Paris",),
        input_text="Where?",
        context="France",
    )
    task_config = {"max_gen_tokens": 512, "prompt_mode": "qwen_chat_no_think"}
    target_value = LONG.evaluate_one(
        engine=FakeEngine(),
        tokenizer=_FakeTokenizer(),
        sample=sample,
        mode="target_standalone",
        task_config=task_config,
        reference_score_cap=1024,
        target_checkpoint=None,
    )
    target_payload = {
        "schema_version": LONG.SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": "fingerprint",
        "task": sample.task,
        "mode": "target_standalone",
        "sample_index": sample.index,
        "sample_id": sample.sample_id,
        "record_sha256": sample.record_sha256,
        "primary_metric": "relaxed_qa_f1",
        **target_value,
    }
    mapped_value = LONG.evaluate_one(
        engine=FakeEngine(),
        tokenizer=_FakeTokenizer(),
        sample=sample,
        mode="mapped",
        task_config=task_config,
        reference_score_cap=1024,
        target_checkpoint=target_payload,
    )
    assert mapped_value["metrics"]["relaxed_qa_f1"] == 1.0
    assert mapped_value["target_checkpoint_binding"][
        "target_checkpoint_canonical_sha256"
    ] == LONG.canonical_digest(target_payload)


def test_per_sample_checkpoint_requires_full_identity_and_primary_metric(tmp_path):
    class FakeEngine:
        max_length = 800

        def generate_token_ids(self, prompt_ids, kwargs, truncation):
            assert prompt_ids and kwargs["max_gen_toks"] == 512
            assert truncation == "error"
            return SimpleNamespace(
                text="gold",
                generated_token_count=1,
                stop_reason="eos",
                cap_hit=False,
                generated_token_ids=(17,),
            )

    sample = LONG.EvaluationSample(
        task="hotpotqa",
        index=7,
        sample_id="sample-7",
        record_sha256="c" * 64,
        references=("gold",),
        input_text="question",
        context="context",
    )
    path = tmp_path / "checkpoint.json"
    tokenizer = _FakeTokenizer()
    task_config = {"max_gen_tokens": 512, "prompt_mode": "qwen_chat_no_think"}
    target_value = LONG.evaluate_one(
        engine=FakeEngine(),
        tokenizer=tokenizer,
        sample=sample,
        mode="target_standalone",
        task_config=task_config,
        reference_score_cap=100,
        target_checkpoint=None,
    )
    target = {
        "schema_version": LONG.SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": "fingerprint",
        "task": "hotpotqa",
        "mode": "target_standalone",
        "sample_index": 7,
        "sample_id": "sample-7",
        "record_sha256": "c" * 64,
        "source_length": None,
        "primary_metric": "relaxed_qa_f1",
        **target_value,
    }
    mapped_value = LONG.evaluate_one(
        engine=FakeEngine(),
        tokenizer=tokenizer,
        sample=sample,
        mode="mapped",
        task_config=task_config,
        reference_score_cap=100,
        target_checkpoint=target,
    )
    payload = {
        "schema_version": LONG.SCHEMA_VERSION,
        "status": "complete",
        "request_fingerprint": "fingerprint",
        "task": "hotpotqa",
        "mode": "mapped",
        "sample_index": 7,
        "sample_id": "sample-7",
        "record_sha256": "c" * 64,
        "source_length": None,
        "primary_metric": "relaxed_qa_f1",
        **mapped_value,
    }
    atomic_write_json(path, payload)
    assert LONG.checkpoint_complete(
        path,
        fingerprint="fingerprint",
        task="hotpotqa",
        mode="mapped",
        sample=sample,
        tokenizer=tokenizer,
        max_length=800,
        reference_score_cap=100,
        target_checkpoint=target,
    )
    assert not LONG.checkpoint_complete(
        path,
        fingerprint="different",
        task="hotpotqa",
        mode="mapped",
        sample=sample,
        tokenizer=tokenizer,
        max_length=800,
        reference_score_cap=100,
        target_checkpoint=target,
    )
    payload["metrics"]["relaxed_qa_f1"] = float("nan")
    atomic_write_json(path, payload)
    assert not LONG.checkpoint_complete(
        path,
        fingerprint="fingerprint",
        task="hotpotqa",
        mode="mapped",
        sample=sample,
        tokenizer=tokenizer,
        max_length=800,
        reference_score_cap=100,
        target_checkpoint=target,
    )
