"""CPU-only preparation of sealed, exact-token method-development inputs."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import unicodedata

from .contracts import (
    ArtifactRef,
    ContractError,
    DATASETS,
    DEFAULT_CONFIG,
    PREPARED_ROOT,
    ROOT,
    atomic_json,
    atomic_jsonl,
    canonical_bytes,
    canonical_sha256,
    code_inventory,
    load_config,
    read_json,
    require_child,
    sha256_file,
    token_ids_sha256,
    write_sidecar,
)


def normalized_question_group(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("question must be a non-empty string")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def selection_rank(namespace: str, record_id: str, source_index: int) -> tuple[str, int]:
    if not namespace or not record_id:
        raise ContractError("selection namespace and record ID must be non-empty")
    digest = hashlib.sha256(
        namespace.encode("utf-8") + b"\0" + record_id.encode("utf-8")
    ).hexdigest()
    return digest, int(source_index)


def head_tail_token_ids(
    token_ids: Sequence[int], budget: int
) -> tuple[list[int], dict[str, int | str | bool]]:
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ContractError("token budget must be a positive integer")
    values: list[int] = []
    for value in token_ids:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError("prompt token IDs must be non-negative integers")
        values.append(value)
    original = len(values)
    if original <= budget:
        return values, {
            "policy": "head_tail",
            "truncated": False,
            "original_tokens": original,
            "used_tokens": original,
            "removed_tokens": 0,
            "head_tokens": original,
            "tail_tokens": 0,
        }
    head = (budget + 1) // 2
    tail = budget - head
    result = values[:head] + (values[-tail:] if tail else [])
    return result, {
        "policy": "head_tail",
        "truncated": True,
        "original_tokens": original,
        "used_tokens": len(result),
        "removed_tokens": original - len(result),
        "head_tokens": head,
        "tail_tokens": tail,
    }


def _require_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ContractError(f"{label} must be a {'possibly empty ' if allow_empty else ''}string")
    return value


def _hotpot_context(record: Mapping[str, Any], label: str) -> str:
    context = record.get("context")
    if not isinstance(context, list) or not context:
        raise ContractError(f"{label}.context must be a non-empty list")
    passages: list[str] = []
    for index, passage in enumerate(context):
        if not isinstance(passage, list) or len(passage) != 2:
            raise ContractError(f"{label}.context[{index}] is malformed")
        title = _require_text(passage[0], f"{label}.context[{index}].title")
        sentences = passage[1]
        if not isinstance(sentences, list) or not sentences:
            raise ContractError(f"{label}.context[{index}].sentences is malformed")
        text = "".join(
            _require_text(sentence, f"{label}.sentence", allow_empty=True)
            for sentence in sentences
        ).strip()
        passages.append(f"[{index}] {title}\n{text}")
    return "\n\n".join(passages)


def _musique_context(record: Mapping[str, Any], label: str) -> str:
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ContractError(f"{label}.paragraphs must be non-empty")
    ordered = sorted(paragraphs, key=lambda value: int(value.get("idx", -1)))
    if [int(value.get("idx", -1)) for value in ordered] != list(range(len(ordered))):
        raise ContractError(f"{label}.paragraph indices are not contiguous")
    result: list[str] = []
    for paragraph in ordered:
        if not isinstance(paragraph, Mapping):
            raise ContractError(f"{label}.paragraph is malformed")
        index = int(paragraph["idx"])
        title = _require_text(paragraph.get("title"), f"{label}.title")
        text = _require_text(paragraph.get("paragraph_text"), f"{label}.text")
        result.append(f"[{index}] {title}\n{text}")
    return "\n\n".join(result)


def _prompt(context: str, question: str) -> str:
    return (
        "Answer the multi-hop question using only the supplied context. "
        "Return only the shortest answer span and no explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, Mapping) and "input_ids" in value:
        value = value["input_ids"]
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise ContractError("tokenizer unexpectedly returned a batch")
        value = value[0]
    if not isinstance(value, Sequence):
        raise ContractError("tokenizer did not return token IDs")
    result = [int(item) for item in value]
    if not result or any(item < 0 for item in result):
        raise ContractError("tokenizer returned empty or negative token IDs")
    return result


def apply_qwen_chat_template(tokenizer: Any, prompt: str) -> list[int]:
    value = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return _as_token_ids(value)


def _tokenizer_identity(source: Any, target: Any) -> dict[str, Any]:
    source_vocab = source.get_vocab()
    target_vocab = target.get_vocab()
    if source_vocab != target_vocab:
        raise ContractError("Qwen3-32B and Qwen3-14B tokenizer vocabularies differ")
    fields = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    source_special = {field: getattr(source, field, None) for field in fields}
    target_special = {field: getattr(target, field, None) for field in fields}
    if source_special != target_special:
        raise ContractError("source/target tokenizer special IDs differ")
    source_template = getattr(source, "chat_template", None)
    target_template = getattr(target, "chat_template", None)
    if not isinstance(source_template, str) or source_template != target_template:
        raise ContractError("source/target Qwen chat templates differ")
    return {
        "source_name_or_path": str(getattr(source, "name_or_path", "")),
        "target_name_or_path": str(getattr(target, "name_or_path", "")),
        "vocabulary_size": len(source_vocab),
        "vocabulary_sha256": canonical_sha256(sorted(source_vocab.items())),
        "special_token_ids": source_special,
        "chat_template_sha256": hashlib.sha256(source_template.encode("utf-8")).hexdigest(),
        "template_arguments": {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    }


def select_hotpot_records(
    records: Sequence[Mapping[str, Any]],
    *,
    screen_namespace: str,
    confirm_namespace: str,
    screen_count: int = 128,
    confirm_count: int = 256,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ids: set[str] = set()
    question_groups: Counter[str] = Counter()
    validated: list[tuple[int, Mapping[str, Any], str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ContractError(f"HotpotQA row {index} is not an object")
        sample_id = _require_text(record.get("_id"), f"hotpot[{index}]._id")
        if sample_id in ids:
            raise ContractError(f"duplicate HotpotQA ID: {sample_id}")
        ids.add(sample_id)
        question = _require_text(record.get("question"), f"hotpot[{index}].question")
        group = normalized_question_group(question)
        question_groups[group] += 1
        validated.append((index, record, sample_id, group))
    eligible = [item for item in validated if question_groups[item[3]] == 1]
    if len(eligible) < screen_count + confirm_count:
        raise ContractError("too few unique HotpotQA normalized-question groups")

    screen_ranked = sorted(
        eligible,
        key=lambda item: selection_rank(screen_namespace, item[2], item[0]),
    )
    screen_values = screen_ranked[:screen_count]
    screen_ids = {item[2] for item in screen_values}
    confirm_ranked = sorted(
        (item for item in eligible if item[2] not in screen_ids),
        key=lambda item: selection_rank(confirm_namespace, item[2], item[0]),
    )
    confirm_values = confirm_ranked[:confirm_count]

    def convert(
        values: Sequence[tuple[int, Mapping[str, Any], str, str]], namespace: str
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for rank_index, (source_index, record, sample_id, group) in enumerate(values):
            rank_digest, _ = selection_rank(namespace, sample_id, source_index)
            result.append(
                {
                    "source_index": source_index,
                    "sample_id": sample_id,
                    "selection_rank": rank_index,
                    "selection_rank_sha256": rank_digest,
                    "normalized_question_group_sha256": hashlib.sha256(
                        group.encode("utf-8")
                    ).hexdigest(),
                    "source_record_sha256": canonical_sha256(record),
                    "record": record,
                }
            )
        return result

    selections = {
        "screen": convert(screen_values, screen_namespace),
        "confirm": convert(confirm_values, confirm_namespace),
    }
    selected_groups = {
        partition: {row["normalized_question_group_sha256"] for row in rows}
        for partition, rows in selections.items()
    }
    if selected_groups["screen"] & selected_groups["confirm"]:
        raise ContractError("HotpotQA normalized-question group crosses partitions")
    audit = {
        "total_rows": len(validated),
        "unique_ids": len(ids),
        "duplicate_normalized_question_groups_quarantined": sum(
            count > 1 for count in question_groups.values()
        ),
        "duplicate_normalized_question_rows_quarantined": sum(
            count for count in question_groups.values() if count > 1
        ),
        "eligible_unique_question_rows": len(eligible),
        "screen_count": len(selections["screen"]),
        "confirm_count": len(selections["confirm"]),
        "cross_partition_id_overlap": 0,
        "cross_partition_normalized_question_overlap": 0,
    }
    return selections, audit


def _validate_musique_registries(
    config: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    dataset = config["datasets"]["musique_train"]
    musique_config_ref = ArtifactRef.parse(dataset["config"], label="MuSiQue config")
    musique_config = read_json(musique_config_ref.verify(label="MuSiQue config"))
    generation = musique_config.get("generation_protocol")
    if not isinstance(generation, Mapping) or generation != {
        "model_modes": ["target_standalone", "mapped"],
        "prompt": "fixed_no_thinking_completion_prompt_with_all_official_paragraphs",
        "paragraph_order": "official_idx_ascending",
        "batch_size": 1,
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 256,
        "max_total_tokens": 16384,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
    }:
        raise ContractError("MuSiQue 06 generation/prompt protocol changed")
    prompt_implementation = ArtifactRef.parse(
        config["provenance"]["musique_prompt_implementation"],
        label="MuSiQue 06 prompt implementation",
    )
    prompt_source = prompt_implementation.verify(label="MuSiQue 06 prompt implementation").read_text(
        encoding="utf-8"
    )
    expected_prologue = config["prompts"]["hotpotqa_template"].split("Context:\\n", 1)[0]
    expected_fragments = [
        "Answer the multi-hop question using only the supplied context. ",
        r"Return only the shortest answer span and no explanation.\n\n",
    ]
    expected_render = 'f"Context:\\n{context}\\n\\nQuestion: {record[\'question\']}\\nAnswer:"'
    if (
        "def build_prompt(record: Mapping[str, Any]) -> str:" not in prompt_source
        or not expected_prologue.replace("\n", r"\n").startswith("".join(expected_fragments))
        or any(fragment not in prompt_source for fragment in expected_fragments)
        or expected_render not in prompt_source
    ):
        raise ContractError("MuSiQue 06 prompt implementation no longer contains the frozen prompt")
    official_metric = musique_config.get("official_metric")
    if (
        not isinstance(official_metric, Mapping)
        or official_metric.get("headline") != "answer_f1"
        or official_metric.get("secondary") != "answer_em"
        or official_metric.get("aliases") != "max_over_primary_answer_and_answer_aliases"
        or official_metric.get("implementation_sha256")
        != config["metrics"]["musique_official"]["sha256"]
    ):
        raise ContractError("MuSiQue 06 official metric contract changed")
    source_spec = musique_config.get("source", {}).get("answerable_train", {})
    source_ref = ArtifactRef.parse(dataset["source"], label="MuSiQue train source")
    if (
        source_spec.get("count") != dataset["rows"]
        or source_spec.get("nbytes") != source_ref.nbytes
        or source_spec.get("sha256") != source_ref.sha256
    ):
        raise ContractError("MuSiQue 06 train source binding changed")
    manifest = read_json(ArtifactRef.parse(dataset["registry_manifest"], label="MuSiQue manifest").verify(label="MuSiQue manifest"))
    screen = read_json(ArtifactRef.parse(dataset["screen_registry"], label="MuSiQue screen registry").verify(label="MuSiQue screen registry"))
    confirm = read_json(ArtifactRef.parse(dataset["confirm_registry"], label="MuSiQue confirm registry").verify(label="MuSiQue confirm registry"))
    locked = read_json(ArtifactRef.parse(dataset["official_dev_lock"], label="MuSiQue official-dev lock").verify(label="MuSiQue official-dev lock"))
    if manifest.get("status") != "complete" or manifest.get("gpu_used") is not False:
        raise ContractError("MuSiQue registry manifest is not a complete CPU registry")
    if manifest.get("locked_outcomes_evaluated") is not False:
        raise ContractError("MuSiQue registry reports locked outcome access")
    expected = {
        "screen": (screen, "train_screen", 256),
        "confirm": (confirm, "train_confirm", 512),
    }
    result: dict[str, Mapping[str, Any]] = {}
    for partition, (value, split, count) in expected.items():
        if (
            value.get("schema_version") != 1
            or value.get("kind") != "musique_method_dev_selection"
            or value.get("tier") != "method-dev"
            or value.get("split") != split
            or value.get("count") != count
            or not isinstance(value.get("records"), list)
            or len(value["records"]) != count
        ):
            raise ContractError(f"MuSiQue {partition} registry contract changed")
        if (
            value.get("config_sha256") != musique_config_ref.sha256
            or value.get("official_metric_sha256")
            != config["metrics"]["musique_official"]["sha256"]
        ):
            raise ContractError(f"MuSiQue {partition} config/metric binding changed")
        if canonical_sha256(value["records"]) != value.get("records_sha256"):
            # Registry v1 appends a newline before hashing canonical records.
            digest = hashlib.sha256(canonical_bytes(value["records"], newline=True)).hexdigest()
            if digest != value.get("records_sha256"):
                raise ContractError(f"MuSiQue {partition} identity digest changed")
        result[partition] = value
    screen_ids = {row["id"] for row in screen["records"]}
    confirm_ids = {row["id"] for row in confirm["records"]}
    if screen_ids & confirm_ids:
        raise ContractError("MuSiQue method-dev registries overlap")
    if (
        locked.get("tier") != "locked-test"
        or locked.get("split") != "official_dev"
        or locked.get("state") != "sealed_until_candidate_freeze"
        or locked.get("records_embedded") is not False
    ):
        raise ContractError("MuSiQue official dev is not sealed")
    if (
        locked.get("config_sha256") != musique_config_ref.sha256
        or locked.get("official_metric_sha256")
        != config["metrics"]["musique_official"]["sha256"]
    ):
        raise ContractError("MuSiQue official-dev lock binding changed")
    selections = manifest.get("selections")
    for name, field in (("train_screen", "screen_registry"), ("train_confirm", "confirm_registry"), ("official_dev_locked", "official_dev_lock")):
        entry = selections.get(name) if isinstance(selections, Mapping) else None
        reference = ArtifactRef.parse(dataset[field], label=f"MuSiQue {field}")
        if not isinstance(entry, Mapping) or entry.get("nbytes") != reference.nbytes or entry.get("sha256") != reference.sha256:
            raise ContractError(f"MuSiQue manifest does not bind {name}")
    return result, {
        "registry_manifest": dataset["registry_manifest"],
        "screen_registry": dataset["screen_registry"],
        "confirm_registry": dataset["confirm_registry"],
        "official_dev_lock": dataset["official_dev_lock"],
        "official_dev_state": "sealed_until_candidate_freeze",
        "official_dev_records_opened": False,
        "screen_confirm_id_overlap": 0,
    }


def load_selected_musique_records(
    source: Path, registries: Mapping[str, Mapping[str, Any]], expected_rows: int
) -> dict[str, list[dict[str, Any]]]:
    by_index: dict[int, tuple[str, Mapping[str, Any]]] = {}
    for partition, registry in registries.items():
        for identity in registry["records"]:
            index = identity.get("source_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ContractError("MuSiQue registry source_index is malformed")
            if index in by_index:
                raise ContractError("MuSiQue registry source_index is duplicated")
            by_index[index] = (partition, identity)
    found: dict[str, list[dict[str, Any]]] = {"screen": [], "confirm": []}
    observed_rows = 0
    with source.open("r", encoding="utf-8") as handle:
        for source_index, line in enumerate(handle):
            if not line.strip():
                raise ContractError(f"blank MuSiQue row {source_index}")
            observed_rows += 1
            selected = by_index.get(source_index)
            if selected is None:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid MuSiQue row {source_index}") from exc
            if not isinstance(record, Mapping):
                raise ContractError(f"MuSiQue row {source_index} is not an object")
            partition, identity = selected
            if record.get("id") != identity.get("id"):
                raise ContractError("MuSiQue registry ID/source_index mismatch")
            record_hash = hashlib.sha256(canonical_bytes(record, newline=True)).hexdigest()
            question = _require_text(record.get("question"), "MuSiQue question")
            question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
            if record_hash != identity.get("record_sha256") or question_hash != identity.get("question_sha256"):
                raise ContractError("MuSiQue selected record content changed")
            found[partition].append(
                {
                    "source_index": source_index,
                    "sample_id": str(record["id"]),
                    "selection_rank": len(found[partition]),
                    "selection_rank_sha256": hashlib.sha256(
                        (str(registries[partition]["namespace"]) + "\0" + str(record["id"])).encode("utf-8")
                    ).hexdigest(),
                    "normalized_question_group_sha256": hashlib.sha256(
                        normalized_question_group(question).encode("utf-8")
                    ).hexdigest(),
                    "source_record_sha256": record_hash,
                    "record": record,
                }
            )
    if observed_rows != expected_rows:
        raise ContractError(f"MuSiQue row count changed: {observed_rows} != {expected_rows}")
    for partition, registry in registries.items():
        if len(found[partition]) != registry["count"]:
            raise ContractError(f"MuSiQue {partition} selected records are incomplete")
        expected_order = [row["id"] for row in registry["records"]]
        found[partition].sort(key=lambda row: expected_order.index(row["sample_id"]))
        for rank_index, row in enumerate(found[partition]):
            row["selection_rank"] = rank_index
    return found


def _prepared_row(
    *,
    dataset_id: str,
    partition: str,
    identity: Mapping[str, Any],
    tokenizer: Any,
    prompt_budget: int,
    cap: int,
    smoke_count: int,
) -> dict[str, Any]:
    record = identity["record"]
    question = _require_text(record.get("question"), f"{dataset_id}.question")
    if dataset_id == "hotpotqa_train":
        context = _hotpot_context(record, f"hotpot[{identity['source_index']}]")
        references = [_require_text(record.get("answer"), "HotpotQA answer")]
        stratum = f"{record.get('type', 'unknown')}/{record.get('level', 'unknown')}"
    elif dataset_id == "musique_train":
        if record.get("answerable") is not True:
            raise ContractError("selected MuSiQue row is not answerable")
        context = _musique_context(record, f"musique[{identity['source_index']}]")
        answer = _require_text(record.get("answer"), "MuSiQue answer")
        aliases = record.get("answer_aliases")
        if not isinstance(aliases, list) or any(not isinstance(value, str) for value in aliases):
            raise ContractError("MuSiQue aliases are malformed")
        references = [answer, *aliases]
        stratum = f"{len(record.get('question_decomposition', []))}hop"
    else:
        raise ContractError(f"unsupported dataset: {dataset_id}")
    rendered = _prompt(context, question)
    original_ids = apply_qwen_chat_template(tokenizer, rendered)
    used_ids, truncation = head_tail_token_ids(original_ids, prompt_budget)
    if len(used_ids) < 33:
        raise ContractError("prepared prompt cannot implement R=32 causal split")
    profile_membership = [partition]
    if partition == "screen" and int(identity["selection_rank"]) < smoke_count:
        profile_membership.insert(0, "smoke")
    row: dict[str, Any] = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "partition": partition,
        "profile_membership": profile_membership,
        "sample_id": identity["sample_id"],
        "source_index": identity["source_index"],
        "selection_rank": identity["selection_rank"],
        "selection_rank_sha256": identity["selection_rank_sha256"],
        "normalized_question_group_sha256": identity["normalized_question_group_sha256"],
        "source_record_sha256": identity["source_record_sha256"],
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "stratum": stratum,
        "references": references,
        "references_sha256": canonical_sha256(references),
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "original_prompt_token_count": len(original_ids),
        "original_prompt_token_ids_sha256": token_ids_sha256(original_ids),
        "prompt_token_ids": used_ids,
        "prompt_token_count": len(used_ids),
        "prompt_token_ids_sha256": token_ids_sha256(used_ids),
        "truncation": truncation,
        "generation_cap": cap,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
    }
    row["sample_binding_sha256"] = canonical_sha256(row)
    return row


def validate_prepared_row(row: Mapping[str, Any], *, max_length: int, cap: int) -> None:
    if row.get("schema_version") != 1 or row.get("dataset_id") not in DATASETS:
        raise ContractError("prepared row schema/dataset is invalid")
    if row.get("partition") not in {"screen", "confirm"}:
        raise ContractError("prepared row partition is invalid")
    unsigned = dict(row)
    binding = unsigned.pop("sample_binding_sha256", None)
    if binding != canonical_sha256(unsigned):
        raise ContractError("prepared sample binding changed")
    ids = row.get("prompt_token_ids")
    if not isinstance(ids, list) or any(type(value) is not int for value in ids):
        raise ContractError("prepared prompt token IDs are malformed")
    if token_ids_sha256(ids) != row.get("prompt_token_ids_sha256"):
        raise ContractError("prepared prompt token IDs changed")
    if len(ids) != row.get("prompt_token_count") or len(ids) < 33:
        raise ContractError("prepared prompt token count is invalid")
    if row.get("generation_cap") != cap or len(ids) + cap > max_length:
        raise ContractError("prepared generation cap/context budget changed")
    if row.get("stop_sequences") != ["\nQuestion:", "\nContext:"]:
        raise ContractError("prepared stop strings changed")
    references = row.get("references")
    if not isinstance(references, list) or not references:
        raise ContractError("prepared references are missing")
    if canonical_sha256(references) != row.get("references_sha256"):
        raise ContractError("prepared references changed")


def prepare_payloads(config_path: str | Path = DEFAULT_CONFIG) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    config = load_config(config_path, verify_inputs=True)
    from transformers import AutoTokenizer

    runtime = config["runtime"]
    source_tokenizer = AutoTokenizer.from_pretrained(
        runtime["source_model"], local_files_only=True, use_fast=True
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        runtime["target_model"], local_files_only=True, use_fast=True
    )
    tokenizer_identity = _tokenizer_identity(source_tokenizer, target_tokenizer)

    hotpot_source = ArtifactRef.parse(
        config["datasets"]["hotpotqa_train"]["source"], label="HotpotQA train"
    ).verify(label="HotpotQA train")
    hotpot_records = read_json(hotpot_source)
    if not isinstance(hotpot_records, list) or len(hotpot_records) != 90447:
        raise ContractError("HotpotQA official train row count changed")
    selection = config["selection"]
    hotpot, hotpot_audit = select_hotpot_records(
        hotpot_records,
        screen_namespace=selection["hotpotqa_screen_namespace"],
        confirm_namespace=selection["hotpotqa_confirm_namespace"],
    )

    musique_registries, musique_audit = _validate_musique_registries(config)
    musique_source = ArtifactRef.parse(
        config["datasets"]["musique_train"]["source"], label="MuSiQue train"
    ).verify(label="MuSiQue train")
    musique = load_selected_musique_records(
        musique_source, musique_registries, int(config["datasets"]["musique_train"]["rows"])
    )

    max_length = int(config["generation"]["max_length"])
    cap = int(config["generation"]["max_new_tokens"])
    budget = max_length - cap
    rows: list[dict[str, Any]] = []
    for dataset_id, selections in (("hotpotqa_train", hotpot), ("musique_train", musique)):
        smoke_count = int(config["profiles"]["smoke"][dataset_id])
        for partition in ("screen", "confirm"):
            expected = int(config["profiles"][partition][dataset_id])
            if len(selections[partition]) != expected:
                raise ContractError(f"{dataset_id}/{partition} selection count changed")
            for identity in selections[partition]:
                row = _prepared_row(
                    dataset_id=dataset_id,
                    partition=partition,
                    identity=identity,
                    tokenizer=target_tokenizer,
                    prompt_budget=budget,
                    cap=cap,
                    smoke_count=smoke_count,
                )
                validate_prepared_row(row, max_length=max_length, cap=cap)
                rows.append(row)

    ids_by_dataset_partition = {
        dataset: {
            partition: [
                row["sample_id"]
                for row in rows
                if row["dataset_id"] == dataset and row["partition"] == partition
            ]
            for partition in ("screen", "confirm")
        }
        for dataset in DATASETS
    }
    for dataset, partitions in ids_by_dataset_partition.items():
        if set(partitions["screen"]) & set(partitions["confirm"]):
            raise ContractError(f"{dataset} screen/confirm ID overlap")
    profile_counts = {
        profile: {
            dataset: sum(
                profile in row["profile_membership"] and row["dataset_id"] == dataset
                for row in rows
            )
            for dataset in DATASETS
        }
        for profile in ("smoke", "screen", "confirm")
    }
    if profile_counts != config["profiles"]:
        raise ContractError(f"prepared profile counts changed: {profile_counts}")

    registry = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "tier": "method-dev",
        "datasets": config["datasets"],
        "selection": config["selection"],
        "profile_counts": profile_counts,
        "sample_ids_sha256": {
            dataset: {
                partition: canonical_sha256(ids)
                for partition, ids in partitions.items()
            }
            for dataset, partitions in ids_by_dataset_partition.items()
        },
        "hotpotqa_selection_audit": hotpot_audit,
        "musique_registry_audit": musique_audit,
        "official_dev_raw_path_opened": False,
        "locked_test_metrics_observed": False,
    }
    preflight = {
        "schema_version": 1,
        "status": "passed_cpu_prepare",
        "gpu_used": False,
        "cuda_api_called": False,
        "protocol_id": config["protocol_id"],
        "config_sha256": config["config_sha256"],
        "tokenizer": tokenizer_identity,
        "generation": config["generation"],
        "prompt_template_sha256": hashlib.sha256(
            config["prompts"]["hotpotqa_template"].encode("utf-8")
        ).hexdigest(),
        "profile_counts": profile_counts,
        "prompt_lengths": {
            "minimum_used": min(row["prompt_token_count"] for row in rows),
            "maximum_used": max(row["prompt_token_count"] for row in rows),
            "maximum_original": max(row["original_prompt_token_count"] for row in rows),
            "truncated_rows": sum(bool(row["truncation"]["truncated"]) for row in rows),
            "all_support_r32": all(row["prompt_token_count"] >= 33 for row in rows),
        },
        "checks": {
            "hotpotqa_official_train_only": "passed",
            "hotpotqa_deterministic_id_sha_rank": "passed",
            "normalized_question_group_isolation": "passed",
            "musique_registry_v1_screen_confirm_exact": "passed",
            "musique_official_dev_sealed": "passed",
            "source_target_tokenizer_identity": "passed",
            "exact_runtime_prompt_ids_frozen": "passed",
            "uniform_generation_cap_256": "passed",
            "candidate_outputs_not_used_for_selection": "passed",
        },
    }
    return rows, registry, preflight


def prepare(
    config_path: str | Path = DEFAULT_CONFIG,
    output_dir: str | Path = PREPARED_ROOT,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    config = load_config(config_path, verify_inputs=True)
    output = require_child(ROOT / "artifacts", output_dir, label="prepared output")
    rows, registry, preflight = prepare_payloads(config_path)
    result = {
        "schema_version": 1,
        "status": "validated_dry_run" if dry_run else "prepared",
        "gpu_used": False,
        "cuda_api_called": False,
        "output": str(output),
        "rows": len(rows),
        "profile_counts": registry["profile_counts"],
        "config_sha256": config["config_sha256"],
    }
    if dry_run:
        return result
    output.mkdir(parents=True, exist_ok=True)
    rows_ref = atomic_jsonl(output / "samples.jsonl", rows)
    registry_ref = atomic_json(output / "data_registry.json", registry)
    preflight_ref = atomic_json(output / "preflight.json", preflight)
    for reference in (rows_ref, registry_ref, preflight_ref):
        write_sidecar(reference)
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "protocol_id": config["protocol_id"],
        "config": {
            "path": config["config_path"],
            "nbytes": Path(config["config_path"]).stat().st_size,
            "sha256": config["config_sha256"],
        },
        "files": {
            "samples.jsonl": rows_ref.to_dict(),
            "data_registry.json": registry_ref.to_dict(),
            "preflight.json": preflight_ref.to_dict(),
        },
        "rows": len(rows),
        "profile_counts": registry["profile_counts"],
        "sample_bindings_sha256": canonical_sha256(
            [row["sample_binding_sha256"] for row in rows]
        ),
        "code": code_inventory(),
        "claim_boundary": config["claim_boundary"],
    }
    manifest_ref = atomic_json(output / "manifest.json", manifest)
    write_sidecar(manifest_ref)
    return {**result, "manifest": manifest_ref.to_dict()}


__all__ = [
    "apply_qwen_chat_template",
    "head_tail_token_ids",
    "load_selected_musique_records",
    "normalized_question_group",
    "prepare",
    "prepare_payloads",
    "select_hotpot_records",
    "selection_rank",
    "validate_prepared_row",
]
