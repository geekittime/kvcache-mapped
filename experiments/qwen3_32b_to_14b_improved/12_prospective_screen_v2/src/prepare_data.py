"""Prepare answers only *after* outcome-independent IDs have been frozen.

The registry contains IDs/hashes/ranks, never answer or context text.  This
module first validates registry publication and G0, then dereferences only the
selected indices in the pinned official train sources.  There is no dev/test
loader or path in this package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .configuration import load_config
from .contracts import (
    ArtifactRef,
    ContractError,
    DATASETS,
    PROFILES,
    SHA256_RE,
    artifact_ref,
    canonical_sha256,
    code_inventory,
    immutable_json,
    immutable_jsonl,
    read_json,
    read_jsonl,
    token_ids_sha256,
    verify_sidecar,
    write_sidecar,
)
from .configuration import validated_profile_counts
from .registry_gate import validate_data_gates


PREPARED_ROW_FIELDS = {
    "schema_version",
    "tier",
    "dataset_id",
    "partition",
    "sample_id",
    "source_index",
    "stratum",
    "stratum_selection_rank",
    "selection_rank_sha256",
    "evaluation_rank",
    "source_projection_sha256",
    "references",
    "prompt_token_ids",
    "prompt_token_count",
    "prompt_token_ids_sha256",
    "truncation",
    "generation_cap",
    "stop_sequences",
    "no_think",
    "sample_binding_sha256",
}


def _text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ContractError(f"{label} must be a {'possibly empty ' if allow_empty else ''}string")
    return value


def _hotpot_context(record: Mapping[str, Any], label: str) -> str:
    context = record.get("context")
    if not isinstance(context, list) or not context:
        raise ContractError(f"{label}.context is malformed")
    passages: list[str] = []
    for index, passage in enumerate(context):
        if not isinstance(passage, list) or len(passage) != 2:
            raise ContractError(f"{label}.context[{index}] is malformed")
        title = _text(passage[0], f"{label}.context[{index}].title")
        sentences = passage[1]
        if not isinstance(sentences, list) or not sentences:
            raise ContractError(f"{label}.context[{index}].sentences is malformed")
        body = "".join(_text(item, f"{label}.sentence", allow_empty=True) for item in sentences).strip()
        passages.append(f"[{index}] {title}\n{body}")
    return "\n\n".join(passages)


def _musique_context(record: Mapping[str, Any], label: str) -> str:
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ContractError(f"{label}.paragraphs is malformed")
    if any(not isinstance(item, Mapping) for item in paragraphs):
        raise ContractError(f"{label}.paragraph is malformed")
    ordered = sorted(paragraphs, key=lambda item: int(item.get("idx", -1)))
    if [int(item.get("idx", -1)) for item in ordered] != list(range(len(ordered))):
        raise ContractError(f"{label}.paragraph indices are not contiguous")
    return "\n\n".join(
        f"[{int(item['idx'])}] {_text(item.get('title'), f'{label}.title', allow_empty=True)}\n"
        f"{_text(item.get('paragraph_text'), f'{label}.paragraph_text')}"
        for item in ordered
    )


def build_prompt(context: str, question: str) -> str:
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
        raise ContractError("tokenizer returned empty/negative token IDs")
    return result


def apply_no_think_chat_template(tokenizer: Any, prompt: str) -> list[int]:
    return _as_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def head_tail_token_ids(values: Sequence[int], budget: int) -> tuple[list[int], dict[str, Any]]:
    if type(budget) is not int or budget <= 0:
        raise ContractError("prompt budget must be a positive integer")
    ids = list(values)
    if any(type(item) is not int or item < 0 for item in ids):
        raise ContractError("prompt token IDs are malformed")
    original = len(ids)
    if original <= budget:
        return ids, {
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
    output = ids[:head] + (ids[-tail:] if tail else [])
    return output, {
        "policy": "head_tail",
        "truncated": True,
        "original_tokens": original,
        "used_tokens": len(output),
        "removed_tokens": original - len(output),
        "head_tokens": head,
        "tail_tokens": tail,
    }


def tokenizer_identity(source: Any, target: Any) -> dict[str, Any]:
    source_vocab = source.get_vocab()
    target_vocab = target.get_vocab()
    if source_vocab != target_vocab:
        raise ContractError("source/target tokenizer vocabularies differ")
    fields = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    source_special = {field: getattr(source, field, None) for field in fields}
    target_special = {field: getattr(target, field, None) for field in fields}
    if source_special != target_special:
        raise ContractError("source/target special token IDs differ")
    source_template = getattr(source, "chat_template", None)
    target_template = getattr(target, "chat_template", None)
    if not isinstance(source_template, str) or source_template != target_template:
        raise ContractError("source/target chat templates differ")
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


def _load_train_source(config: Mapping[str, Any], dataset_id: str) -> list[dict[str, Any]]:
    source = config["train_sources"][dataset_id]
    path = ArtifactRef.parse(source["artifact"], label=f"{dataset_id} train source").verify(
        label=f"{dataset_id} train source"
    )
    if source["format"] == "json":
        rows = read_json(path)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ContractError(f"{dataset_id} JSON train root is malformed")
    elif source["format"] == "jsonl":
        rows = read_jsonl(path)
    else:  # blocked by configuration, retained as defense in depth
        raise ContractError(f"unsupported train source format: {source['format']}")
    if len(rows) != source["rows"]:
        raise ContractError(f"{dataset_id} train row count drifted")
    return rows


def _references(dataset_id: str, record: Mapping[str, Any], label: str) -> list[str]:
    answer = _text(record.get("answer"), f"{label}.answer")
    if dataset_id == "hotpotqa_train":
        return [answer]
    aliases = record.get("answer_aliases", [])
    if aliases is None:
        aliases = []
    if not isinstance(aliases, list) or any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
        raise ContractError(f"{label}.answer_aliases is malformed")
    result: list[str] = []
    for candidate in [answer, *aliases]:
        if candidate not in result:
            result.append(candidate)
    return result


def _expected_source_stratum(
    dataset_id: str, record: Mapping[str, Any], *, sample_id: str
) -> str:
    if dataset_id == "hotpotqa_train":
        question_type = _text(record.get("type"), "HotpotQA type")
        level = _text(record.get("level"), "HotpotQA level")
        return f"{question_type}/{level}"
    if dataset_id == "musique_train":
        for prefix in ("2hop", "3hop", "4hop"):
            if sample_id.startswith(prefix):
                return prefix
        raise ContractError("MuSiQue sample ID lacks a frozen 2hop/3hop/4hop prefix")
    raise ContractError(f"unsupported stratum dataset: {dataset_id}")


def source_projection_sha256(
    dataset_id: str,
    record: Mapping[str, Any],
    *,
    sample_id: str,
    stratum: str,
) -> str:
    """Recompute experiment 10's frozen answer-independent source projection."""

    question = _text(record.get("question"), f"{dataset_id}.question")
    passages: list[str] = []
    if dataset_id == "hotpotqa_train":
        context = record.get("context")
        if not isinstance(context, list) or not context:
            raise ContractError("HotpotQA projection context is malformed")
        for index, item in enumerate(context):
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], list):
                raise ContractError(f"HotpotQA projection context[{index}] is malformed")
            title = _text(item[0], f"hotpot projection title {index}", allow_empty=True)
            passages.append(" ".join([title, *(str(sentence) for sentence in item[1])]))
        producer_dataset = "hotpotqa"
    elif dataset_id == "musique_train":
        paragraphs = record.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ContractError("MuSiQue projection paragraphs are malformed")
        if any(not isinstance(item, Mapping) for item in paragraphs):
            raise ContractError("MuSiQue projection paragraph is malformed")
        ordered = sorted(paragraphs, key=lambda item: int(item.get("idx", -1)))
        if [int(item.get("idx", -1)) for item in ordered] != list(range(len(ordered))):
            raise ContractError("MuSiQue projection paragraph indices are not contiguous")
        for index, item in enumerate(ordered):
            title = str(item.get("title") or "")
            paragraph = _text(item.get("paragraph_text"), f"musique projection paragraph {index}")
            passages.append(" ".join([title, paragraph]))
        producer_dataset = "musique"
    else:
        raise ContractError(f"unsupported projection dataset: {dataset_id}")
    projection = {
        "dataset_id": producer_dataset,
        "sample_id": sample_id,
        "question": question,
        "passages": passages,
        "stratum": stratum,
    }
    payload = (
        json.dumps(projection, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _materialize_split(
    config: Mapping[str, Any],
    key: str,
    selected: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    dataset_prefix, partition = key.rsplit("_", 1)
    dataset_id = f"{dataset_prefix}_train"
    id_field = config["train_sources"][dataset_id]["id_field"]
    prompt_budget = config["generation"]["max_length"] - config["generation"]["max_new_tokens"]
    output: list[dict[str, Any]] = []
    for evaluation_rank, selected_row in enumerate(selected):
        index = selected_row["source_index"]
        if index >= len(source_rows):
            raise ContractError(f"{key} source index is outside frozen train source")
        record = source_rows[index]
        sample_id = record.get(id_field)
        if sample_id != selected_row["sample_id"]:
            raise ContractError(f"{key} frozen ID does not match train source index")
        expected_stratum = _expected_source_stratum(
            dataset_id, record, sample_id=sample_id
        )
        if selected_row["stratum"] != expected_stratum:
            raise ContractError(f"{key} frozen stratum differs from the official train record")
        observed_projection = source_projection_sha256(
            dataset_id,
            record,
            sample_id=sample_id,
            stratum=selected_row["stratum"],
        )
        if observed_projection != selected_row["source_projection_sha256"]:
            raise ContractError(f"{key} frozen answer-independent source projection hash differs")
        label = f"{dataset_id}[{index}]"
        question = _text(record.get("question"), f"{label}.question")
        context = (
            _hotpot_context(record, label)
            if dataset_id == "hotpotqa_train"
            else _musique_context(record, label)
        )
        prompt = build_prompt(context, question)
        full_ids = apply_no_think_chat_template(tokenizer, prompt)
        prompt_ids, truncation = head_tail_token_ids(full_ids, prompt_budget)
        if len(prompt_ids) <= config["methods"]["tail_refresh_r32"]["replay_tokens"]:
            raise ContractError(
                f"{label} prompt is too short for the frozen R32 causal partition"
            )
        row = {
            "schema_version": 2,
            "tier": "method_dev_prospective",
            "dataset_id": dataset_id,
            "partition": partition,
            "sample_id": sample_id,
            "source_index": index,
            "stratum": selected_row["stratum"],
            "stratum_selection_rank": selected_row["selection_rank"],
            "selection_rank_sha256": selected_row["selection_rank_sha256"],
            "evaluation_rank": evaluation_rank,
            "source_projection_sha256": selected_row["source_projection_sha256"],
            "references": _references(dataset_id, record, label),
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
            "truncation": truncation,
            "generation_cap": config["generation"]["max_new_tokens"],
            "stop_sequences": config["generation"]["stop_sequences"],
            "no_think": True,
        }
        row["sample_binding_sha256"] = canonical_sha256(row)
        output.append(row)
    return output


def _profile_rows(
    rows: Sequence[Mapping[str, Any]],
    profile: str,
    profiles: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    partition = "screen" if profile in {"smoke", "screen"} else "confirm"
    result: list[dict[str, Any]] = []
    for dataset in DATASETS:
        matching = [dict(row) for row in rows if row["dataset_id"] == dataset and row["partition"] == partition]
        matching.sort(key=lambda row: row["evaluation_rank"])
        expected = profiles[profile][dataset]
        if len(matching) < expected:
            raise ContractError(f"prepared rows lack {profile}/{dataset}")
        result.extend(matching[:expected])
    return result


def prepare(config_path: str | Path, *, source_tokenizer: Any, target_tokenizer: Any) -> dict[str, Any]:
    config = load_config(config_path, verify_inputs=True)
    profiles = validated_profile_counts(config["profiles"])
    registry, g0 = validate_data_gates(config)
    identity = tokenizer_identity(source_tokenizer, target_tokenizer)
    rows: list[dict[str, Any]] = []
    # Keep peak host RAM bounded: HotpotQA and MuSiQue source objects are never
    # resident together.  Each source is released after both frozen partitions
    # for that dataset have been materialized.
    for dataset_id in DATASETS:
        source_rows = _load_train_source(config, dataset_id)
        prefix = dataset_id.removesuffix("_train")
        for partition in ("screen", "confirm"):
            key = f"{prefix}_{partition}"
            rows.extend(
                _materialize_split(
                    config, key, registry[key], source_rows, target_tokenizer
                )
            )
        del source_rows
    if len({(row["dataset_id"], row["sample_id"]) for row in rows}) != len(rows):
        raise ContractError("prepared samples are not globally disjoint")
    for profile in PROFILES:
        _profile_rows(rows, profile, profiles)

    prepared_root = Path(config["outputs"]["prepared_root"])
    if prepared_root.exists() and prepared_root.is_symlink():
        raise ContractError("prepared output is a symlink")
    samples_ref = immutable_jsonl(prepared_root / "samples.jsonl", rows)
    write_sidecar(samples_ref)
    registry_refs = {
        "published_marker": config["registry"]["published_marker"],
        "registry_manifest": config["registry"]["registry_manifest"],
        "publication_receipt": config["registry"]["publication_receipt"],
        "split_artifacts": config["registry"]["split_artifacts"],
    }
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config": {
            "path": config["config_path"],
            "nbytes": Path(config["config_path"]).stat().st_size,
            "sha256": config["config_sha256"],
        },
        "registry": registry_refs,
        "g0_completion_receipt": config["g0"]["completion_receipt"],
        "g0_receipt_content_sha256": canonical_sha256(g0),
        "tokenizer_identity": identity,
        "samples": samples_ref.to_dict(),
        "sample_count": len(rows),
        "counts": {
            f"{dataset}/{partition}": sum(
                row["dataset_id"] == dataset and row["partition"] == partition for row in rows
            )
            for dataset in DATASETS
            for partition in ("screen", "confirm")
        },
        "sample_bindings_sha256": canonical_sha256([row["sample_binding_sha256"] for row in rows]),
        "answers_dereferenced_after_registry_freeze": True,
        "official_dev_or_test_opened": False,
        "code_inventory": code_inventory(),
    }
    manifest_ref = immutable_json(prepared_root / "manifest.json", manifest)
    write_sidecar(manifest_ref)
    receipt = {
        "schema_version": 2,
        "status": "complete",
        "runnable_source_config_had_no_placeholders": True,
        "registry_publication_state": "published_complete",
        "complete_g0_pass": True,
        "samples": samples_ref.to_dict(),
        "manifest": manifest_ref.to_dict(),
        "profiles": {
            profile: {
                "count": len(_profile_rows(rows, profile, profiles)),
                "sample_bindings_sha256": canonical_sha256(
                    [
                        row["sample_binding_sha256"]
                        for row in _profile_rows(rows, profile, profiles)
                    ]
                ),
            }
            for profile in PROFILES
        },
        "official_dev_or_test_opened": False,
    }
    receipt_ref = immutable_json(prepared_root / "completion_receipt.json", receipt)
    write_sidecar(receipt_ref)
    return {**receipt, "completion_receipt_sha256": receipt_ref.sha256}


def _validate_tokenizer_identity(value: Any, config: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "source_name_or_path",
        "target_name_or_path",
        "vocabulary_size",
        "vocabulary_sha256",
        "special_token_ids",
        "chat_template_sha256",
        "template_arguments",
    }:
        raise ContractError("prepared tokenizer identity schema changed")
    if (
        Path(str(value["source_name_or_path"])).resolve()
        != Path(config["runtime"]["source_model"]).resolve()
        or Path(str(value["target_name_or_path"])).resolve()
        != Path(config["runtime"]["target_model"]).resolve()
        or type(value["vocabulary_size"]) is not int
        or value["vocabulary_size"] <= 0
        or not isinstance(value["vocabulary_sha256"], str)
        or SHA256_RE.fullmatch(value["vocabulary_sha256"]) is None
        or not isinstance(value["chat_template_sha256"], str)
        or SHA256_RE.fullmatch(value["chat_template_sha256"]) is None
        or value["template_arguments"]
        != {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
    ):
        raise ContractError("prepared tokenizer identity changed")
    special = value["special_token_ids"]
    if not isinstance(special, Mapping) or set(special) != {
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
    }:
        raise ContractError("prepared tokenizer special-token schema changed")
    if any(item is not None and (type(item) is not int or item < 0) for item in special.values()):
        raise ContractError("prepared tokenizer special-token IDs are malformed")


def _validate_prepared_rows(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, int]],
    registry: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    registry_rows = {
        (row["dataset_id"], row["partition"], row["sample_id"]): row
        for split in registry.values()
        for row in split
    }
    seen: set[tuple[str, str]] = set()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {
        (dataset, partition): []
        for dataset in DATASETS
        for partition in ("screen", "confirm")
    }
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != PREPARED_ROW_FIELDS:
            raise ContractError("prepared sample field set changed")
        dataset = row.get("dataset_id")
        partition = row.get("partition")
        sample_id = row.get("sample_id")
        fixed = {
            "schema_version": 2,
            "tier": "method_dev_prospective",
            "generation_cap": config["generation"]["max_new_tokens"],
            "stop_sequences": config["generation"]["stop_sequences"],
            "no_think": True,
        }
        if any(row.get(field) != expected for field, expected in fixed.items()):
            raise ContractError("prepared sample frozen generation fields changed")
        if (
            dataset not in DATASETS
            or partition not in {"screen", "confirm"}
            or not isinstance(sample_id, str)
            or not sample_id
            or row.get("stratum") not in config["registry"]["strata_order"][dataset]
        ):
            raise ContractError("prepared sample identity/stratum is malformed")
        for field in ("source_index", "stratum_selection_rank", "evaluation_rank"):
            if type(row.get(field)) is not int or row[field] < 0:
                raise ContractError(f"prepared sample {field} is malformed")
        for field in (
            "selection_rank_sha256",
            "source_projection_sha256",
            "prompt_token_ids_sha256",
            "sample_binding_sha256",
        ):
            if not isinstance(row.get(field), str) or SHA256_RE.fullmatch(row[field]) is None:
                raise ContractError(f"prepared sample {field} is malformed")
        references = row.get("references")
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(item, str) or not item.strip() for item in references)
        ):
            raise ContractError("prepared sample references are malformed")
        ids = row.get("prompt_token_ids")
        if not isinstance(ids, list) or not ids or any(type(item) is not int or item < 0 for item in ids):
            raise ContractError("prepared prompt token IDs are malformed")
        if (
            row.get("prompt_token_count") != len(ids)
            or row.get("prompt_token_ids_sha256") != token_ids_sha256(ids)
            or len(ids) <= config["methods"]["tail_refresh_r32"]["replay_tokens"]
        ):
            raise ContractError("prepared prompt token evidence changed")
        truncation = row.get("truncation")
        if not isinstance(truncation, Mapping) or set(truncation) != {
            "policy",
            "truncated",
            "original_tokens",
            "used_tokens",
            "removed_tokens",
            "head_tokens",
            "tail_tokens",
        }:
            raise ContractError("prepared truncation schema changed")
        if (
            truncation["policy"] != "head_tail"
            or type(truncation["truncated"]) is not bool
            or any(
                type(truncation[field]) is not int or truncation[field] < 0
                for field in (
                    "original_tokens",
                    "used_tokens",
                    "removed_tokens",
                    "head_tokens",
                    "tail_tokens",
                )
            )
            or truncation["used_tokens"] != len(ids)
            or truncation["original_tokens"] - truncation["used_tokens"]
            != truncation["removed_tokens"]
            or truncation["head_tokens"] + truncation["tail_tokens"]
            != truncation["used_tokens"]
            or truncation["truncated"] != (truncation["removed_tokens"] > 0)
        ):
            raise ContractError("prepared truncation evidence does not recompute")
        body = {key: value for key, value in row.items() if key != "sample_binding_sha256"}
        if row["sample_binding_sha256"] != canonical_sha256(body):
            raise ContractError("prepared sample binding changed")
        key = (dataset, sample_id)
        if key in seen:
            raise ContractError("prepared sample ID is duplicated")
        seen.add(key)
        frozen = registry_rows.get((dataset, partition, sample_id))
        if frozen is None or {
            "source_index": row["source_index"],
            "stratum": row["stratum"],
            "selection_rank": row["stratum_selection_rank"],
            "selection_rank_sha256": row["selection_rank_sha256"],
            "source_projection_sha256": row["source_projection_sha256"],
        } != {
            "source_index": frozen["source_index"],
            "stratum": frozen["stratum"],
            "selection_rank": frozen["selection_rank"],
            "selection_rank_sha256": frozen["selection_rank_sha256"],
            "source_projection_sha256": frozen["source_projection_sha256"],
        }:
            raise ContractError("prepared sample differs from frozen native registry identity")
        grouped[(dataset, partition)].append(row)

    expected_counts = {
        f"{dataset}/{partition}": profiles[partition][dataset]
        for dataset in DATASETS
        for partition in ("screen", "confirm")
    }
    if manifest.get("counts") != expected_counts:
        raise ContractError("prepared manifest split counts changed")
    for (dataset, partition), values in grouped.items():
        expected = profiles[partition][dataset]
        if len(values) != expected:
            raise ContractError(f"prepared {dataset}/{partition} count changed")
        if [row["evaluation_rank"] for row in values] != list(range(expected)):
            raise ContractError(f"prepared {dataset}/{partition} evaluation ranks changed")
    bindings = [row["sample_binding_sha256"] for row in rows]
    if manifest.get("sample_bindings_sha256") != canonical_sha256(bindings):
        raise ContractError("prepared manifest sample-binding inventory changed")


def load_prepared(config: Mapping[str, Any], profile: str) -> tuple[list[dict[str, Any]], dict[str, Any], ArtifactRef]:
    profiles = validated_profile_counts(config["profiles"])
    if profile not in PROFILES:
        raise ContractError(f"unknown profile: {profile}")
    root = Path(config["outputs"]["prepared_root"]).resolve()
    manifest_path = root / "manifest.json"
    verify_sidecar(manifest_path)
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ContractError("prepared manifest root is malformed")
    if set(manifest) != {
        "schema_version",
        "status",
        "protocol_id",
        "config",
        "registry",
        "g0_completion_receipt",
        "g0_receipt_content_sha256",
        "tokenizer_identity",
        "samples",
        "sample_count",
        "counts",
        "sample_bindings_sha256",
        "answers_dereferenced_after_registry_freeze",
        "official_dev_or_test_opened",
        "code_inventory",
    }:
        raise ContractError("prepared manifest field set changed")
    expected_config = {
        "path": config["config_path"],
        "nbytes": Path(config["config_path"]).stat().st_size,
        "sha256": config["config_sha256"],
    }
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "complete"
        or manifest.get("protocol_id") != config["protocol_id"]
        or manifest.get("config") != expected_config
        or manifest.get("answers_dereferenced_after_registry_freeze") is not True
        or manifest.get("official_dev_or_test_opened") is not False
    ):
        raise ContractError("prepared manifest/config binding changed")
    if manifest.get("registry") != {
        "published_marker": config["registry"]["published_marker"],
        "registry_manifest": config["registry"]["registry_manifest"],
        "publication_receipt": config["registry"]["publication_receipt"],
        "split_artifacts": config["registry"]["split_artifacts"],
    }:
        raise ContractError("prepared registry binding changed")
    if manifest.get("g0_completion_receipt") != config["g0"]["completion_receipt"]:
        raise ContractError("prepared G0 binding changed")
    current_registry, current_g0 = validate_data_gates(config)
    if manifest.get("g0_receipt_content_sha256") != canonical_sha256(current_g0):
        raise ContractError("prepared G0 receipt content binding changed")
    if manifest.get("code_inventory") != code_inventory():
        raise ContractError("preparation code inventory changed")
    _validate_tokenizer_identity(manifest.get("tokenizer_identity"), config)
    samples_ref = ArtifactRef.parse(manifest.get("samples"), label="prepared samples")
    samples_path = samples_ref.verify(label="prepared samples")
    if samples_path != root / "samples.jsonl":
        raise ContractError("prepared samples path changed")
    verify_sidecar(samples_path)
    rows = read_jsonl(samples_path)
    if len(rows) != manifest.get("sample_count"):
        raise ContractError("prepared sample count changed")
    _validate_prepared_rows(config, rows, manifest, profiles, current_registry)

    completion_path = root / "completion_receipt.json"
    verify_sidecar(completion_path)
    completion = read_json(completion_path)
    expected_completion = {
        "schema_version": 2,
        "status": "complete",
        "runnable_source_config_had_no_placeholders": True,
        "registry_publication_state": "published_complete",
        "complete_g0_pass": True,
        "samples": samples_ref.to_dict(),
        "manifest": artifact_ref(manifest_path).to_dict(),
        "profiles": {
            name: {
                "count": sum(profiles[name].values()),
                "sample_bindings_sha256": canonical_sha256(
                    [row["sample_binding_sha256"] for row in _profile_rows(rows, name, profiles)]
                ),
            }
            for name in PROFILES
        },
        "official_dev_or_test_opened": False,
    }
    if completion != expected_completion:
        raise ContractError("prepared completion receipt does not reproduce")
    return _profile_rows(rows, profile, profiles), manifest, artifact_ref(manifest_path)


__all__ = [
    "apply_no_think_chat_template",
    "build_prompt",
    "head_tail_token_ids",
    "load_prepared",
    "prepare",
    "source_projection_sha256",
    "tokenizer_identity",
]
