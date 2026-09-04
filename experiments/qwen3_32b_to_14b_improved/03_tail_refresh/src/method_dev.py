"""Data preparation and validation for tail-refresh method development.

Only official training partitions are accepted.  Selection is a deterministic
SHA-256 rank over sequence IDs inside preregistered strata; answers and model
outputs never enter the rank.  Prepared content is copied into this
candidate's artifact directory so later GPU runs consume a byte-frozen input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CANDIDATE_ROOT / "configs/method_dev.json"
DATASET_IDS = ("hotpotqa_train", "musique_train")
PHASES = ("smoke", "pilot", "expansion")


class MethodDevValidationError(ValueError):
    """The method-dev config, data, or selection violates isolation."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    return canonical_sha256([int(value) for value in values])


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MethodDevValidationError(f"{label} must be an object")
    return value


def _require_exact_path(value: Any, expected: Path, label: str) -> Path:
    observed = Path(str(value))
    for component in (observed, *observed.parents):
        if component.is_symlink():
            raise MethodDevValidationError(f"{label} path contains a symlink")
    observed = observed.resolve()
    if observed != expected.resolve():
        raise MethodDevValidationError(
            f"{label} changed: {observed} != {expected.resolve()}"
        )
    return observed


def load_and_validate_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = read_json(config_path)
    if not isinstance(raw, Mapping):
        raise MethodDevValidationError("config root must be an object")
    fixed = {
        "schema_version": 1,
        "protocol_id": "qwen3_32b_to_14b_tail_refresh_method_dev_v2",
        "status": "method_dev_only_locked_test_forbidden",
    }
    for key, expected in fixed.items():
        if raw.get(key) != expected:
            raise MethodDevValidationError(f"config {key} changed")
    _require_exact_path(raw.get("candidate_root"), CANDIDATE_ROOT, "candidate_root")

    isolation = _require_mapping(raw.get("isolation"), "isolation")
    if isolation.get("allowed_data_tier") != "method-dev":
        raise MethodDevValidationError("only method-dev data are allowed")
    if isolation.get("baseline_tree_write_allowed") is not False:
        raise MethodDevValidationError("baseline writes must remain forbidden")
    if isolation.get("locked_test_access_allowed") is not False:
        raise MethodDevValidationError("locked-test access must remain forbidden")
    if tuple(isolation.get("allowed_splits", ())) != DATASET_IDS:
        raise MethodDevValidationError("allowed method-dev splits changed")

    method = _require_mapping(raw.get("method"), "method")
    if method.get("id") != "native_target_tail_refresh_k20_v1":
        raise MethodDevValidationError("tail-refresh method ID changed")
    if tuple(method.get("replay_tokens", ())) != (1, 8, 32):
        raise MethodDevValidationError("registered replay windows changed")
    if tuple(method.get("builder_native_tail_tokens", ())) != (0, 7, 31):
        raise MethodDevValidationError("builder tail lengths changed")
    if method.get("engine_boundary_tokens") != 1:
        raise MethodDevValidationError("engine must own exactly one boundary token")

    expected_paths = {
        "source_32b": Path("/home/wby/models/Qwen3-32B"),
        "target_14b": Path("/home/wby/models/Qwen3-14B"),
    }
    models = _require_mapping(raw.get("models"), "models")
    for name, expected_path in expected_paths.items():
        entry = _require_mapping(models.get(name), f"models.{name}")
        model_path = _require_exact_path(entry.get("path"), expected_path, name)
        config_file = model_path / "config.json"
        if sha256_file(config_file) != entry.get("config_sha256"):
            raise MethodDevValidationError(f"{name} config SHA256 changed")

    mapper = _require_mapping(raw.get("mapper"), "mapper")
    mapper_path = _require_exact_path(
        mapper.get("manifest"),
        CANDIDATE_ROOT.parent.parent
        / "qwen3_32b_to_14b/artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01/manifest.json",
        "mapper manifest",
    )
    if mapper.get("access") != "read_only_content_addressed":
        raise MethodDevValidationError("baseline mapper access policy changed")
    if sha256_file(mapper_path) != mapper.get("manifest_sha256"):
        raise MethodDevValidationError("baseline mapper manifest SHA256 changed")

    datasets = _require_mapping(raw.get("datasets"), "datasets")
    expected_data_paths = {
        "hotpotqa_train": CANDIDATE_ROOT.parents[2]
        / "data/HotpotQA/raw/hotpot_train_v1.1.json",
        "musique_train": CANDIDATE_ROOT.parent
        / "06_musique_benchmark/upstream_raw/data/musique_ans_v1.0_train.jsonl",
    }
    for dataset_id, expected_path in expected_data_paths.items():
        entry = _require_mapping(datasets.get(dataset_id), f"datasets.{dataset_id}")
        data_path = _require_exact_path(
            entry.get("path"), expected_path, f"datasets.{dataset_id}"
        )
        if "train" not in data_path.name.lower():
            raise MethodDevValidationError(f"{dataset_id} is not a training filename")
        forbidden = tuple(str(value) for value in isolation.get("forbidden_path_fragments", ()))
        if any(fragment in str(data_path) for fragment in forbidden):
            raise MethodDevValidationError(f"{dataset_id} resolves to a forbidden split")
        if sha256_file(data_path) != entry.get("sha256"):
            raise MethodDevValidationError(f"{dataset_id} raw SHA256 changed")

    prompt = _require_mapping(raw.get("prompt"), "prompt")
    if prompt.get("mode") != "qwen_chat_no_think":
        raise MethodDevValidationError("prompt mode changed")
    if prompt.get("truncation") != "forbidden_in_method_dev_selection":
        raise MethodDevValidationError("method-dev prompts must not be truncated")
    if int(prompt.get("max_length", 0)) != 16_384:
        raise MethodDevValidationError("max_length changed")
    if int(prompt.get("max_generation_tokens", 0)) != 256:
        raise MethodDevValidationError("generation cap changed")
    expected_instruction = (
        "Answer the multi-hop question using only the supplied context. "
        "Return only the shortest answer span and no explanation."
    )
    expected_template = (
        expected_instruction
        + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    if prompt.get("instruction") != expected_instruction:
        raise MethodDevValidationError("shortest-answer instruction changed")
    if prompt.get("completion_template") != expected_template:
        raise MethodDevValidationError("completion prompt changed")
    if prompt.get("paragraph_order") != "official_idx_ascending_for_musique":
        raise MethodDevValidationError("MuSiQue paragraph order changed")
    if tuple(prompt.get("stop_sequences", ())) != ("\nQuestion:", "\nContext:"):
        raise MethodDevValidationError("generation stop sequences changed")

    musique_root = CANDIDATE_ROOT.parent / "06_musique_benchmark"
    binding = _require_mapping(
        raw.get("musique_protocol_binding"), "musique_protocol_binding"
    )
    binding_paths = {
        "config": musique_root / "config.json",
        "registry_manifest": musique_root
        / "artifacts/registry_v1/registry_manifest.json",
        "prompt_implementation": musique_root / "src/musique_benchmark.py",
    }
    resolved_binding: dict[str, Path] = {}
    for name, expected_path in binding_paths.items():
        entry = _require_mapping(binding.get(name), f"musique_protocol_binding.{name}")
        path = _require_exact_path(entry.get("path"), expected_path, f"MuSiQue {name}")
        if path.stat().st_size != int(entry.get("nbytes", -1)):
            raise MethodDevValidationError(f"MuSiQue {name} byte size changed")
        if sha256_file(path) != entry.get("sha256"):
            raise MethodDevValidationError(f"MuSiQue {name} SHA256 changed")
        resolved_binding[name] = path
    if binding["prompt_implementation"].get("function") != "build_prompt":
        raise MethodDevValidationError("bound MuSiQue prompt function changed")
    upstream_config = read_json(resolved_binding["config"])
    generation = _require_mapping(
        upstream_config.get("generation_protocol"),
        "MuSiQue generation_protocol",
    )
    expected_generation = {
        "prompt": "fixed_no_thinking_completion_prompt_with_all_official_paragraphs",
        "paragraph_order": "official_idx_ascending",
        "batch_size": 1,
        "do_sample": False,
        "temperature": 0.0,
        "max_new_tokens": 256,
        "max_total_tokens": 16384,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
    }
    for name, expected in expected_generation.items():
        if generation.get(name) != expected:
            raise MethodDevValidationError(
                f"bound MuSiQue generation protocol {name} changed"
            )
    registry = read_json(resolved_binding["registry_manifest"])
    if (
        registry.get("status") != "complete"
        or registry.get("config_sha256")
        != binding["config"]["sha256"]
        or registry.get("locked_outcomes_evaluated") is not False
    ):
        raise MethodDevValidationError("MuSiQue registry/config binding is invalid")

    outputs = _require_mapping(raw.get("outputs"), "outputs")
    _require_exact_path(
        outputs.get("prepared"),
        CANDIDATE_ROOT / "artifacts/prepared_method_dev_v2",
        "prepared output",
    )
    _require_exact_path(
        outputs.get("runs_root"), CANDIDATE_ROOT / "runs", "runs root"
    )
    validated = dict(raw)
    validated["config_path"] = str(config_path)
    validated["config_sha256"] = sha256_file(config_path)
    return validated


def _require_text(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise MethodDevValidationError(f"{label} must be a non-empty string")
    return value


def _format_hotpot_context(value: Any, label: str) -> str:
    if not isinstance(value, list) or not value:
        raise MethodDevValidationError(f"{label}.context must be a non-empty list")
    passages: list[str] = []
    for index, passage in enumerate(value, start=1):
        if not isinstance(passage, list) or len(passage) != 2:
            raise MethodDevValidationError(f"{label}.context[{index}] is malformed")
        title = _require_text(passage[0], f"{label}.title")
        sentences = passage[1]
        if not isinstance(sentences, list) or not sentences:
            raise MethodDevValidationError(f"{label}.sentences is malformed")
        text = "".join(
            _require_text(sentence, f"{label}.sentence", allow_empty=True)
            for sentence in sentences
        ).strip()
        passages.append(f"[Passage {index}]\nTitle: {title}\n{text}")
    return "\n\n".join(passages)


def _format_musique_context(value: Any, label: str) -> str:
    if not isinstance(value, list) or not value:
        raise MethodDevValidationError(f"{label}.paragraphs must be non-empty")
    passages: list[str] = []
    seen_indices: set[int] = set()
    validated: list[Mapping[str, Any]] = []
    for order, paragraph in enumerate(value, start=1):
        if not isinstance(paragraph, Mapping):
            raise MethodDevValidationError(f"{label}.paragraphs[{order}] is malformed")
        index = int(paragraph.get("idx", -1))
        if index < 0 or index in seen_indices:
            raise MethodDevValidationError(f"{label} has invalid paragraph index")
        seen_indices.add(index)
        title = _require_text(paragraph.get("title"), f"{label}.title")
        text = _require_text(paragraph.get("paragraph_text"), f"{label}.text")
        validated.append(paragraph)
    if seen_indices != set(range(len(value))):
        raise MethodDevValidationError(
            f"{label} paragraph indices are not contiguous from zero"
        )
    for paragraph in sorted(validated, key=lambda item: int(item["idx"])):
        passages.append(
            f"[{paragraph['idx']}] {paragraph['title']}\n"
            f"{str(paragraph['paragraph_text']).strip()}"
        )
    return "\n\n".join(passages)


@dataclass(frozen=True)
class CandidateSample:
    dataset_id: str
    sample_id: str
    stratum: str
    question: str
    context: str
    references: tuple[str, ...]
    source_record_sha256: str
    metadata: Mapping[str, Any]


def load_hotpot_candidates(path: str | Path) -> tuple[list[CandidateSample], int]:
    records = read_json(path)
    if not isinstance(records, list):
        raise MethodDevValidationError("HotpotQA train must be a JSON array")
    candidates: list[CandidateSample] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise MethodDevValidationError(f"HotpotQA row {index} is malformed")
        sample_id = _require_text(record.get("_id"), f"hotpot[{index}]._id")
        if sample_id in seen:
            raise MethodDevValidationError(f"duplicate HotpotQA ID: {sample_id}")
        seen.add(sample_id)
        kind = _require_text(record.get("type"), f"hotpot[{index}].type")
        level = _require_text(record.get("level"), f"hotpot[{index}].level")
        if kind not in {"bridge", "comparison"}:
            raise MethodDevValidationError(f"unexpected HotpotQA type: {kind}")
        if level not in {"easy", "medium", "hard"}:
            raise MethodDevValidationError(f"unexpected HotpotQA level: {level}")
        candidates.append(
            CandidateSample(
                dataset_id="hotpotqa_train",
                sample_id=sample_id,
                stratum=f"{kind}/{level}",
                question=_require_text(
                    record.get("question"), f"hotpot[{index}].question"
                ),
                context=_format_hotpot_context(record.get("context"), f"hotpot[{index}]"),
                references=(
                    _require_text(record.get("answer"), f"hotpot[{index}].answer"),
                ),
                source_record_sha256=canonical_sha256(record),
                metadata={"source_row": index, "type": kind, "level": level},
            )
        )
    return candidates, len(records)


def load_musique_candidates(path: str | Path) -> tuple[list[CandidateSample], int]:
    candidates: list[CandidateSample] = []
    seen: set[str] = set()
    rows = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise MethodDevValidationError(f"MuSiQue row {index} is malformed")
            sample_id = _require_text(record.get("id"), f"musique[{index}].id")
            if sample_id in seen:
                raise MethodDevValidationError(f"duplicate MuSiQue ID: {sample_id}")
            seen.add(sample_id)
            if record.get("answerable") is not True:
                raise MethodDevValidationError(
                    "answerable MuSiQue train unexpectedly contains an unanswerable row"
                )
            decomposition = record.get("question_decomposition")
            if not isinstance(decomposition, list) or len(decomposition) not in {2, 3, 4}:
                raise MethodDevValidationError("MuSiQue hop count is not 2, 3, or 4")
            answer = _require_text(record.get("answer"), f"musique[{index}].answer")
            aliases = record.get("answer_aliases")
            if not isinstance(aliases, list) or any(
                not isinstance(value, str) for value in aliases
            ):
                raise MethodDevValidationError("MuSiQue answer_aliases is malformed")
            references = tuple(dict.fromkeys([answer, *aliases]))
            candidates.append(
                CandidateSample(
                    dataset_id="musique_train",
                    sample_id=sample_id,
                    stratum=f"{len(decomposition)}hop",
                    question=_require_text(
                        record.get("question"), f"musique[{index}].question"
                    ),
                    context=_format_musique_context(
                        record.get("paragraphs"), f"musique[{index}]"
                    ),
                    references=references,
                    source_record_sha256=canonical_sha256(record),
                    metadata={
                        "source_row": index,
                        "hop_count": len(decomposition),
                        "paragraph_count": len(record["paragraphs"]),
                    },
                )
            )
    return candidates, rows


def _selection_rank(seed: str, dataset_id: str, sample_id: str) -> str:
    payload = f"{seed}\0{dataset_id}\0{sample_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_preregistered_samples(
    candidates: Sequence[CandidateSample],
    *,
    seed: str,
    per_phase: Mapping[str, int],
) -> list[dict[str, Any]]:
    if tuple(per_phase) != PHASES:
        raise MethodDevValidationError("selection phases must be smoke/pilot/expansion")
    quotas = [int(per_phase[phase]) for phase in PHASES]
    if not (0 < quotas[0] <= quotas[1] <= quotas[2]):
        raise MethodDevValidationError("selection phase quotas are not nested")
    by_stratum: dict[str, list[tuple[str, CandidateSample]]] = {}
    for sample in candidates:
        rank = _selection_rank(seed, sample.dataset_id, sample.sample_id)
        by_stratum.setdefault(sample.stratum, []).append((rank, sample))
    selected: list[dict[str, Any]] = []
    for stratum in sorted(by_stratum):
        ranked = sorted(by_stratum[stratum], key=lambda item: (item[0], item[1].sample_id))
        if len(ranked) < quotas[-1]:
            raise MethodDevValidationError(
                f"stratum {stratum} has {len(ranked)} rows, needs {quotas[-1]}"
            )
        for within_rank, (rank_hash, sample) in enumerate(
            ranked[: quotas[-1]], start=1
        ):
            memberships = [
                phase
                for phase, quota in zip(PHASES, quotas, strict=True)
                if within_rank <= quota
            ]
            selected.append(
                {
                    **asdict(sample),
                    "references": list(sample.references),
                    "metadata": dict(sample.metadata),
                    "selection_rank_sha256": rank_hash,
                    "selection_rank_within_stratum": within_rank,
                    "phase_membership": memberships,
                }
            )
    return sorted(selected, key=lambda row: (row["dataset_id"], row["stratum"], row["selection_rank_within_stratum"]))


_SPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize_for_overlap(value: str) -> str:
    value = _SPACE.sub(" ", value.lower()).strip()
    return _SPACE.sub(" ", _NON_ALNUM.sub(" ", value)).strip()


def _word_ngrams(value: str, n: int = 3) -> set[tuple[str, ...]]:
    words = normalize_for_overlap(value).split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + n]) for index in range(len(words) - n + 1)}


def audit_cross_dataset_overlap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hotpot = [row for row in rows if row["dataset_id"] == "hotpotqa_train"]
    musique = [row for row in rows if row["dataset_id"] == "musique_train"]
    exact_ids = sorted(
        {str(row["sample_id"]) for row in hotpot}
        & {str(row["sample_id"]) for row in musique}
    )
    exact_content: list[list[str]] = []
    seen_content: dict[str, tuple[str, str]] = {}
    for row in rows:
        normalized = normalize_for_overlap(
            f"{row['question']}\n{row['context']}"
        )
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        previous = seen_content.get(digest)
        current = (str(row["dataset_id"]), str(row["sample_id"]))
        if previous is not None and previous[0] != current[0]:
            exact_content.append([*previous, *current])
        else:
            seen_content[digest] = current
    near_questions: list[dict[str, Any]] = []
    hotpot_ngrams = [(_word_ngrams(str(row["question"])), row) for row in hotpot]
    musique_ngrams = [(_word_ngrams(str(row["question"])), row) for row in musique]
    for left_grams, left in hotpot_ngrams:
        for right_grams, right in musique_ngrams:
            union = left_grams | right_grams
            similarity = len(left_grams & right_grams) / len(union) if union else 1.0
            if similarity >= 0.90:
                near_questions.append(
                    {
                        "hotpotqa_id": left["sample_id"],
                        "musique_id": right["sample_id"],
                        "word_trigram_jaccard": similarity,
                    }
                )
    status = "passed" if not (exact_ids or exact_content or near_questions) else "failed"
    return {
        "status": status,
        "scope": "selected method-dev HotpotQA-train versus MuSiQue-train rows",
        "exact_cross_dataset_id_collisions": exact_ids,
        "exact_normalized_question_plus_context_collisions": exact_content,
        "near_question_threshold": 0.90,
        "near_question_collisions": near_questions,
        "answer_overlap_not_used_for_selection_or_contamination": True,
    }


def render_and_tokenize_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    template: str,
    max_length: int,
    max_generation_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_budget = max_length - max_generation_tokens
    rendered: list[dict[str, Any]] = []
    lengths: list[int] = []
    for row in rows:
        prompt = template.format(context=row["context"], question=row["question"])
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise MethodDevValidationError("tokenizer returned an unexpected batch")
            token_ids = token_ids[0]
        token_ids = [int(value) for value in token_ids]
        if len(token_ids) > prompt_budget:
            raise MethodDevValidationError(
                f"{row['dataset_id']}/{row['sample_id']} has {len(token_ids)} prompt "
                f"tokens, exceeding no-truncation budget {prompt_budget}"
            )
        # Engine removes one boundary before invoking the builder; R=32 then
        # needs at least 32 remaining tokens so one mapped token is retained.
        if len(token_ids) - 1 < 32:
            raise MethodDevValidationError(
                f"{row['dataset_id']}/{row['sample_id']} is too short for R=32"
            )
        lengths.append(len(token_ids))
        rendered.append(
            {
                **dict(row),
                "prompt_text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_token_ids": token_ids,
                "prompt_token_count": len(token_ids),
                "prompt_token_ids_sha256": token_ids_sha256(token_ids),
                "prompt_truncated": False,
            }
        )
    lengths_sorted = sorted(lengths)
    return rendered, {
        "count": len(lengths),
        "minimum": min(lengths),
        "maximum": max(lengths),
        "median": lengths_sorted[len(lengths_sorted) // 2],
        "prompt_budget": prompt_budget,
        "all_support_R32_with_one_or_more_mapped_tokens": True,
        "truncated_count": 0,
    }


def phase_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {
        phase: {dataset: 0 for dataset in DATASET_IDS} for phase in PHASES
    }
    for row in rows:
        for phase in row["phase_membership"]:
            result[phase][str(row["dataset_id"])] += 1
    return result


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)


def write_immutable(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"immutable artifact differs: {destination}")
        return
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_output_within(root: str | Path, output: str | Path) -> Path:
    root_path = Path(root).resolve()
    lexical = Path(output)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise MethodDevValidationError("output path contains a symlink")
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(root_path)
    except ValueError as exc:
        raise MethodDevValidationError(
            f"output {resolved} escapes isolated root {root_path}"
        ) from exc
    if not relative.parts:
        raise MethodDevValidationError("isolated output root itself is not a run")
    return resolved


__all__ = [
    "CANDIDATE_ROOT",
    "DATASET_IDS",
    "DEFAULT_CONFIG",
    "MethodDevValidationError",
    "PHASES",
    "audit_cross_dataset_overlap",
    "canonical_json_bytes",
    "canonical_sha256",
    "ensure_output_within",
    "jsonl_bytes",
    "load_and_validate_config",
    "load_hotpot_candidates",
    "load_musique_candidates",
    "phase_counts",
    "read_json",
    "render_and_tokenize_samples",
    "select_preregistered_samples",
    "sha256_file",
    "token_ids_sha256",
    "write_immutable",
]
