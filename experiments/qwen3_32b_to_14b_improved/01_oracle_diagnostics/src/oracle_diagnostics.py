"""CPU-only data preparation and deep preflight for A1 cache oracles."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

from cache_oracle import (
    CacheContext,
    OracleContractError,
    canonical_sha256,
    intervention_catalog,
    token_ids_sha256,
    validate_qwen3_config,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "prepare_v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PROMPT_INSTRUCTION = (
    "Answer the multi-hop question using only the supplied context. "
    "Return only the shortest answer span and no explanation."
)


class PreparationError(RuntimeError):
    """Raised when a pinned input or prepared artifact violates A1 policy."""


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"value is not canonical JSON: {exc}") from exc
    return payload + (b"\n" if newline else b"")


def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    nbytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifact_ref(path: str | Path) -> ArtifactRef:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PreparationError(f"artifact is not a regular non-symlink file: {resolved}")
    return ArtifactRef(
        path=str(resolved),
        nbytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    )


def code_artifacts(root: str | Path = ROOT) -> dict[str, dict[str, Any]]:
    """Return the complete executable/test/document identity for this A1 harness."""

    root = Path(root).resolve()
    relative_paths = (
        "README.md",
        "config.json",
        "src/__init__.py",
        "src/cache_oracle.py",
        "src/oracle_diagnostics.py",
        "src/runtime_harness.py",
        "scripts/prepare.py",
        "scripts/preflight.py",
        "tests/test_oracle_diagnostics.py",
    )
    result: dict[str, dict[str, Any]] = {}
    for relative in relative_paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise PreparationError(f"required A1 code/document is missing: {path}")
        result[relative] = artifact_ref(path).to_dict()
    return result


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_marker(reference: ArtifactRef) -> None:
    marker_path = Path(reference.path + ".sha256.json")
    marker = {
        "schema_version": 1,
        "path": reference.path,
        "nbytes": reference.nbytes,
        "sha256": reference.sha256,
    }
    _atomic_bytes(marker_path, canonical_bytes(marker, newline=True))


def atomic_json(path: str | Path, value: Any) -> ArtifactRef:
    destination = Path(path)
    _atomic_bytes(destination, canonical_bytes(value, newline=True))
    reference = artifact_ref(destination)
    _write_marker(reference)
    return reference


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> ArtifactRef:
    payload = b"".join(canonical_bytes(dict(row), newline=True) for row in rows)
    destination = Path(path)
    _atomic_bytes(destination, payload)
    reference = artifact_ref(destination)
    _write_marker(reference)
    return reference


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"JSON root is not an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise PreparationError(f"blank JSONL line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreparationError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise PreparationError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def require_output(path: str | Path) -> Path:
    root = ROOT.resolve()
    candidate = Path(path)
    if candidate.is_symlink():
        raise PreparationError("output directory may not be a symlink")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PreparationError(f"output escapes 01_oracle_diagnostics: {resolved}") from exc
    if not relative.parts:
        raise PreparationError("output may not be the experiment root itself")
    if relative.parts[0] != "artifacts":
        raise PreparationError("prepared output must be under 01_oracle_diagnostics/artifacts")
    return resolved


def _reject_symlink_components(path: Path, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise PreparationError(f"{label} contains a symlink: {current}")
        if current.parent == current:
            break
        current = current.parent


def verify_pinned_file(specification: Mapping[str, Any], *, label: str) -> Path:
    if set(specification) != {"path", "nbytes", "sha256"}:
        raise PreparationError(f"{label} must contain exactly path/nbytes/sha256")
    raw_path = specification.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise PreparationError(f"{label}.path must be absolute")
    path = Path(raw_path)
    _reject_symlink_components(path, label)
    if not path.is_file():
        raise PreparationError(f"{label} is not a regular file: {path}")
    expected_size = specification.get("nbytes")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        raise PreparationError(f"{label}.nbytes is invalid")
    if path.stat().st_size != expected_size:
        raise PreparationError(f"{label} byte size drift: {path.stat().st_size} != {expected_size}")
    expected_hash = specification.get("sha256")
    if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
        raise PreparationError(f"{label}.sha256 is malformed")
    observed_hash = sha256_file(path)
    if observed_hash != expected_hash:
        raise PreparationError(f"{label} SHA-256 drift: {observed_hash} != {expected_hash}")
    return path.resolve()


def _train_only_path(path: Path, *, dataset: str, split: str, tier: str) -> None:
    if tier != "method-dev":
        raise PreparationError(f"{dataset} is not method-dev")
    expected = {
        "hotpotqa_train": (
            "/home/wby/llm-cache/nvidia-kvcache/data/HotpotQA/raw/hotpot_train_v1.1.json",
            "train",
        ),
        "musique_train": (
            "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/06_musique_benchmark/upstream_raw/data/musique_ans_v1.0_train.jsonl",
            "train_screen",
        ),
    }
    if dataset not in expected:
        raise PreparationError(f"dataset is not allowlisted for A1: {dataset}")
    expected_path, expected_split = expected[dataset]
    if path != Path(expected_path).resolve() or split != expected_split:
        raise PreparationError(f"{dataset} path/split is not the frozen train-only input")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != 1 or config.get("phase") != "A1":
        raise PreparationError("unsupported A1 config")
    if config.get("tier") != "method-dev" or config.get("direction") != "qwen3_32b_to_14b":
        raise PreparationError("config tier or direction drift")
    policy = config.get("execution_policy", {})
    if policy.get("gpu_execution_authorized_by_this_config") is not False:
        raise PreparationError("CPU prepare config may not authorize GPU execution")
    if policy.get("locked_or_dev_split_access") is not False:
        raise PreparationError("locked/dev access must remain disabled")
    return config


def iter_json_array(path: str | Path, *, chunk_size: int = 1 << 20) -> Iterator[tuple[int, Any]]:
    """Stream a top-level JSON array with only the Python standard library."""

    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    index = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        def read_more() -> None:
            nonlocal buffer, position, eof
            block = handle.read(chunk_size)
            buffer = buffer[position:] + block
            position = 0
            if not block:
                eof = True

        read_more()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer) or buffer[position] != "[":
            raise PreparationError(f"top-level JSON is not an array: {path}")
        position += 1
        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer):
                    break
                if eof:
                    raise PreparationError(f"unterminated JSON array: {path}")
                read_more()
            if buffer[position] == "]":
                position += 1
                break
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as exc:
                    if eof:
                        raise PreparationError(f"invalid JSON array element {index}: {exc}") from exc
                    read_more()
                    continue
                position = end
                break
            yield index, value
            index += 1
        remainder = buffer[position:] + handle.read()
        if remainder.strip():
            raise PreparationError(f"trailing content after JSON array: {path}")


def _normalized_question(value: str) -> str:
    return " ".join(value.casefold().split())


def _rank(namespace: str, record_id: str, source_index: int) -> tuple[str, int]:
    digest = hashlib.sha256(namespace.encode("utf-8") + b"\0" + record_id.encode("utf-8"))
    return digest.hexdigest(), source_index


def _record_sha256(record: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(dict(record), newline=True)).hexdigest()


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def validate_hotpot_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise PreparationError("Hotpot row is not an object")
    required = {"_id", "question", "answer", "context", "supporting_facts", "level", "type"}
    if set(record) != required:
        raise PreparationError(f"Hotpot schema drift: {sorted(set(record) ^ required)}")
    for field in ("_id", "question", "answer", "level", "type"):
        if not isinstance(record[field], str) or not record[field]:
            raise PreparationError(f"Hotpot {field} must be a non-empty string")
    if not isinstance(record["context"], list) or not record["context"]:
        raise PreparationError("Hotpot context must be a non-empty list")
    for paragraph in record["context"]:
        if not isinstance(paragraph, list) or len(paragraph) != 2:
            raise PreparationError("Hotpot context paragraph schema drift")
        title, sentences = paragraph
        if not isinstance(title, str) or not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise PreparationError("Hotpot context title/sentences are malformed")
    if not isinstance(record["supporting_facts"], list):
        raise PreparationError("Hotpot supporting_facts must be a list")
    return record


def validate_musique_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise PreparationError("MuSiQue row is not an object")
    required = {
        "id",
        "question",
        "paragraphs",
        "answer",
        "answer_aliases",
        "answerable",
        "question_decomposition",
    }
    if not required.issubset(record):
        raise PreparationError(f"MuSiQue row is missing {sorted(required - set(record))}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise PreparationError("MuSiQue ID is invalid")
    if not isinstance(record["question"], str) or not record["question"]:
        raise PreparationError("MuSiQue question is invalid")
    if record["answerable"] is not True or not isinstance(record["answer"], str) or not record["answer"]:
        raise PreparationError("MuSiQue selected row is not answerable")
    if not isinstance(record["answer_aliases"], list) or not all(
        isinstance(value, str) for value in record["answer_aliases"]
    ):
        raise PreparationError("MuSiQue aliases are malformed")
    paragraphs = record["paragraphs"]
    if not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 20:
        raise PreparationError("MuSiQue paragraph count is invalid")
    seen: set[int] = set()
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict) or set(paragraph) != {
            "idx",
            "title",
            "paragraph_text",
            "is_supporting",
        }:
            raise PreparationError("MuSiQue paragraph schema drift")
        index = paragraph["idx"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise PreparationError("MuSiQue paragraph index is invalid")
        seen.add(index)
        if not isinstance(paragraph["title"], str) or not isinstance(
            paragraph["paragraph_text"], str
        ) or not isinstance(paragraph["is_supporting"], bool):
            raise PreparationError("MuSiQue paragraph values are malformed")
    if seen != set(range(len(paragraphs))):
        raise PreparationError("MuSiQue paragraph indices are not contiguous")
    return record


def hotpot_prompt(record: Mapping[str, Any]) -> str:
    record = validate_hotpot_record(dict(record))
    context = "\n\n".join(
        f"[{index}] {title}\n{''.join(sentences)}"
        for index, (title, sentences) in enumerate(record["context"])
    )
    return (
        f"{PROMPT_INSTRUCTION}\n\nContext:\n{context}\n\n"
        f"Question: {record['question']}\nAnswer:"
    )


def musique_prompt(record: Mapping[str, Any]) -> str:
    record = validate_musique_record(dict(record))
    paragraphs = sorted(record["paragraphs"], key=lambda value: value["idx"])
    context = "\n\n".join(
        f"[{paragraph['idx']}] {paragraph['title']}\n{paragraph['paragraph_text']}"
        for paragraph in paragraphs
    )
    return (
        f"{PROMPT_INSTRUCTION}\n\nContext:\n{context}\n\n"
        f"Question: {record['question']}\nAnswer:"
    )


def _unique_nonempty(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def load_musique_selection(
    *,
    raw_path: Path,
    registry_path: Path,
    expected_rows: int,
    expected_count: int,
) -> list[tuple[int, dict[str, Any]]]:
    registry = read_json(registry_path)
    if (
        registry.get("schema_version") != 1
        or registry.get("kind") != "musique_method_dev_selection"
        or registry.get("split") != "train_screen"
        or registry.get("tier") != "method-dev"
        or registry.get("count") != expected_count
    ):
        raise PreparationError("MuSiQue train-screen registry contract drift")
    identities = registry.get("records")
    if not isinstance(identities, list) or len(identities) != expected_count:
        raise PreparationError("MuSiQue train-screen identities are malformed")
    indices: dict[int, Mapping[str, Any]] = {}
    for identity in identities:
        if not isinstance(identity, dict):
            raise PreparationError("MuSiQue registry identity is not an object")
        index = identity.get("source_index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in indices:
            raise PreparationError("MuSiQue registry source index is invalid or duplicated")
        indices[index] = identity
    selected: dict[int, dict[str, Any]] = {}
    row_count = 0
    with raw_path.open("rb") as handle:
        for index, raw in enumerate(handle):
            row_count += 1
            if index not in indices:
                continue
            if not raw.strip():
                raise PreparationError(f"blank selected MuSiQue row {index}")
            try:
                record = validate_musique_record(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise PreparationError(f"invalid selected MuSiQue row {index}") from exc
            identity = indices[index]
            if record["id"] != identity.get("id"):
                raise PreparationError(f"MuSiQue registry ID drift at source row {index}")
            if _record_sha256(record) != identity.get("record_sha256"):
                raise PreparationError(f"MuSiQue registry record hash drift at source row {index}")
            if _question_sha256(record["question"]) != identity.get("question_sha256"):
                raise PreparationError(f"MuSiQue registry question hash drift at source row {index}")
            selected[index] = record
    if row_count != expected_rows:
        raise PreparationError(f"MuSiQue train row count drift: {row_count} != {expected_rows}")
    if set(selected) != set(indices):
        raise PreparationError("MuSiQue train-screen registry references missing rows")
    # Preserve the already-frozen registry order, not source order.
    return [(int(identity["source_index"]), selected[int(identity["source_index"])]) for identity in identities]


def load_hotpot_selection(
    *,
    raw_path: Path,
    namespace: str,
    expected_rows: int,
    expected_count: int,
    cross_dataset_questions: set[str],
) -> tuple[list[tuple[int, dict[str, Any]]], dict[str, int]]:
    counts: Counter[str] = Counter()
    minimal: list[tuple[tuple[str, int], int, str, str]] = []
    ids: set[str] = set()
    rows = 0
    for index, raw_record in iter_json_array(raw_path):
        record = validate_hotpot_record(raw_record)
        rows += 1
        record_id = record["_id"]
        if record_id in ids:
            raise PreparationError(f"duplicate Hotpot train ID: {record_id}")
        ids.add(record_id)
        normalized = _normalized_question(record["question"])
        counts[normalized] += 1
        minimal.append((_rank(namespace, record_id, index), index, record_id, normalized))
    if rows != expected_rows:
        raise PreparationError(f"Hotpot train row count drift: {rows} != {expected_rows}")
    eligible = [
        item
        for item in minimal
        if counts[item[3]] == 1 and item[3] not in cross_dataset_questions
    ]
    selected_minimal = sorted(eligible, key=lambda item: item[0])[:expected_count]
    if len(selected_minimal) != expected_count:
        raise PreparationError("too few unique cross-dataset-clean Hotpot train rows")
    selected_ids = {item[1]: item[2] for item in selected_minimal}
    selected: dict[int, dict[str, Any]] = {}
    for index, raw_record in iter_json_array(raw_path):
        if index not in selected_ids:
            continue
        record = validate_hotpot_record(raw_record)
        if record["_id"] != selected_ids[index]:
            raise PreparationError("Hotpot selection identity changed between streaming passes")
        selected[index] = record
    if set(selected) != set(selected_ids):
        raise PreparationError("Hotpot selection references missing source rows")
    # Preserve SHA-rank order.
    ordered = [(index, selected[index]) for _, index, _, _ in selected_minimal]
    diagnostics = {
        "train_rows": rows,
        "duplicate_normalized_question_groups_quarantined": sum(
            count > 1 for count in counts.values()
        ),
        "duplicate_normalized_question_rows_quarantined": sum(
            count for count in counts.values() if count > 1
        ),
        "cross_dataset_normalized_questions_quarantined": sum(
            normalized in cross_dataset_questions for normalized in counts
        ),
    }
    return ordered, diagnostics


def _serialize_prompt(tokenizer: Any, qa_prompt: str) -> tuple[str, list[int]]:
    try:
        serialized = tokenizer.apply_chat_template(
            [{"role": "user", "content": qa_prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = tokenizer.encode(serialized, add_special_tokens=False)
    except Exception as exc:
        raise PreparationError(f"Qwen chat-template serialization failed: {exc}") from exc
    if not isinstance(serialized, str) or not serialized:
        raise PreparationError("chat template returned an empty/non-string prompt")
    if not isinstance(token_ids, list) or len(token_ids) < 2 or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in token_ids
    ):
        raise PreparationError("tokenizer returned malformed prompt token IDs")
    return serialized, token_ids


def _tokenize_record(
    *,
    dataset: str,
    source_index: int,
    record: Mapping[str, Any],
    target_tokenizer: Any,
    source_tokenizer: Any,
    target_config_sha256: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    if dataset == "hotpotqa_train":
        sample_id = str(record["_id"])
        question = str(record["question"])
        references = [str(record["answer"])]
        qa_prompt = hotpot_prompt(record)
    elif dataset == "musique_train":
        sample_id = str(record["id"])
        question = str(record["question"])
        references = _unique_nonempty([str(record["answer"]), *record["answer_aliases"]])
        qa_prompt = musique_prompt(record)
    else:
        raise PreparationError(f"unsupported A1 dataset: {dataset}")
    target_serialized, target_ids = _serialize_prompt(target_tokenizer, qa_prompt)
    source_serialized, source_ids = _serialize_prompt(source_tokenizer, qa_prompt)
    if target_serialized != source_serialized or target_ids != source_ids:
        raise PreparationError("Qwen3-32B and Qwen3-14B prompt serialization/tokenization differ")
    if len(target_ids) > max_prompt_tokens:
        raise PreparationError(
            f"prompt {dataset}/{sample_id} has {len(target_ids)} tokens > {max_prompt_tokens}; "
            "A1 forbids truncation"
        )
    gold_first_tokens: set[int] = set()
    for reference in references:
        encoded = target_tokenizer.encode(reference, add_special_tokens=False)
        if encoded:
            gold_first_tokens.add(int(encoded[0]))
    if not gold_first_tokens:
        raise PreparationError(f"references for {dataset}/{sample_id} have no first answer token")
    try:
        context = CacheContext.from_prompt(
            dataset=dataset,
            sample_id=sample_id,
            serialized_prompt=target_serialized,
            token_ids=target_ids,
            target_config_sha256=target_config_sha256,
        )
    except OracleContractError as exc:
        raise PreparationError(f"cache context invalid for {dataset}/{sample_id}: {exc}") from exc
    return {
        "schema_version": 1,
        "tier": "method-dev",
        "split": "train_screen",
        "dataset": dataset,
        "sample_id": sample_id,
        "source_index": source_index,
        "source_record_sha256": _record_sha256(record),
        "question": question,
        "question_sha256": _question_sha256(question),
        "references": references,
        "qa_prompt": qa_prompt,
        "qa_prompt_sha256": hashlib.sha256(qa_prompt.encode("utf-8")).hexdigest(),
        "serialized_prompt": target_serialized,
        "serialized_prompt_sha256": context.serialized_prompt_sha256,
        "prompt_token_ids": target_ids,
        "prompt_token_ids_sha256": context.prompt_token_ids_sha256,
        "prefix_token_ids_sha256": context.prefix_token_ids_sha256,
        "boundary_token_id": context.boundary_token_id,
        "prompt_token_count": len(target_ids),
        "prefix_token_count": context.prefix_length,
        "gold_first_token_ids": sorted(gold_first_tokens),
        "max_new_tokens": max_new_tokens,
        "truncated": False,
        "cache_context": asdict(context),
    }


def _load_tokenizers(config: Mapping[str, Any]) -> tuple[Any, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise PreparationError("Transformers is required for CPU prompt preparation") from exc
    target_root = config["models"]["target"]["root"]
    source_root = config["models"]["source"]["root"]
    target = AutoTokenizer.from_pretrained(
        target_root, local_files_only=True, trust_remote_code=False
    )
    source = AutoTokenizer.from_pretrained(
        source_root, local_files_only=True, trust_remote_code=False
    )
    identity = lambda value: (
        value.__class__.__name__,
        value.vocab_size,
        value.eos_token_id,
        value.pad_token_id,
        value.chat_template,
    )
    if identity(target) != identity(source):
        raise PreparationError("source/target tokenizer runtime identities differ")
    return target, source


def _median(values: Sequence[int]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise PreparationError("cannot compute median of an empty list")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _validate_intervention_config(config: Mapping[str, Any]) -> None:
    expected = [value.to_dict() for value in intervention_catalog()]
    observed_ids = list(config["interventions"]["global_kv_factorial"])
    observed_ids.extend(
        value["id"] for value in config["interventions"]["mapped_base_native_kv_layer_replacements"]
    )
    if observed_ids != [value["id"] for value in expected]:
        raise PreparationError("config intervention order differs from executable catalog")
    for configured, executable in zip(
        config["interventions"]["mapped_base_native_kv_layer_replacements"],
        expected[4:],
        strict=True,
    ):
        if configured["layers"] != executable["native_kv_layers"]:
            raise PreparationError(f"configured layer replacement drift: {configured['id']}")


def prepare(root: str | Path = ROOT, output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise PreparationError(f"A1 root must be exactly {ROOT.resolve()}")
    output_dir = require_output(output)
    config = load_config(root / "config.json")
    _validate_intervention_config(config)

    accessed: list[str] = []
    for label, spec in config["protocol_inputs"].items():
        accessed.append(str(verify_pinned_file(spec, label=f"protocol_inputs.{label}")))
    for role in ("source", "target"):
        for key in ("config", "tokenizer_json", "tokenizer_config"):
            accessed.append(
                str(verify_pinned_file(config["models"][role][key], label=f"models.{role}.{key}"))
            )
        model_config = read_json(config["models"][role]["config"]["path"])
        try:
            validate_qwen3_config(model_config, source=role == "source")
        except OracleContractError as exc:
            raise PreparationError(f"{role} model config mismatch: {exc}") from exc
    if (
        config["models"]["source"]["tokenizer_json"]["sha256"]
        != config["models"]["target"]["tokenizer_json"]["sha256"]
        or config["models"]["source"]["tokenizer_config"]["sha256"]
        != config["models"]["target"]["tokenizer_config"]["sha256"]
    ):
        raise PreparationError("source/target tokenizer artifacts are not byte-identical")
    mapper_manifest_path = verify_pinned_file(
        config["mapped_cache"]["manifest"], label="mapped_cache.manifest"
    )
    accessed.append(str(mapper_manifest_path))
    metric_path = verify_pinned_file(
        {key: config["metric_reference"][key] for key in ("path", "nbytes", "sha256")},
        label="metric_reference",
    )
    accessed.append(str(metric_path))

    data_specs = config["method_dev_data"]
    hotpot_spec = data_specs["hotpotqa_train"]
    musique_spec = data_specs["musique_train"]
    hotpot_path = verify_pinned_file(
        {key: hotpot_spec[key] for key in ("path", "nbytes", "sha256")},
        label="method_dev_data.hotpotqa_train",
    )
    _train_only_path(
        hotpot_path,
        dataset="hotpotqa_train",
        split=hotpot_spec["split"],
        tier=hotpot_spec["tier"],
    )
    accessed.append(str(hotpot_path))
    musique_path = verify_pinned_file(
        {key: musique_spec[key] for key in ("path", "nbytes", "sha256")},
        label="method_dev_data.musique_train",
    )
    _train_only_path(
        musique_path,
        dataset="musique_train",
        split=musique_spec["split"],
        tier=musique_spec["tier"],
    )
    accessed.append(str(musique_path))
    musique_registry = verify_pinned_file(
        musique_spec["selection_registry"], label="method_dev_data.musique_train.selection_registry"
    )
    accessed.append(str(musique_registry))

    musique = load_musique_selection(
        raw_path=musique_path,
        registry_path=musique_registry,
        expected_rows=int(musique_spec["row_count"]),
        expected_count=int(musique_spec["selection_count"]),
    )
    musique_questions = {_normalized_question(record["question"]) for _, record in musique}
    if len(musique_questions) != len(musique):
        raise PreparationError("MuSiQue train-screen contains duplicate normalized questions")
    hotpot, hotpot_diagnostics = load_hotpot_selection(
        raw_path=hotpot_path,
        namespace=str(hotpot_spec["namespace"]),
        expected_rows=int(hotpot_spec["row_count"]),
        expected_count=int(hotpot_spec["selection_count"]),
        cross_dataset_questions=musique_questions,
    )
    if {_normalized_question(record["question"]) for _, record in hotpot} & musique_questions:
        raise PreparationError("selected Hotpot/MuSiQue normalized questions overlap")

    target_tokenizer, source_tokenizer = _load_tokenizers(config)
    prompt = config["prompt_protocol"]
    target_config_sha256 = config["models"]["target"]["config"]["sha256"]
    rows: list[dict[str, Any]] = []
    for dataset, values in (("hotpotqa_train", hotpot), ("musique_train", musique)):
        for source_index, record in values:
            rows.append(
                _tokenize_record(
                    dataset=dataset,
                    source_index=source_index,
                    record=record,
                    target_tokenizer=target_tokenizer,
                    source_tokenizer=source_tokenizer,
                    target_config_sha256=target_config_sha256,
                    max_prompt_tokens=int(prompt["max_prompt_tokens"]),
                    max_new_tokens=int(prompt["max_new_tokens"]),
                )
            )
    expected_total = int(hotpot_spec["selection_count"]) + int(musique_spec["selection_count"])
    if len(rows) != expected_total:
        raise PreparationError("prepared row count drift")
    keys = [(row["dataset"], row["sample_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise PreparationError("prepared sample keys are not unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_ref = atomic_jsonl(output_dir / "method_dev_records.jsonl", rows)
    interventions_payload = {
        "schema_version": 1,
        "kind": "a1_oracle_intervention_catalog",
        "count": len(intervention_catalog()),
        "reference_intervention": "native_k_native_v",
        "layer_replacement_base": "mapped_k_mapped_v",
        "interventions": [value.to_dict() for value in intervention_catalog()],
    }
    interventions_ref = atomic_json(output_dir / "interventions.json", interventions_payload)
    plan_rows: list[dict[str, Any]] = []
    for row in rows:
        group = canonical_sha256(
            {
                "dataset": row["dataset"],
                "sample_id": row["sample_id"],
                "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            }
        )
        for intervention in intervention_catalog():
            plan_rows.append(
                {
                    "schema_version": 1,
                    "paired_group_sha256": group,
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
                    "prefix_token_ids_sha256": row["prefix_token_ids_sha256"],
                    "boundary_token_id": row["boundary_token_id"],
                    "intervention_id": intervention.id,
                    "reference_intervention_id": "native_k_native_v",
                    "max_new_tokens": int(prompt["max_new_tokens"]),
                    "do_sample": False,
                }
            )
    plan_ref = atomic_jsonl(output_dir / "paired_run_plan.jsonl", plan_rows)
    lengths = [int(row["prompt_token_count"]) for row in rows]
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "kind": "a1_oracle_cpu_prepared_registry",
        "status": "complete",
        "gpu_used": False,
        "model_loaded": False,
        "network_used": False,
        "tier": "method-dev",
        "splits_opened": ["hotpotqa_train", "musique_train_screen"],
        "dev_or_locked_split_opened": False,
        "config": artifact_ref(root / "config.json").to_dict(),
        "code": code_artifacts(root),
        "mapped_cache_manifest": artifact_ref(mapper_manifest_path).to_dict(),
        "records": records_ref.to_dict(),
        "interventions": interventions_ref.to_dict(),
        "paired_run_plan": plan_ref.to_dict(),
        "dataset_counts": {"hotpotqa_train": len(hotpot), "musique_train": len(musique)},
        "record_count": len(rows),
        "intervention_count": len(intervention_catalog()),
        "paired_plan_count": len(plan_rows),
        "selection_unit": "complete_question_sequence",
        "hotpot_selection_rule": "ascending_sha256(namespace + NUL + record_id), then source_index",
        "hotpot_selection_namespace": hotpot_spec["namespace"],
        "musique_selection_registry_sha256": musique_spec["selection_registry"]["sha256"],
        "cross_dataset_exact_normalized_question_overlap": 0,
        "hotpot_selection_diagnostics": hotpot_diagnostics,
        "prompt_template_id": prompt["template_id"],
        "chat_template": prompt["chat_template"],
        "context_truncation": prompt["context_truncation"],
        "truncated_count": 0,
        "prompt_token_count": {
            "min": min(lengths),
            "median": _median(lengths),
            "max": max(lengths),
        },
        "max_new_tokens": int(prompt["max_new_tokens"]),
        "same_prompt_token_ids_across_interventions": True,
        "cache_boundary": "cache C[:-1], compose target-format K/V, replay C[-1] once",
        "accessed_input_paths": sorted(set(accessed)),
    }
    manifest_ref = atomic_json(output_dir / "prepare_manifest.json", manifest)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "gpu_used": False,
        "cuda_initialized": False,
        "model_loaded": False,
        "dev_or_locked_split_opened": False,
        "output_directory": str(output_dir),
        "manifest": manifest_ref.to_dict(),
        "artifacts": {
            "records": records_ref.to_dict(),
            "interventions": interventions_ref.to_dict(),
            "paired_run_plan": plan_ref.to_dict(),
        },
    }
    receipt_ref = atomic_json(output_dir / "prepare_receipt.json", receipt)
    return {
        "status": "complete",
        "gpu_used": False,
        "record_count": len(rows),
        "paired_plan_count": len(plan_rows),
        "manifest": manifest_ref.to_dict(),
        "receipt": receipt_ref.to_dict(),
    }


def verify_artifact_reference(value: Mapping[str, Any], *, label: str) -> Path:
    if set(value) != {"path", "nbytes", "sha256"}:
        raise PreparationError(f"{label} artifact reference is malformed")
    path = Path(str(value["path"])).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise PreparationError(f"{label} prepared artifact escapes A1 root") from exc
    return verify_pinned_file(value, label=label)


def verify_marker(reference: Mapping[str, Any], *, label: str) -> None:
    marker_path = Path(str(reference["path"]) + ".sha256.json")
    if not marker_path.is_file() or marker_path.is_symlink():
        raise PreparationError(f"{label} SHA marker is missing")
    marker = read_json(marker_path)
    expected = {"schema_version": 1, **dict(reference)}
    if marker != expected:
        raise PreparationError(f"{label} SHA marker disagrees with its artifact reference")


def validate_prepared(output: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output_dir = require_output(output)
    manifest_path = output_dir / "prepare_manifest.json"
    receipt_path = output_dir / "prepare_receipt.json"
    if not manifest_path.is_file() or not receipt_path.is_file():
        raise PreparationError("prepared manifest/receipt is missing")
    manifest = read_json(manifest_path)
    receipt = read_json(receipt_path)
    verify_marker(artifact_ref(manifest_path).to_dict(), label="prepared.manifest")
    verify_marker(artifact_ref(receipt_path).to_dict(), label="prepared.receipt")
    if manifest.get("status") != "complete" or receipt.get("status") != "complete":
        raise PreparationError("prepared artifacts are not complete")
    if manifest.get("gpu_used") is not False or receipt.get("gpu_used") is not False:
        raise PreparationError("CPU prepare artifact claims GPU use")
    if manifest.get("dev_or_locked_split_opened") is not False or receipt.get(
        "dev_or_locked_split_opened"
    ) is not False:
        raise PreparationError("prepared artifact claims dev/locked access")
    records_path = verify_artifact_reference(manifest["records"], label="prepared.records")
    interventions_path = verify_artifact_reference(
        manifest["interventions"], label="prepared.interventions"
    )
    plan_path = verify_artifact_reference(manifest["paired_run_plan"], label="prepared.plan")
    verify_marker(manifest["records"], label="prepared.records")
    verify_marker(manifest["interventions"], label="prepared.interventions")
    verify_marker(manifest["paired_run_plan"], label="prepared.plan")
    # ``prepare_v1`` is an immutable historical registry.  Later independent
    # v2 runner files and documentation must not make its original receipt
    # appear corrupt merely because the working tree evolved.  Verify the
    # complete recorded inventory and its rooted paths here; a caller that
    # needs current executable identity binds it separately (v2 does so in
    # its run definition).
    expected_paths = set(code_artifacts(ROOT))
    recorded_code = manifest.get("code")
    if not isinstance(recorded_code, dict) or set(recorded_code) != expected_paths:
        raise PreparationError("prepared code inventory drift")
    for relative, reference in recorded_code.items():
        if not isinstance(reference, Mapping) or set(reference) != {"path", "nbytes", "sha256"}:
            raise PreparationError(f"prepared code reference malformed: {relative}")
        path = Path(str(reference["path"])).resolve()
        if not isinstance(reference["nbytes"], int) or SHA256_RE.fullmatch(str(reference["sha256"])) is None:
            raise PreparationError(f"prepared code identity malformed: {relative}")
        if path != (ROOT / relative).resolve():
            raise PreparationError(f"prepared code path drift: {relative}")
    rows = read_jsonl(records_path)
    plans = read_jsonl(plan_path)
    interventions = read_json(interventions_path)
    if len(rows) != manifest.get("record_count") or len(rows) != 512:
        raise PreparationError("prepared record count drift")
    if len(plans) != manifest.get("paired_plan_count") or len(plans) != 512 * 11:
        raise PreparationError("prepared paired plan count drift")
    expected_ids = [value.id for value in intervention_catalog()]
    if [value["id"] for value in interventions.get("interventions", [])] != expected_ids:
        raise PreparationError("prepared intervention catalog drift")
    records_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("tier") != "method-dev" or row.get("split") != "train_screen":
            raise PreparationError("prepared record is not train method-dev")
        if row.get("dataset") not in {"hotpotqa_train", "musique_train"}:
            raise PreparationError("prepared record dataset is not allowlisted")
        ids = row.get("prompt_token_ids")
        if not isinstance(ids, list) or token_ids_sha256(ids) != row.get("prompt_token_ids_sha256"):
            raise PreparationError("prepared prompt token commitment drift")
        if token_ids_sha256(ids[:-1]) != row.get("prefix_token_ids_sha256"):
            raise PreparationError("prepared prefix token commitment drift")
        if ids[-1] != row.get("boundary_token_id") or len(ids) != row.get("prompt_token_count"):
            raise PreparationError("prepared boundary/length drift")
        context = row.get("cache_context")
        if not isinstance(context, dict) or context.get("prompt_token_ids_sha256") != row.get(
            "prompt_token_ids_sha256"
        ):
            raise PreparationError("prepared cache context drift")
        key = (row["dataset"], row["sample_id"])
        if key in records_by_key:
            raise PreparationError("prepared sample key is duplicated")
        records_by_key[key] = row
    groups: dict[tuple[str, str], list[str]] = {}
    for plan in plans:
        key = (plan.get("dataset"), plan.get("sample_id"))
        if key not in records_by_key:
            raise PreparationError("paired plan references an unknown sample")
        row = records_by_key[key]
        if plan.get("prompt_token_ids_sha256") != row["prompt_token_ids_sha256"]:
            raise PreparationError("paired plan prompt commitment drift")
        groups.setdefault(key, []).append(str(plan.get("intervention_id")))
    if any(values != expected_ids for values in groups.values()) or set(groups) != set(records_by_key):
        raise PreparationError("paired plan does not contain all 11 ordered interventions per sample")
    return {
        "status": "complete",
        "record_count": len(rows),
        "paired_plan_count": len(plans),
        "manifest_sha256": sha256_file(manifest_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def validate_mapper_manifest(
    config: Mapping[str, Any], *, deep_layer_hashes: bool
) -> dict[str, Any]:
    manifest_path = verify_pinned_file(
        config["mapped_cache"]["manifest"], label="mapped_cache.manifest"
    )
    manifest = read_json(manifest_path)
    architecture = manifest.get("architecture", {})
    if (
        manifest.get("status") != "complete"
        or manifest.get("direction") != "qwen3_32b_to_14b"
        or architecture.get("source_layers") != 64
        or architecture.get("target_layers") != 40
        or architecture.get("selected_source_layers") != 20
        or architecture.get("kv_heads") != 8
        or architecture.get("head_dim") != 128
        or manifest.get("weight_count") != 1_677_721_600
        or float(manifest.get("ridge_lambda")) != 0.01
    ):
        raise PreparationError("k20 mapper manifest contract drift")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise PreparationError("k20 mapper does not publish exactly 40 target layers")
    mapper_root = manifest_path.parent.resolve()
    bytes_hashed = 0
    for expected_layer, layer in enumerate(layers):
        if not isinstance(layer, dict) or layer.get("target_layer") != expected_layer:
            raise PreparationError("k20 mapper target layers are not ordered 0..39")
        selected = layer.get("source_layers_rank_order")
        if (
            not isinstance(selected, list)
            or len(selected) != 20
            or len(set(selected)) != 20
            or any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < 64 for value in selected)
        ):
            raise PreparationError(f"k20 mapper layer {expected_layer} source selection is malformed")
        for path_key, sha_key, suffix in (
            ("path", "sha256", ".safetensors"),
            ("sidecar_path", "sidecar_sha256", ".json"),
        ):
            path = Path(str(layer.get(path_key, "")))
            _reject_symlink_components(path, f"mapper layer {expected_layer} {path_key}")
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(mapper_root)
            except ValueError as exc:
                raise PreparationError(f"mapper layer {expected_layer} escapes mapper root") from exc
            if not relative.parts or not resolved.is_file() or resolved.suffix != suffix:
                raise PreparationError(f"mapper layer {expected_layer} {path_key} is invalid")
            expected_hash = layer.get(sha_key)
            if not isinstance(expected_hash, str) or SHA256_RE.fullmatch(expected_hash) is None:
                raise PreparationError(f"mapper layer {expected_layer} {sha_key} is malformed")
            if deep_layer_hashes:
                observed = sha256_file(resolved)
                bytes_hashed += resolved.stat().st_size
                if observed != expected_hash:
                    raise PreparationError(
                        f"mapper layer {expected_layer} {path_key} SHA drift: {observed}"
                    )
    return {
        "manifest": artifact_ref(manifest_path).to_dict(),
        "layer_count": len(layers),
        "deep_layer_hashes": deep_layer_hashes,
        "layer_payload_bytes_hashed": bytes_hashed,
    }


def deep_preflight(
    root: str | Path = ROOT,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    deep_mapper_hashes: bool = True,
    publish_receipt: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    if root != ROOT.resolve():
        raise PreparationError("preflight root must be exactly 01_oracle_diagnostics")
    config = load_config(root / "config.json")
    _validate_intervention_config(config)
    pinned: dict[str, dict[str, Any]] = {}
    for label, spec in config["protocol_inputs"].items():
        path = verify_pinned_file(spec, label=f"protocol_inputs.{label}")
        pinned[f"protocol_inputs.{label}"] = artifact_ref(path).to_dict()
    for role in ("source", "target"):
        for key in ("config", "tokenizer_json", "tokenizer_config"):
            path = verify_pinned_file(config["models"][role][key], label=f"models.{role}.{key}")
            pinned[f"models.{role}.{key}"] = artifact_ref(path).to_dict()
        model_config = read_json(config["models"][role]["config"]["path"])
        try:
            validate_qwen3_config(model_config, source=role == "source")
        except OracleContractError as exc:
            raise PreparationError(f"{role} model config mismatch: {exc}") from exc
    if config["models"]["source"]["tokenizer_json"]["sha256"] != config["models"]["target"][
        "tokenizer_json"
    ]["sha256"]:
        raise PreparationError("source/target tokenizer JSON differs")
    for dataset, specification in config["method_dev_data"].items():
        path = verify_pinned_file(
            {key: specification[key] for key in ("path", "nbytes", "sha256")},
            label=f"method_dev_data.{dataset}",
        )
        _train_only_path(path, dataset=dataset, split=specification["split"], tier=specification["tier"])
        pinned[f"method_dev_data.{dataset}"] = artifact_ref(path).to_dict()
        if dataset == "musique_train":
            registry = verify_pinned_file(
                specification["selection_registry"],
                label="method_dev_data.musique_train.selection_registry",
            )
            pinned["method_dev_data.musique_train.selection_registry"] = artifact_ref(
                registry
            ).to_dict()
    metric = verify_pinned_file(
        {key: config["metric_reference"][key] for key in ("path", "nbytes", "sha256")},
        label="metric_reference",
    )
    pinned["metric_reference"] = artifact_ref(metric).to_dict()
    mapper = validate_mapper_manifest(config, deep_layer_hashes=deep_mapper_hashes)
    prepared = validate_prepared(output)
    receipt = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "kind": "a1_oracle_cpu_deep_preflight",
        "status": "pass",
        "gpu_used": False,
        "cuda_initialized": False,
        "model_loaded": False,
        "network_used": False,
        "dev_or_locked_split_opened": False,
        "authorized_for_gpu_execution": False,
        "config": artifact_ref(root / "config.json").to_dict(),
        "code": code_artifacts(root),
        "pinned_inputs": pinned,
        "mapper_validation": mapper,
        "prepared_validation": prepared,
        "cache_contract": config["cache_contract"],
        "intervention_ids": [value.id for value in intervention_catalog()],
        "prompt_protocol": config["prompt_protocol"],
    }
    if publish_receipt:
        output_dir = require_output(output)
        receipt_ref = atomic_json(output_dir / "preflight_receipt.json", receipt)
        receipt["receipt"] = receipt_ref.to_dict()
    return receipt


__all__ = [
    "ArtifactRef",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "PreparationError",
    "ROOT",
    "atomic_json",
    "atomic_jsonl",
    "canonical_bytes",
    "code_artifacts",
    "deep_preflight",
    "hotpot_prompt",
    "intervention_catalog",
    "iter_json_array",
    "load_config",
    "load_hotpot_selection",
    "load_musique_selection",
    "musique_prompt",
    "prepare",
    "read_json",
    "read_jsonl",
    "require_output",
    "sha256_file",
    "validate_hotpot_record",
    "validate_mapper_manifest",
    "validate_musique_record",
    "validate_prepared",
    "verify_pinned_file",
]
