"""Fail-closed MuSiQue data registry and official answer metrics.

This module intentionally has no model or CUDA dependency. It freezes data
before any candidate is evaluated and keeps the official dev set sealed until
the method nomination step.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


class MuSiQueError(RuntimeError):
    """Raised when a pinned input or registry invariant is violated."""


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    marker = {
        "schema_version": 1,
        "path": path.name,
        "nbytes": len(payload),
        "sha256": sha256_bytes(payload),
    }
    marker_path = path.with_name(path.name + ".sha256.json")
    marker_payload = canonical_bytes(marker)
    fd, temporary = tempfile.mkstemp(prefix=f".{marker_path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(marker_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, marker_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return marker


def load_config(root: Path) -> dict[str, Any]:
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise MuSiQueError("unsupported config schema")
    return config


def verify_pinned_file(root: Path, specification: Mapping[str, Any]) -> Path:
    path = root / str(specification["relative_path"])
    if path.is_symlink() or not path.is_file():
        raise MuSiQueError(f"pinned input is not a regular file: {path}")
    size = path.stat().st_size
    if size != int(specification["nbytes"]):
        raise MuSiQueError(f"pinned input size drift: {path}: {size}")
    actual = sha256_file(path)
    if actual != specification["sha256"]:
        raise MuSiQueError(f"pinned input hash drift: {path}: {actual}")
    return path


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for index, raw in enumerate(handle):
            if not raw.strip():
                raise MuSiQueError(f"blank JSONL row at {path}:{index + 1}")
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise MuSiQueError(f"invalid JSON at {path}:{index + 1}") from error
            if not isinstance(record, dict):
                raise MuSiQueError(f"non-object JSON at {path}:{index + 1}")
            yield index, record


def _validate_answerable_record(record: Mapping[str, Any], *, require_answer: bool) -> None:
    required = {"id", "question", "paragraphs"}
    if require_answer:
        required |= {"answer", "answer_aliases", "answerable", "question_decomposition"}
    missing = required - set(record)
    if missing:
        raise MuSiQueError(f"record missing fields: {sorted(missing)}")
    if not isinstance(record["id"], str) or not record["id"]:
        raise MuSiQueError("record id must be a non-empty string")
    if not isinstance(record["question"], str) or not record["question"]:
        raise MuSiQueError(f"record {record['id']} has invalid question")
    paragraphs = record["paragraphs"]
    if not isinstance(paragraphs, list) or not 1 <= len(paragraphs) <= 20:
        raise MuSiQueError(f"record {record['id']} must contain 1..20 paragraphs")
    seen_indices: set[int] = set()
    for paragraph in paragraphs:
        if set(paragraph) != {"idx", "title", "paragraph_text", "is_supporting"}:
            raise MuSiQueError(f"record {record['id']} paragraph schema drift")
        idx = paragraph["idx"]
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < len(paragraphs):
            raise MuSiQueError(f"record {record['id']} has invalid paragraph index")
        seen_indices.add(idx)
        if not isinstance(paragraph["title"], str) or not isinstance(
            paragraph["paragraph_text"], str
        ):
            raise MuSiQueError(f"record {record['id']} has invalid paragraph text")
        if not isinstance(paragraph["is_supporting"], bool):
            raise MuSiQueError(f"record {record['id']} has invalid support label")
    if seen_indices != set(range(len(paragraphs))):
        raise MuSiQueError(f"record {record['id']} paragraph indices are not contiguous")
    if require_answer:
        if record["answerable"] is not True:
            raise MuSiQueError(f"answerable split contains unanswerable row {record['id']}")
        if not isinstance(record["answer"], str) or not record["answer"]:
            raise MuSiQueError(f"record {record['id']} has invalid answer")
        aliases = record["answer_aliases"]
        if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
            raise MuSiQueError(f"record {record['id']} has invalid aliases")


def _normalized_question(text: str) -> str:
    return " ".join(text.casefold().split())


def _record_identity(index: int, record: Mapping[str, Any]) -> dict[str, Any]:
    canonical = canonical_bytes(record)
    return {
        "source_index": index,
        "id": record["id"],
        "record_sha256": sha256_bytes(canonical),
        "question_sha256": sha256_bytes(record["question"].encode("utf-8")),
    }


def _rank(namespace: str, record_id: str, source_index: int) -> tuple[str, int]:
    value = hashlib.sha256(
        namespace.encode("utf-8") + b"\0" + record_id.encode("utf-8")
    ).hexdigest()
    return value, source_index


def _digest_identities(values: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_bytes(list(values)))


def prepare_registry(root: Path, output_dir: Path) -> str:
    root = root.resolve()
    output_dir = output_dir.resolve()
    expected_parent = (root / "artifacts").resolve()
    if output_dir == expected_parent or expected_parent not in output_dir.parents:
        raise MuSiQueError("registry output must be a child of this benchmark's artifacts directory")
    if output_dir.is_symlink():
        raise MuSiQueError("registry output cannot be a symlink")

    config = load_config(root)
    source = config["source"]
    archive = verify_pinned_file(root, source["download_archive"])
    train_path = verify_pinned_file(root, source["answerable_train"])
    dev_path = verify_pinned_file(root, source["answerable_dev"])
    leakage_path = verify_pinned_file(root, source["single_hop_leakage_registry"])
    metric_path = root / config["official_metric"]["implementation"]
    if sha256_file(metric_path) != config["official_metric"]["implementation_sha256"]:
        raise MuSiQueError("official metric implementation drift")

    train_values: list[tuple[int, dict[str, Any]]] = []
    train_ids: set[str] = set()
    train_question_counts: Counter[str] = Counter()
    for index, record in iter_jsonl(train_path):
        _validate_answerable_record(record, require_answer=True)
        if record["id"] in train_ids:
            raise MuSiQueError(f"duplicate train id: {record['id']}")
        train_ids.add(record["id"])
        normalized = _normalized_question(record["question"])
        train_question_counts[normalized] += 1
        train_values.append((index, record))
    if len(train_values) != source["answerable_train"]["count"]:
        raise MuSiQueError("train row count drift")

    dev_identities: list[dict[str, Any]] = []
    dev_ids: set[str] = set()
    dev_question_counts: Counter[str] = Counter()
    for index, record in iter_jsonl(dev_path):
        _validate_answerable_record(record, require_answer=True)
        if record["id"] in dev_ids:
            raise MuSiQueError(f"duplicate dev id: {record['id']}")
        dev_ids.add(record["id"])
        normalized = _normalized_question(record["question"])
        dev_question_counts[normalized] += 1
        dev_identities.append(_record_identity(index, record))
    if len(dev_identities) != source["answerable_dev"]["count"]:
        raise MuSiQueError("dev row count drift")
    if train_ids & dev_ids:
        raise MuSiQueError("train/dev record-id overlap")
    if set(train_question_counts) & set(dev_question_counts):
        raise MuSiQueError("train/dev exact normalized-question overlap")

    split_policy = config["split_policy"]
    screen_spec = split_policy["train_screen"]
    confirm_spec = split_policy["train_confirm"]
    # Repeated normalized questions are quarantined from method selection so a
    # paraphrase-equivalent group can never straddle screen and confirmation.
    eligible_train_values = [
        item
        for item in train_values
        if train_question_counts[_normalized_question(item[1]["question"])] == 1
    ]
    ranked = sorted(
        eligible_train_values,
        key=lambda item: _rank(screen_spec["namespace"], item[1]["id"], item[0]),
    )
    screen_pairs = ranked[: int(screen_spec["count"])]
    used_ids = {record["id"] for _, record in screen_pairs}
    confirm_ranked = sorted(
        (item for item in eligible_train_values if item[1]["id"] not in used_ids),
        key=lambda item: _rank(confirm_spec["namespace"], item[1]["id"], item[0]),
    )
    confirm_pairs = confirm_ranked[: int(confirm_spec["count"])]
    screen = [_record_identity(index, record) for index, record in screen_pairs]
    confirm = [_record_identity(index, record) for index, record in confirm_pairs]
    if {value["id"] for value in screen} & {value["id"] for value in confirm}:
        raise MuSiQueError("method-dev split overlap")

    config_sha = sha256_file(root / "config.json")
    common = {
        "schema_version": 1,
        "benchmark_id": config["benchmark_id"],
        "config_sha256": config_sha,
        "source_commit": source["repository"] + "@" + source["commit"],
        "archive": {"nbytes": archive.stat().st_size, "sha256": sha256_file(archive)},
        "official_metric_sha256": sha256_file(metric_path),
        "single_hop_leakage_registry_sha256": sha256_file(leakage_path),
    }
    screen_payload = {
        **common,
        "kind": "musique_method_dev_selection",
        "split": "train_screen",
        "tier": "method-dev",
        "namespace": screen_spec["namespace"],
        "count": len(screen),
        "records_sha256": _digest_identities(screen),
        "records": screen,
    }
    confirm_payload = {
        **common,
        "kind": "musique_method_dev_selection",
        "split": "train_confirm",
        "tier": "method-dev",
        "namespace": confirm_spec["namespace"],
        "count": len(confirm),
        "records_sha256": _digest_identities(confirm),
        "records": confirm,
    }
    locked_payload = {
        **common,
        "kind": "musique_locked_test_commitment",
        "split": "official_dev",
        "tier": "locked-test",
        "state": "sealed_until_candidate_freeze",
        "access_budget": split_policy["official_dev"]["access_budget"],
        "count": len(dev_identities),
        "payload_sha256": source["answerable_dev"]["sha256"],
        "record_identities_sha256": _digest_identities(dev_identities),
        "record_ids_sha256": sha256_bytes(
            canonical_bytes([value["id"] for value in dev_identities])
        ),
        "records_embedded": False,
        "prelock_disclosure": split_policy["official_dev"]["disclosure"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    markers = {
        "train_screen": atomic_json(output_dir / "train_screen.json", screen_payload),
        "train_confirm": atomic_json(output_dir / "train_confirm.json", confirm_payload),
        "official_dev_locked": atomic_json(output_dir / "official_dev_locked.json", locked_payload),
    }
    manifest = {
        **common,
        "kind": "musique_data_registry",
        "status": "complete",
        "gpu_used": False,
        "train_count": len(train_values),
        "dev_count": len(dev_identities),
        "exact_id_overlap_count": 0,
        "exact_normalized_question_overlap_count": 0,
        "train_duplicate_normalized_question_groups_quarantined": sum(
            count > 1 for count in train_question_counts.values()
        ),
        "train_duplicate_normalized_question_rows_quarantined": sum(
            count for count in train_question_counts.values() if count > 1
        ),
        "dev_duplicate_normalized_question_groups": sum(
            count > 1 for count in dev_question_counts.values()
        ),
        "selections": markers,
        "locked_outcomes_evaluated": False,
    }
    marker = atomic_json(output_dir / "registry_manifest.json", manifest)
    return json.dumps(
        {
            "status": "complete",
            "gpu_used": False,
            "manifest": str(output_dir / "registry_manifest.json"),
            "manifest_sha256": marker["sha256"],
        },
        sort_keys=True,
    )


def normalize_answer(text: str) -> str:
    """Exact copy of the official answer normalization semantics."""

    lowered = text.lower()
    punctuation = set(string.punctuation)
    without_punctuation = "".join(character for character in lowered if character not in punctuation)
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation, flags=re.UNICODE)
    return " ".join(without_articles.split())


def answer_exact(gold: str, prediction: str) -> int:
    return int(normalize_answer(gold) == normalize_answer(prediction))


def answer_f1(gold: str, prediction: str) -> float:
    gold_tokens = normalize_answer(gold).split() if gold else []
    prediction_tokens = normalize_answer(prediction).split() if prediction else []
    common = Counter(gold_tokens) & Counter(prediction_tokens)
    count = sum(common.values())
    if not gold_tokens or not prediction_tokens:
        return float(gold_tokens == prediction_tokens)
    if count == 0:
        return 0.0
    precision = count / len(prediction_tokens)
    recall = count / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction: str, answer: str, aliases: Iterable[str]) -> dict[str, float]:
    references = [answer, *aliases]
    if not references or not all(isinstance(value, str) for value in references):
        raise MuSiQueError("answer references must be strings")
    return {
        "answer_em": float(max(answer_exact(reference, prediction) for reference in references)),
        "answer_f1": max(answer_f1(reference, prediction) for reference in references),
    }


def build_prompt(record: Mapping[str, Any]) -> str:
    _validate_answerable_record(record, require_answer=True)
    paragraphs = sorted(record["paragraphs"], key=lambda value: value["idx"])
    context = "\n\n".join(
        f"[{paragraph['idx']}] {paragraph['title']}\n{paragraph['paragraph_text']}"
        for paragraph in paragraphs
    )
    return (
        "Answer the multi-hop question using only the supplied context. "
        "Return only the shortest answer span and no explanation.\n\n"
        f"Context:\n{context}\n\nQuestion: {record['question']}\nAnswer:"
    )
