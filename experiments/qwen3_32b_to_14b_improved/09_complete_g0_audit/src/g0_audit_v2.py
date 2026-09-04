from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tokenizers import Tokenizer


class G0V2Error(RuntimeError):
    """Fail-closed error raised before a G0 v2 claim can be emitted."""


EXPECTED_CONSUMER_CONTRACTS = {
    "g0_consumer_contract",
    "native_registry_consumer_contract",
    "dir13_g0_consumer_contract",
    "dir13_native_registry_consumer_contract",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def ordered_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Formatting-independent, order-sensitive digest of the complete row list."""
    return digest_bytes(canonical_bytes([dict(row) for row in rows]))


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def sidecar(path: Path) -> dict[str, Any]:
    return {"path": str(path), "nbytes": path.stat().st_size, "sha256": sha256_file(path)}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(character if character.isalnum() else " " for character in text).split())


def shingles(text: str, width: int) -> frozenset[str]:
    words = text.split()
    if len(words) < width:
        return frozenset()
    return frozenset(digest("\u241f".join(words[index : index + width]))[:24] for index in range(len(words) - width + 1))


_FINGERPRINT_CHARS = 16
_ROLLING_BASE = 1_000_003
_ROLLING_MASK = (1 << 64) - 1


def substring_fingerprints(text: str, minimum_characters: int) -> tuple[int, ...]:
    """Winnow character hashes with complete recall for the substring gate.

    Any shared substring of at least ``minimum_characters`` contains an entire
    window of ``minimum_characters - k + 1`` identical k-gram hashes.  The
    right-most minimum selected for that same complete window is therefore
    shared by both strings.  Literal containment is always checked after
    retrieval, so rolling-hash collisions cannot create a flag.
    """
    if minimum_characters <= 0:
        raise G0V2Error("substring minimum must be positive")
    if len(text) < minimum_characters:
        return ()
    width = min(_FINGERPRINT_CHARS, minimum_characters)
    window = minimum_characters - width + 1
    power = pow(_ROLLING_BASE, width - 1, 1 << 64)
    value = 0
    for character in text[:width]:
        value = (value * _ROLLING_BASE + ord(character) + 1) & _ROLLING_MASK
    hashes = [value]
    for index in range(width, len(text)):
        outgoing = (ord(text[index - width]) + 1) * power
        value = ((value - outgoing) * _ROLLING_BASE + ord(text[index]) + 1) & _ROLLING_MASK
        hashes.append(value)
    queue: deque[tuple[int, int]] = deque()
    selected: set[int] = set()
    for index, fingerprint in enumerate(hashes):
        while queue and queue[-1][1] >= fingerprint:
            queue.pop()
        queue.append((index, fingerprint))
        first_allowed = index - window + 1
        while queue[0][0] < first_allowed:
            queue.popleft()
        if first_allowed >= 0:
            selected.add(queue[0][1])
    return tuple(sorted(selected))


def _reject_placeholder(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_placeholder(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_placeholder(child, f"{label}[{index}]")
    elif isinstance(value, str):
        folded = value.casefold()
        if (value.startswith("__") and value.endswith("__")) or "placeholder" in folded:
            raise G0V2Error(f"unfrozen placeholder: {label}")


def require_frozen_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 2 or config.get("status") != "frozen_after_registry_publication":
        raise G0V2Error("G0 v2 config is not frozen after registry publication")
    _reject_placeholder(config, "config")
    consumer_contracts = config.get("consumer_contracts")
    if not isinstance(consumer_contracts, Mapping) or set(consumer_contracts) != EXPECTED_CONSUMER_CONTRACTS:
        raise G0V2Error("G0 v2 consumer contract inventory is incomplete or has extras")
    for label, spec in iter_file_specs(config):
        checksum = spec.get("sha256")
        nbytes = spec.get("nbytes")
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum) or checksum == "0" * 64:
            raise G0V2Error(f"unfrozen SHA-256: {label}")
        if not isinstance(nbytes, int) or nbytes <= 0:
            raise G0V2Error(f"unfrozen byte count: {label}")
    for filename, spec in config["registry"]["files"].items():
        ordered = spec.get("ordered_rows_sha256")
        if not isinstance(ordered, str) or not re.fullmatch(r"[0-9a-f]{64}", ordered) or ordered == "0" * 64:
            raise G0V2Error(f"unfrozen ordered row digest: {filename}")


def iter_file_specs(config: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for name, spec in sorted(config["inputs"].items()):
        yield f"inputs.{name}", spec
    for name, spec in sorted(config["consumer_contracts"].items()):
        yield f"consumer_contracts.{name}", spec
    for name in (
        "published_marker", "manifest", "receipt", "independent_audit",
        "strict_capacity_failure", "strict_sequence_partition_failure",
    ):
        yield f"registry.{name}", config["registry"][name]
    for name, spec in sorted(config["registry"]["files"].items()):
        yield f"registry.files.{name}", spec


def artifact_ref(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Project a richer pinned-file specification to the shared byte reference."""

    return {"path": spec["path"], "nbytes": spec["nbytes"], "sha256": spec["sha256"]}


def native_candidate_registry_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Emit exactly the native experiment-10 binding consumed by experiment 12."""

    registry = config["registry"]
    expected_files = {
        "hotpotqa_screen.jsonl",
        "hotpotqa_confirm.jsonl",
        "musique_screen.jsonl",
        "musique_confirm.jsonl",
    }
    files = registry["files"]
    if set(files) != expected_files:
        raise G0V2Error("native registry split artifact set changed")
    return {
        "published_marker": artifact_ref(registry["published_marker"]),
        "registry_manifest": artifact_ref(registry["manifest"]),
        "publication_receipt": artifact_ref(registry["receipt"]),
        "split_artifacts": {
            name.removesuffix(".jsonl"): artifact_ref(files[f"{name}.jsonl"])
            for name in (
                "hotpotqa_screen",
                "hotpotqa_confirm",
                "musique_screen",
                "musique_confirm",
            )
        },
    }


def verify_file(path: Path, spec: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not path.is_file():
        raise G0V2Error(f"missing pinned input: {label}")
    observed = {"nbytes": path.stat().st_size, "sha256": sha256_file(path)}
    if observed != {"nbytes": spec["nbytes"], "sha256": spec["sha256"]}:
        raise G0V2Error(f"pinned input mismatch: {label}")
    return observed


@dataclass(frozen=True)
class Projection:
    dataset_id: str
    sample_id: str
    source_index: int
    stratum: str
    question_text: str
    passages: tuple[str, ...]
    projection_sha256: str


@dataclass(frozen=True)
class SelectedRecord:
    projection: Projection
    partition: str
    registry_filename: str

    @property
    def dataset_id(self) -> str:
        return self.projection.dataset_id

    @property
    def record_identity(self) -> str:
        return digest(f"method-dev\u241f{self.dataset_id}\u241f{self.partition}\u241f{self.projection.sample_id}")


@dataclass(frozen=True)
class Unit:
    dataset_id: str
    partition: str
    record_identity: str
    unit_identity: str
    unit_kind: str
    text: str


@dataclass(frozen=True)
class UnitIndex:
    units: tuple[Unit, ...]
    texts: tuple[str, ...]
    shingle_sets: tuple[frozenset[str], ...]
    exact_inverted: Mapping[str, tuple[int, ...]]
    shingle_inverted: Mapping[str, tuple[int, ...]]
    fingerprint_inverted: Mapping[int, tuple[int, ...]]


_HOTPOT_KEYS = frozenset({"_id", "question", "context", "type", "level"})
_MUSIQUE_KEYS = frozenset({"id", "question", "paragraphs", "idx", "title", "paragraph_text"})


def _project_pairs(allowed: frozenset[str]):
    def project(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        return {key: value for key, value in pairs if key in allowed}
    return project


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise G0V2Error(f"{label} must be a non-empty string")
    return value


def _projection_sha(dataset_family: str, sample_id: str, question_text: str, passages: Sequence[str], stratum: str) -> str:
    return digest_bytes(canonical_bytes({
        "dataset_id": dataset_family,
        "sample_id": sample_id,
        "question": question_text,
        "passages": list(passages),
        "stratum": stratum,
    }))


def _hotpot_passages(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise G0V2Error("HotpotQA context projection mismatch")
    passages = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
            raise G0V2Error("HotpotQA passage projection mismatch")
        title = _required_text(row[0], "HotpotQA title")
        passages.append(" ".join([title, *(str(sentence) for sentence in row[1])]))
    return tuple(passages)


def _musique_passages(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise G0V2Error("MuSiQue paragraph projection mismatch")
    ordered = sorted(value, key=lambda row: int(row.get("idx", -1)))
    if [int(row.get("idx", -1)) for row in ordered] != list(range(len(ordered))):
        raise G0V2Error("MuSiQue paragraph indices are not contiguous")
    passages = []
    for row in ordered:
        if set(row) - {"idx", "title", "paragraph_text"}:
            raise G0V2Error("MuSiQue strict paragraph projection mismatch")
        passages.append(" ".join([str(row.get("title", "")), _required_text(row.get("paragraph_text"), "MuSiQue paragraph")]))
    return tuple(passages)


def _musique_hop(sample_id: str) -> str:
    match = re.match(r"^([234])hop(?:\d+)?__", sample_id)
    if not match:
        raise G0V2Error(f"MuSiQue id does not encode hop stratum: {sample_id}")
    return f"{match.group(1)}hop"


def load_hotpot(path: Path, expected_records: int, *, locked: bool) -> list[Projection]:
    decoder = json.JSONDecoder(object_pairs_hook=_project_pairs(_HOTPOT_KEYS))
    rows = decoder.decode(path.read_text(encoding="utf-8"))
    if len(rows) != expected_records:
        raise G0V2Error("HotpotQA row-count mismatch")
    result = []
    seen: set[str] = set()
    for source_index, row in enumerate(rows):
        required = {"_id", "question", "context"} if locked else {"_id", "question", "context", "type", "level"}
        if not required.issubset(row) or set(row) - _HOTPOT_KEYS:
            raise G0V2Error("HotpotQA strict source projection mismatch")
        sample_id = _required_text(row["_id"], "HotpotQA id")
        if sample_id in seen:
            raise G0V2Error("duplicate HotpotQA id")
        seen.add(sample_id)
        question_text = _required_text(row["question"], "HotpotQA question")
        passages = _hotpot_passages(row["context"])
        stratum = "locked" if locked else f"{row['type']}/{row['level']}"
        dataset_id = "hotpotqa_locked_dev" if locked else "hotpotqa_train"
        result.append(Projection(dataset_id, sample_id, source_index, stratum, question_text, passages, _projection_sha("hotpotqa", sample_id, question_text, passages, stratum)))
    return result


def load_musique(path: Path, expected_records: int, *, locked: bool) -> list[Projection]:
    decoder = json.JSONDecoder(object_pairs_hook=_project_pairs(_MUSIQUE_KEYS))
    result = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for source_index, line in enumerate(stream):
            row = decoder.decode(line)
            if not {"id", "question", "paragraphs"}.issubset(row) or set(row) - _MUSIQUE_KEYS:
                raise G0V2Error("MuSiQue strict source projection mismatch")
            sample_id = _required_text(row["id"], "MuSiQue id")
            if sample_id in seen:
                raise G0V2Error("duplicate MuSiQue id")
            seen.add(sample_id)
            question_text = _required_text(row["question"], "MuSiQue question")
            passages = _musique_passages(row["paragraphs"])
            stratum = "locked" if locked else _musique_hop(sample_id)
            dataset_id = "musique_locked_dev" if locked else "musique_train"
            result.append(Projection(dataset_id, sample_id, source_index, stratum, question_text, passages, _projection_sha("musique", sample_id, question_text, passages, stratum)))
    if len(result) != expected_records:
        raise G0V2Error("MuSiQue row-count mismatch")
    return result


_REGISTRY_FIELDS = frozenset({
    "dataset_id", "partition", "sample_id", "source_index",
    "source_projection_sha256", "selection_rank", "selection_rank_sha256", "stratum",
})


def load_registry_file(path: Path, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != spec["expected_records"]:
        raise G0V2Error(f"registry row-count mismatch: {path.name}")
    for row in rows:
        if set(row) != _REGISTRY_FIELDS:
            raise G0V2Error(f"registry field allowlist mismatch: {path.name}")
        if row["dataset_id"] != spec["dataset_id"] or row["partition"] != spec["partition"]:
            raise G0V2Error(f"registry dataset/partition mismatch: {path.name}")
    stratum_counts = dict(sorted(Counter(row["stratum"] for row in rows).items()))
    if stratum_counts != spec.get("expected_stratum_counts"):
        raise G0V2Error(f"registry stratum coverage mismatch: {path.name}")
    if ordered_rows_sha256(rows) != spec["ordered_rows_sha256"]:
        raise G0V2Error(f"registry ordered row digest mismatch: {path.name}")
    return rows


def rehydrate_registry(
    registry_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    source_rows: Mapping[str, Sequence[Projection]],
    namespaces: Mapping[str, str],
) -> list[SelectedRecord]:
    result: list[SelectedRecord] = []
    identities: set[tuple[str, str]] = set()
    for filename, rows in sorted(registry_rows.items()):
        ranks: Counter[str] = Counter()
        for row in rows:
            dataset_id = row["dataset_id"]
            source_index = row["source_index"]
            if not isinstance(source_index, int) or not 0 <= source_index < len(source_rows[dataset_id]):
                raise G0V2Error(f"registry source index out of bounds: {filename}")
            source = source_rows[dataset_id][source_index]
            if source.sample_id != row["sample_id"] or source.stratum != row["stratum"] or source.projection_sha256 != row["source_projection_sha256"]:
                raise G0V2Error(f"registry source projection binding mismatch: {filename}")
            expected_rank_sha = digest(f"{namespaces[dataset_id]}\0{row['partition']}\0{row['stratum']}\0{row['sample_id']}")
            if row["selection_rank_sha256"] != expected_rank_sha or row["selection_rank"] != ranks[row["stratum"]]:
                raise G0V2Error(f"registry ordered rank binding mismatch: {filename}")
            ranks[row["stratum"]] += 1
            identity = (dataset_id, row["sample_id"])
            if identity in identities:
                raise G0V2Error("duplicate selected identity across four registry files")
            identities.add(identity)
            result.append(SelectedRecord(source, row["partition"], filename))
    return result


def record_units(records: Sequence[SelectedRecord | Projection], kind: str) -> list[Unit]:
    units = []
    for record in records:
        if isinstance(record, SelectedRecord):
            projection = record.projection
            partition = record.partition
            record_identity = record.record_identity
        else:
            projection = record
            partition = "locked"
            record_identity = digest(f"locked-test\u241f{projection.dataset_id}\u241f{projection.sample_id}")
        if kind == "question_unit":
            texts = (projection.question_text,)
        elif kind == "passage_unit":
            texts = projection.passages
        else:
            raise G0V2Error(f"unknown unit kind: {kind}")
        for index, text in enumerate(texts):
            units.append(Unit(
                projection.dataset_id,
                partition,
                record_identity,
                digest(f"{record_identity}\u241f{kind}\u241f{index}"),
                kind,
                normalize(text),
            ))
    return units


def mapper_units(texts: Sequence[str]) -> list[Unit]:
    return [
        Unit("fineweb_edu", "mapper-fit", digest(f"mapper-fit\u241f{index:06d}"), digest(f"mapper-fit\u241f{index:06d}\u241fsequence"), "mapper_sequence", normalize(text))
        for index, text in enumerate(texts)
    ]


def build_unit_index(units: Sequence[Unit], width: int, substring_minimum: int) -> UnitIndex:
    texts = tuple(unit.text for unit in units)
    shingle_sets = tuple(shingles(text, width) for text in texts)
    exact: dict[str, list[int]] = defaultdict(list)
    shingle_index: dict[str, list[int]] = defaultdict(list)
    fingerprint_index: dict[int, list[int]] = defaultdict(list)
    for index, text in enumerate(texts):
        exact[digest(text)].append(index)
        for value in sorted(shingle_sets[index]):
            shingle_index[value].append(index)
        for value in substring_fingerprints(text, substring_minimum):
            fingerprint_index[value].append(index)
    return UnitIndex(
        tuple(units), texts, shingle_sets,
        {key: tuple(value) for key, value in sorted(exact.items())},
        {key: tuple(value) for key, value in sorted(shingle_index.items())},
        {key: tuple(value) for key, value in sorted(fingerprint_index.items())},
    )


def compare_units(
    left: Sequence[Unit],
    right: Sequence[Unit],
    *,
    width: int,
    minimum_intersection: int,
    minimum_containment: float,
    minimum_jaccard: float,
    substring_minimum: int,
    label: str,
) -> dict[str, Any]:
    index = build_unit_index(right, width, substring_minimum)
    flags: list[dict[str, Any]] = []
    comparisons = 0
    for left_unit in left:
        left_set = shingles(left_unit.text, width)
        counts: Counter[int] = Counter()
        for value in sorted(left_set):
            counts.update(index.shingle_inverted.get(value, ()))
        candidates = set(index.exact_inverted.get(digest(left_unit.text), ()))
        candidates.update(counts)
        for value in substring_fingerprints(left_unit.text, substring_minimum):
            candidates.update(index.fingerprint_inverted.get(value, ()))
        for right_index in sorted(candidates):
            comparisons += 1
            right_unit = index.units[right_index]
            right_text = index.texts[right_index]
            reason = None
            intersection = counts.get(right_index, 0)
            containment = 0.0
            jaccard = 0.0
            if left_unit.text == right_text:
                reason = "normalized_exact"
            else:
                shorter, longer = (left_unit.text, right_text) if len(left_unit.text) <= len(right_text) else (right_text, left_unit.text)
                if len(shorter) >= substring_minimum and shorter in longer:
                    reason = "normalized_substring"
                elif intersection:
                    right_set = index.shingle_sets[right_index]
                    denominator = min(len(left_set), len(right_set))
                    containment = intersection / denominator if denominator else 0.0
                    union = len(left_set) + len(right_set) - intersection
                    jaccard = intersection / union if union else 0.0
                    if intersection >= minimum_intersection and (containment >= minimum_containment or jaccard >= minimum_jaccard):
                        reason = "near_shingle"
            if reason is not None:
                flags.append({
                    "pair_hash": digest(f"{label}\u241f{left_unit.unit_identity}\u241f{right_unit.unit_identity}"),
                    "left_record_identity_hash": left_unit.record_identity,
                    "right_record_identity_hash": right_unit.record_identity,
                    "left_dataset": left_unit.dataset_id,
                    "right_dataset": right_unit.dataset_id,
                    "left_partition": left_unit.partition,
                    "right_partition": right_unit.partition,
                    "reason": reason,
                    "intersection": intersection,
                    "containment": round(containment, 12),
                    "jaccard": round(jaccard, 12),
                })
    flags.sort(key=lambda value: value["pair_hash"])
    return {
        "candidate_comparisons": comparisons,
        "flag_count": len(flags),
        "reason_counts": dict(sorted(Counter(value["reason"] for value in flags).items())),
        "unique_left_records": len({value["left_record_identity_hash"] for value in flags}),
        "unique_right_records": len({value["right_record_identity_hash"] for value in flags}),
        "flag_examples_hashed_only": flags[:50],
    }


def _dataset_family(dataset_id: str) -> str:
    if dataset_id.startswith("hotpotqa"):
        return "hotpotqa"
    if dataset_id.startswith("musique"):
        return "musique"
    raise G0V2Error(f"unknown dataset family: {dataset_id}")


def compare_record_exact(left: Sequence[SelectedRecord | Projection], right: Sequence[SelectedRecord | Projection], label: str) -> dict[str, Any]:
    def projection(record: SelectedRecord | Projection) -> Projection:
        return record.projection if isinstance(record, SelectedRecord) else record

    extractors = {
        "raw_id_hash": lambda row: digest(row.sample_id),
        "family_scoped_id_hash": lambda row: digest(f"{_dataset_family(row.dataset_id)}\u241f{row.sample_id}"),
        "raw_question_text_hash": lambda row: digest(row.question_text),
        "normalized_question_text_hash": lambda row: digest(normalize(row.question_text)),
        "normalized_question_context_text_hash": lambda row: digest(normalize(row.question_text) + "\u241f" + normalize("\n\n".join(row.passages))),
    }
    left_rows = [projection(row) for row in left]
    right_rows = [projection(row) for row in right]
    result = {}
    total = 0
    for name, extractor in extractors.items():
        overlap = sorted(set(map(extractor, left_rows)) & set(map(extractor, right_rows)))
        result[name] = {"count": len(overlap), "overlap_hashes": [digest(f"{label}\u241f{name}\u241f{value}") for value in overlap[:50]]}
        total += len(overlap)
    return {"flag_count": total, "fields": result}


def audit_record_cell(
    left: Sequence[SelectedRecord | Projection],
    right: Sequence[SelectedRecord | Projection],
    similarity: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    exact = compare_record_exact(left, right, label)
    questions = compare_units(
        record_units(left, "question_unit"), record_units(right, "question_unit"),
        width=similarity["question_shingle_words"],
        minimum_intersection=similarity["question_min_intersection"],
        minimum_containment=similarity["question_min_containment"],
        minimum_jaccard=similarity["question_min_jaccard"],
        substring_minimum=similarity["exact_normalized_unit_substring_min_characters"],
        label=f"{label}\u241fquestion",
    )
    passages = compare_units(
        record_units(left, "passage_unit"), record_units(right, "passage_unit"),
        width=similarity["passage_shingle_words"],
        minimum_intersection=similarity["passage_min_intersection"],
        minimum_containment=similarity["passage_min_containment"],
        minimum_jaccard=similarity["passage_min_jaccard"],
        substring_minimum=similarity["exact_normalized_unit_substring_min_characters"],
        label=f"{label}\u241fpassage",
    )
    return {
        "left_records": len(left),
        "right_records": len(right),
        "exact_record_flags": exact,
        "question_unit_similarity": questions,
        "passage_unit_similarity": passages,
        "flag_count": exact["flag_count"] + questions["flag_count"] + passages["flag_count"],
    }


def audit_mapper_cell(mapper: Sequence[Unit], method: Sequence[SelectedRecord], similarity: Mapping[str, Any], label: str) -> dict[str, Any]:
    questions = compare_units(
        record_units(method, "question_unit"), mapper,
        width=similarity["question_shingle_words"],
        minimum_intersection=similarity["question_min_intersection"],
        minimum_containment=similarity["question_min_containment"],
        minimum_jaccard=similarity["question_min_jaccard"],
        substring_minimum=similarity["exact_normalized_unit_substring_min_characters"],
        label=f"{label}\u241fquestion-vs-sequence",
    )
    passages = compare_units(
        record_units(method, "passage_unit"), mapper,
        width=similarity["passage_shingle_words"],
        minimum_intersection=similarity["passage_min_intersection"],
        minimum_containment=similarity["passage_min_containment"],
        minimum_jaccard=similarity["passage_min_jaccard"],
        substring_minimum=similarity["exact_normalized_unit_substring_min_characters"],
        label=f"{label}\u241fpassage-vs-sequence",
    )
    return {
        "method_records": len(method),
        "mapper_sequences": len(mapper),
        "comparison_is_symmetric": True,
        "question_unit_vs_mapper_sequence": questions,
        "passage_unit_vs_mapper_sequence": passages,
        "flag_count": questions["flag_count"] + passages["flag_count"],
    }


def load_mapper_texts(config: Mapping[str, Any]) -> list[str]:
    tokenizer = Tokenizer.from_file(config["inputs"]["tokenizer_json"]["path"])
    result = []
    for key in ("mapper_fit_train", "mapper_fit_heldout"):
        spec = config["inputs"][key]
        array = np.load(spec["path"], allow_pickle=False)
        if array.ndim != 2 or len(array) != spec["expected_records"]:
            raise G0V2Error(f"mapper array contract mismatch: {key}")
        result.extend(tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in array)
    if len(result) != 550:
        raise G0V2Error("mapper-fit coverage must be exactly 550 sequences")
    return result


def verify_registry_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    registry = config["registry"]
    published = json.loads(Path(registry["published_marker"]["path"]).read_text())
    manifest = json.loads(Path(registry["manifest"]["path"]).read_text())
    receipt = json.loads(Path(registry["receipt"]["path"]).read_text())
    independent = json.loads(Path(registry["independent_audit"]["path"]).read_text())
    expected_counts = {name: spec["expected_records"] for name, spec in sorted(registry["files"].items())}
    if (
        published.get("schema_version") != 1
        or published.get("registry_id") != registry["registry_id"]
        or published.get("status") != "published_complete"
        or published.get("atomic_last_completion_marker") is not True
    ):
        raise G0V2Error("registry atomic publication marker semantics mismatch")
    publication_inventory = published.get("published_files")
    if not isinstance(publication_inventory, dict) or not publication_inventory:
        raise G0V2Error("registry publication inventory is missing")
    ordered_inventory = [
        {"name": name, **publication_inventory[name]}
        for name in sorted(publication_inventory)
    ]
    if digest_bytes(canonical_bytes(ordered_inventory)) != published.get("ordered_inventory_sha256"):
        raise G0V2Error("registry ordered publication inventory mismatch")
    publication_bindings = {"registry_manifest.json": registry["manifest"], "receipt.json": registry["receipt"]}
    publication_bindings.update({name: spec for name, spec in registry["files"].items()})
    for name, spec in publication_bindings.items():
        observed = publication_inventory.get(name, {})
        if {key: observed.get(key) for key in ("nbytes", "sha256")} != {"nbytes": spec["nbytes"], "sha256": spec["sha256"]}:
            raise G0V2Error(f"publication marker payload binding mismatch: {name}")
    if manifest.get("schema_version") != 2 or manifest.get("registry_id") != registry["registry_id"] or manifest.get("status") != "prepared_decontaminated_method_dev_only":
        raise G0V2Error("registry manifest semantics mismatch")
    if manifest.get("counts") != expected_counts or manifest.get("selection_is_gold_independent") is not True:
        raise G0V2Error("registry manifest count/gold-independence mismatch")
    if manifest.get("old_03_or_07_pilot_outputs_used_as_inputs") is not False:
        raise G0V2Error("registry manifest used old model outcomes")
    publication_contract = manifest.get("artifact_publication", {})
    if publication_contract.get("completion_marker_required") != "PUBLISHED.json" or publication_contract.get("status_pass_receipt_without_completion_marker_is_not_published") is not True:
        raise G0V2Error("registry manifest publication contract mismatch")
    threshold_source = config["inputs"]["g0_v1_threshold_config"]
    if {key: manifest.get("threshold_binding_09", {}).get(key) for key in ("path", "nbytes", "sha256")} != {key: threshold_source[key] for key in ("path", "nbytes", "sha256")}:
        raise G0V2Error("registry manifest threshold-source binding mismatch")
    capacity_failure = registry["strict_capacity_failure"]
    manifest_failure = manifest.get("local_strict_substring_capacity_attempt", {})
    if (
        {key: manifest_failure.get(key) for key in ("path", "nbytes", "sha256")} != {key: capacity_failure[key] for key in ("path", "nbytes", "sha256")}
        or manifest_failure.get("status") != "failed_closed_preserved_append_only"
    ):
        raise G0V2Error("registry manifest strict-capacity failure evidence mismatch")
    for name, spec in registry["files"].items():
        entry = manifest.get("files", {}).get(name, {})
        if {key: entry.get(key) for key in ("nbytes", "sha256", "rows")} != {"nbytes": spec["nbytes"], "sha256": spec["sha256"], "rows": spec["expected_records"]}:
            raise G0V2Error(f"registry manifest file binding mismatch: {name}")
    expected_locked_matrix = {
        "hotpotqa_train": {"hotpotqa_locked_dev": 0, "musique_locked_dev": 0},
        "musique_train": {"hotpotqa_locked_dev": 0, "musique_locked_dev": 0},
    }
    expected_sequence_matrix = {
        "hotpotqa_screen_to_hotpotqa_confirm": 0,
        "hotpotqa_screen_to_musique_confirm": 0,
        "musique_screen_to_hotpotqa_confirm": 0,
        "musique_screen_to_musique_confirm": 0,
    }
    if manifest.get("post_selection_four_dataset_pair_flag_counts") != expected_locked_matrix or manifest.get("screen_confirm_sequence_disjoint_four_cell_flag_counts") != expected_sequence_matrix:
        raise G0V2Error("registry manifest self-audit is nonzero or missing")
    manifest_privacy = manifest.get("privacy", {})
    if (
        manifest_privacy.get("registry_contains_only_allowlisted_identity_and_hash_fields") is not True
        or manifest_privacy.get("quarantine_contains_only_aggregates") is not True
        or manifest_privacy.get("answer_fields_projected_into_Projection") is not False
        or manifest_privacy.get("answer_values_available_to_rank_quota_or_decontamination_logic") is not False
        or manifest_privacy.get("answer_values_persisted_to_registry_or_quarantine") is not False
        or manifest_privacy.get("model_outcome_fields_retained_inspected_used_or_written") is not False
    ):
        raise G0V2Error("registry manifest privacy semantics mismatch")
    if receipt.get("schema_version") != 2 or receipt.get("registry_id") != registry["registry_id"] or receipt.get("status") != "pass" or receipt.get("cpu_only") is not True:
        raise G0V2Error("registry receipt semantics mismatch")
    receipt_publication = receipt.get("artifact_publication", {})
    if receipt_publication.get("completion_marker_required") != "PUBLISHED.json" or receipt_publication.get("status_pass_receipt_without_completion_marker_is_not_published") is not True:
        raise G0V2Error("registry receipt publication contract mismatch")
    if receipt.get("selection_counts") != expected_counts or receipt.get("selected_identities_globally_unique") is not True:
        raise G0V2Error("registry receipt coverage mismatch")
    if receipt.get("screen_confirm_disjoint") != {"hotpotqa": True, "musique": True}:
        raise G0V2Error("registry receipt identity disjointness mismatch")
    if receipt.get("post_selection_four_dataset_pair_flag_counts") != expected_locked_matrix or receipt.get("screen_confirm_sequence_disjoint_four_cell_flag_counts") != expected_sequence_matrix:
        raise G0V2Error("registry receipt isolation matrix mismatch")
    receipt_failure = receipt.get("local_strict_substring_capacity_attempt", {})
    if (
        {key: receipt_failure.get(key) for key in ("path", "nbytes", "sha256")} != {key: capacity_failure[key] for key in ("path", "nbytes", "sha256")}
        or receipt_failure.get("status") != "failed_closed_preserved_append_only"
    ):
        raise G0V2Error("registry receipt strict-capacity failure evidence mismatch")
    capacity_contract = registry["musique_capacity_contract"]
    manifest_quota = manifest.get("musique_quota_derivation", {})
    if (
        manifest_quota.get("reads_answers_or_model_outcomes") is not False
        or manifest_quota.get("observed_clean_4hop_pool") != capacity_contract["observed_clean_4hop_pool"]
        or manifest_quota.get("selected_4hop") != capacity_contract["selected_4hop"]
        or manifest_quota.get("asserted_unused_buffer") != capacity_contract["reserved_unused_4hop_buffer"]
    ):
        raise G0V2Error("registry manifest MuSiQue capacity-only quota mismatch")
    expected_receipt_quota = {
        "clean_pool_frozen_before_sha_ranking": True,
        "capacity_only_no_answer_or_outcome": True,
        **capacity_contract,
    }
    if receipt.get("musique_quota_derivation") != expected_receipt_quota:
        raise G0V2Error("registry receipt MuSiQue capacity-only quota mismatch")
    manifest_binding = receipt.get("manifest", {})
    if {key: manifest_binding.get(key) for key in ("nbytes", "sha256")} != {"nbytes": registry["manifest"]["nbytes"], "sha256": registry["manifest"]["sha256"]}:
        raise G0V2Error("registry receipt-to-manifest binding mismatch")
    if independent.get("schema_version") != 1 or independent.get("audit_id") != "registry_v2_final_independent_readonly_audit_v1" or independent.get("status") != "pass":
        raise G0V2Error("independent registry audit did not pass")
    if independent.get("stop_editing_observed") is not True or independent.get("source_tree_frozen_during_audit") is not True:
        raise G0V2Error("independent audit lacks frozen-tree evidence")
    frozen_tree = independent.get("source_tree_digests", {})
    start_digest = frozen_tree.get("start_sha256")
    end_digest = frozen_tree.get("end_sha256")
    if (
        not isinstance(start_digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", start_digest)
        or start_digest != end_digest
    ):
        raise G0V2Error("independent audit source tree changed or lacks exact digests")
    required_checks = {
        "published_inventory_rehashed",
        "implementation_files_rehashed",
        "registry_jsonl_allowlist_and_counts_recomputed",
        "ordered_rows_sha256_recomputed",
        "global_identity_uniqueness_recomputed",
        "screen_confirm_identity_disjointness_recomputed",
        "deterministic_mirror_core_files_byte_identical",
        "locked_outcomes_and_model_outputs_not_read",
    }
    checks = independent.get("checks", {})
    if set(checks) != required_checks or any(checks.get(name) is not True for name in required_checks):
        raise G0V2Error("independent audit required recomputation checks are incomplete")
    subject = independent.get("subject", {})
    for name in ("published_marker", "manifest", "receipt", "strict_capacity_failure"):
        if {key: subject.get(name, {}).get(key) for key in ("nbytes", "sha256")} != {"nbytes": registry[name]["nbytes"], "sha256": registry[name]["sha256"]}:
            raise G0V2Error(f"independent audit subject mismatch: {name}")
    for name, spec in registry["files"].items():
        observed = subject.get("files", {}).get(name, {})
        expected = {"nbytes": spec["nbytes"], "sha256": spec["sha256"], "rows": spec["expected_records"], "ordered_rows_sha256": spec["ordered_rows_sha256"]}
        if {key: observed.get(key) for key in expected} != expected:
            raise G0V2Error(f"independent audit file/ordered binding mismatch: {name}")
    implementation = manifest.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != {"config", "source", "entrypoint", "validator", "tests"}:
        raise G0V2Error("registry manifest implementation inventory is incomplete")
    if subject.get("implementation") != implementation:
        raise G0V2Error("independent audit implementation inventory mismatch")
    implementation_inventory = []
    for role in ("config", "source", "entrypoint", "validator"):
        implementation_inventory.append({"role": role, **implementation[role]})
    for name, spec in sorted(implementation["tests"].items()):
        implementation_inventory.append({"role": f"tests/{name}", **spec})
    expected_tree_digest = digest_bytes(canonical_bytes(implementation_inventory))
    if start_digest != expected_tree_digest:
        raise G0V2Error("independent audit source tree digest does not bind the implementation inventory")
    if independent.get("selection_counts") != expected_counts:
        raise G0V2Error("independent audit selection counts mismatch")
    if independent.get("musique_capacity_contract") != capacity_contract:
        raise G0V2Error("independent audit MuSiQue capacity contract mismatch")
    if independent.get("post_selection_four_dataset_pair_flag_counts") != expected_locked_matrix or independent.get("screen_confirm_sequence_disjoint_four_cell_flag_counts") != expected_sequence_matrix:
        raise G0V2Error("independent registry audit isolation matrix is nonzero or missing")
    return {
        "atomic_publication_marker_semantics": "verified",
        "manifest_semantics": "verified",
        "receipt_semantics": "verified",
        "independent_readonly_audit_semantics": "verified",
    }


def verify_threshold_binding(config: Mapping[str, Any]) -> None:
    upstream = json.loads(Path(config["inputs"]["g0_v1_threshold_config"]["path"]).read_text())
    if upstream.get("normalization") != config.get("normalization") or upstream.get("similarity") != config.get("similarity"):
        raise G0V2Error("G0 v1 normalization/similarity threshold binding mismatch")


def _matrix_total(matrix: Mapping[str, Mapping[str, Any]]) -> int:
    return sum(cell["flag_count"] for cell in matrix.values())


def _expected_matrix_keys(actual: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    if set(actual) != set(expected):
        raise G0V2Error(f"{label} matrix coverage mismatch")


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise G0V2Error(f"refusing to reuse existing output directory: {output_dir}")
    config = json.loads(config_path.read_text())
    require_frozen_config(config)
    verified: dict[str, Any] = {}
    for label, spec in iter_file_specs(config):
        verified[label] = verify_file(Path(spec["path"]), spec, label)
    verify_threshold_binding(config)
    evidence = verify_registry_evidence(config)

    registry_rows = {
        name: load_registry_file(Path(spec["path"]), spec)
        for name, spec in sorted(config["registry"]["files"].items())
    }
    registry_content_binding = {
        name: {
            "rows": len(rows),
            "stratum_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
            "ordered_rows_sha256": ordered_rows_sha256(rows),
        }
        for name, rows in sorted(registry_rows.items())
    }
    inputs = config["inputs"]
    train_sources = {
        "hotpotqa_train": load_hotpot(Path(inputs["hotpot_train"]["path"]), inputs["hotpot_train"]["expected_records"], locked=False),
        "musique_train": load_musique(Path(inputs["musique_train"]["path"]), inputs["musique_train"]["expected_records"], locked=False),
    }
    selected = rehydrate_registry(registry_rows, train_sources, config["registry"]["selection_namespaces"])
    # The selected records retain their immutable Projection objects.  Release
    # the other ~109k train projections before constructing locked-test indexes.
    del train_sources
    locked = [
        *load_hotpot(Path(inputs["hotpot_locked_dev"]["path"]), inputs["hotpot_locked_dev"]["expected_records"], locked=True),
        *load_musique(Path(inputs["musique_locked_dev"]["path"]), inputs["musique_locked_dev"]["expected_records"], locked=True),
    ]
    mapper = mapper_units(load_mapper_texts(config))
    similarity = config["similarity"]

    by_method_dataset = {dataset: [row for row in selected if row.dataset_id == dataset] for dataset in ("hotpotqa_train", "musique_train")}
    by_locked_dataset = {dataset: [row for row in locked if row.dataset_id == dataset] for dataset in ("hotpotqa_locked_dev", "musique_locked_dev")}
    mapper_matrix = {
        dataset: audit_mapper_cell(mapper, rows, similarity, f"mapper-fit-vs-{dataset}")
        for dataset, rows in by_method_dataset.items()
    }
    locked_matrix = {
        f"{method_dataset}__{locked_dataset}": audit_record_cell(method_rows, by_locked_dataset[locked_dataset], similarity, f"{method_dataset}-vs-{locked_dataset}")
        for method_dataset, method_rows in by_method_dataset.items()
        for locked_dataset in ("hotpotqa_locked_dev", "musique_locked_dev")
    }
    screen_confirm_matrix = {
        f"{screen_dataset}_screen__{confirm_dataset}_confirm": audit_record_cell(
            [row for row in by_method_dataset[screen_dataset] if row.partition == "screen"],
            [row for row in by_method_dataset[confirm_dataset] if row.partition == "confirm"],
            similarity,
            f"{screen_dataset}-screen-vs-{confirm_dataset}-confirm",
        )
        for screen_dataset in ("hotpotqa_train", "musique_train")
        for confirm_dataset in ("hotpotqa_train", "musique_train")
    }
    _expected_matrix_keys(mapper_matrix, ["hotpotqa_train", "musique_train"], "mapper")
    _expected_matrix_keys(locked_matrix, [
        "hotpotqa_train__hotpotqa_locked_dev", "hotpotqa_train__musique_locked_dev",
        "musique_train__hotpotqa_locked_dev", "musique_train__musique_locked_dev",
    ], "method-to-locked")
    _expected_matrix_keys(screen_confirm_matrix, [
        "hotpotqa_train_screen__hotpotqa_train_confirm", "hotpotqa_train_screen__musique_train_confirm",
        "musique_train_screen__hotpotqa_train_confirm", "musique_train_screen__musique_train_confirm",
    ], "screen-to-confirm")

    coverage = {
        "mapper_fit_sequences": len(mapper),
        "method_dev_records": len(selected),
        "method_dev_dataset_counts": dict(sorted(Counter(row.dataset_id for row in selected).items())),
        "method_dev_partition_counts": dict(sorted(Counter(row.partition for row in selected).items())),
        "locked_projection_records": len(locked),
        "locked_dataset_counts": dict(sorted(Counter(row.dataset_id for row in locked).items())),
        "registry_files": {name: len(rows) for name, rows in sorted(registry_rows.items())},
    }
    expected_coverage = config["expected_coverage"]
    if coverage != expected_coverage:
        raise G0V2Error(f"complete G0 v2 coverage mismatch: {coverage}")

    cross_tier_flags = _matrix_total(mapper_matrix) + _matrix_total(locked_matrix)
    sequence_isolation_flags = _matrix_total(screen_confirm_matrix)
    total_flags = cross_tier_flags + sequence_isolation_flags
    compatibility_comparisons = {
        "methoddev_hotpot_vs_mapper_fit": {"flags": mapper_matrix["hotpotqa_train"]["flag_count"]},
        "methoddev_musique_vs_mapper_fit": {"flags": mapper_matrix["musique_train"]["flag_count"]},
        "methoddev_hotpot_vs_locked_eval": {
            "flags": sum(cell["flag_count"] for key, cell in locked_matrix.items() if key.startswith("hotpotqa_train__")),
        },
        "methoddev_musique_vs_locked_eval": {
            "flags": sum(cell["flag_count"] for key, cell in locked_matrix.items() if key.startswith("musique_train__")),
        },
    }
    receipt = {
        "schema_version": 2,
        "audit_id": "complete_g0_decontaminated_registry_v2",
        "status": "complete" if total_flags == 0 else "failed_closed",
        "pass": total_flags == 0,
        "closes_complete_g0": total_flags == 0,
        "scope": "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval",
        "locked_outcomes_opened": False,
        "candidate_registry_binding": native_candidate_registry_binding(config),
        "consumer_contract_binding": {
            name: artifact_ref(spec) for name, spec in sorted(config["consumer_contracts"].items())
        },
        "comparisons": compatibility_comparisons,
        "config_sha256": sha256_file(config_path),
        "input_verification": verified,
        "registry_evidence_verification": evidence,
        "registry_content_binding": registry_content_binding,
        "coverage": coverage,
        "privacy": {
            "strict_source_projection": "identity, prompt text, context passages, and train stratum only; gold/support fields dropped from projected objects",
            "raw_source_values_necessarily_parsed_by_json_decoder": True,
            "gold_or_support_fields_retained_used_or_written": False,
            "model_outputs_read_or_written": False,
            "locked_outcome_metrics_read_or_written": False,
            "raw_prompt_or_context_text_written": False,
            "flag_examples_are_hashes_and_numeric_similarity_only": True,
        },
        "thresholds": similarity,
        "audits": {
            "mapper_fit_vs_method_dev_symmetric": mapper_matrix,
            "method_dev_vs_locked_projection_four_dataset_cells": locked_matrix,
            "screen_vs_confirm_sequence_isolation_four_dataset_cells": screen_confirm_matrix,
        },
        "result": {
            "cross_tier_flag_count": cross_tier_flags,
            "screen_confirm_isolation_flag_count": sequence_isolation_flags,
            "total_isolation_flag_count": total_flags,
            "pass": total_flags == 0,
        },
        "scope_details": {
            "mapper_fit": "pinned FineWeb-Edu train500 plus heldout50 decoded sequences",
            "method_dev": "all four published decontaminated registry v2 JSONL cells, rehydrated and projection-hash verified",
            "locked_test": "HotpotQA distractor dev7405 plus MuSiQue official answerable dev2417 strict identity/prompt/context projection only",
            "screen_confirm_sequence_isolation": "all four screen-dataset to confirm-dataset cells",
            "closes_complete_g0": total_flags == 0,
        },
        "implementation": {
            "source": sidecar(Path(__file__)),
            "entrypoint": sidecar(Path(__file__).resolve().parents[1] / "scripts" / "run_v2.py"),
            "tests": sidecar(Path(__file__).resolve().parents[1] / "tests" / "test_g0_audit_v2.py"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    receipt_path = output_dir / "complete_g0_receipt.json"
    atomic_json(receipt_path, receipt)
    atomic_json(output_dir / "complete_g0_receipt.json.sha256.json", sidecar(receipt_path))
    return receipt
