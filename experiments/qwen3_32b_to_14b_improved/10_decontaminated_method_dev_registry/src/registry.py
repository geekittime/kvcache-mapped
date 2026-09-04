from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from tokenizers import Tokenizer


class RegistryError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: str) -> str:
    return digest_bytes(value.encode("utf-8"))


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def shingles(text: str, width: int) -> frozenset[str]:
    words = text.split()
    if len(words) < width:
        return frozenset()
    return frozenset(digest("\u241f".join(words[i : i + width]))[:24] for i in range(len(words) - width + 1))


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_write(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def sidecar(path: Path) -> dict[str, Any]:
    return {"path": str(path), "nbytes": path.stat().st_size, "sha256": sha256_file(path)}


@dataclass(frozen=True)
class Projection:
    dataset_id: str
    sample_id: str
    source_index: int
    stratum: str
    question: str
    passages: tuple[str, ...]
    projection_sha256: str


@dataclass(frozen=True)
class UnitIndex:
    sets: tuple[frozenset[str], ...]
    texts: tuple[str, ...]
    inverted: Mapping[str, tuple[int, ...]]
    normalized_hashes: frozenset[str]
    substring_fingerprint_inverted: Mapping[int, tuple[int, ...]]


@dataclass(frozen=True)
class ReferenceIndex:
    ids: frozenset[str]
    raw_question_hashes: frozenset[str]
    normalized_question_hashes: frozenset[str]
    normalized_question_context_hashes: frozenset[str]
    question: UnitIndex
    passage: UnitIndex


_HOT_POT_KEYS = frozenset({"_id", "question", "context", "type", "level"})
_MUSIQUE_KEYS = frozenset({"id", "question", "paragraphs", "idx", "title", "paragraph_text"})


def _project_pairs(allowed: frozenset[str]):
    def project(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        return {key: value for key, value in pairs if key in allowed}
    return project


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError(f"{label} must be a non-empty string")
    return value


def _hotpot_passages(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RegistryError("HotpotQA context contract mismatch")
    result = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2 or not isinstance(row[1], list):
            raise RegistryError("HotpotQA passage contract mismatch")
        title = _required_text(row[0], "HotpotQA title")
        result.append(" ".join([title, *(str(sentence) for sentence in row[1])]))
    return tuple(result)


def _musique_passages(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RegistryError("MuSiQue paragraphs contract mismatch")
    ordered = sorted(value, key=lambda item: int(item.get("idx", -1)))
    if [int(item.get("idx", -1)) for item in ordered] != list(range(len(ordered))):
        raise RegistryError("MuSiQue paragraph indices are not contiguous")
    result = []
    for item in ordered:
        if set(item) - {"idx", "title", "paragraph_text"}:
            raise RegistryError("MuSiQue strict paragraph projection mismatch")
        result.append(" ".join([str(item.get("title", "")), _required_text(item.get("paragraph_text"), "MuSiQue paragraph")]))
    return tuple(result)


def _projection_sha(dataset_id: str, sample_id: str, question: str, passages: Sequence[str], stratum: str) -> str:
    # This deliberately hashes only the allowed, gold-independent projection.
    return digest_bytes(canonical_bytes({
        "dataset_id": dataset_id,
        "sample_id": sample_id,
        "question": question,
        "passages": list(passages),
        "stratum": stratum,
    }))


def load_hotpot(path: Path, expected: int, *, locked: bool) -> list[Projection]:
    decoder = json.JSONDecoder(object_pairs_hook=_project_pairs(_HOT_POT_KEYS))
    values = decoder.decode(path.read_text(encoding="utf-8"))
    if len(values) != expected:
        raise RegistryError("HotpotQA row-count mismatch")
    result = []
    seen_ids: set[str] = set()
    for source_index, row in enumerate(values):
        required = {"_id", "question", "context"} if locked else {"_id", "question", "context", "type", "level"}
        if not required.issubset(row) or set(row) - _HOT_POT_KEYS:
            raise RegistryError("HotpotQA strict projection mismatch")
        sample_id = _required_text(row["_id"], "HotpotQA id")
        if sample_id in seen_ids:
            raise RegistryError(f"duplicate HotpotQA sample_id: {sample_id}")
        seen_ids.add(sample_id)
        question = _required_text(row["question"], "HotpotQA question")
        passages = _hotpot_passages(row["context"])
        stratum = "locked" if locked else f"{row['type']}/{row['level']}"
        result.append(Projection("hotpotqa_train" if not locked else "hotpotqa_locked_dev", sample_id, source_index, stratum, question, passages, _projection_sha("hotpotqa", sample_id, question, passages, stratum)))
    return result


def _musique_hop(sample_id: str) -> str:
    # Official MuSiQue IDs use 2hop__, 3hop1__/3hop2__, and
    # 4hop1__/4hop2__/4hop3__; the leading digit is the hop stratum.
    match = re.match(r"^([234])hop(?:\d+)?__", sample_id)
    if not match:
        raise RegistryError(f"MuSiQue id does not encode 2/3/4-hop stratum: {sample_id}")
    return f"{match.group(1)}hop"


def load_musique(path: Path, expected: int, *, locked: bool) -> list[Projection]:
    decoder = json.JSONDecoder(object_pairs_hook=_project_pairs(_MUSIQUE_KEYS))
    result = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for source_index, line in enumerate(stream):
            row = decoder.decode(line)
            required = {"id", "question", "paragraphs"}
            if not required.issubset(row) or set(row) - _MUSIQUE_KEYS:
                raise RegistryError("MuSiQue strict projection mismatch")
            sample_id = _required_text(row["id"], "MuSiQue id")
            if sample_id in seen_ids:
                raise RegistryError(f"duplicate MuSiQue sample_id: {sample_id}")
            seen_ids.add(sample_id)
            question = _required_text(row["question"], "MuSiQue question")
            passages = _musique_passages(row["paragraphs"])
            stratum = "locked" if locked else _musique_hop(sample_id)
            result.append(Projection("musique_train" if not locked else "musique_locked_dev", sample_id, source_index, stratum, question, passages, _projection_sha("musique", sample_id, question, passages, stratum)))
    if len(result) != expected:
        raise RegistryError("MuSiQue row-count mismatch")
    return result


_SUBSTRING_FINGERPRINT_CHARS = 16
_ROLLING_BASE = 1_000_003
_ROLLING_MASK = (1 << 64) - 1


@lru_cache(maxsize=8192)
def _substring_fingerprints(text: str, minimum_characters: int) -> tuple[int, ...]:
    """Return deterministic winnowed fingerprints with complete substring recall.

    A normalized substring of at least ``minimum_characters`` contains a full
    window of ``minimum_characters - k + 1`` identical k-gram hashes.  Selecting
    the right-most minimum from *every* such window therefore guarantees that
    the two units share at least one fingerprint.  Fingerprints only retrieve
    candidates: the caller still verifies the literal substring relation, so a
    64-bit collision cannot create a contamination flag.
    """
    if minimum_characters <= 0:
        raise RegistryError("substring minimum must be positive")
    if len(text) < minimum_characters:
        return ()
    width = min(_SUBSTRING_FINGERPRINT_CHARS, minimum_characters)
    window = minimum_characters - width + 1
    power = pow(_ROLLING_BASE, width - 1, 1 << 64)
    value = 0
    for char in text[:width]:
        value = (value * _ROLLING_BASE + ord(char) + 1) & _ROLLING_MASK
    hashes = [value]
    for index in range(width, len(text)):
        outgoing = (ord(text[index - width]) + 1) * power
        value = ((value - outgoing) * _ROLLING_BASE + ord(text[index]) + 1) & _ROLLING_MASK
        hashes.append(value)

    candidates: deque[tuple[int, int]] = deque()
    selected: set[int] = set()
    for index, fingerprint in enumerate(hashes):
        # >= makes ties right-most, which is deterministic and is the standard
        # winnowing tie rule used by the completeness argument above.
        while candidates and candidates[-1][1] >= fingerprint:
            candidates.pop()
        candidates.append((index, fingerprint))
        first_allowed = index - window + 1
        while candidates[0][0] < first_allowed:
            candidates.popleft()
        if first_allowed >= 0:
            selected.add(candidates[0][1])
    return tuple(sorted(selected))


def build_unit_index(texts: Sequence[str], width: int, substring_min_chars: int) -> UnitIndex:
    normalized = tuple(normalize(value) for value in texts)
    sets = tuple(shingles(value, width) for value in normalized)
    mutable: dict[str, list[int]] = defaultdict(list)
    substring_mutable: dict[int, list[int]] = defaultdict(list)
    for index, values in enumerate(sets):
        for value in sorted(values):
            mutable[value].append(index)
        for fingerprint in _substring_fingerprints(normalized[index], substring_min_chars):
            substring_mutable[fingerprint].append(index)
    return UnitIndex(
        sets=sets,
        texts=normalized,
        inverted={key: tuple(value) for key, value in sorted(mutable.items())},
        normalized_hashes=frozenset(digest(value) for value in normalized),
        substring_fingerprint_inverted={
            key: tuple(value) for key, value in sorted(substring_mutable.items())
        },
    )


def build_reference_index(rows: Sequence[Projection], cfg: Mapping[str, Any]) -> ReferenceIndex:
    question_texts = tuple(row.question for row in rows)
    passage_texts = tuple(text for row in rows for text in row.passages)
    qctx_hashes = frozenset(
        digest(normalize(row.question) + "\u241f" + normalize("\n\n".join(row.passages)))
        for row in rows
    )
    return ReferenceIndex(
        ids=frozenset(row.sample_id for row in rows),
        raw_question_hashes=frozenset(digest(row.question) for row in rows),
        normalized_question_hashes=frozenset(digest(normalize(row.question)) for row in rows),
        normalized_question_context_hashes=qctx_hashes,
        question=build_unit_index(question_texts, cfg["question_shingle_words"], cfg["exact_normalized_unit_substring_min_characters"]),
        passage=build_unit_index(passage_texts, cfg["passage_shingle_words"], cfg["exact_normalized_unit_substring_min_characters"]),
    )


def build_mapper_index(texts: Sequence[str], cfg: Mapping[str, Any], *, question: bool) -> UnitIndex:
    width = cfg["question_shingle_words"] if question else cfg["passage_shingle_words"]
    return build_unit_index(texts, width, cfg["exact_normalized_unit_substring_min_characters"])


def _match_unit(
    normalized: str,
    *,
    width: int,
    reference: UnitIndex,
    minimum_intersection: int,
    minimum_containment: float,
    minimum_jaccard: float,
    substring_min_chars: int,
) -> tuple[bool, str | None]:
    if digest(normalized) in reference.normalized_hashes:
        return True, "normalized_exact"
    candidate_set = shingles(normalized, width)
    counts: Counter[int] = Counter()
    for token in sorted(candidate_set):
        for index in reference.inverted.get(token, ()):
            counts[index] += 1
    candidate_indices = set(counts)
    for fingerprint in _substring_fingerprints(normalized, substring_min_chars):
        candidate_indices.update(reference.substring_fingerprint_inverted.get(fingerprint, ()))
    # Deterministic priority: exact hash above, then substring for sorted
    # reference indices, then near-shingle for sorted reference indices.
    for index in sorted(candidate_indices):
        reference_text = reference.texts[index]
        shorter, longer = (normalized, reference_text) if len(normalized) <= len(reference_text) else (reference_text, normalized)
        if len(shorter) >= substring_min_chars and shorter in longer:
            return True, "normalized_substring"
    for index in sorted(counts):
        intersection = counts[index]
        reference_set = reference.sets[index]
        denominator = min(len(candidate_set), len(reference_set))
        containment = intersection / denominator if denominator else 0.0
        union = len(candidate_set) + len(reference_set) - intersection
        jaccard = intersection / union if union else 0.0
        if intersection >= minimum_intersection and (containment >= minimum_containment or jaccard >= minimum_jaccard):
            return True, "near_shingle"
    return False, None


def contamination_reasons(
    row: Projection,
    locked: ReferenceIndex,
    mapper_question: UnitIndex,
    mapper_passage: UnitIndex,
    cfg: Mapping[str, Any],
    *,
    locked_label: str = "locked",
    include_mapper: bool = True,
) -> frozenset[str]:
    reasons: set[str] = set()
    normalized_question = normalize(row.question)
    if row.sample_id in locked.ids:
        reasons.add(f"{locked_label}_exact_id")
    if digest(row.question) in locked.raw_question_hashes:
        reasons.add(f"{locked_label}_exact_question")
    if digest(normalized_question) in locked.normalized_question_hashes:
        reasons.add(f"{locked_label}_normalized_question")
    normalized_qctx = digest(normalized_question + "\u241f" + normalize("\n\n".join(row.passages)))
    if normalized_qctx in locked.normalized_question_context_hashes:
        reasons.add(f"{locked_label}_normalized_question_context")

    matched, kind = _match_unit(
        normalized_question,
        width=cfg["question_shingle_words"],
        reference=locked.question,
        minimum_intersection=cfg["question_min_intersection"],
        minimum_containment=cfg["question_min_containment"],
        minimum_jaccard=cfg["question_min_jaccard"],
        substring_min_chars=cfg["exact_normalized_unit_substring_min_characters"],
    )
    if matched:
        reasons.add(f"{locked_label}_question_{kind}")

    if include_mapper:
        matched, kind = _match_unit(
            normalized_question,
            width=cfg["question_shingle_words"],
            reference=mapper_question,
            minimum_intersection=cfg["question_min_intersection"],
            minimum_containment=cfg["question_min_containment"],
            minimum_jaccard=cfg["question_min_jaccard"],
            substring_min_chars=cfg["exact_normalized_unit_substring_min_characters"],
        )
        if matched:
            reasons.add(f"mapper_question_{kind}")

    locked_passage_found = False
    mapper_passage_found = not include_mapper
    for passage in row.passages:
        normalized_passage = normalize(passage)
        if not locked_passage_found:
            matched, kind = _match_unit(
                normalized_passage,
                width=cfg["passage_shingle_words"],
                reference=locked.passage,
                minimum_intersection=cfg["passage_min_intersection"],
                minimum_containment=cfg["passage_min_containment"],
                minimum_jaccard=cfg["passage_min_jaccard"],
                substring_min_chars=cfg["exact_normalized_unit_substring_min_characters"],
            )
            if matched:
                reasons.add(f"{locked_label}_passage_{kind}")
                locked_passage_found = True
        if not mapper_passage_found:
            matched, kind = _match_unit(
                normalized_passage,
                width=cfg["passage_shingle_words"],
                reference=mapper_passage,
                minimum_intersection=cfg["passage_min_intersection"],
                minimum_containment=cfg["passage_min_containment"],
                minimum_jaccard=cfg["passage_min_jaccard"],
                substring_min_chars=cfg["exact_normalized_unit_substring_min_characters"],
            )
            if matched:
                reasons.add(f"mapper_passage_{kind}")
                mapper_passage_found = True
        if locked_passage_found and mapper_passage_found:
            break
    return frozenset(reasons)


def train_union_question_collisions(
    rows: Sequence[Projection], cfg: Mapping[str, Any]
) -> tuple[frozenset[tuple[str, str]], dict[str, int]]:
    """Return both endpoints of every exact/normalized/near union collision."""
    width = cfg["question_shingle_words"]
    index = build_unit_index(
        [row.question for row in rows], width,
        cfg["exact_normalized_unit_substring_min_characters"],
    )
    marked: set[tuple[str, str]] = set()
    pair_kinds: Counter[str] = Counter()
    qctx = tuple(
        digest(normalize(row.question) + "\u241f" + normalize("\n\n".join(row.passages)))
        for row in rows
    )
    raw_groups: dict[str, list[int]] = defaultdict(list)
    normalized_groups: dict[str, list[int]] = defaultdict(list)
    qctx_groups: dict[str, list[int]] = defaultdict(list)
    for index_value, row in enumerate(rows):
        raw_groups[digest(row.question)].append(index_value)
        normalized_groups[digest(index.texts[index_value])].append(index_value)
        qctx_groups[qctx[index_value]].append(index_value)
    for kind, groups in (
        ("raw_exact_question", raw_groups),
        ("normalized_exact_question", normalized_groups),
        ("normalized_exact_question_context", qctx_groups),
    ):
        for members in groups.values():
            if len(members) > 1:
                marked.update((rows[value].dataset_id, rows[value].sample_id) for value in members)
                pair_kinds[kind] += len(members) * (len(members) - 1) // 2
    for left, row in enumerate(rows):
        candidate_set = index.sets[left]
        counts: Counter[int] = Counter()
        for token in sorted(candidate_set):
            for right in index.inverted.get(token, ()):
                if right > left:
                    counts[right] += 1
        candidates = set(counts)
        for fingerprint in _substring_fingerprints(
            index.texts[left], cfg["exact_normalized_unit_substring_min_characters"]
        ):
            candidates.update(
                right
                for right in index.substring_fingerprint_inverted.get(fingerprint, ())
                if right > left
            )
        for right in sorted(candidates):
            if index.texts[left] == index.texts[right]:
                continue
            kind = None
            shorter, longer = (
                (index.texts[left], index.texts[right])
                if len(index.texts[left]) <= len(index.texts[right])
                else (index.texts[right], index.texts[left])
            )
            if len(shorter) >= cfg["exact_normalized_unit_substring_min_characters"] and shorter in longer:
                kind = "normalized_question_substring"
            elif right in counts:
                intersection = counts[right]
                denominator = min(len(candidate_set), len(index.sets[right]))
                containment = intersection / denominator if denominator else 0.0
                union = len(candidate_set) + len(index.sets[right]) - intersection
                jaccard = intersection / union if union else 0.0
                if intersection >= cfg["question_min_intersection"] and (
                    containment >= cfg["question_min_containment"]
                    or jaccard >= cfg["question_min_jaccard"]
                ):
                    kind = "near_question"
            if kind is not None:
                marked.add((row.dataset_id, row.sample_id))
                marked.add((rows[right].dataset_id, rows[right].sample_id))
                pair_kinds[kind] += 1
    return frozenset(marked), dict(sorted(pair_kinds.items()))


def selection_rank(namespace: str, partition: str, stratum: str, sample_id: str, source_index: int) -> tuple[str, int]:
    value = digest(f"{namespace}\0{partition}\0{stratum}\0{sample_id}")
    return value, source_index


def select_partitioned(
    eligible: Sequence[Projection],
    *,
    namespace: str,
    strata_order: Sequence[str],
    screen_quotas: Mapping[str, int],
    confirm_quotas: Mapping[str, int],
) -> dict[str, list[dict[str, Any]]]:
    screen = select_one_partition(
        eligible, namespace=namespace, partition="screen",
        strata_order=strata_order, quotas=screen_quotas,
        excluded_identities=frozenset(),
    )
    excluded = frozenset((row["dataset_id"], row["sample_id"]) for row in screen)
    confirm = select_one_partition(
        eligible, namespace=namespace, partition="confirm",
        strata_order=strata_order, quotas=confirm_quotas,
        excluded_identities=excluded,
    )
    return {"screen": screen, "confirm": confirm}


def select_one_partition(
    eligible: Sequence[Projection],
    *,
    namespace: str,
    partition: str,
    strata_order: Sequence[str],
    quotas: Mapping[str, int],
    excluded_identities: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[Projection]] = defaultdict(list)
    for row in eligible:
        by_stratum[row.stratum].append(row)
    if set(by_stratum) - set(strata_order):
        raise RegistryError("unexpected stratum in eligible pool")
    selected: list[dict[str, Any]] = []
    for stratum in strata_order:
        count = int(quotas[stratum])
        available = [
            row for row in by_stratum[stratum]
            if (row.dataset_id, row.sample_id) not in excluded_identities
        ]
        ranked = sorted(available, key=lambda row: selection_rank(namespace, partition, stratum, row.sample_id, row.source_index))
        if len(ranked) < count:
            raise RegistryError(f"fail closed: insufficient eligible pool for {partition}/{stratum}: {len(ranked)} < {count}")
        for rank_index, row in enumerate(ranked[:count]):
            rank_sha, _ = selection_rank(namespace, partition, stratum, row.sample_id, row.source_index)
            selected.append({
                "dataset_id": row.dataset_id,
                "partition": partition,
                "sample_id": row.sample_id,
                "source_index": row.source_index,
                "source_projection_sha256": row.projection_sha256,
                "selection_rank": rank_index,
                "selection_rank_sha256": rank_sha,
                "stratum": stratum,
            })
    identities = [(row["dataset_id"], row["sample_id"]) for row in selected]
    if len(identities) != len(set(identities)):
        raise RegistryError(f"duplicate selected identity in {partition}")
    return selected


def verify_file(path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise RegistryError(f"missing input: {path}")
    observed = {"nbytes": path.stat().st_size, "sha256": sha256_file(path)}
    if observed != {"nbytes": spec["nbytes"], "sha256": spec["sha256"]}:
        raise RegistryError(f"pinned input mismatch: {path}")
    return observed


def verify_threshold_binding(config: Mapping[str, Any]) -> None:
    source_spec = config["threshold_binding"]["source"]
    verify_file(Path(source_spec["path"]), source_spec)
    upstream = json.loads(Path(source_spec["path"]).read_text())
    local = config["decontamination"]
    if upstream["normalization"] != local["normalization"]:
        raise RegistryError("09 normalization binding mismatch")
    for key in (
        "exact_normalized_unit_substring_min_characters", "passage_shingle_words",
        "passage_min_intersection", "passage_min_containment", "passage_min_jaccard",
        "question_shingle_words", "question_min_intersection", "question_min_containment",
        "question_min_jaccard",
    ):
        if upstream["similarity"][key] != local[key]:
            raise RegistryError(f"09 threshold binding mismatch: {key}")


def verify_failure_evidence_semantics(path: Path) -> None:
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("audit_id") != "complete_g0_mapper_fit_method_dev_locked_v1":
        raise RegistryError("09 failure evidence identity mismatch")
    if value.get("config_sha256") != "1f61ec78e74cff7b8f5b61b40d381ae4ddf78be53af8206eff86494e8caa0702":
        raise RegistryError("09 failure evidence config binding mismatch")
    if value.get("result") != {"cross_tier_flag_count": 10270, "pass": False}:
        raise RegistryError("09 failure evidence result semantics mismatch")
    scope = value.get("scope", {})
    if scope.get("closes_complete_g0") is not False:
        raise RegistryError("09 failure evidence must not close G0")
    if "07 prepared_v1" not in str(scope.get("method_dev", "")):
        raise RegistryError("09 failure evidence does not cover candidate 07")
    if "HotpotQA distractor dev7405" not in str(scope.get("locked_test", "")) or "MuSiQue dev2417" not in str(scope.get("locked_test", "")):
        raise RegistryError("09 failure evidence locked scope mismatch")
    coverage = value.get("coverage", {})
    if coverage.get("tier_records") != {"locked-test": 9822, "mapper-fit": 550, "method-dev": 2048}:
        raise RegistryError("09 failure evidence tier coverage mismatch")
    expected_candidates = {
        "01_oracle_prepare_v1": 512,
        "03_tail_refresh_v2": 384,
        "07_method_dev_prepared_v1": 1152,
    }
    observed_candidates = coverage.get("candidate_inventory", {})
    if {
        name: observed_candidates.get(name, {}).get("records")
        for name in expected_candidates
    } != expected_candidates or not all(
        observed_candidates.get(name, {}).get("covered") is True
        for name in expected_candidates
    ):
        raise RegistryError("09 failure evidence candidate coverage mismatch")
    similarity = value.get("cross_tier_similarity", {})
    mapper = similarity.get("mapper-fit_to_method-dev", {})
    locked = similarity.get("method-dev_to_locked-test", {})
    expected_dataset_pairs = {
        "hotpotqa_train__hotpotqa_dev_distractor": 2470,
        "hotpotqa_train__musique_official_answerable_dev": 276,
        "musique_train__hotpotqa_dev_distractor": 794,
        "musique_train__musique_official_answerable_dev": 6730,
    }
    expected_candidate_pairs = {
        "01_oracle_prepare_v1__locked": 2440,
        "03_tail_refresh_v2__locked": 1802,
        "07_method_dev_prepared_v1__locked": 6028,
    }
    summary = locked.get("flag_summary", {})
    if mapper.get("flag_count") != 0 or mapper.get("flag_summary", {}).get("flag_pairs") != 0:
        raise RegistryError("09 failure evidence mapper-fit semantics mismatch")
    if (
        locked.get("flag_count") != 10270
        or locked.get("candidate_comparisons") != 4077182
        or summary.get("flag_pairs") != 10270
        or summary.get("by_dataset_pair") != expected_dataset_pairs
        or summary.get("by_candidate_pair") != expected_candidate_pairs
        or summary.get("unique_left_records") != 1571
        or summary.get("unique_right_records") != 2932
        or len(locked.get("flags", [])) != 10270
    ):
        raise RegistryError("09 failure evidence contamination semantics mismatch")
    exact = value.get("cross_tier_exact", {})
    if exact.get("flag_count") != 0:
        raise RegistryError("09 failure evidence exact-tier semantics mismatch")


def verify_capacity_failure_semantics(path: Path, root: Path) -> None:
    value = json.loads(path.read_text())
    expected = {
        "attempt_id": "strict_substring_capacity_fail_attempt2",
        "status": "failed_closed_preserved_append_only",
        "error": "fail closed: strict clean 4hop capacity changed: 119 != 146",
        "old_expected_clean_4hop_capacity": 146,
        "new_expected_clean_4hop_capacity": 119,
        "formal_registry_files_written": 0,
        "completion_marker_present": False,
        "published": False,
        "thresholds_changed": False,
        "gpu_used": False,
        "model_outcomes_read": False,
        "answer_values_available_to_capacity_or_quota_logic": False,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RegistryError("strict substring capacity failure semantics mismatch")
    if value.get("config") != {
        "nbytes": 7870,
        "sha256": "16ef2a02df731bf0ed49ef77af4df62cfab0b47e640c0b01e431d9d112e93f34",
    } or value.get("source") != {
        "nbytes": 51501,
        "sha256": "1dc83c2fde30e850fc0c954c9ad9891f51019f2552d3bd757f0c6e4a02b25c12",
    }:
        raise RegistryError("strict substring capacity implementation binding mismatch")
    log_spec = value.get("log", {})
    log_path = root / str(log_spec.get("path", ""))
    verify_file(log_path, log_spec)
    artifact_dir = root / str(value.get("artifact_dir", ""))
    if not artifact_dir.is_dir() or sorted(path.name for path in artifact_dir.iterdir()) != ["IN_PROGRESS.json"]:
        raise RegistryError("strict substring failed artifact inventory mismatch")
    verify_file(artifact_dir / "IN_PROGRESS.json", value.get("in_progress_marker", {}))
    if (artifact_dir / "PUBLISHED.json").exists():
        raise RegistryError("strict substring failed attempt must not be published")


def load_mapper_texts(config: Mapping[str, Any]) -> list[str]:
    tokenizer = Tokenizer.from_file(config["inputs"]["tokenizer_json"]["path"])
    result = []
    for key in ("mapper_fit_train", "mapper_fit_heldout"):
        spec = config["inputs"][key]
        values = np.load(spec["path"], allow_pickle=False)
        if values.ndim != 2 or len(values) != spec["expected_records"]:
            raise RegistryError(f"mapper array contract mismatch: {key}")
        result.extend(tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in values)
    if len(result) != 550:
        raise RegistryError("mapper-fit coverage must be exactly 550 sequences")
    return result


def decontaminate(
    train: Sequence[Projection],
    locked_indexes: Mapping[str, ReferenceIndex],
    locked_counts: Mapping[str, int],
    mapper_question: UnitIndex,
    mapper_passage: UnitIndex,
    cfg: Mapping[str, Any],
    train_question_collisions: frozenset[tuple[str, str]],
) -> tuple[list[Projection], dict[str, Any]]:
    eligible = []
    reason_counts: Counter[str] = Counter()
    quarantined_by_stratum: Counter[str] = Counter()
    eligible_by_stratum: Counter[str] = Counter()
    for row in train:
        reasons: set[str] = set()
        for locked_index, (locked_label, index) in enumerate(sorted(locked_indexes.items())):
            reasons.update(contamination_reasons(
                row, index, mapper_question, mapper_passage, cfg,
                locked_label=locked_label, include_mapper=locked_index == 0,
            ))
        if (row.dataset_id, row.sample_id) in train_question_collisions:
            reasons.add("train_union_exact_normalized_or_near_question_collision")
        if reasons:
            quarantined_by_stratum[row.stratum] += 1
            reason_counts.update(reasons)
        else:
            eligible.append(row)
            eligible_by_stratum[row.stratum] += 1
    aggregate = {
        "source_records": len(train),
        "locked_records_union": dict(sorted(locked_counts.items())),
        "mapper_fit_sequences": len(mapper_question.texts),
        "quarantined_records": len(train) - len(eligible),
        "eligible_records": len(eligible),
        "reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
        "quarantined_by_stratum": dict(sorted(quarantined_by_stratum.items())),
        "eligible_by_stratum": dict(sorted(eligible_by_stratum.items())),
        "raw_text_or_identity_written": False,
    }
    return eligible, aggregate


def run(
    config_path: Path,
    root: Path,
    artifact_dir_override: str | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    configured_artifact_dir = config["outputs"]["artifact_dir"]
    artifact_dir = artifact_dir_override or configured_artifact_dir
    if Path(artifact_dir).is_absolute() or ".." in Path(artifact_dir).parts:
        raise RegistryError("artifact directory must be a relative path below this experiment")
    output_dir = root / artifact_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    writer_lock_path = output_dir.parent / f".{output_dir.name}.single_writer.lock"
    writer_lock = writer_lock_path.open("a+")
    try:
        fcntl.flock(writer_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        writer_lock.close()
        raise RegistryError(f"another writer owns artifact publication: {output_dir}") from error
    writer_lock.seek(0)
    writer_lock.truncate()
    writer_lock.write(f"pid={os.getpid()}\n")
    writer_lock.flush()
    os.fsync(writer_lock.fileno())
    try:
        if output_dir.exists():
            raise RegistryError(f"refusing to reuse existing artifact directory: {output_dir}")
        # Claim the artifact version while holding the advisory writer lock.
        # From this point onward the non-reusable IN_PROGRESS marker is the
        # durable single-writer reservation, so the fd can be released even if
        # later validation or computation raises.
        output_dir.mkdir(parents=True, exist_ok=False)
        in_progress_path = output_dir / "IN_PROGRESS.json"
        atomic_json(in_progress_path, {
            "schema_version": 1,
            "registry_id": config["registry_id"],
            "status": "in_progress_not_published",
            "completion_marker_required": "PUBLISHED.json",
        })
    finally:
        fcntl.flock(writer_lock.fileno(), fcntl.LOCK_UN)
        writer_lock.close()
    verify_threshold_binding(config)
    verified = {}
    for name, spec in sorted(config["inputs"].items()):
        verified[name] = verify_file(Path(spec["path"]), spec)
    failure_evidence = config["failure_evidence"]["complete_g0_v1_receipt"]
    verified_failure_evidence = verify_file(Path(failure_evidence["path"]), failure_evidence)
    verify_failure_evidence_semantics(Path(failure_evidence["path"]))
    local_failure_spec = config["failure_evidence"]["local_equal_quota_attempt"]
    local_failure_path = root / local_failure_spec["path"]
    verified_local_failure = {"path": str(local_failure_path), **verify_file(local_failure_path, local_failure_spec)}
    complexity_failure_spec = config["failure_evidence"]["local_complexity_terminated_attempt"]
    complexity_failure_path = root / complexity_failure_spec["path"]
    verified_complexity_failure = {"path": str(complexity_failure_path), **verify_file(complexity_failure_path, complexity_failure_spec)}
    capacity_failure_spec = config["failure_evidence"]["local_strict_substring_capacity_attempt"]
    capacity_failure_path = root / capacity_failure_spec["path"]
    verified_capacity_failure = {"path": str(capacity_failure_path), **verify_file(capacity_failure_path, capacity_failure_spec)}
    verify_capacity_failure_semantics(capacity_failure_path, root)
    inputs = config["inputs"]
    mapper_texts = load_mapper_texts(config)
    hotpot_train = load_hotpot(Path(inputs["hotpot_train"]["path"]), inputs["hotpot_train"]["expected_records"], locked=False)
    hotpot_locked = load_hotpot(Path(inputs["hotpot_locked_dev"]["path"]), inputs["hotpot_locked_dev"]["expected_records"], locked=True)
    musique_train = load_musique(Path(inputs["musique_train"]["path"]), inputs["musique_train"]["expected_records"], locked=False)
    musique_locked = load_musique(Path(inputs["musique_locked_dev"]["path"]), inputs["musique_locked_dev"]["expected_records"], locked=True)
    locked_indexes = {
        "hotpotqa_locked_dev": build_reference_index(hotpot_locked, config["decontamination"]),
        "musique_locked_dev": build_reference_index(musique_locked, config["decontamination"]),
    }
    locked_counts = {"hotpotqa_locked_dev": len(hotpot_locked), "musique_locked_dev": len(musique_locked)}
    mapper_question = build_mapper_index(mapper_texts, config["decontamination"], question=True)
    mapper_passage = build_mapper_index(mapper_texts, config["decontamination"], question=False)
    train_question_collisions, train_question_collision_pairs = train_union_question_collisions(
        [*hotpot_train, *musique_train], config["decontamination"]
    )
    hotpot_eligible, hotpot_quarantine = decontaminate(
        hotpot_train, locked_indexes, locked_counts, mapper_question,
        mapper_passage, config["decontamination"], train_question_collisions,
    )
    musique_eligible, musique_quarantine = decontaminate(
        musique_train, locked_indexes, locked_counts, mapper_question,
        mapper_passage, config["decontamination"], train_question_collisions,
    )
    hp_sel = config["selection"]["hotpotqa"]
    mq_sel = config["selection"]["musique"]
    observed_clean_4hop = musique_quarantine["eligible_by_stratum"].get("4hop", 0)
    if observed_clean_4hop != mq_sel["expected_clean_4hop_capacity"]:
        raise RegistryError(
            f"fail closed: strict clean 4hop capacity changed: {observed_clean_4hop} != {mq_sel['expected_clean_4hop_capacity']}"
        )
    selected_4hop = mq_sel["screen_quotas"]["4hop"] + mq_sel["confirm_quotas"]["4hop"]
    if observed_clean_4hop - selected_4hop != mq_sel["reserved_clean_4hop_buffer"]:
        raise RegistryError("fail closed: frozen clean 4hop buffer mismatch")

    hotpot_screen = select_one_partition(
        hotpot_eligible, namespace=hp_sel["namespace"], partition="screen",
        strata_order=hp_sel["strata_order"], quotas=hp_sel["screen_quotas"],
        excluded_identities=frozenset(),
    )
    musique_screen = select_one_partition(
        musique_eligible, namespace=mq_sel["namespace"], partition="screen",
        strata_order=mq_sel["strata_order"], quotas=mq_sel["screen_quotas"],
        excluded_identities=frozenset(),
    )

    selected_projection = {
        "hotpotqa_train": {row.sample_id: row for row in hotpot_eligible},
        "musique_train": {row.sample_id: row for row in musique_eligible},
    }
    screen_projection = {
        "hotpotqa_screen": [selected_projection["hotpotqa_train"][row["sample_id"]] for row in hotpot_screen],
        "musique_screen": [selected_projection["musique_train"][row["sample_id"]] for row in musique_screen],
    }
    screen_indexes = {
        label: build_reference_index(rows, config["decontamination"])
        for label, rows in screen_projection.items()
    }
    empty_q = build_mapper_index([], config["decontamination"], question=True)
    empty_p = build_mapper_index([], config["decontamination"], question=False)
    screen_identities = frozenset(
        (row.dataset_id, row.sample_id)
        for rows in screen_projection.values() for row in rows
    )
    partition_excluded: set[tuple[str, str]] = set(screen_identities)
    partition_quarantine_counts: Counter[str] = Counter()
    for row in [*hotpot_eligible, *musique_eligible]:
        identity = (row.dataset_id, row.sample_id)
        if identity in screen_identities:
            continue
        for screen_label, screen_index in sorted(screen_indexes.items()):
            reasons = contamination_reasons(
                row, screen_index, empty_q, empty_p, config["decontamination"],
                locked_label=screen_label, include_mapper=False,
            )
            if reasons:
                partition_excluded.add(identity)
                partition_quarantine_counts[f"{screen_label}_to_{row.dataset_id}_confirm_pool"] += 1

    excluded_frozen = frozenset(partition_excluded)
    hotpot_confirm = select_one_partition(
        hotpot_eligible, namespace=hp_sel["namespace"], partition="confirm",
        strata_order=hp_sel["strata_order"], quotas=hp_sel["confirm_quotas"],
        excluded_identities=excluded_frozen,
    )
    musique_confirm = select_one_partition(
        musique_eligible, namespace=mq_sel["namespace"], partition="confirm",
        strata_order=mq_sel["strata_order"], quotas=mq_sel["confirm_quotas"],
        excluded_identities=excluded_frozen,
    )
    hotpot_selected = {"screen": hotpot_screen, "confirm": hotpot_confirm}
    musique_selected = {"screen": musique_screen, "confirm": musique_confirm}

    all_selected_identities = [
        (row["dataset_id"], row["sample_id"])
        for selections in (hotpot_selected, musique_selected)
        for partition in ("screen", "confirm") for row in selections[partition]
    ]
    if len(all_selected_identities) != len(set(all_selected_identities)):
        raise RegistryError("duplicate identity across final registry partitions")

    post_selection_matrix: dict[str, dict[str, int]] = {}
    for train_name, selections in (("hotpotqa_train", hotpot_selected), ("musique_train", musique_selected)):
        chosen = [selected_projection[train_name][item["sample_id"]] for part in ("screen", "confirm") for item in selections[part]]
        post_selection_matrix[train_name] = {}
        for locked_label, index in sorted(locked_indexes.items()):
            flagged = sum(bool(contamination_reasons(row, index, empty_q, empty_p, config["decontamination"], locked_label=locked_label, include_mapper=False)) for row in chosen)
            post_selection_matrix[train_name][locked_label] = flagged
    if any(value for row in post_selection_matrix.values() for value in row.values()):
        raise RegistryError(f"post-selection four-way locked audit failed: {post_selection_matrix}")

    sequence_disjoint_matrix: dict[str, int] = {}
    for screen_label, screen_index in sorted(screen_indexes.items()):
        for confirm_name, rows in (
            ("hotpotqa_confirm", hotpot_confirm),
            ("musique_confirm", musique_confirm),
        ):
            chosen = [selected_projection[row["dataset_id"]][row["sample_id"]] for row in rows]
            flags = sum(bool(contamination_reasons(
                row, screen_index, empty_q, empty_p, config["decontamination"],
                locked_label=screen_label, include_mapper=False,
            )) for row in chosen)
            sequence_disjoint_matrix[f"{screen_label}_to_{confirm_name}"] = flags
    if any(sequence_disjoint_matrix.values()):
        raise RegistryError(f"screen-confirm sequence-disjoint audit failed: {sequence_disjoint_matrix}")

    outputs = {
        "hotpotqa_screen.jsonl": hotpot_selected["screen"],
        "hotpotqa_confirm.jsonl": hotpot_selected["confirm"],
        "musique_screen.jsonl": musique_selected["screen"],
        "musique_confirm.jsonl": musique_selected["confirm"],
    }
    for name, rows in outputs.items():
        atomic_jsonl(output_dir / name, rows)
    quarantine = {
        "schema_version": 2,
        "registry_id": config["registry_id"],
        "hotpotqa": hotpot_quarantine,
        "musique": musique_quarantine,
        "train_union_question_collision_pairs": train_question_collision_pairs,
        "train_union_question_collision_records": len(train_question_collisions),
        "screen_to_confirm_sequence_quarantine_counts": dict(sorted(partition_quarantine_counts.items())),
        "privacy": "aggregate counts only; no IDs, questions, contexts, answers, or outcomes",
    }
    atomic_json(output_dir / "quarantine_aggregate.json", quarantine)

    expected_counts = {"hotpotqa_screen.jsonl": 128, "hotpotqa_confirm.jsonl": 256, "musique_screen.jsonl": 256, "musique_confirm.jsonl": 512}
    actual_counts = {name: len(rows) for name, rows in outputs.items()}
    if actual_counts != expected_counts:
        raise RegistryError(f"registry count mismatch: {actual_counts}")
    manifest_files = {name: {**sidecar(output_dir / name), "rows": len(rows)} for name, rows in sorted(outputs.items())}
    manifest_files["quarantine_aggregate.json"] = sidecar(output_dir / "quarantine_aggregate.json")
    implementation = {
        "config": sidecar(config_path),
        "source": sidecar(root / "src" / "registry.py"),
        "entrypoint": sidecar(root / "scripts" / "prepare.py"),
        "validator": sidecar(root / "scripts" / "validate.py"),
        "tests": {
            path.name: sidecar(path)
            for path in sorted((root / "tests").glob("*.py"))
        },
    }
    manifest = {
        "schema_version": 2,
        "registry_id": config["registry_id"],
        "status": "prepared_decontaminated_method_dev_only",
        "artifact_publication": {
            "single_writer_lock": str(writer_lock_path),
            "configured_artifact_dir": configured_artifact_dir,
            "effective_artifact_dir": artifact_dir,
            "artifact_dir_override": artifact_dir_override is not None,
            "completion_marker_required": "PUBLISHED.json",
            "status_pass_receipt_without_completion_marker_is_not_published": True,
        },
        "implementation": implementation,
        "threshold_binding_09": config["threshold_binding"]["source"],
        "failure_evidence_09": {**failure_evidence, **verified_failure_evidence},
        "local_fail_closed_attempt": verified_local_failure,
        "local_complexity_terminated_attempt": verified_complexity_failure,
        "local_strict_substring_capacity_attempt": {**capacity_failure_spec, **verified_capacity_failure},
        "verified_inputs": verified,
        "files": manifest_files,
        "counts": actual_counts,
        "selection_is_gold_independent": True,
        "old_registry_policy": "06/07 v1 registries preserved and superseded for future 07/01 runs only",
        "old_03_or_07_pilot_outputs_used_as_inputs": False,
        "post_selection_four_dataset_pair_flag_counts": post_selection_matrix,
        "screen_confirm_sequence_disjoint_four_cell_flag_counts": sequence_disjoint_matrix,
        "exact_normalized_substring_recall": {
            "minimum_characters": config["decontamination"]["exact_normalized_unit_substring_min_characters"],
            "candidate_recall": "complete deterministic character winnowing",
            "final_decision": "literal normalized shorter-in-longer verification",
            "full_reference_scan": False,
        },
        "musique_quota_derivation": {
            "rule": mq_sel["quota_derivation"],
            "reads_answers_or_model_outcomes": mq_sel["quota_derivation_reads_answers_or_model_outcomes"],
            "reserved_clean_4hop_buffer": mq_sel["reserved_clean_4hop_buffer"],
            "observed_clean_4hop_pool": musique_quarantine["eligible_by_stratum"]["4hop"],
            "selected_4hop": mq_sel["screen_quotas"]["4hop"] + mq_sel["confirm_quotas"]["4hop"],
            "asserted_unused_buffer": mq_sel["reserved_clean_4hop_buffer"],
        },
        "privacy": {
            "registry_contains_only_allowlisted_identity_and_hash_fields": True,
            "quarantine_contains_only_aggregates": True,
            "raw_source_file_bytes_necessarily_read_by_json_decoder": True,
            "raw_answer_values_transiently_parsed_into_decoder_object_pairs": True,
            "answer_fields_projected_into_Projection": False,
            "answer_values_available_to_rank_quota_or_decontamination_logic": False,
            "answer_values_persisted_to_registry_or_quarantine": False,
            "model_outcome_fields_retained_inspected_used_or_written": False,
        },
    }
    atomic_json(output_dir / "registry_manifest.json", manifest)
    receipt = {
        "schema_version": 2,
        "registry_id": config["registry_id"],
        "status": "pass",
        "cpu_only": True,
        "artifact_publication": {
            "single_writer": True,
            "nonexistent_artifact_directory_required_at_start": True,
            "completion_marker_required": "PUBLISHED.json",
            "status_pass_receipt_without_completion_marker_is_not_published": True,
        },
        "input_access_inventory": [
            config["threshold_binding"]["source"]["path"],
            failure_evidence["path"],
            str(local_failure_path),
            str(complexity_failure_path),
            str(capacity_failure_path),
            *(spec["path"] for _, spec in sorted(config["inputs"].items())),
        ],
        "old_03_or_07_pilot_outputs_used_as_inputs": False,
        "local_strict_substring_capacity_attempt": {**capacity_failure_spec, **verified_capacity_failure},
        "post_selection_four_dataset_pair_flag_counts": post_selection_matrix,
        "screen_confirm_sequence_disjoint_four_cell_flag_counts": sequence_disjoint_matrix,
        "exact_normalized_substring_recall": {
            "minimum_characters": config["decontamination"]["exact_normalized_unit_substring_min_characters"],
            "candidate_recall": "complete deterministic character winnowing",
            "final_decision": "literal normalized shorter-in-longer verification",
            "full_reference_scan": False,
        },
        "implementation": implementation,
        "musique_quota_derivation": {
            "clean_pool_frozen_before_sha_ranking": True,
            "capacity_only_no_answer_or_outcome": True,
            "observed_clean_4hop_pool": musique_quarantine["eligible_by_stratum"]["4hop"],
            "selected_4hop": mq_sel["screen_quotas"]["4hop"] + mq_sel["confirm_quotas"]["4hop"],
            "reserved_unused_4hop_buffer": musique_quarantine["eligible_by_stratum"]["4hop"] - mq_sel["screen_quotas"]["4hop"] - mq_sel["confirm_quotas"]["4hop"],
        },
        "thresholds": config["decontamination"],
        "quarantine": {"hotpotqa": hotpot_quarantine, "musique": musique_quarantine},
        "selection_counts": actual_counts,
        "screen_confirm_disjoint": {
            "hotpotqa": not ({row["sample_id"] for row in hotpot_selected["screen"]} & {row["sample_id"] for row in hotpot_selected["confirm"]}),
            "musique": not ({row["sample_id"] for row in musique_selected["screen"]} & {row["sample_id"] for row in musique_selected["confirm"]}),
        },
        "selected_identity_count": len(all_selected_identities),
        "selected_identities_globally_unique": len(all_selected_identities) == len(set(all_selected_identities)),
        "manifest": sidecar(output_dir / "registry_manifest.json"),
        "claim_boundary": "new v2 method-dev registry; old 06/07 selections remain immutable; no model outcomes or locked-test metrics evaluated",
    }
    atomic_json(output_dir / "receipt.json", receipt)
    core_payload_names = sorted([
        *outputs,
        "quarantine_aggregate.json",
        "registry_manifest.json",
        "receipt.json",
    ])
    for name in core_payload_names:
        path = output_dir / name
        if path.is_file():
            atomic_json(output_dir / f"{path.name}.sha256.json", sidecar(path))
    for name in core_payload_names:
        path = output_dir / name
        observed_sidecar = json.loads((output_dir / f"{name}.sha256.json").read_text())
        if observed_sidecar != sidecar(path):
            raise RegistryError(f"publication sidecar self-check failed: {name}")
    in_progress_path.unlink()
    published_paths = sorted(path for path in output_dir.iterdir() if path.is_file())
    expected_names = sorted([
        *core_payload_names,
        *(f"{name}.sha256.json" for name in core_payload_names),
    ])
    if [path.name for path in published_paths] != expected_names:
        raise RegistryError("publication inventory has missing or extra files before commit")
    published_files = {
        path.name: {"nbytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in published_paths
    }
    ordered_inventory = [
        {"name": name, **published_files[name]}
        for name in sorted(published_files)
    ]
    publication = {
        "schema_version": 1,
        "registry_id": config["registry_id"],
        "status": "published_complete",
        "atomic_last_completion_marker": True,
        "published_files": published_files,
        "ordered_inventory_sha256": digest_bytes(canonical_bytes(ordered_inventory)),
        "receipt": published_files["receipt.json"],
        "receipt_sidecar": published_files["receipt.json.sha256.json"],
    }
    # This is deliberately the final artifact write. Consumers and validators
    # must reject a directory that lacks this marker, even if receipt.json says
    # status=pass after a crash between payload and publication completion.
    atomic_json(output_dir / "PUBLISHED.json", publication)
    return receipt
