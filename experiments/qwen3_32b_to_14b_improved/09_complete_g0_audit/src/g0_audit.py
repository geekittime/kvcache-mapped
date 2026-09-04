from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tokenizers import Tokenizer


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def shingles(text: str, width: int) -> frozenset[str]:
    words = text.split()
    if len(words) < width:
        return frozenset()
    return frozenset(digest("\u241f".join(words[i : i + width]))[:24] for i in range(len(words) - width + 1))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@dataclass(frozen=True)
class Record:
    tier: str
    candidate: str
    dataset: str
    identity: str
    raw_id_hash: str
    question: str
    context: str
    passages: tuple[str, ...]


@dataclass(frozen=True)
class Unit:
    tier: str
    candidate: str
    dataset: str
    identity: str
    record_identity: str
    kind: str
    text: str


def _verify_inputs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verified = {}
    for name, spec in sorted(config["inputs"].items()):
        path = Path(spec["path"])
        if not path.is_file():
            raise RuntimeError(f"missing pinned input: {name}")
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != spec["nbytes"] or actual_sha != spec["sha256"]:
            raise RuntimeError(f"pinned input mismatch: {name}")
        verified[name] = {"nbytes": actual_size, "sha256": actual_sha}
    return verified


def _extract_01_context(prompt: str) -> str:
    marker = "\n\nContext:\n"
    tail = "\n\nQuestion:"
    if marker not in prompt or tail not in prompt:
        raise RuntimeError("candidate 01 prompt contract mismatch")
    return prompt.split(marker, 1)[1].rsplit(tail, 1)[0]


def _split_passages(context: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in re.split(r"\n\s*\n(?=\[(?:Passage\s+)?\d+\])", context) if x.strip())


def _load_method(path: Path, candidate: str, expected: dict[str, Any]) -> list[Record]:
    rows: list[Record] = []
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            obj = json.loads(line)
            if candidate.startswith("01"):
                dataset = obj["dataset"]
                context = _extract_01_context(obj["qa_prompt"])
            else:
                dataset = obj["dataset_id"]
                context = obj["context"]
            question = obj["question"]
            raw_id = f"{dataset}::{obj['sample_id']}"
            ident = digest(f"method-dev\u241f{candidate}\u241f{raw_id}")
            rows.append(Record("method-dev", candidate, dataset, ident, digest(raw_id), question, context, _split_passages(context)))
            counts[dataset] += 1
    if len(rows) != expected["expected_records"] or dict(sorted(counts.items())) != expected["expected_dataset_counts"]:
        raise RuntimeError(f"candidate coverage mismatch: {candidate}")
    return rows


def _extract_decoded_prompt(decoded: str) -> tuple[str, str]:
    context_marker = "\n\nContext:\n"
    question_marker = "\n\nQuestion: "
    answer_marker = "\nAnswer:"
    if context_marker not in decoded or question_marker not in decoded or answer_marker not in decoded:
        raise RuntimeError("candidate 07 decoded prompt contract mismatch")
    context_and_tail = decoded.split(context_marker, 1)[1]
    context, tail = context_and_tail.rsplit(question_marker, 1)
    question = tail.split(answer_marker, 1)[0]
    if not context.strip() or not question.strip():
        raise RuntimeError("candidate 07 empty decoded projection")
    return context, question


def _load_07(path: Path, manifest_path: Path, expected: dict[str, Any], tokenizer_path: Path) -> list[Record]:
    manifest = json.loads(manifest_path.read_text())
    file_ref = manifest.get("files", {}).get("samples.jsonl", {})
    if (
        manifest.get("status") != "prepared"
        or manifest.get("rows") != expected["expected_records"]
        or file_ref.get("sha256") != expected["sha256"]
        or file_ref.get("nbytes") != expected["nbytes"]
    ):
        raise RuntimeError("candidate 07 manifest binding mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    rows: list[Record] = []
    datasets: Counter[str] = Counter()
    partitions: Counter[str] = Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            dataset = obj["dataset_id"]
            partition = obj["partition"]
            ids = obj["prompt_token_ids"]
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            context, question = _extract_decoded_prompt(decoded)
            if digest(question) != obj["question_sha256"]:
                raise RuntimeError("candidate 07 decoded question binding mismatch")
            raw_id = f"{dataset}::{obj['sample_id']}"
            ident = digest(f"method-dev\u241f07_method_dev_prepared_v1\u241f{raw_id}")
            rows.append(Record("method-dev", "07_method_dev_prepared_v1", dataset, ident, digest(raw_id), question, context, _split_passages(context)))
            datasets[dataset] += 1
            partitions[partition] += 1
    if (
        len(rows) != expected["expected_records"]
        or dict(sorted(datasets.items())) != expected["expected_dataset_counts"]
        or dict(sorted(partitions.items())) != expected["expected_partition_counts"]
    ):
        raise RuntimeError("candidate coverage mismatch: 07_method_dev_prepared_v1")
    return rows


# Strict locked parser projection.  The JSON decoder drops every mapping key
# except the structural fields needed to build id/question/context projections.
# Ground-truth, decomposition, supporting-fact, and outcome fields are never
# retained or accessed by audit code.
_LOCKED_ALLOWED_KEYS = frozenset({"_id", "id", "question", "context", "paragraphs", "title", "paragraph_text"})


def _project_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    return {key: value for key, value in pairs if key in _LOCKED_ALLOWED_KEYS}


def _hotpot_context(value: list[Any]) -> tuple[str, tuple[str, ...]]:
    passages = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError("locked HotpotQA context schema mismatch")
        title, sentences = item
        passages.append(" ".join([str(title), *(str(x) for x in sentences)]))
    return "\n\n".join(passages), tuple(passages)


def _musique_context(value: list[Any]) -> tuple[str, tuple[str, ...]]:
    passages = []
    for item in value:
        if set(item) - {"title", "paragraph_text"} or "paragraph_text" not in item:
            raise RuntimeError("locked MuSiQue paragraph projection mismatch")
        passages.append(" ".join([str(item.get("title", "")), str(item["paragraph_text"])]))
    return "\n\n".join(passages), tuple(passages)


def _locked_record(dataset: str, obj: dict[str, Any], is_hotpot: bool) -> Record:
    required = {"question", "context" if is_hotpot else "paragraphs", "_id" if is_hotpot else "id"}
    if not required.issubset(obj) or set(obj) - _LOCKED_ALLOWED_KEYS:
        raise RuntimeError(f"locked {dataset} projection schema mismatch")
    raw_id = f"{dataset}::{obj['_id' if is_hotpot else 'id']}"
    context, passages = _hotpot_context(obj["context"]) if is_hotpot else _musique_context(obj["paragraphs"])
    return Record("locked-test", "locked", dataset, digest(f"locked-test\u241f{raw_id}"), digest(raw_id), obj["question"], context, passages)


def _load_locked(hotpot_path: Path, musique_path: Path, hp_count: int, mq_count: int) -> list[Record]:
    decoder = json.JSONDecoder(object_pairs_hook=_project_pairs)
    hotpot = decoder.decode(hotpot_path.read_text(encoding="utf-8"))
    if len(hotpot) != hp_count:
        raise RuntimeError("locked HotpotQA count mismatch")
    rows = [_locked_record("hotpotqa_dev_distractor", x, True) for x in hotpot]
    musique = []
    with musique_path.open(encoding="utf-8") as f:
        for line in f:
            musique.append(_locked_record("musique_official_answerable_dev", decoder.decode(line), False))
    if len(musique) != mq_count:
        raise RuntimeError("locked MuSiQue count mismatch")
    return rows + musique


def _load_mapper(config: dict[str, Any]) -> list[Unit]:
    tokenizer = Tokenizer.from_file(config["inputs"]["tokenizer_json"]["path"])
    units = []
    for split, key in (("train", "mapper_fit_train"), ("heldout", "mapper_fit_heldout")):
        spec = config["inputs"][key]
        array = np.load(spec["path"], allow_pickle=False)
        if array.ndim != 2 or len(array) != spec["expected_records"]:
            raise RuntimeError(f"mapper-fit array contract mismatch: {split}")
        for i, ids in enumerate(array):
            ident = digest(f"mapper-fit\u241f{split}\u241f{i:06d}")
            units.append(Unit("mapper-fit", split, "fineweb_edu", ident, ident, "decoded_sequence", normalize(tokenizer.decode(ids.tolist(), skip_special_tokens=True))))
    return units


def _record_units(records: Iterable[Record]) -> list[Unit]:
    result = []
    for r in records:
        result.append(Unit(r.tier, r.candidate, r.dataset, digest(r.identity + "\u241fquestion"), r.identity, "question", normalize(r.question)))
        for i, passage in enumerate(r.passages):
            result.append(Unit(r.tier, r.candidate, r.dataset, digest(r.identity + f"\u241fpassage\u241f{i}"), r.identity, "passage", normalize(passage)))
    return result


def _duplicate_summary(records: list[Record], tier: str) -> dict[str, Any]:
    fields = {
        "id": [r.raw_id_hash for r in records],
        "normalized_question_hash": [digest(normalize(r.question)) for r in records],
        "normalized_question_context_hash": [digest(normalize(r.question) + "\u241f" + normalize(r.context)) for r in records],
    }
    result: dict[str, Any] = {}
    for name, values in fields.items():
        groups = sorted((key, count) for key, count in Counter(values).items() if count > 1)
        result[name] = {
            "duplicate_groups": len(groups),
            "duplicate_records": sum(n for _, n in groups),
            "group_hashes": [digest(f"{tier}\u241f{name}\u241f{k}\u241f{n}") for k, n in groups[:20]],
        }
    return result


def _cross_exact(left: list[Record], right: list[Record]) -> dict[str, Any]:
    extractors = {
        "id": lambda r: r.raw_id_hash,
        "normalized_question_hash": lambda r: digest(normalize(r.question)),
        "normalized_question_context_hash": lambda r: digest(normalize(r.question) + "\u241f" + normalize(r.context)),
    }
    out = {}
    for name, fn in extractors.items():
        overlap = sorted(set(map(fn, left)) & set(map(fn, right)))
        out[name] = {"count": len(overlap), "identity_hashes": [digest(name + "\u241f" + x) for x in overlap[:20]]}
    return out


def _near_pairs(left: list[Unit], right: list[Unit], cfg: dict[str, Any], label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    passage_width = cfg["passage_shingle_words"]
    q_width = cfg["question_shingle_words"]
    indexes: dict[tuple[str, int], dict[str, list[int]]] = {}
    right_sets: list[frozenset[str]] = []
    for j, unit in enumerate(right):
        width = q_width if unit.kind == "question" else passage_width
        s = shingles(unit.text, width)
        right_sets.append(s)
        index = indexes.setdefault((unit.kind, width), defaultdict(list))
        for token in s:
            index[token].append(j)
    flags: list[dict[str, Any]] = []
    top_heap: list[tuple[float, str, dict[str, Any]]] = []
    comparisons = 0
    for lunit in left:
        kinds = ("question",) if lunit.kind == "question" else ("passage",)
        for kind in kinds:
            width = q_width if kind == "question" else passage_width
            ls = shingles(lunit.text, width)
            counts: Counter[int] = Counter()
            for token in ls:
                for j in indexes.get((kind, width), {}).get(token, ()):
                    counts[j] += 1
            for j, inter in counts.items():
                runit = right[j]
                rs = right_sets[j]
                comparisons += 1
                denom = min(len(ls), len(rs))
                containment = inter / denom if denom else 0.0
                union = len(ls) + len(rs) - inter
                jaccard = inter / union if union else 0.0
                shorter, longer = (lunit.text, runit.text) if len(lunit.text) <= len(runit.text) else (runit.text, lunit.text)
                substring = len(shorter) >= cfg["exact_normalized_unit_substring_min_characters"] and shorter in longer
                if kind == "question":
                    threshold = inter >= cfg["question_min_intersection"] and (containment >= cfg["question_min_containment"] or jaccard >= cfg["question_min_jaccard"])
                else:
                    threshold = inter >= cfg["passage_min_intersection"] and (containment >= cfg["passage_min_containment"] or jaccard >= cfg["passage_min_jaccard"])
                item = {
                    "pair_hash": digest(label + "\u241f" + lunit.identity + "\u241f" + runit.identity),
                    "left_identity_hash": lunit.identity,
                    "right_identity_hash": runit.identity,
                    "left_record_identity_hash": lunit.record_identity,
                    "right_record_identity_hash": runit.record_identity,
                    "left_candidate": lunit.candidate,
                    "right_candidate": runit.candidate,
                    "left_dataset": lunit.dataset,
                    "right_dataset": runit.dataset,
                    "kind": kind,
                    "intersection": inter,
                    "containment": round(containment, 12),
                    "jaccard": round(jaccard, 12),
                    "normalized_substring": substring,
                }
                score = max(containment, jaccard)
                key = item["pair_hash"]
                if len(top_heap) < 20:
                    heapq.heappush(top_heap, (score, key, item))
                elif (score, key) > (top_heap[0][0], top_heap[0][1]):
                    heapq.heapreplace(top_heap, (score, key, item))
                if substring or threshold:
                    flags.append(item)
    return sorted(flags, key=lambda x: x["pair_hash"]), [x[2] for x in sorted(top_heap, reverse=True)], comparisons


def _flag_summary(flags: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair = Counter(f"{x['left_dataset']}__{x['right_dataset']}" for x in flags)
    by_candidate_pair = Counter(f"{x['left_candidate']}__{x['right_candidate']}" for x in flags)
    unique_left = {(x["left_dataset"], x["left_record_identity_hash"]) for x in flags}
    unique_right = {(x["right_dataset"], x["right_record_identity_hash"]) for x in flags}
    return {
        "flag_pairs": len(flags),
        "by_dataset_pair": dict(sorted(by_pair.items())),
        "by_candidate_pair": dict(sorted(by_candidate_pair.items())),
        "unique_left_records": len({x["left_record_identity_hash"] for x in flags}),
        "unique_right_records": len({x["right_record_identity_hash"] for x in flags}),
        "unique_records_by_dataset": {
            "left": dict(sorted(Counter(dataset for dataset, _ in unique_left).items())),
            "right": dict(sorted(Counter(dataset for dataset, _ in unique_right).items())),
        },
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    verified = _verify_inputs(config)
    specs = config["inputs"]
    method_01 = _load_method(Path(specs["candidate_01_oracle_prepare_v1"]["path"]), "01_oracle_prepare_v1", specs["candidate_01_oracle_prepare_v1"])
    method_03 = _load_method(Path(specs["candidate_03_tail_refresh_v2"]["path"]), "03_tail_refresh_v2", specs["candidate_03_tail_refresh_v2"])
    method_07 = _load_07(
        Path(specs["candidate_07_method_dev_samples"]["path"]),
        Path(specs["candidate_07_method_dev_manifest"]["path"]),
        specs["candidate_07_method_dev_samples"],
        Path(specs["tokenizer_json"]["path"]),
    )
    method = method_01 + method_03 + method_07
    locked = _load_locked(
        Path(specs["locked_hotpotqa_dev_distractor"]["path"]),
        Path(specs["locked_musique_official_answerable_dev"]["path"]),
        specs["locked_hotpotqa_dev_distractor"]["expected_records"],
        specs["locked_musique_official_answerable_dev"]["expected_records"],
    )
    mapper_units = _load_mapper(config)
    method_units = _record_units(method)
    locked_units = _record_units(locked)

    shared_exact = _cross_exact(method_01, method_03)
    shared_01_07 = _cross_exact(method_01, method_07)
    shared_03_07 = _cross_exact(method_03, method_07)
    cross_exact = _cross_exact(method, locked)
    sim = config["similarity"]
    mapper_flags, mapper_top, mapper_comparisons = _near_pairs(mapper_units, method_units, sim, "mapper-fit_to_method-dev")
    method_flags, method_top, method_comparisons = _near_pairs(method_units, locked_units, sim, "method-dev_to_locked-test")
    exact_cross_flag_count = sum(x["count"] for x in cross_exact.values())
    flag_count = exact_cross_flag_count + len(mapper_flags) + len(method_flags)

    candidate_coverage = {
        "01_oracle_prepare_v1": {"records": len(method_01), "dataset_counts": dict(sorted(Counter(r.dataset for r in method_01).items())), "covered": True},
        "03_tail_refresh_v2": {"records": len(method_03), "dataset_counts": dict(sorted(Counter(r.dataset for r in method_03).items())), "covered": True},
        "07_method_dev_prepared_v1": {"records": len(method_07), "dataset_counts": dict(sorted(Counter(r.dataset for r in method_07).items())), "partition_counts": dict(sorted(Counter(json.loads(x)["partition"] for x in Path(specs["candidate_07_method_dev_samples"]["path"]).read_text().splitlines()).items())), "covered": True},
    }
    receipt = {
        "schema_version": 1,
        "audit_id": "complete_g0_mapper_fit_method_dev_locked_v1",
        "config_sha256": sha256_file(config_path),
        "input_verification": verified,
        "privacy": {
            "locked_parser": "strict id/question/context projection; all other mapping fields dropped and never accessed",
            "raw_text_written": False,
            "references_written": False,
            "model_outputs_read_or_written": False,
            "locked_outcome_metrics_read_or_written": False,
        },
        "coverage": {
            "candidate_inventory": candidate_coverage,
            "tier_records": {"mapper-fit": len(mapper_units), "method-dev": len(method), "locked-test": len(locked)},
            "tier_units": {"mapper-fit_decoded_sequences": len(mapper_units), "method-dev_questions_and_passages": len(method_units), "locked_test_questions_and_passages": len(locked_units)},
        },
        "within_tier_exact_inventory": {
            "mapper-fit": {
                "id": {"duplicate_groups": 0, "duplicate_records": 0, "group_hashes": []},
                "normalized_decoded_sequence_hash": {
                    "duplicate_groups": sum(n > 1 for n in Counter(digest(u.text) for u in mapper_units).values()),
                    "duplicate_records": sum(n for n in Counter(digest(u.text) for u in mapper_units).values() if n > 1),
                    "group_hashes": [digest("mapper-fit\u241fdecoded\u241f" + k + f"\u241f{n}") for k, n in sorted(Counter(digest(u.text) for u in mapper_units).items()) if n > 1][:20],
                },
                "normalized_question_hash": "not_applicable_no_question_field",
                "normalized_question_context_hash": "not_applicable_no_question_or_context_fields"
            },
            "method-dev": _duplicate_summary(method, "method-dev"),
            "locked-test": _duplicate_summary(locked, "locked-test"),
            "interpretation": "within-tier repetitions are inventory observations, not cross-tier leakage flags",
        },
        "shared_method_dev_candidate_inventory": {
            "classification": "same_method_dev_tier_shared_samples_not_cross_tier_leakage",
            "candidate_pairs": {
                "01_oracle_prepare_v1__03_tail_refresh_v2": shared_exact,
                "01_oracle_prepare_v1__07_method_dev_prepared_v1": shared_01_07,
                "03_tail_refresh_v2__07_method_dev_prepared_v1": shared_03_07
            },
        },
        "cross_tier_exact": {"method-dev_to_locked-test": cross_exact, "flag_count": exact_cross_flag_count},
        "cross_tier_similarity": {
            "mapper-fit_to_method-dev": {"candidate_comparisons": mapper_comparisons, "flag_count": len(mapper_flags), "flag_summary": _flag_summary(mapper_flags), "flags": mapper_flags, "top_nonprivate_candidates": mapper_top},
            "method-dev_to_locked-test": {"candidate_comparisons": method_comparisons, "flag_count": len(method_flags), "flag_summary": _flag_summary(method_flags), "flags": method_flags, "top_nonprivate_candidates": method_top},
        },
        "thresholds": sim,
        "result": {"cross_tier_flag_count": flag_count, "pass": flag_count == 0},
        "scope": {
            "mapper_fit": "FineWeb-Edu train500 plus heldout50 decoded token sequences",
            "method_dev": "01 prepare_v1 Hotpot256+MuSiQue256, 03 prepared_v2 Hotpot192+MuSiQue192, and 07 prepared_v1 screen/confirm Hotpot384+MuSiQue768",
            "locked_test": "HotpotQA distractor dev7405 and official answerable MuSiQue dev2417 identity/question/context projection only",
            "closes_complete_g0": flag_count == 0 and all(x["covered"] for x in candidate_coverage.values()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "complete_g0_receipt.json"
    atomic_json(receipt_path, receipt)
    sidecar = {"path": str(receipt_path), "nbytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path)}
    atomic_json(output_dir / "complete_g0_receipt.json.sha256.json", sidecar)
    return receipt
