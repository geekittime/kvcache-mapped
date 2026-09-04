#!/usr/bin/env python3
"""Deterministic CPU-only audit of mapper-fit text against method-dev passages.

The persisted receipt deliberately contains identities and numeric summaries only.
It never serializes decoded calibration text, method-dev source text, or answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from tokenizers import Tokenizer


PASSAGE_PATTERN = re.compile(r"(?m)^\[Passage ([1-9][0-9]*)\]\n")
MASK64 = (1 << 64) - 1
ROLLING_BASE = 1_000_003
EXACT_ANCHOR_CHARS = 30


class AuditError(RuntimeError):
    """Raised when a locked contract cannot be verified (fail closed)."""


@dataclass(frozen=True)
class CalibrationText:
    split: str
    sequence_index: int
    token_ids_sha256: str
    normalized_sha256: str
    normalized: str
    shingles: frozenset[bytes]


@dataclass(frozen=True)
class Passage:
    sample_id: str
    dataset_id: str
    source_record_sha256: str
    passage_index: int
    normalized_sha256: str
    normalized: str
    shingles: frozenset[bytes]


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"path": str(path.resolve()), "nbytes": size, "sha256": digest.hexdigest()}


def verify_file(spec: dict[str, Any], label: str) -> dict[str, Any]:
    if set(spec) != {"path", "sha256"}:
        raise AuditError(f"{label}: input spec must contain exactly path and sha256")
    path = Path(spec["path"])
    if not path.is_file():
        raise AuditError(f"{label}: missing locked input: {path}")
    actual = file_identity(path)
    if actual["sha256"] != spec["sha256"]:
        raise AuditError(
            f"{label}: SHA-256 mismatch: expected {spec['sha256']}, got {actual['sha256']}"
        )
    return actual


def normalize_text(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in folded).split())


def word_shingles(normalized: str, size: int = 5) -> frozenset[bytes]:
    words = normalized.split()
    if len(words) < size:
        return frozenset()
    return frozenset(
        hashlib.blake2b("\x1f".join(words[i : i + size]).encode("utf-8"), digest_size=16).digest()
        for i in range(len(words) - size + 1)
    )


def digest_int32_tokens(tokens: np.ndarray) -> str:
    canonical = np.asarray(tokens, dtype="<i4", order="C")
    return sha256_bytes(canonical.tobytes(order="C"))


def load_calibration(
    input_paths: dict[str, Path], tokenizer_path: Path, expected: dict[str, int], shingle_size: int
) -> list[CalibrationText]:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    result: list[CalibrationText] = []
    for split, key, expected_rows in (
        ("train", "train_tokens", expected["train_sequences"]),
        ("heldout", "heldout_tokens", expected["heldout_sequences"]),
    ):
        array = np.load(input_paths[key], mmap_mode="r", allow_pickle=False)
        wanted_shape = (expected_rows, expected["tokens_per_sequence"])
        if array.shape != wanted_shape or array.dtype != np.dtype("int32"):
            raise AuditError(
                f"{key}: expected int32 shape {wanted_shape}, got {array.dtype} shape {array.shape}"
            )
        for sequence_index, row in enumerate(array):
            ids = [int(token) for token in row]
            try:
                decoded = tokenizer.decode(ids, skip_special_tokens=False)
            except Exception as exc:  # pragma: no cover - tokenizer supplies details
                raise AuditError(f"{key}[{sequence_index}]: token decode failed: {exc}") from exc
            normalized = normalize_text(decoded)
            if not normalized:
                raise AuditError(f"{key}[{sequence_index}]: normalization produced empty text")
            result.append(
                CalibrationText(
                    split=split,
                    sequence_index=sequence_index,
                    token_ids_sha256=digest_int32_tokens(row),
                    normalized_sha256=sha256_bytes(normalized.encode("utf-8")),
                    normalized=normalized,
                    shingles=word_shingles(normalized, shingle_size),
                )
            )
    return result


def split_explicit_passages(context: str) -> list[tuple[int, str]]:
    markers = list(PASSAGE_PATTERN.finditer(context))
    if not markers:
        return []
    indices = [int(marker.group(1)) for marker in markers]
    if indices != list(range(1, len(indices) + 1)):
        raise AuditError(f"non-contiguous explicit passage indices: {indices}")
    passages: list[tuple[int, str]] = []
    for offset, marker in enumerate(markers):
        end = markers[offset + 1].start() if offset + 1 < len(markers) else len(context)
        block = context[marker.end() : end].strip()
        title_line, separator, text = block.partition("\n")
        if not separator or not title_line.startswith("Title: ") or not title_line.removeprefix("Title: ").strip():
            raise AuditError(f"explicit passage {indices[offset]} lacks a non-empty Title line")
        text = text.strip()
        if not text:
            raise AuditError(f"empty explicit passage {indices[offset]}")
        passages.append((indices[offset], text))
    return passages


def load_passages(
    selected_path: Path, manifest_path: Path, expected: dict[str, int], shingle_size: int
) -> tuple[list[Passage], int, int]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_manifest = manifest.get("files", {}).get("selected_samples.jsonl", {})
    if selected_manifest.get("rows") != expected["method_dev_rows"]:
        raise AuditError("manifest row count disagrees with locked expectation")
    selected_identity = file_identity(selected_path)
    if selected_manifest.get("sha256") != selected_identity["sha256"]:
        raise AuditError("manifest selected_samples SHA-256 disagrees with locked file")

    result: list[Passage] = []
    seen_samples: set[str] = set()
    row_count = 0
    eligible_rows = 0
    with selected_path.open("r", encoding="utf-8") as handle:
        for row_count, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditError(f"selected row {row_count}: invalid JSON: {exc}") from exc
            required = {"sample_id", "dataset_id", "source_record_sha256", "context"}
            if not required.issubset(row) or not all(isinstance(row[key], str) for key in required):
                raise AuditError(f"selected row {row_count}: missing or invalid identity/context field")
            if row["sample_id"] in seen_samples:
                raise AuditError(f"selected row {row_count}: duplicate sample_id {row['sample_id']}")
            seen_samples.add(row["sample_id"])
            explicit = split_explicit_passages(row["context"])
            if explicit:
                eligible_rows += 1
            for passage_index, raw_text in explicit:
                normalized = normalize_text(raw_text)
                if not normalized:
                    raise AuditError(
                        f"selected row {row_count} passage {passage_index}: normalization produced empty text"
                    )
                result.append(
                    Passage(
                        sample_id=row["sample_id"],
                        dataset_id=row["dataset_id"],
                        source_record_sha256=row["source_record_sha256"],
                        passage_index=passage_index,
                        normalized_sha256=sha256_bytes(normalized.encode("utf-8")),
                        normalized=normalized,
                        shingles=word_shingles(normalized, shingle_size),
                    )
                )
    if row_count != expected["method_dev_rows"]:
        raise AuditError(f"expected {expected['method_dev_rows']} selected rows, got {row_count}")
    if eligible_rows != expected["eligible_method_dev_rows"]:
        raise AuditError(
            f"expected {expected['eligible_method_dev_rows']} explicit-passage rows, got {eligible_rows}"
        )
    if len(result) != expected["passages"]:
        raise AuditError(f"expected {expected['passages']} explicit passages, got {len(result)}")
    return result, row_count, eligible_rows


def rolling_hashes(text: str, width: int) -> Iterator[tuple[int, int]]:
    """Yield (start, uint64 polynomial hash) for all fixed-width windows."""
    if len(text) < width:
        return
    power = pow(ROLLING_BASE, width - 1, 1 << 64)
    value = 0
    for ch in text[:width]:
        value = ((value * ROLLING_BASE) + ord(ch) + 1) & MASK64
    yield 0, value
    for start in range(1, len(text) - width + 1):
        outgoing = ord(text[start - 1]) + 1
        incoming = ord(text[start + width - 1]) + 1
        value = ((value - outgoing * power) * ROLLING_BASE + incoming) & MASK64
        yield start, value


def exact_common_length_around_anchor(a: str, a_pos: int, b: str, b_pos: int, width: int) -> int:
    if a[a_pos : a_pos + width] != b[b_pos : b_pos + width]:
        return 0
    left = 0
    while a_pos - left > 0 and b_pos - left > 0 and a[a_pos - left - 1] == b[b_pos - left - 1]:
        left += 1
    right = width
    while a_pos + right < len(a) and b_pos + right < len(b) and a[a_pos + right] == b[b_pos + right]:
        right += 1
    return left + right


def find_exact_pairs(
    calibration: Sequence[CalibrationText], passages: Sequence[Passage], min_chars: int
) -> dict[tuple[int, int], int]:
    """Find all >=min_chars exact pairs via a complete aligned-anchor index.

    Every min_chars interval contains a calibration-grid anchor because the anchor
    width is at most half min_chars. Hash hits are always verified byte-for-byte.
    """
    anchor_width = min(EXACT_ANCHOR_CHARS, min_chars // 2)
    if anchor_width < 1:
        raise AuditError("exact_min_normalized_chars must be at least 2")
    anchors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for calibration_index, item in enumerate(calibration):
        hashes = dict(rolling_hashes(item.normalized, anchor_width))
        for position in range(0, len(item.normalized) - anchor_width + 1, anchor_width):
            anchors[hashes[position]].append((calibration_index, position))

    matches: dict[tuple[int, int], int] = {}
    for passage_list_index, passage in enumerate(passages):
        for passage_pos, value in rolling_hashes(passage.normalized, anchor_width):
            for calibration_index, calibration_pos in anchors.get(value, ()):
                key = (calibration_index, passage_list_index)
                if key in matches:
                    continue
                common = exact_common_length_around_anchor(
                    calibration[calibration_index].normalized,
                    calibration_pos,
                    passage.normalized,
                    passage_pos,
                    anchor_width,
                )
                if common >= min_chars:
                    matches[key] = common
    return matches


def pair_metrics(
    calibration: Sequence[CalibrationText], passages: Sequence[Passage]
) -> dict[tuple[int, int], tuple[int, float, float]]:
    inverted: dict[bytes, list[int]] = defaultdict(list)
    for calibration_index, item in enumerate(calibration):
        for digest in item.shingles:
            inverted[digest].append(calibration_index)
    metrics: dict[tuple[int, int], tuple[int, float, float]] = {}
    for passage_index, passage in enumerate(passages):
        intersections: Counter[int] = Counter()
        for digest in passage.shingles:
            intersections.update(inverted.get(digest, ()))
        for calibration_index, intersection in intersections.items():
            passage_containment = intersection / len(passage.shingles) if passage.shingles else 0.0
            union = len(passage.shingles) + len(calibration[calibration_index].shingles) - intersection
            jaccard = intersection / union if union else 0.0
            metrics[(calibration_index, passage_index)] = (
                intersection,
                passage_containment,
                jaccard,
            )
    return metrics


def public_identity(item: CalibrationText | Passage) -> dict[str, Any]:
    if isinstance(item, CalibrationText):
        return {
            "split": item.split,
            "sequence_index": item.sequence_index,
            "token_ids_sha256": item.token_ids_sha256,
            "normalized_text_sha256": item.normalized_sha256,
        }
    return {
        "sample_id": item.sample_id,
        "dataset_id": item.dataset_id,
        "source_record_sha256": item.source_record_sha256,
        "passage_index": item.passage_index,
        "normalized_text_sha256": item.normalized_sha256,
    }


def audit_pairs(
    calibration: Sequence[CalibrationText], passages: Sequence[Passage], algorithm: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exact_pairs = find_exact_pairs(calibration, passages, algorithm["exact_min_normalized_chars"])
    metrics = pair_metrics(calibration, passages)
    candidate_keys = set(metrics) | set(exact_pairs)
    rows: list[dict[str, Any]] = []
    counts = {"exact_substring": 0, "intersection_and_passage_containment": 0, "jaccard": 0}
    flagged_pairs = 0
    for key in candidate_keys:
        calibration_index, passage_index = key
        intersection, containment, jaccard = metrics.get(key, (0, 0.0, 0.0))
        criteria = {
            "exact_substring": key in exact_pairs,
            "intersection_and_passage_containment": intersection >= algorithm["near_min_intersection"]
            and containment >= algorithm["near_min_passage_containment"],
            "jaccard": jaccard >= algorithm["near_min_jaccard"],
        }
        flagged = any(criteria.values())
        if flagged:
            flagged_pairs += 1
            for criterion, triggered in criteria.items():
                counts[criterion] += int(triggered)
        rows.append(
            {
                "calibration": public_identity(calibration[calibration_index]),
                "passage": public_identity(passages[passage_index]),
                "metrics": {
                    "exact_substring_ge_min": criteria["exact_substring"],
                    "verified_exact_common_chars": exact_pairs.get(key, 0),
                    "shingle_intersection": intersection,
                    "passage_containment": round(containment, 12),
                    "jaccard": round(jaccard, 12),
                },
                "flagged": flagged,
                "triggered_criteria": [name for name, triggered in criteria.items() if triggered],
            }
        )
    rows.sort(
        key=lambda row: (
            not row["flagged"],
            -max(row["metrics"]["passage_containment"], row["metrics"]["jaccard"]),
            -row["metrics"]["shingle_intersection"],
            row["calibration"]["split"],
            row["calibration"]["sequence_index"],
            row["passage"]["sample_id"],
            row["passage"]["passage_index"],
        )
    )
    top = rows[: algorithm["top_candidates"]]
    maximum = {
        "verified_exact_common_chars_ge_min": max(exact_pairs.values(), default=0),
        "shingle_intersection": max((row["metrics"]["shingle_intersection"] for row in rows), default=0),
        "passage_containment": max((row["metrics"]["passage_containment"] for row in rows), default=0.0),
        "jaccard": max((row["metrics"]["jaccard"] for row in rows), default=0.0),
    }
    max_containment_candidate = (
        max(
            rows,
            key=lambda row: (
                row["metrics"]["passage_containment"],
                row["metrics"]["shingle_intersection"],
                row["metrics"]["jaccard"],
            ),
        )
        if rows
        else None
    )
    summary = {
        "flagged_pairs": flagged_pairs,
        "flag_counts_by_criterion": counts,
        "maxima_over_nonzero_candidates": maximum,
        "max_passage_containment_candidate": max_containment_candidate,
        "nonzero_shingle_candidate_pairs": len(metrics),
        "exact_candidate_pairs": len(exact_pairs),
    }
    return top, summary


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_receipt_and_sidecar(output_dir: Path, receipt: dict[str, Any]) -> tuple[Path, Path]:
    receipt_path = output_dir / "receipt.json"
    receipt_bytes = canonical_json_bytes(receipt)
    atomic_write(receipt_path, receipt_bytes)
    sidecar = {
        "schema_version": 1,
        "file": receipt_path.name,
        "nbytes": len(receipt_bytes),
        "sha256": sha256_bytes(receipt_bytes),
    }
    sidecar_path = output_dir / "receipt.json.sha256.json"
    atomic_write(sidecar_path, canonical_json_bytes(sidecar))
    return receipt_path, sidecar_path


def validate_config(config: dict[str, Any]) -> None:
    required_top = {"schema_version", "audit_id", "inputs", "expected", "algorithm", "output_dir"}
    if set(config) != required_top or config["schema_version"] != 1:
        raise AuditError("unsupported or malformed config schema")
    required_inputs = {
        "train_tokens",
        "heldout_tokens",
        "tokenizer_json",
        "method_dev_selected",
        "method_dev_manifest",
    }
    if set(config["inputs"]) != required_inputs:
        raise AuditError("config must name exactly the five locked inputs")
    algorithm = config["algorithm"]
    locked_algorithm = {
        "normalization": "NFKC + casefold + map non-alphanumeric runs to one ASCII space + strip",
        "passage_marker_regex": r"(?m)^\[Passage ([1-9][0-9]*)\]\n",
        "word_shingle_size": 5,
        "shingle_hash": "BLAKE2b-128",
        "exact_min_normalized_chars": 120,
        "near_min_intersection": 20,
        "near_min_passage_containment": 0.5,
        "near_min_jaccard": 0.3,
        "top_candidates": 25,
    }
    if algorithm != locked_algorithm:
        raise AuditError("algorithm parameters differ from the frozen v1 contract")


def execute(config_path: Path, script_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load config: {exc}") from exc
    validate_config(config)
    input_identities = {
        label: verify_file(spec, label) for label, spec in sorted(config["inputs"].items())
    }
    paths = {label: Path(identity["path"]) for label, identity in input_identities.items()}
    calibration = load_calibration(
        paths, paths["tokenizer_json"], config["expected"], config["algorithm"]["word_shingle_size"]
    )
    passages, method_dev_rows, eligible_rows = load_passages(
        paths["method_dev_selected"],
        paths["method_dev_manifest"],
        config["expected"],
        config["algorithm"]["word_shingle_size"],
    )
    top_candidates, summary = audit_pairs(calibration, passages, config["algorithm"])
    receipt = {
        "schema_version": 1,
        "audit_id": config["audit_id"],
        "status": "pass" if summary["flagged_pairs"] == 0 else "flagged",
        "scope": {
            "gate": "tail_pilot_mapper_fit_vs_selected_method_dev_subgate",
            "covered": ["mapper-fit FineWeb calibration train+heldout vs 03 selected explicit passages"],
            "not_covered": [
                "method-dev vs prospective locked-test inputs",
                "locked-test identity/hash collision checks",
                "the complete research-protocol G0 gate",
            ],
            "closes_complete_g0": False,
        },
        "execution": {
            "device": "cpu",
            "gpu_codepaths": False,
            "deterministic": True,
            "fail_closed": True,
        },
        "privacy": {
            "contains_decoded_text": False,
            "contains_method_dev_original_text": False,
            "contains_answers": False,
            "identity_and_numeric_metrics_only": True,
        },
        "code": {"audit_script": file_identity(script_path), "config": file_identity(config_path)},
        "inputs": input_identities,
        "algorithm": config["algorithm"],
        "population": {
            "calibration_sequences": len(calibration),
            "train_sequences": sum(item.split == "train" for item in calibration),
            "heldout_sequences": sum(item.split == "heldout" for item in calibration),
            "method_dev_rows": method_dev_rows,
            "eligible_explicit_passage_rows": eligible_rows,
            "passages": len(passages),
            "pair_comparisons": len(calibration) * len(passages),
        },
        "result": summary,
        "top_candidates": top_candidates,
    }
    output_dir = Path(config["output_dir"])
    receipt_path, sidecar_path = write_receipt_and_sidecar(output_dir, receipt)
    return receipt_path, sidecar_path, receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt_path, sidecar_path, receipt = execute(args.config.resolve(), Path(__file__).resolve())
    except AuditError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "flagged_pairs": receipt["result"]["flagged_pairs"],
                "receipt": str(receipt_path),
                "sidecar": str(sidecar_path),
            },
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
