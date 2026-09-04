"""Deterministic calibration-data preparation for KV-cache transfer.

The paper specifies 500 FineWeb-Edu sequences of 1,024 tokens and a token
stride of four, but it does not publish the exact dataset revision, shard,
documents, or tokenization/window-selection procedure.  This module therefore
keeps those choices explicit in the configuration and output manifest.  The
selection used by this reproduction is a stable, order-independent bottom-k
SHA256 reservoir over eligible *documents*, followed by a second independent
hash for selecting one within-document window.

No document is concatenated with another document.  Tokenization is performed
with ``add_special_tokens=False``.  Those are local reproduction conventions,
not claims about unpublished details of the paper.
"""

from __future__ import annotations

import glob
import hashlib
import heapq
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "sha256-bottom-k-v1"
WINDOW_ALGORITHM = "sha256-modulo-valid-offsets-v1"
TOKEN_HASH_ALGORITHM = "sha256-little-endian-uint32-v1"

DEFAULT_LOCAL_CONVENTIONS = (
    "HuggingFaceFW/fineweb-edu sample-10BT is a local dataset-config choice; "
    "the paper only names FineWeb-Edu.",
    "The pinned dataset revision and Parquet shard are local choices because "
    "the paper does not publish them.",
    "Bottom-k SHA256 document sampling and hash-derived within-document "
    "windows are local deterministic conventions.",
    "Documents are not concatenated and tokenization uses "
    "add_special_tokens=false; the paper does not publish these details.",
    "The last 50 selected documents are a local held-out diagnostic split and "
    "are not part of the paper's 500-sequence calibration set.",
)


class CalibrationDataError(RuntimeError):
    """Base class for deterministic calibration preparation failures."""


class IncompleteSourceError(CalibrationDataError):
    """Raised when a source has too few eligible unique documents."""


class ResumeMismatchError(CalibrationDataError):
    """Raised when an in-progress output belongs to a different run spec."""


class ArtifactVerificationError(CalibrationDataError):
    """Raised when a completed output fails shape or checksum validation."""


@dataclass(frozen=True)
class SourceRecord:
    """A record plus auditable source-location information."""

    data: Mapping[str, Any]
    locator: str
    record_index: int


@dataclass(frozen=True)
class WindowCandidate:
    """One eligible document's deterministically selected token window."""

    selection_hash: str
    record_uid: str
    raw_text_sha256: str
    token_ids_sha256: str
    document_token_count: int
    token_offset: int
    token_ids: tuple[int, ...]
    identity: Mapping[str, Any]
    source_locator: str
    source_record_index: int

    def checkpoint_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["token_ids"] = list(self.token_ids)
        return result

    @classmethod
    def from_checkpoint_dict(cls, value: Mapping[str, Any]) -> "WindowCandidate":
        return cls(
            selection_hash=str(value["selection_hash"]),
            record_uid=str(value["record_uid"]),
            raw_text_sha256=str(value["raw_text_sha256"]),
            token_ids_sha256=str(value["token_ids_sha256"]),
            document_token_count=int(value["document_token_count"]),
            token_offset=int(value["token_offset"]),
            token_ids=tuple(int(token) for token in value["token_ids"]),
            identity=dict(value["identity"]),
            source_locator=str(value["source_locator"]),
            source_record_index=int(value["source_record_index"]),
        )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible data with a stable byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_selection_hash(seed: str | int, record_uid: str) -> str:
    payload = b"kvcache-transfer/selection/v1\0"
    payload += str(seed).encode("utf-8") + b"\0" + record_uid.encode("ascii")
    return sha256_bytes(payload)


def stable_window_offset(
    seed: str | int, record_uid: str, document_length: int, sequence_length: int
) -> int:
    if document_length < sequence_length:
        raise ValueError("document is shorter than the requested sequence")
    number_of_offsets = document_length - sequence_length + 1
    payload = b"kvcache-transfer/window/v1\0"
    payload += str(seed).encode("utf-8") + b"\0" + record_uid.encode("ascii")
    return int(sha256_bytes(payload), 16) % number_of_offsets


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    tokens = np.asarray(token_ids, dtype="<u4")
    return sha256_bytes(tokens.tobytes(order="C"))


def _json_safe_identity_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError):
        return str(value)
    return value


def document_identity(
    record: Mapping[str, Any],
    text: str,
    identity_fields: Sequence[str],
) -> tuple[str, str, dict[str, Any]]:
    """Return (record UID, text SHA256, recorded stable identity fields)."""

    text_hash = sha256_bytes(text.encode("utf-8"))
    identity = {
        field: _json_safe_identity_value(record[field])
        for field in identity_fields
        if field in record and record[field] not in (None, "")
    }
    # Including the text hash prevents a reused URL/id with changed content from
    # silently selecting a different calibration sequence under the same UID.
    material = {"identity": identity, "raw_text_sha256": text_hash}
    return sha256_bytes(canonical_json_bytes(material)), text_hash, identity


class BottomKReservoir:
    """Memory-bounded bottom-k set keyed by a SHA256 integer.

    Selection is independent of stream order.  A dictionary suppresses repeated
    occurrences of the same record UID while keeping memory bounded by k.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: dict[str, WindowCandidate] = {}
        self._heap: list[tuple[int, str]] = []

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def composite_key(selection_hash: str, record_uid: str) -> int:
        """Return the total-order key used by the bottom-k reservoir."""

        if len(selection_hash) != 64 or len(record_uid) != 64:
            raise ValueError("selection_hash and record_uid must be SHA256 hex digests")
        try:
            return (int(selection_hash, 16) << 256) | int(record_uid, 16)
        except ValueError as error:
            raise ValueError("selection_hash and record_uid must be hexadecimal") from error

    def could_accept(self, selection_hash: str, record_uid: str) -> bool:
        """Whether an eligible UID could currently enter the bottom-k set.

        This predicate permits a major but exact streaming optimization.  Once
        the reservoir is full, its worst key can only decrease.  A document
        whose key is already no better can therefore never enter the final
        set, regardless of its tokenized length, and expensive tokenization is
        unnecessary.  Eligibility is still checked for every document which
        can affect the result.
        """

        if record_uid in self._items:
            return False
        if len(self._items) < self.capacity:
            return True
        return self.composite_key(selection_hash, record_uid) < -self._heap[0][0]

    def add(self, candidate: WindowCandidate) -> bool:
        if candidate.record_uid in self._items:
            return False
        # The UID is a second 256-bit deterministic tie-breaker.  SHA256
        # collisions are fantastically unlikely, but defining the tie behavior
        # keeps the algorithm total and independently implementable.
        composite_key = self.composite_key(
            candidate.selection_hash, candidate.record_uid
        )
        if len(self._items) < self.capacity:
            self._items[candidate.record_uid] = candidate
            heapq.heappush(self._heap, (-composite_key, candidate.record_uid))
            return True

        worst_key = -self._heap[0][0]
        if composite_key >= worst_key:
            return False
        _, worst_uid = heapq.heapreplace(
            self._heap, (-composite_key, candidate.record_uid)
        )
        del self._items[worst_uid]
        self._items[candidate.record_uid] = candidate
        return True

    def sorted_candidates(self) -> list[WindowCandidate]:
        return sorted(
            self._items.values(),
            key=lambda item: (item.selection_hash, item.record_uid),
        )

    def checkpoint_values(self) -> list[dict[str, Any]]:
        return [item.checkpoint_dict() for item in self.sorted_candidates()]

    @classmethod
    def from_checkpoint(
        cls, capacity: int, values: Sequence[Mapping[str, Any]]
    ) -> "BottomKReservoir":
        reservoir = cls(capacity)
        for value in values:
            reservoir.add(WindowCandidate.from_checkpoint_dict(value))
        if len(reservoir) != len(values):
            raise ArtifactVerificationError(
                "resume checkpoint contains duplicate record UIDs"
            )
        return reservoir


def iter_jsonl_records(
    paths: Sequence[str | os.PathLike[str]],
) -> Iterator[SourceRecord]:
    """Stream mappings from one or more local JSONL files."""

    for raw_path in paths:
        path = Path(raw_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CalibrationDataError(
                        f"invalid JSON at {path}:{index + 1}: {error}"
                    ) from error
                if not isinstance(value, Mapping):
                    raise CalibrationDataError(
                        f"JSONL record at {path}:{index + 1} is not an object"
                    )
                yield SourceRecord(value, str(path), index)


def iter_parquet_records(
    paths: Sequence[str | os.PathLike[str]], batch_size: int = 256
) -> Iterator[SourceRecord]:
    """Stream mappings from local Parquet row groups without loading a file."""

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment validation
        raise CalibrationDataError("pyarrow is required for Parquet input") from error

    for raw_path in paths:
        path = Path(raw_path).resolve()
        parquet = pq.ParquetFile(path)
        index = 0
        for batch in parquet.iter_batches(batch_size=batch_size):
            for value in batch.to_pylist():
                if not isinstance(value, Mapping):
                    raise CalibrationDataError(
                        f"Parquet row {index} in {path} is not a mapping"
                    )
                yield SourceRecord(value, str(path), index)
                index += 1


def _validate_hf_source_metadata(source: Mapping[str, Any]) -> None:
    expected = source.get("expected_lfs_sha256") or {}
    expected_sizes = source.get("expected_sizes") or {}
    if not expected and not expected_sizes:
        return
    try:
        from huggingface_hub import HfApi
    except ImportError as error:  # pragma: no cover - environment validation
        raise CalibrationDataError(
            "huggingface-hub is required to validate HF shard metadata"
        ) from error

    info = HfApi().dataset_info(
        repo_id=str(source["dataset_name"]),
        revision=str(source["revision"]),
        files_metadata=True,
    )
    siblings = {sibling.rfilename: sibling for sibling in info.siblings}
    for relative_path, expected_hash in expected.items():
        if relative_path not in siblings:
            raise ArtifactVerificationError(
                f"configured HF shard does not exist: {relative_path}"
            )
        lfs = getattr(siblings[relative_path], "lfs", None)
        actual_hash = getattr(lfs, "sha256", None)
        if actual_hash != expected_hash:
            raise ArtifactVerificationError(
                f"HF LFS SHA256 mismatch for {relative_path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
    for relative_path, expected_size in expected_sizes.items():
        if relative_path not in siblings:
            raise ArtifactVerificationError(
                f"configured HF shard does not exist: {relative_path}"
            )
        actual_size = getattr(siblings[relative_path], "size", None)
        if actual_size != int(expected_size):
            raise ArtifactVerificationError(
                f"HF size mismatch for {relative_path}: "
                f"expected {expected_size}, got {actual_size}"
            )


def iter_hf_streaming_records(source: Mapping[str, Any]) -> Iterator[SourceRecord]:
    """Stream an explicitly pinned Hugging Face dataset source."""

    revision = str(source.get("revision", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise CalibrationDataError(
            "HF streaming requires an immutable 40-hex commit revision"
        )
    if source.get("verify_remote_files_metadata", True):
        _validate_hf_source_metadata(source)

    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - environment validation
        raise CalibrationDataError("datasets is required for HF input") from error

    data_files = source.get("data_files")
    if isinstance(data_files, list):
        data_files = {str(source.get("split", "train")): data_files}
    dataset = load_dataset(
        str(source["dataset_name"]),
        name=source.get("config_name"),
        split=str(source.get("split", "train")),
        revision=revision,
        data_files=data_files,
        streaming=True,
    )
    locator = (
        f"hf://datasets/{source['dataset_name']}@{revision}/"
        f"{source.get('config_name', '')}/{source.get('split', 'train')}"
    )
    for index, value in enumerate(dataset):
        if not isinstance(value, Mapping):
            raise CalibrationDataError(f"HF record {index} is not a mapping")
        yield SourceRecord(value, locator, index)


def iter_pinned_hf_local_mirror_records(
    source: Mapping[str, Any], base_dir: str | os.PathLike[str] = "."
) -> Iterator[SourceRecord]:
    """Read a local mirror while preserving and verifying formal HF identity.

    A mirror is not a local-source/debug override: the source manifest still
    pins the HF repository, commit and remote Parquet path.  Before yielding any
    row this function requires the local file's complete SHA256 and byte size to
    equal the pinned Hugging Face LFS metadata.
    """

    revision = str(source.get("revision", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise CalibrationDataError(
            "formal HF local mirror requires an immutable 40-hex commit revision"
        )
    remote_paths = source.get("data_files") or []
    if isinstance(remote_paths, Mapping):
        flattened: list[str] = []
        for values in remote_paths.values():
            flattened.extend(values if isinstance(values, list) else [values])
        remote_paths = flattened
    if len(remote_paths) != 1:
        raise CalibrationDataError(
            "local_mirror_path currently requires exactly one pinned HF data file"
        )
    remote_path = str(remote_paths[0])
    mirror = Path(str(source["local_mirror_path"]))
    mirror = mirror.resolve() if mirror.is_absolute() else (Path(base_dir) / mirror).resolve()
    if not mirror.is_file():
        raise FileNotFoundError(
            f"formal HF local mirror is missing or incomplete: {mirror}"
        )

    expected_hashes = source.get("expected_lfs_sha256") or {}
    expected_sizes = source.get("expected_sizes") or {}
    if remote_path not in expected_hashes or remote_path not in expected_sizes:
        raise CalibrationDataError(
            "formal HF local mirror requires expected_lfs_sha256 and "
            f"expected_sizes for {remote_path}"
        )
    expected_size = int(expected_sizes[remote_path])
    actual_size = mirror.stat().st_size
    if actual_size != expected_size:
        raise ArtifactVerificationError(
            f"local mirror size mismatch for {mirror}: "
            f"expected {expected_size}, got {actual_size}"
        )
    expected_hash = str(expected_hashes[remote_path])
    actual_hash = sha256_file(mirror)
    if actual_hash != expected_hash:
        raise ArtifactVerificationError(
            f"local mirror SHA256 mismatch for {mirror}: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    if source.get("verify_remote_files_metadata", True):
        _validate_hf_source_metadata(source)

    locator = (
        f"hf://datasets/{source['dataset_name']}@{revision}/{remote_path}"
        f"#verified-local-mirror={mirror}"
    )
    for record in iter_parquet_records(
        [mirror], batch_size=int(source.get("batch_size", 256))
    ):
        yield SourceRecord(record.data, locator, record.record_index)


def _resolve_local_paths(
    path_patterns: Sequence[str | os.PathLike[str]], base_dir: Path
) -> list[Path]:
    paths: set[Path] = set()
    for pattern_value in path_patterns:
        pattern_path = Path(pattern_value)
        pattern = str(
            pattern_path if pattern_path.is_absolute() else base_dir / pattern_path
        )
        matches = glob.glob(pattern)
        if not matches and not glob.has_magic(pattern):
            raise FileNotFoundError(pattern)
        paths.update(Path(match).resolve() for match in matches)
    return sorted(paths)


def _verify_local_source_hashes(
    paths: Sequence[Path], expected_sha256: Mapping[str, str]
) -> None:
    if not expected_sha256:
        return
    by_name = {path.name: path for path in paths}
    by_path = {str(path): path for path in paths}
    for configured_path, expected_hash in expected_sha256.items():
        path = by_path.get(configured_path) or by_name.get(Path(configured_path).name)
        if path is None:
            raise ArtifactVerificationError(
                f"expected local source hash has no matching path: {configured_path}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ArtifactVerificationError(
                f"source SHA256 mismatch for {path}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


def iter_source_records(
    source: Mapping[str, Any], base_dir: str | os.PathLike[str] = "."
) -> Iterator[SourceRecord]:
    """Create an iterator for ``jsonl``, ``parquet``, or ``hf_streaming``."""

    source_type = str(source.get("type", ""))
    if source_type == "hf_streaming":
        if source.get("local_mirror_path"):
            yield from iter_pinned_hf_local_mirror_records(source, base_dir)
        else:
            yield from iter_hf_streaming_records(source)
        return
    if source_type not in {"jsonl", "parquet"}:
        raise CalibrationDataError(f"unsupported source type: {source_type!r}")
    path_values = source.get("paths") or []
    if not path_values:
        raise CalibrationDataError(f"{source_type} source requires non-empty paths")
    paths = _resolve_local_paths(path_values, Path(base_dir).resolve())
    _verify_local_source_hashes(paths, source.get("expected_sha256") or {})
    if source_type == "jsonl":
        yield from iter_jsonl_records(paths)
    else:
        yield from iter_parquet_records(
            paths, batch_size=int(source.get("batch_size", 256))
        )


def tokenizer_provenance_from_path(
    tokenizer_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Hash tokenizer files which can affect token IDs."""

    root = Path(tokenizer_path).resolve()
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "tokenizer.model",
    )
    files = {
        name: sha256_file(root / name) for name in names if (root / name).is_file()
    }
    if not files:
        raise FileNotFoundError(f"no tokenizer files found under {root}")
    aggregate = sha256_bytes(canonical_json_bytes(files))
    return {"path": str(root), "files_sha256": files, "aggregate_sha256": aggregate}


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _atomic_write_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalise_record(
    value: SourceRecord | Mapping[str, Any], fallback_index: int
) -> SourceRecord:
    if isinstance(value, SourceRecord):
        return value
    if not isinstance(value, Mapping):
        raise CalibrationDataError(f"record {fallback_index} is not a mapping")
    return SourceRecord(value, "python-iterable", fallback_index)


def _tokenize_document(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if len(token_ids) > 0 and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise CalibrationDataError("tokenizer returned an unexpected batch")
        token_ids = token_ids[0]
    result = [int(token) for token in token_ids]
    if any(token < 0 or token > np.iinfo(np.int32).max for token in result):
        raise CalibrationDataError("token ID is outside int32 output range")
    return result


def _candidate_from_record(
    record: SourceRecord,
    tokenizer: Any,
    *,
    text_field: str,
    identity_fields: Sequence[str],
    seed: str | int,
    sequence_length: int,
) -> WindowCandidate | None:
    text = record.data.get(text_field)
    if not isinstance(text, str) or not text:
        return None
    record_uid, raw_text_hash, identity = document_identity(
        record.data, text, identity_fields
    )
    token_ids = _tokenize_document(tokenizer, text)
    if len(token_ids) < sequence_length:
        return WindowCandidate(
            selection_hash="",
            record_uid=record_uid,
            raw_text_sha256=raw_text_hash,
            token_ids_sha256="",
            document_token_count=len(token_ids),
            token_offset=-1,
            token_ids=(),
            identity=identity,
            source_locator=record.locator,
            source_record_index=record.record_index,
        )
    offset = stable_window_offset(seed, record_uid, len(token_ids), sequence_length)
    window = tuple(token_ids[offset : offset + sequence_length])
    return WindowCandidate(
        selection_hash=stable_selection_hash(seed, record_uid),
        record_uid=record_uid,
        raw_text_sha256=raw_text_hash,
        token_ids_sha256=token_ids_sha256(window),
        document_token_count=len(token_ids),
        token_offset=offset,
        token_ids=window,
        identity=identity,
        source_locator=record.locator,
        source_record_index=record.record_index,
    )


def _record_identity_and_hash(
    record: SourceRecord,
    *,
    text_field: str,
    identity_fields: Sequence[str],
    seed: str | int,
) -> tuple[str, str, str, dict[str, Any], str] | None:
    """Compute cheap selection material before invoking the tokenizer."""

    text = record.data.get(text_field)
    if not isinstance(text, str) or not text:
        return None
    record_uid, raw_text_hash, identity = document_identity(
        record.data, text, identity_fields
    )
    return (
        text,
        record_uid,
        raw_text_hash,
        identity,
        stable_selection_hash(seed, record_uid),
    )


def _candidate_from_identity(
    record: SourceRecord,
    tokenizer: Any,
    identity_material: tuple[str, str, str, dict[str, Any], str],
    *,
    seed: str | int,
    sequence_length: int,
) -> WindowCandidate:
    """Tokenize one document already known to be able to affect bottom-k."""

    text, record_uid, raw_text_hash, identity, selection_hash = identity_material
    token_ids = _tokenize_document(tokenizer, text)
    if len(token_ids) < sequence_length:
        return WindowCandidate(
            selection_hash=selection_hash,
            record_uid=record_uid,
            raw_text_sha256=raw_text_hash,
            token_ids_sha256="",
            document_token_count=len(token_ids),
            token_offset=-1,
            token_ids=(),
            identity=identity,
            source_locator=record.locator,
            source_record_index=record.record_index,
        )
    offset = stable_window_offset(seed, record_uid, len(token_ids), sequence_length)
    window = tuple(token_ids[offset : offset + sequence_length])
    return WindowCandidate(
        selection_hash=selection_hash,
        record_uid=record_uid,
        raw_text_sha256=raw_text_hash,
        token_ids_sha256=token_ids_sha256(window),
        document_token_count=len(token_ids),
        token_offset=offset,
        token_ids=window,
        identity=identity,
        source_locator=record.locator,
        source_record_index=record.record_index,
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ArtifactVerificationError(f"expected JSON object in {path}")
    return value


def verify_calibration_output(
    output_dir: str | os.PathLike[str], expected_config_fingerprint: str | None = None
) -> dict[str, Any]:
    """Validate a completed calibration dataset and return its manifest."""

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactVerificationError(f"missing manifest: {manifest_path}")
    manifest = _load_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ArtifactVerificationError(f"output is not complete: {root}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactVerificationError("unsupported calibration manifest schema")
    if (
        expected_config_fingerprint is not None
        and manifest.get("config_fingerprint") != expected_config_fingerprint
    ):
        raise ResumeMismatchError(
            f"completed output {root} belongs to a different configuration"
        )

    artifacts = manifest.get("artifacts") or {}
    for relative_path, metadata in artifacts.items():
        artifact = (root / relative_path).resolve()
        try:
            artifact.relative_to(root.resolve())
        except ValueError as error:
            raise ArtifactVerificationError(
                f"artifact path escapes output directory: {relative_path}"
            ) from error
        if not artifact.is_file():
            raise ArtifactVerificationError(f"missing artifact: {artifact}")
        actual_hash = sha256_file(artifact)
        if actual_hash != metadata["sha256"]:
            raise ArtifactVerificationError(
                f"artifact SHA256 mismatch for {relative_path}: "
                f"expected {metadata['sha256']}, got {actual_hash}"
            )

    for split in ("train", "heldout"):
        metadata = manifest["splits"][split]
        array = np.load(root / metadata["file"], mmap_mode="r", allow_pickle=False)
        if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
            raise ArtifactVerificationError(
                f"{split} token array shape/dtype does not match manifest"
            )
    return manifest


def _run_spec(
    *,
    train_count: int,
    heldout_count: int,
    sequence_length: int,
    stride: int,
    seed: str | int,
    text_field: str,
    identity_fields: Sequence[str],
    source_provenance: Mapping[str, Any],
    tokenizer_provenance: Mapping[str, Any],
    local_conventions: Sequence[str],
    max_records: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_protocol": {
            "dataset": "FineWeb-Edu",
            "train_sequences": train_count,
            "sequence_length": sequence_length,
            "activation_token_stride": stride,
        },
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": str(seed),
            "within_document_only": True,
            "window_algorithm": WINDOW_ALGORITHM,
            "text_field": text_field,
            "identity_fields": list(identity_fields),
            "add_special_tokens": False,
            "heldout_count": heldout_count,
            "max_records": max_records,
        },
        "source": dict(source_provenance),
        "tokenizer": dict(tokenizer_provenance),
        "local_conventions": list(local_conventions),
    }


def _write_completed_dataset(
    staging: Path,
    reservoir: BottomKReservoir,
    *,
    run_spec: Mapping[str, Any],
    config_fingerprint: str,
    train_count: int,
    heldout_count: int,
    sequence_length: int,
    stats: Mapping[str, int],
) -> dict[str, Any]:
    candidates = reservoir.sorted_candidates()
    expected = train_count + heldout_count
    if len(candidates) != expected:
        raise IncompleteSourceError(
            f"needed {expected} eligible unique documents, found {len(candidates)}"
        )

    train = np.asarray(
        [candidate.token_ids for candidate in candidates[:train_count]],
        dtype=np.int32,
    ).reshape(train_count, sequence_length)
    heldout = np.asarray(
        [candidate.token_ids for candidate in candidates[train_count:]],
        dtype=np.int32,
    ).reshape(heldout_count, sequence_length)
    if train.shape != (train_count, sequence_length):
        raise ArtifactVerificationError(f"unexpected train shape: {train.shape}")
    if heldout.shape != (heldout_count, sequence_length):
        raise ArtifactVerificationError(f"unexpected heldout shape: {heldout.shape}")

    train_path = staging / "train_tokens.npy"
    heldout_path = staging / "heldout_tokens.npy"
    samples_path = staging / "samples.jsonl"
    _atomic_write_npy(train_path, train)
    _atomic_write_npy(heldout_path, heldout)

    sample_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates):
        split = "train" if rank < train_count else "heldout"
        split_index = rank if split == "train" else rank - train_count
        row = candidate.checkpoint_dict()
        del row["token_ids"]
        row.update(
            {
                "selection_rank": rank,
                "split": split,
                "split_index": split_index,
            }
        )
        sample_rows.append(row)
    samples_payload = b"".join(
        canonical_json_bytes(row) + b"\n" for row in sample_rows
    )
    _atomic_write_bytes(samples_path, samples_payload)

    stable_sample_material = [
        {
            key: row[key]
            for key in (
                "selection_hash",
                "record_uid",
                "raw_text_sha256",
                "token_ids_sha256",
                "document_token_count",
                "token_offset",
                "identity",
                "selection_rank",
                "split",
                "split_index",
            )
        }
        for row in sample_rows
    ]
    dataset_fingerprint = sha256_bytes(
        canonical_json_bytes(
            {
                "config_fingerprint": config_fingerprint,
                "samples": stable_sample_material,
            }
        )
    )
    artifacts = {
        "train_tokens.npy": {
            "sha256": sha256_file(train_path),
            "bytes": train_path.stat().st_size,
        },
        "heldout_tokens.npy": {
            "sha256": sha256_file(heldout_path),
            "bytes": heldout_path.stat().st_size,
        },
        "samples.jsonl": {
            "sha256": sha256_file(samples_path),
            "bytes": samples_path.stat().st_size,
            "rows": expected,
        },
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "run_spec": dict(run_spec),
        "stats": dict(stats),
        "splits": {
            "train": {
                "file": "train_tokens.npy",
                "shape": list(train.shape),
                "dtype": str(train.dtype),
                "token_ids_payload_sha256": token_ids_sha256(train.reshape(-1)),
            },
            "heldout": {
                "file": "heldout_tokens.npy",
                "shape": list(heldout.shape),
                "dtype": str(heldout.dtype),
                "token_ids_payload_sha256": token_ids_sha256(heldout.reshape(-1)),
            },
        },
        "sample_manifest": {
            "file": "samples.jsonl",
            "rows": expected,
            "hash_fields": {
                "selection_hash": SELECTION_ALGORITHM,
                "token_ids_sha256": TOKEN_HASH_ALGORITHM,
                "raw_text_sha256": "sha256-utf8-v1",
            },
        },
        "artifacts": artifacts,
    }
    _atomic_write_json(staging / "manifest.json", manifest)
    return manifest


def prepare_calibration_dataset(
    records: Iterable[SourceRecord | Mapping[str, Any]],
    tokenizer: Any,
    output_dir: str | os.PathLike[str],
    *,
    train_count: int = 500,
    heldout_count: int = 50,
    sequence_length: int = 1024,
    stride: int = 4,
    seed: str | int = "20260813",
    text_field: str = "text",
    identity_fields: Sequence[str] = ("id", "url"),
    add_special_tokens: bool = False,
    source_provenance: Mapping[str, Any] | None = None,
    tokenizer_provenance: Mapping[str, Any] | None = None,
    local_conventions: Sequence[str] = DEFAULT_LOCAL_CONVENTIONS,
    checkpoint_every: int = 10_000,
    max_records: int | None = None,
    progress: Callable[[Mapping[str, int]], None] | None = None,
) -> dict[str, Any]:
    """Prepare, validate, and atomically publish calibration token arrays.

    An interrupted scan leaves ``.<output-name>.inprogress/checkpoint.json``.
    Invoking this function again with the same run specification skips the
    already-scanned source records and resumes the bottom-k reservoir.  The
    final output directory appears only after all artifacts and their manifest
    have been fsynced.  An already completed and valid output is reused.
    """

    if add_special_tokens:
        raise ValueError(
            "this reproduction requires add_special_tokens=False; it is a "
            "documented local convention"
        )
    if train_count <= 0 or heldout_count < 0:
        raise ValueError("train_count must be positive and heldout_count nonnegative")
    if sequence_length <= 0 or stride <= 0:
        raise ValueError("sequence_length and stride must be positive")
    if sequence_length % stride:
        raise ValueError("sequence_length must be divisible by stride")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if max_records is not None and max_records <= 0:
        raise ValueError("max_records must be positive when supplied")

    source_provenance = source_provenance or {"type": "python-iterable"}
    tokenizer_provenance = tokenizer_provenance or {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "unknown"))
    }
    run_spec = _run_spec(
        train_count=train_count,
        heldout_count=heldout_count,
        sequence_length=sequence_length,
        stride=stride,
        seed=seed,
        text_field=text_field,
        identity_fields=identity_fields,
        source_provenance=source_provenance,
        tokenizer_provenance=tokenizer_provenance,
        local_conventions=local_conventions,
        max_records=max_records,
    )
    config_fingerprint = sha256_bytes(canonical_json_bytes(run_spec))

    output = Path(output_dir).resolve()
    staging = output.parent / f".{output.name}.inprogress"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_dir():
            raise ArtifactVerificationError(f"output exists and is not a directory: {output}")
        return verify_calibration_output(output, config_fingerprint)

    if (staging / "manifest.json").is_file():
        verify_calibration_output(staging, config_fingerprint)
        (staging / "checkpoint.json").unlink(missing_ok=True)
        _fsync_directory(staging)
        os.replace(staging, output)
        _fsync_directory(output.parent)
        return verify_calibration_output(output, config_fingerprint)

    staging.mkdir(parents=True, exist_ok=True)
    checkpoint_path = staging / "checkpoint.json"
    total_count = train_count + heldout_count
    if checkpoint_path.is_file():
        checkpoint = _load_json(checkpoint_path)
        if checkpoint.get("config_fingerprint") != config_fingerprint:
            raise ResumeMismatchError(
                f"in-progress output {staging} belongs to a different configuration"
            )
        records_seen = int(checkpoint["records_seen"])
        stats = {key: int(value) for key, value in checkpoint["stats"].items()}
        reservoir = BottomKReservoir.from_checkpoint(
            total_count, checkpoint["candidates"]
        )
    else:
        records_seen = 0
        stats = {
            "records_seen": 0,
            "missing_or_empty_text": 0,
            "too_short": 0,
            # Once bottom-k is full, documents whose hash cannot improve the
            # set are provably irrelevant and are not tokenized.  Consequently
            # this is the number of eligible *tokenized* documents, not a
            # claim about the eligibility rate of the complete shard.
            "eligible": 0,
            "tokenized_for_eligibility": 0,
            "selection_pruned_without_tokenization": 0,
            "reservoir_updates": 0,
        }
        reservoir = BottomKReservoir(total_count)
        _atomic_write_json(
            staging / "run_spec.json",
            {"config_fingerprint": config_fingerprint, "run_spec": run_spec},
        )

    def save_checkpoint() -> None:
        _atomic_write_json(
            checkpoint_path,
            {
                "schema_version": SCHEMA_VERSION,
                "config_fingerprint": config_fingerprint,
                "records_seen": records_seen,
                "stats": stats,
                "candidates": reservoir.checkpoint_values(),
            },
        )

    for stream_index, raw_record in enumerate(records):
        if stream_index < records_seen:
            continue
        if max_records is not None and records_seen >= max_records:
            break
        record = _normalise_record(raw_record, stream_index)
        identity_material = _record_identity_and_hash(
            record,
            text_field=text_field,
            identity_fields=identity_fields,
            seed=seed,
        )
        if identity_material is None:
            stats["missing_or_empty_text"] += 1
        else:
            _, record_uid, _, _, selection_hash = identity_material
            if not reservoir.could_accept(selection_hash, record_uid):
                stats["selection_pruned_without_tokenization"] = (
                    stats.get("selection_pruned_without_tokenization", 0) + 1
                )
            else:
                stats["tokenized_for_eligibility"] = (
                    stats.get("tokenized_for_eligibility", 0) + 1
                )
                candidate = _candidate_from_identity(
                    record,
                    tokenizer,
                    identity_material,
                    seed=seed,
                    sequence_length=sequence_length,
                )
                if candidate.token_offset < 0:
                    stats["too_short"] += 1
                else:
                    stats["eligible"] += 1
                    if reservoir.add(candidate):
                        stats["reservoir_updates"] += 1
        records_seen += 1
        stats["records_seen"] = records_seen
        if records_seen % checkpoint_every == 0:
            save_checkpoint()
            if progress is not None:
                progress(dict(stats) | {"selected": len(reservoir)})

    save_checkpoint()
    if len(reservoir) < total_count:
        raise IncompleteSourceError(
            f"source ended after {records_seen} records with only "
            f"{len(reservoir)}/{total_count} eligible unique documents; "
            f"resume checkpoint kept at {checkpoint_path}"
        )

    manifest = _write_completed_dataset(
        staging,
        reservoir,
        run_spec=run_spec,
        config_fingerprint=config_fingerprint,
        train_count=train_count,
        heldout_count=heldout_count,
        sequence_length=sequence_length,
        stats=stats,
    )
    verify_calibration_output(staging, config_fingerprint)
    checkpoint_path.unlink(missing_ok=True)
    _fsync_directory(staging)
    if output.exists():  # Race-safe: never overwrite another completed run.
        raise CalibrationDataError(f"output appeared while publishing: {output}")
    os.replace(staging, output)
    _fsync_directory(output.parent)
    return verify_calibration_output(output, config_fingerprint)


def discard_incomplete_output(output_dir: str | os.PathLike[str]) -> None:
    """Explicitly delete only this module's named in-progress directory.

    This is intentionally not called automatically.  A mismatched checkpoint
    may contain hours of work, so callers must opt in to removing it.
    """

    output = Path(output_dir).resolve()
    staging = output.parent / f".{output.name}.inprogress"
    if staging.is_dir():
        shutil.rmtree(staging)
