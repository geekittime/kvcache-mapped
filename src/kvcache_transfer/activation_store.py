"""Atomic, memory-mappable storage for BF16 KV-cache activations.

NumPy does not expose BF16 consistently across supported versions, so payload
files contain the raw BF16 bit patterns as little-endian unsigned 16-bit
integers.  A JSON sidecar records the logical shape, byte order, byte count,
SHA256 checksum, and provenance supplied by the extraction pipeline.

The sidecar is the commit marker.  Writers create uniquely named temporary
files in the destination directory, fsync and validate the payload, atomically
replace the final payload, then atomically replace the sidecar last.  Readers
never accept a payload without valid committed metadata.  This prevents an
interrupted extraction from masquerading as a usable activation checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


FORMAT_NAME = "kvcache-transfer-bf16-bits"
FORMAT_VERSION = 1
DTYPE_NAME = "bfloat16"
STORAGE_DTYPE = "uint16"
BYTE_ORDER = "little"


class ActivationStoreError(RuntimeError):
    """Base class for activation payload or metadata failures."""


class ActivationIntegrityError(ActivationStoreError):
    """Raised when stored byte counts, checksums, or metadata are invalid."""


def _payload_path(path: str | os.PathLike[str]) -> Path:
    result = Path(path)
    if result.name.endswith(".json"):
        raise ActivationStoreError("payload path must not point to the JSON sidecar")
    return result


def sidecar_path(path: str | os.PathLike[str]) -> Path:
    """Return the JSON sidecar path for a BF16 payload.

    ``layer_000.bf16.bin`` maps to ``layer_000.json`` as specified in the
    reproduction plan.  Other suffixes map to ``<filename>.json``.
    """

    payload = _payload_path(path)
    if payload.name.endswith(".bf16.bin"):
        return payload.with_name(payload.name[: -len(".bf16.bin")] + ".json")
    return payload.with_name(payload.name + ".json")


def _validate_shape(shape: Sequence[int]) -> tuple[int, ...]:
    try:
        result = tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError) as exc:
        raise ActivationStoreError(f"invalid activation shape: {shape!r}") from exc
    if not result or any(dimension <= 0 for dimension in result):
        raise ActivationStoreError(f"activation shape must be non-empty and positive: {result}")
    return result


def _expected_nbytes(shape: Sequence[int]) -> int:
    return math.prod(shape) * 2


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(chunk_bytes):
            digest.update(data)
    return digest.hexdigest()


def _fsync_directory(directory: Path) -> None:
    """Best-effort directory fsync so atomic renames survive a host crash."""

    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary_path(destination: Path) -> Path:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    return Path(temporary)


@dataclass(frozen=True)
class ActivationMetadata:
    """Validated JSON metadata for one raw BF16 activation payload."""

    shape: tuple[int, ...]
    nbytes: int
    sha256: str
    provenance: Mapping[str, Any]
    created_at_utc: str
    format: str = FORMAT_NAME
    format_version: int = FORMAT_VERSION
    dtype: str = DTYPE_NAME
    storage_dtype: str = STORAGE_DTYPE
    byte_order: str = BYTE_ORDER

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-serializable sidecar representation."""

        return {
            "format": self.format,
            "format_version": self.format_version,
            "dtype": self.dtype,
            "storage_dtype": self.storage_dtype,
            "byte_order": self.byte_order,
            "shape": list(self.shape),
            "numel": self.numel,
            "nbytes": self.nbytes,
            "sha256": self.sha256,
            "created_at_utc": self.created_at_utc,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ActivationMetadata":
        """Validate and construct metadata loaded from JSON."""

        required = {
            "format",
            "format_version",
            "dtype",
            "storage_dtype",
            "byte_order",
            "shape",
            "numel",
            "nbytes",
            "sha256",
            "created_at_utc",
            "provenance",
        }
        missing = sorted(required - raw.keys())
        if missing:
            raise ActivationIntegrityError(f"activation metadata is missing keys: {missing}")
        shape = _validate_shape(raw["shape"])
        metadata = cls(
            format=str(raw["format"]),
            format_version=int(raw["format_version"]),
            dtype=str(raw["dtype"]),
            storage_dtype=str(raw["storage_dtype"]),
            byte_order=str(raw["byte_order"]),
            shape=shape,
            nbytes=int(raw["nbytes"]),
            sha256=str(raw["sha256"]),
            created_at_utc=str(raw["created_at_utc"]),
            provenance=raw["provenance"],
        )
        if metadata.format != FORMAT_NAME or metadata.format_version != FORMAT_VERSION:
            raise ActivationIntegrityError(
                f"unsupported activation format {metadata.format!r} v{metadata.format_version}"
            )
        if metadata.dtype != DTYPE_NAME or metadata.storage_dtype != STORAGE_DTYPE:
            raise ActivationIntegrityError(
                f"unsupported activation dtype/storage: {metadata.dtype}/{metadata.storage_dtype}"
            )
        if metadata.byte_order != BYTE_ORDER:
            raise ActivationIntegrityError(
                f"unsupported byte order {metadata.byte_order!r}; expected {BYTE_ORDER!r}"
            )
        if int(raw["numel"]) != metadata.numel:
            raise ActivationIntegrityError("metadata numel does not match shape")
        if metadata.nbytes != _expected_nbytes(shape):
            raise ActivationIntegrityError("metadata nbytes does not match BF16 shape")
        if len(metadata.sha256) != 64 or any(c not in "0123456789abcdef" for c in metadata.sha256):
            raise ActivationIntegrityError("metadata sha256 is not a lowercase SHA256 hex digest")
        if not isinstance(metadata.provenance, Mapping):
            raise ActivationIntegrityError("metadata provenance must be a JSON object")
        return metadata


@dataclass
class MappedBF16Activations:
    """Read-only mmap view of BF16 bit patterns plus validated metadata.

    The underlying NumPy array is a zero-copy, little-endian ``uint16`` mmap.
    PyTorch has no direct zero-copy reinterpretation from a NumPy uint16 array
    into BF16.  :meth:`to_torch` therefore copies only the requested slice and
    views those identical bits as BF16.
    """

    payload_path: Path
    metadata_path: Path
    metadata: ActivationMetadata
    bits: np.memmap

    def close(self) -> None:
        """Close the mmap; subsequent access to ``bits`` is invalid."""

        mmap_object = getattr(self.bits, "_mmap", None)
        if mmap_object is not None:
            mmap_object.close()

    def __enter__(self) -> "MappedBF16Activations":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def to_torch(
        self,
        selection: object = slice(None),
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Materialize a selected mmap region as a PyTorch tensor.

        The default returns BF16 with bitwise identity to the file.  FP32 is
        useful for ridge accumulation.  Other floating dtypes are accepted;
        integral output dtypes are rejected to avoid confusing raw bits with
        numerical BF16 values.
        """

        if not dtype.is_floating_point:
            raise ActivationStoreError("to_torch dtype must be floating point")
        selected = np.asarray(self.bits[selection], dtype=np.dtype("<u2"))
        # np.copy is deliberate: mmap arrays are read-only and PyTorch warns
        # about exposing a non-writable buffer.  The copied int16 tensor owns
        # its bytes; view(torch.bfloat16) preserves their bit pattern exactly.
        signed_bits = selected.copy().view(np.int16)
        bf16 = torch.from_numpy(signed_bits).view(torch.bfloat16)
        if dtype != torch.bfloat16:
            bf16 = bf16.to(dtype=dtype)
        return bf16.to(device=device)

    def iter_torch_chunks(
        self,
        chunk_samples: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Iterator[torch.Tensor]:
        """Yield chunks along the leading sample dimension."""

        if chunk_samples <= 0:
            raise ActivationStoreError("chunk_samples must be positive")
        for start in range(0, self.metadata.shape[0], chunk_samples):
            stop = min(start + chunk_samples, self.metadata.shape[0])
            yield self.to_torch(slice(start, stop), device=device, dtype=dtype)


class AtomicBF16Writer:
    """Append BF16 chunks and commit a verified payload/sidecar atomically.

    Use as a context manager.  On normal exit :meth:`commit` is called; on an
    exception temporary files are removed.  Existing committed files are not
    replaced unless ``overwrite=True``.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        shape: Sequence[int],
        *,
        provenance: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = _payload_path(path)
        self.metadata_path = sidecar_path(self.path)
        self.shape = _validate_shape(shape)
        self.provenance = dict(provenance or {})
        self.overwrite = overwrite
        self.expected_numel = math.prod(self.shape)
        self.expected_nbytes = _expected_nbytes(self.shape)
        self._written_numel = 0
        self._digest = hashlib.sha256()
        self._handle: Any | None = None
        self._temporary_payload: Path | None = None
        self._committed = False

    def __enter__(self) -> "AtomicBF16Writer":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.overwrite and (self.path.exists() or self.metadata_path.exists()):
            raise FileExistsError(
                f"activation output already exists: {self.path} or {self.metadata_path}"
            )
        self._temporary_payload = _temporary_path(self.path)
        self._handle = self._temporary_payload.open("wb")
        return self

    def append(self, values: torch.Tensor | np.ndarray) -> None:
        """Append a contiguous tensor/array in row-major order.

        PyTorch inputs must be BF16.  NumPy inputs must be ``uint16`` and are
        interpreted as already encoded BF16 bits.  Accepting raw NumPy bits is
        useful for byte-for-byte copy and storage tests while preventing an
        accidental FP32-to-uint16 numerical cast.
        """

        if self._handle is None:
            raise ActivationStoreError("AtomicBF16Writer must be entered before append")
        if isinstance(values, torch.Tensor):
            if values.dtype != torch.bfloat16:
                raise ActivationStoreError(
                    f"torch activation chunks must be bfloat16, got {values.dtype}"
                )
            tensor = values.detach().to(device="cpu").contiguous()
            # PyTorch permits a BF16->uint16 view even though NumPy cannot
            # directly represent BF16.
            bits = tensor.view(torch.uint16).numpy()
        else:
            bits = np.asarray(values)
            if bits.dtype.kind != "u" or bits.dtype.itemsize != 2:
                raise ActivationStoreError(
                    "NumPy activation chunks must contain raw uint16 BF16 bits"
                )
            bits = np.ascontiguousarray(bits)

        if bits.ndim != len(self.shape) or tuple(bits.shape[1:]) != self.shape[1:]:
            raise ActivationStoreError(
                "activation chunks must preserve the payload's non-sample dimensions: "
                f"expected [chunk, {self.shape[1:]}], got {tuple(bits.shape)}"
            )

        chunk_numel = int(bits.size)
        if self._written_numel + chunk_numel > self.expected_numel:
            raise ActivationStoreError(
                f"activation writer overflow: expected {self.expected_numel} elements"
            )
        little_endian = np.ascontiguousarray(bits.astype(np.dtype("<u2"), copy=False))
        payload = memoryview(little_endian).cast("B")
        self._handle.write(payload)
        self._digest.update(payload)
        self._written_numel += chunk_numel

    def commit(self) -> ActivationMetadata:
        """Fsync, validate, and publish the payload followed by its sidecar."""

        if self._committed:
            raise ActivationStoreError("writer has already been committed")
        if self._handle is None or self._temporary_payload is None:
            raise ActivationStoreError("AtomicBF16Writer must be entered before commit")
        if self._written_numel != self.expected_numel:
            raise ActivationStoreError(
                f"incomplete activation payload: wrote {self._written_numel} of "
                f"{self.expected_numel} elements"
            )

        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        if self._temporary_payload.stat().st_size != self.expected_nbytes:
            raise ActivationIntegrityError("temporary payload byte count is incorrect")
        checksum = self._digest.hexdigest()
        # Re-read once before publication.  This detects implementation/storage
        # errors independently of the incremental digest.
        if _sha256_file(self._temporary_payload) != checksum:
            raise ActivationIntegrityError("temporary payload checksum verification failed")

        try:
            json.dumps(self.provenance, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ActivationStoreError("provenance must be strict JSON-serializable data") from exc
        metadata = ActivationMetadata(
            shape=self.shape,
            nbytes=self.expected_nbytes,
            sha256=checksum,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            provenance=self.provenance,
        )
        metadata_json = (
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        temporary_metadata = _temporary_path(self.metadata_path)
        try:
            with temporary_metadata.open("wb") as handle:
                handle.write(metadata_json)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(self._temporary_payload, self.path)
            self._temporary_payload = None
            os.replace(temporary_metadata, self.metadata_path)
            _fsync_directory(self.path.parent)
        finally:
            temporary_metadata.unlink(missing_ok=True)
        self._committed = True
        return metadata

    def abort(self) -> None:
        """Close and remove uncommitted temporary output."""

        if self._handle is not None:
            self._handle.close()
            self._handle = None
        if self._temporary_payload is not None:
            self._temporary_payload.unlink(missing_ok=True)
            self._temporary_payload = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            if self._committed:
                return
            try:
                self.commit()
            except BaseException:
                self.abort()
                raise
        else:
            self.abort()


def write_bf16_activations(
    path: str | os.PathLike[str],
    values: torch.Tensor,
    *,
    provenance: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> ActivationMetadata:
    """Atomically write one materialized BF16 tensor and return its metadata."""

    if not isinstance(values, torch.Tensor):
        raise ActivationStoreError("write_bf16_activations expects a torch.Tensor")
    with AtomicBF16Writer(
        path, values.shape, provenance=provenance, overwrite=overwrite
    ) as writer:
        writer.append(values)
    return read_activation_metadata(path)


def commit_existing_bf16_payload(
    path: str | os.PathLike[str],
    shape: Sequence[int],
    *,
    provenance: Mapping[str, Any] | None = None,
    overwrite_sidecar: bool = False,
) -> ActivationMetadata:
    """Publish metadata for a fully written raw BF16 payload.

    This is the commit primitive used by the resumable multi-layer capture
    assembler.  Its payloads are preallocated and filled under a separate
    progress checkpoint, so rewriting them through :class:`AtomicBF16Writer`
    would discard valid sequence-level work.  The caller must keep the payload
    in an unpublished staging directory until this function has committed all
    layer sidecars and a higher-level manifest.

    The payload is fsynced, byte-count checked and SHA256-verified before the
    sidecar (the per-file commit marker) is atomically installed last.
    """

    payload = _payload_path(path)
    metadata_path = sidecar_path(payload)
    validated_shape = _validate_shape(shape)
    expected_nbytes = _expected_nbytes(validated_shape)
    if not payload.is_file():
        raise ActivationIntegrityError(f"cannot commit missing activation payload: {payload}")
    if metadata_path.exists() and not overwrite_sidecar:
        raise FileExistsError(f"activation sidecar already exists: {metadata_path}")
    actual_nbytes = payload.stat().st_size
    if actual_nbytes != expected_nbytes:
        raise ActivationIntegrityError(
            f"payload byte count mismatch: expected={expected_nbytes}, file={actual_nbytes}"
        )
    with payload.open("rb") as handle:
        os.fsync(handle.fileno())
    checksum = _sha256_file(payload)
    provenance_value = dict(provenance or {})
    try:
        json.dumps(provenance_value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ActivationStoreError(
            "provenance must be strict JSON-serializable data"
        ) from exc
    metadata = ActivationMetadata(
        shape=validated_shape,
        nbytes=expected_nbytes,
        sha256=checksum,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        provenance=provenance_value,
    )
    encoded = (
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary_metadata = _temporary_path(metadata_path)
    try:
        with temporary_metadata.open("wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_metadata, metadata_path)
        _fsync_directory(payload.parent)
    finally:
        temporary_metadata.unlink(missing_ok=True)
    # Independently parse what was published before returning success.
    return read_activation_metadata(payload)


def read_activation_metadata(path: str | os.PathLike[str]) -> ActivationMetadata:
    """Load and strictly validate an activation sidecar without hashing data."""

    metadata_path = sidecar_path(path)
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ActivationIntegrityError(
            f"cannot read activation metadata {metadata_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise ActivationIntegrityError("activation metadata root must be a JSON object")
    return ActivationMetadata.from_dict(raw)


def open_bf16_activations(
    path: str | os.PathLike[str], *, verify_checksum: bool = True
) -> MappedBF16Activations:
    """Open a committed BF16 payload as a read-only zero-copy uint16 mmap.

    Payload size is always checked.  SHA256 verification is on by default for
    auditability; repeated performance-critical opens may turn it off only
    after an earlier verified open in the same run.
    """

    payload = _payload_path(path)
    metadata_path = sidecar_path(payload)
    metadata = read_activation_metadata(payload)
    try:
        actual_nbytes = payload.stat().st_size
    except OSError as exc:
        raise ActivationIntegrityError(f"cannot stat activation payload {payload}: {exc}") from exc
    if actual_nbytes != metadata.nbytes:
        raise ActivationIntegrityError(
            f"payload byte count mismatch: metadata={metadata.nbytes}, file={actual_nbytes}"
        )
    if verify_checksum:
        actual_checksum = _sha256_file(payload)
        if actual_checksum != metadata.sha256:
            raise ActivationIntegrityError(
                f"payload SHA256 mismatch: metadata={metadata.sha256}, file={actual_checksum}"
            )
    try:
        bits = np.memmap(
            payload, mode="r", dtype=np.dtype("<u2"), shape=metadata.shape, order="C"
        )
    except (OSError, ValueError) as exc:
        raise ActivationIntegrityError(f"cannot mmap activation payload {payload}: {exc}") from exc
    return MappedBF16Activations(
        payload_path=payload,
        metadata_path=metadata_path,
        metadata=metadata,
        bits=bits,
    )


def validate_bf16_activations(path: str | os.PathLike[str]) -> ActivationMetadata:
    """Fully validate metadata, byte count, and SHA256, then close the mmap."""

    with open_bf16_activations(path, verify_checksum=True) as mapped:
        return mapped.metadata


__all__ = [
    "ActivationIntegrityError",
    "ActivationMetadata",
    "ActivationStoreError",
    "AtomicBF16Writer",
    "BYTE_ORDER",
    "DTYPE_NAME",
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MappedBF16Activations",
    "STORAGE_DTYPE",
    "commit_existing_bf16_payload",
    "open_bf16_activations",
    "read_activation_metadata",
    "sidecar_path",
    "validate_bf16_activations",
    "write_bf16_activations",
]
