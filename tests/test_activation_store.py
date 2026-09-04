from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from kvcache_transfer.activation_store import (
    ActivationIntegrityError,
    ActivationStoreError,
    AtomicBF16Writer,
    open_bf16_activations,
    read_activation_metadata,
    sidecar_path,
    validate_bf16_activations,
    write_bf16_activations,
)


def _bf16_values(shape: tuple[int, ...]) -> torch.Tensor:
    total = int(np.prod(shape))
    return torch.linspace(-17.0, 19.0, total, dtype=torch.float32).reshape(shape).bfloat16()


def test_bf16_round_trip_is_bitwise_and_metadata_is_auditable(tmp_path: Path) -> None:
    path = tmp_path / "layer_000.bf16.bin"
    values = _bf16_values((7, 2, 4))
    provenance = {
        "model_sha256": "a" * 64,
        "token_manifest_sha256": "b" * 64,
        "rope_implementation": "qwen3_content_v1",
        "layer": 0,
        "cache_type": "k_content",
    }
    metadata = write_bf16_activations(path, values, provenance=provenance)

    assert sidecar_path(path) == tmp_path / "layer_000.json"
    assert metadata.shape == tuple(values.shape)
    assert metadata.nbytes == values.numel() * 2
    assert metadata.provenance == provenance
    assert metadata.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    with open_bf16_activations(path) as mapped:
        assert isinstance(mapped.bits, np.memmap)
        assert mapped.bits.flags.writeable is False
        restored = mapped.to_torch()
        assert torch.equal(restored.view(torch.uint16), values.view(torch.uint16))
        assert torch.equal(mapped.to_torch(dtype=torch.float32), values.float())
    assert validate_bf16_activations(path) == metadata


def test_chunked_writer_preserves_shape_order_and_iteration(tmp_path: Path) -> None:
    path = tmp_path / "layer_003.bf16.bin"
    values = _bf16_values((11, 3, 2))
    with AtomicBF16Writer(path, values.shape) as writer:
        writer.append(values[:2])
        writer.append(values[2:9])
        writer.append(values[9:])

    with open_bf16_activations(path) as mapped:
        chunks = list(mapped.iter_torch_chunks(4, dtype=torch.bfloat16))
    restored = torch.cat(chunks)
    assert [chunk.shape[0] for chunk in chunks] == [4, 4, 3]
    assert torch.equal(restored.view(torch.uint16), values.view(torch.uint16))


def test_writer_accepts_only_bf16_or_explicit_raw_uint16_bits(tmp_path: Path) -> None:
    path = tmp_path / "layer_001.bf16.bin"
    with pytest.raises(ActivationStoreError, match="must be bfloat16"):
        with AtomicBF16Writer(path, (2, 3)) as writer:
            writer.append(torch.ones(2, 3, dtype=torch.float32))
    assert not path.exists()
    assert not sidecar_path(path).exists()

    values = _bf16_values((2, 3))
    raw_bits = values.view(torch.uint16).numpy()
    with AtomicBF16Writer(path, values.shape) as writer:
        writer.append(raw_bits)
    with open_bf16_activations(path) as mapped:
        assert torch.equal(mapped.to_torch().view(torch.uint16), values.view(torch.uint16))


def test_incomplete_or_failed_write_is_never_committed(tmp_path: Path) -> None:
    path = tmp_path / "layer_002.bf16.bin"
    values = _bf16_values((8, 2, 2))
    with pytest.raises(ActivationStoreError, match="incomplete"):
        with AtomicBF16Writer(path, values.shape) as writer:
            writer.append(values[:3])

    assert not path.exists()
    assert not sidecar_path(path).exists()
    assert list(tmp_path.glob("*.tmp")) == []
    assert list(tmp_path.glob(".*.tmp")) == []


def test_corruption_and_truncation_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "layer_004.bf16.bin"
    values = _bf16_values((5, 2, 3))
    write_bf16_activations(path, values)

    with path.open("r+b") as handle:
        handle.seek(3)
        original = handle.read(1)
        handle.seek(3)
        handle.write(bytes([original[0] ^ 0xFF]))
    with pytest.raises(ActivationIntegrityError, match="SHA256 mismatch"):
        open_bf16_activations(path)

    # Restoring via explicit overwrite must create a new matching payload and
    # sidecar rather than trusting the previous checksum.
    write_bf16_activations(path, values, overwrite=True)
    with path.open("r+b") as handle:
        handle.truncate(path.stat().st_size - 2)
    with pytest.raises(ActivationIntegrityError, match="byte count mismatch"):
        open_bf16_activations(path)


def test_tampered_metadata_is_rejected_before_mmap(tmp_path: Path) -> None:
    path = tmp_path / "layer_005.bf16.bin"
    write_bf16_activations(path, _bf16_values((3, 2, 2)))
    metadata_path = sidecar_path(path)
    raw = json.loads(metadata_path.read_text())
    raw["shape"] = [999, 2, 2]
    metadata_path.write_text(json.dumps(raw))

    with pytest.raises(ActivationIntegrityError, match="numel does not match shape"):
        read_activation_metadata(path)


def test_existing_outputs_require_explicit_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "layer_006.bf16.bin"
    original = _bf16_values((2, 2, 2))
    replacement = (original.float() + 4).bfloat16()
    write_bf16_activations(path, original)
    with pytest.raises(FileExistsError):
        write_bf16_activations(path, replacement)

    write_bf16_activations(path, replacement, overwrite=True)
    with open_bf16_activations(path) as mapped:
        assert torch.equal(mapped.to_torch().view(torch.uint16), replacement.view(torch.uint16))


def test_chunks_must_preserve_non_sample_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "layer_007.bf16.bin"
    with pytest.raises(ActivationStoreError, match="non-sample dimensions"):
        with AtomicBF16Writer(path, (4, 2, 3)) as writer:
            writer.append(torch.zeros(4, 6, dtype=torch.bfloat16))

