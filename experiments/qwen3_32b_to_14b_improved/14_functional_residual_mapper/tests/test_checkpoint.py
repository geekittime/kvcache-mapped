from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from src.checkpoint import load_checkpoint, save_checkpoint, validate_checkpoint
from src.common import ContractError


def test_checkpoint_is_atomic_safe_and_resumable(tmp_path: Path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    checkpoint = save_checkpoint(
        root,
        "step_0001",
        {"layer_07_key_encoder": torch.ones(3, 2), "layer_07_key_decoder": torch.zeros(1)},
        {"step": 1, "system_class": "functional_residual_r1"},
    )
    report = validate_checkpoint(checkpoint)
    assert report["tensor_count"] == 2
    tensors, metadata, envelope = load_checkpoint(checkpoint)
    assert sorted(tensors) == ["layer_07_key_decoder", "layer_07_key_encoder"]
    assert metadata["step"] == 1
    assert envelope["commit"]["status"] == "COMMITTED"
    assert not list(root.glob("*.staging.*"))
    with pytest.raises(ContractError, match="overwrite"):
        save_checkpoint(root, "step_0001", {"x": torch.ones(1)}, {})


def test_checkpoint_tamper_is_detected(tmp_path: Path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    checkpoint = save_checkpoint(root, "step_0002", {"x": torch.ones(2)}, {})
    metadata = checkpoint / "metadata.json"
    os.chmod(metadata, 0o644)
    with metadata.open("ab") as handle:
        handle.write(b" ")
    os.chmod(metadata, 0o444)
    with pytest.raises(ContractError, match="reference mismatch"):
        validate_checkpoint(checkpoint)


def test_checkpoint_rejects_nonfinite_and_bad_ids(tmp_path: Path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    with pytest.raises(ContractError):
        save_checkpoint(root, "../escape", {"x": torch.ones(1)}, {})
    with pytest.raises(ContractError, match="NaN"):
        save_checkpoint(root, "bad", {"x": torch.tensor([float("nan")])}, {})

