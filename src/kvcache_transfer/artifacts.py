"""Mapper serialization with explicit layer ordering and checksums."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .io import atomic_write_json, sha256_file
from .mapper import LayerMapper


def save_layer_mapper(
    directory: str | Path,
    target_layer: int,
    mapper: LayerMapper,
    metadata: dict[str, Any] | None = None,
    *,
    source_heads: int = 8,
    head_dim: int = 128,
) -> Path:
    """Atomically publish one target-layer mapper and its commit sidecar.

    ``source_heads`` and ``head_dim`` default to the paper's Qwen3 layout.
    Keeping them explicit in the sidecar lets small synthetic integration
    tests exercise the exact serialization path without allocating a full
    8192x1024 production matrix.  Production callers still validate against
    8 KV heads and a 128-wide head.
    """

    mapper.validate(source_heads=source_heads, head_dim=head_dim)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"layer_{target_layer:03d}.safetensors"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=output
    )
    os.close(descriptor)
    try:
        save_file(
            {
                "key_weight": mapper.key_weight.detach().cpu().contiguous(),
                "key_bias": mapper.key_bias.detach().cpu().contiguous(),
                "value_weight": mapper.value_weight.detach().cpu().contiguous(),
                "value_bias": mapper.value_bias.detach().cpu().contiguous(),
            },
            temporary_name,
            metadata={
                "target_layer": str(target_layer),
                "source_layers_rank_order": ",".join(map(str, mapper.source_layers)),
                "source_heads": str(source_heads),
                "head_dim": str(head_dim),
            },
        )
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    sidecar = {
        "target_layer": target_layer,
        "source_layers_rank_order": list(mapper.source_layers),
        "sha256": sha256_file(destination),
        "dtype": str(mapper.key_weight.dtype).removeprefix("torch."),
        "source_heads": int(source_heads),
        "head_dim": int(head_dim),
        "tensor_shapes": {
            "key_weight": list(mapper.key_weight.shape),
            "key_bias": list(mapper.key_bias.shape),
            "value_weight": list(mapper.value_weight.shape),
            "value_bias": list(mapper.value_bias.shape),
        },
        "weight_numel": int(mapper.key_weight.numel() + mapper.value_weight.numel()),
        "bias_numel": int(mapper.key_bias.numel() + mapper.value_bias.numel()),
        "metadata": metadata or {},
    }
    atomic_write_json(destination.with_suffix(".json"), sidecar)
    return destination


def load_layer_mapper(path: str | Path, device: str | torch.device = "cpu") -> LayerMapper:
    path = Path(path)
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256_file(path) != sidecar["sha256"]:
        raise ValueError(f"Mapper checksum mismatch: {path}")
    tensors = load_file(path, device=str(device))
    mapper = LayerMapper(
        source_layers=tuple(sidecar["source_layers_rank_order"]),
        key_weight=tensors["key_weight"],
        key_bias=tensors["key_bias"],
        value_weight=tensors["value_weight"],
        value_bias=tensors["value_bias"],
    )
    mapper.validate(
        source_heads=int(sidecar.get("source_heads", 8)),
        head_dim=int(sidecar.get("head_dim", 128)),
    )
    return mapper
