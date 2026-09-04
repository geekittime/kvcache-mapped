from __future__ import annotations

import torch
import pytest

from kvcache_transfer.mapper import LayerMapper, apply_layer_mapper
from reverse_kv.protocol import atomic_write_json, sha256_file
from reverse_kv.runtime import (
    REVERSE_ARCHITECTURE,
    ReverseMapperBundle,
    ReverseRuntimeValidationError,
)


def test_runtime_concatenates_twenty_layers_in_rank_order() -> None:
    selected = tuple(range(20))
    # Tiny heads/dim exercise the exact runtime without a production 6.25-GiB fixture.
    source_keys = [torch.full((1, 1, 3, 2), float(layer)) for layer in range(20)]
    source_values = [value + 0.5 for value in source_keys]
    weight = torch.zeros((40, 2), dtype=torch.float32)
    weight[0, 0] = 1.0
    weight[-1, 1] = 1.0
    mapper = LayerMapper(
        source_layers=selected,
        key_weight=weight,
        key_bias=torch.zeros(2),
        value_weight=weight.clone(),
        value_bias=torch.zeros(2),
    )
    cos = torch.ones((1, 3, 2), dtype=torch.float32)
    sin = torch.zeros_like(cos)
    key, value = apply_layer_mapper(
        source_keys,
        source_values,
        mapper,
        cos,
        sin,
        output_dtype=torch.float32,
        token_chunk=2,
        source_heads=1,
        head_dim=2,
    )
    assert key.shape == (1, 1, 3, 2)
    assert value.shape == (1, 1, 3, 2)
    assert torch.all(key[..., 0] == 0)
    assert torch.all(key[..., 1] == 19)
    assert torch.all(value[..., 0] == 0.5)
    assert torch.all(value[..., 1] == 19.5)


def test_reverse_runtime_architecture_is_frozen() -> None:
    assert REVERSE_ARCHITECTURE.source_layers == 64
    assert REVERSE_ARCHITECTURE.target_layers == 40
    assert REVERSE_ARCHITECTURE.selected_source_layers == 20
    assert REVERSE_ARCHITECTURE.source_features == 20_480
    assert REVERSE_ARCHITECTURE.mapper_weight_count == 1_677_721_600


def test_bundle_rejects_wrong_direction_before_loading_layers(tmp_path) -> None:
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "direction": "wrong_direction",
        "method": "closed-form-centered-ridge-kv-cache-transfer",
        "dtype": "float32",
        "allow_tf32": False,
        "weight_count": 1_677_721_600,
        "bias_count": 81_920,
    }
    manifest_path = tmp_path / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        tmp_path / "manifest.sha256.json",
        {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
    )
    with pytest.raises(ReverseRuntimeValidationError, match="direction"):
        ReverseMapperBundle.open(tmp_path)
