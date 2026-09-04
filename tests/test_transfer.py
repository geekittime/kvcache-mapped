from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvcache_transfer.artifacts import save_layer_mapper
from kvcache_transfer.config import ArchitectureSpec
from kvcache_transfer.io import atomic_write_json, sha256_file
from kvcache_transfer.mapper import LayerMapper
from kvcache_transfer.transfer import (
    MapperBundle,
    TransferValidationError,
    source_cache_to_content,
    split_context_for_transfer,
    target_own_cache_boundary_oracle,
    transfer_source_cache,
)


def tiny_qwen3(*, seed: int = 17) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=101,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=None,
        _attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).eval()


def tiny_architecture() -> ArchitectureSpec:
    return ArchitectureSpec(
        source_layers=2,
        target_layers=2,
        source_kv_heads=2,
        target_kv_heads=2,
        source_head_dim=16,
        target_head_dim=16,
        selected_source_layers=1,
    )


def write_tiny_mapper_bundle(
    directory: Path,
    *,
    architecture: ArchitectureSpec,
    selected_layers: tuple[tuple[int, ...], ...] = ((0,), (1,)),
) -> MapperBundle:
    layers_directory = directory / "layers"
    fit_fingerprint = "a" * 64
    entries = []
    identity = torch.eye(architecture.source_features, architecture.target_features)
    for target_layer in range(architecture.target_layers):
        mapper = LayerMapper(
            source_layers=selected_layers[target_layer],
            key_weight=identity.clone(),
            key_bias=torch.zeros(architecture.target_features),
            value_weight=identity.clone(),
            value_bias=torch.zeros(architecture.target_features),
        )
        path = save_layer_mapper(
            layers_directory,
            target_layer,
            mapper,
            metadata={"fit_fingerprint": fit_fingerprint},
            source_heads=architecture.source_kv_heads,
            head_dim=architecture.source_head_dim,
        )
        sidecar = json.loads(path.with_suffix(".json").read_text())
        entries.append(
            {
                "target_layer": target_layer,
                "source_layers_rank_order": list(selected_layers[target_layer]),
                "path": str(path.resolve()),
                "sha256": sidecar["sha256"],
                "sidecar_path": str(path.with_suffix(".json").resolve()),
                "sidecar_sha256": sha256_file(path.with_suffix(".json")),
                "weight_numel": sidecar["weight_numel"],
                "bias_numel": sidecar["bias_numel"],
                "diagnostics": {},
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "direction": "qwen3_14b_to_32b",
        "method": "closed-form-centered-ridge-kv-cache-transfer",
        "fit_fingerprint": fit_fingerprint,
        "fit_definition": {
            "method": "two-pass-centered-ridge",
            "ridge_lambda_added_directly_to_unnormalized_gram": 0.01,
            "standardize_features": False,
            "regularize_bias": False,
            "covariance_dtype": "float32",
            "mapper_dtype": "float32",
            "allow_tf32": False,
            "feature_order": "selected_layer_rank,kv_head,head_dim",
            "output_layout": "target_kv_head,target_head_dim",
        },
        "architecture": {
            "source_layers": architecture.source_layers,
            "target_layers": architecture.target_layers,
            "kv_heads": architecture.source_kv_heads,
            "head_dim": architecture.source_head_dim,
            "selected_source_layers": architecture.selected_source_layers,
            "n_samples": 64,
        },
        "ridge_lambda": 0.01,
        "solver": "cholesky",
        "allow_tf32": False,
        "weight_count": architecture.mapper_weight_count,
        "bias_count": architecture.mapper_bias_count,
        "dtype": "float32",
        "selection": {
            "selected_source_layers_rank_order": [list(row) for row in selected_layers]
        },
        "layers": entries,
    }
    atomic_write_json(directory / "manifest.json", manifest)
    atomic_write_json(
        directory / "manifest.sha256.json",
        {
            "path": str((directory / "manifest.json").resolve()),
            "sha256": sha256_file(directory / "manifest.json"),
        },
    )
    return MapperBundle.open(directory, expected_architecture=architecture)


def test_mapper_bundle_detects_post_validation_layer_tampering(tmp_path: Path) -> None:
    architecture = tiny_architecture()
    bundle = write_tiny_mapper_bundle(tmp_path / "mapper", architecture=architecture)
    payload = bundle.layers[0].path
    with payload.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(TransferValidationError, match="changed after bundle validation"):
        bundle.load_layer(0)


def test_mapper_bundle_can_move_despite_audited_absolute_paths(tmp_path: Path) -> None:
    architecture = tiny_architecture()
    original = tmp_path / "original"
    write_tiny_mapper_bundle(original, architecture=architecture)
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)
    bundle = MapperBundle.open(moved, expected_architecture=architecture)
    assert bundle.layers[0].path.parent == moved / "layers"
    assert bundle.load_layer(1).source_layers == (1,)


def test_split_context_is_explicit_c_minus_one_boundary() -> None:
    context = torch.tensor([[2, 3, 5, 7]])
    prefix, boundary, mask = split_context_for_transfer(context)
    torch.testing.assert_close(prefix, torch.tensor([[2, 3, 5]]))
    torch.testing.assert_close(boundary, torch.tensor([[7]]))
    assert mask is None
    with pytest.raises(TransferValidationError, match="at least two"):
        split_context_for_transfer(torch.tensor([[2]]))


def test_target_own_random_qwen3_boundary_oracle_matches_full_forward() -> None:
    model = tiny_qwen3()
    context = torch.tensor([[7, 5, 19, 2, 31, 11, 3]])
    with torch.inference_mode():
        full = model(input_ids=context, use_cache=False)
        oracle = target_own_cache_boundary_oracle(model, context)
    torch.testing.assert_close(
        oracle.logits[:, -1], full.logits[:, -1], atol=2e-5, rtol=2e-5
    )
    assert oracle.past_key_values.get_seq_length() == context.shape[1]


def test_identity_mapper_converts_real_qwen3_cache_and_replays_exactly(
    tmp_path: Path,
) -> None:
    model = tiny_qwen3()
    architecture = tiny_architecture()
    bundle = write_tiny_mapper_bundle(tmp_path / "mapper", architecture=architecture)
    context = torch.tensor([[7, 5, 19, 2, 31, 11, 3]])
    prefix = context[:, :-1]
    with torch.inference_mode():
        own_cache = model.model(input_ids=prefix, use_cache=True).past_key_values
    mapped = transfer_source_cache(
        own_cache,
        source_config=model.config,
        target_model=model,
        mapper_bundle=bundle,
        output_dtype=None,
        token_chunk=2,
    )
    assert mapped is not own_cache
    for target_layer, source_layer in enumerate((0, 1)):
        torch.testing.assert_close(
            mapped.key_cache[target_layer],
            own_cache.key_cache[source_layer],
            atol=3e-6,
            rtol=3e-6,
        )
        torch.testing.assert_close(
            mapped.value_cache[target_layer], own_cache.value_cache[source_layer]
        )
    with torch.inference_mode():
        full = model(input_ids=context, use_cache=False)
        replay = model(
            input_ids=context[:, -1:],
            past_key_values=mapped,
            attention_mask=torch.ones_like(context),
            position_ids=torch.tensor([[context.shape[1] - 1]]),
            cache_position=torch.tensor([context.shape[1] - 1]),
            use_cache=True,
        )
    torch.testing.assert_close(
        replay.logits[:, -1], full.logits[:, -1], atol=3e-5, rtol=3e-5
    )


def test_inverse_rope_accepts_explicit_nonzero_positions() -> None:
    model = tiny_qwen3()
    input_ids = torch.tensor([[2, 3, 5]])
    position_ids = torch.tensor([[9, 10, 11]])
    with torch.inference_mode():
        cache = model.model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
        ).past_key_values
    content, values, returned_positions = source_cache_to_content(
        cache,
        model.config,
        position_ids=position_ids,
        expected_layers=2,
        expected_heads=2,
        expected_head_dim=16,
    )
    assert len(content) == len(values) == 2
    torch.testing.assert_close(returned_positions, position_ids)
    assert all(tensor.dtype == torch.float32 for tensor in content)
