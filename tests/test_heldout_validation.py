from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest
import torch

from kvcache_transfer.activation_store import sidecar_path, write_bf16_activations
from kvcache_transfer.artifacts import save_layer_mapper
from kvcache_transfer.heldout_validation import (
    MapperArchitecture,
    evaluate_heldout_mapper,
    validate_heldout_inputs,
)
from kvcache_transfer.io import atomic_write_json, sha256_file
from kvcache_transfer.mapper import LayerMapper


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _write_training_inventory(
    path: Path, architecture: MapperArchitecture, token_sha: str
) -> dict[str, object]:
    entries = []
    for role, layers in (
        ("source", architecture.source_layers),
        ("target", architecture.target_layers),
    ):
        for cache_type in ("k_content", "v"):
            for layer in range(layers):
                entries.append(
                    {
                        "role": role,
                        "cache_type": cache_type,
                        "layer": layer,
                        "shape": [
                            architecture.n_samples,
                            architecture.kv_heads,
                            architecture.head_dim,
                        ],
                        "payload_sha256": hashlib.sha256(
                            f"{role}-{cache_type}-{layer}".encode()
                        ).hexdigest(),
                        "provenance": {
                            "token_sha256": token_sha,
                            "split": "train",
                        },
                    }
                )
    inventory: dict[str, object] = {
        "schema_version": 1,
        "payload_checksums_verified": True,
        "expected_shape": [
            architecture.n_samples,
            architecture.kv_heads,
            architecture.head_dim,
        ],
        "shared_provenance": {
            "token_sha256": token_sha,
            "split": "train",
            "stride": "4",
            "stride_start": "0",
            "sequences": "1",
        },
        "entries": entries,
    }
    inventory["inventory_digest"] = _canonical(inventory)
    atomic_write_json(path, inventory)
    return inventory


def _write_capture(
    root: Path,
    *,
    role: str,
    values: dict[tuple[str, int], torch.Tensor],
    architecture: MapperArchitecture,
    token_sha: str,
    sequences: int = 2,
    sequence_length: int = 6,
    stride: int = 2,
) -> None:
    model_name = "qwen3_14b" if role == "source" else "qwen3_32b"
    entries = []
    for (cache_type, layer), tensor in sorted(values.items()):
        payload = root / cache_type / f"layer_{layer:03d}.bf16.bin"
        metadata = write_bf16_activations(
            payload,
            tensor.to(torch.bfloat16),
            provenance={
                "model_name": model_name,
                "model_role": role,
                "cache_type": cache_type,
                "layer": layer,
                "split": "heldout",
                "stride": stride,
                "stride_start": 0,
                "sequences": sequences,
                "forward_dtype": "bfloat16",
                "content_key_inverse_dtype": "float32",
                "token_sha256": token_sha,
                "calibration_manifest_sha256": "c" * 64,
                "dataset_fingerprint": "dataset-fingerprint",
            },
        )
        entries.append(
            {
                "cache_type": cache_type,
                "layer": layer,
                "path": str(payload.relative_to(root)),
                "shape": list(metadata.shape),
                "nbytes": metadata.nbytes,
                "sha256": metadata.sha256,
                "sidecar_sha256": sha256_file(sidecar_path(payload)),
            }
        )
    definition = {
        "model_name": model_name,
        "model": {
            "role": role,
            "architecture": {
                "num_hidden_layers": (
                    architecture.source_layers
                    if role == "source"
                    else architecture.target_layers
                ),
                "num_key_value_heads": architecture.kv_heads,
                "head_dim": architecture.head_dim,
            },
        },
        "calibration": {"split": "heldout", "token_sha256": token_sha},
        "sequences": sequences,
        "sequence_length": sequence_length,
        "sample_stride": stride,
        "sample_stride_start": 0,
        "sampled_positions_per_sequence": sequence_length // stride,
        "forward_dtype": "bfloat16",
        "inverse_rope_dtype": "float32",
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "run_definition": definition,
        "entries": entries,
        "inventory_sha256": _canonical(entries),
    }
    atomic_write_json(root / "manifest.json", manifest)


def _build_case(tmp_path: Path, *, same_token: bool = False):
    torch.manual_seed(11)
    architecture = MapperArchitecture(2, 2, 1, 2, 1, 256)
    train_token = "a" * 64
    heldout_token = train_token if same_token else "b" * 64
    mapper_root = tmp_path / "mapper"
    mapper_root.mkdir()
    selection_path = mapper_root / "inputs/selected_layers.json"
    selection = {"selected_source_layers_rank_order": [[0], [1]]}
    atomic_write_json(selection_path, selection)
    selection_sha = sha256_file(selection_path)
    inventory_path = mapper_root / "inputs/activation_inventory.json"
    inventory = _write_training_inventory(inventory_path, architecture, train_token)

    fit_definition = {
        "schema_version": 1,
        "method": "two-pass-centered-ridge",
        "architecture": asdict(architecture),
        "ridge_lambda_added_directly_to_unnormalized_gram": 0.01,
        "standardize_features": False,
        "regularize_bias": False,
        "covariance_dtype": "float32",
        "mapper_dtype": "float32",
        "allow_tf32": False,
        "solver": "cholesky",
        "feature_order": "selected_layer_rank,kv_head,head_dim",
        "output_layout": "target_kv_head,target_head_dim",
        "selection_sha256": selection_sha,
        "activation_inventory_digest": inventory["inventory_digest"],
    }
    fit_fingerprint = _canonical(fit_definition)
    matrices = [
        torch.tensor([[1.25, -0.5], [0.25, 0.75]], dtype=torch.float32),
        torch.tensor([[0.6, 0.4], [-0.2, 1.1]], dtype=torch.float32),
    ]
    biases = [
        torch.tensor([0.1, -0.2], dtype=torch.float32),
        torch.tensor([-0.15, 0.05], dtype=torch.float32),
    ]
    layer_entries = []
    for target_layer, source_layer in enumerate((0, 1)):
        mapper = LayerMapper(
            source_layers=(source_layer,),
            key_weight=matrices[target_layer],
            key_bias=biases[target_layer],
            value_weight=matrices[target_layer] * 0.8,
            value_bias=biases[target_layer] * 0.5,
        )
        payload = save_layer_mapper(
            mapper_root / "layers",
            target_layer,
            mapper,
            source_heads=1,
            head_dim=2,
            metadata={
                "fit_fingerprint": fit_fingerprint,
                "selection_manifest_sha256": selection_sha,
                "selected_source_layers_rank_order": [source_layer],
                "feature_order": "selected_layer_rank,kv_head,head_dim",
                "mapper_dtype": "float32",
                "covariance_dtype": "float32",
                "analysis_dtype": "float64",
                "allow_tf32": False,
                "standardize_features": False,
                "regularize_bias": False,
            },
        )
        sidecar = payload.with_suffix(".json")
        layer_entries.append(
            {
                "target_layer": target_layer,
                "source_layers_rank_order": [source_layer],
                "path": str(payload.resolve()),
                "sha256": sha256_file(payload),
                "sidecar_path": str(sidecar.resolve()),
                "sidecar_sha256": sha256_file(sidecar),
                "weight_numel": 8,
                "bias_numel": 4,
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "direction": "qwen3_14b_to_32b",
        "method": "closed-form-centered-ridge-kv-cache-transfer",
        "fit_fingerprint": fit_fingerprint,
        "fit_definition": fit_definition,
        "architecture": asdict(architecture),
        "ridge_lambda": 0.01,
        "solver": "cholesky",
        "allow_tf32": False,
        "dtype": "float32",
        "weight_count": architecture.expected_weight_count,
        "bias_count": architecture.expected_bias_count,
        "selection": {
            "path": str(selection_path.resolve()),
            "source_file_sha256_used_by_fit": selection_sha,
            "snapshot_sha256": selection_sha,
            "selected_source_layers_rank_order": [[0], [1]],
        },
        "activation_inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": sha256_file(inventory_path),
            "inventory_digest": inventory["inventory_digest"],
            "payload_checksums_verified": True,
        },
        "layers": layer_entries,
    }
    manifest_path = mapper_root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(
        mapper_root / "manifest.sha256.json",
        {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
    )

    samples = 6
    source_values: dict[tuple[str, int], torch.Tensor] = {}
    target_values: dict[tuple[str, int], torch.Tensor] = {}
    for cache_type in ("k_content", "v"):
        source_layers = [torch.randn(samples, 1, 2).to(torch.bfloat16) for _ in range(2)]
        for layer, values in enumerate(source_layers):
            source_values[(cache_type, layer)] = values
        for target_layer, source_layer in enumerate((0, 1)):
            weight = matrices[target_layer]
            bias = biases[target_layer]
            if cache_type == "v":
                weight = weight * 0.8
                bias = bias * 0.5
            target_values[(cache_type, target_layer)] = (
                source_layers[source_layer].flatten(1).float() @ weight + bias
            ).reshape(samples, 1, 2)
    source_root = tmp_path / "source-heldout"
    target_root = tmp_path / "target-heldout"
    _write_capture(
        source_root,
        role="source",
        values=source_values,
        architecture=architecture,
        token_sha=heldout_token,
    )
    _write_capture(
        target_root,
        role="target",
        values=target_values,
        architecture=architecture,
        token_sha=heldout_token,
    )
    return manifest_path, source_root, target_root


def test_strict_synthetic_heldout_reconstruction_is_chunked_and_exact(tmp_path: Path) -> None:
    manifest, source, target = _build_case(tmp_path)
    inputs = validate_heldout_inputs(
        manifest,
        source,
        target,
        formal_protocol=False,
        heldout_sequences=2,
        sequence_length=6,
        stride=2,
    )
    assert inputs.heldout_samples == 6
    assert inputs.training_token_sha256 != inputs.heldout_token_sha256
    result = evaluate_heldout_mapper(
        inputs,
        chunk_tokens=2,
        compute_device="cpu",
        analysis_device="cpu",
    )
    assert result["status"] == "complete"
    assert result["summary"]["target_layers"] == 2
    assert result["summary"]["key_uniform_channel_r2_fp64"][
        "mean_across_target_layers"
    ] > 0.9999
    assert result["summary"]["value_uniform_channel_r2_fp64"][
        "mean_across_target_layers"
    ] > 0.9999
    assert all(layer["source_layers_rank_order"] in ([0], [1]) for layer in result["layers"])


def test_rejects_training_tokens_reused_as_heldout(tmp_path: Path) -> None:
    manifest, source, target = _build_case(tmp_path, same_token=True)
    with pytest.raises(ValueError, match="same token SHA256"):
        validate_heldout_inputs(
            manifest,
            source,
            target,
            formal_protocol=False,
            heldout_sequences=2,
            sequence_length=6,
            stride=2,
        )


def test_rejects_mapper_payload_changed_after_manifest_commit(tmp_path: Path) -> None:
    manifest, source, target = _build_case(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    payload = Path(raw["layers"][0]["path"])
    with payload.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="payload SHA256 mismatch"):
        validate_heldout_inputs(
            manifest,
            source,
            target,
            formal_protocol=False,
            heldout_sequences=2,
            sequence_length=6,
            stride=2,
        )
