from __future__ import annotations

import json
import time
from pathlib import Path

import torch

import scripts.fit_mapper as fit_mapper_module

from kvcache_transfer.activation_store import write_bf16_activations
from kvcache_transfer.artifacts import load_layer_mapper
from kvcache_transfer.io import atomic_write_json, sha256_file
from scripts.fit_mapper import (
    FitArchitecture,
    WorkerJob,
    _worker_entry,
    assign_target_layers,
    collect_activation_inventory,
    fit_one_target_layer,
    load_selected_layers,
    make_fit_fingerprint,
    publish_complete_manifest,
    validate_selection_provenance,
)


def test_worker_events_take_fresh_timestamps(
    tmp_path: Path, monkeypatch
) -> None:
    """Start/complete events must measure the work, not reuse one timestamp."""

    monkeypatch.setattr(
        fit_mapper_module, "validate_completed_layer", lambda *args, **kwargs: None
    )

    def fake_fit_one_target_layer(**kwargs):
        time.sleep(0.01)
        return {"sha256": "a" * 64}

    monkeypatch.setattr(
        fit_mapper_module, "fit_one_target_layer", fake_fit_one_target_layer
    )
    job = WorkerJob(
        rank=0,
        device="cpu",
        target_layers=(0,),
        selected_layers=((0,),),
        source_activation_root=str(tmp_path / "source"),
        target_activation_root=str(tmp_path / "target"),
        output_directory=str(tmp_path / "mapper"),
        run_directory=str(tmp_path / "run"),
        fit_fingerprint="f" * 64,
        selection_sha256="s" * 64,
        architecture=FitArchitecture(1, 1, 1, 2, 1, 8),
        ridge_lambda=0.01,
        chunk_tokens=4,
        solver="cholesky",
        analysis_device="cpu",
        resume=True,
        cpu_threads=1,
    )
    _worker_entry(job)
    events = [
        json.loads(line)
        for line in (tmp_path / "run/workers/worker_00.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["event"] for event in events] == ["fit-start", "fit-complete"]
    assert events[0]["utc"] < events[1]["utc"]


def _activation_path(root: Path, cache_type: str, layer: int) -> Path:
    return root / cache_type / f"layer_{layer:03d}.bf16.bin"


def _write_small_activation(
    root: Path,
    cache_type: str,
    layer: int,
    values: torch.Tensor,
    *,
    sequences: int = 1,
    model_name: str,
) -> None:
    write_bf16_activations(
        _activation_path(root, cache_type, layer),
        values.to(torch.bfloat16),
        provenance={
            "token_sha256": "0" * 64,
            "split": "train",
            "stride": 4,
            "stride_start": 0,
            "sequences": sequences,
            "cache_type": cache_type,
            "layer": layer,
            "model_name": model_name,
            "forward_dtype": "bfloat16",
            "content_key_inverse_dtype": "float32",
        },
    )


def test_assignment_is_target_layer_modulo_worker() -> None:
    assignments = assign_target_layers(range(10), ("cuda:0", "cuda:1", "cuda:2"))
    assert assignments == {
        "cuda:0": (0, 3, 6, 9),
        "cuda:1": (1, 4, 7),
        "cuda:2": (2, 5, 8),
    }


def test_selection_manifest_preserves_rank_order(tmp_path: Path) -> None:
    architecture = FitArchitecture(4, 2, 1, 2, 2, 8)
    path = tmp_path / "selected_layers.json"
    atomic_write_json(
        path,
        {
            "source_layers": 4,
            "target_layers": 2,
            "kv_heads": 1,
            "head_dim": 2,
            "k": 2,
            "n_samples": 8,
            "selected_source_layers_rank_order": [[3, 1], [0, 2]],
        },
    )
    assert load_selected_layers(path, architecture) == ((3, 1), (0, 2))


def test_small_mmap_fit_checkpoint_and_complete_manifest(tmp_path: Path) -> None:
    torch.manual_seed(7)
    architecture = FitArchitecture(
        source_layers=2,
        target_layers=1,
        kv_heads=1,
        head_dim=2,
        selected_source_layers=1,
        n_samples=64,
    )
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_values: dict[str, torch.Tensor] = {}
    for cache_type, offset in (("k_content", 0.0), ("v", 0.75)):
        source0 = torch.randn(64, 1, 2) + offset
        source1 = torch.randn(64, 1, 2) + offset
        source_values[cache_type] = source1
        _write_small_activation(
            source_root,
            cache_type,
            0,
            source0,
            sequences=0,
            model_name="qwen3_14b",
        )
        _write_small_activation(
            source_root,
            cache_type,
            1,
            source1,
            sequences=0,
            model_name="qwen3_14b",
        )
        matrix = (
            torch.tensor([[1.5, -0.25], [0.5, 2.0]])
            if cache_type == "k_content"
            else torch.tensor([[0.75, 0.4], [-0.5, 1.25]])
        )
        bias = torch.tensor([0.2, -0.3])
        target = (source1.flatten(1).float() @ matrix + bias).reshape(64, 1, 2)
        _write_small_activation(
            target_root,
            cache_type,
            0,
            target,
            sequences=0,
            model_name="qwen3_32b",
        )

    inventory = collect_activation_inventory(
        source_root,
        target_root,
        architecture,
        verify_payload_checksum=True,
    )
    # Recreate Phase 3's immutable activation catalog so Phase 4 proves that
    # the top-k rows came from these exact payloads, not merely same-shaped KV.
    catalog_entries = {"source": [], "target": []}
    for entry in inventory["entries"]:
        catalog_entries[entry["role"]].append(
            {
                "model_role": entry["role"],
                "cache_type": entry["cache_type"],
                "layer": entry["layer"],
                "path": entry["payload_path"],
                "shape": entry["shape"],
                "nbytes": entry["nbytes"],
                "sha256": entry["payload_sha256"],
                "provenance": entry["provenance"],
            }
        )
    catalog = {
        "schema_version": 1,
        "architecture": {
            "source_layers": 2,
            "target_layers": 1,
            "kv_heads": 1,
            "head_dim": 2,
            "top_k": 1,
        },
        **catalog_entries,
    }
    import hashlib

    catalog["input_fingerprint"] = hashlib.sha256(
        json.dumps(
            catalog,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    catalog_path = tmp_path / "activation_inputs.json"
    atomic_write_json(catalog_path, catalog)
    scores_path = tmp_path / "scores.npz"
    scores_path.write_bytes(b"synthetic-score-checkpoint")
    selection_path = tmp_path / "selected_layers.json"
    atomic_write_json(
        selection_path,
        {
            "source_layers": 2,
            "target_layers": 1,
            "kv_heads": 1,
            "head_dim": 2,
            "k": 1,
            "n_samples": 64,
            "direction": "qwen3_14b_to_32b",
            "formal_protocol": False,
            "input_fingerprint": catalog["input_fingerprint"],
            "token_sha256": ["0" * 64],
            "feature_order": (
                "selected-source-layer rank, then KV head 0..7, then head dimension 0..127"
            ),
            "selection_rule": {
                "channel_reduction": "uniform arithmetic mean of per-channel R2 values",
                "head_reduction": "uniform arithmetic mean over KV heads",
                "kv_reduction": "equal arithmetic mean of K and V scores",
                "benchmark_data_used": False,
            },
            "sufficient_statistics": {
                "activation_inputs": {
                    "path": str(catalog_path),
                    "sha256": sha256_file(catalog_path),
                }
            },
            "score_artifacts": {
                "npz": {"path": str(scores_path), "sha256": sha256_file(scores_path)}
            },
            "selected_source_layers_rank_order": [[1]],
        },
    )
    validated_provenance = validate_selection_provenance(
        selection_path,
        inventory,
        require_formal_protocol=False,
    )
    assert validated_provenance["selection_input_fingerprint"] == catalog[
        "input_fingerprint"
    ]
    selection_sha = sha256_file(selection_path)
    fingerprint, definition = make_fit_fingerprint(
        architecture=architecture,
        ridge_lambda=0.01,
        allow_tf32=False,
        solver="cholesky",
        chunk_tokens=13,
        selection_sha256=selection_sha,
        activation_inventory_digest=inventory["inventory_digest"],
    )
    output = tmp_path / "mapper"
    fitted = fit_one_target_layer(
        target_layer=0,
        selected_layers=(1,),
        source_activation_root=source_root,
        target_activation_root=target_root,
        output_directory=output,
        architecture=architecture,
        ridge_lambda=0.01,
        chunk_tokens=13,
        compute_device="cpu",
        solver="cholesky",
        analysis_device="cpu",
        fit_fingerprint=fingerprint,
        selection_sha256=selection_sha,
    )
    mapper = load_layer_mapper(fitted["payload"])
    assert mapper.source_layers == (1,)
    assert (output / "components/k_content/layer_000.safetensors").is_file()
    assert (output / "components/k_content/layer_000.json").is_file()
    assert (output / "components/v/layer_000.safetensors").is_file()
    assert (output / "components/v/layer_000.json").is_file()
    key_prediction = source_values["k_content"].flatten(1).float() @ mapper.key_weight
    key_prediction += mapper.key_bias
    target_key_path = _activation_path(target_root, "k_content", 0)
    sidecar = json.loads(target_key_path.with_name("layer_000.json").read_text())
    assert sidecar["shape"] == [64, 1, 2]
    layer_sidecar = json.loads(Path(fitted["payload"]).with_suffix(".json").read_text())
    diagnostics = layer_sidecar["metadata"]["diagnostics"]
    assert diagnostics["key"]["train_r2_uniform_channel_average_fp64"] > 0.999
    assert diagnostics["value"]["train_r2_uniform_channel_average_fp64"] > 0.999
    assert diagnostics["key"]["gram_normalized_by_n"] is False
    assert diagnostics["key"]["bias_regularized"] is False
    assert key_prediction.shape == (64, 2)

    # If the combined publication was interrupted after both independent
    # regressions committed, rebuilding the layer must reuse K and V rather
    # than recompute either Gram/solve.
    Path(fitted["payload"]).unlink()
    Path(fitted["payload"]).with_suffix(".json").unlink()
    rebuilt = fit_one_target_layer(
        target_layer=0,
        selected_layers=(1,),
        source_activation_root=source_root,
        target_activation_root=target_root,
        output_directory=output,
        architecture=architecture,
        ridge_lambda=0.01,
        chunk_tokens=13,
        compute_device="cpu",
        solver="cholesky",
        analysis_device="cpu",
        fit_fingerprint=fingerprint,
        selection_sha256=selection_sha,
    )
    rebuilt_sidecar = json.loads(Path(rebuilt["payload"]).with_suffix(".json").read_text())
    assert rebuilt_sidecar["metadata"]["component_checkpoints"]["key"]["resumed"]
    assert rebuilt_sidecar["metadata"]["component_checkpoints"]["value"]["resumed"]

    inventory_path = output / "inputs" / "activation_inventory.json"
    atomic_write_json(inventory_path, inventory)
    manifest = publish_complete_manifest(
        output,
        architecture=architecture,
        fit_fingerprint=fingerprint,
        fit_definition=definition,
        selection_path=selection_path,
        selection_sha256=selection_sha,
        activation_inventory_path=inventory_path,
        activation_inventory=inventory,
        selected_layers=((1,),),
        ridge_lambda=0.01,
        solver="cholesky",
    )
    assert manifest["status"] == "complete"
    assert manifest["weight_count"] == 8
    assert manifest["bias_count"] == 4
    assert (output / "manifest.sha256.json").is_file()
