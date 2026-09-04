from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import reverse_kv.fit_adapter as fit_adapter
import reverse_kv.provenance as provenance
import reverse_kv.selection_adapter as selection_adapter
from reverse_kv.fit_adapter import (
    _collect_exact_reverse_inventory,
    _sanitized_fit_argv,
    _validate_layer_payload,
)
from reverse_kv.heldout import heldout_preflight
from reverse_kv.protocol import (
    DIRECTION,
    SOURCE,
    TARGET,
    atomic_write_json,
    canonical_sha256,
    fit_activation_inventory,
    sha256_file,
)
from reverse_kv.provenance import (
    assert_model_instance_identity,
    load_verified_train_inventory,
)


def _formal_architecture() -> SimpleNamespace:
    return SimpleNamespace(
        source_layers=64,
        target_layers=40,
        kv_heads=8,
        head_dim=128,
        selected_source_layers=20,
        n_samples=128_000,
        source_features=20_480,
        target_features=1_024,
        expected_weight_count=1_677_721_600,
        expected_bias_count=81_920,
    )


def test_fit_cli_cannot_override_immutable_inputs() -> None:
    assert _sanitized_fit_argv(
        ["--devices", "0,1", "--target-layers", "0-3", "--dry-run"]
    ) == ["--devices", "0,1", "--target-layers", "0-3", "--dry-run"]
    forbidden = (
        ["--source-activations", "/tmp/evil"],
        ["--target-activations=/tmp/evil"],
        ["--selection", "/tmp/evil.json"],
        ["--output", "/tmp/evil"],
        ["--config", "/tmp/evil.yaml"],
        ["--ridge-lambda", "1"],
        ["--skip-payload-checksum"],
        ["--sour", "/tmp/abbreviation"],
    )
    for argv in forbidden:
        with pytest.raises(SystemExit):
            _sanitized_fit_argv(argv)


def test_fit_config_path_is_immutable(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="config is immutable"):
        fit_adapter.run_fit(config_path=tmp_path / "mapper.yaml")


def test_fit_config_cannot_redirect_selection_or_mapper(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        fit_adapter,
        "validate_config",
        lambda _path: {
            "outputs": {"selection": tmp_path / "selection.json", "mapper": tmp_path}
        },
    )
    with pytest.raises(ValueError, match="selection path is immutable"):
        fit_adapter.run_fit(config_path=fit_adapter.DEFAULT_CONFIG)


def test_fit_inventory_uses_the_actual_worker_roots_and_architecture(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        fit_adapter,
        "fit_activation_inventory",
        lambda *, verify_payloads: calls.append(verify_payloads) or {"verified": True},
    )
    architecture = _formal_architecture()
    assert _collect_exact_reverse_inventory(
        SOURCE.train_root,
        TARGET.train_root,
        architecture,
        verify_payload_checksum=True,
    ) == {"verified": True}
    assert calls == [True]
    with pytest.raises(ValueError, match="frozen 32B"):
        _collect_exact_reverse_inventory(
            "/tmp/evil",
            TARGET.train_root,
            architecture,
            verify_payload_checksum=True,
        )
    with pytest.raises(ValueError, match="full activation payload"):
        _collect_exact_reverse_inventory(
            SOURCE.train_root,
            TARGET.train_root,
            architecture,
            verify_payload_checksum=False,
        )


def _selection_fixture(root: Path) -> Path:
    statistics = root / "statistics"
    output = root / "selection"
    statistics.mkdir(parents=True)
    output.mkdir(parents=True)
    activation = {
        "schema_version": 1,
        "architecture": {
            "source_layers": 64,
            "target_layers": 40,
            "kv_heads": 8,
            "head_dim": 128,
            "top_k": 20,
        },
        "source": [{} for _ in range(128)],
        "target": [{} for _ in range(80)],
    }
    activation["input_fingerprint"] = canonical_sha256(activation)
    activation_path = statistics / "activation_inputs.json"
    atomic_write_json(activation_path, activation)
    moments_path = statistics / "moments_manifest.json"
    pairs_path = statistics / "single_source_manifest.json"
    atomic_write_json(
        moments_path, {"input_fingerprint": activation["input_fingerprint"]}
    )
    atomic_write_json(
        pairs_path, {"input_fingerprint": activation["input_fingerprint"]}
    )

    generator = np.random.default_rng(123)
    key = generator.normal(size=(40, 64, 8)).astype(np.float64)
    value = generator.normal(size=(40, 64, 8)).astype(np.float64)
    key_average = key.mean(axis=-1)
    value_average = value.mean(axis=-1)
    combined = (key_average + value_average) / 2.0
    arrays = {
        "key_head_uniform_r2": key,
        "value_head_uniform_r2": value,
        "key_head_global_r2_diagnostic": key.copy(),
        "value_head_global_r2_diagnostic": value.copy(),
        "key_head_averaged_uniform_r2": key_average,
        "value_head_averaged_uniform_r2": value_average,
        "combined_equal_kv_uniform_r2": combined,
        "solver_code": np.zeros((40, 2, 64, 8), dtype=np.int8),
    }
    score_path = output / "scores.npz"
    with score_path.open("wb") as handle:
        np.savez(handle, **arrays)
    score_sidecar = {
        "schema_version": 1,
        "payload": score_path.name,
        "sha256": sha256_file(score_path),
        "nbytes": score_path.stat().st_size,
        "arrays": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in sorted(arrays.items())
        },
        "kind": "selection_scores",
        "input_fingerprint": activation["input_fingerprint"],
        "method": selection_adapter.METHOD,
        "selection_score": "uniform_channel_r2_only",
        "global_r2_role": "diagnostic_only_never_used_for_ranking",
    }
    atomic_write_json(score_path.with_suffix(".json"), score_sidecar)
    heatmaps = {}
    for name in ("key", "value", "combined"):
        heatmap = output / f"heatmap_{name}.png"
        heatmap.write_bytes(b"synthetic-png")
        heatmaps[name] = {"path": str(heatmap), "sha256": sha256_file(heatmap)}
    indices = np.arange(64)
    rows = [
        [int(item) for item in np.lexsort((indices, -row))[:20]]
        for row in combined
    ]
    selection = {
        "schema_version": 1,
        "direction": "qwen3_14b_to_32b",
        "formal_protocol": True,
        "method": selection_adapter.METHOD,
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "k": 20,
        "n_samples": 128_000,
        "input_fingerprint": activation["input_fingerprint"],
        "token_sha256": ["a" * 64],
        "selected_source_layers_rank_order": rows,
        "feature_order": selection_adapter.FEATURE_ORDER,
        "selection_rule": dict(selection_adapter.SELECTION_RULE),
        "score_artifacts": {
            "npz": {"path": str(score_path), "sha256": sha256_file(score_path)},
            "heatmaps": heatmaps,
        },
        "sufficient_statistics": {
            "activation_inputs": {
                "path": str(activation_path),
                "sha256": sha256_file(activation_path),
            },
            "moments_manifest": {
                "path": str(moments_path),
                "sha256": sha256_file(moments_path),
            },
            "single_source_manifest": {
                "path": str(pairs_path),
                "sha256": sha256_file(pairs_path),
            },
        },
    }
    selection_path = output / "selected_layers.json"
    atomic_write_json(selection_path, selection)
    return selection_path


def test_selection_rows_are_recomputed_from_scores(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(selection_adapter, "EXPERIMENT_ROOT", tmp_path.resolve())
    selection_path = _selection_fixture(tmp_path)
    selection_adapter.validate_reverse_selection_artifact(
        selection_path,
        expected_direction="qwen3_14b_to_32b",
        require_implementation=False,
    )
    raw = json.loads(selection_path.read_text())
    raw["selected_source_layers_rank_order"][0] = list(reversed(range(20)))
    atomic_write_json(selection_path, raw)
    with pytest.raises(ValueError, match="score-derived top-20"):
        selection_adapter.validate_reverse_selection_artifact(
            selection_path,
            expected_direction="qwen3_14b_to_32b",
            require_implementation=False,
        )


def test_selection_rule_and_score_hash_tampering_fail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(selection_adapter, "EXPERIMENT_ROOT", tmp_path.resolve())
    selection_path = _selection_fixture(tmp_path)
    raw = json.loads(selection_path.read_text())
    raw["selection_rule"]["probe"] = "not same-head OLS"
    atomic_write_json(selection_path, raw)
    with pytest.raises(ValueError, match="selection_rule"):
        selection_adapter.validate_reverse_selection_artifact(
            selection_path,
            expected_direction="qwen3_14b_to_32b",
            require_implementation=False,
        )
    selection_path = _selection_fixture(tmp_path / "second")
    monkeypatch.setattr(selection_adapter, "EXPERIMENT_ROOT", (tmp_path / "second").resolve())
    with (selection_path.parent / "scores.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="SHA256"):
        selection_adapter.validate_reverse_selection_artifact(
            selection_path,
            expected_direction="qwen3_14b_to_32b",
            require_implementation=False,
        )


def _write_layer_fixture(root: Path, *, wrong_shapes: bool) -> tuple[str, str]:
    fingerprint = "f" * 64
    selection_sha = "s" * 64
    payload = root / "layers/layer_000.safetensors"
    payload.parent.mkdir(parents=True)
    tensors = (
        {
            "key_weight": torch.zeros(1, 4),
            "key_bias": torch.zeros(1),
            "value_weight": torch.zeros(4, 1),
            "value_bias": torch.zeros(3),
        }
        if wrong_shapes
        else {
            "key_weight": torch.zeros(2, 2),
            "key_bias": torch.zeros(2),
            "value_weight": torch.zeros(2, 2),
            "value_bias": torch.zeros(2),
        }
    )
    save_file(
        tensors,
        str(payload),
        metadata={
            "target_layer": "0",
            "source_layers_rank_order": "0",
            "source_heads": "1",
            "head_dim": "2",
        },
    )
    shapes = {
        "key_weight": [2, 2],
        "key_bias": [2],
        "value_weight": [2, 2],
        "value_bias": [2],
    }
    atomic_write_json(
        payload.with_suffix(".json"),
        {
            "target_layer": 0,
            "source_layers_rank_order": [0],
            "sha256": sha256_file(payload),
            "dtype": "float32",
            "source_heads": 1,
            "head_dim": 2,
            "tensor_shapes": shapes,
            "weight_numel": 8,
            "bias_numel": 4,
            "metadata": {
                "fit_fingerprint": fingerprint,
                "selection_manifest_sha256": selection_sha,
                "feature_order": "selected_layer_rank,kv_head,head_dim",
                "selected_source_layers_rank_order": [0],
                "ridge_lambda_added_directly_to_unnormalized_gram": 0.01,
                "standardize_features": False,
                "regularize_bias": False,
                "forward_activation_dtype": "bfloat16",
                "covariance_dtype": "float32",
                "mapper_dtype": "float32",
                "analysis_dtype": "float64",
                "allow_tf32": False,
            },
        },
    )
    return fingerprint, selection_sha


def test_manifest_payload_gate_loads_and_checks_actual_tensor_shapes(tmp_path: Path) -> None:
    fingerprint, selection_sha = _write_layer_fixture(tmp_path / "valid", wrong_shapes=False)
    result = _validate_layer_payload(
        tmp_path / "valid",
        target_layer=0,
        selected_layers=[0],
        source_features=2,
        target_features=2,
        kv_heads=1,
        head_dim=2,
        fit_fingerprint=fingerprint,
        selection_sha256=selection_sha,
    )
    assert result["tensor_numel"] == {
        "key_weight": 4,
        "key_bias": 2,
        "value_weight": 4,
        "value_bias": 2,
    }
    fingerprint, selection_sha = _write_layer_fixture(
        tmp_path / "tampered", wrong_shapes=True
    )
    with pytest.raises(ValueError, match="tensor key_weight"):
        _validate_layer_payload(
            tmp_path / "tampered",
            target_layer=0,
            selected_layers=[0],
            source_features=2,
            target_features=2,
            kv_heads=1,
            head_dim=2,
            fit_fingerprint=fingerprint,
            selection_sha256=selection_sha,
        )


def _write_inventory_snapshot(root: Path) -> tuple[dict, dict]:
    inventory = fit_activation_inventory(verify_payloads=False)
    inventory["payload_checksums_verified"] = True
    digest_value = dict(inventory)
    digest_value.pop("inventory_digest", None)
    inventory["inventory_digest"] = canonical_sha256(digest_value)
    path = root / "inputs/activation_inventory.json"
    atomic_write_json(path, inventory)
    manifest = {
        "activation_inventory": {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "inventory_digest": inventory["inventory_digest"],
            "payload_checksums_verified": True,
        }
    }
    return inventory, manifest


def test_mapper_inventory_binds_current_captures_and_models(tmp_path: Path) -> None:
    _, manifest = _write_inventory_snapshot(tmp_path)
    _, source_identity, target_identity = load_verified_train_inventory(tmp_path, manifest)
    source_model = SimpleNamespace(name_or_path=str(SOURCE.model_path))
    target_model = SimpleNamespace(name_or_path=str(TARGET.model_path))
    assert_model_instance_identity(source_model, source_identity, semantic_role="source")
    assert_model_instance_identity(target_model, target_identity, semantic_role="target")
    tampered = dict(source_identity)
    tampered["model_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="model_config_sha256"):
        assert_model_instance_identity(source_model, tampered, semantic_role="source")


def test_mapper_inventory_tamper_is_rejected(tmp_path: Path) -> None:
    inventory, manifest = _write_inventory_snapshot(tmp_path)
    inventory["entries"][0]["provenance"]["model_config_sha256"] = "0" * 64
    digest_value = dict(inventory)
    digest_value.pop("inventory_digest", None)
    inventory["inventory_digest"] = canonical_sha256(digest_value)
    path = tmp_path / "inputs/activation_inventory.json"
    atomic_write_json(path, inventory)
    manifest["activation_inventory"].update(
        sha256=sha256_file(path), inventory_digest=inventory["inventory_digest"]
    )
    with pytest.raises(ValueError, match="current captures"):
        load_verified_train_inventory(tmp_path, manifest)


def test_heldout_cannot_disable_payload_hashing() -> None:
    with pytest.raises(ValueError, match="full payload SHA256"):
        heldout_preflight(verify_payloads=False)


def test_reverse_heatmap_renderer_has_direction_correct_labels() -> None:
    source = inspect.getsource(selection_adapter._write_reverse_heatmap)
    assert "source layer (Qwen3-32B)" in source
    assert "target layer (Qwen3-14B)" in source
    assert "source layer (Qwen3-14B)" not in source
