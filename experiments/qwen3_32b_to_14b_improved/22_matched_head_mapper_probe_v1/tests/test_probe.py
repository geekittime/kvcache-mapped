from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import numpy as np
import pytest

import src.probe as probe
from src.probe import ProtocolError


def test_config_contract_and_model_bindings_are_strict():
    config = probe.load_config()
    assert config["upstream"]["source_model"]["inventory_digest"] == "6de586248489a6ea41f95f210ef1b2b9542537c0d29fd170ec7e2eb092f4e3d0"
    assert config["upstream"]["target_model"]["inventory_digest"] == "78f4ba04683d6d2dd58256d56527d6848690b185636f87dbf2eeb9131775f902"
    assert config["protocol"]["probe_target_layers"] == [0, 7, 15, 23, 31, 39]
    assert config["protocol"]["locked_activation_evaluations"] == 0
    bad = copy.deepcopy(config)
    bad["upstream"]["source_model"]["shard_count"] = True
    with pytest.raises(ProtocolError):
        probe.validate_config_shape(bad)


def test_source_closure_is_regular_and_deterministic():
    closure = probe.source_byte_closure()
    assert closure["file_count"] == len(closure["files"])
    assert len(closure["digest"]) == 64
    assert [row["path"] for row in closure["files"]] == sorted(row["path"] for row in closure["files"])
    assert all(not row["path"].endswith(".pyc") for row in closure["files"])


def test_symlink_and_noncanonical_refs_fail_closed(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"x")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    ref = {"path": str(link), "nbytes": 1, "sha256": probe.sha256_file(target)}
    with pytest.raises(ProtocolError):
        probe._strict_ref(ref, label="evil")
    relative = dict(ref, path="../target.txt")
    with pytest.raises(ProtocolError):
        probe._strict_ref(relative, label="evil")


def test_sidecar_is_content_bound_and_immutable(tmp_path: Path):
    # Temporarily point output safety at a fixture root; this does not touch
    # the experiment's real artifacts.
    old_root = probe.ROOT
    probe.ROOT = tmp_path
    try:
        destination = tmp_path / "artifacts" / "x.json"
        ref = probe.immutable_json(destination, {"a": 1})
        side = probe.write_sidecar(destination)
        assert ref["nbytes"] == destination.stat().st_size
        probe.verify_sidecar(destination)
        assert side["sha256"] == probe.sha256_file(destination.with_name("x.json.sha256.json"))
        with pytest.raises(ProtocolError):
            probe.immutable_json(destination, {"a": 2})
        destination.write_bytes(b"tampered")
        with pytest.raises(ProtocolError):
            probe.verify_sidecar(destination)
    finally:
        probe.ROOT = old_root


def test_output_parent_symlink_is_rejected(tmp_path: Path):
    old_root = probe.ROOT
    probe.ROOT = tmp_path
    try:
        outside = tmp_path / "outside"
        outside.mkdir()
        (tmp_path / "artifacts").symlink_to(outside, target_is_directory=True)
        with pytest.raises(ProtocolError):
            probe.atomic_json(tmp_path / "artifacts" / "x.json", {"x": 1})
    finally:
        probe.ROOT = old_root


def test_tuning_boundary_rejects_locked_ids(monkeypatch):
    config = probe.load_config()
    monkeypatch.setattr(probe, "_load_split", lambda _config: {
        "tuning_sequence_ids": [0, 1],
        "tuning_sequence_ids_sha256": "0" * 64,
        "locked_sequence_count": 1,
        "locked_sequence_ids_sha256": "1" * 64,
        "locked_rows_indexed": False,
        "fold_manifest_sha256": "2" * 64,
    })
    probe.assert_tuning_ids([0, 1], config)
    with pytest.raises(ProtocolError):
        probe.assert_tuning_ids([2], config)
    with pytest.raises(ProtocolError):
        probe.assert_tuning_ids([True], config)


def test_head_local_numpy_fit_and_metrics():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 2560))
    true_w = rng.normal(scale=0.01, size=(2560, 128))
    true_b = rng.normal(size=(128,))
    y = x @ true_w + true_b
    w, b, diagnostics = probe.fit_head_local_numpy(x, y)
    prediction = probe.predict_head_local(x, w, b)
    assert diagnostics["ridge_lambda"] == 0.01
    assert prediction.shape == y.shape
    metric_target = np.zeros((24, 8, 128), dtype=np.float64)
    metric_prediction = metric_target.copy()
    metric_prediction[0, 0, 0] = 1.0
    metrics = probe.activation_metrics(metric_target, metric_prediction)
    assert metrics["sse"] >= 0.0
    assert metrics["observations"] == 24


def test_block_diagonal_layout():
    w = np.zeros((8, 2560, 128), dtype=np.float32)
    b = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    joint, flat_bias = probe.block_diagonal_head_weights(w, b)
    assert joint.shape == (20480, 1024)
    assert flat_bias.shape == (1024,)
    assert np.array_equal(flat_bias, b.reshape(-1))


def test_execute_requires_explicit_flag_and_frozen_stop():
    with pytest.raises(ProtocolError):
        probe.run_stage(probe.ROOT / "config.json", "probe6", execute=False)


def test_import_is_cpu_only_subprocess(tmp_path: Path):
    prefix = tmp_path / "absent-leaf"
    env = os.environ.copy()
    env.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(prefix),
        "CUDA_VISIBLE_DEVICES": "",
    })
    env.pop("PYTHONPATH", None)
    code = "import sys; sys.path.insert(0, '.'); import src.probe; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=probe.ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not prefix.exists()


def test_model_inventory_synthetic(tmp_path: Path):
    root = tmp_path / "model"
    root.mkdir()
    config_value = {
        "model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40,
        "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
        "torch_dtype": "bfloat16", "vocab_size": 151936,
        "architectures": ["Qwen3ForCausalLM"],
    }
    (root / "config.json").write_text(json.dumps(config_value), encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    for name in probe.MODEL_INVENTORY_SUPPORTING_FILES[1:]:
        (root / name).write_text("{}", encoding="utf-8")
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    index_value = {"metadata": {"total_size": shard.stat().st_size}, "weight_map": {"x": shard.name}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index_value), encoding="utf-8")
    shard_refs = [probe.file_ref(shard)]
    supporting = [probe.file_ref(root / name) for name in probe.MODEL_INVENTORY_SUPPORTING_FILES]
    digest = probe.canonical_sha256(shard_refs + supporting)
    entry = {
        "path": str(root),
        "config": probe.file_ref(root / "config.json"),
        "weight_index": probe.file_ref(root / "model.safetensors.index.json"),
        "tokenizer": probe.file_ref(root / "tokenizer.json"),
        "inventory_digest": digest,
        "metadata_total_size": shard.stat().st_size,
        "filesystem_total_size": shard.stat().st_size,
        "shard_count": 1,
        "expected_num_hidden_layers": 40,
        "expected_num_attention_heads": 40,
        "expected_num_key_value_heads": 8,
        "expected_head_dim": 128,
    }
    observed = probe.model_shard_inventory(entry, label="synthetic")
    assert observed["digest"] == digest
    assert observed["shard_count"] == 1
