from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import src.probe as probe
from src.probe import ProtocolError


def test_config_contract_is_full40_train_to_heldout():
    config = probe.load_config()
    assert config["experiment_id"] == "qwen3_32b_to_14b_cachebridge_head_local_full40_v2"
    assert config["protocol"]["full_target_layers"] == list(range(40))
    assert config["protocol"]["fit_split"] == "train"
    assert config["protocol"]["eval_split"] == "heldout_tuning"
    assert config["protocol"]["fit_sequence_count"] == 500
    assert config["protocol"]["eval_sequence_count"] == 40
    assert config["protocol"]["locked_sequence_count"] == 10
    assert config["outputs"]["stop_receipt"].endswith("PRE_FORMAL_STOP_v1.json")
    superseded = probe._validate_superseded_attempt31(config)
    assert superseded["status"] == "aborted_before_gpu_no_results"
    assert superseded["log"]["nbytes"] == 1895
    assert config["protocol"]["promotion_gate"]["primary_generalization_baseline"] == "nvidia_k20_train500"
    assert config["protocol"]["promotion_gate"]["shrinkage_comparator_status"] == "validation_tuned_on_heldout_tuning40_exploratory_only"
    bad = copy.deepcopy(config)
    bad["protocol"]["fit_sequence_count"] = True
    with pytest.raises(ProtocolError):
        probe.validate_config_shape(bad)


def test_split_namespaces_are_disjoint_even_when_numeric_ids_overlap(monkeypatch):
    config = probe.load_config()
    # Numeric IDs intentionally overlap (heldout tuning includes 0..49 in the
    # frozen source), so the contract is split-namespaced rather than a raw
    # integer-set comparison.
    probe.assert_split_ids([0, 499], config, "train")
    with pytest.raises(ProtocolError):
        probe.assert_split_ids([500], config, "train")
    heldout = (3, 6, 8, 15, 43, 31, 32, 35, 0, 37, 40, 19, 4, 16, 17, 28, 1, 45, 21, 26, 23, 38, 14, 25, 29, 10, 44, 33, 47, 24, 7, 46, 11, 42, 12, 36, 20, 2, 5, 30)
    monkeypatch.setattr(probe, "_load_split", lambda _config: {"tuning_sequence_ids": list(heldout), "locked_sequence_count": 10})
    probe.assert_split_ids([0, 30], config, "heldout_tuning")
    with pytest.raises(ProtocolError):
        probe.assert_split_ids([41], config, "heldout_tuning")
    probe.assert_fit_eval_split_disjoint(config)


def test_fit_eval_disjoint_guard_rejects_aliases():
    config = probe.load_config()
    bad = copy.deepcopy(config)
    bad["upstream"]["fit_source_activations"] = bad["upstream"]["source_activations"]
    with pytest.raises(ProtocolError):
        probe.assert_fit_eval_split_disjoint(bad)


def test_superseded_dir31_log_binding_is_strict():
    config = probe.load_config()
    bad = copy.deepcopy(config)
    bad["superseded_attempt31"]["gpu_started"] = True
    with pytest.raises(ProtocolError):
        probe.validate_config_shape(bad)
    bad = copy.deepcopy(config)
    bad["superseded_attempt31"]["log"]["sha256"] = "0" * 64
    with pytest.raises(ProtocolError):
        probe.validate_config_shape(bad)
    bad = copy.deepcopy(config)
    bad["upstream"]["train_tokens"] = copy.deepcopy(bad["upstream"]["heldout_tokens"])
    with pytest.raises(ProtocolError):
        probe.assert_fit_eval_split_disjoint(bad)


def test_selection_snapshot_is_complete_and_matches_both_manifests():
    config = probe.load_config()
    k20 = probe.strict_json(probe._verify_config_ref(config, "k20_manifest"))
    shrink = probe.strict_json(probe._verify_config_ref(config, "shrinkage_manifest"))
    rows = probe._validate_complete_selection_snapshot(config, k20, shrink)
    assert len(rows) == 40
    assert set(value for row in rows for value in row) == set(range(64))


def _gate_rows(candidate: float = 1.0, nvidia: float = 2.0, shrink: float = 3.0):
    return [
        {
            "target_layer": layer,
            "component": component,
            "eval_split": "heldout_tuning",
            "observations": 40 * 256,
            "candidate_sse": float(candidate),
            "nvidia_k20_sse": float(nvidia),
            "shrinkage_sse": float(shrink),
            "strictly_below_both": bool(candidate < nvidia and candidate < shrink),
        }
        for layer in probe.FULL_LAYERS
        for component in probe.COMPONENTS
    ]


def test_full40_promotion_gate_exact_80_rows_and_split():
    rows = _gate_rows()
    assert probe._validate_full_gate_rows(rows) is True
    rows[0]["candidate_sse"] = 4.0
    rows[0]["strictly_below_both"] = False
    assert probe._validate_full_gate_rows(rows) is False
    rows[0]["strictly_below_both"] = True
    with pytest.raises(ProtocolError):
        probe._validate_full_gate_rows(rows)
    rows = _gate_rows()
    rows[0]["eval_split"] = "train"
    with pytest.raises(ProtocolError):
        probe._validate_full_gate_rows(rows)
    rows = _gate_rows()
    rows[0]["candidate_sse"] = True
    with pytest.raises(ProtocolError):
        probe._validate_full_gate_rows(rows)
    with pytest.raises(ProtocolError):
        probe._validate_full_gate_rows(rows[:-1])


def _synthetic_result(pre_hash: str = "a" * 64, passed: bool = True):
    rows = _gate_rows()
    layers = {
        str(layer): {
            component: {
                "fit": {"n_observations": 500 * 256},
                "promotion": rows[layer * 2 + (0 if component == "key" else 1)],
            }
            for component in probe.COMPONENTS
        }
        for layer in probe.FULL_LAYERS
    }
    return {
        "schema_version": 1,
        "protocol": probe.PROTOCOL,
        "status": "complete",
        "stage": "full40",
        "decision": "GO" if passed else "NO_GO",
        "promotion_gate": {"required": True, "passed": passed, "rows": rows},
        "layers": layers,
        "input_inventory_sha256": pre_hash,
        "superseded_attempt31": probe._validate_superseded_attempt31(probe.load_config()),
        "fit_split": "train",
        "eval_split": "heldout_tuning",
        "fit_sequence_count": 500,
        "eval_sequence_count": 40,
        "fit_observations": 500 * 256,
        "eval_observations": 40 * 256,
        "fit_eval_ids_disjoint": True,
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "answers_or_outcomes_read": False,
        "prior_probe6_is_not_qa_proof": True,
        "shrinkage_comparator_status": "validation_tuned_on_heldout_tuning40_exploratory_only",
    }


def test_full40_result_schema_and_tamper_guard():
    pre = {"input_inventory_sha256": "a" * 64, "superseded_attempt31": probe._validate_superseded_attempt31(probe.load_config())}
    result = _synthetic_result()
    assert probe._validate_full40_result(result, pre) is True
    bad = copy.deepcopy(result)
    bad["fit_split"] = "heldout_tuning"
    with pytest.raises(ProtocolError):
        probe._validate_full40_result(bad, pre)
    bad = copy.deepcopy(result)
    bad["layers"]["0"]["key"]["fit"]["n_observations"] = 40 * 256
    with pytest.raises(ProtocolError):
        probe._validate_full40_result(bad, pre)
    bad = copy.deepcopy(result)
    bad["prior_probe6_is_not_qa_proof"] = False
    with pytest.raises(ProtocolError):
        probe._validate_full40_result(bad, pre)
    bad = copy.deepcopy(result)
    bad["superseded_attempt31"]["status"] = "complete"
    with pytest.raises(ProtocolError):
        probe._validate_full40_result(bad, pre)


def test_shrinkage_alpha_is_explicitly_heldout_tuned():
    config = probe.load_config()
    shrink = probe.strict_json(probe._verify_config_ref(config, "shrinkage_manifest"))
    split = probe._load_split(config)
    root = Path(config["upstream"]["shrinkage_root"])
    binding = probe._validate_shrinkage_alpha_binding(config, shrink, root, split)
    assert binding["fit_split"] == "heldout_tuning"
    assert binding["status"] == "validation_tuned_exploratory_only"
    assert binding["fit_sequence_count"] == 40
    assert binding["locked_sequence_count"] == 10


def test_calibration_binding_covers_both_token_files():
    config = probe.load_config()
    binding = probe._load_calibration_binding(config)
    assert binding["splits"]["train"]["sequence_count"] == 500
    assert binding["splits"]["heldout"]["sequence_count"] == 50
    assert binding["splits"]["train"]["token_ref"]["sha256"] == config["upstream"]["train_tokens"]["sha256"]
    assert binding["splits"]["heldout"]["token_ref"]["sha256"] == config["upstream"]["heldout_tokens"]["sha256"]


def test_source_closure_is_sorted_and_rejects_generated_files(tmp_path: Path):
    root = tmp_path / "snapshot"
    (root / "src").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "tests").mkdir()
    (root / "contracts").mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "README.md").write_text("x", encoding="utf-8")
    (root / "src" / "a.py").write_text("x=1\n", encoding="utf-8")
    old = probe.ROOT
    probe.ROOT = root
    try:
        closure = probe.source_byte_closure()
        assert closure["file_count"] == len(closure["files"])
        assert [row["path"] for row in closure["files"]] == sorted(row["path"] for row in closure["files"])
        (root / "src" / "__pycache__").mkdir()
        (root / "src" / "__pycache__" / "x.pyc").write_bytes(b"bad")
        with pytest.raises(ProtocolError):
            probe.source_byte_closure()
    finally:
        probe.ROOT = old


def test_symlink_and_noncanonical_refs_fail_closed(tmp_path: Path):
    target = tmp_path / "target.txt"
    target.write_bytes(b"x")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    ref = {"path": str(link), "nbytes": 1, "sha256": probe.sha256_file(target)}
    with pytest.raises(ProtocolError):
        probe._strict_ref(ref, label="evil")
    with pytest.raises(ProtocolError):
        probe._strict_ref({**ref, "path": "../target.txt"}, label="evil")


def test_sidecar_is_content_bound_and_immutable(tmp_path: Path):
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


def test_numpy_head_local_fit_metrics_and_layout():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 2560))
    true_w = rng.normal(scale=0.01, size=(2560, 128))
    true_b = rng.normal(size=(128,))
    y = x @ true_w + true_b
    w, b, diagnostics = probe.fit_head_local_numpy(x, y)
    prediction = probe.predict_head_local(x, w, b)
    assert diagnostics["n_observations"] == 24
    assert prediction.shape == y.shape
    target = np.zeros((24, 8, 128), dtype=np.float64)
    pred = target.copy()
    pred[0, 0, 0] = 1.0
    assert probe.activation_metrics(target, pred)["observations"] == 24
    joint, flat_bias = probe.block_diagonal_head_weights(np.zeros((8, 2560, 128), np.float32), np.zeros((8, 128), np.float32))
    assert joint.shape == (20480, 1024)
    assert flat_bias.shape == (1024,)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_full40_source_axis_layout_is_numeric_and_batch_safe(batch_size: int):
    """A small tensor catches the dim=1 stack/reshape regression.

    The production stack is [B,L,P,H,D].  Every value encodes its original
    batch/layer/position, so a wrong permutation visibly swaps source-layer
    and position values for both B=1 and B>1.
    """

    import torch

    layers, positions, heads, dims = 3, 4, 1, 1
    stacked = torch.empty((batch_size, layers, positions, heads, dims), dtype=torch.int64)
    for b in range(batch_size):
        for layer in range(layers):
            for position in range(positions):
                stacked[b, layer, position, 0, 0] = b * 1000 + layer * 100 + position * 10
    converted = probe._source_batch_to_position_layer_major(stacked)
    assert tuple(converted.shape) == (batch_size, positions, layers, heads, dims)
    for b in range(batch_size):
        for position in range(positions):
            expected = [b * 1000 + layer * 100 + position * 10 for layer in range(layers)]
            assert converted[b, position, :, 0, 0].tolist() == expected


def test_full40_source_axis_layout_rejects_non_five_axis_input():
    import torch

    with pytest.raises(ProtocolError):
        probe._source_batch_to_position_layer_major(torch.zeros((2, 3, 4, 1)))


def test_model_inventory_synthetic(tmp_path: Path):
    root = tmp_path / "model"
    root.mkdir()
    model_config = {
        "model_type": "qwen3", "hidden_size": 5120, "num_hidden_layers": 40,
        "num_attention_heads": 40, "num_key_value_heads": 8, "head_dim": 128,
        "torch_dtype": "bfloat16", "vocab_size": 151936,
        "architectures": ["Qwen3ForCausalLM"],
    }
    (root / "config.json").write_text(json.dumps(model_config), encoding="utf-8")
    for name in probe.MODEL_INVENTORY_SUPPORTING_FILES:
        (root / name).write_text("{}", encoding="utf-8")
    shard = root / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 7}, "weight_map": {"x": shard.name}}), encoding="utf-8")
    digest = probe.canonical_sha256([probe.file_ref(shard), *[probe.file_ref(root / n) for n in probe.MODEL_INVENTORY_SUPPORTING_FILES]])
    entry = {
        "path": str(root),
        "config": probe.file_ref(root / "config.json"),
        "weight_index": probe.file_ref(root / "model.safetensors.index.json"),
        "tokenizer": probe.file_ref(root / "tokenizer.json"),
        "inventory_digest": digest,
        "metadata_total_size": 7,
        "filesystem_total_size": 7,
        "shard_count": 1,
        "expected_num_hidden_layers": 40,
        "expected_num_attention_heads": 40,
        "expected_num_key_value_heads": 8,
        "expected_head_dim": 128,
    }
    assert probe.model_shard_inventory(entry, label="synthetic")["digest"] == digest


def test_import_is_cpu_only_with_fresh_external_prefix(tmp_path: Path):
    prefix = tmp_path / "absent-leaf"
    env = os.environ.copy()
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": str(prefix), "CUDA_VISIBLE_DEVICES": ""})
    env.pop("PYTHONPATH", None)
    code = "import sys; sys.path.insert(0, '.'); import src.probe; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], cwd=probe.ROOT, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not prefix.exists()


def test_execute_requires_explicit_full40_flag():
    with pytest.raises(ProtocolError):
        probe.run_stage(probe.ROOT / "config.json", "full40", execute=False)
    with pytest.raises(ProtocolError):
        probe.run_stage(probe.ROOT / "config.json", "probe6", execute=False)
