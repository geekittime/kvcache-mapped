from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import reverse_kv.mechanism as mechanism
from reverse_kv.mechanism import (
    MechanismAccumulator,
    StreamingR2State,
    layer_mechanism_metrics,
    load_checkpoint,
    load_mechanism_config,
    preprocess_hellaswag,
    resume_accumulator,
    save_checkpoint,
    select_documents,
)


def _row(index: int, label: str = "0") -> dict[str, object]:
    return {
        "ind": index + 100,
        "activity_label": "An activity",
        "ctx_a": "A person moves",
        "ctx_b": "THE Camera pans",
        "endings": ["one", "two", "three", "four"],
        "source_id": f"source-{index}",
        "label": label,
    }


def _plan() -> list[dict[str, object]]:
    return [
        {
            "request_index": index,
            "input_token_ids_sha256": f"{index:064x}",
        }
        for index in range(400)
    ]


def test_mechanism_config_is_sealed() -> None:
    config = load_mechanism_config()
    assert config["paper_reference"]["qwen3_32b_to_14b_single_pair_values_published"] is False
    assert config["paper_reference"]["published_cross_pair_evaluations"] == 12
    assert config["protocol"]["scoring_sequences"] == 400
    assert config["protocol"]["r2_key_space"] == (
        "both_target_rope_stripped_content_primary_and_target_rope_cache_diagnostic"
    )
    assert config["runtime"]["source_layer_counts"] == [12, 13, 14, 13, 12]
    assert config["runtime"]["target_layer_counts"] == [13, 14, 13]


def test_hellaswag_preprocessing_matches_pinned_task() -> None:
    assert preprocess_hellaswag("  A [artifact]  B [title] C  ") == "A  B. C"


def test_sha_selection_is_deterministic_and_gold_independent() -> None:
    rows = [_row(index, str(index % 4)) for index in range(12)]
    changed = copy.deepcopy(rows)
    for row in changed:
        row["label"] = str((int(row["label"]) + 1) % 4)
    first = select_documents(rows, count=5, seed="fixed")
    second = select_documents(changed, count=5, seed="fixed")
    assert [(row["row_index"], row["rank_sha256"]) for row in first] == [
        (row["row_index"], row["rank_sha256"]) for row in second
    ]
    # Python str.capitalize lower-cases the rest; this catches a subtle drift
    # from the vendored HellaSwag process_docs implementation.
    assert first[0]["query"].endswith("The camera pans")


def test_streaming_r2_matches_direct_channel_formula() -> None:
    target = torch.tensor(
        [[1.0, 4.0], [2.0, 5.0], [4.0, 9.0], [8.0, 10.0]], dtype=torch.float32
    )
    prediction = target + torch.tensor(
        [[0.1, -0.2], [0.0, 0.3], [-0.4, 0.1], [0.2, -0.1]]
    )
    state = StreamingR2State(2, 2)
    state.update(0, target[:2], prediction[:2])
    state.update(0, target[2:], prediction[2:])
    result = state.finalize_layer(0)
    true = target.double().numpy()
    pred = prediction.double().numpy()
    sse = np.square(true - pred).sum(axis=0)
    sst = np.square(true - true.mean(axis=0)).sum(axis=0)
    expected = np.mean(1.0 - sse / sst)
    assert result["observations"] == 4
    assert result["uniform_channel_r2_fp64"] == pytest.approx(expected, abs=1e-14)


def test_layer_mechanism_key_concentration_matches_closed_form() -> None:
    # Q^TQ=diag(4,1); K-error covariance=diag(1,0), hence
    # concentration = d*4/(5*1) = 1.6.
    query = torch.tensor([[[[2.0, 0.0], [0.0, 1.0]]]])
    gt_key = torch.tensor([[[[0.2, 0.1], [0.3, -0.1]]]])
    mapped_key = gt_key + torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    gt_value = torch.tensor([[[[1.0, 0.5], [0.2, -0.4]]]])
    mapped_value = gt_value + torch.tensor([[[[0.2, 0.1], [-0.3, 0.4]]]])
    result = layer_mechanism_metrics(
        query, gt_key, gt_value, mapped_key, mapped_value
    )
    assert result["key_concentration_per_kv_head"].tolist() == pytest.approx([1.6])
    assert result["value_concentration_per_kv_head"].shape == (1,)
    assert result["attention_cosine_count_per_query_head"].tolist() == [2]
    assert math.isfinite(result["attention_cosine_sum_per_query_head"].item())


def test_layer_mechanism_rejects_undefined_zero_error() -> None:
    query = torch.ones(1, 2, 3, 4)
    key = torch.ones(1, 1, 3, 4)
    value = torch.ones(1, 1, 3, 4)
    with pytest.raises(ValueError, match="K concentration is undefined"):
        layer_mechanism_metrics(query, key, value, key.clone(), value + 0.1)


def test_checkpoint_is_pickle_free_and_round_trips(tmp_path: Path) -> None:
    accumulator = MechanismAccumulator()
    # A valid checkpoint may represent only a completed prefix; its untouched
    # layers are still well-typed sufficient-statistic arrays.
    accumulator.key_r2.update(0, torch.ones(2, 1024), torch.zeros(2, 1024))
    accumulator.key_rope_r2.update(
        0, torch.ones(2, 1024), torch.zeros(2, 1024)
    )
    accumulator.value_r2.update(0, torch.ones(2, 1024), torch.zeros(2, 1024))
    plan = _plan()
    path = save_checkpoint(
        tmp_path,
        accumulator,
        completed=10,
        fingerprint="a" * 64,
        plan=plan,
    )
    values, completed = load_checkpoint(path, fingerprint="a" * 64, plan=plan)
    restored, resumed, latest = resume_accumulator(
        tmp_path, fingerprint="a" * 64, plan=plan
    )
    assert completed == resumed == 10
    assert latest == path
    assert np.array_equal(restored.key_r2.sse, values["key_r2_sse"])
    assert np.array_equal(restored.key_rope_r2.sse, values["key_rope_r2_sse"])
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(path, fingerprint="b" * 64, plan=plan)
    sidecar = path.with_name(f"{path.name}.sha256.json")
    changed = json.loads(sidecar.read_text(encoding="utf-8"))
    changed["plan_prefix_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar changed"):
        load_checkpoint(path, fingerprint="a" * 64, plan=plan)


def test_resume_rejects_missing_checkpoint_sidecar_without_recreating(
    tmp_path: Path,
) -> None:
    accumulator = MechanismAccumulator()
    plan = _plan()
    path = save_checkpoint(
        tmp_path,
        accumulator,
        completed=10,
        fingerprint="a" * 64,
        plan=plan,
    )
    sidecar = path.with_name(f"{path.name}.sha256.json")
    sidecar.unlink()
    with pytest.raises(ValueError, match="SHA sidecar is missing"):
        resume_accumulator(tmp_path, fingerprint="a" * 64, plan=plan)
    assert not sidecar.exists()


def test_r2_state_rejects_nonfinite_inputs() -> None:
    state = StreamingR2State(1, 2)
    with pytest.raises(ValueError, match="NaN/Inf"):
        state.update(0, torch.tensor([[1.0, float("nan")]]), torch.zeros(1, 2))


def test_formal_receipt_recomputes_checkpoint_semantics_after_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coordinated result/sidecar edits still cannot outrun final-state replay."""

    monkeypatch.setattr(mechanism, "TARGET_LAYERS", 1)
    monkeypatch.setattr(mechanism, "KV_HEADS", 1)
    monkeypatch.setattr(mechanism, "QUERY_HEADS", 1)
    monkeypatch.setattr(mechanism, "CHANNELS", 2)
    monkeypatch.setattr(mechanism, "REQUESTS", 1)
    config = {
        "paper_reference": {"sealed": "paper-boundary"},
        "protocol": {"selection_seed": "sealed-seed"},
        "runtime": {
            "source_devices": [0],
            "source_layer_counts": [1],
            "target_devices": [1],
            "target_layer_counts": [1],
            "mapper_token_chunk": 1024,
        },
    }
    monkeypatch.setattr(mechanism, "load_mechanism_config", lambda: config)
    monkeypatch.setattr(mechanism, "validate_gpu_idle_gate", lambda _output: {})

    output = tmp_path / "mechanism"
    output.mkdir()
    upstream = tmp_path / "hellaswag" / "formal_hellaswag_receipt.json"
    upstream.parent.mkdir()
    mechanism.hashed_json(upstream, {"status": "complete"})
    live_hellaswag = {
        "receipt": str(upstream.resolve()),
        "receipt_sha256": mechanism.sha256_file(upstream),
        "receipt_nbytes": upstream.stat().st_size,
        "request_fingerprint": "h" * 64,
        "target_standalone_acc_norm": 0.8,
        "mapped_acc_norm": 0.7,
        "mapped_over_target_retention_percent": 87.5,
    }
    monkeypatch.setattr(
        mechanism, "_hellaswag_binding", lambda _root: live_hellaswag
    )
    heldout_path = tmp_path / "heldout.json"
    mechanism.hashed_json(heldout_path, {"status": "complete"})
    live_heldout = {
        "scope": "local_independent_fineweb_heldout_not_paper_disclosed_calibration_scalar",
        "path": str(heldout_path.resolve()),
        "sha256": mechanism.sha256_file(heldout_path),
        "nbytes": heldout_path.stat().st_size,
    }
    monkeypatch.setattr(mechanism, "_heldout_binding", lambda _path: live_heldout)

    selection_rows = [
        {
            "selection_index": index,
            "row_index": index,
            "rank_sha256": f"{index:064x}",
            "ind": index,
            "source_id": f"source-{index}",
        }
        for index in range(100)
    ]
    selection = {
        "schema_version": 1,
        "method": "sha256_rank_without_gold_label",
        "seed": "sealed-seed",
        "dataset_sha256": mechanism.EXPECTED_VALIDATION_SHA256,
        "selected_documents": selection_rows,
        "selection_sha256": mechanism.canonical_sha256(selection_rows),
    }
    mechanism.hashed_json(output / "selection.json", selection)
    token_ids = [11, 12]
    plan = [
        {
            "request_index": 0,
            "selection_index": 0,
            "choice_index": 0,
            "row_index": 0,
            "rank_sha256": f"{0:064x}",
            "ind": 0,
            "source_id": "source-0",
            "input_token_ids": token_ids,
            "input_token_count": len(token_ids),
            "input_token_ids_sha256": mechanism.token_ids_sha256(token_ids),
        }
    ]
    plan_marker = mechanism._persist_immutable_jsonl(output / "token_plan.jsonl", plan)
    plan_identity = {
        "path": str((output / "token_plan.jsonl").resolve()),
        **plan_marker,
        "token_count_min": 2,
        "token_count_max": 2,
        "token_count_total": 2,
    }
    source_model = {
        "path": "/models/source",
        "content_hashed_files": {
            "config.json": {"sha256": "1" * 64},
            "tokenizer.json": {"sha256": "2" * 64},
            "model.safetensors.index.json": {"sha256": "3" * 64},
        },
    }
    target_model = {
        "path": "/models/target",
        "content_hashed_files": {
            "config.json": {"sha256": "4" * 64},
            "tokenizer.json": {"sha256": "2" * 64},
            "model.safetensors.index.json": {"sha256": "5" * 64},
        },
    }
    mapper_artifact = {
        "manifest": "/mapper/manifest.json",
        "manifest_sha256": "6" * 64,
        "fit_fingerprint": "7" * 64,
        "target_layer_count": 1,
    }
    source_capture_identity = {
        "model_name": "qwen3_32b",
        "model_path": "/models/source",
        "model_role": "target",
        "model_config_sha256": "1" * 64,
        "model_tokenizer_sha256": "2" * 64,
        "model_weight_index_sha256": "3" * 64,
        "capture_run_fingerprint": "8" * 64,
    }
    target_capture_identity = {
        "model_name": "qwen3_14b",
        "model_path": "/models/target",
        "model_role": "source",
        "model_config_sha256": "4" * 64,
        "model_tokenizer_sha256": "2" * 64,
        "model_weight_index_sha256": "5" * 64,
        "capture_run_fingerprint": "9" * 64,
    }
    definition = {
        "schema_version": 1,
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism_local_completion",
        "direction": mechanism.DIRECTION,
        "paper_reference": config["paper_reference"],
        "local_protocol": config["protocol"],
        "runtime": config["runtime"],
        "artifacts": {
            "source_model": source_model,
            "target_model": target_model,
            "mapper": mapper_artifact,
            "formal_hellaswag": live_hellaswag,
            "local_heldout_reconstruction": live_heldout,
        },
        "token_plan": plan_identity,
    }
    current_definition = copy.deepcopy(definition)
    monkeypatch.setattr(
        mechanism,
        "_definition",
        lambda _configuration, _plan, _hellaswag, _heldout: copy.deepcopy(
            current_definition
        ),
    )
    monkeypatch.setattr(
        mechanism,
        "_current_runtime_mapper_identity",
        lambda _configuration: {
            **mapper_artifact,
            "layers": 1,
            "selected_source_layers": mechanism.TOP_K,
            "source_capture_model_identity": source_capture_identity,
            "target_capture_model_identity": target_capture_identity,
        },
    )
    fingerprint = mechanism.canonical_sha256(definition)
    mechanism.hashed_json(
        output / "run_definition.json",
        {**definition, "request_fingerprint": fingerprint},
    )
    mechanism.hashed_json(output / "gpu_idle_gate.json", {"status": "passed"})

    def sharding(*, device: int, role: str) -> dict[str, object]:
        return {
            "status": "passed",
            "role": role,
            "mode": "explicit_exact_counts",
            "devices": [device],
            "declared_layer_counts": [1],
            "expected_layer_counts": [1],
            "actual_layer_counts": [1],
            "total_layers": 1,
            "layer_devices": [device],
            "contiguous_ranges": [
                {
                    "device": device,
                    "start_layer": 0,
                    "end_layer_exclusive": 1,
                    "count": 1,
                }
            ],
            "actual_parameters_inspected_for_every_layer": True,
            "parameter_tensors_inspected": 10,
            "auxiliary_modules": {
                name: {
                    "expected_device": device,
                    "actual_device": device,
                    "parameter_tensors_inspected": 1,
                }
                for name in ("model.embed_tokens", "model.norm", "lm_head")
            },
            "auxiliary_parameter_tensors_inspected": 3,
            "auxiliary_parameters_inspected_for_every_module": True,
        }

    one_ms = {
        "unit": "ms",
        "count": 1,
        "mean": 1.0,
        "median": 1.0,
        "p10": 1.0,
        "p50": 1.0,
        "p90": 1.0,
        "std_population": 0.0,
        "min": 1.0,
        "max": 1.0,
        "samples": [1.0],
    }
    two_ms = {**one_ms, **{
        field: 2.0
        for field in ("mean", "median", "p10", "p50", "p90", "min", "max")
    }, "samples": [2.0]}
    runtime_setup = {
        "schema_version": 1,
        "status": "ready",
        "request_fingerprint": fingerprint,
        "source_sharding": sharding(device=0, role="target"),
        "target_sharding": sharding(device=1, role="source"),
        "mapper": {
            "direction": mechanism.DIRECTION,
            "manifest": mapper_artifact["manifest"],
            "manifest_sha256": mapper_artifact["manifest_sha256"],
            "fit_fingerprint": mapper_artifact["fit_fingerprint"],
            "layers": 1,
            "selected_source_layers": mechanism.TOP_K,
            "token_chunk": 1024,
            "execution": (
                "fp32_mapper_weights_resident_on_target_layer_devices"
            ),
            "target_layer_devices": ["cuda:1"],
            "resident_parameter_bytes": (
                mechanism.WEIGHT_COUNT + mechanism.BIAS_COUNT
            ) * 4,
            "materialization": {
                "scope": "one_time_setup_excluded_from_per_request_latency",
                "host_load_and_hash": one_ms,
                "weight_host_to_device": {
                    **two_ms,
                    "timing_backends": [
                        "host_perf_counter_with_cuda_synchronize"
                    ],
                },
                "per_layer": [
                    {
                        "target_layer": 0,
                        "device": "cuda:1",
                        "host_load_and_hash_ms": 1.0,
                        "weight_host_to_device_ms": 2.0,
                    }
                ],
            },
            "source_model_artifact": {
                "model_path": "/models/source",
                "model_config_sha256": "1" * 64,
                "model_tokenizer_sha256": "2" * 64,
                "model_weight_index_sha256": "3" * 64,
            },
            "target_model_artifact": {
                "model_path": "/models/target",
                "model_config_sha256": "4" * 64,
                "model_tokenizer_sha256": "2" * 64,
                "model_weight_index_sha256": "5" * 64,
            },
            "source_capture_model_identity": source_capture_identity,
            "target_capture_model_identity": target_capture_identity,
        },
    }
    runtime_setup_path = output / "runtime_setup.json"
    mechanism.hashed_json(runtime_setup_path, runtime_setup)

    accumulator = mechanism.MechanismAccumulator()
    truth = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    prediction = truth + torch.tensor([[0.1, -0.2], [0.2, 0.1]])
    accumulator.key_r2.update(0, truth, prediction)
    accumulator.key_rope_r2.update(0, truth, prediction)
    accumulator.value_r2.update(0, truth, prediction)
    accumulator.key_concentration_sum[:] = 1.2
    accumulator.key_concentration_count[:] = 1
    accumulator.value_concentration_sum[:] = 0.8
    accumulator.value_concentration_count[:] = 1
    accumulator.cosine_sum[:] = 1.8
    accumulator.cosine_count[:] = 2
    checkpoint = mechanism.save_checkpoint(
        output / "checkpoints",
        accumulator,
        completed=1,
        fingerprint=fingerprint,
        plan=plan,
    )
    summary = mechanism.summarize_accumulator(accumulator)
    result_value = {
        "schema_version": 1,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism",
        "direction": mechanism.DIRECTION,
        "request_fingerprint": fingerprint,
        "documents": 100,
        "choices_per_document": 4,
        "scoring_sequences": 1,
        "target_layers": 1,
        "metric_scope": (
            "deterministic local completion of paper section 4.5 on HellaSwag scoring tokens"
        ),
        **summary,
        "local_heldout_reconstruction": live_heldout,
        "hellaswag": live_hellaswag,
        "paper_comparison": {
            "qwen3_32b_to_14b_published_reference": None,
            "reason": "paper Table 4 omits Qwen3 32B->14B single-pair mechanism scalars",
            "published_attention_cosine_retention_pearson_r": 0.57,
            "published_calibration_r2k_retention_pearson_r": -0.20,
            "published_correlations_scope": "12 matched-KV pair-direction evaluations",
            "single_pair_correlation_reproduction": "not mathematically defined",
        },
    }
    result_path = output / mechanism.RESULT_NAME
    mechanism.hashed_json(result_path, result_value)
    result_identity = mechanism.validate_mechanism_result(
        result_path, fingerprint=fingerprint
    )
    artifact = mechanism._checkpoint_artifact
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism_receipt",
        "request_fingerprint": fingerprint,
        "result": result_identity,
        "run_definition": artifact(output / "run_definition.json"),
        "token_plan": artifact(output / "token_plan.jsonl"),
        "selection": artifact(output / "selection.json"),
        "runtime_setup": artifact(runtime_setup_path),
        "runtime_setup_sha256_sidecar": artifact(
            runtime_setup_path.with_name(f"{runtime_setup_path.name}.sha256.json")
        ),
        "final_checkpoint": artifact(checkpoint),
        "final_checkpoint_sha256_sidecar": artifact(
            checkpoint.with_name(f"{checkpoint.name}.sha256.json")
        ),
        "formal_hellaswag_receipt": {
            "path": str(upstream.resolve()),
            "sha256": mechanism.sha256_file(upstream),
            "nbytes": upstream.stat().st_size,
        },
        "local_heldout_reconstruction": {
            "path": str(heldout_path.resolve()),
            "sha256": mechanism.sha256_file(heldout_path),
            "nbytes": heldout_path.stat().st_size,
        },
        "gpu_idle_gate": artifact(output / "gpu_idle_gate.json"),
        "paper_reference_boundary": {
            "single_pair_reverse_values_published": False,
            "correlations_are_cross_pair_n12_only": True,
        },
    }
    receipt_path = output / mechanism.RECEIPT_NAME
    mechanism.hashed_json(receipt_path, receipt)
    assert mechanism.validate_completed_output(output) == receipt

    # A fully self-consistent old result cannot be adopted after the current
    # code/model/mapper/freeze definition changes.
    current_definition["code"] = {"reverse_experiment": {"sha256": "a" * 64}}
    with pytest.raises(ValueError, match="current code/model/mapper/freeze"):
        mechanism.validate_completed_output(output)
    current_definition.pop("code")
    assert mechanism.validate_completed_output(output) == receipt

    # Even if an attacker rewrites runtime_setup, its sidecar, and the formal
    # receipt together, semantic runtime replay must reject the change.
    runtime_setup["mapper"]["token_chunk"] = 2048
    mechanism.hashed_json(runtime_setup_path, runtime_setup)
    receipt["runtime_setup"] = artifact(runtime_setup_path)
    receipt["runtime_setup_sha256_sidecar"] = artifact(
        runtime_setup_path.with_name(f"{runtime_setup_path.name}.sha256.json")
    )
    mechanism.hashed_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="runtime mapper token_chunk"):
        mechanism.validate_completed_output(output)
    runtime_setup["mapper"]["token_chunk"] = 1024
    mechanism.hashed_json(runtime_setup_path, runtime_setup)
    receipt["runtime_setup"] = artifact(runtime_setup_path)
    receipt["runtime_setup_sha256_sidecar"] = artifact(
        runtime_setup_path.with_name(f"{runtime_setup_path.name}.sha256.json")
    )
    mechanism.hashed_json(receipt_path, receipt)
    assert mechanism.validate_completed_output(output) == receipt

    # Rehash the edited result and receipt, simulating a coordinated rewrite
    # rather than merely breaking a sidecar.  Replay from state_0001 must win.
    result_value["layers"][0]["key_concentration"][
        "mean_across_kv_heads"
    ] = 9.0
    result_value["headline"]["key_error_concentration"] = 9.0
    mechanism.hashed_json(result_path, result_value)
    receipt["result"] = mechanism.validate_mechanism_result(
        result_path, fingerprint=fingerprint
    )
    mechanism.hashed_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="final sufficient statistics"):
        mechanism.validate_completed_output(output)
