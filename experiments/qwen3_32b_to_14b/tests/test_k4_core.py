from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import reverse_kv.k4_heldout as k4_heldout
import reverse_kv.k4_fit as k4_fit
import reverse_kv.k4_pipeline as k4_pipeline
import reverse_kv.k4_selection as k4_selection
from reverse_kv import coqa_eval
from reverse_kv.coqa_k4_eval import (
    CURVE_VARIANT,
    k4_variant,
    load_k4_coqa_config,
)
from reverse_kv.k4_fit import assert_k4_fit
from reverse_kv.k4_protocol import (
    K4_BIAS_COUNT,
    K4_SOURCE_FEATURES,
    K4_TARGET_FEATURES,
    K4_TOP_K,
    K4_WEIGHT_COUNT,
    validate_k4_mapper_config,
)
from reverse_kv.k4_runtime import K4ReverseMapperBundle, K4RuntimeValidationError
from reverse_kv.protocol import atomic_write_json, canonical_sha256, sha256_file
from reverse_kv.selection_adapter import SCORE_SHAPES_DTYPES


def test_k4_dimensions_are_4096_to_1024_and_have_independent_counts() -> None:
    assert K4_TOP_K == 4
    assert K4_SOURCE_FEATURES == 4 * 8 * 128 == 4096
    assert K4_TARGET_FEATURES == 8 * 128 == 1024
    assert K4_WEIGHT_COUNT == 2 * 40 * 4096 * 1024 == 335_544_320
    assert K4_BIAS_COUNT == 2 * 40 * 1024 == 81_920


def test_k4_mapper_config_is_pinned_to_independent_outputs() -> None:
    configuration = validate_k4_mapper_config()
    assert configuration["raw"]["paper"]["selected_source_layers"] == 4
    assert configuration["outputs"]["selection"].parent.name == "k4_retuned"
    assert configuration["outputs"]["mapper"].name == "k4_retuned_lambda0.01"
    assert "k20_lambda0.01" not in str(configuration["outputs"]["mapper"])


def test_k4_fit_architecture_rejects_the_task_description_2048_typo() -> None:
    valid = SimpleNamespace(
        source_layers=64,
        target_layers=40,
        kv_heads=8,
        head_dim=128,
        selected_source_layers=4,
        n_samples=128_000,
        source_features=4096,
        target_features=1024,
        expected_weight_count=K4_WEIGHT_COUNT,
        expected_bias_count=K4_BIAS_COUNT,
    )
    assert_k4_fit(valid, 0.01)
    invalid = SimpleNamespace(**{**vars(valid), "source_features": 2048})
    with pytest.raises(ValueError, match="architecture mismatch"):
        assert_k4_fit(invalid, 0.01)


def test_k4_fit_driver_uses_only_independent_selection_and_mapper_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    class FakeImplementation:
        assert_paper_fit = staticmethod(lambda *args, **kwargs: None)
        collect_activation_inventory = staticmethod(lambda *args, **kwargs: {})
        validate_selection_provenance = staticmethod(lambda *args, **kwargs: {})
        publish_complete_manifest = staticmethod(lambda *args, **kwargs: {})
        make_run_directory = staticmethod(lambda *args, **kwargs: None)
        _strict_json = staticmethod(lambda *args, **kwargs: {})

        @staticmethod
        def main(argv):
            observed["argv"] = list(argv)
            observed["assert"] = FakeImplementation.assert_paper_fit
            return 0

    configuration = {
        "outputs": {
            "selection": k4_fit.K4_SELECTION.resolve(),
            "mapper": k4_fit.K4_MAPPER_ROOT.resolve(),
        }
    }
    monkeypatch.setattr(k4_fit, "validate_k4_mapper_config", lambda path: configuration)
    monkeypatch.setattr(k4_fit, "_load_implementation", lambda: FakeImplementation)
    original_assert = FakeImplementation.assert_paper_fit
    assert k4_fit.run_k4_fit(["--devices", "0", "--dry-run"]) == 0
    argv = observed["argv"]
    assert isinstance(argv, list)
    assert str(k4_fit.K4_SELECTION) in argv
    assert str(k4_fit.K4_MAPPER_ROOT) in argv
    assert all("k20_lambda0.01" not in str(value) for value in argv)
    assert observed["assert"] is k4_fit.assert_k4_fit
    assert FakeImplementation.assert_paper_fit is original_assert


def test_top4_is_recomputed_over_all_64_scores_with_lower_layer_tie_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment = tmp_path / "experiment"
    parent_path = experiment / "selection/selected_layers.json"
    score_path = experiment / "statistics/selection_scores.npz"
    score_path.parent.mkdir(parents=True)
    arrays = {
        name: np.zeros(shape, dtype=dtype)
        for name, (shape, dtype) in SCORE_SHAPES_DTYPES.items()
    }
    # The combined score is reconstructed from the head tensors.  Source 63 is
    # deliberately outside any hypothetical first-20-index truncation.
    for target in range(40):
        arrays["key_head_uniform_r2"][target, 63, :] = 10.0
        arrays["value_head_uniform_r2"][target, 63, :] = 10.0
        arrays["key_head_uniform_r2"][target, 3, :] = 9.0
        arrays["value_head_uniform_r2"][target, 3, :] = 9.0
        arrays["key_head_uniform_r2"][target, 5, :] = 9.0
        arrays["value_head_uniform_r2"][target, 5, :] = 9.0
        arrays["key_head_uniform_r2"][target, 42, :] = 8.0
        arrays["value_head_uniform_r2"][target, 42, :] = 8.0
    key = arrays["key_head_uniform_r2"].mean(axis=-1)
    value = arrays["value_head_uniform_r2"].mean(axis=-1)
    arrays["key_head_averaged_uniform_r2"] = key
    arrays["value_head_averaged_uniform_r2"] = value
    arrays["combined_equal_kv_uniform_r2"] = (key + value) / 2.0
    np.savez(score_path, **arrays)
    parent_path.parent.mkdir(parents=True)
    parent = {
        "score_artifacts": {
            "npz": {"path": str(score_path), "sha256": sha256_file(score_path)}
        }
    }
    atomic_write_json(parent_path, parent)
    monkeypatch.setattr(k4_selection, "EXPERIMENT_ROOT", experiment)
    monkeypatch.setattr(
        k4_selection, "validate_reverse_selection_artifact", lambda path: parent
    )
    rows, provenance = k4_selection.recompute_top4_rows(parent_path)
    assert rows == [[63, 3, 5, 42] for _ in range(40)]
    assert provenance["ranking_scope"] == (
        "all_64_source_layers_per_target_not_top20_row_truncation"
    )


def test_k4_runtime_rejects_k20_aggregate_before_loading_any_layer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "k20"
    root.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "curve_variant": "figure4_large_to_small_retuned_k4",
        "method": "closed-form-centered-ridge-kv-cache-transfer",
        "dtype": "float32",
        "allow_tf32": False,
        "weight_count": 1_677_721_600,
        "bias_count": 81_920,
        "ridge_lambda": 0.01,
    }
    atomic_write_json(root / "manifest.json", manifest)
    atomic_write_json(
        root / "manifest.sha256.json",
        {
            "path": str(root / "manifest.json"),
            "sha256": sha256_file(root / "manifest.json"),
        },
    )
    with pytest.raises(K4RuntimeValidationError, match="weight_count"):
        K4ReverseMapperBundle.open(root)


def _metric(value: float) -> dict[str, object]:
    channel_sse = [1.0 - value] * 1_024
    channel_sst = [1.0] * 1_024
    global_sse = sum(channel_sse)
    global_sst = sum(channel_sst)
    return {
        "uniform_channel_r2_fp64": value,
        "global_r2_fp64": value,
        "flattened_global_r2_fp64": value,
        "mse_fp64": global_sse / (12_800 * 1_024),
        "global_sse_fp64": global_sse,
        "global_sst_fp64": global_sst,
        "per_channel_sse_fp64": channel_sse,
        "per_channel_sst_fp64": channel_sst,
        "flattened_global_sst_fp64": global_sst,
        "observations": 12_800,
        "channels": 1_024,
        "seconds": 0.1,
    }


def test_k4_heldout_acceptance_recomputes_summary_and_binds_every_top4_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mapper_root = tmp_path / "mapper"
    mapper_root.mkdir()
    rows = [[layer % 64, (layer + 1) % 64, (layer + 2) % 64, (layer + 3) % 64] for layer in range(40)]
    atomic_write_json(
        mapper_root / "manifest.json",
        {"selection": {"selected_source_layers_rank_order": rows}},
    )
    manifest_sha = sha256_file(mapper_root / "manifest.json")
    fit_fingerprint = "f" * 64
    training_binding = {
        "manifest_sha256": manifest_sha,
        "fit_fingerprint": fit_fingerprint,
    }
    destination = tmp_path / "heldout.json"
    layers = [
        {
            "target_layer": layer,
            "selected_source_layers_rank_order": rows[layer],
            "key": _metric(0.5 + layer / 1000),
            "value": _metric(0.4 + layer / 1000),
        }
        for layer in range(40)
    ]
    key = [float(row["key"]["uniform_channel_r2_fp64"]) for row in layers]
    value_scores = [float(row["value"]["uniform_channel_r2_fp64"]) for row in layers]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "curve_variant": "figure4_large_to_small_retuned_k4",
        "metric_scope": "held-out activation reconstruction; not a benchmark score",
        "preflight": {
            "schema_version": 1,
            "status": "passed",
            "direction": "qwen3_32b_to_14b",
            "curve_variant": "figure4_large_to_small_retuned_k4",
            "protocol": {
                "scope": "local_validation_not_disclosed_by_paper",
                "purpose": "mandatory_pre_CoQA_k4_mapper_acceptance_gate",
                "sequences": 50,
                "sequence_length": 1024,
                "stride": 4,
                "observations": 12_800,
                "selected_source_layers": 4,
                "source_features": 4096,
                "target_features": 1024,
                "mapper_compute_dtype": "float32",
                "analysis_dtype": "float64",
                "quality_threshold": None,
                "acceptance_rule": "complete_finite_all_40_layers_K_and_V_with_exact_bindings",
            },
            "source": {
                "model_name": "qwen3_32b",
                "capture_native_role": "target",
                "manifest_sha256": "a" * 64,
                "token_sha256": "b" * 64,
            },
            "target": {
                "model_name": "qwen3_14b",
                "capture_native_role": "source",
                "manifest_sha256": "c" * 64,
                "token_sha256": "b" * 64,
            },
            "train_token_sha256": "d" * 64,
            "train_heldout_tokens_differ": True,
            "payload_checksums_verified": True,
            "mapper_root": str(mapper_root),
            "mapper_training_binding": training_binding,
        },
        "mapper": {
            "manifest": str(mapper_root / "manifest.json"),
            "manifest_sha256": manifest_sha,
            "fit_fingerprint": fit_fingerprint,
        },
        "compute_device": "cuda:0",
        "analysis_device": "cpu",
        "chunk_samples": 1_024,
        "layers": layers,
        "summary": {
            "target_layers": 40,
            "key_uniform_channel_r2_fp64_mean_across_layers": sum(key) / 40,
            "value_uniform_channel_r2_fp64_mean_across_layers": sum(value_scores) / 40,
            "key_uniform_channel_r2_fp64_min": min(key),
            "value_uniform_channel_r2_fp64_min": min(value_scores),
        },
        "acceptance": {
            "status": "passed",
            "rule": "complete_finite_all_40_layers_K_and_V_with_exact_bindings",
            "threshold_applied": False,
            "reason_no_numeric_threshold": "paper does not disclose a held-out activation threshold",
            "raw_layers_digest": canonical_sha256(layers),
        },
        "elapsed_seconds": 1.0,
        "completed_utc": "2026-08-25T00:00:00+00:00",
    }
    atomic_write_json(destination, payload)
    atomic_write_json(
        destination.with_suffix(".sha256.json"),
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    monkeypatch.setattr(k4_heldout, "K4_MAPPER_ROOT", mapper_root)
    monkeypatch.setattr(k4_heldout, "K4_HELDOUT", destination)
    monkeypatch.setattr(k4_heldout, "_training_binding", lambda: training_binding)
    authoritative_preflight = copy.deepcopy(payload["preflight"])
    monkeypatch.setattr(
        k4_heldout,
        "k4_heldout_preflight",
        lambda **kwargs: copy.deepcopy(authoritative_preflight),
    )
    monkeypatch.setattr(
        k4_heldout,
        "validate_capture_pair",
        lambda **kwargs: (
            {"manifest_sha256": "a" * 64, "token_sha256": "b" * 64},
            {"manifest_sha256": "c" * 64, "token_sha256": "b" * 64},
        ),
    )
    monkeypatch.setattr(
        k4_heldout.K4ReverseMapperBundle,
        "open",
        lambda root: SimpleNamespace(
            manifest_path=mapper_root / "manifest.json",
            manifest_sha256=manifest_sha,
            fit_fingerprint=fit_fingerprint,
            architecture=SimpleNamespace(
                source_layers=64,
                target_layers=40,
                source_features=4096,
                target_features=1024,
                selected_source_layers=4,
            ),
        ),
    )
    assert k4_heldout.validate_k4_heldout_artifact(destination)["acceptance"]["status"] == "passed"

    payload["layers"][7]["selected_source_layers_rank_order"] = [60, 61, 62, 63]
    atomic_write_json(destination, payload)
    atomic_write_json(
        destination.with_suffix(".sha256.json"),
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    with pytest.raises(ValueError, match="selection changed"):
        k4_heldout.validate_k4_heldout_artifact(destination)

    # Coordinating the metric edit with the derived summary, acceptance
    # digest, and outer SHA marker must still fail against the per-channel
    # sufficient statistics.
    payload["layers"][7]["selected_source_layers_rank_order"] = rows[7]
    payload["layers"][7]["key"]["uniform_channel_r2_fp64"] += 0.01
    changed_key = [
        float(row["key"]["uniform_channel_r2_fp64"])
        for row in payload["layers"]
    ]
    payload["summary"][
        "key_uniform_channel_r2_fp64_mean_across_layers"
    ] = sum(changed_key) / 40
    payload["summary"]["key_uniform_channel_r2_fp64_min"] = min(changed_key)
    payload["acceptance"]["raw_layers_digest"] = canonical_sha256(payload["layers"])
    atomic_write_json(destination, payload)
    atomic_write_json(
        destination.with_suffix(".sha256.json"),
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    with pytest.raises(ValueError, match="uniform R2 changed"):
        k4_heldout.validate_k4_heldout_artifact(destination)

    # The preflight is recomputed from current captures/mapper; editing both
    # the copied preflight and the artifact checksum is therefore rejected.
    payload["layers"][7]["key"]["uniform_channel_r2_fp64"] -= 0.01
    payload["summary"][
        "key_uniform_channel_r2_fp64_mean_across_layers"
    ] = sum(key) / 40
    payload["summary"]["key_uniform_channel_r2_fp64_min"] = min(key)
    payload["acceptance"]["raw_layers_digest"] = canonical_sha256(payload["layers"])
    payload["preflight"]["source"]["manifest_sha256"] = "9" * 64
    atomic_write_json(destination, payload)
    atomic_write_json(
        destination.with_suffix(".sha256.json"),
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    with pytest.raises(ValueError, match="preflight differs"):
        k4_heldout.validate_k4_heldout_artifact(destination)


def _acceptance() -> dict[str, object]:
    return {
        "status": "passed",
        "kind": "mandatory_pre_CoQA_k4_mapper_acceptance_gate",
        "mapper_manifest_sha256": "a" * 64,
        "fit_fingerprint": "b" * 64,
    }


def test_default_k20_variant_preserves_the_preexisting_run_definition_fields() -> None:
    fields = coqa_eval._variant_run_fields(coqa_eval.DEFAULT_RUN_VARIANT)
    assert fields == {
        "kind": "qwen3_32b_to_14b_coqa_multi_turn_handoff",
        "paper_reference": {
            "paper": "arXiv:2608.03893v1",
            "section": "4.6",
            "paper_disclosed": (
                "100 conversations, five domains, word F1 at handoff depths "
                "1,3,5,7,10, k=20"
            ),
            "figure_evidence": (
                "official arXiv source figures/retention_vs_turns.png; "
                "x-axis Turn depth at handoff"
            ),
            "local_completion_due_to_undisclosed_details": True,
            "paper_bit_exact_claimed": False,
        },
    }
    assert coqa_eval.DEFAULT_RUN_VARIANT.code_files == coqa_eval.CODE_FILES
    coqa_eval._validate_run_variant(coqa_eval.DEFAULT_RUN_VARIANT, None)


def test_k4_coqa_config_changes_only_curve_mapper_and_prerequisite() -> None:
    configuration = load_k4_coqa_config()
    base = coqa_eval.load_coqa_config()
    for field in ("dataset", "selection", "prompt", "generation", "metric", "runtime"):
        assert configuration[field] == base[field]
    assert configuration["curve_variant"] == CURVE_VARIANT
    assert configuration["paths"]["mapper_manifest"].parent.name == (
        "k4_retuned_lambda0.01"
    )
    assert configuration["paths"]["heldout_acceptance"] == k4_heldout.K4_HELDOUT


def test_k4_variant_requires_dedicated_builder_and_rejects_k20_bundle() -> None:
    from reverse_kv.k4_runtime import K4ReverseMappedPrefixCacheBuilder

    variant = k4_variant(_acceptance())
    coqa_eval._validate_run_variant(variant, K4ReverseMappedPrefixCacheBuilder)
    with pytest.raises(ValueError, match="dedicated fail-closed k4 builder"):
        coqa_eval._validate_run_variant(variant, object)
    identity = {"manifest_sha256": "a" * 64, "fit_fingerprint": "b" * 64}
    k4_manifest = {
        "architecture": {"selected_source_layers": 4},
        "curve_variant": CURVE_VARIANT,
    }
    coqa_eval._validate_mapper_for_variant(k4_manifest, identity, variant)
    with pytest.raises(ValueError, match="top-k"):
        coqa_eval._validate_mapper_for_variant(
            {"architecture": {"selected_source_layers": 20}}, identity, variant
        )
    with pytest.raises(ValueError, match="top-k"):
        coqa_eval._validate_mapper_for_variant(
            k4_manifest, identity, coqa_eval.DEFAULT_RUN_VARIANT
        )


def test_k4_run_fields_are_isolated_and_bind_the_heldout_gate() -> None:
    acceptance = _acceptance()
    fields = coqa_eval._variant_run_fields(k4_variant(acceptance))
    assert fields["curve_variant"] == CURVE_VARIANT
    assert fields["selected_source_layers"] == 4
    assert fields["prerequisite_acceptance"] == acceptance
    assert fields["paper_reference"]["paper_undisclosed"]


def test_k4_pipeline_append_contract_is_disjoint_from_the_existing_ten_stages() -> None:
    expected = (
        "11_k4_selection",
        "12_k4_fit_target0",
        "13_k4_fit_full",
        "14_k4_heldout",
        "15_k4_coqa",
    )
    assert k4_pipeline.K4_STAGE_ORDER == expected
    commands = k4_pipeline.k4_stage_commands(Path("/usr/bin/python3"))
    assert tuple(commands) == expected
    assert tuple(k4_pipeline.k4_validator_map()) == expected
    assert commands["12_k4_fit_target0"][commands["12_k4_fit_target0"].index("--target-layers") + 1] == "0"
    assert commands["13_k4_fit_full"][commands["13_k4_fit_full"].index("--devices") + 1] == "0,1,2,3,4,5,6,7"
    flattened = [value for command in commands.values() for value in command]
    assert all("k20_lambda0.01" not in value for value in flattened)
    assert k4_pipeline.K4_PIPELINE_INTEGRATION[
        "production_k20_stage_commands_changed"
    ] is False
    assert k4_pipeline.K4_PIPELINE_INTEGRATION[
        "completion_receipt_validation_must_rerun_k4_validators"
    ] is True
    required = k4_pipeline.K4_PIPELINE_INTEGRATION["required_completion_artifacts"]
    assert str(k4_pipeline.K4_TRAINING_IDLE_GATE) in required
    assert str(k4_pipeline.K4_COQA_IDLE_GATE) in required
