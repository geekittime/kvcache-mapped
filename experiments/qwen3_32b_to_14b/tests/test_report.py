from __future__ import annotations

import copy
from pathlib import Path

import pytest

from reverse_kv import coqa_eval
from reverse_kv.eval_common import hashed_json
from reverse_kv.fit_adapter import assert_reverse_paper_fit
from reverse_kv.k4_fit import assert_k4_fit
from reverse_kv.k4_protocol import K4_TRAIN_OBSERVATIONS
from reverse_kv.k4_runtime import K4_REVERSE_ARCHITECTURE
from reverse_kv.protocol import TRAIN_OBSERVATIONS
from reverse_kv.protocol import canonical_sha256, sha256_file
from reverse_kv.report import (
    COMPLETE_FILENAMES,
    PAPER_COQA_PUBLIC,
    PAPER_HELLASWAG_PUBLIC,
    ReportValidationError,
    _coqa_report_from_summary,
    _fit_architecture_view,
    _validate_selection_manifest,
    _heldout_report_from_layers,
    _validate_report_bundle,
    _validate_stage_receipt_set,
    publish_report_bundle,
    report_stage_command,
)
from reverse_kv.runtime import REVERSE_ARCHITECTURE


def _heldout_payload(k: int) -> dict:
    layers = []
    for layer in range(40):
        metric = {
            "uniform_channel_r2_fp64": 0.8 + layer / 1000,
            "global_r2_fp64": 0.81,
            "flattened_global_r2_fp64": 0.82,
            "mse_fp64": 0.1,
            "global_sse_fp64": 10.0,
            "global_sst_fp64": 100.0,
            "observations": 12_800,
            "channels": 1_024,
        }
        layers.append(
            {
                "target_layer": layer,
                "selected_source_layers_rank_order": list(range(k)),
                "key": dict(metric),
                "value": {**metric, "uniform_channel_r2_fp64": 0.7 + layer / 1000},
            }
        )
    key = [row["key"]["uniform_channel_r2_fp64"] for row in layers]
    val = [row["value"]["uniform_channel_r2_fp64"] for row in layers]
    return {
        "layers": layers,
        "summary": {
            "target_layers": 40,
            "key_uniform_channel_r2_fp64_mean_across_layers": sum(key) / 40,
            "value_uniform_channel_r2_fp64_mean_across_layers": sum(val) / 40,
            "key_uniform_channel_r2_fp64_min": min(key),
            "value_uniform_channel_r2_fp64_min": min(val),
        },
    }


def _coqa_summary() -> dict:
    depths = (1, 3, 5, 7, 10)
    target_by_depth = {}
    mapped_by_depth = {}
    drift = {}
    targets = []
    mapped = []
    for depth in depths:
        target = 0.80
        mapped_value = target - (0.001 * depth)
        targets.append(target)
        mapped.append(mapped_value)
        target_by_depth[str(depth)] = {
            "official_word_f1": target,
            "cap_hits": 0,
        }
        mapped_by_depth[str(depth)] = {
            "official_word_f1": mapped_value,
            "cap_hits": 1,
        }
        drift[str(depth)] = {
            "drift_percentage_points": (target - mapped_value) * 100.0,
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "primary_metric": "official_normalized_word_f1",
        "modes": {
            "target_standalone": {
                "official_word_f1": sum(targets) / len(targets),
                "by_handoff_depth": target_by_depth,
            },
            "mapped": {
                "official_word_f1": sum(mapped) / len(mapped),
                "by_handoff_depth": mapped_by_depth,
            },
        },
        "drift_by_handoff_depth": drift,
    }


def _mapper(variant: str, k: int, source_features: int, gib: float) -> dict:
    return {
        "variant": variant,
        "k": k,
        "source_features": source_features,
        "target_features": 1024,
        "weight_count": 1,
        "bias_count": 1,
        "fp32_parameter_gib": gib,
        "train_key_uniform_channel_r2_mean_across_layers": 0.8,
        "train_value_uniform_channel_r2_mean_across_layers": 0.7,
        "normal_equation_relative_residual_max": 1e-6,
    }


def _formal_report_fixture() -> dict:
    coqa20 = _coqa_report_from_summary(
        _coqa_summary(), variant="production_k20", k=20
    )
    coqa4 = _coqa_report_from_summary(
        _coqa_summary(), variant="figure4_retuned_k4", k=4
    )
    held20 = _heldout_report_from_layers(
        _heldout_payload(20), variant="production_k20", expected_k=20
    )
    held4 = _heldout_report_from_layers(
        _heldout_payload(4), variant="figure4_retuned_k4", expected_k=4
    )
    selections = [
        {
            "variant": "production_k20",
            "rows": [list(range(20)) for _ in range(40)],
        },
        {
            "variant": "figure4_retuned_k4",
            "rows": [list(range(4)) for _ in range(40)],
        },
    ]
    hella = {
        "sample_count_per_mode": 10_042,
        "target_standalone": {
            "correct": 7_900,
            "acc_norm": 7_900 / 10_042,
            "acc_norm_percent": 7_900 / 10_042 * 100,
        },
        "mapped": {
            "correct": 7_700,
            "acc_norm": 7_700 / 10_042,
            "acc_norm_percent": 7_700 / 10_042 * 100,
        },
        "mapped_over_target_retention_percent": 7_700 / 7_900 * 100,
        "paper_public_values": dict(PAPER_HELLASWAG_PUBLIC),
    }
    mechanism = {
        "hellaswag_eval_domain_r2_k_content": 0.7,
        "hellaswag_eval_domain_r2_k_rope_cache": 0.69,
        "hellaswag_eval_domain_r2_v": 0.8,
        "key_error_concentration": 1.1,
        "value_error_concentration": 1.0,
        "attention_output_cosine": 0.9,
    }
    latency_rows = []
    for length in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768):
        target = float(length) / 10
        mapper = float(length) / 20
        latency_rows.append(
            {
                "prefix_length": length,
                "local_target_reprefill_median_ms": target,
                "local_mapper_median_ms": mapper,
                "local_speedup_reprefill_over_mapper": target / mapper,
                "paper_h100_reference": (
                    {
                        "target_reprefill_ms": 39.2,
                        "mapper_ms": 11.6,
                        "speedup_reprefill_over_mapper": 39.2 / 11.6,
                    }
                    if length == 64
                    else None
                ),
            }
        )
    source_inventory = {"fixture": True}
    return {
        "schema_version": 1,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_formal_reproduction_report",
        "direction": "qwen3_32b_to_14b",
        "source_inventory": source_inventory,
        "source_inventory_sha256": canonical_sha256(source_inventory),
        "selection": selections,
        "mappers": [
            _mapper("production_k20", 20, 20_480, 6.25),
            _mapper("figure4_retuned_k4", 4, 4_096, 1.25),
        ],
        "heldout_activation_reconstruction": [held20, held4],
        "hellaswag": hella,
        "mechanism": mechanism,
        "coqa": {
            "variants": [coqa20, coqa4],
            "paper_public_values": dict(PAPER_COQA_PUBLIC),
        },
        "latency": {"rows": latency_rows},
    }


def test_heldout_recomputed_from_layers_rejects_summary_tamper() -> None:
    value = _heldout_payload(20)
    result = _heldout_report_from_layers(
        value, variant="production_k20", expected_k=20
    )
    assert result["key_uniform_channel_r2_fp64_mean_across_layers"] > 0.8
    changed = copy.deepcopy(value)
    changed["summary"]["key_uniform_channel_r2_fp64_mean_across_layers"] += 0.01
    with pytest.raises(ReportValidationError, match="differs from raw"):
        _heldout_report_from_layers(
            changed, variant="production_k20", expected_k=20
        )


def test_coqa_slope_is_derived_from_depth_rows_and_tamper_closes() -> None:
    summary = _coqa_summary()
    result = _coqa_report_from_summary(
        summary, variant="production_k20", k=20
    )
    assert result["drift_linear_slope_pp_per_turn_over_depths_1_3_5_7_10"] == pytest.approx(
        0.1
    )
    summary["drift_by_handoff_depth"]["7"]["drift_percentage_points"] += 1
    with pytest.raises(ReportValidationError, match="depth drift"):
        _coqa_report_from_summary(summary, variant="production_k20", k=20)


def test_stage_receipt_revalidates_bound_artifact_and_detects_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    artifact = root / "formal.json"
    artifact.write_text('{"status":"complete"}\n', encoding="utf-8")
    receipt_root = root / "receipts"
    receipt_root.mkdir()
    row = {
        "path": str(artifact.resolve()),
        "relative_path": "formal.json",
        "sha256": sha256_file(artifact),
        "nbytes": artifact.stat().st_size,
    }
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "stage": "stage_a",
        "definition_digest": "a" * 64,
        "command": ["python"],
        "execution": "executed",
        "artifacts": [row],
        "artifacts_sha256": canonical_sha256([row]),
    }
    hashed_json(receipt_root / "stage_a.json", receipt)
    validated = _validate_stage_receipt_set(
        receipt_root,
        {"stage_a": (artifact,)},
        experiment_root=root,
    )
    assert validated["pipeline_definition_digest"] == "a" * 64
    artifact.write_text('{"status":"changed"}\n', encoding="utf-8")
    with pytest.raises(ReportValidationError, match="changed after receipt"):
        _validate_stage_receipt_set(
            receipt_root,
            {"stage_a": (artifact,)},
            experiment_root=root,
        )


def test_atomic_report_bundle_exact_inventory_and_payload_tamper(
    tmp_path: Path,
) -> None:
    report = _formal_report_fixture()
    output = tmp_path / "formal_report"
    artifacts = publish_report_bundle(output, report)
    assert {path.name for path in output.iterdir()} == set(COMPLETE_FILENAMES)
    assert len(artifacts) == len(COMPLETE_FILENAMES)
    # A valid existing output is immutable and may be adopted.
    assert publish_report_bundle(output, report) == artifacts
    path = output / "latency.csv"
    path.write_bytes(path.read_bytes() + b"tamper\n")
    with pytest.raises(ReportValidationError, match="derived artifact changed"):
        _validate_report_bundle(output, report)


def test_report_manifest_and_receipt_sidecar_tamper_close(tmp_path: Path) -> None:
    report = _formal_report_fixture()
    output = tmp_path / "formal_report"
    publish_report_bundle(output, report)
    manifest = output / "sha256_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ReportValidationError, match="manifest changed"):
        _validate_report_bundle(output, report)

    output2 = tmp_path / "formal_report_2"
    publish_report_bundle(output2, report)
    sidecar = output2 / "formal_report_receipt.json.sha256.json"
    sidecar.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA sidecar mismatch"):
        _validate_report_bundle(output2, report)


def test_paper_boundary_leaves_undisclosed_reverse_values_null() -> None:
    assert PAPER_HELLASWAG_PUBLIC[
        "qwen3_32b_to_14b_mapped_acc_norm_percent"
    ] is None
    assert PAPER_COQA_PUBLIC["exact_qwen3_32b_to_14b_k20_curve_points"] is None
    assert PAPER_COQA_PUBLIC["exact_qwen3_32b_to_14b_k4_curve_points"] is None


def test_report_pipeline_command_is_fixed_cpu_stage(tmp_path: Path) -> None:
    command = report_stage_command(tmp_path / "python")
    assert command[0] == str((tmp_path / "python").resolve())
    assert command[1].endswith("experiments/qwen3_32b_to_14b/scripts/build_report.py")
    assert "CUDA" not in " ".join(command)


def test_report_adapts_verified_runtime_architectures_for_payload_audit() -> None:
    k20 = _fit_architecture_view(
        REVERSE_ARCHITECTURE,
        n_samples=TRAIN_OBSERVATIONS,
    )
    assert_reverse_paper_fit(k20, 0.01)
    k4 = _fit_architecture_view(
        K4_REVERSE_ARCHITECTURE,
        n_samples=K4_TRAIN_OBSERVATIONS,
    )
    assert_k4_fit(k4, 0.01)


def test_report_accepts_the_protocols_one_based_coqa_domain_ranks() -> None:
    rows = []
    for domain in coqa_eval.PAPER_DOMAINS:
        for rank in range(1, 21):
            rows.append(
                {
                    "selection_index": len(rows),
                    "domain": domain,
                    "rank_within_domain": rank,
                }
            )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "paper_bit_exact_claimed": False,
        "selection_uses_answers_or_gold": False,
        "rows": rows,
        "rows_sha256": canonical_sha256(rows),
    }
    assert set(_validate_selection_manifest(manifest)) == set(range(100))

    changed = copy.deepcopy(manifest)
    changed["rows"][0]["rank_within_domain"] = 0
    changed["rows_sha256"] = canonical_sha256(changed["rows"])
    with pytest.raises(ReportValidationError, match="20 rows per paper domain"):
        _validate_selection_manifest(changed)
