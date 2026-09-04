from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import pytest

import reverse_kv.coqa_eval as coqa_eval
import reverse_kv.coqa_k4_eval as coqa_k4_eval
import reverse_kv.heldout as heldout
import reverse_kv.k4_runtime as k4_runtime
import reverse_kv.runtime as reverse_runtime
from reverse_kv.eval_common import hashed_json, hashed_jsonl
from reverse_kv.protocol import (
    atomic_write_json,
    canonical_sha256,
    read_json,
    sha256_file,
)


def _heldout_metric(value: float) -> dict[str, object]:
    per_channel_sse = [1.0 - value] * 1_024
    per_channel_sst = [1.0] * 1_024
    sse = sum(per_channel_sse)
    sst = sum(per_channel_sst)
    return {
        "uniform_channel_r2_fp64": value,
        "global_r2_fp64": value,
        "flattened_global_r2_fp64": value,
        "mse_fp64": sse / (12_800 * 1_024),
        "global_sse_fp64": sse,
        "global_sst_fp64": sst,
        "per_channel_sse_fp64": per_channel_sse,
        "per_channel_sst_fp64": per_channel_sst,
        "flattened_global_sst_fp64": sst,
        "observations": 12_800,
        "channels": 1_024,
        "seconds": 0.1,
    }


@pytest.mark.parametrize("tamper", ("selection", "metric"))
def test_k20_heldout_rejects_coordinated_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tamper: str
) -> None:
    mapper_root = tmp_path / "mapper"
    mapper_root.mkdir()
    rows = [
        [int((layer + offset) % 64) for offset in range(20)]
        for layer in range(40)
    ]
    atomic_write_json(
        mapper_root / "manifest.json",
        {"selection": {"selected_source_layers_rank_order": rows}},
    )
    manifest_sha = sha256_file(mapper_root / "manifest.json")
    fingerprint = "f" * 64
    destination = tmp_path / "heldout.json"
    training_binding = {
        "manifest_sha256": manifest_sha,
        "fit_fingerprint": fingerprint,
    }
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "direction": "qwen3_32b_to_14b",
        "mapper_training_binding": training_binding,
    }
    layers = [
        {
            "target_layer": layer,
            "selected_source_layers_rank_order": rows[layer],
            "key": _heldout_metric(0.5 + layer / 1_000),
            "value": _heldout_metric(0.4 + layer / 1_000),
        }
        for layer in range(40)
    ]
    key = [float(row["key"]["uniform_channel_r2_fp64"]) for row in layers]
    value = [float(row["value"]["uniform_channel_r2_fp64"]) for row in layers]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "metric_scope": "held-out activation reconstruction; not a benchmark score",
        "preflight": preflight,
        "mapper": {
            "manifest": str(mapper_root / "manifest.json"),
            "manifest_sha256": manifest_sha,
            "fit_fingerprint": fingerprint,
        },
        "compute_device": "cuda:0",
        "analysis_device": "cpu",
        "chunk_samples": 1_024,
        "layers": layers,
        "summary": {
            "target_layers": 40,
            "key_uniform_channel_r2_fp64_mean_across_layers": sum(key) / 40,
            "value_uniform_channel_r2_fp64_mean_across_layers": sum(value) / 40,
            "key_uniform_channel_r2_fp64_min": min(key),
            "value_uniform_channel_r2_fp64_min": min(value),
        },
        "acceptance": {
            "status": "passed",
            "rule": "complete_finite_all_40_layers_K_and_V_with_exact_bindings",
            "threshold_applied": False,
            "reason_no_numeric_threshold": (
                "paper does not disclose a held-out activation threshold"
            ),
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
    bundle = SimpleNamespace(
        manifest_path=mapper_root / "manifest.json",
        manifest_sha256=manifest_sha,
        fit_fingerprint=fingerprint,
        architecture=SimpleNamespace(
            source_layers=64,
            target_layers=40,
            source_features=20_480,
            target_features=1_024,
            selected_source_layers=20,
        ),
    )
    monkeypatch.setattr(heldout, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(
        heldout,
        "validate_config",
        lambda: {
            "outputs": {
                "mapper": mapper_root,
                "heldout_validation": destination,
            }
        },
    )
    monkeypatch.setattr(heldout, "heldout_preflight", lambda **_: preflight)
    monkeypatch.setattr(heldout.ReverseMapperBundle, "open", lambda _: bundle)
    assert heldout.validate_heldout_artifact(destination) == payload

    # Rewrite the scientific payload and its digest/outer checksum together.
    # The validator must still recover the authoritative row from the mapper.
    if tamper == "selection":
        payload["layers"][9]["selected_source_layers_rank_order"] = list(
            reversed(rows[9])
        )
        error = "selection changed"
    else:
        payload["layers"][9]["key"]["uniform_channel_r2_fp64"] += 0.01
        changed_key = [
            float(row["key"]["uniform_channel_r2_fp64"])
            for row in payload["layers"]
        ]
        payload["summary"][
            "key_uniform_channel_r2_fp64_mean_across_layers"
        ] = sum(changed_key) / 40
        payload["summary"]["key_uniform_channel_r2_fp64_min"] = min(changed_key)
        error = "uniform R2 changed"
    payload["acceptance"]["raw_layers_digest"] = canonical_sha256(payload["layers"])
    atomic_write_json(destination, payload)
    atomic_write_json(
        destination.with_suffix(".sha256.json"),
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    with pytest.raises(ValueError, match=error):
        heldout.validate_heldout_artifact(destination)


class _CharacterTokenizer:
    eos_token_id = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(value) for value in token_ids)


def _coqa_configuration(tmp_path: Path, mapper_root: Path) -> dict[str, object]:
    return {
        "dataset": {"file": tmp_path / "coqa.parquet"},
        "selection": {
            "namespace": "qwen3_32b_to_14b_coqa_v1",
            "ranking": "ascending sha256(namespace + NUL + conversation_id)",
            "minimum_turns": 10,
            "conversations_per_domain": 20,
        },
        "prompt": {"frozen": "prompt"},
        "generation": {
            "max_new_tokens": 128,
            "stop_sequences": ["\nQ:"],
        },
        "metric": {"frozen": "metric"},
        "runtime": {
            "source_devices": [0, 1, 2, 3, 4],
            "source_layer_counts": [12, 13, 14, 13, 12],
            "target_devices": [5, 6, 7],
            "target_layer_counts": [13, 14, 13],
            "attention_implementation": "flash_attention_2",
            "mapper_token_chunk": 1_024,
            "max_context_tokens": 32_768,
        },
        "paths": {
            "source_model": tmp_path / "Qwen3-32B",
            "target_model": tmp_path / "Qwen3-14B",
            "mapper_manifest": mapper_root / "manifest.json",
        },
    }


def _coqa_conversations() -> list[coqa_eval.CoQAConversation]:
    result = []
    row_index = 0
    for domain in coqa_eval.PAPER_DOMAINS:
        for index in range(20):
            result.append(
                coqa_eval.CoQAConversation(
                    row_index=row_index,
                    conversation_id=f"{domain}-{index:02d}",
                    domain=domain,
                    story=f"Story {domain} {index}.",
                    questions=tuple(f"Question {turn}?" for turn in range(1, 11)),
                    primary_answers=tuple("blue" for _ in range(10)),
                    references=tuple(("blue",) for _ in range(10)),
                    raw_domain=domain,
                )
            )
            row_index += 1
    return result


def _sharding(devices: list[int], counts: list[int], role: str) -> dict[str, object]:
    layer_devices = [device for device, count in zip(devices, counts, strict=True) for _ in range(count)]
    ranges = []
    start = 0
    for device, count in zip(devices, counts, strict=True):
        ranges.append(
            {
                "device": device,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "count": count,
            }
        )
        start += count
    auxiliary = {
        "model.embed_tokens": {
            "expected_device": devices[0],
            "actual_device": devices[0],
            "parameter_tensors_inspected": 1,
        },
        "model.norm": {
            "expected_device": devices[-1],
            "actual_device": devices[-1],
            "parameter_tensors_inspected": 1,
        },
        "lm_head": {
            "expected_device": devices[-1],
            "actual_device": devices[-1],
            "parameter_tensors_inspected": 1,
        },
    }
    return {
        "status": "passed",
        "role": role,
        "mode": "explicit_exact_counts",
        "devices": devices,
        "declared_layer_counts": counts,
        "expected_layer_counts": counts,
        "actual_layer_counts": counts,
        "total_layers": sum(counts),
        "layer_devices": layer_devices,
        "contiguous_ranges": ranges,
        "actual_parameters_inspected_for_every_layer": True,
        "parameter_tensors_inspected": sum(counts),
        "auxiliary_modules": auxiliary,
        "auxiliary_parameter_tensors_inspected": 3,
        "auxiliary_parameters_inspected_for_every_module": True,
    }


def _build_complete_coqa_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, k4: bool = False
) -> tuple[
    Path,
    list[dict[str, object]],
    dict[str, object],
    coqa_eval.CoQARunVariant,
    dict[str, object],
]:
    output = tmp_path / "run"
    output.mkdir()
    mapper_root = tmp_path / "mapper"
    mapper_root.mkdir()
    mapper_manifest: dict[str, object] = {
        "architecture": {"selected_source_layers": 4 if k4 else 20}
    }
    if k4:
        mapper_manifest["curve_variant"] = "figure4_large_to_small_retuned_k4"
    atomic_write_json(mapper_root / "manifest.json", mapper_manifest)
    configuration = _coqa_configuration(tmp_path, mapper_root)
    conversations = _coqa_conversations()
    parquet_schema = {
        "rows": 100,
        "columns": ["id", "source", "story", "questions", "answers"],
        "schema": "test-schema",
        "schema_sha256": "1" * 64,
    }
    dataset_identity = {
        "provider": "test",
        "repository": "test/coqa",
        "revision": "frozen",
        "split": "validation",
        "path": str((tmp_path / "coqa.parquet").resolve()),
        "sha256": "2" * 64,
        "nbytes": 123,
        "identity_scope": "full_parquet_payload_sha256",
    }
    selected, selection_manifest = coqa_eval.select_conversations(
        conversations, configuration
    )
    receipt = {
        "schema_version": 1,
        "status": "ready",
        "dataset": dataset_identity,
        "parquet": parquet_schema,
        "selection_manifest_sha256": canonical_sha256(selection_manifest),
    }
    hashed_json(output / "dataset_receipt.json", receipt)
    hashed_json(output / "selection_manifest.json", selection_manifest)

    tokenizer_sha = "3" * 64
    source_identity = {
        "path": str(tmp_path / "Qwen3-32B"),
        "content_hashed_files": {"tokenizer.json": {"sha256": tokenizer_sha}},
        "weight_shards": [],
    }
    target_identity = {
        "path": str(tmp_path / "Qwen3-14B"),
        "content_hashed_files": {"tokenizer.json": {"sha256": tokenizer_sha}},
        "weight_shards": [],
    }
    mapper_identity = {
        "root": str(mapper_root),
        "manifest": str(mapper_root / "manifest.json"),
        "manifest_sha256": sha256_file(mapper_root / "manifest.json"),
        "checksum": str(mapper_root / "manifest.sha256.json"),
        "checksum_sha256": "4" * 64,
        "fit_fingerprint": "5" * 64,
        "target_layer_count": 40,
    }
    acceptance: dict[str, object] = {
        "status": "passed",
        "kind": "mandatory_pre_CoQA_k4_mapper_acceptance_gate",
        "mapper_manifest_sha256": mapper_identity["manifest_sha256"],
        "fit_fingerprint": mapper_identity["fit_fingerprint"],
    }
    variant = (
        coqa_k4_eval.k4_variant(acceptance)
        if k4
        else coqa_eval.DEFAULT_RUN_VARIANT
    )
    monkeypatch.setattr(coqa_eval, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(coqa_eval, "verify_dataset_file", lambda _: dataset_identity)
    monkeypatch.setattr(
        coqa_eval, "load_conversations", lambda _: (conversations, parquet_schema)
    )
    monkeypatch.setattr(coqa_eval, "_load_validation_tokenizer", lambda _: _CharacterTokenizer())
    monkeypatch.setattr(
        coqa_eval,
        "model_artifact_identity",
        lambda path: source_identity if "32B" in str(path) else target_identity,
    )
    monkeypatch.setattr(coqa_eval, "mapper_manifest_identity", lambda _: mapper_identity)
    monkeypatch.setattr(
        coqa_eval,
        "code_identity",
        lambda paths, **kwargs: {
            "paths": list(paths),
            "root": str(kwargs.get("root", coqa_eval.EXPERIMENT_ROOT)),
        },
    )
    run_definition, _, _ = coqa_eval._build_run_definition(
        configuration,
        output,
        receipt,
        selection_manifest,
        variant,
    )
    hashed_json(output / "run_definition.json", run_definition)
    fingerprint = str(run_definition["request_fingerprint"])

    capture_source = {
        "model_path": str(tmp_path / "Qwen3-32B"),
        "model_config_sha256": "6" * 64,
        "model_tokenizer_sha256": tokenizer_sha,
        "model_weight_index_sha256": "7" * 64,
        "model_name": "qwen3_32b",
    }
    capture_target = {
        "model_path": str(tmp_path / "Qwen3-14B"),
        "model_config_sha256": "8" * 64,
        "model_tokenizer_sha256": tokenizer_sha,
        "model_weight_index_sha256": "9" * 64,
        "model_name": "qwen3_14b",
    }
    bundle = SimpleNamespace(
        manifest_path=mapper_root / "manifest.json",
        manifest_sha256=mapper_identity["manifest_sha256"],
        fit_fingerprint=mapper_identity["fit_fingerprint"],
        source_model_identity=capture_source,
        target_model_identity=capture_target,
    )
    if k4:
        monkeypatch.setattr(k4_runtime.K4ReverseMapperBundle, "open", lambda _: bundle)
    else:
        monkeypatch.setattr(reverse_runtime.ReverseMapperBundle, "open", lambda _: bundle)
    target_devices = [
        f"cuda:{device}"
        for device, count in zip([5, 6, 7], [13, 14, 13], strict=True)
        for _ in range(count)
    ]
    mapper_runtime = {
        "direction": "qwen3_32b_to_14b",
        "manifest": str(mapper_root / "manifest.json"),
        "manifest_sha256": mapper_identity["manifest_sha256"],
        "fit_fingerprint": mapper_identity["fit_fingerprint"],
        "layers": 40,
        "selected_source_layers": 4 if k4 else 20,
        "token_chunk": 1_024,
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
        "target_layer_devices": target_devices,
        "resident_parameter_bytes": (
            2 * 40 * ((4 if k4 else 20) * 8 * 128 * 1_024 + 1_024) * 4
        ),
        "materialization": {
            "scope": "one_time_setup_excluded_from_per_request_latency",
            "host_load_and_hash": {},
            "weight_host_to_device": {},
            "per_layer": [
                {
                    "target_layer": layer,
                    "device": target_devices[layer],
                    "host_load_and_hash_ms": 0.1,
                    "weight_host_to_device_ms": 0.1,
                }
                for layer in range(40)
            ],
        },
        "source_model_artifact": {
            field: capture_source[field]
            for field in (
                "model_path",
                "model_config_sha256",
                "model_tokenizer_sha256",
                "model_weight_index_sha256",
            )
        },
        "target_model_artifact": {
            field: capture_target[field]
            for field in (
                "model_path",
                "model_config_sha256",
                "model_tokenizer_sha256",
                "model_weight_index_sha256",
            )
        },
        "source_capture_model_identity": capture_source,
        "target_capture_model_identity": capture_target,
    }
    if k4:
        mapper_runtime.update(
            {
                "curve_variant": "figure4_large_to_small_retuned_k4",
                "source_features": 4_096,
                "target_features": 1_024,
                "independently_refit_not_k20_slice": True,
            }
        )
    runtime_setup = {
        "schema_version": 1,
        "status": "ready",
        "request_fingerprint": fingerprint,
        "semantic_role_binding": {
            "source": {"model": "Qwen3-32B", "capture_native_forward_role": "target"},
            "target": {"model": "Qwen3-14B", "capture_native_forward_role": "source"},
        },
        "source_model_sharding": _sharding([0, 1, 2, 3, 4], [12, 13, 14, 13, 12], "target"),
        "target_model_sharding": _sharding([5, 6, 7], [13, 14, 13], "source"),
        "mapper": mapper_runtime,
        "cache_execution": {
            "parent_cache_scope": "one_persistent_parent_per_conversation_and_mode",
            "initial_full_history_prefills": 1,
            "subsequent_history_updates": "incremental_token_deltas_only",
            "headline_handoff_depths": list(coqa_eval.HEADLINE_DEPTHS),
            "mapped_parent_model": "Qwen3-32B",
            "target_standalone_parent_model": "Qwen3-14B",
            "decode_branch": "deep_clone_then_discard",
        },
    }
    hashed_json(output / "runtime_setup.json", runtime_setup)

    tokenizer = _CharacterTokenizer()
    token_rows, plans_by_index = coqa_eval._expected_token_plans(
        selected, tokenizer, configuration
    )
    token_sidecar = hashed_jsonl(output / "token_plans.jsonl", token_rows)
    records: list[dict[str, object]] = []
    checkpoints = []
    for mode in coqa_eval.MODES:
        for item in selected:
            plans = plans_by_index[item.selection_index]
            for depth in coqa_eval.HEADLINE_DEPTHS:
                plan = plans[depth - 1]
                expected = coqa_eval._expected_record_identity(
                    item=item,
                    plan=plan,
                    all_plans=plans,
                    mode=mode,
                    fingerprint=fingerprint,
                    variant=variant,
                )
                token_ids = [ord(character) for character in "blue"]
                record: dict[str, object] = {
                    "schema_version": 1,
                    "status": "complete",
                    **copy.deepcopy(expected),
                    "references": list(item.conversation.references[depth - 1]),
                    "prediction": "blue",
                    "relaxed_prediction": "blue",
                    "official_word_f1": 1.0,
                    "official_word_f1_percent": 100.0,
                    "format_relaxed_word_f1": 1.0,
                    "format_relaxed_word_f1_percent": 100.0,
                    "generation": {
                        "generated_token_ids": token_ids,
                        "generated_token_count": len(token_ids),
                        "stop_reason": "eos",
                        "cap_hit": False,
                        "requested_max_tokens": 128,
                        "original_context_tokens": len(plan.prompt_token_ids),
                        "used_context_tokens": len(plan.prompt_token_ids),
                        "truncation": "error",
                    },
                }
                path = coqa_eval._record_path(output, mode, item, depth)
                hashed_json(path, record)
                records.append(record)
                checkpoints.append(
                    {
                        "path": str(path.relative_to(output)),
                        "sha256": sha256_file(path),
                        "nbytes": path.stat().st_size,
                    }
                )
    raw_sidecar = hashed_jsonl(output / "raw_records.jsonl", records)
    summary_sidecar = hashed_json(output / "summary.json", coqa_eval.summarize_records(records))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": fingerprint,
        "direction": "qwen3_32b_to_14b",
        "task": "coqa_multi_turn_handoff",
        "paper_bit_exact_claimed": False,
        "headline_handoff_depths": list(coqa_eval.HEADLINE_DEPTHS),
        "records": 1_000,
        "expected_records": 1_000,
        "run_definition": {
            "sha256": sha256_file(output / "run_definition.json"),
            "nbytes": (output / "run_definition.json").stat().st_size,
        },
        "dataset_receipt": {
            "sha256": sha256_file(output / "dataset_receipt.json"),
            "nbytes": (output / "dataset_receipt.json").stat().st_size,
        },
        "selection_manifest": {
            "sha256": sha256_file(output / "selection_manifest.json"),
            "nbytes": (output / "selection_manifest.json").stat().st_size,
        },
        "runtime_setup": {
            "sha256": sha256_file(output / "runtime_setup.json"),
            "nbytes": (output / "runtime_setup.json").stat().st_size,
        },
        "token_plans": token_sidecar,
        "raw_records": raw_sidecar,
        "summary": summary_sidecar,
        "checkpoint_manifest_sha256": canonical_sha256(checkpoints),
        "checkpoints": checkpoints,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if k4:
        manifest.update(
            {
                "curve_variant": "figure4_large_to_small_retuned_k4",
                "selected_source_layers": 4,
                "prerequisite_acceptance": acceptance,
            }
        )
    hashed_json(output / "manifest.json", manifest)
    return output, records, configuration, variant, acceptance


def test_coqa_completed_output_rejects_coordinated_checkpoint_chain_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, records, configuration, _, _ = _build_complete_coqa_output(
        tmp_path, monkeypatch
    )
    manifest = coqa_eval.validate_completed_output(
        output, configuration=configuration
    )
    assert manifest is not None and manifest["records"] == 1_000

    # Change a cache-history fact, then consistently rewrite checkpoint SHA,
    # raw JSONL, summary link, checkpoint inventory, and final manifest SHA.
    changed: list[tuple[int, Path, dict[str, object]]] = []
    # Mutate the paired target/mapped rows together so even the aggregate's
    # target-vs-mapped history comparison remains internally consistent.
    for index in (0, 500):
        path = output / str(manifest["checkpoints"][index]["path"])
        record = read_json(path)
        record["cache_execution"]["parent_token_count_before_handoff"] += 1
        hashed_json(path, record)
        records[index] = record
        changed.append((index, path, record))
    raw_sidecar = hashed_jsonl(output / "raw_records.jsonl", records)
    summary_sidecar = hashed_json(output / "summary.json", coqa_eval.summarize_records(records))
    mutated = read_json(output / "manifest.json")
    for index, path, _ in changed:
        mutated["checkpoints"][index] = {
            "path": str(path.relative_to(output)),
            "sha256": sha256_file(path),
            "nbytes": path.stat().st_size,
        }
    mutated["checkpoint_manifest_sha256"] = canonical_sha256(mutated["checkpoints"])
    mutated["raw_records"] = raw_sidecar
    mutated["summary"] = summary_sidecar
    hashed_json(output / "manifest.json", mutated)
    with pytest.raises(ValueError, match="invalid CoQA checkpoint"):
        coqa_eval.validate_completed_output(output, configuration=configuration)


def test_k4_coqa_reuses_strong_variant_validator_and_rejects_generation_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, records, configuration, variant, acceptance = _build_complete_coqa_output(
        tmp_path, monkeypatch, k4=True
    )
    manifest = coqa_eval.validate_completed_output(
        output,
        configuration=configuration,
        variant=variant,
        mapper_builder_class=k4_runtime.K4ReverseMappedPrefixCacheBuilder,
        mapper_bundle_class=k4_runtime.K4ReverseMapperBundle,
    )
    assert manifest is not None and manifest["selected_source_layers"] == 4

    mapper_root = Path(configuration["paths"]["mapper_manifest"]).parent
    mapper_identity = coqa_eval.mapper_manifest_identity(mapper_root / "manifest.json")
    monkeypatch.setattr(coqa_k4_eval, "EXPERIMENT_ROOT", tmp_path)
    monkeypatch.setattr(coqa_k4_eval, "K4_MAPPER_ROOT", mapper_root)
    monkeypatch.setattr(
        coqa_k4_eval, "load_k4_coqa_config", lambda: configuration
    )
    monkeypatch.setattr(
        coqa_k4_eval,
        "heldout_acceptance_identity",
        lambda _: copy.deepcopy(acceptance),
    )
    monkeypatch.setattr(
        coqa_k4_eval, "mapper_manifest_identity", lambda _: mapper_identity
    )
    monkeypatch.setattr(
        coqa_k4_eval,
        "code_identity",
        lambda paths, **kwargs: {
            "paths": list(paths),
            "root": str(kwargs.get("root", coqa_eval.EXPERIMENT_ROOT)),
        },
    )
    assert coqa_k4_eval.validate_k4_coqa_output(output)["mode_counts"] == {
        "target_standalone": 500,
        "mapped": 500,
    }

    # Rewrite the generated-token evidence, checkpoint/sidecar, raw JSONL,
    # summary link, checkpoint inventory and manifest together.  The legacy
    # aggregate-only validator accepted this because prediction/F1 stayed
    # unchanged; the shared strong validator must decode the tokens and fail.
    changed_path = output / str(manifest["checkpoints"][0]["path"])
    changed_record = read_json(changed_path)
    changed_record["generation"]["generated_token_ids"] = [
        ord(character) for character in "green"
    ]
    changed_record["generation"]["generated_token_count"] = 5
    hashed_json(changed_path, changed_record)
    records[0] = changed_record
    raw_sidecar = hashed_jsonl(output / "raw_records.jsonl", records)
    summary_sidecar = hashed_json(
        output / "summary.json", coqa_eval.summarize_records(records)
    )
    mutated_manifest = read_json(output / "manifest.json")
    mutated_manifest["checkpoints"][0] = {
        "path": str(changed_path.relative_to(output)),
        "sha256": sha256_file(changed_path),
        "nbytes": changed_path.stat().st_size,
    }
    mutated_manifest["checkpoint_manifest_sha256"] = canonical_sha256(
        mutated_manifest["checkpoints"]
    )
    mutated_manifest["raw_records"] = raw_sidecar
    mutated_manifest["summary"] = summary_sidecar
    hashed_json(output / "manifest.json", mutated_manifest)
    with pytest.raises(ValueError, match="invalid CoQA checkpoint"):
        coqa_k4_eval.validate_k4_coqa_output(output)
