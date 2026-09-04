import copy
import hashlib
import json
from pathlib import Path

import pytest

import src.configuration as configuration
from src.checkpoints import result_path
from src.checkpoints import expected_r32_builder_audit
from src.contracts import (
    ContractError,
    artifact_ref,
    atomic_json,
    canonical_sha256,
    code_inventory,
    immutable_json,
    immutable_jsonl,
    read_json,
    read_jsonl,
    sha256_file,
    token_ids_sha256,
    write_sidecar,
)
from src.metrics import score_answer
from src.gpu_runner import (
    ATTEMPT_EVIDENCE_FILENAMES,
    _validate_deep_runtime_validation,
    seal_raw_completion_receipt,
    validate_raw_completion_receipt,
)
from src.gpu_runtime import validate_method_bindings
from src.run_protocol import (
    _validate_screen_receipt_for_confirm,
    build_run_definition,
    freeze_run_definition,
    seal_completed_run,
)
from src.validator import validate_completed_run, write_validation_receipt


ROOT = Path(__file__).resolve().parents[1]


class _Tokenizer:
    _prospective_frozen_eos_token_ids = (2,)

    def decode(self, ids, skip_special_tokens=True):
        return "answer" if list(ids) == [17] else ""


def _ref(path, content="x\n"):
    path.write_text(content, encoding="utf-8")
    return artifact_ref(path).to_dict()


def _formal_config(tmp_path, monkeypatch):
    config = json.loads((ROOT / "config.template.json").read_text(encoding="utf-8"))
    config["status"] = "frozen_ready"
    config["runnable"] = True
    config["profiles"] = {
        "smoke": {"hotpotqa_train": 2, "musique_train": 2},
        "screen": {"hotpotqa_train": 2, "musique_train": 2},
        "confirm": {"hotpotqa_train": 2, "musique_train": 2},
    }
    dummy = _ref(tmp_path / "pinned.txt")
    split_counts = {
        "hotpotqa_screen": 2,
        "hotpotqa_confirm": 2,
        "musique_screen": 2,
        "musique_confirm": 2,
    }
    strata = config["registry"]["strata_order"]
    splits = {}
    for key, count in split_counts.items():
        prefix, partition = key.rsplit("_", 1)
        dataset = f"{prefix}_train"
        base_index = 1000 if partition == "confirm" else 0
        rows = [
            {
                "dataset_id": dataset,
                "partition": partition,
                "sample_id": f"{key}-{index}",
                "source_index": base_index + index,
                "source_projection_sha256": f"{index + 1:064x}",
                "selection_rank": index,
                "selection_rank_sha256": f"{index + 2:064x}",
                "stratum": strata[dataset][0],
            }
            for index in range(count)
        ]
        path = tmp_path / f"{key}.jsonl"
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        splits[key] = artifact_ref(path).to_dict()
    config["registry"]["split_artifacts"] = splits
    config["registry"]["registry_manifest"] = _ref(
        tmp_path / "registry_manifest.json",
        json.dumps({"status": "prepared_decontaminated_method_dev_only"}) + "\n",
    )
    config["registry"]["publication_receipt"] = _ref(
        tmp_path / "receipt.json",
        json.dumps(
            {
                "status": "pass",
                "artifact_publication": {
                    "completion_marker_required": "PUBLISHED.json",
                    "status_pass_receipt_without_completion_marker_is_not_published": True,
                },
            }
        )
        + "\n",
    )
    quarantine = _ref(
        tmp_path / "quarantine_aggregate.json",
        json.dumps({"status": "pass", "flags": 0}) + "\n",
    )
    published_refs = [
        config["registry"]["registry_manifest"],
        config["registry"]["publication_receipt"],
        quarantine,
        *splits.values(),
    ]
    sidecar_refs = []
    for ref in published_refs:
        core = Path(ref["path"])
        sidecar_refs.append(
            _ref(
                core.with_name(core.name + ".sha256.json"),
                json.dumps(
                    {
                        "path": ref["path"],
                        "nbytes": ref["nbytes"],
                        "sha256": ref["sha256"],
                    },
                    sort_keys=True,
                )
                + "\n",
            )
        )
    published_files = {
        Path(ref["path"]).name: {"nbytes": ref["nbytes"], "sha256": ref["sha256"]}
        for ref in [*published_refs, *sidecar_refs]
    }
    inventory = [{"name": name, **published_files[name]} for name in sorted(published_files)]
    inventory_payload = (
        json.dumps(inventory, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    config["registry"]["published_marker"] = _ref(
        tmp_path / "PUBLISHED.json",
        json.dumps(
            {
                "schema_version": 1,
                "registry_id": "fixture",
                "status": "published_complete",
                "atomic_last_completion_marker": True,
                "published_files": published_files,
                "ordered_inventory_sha256": hashlib.sha256(inventory_payload).hexdigest(),
                "receipt": published_files["receipt.json"],
                "receipt_sidecar": published_files["receipt.json.sha256.json"],
            }
        )
        + "\n",
    )
    registry_binding = {
        "published_marker": config["registry"]["published_marker"],
        "registry_manifest": config["registry"]["registry_manifest"],
        "publication_receipt": config["registry"]["publication_receipt"],
        "split_artifacts": splits,
    }
    g0 = {
        "status": "complete",
        "pass": True,
        "closes_complete_g0": True,
        "scope": "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval",
        "locked_outcomes_opened": False,
        "candidate_registry_binding": registry_binding,
        "comparisons": {
            name: {"flags": 0}
            for name in (
                "methoddev_hotpot_vs_mapper_fit",
                "methoddev_musique_vs_mapper_fit",
                "methoddev_hotpot_vs_locked_eval",
                "methoddev_musique_vs_locked_eval",
            )
        },
    }
    g0_path = tmp_path / "g0.json"
    g0_path.write_text(json.dumps(g0, sort_keys=True) + "\n", encoding="utf-8")
    config["g0"]["completion_receipt"] = artifact_ref(g0_path).to_dict()
    for dataset in config["train_sources"].values():
        dataset["artifact"] = dummy
    config["metrics"]["hotpotqa_reference"] = dummy
    config["metrics"]["musique_reference"] = dummy
    for name in ("config", "method", "protocol", "receipt"):
        config["hypothesis_provenance"][name] = dummy
    config["design_reference_inventory"] = dummy
    config["outputs"] = {
        "prepared_root": str((tmp_path / "prepared").resolve()),
        "runs_root": str((tmp_path / "runs").resolve()),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setattr(configuration, "ROOT", tmp_path)
    return config_path, config


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("required_python_dont_write_bytecode", False),
        ("required_python_pycache_prefix", "/tmp/other-pycache-prefix"),
    ),
)
def test_formal_config_rejects_bytecode_policy_drift(
    tmp_path, monkeypatch, field, replacement
):
    config_path, config = _formal_config(tmp_path, monkeypatch)
    config["runtime"][field] = replacement
    config_path.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="runtime/topology contract changed"):
        configuration.load_config(config_path, verify_inputs=False)


def _prepared(config_path, config):
    rows = []
    for split_key, split_ref in config["registry"]["split_artifacts"].items():
        for selected in read_jsonl(split_ref["path"]):
            body = {
                "schema_version": 2,
                "tier": "method_dev_prospective",
                "dataset_id": selected["dataset_id"],
                "partition": selected["partition"],
                "sample_id": selected["sample_id"],
                "source_index": selected["source_index"],
                "stratum": selected["stratum"],
                "stratum_selection_rank": selected["selection_rank"],
                "selection_rank_sha256": selected["selection_rank_sha256"],
                "evaluation_rank": selected["selection_rank"],
                "source_projection_sha256": selected["source_projection_sha256"],
                "references": ["answer"],
                "prompt_token_ids": list(range(40)),
                "prompt_token_count": 40,
                "prompt_token_ids_sha256": token_ids_sha256(list(range(40))),
                "truncation": {
                    "policy": "head_tail",
                    "truncated": False,
                    "original_tokens": 40,
                    "used_tokens": 40,
                    "removed_tokens": 0,
                    "head_tokens": 40,
                    "tail_tokens": 0,
                },
                "generation_cap": 256,
                "stop_sequences": ["\nQuestion:", "\nContext:"],
                "no_think": True,
            }
            rows.append({**body, "sample_binding_sha256": canonical_sha256(body)})
    prepared = Path(config["outputs"]["prepared_root"])
    samples_ref = immutable_jsonl(prepared / "samples.jsonl", rows)
    write_sidecar(samples_ref)
    config_ref = artifact_ref(config_path).to_dict()
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "protocol_id": config["protocol_id"],
        "config": config_ref,
        "registry": {
            "published_marker": config["registry"]["published_marker"],
            "registry_manifest": config["registry"]["registry_manifest"],
            "publication_receipt": config["registry"]["publication_receipt"],
            "split_artifacts": config["registry"]["split_artifacts"],
        },
        "g0_completion_receipt": config["g0"]["completion_receipt"],
        "g0_receipt_content_sha256": canonical_sha256(
            json.loads(Path(config["g0"]["completion_receipt"]["path"]).read_text(encoding="utf-8"))
        ),
        "tokenizer_identity": {
            "source_name_or_path": config["runtime"]["source_model"],
            "target_name_or_path": config["runtime"]["target_model"],
            "vocabulary_size": 10,
            "vocabulary_sha256": "a" * 64,
            "special_token_ids": {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "pad_token_id": None,
                "unk_token_id": None,
            },
            "chat_template_sha256": "b" * 64,
            "template_arguments": {
                "tokenize": True,
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        },
        "samples": samples_ref.to_dict(),
        "sample_count": len(rows),
        "counts": {
            f"{dataset}/{partition}": config["profiles"][partition][dataset]
            for dataset in ("hotpotqa_train", "musique_train")
            for partition in ("screen", "confirm")
        },
        "sample_bindings_sha256": canonical_sha256(
            [row["sample_binding_sha256"] for row in rows]
        ),
        "answers_dereferenced_after_registry_freeze": True,
        "official_dev_or_test_opened": False,
        "code_inventory": code_inventory(),
    }
    manifest_ref = immutable_json(prepared / "manifest.json", manifest)
    write_sidecar(manifest_ref)
    profile_rows = {}
    for profile in ("smoke", "screen", "confirm"):
        partition = "screen" if profile in {"smoke", "screen"} else "confirm"
        selected_rows = []
        for dataset in ("hotpotqa_train", "musique_train"):
            matching = [
                row
                for row in rows
                if row["dataset_id"] == dataset and row["partition"] == partition
            ]
            selected_rows.extend(matching[: config["profiles"][profile][dataset]])
        profile_rows[profile] = selected_rows
    completion_ref = immutable_json(
        prepared / "completion_receipt.json",
        {
            "schema_version": 2,
            "status": "complete",
            "runnable_source_config_had_no_placeholders": True,
            "registry_publication_state": "published_complete",
            "complete_g0_pass": True,
            "samples": samples_ref.to_dict(),
            "manifest": manifest_ref.to_dict(),
            "profiles": {
                profile: {
                    "count": len(values),
                    "sample_bindings_sha256": canonical_sha256(
                        [row["sample_binding_sha256"] for row in values]
                    ),
                }
                for profile, values in profile_rows.items()
            },
            "official_dev_or_test_opened": False,
        },
    )
    write_sidecar(completion_ref)
    return profile_rows["smoke"]


def _raw_value(row, mode, definition, *, native_sha=None, baseline_sha=None):
    ids = [17]
    prediction = "answer"
    value = {
        "schema_version": 2,
        "status": "complete",
        "result_kind": "raw_model_run",
        "request_fingerprint": definition["request_fingerprint"],
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "sample_binding_sha256": row["sample_binding_sha256"],
        "source_projection_sha256": row["source_projection_sha256"],
        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        "mode": mode,
        "method_binding_sha256": definition["method_binding_sha256"][mode],
        "execution_attempt_id": "attempt_0001",
        "references": row["references"],
        "truncation": row["truncation"],
        "prediction": prediction,
        "generation": {
            "requested_max_tokens": 256,
            "generated_tokens": 1,
            "generated_token_ids": ids,
            "generated_token_ids_sha256": token_ids_sha256(ids),
            "decoded_generated_token_ids": prediction,
            "stop_reason": "eos",
            "terminal_token_id": 2,
            "frozen_eos_token_ids": [2],
            "stop_sequences": row["stop_sequences"],
            "cap_hit": False,
        },
        "metrics": score_answer(row["dataset_id"], prediction, row["references"]),
        "runtime_cost": {
            "wall_seconds": 1.0,
            "phase_timing": {
                "available": False,
                "prefill_seconds": None,
                "decode_seconds": None,
                "scope": "monolithic_engine_generate_token_ids",
            },
            "peak_gpu_allocated_bytes": {
                str(index): 100 + index for index in range(8)
            },
        },
        "builder_audit": None,
        "comparison_bindings": {
            "native14_raw_file_sha256": None if mode == "native14" else native_sha,
            "nvidia_k20_raw_file_sha256": baseline_sha if mode in {"shrinkage", "tail_refresh_r32"} else None,
        },
    }
    if mode == "tail_refresh_r32":
        value["builder_audit"] = expected_r32_builder_audit(row["prompt_token_count"])
    return value


def _write_raw_grid(output, rows, definition):
    for row in rows:
        native_path = result_path(output, row, "native14")
        native_ref = immutable_json(native_path, _raw_value(row, "native14", definition))
        write_sidecar(native_ref)
        baseline_path = result_path(output, row, "nvidia_k20")
        baseline_ref = immutable_json(
            baseline_path,
            _raw_value(row, "nvidia_k20", definition, native_sha=native_ref.sha256),
        )
        write_sidecar(baseline_ref)
        for mode in ("shrinkage", "tail_refresh_r32"):
            ref = immutable_json(
                result_path(output, row, mode),
                _raw_value(
                    row,
                    mode,
                    definition,
                    native_sha=native_ref.sha256,
                    baseline_sha=baseline_ref.sha256,
                ),
            )
            write_sidecar(ref)


def _sharding(devices, counts, *, total, role):
    layer_devices = [
        device
        for device, count in zip(devices, counts, strict=True)
        for _ in range(count)
    ]
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
    return {
        "status": "passed",
        "role": role,
        "mode": "explicit_exact_counts",
        "devices": devices,
        "declared_layer_counts": counts,
        "expected_layer_counts": counts,
        "actual_layer_counts": counts,
        "total_layers": total,
        "layer_devices": layer_devices,
        "contiguous_ranges": ranges,
        "actual_parameters_inspected_for_every_layer": True,
        "parameter_tensors_inspected": total,
        "auxiliary_modules": {},
        "auxiliary_parameter_tensors_inspected": 3,
        "auxiliary_parameters_inspected_for_every_module": True,
    }


def _write_mock_gpu_completion(config, output, rows, definition):
    attempt = output / "gpu_attempts" / "attempt_0001"
    contract = config["runtime"]["idle_gate"]
    samples = [
        {
            "sample_index": index,
            "elapsed_seconds": float(index),
            "memory_used_mib": {str(device): 0 for device in range(8)},
            "compute_process_count": 0,
            "idle": True,
            "consecutive_idle": index + 1,
        }
        for index in range(contract["consecutive_samples"])
    ]
    gate_ref = immutable_json(
        attempt / "gpu_idle_gate.json",
        {
            "schema_version": 2,
            "status": "passed",
            "probe": "nvidia-smi_without_CUDA_runtime",
            "gpu_used_by_gate": False,
            "physical_gpu_indices": list(range(8)),
            "contract": contract,
            "samples": samples,
            "final_consecutive_idle_samples": contract["consecutive_samples"],
            "elapsed_seconds": float(contract["consecutive_samples"]),
        },
    )
    write_sidecar(gate_ref)
    bindings = validate_method_bindings(config, deep_model_hash=False)
    frozen_runtime = read_json(bindings.runtime["receipt"]["path"])["runtime"]
    expected_mapper_payloads = {
        mode: {**dict(evidence), "deep_hash_verified": True}
        for mode, evidence in bindings.mapper_payload_validation.items()
    }
    deep_ref = immutable_json(
        attempt / "deep_runtime_validation.json",
        {
            "schema_version": 2,
            "status": "passed_before_cuda_import",
            "deep_model_hash_verified": True,
            "runtime_environment_verified": True,
            "runtime_binding_digest": bindings.runtime["binding_digest"],
            "validated_method_bindings_digest": bindings.digest,
            "validated_mode_digests": dict(bindings.mode_digests),
            "runtime_provenance_receipt": bindings.runtime["receipt"],
            "source_model_validation": bindings.runtime["source_model"],
            "target_model_validation": bindings.runtime["target_model"],
            "mapper_manifest": bindings.runtime["mapper_manifest"],
            "runtime_code": bindings.runtime["runtime_code"],
            "runtime_environment": {
                "python": frozen_runtime["python"],
                "distributions": frozen_runtime["distributions"],
                "python_dont_write_bytecode": True,
                "python_pycache_prefix": config["runtime"][
                    "required_python_pycache_prefix"
                ],
                "python_pycache_prefix_absent": True,
            },
            "mapper_payload_validation": expected_mapper_payloads,
        },
    )
    write_sidecar(deep_ref)
    cuda_ref = immutable_json(
        attempt / "cuda_runtime_validation.json",
        {
            "schema_version": 2,
            "status": "passed_before_model_load",
            "torch_version": frozen_runtime["torch"],
            "torch_cuda_version": frozen_runtime["torch_cuda"],
            "cuda_available": True,
            "visible_cuda_device_count": 8,
            "cuda_visible_devices": config["runtime"]["required_cuda_visible_devices"],
            "validated_before_model_load": True,
        },
    )
    write_sidecar(cuda_ref)
    baseline = config["methods"]["nvidia_k20"]["mapper_artifact"]
    shrinkage = config["methods"]["shrinkage"]["mapper_artifact"]
    provenance_ref = immutable_json(
        attempt / "method_provenance.json",
        {
            "native14": {"method_id": config["methods"]["native14"]["method_id"]},
            "nvidia_k20": {
                "direction": config["direction"],
                "manifest": baseline["path"],
                "manifest_sha256": baseline["sha256"],
                "layers": 40,
                "selected_source_layers": 20,
                "token_chunk": 1024,
            },
            "shrinkage": {
                "direction": config["direction"],
                "method": read_json(shrinkage["path"])["method"],
                "manifest": shrinkage["path"],
                "manifest_sha256": shrinkage["sha256"],
                "layers": 40,
                "selected_source_layers": 20,
                "token_chunk": 1024,
            },
            "tail_refresh_r32": {
                "method": config["methods"]["tail_refresh_r32"]["method_id"],
                "manifest": baseline["path"],
                "manifest_sha256": baseline["sha256"],
                "replay_tokens_R": 32,
                "builder_native_tail_tokens": 31,
                "engine_boundary_tokens": 1,
                "resident_mapper_shared_across_variants": True,
            },
        },
    )
    write_sidecar(provenance_ref)
    runtime = config["runtime"]
    sharding_ref = immutable_json(
        attempt / "model_sharding.json",
        {
            "source_32b": _sharding(
                runtime["source_devices"],
                runtime["source_layer_counts"],
                total=64,
                role="target",
            ),
            "target_14b": _sharding(
                runtime["target_devices"],
                runtime["target_layer_counts"],
                total=40,
                role="source",
            ),
        },
    )
    write_sidecar(sharding_ref)
    return seal_raw_completion_receipt(
        config=config,
        output=output,
        profile="smoke",
        definition=definition,
        rows=rows,
        attempt_id="attempt_0001",
        gpu_idle_gate_ref=gate_ref,
        deep_runtime_validation_ref=deep_ref,
        cuda_runtime_validation_ref=cuda_ref,
        method_provenance_ref=provenance_ref,
        model_sharding_ref=sharding_ref,
        execution_counts={
            "completed_raw_results": len(rows) * 4,
            "generated_now": len(rows) * 4,
            "resumed_existing": 0,
        },
        tokenizer=_Tokenizer(),
    )


def test_smoke_receipts_recompute_and_semantic_tamper_is_rejected(tmp_path, monkeypatch):
    config_path, config = _formal_config(tmp_path, monkeypatch)
    rows = _prepared(config_path, config)
    output = Path(config["outputs"]["runs_root"]) / "smoke_v2"
    definition = freeze_run_definition(config_path, profile="smoke", output_dir=output)
    _write_raw_grid(output, rows, definition)
    _write_mock_gpu_completion(config, output, rows, definition)
    tokenizer = _Tokenizer()
    completion = seal_completed_run(
        config_path, profile="smoke", output_dir=output, tokenizer=tokenizer
    )
    assert completion["result_count"] == 20
    validation = validate_completed_run(
        config_path, profile="smoke", output_dir=output, tokenizer=tokenizer
    )
    assert validation["status"] == "validated_complete"
    assert validation["validator_used_gpu"] is False
    written = write_validation_receipt(
        config_path, profile="smoke", output_dir=output, tokenizer=tokenizer
    )
    assert written["validation_receipt_sha256"]

    raw_path = result_path(output, rows[0], "shrinkage")
    tampered = read_json(raw_path)
    tampered["metrics"]["official_answer_f1"] = 0.5
    raw_path.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    sidecar = raw_path.with_name(raw_path.name + ".sha256.json")
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "path": raw_path.name,
                "nbytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="raw metrics do not recompute"):
        validate_completed_run(
            config_path, profile="smoke", output_dir=output, tokenizer=tokenizer
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("python_dont_write_bytecode", False),
        ("python_pycache_prefix", "/tmp/attacker-pycache"),
        ("python_pycache_prefix_absent", False),
    ),
)
def test_deep_runtime_bytecode_policy_evidence_tamper_is_rejected(
    tmp_path, monkeypatch, field, replacement
):
    config_path, config = _formal_config(tmp_path, monkeypatch)
    rows = _prepared(config_path, config)
    output = Path(config["outputs"]["runs_root"]) / "smoke_v2"
    definition = freeze_run_definition(config_path, profile="smoke", output_dir=output)
    _write_raw_grid(output, rows, definition)
    _write_mock_gpu_completion(config, output, rows, definition)
    deep_path = output / "gpu_attempts" / "attempt_0001" / "deep_runtime_validation.json"
    tampered = read_json(deep_path)
    tampered["runtime_environment"][field] = replacement
    with pytest.raises(ContractError, match="deep pre-CUDA runtime validation evidence changed"):
        _validate_deep_runtime_validation(tampered, config, definition)


def test_zero_new_cell_resume_seals_final_attempt_and_binds_prior_contributor(
    tmp_path, monkeypatch
):
    config_path, config = _formal_config(tmp_path, monkeypatch)
    rows = _prepared(config_path, config)
    output = Path(config["outputs"]["runs_root"]) / "smoke_v2"
    definition = freeze_run_definition(
        config_path, profile="smoke", output_dir=output
    )
    _write_raw_grid(output, rows, definition)
    unexpected = output / "results" / "raw" / "unexpected.json"
    atomic_json(unexpected, {"status": "unexpected"})
    with pytest.raises(ContractError, match="missing/unexpected/failure JSON"):
        _write_mock_gpu_completion(config, output, rows, definition)
    assert not (output / "raw_completion_receipt.json").exists()
    unexpected.unlink()
    _write_mock_gpu_completion(config, output, rows, definition)

    # Simulate SIGKILL after the raw receipt's atomic JSON rename but before
    # its sidecar rename.  Recovery must perform the full cell/tree/evidence
    # validation before publishing the missing sidecar.
    raw_sidecar = output / "raw_completion_receipt.json.sha256.json"
    raw_sidecar.unlink()
    with pytest.raises(ContractError, match="missing SHA sidecar"):
        validate_raw_completion_receipt(
            config, definition, rows, output, tokenizer=_Tokenizer()
        )
    validate_raw_completion_receipt(
        config,
        definition,
        rows,
        output,
        repair_missing_sidecar=True,
        tokenizer=_Tokenizer(),
    )
    assert raw_sidecar.is_file()

    # Simulate a later invocation that finds a fully complete raw grid.  The
    # final invocation still has its own pre-CUDA/CUDA/model evidence, while
    # every raw cell remains bound to the original contributing attempt.
    (output / "raw_completion_receipt.json.sha256.json").unlink()
    (output / "raw_completion_receipt.json").unlink()
    first = output / "gpu_attempts" / "attempt_0001"
    second = output / "gpu_attempts" / "attempt_0002"
    second_refs = {}
    for field, filename in ATTEMPT_EVIDENCE_FILENAMES.items():
        reference = immutable_json(second / filename, read_json(first / filename))
        write_sidecar(reference)
        second_refs[field] = reference
    with pytest.raises(ContractError, match="differ from raw-cell attempt ledger"):
        seal_raw_completion_receipt(
            config=config,
            output=output,
            profile="smoke",
            definition=definition,
            rows=rows,
            attempt_id="attempt_0002",
            gpu_idle_gate_ref=second_refs["gpu_idle_gate"],
            deep_runtime_validation_ref=second_refs["deep_runtime_validation"],
            cuda_runtime_validation_ref=second_refs["cuda_runtime_validation"],
            method_provenance_ref=second_refs["method_provenance"],
            model_sharding_ref=second_refs["model_sharding"],
            execution_counts={
                "completed_raw_results": len(rows) * 4,
                "generated_now": 1,
                "resumed_existing": len(rows) * 4 - 1,
            },
            tokenizer=_Tokenizer(),
        )
    seal_raw_completion_receipt(
        config=config,
        output=output,
        profile="smoke",
        definition=definition,
        rows=rows,
        attempt_id="attempt_0002",
        gpu_idle_gate_ref=second_refs["gpu_idle_gate"],
        deep_runtime_validation_ref=second_refs["deep_runtime_validation"],
        cuda_runtime_validation_ref=second_refs["cuda_runtime_validation"],
        method_provenance_ref=second_refs["method_provenance"],
        model_sharding_ref=second_refs["model_sharding"],
        execution_counts={
            "completed_raw_results": len(rows) * 4,
            "generated_now": 0,
            "resumed_existing": len(rows) * 4,
        },
        tokenizer=_Tokenizer(),
    )
    receipt = validate_raw_completion_receipt(
        config, definition, rows, output, tokenizer=_Tokenizer()
    )
    assert receipt["attempt_id"] == "attempt_0002"
    assert set(receipt["contributing_attempts"]) == {"attempt_0001"}
    assert all(
        "/attempt_0002/" in receipt[field]["path"]
        for field in ATTEMPT_EVIDENCE_FILENAMES
    )

    # JSON false compares equal to integer zero in Python.  The receipt path
    # must nevertheless reject that alias for a zero-new-cell resume.
    receipt_path = output / "raw_completion_receipt.json"
    receipt_sidecar = output / "raw_completion_receipt.json.sha256.json"
    original_receipt = read_json(receipt_path)
    bool_alias = copy.deepcopy(original_receipt)
    bool_alias["execution_counts"]["generated_now"] = False
    atomic_json(receipt_path, bool_alias)
    bool_alias_ref = artifact_ref(receipt_path)
    atomic_json(
        receipt_sidecar,
        {
            "schema_version": 1,
            "path": receipt_path.name,
            "nbytes": bool_alias_ref.nbytes,
            "sha256": bool_alias_ref.sha256,
        },
    )
    with pytest.raises(ContractError, match="execution counts are malformed"):
        validate_raw_completion_receipt(
            config, definition, rows, output, tokenizer=_Tokenizer()
        )
    atomic_json(receipt_path, original_receipt)
    restored_ref = artifact_ref(receipt_path)
    atomic_json(
        receipt_sidecar,
        {
            "schema_version": 1,
            "path": receipt_path.name,
            "nbytes": restored_ref.nbytes,
            "sha256": restored_ref.sha256,
        },
    )

    # Even if an attacker updates the contributor's local sidecar, the final
    # receipt's content-addressed contributor ledger must reject the change.
    gate_path = first / "gpu_idle_gate.json"
    tampered_gate = read_json(gate_path)
    tampered_gate["status"] = "tampered"
    atomic_json(gate_path, tampered_gate)
    gate_ref = artifact_ref(gate_path)
    atomic_json(
        gate_path.with_name(gate_path.name + ".sha256.json"),
        {
            "schema_version": 1,
            "path": gate_path.name,
            "nbytes": gate_ref.nbytes,
            "sha256": gate_ref.sha256,
        },
    )
    with pytest.raises(ContractError, match="contributing-attempt ledger changed"):
        validate_raw_completion_receipt(
            config, definition, rows, output, tokenizer=_Tokenizer()
        )


def test_confirm_prerequisite_is_exact_validated_passing_screen_receipt(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="utf-8")
    runs = tmp_path / "runs"
    screen = runs / "screen_v2"
    config = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "outputs": {"runs_root": str(runs.resolve())},
        "confirmation_gate": {"rule_id": "screen_pass_then_disjoint_confirm_v2"},
    }
    serialized = {
        "schema_version": 2,
        "status": "validated_complete",
        "profile": "screen",
        "promotion_status": "pass",
        "authorizes_confirm": True,
        "config": artifact_ref(config_path).to_dict(),
        "official_dev_or_test_opened": False,
    }
    reference = immutable_json(screen / "validation_receipt.json", serialized)
    write_sidecar(reference)
    monkeypatch.setattr("src.validator.validate_completed_run", lambda *_args, **_kwargs: serialized)
    prerequisite = _validate_screen_receipt_for_confirm(config, reference.to_dict())
    assert prerequisite["validated_screen"] == serialized
    assert prerequisite["screen_validation_receipt"] == reference.to_dict()

    bad = dict(serialized)
    bad["promotion_status"] = "fail"
    bad["authorizes_confirm"] = False
    bad_ref = immutable_json(runs / "bad_screen" / "validation_receipt.json", bad)
    write_sidecar(bad_ref)
    monkeypatch.setattr("src.validator.validate_completed_run", lambda *_args, **_kwargs: bad)
    with pytest.raises(ContractError, match="promotion_status"):
        _validate_screen_receipt_for_confirm(config, bad_ref.to_dict())
