#!/usr/bin/env python3
"""Prepare byte-frozen HotpotQA-train/MuSiQue-train method-dev inputs.

This command is CPU-only.  ``--dry-run`` performs every read, hash, selection,
overlap, tokenizer, prompt-length, and output-isolation check but writes
nothing.  ``--prepare`` writes only below this candidate's ``artifacts/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
if str(CANDIDATE_ROOT) not in sys.path:
    sys.path.insert(0, str(CANDIDATE_ROOT))

from src.method_dev import (  # noqa: E402
    DATASET_IDS,
    DEFAULT_CONFIG,
    audit_cross_dataset_overlap,
    canonical_json_bytes,
    canonical_sha256,
    ensure_output_within,
    jsonl_bytes,
    load_and_validate_config,
    load_hotpot_candidates,
    load_musique_candidates,
    phase_counts,
    render_and_tokenize_samples,
    select_preregistered_samples,
    sha256_file,
    write_immutable,
)


def _tokenizer_identity(source: Any, target: Any) -> dict[str, Any]:
    source_vocab = source.get_vocab()
    target_vocab = target.get_vocab()
    if source_vocab != target_vocab:
        raise ValueError("source and target tokenizer vocabularies differ")
    fields = ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")
    source_special = {name: getattr(source, name, None) for name in fields}
    target_special = {name: getattr(target, name, None) for name in fields}
    if source_special != target_special:
        raise ValueError("source and target tokenizer special IDs differ")
    source_template = getattr(source, "chat_template", None)
    target_template = getattr(target, "chat_template", None)
    if source_template != target_template or not isinstance(source_template, str):
        raise ValueError("source/target Qwen chat templates differ or are missing")
    return {
        "source_name_or_path": str(getattr(source, "name_or_path", "")),
        "target_name_or_path": str(getattr(target, "name_or_path", "")),
        "vocabulary_size": len(source_vocab),
        "vocabulary_sha256": canonical_sha256(sorted(source_vocab.items())),
        "special_token_ids": source_special,
        "chat_template_sha256": hashlib.sha256(
            source_template.encode("utf-8")
        ).hexdigest(),
        "template_arguments": {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
    }


def _candidate_code_inventory() -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(CANDIDATE_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(CANDIDATE_ROOT)
        if relative.parts[0] in {"artifacts", "runs", "logs"}:
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[str(relative)] = {
            "sha256": sha256_file(path),
            "nbytes": path.stat().st_size,
        }
    return files


def build_prepared_payloads(config_path: Path) -> dict[str, bytes]:
    # Importing transformers is intentionally delayed so basic CLI/help and
    # unit tests do not pay its startup cost.
    from transformers import AutoTokenizer

    config = load_and_validate_config(config_path)
    datasets = config["datasets"]
    hotpot, hotpot_rows = load_hotpot_candidates(datasets["hotpotqa_train"]["path"])
    musique, musique_rows = load_musique_candidates(datasets["musique_train"]["path"])
    observed_rows = {
        "hotpotqa_train": hotpot_rows,
        "musique_train": musique_rows,
    }
    for dataset_id, count in observed_rows.items():
        expected = int(datasets[dataset_id]["rows"])
        if count != expected:
            raise ValueError(f"{dataset_id} rows {count} != pinned {expected}")

    selection = config["selection"]
    selected: list[dict[str, Any]] = []
    for dataset_id, candidates in (
        ("hotpotqa_train", hotpot),
        ("musique_train", musique),
    ):
        selected.extend(
            select_preregistered_samples(
                candidates,
                seed=str(selection["rank_seed"]),
                per_phase=selection["per_stratum"][dataset_id],
            )
        )
    expected_strata = {
        "hotpotqa_train": {
            "bridge/easy",
            "bridge/medium",
            "bridge/hard",
            "comparison/easy",
            "comparison/medium",
            "comparison/hard",
        },
        "musique_train": {"2hop", "3hop", "4hop"},
    }
    for dataset_id, expected in expected_strata.items():
        observed = {
            str(row["stratum"])
            for row in selected
            if row["dataset_id"] == dataset_id
        }
        if observed != expected:
            raise ValueError(f"{dataset_id} strata changed: {sorted(observed)}")

    overlap = audit_cross_dataset_overlap(selected)
    if overlap["status"] != "passed":
        raise ValueError("selected method-dev rows failed cross-dataset overlap audit")

    source_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["source_32b"]["path"],
        local_files_only=True,
        use_fast=True,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["target_14b"]["path"],
        local_files_only=True,
        use_fast=True,
    )
    tokenizer_identity = _tokenizer_identity(source_tokenizer, target_tokenizer)
    prompt = config["prompt"]
    selected, length_summary = render_and_tokenize_samples(
        selected,
        tokenizer=target_tokenizer,
        template=str(prompt["completion_template"]),
        max_length=int(prompt["max_length"]),
        max_generation_tokens=int(prompt["max_generation_tokens"]),
    )
    counts = phase_counts(selected)
    if counts != {
        "smoke": {"hotpotqa_train": 6, "musique_train": 3},
        "pilot": {"hotpotqa_train": 48, "musique_train": 48},
        "expansion": {"hotpotqa_train": 192, "musique_train": 192},
    }:
        raise ValueError(f"prepared phase counts changed: {counts}")

    identities = {
        dataset_id: [
            [row["sample_id"], row["source_record_sha256"]]
            for row in selected
            if row["dataset_id"] == dataset_id
        ]
        for dataset_id in DATASET_IDS
    }
    registry = {
        "schema_version": 1,
        "tier": "method-dev",
        "protocol_id": config["protocol_id"],
        "split_namespace": selection["split_namespace"],
        "split_salt_commitment": selection["split_salt_commitment"],
        "split_unit": "complete question/document group before tokenization",
        "token_level_random_split": False,
        "selection_uses_answers_or_model_metrics": False,
        "datasets": {
            dataset_id: {
                **dict(datasets[dataset_id]),
                "selected_expansion_rows": len(identities[dataset_id]),
                "selected_id_and_record_sha256_digest": canonical_sha256(
                    identities[dataset_id]
                ),
                "selected_ids_sha256": canonical_sha256(
                    [value[0] for value in identities[dataset_id]]
                ),
            }
            for dataset_id in DATASET_IDS
        },
        "tokenizer": tokenizer_identity,
        "prompt_template_sha256": hashlib.sha256(
            str(prompt["completion_template"]).encode("utf-8")
        ).hexdigest(),
        "musique_protocol_binding": config["musique_protocol_binding"],
        "phase_counts": counts,
        "cross_dataset_overlap_audit": overlap,
        "locked_test_files_opened": [],
        "locked_test_metrics_observed": False,
        "mapper_fit_cross_tier_registry_audit": {
            "status": "not_claimed_by_this_candidate",
            "reason": (
                "the project-wide mapper-fit registry must be validated by the "
                "00_research_protocol gate before any nomination; this local "
                "prepare step proves only its two declared training inputs"
            ),
        },
    }
    preflight = {
        "schema_version": 1,
        "status": "passed_method_dev_local_preflight",
        "gpu_used": False,
        "cuda_api_called": False,
        "protocol_id": config["protocol_id"],
        "checks": {
            "config_and_model_config_hashes": "passed",
            "baseline_mapper_manifest_hash_read_only": "passed",
            "official_training_split_paths_and_hashes": "passed",
            "locked_test_path_access": "zero",
            "deterministic_sequence_id_selection": "passed",
            "cross_dataset_exact_and_near_question_overlap": "passed",
            "source_target_tokenizer_identity": "passed",
            "shared_shortest_answer_instruction_for_both_datasets": "passed",
            "musique_all_paragraphs_official_idx_ascending": "passed",
            "musique_config_sha256_binding": config["musique_protocol_binding"]["config"]["sha256"],
            "musique_registry_manifest_sha256_binding": config["musique_protocol_binding"]["registry_manifest"]["sha256"],
            "generation_cap_and_stop_rules_match_musique_locked_protocol": "passed_256_tokens_and_two_stop_sequences",
            "prompt_no_truncation": "passed",
            "every_prompt_supports_R32_with_mapped_prefix": "passed",
            "R1_R8_R32_builder_tail_counts": "0_7_31",
            "engine_owned_boundary_count": 1,
            "candidate_output_isolation": "passed",
        },
        "prompt_token_lengths": length_summary,
        "phase_counts": counts,
        "scientific_boundary": (
            "HotpotQA official train and MuSiQue answerable official train only, "
            "using one shared shortest-answer/no-explanation instruction; "
            "no HotpotQA dev/test, MuSiQue dev/test, or prior formal result is an input"
        ),
        "global_G0_complete": False,
        "global_G0_pending_item": (
            "project-wide mapper-fit/method-dev/locked-test exact and near-duplicate registry"
        ),
    }
    selected_payload = jsonl_bytes(selected)
    registry_payload = canonical_json_bytes(registry) + b"\n"
    preflight_payload = canonical_json_bytes(preflight) + b"\n"
    interface_path = CANDIDATE_ROOT / "src/tail_refresh.py"
    interface = {
        "schema_version": 1,
        "status": "frozen_method_dev_interface",
        "module": "src.tail_refresh",
        "implementation": {
            "path": str(interface_path.resolve()),
            "sha256": sha256_file(interface_path),
            "nbytes": interface_path.stat().st_size,
        },
        "builder_class": "TailRefreshMappedPrefixCacheBuilder",
        "constructor": {
            "required_keywords": [
                "source_model",
                "target_model",
                "mapper_manifest",
                "replay_tokens",
            ],
            "optional_keywords": {"token_chunk": 1024},
            "replay_tokens_allowed": [1, 8, 32],
        },
        "shared_mapper_apis": [
            "TailRefreshMappedPrefixCacheBuilder.from_shared_mapper(verified_reverse_builder, replay_tokens=R)",
            "existing_tail_builder.variant(R)",
        ],
        "accepted_underlying_type": "reverse_kv.runtime.ReverseMappedPrefixCacheBuilder",
        "causal_contract": {
            "builder_input": "C[:-1]",
            "builder_input_length_symbol": "P=len(C)-1",
            "mapped_prefix": "C[:P-(R-1)]",
            "builder_native_tail": "C[P-(R-1):P]",
            "builder_native_tail_count": "R-1",
            "engine_boundary": "C[-1]",
            "engine_boundary_count": 1,
            "total_native_prompt_tokens": "(R-1)+1=R",
        },
        "r1_contract": {
            "target_forward_calls": 0,
            "cache_clone_or_cast": False,
            "returns_underlying_cache_object_directly": True,
        },
        "last_call_audit_required_fields": [
            "builder_prefix_tokens",
            "full_prompt_tokens",
            "requested_total_native_prompt_tokens",
            "mapped_prefix_tokens",
            "builder_native_tail_tokens",
            "engine_boundary_tokens",
            "native_tail_position_ids",
            "engine_boundary_position_id",
            "target_forward_calls",
            "target_forward_query_tokens",
            "cache_length_before_refresh",
            "cache_length_after_refresh",
            "r1_returned_underlying_cache_object_directly",
        ],
        "hybrid_audit_additional_fields": [
            "attention_mask_shape",
            "position_ids",
            "cache_position",
        ],
    }
    interface_payload = canonical_json_bytes(interface) + b"\n"
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "protocol_id": config["protocol_id"],
        "config": {
            "path": config["config_path"],
            "sha256": config["config_sha256"],
        },
        "files": {
            "selected_samples.jsonl": {
                "sha256": hashlib.sha256(selected_payload).hexdigest(),
                "nbytes": len(selected_payload),
                "rows": len(selected),
            },
            "data_registry.json": {
                "sha256": hashlib.sha256(registry_payload).hexdigest(),
                "nbytes": len(registry_payload),
            },
            "preflight.json": {
                "sha256": hashlib.sha256(preflight_payload).hexdigest(),
                "nbytes": len(preflight_payload),
            },
            "interface_receipt.json": {
                "sha256": hashlib.sha256(interface_payload).hexdigest(),
                "nbytes": len(interface_payload),
            },
        },
        "selection_digest": canonical_sha256(
            [
                [
                    row["dataset_id"],
                    row["sample_id"],
                    row["stratum"],
                    row["selection_rank_sha256"],
                    row["phase_membership"],
                    row["prompt_token_ids_sha256"],
                ]
                for row in selected
            ]
        ),
        "phase_counts": counts,
        "candidate_code_inventory": _candidate_code_inventory(),
    }
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    return {
        "selected_samples.jsonl": selected_payload,
        "data_registry.json": registry_payload,
        "preflight.json": preflight_payload,
        "interface_receipt.json": interface_payload,
        "manifest.json": manifest_payload,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payloads = build_prepared_payloads(args.config)
    config = load_and_validate_config(args.config)
    prepared_root = ensure_output_within(
        CANDIDATE_ROOT / "artifacts", config["outputs"]["prepared"]
    )
    summary = {
        "status": "dry_run_passed" if args.dry_run else "prepared",
        "gpu_used": False,
        "output": str(prepared_root),
        "files": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "nbytes": len(payload),
            }
            for name, payload in payloads.items()
        },
    }
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    for name, payload in payloads.items():
        write_immutable(prepared_root / name, payload)
    receipt_path = prepared_root / "prepare_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("config_sha256") != config["config_sha256"]
            or receipt.get("files") != summary["files"]
        ):
            raise FileExistsError("existing prepare receipt differs from current inputs")
    else:
        receipt = {
            **summary,
            "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_sha256": config["config_sha256"],
        }
        write_immutable(
            receipt_path,
            canonical_json_bytes(receipt) + b"\n",
        )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
