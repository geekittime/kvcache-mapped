from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from src.contracts import (
    ContractError,
    ROOT,
    adapt_source,
    artifact_ref,
    canonical_sha256,
    contract_preflight,
    ensure_formal_config_path,
    ensure_inside_output_root,
    find_placeholders,
    intervention_catalog,
    read_json,
    sha256_file,
    source_projection,
    verify_ref,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def build_final_fixture(
    tmp_path: Path,
    *,
    answer_allowed: object = True,
    answer_policy_extra: bool = False,
    g0_flags: int = 0,
    outcome_field: bool = False,
    projection_mismatch: bool = False,
    receipt_projection_algorithm: str = "canonical_question_passages_v1",
    receipt_projection_excludes_answers: bool = True,
) -> Path:
    hotpot_raw = {
        "_id": "hotpot-train-0",
        "question": "Where is the answer?",
        "answer": "Paris",
        "context": [["France", ["Paris is the capital of France."]]],
    }
    musique_raw = {
        "id": "musique-train-0",
        "question": "Who wrote the work?",
        "answer": "Ada",
        "paragraphs": [{"title": "Work", "paragraph_text": "Ada wrote the work."}],
    }
    hotpot_source = tmp_path / "data" / "hotpot_train.json"
    musique_source = tmp_path / "data" / "musique_train.jsonl"
    write_json(hotpot_source, [hotpot_raw])
    write_jsonl(musique_source, [musique_raw])
    algorithm = "canonical_question_passages_v1"
    hotpot_projection = canonical_sha256(source_projection("hotpotqa", adapt_source("hotpotqa", "hotpotqa_train_v1", hotpot_raw), algorithm))
    musique_projection = canonical_sha256(source_projection("musique", adapt_source("musique", "musique_train_v1", musique_raw), algorithm))
    if projection_mismatch:
        hotpot_projection = "0" * 64

    registry_dir = tmp_path / "registry"
    screen_rows = {
        "hotpotqa_screen": [{
            "dataset": "hotpotqa", "split": "train", "partition": "screen", "tier": "method-dev",
            "source_index": 0, "selection_rank": 0, "stratum": "all", "source_projection_sha256": hotpot_projection,
        }],
        "musique_screen": [{
            "dataset": "musique", "split": "train", "partition": "screen", "tier": "method-dev",
            "source_index": 0, "selection_rank": 0, "stratum": "all", "source_projection_sha256": musique_projection,
        }],
    }
    if outcome_field:
        screen_rows["hotpotqa_screen"][0]["f1"] = 1.0
    split_paths: dict[str, Path] = {}
    for key in ("hotpotqa_screen", "hotpotqa_confirm", "musique_screen", "musique_confirm"):
        source_key = key.replace("confirm", "screen")
        rows = copy.deepcopy(screen_rows[source_key])
        for row in rows:
            row["partition"] = "screen" if key.endswith("screen") else key
        path = registry_dir / f"{key}.jsonl"
        write_jsonl(path, rows)
        split_paths[key] = path

    receipt_path = registry_dir / "receipt.json"
    answer_policy = {
        "policy_id": "train_method_dev_oracle_metrics_only",
        "allowed": answer_allowed,
        "allowed_uses": ["gold_first_token_margin", "qa_f1", "qa_em"],
        "forbidden_uses": ["selection", "ranking", "quota", "partition_assignment"],
    }
    if answer_policy_extra:
        answer_policy["selection_allowed"] = False
    write_json(receipt_path, {
        "schema_version": 1,
        "status": "PUBLISHED",
        "pass": True,
        "tier": "method-dev",
        "locked_outcomes_opened": False,
        "projection": {
            "algorithm": receipt_projection_algorithm,
            "excludes_answers": receipt_projection_excludes_answers,
        },
        "train_sources": {"hotpotqa": artifact_ref(hotpot_source), "musique": artifact_ref(musique_source)},
        "answer_access_policy": answer_policy,
    })
    manifest_path = registry_dir / "manifest.json"
    manifest_artifacts = {"receipt": artifact_ref(receipt_path), **{key: artifact_ref(path) for key, path in split_paths.items()}}
    write_json(manifest_path, {"schema_version": 1, "artifacts": manifest_artifacts})
    published_path = registry_dir / "PUBLISHED.json"
    write_json(published_path, {
        "schema_version": 1,
        "status": "PUBLISHED",
        "pass": True,
        "manifest": artifact_ref(manifest_path),
        "receipt": artifact_ref(receipt_path),
    })
    registry_refs = {
        "published": artifact_ref(published_path),
        "manifest": artifact_ref(manifest_path),
        "receipt": artifact_ref(receipt_path),
        **{key: artifact_ref(path) for key, path in split_paths.items()},
    }
    g0_path = tmp_path / "g0_v2.json"
    comparisons = {
        name: {"flags": g0_flags}
        for name in (
            "methoddev_hotpot_vs_mapper_fit",
            "methoddev_musique_vs_mapper_fit",
            "methoddev_hotpot_vs_locked_eval",
            "methoddev_musique_vs_locked_eval",
        )
    }
    nested_registry_binding = {
        "published_marker": registry_refs["published"],
        "registry_manifest": registry_refs["manifest"],
        "publication_receipt": registry_refs["receipt"],
        "split_artifacts": {key: registry_refs[key] for key in split_paths},
    }
    write_json(g0_path, {
        "schema_version": 2,
        "status": "complete",
        "pass": True,
        "closes_complete_g0": True,
        "scope": "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval",
        "locked_outcomes_opened": False,
        "comparisons": comparisons,
        "candidate_registry_binding": nested_registry_binding,
    })

    config = read_json(ROOT / "config.template.json")
    config["status"] = "formal_fixture"
    config["execution_policy"]["gpu_execution_authorized_by_this_config"] = True
    config["complete_g0_v2"] = artifact_ref(g0_path)
    config["native_registry"]["published_marker"] = registry_refs["published"]
    config["native_registry"]["registry_manifest"] = registry_refs["manifest"]
    config["native_registry"]["publication_receipt"] = registry_refs["receipt"]
    config["native_registry"]["split_artifacts"] = {key: registry_refs[key] for key in split_paths}
    config["native_registry"]["strata_order"] = {"hotpotqa": ["all"], "musique": ["all"]}
    config["train_sources"]["hotpotqa"]["artifact"] = artifact_ref(hotpot_source)
    config["train_sources"]["musique"]["artifact"] = artifact_ref(musique_source)
    config["registry_projection"]["algorithm"] = algorithm
    config["profiles"] = {"smoke": {"hotpotqa": 1, "musique": 1}, "diagnostic": {"hotpotqa": 1, "musique": 1}}
    config["outputs_root"] = str((ROOT / "runs").resolve())
    config_path = tmp_path / "formal_config.json"
    write_json(config_path, config)
    return config_path


def test_checked_in_template_is_deliberately_nonformal_and_placeholder_blocked() -> None:
    config = read_json(ROOT / "config.template.json")
    assert config["execution_policy"]["gpu_execution_authorized_by_this_config"] is False
    assert find_placeholders(config)
    with pytest.raises(ContractError, match="unresolved placeholders"):
        contract_preflight(ROOT / "config.template.json", profile="smoke")


def test_valid_synthetic_final_contract_is_cpu_only(tmp_path: Path) -> None:
    config_path = build_final_fixture(tmp_path)
    torch_before = sys.modules.get("torch")
    report, samples = contract_preflight(config_path, profile="diagnostic")
    assert report["status"] == "passed_cpu_contract_preflight"
    assert report["gpu_used"] is False and report["cuda_api_called"] is False
    assert report["datasets"] == {"hotpotqa": 1, "musique": 1}
    assert len(samples) == 2
    assert sys.modules.get("torch") is torch_before


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"answer_allowed": False}, "exact preregistered"),
        ({"answer_allowed": 1}, "exact preregistered"),
        ({"answer_policy_extra": True}, "exact preregistered"),
        ({"g0_flags": 1}, "must be exactly"),
        ({"outcome_field": True}, "forbidden model-outcome field"),
        ({"projection_mismatch": True}, "source projection changed"),
    ],
)
def test_contamination_and_policy_tampering_fail_closed(tmp_path: Path, kwargs: dict, match: str) -> None:
    config_path = build_final_fixture(tmp_path, **kwargs)
    with pytest.raises(ContractError, match=match):
        contract_preflight(config_path, profile="smoke")


def test_pinned_source_byte_tamper_is_detected(tmp_path: Path) -> None:
    config_path = build_final_fixture(tmp_path)
    config = read_json(config_path)
    source = Path(config["train_sources"]["hotpotqa"]["artifact"]["path"])
    source.write_text(source.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ContractError, match="byte count changed"):
        contract_preflight(config_path, profile="smoke")


def test_gold_label_source_cannot_be_swapped_outside_registry_receipt(tmp_path: Path) -> None:
    config_path = build_final_fixture(tmp_path)
    config = read_json(config_path)
    original = Path(config["train_sources"]["hotpotqa"]["artifact"]["path"])
    swapped = tmp_path / "data" / "hotpot_swapped_answer.json"
    raw = json.loads(original.read_text(encoding="utf-8"))
    raw[0]["answer"] = "London"
    write_json(swapped, raw)
    config["train_sources"]["hotpotqa"]["artifact"] = artifact_ref(swapped)
    write_json(config_path, config)
    with pytest.raises(ContractError, match="does not exactly bind both raw train-source"):
        contract_preflight(config_path, profile="smoke")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"receipt_projection_algorithm": "legacy_unfrozen_projection"}, "algorithm is absent or differs"),
        ({"receipt_projection_excludes_answers": False}, "not declared answer-free"),
    ],
)
def test_registry_receipt_projection_is_still_checked_directly(
    tmp_path: Path, kwargs: dict, match: str
) -> None:
    config_path = build_final_fixture(tmp_path, **kwargs)
    with pytest.raises(ContractError, match=match):
        contract_preflight(config_path, profile="smoke")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda config: config["native_registry"]["required_answer_access_policy"].update(
            allowed_uses=["gold_first_token_margin", "qa_f1", "qa_em", "selection"]
        ),
        lambda config: config["native_registry"]["required_answer_access_policy"].update(
            extra_policy_switch=False
        ),
        lambda config: config["native_registry"].update(
            alternate_answer_access_policy="selection_allowed"
        ),
        lambda config: config["execution_policy"].update(
            answers_require_registry_authorization=False
        ),
    ],
)
def test_formal_config_cannot_drift_answer_access_policy(
    tmp_path: Path, mutation
) -> None:
    config_path = build_final_fixture(tmp_path)
    config = read_json(config_path)
    mutation(config)
    write_json(config_path, config)
    with pytest.raises(ContractError, match="answer|execution policy|native-registry policy"):
        contract_preflight(config_path, profile="smoke")


def test_artifact_and_formal_config_paths_reject_live_and_dangling_symlink_chains(
    tmp_path: Path,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    payload = real / "payload.json"
    write_json(payload, {"value": 1})
    payload_spec = {
        "path": "",
        "nbytes": payload.stat().st_size,
        "sha256": sha256_file(payload),
    }

    live = tmp_path / "live"
    live.symlink_to(real, target_is_directory=True)
    with pytest.raises(ContractError, match="symlink before resolution"):
        verify_ref(
            {**payload_spec, "path": str(live / "payload.json")},
            label="live-chain artifact",
        )
    with pytest.raises(ContractError, match="symlink before resolution"):
        artifact_ref(live / "payload.json")
    with pytest.raises(ContractError, match="symlink before resolution"):
        read_json(live / "payload.json")

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(ContractError, match="symlink before resolution"):
        verify_ref(
            {**payload_spec, "path": str(dangling / "payload.json")},
            label="dangling-chain artifact",
        )

    formal_target = real / "config.formal.target.json"
    write_json(formal_target, {"schema_version": 1})
    formal_link = tmp_path / "config.formal.link.json"
    formal_link.symlink_to(formal_target)
    with pytest.raises(ContractError, match="symlink before resolution"):
        ensure_formal_config_path(formal_link)


def test_borrowed_interface_hash_tamper_is_detected_before_registry(tmp_path: Path) -> None:
    config_path = build_final_fixture(tmp_path)
    config = read_json(config_path)
    config["borrowed_read_only_interfaces"]["cache_oracle"]["sha256"] = "0" * 64
    write_json(config_path, config)
    with pytest.raises(ContractError, match="borrowed interface cache_oracle config binding drifted"):
        contract_preflight(config_path, profile="smoke")


def mutate_pinned_g0(config_path: Path, mutation) -> None:
    config = read_json(config_path)
    path = Path(config["complete_g0_v2"]["path"])
    value = read_json(path)
    mutation(value)
    write_json(path, value)
    config["complete_g0_v2"] = artifact_ref(path)
    write_json(config_path, config)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value.update(candidate_registry_binding={
                "published": value["candidate_registry_binding"]["published_marker"],
                "manifest": value["candidate_registry_binding"]["registry_manifest"],
                "receipt": value["candidate_registry_binding"]["publication_receipt"],
                **value["candidate_registry_binding"]["split_artifacts"],
            }),
            "exact nested",
        ),
        (
            lambda value: value["candidate_registry_binding"].update(extra={}),
            "exact nested",
        ),
        (
            lambda value: value["candidate_registry_binding"]["split_artifacts"].pop("musique_confirm"),
            "exact nested",
        ),
        (
            lambda value: value["comparisons"].update(unregistered_direction={"flags": 0}),
            "incomplete or has extras",
        ),
        (
            lambda value: value["comparisons"]["methoddev_hotpot_vs_mapper_fit"].update(extra=0),
            "must be exactly",
        ),
        (
            lambda value: value["comparisons"]["methoddev_hotpot_vs_mapper_fit"].update(flags=False),
            "must be exactly",
        ),
        (
            lambda value: value["comparisons"]["methoddev_hotpot_vs_mapper_fit"].update(flags=0.0),
            "must be exactly",
        ),
        (
            lambda value: value["comparisons"].pop("methoddev_musique_vs_locked_eval"),
            "incomplete or has extras",
        ),
        (lambda value: value.__setitem__("pass", 1), "does not satisfy pass=True"),
    ],
)
def test_frozen_g0_nested_exact4_and_comparison_matrix_fail_closed(
    tmp_path: Path, mutation, match: str
) -> None:
    config_path = build_final_fixture(tmp_path)
    mutate_pinned_g0(config_path, mutation)
    with pytest.raises(ContractError, match=match):
        contract_preflight(config_path, profile="smoke")


def test_intervention_catalog_is_exact_preregistered_order() -> None:
    values = intervention_catalog()
    assert [item["id"] for item in values[:4]] == [
        "native_k_native_v", "mapped_k_native_v", "native_k_mapped_v", "mapped_k_mapped_v"
    ]
    assert values[4]["native_kv_layers"] == list(range(0, 10))
    assert values[7]["native_kv_layers"] == list(range(30, 40))
    assert values[8]["native_kv_layers"] == [34]
    assert values[9]["native_kv_layers"] == [35]
    assert values[10]["native_kv_layers"] == [34, 35]


def test_output_must_remain_in_independent_dir13() -> None:
    config = read_json(ROOT / "config.template.json")
    with pytest.raises(ContractError, match="escapes"):
        ensure_inside_output_root(config, "/tmp/oracle-escape")
