import hashlib
import json

import pytest

from src.contracts import ContractError, SPLIT_KEYS, artifact_ref
from src.registry_gate import validate_complete_g0, validate_registry


STRATA = {
    "hotpotqa_train": ["bridge/easy", "bridge/medium", "bridge/hard", "comparison/easy", "comparison/medium", "comparison/hard"],
    "musique_train": ["2hop", "3hop", "4hop"],
}


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return artifact_ref(path).to_dict()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return artifact_ref(path).to_dict()


def _inventory_digest(files):
    rows = [{"name": name, **files[name]} for name in sorted(files)]
    payload = (json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _publish(tmp_path):
    core_names = (
        "hotpotqa_screen.jsonl",
        "hotpotqa_confirm.jsonl",
        "musique_screen.jsonl",
        "musique_confirm.jsonl",
        "quarantine_aggregate.json",
        "registry_manifest.json",
        "receipt.json",
    )
    core = {name: artifact_ref(tmp_path / name).to_dict() for name in core_names}
    sidecars = {}
    for name, reference in core.items():
        sidecars[f"{name}.sha256.json"] = _write_json(
            tmp_path / f"{name}.sha256.json",
            {
                "path": reference["path"],
                "nbytes": reference["nbytes"],
                "sha256": reference["sha256"],
            },
        )
    published_files = {
        name: {"nbytes": reference["nbytes"], "sha256": reference["sha256"]}
        for name, reference in {**core, **sidecars}.items()
    }
    return _write_json(
        tmp_path / "PUBLISHED.json",
        {
            "schema_version": 1,
            "registry_id": "fixture",
            "status": "published_complete",
            "atomic_last_completion_marker": True,
            "published_files": published_files,
            "ordered_inventory_sha256": _inventory_digest(published_files),
            "receipt": published_files["receipt.json"],
            "receipt_sidecar": published_files["receipt.json.sha256.json"],
        },
    )


def _rows(key):
    prefix, partition = key.rsplit("_", 1)
    dataset = f"{prefix}_train"
    count = {"hotpotqa_screen": 128, "hotpotqa_confirm": 256, "musique_screen": 256, "musique_confirm": 512}[key]
    stratum = STRATA[dataset][0]
    return [
        {
            "dataset_id": dataset,
            "partition": partition,
            "sample_id": f"{key}-{index}",
            "source_index": index + (1000 if partition == "confirm" else 0),
            "source_projection_sha256": f"{index + 1:064x}",
            "selection_rank": index,
            "selection_rank_sha256": f"{index + 2:064x}",
            "stratum": stratum,
        }
        for index in range(count)
    ]


def _published_config(tmp_path):
    splits = {key: _write_jsonl(tmp_path / f"{key}.jsonl", _rows(key)) for key in SPLIT_KEYS}
    manifest = _write_json(tmp_path / "registry_manifest.json", {"status": "prepared_decontaminated_method_dev_only"})
    receipt = _write_json(
        tmp_path / "receipt.json",
        {
            "status": "pass",
            "artifact_publication": {
                "completion_marker_required": "PUBLISHED.json",
                "status_pass_receipt_without_completion_marker_is_not_published": True,
            },
        },
    )
    _write_json(tmp_path / "quarantine_aggregate.json", {"status": "pass", "flags": 0})
    marker = _publish(tmp_path)
    return {
        "profiles": {
            "smoke": {"hotpotqa_train": 2, "musique_train": 2},
            "screen": {"hotpotqa_train": 128, "musique_train": 256},
            "confirm": {"hotpotqa_train": 256, "musique_train": 512},
        },
        "registry": {
            "published_marker": marker,
            "registry_manifest": manifest,
            "publication_receipt": receipt,
            "split_artifacts": splits,
            "strata_order": STRATA,
        }
    }


def test_registry_accepts_exact_native_published_jsonl(tmp_path):
    config = _published_config(tmp_path)
    observed = validate_registry(config)
    assert {key: len(value) for key, value in observed.items()} == {
        "hotpotqa_screen": 128,
        "hotpotqa_confirm": 256,
        "musique_screen": 256,
        "musique_confirm": 512,
    }


def test_registry_rejects_non_native_extra_key(tmp_path):
    config = _published_config(tmp_path)
    path = tmp_path / "hotpotqa_screen.jsonl"
    rows = _rows("hotpotqa_screen")
    rows[0]["answer"] = "leak"
    config["registry"]["split_artifacts"]["hotpotqa_screen"] = _write_jsonl(path, rows)
    # Re-publish a consistent marker so rejection is specifically native row schema.
    config["registry"]["published_marker"] = _publish(tmp_path)
    with pytest.raises(ContractError, match="native key set changed"):
        validate_registry(config)


def test_incomplete_marker_fails_before_missing_native_payload_is_opened(tmp_path):
    missing = {"path": str((tmp_path / "missing.json").resolve()), "nbytes": 1, "sha256": "0" * 64}
    marker = _write_json(
        tmp_path / "PUBLISHED.json",
        {"schema_version": 1, "status": "in_progress", "atomic_last_completion_marker": False},
    )
    config = {
        "profiles": {
            "smoke": {"hotpotqa_train": 2, "musique_train": 2},
            "screen": {"hotpotqa_train": 2, "musique_train": 2},
            "confirm": {"hotpotqa_train": 2, "musique_train": 2},
        },
        "registry": {
            "published_marker": marker,
            "registry_manifest": missing,
            "publication_receipt": missing,
            "split_artifacts": {key: missing for key in SPLIT_KEYS},
            "strata_order": STRATA,
        }
    }
    with pytest.raises(ContractError, match="not the atomic complete marker"):
        validate_registry(config)


def test_complete_g0_requires_exact_native_binding_and_zero_flags(tmp_path):
    config = _published_config(tmp_path)
    config["g0"] = {"required_scope": "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval"}
    binding = {
        "published_marker": config["registry"]["published_marker"],
        "registry_manifest": config["registry"]["registry_manifest"],
        "publication_receipt": config["registry"]["publication_receipt"],
        "split_artifacts": config["registry"]["split_artifacts"],
    }
    receipt = {
        "status": "complete",
        "pass": True,
        "closes_complete_g0": True,
        "scope": config["g0"]["required_scope"],
        "locked_outcomes_opened": False,
        "candidate_registry_binding": binding,
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
    path = tmp_path / "g0.json"
    config["g0"]["completion_receipt"] = _write_json(path, receipt)
    assert validate_complete_g0(config)["pass"] is True
    receipt["comparisons"]["methoddev_hotpot_vs_locked_eval"]["flags"] = 1
    config["g0"]["completion_receipt"] = _write_json(path, receipt)
    with pytest.raises(ContractError, match="exact flags:int0"):
        validate_complete_g0(config)


@pytest.mark.parametrize("tamper", ("extra_direction", "extra_direction_field"))
def test_complete_g0_rejects_non_exact_comparison_matrix(tmp_path, tamper):
    config = _published_config(tmp_path)
    config["g0"] = {
        "required_scope": "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval"
    }
    binding = {
        "published_marker": config["registry"]["published_marker"],
        "registry_manifest": config["registry"]["registry_manifest"],
        "publication_receipt": config["registry"]["publication_receipt"],
        "split_artifacts": config["registry"]["split_artifacts"],
    }
    comparisons = {
        name: {"flags": 0}
        for name in (
            "methoddev_hotpot_vs_mapper_fit",
            "methoddev_musique_vs_mapper_fit",
            "methoddev_hotpot_vs_locked_eval",
            "methoddev_musique_vs_locked_eval",
        )
    }
    if tamper == "extra_direction":
        comparisons["unregistered_direction"] = {"flags": 0}
    else:
        comparisons["methoddev_hotpot_vs_mapper_fit"]["flag_count"] = 0
    receipt = {
        "status": "complete",
        "pass": True,
        "closes_complete_g0": True,
        "scope": config["g0"]["required_scope"],
        "locked_outcomes_opened": False,
        "candidate_registry_binding": binding,
        "comparisons": comparisons,
    }
    config["g0"]["completion_receipt"] = _write_json(tmp_path / "g0.json", receipt)
    message = "direction set changed" if tamper == "extra_direction" else "exact flags:int0"
    with pytest.raises(ContractError, match=message):
        validate_complete_g0(config)
