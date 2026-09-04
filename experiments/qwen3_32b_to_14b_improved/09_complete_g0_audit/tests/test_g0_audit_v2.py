from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import g0_audit_v2 as audit_v2

from g0_audit_v2 import (
    G0V2Error,
    Projection,
    SelectedRecord,
    Unit,
    _projection_sha,
    audit_record_cell,
    canonical_bytes,
    compare_record_exact,
    compare_units,
    digest,
    load_hotpot,
    load_registry_file,
    normalize,
    ordered_rows_sha256,
    rehydrate_registry,
    require_frozen_config,
    run,
    shingles,
    substring_fingerprints,
    verify_registry_evidence,
)


SIMILARITY = {
    "passage_shingle_words": 5,
    "passage_min_intersection": 20,
    "passage_min_containment": 0.5,
    "passage_min_jaccard": 0.3,
    "exact_normalized_unit_substring_min_characters": 120,
    "question_shingle_words": 3,
    "question_min_intersection": 4,
    "question_min_containment": 0.8,
    "question_min_jaccard": 0.7,
}


def unit(identity: str, text: str, *, dataset: str = "left", partition: str = "screen", kind: str = "passage_unit") -> Unit:
    hashed = digest(identity)
    return Unit(dataset, partition, hashed, digest(identity + "-unit"), kind, normalize(text))


def projection(
    dataset: str,
    sample_id: str,
    source_index: int,
    question: str,
    passage: str,
    *,
    stratum: str = "bridge/easy",
) -> Projection:
    family = "hotpotqa" if dataset.startswith("hotpotqa") else "musique"
    passages = (passage,)
    return Projection(dataset, sample_id, source_index, stratum, question, passages, _projection_sha(family, sample_id, question, passages, stratum))


def compare(left: str, right: str, *, width: int = 5, minimum_intersection: int = 20) -> dict:
    return compare_units(
        [unit("left", left)], [unit("right", right, dataset="right", partition="confirm")],
        width=width,
        minimum_intersection=minimum_intersection,
        minimum_containment=0.5 if width == 5 else 0.8,
        minimum_jaccard=0.3 if width == 5 else 0.7,
        substring_minimum=120,
        label="synthetic",
    )


def test_template_is_deliberately_fail_closed() -> None:
    template = json.loads((ROOT / "config_v2.template.json").read_text())
    with pytest.raises(G0V2Error, match="not frozen"):
        require_frozen_config(template)


def test_zero_sha_remains_fail_closed_even_if_status_is_changed() -> None:
    template = json.loads((ROOT / "config_v2.template.json").read_text())
    template["status"] = "frozen_after_registry_publication"
    with pytest.raises(G0V2Error, match="placeholder"):
        require_frozen_config(template)


def test_fully_replaced_template_can_cross_only_the_freeze_syntax_gate() -> None:
    template = json.loads((ROOT / "config_v2.template.json").read_text())
    template["status"] = "frozen_after_registry_publication"
    for name in ("published_marker", "manifest", "receipt", "independent_audit", "strict_sequence_partition_failure"):
        template["registry"][name] = {"path": f"/frozen/{name}.json", "nbytes": 1, "sha256": "a" * 64}
    for index, (name, spec) in enumerate(sorted(template["registry"]["files"].items()), 1):
        spec.update({"path": f"/frozen/{name}", "nbytes": index, "sha256": f"{index:064x}", "ordered_rows_sha256": f"{index + 10:064x}"})
    template["registry"]["musique_capacity_contract"] = {
        "observed_clean_4hop_pool": 3, "selected_4hop": 2, "reserved_unused_4hop_buffer": 1,
    }
    template["registry"]["files"]["musique_screen.jsonl"].update({"expected_records": 1, "expected_stratum_counts": {"4hop": 1}})
    template["registry"]["files"]["musique_confirm.jsonl"].update({"expected_records": 1, "expected_stratum_counts": {"4hop": 1}})
    for name in template["consumer_contracts"]:
        template["consumer_contracts"][name] = {"path": f"/frozen/{name}.json", "nbytes": 1, "sha256": "b" * 64}
    template["expected_coverage"].update({
        "method_dev_records": 386,
        "method_dev_dataset_counts": {"hotpotqa_train": 384, "musique_train": 2},
        "method_dev_partition_counts": {"confirm": 257, "screen": 129},
        "registry_files": {"hotpotqa_confirm.jsonl": 256, "hotpotqa_screen.jsonl": 128, "musique_confirm.jsonl": 1, "musique_screen.jsonl": 1},
    })
    require_frozen_config(template)
    missing = copy.deepcopy(template)
    missing["consumer_contracts"].pop("dir13_g0_consumer_contract")
    with pytest.raises(G0V2Error, match="consumer contract inventory"):
        require_frozen_config(missing)
    extra = copy.deepcopy(template)
    extra["consumer_contracts"]["unregistered_consumer"] = {
        "path": "/frozen/unregistered.json", "nbytes": 1, "sha256": "c" * 64,
    }
    with pytest.raises(G0V2Error, match="consumer contract inventory"):
        require_frozen_config(extra)


def test_template_defers_consumer_contract_pins_until_stop_editing() -> None:
    template = json.loads((ROOT / "config_v2.template.json").read_text())
    assert set(template["consumer_contracts"]) == {
        "g0_consumer_contract", "native_registry_consumer_contract",
        "dir13_g0_consumer_contract", "dir13_native_registry_consumer_contract",
    }
    for spec in template["consumer_contracts"].values():
        assert spec["path"].startswith(("__FINAL_12_", "__FINAL_13_"))
        assert spec["nbytes"] == 0 and spec["sha256"] == "0" * 64


def test_native_candidate_binding_is_exact_and_rejects_split_drift() -> None:
    split_specs = {
        f"{name}.jsonl": {
            "path": f"/native/{name}.jsonl", "nbytes": index,
            "sha256": f"{index:064x}", "extra_producer_metadata": True,
        }
        for index, name in enumerate(
            ("hotpotqa_screen", "hotpotqa_confirm", "musique_screen", "musique_confirm"), 1
        )
    }
    config = {"registry": {
        "published_marker": {"path": "/native/PUBLISHED.json", "nbytes": 5, "sha256": "5" * 64},
        "manifest": {"path": "/native/registry_manifest.json", "nbytes": 6, "sha256": "6" * 64},
        "receipt": {"path": "/native/receipt.json", "nbytes": 7, "sha256": "7" * 64},
        "files": split_specs,
    }}
    binding = audit_v2.native_candidate_registry_binding(config)
    assert set(binding) == {
        "published_marker", "registry_manifest", "publication_receipt", "split_artifacts",
    }
    assert set(binding["split_artifacts"]) == {
        "hotpotqa_screen", "hotpotqa_confirm", "musique_screen", "musique_confirm",
    }
    assert all(set(ref) == {"path", "nbytes", "sha256"} for ref in binding["split_artifacts"].values())
    drifted = copy.deepcopy(config)
    drifted["registry"]["files"]["unexpected.jsonl"] = split_specs["hotpotqa_screen.jsonl"]
    with pytest.raises(G0V2Error, match="split artifact set changed"):
        audit_v2.native_candidate_registry_binding(drifted)


def test_normalization_and_ordered_digest_contract() -> None:
    assert normalize("Ｆoo—BAR\n baz") == "foo bar baz"
    first = [{"x": 1}, {"x": 2}]
    second = list(reversed(first))
    assert ordered_rows_sha256(first) == hashlib.sha256(canonical_bytes(first)).hexdigest()
    assert ordered_rows_sha256(first) != ordered_rows_sha256(second)


def test_winnowing_has_complete_recall_under_boundary_destruction() -> None:
    randomizer = random.Random(94731)
    alphabet = "abcdef ABCDEF0123456789"
    for length in range(120, 181):
        for _ in range(8):
            source = "".join(randomizer.choice(alphabet) for _ in range(length))
            start = randomizer.randrange(length - 119)
            stop = randomizer.randrange(start + 120, length + 1)
            shared = source[start:stop]
            containing = "PREFIX_BOUNDARY" + shared + "SUFFIX_BOUNDARY"
            assert set(substring_fingerprints(shared, 120)) & set(substring_fingerprints(containing, 120))


def test_indexed_matcher_agrees_with_bruteforce_on_seeded_fuzz() -> None:
    randomizer = random.Random(20260831)
    for width, minimum_intersection, minimum_containment, minimum_jaccard in ((3, 4, 0.8, 0.7), (5, 20, 0.5, 0.3)):
        for case in range(160):
            words = [f"w{randomizer.randrange(500)}" for _ in range(randomizer.randrange(8, 70))]
            left = " ".join(words)
            if case % 4 == 0:
                right = left
            elif case % 4 == 1:
                shared = (left + " ") * max(2, 121 // max(1, len(left)))
                right = "prefixboundary" + shared + "suffixboundary"
                left = shared
            elif case % 4 == 2:
                right = " ".join([*words[: max(width + 1, len(words) - 2)], "tailunique"])
            else:
                right = " ".join(f"z{randomizer.randrange(500, 1000)}" for _ in range(len(words)))
            left_normalized = normalize(left)
            right_normalized = normalize(right)
            left_set = shingles(left_normalized, width)
            right_set = shingles(right_normalized, width)
            intersection = len(left_set & right_set)
            denominator = min(len(left_set), len(right_set))
            containment = intersection / denominator if denominator else 0.0
            union = len(left_set | right_set)
            jaccard = intersection / union if union else 0.0
            shorter, longer = (left_normalized, right_normalized) if len(left_normalized) <= len(right_normalized) else (right_normalized, left_normalized)
            brute = (
                left_normalized == right_normalized
                or (len(shorter) >= 120 and shorter in longer)
                or (intersection >= minimum_intersection and (containment >= minimum_containment or jaccard >= minimum_jaccard))
            )
            indexed = compare_units(
                [unit(f"left-{width}-{case}", left)], [unit(f"right-{width}-{case}", right)],
                width=width, minimum_intersection=minimum_intersection,
                minimum_containment=minimum_containment, minimum_jaccard=minimum_jaccard,
                substring_minimum=120, label="seeded-fuzz",
            )
            assert bool(indexed["flag_count"]) is brute


def test_exact_substring_recall_survives_destroyed_word_shingles() -> None:
    shared = "alpha beta gamma delta " + "z" * 100
    result = compare("prefix" + shared + "suffix", shared)
    assert result["flag_count"] == 1
    assert result["reason_counts"] == {"normalized_substring": 1}


def test_reverse_containment_and_long_single_token_are_detected() -> None:
    shared = "q" * 140
    result = compare(shared, "prefix" + shared + "suffix")
    assert result["flag_count"] == 1
    assert result["reason_counts"] == {"normalized_substring": 1}


def test_near_shingle_positive_and_unrelated_negative() -> None:
    shared = " ".join(f"token{index}" for index in range(45))
    positive = compare("prefix " + shared + " suffix", shared)
    assert positive["flag_count"] == 1
    unrelated = compare(" ".join(f"alpha{index}" for index in range(50)), " ".join(f"omega{index}" for index in range(50)))
    assert unrelated["flag_count"] == 0


def test_short_normalized_exact_has_no_length_gate() -> None:
    result = compare("Short, exact.", "ＳＨＯＲＴ EXACT!")
    assert result["reason_counts"] == {"normalized_exact": 1}


def test_family_scoped_id_and_normalized_prompt_context_are_explicit() -> None:
    left = projection("hotpotqa_train", "same-id", 0, "Same prompt?", "Same context.")
    right = projection("hotpotqa_locked_dev", "same-id", 0, "ＳＡＭＥ ＰＲＯＭＰＴ！", "SAME, CONTEXT!")
    exact = compare_record_exact([left], [right], "synthetic")
    assert exact["fields"]["raw_id_hash"]["count"] == 1
    assert exact["fields"]["family_scoped_id_hash"]["count"] == 1
    assert exact["fields"]["normalized_question_text_hash"]["count"] == 1
    assert exact["fields"]["normalized_question_context_text_hash"]["count"] == 1


def test_record_cell_counts_question_and_passage_gates() -> None:
    shared = " ".join(f"fact{index}" for index in range(45))
    left = projection("hotpotqa_train", "train", 0, "Which city contains the old museum by the river?", shared)
    right = projection("hotpotqa_locked_dev", "dev", 0, "Which city contains the old museum by the river today?", shared)
    cell = audit_record_cell([left], [right], SIMILARITY, "synthetic")
    assert cell["question_unit_similarity"]["flag_count"] == 1
    assert cell["passage_unit_similarity"]["flag_count"] == 1
    assert cell["flag_count"] >= 2


def test_registry_rehydration_verifies_projection_rank_and_global_uniqueness() -> None:
    source = projection("hotpotqa_train", "sample", 0, "A prompt?", "A passage.")
    namespace = "namespace"
    row = {
        "dataset_id": "hotpotqa_train",
        "partition": "screen",
        "sample_id": "sample",
        "source_index": 0,
        "source_projection_sha256": source.projection_sha256,
        "selection_rank": 0,
        "selection_rank_sha256": digest(f"{namespace}\0screen\0bridge/easy\0sample"),
        "stratum": "bridge/easy",
    }
    selected = rehydrate_registry({"hotpotqa_screen.jsonl": [row]}, {"hotpotqa_train": [source]}, {"hotpotqa_train": namespace})
    assert selected == [SelectedRecord(source, "screen", "hotpotqa_screen.jsonl")]
    tampered = copy.deepcopy(row)
    tampered["source_projection_sha256"] = "0" * 64
    with pytest.raises(G0V2Error, match="projection binding"):
        rehydrate_registry({"hotpotqa_screen.jsonl": [tampered]}, {"hotpotqa_train": [source]}, {"hotpotqa_train": namespace})


def test_registry_file_requires_allowlist_count_and_ordered_digest(tmp_path: Path) -> None:
    row = {
        "dataset_id": "hotpotqa_train", "partition": "screen", "sample_id": "x", "source_index": 0,
        "source_projection_sha256": "a" * 64, "selection_rank": 0,
        "selection_rank_sha256": "b" * 64, "stratum": "bridge/easy",
    }
    path = tmp_path / "cell.jsonl"
    path.write_text(json.dumps(row) + "\n")
    spec = {
        "expected_records": 1, "expected_stratum_counts": {"bridge/easy": 1},
        "dataset_id": "hotpotqa_train", "partition": "screen", "ordered_rows_sha256": ordered_rows_sha256([row]),
    }
    assert load_registry_file(path, spec) == [row]
    row["answer"] = "forbidden"
    path.write_text(json.dumps(row) + "\n")
    spec["ordered_rows_sha256"] = ordered_rows_sha256([row])
    with pytest.raises(G0V2Error, match="allowlist"):
        load_registry_file(path, spec)


def test_strict_locked_projection_drops_gold_sentinels(tmp_path: Path) -> None:
    sentinel = "SECRET_GOLD_MUST_NOT_SURVIVE"
    path = tmp_path / "locked.json"
    path.write_text(json.dumps([{
        "_id": "x", "question": "A prompt?", "context": [["Title", ["Sentence."]]],
        "answer": sentinel, "supporting_facts": [["Title", 0]],
    }]))
    rows = load_hotpot(path, 1, locked=True)
    assert sentinel not in repr(rows)
    assert rows[0].dataset_id == "hotpotqa_locked_dev"


def evidence_fixture(tmp_path: Path) -> dict:
    registry_id = "qwen3_32b_to_14b_decontaminated_method_dev_registry_v2"
    threshold_path = tmp_path / "threshold.json"
    threshold_path.write_text("{}")
    threshold_spec = {"path": str(threshold_path), "nbytes": threshold_path.stat().st_size, "sha256": hashlib.sha256(threshold_path.read_bytes()).hexdigest()}
    capacity_failure_path = tmp_path / "capacity-failure.json"
    capacity_failure_path.write_text("{}")
    capacity_failure_spec = {"path": str(capacity_failure_path), "nbytes": capacity_failure_path.stat().st_size, "sha256": hashlib.sha256(capacity_failure_path.read_bytes()).hexdigest()}
    capacity_contract = {"observed_clean_4hop_pool": 119, "selected_4hop": 117, "reserved_unused_4hop_buffer": 2}
    failure_binding = {**capacity_failure_spec, "status": "failed_closed_preserved_append_only"}
    counts = {
        "hotpotqa_confirm.jsonl": 256, "hotpotqa_screen.jsonl": 128,
        "musique_confirm.jsonl": 512, "musique_screen.jsonl": 256,
    }
    files = {}
    manifest_files = {}
    subject_files = {}
    for index, (name, rows) in enumerate(sorted(counts.items()), 1):
        files[name] = {
            "path": str(tmp_path / name), "nbytes": index, "sha256": f"{index:064x}",
            "ordered_rows_sha256": f"{index + 10:064x}", "expected_records": rows,
        }
        manifest_files[name] = {"nbytes": index, "sha256": f"{index:064x}", "rows": rows}
        subject_files[name] = {**manifest_files[name], "ordered_rows_sha256": f"{index + 10:064x}"}
    locked_matrix = {
        "hotpotqa_train": {"hotpotqa_locked_dev": 0, "musique_locked_dev": 0},
        "musique_train": {"hotpotqa_locked_dev": 0, "musique_locked_dev": 0},
    }
    sequence_matrix = {
        "hotpotqa_screen_to_hotpotqa_confirm": 0, "hotpotqa_screen_to_musique_confirm": 0,
        "musique_screen_to_hotpotqa_confirm": 0, "musique_screen_to_musique_confirm": 0,
    }
    privacy = {
        "registry_contains_only_allowlisted_identity_and_hash_fields": True,
        "quarantine_contains_only_aggregates": True,
        "answer_fields_projected_into_Projection": False,
        "answer_values_available_to_rank_quota_or_decontamination_logic": False,
        "answer_values_persisted_to_registry_or_quarantine": False,
        "model_outcome_fields_retained_inspected_used_or_written": False,
    }
    implementation = {
        "config": {"path": "/frozen/config", "nbytes": 1, "sha256": "1" * 64},
        "source": {"path": "/frozen/source", "nbytes": 2, "sha256": "2" * 64},
        "entrypoint": {"path": "/frozen/entrypoint", "nbytes": 3, "sha256": "3" * 64},
        "validator": {"path": "/frozen/validator", "nbytes": 4, "sha256": "4" * 64},
        "tests": {"test_registry.py": {"path": "/frozen/test", "nbytes": 5, "sha256": "5" * 64}},
    }
    implementation_inventory = [
        {"role": role, **implementation[role]}
        for role in ("config", "source", "entrypoint", "validator")
    ]
    implementation_inventory.extend(
        {"role": f"tests/{name}", **spec}
        for name, spec in sorted(implementation["tests"].items())
    )
    implementation_tree_sha = hashlib.sha256(canonical_bytes(implementation_inventory)).hexdigest()
    manifest = {
        "schema_version": 2, "registry_id": registry_id, "status": "prepared_decontaminated_method_dev_only",
        "counts": counts, "files": manifest_files, "selection_is_gold_independent": True,
        "old_03_or_07_pilot_outputs_used_as_inputs": False,
        "threshold_binding_09": threshold_spec,
        "artifact_publication": {"completion_marker_required": "PUBLISHED.json", "status_pass_receipt_without_completion_marker_is_not_published": True},
        "post_selection_four_dataset_pair_flag_counts": locked_matrix,
        "screen_confirm_sequence_disjoint_four_cell_flag_counts": sequence_matrix,
        "privacy": privacy,
        "implementation": implementation,
        "local_strict_substring_capacity_attempt": failure_binding,
        "musique_quota_derivation": {
            "reads_answers_or_model_outcomes": False, "observed_clean_4hop_pool": 119,
            "selected_4hop": 117, "asserted_unused_buffer": 2,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_spec = {"path": str(manifest_path), "nbytes": manifest_path.stat().st_size, "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    receipt = {
        "schema_version": 2, "registry_id": registry_id, "status": "pass", "cpu_only": True,
        "selection_counts": counts, "selected_identities_globally_unique": True,
        "screen_confirm_disjoint": {"hotpotqa": True, "musique": True},
        "post_selection_four_dataset_pair_flag_counts": locked_matrix,
        "screen_confirm_sequence_disjoint_four_cell_flag_counts": sequence_matrix,
        "manifest": manifest_spec,
        "artifact_publication": {"completion_marker_required": "PUBLISHED.json", "status_pass_receipt_without_completion_marker_is_not_published": True},
        "local_strict_substring_capacity_attempt": failure_binding,
        "musique_quota_derivation": {
            "clean_pool_frozen_before_sha_ranking": True, "capacity_only_no_answer_or_outcome": True,
            **capacity_contract,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    receipt_spec = {"path": str(receipt_path), "nbytes": receipt_path.stat().st_size, "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
    publication_inventory = {
        "registry_manifest.json": {"nbytes": manifest_spec["nbytes"], "sha256": manifest_spec["sha256"]},
        "receipt.json": {"nbytes": receipt_spec["nbytes"], "sha256": receipt_spec["sha256"]},
    }
    publication_inventory.update({name: {"nbytes": spec["nbytes"], "sha256": spec["sha256"]} for name, spec in files.items()})
    ordered_inventory = [{"name": name, **publication_inventory[name]} for name in sorted(publication_inventory)]
    published = {
        "schema_version": 1, "registry_id": registry_id, "status": "published_complete", "atomic_last_completion_marker": True,
        "published_files": publication_inventory,
        "ordered_inventory_sha256": hashlib.sha256(canonical_bytes(ordered_inventory)).hexdigest(),
    }
    published_path = tmp_path / "PUBLISHED.json"
    published_path.write_text(json.dumps(published))
    published_spec = {"path": str(published_path), "nbytes": published_path.stat().st_size, "sha256": hashlib.sha256(published_path.read_bytes()).hexdigest()}
    independent = {
        "schema_version": 1, "audit_id": "registry_v2_final_independent_readonly_audit_v1", "status": "pass",
        "stop_editing_observed": True, "source_tree_frozen_during_audit": True,
        "source_tree_digests": {"start_sha256": implementation_tree_sha, "end_sha256": implementation_tree_sha},
        "checks": {
            "published_inventory_rehashed": True, "implementation_files_rehashed": True,
            "registry_jsonl_allowlist_and_counts_recomputed": True, "ordered_rows_sha256_recomputed": True,
            "global_identity_uniqueness_recomputed": True, "screen_confirm_identity_disjointness_recomputed": True,
            "deterministic_mirror_core_files_byte_identical": True, "locked_outcomes_and_model_outputs_not_read": True,
        },
        "subject": {
            "published_marker": published_spec, "manifest": manifest_spec, "receipt": receipt_spec,
            "strict_capacity_failure": capacity_failure_spec, "files": subject_files, "implementation": implementation,
        },
        "selection_counts": counts,
        "musique_capacity_contract": capacity_contract,
        "post_selection_four_dataset_pair_flag_counts": locked_matrix,
        "screen_confirm_sequence_disjoint_four_cell_flag_counts": sequence_matrix,
    }
    independent_path = tmp_path / "independent.json"
    independent_path.write_text(json.dumps(independent))
    independent_spec = {"path": str(independent_path), "nbytes": independent_path.stat().st_size, "sha256": hashlib.sha256(independent_path.read_bytes()).hexdigest()}
    return {
        "inputs": {"g0_v1_threshold_config": threshold_spec},
        "registry": {
            "registry_id": registry_id, "published_marker": published_spec, "manifest": manifest_spec,
            "receipt": receipt_spec, "independent_audit": independent_spec,
            "strict_capacity_failure": capacity_failure_spec, "musique_capacity_contract": capacity_contract,
            "files": files,
        },
    }


def test_registry_evidence_requires_complete_independent_pass(tmp_path: Path) -> None:
    config = evidence_fixture(tmp_path)
    assert verify_registry_evidence(config) == {
        "atomic_publication_marker_semantics": "verified", "manifest_semantics": "verified",
        "receipt_semantics": "verified", "independent_readonly_audit_semantics": "verified",
    }
    independent_path = Path(config["registry"]["independent_audit"]["path"])
    independent = json.loads(independent_path.read_text())
    independent["screen_confirm_sequence_disjoint_four_cell_flag_counts"].pop("musique_screen_to_musique_confirm")
    independent_path.write_text(json.dumps(independent))
    with pytest.raises(G0V2Error, match="matrix"):
        verify_registry_evidence(config)


def test_existing_output_directory_fails_before_config_or_locked_data_is_read(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(G0V2Error, match="existing output"):
        run(tmp_path / "missing-config.json", output)


def test_synthetic_end_to_end_run_emits_a_closed_zero_flag_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    namespace_hp = "synthetic/hotpot"
    namespace_mq = "synthetic/musique"
    hp_train = [
        projection("hotpotqa_train", "hp-screen", 0, "hp screen unique", "hp screen passage"),
        projection("hotpotqa_train", "hp-confirm", 1, "hp confirm unique", "hp confirm passage"),
    ]
    mq_train = [
        projection("musique_train", "2hop__mq_screen", 0, "mq screen unique", "mq screen passage", stratum="2hop"),
        projection("musique_train", "2hop__mq_confirm", 1, "mq confirm unique", "mq confirm passage", stratum="2hop"),
    ]
    hp_locked = [projection("hotpotqa_locked_dev", "hp-locked", 0, "locked hp distinct", "locked hp material", stratum="locked")]
    mq_locked = [projection("musique_locked_dev", "2hop__mq_locked", 0, "locked mq distinct", "locked mq material", stratum="locked")]

    def registry_row(source: Projection, partition: str, namespace: str) -> dict:
        return {
            "dataset_id": source.dataset_id, "partition": partition, "sample_id": source.sample_id,
            "source_index": source.source_index, "source_projection_sha256": source.projection_sha256,
            "selection_rank": 0,
            "selection_rank_sha256": digest(f"{namespace}\0{partition}\0{source.stratum}\0{source.sample_id}"),
            "stratum": source.stratum,
        }

    rows = {
        "hotpotqa_screen.jsonl": [registry_row(hp_train[0], "screen", namespace_hp)],
        "hotpotqa_confirm.jsonl": [registry_row(hp_train[1], "confirm", namespace_hp)],
        "musique_screen.jsonl": [registry_row(mq_train[0], "screen", namespace_mq)],
        "musique_confirm.jsonl": [registry_row(mq_train[1], "confirm", namespace_mq)],
    }
    file_specs = {
        name: {
            "path": str(tmp_path / name), "nbytes": 1, "sha256": f"{index:064x}",
            "ordered_rows_sha256": ordered_rows_sha256(value), "expected_records": 1,
            "expected_stratum_counts": {value[0]["stratum"]: 1},
            "dataset_id": value[0]["dataset_id"], "partition": value[0]["partition"],
        }
        for index, (name, value) in enumerate(sorted(rows.items()), 1)
    }
    frozen_stub = lambda name: {"path": f"/frozen/{name}.json", "nbytes": 1, "sha256": "a" * 64}
    config = {
        "schema_version": 2, "status": "frozen_after_registry_publication", "normalization": "unicode_nfkc_casefold_keep_alnum_collapse_whitespace",
        "similarity": SIMILARITY,
        "inputs": {
            "hotpot_train": {"path": "/frozen/hp-train", "expected_records": 2},
            "musique_train": {"path": "/frozen/mq-train", "expected_records": 2},
            "hotpot_locked_dev": {"path": "/frozen/hp-dev", "expected_records": 1},
            "musique_locked_dev": {"path": "/frozen/mq-dev", "expected_records": 1},
        },
        "registry": {
            "registry_id": "synthetic-registry", "published_marker": frozen_stub("published"),
            "manifest": frozen_stub("manifest"), "receipt": frozen_stub("receipt"),
            "independent_audit": frozen_stub("independent"), "files": file_specs,
            "selection_namespaces": {"hotpotqa_train": namespace_hp, "musique_train": namespace_mq},
        },
        "consumer_contracts": {
            "g0_consumer_contract": frozen_stub("g0-consumer-contract"),
            "native_registry_consumer_contract": frozen_stub("native-registry-consumer-contract"),
            "dir13_g0_consumer_contract": frozen_stub("dir13-g0-consumer-contract"),
            "dir13_native_registry_consumer_contract": frozen_stub("dir13-native-registry-consumer-contract"),
        },
        "expected_coverage": {
            "mapper_fit_sequences": 550, "method_dev_records": 4,
            "method_dev_dataset_counts": {"hotpotqa_train": 2, "musique_train": 2},
            "method_dev_partition_counts": {"confirm": 2, "screen": 2},
            "locked_projection_records": 2,
            "locked_dataset_counts": {"hotpotqa_locked_dev": 1, "musique_locked_dev": 1},
            "registry_files": {name: 1 for name in sorted(rows)},
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(audit_v2, "iter_file_specs", lambda _config: ())
    monkeypatch.setattr(audit_v2, "verify_threshold_binding", lambda _config: None)
    monkeypatch.setattr(audit_v2, "verify_registry_evidence", lambda _config: {"synthetic": "verified"})
    monkeypatch.setattr(audit_v2, "load_registry_file", lambda path, _spec: rows[path.name])
    monkeypatch.setattr(audit_v2, "load_hotpot", lambda _path, _count, locked: hp_locked if locked else hp_train)
    monkeypatch.setattr(audit_v2, "load_musique", lambda _path, _count, locked: mq_locked if locked else mq_train)
    monkeypatch.setattr(audit_v2, "load_mapper_texts", lambda _config: [f"mapper singleton{i}" for i in range(550)])
    output = tmp_path / "formal"
    receipt = audit_v2.run(config_path, output)
    assert receipt["result"] == {
        "cross_tier_flag_count": 0, "screen_confirm_isolation_flag_count": 0,
        "total_isolation_flag_count": 0, "pass": True,
    }
    assert receipt["status"] == "complete"
    assert receipt["scope"] == "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval"
    assert receipt["closes_complete_g0"] is True
    assert receipt["locked_outcomes_opened"] is False
    assert receipt["candidate_registry_binding"] == {
        "published_marker": frozen_stub("published"),
        "registry_manifest": frozen_stub("manifest"),
        "publication_receipt": frozen_stub("receipt"),
        "split_artifacts": {
            name.removesuffix(".jsonl"): audit_v2.artifact_ref(file_specs[name])
            for name in (
                "hotpotqa_screen.jsonl", "hotpotqa_confirm.jsonl",
                "musique_screen.jsonl", "musique_confirm.jsonl",
            )
        },
    }
    assert receipt["consumer_contract_binding"] == config["consumer_contracts"]
    assert set(receipt["comparisons"]) == {
        "methoddev_hotpot_vs_mapper_fit", "methoddev_musique_vs_mapper_fit",
        "methoddev_hotpot_vs_locked_eval", "methoddev_musique_vs_locked_eval",
    }
    assert all(value == {"flags": 0} for value in receipt["comparisons"].values())
    assert (output / "complete_g0_receipt.json").is_file()
    serialized = json.dumps(receipt)
    assert "hp screen unique" not in serialized and "mq confirm passage" not in serialized


def test_v2_is_independent_of_old_candidate_and_registry_implementation() -> None:
    source = (ROOT / "src" / "g0_audit_v2.py").read_text()
    forbidden = ("01_oracle_prepare_v1", "03_tail_refresh_v2", "07_method_dev_prepared_v1", "from src.registry", "import registry")
    assert not any(token in source for token in forbidden)
