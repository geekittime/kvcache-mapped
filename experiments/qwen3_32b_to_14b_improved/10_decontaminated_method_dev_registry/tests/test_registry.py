from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.registry import (
    Projection,
    RegistryError,
    UnitIndex,
    _match_unit,
    build_mapper_index,
    build_reference_index,
    contamination_reasons,
    decontaminate,
    load_hotpot,
    _musique_hop,
    normalize,
    select_partitioned,
    train_union_question_collisions,
    run,
    verify_file,
    verify_capacity_failure_semantics,
    verify_failure_evidence_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config.json").read_text())
SIM = CONFIG["decontamination"]


def projection(sample_id: str, question: str, passage: str, *, stratum: str = "bridge/easy") -> Projection:
    return Projection("synthetic", sample_id, int(sample_id.lstrip("x") or 0), stratum, question, (passage,), hashlib.sha256(sample_id.encode()).hexdigest())


def test_normalization_and_locked_near_question_collision() -> None:
    locked = [projection("x1", "Who wrote the famous old red blue green book?", "unrelated passage words here")]
    candidate = projection("x2", "WHO wrote the famous old red blue green book!", "another unrelated passage here")
    index = build_reference_index(locked, SIM)
    empty_q = build_mapper_index([], SIM, question=True)
    empty_p = build_mapper_index([], SIM, question=False)
    reasons = contamination_reasons(candidate, index, empty_q, empty_p, SIM)
    assert "locked_normalized_question" in reasons
    assert normalize("Ａ, B!") == "a b"


def test_official_musique_id_variants_map_to_hop_strata() -> None:
    assert _musique_hop("2hop__a_b") == "2hop"
    assert _musique_hop("3hop2__a_b_c") == "3hop"
    assert _musique_hop("4hop3__a_b_c_d") == "4hop"


def test_passage_threshold_and_mapper_question_are_enforced() -> None:
    locked_passage = " ".join(f"token{i}" for i in range(45))
    candidate_passage = "prefix " + locked_passage + " suffix"
    locked = [projection("x1", "totally unrelated question text", locked_passage)]
    candidate = projection("x2", "which mapper phrase has enough shared words today", candidate_passage)
    mapper = ["A document asks which mapper phrase has enough shared words today and then continues."]
    index = build_reference_index(locked, SIM)
    mapper_q = build_mapper_index(mapper, SIM, question=True)
    mapper_p = build_mapper_index(mapper, SIM, question=False)
    reasons = contamination_reasons(candidate, index, mapper_q, mapper_p, SIM)
    assert any(reason.startswith("locked_passage_") for reason in reasons)
    assert any(reason.startswith("mapper_question_") for reason in reasons)


def test_long_single_token_exact_substring_has_no_shingle_blind_spot() -> None:
    long_token = "z" * 140
    locked = [projection("x1", "unrelated locked question", long_token)]
    candidate = projection("x2", "another unrelated question", f"prefix{long_token}suffix")
    index = build_reference_index(locked, SIM)
    empty_q = build_mapper_index([], SIM, question=True)
    empty_p = build_mapper_index([], SIM, question=False)
    reasons = contamination_reasons(candidate, index, empty_q, empty_p, SIM)
    assert "locked_passage_normalized_substring" in reasons


def test_short_exact_normalized_passage_has_no_length_gate() -> None:
    locked = [projection("x1", "unrelated locked question", "Short exact passage.")]
    candidate = projection("x2", "another unrelated question", "SHORT, EXACT PASSAGE!")
    index = build_reference_index(locked, SIM)
    empty_q = build_mapper_index([], SIM, question=True)
    empty_p = build_mapper_index([], SIM, question=False)
    reasons = contamination_reasons(candidate, index, empty_q, empty_p, SIM)
    assert "locked_passage_normalized_exact" in reasons


def test_normal_path_does_not_enumerate_all_reference_sets() -> None:
    class BombSequence:
        def __iter__(self):
            raise AssertionError("normal miss must not full-scan reference sets")
        def __getitem__(self, index):
            raise AssertionError("no shingle candidate should index reference sets")
    reference = UnitIndex(
        sets=BombSequence(), texts=("unrelated reference text",), inverted={},
        normalized_hashes=frozenset(), substring_fingerprint_inverted={},
    )
    assert _match_unit(
        "ordinary candidate words here", width=5, reference=reference,
        minimum_intersection=20, minimum_containment=.5,
        minimum_jaccard=.3, substring_min_chars=120,
    ) == (False, None)


def test_match_reason_is_byte_stable_across_python_hash_seeds() -> None:
    code = r'''
import json
from src.registry import Projection, build_reference_index, build_mapper_index, contamination_reasons
cfg=json.load(open("config.json"))["decontamination"]
p=lambda i,q,x: Projection("d",i,0,"s",q,(x,),i)
needle=" ".join("word"+str(i) for i in range(45))
refs=[p("a","unrelated question one",needle),p("b","unrelated question two","prefix "+needle+" suffix")]
candidate=p("c","distinct candidate question","prefix "+needle+" suffix more")
idx=build_reference_index(refs,cfg)
e1=build_mapper_index([],cfg,question=True); e2=build_mapper_index([],cfg,question=False)
print(json.dumps(sorted(contamination_reasons(candidate,idx,e1,e2,cfg))))
'''
    outputs = []
    for seed in ("1", "2", "3", "7", "11"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT))
        outputs.append(subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True))
    assert len(set(outputs)) == 1


def test_train_question_collisions_cover_cross_dataset_short_exact() -> None:
    left = projection("x1", "Same?", "left passage")
    right = Projection("other", "x2", 1, "2hop", "ＳＡＭＥ！", ("right passage",), "sha")
    marked, counts = train_union_question_collisions([left, right], SIM)
    assert marked == frozenset({("synthetic", "x1"), ("other", "x2")})
    assert counts["normalized_exact_question"] == 1


@pytest.mark.parametrize("placement", ["prefix", "suffix", "internal"])
def test_exact_substring_recall_survives_destroyed_boundary_shingles(placement: str) -> None:
    # Five long words make the normalized unit longer than 120 characters, but
    # changing either boundary destroys its only width-5 word shingle.
    locked_text = " ".join(chr(97 + index) * 30 for index in range(5))
    if placement == "prefix":
        candidate_text = "prefix" + locked_text
    elif placement == "suffix":
        candidate_text = locked_text + "suffix"
    else:
        candidate_text = "prefix" + locked_text + "suffix"
    locked = [projection("x1", "unrelated locked question", locked_text)]
    candidate = projection("x2", "another unrelated question", candidate_text)
    index = build_reference_index(locked, SIM)
    empty_q = build_mapper_index([], SIM, question=True)
    empty_p = build_mapper_index([], SIM, question=False)
    reasons = contamination_reasons(candidate, index, empty_q, empty_p, SIM)
    assert "locked_passage_normalized_substring" in reasons


def test_train_union_exact_substring_recall_is_cross_dataset_and_width3_complete() -> None:
    left_question = " ".join(chr(97 + index) * 42 for index in range(3))
    left = projection("x1", left_question, "left passage")
    right = Projection(
        "other", "x2", 1, "2hop", "prefix" + left_question,
        ("right passage",), "sha",
    )
    marked, counts = train_union_question_collisions([left, right], SIM)
    assert marked == frozenset({("synthetic", "x1"), ("other", "x2")})
    assert counts["normalized_question_substring"] == 1

    # Reverse source order exercises the opposite containment direction.
    marked_reversed, counts_reversed = train_union_question_collisions([right, left], SIM)
    assert marked_reversed == marked
    assert counts_reversed["normalized_question_substring"] == 1


def test_exact_substring_recall_handles_reverse_containment_and_empty_shingles() -> None:
    candidate_text = "z" * 140
    locked_text = "prefix" + candidate_text + "suffix"
    locked = [projection("x1", "unrelated locked question", locked_text)]
    candidate = projection("x2", "another unrelated question", candidate_text)
    index = build_reference_index(locked, SIM)
    empty_q = build_mapper_index([], SIM, question=True)
    empty_p = build_mapper_index([], SIM, question=False)
    reasons = contamination_reasons(candidate, index, empty_q, empty_p, SIM)
    assert "locked_passage_normalized_substring" in reasons


def test_stratified_sha_selection_is_deterministic_and_disjoint() -> None:
    rows = [projection(f"x{i}", f"question {i}", f"passage {i}", stratum="a" if i < 10 else "b") for i in range(20)]
    kwargs = {
        "namespace": "test-v2",
        "strata_order": ["a", "b"],
        "screen_quotas": {"a": 2, "b": 2},
        "confirm_quotas": {"a": 3, "b": 3},
    }
    first = select_partitioned(rows, **kwargs)
    second = select_partitioned(list(reversed(rows)), **kwargs)
    assert first == second
    assert not ({row["sample_id"] for row in first["screen"]} & {row["sample_id"] for row in first["confirm"]})
    assert {row["stratum"] for row in first["screen"]} == {"a", "b"}


def test_fail_closed_when_clean_stratum_pool_is_too_small() -> None:
    rows = [projection("x1", "q one", "p one", stratum="a")]
    with pytest.raises(RegistryError, match="fail closed"):
        select_partitioned(rows, namespace="n", strata_order=["a"], screen_quotas={"a": 1}, confirm_quotas={"a": 1})


def test_strict_locked_projection_drops_answer_sentinel(tmp_path: Path) -> None:
    sentinel = "SECRET_GOLD_MUST_NOT_SURVIVE"
    source = tmp_path / "locked.json"
    source.write_text(json.dumps([{
        "_id": "x1",
        "question": "A valid question?",
        "context": [["Title", ["Sentence."]]],
        "answer": sentinel,
        "supporting_facts": [["Title", 0]],
    }]))
    rows = load_hotpot(source, 1, locked=True)
    assert sentinel not in repr(rows)
    assert not hasattr(rows[0], "answer")


def test_duplicate_train_sample_id_fails_closed(tmp_path: Path) -> None:
    row = {"_id": "duplicate", "question": "A valid question?", "context": [["Title", ["Sentence."]]], "type": "bridge", "level": "easy"}
    source = tmp_path / "train.json"
    source.write_text(json.dumps([row, row]))
    with pytest.raises(RegistryError, match="duplicate HotpotQA sample_id"):
        load_hotpot(source, 2, locked=False)


def test_normalized_question_context_exact_gate_is_explicit() -> None:
    locked = [projection("x1", "same question", "same short context")]
    candidate = projection("x2", "SAME QUESTION!", "SAME, SHORT CONTEXT!")
    index = build_reference_index(locked, SIM)
    empty_q = build_mapper_index([], SIM, question=True)
    empty_p = build_mapper_index([], SIM, question=False)
    reasons = contamination_reasons(candidate, index, empty_q, empty_p, SIM)
    assert "locked_normalized_question_context" in reasons


def test_pinned_input_tamper_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    spec = {"nbytes": 8, "sha256": hashlib.sha256(b"original").hexdigest()}
    assert verify_file(source, spec) == spec
    source.write_bytes(b"tampered")
    with pytest.raises(RegistryError, match="pinned input mismatch"):
        verify_file(source, spec)


def test_nonempty_artifact_directory_is_never_reused(tmp_path: Path) -> None:
    config = dict(CONFIG)
    config["outputs"] = {"artifact_dir": "artifacts/final"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    artifact = tmp_path / "artifacts" / "final"
    artifact.mkdir(parents=True)
    (artifact / "stale_pass_receipt.json").write_text("{}")
    with pytest.raises(RegistryError, match="refusing to reuse existing"):
        run(config_path, tmp_path)
    # The exception path explicitly releases the advisory fd: a second call
    # reaches the durable-directory rejection, not a stale writer-lock error.
    with pytest.raises(RegistryError, match="refusing to reuse existing"):
        run(config_path, tmp_path)


def test_empty_artifact_directory_is_also_never_reused(tmp_path: Path) -> None:
    config = dict(CONFIG)
    config["outputs"] = {"artifact_dir": "artifacts/empty"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    (tmp_path / "artifacts" / "empty").mkdir(parents=True)
    with pytest.raises(RegistryError, match="refusing to reuse existing"):
        run(config_path, tmp_path)


def test_09_failure_evidence_semantics_are_strongly_checked() -> None:
    path = Path(CONFIG["failure_evidence"]["complete_g0_v1_receipt"]["path"])
    verify_failure_evidence_semantics(path)


def test_attempt2_capacity_failure_evidence_is_append_only_and_strongly_checked() -> None:
    spec = CONFIG["failure_evidence"]["local_strict_substring_capacity_attempt"]
    path = ROOT / spec["path"]
    assert verify_file(path, spec) == {"nbytes": spec["nbytes"], "sha256": spec["sha256"]}
    verify_capacity_failure_semantics(path, ROOT)


def test_privacy_claim_matches_standard_json_decoder_behavior() -> None:
    privacy = CONFIG["privacy"]
    assert privacy["raw_answer_values_transiently_parsed_into_decoder_object_pairs"] is True
    assert privacy["answer_fields_projected_into_Projection"] is False
    assert privacy["answer_values_available_to_rank_quota_or_decontamination_logic"] is False
    assert privacy["answer_values_persisted_to_registry_or_quarantine"] is False


def test_config_has_no_old_pilot_output_inputs() -> None:
    inventory = [CONFIG["threshold_binding"]["source"]["path"], CONFIG["failure_evidence"]["complete_g0_v1_receipt"]["path"]]
    inventory.extend(spec["path"] for spec in CONFIG["inputs"].values())
    forbidden = ("03_tail_refresh/runs", "07_method_dev_evaluation/runs", "pilot_output", "model_output")
    assert not any(token in path for path in inventory for token in forbidden)


def test_union_policy_and_failure_evidence_are_frozen() -> None:
    assert "union" in CONFIG["decontamination"]["locked_comparison_scope"]
    evidence = CONFIG["failure_evidence"]["complete_g0_v1_receipt"]
    assert evidence["sha256"] == "4b66efa69e269dae9bf5d710e2dd4adb8041b3c4651ee3f140c6a44031ae71f8"
    quotas = CONFIG["selection"]["musique"]
    assert quotas["screen_quotas"] == {"2hop": 109, "3hop": 108, "4hop": 39}
    assert quotas["confirm_quotas"] == {"2hop": 217, "3hop": 217, "4hop": 78}
    assert quotas["expected_clean_4hop_capacity"] == 119
    assert quotas["reserved_clean_4hop_buffer"] == 2
    assert CONFIG["failure_evidence"]["local_strict_substring_capacity_attempt"]["sha256"] == "ba4431d54514153811e8b843dbce28b4911cf0998bbc66744ecfaeb23fd9901d"


def test_cross_dataset_locked_union_quarantines_candidate() -> None:
    candidate = projection("x9", "clean unique question with many terms", " ".join(f"shared{i}" for i in range(50)))
    unrelated = projection("x1", "other locked question words here", "totally unrelated short passage")
    cross_locked = projection("x2", "second locked question words here", candidate.passages[0])
    indexes = {
        "hotpotqa_locked_dev": build_reference_index([unrelated], SIM),
        "musique_locked_dev": build_reference_index([cross_locked], SIM),
    }
    empty_q = build_mapper_index([], SIM, question=True)
    empty_p = build_mapper_index([], SIM, question=False)
    eligible, aggregate = decontaminate(
        [candidate], indexes,
        {"hotpotqa_locked_dev": 1, "musique_locked_dev": 1},
        empty_q, empty_p, SIM,
        frozenset(),
    )
    assert eligible == []
    assert aggregate["reason_counts_nonexclusive"]["musique_locked_dev_passage_normalized_exact"] == 1


def test_actual_registry_privacy_counts_and_sidecars() -> None:
    artifact = ROOT / CONFIG["outputs"]["artifact_dir"]
    publication = json.loads((artifact / "PUBLISHED.json").read_text())
    assert publication["status"] == "published_complete"
    assert publication["atomic_last_completion_marker"] is True
    expected_published = sorted(path.name for path in artifact.iterdir() if path.name != "PUBLISHED.json")
    assert sorted(publication["published_files"]) == expected_published
    for name, observed in publication["published_files"].items():
        path = artifact / name
        assert observed == {
            "nbytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = json.loads((artifact / "registry_manifest.json").read_text())
    expected = {
        "hotpotqa_screen.jsonl": 128,
        "hotpotqa_confirm.jsonl": 256,
        "musique_screen.jsonl": 256,
        "musique_confirm.jsonl": 512,
    }
    assert manifest["counts"] == expected
    allowed = set(CONFIG["privacy"]["registry_output_fields"])
    datasets: dict[str, dict[str, set[str]]] = {}
    for name, count in expected.items():
        path = artifact / name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == count
        assert all(set(row) == allowed for row in rows)
        assert all(not ({"question", "context", "answer", "answers", "prediction"} & set(row)) for row in rows)
        dataset = name.split("_", 1)[0]
        partition = "screen" if "screen" in name else "confirm"
        datasets.setdefault(dataset, {})[partition] = {row["sample_id"] for row in rows}
        sidecar = json.loads((artifact / f"{name}.sha256.json").read_text())
        assert sidecar["nbytes"] == path.stat().st_size
        assert sidecar["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (datasets["hotpotqa"]["screen"] & datasets["hotpotqa"]["confirm"])
    assert not (datasets["musique"]["screen"] & datasets["musique"]["confirm"])
    quarantine = json.loads((artifact / "quarantine_aggregate.json").read_text())
    serialized = json.dumps(quarantine).casefold()
    assert "sample_id" not in serialized and "question_sha" not in serialized
    assert quarantine["hotpotqa"]["source_records"] == 90447
    assert quarantine["musique"]["source_records"] == 19938
