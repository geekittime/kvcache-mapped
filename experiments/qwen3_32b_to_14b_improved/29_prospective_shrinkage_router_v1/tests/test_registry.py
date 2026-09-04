from __future__ import annotations

import hashlib

import pytest

from src.registry import RegistryError, router_observations, validate_manifest


TOKENIZER_SHA = "a" * 64


def make_manifest(calibration: int = 24, sealed: int = 48) -> dict:
    splits = {"calibration": [], "sealed_eval": []}
    counts = {"calibration": {}, "sealed_eval": {}}
    for split, size in (("calibration", calibration), ("sealed_eval", sealed)):
        counts[split] = {"hotpotqa_train": size, "musique_train": size}
        for dataset in ("hotpotqa_train", "musique_train"):
            for index in range(size):
                sample_id = f"{split}-{dataset}-{index}"
                splits[split].append(
                    {
                        "sample_id": sample_id,
                        "dataset_id": dataset,
                        "source_record_sha256": hashlib.sha256(sample_id.encode()).hexdigest(),
                    }
                )
    return {
        "schema_version": 1,
        "registry_id": "fresh-test-registry",
        "tokenizer": {"path": "/tmp/tokenizer.json", "sha256": TOKENIZER_SHA},
        "labels_hidden_from_router": True,
        "splits": splits,
        "dataset_counts": counts,
    }


def test_valid_manifest_and_disjoint_split() -> None:
    result = validate_manifest(make_manifest(), expected_tokenizer_sha256=TOKENIZER_SHA)
    assert len(result.records) == 144
    assert len(result.split_records["calibration"]) == 48
    assert len(result.split_records["sealed_eval"]) == 96
    assert len(result.manifest_sha256) == 64


@pytest.mark.parametrize(
    "mutator",
    [
        lambda m: m["labels_hidden_from_router"].__class__ and m.update(labels_hidden_from_router=False),
        lambda m: m["splits"]["sealed_eval"].append(m["splits"]["calibration"][0]),
        lambda m: m["tokenizer"].update(sha256="b" * 64),
        lambda m: m["dataset_counts"]["sealed_eval"].update(hotpotqa_train=47),
    ],
)
def test_manifest_mutations_fail_closed(mutator) -> None:
    manifest = make_manifest()
    mutator(manifest)
    with pytest.raises(RegistryError):
        validate_manifest(manifest, expected_tokenizer_sha256=TOKENIZER_SHA)


def test_manifest_shortfall_fails() -> None:
    with pytest.raises(RegistryError):
        validate_manifest(make_manifest(calibration=23), expected_tokenizer_sha256=TOKENIZER_SHA)
    with pytest.raises(RegistryError):
        validate_manifest(make_manifest(sealed=47), expected_tokenizer_sha256=TOKENIZER_SHA)


def test_router_projection_rejects_labels_and_extra_fields() -> None:
    with pytest.raises(RegistryError):
        router_observations(
            [{"prompt_tokens": 3000, "r1_generated_tokens": 20, "first_pass_complete": True, "dataset_id": "hotpotqa_train"}]
        )
    observations = router_observations(
        [{"prompt_tokens": 3000, "r1_generated_tokens": 20, "first_pass_complete": True}]
    )
    assert observations[0].prompt_tokens == 3000

