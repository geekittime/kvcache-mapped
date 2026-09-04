import json
import os
from pathlib import Path

import pytest

from src.configuration import load_config, validated_profile_counts
import src.contracts as contracts
from src.contracts import (
    ArtifactRef,
    ContractError,
    RAW_MODES,
    RUNTIME_PACKAGE_CLOSURE_FILES,
    RUNTIME_PACKAGE_CLOSURE_ROOT,
    artifact_ref,
    canonical_sha256,
    read_json,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]


def test_template_is_deliberately_non_runnable_and_fails_before_path_access():
    with pytest.raises(ContractError, match="unresolved placeholders"):
        load_config(ROOT / "config.template.json", verify_inputs=True)


def test_raw_modes_are_exact_and_exclude_r8():
    assert RAW_MODES == ("native14", "nvidia_k20", "shrinkage", "tail_refresh_r32")
    assert all("r8" not in mode for mode in RAW_MODES)


def test_final_publication_counts_are_config_bound_but_scope_bounded():
    value = {
        "smoke": {"hotpotqa_train": 2, "musique_train": 2},
        "screen": {"hotpotqa_train": 120, "musique_train": 200},
        "confirm": {"hotpotqa_train": 240, "musique_train": 400},
    }
    assert validated_profile_counts(value) == value
    value["screen"]["musique_train"] = 257
    with pytest.raises(ContractError, match="<= 256"):
        validated_profile_counts(value)


def test_artifact_ref_rejects_hash_tamper(tmp_path):
    path = tmp_path / "value.json"
    path.write_text('{"x":1}\n', encoding="utf-8")
    reference = artifact_ref(path)
    path.write_text('{"x":2}\n', encoding="utf-8")
    with pytest.raises(ContractError, match="SHA-256 drifted"):
        reference.verify(label="tampered")


def test_reference_inventory_exactly_matches_read_only_files():
    inventory = read_json(ROOT / "reference_interfaces.json")
    assert inventory["status"] == "read_only_design_references_frozen"
    for group in ("legacy_07", "watchdog_11_hypothesis_only"):
        for name, value in inventory[group].items():
            if name == "role":
                continue
            reference = ArtifactRef.parse(value, label=f"{group}.{name}")
            assert reference.verify(label=f"{group}.{name}").is_file()


def test_legacy_code_is_not_imported_by_new_source():
    for path in (ROOT / "src").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "07_method_dev_evaluation" not in text or path.name == "contracts.py"
        assert "from src.runner" not in text


def test_code_inventory_rejects_symlinks_and_nonregular_entries(tmp_path, monkeypatch):
    root = tmp_path / "inventory"
    root.mkdir()
    (root / "real.py").write_text("value = 1\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("value = 2\n", encoding="utf-8")

    monkeypatch.setattr(contracts, "ROOT", root)
    (root / "link.py").symlink_to(outside)
    with pytest.raises(ContractError, match="refuses symlink"):
        contracts.code_inventory()

    (root / "link.py").unlink()
    fifo = root / "pipe"
    os.mkfifo(fifo)
    try:
        with pytest.raises(ContractError, match="refuses non-regular"):
            contracts.code_inventory()
    finally:
        fifo.unlink()


def test_runtime_package_closure_is_frozen_and_content_addressed():
    config = read_json(ROOT / "config.template.json")
    closure = config["runtime_package_closure"]
    assert closure["root"] == RUNTIME_PACKAGE_CLOSURE_ROOT
    assert closure["files"] == RUNTIME_PACKAGE_CLOSURE_FILES
    assert closure["files_sha256"] == canonical_sha256(RUNTIME_PACKAGE_CLOSURE_FILES)
    for relative, expected in closure["files"].items():
        path = Path(closure["root"]) / relative
        assert path.is_file()
        assert path.stat().st_size == expected["nbytes"]
        assert sha256_file(path) == expected["sha256"]
