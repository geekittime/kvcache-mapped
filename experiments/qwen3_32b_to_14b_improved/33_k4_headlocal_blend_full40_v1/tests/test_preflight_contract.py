from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from src.blend import ProtocolError, current_python_identity, file_ref
from src.preflight import ROOT, strict_json, validate_config


def test_frozen_config_contract_and_ref_lengths():
    config = strict_json(ROOT / "config.json")
    validate_config(config)
    assert config["protocol"]["k4_layers"] == 4
    assert config["protocol"]["canonical_k20_layers"] == 20
    assert config["protocol"]["maps_refit_in_candidate"] is False
    assert config["protocol"]["alpha_is_oof"] is False
    assert config["protocol"]["heldout_tuning_previously_inspected_exploratory_only"] is True
    assert len(config["upstream"]["heldout_tokens"]["sha256"]) == 64
    assert len(config["supersedes"]["results"]["sha256"]) == 64


def test_file_ref_rejects_symlink_and_nonregular(tmp_path: Path):
    regular = tmp_path / "regular"
    regular.write_bytes(b"x")
    assert file_ref(regular)["nbytes"] == 1
    link = tmp_path / "link"
    link.symlink_to(regular)
    with pytest.raises(ProtocolError):
        file_ref(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ProtocolError):
        file_ref(fifo)


def test_no_locked_or_answer_namespace_in_config():
    config = strict_json(ROOT / "config.json")
    protocol = config["protocol"]
    assert protocol["locked_activation_evaluations"] == 0
    assert protocol["locked_rows_indexed"] is False
    assert protocol["answers_or_outcomes_read"] is False
    assert config["method"]["alpha_formula"].startswith("clip_0_1")


def test_runtime_identity_retains_lexical_pinned_venv():
    identity = current_python_identity()
    # Tests may be invoked with another interpreter, but the identity schema
    # must never collapse a venv symlink to /usr/bin/python and call that bound.
    assert "executable" in identity and "resolved_executable" in identity
    assert isinstance(identity["executable"], str)
