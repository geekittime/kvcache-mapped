from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.common import ContractError, load_json_strict
from src.preflight import validate_config, validate_template


ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_template_is_valid_but_formal_execution_is_blocked():
    report = validate_template(ROOT / "config.template.json")
    assert report["status"] == "PRE_FORMAL_TEMPLATE_VALID"
    assert report["formal_execution_blocked"] is True
    assert report["gpu_execution_performed"] is False
    assert report["locked_outcomes_opened"] is False
    assert report["placeholder_count"] >= 10
    assert validate_config(ROOT / "config.template.json", expect_template_blocked=True)["placeholder_count"] == report["placeholder_count"]
    with pytest.raises(ContractError, match="intentionally blocked"):
        validate_config(ROOT / "config.template.json", expect_template_blocked=False)


def test_template_rejects_premature_gpu_authorization(tmp_path: Path):
    config = load_json_strict(ROOT / "config.template.json")
    config["experiment_root"] = str(tmp_path)
    config["outputs"] = {key: str(tmp_path / Path(value).name) for key, value in config["outputs"].items()}
    config["execution_policy"]["gpu_execution_authorized_by_this_config"] = True
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContractError, match="GPU-disabled"):
        validate_template(config_path)


def test_template_rejects_lexical_output_escape(tmp_path: Path):
    config = load_json_strict(ROOT / "config.template.json")
    config["experiment_root"] = str(tmp_path)
    config["outputs"]["runs_root"] = str(tmp_path / ".." / "escape")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContractError, match="escapes"):
        validate_template(config_path)

