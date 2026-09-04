from __future__ import annotations

import copy
from pathlib import Path
import sys
import types

import pytest

from src.contracts import ContractError, atomic_json, write_sidecar
from src import validator


def _config(runs_root: Path) -> dict:
    return {
        "outputs": {"runs_root": str(runs_root)},
        "runtime": {"target_model": "unused-in-regression-test"},
        "confirmation_gate": {
            "rule_id": "screen_completed_then_confirm_v1",
            "required_screen_profile": "screen",
            "required_screen_validation_status": "validated_complete",
        },
    }


def _screen_validation(receipt_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "status": "validated_complete",
        "gpu_used_by_validator": False,
        "profile": "screen",
        "request_fingerprint": "a" * 64,
        "result_count": 1920,
        "completion_receipt_sha256": receipt_sha256,
        "code_inventory_digest": "b" * 64,
        "official_dev_opened": False,
    }


def _confirm_definition(receipt: dict, validation: dict) -> dict:
    return {
        "profile": "confirm",
        "confirmation_prerequisite": {
            "rule_id": "screen_completed_then_confirm_v1",
            "validated_screen": validation,
            "screen_completion_receipt": receipt,
        },
    }


def test_confirm_validator_accepts_the_exact_revalidated_screen_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    receipt_ref = atomic_json(
        runs_root / "screen_v1" / "completion_receipt.json", {"status": "complete"}
    )
    write_sidecar(receipt_ref)
    expected_validation = _screen_validation(receipt_ref.sha256)
    calls: list[tuple[str, str, Path]] = []

    def fake_validate(config_path, *, profile, output_dir):
        calls.append((str(config_path), profile, Path(output_dir).resolve()))
        return expected_validation

    monkeypatch.setattr(validator, "validate_completed_run", fake_validate)
    definition = _confirm_definition(receipt_ref.to_dict(), expected_validation)

    assert validator._validated_confirmation_prerequisite(
        "config.json", _config(runs_root), definition
    ) == definition["confirmation_prerequisite"]
    assert calls == [("config.json", "screen", (runs_root / "screen_v1").resolve())]


def test_completed_confirm_validator_rebuilds_with_its_frozen_prerequisite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for the old omission of the prerequisite during rebuild."""

    runs_root = tmp_path / "runs"
    screen_receipt = atomic_json(
        runs_root / "screen_v1" / "completion_receipt.json", {"status": "complete"}
    )
    write_sidecar(screen_receipt)
    screen_validation = _screen_validation(screen_receipt.sha256)
    frozen_definition = _confirm_definition(screen_receipt.to_dict(), screen_validation)
    frozen_definition["request_fingerprint"] = "c" * 64
    definition_ref = atomic_json(runs_root / "confirm_v1" / "run_definition.json", frozen_definition)
    write_sidecar(definition_ref)
    config = _config(runs_root)
    observed: dict[str, object] = {}

    real_validate = validator.validate_completed_run
    monkeypatch.setattr(
        validator,
        "validate_completed_run",
        lambda _config_path, *, profile, output_dir: screen_validation,
    )
    monkeypatch.setattr(validator, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(validator, "load_prepared", lambda *_args, **_kwargs: ([], {}, object()))
    monkeypatch.setattr(validator, "validate_method_bindings", lambda *_args, **_kwargs: object())

    def fake_build(_config, _profile, _rows, _prepared_ref, _bindings, *, confirmation_prerequisite=None):
        observed["confirmation_prerequisite"] = confirmation_prerequisite
        return frozen_definition

    monkeypatch.setattr(validator, "build_run_definition", fake_build)

    class StopBeforeCheckpoint(RuntimeError):
        pass

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise StopBeforeCheckpoint

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTokenizer))
    with pytest.raises(StopBeforeCheckpoint):
        real_validate("config.json", profile="confirm", output_dir=runs_root / "confirm_v1")
    assert observed["confirmation_prerequisite"] == frozen_definition["confirmation_prerequisite"]


def test_confirm_validator_rejects_tampered_serialized_screen_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_root = tmp_path / "runs"
    receipt_ref = atomic_json(
        runs_root / "screen_v1" / "completion_receipt.json", {"status": "complete"}
    )
    write_sidecar(receipt_ref)
    expected_validation = _screen_validation(receipt_ref.sha256)
    monkeypatch.setattr(
        validator,
        "validate_completed_run",
        lambda _config_path, *, profile, output_dir: expected_validation,
    )
    definition = _confirm_definition(receipt_ref.to_dict(), expected_validation)
    tampered = copy.deepcopy(definition)
    tampered["confirmation_prerequisite"]["validated_screen"]["request_fingerprint"] = "0" * 64

    with pytest.raises(ContractError, match="validated screen evidence changed"):
        validator._validated_confirmation_prerequisite("config.json", _config(runs_root), tampered)
