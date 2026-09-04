from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import pytest

from reverse_kv.eval_common import hashed_json
from reverse_kv.protocol import EXPERIMENT_ROOT, canonical_sha256
from reverse_kv.report_amendment import (
    BASE_STAGE_NAMES,
    ReportAmendmentError,
    _amendment_document,
    _identity,
    _publish_document,
    _verify_original_freeze,
    _verify_identity,
)


def _temporary_experiment_directory() -> tempfile.TemporaryDirectory[str]:
    (EXPERIMENT_ROOT / "runs").mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(
        prefix="report_amendment_test_", dir=EXPERIMENT_ROOT / "runs"
    )


def test_amendment_binds_every_base_stage_receipt_name() -> None:
    assert BASE_STAGE_NAMES == (
        "01_preflight",
        "02_selection_target0",
        "03_selection_full",
        "04_fit_target0",
        "05_fit_full",
        "06_heldout",
        "07_hellaswag",
        "08_mechanism",
        "09_coqa",
        "10_latency",
        "11_k4_selection",
        "12_k4_fit_target0",
        "13_k4_fit_full",
        "14_k4_heldout",
        "15_k4_coqa",
    )


def test_amendment_preserves_and_reads_the_original_freeze_sidecar() -> None:
    identity, freeze = _verify_original_freeze()
    assert identity["artifact"]["sha256"] == (
        "72a3d03e63ea18fb8f5005383d161753873aa647acb6dd09f3b2d94cf39ffe96"
    )
    assert identity["inventory_digest"] == freeze["inventory_digest"]


def test_recovery_identity_rejects_payload_tamper() -> None:
    with _temporary_experiment_directory() as directory:
        path = Path(directory) / "evidence.txt"
        path.write_text("original\n", encoding="utf-8")
        identity = _identity(path)
        assert _verify_identity(identity) == path.resolve()
        path.write_text("tampered\n", encoding="utf-8")
        with pytest.raises(ReportAmendmentError, match="changed"):
            _verify_identity(identity)


def test_append_only_document_rejects_coordinated_payload_and_sidecar_rewrite() -> None:
    with _temporary_experiment_directory() as directory:
        path = Path(directory) / "amendment.json"
        original = {"schema_version": 1, "status": "complete", "value": "original"}
        _publish_document(path, original)
        # An attacker may rewrite both JSON and its SHA sidecar.  Exact
        # reconstruction must still reject the changed semantic payload.
        hashed_json(path, {**original, "value": "coordinated-tamper"})
        with pytest.raises(ReportAmendmentError, match="append-only"):
            _publish_document(path, original)


def test_amendment_policy_is_explicit_and_patch_digest_is_semantic() -> None:
    base = {"freeze": "old"}
    patch = {"patched_original_files": [{"old_sha256": "a", "new_sha256": "b"}]}
    tests = {"passed": 1, "exit_code": 0}
    report = {"receipt": "bound"}
    value = _amendment_document(base, patch, tests, report)
    assert value["recovery_policy"] == {
        "append_only": True,
        "cpu_only": True,
        "gpu_stages_rerun": False,
        "original_freeze_replaced": False,
        "original_pipeline_state_rewritten": False,
        "ordinary_stage_16_receipt_created": False,
        "base_stages_relabelled_under_hotfix_code": False,
    }
    assert value["hotfix_digest"] == canonical_sha256(
        {"patch": patch, "tests": tests}
    )
    changed = copy.deepcopy(value)
    changed["hotfix"]["patch"]["patched_original_files"][0]["new_sha256"] = "c"
    assert canonical_sha256(changed["hotfix"]) != value["hotfix_digest"]
