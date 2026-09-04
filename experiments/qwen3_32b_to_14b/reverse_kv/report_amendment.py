"""Append-only, CPU-only recovery for the failed formal report stage.

Stages 01--15 were executed under the original code freeze and pipeline
definition.  The original report-only implementation then failed before it
published any scientific output.  This module deliberately does not rewrite
that history: it binds the original freeze, definition, receipts, failed
state, and failure log, then publishes a separately named amended report and
three separately named recovery documents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import re
from typing import Any

from .eval_common import immutable_hashed_json, verify_hashed_json
from .protocol import (
    EXPERIMENT_ROOT,
    canonical_sha256,
    read_json,
    sha256_file,
)
from .report import (
    COMPLETE_FILENAMES,
    MANIFEST_FILENAME,
    RECEIPT_FILENAME,
    REPORT_ROOT,
    STAGE_RECEIPT_ROOT,
    _validate_report_bundle,
    collect_formal_results,
    publish_report_bundle,
    validate_report_output,
)


SCHEMA_VERSION = 1
PIPELINE_ROOT = EXPERIMENT_ROOT / "runs/formal_pipeline_v1"
AMENDED_REPORT_ROOT = PIPELINE_ROOT / "final_report_amended_v1"
AMENDMENT_PATH = PIPELINE_ROOT / "report_hotfix_amendment.json"
AMENDED_STAGE_RECEIPT = STAGE_RECEIPT_ROOT / "16_final_report_amended.json"
AMENDED_COMPLETION = PIPELINE_ROOT / "pipeline_completion_amended.json"
TEST_LOG = PIPELINE_ROOT / "report_hotfix_tests.log"

CODE_FREEZE = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
CODE_FREEZE_SIDECAR = CODE_FREEZE.with_suffix(".sha256.json")
PIPELINE_DEFINITION = PIPELINE_ROOT / "pipeline_definition.json"
FAILED_STATE = PIPELINE_ROOT / "pipeline_state.json"
FAILED_LOG = PIPELINE_ROOT / "logs/16_final_report.log"

BASE_STAGE_NAMES = (
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

PATCHED_FILES = (
    EXPERIMENT_ROOT / "reverse_kv/report.py",
    EXPERIMENT_ROOT / "tests/test_report.py",
)
AMENDMENT_IMPLEMENTATION_FILES = (
    EXPERIMENT_ROOT / "reverse_kv/report_amendment.py",
    EXPERIMENT_ROOT / "scripts/build_report_amendment.py",
    EXPERIMENT_ROOT / "tests/test_report_amendment.py",
)
TEST_COMMAND = (
    os.path.abspath(EXPERIMENT_ROOT.parents[1] / ".venv/bin/python"),
    "-m",
    "pytest",
    "-q",
    "-p",
    "no:cacheprovider",
    "experiments/qwen3_32b_to_14b/tests",
)


class ReportAmendmentError(ValueError):
    """An append-only report recovery invariant was violated."""


def _sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256.json")


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        relative = str(resolved.relative_to(EXPERIMENT_ROOT.resolve()))
    except ValueError as exc:
        raise ReportAmendmentError(f"recovery artifact escaped experiment: {resolved}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "relative_path": relative,
        "sha256": sha256_file(resolved),
        "nbytes": resolved.stat().st_size,
    }


def _verify_identity(row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    expected = _identity(path)
    if dict(row) != expected:
        raise ReportAmendmentError(f"recovery-bound artifact changed: {path}")
    return path


def _identity_pair(path: Path) -> dict[str, Any]:
    verify_hashed_json(path)
    return {"artifact": _identity(path), "checksum": _identity(_sidecar(path))}


def _verify_identity_pair(value: Mapping[str, Any]) -> Path:
    if set(value) != {"artifact", "checksum"}:
        raise ReportAmendmentError("hashed recovery identity has a changed schema")
    artifact = _verify_identity(value["artifact"])
    checksum = _verify_identity(value["checksum"])
    if checksum != _sidecar(artifact).resolve():
        raise ReportAmendmentError("hashed recovery identity points at the wrong sidecar")
    verify_hashed_json(artifact)
    return artifact


def _verify_original_freeze() -> tuple[dict[str, Any], Mapping[str, Any]]:
    freeze = read_json(CODE_FREEZE)
    sidecar = read_json(CODE_FREEZE_SIDECAR)
    expected_sidecar = {"path": str(CODE_FREEZE), "sha256": sha256_file(CODE_FREEZE)}
    if sidecar != expected_sidecar:
        raise ReportAmendmentError("original code-freeze sidecar changed")
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "frozen"
        or not isinstance(freeze.get("inventory_digest"), str)
        or not isinstance(freeze.get("local_files"), list)
    ):
        raise ReportAmendmentError("original code freeze is malformed")
    return {
        "artifact": _identity(CODE_FREEZE),
        "checksum": _identity(CODE_FREEZE_SIDECAR),
        "inventory_digest": freeze["inventory_digest"],
    }, freeze


def _verify_pipeline_definition(
    freeze_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    verify_hashed_json(PIPELINE_DEFINITION)
    definition = read_json(PIPELINE_DEFINITION)
    digest = canonical_sha256(definition)
    if (
        definition.get("direction") != "qwen3_32b_to_14b"
        or definition.get("code_freeze_sha256")
        != freeze_identity["artifact"]["sha256"]
        or definition.get("code_inventory_digest")
        != freeze_identity["inventory_digest"]
    ):
        raise ReportAmendmentError("base pipeline definition is not bound to the original freeze")
    return {
        **_identity_pair(PIPELINE_DEFINITION),
        "definition_digest": digest,
    }, digest


def _collect_base_receipts(definition_digest: str) -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for stage in BASE_STAGE_NAMES:
        path = STAGE_RECEIPT_ROOT / f"{stage}.json"
        pair = _identity_pair(path)
        value = read_json(path)
        if (
            value.get("schema_version") != 1
            or value.get("status") != "complete"
            or value.get("stage") != stage
            or value.get("definition_digest") != definition_digest
            or value.get("artifacts_sha256")
            != canonical_sha256(value.get("artifacts"))
        ):
            raise ReportAmendmentError(f"base stage receipt changed: {stage}")
        receipts[stage] = pair
    return {
        "stages": list(BASE_STAGE_NAMES),
        "receipts": receipts,
        "receipt_inventory_sha256": canonical_sha256(receipts),
    }


def _failure_evidence() -> dict[str, Any]:
    state = read_json(FAILED_STATE)
    if (
        state.get("status") != "failed"
        or state.get("active_stage") != "16_final_report"
        or state.get("last_completed_stage") != "15_k4_coqa"
        or state.get("error_type") != "PipelineError"
    ):
        raise ReportAmendmentError("base failure state no longer describes stage 16")
    log_text = FAILED_LOG.read_text(encoding="utf-8")
    required = (
        "command=",
        "reverse_kv/report.py",
        "AttributeError: 'ArchitectureSpec' object has no attribute 'kv_heads'",
    )
    if any(fragment not in log_text for fragment in required):
        raise ReportAmendmentError("stage-16 failure log no longer proves the original error")
    return {
        "pipeline_state": _identity(FAILED_STATE),
        "stage_log": _identity(FAILED_LOG),
        "failed_stage": "16_final_report",
        "last_completed_stage": "15_k4_coqa",
        "original_exception": "ArchitectureSpec missing fit-validator attribute kv_heads",
    }


def _old_local_file_rows(freeze: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in freeze["local_files"]:
        if not isinstance(row, Mapping):
            raise ReportAmendmentError("original freeze local-file row is malformed")
        rows[str(Path(str(row.get("path", ""))).resolve())] = row
    return rows


def _patch_evidence(freeze: Mapping[str, Any]) -> dict[str, Any]:
    old_rows = _old_local_file_rows(freeze)
    files = []
    for path in PATCHED_FILES:
        resolved = path.resolve()
        old = old_rows.get(str(resolved))
        if old is None:
            raise ReportAmendmentError(f"patched file absent from original freeze: {resolved}")
        current = _identity(resolved)
        if old.get("sha256") == current["sha256"]:
            raise ReportAmendmentError(f"hotfix file did not change: {resolved}")
        files.append(
            {
                "path": str(resolved),
                "relative_path": current["relative_path"],
                "old_sha256": old.get("sha256"),
                "old_nbytes": old.get("nbytes"),
                "new_sha256": current["sha256"],
                "new_nbytes": current["nbytes"],
            }
        )
    implementation = [_identity(path) for path in AMENDMENT_IMPLEMENTATION_FILES]
    return {
        "scope": "report_only_no_scientific_gpu_stage_code_or_artifact_changed",
        "patched_original_files": files,
        "amendment_implementation_files": implementation,
        "patch_digest": canonical_sha256(files),
        "implementation_digest": canonical_sha256(implementation),
    }


_PYTEST_SUMMARY = re.compile(r"(?m)^(\d+) passed(?:, \d+ warning[s]?)? in .+$")


def _test_evidence() -> dict[str, Any]:
    text = TEST_LOG.read_text(encoding="utf-8")
    matches = _PYTEST_SUMMARY.findall(text)
    if not matches or " failed" in text or " error" in text.lower():
        raise ReportAmendmentError("hotfix test log does not prove a clean full-suite run")
    return {
        "command": list(TEST_COMMAND),
        "exit_code": 0,
        "scope": "full_reverse_cpu_test_suite",
        "passed": int(matches[-1]),
        "log": _identity(TEST_LOG),
    }


def _base_evidence() -> tuple[dict[str, Any], Mapping[str, Any]]:
    freeze_identity, freeze = _verify_original_freeze()
    definition, definition_digest = _verify_pipeline_definition(freeze_identity)
    receipts = _collect_base_receipts(definition_digest)
    return {
        "original_code_freeze": freeze_identity,
        "original_pipeline_definition": definition,
        "base_stage_receipts_01_through_15": receipts,
        "original_failure": _failure_evidence(),
    }, freeze


def _report_evidence() -> dict[str, Any]:
    return {
        "root": str(AMENDED_REPORT_ROOT.resolve()),
        "receipt": _identity_pair(AMENDED_REPORT_ROOT / RECEIPT_FILENAME),
        "manifest": _identity(AMENDED_REPORT_ROOT / MANIFEST_FILENAME),
        "files": [_identity(AMENDED_REPORT_ROOT / name) for name in COMPLETE_FILENAMES],
    }


def _amendment_document(
    base: Mapping[str, Any],
    patch: Mapping[str, Any],
    tests: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    hotfix = {"patch": patch, "tests": tests}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_report_hotfix_amendment",
        "direction": "qwen3_32b_to_14b",
        "recovery_policy": {
            "append_only": True,
            "cpu_only": True,
            "gpu_stages_rerun": False,
            "original_freeze_replaced": False,
            "original_pipeline_state_rewritten": False,
            "ordinary_stage_16_receipt_created": False,
            "base_stages_relabelled_under_hotfix_code": False,
        },
        "base": dict(base),
        "hotfix": hotfix,
        "hotfix_digest": canonical_sha256(hotfix),
        "amended_report": dict(report),
    }


def _amended_stage_receipt(amendment: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = [_identity(AMENDED_REPORT_ROOT / name) for name in COMPLETE_FILENAMES]
    artifacts.extend((_identity(AMENDMENT_PATH), _identity(_sidecar(AMENDMENT_PATH))))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_amended_stage_receipt",
        "stage": "16_final_report_amended",
        "execution": "executed_cpu_only_report_hotfix",
        "base_definition_digest": amendment["base"]["original_pipeline_definition"][
            "definition_digest"
        ],
        "hotfix_digest": amendment["hotfix_digest"],
        "ordinary_definition_digest_claimed": False,
        "command": [
            os.path.abspath(EXPERIMENT_ROOT.parents[1] / ".venv/bin/python"),
            str((EXPERIMENT_ROOT / "scripts/build_report_amendment.py").resolve()),
        ],
        "amendment": _identity_pair(AMENDMENT_PATH),
        "artifacts": artifacts,
        "artifacts_sha256": canonical_sha256(artifacts),
    }


def _amended_completion(
    amendment: Mapping[str, Any], stage_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_with_report_hotfix_amendment",
        "kind": "qwen3_32b_to_14b_pipeline_completion_amended",
        "direction": "qwen3_32b_to_14b",
        "base_definition_digest": stage_receipt["base_definition_digest"],
        "base_stage_receipt_inventory_sha256": amendment["base"]
        ["base_stage_receipts_01_through_15"]["receipt_inventory_sha256"],
        "original_code_freeze": amendment["base"]["original_code_freeze"],
        "original_failure": amendment["base"]["original_failure"],
        "hotfix_digest": amendment["hotfix_digest"],
        "amendment": _identity_pair(AMENDMENT_PATH),
        "amended_stage_receipt": _identity_pair(AMENDED_STAGE_RECEIPT),
        "amended_report_receipt": _identity_pair(
            AMENDED_REPORT_ROOT / RECEIPT_FILENAME
        ),
        "gpu_stages_rerun": False,
        "original_pipeline_state_preserved_failed": True,
    }


def _publish_document(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        verify_hashed_json(path)
        if read_json(path) != dict(value):
            raise ReportAmendmentError(f"append-only recovery artifact changed: {path}")
        return
    immutable_hashed_json(path, value)


def _assert_standard_history_untouched() -> None:
    ordinary_receipt = STAGE_RECEIPT_ROOT / "16_final_report.json"
    if REPORT_ROOT.exists() or ordinary_receipt.exists() or _sidecar(ordinary_receipt).exists():
        raise ReportAmendmentError(
            "ordinary stage-16 output/receipt exists; amended recovery refuses to overwrite history"
        )


def build_formal_report_amendment() -> dict[str, Any]:
    """Build the independent report and append-only recovery receipts."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ReportAmendmentError("report amendment requires CUDA_VISIBLE_DEVICES='' exactly")
    _assert_standard_history_untouched()
    before, freeze = _base_evidence()
    patch = _patch_evidence(freeze)
    tests = _test_evidence()
    formal_report = collect_formal_results()
    publish_report_bundle(AMENDED_REPORT_ROOT, formal_report)
    report = _report_evidence()
    amendment = _amendment_document(before, patch, tests, report)
    _publish_document(AMENDMENT_PATH, amendment)
    stage = _amended_stage_receipt(amendment)
    _publish_document(AMENDED_STAGE_RECEIPT, stage)
    completion = _amended_completion(amendment, stage)
    _publish_document(AMENDED_COMPLETION, completion)
    after, _ = _base_evidence()
    if after != before:
        raise ReportAmendmentError("original pipeline evidence changed during recovery")
    _validate_report_bundle(AMENDED_REPORT_ROOT, formal_report)
    return validate_formal_report_amendment(deep_report=False)


def validate_formal_report_amendment(*, deep_report: bool = True) -> dict[str, Any]:
    """Reconstruct and validate every append-only recovery binding."""

    _assert_standard_history_untouched()
    base, freeze = _base_evidence()
    patch = _patch_evidence(freeze)
    tests = _test_evidence()
    amendment = read_json(AMENDMENT_PATH)
    verify_hashed_json(AMENDMENT_PATH)
    expected_amendment = _amendment_document(base, patch, tests, _report_evidence())
    if amendment != expected_amendment:
        raise ReportAmendmentError("report hotfix amendment changed")
    stage = read_json(AMENDED_STAGE_RECEIPT)
    verify_hashed_json(AMENDED_STAGE_RECEIPT)
    if stage != _amended_stage_receipt(amendment):
        raise ReportAmendmentError("amended stage-16 receipt changed")
    completion = read_json(AMENDED_COMPLETION)
    verify_hashed_json(AMENDED_COMPLETION)
    if completion != _amended_completion(amendment, stage):
        raise ReportAmendmentError("amended pipeline completion changed")
    for pair in (
        stage["amendment"],
        completion["amendment"],
        completion["amended_stage_receipt"],
        completion["amended_report_receipt"],
    ):
        _verify_identity_pair(pair)
    for row in stage["artifacts"]:
        _verify_identity(row)
    if stage["artifacts_sha256"] != canonical_sha256(stage["artifacts"]):
        raise ReportAmendmentError("amended stage artifact inventory changed")
    if deep_report:
        validate_report_output(AMENDED_REPORT_ROOT)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv or ())
    if arguments not in ([], ["--validate-only"]):
        raise ReportAmendmentError("report amendment accepts only --validate-only")
    result = (
        validate_formal_report_amendment(deep_report=True)
        if arguments == ["--validate-only"]
        else build_formal_report_amendment()
    )
    print(
        f"status={result['status']} completion={AMENDED_COMPLETION} "
        f"report={AMENDED_REPORT_ROOT}"
    )
    return 0


__all__ = [
    "AMENDED_COMPLETION",
    "AMENDED_REPORT_ROOT",
    "AMENDED_STAGE_RECEIPT",
    "AMENDMENT_PATH",
    "BASE_STAGE_NAMES",
    "ReportAmendmentError",
    "build_formal_report_amendment",
    "validate_formal_report_amendment",
    "_identity",
    "_verify_identity",
]
