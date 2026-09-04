"""Fail-closed formal pipeline for the Qwen3 32B -> 14B reproduction.

The launcher deliberately has no implicit cleanup or overwrite path.  A
checkpoint is either absent, internally valid and resumable, or fatal.  GPU
work is serialized with the repository-wide lock used by the forward runs.
"""

from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from .eval_common import hashed_json, immutable_hashed_json, verify_hashed_json
from .freeze_spec import expected_groups as expected_freeze_groups
from .k4_heldout import K4_HELDOUT
from .k4_pipeline import (
    K4_COQA_IDLE_GATE,
    K4_COQA_ROOT,
    K4_FIT_SMOKE_EVIDENCE,
    K4_STAGE_ORDER,
    K4_TRAINING_IDLE_GATE,
    k4_stage_commands,
    k4_validator_map,
    validate_k4_coqa_idle_gate,
    validate_k4_training_idle_gate,
)
from .k4_protocol import K4_MAPPER_ROOT, K4_SELECTION
from .protocol import (
    DIRECTION,
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    atomic_write_json,
    canonical_sha256,
    read_json,
    sha256_file,
)
from .report import (
    REPORT_ROOT,
    REPORT_STAGE_NAME,
    report_stage_command,
    validate_report_stage,
)


SCHEMA_VERSION = 1
PIPELINE_ID = "qwen3_32b_to_14b_formal_v1"
VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
GLOBAL_GPU_LOCK = REPOSITORY_ROOT / "runs/gpu8_experiment.lock"
FORWARD_ROOT = (
    REPOSITORY_ROOT
    / "runs/20260824_qwen3_14b_to_32b_multinews_bottom500_full_v1"
)
FORWARD_FINGERPRINT = (
    "e6152b67f6c720b30550cd06ca6d148485ae035e0dc4d94994c88b0e7812e966"
)
PIPELINE_ROOT = EXPERIMENT_ROOT / "runs/formal_pipeline_v1"
INPUT_ROOT = EXPERIMENT_ROOT / "artifacts/inputs"
FORWARD_RECEIPT = INPUT_ROOT / "multinews500_forward_completion_receipt.json"
CODE_FREEZE = INPUT_ROOT / "code_freeze.json"
PREFLIGHT = INPUT_ROOT / "preflight.json"
SELECTION = (
    EXPERIMENT_ROOT
    / "artifacts/selection/qwen3_32b_to_14b/selected_layers.json"
)
SELECTION_STATISTICS = (
    EXPERIMENT_ROOT / "artifacts/statistics/qwen3_32b_to_14b/selection"
)
MAPPER_ROOT = (
    EXPERIMENT_ROOT / "artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01"
)
HELDOUT = (
    EXPERIMENT_ROOT
    / "artifacts/validation/qwen3_32b_to_14b/heldout_activation_reconstruction.json"
)
HELLASWAG_ROOT = PIPELINE_ROOT / "hellaswag_full_k20"
MECHANISM_ROOT = PIPELINE_ROOT / "mechanism_hellaswag_k20"
COQA_ROOT = PIPELINE_ROOT / "coqa_reverse_k20"
LATENCY_ROOT = PIPELINE_ROOT / "reverse_latency_k20"
COQA_FILE = REPOSITORY_ROOT / "data/coqa/coqa_validation.parquet"
COQA_SHA256 = "5ab4d352067c30ef4925efb0dfb2abeeed4cdfb236304aa7ee9df62ea781281f"
COQA_NBYTES = 2_221_232
COQA_REVISION = "5153eac35fe087e44d9ce99ebb2d94c3dbc81583"
MS_HUB = Path("/home/wby/.venvs/modelscope/bin/ms-hub")
SMOKE_LIMIT_MIB = 18 * 1024
PAPER_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)


class PipelineError(RuntimeError):
    """A fail-closed pipeline invariant was violated."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _artifact(path: Path, *, base: Path = EXPERIMENT_ROOT) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        relative = str(resolved.relative_to(base.resolve()))
    except ValueError:
        relative = str(resolved)
    return {
        "path": str(resolved),
        "relative_path": relative,
        "sha256": sha256_file(resolved),
        "nbytes": resolved.stat().st_size,
    }


def _verify_artifact(row: Mapping[str, Any]) -> Path:
    path = Path(str(row.get("path", ""))).resolve()
    if not path.is_file():
        raise PipelineError(f"recorded artifact is missing: {path}")
    if path.stat().st_size != row.get("nbytes") or sha256_file(path) != row.get("sha256"):
        raise PipelineError(f"recorded artifact drifted: {path}")
    return path


def _hashed_sidecar(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256.json")


def verify_forward_completion(*, publish: bool = True) -> dict[str, Any]:
    """Verify all 1,000 formal forward checkpoints and publish a stable receipt."""

    root = FORWARD_ROOT.resolve()
    control = {name: read_json(root / name) for name in (
        "run_definition.json", "status.json", "summary.json", "results.manifest.json"
    )}
    definition = control["run_definition.json"]
    status = control["status.json"]
    summary = control["summary.json"]
    manifest = control["results.manifest.json"]
    if definition.get("request_fingerprint") != FORWARD_FINGERPRINT:
        raise PipelineError("MultiNews run definition fingerprint changed")
    for label, value in (("status", status), ("summary", summary), ("manifest", manifest)):
        if value.get("status") != "complete":
            raise PipelineError(f"MultiNews {label} is not complete")
        if value.get("request_fingerprint") != FORWARD_FINGERPRINT:
            raise PipelineError(f"MultiNews {label} fingerprint changed")
    if status.get("failures") != 0:
        raise PipelineError("MultiNews completed with failures")
    files = manifest.get("files")
    if manifest.get("expected_result_count") != 1000 or not isinstance(files, list) or len(files) != 1000:
        raise PipelineError("MultiNews result inventory is not exactly 1,000 files")
    seen: set[str] = set()
    mode_counts = {"target_standalone": 0, "mapped": 0}
    verified: list[dict[str, Any]] = []
    for raw in files:
        if not isinstance(raw, Mapping):
            raise PipelineError("MultiNews result manifest has a malformed row")
        relative = str(raw.get("path", ""))
        path = (root / relative).resolve()
        if root not in path.parents or relative in seen:
            raise PipelineError(f"invalid or duplicate MultiNews result path: {relative}")
        seen.add(relative)
        if not path.is_file() or path.stat().st_size != raw.get("nbytes"):
            raise PipelineError(f"MultiNews result is absent or changed size: {path}")
        digest = sha256_file(path)
        if digest != raw.get("sha256"):
            raise PipelineError(f"MultiNews result SHA256 mismatch: {path}")
        payload = read_json(path)
        if payload.get("status") != "complete" or payload.get("request_fingerprint") != FORWARD_FINGERPRINT:
            raise PipelineError(f"MultiNews result scientific identity changed: {path}")
        mode = str(payload.get("mode", ""))
        if mode not in mode_counts or payload.get("task") != "multi_news":
            raise PipelineError(f"MultiNews result mode/task changed: {path}")
        mode_counts[mode] += 1
        verified.append({"path": relative, "sha256": digest, "nbytes": path.stat().st_size})
    if mode_counts != {"target_standalone": 500, "mapped": 500}:
        raise PipelineError(f"MultiNews mode coverage changed: {mode_counts}")
    manifest_identity = summary.get("results_manifest")
    if not isinstance(manifest_identity, Mapping):
        raise PipelineError("MultiNews summary lacks its result-manifest identity")
    manifest_path = root / "results.manifest.json"
    if manifest_identity.get("sha256") != sha256_file(manifest_path) or manifest_identity.get("nbytes") != manifest_path.stat().st_size:
        raise PipelineError("MultiNews summary/result-manifest binding changed")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "kind": "qwen3_14b_to_32b_multinews_bottom500_completion",
        "request_fingerprint": FORWARD_FINGERPRINT,
        "run_root": str(root),
        "expected_results": 1000,
        "mode_counts": mode_counts,
        "result_inventory_sha256": canonical_sha256(verified),
        "controls": {name: _artifact(root / name, base=root) for name in control},
    }
    if publish:
        immutable_hashed_json(FORWARD_RECEIPT, receipt)
    return receipt


def wait_for_forward_completion(
    *, timeout_seconds: float = 7 * 24 * 60 * 60, poll_seconds: float = 60.0
) -> dict[str, Any]:
    """Wait for the exact managed MultiNews run; fail immediately on drift/failure."""

    deadline = time.monotonic() + timeout_seconds
    definition_path = FORWARD_ROOT / "run_definition.json"
    status_path = FORWARD_ROOT / "status.json"
    while True:
        if definition_path.is_file():
            definition = read_json(definition_path)
            if definition.get("request_fingerprint") != FORWARD_FINGERPRINT:
                raise PipelineError("MultiNews run definition fingerprint changed while waiting")
        if status_path.is_file():
            status = read_json(status_path)
            if status.get("request_fingerprint") != FORWARD_FINGERPRINT:
                raise PipelineError("MultiNews status fingerprint changed while waiting")
            current = status.get("status")
            if current == "failed":
                raise PipelineError("MultiNews formal run failed while the reverse queue waited")
            if current == "complete":
                return verify_forward_completion(publish=True)
        if time.monotonic() >= deadline:
            raise PipelineError("timed out waiting for the formal MultiNews completion receipt")
        time.sleep(poll_seconds)


def verify_code_freeze() -> dict[str, Any]:
    if not CODE_FREEZE.is_file():
        raise PipelineError("final code_freeze.json has not been published")
    sidecar = CODE_FREEZE.with_suffix(".sha256.json")
    if not sidecar.is_file():
        raise PipelineError("code freeze SHA256 commit marker is missing")
    declared = read_json(sidecar)
    if declared.get("sha256") != sha256_file(CODE_FREEZE):
        raise PipelineError("code freeze commit marker disagrees with the payload")
    value = read_json(CODE_FREEZE)
    if value.get("status") != "frozen" or value.get("direction") != DIRECTION:
        raise PipelineError("code freeze direction/status changed")
    digest = value.get("inventory_digest")
    unsigned = dict(value)
    unsigned.pop("inventory_digest", None)
    if digest != canonical_sha256(unsigned):
        raise PipelineError("code freeze inventory digest is invalid")
    entries: list[Mapping[str, Any]] = []
    actual_groups: dict[str, set[str]] = {}
    freeze_groups = (
        "local_files",
        "read_only_shared_dependencies",
        "benchmark_protocol_dependencies",
        "vendored_harness_dependencies",
    )
    for key in freeze_groups:
        rows = value.get(key)
        if not isinstance(rows, list) or not rows:
            raise PipelineError(f"code freeze lacks {key}")
        group_paths: set[str] = set()
        for row in rows:
            if not isinstance(row, Mapping):
                raise PipelineError("code freeze contains a malformed file row")
            path = _verify_artifact(row)
            allowed_root = EXPERIMENT_ROOT if key == "local_files" else REPOSITORY_ROOT
            try:
                path.relative_to(allowed_root.resolve())
            except ValueError as exc:
                raise PipelineError(f"code freeze {key} path escaped its root: {path}") from exc
            if str(path) in group_paths:
                raise PipelineError(f"duplicate code freeze path inside {key}: {path}")
            group_paths.add(str(path))
            entries.append(row)
        actual_groups[key] = group_paths
    paths: set[str] = set()
    for row in entries:
        path = _verify_artifact(row)
        if str(path) in paths:
            raise PipelineError(f"duplicate code freeze path: {path}")
        paths.add(str(path))
    expected_groups = {
        key: {str(path.resolve()) for path in group}
        for key, group in expected_freeze_groups().items()
    }
    for key in freeze_groups:
        if actual_groups[key] != expected_groups[key]:
            missing = sorted(expected_groups[key] - actual_groups[key])
            extra = sorted(actual_groups[key] - expected_groups[key])
            raise PipelineError(
                f"code freeze {key} coverage changed; missing={missing}, extra={extra}"
            )
    return {"sha256": sha256_file(CODE_FREEZE), "inventory_digest": digest, "files": len(entries)}


def _validate_pipeline_idle_gate(path: Path, *, expected_name: str) -> None:
    """Recompute the three launcher-owned physical-idle gates."""

    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "name": expected_name,
        "required_consecutive_samples": 12,
        "memory_limit_mib": 64,
        "sample_interval_seconds": 5.0,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise PipelineError(
                f"pipeline GPU idle gate {expected_name} field {field} changed"
            )
    observations = value.get("observations")
    if not isinstance(observations, list) or len(observations) < 12:
        raise PipelineError(
            f"pipeline GPU idle gate {expected_name} lacks twelve samples"
        )
    for observation in observations[-12:]:
        if not isinstance(observation, Mapping) or observation.get("passed") is not True:
            raise PipelineError(
                f"pipeline GPU idle gate {expected_name} lacks twelve final passes"
            )
        memory = observation.get("memory")
        processes = observation.get("processes")
        if (
            not isinstance(memory, list)
            or len(memory) != 8
            or {row.get("index") for row in memory if isinstance(row, Mapping)}
            != set(range(8))
            or any(
                not isinstance(row, Mapping)
                or int(row.get("memory_used_mib", 65)) > 64
                for row in memory
            )
            or processes != []
        ):
            raise PipelineError(
                f"pipeline GPU idle gate {expected_name} contains a non-idle sample"
            )


def _gate_artifacts(
    path: Path, *, pipeline_gate_name: str | None = None
) -> list[dict[str, Any]]:
    if pipeline_gate_name is not None:
        _validate_pipeline_idle_gate(path, expected_name=pipeline_gate_name)
    verify_hashed_json(path)
    value = read_json(path)
    if value.get("status") != "passed":
        raise PipelineError(f"GPU idle gate is not passed: {path}")
    return [_artifact(path), _artifact(_hashed_sidecar(path))]


def required_gpu_gate_artifacts() -> list[dict[str, Any]]:
    """Return every independently hashed idle gate required by the pipeline."""

    paths = (
        (
            PIPELINE_ROOT / "gates/training_segment_idle.json",
            "training_segment_idle",
        ),
        (
            PIPELINE_ROOT / "gates/hellaswag_segment_idle.json",
            "hellaswag_segment_idle",
        ),
        (HELLASWAG_ROOT / "gpu_idle_gate.json", None),
        (MECHANISM_ROOT / "gpu_idle_gate.json", None),
        (
            PIPELINE_ROOT / "gates/coqa_segment_idle.json",
            "coqa_segment_idle",
        ),
        (LATENCY_ROOT / "gpu_idle_gate.json", None),
        (K4_TRAINING_IDLE_GATE, None),
        (K4_COQA_IDLE_GATE, None),
    )
    artifacts: list[dict[str, Any]] = []
    for path, pipeline_gate_name in paths:
        artifacts.extend(
            _gate_artifacts(path, pipeline_gate_name=pipeline_gate_name)
        )
    return artifacts


def coqa_download_command() -> list[str]:
    return [
        str(MS_HUB), "download", "EleutherAI/coqa",
        "coqa_validation.parquet",
        "--repo-type", "dataset", "--revision", COQA_REVISION,
        "--local-dir", str(COQA_FILE.parent), "--max-workers", "8",
    ]


def verify_coqa_file() -> dict[str, Any] | None:
    if not COQA_FILE.is_file():
        return None
    if COQA_FILE.stat().st_size != COQA_NBYTES or sha256_file(COQA_FILE) != COQA_SHA256:
        raise PipelineError("pinned CoQA parquet exists but fails size/SHA256")
    return _artifact(COQA_FILE, base=REPOSITORY_ROOT)


def _query_gpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    memory = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        check=False, capture_output=True, text=True,
    )
    if memory.returncode or processes.returncode:
        raise PipelineError("nvidia-smi GPU query failed")
    memory_rows = []
    for line in memory.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 3:
            raise PipelineError("malformed nvidia-smi memory row")
        memory_rows.append({"index": int(pieces[0]), "uuid": pieces[1], "memory_used_mib": int(pieces[2])})
    process_rows = []
    for line in processes.stdout.splitlines():
        if not line.strip():
            continue
        pieces = [piece.strip() for piece in line.split(",")]
        if len(pieces) != 3:
            raise PipelineError("malformed nvidia-smi process row")
        used = None if pieces[2] in {"N/A", "[Not Supported]"} else int(pieces[2])
        process_rows.append({"uuid": pieces[0], "pid": int(pieces[1]), "used_memory_mib": used})
    return memory_rows, process_rows


@contextmanager
def repository_gpu_lock() -> Iterator[None]:
    GLOBAL_GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with GLOBAL_GPU_LOCK.open("a+b") as handle:
        try:
            # This is a persistent queue, so a valid managed predecessor is a
            # wait condition rather than a failure.  The idle gate below still
            # rejects unmanaged GPU users after the lock is acquired.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            raise PipelineError("failed to acquire the repository GPU lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def wait_for_idle(
    *, name: str, interval_seconds: float = 5.0, timeout_seconds: float = 86_400.0
) -> dict[str, Any]:
    observations = []
    consecutive = 0
    deadline = time.monotonic() + timeout_seconds
    while consecutive < 12:
        if time.monotonic() >= deadline:
            raise PipelineError(f"timed out waiting for eight idle GPUs: {name}")
        memory, processes = _query_gpus()
        passed = (
            len(memory) == 8
            and {row["index"] for row in memory} == set(range(8))
            and not processes
            and all(row["memory_used_mib"] <= 64 for row in memory)
        )
        consecutive = consecutive + 1 if passed else 0
        observations.append({"utc": utc_now(), "passed": passed, "memory": memory, "processes": processes})
        if consecutive < 12:
            time.sleep(interval_seconds)
    evidence = {
        "schema_version": SCHEMA_VERSION, "status": "passed", "name": name,
        "required_consecutive_samples": 12, "memory_limit_mib": 64,
        "sample_interval_seconds": interval_seconds,
        "timeout_seconds": timeout_seconds, "observations": observations,
    }
    path = PIPELINE_ROOT / "gates" / f"{name}.json"
    hashed_json(path, evidence)
    return _artifact(path)


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_command(
    command: Sequence[str],
    *,
    stage: str,
    monitor_fit_smoke: bool = False,
    fit_smoke_evidence_path: Path | None = None,
) -> dict[str, Any] | None:
    logs = PIPELINE_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{stage}.log"
    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    environment.update(CUDA_VISIBLE_DEVICES=VISIBLE_DEVICES, PYTHONUNBUFFERED="1")
    samples: list[dict[str, Any]] = []
    peak = 0
    observed_owner = False
    owner_identities: set[tuple[str, int]] = set()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] command={json.dumps(list(command))}\n")
        handle.flush()
        process = subprocess.Popen(
            list(command), cwd=REPOSITORY_ROOT, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            while process.poll() is None:
                if monitor_fit_smoke:
                    memory, processes = _query_gpus()
                    by_uuid = {row["uuid"]: row["index"] for row in memory}
                    occupied = [{**row, "index": by_uuid.get(row["uuid"])} for row in processes]
                    if any(row["index"] != 0 for row in occupied) or len(occupied) > 1:
                        raise PipelineError("fit smoke lost single-owner GPU0 isolation")
                    if any(row["index"] != 0 and row["memory_used_mib"] > 64 for row in memory):
                        raise PipelineError("fit smoke observed unexpected memory on GPUs 1-7")
                    owner_identities.update((str(row["uuid"]), int(row["pid"])) for row in occupied)
                    if len(owner_identities) > 1:
                        raise PipelineError("fit smoke observed more than one compute owner over time")
                    observed_owner = observed_owner or bool(occupied)
                    gpu0 = next(row for row in memory if row["index"] == 0)
                    peak = max(peak, int(gpu0["memory_used_mib"]))
                    samples.append({"utc": utc_now(), "memory": memory, "processes": occupied})
                time.sleep(0.25 if monitor_fit_smoke else 1.0)
        except BaseException:
            _terminate_process(process)
            raise
        returncode = process.wait()
    if returncode:
        raise PipelineError(f"stage {stage} failed with exit code {returncode}; see {log}")
    if not monitor_fit_smoke:
        return None
    evidence = {
        "schema_version": SCHEMA_VERSION, "status": "passed", "stage": stage,
        "single_compute_owner": True, "compute_owner_observed": observed_owner,
        "compute_owner_identities": [list(value) for value in sorted(owner_identities)],
        "allowed_gpu": 0, "peak_memory_used_mib": peak,
        "maximum_allowed_peak_mib": SMOKE_LIMIT_MIB, "samples": samples,
    }
    if not observed_owner:
        raise PipelineError("fit smoke produced no observable GPU compute owner")
    if peak > SMOKE_LIMIT_MIB:
        raise PipelineError(f"fit smoke peak {peak} MiB exceeds {SMOKE_LIMIT_MIB} MiB")
    path = fit_smoke_evidence_path or (
        PIPELINE_ROOT / "evidence/fit_target0_gpu_monitor.json"
    )
    try:
        path.resolve().relative_to(PIPELINE_ROOT.resolve())
    except ValueError as exc:
        raise PipelineError("fit-smoke evidence path escaped the pipeline root") from exc
    hashed_json(path, evidence)
    return evidence


def _validate_preflight() -> list[dict[str, Any]] | None:
    if not PREFLIGHT.is_file():
        return None
    value = read_json(PREFLIGHT)
    if value.get("status") != "passed" or value.get("direction") != DIRECTION:
        raise PipelineError("reverse full-payload preflight is invalid")
    # A CPU-only preliminary receipt may exist from code preparation.  It is
    # not a formal checkpoint and is deliberately upgraded in place by the
    # --verify-payloads command.
    if value.get("payload_checksums_verified") is False:
        return None
    if value.get("payload_checksums_verified") is not True:
        raise PipelineError("reverse preflight payload-verification flag is malformed")
    captures = value.get("captures")
    if not isinstance(captures, Mapping) or len(captures) != 4 or any(row.get("payload_checksums_verified") is not True for row in captures.values()):
        raise PipelineError("reverse preflight did not hash every activation payload")
    return [_artifact(PREFLIGHT), _artifact(PREFLIGHT.with_suffix(".sha256.json"))]


def _validate_selection_smoke() -> list[dict[str, Any]] | None:
    payload = SELECTION_STATISTICS / "single_source/target_layer_000.npz"
    sidecar = payload.with_suffix(".json")
    if not payload.exists() and not sidecar.exists():
        return None
    if not payload.is_file() or not sidecar.is_file():
        raise PipelineError("selection target0 smoke is only partially committed")
    raw = read_json(sidecar)
    if raw.get("target_layer") != 0 or raw.get("kind") != "single_source_target_layer":
        raise PipelineError("selection target0 smoke identity changed")
    if raw.get("sha256") != sha256_file(payload) or raw.get("nbytes") != payload.stat().st_size:
        raise PipelineError("selection target0 NPZ commit marker changed")
    with np.load(payload, allow_pickle=False) as archive:
        if not archive.files or any(not np.isfinite(archive[name]).all() for name in archive.files if archive[name].dtype.kind == "f"):
            raise PipelineError("selection target0 NPZ is empty or non-finite")
    return [_artifact(payload), _artifact(sidecar)]


def _validate_selection_full() -> list[dict[str, Any]] | None:
    if not SELECTION.exists():
        return None
    from .selection_adapter import validate_reverse_selection_artifact
    validate_reverse_selection_artifact(SELECTION)
    return [_artifact(SELECTION), _artifact(SELECTION.with_suffix(".sha256.json"))]


def _validate_fit_monitor(path: Path, *, expected_stage: str) -> dict[str, Any]:
    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "stage": expected_stage,
        "single_compute_owner": True,
        "compute_owner_observed": True,
        "allowed_gpu": 0,
        "maximum_allowed_peak_mib": SMOKE_LIMIT_MIB,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise PipelineError(f"fit-smoke monitor field {field} changed")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise PipelineError("fit-smoke monitor has no physical GPU samples")
    owners: set[tuple[str, int]] = set()
    observed_owner = False
    recomputed_peak = 0
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise PipelineError("fit-smoke monitor contains a malformed sample")
        memory = sample.get("memory")
        processes = sample.get("processes")
        if (
            not isinstance(memory, list)
            or len(memory) != 8
            or {row.get("index") for row in memory if isinstance(row, Mapping)}
            != set(range(8))
            or not isinstance(processes, list)
            or len(processes) > 1
        ):
            raise PipelineError("fit-smoke monitor GPU inventory changed")
        if any(
            not isinstance(row, Mapping)
            or (row.get("index") != 0 and int(row.get("memory_used_mib", 65)) > 64)
            for row in memory
        ):
            raise PipelineError("fit-smoke monitor observed memory on GPUs 1-7")
        gpu0 = next(row for row in memory if row.get("index") == 0)
        recomputed_peak = max(recomputed_peak, int(gpu0.get("memory_used_mib", -1)))
        for process in processes:
            if not isinstance(process, Mapping) or process.get("index") != 0:
                raise PipelineError("fit-smoke monitor observed a non-GPU0 owner")
            identity = (str(process.get("uuid", "")), int(process.get("pid", -1)))
            if not identity[0] or identity[1] < 1:
                raise PipelineError("fit-smoke monitor owner identity is malformed")
            owners.add(identity)
            observed_owner = True
    recorded_owners = value.get("compute_owner_identities")
    if (
        observed_owner is not True
        or len(owners) != 1
        or recorded_owners != [list(identity) for identity in sorted(owners)]
        or value.get("peak_memory_used_mib") != recomputed_peak
        or recomputed_peak > SMOKE_LIMIT_MIB
    ):
        raise PipelineError("fit-smoke monitor aggregate evidence changed")
    return value


def _validate_fit_smoke() -> list[dict[str, Any]] | None:
    payload = MAPPER_ROOT / "layers/layer_000.safetensors"
    evidence = PIPELINE_ROOT / "evidence/fit_target0_gpu_monitor.json"
    if not payload.exists() and not evidence.exists():
        return None
    if not payload.is_file() or not evidence.is_file():
        raise PipelineError("fit target0 smoke lacks mapper or GPU evidence")
    _validate_fit_monitor(evidence, expected_stage="04_fit_target0")
    selection = read_json(SELECTION)
    row = selection.get("selected_source_layers_rank_order", [None])[0]
    sidecar = read_json(payload.with_suffix(".json"))
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PipelineError("fit target0 layer metadata is missing")
    from .fit_adapter import _validate_layer_payload
    _validate_layer_payload(
        MAPPER_ROOT, target_layer=0, selected_layers=row,
        source_features=20_480, target_features=1_024, kv_heads=8, head_dim=128,
        fit_fingerprint=str(metadata.get("fit_fingerprint")),
        selection_sha256=sha256_file(SELECTION),
    )
    result = [_artifact(payload), _artifact(payload.with_suffix(".json")), _artifact(evidence), _artifact(_hashed_sidecar(evidence))]
    for cache_type in ("k_content", "v"):
        component = MAPPER_ROOT / "components" / cache_type / "layer_000.safetensors"
        result.extend((_artifact(component), _artifact(component.with_suffix(".json"))))
    return result


def _validate_fit_full() -> list[dict[str, Any]] | None:
    manifest = MAPPER_ROOT / "manifest.json"
    if not manifest.exists():
        return None
    from .eval_common import mapper_manifest_identity
    identity = mapper_manifest_identity(manifest)
    value = read_json(manifest)
    if identity.get("target_layer_count") != 40 or value.get("weight_count") != 1_677_721_600 or value.get("bias_count") != 81_920:
        raise PipelineError("complete mapper aggregate counts changed")
    validation = value.get("payload_validation")
    if not isinstance(validation, Mapping) or validation.get("status") != "passed" or validation.get("layers_actually_loaded") != 40:
        raise PipelineError("complete mapper payload audit is missing")
    return [_artifact(manifest), _artifact(MAPPER_ROOT / "manifest.sha256.json")]


def _validate_heldout() -> list[dict[str, Any]] | None:
    if not HELDOUT.exists():
        return None
    from .heldout import validate_heldout_artifact

    value = validate_heldout_artifact(HELDOUT)
    if (
        value.get("status") != "complete"
        or value.get("direction") != DIRECTION
        or len(value.get("layers", ())) != 40
    ):
        raise PipelineError("held-out reconstruction artifact is invalid")
    sidecar = HELDOUT.with_suffix(".sha256.json")
    if read_json(sidecar).get("sha256") != sha256_file(HELDOUT):
        raise PipelineError("held-out reconstruction SHA marker changed")
    return [_artifact(HELDOUT), _artifact(sidecar)]


def _validate_hellaswag() -> list[dict[str, Any]] | None:
    receipt = HELLASWAG_ROOT / "formal_hellaswag_receipt.json"
    if not receipt.exists():
        return None
    from .hellaswag_eval import _validate_existing_output
    _validate_existing_output(HELLASWAG_ROOT)
    verify_hashed_json(receipt)
    return [
        _artifact(receipt),
        _artifact(_hashed_sidecar(receipt)),
        *_gate_artifacts(HELLASWAG_ROOT / "gpu_idle_gate.json"),
    ]


def _validate_mechanism() -> list[dict[str, Any]] | None:
    receipt = MECHANISM_ROOT / "formal_mechanism_receipt.json"
    if not receipt.exists():
        return None
    from .mechanism import validate_completed_output
    value = validate_completed_output(MECHANISM_ROOT)
    if value is None:
        raise PipelineError("mechanism receipt exists but validation did not complete")
    result = MECHANISM_ROOT / "mechanism_result.json"
    checkpoint = Path(str(value["final_checkpoint"]["path"]))
    return [
        _artifact(receipt),
        _artifact(_hashed_sidecar(receipt)),
        _artifact(result),
        _artifact(_hashed_sidecar(result)),
        _artifact(checkpoint),
        _artifact(checkpoint.with_name(f"{checkpoint.name}.sha256.json")),
        *_gate_artifacts(MECHANISM_ROOT / "gpu_idle_gate.json"),
    ]


def _validate_coqa() -> list[dict[str, Any]] | None:
    manifest = COQA_ROOT / "manifest.json"
    if not manifest.exists():
        return None
    from .coqa_eval import validate_completed_output as validate_coqa_completed_output

    value = validate_coqa_completed_output(COQA_ROOT)
    if value is None:
        raise PipelineError("CoQA manifest exists but strict validation is incomplete")
    definition_path = COQA_ROOT / "run_definition.json"
    summary = COQA_ROOT / "summary.json"
    raw = COQA_ROOT / "raw_records.jsonl"
    runtime_path = COQA_ROOT / "runtime_setup.json"
    raw_sidecar_path = _hashed_sidecar(raw)
    return [
        _artifact(manifest), _artifact(_hashed_sidecar(manifest)),
        _artifact(definition_path), _artifact(_hashed_sidecar(definition_path)),
        _artifact(summary), _artifact(_hashed_sidecar(summary)),
        _artifact(raw), _artifact(raw_sidecar_path),
        _artifact(runtime_path), _artifact(_hashed_sidecar(runtime_path)),
        *_gate_artifacts(
            PIPELINE_ROOT / "gates/coqa_segment_idle.json",
            pipeline_gate_name="coqa_segment_idle",
        ),
    ]


def _validate_latency() -> list[dict[str, Any]] | None:
    manifest = LATENCY_ROOT / "manifest.json"
    if not manifest.exists():
        return None
    from .reverse_latency import verify_latency_completion

    value = verify_latency_completion(LATENCY_ROOT)
    if value is None:
        raise PipelineError("latency formal completion receipt is missing")
    staging = []
    for row in value["staging_receipts"]:
        path = LATENCY_ROOT / str(row["path"])
        staging.extend((_artifact(path), _artifact(_hashed_sidecar(path))))
    return [
        _artifact(manifest), _artifact(_hashed_sidecar(manifest)),
        _artifact(LATENCY_ROOT / "formal_latency_receipt.json"),
        _artifact(_hashed_sidecar(LATENCY_ROOT / "formal_latency_receipt.json")),
        _artifact(LATENCY_ROOT / "summary.json"),
        _artifact(_hashed_sidecar(LATENCY_ROOT / "summary.json")),
        _artifact(LATENCY_ROOT / "endpoint_memory_smoke.json"),
        _artifact(_hashed_sidecar(LATENCY_ROOT / "endpoint_memory_smoke.json")),
        _artifact(LATENCY_ROOT / "runtime_setup.json"),
        _artifact(_hashed_sidecar(LATENCY_ROOT / "runtime_setup.json")),
        *staging,
        *_gate_artifacts(LATENCY_ROOT / "gpu_idle_gate.json"),
    ]


Validator = Callable[[], list[dict[str, Any]] | None]


class FormalPipeline:
    def __init__(self, *, python: Path | None = None) -> None:
        # Do not resolve the venv launcher symlink: on this host it points at
        # /usr/bin/python3.12, whose import environment does not contain the
        # frozen experiment dependencies.
        self.python = Path(os.path.abspath(os.fspath(python or sys.executable)))
        self.definition: dict[str, Any] = {}
        self.definition_digest = ""
        self.state_path = PIPELINE_ROOT / "pipeline_state.json"
        self.receipt_root = PIPELINE_ROOT / "stage_receipts"

    def _write_state(self, status: str, **extra: Any) -> None:
        prior: dict[str, Any] = {}
        if self.state_path.is_file():
            verify_hashed_json(self.state_path)
            prior = read_json(self.state_path)
            if prior.get("definition_digest") != self.definition_digest:
                raise PipelineError("pipeline state belongs to another definition")
        value = {**prior, "schema_version": SCHEMA_VERSION, "pipeline_id": PIPELINE_ID,
                 "direction": DIRECTION, "definition_digest": self.definition_digest,
                 "status": status, "updated_utc": utc_now(), **extra}
        hashed_json(self.state_path, value)

    def _receipt_path(self, stage: str) -> Path:
        return self.receipt_root / f"{stage}.json"

    def _stage(
        self,
        stage: str,
        command: Sequence[str],
        validator: Validator,
        *,
        monitor: bool = False,
        monitor_evidence_path: Path | None = None,
    ) -> list[dict[str, Any]]:
        receipt_path = self._receipt_path(stage)
        if receipt_path.is_file():
            verify_hashed_json(receipt_path)
            receipt = read_json(receipt_path)
            if receipt.get("status") != "complete" or receipt.get("stage") != stage or receipt.get("definition_digest") != self.definition_digest or receipt.get("command") != list(command):
                raise PipelineError(f"stage receipt identity changed: {stage}")
            artifacts = validator()
            if (
                artifacts is None
                or receipt.get("artifacts") != artifacts
                or canonical_sha256(artifacts) != receipt.get("artifacts_sha256")
            ):
                raise PipelineError(f"completed stage artifacts drifted: {stage}")
            return artifacts
        existing = validator()
        execution = "adopted_valid_existing"
        if existing is None:
            execution = "executed"
            self._write_state("running", active_stage=stage)
            run_command(
                command,
                stage=stage,
                monitor_fit_smoke=monitor,
                fit_smoke_evidence_path=monitor_evidence_path,
            )
            existing = validator()
            if existing is None:
                raise PipelineError(f"stage exited zero without accepted artifacts: {stage}")
        receipt = {
            "schema_version": SCHEMA_VERSION, "status": "complete", "stage": stage,
            "definition_digest": self.definition_digest, "command": list(command),
            "execution": execution, "artifacts": existing,
            "artifacts_sha256": canonical_sha256(existing),
        }
        immutable_hashed_json(receipt_path, receipt)
        self._write_state("running", active_stage=None, last_completed_stage=stage,
                          last_stage_receipt_sha256=sha256_file(receipt_path))
        return existing

    def _initialize(self) -> None:
        forward = wait_for_forward_completion()
        freeze = verify_code_freeze()
        definition = {
            "schema_version": SCHEMA_VERSION, "pipeline_id": PIPELINE_ID,
            "direction": DIRECTION, "python": str(self.python),
            "cuda_visible_devices": VISIBLE_DEVICES,
            "global_gpu_lock": str(GLOBAL_GPU_LOCK),
            "forward_request_fingerprint": FORWARD_FINGERPRINT,
            "forward_receipt_sha256": sha256_file(FORWARD_RECEIPT),
            "forward_inventory_sha256": forward["result_inventory_sha256"],
            "code_freeze_sha256": freeze["sha256"],
            "code_inventory_digest": freeze["inventory_digest"],
            "smoke_peak_limit_mib": SMOKE_LIMIT_MIB,
            "idle_gate_policy": {
                "memory_limit_mib": 64,
                "required_consecutive_samples": 12,
                "sample_interval_seconds": 5,
                "all_gate_payloads_and_sha_sidecars_bound_in_completion_receipt": True,
            },
            "outputs": {
                "selection": str(SELECTION), "mapper": str(MAPPER_ROOT),
                "heldout": str(HELDOUT), "hellaswag": str(HELLASWAG_ROOT),
                "mechanism": str(MECHANISM_ROOT),
                "coqa": str(COQA_ROOT), "latency": str(LATENCY_ROOT),
                "k4_selection": str(K4_SELECTION),
                "k4_mapper": str(K4_MAPPER_ROOT),
                "k4_heldout": str(K4_HELDOUT),
                "k4_coqa": str(K4_COQA_ROOT),
                "final_report": str(REPORT_ROOT),
            },
        }
        self.definition = definition
        self.definition_digest = canonical_sha256(definition)
        PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
        immutable_hashed_json(PIPELINE_ROOT / "pipeline_definition.json", definition)
        if self.state_path.is_file():
            verify_hashed_json(self.state_path)
            if read_json(self.state_path).get("definition_digest") != self.definition_digest:
                raise PipelineError("existing pipeline state belongs to another definition")
        else:
            self._write_state("initialized")

    def _ensure_coqa(self) -> None:
        identity = verify_coqa_file()
        command = coqa_download_command()
        receipt = INPUT_ROOT / "coqa_dataset_receipt.json"
        if receipt.is_file():
            verify_hashed_json(receipt)
            value = read_json(receipt)
            if identity is None or value.get("dataset") != identity or value.get("download_command") != command:
                raise PipelineError("CoQA dataset receipt or payload drifted")
            return
        if identity is None:
            if not MS_HUB.is_file():
                raise PipelineError(f"ModelScope ms-hub executable is missing: {MS_HUB}")
            COQA_FILE.parent.mkdir(parents=True, exist_ok=True)
            self._write_state("downloading_coqa")
            run_command(command, stage="coqa_download")
            identity = verify_coqa_file()
            if identity is None:
                raise PipelineError("ModelScope download completed without the pinned CoQA file")
        immutable_hashed_json(receipt, {
            "schema_version": SCHEMA_VERSION, "status": "complete",
            "provider": "ModelScope", "revision": COQA_REVISION,
            "download_command": command, "dataset": identity,
        })

    def _commands(self) -> dict[str, list[str]]:
        py = str(self.python)
        scripts = EXPERIMENT_ROOT / "scripts"
        commands = {
            "01_preflight": [py, str(scripts / "prepare.py"), "--verify-payloads", "--output", str(PREFLIGHT)],
            "02_selection_target0": [py, str(scripts / "select_layers.py"), "--devices", "0", "--chunk-tokens", "1024", "--only-target-layers", "0", "--ols-fallback", "error"],
            "03_selection_full": [py, str(scripts / "select_layers.py"), "--devices", VISIBLE_DEVICES, "--chunk-tokens", "1024", "--ols-fallback", "error"],
            "04_fit_target0": [py, str(scripts / "fit_mapper.py"), "--devices", "0", "--target-layers", "0", "--chunk-tokens", "1024", "--solver", "cholesky", "--analysis-device", "cpu", "--cpu-threads-per-worker", "2"],
            "05_fit_full": [py, str(scripts / "fit_mapper.py"), "--devices", VISIBLE_DEVICES, "--chunk-tokens", "1024", "--solver", "cholesky", "--analysis-device", "cpu", "--cpu-threads-per-worker", "2"],
            "06_heldout": [py, str(scripts / "validate_heldout.py"), "--compute-device", "cuda:0", "--analysis-device", "cpu", "--chunk-samples", "1024"],
            "07_hellaswag": [py, str(scripts / "evaluate_hellaswag.py"), "--output-dir", str(HELLASWAG_ROOT)],
            "08_mechanism": [py, str(scripts / "evaluate_mechanism.py"), "--output-dir", str(MECHANISM_ROOT)],
            "09_coqa": [py, str(scripts / "evaluate_coqa.py"), "--output-dir", str(COQA_ROOT)],
            "10_latency": [py, str(scripts / "benchmark_latency.py"), "--output-dir", str(LATENCY_ROOT)],
        }
        commands.update(k4_stage_commands(self.python))
        if tuple(commands)[-len(K4_STAGE_ORDER):] != K4_STAGE_ORDER:
            raise PipelineError("k4 formal stage order changed")
        commands[REPORT_STAGE_NAME] = report_stage_command(self.python)
        return commands

    def _validators(self) -> dict[str, Validator]:
        validators: dict[str, Validator] = {
            "01_preflight": _validate_preflight,
            "02_selection_target0": _validate_selection_smoke,
            "03_selection_full": _validate_selection_full,
            "04_fit_target0": _validate_fit_smoke,
            "05_fit_full": _validate_fit_full,
            "06_heldout": _validate_heldout,
            "07_hellaswag": _validate_hellaswag,
            "08_mechanism": _validate_mechanism,
            "09_coqa": _validate_coqa,
            "10_latency": _validate_latency,
        }
        validators.update(k4_validator_map())
        validators[REPORT_STAGE_NAME] = validate_report_stage
        return validators

    def _validated_stage_receipts(
        self, commands: Mapping[str, Sequence[str]]
    ) -> list[dict[str, Any]]:
        validators = self._validators()
        if tuple(validators) != tuple(commands):
            raise PipelineError("formal command/validator stage inventory changed")
        rows: list[dict[str, Any]] = []
        for stage, command in commands.items():
            path = self._receipt_path(stage)
            verify_hashed_json(path)
            receipt = read_json(path)
            if set(receipt) != {
                "schema_version",
                "status",
                "stage",
                "definition_digest",
                "command",
                "execution",
                "artifacts",
                "artifacts_sha256",
            }:
                raise PipelineError(f"stage receipt schema changed: {stage}")
            if (
                receipt.get("schema_version") != SCHEMA_VERSION
                or receipt.get("status") != "complete"
                or receipt.get("stage") != stage
                or receipt.get("definition_digest") != self.definition_digest
                or receipt.get("command") != list(command)
                or receipt.get("execution")
                not in {"executed", "adopted_valid_existing"}
            ):
                raise PipelineError(f"stage receipt identity changed: {stage}")
            artifacts = validators[stage]()
            if artifacts is None:
                raise PipelineError(f"completed stage no longer validates: {stage}")
            if (
                receipt.get("artifacts") != artifacts
                or receipt.get("artifacts_sha256") != canonical_sha256(artifacts)
            ):
                raise PipelineError(f"stage receipt artifacts changed: {stage}")
            rows.append(_artifact(path))
        return rows

    def _expected_final_artifacts(self) -> list[dict[str, Any]]:
        validate_k4_training_idle_gate()
        validate_k4_coqa_idle_gate()
        return [
            _artifact(SELECTION),
            _artifact(MAPPER_ROOT / "manifest.json"),
            _artifact(HELDOUT),
            _artifact(HELLASWAG_ROOT / "formal_hellaswag_receipt.json"),
            _artifact(MECHANISM_ROOT / "formal_mechanism_receipt.json"),
            _artifact(COQA_ROOT / "manifest.json"),
            _artifact(LATENCY_ROOT / "manifest.json"),
            _artifact(K4_SELECTION),
            _artifact(K4_MAPPER_ROOT / "manifest.json"),
            _artifact(K4_HELDOUT),
            _artifact(K4_COQA_ROOT / "formal_k4_coqa_receipt.json"),
            _artifact(REPORT_ROOT / "formal_report_receipt.json"),
            *required_gpu_gate_artifacts(),
        ]

    def _validate_forward_receipt(self) -> dict[str, Any]:
        verify_hashed_json(FORWARD_RECEIPT)
        stored = read_json(FORWARD_RECEIPT)
        recomputed = verify_forward_completion(publish=False)
        if stored != recomputed:
            raise PipelineError("forward completion receipt semantic content changed")
        return stored

    def _completion_payload(
        self, commands: Mapping[str, Sequence[str]]
    ) -> dict[str, Any]:
        freeze = verify_code_freeze()
        self._validate_forward_receipt()
        stage_rows = self._validated_stage_receipts(commands)
        final_artifacts = self._expected_final_artifacts()
        verify_hashed_json(self.state_path)
        state = read_json(self.state_path)
        if (
            state.get("schema_version") != SCHEMA_VERSION
            or state.get("pipeline_id") != PIPELINE_ID
            or state.get("direction") != DIRECTION
            or state.get("definition_digest") != self.definition_digest
            or state.get("status") != "complete"
            or state.get("active_stage") is not None
            or state.get("completed_stages") != list(commands)
        ):
            raise PipelineError("completed pipeline state semantic content changed")
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "kind": "qwen3_32b_to_14b_formal_pipeline_completion",
            "pipeline_id": PIPELINE_ID,
            "definition_digest": self.definition_digest,
            "freeze_sha256": freeze["sha256"],
            "freeze_inventory_digest": freeze["inventory_digest"],
            "forward_completion_receipt": _artifact(FORWARD_RECEIPT),
            "pipeline_state": _artifact(self.state_path),
            "stage_receipts": stage_rows,
            "stage_receipts_sha256": canonical_sha256(stage_rows),
            "final_artifacts": final_artifacts,
            "final_artifacts_sha256": canonical_sha256(final_artifacts),
        }

    def _validate_completion_receipt(self) -> bool:
        path = PIPELINE_ROOT / "pipeline_completion_receipt.json"
        if not path.is_file():
            return False
        verify_hashed_json(path)
        receipt = read_json(path)
        expected = self._completion_payload(self._commands())
        if receipt != expected:
            raise PipelineError("pipeline completion receipt semantic content changed")
        return True

    def run(self) -> dict[str, Any]:
        self._initialize()
        if self._validate_completion_receipt():
            return read_json(PIPELINE_ROOT / "pipeline_completion_receipt.json")
        self._ensure_coqa()
        commands = self._commands()
        try:
            with repository_gpu_lock():
                wait_for_idle(name="training_segment_idle")
                self._stage("01_preflight", commands["01_preflight"], _validate_preflight)
                self._stage("02_selection_target0", commands["02_selection_target0"], _validate_selection_smoke)
                self._stage("03_selection_full", commands["03_selection_full"], _validate_selection_full)
                self._stage("04_fit_target0", commands["04_fit_target0"], _validate_fit_smoke, monitor=True)
                self._stage("05_fit_full", commands["05_fit_full"], _validate_fit_full)
                self._stage("06_heldout", commands["06_heldout"], _validate_heldout)
                # Prove the training/validation processes released every CUDA
                # context before handing the lock to HellaSwag's sealed entry.
                wait_for_idle(name="hellaswag_segment_idle")
            # HellaSwag owns the same global lock internally; do not nest it.
            self._stage("07_hellaswag", commands["07_hellaswag"], _validate_hellaswag)
            # Mechanism diagnostics bind the completed HellaSwag retention and
            # independently own the global lock plus physical idle gate.
            self._stage("08_mechanism", commands["08_mechanism"], _validate_mechanism)
            with repository_gpu_lock():
                wait_for_idle(name="coqa_segment_idle")
                self._stage("09_coqa", commands["09_coqa"], _validate_coqa)
            # Latency owns the same lock and its own 12-sample idle gate.
            self._stage("10_latency", commands["10_latency"], _validate_latency)
            with repository_gpu_lock():
                wait_for_idle(name="k4_training_segment_idle")
                validators = k4_validator_map()
                self._stage(
                    "11_k4_selection",
                    commands["11_k4_selection"],
                    validators["11_k4_selection"],
                )
                self._stage(
                    "12_k4_fit_target0",
                    commands["12_k4_fit_target0"],
                    validators["12_k4_fit_target0"],
                    monitor=True,
                    monitor_evidence_path=K4_FIT_SMOKE_EVIDENCE,
                )
                self._stage(
                    "13_k4_fit_full",
                    commands["13_k4_fit_full"],
                    validators["13_k4_fit_full"],
                )
                self._stage(
                    "14_k4_heldout",
                    commands["14_k4_heldout"],
                    validators["14_k4_heldout"],
                )
                wait_for_idle(name="coqa_k4_segment_idle")
                self._stage(
                    "15_k4_coqa",
                    commands["15_k4_coqa"],
                    validators["15_k4_coqa"],
                )
            self._stage(
                REPORT_STAGE_NAME,
                commands[REPORT_STAGE_NAME],
                validate_report_stage,
            )
        except BaseException as exc:
            self._write_state("failed", error_type=type(exc).__name__, error=str(exc))
            raise

        self._write_state("complete", active_stage=None, completed_stages=list(commands))
        completion = self._completion_payload(commands)
        path = PIPELINE_ROOT / "pipeline_completion_receipt.json"
        immutable_hashed_json(path, completion)
        return completion


def main(argv: Sequence[str] | None = None) -> int:
    if argv:
        raise PipelineError("formal pipeline accepts no mutable CLI overrides")
    result = FormalPipeline().run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "FormalPipeline", "PipelineError", "coqa_download_command", "main",
    "verify_code_freeze", "verify_coqa_file", "verify_forward_completion",
    "wait_for_forward_completion", "wait_for_idle",
]
