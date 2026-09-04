"""CPU-gated, resumable downstream probe for the frozen old-03 pilot.

The module deliberately imports only the Python standard library at import
time.  Validation and preflight therefore cannot create a CUDA context.  GPU
and model dependencies are imported lazily by :func:`execute` after all
content-addressed inputs, source bytes, and the eight-card idle gate have
passed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import time
import traceback
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path("/home/wby/llm-cache/nvidia-kvcache").resolve()
OLD03_ROOT = REPO_ROOT / "experiments/qwen3_32b_to_14b_improved/03_tail_refresh"
SHRINKAGE_ROOT = REPO_ROOT / "experiments/qwen3_32b_to_14b_improved/02_shrinkage_mapper"
BASELINE_RUN_ROOT = OLD03_ROOT / "runs/method_dev_pilot_r1_r8_r32_v2"
BASELINE_MODES = {"native14": "target_standalone", "nvidia_k20": "mapped_r1"}
DATASETS = ("hotpotqa_train", "musique_train")
METRIC_NAMES = (
    "relaxed_qa_f1",
    "relaxed_qa_exact_match",
    "normalized_gold_containment",
    "musique_official_answer_f1",
    "musique_official_answer_em",
)

# Make the repository's namespace packages importable only when the lazy GPU
# path is entered.  This is a path operation, not a torch/CUDA import.
for _path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


class ProtocolError(RuntimeError):
    """A frozen input, output, or execution contract is invalid."""


def _absolute_unresolved_path(path: str | Path) -> Path:
    """Return an absolute spelling without erasing symlink evidence."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    # Parent traversal makes a component-wise containment check ambiguous.
    if ".." in candidate.parts:
        raise ProtocolError(f"non-canonical parent traversal in path: {candidate}")
    return candidate


def _lstat_component_chain(
    path: str | Path,
    *,
    label: str,
    allow_missing_tail: bool,
) -> Path:
    """Reject live and dangling symlinks before any ``Path.resolve`` call.

    A prospective output may have an absent suffix, but every existing
    component is still inspected with ``lstat``.  This closes the common
    ``resolve()`` pitfall where a symlink is silently normalized away.
    """

    candidate = _absolute_unresolved_path(path)
    current = Path(candidate.anchor)
    for index, part in enumerate(candidate.parts[1:]):
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            if allow_missing_tail:
                return candidate
            raise ProtocolError(f"{label} component is absent: {current}") from exc
        except OSError as exc:
            raise ProtocolError(f"cannot lstat {label} at {current}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError(
                f"{label} traverses a live or dangling symlink: {current}"
            )
    return candidate


def _assert_output_path(
    path: str | Path,
    *,
    label: str,
    output_root: Path | None = None,
    allow_missing_tail: bool = True,
) -> Path:
    """Check component-wise non-symlink and lexical containment for outputs."""

    candidate = _lstat_component_chain(
        path, label=label, allow_missing_tail=allow_missing_tail
    )
    # The experiment root itself is fixed and must not be traversed through a
    # symlink.  ``ROOT`` is already resolved from this source file.
    root = ROOT.resolve(strict=True)
    if output_root is None:
        containment_root = root
    else:
        containment_root = _lstat_component_chain(
            output_root,
            label="output root",
            allow_missing_tail=False,
        ).resolve(strict=True)
    resolved = candidate.resolve(strict=False)
    if resolved != candidate:
        raise ProtocolError(f"{label} changed during symlink-safe resolution: {candidate}")
    try:
        resolved.relative_to(containment_root)
    except ValueError as exc:
        raise ProtocolError(
            f"{label} escapes containment root {containment_root}: {candidate}"
        ) from exc
    return candidate


def _assert_output_root(path: str | Path, *, allow_missing: bool = True) -> Path:
    """Validate the configured run root and keep it inside this experiment."""

    candidate = _assert_output_path(
        path,
        label="configured output root",
        allow_missing_tail=allow_missing,
    )
    if candidate == ROOT or candidate == ROOT.resolve():
        raise ProtocolError("configured output root cannot be the experiment root")
    return candidate


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"{label} is not a regular non-symlink file: {path}")
    return path.resolve()


def artifact(path: str | Path) -> dict[str, Any]:
    resolved = _regular(Path(path), "artifact")
    return {
        "path": str(resolved),
        "nbytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify_ref(ref: Mapping[str, Any], label: str) -> Path:
    if set(ref) != {"path", "nbytes", "sha256"}:
        raise ProtocolError(f"{label} artifact reference schema changed")
    # Inspect every existing parent with lstat before resolving.  A final-file
    # check alone still permits a symlinked directory to redirect an upstream
    # reference between preflight and execution.
    candidate = _lstat_component_chain(
        Path(str(ref["path"])), label=label, allow_missing_tail=False
    )
    path = _regular(candidate, label)
    if type(ref["nbytes"]) is not int or path.stat().st_size != ref["nbytes"]:
        raise ProtocolError(f"{label} byte size changed: {path}")
    if not is_sha256(ref["sha256"]) or sha256_file(path) != ref["sha256"]:
        raise ProtocolError(f"{label} SHA256 changed: {path}")
    return path


def atomic_json(path: str | Path, value: Any) -> dict[str, Any]:
    """Write strict JSON with fsync and atomic rename; return its artifact ref."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.tmp.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        return artifact(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def immutable_json(path: str | Path, value: Any) -> dict[str, Any]:
    destination = Path(path)
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ProtocolError(f"immutable artifact differs: {destination}")
    else:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.tmp.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return artifact(destination)


def write_sidecar(path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    sidecar = destination.with_name(destination.name + ".sha256.json")
    value = {"path": destination.name, "nbytes": destination.stat().st_size, "sha256": sha256_file(destination)}
    return immutable_json(sidecar, value)


def verify_sidecar(path: str | Path) -> None:
    destination = _regular(Path(path), "sidecar target")
    sidecar = destination.with_name(destination.name + ".sha256.json")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ProtocolError(f"sidecar is missing or not a regular file: {sidecar}")
    value = read_json(sidecar)
    expected = {"path": destination.name, "nbytes": destination.stat().st_size, "sha256": sha256_file(destination)}
    if value != expected:
        raise ProtocolError(f"sidecar mismatch: {destination}")


def token_ids_sha256(values: Sequence[int]) -> str:
    return canonical_sha256([int(value) for value in values])


def model_shard_inventory(model_entry: Mapping[str, Any], label: str) -> dict[str, Any]:
    """Hash every safetensors shard named by a model index before CUDA import."""

    model_root = Path(str(model_entry["path"])).resolve()
    index_path = verify_ref(model_entry["weight_index"], f"{label} weight index")
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, Mapping) or not weight_map or not isinstance(metadata, Mapping):
        raise ProtocolError(f"{label} model index is malformed")
    names = sorted({str(value) for value in weight_map.values()})
    if not names or any(not name.endswith(".safetensors") for name in names):
        raise ProtocolError(f"{label} model shard list is malformed")
    shards: list[dict[str, Any]] = []
    for name in names:
        shard = (model_root / name).resolve()
        try:
            shard.relative_to(model_root)
        except ValueError as exc:
            raise ProtocolError(f"{label} shard escapes model root: {name}") from exc
        shards.append(artifact(shard))
    expected_total = int(metadata.get("total_size", -1))
    observed_total = sum(int(item["nbytes"]) for item in shards)
    # ``metadata.total_size`` is tensor payload bytes, whereas filesystem
    # sizes also include safetensors headers.  Require the latter to cover the
    # declared payload and retain both values for auditability.
    if observed_total < expected_total:
        raise ProtocolError(f"{label} shard byte total is below declared tensor bytes: {observed_total} < {expected_total}")
    supporting: list[dict[str, Any]] = []
    # These are the small local assets AutoTokenizer/AutoConfig may consume;
    # binding them prevents a silent tokenizer/config drift between preflight
    # and execution without ever hashing a second, unindexed weight file.
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "configuration.json"):
        candidate = model_root / name
        if candidate.exists():
            supporting.append(artifact(candidate))
    # Include supporting loader assets in the same content digest as the
    # indexed shards.  A tokenizer/config drift must invalidate the freeze too.
    inventory_files = [*shards, *supporting]
    return {
        "model_root": str(model_root),
        "weight_index": artifact(index_path),
        "metadata_total_size": expected_total,
        "filesystem_total_size": observed_total,
        "shards": shards,
        "supporting_files": supporting,
        "digest": canonical_sha256(inventory_files),
    }


def _validate_prior_candidate_inventory(
    prior_attempt: Mapping[str, Any],
    prior_root: Path,
) -> dict[str, Any]:
    """Validate the append-only candidate inventory from the failed attempt.

    Dir25 must never silently accept an unbound subset of dir23 outputs.  The
    inventory is therefore content-addressed itself, every one of its 96
    result files is re-hashed, and the canonical list digest is recomputed.
    This runs during CPU config validation, before any CUDA/model import.
    """

    inventory_ref = prior_attempt.get("candidate_inventory")
    if not isinstance(inventory_ref, Mapping) or set(inventory_ref) != {"path", "nbytes", "sha256"}:
        raise ProtocolError("prior attempt candidate inventory binding is malformed")
    inventory_path = verify_ref(inventory_ref, "prior attempt candidate inventory")
    verify_sidecar(inventory_path)
    # The inventory is an immutable artifact from the current/parent attempt,
    # while all candidate result files it names must remain in the frozen
    # dir23 run tree.  Permit a parent-attempt inventory path but keep it
    # inside the repository and reject any symlink traversal below.
    try:
        inventory_path.relative_to(REPO_ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ProtocolError("prior candidate inventory is outside the repository") from exc
    inventory = read_json(inventory_path)
    if (
        inventory.get("schema_version") != 1
        or inventory.get("status") != "complete"
        or inventory.get("source_attempt") != "23_shrinkage_old_pilot_eval_v1"
        or inventory.get("candidate_mode") != "shrinkage_r1"
    ):
        raise ProtocolError("prior candidate inventory identity changed")
    refs = inventory.get("results")
    count = inventory.get("count")
    if type(count) is not int or count != 96 or not isinstance(refs, list) or len(refs) != count:
        raise ProtocolError("prior candidate inventory cardinality changed")
    if not is_sha256(inventory.get("results_sha256")) or canonical_sha256(refs) != inventory["results_sha256"]:
        raise ProtocolError("prior candidate inventory canonical digest changed")
    expected_results_root = prior_root / "runs/pilot_shrinkage_r1_attempt23_v1/results"
    expected_results_root = expected_results_root.resolve(strict=True)
    seen_paths: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    dataset_counts: dict[str, int] = {}
    fingerprints: set[str] = set()
    for index, ref in enumerate(refs):
        if not isinstance(ref, Mapping) or set(ref) != {"path", "nbytes", "sha256"}:
            raise ProtocolError(f"prior candidate result reference {index} is malformed")
        raw_path = Path(str(ref.get("path")))
        checked = _lstat_component_chain(raw_path, label="prior candidate result", allow_missing_tail=False)
        resolved = checked.resolve(strict=True)
        if checked != resolved:
            raise ProtocolError(f"prior candidate result path is not canonical: {checked}")
        try:
            relative = resolved.relative_to(expected_results_root)
        except ValueError as exc:
            raise ProtocolError(f"prior candidate result escapes dir23 results: {resolved}") from exc
        if len(relative.parts) != 3 or relative.parts[2].split(".")[-1] != "json":
            raise ProtocolError(f"prior candidate result has an unexpected path: {resolved}")
        if relative.parts[1] != "shrinkage_r1" or relative.parts[2].endswith(".sha256.json") or len(relative.parts[2].removesuffix(".json")) != 20:
            raise ProtocolError(f"prior candidate result mode/sidecar changed: {resolved}")
        key_path = str(resolved)
        if key_path in seen_paths:
            raise ProtocolError("prior candidate inventory contains duplicate paths")
        seen_paths.add(key_path)
        verify_ref(ref, f"prior candidate result {index}")
        value = read_json(resolved)
        if value.get("status") != "complete" or value.get("mode") != "shrinkage_r1":
            raise ProtocolError(f"prior candidate result {index} is not complete shrinkage_r1")
        dataset = value.get("dataset_id")
        sample = value.get("sample_id")
        if dataset not in DATASETS or not isinstance(sample, str) or not sample:
            raise ProtocolError(f"prior candidate result {index} identity is malformed")
        key = (str(dataset), sample)
        if key in seen_keys:
            raise ProtocolError(f"prior candidate inventory duplicates row {key}")
        seen_keys.add(key)
        expected_name = hashlib.sha256(f"{dataset}\0{sample}".encode("utf-8")).hexdigest()[:20] + ".json"
        if relative.parts[2] != expected_name:
            raise ProtocolError(f"prior candidate result filename is not identity-bound: {resolved}")
        fingerprint = value.get("request_fingerprint")
        if not is_sha256(fingerprint):
            raise ProtocolError(f"prior candidate result {index} fingerprint is malformed")
        fingerprints.add(fingerprint)
        dataset_counts[str(dataset)] = dataset_counts.get(str(dataset), 0) + 1
    if dataset_counts != {"hotpotqa_train": 48, "musique_train": 48} or len(seen_keys) != 96:
        raise ProtocolError(f"prior candidate dataset counts changed: {dataset_counts}")
    if len(fingerprints) != 1:
        raise ProtocolError("prior candidate results do not share one request fingerprint")
    return {
        "inventory": artifact(inventory_path),
        "count": count,
        "results_sha256": inventory["results_sha256"],
        "dataset_counts": dataset_counts,
        "request_fingerprint": next(iter(fingerprints)),
    }


def _validate_prior23_evidence(prior23: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the nested dir23 evidence carried by the failed dir24 run."""

    if prior23.get("attempt_id") != "qwen3_32b_to_14b_shrinkage_old_pilot_eval_attempt23_v1":
        raise ProtocolError("nested dir23 attempt identity changed")
    if not isinstance(prior23.get("experiment_root"), str):
        raise ProtocolError("nested dir23 experiment root is missing")
    prior_root = Path(str(prior23["experiment_root"]))
    try:
        prior_root.resolve(strict=True).relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ProtocolError("nested dir23 evidence escaped repository root") from exc
    for evidence_name in ("log", "idle_gate", "run_definition", "status", "paired_summary", "candidate_inventory"):
        evidence = prior23.get(evidence_name)
        if not isinstance(evidence, Mapping) or set(evidence) != {"path", "nbytes", "sha256"}:
            raise ProtocolError(f"nested dir23 {evidence_name} binding is malformed")
        evidence_path = verify_ref(evidence, f"nested dir23 {evidence_name}")
        if evidence_name != "candidate_inventory":
            try:
                evidence_path.relative_to(prior_root.resolve(strict=True))
            except ValueError as exc:
                raise ProtocolError(f"nested dir23 {evidence_name} escaped its experiment root") from exc
    prior_definition = read_json(verify_ref(prior23["run_definition"], "nested dir23 run definition"))
    if prior_definition.get("experiment_id") != prior23.get("attempt_id") or prior_definition.get("candidate_mode") != "shrinkage_r1" or prior_definition.get("sample_count") != 96:
        raise ProtocolError("nested dir23 run-definition protocol changed")
    prior_status = read_json(verify_ref(prior23["status"], "nested dir23 status"))
    if prior_status.get("request_fingerprint") != prior_definition.get("request_fingerprint") or prior_status.get("status") not in {"running", "failed"} or prior_status.get("completed_candidate_results") != 96:
        raise ProtocolError("nested dir23 status does not record the failed 96-row run")
    prior_summary = read_json(verify_ref(prior23["paired_summary"], "nested dir23 paired summary"))
    if prior_summary.get("status") != "complete" or prior_summary.get("request_fingerprint") != prior_definition.get("request_fingerprint") or prior_summary.get("complete_candidate_results") != 96:
        raise ProtocolError("nested dir23 paired summary binding changed")
    prior_inventory = _validate_prior_candidate_inventory(prior23, prior_root)
    if prior_inventory["count"] != prior23.get("candidate_count") or prior_inventory["results_sha256"] != prior23.get("candidate_results_sha256") or prior_inventory["request_fingerprint"] != prior_definition.get("request_fingerprint"):
        raise ProtocolError("nested dir23 candidate inventory summary changed")
    return prior_inventory


def _validate_prior24_evidence(prior24: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the immutable, deliberately invalidated dir24 attempt."""

    expected_id = "qwen3_32b_to_14b_shrinkage_old_pilot_eval_attempt24_v1"
    if prior24.get("attempt_id") != expected_id:
        raise ProtocolError("nested dir24 prior attempt identity changed")
    if not isinstance(prior24.get("experiment_root"), str):
        raise ProtocolError("nested dir24 experiment root is missing")
    prior_root = Path(str(prior24["experiment_root"]))
    try:
        prior_root = prior_root.resolve(strict=True)
        prior_root.relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ProtocolError("nested dir24 evidence escaped repository root") from exc
    for evidence_name in ("preflight", "stop", "config", "runner", "tests", "inventory"):
        evidence = prior24.get(evidence_name)
        if not isinstance(evidence, Mapping) or set(evidence) != {"path", "nbytes", "sha256"}:
            raise ProtocolError(f"nested dir24 {evidence_name} binding is malformed")
        evidence_path = verify_ref(evidence, f"nested dir24 {evidence_name}")
        try:
            evidence_path.relative_to(prior_root)
        except ValueError as exc:
            raise ProtocolError(f"nested dir24 {evidence_name} escaped its experiment root") from exc
    prior_preflight_path = verify_ref(prior24["preflight"], "nested dir24 preflight")
    prior_stop_path = verify_ref(prior24["stop"], "nested dir24 STOP")
    inventory_path = verify_ref(prior24["inventory"], "nested dir24 inventory")
    verify_sidecar(prior_preflight_path)
    verify_sidecar(prior_stop_path)
    verify_sidecar(inventory_path)
    prior_preflight = read_json(prior_preflight_path)
    prior_stop = read_json(prior_stop_path)
    if (
        prior_preflight.get("status") != "passed_cpu_only_preflight"
        or prior_preflight.get("gpu_used") is not False
        or prior_preflight.get("cuda_api_called") is not False
        or prior_preflight.get("definition", {}).get("experiment_id") != expected_id
    ):
        raise ProtocolError("nested dir24 preflight witness changed")
    if (
        prior_stop.get("status") != "STOP_WAITING_FOR_OPERATOR_GPU_CONFIRMATION"
        or prior_stop.get("gpu_used") is not False
        or prior_stop.get("cuda_api_called") is not False
        or prior_stop.get("execute_authorized") is not False
        or prior_stop.get("preflight") != dict(prior24["preflight"])
    ):
        raise ProtocolError("nested dir24 STOP witness changed")
    if prior_stop.get("request_fingerprint") != prior_preflight.get("request_fingerprint"):
        raise ProtocolError("nested dir24 preflight/STOP fingerprint changed")
    if prior24.get("v2_preflight_interrupted_exit_code") != 130:
        raise ProtocolError("nested dir24 interruption witness changed")
    nested23 = prior24.get("prior23_evidence")
    if not isinstance(nested23, Mapping):
        raise ProtocolError("nested dir23 evidence is missing from dir24")
    prior23_inventory = _validate_prior23_evidence(nested23)
    if prior23_inventory["inventory"] != dict(prior24["inventory"]):
        raise ProtocolError("dir24 inventory does not bind nested dir23 inventory")
    if (
        prior24.get("candidate_count") != prior23_inventory["count"]
        or prior24.get("candidate_results_sha256") != prior23_inventory["results_sha256"]
    ):
        raise ProtocolError("dir24 candidate inventory summary changed")
    return prior23_inventory


def _validate_prior25_evidence(prior25: Mapping[str, Any]) -> dict[str, Any]:
    """Validate dir25's failure with exact, canonical, fail-closed bindings."""

    expected_id = "qwen3_32b_to_14b_shrinkage_old_pilot_eval_attempt25_v1"
    expected_fp = "80c74d0a2167dad62dd65eead0c86bf3c0cdc844699ac18aa835991f6bdb602c"
    expected_root = (REPO_ROOT / "experiments/qwen3_32b_to_14b_improved/25_shrinkage_old_pilot_eval_v1").resolve(strict=True)
    if prior25.get("attempt_id") != expected_id or prior25.get("request_fingerprint") != expected_fp:
        raise ProtocolError("prior25 identity or request fingerprint changed")
    raw_root = Path(str(prior25.get("experiment_root", "")))
    if not raw_root.is_absolute() or raw_root != expected_root:
        raise ProtocolError("prior25 experiment_root is not the exact canonical dir25 path")
    prior_root = _lstat_component_chain(raw_root, label="prior25 experiment root", allow_missing_tail=False)
    if prior_root != expected_root:
        raise ProtocolError("prior25 experiment_root changed during lstat")

    expected_paths = {
        "preflight": prior_root / "artifacts/preflight_pilot_shrinkage_r1_attempt25_v1.json",
        "preflight_sidecar": prior_root / "artifacts/preflight_pilot_shrinkage_r1_attempt25_v1.json.sha256.json",
        "stop": prior_root / "artifacts/STOP_PILOT_SHRINKAGE_R1_attempt25_v1.json",
        "stop_sidecar": prior_root / "artifacts/STOP_PILOT_SHRINKAGE_R1_attempt25_v1.json.sha256.json",
        "log": prior_root / "logs/shrinkage_old_pilot_eval_attempt25_v1.log",
        "config": prior_root / "config.json",
        "runner": prior_root / "src/runner.py",
        "tests": prior_root / "tests/test_runner.py",
        "status_witness": ROOT / "artifacts/prior25_status_witness_v2.json",
        "status_witness_sidecar": ROOT / "artifacts/prior25_status_witness_v2.json.sha256.json",
        "candidate_inventory": REPO_ROOT / "experiments/qwen3_32b_to_14b_improved/24_shrinkage_old_pilot_eval_v1/artifacts/prior23_candidate_inventory.json",
    }
    for evidence_name, expected_path in expected_paths.items():
        evidence = prior25.get(evidence_name)
        if not isinstance(evidence, Mapping) or set(evidence) != {"path", "nbytes", "sha256"}:
            raise ProtocolError(f"prior25 {evidence_name} binding is malformed")
        raw_path = Path(str(evidence["path"]))
        if not raw_path.is_absolute() or raw_path != expected_path:
            raise ProtocolError(f"prior25 {evidence_name} path is not canonical")
        checked = _lstat_component_chain(raw_path, label=f"prior25 {evidence_name}", allow_missing_tail=False)
        if checked != expected_path:
            raise ProtocolError(f"prior25 {evidence_name} path changed during lstat")
        verify_ref(evidence, f"prior25 {evidence_name}")
    preflight_path = verify_ref(prior25["preflight"], "prior25 preflight")
    stop_path = verify_ref(prior25["stop"], "prior25 STOP")
    preflight_sidecar_path = verify_ref(prior25["preflight_sidecar"], "prior25 preflight sidecar")
    stop_sidecar_path = verify_ref(prior25["stop_sidecar"], "prior25 STOP sidecar")
    if preflight_sidecar_path != expected_paths["preflight_sidecar"] or stop_sidecar_path != expected_paths["stop_sidecar"]:
        raise ProtocolError("prior25 explicit sidecar path changed")
    verify_sidecar(preflight_path)
    verify_sidecar(stop_path)
    preflight = read_json(preflight_path)
    stop = read_json(stop_path)
    if (
        preflight.get("status") != "passed_cpu_only_preflight"
        or preflight.get("gpu_used") is not False
        or preflight.get("cuda_api_called") is not False
        or not isinstance(preflight.get("definition"), Mapping)
        or preflight["definition"].get("experiment_id") != expected_id
        or preflight.get("sample_count") != 96
        or preflight.get("request_fingerprint") != expected_fp
    ):
        raise ProtocolError("prior25 preflight witness changed")
    if (
        stop.get("status") != "STOP_WAITING_FOR_OPERATOR_GPU_CONFIRMATION"
        or stop.get("gpu_used") is not False
        or stop.get("cuda_api_called") is not False
        or stop.get("execute_authorized") is not False
        or stop.get("preflight") != dict(prior25["preflight"])
        or stop.get("request_fingerprint") != expected_fp
    ):
        raise ProtocolError("prior25 STOP witness changed")
    log_path = verify_ref(prior25["log"], "prior25 failure log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if "TypeError" not in log_text or "dict(preflight_ref)" not in log_text or "PosixPath" not in log_text:
        raise ProtocolError("prior25 log does not bind the observed pre-GPU TypeError")

    status_path = verify_ref(prior25["status_witness"], "prior25 status witness")
    status_sidecar_path = verify_ref(prior25["status_witness_sidecar"], "prior25 status witness sidecar")
    if status_sidecar_path != expected_paths["status_witness_sidecar"]:
        raise ProtocolError("prior25 status witness sidecar path changed")
    verify_sidecar(status_path)
    status = read_json(status_path)
    expected_reason = "pre-GPU execute validation failed before CUDA_VISIBLE_DEVICES guard: _validate_stop_receipt received a PosixPath instead of the preflight artifact Mapping"
    expected_error = "dict(preflight_ref): PosixPath object is not iterable"
    if (
        status.get("schema_version") != 1
        or status.get("status") != "pre_gpu_validation_failed"
        or status.get("attempt_id") != expected_id
        or status.get("request_fingerprint") != expected_fp
        or status.get("failure_stage") != "before_cuda_visible_devices_guard"
        or status.get("failure_reason") != expected_reason
        or status.get("error_type") != "TypeError"
        or status.get("error") != expected_error
        or status.get("gpu_used") is not False
        or status.get("cuda_api_called") is not False
    ):
        raise ProtocolError("prior25 status witness changed")
    # Every failure descriptor is duplicated in the map and the witness so a
    # single-field mutation cannot be silently accepted.
    for field in ("failure_stage", "failure_reason", "error_type", "error"):
        if prior25.get(field) != status.get(field):
            raise ProtocolError(f"prior25 {field} is not witness-bound")
    expected_run_root = str(prior_root / "runs/pilot_shrinkage_r1_attempt25_v1")
    expected_status_path = expected_run_root + "/status.json"
    if (
        prior25.get("run_root") != expected_run_root
        or prior25.get("status_path") != expected_status_path
        or status.get("run_root") != expected_run_root
        or status.get("status_path") != expected_status_path
        or prior25.get("run_root_exists") is not False
        or prior25.get("status_path_exists") is not False
        or status.get("run_root_exists") is not False
        or status.get("status_path_exists") is not False
        or Path(expected_run_root).exists()
        or Path(expected_status_path).exists()
    ):
        raise ProtocolError("prior25 run_root/status_path witness changed")
    source_witness = status.get("source_witness")
    if not isinstance(source_witness, Mapping) or set(source_witness) != {"path", "nbytes", "sha256"}:
        raise ProtocolError("prior25 source witness binding is malformed")
    source_expected = REPO_ROOT / "experiments/qwen3_32b_to_14b_improved/26_shrinkage_old_pilot_eval_v1/artifacts/prior25_status_witness.json"
    if Path(str(source_witness["path"])) != source_expected:
        raise ProtocolError("prior25 source witness path changed")
    verify_ref(source_witness, "prior25 source witness")

    inventory_ref = prior25["candidate_inventory"]
    prior24 = prior25.get("prior24_evidence")
    if not isinstance(prior24, Mapping):
        raise ProtocolError("prior24 evidence is missing from prior25")
    prior23_inventory = _validate_prior24_evidence(prior24)
    if prior23_inventory["inventory"] != dict(inventory_ref):
        raise ProtocolError("prior25 candidate inventory does not bind prior24")
    if (
        prior25.get("candidate_count") != prior23_inventory["count"]
        or prior25.get("candidate_results_sha256") != prior23_inventory["results_sha256"]
    ):
        raise ProtocolError("prior25 candidate inventory summary changed")
    return prior23_inventory


def _validate_prior26_evidence(prior26: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the frozen dir26 attempt and its prior25 failure chain."""

    expected_id = "qwen3_32b_to_14b_shrinkage_old_pilot_eval_attempt26_v1"
    expected_fp = "051376d7df2e05465740559b4c116e08d677390a8b542dc4f99fa1dda03f35ff"
    expected_closure = "a503bd8134c12403569f77513fe8650ebcbffaa4592a5c96f66b489fdff7f14c"
    expected_root = (REPO_ROOT / "experiments/qwen3_32b_to_14b_improved/26_shrinkage_old_pilot_eval_v1").resolve(strict=True)
    if (
        prior26.get("attempt_id") != expected_id
        or prior26.get("protocol_id") != "qwen3_32b_to_14b_shrinkage_old_pilot_eval_v1"
        or prior26.get("request_fingerprint") != expected_fp
        or prior26.get("source_byte_closure_digest") != expected_closure
    ):
        raise ProtocolError("prior26 identity, protocol, fingerprint, or closure changed")
    raw_root = Path(str(prior26.get("experiment_root", "")))
    if not raw_root.is_absolute() or raw_root != expected_root:
        raise ProtocolError("prior26 experiment_root is not the exact canonical dir26 path")
    prior_root = _lstat_component_chain(raw_root, label="prior26 experiment root", allow_missing_tail=False)
    if prior_root != expected_root:
        raise ProtocolError("prior26 experiment_root changed during lstat")
    expected_paths = {
        "preflight": prior_root / "artifacts/preflight_pilot_shrinkage_r1_attempt26_v1.json",
        "preflight_sidecar": prior_root / "artifacts/preflight_pilot_shrinkage_r1_attempt26_v1.json.sha256.json",
        "stop": prior_root / "artifacts/STOP_PILOT_SHRINKAGE_R1_attempt26_v1.json",
        "stop_sidecar": prior_root / "artifacts/STOP_PILOT_SHRINKAGE_R1_attempt26_v1.json.sha256.json",
        "config": prior_root / "config.json",
        "runner": prior_root / "src/runner.py",
        "tests": prior_root / "tests/test_runner.py",
        "readme": prior_root / "README.md",
    }
    for evidence_name, expected_path in expected_paths.items():
        evidence = prior26.get(evidence_name)
        if not isinstance(evidence, Mapping) or set(evidence) != {"path", "nbytes", "sha256"}:
            raise ProtocolError(f"prior26 {evidence_name} binding is malformed")
        raw_path = Path(str(evidence["path"]))
        if not raw_path.is_absolute() or raw_path != expected_path:
            raise ProtocolError(f"prior26 {evidence_name} path is not canonical")
        if _lstat_component_chain(raw_path, label=f"prior26 {evidence_name}", allow_missing_tail=False) != expected_path:
            raise ProtocolError(f"prior26 {evidence_name} path changed during lstat")
        verify_ref(evidence, f"prior26 {evidence_name}")
    preflight_path = verify_ref(prior26["preflight"], "prior26 preflight")
    stop_path = verify_ref(prior26["stop"], "prior26 STOP")
    verify_ref(prior26["preflight_sidecar"], "prior26 preflight sidecar")
    verify_ref(prior26["stop_sidecar"], "prior26 STOP sidecar")
    verify_sidecar(preflight_path)
    verify_sidecar(stop_path)
    preflight = read_json(preflight_path)
    stop = read_json(stop_path)
    definition = preflight.get("definition")
    if (
        preflight.get("status") != "passed_cpu_only_preflight"
        or preflight.get("gpu_used") is not False
        or preflight.get("cuda_api_called") is not False
        or preflight.get("request_fingerprint") != expected_fp
        or preflight.get("source_byte_closure_digest") != expected_closure
        or not isinstance(definition, Mapping)
        or definition.get("experiment_id") != expected_id
        or definition.get("protocol_id") != "qwen3_32b_to_14b_shrinkage_old_pilot_eval_v1"
        or definition.get("sample_count") != 96
        or definition.get("request_fingerprint") != expected_fp
    ):
        raise ProtocolError("prior26 preflight witness changed")
    if (
        stop.get("status") != "STOP_WAITING_FOR_OPERATOR_GPU_CONFIRMATION"
        or stop.get("gpu_used") is not False
        or stop.get("cuda_api_called") is not False
        or stop.get("execute_authorized") is not False
        or stop.get("request_fingerprint") != expected_fp
        or stop.get("preflight") != dict(prior26["preflight"])
        or stop.get("source_byte_closure_digest") != expected_closure
    ):
        raise ProtocolError("prior26 STOP witness changed")
    command = stop.get("唯一启动命令")
    if not isinstance(command, str) or str(expected_paths["config"]) not in command or "--execute" not in command or "--preflight-only" in command:
        raise ProtocolError("prior26 STOP command is not bound to dir26")
    nested25 = prior26.get("prior25_evidence")
    if not isinstance(nested25, Mapping):
        raise ProtocolError("prior25 evidence is missing from prior26")
    prior23_inventory = _validate_prior25_evidence(nested25)
    return prior23_inventory


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    # Freeze the config spelling itself before resolving it.  This rejects a
    # live/dangling symlink in any parent component rather than silently
    # accepting the target bytes.
    config_candidate = _lstat_component_chain(
        path or ROOT / "config.json",
        label="dir27 config",
        allow_missing_tail=False,
    )
    config_path = config_candidate.resolve(strict=True)
    if config_path != config_candidate:
        raise ProtocolError("dir27 config changed during symlink-safe resolution")
    try:
        config_path.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ProtocolError("dir27 config escaped experiment root") from exc
    config = read_json(config_path)
    if config.get("schema_version") != 1 or config.get("protocol_id") != "qwen3_32b_to_14b_shrinkage_old_pilot_eval_v1" or config.get("direction") != "qwen3_32b_to_14b" or config.get("experiment_id") != "qwen3_32b_to_14b_shrinkage_old_pilot_eval_attempt27_v1":
        raise ProtocolError("dir27 config identity changed")
    if config.get("status") != "method_dev_only_gpu_execution_requires_operator_confirmation":
        raise ProtocolError("dir27 execution status changed")
    protocol = config.get("protocol")
    runtime = config.get("runtime")
    outputs = config.get("outputs")
    if not isinstance(protocol, Mapping) or not isinstance(runtime, Mapping) or not isinstance(outputs, Mapping):
        raise ProtocolError("dir27 config sections are malformed")
    if set(outputs) != {"run_root", "preflight_receipt", "stop_receipt"}:
        raise ProtocolError("dir27 output schema changed")
    if any(type(outputs[name]) is not str or not Path(outputs[name]).is_absolute() for name in outputs):
        raise ProtocolError("dir27 output paths must be absolute strings")
    configured_output_root = _assert_output_root(outputs["run_root"], allow_missing=True)
    for output_name in ("preflight_receipt", "stop_receipt"):
        _assert_output_path(
            outputs[output_name],
            label=f"configured output {output_name}",
            allow_missing_tail=True,
        )
    # Receipts must be in this experiment's artifact tree; this also avoids a
    # config that writes one immutable component into another experiment.
    for output_name in ("preflight_receipt", "stop_receipt"):
        receipt_path = Path(str(outputs[output_name]))
        try:
            receipt_path.resolve(strict=False).relative_to(ROOT.resolve(strict=True))
        except ValueError as exc:
            raise ProtocolError(f"configured output {output_name} escaped dir27") from exc
    fixed = {
        "phase": "pilot",
        "candidate_mode": "shrinkage_r1",
        "sample_count": 96,
        "max_length": 16384,
        "max_generation_tokens": 256,
        "prompt_mode": "qwen_chat_no_think",
        "truncation": "error",
        "greedy": True,
        "temperature": 0.0,
        "r1_boundary_tokens": 1,
        "candidate_builder_native_tail_tokens": 0,
    }
    for key, expected in fixed.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"protocol {key} changed")
    if protocol.get("stop_sequences") != ["\nQuestion:", "\nContext:"]:
        raise ProtocolError("stop sequences changed")
    if protocol.get("dataset_counts") != {"hotpotqa_train": 48, "musique_train": 48}:
        raise ProtocolError("pilot dataset counts changed")
    if runtime.get("cuda_visible_devices") != "0,1,2,3,4,5,6,7":
        raise ProtocolError("eight-card visibility contract changed")
    if runtime.get("source_devices") != [0, 1, 2, 3, 4] or runtime.get("source_layer_counts") != [12, 13, 14, 13, 12]:
        raise ProtocolError("5-card source topology changed")
    if runtime.get("target_devices") != [5, 6, 7] or runtime.get("target_layer_counts") != [13, 14, 13]:
        raise ProtocolError("3-card target topology changed")
    if runtime.get("idle_gate_samples") != 5 or runtime.get("idle_gate_interval_seconds") != 2:
        raise ProtocolError("idle gate contract changed")
    if configured_output_root.resolve(strict=False).is_relative_to(OLD03_ROOT.resolve()):
        raise ProtocolError("candidate output is inside old03 tree")
    # Every static input reference is checked before any model code is imported.
    refs = config.get("upstream")
    if not isinstance(refs, Mapping):
        raise ProtocolError("upstream references are missing")
    prior_attempt = refs.get("prior_attempt")
    if not isinstance(prior_attempt, Mapping) or not isinstance(prior_attempt.get("experiment_root"), str):
        raise ProtocolError("prior attempt evidence binding is missing")
    if prior_attempt.get("attempt_id") != "qwen3_32b_to_14b_shrinkage_old_pilot_eval_attempt26_v1":
        raise ProtocolError("prior26 attempt identity changed")
    _validate_prior26_evidence(prior_attempt)
    for key, value in refs.items():
        if isinstance(value, Mapping) and set(value) == {"path", "nbytes", "sha256"}:
            verify_ref(value, f"upstream.{key}")
        elif key.endswith("_model"):
            if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
                raise ProtocolError(f"upstream.{key} is malformed")
            for subkey in ("config", "weight_index", "tokenizer"):
                if subkey in value:
                    verify_ref(value[subkey], f"upstream.{key}.{subkey}")
    config["config_path"] = str(config_path)
    config["config_sha256"] = sha256_file(config_path)
    return config


def _ref_from_manifest(path: Path, entry: Mapping[str, Any], label: str) -> dict[str, Any]:
    if set(entry) < {"nbytes", "sha256"}:
        raise ProtocolError(f"{label} manifest entry is malformed")
    ref = {"path": str(path.resolve()), "nbytes": int(entry["nbytes"]), "sha256": str(entry["sha256"])}
    verify_ref(ref, label)
    return ref


def load_pilot_rows(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = verify_ref(config["upstream"]["prepared_manifest"], "prepared manifest")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "prepared" or manifest.get("protocol_id") != "qwen3_32b_to_14b_tail_refresh_method_dev_v2":
        raise ProtocolError("old03 prepared manifest is not the frozen method-dev preparation")
    prepared_root = manifest_path.parent
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ProtocolError("prepared file inventory is malformed")
    for name in ("selected_samples.jsonl", "data_registry.json", "preflight.json", "interface_receipt.json"):
        entry = files.get(name)
        if not isinstance(entry, Mapping):
            raise ProtocolError(f"prepared file entry missing: {name}")
        ref = _ref_from_manifest(prepared_root / name, entry, f"prepared {name}")
        if name == "selected_samples.jsonl":
            expected = config["upstream"]["selected_samples"]
            if ref["sha256"] != expected["sha256"] or ref["nbytes"] != expected["nbytes"]:
                raise ProtocolError("selected-samples reference does not match config")
    preflight = read_json(prepared_root / "preflight.json")
    if preflight.get("status") != "passed_method_dev_local_preflight" or preflight.get("gpu_used") is not False:
        raise ProtocolError("old03 CPU preparation preflight is not passed")
    interface = read_json(prepared_root / "interface_receipt.json")
    if interface.get("status") != "frozen_method_dev_interface":
        raise ProtocolError("old03 interface receipt is not frozen")
    rows: list[dict[str, Any]] = []
    with (prepared_root / "selected_samples.jsonl").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ProtocolError(f"selected row {line_number} is not an object")
            if "pilot" in value.get("phase_membership", []):
                rows.append(value)
    if len(rows) != 96:
        raise ProtocolError(f"old03 pilot row count changed: {len(rows)}")
    seen: set[tuple[str, str]] = set()
    counts = {dataset: 0 for dataset in DATASETS}
    for row in rows:
        dataset = row.get("dataset_id")
        sample = row.get("sample_id")
        if dataset not in DATASETS or not isinstance(sample, str) or not sample:
            raise ProtocolError("pilot row identity is malformed")
        key = (str(dataset), sample)
        if key in seen:
            raise ProtocolError(f"duplicate pilot row: {key}")
        seen.add(key)
        ids = row.get("prompt_token_ids")
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ProtocolError(f"prompt token IDs malformed for {key}")
        if token_ids_sha256(ids) != row.get("prompt_token_ids_sha256"):
            raise ProtocolError(f"prompt token hash changed for {key}")
        if row.get("prompt_token_count") != len(ids) or row.get("prompt_truncated") is not False:
            raise ProtocolError(f"prompt contract changed for {key}")
        if len(ids) > 16384 or len(ids) - 1 < 1:
            raise ProtocolError(f"prompt length is invalid for {key}")
        references = row.get("references")
        if not isinstance(references, list) or not references or any(not isinstance(x, str) for x in references):
            raise ProtocolError(f"references malformed for {key}")
        if not isinstance(row.get("source_record_sha256"), str) or not isinstance(row.get("stratum"), str):
            raise ProtocolError(f"source identity malformed for {key}")
        counts[str(dataset)] += 1
    if counts != {"hotpotqa_train": 48, "musique_train": 48}:
        raise ProtocolError(f"pilot strata counts changed: {counts}")
    return rows, artifact(manifest_path)


def old_result_path(row: Mapping[str, Any], mode: str) -> Path:
    identity = hashlib.sha256(f"{row['dataset_id']}\0{row['sample_id']}".encode("utf-8")).hexdigest()[:20]
    return BASELINE_RUN_ROOT / "results" / str(row["dataset_id"]) / mode / f"{identity}.json"


def candidate_result_path(output: Path, row: Mapping[str, Any]) -> Path:
    identity = hashlib.sha256(f"{row['dataset_id']}\0{row['sample_id']}".encode("utf-8")).hexdigest()[:20]
    return output / "results" / str(row["dataset_id"]) / "shrinkage_r1" / f"{identity}.json"


def _guard_output_component(path: Path, output: Path, *, label: str, allow_missing_tail: bool = True) -> Path:
    """Guard a mutable output path against symlink traversal and escape."""

    _assert_output_root(output, allow_missing=False)
    return _assert_output_path(
        path,
        label=label,
        output_root=output,
        allow_missing_tail=allow_missing_tail,
    )


def _assert_output_tree_nonsymlink(output: Path) -> None:
    """Fail closed if a mutable output tree contains any symlink component."""

    _assert_output_root(output, allow_missing=False)
    if not output.exists():
        return
    if not output.is_dir():
        raise ProtocolError(f"output root is not a directory: {output}")
    for path in output.rglob("*"):
        try:
            mode = os.lstat(path).st_mode
        except OSError as exc:
            raise ProtocolError(f"cannot lstat output component {path}: {exc}") from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"output tree contains a symlink: {path}")


def _failure_marker_path(destination: Path) -> Path:
    return destination.with_suffix(".failure.json")


def _superseded_failure_marker_path(destination: Path) -> Path:
    return destination.with_suffix(".failure.superseded.json")


def _validate_failure_marker(
    failure_path: Path,
    row: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and bind a failure marker before it can be superseded."""

    if failure_path.is_symlink() or not failure_path.is_file():
        raise ProtocolError(f"failure marker is not a regular file: {failure_path}")
    value = read_json(failure_path)
    expected = {
        "status": "failed",
        "request_fingerprint": definition["request_fingerprint"],
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "mode": "shrinkage_r1",
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ProtocolError(f"failure marker binding changed: {failure_path}")
    for field in ("error_type", "error", "traceback"):
        if not isinstance(value.get(field), str):
            raise ProtocolError(f"failure marker field {field} is malformed: {failure_path}")
    sidecar = failure_path.with_name(failure_path.name + ".sha256.json")
    if sidecar.exists() or sidecar.is_symlink():
        if sidecar.is_symlink():
            raise ProtocolError(f"failure marker sidecar is a symlink: {sidecar}")
        verify_sidecar(failure_path)
    return artifact(failure_path), value


def _validate_superseded_failure_marker(
    marker_path: Path,
    row: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> dict[str, Any]:
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ProtocolError(f"superseded failure marker is not a regular file: {marker_path}")
    value = read_json(marker_path)
    expected = {
        "status": "superseded",
        "request_fingerprint": definition["request_fingerprint"],
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "mode": "shrinkage_r1",
    }
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ProtocolError(f"superseded failure marker binding changed: {marker_path}")
    old_ref = value.get("superseded_failure")
    if not isinstance(old_ref, Mapping) or set(old_ref) != {"path", "nbytes", "sha256"}:
        raise ProtocolError(f"superseded failure provenance is malformed: {marker_path}")
    if not isinstance(old_ref.get("path"), str) or type(old_ref.get("nbytes")) is not int or not is_sha256(old_ref.get("sha256")):
        raise ProtocolError(f"superseded failure provenance is malformed: {marker_path}")
    sidecar = marker_path.with_name(marker_path.name + ".sha256.json")
    if sidecar.exists() or sidecar.is_symlink():
        if sidecar.is_symlink():
            raise ProtocolError(f"superseded failure sidecar is a symlink: {sidecar}")
        verify_sidecar(marker_path)
    return value


def _validate_failure_state_if_present(
    destination: Path,
    row: Mapping[str, Any],
    definition: Mapping[str, Any],
    output: Path,
) -> None:
    """Validate any old failure/tombstone before resuming or rerunning a cell."""

    _guard_output_component(destination, output, label="candidate checkpoint")
    failure = _failure_marker_path(destination)
    failure_sidecar = failure.with_name(failure.name + ".sha256.json")
    superseded = _superseded_failure_marker_path(destination)
    superseded_sidecar = superseded.with_name(superseded.name + ".sha256.json")
    if failure.exists() or failure.is_symlink():
        _validate_failure_marker(failure, row, definition)
    elif failure_sidecar.exists() or failure_sidecar.is_symlink():
        raise ProtocolError(f"orphan failure sidecar blocks resume: {failure_sidecar}")
    if superseded.exists() or superseded.is_symlink():
        _validate_superseded_failure_marker(superseded, row, definition)
    elif superseded_sidecar.exists() or superseded_sidecar.is_symlink():
        raise ProtocolError(f"orphan superseded-failure sidecar blocks resume: {superseded_sidecar}")


def _supersede_failure_marker(
    destination: Path,
    row: Mapping[str, Any],
    definition: Mapping[str, Any],
    output: Path,
) -> dict[str, Any] | None:
    """Atomically publish a bound superseded tombstone and clear old failure.

    The old marker is never silently discarded: it is hash-bound into the
    tombstone.  The tombstone is written with an atomic rename first, then the
    old marker (and optional sidecar) is removed while the global GPU lock is
    held.  A crash between these operations leaves no valid completion receipt;
    the next resume repairs the transition and final publication remains
    fail-closed if any ``*.failure.json`` survives.
    """

    _guard_output_component(destination, output, label="candidate checkpoint")
    failure = _failure_marker_path(destination)
    failure_sidecar = failure.with_name(failure.name + ".sha256.json")
    superseded = _superseded_failure_marker_path(destination)
    superseded_sidecar = superseded.with_name(superseded.name + ".sha256.json")
    if not failure.exists() and not failure.is_symlink():
        if failure_sidecar.exists() or failure_sidecar.is_symlink():
            raise ProtocolError(f"orphan failure sidecar blocks publication: {failure_sidecar}")
        if superseded.exists() or superseded.is_symlink():
            _validate_superseded_failure_marker(superseded, row, definition)
            if not superseded_sidecar.exists():
                write_sidecar(superseded)
            return artifact(superseded)
        if superseded_sidecar.exists() or superseded_sidecar.is_symlink():
            raise ProtocolError(f"orphan superseded-failure sidecar blocks publication: {superseded_sidecar}")
        return None

    old_ref, old_value = _validate_failure_marker(failure, row, definition)
    tombstone = {
        "schema_version": 1,
        "status": "superseded",
        "request_fingerprint": definition["request_fingerprint"],
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "mode": "shrinkage_r1",
        "superseded_failure": old_ref,
        "reason": "valid_candidate_checkpoint_published",
    }
    if superseded.exists() or superseded.is_symlink():
        _validate_superseded_failure_marker(superseded, row, definition)
        if read_json(superseded) != tombstone:
            raise ProtocolError(f"existing superseded marker differs: {superseded}")
        if not superseded_sidecar.exists():
            write_sidecar(superseded)
    else:
        immutable_json(superseded, tombstone)
        write_sidecar(superseded)
    # Remove the old marker only after the superseded state is durable.  Both
    # unlink operations are idempotent and are guarded by the global flock.
    if failure_sidecar.exists() or failure_sidecar.is_symlink():
        if failure_sidecar.is_symlink():
            raise ProtocolError(f"failure marker sidecar is a symlink: {failure_sidecar}")
        failure_sidecar.unlink()
    if failure.exists() or failure.is_symlink():
        if failure.is_symlink():
            raise ProtocolError(f"failure marker became a symlink: {failure}")
        failure.unlink()
    if failure.exists() or failure_sidecar.exists():
        raise ProtocolError(f"failure marker transition did not seal: {failure}")
    return artifact(superseded)


def _assert_no_stale_failure_markers(output: Path) -> None:
    """Completion is forbidden while any failure marker is unsealed.

    A superseded marker is itself append-only evidence, so it must be a
    well-formed, sidecar-bound regular file.  This also rejects orphan
    superseded sidecars that could otherwise hide a partial marker transition.
    """

    _assert_output_tree_nonsymlink(output)
    if not output.exists():
        return
    stale: list[Path] = []
    malformed_superseded: list[Path] = []
    for path in output.rglob("*"):
        if path.name.endswith(".failure.json") or path.name.endswith(".failure.json.sha256.json"):
            stale.append(path)
            continue
        if path.name.endswith(".failure.superseded.json.sha256.json"):
            marker = path.with_name(path.name.removesuffix(".sha256.json"))
            if not marker.is_file() or marker.is_symlink():
                malformed_superseded.append(path)
            continue
        if path.name.endswith(".failure.superseded.json"):
            sidecar = path.with_name(path.name + ".sha256.json")
            try:
                value = read_json(path)
                valid = (
                    value.get("schema_version") == 1
                    and value.get("status") == "superseded"
                    and is_sha256(value.get("request_fingerprint"))
                    and isinstance(value.get("superseded_failure"), Mapping)
                    and set(value["superseded_failure"]) == {"path", "nbytes", "sha256"}
                    and sidecar.is_file()
                    and not sidecar.is_symlink()
                )
                if valid:
                    verify_sidecar(path)
                else:
                    malformed_superseded.append(path)
            except (OSError, ProtocolError, json.JSONDecodeError):
                malformed_superseded.append(path)
    if stale:
        raise ProtocolError(f"completion cannot retain stale failure markers: {stale[:4]}")
    if malformed_superseded:
        raise ProtocolError(f"completion cannot retain malformed superseded failure markers: {malformed_superseded[:4]}")


def _superseded_failure_inventory(output: Path) -> list[dict[str, Any]]:
    results = output / "results"
    if not results.exists():
        return []
    return [
        artifact(path)
        for path in sorted(results.rglob("*.failure.superseded.json"))
        if path.is_file() and not path.is_symlink()
    ]


def _validate_baseline_record(value: Mapping[str, Any], row: Mapping[str, Any], mode: str, old_fingerprint: str) -> None:
    if value.get("status") != "complete" or value.get("request_fingerprint") != old_fingerprint:
        raise ProtocolError("old03 baseline checkpoint is incomplete or fingerprinted differently")
    if value.get("phase") != "pilot" or value.get("dataset_id") != row["dataset_id"] or value.get("sample_id") != row["sample_id"]:
        raise ProtocolError("old03 baseline checkpoint identity changed")
    for field in ("source_record_sha256", "prompt_token_ids_sha256"):
        if value.get(field) != row.get(field):
            raise ProtocolError(f"old03 baseline {field} changed")
    if value.get("prompt_token_count") != row.get("prompt_token_count") or value.get("prompt_truncated") is not False:
        raise ProtocolError("old03 baseline prompt binding changed")
    generation = value.get("generation")
    metrics = value.get("metrics")
    if not isinstance(generation, Mapping) or not isinstance(metrics, Mapping):
        raise ProtocolError("old03 baseline generation/metrics missing")
    if generation.get("requested_max_tokens") != 256 or generation.get("stop_reason") not in {"eos", "stop_sequence", "max_tokens"}:
        raise ProtocolError("old03 baseline generation cap/stop changed")
    if generation.get("cap_hit") is not (generation.get("stop_reason") == "max_tokens"):
        raise ProtocolError("old03 baseline cap/termination binding changed")
    if not isinstance(generation.get("generated_token_ids"), list) or generation.get("generated_tokens") != len(generation["generated_token_ids"]):
        raise ProtocolError("old03 baseline generated-token binding changed")
    if any(name not in metrics for name in METRIC_NAMES):
        raise ProtocolError("old03 baseline metric schema changed")


def load_baseline_bindings(config: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    inventory_path = verify_ref(config["upstream"]["baseline_inventory"], "old03 result inventory")
    inventory = read_json(inventory_path)
    entries = inventory.get("files")
    if inventory.get("count") != 384 or not isinstance(entries, list) or len(entries) != 384:
        raise ProtocolError("old03 result inventory cardinality changed")
    old_definition_path = verify_ref(config["upstream"]["baseline_run_definition"], "old03 run definition")
    old_definition = read_json(old_definition_path)
    old_fingerprint = old_definition.get("request_fingerprint")
    if not isinstance(old_fingerprint, str) or old_definition.get("phase") != "pilot" or old_definition.get("sample_count") != 96:
        raise ProtocolError("old03 run definition is not the frozen pilot")
    if old_definition.get("modes") != ["target_standalone", "mapped_r1", "tail_refresh_r8", "tail_refresh_r32"]:
        raise ProtocolError("old03 mode topology changed")
    summary_path = verify_ref(config["upstream"]["baseline_summary"], "old03 summary")
    summary = read_json(summary_path)
    if summary.get("status") != "complete" or summary.get("expected_atomic_results") != 384 or summary.get("complete_atomic_results") != 384:
        raise ProtocolError("old03 summary is incomplete")
    inventory_map: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ProtocolError("old03 inventory entry is malformed")
        rel = Path(str(entry["path"]))
        path = (OLD03_ROOT / rel).resolve()
        path.relative_to(BASELINE_RUN_ROOT.resolve())
        if rel.as_posix().startswith("../"):
            raise ProtocolError("old03 inventory escapes baseline root")
        _regular(path, "old03 result")
        if path.stat().st_size != int(entry.get("nbytes", -1)) or sha256_file(path) != entry.get("sha256"):
            raise ProtocolError(f"old03 result inventory hash changed: {path}")
        inventory_map[str(path)] = entry
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["dataset_id"]), str(row["sample_id"]))
        bindings: dict[str, Any] = {}
        for alias, mode in BASELINE_MODES.items():
            path = old_result_path(row, mode).resolve()
            if str(path) not in inventory_map:
                raise ProtocolError(f"old03 inventory lacks {key}/{mode}")
            ref = artifact(path)
            value = read_json(path)
            _validate_baseline_record(value, row, mode, old_fingerprint)
            bindings[alias] = {"artifact": ref, "mode": mode, "record": value}
        result[key] = bindings
    return result


def _source_paths() -> list[Path]:
    paths: set[Path] = set()
    for root in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        if root.exists():
            paths.update(path for path in root.rglob("*.py") if path.is_file() and not path.is_symlink())
    # Hash the complete imported implementation closure, not just this runner.
    for root in (
        REPO_ROOT / "src/kvcache_transfer",
        REPO_ROOT / "experiments/qwen3_32b_to_14b/reverse_kv",
    ):
        paths.update(path for path in root.rglob("*.py") if path.is_file() and not path.is_symlink())
    paths.add(SHRINKAGE_ROOT / "src/runtime.py")
    paths.add(OLD03_ROOT / "src/qa_metrics.py")
    return sorted(paths, key=lambda p: str(p.resolve()))


def source_byte_closure() -> dict[str, Any]:
    files = [artifact(path) for path in _source_paths()]
    return {"files": files, "digest": canonical_sha256(files)}


def validate_candidate_mapper(config: Mapping[str, Any], *, deep_hash: bool = True) -> dict[str, Any]:
    manifest_path = verify_ref(config["upstream"]["candidate_mapper_manifest"], "shrinkage mapper manifest")
    root = manifest_path.parent
    marker_path = verify_ref(config["upstream"]["candidate_mapper_marker"], "shrinkage mapper marker")
    marker = read_json(marker_path)
    if marker != {"path": "manifest.json", "sha256": sha256_file(manifest_path)}:
        raise ProtocolError("shrinkage mapper marker changed")
    manifest = read_json(manifest_path)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "method": "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers",
        "dtype": "float32",
        "allow_tf32": False,
        "architecture": {"source_layers": 64, "target_layers": 40, "kv_heads": 8, "head_dim": 128, "selected_source_layers": 20},
    }
    for field, expected in fixed.items():
        if manifest.get(field) != expected:
            raise ProtocolError(f"shrinkage mapper {field} changed")
    if not is_sha256(manifest.get("mapper_fingerprint")):
        raise ProtocolError("shrinkage mapper fingerprint is malformed")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ProtocolError("shrinkage mapper layer count changed")
    layer_refs: list[dict[str, Any]] = []
    for index, entry in enumerate(layers):
        if not isinstance(entry, Mapping) or entry.get("target_layer") != index:
            raise ProtocolError(f"shrinkage layer order changed at {index}")
        for field in ("path", "sidecar_path", "sha256", "sidecar_sha256"):
            if not isinstance(entry.get(field), str):
                raise ProtocolError(f"shrinkage layer {index} lacks {field}")
        payload = (root / str(entry["path"])).resolve()
        sidecar = (root / str(entry["sidecar_path"])).resolve()
        payload.relative_to(root.resolve())
        sidecar.relative_to(root.resolve())
        payload_ref = artifact(payload)
        sidecar_ref = artifact(sidecar)
        if payload_ref["sha256"] != entry["sha256"] or sidecar_ref["sha256"] != entry["sidecar_sha256"]:
            raise ProtocolError(f"shrinkage layer {index} payload hash changed")
        side = read_json(sidecar)
        if side.get("target_layer") != index or side.get("sha256") != entry["sha256"] or side.get("dtype") != "float32":
            raise ProtocolError(f"shrinkage layer {index} sidecar changed")
        if len(entry.get("source_layers_rank_order", [])) != 20 or len(set(entry.get("source_layers_rank_order", []))) != 20:
            raise ProtocolError(f"shrinkage layer {index} source selection changed")
        layer_refs.extend((payload_ref, sidecar_ref))
    completion_path = verify_ref(config["upstream"]["candidate_completion_receipt"], "shrinkage completion receipt")
    completion = read_json(completion_path)
    if completion.get("status") != "complete_by_append_only_amendment" or completion.get("amendment_scope") != "receipt-serialization-only":
        raise ProtocolError("shrinkage completion amendment changed")
    if completion.get("mapper", {}).get("manifest_sha256") != sha256_file(manifest_path):
        raise ProtocolError("shrinkage completion/mapper binding changed")
    completion_manifest = verify_ref(config["upstream"]["candidate_completion_manifest"], "shrinkage completion manifest")
    completion_marker = verify_ref(config["upstream"]["candidate_completion_marker"], "shrinkage completion marker")
    if read_json(completion_marker) != {"path": "manifest.json", "sha256": sha256_file(completion_manifest)}:
        raise ProtocolError("shrinkage completion marker changed")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, Mapping) or not isinstance(upstream.get("k20_manifest"), Mapping):
        raise ProtocolError("shrinkage upstream k20 binding missing")
    k20 = upstream["k20_manifest"]
    k20_path = verify_ref(k20, "shrinkage upstream k20 manifest")
    old03_mapper_path = Path(read_json(verify_ref(config["upstream"]["old03_config"], "old03 config"))["mapper"]["manifest"]).resolve()
    old03_mapper_hash = str(read_json(verify_ref(config["upstream"]["old03_config"], "old03 config"))["mapper"]["manifest_sha256"])
    if str(k20_path) != str(old03_mapper_path) or sha256_file(k20_path) != old03_mapper_hash:
        raise ProtocolError("shrinkage upstream k20 is not the hash-pinned old03 mapper")
    if str(k20_path) != str(Path(config["upstream"]["old03_root"]).resolve().parent.parent / "qwen3_32b_to_14b/artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01/manifest.json"):
        # The exact path is pinned by old03 config below; this branch mainly
        # prevents a candidate from silently pointing at another mapper.
        raise ProtocolError("shrinkage upstream k20 path changed")
    return {
        "manifest": artifact(manifest_path),
        "marker": artifact(marker_path),
        "completion_receipt": artifact(completion_path),
        "completion_manifest": artifact(completion_manifest),
        "completion_marker": artifact(completion_marker),
        "mapper_fingerprint": manifest["mapper_fingerprint"],
        "layers": 40,
        "payloads_deep_hashed": bool(deep_hash),
        "layer_artifacts": layer_refs,
    }


def _validate_old03_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    old_path = verify_ref(config["upstream"]["old03_config"], "old03 config")
    old = read_json(old_path)
    if old.get("protocol_id") != "qwen3_32b_to_14b_tail_refresh_method_dev_v2" or old.get("status") != "method_dev_only_locked_test_forbidden":
        raise ProtocolError("old03 protocol identity changed")
    if old.get("prompt", {}).get("max_generation_tokens") != 256 or old.get("prompt", {}).get("stop_sequences") != ["\nQuestion:", "\nContext:"]:
        raise ProtocolError("old03 prompt generation contract changed")
    if old.get("prompt", {}).get("max_length") != 16384 or old.get("prompt", {}).get("mode") != "qwen_chat_no_think":
        raise ProtocolError("old03 prompt length/mode changed")
    if old.get("runtime", {}).get("source_devices") != [0, 1, 2, 3, 4] or old.get("runtime", {}).get("target_devices") != [5, 6, 7]:
        raise ProtocolError("old03 topology changed")
    if old.get("models", {}).get("source_32b", {}).get("config_sha256") != config["upstream"]["source_model"]["config"]["sha256"]:
        raise ProtocolError("source model config is not old03-bound")
    if old.get("models", {}).get("target_14b", {}).get("config_sha256") != config["upstream"]["target_model"]["config"]["sha256"]:
        raise ProtocolError("target model config is not old03-bound")
    metrics_path = verify_ref(config["upstream"]["old03_metrics"], "old03 metrics implementation")
    prompt_path = verify_ref(config["upstream"]["old03_prompt_impl"], "old03 prompt implementation")
    musique_config = verify_ref(config["upstream"]["old03_musique_config"], "MuSiQue config")
    musique_registry = verify_ref(config["upstream"]["old03_musique_registry"], "MuSiQue registry")
    if read_json(musique_registry).get("status") != "complete":
        raise ProtocolError("MuSiQue registry is not complete")
    return {"config": artifact(old_path), "metrics": artifact(metrics_path), "prompt_impl": artifact(prompt_path), "musique_config": artifact(musique_config), "musique_registry": artifact(musique_registry)}


def build_definition(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    source_closure: Mapping[str, Any],
    mapper: Mapping[str, Any],
    old_protocol: Mapping[str, Any],
    baseline: Mapping[tuple[str, str], Mapping[str, Any]],
    prepared_ref: Mapping[str, Any],
    model_inventories: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_digest = canonical_sha256(
        [
            [key[0], key[1], bindings["native14"]["artifact"], bindings["nvidia_k20"]["artifact"]]
            for key, bindings in sorted(baseline.items())
        ]
    )
    body = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "experiment_id": config["experiment_id"],
        "direction": config["direction"],
        "phase": "pilot",
        "candidate_mode": "shrinkage_r1",
        "baseline_aliases": BASELINE_MODES,
        "sample_count": len(rows),
        "dataset_counts": {dataset: sum(row["dataset_id"] == dataset for row in rows) for dataset in DATASETS},
        "sample_identity_digest": canonical_sha256([[row["dataset_id"], row["sample_id"], row["source_record_sha256"], row["prompt_token_ids_sha256"]] for row in rows]),
        "prepared_manifest": prepared_ref,
        "old03_protocol": old_protocol,
        "old03_baseline_run_definition": artifact(verify_ref(config["upstream"]["baseline_run_definition"], "old03 run definition")),
        "old03_baseline_inventory": artifact(verify_ref(config["upstream"]["baseline_inventory"], "old03 inventory")),
        "old03_baseline_summary": artifact(verify_ref(config["upstream"]["baseline_summary"], "old03 summary")),
        "baseline_binding_digest": baseline_digest,
        "candidate_mapper": mapper,
        "source_byte_closure": source_closure,
        "config": artifact(Path(config["config_path"])),
        "models": {
            "source_32b": config["upstream"]["source_model"],
            "target_14b": config["upstream"]["target_model"],
        },
        "complete_model_shard_inventories": dict(model_inventories),
        "prompt": {
            "mode": config["protocol"]["prompt_mode"],
            "max_length": config["protocol"]["max_length"],
            "max_generation_tokens": config["protocol"]["max_generation_tokens"],
            "stop_sequences": config["protocol"]["stop_sequences"],
            "truncation": config["protocol"]["truncation"],
            "greedy": config["protocol"]["greedy"],
            "temperature": config["protocol"]["temperature"],
        },
        "topology": {
            "source_devices": config["runtime"]["source_devices"],
            "source_layer_counts": config["runtime"]["source_layer_counts"],
            "target_devices": config["runtime"]["target_devices"],
            "target_layer_counts": config["runtime"]["target_layer_counts"],
        },
        "execution_policy": {
            "baseline_rerun": False,
            "native14_reused": True,
            "nvidia_k20_reused": True,
            "locked_test_access": False,
            "official_dev_or_test_opened": False,
            "claim_boundary": "method-dev pilot only; no locked-test or paper-superiority claim",
        },
        "prior_attempt_evidence": config["upstream"]["prior_attempt"],
    }
    return {**body, "request_fingerprint": canonical_sha256(body)}


def _preflight_artifacts(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows, prepared_ref = load_pilot_rows(config)
    baseline = load_baseline_bindings(config, rows)
    mapper = validate_candidate_mapper(config, deep_hash=True)
    old_protocol = _validate_old03_protocol(config)
    closure = source_byte_closure()
    model_inventories = {
        "source_32b": model_shard_inventory(config["upstream"]["source_model"], "source Qwen3-32B"),
        "target_14b": model_shard_inventory(config["upstream"]["target_model"], "target Qwen3-14B"),
    }
    return rows, prepared_ref, baseline, mapper, {"old_protocol": old_protocol, "source_closure": closure}, model_inventories


def preflight(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    rows, prepared_ref, baseline, mapper, extras, model_inventories = _preflight_artifacts(config)
    definition = build_definition(config, rows, extras["source_closure"], mapper, extras["old_protocol"], baseline, prepared_ref, model_inventories)
    preflight_value = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "passed_cpu_only_preflight",
        "gpu_used": False,
        "cuda_api_called": False,
        "phase": "pilot",
        "candidate_mode": "shrinkage_r1",
        "sample_count": len(rows),
        "dataset_counts": {dataset: sum(row["dataset_id"] == dataset for row in rows) for dataset in DATASETS},
        "expected_candidate_results": len(rows),
        "expected_paired_rows": len(rows),
        "request_fingerprint": definition["request_fingerprint"],
        "definition_digest": canonical_sha256(definition),
        "definition": definition,
        "candidate_mapper": mapper,
        "baseline_binding_digest": definition["baseline_binding_digest"],
        "source_byte_closure_digest": extras["source_closure"]["digest"],
        "complete_model_shard_inventories": model_inventories,
        "stop_sequences": config["protocol"]["stop_sequences"],
        "generation_cap": config["protocol"]["max_generation_tokens"],
        "topology": definition["topology"],
    }
    preflight_path = _assert_output_path(
        config["outputs"]["preflight_receipt"],
        label="preflight receipt output",
        allow_missing_tail=True,
    )
    preflight_ref = immutable_json(preflight_path, preflight_value)
    write_sidecar(preflight_path)
    stop_value = {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": "STOP_WAITING_FOR_OPERATOR_GPU_CONFIRMATION",
        "request_fingerprint": definition["request_fingerprint"],
        "preflight": preflight_ref,
        "source_byte_closure_digest": extras["source_closure"]["digest"],
        "source_byte_closure_file_count": len(extras["source_closure"]["files"]),
        "implementation_inventory_digest": extras["source_closure"]["digest"],
        "gpu_used": False,
        "cuda_api_called": False,
        "execute_authorized": False,
        "唯一启动命令": (
            f"CUDA_VISIBLE_DEVICES={config['runtime']['cuda_visible_devices']} "
            f"PYTHONDONTWRITEBYTECODE=1 "
            f"PYTHONPYCACHEPREFIX={config['runtime']['required_pycache_prefix']} "
            f"{config['runtime']['required_python']} {ROOT / 'scripts/run.py'} "
            f"--config {config['config_path']} --execute"
        ),
        "lock": config["runtime"]["global_gpu_lock"],
        "idle_gate": {
            "cards": 8,
            "samples": config["runtime"]["idle_gate_samples"],
            "interval_seconds": config["runtime"]["idle_gate_interval_seconds"],
            "max_memory_mib": config["runtime"]["idle_gate_max_memory_mib"],
        },
        "claim_boundary": "method-dev pilot only; no locked-test or paper-superiority claim",
    }
    stop_path = _assert_output_path(
        config["outputs"]["stop_receipt"],
        label="STOP receipt output",
        allow_missing_tail=True,
    )
    stop_ref = immutable_json(stop_path, stop_value)
    write_sidecar(stop_path)
    print(json.dumps({"preflight": preflight_ref, "stop": stop_ref, "request_fingerprint": definition["request_fingerprint"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return {"config": config, "rows": rows, "baseline": baseline, "definition": definition, "preflight": preflight_value, "preflight_ref": preflight_ref, "stop_ref": stop_ref}


def _validate_bytecode_environment(config: Mapping[str, Any]) -> None:
    expected_python = Path(str(config["runtime"]["required_python"])).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ProtocolError(f"wrong Python executable: {sys.executable}")
    prefix = Path(str(config["runtime"]["required_pycache_prefix"])).resolve()
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ProtocolError("PYTHONDONTWRITEBYTECODE=1 is required")
    observed_prefix = getattr(sys, "pycache_prefix", None)
    if os.environ.get("PYTHONPYCACHEPREFIX") != str(prefix) or observed_prefix != str(prefix):
        raise ProtocolError("formal pycache prefix is not bound")
    if os.path.lexists(prefix):
        raise ProtocolError(f"formal pycache prefix must be fresh/absent: {prefix}")


def _validate_stop_receipt(
    config: Mapping[str, Any],
    definition: Mapping[str, Any],
    preflight_ref: Mapping[str, Any],
) -> dict[str, Any]:
    """Require an unmodified CPU preflight and operator-gated STOP witness."""

    stop_path = _assert_output_path(
        config["outputs"]["stop_receipt"],
        label="dir27 STOP receipt",
        allow_missing_tail=False,
    )
    stop_ref = verify_ref(artifact(stop_path), "dir27 STOP receipt")
    verify_sidecar(stop_path)
    stop = read_json(stop_path)
    if (
        stop.get("schema_version") != 1
        or stop.get("protocol_id") != config.get("protocol_id")
        or stop.get("status") != "STOP_WAITING_FOR_OPERATOR_GPU_CONFIRMATION"
        or stop.get("request_fingerprint") != definition["request_fingerprint"]
        or stop.get("preflight") != dict(preflight_ref)
        or stop.get("gpu_used") is not False
        or stop.get("cuda_api_called") is not False
        or stop.get("execute_authorized") is not False
    ):
        raise ProtocolError("STOP receipt is missing or bound to different preflight inputs")
    command = stop.get("唯一启动命令")
    if not isinstance(command, str) or "--execute" not in command or "--preflight-only" in command or str(config["config_path"]) not in command:
        raise ProtocolError("STOP receipt launch command is not bound to dir27 execute")
    if stop.get("lock") != config["runtime"]["global_gpu_lock"]:
        raise ProtocolError("STOP receipt lock binding changed")
    return stop_ref


def driver_state() -> dict[str, Any]:
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    cards = []
    for line in query.stdout.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            raise ProtocolError("nvidia-smi GPU query schema changed")
        cards.append({"index": int(fields[0]), "memory_used_mib": int(fields[1]), "utilization_percent": int(fields[2])})
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True)
    return {"gpus": cards, "compute_processes": [line for line in processes.stdout.splitlines() if line.strip()]}


def acquire_gpu_lock(path: str | Path) -> BinaryIO:
    lock_path = _lstat_component_chain(
        path,
        label="global GPU lock",
        allow_missing_tail=True,
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def idle_gate(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    lock_path = Path(config["runtime"]["global_gpu_lock"])
    samples: list[dict[str, Any]] = []
    for index in range(int(config["runtime"]["idle_gate_samples"])):
        state = driver_state()
        passed = (
            len(state["gpus"]) == 8
            and [row["index"] for row in state["gpus"]] == list(range(8))
            and not state["compute_processes"]
            and max((row["memory_used_mib"] for row in state["gpus"]), default=10**9) <= int(config["runtime"]["idle_gate_max_memory_mib"])
        )
        samples.append({"index": index, "passed": passed, **state})
        if not passed:
            raise ProtocolError("eight-card GPU idle gate failed")
        if index + 1 < int(config["runtime"]["idle_gate_samples"]):
            time.sleep(int(config["runtime"]["idle_gate_interval_seconds"]))
    return {"schema_version": 1, "status": "pass", "lock": str(lock_path), "samples": samples}


def _completion_paths(output: Path) -> tuple[Path, Path]:
    """Return and symlink-guard the terminal receipt and its sidecar paths."""

    _assert_output_root(output, allow_missing=False)
    completion_path = output / "completion_receipt.json"
    completion_sidecar_path = completion_path.with_name(completion_path.name + ".sha256.json")
    _guard_output_component(completion_path, output, label="completion receipt")
    _guard_output_component(completion_sidecar_path, output, label="completion receipt sidecar")
    return completion_path, completion_sidecar_path


def _terminal_completion_summary(
    output: Path,
    definition: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a validated terminal summary, or ``None`` if run is open.

    This helper is intentionally called both before and *after* acquiring the
    global GPU flock.  The in-lock call is authoritative and closes the race
    where two workers pass an optimistic pre-check concurrently.
    """

    completion_path, completion_sidecar = _completion_paths(output)
    if not completion_path.exists() and not completion_path.is_symlink():
        if completion_sidecar.exists() or completion_sidecar.is_symlink():
            raise ProtocolError("orphan completion sidecar blocks execution")
        return None
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ProtocolError("completion receipt is not a regular file")
    verify_sidecar(completion_path)
    completed = read_json(completion_path)
    if (
        completed.get("schema_version") != 1
        or completed.get("status") != "complete"
        or completed.get("experiment_id") != definition.get("experiment_id")
        or completed.get("attempt_id") != definition.get("experiment_id")
        or completed.get("request_fingerprint") != definition["request_fingerprint"]
        or completed.get("candidate_result_count") != definition.get("sample_count")
        or completed.get("expected_candidate_result_count") != definition.get("sample_count")
    ):
        raise ProtocolError("existing completion receipt is invalid or bound to another run")
    definition_path = output / "run_definition.json"
    _guard_output_component(definition_path, output, label="run definition", allow_missing_tail=False)
    if definition_path.is_symlink() or not definition_path.is_file():
        raise ProtocolError("terminal completion lacks a regular run definition")
    verify_sidecar(definition_path)
    if completed.get("run_definition") != artifact(definition_path):
        raise ProtocolError("terminal completion run-definition binding changed")
    summary_path = output / "paired_summary.json"
    _guard_output_component(summary_path, output, label="paired summary", allow_missing_tail=False)
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ProtocolError("terminal completion lacks a regular paired summary")
    summary = read_json(summary_path)
    if summary.get("status") != "complete" or summary.get("request_fingerprint") != definition["request_fingerprint"]:
        raise ProtocolError("terminal paired summary is invalid")
    if completed.get("paired_summary") != artifact(summary_path):
        raise ProtocolError("terminal completion paired-summary binding changed")
    candidate_refs = completed.get("candidate_results")
    if not isinstance(candidate_refs, list) or len(candidate_refs) != int(definition.get("sample_count", -1)):
        raise ProtocolError("terminal completion candidate inventory cardinality changed")
    candidate_root = output / "results"
    seen: set[str] = set()
    for index, ref in enumerate(candidate_refs):
        if not isinstance(ref, Mapping) or set(ref) != {"path", "nbytes", "sha256"}:
            raise ProtocolError(f"terminal candidate reference {index} is malformed")
        path = verify_ref(ref, f"terminal candidate reference {index}")
        try:
            relative = path.relative_to(candidate_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ProtocolError(f"terminal candidate reference {index} escapes output results") from exc
        if len(relative.parts) != 3 or not relative.parts[-1].endswith(".json") or relative.parts[-1].endswith(".sha256.json"):
            raise ProtocolError(f"terminal candidate reference {index} has an unexpected path")
        if str(path) in seen:
            raise ProtocolError("terminal candidate inventory contains duplicate paths")
        seen.add(str(path))
    superseded = completed.get("superseded_failure_markers")
    if not isinstance(superseded, list):
        raise ProtocolError("terminal superseded-failure inventory is malformed")
    for index, ref in enumerate(superseded):
        if not isinstance(ref, Mapping) or set(ref) != {"path", "nbytes", "sha256"}:
            raise ProtocolError(f"terminal superseded-failure reference {index} is malformed")
        path = verify_ref(ref, f"terminal superseded-failure reference {index}")
        try:
            path.relative_to(candidate_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ProtocolError(f"terminal superseded-failure reference {index} escapes output results") from exc
    _assert_no_stale_failure_markers(output)
    return summary


def _load_module_from_path(module_name: str, path: Path) -> Any:
    """Load a lazy module with a proper ``sys.modules`` binding.

    Python 3.12's :func:`dataclasses.dataclass` resolves forward references
    through ``sys.modules[cls.__module__]`` while a class is being created.
    ``module_from_spec(...).exec_module(...)`` alone leaves that entry absent
    and crashes the frozen shrinkage runtime at import time.  Install the
    module before execution and restore any prior entry on failure.
    """

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ProtocolError(f"cannot load module {module_name}: {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _load_qa_metrics() -> Any:
    path = OLD03_ROOT / "src/qa_metrics.py"
    module = _load_module_from_path("dir27_old03_qa_metrics", path)
    return module.qa_metrics


def _load_shrinkage_builder() -> Any:
    path = SHRINKAGE_ROOT / "src/runtime.py"
    module = _load_module_from_path("dir27_shrinkage_runtime", path)
    return module.ShrinkageMappedPrefixCacheBuilder


def _sync_cuda(torch: Any) -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def _gpu_peaks(torch: Any) -> dict[str, Any]:
    return {str(index): {"allocated_bytes": int(torch.cuda.max_memory_allocated(index)), "reserved_bytes": int(torch.cuda.max_memory_reserved(index))} for index in range(torch.cuda.device_count())}


def _candidate_checkpoint_complete(path: Path, row: Mapping[str, Any], definition: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        value = read_json(path)
    except (OSError, json.JSONDecodeError, ProtocolError):
        return False
    key = (str(row["dataset_id"]), str(row["sample_id"]))
    expected_bindings = baseline[key]
    return (
        value.get("status") == "complete"
        and value.get("request_fingerprint") == definition["request_fingerprint"]
        and value.get("dataset_id") == row["dataset_id"]
        and value.get("sample_id") == row["sample_id"]
        and value.get("mode") == "shrinkage_r1"
        and value.get("source_record_sha256") == row["source_record_sha256"]
        and value.get("prompt_token_ids_sha256") == row["prompt_token_ids_sha256"]
        and value.get("prompt_token_count") == row["prompt_token_count"]
        and value.get("prompt_truncated") is False
        and value.get("baseline_bindings") == {alias: binding["artifact"] for alias, binding in expected_bindings.items()}
        and isinstance(value.get("metrics"), Mapping)
        and isinstance(value.get("generation"), Mapping)
    )


def _summary_value(output: Path, rows: Sequence[Mapping[str, Any]], definition: Mapping[str, Any], baseline: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    complete: list[dict[str, Any]] = []
    missing: list[dict[str, str]] = []
    for row in rows:
        path = candidate_result_path(output, row)
        _guard_output_component(path, output, label="candidate checkpoint")
        key = (str(row["dataset_id"]), str(row["sample_id"]))
        if not _candidate_checkpoint_complete(path, row, definition, baseline):
            missing.append({"dataset_id": key[0], "sample_id": key[1], "mode": "shrinkage_r1"})
            continue
        complete.append(read_json(path))
    aggregates: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset_id"] == dataset]
        candidate_records = [value for value in complete if value.get("dataset_id") == dataset]
        aliases = {"native14": [], "nvidia_k20": [], "shrinkage_r1": candidate_records}
        for row in dataset_rows:
            key = (str(row["dataset_id"]), str(row["sample_id"]))
            if key not in baseline or not any(value.get("sample_id") == key[1] for value in candidate_records):
                continue
            for alias in ("native14", "nvidia_k20"):
                aliases[alias].append(baseline[key][alias]["record"])
            candidate = next(value for value in candidate_records if value.get("sample_id") == key[1])
            pair = {
                "dataset_id": key[0],
                "sample_id": key[1],
                "native14": artifact(old_result_path(row, BASELINE_MODES["native14"])),
                "nvidia_k20": artifact(old_result_path(row, BASELINE_MODES["nvidia_k20"])),
                "shrinkage_r1": artifact(candidate_result_path(output, row)),
                "delta_relaxed_f1_vs_nvidia_k20": float(candidate["metrics"]["relaxed_qa_f1"]) - float(baseline[key]["nvidia_k20"]["record"]["metrics"]["relaxed_qa_f1"]),
                "delta_relaxed_em_vs_nvidia_k20": float(candidate["metrics"]["relaxed_qa_exact_match"]) - float(baseline[key]["nvidia_k20"]["record"]["metrics"]["relaxed_qa_exact_match"]),
            }
            paired_rows.append(pair)
        aggregates[dataset] = {}
        for alias, values in aliases.items():
            if not values:
                continue
            aggregates[dataset][alias] = {
                "count": len(values),
                **{metric: sum(float(value["metrics"][metric]) for value in values) / len(values) for metric in METRIC_NAMES},
                "mean_generated_tokens": sum(int(value["generation"]["generated_tokens"]) for value in values) / len(values),
                "eos_rate": sum(value["generation"]["stop_reason"] == "eos" for value in values) / len(values),
                "stop_sequence_rate": sum(value["generation"]["stop_reason"] == "stop_sequence" for value in values) / len(values),
                "cap_rate": sum(bool(value["generation"]["cap_hit"]) for value in values) / len(values),
                "mean_wall_seconds": sum(float(value["systems"]["wall_seconds"]) for value in values) / len(values),
            }
    # Paired deltas are only complete when all candidate rows are present.
    paired = {}
    if not missing:
        for dataset in DATASETS:
            by_mode = aggregates[dataset]
            paired[dataset] = {
                "shrinkage_r1": {
                    "delta_relaxed_f1_vs_nvidia_k20": by_mode["shrinkage_r1"]["relaxed_qa_f1"] - by_mode["nvidia_k20"]["relaxed_qa_f1"],
                    "delta_relaxed_em_vs_nvidia_k20": by_mode["shrinkage_r1"]["relaxed_qa_exact_match"] - by_mode["nvidia_k20"]["relaxed_qa_exact_match"],
                    "delta_relaxed_f1_vs_native14": by_mode["shrinkage_r1"]["relaxed_qa_f1"] - by_mode["native14"]["relaxed_qa_f1"],
                }
            }
    return {
        "schema_version": 1,
        "status": "complete" if not missing else "incomplete",
        "request_fingerprint": definition["request_fingerprint"],
        "phase": "pilot",
        "candidate_mode": "shrinkage_r1",
        "expected_candidate_results": len(rows),
        "complete_candidate_results": len(complete),
        "missing": missing,
        "aggregates": aggregates,
        "paired": paired,
        "paired_rows": sorted(paired_rows, key=lambda value: (value["dataset_id"], value["sample_id"])),
        "baseline_binding_digest": definition["baseline_binding_digest"],
        "claim_boundary": "method-dev pilot only; not a locked-test or paper-superiority claim",
    }


def execute(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    _validate_bytecode_environment(config)
    # Recompute every CPU binding and require the preflight freeze to match.
    rows, prepared_ref, baseline, mapper, extras, model_inventories = _preflight_artifacts(config)
    definition = build_definition(config, rows, extras["source_closure"], mapper, extras["old_protocol"], baseline, prepared_ref, model_inventories)
    preflight_path = _assert_output_path(
        config["outputs"]["preflight_receipt"],
        label="dir27 preflight receipt",
        allow_missing_tail=True,
    )
    preflight_ref = artifact(preflight_path) if preflight_path.is_file() else None
    if preflight_ref is None:
        raise ProtocolError("run preflight first; no preflight receipt exists")
    verify_ref(preflight_ref, "dir27 preflight receipt")
    verify_sidecar(preflight_path)
    frozen_preflight = read_json(preflight_path)
    if frozen_preflight.get("status") != "passed_cpu_only_preflight" or frozen_preflight.get("protocol_id") != config.get("protocol_id") or frozen_preflight.get("request_fingerprint") != definition["request_fingerprint"]:
        raise ProtocolError("current inputs do not reproduce the frozen CPU preflight")
    _validate_stop_receipt(config, definition, preflight_ref)
    output = _assert_output_root(config["outputs"]["run_root"], allow_missing=True)
    if output.exists() and not output.is_dir():
        raise ProtocolError(f"candidate output root is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Re-lstat after mkdir: a concurrent replacement of any component is a
    # hard failure before a mutable artifact is opened.
    output = _assert_output_root(output, allow_missing=False)
    _assert_output_tree_nonsymlink(output)
    # Bind the terminal paths once, before any GPU work, and reuse the exact
    # objects during finalization.  This prevents a late undefined-variable
    # failure after all 96 expensive candidate generations have completed.
    completion_path = output / "completion_receipt.json"
    completion_sidecar_path = completion_path.with_name(completion_path.name + ".sha256.json")
    _guard_output_component(completion_path, output, label="completion receipt")
    _guard_output_component(completion_sidecar_path, output, label="completion receipt sidecar")
    definition_path = output / "run_definition.json"
    _guard_output_component(definition_path, output, label="run definition")
    immutable_json(definition_path, definition)
    write_sidecar(definition_path)
    # A completed run is terminal and can be inspected without reacquiring GPUs.
    terminal = _terminal_completion_summary(output, definition)
    if terminal is not None:
        return terminal
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["runtime"]["cuda_visible_devices"]:
        raise ProtocolError("GPU execution requires CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7")
    lock = acquire_gpu_lock(config["runtime"]["global_gpu_lock"])
    try:
        # Authoritative second terminal check *after* flock acquisition.  Two
        # workers may both pass the optimistic check above; only this check
        # decides whether the second worker may proceed to idle-gate/model
        # loading.
        _assert_output_root(output, allow_missing=False)
        _assert_output_tree_nonsymlink(output)
        terminal = _terminal_completion_summary(output, definition)
        if terminal is not None:
            return terminal
        gate = idle_gate(config, output)
        atomic_json(output / "idle_gate.json", gate)
        # Only now import CUDA/model packages.
        import torch
        from transformers import AutoTokenizer
        from kvcache_transfer.evaluation import TransferEvaluationEngine
        from kvcache_transfer.modeling import load_qwen3

        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            raise ProtocolError(f"exactly eight visible CUDA cards are required, got {torch.cuda.device_count()}")
        atomic_json(output / "status.json", {"schema_version": 1, "status": "loading_models", "request_fingerprint": definition["request_fingerprint"]})
        tokenizer = AutoTokenizer.from_pretrained(config["upstream"]["target_model"]["path"], local_files_only=True, use_fast=True)
        runtime = config["runtime"]
        target = load_qwen3(config["upstream"]["target_model"]["path"], devices=runtime["target_devices"], layer_counts=runtime["target_layer_counts"], role="source", attention_implementation=runtime["attention_implementation"])
        source = load_qwen3(config["upstream"]["source_model"]["path"], devices=runtime["source_devices"], layer_counts=runtime["source_layer_counts"], role="target", attention_implementation=runtime["attention_implementation"])
        sh_builder_cls = _load_shrinkage_builder()
        builder = sh_builder_cls(source_model=source, target_model=target, mapper_manifest=config["upstream"]["candidate_mapper_manifest"]["path"], token_chunk=int(runtime["mapper_token_chunk"]), verify_mapper_payloads=True)
        atomic_json(output / "model_sharding.json", {"source_32b": getattr(source, "_kvcache_transfer_layer_sharding_evidence"), "target_14b": getattr(target, "_kvcache_transfer_layer_sharding_evidence")})
        atomic_json(output / "method_provenance.json", {"candidate_mode": "shrinkage_r1", "builder": builder.provenance, "baseline_binding_digest": definition["baseline_binding_digest"]})
        engine = TransferEvaluationEngine(target_model=target, tokenizer=tokenizer, mode="mapped", mapped_prefix_builder=builder, max_length=int(config["protocol"]["max_length"]), max_gen_tokens=int(config["protocol"]["max_generation_tokens"]))
        qa_metrics = _load_qa_metrics()
        atomic_json(output / "status.json", {"schema_version": 1, "status": "running", "request_fingerprint": definition["request_fingerprint"], "completed_candidate_results": 0, "total_candidate_results": len(rows)})
        failures = 0
        completed_count = 0
        for row_index, row in enumerate(rows):
            destination = candidate_result_path(output, row)
            _validate_failure_state_if_present(destination, row, definition, output)
            if destination.exists():
                if _candidate_checkpoint_complete(destination, row, definition, baseline):
                    _supersede_failure_marker(destination, row, definition, output)
                    completed_count += 1
                    _summary = _summary_value(output, rows, definition, baseline)
                    atomic_json(output / "paired_summary.json", _summary)
                    continue
                raise ProtocolError(f"existing candidate checkpoint is invalid: {destination}")
            for device in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    # Keep the failure visible in the systems receipt while
                    # allowing execution on the pinned build that cannot reset
                    # an uninitialized allocator counter.
                    pass
            _sync_cuda(torch)
            started = time.perf_counter()
            try:
                generation = engine.generate_token_ids(row["prompt_token_ids"], {"max_gen_toks": 256, "do_sample": False, "temperature": 0.0, "until": config["protocol"]["stop_sequences"]}, truncation="error")
                _sync_cuda(torch)
                elapsed = time.perf_counter() - started
                generated_ids = [int(value) for value in generation.generated_token_ids]
                key = (str(row["dataset_id"]), str(row["sample_id"]))
                payload = {
                    "schema_version": 1,
                    "status": "complete",
                    "request_fingerprint": definition["request_fingerprint"],
                    "phase": "pilot",
                    "dataset_id": row["dataset_id"],
                    "sample_id": row["sample_id"],
                    "stratum": row["stratum"],
                    "source_record_sha256": row["source_record_sha256"],
                    "mode": "shrinkage_r1",
                    "prompt_token_count": row["prompt_token_count"],
                    "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
                    "prompt_truncated": False,
                    "prediction": generation.text,
                    "references": row["references"],
                    "metrics": qa_metrics(generation.text, row["references"]),
                    "generation": {"requested_max_tokens": 256, "generated_tokens": generation.generated_token_count, "generated_token_ids": generated_ids, "generated_token_ids_sha256": token_ids_sha256(generated_ids), "stop_reason": generation.stop_reason, "cap_hit": generation.cap_hit},
                    "baseline_bindings": {alias: baseline[key][alias]["artifact"] for alias in BASELINE_MODES},
                    "baseline_modes": BASELINE_MODES,
                    "systems": {"wall_seconds": elapsed, "gpu_peak": _gpu_peaks(torch)},
                }
                immutable_json(destination, payload)
                # A valid result seals any prior bound failure marker.  If the
                # marker is malformed/mismatched, this raises and completion
                # is intentionally not published.
                _supersede_failure_marker(destination, row, definition, output)
                completed_count += 1
                _summary = _summary_value(output, rows, definition, baseline)
                atomic_json(output / "paired_summary.json", _summary)
                print(f"[complete] {row_index + 1}/{len(rows)} {key[0]}/{key[1]} f1={payload['metrics']['relaxed_qa_f1']:.6f}")
            except BaseException as exc:
                failures += 1
                failure_path = _failure_marker_path(destination)
                failure_sidecar = failure_path.with_name(failure_path.name + ".sha256.json")
                if failure_sidecar.exists() or failure_sidecar.is_symlink():
                    if failure_sidecar.is_symlink():
                        raise ProtocolError(f"failure marker sidecar is a symlink: {failure_sidecar}")
                    # The marker is deliberately mutable while a run is open;
                    # discard only its old digest before replacing both files
                    # atomically in sequence.
                    failure_sidecar.unlink()
                _guard_output_component(failure_path, output, label="failure marker")
                atomic_json(failure_path, {"schema_version": 1, "status": "failed", "request_fingerprint": definition["request_fingerprint"], "dataset_id": row["dataset_id"], "sample_id": row["sample_id"], "mode": "shrinkage_r1", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
                print(f"[failed] {row['dataset_id']}/{row['sample_id']}: {exc}", file=sys.stderr)
                break
            finally:
                engine.clear_prefix_cache()
            atomic_json(output / "status.json", {"schema_version": 1, "status": "failed" if failures else "running", "request_fingerprint": definition["request_fingerprint"], "completed_candidate_results": completed_count, "total_candidate_results": len(rows), "failures": failures})
            if failures:
                break
        summary = _summary_value(output, rows, definition, baseline)
        atomic_json(output / "paired_summary.json", summary)
        if failures or summary["status"] != "complete":
            atomic_json(output / "status.json", {"schema_version": 1, "status": "failed", "request_fingerprint": definition["request_fingerprint"], "completed_candidate_results": summary["complete_candidate_results"], "total_candidate_results": len(rows), "failures": failures})
            raise ProtocolError("candidate pilot did not complete; resume after fixing the failure")
        # Never seal a run while a stale failure marker (or symlinked output
        # component) remains anywhere beneath results.
        _assert_no_stale_failure_markers(output)
        inventory = [artifact(candidate_result_path(output, row)) for row in rows]
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "experiment_id": definition["experiment_id"],
            "attempt_id": definition["experiment_id"],
            "request_fingerprint": definition["request_fingerprint"],
            "run_definition": artifact(definition_path),
            "preflight": preflight_ref,
            "idle_gate": artifact(output / "idle_gate.json"),
            "model_sharding": artifact(output / "model_sharding.json"),
            "method_provenance": artifact(output / "method_provenance.json"),
            "paired_summary": artifact(output / "paired_summary.json"),
            "candidate_results": inventory,
            "candidate_result_count": len(inventory),
            "expected_candidate_result_count": len(rows),
            "superseded_failure_markers": _superseded_failure_inventory(output),
            "baseline_binding_digest": definition["baseline_binding_digest"],
            "baseline_rerun": False,
            "locked_test_access": False,
            "official_dev_or_test_opened": False,
            "claim_boundary": "method-dev pilot only; not a locked-test or paper-superiority claim",
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        # Re-guard immediately before publication in case an external actor
        # replaced a path while the model was running.
        completion_path = output / "completion_receipt.json"
        completion_sidecar_path = completion_path.with_name(completion_path.name + ".sha256.json")
        _guard_output_component(completion_path, output, label="completion receipt")
        _guard_output_component(completion_sidecar_path, output, label="completion receipt sidecar")
        immutable_json(completion_path, receipt)
        write_sidecar(completion_path)
        verify_sidecar(completion_path)
        if not completion_sidecar_path.is_file() or completion_sidecar_path.is_symlink():
            raise ProtocolError("completion receipt sidecar was not durably published")
        atomic_json(output / "status.json", {"schema_version": 1, "status": "complete", "request_fingerprint": definition["request_fingerprint"], "completed_candidate_results": len(inventory), "completion_receipt_sha256": sha256_file(completion_path)})
        return summary
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        preflight(args.config)
        return 0
    execute(args.config)
    return 0


__all__ = [
    "ProtocolError",
    "artifact",
    "build_definition",
    "candidate_result_path",
    "canonical_sha256",
    "execute",
    "load_baseline_bindings",
    "load_config",
    "load_pilot_rows",
    "main",
    "old_result_path",
    "preflight",
    "source_byte_closure",
    "token_ids_sha256",
]
