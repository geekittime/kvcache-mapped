"""Stage orchestration and immutable PRE-FORMAL stop handling."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .allocation import solve_allocation_rows, validate_allocation_report
from .common import (
    ProtocolError,
    atomic_json,
    atomic_jsonl,
    config_file_binding,
    claim_artifact,
    file_spec,
    finalize_publication,
    formal_config_path,
    implementation_inventory,
    implementation_inventory_digest,
    json_exact,
    require_fresh_python_bootstrap,
    safe_child,
    strict_json_load,
    strict_json_loads,
    validate_runtime_binding,
    validate_publication,
    verify_config_file_binding,
)
from .inputs import InputBundle, load_config, load_input_bundle
from .preflight import (
    preflight,
    validate_config_shape,
    validate_formal_config,
    validate_preformal_stop_binding,
)
from .quota import solve_quota_search, targets_from_contract, validate_quota_proof, validate_quota_vector


STAGE_IDENTITIES = {
    "quota_search": {"protocol_id": "quota_capacity_repair_v2", "artifact_kind": "quota_search"},
    "allocation": {"protocol_id": "quota_capacity_repair_v2", "artifact_kind": "repaired_allocation"},
    "deep_validation": {"protocol_id": "quota_capacity_repair_v2", "artifact_kind": "deep_validation"},
}
STAGE_STATUSES_BLOCKED = {
    "quota_search": "published_preformal_quota_search_placeholder_gate",
    "allocation": "published_preformal_allocation_placeholder_gate",
    "deep_validation": "published_preformal_deep_validation_placeholder_gate",
}


def _active_implementation_snapshot(root: Path, config_path: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Require the active PRE-FORMAL marker and return a fresh source digest."""

    validate_preformal_stop_binding(config_path, root, config, require_current_inventory=True)
    rows = implementation_inventory(root)
    return rows, implementation_inventory_digest(rows)


def _assert_implementation_snapshot(
    root: Path,
    expected_rows: list[dict[str, Any]],
    expected_digest: str,
    *,
    name: str,
) -> None:
    """Fail closed if code/docs/tests drift during a worker run."""

    observed = implementation_inventory(root)
    digest = implementation_inventory_digest(observed)
    if digest != expected_digest or observed != expected_rows:
        raise ProtocolError(f"implementation inventory drifted during {name}")


def _stage_dir(root: Path, config: Mapping[str, Any], stage: str) -> Path:
    return safe_child(root, config["outputs"][stage], allow_missing=True)


def _write_blocked_stage(root: Path, config: Mapping[str, Any], stage: str, pf: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = _stage_dir(root, config, stage)
    identity = STAGE_IDENTITIES[stage]
    claim_artifact(output_dir, identity)
    receipt = {
        "schema_version": 1,
        **identity,
        "status": "PRE_FORMAL_PLACEHOLDER_GATE",
        "cpu_only": True,
        "completed": False,
        "gate": "allocation_v3_deep_receipt",
        "gate_status": pf["allocation_v3_deep_receipt"],
        "source_graph_PUBLISHED": pf["source_bindings"]["graph_report"],
        "source_hyperedges": pf["source_bindings"]["hyperedges"],
        "thresholds_reused_byte_for_byte": True,
        "answer_values_read": False,
        "model_outcomes_read": False,
        "formal_execution_performed": False,
        "gpu_execution_performed": False,
        "claim_boundary": "PRE-FORMAL STOP placeholder only; no quota/allocation/deep result claimed",
    }
    atomic_json(output_dir / "receipt.json", receipt)
    marker = finalize_publication(
        output_dir,
        ["receipt.json"],
        identity=identity,
        status=STAGE_STATUSES_BLOCKED[stage],
    )
    return {
        "artifact": str(output_dir),
        "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
        "receipt": file_spec(output_dir / "receipt.json"),
        "status": receipt["status"],
        "ordered_inventory_sha256": marker["ordered_inventory_sha256"],
    }


def publish_placeholder_stages(root: Path, config: Mapping[str, Any], pf: Mapping[str, Any]) -> dict[str, Any]:
    """Publish three independent blocked markers, never touching source dirs."""

    stages: dict[str, Any] = {}
    for stage in ("quota_search", "allocation", "deep_validation"):
        output_dir = _stage_dir(root, config, stage)
        # A prior crash may have completed an early marker before the next
        # parent directory was created.  Reuse it only after a full immutable
        # validation; never overwrite or append to an existing stage.
        if output_dir.exists():
            marker = validate_publication(
                output_dir,
                ["receipt.json"],
                identity=STAGE_IDENTITIES[stage],
                status=STAGE_STATUSES_BLOCKED[stage],
            )
            stages[stage] = {
                "artifact": str(output_dir),
                "PUBLISHED": file_spec(output_dir / "PUBLISHED.json"),
                "receipt": file_spec(output_dir / "receipt.json"),
                "status": "PRE_FORMAL_PLACEHOLDER_GATE",
                "ordered_inventory_sha256": marker["ordered_inventory_sha256"],
                "reused_valid_marker": True,
            }
        else:
            stages[stage] = _write_blocked_stage(root, config, stage, pf)
    return stages


def _worker_environment(
    config: Mapping[str, Any],
    config_binding: Mapping[str, Any],
    cache_prefix: str,
    *,
    seed: str,
    kind: str,
) -> dict[str, str]:
    """Build the exact child environment shared by every worker kind."""

    runtime = config.get("runtime_binding")
    if not isinstance(runtime, Mapping):
        raise ProtocolError("runtime binding is missing before worker launch")
    validate_runtime_binding(runtime)
    environment = {
        **os.environ,
        **dict(config["solver"]["required_environment"]),
        "PYTHONHASHSEED": seed,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": cache_prefix,
        "CUDA_VISIBLE_DEVICES": "",
        "FORMAL_CONFIG_PATH": str(config_binding["path"]),
        "FORMAL_CONFIG_NBYTES": str(config_binding["nbytes"]),
        "FORMAL_CONFIG_SHA256": str(config_binding["sha256"]),
        "FORMAL_PYTHON_EXECUTABLE": str(runtime["python_executable"]),
        "VIRTUAL_ENV": str(runtime["python_prefix"]),
    }
    gate = config.get("preformal_stop_binding")
    if not isinstance(gate, Mapping):
        raise ProtocolError("local PRE-FORMAL stop binding is absent before worker launch")
    environment.update({
        "PREFORMAL_STOP_PATH": str(gate["path"]),
        "PREFORMAL_STOP_NBYTES": str(gate["nbytes"]),
        "PREFORMAL_STOP_SHA256": str(gate["sha256"]),
        "PREFORMAL_INVENTORY_SHA256": str(gate["implementation_inventory_sha256"]),
    })
    if kind == "quota":
        environment["QUOTA_REPAIR_SEED"] = seed
    elif kind == "allocation":
        environment["HYPERGRAPH_HIGHS_RANDOM_SEED"] = seed
    else:
        raise ProtocolError(f"unknown worker kind: {kind}")
    return environment


def _worker_config_args(config_path: Path, config_binding: Mapping[str, Any]) -> list[str]:
    """Return mandatory config path/content arguments for every worker."""

    return [
        "--config",
        str(config_path),
        "--config-nbytes",
        str(config_binding["nbytes"]),
        "--config-sha256",
        str(config_binding["sha256"]),
    ]


def _run_quota_replica(
    bundle: InputBundle,
    config: Mapping[str, Any],
    root: Path,
    config_path: Path,
    config_binding: Mapping[str, Any],
    seed: str,
    temporary_output: Path,
) -> dict[str, Any]:
    """Run one quota-search worker in a distinct process and validate stdout."""

    cache_prefix = require_fresh_python_bootstrap()
    verify_config_file_binding(config_path, config_binding)
    environment = _worker_environment(config, config_binding, cache_prefix, seed=seed, kind="quota")
    runtime = config["runtime_binding"]
    command = [
        str(runtime["python_executable"]),
        str(root / "scripts" / "quota_worker.py"),
        *_worker_config_args(config_path, config_binding),
        "--output",
        str(temporary_output),
    ]
    result = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True, check=False)
    if result.returncode != 0 or result.stderr != "":
        raise ProtocolError(f"quota-search replica {seed} failed or emitted stderr")
    try:
        worker = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ProtocolError("quota-search worker stdout is not JSON") from exc
    if not isinstance(worker, dict) or set(worker) != {"process_id", "python_hash_seed", "seed", "result", "config_binding", "runtime_identity", "preformal_stop_binding"}:
        raise ProtocolError("quota-search worker stdout schema mismatch")
    if type(worker.get("process_id")) is not int or worker["process_id"] <= 0 or worker.get("python_hash_seed") != seed or worker.get("seed") != int(seed) or not isinstance(worker.get("result"), dict):
        raise ProtocolError("quota-search worker PID/seed binding mismatch")
    if not json_exact(worker.get("config_binding"), dict(config_binding)):
        raise ProtocolError("quota-search worker config binding mismatch")
    if not json_exact(worker.get("preformal_stop_binding"), dict(config["preformal_stop_binding"])):
        raise ProtocolError("quota-search worker PRE-FORMAL binding mismatch")
    _validate_worker_runtime_identity(worker.get("runtime_identity"), config)
    verify_config_file_binding(config_path, config_binding)
    payload = temporary_output.read_bytes()
    parsed = strict_json_loads(payload)
    if not isinstance(parsed, dict) or set(parsed) != {"quota_vector", "proof"}:
        raise ProtocolError("quota-search worker payload schema mismatch")
    if not json_exact(parsed, worker["result"]):
        raise ProtocolError("quota-search worker stdout/output mismatch")
    return {
        "seed": int(seed),
        "python_hash_seed": seed,
        "process_id": worker["process_id"],
        "result": parsed,
        "bytes": payload,
        "config_binding": dict(config_binding),
        "runtime_identity": dict(worker["runtime_identity"]),
        "preformal_stop_binding": dict(worker["preformal_stop_binding"]),
    }


def _validate_worker_runtime_identity(identity: Any, config: Mapping[str, Any]) -> None:
    """Validate a child runtime identity against the frozen venv binding."""

    if not isinstance(identity, Mapping) or set(identity) != {
        "python_executable",
        "python_prefix",
        "python_real_executable",
        "python_real_executable_sha256",
        "python_major_minor",
        "python_version",
    }:
        raise ProtocolError("worker runtime identity schema mismatch")
    expected = config.get("runtime_binding")
    if not isinstance(expected, Mapping):
        raise ProtocolError("worker runtime binding is absent from config")
    if identity.get("python_executable") != expected.get("python_executable") or identity.get("python_prefix") != expected.get("python_prefix") or identity.get("python_real_executable") != expected.get("python_real_executable") or identity.get("python_real_executable_sha256") != expected.get("python_real_executable_sha256") or identity.get("python_major_minor") != expected.get("python_major_minor"):
        raise ProtocolError("worker runtime identity does not match config")
    if not isinstance(identity.get("python_version"), str) or not identity["python_version"]:
        raise ProtocolError("worker Python version identity is invalid")


def run_cpu_repair(
    config_path: Path,
    root: Path,
    *,
    allow_execution: bool = False,
) -> dict[str, Any]:
    """Run the eventual CPU repair only when explicitly enabled.

    The default is intentionally a no-op placeholder path.  This prevents a
    pre-formal command from accidentally launching the expensive 31k-row MILP
    while allocation-v3's independent deep receipt is still being produced.
    """

    require_fresh_python_bootstrap()
    config_path = formal_config_path(root, Path(config_path))
    config = load_config(config_path)
    validate_config_shape(config)
    config_binding = validate_formal_config(config_path, root, config)
    pf = preflight(config_path)
    if not allow_execution:
        return {"status": "PRE_FORMAL_STOP", "preflight": pf, "stages": publish_placeholder_stages(root, config, pf)}
    if pf["status"] != "PRE_FORMAL_READY_FOR_CPU_REPAIR":
        raise ProtocolError("CPU repair requires the completed allocation_v3 deep receipt")
    if config["preformal_stop_binding"].get("role") != "active_snapshot":
        raise ProtocolError("CPU repair requires an active local PRE-FORMAL source snapshot")
    inventory_rows, inventory_digest = _active_implementation_snapshot(root, config_path, config)
    bundle = load_input_bundle(config, require_deep_allocation=True)
    result = _run_ready_repair(
        bundle,
        config,
        root,
        config_path,
        config_binding,
        inventory_rows=inventory_rows,
        inventory_digest=inventory_digest,
    )
    _assert_implementation_snapshot(root, inventory_rows, inventory_digest, name="CPU repair completion")
    return result


def _run_ready_repair(
    bundle: InputBundle,
    config: Mapping[str, Any],
    root: Path,
    config_path: Path,
    config_binding: Mapping[str, Any],
    *,
    inventory_rows: list[dict[str, Any]] | None = None,
    inventory_digest: str | None = None,
) -> dict[str, Any]:
    """Execute stages A--D after all immutable external gates are ready."""

    cache_prefix = require_fresh_python_bootstrap()
    formal_config_path(root, config_path)
    verify_config_file_binding(config_path, config_binding)
    validate_runtime_binding(config["runtime_binding"])
    if inventory_rows is None or inventory_digest is None:
        inventory_rows, inventory_digest = _active_implementation_snapshot(root, config_path, config)
    else:
        _assert_implementation_snapshot(root, inventory_rows, inventory_digest, name="CPU repair start")
    quota_dir = _stage_dir(root, config, "quota_search")
    identity = STAGE_IDENTITIES["quota_search"]
    claim_artifact(quota_dir, identity)
    seeds = config["solver"]["replica_python_hash_seeds"]
    if seeds != ["1", "777"] or config["solver"]["replica_highs_random_seeds"] != [1, 777]:
        raise ProtocolError("quota-search seed pairing is not frozen")
    with tempfile.TemporaryDirectory(prefix="quota_repair_replicas_") as temporary:
        replicas = [
            _run_quota_replica(bundle, config, root, config_path, config_binding, seed, Path(temporary) / f"result_{seed}.json")
            for seed in seeds
        ]
    _assert_implementation_snapshot(root, inventory_rows, inventory_digest, name="quota worker completion")
    verify_config_file_binding(config_path, config_binding)
    if len({replica["process_id"] for replica in replicas}) != 2 or len({replica["bytes"] for replica in replicas}) != 1:
        raise ProtocolError("quota-search replicas are not distinct/byte-identical")
    quota_payload = replicas[0]["result"]
    targets = targets_from_contract(bundle.contract, config["upper_quotas"])
    proof = quota_payload.get("proof")
    quota_vector = quota_payload.get("quota_vector")
    if not isinstance(proof, dict) or not isinstance(quota_vector, dict):
        raise ProtocolError("quota-search result missing proof/vector")
    validate_quota_vector(quota_vector, targets)
    validate_quota_proof(proof, targets)
    atomic_json(quota_dir / "quota.json", quota_vector)
    atomic_json(
        quota_dir / "quota_report.json",
        {
            "schema_version": 1,
            **identity,
            "status": "quota_search_proven_optimal",
            "quota_vector": quota_vector,
            "proof": proof,
            "replicas": [
                {key: value for key, value in replica.items() if key != "bytes"}
                for replica in replicas
            ],
            "replica_quota_bytes_identical": True,
            "formal_config": dict(config_binding),
            "answer_values_read": False,
            "model_outcomes_read": False,
        },
    )
    atomic_json(
        quota_dir / "receipt.json",
        {
            "schema_version": 1,
            **identity,
            "status": "pass_quota_capacity_repair",
            "cpu_only": True,
            "completed": True,
            "quota": file_spec(quota_dir / "quota.json"),
            "report": file_spec(quota_dir / "quota_report.json"),
            "allocation_v3_deep_receipt": bundle.allocation_deep_binding,
            "formal_config": dict(config_binding),
            "claim_boundary": "capacity quota proof only; no model outcomes or locked metrics evaluated",
        },
    )
    quota_marker = finalize_publication(quota_dir, ["quota.json", "quota_report.json", "receipt.json"], identity=identity, status="published_complete_quota_search")

    # Stage D uses only C's frozen vector.
    allocation_dir = _stage_dir(root, config, "allocation")
    allocation_identity = STAGE_IDENTITIES["allocation"]
    claim_artifact(allocation_dir, allocation_identity)
    with tempfile.TemporaryDirectory(prefix="quota_repair_alloc_replicas_") as temporary:
        alloc_replicas = []
        for seed in seeds:
            env = _worker_environment(config, config_binding, cache_prefix, seed=seed, kind="allocation")
            out = Path(temporary) / f"selection_{seed}.jsonl"
            # The worker itself is intentionally tiny and emits a strict JSON
            # envelope on stdout; it consumes the already loaded immutable rows
            # through the pinned paths, never answers/outcomes.
            command = [
                str(config["runtime_binding"]["python_executable"]),
                str(root / "scripts" / "allocation_worker.py"),
                *_worker_config_args(config_path, config_binding),
                "--quota",
                str(quota_dir / "quota.json"),
                "--output",
                str(out),
            ]
            result = subprocess.run(command, cwd=root, env=env, text=True, capture_output=True, check=False)
            if result.returncode != 0 or result.stderr != "":
                raise ProtocolError(f"allocation replica {seed} failed or emitted stderr")
            try:
                worker = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                raise ProtocolError("allocation worker stdout is not JSON") from exc
            if not isinstance(worker, dict) or set(worker) != {"process_id", "python_hash_seed", "seed", "selection", "solver", "config_binding", "runtime_identity", "preformal_stop_binding"}:
                raise ProtocolError("allocation worker stdout schema mismatch")
            if type(worker.get("process_id")) is not int or worker["process_id"] <= 0 or worker.get("python_hash_seed") != seed or worker.get("seed") != int(seed):
                raise ProtocolError("allocation worker PID/seed typed mismatch")
            if not json_exact(worker.get("config_binding"), dict(config_binding)):
                raise ProtocolError("allocation worker config binding mismatch")
            if not json_exact(worker.get("preformal_stop_binding"), dict(config["preformal_stop_binding"])):
                raise ProtocolError("allocation worker PRE-FORMAL binding mismatch")
            _validate_worker_runtime_identity(worker.get("runtime_identity"), config)
            selection_bytes = out.read_bytes()
            alloc_replicas.append({"seed": int(seed), "python_hash_seed": seed, "process_id": worker["process_id"], "worker": worker, "selection_bytes": selection_bytes, "config_binding": dict(config_binding), "preformal_stop_binding": dict(worker["preformal_stop_binding"])})
            verify_config_file_binding(config_path, config_binding)
    verify_config_file_binding(config_path, config_binding)
    if len({replica["process_id"] for replica in alloc_replicas}) != 2 or len({replica["selection_bytes"] for replica in alloc_replicas}) != 1:
        raise ProtocolError("allocation replicas are not distinct/byte-identical")
    for replica in alloc_replicas:
        validate_allocation_report(replica["worker"].get("solver"), config, expected_seed=replica["seed"])
    selection_bytes = alloc_replicas[0]["selection_bytes"]
    atomic_write_selection = allocation_dir / "allocation.jsonl"
    from .common import atomic_write  # local import avoids a public API cycle
    atomic_write(atomic_write_selection, selection_bytes)
    atomic_json(
        allocation_dir / "allocation_report.json",
        {
            "schema_version": 1,
            **allocation_identity,
            "status": "allocation_proven_optimal",
            "quota": file_spec(quota_dir / "quota.json"),
            "replicas": [
                {key: value for key, value in replica.items() if key != "selection_bytes"}
                for replica in alloc_replicas
            ],
            "replica_selection_byte_identical": True,
            "selection": file_spec(atomic_write_selection),
            "formal_config": dict(config_binding),
            "answer_values_read": False,
            "model_outcomes_read": False,
        },
    )
    atomic_json(
        allocation_dir / "receipt.json",
        {
            "schema_version": 1,
            **allocation_identity,
            "status": "pass_repaired_allocation",
            "cpu_only": True,
            "completed": True,
            "quota_search": file_spec(quota_dir / "PUBLISHED.json"),
            "report": file_spec(allocation_dir / "allocation_report.json"),
            "formal_config": dict(config_binding),
            "claim_boundary": "SHA-cost allocation under repaired quotas only; no model outcomes or locked metrics evaluated",
        },
    )
    allocation_marker = finalize_publication(allocation_dir, ["allocation.jsonl", "allocation_report.json", "receipt.json"], identity=allocation_identity, status="published_complete_repaired_allocation")
    _assert_implementation_snapshot(root, inventory_rows, inventory_digest, name="allocation worker completion")

    # A separate deep process is required by the protocol.  The caller may run
    # scripts/deep_validate.py; we expose bindings here but never label the
    # pre-formal stop as deep-complete implicitly.
    return {
        "status": "CPU_REPAIR_COMPLETE_A_B_C_D_AWAITING_DEEP",
        "quota_search": {"PUBLISHED": file_spec(quota_dir / "PUBLISHED.json"), "ordered_inventory_sha256": quota_marker["ordered_inventory_sha256"]},
        "allocation": {"PUBLISHED": file_spec(allocation_dir / "PUBLISHED.json"), "ordered_inventory_sha256": allocation_marker["ordered_inventory_sha256"]},
        "deep_validation_required": True,
    }


def validate_stage(stage: str, root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one independent stage marker; useful for tamper tests."""

    if stage not in STAGE_IDENTITIES:
        raise ProtocolError(f"unknown stage: {stage}")
    output_dir = _stage_dir(root, config, stage)
    if stage in STAGE_STATUSES_BLOCKED:
        marker = validate_publication(output_dir, ["receipt.json"], identity=STAGE_IDENTITIES[stage], status=STAGE_STATUSES_BLOCKED[stage])
        receipt = strict_json_load(output_dir / "receipt.json")
        if not isinstance(receipt, dict) or receipt.get("status") != "PRE_FORMAL_PLACEHOLDER_GATE" or receipt.get("completed") is not False or receipt.get("formal_execution_performed") is not False or receipt.get("gpu_execution_performed") is not False:
            raise ProtocolError("blocked stage marker could be mistaken for a completed/formal result")
        return marker
    raise ProtocolError("ready-stage validator is intentionally delegated to deep_validate.py")
