"""CPU-only preflight and the external allocation-v3 deep-receipt gate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .common import (
    ProtocolError,
    config_file_binding,
    formal_config_path,
    implementation_inventory,
    implementation_inventory_digest,
    require_exact_keys,
    runtime_binding_shape,
    strict_json_load,
    validate_runtime_binding,
    verify_config_file_binding,
)
from .inputs import InputBundle, load_config, load_input_bundle


def _contains_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def validate_config_shape(config: Mapping[str, Any]) -> None:
    """Validate immutable protocol fields before touching external inputs."""

    if config.get("schema_version") != 1 or config.get("protocol_id") != "quota_capacity_repair_v2":
        raise ProtocolError("quota-repair config identity mismatch")
    if config.get("status") != "PRE_FORMAL_CPU_ONLY":
        raise ProtocolError("quota-repair config must remain PRE_FORMAL_CPU_ONLY")
    if config.get("source_protocol") != "pairwise_direct_collision_hypergraph_feasibility_v3":
        raise ProtocolError("source protocol binding mismatch")
    if config.get("formal_execution", {}).get("allowed") is not False or config.get("formal_execution", {}).get("gpu_allowed") is not False:
        raise ProtocolError("formal/GPU execution must be disabled in PRE-FORMAL config")
    bootstrap = config.get("runtime_bootstrap")
    if bootstrap != {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "absolute_path_must_be_absent_before_local_import",
        "sys_dont_write_bytecode": True,
        "sys_pycache_prefix_matches_env": True,
        "prefix_parent_policy": "all_existing_parents_real_directories_no_symlinks",
        "local_src_import_guard": True,
    }:
        raise ProtocolError("fresh no-bytecode bootstrap policy is not frozen")
    if config.get("partitions_order") != ["screen", "confirm"]:
        raise ProtocolError("partition order is not frozen")
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {"quota_search", "allocation", "deep_validation", "preformal_stop"}:
        raise ProtocolError("output path schema is not frozen")
    if outputs.get("quota_search") != "artifacts/quota_search" or outputs.get("allocation") != "artifacts/allocation" or outputs.get("deep_validation") != "artifacts/validations/deep":
        raise ProtocolError("stage output paths are not canonical")
    if not isinstance(outputs.get("preformal_stop"), str) or not outputs["preformal_stop"].startswith("artifacts/preformal/PRE_FORMAL_STOP_") or not outputs["preformal_stop"].endswith(".json"):
        raise ProtocolError("PRE-FORMAL output marker path is not canonical")
    target_order = config.get("target_order")
    if target_order != list(range(18)):
        raise ProtocolError("public target order must be exactly 0..17")
    ratio = config.get("ratio_constraint")
    if not isinstance(ratio, dict) or ratio.get("absolute_difference_maximum") != 1:
        raise ProtocolError("screen/confirm ratio tolerance is not frozen to one")
    screen_to_confirm = ratio.get("screen_to_confirm")
    if not isinstance(screen_to_confirm, dict) or screen_to_confirm.get("screen_multiplier") != 2 or screen_to_confirm.get("confirm_multiplier") != 1:
        raise ProtocolError("screen/confirm ratio multipliers are not frozen")
    solver = config.get("solver")
    if not isinstance(solver, dict):
        raise ProtocolError("solver configuration missing")
    if solver.get("replica_python_hash_seeds") != ["1", "777"] or solver.get("replica_highs_random_seeds") != [1, 777]:
        raise ProtocolError("two required replica seeds are not frozen")
    env = solver.get("required_environment")
    if env != {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}:
        raise ProtocolError("single-thread solver environment is not frozen")
    forbidden = config.get("forbidden_data")
    if not isinstance(forbidden, dict) or forbidden.get("model_outcomes_read") is not False or forbidden.get("answer_values_read") is not False:
        raise ProtocolError("answer/outcome policy is not fail-closed")
    formal = config.get("formal_config")
    if not isinstance(formal, dict) or set(formal) != {"relative_path", "single_canonical_path", "worker_binding"} or formal.get("relative_path") != "config.json" or formal.get("single_canonical_path") is not True or not isinstance(formal.get("worker_binding"), str):
        raise ProtocolError("formal config path policy is not frozen")
    runtime = config.get("runtime_binding")
    runtime_binding_shape(runtime)
    upstream = config.get("superseded_upstream")
    if not isinstance(upstream, dict) or set(upstream) != {"experiment_root", "preformal_stop_v11", "status", "reuse_policy"}:
        raise ProtocolError("superseded upstream binding is missing")
    if not isinstance(upstream.get("experiment_root"), str) or not isinstance(upstream.get("status"), str) or not isinstance(upstream.get("reuse_policy"), str):
        raise ProtocolError("superseded upstream binding types mismatch")
    previous = upstream.get("preformal_stop_v11")
    if not isinstance(previous, dict) or set(previous) != {"path", "nbytes", "sha256"} or type(previous.get("nbytes")) is not int or previous["nbytes"] <= 0 or not isinstance(previous.get("path"), str) or not isinstance(previous.get("sha256"), str) or len(previous["sha256"]) != 64:
        raise ProtocolError("superseded v11 receipt binding mismatch")
    preformal = config.get("preformal_stop_binding")
    expected_preformal_keys = {
        "path",
        "nbytes",
        "sha256",
        "implementation_inventory_sha256",
        "status",
        "role",
        "inventory_policy",
    }
    if not isinstance(preformal, Mapping) or set(preformal) != expected_preformal_keys:
        raise ProtocolError("local PRE-FORMAL stop binding is missing or has extra keys")
    if not isinstance(preformal.get("path"), str) or not preformal["path"].startswith("/"):
        raise ProtocolError("local PRE-FORMAL stop path must be absolute")
    if type(preformal.get("nbytes")) is not int or preformal["nbytes"] <= 0:
        raise ProtocolError("local PRE-FORMAL stop nbytes must be a positive integer")
    for key in ("sha256", "implementation_inventory_sha256"):
        value = preformal.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ProtocolError(f"local PRE-FORMAL stop {key} must be lowercase hexadecimal")
    if preformal.get("status") != "PRE_FORMAL_STOP":
        raise ProtocolError("local PRE-FORMAL stop status mismatch")
    if preformal.get("role") not in {"previous_snapshot", "active_snapshot"}:
        raise ProtocolError("local PRE-FORMAL stop role is invalid")
    if preformal.get("inventory_policy") != {
        "algorithm": "canonical_json_array_sha256",
        "excludes": ["artifacts", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "config.json"],
        "reject_symlink": True,
        "reject_nonregular": True,
    }:
        raise ProtocolError("local PRE-FORMAL inventory policy is not frozen")


def validate_preformal_stop_binding(
    config_path: Path,
    root: Path,
    config: Mapping[str, Any],
    *,
    require_current_inventory: bool | None = None,
) -> dict[str, Any]:
    """Verify the local immutable PRE-FORMAL marker and source inventory.

    A ``previous_snapshot`` marker is accepted only as a transition input for
    publishing the next append-only marker.  Any execution-capable path must
    use ``active_snapshot`` and therefore rehash the implementation tree.
    """

    binding = config.get("preformal_stop_binding")
    if not isinstance(binding, Mapping):
        raise ProtocolError("local PRE-FORMAL stop binding is missing")
    marker_path = Path(binding["path"])
    # Marker must remain inside this experiment's preformal directory and may
    # not be redirected through a symlinked parent.
    expected_dir = Path(root) / "artifacts" / "preformal"
    try:
        marker_path.relative_to(expected_dir)
    except ValueError as exc:
        raise ProtocolError("local PRE-FORMAL marker escapes experiment root") from exc
    if binding.get("role") == "active_snapshot":
        try:
            relative_marker = marker_path.relative_to(Path(root)).as_posix()
        except ValueError as exc:
            raise ProtocolError("active PRE-FORMAL marker is outside experiment root") from exc
        if config.get("outputs", {}).get("preformal_stop") != relative_marker:
            raise ProtocolError("active PRE-FORMAL marker/output path mismatch")
    verify_config_file_binding(
        marker_path,
        {key: binding[key] for key in ("path", "nbytes", "sha256")},
        name="local PRE-FORMAL stop marker",
    )
    marker = strict_json_load(marker_path)
    if not isinstance(marker, Mapping) or marker.get("status") != "PRE_FORMAL_STOP" or marker.get("experiment_root") != str(root):
        raise ProtocolError("local PRE-FORMAL marker semantics mismatch")
    marker_digest = marker.get("implementation_inventory_sha256")
    if marker_digest != binding.get("implementation_inventory_sha256"):
        raise ProtocolError("local PRE-FORMAL implementation digest binding mismatch")
    if require_current_inventory is None:
        require_current_inventory = binding.get("role") == "active_snapshot"
    if require_current_inventory:
        inventory = implementation_inventory(Path(root))
        digest = implementation_inventory_digest(inventory)
        if digest != binding["implementation_inventory_sha256"]:
            raise ProtocolError("implementation inventory drifted from active PRE-FORMAL marker")
        marker_inventory = marker.get("implementation_inventory")
        if not isinstance(marker_inventory, list) or marker_inventory != inventory:
            raise ProtocolError("active PRE-FORMAL marker inventory rows mismatch")
        if _contains_key(marker, "formal_config"):
            raise ProtocolError("active PRE-FORMAL marker contains a mutable config echo")
        if marker.get("implementation_inventory_start_sha256") != digest or marker.get("implementation_inventory_end_sha256") != digest or marker.get("implementation_inventory_start_equals_end") is not True:
            raise ProtocolError("active PRE-FORMAL marker start/end inventory evidence mismatch")
        cpu_tests = marker.get("cpu_tests")
        if not isinstance(cpu_tests, Mapping):
            raise ProtocolError("active PRE-FORMAL marker CPU test evidence is missing")
        environment = cpu_tests.get("environment")
        required_environment = {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": environment.get("PYTHONPYCACHEPREFIX") if isinstance(environment, Mapping) else None,
            "PYTHONHASHSEED": environment.get("PYTHONHASHSEED") if isinstance(environment, Mapping) else None,
            **dict(config["solver"]["required_environment"]),
        }
        if not isinstance(environment, Mapping) or environment != required_environment or environment.get("PYTHONHASHSEED") not in {"1", "777"} or not isinstance(environment.get("PYTHONPYCACHEPREFIX"), str) or not environment["PYTHONPYCACHEPREFIX"].startswith("/"):
            raise ProtocolError("active PRE-FORMAL CPU environment evidence is not exact")
        bootstrap = cpu_tests.get("bootstrap")
        if bootstrap != {
            "prefix_initially_absent": True,
            "prefix_parent_policy": "all_existing_parents_real_directories_no_symlinks",
            "sys_dont_write_bytecode": True,
        }:
            raise ProtocolError("active PRE-FORMAL bootstrap evidence is not exact")
    return {
        "marker": dict(binding),
        "marker_payload": dict(marker),
        "inventory_verified": bool(require_current_inventory),
    }


def validate_formal_config(config_path: Path, root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a run to one immutable config path and the pinned interpreter."""

    path = formal_config_path(root, config_path)
    # The config's own path policy is checked before loading any worker input.
    formal = config.get("formal_config")
    if not isinstance(formal, Mapping) or formal.get("relative_path") != path.relative_to(root).as_posix():
        raise ProtocolError("formal config relative path mismatch")
    binding = config_file_binding(path)
    validate_runtime_binding(config["runtime_binding"])
    previous = config["superseded_upstream"]["preformal_stop_v11"]
    previous_path = Path(previous["path"])
    verify_config_file_binding(previous_path, previous, name="superseded v11 PRE_FORMAL marker")
    validate_preformal_stop_binding(config_path, root, config)
    return binding


def preflight(config_path: Path, *, require_deep_allocation: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config_shape(config)
    root = Path(__file__).resolve().parents[1]
    config_binding = validate_formal_config(config_path, root, config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ProtocolError("PRE-FORMAL preflight refuses visible GPU devices")
    bundle = load_input_bundle(config, require_deep_allocation=require_deep_allocation)
    gate = dict(bundle.allocation_deep_binding)
    status = "PRE_FORMAL_READY_FOR_CPU_REPAIR" if bundle.allocation_deep_ready else "PRE_FORMAL_PLACEHOLDER_GATE"
    return {
        "schema_version": 1,
        "protocol_id": config["protocol_id"],
        "status": status,
        "cpu_only": True,
        "gpu_execution_performed": False,
        "models_loaded": False,
        "source_protocol": config["source_protocol"],
        "source_bindings": {
            "graph_dir": str(bundle.graph_dir),
            "records": dict(config["source_bindings"]["records"]),
            "hyperedges": dict(config["source_bindings"]["hyperedges"]),
            "graph_report": dict(config["source_bindings"]["graph_report"]),
            "solver_contract": dict(config["source_bindings"]["solver_contract"]),
            "graph_deep_receipt": dict(config["source_bindings"]["graph_deep_receipt"]),
            "allocation_receipt": dict(config["source_bindings"]["allocation_receipt"]),
        },
        "graph_records": len(bundle.records),
        "collision_hyperedges": len(bundle.hyperedges),
        "thresholds_reused_byte_for_byte": True,
        "model_outcomes_read": False,
        "answer_values_read": False,
        "allocation_v3_infeasibility": bundle.bindings["allocation_infeasible"],
        "allocation_v3_deep_receipt": gate,
        "formal_execution_blocked": True,
        "next_gate": "allocation_v3_deep_receipt" if not bundle.allocation_deep_ready else "none_for_pre_formal_stop",
        "formal_config": config_binding,
        "runtime_binding": dict(config["runtime_binding"]),
    }


def load_preflight_bundle(config_path: Path, *, require_deep_allocation: bool = False) -> tuple[InputBundle, dict[str, Any]]:
    config = load_config(config_path)
    validate_config_shape(config)
    bundle = load_input_bundle(config, require_deep_allocation=require_deep_allocation)
    return bundle, preflight(config_path, require_deep_allocation=require_deep_allocation)
