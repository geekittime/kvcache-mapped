"""CPU-only preflight and the external allocation-v3 deep-receipt gate."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .common import ProtocolError, require_exact_keys
from .inputs import InputBundle, load_config, load_input_bundle


def validate_config_shape(config: Mapping[str, Any]) -> None:
    """Validate immutable protocol fields before touching external inputs."""

    if config.get("schema_version") != 1 or config.get("protocol_id") != "quota_capacity_repair_v1":
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


def preflight(config_path: Path, *, require_deep_allocation: bool = False) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config_shape(config)
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
    }


def load_preflight_bundle(config_path: Path, *, require_deep_allocation: bool = False) -> tuple[InputBundle, dict[str, Any]]:
    config = load_config(config_path)
    validate_config_shape(config)
    bundle = load_input_bundle(config, require_deep_allocation=require_deep_allocation)
    return bundle, preflight(config_path, require_deep_allocation=require_deep_allocation)
