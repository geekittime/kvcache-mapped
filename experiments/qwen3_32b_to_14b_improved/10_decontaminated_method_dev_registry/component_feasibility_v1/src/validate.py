from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from .feasibility import (
    FeasibilityError,
    build_components,
    canonical_bytes,
    component_inventory,
    file_spec,
    load_registry_core,
    make_report,
    max_flow_feasibility,
    search_maximum_balanced,
    sha256_file,
    statistics_from_inventory,
    prepare_eligible,
    verify_attempt3_failure,
    verify_file,
    verify_sidecar,
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_INVENTORY_KEYS = {
    "component_id",
    "eligible_cell_mask",
    "eligible_cells",
    "member_record_count",
    "member_record_count_by_cell",
    "representative_projection_sha256_by_cell",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _verify_implementation(root: Path, implementation: Mapping[str, Any]) -> None:
    expected_paths = {
        "config": root / "config.json",
        "source": root / "src" / "feasibility.py",
        "validation_source": root / "src" / "validate.py",
        "entrypoint": root / "scripts" / "run.py",
        "validator": root / "scripts" / "validate.py",
    }
    if set(implementation) != {*expected_paths, "tests"}:
        raise FeasibilityError("manifest implementation keys mismatch")
    for name, path in expected_paths.items():
        if implementation[name] != file_spec(path):
            raise FeasibilityError(f"implementation binding mismatch: {name}")
    test_paths = sorted((root / "tests").glob("*.py"))
    expected_tests = {path.name: file_spec(path) for path in test_paths}
    if implementation["tests"] != expected_tests:
        raise FeasibilityError("test implementation binding mismatch")


def _load_inventory(path: Path, cells_order: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = ""
    valid_cells = set(cells_order)
    for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
        row = json.loads(line)
        if set(row) != _INVENTORY_KEYS:
            raise FeasibilityError(f"component inventory keys mismatch at line {line_number}")
        component_id = row["component_id"]
        if not isinstance(component_id, str) or not _HEX64.fullmatch(component_id):
            raise FeasibilityError("invalid component id")
        if component_id <= previous:
            raise FeasibilityError("component inventory is not uniquely sorted")
        previous = component_id
        mask = row["eligible_cell_mask"]
        if not isinstance(mask, int) or isinstance(mask, bool) or not 0 < mask < (1 << len(cells_order)):
            raise FeasibilityError("invalid eligible cell mask")
        cells = row["eligible_cells"]
        expected_cells = [cell for index, cell in enumerate(cells_order) if mask & (1 << index)]
        if cells != expected_cells or not set(cells) <= valid_cells:
            raise FeasibilityError("eligible cell mask/list mismatch")
        size = row["member_record_count"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise FeasibilityError("invalid component member count")
        representatives = row["representative_projection_sha256_by_cell"]
        if set(representatives) != set(cells) or any(
            not isinstance(value, str) or not _HEX64.fullmatch(value)
            for value in representatives.values()
        ):
            raise FeasibilityError("invalid component representative projections")
        member_counts = row["member_record_count_by_cell"]
        if set(member_counts) != set(cells) or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in member_counts.values()
        ) or sum(member_counts.values()) != size:
            raise FeasibilityError("invalid per-cell component member counts")
        # The allowlist above is the privacy gate: IDs/questions/contexts/answers
        # and outcomes have no field in this artifact.
        result.append(row)
    if not result:
        raise FeasibilityError("component inventory must be non-empty")
    return result


def validate(config_path: Path, root: Path, *, deep: bool = False) -> dict[str, Any]:
    config = _read_json(config_path)
    bindings = config["bindings"]
    for spec in bindings.values():
        verify_file(Path(spec["path"]), spec)
    verify_sidecar(
        Path(bindings["attempt2_capacity_failure_sidecar"]["path"]),
        Path(bindings["attempt2_capacity_failure"]["path"]),
        bindings["attempt2_capacity_failure"],
    )
    verify_sidecar(
        Path(bindings["attempt3_sequence_failure_sidecar"]["path"]),
        Path(bindings["attempt3_sequence_failure"]["path"]),
        bindings["attempt3_sequence_failure"],
    )
    audit_root = Path(bindings["registry_config"]["path"]).parent
    verify_attempt3_failure(Path(bindings["attempt3_sequence_failure"]["path"]), audit_root)

    output_dir = root / config["outputs"]["artifact_dir"]
    marker_path = output_dir / "PUBLISHED.json"
    if not marker_path.is_file():
        raise FeasibilityError("missing atomic completion marker")
    payloads = ["component_inventory.jsonl", "feasibility_report.json", "manifest.json", "receipt.json"]
    sidecars = [f"{name}.sha256.json" for name in payloads]
    published_names = sorted([*payloads, *sidecars])
    actual_names = sorted(path.name for path in output_dir.iterdir())
    if actual_names != sorted([*published_names, "PUBLISHED.json"]):
        raise FeasibilityError("published artifact exact inventory mismatch")

    marker = _read_json(marker_path)
    if set(marker) != {
        "schema_version", "audit_id", "status", "atomic_last_completion_marker",
        "published_files", "ordered_inventory_sha256", "receipt", "receipt_sidecar",
    } or marker["schema_version"] != 1 or marker["audit_id"] != config["audit_id"] or marker["status"] != "published_complete_feasibility_only" or marker["atomic_last_completion_marker"] is not True:
        raise FeasibilityError("completion marker semantics mismatch")
    if list(marker["published_files"]) != published_names:
        raise FeasibilityError("completion marker ordered filenames mismatch")
    observed_published = {
        name: file_spec(output_dir / name, include_path=False) for name in published_names
    }
    if marker["published_files"] != observed_published:
        raise FeasibilityError("completion marker file bindings mismatch")
    ordered = [{"name": name, **observed_published[name]} for name in published_names]
    if marker["ordered_inventory_sha256"] != hashlib.sha256(canonical_bytes(ordered)).hexdigest():
        raise FeasibilityError("completion marker ordered digest mismatch")
    if marker["receipt"] != observed_published["receipt.json"] or marker["receipt_sidecar"] != observed_published["receipt.json.sha256.json"]:
        raise FeasibilityError("completion marker receipt binding mismatch")
    for payload in payloads:
        sidecar_path = output_dir / f"{payload}.sha256.json"
        expected_sidecar = file_spec(output_dir / payload)
        if _read_json(sidecar_path) != expected_sidecar:
            raise FeasibilityError(f"payload sidecar mismatch: {payload}")

    manifest = _read_json(output_dir / "manifest.json")
    report = _read_json(output_dir / "feasibility_report.json")
    receipt = _read_json(output_dir / "receipt.json")
    if manifest.get("schema_version") != 1 or manifest.get("audit_id") != config["audit_id"] or manifest.get("status") != "prepared_feasibility_only_not_registry":
        raise FeasibilityError("manifest identity/status mismatch")
    if manifest.get("bindings") != bindings:
        raise FeasibilityError("manifest input bindings mismatch")
    _verify_implementation(root, manifest["implementation"])
    if report.get("schema_version") != 1 or report.get("audit_id") != config["audit_id"] or report.get("status") != "feasibility_computed_not_registry":
        raise FeasibilityError("report identity/status mismatch")
    if report.get("claim_boundary") != "CPU-only feasibility; no sample registry selected or published":
        raise FeasibilityError("report claim boundary mismatch")
    residual = report.get("eligible_question_collision_residual_assertion", {})
    if residual.get("pass") is not True or any(
        value for key, value in residual.items() if key != "pass"
    ):
        raise FeasibilityError("eligible question-collision residual assertion mismatch")
    registry_config = _read_json(Path(bindings["registry_config"]["path"]))
    if report.get("thresholds") != registry_config["decontamination"]:
        raise FeasibilityError("report threshold binding mismatch")
    if receipt.get("schema_version") != 1 or receipt.get("audit_id") != config["audit_id"] or receipt.get("status") != "pass_feasibility_computed" or receipt.get("cpu_only") is not True or receipt.get("not_a_sample_registry") is not True:
        raise FeasibilityError("receipt identity/status mismatch")

    inventory = _load_inventory(output_dir / "component_inventory.jsonl", config["cells_order"])
    mask_counts, statistics = statistics_from_inventory(inventory, config["cells_order"])
    if report["component_statistics"] != statistics:
        raise FeasibilityError("component statistics do not recompute")
    if statistics["eligible_records"] != sum(
        int(report["decontamination_aggregate"][dataset]["eligible_records"])
        for dataset in ("hotpotqa", "musique")
    ):
        raise FeasibilityError("eligible record aggregate mismatch")
    if report.get("component_policy") != config["component_policy"] or report.get("component_buffer") != config["component_buffer"]:
        raise FeasibilityError("component policy mismatch")
    if report.get("original_quotas") != config["original_quotas"]:
        raise FeasibilityError("original quota binding mismatch")
    buffer = config["component_buffer"]
    buffer_cell = buffer["cell"]
    buffer_reserve = int(buffer["minimum_completely_unselected_supporting_components"])
    original_flow = max_flow_feasibility(
        mask_counts,
        config["cells_order"],
        config["original_quotas"],
        buffer_cell=buffer_cell,
        minimum_unselected_buffer_components=buffer_reserve,
    )
    if report["original_quota_flow"] != original_flow:
        raise FeasibilityError("original quota max-flow does not recompute")
    maximum = search_maximum_balanced(
        mask_counts,
        config["cells_order"],
        statistics["supporting_component_count_by_cell"],
        buffer_cell=buffer_cell,
        minimum_unselected_buffer_components=buffer_reserve,
    )
    if report["maximum_balanced_common_scale"] != maximum:
        raise FeasibilityError("maximum balanced quota search does not recompute")

    expected_manifest_files = {
        "component_inventory.jsonl": {
            **file_spec(output_dir / "component_inventory.jsonl"),
            "rows": len(inventory),
        },
        "feasibility_report.json": file_spec(output_dir / "feasibility_report.json"),
    }
    if manifest.get("files") != expected_manifest_files or manifest.get("original_quota_feasible") != original_flow["pass"] or manifest.get("maximum_balanced_H") != maximum["H"]:
        raise FeasibilityError("manifest result bindings mismatch")
    if receipt.get("component_count") != len(inventory) or receipt.get("original_quota_feasible") != original_flow["pass"] or receipt.get("original_quota_maximum_flow_records") != original_flow["maximum_flow_records"] or receipt.get("original_quota_requested_records") != original_flow["requested_records"] or receipt.get("maximum_balanced_common_scale") != maximum:
        raise FeasibilityError("receipt result bindings mismatch")
    if receipt.get("manifest") != file_spec(output_dir / "manifest.json"):
        raise FeasibilityError("receipt manifest binding mismatch")
    if receipt.get("claim_boundary") != "feasibility only; a later registry needs a separately frozen allocation algorithm and full validation":
        raise FeasibilityError("receipt claim boundary mismatch")
    privacy = report.get("privacy", {})
    if privacy != {
        "component_inventory_contains_raw_id_question_context_answer_or_outcome": False,
        "answer_values_available_to_component_or_flow_logic": False,
        "model_outcomes_read": False,
    }:
        raise FeasibilityError("privacy statement mismatch")
    required_validation = {
        "structural_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/validate.py",
        "deep_recompute_command": "PYTHONDONTWRITEBYTECODE=1 python scripts/validate.py --deep",
        "deep_recomputes_clean_pool_components_inventory_and_report_from_pinned_sources": True,
    }
    if receipt.get("required_post_publication_validation") != required_validation:
        raise FeasibilityError("required post-publication validation contract mismatch")

    deep_result: dict[str, Any] = {"performed": False}
    if deep:
        def progress(value: Mapping[str, Any]) -> None:
            print(json.dumps({"deep_validation": dict(value)}, sort_keys=True), file=sys.stderr, flush=True)

        core = load_registry_core(Path(bindings["registry_source"]["path"]))
        replica_eligible, replica_quarantine = prepare_eligible(
            core, registry_config, audit_root, progress
        )
        replica_uf, replica_relation_stats = build_components(
            replica_eligible,
            registry_config["decontamination"],
            core,
            progress=progress,
        )
        replica_inventory, replica_masks, replica_stats = component_inventory(
            replica_eligible, replica_uf, config["cells_order"]
        )
        inventory_hash = hashlib.sha256()
        inventory_nbytes = 0
        for row in replica_inventory:
            payload = canonical_bytes(row)
            inventory_hash.update(payload)
            inventory_nbytes += len(payload)
        observed_inventory_spec = file_spec(
            output_dir / "component_inventory.jsonl", include_path=False
        )
        expected_inventory_spec = {
            "nbytes": inventory_nbytes,
            "sha256": inventory_hash.hexdigest(),
        }
        if observed_inventory_spec != expected_inventory_spec or replica_inventory != inventory:
            raise FeasibilityError("deep component inventory replica is not byte-identical")
        replica_flow = max_flow_feasibility(
            replica_masks,
            config["cells_order"],
            config["original_quotas"],
            buffer_cell=buffer_cell,
            minimum_unselected_buffer_components=buffer_reserve,
        )
        replica_maximum = search_maximum_balanced(
            replica_masks,
            config["cells_order"],
            replica_stats["supporting_component_count_by_cell"],
            buffer_cell=buffer_cell,
            minimum_unselected_buffer_components=buffer_reserve,
        )
        replica_report = make_report(
            config,
            registry_config,
            replica_relation_stats,
            replica_stats,
            replica_flow,
            replica_maximum,
            replica_quarantine,
        )
        replica_report_payload = canonical_bytes(replica_report)
        observed_report_payload = (output_dir / "feasibility_report.json").read_bytes()
        if replica_report_payload != observed_report_payload:
            raise FeasibilityError("deep feasibility report replica is not byte-identical")
        deep_result = {
            "performed": True,
            "recomputed_from_pinned_sources": True,
            "inventory_byte_identical": True,
            "report_byte_identical": True,
            "inventory": expected_inventory_spec,
            "report": {
                "nbytes": len(replica_report_payload),
                "sha256": hashlib.sha256(replica_report_payload).hexdigest(),
            },
        }

    return {
        "status": "pass",
        "artifact": str(output_dir),
        "components": len(inventory),
        "eligible_records": statistics["eligible_records"],
        "original_quota_feasible": original_flow["pass"],
        "original_quota_maximum_flow_records": original_flow["maximum_flow_records"],
        "maximum_balanced_H": maximum["H"],
        "deep_validation": deep_result,
        "PUBLISHED": {"nbytes": marker_path.stat().st_size, "sha256": sha256_file(marker_path)},
    }
