"""Figure-4 retuned k=4 CoQA variant with a mandatory held-out gate."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from . import coqa_eval
from .coqa_eval import CoQARunVariant
from .eval_common import (
    EVALUATION_SCHEMA_VERSION,
    assert_evaluation_output,
    canonical_jsonl_bytes,
    code_identity,
    immutable_hashed_json,
    mapper_manifest_identity,
    verify_hashed_json,
)
from .k4_heldout import K4_HELDOUT, validate_k4_heldout_artifact
from .k4_protocol import K4_MAPPER_ROOT, K4_TOP_K
from .k4_runtime import K4ReverseMappedPrefixCacheBuilder, K4ReverseMapperBundle
from .protocol import EXPERIMENT_ROOT, REPOSITORY_ROOT, canonical_sha256, read_json, sha256_file


DEFAULT_CONFIG = EXPERIMENT_ROOT / "config/coqa_k4.yaml"
CURVE_VARIANT = "figure4_large_to_small_retuned_k4"
RUN_KIND = "qwen3_32b_to_14b_coqa_multi_turn_handoff_k4_retuned"
RECEIPT_NAME = "formal_k4_coqa_receipt.json"
CODE_FILES = (
    "reverse_kv/__init__.py",
    "reverse_kv/coqa_eval.py",
    "reverse_kv/coqa_k4_eval.py",
    "reverse_kv/eval_common.py",
    "reverse_kv/provenance.py",
    "reverse_kv/protocol.py",
    "reverse_kv/k4_protocol.py",
    "reverse_kv/k4_selection.py",
    "reverse_kv/k4_fit.py",
    "reverse_kv/k4_runtime.py",
    "reverse_kv/k4_heldout.py",
    "scripts/evaluate_coqa_k4.py",
    "config/coqa_k4.yaml",
    "config/mapper_k4_coqa.yaml",
    "artifacts/inputs/code_freeze.json",
)


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_k4_coqa_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise ValueError("Figure-4 k4 CoQA config is immutable")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("k4 CoQA config must be a mapping")
    base_raw = yaml.safe_load(coqa_eval.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(base_raw, Mapping):
        raise ValueError("base k20 CoQA config is malformed")
    for field in ("dataset", "selection", "prompt", "generation", "metric", "runtime"):
        if raw.get(field) != base_raw.get(field):
            raise ValueError(f"k4 CoQA {field} must exactly match the frozen k20 protocol")
    fixed = {
        "schema_version": 1,
        "direction": "qwen3_32b_to_14b",
        "curve_variant": CURVE_VARIANT,
    }
    for field, expected in fixed.items():
        if raw.get(field) != expected:
            raise ValueError(f"k4 CoQA {field} changed")
    paper = raw.get("paper_reference")
    paper_expected = {
        "paper": "arXiv:2608.03893v1",
        "section": "4.6",
        "disclosed": (
            "100 conversations of about 15 turns across five domains; word F1; "
            "Figure 4 dotted L-to-S curve is k=4, re-tuned"
        ),
        "figure_evidence": (
            "official arXiv source figures/retention_vs_turns.png; legend "
            "L→S: 32B→14B handoff (k=4, re-tuned)"
        ),
        "undisclosed": (
            "multi-turn k-sweep objective/tie break, conversation IDs, selection seed, "
            "prompt, history policy, decoding settings, exact scorer implementation"
        ),
    }
    if paper != paper_expected:
        raise ValueError("k4 CoQA paper disclosure boundary changed")
    paths = raw.get("paths")
    if not isinstance(paths, Mapping):
        raise ValueError("k4 CoQA paths are missing")
    base_paths = base_raw["paths"]
    if paths.get("source_model") != base_paths.get("source_model") or paths.get(
        "target_model"
    ) != base_paths.get("target_model"):
        raise ValueError("k4 CoQA endpoint models differ from k20")
    resolved_paths = {
        key: _resolve(config_path.parent, value) for key, value in paths.items()
    }
    if resolved_paths != {
        "source_model": Path("/home/wby/models/Qwen3-32B"),
        "target_model": Path("/home/wby/models/Qwen3-14B"),
        "mapper_manifest": (K4_MAPPER_ROOT / "manifest.json").resolve(),
        "heldout_acceptance": K4_HELDOUT.resolve(),
    }:
        raise ValueError("k4 CoQA mapper/held-out path lock changed")
    result = copy.deepcopy(dict(raw))
    result["config_path"] = config_path
    result["dataset"]["file"] = _resolve(config_path.parent, result["dataset"]["file"])
    result["paths"] = resolved_paths
    return result


def _artifact(path: Path, *, base: Path | None = None) -> dict[str, Any]:
    root = (base or EXPERIMENT_ROOT).resolve()
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(root))
    except ValueError:
        relative = str(resolved)
    return {
        "path": str(resolved),
        "relative_path": relative,
        "sha256": sha256_file(resolved),
        "nbytes": resolved.stat().st_size,
    }


def heldout_acceptance_identity(configuration: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(configuration["paths"]["heldout_acceptance"]).resolve()
    if path != K4_HELDOUT.resolve():
        raise ValueError("k4 CoQA held-out gate path changed")
    value = validate_k4_heldout_artifact(path)
    return {
        "status": "passed",
        "kind": "mandatory_pre_CoQA_k4_mapper_acceptance_gate",
        "artifact": _artifact(path),
        "checksum": _artifact(path.with_suffix(".sha256.json")),
        "mapper_manifest_sha256": value["mapper"]["manifest_sha256"],
        "fit_fingerprint": value["mapper"]["fit_fingerprint"],
        "acceptance": value["acceptance"],
    }


def k4_variant(acceptance: Mapping[str, Any]) -> CoQARunVariant:
    return CoQARunVariant(
        kind=RUN_KIND,
        selected_source_layers=K4_TOP_K,
        paper_disclosed=(
            "100 conversations of about 15 turns across five domains; word F1; "
            "Figure 4 dotted large-to-small curve uses k=4 re-tuned"
        ),
        figure_evidence=(
            "official arXiv source figures/retention_vs_turns.png; "
            "legend L→S: 32B→14B handoff (k=4, re-tuned)"
        ),
        curve_variant=CURVE_VARIANT,
        paper_undisclosed=(
            "multi-turn k-sweep objective/tie break; conversation IDs; prompt; "
            "history construction; decoding settings; exact scorer implementation"
        ),
        code_files=CODE_FILES,
        prerequisite_acceptance=dict(acceptance),
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_control_link(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    field: str,
    filename: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = output / filename
    verify_hashed_json(path)
    expected = {"sha256": sha256_file(path), "nbytes": path.stat().st_size}
    if manifest.get(field) != expected:
        raise ValueError(f"k4 CoQA manifest {field} binding changed")
    return _artifact(path, base=output), _artifact(
        path.with_name(f"{path.name}.sha256.json"), base=output
    )


def _read_verified_jsonl(
    output: Path,
    manifest: Mapping[str, Any],
    *,
    field: str,
    filename: str,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]]:
    path = output / filename
    sidecar_path = path.with_name(f"{path.name}.sha256.json")
    sidecar = read_json(sidecar_path)
    expected_sidecar = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "rows": expected_rows,
    }
    if sidecar != expected_sidecar or manifest.get(field) != expected_sidecar:
        raise ValueError(f"k4 CoQA {field} JSONL binding changed")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"k4 CoQA {field} line {line_number} is invalid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"k4 CoQA {field} line {line_number} is not an object")
            rows.append(value)
    if len(rows) != expected_rows:
        raise ValueError(f"k4 CoQA {field} row coverage changed")
    return rows, (
        _artifact(path, base=output),
        _artifact(sidecar_path, base=output),
    )


def _validate_token_plans(
    rows: Sequence[Mapping[str, Any]], selection_manifest: Mapping[str, Any]
) -> None:
    selection_rows = selection_manifest.get("rows")
    if not isinstance(selection_rows, list) or len(selection_rows) != 100:
        raise ValueError("k4 CoQA selection manifest is not 100 conversations")
    selected: dict[int, Mapping[str, Any]] = {}
    for row in selection_rows:
        if not isinstance(row, Mapping):
            raise ValueError("k4 CoQA selection manifest contains a malformed row")
        index = row.get("selection_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in selected:
            raise ValueError("k4 CoQA selection indices are malformed or duplicated")
        selected[index] = row
    if set(selected) != set(range(100)):
        raise ValueError("k4 CoQA selection indices are not exactly 0..99")

    observed: set[tuple[int, int]] = set()
    for row in rows:
        index = row.get("selection_index")
        turn = row.get("turn")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or isinstance(turn, bool)
            or not isinstance(turn, int)
            or index not in selected
            or not 1 <= turn <= 10
            or (index, turn) in observed
        ):
            raise ValueError("k4 CoQA token-plan key is malformed or duplicated")
        observed.add((index, turn))
        selection = selected[index]
        if row.get("conversation_id") != selection.get("conversation_id") or row.get(
            "domain"
        ) != selection.get("domain"):
            raise ValueError("k4 CoQA token plan differs from its selected conversation")
        if row.get("is_headline_handoff_depth") is not (
            turn in coqa_eval.HEADLINE_DEPTHS
        ):
            raise ValueError("k4 CoQA token-plan handoff-depth flag changed")
        prompt_count = row.get("prompt_token_count")
        parent_count = row.get("parent_token_count")
        append_count = row.get("append_after_turn_token_count")
        if (
            isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or prompt_count < 2
            or parent_count != prompt_count - 1
            or isinstance(append_count, bool)
            or not isinstance(append_count, int)
            or append_count < 0
        ):
            raise ValueError("k4 CoQA token-plan lengths changed")
        for field in (
            "prompt_sha256",
            "prompt_token_ids_sha256",
            "parent_token_ids_sha256",
        ):
            if not _is_sha256(row.get(field)):
                raise ValueError(f"k4 CoQA token plan {field} is malformed")
        append_sha = row.get("append_after_turn_token_ids_sha256")
        if turn < 10:
            if append_count <= 0 or not _is_sha256(append_sha):
                raise ValueError("k4 CoQA non-final token plan lacks its cache delta")
        elif append_count != 0 or append_sha is not None:
            raise ValueError("k4 CoQA final token plan unexpectedly appends history")
        boundary = row.get("boundary_token_id")
        if isinstance(boundary, bool) or not isinstance(boundary, int) or boundary < 0:
            raise ValueError("k4 CoQA token-plan boundary token is malformed")
        if row.get("canonical_full_prompt_token_equality") is not True:
            raise ValueError("k4 CoQA token plan lacks canonical prompt equality")
    if observed != {(index, turn) for index in range(100) for turn in range(1, 11)}:
        raise ValueError("k4 CoQA token-plan coverage is incomplete")


def validate_k4_coqa_output(output_directory: str | Path) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    try:
        output.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("k4 CoQA output escaped the reverse experiment") from exc
    manifest_path = output / "manifest.json"
    summary_path = output / "summary.json"
    run_definition_path = output / "run_definition.json"
    for path in (manifest_path, summary_path, run_definition_path):
        verify_hashed_json(path)
    manifest = read_json(manifest_path)
    definition = read_json(run_definition_path)
    expected_manifest = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "direction": "qwen3_32b_to_14b",
        "task": "coqa_multi_turn_handoff",
        "curve_variant": CURVE_VARIANT,
        "selected_source_layers": K4_TOP_K,
        "records": 1000,
        "expected_records": 1000,
        "headline_handoff_depths": list(coqa_eval.HEADLINE_DEPTHS),
        "paper_bit_exact_claimed": False,
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"k4 CoQA manifest {field} changed")
    if definition.get("kind") != RUN_KIND:
        raise ValueError("k4 CoQA run kind changed")
    if definition.get("curve_variant") != CURVE_VARIANT:
        raise ValueError("k4 CoQA run definition lacks curve identity")
    if definition.get("selected_source_layers") != K4_TOP_K:
        raise ValueError("k4 CoQA run definition selected-layer count changed")
    definition_expected = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "direction": "qwen3_32b_to_14b",
        "modes": list(coqa_eval.MODES),
    }
    for field, expected in definition_expected.items():
        if definition.get(field) != expected:
            raise ValueError(f"k4 CoQA run definition {field} changed")
    configuration = load_k4_coqa_config()
    acceptance = heldout_acceptance_identity(configuration)
    variant = k4_variant(acceptance)
    # Reuse the production validator's full scientific chain with the sealed
    # k=4 variant injected: reselect from the pinned parquet, regenerate all
    # 1,000 token plans, validate runtime placement/materialization, and run
    # `_validate_complete_record` over all 1,000 checkpoints so prediction,
    # generated token IDs, stop/cap evidence, cache evidence, references,
    # raw JSONL, summary, and manifest cannot be coordinatedly rewritten.
    strong_manifest = coqa_eval.validate_completed_output(
        output,
        configuration=configuration,
        variant=variant,
        mapper_builder_class=K4ReverseMappedPrefixCacheBuilder,
        mapper_bundle_class=K4ReverseMapperBundle,
    )
    if strong_manifest is None or strong_manifest != manifest:
        raise ValueError("k4 CoQA failed the shared strong completion validator")
    if definition.get("prerequisite_acceptance") != acceptance:
        raise ValueError("k4 CoQA no longer binds the current held-out acceptance gate")
    if manifest.get("prerequisite_acceptance") != acceptance:
        raise ValueError("k4 CoQA manifest no longer binds the held-out gate")
    mapper = mapper_manifest_identity(K4_MAPPER_ROOT / "manifest.json")
    if definition.get("artifacts", {}).get("mapper") != mapper:
        raise ValueError("k4 CoQA mapper identity changed")
    fingerprint = str(definition.get("request_fingerprint", ""))
    unsigned_definition = copy.deepcopy(definition)
    unsigned_definition.pop("request_fingerprint", None)
    if not _is_sha256(fingerprint) or canonical_sha256(unsigned_definition) != fingerprint:
        raise ValueError("k4 CoQA run-definition fingerprint is invalid")
    if manifest.get("request_fingerprint") != fingerprint:
        raise ValueError("k4 CoQA manifest fingerprint changed")
    expected_code = {
        "reverse_experiment": code_identity(CODE_FILES),
        "shared_read_only_runtime": code_identity(
            coqa_eval.SHARED_CODE_FILES, root=REPOSITORY_ROOT
        ),
    }
    if definition.get("code") != expected_code:
        raise ValueError("k4 CoQA code identity differs from the current frozen code")
    for field in ("prompt", "generation", "metric", "runtime"):
        if definition.get(field) != configuration[field]:
            raise ValueError(f"k4 CoQA run-definition {field} changed")
    run_fields = coqa_eval._variant_run_fields(variant)
    if definition.get("paper_reference") != run_fields["paper_reference"]:
        raise ValueError("k4 CoQA run-definition paper boundary changed")

    controls: list[dict[str, Any]] = []
    for field, filename in (
        ("run_definition", "run_definition.json"),
        ("dataset_receipt", "dataset_receipt.json"),
        ("selection_manifest", "selection_manifest.json"),
        ("runtime_setup", "runtime_setup.json"),
    ):
        controls.extend(
            _verify_control_link(
                output, manifest, field=field, filename=filename
            )
        )
    dataset_receipt = read_json(output / "dataset_receipt.json")
    selection_manifest = read_json(output / "selection_manifest.json")
    runtime_setup = read_json(output / "runtime_setup.json")
    if definition.get("dataset_receipt") != dataset_receipt:
        raise ValueError("k4 CoQA run definition differs from its dataset receipt")
    if definition.get("selection_manifest_sha256") != sha256_file(
        output / "selection_manifest.json"
    ) or definition.get("selection_rows_sha256") != selection_manifest.get(
        "rows_sha256"
    ):
        raise ValueError("k4 CoQA run definition selection binding changed")
    selection_rows_raw = selection_manifest.get("rows")
    if not isinstance(selection_rows_raw, list) or canonical_sha256(
        selection_rows_raw
    ) != selection_manifest.get("rows_sha256"):
        raise ValueError("k4 CoQA selection row digest changed")
    if dataset_receipt.get("dataset") != coqa_eval.verify_dataset_file(configuration):
        raise ValueError("k4 CoQA dataset receipt differs from the pinned local payload")
    if runtime_setup.get("request_fingerprint") != fingerprint:
        raise ValueError("k4 CoQA runtime setup fingerprint changed")
    runtime_mapper = runtime_setup.get("mapper")
    runtime_mapper_expected = {
        "curve_variant": CURVE_VARIANT,
        "manifest": str((K4_MAPPER_ROOT / "manifest.json").resolve()),
        "manifest_sha256": mapper["manifest_sha256"],
        "fit_fingerprint": mapper["fit_fingerprint"],
        "layers": 40,
        "selected_source_layers": K4_TOP_K,
        "source_features": 4096,
        "target_features": 1024,
        "independently_refit_not_k20_slice": True,
    }
    if not isinstance(runtime_mapper, Mapping) or any(
        runtime_mapper.get(field) != expected
        for field, expected in runtime_mapper_expected.items()
    ):
        raise ValueError("k4 CoQA runtime setup used the wrong mapper family")

    token_plans, token_plan_artifacts = _read_verified_jsonl(
        output,
        manifest,
        field="token_plans",
        filename="token_plans.jsonl",
        expected_rows=1000,
    )
    _validate_token_plans(token_plans, selection_manifest)
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1000:
        raise ValueError("k4 CoQA checkpoint inventory is incomplete")
    if manifest.get("checkpoint_manifest_sha256") != canonical_sha256(checkpoints):
        raise ValueError("k4 CoQA checkpoint inventory digest changed")
    records = []
    mode_counts = {"target_standalone": 0, "mapped": 0}
    checkpoint_paths: set[str] = set()
    selection_rows = {
        int(row["selection_index"]): row for row in selection_manifest["rows"]
    }
    for row in checkpoints:
        if not isinstance(row, Mapping):
            raise ValueError("k4 CoQA checkpoint row is malformed")
        relative = str(row.get("path", ""))
        path = (output / relative).resolve()
        if output not in path.parents or relative in checkpoint_paths:
            raise ValueError("k4 CoQA checkpoint escaped its run root")
        checkpoint_paths.add(relative)
        if path.stat().st_size != row.get("nbytes") or sha256_file(path) != row.get("sha256"):
            raise ValueError(f"k4 CoQA checkpoint changed: {path}")
        verify_hashed_json(path)
        record = read_json(path)
        if (
            record.get("status") != "complete"
            or record.get("request_fingerprint") != fingerprint
            or record.get("curve_variant") != CURVE_VARIANT
            or record.get("selected_source_layers") != K4_TOP_K
        ):
            raise ValueError(f"k4 CoQA checkpoint scientific identity changed: {path}")
        mode = str(record.get("mode", ""))
        if mode not in mode_counts:
            raise ValueError("k4 CoQA checkpoint mode changed")
        index = record.get("selection_index")
        depth = record.get("handoff_depth")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in selection_rows
            or isinstance(depth, bool)
            or not isinstance(depth, int)
            or depth not in coqa_eval.HEADLINE_DEPTHS
        ):
            raise ValueError("k4 CoQA checkpoint selection/depth changed")
        selected = selection_rows[index]
        expected_relative = (
            f"records/{mode}/{index:03d}_"
            f"{hashlib.sha256(str(selected['conversation_id']).encode('utf-8')).hexdigest()[:16]}_"
            f"depth_{depth:02d}.json"
        )
        if relative != expected_relative:
            raise ValueError("k4 CoQA checkpoint path differs from its scientific key")
        for field in ("conversation_id", "domain", "rank_within_domain", "rank_sha256"):
            if record.get(field) != selected.get(field):
                raise ValueError(f"k4 CoQA checkpoint differs from selection field {field}")
        if record.get("handoff_executed") is not (mode == "mapped"):
            raise ValueError("k4 CoQA checkpoint handoff evidence changed")
        mode_counts[mode] += 1
        records.append(record)
    if mode_counts != {"target_standalone": 500, "mapped": 500}:
        raise ValueError("k4 CoQA mode coverage changed")
    recomputed = coqa_eval.summarize_records(records)
    if read_json(summary_path) != recomputed:
        raise ValueError("k4 CoQA summary differs from raw checkpoints")
    raw_records, raw_artifacts = _read_verified_jsonl(
        output,
        manifest,
        field="raw_records",
        filename="raw_records.jsonl",
        expected_rows=1000,
    )
    if raw_records != records or (output / "raw_records.jsonl").read_bytes() != canonical_jsonl_bytes(records):
        raise ValueError("k4 CoQA raw-record JSONL differs from exact checkpoints")
    summary_sidecar = verify_hashed_json(summary_path)
    if manifest.get("summary") != summary_sidecar:
        raise ValueError("k4 CoQA manifest summary binding changed")
    return {
        "manifest": _artifact(manifest_path, base=output),
        "summary": _artifact(summary_path, base=output),
        "run_definition": _artifact(run_definition_path, base=output),
        "controls": controls,
        "token_plans": list(token_plan_artifacts),
        "raw_records": list(raw_artifacts),
        "request_fingerprint": fingerprint,
        "mode_counts": mode_counts,
        "checkpoint_inventory_sha256": canonical_sha256(checkpoints),
        "mapper": mapper,
        "prerequisite_acceptance": acceptance,
    }


def validate_k4_receipt(output_directory: str | Path) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    validation = validate_k4_coqa_output(output)
    path = output / RECEIPT_NAME
    verify_hashed_json(path)
    value = read_json(path)
    fixed = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "kind": "figure4_large_to_small_retuned_k4_coqa_curve",
        "direction": "qwen3_32b_to_14b",
        "curve_variant": CURVE_VARIANT,
        "selected_source_layers": K4_TOP_K,
        "paper_bit_exact_claimed": False,
        "paper_disclosure_boundary": {
            "disclosed": "Figure 4 labels the retuned L-to-S curve k=4",
            "undisclosed": (
                "multi-turn sweep criterion and all exact local CoQA protocol details"
            ),
        },
        "validation": validation,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"formal k4 CoQA receipt {field} changed")
    finished = value.get("finished_utc")
    if not isinstance(finished, str) or not finished:
        raise ValueError("formal k4 CoQA receipt lacks completion time")
    return value


def publish_k4_receipt(output_directory: str | Path) -> dict[str, Any]:
    output = Path(output_directory).resolve()
    receipt_path = output / RECEIPT_NAME
    if receipt_path.exists():
        return validate_k4_receipt(output)
    validation = validate_k4_coqa_output(output)
    receipt = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "kind": "figure4_large_to_small_retuned_k4_coqa_curve",
        "direction": "qwen3_32b_to_14b",
        "curve_variant": CURVE_VARIANT,
        "selected_source_layers": K4_TOP_K,
        "paper_bit_exact_claimed": False,
        "paper_disclosure_boundary": {
            "disclosed": "Figure 4 labels the retuned L-to-S curve k=4",
            "undisclosed": (
                "multi-turn sweep criterion and all exact local CoQA protocol details"
            ),
        },
        "validation": validation,
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    immutable_hashed_json(receipt_path, receipt)
    return validate_k4_receipt(output)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configuration = load_k4_coqa_config(args.config)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = assert_evaluation_output(
        args.output_dir
        or EXPERIMENT_ROOT / "runs" / f"{stamp}_coqa_reverse_k4_retuned"
    )
    if args.prepare_only and args.validate_only:
        raise ValueError("choose at most one k4 CoQA inspection mode")
    if args.prepare_only:
        selected, receipt = coqa_eval.prepare_coqa(configuration, output)
        print(
            json.dumps(
                {
                    "selected": len(selected) if selected is not None else None,
                    "receipt": receipt,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return 0
    if args.validate_only:
        validate_k4_receipt(output)
        return 0
    acceptance = heldout_acceptance_identity(configuration)
    result = coqa_eval.run_evaluation(
        configuration,
        output,
        variant=k4_variant(acceptance),
        mapper_builder_class=K4ReverseMappedPrefixCacheBuilder,
    )
    if result:
        return int(result)
    publish_k4_receipt(output)
    return 0


__all__ = [
    "CODE_FILES",
    "CURVE_VARIANT",
    "DEFAULT_CONFIG",
    "heldout_acceptance_identity",
    "k4_variant",
    "load_k4_coqa_config",
    "main",
    "publish_k4_receipt",
    "validate_k4_receipt",
    "validate_k4_coqa_output",
]
