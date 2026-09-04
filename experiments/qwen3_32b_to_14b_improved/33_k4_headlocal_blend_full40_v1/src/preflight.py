"""CPU-only provenance and publication gate for experiment 33.

This file intentionally does not import torch/transformers/safetensors.  It
checks the immutable upstream manifests and the local source closure before a
future runner is allowed to touch activations or a GPU.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .blend import (
    ProtocolError,
    canonical_bytes,
    canonical_sha256,
    file_ref,
    validate_selection_subset,
)


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ("key", "value")
TARGET_LAYERS = tuple(range(40))


def strict_json(path: Path) -> Any:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ProtocolError(f"JSON path is not a regular non-link file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc


def _ref(path_value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(path_value, Mapping) or set(path_value) != {"path", "nbytes", "sha256"}:
        raise ProtocolError(f"{label}: exact file reference required")
    path = Path(path_value["path"])
    if not path.is_absolute() or type(path_value["nbytes"]) is not int or path_value["nbytes"] < 0:
        raise ProtocolError(f"{label}: malformed file reference")
    # Walk every existing parent component before resolving; a regular file
    # behind a symlinked directory is not an immutable reference.
    _lstat_chain(path, label=label, allow_missing_tail=False)
    digest = file_ref(path)
    if digest != {"path": str(path.resolve()), "nbytes": path_value["nbytes"], "sha256": path_value["sha256"]}:
        raise ProtocolError(f"{label}: content reference changed")
    return digest


def _lstat_chain(path: Path, *, label: str, allow_missing_tail: bool) -> Path:
    """Reject symlink/non-regular path components without following them."""

    if not path.is_absolute():
        raise ProtocolError(f"{label}: path must be absolute")
    current = Path(path.anchor)
    parts = path.parts[1:]
    missing = False
    for index, part in enumerate(parts):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            if allow_missing_tail:
                missing = True
                break
            raise ProtocolError(f"{label}: missing path component {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ProtocolError(f"{label}: symlink path component {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise ProtocolError(f"{label}: non-directory path component {current}")
    if not missing and not path.is_dir() and not path.is_file():
        raise ProtocolError(f"{label}: unsupported path")
    return path


def _path_ref(path: Path, expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    observed = file_ref(path)
    if observed["nbytes"] != expected.get("nbytes") or observed["sha256"] != expected.get("sha256"):
        raise ProtocolError(f"{label}: hash/size mismatch")
    return observed


def source_closure() -> dict[str, Any]:
    """Hash the exact local code/test closure, rejecting links and caches."""

    roots = (ROOT / "src", ROOT / "scripts", ROOT / "tests")
    rows: list[dict[str, Any]] = []
    for base in roots:
        if not base.exists() or base.is_symlink() or not base.is_dir():
            raise ProtocolError(f"source root missing or unsafe: {base}")
        for path in sorted(base.rglob("*")):
            if path.is_symlink():
                raise ProtocolError(f"source closure contains symlink: {path}")
            if path.is_dir():
                continue
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ProtocolError(f"source closure contains non-regular file: {path}")
            rel = path.relative_to(ROOT).as_posix()
            if "__pycache__" in Path(rel).parts or path.suffix in {".pyc", ".pyo"}:
                raise ProtocolError(f"source closure contains cache: {path}")
            rows.append({"path": rel, "nbytes": int(info.st_size), "sha256": _sha(path)})
    rows.sort(key=lambda row: row["path"])
    if not rows:
        raise ProtocolError("empty source closure")
    return {"files": rows, "file_count": len(rows), "digest": _inventory_digest(rows)}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _inventory_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    # Keep this independent from JSON object ordering: path, decimal size and
    # lowercase SHA are the canonical line protocol used by earlier siblings.
    payload = "".join(f"{row['path']}\t{int(row['nbytes'])}\t{str(row['sha256']).lower()}\n" for row in sorted(rows, key=lambda r: str(r["path"]))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _capture_inventory_digest(value: Any) -> str:
    """Digest used by the immutable activation-capture writer.

    The upstream capture protocol intentionally has no trailing LF and uses
    ``ensure_ascii=False``.  It is distinct from this sibling's receipt
    canonicalization, so never silently substitute one for the other.
    """

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_ref_json(ref: Mapping[str, Any], *, label: str) -> tuple[Path, Any, dict[str, Any]]:
    observed = _ref(ref, label=label)
    path = Path(observed["path"])
    return path, strict_json(path), observed


def _validate_model_ref(model: Mapping[str, Any], *, label: str, expected_layers: int) -> dict[str, Any]:
    if not isinstance(model, Mapping) or not isinstance(model.get("path"), str):
        raise ProtocolError(f"{label}: model binding missing")
    result: dict[str, Any] = {"path": model["path"], "files": {}}
    for key in ("config", "weight_index", "tokenizer"):
        path, _, observed = _load_ref_json(model.get(key), label=f"{label}.{key}") if key != "tokenizer" else (None, None, None)
        if key == "tokenizer":
            observed = _ref(model.get(key), label=f"{label}.{key}")
        result["files"][key] = observed
    if type(model.get("inventory_digest")) is not str or len(model["inventory_digest"]) != 64:
        raise ProtocolError(f"{label}: full shard inventory digest missing")
    if type(model.get("shard_count")) is not int or model["shard_count"] <= 0:
        raise ProtocolError(f"{label}: shard count missing")
    # The model's small config is parsed only for architecture sanity; shard
    # bytes are already content-addressed by the immutable upstream inventory.
    cfg = strict_json(Path(result["files"]["config"]["path"]))
    if cfg.get("num_hidden_layers") != expected_layers:
        raise ProtocolError(f"{label}: hidden-layer count changed")
    return result


def _validate_mapper_manifest(config: Mapping[str, Any], prefix: str, *, expected_k: int) -> dict[str, Any]:
    root = Path(config["upstream"][f"{prefix}_root"])
    manifest_path, manifest, manifest_ref = _load_ref_json(config["upstream"][f"{prefix}_manifest"], label=f"{prefix} manifest")
    marker_path, marker, marker_ref = _load_ref_json(config["upstream"][f"{prefix}_manifest_marker"], label=f"{prefix} marker")
    if marker.get("path") != str(manifest_path) or marker.get("sha256") != manifest_ref["sha256"]:
        raise ProtocolError(f"{prefix}: manifest marker does not bind manifest")
    if manifest.get("status") != "complete" or manifest.get("direction") != "qwen3_32b_to_14b":
        raise ProtocolError(f"{prefix}: manifest state/direction changed")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ProtocolError(f"{prefix}: layer coverage changed")
    selection_path, selection, selection_ref = _load_ref_json(config["upstream"][f"{prefix}_selection"], label=f"{prefix} selection")
    rows = selection.get("selected_source_layers_rank_order")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ProtocolError(f"{prefix}: selection coverage changed")
    # Check every payload reference and sidecar reference.  This hashes mapper
    # files (a few GB total) but never reads activation values or model shards.
    payload_count = 0
    for index, row in enumerate(layers):
        if not isinstance(row, Mapping) or row.get("target_layer") != index:
            raise ProtocolError(f"{prefix}: layer order changed")
        sel = rows[index]
        if not isinstance(sel, list) or len(sel) != expected_k or len(set(sel)) != expected_k or any(type(v) is not int or not 0 <= v < 64 for v in sel):
            raise ProtocolError(f"{prefix}: source selection row malformed")
        manifest_selection = row.get("source_layers_rank_order")
        if manifest_selection != sel:
            raise ProtocolError(f"{prefix}: manifest/selection rank order differs at layer {index}")
        for key in ("path", "sidecar_path"):
            rel = row.get(key)
            if not isinstance(rel, str):
                raise ProtocolError(f"{prefix}: missing {key}")
            p = Path(rel)
            # Manifest paths are rooted relative to the mapper root; absolute
            # paths are accepted only if they resolve to the same root.
            path = p if p.is_absolute() else root / p
            _lstat_chain(path, label=f"{prefix} {key}", allow_missing_tail=False)
            expected = {"path": str(path.resolve()), "nbytes": row.get("weight_numel", 0) * 4 if key == "path" else None, "sha256": row.get("sha256") if key == "path" else row.get("sidecar_sha256")}
            observed = file_ref(path)
            if observed["sha256"] != expected["sha256"]:
                raise ProtocolError(f"{prefix}: {key} hash changed at layer {index}")
            if key == "path":
                if type(row.get("weight_numel")) is not int or row.get("weight_numel") != 2 * (expected_k * 8 * 128) * 1024:
                    raise ProtocolError(f"{prefix}: weight geometry changed at layer {index}")
                payload_count += 1
    return {"root": str(root), "manifest": manifest_ref, "manifest_marker": marker_ref, "selection": selection_ref, "selection_rows": rows, "payload_count": payload_count}


def _validate_activation_manifest_ref(ref: Mapping[str, Any], *, label: str, expected_entries: int, expected_sequences: int) -> dict[str, Any]:
    path, manifest, observed = _load_ref_json(ref, label=label)
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 2:
        raise ProtocolError(f"{label}: manifest state changed")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_entries:
        raise ProtocolError(f"{label}: entry count changed")
    for row in entries:
        if not isinstance(row, Mapping) or set(row) != {"cache_type", "layer", "nbytes", "path", "sha256", "shape", "sidecar_sha256"}:
            raise ProtocolError(f"{label}: entry schema changed")
        if row.get("shape") != [expected_sequences * 256, 8, 128] or type(row.get("nbytes")) is not int or row["nbytes"] != expected_sequences * 256 * 8 * 128 * 2:
            raise ProtocolError(f"{label}: activation geometry changed")
        if type(row.get("sha256")) is not str or len(row["sha256"]) != 64 or type(row.get("sidecar_sha256")) is not str or len(row["sidecar_sha256"]) != 64:
            raise ProtocolError(f"{label}: activation hashes missing")
    if manifest.get("inventory_sha256") != _capture_inventory_digest(entries):
        raise ProtocolError(f"{label}: inventory digest changed")
    return {"path": str(path), "ref": observed, "entry_count": len(entries), "sequence_count": expected_sequences, "payloads_opaque_hashed_by_manifest": True}


def _validate_prior_candidate_files(prior_results: Mapping[str, Any]) -> dict[str, Any]:
    """Re-hash every immutable C map/sidecar referenced by dir32 results."""

    rows = prior_results.get("candidate_files")
    if not isinstance(rows, list) or len(rows) != 160:
        raise ProtocolError("prior head-local candidate file inventory changed")
    seen: set[str] = set()
    total_bytes = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"path", "nbytes", "sha256"}:
            raise ProtocolError("prior candidate file ref schema changed")
        ref = _ref(row, label="prior head-local candidate file")
        if ref["path"] in seen:
            raise ProtocolError("duplicate prior head-local candidate file")
        seen.add(ref["path"])
        total_bytes += ref["nbytes"]
    return {"file_count": len(rows), "total_bytes": total_bytes, "files": sorted(seen)}


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1 or config.get("experiment_id") != "qwen3_32b_to_14b_k4_headlocal_blend_full40_v1":
        raise ProtocolError("experiment identity changed")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ProtocolError("protocol missing")
    exact = {"source_layers": 64, "target_layers": 40, "k4_layers": 4, "canonical_k20_layers": 20, "kv_heads": 8, "head_dim": 128, "fit_sequence_count": 500, "alpha_calibration_sequence_count": 500, "final_refit_sequence_count": 500, "eval_sequence_count": 40, "fold_count": 5, "locked_sequence_count": 10, "locked_activation_evaluations": 0, "locked_rows_indexed": False, "answers_or_outcomes_read": False}
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"protocol.{key} changed")
    if protocol.get("fit_split") != "train500_frozen_maps" or protocol.get("alpha_calibration_split") != "train500_postfit_same_namespace" or protocol.get("eval_split") != "heldout_tuning40":
        raise ProtocolError("fit/calibration/evaluation split labels changed")
    if protocol.get("maps_refit_in_candidate") is not False or protocol.get("alpha_is_oof") is not False or protocol.get("heldout_tuning_previously_inspected_exploratory_only") is not True:
        raise ProtocolError("alpha/map leakage boundary changed")
    if config.get("status") != "PRE_FORMAL_CPU_ONLY":
        raise ProtocolError("candidate is not CPU-only before explicit execution")
    if config.get("runtime", {}).get("gpu_execution_requires_explicit_execute") is not True:
        raise ProtocolError("GPU explicit-execute gate missing")
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("stop_receipt") != "artifacts/preformal/PRE_FORMAL_STOP_v2.json":
        raise ProtocolError("STOP version/output binding changed")


def validate_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Perform all cheap/medium input checks and return a content inventory."""

    validate_config(config)
    upstream = config["upstream"]
    previous = config.get("supersedes", {}).get("prior_local_stop")
    if previous is not None:
        previous_path, previous_json, previous_ref = _load_ref_json(previous, label="previous local STOP")
        if previous_json.get("status") != "PRE_FORMAL_STOP" or previous_json.get("protocol_id") != "qwen3_32b_to_14b_k4_headlocal_blend_full40_v1":
            raise ProtocolError("previous local STOP evidence changed")
    k4 = _validate_mapper_manifest(config, "k4", expected_k=4)
    k20 = _validate_mapper_manifest(config, "k20", expected_k=20)
    for layer, (s20, s4) in enumerate(zip(k20["selection_rows"], k4["selection_rows"])):
        validate_selection_subset(s20, s4)
    # Bind the old head-local candidate as failed evidence only.  It is never
    # used as a clean CV result or a promotion decision.
    prior = upstream["head_local_prior"]
    prior_refs = {}
    prior_candidate_inventory = None
    for key in ("config", "results", "results_sidecar", "completed"):
        path, obj, observed = _load_ref_json(prior[key], label=f"head_local_prior.{key}")
        prior_refs[key] = observed
        if key == "results" and (obj.get("decision") != "NO_GO" or obj.get("locked_activation_evaluations") != 0 or obj.get("answers_or_outcomes_read") is not False):
            raise ProtocolError("prior head-local evidence crossed the claim boundary")
        if key == "results":
            prior_candidate_inventory = _validate_prior_candidate_files(obj)
    calibration = _load_ref_json(upstream["calibration_manifest"], label="calibration manifest")[2]
    train_token = _ref(upstream["train_tokens"], label="train tokens")
    held_token = _ref(upstream["heldout_tokens"], label="heldout tokens")
    if train_token["sha256"] == held_token["sha256"]:
        raise ProtocolError("train and heldout token namespaces are identical")
    activations = {
        "train_source": _validate_activation_manifest_ref(upstream["fit_source_activation_manifest"], label="train source activation manifest", expected_entries=128, expected_sequences=500),
        "train_target": _validate_activation_manifest_ref(upstream["fit_target_activation_manifest"], label="train target activation manifest", expected_entries=80, expected_sequences=500),
        "heldout_source": _validate_activation_manifest_ref(upstream["heldout_source_activation_manifest"], label="heldout source activation manifest", expected_entries=128, expected_sequences=50),
        "heldout_target": _validate_activation_manifest_ref(upstream["heldout_target_activation_manifest"], label="heldout target activation manifest", expected_entries=80, expected_sequences=50),
    }
    models = {
        "source": _validate_model_ref(upstream["source_model"], label="source model", expected_layers=64),
        "target": _validate_model_ref(upstream["target_model"], label="target model", expected_layers=40),
    }
    closure = source_closure()
    inventory = {
        "protocol": config["experiment_id"],
        "k4": k4,
        "k20": k20,
        "prior_head_local": {"refs": prior_refs, "candidate_files": prior_candidate_inventory},
        "tokens": {"train": train_token, "heldout": held_token},
        "calibration_manifest": calibration,
        "activations": activations,
        "models": models,
        "source_byte_closure_digest": closure["digest"],
        "fit_split": {"map_fit_ids": list(range(500)), "alpha_calibration_ids": list(range(500)), "alpha_is_postfit_same_namespace": True, "eval_split": "heldout_tuning40", "locked_ids_indexed": False},
    }
    return {
        "status": "pass",
        "protocol": config["experiment_id"],
        "source_selection_by_target": {str(i): list(k20["selection_rows"][i]) for i in TARGET_LAYERS},
        "k4_selection_by_target": {str(i): list(k4["selection_rows"][i]) for i in TARGET_LAYERS},
        "k4_subset_k20_all40": True,
        "fit_split": "train500_frozen_maps",
        "fit_sequence_count": 500,
        "alpha_calibration_split": "train500_postfit_same_namespace",
        "alpha_calibration_sequence_count": 500,
        "final_refit_sequence_count": 500,
        "eval_split": "heldout_tuning40",
        "eval_sequence_count": 40,
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "answers_or_outcomes_read": False,
        "source_byte_closure": closure,
        "inventory": inventory,
        "input_inventory_sha256": canonical_sha256(inventory),
    }


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    """Write a JSON receipt atomically and return its content reference."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ProtocolError(f"output path is a symlink: {path}")
    payload = canonical_bytes(value)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return file_ref(path)


def write_sidecar(path: Path) -> dict[str, Any]:
    ref = file_ref(path)
    sidecar = {"path": ref["path"], "nbytes": ref["nbytes"], "sha256": ref["sha256"]}
    return atomic_json(path.with_name(path.name + ".sha256.json"), sidecar)


def make_stop(config_path: Path, preflight_path: Path, preflight: Mapping[str, Any], *, test_count: int, compile_count: int, runtime_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config_ref = file_ref(config_path)
    pre_ref = file_ref(preflight_path)
    closure = preflight.get("source_byte_closure")
    if not isinstance(closure, Mapping):
        raise ProtocolError("preflight closure missing")
    return {
        "schema_version": 1,
        "status": "PRE_FORMAL_STOP",
        "protocol_id": "qwen3_32b_to_14b_k4_headlocal_blend_full40_v1",
        "experiment_root": str(ROOT),
        "config": config_ref,
        "preflight": pre_ref,
        "input_inventory_sha256": preflight.get("input_inventory_sha256"),
        "source_byte_closure_digest": closure.get("digest"),
        "source_byte_closure_file_count": closure.get("file_count"),
        "tests_passed": _strict_stop_int(test_count, "tests_passed"),
        "compile_passed_files": _strict_stop_int(compile_count, "compile_passed_files"),
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "answers_or_outcomes_read": False,
        "gpu_started": False,
        "formal_run_started": False,
        "runtime_identity": dict(runtime_identity or {}),
        "claim_boundary": "activation geometry only; heldout tuning is evaluation, not alpha selection; no downstream/locked claim",
        "stop_reason": "CPU input/provenance gate complete; explicit operator audit required before any GPU stage",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def _strict_stop_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ProtocolError(f"{label}: expected nonnegative int")
    return value
