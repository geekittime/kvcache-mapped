"""CacheBridge Head-Local activation probe.

The probe is deliberately split into a CPU-only input gate and an explicit
GPU execution path.  The scientific candidate is a block-diagonal (one KV
head at a time) affine ridge mapper:

    y[h] = x[h] @ W[h] + b[h]

where ``x[h]`` concatenates the 128 channels of the same KV head over the 20
frozen source layers (20 * 128 = 2560).  It is an ordinary, unweighted SSE
objective; no attention/readout weights are used.  The mapper is fitted on
the immutable 500-sequence ``train`` split and evaluated only on the forty
``heldout_tuning`` rows from the frozen ``02_shrinkage_mapper`` split.  The
ten locked rows are represented by a count/hash boundary and are never
indexed.

All heavyweight imports (torch, safetensors) are lazy.  This lets the
preflight and the tests prove the protocol without touching CUDA or loading a
model.  Future GPU runs must pass ``--execute`` and the eight-card idle lock.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = 1
PROTOCOL = "cachebridge_head_local_unweighted_k20_full40_v1"
PROBE_LAYERS = (0, 7, 15, 23, 31, 39)
FULL_LAYERS = tuple(range(40))
ACTIVE_LAYERS = FULL_LAYERS
COMPONENTS = ("key", "value")
COMPONENT_DIR = {"key": "k_content", "value": "v"}
EXPECTED_PAYLOAD_BYTES = 50 * 256 * 8 * 128 * 2
TRAIN_PAYLOAD_BYTES = 500 * 256 * 8 * 128 * 2
EXPECTED_METADATA_SHAPE = [12800, 8, 128]
REPO_ROOT = Path("/home/wby/llm-cache/nvidia-kvcache").resolve()
MODEL_INVENTORY_SUPPORTING_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
    "configuration.json",
)


class ProtocolError(RuntimeError):
    """Raised whenever a scientific or provenance contract is violated."""


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON bytes used for all protocol digests."""

    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _absolute_unresolved_path(path: str | Path) -> Path:
    """Return an absolute lexical path without erasing symlink evidence."""

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if ".." in candidate.parts:
        raise ProtocolError(f"parent traversal is not permitted: {candidate}")
    return candidate


def _lstat_component_chain(
    path: str | Path, *, label: str, allow_missing_tail: bool
) -> Path:
    """Reject live/dangling symlinks in every existing path component.

    ``Path.resolve`` alone follows a symlink before a caller can notice it.
    This lexical walk is used for both external inputs and output paths.  A
    missing suffix is allowed only when creating a new artifact; once the
    parent is created callers repeat the walk before opening the file.
    """

    candidate = _absolute_unresolved_path(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError as exc:
            if allow_missing_tail:
                return candidate
            raise ProtocolError(f"{label} component is absent: {current}") from exc
        except OSError as exc:
            raise ProtocolError(f"cannot inspect {label}: {current}") from exc
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"{label} traverses a symlink: {current}")
    return candidate


def _assert_inside(root: Path, path: Path, *, label: str) -> Path:
    root_checked = _lstat_component_chain(root, label=f"{label} root", allow_missing_tail=False)
    path_checked = _lstat_component_chain(path, label=label, allow_missing_tail=True)
    # ``resolve(strict=False)`` is safe only after the component walk.  It is
    # also a race check: a parent replaced by a symlink between the two calls
    # makes the resolved spelling differ and therefore fails closed.
    if path_checked.resolve(strict=False) != path_checked:
        raise ProtocolError(f"{label} changed during symlink-safe resolution")
    try:
        path_checked.relative_to(root_checked)
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes {root_checked}") from exc
    return path_checked


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    path = _lstat_component_chain(path, label="file", allow_missing_tail=False)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ProtocolError(f"expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_ref(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    path = _lstat_component_chain(path, label="file reference", allow_missing_tail=False)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ProtocolError(f"not a regular non-symlink file: {path}")
    name = str(path.relative_to(relative_to)) if relative_to is not None else str(path.resolve())
    return {"path": name, "nbytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def strict_json(path: Path) -> dict[str, Any]:
    path = _lstat_component_chain(path, label="JSON file", allow_missing_tail=False)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ProtocolError(f"missing/nonregular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot parse JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON root must be an object: {path}")
    return value


def atomic_bytes(path: Path, payload: bytes) -> None:
    """Durably write one file, committing by atomic rename."""

    path = _assert_inside(ROOT, path, label="output")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Recheck every parent after mkdir to close a symlink substitution race.
    _lstat_component_chain(path.parent, label="output parent", allow_missing_tail=False)
    if os.path.lexists(path):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise ProtocolError(f"refusing to replace nonregular path: {path}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        _lstat_component_chain(path.parent, label="output parent", allow_missing_tail=False)
        if os.path.lexists(path) and stat.S_ISLNK(path.lstat().st_mode):
            raise ProtocolError(f"output destination became a symlink: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def immutable_json(path: Path, value: Any) -> dict[str, Any]:
    """Create an artifact once; a later different byte sequence is fatal."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    path = _assert_inside(ROOT, path, label="immutable output")
    if os.path.lexists(path):
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ProtocolError(f"immutable destination is not a regular file: {path}")
        if path.read_bytes() != payload:
            raise ProtocolError(f"immutable artifact differs: {path}")
    else:
        atomic_bytes(path, payload)
    return file_ref(path)


def write_sidecar(path: Path) -> dict[str, Any]:
    path = _lstat_component_chain(path, label="sidecar target", allow_missing_tail=False)
    sidecar = path.with_name(path.name + ".sha256.json")
    value = {"path": path.name, "nbytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    return immutable_json(sidecar, value)


def verify_sidecar(path: Path) -> None:
    path = _lstat_component_chain(path, label="sidecar target", allow_missing_tail=False)
    sidecar = _lstat_component_chain(path.with_name(path.name + ".sha256.json"), label="sidecar", allow_missing_tail=False)
    value = strict_json(sidecar)
    expected = {"path": path.name, "nbytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    if value != expected:
        raise ProtocolError(f"sidecar mismatch: {path}")


def _strict_ref(ref: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(ref, Mapping):
        raise ProtocolError(f"{label}: reference must be an object")
    if set(ref) != {"path", "nbytes", "sha256"}:
        raise ProtocolError(f"{label}: reference keys changed")
    if type(ref.get("nbytes")) is not int or ref["nbytes"] < 0:
        raise ProtocolError(f"{label}: nbytes must be an int")
    digest = ref.get("sha256")
    if type(digest) is not str or len(digest) != 64 or digest.lower() != digest:
        raise ProtocolError(f"{label}: SHA256 must be lowercase hex")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise ProtocolError(f"{label}: invalid SHA256") from exc
    raw_path = ref.get("path")
    if type(raw_path) is not str or not raw_path or not Path(raw_path).is_absolute():
        raise ProtocolError(f"{label}: path must be an absolute string")
    path = _lstat_component_chain(raw_path, label=label, allow_missing_tail=False)
    if path.resolve(strict=True) != path:
        raise ProtocolError(f"{label}: path is not canonical")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ProtocolError(f"{label}: missing/nonregular path {path}")
    observed = file_ref(path)
    if observed["nbytes"] != ref["nbytes"] or observed["sha256"] != digest:
        raise ProtocolError(f"{label}: byte binding mismatch")
    return path.resolve()


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def _strict_int(value: Any, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ProtocolError(f"{label}: expected strict int")
    if minimum is not None and value < minimum:
        raise ProtocolError(f"{label}: below minimum")
    return value


def validate_config_shape(config: Mapping[str, Any]) -> None:
    """Validate the immutable method/protocol contract without reading data."""

    _expect(type(config.get("schema_version")) is int and config["schema_version"] == 1, "schema_version changed")
    _expect(config.get("experiment_id") == "qwen3_32b_to_14b_cachebridge_head_local_full40_v1", "experiment identity changed")
    _expect(config.get("direction") == "qwen3_32b_to_14b", "direction changed")
    method = config.get("method")
    _expect(isinstance(method, Mapping), "method section missing")
    _expect(method.get("method_id") == "cachebridge_head_local_unweighted_k20", "method id changed")
    _expect(method.get("source_feature_width") == 2560 and method.get("target_feature_width") == 128, "head-local dimensions changed")
    _expect(type(method.get("ridge_lambda")) is float and method["ridge_lambda"] == 0.01, "ridge lambda must be exactly 0.01")
    _expect(method.get("weighted_loss") is False and method.get("regularize_bias") is False, "weighted/regularized objective changed")
    _expect(method.get("solver") == "cholesky" and method.get("centered_two_pass") is True, "ridge solver contract changed")
    upstream = config.get("upstream")
    _expect(isinstance(upstream, Mapping), "upstream section missing")
    required_upstream = {
        "k20_root", "k20_manifest", "k20_manifest_marker", "k20_selection", "k20_fit_definition",
        "shrinkage_root", "shrinkage_manifest", "shrinkage_manifest_marker", "split_manifest",
        "calibration_manifest", "train_tokens", "heldout_tokens",
        "source_activations", "target_activations", "fit_source_activations", "fit_target_activations",
        "heldout_source_activation_manifest", "heldout_target_activation_manifest",
        "fit_source_activation_manifest", "fit_target_activation_manifest",
        "source_model", "target_model",
    }
    _expect(set(upstream) == required_upstream, "upstream key set changed")
    model_keys = {
        "path", "config", "weight_index", "tokenizer", "inventory_digest",
        "metadata_total_size", "filesystem_total_size", "shard_count",
        "expected_num_hidden_layers", "expected_num_attention_heads",
        "expected_num_key_value_heads", "expected_head_dim",
    }
    for role in ("source_model", "target_model"):
        entry = upstream.get(role)
        _expect(type(entry) is dict and set(entry) == model_keys, f"{role} model binding shape changed")
        _expect(type(entry.get("path")) is str and Path(entry["path"]).is_absolute(), f"{role} model root changed")
        for field in ("metadata_total_size", "filesystem_total_size", "shard_count", "expected_num_hidden_layers", "expected_num_attention_heads", "expected_num_key_value_heads", "expected_head_dim"):
            _expect(type(entry.get(field)) is int and entry[field] >= 0, f"{role} model {field} must be a strict integer")
        digest = entry.get("inventory_digest")
        _expect(type(digest) is str and len(digest) == 64 and digest.lower() == digest, f"{role} model inventory digest changed")
    prior = config.get("prior_probe6")
    _expect(type(prior) is dict and set(prior) == {"role", "experiment_root", "config", "preflight", "stop"}, "prior probe6 binding changed")
    _expect(prior.get("role") == "read_only_mechanistic_prior_not_qa_proof", "prior probe6 role changed")
    _expect(type(prior.get("experiment_root")) is str and Path(prior["experiment_root"]).is_absolute(), "prior probe6 root changed")
    for name in ("config", "preflight", "stop"):
        ref = prior.get(name)
        _expect(type(ref) is dict and set(ref) == {"path", "nbytes", "sha256"}, f"prior probe6 {name} ref changed")
        _expect(type(ref.get("nbytes")) is int and ref["nbytes"] >= 0, f"prior probe6 {name} size changed")
        _expect(type(ref.get("sha256")) is str and len(ref["sha256"]) == 64 and ref["sha256"].lower() == ref["sha256"], f"prior probe6 {name} hash changed")
    protocol = config.get("protocol")
    _expect(isinstance(protocol, Mapping), "protocol section missing")
    _expect(protocol.get("source_layers") == 64 and protocol.get("target_layers") == 40, "model layer counts changed")
    _expect(protocol.get("selected_source_layers") == 20 and protocol.get("kv_heads") == 8 and protocol.get("head_dim") == 128, "KV topology changed")
    _expect(protocol.get("components") == list(COMPONENTS), "component order changed")
    _expect(protocol.get("probe_target_layers") == list(PROBE_LAYERS), "probe reference layer order changed")
    _expect(protocol.get("full_target_layers") == list(FULL_LAYERS), "full layer registration changed")
    _expect(protocol.get("tuning_sequence_count") == 40 and protocol.get("locked_sequence_count") == 10, "split cardinality changed")
    _expect(protocol.get("fit_split") == "train" and protocol.get("eval_split") == "heldout_tuning", "fit/eval split contract changed")
    _expect(protocol.get("fit_sequence_count") == 500 and protocol.get("eval_sequence_count") == 40, "fit/eval sequence counts changed")
    _expect(protocol.get("observations_per_sequence") == 256 and protocol.get("activation_stride") == 4, "activation geometry changed")
    _expect(protocol.get("evaluation_scope") == "02_frozen_tuning40_only", "evaluation scope changed")
    _expect(protocol.get("locked_activation_evaluations") == 0 and protocol.get("locked_rows_indexed") is False, "locked boundary changed")
    _expect(protocol.get("require_k20_and_shrinkage_selection_equal") is True, "selection equality guard missing")
    _expect(protocol.get("comparison_methods") == ["nvidia_k20", "shrinkage", "cachebridge_head_local"], "comparison methods changed")
    gate = protocol.get("promotion_gate")
    _expect(isinstance(gate, Mapping), "promotion gate missing")
    _expect(gate.get("full40_candidate_sse_strictly_below_k20_and_shrinkage_for_each_component") is True, "full40 gate changed")
    _expect(gate.get("primary_generalization_baseline") == "nvidia_k20_train500" and gate.get("shrinkage_comparator_status") == "validation_tuned_on_heldout_tuning40_exploratory_only" and gate.get("candidate_fit_split") == "train500" and gate.get("candidate_eval_split") == "heldout_tuning40", "full40 comparator split contract changed")
    _expect(gate.get("full40_gate_rows") == 80 and gate.get("prior_probe6_is_not_qa_proof") is True and gate.get("locked_activation_evaluations") == 0, "full40 gate contract changed")
    runtime = config.get("runtime")
    _expect(isinstance(runtime, Mapping), "runtime section missing")
    _expect(runtime.get("gpu_count") == 8 and runtime.get("cuda_visible_devices") == "0,1,2,3,4,5,6,7", "GPU topology changed")
    _expect(runtime.get("layer_sharding") == "target_layer_modulo_visible_gpu_count", "layer sharding changed")
    _expect(runtime.get("global_gpu_lock") == "/home/wby/llm-cache/nvidia-kvcache/runs/gpu8_experiment.lock", "GPU lock path changed")
    _expect(runtime.get("required_python") == "/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python", "Python binding changed")
    _expect(runtime.get("required_python_major_minor") == "3.12", "Python version binding changed")
    _expect(runtime.get("fresh_pycache_policy") == "external_absolute_absent_prefix_per_invocation", "pycache policy changed")
    _expect(runtime.get("gpu_execution_requires_explicit_execute") is True and runtime.get("models_loaded") is False, "execution gate changed")
    _expect(runtime.get("preflight_cpu_only") is True and runtime.get("cuda_visible_devices_preflight") == "", "CPU preflight gate changed")
    outputs = config.get("outputs")
    _expect(isinstance(outputs, Mapping), "outputs section missing")
    _expect(outputs.get("preflight_receipt") == "artifacts/preflight/input_preflight.json", "preflight output changed")
    _expect(outputs.get("stop_receipt") == "artifacts/preformal/PRE_FORMAL_STOP_v2.json", "stop output changed")
    _expect(set(outputs) == {"preflight_receipt", "stop_receipt", "full40_run_root"}, "output key set changed")
    _expect(outputs.get("full40_run_root") == "runs/full40", "run root changed")


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = ROOT / "config.json"
    path = _lstat_component_chain(config_path, label="config", allow_missing_tail=False)
    if path.resolve(strict=True) != path:
        raise ProtocolError("config path is not canonical")
    _expect(path == ROOT / "config.json", "only the sibling's canonical config.json is accepted")
    config = strict_json(path)
    validate_config_shape(config)
    assert_fit_eval_split_disjoint(config)
    return config


def _verify_config_ref(config: Mapping[str, Any], key: str) -> Path:
    return _strict_ref(config["upstream"][key], label=f"upstream.{key}")


def _model_ref_under_root(model_root: Path, ref: Mapping[str, Any], *, label: str, filename: str) -> Path:
    """Verify a model metadata ref and require its canonical expected name."""

    path = _strict_ref(ref, label=label)
    expected = model_root / filename
    if path != expected:
        raise ProtocolError(f"{label}: expected canonical path {expected}")
    return path


def model_shard_inventory(model_entry: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Hash every indexed model shard and loader asset before any torch import.

    The digest intentionally matches the audited model-inventory convention
    used by the earlier experiments: indexed safetensors followed by the six
    small tokenizer/generation assets, each represented by an absolute,
    content-addressed file reference.  Model ``config.json`` and the index
    itself are separately bound refs and are also checked for architecture.
    """

    required = {
        "path", "config", "weight_index", "tokenizer", "inventory_digest",
        "metadata_total_size", "filesystem_total_size", "shard_count",
        "expected_num_hidden_layers", "expected_num_attention_heads",
        "expected_num_key_value_heads", "expected_head_dim",
    }
    if not isinstance(model_entry, Mapping) or set(model_entry) != required:
        raise ProtocolError(f"{label}: model binding keys changed")
    raw_root = model_entry.get("path")
    if type(raw_root) is not str or not raw_root or not Path(raw_root).is_absolute():
        raise ProtocolError(f"{label}: model root must be an absolute string")
    model_root = _lstat_component_chain(raw_root, label=f"{label} root", allow_missing_tail=False)
    if model_root.resolve(strict=True) != model_root or not stat.S_ISDIR(model_root.lstat().st_mode):
        raise ProtocolError(f"{label}: model root must be a canonical directory")
    config_path = _model_ref_under_root(model_root, model_entry["config"], label=f"{label} config", filename="config.json")
    index_path = _model_ref_under_root(model_root, model_entry["weight_index"], label=f"{label} weight index", filename="model.safetensors.index.json")
    tokenizer_path = _model_ref_under_root(model_root, model_entry["tokenizer"], label=f"{label} tokenizer", filename="tokenizer.json")
    model_config = strict_json(config_path)
    expected_arch = {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "num_hidden_layers": _strict_int(model_entry["expected_num_hidden_layers"], label=f"{label} expected layers", minimum=1),
        "num_attention_heads": _strict_int(model_entry["expected_num_attention_heads"], label=f"{label} expected attention heads", minimum=1),
        "num_key_value_heads": _strict_int(model_entry["expected_num_key_value_heads"], label=f"{label} expected KV heads", minimum=1),
        "head_dim": _strict_int(model_entry["expected_head_dim"], label=f"{label} expected head dim", minimum=1),
        "torch_dtype": "bfloat16",
        "vocab_size": 151936,
    }
    for key, expected in expected_arch.items():
        if model_config.get(key) != expected:
            raise ProtocolError(f"{label}: model config {key} changed")
    if model_config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ProtocolError(f"{label}: architecture list changed")
    index = strict_json(index_path)
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, Mapping) or not weight_map or not isinstance(metadata, Mapping):
        raise ProtocolError(f"{label}: model index is malformed")
    names = sorted({value for value in weight_map.values() if type(value) is str})
    if len(names) != len({value for value in weight_map.values()}):
        raise ProtocolError(f"{label}: model shard names are not strict strings")
    if not names or any(not name.endswith(".safetensors") or Path(name).name != name or ".." in Path(name).parts for name in names):
        raise ProtocolError(f"{label}: model shard list is malformed")
    expected_count = _strict_int(model_entry["shard_count"], label=f"{label} shard count", minimum=1)
    if len(names) != expected_count:
        raise ProtocolError(f"{label}: shard count changed")
    shards: list[dict[str, Any]] = []
    for name in names:
        shard = _lstat_component_chain(model_root / name, label=f"{label} shard {name}", allow_missing_tail=False)
        if shard.resolve(strict=True) != shard or not stat.S_ISREG(shard.lstat().st_mode):
            raise ProtocolError(f"{label}: shard is not a regular canonical file")
        shards.append(file_ref(shard))
    # Reject unindexed safetensors, including a malicious replacement dropped
    # beside the indexed files.
    observed_safetensors = sorted(
        p.name for p in model_root.iterdir()
        if not p.is_symlink() and stat.S_ISREG(p.lstat().st_mode) and p.name.endswith(".safetensors")
    )
    if observed_safetensors != names:
        raise ProtocolError(f"{label}: unindexed or missing safetensors detected")
    expected_metadata_total = _strict_int(model_entry["metadata_total_size"], label=f"{label} metadata total", minimum=0)
    index_metadata_total = _strict_int(metadata.get("total_size"), label=f"{label} index metadata total", minimum=0)
    if index_metadata_total != expected_metadata_total:
        raise ProtocolError(f"{label}: index metadata total changed")
    observed_total = sum(int(item["nbytes"]) for item in shards)
    expected_filesystem_total = _strict_int(model_entry["filesystem_total_size"], label=f"{label} filesystem total", minimum=0)
    if observed_total != expected_filesystem_total or observed_total < expected_metadata_total:
        raise ProtocolError(f"{label}: shard byte total changed")
    supporting: list[dict[str, Any]] = []
    for name in MODEL_INVENTORY_SUPPORTING_FILES:
        candidate = _lstat_component_chain(model_root / name, label=f"{label} supporting {name}", allow_missing_tail=False)
        if candidate.resolve(strict=True) != candidate or not stat.S_ISREG(candidate.lstat().st_mode):
            raise ProtocolError(f"{label}: supporting loader asset is not regular")
        supporting.append(file_ref(candidate))
    inventory_files = [*shards, *supporting]
    digest = canonical_sha256(inventory_files)
    expected_digest = model_entry["inventory_digest"]
    if type(expected_digest) is not str or len(expected_digest) != 64 or expected_digest.lower() != expected_digest or digest != expected_digest:
        raise ProtocolError(f"{label}: complete shard inventory digest changed")
    return {
        "model_root": str(model_root),
        "config": file_ref(config_path),
        "weight_index": file_ref(index_path),
        "tokenizer": file_ref(tokenizer_path),
        "metadata_total_size": expected_metadata_total,
        "filesystem_total_size": observed_total,
        "shards": shards,
        "supporting_files": supporting,
        "shard_count": len(shards),
        "architecture": {key: expected_arch[key] for key in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "torch_dtype", "vocab_size")},
        "digest": digest,
    }


def source_byte_closure() -> dict[str, Any]:
    """Return a deterministic digest of all executable/config source bytes.

    Generated artifacts and run roots are intentionally outside this closure;
    they are bound by their own receipts.  Every existing component is
    inspected with ``lstat`` so a symlink, FIFO, pyc, or cache directory cannot
    silently enter the execution surface.
    """

    roots = [ROOT / "config.json", ROOT / "README.md", ROOT / "src", ROOT / "scripts", ROOT / "tests", ROOT / "contracts"]
    files: list[dict[str, Any]] = []

    def visit(path: Path) -> None:
        mode = os.lstat(path).st_mode
        if stat.S_ISLNK(mode):
            raise ProtocolError(f"source closure traverses symlink: {path}")
        if stat.S_ISREG(mode):
            if path.suffix == ".pyc" or "__pycache__" in path.parts or path.name.startswith("."):
                raise ProtocolError(f"generated/hidden file in source closure: {path}")
            relative = path.relative_to(ROOT).as_posix()
            files.append({"path": relative, "nbytes": int(path.stat().st_size), "sha256": sha256_file(path)})
            return
        if not stat.S_ISDIR(mode):
            raise ProtocolError(f"non-directory/non-regular source entry: {path}")
        for child in sorted(path.iterdir(), key=lambda item: item.name):
            visit(child)

    for root in roots:
        if not os.path.lexists(root):
            # Optional documentation/test/contract roots are allowed while a
            # snapshot is being assembled; the mandatory config/src roots are
            # checked explicitly below.
            if root.name in {"config.json", "src"}:
                raise ProtocolError(f"mandatory source entry missing: {root}")
            continue
        visit(root)
    files.sort(key=lambda item: item["path"])
    return {"root": str(ROOT), "files": files, "file_count": len(files), "digest": canonical_sha256(files)}


def validate_source_closure(expected: Mapping[str, Any]) -> dict[str, Any]:
    observed = source_byte_closure()
    if observed != dict(expected):
        raise ProtocolError("source byte closure changed after freeze")
    return observed


def _manifest_layers(manifest: Mapping[str, Any], *, label: str) -> list[Mapping[str, Any]]:
    _expect(type(manifest.get("schema_version")) is int and manifest.get("schema_version") == 1, f"{label}: manifest schema changed")
    _expect(manifest.get("status") == "complete" and manifest.get("direction") == "qwen3_32b_to_14b", f"{label}: manifest state changed")
    arch = manifest.get("architecture")
    _expect(isinstance(arch, Mapping), f"{label}: architecture missing")
    _expect(arch.get("source_layers") == 64 and arch.get("target_layers") == 40, f"{label}: layer counts changed")
    _expect(arch.get("selected_source_layers") == 20 and arch.get("kv_heads") == 8 and arch.get("head_dim") == 128, f"{label}: topology changed")
    layers = manifest.get("layers")
    _expect(isinstance(layers, list) and len(layers) == 40, f"{label}: layer inventory changed")
    for index, entry in enumerate(layers):
        _expect(isinstance(entry, Mapping) and entry.get("target_layer") == index, f"{label}: target order changed")
        selected = entry.get("source_layers_rank_order")
        _expect(isinstance(selected, list) and len(selected) == 20 and len(set(selected)) == 20, f"{label}: source selection cardinality changed")
        _expect(all(type(v) is int and 0 <= v < 64 for v in selected), f"{label}: invalid source layer")
    return layers


def _check_manifest_fit_definition(manifest: Mapping[str, Any], *, label: str) -> None:
    fit = manifest.get("fit_definition")
    if label == "k20":
        _expect(isinstance(fit, Mapping), "k20 fit definition missing")
        _expect(fit.get("method") == "two-pass-centered-ridge", "k20 method changed")
        _expect(fit.get("ridge_lambda_added_directly_to_unnormalized_gram") == 0.01, "k20 lambda changed")
        _expect(fit.get("feature_order") == "selected_layer_rank,kv_head,head_dim", "k20 feature order changed")
        arch = fit.get("architecture")
        _expect(isinstance(arch, Mapping), "k20 fit architecture missing")
        _expect(arch.get("n_samples") == 128000 and arch.get("source_layers") == 64 and arch.get("target_layers") == 40 and arch.get("selected_source_layers") == 20 and arch.get("kv_heads") == 8 and arch.get("head_dim") == 128, "k20 fit sample/topology changed")
    else:
        _expect(isinstance(manifest.get("method_definition"), Mapping), "shrinkage method definition missing")
        definition = manifest["method_definition"]
        _expect(definition.get("feature_order") == "selected_layer_rank,kv_head,head_dim", "shrinkage feature order changed")
        _expect(definition.get("method") == "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers", "shrinkage method changed")
        _expect(definition.get("formula") == "f4 + alpha * (f20 - f4)", "shrinkage formula changed")
        _expect(definition.get("locked_audit_used_for_alpha") is False, "shrinkage alpha touched locked rows")
        _expect(manifest.get("status") == "complete", "shrinkage status changed")
        alpha = manifest.get("alpha")
        _expect(isinstance(alpha, Mapping) and type(alpha.get("path")) is str and type(alpha.get("sha256")) is str, "shrinkage alpha binding missing")
        # The alpha manifest is rooted at the shrinkage artifact root and is
        # checked by callers once the root is known.  Here we only require the
        # immutable split semantics exposed in the parent manifest.
        split = manifest.get("split")
        _expect(isinstance(split, Mapping) and split.get("sha256") == definition.get("split_manifest_sha256"), "shrinkage split binding changed")


def _validate_shrinkage_alpha_binding(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    shrinkage_root: Path,
    split: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate the shrinkage comparator's heldout-tuned alpha artifact.

    The comparator is retained for continuity with the earlier experiments,
    but its alpha was selected on the same forty tuning sequences used for
    this sibling's evaluation.  We bind that fact explicitly and expose it
    as exploratory metadata; it is never presented as a clean generalization
    baseline (the train-500 NVIDIA k20 mapper is the primary baseline).
    """

    alpha_ref = manifest.get("alpha")
    _expect(isinstance(alpha_ref, Mapping), "shrinkage alpha reference missing")
    alpha_path = _resolve_under(shrinkage_root, str(alpha_ref.get("path")), label="shrinkage alpha manifest")
    observed_alpha = file_ref(alpha_path)
    _expect(observed_alpha["nbytes"] == alpha_ref.get("nbytes") and observed_alpha["sha256"] == alpha_ref.get("sha256"), "shrinkage alpha artifact binding changed")
    alpha = strict_json(alpha_path)
    _expect(type(alpha.get("schema_version")) is int and alpha.get("schema_version") == 1 and alpha.get("status") == "frozen_before_locked_audit", "shrinkage alpha state changed")
    _expect(alpha.get("formula") == "f4 + alpha * (f20 - f4)" and alpha.get("granularity") == "target_layer,cache_component,target_kv_head", "shrinkage alpha definition changed")
    _expect(alpha.get("locked_audit_used_for_alpha") is False, "shrinkage alpha used locked rows")
    fit_ids = alpha.get("fit_sequence_ids")
    locked_ids = alpha.get("locked_audit_sequence_ids")
    _expect(isinstance(fit_ids, list) and len(fit_ids) == 40 and all(type(v) is int for v in fit_ids), "shrinkage alpha fit IDs changed")
    _expect(isinstance(locked_ids, list) and len(locked_ids) == 10 and all(type(v) is int for v in locked_ids), "shrinkage alpha locked IDs changed")
    _expect(set(fit_ids) == set(split["tuning_sequence_ids"]) and set(locked_ids).isdisjoint(fit_ids), "shrinkage alpha split IDs changed")
    # Obtain the frozen locked namespace directly from the split manifest so
    # numeric IDs cannot be mistaken for the train namespace.
    split_path = _strict_ref(config["upstream"]["split_manifest"], label="upstream.split_manifest")
    split_json = strict_json(split_path)
    frozen_locked = split_json.get("locked_audit_sequence_ids")
    _expect(isinstance(frozen_locked, list) and set(locked_ids) == set(frozen_locked), "shrinkage alpha locked namespace changed")
    layers = alpha.get("layers")
    _expect(isinstance(layers, list) and len(layers) == 40, "shrinkage alpha layer coverage changed")
    for index, row in enumerate(layers):
        _expect(isinstance(row, Mapping) and row.get("target_layer") == index, "shrinkage alpha target order changed")
        for component in COMPONENTS:
            comp = row.get(component)
            _expect(isinstance(comp, Mapping), "shrinkage alpha component missing")
            values = comp.get("alpha")
            _expect(isinstance(values, list) and len(values) == 8, "shrinkage alpha head cardinality changed")
            for value in values:
                _expect(type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0, "shrinkage alpha value changed")
    return {"path": str(alpha_path), **observed_alpha, "fit_sequence_count": len(fit_ids), "locked_sequence_count": len(locked_ids), "fit_split": "heldout_tuning", "status": "validation_tuned_exploratory_only"}


def _validate_complete_selection_snapshot(
    config: Mapping[str, Any],
    k20_manifest: Mapping[str, Any],
    shrink_manifest: Mapping[str, Any],
) -> list[tuple[int, ...]]:
    """Cross-check the standalone selection snapshot for every target layer.

    The full40 runner must never silently fall back to the six-layer probe
    subset.  The manifest and its standalone ``selected_layers.json`` are
    therefore parsed independently and compared byte-derived, layer by layer.
    """

    selection_path = _verify_config_ref(config, "k20_selection")
    snapshot = strict_json(selection_path)
    _expect(snapshot.get("direction") == "qwen3_32b_to_14b", "selection direction changed")
    _expect(snapshot.get("source_layers") == 64 and snapshot.get("target_layers") == 40, "selection topology changed")
    _expect(snapshot.get("k") == 20 and snapshot.get("kv_heads") == 8 and snapshot.get("head_dim") == 128, "selection dimensions changed")
    rows = snapshot.get("selected_source_layers_rank_order")
    _expect(isinstance(rows, list) and len(rows) == len(ACTIVE_LAYERS), "complete selection snapshot is not 40-layer")
    manifest_rows = k20_manifest.get("layers")
    shrink_rows = shrink_manifest.get("layers")
    _expect(isinstance(manifest_rows, list) and isinstance(shrink_rows, list), "selection manifests are missing")
    selected: list[tuple[int, ...]] = []
    for target in ACTIVE_LAYERS:
        raw = rows[target]
        _expect(isinstance(raw, list) and len(raw) == 20, f"selection layer {target} cardinality changed")
        values = tuple(_strict_int(v, label=f"selection layer {target} source", minimum=0) for v in raw)
        _expect(len(set(values)) == 20 and all(v < 64 for v in values), f"selection layer {target} values changed")
        _expect(manifest_rows[target].get("source_layers_rank_order") == list(values), f"k20 manifest selection differs at layer {target}")
        _expect(shrink_rows[target].get("source_layers_rank_order") == list(values), f"shrinkage manifest selection differs at layer {target}")
        selected.append(values)
    # This field is the exact selection file hash recorded by the upstream fit;
    # requiring it prevents a merely path-compatible replacement from passing.
    _expect(type(snapshot.get("input_fingerprint")) is str and len(snapshot["input_fingerprint"]) == 64 and snapshot["input_fingerprint"].lower() == snapshot["input_fingerprint"], "selection snapshot fingerprint is missing")
    manifest_selection = k20_manifest.get("selection")
    _expect(isinstance(manifest_selection, Mapping) and manifest_selection.get("snapshot_sha256") == config["upstream"]["k20_selection"]["sha256"] and manifest_selection.get("source_file_sha256_used_by_fit") == config["upstream"]["k20_selection"]["sha256"], "selection snapshot byte binding changed")
    fit_path = _verify_config_ref(config, "k20_fit_definition")
    fit = strict_json(fit_path)
    fit_arch = fit.get("architecture")
    _expect(isinstance(fit_arch, Mapping) and fit_arch.get("n_samples") == 128000 and fit_arch.get("selected_source_layers") == 20 and fit_arch.get("source_layers") == 64 and fit_arch.get("target_layers") == 40 and fit_arch.get("kv_heads") == 8 and fit_arch.get("head_dim") == 128, "k20 fit definition topology changed")
    _expect(fit.get("feature_order") == "selected_layer_rank,kv_head,head_dim", "k20 fit feature order changed")
    _expect(fit.get("selection_sha256") == config["upstream"]["k20_selection"]["sha256"], "k20 fit selection binding changed")
    return selected


def _resolve_under(root: Path, relative: str, *, label: str) -> Path:
    if type(relative) is not str or not relative:
        raise ProtocolError(f"{label}: relative path is malformed")
    root = _lstat_component_chain(root, label=f"{label} root", allow_missing_tail=False)
    raw = Path(relative)
    if raw.is_absolute():
        path = _lstat_component_chain(raw, label=label, allow_missing_tail=False)
    else:
        if ".." in raw.parts:
            raise ProtocolError(f"{label}: relative path is malformed")
        path = _lstat_component_chain(root / raw, label=label, allow_missing_tail=False)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProtocolError(f"{label}: path escapes root") from exc
    if path.resolve(strict=True) != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ProtocolError(f"{label}: missing/nonregular path")
    return path


def _mapper_bindings(config: Mapping[str, Any], k20_manifest: Mapping[str, Any], shrink_manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kroot = Path(config["upstream"]["k20_root"]).resolve()
    sroot = Path(config["upstream"]["shrinkage_root"]).resolve()
    k_layers = _manifest_layers(k20_manifest, label="k20")
    s_layers = _manifest_layers(shrink_manifest, label="shrinkage")
    _check_manifest_fit_definition(k20_manifest, label="k20")
    _check_manifest_fit_definition(shrink_manifest, label="shrinkage")
    # Bind the comparator's alpha file and prove that its fit namespace is
    # exactly the forty heldout tuning IDs.  This is intentionally metadata
    # only: candidate fitting below uses the independent train-500 split.
    split = _load_split(config)
    alpha_binding = _validate_shrinkage_alpha_binding(config, shrink_manifest, sroot, split)
    rows: list[dict[str, Any]] = []
    for target in ACTIVE_LAYERS:
        ksel = list(k_layers[target]["source_layers_rank_order"])
        ssel = list(s_layers[target]["source_layers_rank_order"])
        _expect(ksel == ssel, f"target layer {target}: k20/shrinkage selection differs")
        for name, root, manifest_layers in (("nvidia_k20", kroot, k_layers), ("shrinkage", sroot, s_layers)):
            entry = manifest_layers[target]
            payload = _resolve_under(root, str(entry["path"]), label=f"{name} layer {target} payload")
            sidecar = _resolve_under(root, str(entry["sidecar_path"]), label=f"{name} layer {target} sidecar")
            payload_ref = file_ref(payload)
            sidecar_ref = file_ref(sidecar)
            _expect(payload_ref["sha256"] == entry.get("sha256"), f"{name} layer {target}: payload SHA mismatch")
            _expect(sidecar_ref["sha256"] == entry.get("sidecar_sha256"), f"{name} layer {target}: sidecar SHA mismatch")
            rows.append({"method": name, "target_layer": target, "payload": payload_ref, "sidecar": sidecar_ref, "source_layers_rank_order": ksel})
    return rows, alpha_binding


def _load_split(config: Mapping[str, Any]) -> dict[str, Any]:
    split_path = Path(config["upstream"]["split_manifest"]["path"])
    split = strict_json(split_path)
    token_file = split.get("token_file")
    if not isinstance(token_file, Mapping):
        raise ProtocolError("split token binding is missing")
    split_token_path = _strict_ref(token_file, label="split token file")
    heldout_ref = config["upstream"]["heldout_tokens"]
    expected_token_path = _strict_ref(heldout_ref, label="upstream.heldout_tokens")
    if split_token_path != expected_token_path or file_ref(split_token_path) != dict(heldout_ref):
        raise ProtocolError("split token file does not match the frozen heldout token ref")
    tuning_raw = split.get("tuning_sequence_ids")
    locked_raw = split.get("locked_audit_sequence_ids")
    _expect(isinstance(tuning_raw, list) and len(tuning_raw) == 40, "tuning split cardinality changed")
    _expect(isinstance(locked_raw, list) and len(locked_raw) == 10, "locked split cardinality changed")
    tuning = tuple(_strict_int(v, label="tuning sequence id", minimum=0) for v in tuning_raw)
    locked = tuple(_strict_int(v, label="locked sequence id", minimum=0) for v in locked_raw)
    _expect(len(set(tuning)) == 40 and len(set(locked)) == 10, "split contains duplicate ids")
    _expect(set(tuning).isdisjoint(locked) and all(v < 50 for v in tuning + locked), "tuning/locked split overlap")
    _expect(split.get("locked_audit_used_for_alpha") is False, "locked split was used upstream")
    folds = split.get("cv_folds")
    _expect(isinstance(folds, list) and len(folds) == 5, "split fold count changed")
    return {
        "split": "heldout",
        "sequence_count": 50,
        "token_sha256": str(heldout_ref["sha256"]),
        "tuning_sequence_ids": list(tuning),
        "tuning_sequence_ids_sha256": canonical_sha256(list(tuning)),
        "locked_sequence_count": 10,
        "locked_sequence_ids_sha256": canonical_sha256(list(locked)),
        "locked_rows_indexed": False,
        "fold_manifest_sha256": canonical_sha256(folds),
    }


def _load_train_split(config: Mapping[str, Any]) -> dict[str, Any]:
    """Load the immutable 500-sequence fit namespace without opening values."""

    calibration_path = _strict_ref(config["upstream"]["calibration_manifest"], label="calibration manifest")
    calibration = strict_json(calibration_path)
    _expect(calibration.get("status") == "complete" and calibration.get("schema_version") == 1, "calibration manifest state changed")
    splits = calibration.get("splits")
    _expect(isinstance(splits, Mapping) and isinstance(splits.get("train"), Mapping), "train calibration split missing")
    train_meta = splits["train"]
    _expect(train_meta.get("shape") == [500, 1024] and train_meta.get("dtype") == "int32", "train token geometry changed")
    train_ref = _strict_ref(config["upstream"]["train_tokens"], label="train token file")
    expected_path = calibration_path.parent / str(train_meta.get("file"))
    _expect(train_ref == expected_path and file_ref(train_ref) == dict(config["upstream"]["train_tokens"]), "train token file binding changed")
    payload_sha = train_meta.get("token_ids_payload_sha256")
    _expect(type(payload_sha) is str and len(payload_sha) == 64 and payload_sha.lower() == payload_sha, "train token payload digest missing")
    ids = tuple(range(500))
    return {
        "split": "train",
        "sequence_count": 500,
        "token_sha256": str(config["upstream"]["train_tokens"]["sha256"]),
        "fit_sequence_ids": list(ids),
        "fit_sequence_ids_sha256": canonical_sha256(list(ids)),
        "locked_sequence_count": 0,
        "locked_rows_indexed": False,
        "token_ids_payload_sha256": payload_sha,
    }


def _load_calibration_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the shared calibration manifest and both token namespaces."""

    calibration_path = _strict_ref(config["upstream"]["calibration_manifest"], label="calibration manifest")
    calibration = strict_json(calibration_path)
    _expect(type(calibration.get("schema_version")) is int and calibration.get("schema_version") == 1 and calibration.get("status") == "complete", "calibration manifest state changed")
    dataset_fingerprint = calibration.get("dataset_fingerprint")
    config_fingerprint = calibration.get("config_fingerprint")
    _expect(type(dataset_fingerprint) is str and len(dataset_fingerprint) == 64 and dataset_fingerprint.lower() == dataset_fingerprint, "calibration dataset fingerprint changed")
    _expect(type(config_fingerprint) is str and len(config_fingerprint) == 64 and config_fingerprint.lower() == config_fingerprint, "calibration config fingerprint changed")
    splits = calibration.get("splits")
    artifacts = calibration.get("artifacts")
    _expect(isinstance(splits, Mapping) and isinstance(artifacts, Mapping), "calibration split/artifact sections missing")
    expected = {
        "train": (config["upstream"]["train_tokens"], 500, "train_tokens.npy"),
        "heldout": (config["upstream"]["heldout_tokens"], 50, "heldout_tokens.npy"),
    }
    result: dict[str, Any] = {"path": str(calibration_path), "sha256": config["upstream"]["calibration_manifest"]["sha256"], "dataset_fingerprint": dataset_fingerprint, "config_fingerprint": config_fingerprint, "splits": {}}
    for split_name, (ref, count, filename) in expected.items():
        meta = splits.get(split_name)
        _expect(isinstance(meta, Mapping) and meta.get("shape") == [count, 1024] and meta.get("dtype") == "int32", f"calibration {split_name} geometry changed")
        token_ref = _strict_ref(ref, label=f"{split_name} token file")
        _expect(token_ref == calibration_path.parent / filename, f"calibration {split_name} token path changed")
        artifact = artifacts.get(filename)
        _expect(isinstance(artifact, Mapping) and artifact.get("bytes") == ref.get("nbytes") and artifact.get("sha256") == ref.get("sha256"), f"calibration {split_name} token artifact binding changed")
        payload_sha = meta.get("token_ids_payload_sha256")
        _expect(type(payload_sha) is str and len(payload_sha) == 64 and payload_sha.lower() == payload_sha, f"calibration {split_name} payload digest missing")
        result["splits"][split_name] = {"sequence_count": count, "token_ref": dict(ref), "token_ids_payload_sha256": payload_sha}
    return result


def _activation_paths(
    config: Mapping[str, Any],
    selections: Mapping[int, Sequence[int]],
    *,
    source_key: str,
    target_key: str,
) -> tuple[list[Path], list[Path]]:
    source_layers = sorted({int(v) for row in selections.values() for v in row})
    target_layers = list(ACTIVE_LAYERS)
    payloads: list[Path] = []
    metadata: list[Path] = []
    for component in COMPONENTS:
        subdir = COMPONENT_DIR[component]
        source_root = _lstat_component_chain(Path(config["upstream"][source_key]) / subdir, label="source activation root", allow_missing_tail=False)
        target_root = _lstat_component_chain(Path(config["upstream"][target_key]) / subdir, label="target activation root", allow_missing_tail=False)
        for layer in source_layers:
            stem = source_root / f"layer_{layer:03d}"
            payloads.append(_lstat_component_chain(stem.with_suffix(".bf16.bin"), label="source activation payload", allow_missing_tail=False))
            metadata.append(_lstat_component_chain(stem.with_suffix(".json"), label="source activation metadata", allow_missing_tail=False))
        for layer in target_layers:
            stem = target_root / f"layer_{layer:03d}"
            payloads.append(_lstat_component_chain(stem.with_suffix(".bf16.bin"), label="target activation payload", allow_missing_tail=False))
            metadata.append(_lstat_component_chain(stem.with_suffix(".json"), label="target activation metadata", allow_missing_tail=False))
    return sorted(set(payloads)), sorted(set(metadata))


def _validate_activation_manifest(
    manifest: Mapping[str, Any],
    root: str,
    model_inventory: Mapping[str, Any],
    split_name: str,
    sequence_count: int,
    token_sha: str,
    layer_count: int,
    calibration_manifest_sha: str,
    token_path: str,
    dataset_fingerprint: str,
) -> None:
    """Validate one train/heldout activation manifest and its inventory hash."""

    _expect(manifest.get("schema_version") == 2 and manifest.get("status") == "complete", f"{split_name}: activation manifest state changed")
    entries = manifest.get("entries")
    _expect(isinstance(entries, list) and len(entries) == layer_count * len(COMPONENTS), f"{split_name}: activation manifest entry count changed")
    expected_pairs = {(component, layer) for component in COMPONENTS for layer in range(layer_count)}
    seen_pairs: set[tuple[str, int]] = set()
    for entry in entries:
        _expect(isinstance(entry, Mapping) and set(entry) == {"cache_type", "layer", "nbytes", "path", "sha256", "shape", "sidecar_sha256"}, f"{split_name}: activation manifest entry schema changed")
        component = entry.get("cache_type")
        layer = _strict_int(entry.get("layer"), label=f"{split_name} manifest layer", minimum=0)
        _expect(component in COMPONENT_DIR.values() and layer < layer_count, f"{split_name}: activation manifest layer/component changed")
        component_name = next(name for name, directory in COMPONENT_DIR.items() if directory == component)
        pair = (component_name, layer)
        _expect(pair not in seen_pairs, f"{split_name}: duplicate activation manifest entry")
        seen_pairs.add(pair)
        _expect(entry.get("path") == f"{component}/layer_{layer:03d}.bf16.bin", f"{split_name}: activation manifest path changed")
        expected_bytes = int(sequence_count) * 256 * 8 * 128 * 2
        _expect(entry.get("nbytes") == expected_bytes and entry.get("shape") == [int(sequence_count) * 256, 8, 128], f"{split_name}: activation manifest geometry changed")
        payload = _lstat_component_chain(Path(root) / str(entry["path"]), label=f"{split_name} activation manifest payload", allow_missing_tail=False)
        _expect(file_ref(payload)["sha256"] == entry.get("sha256") and file_ref(payload)["nbytes"] == expected_bytes, f"{split_name}: activation manifest payload binding changed")
        companion = payload.with_name(payload.name.removesuffix(".bf16.bin") + ".json")
        _expect(file_ref(companion)["sha256"] == entry.get("sidecar_sha256"), f"{split_name}: activation manifest sidecar binding changed")
    _expect(seen_pairs == expected_pairs, f"{split_name}: activation manifest coverage changed")
    _expect(manifest.get("inventory_sha256") == canonical_sha256(entries), f"{split_name}: activation manifest inventory digest changed")
    run = manifest.get("run_definition")
    _expect(isinstance(run, Mapping), f"{split_name}: activation run definition missing")
    _expect(run.get("sequences") == sequence_count and run.get("sampled_positions_per_sequence") == 256 and run.get("sample_stride") == 4 and run.get("sample_stride_start") == 0, f"{split_name}: activation run geometry changed")
    calibration = run.get("calibration")
    _expect(isinstance(calibration, Mapping) and calibration.get("split") == ("train" if split_name == "train" else "heldout") and calibration.get("token_sha256") == token_sha and calibration.get("manifest_sha256") == calibration_manifest_sha and calibration.get("token_path") == token_path and calibration.get("dataset_fingerprint") == dataset_fingerprint, f"{split_name}: calibration provenance changed")
    model = run.get("model")
    _expect(isinstance(model, Mapping) and model.get("path") == model_inventory["model_root"] and model.get("config_sha256") == model_inventory["config"]["sha256"] and model.get("weight_index_sha256") == model_inventory["weight_index"]["sha256"] and model.get("tokenizer_sha256") == model_inventory["tokenizer"]["sha256"], f"{split_name}: activation model provenance changed")


def _validate_activation_inventory(
    config: Mapping[str, Any],
    selections: Mapping[int, Sequence[int]],
    split: Mapping[str, Any],
    token_sha: str,
    model_inventories: Mapping[str, Mapping[str, Any]],
    *,
    split_name: str,
    sequence_count: int,
    source_key: str,
    target_key: str,
    source_manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    calibration_manifest_sha: str,
    token_path: str,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    payload_paths, metadata_paths = _activation_paths(config, selections, source_key=source_key, target_key=target_key)
    expected_payload_bytes = int(sequence_count) * 256 * 8 * 128 * 2
    expected_shape = [int(sequence_count) * 256, 8, 128]
    expected_source_root = str(Path(config["upstream"][source_key]))
    expected_target_root = str(Path(config["upstream"][target_key]))
    if expected_source_root == expected_target_root:
        raise ProtocolError(f"{split_name}: source and target activation roots are identical")
    _validate_activation_manifest(source_manifest, expected_source_root, model_inventories["source_32b"], split_name, sequence_count, token_sha, 64, calibration_manifest_sha, token_path, dataset_fingerprint)
    _validate_activation_manifest(target_manifest, expected_target_root, model_inventories["target_14b"], split_name, sequence_count, token_sha, len(ACTIVE_LAYERS), calibration_manifest_sha, token_path, dataset_fingerprint)
    payload_inventory: list[dict[str, Any]] = []
    metadata_inventory: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        metadata = strict_json(metadata_path)
        _expect(metadata.get("shape") == expected_shape, f"activation shape changed: {metadata_path}")
        _expect(type(metadata.get("nbytes")) is int and metadata.get("nbytes") == expected_payload_bytes, f"activation metadata size changed: {metadata_path}")
        provenance = metadata.get("provenance")
        _expect(isinstance(provenance, Mapping), f"activation provenance missing: {metadata_path}")
        _expect(provenance.get("token_sha256") == token_sha and provenance.get("split") == split_name, f"activation token/split provenance changed: {metadata_path}")
        _expect(type(provenance.get("sequences")) is int and provenance.get("sequences") == sequence_count and type(provenance.get("stride")) is int and provenance.get("stride") == 4, f"activation sequence provenance changed: {metadata_path}")
        component = next((name for name, directory in COMPONENT_DIR.items() if directory in metadata_path.parts), None)
        _expect(component in COMPONENTS, f"activation component path changed: {metadata_path}")
        layer_name = metadata_path.stem
        _expect(layer_name.startswith("layer_") and layer_name[6:].isdigit(), f"activation layer filename changed: {metadata_path}")
        expected_layer = int(layer_name[6:])
        _expect(provenance.get("layer") == expected_layer and provenance.get("cache_type") == COMPONENT_DIR[component], f"activation layer/component provenance changed: {metadata_path}")
        is_source = expected_source_root in str(metadata_path)
        model_inventory = model_inventories["source_32b" if is_source else "target_14b"]
        model_root = model_inventory["model_root"]
        expected_model_name = "qwen3_32b" if is_source else "qwen3_14b"
        _expect(provenance.get("model_path") == model_root and provenance.get("model_name") == expected_model_name, f"activation model provenance changed: {metadata_path}")
        _expect(provenance.get("model_config_sha256") == model_inventory["config"]["sha256"], f"activation config provenance changed: {metadata_path}")
        _expect(provenance.get("model_weight_index_sha256") == model_inventory["weight_index"]["sha256"], f"activation index provenance changed: {metadata_path}")
        _expect(provenance.get("model_tokenizer_sha256") == model_inventory["tokenizer"]["sha256"], f"activation tokenizer provenance changed: {metadata_path}")
        metadata_inventory.append(file_ref(metadata_path))
    for payload_path in payload_paths:
        _expect(payload_path.stat().st_size == expected_payload_bytes, f"activation payload size changed: {payload_path}")
        payload_ref = file_ref(payload_path)
        # Metadata is deliberately the only parsed companion; payloads are
        # hashed as opaque bytes, never indexed by a sequence id here.
        # ``Path.with_suffix('.json')`` would produce ``layer_000.bf16.json``;
        # the activation contract uses the stem ``layer_000.json``.
        if not payload_path.name.endswith(".bf16.bin"):
            raise ProtocolError(f"activation payload filename changed: {payload_path}")
        companion = payload_path.with_name(payload_path.name.removesuffix(".bf16.bin") + ".json")
        companion_json = strict_json(companion)
        _expect(companion_json.get("sha256") == payload_ref["sha256"], f"activation payload SHA mismatch: {payload_path}")
        payload_inventory.append(payload_ref)
    return {
        "payload_files": payload_inventory,
        "metadata_files": metadata_inventory,
        "payload_bytes_hashed": sum(int(row["nbytes"]) for row in payload_inventory),
        "metadata_bytes_hashed": sum(int(row["nbytes"]) for row in metadata_inventory),
        "split": split_name,
        "sequence_count": int(sequence_count),
        "observations_per_sequence": 256,
        "tuning_sequence_count": len(split["tuning_sequence_ids"]) if split_name == "heldout" else 0,
        "locked_sequence_count": split["locked_sequence_count"] if split_name == "heldout" else 0,
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "opaque_payload_hashing_does_not_index_locked_values": True,
    }


def validate_scientific_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash every full40 input and return a machine-checkable inventory."""

    validate_config_shape(config)
    assert_fit_eval_split_disjoint(config)
    model_inventories = {
        "source_32b": model_shard_inventory(config["upstream"]["source_model"], label="source Qwen3-32B"),
        "target_14b": model_shard_inventory(config["upstream"]["target_model"], label="target Qwen3-14B"),
    }
    _expect(model_inventories["source_32b"]["architecture"]["num_hidden_layers"] == 64, "source model depth changed")
    _expect(model_inventories["target_14b"]["architecture"]["num_hidden_layers"] == 40, "target model depth changed")
    refs = {key: _verify_config_ref(config, key) for key in ("k20_manifest", "k20_manifest_marker", "k20_selection", "k20_fit_definition", "shrinkage_manifest", "shrinkage_manifest_marker", "split_manifest", "calibration_manifest", "train_tokens", "heldout_tokens", "heldout_source_activation_manifest", "heldout_target_activation_manifest", "fit_source_activation_manifest", "fit_target_activation_manifest")}
    k20_manifest = strict_json(refs["k20_manifest"])
    shrink_manifest = strict_json(refs["shrinkage_manifest"])
    _expect(file_ref(refs["k20_manifest"])["sha256"] == config["upstream"]["k20_manifest"]["sha256"], "k20 manifest binding changed")
    _expect(file_ref(refs["shrinkage_manifest"])["sha256"] == config["upstream"]["shrinkage_manifest"]["sha256"], "shrinkage manifest binding changed")
    k_layers = _manifest_layers(k20_manifest, label="k20")
    s_layers = _manifest_layers(shrink_manifest, label="shrinkage")
    selected_rows = _validate_complete_selection_snapshot(config, k20_manifest, shrink_manifest)
    selections = {target: selected_rows[target] for target in ACTIVE_LAYERS}
    _expect(set(v for row in selections.values() for v in row) == set(range(64)), "full40 selection does not cover all source layers")
    split = _load_split(config)
    train_split = _load_train_split(config)
    calibration = _load_calibration_binding(config)
    token_ref = config["upstream"]["heldout_tokens"]
    heldout_source_manifest = strict_json(refs["heldout_source_activation_manifest"])
    heldout_target_manifest = strict_json(refs["heldout_target_activation_manifest"])
    fit_source_manifest = strict_json(refs["fit_source_activation_manifest"])
    fit_target_manifest = strict_json(refs["fit_target_activation_manifest"])
    activation = _validate_activation_inventory(
        config, selections, split, str(token_ref["sha256"]), model_inventories,
        split_name="heldout", sequence_count=50, source_key="source_activations", target_key="target_activations",
        source_manifest=heldout_source_manifest, target_manifest=heldout_target_manifest,
        calibration_manifest_sha=str(calibration["sha256"]), token_path=str(token_ref["path"]), dataset_fingerprint=str(calibration["dataset_fingerprint"]),
    )
    fit_activation = _validate_activation_inventory(
        config, selections, train_split, str(config["upstream"]["train_tokens"]["sha256"]), model_inventories,
        split_name="train", sequence_count=500, source_key="fit_source_activations", target_key="fit_target_activations",
        source_manifest=fit_source_manifest, target_manifest=fit_target_manifest,
        calibration_manifest_sha=str(calibration["sha256"]), token_path=str(config["upstream"]["train_tokens"]["path"]), dataset_fingerprint=str(calibration["dataset_fingerprint"]),
    )
    mapper, comparator_alpha = _mapper_bindings(config, k20_manifest, shrink_manifest)
    closure = source_byte_closure()
    inventory = {
        "models": model_inventories,
        "mapper_files": mapper,
        "comparator_alpha": comparator_alpha,
        "activation": activation,
        "fit_activation": fit_activation,
        "calibration": calibration,
        "upstream_refs": {key: file_ref(path) for key, path in refs.items()},
        "source_byte_closure_digest": closure["digest"],
        "fit_eval_split_binding": {
            "fit_split": "train", "fit_sequence_count": 500,
            "eval_split": "heldout_tuning", "eval_sequence_count": 40,
            "fit_ids_are_namespaced_from_eval_ids": True,
        },
    }
    return {
        "status": "pass",
        "protocol": PROTOCOL,
        "probe_target_layers": list(PROBE_LAYERS),
        "full_target_layers_registered": list(FULL_LAYERS),
        "fit_split": "train",
        "fit_sequence_count": 500,
        "fit_observations": 500 * 256,
        "eval_split": "heldout_tuning",
        "eval_sequence_count": 40,
        "eval_observations": 40 * 256,
        "fit_eval_ids_disjoint": True,
        "shrinkage_comparator_status": "validation_tuned_on_heldout_tuning40_exploratory_only",
        "source_selection_by_target": {str(k): list(v) for k, v in selections.items()},
        "split": split,
        "fit_split_info": train_split,
        "mapper_files": mapper,
        "comparator_alpha": comparator_alpha,
        "activation": activation,
        "fit_activation": fit_activation,
        "upstream_refs": {key: file_ref(path) for key, path in refs.items()},
        "models": model_inventories,
        "source_byte_closure": closure,
        "input_inventory_sha256": canonical_sha256(inventory),
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "answers_or_outcomes_read": False,
        "fit_eval_split_binding": {
            "fit_split": "train", "fit_sequence_count": 500,
            "eval_split": "heldout_tuning", "eval_sequence_count": 40,
            "fit_ids_are_namespaced_from_eval_ids": True,
        },
    }


def tuning_ids(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return only the permitted 40 tuning IDs; locked IDs never leave _load_split."""

    return tuple(_load_split(config)["tuning_sequence_ids"])


def assert_tuning_ids(ids: Sequence[int], config: Mapping[str, Any]) -> None:
    allowed = set(tuning_ids(config))
    if any(type(v) is not int or v not in allowed for v in ids):
        raise ProtocolError("an activation access requested a non-tuning sequence")


def fit_head_local_numpy(x: Any, y: Any, ridge_lambda: float = 0.01) -> tuple[Any, Any, dict[str, Any]]:
    """Fit one head with centered, unweighted ridge (CPU oracle/test helper)."""

    import numpy as np

    if type(ridge_lambda) is not float or ridge_lambda != 0.01:
        raise ProtocolError("head-local ridge lambda must be exactly 0.01")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[1] != 2560 or y.shape[1] != 128 or x.shape[0] == 0:
        raise ProtocolError("head-local training shapes must be [N,2560] and [N,128]")
    mean_x = x.mean(axis=0)
    mean_y = y.mean(axis=0)
    xc = x - mean_x
    yc = y - mean_y
    gram = xc.T @ xc
    gram.flat[:: gram.shape[0] + 1] += ridge_lambda
    cross = xc.T @ yc
    try:
        chol = np.linalg.cholesky(gram)
        weight = np.linalg.solve(chol.T, np.linalg.solve(chol, cross))
    except np.linalg.LinAlgError as exc:
        raise ProtocolError("head-local ridge Cholesky failed") from exc
    bias = mean_y - mean_x @ weight
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ProtocolError("head-local ridge produced non-finite parameters")
    return weight.astype(np.float32), bias.astype(np.float32), {
        "n_observations": int(x.shape[0]),
        "source_features": 2560,
        "target_features": 128,
        "ridge_lambda": 0.01,
        "weighted_loss": False,
        "regularize_bias": False,
        "solver": "cholesky",
    }


def predict_head_local(x: Any, weight: Any, bias: Any) -> Any:
    import numpy as np

    x = np.asarray(x)
    weight = np.asarray(weight)
    bias = np.asarray(bias)
    if x.ndim != 2 or x.shape[1] != 2560 or weight.shape != (2560, 128) or bias.shape != (128,):
        raise ProtocolError("head-local prediction shape mismatch")
    return x @ weight + bias


def activation_metrics(target: Any, prediction: Any) -> dict[str, Any]:
    """Return SSE and global/uniform-channel R² for [N,8,128] arrays."""

    import numpy as np

    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 3 or y.shape[1:] != (8, 128):
        raise ProtocolError("metric arrays must both have shape [N,8,128]")
    residual = p - y
    sse_channels = np.sum(residual * residual, axis=0)
    centered = y - np.mean(y, axis=0, keepdims=True)
    sst_channels = np.sum(centered * centered, axis=0)
    sse = float(np.sum(sse_channels))
    sst = float(np.sum(sst_channels))
    r2_channels = np.zeros_like(sst_channels, dtype=np.float64)
    np.divide(sse_channels, sst_channels, out=r2_channels, where=sst_channels > 0.0)
    r2_channels = np.where(sst_channels > 0.0, 1.0 - r2_channels, np.where(sse_channels == 0.0, 1.0, 0.0))
    return {
        "sse": sse,
        "sst": sst,
        "global_r2": float(1.0 - sse / sst) if sst > 0.0 else (1.0 if sse == 0.0 else 0.0),
        "uniform_channel_r2": float(np.mean(r2_channels)),
        "channel_sse": sse_channels.tolist(),
        "channel_sst": sst_channels.tolist(),
        "observations": int(y.shape[0]),
    }


def compare_metric_rows(target: Any, predictions: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): activation_metrics(target, value) for name, value in predictions.items()}


def block_diagonal_head_weights(head_weights: Any, head_bias: Any) -> tuple[Any, Any]:
    """Embed [8,2560,128] head-local maps in the joint [20480,1024] layout."""

    import numpy as np

    w = np.asarray(head_weights)
    b = np.asarray(head_bias)
    if w.shape != (8, 2560, 128) or b.shape != (8, 128):
        raise ProtocolError("head-local parameter layout changed")
    joint = np.zeros((20480, 1024), dtype=w.dtype)
    for head in range(8):
        source_indices = [layer * 8 * 128 + head * 128 + channel for layer in range(20) for channel in range(128)]
        target_indices = [head * 128 + channel for channel in range(128)]
        joint[np.ix_(source_indices, target_indices)] = w[head]
    return joint, b.reshape(-1).copy()


def _mmap_activation(path: Path, sequence_count: int) -> Any:
    """Open a BF16 payload without interpreting any row until requested."""

    import torch

    expected_bytes = int(sequence_count) * 256 * 8 * 128 * 2
    _expect(path.stat().st_size == expected_bytes, f"activation payload size changed: {path}")
    raw = torch.from_file(str(path), shared=False, size=expected_bytes // 2, dtype=torch.uint16)
    return raw.view(torch.bfloat16).reshape(int(sequence_count), 256, 8, 128)


def _split_ids(config: Mapping[str, Any], split: str) -> tuple[int, ...]:
    """Return the immutable ID namespace for one fit/evaluation split."""

    if split == "train":
        return tuple(range(500))
    if split == "heldout_tuning":
        return tuple(_load_split(config)["tuning_sequence_ids"])
    raise ProtocolError(f"unknown activation split: {split}")


def assert_split_ids(ids: Sequence[int], config: Mapping[str, Any], split: str) -> None:
    allowed = set(_split_ids(config, split))
    if any(type(v) is not int or v not in allowed for v in ids):
        raise ProtocolError(f"an activation access requested an ID outside {split}")
    if split == "train" and any(v >= 500 for v in ids):
        raise ProtocolError("train activation ID exceeds 500-sequence fit split")


def assert_fit_eval_split_disjoint(config: Mapping[str, Any]) -> None:
    """Require distinct token files/roots for fit and heldout evaluation."""

    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("fit_split") != "train" or protocol.get("eval_split") != "heldout_tuning":
        raise ProtocolError("fit/eval split labels are not the frozen train/heldout pair")
    up = config.get("upstream")
    if not isinstance(up, Mapping):
        raise ProtocolError("upstream split roots are missing")
    activation_roots = [up.get(name) for name in ("fit_source_activations", "fit_target_activations", "source_activations", "target_activations")]
    if any(type(value) is not str or not Path(value).is_absolute() for value in activation_roots):
        raise ProtocolError("fit/evaluation activation roots must be absolute strings")
    if len(set(activation_roots)) != 4:
        raise ProtocolError("fit and evaluation activation roots must be distinct")
    train_ref = up.get("train_tokens")
    held_ref = up.get("heldout_tokens")
    if not isinstance(train_ref, Mapping) or not isinstance(held_ref, Mapping) or train_ref.get("path") == held_ref.get("path") or train_ref.get("sha256") == held_ref.get("sha256"):
        raise ProtocolError("fit and evaluation token refs must be distinct")
    fit_ids = tuple(range(500))
    eval_ids = tuple(_load_split(config)["tuning_sequence_ids"])
    if len(fit_ids) != 500 or len(eval_ids) != 40:
        raise ProtocolError("fit/evaluation ID cardinality changed")
    # Raw integer IDs overlap by design because each activation file has its
    # own local sequence axis.  The only valid identity is (split, local_id).
    fit_namespace = {("train", value) for value in fit_ids}
    eval_namespace = {("heldout_tuning", value) for value in eval_ids}
    if not fit_namespace.isdisjoint(eval_namespace):
        raise ProtocolError("fit/evaluation namespaced IDs overlap")


def _activation_batch(
    config: Mapping[str, Any],
    target_layer: int,
    component: str,
    ids: Sequence[int],
    selection: Sequence[int],
    split: str,
) -> tuple[Any, Any]:
    """Materialize one split-specific batch in [tokens,features] form."""

    assert_split_ids(ids, config, split)
    import torch

    if split == "train":
        source_base = config["upstream"]["fit_source_activations"]
        target_base = config["upstream"]["fit_target_activations"]
        sequence_count = 500
    elif split == "heldout_tuning":
        source_base = config["upstream"]["source_activations"]
        target_base = config["upstream"]["target_activations"]
        sequence_count = 50
    else:
        raise ProtocolError(f"unknown activation split: {split}")
    source_root = Path(source_base) / COMPONENT_DIR[component]
    target_root = Path(target_base) / COMPONENT_DIR[component]
    source = [_mmap_activation(source_root / f"layer_{int(layer):03d}.bf16.bin", sequence_count) for layer in selection]
    target = _mmap_activation(target_root / f"layer_{int(target_layer):03d}.bf16.bin", sequence_count)
    # The list comprehension is intentionally over ``ids`` only.  No locked
    # sequence id is available to this function.
    x = torch.stack([source_rank[list(ids)] for source_rank in source], dim=1)
    y = target[list(ids)]
    return x, y


def _fit_head_local_torch(config: Mapping[str, Any], target_layer: int, component: str, selection: Sequence[int], ids: Sequence[int], device: str, split: str = "train") -> tuple[Any, Any, dict[str, Any]]:
    """Two-pass streaming GPU fit for all eight heads of one layer/component."""

    import torch

    assert_split_ids(ids, config, split)
    dev = torch.device(device)
    chunk = int(config["runtime"]["chunk_sequences"])
    means_x = torch.zeros((8, 2560), dtype=torch.float32, device=dev)
    means_y = torch.zeros((8, 128), dtype=torch.float32, device=dev)
    count = 0
    for first in range(0, len(ids), chunk):
        batch_ids = list(ids[first:first + chunk])
        x_cpu, y_cpu = _activation_batch(config, target_layer, component, batch_ids, selection, split)
        # ``_activation_batch`` stacks source layers first, yielding
        # ``[20, batch, position, head, dim]``.  Move the batch/position
        # axes in front before flattening; a plain reshape here would pair
        # rows from different sequences with the wrong source layer and
        # silently fit a different (incorrect) feature matrix.
        x = x_cpu.to(dev, dtype=torch.float32).permute(1, 2, 0, 3, 4).reshape(len(batch_ids) * 256, 20, 8, 128)
        y = y_cpu.to(dev, dtype=torch.float32).reshape(len(batch_ids) * 256, 8, 128)
        n = x.shape[0]
        means_x += x.permute(2, 0, 1, 3).reshape(8, n, 2560).sum(dim=1)
        means_y += y.permute(1, 0, 2).reshape(8, n, 128).sum(dim=1)
        count += n
        del x_cpu, y_cpu, x, y
    if count <= 0:
        raise ProtocolError("no tuning observations")
    means_x /= float(count)
    means_y /= float(count)
    gram = torch.zeros((8, 2560, 2560), dtype=torch.float32, device=dev)
    cross = torch.zeros((8, 2560, 128), dtype=torch.float32, device=dev)
    seen = 0
    for first in range(0, len(ids), chunk):
        batch_ids = list(ids[first:first + chunk])
        x_cpu, y_cpu = _activation_batch(config, target_layer, component, batch_ids, selection, split)
        x = x_cpu.to(dev, dtype=torch.float32).permute(1, 2, 0, 3, 4).reshape(len(batch_ids) * 256, 20, 8, 128)
        y = y_cpu.to(dev, dtype=torch.float32).reshape(len(batch_ids) * 256, 8, 128)
        n = x.shape[0]
        xh = x.permute(2, 0, 1, 3).reshape(8, n, 2560) - means_x[:, None, :]
        yh = y.permute(1, 0, 2).reshape(8, n, 128) - means_y[:, None, :]
        gram += torch.matmul(xh.transpose(1, 2), xh)
        cross += torch.matmul(xh.transpose(1, 2), yh)
        seen += n
        del x_cpu, y_cpu, x, y, xh, yh
    _expect(seen == count, "ridge passes observed different counts")
    eye = torch.eye(2560, dtype=torch.float32, device=dev).expand(8, -1, -1)
    system = gram + 0.01 * eye
    factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
    if int(torch.max(info).item()) != 0:
        raise ProtocolError("head-local Cholesky failed")
    weights = torch.cholesky_solve(cross, factor)
    biases = means_y - torch.matmul(means_x[:, None, :], weights).squeeze(1)
    if not torch.isfinite(weights).all() or not torch.isfinite(biases).all():
        raise ProtocolError("head-local fit produced non-finite values")
    return weights.detach().cpu(), biases.detach().cpu(), {
        "n_observations": count,
        "ridge_lambda": 0.01,
        "solver": "cholesky",
        "weighted_loss": False,
        "regularize_bias": False,
        "source_features_per_head": 2560,
        "target_features_per_head": 128,
        "cholesky_info_max": int(torch.max(info).item()),
    }


def _driver_state() -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    gpus = []
    for line in result.stdout.splitlines():
        fields = [v.strip() for v in line.split(",")]
        if len(fields) != 3:
            raise ProtocolError("nvidia-smi GPU output changed")
        gpus.append({"index": int(fields[0]), "memory_used_mib": int(fields[1]), "utilization_percent": int(fields[2])})
    proc = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True)
    return {"gpus": gpus, "compute_processes": [line for line in proc.stdout.splitlines() if line.strip()]}


def idle_gate(config: Mapping[str, Any], run_root: Path) -> Any:
    """Acquire the shared eight-GPU lock and require consecutive idle samples."""

    lock_path = Path(config["runtime"]["global_gpu_lock"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ProtocolError("global GPU lock is held") from exc
    samples = []
    try:
        for index in range(int(config["runtime"]["idle_gate_samples"])):
            state = _driver_state()
            passed = len(state["gpus"]) == 8 and not state["compute_processes"] and max((row["memory_used_mib"] for row in state["gpus"]), default=10**9) <= int(config["runtime"]["idle_gate_max_memory_mib"])
            samples.append({"index": index, "passed": passed, **state})
            if not passed:
                raise ProtocolError("eight-GPU idle gate failed")
            if index + 1 < int(config["runtime"]["idle_gate_samples"]):
                time.sleep(float(config["runtime"]["idle_gate_interval_seconds"]))
        atomic_json(run_root / "idle_gate.json", {"status": "pass", "samples": samples})
        return handle
    except BaseException:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise


def _fresh_bootstrap() -> str:
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or sys.dont_write_bytecode is not True:
        raise ProtocolError("PYTHONDONTWRITEBYTECODE=1 is required")
    prefix = os.environ.get("PYTHONPYCACHEPREFIX")
    if not isinstance(prefix, str) or not prefix or not os.path.isabs(prefix) or os.path.lexists(prefix):
        raise ProtocolError("PYTHONPYCACHEPREFIX must be an initially absent absolute path")
    if type(sys.pycache_prefix) is not str or sys.pycache_prefix != prefix:
        raise ProtocolError("sys.pycache_prefix does not match PYTHONPYCACHEPREFIX")
    for parent in Path(prefix).parents:
        if parent.is_symlink() or not parent.is_dir():
            raise ProtocolError("PYTHONPYCACHEPREFIX parent is missing/non-directory/symlink")
    return prefix


def _cpu_preflight_environment(config: Mapping[str, Any]) -> dict[str, Any]:
    """Assert the preflight process cannot accidentally touch CUDA."""

    expected_python_raw = config["runtime"]["required_python"]
    if type(expected_python_raw) is not str or not Path(expected_python_raw).is_absolute():
        raise ProtocolError("configured Python path is malformed")
    # Keep both the invocation spelling and the resolved target.  A generic
    # `/usr/bin/python` that happens to be the same binary is not accepted:
    # the venv path itself is part of the reproducibility contract.
    if sys.executable != expected_python_raw:
        raise ProtocolError(f"preflight must use the exact configured Python path: {sys.executable}")
    expected_python = Path(expected_python_raw)
    # The venv launcher is intentionally a symlink to its pinned interpreter;
    # reject symlinked *parents* but allow this final, explicitly bound link.
    _lstat_component_chain(expected_python.parent, label="configured Python parent", allow_missing_tail=False)
    if not os.path.lexists(expected_python):
        raise ProtocolError("configured Python executable is missing")
    resolved_python = expected_python.resolve(strict=True)
    if not stat.S_ISREG(resolved_python.lstat().st_mode) or resolved_python != Path(sys.executable).resolve(strict=True) or not os.access(resolved_python, os.X_OK):
        raise ProtocolError("configured Python executable does not match the running interpreter")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("CPU preflight requires CUDA_VISIBLE_DEVICES='' (empty)")
    if "torch" in sys.modules or "transformers" in sys.modules or "safetensors" in sys.modules:
        raise ProtocolError("heavy model/CUDA modules were imported before CPU preflight")
    prefix = _fresh_bootstrap()
    return {
        "executable": sys.executable,
        "resolved_executable": str(Path(sys.executable).resolve(strict=True)),
        "configured_executable": expected_python_raw,
        "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_visible_devices": "",
        "pythondontwritebytecode": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "pythonpycacheprefix": prefix,
        "sys_dont_write_bytecode": bool(sys.dont_write_bytecode),
        "torch_imported": False,
        "transformers_imported": False,
        "safetensors_imported": False,
    }


def _relative_output(config: Mapping[str, Any], key: str) -> Path:
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping) or type(outputs.get(key)) is not str:
        raise ProtocolError(f"output {key} is missing")
    relative = Path(outputs[key])
    if relative.is_absolute() or ".." in relative.parts:
        raise ProtocolError(f"output {key} must be a relative path inside this experiment")
    return _assert_inside(ROOT, ROOT / relative, label=f"output {key}")


def _config_file_ref() -> dict[str, Any]:
    return file_ref(_lstat_component_chain(ROOT / "config.json", label="config", allow_missing_tail=False))


def _validate_prior_probe6_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind dir30 as read-only mechanistic context, never as QA evidence."""

    binding = config.get("prior_probe6")
    if not isinstance(binding, Mapping):
        raise ProtocolError("prior probe6 binding is missing")
    if binding.get("role") != "read_only_mechanistic_prior_not_qa_proof":
        raise ProtocolError("prior probe6 role is not read-only/non-QA")
    raw_root = binding.get("experiment_root")
    if type(raw_root) is not str or not Path(raw_root).is_absolute():
        raise ProtocolError("prior probe6 root is not absolute")
    prior_root = _lstat_component_chain(raw_root, label="prior probe6 root", allow_missing_tail=False)
    if prior_root.resolve(strict=True) != prior_root or prior_root.name != "30_matched_head_mapper_probe_v2":
        raise ProtocolError("prior probe6 root is not canonical dir30")
    refs: dict[str, dict[str, Any]] = {}
    expected_paths = {
        "config": Path("config.json"),
        "preflight": Path("artifacts/preflight/input_preflight.json"),
        "stop": Path("artifacts/preformal/PRE_FORMAL_STOP_v2.json"),
    }
    for name in ("config", "preflight", "stop"):
        raw = binding.get(name)
        if not isinstance(raw, Mapping):
            raise ProtocolError(f"prior probe6 {name} binding is malformed")
        path = _strict_ref(raw, label=f"prior probe6 {name}")
        try:
            relative_path = path.relative_to(prior_root)
        except ValueError as exc:
            raise ProtocolError(f"prior probe6 {name} path escapes prior root") from exc
        if relative_path != expected_paths[name]:
            raise ProtocolError(f"prior probe6 {name} path changed")
        if name != "config":
            verify_sidecar(path)
        value = strict_json(path)
        if name == "config":
            if value.get("experiment_id") != "qwen3_32b_to_14b_cachebridge_head_local_probe_v2":
                raise ProtocolError("prior probe6 config identity changed")
        elif name == "preflight":
            if value.get("status") != "passed_cpu_only_preflight" or value.get("gpu_used") is not False or value.get("models_loaded") is not False:
                raise ProtocolError("prior probe6 preflight is not CPU-only")
        else:
            if value.get("status") != "PRE_FORMAL_STOP" or value.get("execute_authorized") is not False:
                raise ProtocolError("prior probe6 STOP is not an unexecuted snapshot")
            # Explicitly reject using a future result/accuracy artifact as a
            # promotion signal; this sibling only records the mechanistic prior.
            if value.get("answers_or_outcomes_read") is not False or value.get("locked_rows_indexed") is not False:
                raise ProtocolError("prior probe6 crossed the locked/QA boundary")
        refs[name] = dict(raw)
    return {"role": binding["role"], "experiment_root": str(prior_root), **refs}


def _preflight_definition(config: Mapping[str, Any], inputs: Mapping[str, Any], closure: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "experiment_id": config["experiment_id"],
        "direction": config["direction"],
        "method": config["method"],
        "probe_target_layers": list(PROBE_LAYERS),
        "full_target_layers_registered": list(FULL_LAYERS),
        "inputs": inputs,
        "source_byte_closure": closure,
        "prior_probe6": _validate_prior_probe6_snapshot(config),
        "config": _config_file_ref(),
        "environment": environment,
        "execution_policy": {
            "cpu_preflight_only": True,
            "gpu_execution_performed": False,
            "models_loaded": False,
            "locked_activation_evaluations": 0,
            "locked_rows_indexed": False,
            "answers_or_outcomes_read": False,
            "downstream_accuracy_claim_permitted": False,
        },
    }
    return {**body, "request_fingerprint": canonical_sha256(body)}


def preflight(config_path: Path | None = None) -> dict[str, Any]:
    """Run the complete CPU input gate and publish immutable PF/STOP receipts.

    This function performs no model loading and no CUDA API call.  A second
    invocation is idempotent only if every output byte is identical; otherwise
    it fails closed instead of silently replacing a prior snapshot.
    """

    if config_path is None:
        config_path = ROOT / "config.json"
    # Shape validation precedes environment probing, so malformed configs
    # cannot influence the process or output paths.
    config = load_config(Path(config_path))
    prior_probe6 = _validate_prior_probe6_snapshot(config)
    environment = _cpu_preflight_environment(config)
    inputs = validate_scientific_inputs(config)
    closure = inputs["source_byte_closure"]
    definition = _preflight_definition(config, inputs, closure, environment)
    preflight_value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "experiment_id": config["experiment_id"],
        "status": "passed_cpu_only_preflight",
        "gpu_used": False,
        "cuda_api_called": False,
        "models_loaded": False,
        "request_fingerprint": definition["request_fingerprint"],
        "definition_digest": canonical_sha256(definition),
        "definition": definition,
        "prior_probe6": prior_probe6,
        "input_inventory_sha256": inputs["input_inventory_sha256"],
        "source_byte_closure_digest": closure["digest"],
        "source_byte_closure_file_count": closure["file_count"],
        "model_inventory_digests": {key: value["digest"] for key, value in inputs["models"].items()},
        "fit_split": "train",
        "fit_sequence_count": 500,
        "fit_observations": 500 * 256,
        "eval_split": "heldout_tuning",
        "eval_sequence_count": 40,
        "eval_observations": 40 * 256,
        "fit_eval_ids_disjoint": True,
        "shrinkage_comparator_status": "validation_tuned_on_heldout_tuning40_exploratory_only",
        "probe_target_layers": list(PROBE_LAYERS),
        "tuning_sequence_count": 40,
        "locked_sequence_count": 10,
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "answers_or_outcomes_read": False,
        "execution_boundary": "CPU preflight only; no mapper tensors are fitted and no downstream outcomes are opened",
    }
    preflight_path = _relative_output(config, "preflight_receipt")
    preflight_ref = immutable_json(preflight_path, preflight_value)
    preflight_sidecar = write_sidecar(preflight_path)
    stop_value = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "experiment_id": config["experiment_id"],
        "status": "PRE_FORMAL_STOP",
        "stop_mode": "CPU_PREFLIGHT_COMPLETE_WAITING_FOR_OPERATOR_GPU_GATE",
        "request_fingerprint": definition["request_fingerprint"],
        "definition_digest": canonical_sha256(definition),
        "prior_probe6": prior_probe6,
        "config": _config_file_ref(),
        "preflight": preflight_ref,
        "preflight_sidecar": preflight_sidecar,
        "source_byte_closure_digest": closure["digest"],
        "source_byte_closure_file_count": closure["file_count"],
        "input_inventory_sha256": inputs["input_inventory_sha256"],
        "fit_split": "train",
        "fit_sequence_count": 500,
        "fit_observations": 500 * 256,
        "eval_split": "heldout_tuning",
        "eval_sequence_count": 40,
        "eval_observations": 40 * 256,
        "fit_eval_ids_disjoint": True,
        "shrinkage_comparator_status": "validation_tuned_on_heldout_tuning40_exploratory_only",
        "gpu_used": False,
        "cuda_api_called": False,
        "models_loaded": False,
        "execute_authorized": False,
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "answers_or_outcomes_read": False,
        "claim_boundary": config["claim_boundary"],
        "environment": environment,
        "only_future_execute_command": (
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONDONTWRITEBYTECODE=1 "
            "PYTHONPYCACHEPREFIX=<fresh-absent-absolute-leaf> "
            f"{config['runtime']['required_python']} {ROOT / 'scripts/run.py'} "
            f"--config {ROOT / 'config.json'} --stage full40 --execute"
        ),
        "unfinished_boundaries": [
            "full40 mapper fitting and activation metric comparison",
            "all GPU/model loading and downstream accuracy evaluation",
            "locked10 activation evaluation (permanently prohibited in this probe)",
        ],
    }
    stop_path = _relative_output(config, "stop_receipt")
    stop_ref = immutable_json(stop_path, stop_value)
    stop_sidecar = write_sidecar(stop_path)
    return {
        "config": config,
        "definition": definition,
        "preflight": preflight_value,
        "preflight_ref": preflight_ref,
        "preflight_sidecar": preflight_sidecar,
        "stop_ref": stop_ref,
        "stop_sidecar": stop_sidecar,
    }


def _validate_frozen_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    """Revalidate PF/STOP and source bytes immediately before any GPU path."""

    stop_path = _relative_output(config, "stop_receipt")
    stop = strict_json(stop_path)
    verify_sidecar(stop_path)
    if stop.get("status") != "PRE_FORMAL_STOP" or stop.get("protocol") != PROTOCOL or stop.get("experiment_id") != config.get("experiment_id") or stop.get("execute_authorized") is not False:
        raise ProtocolError("STOP receipt is not an unexecuted CPU snapshot")
    preflight_ref = stop.get("preflight")
    if not isinstance(preflight_ref, Mapping):
        raise ProtocolError("STOP preflight binding is malformed")
    preflight_path = _strict_ref(preflight_ref, label="STOP preflight")
    verify_sidecar(preflight_path)
    preflight = strict_json(preflight_path)
    if preflight.get("status") != "passed_cpu_only_preflight" or preflight.get("protocol") != PROTOCOL or preflight.get("experiment_id") != config.get("experiment_id") or preflight.get("gpu_used") is not False or preflight.get("models_loaded") is not False:
        raise ProtocolError("preflight receipt is not CPU-only")
    expected_split_fields = {
        "fit_split": "train", "fit_sequence_count": 500, "fit_observations": 500 * 256,
        "eval_split": "heldout_tuning", "eval_sequence_count": 40, "eval_observations": 40 * 256,
        "fit_eval_ids_disjoint": True,
        "shrinkage_comparator_status": "validation_tuned_on_heldout_tuning40_exploratory_only",
    }
    if any(preflight.get(key) != value for key, value in expected_split_fields.items()):
        raise ProtocolError("preflight fit/eval split evidence changed")
    if stop.get("source_byte_closure_digest") != preflight.get("source_byte_closure_digest"):
        raise ProtocolError("STOP/preflight source closure differs")
    if any(stop.get(key) != value for key, value in expected_split_fields.items()):
        raise ProtocolError("STOP fit/eval split evidence changed")
    definition = preflight.get("definition")
    if not isinstance(definition, Mapping):
        raise ProtocolError("preflight definition is missing")
    closure = definition.get("source_byte_closure")
    if not isinstance(closure, Mapping):
        raise ProtocolError("preflight source closure is missing")
    validate_source_closure(closure)
    if file_ref(ROOT / "config.json") != stop.get("config"):
        raise ProtocolError("config changed after PRE_FORMAL_STOP")
    _validate_prior_probe6_snapshot(config)
    return stop


def _stage_layers(stage: str) -> tuple[int, ...]:
    if stage == "full40":
        return ACTIVE_LAYERS
    raise ProtocolError("full40 is the only executable stage in this sibling")


def _finite_float(value: Any, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ProtocolError(f"{label}: expected finite JSON float")
    return value


def _baseline_payload(pre: Mapping[str, Any], method: str, target_layer: int) -> Path:
    rows = [row for row in pre.get("mapper_files", []) if row.get("method") == method and row.get("target_layer") == target_layer]
    if len(rows) != 1 or not isinstance(rows[0].get("payload"), Mapping):
        raise ProtocolError(f"missing frozen {method} mapper for layer {target_layer}")
    return _strict_ref(rows[0]["payload"], label=f"{method} layer {target_layer} payload")


def _evaluate_full_gate_torch(
    config: Mapping[str, Any],
    pre: Mapping[str, Any],
    target_layer: int,
    component: str,
    selection: Sequence[int],
    ids: Sequence[int],
    candidate_weight: Any,
    candidate_bias: Any,
    device: str,
) -> dict[str, Any]:
    """Compute candidate and both baseline SSEs on the tuning rows only."""

    import torch
    from safetensors.torch import safe_open

    assert_split_ids(ids, config, "heldout_tuning")
    dev = torch.device(device)
    baseline_tensors: dict[str, tuple[Any, Any]] = {}
    key = "key" if component == "key" else "value"
    for method in ("nvidia_k20", "shrinkage"):
        path = _baseline_payload(pre, method, target_layer)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = set(handle.keys())
            expected_names = {"key_weight", "key_bias", "value_weight", "value_bias"}
            if names != expected_names:
                raise ProtocolError(f"{method} mapper tensor keys changed")
            bw = handle.get_tensor(f"{key}_weight")
            bb = handle.get_tensor(f"{key}_bias")
        if tuple(bw.shape) != (20480, 1024) or tuple(bb.shape) != (1024,):
            raise ProtocolError(f"{method} mapper tensor shape changed")
        baseline_tensors[method] = (bw.to(dev, dtype=torch.float32), bb.to(dev, dtype=torch.float32))
    cw = candidate_weight.to(dev, dtype=torch.float32)
    cb = candidate_bias.to(dev, dtype=torch.float32)
    if tuple(cw.shape) != (8, 2560, 128) or tuple(cb.shape) != (8, 128):
        raise ProtocolError("candidate head-local tensor shape changed")
    totals = {"cachebridge_head_local": torch.zeros((), dtype=torch.float64, device=dev), "nvidia_k20": torch.zeros((), dtype=torch.float64, device=dev), "shrinkage": torch.zeros((), dtype=torch.float64, device=dev)}
    observations = 0
    chunk = _strict_int(config["runtime"]["chunk_sequences"], label="runtime chunk", minimum=1)
    for first in range(0, len(ids), chunk):
        batch_ids = list(ids[first:first + chunk])
        x_cpu, y_cpu = _activation_batch(config, target_layer, component, batch_ids, selection, "heldout_tuning")
        n = len(batch_ids) * 256
        x = x_cpu.to(dev, dtype=torch.float32).permute(1, 2, 0, 3, 4).reshape(n, 20, 8, 128)
        y = y_cpu.to(dev, dtype=torch.float32).reshape(n, 8, 128)
        xh = x.permute(2, 0, 1, 3).reshape(8, n, 2560)
        yh = y.permute(1, 0, 2)
        candidate = torch.bmm(xh, cw) + cb[:, None, :]
        totals["cachebridge_head_local"] += torch.sum((candidate - yh).to(torch.float64) ** 2)
        xjoint = x.reshape(n, 20480)
        for method, (bw, bb) in baseline_tensors.items():
            prediction = (xjoint @ bw + bb).reshape(n, 8, 128).permute(1, 0, 2)
            totals[method] += torch.sum((prediction - yh).to(torch.float64) ** 2)
        observations += n
        del x_cpu, y_cpu, x, y, xh, yh, candidate, xjoint
    result = {
        "target_layer": int(target_layer),
        "component": component,
        "eval_split": "heldout_tuning",
        "observations": int(observations),
        "candidate_sse": float(totals["cachebridge_head_local"].item()),
        "nvidia_k20_sse": float(totals["nvidia_k20"].item()),
        "shrinkage_sse": float(totals["shrinkage"].item()),
    }
    for name in ("candidate_sse", "nvidia_k20_sse", "shrinkage_sse"):
        _finite_float(result[name], label=f"promotion {name}")
    result["strictly_below_both"] = bool(result["candidate_sse"] < result["nvidia_k20_sse"] and result["candidate_sse"] < result["shrinkage_sse"])
    return result


def _validate_full_gate_rows(rows: Any) -> bool:
    """Validate the exact 40-layer × 2-component promotion grid."""

    if not isinstance(rows, list) or len(rows) != len(ACTIVE_LAYERS) * len(COMPONENTS):
        raise ProtocolError("full40 promotion gate must contain exactly 80 layer/component rows")
    seen: set[tuple[int, str]] = set()
    passed = True
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"target_layer", "component", "eval_split", "observations", "candidate_sse", "nvidia_k20_sse", "shrinkage_sse", "strictly_below_both"}:
            raise ProtocolError("full40 promotion row schema changed")
        layer = _strict_int(row["target_layer"], label="promotion target layer", minimum=0)
        component = row["component"]
        if component not in COMPONENTS or (layer, component) in seen or layer not in ACTIVE_LAYERS:
            raise ProtocolError("full40 promotion row identity changed")
        if row.get("eval_split") != "heldout_tuning":
            raise ProtocolError("promotion row evaluated the wrong split")
        seen.add((layer, component))
        observations = _strict_int(row["observations"], label="promotion observations", minimum=1)
        if observations != 40 * 256:
            raise ProtocolError("promotion observation count is not heldout tuning40")
        candidate = _finite_float(row["candidate_sse"], label="candidate SSE")
        nvidia = _finite_float(row["nvidia_k20_sse"], label="NVIDIA SSE")
        shrink = _finite_float(row["shrinkage_sse"], label="shrinkage SSE")
        strict = row["strictly_below_both"]
        if type(strict) is not bool:
            raise ProtocolError("promotion strict flag must be bool")
        expected = candidate < nvidia and candidate < shrink
        if strict is not expected:
            raise ProtocolError("promotion strict flag does not match SSE values")
        passed = passed and expected
    if seen != {(layer, component) for layer in ACTIVE_LAYERS for component in COMPONENTS}:
        raise ProtocolError("full40 promotion rows are incomplete")
    return passed


def _validate_full40_result(result: Mapping[str, Any], pre: Mapping[str, Any]) -> bool:
    """Validate an in-memory or persisted full40 result before publication."""

    if not isinstance(result, Mapping):
        raise ProtocolError("full40 result is not an object")
    if result.get("schema_version") != 1 or result.get("protocol") != PROTOCOL or result.get("status") != "complete" or result.get("stage") != "full40":
        raise ProtocolError("full40 result identity changed")
    if result.get("input_inventory_sha256") != pre.get("input_inventory_sha256"):
        raise ProtocolError("full40 result input binding changed")
    if result.get("fit_split") != "train" or result.get("eval_split") != "heldout_tuning" or result.get("fit_sequence_count") != 500 or result.get("eval_sequence_count") != 40:
        raise ProtocolError("full40 result fit/eval split binding changed")
    if result.get("fit_observations") != 500 * 256 or result.get("eval_observations") != 40 * 256 or result.get("fit_eval_ids_disjoint") is not True:
        raise ProtocolError("full40 result fit/eval observation evidence changed")
    if result.get("prior_probe6_is_not_qa_proof") is not True or result.get("shrinkage_comparator_status") != "validation_tuned_on_heldout_tuning40_exploratory_only":
        raise ProtocolError("full40 prior/comparator boundary changed")
    if result.get("locked_activation_evaluations") != 0 or result.get("locked_rows_indexed") is not False or result.get("answers_or_outcomes_read") is not False:
        raise ProtocolError("full40 result crossed locked/answer boundary")
    gate = result.get("promotion_gate")
    if not isinstance(gate, Mapping) or set(gate) != {"required", "passed", "rows"} or gate.get("required") is not True:
        raise ProtocolError("full40 promotion gate schema changed")
    passed = _validate_full_gate_rows(gate.get("rows"))
    if type(gate.get("passed")) is not bool or gate["passed"] is not passed:
        raise ProtocolError("full40 promotion gate pass bit changed")
    expected_decision = "GO" if passed else "NO_GO"
    if result.get("decision") != expected_decision:
        raise ProtocolError("full40 decision does not match strict gate")
    layers = result.get("layers")
    if not isinstance(layers, Mapping) or set(layers) != {str(layer) for layer in ACTIVE_LAYERS}:
        raise ProtocolError("full40 layer result set changed")
    for layer in ACTIVE_LAYERS:
        entry = layers[str(layer)]
        if not isinstance(entry, Mapping) or set(entry) != set(COMPONENTS):
            raise ProtocolError("full40 layer/component result schema changed")
        for component in COMPONENTS:
            item = entry[component]
            if not isinstance(item, Mapping) or set(item) != {"fit", "promotion"}:
                raise ProtocolError("full40 component result schema changed")
            if not isinstance(item["fit"], Mapping) or not isinstance(item["promotion"], Mapping):
                raise ProtocolError("full40 component diagnostics missing")
            if item["fit"].get("n_observations") != 500 * 256 or item["promotion"].get("observations") != 40 * 256:
                raise ProtocolError("full40 fit/eval observation counts changed")
    return passed


def run_stage(config_path: Path, stage: str, *, execute: bool = False) -> dict[str, Any]:
    """Run a registered stage; never starts without explicit ``execute``."""

    if not execute:
        raise ProtocolError("GPU stages require explicit --execute")
    _fresh_bootstrap()
    config = load_config(config_path)
    stop = _validate_frozen_preflight(config)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["runtime"]["cuda_visible_devices"]:
        raise ProtocolError("CUDA_VISIBLE_DEVICES must expose exactly eight configured GPUs")
    pre = validate_scientific_inputs(config)
    if pre["input_inventory_sha256"] != stop.get("input_inventory_sha256"):
        raise ProtocolError("scientific inputs changed after PRE_FORMAL_STOP")
    if stage != "full40":
        raise ProtocolError("this sibling exposes only the full40 stage")
    assert_fit_eval_split_disjoint(config)
    run_root = _relative_output(config, f"{stage}_run_root")
    if os.path.lexists(run_root):
        raise ProtocolError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    _lstat_component_chain(run_root, label="run root", allow_missing_tail=False)
    atomic_json(run_root / "run_definition.json", {"schema_version": 1, "status": "running", "stage": stage, "config": file_ref(config_path.resolve()), "input_inventory_sha256": pre["input_inventory_sha256"], "target_layers": list(_stage_layers(stage)), "tuning_sequence_count": 40, "locked_sequence_count": 10, "locked_rows_indexed": False, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    lock = idle_gate(config, run_root)
    try:
        import torch
        from safetensors.torch import save_file

        torch.backends.cuda.matmul.allow_tf32 = False
        fit_ids = _split_ids(config, "train")
        eval_ids = _split_ids(config, "heldout_tuning")
        if len(fit_ids) != 500 or len(eval_ids) != 40:
            raise ProtocolError("fit/eval split cardinality changed")
        selections = {int(k): tuple(v) for k, v in pre["source_selection_by_target"].items()}
        layer_metrics: dict[str, Any] = {}
        candidate_refs: list[dict[str, Any]] = []
        for target_layer in _stage_layers(stage):
            device = f"cuda:{target_layer % 8}"
            component_metrics: dict[str, Any] = {}
            for component in COMPONENTS:
                weights, biases, diagnostics = _fit_head_local_torch(config, target_layer, component, selections[target_layer], fit_ids, device, "train")
                destination = _assert_inside(ROOT, run_root / "mappers" / f"layer_{target_layer:03d}_{component}.safetensors", label="candidate mapper output")
                destination.parent.mkdir(parents=True, exist_ok=True)
                _lstat_component_chain(destination.parent, label="candidate mapper parent", allow_missing_tail=False)
                temp = destination.with_name(f".{destination.name}.tmp")
                save_file({"weight": weights.contiguous(), "bias": biases.contiguous()}, str(temp), metadata={"method": "cachebridge_head_local_unweighted_k20", "target_layer": str(target_layer), "component": component})
                with temp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp, destination)
                sidecar = destination.with_suffix(".json")
                atomic_json(sidecar, {"schema_version": 1, "target_layer": target_layer, "component": component, "source_layers_rank_order": list(selections[target_layer]), "weight_shape": list(weights.shape), "bias_shape": list(biases.shape), "ridge_lambda": 0.01, "weighted_loss": False, "regularize_bias": False, "fit_split": "train", "eval_split": "heldout_tuning", "fit_sequence_count": 500, "eval_sequence_count": 40, "diagnostics": diagnostics, "sha256": sha256_file(destination)})
                candidate_refs.append(file_ref(destination))
                candidate_refs.append(file_ref(sidecar))
                component_metrics[component] = {
                    "fit": diagnostics,
                    "promotion": _evaluate_full_gate_torch(config, pre, target_layer, component, selections[target_layer], eval_ids, weights, biases, device),
                }
            layer_metrics[str(target_layer)] = component_metrics
        gate_rows = [layer_metrics[str(layer)][component]["promotion"] for layer in ACTIVE_LAYERS for component in COMPONENTS]
        gate_passed = _validate_full_gate_rows(gate_rows)
        results = {"schema_version": 1, "protocol": PROTOCOL, "status": "complete", "stage": stage, "decision": "GO" if gate_passed else "NO_GO", "promotion_gate": {"required": True, "passed": bool(gate_passed), "rows": gate_rows}, "layers": layer_metrics, "candidate_files": candidate_refs, "input_inventory_sha256": pre["input_inventory_sha256"], "fit_split": "train", "eval_split": "heldout_tuning", "fit_sequence_count": 500, "eval_sequence_count": 40, "fit_observations": 500 * 256, "eval_observations": 40 * 256, "fit_eval_ids_disjoint": True, "locked_activation_evaluations": 0, "locked_rows_indexed": False, "answers_or_outcomes_read": False, "prior_probe6_is_not_qa_proof": True, "shrinkage_comparator_status": "validation_tuned_on_heldout_tuning40_exploratory_only", "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        _validate_full40_result(results, pre)
        atomic_json(run_root / "results.json", results)
        results_ref = file_ref(run_root / "results.json")
        write_sidecar(run_root / "results.json")
        atomic_json(run_root / "COMPLETED.json", {"status": "complete", "stage": stage, "results": results_ref, "results_sidecar": file_ref(run_root / "results.json.sha256.json"), "input_inventory_sha256": pre["input_inventory_sha256"], "decision": results["decision"]})
        return results
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


__all__ = [
    "ProtocolError", "canonical_bytes", "canonical_sha256", "sha256_file", "file_ref", "strict_json", "atomic_json",
    "immutable_json", "write_sidecar", "verify_sidecar", "model_shard_inventory", "source_byte_closure", "validate_source_closure",
    "validate_config_shape", "load_config", "validate_scientific_inputs", "preflight", "tuning_ids", "assert_tuning_ids",
    "fit_head_local_numpy", "predict_head_local", "activation_metrics", "compare_metric_rows", "block_diagonal_head_weights",
    "idle_gate", "run_stage", "_validate_full_gate_rows", "_validate_full40_result", "PROBE_LAYERS", "FULL_LAYERS", "ACTIVE_LAYERS", "COMPONENTS",
]
