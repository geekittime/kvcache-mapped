"""CacheBridge Head-Local activation probe.

The probe is deliberately split into a CPU-only input gate and an explicit
GPU execution path.  The scientific candidate is a block-diagonal (one KV
head at a time) affine ridge mapper:

    y[h] = x[h] @ W[h] + b[h]

where ``x[h]`` concatenates the 128 channels of the same KV head over the 20
frozen source layers (20 * 128 = 2560).  It is an ordinary, unweighted SSE
objective; no attention/readout weights are used.  Only the forty tuning rows
from the frozen ``02_shrinkage_mapper`` split may be interpreted.  The ten
locked rows are represented by a count/hash boundary and are never indexed.

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
PROBE_LAYERS = (0, 7, 15, 23, 31, 39)
FULL_LAYERS = tuple(range(40))
COMPONENTS = ("key", "value")
COMPONENT_DIR = {"key": "k_content", "value": "v"}
EXPECTED_PAYLOAD_BYTES = 50 * 256 * 8 * 128 * 2
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
    _expect(config.get("experiment_id") == "qwen3_32b_to_14b_cachebridge_head_local_probe_v1", "experiment identity changed")
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
        "heldout_tokens", "source_activations", "target_activations", "source_model", "target_model",
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
    protocol = config.get("protocol")
    _expect(isinstance(protocol, Mapping), "protocol section missing")
    _expect(protocol.get("source_layers") == 64 and protocol.get("target_layers") == 40, "model layer counts changed")
    _expect(protocol.get("selected_source_layers") == 20 and protocol.get("kv_heads") == 8 and protocol.get("head_dim") == 128, "KV topology changed")
    _expect(protocol.get("components") == list(COMPONENTS), "component order changed")
    _expect(protocol.get("probe_target_layers") == list(PROBE_LAYERS), "probe layer order changed")
    _expect(protocol.get("full_target_layers") == list(FULL_LAYERS), "full layer registration changed")
    _expect(protocol.get("tuning_sequence_count") == 40 and protocol.get("locked_sequence_count") == 10, "split cardinality changed")
    _expect(protocol.get("observations_per_sequence") == 256 and protocol.get("activation_stride") == 4, "activation geometry changed")
    _expect(protocol.get("evaluation_scope") == "02_frozen_tuning40_only", "evaluation scope changed")
    _expect(protocol.get("locked_activation_evaluations") == 0 and protocol.get("locked_rows_indexed") is False, "locked boundary changed")
    _expect(protocol.get("require_k20_and_shrinkage_selection_equal") is True, "selection equality guard missing")
    _expect(protocol.get("comparison_methods") == ["nvidia_k20", "shrinkage", "cachebridge_head_local"], "comparison methods changed")
    gate = protocol.get("promotion_gate")
    _expect(isinstance(gate, Mapping), "promotion gate missing")
    _expect(gate.get("probe6_candidate_sse_strictly_below_k20_and_shrinkage_for_each_component") is True, "probe gate changed")
    _expect(gate.get("full40_requires_probe6_go") is True and gate.get("locked_activation_evaluations") == 0, "full-stage gate changed")
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
    _expect(outputs.get("stop_receipt") == "artifacts/preformal/PRE_FORMAL_STOP_v1.json", "stop output changed")
    _expect(outputs.get("probe6_run_root") == "runs/probe6" and outputs.get("full40_run_root") == "runs/full40", "run roots changed")


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = ROOT / "config.json"
    path = _lstat_component_chain(config_path, label="config", allow_missing_tail=False)
    if path.resolve(strict=True) != path:
        raise ProtocolError("config path is not canonical")
    _expect(path == ROOT / "config.json", "only the sibling's canonical config.json is accepted")
    config = strict_json(path)
    validate_config_shape(config)
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
    else:
        _expect(isinstance(manifest.get("method_definition"), Mapping), "shrinkage method definition missing")
        _expect(manifest["method_definition"].get("feature_order") == "selected_layer_rank,kv_head,head_dim", "shrinkage feature order changed")


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


def _mapper_bindings(config: Mapping[str, Any], k20_manifest: Mapping[str, Any], shrink_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    kroot = Path(config["upstream"]["k20_root"]).resolve()
    sroot = Path(config["upstream"]["shrinkage_root"]).resolve()
    k_layers = _manifest_layers(k20_manifest, label="k20")
    s_layers = _manifest_layers(shrink_manifest, label="shrinkage")
    _check_manifest_fit_definition(k20_manifest, label="k20")
    _check_manifest_fit_definition(shrink_manifest, label="shrinkage")
    rows: list[dict[str, Any]] = []
    for target in PROBE_LAYERS:
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
    return rows


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
        "tuning_sequence_ids": list(tuning),
        "tuning_sequence_ids_sha256": canonical_sha256(list(tuning)),
        "locked_sequence_count": 10,
        "locked_sequence_ids_sha256": canonical_sha256(list(locked)),
        "locked_rows_indexed": False,
        "fold_manifest_sha256": canonical_sha256(folds),
    }


def _activation_paths(config: Mapping[str, Any], selections: Mapping[int, Sequence[int]]) -> tuple[list[Path], list[Path]]:
    source_layers = sorted({int(v) for row in selections.values() for v in row})
    target_layers = list(PROBE_LAYERS)
    payloads: list[Path] = []
    metadata: list[Path] = []
    for component in COMPONENTS:
        subdir = COMPONENT_DIR[component]
        source_root = _lstat_component_chain(Path(config["upstream"]["source_activations"]) / subdir, label="source activation root", allow_missing_tail=False)
        target_root = _lstat_component_chain(Path(config["upstream"]["target_activations"]) / subdir, label="target activation root", allow_missing_tail=False)
        for layer in source_layers:
            stem = source_root / f"layer_{layer:03d}"
            payloads.append(_lstat_component_chain(stem.with_suffix(".bf16.bin"), label="source activation payload", allow_missing_tail=False))
            metadata.append(_lstat_component_chain(stem.with_suffix(".json"), label="source activation metadata", allow_missing_tail=False))
        for layer in target_layers:
            stem = target_root / f"layer_{layer:03d}"
            payloads.append(_lstat_component_chain(stem.with_suffix(".bf16.bin"), label="target activation payload", allow_missing_tail=False))
            metadata.append(_lstat_component_chain(stem.with_suffix(".json"), label="target activation metadata", allow_missing_tail=False))
    return sorted(set(payloads)), sorted(set(metadata))


def _validate_activation_inventory(
    config: Mapping[str, Any],
    selections: Mapping[int, Sequence[int]],
    split: Mapping[str, Any],
    token_sha: str,
    model_inventories: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    payload_paths, metadata_paths = _activation_paths(config, selections)
    payload_inventory: list[dict[str, Any]] = []
    metadata_inventory: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        metadata = strict_json(metadata_path)
        _expect(metadata.get("shape") == EXPECTED_METADATA_SHAPE, f"activation shape changed: {metadata_path}")
        _expect(type(metadata.get("nbytes")) is int and metadata.get("nbytes") == EXPECTED_PAYLOAD_BYTES, f"activation metadata size changed: {metadata_path}")
        provenance = metadata.get("provenance")
        _expect(isinstance(provenance, Mapping), f"activation provenance missing: {metadata_path}")
        _expect(provenance.get("token_sha256") == token_sha and provenance.get("split") == "heldout", f"activation token/split provenance changed: {metadata_path}")
        _expect(type(provenance.get("sequences")) is int and provenance.get("sequences") == 50 and type(provenance.get("stride")) is int and provenance.get("stride") == 4, f"activation sequence provenance changed: {metadata_path}")
        component = next((name for name, directory in COMPONENT_DIR.items() if directory in metadata_path.parts), None)
        _expect(component in COMPONENTS, f"activation component path changed: {metadata_path}")
        layer_name = metadata_path.stem
        _expect(layer_name.startswith("layer_") and layer_name[6:].isdigit(), f"activation layer filename changed: {metadata_path}")
        expected_layer = int(layer_name[6:])
        _expect(provenance.get("layer") == expected_layer and provenance.get("cache_type") == COMPONENT_DIR[component], f"activation layer/component provenance changed: {metadata_path}")
        is_source = str(config["upstream"]["source_activations"]) in str(metadata_path)
        model_inventory = model_inventories["source_32b" if is_source else "target_14b"]
        model_root = model_inventory["model_root"]
        expected_model_name = "qwen3_32b" if is_source else "qwen3_14b"
        _expect(provenance.get("model_path") == model_root and provenance.get("model_name") == expected_model_name, f"activation model provenance changed: {metadata_path}")
        _expect(provenance.get("model_config_sha256") == model_inventory["config"]["sha256"], f"activation config provenance changed: {metadata_path}")
        _expect(provenance.get("model_weight_index_sha256") == model_inventory["weight_index"]["sha256"], f"activation index provenance changed: {metadata_path}")
        _expect(provenance.get("model_tokenizer_sha256") == model_inventory["tokenizer"]["sha256"], f"activation tokenizer provenance changed: {metadata_path}")
        metadata_inventory.append(file_ref(metadata_path))
    for payload_path in payload_paths:
        _expect(payload_path.stat().st_size == EXPECTED_PAYLOAD_BYTES, f"activation payload size changed: {payload_path}")
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
        "tuning_sequence_count": len(split["tuning_sequence_ids"]),
        "locked_sequence_count": split["locked_sequence_count"],
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "opaque_payload_hashing_does_not_index_locked_values": True,
    }


def validate_scientific_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash every probe6 input and return a machine-checkable inventory."""

    validate_config_shape(config)
    model_inventories = {
        "source_32b": model_shard_inventory(config["upstream"]["source_model"], label="source Qwen3-32B"),
        "target_14b": model_shard_inventory(config["upstream"]["target_model"], label="target Qwen3-14B"),
    }
    _expect(model_inventories["source_32b"]["architecture"]["num_hidden_layers"] == 64, "source model depth changed")
    _expect(model_inventories["target_14b"]["architecture"]["num_hidden_layers"] == 40, "target model depth changed")
    refs = {key: _verify_config_ref(config, key) for key in ("k20_manifest", "k20_manifest_marker", "k20_selection", "k20_fit_definition", "shrinkage_manifest", "shrinkage_manifest_marker", "split_manifest", "heldout_tokens")}
    k20_manifest = strict_json(refs["k20_manifest"])
    shrink_manifest = strict_json(refs["shrinkage_manifest"])
    _expect(file_ref(refs["k20_manifest"])["sha256"] == config["upstream"]["k20_manifest"]["sha256"], "k20 manifest binding changed")
    _expect(file_ref(refs["shrinkage_manifest"])["sha256"] == config["upstream"]["shrinkage_manifest"]["sha256"], "shrinkage manifest binding changed")
    k_layers = _manifest_layers(k20_manifest, label="k20")
    s_layers = _manifest_layers(shrink_manifest, label="shrinkage")
    selections = {target: tuple(int(v) for v in k_layers[target]["source_layers_rank_order"]) for target in PROBE_LAYERS}
    for target in PROBE_LAYERS:
        _expect(selections[target] == tuple(int(v) for v in s_layers[target]["source_layers_rank_order"]), f"target {target}: selection mismatch")
    split = _load_split(config)
    token_ref = config["upstream"]["heldout_tokens"]
    activation = _validate_activation_inventory(config, selections, split, str(token_ref["sha256"]), model_inventories)
    mapper = _mapper_bindings(config, k20_manifest, shrink_manifest)
    closure = source_byte_closure()
    inventory = {
        "models": model_inventories,
        "mapper_files": mapper,
        "activation": activation,
        "upstream_refs": {key: file_ref(path) for key, path in refs.items()},
        "source_byte_closure_digest": closure["digest"],
    }
    return {
        "status": "pass",
        "protocol": "cachebridge_head_local_unweighted_k20",
        "probe_target_layers": list(PROBE_LAYERS),
        "full_target_layers_registered": list(FULL_LAYERS),
        "source_selection_by_target": {str(k): list(v) for k, v in selections.items()},
        "split": split,
        "mapper_files": mapper,
        "activation": activation,
        "upstream_refs": {key: file_ref(path) for key, path in refs.items()},
        "models": model_inventories,
        "source_byte_closure": closure,
        "input_inventory_sha256": canonical_sha256(inventory),
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "answers_or_outcomes_read": False,
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


def _mmap_activation(path: Path) -> Any:
    """Open a BF16 payload without interpreting any row until requested."""

    import torch

    _expect(path.stat().st_size == EXPECTED_PAYLOAD_BYTES, f"activation payload size changed: {path}")
    raw = torch.from_file(str(path), shared=False, size=EXPECTED_PAYLOAD_BYTES // 2, dtype=torch.uint16)
    return raw.view(torch.bfloat16).reshape(50, 256, 8, 128)


def _activation_batch(config: Mapping[str, Any], target_layer: int, component: str, ids: Sequence[int], selection: Sequence[int]) -> tuple[Any, Any]:
    """Materialize one tuning-only batch in [tokens,features] form."""

    assert_tuning_ids(ids, config)
    import torch

    source_root = Path(config["upstream"]["source_activations"]) / COMPONENT_DIR[component]
    target_root = Path(config["upstream"]["target_activations"]) / COMPONENT_DIR[component]
    source = [_mmap_activation(source_root / f"layer_{int(layer):03d}.bf16.bin") for layer in selection]
    target = _mmap_activation(target_root / f"layer_{int(target_layer):03d}.bf16.bin")
    # The list comprehension is intentionally over ``ids`` only.  No locked
    # sequence id is available to this function.
    x = torch.stack([source_rank[list(ids)] for source_rank in source], dim=1)
    y = target[list(ids)]
    return x, y


def _fit_head_local_torch(config: Mapping[str, Any], target_layer: int, component: str, selection: Sequence[int], ids: Sequence[int], device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Two-pass streaming GPU fit for all eight heads of one layer/component."""

    import torch

    assert_tuning_ids(ids, config)
    dev = torch.device(device)
    chunk = int(config["runtime"]["chunk_sequences"])
    means_x = torch.zeros((8, 2560), dtype=torch.float32, device=dev)
    means_y = torch.zeros((8, 128), dtype=torch.float32, device=dev)
    count = 0
    for first in range(0, len(ids), chunk):
        batch_ids = list(ids[first:first + chunk])
        x_cpu, y_cpu = _activation_batch(config, target_layer, component, batch_ids, selection)
        x = x_cpu.to(dev, dtype=torch.float32).permute(0, 1, 2, 3, 4).reshape(len(batch_ids) * 256, 20, 8, 128)
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
        x_cpu, y_cpu = _activation_batch(config, target_layer, component, batch_ids, selection)
        x = x_cpu.to(dev, dtype=torch.float32).reshape(len(batch_ids) * 256, 20, 8, 128)
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

    expected_python = Path(str(config["runtime"]["required_python"])).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise ProtocolError(f"preflight must use the configured Python: {sys.executable}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ProtocolError("CPU preflight requires CUDA_VISIBLE_DEVICES='' (empty)")
    if "torch" in sys.modules or "transformers" in sys.modules or "safetensors" in sys.modules:
        raise ProtocolError("heavy model/CUDA modules were imported before CPU preflight")
    prefix = _fresh_bootstrap()
    return {
        "executable": str(Path(sys.executable).resolve()),
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


def _preflight_definition(config: Mapping[str, Any], inputs: Mapping[str, Any], closure: Mapping[str, Any], environment: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": 1,
        "protocol": "cachebridge_head_local_unweighted_k20_probe_v1",
        "experiment_id": config["experiment_id"],
        "direction": config["direction"],
        "method": config["method"],
        "probe_target_layers": list(PROBE_LAYERS),
        "full_target_layers_registered": list(FULL_LAYERS),
        "inputs": inputs,
        "source_byte_closure": closure,
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
    environment = _cpu_preflight_environment(config)
    inputs = validate_scientific_inputs(config)
    closure = inputs["source_byte_closure"]
    definition = _preflight_definition(config, inputs, closure, environment)
    preflight_value = {
        "schema_version": 1,
        "protocol": "cachebridge_head_local_unweighted_k20_probe_v1",
        "experiment_id": config["experiment_id"],
        "status": "passed_cpu_only_preflight",
        "gpu_used": False,
        "cuda_api_called": False,
        "models_loaded": False,
        "request_fingerprint": definition["request_fingerprint"],
        "definition_digest": canonical_sha256(definition),
        "definition": definition,
        "input_inventory_sha256": inputs["input_inventory_sha256"],
        "source_byte_closure_digest": closure["digest"],
        "source_byte_closure_file_count": closure["file_count"],
        "model_inventory_digests": {key: value["digest"] for key, value in inputs["models"].items()},
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
        "protocol": "cachebridge_head_local_unweighted_k20_probe_v1",
        "experiment_id": config["experiment_id"],
        "status": "PRE_FORMAL_STOP",
        "stop_mode": "CPU_PREFLIGHT_COMPLETE_WAITING_FOR_OPERATOR_GPU_GATE",
        "request_fingerprint": definition["request_fingerprint"],
        "definition_digest": canonical_sha256(definition),
        "config": _config_file_ref(),
        "preflight": preflight_ref,
        "preflight_sidecar": preflight_sidecar,
        "source_byte_closure_digest": closure["digest"],
        "source_byte_closure_file_count": closure["file_count"],
        "input_inventory_sha256": inputs["input_inventory_sha256"],
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
            f"--config {ROOT / 'config.json'} --stage probe6 --execute"
        ),
        "unfinished_boundaries": [
            "probe6/full40 mapper fitting and activation metric comparison",
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
    if stop.get("status") != "PRE_FORMAL_STOP" or stop.get("execute_authorized") is not False:
        raise ProtocolError("STOP receipt is not an unexecuted CPU snapshot")
    preflight_ref = stop.get("preflight")
    if not isinstance(preflight_ref, Mapping):
        raise ProtocolError("STOP preflight binding is malformed")
    preflight_path = _strict_ref(preflight_ref, label="STOP preflight")
    verify_sidecar(preflight_path)
    preflight = strict_json(preflight_path)
    if preflight.get("status") != "passed_cpu_only_preflight" or preflight.get("gpu_used") is not False or preflight.get("models_loaded") is not False:
        raise ProtocolError("preflight receipt is not CPU-only")
    if stop.get("source_byte_closure_digest") != preflight.get("source_byte_closure_digest"):
        raise ProtocolError("STOP/preflight source closure differs")
    definition = preflight.get("definition")
    if not isinstance(definition, Mapping):
        raise ProtocolError("preflight definition is missing")
    closure = definition.get("source_byte_closure")
    if not isinstance(closure, Mapping):
        raise ProtocolError("preflight source closure is missing")
    validate_source_closure(closure)
    if file_ref(ROOT / "config.json") != stop.get("config"):
        raise ProtocolError("config changed after PRE_FORMAL_STOP")
    return stop


def _stage_layers(stage: str) -> tuple[int, ...]:
    if stage == "probe6":
        return PROBE_LAYERS
    if stage == "full40":
        return FULL_LAYERS
    raise ProtocolError("stage must be probe6 or full40")


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
    if stage == "full40":
        probe_root = ROOT / config["outputs"]["probe6_run_root"]
        completed = probe_root / "COMPLETED.json"
        if not completed.is_file():
            raise ProtocolError("full40 is gated on a completed probe6 result")
        probe_result = strict_json(probe_root / "results.json")
        _expect(probe_result.get("decision") == "GO", "probe6 did not pass the pre-registered GO gate")
        _expect(probe_result.get("locked_activation_evaluations") == 0, "probe6 locked boundary changed")
    run_root = ROOT / config["outputs"][f"{stage}_run_root"]
    if run_root.exists():
        raise ProtocolError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    atomic_json(run_root / "run_definition.json", {"schema_version": 1, "status": "running", "stage": stage, "config": file_ref(config_path.resolve()), "input_inventory_sha256": pre["input_inventory_sha256"], "target_layers": list(_stage_layers(stage)), "tuning_sequence_count": 40, "locked_sequence_count": 10, "locked_rows_indexed": False, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    lock = idle_gate(config, run_root)
    try:
        import torch
        from safetensors.torch import save_file

        torch.backends.cuda.matmul.allow_tf32 = False
        ids = tuning_ids(config)
        selections = {int(k): tuple(v) for k, v in pre["source_selection_by_target"].items()}
        layer_metrics: dict[str, Any] = {}
        candidate_refs: list[dict[str, Any]] = []
        for target_layer in _stage_layers(stage):
            device = f"cuda:{target_layer % 8}"
            component_metrics: dict[str, Any] = {}
            for component in COMPONENTS:
                weights, biases, diagnostics = _fit_head_local_torch(config, target_layer, component, selections[target_layer], ids, device)
                destination = run_root / "mappers" / f"layer_{target_layer:03d}_{component}.safetensors"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temp = destination.with_name(f".{destination.name}.tmp")
                save_file({"weight": weights.contiguous(), "bias": biases.contiguous()}, str(temp), metadata={"method": "cachebridge_head_local_unweighted_k20", "target_layer": str(target_layer), "component": component})
                with temp.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temp, destination)
                sidecar = destination.with_suffix(".json")
                atomic_json(sidecar, {"schema_version": 1, "target_layer": target_layer, "component": component, "source_layers_rank_order": list(selections[target_layer]), "weight_shape": list(weights.shape), "bias_shape": list(biases.shape), "ridge_lambda": 0.01, "weighted_loss": False, "regularize_bias": False, "diagnostics": diagnostics, "sha256": sha256_file(destination)})
                candidate_refs.append(file_ref(destination))
                candidate_refs.append(file_ref(sidecar))
                component_metrics[component] = {"fit": diagnostics}
            layer_metrics[str(target_layer)] = component_metrics
        results = {"schema_version": 1, "status": "complete", "stage": stage, "decision": "PENDING_METRIC_COMPARISON", "layers": layer_metrics, "candidate_files": candidate_refs, "input_inventory_sha256": pre["input_inventory_sha256"], "locked_activation_evaluations": 0, "locked_rows_indexed": False, "answers_or_outcomes_read": False, "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(run_root / "results.json", results)
        atomic_json(run_root / "COMPLETED.json", {"status": "complete", "results": file_ref(run_root / "results.json")})
        return results
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


__all__ = [
    "ProtocolError", "canonical_bytes", "canonical_sha256", "sha256_file", "file_ref", "strict_json", "atomic_json",
    "immutable_json", "write_sidecar", "verify_sidecar", "model_shard_inventory", "source_byte_closure", "validate_source_closure",
    "validate_config_shape", "load_config", "validate_scientific_inputs", "preflight", "tuning_ids", "assert_tuning_ids",
    "fit_head_local_numpy", "predict_head_local", "activation_metrics", "compare_metric_rows", "block_diagonal_head_weights",
    "idle_gate", "run_stage", "PROBE_LAYERS", "FULL_LAYERS", "COMPONENTS",
]
