from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


class ProtocolError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def artifact(path: Path) -> dict[str, Any]:
    st = path.lstat()
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"artifact is not a regular non-symlink file: {path}")
    return {"path": str(path.resolve()), "nbytes": st.st_size, "sha256": sha256_file(path)}


def verify_ref(ref: Mapping[str, Any]) -> Path:
    if set(ref) != {"path", "nbytes", "sha256"}:
        raise ProtocolError("artifact ref schema changed")
    path = Path(str(ref["path"]))
    if path.is_symlink() or not path.is_file():
        raise ProtocolError(f"missing/nonregular artifact: {path}")
    if type(ref["nbytes"]) is not int or path.stat().st_size != ref["nbytes"]:
        raise ProtocolError(f"artifact size mismatch: {path}")
    if not isinstance(ref["sha256"], str) or sha256_file(path) != ref["sha256"]:
        raise ProtocolError(f"artifact hash mismatch: {path}")
    return path.resolve()


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != 1 or config.get("direction") != "qwen3_32b_to_14b":
        raise ProtocolError("config identity changed")
    for field in ("shrinkage_manifest", "split_manifest", "heldout_tokens"):
        verify_ref(config["upstream"][field])
    protocol = config["protocol"]
    if protocol.get("target_layers") != [7, 8, 9, 10, 34, 35]:
        raise ProtocolError("target layer set changed")
    if protocol.get("components") != ["key", "value"]:
        raise ProtocolError("component order changed")
    if protocol.get("go_gate", {}).get("locked_activation_evaluations") != 0:
        raise ProtocolError("locked activation use is forbidden")
    return config


def split_rows(config: Mapping[str, Any]) -> tuple[list[int], list[list[int]], list[int]]:
    split = read_json(Path(config["upstream"]["split_manifest"]["path"]))
    tuning = [int(value) for value in split["tuning_sequence_ids"]]
    locked = [int(value) for value in split["locked_audit_sequence_ids"]]
    folds = [[int(value) for value in row["validation_sequence_ids"]] for row in split["cv_folds"]]
    if len(tuning) != 40 or len(set(tuning)) != 40 or len(locked) != 10 or set(tuning) & set(locked):
        raise ProtocolError("tuning/locked split changed")
    if len(folds) != 5 or any(len(row) != 8 for row in folds):
        raise ProtocolError("fold cardinality changed")
    if sorted(value for row in folds for value in row) != sorted(tuning):
        raise ProtocolError("folds do not partition tuning rows")
    return tuning, folds, locked


def validate_scientific_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-bind every mapper/activation file used by the probe.

    Activation payload hashing is permitted because it treats the file as
    opaque bytes.  Only the 40 tuning sequence indices are ever interpreted
    by ``collect_stats``.
    """
    manifest_path = Path(config["upstream"]["shrinkage_manifest"]["path"])
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("direction") != "qwen3_32b_to_14b":
        raise ProtocolError("shrinkage manifest state changed")
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != 40:
        raise ProtocolError("shrinkage layer inventory changed")
    root = Path(config["upstream"]["shrinkage_root"]).resolve()
    token_sha = str(config["upstream"]["heldout_tokens"]["sha256"])
    payload_paths: set[Path] = set()
    metadata_paths: set[Path] = set()
    mapper_paths: set[Path] = set()
    for target_layer in config["protocol"]["target_layers"]:
        entry = layers[int(target_layer)]
        if entry.get("target_layer") != int(target_layer):
            raise ProtocolError("shrinkage target layer order changed")
        mapper_path = (root / str(entry["path"])).resolve()
        mapper_path.relative_to(root)
        if artifact(mapper_path)["sha256"] != entry.get("sha256"):
            raise ProtocolError(f"shrinkage layer hash mismatch: {target_layer}")
        sidecar_path = (root / str(entry["sidecar_path"])).resolve()
        sidecar_path.relative_to(root)
        if artifact(sidecar_path)["sha256"] != entry.get("sidecar_sha256"):
            raise ProtocolError(f"shrinkage sidecar hash mismatch: {target_layer}")
        mapper_paths.update((mapper_path, sidecar_path))
        selected = [int(value) for value in entry["source_layers_rank_order"]]
        if len(selected) != 20 or len(set(selected)) != 20:
            raise ProtocolError("source selection cardinality changed")
        for component in config["protocol"]["components"]:
            activation_component = "k_content" if component == "key" else "v"
            source_root = Path(config["upstream"]["source_activations"]) / activation_component
            target_root = Path(config["upstream"]["target_activations"]) / activation_component
            for source_layer in selected:
                stem = source_root / f"layer_{source_layer:03d}"
                payload_paths.add(stem.with_suffix(".bf16.bin"))
                metadata_paths.add(stem.with_suffix(".json"))
            stem = target_root / f"layer_{int(target_layer):03d}"
            payload_paths.add(stem.with_suffix(".bf16.bin"))
            metadata_paths.add(stem.with_suffix(".json"))
    expected_bytes = 50 * 256 * 8 * 128 * 2
    for metadata_path in sorted(metadata_paths):
        metadata = read_json(metadata_path)
        provenance = metadata.get("provenance")
        if metadata.get("shape") != [12800, 8, 128] or metadata.get("nbytes") != expected_bytes:
            raise ProtocolError(f"activation metadata shape/size changed: {metadata_path}")
        if not isinstance(provenance, Mapping) or provenance.get("token_sha256") != token_sha:
            raise ProtocolError(f"activation token provenance changed: {metadata_path}")
        if provenance.get("split") != "heldout" or provenance.get("sequences") != 50 or provenance.get("stride") != 4:
            raise ProtocolError(f"activation split provenance changed: {metadata_path}")
    payload_inventory = []
    for payload_path in sorted(payload_paths):
        if payload_path.stat().st_size != expected_bytes:
            raise ProtocolError(f"activation payload size changed: {payload_path}")
        payload_inventory.append(artifact(payload_path))
    return {
        "status": "pass",
        "mapper_files": [artifact(path) for path in sorted(mapper_paths)],
        "activation_metadata_files": [artifact(path) for path in sorted(metadata_paths)],
        "activation_payload_files": payload_inventory,
        "activation_payload_bytes_hashed": sum(row["nbytes"] for row in payload_inventory),
        "opaque_hashing_does_not_index_locked_values": True,
    }


def fit_head_scale_channel_bias(stats: Mapping[str, Any], rows: Sequence[int], bounds: Sequence[float]) -> tuple[Any, Any]:
    import numpy as np

    ids = np.asarray(rows, dtype=np.int64)
    n = float(len(ids) * 256)
    sp = np.asarray(stats["sum_p"], dtype=np.float64)[ids].sum(axis=0)
    sy = np.asarray(stats["sum_y"], dtype=np.float64)[ids].sum(axis=0)
    spp = np.asarray(stats["sum_p2"], dtype=np.float64)[ids].sum(axis=0)
    spy = np.asarray(stats["sum_py"], dtype=np.float64)[ids].sum(axis=0)
    varp = np.maximum(spp - sp * sp / n, 0.0)
    cov = spy - sp * sy / n
    denominator = varp.sum(axis=1)
    scale = np.ones(8, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).tiny
    scale[valid] = cov.sum(axis=1)[valid] / denominator[valid]
    scale = np.clip(scale, float(bounds[0]), float(bounds[1]))
    bias = sy / n - scale[:, None] * (sp / n)
    if not np.isfinite(scale).all() or not np.isfinite(bias).all():
        raise ProtocolError("non-finite calibration")
    return scale, bias


def sse_for(stats: Mapping[str, Any], rows: Sequence[int], scale: Any, bias: Any, beta: float) -> Any:
    import numpy as np

    ids = np.asarray(rows, dtype=np.int64)
    n = float(len(ids) * 256)
    sp = np.asarray(stats["sum_p"], dtype=np.float64)[ids].sum(axis=0)
    sy = np.asarray(stats["sum_y"], dtype=np.float64)[ids].sum(axis=0)
    spp = np.asarray(stats["sum_p2"], dtype=np.float64)[ids].sum(axis=0)
    spy = np.asarray(stats["sum_py"], dtype=np.float64)[ids].sum(axis=0)
    syy = np.asarray(stats["sum_y2"], dtype=np.float64)[ids].sum(axis=0)
    effective_scale = 1.0 + float(beta) * (np.asarray(scale, dtype=np.float64)[:, None] - 1.0)
    effective_bias = float(beta) * np.asarray(bias, dtype=np.float64)
    result = (
        syy
        + effective_scale * effective_scale * spp
        + n * effective_bias * effective_bias
        - 2.0 * effective_scale * spy
        - 2.0 * effective_bias * sy
        + 2.0 * effective_scale * effective_bias * sp
    )
    tolerance = 1e-8 * max(1.0, float(np.max(syy)))
    if float(np.min(result)) < -tolerance:
        raise ProtocolError("negative SSE beyond roundoff")
    return np.maximum(result, 0.0)


def target_sst(stats: Mapping[str, Any]) -> Any:
    import numpy as np

    n = float(len(stats["sum_y"]) * 256)
    sy = np.asarray(stats["sum_y"], dtype=np.float64).sum(axis=0)
    syy = np.asarray(stats["sum_y2"], dtype=np.float64).sum(axis=0)
    return np.maximum(syy - sy * sy / n, 0.0)


def nested_cv(component_stats: Sequence[Mapping[str, Any]], folds: Sequence[Sequence[int]], beta_grid: Sequence[float], bounds: Sequence[float]) -> dict[str, Any]:
    import numpy as np

    all_rows = sorted(value for fold in folds for value in fold)
    oof_base = [np.zeros((8, 128), dtype=np.float64) for _ in component_stats]
    oof_candidate = [np.zeros((8, 128), dtype=np.float64) for _ in component_stats]
    outer = []
    for outer_index, validation in enumerate(folds):
        validation_set = set(validation)
        training = [value for value in all_rows if value not in validation_set]
        inner = []
        for inner_index, inner_validation in enumerate(folds):
            if inner_index == outer_index:
                continue
            inner_val = [value for value in inner_validation if value in training]
            inner_train = [value for value in training if value not in set(inner_val)]
            inner.append((inner_train, inner_val))
        losses = []
        for beta in beta_grid:
            loss = 0.0
            for stats in component_stats:
                for inner_train, inner_val in inner:
                    scale, bias = fit_head_scale_channel_bias(stats, inner_train, bounds)
                    loss += float(sse_for(stats, inner_val, scale, bias, beta).sum())
            losses.append(loss)
        best_index = min(range(len(losses)), key=lambda index: (losses[index], index))
        beta = float(beta_grid[best_index])
        for layer_index, stats in enumerate(component_stats):
            scale, bias = fit_head_scale_channel_bias(stats, training, bounds)
            oof_base[layer_index] += sse_for(stats, validation, scale, bias, 0.0)
            oof_candidate[layer_index] += sse_for(stats, validation, scale, bias, beta)
        outer.append({"fold": outer_index, "training_rows": training, "validation_rows": list(validation), "selected_beta": beta, "inner_sse_by_beta": [float(v) for v in losses]})

    deployment_losses = []
    for beta in beta_grid:
        loss = 0.0
        for stats in component_stats:
            for validation in folds:
                training = [value for value in all_rows if value not in set(validation)]
                scale, bias = fit_head_scale_channel_bias(stats, training, bounds)
                loss += float(sse_for(stats, validation, scale, bias, beta).sum())
        deployment_losses.append(loss)
    deployment_index = min(range(len(deployment_losses)), key=lambda index: (deployment_losses[index], index))
    deployment_beta = float(beta_grid[deployment_index])

    base_sse = float(sum(value.sum() for value in oof_base))
    candidate_sse = float(sum(value.sum() for value in oof_candidate))
    total_sst = float(sum(target_sst(stats).sum() for stats in component_stats))
    uniform_base = []
    uniform_candidate = []
    per_layer = []
    for layer_index, stats in enumerate(component_stats):
        sst = target_sst(stats)
        nonzero = sst > 0
        r2_base = np.where(nonzero, 1.0 - oof_base[layer_index] / sst, np.where(oof_base[layer_index] == 0, 1.0, 0.0))
        r2_candidate = np.where(nonzero, 1.0 - oof_candidate[layer_index] / sst, np.where(oof_candidate[layer_index] == 0, 1.0, 0.0))
        uniform_base.append(float(r2_base.mean()))
        uniform_candidate.append(float(r2_candidate.mean()))
        per_layer.append({"index": layer_index, "base_sse": float(oof_base[layer_index].sum()), "candidate_sse": float(oof_candidate[layer_index].sum()), "base_uniform_r2": uniform_base[-1], "candidate_uniform_r2": uniform_candidate[-1]})

    fitted = []
    for stats in component_stats:
        scale, bias = fit_head_scale_channel_bias(stats, all_rows, bounds)
        effective_scale = 1.0 + deployment_beta * (scale - 1.0)
        effective_bias = deployment_beta * bias
        fitted.append({"scale": effective_scale.tolist(), "bias": effective_bias.tolist()})
    return {
        "outer_folds": outer,
        "beta_grid": [float(value) for value in beta_grid],
        "deployment_cv_sse_by_beta": [float(value) for value in deployment_losses],
        "deployment_beta": deployment_beta,
        "oof": {
            "base_sse": base_sse,
            "candidate_sse": candidate_sse,
            "relative_sse_reduction": (base_sse - candidate_sse) / base_sse,
            "base_global_r2": 1.0 - base_sse / total_sst,
            "candidate_global_r2": 1.0 - candidate_sse / total_sst,
            "base_uniform_r2": float(np.mean(uniform_base)),
            "candidate_uniform_r2": float(np.mean(uniform_candidate)),
            "per_layer": per_layer,
        },
        "fitted_parameters": fitted,
    }


def _mmap_activation(path: Path) -> Any:
    import torch

    expected = 50 * 256 * 8 * 128
    if path.stat().st_size != expected * 2:
        raise ProtocolError(f"activation size changed: {path}")
    raw = torch.from_file(str(path), shared=False, size=expected, dtype=torch.uint16)
    return raw.view(torch.bfloat16).reshape(50, 256, 8, 128)


def collect_stats(config: Mapping[str, Any], layer: int, component: str, tuning_ids: Sequence[int]) -> dict[str, Any]:
    import numpy as np
    import torch
    from safetensors.torch import load_file

    manifest = read_json(Path(config["upstream"]["shrinkage_manifest"]["path"]))
    entry = manifest["layers"][layer]
    if entry["target_layer"] != layer or len(entry["source_layers_rank_order"]) != 20:
        raise ProtocolError("mapper layer entry changed")
    selected = [int(value) for value in entry["source_layers_rank_order"]]
    source_component = "k_content" if component == "key" else "v"
    source_root = Path(config["upstream"]["source_activations"])
    target_root = Path(config["upstream"]["target_activations"])
    source = [_mmap_activation(source_root / source_component / f"layer_{value:03d}.bf16.bin") for value in selected]
    target = _mmap_activation(target_root / source_component / f"layer_{layer:03d}.bf16.bin")
    tensors = load_file(str(Path(config["upstream"]["shrinkage_root"]) / entry["path"]), device="cpu")
    weight = tensors[f"{component}_weight"].to(config["runtime"]["device"], dtype=torch.float32)
    bias = tensors[f"{component}_bias"].to(config["runtime"]["device"], dtype=torch.float32)
    arrays = {name: np.zeros((40, 8, 128), dtype=np.float64) for name in ("sum_p", "sum_y", "sum_p2", "sum_py", "sum_y2")}
    chunk = int(config["runtime"]["chunk_sequences"])
    with torch.inference_mode():
        for first in range(0, len(tuning_ids), chunk):
            global_ids = [int(value) for value in tuning_ids[first:first + chunk]]
            x_cpu = torch.stack([torch.cat([value[index] for index in global_ids], dim=0) for value in source], dim=1).reshape(len(global_ids) * 256, 20480)
            y_cpu = torch.cat([target[index] for index in global_ids], dim=0).reshape(len(global_ids) * 256, 1024)
            x = x_cpu.to(config["runtime"]["device"], dtype=torch.float32)
            y = y_cpu.to(config["runtime"]["device"], dtype=torch.float32)
            p = torch.addmm(bias, x, weight).reshape(len(global_ids), 256, 8, 128).to(torch.float64)
            y64 = y.reshape(len(global_ids), 256, 8, 128).to(torch.float64)
            sl = slice(first, first + len(global_ids))
            arrays["sum_p"][sl] = p.sum(1).cpu().numpy()
            arrays["sum_y"][sl] = y64.sum(1).cpu().numpy()
            arrays["sum_p2"][sl] = (p * p).sum(1).cpu().numpy()
            arrays["sum_py"][sl] = (p * y64).sum(1).cpu().numpy()
            arrays["sum_y2"][sl] = (y64 * y64).sum(1).cpu().numpy()
            del x_cpu, y_cpu, x, y, p, y64
    del source, target, tensors, weight, bias
    torch.cuda.empty_cache()
    return {"sequence_ids": np.asarray(tuning_ids, dtype=np.int64), **arrays}


def driver_state() -> dict[str, Any]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        rows.append({"index": int(fields[0]), "memory_used_mib": int(fields[1]), "utilization_percent": int(fields[2])})
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"], check=False, capture_output=True, text=True)
    return {"gpus": rows, "compute_processes": [line for line in processes.stdout.splitlines() if line.strip()]}


def idle_gate(config: Mapping[str, Any], run_root: Path) -> Any:
    lock_path = Path(config["runtime"]["global_gpu_lock"])
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        handle.close()
        raise ProtocolError("global GPU lock is held") from error
    samples = []
    try:
        for index in range(int(config["runtime"]["idle_gate_samples"])):
            state = driver_state()
            passed = len(state["gpus"]) == 8 and not state["compute_processes"] and max(row["memory_used_mib"] for row in state["gpus"]) <= int(config["runtime"]["idle_gate_max_memory_mib"])
            samples.append({"index": index, "passed": passed, **state})
            if not passed:
                raise ProtocolError("GPU idle gate failed")
            if index + 1 < int(config["runtime"]["idle_gate_samples"]):
                time.sleep(int(config["runtime"]["idle_gate_interval_seconds"]))
        atomic_json(run_root / "idle_gate.json", {"status": "pass", "samples": samples})
        return handle
    except BaseException:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        raise


def run(config_path: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    config_path = config_path.resolve()
    config = load_config(config_path)
    if Path(sys.executable).resolve() != Path(config["runtime"]["required_python"]).resolve():
        raise ProtocolError("Python executable changed")
    prefix = Path(config["runtime"]["required_pycache_prefix"])
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1" or not sys.dont_write_bytecode:
        raise ProtocolError("bytecode writes are not disabled")
    if os.environ.get("PYTHONPYCACHEPREFIX") != str(prefix) or sys.pycache_prefix != str(prefix) or os.path.lexists(prefix):
        raise ProtocolError("fresh pycache prefix gate failed")
    tuning, global_folds, locked = split_rows(config)
    if set(tuning) & set(locked):
        raise ProtocolError("locked rows leaked")
    input_validation = validate_scientific_inputs(config)
    local = {value: index for index, value in enumerate(tuning)}
    folds = [[local[value] for value in fold] for fold in global_folds]
    run_root = Path(config["outputs"]["run_root"])
    if run_root.exists():
        raise ProtocolError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)
    atomic_json(run_root / "input_preflight.json", input_validation)
    atomic_json(run_root / "run_definition.json", {"schema_version": 1, "status": "running", "config": artifact(config_path), "tuning_sequence_ids": tuning, "locked_sequence_ids_sha256": hashlib.sha256(canonical_bytes(locked)).hexdigest(), "locked_rows_indexed": False, "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    lock = idle_gate(config, run_root)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        all_stats: dict[str, list[dict[str, Any]]] = {"key": [], "value": []}
        for component in config["protocol"]["components"]:
            for layer in config["protocol"]["target_layers"]:
                stats = collect_stats(config, int(layer), str(component), tuning)
                path = run_root / "stats" / f"layer_{int(layer):03d}_{component}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, **stats)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                all_stats[str(component)].append(stats)
        results = {}
        for component in config["protocol"]["components"]:
            results[str(component)] = nested_cv(all_stats[str(component)], folds, config["protocol"]["beta_grid"], config["protocol"]["scale_bounds"])
        reductions = [results[name]["oof"]["relative_sse_reduction"] for name in config["protocol"]["components"]]
        gate = config["protocol"]["go_gate"]
        decision = "GO_GEOMETRY_ONLY" if all(value > 0.0 for value in reductions) and max(reductions) >= float(gate["minimum_relative_sse_reduction_in_at_least_one_component"]) else "NO_GO"
        receipt = {"schema_version": 1, "status": "complete", "decision": decision, "results": results, "locked_activation_evaluations": 0, "locked_rows_indexed": False, "downstream_claim_permitted": False, "stats_files": [artifact(path) for path in sorted((run_root / "stats").glob("*.npz"))], "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
        atomic_json(run_root / "results.json", receipt)
        atomic_json(run_root / "COMPLETED.json", {"status": "complete", "results": artifact(run_root / "results.json")})
        return receipt
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
