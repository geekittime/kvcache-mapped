"""Real eight-GPU executor for the decontaminated causal oracle.

Nothing in this module imports torch at module-import or CPU-preflight time.
The order is deliberate: final placeholders, complete G0-v2, the native
registry, train-only answer policy, all pinned files, the global lock, GPU
idleness, and deep model/mapper hashes are checked before torch is imported.
No GPU execution is authorized by the checked-in template.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import fcntl
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterator, Mapping, Sequence

from .contracts import (
    ContractError,
    ROOT,
    SelectedSample,
    artifact_ref,
    canonical_bytes,
    canonical_sha256,
    contract_preflight,
    ensure_formal_config_path,
    ensure_inside_output_root,
    intervention_catalog,
    load_config,
    prospective_nonsymlink_path,
    read_json,
    require_nonsymlink_directory,
    require_nonsymlink_file,
    sha256_file,
    verify_borrowed_interfaces,
)
from .validator import result_path, validate_atomic_result, validate_run


class GPUOracleError(RuntimeError):
    """The concrete runtime cannot preserve the preregistered oracle contract."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(_jsonable(value), newline=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _write_or_verify(path: Path, value: Any, *, label: str) -> None:
    normalized = _jsonable(value)
    if path.exists():
        if read_json(path) != normalized:
            raise GPUOracleError(f"existing {label} differs; refusing to overwrite {path}")
        return
    _atomic_json(path, normalized)


def _deep_model_identity(model_root: str | Path, expected_config_sha256: str) -> dict[str, Any]:
    root = require_nonsymlink_directory(model_root, label="model root")
    config_path = root / "config.json"
    if sha256_file(config_path) != expected_config_sha256:
        raise GPUOracleError(f"model config changed: {config_path}")
    index_path = root / "model.safetensors.index.json"
    index = read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise GPUOracleError(f"model index has no weight map: {index_path}")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and name.endswith(".safetensors") for name in shard_names):
        raise GPUOracleError(f"model index has malformed shard names: {index_path}")
    metadata_names = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    metadata = {}
    for name in metadata_names:
        path = root / name
        if not path.is_file():
            raise GPUOracleError(f"required model metadata is absent: {path}")
        metadata[name] = artifact_ref(path)
    shards = []
    for name in shard_names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise GPUOracleError(f"model shard name is not a safe relative path: {name}")
        path = require_nonsymlink_file(root / relative, label="model shard")
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise GPUOracleError(f"model shard escapes model root: {name}") from exc
        shards.append(artifact_ref(path))
    return {
        "path": str(root),
        "config_sha256": expected_config_sha256,
        "metadata": metadata,
        "weight_shards": shards,
        "weight_shards_deep_hashed": True,
        "weight_inventory_sha256": canonical_sha256(shards),
    }


def _deep_mapper_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    mapper = config["mapper"]
    manifest = require_nonsymlink_file(mapper["manifest"], label="k20 mapper manifest")
    if manifest.stat().st_size != mapper["manifest_nbytes"] or sha256_file(manifest) != mapper["manifest_sha256"]:
        raise GPUOracleError("k20 mapper manifest identity changed")
    root = require_nonsymlink_directory(manifest.parent, label="k20 mapper root")
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise GPUOracleError(f"mapper directory contains a symlink: {path}")
        if path.is_file():
            entry = artifact_ref(path)
            entry["relative_path"] = path.relative_to(root).as_posix()
            files.append(entry)
    if not files or not any(item["relative_path"] == "manifest.json" for item in files):
        raise GPUOracleError("deep mapper inventory is empty or lacks manifest.json")
    return {
        "root": str(root),
        "manifest": artifact_ref(manifest),
        "files": files,
        "all_regular_files_deep_hashed": True,
        "file_inventory_sha256": canonical_sha256(files),
    }


def _implementation_identity(config_path: Path) -> dict[str, Any]:
    paths = [require_nonsymlink_file(config_path, label="formal GPU config"), ROOT / "g0_consumer_contract.json", ROOT / "native_registry_consumer_contract.json"]
    paths.extend(sorted((ROOT / "src").glob("*.py")))
    paths.extend(sorted((ROOT / "scripts").glob("*.py")))
    entries = [artifact_ref(path) for path in paths]
    return {"files": entries, "inventory_sha256": canonical_sha256(entries)}


def _nvidia_memory_used_mib() -> list[int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise GPUOracleError(f"cannot query GPU idleness with nvidia-smi: {exc}") from exc
    values: dict[int, int] = {}
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise GPUOracleError(f"malformed nvidia-smi row: {line!r}")
        try:
            index, used = int(fields[0]), int(fields[1])
        except ValueError as exc:
            raise GPUOracleError(f"non-integer nvidia-smi row: {line!r}") from exc
        values[index] = used
    if sorted(values) != list(range(8)):
        raise GPUOracleError(f"exactly physical GPUs 0..7 are required, observed {sorted(values)}")
    return [values[index] for index in range(8)]


def _wait_for_idle_gpus(runtime: Mapping[str, Any]) -> dict[str, Any]:
    limit = runtime["idle_memory_limit_mib"]
    required = runtime["idle_consecutive_samples"]
    interval = runtime["idle_sample_interval_seconds"]
    timeout = runtime["idle_timeout_seconds"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (limit, required, interval, timeout)):
        raise GPUOracleError("GPU idle-gate parameters must be positive integers")
    start = time.monotonic()
    consecutive = 0
    final_window: list[dict[str, Any]] = []
    while True:
        memory = _nvidia_memory_used_mib()
        passed = all(value <= limit for value in memory)
        consecutive = consecutive + 1 if passed else 0
        final_window.append({"elapsed_seconds": time.monotonic() - start, "memory_used_mib": memory, "passed": passed})
        final_window = final_window[-required:]
        if consecutive >= required:
            return {
                "status": "passed",
                "limit_mib": limit,
                "required_consecutive_samples": required,
                "observed_consecutive_samples": consecutive,
                "sample_interval_seconds": interval,
                "final_window": final_window,
            }
        if time.monotonic() - start >= timeout:
            raise GPUOracleError("timed out waiting for eight idle GPUs")
        time.sleep(interval)


@contextmanager
def _global_gpu_lock(path: str | Path) -> Iterator[None]:
    lock_path = prospective_nonsymlink_path(path, label="global GPU lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _import_verified_legacy(config: Mapping[str, Any]) -> tuple[Any, Any]:
    """Import the read-only dir01 interfaces only after rechecking their hashes."""

    refs = verify_borrowed_interfaces(config)
    source_dirs = {str(Path(spec["path"]).parent) for spec in refs.values()}
    if len(source_dirs) != 1:
        raise GPUOracleError("borrowed dir01 interfaces do not share one pinned source directory")
    source_dir = next(iter(source_dirs))
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)
    cache_oracle = importlib.import_module("cache_oracle")
    gpu_oracle = importlib.import_module("gpu_oracle")
    if require_nonsymlink_file(cache_oracle.__file__, label="imported cache_oracle") != require_nonsymlink_file(refs["cache_oracle"]["path"], label="pinned cache_oracle"):
        raise GPUOracleError("cache_oracle was imported from an unpinned location")
    if require_nonsymlink_file(gpu_oracle.__file__, label="imported gpu_oracle") != require_nonsymlink_file(refs["gpu_oracle"]["path"], label="pinned gpu_oracle"):
        raise GPUOracleError("gpu_oracle was imported from an unpinned location")
    observed = [item.to_dict() for item in cache_oracle.intervention_catalog()]
    expected = intervention_catalog()
    if observed != expected:
        raise GPUOracleError("borrowed causal intervention catalog differs from dir13 preregistration")
    return cache_oracle, gpu_oracle


def _serialize_and_tokenize(tokenizer: Any, sample: SelectedSample, prompt_config: Mapping[str, Any]) -> dict[str, Any]:
    numbered = "\n\n".join(
        f"Passage {index + 1} — {title}\n{text}"
        for index, (title, text) in enumerate(sample.passages)
    )
    user = f"{numbered}\n\nQuestion: {sample.question}"
    messages = [
        {"role": "system", "content": prompt_config["system_text"]},
        {"role": "user", "content": user},
    ]
    serialized = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(serialized, str) or not serialized:
        raise GPUOracleError("target tokenizer returned an empty serialized prompt")
    token_ids = tokenizer.encode(serialized, add_special_tokens=False)
    if not isinstance(token_ids, list) or any(type(item) is not int or item < 0 for item in token_ids):
        raise GPUOracleError("target tokenizer returned malformed prompt token IDs")
    if not 2 <= len(token_ids) <= prompt_config["maximum_serialized_tokens"]:
        raise GPUOracleError("serialized prompt is empty/oversized; truncation is forbidden")
    gold: set[int] = set()
    for reference in sample.references:
        for variant in (reference, " " + reference):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if ids:
                gold.add(int(ids[0]))
    if not gold:
        raise GPUOracleError("no answer alias has a valid first target token")
    prefix = token_ids[:-1]
    binding = {
        **sample.public_binding(),
        "prompt_token_count": len(token_ids),
        "serialized_prompt_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": canonical_sha256(token_ids),
        "prefix_token_ids_sha256": canonical_sha256(prefix),
        "references_sha256": canonical_sha256(list(sample.references)),
        "gold_first_token_ids_sha256": canonical_sha256(sorted(gold)),
    }
    return {
        "sample": sample,
        "serialized_prompt": serialized,
        "token_ids": tuple(token_ids),
        "gold_first_token_ids": tuple(sorted(gold)),
        "binding": binding,
    }


def _tokenized_plans(tokenizer: Any, samples: Sequence[SelectedSample], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt = config["prompt"]
    if prompt.get("schema") != "qa_passages_chat_v1" or prompt.get("enable_thinking") is not False:
        raise GPUOracleError("prompt schema/thinking policy drifted")
    if prompt.get("truncation_forbidden") is not True or prompt.get("maximum_serialized_tokens") != 8192:
        raise GPUOracleError("prompt truncation contract drifted")
    return [_serialize_and_tokenize(tokenizer, sample, prompt) for sample in samples]


def _execute_prompt(
    *,
    plan: Mapping[str, Any],
    request_fingerprint: str,
    config: Mapping[str, Any],
    tokenizer: Any,
    target: Any,
    builder: Any,
    cache_oracle: Any,
    gpu_oracle: Any,
) -> dict[str, Any]:
    """Materialize native and mapped prefix once, then replay all 11 branches."""

    torch = sys.modules.get("torch")
    if torch is None:
        raise GPUOracleError("torch is unavailable after verified GPU import")
    from kvcache_transfer.cache_adapter import boundary_replay_inputs, cache_to_legacy

    sample: SelectedSample = plan["sample"]
    token_ids = list(plan["token_ids"])
    prefix = token_ids[:-1]
    context = cache_oracle.CacheContext.from_prompt(
        dataset=sample.dataset,
        sample_id=sample.sample_id,
        serialized_prompt=plan["serialized_prompt"],
        token_ids=token_ids,
        target_config_sha256=config["models"]["target_config_sha256"],
    )
    base = getattr(target, "model", target)
    device = gpu_oracle._device(target)
    full = base(**gpu_oracle._full_inputs(token_ids, device))
    native_prefix = base(**gpu_oracle._full_inputs(prefix, device))
    native_envelope = cache_oracle.wrap_cache(
        cache_to_legacy(native_prefix.past_key_values),
        context=context,
        provenance=cache_oracle.CacheProvenance(
            "native",
            "Qwen3-14B exact C[:-1]",
            config["models"]["target_config_sha256"],
        ),
    )
    boundary = torch.tensor([[context.boundary_token_id]], dtype=torch.long, device=device)
    with gpu_oracle._capture_attn_inputs(target) as captures:
        native_replay = target(
            **boundary_replay_inputs(boundary, native_prefix.past_key_values),
            return_dict=True,
        )
    gpu_oracle._assert_boundary_identity(full, native_replay, context.prefix_length)
    native_queries, boundary_keys, boundary_values = gpu_oracle._native_boundary_qkv(
        target, captures, context.prefix_length
    )
    for layer, (key, value) in enumerate(cache_to_legacy(native_replay.past_key_values)):
        if not (
            torch.allclose(key[..., -1:, :].float(), boundary_keys[layer].float(), rtol=2e-3, atol=2e-3)
            and torch.allclose(value[..., -1:, :].float(), boundary_values[layer].float(), rtol=2e-3, atol=2e-3)
        ):
            raise GPUOracleError(f"native post-RoPE boundary K/V identity failed at layer {layer}")

    # The only mapped-cache build for this prompt.  Every intervention below
    # composes a cloned branch from this same materialization.
    mapped_cache = builder(prefix)
    mapped_materializations = 1
    mapped_envelope = cache_oracle.wrap_cache(
        cache_to_legacy(mapped_cache),
        context=context,
        provenance=cache_oracle.CacheProvenance(
            "mapped",
            "Qwen3-32B exact C[:-1] then frozen NVIDIA k20",
            config["models"]["source_config_sha256"],
            config["mapper"]["manifest_sha256"],
        ),
    )
    reference = None
    paired: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    compositions: list[dict[str, Any]] = []
    for intervention in cache_oracle.intervention_catalog():
        composed, composition = cache_oracle.compose_legacy_cache(
            native_envelope,
            mapped_envelope,
            intervention,
            clone=True,
        )
        observation, trace = gpu_oracle._observation(
            target,
            tokenizer,
            context,
            composed,
            native_queries,
            boundary_keys,
            boundary_values,
        )
        if intervention.id == "native_k_native_v":
            reference = observation
        if reference is None:
            raise GPUOracleError("native K/native V reference was not executed first")
        paired.append(
            cache_oracle.score_paired_observations(
                context=context,
                intervention=intervention,
                references=sample.references,
                gold_first_token_ids=plan["gold_first_token_ids"],
                native=reference,
                candidate=observation,
            )
        )
        traces.append({"intervention_id": intervention.id, **trace})
        compositions.append(composition.to_dict())
    if mapped_materializations != 1:
        raise GPUOracleError("internal k20 materialization counter drifted")
    gpu_oracle._sync()
    binding = plan["binding"]
    return {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": request_fingerprint,
        **binding,
        "causal_boundary": {
            "prefix": "C[:-1]",
            "replay": "C[-1]",
            "full_prompt_identity_passed": True,
            "native_boundary_qkv_identity_passed": True,
        },
        "exact_cache_envelope": {
            "layers": 40,
            "batch": 1,
            "kv_heads": 8,
            "head_dim": 128,
            "prefix_length": context.prefix_length,
            "positions": "contiguous_zero_based_C_prefix",
            "key_space": "post_target_qwen3_rope_cache",
            "value_space": "target_cache_value_space",
            "finite_checked": True,
            "native_mapped_shape_dtype_device_equal": True,
            "context_sha256": cache_oracle.canonical_sha256(asdict(context)),
        },
        "mapped_cache_materializations": mapped_materializations,
        "paired": paired,
        "intervention_generation": traces,
        "compositions": compositions,
        "systems": {"gpu_peak": gpu_oracle._peaks()},
    }


def preflight(config_path: str | Path, *, profile: str) -> dict[str, Any]:
    report, _samples = contract_preflight(config_path, profile=profile)
    return report


def execute(config_path: str | Path, *, profile: str, output_dir: str | Path) -> int:
    """Run the frozen oracle; the checked-in template always refuses this call."""

    config_path = ensure_formal_config_path(config_path)
    config = load_config(config_path)
    preflight_report, samples = contract_preflight(config_path, profile=profile)
    output = ensure_inside_output_root(config, output_dir)
    if (output / "completion_receipt.json").is_file():
        validate_run(config_path, profile=profile, output_dir=output)
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["runtime"]["required_cuda_visible_devices"]:
        raise GPUOracleError("CUDA_VISIBLE_DEVICES must be exactly 0,1,2,3,4,5,6,7")
    output.mkdir(parents=True, exist_ok=True)
    with _global_gpu_lock(config["runtime"]["global_gpu_lock"]):
        _atomic_json(output / "status.json", {"status": "waiting_for_idle_gpus", "gpu_used": False})
        idle = _wait_for_idle_gpus(config["runtime"])
        idle_path = output / "gpu_idle_gates" / f"{time.time_ns()}.json"
        _atomic_json(idle_path, idle)
        deep_identity = {
            "schema_version": 1,
            "status": "complete_before_torch_import",
            "source_32b": _deep_model_identity(
                config["models"]["source_32b"], config["models"]["source_config_sha256"]
            ),
            "target_14b": _deep_model_identity(
                config["models"]["target_14b"], config["models"]["target_config_sha256"]
            ),
            "mapper_k20": _deep_mapper_identity(config),
            "borrowed_read_only_interfaces": verify_borrowed_interfaces(config),
            "implementation": _implementation_identity(config_path),
            "torch_imported_when_written": "torch" in sys.modules,
            "cuda_api_called_when_written": False,
        }
        if deep_identity["torch_imported_when_written"] is not False:
            raise GPUOracleError("torch was imported before deep input identity was built")
        _write_or_verify(output / "deep_input_identity.json", deep_identity, label="deep input identity")

        cache_oracle, gpu_oracle = _import_verified_legacy(config)
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
            raise GPUOracleError("exactly eight visible CUDA devices are required")
        tokenizer, source, target, builder = gpu_oracle._load_components(config)
        plans = _tokenized_plans(tokenizer, samples, config)
        runtime_provenance = {
            "schema_version": 1,
            "status": "loaded",
            "source_devices": config["models"]["source_devices"],
            "source_layer_counts": config["models"]["source_layer_counts"],
            "target_devices": config["models"]["target_devices"],
            "target_layer_counts": config["models"]["target_layer_counts"],
            "source_sharding_evidence": _jsonable(getattr(source, "_kvcache_transfer_layer_sharding_evidence")),
            "target_sharding_evidence": _jsonable(getattr(target, "_kvcache_transfer_layer_sharding_evidence")),
            "mapper_builder_provenance": _jsonable(builder.provenance),
            "k20_materialization_policy": "exactly_once_per_prompt_then_clone_11_compositions",
            "borrowed_interfaces_are_read_only": True,
            "output_root": str(output),
        }
        _write_or_verify(output / "runtime_provenance.json", runtime_provenance, label="runtime provenance")
        definition_unsigned = {
            "schema_version": 1,
            "status": "frozen_run_definition",
            "config": artifact_ref(config_path),
            "profile": profile,
            "preflight": preflight_report,
            "selected_bindings_sha256": preflight_report["selected_bindings_sha256"],
            "samples": [plan["binding"] for plan in plans],
            "deep_input_identity": artifact_ref(output / "deep_input_identity.json"),
            "runtime_provenance": artifact_ref(output / "runtime_provenance.json"),
            "intervention_ids": [item["id"] for item in intervention_catalog()],
            "mapped_cache_materializations_per_prompt": 1,
        }
        fingerprint = canonical_sha256(definition_unsigned)
        definition = {**definition_unsigned, "request_fingerprint": fingerprint}
        _write_or_verify(output / "run_definition.json", definition, label="run definition")
        _atomic_json(
            output / "status.json",
            {"status": "running", "request_fingerprint": fingerprint, "completed_prompts": 0, "total_prompts": len(plans)},
        )
        completed = 0
        for plan in plans:
            destination = result_path(output, plan["binding"])
            if destination.exists():
                validate_atomic_result(
                    destination,
                    request_fingerprint=fingerprint,
                    plan_binding=plan["binding"],
                )
                completed += 1
                continue
            try:
                payload = _execute_prompt(
                    plan=plan,
                    request_fingerprint=fingerprint,
                    config=config,
                    tokenizer=tokenizer,
                    target=target,
                    builder=builder,
                    cache_oracle=cache_oracle,
                    gpu_oracle=gpu_oracle,
                )
                _atomic_json(destination, payload)
                validate_atomic_result(
                    destination,
                    request_fingerprint=fingerprint,
                    plan_binding=plan["binding"],
                )
            except BaseException as exc:
                failure = output / "failures" / plan["binding"]["dataset"] / (
                    f"{plan['binding']['sample_id']}.{time.time_ns()}.json"
                )
                _atomic_json(
                    failure,
                    {
                        "status": "failed",
                        "request_fingerprint": fingerprint,
                        "sample": plan["binding"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                raise
            finally:
                torch.cuda.empty_cache()
            completed += 1
            _atomic_json(
                output / "status.json",
                {
                    "status": "running",
                    "request_fingerprint": fingerprint,
                    "completed_prompts": completed,
                    "total_prompts": len(plans),
                },
            )
        inventory = [artifact_ref(result_path(output, plan["binding"])) for plan in plans]
        receipt = {
            "schema_version": 1,
            "status": "complete",
            "request_fingerprint": fingerprint,
            "atomic_prompt_results": len(plans),
            "paired_metric_rows": len(plans) * 11,
            "result_inventory": inventory,
            "result_inventory_sha256": canonical_sha256(inventory),
            "method_dev_train_only": True,
            "locked_outcomes_opened": False,
            "answers_used_for_selection": False,
            "answers_used_only_for": ["gold_first_token_margin", "qa_f1", "qa_em"],
            "mapped_cache_materializations_per_prompt": 1,
            "final_session_idle_gate": artifact_ref(idle_path),
            "run_artifacts": {
                "deep_input_identity": artifact_ref(output / "deep_input_identity.json"),
                "runtime_provenance": artifact_ref(output / "runtime_provenance.json"),
                "run_definition": artifact_ref(output / "run_definition.json"),
            },
        }
        _write_or_verify(output / "completion_receipt.json", receipt, label="completion receipt")
        _atomic_json(
            output / "status.json",
            {"status": "complete", "request_fingerprint": fingerprint, "completed_prompts": len(plans)},
        )
    return 0


__all__ = ["GPUOracleError", "execute", "preflight"]
