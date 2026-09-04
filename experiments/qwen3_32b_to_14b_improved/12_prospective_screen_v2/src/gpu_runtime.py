"""Real Qwen3-32B/14B runtime construction for the four prospective raw modes.

No CUDA module is imported at module import time.  All receipts, mapper files,
model identities, and code hashes can therefore be validated before the first
CUDA API call.  ``build_runtime_components`` is the only model-loading entry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

from .contracts import (
    ArtifactRef,
    ContractError,
    canonical_sha256,
    read_json,
    require_child,
)
from .runtime_binding import validate_runtime_provenance


REPOSITORY_ROOT = Path("/home/wby/llm-cache/nvidia-kvcache")
REPOSITORY_SRC = REPOSITORY_ROOT / "src"
BASELINE_EXPERIMENT = REPOSITORY_ROOT / "experiments" / "qwen3_32b_to_14b"


@dataclass(frozen=True)
class MethodBindings:
    runtime: Mapping[str, Any]
    baseline_manifest: ArtifactRef
    shrinkage: Mapping[str, Any]
    tail_refresh_r32: Mapping[str, Any]
    mapper_payload_validation: Mapping[str, Any]
    deep_mapper_hash_verified: bool
    mode_digests: Mapping[str, str]
    digest: str


@dataclass(frozen=True)
class RuntimeComponents:
    tokenizer: Any
    source_model: Any
    target_model: Any
    engines: Mapping[str, Any]
    builders: Mapping[str, Any]
    method_provenance: Mapping[str, Any]
    model_sharding: Mapping[str, Any]


@dataclass(frozen=True)
class AuditedGenerationResult:
    """Local generation result with the excluded terminal token preserved."""

    text: str
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    terminal_token_id: int | None
    cap_hit: bool
    requested_max_tokens: int
    original_context_tokens: int
    used_context_tokens: int
    truncation: str


def _eos_id_values(value: Any, *, label: str) -> set[int]:
    if type(value) is int and value >= 0:
        return {value}
    if (
        isinstance(value, list)
        and value
        and all(type(item) is int and item >= 0 for item in value)
    ):
        return set(value)
    raise ContractError(f"{label} EOS token IDs are malformed")


def frozen_target_eos_token_ids(model_root: str | Path) -> tuple[int, ...]:
    """Derive the exact EOS set from content-hashed target model metadata."""

    root = Path(model_root).resolve()
    values: set[int] = set()
    for filename in ("config.json", "generation_config.json"):
        payload = read_json(root / filename)
        if not isinstance(payload, Mapping):
            raise ContractError(f"target {filename} root is malformed")
        values.update(
            _eos_id_values(
                payload.get("eos_token_id"), label=f"target {filename}"
            )
        )
    if not values:  # pragma: no cover - _eos_id_values is already non-empty
        raise ContractError("frozen target EOS token set is empty")
    return tuple(sorted(values))


def _observed_runtime_eos_token_ids(tokenizer: Any, target_model: Any) -> tuple[int, ...]:
    values = _eos_id_values(
        getattr(tokenizer, "eos_token_id", None), label="runtime tokenizer"
    )
    values.update(
        _eos_id_values(
            getattr(getattr(target_model, "generation_config", None), "eos_token_id", None),
            label="runtime target generation config",
        )
    )
    return tuple(sorted(values))


def _audited_engine_class(base_class: type[Any], evaluation_module: Any) -> type[Any]:
    """Extend the frozen engine loop only to retain its observed EOS argmax.

    This is a local implementation rather than a post-hoc inference from the
    upstream ``stop_reason``.  All causal-cache helpers are still taken from
    the runtime-provenance-bound core module, while this file is independently
    covered by the prospective code inventory.
    """

    torch_module = evaluation_module.torch

    class AuditedTransferEvaluationEngine(base_class):
        @torch_module.inference_mode()
        def generate_token_ids(
            self,
            context_token_ids: Sequence[int],
            gen_kwargs: Mapping[str, Any],
            *,
            truncation: str = "suffix",
        ) -> AuditedGenerationResult:
            max_tokens, stop_strings = self._generation_kwargs(gen_kwargs)
            if max_tokens >= self.max_length:
                raise ValueError("max_gen_toks must be smaller than max_length")
            if truncation not in {"suffix", "head_tail", "error"}:
                raise ValueError("truncation must be suffix, head_tail, or error")
            context_ids = [int(token) for token in context_token_ids]
            original_context_tokens = len(context_ids)
            if not context_ids:
                context_ids = [self.prefix_token_id]
            context_budget = self.max_length - max_tokens
            if len(context_ids) > context_budget:
                if truncation == "suffix":
                    context_ids = context_ids[-context_budget:]
                elif truncation == "head_tail":
                    context_ids, _ = evaluation_module.head_tail_truncate_token_ids(
                        context_ids, context_budget
                    )
                else:
                    raise ValueError(
                        f"generation context has {len(context_ids)} tokens, exceeding "
                        f"budget={context_budget}"
                    )

            if self.mode in evaluation_module.STANDALONE_MODES or len(context_ids) == 1:
                ids = torch_module.tensor(
                    [context_ids], dtype=torch_module.long, device=self.device
                )
                output = self.target_model(
                    input_ids=ids,
                    attention_mask=torch_module.ones_like(ids),
                    position_ids=torch_module.arange(
                        ids.shape[1], device=self.device
                    ).unsqueeze(0),
                    cache_position=torch_module.arange(
                        ids.shape[1], device=self.device
                    ),
                    use_cache=True,
                    return_dict=True,
                )
            else:
                parent = self._parent_prefix_cache(context_ids[:-1])
                boundary = torch_module.tensor(
                    [[context_ids[-1]]], dtype=torch_module.long, device=self.device
                )
                inputs = evaluation_module.boundary_replay_inputs(
                    boundary, parent, clone_cache=True
                )
                output = self.target_model(**inputs, return_dict=True)

            raw_frozen_eos = getattr(
                self.tokenizer, "_prospective_frozen_eos_token_ids", None
            )
            if (
                not isinstance(raw_frozen_eos, tuple)
                or not raw_frozen_eos
                or any(type(item) is not int or item < 0 for item in raw_frozen_eos)
                or tuple(sorted(set(raw_frozen_eos))) != raw_frozen_eos
            ):
                raise ContractError("runtime engine lacks frozen EOS token IDs")
            eos_ids = set(raw_frozen_eos)
            generated: list[int] = []
            for step in range(max_tokens):
                next_token = int(output.logits[:, -1, :].argmax(-1).item())
                reached_eos = next_token in eos_ids
                if not reached_eos:
                    generated.append(next_token)
                if stop_strings:
                    text = evaluation_module._tokenizer_decode(
                        self.tokenizer, generated
                    )
                    stop_at = min(
                        (text.find(stop) for stop in stop_strings if stop in text),
                        default=-1,
                    )
                    if stop_at >= 0:
                        return AuditedGenerationResult(
                            text=text[:stop_at],
                            generated_token_ids=tuple(generated),
                            stop_reason="stop_sequence",
                            terminal_token_id=None,
                            cap_hit=False,
                            requested_max_tokens=max_tokens,
                            original_context_tokens=original_context_tokens,
                            used_context_tokens=len(context_ids),
                            truncation=truncation,
                        )
                if reached_eos:
                    return AuditedGenerationResult(
                        text=evaluation_module._tokenizer_decode(
                            self.tokenizer, generated
                        ),
                        generated_token_ids=tuple(generated),
                        stop_reason="eos",
                        terminal_token_id=next_token,
                        cap_hit=False,
                        requested_max_tokens=max_tokens,
                        original_context_tokens=original_context_tokens,
                        used_context_tokens=len(context_ids),
                        truncation=truncation,
                    )
                if step + 1 == max_tokens:
                    break

                past = output.past_key_values
                prefix_length = evaluation_module.cache_sequence_length(past)
                token = torch_module.tensor(
                    [[next_token]], dtype=torch_module.long, device=self.device
                )
                attention_mask = evaluation_module.full_attention_mask(
                    prefix_length, 1, batch_size=1, device=self.device
                )
                output = self.target_model(
                    input_ids=token,
                    past_key_values=past,
                    attention_mask=attention_mask,
                    position_ids=evaluation_module.replay_position_ids(
                        attention_mask, 1
                    ),
                    cache_position=evaluation_module.replay_cache_position(
                        prefix_length, 1, device=self.device
                    ),
                    use_cache=True,
                    return_dict=True,
                )
            return AuditedGenerationResult(
                text=evaluation_module._tokenizer_decode(self.tokenizer, generated),
                generated_token_ids=tuple(generated),
                stop_reason="max_tokens",
                terminal_token_id=None,
                cap_hit=True,
                requested_max_tokens=max_tokens,
                original_context_tokens=original_context_tokens,
                used_context_tokens=len(context_ids),
                truncation=truncation,
            )

    AuditedTransferEvaluationEngine.__name__ = "AuditedTransferEvaluationEngine"
    return AuditedTransferEvaluationEngine


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _mapper_payload_inventory(
    reference: ArtifactRef, *, label: str, deep_hash: bool
) -> dict[str, Any]:
    manifest = read_json(reference.path)
    layers = manifest.get("layers") if isinstance(manifest, Mapping) else None
    if not isinstance(layers, list) or len(layers) != 40:
        raise ContractError(f"{label} mapper must contain 40 layers")
    root = Path(reference.path).resolve().parent
    inventory: list[dict[str, Any]] = []
    observed_layers: set[int] = set()
    for raw in layers:
        entry = _mapping(raw, f"{label} layer")
        layer = entry.get("target_layer")
        if type(layer) is not int or not 0 <= layer < 40 or layer in observed_layers:
            raise ContractError(f"{label} mapper target-layer coverage changed")
        observed_layers.add(layer)
        row: dict[str, Any] = {"target_layer": layer}
        for path_field, sha_field, output_prefix in (
            ("path", "sha256", "payload"),
            ("sidecar_path", "sidecar_sha256", "sidecar"),
        ):
            raw_path = entry.get(path_field)
            if not isinstance(raw_path, str) or not raw_path:
                raise ContractError(f"{label} mapper {path_field} is malformed")
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            path = require_child(root, candidate, label=f"{label} mapper {path_field}")
            if not path.is_file():
                raise ContractError(f"{label} mapper payload is missing: {path}")
            payload_ref = ArtifactRef.parse(
                {
                    "path": str(path),
                    "nbytes": path.stat().st_size,
                    "sha256": entry.get(sha_field),
                },
                label=f"{label} mapper {path_field}",
            )
            payload_ref.verify(
                label=f"{label} mapper {path_field}", deep_hash=deep_hash
            )
            row[f"{output_prefix}_path"] = str(path.relative_to(root))
            row[f"{output_prefix}_nbytes"] = payload_ref.nbytes
            row[f"{output_prefix}_sha256"] = payload_ref.sha256
        inventory.append(row)
    if observed_layers != set(range(40)):
        raise ContractError(f"{label} mapper does not cover target layers 0..39")
    inventory.sort(key=lambda item: item["target_layer"])
    return {
        "manifest": reference.to_dict(),
        "layer_count": 40,
        "artifact_count": 80,
        "inventory_sha256": canonical_sha256(inventory),
        "total_nbytes": sum(
            item["payload_nbytes"] + item["sidecar_nbytes"] for item in inventory
        ),
        "deep_hash_verified": deep_hash,
    }


def _verify_shrinkage(config: Mapping[str, Any]) -> dict[str, Any]:
    method = _mapping(config["methods"]["shrinkage"], "shrinkage method")
    manifest_ref = ArtifactRef.parse(method.get("mapper_artifact"), label="shrinkage mapper")
    completion_ref = ArtifactRef.parse(method.get("completion_receipt"), label="shrinkage completion")
    algebra_ref = ArtifactRef.parse(method.get("algebra_receipt"), label="shrinkage algebra")
    interface_ref = ArtifactRef.parse(method.get("runtime_interface"), label="shrinkage interface")
    for label, reference in (
        ("shrinkage mapper", manifest_ref),
        ("shrinkage completion", completion_ref),
        ("shrinkage algebra", algebra_ref),
        ("shrinkage interface", interface_ref),
    ):
        reference.verify(label=label)
    manifest = read_json(manifest_ref.path)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("direction") != "qwen3_32b_to_14b"
        or manifest.get("method")
        != "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers"
        or len(manifest.get("layers", [])) != 40
    ):
        raise ContractError("shrinkage mapper manifest contract changed")
    k20 = _mapping(_mapping(manifest.get("upstream"), "shrinkage upstream").get("k20_manifest"), "shrinkage k20")
    baseline = config["methods"]["nvidia_k20"]["mapper_artifact"]
    if (
        Path(str(k20.get("path", ""))).resolve() != Path(baseline["path"]).resolve()
        or k20.get("sha256") != baseline["sha256"]
    ):
        raise ContractError("shrinkage mapper is not anchored to paired NVIDIA k20")
    completion = read_json(completion_ref.path)
    if (
        not isinstance(completion, Mapping)
        or completion.get("status") != "complete_by_append_only_amendment"
        or completion.get("amendment_scope") != "receipt-serialization-only"
        or completion.get("mapper", {}).get("manifest_sha256") != manifest_ref.sha256
        or completion.get("mapper", {}).get("payloads_deep_hashed") is not True
    ):
        raise ContractError("shrinkage completion receipt changed")
    execution = _mapping(completion.get("execution"), "shrinkage amendment execution")
    forbidden_true = (
        "activation_recomputed",
        "alpha_modified",
        "cuda_invoked",
        "locked_audit_recomputed",
        "mapper_modified",
    )
    if any(execution.get(field) is not False for field in forbidden_true):
        raise ContractError("shrinkage completion amendment exceeded serialization scope")
    algebra = read_json(algebra_ref.path)
    if (
        not isinstance(algebra, Mapping)
        or algebra.get("schema_version") != 1
        or algebra.get("status") != "complete"
        or algebra.get("cuda_invoked") is not False
        or algebra.get("amendment_v1_receipt", {}).get("sha256") != completion_ref.sha256
    ):
        raise ContractError("shrinkage algebra receipt changed")
    runtime_entries = [
        item
        for item in algebra.get("code", {}).get("files", [])
        if isinstance(item, Mapping) and item.get("path") == "src/runtime.py"
    ]
    if len(runtime_entries) != 1 or (
        runtime_entries[0].get("sha256") != interface_ref.sha256
        or runtime_entries[0].get("nbytes") != interface_ref.nbytes
    ):
        raise ContractError("shrinkage algebra does not bind runtime interface")
    value = {
        "method": dict(method),
        "manifest": manifest_ref.to_dict(),
        "completion": completion_ref.to_dict(),
        "algebra": algebra_ref.to_dict(),
        "interface": interface_ref.to_dict(),
    }
    return {**value, "binding_digest": canonical_sha256(value)}


def _verify_tail_refresh(config: Mapping[str, Any]) -> dict[str, Any]:
    method = _mapping(config["methods"]["tail_refresh_r32"], "R32 method")
    completion_ref = ArtifactRef.parse(method.get("completion_receipt"), label="R32 preflight")
    interface_receipt_ref = ArtifactRef.parse(method.get("interface_receipt"), label="R32 interface receipt")
    interface_ref = ArtifactRef.parse(method.get("runtime_interface"), label="R32 runtime interface")
    for label, reference in (
        ("R32 preflight", completion_ref),
        ("R32 interface receipt", interface_receipt_ref),
        ("R32 runtime interface", interface_ref),
    ):
        reference.verify(label=label)
    completion = read_json(completion_ref.path)
    if (
        not isinstance(completion, Mapping)
        or completion.get("schema_version") != 2
        or completion.get("status") != "passed"
        or completion.get("gpu_used") is not False
        or completion.get("cuda_execution_started") is not False
        or completion.get("tests", {}).get("failed") != 0
        or completion.get("implementation", {}).get("sha256") != interface_ref.sha256
        or completion.get("implementation", {}).get("nbytes") != interface_ref.nbytes
        or completion.get("prepared_v2", {}).get("interface_sha256") != interface_receipt_ref.sha256
    ):
        raise ContractError("R32 CPU preflight receipt changed")
    interface_receipt = read_json(interface_receipt_ref.path)
    if (
        not isinstance(interface_receipt, Mapping)
        or interface_receipt.get("schema_version") != 1
        or interface_receipt.get("status") != "frozen_method_dev_interface"
        or interface_receipt.get("builder_class") != method.get("symbol")
        or interface_receipt.get("implementation") != interface_ref.to_dict()
        or interface_receipt.get("constructor", {}).get("replay_tokens_allowed") != [1, 8, 32]
        or interface_receipt.get("causal_contract", {}).get("builder_native_tail_count") != "R-1"
        or interface_receipt.get("causal_contract", {}).get("engine_boundary_count") != 1
    ):
        raise ContractError("R32 interface receipt changed")
    value = {
        "method": dict(method),
        "completion": completion_ref.to_dict(),
        "interface_receipt": interface_receipt_ref.to_dict(),
        "interface": interface_ref.to_dict(),
        "underlying_k20": config["methods"]["nvidia_k20"]["mapper_artifact"],
    }
    return {**value, "binding_digest": canonical_sha256(value)}


def validate_method_bindings(
    config: Mapping[str, Any], *, deep_model_hash: bool = False, verify_environment: bool = False
) -> MethodBindings:
    runtime = validate_runtime_provenance(
        config,
        deep_model_hash=deep_model_hash,
        verify_environment=verify_environment,
    )
    baseline = ArtifactRef.parse(
        config["methods"]["nvidia_k20"]["mapper_artifact"], label="NVIDIA k20 mapper"
    )
    baseline.verify(label="NVIDIA k20 mapper")
    shrinkage = _verify_shrinkage(config)
    tail = _verify_tail_refresh(config)
    mapper_payload_validation = {
        "nvidia_k20": _mapper_payload_inventory(
            baseline, label="NVIDIA k20", deep_hash=deep_model_hash
        ),
        "shrinkage": _mapper_payload_inventory(
            ArtifactRef.parse(
                config["methods"]["shrinkage"]["mapper_artifact"],
                label="shrinkage mapper",
            ),
            label="shrinkage",
            deep_hash=deep_model_hash,
        ),
    }
    mode_digests = {
        "native14": canonical_sha256(
            {"target_model": runtime["target_model"], "method": config["methods"]["native14"]}
        ),
        "nvidia_k20": canonical_sha256(
            {"runtime": runtime["binding_digest"], "mapper": baseline.to_dict()}
        ),
        "shrinkage": shrinkage["binding_digest"],
        "tail_refresh_r32": tail["binding_digest"],
    }
    value = {
        "runtime": runtime["binding_digest"],
        "baseline": baseline.to_dict(),
        "mode_digests": mode_digests,
    }
    return MethodBindings(
        runtime=runtime,
        baseline_manifest=baseline,
        shrinkage=shrinkage,
        tail_refresh_r32=tail,
        mapper_payload_validation=mapper_payload_validation,
        deep_mapper_hash_verified=deep_model_hash,
        mode_digests=mode_digests,
        digest=canonical_sha256(value),
    )


def _load_verified_module(reference: ArtifactRef, module_name: str) -> ModuleType:
    """Execute exactly the bytes covered by ``reference``.

    The standard source-file loader may transparently consume a timestamp
    based adjacent ``.pyc`` after a successful source verification.  Reading
    and hashing the bytes that are subsequently compiled makes the operation
    content-addressed even when a caller forgot to install the formal fresh
    pycache-prefix bootstrap.  The bootstrap still protects all transitive
    imports performed by the verified interface.
    """

    path = reference.verify(label=f"verified interface {module_name}")
    try:
        source = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read verified interface: {path}") from exc
    if len(source) != reference.nbytes:
        raise ContractError(f"verified interface byte size changed: {path}")
    if hashlib.sha256(source).hexdigest() != reference.sha256:
        raise ContractError(f"verified interface SHA-256 changed: {path}")

    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__loader__ = None
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        code = compile(source, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException as exc:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"verified interface execution failed: {path}") from exc
    if Path(str(module.__file__)).resolve() != path:
        if sys.modules.get(module_name) is module:
            del sys.modules[module_name]
        raise ContractError("verified interface import path changed")
    return module


def _symbol(module: ModuleType, name: str) -> Any:
    value = getattr(module, name, None)
    if value is None:
        raise ContractError(f"verified interface lacks symbol: {name}")
    return value


def build_runtime_components(
    config: Mapping[str, Any], bindings: MethodBindings
) -> RuntimeComponents:
    """Load real 32B/14B models on frozen 5+3 devices and construct 4 engines."""

    for path in (REPOSITORY_ROOT, REPOSITORY_SRC, BASELINE_EXPERIMENT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from transformers import AutoTokenizer
    from kvcache_transfer import evaluation as evaluation_module
    from kvcache_transfer.modeling import load_qwen3
    from reverse_kv.runtime import ReverseMappedPrefixCacheBuilder

    audited_engine = _audited_engine_class(
        evaluation_module.TransferEvaluationEngine, evaluation_module
    )

    runtime = config["runtime"]
    # These files were already deep-hash verified by validate_method_bindings
    # before CUDA entry; derive the expected terminal set before model load.
    frozen_eos_ids = frozen_target_eos_token_ids(runtime["target_model"])
    tokenizer = AutoTokenizer.from_pretrained(
        runtime["target_model"], local_files_only=True, use_fast=True
    )
    target_model = load_qwen3(
        runtime["target_model"],
        devices=runtime["target_devices"],
        layer_counts=runtime["target_layer_counts"],
        role="source",
        attention_implementation=runtime["attention_implementation"],
    )
    source_model = load_qwen3(
        runtime["source_model"],
        devices=runtime["source_devices"],
        layer_counts=runtime["source_layer_counts"],
        role="target",
        attention_implementation=runtime["attention_implementation"],
    )
    observed_eos_ids = _observed_runtime_eos_token_ids(tokenizer, target_model)
    if observed_eos_ids != frozen_eos_ids:
        raise ContractError(
            "runtime EOS token IDs differ from content-hashed target metadata"
        )
    setattr(tokenizer, "_prospective_frozen_eos_token_ids", frozen_eos_ids)
    baseline_builder = ReverseMappedPrefixCacheBuilder(
        source_model=source_model,
        target_model=target_model,
        mapper_manifest=bindings.baseline_manifest.path,
        token_chunk=int(runtime["mapper_token_chunk"]),
    )

    shrinkage_method = config["methods"]["shrinkage"]
    shrinkage_ref = ArtifactRef.parse(
        shrinkage_method["runtime_interface"], label="shrinkage runtime interface"
    )
    shrinkage_module = _load_verified_module(
        shrinkage_ref, str(shrinkage_method["module_name"])
    )
    shrinkage_class = _symbol(shrinkage_module, str(shrinkage_method["symbol"]))
    shrinkage_builder = shrinkage_class(
        source_model=source_model,
        target_model=target_model,
        mapper_manifest=shrinkage_method["mapper_artifact"]["path"],
        token_chunk=int(runtime["mapper_token_chunk"]),
        verify_mapper_payloads=True,
    )

    tail_method = config["methods"]["tail_refresh_r32"]
    tail_ref = ArtifactRef.parse(tail_method["runtime_interface"], label="R32 runtime interface")
    tail_module = _load_verified_module(tail_ref, str(tail_method["module_name"]))
    tail_class = _symbol(tail_module, str(tail_method["symbol"]))
    tail_builder = tail_class.from_shared_mapper(
        baseline_builder, replay_tokens=int(tail_method["replay_tokens"])
    )
    if getattr(tail_builder, "_mapped_builder", None) is not baseline_builder:
        raise ContractError("R32 builder does not wrap the exact paired k20 builder")

    max_length = int(config["generation"]["max_length"])
    cap = int(config["generation"]["max_new_tokens"])
    builders = {
        "nvidia_k20": baseline_builder,
        "shrinkage": shrinkage_builder,
        "tail_refresh_r32": tail_builder,
    }
    engines = {
        "native14": audited_engine(
            target_model=target_model,
            tokenizer=tokenizer,
            mode="target_standalone",
            max_length=max_length,
            max_gen_tokens=cap,
        ),
        **{
            mode: audited_engine(
                target_model=target_model,
                tokenizer=tokenizer,
                mode="mapped",
                mapped_prefix_builder=builder,
                max_length=max_length,
                max_gen_tokens=cap,
            )
            for mode, builder in builders.items()
        },
    }
    if tuple(engines) != ("native14", "nvidia_k20", "shrinkage", "tail_refresh_r32"):
        raise ContractError("runtime engine ordering/modes changed")
    provenance = {
        "native14": {"method_id": config["methods"]["native14"]["method_id"]},
        "nvidia_k20": dict(getattr(baseline_builder, "provenance")),
        "shrinkage": dict(getattr(shrinkage_builder, "provenance")),
        "tail_refresh_r32": dict(getattr(tail_builder, "provenance")),
    }
    sharding = {
        "source_32b": getattr(source_model, "_kvcache_transfer_layer_sharding_evidence"),
        "target_14b": getattr(target_model, "_kvcache_transfer_layer_sharding_evidence"),
    }
    return RuntimeComponents(
        tokenizer=tokenizer,
        source_model=source_model,
        target_model=target_model,
        engines=engines,
        builders=builders,
        method_provenance=provenance,
        model_sharding=sharding,
    )


__all__ = [
    "AuditedGenerationResult",
    "MethodBindings",
    "RuntimeComponents",
    "build_runtime_components",
    "frozen_target_eos_token_ids",
    "validate_method_bindings",
]
