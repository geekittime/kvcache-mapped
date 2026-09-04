"""Independent adapter around content-addressed reverse long-context runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping

from .contracts import (
    ArtifactRef,
    ContractError,
    ROOT,
    canonical_sha256,
    load_config,
    read_json,
    sha256_file,
)


REPOSITORY_ROOT = Path("/home/wby/llm-cache/nvidia-kvcache")
BASELINE_EXPERIMENT = REPOSITORY_ROOT / "experiments" / "qwen3_32b_to_14b"
IMPROVED_ROOT = ROOT.parent


@dataclass(frozen=True)
class MethodBindings:
    runtime_receipt: Mapping[str, Any]
    runtime_receipt_ref: ArtifactRef
    baseline_manifest_ref: ArtifactRef
    shrinkage: Mapping[str, Any]
    tail_refresh: Mapping[str, Any]
    cross_tier_audit_ref: ArtifactRef
    digest: str


@dataclass(frozen=True)
class RuntimeComponents:
    tokenizer: Any
    source_model: Any
    target_model: Any
    target_engine: Any
    engines: Mapping[str, Any]
    builders: Mapping[str, Any]
    method_provenance: Mapping[str, Any]


def _expect_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _verify_receipt_fingerprint(receipt: Mapping[str, Any]) -> None:
    declared = receipt.get("request_fingerprint")
    if not isinstance(declared, str):
        raise ContractError("upstream runtime receipt lacks request_fingerprint")
    unsigned = dict(receipt)
    unsigned.pop("request_fingerprint", None)
    if canonical_sha256(unsigned) != declared:
        raise ContractError("upstream runtime receipt fingerprint is invalid")


def _verify_upstream_code(receipt: Mapping[str, Any]) -> None:
    code = _expect_mapping(receipt.get("code"), "upstream runtime code")
    generic = _expect_mapping(code.get("files"), "upstream generic code files")
    for relative, identity in generic.items():
        entry = _expect_mapping(identity, f"upstream code {relative}")
        path = (REPOSITORY_ROOT / str(relative)).resolve()
        expected = ArtifactRef(
            path=str(path), nbytes=int(entry.get("nbytes", -1)), sha256=str(entry.get("sha256", ""))
        )
        expected.verify(label=f"upstream code {relative}")
    reverse = _expect_mapping(
        code.get("reverse_runtime_files"), "upstream reverse runtime files"
    )
    for relative, identity in reverse.items():
        entry = _expect_mapping(identity, f"reverse runtime {relative}")
        path = (BASELINE_EXPERIMENT / str(relative)).resolve()
        expected = ArtifactRef(
            path=str(path), nbytes=int(entry.get("nbytes", -1)), sha256=str(entry.get("sha256", ""))
        )
        expected.verify(label=f"reverse runtime {relative}")


def _verify_model_identity(
    value: Any, *, expected_path: str, label: str, deep_hash: bool
) -> None:
    identity = _expect_mapping(value, label)
    root = Path(str(identity.get("path", ""))).resolve()
    if root != Path(expected_path).resolve():
        raise ContractError(f"{label} path changed")
    files = _expect_mapping(identity.get("content_hashed_files"), f"{label} metadata")
    for relative, raw in files.items():
        entry = _expect_mapping(raw, f"{label}/{relative}")
        reference = ArtifactRef(
            path=str(root / str(relative)),
            nbytes=int(entry.get("nbytes", -1)),
            sha256=str(entry.get("sha256", "")),
        )
        reference.verify(label=f"{label}/{relative}", deep_hash=True)
    shards = identity.get("weight_shards")
    if not isinstance(shards, list) or len(shards) != identity.get("weight_shard_count"):
        raise ContractError(f"{label} weight shard inventory changed")
    observed: list[dict[str, Any]] = []
    for raw in shards:
        entry = _expect_mapping(raw, f"{label} shard")
        path = (root / str(entry.get("path", ""))).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError as exc:
            raise ContractError(f"{label} shard escapes model root") from exc
        reference = ArtifactRef(
            path=str(path),
            nbytes=int(entry.get("nbytes", -1)),
            sha256=str(entry.get("sha256", "")),
        )
        reference.verify(label=f"{label} shard {relative}", deep_hash=deep_hash)
        observed.append({"path": relative, "nbytes": reference.nbytes, "sha256": reference.sha256})
    if canonical_sha256(observed) != identity.get("weight_shards_sha256"):
        raise ContractError(f"{label} weight shard inventory digest changed")
    if sum(entry["nbytes"] for entry in observed) != identity.get("weight_shard_total_nbytes"):
        raise ContractError(f"{label} weight byte count changed")


def validate_upstream_runtime(
    config: Mapping[str, Any], *, deep_model_hash: bool = False
) -> tuple[dict[str, Any], ArtifactRef]:
    runtime = _expect_mapping(config.get("runtime"), "runtime")
    reference = ArtifactRef.parse(runtime.get("upstream_receipt"), label="upstream runtime receipt")
    path = reference.verify(label="upstream runtime receipt")
    receipt = read_json(path)
    if not isinstance(receipt, dict):
        raise ContractError("upstream runtime receipt must be an object")
    _verify_receipt_fingerprint(receipt)
    fixed = {
        "schema_version": 1,
        "direction": "qwen3_32b_to_14b",
        "max_length": 16384,
        "mapper_token_chunk": 1024,
        "attention_implementation": "flash_attention_2",
    }
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise ContractError(f"upstream runtime {field} changed")
    expected_topology = {
        "source": {
            "model": runtime["source_model"],
            "devices": runtime["source_devices"],
            "layer_counts": runtime["source_layer_counts"],
        },
        "target": {
            "model": runtime["target_model"],
            "devices": runtime["target_devices"],
            "layer_counts": runtime["target_layer_counts"],
        },
    }
    if receipt.get("formal_topology") != expected_topology:
        raise ContractError("upstream runtime topology changed")
    if receipt.get("decoding") != {"batch_size": 1, "do_sample": False, "temperature": 0.0}:
        raise ContractError("upstream deterministic decoding contract changed")
    _verify_upstream_code(receipt)
    _verify_model_identity(
        receipt.get("source_model"),
        expected_path=runtime["source_model"],
        label="source Qwen3-32B",
        deep_hash=deep_model_hash,
    )
    _verify_model_identity(
        receipt.get("target_model"),
        expected_path=runtime["target_model"],
        label="target Qwen3-14B",
        deep_hash=deep_model_hash,
    )
    baseline = ArtifactRef.parse(
        config["methods"]["nvidia_k20"]["mapper_manifest"],
        label="NVIDIA k20 mapper manifest",
    )
    baseline.verify(label="NVIDIA k20 mapper manifest")
    mapper = _expect_mapping(receipt.get("mapper"), "upstream mapper identity")
    if (
        Path(str(mapper.get("manifest", ""))).resolve() != Path(baseline.path).resolve()
        or mapper.get("manifest_sha256") != baseline.sha256
        or mapper.get("target_layer_count") != 40
    ):
        raise ContractError("upstream runtime is not bound to the configured NVIDIA k20 mapper")
    return receipt, reference


def _verify_shrinkage(config: Mapping[str, Any]) -> dict[str, Any]:
    method = _expect_mapping(config["methods"].get("shrinkage"), "shrinkage binding")
    if method.get("state") != "ready" or method.get("factory") != "constructor":
        raise ContractError("shrinkage binding is not ready")
    manifest_ref = ArtifactRef.parse(method.get("mapper_manifest"), label="shrinkage mapper")
    completion_ref = ArtifactRef.parse(method.get("completion_receipt"), label="shrinkage completion receipt")
    algebra_ref = ArtifactRef.parse(method.get("algebra_receipt"), label="shrinkage algebra receipt")
    interface_ref = ArtifactRef.parse(method.get("interface"), label="shrinkage runtime interface")
    for label, reference in (
        ("shrinkage mapper", manifest_ref),
        ("shrinkage completion receipt", completion_ref),
        ("shrinkage algebra receipt", algebra_ref),
        ("shrinkage runtime interface", interface_ref),
    ):
        reference.verify(label=label)
    manifest = read_json(manifest_ref.path)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("direction") != "qwen3_32b_to_14b"
        or manifest.get("method") != "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers"
        or len(manifest.get("layers", [])) != 40
    ):
        raise ContractError("shrinkage mapper manifest contract changed")
    upstream = _expect_mapping(manifest.get("upstream"), "shrinkage upstream")
    k20 = _expect_mapping(upstream.get("k20_manifest"), "shrinkage k20 binding")
    baseline = config["methods"]["nvidia_k20"]["mapper_manifest"]
    if k20.get("sha256") != baseline["sha256"] or Path(str(k20.get("path"))).resolve() != Path(baseline["path"]).resolve():
        raise ContractError("shrinkage candidate is not anchored to the paired k20 baseline")
    completion = read_json(completion_ref.path)
    if (
        completion.get("status") != "complete_by_append_only_amendment"
        or completion.get("amendment_scope") != "receipt-serialization-only"
        or completion.get("mapper", {}).get("manifest_sha256") != manifest_ref.sha256
        or completion.get("mapper", {}).get("payloads_deep_hashed") is not True
    ):
        raise ContractError("shrinkage completion receipt changed")
    execution = _expect_mapping(completion.get("execution"), "shrinkage completion execution")
    if any(
        execution.get(field) is not False
        for field in ("activation_recomputed", "alpha_modified", "cuda_invoked", "locked_audit_recomputed", "mapper_modified")
    ):
        raise ContractError("shrinkage completion amendment exceeded its scope")
    algebra = read_json(algebra_ref.path)
    if (
        algebra.get("schema_version") != 1
        or algebra.get("status") != "complete"
        or algebra.get("cuda_invoked") is not False
        or algebra.get("amendment_v1_receipt", {}).get("sha256") != completion_ref.sha256
    ):
        raise ContractError("shrinkage algebra receipt changed")
    code_files = algebra.get("code", {}).get("files", [])
    runtime_entries = [entry for entry in code_files if entry.get("path") == "src/runtime.py"]
    if len(runtime_entries) != 1 or runtime_entries[0].get("sha256") != interface_ref.sha256 or runtime_entries[0].get("nbytes") != interface_ref.nbytes:
        raise ContractError("shrinkage algebra receipt does not bind the runtime interface")
    return {
        **dict(method),
        "binding_digest": canonical_sha256(
            {
                "mapper_manifest": manifest_ref.to_dict(),
                "completion_receipt": completion_ref.to_dict(),
                "algebra_receipt": algebra_ref.to_dict(),
                "interface": interface_ref.to_dict(),
            }
        ),
    }


def _verify_tail_refresh(config: Mapping[str, Any]) -> dict[str, Any]:
    method = _expect_mapping(config["methods"].get("tail_refresh"), "tail-refresh binding")
    if method.get("state") != "ready" or method.get("factory") != "from_shared_mapper":
        raise ContractError("tail-refresh binding is not ready")
    completion_ref = ArtifactRef.parse(method.get("completion_receipt"), label="tail CPU preflight receipt")
    receipt_ref = ArtifactRef.parse(method.get("interface_receipt"), label="tail interface receipt")
    interface_ref = ArtifactRef.parse(method.get("interface"), label="tail runtime interface")
    for label, reference in (
        ("tail CPU preflight receipt", completion_ref),
        ("tail interface receipt", receipt_ref),
        ("tail runtime interface", interface_ref),
    ):
        reference.verify(label=label)
    completion = read_json(completion_ref.path)
    if (
        completion.get("schema_version") != 2
        or completion.get("status") != "passed"
        or completion.get("gpu_used") is not False
        or completion.get("cuda_execution_started") is not False
        or completion.get("tests", {}).get("failed") != 0
        or completion.get("implementation", {}).get("sha256") != interface_ref.sha256
        or completion.get("implementation", {}).get("nbytes") != interface_ref.nbytes
        or completion.get("prepared_v2", {}).get("interface_sha256") != receipt_ref.sha256
        or completion.get("musique_protocol_binding", {}).get("config_sha256") != config["datasets"]["musique_train"]["config"]["sha256"]
        or completion.get("musique_protocol_binding", {}).get("registry_manifest_sha256") != config["datasets"]["musique_train"]["registry_manifest"]["sha256"]
    ):
        raise ContractError("tail-refresh CPU preflight receipt changed")
    interface_receipt = read_json(receipt_ref.path)
    if (
        interface_receipt.get("schema_version") != 1
        or interface_receipt.get("status") != "frozen_method_dev_interface"
        or interface_receipt.get("builder_class") != method.get("symbol")
        or interface_receipt.get("implementation") != interface_ref.to_dict()
        or interface_receipt.get("constructor", {}).get("replay_tokens_allowed") != [1, 8, 32]
        or interface_receipt.get("causal_contract", {}).get("builder_native_tail_count") != "R-1"
        or interface_receipt.get("causal_contract", {}).get("engine_boundary_count") != 1
    ):
        raise ContractError("tail-refresh interface receipt changed")
    return {
        **dict(method),
        "binding_digest": canonical_sha256(
            {
                "completion_receipt": completion_ref.to_dict(),
                "interface_receipt": receipt_ref.to_dict(),
                "interface": interface_ref.to_dict(),
            }
        ),
    }


def _verify_cross_tier_audit(config: Mapping[str, Any]) -> ArtifactRef:
    """Bind the completed CPU-only 08 leakage sub-gate to this harness."""

    reference = ArtifactRef.parse(
        config["provenance"]["cross_tier_audit_receipt"], label="08 cross-tier audit receipt"
    )
    receipt = read_json(reference.verify(label="08 cross-tier audit receipt"))
    if not isinstance(receipt, Mapping):
        raise ContractError("08 cross-tier audit receipt must be an object")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("audit_id") != "qwen3_mapper_fit_vs_method_dev_v2_cross_tier_v1"
        or receipt.get("status") != "pass"
        or receipt.get("execution") != {
            "deterministic": True,
            "device": "cpu",
            "fail_closed": True,
            "gpu_codepaths": False,
        }
        or receipt.get("result", {}).get("flagged_pairs") != 0
        or receipt.get("scope", {}).get("gate")
        != "tail_pilot_mapper_fit_vs_selected_method_dev_subgate"
    ):
        raise ContractError("08 cross-tier audit receipt contract changed")
    return reference


def validate_method_bindings(
    config_or_path: Mapping[str, Any] | str | Path,
    *,
    deep_model_hash: bool = False,
) -> MethodBindings:
    config = (
        load_config(config_or_path, verify_inputs=True)
        if not isinstance(config_or_path, Mapping)
        else dict(config_or_path)
    )
    receipt, receipt_ref = validate_upstream_runtime(config, deep_model_hash=deep_model_hash)
    baseline_ref = ArtifactRef.parse(
        config["methods"]["nvidia_k20"]["mapper_manifest"], label="NVIDIA k20 mapper"
    )
    shrinkage = _verify_shrinkage(config)
    tail = _verify_tail_refresh(config)
    cross_tier_audit_ref = _verify_cross_tier_audit(config)
    digest = canonical_sha256(
        {
            "upstream_runtime_receipt": receipt_ref.to_dict(),
            "nvidia_k20": baseline_ref.to_dict(),
            "shrinkage": shrinkage["binding_digest"],
            "tail_refresh": tail["binding_digest"],
            "cross_tier_audit_receipt": cross_tier_audit_ref.to_dict(),
        }
    )
    return MethodBindings(
        runtime_receipt=receipt,
        runtime_receipt_ref=receipt_ref,
        baseline_manifest_ref=baseline_ref,
        shrinkage=shrinkage,
        tail_refresh=tail,
        cross_tier_audit_ref=cross_tier_audit_ref,
        digest=digest,
    )


def _load_file_module(reference: ArtifactRef, module_name: str) -> ModuleType:
    path = reference.verify(label=f"interface {module_name}")
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ContractError(f"cannot import verified interface: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    if Path(str(module.__file__)).resolve() != path:
        raise ContractError(f"imported interface path changed: {module.__file__}")
    return module


def _symbol(module: ModuleType, name: str) -> Any:
    value = getattr(module, name, None)
    if value is None:
        raise ContractError(f"verified interface lacks symbol {name}")
    return value


def build_runtime_components(config: Mapping[str, Any], bindings: MethodBindings) -> RuntimeComponents:
    """Load both models and all five paired modes; call only after GPU gate."""

    for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src", BASELINE_EXPERIMENT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from transformers import AutoTokenizer
    from kvcache_transfer.evaluation import TransferEvaluationEngine
    from kvcache_transfer.modeling import load_qwen3
    from reverse_kv.runtime import ReverseMappedPrefixCacheBuilder

    runtime = config["runtime"]
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
    baseline_builder = ReverseMappedPrefixCacheBuilder(
        source_model=source_model,
        target_model=target_model,
        mapper_manifest=bindings.baseline_manifest_ref.path,
        token_chunk=int(runtime["mapper_token_chunk"]),
    )

    shrinkage_ref = ArtifactRef.parse(bindings.shrinkage["interface"], label="shrinkage interface")
    shrinkage_module = _load_file_module(shrinkage_ref, str(bindings.shrinkage["module_name"]))
    shrinkage_class = _symbol(shrinkage_module, str(bindings.shrinkage["symbol"]))
    shrinkage_builder = shrinkage_class(
        source_model=source_model,
        target_model=target_model,
        mapper_manifest=bindings.shrinkage["mapper_manifest"]["path"],
        token_chunk=int(runtime["mapper_token_chunk"]),
        verify_mapper_payloads=True,
    )

    tail_ref = ArtifactRef.parse(bindings.tail_refresh["interface"], label="tail interface")
    tail_module = _load_file_module(tail_ref, str(bindings.tail_refresh["module_name"]))
    tail_class = _symbol(tail_module, str(bindings.tail_refresh["symbol"]))
    tail_r1 = tail_class.from_shared_mapper(baseline_builder, replay_tokens=1)
    tail_r8 = tail_r1.variant(8)
    tail_r32 = tail_r1.variant(32)
    if getattr(tail_r1, "_mapped_builder", None) is not baseline_builder:
        raise ContractError("tail R1 does not wrap the exact paired baseline builder")
    if any(getattr(builder, "_mapped_builder", None) is not baseline_builder for builder in (tail_r8, tail_r32)):
        raise ContractError("tail variants do not share the exact paired baseline builder")

    cap = int(config["generation"]["max_new_tokens"])
    max_length = int(config["generation"]["max_length"])
    builders = {
        "nvidia_k20": baseline_builder,
        "shrinkage": shrinkage_builder,
        "tail_refresh_r8": tail_r8,
        "tail_refresh_r32": tail_r32,
    }
    target_engine = TransferEvaluationEngine(
        target_model=target_model,
        tokenizer=tokenizer,
        mode="target_standalone",
        max_length=max_length,
        max_gen_tokens=cap,
    )
    engines = {
        "native14": target_engine,
        **{
            mode: TransferEvaluationEngine(
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
    provenance = {
        "native14": {"method_id": config["methods"]["native14"]["method_id"]},
        "nvidia_k20": dict(getattr(baseline_builder, "provenance")),
        "shrinkage": dict(getattr(shrinkage_builder, "provenance")),
        "tail_refresh_r1_contract": dict(getattr(tail_r1, "provenance")),
        "tail_refresh_r8": dict(getattr(tail_r8, "provenance")),
        "tail_refresh_r32": dict(getattr(tail_r32, "provenance")),
    }
    return RuntimeComponents(
        tokenizer=tokenizer,
        source_model=source_model,
        target_model=target_model,
        target_engine=target_engine,
        engines=engines,
        builders=builders,
        method_provenance=provenance,
    )


__all__ = [
    "MethodBindings",
    "RuntimeComponents",
    "build_runtime_components",
    "validate_method_bindings",
    "validate_upstream_runtime",
]
