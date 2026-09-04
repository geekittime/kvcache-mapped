"""Independent validation of frozen model/topology/mapper runtime provenance."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as distribution_version
import os
from pathlib import Path
import sys
from typing import Any, Mapping

from .contracts import (
    ArtifactRef,
    ContractError,
    RUNTIME_PACKAGE_CLOSURE_FILES,
    canonical_sha256,
    read_json,
    require_child,
)


REPOSITORY_ROOT = Path("/home/wby/llm-cache/nvidia-kvcache")
BASELINE_EXPERIMENT_ROOT = (
    REPOSITORY_ROOT / "experiments" / "qwen3_32b_to_14b"
)
REQUIRED_CORE_RUNTIME_FILES = {
    # Package import metadata and the shared I/O helper execute transitively
    # when the baseline modules are imported.  Keep them in this protocol's
    # complete runtime closure even though the historical receipt predates
    # their enumeration.
    "src/kvcache_transfer/__init__.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/io.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/transfer.py",
}
REQUIRED_RECEIPT_CORE_RUNTIME_FILES = REQUIRED_CORE_RUNTIME_FILES - {
    "src/kvcache_transfer/__init__.py",
    "src/kvcache_transfer/io.py",
}
REQUIRED_PACKAGE_RUNTIME_FILES = set(RUNTIME_PACKAGE_CLOSURE_FILES)
REQUIRED_REVERSE_RUNTIME_FILES = {
    "reverse_kv/__init__.py",
    "reverse_kv/protocol.py",
    "reverse_kv/provenance.py",
    "reverse_kv/runtime.py",
}


def _expected_bytecode_policy(runtime: Mapping[str, Any]) -> tuple[bool, str]:
    """Return the exact interpreter bytecode policy required by formal runs."""

    dont_write = runtime.get("required_python_dont_write_bytecode")
    prefix = runtime.get("required_python_pycache_prefix")
    if type(dont_write) is not bool or dont_write is not True:
        raise ContractError(
            "runtime bytecode policy must require exact true dont_write_bytecode"
        )
    if type(prefix) is not str or not prefix or not Path(prefix).is_absolute():
        raise ContractError("runtime bytecode policy prefix must be absolute")
    return dont_write, prefix


def verify_python_bytecode_policy(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless this interpreter cannot read or create source-adjacent pyc.

    ``PYTHONDONTWRITEBYTECODE`` prevents writes but does not prevent reads.  A
    fresh, absent ``PYTHONPYCACHEPREFIX`` is therefore part of the frozen
    launch contract, and must be checked after interpreter startup (when
    ``sys.pycache_prefix`` has been populated) and before local imports.
    """

    required_dont_write, required_prefix = _expected_bytecode_policy(runtime)
    if os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise ContractError("formal run requires PYTHONDONTWRITEBYTECODE=1")
    if os.environ.get("PYTHONPYCACHEPREFIX") != required_prefix:
        raise ContractError(
            "formal run requires the exact frozen PYTHONPYCACHEPREFIX"
        )
    if sys.dont_write_bytecode is not required_dont_write:
        raise ContractError("interpreter dont_write_bytecode flag changed")
    if type(sys.pycache_prefix) is not str or sys.pycache_prefix != required_prefix:
        raise ContractError("interpreter pycache prefix differs from frozen policy")
    prefix_path = Path(required_prefix)
    for component in (prefix_path, *prefix_path.parents):
        if component.is_symlink():
            raise ContractError(
                "formal PYTHONPYCACHEPREFIX cannot contain a symlink component"
            )
    # ``exists`` misses a dangling symlink; lexists deliberately rejects it.
    if os.path.lexists(required_prefix):
        raise ContractError(
            "formal PYTHONPYCACHEPREFIX must be absent before local imports"
        )
    return {
        "python_dont_write_bytecode": required_dont_write,
        "python_pycache_prefix": required_prefix,
        "python_pycache_prefix_absent": True,
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{label} must be an object")
    return value


def _verify_declared_code_files(
    value: Any,
    *,
    root: Path,
    required: set[str],
    label: str,
) -> dict[str, Any]:
    files = _mapping(value, label)
    if not required.issubset(files):
        raise ContractError(f"{label} lacks required runtime closure")
    for relative, raw in files.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ContractError(f"{label} contains a malformed relative path")
        entry = _mapping(raw, f"{label}.{relative}")
        if set(entry) != {"nbytes", "sha256"}:
            raise ContractError(f"{label}.{relative} artifact schema changed")
        path = require_child(root, root / relative, label=f"{label}.{relative}")
        ArtifactRef.parse(
            {
                "path": str(path),
                "nbytes": entry.get("nbytes"),
                "sha256": entry.get("sha256"),
            },
            label=f"{label}.{relative}",
        ).verify(label=f"{label}.{relative}")
    return {"file_count": len(files), "inventory_sha256": canonical_sha256(dict(files))}


def _verify_package_closure(config: Mapping[str, Any]) -> dict[str, Any]:
    """Verify package files omitted by the immutable historical receipt.

    The baseline run definition is content-addressed and cannot be rewritten.
    Experiment 12 therefore carries a tiny independent closure manifest for
    ``kvcache_transfer``'s package initializer and shared I/O helper.  Requiring
    the exact key set prevents a caller from silently substituting a different
    package root or leaving newly executable files outside the digest.
    """

    closure = _mapping(config.get("runtime_package_closure"), "runtime package closure")
    if set(closure) != {"root", "files", "files_sha256"}:
        raise ContractError("runtime package closure schema changed")
    closure_root = Path(str(closure.get("root", "")))
    if not closure_root.is_absolute() or closure_root != REPOSITORY_ROOT:
        raise ContractError("runtime package closure root changed")
    files = _mapping(closure.get("files"), "runtime package closure files")
    if set(files) != REQUIRED_PACKAGE_RUNTIME_FILES:
        raise ContractError("runtime package closure file set changed")
    if files != RUNTIME_PACKAGE_CLOSURE_FILES:
        raise ContractError("runtime package closure file bindings changed")
    observed = _verify_declared_code_files(
        files,
        root=closure_root,
        required=REQUIRED_PACKAGE_RUNTIME_FILES,
        label="runtime package closure files",
    )
    if closure.get("files_sha256") != observed["inventory_sha256"]:
        raise ContractError("runtime package closure inventory digest changed")
    # Bind the exact config values in the returned evidence; callers include
    # this object in the run-definition/runtime binding digest.
    return {
        "root": str(closure_root),
        "files": dict(files),
        "files_sha256": closure["files_sha256"],
    }


def _embedded_ref(value: Any, label: str) -> dict[str, Any]:
    item = _mapping(value, label)
    reference = ArtifactRef.parse(
        {
            "path": item.get("path"),
            "nbytes": item.get("nbytes"),
            "sha256": item.get("sha256"),
        },
        label=label,
    )
    reference.verify(label=label)
    return reference.to_dict()


def _verify_runtime_code(
    receipt: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    code = _mapping(receipt.get("code"), "runtime provenance code")
    core = _verify_declared_code_files(
        code.get("files"),
        root=REPOSITORY_ROOT,
        # Keep the historical receipt's own fingerprint intact.  The two
        # package files added by this protocol are verified from the separate
        # config-bound closure below.
        required=REQUIRED_RECEIPT_CORE_RUNTIME_FILES,
        label="runtime provenance core files",
    )
    if code.get("files_sha256") != core["inventory_sha256"]:
        raise ContractError("runtime provenance core inventory digest changed")
    reverse = _verify_declared_code_files(
        code.get("reverse_runtime_files"),
        root=BASELINE_EXPERIMENT_ROOT,
        required=REQUIRED_REVERSE_RUNTIME_FILES,
        label="runtime provenance reverse files",
    )
    extension_freeze = _embedded_ref(
        code.get("long_context_extension_code_freeze"),
        "long-context code freeze",
    )
    reverse_freeze = _embedded_ref(
        code.get("reverse_base_code_freeze"),
        "reverse base code freeze",
    )
    package_closure = _verify_package_closure(config)
    return {
        "core": core,
        "package_closure": package_closure,
        "reverse": reverse,
        "long_context_extension_code_freeze": extension_freeze,
        "reverse_base_code_freeze": reverse_freeze,
    }


def _verify_preimport_environment(
    receipt: Mapping[str, Any], runtime_contract: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = _mapping(receipt.get("runtime"), "runtime environment")
    distributions = _mapping(runtime.get("distributions"), "runtime distributions")
    if sys.version != runtime.get("python"):
        raise ContractError("runtime Python build differs from frozen provenance")
    observed: dict[str, str] = {}
    for name, expected in distributions.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise ContractError("runtime distribution contract is malformed")
        try:
            installed = distribution_version(name)
        except PackageNotFoundError as exc:
            raise ContractError(f"required runtime distribution is missing: {name}") from exc
        if installed != expected:
            raise ContractError(
                f"runtime distribution {name} changed: {installed!r} != {expected!r}"
            )
        observed[name] = installed
    return {
        "python": sys.version,
        "distributions": observed,
        **verify_python_bytecode_policy(runtime_contract),
    }


def _verify_model(
    value: Any,
    *,
    expected_path: str,
    label: str,
    deep_model_hash: bool,
) -> dict[str, Any]:
    identity = _mapping(value, label)
    root = Path(str(identity.get("path", ""))).resolve()
    if root != Path(expected_path).resolve() or not root.is_dir():
        raise ContractError(f"{label} model path changed/missing")
    metadata = _mapping(identity.get("content_hashed_files"), f"{label} metadata")
    for relative, raw in metadata.items():
        path = (root / str(relative)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ContractError(f"{label} metadata path escapes model root") from exc
        entry = _mapping(raw, f"{label} metadata {relative}")
        ArtifactRef.parse(
            {
                "path": str(path),
                "nbytes": entry.get("nbytes"),
                "sha256": entry.get("sha256"),
            },
            label=f"{label} metadata {relative}",
        ).verify(label=f"{label} metadata {relative}", deep_hash=True)
    shards = identity.get("weight_shards")
    if not isinstance(shards, list) or len(shards) != identity.get("weight_shard_count"):
        raise ContractError(f"{label} weight shard inventory changed")
    observed: list[dict[str, Any]] = []
    for index, raw in enumerate(shards):
        entry = _mapping(raw, f"{label} shard {index}")
        path = (root / str(entry.get("path", ""))).resolve()
        try:
            relative = str(path.relative_to(root))
        except ValueError as exc:
            raise ContractError(f"{label} shard path escapes model root") from exc
        reference = ArtifactRef.parse(
            {
                "path": str(path),
                "nbytes": entry.get("nbytes"),
                "sha256": entry.get("sha256"),
            },
            label=f"{label} shard {index}",
        )
        reference.verify(label=f"{label} shard {relative}", deep_hash=deep_model_hash)
        observed.append(
            {"path": relative, "nbytes": reference.nbytes, "sha256": reference.sha256}
        )
    if canonical_sha256(observed) != identity.get("weight_shards_sha256"):
        raise ContractError(f"{label} weight shard inventory digest changed")
    if sum(item["nbytes"] for item in observed) != identity.get("weight_shard_total_nbytes"):
        raise ContractError(f"{label} weight shard byte total changed")
    return {
        "path": str(root),
        "metadata_files": len(metadata),
        "weight_shards": len(observed),
        "weight_shards_sha256": identity.get("weight_shards_sha256"),
    }


def validate_runtime_provenance(
    config: Mapping[str, Any], *, deep_model_hash: bool = False, verify_environment: bool = False
) -> dict[str, Any]:
    reference = ArtifactRef.parse(
        config.get("runtime_provenance_receipt"), label="runtime provenance receipt"
    )
    reference.verify(label="runtime provenance receipt")
    receipt = read_json(reference.path)
    if not isinstance(receipt, dict):
        raise ContractError("runtime provenance receipt root is malformed")
    declared = receipt.get("request_fingerprint")
    unsigned = dict(receipt)
    unsigned.pop("request_fingerprint", None)
    if not isinstance(declared, str) or canonical_sha256(unsigned) != declared:
        raise ContractError("runtime provenance request fingerprint is invalid")
    fixed = {
        "schema_version": 1,
        "direction": "qwen3_32b_to_14b",
        "max_length": 16384,
        "mapper_token_chunk": 1024,
        "attention_implementation": "flash_attention_2",
    }
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise ContractError(f"runtime provenance {field} changed")
    runtime = config["runtime"]
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
        raise ContractError("runtime provenance topology changed")
    if receipt.get("decoding") != {"batch_size": 1, "do_sample": False, "temperature": 0.0}:
        raise ContractError("runtime deterministic decoding contract changed")
    runtime_code = _verify_runtime_code(receipt, config)
    mapper_ref = ArtifactRef.parse(
        config["methods"]["nvidia_k20"]["mapper_artifact"], label="NVIDIA k20 mapper"
    )
    mapper_ref.verify(label="NVIDIA k20 mapper")
    mapper = _mapping(receipt.get("mapper"), "runtime mapper")
    if (
        Path(str(mapper.get("manifest", ""))).resolve() != Path(mapper_ref.path).resolve()
        or mapper.get("manifest_sha256") != mapper_ref.sha256
        or mapper.get("target_layer_count") != 40
    ):
        raise ContractError("runtime provenance is not bound to configured NVIDIA k20 mapper")
    source = _verify_model(
        receipt.get("source_model"),
        expected_path=runtime["source_model"],
        label="Qwen3-32B",
        deep_model_hash=deep_model_hash,
    )
    target = _verify_model(
        receipt.get("target_model"),
        expected_path=runtime["target_model"],
        label="Qwen3-14B",
        deep_model_hash=deep_model_hash,
    )
    # Verification depth is execution evidence, not part of model identity.
    # Keeping it outside ``value`` makes the shallow pre-CUDA freeze digest
    # exactly equal to the later full-weight-hash digest.  Otherwise every
    # real launch would reject its own frozen run definition solely because
    # the validation became stronger.
    value = {
        "receipt": reference.to_dict(),
        "receipt_request_fingerprint": declared,
        "source_model": source,
        "target_model": target,
        "mapper_manifest": mapper_ref.to_dict(),
        "runtime_code": runtime_code,
    }
    return {
        **value,
        "deep_model_hash_verified": deep_model_hash,
        "runtime_environment_verified": verify_environment,
        "runtime_environment": (
            _verify_preimport_environment(receipt, config["runtime"])
            if verify_environment
            else None
        ),
        "binding_digest": canonical_sha256(value),
    }


__all__ = ["validate_runtime_provenance", "verify_python_bytecode_policy"]
