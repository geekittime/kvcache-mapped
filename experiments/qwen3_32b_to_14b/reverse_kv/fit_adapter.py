"""Isolated production-fit driver for the formal 32B -> 14B mapper.

The audited matrix construction and two-pass centered-ridge numerical kernels
are direction-neutral, so this module calls them without copying a second
implementation.  All direction/architecture gates, activation role binding,
and aggregate publication are replaced here.  The shared repository is read
only; every checkpoint and run record is rooted below ``EXPERIMENT_ROOT``.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open

from .protocol import (
    BIAS_COUNT,
    DEFAULT_CONFIG,
    DIRECTION,
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    SOURCE,
    SOURCE_FEATURES,
    TARGET,
    TARGET_FEATURES,
    TOP_K,
    TRAIN_OBSERVATIONS,
    WEIGHT_COUNT,
    atomic_write_json,
    canonical_sha256,
    fit_activation_inventory,
    read_json,
    sha256_file,
    validate_config,
)
from .selection_adapter import validate_reverse_selection_artifact


def _sanitized_fit_argv(extra_argv: Sequence[str]) -> list[str]:
    """Accept only operational controls; immutable inputs cannot be overridden."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--devices")
    parser.add_argument("--target-layers")
    parser.add_argument("--chunk-tokens", type=int)
    parser.add_argument("--solver", choices=("cholesky", "lstsq"))
    parser.add_argument("--analysis-device")
    parser.add_argument("--cpu-threads-per-worker", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parsed = parser.parse_args(list(map(str, extra_argv)))
    result: list[str] = []
    for option, value in (
        ("--devices", parsed.devices),
        ("--target-layers", parsed.target_layers),
        ("--chunk-tokens", parsed.chunk_tokens),
        ("--solver", parsed.solver),
        ("--analysis-device", parsed.analysis_device),
        ("--cpu-threads-per-worker", parsed.cpu_threads_per_worker),
    ):
        if value is not None:
            result.extend((option, str(value)))
    if parsed.no_resume:
        result.append("--no-resume")
    if parsed.dry_run:
        result.append("--dry-run")
    return result


def _collect_exact_reverse_inventory(
    source_root: str | Path,
    target_root: str | Path,
    architecture: Any,
    *,
    verify_payload_checksum: bool,
) -> dict[str, Any]:
    """Bind the inventory to the same roots/dimensions workers will consume."""

    if Path(source_root).resolve() != SOURCE.train_root.resolve():
        raise ValueError("fit source activations are not the frozen 32B capture")
    if Path(target_root).resolve() != TARGET.train_root.resolve():
        raise ValueError("fit target activations are not the frozen 14B capture")
    assert_reverse_paper_fit(architecture, 0.01)
    if verify_payload_checksum is not True:
        raise ValueError("formal reverse fit requires full activation payload SHA256")
    return fit_activation_inventory(verify_payloads=True)


def _load_implementation() -> Any:
    scripts = REPOSITORY_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    module = importlib.import_module("fit_mapper")
    expected = (scripts / "fit_mapper.py").resolve()
    if Path(module.__file__).resolve() != expected:
        raise ImportError(f"unexpected fit implementation: {module.__file__}")
    return module


def assert_reverse_paper_fit(architecture: Any, ridge_lambda: float) -> None:
    observed = {
        "source_layers": int(architecture.source_layers),
        "target_layers": int(architecture.target_layers),
        "kv_heads": int(architecture.kv_heads),
        "head_dim": int(architecture.head_dim),
        "selected_source_layers": int(architecture.selected_source_layers),
        "n_samples": int(architecture.n_samples),
        "source_features": int(architecture.source_features),
        "target_features": int(architecture.target_features),
        "weight_count": int(architecture.expected_weight_count),
        "bias_count": int(architecture.expected_bias_count),
    }
    expected = {
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "selected_source_layers": TOP_K,
        "n_samples": TRAIN_OBSERVATIONS,
        "source_features": SOURCE_FEATURES,
        "target_features": TARGET_FEATURES,
        "weight_count": WEIGHT_COUNT,
        "bias_count": BIAS_COUNT,
    }
    if observed != expected:
        raise ValueError(f"reverse paper architecture mismatch: {observed!r} != {expected!r}")
    if float(ridge_lambda) != 0.01:
        raise ValueError(f"reverse paper ridge lambda must be 0.01, got {ridge_lambda!r}")


def _validate_reverse_selection(
    implementation: Any,
    selection_path: str | Path,
    inventory: Mapping[str, Any],
    *,
    require_formal_protocol: bool,
) -> dict[str, Any]:
    """Use the audited provenance validator with a memory-only direction view.

    The audited implementation has exactly one non-numerical hard-code: its
    direction string.  Presenting an in-memory forward label lets it perform
    all catalog, payload, rule, fingerprint, and score checks without changing
    the reverse selection artifact on disk.
    """

    path = Path(selection_path).resolve()
    selection = validate_reverse_selection_artifact(path)
    expected = {
        "direction": DIRECTION,
        "formal_protocol": True,
        "source_layers": 64,
        "target_layers": 40,
        "kv_heads": 8,
        "head_dim": 128,
        "k": TOP_K,
        "n_samples": TRAIN_OBSERVATIONS,
    }
    for field, wanted in expected.items():
        if selection.get(field) != wanted:
            raise ValueError(
                f"reverse selection {field}={selection.get(field)!r}, expected {wanted!r}"
            )
    role_binding = selection.get("semantic_role_binding")
    if role_binding != {
        "source": {
            "model_name": SOURCE.model_name,
            "capture_native_role": SOURCE.capture_native_role,
        },
        "target": {
            "model_name": TARGET.model_name,
            "capture_native_role": TARGET.capture_native_role,
        },
    }:
        raise ValueError("reverse selection capture-native role binding is invalid")

    original_reader = implementation._strict_json

    def direction_view(candidate: str | Path) -> dict[str, Any]:
        raw = original_reader(candidate)
        if Path(candidate).resolve() == path:
            raw = dict(raw)
            raw["direction"] = "qwen3_14b_to_32b"
        return raw

    implementation._strict_json = direction_view
    try:
        result = implementation._reverse_original_validate_selection(
            path,
            inventory,
            require_formal_protocol=require_formal_protocol,
        )
    finally:
        implementation._strict_json = original_reader
    result = dict(result)
    result["semantic_direction"] = DIRECTION
    result["capture_native_roles"] = {
        "source": SOURCE.capture_native_role,
        "target": TARGET.capture_native_role,
    }
    return result


def _validate_layer_payload(
    output_directory: str | Path,
    *,
    target_layer: int,
    selected_layers: Sequence[int],
    source_features: int,
    target_features: int,
    kv_heads: int,
    head_dim: int,
    fit_fingerprint: str,
    selection_sha256: str,
) -> dict[str, Any]:
    """Actually load and validate one committed four-tensor safetensors layer."""

    output = Path(output_directory).resolve()
    payload = output / "layers" / f"layer_{target_layer:03d}.safetensors"
    sidecar_path = payload.with_suffix(".json")
    if not payload.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"mapper layer {target_layer} is not fully committed")
    sidecar = read_json(sidecar_path)
    selected = [int(value) for value in selected_layers]
    shapes = {
        "key_weight": [source_features, target_features],
        "key_bias": [target_features],
        "value_weight": [source_features, target_features],
        "value_bias": [target_features],
    }
    sidecar_expected = {
        "target_layer": target_layer,
        "source_layers_rank_order": selected,
        "sha256": sha256_file(payload),
        "dtype": "float32",
        "source_heads": kv_heads,
        "head_dim": head_dim,
        "tensor_shapes": shapes,
        "weight_numel": 2 * source_features * target_features,
        "bias_numel": 2 * target_features,
    }
    for field, expected in sidecar_expected.items():
        if sidecar.get(field) != expected:
            raise ValueError(f"mapper layer {target_layer} sidecar {field} mismatch")
    metadata = sidecar.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"mapper layer {target_layer} lacks fit metadata")
    metadata_expected = {
        "fit_fingerprint": fit_fingerprint,
        "selection_manifest_sha256": selection_sha256,
        "feature_order": "selected_layer_rank,kv_head,head_dim",
        "selected_source_layers_rank_order": selected,
        "ridge_lambda_added_directly_to_unnormalized_gram": 0.01,
        "standardize_features": False,
        "regularize_bias": False,
        "forward_activation_dtype": "bfloat16",
        "covariance_dtype": "float32",
        "mapper_dtype": "float32",
        "analysis_dtype": "float64",
        "allow_tf32": False,
    }
    for field, expected in metadata_expected.items():
        if metadata.get(field) != expected:
            raise ValueError(f"mapper layer {target_layer} metadata {field} mismatch")

    with safe_open(str(payload), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(shapes):
            raise ValueError(f"mapper layer {target_layer} safetensors key set mismatch")
        tensor_metadata = handle.metadata() or {}
        expected_tensor_metadata = {
            "target_layer": str(target_layer),
            "source_layers_rank_order": ",".join(map(str, selected)),
            "source_heads": str(kv_heads),
            "head_dim": str(head_dim),
        }
        if tensor_metadata != expected_tensor_metadata:
            raise ValueError(f"mapper layer {target_layer} safetensors metadata mismatch")
        loaded = {}
        for name, expected_shape in shapes.items():
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32 or list(tensor.shape) != expected_shape:
                raise ValueError(
                    f"mapper layer {target_layer} tensor {name} has "
                    f"{list(tensor.shape)}/{tensor.dtype}, expected {expected_shape}/float32"
                )
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(
                    f"mapper layer {target_layer} tensor {name} contains NaN/Inf"
                )
            loaded[name] = int(tensor.numel())
    return {
        "target_layer": target_layer,
        "payload": str(payload),
        "payload_sha256": sidecar_expected["sha256"],
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sha256_file(sidecar_path),
        "tensor_numel": loaded,
    }


def _validate_complete_payloads(
    output_directory: str | Path,
    *,
    architecture: Any,
    fit_fingerprint: str,
    selection_sha256: str,
    selected_layers: Sequence[Sequence[int]],
) -> dict[str, Any]:
    assert_reverse_paper_fit(architecture, 0.01)
    if len(selected_layers) != 40:
        raise ValueError("complete reverse mapper must contain 40 selection rows")
    entries = [
        _validate_layer_payload(
            output_directory,
            target_layer=target_layer,
            selected_layers=selected_layers[target_layer],
            source_features=SOURCE_FEATURES,
            target_features=TARGET_FEATURES,
            kv_heads=8,
            head_dim=128,
            fit_fingerprint=fit_fingerprint,
            selection_sha256=selection_sha256,
        )
        for target_layer in range(40)
    ]
    return {
        "status": "passed",
        "layers_actually_loaded": 40,
        "tensors_actually_loaded": 160,
        "expected_tensor_shapes": {
            "key_weight": [SOURCE_FEATURES, TARGET_FEATURES],
            "key_bias": [TARGET_FEATURES],
            "value_weight": [SOURCE_FEATURES, TARGET_FEATURES],
            "value_bias": [TARGET_FEATURES],
        },
        "entries_digest": canonical_sha256(entries),
    }


def _publish_reverse_manifest(implementation: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    output_directory = Path(args[0] if args else kwargs["output_directory"]).resolve()
    try:
        output_directory.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("reverse mapper output escaped isolated experiment") from exc
    payload_validation = _validate_complete_payloads(
        output_directory,
        architecture=kwargs["architecture"],
        fit_fingerprint=str(kwargs["fit_fingerprint"]),
        selection_sha256=str(kwargs["selection_sha256"]),
        selected_layers=kwargs["selected_layers"],
    )
    manifest = implementation._reverse_original_publish(*args, **kwargs)
    manifest = dict(manifest)
    manifest["direction"] = DIRECTION
    manifest["paper_reference"] = {
        "paper": "arXiv:2608.03893",
        "sections": ["4.6", "4.7"],
        "table": 16,
        "selected_source_layers": TOP_K,
    }
    manifest["semantic_role_binding"] = {
        "source": {
            "model_name": SOURCE.model_name,
            "capture_native_role": SOURCE.capture_native_role,
        },
        "target": {
            "model_name": TARGET.model_name,
            "capture_native_role": TARGET.capture_native_role,
        },
    }
    manifest["payload_validation"] = payload_validation
    destination = output_directory / "manifest.json"
    atomic_write_json(destination, manifest)
    atomic_write_json(
        output_directory / "manifest.sha256.json",
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    return manifest


def run_fit(extra_argv: Sequence[str] = (), *, config_path: Path = DEFAULT_CONFIG) -> int:
    resolved_config = Path(config_path).resolve()
    if resolved_config != DEFAULT_CONFIG.resolve():
        raise ValueError(
            "formal reverse fit config is immutable; use the experiment mapper.yaml"
        )
    configuration = validate_config(resolved_config)
    outputs = configuration["outputs"]
    expected_outputs = {
        "selection": (
            EXPERIMENT_ROOT
            / "artifacts/selection/qwen3_32b_to_14b/selected_layers.json"
        ).resolve(),
        "mapper": (
            EXPERIMENT_ROOT
            / "artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01"
        ).resolve(),
    }
    for field, expected in expected_outputs.items():
        if outputs.get(field) != expected:
            raise ValueError(
                f"formal reverse fit {field} path is immutable: "
                f"{outputs.get(field)!s} != {expected!s}"
            )
    sanitized_argv = _sanitized_fit_argv(extra_argv)
    implementation = _load_implementation()
    implementation.assert_paper_fit = assert_reverse_paper_fit
    implementation.collect_activation_inventory = _collect_exact_reverse_inventory
    if not hasattr(implementation, "_reverse_original_validate_selection"):
        implementation._reverse_original_validate_selection = (
            implementation.validate_selection_provenance
        )
    implementation.validate_selection_provenance = (
        lambda selection_path, inventory, require_formal_protocol: _validate_reverse_selection(
            implementation,
            selection_path,
            inventory,
            require_formal_protocol=require_formal_protocol,
        )
    )
    if not hasattr(implementation, "_reverse_original_publish"):
        implementation._reverse_original_publish = implementation.publish_complete_manifest
    implementation.publish_complete_manifest = (
        lambda *args, **kwargs: _publish_reverse_manifest(implementation, *args, **kwargs)
    )

    argv = [
        "--config",
        str(resolved_config),
        "--selection",
        str(outputs["selection"]),
        "--source-activations",
        str(SOURCE.train_root),
        "--target-activations",
        str(TARGET.train_root),
        "--output",
        str(outputs["mapper"]),
        *sanitized_argv,
    ]
    return int(implementation.main(argv))


__all__ = [
    "assert_reverse_paper_fit",
    "run_fit",
    "_collect_exact_reverse_inventory",
    "_sanitized_fit_argv",
    "_validate_layer_payload",
]
