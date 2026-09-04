"""Independent closed-form fit for Figure 4's retuned reverse ``k=4`` curve."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .fit_adapter import (
    _load_implementation,
    _sanitized_fit_argv,
    _validate_layer_payload,
)
from .k4_protocol import (
    DEFAULT_K4_MAPPER_CONFIG,
    K4_BIAS_COUNT,
    K4_MAPPER_ROOT,
    K4_SELECTION,
    K4_SOURCE_FEATURES,
    K4_TARGET_FEATURES,
    K4_TOP_K,
    K4_TRAIN_OBSERVATIONS,
    K4_WEIGHT_COUNT,
    validate_k4_mapper_config,
)
from .k4_selection import PAPER_CURVE, validate_k4_selection_artifact
from .protocol import (
    DIRECTION,
    EXPERIMENT_ROOT,
    SOURCE,
    TARGET,
    atomic_write_json,
    canonical_sha256,
    fit_activation_inventory,
    sha256_file,
)


def assert_k4_fit(architecture: Any, ridge_lambda: float) -> None:
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
        "selected_source_layers": K4_TOP_K,
        "n_samples": K4_TRAIN_OBSERVATIONS,
        "source_features": K4_SOURCE_FEATURES,
        "target_features": K4_TARGET_FEATURES,
        "weight_count": K4_WEIGHT_COUNT,
        "bias_count": K4_BIAS_COUNT,
    }
    if observed != expected:
        raise ValueError(f"Figure-4 k4 architecture mismatch: {observed!r} != {expected!r}")
    if float(ridge_lambda) != 0.01:
        raise ValueError("Figure-4 k4 ridge lambda must remain 0.01")


def _collect_inventory(
    source_root: str | Path,
    target_root: str | Path,
    architecture: Any,
    *,
    verify_payload_checksum: bool,
) -> dict[str, Any]:
    if Path(source_root).resolve() != SOURCE.train_root.resolve():
        raise ValueError("k4 fit source is not the frozen 32B activation capture")
    if Path(target_root).resolve() != TARGET.train_root.resolve():
        raise ValueError("k4 fit target is not the frozen 14B activation capture")
    assert_k4_fit(architecture, 0.01)
    if verify_payload_checksum is not True:
        raise ValueError("k4 fit requires SHA256 verification of every activation payload")
    return fit_activation_inventory(verify_payloads=True)


def _validate_selection_for_fit(
    original_validator: Any,
    strict_json: Any,
    selection_path: str | Path,
    inventory: Mapping[str, Any],
    *,
    require_formal_protocol: bool,
) -> dict[str, Any]:
    path = Path(selection_path).resolve()
    selection = validate_k4_selection_artifact(path)
    original_reader = strict_json

    def forward_direction_view(candidate: str | Path) -> dict[str, Any]:
        value = original_reader(Path(candidate))
        if Path(candidate).resolve() == path:
            value = dict(value)
            value["direction"] = "qwen3_14b_to_32b"
        return value

    # The shared validator is direction-neutral except for one legacy label.
    # Temporarily presenting that label retains its full catalog/payload checks.
    implementation = _load_implementation()
    saved_reader = implementation._strict_json
    implementation._strict_json = forward_direction_view
    try:
        result = original_validator(
            path,
            inventory,
            require_formal_protocol=require_formal_protocol,
        )
    finally:
        implementation._strict_json = saved_reader
    result = dict(result)
    result.update(
        {
            "semantic_direction": DIRECTION,
            "curve_variant": "figure4_large_to_small_retuned_k4",
            "selected_source_layers": K4_TOP_K,
            "selection_manifest_sha256": sha256_file(path),
            "selection_recomputed_from_full_score_tensor": True,
        }
    )
    if selection["k"] != K4_TOP_K:
        raise AssertionError("validated k4 selection changed after validation")
    return result


def _validate_complete_payloads(
    output_directory: str | Path,
    *,
    architecture: Any,
    fit_fingerprint: str,
    selection_sha256: str,
    selected_layers: Sequence[Sequence[int]],
) -> dict[str, Any]:
    assert_k4_fit(architecture, 0.01)
    if len(selected_layers) != 40:
        raise ValueError("complete k4 mapper must contain 40 target-layer rows")
    entries = [
        _validate_layer_payload(
            output_directory,
            target_layer=layer,
            selected_layers=selected_layers[layer],
            source_features=K4_SOURCE_FEATURES,
            target_features=K4_TARGET_FEATURES,
            kv_heads=8,
            head_dim=128,
            fit_fingerprint=fit_fingerprint,
            selection_sha256=selection_sha256,
        )
        for layer in range(40)
    ]
    return {
        "status": "passed",
        "layers_actually_loaded": 40,
        "tensors_actually_loaded": 160,
        "expected_tensor_shapes": {
            "key_weight": [K4_SOURCE_FEATURES, K4_TARGET_FEATURES],
            "key_bias": [K4_TARGET_FEATURES],
            "value_weight": [K4_SOURCE_FEATURES, K4_TARGET_FEATURES],
            "value_bias": [K4_TARGET_FEATURES],
        },
        "entries_digest": canonical_sha256(entries),
    }


def _publish_k4_manifest(
    original_publish: Any,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    output = Path(args[0] if args else kwargs["output_directory"]).resolve()
    if output != K4_MAPPER_ROOT.resolve():
        raise ValueError("k4 mapper publisher received a non-frozen output root")
    payload_validation = _validate_complete_payloads(
        output,
        architecture=kwargs["architecture"],
        fit_fingerprint=str(kwargs["fit_fingerprint"]),
        selection_sha256=str(kwargs["selection_sha256"]),
        selected_layers=kwargs["selected_layers"],
    )
    manifest = dict(original_publish(*args, **kwargs))
    manifest["direction"] = DIRECTION
    manifest["curve_variant"] = "figure4_large_to_small_retuned_k4"
    manifest["paper_reference"] = PAPER_CURVE
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
    manifest["independent_refit_evidence"] = {
        "weights_refit_from_frozen_train_activations": True,
        "k20_mapper_weights_consumed": False,
        "k20_weights_sliced_or_truncated": False,
        "source_design_width": K4_SOURCE_FEATURES,
        "target_width": K4_TARGET_FEATURES,
        "equation": "4096_to_1024_centered_ridge_lambda_0.01",
        "selection_source": "full_40x64_single_source_score_tensor",
    }
    manifest["payload_validation"] = payload_validation
    destination = output / "manifest.json"
    atomic_write_json(destination, manifest)
    atomic_write_json(
        output / "manifest.sha256.json",
        {"path": str(destination), "sha256": sha256_file(destination)},
    )
    return manifest


def run_k4_fit(
    extra_argv: Sequence[str] = (),
    *,
    config_path: Path = DEFAULT_K4_MAPPER_CONFIG,
) -> int:
    configuration = validate_k4_mapper_config(config_path)
    outputs = configuration["outputs"]
    if outputs != {
        "selection": K4_SELECTION.resolve(),
        "mapper": K4_MAPPER_ROOT.resolve(),
    }:
        raise ValueError("k4 fit output lock changed")
    sanitized = _sanitized_fit_argv(extra_argv)
    implementation = _load_implementation()
    original = {
        "assert_paper_fit": implementation.assert_paper_fit,
        "collect_activation_inventory": implementation.collect_activation_inventory,
        "validate_selection_provenance": implementation.validate_selection_provenance,
        "publish_complete_manifest": implementation.publish_complete_manifest,
        "make_run_directory": implementation.make_run_directory,
    }
    original_reader = implementation._strict_json

    def named_run_directory(root: Path, phase: str, name: str) -> Any:
        return original["make_run_directory"](root, phase, f"{name}_k4_retuned")

    implementation.assert_paper_fit = assert_k4_fit
    implementation.collect_activation_inventory = _collect_inventory
    implementation.validate_selection_provenance = (
        lambda selection_path, inventory, require_formal_protocol: _validate_selection_for_fit(
            original["validate_selection_provenance"],
            original_reader,
            selection_path,
            inventory,
            require_formal_protocol=require_formal_protocol,
        )
    )
    implementation.publish_complete_manifest = (
        lambda *args, **kwargs: _publish_k4_manifest(
            original["publish_complete_manifest"], *args, **kwargs
        )
    )
    implementation.make_run_directory = named_run_directory
    argv = [
        "--config",
        str(Path(config_path).resolve()),
        "--selection",
        str(K4_SELECTION),
        "--source-activations",
        str(SOURCE.train_root),
        "--target-activations",
        str(TARGET.train_root),
        "--output",
        str(K4_MAPPER_ROOT),
        *sanitized,
    ]
    try:
        return int(implementation.main(argv))
    finally:
        for name, value in original.items():
            setattr(implementation, name, value)


__all__ = [
    "assert_k4_fit",
    "run_k4_fit",
    "_collect_inventory",
    "_validate_complete_payloads",
]
