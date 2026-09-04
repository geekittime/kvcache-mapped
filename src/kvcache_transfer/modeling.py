"""Validated loading and deterministic device maps for the Qwen3 pair."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from .config import assert_paper_architecture


def resolve_contiguous_layer_counts(
    *,
    num_hidden_layers: int,
    devices: Iterable[int],
    layer_counts: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Resolve and strictly validate one contiguous decoder-layer allocation.

    ``layer_counts=None`` preserves the repository's original approximately
    even contiguous assignment exactly.  Explicit counts are intentionally
    strict: every value must be an exact positive integer, there must be one
    value per device, and their sum must equal the checkpoint's decoder-layer
    count.  This makes a weighted memory layout part of the reproducible model
    loading contract rather than a best-effort Accelerate hint.
    """

    if type(num_hidden_layers) is not int or num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be an exact positive integer")
    device_ids = tuple(int(device) for device in devices)
    if not device_ids:
        raise ValueError("at least one CUDA device is required")

    if layer_counts is None:
        counts = [0] * len(device_ids)
        for layer in range(num_hidden_layers):
            bucket = min(
                layer * len(device_ids) // num_hidden_layers,
                len(device_ids) - 1,
            )
            counts[bucket] += 1
        return tuple(counts)

    try:
        raw_counts = tuple(layer_counts)
    except TypeError as exc:
        raise ValueError(
            "layer_counts must be an iterable of exact positive integers"
        ) from exc
    if any(type(count) is not int for count in raw_counts):
        raise ValueError(
            "layer_counts must contain exact integers, not bools or floats"
        )
    if any(count <= 0 for count in raw_counts):
        raise ValueError("layer_counts must contain only positive integers")
    if len(raw_counts) != len(device_ids):
        raise ValueError(
            "layer_counts length must equal the number of devices "
            f"({len(raw_counts)} != {len(device_ids)})"
        )
    if sum(raw_counts) != num_hidden_layers:
        raise ValueError(
            "layer_counts must sum exactly to num_hidden_layers "
            f"({sum(raw_counts)} != {num_hidden_layers})"
        )
    return raw_counts


def _expected_layer_devices(
    device_ids: Sequence[int],
    counts: Sequence[int],
) -> tuple[int, ...]:
    return tuple(
        device
        for device, count in zip(device_ids, counts, strict=True)
        for _ in range(count)
    )


def contiguous_device_map(
    model_path: str | Path,
    devices: Iterable[int],
    *,
    layer_counts: Iterable[int] | None = None,
) -> dict[str, int]:
    """Assign contiguous decoder-layer ranges and pin embeddings/head sanely."""
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    device_ids = tuple(int(device) for device in devices)
    counts = resolve_contiguous_layer_counts(
        num_hidden_layers=int(config.num_hidden_layers),
        devices=device_ids,
        layer_counts=layer_counts,
    )
    result: dict[str, int] = {
        "model.embed_tokens": device_ids[0],
        "model.norm": device_ids[-1],
        "lm_head": device_ids[-1],
    }
    for layer, device in enumerate(_expected_layer_devices(device_ids, counts)):
        result[f"model.layers.{layer}"] = device
    return result


def model_layer_sharding_evidence(
    model: torch.nn.Module,
    *,
    devices: Iterable[int],
    layer_counts: Iterable[int] | None = None,
    role: str | None = None,
) -> dict[str, Any]:
    """Assert the loaded decoder layers match the requested contiguous map.

    The assertion inspects the real device of every loaded decoder layer, not
    merely ``hf_device_map``.  Its JSON-compatible return value is suitable for
    immutable runtime/setup evidence.
    """

    device_ids = tuple(int(device) for device in devices)
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise ValueError("model has no decoder layer collection at model.layers")
    total_layers = len(layers)
    counts = resolve_contiguous_layer_counts(
        num_hidden_layers=total_layers,
        devices=device_ids,
        layer_counts=layer_counts,
    )
    expected_devices = _expected_layer_devices(device_ids, counts)
    if len(expected_devices) != total_layers:
        raise RuntimeError(
            "resolved layer-device map length differs from the loaded model"
        )

    def inspect_module_device(
        module: Any,
        *,
        module_name: str,
        expected_device: int,
    ) -> dict[str, int]:
        parameter_count = 0
        actual_device: int | None = None
        for parameter_index, parameter in enumerate(module.parameters()):
            parameter_count += 1
            actual = parameter.device
            if actual.type != "cuda" or actual.index is None:
                raise RuntimeError(
                    f"{module_name} parameter {parameter_index} is on {actual}, "
                    f"expected cuda:{expected_device}"
                )
            current_device = int(actual.index)
            if current_device != expected_device:
                raise RuntimeError(
                    f"{module_name} parameter {parameter_index} is on "
                    f"cuda:{current_device}, expected cuda:{expected_device}"
                )
            if actual_device is None:
                actual_device = current_device
        if actual_device is None:
            raise RuntimeError(f"{module_name} has no parameters to locate")
        return {
            "expected_device": expected_device,
            "actual_device": actual_device,
            "parameter_tensors_inspected": parameter_count,
        }

    actual_devices: list[int] = []
    parameter_tensors_inspected = 0
    for index, (layer, expected_device) in enumerate(
        zip(layers, expected_devices, strict=True)
    ):
        layer_device: int | None = None
        for parameter_index, parameter in enumerate(layer.parameters()):
            parameter_tensors_inspected += 1
            actual = parameter.device
            if actual.type != "cuda" or actual.index is None:
                raise RuntimeError(
                    f"decoder layer {index} parameter {parameter_index} is on {actual}, "
                    f"expected cuda:{expected_device}"
                )
            actual_device = int(actual.index)
            if actual_device != expected_device:
                raise RuntimeError(
                    f"decoder layer {index} parameter {parameter_index} is on "
                    f"cuda:{actual_device}, expected cuda:{expected_device}"
                )
            if layer_device is None:
                layer_device = actual_device
        if layer_device is None:
            raise RuntimeError(f"decoder layer {index} has no parameters to locate")
        actual_devices.append(layer_device)

    auxiliary_specs = (
        ("model.embed_tokens", getattr(base, "embed_tokens", None), device_ids[0]),
        ("model.norm", getattr(base, "norm", None), device_ids[-1]),
        ("lm_head", getattr(model, "lm_head", None), device_ids[-1]),
    )
    auxiliary_modules: dict[str, dict[str, int]] = {}
    for module_name, module, expected_device in auxiliary_specs:
        if module is None:
            raise ValueError(f"model has no auxiliary module {module_name}")
        auxiliary_modules[module_name] = inspect_module_device(
            module,
            module_name=module_name,
            expected_device=expected_device,
        )
    auxiliary_parameter_tensors_inspected = sum(
        item["parameter_tensors_inspected"] for item in auxiliary_modules.values()
    )

    actual_counts = tuple(actual_devices.count(device) for device in device_ids)
    if actual_counts != counts:
        raise RuntimeError(
            f"actual decoder-layer counts {actual_counts} differ from expected {counts}"
        )
    ranges: list[dict[str, int]] = []
    start = 0
    for device, count in zip(device_ids, counts, strict=True):
        ranges.append(
            {
                "device": device,
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "count": count,
            }
        )
        start += count
    return {
        "status": "passed",
        "role": role,
        "mode": (
            "explicit_exact_counts"
            if layer_counts is not None
            else "legacy_even_contiguous"
        ),
        "devices": list(device_ids),
        "declared_layer_counts": list(counts) if layer_counts is not None else None,
        "expected_layer_counts": list(counts),
        "actual_layer_counts": list(actual_counts),
        "total_layers": total_layers,
        "layer_devices": actual_devices,
        "contiguous_ranges": ranges,
        "actual_parameters_inspected_for_every_layer": True,
        "parameter_tensors_inspected": parameter_tensors_inspected,
        "auxiliary_modules": auxiliary_modules,
        "auxiliary_parameter_tensors_inspected": auxiliary_parameter_tensors_inspected,
        "auxiliary_parameters_inspected_for_every_module": True,
    }


def load_qwen3(
    model_path: str | Path,
    *,
    devices: Iterable[int],
    layer_counts: Iterable[int] | None = None,
    role: str | None = None,
    attention_implementation: str = "flash_attention_2",
) -> torch.nn.Module:
    """Load one local checkpoint in BF16 after asserting paper invariants."""
    device_ids = tuple(int(device) for device in devices)
    declared_counts = None if layer_counts is None else tuple(layer_counts)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    assert_paper_architecture(config, role=role)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention_implementation,
        device_map=contiguous_device_map(
            model_path,
            device_ids,
            layer_counts=declared_counts,
        ),
        low_cpu_mem_usage=True,
    )
    model.eval()
    if model.training:
        raise RuntimeError("model must be in evaluation mode")
    evidence = model_layer_sharding_evidence(
        model,
        devices=device_ids,
        layer_counts=declared_counts,
        role=role,
    )
    # Retain the exact post-load assertion for callers that need to persist it
    # without re-inspecting the model.
    setattr(model, "_kvcache_transfer_layer_sharding_evidence", evidence)
    return model
