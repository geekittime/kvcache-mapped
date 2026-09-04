from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from kvcache_transfer.modeling import (
    contiguous_device_map,
    model_layer_sharding_evidence,
    resolve_contiguous_layer_counts,
)


class _FakeLayer:
    def __init__(self, device: str, parameter_count: int = 1) -> None:
        self._parameters = tuple(
            SimpleNamespace(device=torch.device(device)) for _ in range(parameter_count)
        )

    def parameters(self):
        return iter(self._parameters)


def _fake_model(layer_devices: list[str], *, layer_parameter_count: int = 1):
    first = layer_devices[0]
    last = layer_devices[-1]
    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                _FakeLayer(device, parameter_count=layer_parameter_count)
                for device in layer_devices
            ],
            embed_tokens=_FakeLayer(first),
            norm=_FakeLayer(last),
        ),
        lm_head=_FakeLayer(last),
    )


def test_legacy_layer_count_resolution_preserves_original_bucket_formula() -> None:
    assert resolve_contiguous_layer_counts(
        num_hidden_layers=8,
        devices=(2, 4, 6),
    ) == (3, 3, 2)
    assert resolve_contiguous_layer_counts(
        num_hidden_layers=64,
        devices=(0, 1, 2, 3, 4),
    ) == (13, 13, 13, 13, 12)


def test_explicit_layer_counts_are_exact_and_allow_repeated_values() -> None:
    assert resolve_contiguous_layer_counts(
        num_hidden_layers=64,
        devices=(0, 1, 2, 3, 4),
        layer_counts=(8, 15, 14, 14, 13),
    ) == (8, 15, 14, 14, 13)
    assert resolve_contiguous_layer_counts(
        num_hidden_layers=40,
        devices=(5, 6, 7),
        layer_counts=(14, 13, 13),
    ) == (14, 13, 13)


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        ((8, 14), "length"),
        ((8, 14, 14, 14, 13), "sum exactly"),
        ((8, 14, 14, 14, 0), "positive"),
        ((8, 14, 14, 14, True), "exact integers"),
        ((8, 15, 14, 14, 13.0), "exact integers"),
    ],
)
def test_explicit_layer_count_validation_fails_closed(counts, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_contiguous_layer_counts(
            num_hidden_layers=64,
            devices=(0, 1, 2, 3, 4),
            layer_counts=counts,
        )


def test_contiguous_device_map_uses_exact_declared_ranges(tmp_path) -> None:
    Qwen3Config(num_hidden_layers=8).save_pretrained(tmp_path)
    observed = contiguous_device_map(
        tmp_path,
        (2, 4, 6),
        layer_counts=(1, 3, 4),
    )
    assert observed["model.embed_tokens"] == 2
    assert observed["model.norm"] == observed["lm_head"] == 6
    assert [observed[f"model.layers.{index}"] for index in range(8)] == [
        2,
        4,
        4,
        4,
        6,
        6,
        6,
        6,
    ]


def test_actual_loaded_layer_devices_are_asserted_and_serialized() -> None:
    model = _fake_model(["cuda:0"] * 2 + ["cuda:1"] * 3 + ["cuda:2"] * 3)
    evidence = model_layer_sharding_evidence(
        model,
        devices=(0, 1, 2),
        layer_counts=(2, 3, 3),
        role="target",
    )
    assert evidence == {
        "status": "passed",
        "role": "target",
        "mode": "explicit_exact_counts",
        "devices": [0, 1, 2],
        "declared_layer_counts": [2, 3, 3],
        "expected_layer_counts": [2, 3, 3],
        "actual_layer_counts": [2, 3, 3],
        "total_layers": 8,
        "layer_devices": [0, 0, 1, 1, 1, 2, 2, 2],
        "contiguous_ranges": [
            {"device": 0, "start_layer": 0, "end_layer_exclusive": 2, "count": 2},
            {"device": 1, "start_layer": 2, "end_layer_exclusive": 5, "count": 3},
            {"device": 2, "start_layer": 5, "end_layer_exclusive": 8, "count": 3},
        ],
        "actual_parameters_inspected_for_every_layer": True,
        "parameter_tensors_inspected": 8,
        "auxiliary_modules": {
            "model.embed_tokens": {
                "expected_device": 0,
                "actual_device": 0,
                "parameter_tensors_inspected": 1,
            },
            "model.norm": {
                "expected_device": 2,
                "actual_device": 2,
                "parameter_tensors_inspected": 1,
            },
            "lm_head": {
                "expected_device": 2,
                "actual_device": 2,
                "parameter_tensors_inspected": 1,
            },
        },
        "auxiliary_parameter_tensors_inspected": 3,
        "auxiliary_parameters_inspected_for_every_module": True,
    }


def test_actual_loaded_layer_device_mismatch_fails_closed() -> None:
    model = _fake_model(["cuda:0", "cuda:1", "cuda:1", "cuda:1"])
    with pytest.raises(RuntimeError, match="decoder layer 1.*expected cuda:0"):
        model_layer_sharding_evidence(
            model,
            devices=(0, 1),
            layer_counts=(2, 2),
            role="source",
        )


def test_actual_loaded_layer_on_cpu_is_rejected() -> None:
    model = _fake_model(["cpu", "cuda:1"])
    with pytest.raises(RuntimeError, match="is on cpu"):
        model_layer_sharding_evidence(
            model,
            devices=(0, 1),
            layer_counts=(1, 1),
        )


def test_actual_auxiliary_module_device_mismatch_is_rejected() -> None:
    model = _fake_model(["cuda:0", "cuda:1"])
    model.lm_head = _FakeLayer("cuda:0")
    with pytest.raises(RuntimeError, match="lm_head parameter 0.*expected cuda:1"):
        model_layer_sharding_evidence(
            model,
            devices=(0, 1),
            layer_counts=(1, 1),
            role="target",
        )


@pytest.mark.parametrize(
    ("role", "devices", "counts", "expected_parameter_tensors"),
    [
        ("source", (5, 6, 7), (14, 13, 13), 440),
        ("target", (0, 1, 2, 3, 4), (8, 15, 14, 14, 13), 704),
    ],
)
def test_qwen3_production_sharding_evidence_has_exact_parameter_tensor_counts(
    role: str,
    devices: tuple[int, ...],
    counts: tuple[int, ...],
    expected_parameter_tensors: int,
) -> None:
    layer_devices = [
        f"cuda:{device}"
        for device, count in zip(devices, counts, strict=True)
        for _ in range(count)
    ]
    # Qwen3 decoder layers contain 11 parameter tensors: q/k/v/o projections,
    # q/k norms, gate/up/down projections, and two RMSNorm weights.
    model = _fake_model(layer_devices, layer_parameter_count=11)
    evidence = model_layer_sharding_evidence(
        model,
        devices=devices,
        layer_counts=counts,
        role=role,
    )
    assert evidence["parameter_tensors_inspected"] == expected_parameter_tensors
    assert evidence["auxiliary_parameter_tensors_inspected"] == 3
    assert {
        name: item["parameter_tensors_inspected"]
        for name, item in evidence["auxiliary_modules"].items()
    } == {"model.embed_tokens": 1, "model.norm": 1, "lm_head": 1}
