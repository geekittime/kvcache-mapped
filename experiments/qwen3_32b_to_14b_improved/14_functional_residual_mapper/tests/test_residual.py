from __future__ import annotations

import torch
import pytest

from src.residual import (
    FunctionalResidualMapper,
    HeadStructuredResidual,
    estimated_state_bytes,
    parameter_count_per_active_layer,
)


def make_module(**kwargs):
    defaults = dict(
        input_dim=12,
        rank=3,
        kv_heads=2,
        head_dim=4,
        feature_mean=torch.zeros(12),
        feature_inv_std=torch.ones(12),
        rms_q99=2.0,
        absmax_q99=4.0,
    )
    defaults.update(kwargs)
    return HeadStructuredResidual(**defaults)


def test_zero_initialized_decoder_is_exact_anchor():
    torch.manual_seed(2)
    module = make_module()
    source = torch.randn(2, 5, 12)
    anchor = torch.randn(2, 5, 2, 4)
    corrected, diagnostics = module(source, anchor)
    assert torch.equal(corrected, anchor)
    assert torch.all(diagnostics.ood_gate == 1)
    assert torch.all(diagnostics.cap_scale == 1)


def test_ood_and_nonfinite_inputs_fall_back_to_anchor():
    module = make_module()
    anchor = torch.randn(1, 2, 2, 4)
    too_far = torch.full((1, 2, 12), 100.0)
    corrected_far, diag_far = module(too_far, anchor)
    assert torch.equal(corrected_far, anchor)
    assert torch.all(diag_far.fallback_mask)
    nonfinite = torch.zeros(1, 2, 12)
    nonfinite[..., 0] = float("nan")
    corrected_nan, diag_nan = module(nonfinite, anchor)
    assert torch.equal(corrected_nan, anchor)
    assert torch.all(diag_nan.fallback_mask)


def test_residual_is_finite_and_norm_capped():
    torch.manual_seed(3)
    module = make_module(rms_q99=0.1, absmax_q99=0.1)
    with torch.no_grad():
        module.decoder.fill_(10.0)
        module.delta_bias.fill_(10.0)
    source = torch.randn(2, 3, 12) * 0.01
    anchor = torch.randn(2, 3, 2, 4) + 3.0
    corrected, diagnostics = module(source, anchor)
    assert torch.isfinite(corrected).all()
    actual_ratio = (corrected - anchor).square().mean(dim=-1).sqrt() / anchor.square().mean(dim=-1).sqrt().clamp_min(1e-12)
    assert float(actual_ratio.max()) <= 0.250001
    assert torch.all((diagnostics.ood_gate >= 0) & (diagnostics.ood_gate <= 1))


def test_parameter_formula_and_strict_dimension_checks():
    assert parameter_count_per_active_layer(input_dim=20480, rank=64, kv_heads=8, head_dim=128) == 2754576
    assert estimated_state_bytes(10, 16) == 160
    with pytest.raises(ValueError):
        parameter_count_per_active_layer(input_dim=1, rank=True, kv_heads=1, head_dim=1)
    with pytest.raises(ValueError):
        make_module(feature_inv_std=torch.ones(11))


def test_mapper_only_corrects_declared_active_layers():
    stats = {}
    for component in ("key", "value"):
        stats[(7, component)] = {
            "feature_mean": torch.zeros(12),
            "feature_inv_std": torch.ones(12),
            "rms_q99": 2.0,
            "absmax_q99": 4.0,
        }
    mapper = FunctionalResidualMapper(
        active_layers=[7],
        layer_loss_weights={7: 1.0},
        input_dim=12,
        rank=3,
        kv_heads=2,
        head_dim=4,
        statistics=stats,
        target_layers=40,
    )
    assert mapper.expected_parameter_count() == 2 * (12 * 3 + 2 * 3 * 4 + 2 * 4 + 2)
    with pytest.raises(KeyError):
        mapper.module_for(8, "key")
