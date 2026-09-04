from __future__ import annotations

from types import SimpleNamespace

import pytest

from reverse_kv.fit_adapter import assert_reverse_paper_fit
from reverse_kv.protocol import (
    BIAS_COUNT,
    SOURCE_FEATURES,
    TARGET_FEATURES,
    TOP_K,
    TRAIN_OBSERVATIONS,
    WEIGHT_COUNT,
    validate_config,
)


def test_formal_reverse_dimensions() -> None:
    assert TOP_K == 20
    assert SOURCE_FEATURES == 20 * 8 * 128 == 20_480
    assert TARGET_FEATURES == 8 * 128 == 1_024
    assert WEIGHT_COUNT == 2 * 40 * SOURCE_FEATURES * TARGET_FEATURES
    assert WEIGHT_COUNT == 1_677_721_600
    assert BIAS_COUNT == 2 * 40 * TARGET_FEATURES == 81_920
    assert (WEIGHT_COUNT + BIAS_COUNT) * 4 / (1024**3) == 6.25030517578125


def test_config_is_formal_and_isolated() -> None:
    configuration = validate_config()
    assert configuration["raw"]["paper"]["selected_source_layers"] == TOP_K
    assert configuration["outputs"]["mapper"].name == "k20_lambda0.01"


def test_fit_gate_accepts_only_reverse_paper_shape() -> None:
    architecture = SimpleNamespace(
        source_layers=64,
        target_layers=40,
        kv_heads=8,
        head_dim=128,
        selected_source_layers=TOP_K,
        n_samples=TRAIN_OBSERVATIONS,
        source_features=SOURCE_FEATURES,
        target_features=TARGET_FEATURES,
        expected_weight_count=WEIGHT_COUNT,
        expected_bias_count=BIAS_COUNT,
    )
    assert_reverse_paper_fit(architecture, 0.01)
    architecture.selected_source_layers = 8
    architecture.source_features = 8 * 8 * 128
    architecture.expected_weight_count = 2 * 40 * architecture.source_features * 1024
    with pytest.raises(ValueError, match="reverse paper architecture mismatch"):
        assert_reverse_paper_fit(architecture, 0.01)
