from types import SimpleNamespace

import pytest

from kvcache_transfer.config import PAPER_ARCHITECTURE, assert_paper_architecture


def test_exact_mapper_size() -> None:
    assert PAPER_ARCHITECTURE.source_features == 8192
    assert PAPER_ARCHITECTURE.target_features == 1024
    assert PAPER_ARCHITECTURE.mapper_weight_count == 1_073_741_824
    assert PAPER_ARCHITECTURE.mapper_bias_count == 131_072


def test_architecture_gate() -> None:
    valid = SimpleNamespace(
        model_type="qwen3",
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=1_000_000,
        use_sliding_window=False,
        sliding_window=None,
    )
    assert_paper_architecture(valid)
    valid.head_dim = 64
    with pytest.raises(ValueError, match="head_dim"):
        assert_paper_architecture(valid)


def test_role_specific_architecture_gate() -> None:
    source = SimpleNamespace(
        model_type="qwen3",
        num_key_value_heads=8,
        head_dim=128,
        rope_theta=1_000_000,
        use_sliding_window=False,
        sliding_window=None,
        num_hidden_layers=40,
        num_attention_heads=40,
        hidden_size=5120,
        intermediate_size=17408,
    )
    assert_paper_architecture(source, role="source")
    with pytest.raises(ValueError, match="num_hidden_layers"):
        assert_paper_architecture(source, role="target")
