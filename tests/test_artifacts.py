import torch

from kvcache_transfer.artifacts import load_layer_mapper, save_layer_mapper
from kvcache_transfer.mapper import LayerMapper


def test_mapper_round_trip(tmp_path) -> None:
    mapper = LayerMapper(
        source_layers=(3,),
        key_weight=torch.randn(1024, 1024),
        key_bias=torch.randn(1024),
        value_weight=torch.randn(1024, 1024),
        value_bias=torch.randn(1024),
    )
    path = save_layer_mapper(tmp_path, 2, mapper)
    restored = load_layer_mapper(path)
    assert restored.source_layers == (3,)
    torch.testing.assert_close(restored.key_weight, mapper.key_weight)
    torch.testing.assert_close(restored.value_bias, mapper.value_bias)

