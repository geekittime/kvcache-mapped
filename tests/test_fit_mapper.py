import torch

from kvcache_transfer.fit_mapper import flatten_selected_layers, target_all_heads


def test_flatten_order_is_layer_then_head_then_dim() -> None:
    layer0 = torch.arange(2 * 2 * 3).reshape(2, 2, 3)
    layer1 = 100 + layer0
    flattened = flatten_selected_layers([layer0, layer1], [1, 0])
    torch.testing.assert_close(flattened[0], torch.tensor([100, 101, 102, 103, 104, 105, 0, 1, 2, 3, 4, 5]))


def test_target_all_heads() -> None:
    target = torch.arange(12).reshape(2, 2, 3)
    assert target_all_heads(target).shape == (2, 6)

