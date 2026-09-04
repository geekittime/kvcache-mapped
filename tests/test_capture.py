import torch

from kvcache_transfer.capture import sampled_positions


def test_paper_stride_positions() -> None:
    positions = sampled_positions(1024, stride=4)
    assert positions.shape == (256,)
    assert positions[0].item() == 0
    assert positions[-1].item() == 1020

