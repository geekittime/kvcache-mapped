import torch

from kvcache_transfer.mapper import LayerMapper, apply_layer_mapper
from kvcache_transfer.rope import apply_rope


def test_identity_like_mapper_preserves_selected_layer() -> None:
    torch.manual_seed(1)
    source_k = [torch.randn(1, 8, 5, 128)]
    source_v = [torch.randn(1, 8, 5, 128)]
    identity = torch.eye(1024)
    mapper = LayerMapper(
        source_layers=(0,),
        key_weight=identity,
        key_bias=torch.zeros(1024),
        value_weight=identity,
        value_bias=torch.zeros(1024),
    )
    cos = torch.ones(1, 5, 128)
    sin = torch.zeros(1, 5, 128)
    key, value = apply_layer_mapper(
        source_k, source_v, mapper, cos, sin, output_dtype=torch.float32
    )
    torch.testing.assert_close(key, source_k[0])
    torch.testing.assert_close(value, source_v[0])


def test_mapper_gathers_chunks_from_layer_devices() -> None:
    """Selected cache layers may reside on different Accelerate devices."""

    if not torch.cuda.is_available():
        return
    devices = [torch.device("cuda:0")]
    if torch.cuda.device_count() > 1:
        devices.append(torch.device("cuda:1"))
    else:
        devices.append(devices[0])
    source_k = [
        torch.randn(1, 8, 7, 128, device=devices[0]),
        torch.randn(1, 8, 7, 128, device=devices[1]),
    ]
    source_v = [value.clone() for value in source_k]
    expected = torch.cat(
        [value.cpu().permute(0, 2, 1, 3).flatten(2) for value in source_k], dim=-1
    )
    weight = torch.zeros(2048, 1024, device=devices[0])
    weight[:1024] = torch.eye(1024, device=devices[0])
    mapper = LayerMapper(
        source_layers=(0, 1),
        key_weight=weight,
        key_bias=torch.zeros(1024, device=devices[0]),
        value_weight=weight.clone(),
        value_bias=torch.zeros(1024, device=devices[0]),
    )
    cos = torch.ones(1, 7, 128, device=devices[0])
    sin = torch.zeros_like(cos)
    key, value = apply_layer_mapper(
        source_k,
        source_v,
        mapper,
        cos,
        sin,
        output_dtype=torch.float32,
        token_chunk=3,
    )
    expected_first_layer = expected[..., :1024].unflatten(-1, (8, 128)).transpose(1, 2)
    torch.testing.assert_close(key.cpu(), expected_first_layer)
    torch.testing.assert_close(value.cpu(), expected_first_layer)
