"""Production builders for top-k design matrices and full ridge layers.

The on-disk production pipeline lives in :mod:`scripts.fit_mapper`, while the
pure functions here deliberately stay independent of CLI and multiprocessing
state.  In particular, :class:`MMapLayerChunkFactory` fixes the design-matrix
feature order to ``[selected-layer rank, KV head, head dimension]`` and can be
replayed for the two centered-ridge passes plus FP64 reconstruction analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from .activation_store import MappedBF16Activations, open_bf16_activations
from .mapper import LayerMapper
from .ridge import CenteredRidgeStatistics, RidgeSolution, fit_centered_ridge_streaming


ActivationReader = Callable[[int, str, slice], torch.Tensor]


class MMapLayerChunkFactory:
    """Replayable BF16 mmap reader for one target layer/cache type.

    Source paths must already be in the selected layer's descending score
    order.  Each payload has logical shape ``[samples, heads, head_dim]``.
    Chunks remain BF16 on CPU; the ridge accumulator performs the only FP32
    conversion directly on its compute device.
    """

    def __init__(
        self,
        source_paths_rank_order: Sequence[str | Path],
        target_path: str | Path,
        *,
        chunk_samples: int,
        verify_checksum: bool = False,
    ) -> None:
        if not source_paths_rank_order:
            raise ValueError("at least one source activation path is required")
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        self.source_paths = tuple(Path(path) for path in source_paths_rank_order)
        self.target_path = Path(target_path)
        self.chunk_samples = int(chunk_samples)
        self.verify_checksum = bool(verify_checksum)
        self._sources: list[MappedBF16Activations] = []
        self._target: MappedBF16Activations | None = None
        self._shape: tuple[int, int, int] | None = None

    def __enter__(self) -> "MMapLayerChunkFactory":
        if self._sources or self._target is not None:
            raise RuntimeError("MMapLayerChunkFactory is already open")
        try:
            self._sources = [
                open_bf16_activations(path, verify_checksum=self.verify_checksum)
                for path in self.source_paths
            ]
            self._target = open_bf16_activations(
                self.target_path, verify_checksum=self.verify_checksum
            )
            shapes = [mapped.metadata.shape for mapped in self._sources]
            shapes.append(self._target.metadata.shape)
            if any(len(shape) != 3 for shape in shapes):
                raise ValueError(f"activation payloads must be [samples,heads,dim], got {shapes}")
            if len(set(shapes)) != 1:
                raise ValueError(f"source/target activation shapes differ: {shapes}")
            self._shape = tuple(int(value) for value in shapes[0])  # type: ignore[assignment]
        except BaseException:
            self.close()
            raise
        return self

    @property
    def activation_shape(self) -> tuple[int, int, int]:
        if self._shape is None:
            raise RuntimeError("MMapLayerChunkFactory must be opened first")
        return self._shape

    @property
    def source_features(self) -> int:
        _, heads, head_dim = self.activation_shape
        return len(self.source_paths) * heads * head_dim

    @property
    def target_features(self) -> int:
        _, heads, head_dim = self.activation_shape
        return heads * head_dim

    def __call__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        if self._target is None or self._shape is None:
            raise RuntimeError("MMapLayerChunkFactory must be opened first")
        n_samples = self._shape[0]
        for start in range(0, n_samples, self.chunk_samples):
            stop = min(start + self.chunk_samples, n_samples)
            selection = slice(start, stop)
            source_pieces = [
                mapped.to_torch(selection, dtype=torch.bfloat16).flatten(1)
                for mapped in self._sources
            ]
            x = torch.cat(source_pieces, dim=1)
            y = self._target.to_torch(selection, dtype=torch.bfloat16).flatten(1)
            yield x, y

    def close(self) -> None:
        for mapped in self._sources:
            mapped.close()
        self._sources = []
        if self._target is not None:
            self._target.close()
            self._target = None
        self._shape = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def flatten_selected_layers(
    tensors: Sequence[torch.Tensor], selected_layers: Sequence[int]
) -> torch.Tensor:
    """Flatten ``[N,H,D]`` layers into ``[N, layer×head×dim]`` rank order."""
    if not selected_layers:
        raise ValueError("at least one source layer is required")
    pieces = []
    samples = None
    for layer_index in selected_layers:
        value = tensors[layer_index]
        if value.ndim != 3:
            raise ValueError("calibration activation must have [tokens,heads,dim] shape")
        if samples is None:
            samples = value.shape[0]
        elif value.shape[0] != samples:
            raise ValueError("selected source layers have different token counts")
        pieces.append(value.flatten(1))
    return torch.cat(pieces, dim=1)


def target_all_heads(target: torch.Tensor) -> torch.Tensor:
    if target.ndim != 3:
        raise ValueError("target activation must have [tokens,heads,dim] shape")
    return target.flatten(1)


def fit_full_target_layer(
    chunk_factory_key: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    chunk_factory_value: Callable[[], Iterator[tuple[torch.Tensor, torch.Tensor]]],
    selected_layers: Sequence[int],
    *,
    ridge_lambda: float = 0.01,
    compute_device: str | torch.device = "cuda",
    allow_tf32: bool = False,
    source_heads: int = 8,
    head_dim: int = 128,
) -> tuple[LayerMapper, dict[str, CenteredRidgeStatistics]]:
    """Fit both independent K/V maps for one complete target layer."""
    key_solution, key_stats = fit_centered_ridge_streaming(
        chunk_factory_key,
        ridge_lambda,
        compute_device=compute_device,
        allow_tf32=allow_tf32,
    )
    value_solution, value_stats = fit_centered_ridge_streaming(
        chunk_factory_value,
        ridge_lambda,
        compute_device=compute_device,
        allow_tf32=allow_tf32,
    )
    mapper = LayerMapper(
        source_layers=tuple(int(index) for index in selected_layers),
        key_weight=key_solution.weight,
        key_bias=key_solution.bias,
        value_weight=value_solution.weight,
        value_bias=value_solution.bias,
    )
    mapper.validate(source_heads=source_heads, head_dim=head_dim)
    return mapper, {"key": key_stats, "value": value_stats}


__all__ = [
    "ActivationReader",
    "MMapLayerChunkFactory",
    "fit_full_target_layer",
    "flatten_selected_layers",
    "target_all_heads",
]
