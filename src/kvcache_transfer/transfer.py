"""End-to-end Qwen3 KV-cache transfer and boundary replay runtime.

This module implements the serving half of the paper's method:

1. consume the source model's *real* post-RoPE :class:`DynamicCache`;
2. invert source RoPE in FP32 to recover position-independent content keys;
3. consume each FP32 mapper layer on the corresponding target decoder-layer
   device (either lazy-streamed or pre-materialized for repeated requests);
4. apply the independent K/V maps in persisted
   ``[selected-layer rank, KV head, head dimension]`` order;
5. apply target RoPE in FP32 and build a layer-sharded BF16 DynamicCache;
6. replay ``C[-1]`` on the target after mapping the source prefix ``C[:-1]``.

The low-memory runtime may stream the 4-GiB mapper, while benchmark evaluation
and latency runs may keep it resident so disk I/O is never charged per request.
Every safetensors payload and sidecar is hashed before first use.  The final
mapper manifest is itself guarded by ``manifest.sha256.json`` and must declare
all target layers complete before a production runtime is constructed.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from transformers.cache_utils import Cache, DynamicCache

from .artifacts import load_layer_mapper
from .cache_adapter import (
    boundary_replay_inputs,
    cache_sequence_length,
    cache_to_legacy,
    continuation_scoring_inputs,
    qwen3_layer_devices,
)
from .config import ArchitectureSpec, PAPER_ARCHITECTURE
from .io import sha256_file
from .mapper import LayerMapper, apply_layer_mapper
from .rope import qwen3_rope_cos_sin_from_config, remove_rope


class TransferValidationError(ValueError):
    """Raised when a mapper, cache, position, or model invariant is violated."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise TransferValidationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TransferValidationError(f"expected a JSON object in {path}")
    return value


def _layer_artifact_path(mapper_root: Path, value: Any, *, expected_name: str) -> Path:
    """Resolve a layer artifact while allowing a complete bundle to move.

    The training manifest records absolute paths for auditability.  A copied
    mapper must nevertheless remain self-contained, so ``root/layers`` takes
    precedence whenever the expected file is present.  A relative manifest
    path is resolved below the current bundle root.  The basename is fixed by
    target-layer index so a manifest cannot substitute an unrelated file.
    """

    candidate = Path(str(value))
    if candidate.name != expected_name:
        raise TransferValidationError(
            f"mapper layer artifact {candidate} has unexpected name; expected {expected_name}"
        )
    local = mapper_root / "layers" / expected_name
    if local.is_file():
        return local.resolve()
    if candidate.is_absolute():
        return candidate
    resolved = (mapper_root / candidate).resolve()
    try:
        resolved.relative_to(mapper_root)
    except ValueError as exc:
        raise TransferValidationError(
            f"relative mapper artifact escapes its bundle: {candidate}"
        ) from exc
    return resolved


def _architecture_from_manifest(raw: Mapping[str, Any]) -> ArchitectureSpec:
    try:
        return ArchitectureSpec(
            source_layers=int(raw["source_layers"]),
            target_layers=int(raw["target_layers"]),
            source_kv_heads=int(raw["kv_heads"]),
            target_kv_heads=int(raw["kv_heads"]),
            source_head_dim=int(raw["head_dim"]),
            target_head_dim=int(raw["head_dim"]),
            selected_source_layers=int(raw["selected_source_layers"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TransferValidationError("mapper manifest has an invalid architecture block") from exc


@dataclass(frozen=True)
class MapperLayerArtifact:
    target_layer: int
    path: Path
    sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    source_layers_rank_order: tuple[int, ...]


@dataclass(frozen=True)
class MapperBundle:
    """Fully verified, lazy-loading production mapper bundle."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    fit_fingerprint: str
    architecture: ArchitectureSpec
    layers: tuple[MapperLayerArtifact, ...]
    dtype: torch.dtype = torch.float32

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        expected_architecture: ArchitectureSpec | None = PAPER_ARCHITECTURE,
    ) -> "MapperBundle":
        """Verify a complete mapper manifest and every per-layer hash/shape."""

        root = Path(directory).resolve()
        manifest_path = root / "manifest.json"
        checksum_path = root / "manifest.sha256.json"
        try:
            manifest_hash = sha256_file(manifest_path)
        except OSError as exc:
            raise TransferValidationError(
                f"mapper bundle has no readable manifest: {manifest_path}"
            ) from exc
        checksum = _read_json(checksum_path)
        if checksum.get("sha256") != manifest_hash:
            raise TransferValidationError("mapper aggregate manifest SHA256 mismatch")
        manifest = _read_json(manifest_path)
        required_values = {
            "schema_version": 1,
            "status": "complete",
            "direction": "qwen3_14b_to_32b",
            "method": "closed-form-centered-ridge-kv-cache-transfer",
            "dtype": "float32",
            "allow_tf32": False,
        }
        for field, expected in required_values.items():
            if manifest.get(field) != expected:
                raise TransferValidationError(
                    f"mapper manifest {field}={manifest.get(field)!r}, expected {expected!r}"
                )
        architecture_raw = manifest.get("architecture")
        if not isinstance(architecture_raw, Mapping):
            raise TransferValidationError("mapper manifest lacks architecture")
        architecture = _architecture_from_manifest(architecture_raw)
        if expected_architecture is not None and architecture != expected_architecture:
            raise TransferValidationError(
                f"mapper architecture {architecture!r} does not match {expected_architecture!r}"
            )
        if architecture.source_kv_heads != architecture.target_kv_heads:
            raise TransferValidationError("runtime requires matched source/target KV-head counts")
        if architecture.source_head_dim != architecture.target_head_dim:
            raise TransferValidationError("runtime requires matched source/target head dimensions")
        if int(manifest.get("weight_count", -1)) != architecture.mapper_weight_count:
            raise TransferValidationError("mapper aggregate weight count is invalid")
        if int(manifest.get("bias_count", -1)) != architecture.mapper_bias_count:
            raise TransferValidationError("mapper aggregate bias count is invalid")
        fit_definition = manifest.get("fit_definition")
        if not isinstance(fit_definition, Mapping):
            raise TransferValidationError("mapper manifest lacks fit_definition")
        fit_invariants = {
            "method": "two-pass-centered-ridge",
            "ridge_lambda_added_directly_to_unnormalized_gram": 0.01,
            "standardize_features": False,
            "regularize_bias": False,
            "covariance_dtype": "float32",
            "mapper_dtype": "float32",
            "allow_tf32": False,
            "feature_order": "selected_layer_rank,kv_head,head_dim",
            "output_layout": "target_kv_head,target_head_dim",
        }
        for field, expected in fit_invariants.items():
            if fit_definition.get(field) != expected:
                raise TransferValidationError(
                    f"fit definition {field}={fit_definition.get(field)!r}, expected {expected!r}"
                )
        if float(manifest.get("ridge_lambda", -1.0)) != 0.01:
            raise TransferValidationError("production mapper ridge lambda is not 0.01")
        fit_fingerprint = str(manifest.get("fit_fingerprint", ""))
        if len(fit_fingerprint) != 64:
            raise TransferValidationError("mapper fit fingerprint is invalid")

        raw_layers = manifest.get("layers")
        if not isinstance(raw_layers, list) or len(raw_layers) != architecture.target_layers:
            raise TransferValidationError(
                f"mapper manifest must contain {architecture.target_layers} target layers"
            )
        layer_artifacts: list[MapperLayerArtifact | None] = [None] * architecture.target_layers
        selected_rows = manifest.get("selection", {}).get(
            "selected_source_layers_rank_order"
        )
        if not isinstance(selected_rows, list) or len(selected_rows) != architecture.target_layers:
            raise TransferValidationError("mapper manifest has invalid selected-layer rows")
        per_layer_weight_count = (
            2 * architecture.source_features * architecture.target_features
        )
        per_layer_bias_count = 2 * architecture.target_features
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, Mapping):
                raise TransferValidationError("mapper layer entry is not an object")
            try:
                target_layer = int(raw_layer["target_layer"])
            except (KeyError, TypeError, ValueError) as exc:
                raise TransferValidationError("mapper layer entry has no valid index") from exc
            if not 0 <= target_layer < architecture.target_layers:
                raise TransferValidationError(f"target layer {target_layer} is out of range")
            if layer_artifacts[target_layer] is not None:
                raise TransferValidationError(f"duplicate mapper target layer {target_layer}")
            selected = tuple(int(value) for value in raw_layer.get("source_layers_rank_order", ()))
            expected_selected = tuple(int(value) for value in selected_rows[target_layer])
            if selected != expected_selected:
                raise TransferValidationError(
                    f"target layer {target_layer} selection disagrees with aggregate selection"
                )
            if len(selected) != architecture.selected_source_layers or len(set(selected)) != len(selected):
                raise TransferValidationError(
                    f"target layer {target_layer} must select unique top-k source layers"
                )
            if any(value < 0 or value >= architecture.source_layers for value in selected):
                raise TransferValidationError(
                    f"target layer {target_layer} selects an out-of-range source layer"
                )
            if int(raw_layer.get("weight_numel", -1)) != per_layer_weight_count:
                raise TransferValidationError(f"target layer {target_layer} weight count is invalid")
            if int(raw_layer.get("bias_numel", -1)) != per_layer_bias_count:
                raise TransferValidationError(f"target layer {target_layer} bias count is invalid")
            stem = f"layer_{target_layer:03d}"
            payload_path = _layer_artifact_path(
                root,
                raw_layer.get("path"),
                expected_name=f"{stem}.safetensors",
            )
            sidecar_path = _layer_artifact_path(
                root,
                raw_layer.get("sidecar_path"),
                expected_name=f"{stem}.json",
            )
            if payload_path.with_suffix(".json") != sidecar_path:
                raise TransferValidationError(
                    f"target layer {target_layer} payload and sidecar are not colocated"
                )
            try:
                payload_hash = sha256_file(payload_path)
                sidecar_hash = sha256_file(sidecar_path)
            except OSError as exc:
                raise TransferValidationError(
                    f"target layer {target_layer} mapper artifact is missing: {exc}"
                ) from exc
            if payload_hash != raw_layer.get("sha256"):
                raise TransferValidationError(
                    f"target layer {target_layer} safetensors SHA256 mismatch"
                )
            if sidecar_hash != raw_layer.get("sidecar_sha256"):
                raise TransferValidationError(
                    f"target layer {target_layer} sidecar SHA256 mismatch"
                )
            sidecar = _read_json(sidecar_path)
            sidecar_expectations = {
                "target_layer": target_layer,
                "source_layers_rank_order": list(selected),
                "sha256": payload_hash,
                "dtype": "float32",
                "source_heads": architecture.source_kv_heads,
                "head_dim": architecture.source_head_dim,
                "weight_numel": per_layer_weight_count,
                "bias_numel": per_layer_bias_count,
            }
            for field, expected in sidecar_expectations.items():
                if sidecar.get(field) != expected:
                    raise TransferValidationError(
                        f"target layer {target_layer} sidecar {field} is invalid"
                    )
            metadata = sidecar.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("fit_fingerprint") != fit_fingerprint:
                raise TransferValidationError(
                    f"target layer {target_layer} fit fingerprint is invalid"
                )
            expected_shapes = {
                "key_weight": [architecture.source_features, architecture.target_features],
                "key_bias": [architecture.target_features],
                "value_weight": [architecture.source_features, architecture.target_features],
                "value_bias": [architecture.target_features],
            }
            if sidecar.get("tensor_shapes") != expected_shapes:
                raise TransferValidationError(
                    f"target layer {target_layer} mapper tensor shapes are invalid"
                )
            layer_artifacts[target_layer] = MapperLayerArtifact(
                target_layer=target_layer,
                path=payload_path,
                sha256=payload_hash,
                sidecar_path=sidecar_path,
                sidecar_sha256=sidecar_hash,
                source_layers_rank_order=selected,
            )
        if any(layer is None for layer in layer_artifacts):
            raise TransferValidationError("mapper target-layer coverage is incomplete")
        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_hash,
            fit_fingerprint=fit_fingerprint,
            architecture=architecture,
            layers=tuple(layer for layer in layer_artifacts if layer is not None),
        )

    def load_layer(
        self, target_layer: int, *, device: torch.device | str = "cpu"
    ) -> LayerMapper:
        """Load and re-verify one mapper layer on the requested device."""

        if not 0 <= target_layer < len(self.layers):
            raise TransferValidationError(f"target layer {target_layer} is out of range")
        artifact = self.layers[target_layer]
        try:
            current_hash = sha256_file(artifact.path)
            current_sidecar_hash = sha256_file(artifact.sidecar_path)
        except OSError as exc:
            raise TransferValidationError(
                f"target layer {target_layer} disappeared after bundle validation: {exc}"
            ) from exc
        if current_hash != artifact.sha256:
            raise TransferValidationError(
                f"target layer {target_layer} changed after bundle validation"
            )
        if current_sidecar_hash != artifact.sidecar_sha256:
            raise TransferValidationError(
                f"target layer {target_layer} sidecar changed after bundle validation"
            )
        mapper = load_layer_mapper(artifact.path, device=device)
        if mapper.source_layers != artifact.source_layers_rank_order:
            raise TransferValidationError(
                f"target layer {target_layer} loaded a different selected-layer order"
            )
        mapper.validate(
            source_heads=self.architecture.source_kv_heads,
            head_dim=self.architecture.source_head_dim,
        )
        tensor_dtypes = {
            mapper.key_weight.dtype,
            mapper.key_bias.dtype,
            mapper.value_weight.dtype,
            mapper.value_bias.dtype,
        }
        if tensor_dtypes != {torch.float32}:
            raise TransferValidationError(
                f"target layer {target_layer} mapper tensors are not all FP32: {tensor_dtypes}"
            )
        return mapper

    def iter_layers(
        self, layer_devices: Sequence[torch.device | str]
    ) -> Iterator[tuple[int, LayerMapper]]:
        """Stream exactly one mapper layer to each target-layer device."""

        if len(layer_devices) != self.architecture.target_layers:
            raise TransferValidationError(
                f"received {len(layer_devices)} target devices, expected "
                f"{self.architecture.target_layers}"
            )
        for target_layer, device in enumerate(layer_devices):
            yield target_layer, self.load_layer(target_layer, device=device)


def _model_architecture(model: torch.nn.Module) -> tuple[int, int, int]:
    config = getattr(model, "config", None)
    if config is None:
        raise TransferValidationError("model has no config")
    try:
        layers = int(config.num_hidden_layers)
        heads = int(config.num_key_value_heads)
        head_dim = int(config.head_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TransferValidationError("model lacks Qwen3 layer/KV/head dimensions") from exc
    return layers, heads, head_dim


def _validate_target_model(model: torch.nn.Module, architecture: ArchitectureSpec) -> None:
    observed = _model_architecture(model)
    expected = (
        architecture.target_layers,
        architecture.target_kv_heads,
        architecture.target_head_dim,
    )
    if observed != expected:
        raise TransferValidationError(
            f"target model layer/KV/head dimensions {observed}, expected {expected}"
        )
    config = model.config
    if float(config.rope_theta) != 1_000_000 or getattr(config, "rope_scaling", None) not in (
        None,
        {},
    ):
        raise TransferValidationError("runtime only accepts the paper's default Qwen3 RoPE")
    if getattr(config, "sliding_window", None) is not None:
        raise TransferValidationError("runtime does not support a sliding-window target cache")


def _validate_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    if position_ids is None:
        return torch.arange(sequence_length, device=device, dtype=torch.long).unsqueeze(0).expand(
            batch_size, -1
        )
    if not isinstance(position_ids, torch.Tensor) or position_ids.shape != (
        batch_size,
        sequence_length,
    ):
        raise TransferValidationError(
            f"position_ids must be [{batch_size},{sequence_length}]"
        )
    if position_ids.dtype not in (torch.int32, torch.int64):
        raise TransferValidationError("position_ids must contain integers")
    positions = position_ids.to(device=device, dtype=torch.long)
    if bool(torch.any(positions < 0)):
        raise TransferValidationError("position_ids must be non-negative")
    return positions


@torch.inference_mode()
def source_cache_to_content(
    source_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    source_config: Any,
    *,
    position_ids: torch.Tensor | None = None,
    expected_layers: int | None = None,
    expected_heads: int | None = None,
    expected_head_dim: int | None = None,
    strict_validation: bool = True,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
    """Invert source-cache RoPE in FP32 while preserving layer placement.

    ``strict_validation=True`` is the safe public default and performs the
    device reductions which diagnose singular RoPE frequencies and non-finite
    recovered keys.  A latency benchmark may set it to ``False`` only after an
    untimed strict conversion of the exact same cache/model/mapper request has
    passed.  Shape, dtype, architecture, and output validation remain enabled
    in both modes.
    """

    layers = cache_to_legacy(source_cache)
    if expected_layers is not None and len(layers) != expected_layers:
        raise TransferValidationError(
            f"source cache has {len(layers)} layers, expected {expected_layers}"
        )
    batch_size, heads, sequence_length, head_dim = layers[0][0].shape
    if expected_heads is not None and heads != expected_heads:
        raise TransferValidationError(f"source cache has {heads} heads, expected {expected_heads}")
    if expected_head_dim is not None and head_dim != expected_head_dim:
        raise TransferValidationError(
            f"source cache head dim is {head_dim}, expected {expected_head_dim}"
        )
    positions = _validate_position_ids(
        position_ids,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=layers[0][0].device,
    )
    keys: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    for layer_idx, (key, value) in enumerate(layers):
        if key.shape != (batch_size, heads, sequence_length, head_dim):
            raise TransferValidationError(
                f"source cache layer {layer_idx} has shape {tuple(key.shape)}"
            )
        if key.dtype != value.dtype:
            raise TransferValidationError(
                f"source cache layer {layer_idx} K/V dtypes differ"
            )
        layer_positions = positions.to(key.device)
        cos, sin = qwen3_rope_cos_sin_from_config(
            layer_positions,
            source_config,
            device=key.device,
            dtype=torch.float32,
        )
        content = remove_rope(
            key,
            cos,
            sin,
            output_dtype=torch.float32,
            validate=strict_validation,
        )
        if strict_validation and not bool(torch.isfinite(content).all()):
            raise TransferValidationError(
                f"source cache layer {layer_idx} inverse RoPE produced NaN/Inf"
            )
        keys.append(content)
        # V is already content-space.  Preserve BF16 storage and layer device;
        # apply_layer_mapper converts only the current token chunk to FP32.
        values.append(value.detach())
    return tuple(keys), tuple(values), positions


@torch.inference_mode()
def transfer_source_cache(
    source_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    source_config: Any,
    target_model: torch.nn.Module,
    mapper_bundle: MapperBundle,
    position_ids: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    token_chunk: int = 1024,
    strict_validation: bool = True,
) -> DynamicCache:
    """Convert a real source cache to a target-layer-sharded DynamicCache.

    Strict numerical validation is enabled by default.  The opt-out exists for
    synchronized latency loops, where those diagnostic reductions would add
    dozens of host/device barriers; callers must first validate the identical
    request with the default strict path.
    """

    if token_chunk <= 0:
        raise TransferValidationError("token_chunk must be positive")
    architecture = mapper_bundle.architecture
    _validate_target_model(target_model, architecture)
    target_devices = qwen3_layer_devices(target_model)
    base = getattr(target_model, "model", target_model)
    target_layers_modules = getattr(base, "layers")
    target_layer_dtypes: list[torch.dtype] = []
    for layer_idx, layer in enumerate(target_layers_modules):
        try:
            layer_dtype = next(layer.parameters()).dtype
        except StopIteration as exc:
            raise TransferValidationError(
                f"target decoder layer {layer_idx} has no parameters"
            ) from exc
        if not layer_dtype.is_floating_point:
            raise TransferValidationError(
                f"target decoder layer {layer_idx} has non-floating dtype {layer_dtype}"
            )
        target_layer_dtypes.append(layer_dtype)
    if output_dtype is None:
        if len(set(target_layer_dtypes)) != 1:
            raise TransferValidationError(
                f"target decoder layers use mixed dtypes: {set(target_layer_dtypes)}"
            )
        resolved_output_dtype = target_layer_dtypes[0]
    else:
        if not output_dtype.is_floating_point:
            raise TransferValidationError("mapped cache output dtype must be floating point")
        mismatches = [
            index
            for index, layer_dtype in enumerate(target_layer_dtypes)
            if layer_dtype != output_dtype
        ]
        if mismatches:
            raise TransferValidationError(
                f"explicit mapped-cache dtype {output_dtype} differs from target layer dtype "
                f"at layers {mismatches[:8]}"
            )
        resolved_output_dtype = output_dtype
    source_layers, source_heads, source_head_dim = (
        int(source_config.num_hidden_layers),
        int(source_config.num_key_value_heads),
        int(source_config.head_dim),
    )
    expected_source = (
        architecture.source_layers,
        architecture.source_kv_heads,
        architecture.source_head_dim,
    )
    if (source_layers, source_heads, source_head_dim) != expected_source:
        raise TransferValidationError(
            f"source config layer/KV/head dimensions "
            f"{(source_layers, source_heads, source_head_dim)}, expected {expected_source}"
        )
    if float(source_config.rope_theta) != 1_000_000 or getattr(
        source_config, "rope_scaling", None
    ) not in (None, {}):
        raise TransferValidationError("runtime only accepts the paper's default source RoPE")
    content_keys, values, positions = source_cache_to_content(
        source_cache,
        source_config,
        position_ids=position_ids,
        expected_layers=architecture.source_layers,
        expected_heads=architecture.source_kv_heads,
        expected_head_dim=architecture.source_head_dim,
        strict_validation=strict_validation,
    )
    result = DynamicCache()
    for target_layer, mapper in mapper_bundle.iter_layers(target_devices):
        target_device = target_devices[target_layer]
        cos, sin = qwen3_rope_cos_sin_from_config(
            positions.to(target_device),
            target_model.config,
            device=target_device,
            dtype=torch.float32,
        )
        key, value = apply_layer_mapper(
            content_keys,
            values,
            mapper,
            cos,
            sin,
            output_dtype=resolved_output_dtype,
            token_chunk=token_chunk,
            source_heads=architecture.source_kv_heads,
            head_dim=architecture.source_head_dim,
        )
        expected_shape = (
            content_keys[0].shape[0],
            architecture.target_kv_heads,
            content_keys[0].shape[2],
            architecture.target_head_dim,
        )
        if tuple(key.shape) != expected_shape or tuple(value.shape) != expected_shape:
            raise TransferValidationError(
                f"mapped target layer {target_layer} has invalid K/V shapes"
            )
        if key.device != target_device or value.device != target_device:
            raise TransferValidationError(
                f"mapped target layer {target_layer} is not on its decoder device"
            )
        if key.dtype != resolved_output_dtype or value.dtype != resolved_output_dtype:
            raise TransferValidationError(
                f"mapped target layer {target_layer} has the wrong cache dtype"
            )
        result.update(key.detach(), value.detach(), target_layer)
        # Drop this iteration's local reference.  A lazy bundle may now
        # reclaim the layer; a resident bundle intentionally retains it.
        del mapper
    if len(result) != architecture.target_layers:
        raise TransferValidationError("mapped DynamicCache is missing target layers")
    if cache_sequence_length(result) != positions.shape[1]:
        raise TransferValidationError("mapped cache sequence length changed")
    return result


def split_context_for_transfer(
    context_input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return source ``C[:-1]``, target boundary ``C[-1]``, and prefix mask."""

    if not isinstance(context_input_ids, torch.Tensor) or context_input_ids.ndim != 2:
        raise TransferValidationError("context_input_ids must have shape [batch,sequence]")
    if context_input_ids.dtype not in (torch.int32, torch.int64):
        raise TransferValidationError("context_input_ids must contain integers")
    if context_input_ids.shape[1] < 2:
        raise TransferValidationError("transfer context needs at least two tokens")
    prefix_mask = None
    if attention_mask is not None:
        if attention_mask.shape != context_input_ids.shape:
            raise TransferValidationError("attention_mask shape must equal context input shape")
        if not bool(torch.all((attention_mask == 0) | (attention_mask == 1))):
            raise TransferValidationError("attention_mask must be binary")
        if not bool(torch.all(attention_mask[:, -1] == 1)):
            raise TransferValidationError("the boundary token may not be padding")
        prefix_mask = attention_mask[:, :-1]
        if not bool(torch.all(prefix_mask == 1)):
            raise TransferValidationError(
                "mapped runtime currently requires an unpadded source prefix"
            )
    return context_input_ids[:, :-1], context_input_ids[:, -1:], prefix_mask


@torch.inference_mode()
def prefill_source_prefix(
    source_model: torch.nn.Module,
    context_input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[Cache, torch.Tensor, torch.Tensor | None]:
    """Prefill only ``C[:-1]`` on the source and return its real cache."""

    prefix_ids, boundary_ids, prefix_mask = split_context_for_transfer(
        context_input_ids, attention_mask=attention_mask
    )
    base = getattr(source_model, "model", source_model)
    embedding = getattr(base, "embed_tokens", None)
    if embedding is None:
        raise TransferValidationError("source model has no model.embed_tokens")
    source_device = embedding.weight.device
    prefix_ids = prefix_ids.to(source_device)
    model_inputs: dict[str, Any] = {
        "input_ids": prefix_ids,
        "use_cache": True,
        "return_dict": True,
    }
    if prefix_mask is not None:
        model_inputs["attention_mask"] = prefix_mask.to(source_device)
    output = base(**model_inputs)
    cache = output.past_key_values
    if not isinstance(cache, Cache):
        raise TransferValidationError("source model did not return a Transformers Cache")
    if cache_sequence_length(cache) != prefix_ids.shape[1]:
        raise TransferValidationError("source prefix cache has an unexpected length")
    return cache, boundary_ids, prefix_mask


@torch.inference_mode()
def target_boundary_replay(
    target_model: torch.nn.Module,
    boundary_input_ids: torch.Tensor,
    prefix_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    prefix_attention_mask: torch.Tensor | None = None,
    clone_cache: bool = True,
) -> Any:
    """Replay one boundary token on the target after an injected prefix."""

    base = getattr(target_model, "model", target_model)
    embedding = getattr(base, "embed_tokens", None)
    if embedding is None:
        raise TransferValidationError("target model has no model.embed_tokens")
    device = embedding.weight.device
    boundary = boundary_input_ids.to(device)
    mask = None if prefix_attention_mask is None else prefix_attention_mask.to(device)
    # A layer-sharded DynamicCache must already place each tensor beside the
    # corresponding decoder layer.  HF/Accelerate does not relocate past KV.
    layer_devices = qwen3_layer_devices(target_model)
    cache_layers = cache_to_legacy(prefix_cache)
    if len(cache_layers) != len(layer_devices):
        raise TransferValidationError(
            f"target prefix cache has {len(cache_layers)} layers, model has {len(layer_devices)}"
        )
    for layer_idx, ((key, value), expected_device) in enumerate(
        zip(cache_layers, layer_devices, strict=True)
    ):
        if key.device != expected_device or value.device != expected_device:
            raise TransferValidationError(
                f"target cache layer {layer_idx} is on {key.device}/{value.device}, "
                f"expected {expected_device}"
            )
    inputs = boundary_replay_inputs(
        boundary,
        prefix_cache,
        prefix_attention_mask=mask,
        clone_cache=clone_cache,
    )
    return target_model(**inputs)


@torch.inference_mode()
def transfer_prefill_and_boundary_replay(
    source_model: torch.nn.Module,
    target_model: torch.nn.Module,
    context_input_ids: torch.Tensor,
    mapper_bundle: MapperBundle,
    *,
    attention_mask: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    token_chunk: int = 1024,
) -> Any:
    """End-to-end paper protocol: source ``C[:-1]`` then target ``C[-1]``."""

    source_cache, boundary_ids, prefix_mask = prefill_source_prefix(
        source_model,
        context_input_ids,
        attention_mask=attention_mask,
    )
    mapped_cache = transfer_source_cache(
        source_cache,
        source_config=source_model.config,
        target_model=target_model,
        mapper_bundle=mapper_bundle,
        output_dtype=output_dtype,
        token_chunk=token_chunk,
    )
    return target_boundary_replay(
        target_model,
        boundary_ids,
        mapped_cache,
        prefix_attention_mask=prefix_mask,
        clone_cache=False,
    )


@torch.inference_mode()
def target_own_cache_boundary_oracle(
    target_model: torch.nn.Module,
    context_input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> Any:
    """Target-own-cache control using the identical ``C[:-1]`` replay protocol."""

    prefix_ids, boundary_ids, prefix_mask = split_context_for_transfer(
        context_input_ids, attention_mask=attention_mask
    )
    base = getattr(target_model, "model", target_model)
    embedding = getattr(base, "embed_tokens", None)
    if embedding is None:
        raise TransferValidationError("target model has no model.embed_tokens")
    device = embedding.weight.device
    inputs: dict[str, Any] = {
        "input_ids": prefix_ids.to(device),
        "use_cache": True,
        "return_dict": True,
    }
    if prefix_mask is not None:
        inputs["attention_mask"] = prefix_mask.to(device)
    prefix = base(**inputs).past_key_values
    return target_boundary_replay(
        target_model,
        boundary_ids,
        prefix,
        prefix_attention_mask=prefix_mask,
        clone_cache=False,
    )


def transfer_continuation_inputs(
    boundary_input_ids: torch.Tensor,
    continuation_input_ids: torch.Tensor,
    mapped_prefix_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    target_model: torch.nn.Module,
    prefix_attention_mask: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor | DynamicCache | bool], torch.Tensor]:
    """Prepare ``C[-1] + Y[:-1]`` for exact transfer log-likelihood."""

    base = getattr(target_model, "model", target_model)
    embedding = getattr(base, "embed_tokens", None)
    if embedding is None:
        raise TransferValidationError("target model has no model.embed_tokens")
    device = embedding.weight.device
    return continuation_scoring_inputs(
        boundary_input_ids.to(device),
        continuation_input_ids.to(device),
        mapped_prefix_cache,
        prefix_attention_mask=None
        if prefix_attention_mask is None
        else prefix_attention_mask.to(device),
        clone_cache=True,
    )


__all__ = [
    "MapperBundle",
    "MapperLayerArtifact",
    "TransferValidationError",
    "prefill_source_prefix",
    "source_cache_to_content",
    "split_context_for_transfer",
    "target_boundary_replay",
    "target_own_cache_boundary_oracle",
    "transfer_continuation_inputs",
    "transfer_prefill_and_boundary_replay",
    "transfer_source_cache",
]
