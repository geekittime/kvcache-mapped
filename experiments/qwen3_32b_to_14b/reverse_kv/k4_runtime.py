"""Fail-closed runtime for the Figure-4 retuned Qwen3 32B -> 14B k=4 mapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .k4_protocol import (
    K4_BIAS_COUNT,
    K4_SOURCE_FEATURES,
    K4_TARGET_FEATURES,
    K4_TOP_K,
    K4_WEIGHT_COUNT,
)
from .k4_selection import PAPER_CURVE, validate_k4_selection_artifact
from .protocol import DIRECTION, canonical_sha256, read_json, sha256_file
from .provenance import assert_model_instance_identity, load_verified_train_inventory

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPOSITORY_ROOT / "src"
import sys

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.artifacts import load_layer_mapper  # noqa: E402
from kvcache_transfer.cache_adapter import cache_sequence_length, qwen3_layer_devices  # noqa: E402
from kvcache_transfer.config import ArchitectureSpec  # noqa: E402
from kvcache_transfer.latency import ResidentMapperBundle, materialize_mapper_on_target  # noqa: E402
from kvcache_transfer.mapper import LayerMapper  # noqa: E402
from kvcache_transfer.transfer import transfer_source_cache  # noqa: E402


K4_REVERSE_ARCHITECTURE = ArchitectureSpec(
    source_layers=64,
    target_layers=40,
    source_kv_heads=8,
    target_kv_heads=8,
    source_head_dim=128,
    target_head_dim=128,
    selected_source_layers=K4_TOP_K,
)


class K4RuntimeValidationError(ValueError):
    """A k4 artifact/model violates the frozen Figure-4 variant."""


def _local_artifact(root: Path, value: Any, name: str) -> Path:
    recorded = Path(str(value)).resolve()
    local = (root / "layers" / name).resolve()
    if recorded != local or not local.is_file():
        raise K4RuntimeValidationError(f"k4 bundle is not self-contained at {local}")
    return local


@dataclass(frozen=True)
class K4LayerArtifact:
    target_layer: int
    path: Path
    sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    source_layers_rank_order: tuple[int, ...]


@dataclass(frozen=True)
class K4ReverseMapperBundle:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    fit_fingerprint: str
    architecture: ArchitectureSpec
    layers: tuple[K4LayerArtifact, ...]
    source_model_identity: Mapping[str, str]
    target_model_identity: Mapping[str, str]
    dtype: torch.dtype = torch.float32

    @classmethod
    def open(cls, directory: str | Path) -> "K4ReverseMapperBundle":
        candidate = Path(directory).resolve()
        root = candidate if candidate.is_dir() else candidate.parent
        manifest_path = root / "manifest.json"
        checksum_path = root / "manifest.sha256.json"
        if not manifest_path.is_file() or not checksum_path.is_file():
            raise K4RuntimeValidationError("k4 manifest/checksum is missing")
        manifest_hash = sha256_file(manifest_path)
        checksum = read_json(checksum_path)
        if checksum != {"path": str(manifest_path), "sha256": manifest_hash}:
            raise K4RuntimeValidationError("k4 manifest checksum marker changed")
        manifest = read_json(manifest_path)
        fixed = {
            "schema_version": 1,
            "status": "complete",
            "direction": DIRECTION,
            "curve_variant": "figure4_large_to_small_retuned_k4",
            "method": "closed-form-centered-ridge-kv-cache-transfer",
            "dtype": "float32",
            "allow_tf32": False,
            "weight_count": K4_WEIGHT_COUNT,
            "bias_count": K4_BIAS_COUNT,
            "ridge_lambda": 0.01,
        }
        for field, expected in fixed.items():
            if manifest.get(field) != expected:
                raise K4RuntimeValidationError(f"k4 manifest {field} is invalid")
        if manifest.get("paper_reference") != PAPER_CURVE:
            raise K4RuntimeValidationError("k4 paper/curve reference is invalid")
        if manifest.get("semantic_role_binding") != {
            "source": {"model_name": "qwen3_32b", "capture_native_role": "target"},
            "target": {"model_name": "qwen3_14b", "capture_native_role": "source"},
        }:
            raise K4RuntimeValidationError("k4 semantic role binding is invalid")
        independence = manifest.get("independent_refit_evidence")
        expected_independence = {
            "weights_refit_from_frozen_train_activations": True,
            "k20_mapper_weights_consumed": False,
            "k20_weights_sliced_or_truncated": False,
            "source_design_width": K4_SOURCE_FEATURES,
            "target_width": K4_TARGET_FEATURES,
            "equation": "4096_to_1024_centered_ridge_lambda_0.01",
            "selection_source": "full_40x64_single_source_score_tensor",
        }
        if independence != expected_independence:
            raise K4RuntimeValidationError("k4 mapper is not proven to be an independent refit")
        payload = manifest.get("payload_validation")
        payload_expected = {
            "status": "passed",
            "layers_actually_loaded": 40,
            "tensors_actually_loaded": 160,
            "expected_tensor_shapes": {
                "key_weight": [K4_SOURCE_FEATURES, K4_TARGET_FEATURES],
                "key_bias": [K4_TARGET_FEATURES],
                "value_weight": [K4_SOURCE_FEATURES, K4_TARGET_FEATURES],
                "value_bias": [K4_TARGET_FEATURES],
            },
        }
        if not isinstance(payload, Mapping):
            raise K4RuntimeValidationError("k4 manifest lacks payload validation")
        for field, expected in payload_expected.items():
            if payload.get(field) != expected:
                raise K4RuntimeValidationError(f"k4 payload validation {field} is invalid")
        payload_digest = str(payload.get("entries_digest", ""))
        if len(payload_digest) != 64 or any(c not in "0123456789abcdef" for c in payload_digest):
            raise K4RuntimeValidationError("k4 payload validation digest is invalid")

        try:
            inventory, source_identity, target_identity = load_verified_train_inventory(
                root, manifest
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise K4RuntimeValidationError(f"k4 training provenance is invalid: {exc}") from exc
        architecture_raw = manifest.get("architecture")
        if not isinstance(architecture_raw, Mapping):
            raise K4RuntimeValidationError("k4 manifest lacks architecture")
        architecture = ArchitectureSpec(
            source_layers=int(architecture_raw.get("source_layers", -1)),
            target_layers=int(architecture_raw.get("target_layers", -1)),
            source_kv_heads=int(architecture_raw.get("kv_heads", -1)),
            target_kv_heads=int(architecture_raw.get("kv_heads", -1)),
            source_head_dim=int(architecture_raw.get("head_dim", -1)),
            target_head_dim=int(architecture_raw.get("head_dim", -1)),
            selected_source_layers=int(architecture_raw.get("selected_source_layers", -1)),
        )
        if architecture != K4_REVERSE_ARCHITECTURE:
            raise K4RuntimeValidationError("k4 architecture changed")
        if architecture.source_features != K4_SOURCE_FEATURES:
            raise K4RuntimeValidationError("k4 source width is not 4096")
        if architecture.target_features != K4_TARGET_FEATURES:
            raise K4RuntimeValidationError("k4 target width is not 1024")

        fit = manifest.get("fit_definition")
        fit_expected = {
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
        if not isinstance(fit, Mapping):
            raise K4RuntimeValidationError("k4 manifest lacks fit definition")
        for field, expected in fit_expected.items():
            if fit.get(field) != expected:
                raise K4RuntimeValidationError(f"k4 fit invariant {field} changed")
        fingerprint = str(manifest.get("fit_fingerprint", ""))
        if canonical_sha256(fit) != fingerprint:
            raise K4RuntimeValidationError("k4 fit fingerprint is invalid")
        if fit.get("activation_inventory_digest") != inventory.get("inventory_digest"):
            raise K4RuntimeValidationError("k4 fit is not bound to its activation inventory")

        selection = manifest.get("selection")
        if not isinstance(selection, Mapping):
            raise K4RuntimeValidationError("k4 manifest lacks selection provenance")
        selection_snapshot = (root / "inputs/selected_layers.json").resolve()
        if Path(str(selection.get("path", ""))).resolve() != selection_snapshot:
            raise K4RuntimeValidationError("k4 selection is not the local immutable snapshot")
        snapshot = validate_k4_selection_artifact(selection_snapshot)
        selection_sha = sha256_file(selection_snapshot)
        if (
            selection.get("source_file_sha256_used_by_fit") != selection_sha
            or selection.get("snapshot_sha256") != selection_sha
            or fit.get("selection_sha256") != selection_sha
        ):
            raise K4RuntimeValidationError("k4 selection snapshot/hash binding changed")
        rows = selection.get("selected_source_layers_rank_order")
        if rows != snapshot.get("selected_source_layers_rank_order"):
            raise K4RuntimeValidationError("k4 manifest rows differ from its selection snapshot")
        if not isinstance(rows, list) or len(rows) != 40:
            raise K4RuntimeValidationError("k4 selection does not cover 40 target layers")

        raw_layers = manifest.get("layers")
        if not isinstance(raw_layers, list) or len(raw_layers) != 40:
            raise K4RuntimeValidationError("k4 manifest does not contain 40 mapper layers")
        layer_slots: list[K4LayerArtifact | None] = [None] * 40
        expected_shapes = {
            "key_weight": [K4_SOURCE_FEATURES, K4_TARGET_FEATURES],
            "key_bias": [K4_TARGET_FEATURES],
            "value_weight": [K4_SOURCE_FEATURES, K4_TARGET_FEATURES],
            "value_bias": [K4_TARGET_FEATURES],
        }
        per_layer_weights = 2 * K4_SOURCE_FEATURES * K4_TARGET_FEATURES
        per_layer_biases = 2 * K4_TARGET_FEATURES
        for entry in raw_layers:
            if not isinstance(entry, Mapping):
                raise K4RuntimeValidationError("malformed k4 layer entry")
            layer = int(entry.get("target_layer", -1))
            if not 0 <= layer < 40 or layer_slots[layer] is not None:
                raise K4RuntimeValidationError(f"invalid/duplicate k4 target layer {layer}")
            selected = tuple(int(value) for value in entry.get("source_layers_rank_order", ()))
            if selected != tuple(rows[layer]) or len(selected) != K4_TOP_K:
                raise K4RuntimeValidationError(f"invalid k4 selection row at target {layer}")
            if len(set(selected)) != K4_TOP_K or any(value < 0 or value >= 64 for value in selected):
                raise K4RuntimeValidationError(f"out-of-range/duplicate k4 row at target {layer}")
            if entry.get("weight_numel") != per_layer_weights or entry.get("bias_numel") != per_layer_biases:
                raise K4RuntimeValidationError(f"k4 parameter count changed at target {layer}")
            stem = f"layer_{layer:03d}"
            layer_path = _local_artifact(root, entry.get("path"), f"{stem}.safetensors")
            sidecar_path = _local_artifact(root, entry.get("sidecar_path"), f"{stem}.json")
            layer_sha = sha256_file(layer_path)
            sidecar_sha = sha256_file(sidecar_path)
            if layer_sha != entry.get("sha256") or sidecar_sha != entry.get("sidecar_sha256"):
                raise K4RuntimeValidationError(f"k4 artifact SHA changed at target {layer}")
            sidecar = read_json(sidecar_path)
            sidecar_expected = {
                "target_layer": layer,
                "source_layers_rank_order": list(selected),
                "sha256": layer_sha,
                "dtype": "float32",
                "source_heads": 8,
                "head_dim": 128,
                "tensor_shapes": expected_shapes,
                "weight_numel": per_layer_weights,
                "bias_numel": per_layer_biases,
            }
            for field, expected in sidecar_expected.items():
                if sidecar.get(field) != expected:
                    raise K4RuntimeValidationError(
                        f"k4 layer {layer} sidecar {field} changed"
                    )
            metadata = sidecar.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("fit_fingerprint") != fingerprint:
                raise K4RuntimeValidationError(f"k4 layer {layer} fit fingerprint changed")
            if metadata.get("selection_manifest_sha256") != selection_sha:
                raise K4RuntimeValidationError(f"k4 layer {layer} selection binding changed")
            layer_slots[layer] = K4LayerArtifact(
                target_layer=layer,
                path=layer_path,
                sha256=layer_sha,
                sidecar_path=sidecar_path,
                sidecar_sha256=sidecar_sha,
                source_layers_rank_order=selected,
            )
        if any(value is None for value in layer_slots):
            raise K4RuntimeValidationError("k4 target-layer coverage is incomplete")
        if sum(int(row["weight_numel"]) for row in raw_layers) != K4_WEIGHT_COUNT:
            raise K4RuntimeValidationError("k4 aggregate weight count changed")
        if sum(int(row["bias_numel"]) for row in raw_layers) != K4_BIAS_COUNT:
            raise K4RuntimeValidationError("k4 aggregate bias count changed")
        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_hash,
            fit_fingerprint=fingerprint,
            architecture=architecture,
            layers=tuple(value for value in layer_slots if value is not None),
            source_model_identity=source_identity,
            target_model_identity=target_identity,
        )

    def load_layer(
        self, target_layer: int, *, device: torch.device | str = "cpu"
    ) -> LayerMapper:
        if not 0 <= target_layer < 40:
            raise K4RuntimeValidationError("k4 target layer is out of range")
        artifact = self.layers[target_layer]
        if sha256_file(artifact.path) != artifact.sha256:
            raise K4RuntimeValidationError("k4 mapper changed after bundle validation")
        if sha256_file(artifact.sidecar_path) != artifact.sidecar_sha256:
            raise K4RuntimeValidationError("k4 mapper sidecar changed after validation")
        mapper = load_layer_mapper(artifact.path, device=device)
        if mapper.source_layers != artifact.source_layers_rank_order:
            raise K4RuntimeValidationError("loaded k4 source-layer order changed")
        mapper.validate(source_heads=8, head_dim=128)
        if mapper.key_weight.shape != (K4_SOURCE_FEATURES, K4_TARGET_FEATURES):
            raise K4RuntimeValidationError("loaded k4 key weight shape changed")
        if mapper.value_weight.shape != (K4_SOURCE_FEATURES, K4_TARGET_FEATURES):
            raise K4RuntimeValidationError("loaded k4 value weight shape changed")
        if {
            mapper.key_weight.dtype,
            mapper.key_bias.dtype,
            mapper.value_weight.dtype,
            mapper.value_bias.dtype,
        } != {torch.float32}:
            raise K4RuntimeValidationError("k4 mapper tensors must be FP32")
        return mapper

    def iter_layers(
        self, layer_devices: Sequence[torch.device | str]
    ) -> Iterator[tuple[int, LayerMapper]]:
        if len(layer_devices) != 40:
            raise K4RuntimeValidationError("k4 target layer-device count must be 40")
        for layer, device in enumerate(layer_devices):
            yield layer, self.load_layer(layer, device=device)


def _input_device(model: torch.nn.Module) -> torch.device:
    try:
        device = next(model.get_input_embeddings().parameters()).device
    except StopIteration as exc:
        raise K4RuntimeValidationError("model embeddings have no parameters") from exc
    if device.type == "meta":
        raise K4RuntimeValidationError("model embeddings remain on meta device")
    return device


class K4ReverseMappedPrefixCacheBuilder:
    """Prefill 32B and map its prefix with independently fitted top-4 weights."""

    def __init__(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        mapper_manifest: str | Path,
        token_chunk: int = 1024,
    ) -> None:
        if int(token_chunk) <= 0:
            raise ValueError("k4 mapper token chunk must be positive")
        self.source_model = source_model
        self.target_model = target_model
        self.token_chunk = int(token_chunk)
        source_arch = (
            int(source_model.config.num_hidden_layers),
            int(source_model.config.num_key_value_heads),
            int(source_model.config.head_dim),
        )
        target_arch = (
            int(target_model.config.num_hidden_layers),
            int(target_model.config.num_key_value_heads),
            int(target_model.config.head_dim),
        )
        if source_arch != (64, 8, 128) or target_arch != (40, 8, 128):
            raise K4RuntimeValidationError(
                f"k4 endpoint architecture changed: {source_arch} -> {target_arch}"
            )
        bundle = K4ReverseMapperBundle.open(mapper_manifest)
        try:
            self.source_model_artifact = assert_model_instance_identity(
                source_model, bundle.source_model_identity, semantic_role="source"
            )
            self.target_model_artifact = assert_model_instance_identity(
                target_model, bundle.target_model_identity, semantic_role="target"
            )
        except (OSError, ValueError) as exc:
            raise K4RuntimeValidationError(str(exc)) from exc
        devices = qwen3_layer_devices(target_model)
        resident, materialization = materialize_mapper_on_target(bundle, devices)
        self.mapper_bundle: ResidentMapperBundle = resident
        self.materialization = materialization
        self.manifest_path = bundle.manifest_path
        self.target_layer_devices = tuple(str(value) for value in devices)
        self.source_capture_model_identity = dict(bundle.source_model_identity)
        self.target_capture_model_identity = dict(bundle.target_model_identity)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "direction": DIRECTION,
            "curve_variant": "figure4_large_to_small_retuned_k4",
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.mapper_bundle.manifest_sha256,
            "fit_fingerprint": self.mapper_bundle.fit_fingerprint,
            "layers": len(self.mapper_bundle.layers),
            "selected_source_layers": K4_TOP_K,
            "source_features": K4_SOURCE_FEATURES,
            "target_features": K4_TARGET_FEATURES,
            "independently_refit_not_k20_slice": True,
            "token_chunk": self.token_chunk,
            "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
            "target_layer_devices": list(self.target_layer_devices),
            "resident_parameter_bytes": self.mapper_bundle.resident_parameter_bytes,
            "materialization": self.materialization,
            "source_model_artifact": self.source_model_artifact,
            "target_model_artifact": self.target_model_artifact,
            "source_capture_model_identity": self.source_capture_model_identity,
            "target_capture_model_identity": self.target_capture_model_identity,
        }

    @torch.inference_mode()
    def __call__(self, prefix_token_ids: Sequence[int]) -> Any:
        prefix = [int(token) for token in prefix_token_ids]
        if not prefix:
            raise ValueError("k4 mapped prefix must contain at least one token")
        source_base = getattr(self.source_model, "model", self.source_model)
        device = _input_device(self.source_model)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(prefix), device=device).unsqueeze(0)
        output = source_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.arange(len(prefix), device=device),
            use_cache=True,
            return_dict=True,
        )
        cache = transfer_source_cache(
            output.past_key_values,
            source_config=self.source_model.config,
            target_model=self.target_model,
            mapper_bundle=self.mapper_bundle,
            position_ids=position_ids,
            output_dtype=None,
            token_chunk=self.token_chunk,
        )
        if cache_sequence_length(cache) != len(prefix):
            raise K4RuntimeValidationError("k4 mapped cache sequence length changed")
        return cache


__all__ = [
    "K4_REVERSE_ARCHITECTURE",
    "K4ReverseMappedPrefixCacheBuilder",
    "K4ReverseMapperBundle",
    "K4RuntimeValidationError",
]
