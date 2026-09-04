"""Strict runtime for the formal top-20 Qwen3-32B -> Qwen3-14B mapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .protocol import (
    BIAS_COUNT,
    canonical_sha256,
    DIRECTION,
    SOURCE_FEATURES,
    TARGET_FEATURES,
    TOP_K,
    WEIGHT_COUNT,
    read_json,
    sha256_file,
)

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

from .provenance import (  # noqa: E402
    assert_model_instance_identity,
    load_verified_train_inventory,
)


REVERSE_ARCHITECTURE = ArchitectureSpec(
    source_layers=64,
    target_layers=40,
    source_kv_heads=8,
    target_kv_heads=8,
    source_head_dim=128,
    target_head_dim=128,
    selected_source_layers=TOP_K,
)


class ReverseRuntimeValidationError(ValueError):
    """A reverse mapper artifact or model violates the frozen protocol."""


def _local_artifact(root: Path, value: Any, name: str) -> Path:
    recorded = Path(str(value))
    if recorded.name != name:
        raise ReverseRuntimeValidationError(
            f"mapper artifact name {recorded.name!r}, expected {name!r}"
        )
    local = root / "layers" / name
    if not local.is_file():
        raise ReverseRuntimeValidationError(
            f"reverse bundle is not self-contained; missing {local}"
        )
    return local.resolve()


@dataclass(frozen=True)
class ReverseLayerArtifact:
    target_layer: int
    path: Path
    sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    source_layers_rank_order: tuple[int, ...]


@dataclass(frozen=True)
class ReverseMapperBundle:
    """Verified, self-contained, lazy-loading reverse mapper bundle."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    fit_fingerprint: str
    architecture: ArchitectureSpec
    layers: tuple[ReverseLayerArtifact, ...]
    source_model_identity: Mapping[str, str]
    target_model_identity: Mapping[str, str]
    dtype: torch.dtype = torch.float32

    @classmethod
    def open(cls, directory: str | Path) -> "ReverseMapperBundle":
        root = Path(directory).resolve()
        manifest_path = root / "manifest.json"
        checksum_path = root / "manifest.sha256.json"
        manifest_hash = sha256_file(manifest_path)
        checksum = read_json(checksum_path)
        if checksum.get("sha256") != manifest_hash:
            raise ReverseRuntimeValidationError("mapper manifest SHA256 mismatch")
        manifest = read_json(manifest_path)
        fixed = {
            "schema_version": 1,
            "status": "complete",
            "direction": DIRECTION,
            "method": "closed-form-centered-ridge-kv-cache-transfer",
            "dtype": "float32",
            "allow_tf32": False,
            "weight_count": WEIGHT_COUNT,
            "bias_count": BIAS_COUNT,
        }
        for field, expected in fixed.items():
            if manifest.get(field) != expected:
                raise ReverseRuntimeValidationError(
                    f"manifest {field}={manifest.get(field)!r}, expected {expected!r}"
                )
        if manifest.get("paper_reference") != {
            "paper": "arXiv:2608.03893",
            "sections": ["4.6", "4.7"],
            "table": 16,
            "selected_source_layers": TOP_K,
        }:
            raise ReverseRuntimeValidationError("paper protocol reference is missing/invalid")
        if manifest.get("semantic_role_binding") != {
            "source": {
                "model_name": "qwen3_32b",
                "capture_native_role": "target",
            },
            "target": {
                "model_name": "qwen3_14b",
                "capture_native_role": "source",
            },
        }:
            raise ReverseRuntimeValidationError("semantic/capture-native role binding is invalid")
        payload_validation = manifest.get("payload_validation")
        expected_payload_validation = {
            "status": "passed",
            "layers_actually_loaded": 40,
            "tensors_actually_loaded": 160,
            "expected_tensor_shapes": {
                "key_weight": [SOURCE_FEATURES, TARGET_FEATURES],
                "key_bias": [TARGET_FEATURES],
                "value_weight": [SOURCE_FEATURES, TARGET_FEATURES],
                "value_bias": [TARGET_FEATURES],
            },
        }
        if not isinstance(payload_validation, Mapping):
            raise ReverseRuntimeValidationError("manifest lacks complete payload validation")
        for field, expected in expected_payload_validation.items():
            if payload_validation.get(field) != expected:
                raise ReverseRuntimeValidationError(
                    f"manifest payload validation {field} is invalid"
                )
        payload_digest = str(payload_validation.get("entries_digest", ""))
        if len(payload_digest) != 64 or any(
            character not in "0123456789abcdef" for character in payload_digest
        ):
            raise ReverseRuntimeValidationError("manifest payload-validation digest is invalid")
        try:
            train_inventory, source_model_identity, target_model_identity = (
                load_verified_train_inventory(root, manifest)
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ReverseRuntimeValidationError(
                f"mapper training provenance is invalid: {exc}"
            ) from exc
        architecture_raw = manifest.get("architecture")
        if not isinstance(architecture_raw, Mapping):
            raise ReverseRuntimeValidationError("manifest lacks architecture")
        architecture = ArchitectureSpec(
            source_layers=int(architecture_raw.get("source_layers", -1)),
            target_layers=int(architecture_raw.get("target_layers", -1)),
            source_kv_heads=int(architecture_raw.get("kv_heads", -1)),
            target_kv_heads=int(architecture_raw.get("kv_heads", -1)),
            source_head_dim=int(architecture_raw.get("head_dim", -1)),
            target_head_dim=int(architecture_raw.get("head_dim", -1)),
            selected_source_layers=int(
                architecture_raw.get("selected_source_layers", -1)
            ),
        )
        if architecture != REVERSE_ARCHITECTURE:
            raise ReverseRuntimeValidationError(
                f"mapper architecture {architecture!r} != {REVERSE_ARCHITECTURE!r}"
            )
        if architecture.source_features != SOURCE_FEATURES:
            raise ReverseRuntimeValidationError("source feature count is not 20480")
        if architecture.target_features != TARGET_FEATURES:
            raise ReverseRuntimeValidationError("target feature count is not 1024")
        if float(manifest.get("ridge_lambda", -1)) != 0.01:
            raise ReverseRuntimeValidationError("ridge lambda is not 0.01")
        fit_definition = manifest.get("fit_definition")
        if not isinstance(fit_definition, Mapping):
            raise ReverseRuntimeValidationError("manifest lacks fit definition")
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
        for field, expected in fit_expected.items():
            if fit_definition.get(field) != expected:
                raise ReverseRuntimeValidationError(f"fit invariant {field} is invalid")
        fingerprint = str(manifest.get("fit_fingerprint", ""))
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
            or canonical_sha256(fit_definition) != fingerprint
        ):
            raise ReverseRuntimeValidationError("fit fingerprint is invalid")
        if (
            fit_definition.get("activation_inventory_digest")
            != train_inventory.get("inventory_digest")
        ):
            raise ReverseRuntimeValidationError(
                "fit definition is not bound to the mapper activation inventory"
            )

        selection = manifest.get("selection")
        rows = selection.get("selected_source_layers_rank_order") if isinstance(selection, Mapping) else None
        if not isinstance(rows, list) or len(rows) != 40:
            raise ReverseRuntimeValidationError("manifest must contain 40 selection rows")
        selection_snapshot = (root / "inputs/selected_layers.json").resolve()
        if not isinstance(selection, Mapping):
            raise ReverseRuntimeValidationError("manifest selection provenance is malformed")
        recorded_selection_path = Path(str(selection.get("path", ""))).resolve()
        if recorded_selection_path != selection_snapshot or not selection_snapshot.is_file():
            raise ReverseRuntimeValidationError(
                "mapper selection is not the self-contained input snapshot"
            )
        selection_sha = sha256_file(selection_snapshot)
        if (
            selection.get("source_file_sha256_used_by_fit") != selection_sha
            or selection.get("snapshot_sha256") != selection_sha
            or fit_definition.get("selection_sha256") != selection_sha
        ):
            raise ReverseRuntimeValidationError(
                "mapper selection snapshot/hash/fit-definition binding is invalid"
            )
        selection_snapshot_raw = read_json(selection_snapshot)
        selection_exact = {
            "direction": DIRECTION,
            "formal_protocol": True,
            "method": "same-head-single-source-ols-affine-uniform-channel-r2",
            "source_layers": 64,
            "target_layers": 40,
            "kv_heads": 8,
            "head_dim": 128,
            "k": TOP_K,
            "n_samples": 128_000,
        }
        for field, expected in selection_exact.items():
            if selection_snapshot_raw.get(field) != expected:
                raise ReverseRuntimeValidationError(
                    f"mapper selection snapshot {field} is invalid"
                )
        if selection_snapshot_raw.get("selected_source_layers_rank_order") != rows:
            raise ReverseRuntimeValidationError(
                "manifest rows differ from the fitted selection snapshot"
            )
        raw_layers = manifest.get("layers")
        if not isinstance(raw_layers, list) or len(raw_layers) != 40:
            raise ReverseRuntimeValidationError("manifest must contain 40 mapper layers")
        layer_slots: list[ReverseLayerArtifact | None] = [None] * 40
        per_layer_weights = 2 * SOURCE_FEATURES * TARGET_FEATURES
        per_layer_biases = 2 * TARGET_FEATURES
        expected_shapes = {
            "key_weight": [SOURCE_FEATURES, TARGET_FEATURES],
            "key_bias": [TARGET_FEATURES],
            "value_weight": [SOURCE_FEATURES, TARGET_FEATURES],
            "value_bias": [TARGET_FEATURES],
        }
        for entry in raw_layers:
            if not isinstance(entry, Mapping):
                raise ReverseRuntimeValidationError("malformed mapper layer entry")
            layer = int(entry.get("target_layer", -1))
            if not 0 <= layer < 40 or layer_slots[layer] is not None:
                raise ReverseRuntimeValidationError(f"invalid/duplicate target layer {layer}")
            selected = tuple(int(value) for value in entry.get("source_layers_rank_order", ()))
            row = tuple(int(value) for value in rows[layer])
            if selected != row or len(selected) != TOP_K or len(set(selected)) != TOP_K:
                raise ReverseRuntimeValidationError(f"invalid top-{TOP_K} row at layer {layer}")
            if any(value < 0 or value >= 64 for value in selected):
                raise ReverseRuntimeValidationError(f"out-of-range source layer at target {layer}")
            if int(entry.get("weight_numel", -1)) != per_layer_weights:
                raise ReverseRuntimeValidationError(f"weight count mismatch at target {layer}")
            if int(entry.get("bias_numel", -1)) != per_layer_biases:
                raise ReverseRuntimeValidationError(f"bias count mismatch at target {layer}")
            stem = f"layer_{layer:03d}"
            payload = _local_artifact(root, entry.get("path"), f"{stem}.safetensors")
            sidecar_path = _local_artifact(root, entry.get("sidecar_path"), f"{stem}.json")
            payload_hash = sha256_file(payload)
            sidecar_hash = sha256_file(sidecar_path)
            if payload_hash != entry.get("sha256") or sidecar_hash != entry.get("sidecar_sha256"):
                raise ReverseRuntimeValidationError(f"artifact SHA mismatch at target {layer}")
            sidecar = read_json(sidecar_path)
            sidecar_expected = {
                "target_layer": layer,
                "source_layers_rank_order": list(selected),
                "sha256": payload_hash,
                "dtype": "float32",
                "source_heads": 8,
                "head_dim": 128,
                "tensor_shapes": expected_shapes,
                "weight_numel": per_layer_weights,
                "bias_numel": per_layer_biases,
            }
            for field, expected in sidecar_expected.items():
                if sidecar.get(field) != expected:
                    raise ReverseRuntimeValidationError(
                        f"sidecar {field} mismatch at target {layer}"
                    )
            metadata = sidecar.get("metadata")
            if not isinstance(metadata, Mapping) or metadata.get("fit_fingerprint") != fingerprint:
                raise ReverseRuntimeValidationError(f"fit fingerprint mismatch at target {layer}")
            layer_slots[layer] = ReverseLayerArtifact(
                target_layer=layer,
                path=payload,
                sha256=payload_hash,
                sidecar_path=sidecar_path,
                sidecar_sha256=sidecar_hash,
                source_layers_rank_order=selected,
            )
        if any(value is None for value in layer_slots):
            raise ReverseRuntimeValidationError("mapper target-layer coverage is incomplete")
        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_hash,
            fit_fingerprint=fingerprint,
            architecture=architecture,
            layers=tuple(value for value in layer_slots if value is not None),
            source_model_identity=source_model_identity,
            target_model_identity=target_model_identity,
        )

    def load_layer(self, target_layer: int, *, device: torch.device | str = "cpu") -> LayerMapper:
        if not 0 <= target_layer < len(self.layers):
            raise ReverseRuntimeValidationError("target layer is out of range")
        artifact = self.layers[target_layer]
        if sha256_file(artifact.path) != artifact.sha256:
            raise ReverseRuntimeValidationError("mapper changed after bundle validation")
        if sha256_file(artifact.sidecar_path) != artifact.sidecar_sha256:
            raise ReverseRuntimeValidationError("mapper sidecar changed after validation")
        mapper = load_layer_mapper(artifact.path, device=device)
        if mapper.source_layers != artifact.source_layers_rank_order:
            raise ReverseRuntimeValidationError("loaded source-layer order changed")
        mapper.validate(source_heads=8, head_dim=128)
        if {
            mapper.key_weight.dtype,
            mapper.key_bias.dtype,
            mapper.value_weight.dtype,
            mapper.value_bias.dtype,
        } != {torch.float32}:
            raise ReverseRuntimeValidationError("mapper tensors must be FP32")
        return mapper

    def iter_layers(
        self, layer_devices: Sequence[torch.device | str]
    ) -> Iterator[tuple[int, LayerMapper]]:
        if len(layer_devices) != 40:
            raise ReverseRuntimeValidationError("target layer-device count must be 40")
        for layer, device in enumerate(layer_devices):
            yield layer, self.load_layer(layer, device=device)


def _input_device(model: torch.nn.Module) -> torch.device:
    parameters = model.get_input_embeddings().parameters()
    try:
        device = next(parameters).device
    except StopIteration as exc:
        raise ReverseRuntimeValidationError("model embeddings have no parameters") from exc
    if device.type == "meta":
        raise ReverseRuntimeValidationError("model embeddings remain on meta device")
    return device


class ReverseMappedPrefixCacheBuilder:
    """Prefill 32B on a prefix and map it into a resident 14B cache."""

    def __init__(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        mapper_manifest: str | Path,
        token_chunk: int = 1024,
    ) -> None:
        self.source_model = source_model
        self.target_model = target_model
        self.token_chunk = int(token_chunk)
        if self.token_chunk <= 0:
            raise ValueError("token_chunk must be positive")
        observed_source = (
            int(source_model.config.num_hidden_layers),
            int(source_model.config.num_key_value_heads),
            int(source_model.config.head_dim),
        )
        observed_target = (
            int(target_model.config.num_hidden_layers),
            int(target_model.config.num_key_value_heads),
            int(target_model.config.head_dim),
        )
        if observed_source != (64, 8, 128):
            raise ReverseRuntimeValidationError(f"source architecture is {observed_source}")
        if observed_target != (40, 8, 128):
            raise ReverseRuntimeValidationError(f"target architecture is {observed_target}")
        manifest_path = Path(mapper_manifest).resolve()
        bundle = ReverseMapperBundle.open(
            manifest_path if manifest_path.is_dir() else manifest_path.parent
        )
        try:
            self.source_model_artifact = assert_model_instance_identity(
                source_model, bundle.source_model_identity, semantic_role="source"
            )
            self.target_model_artifact = assert_model_instance_identity(
                target_model, bundle.target_model_identity, semantic_role="target"
            )
        except (OSError, ValueError) as exc:
            raise ReverseRuntimeValidationError(str(exc)) from exc
        devices = qwen3_layer_devices(target_model)
        resident, materialization = materialize_mapper_on_target(bundle, devices)
        self.mapper_bundle: ResidentMapperBundle = resident
        self.source_capture_model_identity = dict(bundle.source_model_identity)
        self.target_capture_model_identity = dict(bundle.target_model_identity)
        self.materialization = materialization
        self.manifest_path = bundle.manifest_path
        self.target_layer_devices = tuple(str(value) for value in devices)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "direction": DIRECTION,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.mapper_bundle.manifest_sha256,
            "fit_fingerprint": self.mapper_bundle.fit_fingerprint,
            "layers": len(self.mapper_bundle.layers),
            "selected_source_layers": TOP_K,
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
            raise ValueError("mapped prefix must contain at least one token")
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
            raise ReverseRuntimeValidationError("mapped cache sequence length changed")
        return cache


__all__ = [
    "REVERSE_ARCHITECTURE",
    "ReverseMapperBundle",
    "ReverseMappedPrefixCacheBuilder",
    "ReverseRuntimeValidationError",
]
