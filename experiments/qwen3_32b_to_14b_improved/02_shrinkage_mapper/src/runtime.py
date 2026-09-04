"""Fail-closed runtime for the independently published shrinkage mapper."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path("/home/wby/llm-cache/nvidia-kvcache").resolve()
SRC_ROOT = REPOSITORY_ROOT / "src"
for value in (REPOSITORY_ROOT, SRC_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from kvcache_transfer.artifacts import load_layer_mapper  # noqa: E402
from kvcache_transfer.cache_adapter import cache_sequence_length, qwen3_layer_devices  # noqa: E402
from kvcache_transfer.config import ArchitectureSpec  # noqa: E402
from kvcache_transfer.latency import ResidentMapperBundle, materialize_mapper_on_target  # noqa: E402
from kvcache_transfer.mapper import LayerMapper  # noqa: E402
from kvcache_transfer.transfer import transfer_source_cache  # noqa: E402
from experiments.qwen3_32b_to_14b.reverse_kv.provenance import (  # noqa: E402
    assert_model_instance_identity,
)
from experiments.qwen3_32b_to_14b.reverse_kv.runtime import (  # noqa: E402
    ReverseMapperBundle as UpstreamK20Bundle,
)


DIRECTION = "qwen3_32b_to_14b"
METHOD = "headwise-convex-shrinkage-of-independent-k4-and-k20-affine-mappers"
DEFAULT_MAPPER_MANIFEST = (
    EXPERIMENT_ROOT / "artifacts" / "headwise_shrinkage_v1" / "manifest.json"
)
COMPLETION_RECEIPT = (
    EXPERIMENT_ROOT
    / "runs"
    / "activation_cv_v3_completion_amendment_v1"
    / "receipt_amended.json"
)
ARCHITECTURE = ArchitectureSpec(
    source_layers=64,
    target_layers=40,
    source_kv_heads=8,
    target_kv_heads=8,
    source_head_dim=128,
    target_head_dim=128,
    selected_source_layers=20,
)


class ShrinkageRuntimeValidationError(ValueError):
    """A candidate artifact or model violates the frozen protocol."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ShrinkageRuntimeValidationError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_completion_receipt(mapper_manifest_sha256: str) -> dict[str, Any]:
    receipt = _read_json(COMPLETION_RECEIPT)
    fixed = {
        "status": "complete_by_append_only_amendment",
        "amendment_scope": "receipt-serialization-only",
    }
    for field, expected in fixed.items():
        if receipt.get(field) != expected:
            raise ShrinkageRuntimeValidationError(f"completion receipt {field} changed")
    execution = receipt.get("execution")
    if not isinstance(execution, Mapping) or any(
        execution.get(field) is not False
        for field in (
            "cuda_invoked",
            "activation_recomputed",
            "locked_audit_recomputed",
            "mapper_modified",
            "alpha_modified",
        )
    ):
        raise ShrinkageRuntimeValidationError("completion amendment scope changed")
    mapper = receipt.get("mapper")
    if not isinstance(mapper, Mapping) or mapper.get("manifest_sha256") != mapper_manifest_sha256:
        raise ShrinkageRuntimeValidationError("receipt/mapper manifest identity mismatch")
    amendment_root = COMPLETION_RECEIPT.parent
    marker = _read_json(amendment_root / "manifest.sha256.json")
    if marker != {
        "path": "manifest.json",
        "sha256": _sha256(amendment_root / "manifest.json"),
    }:
        raise ShrinkageRuntimeValidationError("completion amendment marker changed")
    return {
        "path": str(COMPLETION_RECEIPT),
        "sha256": _sha256(COMPLETION_RECEIPT),
        "status": receipt["status"],
        "locked_audit_sha256": receipt["locked_audit"]["sha256"],
    }


@dataclass(frozen=True)
class ShrinkageLayerArtifact:
    target_layer: int
    path: Path
    sha256: str
    sidecar_path: Path
    sidecar_sha256: str
    source_layers_rank_order: tuple[int, ...]


@dataclass(frozen=True)
class ShrinkageMapperBundle:
    """Verified, self-contained, lazy-loading candidate mapper bundle."""

    root: Path
    manifest_path: Path
    manifest_sha256: str
    fit_fingerprint: str
    architecture: ArchitectureSpec
    layers: tuple[ShrinkageLayerArtifact, ...]
    source_model_identity: Mapping[str, str]
    target_model_identity: Mapping[str, str]
    completion_receipt: Mapping[str, Any]
    dtype: torch.dtype = torch.float32

    @classmethod
    def open(
        cls, directory: str | Path = DEFAULT_MAPPER_MANIFEST, *, verify_payloads: bool = True
    ) -> "ShrinkageMapperBundle":
        value = Path(directory).resolve()
        root = value if value.is_dir() else value.parent
        manifest_path = root / "manifest.json"
        manifest_sha = _sha256(manifest_path)
        marker = _read_json(root / "manifest.sha256.json")
        if marker != {"path": "manifest.json", "sha256": manifest_sha}:
            raise ShrinkageRuntimeValidationError("mapper manifest marker mismatch")
        manifest = _read_json(manifest_path)
        fixed = {
            "schema_version": 1,
            "status": "complete",
            "direction": DIRECTION,
            "method": METHOD,
            "dtype": "float32",
            "allow_tf32": False,
            "weight_count": 1_677_721_600,
            "bias_count": 81_920,
        }
        for field, expected in fixed.items():
            if manifest.get(field) != expected:
                raise ShrinkageRuntimeValidationError(f"mapper {field} changed")
        architecture_raw = manifest.get("architecture")
        expected_architecture = {
            "source_layers": 64,
            "target_layers": 40,
            "kv_heads": 8,
            "head_dim": 128,
            "selected_source_layers": 20,
        }
        if architecture_raw != expected_architecture:
            raise ShrinkageRuntimeValidationError("mapper architecture changed")
        fingerprint = str(manifest.get("mapper_fingerprint", ""))
        if len(fingerprint) != 64:
            raise ShrinkageRuntimeValidationError("mapper fingerprint is malformed")
        receipt = _validate_completion_receipt(manifest_sha)
        upstream = manifest.get("upstream")
        if not isinstance(upstream, Mapping):
            raise ShrinkageRuntimeValidationError("upstream identity missing")
        k20_manifest_entry = upstream.get("k20_manifest")
        if not isinstance(k20_manifest_entry, Mapping):
            raise ShrinkageRuntimeValidationError("upstream k20 identity missing")
        k20_manifest = Path(str(k20_manifest_entry["path"])).resolve()
        if _sha256(k20_manifest) != k20_manifest_entry.get("sha256"):
            raise ShrinkageRuntimeValidationError("upstream k20 manifest changed")
        upstream_bundle = UpstreamK20Bundle.open(k20_manifest.parent)
        rows = manifest.get("selection", {}).get("selected_source_layers_rank_order")
        if not isinstance(rows, list) or len(rows) != 40:
            raise ShrinkageRuntimeValidationError("mapper selection changed")
        raw_layers = manifest.get("layers")
        if not isinstance(raw_layers, list) or len(raw_layers) != 40:
            raise ShrinkageRuntimeValidationError("mapper layer inventory changed")
        artifacts: list[ShrinkageLayerArtifact] = []
        for target_layer, entry in enumerate(raw_layers):
            if entry.get("target_layer") != target_layer:
                raise ShrinkageRuntimeValidationError("mapper layers are unordered")
            selected = tuple(int(item) for item in entry.get("source_layers_rank_order", ()))
            if list(selected) != rows[target_layer] or len(selected) != 20:
                raise ShrinkageRuntimeValidationError(f"selection mismatch at {target_layer}")
            payload = (root / str(entry["path"])).resolve()
            sidecar = (root / str(entry["sidecar_path"])).resolve()
            payload.relative_to(root)
            sidecar.relative_to(root)
            sidecar_hash = _sha256(sidecar)
            if sidecar_hash != entry.get("sidecar_sha256"):
                raise ShrinkageRuntimeValidationError(f"sidecar changed at {target_layer}")
            if verify_payloads and _sha256(payload) != entry.get("sha256"):
                raise ShrinkageRuntimeValidationError(f"payload changed at {target_layer}")
            artifacts.append(
                ShrinkageLayerArtifact(
                    target_layer=target_layer,
                    path=payload,
                    sha256=str(entry["sha256"]),
                    sidecar_path=sidecar,
                    sidecar_sha256=sidecar_hash,
                    source_layers_rank_order=selected,
                )
            )
        return cls(
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            fit_fingerprint=fingerprint,
            architecture=ARCHITECTURE,
            layers=tuple(artifacts),
            source_model_identity=upstream_bundle.source_model_identity,
            target_model_identity=upstream_bundle.target_model_identity,
            completion_receipt=receipt,
        )

    def load_layer(
        self, target_layer: int, *, device: torch.device | str = "cpu"
    ) -> LayerMapper:
        if not 0 <= target_layer < 40:
            raise ShrinkageRuntimeValidationError("target layer out of range")
        entry = self.layers[target_layer]
        if _sha256(entry.path) != entry.sha256:
            raise ShrinkageRuntimeValidationError("mapper payload changed after open")
        if _sha256(entry.sidecar_path) != entry.sidecar_sha256:
            raise ShrinkageRuntimeValidationError("mapper sidecar changed after open")
        mapper = load_layer_mapper(entry.path, device=device)
        if mapper.source_layers != entry.source_layers_rank_order:
            raise ShrinkageRuntimeValidationError("loaded source layer order changed")
        mapper.validate(source_heads=8, head_dim=128)
        if {
            mapper.key_weight.dtype,
            mapper.key_bias.dtype,
            mapper.value_weight.dtype,
            mapper.value_bias.dtype,
        } != {torch.float32}:
            raise ShrinkageRuntimeValidationError("mapper tensors must remain FP32")
        return mapper

    def iter_layers(
        self, layer_devices: Sequence[torch.device | str]
    ) -> Iterator[tuple[int, LayerMapper]]:
        if len(layer_devices) != 40:
            raise ShrinkageRuntimeValidationError("target layer-device count must be 40")
        for layer, device in enumerate(layer_devices):
            yield layer, self.load_layer(layer, device=device)


def _input_device(model: torch.nn.Module) -> torch.device:
    try:
        device = next(model.get_input_embeddings().parameters()).device
    except StopIteration as exc:
        raise ShrinkageRuntimeValidationError("model embeddings have no parameters") from exc
    if device.type == "meta":
        raise ShrinkageRuntimeValidationError("model embeddings remain on meta")
    return device


class ShrinkageMappedPrefixCacheBuilder:
    """Prefill frozen 32B, then map the prefix into a frozen 14B cache."""

    def __init__(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        mapper_manifest: str | Path = DEFAULT_MAPPER_MANIFEST,
        token_chunk: int = 1024,
        verify_mapper_payloads: bool = True,
    ) -> None:
        self.source_model = source_model
        self.target_model = target_model
        self.token_chunk = int(token_chunk)
        if self.token_chunk <= 0:
            raise ValueError("token_chunk must be positive")
        source_shape = (
            int(source_model.config.num_hidden_layers),
            int(source_model.config.num_key_value_heads),
            int(source_model.config.head_dim),
        )
        target_shape = (
            int(target_model.config.num_hidden_layers),
            int(target_model.config.num_key_value_heads),
            int(target_model.config.head_dim),
        )
        if source_shape != (64, 8, 128) or target_shape != (40, 8, 128):
            raise ShrinkageRuntimeValidationError(
                f"model architecture mismatch: source={source_shape}, target={target_shape}"
            )
        bundle = ShrinkageMapperBundle.open(
            mapper_manifest, verify_payloads=verify_mapper_payloads
        )
        try:
            self.source_model_artifact = assert_model_instance_identity(
                source_model, bundle.source_model_identity, semantic_role="source"
            )
            self.target_model_artifact = assert_model_instance_identity(
                target_model, bundle.target_model_identity, semantic_role="target"
            )
        except (OSError, ValueError) as exc:
            raise ShrinkageRuntimeValidationError(str(exc)) from exc
        devices = qwen3_layer_devices(target_model)
        resident, materialization = materialize_mapper_on_target(bundle, devices)
        self.mapper_bundle: ResidentMapperBundle = resident
        self.manifest_path = bundle.manifest_path
        self.completion_receipt = dict(bundle.completion_receipt)
        self.target_layer_devices = tuple(str(value) for value in devices)
        self.materialization = materialization

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "direction": DIRECTION,
            "method": METHOD,
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.mapper_bundle.manifest_sha256,
            "mapper_fingerprint": self.mapper_bundle.fit_fingerprint,
            "completion_receipt": self.completion_receipt,
            "layers": len(self.mapper_bundle.layers),
            "selected_source_layers": 20,
            "token_chunk": self.token_chunk,
            "execution": "one_fp32_affine_mapper_per_target_layer_and_component",
            "target_layer_devices": list(self.target_layer_devices),
            "resident_parameter_bytes": self.mapper_bundle.resident_parameter_bytes,
            "materialization": self.materialization,
            "source_model_artifact": self.source_model_artifact,
            "target_model_artifact": self.target_model_artifact,
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
            raise ShrinkageRuntimeValidationError("mapped cache length changed")
        return cache


__all__ = [
    "ARCHITECTURE",
    "COMPLETION_RECEIPT",
    "DEFAULT_MAPPER_MANIFEST",
    "ShrinkageMappedPrefixCacheBuilder",
    "ShrinkageMapperBundle",
    "ShrinkageRuntimeValidationError",
]

