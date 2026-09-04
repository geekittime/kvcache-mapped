"""Held-out activation reconstruction for a completed production mapper.

This module deliberately does not import Transformers or load either language
model.  It consumes the immutable BF16 activation stores produced by
``capture_activations.py`` and the completed per-layer mapper checkpoints.
The primary metric is the paper's arithmetic mean of independently centered
per-channel R2 values.  Predictions are formed in FP32 and the streaming R2
state uses stable FP64 Welford updates via :class:`R2Accumulator`.

The 50-sequence held-out split is a local generalization diagnostic.  It was
not used to select source layers or fit the mapper, and it is not a benchmark
metric reported by the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file
import torch

from .activation_store import read_activation_metadata, sidecar_path
from .fit_mapper import MMapLayerChunkFactory
from .io import sha256_file
from .metrics import R2Accumulator, R2Diagnostics


SCHEMA_VERSION = 1
CAPTURE_SCHEMA_VERSION = 2
PAPER_SOURCE_LAYERS = 40
PAPER_TARGET_LAYERS = 64
PAPER_KV_HEADS = 8
PAPER_HEAD_DIM = 128
PAPER_SELECTED_LAYERS = 8
PAPER_TRAIN_SEQUENCES = 500
PAPER_HELDOUT_SEQUENCES = 50
PAPER_SEQUENCE_LENGTH = 1024
PAPER_STRIDE = 4
PAPER_TRAIN_SAMPLES = 128_000
PAPER_HELDOUT_SAMPLES = 12_800


@dataclass(frozen=True)
class MapperArchitecture:
    """Dimensions persisted by the completed mapper manifest."""

    source_layers: int
    target_layers: int
    kv_heads: int
    head_dim: int
    selected_source_layers: int
    n_samples: int

    @property
    def source_features(self) -> int:
        return self.selected_source_layers * self.kv_heads * self.head_dim

    @property
    def target_features(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def expected_weight_count(self) -> int:
        return 2 * self.target_layers * self.source_features * self.target_features

    @property
    def expected_bias_count(self) -> int:
        return 2 * self.target_layers * self.target_features

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MapperArchitecture":
        required = (
            "source_layers",
            "target_layers",
            "kv_heads",
            "head_dim",
            "selected_source_layers",
            "n_samples",
        )
        missing = [field for field in required if field not in value]
        if missing:
            raise ValueError(f"mapper architecture lacks fields: {missing}")
        result = cls(**{field: int(value[field]) for field in required})
        if any(component <= 0 for component in asdict(result).values()):
            raise ValueError(f"mapper architecture must be positive: {result!r}")
        return result


@dataclass(frozen=True)
class LayerArtifact:
    target_layer: int
    source_layers: tuple[int, ...]
    payload: Path
    payload_sha256: str
    sidecar: Path
    sidecar_sha256: str


@dataclass(frozen=True)
class ValidatedHeldoutInputs:
    architecture: MapperArchitecture
    mapper_manifest: Path
    mapper_manifest_sha256: str
    fit_fingerprint: str
    selection_sha256: str
    training_token_sha256: str
    heldout_token_sha256: str
    source_capture_manifest_sha256: str
    target_capture_manifest_sha256: str
    source_capture_inventory_sha256: str
    target_capture_inventory_sha256: str
    source_root: Path
    target_root: Path
    layers: tuple[LayerArtifact, ...]
    heldout_samples: int
    heldout_sequences: int
    sequence_length: int
    stride: int


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read strict JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _linked_path(raw: object, anchor: Path, *, description: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{description} is not a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        path = anchor / path
    if not path.is_file():
        raise ValueError(f"{description} is missing: {path}")
    return path.resolve()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _assert_sha256(value: object, description: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{description} is not a lowercase SHA256 digest")
    return str(value)


def _assert_paper_architecture(architecture: MapperArchitecture) -> None:
    expected = MapperArchitecture(
        source_layers=PAPER_SOURCE_LAYERS,
        target_layers=PAPER_TARGET_LAYERS,
        kv_heads=PAPER_KV_HEADS,
        head_dim=PAPER_HEAD_DIM,
        selected_source_layers=PAPER_SELECTED_LAYERS,
        n_samples=PAPER_TRAIN_SAMPLES,
    )
    if architecture != expected:
        raise ValueError(
            f"formal held-out validation requires paper architecture {expected!r}, "
            f"observed {architecture!r}"
        )


def _validate_training_inventory(
    mapper: Mapping[str, Any],
    manifest_path: Path,
    architecture: MapperArchitecture,
    *,
    formal_protocol: bool,
) -> tuple[str, str]:
    reference = mapper.get("activation_inventory")
    if not isinstance(reference, Mapping):
        raise ValueError("mapper manifest lacks its training activation inventory")
    inventory_path = _linked_path(
        reference.get("path"), manifest_path.parent, description="training inventory"
    )
    inventory_sha = sha256_file(inventory_path)
    if inventory_sha != reference.get("sha256"):
        raise ValueError("training activation inventory SHA256 mismatch")
    inventory = _strict_json(inventory_path)
    recorded_digest = inventory.get("inventory_digest")
    digest_input = dict(inventory)
    digest_input.pop("inventory_digest", None)
    if recorded_digest != _canonical_sha256(digest_input):
        raise ValueError("training activation inventory canonical digest mismatch")
    if recorded_digest != reference.get("inventory_digest"):
        raise ValueError("mapper and training inventory digests disagree")
    if mapper.get("fit_definition", {}).get("activation_inventory_digest") != recorded_digest:
        raise ValueError("fit definition is not bound to the training inventory")
    if reference.get("payload_checksums_verified") is not True:
        raise ValueError("mapper fit did not record verified training payload checksums")

    shared = inventory.get("shared_provenance")
    if not isinstance(shared, Mapping):
        raise ValueError("training inventory lacks shared provenance")
    token_sha = _assert_sha256(shared.get("token_sha256"), "training token SHA256")
    expected_shared = {
        "split": "train",
        "stride": str(PAPER_STRIDE if formal_protocol else 4),
        "stride_start": "0",
        "sequences": str(
            PAPER_TRAIN_SEQUENCES
            if formal_protocol
            else architecture.n_samples // (PAPER_SEQUENCE_LENGTH // 4)
        ),
    }
    for field, expected in expected_shared.items():
        if str(shared.get(field)) != expected:
            raise ValueError(
                f"training inventory {field}={shared.get(field)!r}, expected {expected!r}"
            )
    expected_shape = [architecture.n_samples, architecture.kv_heads, architecture.head_dim]
    if inventory.get("expected_shape") != expected_shape:
        raise ValueError(
            f"training inventory shape={inventory.get('expected_shape')!r}, "
            f"expected {expected_shape!r}"
        )
    entries = inventory.get("entries")
    expected_entries = 2 * (architecture.source_layers + architecture.target_layers)
    if not isinstance(entries, list) or len(entries) != expected_entries:
        raise ValueError(
            f"training inventory must contain {expected_entries} activation entries"
        )
    expected_keys = {
        (role, cache_type, layer)
        for role, layers in (
            ("source", architecture.source_layers),
            ("target", architecture.target_layers),
        )
        for cache_type in ("k_content", "v")
        for layer in range(layers)
    }
    observed_keys: set[tuple[str, str, int]] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("training inventory contains a non-object entry")
        key = (str(entry.get("role")), str(entry.get("cache_type")), int(entry.get("layer", -1)))
        if key in observed_keys:
            raise ValueError(f"training inventory duplicates {key}")
        observed_keys.add(key)
        if entry.get("shape") != expected_shape:
            raise ValueError(f"training inventory entry {key} has the wrong shape")
        _assert_sha256(entry.get("payload_sha256"), f"training payload {key} SHA256")
        provenance = entry.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"training inventory entry {key} lacks provenance")
        if provenance.get("token_sha256") != token_sha or provenance.get("split") != "train":
            raise ValueError(f"training inventory entry {key} disagrees on token/split")
    if observed_keys != expected_keys:
        raise ValueError("training inventory does not cover the exact source/target K/V layers")
    return token_sha, inventory_sha


def _validate_selection(
    mapper: Mapping[str, Any], manifest_path: Path, architecture: MapperArchitecture
) -> tuple[tuple[tuple[int, ...], ...], str]:
    selection = mapper.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("mapper manifest lacks selection provenance")
    selection_path = _linked_path(
        selection.get("path"), manifest_path.parent, description="selection snapshot"
    )
    selection_sha = sha256_file(selection_path)
    for field in ("source_file_sha256_used_by_fit", "snapshot_sha256"):
        if selection.get(field) != selection_sha:
            raise ValueError(f"selection {field} does not match the snapshot SHA256")
    fit_definition = mapper.get("fit_definition")
    if not isinstance(fit_definition, Mapping) or fit_definition.get("selection_sha256") != selection_sha:
        raise ValueError("fit definition is not bound to the selection snapshot")
    snapshot = _strict_json(selection_path)
    manifest_rows = selection.get("selected_source_layers_rank_order")
    snapshot_rows = snapshot.get("selected_source_layers_rank_order")
    if manifest_rows != snapshot_rows or not isinstance(manifest_rows, list):
        raise ValueError("mapper and selection snapshot top-k rows disagree")
    if len(manifest_rows) != architecture.target_layers:
        raise ValueError("selection row count does not equal target-layer count")
    rows: list[tuple[int, ...]] = []
    for target_layer, raw in enumerate(manifest_rows):
        if not isinstance(raw, list):
            raise ValueError(f"selection row {target_layer} is not a list")
        row = tuple(int(value) for value in raw)
        if len(row) != architecture.selected_source_layers or len(set(row)) != len(row):
            raise ValueError(f"selection row {target_layer} is not a unique top-k row")
        if any(layer < 0 or layer >= architecture.source_layers for layer in row):
            raise ValueError(f"selection row {target_layer} contains an out-of-range layer")
        rows.append(row)
    return tuple(rows), selection_sha


def _validate_mapper_layers(
    mapper: Mapping[str, Any],
    manifest_path: Path,
    architecture: MapperArchitecture,
    rows: Sequence[Sequence[int]],
    selection_sha: str,
) -> tuple[LayerArtifact, ...]:
    entries = mapper.get("layers")
    if not isinstance(entries, list) or len(entries) != architecture.target_layers:
        raise ValueError("mapper manifest does not contain every target layer")
    by_target: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("mapper layer manifest contains a non-object entry")
        target_layer = int(entry.get("target_layer", -1))
        if target_layer in by_target:
            raise ValueError(f"mapper layer manifest duplicates target layer {target_layer}")
        by_target[target_layer] = entry
    if set(by_target) != set(range(architecture.target_layers)):
        raise ValueError("mapper layer manifest target-layer coverage is incomplete")

    source_features = architecture.source_features
    target_features = architecture.target_features
    expected_shapes = {
        "key_weight": [source_features, target_features],
        "key_bias": [target_features],
        "value_weight": [source_features, target_features],
        "value_bias": [target_features],
    }
    expected_weight_numel = 2 * source_features * target_features
    expected_bias_numel = 2 * target_features
    fit_fingerprint = str(mapper["fit_fingerprint"])
    artifacts: list[LayerArtifact] = []
    for target_layer in range(architecture.target_layers):
        entry = by_target[target_layer]
        expected_row = tuple(int(value) for value in rows[target_layer])
        if tuple(entry.get("source_layers_rank_order", ())) != expected_row:
            raise ValueError(f"mapper entry {target_layer} does not use production top-k order")
        if int(entry.get("weight_numel", -1)) != expected_weight_numel:
            raise ValueError(f"mapper entry {target_layer} weight count mismatch")
        if int(entry.get("bias_numel", -1)) != expected_bias_numel:
            raise ValueError(f"mapper entry {target_layer} bias count mismatch")
        payload = _linked_path(
            entry.get("path"), manifest_path.parent, description=f"mapper layer {target_layer}"
        )
        sidecar = _linked_path(
            entry.get("sidecar_path"),
            manifest_path.parent,
            description=f"mapper layer {target_layer} sidecar",
        )
        if payload.name != f"layer_{target_layer:03d}.safetensors":
            raise ValueError(f"mapper layer {target_layer} payload filename is inconsistent")
        if sidecar != payload.with_suffix(".json"):
            raise ValueError(f"mapper layer {target_layer} sidecar path is inconsistent")
        sidecar_sha = sha256_file(sidecar)
        if sidecar_sha != entry.get("sidecar_sha256"):
            raise ValueError(f"mapper layer {target_layer} sidecar SHA256 mismatch")
        raw = _strict_json(sidecar)
        expectations = {
            "target_layer": target_layer,
            "source_layers_rank_order": list(expected_row),
            "dtype": "float32",
            "source_heads": architecture.kv_heads,
            "head_dim": architecture.head_dim,
            "tensor_shapes": expected_shapes,
            "weight_numel": expected_weight_numel,
            "bias_numel": expected_bias_numel,
        }
        for field, expected in expectations.items():
            if raw.get(field) != expected:
                raise ValueError(
                    f"mapper layer {target_layer} sidecar {field}={raw.get(field)!r}, "
                    f"expected {expected!r}"
                )
        metadata = raw.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError(f"mapper layer {target_layer} sidecar lacks metadata")
        metadata_expectations = {
            "fit_fingerprint": fit_fingerprint,
            "selection_manifest_sha256": selection_sha,
            "selected_source_layers_rank_order": list(expected_row),
            "feature_order": "selected_layer_rank,kv_head,head_dim",
            "mapper_dtype": "float32",
            "covariance_dtype": "float32",
            "analysis_dtype": "float64",
            "allow_tf32": False,
            "standardize_features": False,
            "regularize_bias": False,
        }
        for field, expected in metadata_expectations.items():
            if metadata.get(field) != expected:
                raise ValueError(f"mapper layer {target_layer} metadata {field} mismatch")
        payload_sha = sha256_file(payload)
        if payload_sha != entry.get("sha256") or payload_sha != raw.get("sha256"):
            raise ValueError(f"mapper layer {target_layer} payload SHA256 mismatch")
        with safe_open(payload, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(expected_shapes):
                raise ValueError(f"mapper layer {target_layer} tensor set is invalid")
            for name, shape in expected_shapes.items():
                tensor_slice = handle.get_slice(name)
                if tensor_slice.get_shape() != shape or tensor_slice.get_dtype() != "F32":
                    raise ValueError(
                        f"mapper layer {target_layer} {name} header is not FP32 {shape}"
                    )
        artifacts.append(
            LayerArtifact(
                target_layer=target_layer,
                source_layers=expected_row,
                payload=payload,
                payload_sha256=payload_sha,
                sidecar=sidecar,
                sidecar_sha256=sidecar_sha,
            )
        )
    return tuple(artifacts)


def _validate_mapper_manifest(
    manifest_path: Path, *, formal_protocol: bool
) -> tuple[
    MapperArchitecture,
    tuple[LayerArtifact, ...],
    str,
    str,
    str,
    str,
]:
    manifest_path = manifest_path.resolve()
    mapper = _strict_json(manifest_path)
    manifest_sha = sha256_file(manifest_path)
    checksum_path = manifest_path.with_name("manifest.sha256.json")
    checksum = _strict_json(checksum_path)
    if checksum.get("sha256") != manifest_sha:
        raise ValueError("completed mapper manifest SHA256 sidecar mismatch")
    if Path(str(checksum.get("path", ""))).resolve() != manifest_path:
        raise ValueError("completed mapper checksum sidecar points at another manifest")
    common_expectations = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "direction": "qwen3_14b_to_32b",
        "method": "closed-form-centered-ridge-kv-cache-transfer",
        "dtype": "float32",
        "ridge_lambda": 0.01,
        "solver": "cholesky",
        "allow_tf32": False,
    }
    for field, expected in common_expectations.items():
        if mapper.get(field) != expected:
            raise ValueError(f"mapper manifest {field}={mapper.get(field)!r}, expected {expected!r}")
    architecture_raw = mapper.get("architecture")
    if not isinstance(architecture_raw, Mapping):
        raise ValueError("mapper manifest lacks architecture")
    architecture = MapperArchitecture.from_mapping(architecture_raw)
    if formal_protocol:
        _assert_paper_architecture(architecture)
    if mapper.get("weight_count") != architecture.expected_weight_count:
        raise ValueError("mapper aggregate weight count mismatch")
    if mapper.get("bias_count") != architecture.expected_bias_count:
        raise ValueError("mapper aggregate bias count mismatch")
    fit_definition = mapper.get("fit_definition")
    if not isinstance(fit_definition, Mapping):
        raise ValueError("mapper manifest lacks fit definition")
    if fit_definition.get("architecture") != asdict(architecture):
        raise ValueError("fit definition and mapper architecture disagree")
    fit_expectations = {
        "method": "two-pass-centered-ridge",
        "ridge_lambda_added_directly_to_unnormalized_gram": 0.01,
        "standardize_features": False,
        "regularize_bias": False,
        "covariance_dtype": "float32",
        "mapper_dtype": "float32",
        "allow_tf32": False,
        "solver": "cholesky",
        "feature_order": "selected_layer_rank,kv_head,head_dim",
        "output_layout": "target_kv_head,target_head_dim",
    }
    for field, expected in fit_expectations.items():
        if fit_definition.get(field) != expected:
            raise ValueError(f"fit definition {field} mismatch")
    fit_fingerprint = _assert_sha256(mapper.get("fit_fingerprint"), "fit fingerprint")
    if _canonical_sha256(fit_definition) != fit_fingerprint:
        raise ValueError("fit fingerprint is not the canonical fit-definition digest")
    rows, selection_sha = _validate_selection(mapper, manifest_path, architecture)
    training_token_sha, _ = _validate_training_inventory(
        mapper, manifest_path, architecture, formal_protocol=formal_protocol
    )
    layers = _validate_mapper_layers(
        mapper, manifest_path, architecture, rows, selection_sha
    )
    return (
        architecture,
        layers,
        manifest_sha,
        fit_fingerprint,
        selection_sha,
        training_token_sha,
    )


def _capture_payload_path(root: Path, cache_type: str, layer: int) -> Path:
    return root / cache_type / f"layer_{layer:03d}.bf16.bin"


def _validate_capture(
    root: Path,
    *,
    role: str,
    layer_count: int,
    architecture: MapperArchitecture,
    heldout_sequences: int,
    sequence_length: int,
    stride: int,
    verify_payload_checksums: bool,
    formal_protocol: bool,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    manifest = _strict_json(manifest_path)
    accepted_capture_schemas = (
        {CAPTURE_SCHEMA_VERSION} if formal_protocol else {1, CAPTURE_SCHEMA_VERSION}
    )
    if (
        manifest.get("schema_version") not in accepted_capture_schemas
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"held-out {role} capture manifest is not complete")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or manifest.get("inventory_sha256") != _canonical_sha256(entries):
        raise ValueError(f"held-out {role} capture inventory digest mismatch")
    definition = manifest.get("run_definition")
    if not isinstance(definition, Mapping):
        raise ValueError(f"held-out {role} capture lacks run definition")
    model = definition.get("model")
    calibration = definition.get("calibration")
    if not isinstance(model, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError(f"held-out {role} capture lacks model/calibration identity")
    model_architecture = model.get("architecture")
    if not isinstance(model_architecture, Mapping):
        raise ValueError(f"held-out {role} capture lacks model architecture")
    expected_model_name = "qwen3_14b" if role == "source" else "qwen3_32b"
    definition_expectations = {
        "model_name": expected_model_name,
        "sequences": heldout_sequences,
        "sequence_length": sequence_length,
        "sample_stride": stride,
        "sample_stride_start": 0,
        "sampled_positions_per_sequence": sequence_length // stride,
        "forward_dtype": "bfloat16",
        "inverse_rope_dtype": "float32",
    }
    for field, expected in definition_expectations.items():
        if definition.get(field) != expected:
            raise ValueError(
                f"held-out {role} run definition {field}={definition.get(field)!r}, "
                f"expected {expected!r}"
            )
    if model.get("role") != role:
        raise ValueError(f"held-out {role} model role mismatch")
    model_architecture_expectations = {
        "num_hidden_layers": layer_count,
        "num_key_value_heads": architecture.kv_heads,
        "head_dim": architecture.head_dim,
    }
    for field, expected in model_architecture_expectations.items():
        if int(model_architecture.get(field, -1)) != expected:
            raise ValueError(f"held-out {role} model architecture {field} mismatch")
    if calibration.get("split") != "heldout":
        raise ValueError(f"held-out {role} calibration split is not heldout")
    token_sha = _assert_sha256(calibration.get("token_sha256"), f"held-out {role} token SHA256")

    expected_samples = heldout_sequences * (sequence_length // stride)
    expected_shape = (expected_samples, architecture.kv_heads, architecture.head_dim)
    expected_keys = {
        (cache_type, layer)
        for cache_type in ("k_content", "v")
        for layer in range(layer_count)
    }
    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"held-out {role} capture contains a non-object entry")
        key = (str(entry.get("cache_type")), int(entry.get("layer", -1)))
        if key in by_key:
            raise ValueError(f"held-out {role} capture duplicates {key}")
        by_key[key] = entry
    if set(by_key) != expected_keys:
        raise ValueError(f"held-out {role} capture layer/KV coverage is incomplete")

    shared_fields = {
        "token_sha256": set(),
        "calibration_manifest_sha256": set(),
        "dataset_fingerprint": set(),
        "split": set(),
        "stride": set(),
        "stride_start": set(),
        "sequences": set(),
    }
    for cache_type, layer in sorted(expected_keys):
        entry = by_key[(cache_type, layer)]
        payload = _capture_payload_path(root, cache_type, layer)
        recorded_path = root / str(entry.get("path", ""))
        if recorded_path.resolve() != payload.resolve():
            raise ValueError(f"held-out {role} entry {(cache_type, layer)} path mismatch")
        metadata = read_activation_metadata(payload)
        if metadata.shape != expected_shape or entry.get("shape") != list(expected_shape):
            raise ValueError(f"held-out {role} {(cache_type, layer)} activation shape mismatch")
        if payload.stat().st_size != metadata.nbytes or int(entry.get("nbytes", -1)) != metadata.nbytes:
            raise ValueError(f"held-out {role} {(cache_type, layer)} byte count mismatch")
        metadata_path = sidecar_path(payload)
        if sha256_file(metadata_path) != entry.get("sidecar_sha256"):
            raise ValueError(f"held-out {role} {(cache_type, layer)} sidecar SHA256 mismatch")
        if metadata.sha256 != entry.get("sha256"):
            raise ValueError(f"held-out {role} {(cache_type, layer)} metadata SHA256 mismatch")
        if verify_payload_checksums and sha256_file(payload) != metadata.sha256:
            raise ValueError(f"held-out {role} {(cache_type, layer)} payload SHA256 mismatch")
        provenance = metadata.provenance
        provenance_expectations = {
            "model_name": expected_model_name,
            "model_role": role,
            "cache_type": cache_type,
            "layer": layer,
            "split": "heldout",
            "stride": stride,
            "stride_start": 0,
            "sequences": heldout_sequences,
            "forward_dtype": "bfloat16",
            "content_key_inverse_dtype": "float32",
        }
        for field, expected in provenance_expectations.items():
            if provenance.get(field) != expected:
                raise ValueError(
                    f"held-out {role} {(cache_type, layer)} provenance {field} mismatch"
                )
        for field in shared_fields:
            if field not in provenance:
                raise ValueError(
                    f"held-out {role} {(cache_type, layer)} provenance lacks {field}"
                )
            shared_fields[field].add(str(provenance[field]))
    for field, values in shared_fields.items():
        if len(values) != 1:
            raise ValueError(f"held-out {role} capture disagrees on shared {field}")
    if next(iter(shared_fields["token_sha256"])) != token_sha:
        raise ValueError(f"held-out {role} manifest and payload token hashes disagree")
    if formal_protocol and expected_samples != PAPER_HELDOUT_SAMPLES:
        raise AssertionError("formal held-out sample-count invariant failed")
    return {
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "inventory_sha256": str(manifest["inventory_sha256"]),
        "token_sha256": token_sha,
        "calibration_manifest_sha256": next(
            iter(shared_fields["calibration_manifest_sha256"])
        ),
        "dataset_fingerprint": next(iter(shared_fields["dataset_fingerprint"])),
        "samples": expected_samples,
    }


def validate_heldout_inputs(
    mapper_manifest: str | Path,
    source_root: str | Path,
    target_root: str | Path,
    *,
    verify_payload_checksums: bool = True,
    formal_protocol: bool = True,
    heldout_sequences: int = PAPER_HELDOUT_SEQUENCES,
    sequence_length: int = PAPER_SEQUENCE_LENGTH,
    stride: int = PAPER_STRIDE,
) -> ValidatedHeldoutInputs:
    """Strictly bind a complete mapper to matching, unseen held-out stores."""

    if heldout_sequences <= 0 or sequence_length <= 0 or stride <= 0:
        raise ValueError("held-out sequences, sequence length, and stride must be positive")
    if sequence_length % stride:
        raise ValueError("sequence length must be divisible by the activation stride")
    if formal_protocol and (
        heldout_sequences,
        sequence_length,
        stride,
    ) != (PAPER_HELDOUT_SEQUENCES, PAPER_SEQUENCE_LENGTH, PAPER_STRIDE):
        raise ValueError("formal held-out protocol is fixed to 50x1024 with stride 4")
    mapper_path = Path(mapper_manifest).resolve()
    (
        architecture,
        layers,
        mapper_sha,
        fit_fingerprint,
        selection_sha,
        training_token_sha,
    ) = _validate_mapper_manifest(mapper_path, formal_protocol=formal_protocol)
    source = _validate_capture(
        Path(source_root),
        role="source",
        layer_count=architecture.source_layers,
        architecture=architecture,
        heldout_sequences=heldout_sequences,
        sequence_length=sequence_length,
        stride=stride,
        verify_payload_checksums=verify_payload_checksums,
        formal_protocol=formal_protocol,
    )
    target = _validate_capture(
        Path(target_root),
        role="target",
        layer_count=architecture.target_layers,
        architecture=architecture,
        heldout_sequences=heldout_sequences,
        sequence_length=sequence_length,
        stride=stride,
        verify_payload_checksums=verify_payload_checksums,
        formal_protocol=formal_protocol,
    )
    cross_fields = (
        "token_sha256",
        "calibration_manifest_sha256",
        "dataset_fingerprint",
        "samples",
    )
    for field in cross_fields:
        if source[field] != target[field]:
            raise ValueError(f"source and target held-out captures disagree on {field}")
    heldout_token_sha = str(source["token_sha256"])
    if heldout_token_sha == training_token_sha:
        raise ValueError("training and held-out activation stores use the same token SHA256")
    return ValidatedHeldoutInputs(
        architecture=architecture,
        mapper_manifest=mapper_path,
        mapper_manifest_sha256=mapper_sha,
        fit_fingerprint=fit_fingerprint,
        selection_sha256=selection_sha,
        training_token_sha256=training_token_sha,
        heldout_token_sha256=heldout_token_sha,
        source_capture_manifest_sha256=str(source["manifest_sha256"]),
        target_capture_manifest_sha256=str(target["manifest_sha256"]),
        source_capture_inventory_sha256=str(source["inventory_sha256"]),
        target_capture_inventory_sha256=str(target["inventory_sha256"]),
        source_root=Path(source["root"]),
        target_root=Path(target["root"]),
        layers=layers,
        heldout_samples=int(source["samples"]),
        heldout_sequences=heldout_sequences,
        sequence_length=sequence_length,
        stride=stride,
    )


def _diagnostics_dict(value: R2Diagnostics) -> dict[str, Any]:
    result = {
        "n_samples": value.n_samples,
        "n_channels": value.n_channels,
        "r2_uniform_channel_average_fp64": value.uniform_average,
        "r2_global_fp64": value.global_r2,
        "r2_flattened_global_fp64": value.flattened_global_r2,
        "mse_fp64": value.mse,
        "global_sse_fp64": value.global_sse,
        "global_sst_fp64": value.global_sst,
    }
    if not all(math.isfinite(float(number)) for number in result.values()):
        raise FloatingPointError("held-out R2 diagnostics contain NaN or Inf")
    return result


@torch.no_grad()
def _evaluate_cache_type(
    *,
    source_root: Path,
    target_root: Path,
    cache_type: str,
    target_layer: int,
    source_layers: Sequence[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    chunk_tokens: int,
    compute_device: torch.device,
    analysis_device: torch.device,
) -> dict[str, Any]:
    source_paths = [
        _capture_payload_path(source_root, cache_type, source_layer)
        for source_layer in source_layers
    ]
    target_path = _capture_payload_path(target_root, cache_type, target_layer)
    accumulator = R2Accumulator(analysis_device=analysis_device)
    started = time.perf_counter()
    with MMapLayerChunkFactory(
        source_paths,
        target_path,
        chunk_samples=chunk_tokens,
        # The complete held-out stores were hashed once in the preflight.  Do
        # not hash the same selected source file repeatedly for every target.
        verify_checksum=False,
    ) as chunks:
        for x_bf16, y_bf16 in chunks():
            x = x_bf16.to(device=compute_device, dtype=torch.float32)
            prediction = x @ weight + bias
            accumulator.update(y_bf16, prediction)
    if compute_device.type == "cuda":
        torch.cuda.synchronize(compute_device)
    result = _diagnostics_dict(accumulator.finalize())
    result["elapsed_seconds"] = time.perf_counter() - started
    return result


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError("cannot summarize empty or non-finite held-out scores")
    minimum_index = min(range(len(values)), key=lambda index: values[index])
    maximum_index = max(range(len(values)), key=lambda index: values[index])
    return {
        "mean_across_target_layers": statistics.fmean(values),
        "median_across_target_layers": statistics.median(values),
        "min": float(values[minimum_index]),
        "min_target_layer": minimum_index,
        "max": float(values[maximum_index]),
        "max_target_layer": maximum_index,
    }


@torch.no_grad()
def evaluate_heldout_mapper(
    inputs: ValidatedHeldoutInputs,
    *,
    chunk_tokens: int = 256,
    compute_device: str | torch.device = "cpu",
    analysis_device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Apply every production mapper and return layer/aggregate held-out R2."""

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    compute = torch.device(compute_device)
    analysis = compute if str(analysis_device) == "compute" else torch.device(analysis_device)
    for role, device in (("compute", compute), ("analysis", analysis)):
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(f"{role} device requests CUDA, but CUDA is unavailable")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError(f"{role} device {device} is out of range")
    if compute.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    layer_results: list[dict[str, Any]] = []
    expected_shapes = {
        "key_weight": (inputs.architecture.source_features, inputs.architecture.target_features),
        "key_bias": (inputs.architecture.target_features,),
        "value_weight": (inputs.architecture.source_features, inputs.architecture.target_features),
        "value_bias": (inputs.architecture.target_features,),
    }
    for artifact in inputs.layers:
        tensors = load_file(artifact.payload, device=str(compute))
        if set(tensors) != set(expected_shapes):
            raise ValueError(f"mapper layer {artifact.target_layer} tensor set changed after preflight")
        for name, shape in expected_shapes.items():
            tensor = tensors[name]
            if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
                raise ValueError(f"mapper layer {artifact.target_layer} {name} shape/dtype mismatch")
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(f"mapper layer {artifact.target_layer} {name} is non-finite")
        key = _evaluate_cache_type(
            source_root=inputs.source_root,
            target_root=inputs.target_root,
            cache_type="k_content",
            target_layer=artifact.target_layer,
            source_layers=artifact.source_layers,
            weight=tensors["key_weight"],
            bias=tensors["key_bias"],
            chunk_tokens=chunk_tokens,
            compute_device=compute,
            analysis_device=analysis,
        )
        value = _evaluate_cache_type(
            source_root=inputs.source_root,
            target_root=inputs.target_root,
            cache_type="v",
            target_layer=artifact.target_layer,
            source_layers=artifact.source_layers,
            weight=tensors["value_weight"],
            bias=tensors["value_bias"],
            chunk_tokens=chunk_tokens,
            compute_device=compute,
            analysis_device=analysis,
        )
        key_score = float(key["r2_uniform_channel_average_fp64"])
        value_score = float(value["r2_uniform_channel_average_fp64"])
        layer_results.append(
            {
                "target_layer": artifact.target_layer,
                "source_layers_rank_order": list(artifact.source_layers),
                "mapper_payload_sha256": artifact.payload_sha256,
                "key": key,
                "value": value,
                "key_value_equal_mean_uniform_channel_r2_fp64": (
                    key_score + value_score
                )
                / 2.0,
            }
        )
        del tensors
        if compute.type == "cuda":
            torch.cuda.empty_cache()

    key_values = [
        float(layer["key"]["r2_uniform_channel_average_fp64"])
        for layer in layer_results
    ]
    value_values = [
        float(layer["value"]["r2_uniform_channel_average_fp64"])
        for layer in layer_results
    ]
    combined_values = [
        float(layer["key_value_equal_mean_uniform_channel_r2_fp64"])
        for layer in layer_results
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "direction": "qwen3_14b_to_32b",
        "method": "heldout-bf16-activation-mapper-reconstruction",
        "paper_alignment": {
            "primary_metric": "uniform arithmetic mean of independently centered per-channel R2",
            "target_layer_reduction": "uniform arithmetic mean across target layers",
            "key_value_reduction": "reported separately; equal arithmetic mean is an additional summary",
            "prediction_dtype": "float32",
            "analysis_dtype": "float64",
            "benchmark_data_used": False,
            "heldout_split_note": (
                "local 50-sequence generalization diagnostic; not a paper benchmark result "
                "and not used for source-layer selection or mapper fitting"
            ),
        },
        "protocol": {
            "split": "heldout",
            "sequences": inputs.heldout_sequences,
            "sequence_length": inputs.sequence_length,
            "position_stride": inputs.stride,
            "observations": inputs.heldout_samples,
            "chunk_tokens": chunk_tokens,
            "compute_device": str(compute),
            "analysis_device": str(analysis),
            "feature_order": "selected_layer_rank,kv_head,head_dim",
            "apply_equation": "Y_hat = FP32(X) @ FP32(W) + FP32(b)",
        },
        "inputs": {
            "mapper_manifest": str(inputs.mapper_manifest),
            "mapper_manifest_sha256": inputs.mapper_manifest_sha256,
            "fit_fingerprint": inputs.fit_fingerprint,
            "selection_sha256": inputs.selection_sha256,
            "source_activation_root": str(inputs.source_root),
            "source_capture_manifest_sha256": inputs.source_capture_manifest_sha256,
            "source_capture_inventory_sha256": inputs.source_capture_inventory_sha256,
            "target_activation_root": str(inputs.target_root),
            "target_capture_manifest_sha256": inputs.target_capture_manifest_sha256,
            "target_capture_inventory_sha256": inputs.target_capture_inventory_sha256,
            "training_token_sha256": inputs.training_token_sha256,
            "heldout_token_sha256": inputs.heldout_token_sha256,
            "training_and_heldout_token_sha256_differ": True,
        },
        "architecture": asdict(inputs.architecture),
        "layers": layer_results,
        "summary": {
            "target_layers": len(layer_results),
            "key_uniform_channel_r2_fp64": _summary(key_values),
            "value_uniform_channel_r2_fp64": _summary(value_values),
            "key_value_equal_mean_uniform_channel_r2_fp64": _summary(combined_values),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


__all__ = [
    "MapperArchitecture",
    "LayerArtifact",
    "ValidatedHeldoutInputs",
    "evaluate_heldout_mapper",
    "validate_heldout_inputs",
]
