"""Frozen reverse-direction protocol and capture-role provenance checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import yaml


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config/mapper.yaml"
DIRECTION = "qwen3_32b_to_14b"
TOP_K = 20
TRAIN_OBSERVATIONS = 128_000
SOURCE_FEATURES = 20_480
TARGET_FEATURES = 1_024
WEIGHT_COUNT = 1_677_721_600
BIAS_COUNT = 81_920


@dataclass(frozen=True)
class EndpointBinding:
    semantic_role: str
    model_name: str
    model_path: Path
    layers: int
    capture_native_role: str
    train_root: Path
    heldout_root: Path


SOURCE = EndpointBinding(
    semantic_role="source",
    model_name="qwen3_32b",
    model_path=Path("/home/wby/models/Qwen3-32B"),
    layers=64,
    capture_native_role="target",
    train_root=REPOSITORY_ROOT / "data/activations/qwen3_32b/train",
    heldout_root=REPOSITORY_ROOT / "data/activations/qwen3_32b/heldout",
)
TARGET = EndpointBinding(
    semantic_role="target",
    model_name="qwen3_14b",
    model_path=Path("/home/wby/models/Qwen3-14B"),
    layers=40,
    capture_native_role="source",
    train_root=REPOSITORY_ROOT / "data/activations/qwen3_14b/train",
    heldout_root=REPOSITORY_ROOT / "data/activations/qwen3_14b/heldout",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("reverse mapper config must be a YAML mapping")
    return value


def _resolved(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def assert_inside_experiment(path: str | Path, *, description: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError(f"{description} escapes isolated experiment root: {resolved}") from exc
    return resolved


def validate_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = load_config(config_path)
    fixed = {
        "project_root": EXPERIMENT_ROOT,
        "shared_repository": REPOSITORY_ROOT,
        "source_model": SOURCE.model_path,
        "target_model": TARGET.model_path,
    }
    for field, expected in fixed.items():
        observed = Path(str(raw.get(field, ""))).resolve()
        if observed != expected.resolve():
            raise ValueError(f"config {field}={observed}, expected {expected.resolve()}")
    if raw.get("direction") != DIRECTION:
        raise ValueError(f"config direction must be {DIRECTION}")

    paper = raw.get("paper")
    if not isinstance(paper, Mapping):
        raise ValueError("config lacks paper protocol")
    paper_expected = {
        "calibration_sequences": 500,
        "sequence_length": 1024,
        "position_stride": 4,
        "selected_source_layers": TOP_K,
        "ridge_lambda": 0.01,
        "forward_dtype": "bfloat16",
        "covariance_dtype": "float32",
        "analysis_dtype": "float64",
    }
    for field, expected in paper_expected.items():
        if paper.get(field) != expected:
            raise ValueError(f"paper.{field}={paper.get(field)!r}, expected {expected!r}")

    local_validation = raw.get("local_validation")
    local_validation_expected = {
        "heldout_sequences": 50,
        "sequence_length": 1024,
        "position_stride": 4,
    }
    if not isinstance(local_validation, Mapping):
        raise ValueError("config lacks local validation protocol")
    for field, expected in local_validation_expected.items():
        if local_validation.get(field) != expected:
            raise ValueError(
                f"local_validation.{field}={local_validation.get(field)!r}, "
                f"expected {expected!r}"
            )

    architecture = raw.get("architecture")
    architecture_expected = {
        "source_layers": 64,
        "target_layers": 40,
        "source_kv_heads": 8,
        "target_kv_heads": 8,
        "source_head_dim": 128,
        "target_head_dim": 128,
        "rope_theta": 1_000_000,
    }
    if not isinstance(architecture, Mapping):
        raise ValueError("config lacks architecture")
    for field, expected in architecture_expected.items():
        if architecture.get(field) != expected:
            raise ValueError(
                f"architecture.{field}={architecture.get(field)!r}, expected {expected!r}"
            )

    captures = raw.get("captures")
    if not isinstance(captures, Mapping):
        raise ValueError("config lacks capture bindings")
    capture_expected = {
        "source_train": SOURCE.train_root,
        "target_train": TARGET.train_root,
        "source_heldout": SOURCE.heldout_root,
        "target_heldout": TARGET.heldout_root,
    }
    for field, expected in capture_expected.items():
        observed = Path(str(captures.get(field, ""))).resolve()
        if observed != expected.resolve():
            raise ValueError(f"captures.{field}={observed}, expected {expected.resolve()}")
        try:
            observed.relative_to(EXPERIMENT_ROOT)
        except ValueError:
            pass
        else:
            raise ValueError(f"reused capture must remain outside writable experiment: {observed}")
    if captures.get("source_native_role") != SOURCE.capture_native_role:
        raise ValueError("32B source capture must retain native target role")
    if captures.get("target_native_role") != TARGET.capture_native_role:
        raise ValueError("14B target capture must retain native source role")

    outputs = raw.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ValueError("config lacks isolated outputs")
    resolved_outputs = {
        field: assert_inside_experiment(
            _resolved(EXPERIMENT_ROOT, value), description=f"outputs.{field}"
        )
        for field, value in outputs.items()
    }
    return {
        "path": config_path,
        "sha256": sha256_file(config_path),
        "raw": raw,
        "outputs": resolved_outputs,
    }


def _payload_path(root: Path, cache_type: str, layer: int) -> Path:
    return root / cache_type / f"layer_{layer:03d}.bf16.bin"


def _sidecar_path(payload: Path) -> Path:
    return payload.with_suffix("").with_suffix(".json")


def scan_capture(
    binding: EndpointBinding,
    *,
    split: str,
    verify_payloads: bool = False,
) -> dict[str, Any]:
    if split not in ("train", "heldout"):
        raise ValueError(f"unknown split: {split}")
    root = binding.train_root if split == "train" else binding.heldout_root
    sequences = 500 if split == "train" else 50
    observations = sequences * 256
    expected_shape = [observations, 8, 128]
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 2 or manifest.get("status") != "complete":
        raise ValueError(f"capture is not a complete schema-2 store: {manifest_path}")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 2 * binding.layers:
        raise ValueError(f"{root}: expected {2 * binding.layers} manifest entries")
    if manifest.get("inventory_sha256") != canonical_sha256(entries):
        raise ValueError(f"capture inventory digest mismatch: {root}")

    definition = manifest.get("run_definition")
    if not isinstance(definition, Mapping):
        raise ValueError(f"capture lacks run definition: {root}")
    model = definition.get("model")
    calibration = definition.get("calibration")
    if not isinstance(model, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError(f"capture lacks model/calibration provenance: {root}")
    definition_expected = {
        "model_name": binding.model_name,
        "sequences": sequences,
        "sequence_length": 1024,
        "sample_stride": 4,
        "sample_stride_start": 0,
        "sampled_positions_per_sequence": 256,
        "forward_dtype": "bfloat16",
        "inverse_rope_dtype": "float32",
    }
    for field, expected in definition_expected.items():
        if definition.get(field) != expected:
            raise ValueError(f"{root}: run_definition.{field} mismatch")
    if model.get("role") != binding.capture_native_role:
        raise ValueError(
            f"{root}: native role {model.get('role')!r}, expected "
            f"{binding.capture_native_role!r} for reverse semantic {binding.semantic_role!r}"
        )
    model_architecture = model.get("architecture")
    if not isinstance(model_architecture, Mapping):
        raise ValueError(f"{root}: missing model architecture")
    architecture_expected = {
        "num_hidden_layers": binding.layers,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rope_theta": 1_000_000.0,
    }
    for field, expected in architecture_expected.items():
        if model_architecture.get(field) != expected:
            raise ValueError(f"{root}: model architecture {field} mismatch")
    if calibration.get("split") != split:
        raise ValueError(f"{root}: calibration split mismatch")
    token_sha = str(calibration.get("token_sha256", ""))
    if len(token_sha) != 64:
        raise ValueError(f"{root}: invalid token SHA256")

    by_key: dict[tuple[str, int], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{root}: malformed manifest entry")
        key = (str(entry.get("cache_type")), int(entry.get("layer", -1)))
        if key in by_key:
            raise ValueError(f"{root}: duplicate manifest entry {key}")
        by_key[key] = entry
    expected_keys = {
        (cache_type, layer)
        for cache_type in ("k_content", "v")
        for layer in range(binding.layers)
    }
    if set(by_key) != expected_keys:
        raise ValueError(f"{root}: incomplete K/V layer inventory")

    inventory_entries: list[dict[str, Any]] = []
    shared: dict[str, set[str]] = {
        key: set()
        for key in (
            "token_sha256",
            "split",
            "stride",
            "stride_start",
            "sequences",
            "calibration_manifest_sha256",
            "dataset_fingerprint",
        )
    }
    for cache_type, layer in sorted(expected_keys):
        recorded = by_key[(cache_type, layer)]
        payload = _payload_path(root, cache_type, layer)
        sidecar = _sidecar_path(payload)
        metadata = read_json(sidecar)
        if not payload.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"capture payload/sidecar missing: {payload}")
        if (root / str(recorded.get("path", ""))).resolve() != payload.resolve():
            raise ValueError(f"{root}: entry path mismatch for {(cache_type, layer)}")
        if metadata.get("shape") != expected_shape or recorded.get("shape") != expected_shape:
            raise ValueError(f"{root}: shape mismatch for {(cache_type, layer)}")
        expected_nbytes = observations * 8 * 128 * 2
        if payload.stat().st_size != expected_nbytes:
            raise ValueError(f"{payload}: byte count mismatch")
        if int(metadata.get("nbytes", -1)) != expected_nbytes:
            raise ValueError(f"{sidecar}: byte count mismatch")
        sidecar_sha = sha256_file(sidecar)
        if sidecar_sha != recorded.get("sidecar_sha256"):
            raise ValueError(f"{sidecar}: SHA256 mismatch")
        payload_sha = str(metadata.get("sha256", ""))
        if payload_sha != recorded.get("sha256"):
            raise ValueError(f"{payload}: manifest/sidecar SHA256 mismatch")
        if verify_payloads and sha256_file(payload) != payload_sha:
            raise ValueError(f"{payload}: payload SHA256 mismatch")
        provenance = metadata.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError(f"{sidecar}: missing provenance")
        provenance_expected = {
            "model_name": binding.model_name,
            "model_role": binding.capture_native_role,
            "cache_type": cache_type,
            "layer": layer,
            "split": split,
            "stride": 4,
            "stride_start": 0,
            "sequences": sequences,
            "forward_dtype": "bfloat16",
            "content_key_inverse_dtype": "float32",
        }
        for field, expected in provenance_expected.items():
            if provenance.get(field) != expected:
                raise ValueError(f"{sidecar}: provenance {field} mismatch")
        for field in shared:
            if field not in provenance:
                raise ValueError(f"{sidecar}: provenance lacks {field}")
            shared[field].add(str(provenance[field]))
        inventory_entries.append(
            {
                "role": binding.semantic_role,
                "capture_native_role": binding.capture_native_role,
                "cache_type": cache_type,
                "layer": layer,
                "payload_path": str(payload.resolve()),
                "nbytes": expected_nbytes,
                "payload_sha256": payload_sha,
                "metadata_path": str(sidecar.resolve()),
                "metadata_sha256": sidecar_sha,
                "shape": expected_shape,
                "provenance": dict(provenance),
                "provenance_sha256": canonical_sha256(provenance),
            }
        )
    for field, values in shared.items():
        if len(values) != 1:
            raise ValueError(f"{root}: payloads disagree on provenance {field}")
    if next(iter(shared["token_sha256"])) != token_sha:
        raise ValueError(f"{root}: manifest and payload token hashes disagree")
    return {
        "semantic_role": binding.semantic_role,
        "capture_native_role": binding.capture_native_role,
        "model_name": binding.model_name,
        "root": str(root.resolve()),
        "split": split,
        "layers": binding.layers,
        "observations": observations,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "capture_inventory_sha256": str(manifest["inventory_sha256"]),
        "token_sha256": token_sha,
        "calibration_manifest_sha256": next(iter(shared["calibration_manifest_sha256"])),
        "dataset_fingerprint": next(iter(shared["dataset_fingerprint"])),
        "payload_checksums_verified": verify_payloads,
        "entries": inventory_entries,
    }


def validate_capture_pair(
    *, split: str, verify_payloads: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = scan_capture(SOURCE, split=split, verify_payloads=verify_payloads)
    target = scan_capture(TARGET, split=split, verify_payloads=verify_payloads)
    for field in (
        "token_sha256",
        "calibration_manifest_sha256",
        "dataset_fingerprint",
        "observations",
    ):
        if source[field] != target[field]:
            raise ValueError(f"reverse source/target captures disagree on {field}")
    return source, target


def fit_activation_inventory(
    *, verify_payloads: bool
) -> dict[str, Any]:
    source, target = validate_capture_pair(split="train", verify_payloads=verify_payloads)
    entries = [*source["entries"], *target["entries"]]
    shared = {
        "token_sha256": source["token_sha256"],
        "split": "train",
        "stride": "4",
        "stride_start": "0",
        "sequences": "500",
    }
    inventory = {
        "schema_version": 1,
        "direction": DIRECTION,
        "semantic_capture_role_binding": {
            "source": SOURCE.capture_native_role,
            "target": TARGET.capture_native_role,
        },
        "payload_checksums_verified": verify_payloads,
        "expected_shape": [128_000, 8, 128],
        "shared_provenance": shared,
        "entries": entries,
    }
    inventory["inventory_digest"] = canonical_sha256(inventory)
    return inventory


def protocol_summary(*, verify_payloads: bool = False) -> dict[str, Any]:
    config = validate_config()
    train_source, train_target = validate_capture_pair(
        split="train", verify_payloads=verify_payloads
    )
    held_source, held_target = validate_capture_pair(
        split="heldout", verify_payloads=verify_payloads
    )
    if train_source["token_sha256"] == held_source["token_sha256"]:
        raise ValueError("train and held-out captures reuse the same token payload")
    return {
        "schema_version": 1,
        "status": "passed",
        "direction": DIRECTION,
        "config": {"path": str(config["path"]), "sha256": config["sha256"]},
        "paper_disclosed_protocol": {
            "train": "500x1024_stride4",
            "top_k": TOP_K,
            "ridge_lambda": 0.01,
            "source_features": SOURCE_FEATURES,
            "target_features": TARGET_FEATURES,
            "target_layers": 40,
        },
        "local_validation_protocol": {
            "heldout": "50x1024_stride4",
            "purpose": "independent activation reconstruction audit; not paper-disclosed",
            "train_heldout_tokens_differ": True,
        },
        "locally_derived_mapper_size": {
            "derivation": "paper k=20 and mapper formula applied to local Qwen3 architecture",
            "paper_directly_reports_exact_reverse_storage_bytes": False,
            "weight_count": WEIGHT_COUNT,
            "bias_count": BIAS_COUNT,
            "fp32_parameter_bytes": (WEIGHT_COUNT + BIAS_COUNT) * 4,
            "fp32_parameter_gib": (WEIGHT_COUNT + BIAS_COUNT) * 4 / (1024**3),
        },
        "role_binding": {
            "source": {
                "semantic_model": SOURCE.model_name,
                "capture_native_role": SOURCE.capture_native_role,
            },
            "target": {
                "semantic_model": TARGET.model_name,
                "capture_native_role": TARGET.capture_native_role,
            },
        },
        "captures": {
            "source_train": {k: v for k, v in train_source.items() if k != "entries"},
            "target_train": {k: v for k, v in train_target.items() if k != "entries"},
            "source_heldout": {k: v for k, v in held_source.items() if k != "entries"},
            "target_heldout": {k: v for k, v in held_target.items() if k != "entries"},
        },
        "train_heldout_tokens_differ": True,
        "payload_checksums_verified": verify_payloads,
    }


def endpoint_asdict(binding: EndpointBinding) -> dict[str, Any]:
    value = asdict(binding)
    return {key: str(item) if isinstance(item, Path) else item for key, item in value.items()}
