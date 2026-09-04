"""Content-addressed, fail-closed contracts shared by prepare and execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
PREPARED_ROOT = ROOT / "artifacts" / "prepared_v1"
RUNS_ROOT = ROOT / "runs"
DATASETS = ("hotpotqa_train", "musique_train")
PARTITIONS = ("screen", "confirm")
PROFILES = ("smoke", "screen", "confirm")
MODES = (
    "native14",
    "nvidia_k20",
    "shrinkage",
    "tail_refresh_r8",
    "tail_refresh_r32",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ContractError(RuntimeError):
    """Raised whenever data, code, a receipt, or a checkpoint drifts."""


def canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical-JSON serializable: {exc}") from exc
    return payload + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError("token IDs must be non-negative integers")
        normalized.append(value)
    return canonical_sha256(normalized)


def _reject_symlink_components(path: Path, label: str) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ContractError(f"{label} contains a symlink: {component}")


def require_child(root: str | Path, value: str | Path, *, label: str) -> Path:
    root_path = Path(root)
    value_path = Path(value)
    _reject_symlink_components(root_path, f"{label} root")
    _reject_symlink_components(value_path, label)
    resolved_root = root_path.resolve()
    resolved = value_path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractError(f"{label} escapes {resolved_root}: {resolved}") from exc
    if not relative.parts:
        raise ContractError(f"{label} cannot be the root itself")
    return resolved


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    nbytes: int
    sha256: str

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "ArtifactRef":
        if not isinstance(value, Mapping) or set(value) != {"path", "nbytes", "sha256"}:
            raise ContractError(f"{label} must contain exactly path, nbytes, sha256")
        raw_path = value.get("path")
        raw_size = value.get("nbytes")
        raw_hash = value.get("sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise ContractError(f"{label}.path must be absolute")
        if isinstance(raw_size, bool) or not isinstance(raw_size, int) or raw_size < 0:
            raise ContractError(f"{label}.nbytes must be a non-negative integer")
        if not isinstance(raw_hash, str) or SHA256_RE.fullmatch(raw_hash) is None:
            raise ContractError(f"{label}.sha256 is malformed")
        return cls(path=raw_path, nbytes=raw_size, sha256=raw_hash)

    def verify(self, *, label: str = "artifact", deep_hash: bool = True) -> Path:
        lexical = Path(self.path)
        _reject_symlink_components(lexical, label)
        path = lexical.resolve()
        if not path.is_file():
            raise ContractError(f"{label} is not a regular file: {path}")
        observed_size = path.stat().st_size
        if observed_size != self.nbytes:
            raise ContractError(
                f"{label} size drift: {observed_size} != {self.nbytes}: {path}"
            )
        if deep_hash:
            observed_hash = sha256_file(path)
            if observed_hash != self.sha256:
                raise ContractError(
                    f"{label} SHA-256 drift: {observed_hash} != {self.sha256}: {path}"
                )
        return path

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def artifact_ref(path: str | Path) -> ArtifactRef:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ContractError(f"artifact does not exist: {resolved}")
    return ArtifactRef(
        path=str(resolved), nbytes=resolved.stat().st_size, sha256=sha256_file(resolved)
    )


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ContractError(f"invalid JSON: {path}: {exc}") from exc


def read_json_ref(value: Any, *, label: str) -> tuple[dict[str, Any], ArtifactRef]:
    reference = ArtifactRef.parse(value, label=label)
    path = reference.verify(label=label)
    document = read_json(path)
    if not isinstance(document, dict):
        raise ContractError(f"{label} JSON root must be an object")
    return document, reference


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ContractError(f"blank JSONL row at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ContractError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    encoded = [canonical_bytes(dict(row)) for row in rows]
    return (b"\n".join(encoded) + b"\n") if encoded else b""


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: str | Path, value: Any) -> ArtifactRef:
    destination = Path(path)
    payload = canonical_bytes(value, newline=True)
    _atomic_bytes(destination, payload)
    return artifact_ref(destination)


def atomic_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> ArtifactRef:
    destination = Path(path)
    payload = jsonl_bytes(rows)
    _atomic_bytes(destination, payload)
    return artifact_ref(destination)


def immutable_json(path: str | Path, value: Any) -> ArtifactRef:
    destination = Path(path)
    payload = canonical_bytes(value, newline=True)
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ContractError(f"immutable artifact is not a regular file: {destination}")
        if destination.read_bytes() != payload:
            raise ContractError(f"immutable artifact differs: {destination}")
        return artifact_ref(destination)
    _atomic_bytes(destination, payload)
    return artifact_ref(destination)


def write_sidecar(reference: ArtifactRef) -> ArtifactRef:
    path = Path(reference.path)
    return atomic_json(
        path.with_name(path.name + ".sha256.json"),
        {
            "schema_version": 1,
            "path": path.name,
            "nbytes": reference.nbytes,
            "sha256": reference.sha256,
        },
    )


def finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be finite")
    return result


def load_config(path: str | Path = DEFAULT_CONFIG, *, verify_inputs: bool = True) -> dict[str, Any]:
    config_path = Path(path).resolve()
    config = read_json(config_path)
    if not isinstance(config, dict):
        raise ContractError("config root must be an object")
    fixed = {
        "schema_version": 1,
        "protocol_id": "qwen3_32b_to_14b_method_dev_paired_v1",
        "status": "method_dev_only_official_dev_sealed",
        "direction": "qwen3_32b_to_14b",
    }
    for field, expected in fixed.items():
        if config.get(field) != expected:
            raise ContractError(f"config {field} changed")
    generation = config.get("generation")
    if not isinstance(generation, Mapping) or generation != {
        "max_length": 16384,
        "max_new_tokens": 256,
        "batch_size": 1,
        "do_sample": False,
        "temperature": 0.0,
        "stop_sequences": ["\nQuestion:", "\nContext:"],
        "overlength_policy": "exact_token_id_head_tail_before_runtime",
    }:
        raise ContractError("generation contract changed")
    profiles = config.get("profiles")
    expected_profiles = {
        "smoke": {"hotpotqa_train": 2, "musique_train": 2},
        "screen": {"hotpotqa_train": 128, "musique_train": 256},
        "confirm": {"hotpotqa_train": 256, "musique_train": 512},
    }
    if profiles != expected_profiles:
        raise ContractError("profile counts changed")
    expected_selection = {
        "hotpotqa_screen_namespace": "qwen3-32to14-improved/hotpotqa/train-screen/method-dev-eval-v1",
        "hotpotqa_confirm_namespace": "qwen3-32to14-improved/hotpotqa/train-confirm/method-dev-eval-v1",
        "rank_rule": "ascending_sha256(namespace + NUL + record_id), then source_index",
        "normalized_question_group_rule": "NFKC_then_casefold_then_whitespace_collapse",
        "duplicate_group_policy": "quarantine_all_rows_in_repeated_normalized_question_groups",
        "selection_reads_answers_or_model_outputs": False,
    }
    if config.get("selection") != expected_selection:
        raise ContractError("selection contract changed")
    expected_prompt = (
        "Answer the multi-hop question using only the supplied context. "
        "Return only the shortest answer span and no explanation.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    if config.get("prompts") != {
        "mode": "qwen_chat_no_think",
        "hotpotqa_template": expected_prompt,
        "musique_source": "byte_exact_semantics_from_06_config_and_build_prompt",
        "paragraph_order": "official_order_hotpotqa; official_idx_ascending_musique",
    }:
        raise ContractError("prompt contract changed")
    datasets = config.get("datasets")
    if not isinstance(datasets, Mapping) or tuple(datasets) != DATASETS:
        raise ContractError("dataset ordering/content changed")
    expected_dataset_fields = {
        "hotpotqa_train": (90447, "official_train_v1.1", "method-dev", "_id"),
        "musique_train": (19938, "official_answerable_train_v1.0", "method-dev", "id"),
    }
    for dataset_id, expected in expected_dataset_fields.items():
        value = datasets.get(dataset_id)
        if not isinstance(value, Mapping):
            raise ContractError(f"dataset {dataset_id} is malformed")
        observed = (value.get("rows"), value.get("split"), value.get("tier"), value.get("grouping_key"))
        if observed != expected:
            raise ContractError(f"dataset {dataset_id} split contract changed")
    outputs = config.get("outputs")
    if not isinstance(outputs, Mapping):
        raise ContractError("output config is malformed")
    if Path(str(outputs.get("prepared"))).resolve() != PREPARED_ROOT.resolve():
        raise ContractError("prepared output path changed")
    if Path(str(outputs.get("runs_root"))).resolve() != RUNS_ROOT.resolve():
        raise ContractError("runs output path changed")

    artifact_fields = (
        ("datasets.hotpotqa_train.source", config["datasets"]["hotpotqa_train"]["source"]),
        ("datasets.musique_train.source", config["datasets"]["musique_train"]["source"]),
        ("datasets.musique_train.config", config["datasets"]["musique_train"]["config"]),
        ("datasets.musique_train.registry_manifest", config["datasets"]["musique_train"]["registry_manifest"]),
        ("datasets.musique_train.screen_registry", config["datasets"]["musique_train"]["screen_registry"]),
        ("datasets.musique_train.confirm_registry", config["datasets"]["musique_train"]["confirm_registry"]),
        ("datasets.musique_train.official_dev_lock", config["datasets"]["musique_train"]["official_dev_lock"]),
        ("metrics.hotpotqa_reference", config["metrics"]["hotpotqa_reference"]),
        ("metrics.musique_official", config["metrics"]["musique_official"]),
        ("provenance.musique_prompt_implementation", config.get("provenance", {}).get("musique_prompt_implementation")),
        ("provenance.cross_tier_audit_receipt", config.get("provenance", {}).get("cross_tier_audit_receipt")),
        ("runtime.upstream_receipt", config["runtime"]["upstream_receipt"]),
        ("methods.nvidia_k20.mapper_manifest", config["methods"]["nvidia_k20"]["mapper_manifest"]),
    )
    for label, value in artifact_fields:
        reference = ArtifactRef.parse(value, label=label)
        if verify_inputs:
            reference.verify(label=label)

    expected_confirmation_gate = {
        "rule_id": "screen_completed_then_confirm_v1",
        "required_screen_profile": "screen",
        "required_screen_validation_status": "validated_complete",
        "selection_is_pre_registered_and_outcome_independent": True,
        "outcome_threshold": None,
        "confirm_requires_explicit_validated_screen_output": True,
    }
    if config.get("confirmation_gate") != expected_confirmation_gate:
        raise ContractError("confirmation gate contract changed")

    methods = config.get("methods")
    if not isinstance(methods, Mapping) or tuple(methods) != (
        "native14",
        "nvidia_k20",
        "shrinkage",
        "tail_refresh",
    ):
        raise ContractError("method ordering/content changed")
    if methods["native14"].get("state") != "ready" or methods["nvidia_k20"].get("state") != "ready":
        raise ContractError("native/baseline binding is not ready")
    runtime = config.get("runtime")
    expected_runtime = {
        "source_model": "/home/wby/models/Qwen3-32B",
        "target_model": "/home/wby/models/Qwen3-14B",
        "source_devices": [0, 1, 2, 3, 4],
        "source_layer_counts": [12, 13, 14, 13, 12],
        "target_devices": [5, 6, 7],
        "target_layer_counts": [13, 14, 13],
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "required_cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "global_gpu_lock": "/home/wby/llm-cache/nvidia-kvcache/runs/gpu8_experiment.lock",
    }
    if not isinstance(runtime, Mapping):
        raise ContractError("runtime config is malformed")
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise ContractError(f"runtime {field} changed")
    return {**config, "config_path": str(config_path), "config_sha256": sha256_file(config_path)}


def code_inventory() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in {"artifacts", "runs", "logs"}:
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files[str(relative)] = {
            "nbytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {"files": files, "digest": canonical_sha256(files)}


__all__ = [
    "ArtifactRef",
    "ContractError",
    "DATASETS",
    "DEFAULT_CONFIG",
    "MODES",
    "PARTITIONS",
    "PREPARED_ROOT",
    "PROFILES",
    "ROOT",
    "RUNS_ROOT",
    "artifact_ref",
    "atomic_json",
    "atomic_jsonl",
    "canonical_bytes",
    "canonical_sha256",
    "code_inventory",
    "finite_float",
    "immutable_json",
    "jsonl_bytes",
    "load_config",
    "read_json",
    "read_json_ref",
    "read_jsonl",
    "require_child",
    "sha256_file",
    "token_ids_sha256",
    "write_sidecar",
]
