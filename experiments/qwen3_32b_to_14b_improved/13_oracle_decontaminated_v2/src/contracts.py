"""CPU-only, fail-closed contracts for the decontaminated causal oracle.

This module intentionally imports only the Python standard library.  In
particular, a malformed placeholder, registry, or complete-G0 receipt is
rejected before ``torch`` can be imported or a CUDA API can be called.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import stat
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "config.template.json"
HEX64 = re.compile(r"[0-9a-f]{64}")
PLACEHOLDER_PREFIX = "__FINAL_"
EXPECTED_BORROWED_INTERFACES = {
    "cache_oracle": {
        "path": "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/01_oracle_diagnostics/src/cache_oracle.py",
        "nbytes": 41062,
        "sha256": "c61722344ebe71e3b58609d3db526692085596539ae57561692c8c02f02cea02",
    },
    "gpu_oracle": {
        "path": "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/01_oracle_diagnostics/src/gpu_oracle.py",
        "nbytes": 24800,
        "sha256": "cd89dffbbfa449cd763a644703d003087bd8e411690adaf3cc853015edb90bb0",
    },
    "runtime_harness": {
        "path": "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/01_oracle_diagnostics/src/runtime_harness.py",
        "nbytes": 6438,
        "sha256": "de5c55b0aee52976b1a27666f668245aea7eb7c8e3e4a011c96b8070e93beec4",
    },
    "oracle_diagnostics": {
        "path": "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/01_oracle_diagnostics/src/oracle_diagnostics.py",
        "nbytes": 52357,
        "sha256": "6401d316cdae11d15081151d167f7dab6849ccd728a36db112dab5a6e7fdecdd",
    },
}
EXPECTED_ANSWER_ACCESS_POLICY = {
    "policy_id": "train_method_dev_oracle_metrics_only",
    "allowed": True,
    "allowed_uses": ["gold_first_token_margin", "qa_f1", "qa_em"],
    "forbidden_uses": ["selection", "ranking", "quota", "partition_assignment"],
}


class ContractError(RuntimeError):
    """A frozen input or causal/data-isolation contract was violated."""


def _absolute_unresolved_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if ".." in candidate.parts:
        raise ContractError(f"path contains a non-canonical parent traversal: {candidate}")
    return candidate


def _reject_symlink_chain_before_resolve(
    path: str | Path,
    *,
    label: str,
    allow_missing_tail: bool,
) -> Path:
    """Reject every live or dangling symlink before any call to ``resolve``.

    ``Path.resolve`` erases the evidence that an ArtifactRef or formal config
    traversed a symlink.  Walk the unresolved spelling with ``lstat`` first so
    both a live link and a dangling link fail closed.  Mutable output targets
    may have an absent suffix, but every existing prefix is still checked.
    """

    candidate = _absolute_unresolved_path(path)
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            if allow_missing_tail:
                return candidate
            raise ContractError(
                f"{label} is absent before resolution (possibly a dangling path): {current}"
            ) from exc
        except OSError as exc:
            raise ContractError(
                f"cannot lstat {label} before resolution at {current}: {exc}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise ContractError(
                f"{label} traverses a live or dangling symlink before resolution: {current}"
            )
    return candidate


def require_nonsymlink_file(path: str | Path, *, label: str) -> Path:
    candidate = _reject_symlink_chain_before_resolve(
        path, label=label, allow_missing_tail=False
    )
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:  # The component walk normally provides the detail.
        raise ContractError(f"cannot stat {label}: {candidate}: {exc}") from exc
    if not stat.S_ISREG(mode):
        raise ContractError(f"{label} is not a regular non-symlink file: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ContractError(f"{label} path changed during strict resolution: {candidate}")
    return candidate


def require_nonsymlink_directory(path: str | Path, *, label: str) -> Path:
    candidate = _reject_symlink_chain_before_resolve(
        path, label=label, allow_missing_tail=False
    )
    try:
        mode = candidate.lstat().st_mode
    except OSError as exc:
        raise ContractError(f"cannot stat {label}: {candidate}: {exc}") from exc
    if not stat.S_ISDIR(mode):
        raise ContractError(f"{label} is not a regular non-symlink directory: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ContractError(f"{label} path changed during strict resolution: {candidate}")
    return candidate


def prospective_nonsymlink_path(path: str | Path, *, label: str) -> Path:
    candidate = _reject_symlink_chain_before_resolve(
        path, label=label, allow_missing_tail=True
    )
    resolved = candidate.resolve(strict=False)
    if resolved != candidate:
        raise ContractError(f"{label} path changed during resolution: {candidate}")
    return candidate


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
        raise ContractError(f"value is not canonical JSON: {exc}") from exc
    return payload + (b"\n" if newline else b"")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    checked = require_nonsymlink_file(path, label="SHA-256 input")
    result = hashlib.sha256()
    with checked.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def artifact_ref(path: str | Path) -> dict[str, Any]:
    resolved = require_nonsymlink_file(path, label="artifact")
    return {
        "path": str(resolved),
        "nbytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_json(path: str | Path) -> dict[str, Any]:
    checked = require_nonsymlink_file(path, label="JSON input")
    try:
        with checked.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    checked = require_nonsymlink_file(path, label="JSONL input")
    result: list[dict[str, Any]] = []
    try:
        with checked.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ContractError(f"blank JSONL row at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ContractError(f"JSONL row is not an object at {path}:{line_number}")
                result.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSONL {path}: {exc}") from exc
    return result


def find_placeholders(value: Any, *, prefix: str = "$") -> list[str]:
    result: list[str] = []
    if isinstance(value, str) and value.startswith(PLACEHOLDER_PREFIX):
        result.append(prefix)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            result.extend(find_placeholders(item, prefix=f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(find_placeholders(item, prefix=f"{prefix}[{index}]"))
    return result


def _require_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ContractError(f"{label} is missing keys: {missing}")


def verify_ref(spec: Mapping[str, Any], *, label: str) -> Path:
    if not isinstance(spec, Mapping):
        raise ContractError(f"{label} artifact reference is not an object")
    if set(spec) != {"path", "nbytes", "sha256"}:
        raise ContractError(f"{label} artifact reference must have exactly path/nbytes/sha256")
    if find_placeholders(spec):
        raise ContractError(f"{label} artifact reference still contains a finalization placeholder")
    path_raw, nbytes, digest = spec["path"], spec["nbytes"], spec["sha256"]
    if not isinstance(path_raw, str) or not path_raw or not Path(path_raw).is_absolute():
        raise ContractError(f"{label} path must be a non-empty absolute path")
    if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 0:
        raise ContractError(f"{label} nbytes must be a non-negative integer")
    if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
        raise ContractError(f"{label} SHA-256 is malformed")
    path = require_nonsymlink_file(path_raw, label=label)
    if path.stat().st_size != nbytes:
        raise ContractError(f"{label} byte count changed")
    if sha256_file(path) != digest:
        raise ContractError(f"{label} SHA-256 changed")
    return path


def load_config(path: str | Path = TEMPLATE) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != 1:
        raise ContractError("unsupported oracle config schema")
    if value.get("experiment_id") != "qwen3_32b_to_14b_oracle_decontaminated_v2":
        raise ContractError("unexpected oracle experiment identity")
    if value.get("tier") != "method-dev-train-only":
        raise ContractError("oracle tier is not train-only method-dev")
    return value


def _formal_config_guard(config: Mapping[str, Any], *, profile: str) -> None:
    placeholders = find_placeholders(config)
    if placeholders:
        preview = ", ".join(placeholders[:8])
        raise ContractError(f"formal execution refused: unresolved placeholders at {preview}")
    policy = config.get("execution_policy")
    expected_execution_policy = {
        "gpu_execution_authorized_by_this_config": True,
        "locked_outcomes_must_remain_unopened": True,
        "answers_require_registry_authorization": True,
    }
    if not isinstance(policy, Mapping) or canonical_bytes(policy) != canonical_bytes(
        expected_execution_policy
    ):
        raise ContractError(
            "formal execution policy must exactly authorize this config while retaining locked/answer gates"
        )
    registry = config.get("native_registry")
    expected_registry_keys = {
        "published_marker",
        "registry_manifest",
        "publication_receipt",
        "split_artifacts",
        "consumed_partitions",
        "strata_order",
        "required_answer_access_policy",
    }
    if not isinstance(registry, Mapping) or set(registry) != expected_registry_keys:
        raise ContractError("formal config native-registry policy inventory drifted")
    required_policy = registry.get("required_answer_access_policy") if isinstance(registry, Mapping) else None
    if not isinstance(required_policy, Mapping) or canonical_bytes(
        required_policy
    ) != canonical_bytes(EXPECTED_ANSWER_ACCESS_POLICY):
        raise ContractError("formal config required answer-access policy drifted")
    generation = config.get("generation")
    if generation != {"max_new_tokens": 256, "do_sample": False, "stop": "eos_only"}:
        raise ContractError("generation policy drifted from greedy EOS-only cap 256")
    intervention = config.get("intervention_contract")
    if not isinstance(intervention, Mapping) or (
        intervention.get("catalog"),
        intervention.get("expected_count"),
        intervention.get("target_layers"),
        intervention.get("layer_ids_are_zero_based"),
    ) != (
        "native_kv_2x2_plus_preregistered_layer_repairs_v1",
        11,
        40,
        True,
    ):
        raise ContractError("intervention contract drifted")
    profiles = config.get("profiles")
    if not isinstance(profiles, Mapping) or profile not in profiles:
        raise ContractError(f"unknown profile: {profile}")
    counts = profiles[profile]
    if not isinstance(counts, Mapping) or set(counts) != {"hotpotqa", "musique"}:
        raise ContractError("profile must contain exactly HotpotQA and MuSiQue counts")
    limits = config.get("profile_limits")
    if not isinstance(limits, Mapping):
        raise ContractError("profile limits are absent")
    total = 0
    for dataset in ("hotpotqa", "musique"):
        count = counts[dataset]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ContractError(f"{profile}/{dataset} count must be a positive integer")
        if count > limits.get("maximum_per_dataset"):
            raise ContractError(f"{profile}/{dataset} exceeds the preregistered small-scale cap")
        total += count
    if total > limits.get("maximum_total"):
        raise ContractError(f"{profile} exceeds the preregistered total cap")
    strata = config.get("native_registry", {}).get("strata_order")
    if not isinstance(strata, Mapping) or set(strata) != {"hotpotqa", "musique"}:
        raise ContractError("registry strata order is absent")
    for dataset, order in strata.items():
        if not isinstance(order, list) or not order or not all(
            isinstance(item, str) and item for item in order
        ) or len(set(order)) != len(order):
            raise ContractError(f"{dataset} strata order must be a non-empty unique string list")
    models = config.get("models")
    if not isinstance(models, Mapping) or (
        models.get("source_devices"),
        models.get("source_layer_counts"),
        models.get("target_devices"),
        models.get("target_layer_counts"),
    ) != ([0, 1, 2, 3, 4], [12, 13, 14, 13, 12], [5, 6, 7], [13, 14, 13]):
        raise ContractError("model sharding drifted from preregistered 5+3 placement")
    if sum(models["source_layer_counts"]) != 64 or sum(models["target_layer_counts"]) != 40:
        raise ContractError("model layer totals are invalid")
    if (
        models.get("source_32b"),
        models.get("source_config_sha256"),
        models.get("target_14b"),
        models.get("target_config_sha256"),
        models.get("attention_implementation"),
        models.get("mapper_token_chunk"),
    ) != (
        "/home/wby/models/Qwen3-32B",
        "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb",
        "/home/wby/models/Qwen3-14B",
        "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a",
        "flash_attention_2",
        1024,
    ):
        raise ContractError("model identity or loading policy drifted")
    mapper = config.get("mapper")
    if not isinstance(mapper, Mapping) or (
        mapper.get("manifest"), mapper.get("manifest_nbytes"), mapper.get("manifest_sha256"), mapper.get("deep_hash_scope")
    ) != (
        "/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b/artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01/manifest.json",
        141889,
        "382a1deb6f4fd33c824fe782d956f46bdfb2b876081cb36487616e4de92a8ee2",
        "all_regular_files_under_manifest_directory",
    ):
        raise ContractError("frozen NVIDIA k20 mapper identity drifted")
    prompt = config.get("prompt")
    if not isinstance(prompt, Mapping) or prompt != {
        "schema": "qa_passages_chat_v1",
        "system_text": "Answer the question using the supplied passages. Give only the short answer.",
        "enable_thinking": False,
        "add_generation_prompt": True,
        "maximum_serialized_tokens": 8192,
        "truncation_forbidden": True,
    }:
        raise ContractError("prompt serialization policy drifted")
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping) or runtime != {
        "required_cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "global_gpu_lock": "/home/wby/llm-cache/nvidia-kvcache/runs/gpu8_experiment.lock",
        "idle_memory_limit_mib": 64,
        "idle_consecutive_samples": 12,
        "idle_sample_interval_seconds": 5,
        "idle_timeout_seconds": 86400,
    }:
        raise ContractError("eight-GPU lock or idle-gate policy drifted")
    outputs = Path(str(config.get("outputs_root", "")))
    if not outputs.is_absolute():
        raise ContractError("outputs_root must be absolute")
    checked_outputs = require_nonsymlink_directory(outputs, label="outputs_root")
    if checked_outputs != require_nonsymlink_directory(ROOT / "runs", label="dir13 runs root"):
        raise ContractError("outputs_root must equal the independent dir13/runs root")


def intervention_catalog() -> tuple[dict[str, Any], ...]:
    """The preregistered 2x2 and mapped-base native-layer repair catalog."""

    values = (
        {"id": "native_k_native_v", "default_k": "native", "default_v": "native", "native_kv_layers": []},
        {"id": "mapped_k_native_v", "default_k": "mapped", "default_v": "native", "native_kv_layers": []},
        {"id": "native_k_mapped_v", "default_k": "native", "default_v": "mapped", "native_kv_layers": []},
        {"id": "mapped_k_mapped_v", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": []},
        {"id": "native_kv_layers_0_9", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": list(range(0, 10))},
        {"id": "native_kv_layers_10_19", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": list(range(10, 20))},
        {"id": "native_kv_layers_20_29", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": list(range(20, 30))},
        {"id": "native_kv_layers_30_39", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": list(range(30, 40))},
        {"id": "native_kv_layer_34", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": [34]},
        {"id": "native_kv_layer_35", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": [35]},
        {"id": "native_kv_layers_34_35", "default_k": "mapped", "default_v": "mapped", "native_kv_layers": [34, 35]},
    )
    if len(values) != 11 or len({item["id"] for item in values}) != 11:
        raise ContractError("internal intervention catalog is malformed")
    return values


def _flat_registry_refs(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    registry = config.get("native_registry")
    if not isinstance(registry, Mapping):
        raise ContractError("native registry config is absent")
    splits = registry.get("split_artifacts")
    if not isinstance(splits, Mapping) or set(splits) != {
        "hotpotqa_screen",
        "hotpotqa_confirm",
        "musique_screen",
        "musique_confirm",
    }:
        raise ContractError("native registry must bind exactly four split artifacts")
    return {
        "published": registry["published_marker"],
        "manifest": registry["registry_manifest"],
        "receipt": registry["publication_receipt"],
        **dict(splits),
    }


def _candidate_registry_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    """Project dir13 config to the frozen nested experiment-09/12 binding."""

    refs = _flat_registry_refs(config)
    return {
        "published_marker": dict(refs["published"]),
        "registry_manifest": dict(refs["manifest"]),
        "publication_receipt": dict(refs["receipt"]),
        "split_artifacts": {
            key: dict(refs[key])
            for key in (
                "hotpotqa_screen",
                "hotpotqa_confirm",
                "musique_screen",
                "musique_confirm",
            )
        },
    }


def verify_borrowed_interfaces(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    specs = config.get("borrowed_read_only_interfaces")
    expected = {"cache_oracle", "gpu_oracle", "runtime_harness", "oracle_diagnostics"}
    if not isinstance(specs, Mapping) or set(specs) != expected:
        raise ContractError("read-only borrowed-interface inventory drifted")
    result: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        if specs[name] != EXPECTED_BORROWED_INTERFACES[name]:
            raise ContractError(f"borrowed interface {name} config binding drifted")
        path = verify_ref(specs[name], label=f"borrowed interface {name}")
        result[name] = artifact_ref(path)
    return result


def _validate_g0(config: Mapping[str, Any]) -> dict[str, Any]:
    path = verify_ref(config["complete_g0_v2"], label="complete G0-v2 receipt")
    value = read_json(path)
    required = {
        "status": "complete",
        "pass": True,
        "closes_complete_g0": True,
        "scope": "decontaminated_registry_v2_vs_mapper_fit_and_locked_eval",
        "locked_outcomes_opened": False,
    }
    for key, expected in required.items():
        observed = value.get(key)
        if isinstance(expected, bool):
            matched = observed is expected
        else:
            matched = observed == expected
        if not matched:
            raise ContractError(f"complete G0-v2 receipt does not satisfy {key}={expected!r}")
    comparisons = value.get("comparisons")
    required_names = {
        "methoddev_hotpot_vs_mapper_fit",
        "methoddev_musique_vs_mapper_fit",
        "methoddev_hotpot_vs_locked_eval",
        "methoddev_musique_vs_locked_eval",
    }
    if not isinstance(comparisons, Mapping) or set(comparisons) != required_names:
        raise ContractError("complete G0-v2 comparison inventory is incomplete or has extras")
    for name in sorted(required_names):
        comparison = comparisons[name]
        if (
            not isinstance(comparison, Mapping)
            or set(comparison) != {"flags"}
            or type(comparison["flags"]) is not int
            or comparison["flags"] != 0
        ):
            raise ContractError(
                f"complete G0-v2 comparison {name} must be exactly {{'flags': 0}}"
            )
    binding = value.get("candidate_registry_binding")
    expected_binding = _candidate_registry_binding(config)
    if binding != expected_binding:
        raise ContractError(
            "complete G0-v2 candidate_registry_binding is not the exact nested dir10 binding"
        )
    return {"artifact": artifact_ref(path), "receipt_sha256": sha256_file(path), "status": "complete_pass"}


def _validate_answer_policy(
    receipt: Mapping[str, Any], required_policy: Mapping[str, Any]
) -> None:
    policy = receipt.get("answer_access_policy")
    if not isinstance(policy, Mapping):
        raise ContractError("registry receipt has no explicit answer-access policy")
    if canonical_bytes(required_policy) != canonical_bytes(EXPECTED_ANSWER_ACCESS_POLICY):
        raise ContractError("formal config required answer-access policy drifted")
    if set(policy) != set(EXPECTED_ANSWER_ACCESS_POLICY) or canonical_bytes(
        policy
    ) != canonical_bytes(EXPECTED_ANSWER_ACCESS_POLICY):
        raise ContractError(
            "registry answer-access policy must have the exact preregistered keys and typed values"
        )


def _validate_registry_publication(
    config: Mapping[str, Any], refs: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Path], dict[str, Any]]:
    paths = {key: verify_ref(spec, label=f"native registry {key}") for key, spec in refs.items()}
    published = read_json(paths["published"])
    if published.get("status") != "PUBLISHED" or published.get("pass") is not True:
        raise ContractError("native registry PUBLISHED marker is not a passing publication")
    receipt = read_json(paths["receipt"])
    if receipt.get("status") != "PUBLISHED" or receipt.get("pass") is not True:
        raise ContractError("native registry receipt is not PUBLISHED/pass")
    if receipt.get("tier") != "method-dev" or receipt.get("locked_outcomes_opened") is not False:
        raise ContractError("native registry receipt does not prove method-dev/locked isolation")
    required_policy = config["native_registry"]["required_answer_access_policy"]
    _validate_answer_policy(receipt, required_policy)
    expected_train_sources = {
        dataset: dict(config["train_sources"][dataset]["artifact"])
        for dataset in ("hotpotqa", "musique")
    }
    if receipt.get("train_sources") != expected_train_sources:
        raise ContractError("registry receipt does not exactly bind both raw train-source artifacts")
    projection = receipt.get("projection")
    configured_algorithm = config.get("registry_projection", {}).get("algorithm")
    if not isinstance(projection, Mapping) or projection.get("algorithm") != configured_algorithm:
        raise ContractError("registry projection algorithm is absent or differs from config")
    if projection.get("excludes_answers") is not True:
        raise ContractError("registry projection is not declared answer-free")
    manifest = read_json(paths["manifest"])
    artifacts = manifest.get("artifacts")
    expected_manifest_keys = set(refs) - {"manifest", "published"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_manifest_keys:
        raise ContractError("registry manifest artifact inventory is incomplete or circular")
    for key in sorted(artifacts):
        if artifacts[key] != dict(refs[key]):
            raise ContractError(f"registry manifest binding changed for {key}")
    if published.get("manifest") != dict(refs["manifest"]) or published.get("receipt") != dict(refs["receipt"]):
        raise ContractError("PUBLISHED marker does not bind the pinned manifest and receipt")
    return paths, receipt


FORBIDDEN_OUTCOME_KEYS = {
    "prediction",
    "predictions",
    "generated_text",
    "generated_token_ids",
    "f1",
    "em",
    "accuracy",
    "metric",
    "metrics",
    "score",
    "scores",
    "logit",
    "logits",
    "outcome",
    "outcomes",
}


def _assert_no_outcome_fields(value: Any, *, prefix: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_OUTCOME_KEYS:
                raise ContractError(f"registry row contains forbidden model-outcome field {prefix}.{key}")
            _assert_no_outcome_fields(item, prefix=f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcome_fields(item, prefix=f"{prefix}[{index}]")


def _validated_registry_rows(
    path: Path,
    *,
    dataset: str,
    artifact_key: str,
    strata_order: Sequence[str],
) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    required = {
        "dataset",
        "split",
        "partition",
        "tier",
        "source_index",
        "selection_rank",
        "stratum",
        "source_projection_sha256",
    }
    seen: set[tuple[str, int]] = set()
    seen_sources: set[int] = set()
    stratum_index = {name: index for index, name in enumerate(strata_order)}
    result: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, 1):
        _assert_no_outcome_fields(row, prefix=f"{artifact_key}[{row_number}]")
        _require_keys(row, required, f"{artifact_key} row {row_number}")
        if row["dataset"] != dataset or row["split"] != "train" or row["tier"] != "method-dev":
            raise ContractError(f"{artifact_key} row {row_number} is not {dataset} train method-dev")
        if row["partition"] not in {"screen", artifact_key}:
            raise ContractError(f"{artifact_key} row {row_number} has the wrong partition")
        source_index, rank, stratum = row["source_index"], row["selection_rank"], row["stratum"]
        if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
            raise ContractError(f"{artifact_key} row {row_number} has invalid source_index")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ContractError(f"{artifact_key} row {row_number} has invalid selection_rank")
        if not isinstance(stratum, str) or stratum not in stratum_index:
            raise ContractError(f"{artifact_key} row {row_number} has unknown stratum")
        digest = row["source_projection_sha256"]
        if not isinstance(digest, str) or HEX64.fullmatch(digest) is None:
            raise ContractError(f"{artifact_key} row {row_number} has malformed projection hash")
        local_key = (stratum, rank)
        if local_key in seen or source_index in seen_sources:
            raise ContractError(f"{artifact_key} has duplicate stratum-rank or source index")
        seen.add(local_key)
        seen_sources.add(source_index)
        result.append(dict(row))
    result.sort(key=lambda row: (stratum_index[row["stratum"]], row["selection_rank"], row["source_index"]))
    return result


def _load_source_records(path: Path, format_name: str) -> list[dict[str, Any]]:
    if format_name == "jsonl":
        return read_jsonl(path)
    if format_name != "json":
        raise ContractError(f"unsupported train-source format: {format_name}")
    checked = require_nonsymlink_file(path, label="train JSON source")
    try:
        with checked.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read train JSON {path}: {exc}") from exc
    if isinstance(value, Mapping) and isinstance(value.get("data"), list):
        value = value["data"]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ContractError(f"train JSON {path} must contain an object list")
    return [dict(item) for item in value]


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label} must be a non-empty string")
    return value.strip()


def _adapt_hotpot(record: Mapping[str, Any]) -> dict[str, Any]:
    question = _nonempty_string(record.get("question"), "HotpotQA question")
    context = record.get("context")
    passages: list[dict[str, str]] = []
    if isinstance(context, list):
        for index, item in enumerate(context):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ContractError(f"HotpotQA context item {index} is malformed")
            title = _nonempty_string(item[0], f"HotpotQA title {index}")
            sentences = item[1]
            if not isinstance(sentences, list) or not all(isinstance(x, str) for x in sentences):
                raise ContractError(f"HotpotQA sentences {index} are malformed")
            text = _nonempty_string(" ".join(x.strip() for x in sentences), f"HotpotQA passage {index}")
            passages.append({"title": title, "text": text})
    else:
        raise ContractError("HotpotQA context must be a list")
    answer = record.get("answer")
    aliases_raw = record.get("answer_aliases", [])
    aliases = [answer] if isinstance(answer, str) else list(answer) if isinstance(answer, list) else []
    if isinstance(aliases_raw, list):
        aliases.extend(aliases_raw)
    references = sorted({_nonempty_string(item, "HotpotQA answer") for item in aliases})
    if not references:
        raise ContractError("HotpotQA record has no answer reference")
    source_id = str(record.get("_id", record.get("id", "")))
    return {"question": question, "passages": passages, "references": references, "source_id": source_id}


def _adapt_musique(record: Mapping[str, Any]) -> dict[str, Any]:
    question = _nonempty_string(record.get("question"), "MuSiQue question")
    raw_passages = record.get("paragraphs")
    if not isinstance(raw_passages, list) or not raw_passages:
        raise ContractError("MuSiQue paragraphs must be a non-empty list")
    passages: list[dict[str, str]] = []
    for index, item in enumerate(raw_passages):
        if not isinstance(item, Mapping):
            raise ContractError(f"MuSiQue paragraph {index} is malformed")
        title = _nonempty_string(item.get("title", f"Passage {index + 1}"), f"MuSiQue title {index}")
        text = _nonempty_string(
            item.get("paragraph_text", item.get("text")),
            f"MuSiQue paragraph text {index}",
        )
        passages.append({"title": title, "text": text})
    answer = record.get("answer")
    aliases_raw = record.get("answer_aliases", [])
    aliases = [answer] if isinstance(answer, str) else list(answer) if isinstance(answer, list) else []
    if isinstance(aliases_raw, list):
        aliases.extend(aliases_raw)
    references = sorted({_nonempty_string(item, "MuSiQue answer") for item in aliases})
    if not references:
        raise ContractError("MuSiQue record has no answer reference")
    source_id = str(record.get("id", record.get("_id", "")))
    return {"question": question, "passages": passages, "references": references, "source_id": source_id}


def adapt_source(dataset: str, adapter: str, record: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"hotpotqa": "hotpotqa_train_v1", "musique": "musique_train_v1"}
    if expected.get(dataset) != adapter:
        raise ContractError(f"unexpected {dataset} train adapter: {adapter}")
    return _adapt_hotpot(record) if dataset == "hotpotqa" else _adapt_musique(record)


def _normalize_projection_text(value: str) -> str:
    return " ".join(value.split())


def source_projection(dataset: str, adapted: Mapping[str, Any], algorithm: str) -> dict[str, Any]:
    if algorithm != "canonical_question_passages_v1":
        raise ContractError(f"unsupported registry projection algorithm: {algorithm}")
    passages = adapted.get("passages")
    if not isinstance(passages, list):
        raise ContractError("adapted record has no passage list")
    return {
        "schema_version": 1,
        "algorithm": algorithm,
        "dataset": dataset,
        "question": _normalize_projection_text(str(adapted["question"])),
        "passages": [
            {
                "title": _normalize_projection_text(str(item["title"])),
                "text": _normalize_projection_text(str(item["text"])),
            }
            for item in passages
        ],
    }


@dataclass(frozen=True)
class SelectedSample:
    dataset: str
    artifact_key: str
    source_index: int
    selection_rank: int
    stratum: str
    source_projection_sha256: str
    sample_id: str
    question: str
    passages: tuple[tuple[str, str], ...]
    references: tuple[str, ...]
    input_commitment_sha256: str

    def public_binding(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "artifact_key": self.artifact_key,
            "source_index": self.source_index,
            "selection_rank": self.selection_rank,
            "stratum": self.stratum,
            "source_projection_sha256": self.source_projection_sha256,
            "sample_id": self.sample_id,
            "input_commitment_sha256": self.input_commitment_sha256,
        }


def _materialize_selected(
    config: Mapping[str, Any],
    registry_paths: Mapping[str, Path],
    *,
    profile: str,
) -> tuple[list[SelectedSample], dict[str, dict[str, Any]]]:
    registry = config["native_registry"]
    consumed = registry["consumed_partitions"]
    strata_orders = registry["strata_order"]
    counts = config["profiles"][profile]
    train_sources = config.get("train_sources")
    if not isinstance(train_sources, Mapping) or set(train_sources) != {"hotpotqa", "musique"}:
        raise ContractError("train-source inventory must contain exactly HotpotQA and MuSiQue")
    selected: list[SelectedSample] = []
    source_refs: dict[str, dict[str, Any]] = {}
    algorithm = config["registry_projection"]["algorithm"]
    sampling = config.get("sampling")
    if sampling != {
        "algorithm": "registry_screen_round_robin_strata_v1",
        "uses_answers_or_model_outcomes": False,
    }:
        raise ContractError("diagnostic sampling policy drifted")
    for dataset in ("hotpotqa", "musique"):
        artifact_key = consumed.get(dataset)
        expected_key = f"{dataset}_screen"
        if artifact_key != expected_key:
            raise ContractError(f"oracle may consume only {expected_key}, not {artifact_key}")
        rows = _validated_registry_rows(
            registry_paths[artifact_key],
            dataset=dataset,
            artifact_key=artifact_key,
            strata_order=strata_orders[dataset],
        )
        need = counts[dataset]
        if len(rows) < need:
            raise ContractError(f"{artifact_key} has only {len(rows)} rows, needs {need}")
        buckets = {
            stratum: [row for row in rows if row["stratum"] == stratum]
            for stratum in strata_orders[dataset]
        }
        chosen: list[dict[str, Any]] = []
        depth = 0
        while len(chosen) < need:
            progressed = False
            for stratum in strata_orders[dataset]:
                if depth < len(buckets[stratum]):
                    chosen.append(buckets[stratum][depth])
                    progressed = True
                    if len(chosen) == need:
                        break
            if not progressed:
                raise ContractError(f"{artifact_key} round-robin selection exhausted early")
            depth += 1
        source_spec = train_sources[dataset]
        if not isinstance(source_spec, Mapping) or source_spec.get("declared_split") != "train":
            raise ContractError(f"{dataset} source is not declared train")
        source_path = verify_ref(source_spec["artifact"], label=f"pinned {dataset} train source")
        source_refs[dataset] = artifact_ref(source_path)
        source_rows = _load_source_records(source_path, str(source_spec.get("format")))
        for row in chosen:
            index = row["source_index"]
            if index >= len(source_rows):
                raise ContractError(f"{artifact_key} source_index {index} is out of range")
            adapted = adapt_source(dataset, str(source_spec.get("adapter")), source_rows[index])
            projection = source_projection(dataset, adapted, algorithm)
            digest = canonical_sha256(projection)
            if digest != row["source_projection_sha256"]:
                raise ContractError(f"{artifact_key} source projection changed at index {index}")
            source_id_hash = canonical_sha256({"dataset": dataset, "source_id": adapted["source_id"]})
            if row.get("source_id_sha256") not in (None, source_id_hash):
                raise ContractError(f"{artifact_key} source ID commitment changed at index {index}")
            sample_id = hashlib.sha256(f"{dataset}\0{index}\0{digest}".encode("utf-8")).hexdigest()[:24]
            binding = {
                "dataset": dataset,
                "artifact_key": artifact_key,
                "source_index": index,
                "selection_rank": row["selection_rank"],
                "stratum": row["stratum"],
                "source_projection_sha256": digest,
                "source_id_sha256": source_id_hash,
                "references_sha256": canonical_sha256(adapted["references"]),
            }
            selected.append(
                SelectedSample(
                    dataset=dataset,
                    artifact_key=artifact_key,
                    source_index=index,
                    selection_rank=row["selection_rank"],
                    stratum=row["stratum"],
                    source_projection_sha256=digest,
                    sample_id=sample_id,
                    question=adapted["question"],
                    passages=tuple((item["title"], item["text"]) for item in adapted["passages"]),
                    references=tuple(adapted["references"]),
                    input_commitment_sha256=canonical_sha256(binding),
                )
            )
    return selected, source_refs


def contract_preflight(
    config_path: str | Path,
    *,
    profile: str,
) -> tuple[dict[str, Any], list[SelectedSample]]:
    """Validate all final gates, then and only then open pinned train sources."""

    config = load_config(config_path)
    _formal_config_guard(config, profile=profile)
    borrowed = verify_borrowed_interfaces(config)
    refs = _flat_registry_refs(config)
    g0 = _validate_g0(config)
    registry_paths, receipt = _validate_registry_publication(config, refs)
    selected, source_refs = _materialize_selected(config, registry_paths, profile=profile)
    report = {
        "schema_version": 1,
        "status": "passed_cpu_contract_preflight",
        "gpu_used": False,
        "cuda_api_called": False,
        "profile": profile,
        "samples": len(selected),
        "datasets": {
            dataset: sum(item.dataset == dataset for item in selected)
            for dataset in ("hotpotqa", "musique")
        },
        "interventions": [item["id"] for item in intervention_catalog()],
        "expected_atomic_prompt_results": len(selected),
        "expected_paired_metric_rows": len(selected) * 11,
        "complete_g0_v2": g0,
        "native_registry": {
            "artifacts": {key: dict(spec) for key, spec in refs.items()},
            "receipt_policy_id": receipt["answer_access_policy"]["policy_id"],
            "projection_algorithm": receipt["projection"]["algorithm"],
        },
        "train_sources": source_refs,
        "borrowed_read_only_interfaces": borrowed,
        "selected_bindings": [sample.public_binding() for sample in selected],
        "selected_bindings_sha256": canonical_sha256([sample.public_binding() for sample in selected]),
        "data_access": {
            "train_questions_passages_opened": True,
            "train_answers_opened_after_explicit_policy_gate": True,
            "answers_used_only_for": ["gold_first_token_margin", "qa_f1", "qa_em"],
            "answers_used_for_selection": False,
            "locked_outcomes_opened": False,
        },
    }
    return report, selected


def ensure_inside_output_root(config: Mapping[str, Any], output_dir: str | Path) -> Path:
    root = require_nonsymlink_directory(
        Path(str(config["outputs_root"])), label="outputs_root"
    )
    target = prospective_nonsymlink_path(output_dir, label="run output")
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"output path escapes independent dir13 runs: {target}") from exc
    if target == root:
        raise ContractError("run output must be a named child directory, not the shared runs root")
    return target


def ensure_formal_config_path(config_path: str | Path) -> Path:
    path = require_nonsymlink_file(config_path, label="formal GPU config")
    try:
        path.relative_to(require_nonsymlink_directory(ROOT, label="dir13 root"))
    except ValueError as exc:
        raise ContractError("formal GPU config must live inside independent dir13") from exc
    if path == require_nonsymlink_file(TEMPLATE, label="config template") or not path.name.startswith("config.formal") or path.suffix != ".json":
        raise ContractError("formal GPU config must be a named config.formal*.json, not the template")
    return path


def assert_finite_tree(value: Any, *, prefix: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"non-finite metric at {prefix}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite_tree(item, prefix=f"{prefix}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, prefix=f"{prefix}[{index}]")
        return
    raise ContractError(f"unsupported result value type at {prefix}: {type(value).__name__}")


__all__ = [
    "ContractError",
    "ROOT",
    "EXPECTED_ANSWER_ACCESS_POLICY",
    "SelectedSample",
    "TEMPLATE",
    "adapt_source",
    "artifact_ref",
    "assert_finite_tree",
    "canonical_bytes",
    "canonical_sha256",
    "contract_preflight",
    "ensure_inside_output_root",
    "ensure_formal_config_path",
    "find_placeholders",
    "intervention_catalog",
    "load_config",
    "read_json",
    "read_jsonl",
    "require_nonsymlink_directory",
    "require_nonsymlink_file",
    "prospective_nonsymlink_path",
    "sha256_file",
    "source_projection",
    "verify_borrowed_interfaces",
    "verify_ref",
]
