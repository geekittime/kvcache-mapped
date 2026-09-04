#!/usr/bin/env python3
"""Validate and optionally publish the attempt29 CPU preflight receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Mapping


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "router.json"
RECEIPT_PATH = ROOT / "artifacts" / "preflight_router_attempt29_v1.json"
PROTOCOL_ID = "qwen3_32b_to_14b_prospective_shrinkage_router_v1"
EXPERIMENT_ID = "qwen3_32b_to_14b_prospective_shrinkage_router_attempt29_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.registry import validate_manifest  # noqa: E402
from src.router import Budget, Observation, route_batch  # noqa: E402


class PreflightError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_regular(path: Path) -> Path:
    """Return a canonical absolute regular file, rejecting every symlink."""

    if not path.is_absolute():
        raise PreflightError(f"path must be absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise PreflightError(f"missing path component: {current}") from exc
        if stat.S_ISLNK(mode):
            raise PreflightError(f"symlink rejected: {current}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise PreflightError(f"not a regular file: {path}")
    if path.resolve(strict=True) != path:
        raise PreflightError(f"non-canonical file path: {path}")
    return path


def verify_ref(ref: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(ref, Mapping):
        raise PreflightError("artifact ref must be a mapping")
    if type(ref.get("path")) is not str:
        raise PreflightError("artifact ref path must be a string")
    if type(ref.get("nbytes")) is not int or ref["nbytes"] < 0:
        raise PreflightError("artifact ref nbytes must be a non-negative integer")
    if type(ref.get("sha256")) is not str or len(ref["sha256"]) != 64:
        raise PreflightError("artifact ref sha256 must be 64 hex")
    if any(char not in "0123456789abcdef" for char in ref["sha256"]):
        raise PreflightError("artifact ref sha256 must be lowercase hex")
    path = strict_regular(Path(ref["path"]))
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != ref["nbytes"]:
        raise PreflightError(f"nbytes mismatch for {path}")
    if digest != ref["sha256"]:
        raise PreflightError(f"sha256 mismatch for {path}")
    return {"path": str(path), "nbytes": size, "sha256": digest}


def load_config() -> dict[str, object]:
    strict_regular(CONFIG_PATH)
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PreflightError("config must be a JSON object")
    if value.get("schema_version") != 1:
        raise PreflightError("config schema_version mismatch")
    if value.get("protocol_id") != PROTOCOL_ID:
        raise PreflightError("config protocol_id mismatch")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise PreflightError("config experiment_id mismatch")
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping) or runtime.get("gpu_allowed") is not False:
        raise PreflightError("CPU-only runtime contract missing")
    route = value.get("route")
    if not isinstance(route, Mapping):
        raise PreflightError("route block missing")
    expected_route = {
        "default_arm": "nvidia_k20",
        "escalation_arm": "shrinkage_r1",
        "prompt_enter_tokens": 2800,
        "r1_generated_enter_tokens": 16,
        "gate": "prompt_and_r1_generation",
        "first_pass_required": True,
        "tie_break": "original_batch_order",
    }
    if dict(route) != expected_route:
        raise PreflightError("route block differs from frozen contract")
    budget = value.get("budget")
    if not isinstance(budget, Mapping):
        raise PreflightError("budget block missing")
    if budget.get("canonical_96_request_limit") != 28:
        raise PreflightError("canonical budget mismatch")
    if value.get("arms") != ["native14", "nvidia_k20", "shrinkage_r1"]:
        raise PreflightError("three-arm contract mismatch")
    return value


def implementation_inventory() -> dict[str, object]:
    """Hash every stable implementation/config/document file.

    Generated artifacts and caches are excluded.  The STOP receipt separately
    binds the published preflight receipt and its sidecar.
    """

    items: list[dict[str, object]] = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not relative.parts or relative.parts[0] == "artifacts":
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise PreflightError(f"implementation symlink rejected: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise PreflightError(f"non-regular implementation entry: {relative}")
        items.append(
            {
                "path": relative.as_posix(),
                "nbytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    digest = hashlib.sha256(canonical_bytes(items)).hexdigest()
    return {"count": len(items), "digest": digest, "files": items}


def _synthetic_manifest(tokenizer: Mapping[str, object]) -> dict[str, object]:
    splits: dict[str, list[dict[str, str]]] = {"calibration": [], "sealed_eval": []}
    counts = {
        "calibration": {"hotpotqa_train": 24, "musique_train": 24},
        "sealed_eval": {"hotpotqa_train": 48, "musique_train": 48},
    }
    for split, dataset_counts in counts.items():
        for dataset, count in dataset_counts.items():
            for index in range(count):
                ident = f"synthetic-{split}-{dataset}-{index}"
                splits[split].append(
                    {
                        "sample_id": ident,
                        "dataset_id": dataset,
                        "source_record_sha256": hashlib.sha256(ident.encode()).hexdigest(),
                    }
                )
    return {
        "schema_version": 1,
        "registry_id": "synthetic-contract-check-only",
        "tokenizer": dict(tokenizer),
        "labels_hidden_from_router": True,
        "splits": splits,
        "dataset_counts": counts,
    }


def _assert_cpu_only_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise PreflightError("CUDA_VISIBLE_DEVICES must be the empty string")
    for fd_path in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue
        if target.startswith("/dev/nvidia"):
            raise PreflightError(f"GPU device descriptor is open: {target}")
    forbidden_modules = ("torch", "tensorflow", "jax", "cupy")
    loaded = [name for name in forbidden_modules if name in sys.modules]
    if loaded:
        raise PreflightError(f"model/GPU modules loaded: {loaded}")


def run_preflight() -> dict[str, object]:
    _assert_cpu_only_environment()
    config = load_config()
    tokenizer = config.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise PreflightError("tokenizer block missing")
    tokenizer_ref = {
        "path": tokenizer.get("path"),
        "nbytes": strict_regular(Path(str(tokenizer.get("path")))).stat().st_size,
        "sha256": tokenizer.get("sha256"),
    }
    verified_tokenizer = verify_ref(tokenizer_ref)
    refs = config.get("immutable_hypothesis_refs")
    if not isinstance(refs, list) or len(refs) != 5:
        raise PreflightError("exactly five immutable hypothesis refs are required")
    verified_refs = []
    seen_names: set[str] = set()
    for ref in refs:
        if not isinstance(ref, Mapping) or type(ref.get("name")) is not str:
            raise PreflightError("hypothesis ref name missing")
        if ref["name"] in seen_names:
            raise PreflightError("duplicate hypothesis ref name")
        seen_names.add(ref["name"])
        checked = verify_ref(ref)
        checked["name"] = ref["name"]
        verified_refs.append(checked)

    # Route contract smoke: single signals stay on NVIDIA; the dual signal is
    # eligible but only one row may consume a one-unit budget.
    observations = (
        Observation(2799, 15),
        Observation(2800, 15),
        Observation(2799, 16),
        Observation(2800, 16),
        Observation(4000, 256),
    )
    decisions, final_budget = route_batch(observations, budget=Budget(limit=1))
    actions = [decision.action for decision in decisions]
    if actions != ["nvidia_k20", "nvidia_k20", "nvidia_k20", "shrinkage_r1", "nvidia_k20"]:
        raise PreflightError("dual-gate smoke vector mismatch")
    if decisions[-1].reason != "no_go:budget_exhausted" or final_budget.used != 1:
        raise PreflightError("budget-overflow smoke mismatch")

    fresh = config.get("fresh_registry")
    if not isinstance(fresh, Mapping):
        raise PreflightError("fresh_registry block missing")
    validated = validate_manifest(
        _synthetic_manifest(tokenizer),
        expected_tokenizer_sha256=str(tokenizer["sha256"]),
        min_calibration_per_dataset=int(fresh["calibration_min_per_dataset"]),
        min_sealed_per_dataset=int(fresh["sealed_eval_min_per_dataset"]),
    )
    if len(validated.records) != 144:
        raise PreflightError("synthetic registry size mismatch")

    return {
        "schema_version": 1,
        "status": "passed_cpu_preflight",
        "protocol_id": PROTOCOL_ID,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": verify_ref(
            {
                "path": str(CONFIG_PATH),
                "nbytes": CONFIG_PATH.stat().st_size,
                "sha256": sha256_file(CONFIG_PATH),
            }
        ),
        "implementation_inventory": implementation_inventory(),
        "tokenizer": verified_tokenizer,
        "immutable_hypothesis_refs": verified_refs,
        "contract_smoke": {
            "actions": actions,
            "final_budget_used": final_budget.used,
            "fresh_registry_records": len(validated.records),
        },
        "integrity": {
            "gpu_used": False,
            "model_runtime_imported": False,
            "old03_and_attempt28_are_posthoc_only": True,
            "fresh_registry_opened": False,
            "quality_or_reference_inputs_read": False,
        },
        "claim_boundary": config["claim_boundary"],
    }


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise PreflightError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def publish_receipt(receipt: Mapping[str, object]) -> dict[str, object]:
    payload = canonical_bytes(receipt)
    atomic_write(RECEIPT_PATH, payload)
    reference = {
        "path": str(RECEIPT_PATH),
        "nbytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    sidecar_path = RECEIPT_PATH.with_name(RECEIPT_PATH.name + ".sha256.json")
    atomic_write(sidecar_path, canonical_bytes(reference))
    return reference


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    receipt = run_preflight()
    output: Mapping[str, object]
    if args.write_receipt:
        output = {"status": "published", "receipt": publish_receipt(receipt)}
    else:
        output = receipt
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

