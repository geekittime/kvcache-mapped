#!/usr/bin/env python3
"""Read-only, CPU-only validation for the improved-method research protocol."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


PROTOCOL_DIR = Path(__file__).resolve().parent
REQUIRED_DOCUMENTS = {
    "README.md",
    "EVIDENCE.md",
    "PROTOCOL.md",
    "protocol.json",
    "data_policy.json",
    "baseline_refs.json",
}
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "argparse",
    "ast",
    "copy",
    "hashlib",
    "importlib",
    "json",
    "math",
    "pathlib",
    "sys",
    "tempfile",
    "typing",
    "unittest",
}
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer!r}")
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(part)]
        elif isinstance(current, dict):
            current = current[part]
        else:
            raise TypeError(f"cannot descend through {type(current).__name__}")
    return current


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def repository_root(reference_manifest: dict[str, Any], protocol_dir: Path) -> Path:
    relative = reference_manifest["repository_root_relative_to_this_file"]
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("repository root anchor must be relative")
    return (protocol_dir / candidate).resolve()


def verify_reference(
    reference: dict[str, Any], repo_root: Path, baseline_root: Path
) -> list[str]:
    errors: list[str] = []
    ref_id = reference.get("id", "<missing-id>")
    raw_path = Path(reference.get("path", ""))
    if not raw_path.parts or raw_path.is_absolute() or ".." in raw_path.parts:
        return [f"{ref_id}: path must be a nonempty repository-relative path"]

    unresolved = repo_root / raw_path
    resolved = unresolved.resolve()
    if not is_within(resolved, baseline_root):
        errors.append(f"{ref_id}: reference escapes read-only baseline root: {resolved}")
        return errors
    if unresolved.is_symlink():
        errors.append(f"{ref_id}: direct symlink references are forbidden")
    if not resolved.is_file():
        errors.append(f"{ref_id}: missing regular file: {resolved}")
        return errors

    actual_digest = sha256_file(resolved)
    expected_digest = reference.get("sha256")
    if actual_digest != expected_digest:
        errors.append(
            f"{ref_id}: SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
        )

    if reference.get("media_type") != "application/json":
        errors.append(f"{ref_id}: only JSON control artifacts may be referenced")
        return errors
    try:
        document = load_json(resolved)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{ref_id}: cannot parse JSON: {exc}")
        return errors

    for assertion in reference.get("assertions", []):
        pointer = assertion.get("pointer")
        try:
            actual = json_pointer(document, pointer)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            errors.append(f"{ref_id}{pointer}: unresolved assertion: {exc}")
            continue
        if "equals" in assertion and actual != assertion["equals"]:
            errors.append(
                f"{ref_id}{pointer}: expected {assertion['equals']!r}, got {actual!r}"
            )
        if "approx" in assertion:
            expected = assertion["approx"]
            atol = assertion.get("atol", 0.0)
            if (
                isinstance(actual, bool)
                or not isinstance(actual, (int, float))
                or not math.isfinite(float(actual))
                or not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol)
            ):
                errors.append(
                    f"{ref_id}{pointer}: expected {expected!r}±{atol}, got {actual!r}"
                )
    return errors


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def validate_protocol(protocol_dir: Path = PROTOCOL_DIR) -> list[str]:
    protocol_dir = protocol_dir.resolve()
    errors: list[str] = []

    missing = sorted(name for name in REQUIRED_DOCUMENTS if not (protocol_dir / name).is_file())
    if missing:
        errors.append(f"missing required documents: {missing}")
        return errors

    try:
        protocol = load_json(protocol_dir / "protocol.json")
        data_policy = load_json(protocol_dir / "data_policy.json")
        refs = load_json(protocol_dir / "baseline_refs.json")
    except (OSError, json.JSONDecodeError) as exc:
        return [f"protocol JSON parse failure: {exc}"]

    for name, document in (
        ("protocol.json", protocol),
        ("data_policy.json", data_policy),
        ("baseline_refs.json", refs),
    ):
        if document.get("schema_version") != 1:
            errors.append(f"{name}: schema_version must be 1")

    try:
        repo_root = repository_root(refs, protocol_dir)
    except (KeyError, ValueError) as exc:
        errors.append(f"baseline_refs.json: invalid repository anchor: {exc}")
        return errors

    expected_repo_root = protocol_dir.parents[2]
    if repo_root != expected_repo_root:
        errors.append(
            f"repository anchor resolves to {repo_root}, expected {expected_repo_root}"
        )

    baseline_relative = Path(refs.get("baseline_root", ""))
    if baseline_relative.is_absolute() or ".." in baseline_relative.parts:
        errors.append("baseline_root must be repository-relative without '..'")
        return errors
    baseline_root = (repo_root / baseline_relative).resolve()
    expected_baseline = (repo_root / protocol["isolation"]["baseline_directory"]).resolve()
    if baseline_root != expected_baseline:
        errors.append("baseline root disagrees between manifests")
    if refs.get("write_policy") != "forbidden":
        errors.append("baseline write policy must be forbidden")

    references = refs.get("references", [])
    reference_ids = [reference.get("id") for reference in references]
    if len(references) < 8:
        errors.append("at least eight baseline control references are required")
    if len(reference_ids) != len(set(reference_ids)) or None in reference_ids:
        errors.append("baseline reference IDs must be present and unique")
    for reference in references:
        errors.extend(verify_reference(reference, repo_root, baseline_root))

    tiers = data_policy.get("tiers", [])
    tier_ids = [tier.get("id") for tier in tiers]
    if tier_ids != ["mapper-fit", "method-dev", "locked-test"]:
        errors.append("data tiers must be ordered mapper-fit, method-dev, locked-test")
    namespaces = [tier.get("namespace") for tier in tiers]
    if len(namespaces) != len(set(namespaces)) or None in namespaces:
        errors.append("data tier namespaces must be present and unique")
    if not data_policy.get("token_level_random_split_forbidden"):
        errors.append("token-level random splitting must be forbidden")
    if data_policy.get("cross_tier_exact_id_overlap_allowed") is not False:
        errors.append("cross-tier exact ID overlap must be forbidden")
    exposure = data_policy.get("historical_exposure", {})
    if exposure.get("aggregate_results_previously_observed") is not True:
        errors.append("historical test-result exposure must be disclosed")
    if "not historically blind" not in exposure.get("required_report_language", ""):
        errors.append("historical exposure language is incomplete")
    if tiers and tiers[-1].get("state") != "sealed_pending_candidate_freeze":
        errors.append("locked-test must remain sealed pending candidate freeze")

    if protocol.get("status") != "preregistered_before_candidate_gpu_runs":
        errors.append("protocol status must precede candidate GPU runs")
    if protocol.get("direction") != "qwen3_32b_to_14b":
        errors.append("model direction mismatch")
    isolation = protocol.get("isolation", {})
    if isolation.get("baseline_access") != "read_only_content_addressed":
        errors.append("baseline access must be read-only and content-addressed")
    if isolation.get("mixed_candidate_artifacts_allowed") is not False:
        errors.append("mixed candidate artifacts must be forbidden")
    forbidden_roots = isolation.get("forbidden_write_roots", [])
    if "experiments/qwen3_32b_to_14b" not in forbidden_roots:
        errors.append("existing experiment tree must be a forbidden write root")

    system_classes = protocol.get("system_classes", [])
    replay_windows = [item.get("target_replay_tokens") for item in system_classes]
    if replay_windows != [1, 8, 32]:
        errors.append("system classes must be separately declared as R=1,8,32")
    if system_classes and any(
        item.get("claim_class") == system_classes[0].get("claim_class")
        for item in system_classes[1:]
    ):
        errors.append("pure and hybrid modes must use distinct claim classes")

    candidates = protocol.get("candidates", [])
    orders = [candidate.get("order") for candidate in candidates]
    if orders != list(range(len(candidates))):
        errors.append("candidate ablation order must be contiguous and immutable")
    candidate_ids = [candidate.get("id") for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or None in candidate_ids:
        errors.append("candidate IDs must be present and unique")
    for candidate in candidates:
        has_locked = "locked-test" in candidate.get("data_access", [])
        if has_locked != bool(candidate.get("locked_test_allowed")):
            errors.append(f"{candidate.get('id')}: locked-test access flag mismatch")
        if candidate.get("order", -1) < 9 and has_locked:
            errors.append(f"{candidate.get('id')}: premature locked-test access")

    if protocol.get("primary_method", {}).get("status") != "unvalidated_hypothesis":
        errors.append("SARR-KV must remain labeled an unvalidated hypothesis")
    gates = {gate.get("id"): gate for gate in protocol.get("gates", [])}
    if set(gates) != {
        "G0_integrity",
        "G1_generalization",
        "G2_functional",
        "G3_nomination",
        "G4_confirmatory_claim",
    }:
        errors.append("GO/NO-GO gate set is incomplete")
    claim_policy = protocol.get("claim_policy", {})
    if claim_policy.get("pure_and_hybrid_results_may_be_merged") is not False:
        errors.append("pure and hybrid results must not be merged")
    if claim_policy.get("paper_bit_exact_claim_allowed") is not False:
        errors.append("paper bit-exact claim must be forbidden")

    for path in protocol_dir.rglob("*"):
        if path.is_symlink():
            errors.append(f"symlink forbidden in protocol directory: {path}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            errors.append(f"model/data artifact forbidden in protocol directory: {path}")
        if path.is_file() and path.suffix == ".py":
            try:
                extra_imports = imported_roots(path) - ALLOWED_IMPORT_ROOTS
            except SyntaxError as exc:
                errors.append(f"Python syntax error in {path}: {exc}")
                continue
            if extra_imports:
                errors.append(f"non-stdlib imports in {path}: {sorted(extra_imports)}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit a JSON receipt to stdout")
    args = parser.parse_args(argv)
    errors = validate_protocol()
    result = {
        "status": "passed" if not errors else "failed",
        "protocol_directory": str(PROTOCOL_DIR),
        "gpu_or_model_operations": 0,
        "errors": errors,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        print("protocol validation failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print("protocol validation passed (read-only, CPU-only)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
