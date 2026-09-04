#!/usr/bin/env python3
"""Strict, CPU-only validator and paired analyzer for the frozen tail pilot."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROOT = ANALYSIS_ROOT.parent
RUN_ROOT = CANDIDATE_ROOT / "runs/method_dev_pilot_r1_r8_r32_v2"
PREPARED = CANDIDATE_ROOT / "artifacts/prepared_method_dev_v2"
TOKENIZER_ROOT = Path("/home/wby/models/Qwen3-14B")
MODES = ("target_standalone", "mapped_r1", "tail_refresh_r8", "tail_refresh_r32")
DATASETS = ("hotpotqa_train", "musique_train")
R_BY_MODE = {"mapped_r1": 1, "tail_refresh_r8": 8, "tail_refresh_r32": 32}
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_FINGERPRINT = "4c113cec99c56e9e48d47e79ad9933d34720ebb4b8851e8e1efce1e96b57e723"
EXPECTED_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
EXPECTED_PREPARED_MANIFEST_SHA256 = "dea34a19ca965f58d8d5f3ffb4c481b22d2420d71f30254287d8247e25062788"
EXPECTED_SELECTED_SHA256 = "aa8b94367d57b088ba8c12a13068e9c82c6b95d18fc1579b5ccf66270c8aed53"

for path in (CANDIDATE_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.qa_metrics import qa_metrics  # noqa: E402


class ValidationError(ValueError):
    """A frozen result or its provenance is incomplete or inconsistent."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_selected() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (PREPARED / "selected_samples.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if "pilot" in value["phase_membership"]:
                    rows.append(value)
    if len(rows) != 96 or Counter(row["dataset_id"] for row in rows) != Counter({dataset: 48 for dataset in DATASETS}):
        raise ValidationError("pilot selection is not exactly 48 HotpotQA + 48 MuSiQue samples")
    identities = {(row["dataset_id"], row["sample_id"]) for row in rows}
    if len(identities) != 96:
        raise ValidationError("pilot selection contains duplicate identities")
    for row in rows:
        ids = row.get("prompt_token_ids")
        if not isinstance(ids, list) or any(type(token) is not int for token in ids):
            raise ValidationError("prepared prompt token IDs are malformed")
        if canonical_sha256(ids) != row.get("prompt_token_ids_sha256"):
            raise ValidationError("prepared prompt token ID hash differs")
        if len(ids) != row.get("prompt_token_count") or row.get("prompt_truncated") is not False:
            raise ValidationError("prepared prompt count/truncation contract differs")
    return rows


def result_identity(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{row['dataset_id']}\0{row['sample_id']}".encode()).hexdigest()[:20]


def expected_result_path(row: Mapping[str, Any], mode: str) -> Path:
    return RUN_ROOT / "results" / str(row["dataset_id"]) / mode / f"{result_identity(row)}.json"


def _equal_number(observed: Any, expected: Any, label: str, *, atol: float = 1e-12) -> None:
    if isinstance(expected, float):
        if not isinstance(observed, (float, int)) or not math.isclose(float(observed), expected, abs_tol=atol, rel_tol=0.0):
            raise ValidationError(f"{label} differs: {observed!r} != {expected!r}")
    elif observed != expected:
        raise ValidationError(f"{label} differs: {observed!r} != {expected!r}")


def validate_tail_audit(audit: Any, prompt_tokens: int, mode: str) -> None:
    if mode == "target_standalone":
        if audit is not None:
            raise ValidationError("standalone result unexpectedly has a tail audit")
        return
    if not isinstance(audit, Mapping):
        raise ValidationError(f"{mode} tail audit is missing")
    r_value = R_BY_MODE[mode]
    prefix = prompt_tokens - 1
    mapped = prompt_tokens - r_value
    native = r_value - 1
    exact = {
        "full_prompt_tokens": prompt_tokens,
        "builder_prefix_tokens": prefix,
        "requested_total_native_prompt_tokens": r_value,
        "builder_native_tail_tokens": native,
        "engine_boundary_tokens": 1,
        "engine_boundary_position_id": prefix,
        "mapped_prefix_tokens": mapped,
        "mapped_token_slice": [0, mapped],
        "native_tail_token_slice": [mapped, prefix],
        "cache_length_before_refresh": mapped,
        "cache_length_after_refresh": prefix,
        "native_tail_position_ids": list(range(mapped, prefix)),
        "target_forward_calls": 0 if r_value == 1 else 1,
        "target_forward_query_tokens": native,
        "r1_returned_underlying_cache_object_directly": r_value == 1,
    }
    for key, expected in exact.items():
        if audit.get(key) != expected:
            raise ValidationError(f"{mode} tail audit {key} differs")
    if r_value == 1:
        for forbidden in ("attention_mask_shape", "cache_position", "position_ids"):
            if forbidden in audit:
                raise ValidationError(f"R=1 zero-op unexpectedly records {forbidden}")
    else:
        for key in ("cache_position", "position_ids"):
            if audit.get(key) != list(range(mapped, prefix)):
                raise ValidationError(f"{mode} {key} is not the exact native tail")
        if audit.get("attention_mask_shape") != [1, prefix]:
            raise ValidationError(f"{mode} attention mask shape differs")


def validate_generation(value: Mapping[str, Any], tokenizer: Any) -> None:
    generation = value.get("generation")
    if not isinstance(generation, Mapping):
        raise ValidationError("generation record is missing")
    ids = generation.get("generated_token_ids")
    if not isinstance(ids, list) or any(type(token) is not int for token in ids):
        raise ValidationError("generated token IDs are malformed")
    if any(token < 0 or token >= len(tokenizer) for token in ids):
        raise ValidationError("generated token ID is outside tokenizer vocabulary")
    if generation.get("generated_tokens") != len(ids):
        raise ValidationError("generated token count differs from ID count")
    if generation.get("generated_token_ids_sha256") != canonical_sha256(ids):
        raise ValidationError("generated token ID hash differs")
    if generation.get("requested_max_tokens") != 256 or len(ids) > 256:
        raise ValidationError("generation cap contract differs")
    reason = generation.get("stop_reason")
    cap_hit = generation.get("cap_hit")
    if reason not in {"eos", "stop_sequence", "max_tokens"} or type(cap_hit) is not bool:
        raise ValidationError("invalid stop reason/cap flag")
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    prediction = value.get("prediction")
    if not isinstance(prediction, str):
        raise ValidationError("prediction is not text")
    stop_sequences = ("\nQuestion:", "\nContext:")
    positions = [decoded.find(stop) for stop in stop_sequences if stop in decoded]
    stop_at = min(positions) if positions else -1
    if reason == "stop_sequence":
        if stop_at < 0 or prediction != decoded[:stop_at] or cap_hit:
            raise ValidationError("stop-sequence decode contract differs")
    else:
        if prediction != decoded:
            raise ValidationError("decoded generated IDs differ from prediction")
        if stop_at >= 0:
            raise ValidationError("unreported stop sequence appears in decoded output")
    if reason == "max_tokens":
        if len(ids) != 256 or cap_hit is not True:
            raise ValidationError("max-token termination contract differs")
    elif cap_hit is not False or len(ids) >= 256:
        raise ValidationError("non-cap termination has inconsistent cap state")


def validate_result(value: Mapping[str, Any], row: Mapping[str, Any], mode: str, tokenizer: Any, target_hash: str) -> None:
    exact = {
        "schema_version": 1,
        "status": "complete",
        "request_fingerprint": EXPECTED_FINGERPRINT,
        "phase": "pilot",
        "dataset_id": row["dataset_id"],
        "sample_id": row["sample_id"],
        "stratum": row["stratum"],
        "source_record_sha256": row["source_record_sha256"],
        "mode": mode,
        "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        "prompt_token_count": row["prompt_token_count"],
        "prompt_truncated": False,
        "references": row["references"],
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            raise ValidationError(f"{row['sample_id']}/{mode}: {key} differs")
    binding = value.get("target_checkpoint_binding")
    if mode == "target_standalone":
        if binding is not None:
            raise ValidationError("standalone target binding must be null")
    elif binding != target_hash:
        raise ValidationError(f"{row['sample_id']}/{mode}: target checkpoint binding differs")
    validate_generation(value, tokenizer)
    recomputed = qa_metrics(value["prediction"], row["references"])
    observed = value.get("metrics")
    if not isinstance(observed, Mapping) or set(observed) != set(recomputed):
        raise ValidationError("QA metric schema differs")
    for key, expected in recomputed.items():
        _equal_number(observed[key], expected, f"{row['sample_id']}/{mode}/{key}")
    validate_tail_audit(value.get("tail_refresh_audit"), int(row["prompt_token_count"]), mode)
    systems = value.get("systems")
    if not isinstance(systems, Mapping) or not isinstance(systems.get("wall_seconds"), (float, int)) or systems["wall_seconds"] <= 0:
        raise ValidationError("wall-time record is invalid")


def percentile_ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _label_seed(label: str) -> int:
    suffix = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    return (BOOTSTRAP_SEED + suffix) % (2**63 - 1)


def paired_bootstrap_delta(candidate: Sequence[float], baseline: Sequence[float], label: str) -> dict[str, Any]:
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValidationError("paired bootstrap inputs differ")
    rng = np.random.default_rng(_label_seed(label))
    indices = rng.integers(0, len(a), size=(BOOTSTRAP_REPLICATES, len(a)))
    draws = (a[indices] - b[indices]).mean(axis=1)
    return {
        "estimate": float((a - b).mean()),
        "ci95": percentile_ci(draws),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "paired_sample_id_cluster",
    }


def gap_closure_bootstrap(candidate: Sequence[float], baseline: Sequence[float], target: Sequence[float], label: str) -> dict[str, Any] | None:
    a, b, t = (np.asarray(values, dtype=np.float64) for values in (candidate, baseline, target))
    denominator = float((t - b).mean())
    if denominator <= 0:
        return None
    rng = np.random.default_rng(_label_seed(label))
    indices = rng.integers(0, len(a), size=(BOOTSTRAP_REPLICATES, len(a)))
    numerators = (a[indices] - b[indices]).mean(axis=1)
    denominators = (t[indices] - b[indices]).mean(axis=1)
    valid = denominators > 0
    draws = numerators[valid] / denominators[valid]
    return {
        "estimate": float((a - b).mean() / denominator),
        "ci95": percentile_ci(draws),
        "valid_replicates": int(valid.sum()),
        "discarded_nonpositive_denominator_replicates": int((~valid).sum()),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "definition": "(candidate-R1)/(native14-R1); defined only when aggregate native14>R1",
    }


def mean(values: Iterable[float]) -> float:
    sequence = list(values)
    return float(sum(sequence) / len(sequence))


def summarize_mode(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    generations = [value["generation"] for value in values]
    walls = [float(value["systems"]["wall_seconds"]) for value in values]
    return {
        "count": len(values),
        "relaxed_qa_f1": mean(float(value["metrics"]["relaxed_qa_f1"]) for value in values),
        "relaxed_qa_em": mean(float(value["metrics"]["relaxed_qa_exact_match"]) for value in values),
        "raw_qa_f1": mean(float(value["metrics"]["qa_f1"]) for value in values),
        "raw_qa_em": mean(float(value["metrics"]["qa_exact_match"]) for value in values),
        "gold_containment": mean(float(value["metrics"]["normalized_gold_containment"]) for value in values),
        "mean_generated_tokens": mean(int(value["generated_tokens"]) for value in generations),
        "median_generated_tokens": float(statistics.median(int(value["generated_tokens"]) for value in generations)),
        "eos_rate": mean(value["stop_reason"] == "eos" for value in generations),
        "stop_sequence_rate": mean(value["stop_reason"] == "stop_sequence" for value in generations),
        "cap_rate": mean(bool(value["cap_hit"]) for value in generations),
        "mean_wall_seconds": mean(walls),
        "median_wall_seconds": float(statistics.median(walls)),
    }


def transitions(reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ref = [bool(value["metrics"]["relaxed_qa_exact_match"]) for value in reference]
    cand = [bool(value["metrics"]["relaxed_qa_exact_match"]) for value in candidate]
    return {
        "correct_to_wrong": sum(a and not b for a, b in zip(ref, cand)),
        "wrong_to_correct": sum(not a and b for a, b in zip(ref, cand)),
        "correct_to_correct": sum(a and b for a, b in zip(ref, cand)),
        "wrong_to_wrong": sum(not a and not b for a, b in zip(ref, cand)),
    }


def stratum_summary(rows: Sequence[Mapping[str, Any]], paired: Mapping[tuple[str, str], Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    strata = sorted({str(row["stratum"]) for row in rows if row["dataset_id"] == dataset})
    for stratum in strata:
        selected = [row for row in rows if row["dataset_id"] == dataset and row["stratum"] == stratum]
        output[stratum] = {
            mode: summarize_mode([paired[(str(row["sample_id"]), mode)] for row in selected])
            for mode in MODES
        }
    return output


def analyze(rows: Sequence[Mapping[str, Any]], paired: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "validated_complete",
        "claim_boundary": "method-dev pilot only; no locked-test or paper-superiority claim",
        "bootstrap": {"seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
        "datasets": {},
        "strata": {},
    }
    sample_rows: list[dict[str, Any]] = []
    strict_dataset_pass: dict[str, dict[str, bool]] = defaultdict(dict)
    for dataset in DATASETS:
        selected = [row for row in rows if row["dataset_id"] == dataset]
        by_mode = {mode: [paired[(str(row["sample_id"]), mode)] for row in selected] for mode in MODES}
        summaries = {mode: summarize_mode(values) for mode, values in by_mode.items()}
        comparisons: dict[str, Any] = {}
        for mode in ("tail_refresh_r8", "tail_refresh_r32"):
            mode_f1 = [float(v["metrics"]["relaxed_qa_f1"]) for v in by_mode[mode]]
            r1_f1 = [float(v["metrics"]["relaxed_qa_f1"]) for v in by_mode["mapped_r1"]]
            target_f1 = [float(v["metrics"]["relaxed_qa_f1"]) for v in by_mode["target_standalone"]]
            mode_em = [float(v["metrics"]["relaxed_qa_exact_match"]) for v in by_mode[mode]]
            r1_em = [float(v["metrics"]["relaxed_qa_exact_match"]) for v in by_mode["mapped_r1"]]
            cap_ok = summaries[mode]["cap_rate"] <= summaries["mapped_r1"]["cap_rate"]
            termination_ok = cap_ok and summaries[mode]["eos_rate"] + summaries[mode]["stop_sequence_rate"] >= summaries["mapped_r1"]["eos_rate"] + summaries["mapped_r1"]["stop_sequence_rate"]
            improved = summaries[mode]["relaxed_qa_f1"] > summaries["mapped_r1"]["relaxed_qa_f1"]
            strict_dataset_pass[mode][dataset] = improved and termination_ok
            comparisons[mode] = {
                "f1_delta_vs_r1": paired_bootstrap_delta(mode_f1, r1_f1, f"{dataset}/{mode}/f1"),
                "em_delta_vs_r1": paired_bootstrap_delta(mode_em, r1_em, f"{dataset}/{mode}/em"),
                "gap_closure": gap_closure_bootstrap(mode_f1, r1_f1, target_f1, f"{dataset}/{mode}/gap"),
                "r1_em_transitions": transitions(by_mode["mapped_r1"], by_mode[mode]),
                "native_em_transitions": transitions(by_mode["target_standalone"], by_mode[mode]),
                "termination_nonworse": termination_ok,
                "pilot_dataset_promising": strict_dataset_pass[mode][dataset],
            }
        consistency = {
            "exact_prediction_agreement": sum(a["prediction"] == b["prediction"] for a, b in zip(by_mode["tail_refresh_r8"], by_mode["tail_refresh_r32"])),
            "exact_generated_token_id_agreement": sum(a["generation"]["generated_token_ids"] == b["generation"]["generated_token_ids"] for a, b in zip(by_mode["tail_refresh_r8"], by_mode["tail_refresh_r32"])),
            "relaxed_f1_agreement": sum(math.isclose(float(a["metrics"]["relaxed_qa_f1"]), float(b["metrics"]["relaxed_qa_f1"]), abs_tol=1e-12) for a, b in zip(by_mode["tail_refresh_r8"], by_mode["tail_refresh_r32"])),
            "relaxed_em_agreement": sum(a["metrics"]["relaxed_qa_exact_match"] == b["metrics"]["relaxed_qa_exact_match"] for a, b in zip(by_mode["tail_refresh_r8"], by_mode["tail_refresh_r32"])),
            "count": len(selected),
        }
        report["datasets"][dataset] = {"modes": summaries, "comparisons": comparisons, "r8_r32_consistency": consistency}
        report["strata"][dataset] = stratum_summary(rows, paired, dataset)
        for row in selected:
            item: dict[str, Any] = {"dataset_id": dataset, "sample_id": row["sample_id"], "stratum": row["stratum"], "modes": {}}
            for mode in MODES:
                value = paired[(str(row["sample_id"]), mode)]
                item["modes"][mode] = {
                    "relaxed_qa_f1": value["metrics"]["relaxed_qa_f1"],
                    "relaxed_qa_em": value["metrics"]["relaxed_qa_exact_match"],
                    "generated_tokens": value["generation"]["generated_tokens"],
                    "stop_reason": value["generation"]["stop_reason"],
                    "cap_hit": value["generation"]["cap_hit"],
                    "wall_seconds": value["systems"]["wall_seconds"],
                }
            sample_rows.append(item)
    report["decision"] = {}
    for mode in ("tail_refresh_r8", "tail_refresh_r32"):
        passes = strict_dataset_pass[mode]
        report["decision"][mode] = {
            "status": "passes_pilot_promising_rule" if all(passes.values()) else "fails_pilot_promising_rule",
            "dataset_passes": dict(passes),
            "rule": "strict: positive relaxed-F1 delta vs R1 and nonworse cap/valid-termination behavior on every registered pilot dataset",
        }
    report["decision"]["nomination"] = "none; neither hybrid passes the strict all-dataset pilot rule"
    return report, sample_rows


def validate_and_load() -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    status = read_json(RUN_ROOT / "status.json")
    summary = read_json(RUN_ROOT / "summary.json")
    definition = read_json(RUN_ROOT / "run_definition.json")
    prepared_manifest_path = PREPARED / "manifest.json"
    selected_path = PREPARED / "selected_samples.jsonl"
    if sha256_file(prepared_manifest_path) != EXPECTED_PREPARED_MANIFEST_SHA256:
        raise ValidationError("prepared manifest identity differs")
    if sha256_file(selected_path) != EXPECTED_SELECTED_SHA256:
        raise ValidationError("selected sample registry identity differs")
    prepared_manifest = read_json(prepared_manifest_path)
    selected_entry = prepared_manifest.get("files", {}).get("selected_samples.jsonl", {})
    if (
        prepared_manifest.get("status") != "prepared"
        or selected_entry.get("sha256") != EXPECTED_SELECTED_SHA256
        or selected_entry.get("nbytes") != selected_path.stat().st_size
    ):
        raise ValidationError("prepared manifest does not bind the selected registry")
    if status.get("status") != "complete" or status.get("failures") != 0 or summary.get("status") != "complete":
        raise ValidationError("runner did not finish cleanly")
    if status.get("request_fingerprint") != EXPECTED_FINGERPRINT or summary.get("request_fingerprint") != EXPECTED_FINGERPRINT:
        raise ValidationError("run fingerprint differs")
    if status.get("summary_sha256") != sha256_file(RUN_ROOT / "summary.json"):
        raise ValidationError("runner summary hash differs")
    if (
        summary.get("expected_atomic_results") != 384
        or summary.get("complete_atomic_results") != 384
        or summary.get("missing") != []
        or definition.get("expected_atomic_results") != 384
        or definition.get("sample_count") != 96
        or definition.get("modes") != list(MODES)
        or definition.get("request_fingerprint") != EXPECTED_FINGERPRINT
        or definition.get("prepared_manifest_sha256") != EXPECTED_PREPARED_MANIFEST_SHA256
    ):
        raise ValidationError("run definition/summary inventory contract differs")
    rows = read_selected()
    expected = {expected_result_path(row, mode).resolve() for row in rows for mode in MODES}
    actual = {path.resolve() for path in (RUN_ROOT / "results").rglob("*.json") if not path.name.endswith(".failure.json")}
    failures = list((RUN_ROOT / "results").rglob("*.failure.json"))
    if failures:
        raise ValidationError(f"failure artifacts exist: {len(failures)}")
    if actual != expected:
        raise ValidationError(f"result inventory differs: missing={len(expected-actual)}, extra={len(actual-expected)}")
    if len(actual) != 384:
        raise ValidationError("result inventory is not exactly 96 x 4")
    if sha256_file(TOKENIZER_ROOT / "tokenizer.json") != EXPECTED_TOKENIZER_SHA256:
        raise ValidationError("target tokenizer identity differs")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ROOT, local_files_only=True, trust_remote_code=False)
    paired: dict[tuple[str, str], dict[str, Any]] = {}
    inventory: list[dict[str, Any]] = []
    for row in rows:
        target_path = expected_result_path(row, "target_standalone")
        target = read_json(target_path)
        target_hash = canonical_sha256(target)
        for mode in MODES:
            path = expected_result_path(row, mode)
            value = read_json(path)
            validate_result(value, row, mode, tokenizer, target_hash)
            key = (str(row["sample_id"]), mode)
            if key in paired:
                raise ValidationError("duplicate paired result key")
            paired[key] = value
            inventory.append({"path": str(path.relative_to(CANDIDATE_ROOT)), "nbytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows, paired, sorted(inventory, key=lambda item: item["path"])


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(report: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]) -> None:
    outputs = ANALYSIS_ROOT / "outputs"
    analysis_path = outputs / "analysis.json"
    samples_path = outputs / "paired_samples.jsonl"
    inventory_path = outputs / "result_inventory.json"
    atomic_write(analysis_path, canonical_bytes(report) + b"\n")
    atomic_write(samples_path, b"".join(canonical_bytes(row) + b"\n" for row in samples))
    atomic_write(inventory_path, canonical_bytes({"count": len(inventory), "files": inventory}) + b"\n")
    bindings = {
        "run_status": RUN_ROOT / "status.json",
        "run_summary": RUN_ROOT / "summary.json",
        "run_definition": RUN_ROOT / "run_definition.json",
        "method_provenance": RUN_ROOT / "method_provenance.json",
        "prepared_manifest": PREPARED / "manifest.json",
        "selected_samples": PREPARED / "selected_samples.jsonl",
        "tokenizer_json": TOKENIZER_ROOT / "tokenizer.json",
        "analyzer": Path(__file__).resolve(),
        "analysis": analysis_path,
        "paired_samples": samples_path,
        "result_inventory": inventory_path,
    }
    receipt = {
        "schema_version": 1,
        "status": "passed_strict_cpu_validation",
        "validated_atomic_results": len(inventory),
        "expected_atomic_results": 384,
        "failures": 0,
        "request_fingerprint": EXPECTED_FINGERPRINT,
        "result_inventory_digest": canonical_sha256(inventory),
        "bindings": {name: {"path": str(path), "nbytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in bindings.items()},
    }
    atomic_write(outputs / "validation_receipt.json", canonical_bytes(receipt) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="atomically write analysis outputs")
    args = parser.parse_args(argv)
    rows, paired, inventory = validate_and_load()
    report, samples = analyze(rows, paired)
    if args.write:
        write_outputs(report, samples, inventory)
    else:
        print(json.dumps(report["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
