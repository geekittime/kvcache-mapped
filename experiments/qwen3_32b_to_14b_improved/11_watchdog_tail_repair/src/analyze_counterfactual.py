#!/usr/bin/env python3
"""Strict CPU-only WGTR counterfactual over the frozen 03 pilot inventory."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from .watchdog import trigger_from_generation


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.json"
RESULTS = ROOT / "results"
PRIMARY_F1 = "qa_f1"
PRIMARY_EM = "qa_exact_match"
VALID_STOPS = {"eos", "stop_sequence", "max_tokens"}


class ValidationError(ValueError):
    """An input binding or raw result violated the frozen contract."""


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


def verify_ref(raw: Mapping[str, Any], label: str) -> Path:
    path = Path(str(raw.get("path", ""))).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValidationError(f"{label} is not a regular file")
    if path.stat().st_size != raw.get("nbytes") or sha256_file(path) != raw.get("sha256"):
        raise ValidationError(f"{label} content identity changed")
    return path


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValidationError("empty bootstrap distribution")
    location = (len(ordered) - 1) * q
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return float(ordered[lower])
    weight = location - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def paired_bootstrap(candidate: Sequence[float], baseline: Sequence[float], *, seed: int, replicates: int) -> dict[str, Any]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValidationError("paired bootstrap inputs differ")
    deltas = [float(a) - float(b) for a, b in zip(candidate, baseline)]
    rng = random.Random(seed)
    draws = [sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas) for _ in range(replicates)]
    return {
        "estimate": sum(deltas) / len(deltas),
        "ci95_percentile": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "seed": seed,
        "replicates": replicates,
        "resampling_unit": "paired_sample_id",
    }


def _validate_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValidationError(f"{label} must be finite in [0,1]")
    return float(value)


def validate_raw_result(value: Mapping[str, Any], *, expected_dataset: str, expected_mode: str, expected_filename: str) -> None:
    for field, expected in {
        "schema_version": 1,
        "status": "complete",
        "phase": "pilot",
        "dataset_id": expected_dataset,
        "mode": expected_mode,
    }.items():
        if value.get(field) != expected:
            raise ValidationError(f"raw result {field} changed")
    sample_id = value.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValidationError("raw sample_id is invalid")
    identity = hashlib.sha256(f"{expected_dataset}\0{sample_id}".encode()).hexdigest()[:20] + ".json"
    if identity != expected_filename:
        raise ValidationError("raw result filename/identity differs")
    generation = value.get("generation")
    if not isinstance(generation, Mapping):
        raise ValidationError("raw generation evidence is missing")
    ids = generation.get("generated_token_ids")
    count = generation.get("generated_tokens")
    cap = generation.get("cap_hit")
    reason = generation.get("stop_reason")
    if not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids):
        raise ValidationError("raw generated token IDs are malformed")
    if count != len(ids) or generation.get("generated_token_ids_sha256") != canonical_sha256(ids):
        raise ValidationError("raw generated token count/hash differs")
    if generation.get("requested_max_tokens") != 256 or reason not in VALID_STOPS or type(cap) is not bool:
        raise ValidationError("raw generation cap/stop contract differs")
    if reason == "max_tokens":
        if count != 256 or cap is not True:
            raise ValidationError("raw max-token evidence is inconsistent")
    elif cap is not False or count >= 256:
        raise ValidationError("raw non-cap evidence is inconsistent")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValidationError("raw metrics are missing")
    for field in (PRIMARY_F1, PRIMARY_EM, "relaxed_qa_f1", "relaxed_qa_exact_match", "musique_official_answer_f1"):
        _validate_number(metrics.get(field), f"raw metric {field}")
    systems = value.get("systems")
    if not isinstance(systems, Mapping) or not isinstance(systems.get("wall_seconds"), (int, float)) or systems["wall_seconds"] <= 0:
        raise ValidationError("raw wall time is invalid")


def load_and_validate(config_path: Path) -> tuple[dict[str, Any], dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    config = read_json(config_path)
    if config.get("schema_version") != 1 or config.get("status") != "post_hoc_hypothesis_only":
        raise ValidationError("config claim boundary changed")
    watchdog = config.get("watchdog", {})
    if watchdog != {
        "stall_budget_generated_tokens": 32,
        "initial_mode": "mapped_r1",
        "repair_mode": "tail_refresh_r32",
        "trigger": "r1_generated_tokens_gt_32_or_cap_hit",
        "gold_blind": True,
        "dataset_blind": True,
    }:
        raise ValidationError("watchdog gate changed")
    upstream = config["upstream"]
    upstream_root = Path(upstream["root"]).resolve()
    receipt_path = verify_ref(upstream["analysis_receipt"], "03 validation receipt")
    inventory_path = verify_ref(upstream["result_inventory"], "03 result inventory")
    paired_path = verify_ref(upstream["paired_samples"], "03 paired samples")
    receipt = read_json(receipt_path)
    inventory = read_json(inventory_path)
    if (
        receipt.get("status") != "passed_strict_cpu_validation"
        or receipt.get("validated_atomic_results") != 384
        or receipt.get("expected_atomic_results") != 384
        or receipt.get("failures") != 0
    ):
        raise ValidationError("03 strict validation receipt did not pass")
    for name, expected in (("result_inventory", upstream["result_inventory"]), ("paired_samples", upstream["paired_samples"])):
        bound = receipt.get("bindings", {}).get(name)
        if bound != expected:
            raise ValidationError(f"03 receipt no longer binds {name}")
    files = inventory.get("files")
    if inventory.get("count") != 384 or not isinstance(files, list) or len(files) != 384:
        raise ValidationError("03 result inventory is not exactly 384 files")
    if receipt.get("result_inventory_digest") != canonical_sha256(files):
        raise ValidationError("03 result inventory digest differs")

    expected_modes = set(upstream["expected_modes"])
    expected_dataset_counts = upstream["expected_datasets"]
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    verified_inventory: list[dict[str, Any]] = []
    for raw in files:
        relative = Path(str(raw.get("path", "")))
        path = (upstream_root / relative).resolve()
        try:
            path.relative_to(upstream_root)
        except ValueError as exc:
            raise ValidationError("inventory path escapes the 03 root") from exc
        if path.is_symlink() or not path.is_file():
            raise ValidationError("raw result is not a regular file")
        if path.stat().st_size != raw.get("nbytes") or sha256_file(path) != raw.get("sha256"):
            raise ValidationError(f"raw inventory member changed: {relative}")
        parts = relative.parts
        if len(parts) < 4 or "results" not in parts:
            raise ValidationError("unexpected raw result path")
        idx = parts.index("results")
        dataset, mode, filename = parts[idx + 1 : idx + 4]
        if mode not in expected_modes or dataset not in expected_dataset_counts or filename != parts[-1]:
            raise ValidationError("unexpected dataset/mode in raw result path")
        value = read_json(path)
        validate_raw_result(value, expected_dataset=dataset, expected_mode=mode, expected_filename=filename)
        key = (value["sample_id"], mode)
        if key in by_key:
            raise ValidationError("duplicate raw sample/mode key")
        by_key[key] = value
        verified_inventory.append({"path": str(relative), "nbytes": path.stat().st_size, "sha256": sha256_file(path)})

    sample_modes: dict[str, set[str]] = defaultdict(set)
    sample_dataset: dict[str, str] = {}
    for (sample_id, mode), value in by_key.items():
        sample_modes[sample_id].add(mode)
        old = sample_dataset.setdefault(sample_id, value["dataset_id"])
        if old != value["dataset_id"]:
            raise ValidationError("sample identity crosses datasets")
    if len(sample_modes) != 96 or any(modes != expected_modes for modes in sample_modes.values()):
        raise ValidationError("raw results are not exactly 96 complete four-mode pairs")
    observed_counts = Counter(sample_dataset.values())
    if dict(observed_counts) != expected_dataset_counts:
        raise ValidationError("per-dataset sample counts changed")
    shared = ("dataset_id", "sample_id", "source_record_sha256", "prompt_token_ids_sha256", "prompt_token_count", "references", "request_fingerprint")
    for sample_id, modes in sample_modes.items():
        anchor = by_key[(sample_id, "mapped_r1")]
        for mode in modes:
            current = by_key[(sample_id, mode)]
            if any(current.get(field) != anchor.get(field) for field in shared):
                raise ValidationError("cross-mode sample bindings differ")
    if sha256_file(paired_path) != upstream["paired_samples"]["sha256"]:
        raise ValidationError("paired samples changed during validation")
    return config, by_key, sorted(verified_inventory, key=lambda item: item["path"])


def _mean(values: Iterable[float]) -> float:
    sequence = list(values)
    return sum(sequence) / len(sequence)


def analyze(config: Mapping[str, Any], by_key: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    budget = int(config["watchdog"]["stall_budget_generated_tokens"])
    bootstrap = config["bootstrap"]
    datasets = config["upstream"]["expected_datasets"]
    report: dict[str, Any] = {
        "schema_version": 1,
        "method_id": config["method_id"],
        "status": "exploratory_post_hoc_counterfactual_complete",
        "claim_boundary": config["claim_boundary"],
        "gate": {
            "stall_budget_generated_tokens": budget,
            "trigger_inputs": ["mapped_r1.generation.generated_tokens", "mapped_r1.generation.cap_hit"],
            "forbidden_gate_inputs": ["references", "metrics", "prediction", "dataset_id", "sample_id"],
        },
        "datasets": {},
    }
    sample_audit: list[dict[str, Any]] = []
    for dataset, expected_count in datasets.items():
        sample_ids = sorted({sample_id for (sample_id, _), value in by_key.items() if value["dataset_id"] == dataset})
        if len(sample_ids) != expected_count:
            raise ValidationError("analysis dataset count differs")
        r1_values = [by_key[(sample_id, "mapped_r1")] for sample_id in sample_ids]
        r32_values = [by_key[(sample_id, "tail_refresh_r32")] for sample_id in sample_ids]
        # Selection happens before and independently of any metric access.
        triggers = [trigger_from_generation(dict(value["generation"]), stall_budget=budget) for value in r1_values]
        selected = [r32 if trigger else r1 for r1, r32, trigger in zip(r1_values, r32_values, triggers)]
        for r1, chosen, trigger in zip(r1_values, selected, triggers):
            if (chosen is r1) != (not trigger):
                raise ValidationError("no-trigger R1 object identity was not preserved")
        r1_f1 = [float(value["metrics"][PRIMARY_F1]) for value in r1_values]
        selected_f1 = [float(value["metrics"][PRIMARY_F1]) for value in selected]
        r1_em = [float(value["metrics"][PRIMARY_EM]) for value in r1_values]
        selected_em = [float(value["metrics"][PRIMARY_EM]) for value in selected]
        trigger_count = sum(triggers)
        r1_steps = sum(int(value["generation"]["generated_tokens"]) for value in r1_values)
        online_steps = sum(
            min(int(r1["generation"]["generated_tokens"]), budget)
            + (int(r32["generation"]["generated_tokens"]) if trigger else 0)
            for r1, r32, trigger in zip(r1_values, r32_values, triggers)
        )
        r1_wall = sum(float(value["systems"]["wall_seconds"]) for value in r1_values)
        conservative_wall = sum(
            float(r1["systems"]["wall_seconds"]) + (float(r32["systems"]["wall_seconds"]) if trigger else 0.0)
            for r1, r32, trigger in zip(r1_values, r32_values, triggers)
        )
        selected_recorded_wall = sum(float(value["systems"]["wall_seconds"]) for value in selected)
        report["datasets"][dataset] = {
            "count": expected_count,
            "primary_metric": "raw normalized-token QA F1 (MuSiQue official-compatible)",
            "r1": {"f1": _mean(r1_f1), "em": _mean(r1_em)},
            "wgtr_r32": {"f1": _mean(selected_f1), "em": _mean(selected_em)},
            "paired_delta": {
                "f1": paired_bootstrap(selected_f1, r1_f1, seed=int(bootstrap["seed"]), replicates=int(bootstrap["replicates"])),
                "em": paired_bootstrap(selected_em, r1_em, seed=int(bootstrap["seed"]) + 1, replicates=int(bootstrap["replicates"])),
            },
            "trigger": {
                "count": trigger_count,
                "rate": trigger_count / expected_count,
                "cap_hits_among_triggers": sum(bool(r1["generation"]["cap_hit"]) for r1, trigger in zip(r1_values, triggers) if trigger),
                "noncap_long_generations_among_triggers": sum(not bool(r1["generation"]["cap_hit"]) for r1, trigger in zip(r1_values, triggers) if trigger),
            },
            "compute_accounting": {
                "exact_generated_decode_steps_r1_recorded": r1_steps,
                "exact_generated_decode_steps_wgtr_counterfactual": online_steps,
                "decode_step_ratio_vs_r1": online_steps / r1_steps,
                "r32_restarts": trigger_count,
                "extra_r32_native_prompt_replay_tokens": 31 * trigger_count,
                "extra_mapped_prefill_rebuilds": trigger_count,
                "recorded_selected_mode_wall_ratio_not_online_runtime": selected_recorded_wall / r1_wall,
                "conservative_full_r1_plus_r32_wall_upper_proxy_ratio": conservative_wall / r1_wall,
                "wall_warning": "03 ran full R1 and R32 independently; a true watchdog aborts R1 at token 32, so neither recorded-wall ratio is a measured online latency.",
            },
        }
        for sample_id, r1, r32, chosen, trigger in zip(sample_ids, r1_values, r32_values, selected, triggers):
            sample_audit.append(
                {
                    "dataset_id": dataset,
                    "sample_id": sample_id,
                    "triggered": trigger,
                    "selected_mode": "tail_refresh_r32" if trigger else "mapped_r1",
                    "r1_generated_tokens": r1["generation"]["generated_tokens"],
                    "r1_cap_hit": r1["generation"]["cap_hit"],
                    "r1_f1": r1["metrics"][PRIMARY_F1],
                    "selected_f1": chosen["metrics"][PRIMARY_F1],
                    "r1_raw_sha256": canonical_sha256(r1),
                    "selected_raw_sha256": canonical_sha256(chosen),
                    "r32_raw_sha256": canonical_sha256(r32),
                }
            )
    return report, sample_audit


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_ref(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "nbytes": path.stat().st_size, "sha256": sha256_file(path)}


def write_sidecar(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256.json")
    atomic_write(sidecar, canonical_bytes({"schema_version": 1, "artifact": file_ref(path)}) + b"\n")


def write_outputs(config_path: Path, config: Mapping[str, Any], report: Mapping[str, Any], samples: Sequence[Mapping[str, Any]], inventory: Sequence[Mapping[str, Any]]) -> None:
    result_path = RESULTS / "analysis.json"
    samples_path = RESULTS / "sample_audit.jsonl"
    atomic_write(result_path, canonical_bytes(report) + b"\n")
    atomic_write(samples_path, b"".join(canonical_bytes(row) + b"\n" for row in samples))
    receipt = {
        "schema_version": 1,
        "status": "passed_strict_cpu_counterfactual",
        "cuda_invoked": False,
        "gpu_used": False,
        "validated_upstream_atomic_results": len(inventory),
        "upstream_inventory_digest": canonical_sha256(inventory),
        "claim_boundary": config["claim_boundary"],
        "bindings": {
            "config": file_ref(config_path),
            "analyzer": file_ref(Path(__file__)),
            "runtime_state_machine": file_ref(ROOT / "src/watchdog.py"),
            "03_validation_receipt": config["upstream"]["analysis_receipt"],
            "03_result_inventory": config["upstream"]["result_inventory"],
            "03_paired_samples": config["upstream"]["paired_samples"],
            "analysis": file_ref(result_path),
            "sample_audit": file_ref(samples_path),
        },
    }
    receipt_path = RESULTS / "receipt.json"
    atomic_write(receipt_path, canonical_bytes(receipt) + b"\n")
    for path in (result_path, samples_path, receipt_path):
        write_sidecar(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config, by_key, inventory = load_and_validate(config_path)
    report, samples = analyze(config, by_key)
    if args.write:
        write_outputs(config_path, config, report, samples, inventory)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
