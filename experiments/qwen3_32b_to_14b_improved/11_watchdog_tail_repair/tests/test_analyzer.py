from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src import analyze_counterfactual as analyzer


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_384_raw_inventory_and_expected_counterfactual():
    config, by_key, inventory = analyzer.load_and_validate(ROOT / "config.json")
    assert len(inventory) == 384
    assert len(by_key) == 384
    report, samples = analyzer.analyze(config, by_key)
    assert len(samples) == 96
    hotpot = report["datasets"]["hotpotqa_train"]
    musique = report["datasets"]["musique_train"]
    assert hotpot["trigger"]["count"] == 1
    assert hotpot["r1"]["f1"] == pytest.approx(0.7314153439153439)
    assert hotpot["wgtr_r32"]["f1"] == pytest.approx(0.7418320105820105)
    assert musique["trigger"]["count"] == 13
    assert musique["r1"]["f1"] == pytest.approx(0.3638786088425927)
    assert musique["wgtr_r32"]["f1"] == pytest.approx(0.4476449275362319)


def test_generation_tamper_is_rejected():
    config, by_key, _ = analyzer.load_and_validate(ROOT / "config.json")
    value = copy.deepcopy(next(iter(by_key.values())))
    value["generation"]["generated_tokens"] += 1
    filename = analyzer.hashlib.sha256(
        f"{value['dataset_id']}\0{value['sample_id']}".encode()
    ).hexdigest()[:20] + ".json"
    with pytest.raises(analyzer.ValidationError, match="count/hash"):
        analyzer.validate_raw_result(
            value,
            expected_dataset=value["dataset_id"],
            expected_mode=value["mode"],
            expected_filename=filename,
        )


def test_pinned_ref_byte_tamper_is_rejected(tmp_path):
    path = tmp_path / "artifact.json"
    path.write_text('{"ok":true}\n', encoding="utf-8")
    ref = {"path": str(path), "nbytes": path.stat().st_size, "sha256": analyzer.sha256_file(path)}
    path.write_text('{"ok":false}\n', encoding="utf-8")
    with pytest.raises(analyzer.ValidationError, match="identity changed"):
        analyzer.verify_ref(ref, "synthetic")


def test_written_receipt_and_sidecars_are_self_consistent():
    receipt_path = ROOT / "results/receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "passed_strict_cpu_counterfactual"
    assert receipt["validated_upstream_atomic_results"] == 384
    assert receipt["gpu_used"] is False and receipt["cuda_invoked"] is False
    for name in ("analysis.json", "sample_audit.jsonl", "receipt.json"):
        artifact = ROOT / "results" / name
        sidecar = json.loads((artifact.with_name(name + ".sha256.json")).read_text(encoding="utf-8"))
        assert sidecar["artifact"]["sha256"] == analyzer.sha256_file(artifact)
        assert sidecar["artifact"]["nbytes"] == artifact.stat().st_size
