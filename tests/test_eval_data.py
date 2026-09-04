from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kvcache_transfer.eval_data import (
    DATASET_LAYOUTS,
    BenchmarkDataError,
    generate_local_task_overrides,
    load_benchmark_config,
    parquet_schema_signature,
    verify_ready_manifest,
)
from kvcache_transfer.io import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/benchmarks_modelscope.yaml"
HARNESS = ROOT / "vendor/lm-evaluation-harness"


def _configuration(tmp_path: Path | None = None):
    configuration = load_benchmark_config(CONFIG, ROOT)
    if tmp_path is not None:
        configuration = copy.deepcopy(configuration)
        configuration["local_task_dir"] = tmp_path / "tasks"
    return configuration


@pytest.fixture(scope="module")
def generated_configuration(tmp_path_factory):
    configuration = _configuration(tmp_path_factory.mktemp("local_tasks"))
    generated = generate_local_task_overrides(configuration, HARNESS)
    return configuration, generated


def test_downloaded_parquets_have_exact_expected_rows_and_schemas():
    configuration = _configuration()
    for name, layout in DATASET_LAYOUTS.items():
        root = Path(configuration["datasets"][name]["local_root"])
        totals: dict[str, int] = {}
        for item in layout.files:
            path = root / item.relative_path
            assert path.is_file(), path
            import pyarrow.parquet as pq

            totals[item.split] = totals.get(item.split, 0) + pq.read_metadata(path).num_rows
            assert parquet_schema_signature(path) == layout.expected_schema
        assert totals == dict(layout.expected_split_rows)


def test_generated_overrides_cannot_replace_prompt_or_metric(generated_configuration):
    _, generated = generated_configuration
    assert len(generated) == 61
    for item in generated.values():
        import yaml

        raw = yaml.safe_load(Path(item["path"]).read_text(encoding="utf-8"))
        assert set(raw) == {"include", "dataset_path", "dataset_name", "dataset_kwargs"}
        assert item["prompt_metric_overrides"] == []


@pytest.mark.parametrize(
    ("generated_name", "canonical_relative"),
    [
        ("hellaswag.yaml", "hellaswag/hellaswag.yaml"),
        ("arc_challenge.yaml", "arc/arc_challenge.yaml"),
        ("winogrande.yaml", "winogrande/default.yaml"),
        ("gsm8k_cot.yaml", "gsm8k/gsm8k-cot.yaml"),
        ("mmlu_abstract_algebra.yaml", "mmlu/default/mmlu_abstract_algebra.yaml"),
    ],
)
def test_recursive_local_config_is_canonical_except_dataset_loader(
    generated_configuration, generated_name, canonical_relative
):
    from lm_eval.tasks._yaml_loader import load_yaml

    configuration, _ = generated_configuration
    local = load_yaml(
        Path(configuration["local_task_dir"]) / generated_name, resolve_func=False
    )
    canonical = load_yaml(
        HARNESS / "lm_eval/tasks" / canonical_relative, resolve_func=False
    )
    for key in ("dataset_path", "dataset_name", "dataset_kwargs"):
        local.pop(key, None)
        canonical.pop(key, None)
    assert local == canonical


def test_generated_tasks_load_selected_data_fully_offline(generated_configuration, monkeypatch):
    from lm_eval.tasks._yaml_loader import load_yaml
    from lm_eval.api.task import ConfigurableTask

    configuration, _ = generated_configuration
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    filenames = [
        "hellaswag.yaml",
        "arc_challenge.yaml",
        "winogrande.yaml",
        "gsm8k_cot.yaml",
        "mmlu_abstract_algebra.yaml",
    ]
    loaded = {}
    for filename in filenames:
        config = load_yaml(Path(configuration["local_task_dir"]) / filename)
        task = ConfigurableTask(config=config)
        loaded[config["task"]] = task
    assert all(task.config.dataset_path == "parquet" for task in loaded.values())
    assert len(loaded["hellaswag"].validation_docs()) == 10_042
    assert len(loaded["arc_challenge"].test_docs()) == 1_172
    assert len(loaded["winogrande"].validation_docs()) == 1_267
    assert len(loaded["gsm8k_cot"].test_docs()) == 1_319
    assert len(loaded["mmlu_abstract_algebra"].test_docs()) == 100
    assert len(loaded["mmlu_abstract_algebra"].fewshot_docs()) == 5


def test_ready_manifest_rejects_changed_task_file(tmp_path):
    configuration = _configuration(tmp_path)
    generated = generate_local_task_overrides(configuration, HARNESS)
    datasets = {}
    for name in DATASET_LAYOUTS:
        source = tmp_path / f"{name}.bin"
        source.write_bytes(name.encode())
        digest = sha256_file(source)
        datasets[name] = {
            "all_files_byte_identical_to_canonical": True,
            "all_files_verified_against_modelscope": True,
            "files": [
                {
                    "path": str(source),
                    "nbytes": source.stat().st_size,
                    "sha256": digest,
                    "canonical_sha256": digest,
                    "modelscope_sha256": digest,
                }
            ],
        }
    manifest = {
        "schema_version": 1,
        "status": "ready",
        "datasets": datasets,
        "local_task_files": generated,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    verify_ready_manifest(path)
    changed = Path(next(iter(generated.values()))["path"])
    changed.write_text(changed.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
    with pytest.raises(BenchmarkDataError, match="override changed"):
        verify_ready_manifest(path)
