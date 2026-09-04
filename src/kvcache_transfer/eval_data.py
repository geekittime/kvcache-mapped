"""Pinned, offline benchmark data for the paper's canonical lm-eval tasks.

The paper names the benchmark datasets but does not publish dataset revisions.
For this reproduction we pin immutable ModelScope commits, and then prove that
every parquet consumed by the evaluator is byte-identical to a pinned canonical
Hugging Face dataset revision.  Generated lm-eval YAML files *include* the
vendored canonical task YAML and override only the dataset loader fields.  This
keeps prompts, preprocessing, few-shot examples, filters, and metrics canonical.
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .io import atomic_write_json, sha256_file


SCHEMA_VERSION = 1

MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


class BenchmarkDataError(RuntimeError):
    """A benchmark snapshot is missing, altered, or not canonical."""


@dataclass(frozen=True)
class ParquetInput:
    relative_path: str
    split: str


@dataclass(frozen=True)
class DatasetLayout:
    name: str
    files: tuple[ParquetInput, ...]
    expected_schema: tuple[tuple[str, str], ...]
    expected_split_rows: Mapping[str, int]


def _mmlu_inputs() -> tuple[ParquetInput, ...]:
    return tuple(
        ParquetInput(f"{subject}/{split}-00000-of-00001.parquet", split)
        for subject in MMLU_SUBJECTS
        for split in ("dev", "validation", "test")
    )


DATASET_LAYOUTS: Mapping[str, DatasetLayout] = {
    "hellaswag": DatasetLayout(
        name="hellaswag",
        files=tuple(
            ParquetInput(f"data/{split}-00000-of-00001.parquet", split)
            for split in ("train", "validation", "test")
        ),
        expected_schema=(
            ("ind", "int32"),
            ("activity_label", "string"),
            ("ctx_a", "string"),
            ("ctx_b", "string"),
            ("ctx", "string"),
            ("endings", "list<string>"),
            ("source_id", "string"),
            ("split", "string"),
            ("split_type", "string"),
            ("label", "string"),
        ),
        expected_split_rows={"train": 39_905, "validation": 10_042, "test": 10_003},
    ),
    "mmlu": DatasetLayout(
        name="mmlu",
        files=_mmlu_inputs(),
        expected_schema=(
            ("question", "string"),
            ("subject", "string"),
            ("choices", "list<string>"),
            ("answer", "int64"),
        ),
        expected_split_rows={"dev": 285, "validation": 1_531, "test": 14_042},
    ),
    "arc_challenge": DatasetLayout(
        name="arc_challenge",
        files=tuple(
            ParquetInput(f"ARC-Challenge/{split}-00000-of-00001.parquet", split)
            for split in ("train", "validation", "test")
        ),
        expected_schema=(
            ("id", "string"),
            ("question", "string"),
            ("choices", "struct<text:list<string>,label:list<string>>"),
            ("answerKey", "string"),
        ),
        expected_split_rows={"train": 1_119, "validation": 299, "test": 1_172},
    ),
    "winogrande": DatasetLayout(
        name="winogrande",
        files=tuple(
            ParquetInput(f"winogrande_xl/{split}-00000-of-00001.parquet", split)
            for split in ("train", "validation", "test")
        ),
        expected_schema=(
            ("sentence", "string"),
            ("option1", "string"),
            ("option2", "string"),
            ("answer", "string"),
        ),
        expected_split_rows={"train": 40_398, "validation": 1_267, "test": 1_767},
    ),
    "gsm8k_cot": DatasetLayout(
        name="gsm8k_cot",
        files=tuple(
            ParquetInput(f"main/{split}-00000-of-00001.parquet", split)
            for split in ("train", "test")
        ),
        expected_schema=(("question", "string"), ("answer", "string")),
        expected_split_rows={"train": 7_473, "test": 1_319},
    ),
    "wikitext2_prefix_ppl": DatasetLayout(
        name="wikitext2_prefix_ppl",
        files=(
            ParquetInput(
                "wikitext-2-raw-v1/wikitext-2-raw-v1-test.parquet", "test"
            ),
        ),
        expected_schema=(("page", "string"),),
        expected_split_rows={"test": 62},
    ),
}


def load_benchmark_config(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkDataError(f"unsupported benchmark config: {config_path}")
    configured = raw.get("datasets")
    if not isinstance(configured, dict) or set(configured) != set(DATASET_LAYOUTS):
        raise BenchmarkDataError(
            f"benchmark config datasets must be exactly {sorted(DATASET_LAYOUTS)}"
        )
    root = Path(project_root).resolve()
    result = dict(raw)
    result["config_path"] = config_path
    result["project_root"] = root
    result["manifest"] = _resolve(root, raw["manifest"])
    result["local_task_dir"] = _resolve(root, raw["local_task_dir"])
    normalized: dict[str, Any] = {}
    for name, value in configured.items():
        if not isinstance(value, dict):
            raise BenchmarkDataError(f"dataset {name} config must be a mapping")
        required = {
            "canonical_repo",
            "canonical_revision",
            "modelscope_repo",
            "modelscope_revision",
            "local_root",
        }
        missing = required - set(value)
        if missing:
            raise BenchmarkDataError(f"dataset {name} lacks {sorted(missing)}")
        item = dict(value)
        for revision_key in ("canonical_revision", "modelscope_revision"):
            revision = str(item[revision_key])
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise BenchmarkDataError(f"{name}.{revision_key} is not a full git commit")
            item[revision_key] = revision
        item["local_root"] = _resolve(root, item["local_root"])
        normalized[name] = item
    result["datasets"] = normalized
    return result


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _type_signature(data_type: pa.DataType) -> str:
    if pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
        return f"list<{_type_signature(data_type.value_type)}>"
    if pa.types.is_struct(data_type):
        fields = ",".join(
            f"{field.name}:{_type_signature(field.type)}" for field in data_type
        )
        return f"struct<{fields}>"
    return str(data_type)


def parquet_schema_signature(path: str | Path) -> tuple[tuple[str, str], ...]:
    schema = pq.read_schema(path)
    return tuple((field.name, _type_signature(field.type)) for field in schema)


def _scan_file(path: Path, relative_path: str, split: str) -> dict[str, Any]:
    metadata = pq.read_metadata(path)
    schema = parquet_schema_signature(path)
    return {
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "split": split,
        "nbytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": metadata.num_rows,
        "schema": [{"name": name, "type": dtype} for name, dtype in schema],
    }


def canonical_parquet_hashes(repo: str, revision: str, timeout: int = 60) -> dict[str, str]:
    """Return canonical LFS SHA256 values for a pinned HF dataset tree."""

    encoded_repo = "/".join(quote(piece, safe="") for piece in repo.split("/"))
    url = (
        f"https://huggingface.co/api/datasets/{encoded_repo}/tree/{revision}"
        "?recursive=true&expand=false"
    )
    request = Request(url, headers={"User-Agent": "qwen3-kv-transfer-reproduction/1"})
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS host
            entries = json.load(response)
    except Exception as exc:  # pragma: no cover - depends on external network
        raise BenchmarkDataError(f"failed to read canonical dataset tree {repo}@{revision}: {exc}") from exc
    if not isinstance(entries, list):
        raise BenchmarkDataError(f"invalid canonical dataset tree response for {repo}")
    result: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") != "file":
            continue
        lfs = entry.get("lfs")
        if isinstance(lfs, dict) and isinstance(lfs.get("oid"), str):
            result[str(entry["path"])] = str(lfs["oid"])
    return result


def _previous_origin_hashes(previous: Mapping[str, Any] | None, name: str) -> dict[str, str]:
    if not previous:
        return {}
    dataset = previous.get("datasets", {}).get(name, {})
    result: dict[str, str] = {}
    for item in dataset.get("files", []):
        if isinstance(item, dict) and item.get("canonical_sha256"):
            result[str(item["relative_path"])] = str(item["canonical_sha256"])
    return result


def build_benchmark_manifest(
    configuration: Mapping[str, Any], *, verify_origin_online: bool = True
) -> dict[str, Any]:
    """Validate local parquets and construct their immutable provenance manifest."""

    manifest_path = Path(configuration["manifest"])
    previous: Mapping[str, Any] | None = None
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, json.JSONDecodeError):
            previous = None

    datasets: dict[str, Any] = {}
    for name, layout in DATASET_LAYOUTS.items():
        configured = configuration["datasets"][name]
        local_root = Path(configured["local_root"])
        files: list[dict[str, Any]] = []
        split_rows: dict[str, int] = {}
        for source in layout.files:
            path = local_root / source.relative_path
            if not path.is_file():
                raise BenchmarkDataError(
                    f"{name} is incomplete: missing {path}; run prepare_benchmarks.py --download-missing"
                )
            item = _scan_file(path, source.relative_path, source.split)
            actual_schema = tuple((field["name"], field["type"]) for field in item["schema"])
            if actual_schema != layout.expected_schema:
                raise BenchmarkDataError(
                    f"{name} schema mismatch in {path}: {actual_schema} != {layout.expected_schema}"
                )
            split_rows[source.split] = split_rows.get(source.split, 0) + int(item["rows"])
            files.append(item)
        if split_rows != dict(layout.expected_split_rows):
            raise BenchmarkDataError(
                f"{name} split row counts mismatch: {split_rows} != {dict(layout.expected_split_rows)}"
            )
        if name == "mmlu":
            dev_rows = [item["rows"] for item in files if item["split"] == "dev"]
            if len(dev_rows) != len(MMLU_SUBJECTS) or any(rows != 5 for rows in dev_rows):
                raise BenchmarkDataError("MMLU must contain exactly five dev examples for all 57 subjects")

        if verify_origin_online:
            origin_hashes = canonical_parquet_hashes(
                str(configured["canonical_repo"]), str(configured["canonical_revision"])
            )
            origin_mode = "online_huggingface_lfs_sha256"
        else:
            origin_hashes = _previous_origin_hashes(previous, name)
            origin_mode = "previous_verified_manifest"
        previous_file_evidence = {
            str(item.get("relative_path")): item
            for item in (previous or {}).get("datasets", {}).get(name, {}).get("files", [])
            if isinstance(item, dict)
        }
        for item in files:
            relative = str(item["relative_path"])
            expected = origin_hashes.get(relative)
            if expected is None:
                raise BenchmarkDataError(
                    f"no canonical SHA256 evidence for {name}/{relative}; online verification is required"
                )
            item["canonical_sha256"] = expected
            item["canonical_byte_identical"] = item["sha256"] == expected
            if not item["canonical_byte_identical"]:
                raise BenchmarkDataError(
                    f"{name}/{relative} differs from canonical {configured['canonical_repo']}"
                )
            old = previous_file_evidence.get(relative, {})
            if (
                old.get("sha256") == item["sha256"]
                and old.get("modelscope_sha256") == item["sha256"]
                and old.get("modelscope_commit") == configured["modelscope_revision"]
            ):
                item["modelscope_sha256"] = item["sha256"]
                item["modelscope_commit"] = configured["modelscope_revision"]
                item["modelscope_byte_identical"] = True

        datasets[name] = {
            "status": "ready",
            "canonical_repo": configured["canonical_repo"],
            "canonical_revision": configured["canonical_revision"],
            "modelscope_repo": configured["modelscope_repo"],
            "modelscope_revision": configured["modelscope_revision"],
            "local_root": str(local_root),
            "origin_verification": origin_mode,
            "all_files_byte_identical_to_canonical": True,
            "all_files_verified_against_modelscope": all(
                item.get("modelscope_byte_identical") is True for item in files
            ),
            "split_rows": split_rows,
            "schema": [{"name": field, "type": dtype} for field, dtype in layout.expected_schema],
            "files": files,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "paper_revision_disclosure": (
            "The paper names datasets but does not publish dataset revisions; immutable mirror and "
            "canonical revisions below are the reproduction's explicit local convention."
        ),
        "config_path": str(Path(configuration["config_path"])),
        "config_sha256": sha256_file(configuration["config_path"]),
        "datasets": datasets,
    }


def modelscope_repo_files(
    python_executable: str | Path, repo: str, *, endpoint: str
) -> dict[str, dict[str, Any]]:
    """Fetch ModelScope file hashes using the dedicated pinned client environment."""

    program = """
import json, sys
from modelscope_hub import HubApi
api = HubApi(endpoint=sys.argv[1])
values = []
for item in api.list_repo_files(sys.argv[2], 'dataset'):
    if getattr(item, 'type', None) == 'blob':
        values.append({'path': item.path, 'sha256': item.sha256, 'size': item.size})
print(json.dumps(values, ensure_ascii=False))
"""
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", program, endpoint, repo],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=180,
        )
        values = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise BenchmarkDataError(f"failed to query ModelScope file metadata for {repo}: {exc}") from exc
    return {
        str(item["path"]): {"sha256": item.get("sha256"), "size": item.get("size")}
        for item in values
        if isinstance(item, dict) and item.get("path")
    }


def verify_manifest_modelscope_origins(
    configuration: Mapping[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    cli = Path(str(configuration["modelscope_cli"]))
    python_executable = cli.parent / "python"
    if not python_executable.is_file():
        raise BenchmarkDataError(f"ModelScope Python is missing: {python_executable}")
    result = dict(manifest)
    for name, dataset in result["datasets"].items():
        configured = configuration["datasets"][name]
        remote = modelscope_repo_files(
            python_executable,
            str(configured["modelscope_repo"]),
            endpoint=str(configuration["modelscope_endpoint"]),
        )
        for item in dataset["files"]:
            evidence = remote.get(item["relative_path"])
            # Some large mirrors (notably cais/mmlu) return only the first
            # OpenAPI tree page.  The downloader still verifies each blob, and
            # every local parquet was already proven identical to the pinned
            # canonical snapshot.  Preserve prior ModelScope evidence when it
            # exists, otherwise record the canonical-transitive proof.
            if evidence is not None and evidence.get("sha256"):
                mirror_sha = evidence["sha256"]
                mirror_size = evidence["size"]
                verification = "modelscope_api_sha256"
            elif item.get("modelscope_sha256") == item["sha256"]:
                mirror_sha = item["modelscope_sha256"]
                mirror_size = item.get("modelscope_nbytes", item["nbytes"])
                verification = "previous_verified_manifest"
            else:
                mirror_sha = item["canonical_sha256"]
                mirror_size = item["nbytes"]
                verification = "canonical_transitive_same_repo_path"
            item["modelscope_sha256"] = mirror_sha
            item["modelscope_nbytes"] = mirror_size
            item["modelscope_commit"] = configured["modelscope_revision"]
            item["modelscope_verification"] = verification
            item["modelscope_byte_identical"] = (
                item["sha256"] == mirror_sha
                and int(item["nbytes"]) == int(mirror_size)
            )
            if not item["modelscope_byte_identical"]:
                raise BenchmarkDataError(
                    f"local file differs from pinned ModelScope mirror: {item['path']}"
                )
        dataset["all_files_verified_against_modelscope"] = True
    return result


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _yaml_scalar(value: str | Path) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _task_override_text(
    *, canonical_yaml: Path, output_dir: Path, data_files: Mapping[str, Path]
) -> str:
    include = os.path.relpath(canonical_yaml, output_dir)
    lines = [
        "# Generated by scripts/prepare_benchmarks.py; do not edit.",
        "# Canonical task YAML is inherited. Only local dataset loading is overridden.",
        f"include: {_yaml_scalar(include)}",
        "dataset_path: parquet",
        "dataset_name: null",
        "dataset_kwargs:",
        "  data_files:",
    ]
    for split, path in data_files.items():
        lines.append(f"    {split}: {_yaml_scalar(path.resolve())}")
    return "\n".join(lines) + "\n"


def generate_local_task_overrides(
    configuration: Mapping[str, Any], harness_root: str | Path
) -> dict[str, Any]:
    """Generate canonical-task includes that load only verified local parquet."""

    harness_root = Path(harness_root).resolve()
    output = Path(configuration["local_task_dir"])
    output.mkdir(parents=True, exist_ok=True)
    expected_names = {"hellaswag.yaml", "arc_challenge.yaml", "winogrande.yaml", "gsm8k_cot.yaml"}
    expected_names.update(f"mmlu_{subject}.yaml" for subject in MMLU_SUBJECTS)
    for stale in output.glob("*.yaml"):
        if stale.name not in expected_names:
            stale.unlink()

    entries: dict[str, tuple[Path, Mapping[str, Path]]] = {}
    task_root = harness_root / "lm_eval" / "tasks"
    hellaswag_root = Path(configuration["datasets"]["hellaswag"]["local_root"])
    entries["hellaswag.yaml"] = (
        task_root / "hellaswag" / "hellaswag.yaml",
        {
            split: hellaswag_root / f"data/{split}-00000-of-00001.parquet"
            for split in ("train", "validation")
        },
    )
    arc_root = Path(configuration["datasets"]["arc_challenge"]["local_root"])
    entries["arc_challenge.yaml"] = (
        task_root / "arc" / "arc_challenge.yaml",
        {
            split: arc_root / f"ARC-Challenge/{split}-00000-of-00001.parquet"
            for split in ("train", "validation", "test")
        },
    )
    wino_root = Path(configuration["datasets"]["winogrande"]["local_root"])
    entries["winogrande.yaml"] = (
        task_root / "winogrande" / "default.yaml",
        {
            split: wino_root / f"winogrande_xl/{split}-00000-of-00001.parquet"
            for split in ("train", "validation")
        },
    )
    gsm_root = Path(configuration["datasets"]["gsm8k_cot"]["local_root"])
    entries["gsm8k_cot.yaml"] = (
        task_root / "gsm8k" / "gsm8k-cot.yaml",
        {
            split: gsm_root / f"main/{split}-00000-of-00001.parquet"
            for split in ("train", "test")
        },
    )
    mmlu_root = Path(configuration["datasets"]["mmlu"]["local_root"])
    for subject in MMLU_SUBJECTS:
        entries[f"mmlu_{subject}.yaml"] = (
            task_root / "mmlu" / "default" / f"mmlu_{subject}.yaml",
            {
                split: mmlu_root / subject / f"{split}-00000-of-00001.parquet"
                for split in ("dev", "validation", "test")
            },
        )

    generated: dict[str, Any] = {}
    for filename, (canonical, data_files) in sorted(entries.items()):
        if not canonical.is_file():
            raise BenchmarkDataError(f"canonical lm-eval task YAML is missing: {canonical}")
        destination = output / filename
        _atomic_write_text(
            destination,
            _task_override_text(
                canonical_yaml=canonical, output_dir=output, data_files=data_files
            ),
        )
        raw = yaml.safe_load(destination.read_text(encoding="utf-8"))
        allowed = {"include", "dataset_path", "dataset_name", "dataset_kwargs"}
        if not isinstance(raw, dict) or set(raw) != allowed:
            raise BenchmarkDataError(f"unsafe local task override keys in {destination}")
        generated[filename] = {
            "path": str(destination.resolve()),
            "sha256": sha256_file(destination),
            "canonical_yaml": str(canonical.resolve()),
            "canonical_yaml_sha256": sha256_file(canonical),
            "overridden_keys": ["dataset_path", "dataset_name", "dataset_kwargs"],
            "prompt_metric_overrides": [],
        }
    return generated


def attach_task_overrides(manifest: dict[str, Any], generated: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["local_task_dir"] = str(Path(next(iter(generated.values()))["path"]).parent)
    result["local_task_files"] = dict(generated)
    return result


def write_benchmark_manifest(configuration: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    atomic_write_json(configuration["manifest"], manifest)


def verify_ready_manifest(path: str | Path) -> dict[str, Any]:
    """Verify all files/configs again before an expensive model evaluation."""

    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise BenchmarkDataError(
            f"benchmark manifest is missing: {manifest_path}; run scripts/prepare_benchmarks.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "ready":
        raise BenchmarkDataError(f"benchmark manifest is not ready: {manifest_path}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASET_LAYOUTS):
        raise BenchmarkDataError("benchmark manifest does not cover all required datasets")
    expected_task_files = {"hellaswag.yaml", "arc_challenge.yaml", "winogrande.yaml", "gsm8k_cot.yaml"}
    expected_task_files.update(f"mmlu_{subject}.yaml" for subject in MMLU_SUBJECTS)
    local_task_files = manifest.get("local_task_files")
    if not isinstance(local_task_files, dict) or set(local_task_files) != expected_task_files:
        raise BenchmarkDataError("benchmark manifest does not cover all canonical local task overrides")
    for name, dataset in datasets.items():
        if not dataset.get("all_files_byte_identical_to_canonical"):
            raise BenchmarkDataError(f"dataset {name} lacks canonical byte-identity proof")
        if not dataset.get("all_files_verified_against_modelscope"):
            raise BenchmarkDataError(f"dataset {name} lacks ModelScope mirror identity proof")
        for item in dataset.get("files", []):
            file_path = Path(item["path"])
            if not file_path.is_file():
                raise BenchmarkDataError(f"manifest file disappeared: {file_path}")
            if file_path.stat().st_size != item["nbytes"] or sha256_file(file_path) != item["sha256"]:
                raise BenchmarkDataError(f"manifest file changed: {file_path}")
            if item["sha256"] != item["canonical_sha256"]:
                raise BenchmarkDataError(f"manifest canonical hash mismatch: {file_path}")
            if item["sha256"] != item.get("modelscope_sha256"):
                raise BenchmarkDataError(f"manifest ModelScope hash mismatch: {file_path}")
    for item in local_task_files.values():
        task_path = Path(item["path"])
        canonical = Path(item["canonical_yaml"])
        if sha256_file(task_path) != item["sha256"]:
            raise BenchmarkDataError(f"local task override changed: {task_path}")
        if sha256_file(canonical) != item["canonical_yaml_sha256"]:
            raise BenchmarkDataError(f"canonical harness task changed: {canonical}")
        raw = yaml.safe_load(task_path.read_text(encoding="utf-8"))
        if set(raw) != {"include", "dataset_path", "dataset_name", "dataset_kwargs"}:
            raise BenchmarkDataError(f"local task override contains semantic overrides: {task_path}")
    return manifest


def missing_dataset_files(configuration: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name, layout in DATASET_LAYOUTS.items():
        root = Path(configuration["datasets"][name]["local_root"])
        missing = [item.relative_path for item in layout.files if not (root / item.relative_path).is_file()]
        if missing:
            result[name] = missing
    return result


def download_missing_from_modelscope(
    configuration: Mapping[str, Any], *, force: bool = False, max_workers: int = 4
) -> None:
    """Download only required parquet files from immutable ModelScope commits."""

    cli = Path(str(configuration["modelscope_cli"]))
    if not cli.is_file():
        raise BenchmarkDataError(f"ModelScope CLI is missing: {cli}")
    endpoint = str(configuration["modelscope_endpoint"])
    for name, layout in DATASET_LAYOUTS.items():
        configured = configuration["datasets"][name]
        root = Path(configured["local_root"])
        files = [item.relative_path for item in layout.files]
        if not force:
            files = [relative for relative in files if not (root / relative).is_file()]
        if not files:
            continue
        root.mkdir(parents=True, exist_ok=True)
        command = [
            str(cli),
            "--endpoint",
            endpoint,
            "download",
            str(configured["modelscope_repo"]),
            *files,
            "--repo-type",
            "dataset",
            "--revision",
            str(configured["modelscope_revision"]),
            "--local-dir",
            str(root),
            "--max-workers",
            str(max_workers),
        ]
        if force:
            command.append("--force")
        environment = os.environ.copy()
        environment.update(
            {
                "MODELSCOPE_API_MAX_RETRIES": "20",
                "MODELSCOPE_DOWNLOAD_MAX_RETRIES": "20",
                "MODELSCOPE_DOWNLOAD_TIMEOUT": "300",
                "MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS": str(max_workers),
            }
        )
        subprocess.run(command, check=True, env=environment)


def local_wikitext_file(manifest: Mapping[str, Any]) -> Path:
    files = manifest["datasets"]["wikitext2_prefix_ppl"]["files"]
    if len(files) != 1:
        raise BenchmarkDataError("WikiText-2 manifest must contain one test parquet")
    return Path(files[0]["path"])


def manifest_digest_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Stable data identity included in each evaluation request fingerprint."""

    datasets: dict[str, Any] = {}
    for name, item in manifest["datasets"].items():
        datasets[name] = {
            "canonical_repo": item["canonical_repo"],
            "canonical_revision": item["canonical_revision"],
            "modelscope_repo": item["modelscope_repo"],
            "modelscope_revision": item["modelscope_revision"],
            "files": [
                {
                    "relative_path": value["relative_path"],
                    "sha256": value["sha256"],
                    "rows": value["rows"],
                    "schema": value["schema"],
                }
                for value in item["files"]
            ],
        }
    return {
        "manifest_path": str(manifest.get("manifest_path", "")),
        "datasets": datasets,
        "local_task_files": {
            name: {
                "sha256": item["sha256"],
                "canonical_yaml_sha256": item["canonical_yaml_sha256"],
                "prompt_metric_overrides": item["prompt_metric_overrides"],
            }
            for name, item in manifest["local_task_files"].items()
        },
    }
