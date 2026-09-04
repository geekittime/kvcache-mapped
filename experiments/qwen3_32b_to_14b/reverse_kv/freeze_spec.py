"""Single source of truth for the reverse experiment's code-freeze scope."""

from __future__ import annotations

from pathlib import Path

from .protocol import EXPERIMENT_ROOT, REPOSITORY_ROOT


LOCAL_DOCUMENTS = ("README.md", "EVALUATION.md", "FIGURE4_K4.md")
LOCAL_DIRECTORIES = ("config", "reverse_kv", "scripts", "tests")
SHARED_RELATIVE = (
    "scripts/select_layers.py",
    "scripts/fit_mapper.py",
    "scripts/evaluate.py",
    "src/kvcache_transfer/__init__.py",
    "src/kvcache_transfer/activation_store.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/eval_data.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/fit_mapper.py",
    "src/kvcache_transfer/io.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/metrics.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/ridge.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/runlog.py",
    "src/kvcache_transfer/select_layers.py",
    "src/kvcache_transfer/transfer.py",
)
BENCHMARK_PROTOCOL_RELATIVE = (
    "data/benchmarks/manifest.json",
    "data/benchmarks/lm_eval_tasks/hellaswag.yaml",
)


def local_paths() -> tuple[Path, ...]:
    paths = [EXPERIMENT_ROOT / name for name in LOCAL_DOCUMENTS]
    for directory in LOCAL_DIRECTORIES:
        paths.extend(
            path
            for path in (EXPERIMENT_ROOT / directory).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    return tuple(sorted(set(paths)))


def shared_paths() -> tuple[Path, ...]:
    return tuple(REPOSITORY_ROOT / relative for relative in SHARED_RELATIVE)


def benchmark_protocol_paths() -> tuple[Path, ...]:
    return tuple(
        REPOSITORY_ROOT / relative for relative in BENCHMARK_PROTOCOL_RELATIVE
    )


def vendored_harness_paths() -> tuple[Path, ...]:
    lm_eval_root = REPOSITORY_ROOT / "vendor/lm-evaluation-harness/lm_eval"
    paths = [
        path
        for path in lm_eval_root.rglob("*.py")
        if "tasks" not in path.relative_to(lm_eval_root).parts
    ]
    paths.extend((lm_eval_root / "tasks").glob("*.py"))
    for pattern in ("*.py", "*.yaml", "*.yml", "*.json"):
        paths.extend((lm_eval_root / "tasks/hellaswag").rglob(pattern))
    return tuple(sorted(set(paths)))


def expected_groups() -> dict[str, tuple[Path, ...]]:
    return {
        "local_files": local_paths(),
        "read_only_shared_dependencies": shared_paths(),
        "benchmark_protocol_dependencies": benchmark_protocol_paths(),
        "vendored_harness_dependencies": vendored_harness_paths(),
    }


__all__ = [
    "BENCHMARK_PROTOCOL_RELATIVE",
    "LOCAL_DIRECTORIES",
    "LOCAL_DOCUMENTS",
    "SHARED_RELATIVE",
    "benchmark_protocol_paths",
    "expected_groups",
    "local_paths",
    "shared_paths",
    "vendored_harness_paths",
]
