"""Immutable scientific contract for the reverse five-task extension.

This package is deliberately outside ``reverse_kv`` and the completed paper
pipeline.  It consumes the already committed k=20 mapper without mutating it.
"""

from __future__ import annotations

from pathlib import Path


EXTENSION_VERSION = "reverse-five-task-k20-v1"
PAPER_BIT_EXACT = False
PAPER_SCOPE_NOTE = (
    "The paper does not publish a Qwen3-32B-to-Qwen3-14B full five-task "
    "accuracy/PPL table. These are protocol-aligned local extension results, "
    "not paper-bit-exact target values."
)

EXTENSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXTENSION_ROOT.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
BASE_EVAL_CONFIG = EXPERIMENT_ROOT / "config/eval.yaml"
EXTENSION_CONFIG = EXTENSION_ROOT / "config.yaml"
EXTENSION_CODE_FREEZE = EXTENSION_ROOT / "code_freeze.json"
DEFAULT_FULL_OUTPUT = EXPERIMENT_ROOT / "runs/five_task_extension_full_k20_v1"
DEFAULT_SMOKE_OUTPUT = EXPERIMENT_ROOT / "runs/five_task_extension_smoke_k20_v1"
HELLASWAG_ROOT = EXPERIMENT_ROOT / "runs/formal_pipeline_v1/hellaswag_full_k20"
MAPPER_MANIFEST = (
    EXPERIMENT_ROOT
    / "artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01/manifest.json"
)

MODES = ("target_standalone", "mapped")
# HellaSwag is excluded here only because the identical reverse formal run is
# bridged after its complete receipt and every raw row are revalidated.
EXECUTED_TASKS = (
    "arc_challenge",
    "winogrande",
    "mmlu",
    "gsm8k_cot",
    "wikitext2_prefix_ppl",
)
ACCURACY_TASKS = (
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "mmlu",
    "gsm8k_cot",
)
TASK_METRICS = {
    "arc_challenge": "acc_norm",
    "hellaswag": "acc_norm",
    "winogrande": "acc",
    "mmlu": "acc",
    "gsm8k_cot": "exact_match",
    "wikitext2_prefix_ppl": "perplexity",
}
EXPECTED_FULL_SAMPLES = {
    "arc_challenge": 1_172,
    "hellaswag": 10_042,
    "winogrande": 1_267,
    "mmlu": 14_042,
    "gsm8k_cot": 1_319,
}
SMOKE_LIMIT = 8
SMOKE_WIKITEXT_CHUNKS = 2

FORMAL_TOPOLOGY = {
    "source": {
        "model": "/home/wby/models/Qwen3-32B",
        "devices": [0, 1, 2, 3, 4],
        "layer_counts": [12, 13, 14, 13, 12],
    },
    "target": {
        "model": "/home/wby/models/Qwen3-14B",
        "devices": [5, 6, 7],
        "layer_counts": [13, 14, 13],
    },
}

RECEIPT_NAME = "five_task_extension_receipt.json"
SUMMARY_NAME = "five_task_extension_summary.json"
SUMMARY_MARKDOWN_NAME = "five_task_extension_summary.md"
GPU_GATES_DIRECTORY = "extension_segment_gates"
SMOKE_RECEIPT_NAME = "five_task_extension_smoke_receipt.json"

CODE_IDENTITY_FILES = (
    "five_task_extension/__init__.py",
    "five_task_extension/protocol.py",
    "five_task_extension/validator.py",
    "five_task_extension/run.py",
    "five_task_extension/adapter.py",
    "five_task_extension/prepare.py",
    "five_task_extension/freeze.py",
    "five_task_extension/config.yaml",
    "five_task_extension/README.md",
)


def output_within_experiment(value: str | Path) -> Path:
    output = Path(value).resolve()
    try:
        output.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("extension output must remain inside the reverse experiment") from exc
    return output


__all__ = [name for name in globals() if name.isupper()] + ["output_within_experiment"]
