"""Frozen contract for the Qwen3-32B -> Qwen3-14B long-context extension.

The paper does not publish these four reverse-direction task results.  This
module therefore describes a protocol-aligned local extension, not a
paper-bit-exact claim.  It deliberately consumes the already committed k=20
mapper and never writes into its artifact directory.
"""

from __future__ import annotations

from pathlib import Path


SCHEMA_VERSION = 1
EXTENSION_VERSION = "reverse-long-context-k20-v1"
DIRECTION = "qwen3_32b_to_14b"
PROTOCOL_NAME = "qwen3_32b_to_14b_long_context_k20_v1"
PAPER_BIT_EXACT = False
PAPER_SCOPE_NOTE = (
    "The paper does not publish Qwen3-32B-to-Qwen3-14B results for HotpotQA, "
    "MultiNews, LMSYS, or ShareGPT4. These are protocol-aligned local extension "
    "results and are not paper-bit-exact targets."
)

EXTENSION_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = EXTENSION_ROOT.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
BASE_RUNNER = REPOSITORY_ROOT / "scripts/evaluate_long_context.py"
CONFIG_PATH = EXTENSION_ROOT / "config.yaml"
DEFAULT_FULL_OUTPUT = EXPERIMENT_ROOT / "runs/long_context_extension_full_k20_v1"
DEFAULT_SMOKE_OUTPUT = EXPERIMENT_ROOT / "runs/long_context_extension_smoke_k20_v1"
RUNS_ROOT = EXPERIMENT_ROOT / "runs"

SOURCE_MODEL = Path("/home/wby/models/Qwen3-32B")
TARGET_MODEL = Path("/home/wby/models/Qwen3-14B")
MAPPER_MANIFEST = (
    EXPERIMENT_ROOT
    / "artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01/manifest.json"
)
EXPECTED_MAPPER_MANIFEST_SHA256 = (
    "382a1deb6f4fd33c824fe782d956f46bdfb2b876081cb36487616e4de92a8ee2"
)
EXPECTED_MAPPER_FIT_FINGERPRINT = (
    "0aa81cb5816b56fe19c0349db7c54496751be6aa1918b8aff895f1e824ac2443"
)

MODES = ("target_standalone", "mapped")
TASKS = ("hotpotqa", "multi_news", "lmsys", "sharegpt4")
DIALOG_TASKS = ("lmsys", "sharegpt4")
EXPECTED_SAMPLE_COUNTS = {
    "hotpotqa": 7_405,
    "multi_news": 500,
    "lmsys": 500,
    "sharegpt4": 500,
}
EXPECTED_RESULT_COUNT = sum(EXPECTED_SAMPLE_COUNTS.values()) * len(MODES)
SMOKE_ENDPOINTS = {
    "hotpotqa": (
        {
            "evaluation_index": 3_469,
            "source_index": 3_469,
            "sample_id": "5adfff0755429925eb1afbce",
            "record_sha256": "8364949ab491c3dd3345c62d5329e7aa296e66408b9d088f895baa74f60726c1",
            "coverage_intent": (
                "maximum frozen HotpotQA source-content length and long-prompt "
                "EOS path"
            ),
        },
        {
            "evaluation_index": 645,
            "source_index": 645,
            "sample_id": "5a72f74a55429901807daf59",
            "record_sha256": "7aae54ae3820ecf298369935b59ecdd304cffa1db50f07c77ba9c99bc3c052cd",
            "coverage_intent": (
                "historical forward mapped 512-token cap hit while target "
                "stopped at EOS; cap-versus-EOS divergence risk"
            ),
        },
    ),
    "multi_news": (
        {
            "evaluation_index": 1_463,
            "source_index": 1_463,
            "sample_id": "multi-news-test-1463",
            "record_sha256": "75934b82ad707f22d6912233b12d0d0e81cf51a1c73568c3df3d91bee928628d",
            "coverage_intent": (
                "maximum frozen MultiNews source-content length; historical "
                "63184-token prompt exercises head-tail truncation"
            ),
        },
        {
            "evaluation_index": 2_938,
            "source_index": 2_938,
            "sample_id": "multi-news-test-2938",
            "record_sha256": "5c74a99d767081bd9b823d58f497c6dcfe407babd68546dfb2298ebf97930941",
            "coverage_intent": (
                "historical forward mapped 1024-token cap hit while target "
                "stopped at EOS; generation-cap path"
            ),
        },
    ),
    "lmsys": (
        {
            "evaluation_index": 125,
            "source_index": 85_754,
            "sample_id": "lmsys-shard-1-row-017403-2cc0b4753694",
            "record_sha256": "ea8b6e3d6dcf7fb552d1a33ead09f797e16eb5597b4cec5c812dbedbfcd31d15",
            "coverage_intent": (
                "maximum frozen LMSYS prompt length; historical mapped "
                "generation cap hit while target stopped at EOS"
            ),
        },
        {
            "evaluation_index": 370,
            "source_index": 202_632,
            "sample_id": "lmsys-shard-2-row-065930-abf126e8bae9",
            "record_sha256": "8bb9834e36dc1e073b4d962409d5b17e2709a3b2cf8e6da8296cb45943028dd9",
            "coverage_intent": (
                "historical 1024-token reference cap and target generation "
                "cap hit while mapped stopped at EOS"
            ),
        },
    ),
    "sharegpt4": (
        {
            "evaluation_index": 341,
            "source_index": 3_775,
            "sample_id": "sharegpt4-line-003775-5a1b3973400a",
            "record_sha256": "5e4c82d3739bc92320a6b4ce7a11fdf6298e7ed74bf0547e955fc87320ae1bbd",
            "coverage_intent": (
                "maximum frozen ShareGPT4 prompt length; historical dialogue "
                "message-drop truncation path"
            ),
        },
        {
            "evaluation_index": 435,
            "source_index": 5_095,
            "sample_id": "sharegpt4-line-005095-c64cfd413550",
            "record_sha256": "06dc5068f6cd7b20a6db6627418b847e706fa93599d0157f3b714054662282f6",
            "coverage_intent": (
                "historical 1024-token reference cap and generation cap hit "
                "in both target and mapped modes"
            ),
        },
    ),
}
SMOKE_SAMPLE_INDICES = {
    task: tuple(endpoint["evaluation_index"] for endpoint in endpoints)
    for task, endpoints in SMOKE_ENDPOINTS.items()
}
EXPECTED_SMOKE_RESULT_COUNT = sum(
    len(indices) for indices in SMOKE_SAMPLE_INDICES.values()
) * len(MODES)

# The user explicitly requested a generous HotpotQA cap for this reverse run.
# All tasks therefore share the same frozen cap, while total context remains
# bounded at 16,384 tokens for the eight 24-GiB GPU host.
EXPECTED_CAPS = {task: 1_024 for task in TASKS}
MAX_LENGTH = 16_384
DIALOG_REFERENCE_SCORE_CAP_TOKENS = 1_024
EXPECTED_TARGET_VOCAB_SIZE = 151_936
MAPPER_TOKEN_CHUNK = 1_024
ATTENTION_IMPLEMENTATION = "flash_attention_2"

FORMAL_TOPOLOGY = {
    "source": {
        "model": SOURCE_MODEL,
        "devices": (0, 1, 2, 3, 4),
        "layer_counts": (12, 13, 14, 13, 12),
    },
    "target": {
        "model": TARGET_MODEL,
        "devices": (5, 6, 7),
        "layer_counts": (13, 14, 13),
    },
}

MULTINEWS_SELECTION_SEED = (
    "qwen3-14b-to-32b-multinews-official-test-bottom500-v1"
)
LMSYS_SELECTION_SEED = "qwen3-14b-to-32b-lmsys-heldout-v1"
SHAREGPT4_SELECTION_SEED = "qwen3-14b-to-32b-sharegpt4-heldout-v1"

# Reusing the forward seeds and exact subsets makes direction comparisons
# paired.  The seed strings retain their historical names by design.
EXPECTED_DATA_IDENTITIES = {
    "hotpotqa": {
        "file_sha256": "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
        "sample_identity_sha256": "21e5a83eea0c95a19502eccd5c8886a67fb14e3983f205bc8254b8ecf6e5e0c8",
    },
    "multi_news": {
        "source_sha256": "138d3ac2dc899cbcd2e3745aaa94d1c1db55fb7058d9df4ba3ef2dac05a3a186",
        "target_sha256": "fa97cf91a62ae82a0af6da88f2ddf8e06eb4e3b90f7971d8e0c516436518fae3",
        "full_identity_sha256": "65a7d20c35aab4051b520a9cf6faeb08c2c7e193edf2a53bfc7965746239fa27",
        "selected_sha256": "47f460b9bb3c69a60cea1b1d44ea8a9a64fd33623d107f4129d48ca6c5e76076",
        "selected_indices_sha256": "d39b44894d60ad2fbbb3aaad96b619fcbfde833a982f60644305f109821f8510",
    },
    "lmsys": {
        "files_sha256": "5a82b44c3c574bba4e0f62fd937c1e4ed82e1ad93959ae16e5e5b78257d8eefb",
        "selected_sha256": "78a996e129f778ac9cd86a5ac9a2391c556d080d48970bd89650e6a0be0cb946",
    },
    "sharegpt4": {
        "file_sha256": "b59e417da0aa7b5f40dfbfc2af77b3d7fd250356e7cd1b0ff002ad0ffd048f43",
        "selected_sha256": "1c3d3e541a5c9c960f3371562dfe3c5e4a75e8a1f8d2278472c5174d4c7404ea",
    },
}

CODE_IDENTITY_FILES = (
    "long_context_extension/__init__.py",
    "long_context_extension/protocol.py",
    "long_context_extension/config.yaml",
    "long_context_extension/adapter.py",
    "long_context_extension/run.py",
    "long_context_extension/selection.py",
    "long_context_extension/metrics.py",
    "long_context_extension/freeze.py",
    "long_context_extension/validator.py",
    "long_context_extension/README.md",
)

REVERSE_RUNTIME_CODE_FILES = (
    "reverse_kv/__init__.py",
    "reverse_kv/protocol.py",
    "reverse_kv/provenance.py",
    "reverse_kv/eval_common.py",
    "reverse_kv/runtime.py",
    "reverse_kv/reverse_latency.py",
)

GPU_IDLE_RUNTIME = {
    "require_cuda_visible_devices": "0,1,2,3,4,5,6,7",
    "idle_memory_limit_mib": 64,
    "idle_consecutive_samples": 12,
    "idle_sample_interval_seconds": 5,
    "idle_timeout_seconds": 86_400,
}


def output_within_experiment(value: str | Path) -> Path:
    """Resolve an output and reject writes outside the reverse experiment."""

    lexical = Path(value)
    if not lexical.is_absolute():
        lexical = Path.cwd() / lexical
    # ``resolve`` erases symlink identity.  Reject every lexical component
    # first, including a dangling final link, so an apparently in-tree output
    # cannot alias a mapper, model, or another run directory.
    for component in (lexical, *lexical.parents):
        if component.is_symlink():
            raise ValueError("long-context output path contains a symlink")
    output = lexical.resolve()
    runs_root = RUNS_ROOT.resolve()
    try:
        relative = output.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError(
            "long-context output must remain below the reverse runs directory"
        ) from exc
    if not relative.parts:
        raise ValueError("the reverse runs root itself is not a valid output")
    return output


__all__ = [name for name in globals() if name.isupper()] + [
    "output_within_experiment"
]
