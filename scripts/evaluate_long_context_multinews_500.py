#!/usr/bin/env python3
"""Evaluate a frozen 500-row MultiNews test subset in Target/Mapped modes.

The shared long-context runner loads the complete official 5,622-row test
split.  This isolated entry point deterministically selects a SHA256 bottom-k
subset without changing the shared runner that is bound to an active dialogue
experiment.  All inference, mapping, scoring, checkpoint, and manifest logic
continues to come from the shared runner.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "scripts/evaluate_long_context.py"
DEFAULT_CONFIG = ROOT / "configs/eval_qwen3_14b_to_32b_multinews_500.yaml"
TASKS = ("multi_news",)
SOURCE_ROWS = 5622
SAMPLE_COUNT = 500
SELECTION_SEED = "qwen3-14b-to-32b-multinews-official-test-bottom500-v1"
PROTOCOL = "qwen3_14b_to_32b_multinews_target_vs_mapped_bottom500_v1"
EXPECTED_SOURCE_SHA256 = "138d3ac2dc899cbcd2e3745aaa94d1c1db55fb7058d9df4ba3ef2dac05a3a186"
EXPECTED_TARGET_SHA256 = "fa97cf91a62ae82a0af6da88f2ddf8e06eb4e3b90f7971d8e0c516436518fae3"
EXPECTED_FULL_IDENTITY_SHA256 = "65a7d20c35aab4051b520a9cf6faeb08c2c7e193edf2a53bfc7965746239fa27"
EXPECTED_SELECTED_SHA256 = "47f460b9bb3c69a60cea1b1d44ea8a9a64fd33623d107f4129d48ca6c5e76076"
EXPECTED_SELECTED_INDICES_SHA256 = "d39b44894d60ad2fbbb3aaad96b619fcbfde833a982f60644305f109821f8510"


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_rank(sample: Any, *, seed: str = SELECTION_SEED) -> str:
    # The rank is intentionally independent of the gold summary.  The complete
    # record hash remains in the selected manifest only as post-selection data
    # identity and checkpoint provenance.
    identity = f"{seed}\0{int(sample.index)}\0{sample.sample_id}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def select_multinews_bottom_k(
    samples: Sequence[Any],
    manifest: Mapping[str, Any],
    *,
    sample_count: int = SAMPLE_COUNT,
    seed: str = SELECTION_SEED,
    enforce_frozen_identity: bool = True,
) -> tuple[list[Any], dict[str, Any]]:
    """Select and describe a deterministic subset of the official test split."""

    if len(samples) != SOURCE_ROWS or manifest.get("rows") != SOURCE_ROWS:
        raise ValueError(
            f"MultiNews source must contain exactly {SOURCE_ROWS} rows, got "
            f"{len(samples)}/{manifest.get('rows')}"
        )
    if sample_count <= 0 or sample_count > SOURCE_ROWS:
        raise ValueError(f"invalid MultiNews sample_count={sample_count}")
    observed_indices = [int(sample.index) for sample in samples]
    if observed_indices != list(range(SOURCE_ROWS)):
        raise ValueError("MultiNews source indices are not the canonical 0..5621 order")
    sample_ids = [str(sample.sample_id) for sample in samples]
    if len(set(sample_ids)) != SOURCE_ROWS:
        raise ValueError("MultiNews source sample IDs are not unique")

    ranked = [
        (_selection_rank(sample, seed=seed), int(sample.index), sample)
        for sample in samples
    ]
    chosen_by_rank = sorted(ranked, key=lambda item: (item[0], item[1]))[:sample_count]
    # Execute in canonical source order while retaining the random-like frozen
    # subset.  This makes checkpoint names stable and easy to cross-reference.
    chosen = sorted(chosen_by_rank, key=lambda item: item[1])
    selected_records = [
        {
            "execution_position": position,
            "source_index": source_index,
            "sample_id": str(sample.sample_id),
            "record_sha256": str(sample.record_sha256),
            "selection_rank_sha256": rank,
        }
        for position, (rank, source_index, sample) in enumerate(chosen)
    ]
    selected_samples = [sample for _, _, sample in chosen]
    selected_sha256 = _canonical_digest(selected_records)
    selected_indices_sha256 = _canonical_digest(
        [record["source_index"] for record in selected_records]
    )
    if enforce_frozen_identity:
        source = manifest.get("source")
        target = manifest.get("target")
        if not isinstance(source, Mapping) or source.get("sha256") != EXPECTED_SOURCE_SHA256:
            raise ValueError("MultiNews source file does not match the frozen SHA256")
        if not isinstance(target, Mapping) or target.get("sha256") != EXPECTED_TARGET_SHA256:
            raise ValueError("MultiNews target file does not match the frozen SHA256")
        if manifest.get("sample_identity_sha256") != EXPECTED_FULL_IDENTITY_SHA256:
            raise ValueError("MultiNews full sample identity does not match the frozen digest")
        if selected_sha256 != EXPECTED_SELECTED_SHA256:
            raise ValueError("MultiNews selected records do not match the frozen digest")
        if selected_indices_sha256 != EXPECTED_SELECTED_INDICES_SHA256:
            raise ValueError("MultiNews selected indices do not match the frozen digest")
    updated_manifest = dict(manifest)
    updated_manifest["formal_selection"] = {
        "name": "sha256_bottom_k",
        "seed": seed,
        "source_split": "official_test",
        "source_rows": SOURCE_ROWS,
        "selected_rows": sample_count,
        "rank_identity": "seed\\0source_index\\0sample_id (gold-independent)",
        "execution_order": "ascending_source_index",
        "selected": selected_records,
        "selected_sha256": selected_sha256,
        "selected_indices_sha256": selected_indices_sha256,
    }
    return selected_samples, updated_manifest


def rouge_l_f1_without_baseexception(
    prediction: str, references: Sequence[str]
) -> float:
    """Pinned ROUGE-L while allowing interrupts/system failures to propagate."""

    if not references:
        raise ValueError("ROUGE sample has no references")
    try:
        from rouge import Rouge
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("rouge==1.0.1 is required for this evaluation") from exc
    scorer = Rouge()
    scores: list[float] = []
    for reference in references:
        try:
            score = scorer.get_scores([prediction], [reference], avg=True)
            scores.append(float(score["rouge-l"]["f"]))
        except ValueError:
            # rouge==1.0.1 raises ValueError for an empty hypothesis/reference.
            # Treat that documented malformed-input case as zero, but allow
            # runtime/system failures to fail the run instead of becoming a
            # silent quality score.
            scores.append(0.0)
    return max(scores)


def _load_base_runner() -> Any:
    module_name = "kvcache_transfer_evaluate_multinews_500_base"
    spec = importlib.util.spec_from_file_location(module_name, BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base long-context runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _has_option(argv: Sequence[str], option: str) -> bool:
    return option in argv or any(value.startswith(f"{option}=") for value in argv)


def _normalized_argv(argv: Sequence[str] | None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(values, "--config"):
        values[0:0] = ["--config", str(DEFAULT_CONFIG)]
    if not _has_option(values, "--tasks"):
        values[0:0] = ["--tasks", "multi_news"]
    return values


def _validate_extension_request(raw: Mapping[str, Any], args: Any) -> None:
    if tuple(args.tasks) != TASKS:
        raise ValueError("the MultiNews-500 entry point requires exactly --tasks multi_news")
    if args.limit is not None or args.sample_indices is not None:
        raise ValueError(
            "the MultiNews-500 formal entry point forbids --limit and "
            "--sample-indices; its SHA256 bottom-500 subset is fixed"
        )
    data = raw.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("configuration requires a data mapping")
    if int(data.get("multi_news_sample_count", -1)) != SAMPLE_COUNT:
        raise ValueError(f"MultiNews formal sample count must be {SAMPLE_COUNT}")
    if str(data.get("multi_news_selection_seed", "")) != SELECTION_SEED:
        raise ValueError("MultiNews selection seed does not match the frozen protocol")


def main(argv: Sequence[str] | None = None) -> int:
    base = _load_base_runner()
    original_validate = base.validate_configuration
    original_loader = base.load_multi_news_samples
    original_build_definition = base.build_run_definition

    def validate_multinews_500(
        raw: Mapping[str, Any], args: Any
    ) -> dict[str, Any]:
        _validate_extension_request(raw, args)
        # The two extension-only keys are not needed by the shared validator.
        compatible_raw = dict(raw)
        compatible_data = dict(raw["data"])
        compatible_data.pop("multi_news_sample_count", None)
        compatible_data.pop("multi_news_selection_seed", None)
        compatible_raw["data"] = compatible_data
        configuration = original_validate(compatible_raw, args)
        configuration["multi_news_sample_count"] = SAMPLE_COUNT
        configuration["multi_news_selection_seed"] = SELECTION_SEED
        return configuration

    def load_multinews_500(
        source_path: str | Path, target_path: str | Path
    ) -> tuple[list[Any], dict[str, Any]]:
        samples, manifest = original_loader(source_path, target_path)
        return select_multinews_bottom_k(samples, manifest)

    def build_multinews_500_definition(**kwargs: Any) -> dict[str, Any]:
        definition = original_build_definition(**kwargs)
        dataset_manifest = kwargs["dataset_manifest"]
        selection = dataset_manifest["multi_news"]["formal_selection"]
        launcher = Path(__file__).resolve()
        definition["protocol"] = PROTOCOL
        definition["sample_scope"] = "formal_sha256_bottom500_of_official_test5622"
        definition["multinews_500_extension"] = {
            "launcher": str(launcher),
            "launcher_sha256": base.sha256_file(launcher),
            "launcher_nbytes": launcher.stat().st_size,
            "source_rows": SOURCE_ROWS,
            "selected_rows": SAMPLE_COUNT,
            "selection_seed": SELECTION_SEED,
            "selected_sha256": selection["selected_sha256"],
            "selected_indices_sha256": selection["selected_indices_sha256"],
            "rouge_exception_policy": (
                "empty-input ValueError becomes zero; all other failures propagate"
            ),
            "base_inference_mapping_scoring_checkpoint_logic_reused": True,
        }
        return definition

    base.validate_configuration = validate_multinews_500
    base.load_multi_news_samples = load_multinews_500
    base.build_run_definition = build_multinews_500_definition
    base.rouge_l_f1 = rouge_l_f1_without_baseexception
    launcher_relative = str(Path(__file__).resolve().relative_to(ROOT))
    if launcher_relative not in base.FORMAL_CODE_FILES:
        base.FORMAL_CODE_FILES = (*base.FORMAL_CODE_FILES, launcher_relative)
    return int(base.main(_normalized_argv(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
