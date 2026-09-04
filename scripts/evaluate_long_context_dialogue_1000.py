#!/usr/bin/env python3
"""Run the frozen LMSYS/ShareGPT4 protocol on 1,000 samples per dataset.

The shared long-context runner deliberately pins the original dialogue pilot
to 200 samples.  This small, auditable entry point reuses that runner verbatim
and changes only the permitted sample count.  It also gives the extension a
distinct protocol name and binds this launcher into the run's code provenance.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER = ROOT / "scripts/evaluate_long_context.py"
DEFAULT_CONFIG = ROOT / "configs/eval_qwen3_14b_to_32b_dialogue_1000.yaml"
DIALOG_TASKS = ("lmsys", "sharegpt4")
SAMPLE_COUNT = 1000
PROTOCOL = "qwen3_14b_to_32b_dialogue_target_vs_mapped_bottom1000_v1"


def _load_base_runner() -> Any:
    module_name = "kvcache_transfer_evaluate_long_context_dialogue_1000_base"
    spec = importlib.util.spec_from_file_location(module_name, BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base long-context runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolve their defining module through sys.modules.
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
        values[0:0] = ["--tasks", ",".join(DIALOG_TASKS)]
    return values


def main(argv: Sequence[str] | None = None) -> int:
    base = _load_base_runner()
    original_validate = base.validate_configuration
    original_build_definition = base.build_run_definition

    def validate_dialogue_1000(
        raw: Mapping[str, Any], args: Any
    ) -> dict[str, Any]:
        if tuple(args.tasks) != DIALOG_TASKS:
            raise ValueError(
                "the dialogue-1000 entry point requires exactly --tasks "
                "lmsys,sharegpt4 in that order"
            )
        if args.limit is not None or args.sample_indices is not None:
            raise ValueError(
                "the dialogue-1000 formal entry point forbids --limit and "
                "--sample-indices; it always evaluates both frozen 1,000-sample sets"
            )
        data = raw.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("configuration requires a data mapping")
        observed_counts = (
            int(data.get("lmsys_sample_count", -1)),
            int(data.get("sharegpt_sample_count", -1)),
        )
        if observed_counts != (SAMPLE_COUNT, SAMPLE_COUNT):
            raise ValueError(
                f"dialogue-1000 requires exactly {SAMPLE_COUNT} LMSYS and "
                f"{SAMPLE_COUNT} ShareGPT4 samples, got {observed_counts}"
            )

        # Exercise every other frozen validation rule using the base runner's
        # original 200-sample contract, then restore only the requested counts
        # in the validated, private configuration object.
        compatible_raw = dict(raw)
        compatible_data = dict(data)
        compatible_data["lmsys_sample_count"] = 200
        compatible_data["sharegpt_sample_count"] = 200
        compatible_raw["data"] = compatible_data
        configuration = original_validate(compatible_raw, args)
        configuration["lmsys_sample_count"] = SAMPLE_COUNT
        configuration["sharegpt_sample_count"] = SAMPLE_COUNT
        return configuration

    def build_dialogue_1000_definition(**kwargs: Any) -> dict[str, Any]:
        definition = original_build_definition(**kwargs)
        launcher = Path(__file__).resolve()
        definition["protocol"] = PROTOCOL
        definition["dialogue_1000_extension"] = {
            "launcher": str(launcher),
            "launcher_sha256": base.sha256_file(launcher),
            "launcher_nbytes": launcher.stat().st_size,
            "sample_count_per_task": SAMPLE_COUNT,
            "tasks": list(DIALOG_TASKS),
            "base_protocol_reused_except_sample_count": True,
        }
        return definition

    base.validate_configuration = validate_dialogue_1000
    base.build_run_definition = build_dialogue_1000_definition
    launcher_relative = str(Path(__file__).resolve().relative_to(ROOT))
    if launcher_relative not in base.FORMAL_CODE_FILES:
        base.FORMAL_CODE_FILES = (*base.FORMAL_CODE_FILES, launcher_relative)
    return int(base.main(_normalized_argv(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
