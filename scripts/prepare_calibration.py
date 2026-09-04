#!/usr/bin/env python3
"""Prepare the pinned FineWeb-Edu calibration tokens.

The default YAML is the formal Qwen3 reproduction configuration.  Local
JSONL/Parquet overrides exist for smoke tests and debugging; they are recorded
in the resulting manifest and cannot be mistaken for the formal HF source.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.data import (  # noqa: E402
    CalibrationDataError,
    iter_source_records,
    prepare_calibration_dataset,
    sha256_file,
    tokenizer_provenance_from_path,
    verify_calibration_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic 500+50 FineWeb-Edu token windows."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/calibration_qwen3.yaml",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source-jsonl",
        type=Path,
        nargs="+",
        help="debug-only local JSONL source override",
    )
    source_group.add_argument(
        "--source-parquet",
        type=Path,
        nargs="+",
        help="debug-only local Parquet source override",
    )
    parser.add_argument(
        "--formal-local-mirror",
        type=Path,
        help=(
            "read this local Parquet as a mirror of the pinned HF shard; its "
            "full SHA256 and size must match formal config"
        ),
    )
    parser.add_argument(
        "--hf-streaming",
        action="store_true",
        help="ignore configured local mirror and stream the same pinned HF shard",
    )
    parser.add_argument("--output", type=Path, help="override output directory")
    parser.add_argument("--tokenizer", type=Path, help="override tokenizer path")
    parser.add_argument(
        "--checkpoint-every", type=int, help="records between atomic checkpoints"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        help="debug-only cap; becomes part of the output fingerprint",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an existing output without opening the data source",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def local_override_source(kind: str, paths: list[Path]) -> dict[str, Any]:
    resolved = [path.resolve() for path in paths]
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "type": kind,
        "paths": [str(path) for path in resolved],
        "expected_sha256": {str(path): sha256_file(path) for path in resolved},
        "provenance_status": "debug-only local CLI override; not paper calibration",
    }


def validate_formal_protocol(config: Mapping[str, Any]) -> None:
    protocol = config["paper_protocol"]
    expected = {
        "dataset": "FineWeb-Edu",
        "train_sequences": 500,
        "sequence_length": 1024,
        "activation_token_stride": 4,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"paper_protocol.{key} must be {value!r}, "
                f"got {protocol.get(key)!r}"
            )
    selection = config["selection"]
    if selection.get("add_special_tokens") is not False:
        raise ValueError("selection.add_special_tokens must be exactly false")
    if int(selection.get("train_count", -1)) != 500:
        raise ValueError("selection.train_count must be 500")
    if int(selection.get("sequence_length", -1)) != 1024:
        raise ValueError("selection.sequence_length must be 1024")
    if int(selection.get("stride", -1)) != 4:
        raise ValueError("selection.stride must be 4")


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    validate_formal_protocol(config)

    output_value = args.output or config["output"]["directory"]
    output = resolve_project_path(output_value)
    if args.verify_only:
        manifest = verify_calibration_output(output)
        print(
            json.dumps(
                {
                    "status": "verified",
                    "output": str(output),
                    "dataset_fingerprint": manifest["dataset_fingerprint"],
                    "splits": manifest["splits"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    source = dict(config["source"])
    local_conventions = list(config["local_conventions"])
    if args.hf_streaming and (
        args.source_jsonl or args.source_parquet or args.formal_local_mirror
    ):
        raise ValueError(
            "--hf-streaming cannot be combined with another source override"
        )
    if args.source_jsonl:
        source = local_override_source("jsonl", args.source_jsonl)
        local_conventions.append(
            "This run used --source-jsonl and is a debug run, not the pinned "
            "FineWeb-Edu calibration source."
        )
    elif args.source_parquet:
        source = local_override_source("parquet", args.source_parquet)
        local_conventions.append(
            "This run used --source-parquet and is a debug run, not the pinned "
            "FineWeb-Edu calibration source."
        )
    if args.formal_local_mirror:
        if args.source_jsonl or args.source_parquet:
            raise ValueError(
                "--formal-local-mirror cannot be combined with a debug source override"
            )
        source["local_mirror_path"] = str(args.formal_local_mirror.resolve())
    elif args.hf_streaming:
        source.pop("local_mirror_path", None)
    elif source.get("local_mirror_path"):
        source["local_mirror_path"] = str(
            resolve_project_path(source["local_mirror_path"])
        )

    tokenizer_config = dict(config["tokenizer"])
    tokenizer_path = resolve_project_path(args.tokenizer or tokenizer_config["path"])
    provenance = tokenizer_provenance_from_path(tokenizer_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=bool(tokenizer_config.get("local_files_only", True)),
        trust_remote_code=bool(tokenizer_config.get("trust_remote_code", False)),
        use_fast=bool(tokenizer_config.get("use_fast", True)),
    )
    provenance.update(
        {
            "class": type(tokenizer).__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "add_special_tokens": False,
        }
    )

    selection = config["selection"]
    checkpoint_every = int(
        args.checkpoint_every
        if args.checkpoint_every is not None
        else config["output"].get("checkpoint_every_records", 10_000)
    )

    def report_progress(stats: Mapping[str, int]) -> None:
        print(json.dumps({"progress": stats}, sort_keys=True), file=sys.stderr)

    manifest = prepare_calibration_dataset(
        iter_source_records(source, base_dir=PROJECT_ROOT),
        tokenizer,
        output,
        train_count=int(selection["train_count"]),
        heldout_count=int(selection["heldout_count"]),
        sequence_length=int(selection["sequence_length"]),
        stride=int(selection["stride"]),
        seed=str(selection["seed"]),
        text_field=str(selection["text_field"]),
        identity_fields=tuple(selection["identity_fields"]),
        add_special_tokens=bool(selection["add_special_tokens"]),
        source_provenance=source,
        tokenizer_provenance=provenance,
        local_conventions=local_conventions,
        checkpoint_every=checkpoint_every,
        max_records=args.max_records,
        progress=report_progress,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "config_fingerprint": manifest["config_fingerprint"],
                "stats": manifest["stats"],
                "splits": manifest["splits"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CalibrationDataError, KeyError, ValueError, FileNotFoundError) as error:
        print(f"calibration preparation failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
