#!/usr/bin/env python3
"""Validate the complete Qwen3 mapper on held-out BF16 KV activations.

No model is loaded.  The script first verifies all immutable manifests,
sidecars, payload hashes, shapes, dtypes, the 50x1024 stride-4 protocol, and
the train/held-out token split.  It then applies every target layer's fixed
production top-8 mapper in FP32 and computes stable streaming FP64 R2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.heldout_validation import (  # noqa: E402
    evaluate_heldout_mapper,
    validate_heldout_inputs,
)
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
)


DEFAULT_MAPPER = (
    ROOT / "artifacts/mappers/qwen3_14b_to_32b/k8_lambda0.01/manifest.json"
)
DEFAULT_SOURCE = ROOT / "data/activations/qwen3_14b/heldout"
DEFAULT_TARGET = ROOT / "data/activations/qwen3_32b/heldout"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/validation/qwen3_14b_to_32b/heldout_activation_reconstruction.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapper-manifest", type=Path, default=DEFAULT_MAPPER)
    parser.add_argument("--source-activations", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target-activations", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-tokens", type=int, default=256)
    parser.add_argument(
        "--compute-device",
        default="cpu",
        help="FP32 mapper matmul device (default: cpu; e.g. cuda:0 is optional)",
    )
    parser.add_argument(
        "--analysis-device",
        default="cpu",
        help="FP64 metric device (default: cpu; 'compute' reuses compute device)",
    )
    parser.add_argument("--cpu-threads", type=int, default=14)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify every input/hash without running mapper matmuls",
    )
    return parser.parse_args(argv)


def _finite_json(value: Any) -> None:
    """Refuse a result that Python's permissive encoder would write as NaN."""

    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("result JSON contains NaN or Inf")
    if isinstance(value, dict):
        for nested in value.values():
            _finite_json(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _finite_json(nested)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.chunk_tokens <= 0:
        raise ValueError("--chunk-tokens must be positive")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before it is used.
        pass

    run = make_run_directory(ROOT, "p4", "validate_mapper_heldout")
    started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
    record_run_start(
        run,
        list(sys.argv if argv is None else [sys.argv[0], *argv]),
        {
            "mapper_manifest": str(args.mapper_manifest.resolve()),
            "source_activations": str(args.source_activations.resolve()),
            "target_activations": str(args.target_activations.resolve()),
            "output": str(args.output.resolve()),
            "chunk_tokens": args.chunk_tokens,
            "compute_device": args.compute_device,
            "analysis_device": args.analysis_device,
            "cpu_threads": args.cpu_threads,
            "preflight_only": args.preflight_only,
            "formal_protocol": True,
        },
    )
    started = time.perf_counter()
    try:
        inputs = validate_heldout_inputs(
            args.mapper_manifest,
            args.source_activations,
            args.target_activations,
            verify_payload_checksums=True,
            formal_protocol=True,
        )
        preflight = {
            "schema_version": 1,
            "status": "passed",
            "mapper_manifest_sha256": inputs.mapper_manifest_sha256,
            "fit_fingerprint": inputs.fit_fingerprint,
            "selection_sha256": inputs.selection_sha256,
            "training_token_sha256": inputs.training_token_sha256,
            "heldout_token_sha256": inputs.heldout_token_sha256,
            "training_and_heldout_token_sha256_differ": (
                inputs.training_token_sha256 != inputs.heldout_token_sha256
            ),
            "source_capture_manifest_sha256": inputs.source_capture_manifest_sha256,
            "target_capture_manifest_sha256": inputs.target_capture_manifest_sha256,
            "source_capture_inventory_sha256": inputs.source_capture_inventory_sha256,
            "target_capture_inventory_sha256": inputs.target_capture_inventory_sha256,
            "target_layers": len(inputs.layers),
            "observations": inputs.heldout_samples,
            "protocol": {
                "split": "heldout",
                "sequences": inputs.heldout_sequences,
                "sequence_length": inputs.sequence_length,
                "stride": inputs.stride,
            },
        }
        atomic_write_json(run.path / "preflight.json", preflight)
        if args.preflight_only:
            result = {
                **preflight,
                "status": "preflight-complete",
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_write_json(run.path / "result.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        result = evaluate_heldout_mapper(
            inputs,
            chunk_tokens=args.chunk_tokens,
            compute_device=args.compute_device,
            analysis_device=args.analysis_device,
        )
        result["started_utc"] = started_utc
        result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        _finite_json(result)
        output = args.output.resolve()
        atomic_write_json(output, result)
        output_sha = sha256_file(output)
        atomic_write_json(
            output.with_suffix(".sha256.json"),
            {"path": str(output), "sha256": output_sha},
        )
        run_result = {
            "schema_version": 1,
            "status": "complete",
            "output": str(output),
            "output_sha256": output_sha,
            "summary": result["summary"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(run.path / "result.json", run_result)
        summary = result["summary"]
        append_reproduction_log(
            ROOT,
            heading=f"P4-HELDOUT-{run.run_id}",
            bullets=[
                "状态：completed / passed；使用未参与选层或 ridge 拟合的 "
                "50×1024 held-out calibration，stride=4，共 12800 observations。",
                "逐 target layer 固定使用 production top-8；FP32 执行 X@W+b，"
                "FP64 streaming Welford 计算逐 channel R² 后等权平均。",
                "held-out uniform-channel R²（64 层等权）："
                f"K={summary['key_uniform_channel_r2_fp64']['mean_across_target_layers']:.10f}，"
                f"V={summary['value_uniform_channel_r2_fp64']['mean_across_target_layers']:.10f}。",
                f"结果：`{output.relative_to(ROOT)}`，SHA256={output_sha}；run=`{run.path.relative_to(ROOT)}`。",
                "该 held-out R² 是本地泛化诊断，不冒充论文 benchmark 指标。",
            ],
        )
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(output),
                    "output_sha256": output_sha,
                    "summary": summary,
                    "run": str(run.path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except BaseException as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "started_utc": started_utc,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_write_json(run.path / "result.json", failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
