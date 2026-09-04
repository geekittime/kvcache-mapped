#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT))

from src.common import ContractError, artifact_ref, atomic_write_json, load_json_strict, sha256_file  # noqa: E402
from src.preflight import validate_template  # noqa: E402


def _inventory() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    excluded = {"__pycache__", ".pytest_cache", ".mypy_cache"}
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in excluded for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[:2] == ("artifacts", "preformal"):
            continue
        entries.append({"path": str(relative), "nbytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish CPU-only pre-formal STOP receipt")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        print("BLOCKED: publish requires CUDA_VISIBLE_DEVICES=''", file=sys.stderr)
        return 2
    try:
        preflight = validate_template(ROOT / "config.template.json")
    except (ContractError, OSError, ValueError) as exc:
        print(f"BLOCKED: template preflight failed: {exc}", file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    test_output = (completed.stdout + ("\n" + completed.stderr if completed.stderr else "")).strip()
    if completed.returncode != 0:
        print(test_output, file=sys.stderr)
        print(f"BLOCKED: CPU tests returned {completed.returncode}", file=sys.stderr)
        return 3

    output_dir = ROOT / "artifacts" / "preformal"
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / f"PRE_FORMAL_STOP_{args.version}.json"
    sidecar_path = Path(str(receipt_path) + ".sha256.json")
    if receipt_path.exists() or sidecar_path.exists():
        print(f"BLOCKED: refusing to overwrite {receipt_path}", file=sys.stderr)
        return 2
    config_ref = artifact_ref(ROOT / "config.template.json")
    receipt = {
        "schema_version": 1,
        "kind": "functional_residual_pre_formal_stop_v1",
        "status": "PRE_FORMAL_STOP",
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_root": str(ROOT),
        "config": config_ref,
        "preflight": preflight,
        "cpu_tests": {
            "command": command,
            "returncode": completed.returncode,
            "output": test_output,
            "cuda_visible_devices": "",
            "gpu_execution_performed": False,
            "models_loaded": False,
        },
        "implementation_inventory": _inventory(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cuda_visible_devices": "",
        },
        "locked_outcomes_opened": False,
        "comparative_claim_permitted": False,
        "unfinished_boundaries": [
            "real_mapper_fit_functional_capture",
            "answer_free_train_only_layer_oracle_receipt",
            "complete_G0_v2_and_decontaminated_registry_binding",
            "differentiable_frozen_14B_receiver_adapter",
            "GPU_memory_and_latency_preflight",
            "GPU_training_and_checkpoint",
            "prospective_R1_method_dev_and_locked_test",
            "optional_R32_hybrid_evaluation",
        ],
        "research_references": [
            "https://arxiv.org/abs/2608.03893",
            "https://arxiv.org/abs/2607.28979",
            "https://arxiv.org/abs/2405.16444",
            "https://arxiv.org/abs/2608.21362",
            "https://arxiv.org/abs/2604.13226",
            "https://arxiv.org/abs/2607.21599",
        ],
    }
    atomic_write_json(receipt_path, receipt)
    atomic_write_json(
        sidecar_path,
        {"path": str(receipt_path), "nbytes": receipt_path.stat().st_size, "sha256": sha256_file(receipt_path)},
    )
    print(f"published {receipt_path}")
    print(f"receipt sha256 {sha256_file(receipt_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

