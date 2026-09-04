#!/usr/bin/env python3
"""Record the Phase-0 CUDA, dependency, model identity, and topology gate."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.config import assert_paper_architecture  # noqa: E402
from kvcache_transfer.io import atomic_write_json, sha256_file  # noqa: E402
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
)


MODELS = [Path("/home/wby/models/Qwen3-14B"), Path("/home/wby/models/Qwen3-32B")]


def main() -> int:
    run = make_run_directory(ROOT, "p0", "environment_smoke")
    record_run_start(run, sys.argv, {})
    model_records = []
    for model_index, model_path in enumerate(MODELS):
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        role = "source" if model_index == 0 else "target"
        assert_paper_architecture(config, role=role)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        incomplete = list(model_path.rglob("*.incomplete"))
        if incomplete:
            raise RuntimeError(f"Incomplete model files under {model_path}: {incomplete}")
        model_records.append(
            {
                "path": str(model_path),
                "layers": config.num_hidden_layers,
                "hidden_size": config.hidden_size,
                "attention_heads": config.num_attention_heads,
                "kv_heads": config.num_key_value_heads,
                "head_dim": config.head_dim,
                "tokenizer_size": len(tokenizer),
                "tokenizer_sha256": sha256_file(model_path / "tokenizer.json"),
            }
        )
    if model_records[0]["tokenizer_sha256"] != model_records[1]["tokenizer_sha256"]:
        raise RuntimeError("Source and target tokenizer hashes differ")

    cuda_records = []
    for index in range(torch.cuda.device_count()):
        device = torch.device("cuda", index)
        left = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
        right = torch.randn(64, 64, device=device, dtype=torch.bfloat16)
        value = left @ right
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite BF16 matmul on GPU {index}")
        cuda_records.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": torch.cuda.get_device_capability(index),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    if len(cuda_records) != 8:
        raise RuntimeError(f"Expected 8 CUDA devices, observed {len(cuda_records)}")
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], text=True, check=True, capture_output=True
    ).stdout
    (run.path / "gpu_topology.txt").write_text(topology, encoding="utf-8")
    metrics = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "models": model_records,
        "cuda_devices": cuda_records,
        "status": "passed",
    }
    atomic_write_json(run.path / "metrics.json", metrics)
    append_reproduction_log(
        ROOT,
        heading=f"ENV-001：Phase 0 环境与模型身份检查（{run.run_id}）",
        bullets=[
            "状态：completed / passed。",
            f"项目隔离环境使用 torch {torch.__version__}、CUDA {torch.version.cuda}、transformers {transformers.__version__}。",
            "8 张 RTX 4090 均通过 BF16 matmul smoke；GPU topology 已写入 run 目录。",
            "Qwen3-14B/32B config、层数、matched-KV 结构及 tokenizer SHA256 均通过断言；无 incomplete shard。",
            f"运行产物：`runs/{run.run_id}/`。",
        ],
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
