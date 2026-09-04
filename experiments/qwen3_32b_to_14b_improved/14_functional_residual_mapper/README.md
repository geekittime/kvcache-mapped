# 14 — functional residual mapper (CPU pre-formal)

Status: **PRE-FORMAL / UNTRAINED**. This directory contains an independent,
CPU-tested framework for a next Qwen3-32B → Qwen3-14B mapper candidate. It has
not loaded either model, has not run on a GPU, has not opened a locked/test
outcome, and has not produced comparative accuracy evidence.

The candidate freezes the content-addressed head-wise shrinkage mapper as its
anchor and adds a zero-initialized, layer/KV/head-structured low-rank residual.
Its primary targets are native-14B fixed-query attention behavior, 8→32 step
native-14B suffix KL, and EOS behavior. KV reconstruction is a guardrail, not
the optimization endpoint. Dynamic OOD and residual-norm guards recover the
anchor when the residual is unsafe.

Only an answer-free train-only oracle/method-dev screen may choose a nonempty
subset of layers 7–10 and 34–35, their functional-loss weights, and one rank
from the registered 64→128→256 sequence. Parameter fitting is mapper-fit only.
The pure R=1 comparison has no tail refresh; optional R=32 is a separately
reported hybrid system and cannot rescue the R=1 claim.

Scalar attention-logit calibration and saliency-weighted ridge (using native
`Q^TQ`/attention read weights) are recorded as separate future-v2 ablations;
they are deliberately not mixed into this candidate or its pre-formal receipt.

Files:

- `METHOD.md`: equations, parameter/VRAM accounting, and research rationale;
- `PROTOCOL.md`: data isolation, staged decisions, gates, and claim boundary;
- `config.template.json`: deliberately blocked, non-GPU template;
- `src/`: residual, functional losses, tier gate, trainer, checkpoint, and
  preflight interfaces;
- `tests/`: CPU synthetic and fail-closed tests;
- `artifacts/preformal/`: immutable implementation inventory and STOP receipt.

CPU commands use the repository environment explicitly:

```bash
ROOT=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/14_functional_residual_mapper
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python

cd "$ROOT"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest -q tests
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/preflight.py \
  --config config.template.json --expect-template-blocked
```

There is intentionally no authorized GPU command in this pre-formal version.
