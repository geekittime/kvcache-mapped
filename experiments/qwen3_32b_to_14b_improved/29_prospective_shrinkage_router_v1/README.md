# 29 — prospective shrinkage router (CPU-only, pre-registered)

This is a new, independent router contract.  It does not modify any earlier
experiment and it does not load a model or start a GPU process.  The contract
decides whether a completed NVIDIA k20 pass is allowed to use the shrinkage
mapper.  The router is deliberately conservative for HotpotQA: both a long
prompt and a long completed NVIDIA/R1 generation are required.

The package contains:

* `src/router.py` — a deterministic, gold-free batch router;
* `src/evaluate.py` — a locked three-arm (native14/NVIDIA/shrinkage) CPU
  evaluator and paired bootstrap gates;
* `src/registry.py` — a fresh-registry split and tokenizer contract;
* `PROTOCOL.md` and `METHOD.md` — the pre-registration and claim boundary;
* `configs/router.json` — the frozen thresholds, budget, and promotion gates;
* `scripts/preflight.py` — no-GPU contract checks;
* `tests/` — synthetic tests only (no model or answer files are read).

The old `03_tail_refresh` pilot and completed `28_shrinkage_old_pilot_eval_v1`
run are immutable, read-only hypothesis evidence.  They are bound by path,
size, and SHA-256 in the CPU receipt; they are not a locked test.  In
particular, the old pilot contains almost no long-prompt HotpotQA examples, so
it cannot validate a Hotpot-specific router.

## Frozen policy

After the first NVIDIA/R1 pass has completed, the serving layer supplies only
`prompt_tokens`, `r1_generated_tokens`, and a transport completion bit.  If
`prompt_tokens >= 2800` **and** `r1_generated_tokens >= 16`, the request is an
eligible `shrinkage_r1` escalation.  All other valid requests remain on
`nvidia_k20`.  Invalid or incomplete telemetry always falls back to NVIDIA and
is recorded as `no_go`.  A batch may escalate at most
`min(32, floor(0.30 * batch_size))` requests, with stable original-order tie
breaking.  No output text, answer, quality metric, wall time, GPU state,
dataset label, cap flag, or hidden score is accepted by the router.

The prospective registry must be frozen before execution and split into a
non-evaluative calibration partition and a sealed evaluation partition.  The
partitions must be disjoint by sample ID and use the exact pinned tokenizer.
Thresholds and gates cannot be changed after any sealed outcome is observed.

## CPU checks

```bash
ROOT=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/29_prospective_shrinkage_router_v1
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
cd "$ROOT"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/router29_cache.XXXXXX)" \
  "$PY" -m pytest -q -p no:cacheprovider tests
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="$(mktemp -d /tmp/router29_pf.XXXXXX)" \
  "$PY" scripts/preflight.py
```

The temporary cache directories must be removed after each command.  A
successful CPU preflight is an implementation receipt only; it is not evidence
that the router improves accuracy or beats the NVIDIA paper.
