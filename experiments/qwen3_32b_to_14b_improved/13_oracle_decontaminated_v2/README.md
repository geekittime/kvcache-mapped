# 13 — decontaminated causal oracle v2

Status: **PRE-FORMAL STOP; no GPU/model execution performed**.

This independent directory contains the fail-closed framework for a small
HotpotQA-train/MuSiQue-train causal diagnostic of the frozen Qwen3-32B → 14B
NVIDIA k20 mapper. It does not reuse the contaminated dir01 configuration or
prepared records. Four low-level dir01 causal-cache interfaces are referenced
read-only and pinned by byte count and SHA-256; all outputs are independent.

The full preregistration is in `PROTOCOL.md`. The catalog covers native K/V
2×2, four ten-layer native repairs, layer 34, layer 35, and layers 34+35. Each
branch reports first-token KL, gold margin, fixed-native-query attention-output
cosine (plus NMSE/JS), and QA F1/EM at the exact `C[:-1]` / `C[-1]` boundary.

The G0 consumer is schema-aligned with the frozen dir12 contract and dir09-v2
producer: `candidate_registry_binding` has exactly three top-level artifact
references plus nested `split_artifacts` exact4, and `comparisons` has exactly
the dataset-specific HotpotQA/MuSiQue × mapper-fit/locked-eval directions.
Every direction must be exactly `{"flags":0}` with no extra field. Dir13 still
directly verifies dir10 receipt `train_sources`, `answer_access_policy`, and
answer-free `projection`; G0 schema alignment does not weaken those gates.
Every content-addressed input is walked with `lstat` before resolution, so a
live or dangling symlink anywhere in its path is rejected rather than hidden
by `resolve()`. The answer-access object has one exact key set and typed value
shared by code, the checked-in template, and the machine consumer contract;
extra policy switches or coordinated config/receipt drift fail closed.

## Why it cannot run yet

`config.template.json` intentionally contains finalization placeholders for:

- the final dir09 complete-G0-v2 receipt;
- final dir10 PUBLISHED/manifest/receipt and all four native split files;
- pinned HotpotQA/MuSiQue train sources;
- final projection algorithm and registry strata order;
- smoke and diagnostic sample counts.

It also sets `gpu_execution_authorized_by_this_config=false`. CPU preflight
therefore fails before any registry/source is opened, before torch is imported,
and before CUDA is called. The old failed/partial G0 receipts are unacceptable.

## Commands

The following checked-in command should report a blocked template:

```bash
CUDA_VISIBLE_DEVICES='' /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/preflight.py --profile smoke
```

Generic CPU tests use only synthetic train data and tampered mock artifacts:

```bash
CUDA_VISIBLE_DEVICES='' /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python -m pytest -q tests
```

Only after a separately audited formal config is frozen and explicitly
authorized would the concrete runner be called:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/run_gpu_oracle.py \
  --config config.formal.json --profile smoke --output-dir runs/smoke_v1 --execute
```

That runner enforces the global lock, 12-sample idle gate, 5+3 model sharding,
deep pre-torch model/mapper hashes, one k20 materialization per prompt, exact
cache envelopes, atomic resume, and a hash-bound completion receipt.

## Layout

- `config.template.json`: deliberately blocked pre-formal configuration;
- `g0_consumer_contract.json`: machine requirements for final dir09;
- `native_registry_consumer_contract.json`: machine requirements for dir10;
- `src/contracts.py`: CPU-only G0/registry/train reconstruction gates;
- `src/gpu_runner.py`: real but currently unauthorized eight-GPU runner;
- `src/validator.py`: independent atomic-result and completion validator;
- `tests/`: synthetic/mock and tamper tests; no model or CUDA use;
- `artifacts/preformal/`: versioned implementation inventories published at each stop;
- `runs/`: independent future run outputs only.
