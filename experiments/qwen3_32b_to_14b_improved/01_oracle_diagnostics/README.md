# A1 exact/mapped cache oracle diagnostics

This directory is the isolated CPU-prepared harness for causal diagnostics of
the frozen Qwen3-32B → Qwen3-14B k20 cache mapper. It does not train or publish
a mapper, does not load either model during preparation/preflight, and does not
authorize a GPU run.

## Current status

- implementation and CPU tests: complete;
- HotpotQA **train** deterministic screen: 256 complete questions;
- MuSiQue **train-screen** frozen registry: 256 complete questions;
- prepared paired plan: 512 prompts × 11 interventions = 5,632 rows;
- maximum serialized prompt length: 4,375 tokens, so no prompt was truncated;
- generation policy: greedy, no thinking, identical prompt IDs, cap 256;
- Hotpot/MuSiQue selected normalized-question overlap: zero;
- dev/validation/test/locked outcomes opened: no;
- GPU/model execution: not performed.

## v2 concrete GPU runner (still not authorized)

`config_v2.json` and `src/gpu_oracle.py` add the real eight-GPU execution
path. It only selects train-only records in two bounded profiles: `smoke`
(2 HotpotQA + 2 MuSiQue) and `screen` (64 + 64); it refuses the old 5,632-row
plan. It loads 32B over GPUs 0--4 and 14B over GPUs 5--7, materializes k20
once, and obtains both caches from the identical token-vector prefix `C[:-1]`.
It verifies full `C` against the prefix-plus-boundary native branch before the
2x2 K/V and seven layer-replacement conditions.

The runner records native post-`q_norm`/`k_norm`+RoPE boundary Q/K/V, checks
the captured K/V against the actual target cache, uses native fixed-Q FP32 GQA
diagnostics, and keeps per-intervention generated token IDs, EOS/cap reason,
token hashes, F1/EM, margins, KL, JS, cosine and NMSE. One atomic result is
checkpointed per prompt and revalidated before it can be reused.

There is deliberately no current GPU authorization: v2 requires an external
*complete* G0 closure receipt with exact byte/hash identity. The partial
cross-tier audit is not treated as closure, so the CPU command reports
`blocked_missing_complete_g0` without loading models or calling CUDA.

```bash
CUDA_VISIBLE_DEVICES='' /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/run_gpu_oracle.py --profile smoke --preflight-only
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/run_gpu_oracle.py --profile smoke --output-dir runs/oracle_v2_smoke --execute
```

The latter intentionally refuses until the independently published complete
G0 receipt is installed and pinned in `config_v2.json`; after that it takes
the repository-wide eight-GPU lock and waits for twelve consecutive idle
samples. Estimated runtime: smoke 30--90 minutes; screen 16--48 hours, mostly
depending on prompt length and the eleven replay branches.

The published CPU receipt is
`artifacts/prepare_v1/preflight_receipt.json`. It verifies the 40 k20
safetensors and sidecars by SHA-256 (6,711,407,831 bytes), as well as all data,
tokenizer, prompt-plan, code, and artifact commitments.

## Exact causal boundary

For one frozen serialized prompt `C = [c0, …, cL-1]`, every mode uses the same
token IDs and this sequence:

1. native Qwen3-14B produces its target-format cache for `C[:-1]`;
2. Qwen3-32B produces a cache for the same `C[:-1]`, and the read-only k20
   mapper converts it into target-format cache tensors;
3. K and V are selected per target layer according to one intervention;
4. native Qwen3-14B replays exactly `C[-1]` at position `L-1` from a fresh
   cache branch, producing the first answer-token logits;
5. greedy target decoding continues for at most 256 new tokens.

Mixing full-prompt caches would not provide comparable first-token logits, and
mixing before/after RoPE would be invalid. The executable cache envelope
therefore requires 40 layers of `[1, 8, L-1, 128]` tensors, identical per-layer
native/mapped shape, dtype, and device, contiguous positions `0..L-2`, target
post-RoPE keys, target value space, `rope_theta=1e6`, no RoPE scaling, no
sliding window, and finite values. Prompt, prefix, positions, target config,
and mapper provenance are SHA-committed. Any mismatch fails before replay.

## Interventions

The four global K/V conditions are:

1. `native_k_native_v` (paired reference);
2. `mapped_k_native_v`;
3. `native_k_mapped_v`;
4. `mapped_k_mapped_v` (k20 baseline).

Starting from mapped K + mapped V, both K and V are then replaced by native
cache tensors for `[0,9]`, `[10,19]`, `[20,29]`, `[30,39]`, layer 34, layer
35, and layers 34+35. Layer IDs are zero-based and inclusive in artifact IDs;
the corresponding mathematical blocks are `[0,10)`, `[10,20)`, `[20,30)`,
and `[30,40)`.

## Measurements

The interfaces require a paired result for every prompt/intervention:

- first-token `KL(native || intervention)` in FP64;
- gold answer margin: best valid first token across answer aliases minus the
  best non-gold vocabulary token, reported for native, candidate, and delta;
- normalized answer EM/F1 and candidate-minus-native paired delta;
- normalized native/candidate answer agreement, EOS/cap reason, and length;
- per-layer native-query attention-output cosine, normalized MSE, and
  attention-probability Jensen-Shannon divergence.

The attention diagnostic freezes native-14B post-RoPE boundary Q. For each
layer, concatenate the selected prefix K/V with the native boundary-token K/V,
then apply the identical native Q through an explicit causal GQA FP32 softmax.
The output is measured per query head before `o_proj`, then uniformly averaged
over the 40 target layers. A SHA-256 commitment to Q and its absolute position
must match between the paired readouts; endogenous or different queries are
rejected. This isolates prefix cache K/V error. First-token logits and
generation remain the separate end-to-end functional measurement through the
actual mixed-cache trajectory.

## Files

- `config.json`: frozen inputs, geometry, prompt/decode policy, and no-GPU
  authorization boundary;
- `src/cache_oracle.py`: cache envelopes, 2×2/layer composition, boundary
  replay inputs, attention/KL/margin/F1 metrics;
- `src/runtime_harness.py`: dependency-injected 11-condition paired executor;
- `src/oracle_diagnostics.py`: train-only selection, prompt preparation,
  artifact publication, and deep preflight;
- `artifacts/prepare_v1/method_dev_records.jsonl`: exact prompt text, token IDs,
  answers, cache context, and hashes;
- `artifacts/prepare_v1/paired_run_plan.jsonl`: ordered paired execution plan;
- `artifacts/prepare_v1/interventions.json`: executable intervention catalog.

## CPU-only commands

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare.py
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
CUDA_VISIBLE_DEVICES='' python scripts/preflight.py
```

`scripts/preflight.py` hashes every k20 layer by default. The
`--manifest-only-mapper` switch is only for a local smoke check and is not a
final preflight. A future GPU executor must be separately authorized and must
consume the published records without changing prompts, cap, mapper, or
intervention order.
