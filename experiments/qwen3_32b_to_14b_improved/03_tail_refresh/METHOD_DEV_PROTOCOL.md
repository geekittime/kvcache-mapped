# Tail-refresh method-dev protocol

## Scientific question

Does replacing only the final 7 or 31 mapped prompt positions with native
Qwen3-14B cache entries recover part of the Qwen3-32B→14B k20 fidelity gap on
multi-hop QA? This is a hybrid quality/target-prefill trade-off, not a new
mapper-only result.

The registered modes are evaluated on the same exact tokenized prompt, with
greedy decoding and a 256-token cap:

| mode | mapper | native target prompt tokens | claim class |
|---|---|---:|---|
| `target_standalone` | none | all | fidelity reference |
| `mapped_r1` | immutable production k20 | 1 boundary | NVIDIA-compatible local baseline |
| `tail_refresh_r8` | same k20 | 7 builder tail + 1 boundary | hybrid |
| `tail_refresh_r32` | same k20 | 31 builder tail + 1 boundary | hybrid |

## Causal/cache contract

For full prompt tokens `C` of length `T`, `TransferEvaluationEngine` calls its
prefix builder with `C[:-1]`, length `P=T-1`, then separately replays `C[-1]`.
The builder must therefore do the following for registered replay window `R`:

1. run 32B and k20 mapping only on `C[:P-(R-1)]`;
2. append `C[P-(R-1):P]` in one native 14B batch of exactly `R-1` tokens;
3. use logical `position_ids` and physical `cache_position` equal to
   `P-(R-1), ..., P-1`;
4. return a cache of length `P`;
5. leave `C[-1]` to the engine at position `P`.

Thus `R=8` means 7 builder tokens plus one engine boundary—not eight plus one.
`R=1` performs no target call and returns the exact underlying k20 cache
object without clone, conversion, or cast. A prompt too short to leave at
least one mapped token fails closed instead of silently becoming native-only.

The production builder validates the exact 64-layer/40-layer architectures,
mapper payload hashes, source/target artifact identities, and resident layer
devices. Tail refresh additionally binds the validated model object instances
and rejects a post-validation swap.

## Method-development data only

No formal HotpotQA dev result and no MuSiQue dev/test row is an input to this
phase.

- HotpotQA: official `hotpot_train_v1.1.json`, stratified by
  `bridge/comparison × easy/medium/hard`.
- MuSiQue: official answerable `musique_ans_v1.0_train.jsonl`, stratified by
  2/3/4 hops.

Within every stratum, samples are ranked only by
`SHA256(seed || dataset_id || sample_id)`. Answers, prompt length, native
outputs, k20 outputs, and metrics do not affect selection. Phases are nested:

| phase | HotpotQA | MuSiQue | total samples | total atomic modes |
|---|---:|---:|---:|---:|
| smoke | 6 | 3 | 9 | 36 |
| pilot | 48 | 48 | 96 | 384 |
| expansion | 192 | 192 | 384 | 1,536 |

All expansion rows are frozen during preparation. The pilot cannot be reranked
after observing results. Both datasets use the same frozen instruction:
“Answer the multi-hop question using only the supplied context. Return only
the shortest answer span and no explanation.” MuSiQue prompt bytes and
official-index paragraph order are bound to `06_musique_benchmark`'s
`build_prompt`; HotpotQA uses the same instruction and completion structure
over its official train passages. Exact prompts use the Qwen chat template
with `enable_thinking=False`, greedy decoding, a 256-token cap, and the same
`\nQuestion:`/`\nContext:` stops as the prospectively locked MuSiQue protocol.
Truncation is forbidden and preflight verifies every prompt fits and supports
R=32. The preparation receipt pins the 06 config and registry-manifest
SHA-256 values.

The candidate-local overlap audit covers exact IDs, normalized
question+context hashes, and cross-dataset question word-trigram Jaccard ≥0.9.
The project-wide mapper-fit/method-dev/locked-test registry remains a separate
G0 prerequisite before nomination; the local receipt does not overclaim that
global gate.

## Metrics and decisions

Report for every dataset and mode:

- raw and gold-independent relaxed QA F1/EM;
- MuSiQue official normalized answer F1/EM (max over aliases);
- normalized gold containment;
- target score gap, delta from R=1, and gap closure only when target exceeds R=1;
- generated-token length, EOS rate, cap rate, wall time, and per-GPU peak memory;
- exact cache split, position IDs, cache positions, and lengths for every
  mapped result.

R=8/32 is promising only if its paired method-dev F1 improves over R=1 without
worse cap/termination behavior. Any later locked-test evaluation requires a
separate freeze and may not reinterpret a hybrid result as mapper-only.

## Execution order

CPU-only preparation and preflight:

```bash
cd /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/03_tail_refresh
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/prepare.py --dry-run
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/prepare.py --prepare
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/run_method_dev.py \
  --phase pilot \
  --output-dir runs/method_dev_pilot_r1_r8_r32_v2 \
  --preflight-only
```

Only after the global GPU lock is free and GPUs are idle may an operator—not
the preparation agent—replace `--preflight-only` with `--execute` and set:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python \
  scripts/run_method_dev.py \
  --phase pilot \
  --output-dir runs/method_dev_pilot_r1_r8_r32_v2 \
  --execute
```

The runner materializes k20 once and shares the same resident mapper object
across R variants. It acquires the repository-wide eight-GPU coordination lock
and writes all scientific outputs exclusively under `03_tail_refresh/runs/`.
