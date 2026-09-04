# Qwen3-32B → Qwen3-14B long-context extension

This directory evaluates the already trained production `k=20` reverse KV
mapper on four datasets that are outside the paper's published reverse-task
table. It is an isolated, protocol-aligned local extension and does not mutate
the completed paper reproduction, mapper weights, or the existing 14B→32B
long-context runs.

## Frozen comparison

Both modes use the same Qwen tokenizer, exact prompt token IDs, greedy decoding,
and a batch size of one:

- `target_standalone`: Qwen3-14B performs the complete prefill and generation.
- `mapped`: Qwen3-32B prefills the prompt, its KV cache is transformed by the
  production `k=20` mapper, and Qwen3-14B continues from the mapped cache.

The mapped checkpoint is cryptographically bound to the corresponding target
checkpoint's sample, references, prompt token IDs, truncation record, and token
cap. Every sample/mode pair is an atomic resumable JSON checkpoint. Each
checkpoint also persists the complete generated token-ID list, its count and
hash, the exact decoded prediction, and an `eos` or `max_tokens` stop proof.
Resume rejects a checkpoint unless all of that evidence recomputes exactly.

## Dataset scope and metrics

| Dataset | Scope | Primary metric | Additional metrics |
|---|---:|---|---|
| HotpotQA | all 7,405 `dev_distractor` rows | relaxed answer F1 | standard/extracted F1 and EM, containment diagnostics |
| MultiNews | fixed bottom-500 of 5,622 official test rows | ROUGE-L F1 | ROUGE-1 and ROUGE-2 F1 |
| LMSYS | deterministic train-derived held-out 500 | reference corpus PPL | reference ROUGE-L, chrF, normalized-token F1, output retention |
| ShareGPT4 | deterministic train-derived held-out 500 | reference corpus PPL | reference ROUGE-L, chrF, normalized-token F1, output retention |

MultiNews has an aligned official reference summary in `test.tgt`. The summary
is therefore scored against gold, but abstractive summarization does not have a
classification-style "accuracy"; ROUGE-L F1 is the headline measure. LMSYS and
ShareGPT4 publish train-only corpora, so their final assistant turns are frozen
as deterministic local held-out references and must not be described as an
official test set.

All four tasks use the Qwen chat template with `enable_thinking=false`, greedy
decoding, `max_length=16,384`, and `max_gen_tokens=1,024`. In particular,
HotpotQA uses the larger 1,024-token cap requested for this reverse experiment;
every EOS/cap stop is recorded rather than inferred from decoded text.

## Runtime topology

The formal simultaneous topology matches the validated reverse evaluator:

- source Qwen3-32B: GPUs `0,1,2,3,4`, layer counts `12,13,14,13,12`;
- target Qwen3-14B plus resident FP32 mapper: GPUs `5,6,7`, layer counts
  `13,14,13`;
- mapper: `20480 → 1024` per target-layer K/V map, 40 target layers,
  `mapper_token_chunk=1024`.

Model roles are intentionally inverted only at the shared loader's native
forward architecture assertion: semantic source 32B is checked with native
role `target`, while semantic target 14B is checked with native role `source`.
The reverse runtime then independently verifies both loaded checkpoints,
mapper direction, all layer artifacts, and the `64-layer → 40-layer` geometry.

## Entrypoints

Run from the repository root with the pinned environment:

```bash
.venv/bin/python experiments/qwen3_32b_to_14b/long_context_extension/freeze.py \
  --write

.venv/bin/python experiments/qwen3_32b_to_14b/long_context_extension/run.py \
  --profile prepare
```

The freeze must be written only after code, tests, configuration, and this
document are final. `prepare` is CPU-only. It loads and freezes dataset
identities, hashes model weight shards, mapper metadata, metric
implementations, and code, then returns before any CUDA check. It holds the
same global writer lock as GPU runs so it cannot overwrite a concurrent full
run's status/control files.

After `prepare`, run a separate two-row-per-task smoke (8 samples and 16 atomic
mode checkpoints):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python experiments/qwen3_32b_to_14b/long_context_extension/run.py \
  --profile smoke
```

Then launch or resume the complete 17,810-checkpoint run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
.venv/bin/python experiments/qwen3_32b_to_14b/long_context_extension/run.py \
  --profile full
```

The default formal output is:

```text
experiments/qwen3_32b_to_14b/runs/long_context_extension_full_k20_v1/
```

`prepare` and `full` intentionally share that output and scientific request
fingerprint. The smoke uses an independent directory and is marked
`frozen_explicit_two_indices_per_task_not_formal`. The frozen smoke endpoints
are self-described in `extension_scope.json`, `run_definition.json`, and the
smoke completion receipt:

| Task | Evaluation index | Source index | Coverage intent |
|---|---:|---:|---|
| HotpotQA | 3469 | 3469 | maximum frozen source-content length and long-prompt EOS path |
| HotpotQA | 645 | 645 | historical mapped 512-token cap hit versus target EOS |
| MultiNews | 1463 | 1463 | maximum source length; historical 63,184-token prompt truncation |
| MultiNews | 2938 | 2938 | historical mapped 1,024-token cap hit versus target EOS |
| LMSYS | 125 | 85754 | maximum frozen prompt; historical mapped cap hit versus target EOS |
| LMSYS | 370 | 202632 | reference cap plus target generation cap versus mapped EOS |
| ShareGPT4 | 341 | 3775 | maximum frozen prompt and dialogue message-drop truncation |
| ShareGPT4 | 435 | 5095 | reference cap and generation cap in both modes |

GPU profiles acquire the repository-wide reverse evaluation lock and require
twelve consecutive idle samples across all eight visible GPUs before model
loading. Completion is checked again after acquiring the lock, so a second
waiter exits after strict receipt validation instead of loading duplicate
models. If the shared runner completed but a receipt was not yet emitted, the
CPU finalizer reconstructs and validates every artifact before publishing it.

## Reused and direction-specific code

The shared [long-context runner](../../../scripts/evaluate_long_context.py)
continues to own prompt construction, truncation, boundary replay, generation,
reference scoring, checkpoint validation, and aggregation. `adapter.py` applies
only fail-closed overrides for:

- reverse model roles and the 5+3 topology;
- `ReverseMappedPrefixCacheBuilder`;
- full model-shard and reverse mapper provenance;
- exact 7,405/500/500/500 sample scopes;
- MultiNews ROUGE-1/2/L scoring;
- reverse-only output containment and protocol disclosure.

Changing any bound code, dataset, model shard, mapper artifact, prompt, cap,
metric implementation, sample selection, or topology changes the request
fingerprint and prevents stale checkpoint reuse.

The finalizer independently reconstructs all 17,810 formal (or 16 smoke)
checkpoints, recomputes task metrics from raw prediction/reference values,
checks dialogue likelihood algebra, rehashes model and mapper provenance, and
publishes a profile-specific hashed receipt. Output is restricted to a
non-root child of `experiments/qwen3_32b_to_14b/runs/`; symlinked outputs and
symlinks anywhere in the result inventory are rejected. GPU gate JSON and its
SHA sidecar are two atomic file replacements rather than one filesystem
transaction: a crash between them leaves a visible incomplete pair that is
intentionally rejected and requires explicit operator audit before retry.
