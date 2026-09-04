# Decontaminated causal-oracle protocol (pre-formal)

## Purpose and claim boundary

This experiment is a small, train-only diagnostic for the frozen NVIDIA k20
Qwen3-32B → Qwen3-14B cache mapper. It is designed to locate whether mapped K,
mapped V, or particular target-layer blocks cause functional loss. It neither
trains a new method nor by itself supports a claim that any method beats the
NVIDIA baseline.

The checked-in configuration is deliberately non-executable. Final dir10
native-registry artifact references, the final dir09 complete-G0-v2 receipt,
the two pinned train-source references, their exact projection algorithm,
strata order, and both profile counts remain placeholders. GPU authorization
is also false. Any one of those conditions blocks before torch is imported.

## Data isolation and deterministic sampling

- Only final `hotpotqa_screen` and `musique_screen` rows from the native dir10
  registry may be consumed. Confirm files are byte/hash-bound for G0
  completeness but their rows are never parsed or consumed by this diagnostic.
- Every consumed registry row must be `split=train`, `tier=method-dev`, and
  contain no prediction, metric, score, logit, or outcome field.
- The pinned raw train record is resolved by `source_index`. Its answer-free
  question-and-passages projection is recomputed and must match
  `source_projection_sha256`.
- Within each declared stratum, rows are ordered by the stratum-local
  `selection_rank` and then `source_index`. Diagnostic rows are selected by a
  frozen round-robin over the configured strata order until the final count is
  reached. Answers and model outputs do not enter this process.
- Answers are opened only after the final registry receipt explicitly grants
  policy `train_method_dev_oracle_metrics_only`. They may be used only for the
  gold first-token margin and QA F1/EM, never for selection, ranking, quotas,
  partition assignment, or intervention choice.
- That policy is an exact four-key typed object (`policy_id`, `allowed`,
  `allowed_uses`, `forbidden_uses`) embedded identically in the formal config
  and receipt contract. Missing, extra, reordered-use, or bool/int-aliased
  values fail before train answers are opened.
- Every pinned path is checked component-by-component with `lstat` before
  canonical resolution. Both live and dangling symlinks are forbidden for
  configs, ArtifactRefs, train sources, models, mapper files, and borrowed
  interfaces.
- No dev/validation/test/locked outcome may be opened.

Complete G0-v2 must use the frozen dir09/dir12 compatibility schema. Its
`candidate_registry_binding` is exactly `{published_marker,
registry_manifest, publication_receipt, split_artifacts}`, with
`split_artifacts` containing exactly the four HotpotQA/MuSiQue screen/confirm
ArtifactRefs. Its comparison matrix contains exactly
`methoddev_hotpot_vs_mapper_fit`, `methoddev_musique_vs_mapper_fit`,
`methoddev_hotpot_vs_locked_eval`, and
`methoddev_musique_vs_locked_eval`; every value is exactly `{"flags":0}`.
Extra directions, missing directions, flattened bindings, or extra fields
inside a comparison fail closed.

The intended formal diagnostic remains at most 32 records per dataset and 64
records total. Exact smoke and diagnostic counts will be frozen only after the
final registry capacity is known.

## Frozen prompt and causal boundary

For each selected train record, all passages and the question are rendered
with the target Qwen3 tokenizer's chat template, a fixed short-answer system
instruction, `enable_thinking=false`, and an assistant generation marker.
Truncation is forbidden; the serialized prompt must contain 2–8192 tokens.

Let the exact token vector be `C = [c0, …, cL-1]`.

1. Native Qwen3-14B builds its target cache for exactly `C[:-1]`.
2. Qwen3-32B builds the source cache for the identical `C[:-1]`; the frozen
   NVIDIA k20 mapper converts it once to target-format K/V.
3. One isolated cloned cache branch is composed for each intervention.
4. Native Qwen3-14B replays exactly `C[-1]` at absolute position `L-1`.
5. First-answer-token logits are measured and greedy decoding continues to
   EOS or 256 content tokens.

The runner also evaluates native full `C` and fails unless its boundary logits
and cache agree with the `C[:-1] + C[-1]` replay. Native boundary Q/K/V are
captured after Q/K normalization and target RoPE; captured K/V must match the
actual target cache. Native and mapped envelopes must contain exactly 40
layers of finite `[1, 8, L-1, 128]` K and V tensors, with equal per-layer
shape, dtype, and device, contiguous zero-based positions, post-target-RoPE
keys, target value space, `rope_theta=1e6`, no RoPE scaling, and no sliding
window.

## Preregistered interventions

The order is immutable:

1. native K + native V;
2. mapped K + native V;
3. native K + mapped V;
4. mapped K + mapped V (NVIDIA k20 reference);
5. mapped/mapped with native K+V restored at layers 0–9;
6. mapped/mapped with native K+V restored at layers 10–19;
7. mapped/mapped with native K+V restored at layers 20–29;
8. mapped/mapped with native K+V restored at layers 30–39;
9. mapped/mapped with native K+V restored at layer 34;
10. mapped/mapped with native K+V restored at layer 35;
11. mapped/mapped with native K+V restored at layers 34+35.

Layer IDs are zero-based. The mapped prefix cache is materialized exactly once
per prompt; all 11 interventions clone and compose from that same object.

## Measurements

Every intervention is paired to native K + native V for the identical prompt.
The atomic result must contain:

- first-token `KL(native || intervention)` in FP64;
- native/candidate gold-answer first-token margin and paired delta;
- native/candidate normalized QA F1 and EM and paired deltas;
- greedy token IDs, EOS/cap reason, and committed decoded-text hash;
- per-layer native-fixed-query attention-output cosine, normalized MSE, and
  attention-probability JS divergence for all 40 layers;
- the uniform layer-mean attention-output cosine.

For attention diagnostics, the native target boundary Q is frozen. Selected
prefix K/V plus the identical native boundary K/V are passed through explicit
causal GQA FP32 attention before `o_proj`. Query hashes and positions must be
identical inside each pair.

## Execution, resume, and audit gates

The formal execution order is fixed:

1. reject placeholders/non-authorization;
2. byte/hash-check the read-only dir01 interface sources;
3. validate complete G0-v2 and its exact seven-artifact registry binding;
4. validate the final registry publication and answer policy;
5. reconstruct selected records from pinned train sources;
6. require exact `CUDA_VISIBLE_DEVICES=0,…,7`;
7. acquire the repository-wide eight-GPU lock;
8. observe 12 consecutive idle samples on all eight cards;
9. deep-hash every model shard and every regular file in the k20 mapper
   directory before importing torch;
10. load 32B over GPUs 0–4 and 14B over GPUs 5–7;
11. write one atomic checkpoint per prompt and independently revalidate it;
12. publish a completion receipt only after every checkpoint passes.

An existing valid checkpoint is resumed. An existing malformed checkpoint or
run definition is never overwritten. Failure records are append-only and kept
separately. All outputs remain under this directory's `runs/` tree.

## Pre-formal stop

This turn ends before a formal configuration is created and before any GPU
command is run. The next authorized step is to pin final dir10/dir09 artifacts,
reconcile their exact machine schemas with the two consumer contracts, freeze
counts/strata/projection, run a fresh independent CPU audit, and only then run
the smoke profile.
