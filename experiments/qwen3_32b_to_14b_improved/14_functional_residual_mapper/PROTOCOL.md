# Functional residual mapper protocol — pre-formal v1

## 1. Status and estimand

This is an untrained CPU-only framework. The primary future estimand is the
paired change in native-Qwen3-14B fidelity produced by one frozen residual R=1
mapper relative to the frozen NVIDIA-compatible k20 R=1 mapper, with IDs,
prompts, tokenizer, decoding, scoring, and model weights held fixed.

R=32 is a secondary hybrid estimand with explicit target replay cost. It is
not pooled with, substituted for, or used to nominate the pure R=1 method.

## 2. Hard data boundary

The three tiers are disjoint by complete sequence/conversation/document before
tokenization:

| tier | permitted use here | forbidden use |
|---|---|---|
| mapper-fit | fit residual parameters, standardization and OOD statistics | benchmark validation/test, locked IDs/text, answers |
| method-dev train | answer-free oracle, layer/rank/loss selection, early stop | gradients on benchmark answers, QA/downstream score selection, locked content |
| locked-test | one frozen prospective confirmation after nomination | any tuning, refit, prompt/stop change, second candidate |

The final tier registry and complete-G0-v2 receipt must prove zero exact ID,
normalized text, near-duplicate, and substring collision under the project
contract. Every input is an exact path/byte/SHA ArtifactRef; symlinks and path
escape fail closed. The framework never parses locked/test artifacts during
training or selection.

Answers are unnecessary for this method's selection and therefore remain
unopened. The answer-free selection allowlist is exactly attention-output
cosine/NMSE, centered attention-logit Huber, native-14B suffix KL, native-14B
EOS BCE, and KV R² guardrail. QA F1/EM, accuracy, gold margin, ROUGE, CHRF,
normalized-token F1, generated answers, and any locked outcome are denied.

## 3. Frozen inputs required before formalization

Before a formal config can exist, all placeholders in `config.template.json`
must be replaced and content-addressed:

1. final complete-G0-v2 receipt;
2. final decontaminated train registry receipt;
3. answer-free train-only causal-oracle receipt;
4. mapper-fit functional-capture manifest;
5. complete three-tier registry;
6. one rank, a nonempty subset of `[7,8,9,10,34,35]`, and one positive weight
   per active layer;
7. an independently reviewed GPU layout and receiver-adapter implementation.

The anchor is already frozen to dir02 `headwise_shrinkage_v1`, manifest
`dabc63b...b4995`. A hash/payload failure aborts the method version. It never
triggers an outcome-dependent switch to k4.

## 4. Formal capture contract

Each mapper-fit example binds one exact prompt/token sequence and contains no
answer. It provides source-32B content-space KV features in the anchor's
per-target-layer k20 order, anchor mapped KV, native-14B KV, frozen native-14B
queries at sampled causal rows, exact query/key positions and masks, identical
teacher-forced suffix tokens, native-14B teacher logits computed online or in
a hash-bound capture, and tokenizer/model/code hashes.

Sequences remain whole for splitting. Full T² attention is not stored: query
rows are the union of tail rows, chunk boundaries, and seeded uniform rows.
No truncation is allowed. Source-32B logits are never a teacher.

## 5. Preregistered method decisions

The decision order is immutable:

1. validate anchor and G0 inputs without reading outcomes;
2. run the answer-free oracle over layers 7,8,9,10,34,35;
3. freeze active layers and clipped/normalized positive loss weights;
4. fit rank 64, then 128, then 256 on mapper-fit only;
5. on method-dev screen, select the first registered rank passing all
   functional and guardrail gates;
6. evaluate that rank once on method-dev confirm;
7. if confirm passes, freeze exactly one pure R1 candidate;
8. after independent audit, perform one prospective locked R1 evaluation;
9. only then report separately frozen optional R32.

Confirm failure ends this method version; another rank/config requires a new
version. Later stages cannot rescue a failed earlier stage.

## 6. Functional objective

For every active layer and K/V independently, a shared 20480→rank encoder and
eight head-specific rank→128 decoders correct the frozen anchor. Decoder and
bias initialize to zero. Fixed native queries define centered attention-logit
Smooth-L1, attention-output cosine, and normalized MSE. End-to-end receiver
forward defines native→candidate KL at horizons 8 then 32 and EOS soft BCE.
Variance-normalized KV Huber/R² is a guardrail only.

Loss curriculum and weights are machine-frozen in the template. Dynamic OOD
smooth gating and a per-token/layer/component/head residual-to-anchor RMS cap
are part of the candidate, not post-hoc evaluation fixes.

## 7. Method-dev gates

All effects are paired by sequence. Required before confirm:

- every active layer has positive answer-free causal-oracle primary effect and
  paired-bootstrap 95% lower bound above zero;
- attention-output cosine improves by at least 0.010 over the frozen anchor,
  with paired-bootstrap lower bound above zero;
- native-14B suffix KL does not worsen by more than 2% versus anchor;
- K and V held-out global R² each decline by no more than 0.002 from anchor;
- finite outputs, exact zero-gate anchor reproduction within `1e-6`, and no
  sequence/tier collision;
- residual weights and measured latency/VRAM satisfy the parent G3 system
  limits before nomination.

Downstream train metrics, if later reported descriptively, cannot choose the
candidate. The final parent protocol's downstream non-inferiority gates still
apply before any nomination or claim.

## 8. Checkpoint and receipt requirements

Each checkpoint is a new directory committed atomically. It contains safe
tensor payload(s), canonical metadata, a manifest with byte/SHA bindings, and
a final `COMMITTED.json`. Existing or malformed checkpoints are never
overwritten. Resume revalidates the full manifest first.

A completion receipt must bind config, source inventory, anchor, G0, registry,
oracle, capture, selected rank/layers/weights, checkpoint, seed, environment,
loss curves, gate distributions, residual cap/fallback rates, measured weight
bytes, per-GPU peak allocated/reserved bytes, and exact R1/R32 class. It must
state `locked_outcomes_opened=false` until the one authorized confirmation.

## 9. Fail-closed conditions

The run stops before torch/model/CUDA work if any placeholder remains, GPU
authorization is false, an ArtifactRef mismatches, a symlink/path escape is
found, a tier/sequence collision exists, an answer/outcome selection field is
present, oracle layers/weights do not match the exact contract, or outputs do
not resolve under this directory.

During training it stops on nonfinite loss/gradient/parameter, guardrail
failure, malformed receiver output, checkpoint validation failure, or GPU lock
loss. Infrastructure failure before metric visibility may resume only from a
valid committed checkpoint; outcome-visible reruns are not silently replaced.

## 10. CPU pre-formal verification and STOP

The checked-in template deliberately has placeholders,
`formal_config=false`, no receiver adapter, and
`gpu_execution_authorized_by_this_config=false`. The expected CPU preflight is
therefore “static contract valid; formal execution blocked.” Synthetic tests
cover residual algebra, zero/fallback invariants, functional losses, tier
tampering, checkpoint atomicity, curriculum/trainer behavior, and config
fail-closed paths.

This version stops after publishing a hash-bound pre-formal receipt. No GPU
command is authorized and no comparative claim is permitted.

