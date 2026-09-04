# Preregistered experiment protocol

## 1. Objective and estimands

The primary scientific estimand is the change in fidelity to native
Qwen3-14B caused by replacing the NVIDIA-compatible k20 ridge mapper with one
candidate mapper while holding model weights, prompts, tokenizer, decoding,
sample IDs, and scoring fixed.

The primary systems estimand is the quality/TTFT trade-off after adding a
declared target-side replay window.  It is reported separately because replay
uses target prefill compute.

The comparison classes are:

- **pure mapper**: `R=1`, matching the existing boundary-token replay;
- **hybrid refresh**: `R=8` or `R=32`, with the last R prompt tokens recomputed
  by native 14B;
- **native target**: complete 14B prefill, the fidelity reference.

## 2. Three-tier data isolation

The machine-readable rules are in `data_policy.json`.  Splits are made by
complete sequence/conversation/document ID before tokenization.  Token-level
random splitting is forbidden because neighboring tokens would leak context.

### mapper-fit

Purpose: fit mapper weights and feature statistics.  It may contain only
general-domain training material or newly generated synthetic material whose
provenance is recorded.  Official benchmark validation/test questions,
answers, conversations, prompts, and existing evaluation outputs are
forbidden.  Native 14B activations/logits may be training targets; 32B logits
may not be used as teacher in the fidelity study.

### method-dev

Purpose: choose source features, rank, lambda, loss weights, layer/head gates,
early stopping, and R.  It uses sequence-disjoint held-out general data and,
where licensing permits, only official **training** partitions of downstream
tasks.  It must not contain IDs from mapper-fit or locked-test.

All causal diagnostics, including native K/V oracle replacement and per-layer
replacement, occur here.  The existing official evaluation aggregates may be
used only as historical baseline context, not as observations in an optimizer
or selection rule.

### locked-test

Purpose: one confirmatory run of one nominated pure mapper, followed by
separately nominated hybrid modes.  Configuration and code hashes are frozen
before unsealing.  Hyperparameters cannot change after results are read.

The project has already observed aggregate results on several official test
sets.  Therefore `locked-test` means *prospectively locked against further
tuning*, not historically blind.  Every report must disclose this prior
exposure.  A stronger future claim requires an external holdout or evaluator
whose outcomes have never been exposed to the development team.

## 3. Contamination controls

Before capture or evaluation, create an immutable registry containing:

- canonical source URI/revision/config/split and license;
- raw payload SHA-256 and normalized sample-ID SHA-256;
- exact sample IDs or a content-addressed bitmap;
- tokenizer ID/revision and serialized prompt SHA-256;
- dedup fingerprints computed before any answer/metric is read;
- split namespace, salt commitment, and sequence grouping key.

Check exact IDs, normalized text hashes, and MinHash/near-duplicate signatures
across all three tiers.  Any collision quarantines the entire sequence group.
The collision list and exclusion decision are append-only artifacts.

## 4. Candidate and ablation order

Only one factor changes at a time.  A phase that fails its gate cannot be
combined with later components to rescue it.

| order | ID | change from previous accepted phase | selection data |
|---:|---|---|---|
| 0 | A0 | reproduce content-addressed k20/k4 baselines; no training | none/read-only |
| 1 | A1 | K/V and layer-range oracle replacements to establish causal targets | method-dev |
| 2 | A2 | sequence-CV k4/k20 shrinkage only | mapper-fit + method-dev |
| 3 | A3 | feature standardization and per-layer/head/K/V ridge lambda only | mapper-fit + method-dev |
| 4 | A4 | linear SARR residual, ranks 64/128/256; zero gate fallback | mapper-fit + method-dev |
| 5 | A5 | attention-distribution/output loss only | mapper-fit + method-dev |
| 6 | A6 | 8-token, then 32-token native-14B suffix KL and EOS loss | mapper-fit + method-dev |
| 7 | A7 | nonlinear residual or mixture only if A6 saturates and budget allows | mapper-fit + method-dev |
| 8 | A8 | freeze and nominate exactly one pure mapper | no locked-test access |
| 9 | A9 | one pure-mapper locked-test run | locked-test |
| 10 | H1 | separately frozen R=8 and R=32 hybrid refresh curves | locked-test |
| 11 | H2 | optional risk-based fallback, with coverage fixed on method-dev | locked-test once |

Layer 34/35 or other suspected layers are not privileged until A1 demonstrates
a paired causal effect.  The dense MLP is deliberately late: it adds capacity
and runtime cost while losing the exact safe fallback provided by the anchor.

## 5. Functional capture plan

Existing activations suffice for A2–A4 reconstruction work.  A5/A6 require a
new, independent capture rooted outside this directory.  The initial capture
budget is 64K tokens; it expands to at most 256K only after an attention gate
passes.

The planned 256K mixture is 64×1K, 32×4K, and 8×8K contiguous sequences,
rounded by actual token count.  It should cover prose, news, dialogue,
short-answer QA, mathematical structure, and long contexts without importing
official evaluation examples or answers.

Record source-32B content K/V, native-14B Q anchors and attention outputs,
sampled attention top-indices/probabilities plus tail mass, 8–32 teacher-forced
suffix logits, labels, exact positions, sequence offsets, and model/code
hashes.  Source and target models may be captured sequentially; simultaneous
loading is not required.

## 6. Metrics

### 6.1 Mechanism and reconstruction

- sequence-held-out normalized K/V Huber and global/uniform-channel R²;
- train→held-out R² gap per layer/head/K/V;
- native-query attention probability JS divergence;
- attention-output cosine and normalized MSE;
- first-token and 8/32-step teacher-forced logit KL at temperature 1 and 2;
- EOS/stop/control-token logit difference;
- residual norm, gate distribution, OOD fallback rate.

### 6.2 Downstream fidelity

For classification report accuracy **and** paired fidelity:

- target/mapped decision agreement;
- target-correct retention;
- target-correct→candidate-wrong regressions;
- target-wrong→candidate-correct gains;
- exact 2×2 candidate-versus-k20 paired table.

For QA/summarization report the registered official/relaxed metric, paired
per-example difference, answer containment where diagnostic, generation
length, EOS rate, cap rate, truncation rate, and stop reason.  CoQA resampling
unit is conversation, not individual turn.

For language modeling report NLL as the inferential quantity and perplexity as
its exponential presentation.  For every mode report mapper time, target
refresh time, end-to-end TTFT, peak allocated/reserved VRAM, weight bytes, and
hardware topology.

### 6.3 Gap closure

For a higher-is-better task on which k20 is below native target:

\[
GC=(M_{candidate}-M_{k20})/(M_{target}-M_{k20}).
\]

For a lower-is-better metric use the sign-adjusted utility.  If k20 equals or
beats native target, gap closure is undefined and must not be reported; use
paired agreement, regression rate, and a non-inferiority test instead.  This
prevents accidental ARC/MMLU gains from being mislabeled as fidelity.

## 7. Statistical plan

- Fixed resampling seed: `20260830`; 10,000 paired replicates.
- Binary outcomes: exact two-sided McNemar for candidate versus k20, plus a
  paired bootstrap confidence interval for accuracy difference.
- Continuous per-example QA/ROUGE/CHRF/NLL: paired bootstrap percentile 95% CI.
- CoQA/chat conversations: cluster bootstrap by conversation ID.
- MultiNews: document-level paired bootstrap.
- Multiple candidates/tasks within one decision family: Holm correction at
  family-wise alpha `0.05`.
- Report raw effect, 95% CI, adjusted p-value, sample count, missing/cap count,
  and all prespecified exclusions.  Statistical significance without the
  minimum practical effect does not pass.
- No seed cherry-picking.  Stochastic decoding, if ever required, uses three
  preregistered seeds and hierarchical resampling; deterministic decoding is
  preferred for fidelity comparison.

## 8. GO/NO-GO gates

### G0: integrity before any fit

GO only if baseline hashes validate, the three-tier registry has no collisions,
all output paths are inside the candidate's independent directory, config/code
hashes are captured, and the GPU job lock is acquired by the eventual runner.
This protocol-validation phase itself never acquires GPUs.

### G1: shrinkage/standardization

On sequence-held-out method-dev:

- K global R² no worse than k20 by more than `0.002`;
- V global R² no worse than the better of k20/k4 by more than `0.002`;
- K and V train→held-out gap each reduced by at least 20% relative to k20;
- no NaN/Inf, and all exact-zero gates reproduce the k4 anchor within the
  registered numerical tolerance.

NO-GO if improvement exists only on training observations.

### G2: functional residual

On method-dev:

- mean attention-output cosine improves by at least `0.010` over `0.761078`;
- paired-bootstrap lower 95% bound of the cosine improvement is above zero;
- suffix KL does not worsen by more than 2%;
- no individual preregistered dev task worsens beyond its non-inferiority
  margin.

### G3: downstream nomination

Nominate at most one pure mapper if:

- macro gap closure over degraded dev tasks is at least 20%;
- its paired-bootstrap lower 95% bound is above zero;
- no classification task is more than `0.5 pp` below k20;
- no numeric/short-QA task is more than `1.0 pp` below k20;
- CoQA is not more than `1.5 pp` below k20;
- target-correct retention does not fall by more than `0.5 pp` on any task;
- mapper weights are below 3 GiB, peak training/inference allocation is
  documented, and R=1 latency is at most `1.05×` current k20 on identical
  hardware.

The nomination record freezes method ID, parameters, data registries, code,
prompts, decoding, metric versions, seeds, and expected output schema.

### G4: locked-test claim

The claim "exceeds NVIDIA k20 with less accuracy loss" requires all of:

- the nominated R=1 configuration was not changed after unsealing;
- positive Holm-adjusted evidence on the registered primary macro fidelity
  endpoint and at least two degraded task families;
- no safety/non-inferiority margin is violated;
- all results, including regressions and cap/stop diagnostics, are reported;
- hashes and a validator receipt reproduce from raw records.

If G4 fails, record the result and return to method-dev under a **new protocol
version**.  Do not run another locked-test variant in the same confirmatory
family.  Hybrid R=8/R=32 claims are labeled separately and include the target
compute cost.

## 9. Reporting language

Permitted before G4: "candidate", "hypothesis", "method-dev improvement".

Permitted after G4 only: "outperforms the local NVIDIA-compatible k20 baseline
under the frozen Qwen3-32B→14B protocol".

Never claim bit-exact reproduction of unpublished paper numbers, never merge
pure and hybrid results, and never generalize local 4090 latency to H100/NVLink.
