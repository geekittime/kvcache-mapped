# Evidence review and method hypothesis

Updated: 2026-08-30

## 1. Scope and evidence labels

The goal is functional fidelity to native Qwen3-14B when a prompt cache is
translated from Qwen3-32B.  Native 14B is therefore the teacher and reference.
Using 32B logits as a teacher would turn the experiment into capability
distillation and requires a separate protocol.

This document uses three evidence levels:

- **Published evidence**: a primary paper reports the claim, possibly in a
  different model or cache-reuse setting.
- **Local measurement**: an existing, hash-addressed artifact in
  `baseline_refs.json` supports the claim.
- **Hypothesis**: a proposed causal explanation or method not yet established
  on frozen Qwen3-32B→14B.

No published method currently establishes superiority on this exact frozen
Qwen3-32B→14B, no-target-prefill protocol.  Related work motivates the
design but cannot substitute for the planned ablations.

## 2. What the NVIDIA baseline establishes

The NVIDIA method selects source layers, fits K in content space and V
directly with top-k ridge regression, then builds a target-model cache.  The
paper reports that attention-output similarity predicts retention better than
calibration reconstruction R²; its reported cross-pair Pearson correlations
are approximately `+0.57` for attention-output cosine and `-0.20` for
calibration K R².  It also reports that an MLP can substantially repair poor
model pairs but can slightly underperform ridge on already-good pairs.

Primary source: [Cross-Model KV Cache Reuse](https://arxiv.org/html/2608.03893).

The implication is limited but important: mean coordinate-space
reconstruction is not a sufficient objective, and unconstrained extra
capacity is not automatically safer.

## 3. Local facts that constrain the new method

All values below come from the content-addressed files in
`baseline_refs.json`.

### 3.1 Capacity and generalization

| mapper | train K R² | held-out K R² | gap | train V R² | held-out V R² | gap |
|---|---:|---:|---:|---:|---:|---:|
| k20 ridge | 0.84371 | 0.74503 | 0.09868 | 0.78879 | 0.65223 | 0.13657 |
| retuned k4 ridge | 0.77504 | 0.74223 | 0.03281 | 0.70146 | 0.65869 | 0.04278 |

The high-capacity k20 fit has a much larger train-to-held-out gap.  k4 has
better held-out V R², yet k20 has slightly better CoQA F1 (`0.72873` versus
`0.72519`).  Thus neither training R² nor held-out R² alone is a safe model
selection target.

### 3.2 Functional mismatch

On the local HellaSwag mechanism run, attention-output cosine is `0.76108`,
while content-space K R² is `0.69511` and V R² is `0.61743`.  Per-layer
attention failures do not rank the same way as Euclidean KV failures.  This
supports measuring the cache through native 14B queries and value readout,
but does not yet prove that directly optimizing that measure improves tasks.

### 3.3 Downstream failure modes

The completed five-task extension reports:

- HellaSwag `0.78699 → 0.77604`;
- WinoGrande `0.72928 → 0.71981`;
- GSM8K numeric-final `0.94996 → 0.90371`;
- WikiText-2 perplexity `7.50203 → 7.80223`;
- ARC and MMLU aggregate accuracy increase, despite substantial target/mapped
  decision flips.

Aggregate accuracy can therefore hide fidelity loss.  Paired decision
agreement, target-correct retention, and the correct→wrong/wrong→correct
matrix are mandatory.

CoQA falls from `0.78221` to `0.72873`.  Existing diagnostics show that many
mapped completions contain a reference answer but cross the desired answer
boundary.  GSM8K shows an even larger strict-format loss than numeric loss and
many long/capped generations.  These observations motivate suffix-logit and
EOS/stop calibration rather than merely increasing the generation cap.

### 3.4 Hardware reality

The local 8×RTX 4090 PCIe system has no NVLink/P2P.  The current FP32 k20
mapper is slower than native target reprefill in the local latency artifact.
Quality improvements do not imply a local serving speedup.  Latency claims
must report this hardware separately from the paper's H100/NVLink setting.

## 4. Related methods and what can safely transfer

| work | transferable idea | boundary of the evidence |
|---|---|---|
| [LatentAlign](https://arxiv.org/abs/2601.06123) | train a translator on target suffix behavior, not reconstruction alone | reported on much smaller models and short contexts |
| [Mixture of Translators](https://arxiv.org/abs/2607.28979) | correct early trajectory/last-state shift with target-context alignment | includes target-side correction/replay and a different scale |
| [KVSculpt](https://arxiv.org/abs/2603.27819) | allocate capacity by attention behavior and layer/head difficulty | same-model KV compression, not cross-model translation |
| [CacheBlend](https://arxiv.org/html/2405.16444) | selectively recompute attention-sensitive tokens | same-model modular RAG cache; cross-model benefit is unproven |
| [C2C](https://arxiv.org/abs/2510.03215) | target-layer gating and learned cache fusion | relies on native target-cache fusion, not pure prefix skipping |
| [See What I See](https://arxiv.org/abs/2606.13594) | two-stage reconstruction then generation alignment | protocol and model pairs differ from this frozen pair |
| [SwiftKV](https://arxiv.org/abs/2410.03960) | knowledge-preserving prefill distillation | changes model weights rather than translating between frozen models |

The common signal is functional alignment plus a native-target anchor.  None
of these papers proves that a large standalone MLP is the best next step here.

## 5. Primary hypothesis: SARR-KV

**SARR-KV (Shrinkage-Anchored Reduced-Rank KV mapping)** keeps the stable k4
ridge solution as an anchor and permits k20 information only through a gated,
standardized, low-rank residual:

\[
\hat y_{\ell,c}=f^{k4}_{\ell,c}(x_{S4})+
g_{\ell,c,h}\,B_{\ell,c}A_{\ell,c}z(x_{S20}).
\]

Here `c` is K or V, `z` is feature standardization, and the residual is
zero-initialized or initialized from the low-rank SVD of a cross-validated
k4/k20 shrinkage solution.  Gate zero is an exact fallback to k4.  Runtime
guards clip residual norm and smoothly close the gate for out-of-distribution
features.

The staged loss is:

\[
L=\lambda_{KV}L_{Huber}+
\lambda_A L_{attn\ JS}+
\lambda_O L_{attn\ output}+
\lambda_D L_{suffix\ KL}+
\lambda_E L_{EOS}+\lambda_R L_{regularization}.
\]

The hypothesis has three falsifiable parts:

1. shrinkage/standardization reduces k20's generalization gap without harming
   functional metrics;
2. a low-rank residual improves native-14B attention output more reliably than
   a dense MLP;
3. suffix/EOS alignment converts mechanism gains into lower task degradation.

Target-tail refresh (`R=8` or `R=32`) is a separate hybrid hypothesis.  It may
produce the largest quality gain by rebuilding the most recent target-model
states, but it spends native 14B prefill compute and cannot be presented as a
pure mapper result.

## 6. Strongest disconfirmation cases

The method should be rejected or revised if any of the following occurs:

- held-out R² rises but attention output and suffix KL do not improve;
- attention output improves but paired downstream confidence intervals do not;
- aggregate accuracy rises by converting target-wrong cases while
  target-correct retention falls;
- a result requires task test examples, gold answers, or repeated locked-test
  access for tuning;
- a hybrid replay gain is attributed to the pure mapper;
- quality improves but the deployment budget or latency claim is evaluated on
  incomparable hardware.
