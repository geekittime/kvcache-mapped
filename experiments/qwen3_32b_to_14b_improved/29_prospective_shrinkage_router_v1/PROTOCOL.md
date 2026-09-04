# Pre-registration protocol (v1)

## Scope and information boundary

This directory is CPU-only.  The frozen router is a prospective policy, not a
model-quality result.  Runtime routing may read only:

* the exact token count of the complete prompt under the pinned tokenizer;
* the number of tokens emitted by the completed NVIDIA/R1 pass; and
* a boolean indicating that the first pass completed and its cache transfer is
  safe.

The router must not receive a dataset/task identifier, sample ID, text output,
reference answer, score, cap flag, wall time, GPU/VRAM state, or any derived
quality signal.  Offline evaluation may use labels after decisions are
immutable.

## Fresh registry and split

Before any model invocation, an independent registry owner must publish a
manifest satisfying `src.registry.validate_manifest`:

* a new registry ID and exact tokenizer SHA-256;
* disjoint `calibration` and `sealed_eval` sample-ID sets;
* at least 24 calibration and 48 sealed examples for each of HotpotQA and
  MuSiQue (or an explicitly recorded shortfall that forces NO-GO);
* immutable source-record hashes and no duplicate IDs within or across splits;
* a declaration that labels are unavailable to the routing process.

Calibration is operational only (tokenizer/counting, budget, and telemetry
checks).  It cannot be used to tune thresholds, priority, budget, or gates.
The sealed partition is the sole source of the primary quality claim.  If any
sealed result is inspected before the policy is frozen, the run is exploratory
and cannot be promoted.

## Frozen route and controls

The route is `shrinkage_r1` exactly when both `prompt_tokens >= 2800` and
`r1_generated_tokens >= 16`, the first pass is complete, and budget remains.
Otherwise it is `nvidia_k20`.  Invalid telemetry and budget overflow are
fail-closed `no_go` outcomes.  The budget is
`min(32, floor(0.30 * n))`, with `n` fixed before the batch starts and stable
original-order tie breaking.

The fresh registry must collect paired outputs for `native14`, `nvidia_k20`,
and `shrinkage_r1` (the latter at least for all routed candidates).  The three
arms are reported separately; the primary contrast is router policy versus
always-`nvidia_k20`.

## Statistical gates (declared before sealed evaluation)

Use the deterministic stratified paired bootstrap in `src.evaluate` (10,000
replicates, seed 2901, resampling within each dataset while preserving its
sample count).  For the router-minus-NVIDIA relaxed official F1:

1. each dataset mean must be at least 0.0 (no average regression);
2. HotpotQA mean F1 must be at least −0.01 and mean EM at least −0.02
   (pre-registered safety margins; a stricter zero margin is preferred);
3. the combined paired 95% CI lower endpoint must be strictly greater than 0
   for a superiority claim;
4. mean EM must not be below −0.02 in either dataset;
5. cap rate may increase by at most 0.01 and complete/EOS rate may decrease by
   at most 0.01 relative to NVIDIA;
6. invalid/no-go decisions must be ≤5%, and actual escalations must not exceed
   the frozen budget.

Failure of any safety gate is `NO-GO`; a positive point estimate without a
positive CI is exploratory only.  The native14 arm is a reference and does not
enter route selection or gate tuning.  No threshold or rule may be changed
after sealed outcomes are observed.

## Claim boundary

The old `03_tail_refresh` pilot and completed `28_shrinkage_old_pilot_eval_v1`
artifacts are bound only as post-hoc hypothesis evidence.  This CPU package
must never claim that the router generalizes or beats the NVIDIA paper until a
fresh sealed registry passes every gate and an independent analysis receipt
verifies the hashes.
