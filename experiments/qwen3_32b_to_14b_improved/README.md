# Qwen3-32B -> Qwen3-14B improved cache-transfer research

This tree is an independent research workspace.  The completed NVIDIA-paper
reproduction under `../qwen3_32b_to_14b/` is immutable input: nothing in this
tree may overwrite its code, mapper weights, run directories, or receipts.

## Directory contract

| Directory | Purpose | May write GPU artifacts? |
|---|---|---|
| `00_research_protocol/` | Evidence, frozen data splits, metrics, statistics, and go/no-go rules | No |
| `01_oracle_diagnostics/` | Exact-K/V and exact-layer causal interventions | Only inside this directory |
| `02_shrinkage_mapper/` | Pure affine k4/k20 shrinkage mapper | Only inside this directory |
| `03_tail_refresh/` | k20 mapper plus native 14B prompt-tail replay | Only inside this directory |
| `04_sarr_functional/` | Function-aligned low-rank residual mapper | Only inside this directory |
| `05_locked_comparison/` | One-shot comparison of frozen winners on locked tests | Only inside this directory |
| `06_musique_benchmark/` | Pinned MuSiQue method-dev and sealed challenge benchmark | Only inside this directory |
| `07_method_dev_evaluation/` | Preserved v1 screening harness; its old registry is contaminated and must not be reused | No new formal run |
| `08_cross_tier_audit/` | Partial mapper-fit vs. tail-pilot collision audit; explicitly not complete G0 | No |
| `09_complete_g0_audit/` | Append-only complete cross-tier collision audits and fail/pass receipts | No |
| `10_decontaminated_method_dev_registry/` | Gold-independent HotpotQA/MuSiQue train registry construction; every failed attempt is preserved | No |
| `11_watchdog_tail_repair/` | Post-hoc two-pass/watchdog hypothesis from the old pilot; exploratory evidence only | No new formal run |
| `12_prospective_screen_v2/` | Fresh, decontaminated prospective comparison of native14, NVIDIA k20, shrinkage, R32 and the frozen two-pass policy | Only inside this directory |
| `13_oracle_decontaminated_v2/` | Planned causal K/V/layer diagnostics on the final clean method-dev registry | Only inside this directory |

Every method directory owns its own `configs/`, `src/`, `scripts/`, `tests/`,
`artifacts/`, `runs/`, and `logs/`.  Cross-directory products are referenced by
absolute path plus byte size and SHA-256; they are never silently copied or
modified.

## Scientific separation

1. `mapper-fit`: unsupervised activation/text data used to fit mapper weights.
2. `method-dev`: disjoint task-training examples used to choose rank, shrinkage,
   replay length, gates, and stopping rules.
3. `locked-test`: existing formal test/dev sets.  They are evaluated only after
   a candidate and all thresholds are frozen.

Historical formal results may motivate hypotheses, but may not select a new
method's hyperparameters.

## Current experiment status (2026-08-31)

- `02_shrinkage_mapper/` is complete at the activation level: the frozen
  affine candidate improves held-out K R2 by 2.05 percentage points and V R2
  by 3.17 points, but it has not yet earned a downstream-quality claim.
- `03_tail_refresh/` completed its first train-only pilot. Fixed R8/R32 did not
  pass the preregistered per-dataset rule, so neither is a winner.
- `11_watchdog_tail_repair/` found a useful post-hoc length/cap signal. Because
  it was discovered after observing the pilot and its online/offline boundary
  was ambiguous, it is only a hypothesis generator.
- `10_decontaminated_method_dev_registry/` completed one component-closure
  feasibility attempt and failed closed: transitive similarity connected the
  clean MuSiQue 4-hop pool into fewer than two components. The failure is
  preserved append-only. A new independent pairwise-hypergraph version keeps
  every direct exact/substring/near collision constraint without propagating
  it through unrelated transitive chains. No threshold is relaxed and no
  locked outcome is read by selection logic.
- `12_prospective_screen_v2/` is the next valid downstream experiment. GPU
  execution is forbidden until `10` publishes a clean registry and `09`
  publishes a complete zero-collision G0-v2 receipt.

No current artifact supports the statement that the new method already beats
the NVIDIA baseline. That claim requires a frozen, paired prospective screen
and confirmation result, with pure-mapper and target-replay costs separated.

## Baseline

The primary comparison is the immutable production k20 mapper:

`../qwen3_32b_to_14b/artifacts/mappers/qwen3_32b_to_14b/k20_lambda0.01/manifest.json`

The independently trained k4 mapper is a shrinkage anchor, not a replacement
baseline:

`../qwen3_32b_to_14b/artifacts/mappers/qwen3_32b_to_14b/k4_retuned_lambda0.01/manifest.json`

## Reporting rule

Mapper-only (`R=1`) and mapper-plus-target-replay (`R>1`) results are reported
separately.  A replay result cannot be presented as a zero-target-prefill
mapper result.  Accuracy/F1/PPL, target-decision agreement, target-correct
retention, EOS/cap behavior, latency, peak VRAM, and artifact size are all
required for a claim of improvement.
