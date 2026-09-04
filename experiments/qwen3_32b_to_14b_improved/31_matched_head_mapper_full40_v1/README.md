# CacheBridge Head-Local full40 evaluation (32B → 14B), v1

This is an independent full-40-layer sibling of the frozen dir30 probe.  The
dir30 snapshot is recorded only as read-only mechanistic context; it is not a
QA result and cannot authorize this run.

For every target layer `0..39`, cache component, and KV head `h`, the mapper
fits the same-head affine ridge map from the 20 source layers selected by the
immutable NVIDIA k20 manifest:

`X_h ∈ R^(N×2560)`, `Y_h ∈ R^(N×128)`,
`W_h=(X_hcᵀX_hc+0.01I)⁻¹X_hcᵀY_hc`,
`b_h=mean(Y_h)-mean(X_h)W_h`.

The candidate is fitted only on the independent 500-sequence `train` split
(128,000 sampled observations per layer/component), then evaluated only on the
40 `heldout_tuning` sequences (10,240 observations).  Fit and evaluation roots
and token files are distinct and are checked byte-for-byte before execution;
numeric sequence IDs are treated as split-namespaced IDs.  The ten locked
sequences, answers, and downstream outcomes are never indexed.  The preflight
independently verifies all 40 per-target source selections, all 64 source
activation layers, all 40 target activation layers, and both baseline mapper
payloads for all 40 layers.

The NVIDIA k20 mapper (fit on train500) is the primary clean generalization
baseline.  The existing headwise-shrinkage payload is retained as a continuity
comparator, but its alpha was selected on heldout tuning40; it is therefore
labelled `validation_tuned_on_heldout_tuning40_exploratory_only` and cannot
support a clean generalization claim.  The six-layer dir30 result is a
read-only mechanistic prior, not QA evidence.

## CPU-only preflight

Use the pinned venv and an absent external pycache leaf.  This command hashes
inputs and writes immutable preflight/STOP receipts but does not import torch,
load a model, or call CUDA:

```bash
ROOT=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/31_matched_head_mapper_full40_v1
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
CACHE_PARENT=$(mktemp -d)
CACHE_PREFIX="$CACHE_PARENT/fresh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$CACHE_PREFIX" \
  "$PY" scripts/run.py --config "$ROOT/config.json" --preflight-only
rmdir "$CACHE_PARENT"
```

Only a separately audited operator may run the future full40 stage:

```bash
ROOT=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/31_matched_head_mapper_full40_v1
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
CACHE_PARENT=$(mktemp -d)
CACHE_PREFIX="$CACHE_PARENT/fresh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="$CACHE_PREFIX" "$PY" scripts/run.py \
  --config "$ROOT/config.json" --stage full40 --execute
rmdir "$CACHE_PARENT"
```

The runner computes candidate, NVIDIA-k20, and shrinkage SSE for all 40 × 2
rows on heldout tuning40.  It emits `GO` only if candidate SSE is strictly
lower than both baselines in every row; otherwise it emits `NO_GO` and no
downstream promotion is permitted.  Because shrinkage is
validation-tuned, the primary scientific comparison is candidate versus the
train500 NVIDIA k20 baseline; this sibling still keeps the stricter
candidate-below-both gate as a conservative exploratory check.  This is an
activation-geometry result, not an accuracy claim.
