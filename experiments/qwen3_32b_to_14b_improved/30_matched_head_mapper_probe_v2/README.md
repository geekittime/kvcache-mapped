# CacheBridge Head-Local matched-head probe (32B → 14B), v2

This is an independent successor to `22_matched_head_mapper_probe_v1`; the
v1 CPU snapshot is retained as immutable upstream evidence. v2 additionally
requires the exact configured venv executable and implements the promotion
gate in the runner itself.

This independent experiment tests a narrower mapper than the NVIDIA-style
joint `k20` map.  For one target layer, cache component, and KV head `h`, its
input is the same head from the frozen 20 selected Qwen3-32B layers:

`X_h ∈ R^(N×2560)`, `Y_h ∈ R^(N×128)` and
`W_h=(X_hcᵀX_hc+0.01I)⁻¹X_hcᵀY_hc`,
`b_h=mean(Y_h)-mean(X_h)W_h`.

The objective is ordinary unweighted SSE. The initial probe is registered for
target layers `[0, 7, 15, 23, 31, 39]`; a full 40-layer stage is permitted only
after all six layers beat both frozen `k20` and shrinkage baselines in SSE for
both key and value. The probe may index only the 40 tuning sequences from the
frozen `02_shrinkage_mapper` split. The 10 locked activation rows, downstream
answers, and downstream outcomes are prohibited.

## CPU-only preflight

The preflight hashes all 25 model shards and loader assets, all selected
activation payloads/metadata, both baseline mapper payloads/sidecars, the
frozen split/token refs, and the complete local source closure. It does not
import torch, load a model, or call CUDA.

```bash
ROOT=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/30_matched_head_mapper_probe_v2
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
CACHE_PARENT=$(mktemp -d)
CACHE_PREFIX="$CACHE_PARENT/fresh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$CACHE_PREFIX" \
  "$PY" scripts/run.py --config "$ROOT/config.json" --preflight-only
rmdir "$CACHE_PARENT"
```

The absent `fresh` leaf is intentional. CPU tests use the same bootstrap and
`pytest -p no:cacheprovider`. No GPU command is authorized by the
`PRE_FORMAL_STOP` receipt; an operator must independently audit the snapshot
before invoking the exact future `probe6 --execute` command recorded there.

The runner computes candidate, NVIDIA-k20, and shrinkage SSE for every one of
the six layers × two components. It emits `GO` only when the candidate is
strictly lower than both baselines in every row; otherwise it emits `NO_GO`.
`full40` rejects any probe result lacking this exact gate, so an editable
`PENDING` marker cannot promote a run.

## Claim boundary

Until the six-layer activation probe and a later downstream HotpotQA
method-development evaluation are completed, this directory supports only a
mechanistic hypothesis. It does not establish an accuracy improvement or a
paper-superiority claim.
