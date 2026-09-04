# k4 + matched-head convex blend (full40, CPU gate)

This is an independent candidate sibling.  It does not modify the immutable
`32_matched_head_mapper_full40_v2` directory and it has no downstream or locked
dataset access.

## Method

For each target layer and cache component, the frozen NVIDIA k4 affine map and
the frozen train500 CacheBridge Head-Local map are expressed in the same
canonical source coordinate system:

`x20 = [selected k20 layer rank, KV head, head dimension]` (width 20×8×128).

The k4 selection is checked to be an ordered subset of the k20 selection.  Its
weight is embedded by `W4_up = E.T @ W4`, where `E` selects the exact k4 ranks.
The head-local tensor `[8, 2560, 128]` is embedded as a block-diagonal dense
`WC_up[20480,1024]`; all cross-head blocks are zero.  A scalar or per-output
convex coefficient then gives the ordinary downstream-compatible affine map:

`W = (1−α) W4_up + α WC_up`, `b = (1−α)b4 + αbC`.

The implementation includes an exact synthetic check that dense prediction is
identical to direct prediction-level blending, including nontrivial selection
order.  The activation axis helper is tested for batch sizes 1, 2, and 3.

## Split and claim boundary

The available k4 and head-local maps are frozen fits on all 500 train
sequences.  Therefore α is a **post-fit train-only diagnostic** (same train500
namespace), and the five-fold values are stability summaries, not out-of-fold
estimates.  This sibling must not call that procedure unbiased CV.  A future
clean attempt should refit both arms on a disjoint train-fit subset before
choosing α, then refit on all train500.

The forty `heldout_tuning` sequences were already inspected by a separate
read-only diagnostic.  Any heldout result from this sibling is exploratory and
cannot support a generalization or QA claim.  The pre-registered safety gate
is aggregate SSE strictly below NVIDIA k20 and every layer/component ratio at
most 1.05; it is not a superiority gate.  NVIDIA k20 remains the primary
clean comparator, k4 is a secondary comparator, and the old shrinkage α is
marked validation-tuned/exploratory.

## CPU-only preflight

Run from a fresh external cache prefix.  The command hashes the local source
closure, k4/k20 manifests and payloads, model metadata refs, activation
manifests, tokens, and the immutable dir32 failure evidence.  It does not
import torch, load a model, or start CUDA.

```bash
ROOT=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/33_k4_headlocal_blend_full40_v1
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
CACHE_PARENT=$(mktemp -d)
CACHE_PREFIX="$CACHE_PARENT/fresh"
cd "$ROOT"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPYCACHEPREFIX="$CACHE_PREFIX" \
  "$PY" scripts/run.py --config "$ROOT/config.json" --preflight-only
rmdir "$CACHE_PARENT"
```

After CPU tests and an independent audit, an operator may publish the
append-only PRE_FORMAL_STOP receipt with `--publish-stop --tests N
--compile-files M`.  No GPU stage is implemented or authorized in this
snapshot; a later stage must use an explicit `--execute`, the global lock, a
fresh output root, and the same source/input hashes.

## Downstream QA

Only a future activation GO on a fresh, uninspected evaluation split can open a
downstream sibling.  That sibling must use the native dense map interface and
the same prompt/decoding/cache rows for 14B, NVIDIA k20, k4, and the blend.
Answers are excluded from registry projection and may be read only by the
oracle metric stage (`gold_first_token_margin`, `qa_f1`, `qa_em`).

