#!/usr/bin/env bash
export KV_LINGO_ROOT=/home/tiger/kvcache-mapped
export TMPDIR="$KV_LINGO_ROOT/.tmp/kv_lingo"
export HF_HOME="$KV_LINGO_ROOT/.cache/huggingface"
export UV_CACHE_DIR="$KV_LINGO_ROOT/.cache/uv"
export TORCH_HOME="$KV_LINGO_ROOT/.cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$KV_LINGO_ROOT/.cache/inductor"
export TRITON_CACHE_DIR="$KV_LINGO_ROOT/.cache/triton"
export XDG_CACHE_HOME="$KV_LINGO_ROOT/.cache"
export PYTHONPYCACHEPREFIX="$KV_LINGO_ROOT/.cache/pycache"
export NLTK_DATA="$KV_LINGO_ROOT/.cache/nltk"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$KV_LINGO_ROOT/experiments/v_readout_ridge"
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$HF_HOME" "$UV_CACHE_DIR" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$NLTK_DATA"
