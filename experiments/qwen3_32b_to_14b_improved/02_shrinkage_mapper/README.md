# Qwen3-32B -> 14B: head-wise shrinkage mapper

This directory is an isolated first candidate for improving the NVIDIA
top-20 ridge mapper.  It does not modify the original reverse experiment,
its code, or its artifacts.

For every target layer, cache component (content-space K or V), and target KV
head, the candidate is

```text
f_alpha(x) = f4(x[:4096]) + alpha * (f20(x) - f4(x[:4096]))
```

The independently refitted top-4 source-layer ordering is first required to
equal the first four entries of the top-20 ordering.  The resulting function
is fused into one 20480 -> 1024 FP32 affine map, so runtime still performs one
matrix multiplication per target layer and component.

## Data separation

The existing 50 held-out FineWeb-Edu sequences are split from token identity
alone, before looking at activation errors:

- the ten token rows with the lowest SHA256 values are a locked audit set;
- the remaining forty rows form five sequence-level CV folds;
- each fold fits alpha on the other 32 tuning sequences and evaluates on its
  eight validation sequences;
- the final alpha is refit on all forty tuning sequences;
- only after every final alpha is frozen is the ten-sequence audit evaluated.

The locked audit is a reconstruction audit, not a downstream benchmark and
not a source of hyperparameter feedback.  A later downstream evaluation must
remain paired against the unchanged k20 baseline.

## Commands

Use the repository environment explicitly because the login-shell Python is
not the experiment environment.

```bash
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python

$PY -m unittest discover -s tests -v
$PY src/shrinkage_mapper.py preflight --deep-hash
$PY src/shrinkage_mapper.py run
$PY src/shrinkage_mapper.py validate
```

`run` takes the repository-wide GPU lock, requires twelve consecutive idle
samples, exposes only physical GPU 0, loads no language model, and records the
gate, exact command, code/config hashes, CUDA peak memory, sufficient
statistics, mapper payload hashes, and completion receipt.  It refuses to
overwrite an existing run or mapper.

