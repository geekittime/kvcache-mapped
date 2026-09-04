# 24 — independent shrinkage retry on the frozen old-03 pilot

This is an independent, method-development-only downstream probe for the
head-wise shrinkage mapper from `02_shrinkage_mapper`.  It evaluates exactly
the 96 pilot prompts already frozen by old-03 (48 official HotpotQA-train and
48 official MuSiQue-train rows), using only the candidate `shrinkage_r1`
mapped path.  The already-completed old-03 `target_standalone` (native14) and
`mapped_r1` (NVIDIA-aligned k20) JSON results are read-only, hash-checked, and
content-bound into every paired record and the final receipt.  They are never
rerun.

The runner is resumable: a matching complete candidate checkpoint is skipped,
while any changed prompt, baseline artifact, mapper, config, or source byte
fails closed.  Candidate checkpoints and `paired_summary.json` are written by
fsync + atomic rename.  The exact old-03 prompt, 256-token cap, greedy decode,
two stop strings, and `truncation="error"` contract are retained.

If a prior attempt left `*.failure.json` beside a now-valid checkpoint, the
marker is content-bound and atomically recorded as
`*.failure.superseded.json` before the stale marker is removed.  A terminal
receipt is refused while any unsuperseded failure marker or output-tree
symlink remains.  The global GPU flock performs a second terminal-completion
check after acquisition, closing concurrent-resume races.

The model topology is fixed to 5+3 GPUs: Qwen3-32B layers `[12,13,14,13,12]`
on cards 0–4 and Qwen3-14B layers `[13,14,13]` on cards 5–7.  GPU execution
requires the repository eight-card lock, five idle-gate samples, and an
explicit `--execute`; preflight never imports CUDA/model code.

This is a fresh attempt directory after the dir23 finalization-only failure
(`NameError: completion_path` after all 96 candidate outputs were written).
Its config content-binds the dir23 log, idle-gate receipt, run definition,
status, paired summary, and a canonical inventory/hash of all 96 candidate
outputs as append-only prior-attempt evidence.  No dir23 result is reused as a
candidate output; the inventory is audited read-only during CPU preflight.
The lazy loaders register their module in `sys.modules` before `exec_module`
and restore the previous entry on import failure (required by Python 3.12
dataclasses).

## CPU-only freeze and STOP

```bash
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
cd /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/24_shrinkage_old_pilot_eval_v1
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/dir24_preflight_pycache \
  "$PY" scripts/run.py --preflight-only
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/dir24_test_pycache \
  "$PY" -m pytest -q
```

The preflight writes an immutable receipt and append-only
`STOP_PILOT_SHRINKAGE_R1_attempt24_v2.json`; dir23's failed receipts/results
remain historical evidence.
Do not replace `--preflight-only` with `--execute` until the operator has
audited the receipt and explicitly authorizes GPU use.

## Explicit GPU command (operator confirmation required)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/24_shrinkage_old_pilot_eval_v1/.formal_empty_pycache \
/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python \
  scripts/run.py --execute
```

This command is intentionally not launched by the preparation agent.
Outputs are method-development evidence only; they cannot authorize a locked
test or a paper-superiority claim.
