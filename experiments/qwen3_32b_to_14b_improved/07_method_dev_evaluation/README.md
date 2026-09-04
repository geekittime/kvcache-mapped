# Qwen3-32B to Qwen3-14B method-development evaluation

This directory is an isolated, paired evaluation harness. It never selects on
HotpotQA dev or MuSiQue official dev. It compares the same native Qwen3-14B
target, immutable NVIDIA-compatible k20 baseline, finalized head-wise
shrinkage candidate, and k20 plus native-target tail refresh at R=8 and R=32.
The raw k20 mode is the R=1 pure-mapper comparison; the frozen tail interface
also validates its own exact R=1 zero-operation wrapper contract.

## Data boundary

- HotpotQA uses only official train v1.1. Complete records with repeated
  NFKC/casefold/whitespace-normalized questions are quarantined. Screen and
  confirmation are disjoint SHA-256 ranks over namespace, NUL, and record ID:
  128 screen rows and a separate 256 confirmation rows.
- MuSiQue consumes the exact 06 registry_v1 train_screen (256) and
  train_confirm (512) identities. The official-dev file is never opened; only
  its sealed commitment is verified.
- Smoke is the first two frozen screen identities per dataset. It is an
  infrastructure check, not a model-selection estimate.
- All five modes receive the same exact prompt token IDs, max length 16,384,
  generation cap 256, deterministic greedy decoding, and registered stop
  strings. Overlength prompts are deterministically truncated once at the
  token-ID level and those exact used IDs are frozen before execution.

The MuSiQue instruction, paragraph ordering, cap, metric implementation, and
registry are content-bound to 06. HotpotQA uses the same shortest-answer-span,
no-explanation instruction semantics. Headline F1/EM always score the raw
decoded prediction with official answer normalization. Gold-independent final
answer extraction is stored only as a diagnostic.

## Content-addressed method inputs

Every external input is an absolute path plus byte size plus SHA-256. The
runner verifies:

- the completed reverse long-context run definition and its current code and
  model inventory;
- the immutable NVIDIA k20 mapper manifest;
- 02 shrinkage mapper, amended completion receipt, algebra audit receipt, and
  ShrinkageMappedPrefixCacheBuilder implementation;
- 03 CPU preflight receipt, frozen interface receipt, and
  TailRefreshMappedPrefixCacheBuilder implementation.

The tail builder receives C[:-1], maps C[:P-(R-1)], appends exactly R-1 native
14B tokens, and leaves C[-1] to the shared evaluation engine. R=8 therefore
means 7 builder tokens plus one boundary token; R=32 means 31 plus one.

## CPU-only preparation and checks

Run from this directory with the repository environment:

    PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
    PYTHONDONTWRITEBYTECODE=1 $PY -m pytest -q tests --basetemp artifacts/pytest_tmp
    PYTHONDONTWRITEBYTECODE=1 $PY scripts/prepare.py --dry-run
    PYTHONDONTWRITEBYTECODE=1 $PY scripts/prepare.py --prepare
    PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_pilot.py --profile smoke --contract-only
    PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_pilot.py --profile screen --contract-only
    PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_pilot.py --profile confirm --contract-only

These commands do not call a CUDA API. Preparation writes only below this
directory's artifacts/prepared_v1. A changed config, code file, data byte,
registry, receipt, prompt ID, or tokenizer identity invalidates the prepared
manifest and requires preparation again.

## Guarded GPU execution (not run during implementation)

Only an explicit execute flag can cross the GPU boundary. The runner then
requires all eight visible devices, takes the repository-wide GPU lock, deep
hashes model shards, verifies both candidate interfaces, waits for the frozen
idle gate, and only then loads models.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_pilot.py \
      --profile smoke --execute --output-dir runs/smoke_v1

After smoke infrastructure validation, screen runs before confirmation. The
pre-registered confirmation gate is outcome-independent: it requires a
CPU-validated completed screen receipt, passed explicitly to the confirm
command, and records that receipt in the confirm run definition.

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_pilot.py \
      --profile screen --execute --output-dir runs/screen_v1

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      PYTHONDONTWRITEBYTECODE=1 $PY scripts/run_pilot.py \
      --profile confirm --execute --output-dir runs/confirm_v1 \
      --screen-output-dir runs/screen_v1

Within each sample the immutable order is native14, nvidia_k20, shrinkage,
tail_refresh_r8, tail_refresh_r32. Every non-target checkpoint binds the
target checkpoint SHA; every candidate checkpoint also binds the k20
checkpoint SHA. Complete checkpoints are validated and resumed, while an
invalid existing checkpoint fails closed. A failure writes a separate atomic
failure record and stops the run.

Each checkpoint records exact generated token IDs and their SHA, their exact
decode, the score-bearing prediction after a registered stop string if any,
stop reason, cap hit, official F1/EM, prompt truncation, method binding,
wall-clock time, and peak GPU memory. Tail modes additionally record the
causal partition and position/cache audit.

On completion, the runner computes paired per-example deltas, deterministic
10,000-replicate percentile bootstrap intervals, raw score retention versus
native 14B, target-correct EM retention, gap closure only when native exceeds
k20, F1 win/tie/loss counts, and exact-match 2x2 flip tables. It writes a
content-addressed completion receipt only after every paired cell validates.

Validate a completed output without using a GPU:

    PYTHONDONTWRITEBYTECODE=1 $PY scripts/validate_run.py \
      --profile smoke --output-dir runs/smoke_v1

All results remain method-development evidence. Pure k20/shrinkage and hybrid
R=8/R=32 results are reported separately and cannot support a locked-test or
paper-superiority claim.
