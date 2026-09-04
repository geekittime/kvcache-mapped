# Qwen3 32B → 14B KV-cache mapping reproduction

This is a source-only export of the Qwen3 32B → 14B reproduction work.  It
contains the implementation, protocol/configuration files, experiment
drivers, and CPU/GPU test code.  Historical attempts are kept in their
numbered directories so that the development and audit trail remains
inspectable; an attempt marked aborted, superseded, or `NO_GO` is not a claim
of a deployable method.

## Repository layout

* `src/kvcache_transfer/` — shared model, cache, mapper, ridge, RoPE, metric,
  and evaluation primitives.
* `scripts/`, `tests/`, `configs/`, `pyproject.toml`, `uv.lock` — shared
  utilities and unit tests used by the experiment drivers.
* `experiments/qwen3_32b_to_14b/` — the baseline reverse-direction pipeline
  (selection, fitting, held-out reconstruction, HellaSwag/CoQA/latency
  extensions, and their tests).
* `experiments/qwen3_32b_to_14b_improved/` — numbered protocol, audit, and
  method-development attempts, including their `src/`, `scripts/`, and
  `tests/` trees.

The root-level shared utilities also contain some direction-neutral or legacy
benchmark helpers.  The reverse-direction entry points and the authoritative
configs are under the two `qwen3_32b_to_14b` experiment trees above.
The shared package's historical default architecture is still the forward
14B → 32B spec; reverse adapters deliberately pass an explicit 32B → 14B
architecture, so use the reverse experiment configs rather than relying on a
root-level default.

## What is deliberately not included

No dataset, prompt/response corpus, model checkpoint, activation store,
mapper/statistics payload, benchmark output, run log, vendored dependency,
virtual environment, or Python cache is committed.  This keeps the repository
small and avoids publishing local data.  The `.gitignore` applies the same
policy to future runs.  The configs and historical protocol documents retain
some machine-specific paths and hashes for provenance; update those paths (or
use the provided templates) after cloning.

The paper's `lm-evaluation-harness` checkout is also intentionally omitted.
Install or clone the pinned harness separately and point the evaluation
configuration at it before running lm-eval-based benchmarks; the CPU/unit
tests that use synthetic fixtures do not need that checkout.

There is currently no license file in the source project; no license is
asserted by this export.  Add the appropriate license before redistributing
the code.

## CPU-only checks

From a checkout with the project dependencies installed:

```bash
export PYTHONPATH="$PWD/src"
python -m pytest -q tests
python -m pytest -q experiments/qwen3_32b_to_14b/tests
```

The numbered improved attempts can be tested independently from each attempt
directory, for example:

```bash
REPO_ROOT="$PWD"
cd "$REPO_ROOT/experiments/qwen3_32b_to_14b_improved/32_matched_head_mapper_full40_v2"
PYTHONPATH="$REPO_ROOT/src" python -m pytest -q tests
```

Tests that validate frozen artifacts or run a model intentionally require the
omitted local data/checkpoint/mapper inputs; without those inputs they should
fail closed rather than silently fabricate results.  Do not run a GPU stage
until its input preflight and protocol gates have passed.

## Recreating the reverse pipeline

The baseline README and `REPRODUCTION.md` give the exact stage order.  In
outline, provision the pinned Python dependencies and Qwen3 checkpoints,
prepare the 32B/14B activation stores, run source-layer selection, fit the
40 target-layer K/V maps, validate on the held-out capture, and then run the
registered evaluation scripts.  Keep generated files in the ignored
`data/`, `artifacts/`, `runs/`, and `logs/` locations.

The exported code is intended for reproducibility and review, not as a claim
that every historical exploratory method improves the paper baseline.
