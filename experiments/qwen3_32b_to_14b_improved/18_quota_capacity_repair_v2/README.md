# 18 — quota capacity repair v2 (CPU PRE-FORMAL)

This directory is an independent, answer-free consumer of the immutable
pairwise-v3 graph.  It addresses the v3 result that the original 18 exact
quotas are infeasible under the published direct-collision hyperedges.  The
new method does not restore quarantined rows, weaken a threshold, merge
overlapping hyperedges, or inspect a model outcome.  It searches for the
largest *capacity-safe* quota vector and leaves SHA-cost row selection as a
separate stage.

The protocol is intentionally stopped before formal/GPU work.  The required
`allocation_v3_deep` receipt is now bound as a read-only READY gate, while the
three independent blocked completion markers below remain immutable historical
evidence from the earlier missing-gate snapshot:

* `artifacts/quota_search/PUBLISHED.json` — Stage A/B/C placeholder;
* `artifacts/allocation/PUBLISHED.json` — Stage D placeholder;
* `artifacts/validations/deep/PUBLISHED.json` — independent deep-stage
  placeholder.

No marker is a claim that a repaired quota or selection has been solved; the
READY snapshot likewise does not execute A/B/C/D.

`PRE_FORMAL_STOP_v12.json`, `PRE_FORMAL_STOP_v13.json`, `PRE_FORMAL_STOP_v14.json`,
and `PRE_FORMAL_STOP_v15.json` are retained as superseded append-only evidence.
The current source-closure/output-safety snapshot is published as v16; its
config binding is activated only after the marker and its sidecar are written.

## Frozen optimization

There are exactly 18 targets, in the public pairwise-v3 order.  Their upper
bounds are copied byte-for-byte from the v3 solver contract.  For each cell,

```
q_screen <= U_screen,  q_confirm <= U_confirm,
abs(q_confirm - 2*q_screen) <= 1.
```

The MuSiQue 4-hop bound remains 117 selected rows from 119 strict-clean rows,
leaving two unselected rows.  Stage A uses an integer-safe finite-breakpoint
certificate: all attainable values `q_t/U_t` are enumerated as exact
fractions, monotone MILP feasibility is binary-searched, and the first higher
breakpoint must be explicitly infeasible.  A continuous lambda solve is kept
only as a diagnostic cross-check; it never decides the final fairness value.

Stage B fixes the proven rational lambda floor and maximizes total selected
rows.  Stage C fixes that total and maximizes each target quota sequentially in
the frozen public target order, yielding a unique lexicographically largest
quota vector.  Stage D then minimizes the existing
`qwen3-32to14-improved/pairwise-hypergraph-v2` SHA-cost objective with two
single-thread replicas (`PYTHONHASHSEED`/HiGHS seeds 1 and 777).  Distinct PIDs,
strict typed status, finite binary incumbents, and byte-identical selections
are mandatory.

## Run only the safe pre-formal stop

```bash
ROOT=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/18_quota_capacity_repair_v2
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
cd "$ROOT"
PREFLIGHT_PARENT=$(mktemp -d /tmp/qcr_v2_preflight.XXXXXX)
PREFLIGHT_CACHE="$PREFLIGHT_PARENT/fresh"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=1 PYTHONPYCACHEPREFIX="$PREFLIGHT_CACHE" "$PY" scripts/preflight.py --config "$ROOT/config.json"
rmdir "$PREFLIGHT_PARENT"
TEST_PARENT=$(mktemp -d /tmp/qcr_v2_tests.XXXXXX)
TEST_CACHE="$TEST_PARENT/fresh"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPYCACHEPREFIX="$TEST_CACHE" "$PY" -m pytest -q -p no:cacheprovider tests
rmdir "$TEST_PARENT"
PUBLISH_PARENT=$(mktemp -d /tmp/qcr_v2_publish.XXXXXX)
PUBLISH_CACHE="$PUBLISH_PARENT/fresh"
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPYCACHEPREFIX="$PUBLISH_CACHE" "$PY" scripts/publish_ready_snapshot.py --config "$ROOT/config.json" --version v16
rmdir "$PUBLISH_PARENT"
```

Every entrypoint requires `PYTHONPYCACHEPREFIX` to name an initially absent
absolute path whose existing parents are real non-symlink directories;
`sys.pycache_prefix` must match it and `sys.dont_write_bytecode` must already be
`True`.  `PYTHONDONTWRITEBYTECODE=1` is mandatory and no command supplies an
implicit cache directory.  `run_preformal.py` refuses a ready gate: any future
CPU repair must be an explicitly audited invocation of
`src.runner.run_cpu_repair(..., allow_execution=True)`, followed by the
separate deep validator.  There is no GPU launcher in this directory.

## Downstream native-registry interface

`contracts/native_registry_projection_contract.json` and
`src/registry_adapter.py` define the hand-off to dirs 10/12/13.  Stage-D rows
are split into `hotpotqa_screen`, `hotpotqa_confirm`, `musique_screen`, and
`musique_confirm`, retaining `stratum`, `source_index`,
`source_projection_sha256`, and zero-based stratum-local `selection_rank`.
`sample_id` and `selection_rank_sha256` are deliberately not fabricated: a
later adapter must resolve them from pinned train sources and pass through the
exact registry receipt `train_sources` and `answer_access_policy`.  This
package never reads answers or model outcomes.
