# Pairwise direct-collision hypergraph feasibility v3

This independent CPU-only v3 protocol replaces the over-conservative connected-component
closure from `component_feasibility_v1`. It does not weaken an exact,
normalized-substring, or near-shingle threshold and does not restore any row
quarantined against locked dev or mapper-fit data.

## Frozen failure and erratum lineage

The original component-v1 failure evidence mislabeled its last periodic
passage-scan checkpoint as a complete scan. Its frozen log event reports left
index `240000` of `244038`: that event was emitted after 240001 left units had
completed, leaving 4037. The values `39817022` candidates and `7797` direct
relations are therefore a partial checkpoint, not terminal counters. The
unchanged v2 implementation completed the same permutation-invariant unordered
unit-pair scan and correctly failed closed when its preregistered expectation
did not match the terminal `39831083` candidates and `7892` relations. It wrote
only `IN_PROGRESS.json` and no `PUBLISHED.json`.

V3 content-binds both append-only records:

- `component_feasibility_v1_partial_checkpoint_erratum.json` preserves and
  corrects only the mislabeled component-v1 field.
- `pairwise_hypergraph_v2_terminal_expectation_fail.json` preserves the v2
  full-scan failure, formal log, old expectation, terminal observation, and
  frozen 16-file implementation inventory.

V3 changes no source input, threshold, candidate/relation formula, owner-union
rule, quota, or solver rule. The passage scan now emits a separate terminal
event only after its loop: `phase=passage_similarity_complete`,
`completed_left_count=unique_units_total=244038`, with the final counters. The
event is persisted in `passage_similarity_statistics` and the structural and
deep validators reject a periodic `unique_units_processed` checkpoint in its
place.

The allocation solver, common publication primitives, allocation validator,
and all five entrypoint scripts are required to remain byte-identical to the
content-bound v2 files; this is enforced both before a formal build and by a
regression test.

Each direct collision clique is one hyperedge constrained by
`sum(record selected across every target) <= 1`. Exact-unit owner sets are
cliques. For a directly matching unit pair A/B, `owners(A) union owners(B)` is
a clique: owners within one unit match exactly and every cross-unit owner pair
inherits the direct A/B relation. Equal member sets are content-deduplicated
with derivation counts. Overlapping hyperedges are never transitively merged,
so `{a,b}` plus `{b,c}` does not prohibit selecting `a` with `c`.

Records are ordered by unique `source_projection_sha256`. A hyperedge ID is
SHA-256 over the canonical JSON line containing its ordered member projection
hashes; hyperedges are published in ID order. Raw IDs, questions, passages,
answers, supports, and model outcomes are not persisted in either graph or
allocation artifacts.

## Publication and proof chain

The four independently completed artifacts are ordered as follows:

1. `artifacts/hypergraph_v3/PUBLISHED.json` completes privacy-safe records,
   direct hyperedges, the graph report, and reusable solver contract.
2. `artifacts/validations/hypergraph_v3_deep/PUBLISHED.json` is written by a
   different PID/hash seed after rebuilding the clean pool and graph from all
   pinned raw inputs and matching all four graph payloads byte-for-byte.
3. `artifacts/allocation_v3/PUBLISHED.json` consumes only those two completed
   publications. Two separate one-thread SciPy/HiGHS workers use Python/HiGHS
   seeds 1 and 777; their selection JSONL must be byte-identical.
4. `artifacts/validations/allocation_v3_deep/PUBLISHED.json` independently
   rebuilds the graph again from pinned raw sources, starts two new solver
   PIDs, and requires their selection bytes and solver proof (apart from the
   explicitly replica-specific PID/seed fields) to match publication.

Every producer rejects an existing output directory. Payloads and sidecars
are regular non-symlink files, inventory is exact, and `PUBLISHED.json` is the
last atomic completion marker. An `IN_PROGRESS.json`, a receipt without its
marker, an extra file, a symlink, a duplicate JSON key, or a nonfinite JSON
number is not a publication.

The graph build is pinned to `PYTHONHASHSEED=1`; the raw-source deep rebuilds
are pinned to `PYTHONHASHSEED=777` and persist PID/seed evidence. All source,
package initializers, entrypoints, tests, README/config, Python/NumPy/SciPy/
NetworkX versions, and the compiled HiGHS wrapper binary are content-bound.
The execution tree must contain no `__pycache__`/`.pyc`; every command uses an
absent isolated `PYTHONPYCACHEPREFIX` plus `PYTHONDONTWRITEBYTECODE=1`.

## MILP claim boundary

Variables are binary target/record assignments. Constraints enforce at most
one target per record, at most one selected record per direct hyperedge, and
all 18 exact screen/confirm quotas. The 4-hop buffer keeps its original record
meaning: at most 117 of 119 strict-clean rows are selected, leaving at least
two rows; it does not require two unused connected components.

Phase 1 has a zero objective. Only SciPy integer status 2 together with
`success=false`, `x=null`, and the pinned HiGHS infeasible message may be
published as infeasible. Limit, unbounded, numerical, interrupted, unknown,
or malformed results fail closed without `PUBLISHED`. A feasible phase 1 and
hash-cost phase 2 both require status 0/Optimal, finite correctly-sized binary
incumbents satisfying the matrix. Phase-2 `fun` and the exact 52-bit integer
objective numerator are independently recomputed from the selected rows.

The phase-2 objective namespace remains
`qwen3-32to14-improved/pairwise-hypergraph-v2` intentionally. It is part of the
frozen mathematical cost formula, not an attempt path; retaining it makes v3's
MILP objective identical to v2 rather than silently changing record tie-breaks.

The JSON decoder necessarily materializes answer/support values transiently
before the frozen allowlist hook drops them. They never enter `Projection` and
are unavailable to graph, ranking, quota, partition, solver, allocation, and
persisted artifacts. Solver inputs are only privacy-safe graph rows, direct
hyperedges, exact quotas, and SHA-256-derived target/record costs. No model
outcome is opened.

## Frozen formal commands

No formal command may run before an independent read-only audit approves the
frozen source/config/tests. The authoritative full strings are also embedded
in `config.json`, the solver contract, and graph receipt. The fixed interpreter
is `/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python` (never bare `python`).

The independent tmux/log identities are frozen in `config.json`:

- graph build: `qwen3_pairwise_hypergraph_v3_graph_build` /
  `logs/pairwise_hypergraph_v3_graph.log`
- graph structural: `qwen3_pairwise_hypergraph_v3_graph_structural` /
  `logs/pairwise_hypergraph_v3_graph_structural.log`
- graph deep: `qwen3_pairwise_hypergraph_v3_graph_deep` /
  `logs/pairwise_hypergraph_v3_graph_deep.log`
- allocation: `qwen3_pairwise_hypergraph_v3_allocation` /
  `logs/pairwise_hypergraph_v3_allocation.log`
- allocation deep: `qwen3_pairwise_hypergraph_v3_allocation_deep` /
  `logs/pairwise_hypergraph_v3_allocation_deep.log`

```bash
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/scripts/build_graph.py
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/scripts/validate_graph.py
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache PYTHONHASHSEED=777 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/scripts/validate_graph.py --deep --write-receipt
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/scripts/solve_allocation.py
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/.formal_empty_pycache PYTHONHASHSEED=777 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v3/scripts/validate_allocation.py --deep
```
