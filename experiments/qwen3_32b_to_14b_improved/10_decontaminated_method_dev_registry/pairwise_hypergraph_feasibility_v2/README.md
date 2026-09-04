# Pairwise direct-collision hypergraph feasibility v2

This CPU-only protocol replaces the over-conservative connected-component
closure from `component_feasibility_v1`. It does not weaken an exact,
normalized-substring, or near-shingle threshold and does not restore any row
quarantined against locked dev or mapper-fit data.

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

1. `artifacts/hypergraph_v2/PUBLISHED.json` completes privacy-safe records,
   direct hyperedges, the graph report, and reusable solver contract.
2. `artifacts/validations/hypergraph_v2_deep/PUBLISHED.json` is written by a
   different PID/hash seed after rebuilding the clean pool and graph from all
   pinned raw inputs and matching all four graph payloads byte-for-byte.
3. `artifacts/allocation_v2/PUBLISHED.json` consumes only those two completed
   publications. Two separate one-thread SciPy/HiGHS workers use Python/HiGHS
   seeds 1 and 777; their selection JSONL must be byte-identical.
4. `artifacts/validations/allocation_v2_deep/PUBLISHED.json` independently
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

```bash
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/scripts/build_graph.py
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/scripts/validate_graph.py
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache PYTHONHASHSEED=777 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/scripts/validate_graph.py --deep --write-receipt
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache PYTHONHASHSEED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/scripts/solve_allocation.py
test ! -e /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache && env PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/.formal_empty_pycache PYTHONHASHSEED=777 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/pairwise_hypergraph_feasibility_v2/scripts/validate_allocation.py --deep
```
