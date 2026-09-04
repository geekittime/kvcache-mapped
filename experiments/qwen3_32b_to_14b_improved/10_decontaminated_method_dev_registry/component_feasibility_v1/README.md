# Connected-component partition feasibility v1

This is an independent, CPU-only feasibility audit. It is not a sample
registry and cannot be consumed by model evaluation.

Attempt 3 proved that choosing a hash-dispersed screen set first left only 40
eligible MuSiQue 3-hop confirm records after the frozen exact/near sequence
gate. This audit therefore constructs the full symmetric collision graph over
the already decontaminated train pool using the identical formal thresholds.
Every connected component may contribute at most one selected record and may
serve only one partition. A deterministic max-flow then tests the original 18
dataset/stratum/partition quotas.

If the original quotas are infeasible, a preregistered descending common-scale
search reports the largest feasible balanced family with global 1:2
screen:confirm ratios, frozen max-min stratum water-filling, and two unused
MuSiQue 4-hop-supporting components. The buffer is a max-flow capacity gate:
if such a component is assigned to any dataset, stratum, or partition, it
counts as used, so at least two are completely unused. This is diagnostic
output only; it does not silently change any formal registry quota.

The standard JSON decoder transiently parses raw answer/supporting values, but
the projection hook drops them before `Projection` objects are constructed.
Answers and model outcomes are unavailable to decontamination, components,
max-flow, ranking, and all persisted artifacts. The component inventory writes
only component hashes, cell labels, sizes, masks, and representative projection
hashes—never sample IDs, questions, contexts, answers, or outcomes.
The report includes both whole-component sizes and, for every frozen stratum,
the number of supporting components plus within-stratum component-size
percentiles and maxima.

Run:

```bash
cd /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry/component_feasibility_v1
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/run.py
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/validate.py
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/validate.py --deep
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python -m pytest -q
```

Only an artifact with a final atomic `PUBLISHED.json` is complete. Freezing the
result additionally requires the post-publication structural validator and the
full pinned-source deep recomputation to pass naturally.
