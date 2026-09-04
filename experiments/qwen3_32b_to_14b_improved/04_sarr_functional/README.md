# SARR-KV functional mapper

This directory is exclusively for the function-aligned low-rank residual
mapper. It may read frozen k4/k20 mapper products and frozen data registries by
absolute path plus SHA-256, but it may not write into those directories.

The method has not yet passed the preregistered gates. Research rationale,
equations, ablation order, and claim boundaries are in
[`RESEARCH_AND_METHOD.md`](RESEARCH_AND_METHOD.md).

Planned owned layout:

- `configs/`: frozen training and pilot configurations;
- `src/`: residual mapper and functional losses;
- `scripts/`: capture, train, validate, and benchmark entrypoints;
- `tests/`: CPU and fail-closed tests;
- `artifacts/`: trained weights and immutable receipts;
- `runs/`: method-dev evaluations only;
- `logs/`: append-only execution logs.

