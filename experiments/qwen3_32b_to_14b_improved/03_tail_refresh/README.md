# 03 — native target-tail refresh

This isolated candidate keeps the immutable Qwen3-32B→14B production k20
mapper and recomputes only the final prompt tail with native Qwen3-14B. It
implements the preregistered R=1/8/32 systems classes without modifying any
baseline code, mapper payload, formal result, shared `src/`, or shared script.

Key files:

- `src/tail_refresh.py` — strict cache builder and causal split audit;
- `METHOD_DEV_PROTOCOL.md` — train-only data and decision protocol;
- `configs/method_dev.json` — pinned paths, hashes, modes, strata, prompts, and
  topology;
- `scripts/prepare.py` — CPU-only frozen selection/registry/preflight;
- `scripts/run_method_dev.py` — explicit-opt-in, resumable GPU pilot;
- `tests/` — tiny-Qwen3 CPU oracle, R=1 object identity, positions, isolation,
  selection, overlap, and metrics tests;
- `artifacts/prepared_method_dev_v2/` — current immutable preparation,
  byte-bound to the shared MuSiQue prompt/cap/stop protocol; the earlier v1
  CPU-only preparation was superseded before any GPU result was produced;
- `runs/` and `logs/` — candidate-local runtime products only.

The central invariant is:

```text
engine input C
  builder receives C[:-1]
    k20 maps first P-(R-1), P=len(C)-1
    14B appends exactly R-1 tail tokens
  engine replays exactly one boundary C[-1]
```

Consequently R=8 appends 7 tokens in the builder, and R=32 appends 31. R=1 is
an exact zero-op wrapper around the existing k20 builder.

These are method-development experiments on official HotpotQA train and
MuSiQue train only. Results cannot be described as locked-test superiority or
as a mapper-only improvement for R>1.
