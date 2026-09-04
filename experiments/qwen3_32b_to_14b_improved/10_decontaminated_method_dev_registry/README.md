# Decontaminated method-development registry v2

This independent, CPU-only directory creates the future `07`/`01` method-dev
selection registry. It does not modify the immutable `03`, `06`, `07`, or `09`
artifacts and does not launch models.

The selector uses only strict source projections:

- HotpotQA train: `_id`, `question`, `context`, `type`, `level`;
- MuSiQue train: `id`, `question`, `paragraphs` (hop is parsed from `id`);
- each dataset's locked dev: ID, question, and context only.

The standard JSON decoder necessarily reads the source file bytes and
transiently parses answer/supporting values into its temporary object-pairs.
The projection hook drops those keys before constructing any `Projection`, so
answer values are not available to decontamination, quota, or SHA-ranking
logic and are not persisted. Prompt length, model outputs, and metrics are not
used. Output registries contain only dataset/partition ID,
source row, projection SHA, SHA selection rank, and stratum. Quarantine output
contains aggregate counts only.

The normalization and exact/near thresholds are content-bound to pinned `09`.
Every train question and passage is compared against the **union of both**
HotpotQA and MuSiQue locked-dev projections, covering all four train-to-dev
dataset pairs, and against all 550 decoded FineWeb mapper-fit sequences. This
stronger rule is bound to the pinned `09` failure receipt that exposed
cross-dataset passage reuse. Repeated
normalized train-question groups are also quarantined. Insufficient clean pool
in any stratum fails closed.

Frozen quotas:

- HotpotQA: screen 128 and disjoint confirm 256, balanced as evenly as integer
  totals permit across `bridge/comparison × easy/medium/hard`;
- MuSiQue: screen 256 (`109/108/39`) and disjoint confirm 512
  (`217/217/78`) across `2hop/3hop/4hop`. The complete character-winnowing
  exact-substring attempt found 119 clean 4-hop records, invalidating the old
  incomplete-recall capacity of 146 and failing closed before publication.
  Capacity-only water-filling reserves exactly two clean 4-hop rows, allocates
  the other 117 in the fixed 1:2 screen/confirm ratio (`39/78`), then splits
  the remaining 651 slots as evenly as integers permit between 2-hop and
  3-hop (`326/325`, frozen strata-order tiebreak). Both prior failed quota
  attempts are retained append-only; no flagged row was restored and no
  similarity threshold was changed.

Run and verify:

```bash
cd /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/10_decontaminated_method_dev_registry
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/prepare.py
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python -m pytest -q
```

Publication is single-writer and fail-closed. The artifact directory must not
exist at startup. `receipt.json` is not sufficient by itself: consumers must
require `PUBLISHED.json`, which is atomically written only after every core
payload and SHA-256 sidecar has passed an internal self-check. A crash before
that final marker leaves an explicitly incomplete, non-reusable attempt.

New consumers must bind
`artifacts/registry_v2_strict_attempt3/registry_manifest.json` by path, byte
size, and SHA-256, and must independently bind the same properties for the
final `artifacts/registry_v2_strict_attempt3/PUBLISHED.json` completion marker.
The old `06`/`07` v1 selections remain preserved; this registry supersedes them
only for future runs.
