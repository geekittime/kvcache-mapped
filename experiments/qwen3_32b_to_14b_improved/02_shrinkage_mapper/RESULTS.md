# Head-wise shrinkage mapper result

## Outcome

The first pure-mapper candidate was completed without changing the original
32B -> 14B experiment.  It fuses an independently fitted k4 mapper and the
production k20 mapper with one scalar alpha per target layer, K/V component,
and target KV head (640 scalars).  Deployment remains one FP32 affine map per
target layer and component.

Alpha was fitted only on forty held-out sequences using five sequence-level
folds.  Ten token-SHA-selected sequences were locked before activation errors
were inspected and were evaluated once after all alphas had been frozen.

| Scope | Component | k4 R2 | k20 R2 | shrinkage R2 | shrinkage - k20 |
|---|---|---:|---:|---:|---:|
| 5-fold tuning OOF | K | 0.741761 | 0.744606 | 0.764819 | +2.021 pp |
| 5-fold tuning OOF | V | 0.658332 | 0.651765 | 0.683470 | +3.170 pp |
| locked 10-sequence audit | K | 0.742405 | 0.745000 | 0.765500 | +2.050 pp |
| locked 10-sequence audit | V | 0.657790 | 0.651585 | 0.683243 | +3.166 pp |

K alpha has mean/median 0.507904/0.520797 and range
0.302563--0.709793.  V alpha has mean/median 0.466974/0.466108 and range
0.196151--0.661688.  None of the 640 values reached either convex bound.

These are activation-reconstruction results, not downstream accuracy.  The
candidate must remain frozen while paired downstream evaluation compares it
with native 14B and unchanged k20.

## Artifact identities

- Mapper manifest: `artifacts/headwise_shrinkage_v1/manifest.json`
  - SHA256: `dabc63b252c5c7b9964fb00fdbe1b5eef440f126fb88c8d4e73a45378d5b4995`
  - 40 safetensors payloads, 6,711,239,568 payload bytes total.
- Frozen alpha: `runs/activation_cv_v3/alpha_manifest.json`
  - SHA256: `b918566884fc4dc2eb7738a1513bd48ffb9a89028781144da363a354c03a92f0`
- Locked audit: `runs/activation_cv_v3/locked_audit.json`
  - SHA256: `99da4cb55e1c52a1fb15a454443ab26f79914f8b71cc3689fa5d9f18f75b4369`
- CPU-only completion amendment:
  `runs/activation_cv_v3_completion_amendment_v1/receipt_amended.json`
  - SHA256: `49450ace00fee697fd2ebd07c6b2def9efe9241721e7cebe82bea8518699dd14`
- Independent mapper algebra audit:
  `runs/activation_cv_v3_mapper_algebra_audit_v2/receipt.json`
  - SHA256: `25e9ff1932bc49f67c4e4619a4f5d934e54df5c6f3478191fc81b38b98f90497`
  - Its validation checked all 40 layers and 160 tensors with `torch.equal`,
    explicitly binding every upstream k20/k4 layer SHA.  It did not parse or
    recompute locked-audit outcomes.
- Runtime: `src/runtime.py`
  - SHA256: `da9c48f00cc81578e9c56d7908ccd0ddb3faf554b9c087dea6160dc15f24de7e`

## Execution record

Attempts v1 and v2 passed independent 12 x 5-second GPU idle gates but failed
before any activation matrix multiplication because the pinned PyTorch build
could not reset allocator statistics before CUDA context creation.  Both
failures are retained and content-bound in the v3 definition and amended
receipt.  V3 created a one-element CUDA context probe after its own fresh idle
gate, then completed all activation work on physical GPU 0 only.  No language
model was loaded.

The scientific artifacts and mapper were atomically published before the v3
process failed to JSON-serialize PyTorch's `_CUuuid` in its final receipt.
They were not recomputed.  The append-only amendment converts only that UUID
to the already frozen nvidia-smi string and deeply validates the pre-existing
alpha, audit, and mapper.  Exact process-local allocator peak memory could not
be recovered after the original process exited, so no retrospective value is
fabricated.

