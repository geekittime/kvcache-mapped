# Qwen3-32B → Qwen3-14B KV mapper reproduction

This directory is the isolated reverse-direction reproduction.  It never
writes into the existing `qwen3_14b_to_32b` selection, mapper, validation, or
run directories.  Existing activation stores are reused read-only.

## Paper-disclosed mapper protocol

- semantic source: Qwen3-32B, 64 layers;
- semantic target: Qwen3-14B, 40 layers;
- 500 × 1024 calibration tokens, position stride 4 (128,000 rows);
- same-head affine OLS probes, equal K/V score, ordered top-20 source layers;
- independent centered ridge K/V maps, lambda 0.01, unnormalised Gram;
- BF16 captured activations, FP32 covariance and mapper, FP64 diagnostics;

The additional 50 × 1024, stride-4 held-out capture is a local independent
reconstruction audit.  The paper does not disclose that held-out activation
split, so it is not presented as a paper protocol item.

The reused captures retain their original native roles.  Therefore the role
binding is intentionally crossed:

| Reverse semantic role | Reused capture | Capture-native role |
|---|---|---|
| source | `data/activations/qwen3_32b` | `target` |
| target | `data/activations/qwen3_14b` | `source` |

`scripts/prepare.py` verifies and records this binding before any GPU job.

## Commands

Run from the repository root:

```bash
.venv/bin/python experiments/qwen3_32b_to_14b/scripts/prepare.py
.venv/bin/python experiments/qwen3_32b_to_14b/scripts/select_layers.py --devices 0,1,2,3,4,5,6,7
.venv/bin/python experiments/qwen3_32b_to_14b/scripts/fit_mapper.py --devices 0,1,2,3,4,5,6,7
.venv/bin/python experiments/qwen3_32b_to_14b/scripts/validate_heldout.py --compute-device cuda:0
```

Use the repository `.venv`; it contains the pinned Torch, Transformers,
Accelerate, and safetensors stack.  Selection partitions the 40 target layers
over the requested GPUs.  Fitting also partitions by target layer and commits
K and V components independently, so an interrupted run resumes without
recomputing completed components.

The held-out reconstruction command consumes the independent local
50-sequence capture.  The paper-disclosed reverse evaluation scope is executed
through the sealed HellaSwag, HellaSwag-domain mechanism, CoQA, and latency entries documented in
`EVALUATION.md`.  The generic `scripts/evaluate.py` five-task/WikiText entry is
retained only as an optional local extension; the paper does not publish a
reverse five-task main table.

CPU-only checks:

```bash
.venv/bin/python -m pytest -q experiments/qwen3_32b_to_14b/tests
.venv/bin/python experiments/qwen3_32b_to_14b/scripts/validate_heldout.py --preflight-only
```

After the implementation has been frozen with `scripts/freeze_code.py`, the
sealed end-to-end queue is:

```bash
.venv/bin/python experiments/qwen3_32b_to_14b/scripts/run_pipeline.py
```

It accepts no mutable CLI overrides.  The queue waits for and cryptographically
audits the formal forward MultiNews500 run, verifies every code-freeze entry,
downloads only the pinned CoQA revision when its exact parquet is absent, and
then runs selection, fitting, held-out validation, HellaSwag, its mechanism
diagnostics, CoQA, and latency
under the repository GPU lock.  Target-layer-0 selection and fit are mandatory
smokes; the fit smoke must retain one GPU owner and stay below 18 GiB according
to the persisted `nvidia-smi` samples.  Every stage has an immutable hashed
receipt.  A rerun skips a stage only after revalidating its artifacts, and the
pipeline publishes `runs/formal_pipeline_v1/pipeline_completion_receipt.json`
only after all formal outputs have passed.

Selection and fitting are durable and resume per target layer.  The reverse
mapper contains 40 target-layer K/V maps.  Each map is `20480 → 1024`; the
complete FP32 mapper has 1,677,721,600 weights and 81,920 biases (about
6.2503 GiB plus metadata).  The paper directly discloses reverse `k=20` and
the general mapper-size formula.  These exact counts and 6.2503 GiB are local
derivations from that formula and the Qwen3 architectures, not a reverse
storage value printed directly in the paper.

Formal HellaSwag and CoQA use an explicit 5-GPU 32B / 3-GPU 14B layout with
layer counts `[12,13,14,13,12]` and `[13,14,13]`.  Formal latency uses a
sequential lifecycle suitable for eight 24-GiB GPUs: 6-GPU 32B prefill,
raw-bit-identical CPU cache staging, proven source-model release, then a
3-GPU 14B+mapper runtime with the source cache restaged on the other five GPUs
outside the timed region.  See `EVALUATION.md` for the exact topology and
receipt guarantees.

## Isolation layout

```text
config/          frozen reverse configs
reverse_kv/      reverse protocol, provenance, bundle, and runtime code
scripts/         executable reverse entry points
tests/           CPU-only unit tests
artifacts/       all reverse selection/statistics/weights/validation outputs
runs/            all reverse run evidence and evaluation results
logs/            launch logs
```

The entry points reuse only direction-agnostic numerical primitives from the
main package.  Direction-specific orchestration and runtime manifest handling
live in this directory.
