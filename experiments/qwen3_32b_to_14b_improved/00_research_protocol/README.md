# Qwen3-32B→14B improved cache reuse: research protocol

This directory is the immutable starting point for experiments intended to
improve on the NVIDIA top-k ridge KV translator.  It contains protocol and
validation material only.  It does **not** contain mapper weights, captured
activations, benchmark outputs, or GPU launch scripts.

## Isolation contract

- The existing `experiments/qwen3_32b_to_14b/` tree is a read-only baseline.
- Shared `src/`, `scripts/`, `configs/`, `data/`, `artifacts/`, and `runs/`
  trees are not output locations for this research program.
- Every attempted method receives a sibling directory under
  `experiments/qwen3_32b_to_14b_improved/`; artifacts from different methods
  must not be pooled into this protocol directory.
- Baseline files are referenced by path and SHA-256 in
  `baseline_refs.json`; they are never copied, rewritten, or silently
  refreshed.
- This directory's validator is Python-standard-library-only and performs no
  model load, CUDA initialization, dataset download, or network request.

## Contents

- `EVIDENCE.md`: paper evidence, local evidence, transfer limits, and the
  resulting research hypotheses.
- `PROTOCOL.md`: data isolation, candidate/ablation order, metrics,
  statistics, and preregistered GO/NO-GO gates.
- `protocol.json`: machine-readable experiment contract.
- `data_policy.json`: three-tier data policy and prior-exposure disclosure.
- `baseline_refs.json`: content-addressed, read-only baseline references.
- `validate_protocol.py`: CPU-only/static protocol validator.
- `tests/test_protocol.py`: protocol and tamper-detection tests.

## CPU-only validation

From this directory:

```bash
python validate_protocol.py
python -m pytest -q tests
```

Validation reads the referenced baseline JSON files to check their hashes and
declared fields.  It does not read mapper tensor payloads and does not touch
the GPUs.

## Scientific status

The proposed SARR-KV method is an evidence-driven hypothesis, not an achieved
result.  A claim that it exceeds the NVIDIA baseline is allowed only after a
single nominated configuration passes the locked-test procedure in
`PROTOCOL.md`.  Pure mapping (`R=1`) and target-side tail refresh (`R>1`) are
different system classes and must be reported separately.
