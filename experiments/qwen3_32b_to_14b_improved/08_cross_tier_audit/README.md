# Mapper-fit / method-dev cross-tier audit

This directory provides the locked, CPU-only contamination audit between the
550 FineWeb mapper-fit calibration sequences and the explicit `[Passage N]`
blocks in `03_tail_refresh/artifacts/prepared_method_dev_v2/selected_samples.jsonl`.
Rows without that explicit marker are deliberately ineligible; the locked
population contract is 384 rows, 192 eligible rows, and 1,902 passages. The
`Title:` line is passage identity metadata and is excluded from the audited
passage body, matching the prompt's passage boundary.

## Gate boundary

This receipt closes only the tail-pilot subgate for mapper-fit FineWeb
train/heldout versus the selected method-dev v2 explicit passages. It does not
compare method-dev with prospective locked-test inputs, perform locked-test
identity/hash collision checks, or claim to close the complete research
protocol G0 gate. Those require separately pinned inputs and receipts.

The audit decodes token sequences with the pinned Qwen3-14B `tokenizer.json`,
normalizes both tiers with NFKC, case folding, and alphanumeric/whitespace
normalization, and hashes set-valued word 5-shingles with BLAKE2b-128. A pair is
flagged when it has an exact normalized substring of at least 120 characters,
or at least 20 shared shingles and passage containment of at least 0.50, or
Jaccard similarity of at least 0.30.

All five inputs are checked against frozen SHA-256 values before parsing. Any
missing file, mismatch, malformed row, unexpected array layout/count, decode
failure, or changed algorithm contract aborts without replacing the prior
receipt. The receipt is written by fsync plus atomic rename, followed by an
equally atomic SHA-256 sidecar. Persisted output contains identities and numeric
metrics only—never decoded text, method-dev source text, questions, or answers.

Run from the repository root with the existing CPU environment:

```bash
.venv/bin/python experiments/qwen3_32b_to_14b_improved/08_cross_tier_audit/scripts/run_audit.py \
  --config experiments/qwen3_32b_to_14b_improved/08_cross_tier_audit/config.json

.venv/bin/python -m pytest -q \
  experiments/qwen3_32b_to_14b_improved/08_cross_tier_audit/tests
```

The command exits 0 for a clean audit, 2 when any pair is flagged, and 1 for a
contract/integrity failure. See `RESULTS.md` for the committed run summary.
