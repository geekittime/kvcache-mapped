# Cross-tier audit result

## Decision and scope

**PASS** for the tail-pilot mapper-fit versus selected method-dev subgate. This
is not a claim that the complete G0 gate is closed: prospective locked-test
inputs and method-dev versus locked-test collision checks are out of scope and
are explicitly recorded as such in the receipt.

The audit was run without importing a GPU framework or invoking a model. The
locked inputs were decoded and compared on CPU only. Persisted outputs contain
identities, hashes, thresholds, aggregates, and numeric candidate metrics; they
contain no decoded/original text, question, reference, or answer fields.

## Locked population and result

| Measure | Result |
|---|---:|
| mapper-fit train sequences | 500 |
| mapper-fit heldout sequences | 50 |
| selected method-dev rows | 384 |
| rows with explicit `[Passage N]` blocks | 192 |
| audited passage bodies | 1,902 |
| pair comparisons | 1,046,100 |
| pairs with nonzero 5-shingle overlap | 698 |
| exact candidates (>=120 normalized characters) | 0 |
| flagged pairs | **0** |
| exact-substring flags | 0 |
| intersection + containment flags | 0 |
| Jaccard flags | 0 |

The maximum passage containment was `0.153846153846`; that candidate shared 2
unique 5-shingles and had Jaccard `0.002355712603`. The independent maximum
intersection was 5, and the maximum Jaccard was `0.005854800937`. Thus none is
close to the frozen thresholds of intersection >=20 with containment >=0.50,
or Jaccard >=0.30.

## Verification performed

- Seven synthetic/unit tests passed: normalization, passage/title boundary,
  exact positive, near-duplicate positive, negative, locked-input tamper
  rejection, and atomic receipt/sidecar binding.
- The real audit completed twice with byte-identical receipt SHA-256.
- The sidecar byte count and SHA-256 were recomputed from the receipt and
  matched exactly.
- The receipt contract was checked for `status=pass`, zero flags, privacy
  assertions, and absence of raw content/answer fields.

## SHA-256 inventory

Locked inputs:

| Input | SHA-256 |
|---|---|
| FineWeb train tokens | `6a045dc9b5c3456b9b1c72b20f6ea5e011bf81eaf0e2ba3c9d079cb93e743f29` |
| FineWeb heldout tokens | `43b7dee7b0b5acf352b53bc3613d176d04f2abe03e811aff955644eb2b026ddd` |
| Qwen3-14B `tokenizer.json` | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| selected method-dev v2 rows | `aa8b94367d57b088ba8c12a13068e9c82c6b95d18fc1579b5ccf66270c8aed53` |
| prepared method-dev v2 manifest | `dea34a19ca965f58d8d5f3ffb4c481b22d2420d71f30254287d8247e25062788` |

Audit products and executable contract:

| File | SHA-256 |
|---|---|
| `artifacts/cross_tier_audit_v1/receipt.json` | `37f69adcafa583d0898579f30701d09e569551895b1339b693d18ada420da84d` |
| `artifacts/cross_tier_audit_v1/receipt.json.sha256.json` | `aabdc82fe71127564db5cbccbbef09630a2a7c479025608de5546da1a9829ff2` |
| `src/cross_tier_audit.py` | `1e52c2bbf88a23d7d8db644230d23c5c5ff53ed8022c745d141c5ee996ebae8b` |
| `scripts/run_audit.py` | `34cfe9314b282918b7417e38f8bef44fef6f73e680a449c2120b8214f8ad6dc8` |
| `tests/test_cross_tier_audit.py` | `e3e82ea7333d60eabab16f03bc1f212a8fb8aff6e2da8e0c98ad710287f4d79a` |
| `config.json` | `65e6818877ec9e6db0fc6651a0126587cb9999f461a2e4fb0117838ab71d6b62` |

The receipt is 21,829 bytes; its atomic sidecar records that byte count and the
full receipt digest above.
