# Complete G0 isolation audit

CPU-only, fail-closed audit covering every currently authorized candidate input:

- mapper-fit: pinned FineWeb-Edu train 500 and heldout 50 token arrays, decoded by the pinned Qwen3-14B tokenizer JSON;
- method-dev: `01` prepare-v1 (HotpotQA 256 + MuSiQue 256), `03` prepared-v2 (HotpotQA 192 + MuSiQue 192), and frozen `07` prepared-v1 screen/confirm samples (HotpotQA 384 + MuSiQue 768);
- locked-test: HotpotQA distractor dev 7,405 and official answerable MuSiQue dev 2,417.

The locked parser is a strict projection: it retains only identity, question,
and context structure. It never accesses ground truth, supporting annotations,
model results, or outcome metrics. Receipts contain hashes and numeric
similarities only—never raw questions, contexts, references, or outputs.

Thresholds were frozen in `config.json` before inspecting results. The passage
rule uses normalized word-5 shingles and flags either normalized whole-unit
substring containment of at least 120 characters, or at least 20 shared
shingles plus containment >= 0.50 or Jaccard >= 0.30. The preregistered
short-question rule uses word-3 shingles and at least four shared shingles plus
containment >= 0.80 or Jaccard >= 0.70.

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/run.py
```

`scope.closes_complete_g0` is true only when all pinned inputs match, all
candidate counts match, and every mapper-fit/method-dev/locked-test cross-tier
check has zero flags. Repetitions shared by `01` and `03` are inventoried as
same-tier reuse, not mislabeled as cross-tier leakage. The `07` question and
context projection is reconstructed only from its pinned frozen prompt token
IDs using the pinned tokenizer; its reference list is never accessed.

## Append-only v2 staging

The old v1 failure evidence above is immutable.  A separate v2 implementation
is staged in `src/g0_audit_v2.py`, `scripts/run_v2.py`, and
`tests/test_g0_audit_v2.py`.  It independently rehydrates every selected v2
registry identity from the pinned train sources and verifies the source row,
projection SHA, selection-rank SHA, byte digest, and a formatting-independent
order-sensitive digest before auditing content.

`config_v2.template.json` is deliberately non-runnable.  Its unpublished
registry paths, byte counts, SHA-256 values, and ordered row digests are explicit
placeholders; the v2 runner rejects it before reading registry or locked data.
Only after the registry has an atomic `PUBLISHED.json`, its writer declares
`STOP EDITING`, and a separate read-only audit passes may a new frozen
`config_v2.json` and formal `artifacts/complete_g0_v2/` be created.

The v2 gates cover both mapper-fit-to-method-dev dataset cells, all four
method-dev-to-locked dataset cells, and all four screen-to-confirm sequence
isolation cells.  Questions and passages are audited separately with the v1
thresholds.  Exact normalized substring retrieval uses deterministic character
winnowing with literal shorter-in-longer verification, including long-token and
destroyed-word-boundary cases.  Any missing binding, missing matrix cell, or
nonzero flag fails closed.

On success, the v2 receipt exposes the exact native experiment-10 publication
binding consumed by experiment 12: the `PUBLISHED.json` marker, registry
manifest, publication receipt, and four JSONL split artifacts.  It also exposes
the four explicit mapper-fit/locked-eval comparison flags required by that
consumer.  No shadow registry or translated row format is created.
Both machine-readable experiment-12 contracts and both experiment-13
G0/native-registry contracts are themselves pinned by path, byte count, and
SHA-256 in the v2 config and success receipt.  These evidence bindings do not
alter the shared nested native candidate-registry schema or its four
dataset-specific zero-flag comparisons.
