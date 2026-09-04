# Complete G0 v1 result

Status: **FAIL (fail closed)**. `scope.closes_complete_g0=false`.

All pinned inputs and all three candidate inventories (`01`, `03`, and `07`)
were covered. FineWeb mapper-fit to the complete method-dev inventory produced
zero flags. Method-dev to locked-test produced 10,270 flagged passage-unit
pairs affecting 1,571 unique method-dev records and 2,932 unique locked
records. These are real exact/near context-passage reuse observations under the
thresholds frozen before the run, so the thresholds were not changed.

Flagged-pair inventory:

| Method-dev dataset | Locked dataset | Pair flags |
|---|---|---:|
| HotpotQA train | HotpotQA distractor dev | 2,470 |
| HotpotQA train | MuSiQue official answerable dev | 276 |
| MuSiQue train | HotpotQA distractor dev | 794 |
| MuSiQue train | MuSiQue official answerable dev | 6,730 |

Unique records affected by dataset: HotpotQA train 581, MuSiQue train 990,
HotpotQA distractor dev 1,495, and MuSiQue dev 1,437. Cross-tier exact ID,
normalized-question-hash, and normalized-question-plus-context-hash overlaps
were all zero; the failure is specifically passage similarity/containment.

Candidate pair flags were `01`: 2,440, `03`: 1,802, and `07`: 6,028. Shared
samples among `01`/`03`/`07` are explicitly classified as same-method-dev-tier
reuse and are not counted as cross-tier leakage.

The deterministic CPU rerun produced the same receipt bytes and SHA-256:

- receipt: `artifacts/complete_g0_v1/complete_g0_receipt.json`
- bytes: `7293564`
- SHA-256: `4b66efa69e269dae9bf5d710e2dd4adb8041b3c4651ee3f140c6a44031ae71f8`
- sidecar SHA-256: `4815be8535fe6f545587935f11b970e24409019d50db786acbd82d72125da7d9`

No raw source text, references, model outputs, or locked outcome metrics are in
the receipt. No locked outcome metric was evaluated.
