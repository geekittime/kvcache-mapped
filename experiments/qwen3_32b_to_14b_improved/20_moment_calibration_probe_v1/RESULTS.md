# Results — moment-calibrated shrinkage probe v1

Decision: **NO-GO**.

The run completed on 2026-09-02 using GPU 0 after the five-sample all-GPU
idle gate.  It interpreted only the 40 pre-existing tuning sequences.  The
ten locked activation rows were neither indexed nor evaluated.  Input
preflight deep-hashed 92 activation payloads (2,411,724,800 bytes), 92
metadata files, and the 12 selected shrinkage mapper/sidecar files.

Nested five-fold group CV selected `beta=0.375` for K and `beta=0.5` for V.
The unbiased outer-fold results over target layers 7, 8, 9, 10, 34, and 35
were:

| Component | Shrinkage OOF SSE | Calibrated OOF SSE | Relative reduction | Uniform-channel R² before | after |
|---|---:|---:|---:|---:|---:|
| K | 16,965,102.2913 | 16,961,089.9284 | 0.02365% | 0.7742415 | 0.7743052 |
| V | 87,432,133.9588 | 87,395,386.2131 | 0.04203% | 0.7230682 | 0.7232999 |

Both components improved numerically, but neither reached the pre-registered
0.1% relative-SSE threshold.  The candidate therefore stops here and will not
consume downstream HotpotQA/MuSiQue evaluation budget.  This negative result
also argues against spending more effort on purely static scale/offset
calibration; the next mapper candidate should change the functional objective
(fixed-query attention/readout and suffix-logit behavior), not merely output
moments.

Authoritative artifacts:

- `runs/probe_v1/input_preflight.json`
- `runs/probe_v1/results.json`
- `runs/probe_v1/COMPLETED.json`
- `runs/probe_v1/completion_receipt_v1.json` (append-only full input/code/result binding)

The result is a mapper-geometry ablation only and supports no downstream or
paper-superiority claim.
