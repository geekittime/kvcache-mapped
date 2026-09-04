# 20 — moment-calibrated shrinkage probe

This is an isolated Qwen3-32B → Qwen3-14B mapper experiment.  It asks a
deliberately cheap question before spending GPU time on a learned residual:
can the frozen head-wise shrinkage mapper be improved by an identity-centred
post-calibration consisting of one scale per target KV head and one bias per
output channel?

For a frozen shrinkage prediction `p`, the candidate is

```
q[h,c] = (1 + beta * (scale[h] - 1)) * p[h,c] + beta * bias[h,c]
```

`scale` and `bias` are fitted on only the 40 tuning sequences already used by
the shrinkage experiment.  `beta` is selected separately for K and V by a
five-fold nested group CV shared over the pre-registered target layers
`7,8,9,10,34,35`.  The ten locked activation sequences are never indexed,
evaluated, or used for selection.  Calibration parameters can be fused into
the affine mapper, so a successful candidate adds no receiver forward and no
extra matrix multiplication.

This probe is a geometry ablation, not a downstream-accuracy claim.  It is a
NO-GO unless nested OOF SSE improves for both K and V and at least one improves
by 0.1% relative.  Even a GO must later beat the NVIDIA-aligned k20 baseline
on fresh HotpotQA/MuSiQue samples.

All outputs stay inside this directory.  The old shrinkage and NVIDIA-aligned
directories are read-only inputs.

