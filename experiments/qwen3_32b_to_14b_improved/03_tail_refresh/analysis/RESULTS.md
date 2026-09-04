# R=1/8/32 tail-refresh method-dev pilot

## Integrity result

The GPU run completed naturally with 384/384 atomic results and no failure
artifact. The independent CPU validator passed all 384 files: exact inventory,
frozen sample and prompt identities, generated-token hashes and decoding,
stop/cap behavior, recomputed QA metrics, per-sample target binding, and the
R=1/8/32 boundary contract (builder-native tail 0/7/31 plus one engine-owned
boundary token). This analysis did not access a locked test split.

## Aggregate pilot results

Values are relaxed answer F1 / EM on 48 examples per dataset. Parentheses show
the paired F1 change from the local NVIDIA-compatible k20 R=1 baseline.

| Dataset | native 14B | mapped R=1 | tail R=8 | tail R=32 |
|---|---:|---:|---:|---:|
| HotpotQA train | 0.7645 / 0.6667 | 0.7314 / 0.6042 | 0.7241 / 0.6250 (-0.0073) | 0.7290 / 0.6250 (-0.0024) |
| MuSiQue train | 0.3487 / 0.2292 | 0.3649 / 0.2708 | 0.3878 / 0.2708 (+0.0228) | 0.4787 / 0.3958 (+0.1138) |

Paired sample-ID bootstrap, fixed seed 20260830 and 10,000 replicates:

| Dataset | Variant | F1 delta vs R=1 (95% CI) | EM delta vs R=1 (95% CI) |
|---|---|---:|---:|
| HotpotQA | R=8 | -0.0073 [-0.0496, 0.0326] | +0.0208 [0.0000, 0.0625] |
| HotpotQA | R=32 | -0.0024 [-0.0677, 0.0681] | +0.0208 [-0.0625, 0.1042] |
| MuSiQue | R=8 | +0.0228 [-0.0688, 0.1144] | 0.0000 [-0.1042, 0.1042] |
| MuSiQue | R=32 | +0.1138 [-0.0353, 0.2617] | +0.1250 [-0.0208, 0.2708] |

HotpotQA native-minus-R=1 F1 is positive, so gap closure is defined: R=8 is
-22.18% and R=32 is -7.41%. The bootstrap intervals are extremely unstable
because the native/R=1 pilot gap is small and changes sign in many resamples;
these values are diagnostics, not evidence of improvement. On MuSiQue, R=1
already exceeds native 14B by 1.62 F1 points, so fidelity gap closure is
undefined. R=32's +11.38-point task-score gain there must not be relabeled as
closing a native fidelity gap.

## Termination and cost diagnostics

| Dataset | Mode | EOS | Cap | Mean tokens | Mean wall seconds |
|---|---|---:|---:|---:|---:|
| HotpotQA | native 14B | 100.0% | 0.0% | 6.23 | 0.717 |
| HotpotQA | R=1 | 100.0% | 0.0% | 6.92 | 1.788 |
| HotpotQA | R=8 | 100.0% | 0.0% | 5.63 | 1.743 |
| HotpotQA | R=32 | 100.0% | 0.0% | 8.33 | 1.888 |
| MuSiQue | native 14B | 100.0% | 0.0% | 9.71 | 1.195 |
| MuSiQue | R=1 | 85.4% | 14.6% | 52.98 | 5.604 |
| MuSiQue | R=8 | 100.0% | 0.0% | 14.17 | 3.351 |
| MuSiQue | R=32 | 100.0% | 0.0% | 9.75 | 3.060 |

The MuSiQue improvement is mechanistically meaningful as a pilot signal:
refresh eliminated all 7/48 R=1 cap hits and reduced generation length. It is
not explained by accepting more truncated answers. These wall times are
end-to-end per-request measurements under this run, not isolated TTFT or
refresh-kernel latency.

## Stratified F1 / EM

| HotpotQA stratum (n=8) | native 14B | R=1 | R=8 | R=32 |
|---|---:|---:|---:|---:|
| bridge/easy | .7708/.6250 | .7708/.6250 | .7708/.6250 | .7708/.6250 |
| bridge/medium | .8221/.7500 | .8125/.7500 | .7554/.7500 | .8125/.7500 |
| bridge/hard | .8125/.7500 | .7986/.6250 | .7986/.6250 | .8125/.7500 |
| comparison/easy | .6331/.6250 | .6750/.6250 | .6875/.6250 | .6994/.6250 |
| comparison/medium | .8875/.6250 | .7458/.5000 | .7250/.6250 | .6714/.5000 |
| comparison/hard | .6607/.6250 | .5857/.5000 | .6071/.5000 | .6071/.5000 |

| MuSiQue hop stratum (n=16) | native 14B | R=1 | R=8 | R=32 |
|---|---:|---:|---:|---:|
| 2-hop | .3978/.2500 | .4695/.3125 | .4291/.1875 | .5513/.4375 |
| 3-hop | .1696/.1250 | .2315/.1250 | .3489/.2500 | .3140/.1875 |
| 4-hop | .4788/.3125 | .3938/.3750 | .3853/.3750 | .5708/.5625 |

## Pairwise behavior and decision

Against R=1 on HotpotQA, R=8 has 0 correct→wrong and 1 wrong→correct EM
transition; R=32 has 2 correct→wrong and 3 wrong→correct. On MuSiQue, R=8 has
3/3 and R=32 has 3/9 respectively. R=8 and R=32 emit exactly the same token
sequence on 36/48 HotpotQA and 12/48 MuSiQue examples; they disagree materially
on the harder benchmark.

Neither hybrid passes the preregistered strict pilot rule, because both have a
negative HotpotQA mean F1 delta even though termination is nonworse. Therefore
there is **no nomination and no locked-test claim** from this pilot. R=32 is the
stronger diagnostic lead: it nearly preserves HotpotQA R=1 F1 while sharply
improving MuSiQue and termination, but its 95% paired intervals include zero.
The honest next step is a frozen method-dev expansion, not a claim that the
paper baseline has already been beaten.
