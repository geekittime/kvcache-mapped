# V weighted ridge vs V readout ridge

Audited full cells: 28/52. Complete: False.

| Direction | Task/mode | n | Metric | Weighted (%) | Readout (%) | Δ (pp) | Paired 95% CI (pp) |
|---|---|---:|---|---:|---:|---:|---|
| 06_17 | arc_off | 1172 | accuracy | 57.082 | 58.447 | +1.365 | [-0.512, +3.328] |
| 06_17 | mmlu_off | 1400 | macro_accuracy | 28.429 | 30.571 | +2.143 | [+0.071, +4.214] |
| 06_17 | ruler_4k_off | 400 | recall | 97.000 | 97.750 | +0.750 | [+0.000, +1.750] |
| 06_17 | ruler_8k_off | 400 | recall | 96.750 | 98.250 | +1.500 | [+0.250, +3.000] |
| 06_17 | ruler_16k_off | 400 | recall | 94.500 | 97.750 | +3.250 | [+1.250, +5.500] |
| 06_17 | ruler_32k_off | 400 | recall | 92.250 | 97.000 | +4.750 | [+2.500, +7.250] |
| 06_17 | arc_on | — | pending | — | — | — | — |
| 06_17 | mmlu_on | — | pending | — | — | — | — |
| 06_17 | ruler_4k_on | — | pending | — | — | — | — |
| 06_17 | ruler_8k_on | — | pending | — | — | — | — |
| 06_17 | ruler_16k_on | — | pending | — | — | — | — |
| 06_17 | ruler_32k_on | — | pending | — | — | — | — |
| 06_17 | hotpot_off | 7405 | em | 8.832 | 13.558 | +4.727 | [+3.957, +5.456] |
| 06_17 | hotpot_off | 7405 | f1 | 20.579 | 25.167 | +4.588 | [+3.870, +5.273] |
| 06_17 | hotpot_off | 7405 | raw_em | 1.675 | 0.797 | -0.878 | [-1.148, -0.621] |
| 06_17 | hotpot_off | 7405 | raw_f1 | 14.420 | 11.165 | -3.255 | [-3.627, -2.904] |
| 17_06 | arc_off | 1172 | accuracy | 66.041 | 72.099 | +6.058 | [+3.584, +8.618] |
| 17_06 | mmlu_off | 1400 | macro_accuracy | 23.714 | 27.929 | +4.214 | [+2.143, +6.429] |
| 17_06 | ruler_4k_off | 400 | recall | 99.500 | 100.000 | +0.500 | [+0.000, +1.250] |
| 17_06 | ruler_8k_off | 400 | recall | 100.000 | 100.000 | +0.000 | [+0.000, +0.000] |
| 17_06 | ruler_16k_off | 400 | recall | 99.500 | 99.750 | +0.250 | [+0.000, +0.750] |
| 17_06 | ruler_32k_off | 400 | recall | 99.500 | 99.500 | +0.000 | [-1.000, +1.000] |
| 17_06 | arc_on | — | pending | — | — | — | — |
| 17_06 | mmlu_on | — | pending | — | — | — | — |
| 17_06 | ruler_4k_on | — | pending | — | — | — | — |
| 17_06 | ruler_8k_on | — | pending | — | — | — | — |
| 17_06 | ruler_16k_on | — | pending | — | — | — | — |
| 17_06 | ruler_32k_on | — | pending | — | — | — | — |
| 17_06 | hotpot_off | 7405 | em | 8.454 | 9.831 | +1.377 | [+0.905, +1.891] |
| 17_06 | hotpot_off | 7405 | f1 | 24.332 | 25.472 | +1.140 | [+0.659, +1.645] |
| 17_06 | hotpot_off | 7405 | raw_em | 1.904 | 4.159 | +2.255 | [+1.904, +2.633] |
| 17_06 | hotpot_off | 7405 | raw_f1 | 19.508 | 21.154 | +1.646 | [+1.246, +2.083] |

Generation diagnostics for complete paired cells (weighted/readout):

| Direction | Task/mode | n | Length stops | Unparsed | Mean generated tokens |
|---|---|---:|---|---|---|
| 06_17 | arc_off | 1172 | 4/5 | 1/2 | 82.80/95.79 |
| 06_17 | mmlu_off | 1400 | 126/150 | 146/143 | 297.98/325.48 |
| 06_17 | ruler_4k_off | 400 | 0/0 | 0/0 | 63.55/51.90 |
| 06_17 | ruler_8k_off | 400 | 1/1 | 0/0 | 57.11/69.06 |
| 06_17 | ruler_16k_off | 400 | 1/0 | 0/0 | 72.65/90.82 |
| 06_17 | ruler_32k_off | 400 | 0/1 | 0/0 | 51.97/88.08 |
| 06_17 | hotpot_off | 7405 | 106/48 | 0/0 | 40.98/57.60 |
| 17_06 | arc_off | 1172 | 0/0 | 6/2 | 49.88/31.93 |
| 17_06 | mmlu_off | 1400 | 9/14 | 32/29 | 101.22/138.03 |
| 17_06 | ruler_4k_off | 400 | 0/0 | 0/0 | 22.93/24.41 |
| 17_06 | ruler_8k_off | 400 | 0/0 | 0/0 | 23.43/24.30 |
| 17_06 | ruler_16k_off | 400 | 0/0 | 0/0 | 24.23/24.48 |
| 17_06 | ruler_32k_off | 400 | 0/0 | 0/0 | 23.43/23.75 |
| 17_06 | hotpot_off | 7405 | 0/1 | 0/0 | 20.36/21.93 |

06_17 = Qwen3-0.6B → 1.7B; 17_06 = reverse. on/off indicates reasoning mode.
MMLU-Pro uses macro accuracy; ARC-C accuracy; RULER NIAH recall at each length; HotpotQA answer EM/F1 and raw scores.
Intervals use 2,000 paired bootstrap draws (MMLU stratified by subject). They do not measure training-seed variability. No tuning on these outcomes.
MC scores use parser v2 for LaTeX option wrappers and Markdown option letters. Original parser scores and all changed records remain in per-cell mc_scoring_v2.json.
Length-capped and unparsed predictions remain in every score denominator. See HOTPOT_DIAGNOSTICS.md for descriptive output-format and question-type effects; extracted-answer and raw Hotpot scores can move in opposite directions.
Full protocol and unavailable author details: README.md. Process and stage evidence: PROCESS.md and checks/.
