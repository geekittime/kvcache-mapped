# HotpotQA complete-output diagnostics

Both conditions use the same frozen prompts and prediction-only extraction. These descriptive subgroups do not alter the primary scores.

| Direction | Group | n | EM weighted/readout (%) | F1 weighted/readout (%) | Raw F1 weighted/readout (%) | Mean tokens weighted/readout |
|---|---|---:|---|---|---|---|
| 06_17 | all | 7405 | 8.832/13.558 | 20.579/25.167 | 14.420/11.165 | 40.98/57.60 |
| 06_17 | yes_no_reference | 458 | 51.528/50.000 | 51.528/50.000 | 0.655/0.218 | 27.51/47.91 |
| 06_17 | span_reference | 6947 | 6.017/11.156 | 18.538/23.530 | 15.327/11.887 | 41.87/58.24 |
| 06_17 | type_bridge | 5918 | 5.323/10.392 | 16.998/22.124 | 13.920/11.276 | 42.92/58.12 |
| 06_17 | type_comparison | 1487 | 22.798/26.160 | 34.830/37.276 | 16.409/10.722 | 33.25/55.52 |
| 17_06 | all | 7405 | 8.454/9.831 | 24.332/25.472 | 19.508/21.154 | 20.36/21.93 |
| 17_06 | yes_no_reference | 458 | 54.367/52.402 | 54.367/52.402 | 0.437/2.402 | 19.22/19.11 |
| 17_06 | span_reference | 6947 | 5.427/7.025 | 22.352/23.696 | 20.765/22.390 | 20.43/22.11 |
| 17_06 | type_bridge | 5918 | 5.762/7.773 | 21.524/23.271 | 19.810/21.835 | 21.26/22.55 |
| 17_06 | type_comparison | 1487 | 19.166/18.023 | 35.507/34.232 | 18.304/18.443 | 16.77/19.46 |

Extraction normalizes explicit answer labels, boxed answers and leading yes/no. Raw scores compare the entire generated response with the reference. Long explanations can lower raw F1 even when they contain a correct answer. Conversely, extraction can miss unlabelled correct spans or select an earlier explicit answer in a contradictory response. Thus neither score alone establishes a causal improvement in reasoning.

The frozen prompt requests a short answer. It is not changed after observing these outputs. No native-model control was run in this comparison; the absolute gap from native performance cannot be inferred.
