# KV-Lingo Stage 1: V weighted ridge versus V readout ridge

User-requested comparison, 2026-10-19. No Stage 2, no MapFlow, no TTFT. Branch `experiment/v-readout-ridge-qwen3-06b-17b`. All files and caches stay under /home/tiger; no push.

Two directions: released **Instruct** Qwen3-0.6B → Qwen3-1.7B and reverse. Two conditions: `weighted` and `readout`. Both use the same K attention-weighted mapper, exactly equal saved K matrices, and λ=0.01 on unnormalised corpus normal-equation sums, as NVIDIA paper Eq.4. Only the V regression design differs. λ is not tuned on benchmark results.

Common architecture: KV-Lingo split K/V, head mixing (all 1024 source V features per 128-dimensional target KV head), 28 identity-matched layers, pre-k-norm/pre-RoPE keys, no bias, ν=1. Frozen BF16 models, FP64 regression products/statistics/solve, FP32 saved maps; target key norm and RoPE applied exactly once. No intercept/centering is added: NVIDIA's affine architecture is not imported, because the user requests only its ridge regularizer while retaining KV-Lingo's linear architecture. Its λ=0.01 Tikhonov term replaces the unregularized eigensolver's cutoff; solve with Cholesky and verify the normal residual.

Attention convention is **prefix-only softmax**, the literal Eq.3.4 interpretation. All continuation queries and all 16 target query heads are used; no query subsampling. The earlier local reproduction used a full causal denominator; those old mappers/results are not controls for this new experiment. Target q/k are captured after native norm and RoPE. The full teacher sequence provides the continuation's queries, but the fitting softmax only includes prefix keys. This is oracle teacher attention during fitting, not attention produced by mapped K during generation.

For a layer, let X be source V, and Y_h the target KV-head h values. Hkv=8, G=2 grouped query heads, m continuation queries. Baseline token weights are w_i = mean over all 16m target queries of A[q,i], matching the paper's averaging over heads and queries. Baseline fits sum_examples sum_i w_i ||X_i W - Y_i||² + .01 ||W||². Readout fits sum_examples sum_h mean over that KV head's Gm queries of ||A_q X W_h - A_q Y_h||² + .01 ||W||². Each native query-head A is aggregated independently, **never averaging A across heads before forming its Gram**. The 1/(Gm) factor matches per-example/per-KV-head scale; it avoids changing ridge strength just because m or G changes. We never form AᵀA. If this protocol is later changed to a subset of causal attention, its mass must not be silently renormalized; an affine model would require the A1 coefficient on bias. The current model is linear and uses normalized prefix attention.

Training: reused verified `data/kv_lingo/training_paper_v7`, pinned Nemotron-SFT-Instruction-Following-Chat-v2 revision 1a9454ed054b8544503ab8d8c0a519d141a44c5b. 400 train /64 held-out conversations, no member UUID overlap; 50% natural (10–10k, uniform length bins), 50% packed (12–16k); reasoning alternates; continuation cap 2048. These exact same rows are shared by both conditions and directions. Raw source hashes, split construction and provenance are retained. Local prefix total is 3,809,293 tokens; paper says approximately 2.8M while describing this mixture. Author IDs and packing code are not public; local exact sample recovery is impossible. Do not claim identical hidden author corpus.

Evaluation: paired inputs and deterministic greedy decoding, prefix translated except last boilerplate token, held-back token forwarded natively. ARC-C full 1172 and MMLU-PRO 1400 (100 per subject, macro primary); both reasoning-off/on budgets 1024/8192 from Table4. RULER NIAH `niah_single_2`, 400 each at 4k/8k/16k/32k, off/on with 2048 budget, recall; this is a fixed local NIAH subtype, not all RULER tasks. HotpotQA full 7405 distractor validation, reasoning-off, 256-token budget, answer EM/F1 and raw metrics. HotpotQA is a user extension, not a KV-Lingo paper benchmark. Existing fixed MC/QA prompts and prediction-only parsers are local implementations because the full author harness is unavailable. All outputs, truncation/unparsed counts and paired deltas are retained; no examples skipped or test-driven hyperparameter selection. A held-out readout/reconstruction diagnostic checks whether improving the chosen loss generalizes before attributing benchmark differences to it.

Stage gates: data/provenance → algebra and real-model capture/injection checks → fit smoke/normal residual/K identity → complete 400-example fit and artifact audit → fresh generation smoke → complete benchmark outputs and independent rescoring. Failed stages stop dependent work. Reports must distinguish running/partial from accepted complete results. Process history: PROCESS.md.

Read-only progress:

```bash
cd /home/tiger/kvcache-mapped
source experiments/v_readout_ridge/env.sh
.venv/bin/python experiments/v_readout_ridge/status.py
```

The already-launched durable coordinator is recorded in `checks/pipeline_process.json`. Do not launch it a second time against the same output paths. It uses the four exact fit processes in `checks/fit_processes.json`, then stage-gated validation and evaluation. `report.py --require-complete` independently audits all full cells and refuses to declare success until all52 pass.
