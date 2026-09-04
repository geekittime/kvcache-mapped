# Reverse-direction paper evaluation

This directory keeps every Qwen3 32B → 14B evaluation entry and output under
the isolated reverse experiment.  None of these entries writes to the existing
14B → 32B mapper or run directories.

## Scope disclosed by the paper

The paper-disclosed large-to-small evaluation scope includes HellaSwag
mechanism analysis, CoQA multi-turn handoff, and prefill latency.  It does
**not** publish a reverse five-task main table.  The reverse mapper is the
content-space ridge mapper with `k=20`.  The paper does not disclose the exact
HellaSwag harness commit or dataset revision; this reproduction freezes a
fully auditable local instance rather than claiming byte-identical hidden
inputs.

## Full HellaSwag

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python \
  experiments/qwen3_32b_to_14b/scripts/evaluate_hellaswag.py \
  --output-dir \
  experiments/qwen3_32b_to_14b/runs/hellaswag_full_k20
```

The entry accepts no mode, task, split, or limit override.  It always evaluates
only `target_standalone,mapped` on all 10,042 HellaSwag validation examples,
zero-shot, with canonical `acc_norm`.  All 10,042 raw rows are always retained;
the formal wrapper rejects incomplete/duplicate sample IDs and recomputes
`acc_norm` from those rows before committing a hashed paired receipt.

The simultaneous runtime uses explicit, contiguous sharding rather than the
loader's legacy-even default: the 32B semantic source uses GPUs 0–4 with
decoder-layer counts `[12,13,14,13,12]`, while the 14B semantic target and its
resident mapper use GPUs 5–7 with counts `[13,14,13]`.  The adapter validates
the semantic role and model path at every load, injects these exact counts,
and records the topology and its SHA256 in the formal receipt.

## HellaSwag-domain mechanism diagnostics

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python \
  experiments/qwen3_32b_to_14b/scripts/evaluate_mechanism.py \
  --output-dir \
  experiments/qwen3_32b_to_14b/runs/formal_pipeline_v1/mechanism_hellaswag_k20
```

This stage runs only after, and cryptographically binds, the complete formal
HellaSwag target/mapped receipt and the earlier 50-sequence local held-out
reconstruction audit.  It reports the section 4.5 diagnostics:
HellaSwag-domain RoPE-stripped content-key R² (primary), cached-RoPE key R²
(diagnostic), raw-value R², K/V error concentration, and cosine between
mapped-cache and ground-truth-cache target attention outputs.  The extra key
view makes the key-space choice auditable because the paper does not disclose
the exact implementation behind its eval-domain scalar.

The paper does not disclose the mechanism sample IDs, sequence construction,
or aggregation implementation.  The frozen local completion SHA-ranks 100
validation documents without reading their gold labels, evaluates all four
multiple-choice scoring sequences (400 sequences), and uses the exact
lm-eval forward inputs: context plus continuation except the final predicted
token.  R² is accumulated in FP64 per channel; attention/concentration math is
FP32.  Both cache conditions are read with the same target-standalone query
states; attention-output cosine is measured per query head before `o_proj`,
and the five query heads sharing each KV head are aggregated for concentration.
This is the paper-style offline mechanism diagnostic: source and target both
prefill the full scoring input and the mapper converts that full source cache;
it is not a second downstream score or a trace of target-generated suffix KV.
Immutable NPZ sufficient-statistic checkpoints make the run resumable.

The paper's `r=+0.57` attention-cosine result and `r=-0.20` calibration-R²
result are Pearson correlations over all 12 pair-direction evaluations, not
reference values for this one pair.  Table 4 does not publish Qwen3
32B→14B's individual R², concentration, or cosine.  Accordingly the result
records those references as unavailable and never claims a single-point
correlation or bit-exact paper match.

The bound FineWeb held-out R² is displayed only as a local domain-shift
reference.  The paper did not disclose this repository's 50-sequence held-out
split, so it is never relabelled as the paper's calibration-domain scalar.

## CoQA multi-turn handoff

The paper does not publish its 100 conversation IDs, prompt, history policy, or
decoding arguments.  The local deterministic completion is therefore marked
as non-bit-exact in every manifest:

- the pinned ModelScope validation parquet is selected at commit
  `5153eac35fe087e44d9ce99ebb2d94c3dbc81583`;
- twenty conversations with at least ten turns are selected from each of the
  five public domains by ascending SHA256 of namespace plus conversation ID;
- the official arXiv figure asset `figures/retention_vs_turns.png` labels its
  x-axis “Turn depth at handoff”; the five headline depths are 1, 3, 5, 7,
  and 10 (they are not ten repeated cascade transfers);
- previous answers are primary gold answers in both modes, eliminating
  cascading generated-answer differences from the paired comparison;
- the prompt is the historical EleutherAI CoQA completion prompt;
- for each conversation and mode, the model performs one initial turn-1
  parent prefill, then advances the same parent cache through turns 2–10 using
  only token deltas.  At each headline depth, mapped mode converts that
  accumulated 32B parent cache once and decodes on a disposable 14B branch;
  target standalone decodes from a disposable branch of its equivalently
  accumulated 14B own cache.  No headline point performs a full-history
  re-prefill;
- segment tokenization is accepted only when it exactly equals whole-prompt
  tokenization, so incremental execution cannot silently change the prompt;
- first-line response postprocessing follows the vendored lm-eval task, while
  the primary scorer implements Stanford's official `evaluate-v1.0.py`
  leave-one-reference-out normalized word F1 (script SHA256
  `d42f70d686abc724a9169947a487f4509087ee69804277cc73950682cb700b4d`).
  It preserves the primary plus three additional human annotations, including
  duplicate texts; the vendored lm-eval `doc_to_target` case-insensitive
  reference deduplication is explicitly not used.  A gold-independent
  answer-label-relaxed F1 is retained only as a diagnostic.

The data is intentionally not downloaded by the code-only preparation step:

```bash
.venv/bin/python \
  experiments/qwen3_32b_to_14b/scripts/evaluate_coqa.py \
  --prepare-only
```

The emitted pinned download command is:

```bash
/home/wby/.venvs/modelscope/bin/ms-hub download \
  EleutherAI/coqa coqa_validation.parquet \
  --repo-type dataset \
  --revision 5153eac35fe087e44d9ce99ebb2d94c3dbc81583 \
  --local-dir /home/wby/llm-cache/nvidia-kvcache/data/coqa \
  --max-workers 8
```

After download, `--prepare-only` verifies the complete payload SHA256
`5ab4d352067c30ef4925efb0dfb2abeeed4cdfb236304aa7ee9df62ea781281f`
and byte size `2,221,232`, then commits the deterministic 100-conversation
selection.  The full run command is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python \
  experiments/qwen3_32b_to_14b/scripts/evaluate_coqa.py \
  --output-dir experiments/qwen3_32b_to_14b/runs/coqa_100_k20
```

Every `(mode, conversation, handoff depth)` is an atomic hashed checkpoint.
The final run contains 1,000 hashed token-plan rows (all ten cumulative turns),
1,000 scored records (two modes × 100 conversations × five depths), raw
prompts, responses, generated token IDs, all human references, cache-length
and incremental-update evidence, per-example F1, per-depth/domain summaries,
drift in percentage points, a complete checkpoint manifest, and
code/model/mapper identities.

CoQA uses the same explicit 5-source/3-target topology as HellaSwag:
`[12,13,14,13,12]` source layers on GPUs 0–4 and `[13,14,13]` target layers on
GPUs 5–7.  Both the config loader and actual model loader fail closed on any
count or device drift, and the runtime setup persists post-load sharding
evidence from every decoder layer.

## Prefill latency

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python \
  experiments/qwen3_32b_to_14b/scripts/benchmark_latency.py \
  --output-dir experiments/qwen3_32b_to_14b/runs/latency_10length_k20
```

The entry is sealed to sequence lengths
`64,128,256,512,1024,2048,4096,8192,16384,32768`, 50 warmups, and 30 timed
samples.  It requires twelve consecutive idle-GPU samples before model load.
The runner acquires the same repository-wide `runs/gpu8_experiment.lock` used
by forward experiments before that idle gate and holds it through the final
manifest, preventing managed forward/reverse jobs from racing the gate.

The 24-GiB runtime is deliberately staged.  First, only the 32B model is loaded
on GPUs 0–5 with layer counts `[10,11,11,11,11,10]`.  It prefills every needed
length and copies each complete 64-layer BF16 cache to CPU.  The runner hashes
all raw K/V bits before and after that copy, then destroys the 32B model and
requires all eight visible CUDA allocators to report zero allocated and zero
reserved bytes before the 14B model is loaded.  The 14B model plus resident
6.2503-GiB FP32 mapper then use GPUs 0–2 with counts `[13,14,13]`.  Immediately
before each cell, its CPU cache is copied onto the disjoint GPUs 3–7 with
counts `[13,13,13,13,12]`, re-hashed, and verified GPU-resident.  CPU staging
and GPU restaging are outside both timers; the mapper timer begins from that
already-resident source KV and includes only source-cache handoff, inverse
source RoPE, FP32 affine mapping, target RoPE, BF16 conversion, and target
cache construction.  The target baseline times only the 14B transformer body
with KV output, excluding the LM head.

Before the ten-length sweep, the same lifecycle must pass a 32K endpoint
memory smoke with at least 2 GiB of observed driver headroom.  Each length
checkpoint binds the CPU raw-bit preparation receipt, source-model release
proof, five-GPU restage proof, timing-scope flags, and post-cell restaged-cache
release.  The final manifest and `formal_latency_receipt.json` revalidate and
bind all ten checkpoints and every staging session present in the sealed
session directory; an orphan payload or checksum sidecar is fatal.  Raw samples and
CUDA/nvidia-smi topology and memory evidence are persisted.  The paper's
H100/NVLink values are stored only as reference anchors; local RTX 4090
absolute latency is not claimed bit-exact to H100.

Before checkpoint reuse, every source/target safetensors shard is read and
content-hashed.  The latency request fingerprint additionally binds Python,
PyTorch/CUDA, Transformers, Accelerate, FlashAttention, safetensors, and the
NVIDIA driver version; changing any of them creates a different formal run.
