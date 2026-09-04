# Qwen3 32B → 14B five-task accuracy extension

This directory adds a sealed evaluation extension to the completed reverse
mapper reproduction. It does not modify `reverse_kv/`, `pipeline.py`,
`report.py`, `code_freeze.json`, mapper weights, or completed pipeline output.

`five_task_extension/code_freeze.json` is an independent current overlay. The
older paper-pipeline freeze is retained as an immutable base reference, but is
explicitly not claimed to describe the current tree: the overlay contains a
machine-verifiable amendment/drift ledger and freshly content-hashes extension
files, all current reverse modules/scripts, shared evaluator/scorer code, the
vendored harness core plus all five task directories, local task YAMLs and the
benchmark manifest. It also records every 32B/14B weight-shard content SHA and
rehashes every k20 mapper payload/sidecar/input byte. Both the smoke and full
runners deep-verify model and mapper bytes before taking the GPU lock.

## Scientific boundary

`paper_bit_exact=false`. The paper does not report a complete reverse
Qwen3-32B→Qwen3-14B ARC-Challenge/HellaSwag/WinoGrande/MMLU/GSM8K table or the
corresponding WikiText-2 prefix PPL table. The numbers produced here are local,
protocol-aligned extension measurements; they must not be described as paper
table values.

The comparison is:

- `target_standalone`: Qwen3-14B itself;
- `mapped`: Qwen3-32B prefill, the committed k=20, λ=0.01 projection, then
  Qwen3-14B continuation/scoring;
- completion requests only, no chat template, batch size 1, boundary replay;
- ARC-Challenge 0-shot `acc_norm` (1,172 test rows);
- HellaSwag 0-shot `acc_norm` (10,042 validation rows);
- WinoGrande 0-shot `acc` (1,267 validation rows);
- MMLU 5-shot `acc` (14,042 test rows across 57 subjects);
- GSM8K CoT 8-shot greedy generation, 1,319 test rows, generation cap 1,024;
- WikiText-2 non-overlapping 2,048-token chunks, 1,024-token mapped prefix and
  1,024 scored tokens, with full prefix PPL as an auxiliary metric.

GSM8K always preserves and independently recomputes canonical lm-eval
`strict-match`. The reporting row additionally uses the already frozen
`numeric-final-result-level-diagnostic-v1`: prediction extraction never sees
the gold answer, selects the last explicit answer marker or final fallback
numeric token, and compares finite decimals by exact rational equality. It is
a reasonable local result-level diagnostic, not the paper's undisclosed
answer filter.

## Reuse decision

The existing reverse HellaSwag result is strictly bridged, not rerun. It
already contains both requested modes under the same reverse config, models,
full-content-hashed weights, mapper, topology, task YAML, dataset SHA, and
full raw 10,042-row protocol. Finalization revalidates both raw files and its
formal receipt before including it.

Old forward-direction Qwen3-14B source results are not used for the formal
target column. Although their task/scoring protocol is scientifically close,
their provenance did not hash every weight-shard byte; their semantic role and
two-GPU topology also differ from the reverse target's three-GPU runtime. The
fresh reverse run is therefore required for ARC, WinoGrande, MMLU, GSM8K and
WikiText-2.

## CPU-only preparation and tests

From the repository root:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python \
  experiments/qwen3_32b_to_14b/five_task_extension/prepare.py \
  --profile full

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q -p no:cacheprovider \
  experiments/qwen3_32b_to_14b/five_task_extension/tests
```

Preparation rehashes current benchmark payloads and all model/mapper payloads, revalidates the HellaSwag
bridge, verifies the evaluation/runtime scientific subset is byte-identical to
the completed Hella segment, records the completed reverse freeze identity,
and prints the exact request fingerprint expected from the shared evaluator.
It does not import CUDA or load a model.

The identities printed by `prepare.py` include:

- exact full/smoke request fingerprint (the extension overlay SHA is inside
  this fingerprint, avoiding a documentation/freeze hash cycle);
- immutable base paper-pipeline code-freeze SHA256 (the extension overlay is
  printed separately by `prepare.py`):
  `72a3d03e63ea18fb8f5005383d161753873aa647acb6dd09f3b2d94cf39ffe96`;
- k20 mapper manifest SHA256:
  `382a1deb6f4fd33c824fe782d956f46bdfb2b876081cb36487616e4de92a8ee2`;
- mapper fit fingerprint:
  `0aa81cb5816b56fe19c0349db7c54496751be6aa1918b8aff895f1e824ac2443`.

`prepare.py` is authoritative: it fails if scientific code/data identities
move and will expose any non-scientific provenance change through a different
expected request fingerprint before a GPU is loaded.

## GPU smoke

Smoke has a separate immutable fingerprint and output. It evaluates eight rows
per lm-eval leaf task (eight per each MMLU subject) and two WikiText chunks. It
is not a formal result.

```bash
tmux new-session -d -s qwen3_reverse_five_task_smoke_v1 \
  "set -Eeuo pipefail; cd /home/wby/llm-cache/nvidia-kvcache && \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
   /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python \
   experiments/qwen3_32b_to_14b/five_task_extension/run.py \
   --profile smoke \
   --output-dir experiments/qwen3_32b_to_14b/runs/five_task_extension_smoke_k20_v1 \
   2>&1 | tee -a experiments/qwen3_32b_to_14b/logs/five_task_extension_smoke_k20_v1.log"
```

## Full GPU run

Start only after the smoke receipt validates:

```bash
tmux new-session -d -s qwen3_reverse_five_task_full_v1 \
  "set -Eeuo pipefail; cd /home/wby/llm-cache/nvidia-kvcache && \
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
   /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python \
   experiments/qwen3_32b_to_14b/five_task_extension/run.py \
   --profile full \
   --output-dir experiments/qwen3_32b_to_14b/runs/five_task_extension_full_k20_v1 \
   2>&1 | tee -a experiments/qwen3_32b_to_14b/logs/five_task_extension_full_k20_v1.log"
```

The runner holds the repository-wide eight-GPU lock, proves all eight GPUs
idle for 12 consecutive five-second samples, then loads the fixed 5-GPU 32B +
3-GPU 14B topology. Every `(task, mode)` is an atomic checkpoint. lm-eval
request databases autocommit each completed request, so an interruption within
a task resumes at request granularity. Before any resume, all existing complete
task files are revalidated from raw sample bits; invalid completed checkpoints
are rejected rather than silently skipped.

GPU idle evidence is append-only: each fresh process segment creates
`extension_segment_gates/segment_NNNN_gpu_idle_gate.json` and never overwrites
an earlier gate. The final receipt binds the complete consecutive gate list, so
resuming cannot erase the startup evidence associated with prior checkpoints.

The final commit marker is `five_task_extension_receipt.json` plus its SHA
sidecar. The validator recomputes all accuracy aggregates from raw rows,
requires target/mapped sample identity equality, recomputes GSM strict and
numeric-final scores, recomputes WikiText PPL from log-likelihood/token counts,
and binds every result, numeric report, Hella bridge, extension code file,
config, GPU idle gate, summary and shared status by SHA256 and byte count.

On this 8×RTX 4090 PCIe server, budget roughly 14–24 hours for the full run.
MMLU contributes 112,336 option-score requests across the two modes; GSM8K
adds 2,638 long greedy generations. Actual time depends mainly on generated
GSM reasoning length and PCIe-host cache-transfer overhead. HellaSwag is not
inside this estimate because it is bridged rather than rerun.
