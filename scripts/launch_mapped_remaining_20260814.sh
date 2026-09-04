#!/usr/bin/env bash
set -u

cd /home/wby/llm-cache/nvidia-kvcache
export CUDA_VISIBLE_DEVICES=4,5,6,7,0,1
export PYTHONUNBUFFERED=1

# Non-generation task: keep the paper-aligned frozen configuration.
.venv/bin/python scripts/evaluate.py \
  --config configs/eval_qwen3_14b_to_32b.yaml \
  --modes mapped \
  --tasks mmlu \
  --log-samples \
  --output-dir runs/20260814_062228_p7_mapped_mmlu
mmlu_status=$?

if [ "$mmlu_status" -ne 0 ]; then
  echo "MMLU mapped evaluation failed with status $mmlu_status; GSM8K was not started." >&2
  exit "$mmlu_status"
fi

# Generation task: use the user-requested larger generation cap and a separate
# output directory/fingerprint.  This is intentionally not mixed with the
# frozen max-256 non-generation run.
.venv/bin/python scripts/evaluate.py \
  --config configs/eval_qwen3_14b_to_32b_max1024.yaml \
  --modes mapped \
  --tasks gsm8k_cot \
  --log-samples \
  --output-dir runs/20260814_062228_p7_mapped_gsm_max1024
exit $?
