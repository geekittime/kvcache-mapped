#!/usr/bin/env bash
set -Eeuo pipefail

# Weighted capacity fallback for the local 8xRTX4090 reproduction.
#
# v2 (4 target + 2 source) failed with a real 32K OOM.  v3 (legacy-even
# 5 target + 3 source) completed its 64-token cell with the resident mapper,
# then failed in the 32K target re-prefill at Qwen3MLP/SiLU on target logical
# GPU0 because that card also owns embeddings.  v4 keeps the audited
# physical/logical topology while declaring
# exact contiguous decoder-layer counts: source=14,13,13 and
# target=8,15,14,14,13.  The lighter first target shard accounts for
# embed_tokens; the lighter last shard accounts for model.norm and lm_head.
# Every queue, endpoint, formal, log, and report path is fresh and cannot reuse
# either failed predecessor.  The paper does not disclose pair-level sharding;
# on this RTX4090/PCIe/no-P2P host, both absolute milliseconds and the measured
# speedup ratio are local context rather than hardware-equivalent H100 results.

readonly PROJECT_ROOT="/home/wby/llm-cache/nvidia-kvcache"
readonly BASE_LAUNCHER="${PROJECT_ROOT}/scripts/launch_latency_after_idle.sh"
readonly EXPECTED_BASE_LAUNCHER_SHA256="72f1cde7190c660748963c4d745fa0daeb87aa10406b8feb6a34640af964432e"

actual_base_sha256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
if [[ "${actual_base_sha256}" != "${EXPECTED_BASE_LAUNCHER_SHA256}" ]]; then
  printf 'refusing weighted-v4 latency launch: shared launcher SHA drifted (%s)\n' \
    "${actual_base_sha256}" >&2
  exit 1
fi

export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v4_weighted_5plus3"
export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v4_weighted_5plus3"
export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v4_weighted_5plus3"
export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v4_weighted_5plus3"

export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v4_weighted_5plus3.log"
export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v4_weighted_5plus3.verify.log"
export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v4_weighted_5plus3.preformal.verify.log"
export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v4_weighted_5plus3.log"
export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v4_weighted_5plus3.verify.log"
export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v4_weighted_5plus3.log"

export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"
export LATENCY_SOURCE_DEVICES="5,6,7"
export LATENCY_TARGET_DEVICES="0,1,2,3,4"
export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"
export LATENCY_TARGET_LAYER_COUNTS="8,15,14,14,13"
export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_low_first_v4"

exec "${BASE_LAUNCHER}" "$@"
