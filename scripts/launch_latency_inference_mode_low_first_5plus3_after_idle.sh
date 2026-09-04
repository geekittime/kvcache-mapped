#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh paper-main latency run after the equal-v7 inference-mode endpoint
# reached a genuine 32K capacity limit on the embedding-bearing first target
# shard.  v7 measured 24,209,522,688 peak-reserved bytes on logical cuda:0
# with only 1,184,169,984 bytes of capacity headroom before a further 1.56-GiB
# allocation failed.  Moving seven target layers away from that shard releases
# resident and 32K-KV capacity; after also charging the failed 1.56-GiB request,
# each 15-layer shard retains about 6.2 GiB of projected headroom.  The fresh
# endpoint remains authoritative and must prove at least 1 GiB of measured
# peak-reserved headroom at both 64 and 32768 before formal execution.
#
# Timing semantics are unchanged: source prefill is outside T_mapper and
# T_reprefill, both paper-main metrics run in inference mode, endpoint is 1/2,
# and the formal sweep is the ten paper lengths with 50/30 samples.

readonly PROJECT_ROOT="/home/wby/llm-cache/nvidia-kvcache"
readonly BASE_LAUNCHER="${PROJECT_ROOT}/scripts/launch_latency_after_idle.sh"
readonly EXPECTED_BASE_LAUNCHER_SHA256="2865c35ffe0aead06728e187b7da5e2f8bcc5d4944b2bf76bb02431b044eaf92"

actual_base_sha256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
if [[ "${actual_base_sha256}" != "${EXPECTED_BASE_LAUNCHER_SHA256}" ]]; then
  printf 'refusing inference-mode low-first-v8 latency launch: shared launcher SHA drifted (%s)\n' \
    "${actual_base_sha256}" >&2
  exit 1
fi

export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v8_inference_mode_low_first_5plus3"
export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3"
export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v8_inference_mode_low_first_5plus3"
export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v8_inference_mode_low_first_5plus3"

export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.log"
export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.verify.log"
export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.preformal.verify.log"
export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v8_inference_mode_low_first_5plus3.postformal.verify.log"
export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v8_inference_mode_low_first_5plus3.log"
export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v8_inference_mode_low_first_5plus3.verify.log"
export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v8_inference_mode_low_first_5plus3.log"

export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"
export LATENCY_SOURCE_DEVICES="5,6,7"
export LATENCY_TARGET_DEVICES="0,1,2,3,4"
export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"
export LATENCY_TARGET_LAYER_COUNTS="6,15,15,15,13"
export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_inference_low_first_v8"

exec "${BASE_LAUNCHER}" "$@"
