#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh v9 paper-main Qwen3-14B -> Qwen3-32B latency reproduction.
#
# Source prefill is performed untimed on logical CUDA 5/6/7 (14/13/13), and
# every requested BF16 source cache is cloned bitwise-exactly to host RAM.  The
# source model is then deleted and its 5/6/7 allocator state is proved empty
# before the target/mapper are loaded on logical CUDA 0..5 (10/11/11/11/11/10).
# Each cell restores only its source cache to logical CUDA 6/7 (20/20) outside
# the peak/timer boundary.  T_mapper starts from that preexisting GPU cache and
# still includes source-to-target transfer.  The paper does not publish model
# residency/sharding; this three-phase topology is the local 8x4090 capacity
# configuration, not a claim about the authors' H100 placement.

readonly PROJECT_ROOT="/home/wby/llm-cache/nvidia-kvcache"
readonly BASE_LAUNCHER="${PROJECT_ROOT}/scripts/launch_latency_after_idle.sh"
readonly EXPECTED_BASE_LAUNCHER_SHA256="9df447fb5c3937a59ca2bcfd7aeeb9d9c093b53f3da66c2bc05e7f5043fd9c5c"

actual_base_sha256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
if [[ "${actual_base_sha256}" != "${EXPECTED_BASE_LAUNCHER_SHA256}" ]]; then
  printf 'refusing staged-v9 latency launch: shared launcher SHA drifted (%s)\n' \
    "${actual_base_sha256}" >&2
  exit 1
fi

export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v9_staged_6target_2cache"
export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v9_staged_6target_2cache"
export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v9_staged_6target_2cache"
export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v9_staged_6target_2cache"

export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v9_staged_6target_2cache.log"
export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v9_staged_6target_2cache.verify.log"
export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v9_staged_6target_2cache.preformal.verify.log"
export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v9_staged_6target_2cache.postformal.verify.log"
export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v9_staged_6target_2cache.log"
export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v9_staged_6target_2cache.verify.log"
export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v9_staged_6target_2cache.log"

export LATENCY_CUDA_VISIBILITY="2,3,4,5,6,7,0,1"
export LATENCY_SOURCE_DEVICES="6,7"
export LATENCY_TARGET_DEVICES="0,1,2,3,4,5"
export LATENCY_SOURCE_LAYER_COUNTS="20,20"
export LATENCY_TARGET_LAYER_COUNTS="10,11,11,11,11,10"
export LATENCY_STAGE_SOURCE_CACHES="1"
export LATENCY_SOURCE_PREFILL_DEVICES="5,6,7"
export LATENCY_SOURCE_PREFILL_LAYER_COUNTS="14,13,13"
export LATENCY_DEVICE_LAYOUT_PROFILE="staged_6target_2cache_v9"

exec "${BASE_LAUNCHER}" "$@"
