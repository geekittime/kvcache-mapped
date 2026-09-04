#!/usr/bin/env bash
set -Eeuo pipefail

# Fresh paper-main latency run after fixing target re-prefill inference mode.
#
# Every historical v2-v6 64-token checkpoint timed target re-prefill with
# autograd enabled while mapper transfer ran in inference mode.  Those values
# are retained only as failed-run diagnostics and must never be resumed or
# mixed into this run.  With autograd graphs removed, the earlier 32K capacity
# failures no longer justify the capacity-distorting v4-v6 splits.  v7 returns
# to the natural contiguous five-target/three-source balance, binding explicit
# source=14,13,13 and target=13,13,13,13,12 counts.  A fresh 64/32768 endpoint
# must still prove at least 1 GiB peak-reserved headroom before the ten-length
# 50-warmup/30-repeat formal sweep is allowed.

readonly PROJECT_ROOT="/home/wby/llm-cache/nvidia-kvcache"
readonly BASE_LAUNCHER="${PROJECT_ROOT}/scripts/launch_latency_after_idle.sh"
readonly EXPECTED_BASE_LAUNCHER_SHA256="dea19e35c15b2208b1be25f24f3ae207e70a6bedc92d6854a961fac34680b19f"

actual_base_sha256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
if [[ "${actual_base_sha256}" != "${EXPECTED_BASE_LAUNCHER_SHA256}" ]]; then
  printf 'refusing inference-mode equal-v7 latency launch: shared launcher SHA drifted (%s)\n' \
    "${actual_base_sha256}" >&2
  exit 1
fi

export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v7_inference_mode_equal_5plus3"
export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3"
export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v7_inference_mode_equal_5plus3"
export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v7_inference_mode_equal_5plus3"

export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.log"
export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.verify.log"
export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.preformal.verify.log"
export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v7_inference_mode_equal_5plus3.postformal.verify.log"
export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v7_inference_mode_equal_5plus3.log"
export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v7_inference_mode_equal_5plus3.verify.log"
export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v7_inference_mode_equal_5plus3.log"

export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"
export LATENCY_SOURCE_DEVICES="5,6,7"
export LATENCY_TARGET_DEVICES="0,1,2,3,4"
export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"
export LATENCY_TARGET_LAYER_COUNTS="13,13,13,13,12"
export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_equal_v7"

exec "${BASE_LAUNCHER}" "$@"
