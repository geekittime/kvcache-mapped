#!/usr/bin/env bash
set -Eeuo pipefail

# Single-first-layer capacity fallback for the local 8xRTX4090 reproduction.
#
# The frozen v5 endpoint completed 64 tokens, then failed during the 32768-token
# target transformer re-prefill in Qwen3 RoPE while requesting 512 MiB on target
# logical GPU0.  That embedding-owning card reached 22.982 GiB peak reserved and
# had only 196.25 MiB driver free at the failed allocation.  Runtime baselines
# show that moving three of its four decoder layers releases about 2.912 GiB of
# resident BF16-model plus FP32-mapper memory; their 32K KV cache is another
# 0.375 GiB.  v6 therefore keeps one real decoder layer beside embed_tokens and
# uses target=1,16,16,16,15 with the unchanged source=14,13,13 split.  Keeping a
# positive first shard uses the already audited core parser/model-loading path
# and preserves four cross-device hidden-state boundaries in target re-prefill.
# This remains a capacity inference, not a success claim: the fresh endpoint
# must prove at least 1 GiB peak-reserved headroom on every selected device.
# The paper does not disclose pair-level sharding; RTX4090/PCIe/no-P2P timings
# remain local context rather than hardware-equivalent H100 results.

readonly PROJECT_ROOT="/home/wby/llm-cache/nvidia-kvcache"
readonly BASE_LAUNCHER="${PROJECT_ROOT}/scripts/launch_latency_after_idle.sh"
readonly EXPECTED_BASE_LAUNCHER_SHA256="b2369daa3c377e5aebf45df4472834757e511423518b3d5defeddbb4ec570595"

actual_base_sha256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
if [[ "${actual_base_sha256}" != "${EXPECTED_BASE_LAUNCHER_SHA256}" ]]; then
  printf 'refusing single-first-v6 latency launch: shared launcher SHA drifted (%s)\n' \
    "${actual_base_sha256}" >&2
  exit 1
fi

export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v6_single_first_5plus3"
export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3"
export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v6_single_first_5plus3"
export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v6_single_first_5plus3"

export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.log"
export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.verify.log"
export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.preformal.verify.log"
export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v6_single_first_5plus3.postformal.verify.log"
export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v6_single_first_5plus3.log"
export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v6_single_first_5plus3.verify.log"
export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v6_single_first_5plus3.log"

export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"
export LATENCY_SOURCE_DEVICES="5,6,7"
export LATENCY_TARGET_DEVICES="0,1,2,3,4"
export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"
export LATENCY_TARGET_LAYER_COUNTS="1,16,16,16,15"
export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_single_first_v6"

exec "${BASE_LAUNCHER}" "$@"
