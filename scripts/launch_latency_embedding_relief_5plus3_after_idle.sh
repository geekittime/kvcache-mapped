#!/usr/bin/env bash
set -Eeuo pipefail

# Embedding-relief capacity fallback for the local 8xRTX4090 reproduction.
#
# The frozen v4 endpoint completed 64 tokens, then failed during the 32768-token
# target transformer re-prefill in Qwen3 RMSNorm while requesting 640 MiB on
# target logical GPU0.  That embedding-owning card reached 22.910 GiB peak
# reserved with only 0.740 GiB capacity headroom.  v5 keeps the audited
# physical/logical topology and source=14,13,13 split, but moves four decoder
# layers off the embedding card: target=4,15,15,15,15.  The other four target
# cards carry 15 layers each, no more than the largest v4 target shard.  This is a
# capacity inference, not a success claim: activations and allocator behavior
# are nonlinear, and the endpoint must still prove >=1 GiB headroom per card.
# All queue, endpoint, formal, log, and report paths are fresh.  The paper does
# not disclose pair-level sharding; RTX4090/PCIe/no-P2P measurements remain
# local context rather than hardware-equivalent H100 results.

readonly PROJECT_ROOT="/home/wby/llm-cache/nvidia-kvcache"
readonly BASE_LAUNCHER="${PROJECT_ROOT}/scripts/launch_latency_after_idle.sh"
readonly EXPECTED_BASE_LAUNCHER_SHA256="71607c4056dcd3de60d679baa875e07b4e93d0d11f29da72ece24783b8b56b1d"

actual_base_sha256="$(sha256sum "${BASE_LAUNCHER}" | awk '{print $1}')"
if [[ "${actual_base_sha256}" != "${EXPECTED_BASE_LAUNCHER_SHA256}" ]]; then
  printf 'refusing embedding-relief-v5 latency launch: shared launcher SHA drifted (%s)\n' \
    "${actual_base_sha256}" >&2
  exit 1
fi

export LATENCY_QUEUE_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_queue_v5_embedding_relief_5plus3"
export LATENCY_ENDPOINT_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3"
export LATENCY_FORMAL_ROOT="${PROJECT_ROOT}/runs/20260818_p8_latency_main_only_formal_v5_embedding_relief_5plus3"
export LATENCY_REPORT_ROOT="${PROJECT_ROOT}/artifacts/evaluation/qwen3_14b_to_32b/latency_main_only_v5_embedding_relief_5plus3"

export LATENCY_ENDPOINT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.log"
export LATENCY_ENDPOINT_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.verify.log"
export LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.preformal.verify.log"
export LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_endpoint_v5_embedding_relief_5plus3.postformal.verify.log"
export LATENCY_FORMAL_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v5_embedding_relief_5plus3.log"
export LATENCY_FORMAL_VERIFY_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_formal_v5_embedding_relief_5plus3.verify.log"
export LATENCY_REPORT_LOG="${PROJECT_ROOT}/runs/logs/20260818_p8_latency_main_only_report_v5_embedding_relief_5plus3.log"

export LATENCY_CUDA_VISIBILITY="3,4,5,6,7,0,1,2"
export LATENCY_SOURCE_DEVICES="5,6,7"
export LATENCY_TARGET_DEVICES="0,1,2,3,4"
export LATENCY_SOURCE_LAYER_COUNTS="14,13,13"
export LATENCY_TARGET_LAYER_COUNTS="4,15,15,15,15"
export LATENCY_DEVICE_LAYOUT_PROFILE="balanced_5target_3source_embedding_relief_v5"

exec "${BASE_LAUNCHER}" "$@"
