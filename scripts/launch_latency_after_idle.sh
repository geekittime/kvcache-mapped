#!/usr/bin/env bash
set -Eeuo pipefail

# Fail-closed production latency queue for the Qwen3-14B -> Qwen3-32B run.
#
# The endpoint and formal runs deliberately use fresh directories.  This
# wrapper waits for a stable, physically idle eight-GPU window, runs the 64 /
# 32768 endpoint, validates its immutable artifacts and a 1-GiB peak-reserved
# headroom policy, waits for a second clean window, and only then starts the
# paper-main 50-warmup / 30-repeat sweep.  Component profiling, boundary
# decode, and protocol-complete timing are deliberately excluded: Appendix G
# compares only target transformer re-prefill with KV-cache transfer.

readonly PROJECT_ROOT="/home/wby/llm-cache/nvidia-kvcache"
readonly PYTHON="${PROJECT_ROOT}/.venv/bin/python"
readonly RUNS_ROOT="${PROJECT_ROOT}/runs"
readonly LOG_ROOT="${RUNS_ROOT}/logs"
readonly QUEUE_ROOT="${LATENCY_QUEUE_ROOT:-${RUNS_ROOT}/20260818_p8_latency_main_only_queue_v2}"
readonly ENDPOINT_ROOT="${LATENCY_ENDPOINT_ROOT:-${RUNS_ROOT}/20260818_p8_latency_main_only_endpoint_v2}"
readonly FORMAL_ROOT="${LATENCY_FORMAL_ROOT:-${RUNS_ROOT}/20260818_p8_latency_main_only_formal_v2}"
readonly REPORT_ROOT="${LATENCY_REPORT_ROOT:-${RUNS_ROOT}/20260818_p8_latency_main_only_report_v2}"
readonly STATE_FILE="${QUEUE_ROOT}/state.tsv"
readonly WAIT_LOG="${QUEUE_ROOT}/wait.log"
readonly ENDPOINT_LOG="${LATENCY_ENDPOINT_LOG:-${LOG_ROOT}/20260818_p8_latency_main_only_endpoint_v2.log}"
readonly ENDPOINT_VERIFY_LOG="${LATENCY_ENDPOINT_VERIFY_LOG:-${LOG_ROOT}/20260818_p8_latency_main_only_endpoint_v2.verify.log}"
readonly ENDPOINT_PREFORMAL_VERIFY_LOG="${LATENCY_ENDPOINT_PREFORMAL_VERIFY_LOG:-${LOG_ROOT}/20260818_p8_latency_main_only_endpoint_v2.preformal.verify.log}"
readonly ENDPOINT_POSTFORMAL_VERIFY_LOG="${LATENCY_ENDPOINT_POSTFORMAL_VERIFY_LOG:-${ENDPOINT_LOG}.postformal.verify}"
readonly FORMAL_LOG="${LATENCY_FORMAL_LOG:-${LOG_ROOT}/20260818_p8_latency_main_only_formal_v2.log}"
readonly FORMAL_VERIFY_LOG="${LATENCY_FORMAL_VERIFY_LOG:-${LOG_ROOT}/20260818_p8_latency_main_only_formal_v2.verify.log}"
readonly REPORT_LOG="${LATENCY_REPORT_LOG:-${LOG_ROOT}/20260818_p8_latency_main_only_report_v2.log}"
readonly REPORT_VERIFY_LOG="${REPORT_LOG}.verify"
readonly LOCK_FILE="${RUNS_ROOT}/.qwen3_latency_queue.lock"

readonly EXPECTED_LATENCY_CODE_DIGEST="00b2a0c71664e062a777ff5b4f1299e744e63229a86f2115d570b92fdfa181f3"
readonly EXPECTED_CLI_SHA256="85b31d2e0eb430107c364df3c5c03327e0b7597cd0b5788eac175de0fe047b75"
readonly EXPECTED_LIBRARY_SHA256="1566894a35cdf52d947d78e9499efcd13d4d0fd5ea530b5c5e73af858b57e582"
readonly EXPECTED_REPORT_LIBRARY_SHA256="18694924856791ede11ba6a1e349bcd3f160a8bc1d752a033e385154d7c45dcf"
readonly EXPECTED_REPORT_CLI_SHA256="4c2cf64f4f843553a17d3131fa034529a01dc6ced65fafd33ba8e2395ca13c9b"
readonly CUDA_VISIBILITY="${LATENCY_CUDA_VISIBILITY:-4,5,6,7,0,1}"
readonly SOURCE_DEVICE_CSV="${LATENCY_SOURCE_DEVICES:-4,5}"
readonly TARGET_DEVICE_CSV="${LATENCY_TARGET_DEVICES:-0,1,2,3}"
readonly SELECTED_DEVICE_CSV="${SOURCE_DEVICE_CSV},${TARGET_DEVICE_CSV}"
readonly SOURCE_LAYER_COUNT_CSV="${LATENCY_SOURCE_LAYER_COUNTS:-}"
readonly TARGET_LAYER_COUNT_CSV="${LATENCY_TARGET_LAYER_COUNTS:-}"
readonly STAGE_SOURCE_CACHES="${LATENCY_STAGE_SOURCE_CACHES:-0}"
readonly SOURCE_PREFILL_DEVICE_CSV="${LATENCY_SOURCE_PREFILL_DEVICES:-}"
readonly SOURCE_PREFILL_LAYER_COUNT_CSV="${LATENCY_SOURCE_PREFILL_LAYER_COUNTS:-}"
readonly DEVICE_LAYOUT_PROFILE="${LATENCY_DEVICE_LAYOUT_PROFILE:-legacy_4target_2source}"
readonly IDLE_MEMORY_LIMIT_MIB=64
readonly IDLE_SAMPLES_REQUIRED=12
readonly IDLE_SAMPLE_INTERVAL_SECONDS=5
readonly ENDPOINT_MIN_HEADROOM_BYTES=$((1 << 30))
CURRENT_STAGE="initializing"

SHARDING_ARGS=()
if [[ -n "${SOURCE_LAYER_COUNT_CSV}" || -n "${TARGET_LAYER_COUNT_CSV}" ]]; then
  SHARDING_ARGS=(
    "--source-layer-counts" "${SOURCE_LAYER_COUNT_CSV}"
    "--target-layer-counts" "${TARGET_LAYER_COUNT_CSV}"
  )
fi
readonly -a SHARDING_ARGS

STAGING_ARGS=()
if [[ "${STAGE_SOURCE_CACHES}" == "1" ]]; then
  STAGING_ARGS=(
    "--stage-source-caches-on-cpu-before-cells"
    "--source-prefill-devices" "${SOURCE_PREFILL_DEVICE_CSV}"
    "--source-prefill-layer-counts" "${SOURCE_PREFILL_LAYER_COUNT_CSV}"
  )
fi
readonly -a STAGING_ARGS

readonly -a ENDPOINT_ARGS=(
  "${PYTHON}" "scripts/benchmark_latency.py"
  "--mode" "production"
  "--source-model" "/home/wby/models/Qwen3-14B"
  "--target-model" "/home/wby/models/Qwen3-32B"
  "--mapper" "${PROJECT_ROOT}/artifacts/mappers/qwen3_14b_to_32b/k8_lambda0.01"
  "--source-devices" "${SOURCE_DEVICE_CSV}"
  "--target-devices" "${TARGET_DEVICE_CSV}"
  "${SHARDING_ARGS[@]}"
  "${STAGING_ARGS[@]}"
  "--lengths" "64,32768"
  "--warmup" "1"
  "--repeats" "2"
  "--mapper-token-chunk" "1024"
  "--attention" "flash_attention_2"
  "--seed" "20260813"
  "--preload-idle-memory-limit-mib" "64"
  "--preload-idle-samples" "12"
  "--preload-idle-sample-interval-seconds" "5"
  "--preload-idle-timeout-seconds" "86400"
  "--output-dir" "${ENDPOINT_ROOT}"
)

readonly -a FORMAL_ARGS=(
  "${PYTHON}" "scripts/benchmark_latency.py"
  "--mode" "production"
  "--source-model" "/home/wby/models/Qwen3-14B"
  "--target-model" "/home/wby/models/Qwen3-32B"
  "--mapper" "${PROJECT_ROOT}/artifacts/mappers/qwen3_14b_to_32b/k8_lambda0.01"
  "--source-devices" "${SOURCE_DEVICE_CSV}"
  "--target-devices" "${TARGET_DEVICE_CSV}"
  "${SHARDING_ARGS[@]}"
  "${STAGING_ARGS[@]}"
  "--lengths" "64,128,256,512,1024,2048,4096,8192,16384,32768"
  "--warmup" "50"
  "--repeats" "30"
  "--mapper-token-chunk" "1024"
  "--attention" "flash_attention_2"
  "--seed" "20260813"
  "--preload-idle-memory-limit-mib" "64"
  "--preload-idle-samples" "12"
  "--preload-idle-sample-interval-seconds" "5"
  "--preload-idle-timeout-seconds" "86400"
  "--output-dir" "${FORMAL_ROOT}"
)

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_state() {
  local stage="$1"
  local detail="${2:-}"
  local temporary="${STATE_FILE}.tmp.$$"
  CURRENT_STAGE="${stage}"
  printf '%s\t%s\t%s\n' "$(timestamp)" "${stage}" "${detail}" >"${temporary}"
  mv "${temporary}" "${STATE_FILE}"
}

log_wait() {
  if [[ -d "${QUEUE_ROOT}" ]]; then
    printf '%s %s\n' "$(timestamp)" "$*" | tee -a "${WAIT_LOG}"
  else
    printf '%s %s\n' "$(timestamp)" "$*" >&2
  fi
}

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

current_latency_digest() {
  env CUDA_VISIBLE_DEVICES='' PYTHONPATH="${PROJECT_ROOT}/src" "${PYTHON}" - <<'PY'
import scripts.benchmark_latency as latency_cli
print(latency_cli.project_code_provenance()["files_digest"])
PY
}

verify_frozen_code() {
  local cli_sha library_sha report_sha report_cli_sha digest
  cli_sha="$(sha256_of "${PROJECT_ROOT}/scripts/benchmark_latency.py")"
  library_sha="$(sha256_of "${PROJECT_ROOT}/src/kvcache_transfer/latency.py")"
  report_sha="$(sha256_of "${PROJECT_ROOT}/src/kvcache_transfer/latency_report.py")"
  report_cli_sha="$(sha256_of "${PROJECT_ROOT}/scripts/summarize_latency.py")"
  digest="$(cd "${PROJECT_ROOT}" && current_latency_digest | tail -n 1)"
  [[ "${cli_sha}" == "${EXPECTED_CLI_SHA256}" ]] || {
    log_wait "ABORT code drift: benchmark_latency.py=${cli_sha}"
    return 1
  }
  [[ "${library_sha}" == "${EXPECTED_LIBRARY_SHA256}" ]] || {
    log_wait "ABORT code drift: latency.py=${library_sha}"
    return 1
  }
  [[ "${report_sha}" == "${EXPECTED_REPORT_LIBRARY_SHA256}" ]] || {
    log_wait "ABORT code drift: latency_report.py=${report_sha}"
    return 1
  }
  [[ "${report_cli_sha}" == "${EXPECTED_REPORT_CLI_SHA256}" ]] || {
    log_wait "ABORT code drift: summarize_latency.py=${report_cli_sha}"
    return 1
  }
  [[ "${digest}" == "${EXPECTED_LATENCY_CODE_DIGEST}" ]] || {
    log_wait "ABORT latency code digest drift: ${digest}"
    return 1
  }
}

verify_layout_contract() {
  case "${SOURCE_DEVICE_CSV}|${TARGET_DEVICE_CSV}|${CUDA_VISIBILITY}|${SOURCE_LAYER_COUNT_CSV}|${TARGET_LAYER_COUNT_CSV}|${STAGE_SOURCE_CACHES}|${SOURCE_PREFILL_DEVICE_CSV}|${SOURCE_PREFILL_LAYER_COUNT_CSV}|${DEVICE_LAYOUT_PROFILE}" in
    "4,5|0,1,2,3|4,5,6,7,0,1|||0|||legacy_4target_2source"|\
    "5,6,7|0,1,2,3,4|3,4,5,6,7,0,1,2|||0|||balanced_5target_3source_v3"|\
    "5,6,7|0,1,2,3,4|3,4,5,6,7,0,1,2|14,13,13|8,15,14,14,13|0|||balanced_5target_3source_low_first_v4"|\
    "5,6,7|0,1,2,3,4|3,4,5,6,7,0,1,2|14,13,13|4,15,15,15,15|0|||balanced_5target_3source_embedding_relief_v5"|\
    "5,6,7|0,1,2,3,4|3,4,5,6,7,0,1,2|14,13,13|1,16,16,16,15|0|||balanced_5target_3source_single_first_v6"|\
    "5,6,7|0,1,2,3,4|3,4,5,6,7,0,1,2|14,13,13|13,13,13,13,12|0|||balanced_5target_3source_equal_v7"|\
    "5,6,7|0,1,2,3,4|3,4,5,6,7,0,1,2|14,13,13|6,15,15,15,13|0|||balanced_5target_3source_inference_low_first_v8"|\
    "6,7|0,1,2,3,4,5|2,3,4,5,6,7,0,1|20,20|10,11,11,11,11,10|1|5,6,7|14,13,13|staged_6target_2cache_v9")
      return 0
      ;;
    *)
      log_wait "ABORT unaudited latency layout/sharding: source_cache=${SOURCE_DEVICE_CSV};target=${TARGET_DEVICE_CSV};CUDA_VISIBLE_DEVICES=${CUDA_VISIBILITY};source_cache_counts=${SOURCE_LAYER_COUNT_CSV:-null};target_counts=${TARGET_LAYER_COUNT_CSV:-null};staged=${STAGE_SOURCE_CACHES};source_prefill=${SOURCE_PREFILL_DEVICE_CSV:-null};source_prefill_counts=${SOURCE_PREFILL_LAYER_COUNT_CSV:-null};profile=${DEVICE_LAYOUT_PROFILE}"
      return 1
      ;;
  esac
}

memory_snapshot() {
  nvidia-smi \
    --query-gpu=index,memory.used \
    --format=csv,noheader,nounits
}

memory_snapshot_is_idle() {
  local snapshot="$1"
  local -a seen=()
  local index used extra
  while IFS=',' read -r index used extra; do
    index="${index//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    [[ -z "${extra:-}" ]] || return 1
    [[ "${index}" =~ ^[0-7]$ ]] || return 1
    [[ "${used}" =~ ^[0-9]+$ ]] || return 1
    [[ -z "${seen[${index}]:-}" ]] || return 1
    seen[${index}]=1
    (( used <= IDLE_MEMORY_LIMIT_MIB )) || return 1
  done <<<"${snapshot}"
  [[ "${#seen[@]}" -eq 8 ]]
}

single_idle_sample() {
  local before after applications
  before="$(memory_snapshot)" || return 1
  memory_snapshot_is_idle "${before}" || return 1
  applications="$(nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>&1)" || return 1
  [[ -z "${applications//[[:space:]]/}" ]] || return 1
  sleep 1 || return 1
  after="$(memory_snapshot)" || return 1
  memory_snapshot_is_idle "${after}" || return 1
  applications="$(nvidia-smi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>&1)" || return 1
  [[ -z "${applications//[[:space:]]/}" ]] || return 1
}

wait_for_clean_gpu_window() {
  local label="$1"
  local streak=0
  local polls=0
  write_state "waiting_for_idle" "${label}"
  log_wait "Waiting for ${label}: eight GPUs <=${IDLE_MEMORY_LIMIT_MIB} MiB, no compute apps, ${IDLE_SAMPLES_REQUIRED} consecutive samples."
  while (( streak < IDLE_SAMPLES_REQUIRED )); do
    ((polls += 1))
    if single_idle_sample; then
      ((streak += 1))
      log_wait "${label} idle streak ${streak}/${IDLE_SAMPLES_REQUIRED}"
    else
      if (( streak > 0 || polls % 60 == 1 )); then
        log_wait "${label} busy; streak reset. Current physical GPU state:"
        nvidia-smi \
          --query-gpu=index,memory.used,memory.free,utilization.gpu \
          --format=csv,noheader | tee -a "${WAIT_LOG}" || true
        nvidia-smi \
          --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
          --format=csv,noheader | tee -a "${WAIT_LOG}" || true
      fi
      streak=0
    fi
    if (( streak < IDLE_SAMPLES_REQUIRED )); then
      sleep "${IDLE_SAMPLE_INTERVAL_SECONDS}"
    fi
  done
  log_wait "${label} passed."
}

verify_host_capacity() {
  local available_memory available_disk
  available_memory="$(awk '/^MemAvailable:/ {printf "%.0f\n", $2 * 1024}' /proc/meminfo)"
  available_disk="$(df -PB1 "${PROJECT_ROOT}" | awk 'NR == 2 {print $4}')"
  (( available_memory >= 200 * 1024 * 1024 * 1024 )) || {
    log_wait "ABORT host MemAvailable is below 200 GiB: ${available_memory} bytes"
    return 1
  }
  (( available_disk >= 100 * 1024 * 1024 * 1024 )) || {
    log_wait "ABORT project filesystem free space is below 100 GiB: ${available_disk} bytes"
    return 1
  }
}

run_logged() {
  local log_path="$1"
  shift
  set +e
  env \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBILITY}" \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    PYTHONUNBUFFERED=1 \
    "$@" 2>&1 | tee "${log_path}"
  local -a pipeline_status=("${PIPESTATUS[@]}")
  local command_status="${pipeline_status[0]}"
  local tee_status="${pipeline_status[1]}"
  set -e
  if (( command_status != 0 )); then
    return "${command_status}"
  fi
  if (( tee_status != 0 )); then
    return 125
  fi
  return 0
}

require_complete_marker() {
  local run_root="$1"
  env RUN_ROOT="${run_root}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
for name in ("result.json", "status.json", "results.manifest.json", "metrics.json"):
    path = root / name
    if not path.is_file():
        raise SystemExit(f"missing completion artifact: {path}")
for name in ("result.json", "status.json", "metrics.json"):
    value = json.loads((root / name).read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise SystemExit(f"{name} is not complete")
PY
}

reject_fatal_log_markers() {
  local log_path="$1"
  if grep -Eqi \
    'CUDA out of memory|OutOfMemoryError|allocator failure|SIGKILL|exit (status|code) 137|Killed process' \
    "${log_path}"; then
    log_wait "ABORT fatal CUDA/process marker found in ${log_path}"
    return 1
  fi
}

verify_endpoint_policy() {
  env \
    ENDPOINT_ROOT="${ENDPOINT_ROOT}" \
    ENDPOINT_MIN_HEADROOM_BYTES="${ENDPOINT_MIN_HEADROOM_BYTES}" \
    EXPECTED_CUDA_VISIBILITY="${CUDA_VISIBILITY}" \
    EXPECTED_SELECTED_DEVICES="${SELECTED_DEVICE_CSV}" \
    EXPECTED_SOURCE_DEVICES="${SOURCE_DEVICE_CSV}" \
    EXPECTED_TARGET_DEVICES="${TARGET_DEVICE_CSV}" \
    EXPECTED_SOURCE_LAYER_COUNTS="${SOURCE_LAYER_COUNT_CSV}" \
    EXPECTED_TARGET_LAYER_COUNTS="${TARGET_LAYER_COUNT_CSV}" \
    EXPECTED_STAGE_SOURCE_CACHES="${STAGE_SOURCE_CACHES}" \
    EXPECTED_SOURCE_PREFILL_DEVICES="${SOURCE_PREFILL_DEVICE_CSV}" \
    EXPECTED_SOURCE_PREFILL_LAYER_COUNTS="${SOURCE_PREFILL_LAYER_COUNT_CSV}" \
    EXPECTED_DEVICE_LAYOUT_PROFILE="${DEVICE_LAYOUT_PROFILE}" \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    "${PYTHON}" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

root = Path(os.environ["ENDPOINT_ROOT"])
minimum_headroom = int(os.environ["ENDPOINT_MIN_HEADROOM_BYTES"])
expected_devices = [int(item) for item in os.environ["EXPECTED_SELECTED_DEVICES"].split(",")]
expected_source_devices = [int(item) for item in os.environ["EXPECTED_SOURCE_DEVICES"].split(",")]
expected_target_devices = [int(item) for item in os.environ["EXPECTED_TARGET_DEVICES"].split(",")]
expected_source_counts = [
    int(item) for item in os.environ["EXPECTED_SOURCE_LAYER_COUNTS"].split(",") if item
]
expected_target_counts = [
    int(item) for item in os.environ["EXPECTED_TARGET_LAYER_COUNTS"].split(",") if item
]
staged_source_lifecycle = os.environ["EXPECTED_STAGE_SOURCE_CACHES"] == "1"
expected_source_prefill_devices = [
    int(item) for item in os.environ["EXPECTED_SOURCE_PREFILL_DEVICES"].split(",") if item
]
expected_source_prefill_counts = [
    int(item) for item in os.environ["EXPECTED_SOURCE_PREFILL_LAYER_COUNTS"].split(",") if item
]
expected_profile = os.environ["EXPECTED_DEVICE_LAYOUT_PROFILE"]
exact_count_profile = expected_profile in {
    "balanced_5target_3source_low_first_v4",
    "balanced_5target_3source_embedding_relief_v5",
    "balanced_5target_3source_single_first_v6",
    "balanced_5target_3source_equal_v7",
    "balanced_5target_3source_inference_low_first_v8",
    "staged_6target_2cache_v9",
}
receipt = {"schema_version": 1, "status": "passed", "lengths": {}}
run_definition_path = root / "run_definition.json"
run_definition = json.loads(run_definition_path.read_text(encoding="utf-8"))
configuration = json.loads((root / "config.json").read_text(encoding="utf-8"))
mapper_setup = json.loads((root / "mapper_setup.json").read_text(encoding="utf-8"))
preload_gate_path = root / "preload_gpu_idle_gate.json"
preload_gate = json.loads(preload_gate_path.read_text(encoding="utf-8"))

def digest(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def sha256_file(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def expected_layer_sharding():
    result = {
        "source": {
            "mode": "explicit_exact_counts",
            "devices": expected_source_devices,
            "declared_layer_counts": expected_source_counts,
            "resolved_layer_counts": expected_source_counts,
            "total_layers": 40,
        },
        "target": {
            "mode": "explicit_exact_counts",
            "devices": expected_target_devices,
            "declared_layer_counts": expected_target_counts,
            "resolved_layer_counts": expected_target_counts,
            "total_layers": 64,
        },
    }
    if staged_source_lifecycle:
        result["source_model_prefill"] = {
            "mode": "explicit_exact_counts",
            "devices": expected_source_prefill_devices,
            "declared_layer_counts": expected_source_prefill_counts,
            "resolved_layer_counts": expected_source_prefill_counts,
            "total_layers": 40,
        }
        result["source_cache_staging"] = dict(result["source"])
    return result

def validate_runtime_layer_sharding(actual):
    if not isinstance(actual, dict) or set(actual) != {"source", "target"}:
        raise SystemExit("endpoint runtime layer-sharding roles mismatch")
    for role, devices, counts, total in (
        ("source", expected_source_devices, expected_source_counts, 40),
        ("target", expected_target_devices, expected_target_counts, 64),
    ):
        ranges = []
        start = 0
        for device, count in zip(devices, counts, strict=True):
            ranges.append(
                {
                    "device": device,
                    "start_layer": start,
                    "end_layer_exclusive": start + count,
                    "count": count,
                }
            )
            start += count
        expected = {
            "status": "passed",
            "role": role,
            "mode": "explicit_exact_counts",
            "devices": devices,
            "declared_layer_counts": counts,
            "expected_layer_counts": counts,
            "actual_layer_counts": counts,
            "total_layers": total,
            "layer_devices": [
                device
                for device, count in zip(devices, counts, strict=True)
                for _ in range(count)
            ],
            "contiguous_ranges": ranges,
            "actual_parameters_inspected_for_every_layer": True,
            "auxiliary_parameters_inspected_for_every_module": True,
        }
        item = actual.get(role)
        if not isinstance(item, dict):
            raise SystemExit(f"endpoint runtime {role} sharding evidence is missing")
        if set(item) != {
            *expected,
            "parameter_tensors_inspected",
            "auxiliary_modules",
            "auxiliary_parameter_tensors_inspected",
        }:
            raise SystemExit(f"endpoint runtime {role} sharding evidence fields mismatch")
        if any(item.get(field) != value for field, value in expected.items()):
            raise SystemExit(f"endpoint runtime {role} sharding evidence mismatch")
        inspected = item.get("parameter_tensors_inspected")
        if type(inspected) is not int or inspected != total * 11:
            raise SystemExit(f"endpoint runtime {role} parameter inspection count mismatch")
        expected_aux_devices = {
            "model.embed_tokens": devices[0],
            "model.norm": devices[-1],
            "lm_head": devices[-1],
        }
        auxiliary = item.get("auxiliary_modules")
        if not isinstance(auxiliary, dict) or set(auxiliary) != set(expected_aux_devices):
            raise SystemExit(f"endpoint runtime {role} auxiliary module coverage mismatch")
        auxiliary_tensors = 0
        for module_name, expected_device in expected_aux_devices.items():
            module = auxiliary[module_name]
            if not isinstance(module, dict) or set(module) != {
                "expected_device", "actual_device", "parameter_tensors_inspected"
            }:
                raise SystemExit(f"endpoint runtime {role} auxiliary module fields mismatch")
            if module.get("expected_device") != expected_device or module.get("actual_device") != expected_device:
                raise SystemExit(f"endpoint runtime {role} auxiliary module device mismatch")
            count = module.get("parameter_tensors_inspected")
            if type(count) is not int or count < 1:
                raise SystemExit(f"endpoint runtime {role} auxiliary parameter inspection is incomplete")
            auxiliary_tensors += count
        if auxiliary_tensors != 3 or item.get("auxiliary_parameter_tensors_inspected") != 3:
            raise SystemExit(f"endpoint runtime {role} auxiliary parameter total mismatch")
    return actual

if exact_count_profile:
    if run_definition.get("cli", {}).get("source_layer_counts") != expected_source_counts:
        raise SystemExit("endpoint source layer-count declaration mismatch")
    if run_definition.get("cli", {}).get("target_layer_counts") != expected_target_counts:
        raise SystemExit("endpoint target layer-count declaration mismatch")
    sharding = expected_layer_sharding()
    if run_definition.get("resolved", {}).get("layer_sharding") != sharding:
        raise SystemExit("endpoint fingerprint-bound resolved layer sharding mismatch")
    if configuration.get("source_layer_counts") != expected_source_counts:
        raise SystemExit("endpoint config source layer-count mismatch")
    if configuration.get("target_layer_counts") != expected_target_counts:
        raise SystemExit("endpoint config target layer-count mismatch")
    if configuration.get("layer_sharding") != sharding:
        raise SystemExit("endpoint config resolved layer sharding mismatch")
    if staged_source_lifecycle:
        runtime_sharding = mapper_setup.get("model_layer_sharding")
        if (
            not isinstance(runtime_sharding, dict)
            or set(runtime_sharding)
            != {"source", "source_model_prefill", "source_cache_staging", "target"}
        ):
            raise SystemExit("endpoint staged runtime layer-sharding roles mismatch")
        if runtime_sharding["source"] != runtime_sharding["source_model_prefill"]:
            raise SystemExit("endpoint staged source/prefill aliases differ")
    else:
        runtime_sharding = validate_runtime_layer_sharding(
            mapper_setup.get("model_layer_sharding")
        )
    actual_mapper_devices = []
    for layer, item in enumerate(mapper_setup.get("per_layer", [])):
        if item.get("target_layer") != layer:
            raise SystemExit("endpoint mapper target layers are incomplete/reordered")
        device = str(item.get("device", ""))
        if not device.startswith("cuda:"):
            raise SystemExit("endpoint mapper target layer has malformed device")
        actual_mapper_devices.append(int(device.removeprefix("cuda:")))
    expected_mapper_devices = runtime_sharding["target"]["layer_devices"]
    if actual_mapper_devices != expected_mapper_devices:
        raise SystemExit("endpoint actual mapper target distribution mismatch")
    receipt.update(
        {
            "device_layout_profile": expected_profile,
            "source_layer_counts": expected_source_counts,
            "target_layer_counts": expected_target_counts,
            "model_layer_sharding": runtime_sharding,
            "actual_mapper_target_layer_counts": expected_target_counts,
        }
    )

if staged_source_lifecycle:
    expected_runtime_topology = {
        "policy": "untimed_cpu_staging_then_gpu_restage_v1",
        "cuda_visible_devices_environment": "2,3,4,5,6,7,0,1",
        "logical_to_physical": {
            str(index): physical
            for index, physical in enumerate((2, 3, 4, 5, 6, 7, 0, 1))
        },
        "source_model_prefill": {
            "logical_devices": expected_source_prefill_devices,
            "physical_devices": [7, 0, 1],
            "layer_counts": expected_source_prefill_counts,
        },
        "source_cache_staging": {
            "logical_devices": expected_source_devices,
            "physical_devices": [0, 1],
            "layer_counts": expected_source_counts,
        },
        "target_timed": {
            "logical_devices": expected_target_devices,
            "physical_devices": [2, 3, 4, 5, 6, 7],
            "layer_counts": expected_target_counts,
        },
        "source_prefill_target_logical_overlap": [5],
        "source_prefill_target_physical_overlap": [7],
        "source_cache_target_logical_overlap": [],
        "source_cache_target_physical_overlap": [],
        "logical_5_unique_sequential_reuse_physical_device": 7,
    }
    if run_definition.get("cli", {}).get("stage_source_caches_on_cpu_before_cells") is not True:
        raise SystemExit("endpoint staged CLI flag is missing")
    if run_definition.get("cli", {}).get("source_prefill_devices") != expected_source_prefill_devices:
        raise SystemExit("endpoint staged source-prefill devices drifted")
    if run_definition.get("cli", {}).get("source_prefill_layer_counts") != expected_source_prefill_counts:
        raise SystemExit("endpoint staged source-prefill counts drifted")
    if configuration.get("staged_runtime_topology") != expected_runtime_topology:
        raise SystemExit("endpoint staged runtime topology drifted")
    if mapper_setup.get("runtime_topology") != expected_runtime_topology:
        raise SystemExit("endpoint mapper staged runtime topology drifted")
    if mapper_setup.get("source_model_live_during_timed_cells") is not False:
        raise SystemExit("endpoint mapper claims a live source model during cells")
    receipt.update(
        {
            "source_model_prefill_devices": expected_source_prefill_devices,
            "source_model_prefill_layer_counts": expected_source_prefill_counts,
            "source_cache_staging_policy": "untimed_cpu_staging_then_gpu_restage_v1",
            "runtime_topology": expected_runtime_topology,
            "staging_session_bindings": {},
        }
    )

receipt.update(
    {
        "request_fingerprint": run_definition["request_fingerprint"],
        "project_code_files_digest": run_definition["project_code"]["files_digest"],
        "runtime_artifacts_digest": digest(run_definition["runtime_artifacts"]),
        "run_definition_sha256": sha256_file(run_definition_path),
        "results_manifest_sha256": sha256_file(root / "results.manifest.json"),
        "preload_gpu_idle_gate_sha256": sha256_file(preload_gate_path),
        "cuda_visible_devices": os.environ["EXPECTED_CUDA_VISIBILITY"],
        "logical_device_order": expected_devices,
        "minimum_required_peak_reserved_headroom_bytes": minimum_headroom,
        "latency_protocol": "paper_main_only",
    }
)
if preload_gate.get("request_fingerprint") != run_definition.get("request_fingerprint"):
    raise SystemExit("pre-load GPU idle gate is not bound to the endpoint request")
if preload_gate.get("immediate_preload_gpu_idle_sandwich", {}).get("status") != "passed":
    raise SystemExit("endpoint lacks the immediate pre-load GPU idle sandwich")
if run_definition.get("resolved", {}).get("latency_protocol") != "paper_main_only":
    raise SystemExit("endpoint is not the paper-main-only latency protocol")
if run_definition.get("cli", {}).get("supplemental_diagnostics") is not False:
    raise SystemExit("endpoint unexpectedly enables supplemental diagnostics")
for length in (64, 32768):
    path = root / f"length_{length:05d}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    preflight = value["strict_transfer_preflight"]
    if preflight != {
        **preflight,
        "status": "passed",
        "untimed": True,
        "strict_validation": True,
        "rope_singularity_reduction": True,
        "source_content_finite_reduction": True,
        "mapped_kv_finite_reduction": True,
        "validated_target_layers": 64,
        "source_cache_length": length,
        "mapped_cache_length": length,
        "mapped_cache_dtypes": ["torch.bfloat16"],
    }:
        raise SystemExit(f"strict preflight policy failed for length {length}")
    expected_protocol = {
        "name": "paper_main_only",
        "paper_main_metrics": ["target_reprefill", "mapper_handoff_end_to_end"],
        "supplemental_diagnostics_enabled": False,
        "component_profile_in_cell": False,
        "boundary_decode_in_cell": False,
        "protocol_complete_in_cell": False,
        "paper_headline_speedup_uses_supplemental_metrics": False,
    }
    if value.get("latency_protocol") != expected_protocol:
        raise SystemExit(f"paper-main protocol contract failed for length {length}")
    forbidden = {
        "mapper_synchronized_component_profile",
        "mapper_kernel_only_profile",
        "boundary_one_token_decode",
        "protocol_complete",
        "median_speedup_target_reprefill_over_protocol_complete",
    }
    present = sorted(forbidden.intersection(value))
    if present:
        raise SystemExit(f"supplemental metrics contaminated length {length}: {present}")
    peak = value["cuda_peak_memory"]
    if peak.get("status") != "captured" or peak.get("selected_device_ids") != expected_devices:
        raise SystemExit(f"peak evidence coverage failed for length {length}")
    expected_peak_scope = (
        "entire_length_cell_from_preexisting_gpu_source_cache_through_final_allocator_cleanup"
        if staged_source_lifecycle
        else "entire_length_cell_from_source_cache_build_through_final_allocator_cleanup"
    )
    if peak.get("measurement_scope") != expected_peak_scope:
        raise SystemExit(f"peak scope failed for length {length}")
    headrooms = {
        device: int(item["capacity_headroom_at_peak_reserved_bytes"])
        for device, item in peak["devices"].items()
    }
    if min(headrooms.values()) < minimum_headroom:
        raise SystemExit(
            f"length {length} peak-reserved headroom below policy: {headrooms}"
        )
    for metric in ("target_reprefill", "mapper_handoff_end_to_end"):
        summary = value[metric]
        if summary.get("inference_mode_enabled") is not True:
            raise SystemExit(
                f"missing {metric} inference-mode evidence for length {length}"
            )
        if summary.get("autograd_graph_disabled") is not True:
            raise SystemExit(
                f"missing {metric} disabled-autograd evidence for length {length}"
            )
        for field in ("mean", "median", "p10", "p50", "p90", "min", "max"):
            number = float(summary[field])
            if not math.isfinite(number) or number < 0:
                raise SystemExit(f"invalid {metric}.{field} for length {length}")
    speedup = float(value["median_speedup_target_reprefill_over_mapper"])
    if not math.isfinite(speedup) or speedup <= 0:
        raise SystemExit(f"invalid speedup for length {length}")
    receipt["lengths"][str(length)] = {
        "checkpoint": path.name,
        "checkpoint_sha256": sha256_file(path),
        "minimum_peak_reserved_headroom_bytes": min(headrooms.values()),
        "per_device_peak_reserved_headroom_bytes": headrooms,
        "median_speedup_target_reprefill_over_mapper": speedup,
    }
    if staged_source_lifecycle:
        lifecycle = value.get("source_cache_lifecycle")
        if not isinstance(lifecycle, dict):
            raise SystemExit(f"missing staged lifecycle for length {length}")
        receipt["staging_session_bindings"][str(length)] = {
            "staging_session_id": lifecycle.get("staging_session_id"),
            "session_evidence_path": lifecycle.get("session_evidence_path"),
            "session_evidence_sha256": lifecycle.get("session_evidence_sha256"),
            "preparation_evidence_sha256": lifecycle.get("preparation_evidence_sha256"),
        }

temporary = root / ".endpoint_gate_receipt.json.tmp"
final = root / "endpoint_gate_receipt.json"
temporary.write_text(
    json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
temporary.replace(final)
sidecar_payload = {
    "schema_version": 1,
    "artifact": final.name,
    "sha256": sha256_file(final),
    "nbytes": final.stat().st_size,
}
sidecar_temporary = root / ".endpoint_gate_receipt.sha256.json.tmp"
sidecar_final = root / "endpoint_gate_receipt.sha256.json"
sidecar_temporary.write_text(
    json.dumps(sidecar_payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
sidecar_temporary.replace(sidecar_final)

# The independent report validator is the final semantic gate for the staged
# lifecycle, including every preparation/session/release/peak binding.
from kvcache_transfer.latency_report import _validate_endpoint_receipt, _validate_run
validated = _validate_run(root, lengths=(64, 32768), warmup=1, repeats=2)
_validate_endpoint_receipt(validated)
PY
}

verify_endpoint_receipt() {
  env \
    ENDPOINT_ROOT="${ENDPOINT_ROOT}" \
    EXPECTED_LATENCY_CODE_DIGEST="${EXPECTED_LATENCY_CODE_DIGEST}" \
    EXPECTED_CUDA_VISIBILITY="${CUDA_VISIBILITY}" \
    EXPECTED_SELECTED_DEVICES="${SELECTED_DEVICE_CSV}" \
    EXPECTED_SOURCE_DEVICES="${SOURCE_DEVICE_CSV}" \
    EXPECTED_TARGET_DEVICES="${TARGET_DEVICE_CSV}" \
    EXPECTED_SOURCE_LAYER_COUNTS="${SOURCE_LAYER_COUNT_CSV}" \
    EXPECTED_TARGET_LAYER_COUNTS="${TARGET_LAYER_COUNT_CSV}" \
    EXPECTED_STAGE_SOURCE_CACHES="${STAGE_SOURCE_CACHES}" \
    EXPECTED_SOURCE_PREFILL_DEVICES="${SOURCE_PREFILL_DEVICE_CSV}" \
    EXPECTED_SOURCE_PREFILL_LAYER_COUNTS="${SOURCE_PREFILL_LAYER_COUNT_CSV}" \
    EXPECTED_DEVICE_LAYOUT_PROFILE="${DEVICE_LAYOUT_PROFILE}" \
    PYTHONPATH="${PROJECT_ROOT}/src" \
    "${PYTHON}" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

root = Path(os.environ["ENDPOINT_ROOT"])
receipt_path = root / "endpoint_gate_receipt.json"
sidecar_path = root / "endpoint_gate_receipt.sha256.json"
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

def sha256_file(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

def digest(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

if sidecar != {
    "schema_version": 1,
    "artifact": receipt_path.name,
    "sha256": sha256_file(receipt_path),
    "nbytes": receipt_path.stat().st_size,
}:
    raise SystemExit("endpoint receipt sidecar mismatch")
exact_count_profiles = {
    "balanced_5target_3source_low_first_v4",
    "balanced_5target_3source_embedding_relief_v5",
    "balanced_5target_3source_single_first_v6",
    "balanced_5target_3source_equal_v7",
    "balanced_5target_3source_inference_low_first_v8",
    "staged_6target_2cache_v9",
}
exact_count_profile = os.environ["EXPECTED_DEVICE_LAYOUT_PROFILE"] in exact_count_profiles
staged_source_lifecycle = os.environ["EXPECTED_STAGE_SOURCE_CACHES"] == "1"
allowed_fields = {
    "schema_version",
    "status",
    "request_fingerprint",
    "project_code_files_digest",
    "runtime_artifacts_digest",
    "run_definition_sha256",
    "results_manifest_sha256",
    "preload_gpu_idle_gate_sha256",
    "cuda_visible_devices",
    "logical_device_order",
    "minimum_required_peak_reserved_headroom_bytes",
    "latency_protocol",
    "lengths",
}
if exact_count_profile:
    allowed_fields.update(
        {
            "device_layout_profile",
            "source_layer_counts",
            "target_layer_counts",
            "model_layer_sharding",
            "actual_mapper_target_layer_counts",
        }
    )
if staged_source_lifecycle:
    allowed_fields.update(
        {
            "source_model_prefill_devices",
            "source_model_prefill_layer_counts",
            "source_cache_staging_policy",
            "runtime_topology",
            "staging_session_bindings",
        }
    )
if not isinstance(receipt, dict) or set(receipt) != allowed_fields:
    raise SystemExit("endpoint receipt fields drifted")
if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
    raise SystemExit("endpoint receipt schema version drifted")
definition_path = root / "run_definition.json"
definition = json.loads(definition_path.read_text(encoding="utf-8"))
expected = {
    "status": "passed",
    "request_fingerprint": definition["request_fingerprint"],
    "project_code_files_digest": definition["project_code"]["files_digest"],
    "runtime_artifacts_digest": digest(definition["runtime_artifacts"]),
    "run_definition_sha256": sha256_file(definition_path),
    "results_manifest_sha256": sha256_file(root / "results.manifest.json"),
    "preload_gpu_idle_gate_sha256": sha256_file(root / "preload_gpu_idle_gate.json"),
    "latency_protocol": "paper_main_only",
    "cuda_visible_devices": os.environ["EXPECTED_CUDA_VISIBILITY"],
    "logical_device_order": [
        int(item) for item in os.environ["EXPECTED_SELECTED_DEVICES"].split(",")
    ],
    "minimum_required_peak_reserved_headroom_bytes": 1 << 30,
}
for field, value in expected.items():
    if digest(receipt.get(field)) != digest(value):
        raise SystemExit(f"endpoint gate receipt drifted field {field}")
if exact_count_profile:
    source_devices = [int(item) for item in os.environ["EXPECTED_SOURCE_DEVICES"].split(",")]
    target_devices = [int(item) for item in os.environ["EXPECTED_TARGET_DEVICES"].split(",")]
    source_counts = [int(item) for item in os.environ["EXPECTED_SOURCE_LAYER_COUNTS"].split(",")]
    target_counts = [int(item) for item in os.environ["EXPECTED_TARGET_LAYER_COUNTS"].split(",")]
    if receipt.get("device_layout_profile") != os.environ["EXPECTED_DEVICE_LAYOUT_PROFILE"]:
        raise SystemExit("endpoint receipt exact-count profile drifted")
    if receipt.get("source_layer_counts") != source_counts:
        raise SystemExit("endpoint receipt source layer counts drifted")
    if receipt.get("target_layer_counts") != target_counts:
        raise SystemExit("endpoint receipt target layer counts drifted")
    if receipt.get("actual_mapper_target_layer_counts") != target_counts:
        raise SystemExit("endpoint receipt mapper target distribution drifted")
    mapper_setup = json.loads((root / "mapper_setup.json").read_text(encoding="utf-8"))
    if receipt.get("model_layer_sharding") != mapper_setup.get("model_layer_sharding"):
        raise SystemExit("endpoint receipt runtime layer-sharding evidence drifted")
    runtime_sharding = receipt["model_layer_sharding"]
    runtime_roles = (
        (("target", target_devices, target_counts, 64),)
        if staged_source_lifecycle
        else (
            ("source", source_devices, source_counts, 40),
            ("target", target_devices, target_counts, 64),
        )
    )
    for role, devices, counts, total in runtime_roles:
        ranges = []
        start = 0
        for device, count in zip(devices, counts, strict=True):
            ranges.append(
                {
                    "device": device,
                    "start_layer": start,
                    "end_layer_exclusive": start + count,
                    "count": count,
                }
            )
            start += count
        expected = {
            "status": "passed",
            "role": role,
            "mode": "explicit_exact_counts",
            "devices": devices,
            "declared_layer_counts": counts,
            "expected_layer_counts": counts,
            "actual_layer_counts": counts,
            "total_layers": total,
            "layer_devices": [
                device
                for device, count in zip(devices, counts, strict=True)
                for _ in range(count)
            ],
            "contiguous_ranges": ranges,
            "actual_parameters_inspected_for_every_layer": True,
            "auxiliary_parameters_inspected_for_every_module": True,
        }
        item = runtime_sharding.get(role)
        if not isinstance(item, dict) or set(item) != {
            *expected,
            "parameter_tensors_inspected",
            "auxiliary_modules",
            "auxiliary_parameter_tensors_inspected",
        }:
            raise SystemExit(f"endpoint receipt runtime {role} fields drifted")
        if any(
            digest(item.get(field)) != digest(value)
            for field, value in expected.items()
        ):
            raise SystemExit(f"endpoint receipt runtime {role} split drifted")
        inspected = item.get("parameter_tensors_inspected")
        if type(inspected) is not int or inspected != total * 11:
            raise SystemExit(f"endpoint receipt runtime {role} inspection total drifted")
        expected_aux_devices = {
            "model.embed_tokens": devices[0],
            "model.norm": devices[-1],
            "lm_head": devices[-1],
        }
        auxiliary = item.get("auxiliary_modules")
        if not isinstance(auxiliary, dict) or set(auxiliary) != set(expected_aux_devices):
            raise SystemExit(f"endpoint receipt runtime {role} auxiliary coverage drifted")
        auxiliary_total = 0
        for name, expected_device in expected_aux_devices.items():
            module = auxiliary[name]
            if not isinstance(module, dict) or set(module) != {
                "expected_device", "actual_device", "parameter_tensors_inspected"
            }:
                raise SystemExit(f"endpoint receipt runtime {role} auxiliary fields drifted")
            if module.get("expected_device") != expected_device or module.get("actual_device") != expected_device:
                raise SystemExit(f"endpoint receipt runtime {role} auxiliary device drifted")
            count = module.get("parameter_tensors_inspected")
            if type(count) is not int or count < 1:
                raise SystemExit(f"endpoint receipt runtime {role} auxiliary inspection drifted")
            auxiliary_total += count
        if auxiliary_total != 3 or item.get("auxiliary_parameter_tensors_inspected") != 3:
            raise SystemExit(f"endpoint receipt runtime {role} auxiliary total drifted")
    expected_mapper_devices = [
        device
        for device, count in zip(target_devices, target_counts, strict=True)
        for _ in range(count)
    ]
    per_layer = mapper_setup.get("per_layer")
    if not isinstance(per_layer, list) or len(per_layer) != 64:
        raise SystemExit("endpoint receipt mapper layer coverage drifted")
    for layer, (item, expected_device) in enumerate(zip(per_layer, expected_mapper_devices, strict=True)):
        if item.get("target_layer") != layer or item.get("device") != f"cuda:{expected_device}":
            raise SystemExit("endpoint receipt mapper target placement drifted")
if staged_source_lifecycle:
    source_prefill_devices = [
        int(item) for item in os.environ["EXPECTED_SOURCE_PREFILL_DEVICES"].split(",")
    ]
    source_prefill_counts = [
        int(item)
        for item in os.environ["EXPECTED_SOURCE_PREFILL_LAYER_COUNTS"].split(",")
    ]
    mapper_setup = json.loads((root / "mapper_setup.json").read_text(encoding="utf-8"))
    expected_staged = {
        "source_model_prefill_devices": source_prefill_devices,
        "source_model_prefill_layer_counts": source_prefill_counts,
        "source_cache_staging_policy": "untimed_cpu_staging_then_gpu_restage_v1",
        "runtime_topology": mapper_setup.get("runtime_topology"),
    }
    for field, wanted in expected_staged.items():
        if digest(receipt.get(field)) != digest(wanted):
            raise SystemExit(f"endpoint receipt staged field {field} drifted")
if receipt.get("project_code_files_digest") != os.environ["EXPECTED_LATENCY_CODE_DIGEST"]:
    raise SystemExit("endpoint code digest is not the frozen launcher digest")
lengths = receipt.get("lengths")
if not isinstance(lengths, dict) or set(lengths) != {"64", "32768"}:
    raise SystemExit("endpoint receipt length set drifted")
expected_devices = [
    int(item) for item in os.environ["EXPECTED_SELECTED_DEVICES"].split(",")
]
expected_device_keys = {str(device) for device in expected_devices}
length_fields = {
    "checkpoint",
    "checkpoint_sha256",
    "minimum_peak_reserved_headroom_bytes",
    "per_device_peak_reserved_headroom_bytes",
    "median_speedup_target_reprefill_over_mapper",
}
for length in (64, 32768):
    checkpoint = root / f"length_{length:05d}.json"
    item = lengths[str(length)]
    if not isinstance(item, dict) or set(item) != length_fields:
        raise SystemExit(f"endpoint receipt length {length} fields drifted")
    if item.get("checkpoint") != checkpoint.name:
        raise SystemExit(f"endpoint receipt checkpoint {length} name drifted")
    current_sha256 = sha256_file(checkpoint)
    if item.get("checkpoint_sha256") != current_sha256:
        raise SystemExit(f"endpoint checkpoint {length} drifted after gate")
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("status") != "complete"
        or value.get("prefix_length") != length
        or value.get("request_fingerprint") != definition["request_fingerprint"]
    ):
        raise SystemExit(f"endpoint checkpoint {length} identity drifted")
    preflight = value.get("strict_transfer_preflight")
    expected_preflight = {
        "status": "passed",
        "untimed": True,
        "strict_validation": True,
        "rope_singularity_reduction": True,
        "source_content_finite_reduction": True,
        "mapped_kv_finite_reduction": True,
        "validated_target_layers": 64,
        "source_cache_length": length,
        "mapped_cache_length": length,
        "mapped_cache_dtypes": ["torch.bfloat16"],
    }
    if not isinstance(preflight, dict) or any(
        digest(preflight.get(field)) != digest(expected)
        for field, expected in expected_preflight.items()
    ):
        raise SystemExit(f"endpoint checkpoint {length} strict preflight drifted")
    preflight_wall_ms = preflight.get("wall_ms_diagnostic_not_latency")
    if (
        type(preflight_wall_ms) not in (int, float)
        or not math.isfinite(float(preflight_wall_ms))
        or float(preflight_wall_ms) < 0.0
    ):
        raise SystemExit(
            f"endpoint checkpoint {length} strict preflight timing drifted"
        )
    peak = value.get("cuda_peak_memory")
    if (
        not isinstance(peak, dict)
        or peak.get("status") != "captured"
        or peak.get("selected_device_ids") != expected_devices
    ):
        raise SystemExit(f"endpoint checkpoint {length} peak coverage drifted")
    expected_peak_scope = (
        "entire_length_cell_from_preexisting_gpu_source_cache_through_final_allocator_cleanup"
        if staged_source_lifecycle
        else "entire_length_cell_from_source_cache_build_through_final_allocator_cleanup"
    )
    if peak.get("measurement_scope") != expected_peak_scope:
        raise SystemExit(f"endpoint checkpoint {length} peak scope drifted")
    peak_devices = peak.get("devices")
    if not isinstance(peak_devices, dict) or set(peak_devices) != expected_device_keys:
        raise SystemExit(f"endpoint checkpoint {length} peak devices drifted")
    headrooms = {}
    for device in expected_device_keys:
        evidence = peak_devices[device]
        if not isinstance(evidence, dict):
            raise SystemExit(f"endpoint checkpoint {length} peak evidence drifted")
        capacity = evidence.get("capacity_bytes")
        peak_reserved = evidence.get("peak_reserved_bytes")
        headroom = evidence.get("capacity_headroom_at_peak_reserved_bytes")
        if (
            type(capacity) is not int
            or type(peak_reserved) is not int
            or type(headroom) is not int
            or capacity < 0
            or peak_reserved < 0
            or peak_reserved > capacity
            or headroom != capacity - peak_reserved
        ):
            raise SystemExit(
                f"endpoint checkpoint {length} peak headroom arithmetic drifted"
            )
        if headroom < (1 << 30):
            raise SystemExit(f"endpoint checkpoint {length} headroom policy failed")
        headrooms[device] = headroom
    minimum_headroom = min(headrooms.values())
    if (
        item.get("minimum_peak_reserved_headroom_bytes") != minimum_headroom
        or digest(item.get("per_device_peak_reserved_headroom_bytes"))
        != digest(headrooms)
    ):
        raise SystemExit(f"endpoint receipt length {length} headroom drifted")
    target = value.get("target_reprefill")
    mapper = value.get("mapper_handoff_end_to_end")
    if not isinstance(target, dict) or not isinstance(mapper, dict):
        raise SystemExit(f"endpoint checkpoint {length} latency summaries drifted")
    for metric, summary in (
        ("target_reprefill", target),
        ("mapper_handoff_end_to_end", mapper),
    ):
        if summary.get("inference_mode_enabled") is not True:
            raise SystemExit(
                f"endpoint checkpoint {length} {metric} inference-mode evidence drifted"
            )
        if summary.get("autograd_graph_disabled") is not True:
            raise SystemExit(
                f"endpoint checkpoint {length} {metric} autograd evidence drifted"
            )
    target_median = target.get("median")
    mapper_median = mapper.get("median")
    if (
        type(target_median) not in (int, float)
        or type(mapper_median) not in (int, float)
        or not math.isfinite(float(target_median))
        or not math.isfinite(float(mapper_median))
        or float(target_median) <= 0.0
        or float(mapper_median) <= 0.0
    ):
        raise SystemExit(f"endpoint checkpoint {length} latency medians drifted")
    recorded_speedup = item.get("median_speedup_target_reprefill_over_mapper")
    checkpoint_speedup = value.get("median_speedup_target_reprefill_over_mapper")
    expected_speedup = float(target_median) / float(mapper_median)
    if (
        type(checkpoint_speedup) not in (int, float)
        or not math.isfinite(float(checkpoint_speedup))
        or float(checkpoint_speedup) <= 0.0
        or not math.isclose(
            float(checkpoint_speedup), expected_speedup, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise SystemExit(f"endpoint checkpoint {length} speedup drifted")
    if (
        type(recorded_speedup) not in (int, float)
        or not math.isfinite(float(recorded_speedup))
        or float(recorded_speedup) <= 0.0
        or not math.isclose(
            float(recorded_speedup),
            float(checkpoint_speedup),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise SystemExit(f"endpoint receipt length {length} speedup drifted")

from kvcache_transfer.latency_report import _validate_endpoint_receipt, _validate_run
validated = _validate_run(root, lengths=(64, 32768), warmup=1, repeats=2)
_validate_endpoint_receipt(validated)
PY
}

verify_formal_identity_against_endpoint() {
  env \
    ENDPOINT_ROOT="${ENDPOINT_ROOT}" \
    FORMAL_ROOT="${FORMAL_ROOT}" \
    EXPECTED_SOURCE_DEVICES="${SOURCE_DEVICE_CSV}" \
    EXPECTED_TARGET_DEVICES="${TARGET_DEVICE_CSV}" \
    EXPECTED_SELECTED_DEVICES="${SELECTED_DEVICE_CSV}" \
    EXPECTED_CUDA_VISIBILITY="${CUDA_VISIBILITY}" \
    EXPECTED_SOURCE_LAYER_COUNTS="${SOURCE_LAYER_COUNT_CSV}" \
    EXPECTED_TARGET_LAYER_COUNTS="${TARGET_LAYER_COUNT_CSV}" \
    EXPECTED_STAGE_SOURCE_CACHES="${STAGE_SOURCE_CACHES}" \
    EXPECTED_SOURCE_PREFILL_DEVICES="${SOURCE_PREFILL_DEVICE_CSV}" \
    EXPECTED_SOURCE_PREFILL_LAYER_COUNTS="${SOURCE_PREFILL_LAYER_COUNT_CSV}" \
    EXPECTED_DEVICE_LAYOUT_PROFILE="${DEVICE_LAYOUT_PROFILE}" \
    "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

endpoint_root = Path(os.environ["ENDPOINT_ROOT"])
formal_root = Path(os.environ["FORMAL_ROOT"])
receipt = json.loads((endpoint_root / "endpoint_gate_receipt.json").read_text(encoding="utf-8"))
formal = json.loads((formal_root / "run_definition.json").read_text(encoding="utf-8"))
formal_config = json.loads((formal_root / "config.json").read_text(encoding="utf-8"))
formal_mapper = json.loads((formal_root / "mapper_setup.json").read_text(encoding="utf-8"))

def digest(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

if digest(formal["runtime_artifacts"]) != receipt["runtime_artifacts_digest"]:
    raise SystemExit("formal model/mapper identity differs from the endpoint gate")
if formal["project_code"]["files_digest"] != receipt["project_code_files_digest"]:
    raise SystemExit("formal code identity differs from the endpoint gate")
if formal.get("resolved", {}).get("latency_protocol") != "paper_main_only":
    raise SystemExit("formal run is not paper-main-only")
if formal.get("cli", {}).get("supplemental_diagnostics") is not False:
    raise SystemExit("formal run unexpectedly enables supplemental diagnostics")
expected_source = [int(item) for item in os.environ["EXPECTED_SOURCE_DEVICES"].split(",")]
expected_target = [int(item) for item in os.environ["EXPECTED_TARGET_DEVICES"].split(",")]
expected_selected = [int(item) for item in os.environ["EXPECTED_SELECTED_DEVICES"].split(",")]
for definition, label in ((formal, "formal"),):
    if definition.get("cli", {}).get("source_devices") != expected_source:
        raise SystemExit(f"{label} source-device layout mismatch")
    if definition.get("cli", {}).get("target_devices") != expected_target:
        raise SystemExit(f"{label} target-device layout mismatch")
    topology = definition.get("topology", {})
    if topology.get("selected_devices") != expected_selected:
        raise SystemExit(f"{label} selected-device order mismatch")
    if topology.get("cuda_visible_devices_environment") != os.environ["EXPECTED_CUDA_VISIBILITY"]:
        raise SystemExit(f"{label} CUDA visibility mismatch")
if receipt.get("logical_device_order") != expected_selected:
    raise SystemExit("endpoint receipt selected-device order mismatch")
if receipt.get("cuda_visible_devices") != os.environ["EXPECTED_CUDA_VISIBILITY"]:
    raise SystemExit("endpoint receipt CUDA visibility mismatch")
if os.environ["EXPECTED_DEVICE_LAYOUT_PROFILE"] in {
    "balanced_5target_3source_low_first_v4",
    "balanced_5target_3source_embedding_relief_v5",
    "balanced_5target_3source_single_first_v6",
    "balanced_5target_3source_equal_v7",
    "balanced_5target_3source_inference_low_first_v8",
    "staged_6target_2cache_v9",
}:
    source_counts = [int(item) for item in os.environ["EXPECTED_SOURCE_LAYER_COUNTS"].split(",")]
    target_counts = [int(item) for item in os.environ["EXPECTED_TARGET_LAYER_COUNTS"].split(",")]
    sharding = {
        "source": {
            "mode": "explicit_exact_counts",
            "devices": expected_source,
            "declared_layer_counts": source_counts,
            "resolved_layer_counts": source_counts,
            "total_layers": 40,
        },
        "target": {
            "mode": "explicit_exact_counts",
            "devices": expected_target,
            "declared_layer_counts": target_counts,
            "resolved_layer_counts": target_counts,
            "total_layers": 64,
        },
    }
    if os.environ["EXPECTED_STAGE_SOURCE_CACHES"] == "1":
        prefill_devices = [
            int(item) for item in os.environ["EXPECTED_SOURCE_PREFILL_DEVICES"].split(",")
        ]
        prefill_counts = [
            int(item)
            for item in os.environ["EXPECTED_SOURCE_PREFILL_LAYER_COUNTS"].split(",")
        ]
        sharding["source_model_prefill"] = {
            "mode": "explicit_exact_counts",
            "devices": prefill_devices,
            "declared_layer_counts": prefill_counts,
            "resolved_layer_counts": prefill_counts,
            "total_layers": 40,
        }
        sharding["source_cache_staging"] = dict(sharding["source"])
        if formal.get("cli", {}).get("stage_source_caches_on_cpu_before_cells") is not True:
            raise SystemExit("formal staged source-cache flag is missing")
        if formal.get("cli", {}).get("source_prefill_devices") != prefill_devices:
            raise SystemExit("formal source-prefill devices mismatch")
        if formal.get("cli", {}).get("source_prefill_layer_counts") != prefill_counts:
            raise SystemExit("formal source-prefill layer counts mismatch")
        if formal_config.get("stage_source_caches_on_cpu_before_cells") is not True:
            raise SystemExit("formal config staged flag mismatch")
        if formal_config.get("source_prefill_devices") != prefill_devices:
            raise SystemExit("formal config source-prefill devices mismatch")
        if formal_config.get("source_prefill_layer_counts") != prefill_counts:
            raise SystemExit("formal config source-prefill counts mismatch")
        if formal_config.get("staged_runtime_topology") != receipt.get("runtime_topology"):
            raise SystemExit("formal/endpoint staged runtime topology differs")
    if formal.get("cli", {}).get("source_layer_counts") != source_counts:
        raise SystemExit("formal source layer-count declaration mismatch")
    if formal.get("cli", {}).get("target_layer_counts") != target_counts:
        raise SystemExit("formal target layer-count declaration mismatch")
    if formal.get("resolved", {}).get("layer_sharding") != sharding:
        raise SystemExit("formal fingerprint-bound layer-sharding mismatch")
    if formal_config.get("source_layer_counts") != source_counts:
        raise SystemExit("formal config source layer-count mismatch")
    if formal_config.get("target_layer_counts") != target_counts:
        raise SystemExit("formal config target layer-count mismatch")
    if formal_config.get("layer_sharding") != sharding:
        raise SystemExit("formal config layer-sharding mismatch")
    if receipt.get("device_layout_profile") != os.environ["EXPECTED_DEVICE_LAYOUT_PROFILE"]:
        raise SystemExit("endpoint receipt lacks the exact-count profile")
    if receipt.get("source_layer_counts") != source_counts:
        raise SystemExit("endpoint/formal source layer counts differ")
    if receipt.get("target_layer_counts") != target_counts:
        raise SystemExit("endpoint/formal target layer counts differ")
    if formal_mapper.get("model_layer_sharding") != receipt.get("model_layer_sharding"):
        raise SystemExit("endpoint/formal actual model layer sharding differs")
    mapper_counts = {device: 0 for device in expected_target}
    mapper_devices = []
    for layer, item in enumerate(formal_mapper.get("per_layer", [])):
        if item.get("target_layer") != layer:
            raise SystemExit("formal mapper target layers are incomplete/reordered")
        device_text = str(item.get("device", ""))
        if not device_text.startswith("cuda:"):
            raise SystemExit("formal mapper target device is malformed")
        device = int(device_text.removeprefix("cuda:"))
        if device not in mapper_counts:
            raise SystemExit("formal mapper target device is unexpected")
        mapper_counts[device] += 1
        mapper_devices.append(device)
    actual_counts = [mapper_counts[device] for device in expected_target]
    expected_devices_by_layer = [
        device
        for device, count in zip(expected_target, target_counts, strict=True)
        for _ in range(count)
    ]
    if actual_counts != target_counts or mapper_devices != expected_devices_by_layer:
        raise SystemExit("formal actual mapper target distribution mismatch")
    if receipt.get("actual_mapper_target_layer_counts") != actual_counts:
        raise SystemExit("endpoint/formal mapper target counts differ")
PY
}

run_endpoint() {
  write_state "running_endpoint" "${ENDPOINT_ROOT}"
  verify_frozen_code
  verify_host_capacity
  [[ ! -e "${ENDPOINT_ROOT}" ]] || {
    log_wait "ABORT endpoint directory already exists: ${ENDPOINT_ROOT}"
    return 1
  }
  single_idle_sample || {
    log_wait "ABORT endpoint lost the clean GPU window immediately before launch"
    return 1
  }
  run_logged "${ENDPOINT_LOG}" "${ENDPOINT_ARGS[@]}"
  reject_fatal_log_markers "${ENDPOINT_LOG}"
  require_complete_marker "${ENDPOINT_ROOT}"
  # A second invocation is allowed only after proving the run is complete; it
  # exercises verify_completed_run and must return reused_complete rather than
  # resuming any partial cell.
  run_logged "${ENDPOINT_VERIFY_LOG}" "${ENDPOINT_ARGS[@]}"
  grep -q '"status": "reused_complete"' "${ENDPOINT_VERIFY_LOG}"
  verify_endpoint_policy
  verify_endpoint_receipt
  write_state "endpoint_passed" "${ENDPOINT_ROOT}/endpoint_gate_receipt.json"
}

run_formal() {
  write_state "running_formal" "${FORMAL_ROOT}"
  verify_frozen_code
  verify_host_capacity
  require_complete_marker "${ENDPOINT_ROOT}"
  run_logged "${ENDPOINT_PREFORMAL_VERIFY_LOG}" "${ENDPOINT_ARGS[@]}"
  grep -q '"status": "reused_complete"' "${ENDPOINT_PREFORMAL_VERIFY_LOG}"
  verify_endpoint_receipt
  [[ ! -e "${FORMAL_ROOT}" ]] || {
    log_wait "ABORT formal directory already exists: ${FORMAL_ROOT}"
    return 1
  }
  single_idle_sample || {
    log_wait "ABORT formal run lost the clean GPU window immediately before launch"
    return 1
  }
  run_logged "${FORMAL_LOG}" "${FORMAL_ARGS[@]}"
  reject_fatal_log_markers "${FORMAL_LOG}"
  require_complete_marker "${FORMAL_ROOT}"
  run_logged "${FORMAL_VERIFY_LOG}" "${FORMAL_ARGS[@]}"
  grep -q '"status": "reused_complete"' "${FORMAL_VERIFY_LOG}"
  # Re-run the frozen endpoint's complete-artifact validator after the long
  # formal run.  The reused_complete path does not load models or run a GPU
  # cell, and closes mutation gaps without duplicating that validator here.
  run_logged "${ENDPOINT_POSTFORMAL_VERIFY_LOG}" "${ENDPOINT_ARGS[@]}"
  grep -q '"status": "reused_complete"' "${ENDPOINT_POSTFORMAL_VERIFY_LOG}"
  verify_endpoint_receipt
  verify_formal_identity_against_endpoint
  write_state "formal_passed" "${FORMAL_ROOT}"
}

verify_report_outputs_against_inputs() {
  # Rebuilding into the same directory is intentionally idempotent: the
  # frozen report builder revalidates both input runs and refuses any byte
  # drift across JSON, CSV, Markdown, or their manifest.
  run_logged \
    "${REPORT_VERIFY_LOG}" \
    "${PYTHON}" \
    "scripts/summarize_latency.py" \
    "${ENDPOINT_ROOT}" \
    "${FORMAL_ROOT}" \
    "--output-dir" \
    "${REPORT_ROOT}"
  grep -q '"status": "complete"' "${REPORT_VERIFY_LOG}"
}

run_report() {
  write_state "running_report" "${REPORT_ROOT}"
  verify_frozen_code
  verify_endpoint_receipt
  [[ ! -e "${REPORT_ROOT}" ]] || {
    log_wait "ABORT report directory already exists: ${REPORT_ROOT}"
    return 1
  }
  run_logged \
    "${REPORT_LOG}" \
    "${PYTHON}" \
    "scripts/summarize_latency.py" \
    "${ENDPOINT_ROOT}" \
    "${FORMAL_ROOT}" \
    "--output-dir" \
    "${REPORT_ROOT}"
  env \
    REPORT_ROOT="${REPORT_ROOT}" \
    EXPECTED_SOURCE_DEVICES="${SOURCE_DEVICE_CSV}" \
    EXPECTED_TARGET_DEVICES="${TARGET_DEVICE_CSV}" \
    EXPECTED_SELECTED_DEVICES="${SELECTED_DEVICE_CSV}" \
    EXPECTED_CUDA_VISIBILITY="${CUDA_VISIBILITY}" \
    EXPECTED_SOURCE_LAYER_COUNTS="${SOURCE_LAYER_COUNT_CSV}" \
    EXPECTED_TARGET_LAYER_COUNTS="${TARGET_LAYER_COUNT_CSV}" \
    EXPECTED_STAGE_SOURCE_CACHES="${STAGE_SOURCE_CACHES}" \
    EXPECTED_SOURCE_PREFILL_DEVICES="${SOURCE_PREFILL_DEVICE_CSV}" \
    EXPECTED_SOURCE_PREFILL_LAYER_COUNTS="${SOURCE_PREFILL_LAYER_COUNT_CSV}" \
    EXPECTED_DEVICE_LAYOUT_PROFILE="${DEVICE_LAYOUT_PROFILE}" \
    "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["REPORT_ROOT"])
manifest = json.loads((root / "sha256_manifest.json").read_text(encoding="utf-8"))
report = json.loads((root / "latency_report.json").read_text(encoding="utf-8"))

def sha256_file(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()

if report.get("status") != "complete":
    raise SystemExit("formal latency report is not complete")
protocol = report.get("protocol", {})
expected_source = [int(item) for item in os.environ["EXPECTED_SOURCE_DEVICES"].split(",")]
expected_target = [int(item) for item in os.environ["EXPECTED_TARGET_DEVICES"].split(",")]
expected_selected = [int(item) for item in os.environ["EXPECTED_SELECTED_DEVICES"].split(",")]
if protocol.get("source_devices") != expected_source:
    raise SystemExit("report source-device layout mismatch")
if protocol.get("target_devices") != expected_target:
    raise SystemExit("report target-device layout mismatch")
if protocol.get("selected_device_order") != expected_selected:
    raise SystemExit("report selected-device order mismatch")
if protocol.get("cuda_visible_devices_environment") != os.environ["EXPECTED_CUDA_VISIBILITY"]:
    raise SystemExit("report CUDA visibility mismatch")
if protocol.get("device_layout_profile") != os.environ["EXPECTED_DEVICE_LAYOUT_PROFILE"]:
    raise SystemExit("report device-layout profile mismatch")
if protocol.get("timed_main_metrics_inference_mode_enabled") is not True:
    raise SystemExit("report inference-mode evidence mismatch")
if protocol.get("timed_main_metrics_autograd_graph_disabled") is not True:
    raise SystemExit("report disabled-autograd evidence mismatch")
if os.environ["EXPECTED_DEVICE_LAYOUT_PROFILE"] in {
    "balanced_5target_3source_low_first_v4",
    "balanced_5target_3source_embedding_relief_v5",
    "balanced_5target_3source_single_first_v6",
    "balanced_5target_3source_equal_v7",
    "balanced_5target_3source_inference_low_first_v8",
    "staged_6target_2cache_v9",
}:
    source_counts = [int(item) for item in os.environ["EXPECTED_SOURCE_LAYER_COUNTS"].split(",")]
    target_counts = [int(item) for item in os.environ["EXPECTED_TARGET_LAYER_COUNTS"].split(",")]
    if protocol.get("source_layer_counts") != source_counts:
        raise SystemExit("exact-count report source layer counts mismatch")
    if protocol.get("target_layer_counts") != target_counts:
        raise SystemExit("exact-count report target layer counts mismatch")
    if protocol.get("actual_mapper_target_layer_counts") != target_counts:
        raise SystemExit("exact-count report mapper target distribution mismatch")
if os.environ["EXPECTED_STAGE_SOURCE_CACHES"] == "1":
    prefill_devices = [
        int(item) for item in os.environ["EXPECTED_SOURCE_PREFILL_DEVICES"].split(",")
    ]
    prefill_counts = [
        int(item)
        for item in os.environ["EXPECTED_SOURCE_PREFILL_LAYER_COUNTS"].split(",")
    ]
    staged_expected = {
        "source_model_prefill_devices": prefill_devices,
        "source_model_prefill_layer_counts": prefill_counts,
        "source_cache_staging_policy": "untimed_cpu_staging_then_gpu_restage_v1",
        "source_model_live_during_timed_cells": False,
        "cpu_staging_and_gpu_restage_included_in_mapper_timing": False,
        "mapper_timing_starts_from_preexisting_gpu_source_cache": True,
        "author_model_residency_and_sharding_published": False,
        "local_staged_sharding_is_4090_capacity_configuration": True,
    }
    for field, wanted in staged_expected.items():
        if protocol.get(field) != wanted:
            raise SystemExit(f"staged report protocol {field} mismatch")
files = manifest.get("files", {})
if set(files) != {"latency_report.json", "latency_table.csv", "latency_table.md"}:
    raise SystemExit("report manifest file set mismatch")
for name, identity in files.items():
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"report artifact missing or symlinked: {name}")
    if identity != {"sha256": sha256_file(path), "nbytes": path.stat().st_size}:
        raise SystemExit(f"report artifact identity mismatch: {name}")
PY
  verify_report_outputs_against_inputs
  write_state "complete" "${REPORT_ROOT}"
}

on_exit() {
  local status="$?"
  if (( status != 0 )); then
    local failed_stage="${CURRENT_STAGE}"
    write_state "failed" "stage=${failed_stage};exit_status=${status}"
    log_wait "Latency queue failed closed with exit status ${status}."
  fi
}

main() {
  cd "${PROJECT_ROOT}"
  mkdir -p "${LOG_ROOT}"
  exec 9>"${LOCK_FILE}"
  flock -n 9 || {
    printf 'another NVIDIA KV-cache latency queue holds %s\n' "${LOCK_FILE}" >&2
    return 1
  }
  verify_frozen_code
  verify_layout_contract
  verify_host_capacity
  if [[ "${LATENCY_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    printf 'latency launcher preflight passed: source=%s target=%s CUDA_VISIBLE_DEVICES=%s\n' \
      "${SOURCE_DEVICE_CSV}" "${TARGET_DEVICE_CSV}" "${CUDA_VISIBILITY}"
    return 0
  fi
  [[ ! -e "${QUEUE_ROOT}" ]] || {
    printf 'queue directory already exists: %s\n' "${QUEUE_ROOT}" >&2
    return 1
  }
  mkdir "${QUEUE_ROOT}"
  trap on_exit EXIT
  trap 'exit 130' HUP INT TERM
  write_state "created" "pid=$$;ppid=${PPID};tmux=${TMUX:-none};fail_closed=true"
  wait_for_clean_gpu_window "endpoint_window"
  run_endpoint
  wait_for_clean_gpu_window "formal_window"
  run_formal
  run_report
  log_wait "Latency endpoint and formal sweep completed successfully."
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
