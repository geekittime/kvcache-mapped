#!/usr/bin/env bash
set -Eeuo pipefail
set -o noclobber

PROJECT_ROOT=/home/wby/llm-cache/nvidia-kvcache
HOT_ROOT="$PROJECT_ROOT/runs/20260820_qwen3_14b_to_32b_hotpotqa_cap512_full_v1"
DIALOGUE_ROOT="$PROJECT_ROOT/runs/20260820_qwen3_14b_to_32b_dialogue_bottom1000_full_v1"
DIALOGUE_LOG="$PROJECT_ROOT/runs/logs/20260820_qwen3_14b_to_32b_dialogue_bottom1000_full_v1.log"

cd "$PROJECT_ROOT"
printf '%s waiting_for_hotpotqa_complete\n' "$(date -u +%FT%TZ)"

while true; do
  if [[ -f "$HOT_ROOT/status.json" ]]; then
    HOT_STATE=$(.venv/bin/python - "$HOT_ROOT/status.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(value.get("status", "missing"))
PY
)
    if [[ "$HOT_STATE" == complete ]]; then
      break
    fi
    if [[ "$HOT_STATE" == failed || "$HOT_STATE" == incomplete ]]; then
      printf '%s hotpotqa_failed_closed state=%s\n' "$(date -u +%FT%TZ)" "$HOT_STATE" >&2
      exit 1
    fi
  fi
  sleep 30
done

.venv/bin/python - "$HOT_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_fingerprint = "3b1d4bc30eb128792427f38a28009b23bf08fbcef704c7f6ecc218596ed230b9"

def load(name: str):
    return json.loads((root / name).read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

status = load("status.json")
summary = load("summary.json")
manifest = load("results.manifest.json")
assert status.get("status") == "complete" and status.get("failures") == 0
assert status.get("request_fingerprint") == expected_fingerprint
assert summary.get("status") == "complete"
assert summary.get("request_fingerprint") == expected_fingerprint
assert manifest.get("status") == "complete"
assert manifest.get("request_fingerprint") == expected_fingerprint
assert manifest.get("expected_result_count") == 14810
files = manifest.get("files")
assert isinstance(files, list) and len(files) == 14810
assert len({item["path"] for item in files}) == 14810
for item in files:
    path = root / item["path"]
    assert path.is_file()
    assert path.stat().st_size == item["nbytes"]
    assert sha256(path) == item["sha256"]
task = summary["tasks"]["hotpotqa"]
assert task.get("expected_samples") == 7405
for mode in ("target_standalone", "mapped"):
    observed = task["modes"][mode]
    assert observed.get("complete") is True
    assert observed.get("completed_samples") == 7405
manifest_identity = status["results_manifest"]
manifest_path = root / manifest_identity["path"]
assert manifest_path.stat().st_size == manifest_identity["nbytes"]
assert sha256(manifest_path) == manifest_identity["sha256"]
assert (root / "summary.json").stat().st_size == status["summary_nbytes"]
assert sha256(root / "summary.json") == status["summary_sha256"]
print("hotpotqa_complete_receipt_passed")
PY

printf '%s waiting_for_8gpu_idle\n' "$(date -u +%FT%TZ)"
IDLE_SAMPLES=0
while (( IDLE_SAMPLES < 12 )); do
  COMPUTE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
  if [[ -z "$COMPUTE_PIDS" && "$MAX_USED" =~ ^[0-9]+$ && "$MAX_USED" -le 64 ]]; then
    IDLE_SAMPLES=$((IDLE_SAMPLES + 1))
  else
    IDLE_SAMPLES=0
  fi
  sleep 5
done

printf '%s  %s\n' \
  d6f03eb454a787ed904be390377881d153ed87bd2c60d81c7a69346a6ff5b698 scripts/evaluate_long_context_dialogue_1000.py \
  b62870addf16f1f73bc6f50e8516caae28ef974fd580646568ec4eec38becf4e configs/eval_qwen3_14b_to_32b_dialogue_1000.yaml \
  9fb29dab32d7bc4cab5fe4f1f3e7a53fdd669f4efb43f0081da88a71f746154b scripts/evaluate_long_context.py \
  5e6a2ea48b7ac8b7c3a7b3673fc2544bbc47880ba9a32f2e7981604128595fd7 src/kvcache_transfer/evaluation.py \
  | sha256sum -c -

test ! -e "$DIALOGUE_ROOT"
test ! -e "$DIALOGUE_LOG"
: > "$DIALOGUE_LOG"
printf '%s starting_dialogue_bottom1000\n' "$(date -u +%FT%TZ)"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/evaluate_long_context_dialogue_1000.py \
  --output-dir "$DIALOGUE_ROOT" 2>&1 | tee -a "$DIALOGUE_LOG"
