#!/usr/bin/env bash
set -Eeuo pipefail
unset PYTHONOPTIMIZE

PROJECT_ROOT=/home/wby/llm-cache/nvidia-kvcache
FORMAL_ROOT="$PROJECT_ROOT/runs/20260824_qwen3_14b_to_32b_multinews_bottom500_full_v1"
LOG_PATH="$PROJECT_ROOT/runs/logs/20260824_qwen3_14b_to_32b_multinews_bottom500_full_v1.resume_v1.log"
STATE_PATH="$PROJECT_ROOT/runs/20260825_qwen3_14b_to_32b_multinews_bottom500_resume_state.tsv"
LOCK_PATH="$PROJECT_ROOT/runs/gpu8_experiment.lock"
FINGERPRINT=e6152b67f6c720b30550cd06ca6d148485ae035e0dc4d94994c88b0e7812e966

cd "$PROJECT_ROOT"

printf '%s  %s\n' \
  318c2794feb9a8ed548445460a40d40a08f74606c04e8292c4837ec11520c496 scripts/evaluate_long_context_multinews_500.py \
  a16df3b94086773c9da2ab2643e54351959c28715a6e63a464bacc0915c524bd configs/eval_qwen3_14b_to_32b_multinews_500.yaml \
  9fb29dab32d7bc4cab5fe4f1f3e7a53fdd669f4efb43f0081da88a71f746154b scripts/evaluate_long_context.py \
  5e6a2ea48b7ac8b7c3a7b3673fc2544bbc47880ba9a32f2e7981604128595fd7 src/kvcache_transfer/evaluation.py \
  785e2d57c8953a0c630f3044c4c06381f81a88799ca651960586f584f569ec08 artifacts/mappers/qwen3_14b_to_32b/k8_lambda0.01/manifest.json \
  | sha256sum -c -

exec 9>"$LOCK_PATH"
flock -n 9 || {
  printf 'another repository-managed eight-GPU experiment holds %s\n' "$LOCK_PATH" >&2
  exit 1
}

.venv/bin/python - "$FORMAL_ROOT" "$FINGERPRINT" <<'PY'
import json
import sys
from pathlib import Path

if sys.flags.optimize:
    raise SystemExit("Python optimization must be disabled")
root = Path(sys.argv[1]).resolve()
fingerprint = sys.argv[2]
definition = json.loads((root / "run_definition.json").read_text(encoding="utf-8"))
status = json.loads((root / "status.json").read_text(encoding="utf-8"))
if definition.get("request_fingerprint") != fingerprint:
    raise SystemExit("formal run-definition fingerprint mismatch")
if status.get("request_fingerprint") != fingerprint or status.get("status") != "failed":
    raise SystemExit("resume source is not the expected failed formal run")
if status.get("failures") != 1:
    raise SystemExit("resume source does not contain exactly one recorded failure")
PY

: > "$STATE_PATH"
printf '%s\twaiting_for_12_idle_samples\n' "$(date -u +%FT%TZ)" >> "$STATE_PATH"
idle_samples=0
while (( idle_samples < 12 )); do
  gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  compute_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d' || true)
  max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -nr | head -1)
  if [[ "$gpu_count" == 8 && -z "$compute_pids" && "$max_used" =~ ^[0-9]+$ && "$max_used" -le 64 ]]; then
    idle_samples=$((idle_samples + 1))
  else
    idle_samples=0
  fi
  sleep 5
done

printf '%s\trunning_resume fingerprint=%s\n' "$(date -u +%FT%TZ)" "$FINGERPRINT" >> "$STATE_PATH"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
  .venv/bin/python scripts/evaluate_long_context_multinews_500.py \
  --output-dir runs/20260824_qwen3_14b_to_32b_multinews_bottom500_full_v1 \
  2>&1 | tee -a "$LOG_PATH"

.venv/bin/python - "$FORMAL_ROOT" "$FINGERPRINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
fingerprint = sys.argv[2]

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
if status.get("status") != "complete" or status.get("failures") != 0:
    raise SystemExit("formal resume did not complete cleanly")
if summary.get("status") != "complete" or manifest.get("status") != "complete":
    raise SystemExit("formal summary or results manifest is incomplete")
if any(value.get("request_fingerprint") != fingerprint for value in (status, summary, manifest)):
    raise SystemExit("formal identity drift after resume")
files = manifest.get("files")
if manifest.get("expected_result_count") != 1000 or not isinstance(files, list) or len(files) != 1000:
    raise SystemExit("formal result inventory is not exactly 1000 checkpoints")
for item in files:
    path = (root / item["path"]).resolve()
    if root not in path.parents or not path.is_file():
        raise SystemExit(f"invalid result path: {path}")
    if path.stat().st_size != item["nbytes"] or sha256(path) != item["sha256"]:
        raise SystemExit(f"result digest mismatch: {path}")
PY

printf '%s\tcomplete fingerprint=%s\n' "$(date -u +%FT%TZ)" "$FINGERPRINT" >> "$STATE_PATH"
