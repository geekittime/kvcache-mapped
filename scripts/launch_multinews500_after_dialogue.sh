#!/usr/bin/env bash
set -Eeuo pipefail
set -o noclobber
unset PYTHONOPTIMIZE

PROJECT_ROOT=/home/wby/llm-cache/nvidia-kvcache
UPSTREAM_ROOT="$PROJECT_ROOT/runs/20260820_qwen3_14b_to_32b_dialogue_bottom1000_full_v1"
PREFLIGHT_ROOT="$PROJECT_ROOT/runs/20260824_qwen3_14b_to_32b_multinews_bottom500_preflight_v1"
SMOKE_ROOT="$PROJECT_ROOT/runs/20260824_qwen3_14b_to_32b_multinews_bottom500_smoke_v1"
FORMAL_ROOT="$PROJECT_ROOT/runs/20260824_qwen3_14b_to_32b_multinews_bottom500_full_v1"
SMOKE_LOG="$PROJECT_ROOT/runs/logs/20260824_qwen3_14b_to_32b_multinews_bottom500_smoke_v1.log"
FORMAL_LOG="$PROJECT_ROOT/runs/logs/20260824_qwen3_14b_to_32b_multinews_bottom500_full_v1.log"
STATE_FILE="$PROJECT_ROOT/runs/20260824_qwen3_14b_to_32b_multinews_bottom500_queue_state.tsv"

UPSTREAM_FINGERPRINT=c37e81d37f40758eebd57f4664d924f7c76ea8e5b738afeb2358eac2d059d614
ENTRY=scripts/evaluate_long_context_multinews_500.py
CONFIG=configs/eval_qwen3_14b_to_32b_multinews_500.yaml
BASE=scripts/evaluate_long_context.py
EVALUATION_CORE=src/kvcache_transfer/evaluation.py
ENTRY_SHA=318c2794feb9a8ed548445460a40d40a08f74606c04e8292c4837ec11520c496
CONFIG_SHA=a16df3b94086773c9da2ab2643e54351959c28715a6e63a464bacc0915c524bd
BASE_SHA=9fb29dab32d7bc4cab5fe4f1f3e7a53fdd669f4efb43f0081da88a71f746154b
EVALUATION_SHA=5e6a2ea48b7ac8b7c3a7b3673fc2544bbc47880ba9a32f2e7981604128595fd7
PROTOCOL=qwen3_14b_to_32b_multinews_target_vs_mapped_bottom500_v1
SELECTED_SHA=47f460b9bb3c69a60cea1b1d44ea8a9a64fd33623d107f4129d48ca6c5e76076
SELECTED_INDICES_SHA=d39b44894d60ad2fbbb3aaad96b619fcbfde833a982f60644305f109821f8510
SMOKE_INDEX=1463

cd "$PROJECT_ROOT"

test ! -e "$PREFLIGHT_ROOT"
test ! -e "$SMOKE_ROOT"
test ! -e "$FORMAL_ROOT"
test ! -e "$SMOKE_LOG"
test ! -e "$FORMAL_LOG"
test ! -e "$STATE_FILE"
: > "$STATE_FILE"

record_state() {
  printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$STATE_FILE"
}

on_exit() {
  status=$?
  if (( status != 0 )); then
    record_state "failed exit_status=$status"
  fi
}
trap on_exit EXIT

check_frozen_hashes() {
  printf '%s  %s\n' \
    "$ENTRY_SHA" "$ENTRY" \
    "$CONFIG_SHA" "$CONFIG" \
    "$BASE_SHA" "$BASE" \
    "$EVALUATION_SHA" "$EVALUATION_CORE" \
    | sha256sum -c -
}

wait_for_idle() {
  local idle_samples=0
  local gpu_count max_used compute_pids
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
}

record_state waiting_for_dialogue_complete
while true; do
  if [[ -f "$UPSTREAM_ROOT/status.json" ]]; then
    upstream_status=$(.venv/bin/python - "$UPSTREAM_ROOT/status.json" <<'PY'
import json
import sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("status", "missing"))
PY
)
    if [[ "$upstream_status" == complete ]]; then
      break
    fi
    if [[ "$upstream_status" == failed || "$upstream_status" == incomplete ]]; then
      record_state "upstream_failed_closed status=$upstream_status"
      exit 1
    fi
  fi
  sleep 30
done

.venv/bin/python - "$UPSTREAM_ROOT" "$UPSTREAM_FINGERPRINT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise SystemExit("Python optimization must be disabled for receipt validation")

root = Path(sys.argv[1]).resolve()
fingerprint = sys.argv[2]

def load(name):
    return json.loads((root / name).read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

status = load("status.json")
summary = load("summary.json")
manifest = load("results.manifest.json")
assert status.get("status") == "complete" and status.get("failures") == 0
assert summary.get("status") == "complete"
assert manifest.get("status") == "complete"
assert all(value.get("request_fingerprint") == fingerprint for value in (status, summary, manifest))
assert manifest.get("expected_result_count") == 4000
files = manifest.get("files")
assert isinstance(files, list) and len(files) == 4000
expected_paths = {
    f"results/{task}/{mode}/{index:06d}.json"
    for task in ("lmsys", "sharegpt4")
    for mode in ("target_standalone", "mapped")
    for index in range(1000)
}
assert {item["path"] for item in files} == expected_paths
for item in files:
    path = (root / item["path"]).resolve()
    assert root in path.parents and path.is_file()
    assert path.stat().st_size == item["nbytes"]
    assert sha256(path) == item["sha256"]
for task in ("lmsys", "sharegpt4"):
    task_summary = summary["tasks"][task]
    assert task_summary.get("expected_samples") == 1000
    for mode in ("target_standalone", "mapped"):
        observed = task_summary["modes"][mode]
        assert observed.get("complete") is True
        assert observed.get("completed_samples") == 1000
manifest_identity = status["results_manifest"]
manifest_path = (root / manifest_identity["path"]).resolve()
assert root in manifest_path.parents
assert manifest_path.stat().st_size == manifest_identity["nbytes"]
assert sha256(manifest_path) == manifest_identity["sha256"]
summary_path = root / "summary.json"
assert summary_path.stat().st_size == status["summary_nbytes"]
assert sha256(summary_path) == status["summary_sha256"]
print("dialogue_complete_receipt_passed")
PY

record_state waiting_for_preflight_idle
wait_for_idle
check_frozen_hashes

record_state running_cpu_preflight
CUDA_VISIBLE_DEVICES='' PYTHONUNBUFFERED=1 \
  .venv/bin/python "$ENTRY" --prepare-only --output-dir "$PREFLIGHT_ROOT"

PREFLIGHT_FINGERPRINT=$(.venv/bin/python - "$PREFLIGHT_ROOT" "$PROTOCOL" "$SELECTED_SHA" "$SELECTED_INDICES_SHA" "$SMOKE_INDEX" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise SystemExit("Python optimization must be disabled for receipt validation")

root = Path(sys.argv[1]).resolve()
protocol, selected_sha, indices_sha = sys.argv[2:5]
smoke_index = int(sys.argv[5])
status = json.loads((root / "status.json").read_text(encoding="utf-8"))
definition = json.loads((root / "run_definition.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
assert status.get("status") == "prepared" and status.get("gpu_used") is False
assert definition.get("protocol") == protocol
assert definition.get("tasks") == ["multi_news"]
assert definition.get("modes") == ["target_standalone", "mapped"]
assert definition.get("sample_scope") == "formal_sha256_bottom500_of_official_test5622"
assert definition.get("limit") is None and definition.get("sample_indices") is None
selection = manifest["multi_news"]["formal_selection"]
assert selection.get("source_rows") == 5622 and selection.get("selected_rows") == 500
assert selection.get("selected_sha256") == selected_sha
assert selection.get("selected_indices_sha256") == indices_sha
assert len(selection.get("selected", [])) == 500
assert len({item["source_index"] for item in selection["selected"]}) == 500
assert smoke_index in {item["source_index"] for item in selection["selected"]}
extension = definition["multinews_500_extension"]
assert extension.get("selected_sha256") == selected_sha
assert extension.get("selected_indices_sha256") == indices_sha
fingerprint = definition["request_fingerprint"]
assert status.get("request_fingerprint") == fingerprint
print(fingerprint)
PY
)
record_state "preflight_passed fingerprint=$PREFLIGHT_FINGERPRINT"

record_state waiting_for_smoke_idle
wait_for_idle
check_frozen_hashes
test ! -e "$SMOKE_LOG"
: > "$SMOKE_LOG"
record_state running_longest_selected_smoke
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
  .venv/bin/python "$BASE" --config "$CONFIG" --tasks multi_news \
  --sample-indices "multi_news:$SMOKE_INDEX" --output-dir "$SMOKE_ROOT" \
  2>&1 | tee -a "$SMOKE_LOG"

.venv/bin/python - "$SMOKE_ROOT" "$SMOKE_INDEX" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise SystemExit("Python optimization must be disabled for receipt validation")

root = Path(sys.argv[1]).resolve()
index = int(sys.argv[2])

def canonical(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

status = load(root / "status.json")
summary = load(root / "summary.json")
manifest = load(root / "results.manifest.json")
assert status.get("status") == "complete" and status.get("failures") == 0
assert summary.get("status") == "complete"
assert manifest.get("expected_result_count") == 2 and len(manifest.get("files", [])) == 2
target = load(root / f"results/multi_news/target_standalone/{index:06d}.json")
mapped = load(root / f"results/multi_news/mapped/{index:06d}.json")
assert target["prompt"] == mapped["prompt"]
assert target["references"] == mapped["references"]
assert target["record_sha256"] == mapped["record_sha256"]
assert target["prompt"]["original_prompt_tokens"] == 63184
assert target["prompt"]["generation_context_truncation"]["truncated"] is True
assert target["generation"]["requested_max_tokens"] == 1024
assert mapped["generation"]["requested_max_tokens"] == 1024
assert mapped["target_checkpoint_binding"]["target_checkpoint_canonical_sha256"] == canonical(target)
for payload in (target, mapped):
    value = float(payload["metrics"]["rouge_l_f1"])
    assert math.isfinite(value) and 0.0 <= value <= 1.0
print("multinews_longest_selected_smoke_passed")
PY

record_state waiting_for_formal_idle
wait_for_idle
check_frozen_hashes

.venv/bin/python - "$PREFLIGHT_ROOT/run_definition.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise SystemExit("Python optimization must be disabled for receipt validation")

definition = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path("/home/wby/llm-cache/nvidia-kvcache")
for relative, identity in definition["code"]["files"].items():
    path = root / relative
    assert path.is_file() and path.stat().st_size == identity["nbytes"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == identity["sha256"], relative
print("preflight_code_identity_still_current")
PY

test ! -e "$FORMAL_LOG"
: > "$FORMAL_LOG"
record_state "running_formal fingerprint=$PREFLIGHT_FINGERPRINT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
  .venv/bin/python "$ENTRY" --output-dir "$FORMAL_ROOT" \
  2>&1 | tee -a "$FORMAL_LOG"

.venv/bin/python - "$FORMAL_ROOT" "$PREFLIGHT_FINGERPRINT" "$SELECTED_SHA" "$SELECTED_INDICES_SHA" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

if sys.flags.optimize != 0:
    raise SystemExit("Python optimization must be disabled for receipt validation")

root = Path(sys.argv[1]).resolve()
fingerprint, selected_sha, indices_sha = sys.argv[2:]

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def canonical(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

status = load(root / "status.json")
summary = load(root / "summary.json")
manifest = load(root / "results.manifest.json")
definition = load(root / "run_definition.json")
dataset = load(root / "dataset_manifest.json")
assert status.get("status") == "complete" and status.get("failures") == 0
assert summary.get("status") == "complete"
assert manifest.get("status") == "complete"
assert all(value.get("request_fingerprint") == fingerprint for value in (status, summary, manifest, definition))
selection = dataset["multi_news"]["formal_selection"]
assert selection["selected_sha256"] == selected_sha
assert selection["selected_indices_sha256"] == indices_sha
indices = [item["source_index"] for item in selection["selected"]]
assert len(indices) == 500 and len(set(indices)) == 500
expected_paths = {
    f"results/multi_news/{mode}/{index:06d}.json"
    for mode in ("target_standalone", "mapped")
    for index in indices
}
files = manifest.get("files")
assert manifest.get("expected_result_count") == 1000
assert isinstance(files, list) and len(files) == 1000
assert {item["path"] for item in files} == expected_paths
for item in files:
    path = (root / item["path"]).resolve()
    assert root in path.parents and path.is_file()
    assert path.stat().st_size == item["nbytes"]
    assert sha256(path) == item["sha256"]
task = summary["tasks"]["multi_news"]
assert task.get("expected_samples") == 500
for mode in ("target_standalone", "mapped"):
    observed = task["modes"][mode]
    assert observed.get("complete") is True and observed.get("completed_samples") == 500
expected_truncation = {
    127: (33159, 17799),
    620: (15908, 548),
    1463: (63184, 47824),
    5562: (23746, 8386),
}
observed_truncated = set()
for index in indices:
    target = load(root / f"results/multi_news/target_standalone/{index:06d}.json")
    mapped = load(root / f"results/multi_news/mapped/{index:06d}.json")
    assert target["prompt"] == mapped["prompt"]
    assert target["references"] == mapped["references"]
    assert target["record_sha256"] == mapped["record_sha256"]
    truncation = target["prompt"]["generation_context_truncation"]
    if index in expected_truncation:
        original, removed = expected_truncation[index]
        assert truncation == {
            "policy": "head_tail",
            "truncated": True,
            "original_tokens": original,
            "used_tokens": 15360,
            "removed_tokens": removed,
            "head_tokens": 7680,
            "tail_tokens": 7680,
        }
        observed_truncated.add(index)
    else:
        assert truncation["truncated"] is False
    assert mapped["target_checkpoint_binding"]["target_checkpoint_canonical_sha256"] == canonical(target)
    for payload in (target, mapped):
        metric = float(payload["metrics"]["rouge_l_f1"])
        assert math.isfinite(metric) and 0.0 <= metric <= 1.0
assert observed_truncated == set(expected_truncation)
manifest_identity = status["results_manifest"]
manifest_path = (root / manifest_identity["path"]).resolve()
assert root in manifest_path.parents
assert manifest_path.stat().st_size == manifest_identity["nbytes"]
assert sha256(manifest_path) == manifest_identity["sha256"]
assert (root / "summary.json").stat().st_size == status["summary_nbytes"]
assert sha256(root / "summary.json") == status["summary_sha256"]
print("multinews_bottom500_formal_receipt_passed")
PY

record_state "complete fingerprint=$PREFLIGHT_FINGERPRINT"
trap - EXIT
