# 12 — Prospective decontaminated screen v2

这是一个**独立、完整执行路径已实现、尚未运行 GPU**的前瞻评测目录。它的目标是在未查看结果的
去污染 method-dev 数据上，同样本比较 Qwen3-14B native、NVIDIA k20、shrinkage、R32，以及
预先冻结的 two-pass 选择器能否让 32B→14B 的准确率掉点更小。

当前状态：`WAITING_FOR_10_FINAL_PUBLISHED_COUNTS_AND_09_G0_V2_PASS`。目录中只有通用实现、
不可运行模板和测试；没有 `config.json`、prepared 数据、formal run 或新模型输出。
`config.template.json`
含显式 placeholder，任何 CLI 都会在访问 registry split 前拒绝它，因此不会误读 10 的
`IN_PROGRESS` attempt。

## 冻结实验设计

- 数据：只接受 10 最终发布的四个 split：HotpotQA screen/confirm、MuSiQue screen/confirm。
- profile：smoke 固定 `2+2`。screen/confirm 的 exact counts 必须从 10 最终原子发布复制到 formal
  config 并与 JSONL 行数一致；原设计 `128+256` / `256+512` 现在只作为不可扩大的上限，不在
  component-feasibility 结束前预先冻结。
- raw modes（精确四个）：`native14`、`nvidia_k20`、`shrinkage`、
  `tail_refresh_r32`。没有 R8。
- 生成：greedy、no-think、batch=1、cap=256、相同 prompt/token IDs/stop boundaries。
- derived primary candidate：`two_pass_r1_then_r32`。
- headline：官方语义 normalized-token F1 与 EM；10,000 次 sample-ID paired bootstrap。
- official dev/test 始终 sealed；本目录只允许读取固定 train source 中被 registry 选中的记录。

two-pass 的语义与 11 的在线 watchdog 不同：这里必须先让 NVIDIA R1 **完整结束**，随后只读
`generated_tokens > 32` 或 `cap_hit`；触发则选择完整 R32 raw 结果，否则选择原始 R1。
它是对完整 paired raw grid 的 gold-blind 离线选择器，不是 token 32 处 early abort，也不把
11 的 post-hoc pilot 当作新数据验证。11 仅作为假设来源被内容寻址绑定。

## Fail-closed 顺序

1. 10 必须以最后一步原子写出 `PUBLISHED.json`：exact 8-key marker、14-file inventory、7 对
   payload/sidecar 全部 byte-verified，且四个 native JSONL 只有 IDs/hashes/ranks/stratum。
2. 09 complete-G0-v2 必须 `pass=true`、`closes_complete_g0=true`，精确绑定同一 marker/manifest/
   receipt/splits；comparisons 必须精确只有四个注册方向，且每项精确只有 `flags: 0`。
3. 才能从 `config.template.json` 生成新的 formal `config.json`；不得原地把 template 当 config。
4. `scripts/check_readiness.py` 通过后，`scripts/prepare.py` 才会按 frozen source index 读取 train
   answer/context，并冻结 exact no-think token IDs。代码没有 dev/test loader。
5. 依次 freeze + 执行 + seal + validate smoke；通过后才能 freeze screen。
6. screen 只做一次 go/no-go。只有预注册 gate 全部通过，content-addressed
   `validation_receipt.json` 才能授权 disjoint confirm。
7. confirm 必须显式绑定该 screen validation receipt；失败 screen、smoke 或手工 summary 都不能授权。

## Screen 升级门

primary candidate 固定为 two-pass，baseline 固定为 NVIDIA k20。全部条件同时成立才 `pass`：

- HotpotQA 和 MuSiQue 各自 F1 delta ≥ 0；
- pooled paired F1 delta > 0；
- 两个数据集 cap rate 均不升、有效 EOS/stop termination 均不降；
- 两个数据集 R1-correct EM retention 均 ≥ 0.98；
- raw/derived paired cells 完整，零 missing/failure。

screen pass 只授权 confirm，不授权“超过论文”或 locked-test 结论。confirm 内部的更强证据条件是
pooled paired F1 CI 下界 > 0 且两个数据集点估计均非负；即使满足，仍只能表述为 method-dev
confirm 证据。

## 代码入口

- `src/configuration.py`：固定配置、placeholder 拒绝、路径与引用检查。
- `src/registry_gate.py`：COMPLETE publication 与 complete-G0-v2 精确绑定。
- `src/prepare_data.py`：冻结 ID 后才解引用 train answer，exact token preparation。
- `src/gpu_guard.py`：全局 8-GPU flock 与连续 nvidia-smi idle gate（不触发 CUDA）。
- `src/run_state.py`：GPU/CPU seal 共用的 per-run flock 与不可降级 terminal status 状态机。
- `src/gpu_runtime.py`：真实 Qwen3-32B 5 卡 + Qwen3-14B 3 卡加载及 k20/shrinkage/R32 builders。
- `src/gpu_runner.py` / `src/raw_writer.py`：真实四模式逐样本执行、原子 checkpoint、resume、
  full-weight/runtime/code/topology evidence 与 raw completion receipt。
- `src/two_pass_selector.py`：完整 R1 后的 gold/dataset-blind 选择器。
- `src/checkpoints.py`：token、实际 terminal EOS、冻结 EOS 集合、exact 8-GPU cost、metric、R32 causal
  audit 与 cross-mode hash binding。
- `src/statistics.py` / `src/promotion.py`：paired bootstrap、掉点/retention/cost 与升级门。
- `src/run_protocol.py` / `src/validator.py`：run definition、immutable receipts、独立重算验证。
- `reference_interfaces.json`：只读审阅过的 07/11 文件 SHA；本目录不 import 07。

未来正式命令示意（当前 template 会按设计失败）：

```bash
PY=/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python
PYCACHE_PREFIX=/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/12_prospective_screen_v2/.formal_empty_pycache
# The prefix must not exist (including as a dangling symlink) before startup.
test ! -e "$PYCACHE_PREFIX" && test ! -L "$PYCACHE_PREFIX"
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" $PY scripts/check_readiness.py --config config.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" $PY scripts/prepare.py --config config.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" $PY scripts/freeze_run.py --config config.json --profile smoke --output runs/smoke_v2
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY scripts/run_gpu.py \
  --config config.json --profile smoke --output runs/smoke_v2
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" $PY scripts/seal_run.py --config config.json --profile smoke --output runs/smoke_v2
PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$PYCACHE_PREFIX" $PY scripts/validate_run.py --config config.json --profile smoke --output runs/smoke_v2
```

`PYTHONDONTWRITEBYTECODE=1` only disables bytecode writes; it does not disable
reading an adjacent timestamp-based `.pyc`.  The exact, absent
`PYTHONPYCACHEPREFIX` is therefore mandatory for every formal entrypoint, and
each script fails before importing this directory's `src` package if the
policy is missing or changed.

The historical NVIDIA runtime receipt is immutable and predates two executable
package files.  The template therefore carries an independent, exact
`runtime_package_closure` binding for `src/kvcache_transfer/__init__.py` and
`src/kvcache_transfer/io.py`; both files are hash-verified before any model or
CUDA load and are included in the runtime binding digest.

测试：

```bash
pytest -q
```

当前 CPU 测试覆盖 placeholder、动态 final-count binding、IN_PROGRESS fail-before-split-read、14-file
publication/sidecar、answer leakage、G0 exact schema/错绑、真实 runner mock/resume/kill-window、双 runner
与 seal 竞争、idle gate、full model/code binding、selector/terminal-token/EOS-set 边界、exact 8-GPU
cost、token/metric/cross-mode/R32
causal-audit 与 bool/int-alias tamper、attempt-ledger typed counts、paired bootstrap、screen gate、confirm
prerequisite，以及带 raw-GPU evidence 的完整
smoke receipt 重算链。
