# Prospective v2 frozen protocol

## 1. Estimand

同一个 frozen question record、prompt token IDs、greedy decoding contract 下，估计 candidate 相对
`nvidia_k20` 的 paired official-answer F1/EM 增量，以及相对 `native14` 的剩余掉点。统计单位是
完整 question/sample ID，不是 token、answer alias 或独立 mode cell。

primary candidate 在 outcome 打开前固定为 `two_pass_r1_then_r32`。`shrinkage` 与 fixed R32 作为
预先注册的诊断 candidate 报告，但不能在 screen 后根据最好分数替换 primary 再声称 gate 通过。

## 2. Data separation

| Tier | Split | HotpotQA | MuSiQue | 用途 |
|---|---|---:|---:|---|
| smoke | screen 前 2 IDs | 2 | 2 | 执行完整性，不升级 |
| screen | 10 frozen screen | final published count（≤128） | final published count（≤256） | 单次 go/no-go |
| confirm | 10 disjoint confirm | final published count（≤256） | final published count（≤512） | 仅 screen pass 后 |
| locked | official dev/test | 0 opened | 0 opened | 不属于本目录 |

Registry 必须只有 ID、source index、answer-independent source-projection hash、stratum-local rank、
rank hash 与 stratum；任何 answer、question、context、prediction、metric 字段都被拒绝。只有 exact
`PUBLISHED.json` 14-file publication 和 complete G0 v2 都通过后，preparer 才按 exact source index
解引用 pinned train record，重算 producer 的 canonical question/passages/stratum projection SHA，再
读取 answer/context。selection 不会读取 answer 或模型 outcome。

G0 v2 receipt 必须精确绑定 config 中相同的 release receipt 与四 split refs，并显式覆盖：

- methoddev HotpotQA → mapper-fit；
- methoddev MuSiQue → mapper-fit；
- methoddev HotpotQA → locked-eval projection；
- methoddev MuSiQue → locked-eval projection。

comparisons key set 必须精确为上述四个方向，每项精确为 `{"flags": 0}`，不得添加额外方向或额外
zero-valued 字段；receipt 还必须说明 locked outcomes 未打开。

## 3. Model cells

每个样本先形成四个完整 raw checkpoint：

1. `native14`：14B full native prefill；
2. `nvidia_k20`：论文兼容 centered-ridge k20 mapped cache；
3. `shrinkage`：02 frozen headwise convex shrinkage mapper；
4. `tail_refresh_r32`：NVIDIA k20 mapped prefix + 31 target-native prompt tail tokens + 1 engine boundary。

R8 被删除，因为它不是本 protocol 的 raw mode。所有模式 cap=256、no-think、greedy、batch=1，
完全相同 prompt IDs 和 stop sequences。每个 raw file 固定 token IDs/hash、decoded continuation、
stop/cap、actual excluded `terminal_token_id`、`frozen_eos_token_ids`、official metrics、monolithic wall
time、GPU `0..7` 各自 peak allocation 和 method binding。EOS 集合由已绑定 target model 的
`config.json`/`generation_config.json` 重算；EOS terminal 必须在集合中，另外两种 stop reason 必须为
null。当前 runtime 没有独立可审计的 prefill/decode timer，因此 phase timing 固定为 unavailable/null
及唯一 monolithic scope，不能估算冒充实测。所有 schema-defined integer/bool 与嵌套 R32/cost/metric
对象按 JSON 类型递归精确比较，Python bool/int 别名不是合法等价。

Cross-mode file binding 有顺序：native file → NVIDIA file（绑定 native SHA）→ shrinkage/R32 files
（同时绑定 native 与 NVIDIA SHA）。任何配对文件被替换都使 downstream checkpoint 失效。

## 4. Two-pass selector

固定规则：

```text
assert R1.status == complete and R32.status == complete
trigger = (R1.generated_tokens > 32) OR R1.cap_hit
selected = R32 if trigger else R1
```

gate function 只接收 R1 generation evidence；不能接收 dataset、gold、metric 或 prediction。derived
result 必须逐字段复制 selected raw prediction/generation/metrics，并绑定两个 raw file SHA。

这不是 experiment 11 的 online state machine：不会在第 32 token abort R1。报告的 sequential wall
是 separate complete raw runs 的 policy proxy；`online_end_to_end_latency_measured=false`。因此准确率
可作前瞻 selector 验证，真实在线 latency superiority 不能由本矩阵推出。

## 5. Statistics and gates

- normalized-token F1/EM 采用 pinned HotpotQA/MuSiQue official answer semantics；MuSiQue 对 primary +
  aliases 取 max。
- 每个 dataset 与 pooled 都报告 candidate/native/NVIDIA mean、delta、native drop reduction、cap、
  termination、mean/P95 wall、selector trigger rate。
- paired percentile bootstrap：seed 20260830、10,000 replicates、95% CI、sample-ID resampling。
- screen gate 固定在 `configuration.STATIC_PROMOTION`；任何更改都会改变 config/code fingerprint。
- confirm definition 必须嵌入且重新验证通过的 screen validation receipt；validation 会递归重算
  screen raw/derived/summary/gate，而不是信任序列化的 pass 字段。

## 6. Atomicity and claims

Every formal Python entrypoint is launched with the pinned interpreter,
`PYTHONDONTWRITEBYTECODE=1`, and the exact absent
`PYTHONPYCACHEPREFIX` recorded in the runtime contract.  The entrypoint checks
these values before importing local `src`; the deep pre-CUDA receipt records
the same policy and rejects any existing file, directory, or dangling symlink
at that prefix.  The two content-addressed external interfaces are executed
from freshly hashed source bytes rather than through a timestamp-based loader.
Because the historical upstream receipt predates two executable package files,
the formal config also binds an exact `runtime_package_closure` for
`kvcache_transfer/__init__.py` and `kvcache_transfer/io.py`; both are verified
before model/CUDA load and included in the runtime binding digest.

Run definition、每个 raw/derived checkpoint、raw-completion/summary/decision/completion/validation receipts
都有 byte count + SHA sidecar；live status 在 terminal seal 前使用原子替换，terminal status 再冻结
sidecar。真实 runner 另绑定 pre-CUDA full-weight/code hash、nvidia-smi idle、torch/CUDA、method provenance
和每层 5+3 sharding evidence。immutable writer 遇到同路径不同内容直接失败。validator 从 frozen inputs
重建 definition、逐格重算 metric/selector/bootstrap/gate 并要求结果树无多余或 failure 文件。

GPU execution 与 CPU seal 由同一 per-run flock 串行；所有 GPU terminal/raw 检查都在获得锁后重做。
raw receipt 只有在 frozen tokenizer 全量验证四模式 token/stop/metric/cross-mode/R32 evidence 及 exact raw
tree 后才能发布，缺 sidecar 恢复也执行相同验证。terminal `status=complete` 一旦内容寻址后不可被任何
live/failure writer 降级。

允许结论边界：smoke 无效果结论；screen pass 仅授权 confirm；confirm CI 支持时只称“去污染
method-dev confirm 上优于同设置 NVIDIA k20”。official dev/test 没打开前，不得称论文全面超越。
