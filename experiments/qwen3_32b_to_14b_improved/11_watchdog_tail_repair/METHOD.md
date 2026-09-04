# Watchdog-Gated Tail Repair R32（WGTR-R32）

## 方法定义

WGTR-R32 是一个 **post-hoc、待验证的假设**，不是已经冻结的候选，也不是论文超越结论。
它利用一个不读答案的在线失败信号：NVIDIA k20 的 R1 解码如果在前 32 个生成 token
内仍没有 EOS 或注册 stop，则放弃该 session，重新构造 R32 cache 并从头 greedy decode；
否则直接返回 R1 的原始结果对象。

固定状态机如下：

1. 以 `mapped_r1` 启动 greedy decode。
2. 每生成一个 content token，只查看解码器是否报告 `eos`、`stop_sequence` 或
   `max_tokens`。若在第 1–32 个 token 内终止，返回 engine 持有的 R1 outcome；不复制、
   不改写、不重算。
3. 若第 32 个 token 后仍未终止，中止 R1，使用相同 prompt 重新构造
   `tail_refresh_r32` cache，从 token 0 重新 greedy decode 到终止。

离线等价门为：

```text
trigger = (R1.generated_tokens > 32) OR R1.cap_hit
output  = R32 if trigger else R1
```

32 与 R32 的 refresh window 共用同一个固定 stall budget。它不是由 gold、数据集名字或
pilot 分数搜索得到的；但这个假设是在查看 03 pilot 后提出，因此依然属于探索性 post-hoc
分析，必须在新数据上前瞻验证。

## 为什么这个信号可能有效

正常短答案 QA 的 R1 多数很快 EOS；异常长生成或撞 cap 是 cache 映射导致生成轨迹失稳的
可观测症状。R32 用 31 个原生 target prompt token 修复最近上下文状态，03 pilot 中它显著
降低了 MuSiQue 的 cap/长生成问题。WGTR 仅对疑似失稳轨迹支付 repair 成本，避免让所有样本
承担 R32 在 HotpotQA 上偶发的正确到错误回归。

这只是机制解释，不是因果证明。门控可能同时捕捉“答案本来就长”的正常样本，R32 也可能
把一个最终会答对的 R1 轨迹改坏。

## 在线接口

[`src/watchdog.py`](src/watchdog.py) 暴露 `RestartableDecodeEngine`、`DecodeSession` 和
`WatchdogTailRepair`。`request` 是 opaque object，状态机从不解引用；因此即使调用者把
dataset/gold 放在 request 中，门也无法读取。`WatchdogRun.output` 保留底层 engine outcome
的对象身份；unit test 验证 no-trigger 路径 `output is r1_outcome`。

给未来 07/10 runner 的注入点是一个可 restart 的 step-wise engine：

```python
watchdog = WatchdogTailRepair(engine, stall_budget=32)
run = watchdog.run(prepared_request)
checkpoint = run.output              # exact R1 or exact restarted R32 outcome
audit = run.audit                     # no gold, no prediction score
```

未来 runner 应新增独立 mode，而不是覆盖 `nvidia_k20` 或 `tail_refresh_r32`。checkpoint 中应
记录 R1 前缀 token IDs、abort token 数、两个 cache binding、最终 selected mode 和完整
state path。R32 restart 失败时应写 failure artifact，不能静默返回已中止的 R1 半成品。

## 成本口径

触发样本会额外支付一次 R32 cache 构造（包括 31 个 target-native prompt replay token）和
一次从头解码。另一方面，原本很长的 R1 会在第 32 token 被提前中止，所以 generated decode
steps 可能反而下降。03 是四个 mode 的独立离线运行，无法测得真实在线 latency：

- `selected-mode wall` 漏掉触发前 R1 的 32-token 成本；
- `full-R1 + R32 wall` 把本应在 token 32 中止的 R1 全程算入，是保守上界 proxy；
- 两者都不是 WGTR 的实测 wall time，前瞻实验必须直接计时。

## 失败模式

- 正常答案确实需要超过 32 token，导致不必要 repair；
- R1 在 token 33 以后本会正确 EOS，而 R32 变差；
- R32 cache 重建 OOM/异常，增加 tail latency 和失败率；
- batch serving 中单样本 restart 造成调度碎片；
- stop detector 与离线 `generated_tokens` 语义不一致；
- 首次 R1 和 restart R32 若 prompt/tokenizer/position 不完全相同，反事实不成立；
- cap 降低不等于答案正确，必须同时报告 F1/EM、正确保留、cap 和 latency。

## 结论边界

03 pilot 已被查看，因此本文档中的结果只能表述为“同一已观察 pilot 上的探索性
counterfactual”。在未观察的去污染 screen 和 confirm 上按冻结协议成功复现之前，禁止写
“优于 NVIDIA”“已胜出”或“泛化有效”。
