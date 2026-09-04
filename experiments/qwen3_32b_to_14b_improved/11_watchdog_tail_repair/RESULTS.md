# Exploratory counterfactual results

严格 CPU analyzer 对 03 receipt 绑定的 384 个 raw artifacts 逐文件校验通过；96 个 pilot
样本均有 `target / R1 / R8 / R32` 四 mode 完整配对，没有 failure/extra artifact。以下结果
是在已观察 pilot 上提出规则后的 **post-hoc counterfactual**，不是新的 GPU 实验。

| Dataset（pilot, n=48） | R1 normalized-token F1 | WGTR-R32 F1 | paired Δ | 95% bootstrap CI | triggers |
|---|---:|---:|---:|---:|---:|
| HotpotQA train | 0.731415 | 0.741832 | +0.010417（+1.0417 pp） | [0.000000, 0.031250] | 1/48（2.08%） |
| MuSiQue train | 0.363879 | 0.447645 | +0.083766（+8.3766 pp） | [0.015583, 0.163921] | 13/48（27.08%） |

HotpotQA 唯一 trigger 是非 cap 长生成；MuSiQue 13 个 trigger 中 7 个 cap、6 个非 cap 长生成。
WGTR 的选择门在读取 metric 之前完成，输入仅为 R1 `generated_tokens` 与 `cap_hit`。

生成 token-step 的精确反事实计数（不含重复 prompt/cache 构造）为：HotpotQA 332 → 265
（0.798×），MuSiQue 2543 → 722（0.284×）。触发会分别增加 1/13 次 R32 cache rebuild
以及 31/403 个 target-native prompt replay token。真实 wall time 未测：把已经完整跑完的 R1
wall 再加触发 R32 wall 得到的保守 proxy 是 HotpotQA 1.019×、MuSiQue 1.148×，但真正在线
状态机会在第 32 token 中止 R1，因此该 proxy 是上界式估计，不能当 latency 结果。

证据：

- `results/analysis.json`：aggregate、paired bootstrap 与计算口径；
- `results/sample_audit.jsonl`：不含 answer/prediction/reference 文本的逐样本选择审计；
- `results/receipt.json`：384 raw artifacts、03 receipt/inventory、config 与代码 binding；
- 三个 artifacts 均有独立 `.sha256.json` sidecar；
- 11 个 tests（含 byte tamper、generation tamper、exact R1 object identity、token-32 边界）通过。

结论仍为：**promising hypothesis, not a superiority result**。必须在未来 10 的未观察、去污染
screen/confirm 上冻结规则后前瞻运行，才能判断是否泛化。
