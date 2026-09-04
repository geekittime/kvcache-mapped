# WGTR-R32 protocol

## 当前分析（允许做什么）

- 输入只允许是 03 已冻结 pilot 的 384 个 raw results。
- `config.json` 固定绑定 03 strict-validation receipt、result inventory 和 paired sample
  summary 的绝对路径、字节数与 SHA-256。
- analyzer 必须逐个重哈希 384 个 raw files，验证 96 个样本都有完整的四 mode 配对，验证
  generation token count/hash/cap/stop contract，然后才计算 WGTR。
- gate 只能接收 R1 的 `generated_tokens` 和 `cap_hit`。选择完成后才允许读取 metrics 评分。
- 输出不得包含 answer、prediction 或 reference 文本。
- bootstrap 固定 seed `20260830`、10,000 次、paired sample-id resampling。
- 本阶段不用 GPU，不产生新模型输出。

## 前瞻验证（必须完成后才能升级结论）

WGTR 的下一次模型运行必须使用未来 `10_decontaminated_eval/` 中此前未打开 outcome 的
screen/confirm，并在运行前冻结以下内容：

1. stall budget = 32，不扫描 8/16/24/48/64；
2. initial mode = immutable NVIDIA k20 R1；repair mode = R32；
3. greedy、cap=256、stop sequences、prompt template 与 paired baselines 完全一致；
4. 每个数据集 paired sample IDs、input hashes、runner code hashes 和 method binding；
5. primary metric = normalized-token F1；同时报告 EM、cap、termination、target-correct
   retention、R1-correct-to-WGTR-wrong、真实 end-to-end latency、P95 latency 和 peak VRAM；
6. screen 只作一次 go/no-go。通过后不得改门，才进入独立 confirm；若失败则记录失败，不能
   根据 screen gold 调整阈值后仍称 confirm。

建议的 screen gate（在 outcome 打开前写进 10 的冻结 manifest）：两个注册数据集的 paired
F1 均不低于 R1，pooled paired F1 为正，cap/有效终止不差，且 R1-correct retention 不低于
预注册阈值。confirm 应报告 paired bootstrap CI；CI 跨零时只能称不确定，不能称超越。

## 论文比较边界

即使 confirm 胜过 R1，WGTR-R32 仍然是混合系统：触发时执行 target-native tail replay，不能
作为“纯零 target-prefill mapper”与论文 R1 混为一谈。任何论文超越结论还需要同硬件、同
prompt、同样本、同 cap 的 paired accuracy 与系统成本比较，并通过项目 locked-test gate。
