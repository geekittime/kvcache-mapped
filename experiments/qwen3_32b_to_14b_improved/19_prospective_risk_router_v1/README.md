# Prospective R=1→R=32 risk router (v1)

状态：`FROZEN_POSTHOC_PILOT_DEVELOPMENT_ONLY`。

这是一个独立的、仅 CPU 的路由合同。它在完成 R=1（mapped）后，依据输入长度和
R=1 生成长度，决定是否再做一次 R=32 tail refresh。路由器本身不加载模型、不读
文件、不看答案文本、不访问参考答案或质量指标，也不读取计时/GPU 状态。

## 输入与固定规则

`src/risk_router.py` 的 `Observation` 只有四个字段：

* `prompt_tokens`：完整 prompt 的 token 数（必需）；
* `context_tokens`：仅 context 的 token 数（可选，若提供必须不大于 prompt 数）；
* `r1_generated_tokens`：已完成的 R=1 输出 token 数（必需）；
* `first_pass_complete`：第一遍是否完整结束（传输安全位）。

进入阈值与退出阈值均已冻结：

| 信号 | 进入 | 退出 |
|---|---:|---:|
| prompt token 数 | 2800 | 2600 |
| context token 数（可选） | 2800 | 2600 |
| R=1 生成 token 数 | 16 | 12 |

任一可用信号达到进入阈值即可进入候选。优先级固定为：同时满足输入和生成信号
为 3；仅生成信号为 2；仅输入信号为 1；无信号为 0。批处理时先选高优先级，
同优先级保持原始顺序。

迟滞是“每个请求一枚 latch”：从低信号到达进入阈值后保持 active，只有所有可用
维度都严格低于退出阈值才清除。跨请求必须新建 `HysteresisLatch`，不能共享状态。

预算固定为 `floor(0.30 × batch_size)`，并且最多 32 个 R=32 请求；96 条请求的
预算为 28。预算不足、优先级被延后、输入非法或第一遍未完成时，统一 `r1` 回退并
记录 `no_go:*` 原因。

## 旧 pilot 仅用于开发（不是验证）

阈值是在旧 `03_tail_refresh` 的 96 条 pilot（48 HotpotQA + 48 MuSiQue）查看后
事后选定的，因此下列数字都是 post-hoc 描述，不能作为泛化或超过 NVIDIA 方法的
证据：

* `R1_generated_tokens ≥16`：16/96 条，所选样本的 R32−R1 relaxed-F1 均值
  `+0.3706`（9 正、3 平、4 负）；
* `prompt_tokens ≥2800`：20/96 条，所选均值 `+0.3654`（10 正、8 平、2 负）；
* `context_tokens ≥2800`（用冻结 Qwen3 tokenizer 从 context 单独计数）：18/96
  条，所选均值 `+0.3504`（9 正、7 平、2 负）；
* 两类输入/生成信号取 OR：27/96 条（28.1%），所选均值 `+0.3514`，折算到
  全部 96 条的事后 policy uplift 为 `+0.0988` F1 点。

这些选择和均值都受到小样本、任务组成及阈值搜索影响；路由器没有使用这些 pilot
的答案或指标，也没有把 pilot 当作 locked test。

## 未来 fresh registry 的唯一允许用法

1. 先冻结全新的样本/提示 registry、tokenizer 身份和批次预算；冻结后才能运行路由
   器，且运行日志不得包含参考答案或质量指标。
2. 在每条样本上先做 R=1，再按本合同的 `route_batch` 或独立 `route_one` 决定是否
   做 R=32。不得依据 fresh 结果调阈值、调优先级或改预算。
3. 原子地保存每条路由决定、输入信号、预算消耗、`no_go` 原因及 R=1/R=32 输出。
4. 由独立 CPU 分析器在锁定规则下计算 paired official F1/EM、cap/终止率和成本。

建议的预注册 promotion/no-go 门禁：所有注册数据集的 router policy 平均 F1 不得
低于 always-R=1；合并 paired 95% CI 下界必须大于 0；cap 率和有效终止率不得变差；
缺失/非法路由或 no-go 率超过 5% 即 NO-GO。任一门禁失败都只能报告探索性结果，
不能称为超过论文方法。

本目录的 `artifacts/pilot_development_receipt.json` 只证明 CPU 合同和旧 pilot 绑定
完整；它明确禁止 locked-test、泛化或 paper-superiority claim。
