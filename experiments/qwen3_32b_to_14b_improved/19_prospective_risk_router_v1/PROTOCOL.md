# v1 冻结协议

## 目的

建立一个可在未来新 registry 上前瞻执行的、gold-free 的 R=1→R=32 风险路由器。
本协议不是模型训练协议，也不替代独立的 locked-test 统计协议。

## 信息边界

允许的信息严格限于：完整 prompt token 数、可选的 context-only token 数、已完成
R=1 生成 token 数和第一遍传输完成位。实现不得接受或推导输出文本、参考答案、
任何质量分数、任务/数据集标签、wall time、GPU/显存状态。token 计数应绑定到未来
registry 的 tokenizer；若 tokenizer 身份不匹配，应重新建立独立 router 目录，而不
得静默换算阈值。

## 决策状态机

1. `assess` 先做类型、范围、context≤prompt 及 first-pass-complete 检查。
2. 任一进入信号成立后，`HysteresisLatch` 置 active；只有所有可用信号低于退出
   阈值才清除。
3. `route_batch` 按优先级 `(both=3, generation=2, input=1)` 和原始索引稳定排序，
   再从预先声明的预算中保留候选。
4. 所有 no-go 情形都回退到 R=1；回退不得消耗 R=32 预算。

## 预算与记录

预算为 `min(32, floor(0.30*n))`，其中 `n` 是预先冻结的批大小。每次成功升级只
消耗一个单位；优先级延后和预算耗尽必须分别记录。不能通过扩大批次、跨批借用或
重排后重试规避预算。

## 统计门禁（仅未来 fresh registry）

* primary：router policy 与 always-R=1 的 paired official answer F1；
* 每个注册数据集的平均差值 ≥ 0；
* 合并 paired 95% CI 下界 > 0；
* cap 率不增加、有效终止率不降低；
* 决策缺失/非法为 0，no-go 比例 ≤ 5%；
* 任何阈值、优先级、预算或迟滞参数在观察 fresh 质量后改变，整次实验作废。

## 证据等级

旧 03 pilot 只能用于选择并冻结本目录参数，属于 `posthoc method development`。
在新的 registry 完成上述门禁之前，本目录不得产生“优于 NVIDIA paper”或“泛化
有效”的结论。CPU 单元测试通过也不代表模型效果通过。
