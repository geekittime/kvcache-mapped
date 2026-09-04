# Frozen-anchor functional residual mapper

更新时间：2026-08-31

## 1. 目标与当前证据边界

目标是减少 Qwen3-32B cache 转移到冻结 Qwen3-14B 后的功能偏移，而不是单纯
提高 KV 坐标的 R²。本目录目前只完成方法、协议和 CPU synthetic 实现；没有训练
矩阵、没有 GPU 运行、没有下游准确率，也没有“超过 NVIDIA”的证据。

已有本地研究显示，k20 的训练重构更好不等于 native-14B 行为更稳；因此这里把
native-14B 的固定 query attention、suffix logits 和 EOS 行为作为主要训练信号，
KV Huber/R² 只阻止映射器漂移。这个取舍与以下原始工作提供的方向性证据一致：

- [NVIDIA Cross-Model KV Cache Transfer](https://arxiv.org/abs/2608.03893)：
  content-space/RoPE 处理、跨模型映射及 functional retention 的直接背景；
- [Mixture of Translators](https://arxiv.org/abs/2607.28979)：target trajectory /
  context-correction 目标支持从几何拟合转向 receiver 行为；
- [CacheBlend](https://arxiv.org/abs/2405.16444)：少量 receiver 重算可修复 cache
  误差，但它是不同设定，所以本项目只把 R32 当独立混合对照；
- [KVBoost](https://arxiv.org/abs/2608.21362)：偏差定位和选择性修复支持 OOD /
  deviation gate，不证明跨模型结果；
- [KV Packet](https://arxiv.org/abs/2604.13226)：小型 receiver adapter 配合
  self-distillation 的思想支持冻结主模型、训练小修复器；
- [Decoupled Attention Fusion](https://arxiv.org/abs/2607.21599)：attention-sensitive
  修复支持按功能层/token 定位，而不是平均优化所有 KV 元素。

这些论文只用于提出可检验假设，不能替代本模型对上的前瞻配对实验。

## 1.1 保留的后续低风险 ablation（不进入本版本）

有两个值得单独注册、但不能临时插入当前候选的低风险方向：

1. **scalar attention-logit calibration**：在冻结 affine 输出后，仅学习每层/K
   的一个标量温度或中心校正，使 `QK^T/√d` 的尺度与 native query 的尺度对齐；
   该标量可直接折叠进一次 affine 输出，几乎不增加推理矩阵乘法，但可能改变
   softmax 温度，必须先在 answer-free mapper-fit/method-dev 做稳定性和零校正
   测试。
2. **saliency-weighted ridge**：用 native query 的 `Q^TQ`（K）或 native attention
   read weight（V）构造通道/行权重，再解一个加权 ridge；它仍是单次 affine map，
   但权重来自功能信号而非普通 MSE。需要新的 capture、正则化和独立版本，不能
   在看过本版本 screen/confirm 后把它作为“修复”并入 residual。

这两个方向属于 future `v2` ablation；当前配置、参数表、STOP 收据和训练器均不
包含它们，避免把多个变化混成一个候选或事后挑选。

## 2. 冻结 anchor

唯一 anchor 是目录 02 已内容寻址的 `headwise_shrinkage_v1`：

```text
manifest bytes  = 39059
manifest sha256 = dabc63b252c5c7b9964fb00fdbe1b5eef440f126fb88c8d4e73a45378d5b4995
```

formal run 必须深度验证其 40 层 payload，且 residual 第 0 步精确返回 anchor。
若 anchor 校验失败，当前配置整体中止；不允许看过 method-dev 后切换 k4。k4 只
能在一个新版本中作为单独预注册的 control。这避免“哪个 base 结果好就用哪个”
带来的隐性选择偏差。

## 3. 层/KV/head 结构化低秩残差

对 target 层 `l` 和分量 `c ∈ {K,V}`，k20 source features 按冻结 source-layer
rank、KV head、head dim 展平：

\[
x_{l,c,t}\in\mathbb{R}^{20\times8\times128}=\mathbb{R}^{20480}.
\]

mapper-fit 统计给出 `μ`、`1/σ`，令 `z=(x-μ)/σ`。冻结 anchor 输出为
`Y0 ∈ R^(8×128)`。残差为：

\[
u=zA_{l,c},\quad A_{l,c}\in\mathbb{R}^{20480\times r},
\]

\[
r_h=uB_{l,c,h}+\Delta b_{l,c,h},\quad
B_{l,c,h}\in\mathbb{R}^{r\times128}.
\]

同一 layer/component 的八个 output KV heads 共享 encoder `A`，decoder `B_h`
按 head 分开；K 和 V 完全分开。`B` 与 `Δb` 零初始化，因此 step 0 与 anchor
逐元素相等。每个 head 有静态可学习 gate，但任何动态安全 gate 关闭时都精确
回到 anchor：

\[
\hat Y_h=Y_{0,h}+w_l\,g_{ood}(x)\,g_{cap}(r,Y_0)\,
\sigma(a_{l,c,h})\,r_h.
\]

`w_l` 是 train-only oracle 冻结的层 gain；当前实现把层的重要性主要用于 loss
加权，production 配置可把 gain 固定为 1，以免可辨识性重复。

## 4. 参数量与额外状态

每个 active layer 同时包含 K/V 两个模块，参数数为：

\[
2(20480r+8r128+8\cdot128+8)=43008r+2064.
\]

| active 层 | rank | 参数量 | BF16 residual weights | 16 B/param 训练状态上界 |
|---:|---:|---:|---:|---:|
| 6 | 64 | 16,527,456 | 31.524 MiB | 252.19 MiB |
| 6 | 128 | 33,042,528 | 63.024 MiB | 504.19 MiB |
| 6 | 256 | 66,072,672 | 126.024 MiB | 1008.19 MiB |
| 40 | 64 | 110,183,040 | 210.157 MiB | 1.642 GiB |
| 40 | 128 | 220,283,520 | 420.157 MiB | 3.282 GiB |
| 40 | 256 | 440,484,480 | 840.157 MiB | 6.564 GiB |

表中只是 residual 参数及保守 optimizer 状态，不含 anchor、14B weights、KV、
activations 或 allocator reserve。任何正式结果必须记录 `max_memory_allocated`、
`max_memory_reserved` 和实际权重 bytes，不能把这张表当 GPU 实测。

## 5. OOD 与残差幅度安全门

所有阈值仅由 mapper-fit capture 冻结。每个 token/layer/component 计算：

\[
d=\max(\operatorname{RMS}(z)/q^{RMS}_{.99},
       \max|z|/q^{max}_{.99}).
\]

- `d≤1`：OOD gate 完全打开；
- `1<d<1.25`：使用 `1-smoothstep((d-1)/0.25)` 平滑关闭；
- `d≥1.25` 或任何输入非有限：gate 为 0，精确返回 anchor。

每个 output head 还限制 `RMS(residual) ≤ 0.25×RMS(anchor)`；超出时只缩放
残差，不改 anchor。正式 capture 应同时报告 gate 分布、fallback rate、cap rate、
残差/anchor norm 分布，并做“去掉 OOD gate”“去掉 norm cap”两个消融。

## 6. 层选择不能来自答案或 locked outcome

预注册候选层顺序为 `[7,8,9,10,34,35]`，但这不等于预先断言这些层有效。
一个去污染、train-split-only oracle 必须逐层测：

1. native repair 相对 mapped cache 的 first-token KL 减少；
2. native-fixed-query attention-output cosine 增益；
3. centered attention-logit Huber 减少。

只保留 primary effect 为正且 paired bootstrap 95% LCB 大于 0 的层；正 effect
裁到 `[0.25,2]` 后归一到均值 1，成为 loss weights。oracle 记录必须明确：未用
answer、gold margin、F1、EM、accuracy、生成结果或 locked/test outcome。若没有层
通过，方法直接 NO-GO；不能凭以往测试掉点手选 34/35。

## 7. 功能损失

### 7.1 固定 native query 的 attention logit/output

对 native-14B query `Q*`、native cache `(K*,V*)` 与候选 cache `(Khat,Vhat)`：

\[
S^*=Q^*K^{*T}/\sqrt d+M,\qquad
\hat S=Q^*\hat K^T/\sqrt d+M.
\]

logit loss 对每个 query row 先减去合法 key 上的均值，再做 Smooth-L1，避免惩罚
softmax 不敏感的常数平移。Qwen3 GQA 的每个 KV head 对应的全部 query heads都
进入计算。输出为 `O=softmax(S)V`，使用 `1-cos(O*,Ohat)` 与
`||O*-Ohat||²/(||O*||²+ε)`。长序列只取预注册 query rows，不物化完整 T²。

### 7.2 8→32 步 native-14B suffix KL

teacher 与 candidate teacher-force 完全相同的 suffix tokens：teacher 使用
native-14B prefix cache，candidate 使用 corrected mapped cache。方向固定为：

\[
\tau^2 KL(p_{14,native}^{\tau}\Vert p_{14,mapped}^{\tau}).
\]

训练 curriculum 先 attention，再 suffix-8，最后 suffix-32，温度固定 1 与 2
取均值。32B logits 明令禁止作为 teacher，否则研究问题变为 capability
distillation。

### 7.3 EOS soft BCE 与 KV guardrail

EOS 使用 native-14B teacher 的 EOS 概率作 soft target，对 candidate EOS logit
做 BCE。KV loss 是按 native KV 方差归一的 Huber，仅以 0.02 权重稳定训练；
held-out K/V R² 相对 anchor 任一下降超过 0.002 即 NO-GO，即使功能 loss 更好。

## 8. 训练、筛选与确认

1. mapper-fit capture 拟合 standardization、OOD quantile 和 residual 参数；
2. 三个 rank 按 64→128→256 独立从同一 anchor、同一 seed policy 训练；
3. method-dev screen 只用允许的 answer-free functional metrics，选择注册顺序中
   第一个过门槛的 rank，不按最优点 cherry-pick；
4. 被选的唯一 rank 只进入一次 method-dev confirm；confirm 失败则本方法版本
   失败，不能改选另一 rank；
5. 通过后才冻结唯一 R1 candidate；再经过完整 G0 与独立审计后，才可一次打开
   locked-test。

checkpoint 必须原子提交并绑定 config/code/data/anchor/oracle/capture SHA；resume
只接受完整验证的 checkpoint，损坏文件不覆盖。

## 9. R1 与 R32 是两类系统

- `functional_residual_r1`：只保留既有 boundary token replay，不做 tail refresh，
  是与 NVIDIA-compatible k20 R1 的主比较；
- `functional_residual_r32`：可选地让 14B 重放 32 个 prompt tokens，是额外
  target compute 的 hybrid 质量—延迟点。

R32 必须使用同一个已冻结 mapper，单列 target replay time、TTFT 和 VRAM；不能
用 R32 的提高挽救 R1，也不能合并成“同成本超过 NVIDIA”。

## 10. 当前尚未完成的生产边界

本目录尚缺：真实 mapper-fit functional capture、逐层 train-only oracle receipt、
完整 G0-v2/registry binding、14B differentiable receiver adapter、八卡显存预检、
GPU 训练、R1/R32 prospective evaluation 和 locked confirmatory evidence。任何一个
缺失都不允许比较性结论。
