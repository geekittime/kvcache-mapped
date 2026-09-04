# 32B→14B KV 缓存复用：文献、机制与可证伪的新方法

更新时间：2026-09-01（研究记录，不是结果声明）

## 摘要

本项目要解决的不是“把 32B 的 K/V 坐标回归得更像 14B”，而是：在不重新
prefill 全部上下文的条件下，让冻结的 Qwen3-14B 继续做出与其 native cache
尽可能一致的决策和生成。当前 NVIDIA 基线是 content-space、逐 head 的 top-k
跨层 ridge mapper；本地反向 32B→14B 结果显示，几何拟合与下游行为并不等价，
尤其 HotpotQA、CoQA 和 GSM8K 的损失集中在答案边界、EOS 和 query-sensitive
方向。

本记录提出一个可逐层否证的路线：

1. 以 sequence-CV 的 shrinkage ridge 作为稳定 anchor；
2. 只学习零初始化、低秩、可回退的功能残差；
3. 用 native-14B query 的 attention/logit/output 及短 suffix trajectory 监督，
   而不是继续最大化未加权 KV MSE；
4. 把 target-side tail replay（R=8/32）作为单独的质量—延迟点；
5. 用不读答案的预注册 oracle 选层，再在去污染 method-dev 做唯一一次确认，
   最后才打开 locked test。

“超过 NVIDIA”只有在冻结、同 prompt、同 decode、同样本、同 R=1 成本下的
paired 结果满足预注册门槛时才成立。本文档中的设计均不等于已经超过。

## 1. 问题形式化

给定同一 token 序列 `x[0:T)`：

- 32B source 产生每层的 content-space `K_s,V_s`；
- mapper `f` 输出 14B 可执行的 `K_hat,V_hat`；
- 14B decoder 使用 `K_hat,V_hat` 跳过 prefix prefill，然后生成 suffix。

目标是同时控制四种误差：

\[
  \mathcal E_{geom}=\|K^*-\hat K\|,\qquad
  \mathcal E_{att}=D(\operatorname{softmax}(Q^*K^{*T}),
                         \operatorname{softmax}(Q^*\hat K^T)),
\]
\[
  \mathcal E_{traj}=\sum_t KL(p^*_{t}\Vert \hat p_t),\qquad
  \mathcal E_{term}=|p^*_{EOS}-\hat p_{EOS}|.
\]

其中 `*` 始终代表 native 14B；不能用 32B logits 作 teacher，否则问题会变成
能力蒸馏而不是 cache fidelity。最终质量由 paired QA/多选/生成指标决定，
而不是由任何单一误差代理决定。

## 2. NVIDIA 基线及其可改进处

原论文为 [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/abs/2608.03893)。
论文明确采用：

- 每个目标 layer/head 独立的 centered ridge；
- 按单源层 `R²` 选择 top-k source layers；
- 对 K 去 RoPE，在 content space 拟合，再按目标位置加 RoPE；
- 500 个 FineWeb-Edu、每条 1024 token 的 calibration；
- 推理时一次 mapper matmul，避免目标模型 prefix prefill。

论文的公开观察对新方法很关键：多源层使 Qwen3 14B→32B 的单层 K/V 解释率
从约 56%/32% 提升到约 79%/65%；但 across-pair 的 attention-output cosine
与 retention 相关系数约为 `+0.57`，明显比简单 calibration `R²` 更有用；困难
pair 上 MLP 可改善，但容易在已经成功的 pair 上退化。论文还指出误差的方向/位置
比误差总量更重要。这些是“改功能目标”的动机，不是本项目结果。

本地 reverse 结果为同一方向的直接警示：

| 证据 | 观察 | 含义 |
|---|---:|---|
| k20 held-out K/V R² | 0.7450 / 0.6516 | 坐标拟合仍有明显残差 |
| k20 train→held-out gap | 9.87 / 13.66 pp | 大输入特征有泛化风险 |
| shrinkage held-out K/V | 0.7655 / 0.6832 | 简单收缩可改善几何泛化 |
| HellaSwag | −1.095 pp | 低 margin 决策容易翻转 |
| HotpotQA | 约 −8.8 pp F1 | query/readout 与答案边界失稳 |
| GSM8K numeric-final | −4.625 pp | strict-match 额外受格式/EOS影响 |
| GSM mapped 长尾 | median 623 vs target 102 token | runaway 不是单纯 KV MSE 可见 |
| CoQA | −5.35 pp F1 | 很多答案仍被包含，但边界/冗余变差 |

因此“继续加 k 或只调 ridge lambda”不是充分方案。新方法必须至少改善
attention/readout 或终止稳定性，同时保持几何 guardrail。

## 3. 文献调研与可迁移原则

以下只引用论文公开的机制；模型、数据和成本不同的结果不能直接外推到
Qwen3 32B→14B。

### 3.1 直接跨模型映射

- **NVIDIA ridge mapper**：见上节。其最强可迁移点是 RoPE/content 分解和
  top-k 跨层输入；最明显空缺是主损失仍是 K/V 回归。
- **Cache-to-Cache (C2C)**：
  [arXiv:2510.03215](https://arxiv.org/abs/2510.03215)。用投影、融合和可学习
  layer gate 做跨模型语义通信。它支持“zero-init residual + receiver anchor”
  的工程想法，但其模型/任务不是本项目的严格同成本设定。
- **Mixture-of-Translators (MoT)**：
  [arXiv:2607.28979](https://arxiv.org/abs/2607.28979)。用多个 translator
  表达多模态映射，并以 Context Correction Loss 对齐 target trajectory；作者
  还区分 early propagated shift 与 late last-state shift。可迁移的假设是
  “trajectory correction 比单纯 coordinate regression 更接近最终质量”，但
  多 translator 的额外推理成本必须单独核算。

### 3.2 选择性重算和边界修复

- **CacheBlend**：[arXiv:2405.16444](https://arxiv.org/abs/2405.16444)。
  对重用 chunk 选择少量 token 做 receiver recompute，目标是修复跨 chunk
  attention，而非重算全部上下文。它报告在其同模型 RAG 设定中兼顾 TTFT/质量；
  本项目将其缩减为 R=8/32 的 target-tail replay 对照，不能把同模型结果当跨模型
  证明。
- **KVBoost**：[arXiv:2608.21362](https://arxiv.org/abs/2608.21362)。
  用 deviation-guided probe、边界重算和 chunk 策略定位高偏差 token。可迁移
  的原则是运行时 gate 应看功能偏差，而不是固定扩大 replay 窗口。
- **Decoupled Attention Fusion (DAF)**：
  [arXiv:2607.21599](https://arxiv.org/abs/2607.21599)。把重要 token 的
  self-attention、question-document attention 与 state fusion 解耦，以较少
  重算恢复缺失交互。它提示未来可做 attention-sensitive sparse refresh；但
  DAF 的 RAG chunk 假设不同，本轮不直接移植。
- **KV Packet**：[arXiv:2604.13226](https://arxiv.org/abs/2604.13226)。
  用小型 soft-token adapter 和 self-distillation 连接上下文不连续处。可迁移
  的设计约束是冻结主 decoder、让小修复器学习行为；不采用其具体 packet 格式。

### 3.3 Distillation / compression 的功能目标

- **KV-Distill**：[arXiv:2503.10337](https://arxiv.org/abs/2503.10337)。
  用 teacher-student 的输出 KL 训练可压缩 cache，强调 question-independent
  context compression。它支持 suffix-logit KL 作为功能监督。
- **KVSculpt**：[arXiv:2603.27819](https://arxiv.org/abs/2603.27819)。
  直接优化连续 KV 以保留 attention behavior，并按 layer/head 难度分配预算。
  可迁移的启示是 per-layer/head difficulty 很不均匀，不能用一个全局 MSE 权重。

### 3.4 综述性结论

文献共同支持一个“几何 anchor + 功能 correction + 稀疏 receiver refresh”的
组合，但没有一篇论文证明它在本项目的 Qwen3 reverse pair、同成本 R=1、HotpotQA
上一定更好。因此所有引用只进入假设注册，不进入结果列。

## 4. 候选方法：FAR-KV（Functional Anchor-and-Repair KV）

暂定名称 **FAR-KV**，包含三个可分离部件：

### 4.1 稳健 anchor

从已冻结的 k20 mapper 与其前四个完全一致的 k4 子空间构造 sequence-CV
shrinkage：

\[
  y_0=(1-\alpha)y_{k20}+\alpha y_{k4},\quad \alpha\in[0,1].
\]

每个 target layer、K/V、KV head 单独估计 `α`；按完整序列做 group CV，不能把
同一序列的 token 随机拆到 train/validation。融合后仍为一个 k20-shape affine
matmul，R=1 latency 理论上与 k20 相同。

### 4.2 零初始化低秩功能残差

令 `z` 是冻结的 layer/component 标准化输入，
`A∈R^(20480×r)` 为共享 encoder，`B_h∈R^(r×128)` 为每个 KV head 的 decoder：

\[
  y_h=y_{0,h}+g_{ood}(z)g_{cap}(r_h,y_{0,h})\,B_h^T A^Tz.
\]

`B` 和 bias 零初始化，故 step 0 **逐元素等于 anchor**；若 OOD 或 residual
norm 超阈值，平滑 gate 精确回退。先按预注册顺序试线性 rank 64、128、256；不
直接复制论文的 1024×2 MLP，因为本地成功 pair 上大 MLP 可能过拟合，且存储/延迟
不再可比。只有线性残差失败才开一个全新版本尝试非线性。

### 4.3 功能损失

对 native 14B 固定 query：

1. centered attention-logit Smooth-L1（减去每行合法 key 的均值）；
2. attention output cosine + normalized MSE；
3. suffix 8/32 步 native-target KL（temperature 1/2）；
4. EOS/stop token soft target；
5. normalized KV Huber 只作 guardrail。

长序列不物化完整 `T×T` 矩阵，只采样预注册的 tail、chunk boundary、uniform 和
高注意力行，并保存采样覆盖证明。任何 target logits 都来自 native 14B；不读
答案、不用 QA 分数选层。

### 4.4 可选 target-side replay

`R=1` 只代表纯 mapper（含已有 boundary 机制）；`R=8`、`R=32` 是额外 target
compute 的独立 hybrid。先生成 R=1，如果无答案-free runtime gate 触发条件（如
`generated_tokens>32` 或 `cap_hit`），再对同一 prompt 做 R=32；不得把两次结果
合成“同成本”结论。未来可将固定阈值替换为 KVBoost 风格 deviation probe，但
必须另建版本并保存 terminal event trace。

## 5. 为什么这些部件可能减少掉点（可证伪推理）

| 失败模式 | 机制假设 | 可观测验证 |
|---|---|---|
| k20 高维外推 | shrinkage 降低方差、保留低频信号 | held-out R²、attention KL |
| K 误差落在 query-sensitive 方向 | fixed-Q loss 改变误差方向 | attention logit JS/cos |
| late-layer last-state shift | 34/35 或 oracle 层的残差 correction | suffix KL、first-token margin |
| mapped 冗长/不终止 | EOS soft loss + tail replay | EOS rate、cap rate、长度分布 |
| Hotpot 答案被包含但边界错 | trajectory/stop supervision | normalized containment 与 F1 差值 |
| 长上下文边界错 | R32 稀疏 receiver replay | F1 vs replay cost 曲线 |

若 attention cosine 提高而下游 paired CI 不提高，假设即被否证；若只改善 strict
格式而 numeric 不变，只能报告 decoding/format 修复，不能称 cache mapper 超越。

## 6. 严格实验设计

### 6.1 三层数据隔离

1. **mapper-fit**：通用文本；只拟合 anchor、标准化和残差。
2. **method-dev**：去污染 HotpotQA/MuSiQue train；只做 answer-free oracle、
   rank/layer 选择和 confirm。
3. **locked-test**：候选冻结后只跑一次 HotpotQA dev、MuSiQue dev 及后续任务。

完整 G0 必须证明 exact ID、normalized text、substring 和 near duplicate 均为
零；历史已经看过的 Hella/GSM/CoQA/Hotpot 结果只作 baseline，不能用来选超参。

### 6.2 固定比较单元

每个样本必须四路绑定同一个：

- token IDs、chat/no-thinking prompt、position IDs；
- target 14B weights/tokenizer；
- source 32B weights 与 mapper；
- greedy decode、cap、stop rule；
- checkpoint sample ID、record SHA、generated token IDs、stop reason。

比较至少包括 native14、NVIDIA k20、shrinkage、FAR-KV R1；R32 单列。输出
Hotpot normalized answer F1/EM、MuSiQue F1/EM、长度/EOS/cap、first-token KL、
target decision agreement 和 target-correct retention。

### 6.3 统计与门槛

- 二元多选：paired McNemar + exact 95% CI；
- F1/ROUGE：按 conversation/document cluster bootstrap；
- PPL：先聚合 token NLL，再 bootstrap，不直接对 PPL 做 t-test；
- 五任务多重比较：Holm 校正；机制次指标用 BH-FDR；
- 纯 R=1 候选至少在三个关键 QA/推理任务中两个达到 paired CI 下界 >0，且
  不劣化 margin：Hella/MMLU 0.5 pp、GSM/Hotpot 1 pp、CoQA 1.5 pp；
- target-correct retention 不得下降 0.5 pp；
- K/V held-out R² 相对 anchor 下降不得超过 0.002；
- mapper 权重和实测 latency/VRAM 必须在同一 NVIDIA k20 预算内。

任何门槛失败都保留完整 receipt，并把版本标为 NO-GO；不能从已看过的 locked
结果反向调参。

## 7. 试验顺序与停止规则

```text
文献/本地证据
      ↓
anchor shrinkage（低风险、R=1）
      ↓
answer-free layer oracle
      ↓
FAR-KV rank 64 → 128 → 256（固定顺序）
      ↓
method-dev screen → 唯一 confirm
      ↓
独立 G0 + prospective locked R=1
      ↓
另报 R=8/R=32 质量—延迟 Pareto
      ↓（仅若失败）
新目录尝试 nonlinear / deviation-triggered repair
```

停止条件：出现非有限 loss、anchor bit mismatch、跨 tier collision、GPU lock
失效、完整性 receipt 缺失或任何 outcome 泄漏时立即 fail closed。不得因为某个
任务净准确率上升就跳过 target-correct retention 检查。

## 8. 当前实施状态

- `02_shrinkage_mapper`：已完成 mapper-fit/held-out 几何验证；尚无下游提升证明。
- `03_tail_refresh`：R8/R32 exploratory pilot 未满足全数据集 promotion rule，
  只能作假设证据。
- `12_prospective_screen_v2`：通用 GPU runner 已通过 CPU/攻击审计，但要等最终
  去污染 registry 与 G0 receipt 才能生成 formal config。
- `13_oracle_decontaminated_v2`：answer-free oracle 合同 PRE-FORMAL PASS，
  尚未做真实模型运行。
- `14_functional_residual_mapper`：正在独立目录中完成 CPU 协议、训练器和
  preflight；尚无 GPU、训练矩阵或准确率结果。
- `10_decontaminated_method_dev_registry`：pairwise-v3 graph 已发布；deep
  重建与后续 allocation 必须顺序完成，不能消费尚未通过 deep 的 registry。

## 9. 不夸大的结论

截至本记录日期，没有证据可以宣称任何新方法已经超过 NVIDIA。已知最强的
可执行路线是“shrinkage anchor + functional low-rank correction + 独立 tail
replay”，但它仍是待验证假设。最终汇报必须把：

1. 几何指标；
2. native-behavior 指标；
3. 下游 paired 结果；
4. 额外计算/延迟；
5. 数据隔离和失败尝试

分别列出，避免把 R²、PPL、strict parser 修复或 exploratory n=48 结果冒充为
准确率提升。

## 参考文献（原始论文）

1. Heo et al., *Cross-Model KV Cache Transfer in LLM Families*, arXiv:2608.03893.
2. Lee et al., *Mixture-of-Translators*, arXiv:2607.28979.
3. Fu et al., *Cache-to-Cache*, arXiv:2510.03215.
4. Yao et al., *CacheBlend*, arXiv:2405.16444.
5. Unnikrishnan, *KVBoost*, arXiv:2608.21362.
6. Wu et al., *Decoupled Attention Fusion*, arXiv:2607.21599.
7. Chen et al., *KV Packet*, arXiv:2604.13226.
8. Chari et al., *KV-Distill*, arXiv:2503.10337.
9. Jiang & Jin, *KVSculpt*, arXiv:2603.27819.

## 附录 A：挑战性数据集取舍

本轮不为了“找一个更容易赢的集合”而换测试集。数据集的作用是覆盖不同的
cache failure mode，并在候选冻结后一次性锁定：

| 数据集 | 适合检验的现象 | 本轮角色 |
|---|---|---|
| HotpotQA distractor | 10 段、多跳、实体/比较题、答案边界 | 主指标；全量 dev 已绑定 |
| MuSiQue | 组合式多跳、减少单跳 shortcut | method-dev/locked 配对；已绑定官方 train/dev |
| 2WikiMultiHopQA | 结构化+非结构化、最多多跳路径 | 后续独立扩展；不与当前 registry 混用 |
| LongBench v2 | 8k–2M words、长对话/代码/结构化理解 | 仅在长上下文 runner 稳定后选固定子集 |
| MoreHopQA | 在已有多跳上叠加 commonsense/算术/符号推理 | 后续压力测试；需要单独 G0 和答案协议 |
| QASPER | 论文全文信息寻求、证据跨段落 | 长文档质量/引用一致性扩展，不用于当前主门槛 |

HotpotQA 原始论文说明其问题需要多个 supporting documents，并提供 sentence-level
supporting facts；MuSiQue 的公开代码和论文强调从 single-hop composition 构造
多跳问题。2WikiMultiHopQA 的设计加入 reasoning path，专门减少“看一个段落就猜到
答案”的捷径。[HotpotQA 原论文](https://arxiv.org/abs/1809.09600)、
[2WikiMultiHopQA 原论文](https://arxiv.org/abs/2011.01060)、
[LongBench v2](https://arxiv.org/abs/2412.15204)、
[MoreHopQA](https://arxiv.org/abs/2406.13397)、
[QASPER](https://arxiv.org/abs/2105.03011)是数据定义依据。

当前优先 HotpotQA 的原因是本地已观察到最大功能掉点，且其官方答案/支持事实
可复算；MuSiQue 作为第二个不同构造的多跳集合，能检验方法是否只修复 Hotpot
的模板。LongBench v2 和 MoreHopQA 需要重新做完整的三层去污染、prompt/token
边界和评测协议，不能在已有结果上临时追加，否则会把筛选集变成隐性测试集。

## 附录 B：文献结论的可检验翻译

把论文中的方法名直接拼在一起容易产生“看起来先进、实际不可归因”的系统。
更严格的翻译如下：

| 文献观察 | 可移植变量 | 本项目的最小检验 |
|---|---|---|
| NVIDIA：attention-output 比 R² 更预测 retention | loss 的目标空间 | fixed-native-Q cosine/JS 与 paired F1 是否同向 |
| MoT：early shift 与 late shift 是两种失效 | 注入位置/层分块 | 0–9、10–19、20–29、30–39 及 34/35 的 oracle 干预 |
| CacheBlend/KVBoost：边界少量重算有效 | replay budget 与触发器 | R1、R8、R32 单独测，记录 target FLOPs/TTFT |
| C2C：融合而非单投影 | anchor + residual/gate | zero-init residual 是否不破坏 anchor |
| SemPIC/KV Packet：冻结 reader、训练 writer/adapter | offline 小适配器 | residual 只改 mapper，14B 权重与 decode 不变 |
| KVSculpt：layer/head 难度高度不均匀 | 细粒度权重/预算 | 每层每 head 的 attention loss 与 fallback 分布 |
| IAM/DroidSpeak：attention 相似性可跨模型/变体 | attention map 作为信号 | 仅作机制对照，不把同模型结果当跨尺寸证据 |

这张表也定义了否证方式：若 fixed-Q 功能指标变好、但 native 14B 的 paired
answer 指标不变或变差，不能继续扩大 residual；若 R32 变好但 R1 不变，只能
报告质量—延迟 trade-off；若只在已暴露数据上变好，则整条比较链无效。

## 附录 C：从“论文复现”到“超过论文”的可审计差异

论文的公开结论是跨六个 matched-KV pair 的线性 mapper，以及困难 pair 上的
非线性 MLP 对照；它没有给出本项目全部 Qwen3-32B→14B 任务的 bit-exact
目标表。因而本地实验应采用三列而非单列排名：

1. **质量**：native 14B、NVIDIA k20、候选方法的逐样本结果与置信区间；
2. **功能机制**：attention JS/cosine、first-token KL、suffix KL、EOS/cap；
3. **系统成本**：mapper bytes、额外 target replay、TTFT、峰值显存。

只有质量列在预注册非劣门槛内、功能列解释改善方向、系统列仍满足成本约束，
才可把候选称为 Pareto 改进。任何单独的 R²、PPL 或 strict parser 改善都不够。

## 附录 D：2026-09-02 调研增量与新的可证伪方向

这次增量检索没有把同模型 KV 压缩论文的数字直接外推到跨模型映射，而是提取
它们对“哪里值得修”和“何时值得重算”的共同结构。

1. **层/头预算不应均匀。** PyramidKV 把层深度与信息汇聚联系起来，主张按层
   分配不同缓存预算；SnapKV、H2O 及后续 head-wise 方法则用近期 query 的注意力
   估计 token 重要性。对本项目的直接翻译不是删掉 32B token，而是让残差容量、
   replay 预算和 loss 权重按 target layer/KV head 分配。参考：
   [PyramidKV](https://arxiv.org/abs/2406.02069)、
   [SnapKV](https://arxiv.org/abs/2404.14469)。

2. **修复触发器应来自偏差，而非固定长度。** CacheBlend 的核心是只重算少量
   跨块交互敏感 token；KVBoost 进一步用 probe/deviation 选择边界或高偏差 token。
   这支持本项目的两阶段策略：先生成纯 mapper R=1，再用答案盲的 attention-logit
   偏差或终止异常触发 R=8/32；固定每个请求重算 R=32 不是同成本比较。参考：
   [CacheBlend](https://arxiv.org/abs/2405.16444)、
   [KVBoost](https://arxiv.org/abs/2608.21362)。

3. **目标应是 receiver trajectory。** MoT 明确区分 early propagated shift 与
   late last-state shift，并引入 Context Correction Loss；KV-Distill/KVSculpt
   也把输出行为或 attention 行为作为压缩约束。对 32B→14B，最小可行翻译是冻结
   14B，只训练小的 residual/gate，使用 native-14B fixed-Q、首 token margin、
   8/32-step suffix KL 和 EOS loss；32B logits不能作为 teacher。参考：
   [Mixture-of-Translators](https://arxiv.org/abs/2607.28979)、
   [KV-Distill](https://arxiv.org/abs/2503.10337)、
   [KVSculpt](https://arxiv.org/abs/2603.27819)。

4. **本项目新增的最小实验假设（FAR-KV-2）：**

   - 先用 head-wise shrinkage 作为 anchor，保持 NVIDIA k20 的输入/输出几何；
   - 对每个 layer/head 学一个低秩 residual，但用 native attention read-weight
     及 centered logit 误差加权，而不是均匀 KV MSE；
   - 用一个只依赖运行时状态的 deviation probe（不读答案、预测文本或数据集标签）
     决定是否做 R=8/32 tail replay；
   - 对 34/35 等高功能误差层设置更高 residual 上限，同时保留 OOD 时的 exact
     anchor fallback。

   这不是已证明的算法。其可证伪条件是：在未观察的 method-dev 上，fixed-Q
   attention JS/首 token KL 和终止率同时改善；若 paired HotpotQA/MuSiQue F1
   不改善，或者额外 replay 成本超过预注册上限，则 FAR-KV-2 淘汰。这样可把
   “代理指标变好但准确率不变”的情况明确判为失败，而不是继续调参。
