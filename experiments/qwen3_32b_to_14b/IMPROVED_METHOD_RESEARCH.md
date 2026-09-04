# Qwen3 32B→14B KV Cache 映射：现状、失败分析与低损失改进方案

更新日期：2026-08-30

## 1. 研究目标与证据边界

本项目的直接目标是：将 Qwen3-32B 已计算的 prompt KV cache 映射为 Qwen3-14B 可继续使用的 KV cache，在减少 14B 重复 prefill 的同时，尽可能保持原生 Qwen3-14B 的行为和准确率。

全文使用以下标签区分证据强度：

- **[论文结果]**：原论文或相关论文已经公开报告的结果。
- **[本地实测]**：本项目已完成、且通过产物哈希和原始样本复算的结果。
- **[研究假设]**：由机制证据支持、但尚未在本地 Qwen3-32B→14B 设置中验证的方案。

需要特别强调：原论文没有公开 Qwen3-32B→14B 在 ARC、MMLU、GSM8K、HotpotQA、MultiNews、LMSYS、ShareGPT4 等任务上的完整正式结果表。本项目在这些任务上的数值属于**论文协议对齐的本地扩展实验**，不能写成论文原表的逐数值复现。

本研究首先追求“对原生 14B 的功能保真”。如果以后希望把 32B 的能力蒸馏给 14B，必须另建实验、另设 teacher 和评价口径，不能与当前 fidelity 目标混在一起。

## 2. 当前论文方法与本地复现基线

### 2.1 基线方法

当前 production mapper 沿用论文的大模型到小模型协议：

- source：Qwen3-32B，64 个 decoder layers；
- target：Qwen3-14B，40 个 decoder layers；
- 每个 target layer 对 K、V 分别选择 20 个 source layers；
- 每个 source layer 的 KV 宽度为 `8 heads × 128 = 1024`；
- 每个 K/V 映射为 `20480 → 1024`；
- 训练样本为约 128,000 个 token observations；
- centered ridge，固定 `lambda=0.01`；
- K 在 content space 中拟合，即先移除 source RoPE，映射后再施加 target RoPE；
- FP32 covariance、FP32 mapper、FP64 诊断；
- 40 层 K/V 总权重数为 1,677,721,600，bias 为 81,920；
- production FP32 mapper 约 6.2503 GiB。

运行时流程为：32B 完成 prompt prefill → 取出 64 层 KV → K 逆 RoPE → 按每个 14B layer 的 top-20 source layers 拼接 → K/V 线性映射 → K 施加 14B RoPE → 构造 14B cache → 14B replay 边界 token 并继续生成。

### 2.2 已完成的正式本地结果

下表是严格重算后的正式本地五任务扩展结果。`Mapped/Target` 用于观察映射后相对原生 14B 的保留率；PPL 越低越好。

| 数据集 | 指标 | 原生 14B | 32B→14B mapped | 变化 |
|---|---:|---:|---:|---:|
| ARC-Challenge | acc_norm | 60.0683% | 60.7509% | +0.6826 pp |
| HellaSwag | acc_norm | 78.6995% | 77.6041% | -1.0954 pp |
| WinoGrande | acc | 72.9282% | 71.9811% | -0.9471 pp |
| MMLU | acc | 78.8064% | 79.7963% | +0.9899 pp |
| GSM8K | numeric-final | 94.9962% | 90.3715% | -4.6247 pp |
| WikiText-2 | PPL ↓ | 7.5020 | 7.8022 | +4.00%（较差） |

准确率偶然升高不等于映射忠实。例如 ARC 和 MMLU 的总体准确率略升，但 target/mapped 的选择仍发生大量翻转。因此，后续不能只看最终 accuracy，还必须报告 target decision agreement、target-correct retention、gain/regression flip matrix 和 logit divergence。

本地正式产物：

- [五任务最终汇总](/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b/runs/five_task_extension_full_k20_v1/five_task_extension_summary.json)
- [HellaSwag 机制分析](/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b/runs/formal_pipeline_v1/mechanism_hellaswag_k20/mechanism_result.json)
- [held-out KV 重建结果](/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b/artifacts/validation/qwen3_32b_to_14b/heldout_activation_reconstruction.json)
- [CoQA k20 原始记录](/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b/runs/formal_pipeline_v1/coqa_reverse_k20/raw_records.jsonl)
- [长上下文 smoke 汇总](/home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b/runs/long_context_extension_smoke_k20_v1/summary.json)

## 3. 当前准确率损失的根因

### 3.1 calibration 拟合存在明显泛化差距

| Mapper | K train R² | K held-out R² | gap | V train R² | V held-out R² | gap |
|---|---:|---:|---:|---:|---:|---:|
| k20 | 0.8437 | 0.7450 | 0.0987 | 0.7888 | 0.6522 | 0.1366 |
| k4 | 0.7750 | 0.7422 | 0.0328 | 0.7015 | 0.6587 | 0.0428 |

`k20` 的训练 R² 更高，但 train/held-out gap 明显更大。`k4` 在 25/40 个 target layers 上甚至取得更高的 held-out V R²，平均高约 0.00646；然而 CoQA 上 k20 F1 仍高于 k4（72.8731% 对 72.5194%）。这说明：

1. 增大 k 会增加拟合容量，也会放大过拟合和病态条件问题；
2. KV reconstruction R² 不是最终功能质量的充分指标；
3. 不能简单把 k20 替换为 k4，需要以功能损失决定正则化与容量。

当前 ridge 还使用未归一化特征和固定 `lambda=0.01`。K Gram diagonal 的最大/最小比值中位数约 `4.18e8`、最大约 `2.78e9`，V 的中位数约 `4081`。固定 lambda 在不同层、不同 head、K/V 上对应的有效正则强度相差很大，因而有必要测试 feature standardization、whitening 和分层/分头 lambda。

### 3.2 KV R² 与实际 attention 功能不一致

HellaSwag 机制实验的本地结果：

- K content-space R²：0.6951；
- K RoPE-space R²：0.7483；
- V R²：0.6174；
- attention-output cosine：0.7611。

最差 attention-output cosine 出现在 layers 34、35、7、8、37，分别约为 0.316、0.348、0.410、0.446、0.473；这些层的 KV R² 并不一定最低。跨 40 层，attention cosine 与 K/V R² 的相关系数约为 -0.235/-0.306；跨 320 个 layer×KV-head 也只有约 -0.091/-0.148。

这表明当前目标函数存在根本错位：欧氏空间里“重建得像”，不等于经过 14B query、softmax 和 value read 后“功能相同”。原论文报告 attention-output cosine 与 retention 的跨 pair 相关性高于 calibration R²，但这仍只是一项相关性证据，不代表直接优化 cosine 一定提高准确率；需要本地因果消融验证。

### 3.3 自回归边界和 EOS 失稳是 GSM8K 的主要失败模式

GSM8K 上：

- numeric-final：94.9962% → 90.3715%；
- canonical strict：92.7976% → 64.2153%；
- mapped 平均输出约 662 tokens，target 约 147 tokens；
- mapped 中约 338 条输出达到或接近 1,000 tokens，target 约 32 条；
- 87 条 target 正确、mapped 错误的样本中，66 条 mapped 输出达到或接近 1,000 tokens；
- mapped 输出中出现 `</think>` 199 次，target 为 0；
- 351 条 mapped 样本 numeric-final 正确但 strict-match 错误，主要来自冗长和答案格式漂移。

所以 GSM8K 不只是“数学能力下降”，而是映射后首步状态、停止概率和生成轨迹偏移，随后误差自回归累积。仅继续提高 KV R² 很可能无法解决这个问题。

### 3.4 CoQA 的主要损失是回答边界漂移

CoQA F1 从 78.2213% 降至 72.8731%，下降 5.3481 pp。target/mapped cap 数分别为 91/232，但在双方都没有触发 cap 的 223 对样本中仍下降约 5.69 pp，因此长度不是唯一原因。

在 72 个“target 满分、mapped 非满分”的样本中，59 个 mapped 文本实际上包含某个合法参考答案。对两边公平执行“逗号前短语”诊断时，差距可从约 5.35 pp 缩至约 2.03 pp；这是强诊断证据，但不是官方 CoQA 评分，也不能直接作为正式方法或结果。更合理的做法是训练 EOS/答案边界，或者预注册短答案 prompt/stop protocol。

### 3.5 top-k 选层不是当前最高优先级

本地 40 层中，K、V、joint score 的 top-1 source layer 完全一致；joint top-20 与 K/V 独立 top-20 的平均重叠分别约 16.75/16.2。单独对 K/V 选层的代理提升只有约 0.00335/0.00404。

因此，可以测试 joint greedy selection 以减少冗余 source layers，但预计收益小于功能对齐、正则化和 native target tail refresh。

## 4. 改进方法：AAR-KV

建议将主方法命名为：

> **AAR-KV: Attention-Aligned Residual KV Translation with Native Target Refresh**

它由三个层次组成：稳健线性基座、功能对齐残差、可选 native target tail refresh。前两项保持纯 mapper 路径；第三项是质量—延迟可调的系统模式，必须单独报告。

### 4.1 零额外运行成本的 k4/k20 shrinkage

先用已有 k4 和 k20 mapper 构造验证集 shrinkage：

\[
\hat C_\alpha = \hat C_{20}+\alpha(\hat C_{4}-\hat C_{20}),
\]

其中 k4 权重按 k20 source-layer 排列补零。对每个 target layer、K/V，必要时细化到 head，在严格独立的 sequence validation set 上求闭式最优 `alpha`：

\[
\alpha^*=\frac{\langle C-\hat C_{20},\hat C_4-\hat C_{20}\rangle}
{\|\hat C_4-\hat C_{20}\|_2^2}.
\]

然后预融合：

\[
W_\alpha=W_{20}+\alpha(W_{4\rightarrow20}-W_{20}).
\]

部署时仍是一次 k20 matmul，不增加 mapper 参数、显存或在线延迟。这是最便宜、最先应该验证的改进。

**[研究假设]** shrinkage 能利用 k4 更好的泛化和 k20 更高的容量，降低 held-out 与下游损失；尚未在本地正式任务上验证。

### 4.2 Ridge-anchored 低秩功能修正

以现有 ridge 权重 `W0` 为固定基座，学习低秩修正：

\[
W = W_0 + AB,
\]

其中 `A∈R^(20480×r)`、`B∈R^(r×1024)`。初始化时 `A` 使用小尺度正交/随机初始化、`B=0`，保证训练起点严格等于 ridge baseline。训练后把 `AB` 融合回 `W`，因此部署仍是同一个 20480→1024 线性映射，在线参数量、显存和 latency 不增加。

对 40 层、K/V、`r=64`，可训练参数约 110,100,480，BF16 约 0.205 GiB；Adam optimizer 和梯度会增加训练显存，但远小于直接训练 full dense MLP。首轮建议扫描 `r∈{16,32,64}`。

如果线性修正饱和，再测试零初始化的非线性残差：

\[
\hat C = \hat C_{base}+g\cdot \mathrm{GELU}(XA)B,
\]

其中 gate `g` 初始化为 0。非线性版增加在线计算和权重，必须在低秩线性版失败后再做。

原论文中的 full MLP 若直接按本地 reverse geometry 推导，参数量约 14.18B，BF16 约 26.4 GiB、FP32 约 52.8 GiB；并且论文中 MLP 并非对所有 pair 都更好。因此它不适合作为本项目的第一改进路线。

**[研究假设]** ridge-anchored 低秩修正能在不破坏稳定基座的前提下优化下游功能；尚未在本地正式任务上验证。

### 4.3 直接优化 attention 和 target 行为

建议总损失为：

\[
\mathcal L =
\lambda_{kv}\mathcal L_{robust\_kv}
+\lambda_{attn}\mathcal L_{attn}
+\lambda_{out}\mathcal L_{attn\_out}
+\lambda_{logit}\mathcal L_{suffix\_KL}
+\lambda_{traj}\mathcal L_{trajectory}
+\lambda_{eos}\mathcal L_{EOS}.
\]

各项定义：

1. `L_robust_kv`：按 layer/head/channel 标准化后的 Huber 或 normalized MSE，保持数值稳定；
2. `L_attn`：使用原生 14B query，比较 native cache 与 mapped cache 的 attention distribution KL；
3. `L_attn_out`：比较 attention readout 的 cosine 和 norm-normalized MSE；
4. `L_suffix_KL`：让原生 14B cache 与 mapped cache 在后续 8–32 个 teacher-forced tokens 上的 logits 对齐；
5. `L_trajectory`：比较这些 suffix steps 的关键 residual/attention states；
6. `L_EOS`：显式对齐 EOS、换行、答案边界和停止 token 的 logit/BCE，针对 GSM8K 和 CoQA 的生成失控。

teacher 应是**原生 Qwen3-14B**，因为当前目标是保持 14B 行为。使用 32B logits 作为 teacher 会把问题改成能力蒸馏，必须另立实验。

建议先对 layers 7–10、33–38 加权，但不能只凭相关性决定。第一步应做 exact-layer replacement 或 per-layer tail replay 消融，确认这些层是否对 Hella/GSM/CoQA 的损失具有因果贡献。

**[研究假设]** attention/output 与 suffix/EOS 损失比 KV reconstruction R² 更能预测并改善本地下游质量；相关工作只提供机制支持，尚无本地因果证明。

### 4.4 Native target tail refresh

当前 runtime 映射 `C[:-1]` 并由 14B replay 1 个 boundary token。可以把 replay window 扩大为 `R∈{1,8,32,64,128,256}`：

1. 32B prefill 完整 prompt；
2. 对较早的 prefix cache 做 32B→14B 映射；
3. 丢弃 mapped cache 的最后 `R` 个位置；
4. 使用原生 14B 对最后 `R` 个 prompt tokens 增量 prefill，重建近端 KV；
5. 从 native-refreshed 边界开始生成。

该方法直接针对边界状态、EOS 与误差累积，而且成本远小于重新 prefill 全 prompt。长生成可以进一步测试每 2K/4K tokens 做一次 target anchor。

必须把结果分成两个系统档位：

- `R=1`：纯 mapper baseline；
- `R>1`：mapper + partial target replay 的 quality mode。

不能把 `R>1` 的结果写成“完全跳过 target prefill”的纯 mapper 结果。

**[研究假设]** 64–256 token refresh 能显著修复 GSM8K 的 EOS runaway 和长上下文边界漂移；需要用真实 latency/quality 曲线验证。

### 4.5 可选的不确定性 fallback

可从以下信号估计映射风险：

- attention entropy 或 native/mapped attention disagreement；
- suffix first-token KL；
- EOS logit drift；
- mapper residual norm；
- 多个 shrinkage/专家输出之间的 disagreement。

高风险请求执行更长 tail refresh，极高风险请求退回完整 14B prefill。报告必须给出 coverage–quality–latency 曲线，不能只报告选择性保留后的准确率。

## 5. Calibration 数据与训练设计

现有 500×1024 FineWeb-Edu activation 足以做闭式 shrinkage、标准化 ridge 和小规模机制验证，但不足以直接训练 attention/suffix 目标，因为当前 capture 没有完整保存 target query、attention output 和 suffix logits。

建议新增一个与正式测试集严格隔离的功能校准集，约 256K tokens：

- 64×1K 通用文本；
- 32×4K chat/QA/summary 混合文本；
- 8×8K 长上下文文本；
- position 采用 log-uniform 覆盖；
- 包含普通 prose、对话、短答案 QA、摘要、数学/代码结构，但不能包含正式 benchmark 样本或其答案。

建议保存：

- 32B source K-content/V；
- 14B native Q、attention output；
- attention top-128 indices/weights 与 tail mass，而非完整 `T×T` attention；
- native 14B suffix logits；
- exact 14B target KV；
- tokenizer、prompt、sequence、model shard、code SHA。

粗略预计新增 capture 约 75–85 GiB；这是设计预算，不是已经测得的最终占用。为控制显存，可顺序 capture 32B 和 14B，而不是同时保留两个模型；训练低秩 mapper 时使用离线 source cache，并只加载必要的 14B teacher 子路径或缓存后的 target statistics。

数据切分必须按完整 sequence 做，不能按 token 随机切分。正式测试任务仅在所有超参数冻结后运行一次；不能用 Hella/GSM/CoQA 正式结果调 layer、rank、lambda、refresh window 或 early stop。

## 6. 分阶段实验路线

| 阶段 | 方案 | 是否增加在线成本 | 主要目的 |
|---|---|---:|---|
| E0 | exact-layer replacement / layer ablation | 诊断性 | 找到真正因果敏感层 |
| E1 | k4/k20 shrinkage + standardized ridge/lambda sweep | 否 | 低成本降低过拟合 |
| E2 | `R=1/8/32/64/128/256` tail refresh | 是，受 R 控制 | 修复边界与 EOS 漂移 |
| E3 | rank 16/32/64 low-rank attention-aligned linear refinement | 训练增加，部署不增加 | 直接优化 attention 功能 |
| E4 | + 8-token suffix KL + EOS loss | 部署不增加 | 稳定首步和停止行为 |
| E5 | + 32-token KL/trajectory loss | 部署不增加 | 降低自回归累积误差 |
| E6 | zero-init nonlinear residual | 增加 | 在线性容量不足时扩容 |
| E7 | 2-expert token/domain router | 增加 | 处理异质 token/领域 |
| E8 | uncertainty-gated refresh/fallback | 条件增加 | 质量保障模式 |

每阶段先运行 pilot：HellaSwag 1,000、GSM8K 200、CoQA 100 conversations、HotpotQA 500、MultiNews 100、WikiText 小切片。只有机制指标、质量和延迟均满足门槛，才运行正式 full test。

## 7. 评测指标与统计

### 7.1 下游指标

- ARC/Hella/Wino/MMLU：accuracy、target decision agreement、target-correct retention、gain/regression flip matrix、margin flip；
- GSM8K：numeric-final、strict/flexible、EOS/cap rate、生成长度、答案标记和格式；
- CoQA/HotpotQA：F1/EM、answer containment、答案长度、stop/cap；
- MultiNews：ROUGE-1/2/L；
- LMSYS/ShareGPT4：reference corpus PPL、ROUGE-L、chrF、normalized-token F1、output retention；
- WikiText-2：PPL。

### 7.2 机制指标

- held-out K/V R² 和 train–held-out gap；
- attention distribution KL；
- attention-output cosine/norm error；
- first-token 与 8/32-step suffix logit KL；
- EOS/stop logit error；
- per-layer exact replacement gain；
- mapper residual norm 与失败概率相关性。

### 7.3 系统指标

- mapper latency、tail-refresh latency、完整 14B re-prefill latency；
- peak allocated/reserved/driver free memory；
- mapper storage、capture storage、训练耗时；
- coverage–quality–latency 曲线。

### 7.4 统计方法

- 分类/QA 使用 paired bootstrap 和 McNemar；
- 对话按 conversation 做 cluster bootstrap；
- 长文本按 document/block bootstrap；
- 训练至少 3 个种子；
- 多个 ablation 使用 Holm correction；
- 同时报告点估计、95% CI、样本数和 cap/truncation。

建议的研究门槛（用于筛选，不是结果预测）：

- HellaSwag mapped ≥ 78.15%；
- WinoGrande mapped ≥ 72.45%；
- GSM8K numeric-final mapped ≥ 92.68%；
- CoQA mapped ≥ 75.55%；
- WikiText-2 mapped PPL ≤ 7.65；
- ARC/MMLU 不劣于原生 14B，且 decision agreement 明显高于当前约 82%；
- 任何任务不能出现统计显著的 >0.3–0.5 pp 新下降。

这些门槛的作用是提前停止无效方案，不代表最终一定能够达到。

## 8. 推荐优先级

1. **立即做**：exact-layer causality、k4/k20 shrinkage、feature standardization、K/V/layer-specific lambda；
2. **最高 ROI 系统实验**：tail refresh `R=8–256`，重点观察 GSM8K、CoQA、HotpotQA 和长上下文；
3. **主创新方法**：ridge-anchored rank-16/32/64 functional low-rank refinement；
4. **关键训练目标**：attention output + suffix logit KL + EOS calibration；
5. **最后再做**：nonlinear residual、MoE、uncertainty fallback；
6. **不建议优先**：直接训练 14.18B full MLP，或仅凭 reconstruction R² 继续扩大 k。

如果只允许做一个最小改进实验，首选：

> `k4/k20 shrinkage + R∈{1,32,64,128} tail refresh`，在 GSM8K-200、CoQA-100、HellaSwag-1000 上联合评估。

如果要形成一条可投稿的方法主线，首选：

> `standardized ridge/shrinkage base + attention-aligned low-rank correction + suffix/EOS distillation + optional native tail refresh`。

## 9. 风险与停止条件

- 如果 attention/output 指标提高但 target decision agreement 和 GSM/CoQA 不提高，说明当前 functional proxy 仍错位，停止扩大模型容量；
- 如果 `R=32/64` 已恢复大部分质量，则应优先研究边界 refresh，而不是部署更重 mapper；
- 如果不同训练 seed 的 task delta 方差超过平均增益，方法不具备稳定性；
- 如果低秩融合后在线 mapper latency或显存发生变化，说明实现没有真正 fuse，需要单独报告；
- 如果使用 benchmark 数据选择 layer/rank/lambda，必须把结果标为开发集，不得作为正式测试；
- 如果 fallback 只覆盖容易样本，必须报告 coverage，不能用筛选后的准确率替代全量结果。

## 10. 相关工作与可迁移边界

| 工作 | 可借鉴机制 | 不能直接声称的内容 |
|---|---|---|
| Cross-Model KV Cache Transfer | top-k ridge、content-space K、attention-output 机制分析 | 没有公开本地 reverse 全任务数值 |
| See What I See, Know What I Think | dense/functional alignment、trajectory 信号 | 训练任务和 teacher 设置不同，不能直接视为通用 KV 重建证据 |
| Mixture-of-Translators | token/域路由、trajectory loss | 包含 target context replay，不证明完全跳过 target prefill |
| LatentAlign | 非线性 translator 与 latent alignment | 只在较小模型、较短长度验证，translator 很大 |
| Cache-to-Cache | residual/gated cache translation | 强依赖 receiver native cache，和纯映射设置不同 |
| CacheBlend | partial recomputation/refresh | 支持系统直觉，不直接证明 32B→14B 的最佳 R |
| Model Stitching / A2D / IAM | CCA、attention KL、功能映射 | 对象分别是 residual、NMT attention 或 attention matrix，不是本地完整 KV |

上述设计由相关工作的机制证据支持，但尚未在 Qwen3-32B→14B 的完全 prefill-reuse 设置中得到直接验证。

## 11. Primary sources

- [Cross-Model KV Cache Transfer in LLM Families](https://arxiv.org/html/2608.03893)
- [See What I See, Know What I Think](https://arxiv.org/html/2606.13594)
- [Mixture-of-Translators](https://arxiv.org/html/2607.28979)
- [Latent Space Communication via K-V Cache Alignment](https://arxiv.org/html/2601.06123)
- [Cache-to-Cache](https://arxiv.org/html/2510.03215)
- [CacheBlend](https://arxiv.org/abs/2405.16444)
- [KVSculpt](https://arxiv.org/abs/2603.27819)
- [Transferring Linear Features Across Language Models With Model Stitching](https://arxiv.org/html/2506.06609)
- [Align-to-Distill](https://arxiv.org/html/2403.01479)
- [IAM](https://arxiv.org/html/2507.11953)
- [DroidSpeak](https://arxiv.org/html/2411.02820)
- [RADLADS](https://arxiv.org/html/2505.03005)

