# 超越线性 KV 映射的研究方案：SARR-KV

更新时间：2026-08-31

## 1. 研究问题与结论先行

目标不是让 32B 的 KV 坐标在均方误差上更像 14B，而是让冻结的
Qwen3-14B 在接收 32B→14B cache 后，尽可能保持其原生预测、答案边界、
EOS 行为和下游准确率。

当前最有希望的方案是：

1. 保留论文的 content-space RoPE 处理与 ridge mapper，避免从随机网络
   重新学习已经成立的线性关系；
2. 以泛化更稳的 k4 ridge 为锚点，用 k20 的额外源层特征学习一个
   零初始化、低秩、可回退的残差；
3. 训练目标从统一 KV MSE 转为 native 14B 的 attention distribution、
   attention output、短 suffix hidden trajectory 和 logits KL；
4. 另设 R=8/32 的混合模式，只让 14B 原生重放 prompt 尾部少量 token，
   修复生成边界与近期状态。R>1 必须与纯 mapper 的 R=1 分开报告；
5. 只有低秩线性残差失败后才尝试小型非线性残差或 token-routed experts。

这是一组有文献依据、但尚未在 exact Qwen3-32B→14B 上被证明的研究假设。
在锁定测试通过前，不得写成“已经超过 NVIDIA”。

## 2. 为什么不能继续只优化 R²

### 2.1 本地直接证据

| mapper | train K R² | held-out K R² | gap | train V R² | held-out V R² | gap |
|---|---:|---:|---:|---:|---:|---:|
| k20 ridge | 0.84371 | 0.74503 | 0.09868 | 0.78879 | 0.65223 | 0.13657 |
| k4 ridge | 0.77504 | 0.74223 | 0.03281 | 0.70146 | 0.65869 | 0.04278 |

k20 的训练 R² 显著更高，但 train→heldout gap 也显著更大；k4 的 held-out
V R² 甚至更高。然而 CoQA 上 k20 F1=0.72873，k4=0.72519。由此可知：

- 增加 k 可以提高坐标重构，却不保证提高功能保真；
- held-out R² 也不能单独决定下游结果；
- 需要用 14B 实际 query、attention readout 和 logits 来定义训练目标。

机制实验中，平均 attention-output cosine 只有 0.76108；第 34、35 层约
0.316/0.348，是明显功能失配点，但它们的 KV R² 并非最低。跨层 R² 与
attention cosine 不呈正相关，说明误差方向比误差总能量更重要。

### 2.2 下游失败形态

- HellaSwag：78.6995%→77.6041%，下降 1.0954pp；
- CoQA：78.2213%→72.8731%，下降 5.3481pp；
- HotpotQA 全量 baseline：relaxed F1 72.1972%→63.3915%，下降约
  8.8057pp；
- GSM8K numeric-final：94.9962%→90.3715%，下降 4.6247pp；但 canonical
  strict 从 92.7976% 降到 64.2153%，说明大量损失来自生成失控和格式；
- WikiText-2 PPL：7.5020→7.8022，恶化约 4.0%。

HotpotQA 的 mapped 平均生成长度约 22.29 token，native 14B 约 4.93；
mapped cap 69/7405，而 native 仅 1/7405。cap 仍不足以解释全部 F1 下降，
说明 cache 同时破坏了语义状态和答案边界。

ARC/MMLU 的 aggregate accuracy 有时上升，但 17% 左右的样本决策发生改变。
这可能包含 32B 信息带来的偶然 gain，也包含 native 14B 正确样本被破坏。
所以必须额外报告：target decision agreement、target-correct retention、
correct→wrong 与 wrong→correct，而不能只看净准确率。

## 3. 相关工作的可迁移部分与边界

### NVIDIA Cross-Model KV Cache Transfer

[论文](https://arxiv.org/html/2608.03893)直接支持三件事：RoPE 应在
content space 中处理；attention-output cosine 与 retention 更相关；更大
MLP 在困难模型对上可能有效，但在已较好的模型对上也可能退化。它没有证明
32B→14B 上任意 MLP 都优于 ridge，也没有公布本项目所有测试集的目标数值。

### LatentAlign 与 Mixture-of-Translators

[LatentAlign](https://arxiv.org/abs/2601.06123)的消融表明 suffix 行为目标
可能比再加 reconstruction 更重要；[Mixture-of-Translators](https://arxiv.org/abs/2607.28979)
说明 target trajectory/context correction 可以明显优于纯 projection。
但它们的模型规模、运行路径和 target replay 都与本项目不同，只能支持训练
目标的选择，不能把其结果外推为本项目效果。

### CacheBlend、KVShare 与部分重算

[CacheBlend](https://arxiv.org/html/2405.16444)和
[KVShare](https://arxiv.org/abs/2503.16525)说明 attention-sensitive 的少量
token 重算能够修复 cache reuse 误差，且 Jacobian/attention 权重比均匀误差
更合理。它们主要是同模型复用，因此这里只将其迁移为 R=8/32 尾部重放假设。

### KVBoost、KV Packet 与 DAF：修复而不是盲信旧 cache

[KVBoost](https://arxiv.org/abs/2608.21362)把 cache 误差视为可定位的偏差：先用
probe 找到高偏差 token，再重算边界或高偏差区域。这支持本项目把 target-tail
refresh 和未来的 deviation-triggered refresh 作为质量模式，而不是给全部 token
支付 14B prefill。不过它研究的是同模型、分块 cache，不能直接证明跨模型尾部重放
一定有效；本项目的 R8/R32 必须用配对任务指标验证。

[KV Packet](https://arxiv.org/abs/2604.13226)用少量可训练 header/trailer soft
tokens 和 self-distillation 修补 cache 拼接边界，说明“小型 receiver-side adapter +
teacher trajectory”可能比继续增大 KV 坐标回归器更有效。可迁移到本项目的不是其
具体 packet 结构，而是两个原则：冻结主模型、只训练很小的修复参数；用 target
logits/trajectory 监督，而非只用 KV MSE。

[Decoupled Attention Fusion](https://arxiv.org/abs/2607.21599)通过重要 token
attention、question-document attention 和状态融合恢复上下文交互。它进一步提示：
如果固定尾部重放仍不稳定，下一步应在独立目录测试“高偏差 token/层的稀疏 target
recompute”，而不是把 refresh 长度从 32 无限制增大。其 RAG cache composition
设定与本项目不同，因此只作为候选消融依据。

### 与当前实验路线的关系

上述工作共同指向三条、且必须分开报告的路线：

1. **同成本纯 mapper**：head-wise shrinkage 或可融合的低秩残差，保持 R=1；
2. **功能对齐 mapper**：attention-output、suffix-logit、EOS/stop loss；
3. **质量—延迟混合模式**：固定 tail refresh 或由不读答案的运行时偏差信号触发
   稀疏 target recompute。

当前前瞻评测先比较第 1 路的 shrinkage 和第 3 路的冻结 two-pass R32 策略。
two-pass 必须先完整运行 R1，再仅由 `generated_tokens > 32 or cap_hit` 决定是否采用
独立完成的 R32；它不是在线 watchdog，也不能报告成单次解码延迟。如果两者均未在
去污染 screen/confirm 上优于 NVIDIA k20，第 2 路的零初始化低秩功能残差才进入
下一独立目录训练。

### C2C、XKV 与 receiver-native anchor

[Cache-to-Cache](https://arxiv.org/abs/2510.03215)与
[XKV](https://arxiv.org/abs/2608.20617)支持 zero-init residual、layer/head gate
与 receiver-native anchor。但它们通常已经拥有 receiver cache，不等价于
完全跳过 14B prompt prefill。

### 其它边界

- [DroidSpeak](https://arxiv.org/abs/2411.02820)适用于同底座模型变体，不能
  直接把 32B 中间层接到宽度不同的 14B；
- [Cross-Family Speculative Prefill](https://arxiv.org/abs/2603.02631)可作为
  系统级强对照，但它是“32B 选文本、14B 原生 prefill”，不是 KV mapper；
- [Model Stitching](https://arxiv.org/abs/2506.06609)支持 SVCCA/cycle 诊断，
  但 residual stream 对齐不等于 KV 可执行；
- [Characterizing Linear Alignment](https://arxiv.org/abs/2603.18908)进一步
  警告：几何拟合成功不代表生成状态可执行。

## 4. 方法一：无额外运行时开销的 shrinkage mapper

先做最低风险实验。将 k4 的 4096 维输入嵌入 k20 的前 4096 维；本地已经
确认每个 target layer 的 k4 source rows 与 k20 的前四项完全一致。

对每个 target layer、K/V、KV head，令：

\[
f_\alpha(x)=(1-\alpha)f_{k20}(x)+\alpha f_{k4}(x_{1:4}),
\qquad \alpha\in[0,1].
\]

按完整 sequence 做 5-fold group CV，不能随机拆 token。40 条 held-out sequence
用于交叉验证和 refit，另 10 条只做一次 audit。每个 head 独立选 alpha，避免
容易过拟合的层污染全部 head。最终把两套 affine 权重融合成一个 k20 shape
矩阵，所以运行时仍是一轮 matmul，不增加 mapper latency。

该阶段的意义是验证“收缩高方差的 k20”是否能在不增加模型容量的情况下改善
泛化。若 G1 的 held-out K/V、gap 门槛不通过，后续残差不应以此初始化。

## 5. 方法二：SARR-KV 低秩功能残差

对每个 target layer 与 K/V 类型：

\[
Y_0=f_{anchor}(X),
\qquad
\hat Y=Y_0+g_{\ell,c,h}(X)\,B_{\ell,c}A_{\ell,c}z(X)+\Delta b.
\]

- `anchor` 是通过 sequence-CV 的 shrinkage/k4 解；
- `z(X)` 是按 feature/layer/head 冻结的标准化；
- rank 按 64→128→256 的固定顺序试验；
- B 或输出投影零初始化，因此第 0 步精确等于 anchor；
- gate 可按 layer/KV/head 回退到 anchor；
- OOD 时按 calibration p99 residual norm 裁剪，并在标准化输入越界时平滑关门。

若只训练线性残差，训练后可将 BA 融合进原 affine 权重，R=1 推理仍只有一次
线性映射。只有线性方法饱和后，才允许尝试 bottleneck 64/128 的零初始化
SiLU residual；非线性结果必须单独报告运行时成本。

## 6. 功能损失

### 6.1 归一化 KV guardrail

\[
L_{KV}=\sum_{\ell,h,c}
\operatorname{Huber}\left(
\frac{\hat Y_{\ell,h,c}-Y^*_{\ell,h,c}}
{\sqrt{\operatorname{Var}(Y^*_{\ell,h,c})+\epsilon}}
\right).
\]

它只防止 mapper 完全漂移，不作为主选择指标。

### 6.2 Attention distribution 与 output

使用 native 14B query：

\[
A^*=\operatorname{softmax}(Q^*K^{*T}/\sqrt d+M),\quad
\hat A=\operatorname{softmax}(Q^*\hat K^T/\sqrt d+M),
\]

\[
O^*=A^*V^*,\qquad \hat O=\hat A\hat V.
\]

损失：

\[
L_{att}=JS(A^*,\hat A)+
\beta[1-\cos(O^*,\hat O)]+
\gamma\frac{\|O^*-\hat O\|_2^2}{\|O^*\|_2^2+\epsilon}.
\]

Qwen3 是 GQA，一个 KV head 对应的所有 query heads 都必须进入损失。长序列
不保存完整 T² attention，只采样末尾行、chunk 边界行、随机行和高注意力行，
并保存 tail mass 以验证采样近似。

### 6.3 Native 14B suffix distillation

相同文本切成 prefix 与 8/32 个 suffix token：

- teacher：14B native prefix cache；
- candidate：32B prefix cache 经 mapper 得到的 14B cache；
- 两边 teacher-force 完全相同的 suffix tokens。

\[
L_{KD}=\tau^2\sum_t KL(p^{\tau}_{14,native,t}\|p^{\tau}_{14,mapped,t}),
\]

并在预注册层上加入 hidden trajectory cosine，以及 EOS/stop-control token 的
logit 差。teacher 必须是 native 14B；使用 32B logits 会变成能力蒸馏，不能再
声称只提高 cache fidelity。

总损失采用固定 curriculum：先 KV+attention，再加 suffix trajectory，最后才
加入 logits KL/EOS，并逐步降低 KV 权重。

## 7. 混合模式：target tail refresh

对总 prompt 长度 T 和刷新长度 R：

1. 只映射位置 `1..T-R`；
2. 将最后 R 个原始 prompt token 按其原 position id 送入冻结 14B；
3. 在 mapped earlier cache 的条件下得到近期 target-native cache；
4. 再开始生成。

本仓库 engine 已经原生重放最后一个 boundary token，因此 builder 内只额外
重放 R-1 个 token，才能保证总 refresh 精确等于 R。R=1 必须逐 bit 等价当前
baseline；正式只比较 R=8、R=32。刷新产生的 KV 不是完整 native 14B prefill
结果，而是以 mapped earlier prefix 为条件的 target-side correction。

该模式很可能快速改善答案边界、EOS 和最近实体，但花费目标模型算力；所以它
只能称 hybrid quality/latency trade-off，不能与纯 mapper 的速度声明合并。

## 8. 数据隔离和评测顺序

三层数据不可交叉：

1. mapper-fit：通用训练文本、按完整 sequence 划分；
2. method-dev：HotpotQA train、MuSiQue train，以及不重叠的通用 holdout；
3. locked-test：冻结候选后一次性跑 HotpotQA dev、MuSiQue dev，以及预注册的
   其它测试。已有 Hotpot/CoQA/Hella/GSM 等结果已经暴露，不能称历史盲测。

MuSiQue 使用官方 commit `922ac98...`，训练集固定 256 screening + 独立
512 confirmation，正式 dev 2417 条在候选冻结前不计算模型指标。主指标使用
官方 normalized answer F1/EM。它比 Hotpot 更强调多跳组合并减少 shortcut，
适合验证方法是否只拟合了 Hotpot 的输出风格。

严格消融顺序：

1. A0：重验 k20/k4 baseline；
2. A1：native/mapped K/V 2×2 oracle 与层块干预，先定位因果瓶颈；
3. A2：sequence-CV k4/k20 shrinkage；
4. A3：feature standardization + 分量独立 lambda；
5. A4：rank64→128→256 线性残差；
6. A5：只加入 attention 功能损失；
7. A6：再加入 suffix8→32 与 EOS；
8. A7：只有线性饱和才试非线性；
9. A8：冻结唯一纯 mapper；
10. A9：一次 locked confirmatory；
11. H1：分别冻结并测试 R=8/R=32，不回头调 R。

## 9. 怎样才算“超过 NVIDIA 方法”

纯 mapper 必须同时满足：

- 与 k20 相同 Qwen3-32B→14B、prompt、样本、decoding 和 R=1；
- degraded method-dev tasks 的 macro gap closure 至少 20%；
- paired CI 下界为正；
- 分类任务任一退化不超过 0.5pp，短 QA/numeric 不超过 1pp，CoQA 不超过
  1.5pp；
- target-correct retention 不得下降超过 0.5pp；
- mapper 权重不超过 3GiB，延迟不超过 k20 的 1.05 倍；
- locked-test 只允许一个冻结的纯 mapper 配置进入；
- 报告所有 cap、EOS、缺失和失败样本。

R=8/32 若更好，只能声称在相应额外 target compute 下达到更好的质量—延迟
Pareto 点。不能把 hybrid 结果冒充为“同成本 mapper 超过论文”。

## 10. 明确的失败条件

- R² 提高但 attention/suffix 和任务指标不提高：拒绝；
- attention cosine 提高但 paired downstream CI 不提高：不宣称成功；
- 净准确率提高但 native14 正确样本保留率下降：不通过；
- 改 prompt/stop 后才改善：只能算 decoding intervention，不能算 mapper 改进；
- 使用正式测试答案选 rank/loss/R：整版 protocol 失效，必须新建版本；
- 本机 RTX4090 PCIe 延迟不得外推成 H100/NVLink 速度。

这一设计的核心原则是：几何重构只是约束，native 14B 的功能行为才是目标；
纯 mapper 与目标侧少量重算必须科学上分开，所有“超过”结论都必须来自冻结后
的一次配对验证。
