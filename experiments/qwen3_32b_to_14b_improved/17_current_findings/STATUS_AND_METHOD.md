# 32B→14B KV-cache 复用：当前证据、方法假设与下一轮实验

更新时间：2026-09-02 UTC  
范围：Qwen3-32B source → Qwen3-14B receiver；所有新尝试使用独立 sibling 目录，旧 NVIDIA 对照只读保留。

## 结论先行

目前已经找到有希望的组件，但还没有足够的前瞻、去污染、配对证据证明整体方法超过 NVIDIA 方法。最可靠的几何信号是 **head-wise shrinkage**；最强但仍探索性的功能信号是 **按生成状态触发的尾部 target replay（R8/R32）**。下一版候选方法暂名 **FAR-KV（Functional Alignment + Robust shrinkage KV）**：

1. 以现有 k20 ridge mapper 为 anchor，并用独立 held-out 估计的 head/layer shrinkage 把高方差方向拉回 k4/k20 的稳健折中；
2. 不再只优化 KV 重建 (R^2)，而是以固定 native-14B query 的 attention logits/readout、attention-output cosine/NMSE、短 suffix teacher-forced logit KL 和 EOS 稳定性为功能损失；
3. 对 OOD/边界漂移只触发小窗口 replay（R1/R8/R32），而不是所有 token 全量重算；
4. 对低置信度或异常长生成使用 fail-safe anchor/fallback，避免把 cap/runaway 当成“正常答案”。

这个命名和设计是待验证假设，不代表已经取得论文级提升。

## NVIDIA 对照的本地正式结果

权威 receipt：`runs/five_task_extension_full_k20_v1/`（协议对齐的本地 extension；论文没有公布同一 Qwen3-32B→14B 五任务 bit-exact 表，因此 `paper_bit_exact=false`）。

| 任务/指标 | native 14B | NVIDIA k20 mapped | 差值（mapped-native） |
|---|---:|---:|---:|
| ARC-Challenge acc_norm | 60.0683% | 60.7509% | +0.6826 pp |
| HellaSwag acc_norm | 78.6995% | 77.6041% | −1.0954 pp |
| WinoGrande acc | 72.9282% | 71.9811% | −0.9471 pp |
| MMLU acc | 78.8064% | 79.7963% | +0.9899 pp |
| GSM8K numeric-final（诊断） | 94.9962% | 90.3715% | −4.6247 pp |
| GSM8K canonical strict-match | 92.7976% | 64.2153% | −28.5823 pp |
| WikiText2 prefix PPL（越低越好） | 7.5020 | 7.8022 | 恶化约 4.00% |

长上下文 HotpotQA 的旧正式/长跑证据显示约 8–10 pp F1 级别掉点；旧 smoke 只有每任务 n=2，不能作为结论。旧 32B→14B 长跑已停止，结果和目录不再被新方法覆盖。

## 已发现的有效信号

### 1. Shrinkage：最稳定的低风险改进

目录：`02_shrinkage_mapper/`。在 mapper-fit 之外的 held-out 激活上，冻结 k20 与 k4 的前缀关系，按 layer/head 求验证集 shrinkage：

- K uniform-channel (R^2)：约 0.7450 → 0.7655（+2.05 pp）；
- V uniform-channel (R^2)：约 0.6516 → 0.6832（+3.17 pp）；
- 5-fold OOF 与 refit 差异约 (10^{-5})，640 个 alpha 没有撞到 0/1 边界；
- 运行时仍是一次 affine k20 matmul，没有额外 receiver forward。

这证明重建误差可下降，但不证明任务准确率一定上升；CoQA 的历史 k4/k20 对比已经说明 (R^2) 与功能保真并不等价。

### 2. Tail replay：对长推理链可能有效，但高度条件化

目录：`03_tail_refresh/`。独立 pilot（HotpotQA 48、MuSiQue 48；不是 locked test）：

- HotpotQA：native F1 .7645，R1 .7314，R8 .7241，R32 .7290；R8/R32 没有改善；
- MuSiQue：native .3487，R1 .3649，R8 .3878，R32 .4787；R32 相对 R1 +11.38 pp，并将观察到的 7/48 cap-hit 降至 0；bootstrap 区间仍包含 0；
- R1/R8/R32 的 pilot 没有满足“每个数据集都正增益且终止非劣”的晋级规则。

解释：replay 似乎能修复多跳推理中的边界漂移，但 HotpotQA 的错误不只来自末尾生成失控，所以不能把 R32 当通用修复。

### 3. Watchdog：有信号，但之前是 post-hoc

目录：`11_watchdog_tail_repair/`。用生成长度/EOS/cap 状态触发 replay 的离线反事实：HotpotQA +1.04 pp、MuSiQue +8.38 pp；但规则是在看过 pilot 后提出，不能用于优越性声明。下一轮必须在读取任何 method-dev outcome 前冻结规则，并保存 terminal token/event trace。

## 为什么不能继续只追 (R^2)

本地机制分析（`artifacts/validation/qwen3_32b_to_14b/heldout_activation_reconstruction.json`、`runs/formal_pipeline_v1/mechanism_hellaswag_k20/`）给出：

- k20 train→held-out gap：K 约 9.87 pp，V 约 13.66 pp；k4 gap 仅约 3.28/4.28 pp；
- attention-output cosine 与任务 retention 的相关性明显高于 KV (R^2)（论文报告的方向也是如此）；
- layer 34/35 的 attention cosine 最差（约 .316/.348），但局部 K/V (R^2) 并不总是最差；
- HellaSwag、CoQA、GSM 的主要损失分别表现为 margin flip、答案边界/冗长回答、EOS/runaway，而非单一平均 KV MSE；
- MMLU 反而净增约 +0.99 pp，说明 mapper 会改变决策边界，单纯提高重建分数不能保证正确选项保留。

因此主指标应包含 target-decision agreement、target-correct retention、first-token KL、gold margin flip、attention-output cosine、EOS/cap rate 和最终任务分数；(R^2) 只作 guardrail。

## FAR-KV 候选的精确定义

### Anchor

冻结 `02_shrinkage_mapper` 的 40 层 FP32 head-wise affine mapper；逐层绑定 k20/k4 manifest、fit fingerprint 和上游字节哈希。若任何绑定或 algebra audit 失败，整版 fail-closed。

### Functional residual

在全新独立目录 `14_functional_residual_mapper/` 中实现零初始化低秩残差：

- 每个目标层和 K/V 分支共享 encoder (A\in\mathbb{R}^{20480\times r})；每个 KV head 有 decoder (B_h\in\mathbb{R}^{r\times128})；
- 初始输出为零，故候选开始时严格等于 shrinkage anchor；
- 推荐先试 (r=64)，再试 128/256；只允许 mapper 参数更新，冻结两个语言模型；
- OOD smooth gate 与 residual RMS cap 超阈值时回退 anchor，不允许异常残差污染整段 cache。

### 功能损失（按优先级）

1. 固定 native-14B Q，在 boundary/真实 read position 重算 attention centered-logit 与 output cosine/NMSE；
2. 对 8/32 个 suffix token 做 native→candidate teacher-forced logit KL（温度 1/2）；
3. EOS soft BCE 与 cap/runaway penalty；
4. K 使用 query/attention-read 权重，V 使用实际 read weight 加权；
5. KV Huber/(R^2) 仅作稳定性约束。

### Replay policy

R1 为纯 mapper基线；R8/R32 是独立质量模式，不混入 R1 分数。第一版前瞻规则只看运行时可观测的 terminal event、generation count、cap_hit、first-token KL/OOD proxy；不得读取 gold answer 或 locked outcome。

## 实验隔离与验收

目录职责：

- `00/02/03/11/14/15/16/17` 各自独立；旧目录只读，不覆盖、不复用结果文件；
- mapper-fit、method-dev、locked-test 三层固定 ID+SHA；当前曾看过的正式集只能作为历史 baseline，不能再次用于调参；
- `10_decontaminated_method_dev_registry/` 的旧配额尝试全部保留为失败证据，不把 infeasible 当成模型结果；
- 任何 GPU runner 必须先过 G0、native registry、模型/mapper 全哈希、8 卡 idle gate 和新鲜 `PYTHONPYCACHEPREFIX` 门。

前瞻候选晋级建议（paired、以 native 14B 和 NVIDIA k20 同时为参照）：

1. HotpotQA 与 MuSiQue method-dev 至少各 256 条；另设不重叠 screen/confirm；
2. 候选相对 NVIDIA k20 在至少两个长上下文任务上 95% paired bootstrap 区间下界 > 0；
3. semantic macro improvement 区间下界 > 0；
4. 非目标任务不可劣化：Hella/MMLU ≤0.5 pp，GSM numeric/Hotpot ≤1 pp，CoQA ≤1.5 pp；
5. GSM strict-match 的提升只有在 numeric 或 cap/EOS 也改善时才算实质改善；
6. 如果 attention cosine/margin 改善而最终任务没有改善，判为机制正向但方法 NO-GO；如果只有 (R^2) 上升而任务不升，同样 NO-GO。

## 当前工程状态（截至本文件更新时间）

- 旧 NVIDIA 长上下文 full：已停止，旧结果保留；
- pairwise-v3 graph/deep：已完成且从 pinned raw sources 重建一致；
- pairwise-v3 原 allocation：严格证明原 18-target quota 不可行，不能继续使用；
- `16_quota_capacity_repair_v1`：正在生成新 PRE-FORMAL snapshot，尚未取得独立审计 PASS，尚未启动 A/B/C/D 正式搜索或 GPU；
- `12_prospective_screen_v2`：CPU/bytecode/runner 审计 snapshot 已通过内部测试，仍等待最终 registry/G0 contract；
- `13_oracle_decontaminated_v2`：PRE-FORMAL v3 独立审计 PASS；
- `14_functional_residual_mapper`：仅 CPU 框架与契约，尚未进行真实 functional capture/training；
- 因此目前没有任何“已在 locked test 上显著优于 NVIDIA”的结论。

## 参考文献（方法动机，不等于本地结果）

- NVIDIA KV-cache reuse paper: <https://arxiv.org/abs/2608.03893>
- Mixture-of-Tokens / context correction: <https://arxiv.org/abs/2607.28979>
- CacheBlend selective recomputation: <https://arxiv.org/abs/2405.16444>
- KVBoost deviation-guided repair: <https://arxiv.org/abs/2608.21362>
- PyramidKV: <https://arxiv.org/abs/2406.02069>
- SnapKV: <https://arxiv.org/abs/2404.14469>
- KV-Distill: <https://arxiv.org/abs/2503.10337>
- KVSculpt: <https://arxiv.org/abs/2603.27819>
- HotpotQA: <https://arxiv.org/abs/1809.09600>
- MuSiQue: <https://arxiv.org/abs/2011.01060>

补充检索到的最新相关工作：**CacheBridge**（2026-09-01 提交）把 matched-head support、按 causal-attention sensitivity 加权的闭式统计和有界 mapper 结合起来；其摘要报告在 Qwen3 方向的平均 target retention 99.83%，但主要实验方向和本项目的 32B→14B 逆向设置不同，不能直接移植数值或宣称本地已复现。它强化了本项目优先测试“attention/read-weight 加权 + architecture-indexed support”的理由：<https://arxiv.org/abs/2609.00891>。

## 深一层的技术判断：为什么 FAR-KV 可能比静态 ridge 更有机会

静态 ridge 只回答“source activation 如何逼近 target activation”。真正影响生成的是 target query 对整段 cache 的读出：

\[
o_{l,t}=\operatorname{softmax}(q_{l,t}K_l^\top/\sqrt d)V_l,
\qquad z_t=\operatorname{LMHead}(h_t).
\]

所以同样的平均 K/V MSE，若误差落在高 read-weight 或高 query-sensitivity 方向，logit/margin 仍会大幅改变；反之，低 read-weight 方向的较大误差可能几乎没有任务影响。FAR-KV 把目标从“像素重建”改为“被 receiver 读出来以后仍像 native receiver”。这也是为什么候选损失必须同时包含 fixed-Q attention output、suffix logits 和 EOS，而不能只比较 (R^2)。

### 推荐的最小可行消融矩阵

每一行都在同一 mapper-fit、同一 method-dev registry、同一随机种子和同一生成参数下配对比较：

| 编号 | cache 变换 | 额外 receiver 计算 | 目的 |
|---|---|---:|---|
| B0 | NVIDIA k20 ridge | R1 boundary replay | 原始基线 |
| B1 | k4/k20 head-wise shrinkage | R1 | 只测试泛化正则 |
| B2 | B1 + query/read-weighted low-rank residual | R1 | 测功能损失是否优于 MSE |
| B3 | B2 + OOD gate，低风险 R1/高风险 R32 | 条件 R32 | 测质量—延迟折中 |
| B4 | B1 + fixed-Q scalar/head calibration（无低秩训练） | R1 | 低成本机制 sanity check |
| B5 | B2 + layer 34/35 重点 residual | R1 | 测 late-layer 因果性 |

B4 是重要的便宜对照：如果只用每层/每头的标量缩放就能提升 first-token KL 和 HotpotQA，说明主要问题是 readout 温度/尺度；若 B4 无效而 B2 有效，才值得承担低秩训练和额外存储。B5 不能只凭相关系数晋级，必须先做单层 exact replacement 的因果消融。

### 低秩残差的参数与部署约束

对一个 (20480\to1024) 的 target layer，rank (r=64) 的共享 encoder 加 8 个 head decoder 约为

\[
20480r+8\cdot128r \approx 1.36\text{M}
\]

个 FP16/FP32 参数，远小于完整 20-way affine map 的 20.97M 权重。若把 (B=0) 初始化，训练前后 mapper 输出严格连续；将 (W_\text{anchor}+BA) 融合为单个 affine 后，在线仍只有一次矩阵乘，不能把离线训练成本误报成运行时开销。实际报告必须分别列出：融合后 mapper bytes、GPU resident bytes、每 prompt map latency、R1/R32 receiver FLOPs。

### 更可能有效的触发器

不读取答案即可得到的触发器，按侵入性从低到高排列：

1. 映射 residual RMS / anchor disagreement；
2. boundary first-token KL 或正确选项 margin proxy（仅在 method-dev 训练时用 gold 计算阈值，部署时用无标签 proxy）；
3. attention entropy / top-k read-weight 质量；
4. 生成过程中出现第一个 terminal 事件前的长尾（content token 数、cap proximity、重复 n-gram）。

触发器要在 method-dev 之前冻结，且用双阈值 hysteresis，避免单个 noisy token 在 R1/R32 间来回切换。正式结果必须同时报告所有请求的 coverage、平均 replay tokens、p50/p95 latency 和质量；只报告被筛选后样本会产生选择偏差。

### 预期失败模式与停止规则

- **只提高 (R^2)**：若 attention cosine、first-token KL 或任务分数不改善，立即停止该方向；
- **只改善 MuSiQue**：可作为长链特化模式，但不能宣称通用 superiority；至少要在 HotpotQA 和一个非多跳任务上通过非劣门槛；
- **只改善 strict-match**：如果 numeric-final、EOS/cap 没有同步改善，只能称格式修复；
- **增加 replay 后提升**：必须提供质量—延迟曲线，并和相同 replay 预算的 NVIDIA baseline 比较；
- **低秩 residual 破坏高 margin 样本**：启用 OOD gate/fallback，若仍超过预注册 non-inferiority margin，整版 NO-GO；
- **训练/选择依赖已看过的正式集**：结果降级为探索性，重新建立 prospective screen/confirm。

该判断与公开工作的机制方向相符，但具体 FAR-KV 组合仍是本项目的待验证新方法：NVIDIA 的跨模型 KV 映射、CacheBlend 的选择性重算、PyramidKV/SnapKV 的层/头非均匀预算、MoT/KV-Distill/KVSculpt 的功能或轨迹监督只能提供动机，不能替代本地 paired 证据。

## 2026-09-02 状态增量：目前什么算“有效”

这里把“有效”分成三档，避免把漂亮的中间指标误称为准确率提升：

| 证据档位 | 当前结果 | 可以说什么 | 不能说什么 |
|---|---|---|---|
| 几何验证 | shrinkage 在独立 held-out 上 K/V R² 分别约 +2.05/+3.17 pp，OOF 稳定 | 泛化重建误差确实下降 | 不等于 QA/生成准确率上升 |
| 探索性功能 pilot | R32 尾部重算在 MuSiQue 48 条上比 R1 高约 11.38 pp，HotpotQA 上反而低约 0.24 pp；watchdog 反事实为 post-hoc | 长多跳链存在可修复的状态漂移，且触发式 replay 有信号 | 不能据此宣称通用优于 NVIDIA，样本小且规则曾看过结果 |
| 前瞻正式比较 | 尚未开始；12 的 runner 已到 PRE-FORMAL，10 的原始 quota 已被严格证明不可行，容量修复仍待审计 | 目前没有锁定的 superiority 结论 | 不能把任何旧 pilot 当 locked-test 或论文复现 |

因此，当前唯一已经“有效”且足够稳健的结论是：**head-wise shrinkage 是可靠的低风险组件；条件式 R32 replay 是有条件的质量修复器；二者组合是否减少 32B→14B 的 HotpotQA 掉点，仍必须由新建的 prospective screen/confirm 实验决定。**

### 2026-09-02 新增实际 GPU 消融：静态 moment calibration NO-GO

独立目录 `20_moment_calibration_probe_v1/` 在 GPU 0 上测试了一个便宜候选：对冻结 shrinkage 输出增加每层/每 KV head 的尺度和每通道偏置，并向恒等变换收缩。实验只解释原 40 条 tuning 激活，10 条 locked 激活从未索引；beta 由五折嵌套 group CV 选择。K/V 的 OOF SSE 分别只下降 0.02365%/0.04203%，低于预注册 0.1% 门槛，因此严格判定 `NO_GO`，不消耗下游 HotpotQA/MuSiQue 预算。这个负结果排除了“只修均值/尺度”这一便宜路线，进一步支持把下一轮算力集中在 fixed-Q attention/readout、suffix KL/EOS 和条件 replay，而不是继续追静态输出校准。

## 方法选择的因果优先级（比继续堆模型更重要）

对一个 target layer 的 boundary query (q)，映射误差对 attention logit 的一阶影响是

\[
  \delta a_i \simeq q^\top\delta k_i/\sqrt{d},
  \qquad
  \delta o \simeq \sum_i p_i\,\delta v_i +
       \sum_i (\delta p_i) v_i .
\]

这给出一个可检验的优先级，而不是凭 R² 猜测：

1. 先做固定 native-Q 的 K query-covariance 加权回归，和 V 的 causal-read-weight 加权回归；
2. 再在同一 anchor 上训练零初始化的低秩 residual；
3. 最后才加 R8/R32 replay。若第 1 步不能改善 first-token KL/attention-output，直接停止该分支；若只能改善 R²，也停止。

一个可复现的闭式目标是

\[
 \min_{W,b}\;\sum_n (y_n-x_nW-b)^\top G_n(y_n-x_nW-b)
             +\lambda\lVert W\rVert_F^2,
\]

其中 K 分支 (G_n=\epsilon I+q_nq_n^\top)，V 分支则使用 native attention probability 的对角近似。这样仍能用充分统计量和一次 affine map 部署，不把在线成本偷偷变成 target 全量 prefill。实际实现中用低秩/分块近似 (G_n)，并报告近似误差。

## 新增的工程性阻塞（不是方法否定）

去污染图已经从 pinned raw source 完整重建：31,260 条记录、39,447 条 hyperedge；原 18 个 quota 的 allocation 两个 seed 都得到严格 `HiGHS status=2 / explicit infeasible`，并非硬件故障。其根因是 MuSiQue 4-hop 记录的碰撞约束，原先要求的 117 条同时选取超过图的可行容量。容量修复目录 `16_quota_capacity_repair_v1` 已生成 v10 PRE-FORMAL 快照（尚待独立审计），后续只允许在新 quota 上逐阶段 CPU 求解；旧 allocation 结果永不回写或覆盖。

这一步看似与模型方法无关，实际上决定了准确率结论是否可信：如果 screen/confirm 因碰撞泄漏或不可行 quota 被悄悄改动，任何“提升”都可能只是数据选择偏差。故正式 GPU 顺序固定为：容量/registry → G0/native contract → CPU preflight → 2+2 smoke → screen → confirm；每一步失败即停止。

## 预期结果与决策边界

最值得押注的候选不是一个更大的 MLP，而是 `shrinkage anchor + attention-aware low-rank residual + uncertainty-gated R32`。理由是它同时满足三点：初始行为连续（残差零初始化）、训练目标对准 receiver 真正读取的方向、异常样本可以回退到已验证 anchor。若 HotpotQA 和 MuSiQue 的前瞻 screen 中不能同时看到 first-token KL/answer-margin 与 F1 的方向一致改善，则不进入 confirm；若只在 MuSiQue 改善，则记录为多跳特化，不称通用方法；若任何非目标任务超过预设非劣边界，则整版 NO-GO。
