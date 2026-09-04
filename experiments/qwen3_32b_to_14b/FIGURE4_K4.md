# Qwen3 32B→14B CoQA Figure 4：retuned k=4 曲线

## 论文直接公开与未公开边界

论文正文 §4.6 和 Figure 4 直接公开：CoQA、100 个约 15 轮对话、5 个领域、word-level F1、handoff turn 1–10；32B→14B production 曲线使用 k=20，dotted re-tuned 曲线的图例明确使用 k=4。正文还报告 large-to-small drift 约以 0.33 pp/turn 增长，以及 multi-turn tuning 相对 production k 最多改变约 2.0 pp。

论文没有公开：100 个 conversation ID、选择 seed、prompt、answer-history 策略、解码参数、精确 scorer 实现，以及 multi-turn k sweep 的候选结果、目标函数和 tie break。因此本地结果不能声称 bit-exact；这些本地补全部分与已有 k=20 CoQA 协议保持完全相同，使 k20/k4 对比只改变 mapper。

论文页面与原图：

- <https://arxiv.org/html/2608.03893v1#S4.SS6>
- <https://arxiv.org/html/2608.03893v1/figures/retention_vs_turns.png>

## 独立 k=4 mapper

k=4 不是从 k=20 权重中截取。每个 source layer 提供 `8 KV heads × 128 = 1024` 个特征，因此 k=4 的完整 layer mapper 是 `4096→1024`，不是 `2048→1024`。40 个 target layers 的 K/V aggregate 参数量为：

- FP32 weights：`2 × 40 × 4096 × 1024 = 335,544,320`
- FP32 biases：`2 × 40 × 1024 = 81,920`
- FP32 参数总字节：`1,342,504,960`（约 1.2503 GiB）

选择阶段复用 production k=20 阶段已经计算并加 SHA256 的完整 `40×64` single-source score tensor，但会重新从全部 64 个 source layers 排序取 top-4。它不会直接信任或截取 top-20 JSON 行。排序仍是 K/V 等权、head/channel 均匀平均的 R²，平分时选择更低 source-layer index。

拟合使用同一批冻结 FineWeb-Edu `500×1024, stride=4` 激活、centered unnormalized ridge `λ=0.01`、BF16 activation、FP32 statistics/weights、FP64 diagnostics，并把输出放到完全独立的目录：

- selection：`artifacts/selection/qwen3_32b_to_14b/k4_retuned/`
- mapper：`artifacts/mappers/qwen3_32b_to_14b/k4_retuned_lambda0.01/`
- held-out：`artifacts/validation/qwen3_32b_to_14b/k4_retuned/`
- CoQA：正式 pipeline 的 `coqa_reverse_k4_retuned/`

## 执行顺序

```bash
.venv/bin/python experiments/qwen3_32b_to_14b/scripts/select_k4_coqa.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python experiments/qwen3_32b_to_14b/scripts/fit_mapper_k4_coqa.py \
  --devices 0,1,2,3,4,5,6,7 \
  --chunk-tokens 1024 \
  --solver cholesky \
  --analysis-device cpu \
  --cpu-threads-per-worker 2

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python experiments/qwen3_32b_to_14b/scripts/validate_heldout_k4_coqa.py \
  --compute-device cuda:0 \
  --analysis-device cpu \
  --chunk-samples 1024

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  .venv/bin/python experiments/qwen3_32b_to_14b/scripts/evaluate_coqa_k4.py \
  --output-dir experiments/qwen3_32b_to_14b/runs/formal_pipeline_v1/coqa_reverse_k4_retuned
```

正式 CoQA 入口会先验证 held-out acceptance artifact。验收规则不是人为设置论文未公开的 R² 阈值，而是：40 层 K/V 全覆盖、所有原始指标有限、每层 top-4 与 mapper manifest 完全一致、4096→1024 形状一致、train/held-out token 不同、mapper/activation/selection SHA 链完整。任何条件不满足都会 fail closed，不能进入 CoQA。

## Pipeline 接入要求

现有 production k=20 阶段和产物不改。正式 pipeline 在最终 completion receipt 之前追加以下独立 stages：

0. 获取 repository GPU lock 后新建 12 次连续 idle gate `gates/k4_training_segment_idle.json`；旧 training gate 不能替代它。
1. `11_k4_selection`：运行 `select_k4_coqa.py`，用 `validate_k4_selection_artifact` 验证。
2. `12_k4_fit_target0`：仅 layer 0 的 GPU smoke；GPU monitor evidence 必须使用独立文件 `evidence/fit_k4_target0_gpu_monitor.json`，不得覆盖 k20 evidence。
3. `13_k4_fit_full`：完整 40 层重新拟合；验证 weight/bias aggregate、全部 payload、selection snapshot 和 independent-refit evidence。
4. `14_k4_heldout`：运行独立 held-out reconstruction，`validate_k4_heldout_artifact` 为验收器。
5. 新建 12 次连续 idle gate `gates/coqa_k4_segment_idle.json`。
6. `15_k4_coqa`：运行 `evaluate_coqa_k4.py`；`validate_k4_coqa_output` 和 `formal_k4_coqa_receipt.json` 均必须通过。

最终 pipeline definition、stage receipt inventory、GPU gate inventory 和 completion receipt 的 final artifacts 都必须直接绑定 k4 selection、mapper manifest、held-out artifact、CoQA formal receipt 及相应 SHA sidecars。completion receipt 的验证入口必须重新调用 k4 stage validators，不能只验证顶层 receipt 的字节哈希，否则底层 mapper layer/checkpoint 在完成后变化时无法 fail closed。k4 fit smoke monitor 需要让通用 `run_command` 接受 evidence destination 参数；不能复用当前固定的 k20 monitor 路径。
