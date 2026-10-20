# 本机运行与阶段验收

以下命令都从 `/home/tiger/kvcache-mapped` 执行。当前实验已经在运行，**不要再次启动同一输出目录的协调器**；本页的 fit/merge 命令用于理解或在独立的新实验目录重建，不能覆盖本次证据。所有脚本使用断言拒绝模型、数据、权重和协议错配。

## 环境、数据和监控

```bash
cd /home/tiger/kvcache-mapped
source experiments/v_readout_ridge/env.sh
.venv/bin/python experiments/v_readout_ridge/brief_status.py
.venv/bin/python experiments/v_readout_ridge/status.py
```

环境包清单见 `source/environment_freeze.txt`，实际权重、tokenizer、论文和原始训练文件来源见 `checks/provenance.json`。需要本地 Qwen3-0.6B、1.7B、原始 Nemotron JSONL、处理后的训练/验证数据和 benchmark 输入；这些大文件不随 Git 提交。模型和数据没有通过此仓库自动上传。

```bash
.venv/bin/python experiments/v_readout_ridge/check_data.py \
  --data data/kv_lingo/training_paper_v7 \
  --output experiments/v_readout_ridge/checks/data_review.json
.venv/bin/python experiments/v_readout_ridge/audit_training_source.py
.venv/bin/python -m pytest experiments/v_readout_ridge/tests -q
```

训练数据构造参考脚本为 `source/prepare_training_reference.py`：固定 revision、seed42、每个流前100,000条候选、400train/64validation。作者未公开原始样本列表及 packing 实现，本地精确样本构造和 prefix 总量差异必须保留在报告中。数据已经冻结时只做审计，不重新生成覆盖它。全部评测 prompt 的预处理审计程序为 `prepare_eval.py`，会生成 `checks/evaluation_inputs.json` 与本地 `prompt_hashes.json`。

## 拟合与验证

单方向两个 shard，共400条，同时累积 weighted/readout 的统计量。例如方向 `06_17` 的 shard0 命令如下；shard1 使用 `--shard 1 --device cuda:1`，反向使用 `--pair 17_06`、独立目录和 GPU2/3。不要给两个进程相同的 shard 编号。

```bash
.venv/bin/python experiments/v_readout_ridge/fit.py fit \
  --pair 06_17 --shards 2 --shard 0 --device cuda:0 \
  --output experiments/v_readout_ridge/artifacts/06_17/full_v1
```

只有两个 shard 都结束、`shard_00.json`/`shard_01.json` 均标记完成，才能依次执行：

```bash
.venv/bin/python experiments/v_readout_ridge/fit.py merge \
  --pair 06_17 --shards 2 \
  --output experiments/v_readout_ridge/artifacts/06_17/full_v1
.venv/bin/python experiments/v_readout_ridge/audit_fit.py \
  experiments/v_readout_ridge/artifacts/06_17/full_v1
.venv/bin/python experiments/v_readout_ridge/validate.py \
  --pair 06_17 --device cuda:0 \
  --artifact experiments/v_readout_ridge/artifacts/06_17/full_v1 \
  --output experiments/v_readout_ridge/checks/validation_06_17_full
.venv/bin/python experiments/v_readout_ridge/audit_upstream.py \
  --validation experiments/v_readout_ridge/checks/validation_06_17_full
```

反向同理；本次这些步骤已全部完成并通过。两个 mapper 条件共享同一 K 解，不是分别重复拟合 K。独立审计检查112个序列化矩阵、FP32残差、两组目标最优性、原始 moment shard 成员及 hash。

## 正式评测调度

本次活动协调器记录在 `checks/resume_evaluation_process.json`。它先复核完整fit/validation，再运行两条互不重复的 case 队列/GPU。不要仅按进程号重启，必须核对 `/proc/PID/cmdline` 与该文件里的 command 相同。

每个 case 都由独立模型进程先 smoke 再 full。正式生成必须经过 `deterministic_eval.py`，不可直接绕过 wrapper 调用旧的 `evaluate.py`：后者保持原冻结数值实现，但 wrapper 额外启用确定性算法、CUBLAS workspace和seed，并写 `runtime.json`。审计要求 smoke/full 完整首批记录完全相等。

普通任务持共享 GPU 调度锁；8192-token ARC/MMLU reasoning-on 持独占锁。锁等待是正常排队，不是计算卡死。长MC首批可能持续数十分钟，日志按整批输出。不要通过减预算、丢弃未解析题或终止仍在计算的批次改变分数。

已保存的完整预测批次可恢复；已完成并审计的单元保留。若发现异常，先保存失败证据、核对原命令和协议，再恢复。旧非确定性输出在 `runs_nondeterministic_v1/`，OOM历史在 `checks/oom_schedule_v3/`，都不能作为本轮正式结果。

活动监控器为 `guard_execution.py`，身份见 `checks/guard_process.json`，每分钟状态见 `checks/guard_status.json`。它只允许对 CUDA OOM 最多两次自动恢复：先归档日志并停止核对过命令的自有进程组，再以每卡所有任务独占的方式恢复保存的完整批次。任何输入、mapper、数值或评分审计错误都不会自动重试放行，会写 `guard_failure.json`。仅当协调器结束且最终结果严格52/52时才写 `guard_complete.json`；监控器运行本身不表示实验完成。

## 结果生成

```bash
# 中间汇总：只列入已完成、通过独立审计的单元
.venv/bin/python experiments/v_readout_ridge/report.py
# 最终验收：缺少任何一个单元都会失败
.venv/bin/python experiments/v_readout_ridge/report.py --require-complete
```

最终必须52/52。MC主分数由 `rescore_mc.py` 统一处理明确的LaTeX/Markdown选项格式，原预测和v1分数不覆盖，变更明细存 `full/mc_scoring_v2.json`。其他任务使用原评分。结果写 `RESULTS.json`/`RESULTS.md`，配对置信区间、长度截断和未解析数量同时保存。

最终交付监听器 `finalize_delivery.py --wait` 已独立运行，进程身份记录在 `checks/delivery_process.json`，等待状态见 `checks/delivery_status.json`。不要重复启动。它只在执行监控器生成 `guard_complete.json` 后工作：核对52单元、26组配对比较、样本数、分数、首批一致性及证据指纹，然后生成 `FINAL_REPORT_ZH.md` 和 `checks/final_delivery.json`。中文报告在此之前明确标记未完成。此步骤不调用模型、不改变评分、不执行 Git push；验收异常写入 `checks/delivery_failure.json`，不会产出完成声明。

本轮目标仅Stage1两种V目标的比较。不能将NIAH子任务写成全RULER套件综合分；不能将本地训练样本声称为作者未公开的原始样本；不能用训练损失下降代替任务分数。过程见 `PROCESS.md`，缺陷与纠正证据见 `AUDIT_20261019.md`。不push。
