# Formal config publication checklist

现在不要创建 `config.json`。只有以下证据全部存在且稳定后，才能从 template 复制出一个新的 formal
config（保留 template 不变）：

1. 10 的 `PUBLISHED.json` 必须是最后原子 marker：exact 8 keys、`status=published_complete`、
   `atomic_last_completion_marker=true`、exact 14-file inventory，全部 payload/sidecar bytes 通过。
2. 四个 split 的 path/nbytes/SHA 和**最终实际行数**必须从该 publication 复制；字段只能是
   ID/source-index/source-projection-SHA/local-rank/rank-SHA/stratum，screen/confirm disjoint。
   component-feasibility 完成前不得把原始 `128+256` / `256+512` 写成 formal counts；它们仅是上限。
3. formal config 的 marker/manifest/receipt/four splits 与 10 native publication byte-exact 相同；不建
   shadow/translation registry。
4. 09 G0-v2 PASS receipt：`scope=decontaminated_registry_v2_vs_mapper_fit_and_locked_eval`、
   `pass=true`、`closes_complete_g0=true`、`locked_outcomes_opened=false`。
5. G0 `candidate_registry_binding` 与 formal config 的 marker/manifest/receipt/four splits byte-exact
   相同；comparisons 精确只有四个必需 direction，且每项精确只有 `flags: 0`。
6. 对所有 refs 重新计算 path/nbytes/SHA；不接受 latest symlink、directory 或 zero hash。
7. 把 `status` 设为 `frozen_ready`、`runnable` 设为 true；确认全文没有 `__PENDING_`、
   `__PLACEHOLDER_`、`TEMPLATE_ONLY`。
8. 先运行 `check_readiness.py`。若 producer receipt schema 与 `registry_gate.py` 不同，应该在不放宽
   语义的前提下适配 validator 并增加 fixture test；禁止仅为了继续运行而跳过字段。
9. `prepare.py` 成功后冻结 config、code inventory、prepared receipt；从此任何代码变化都需要新
   protocol/run ID，不能静默复用旧 run definition。
10. 用 pinned venv 只先 freeze/run smoke，真实 runner 必须产出 full-weight/code/CUDA/5+3 topology/
    raw-grid receipt。screen outcome 在 smoke validation 完成前不得打开；confirm outcome 在
    screen `promotion_status=pass` 的 validation receipt 产生前不得打开。
11. audited local generation loop 的 `terminal_token_id` CPU oracle、冻结 EOS 集合的 hashed-metadata/
    runtime equality、terminal/cost tamper tests 必须全部通过；raw cost 必须精确覆盖 GPU `0..7` 且 phase
    timing 只能使用冻结的 unavailable schema。
12. generation/R32/cost/attempt ledger 的所有整数字段必须通过 `strict_int`，嵌套 exact evidence 必须
    递归验证 JSON 类型；`true/false` 替换 `1/0` 的全链 tamper tests 必须失败。
13. 六个正式 Python 入口必须以 pinned interpreter、`PYTHONDONTWRITEBYTECODE=1` 和配置中精确的
    `PYTHONPYCACHEPREFIX` 启动；prefix 在启动前必须不存在（含 dangling symlink）。入口在首次本地
    `src` import 前检查 `sys.dont_write_bytecode`/`sys.pycache_prefix`，深运行 receipt 还要记录并
    重验这三个 bytecode-policy 证据字段；独立 `runtime_package_closure` 还必须逐字节绑定并重验
    `kvcache_transfer/__init__.py` 与 `kvcache_transfer/io.py`。

禁止事项：读取 10 的 IN_PROGRESS split、把 09 v1 failure 当 G0 pass、从旧 07 prepared_v1 取样、
打开 official dev/test、根据 screen gold 调 threshold、把 two-pass proxy latency 写成真实在线 latency。
