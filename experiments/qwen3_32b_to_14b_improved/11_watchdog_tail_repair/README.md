# 11 — Watchdog-Gated Tail Repair R32

独立的 CPU-only post-hoc hypothesis 目录。这里不修改 03/07，不调用 GPU，也不生成新的模型
输出。

- `METHOD.md`：方法、在线 state machine、成本与失败模式。
- `PROTOCOL.md`：当前 counterfactual 边界及未来去污染前瞻验证要求。
- `config.json`：固定 gate 和 03 输入的 content-addressed bindings。
- `src/watchdog.py`：可注入未来 runner 的 gold-blind online API。
- `src/analyze_counterfactual.py`：逐个验证 384 raw files 后做 paired 分析。
- `results/`：aggregate、无答案文本的 sample audit、receipt 和 SHA-256 sidecars。

复现命令：

```bash
cd /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/11_watchdog_tail_repair
/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/analyze.py --write
/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python -m pytest -q
```

所有结果均为看过 03 pilot 后的探索性反事实，不能宣称优于 NVIDIA。下一步必须在未来
`10_decontaminated_eval` 未观察 screen/confirm 上以前瞻方式验证。
