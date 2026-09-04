# Tail-refresh pilot strict analysis

This directory is append-only analysis of the completed, frozen GPU run at
`../runs/method_dev_pilot_r1_r8_r32_v2`. It does not edit or rerun the model,
runner, prepared registry, or any scientific result.

The analyzer rejects missing, extra, failed, or malformed atomic results;
recomputes token-ID hashes, tokenizer decode/stop/cap behavior and QA metrics;
checks every target binding; and enforces the exact R=1/8/32 cache boundary
contract. It then performs sample-ID-clustered paired bootstrap analysis with
the preregistered seed `20260830` and 10,000 replicates.

Run on CPU only:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python \
  analysis/src/analyze.py --write
PYTHONDONTWRITEBYTECODE=1 /home/wby/llm-cache/nvidia-kvcache/.venv/bin/python \
  -m pytest -q analysis/tests
```

The pilot decision is deliberately strict: a hybrid is called promising only
if relaxed F1 improves over R=1 and cap/valid-termination behavior is nonworse
on every registered pilot dataset. This is method-development evidence, not a
locked-test or paper-superiority claim.
