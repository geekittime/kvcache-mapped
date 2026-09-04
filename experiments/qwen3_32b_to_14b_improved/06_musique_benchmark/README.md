# MuSiQue independent benchmark

This directory is reserved exclusively for the additional MuSiQue benchmark.
It does not contain mapper weights, HotpotQA artifacts, or outputs from another
candidate. Its purpose is to provide a more shortcut-resistant multi-hop QA
test than repeatedly tuning against the already exposed HotpotQA result.

## Pinned source

- Official repository: <https://github.com/StonyBrookNLP/musique>
- Commit: `922ac98f19a201998dbdae6d7f2887a5258dbdeb`
- Dataset release: MuSiQue v1.0, CC BY 4.0
- Download archive SHA-256:
  `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`

The archive, extracted upstream payloads, vendored official evaluator, and
license are preserved byte-for-byte. The official repository warns that the
constituent single-hop questions originate in existing QA corpora; the released
dev/test single-hop-ID registry is therefore pinned into the data manifest.

## Split policy

- MuSiQue-Answerable **train**, deterministic 256-record screen: method-dev.
- A disjoint deterministic 512-record train confirmation set: method-dev.
- MuSiQue-Answerable **dev**, all 2,417 records: sealed locked-test. It may be
  evaluated once only after one pure mapper and the separately labelled replay
  modes have been frozen.
- MuSiQue test has no public answers and is not used for local metric claims.

One dev record was inspected solely to establish the upstream schema before
the seal. This is disclosed in the lock commitment; no model aggregate has
been computed. The final result is prospective against method selection but
is not claimed to be cryptographically inaccessible to a local operator.

## Metric and prompt

Primary metrics are the official normalized answer F1 and exact match, taking
the maximum across the canonical answer and aliases. Supporting-fact metrics
are out of scope because the cache-transfer runner produces an answer rather
than a paragraph selector. All official paragraphs (normally 20; a small
number of upstream rows contain 16–19) are serialized in
official index order. The prompt requests only the shortest answer span,
Qwen thinking is disabled, generation is greedy, and the cap is 256 tokens.

## Prepare

The following CPU-only command verifies every pinned SHA, checks exact train/dev
ID and normalized-question separation, freezes the two method-dev selections,
quarantines repeated normalized training questions, and writes only
commitments—not dev answers or scores:

```bash
cd /home/wby/llm-cache/nvidia-kvcache/experiments/qwen3_32b_to_14b_improved/06_musique_benchmark
/home/wby/llm-cache/nvidia-kvcache/.venv/bin/python scripts/prepare.py
```

The GPU evaluator will be added only after the candidate runtime interface is
frozen, so a baseline and a candidate necessarily consume identical prompts,
records, stop rules, and scoring code.
