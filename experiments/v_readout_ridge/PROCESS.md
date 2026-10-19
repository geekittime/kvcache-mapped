# Process log

## 2026-10-19: scope and shutdown

Stopped previous MapFlow dispatch, active coordinators/evaluators and their owned model/cache service process groups at user's explicit request. Verified all four GPU memory allocations returned to zero. Preserved old complete/partial results and wrote MapFlow checks/stopped_for_readout_experiment.json. New local branch created without changing prior untracked KV-Lingo folders.

Read KV-Lingo Eq.3.2/3.4, Appendix C/E/O, Table4 and NVIDIA §3.1 Eq.4. Fixed the comparison protocol in README. Ridge is λ=.01; KV-Lingo no-bias architecture retained. Explicitly use prefix softmax for both fresh fits. Existing causal-weighted fits are excluded.

Re-ran corpus checks on 400/64 examples, 50/50 natural/packed and reasoning splits, template boundaries, member isolation, vocabulary/template equality and benchmark counts. All passed: checks/data_review.json. Environment reports torch 2.12.0, transformers 4.51.3 and four CUDA devices. Original datasets remain immutable.

## Evaluation and continuation validation gates

The held-out validation program completed a real-model smoke with both mapper conditions: native/mapped teacher forcing uses the same prefix[:-1] split boundary, and rescoring/aggregation passed. Four generation smokes (both directions × both modes) passed saved-token decoding, prompt retokenization, reference coverage and scorer checks. They deliberately use four-example smoke-fit mappers and a16-token generation cap to test plumbing; their near-zero scores are NOT benchmark scores and do not enter the final report. Formal jobs use400-example fits and the paper's actual generation budgets.

All15 algebra/scoring/resume tests passed. Model safetensors/config/tokenizer hashes, both paper hashes and raw Nemotron source hashes were freshly recorded/verified in checks/provenance.json. prepare_eval.py retokenizes every planned benchmark prompt with both tokenizers, checks model context limits and stores exact prompt hashes. All fitting implementation files are frozen while fitting runs.

Started durable run_pipeline.py after these gates. It verifies exact fit process identities, merges and audits both full fits, runs64-example held-out validation per direction, then dispatches four evaluation workers (one per direction/condition). Every condition/dataset/reasoning mode has fresh smoke/full services at the model-process level; first full batch must exactly equal smoke. Full outputs are independently rescored before the next cell, and final reporting requires52 complete accepted cells. Worker or stage failures are recorded and stop dependent work. Pending cells have no published scores.

## Raw benchmark audit

All13 prompt cases completed preparation with exact tokenizer ID equality and context-budget checks. Every selected MMLU-PRO/ARC-C JSON record equals its downloaded parquet record. Independently reconstructed all7405 HotpotQA prompts from raw validation parquet: all original distractor documents included in order, correct references, no appended gold answer or supporting-fact labels. The legacy Hotpot dataset manifest still mentions first500 (an old experiment); this run explicitly selects all7405 in its own immutable protocol and does not use that old subset annotation. Source audit proofs are in checks/benchmark_source_review.json and checks/hotpot_source_review.json.

## Continued pre-evaluation review

Verified all four full-fit processes and the durable coordinator remain live; no failure records. Added an independent teacher-forcing test against a full forward on a tiny Qwen3 model, covering131 continuation tokens and both64-token chunk boundaries. It passed; this is a new check of the existing validation implementation, with no change to numerical code or fitting protocol.

Ran actual32k RULER cache injection for both directions at the planned batch size4, using the existing smoke mappers and16-token output cap. Both completed all four rows and passed independent prompt retokenization, saved-token decoding and score/reference auditing. These are execution checks, not accepted benchmark results. Evidence: checks/continuation_stage_review.json and checks/ruler32k_capacity_*/. Formal runs retain the2048-token RULER budget and full400-example fitted mappers. The current two directions each reached170/400 fit examples at this review; held-out validation and formal benchmarks are still pending. Existing fitting/evaluation code fingerprints remain unchanged. No push.

## Algebra, smoke and real-model gates passed

Seven mathematical tests passed. Both directions completed four-example smoke fits; all 28 layers passed regularized normal residual and cross-objective optimum checks, K matrices bitwise equal. Actual BF16 identity cache and split logits checks passed for both models. Full-length native 1.7B capture at the longest training prefix passed all 28 layers: independently computed attention mass equal; identity readout cross/target moments equal direct Gram slices. Evidence in checks and artifacts/*/smoke_v1. Started fresh full400 fits on four GPUs, two shards per direction, collecting both conditions together. No old mapper reused.

## 2026-10-19 detailed whole-task audit

Found acceptance gaps rather than a demonstrated fit arithmetic failure: formal evaluation was not bound to the frozen prepared case, and final reporting trusted upstream/smoke stage markers too much. Added protocol guards, independent serialized-FP32 fit audit, independent validation-record checks, repeated model-weight provenance checks and direct saved smoke/full comparison. Numerical fit/inference/scoring code hashes remain unchanged; configs/audit_amendment_v2.json records the audit-only amendment.

Independently rebuilt all 400 training and 64 validation rows from 4,041 raw conversation UUIDs; prefix/continuation tokens matched exactly, raw hashes matched, no train/validation member overlap. All 28 tests passed, including new causal capture and upstream RULER metric checks. Reaudited all six generation/capacity outputs: all remain smoke, none is an accepted full benchmark result. Both four-example fits passed the new serialized-weight audit; the full fits must pass separately.

Stopped only the old waiting coordinator and restarted its amended version (checks/pipeline_process.json), preserving all four fitting processes. Full fits/64-row validations/52 benchmark cells remain pending. The report refuses completion at 0/52. Comprehensive findings, paper-reproduction limitations, proxy-loss caveats and precise evaluation scope are documented in AUDIT_20261019.md. No push.
