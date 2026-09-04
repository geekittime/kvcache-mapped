#!/usr/bin/env python3
"""Validate Qwen3 boundary replay and mapped-cache runtime conventions.

The default ``tiny`` mode creates a deterministic random Qwen3 entirely in
memory.  It checks both the target-own-cache oracle and a target-to-itself
identity mapper that traverses the complete inverse-RoPE -> FP32 map -> target
RoPE -> DynamicCache path.  It never loads the 14B or 32B checkpoints.

After the production environment is ready, ``target-local`` mode can validate
the same target-own-cache boundary convention on the local Qwen3-32B model.
It deliberately does not claim to test transfer accuracy; that requires the
trained 14B -> 32B mapper and benchmark runner.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Sequence

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kvcache_transfer.cache_adapter import (  # noqa: E402
    boundary_replay_inputs,
    cache_to_legacy,
    clone_dynamic_cache,
)
from kvcache_transfer.config import ArchitectureSpec  # noqa: E402
from kvcache_transfer.io import atomic_write_json  # noqa: E402
from kvcache_transfer.mapper import LayerMapper  # noqa: E402
from kvcache_transfer.modeling import load_qwen3  # noqa: E402
from kvcache_transfer.runlog import (  # noqa: E402
    append_reproduction_log,
    make_run_directory,
    record_run_start,
)
from kvcache_transfer.transfer import (  # noqa: E402
    target_boundary_replay,
    transfer_source_cache,
)


@dataclass(frozen=True)
class InMemoryIdentityBundle:
    """Tiny smoke-test double with the MapperBundle streaming interface."""

    architecture: ArchitectureSpec

    def iter_layers(
        self, layer_devices: Sequence[torch.device | str]
    ):
        if len(layer_devices) != self.architecture.target_layers:
            raise ValueError("target device count does not match tiny architecture")
        output_features = self.architecture.target_features
        for target_layer, device_value in enumerate(layer_devices):
            device = torch.device(device_value)
            identity = torch.eye(output_features, dtype=torch.float32, device=device)
            yield target_layer, LayerMapper(
                source_layers=(target_layer,),
                key_weight=identity,
                key_bias=torch.zeros(output_features, dtype=torch.float32, device=device),
                value_weight=identity.clone(),
                value_bias=torch.zeros(output_features, dtype=torch.float32, device=device),
            )


def tiny_model(device: torch.device, dtype: torch.dtype) -> Qwen3ForCausalLM:
    torch.manual_seed(20260813)
    config = Qwen3Config(
        vocab_size=127,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=None,
        _attn_implementation="eager",
    )
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def compare_last_logits(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, object]:
    reference = reference[:, -1].float().cpu()
    candidate = candidate[:, -1].float().cpu()
    difference = (reference - candidate).abs()
    reference_log_probs = reference.log_softmax(-1)
    candidate_log_probs = candidate.log_softmax(-1)
    reference_probabilities = reference_log_probs.exp()
    candidate_probabilities = candidate_log_probs.exp()
    log_prob_difference = (reference_log_probs - candidate_log_probs).abs()
    reference_top1 = reference.argmax(-1)
    candidate_top1 = candidate.argmax(-1)
    top5_width = min(5, reference.shape[-1])
    top10_width = min(10, reference.shape[-1])
    reference_top5 = reference.topk(top5_width, dim=-1).indices
    candidate_top5 = candidate.topk(top5_width, dim=-1).indices
    reference_top10 = reference.topk(top10_width, dim=-1).indices
    candidate_top10 = candidate.topk(top10_width, dim=-1).indices
    reference_norm = reference.norm(dim=-1)
    relative_l2 = (reference - candidate).norm(dim=-1) / reference_norm.clamp_min(
        torch.finfo(reference.dtype).tiny
    )
    kl_divergence = (
        reference_probabilities * (reference_log_probs - candidate_log_probs)
    ).sum(dim=-1)
    return {
        "max_abs_logit": float(difference.max().item()),
        "mean_abs_logit": float(difference.mean().item()),
        "rmse_logit": float((difference.square().mean()).sqrt().item()),
        "reference_logit_rms": float((reference.square().mean()).sqrt().item()),
        "relative_l2_logit": float(relative_l2.max().item()),
        "cosine_similarity_logit": float(
            torch.nn.functional.cosine_similarity(reference, candidate).min().item()
        ),
        "max_abs_log_probability": float(log_prob_difference.max().item()),
        "reference_top1_log_probability_delta": float(
            log_prob_difference.gather(-1, reference_top1.unsqueeze(-1)).max().item()
        ),
        "kl_reference_to_candidate": float(kl_divergence.max().item()),
        "total_variation_probability": float(
            (0.5 * (reference_probabilities - candidate_probabilities).abs().sum(-1))
            .max()
            .item()
        ),
        "reference_top1": reference_top1.tolist(),
        "candidate_top1": candidate_top1.tolist(),
        "top1_equal": bool(torch.equal(reference_top1, candidate_top1)),
        "minimum_top5_overlap": min(
            len(set(left.tolist()) & set(right.tolist()))
            for left, right in zip(reference_top5, candidate_top5, strict=True)
        ),
        "minimum_top10_overlap": min(
            len(set(left.tolist()) & set(right.tolist()))
            for left, right in zip(reference_top10, candidate_top10, strict=True)
        ),
    }


def compare_last_hidden(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    """Compare the post-final-norm state immediately before ``lm_head``."""

    reference = reference[:, -1].float().cpu()
    candidate = candidate[:, -1].float().cpu()
    difference = reference - candidate
    reference_norm = reference.norm(dim=-1)
    return {
        "max_abs_hidden": float(difference.abs().max().item()),
        "mean_abs_hidden": float(difference.abs().mean().item()),
        "rmse_hidden": float((difference.square().mean()).sqrt().item()),
        "reference_hidden_rms": float((reference.square().mean()).sqrt().item()),
        "relative_l2_hidden": float(
            (
                difference.norm(dim=-1)
                / reference_norm.clamp_min(torch.finfo(reference.dtype).tiny)
            )
            .max()
            .item()
        ),
        "cosine_similarity_hidden": float(
            torch.nn.functional.cosine_similarity(reference, candidate).min().item()
        ),
    }


# These gates concern an identity control, not mapper quality.  They are based
# on distributional/ranking quantities that remain meaningful when BF16
# prefill and one-token decode select different attention kernels.  A raw
# maximum-logit bound is deliberately absent: a large tail-vocabulary logit
# delta can coexist with a tiny KL divergence and an unchanged decision.
TARGET_LOCAL_ORACLE_THRESHOLDS: dict[str, float | int] = {
    "relative_l2_logit_max": 0.03,
    "cosine_similarity_logit_min": 0.999,
    "kl_reference_to_candidate_max": 0.005,
    "reference_top1_log_probability_delta_max": 0.1,
    "minimum_top5_overlap_min": 4,
    "relative_l2_hidden_max": 0.03,
    "cosine_similarity_hidden_min": 0.999,
    "exact_control_max_abs": 0.0,
}


def target_local_case_passed(
    replay: dict[str, object],
    hidden: dict[str, float],
    *,
    thresholds: dict[str, float | int] = TARGET_LOCAL_ORACLE_THRESHOLDS,
) -> bool:
    """Classify a target-own-cache replay without masking BF16 kernel drift."""

    return bool(
        replay["top1_equal"]
        and float(replay["relative_l2_logit"])
        <= float(thresholds["relative_l2_logit_max"])
        and float(replay["cosine_similarity_logit"])
        >= float(thresholds["cosine_similarity_logit_min"])
        and float(replay["kl_reference_to_candidate"])
        <= float(thresholds["kl_reference_to_candidate_max"])
        and float(replay["reference_top1_log_probability_delta"])
        <= float(thresholds["reference_top1_log_probability_delta_max"])
        and int(replay["minimum_top5_overlap"])
        >= int(thresholds["minimum_top5_overlap_min"])
        and float(hidden["relative_l2_hidden"])
        <= float(thresholds["relative_l2_hidden_max"])
        and float(hidden["cosine_similarity_hidden"])
        >= float(thresholds["cosine_similarity_hidden_min"])
    )


def diagnostic_context_lengths(total_length: int) -> tuple[int, ...]:
    """Choose fixed short/medium replay cases plus the complete smoke prompt."""

    if total_length < 2:
        raise ValueError("diagnostic context must contain at least two tokens")
    return tuple(sorted({length for length in (2, 4, 8, 15, total_length) if length <= total_length}))


@torch.inference_mode()
def run_tiny(device: torch.device) -> dict[str, object]:
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = tiny_model(device, dtype)
    context = torch.tensor([[7, 5, 19, 2, 31, 11, 3]], device=device)
    full = model(input_ids=context, use_cache=False)
    oracle = target_own_cache_boundary_oracle(model, context)
    oracle_metrics = compare_last_logits(full.logits, oracle.logits)

    architecture = ArchitectureSpec(
        source_layers=2,
        target_layers=2,
        source_kv_heads=2,
        target_kv_heads=2,
        source_head_dim=16,
        target_head_dim=16,
        selected_source_layers=1,
    )
    source_prefix = context[:, :-1]
    source_cache = model.model(input_ids=source_prefix, use_cache=True).past_key_values
    mapped = transfer_source_cache(
        source_cache,
        source_config=model.config,
        target_model=model,
        mapper_bundle=InMemoryIdentityBundle(architecture),  # type: ignore[arg-type]
        output_dtype=dtype,
        token_chunk=2,
    )
    cache_max_abs = max(
        float((mapped.key_cache[layer].float() - source_cache.key_cache[layer].float()).abs().max())
        for layer in range(architecture.target_layers)
    )
    replay = target_boundary_replay(
        model,
        context[:, -1:],
        mapped,
        clone_cache=False,
    )
    mapped_metrics = compare_last_logits(full.logits, replay.logits)
    tolerance = 1e-4 if dtype == torch.float32 else 2e-2
    passed = (
        oracle_metrics["top1_equal"]
        and mapped_metrics["top1_equal"]
        and float(oracle_metrics["max_abs_logit"]) <= tolerance
        and float(mapped_metrics["max_abs_logit"]) <= tolerance
        and cache_max_abs <= tolerance
    )
    return {
        "mode": "tiny",
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "context_length": context.shape[1],
        "prefix_length": context.shape[1] - 1,
        "oracle": oracle_metrics,
        "identity_mapped": mapped_metrics,
        "identity_mapped_key_cache_max_abs": cache_max_abs,
        "tolerance": tolerance,
        "passed": bool(passed),
    }


@torch.inference_mode()
def run_target_local(
    model_path: Path,
    devices: Sequence[int],
    text: str,
    max_tokens: int,
    attention: str,
) -> dict[str, object]:
    model = load_qwen3(
        model_path,
        devices=devices,
        role="target",
        attention_implementation=attention,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    encoded = encoded[:, :max_tokens]
    if encoded.shape[1] < 2:
        raise ValueError("smoke text must tokenize to at least two tokens")
    embedding_device = model.model.embed_tokens.weight.device
    context = encoded.to(embedding_device)
    full = model(
        input_ids=context,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    full_with_cache = model(
        input_ids=context,
        use_cache=True,
        return_dict=True,
    )
    prefix = model.model(
        input_ids=context[:, :-1],
        use_cache=True,
        return_dict=True,
    ).past_key_values

    # This is the exact explicit input dictionary used by production boundary
    # replay.  Keep a cloned branch so subsequent controls cannot observe a
    # cache already extended by this call.
    explicit_inputs = boundary_replay_inputs(
        context[:, -1:],
        prefix,
        clone_cache=True,
    )
    explicit = model(
        **explicit_inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    # Let Transformers infer both positions and mask independently.  Exact
    # agreement with the explicit branch rules out an off-by-one/API error.
    implicit = model(
        input_ids=context[:, -1:],
        past_key_values=clone_dynamic_cache(prefix),
        use_cache=True,
        return_dict=True,
    )

    cache_key_max = 0.0
    cache_value_max = 0.0
    for (prefix_key, prefix_value), (full_key, full_value) in zip(
        cache_to_legacy(prefix),
        cache_to_legacy(full_with_cache.past_key_values),
        strict=True,
    ):
        cache_key_max = max(
            cache_key_max,
            float(
                (prefix_key.float() - full_key[..., :-1, :].float()).abs().max().item()
            ),
        )
        cache_value_max = max(
            cache_value_max,
            float(
                (prefix_value.float() - full_value[..., :-1, :].float())
                .abs()
                .max()
                .item()
            ),
        )

    metrics = compare_last_logits(full.logits, explicit.logits)
    hidden_metrics = compare_last_hidden(full.hidden_states[-1], explicit.hidden_states[-1])
    same_prefill_metrics = compare_last_logits(full.logits, full_with_cache.logits)
    explicit_implicit_metrics = compare_last_logits(explicit.logits, implicit.logits)
    exact_limit = float(TARGET_LOCAL_ORACLE_THRESHOLDS["exact_control_max_abs"])
    exact_controls_passed = bool(
        float(same_prefill_metrics["max_abs_logit"]) <= exact_limit
        and float(explicit_implicit_metrics["max_abs_logit"]) <= exact_limit
        and cache_key_max <= exact_limit
        and cache_value_max <= exact_limit
    )
    semantic_gate_passed = target_local_case_passed(metrics, hidden_metrics)

    sweep: dict[str, dict[str, object]] = {}
    for context_length in diagnostic_context_lengths(context.shape[1]):
        if context_length == context.shape[1]:
            case_logits = metrics
            case_hidden = hidden_metrics
        else:
            case_context = context[:, :context_length]
            case_full = model(
                input_ids=case_context,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            case_prefix = model.model(
                input_ids=case_context[:, :-1],
                use_cache=True,
                return_dict=True,
            ).past_key_values
            case_inputs = boundary_replay_inputs(
                case_context[:, -1:],
                case_prefix,
                clone_cache=False,
            )
            case_replay = model(
                **case_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            case_logits = compare_last_logits(case_full.logits, case_replay.logits)
            case_hidden = compare_last_hidden(
                case_full.hidden_states[-1], case_replay.hidden_states[-1]
            )
        case_passed = target_local_case_passed(case_logits, case_hidden)
        sweep[str(context_length)] = {
            "oracle": case_logits,
            "pre_lm_head_hidden": case_hidden,
            "semantic_gate_passed": case_passed,
        }
    sweep_passed = all(bool(case["semantic_gate_passed"]) for case in sweep.values())
    passed = exact_controls_passed and semantic_gate_passed and sweep_passed
    return {
        "mode": "target-local",
        "model_path": str(model_path.resolve()),
        "devices": list(devices),
        "attention": attention,
        "dtype": "bfloat16",
        "context_length": context.shape[1],
        "prefix_length": context.shape[1] - 1,
        "oracle": metrics,
        "pre_lm_head_hidden": hidden_metrics,
        "same_length_prefill_use_cache_control": same_prefill_metrics,
        "explicit_vs_implicit_position_control": explicit_implicit_metrics,
        "prefix_cache_vs_full_cache_prefix_max_abs": {
            "key": cache_key_max,
            "value": cache_value_max,
        },
        "thresholds": TARGET_LOCAL_ORACLE_THRESHOLDS,
        "exact_controls_passed": exact_controls_passed,
        "semantic_gate_passed": semantic_gate_passed,
        "context_length_sweep": sweep,
        "context_length_sweep_passed": sweep_passed,
        "diagnostic_conclusion": (
            "BF16 attention-kernel path difference is bounded; cache contents, explicit/implicit "
            "positions, top-1 decision, and predictive distribution controls pass"
            if passed
            else "identity boundary replay diagnostic gate failed; investigate before benchmarks"
        ),
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("tiny", "target-local"), default="tiny")
    parser.add_argument("--device", default="cpu", help="tiny mode torch device")
    parser.add_argument("--model", type=Path, default=Path("/home/wby/models/Qwen3-32B"))
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--attention", default="flash_attention_2")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument(
        "--text",
        default=(
            "The purpose of this deterministic cache replay test is to verify the final token. "
            "A second sentence provides enough tokens to compare the prefill and decode paths."
        ),
    )
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = make_run_directory(ROOT, "p5", f"{args.mode}_oracle_smoke")
    record_run_start(
        run,
        sys.argv,
        {
            "mode": args.mode,
            "device": args.device,
            "model": str(args.model),
            "devices": args.devices,
            "attention": args.attention,
            "max_tokens": args.max_tokens,
        },
    )
    if args.mode == "tiny":
        result = run_tiny(torch.device(args.device))
    else:
        devices = tuple(int(value) for value in args.devices.split(",") if value.strip())
        result = run_target_local(
            args.model,
            devices,
            args.text,
            args.max_tokens,
            args.attention,
        )
    atomic_write_json(run.path / "metrics.json", result)
    atomic_write_json(
        run.path / "result.json",
        {"status": "passed" if result["passed"] else "failed", **result},
    )
    if not args.no_ledger:
        append_reproduction_log(
            ROOT,
            heading=f"P5-ORACLE-{run.run_id}",
            bullets=[
                f"运行 `{args.mode}` boundary-cache smoke；结果："
                f"{'通过' if result['passed'] else '失败'}。",
                "协议固定为 source/target prefix C[:-1]，再由 target 重放 C[-1]。",
                *(
                    [
                        "真实模型 gate 使用 top-1、logit relative-L2/cosine、KL、top-1 "
                        "log-prob 与 pre-lm-head hidden 指标；不以 BF16 tail-logit 的单点 "
                        "max-abs 作为等价性判据。",
                        f"诊断结论：{result['diagnostic_conclusion']}。",
                    ]
                    if args.mode == "target-local"
                    else []
                ),
                f"完整证据目录：`{run.path}`。",
            ],
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
