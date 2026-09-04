from __future__ import annotations

import pytest
import torch

from src.losses import (
    eos_soft_bce,
    fixed_query_attention_loss,
    global_r2,
    suffix_kl_native_to_candidate,
    variance_normalized_huber,
)


def attention_case():
    torch.manual_seed(7)
    native_query = torch.randn(1, 4, 3, 2)
    native_key = torch.randn(1, 2, 4, 2)
    native_value = torch.randn(1, 2, 4, 2)
    mask = torch.tensor([[[True, False, False, False], [True, True, False, False], [True, True, True, False]]])
    return native_query, native_key, native_value, mask


def test_fixed_query_attention_is_zero_for_identical_cache():
    q, key, value, mask = attention_case()
    result = fixed_query_attention_loss(
        native_query=q,
        native_key=key,
        native_value=value,
        candidate_key=key,
        candidate_value=value,
        causal_mask=mask,
    )
    assert float(result.centered_logit_huber) < 1e-12
    assert float(result.output_cosine) < 1e-6
    assert float(result.output_nmse) < 1e-12


def test_fixed_query_attention_detects_key_and_value_perturbation():
    q, key, value, mask = attention_case()
    result = fixed_query_attention_loss(
        native_query=q,
        native_key=key,
        native_value=value,
        candidate_key=key + 0.5,
        candidate_value=value - 0.4,
        causal_mask=mask,
    )
    assert float(result.centered_logit_huber) > 0
    assert float(result.output_cosine) > 0
    assert float(result.output_nmse) > 0


def test_attention_rejects_empty_causal_rows_and_non_bool_mask():
    q, key, value, mask = attention_case()
    with pytest.raises(ValueError):
        fixed_query_attention_loss(
            native_query=q,
            native_key=key,
            native_value=value,
            candidate_key=key,
            candidate_value=value,
            causal_mask=torch.zeros_like(mask),
        )
    with pytest.raises(ValueError):
        fixed_query_attention_loss(
            native_query=q,
            native_key=key,
            native_value=value,
            candidate_key=key,
            candidate_value=value,
            causal_mask=mask.to(torch.int64),
        )


def test_suffix_kl_native_to_candidate_direction_and_mask():
    torch.manual_seed(8)
    native = torch.randn(2, 5, 11)
    candidate = native.clone()
    mask = torch.tensor([[True, True, True, False, False], [True, True, False, False, False]])
    same = suffix_kl_native_to_candidate(native_logits=native, candidate_logits=candidate, suffix_mask=mask)
    assert float(same) < 1e-9
    bad = candidate.clone()
    bad[..., 0] -= 4.0
    changed = suffix_kl_native_to_candidate(native_logits=native, candidate_logits=bad, suffix_mask=mask)
    assert float(changed) > 0
    with pytest.raises(ValueError):
        suffix_kl_native_to_candidate(native_logits=native, candidate_logits=candidate, suffix_mask=torch.zeros_like(mask))


def test_eos_soft_bce_and_kv_guardrail():
    native = torch.zeros(1, 3, 5)
    candidate = native.clone()
    mask = torch.ones(1, 3, dtype=torch.bool)
    baseline = eos_soft_bce(native_logits=native, candidate_logits=candidate, suffix_mask=mask, eos_token_ids=[4])
    candidate[..., 4] = -10
    degraded = eos_soft_bce(native_logits=native, candidate_logits=candidate, suffix_mask=mask, eos_token_ids=[4])
    assert float(degraded) > float(baseline)
    target = torch.randn(1, 2, 2, 4)
    variance = torch.ones(2, 4)
    assert float(variance_normalized_huber(candidate=target, native=target, variance=variance)) == 0.0
    assert float(global_r2(candidate=target, native=target)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        eos_soft_bce(native_logits=native, candidate_logits=candidate, suffix_mask=mask, eos_token_ids=[8])

