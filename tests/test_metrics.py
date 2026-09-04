from __future__ import annotations

import math

import pytest
import torch

from kvcache_transfer.metrics import (
    MetricInputError,
    R2Accumulator,
    global_r2,
    mean_row_cosine_similarity,
    mean_squared_error,
    r2_diagnostics,
    r2_uniform_average,
    uniform_average_r2,
)


def test_uniform_channel_r2_and_global_diagnostic_are_distinct() -> None:
    # Channel 0 has SST=2 and perfect prediction => R2=1.
    # Channel 1 has SST=200 and zero prediction => SSE=200 => R2=0.
    # Uniform average is 0.5 while pooled channel-centered SSE/SST is 1-200/202.
    target = torch.tensor([[-1.0, -10.0], [0.0, 0.0], [1.0, 10.0]])
    prediction = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    diagnostics = r2_diagnostics(target, prediction)

    torch.testing.assert_close(diagnostics.per_channel, torch.tensor([1.0, 0.0], dtype=torch.float64))
    assert diagnostics.uniform_average == pytest.approx(0.5)
    assert diagnostics.global_r2 == pytest.approx(1.0 - 200.0 / 202.0)
    assert uniform_average_r2(target, prediction) == pytest.approx(0.5)
    assert r2_uniform_average(target, prediction) == pytest.approx(0.5)
    assert global_r2(target, prediction) == pytest.approx(1.0 - 200.0 / 202.0)


def test_flattened_global_diagnostic_uses_one_scalar_target_mean() -> None:
    target = torch.tensor([[100.0, -1.0], [102.0, 1.0]])
    prediction = target + 1.0
    diagnostics = r2_diagnostics(target, prediction)

    # Per-channel SST is 4 total, whereas centering all four values around one
    # scalar mean produces a much larger diagnostic SST.
    assert diagnostics.global_sst == pytest.approx(4.0)
    expected_flat_sst = torch.sum((target.double() - target.double().mean()).square()).item()
    assert diagnostics.flattened_global_sst == pytest.approx(expected_flat_sst)
    assert diagnostics.flattened_global_r2 == pytest.approx(
        1.0 - diagnostics.global_sse / expected_flat_sst
    )


def test_streaming_r2_matches_batch_for_irregular_chunks_and_high_rank_targets() -> None:
    generator = torch.Generator().manual_seed(99)
    target = torch.randn(101, 2, 4, generator=generator)
    prediction = target + 0.2 * torch.randn(101, 2, 4, generator=generator)
    batch = r2_diagnostics(target, prediction)

    accumulator = R2Accumulator()
    for start, stop in ((0, 1), (1, 29), (29, 30), (30, 101)):
        accumulator.update(target[start:stop], prediction[start:stop])
    streamed = accumulator.finalize()

    torch.testing.assert_close(streamed.per_channel, batch.per_channel, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(
        streamed.sst_per_channel, batch.sst_per_channel, rtol=1e-13, atol=1e-13
    )
    assert streamed.uniform_average == pytest.approx(batch.uniform_average, abs=1e-14)
    assert streamed.global_r2 == pytest.approx(batch.global_r2, abs=1e-14)


def test_analysis_outputs_are_fp64_even_for_bf16_inputs() -> None:
    target = torch.arange(24, dtype=torch.float32).reshape(6, 4).bfloat16()
    prediction = (target.float() + 0.5).bfloat16()
    diagnostics = r2_diagnostics(target, prediction)

    assert diagnostics.per_channel.dtype == torch.float64
    assert diagnostics.sse_per_channel.dtype == torch.float64
    assert diagnostics.sst_per_channel.dtype == torch.float64


def test_constant_channel_policy_is_explicit() -> None:
    target = torch.ones(5, 3)
    exact = target.clone()
    wrong = target.clone()
    wrong[:, 1] = 2.0

    exact_result = r2_diagnostics(target, exact)
    wrong_result = r2_diagnostics(target, wrong)
    nan_result = r2_diagnostics(target, exact, constant_policy="nan")
    torch.testing.assert_close(exact_result.per_channel, torch.ones(3, dtype=torch.float64))
    torch.testing.assert_close(
        wrong_result.per_channel, torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64)
    )
    assert torch.isnan(nan_result.per_channel).all()
    assert math.isnan(nan_result.uniform_average)


def test_mse_and_row_cosine_have_deterministic_zero_vector_rules() -> None:
    target = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    prediction = torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])

    assert mean_squared_error(target, prediction) == pytest.approx(1.0 / 6.0)
    # Row cosines are [1, 1 (both zero), 0 (only target zero)].
    assert mean_row_cosine_similarity(target, prediction) == pytest.approx(2.0 / 3.0)


def test_metrics_reject_shape_mismatch_and_nonfinite_values() -> None:
    with pytest.raises(MetricInputError, match="identical shape"):
        r2_diagnostics(torch.ones(3, 2), torch.ones(3, 3))
    with pytest.raises(MetricInputError, match="NaN or Inf"):
        r2_diagnostics(torch.tensor([[float("nan")]]), torch.zeros(1, 1))

