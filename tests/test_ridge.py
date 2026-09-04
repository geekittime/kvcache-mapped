from __future__ import annotations

import numpy as np
import pytest
import torch

from kvcache_transfer.ridge import (
    CenteredRidgeAccumulator,
    RidgeInputError,
    RidgeSolveError,
    StreamingFeatureMeans,
    fit_centered_ridge,
    fit_centered_ridge_streaming,
    normal_equation_relative_residual,
)


def _synthetic(
    samples: int = 257, source_features: int = 7, target_features: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260813)
    x = torch.randn(samples, source_features, generator=generator)
    weight = torch.randn(source_features, target_features, generator=generator)
    bias = torch.linspace(-3.0, 2.0, target_features)
    noise = 0.01 * torch.randn(samples, target_features, generator=generator)
    return x, x @ weight + bias + noise


def _numpy_reference(
    x: torch.Tensor, y: torch.Tensor, ridge_lambda: float
) -> tuple[np.ndarray, np.ndarray]:
    x64 = x.double().numpy()
    y64 = y.double().numpy()
    mean_x = x64.mean(axis=0)
    mean_y = y64.mean(axis=0)
    xc = x64 - mean_x
    yc = y64 - mean_y
    gram = xc.T @ xc
    weight = np.linalg.solve(gram + ridge_lambda * np.eye(gram.shape[0]), xc.T @ yc)
    bias = mean_y - mean_x @ weight
    return weight, bias


def test_centered_ridge_matches_fp64_reference_and_recovers_bias() -> None:
    x, y = _synthetic()
    solution = fit_centered_ridge(x, y, ridge_lambda=0.01)
    expected_weight, expected_bias = _numpy_reference(x, y, 0.01)

    assert solution.weight.dtype == torch.float32
    assert solution.bias.dtype == torch.float32
    np.testing.assert_allclose(solution.weight.numpy(), expected_weight, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(solution.bias.numpy(), expected_bias, rtol=2e-5, atol=2e-5)


def test_lambda_is_added_to_unnormalized_gram() -> None:
    # Centered values are [-2, 0, 2], hence Xc.T@Xc == 8.  With Y == 3X,
    # Xc.T@Yc == 24 and lambda=2 must give W=24/(8+2)=2.4.  Dividing the
    # Gram by N (or scaling lambda by N) would produce a different answer.
    x = torch.tensor([[1.0], [3.0], [5.0]])
    y = 3.0 * x + 7.0
    solution = fit_centered_ridge(x, y, ridge_lambda=2.0)

    torch.testing.assert_close(solution.weight, torch.tensor([[2.4]]))
    # The intercept is derived after centering and is never penalized.
    torch.testing.assert_close(solution.bias, torch.tensor([8.8]))


def test_streaming_two_pass_statistics_equal_direct_centered_products() -> None:
    x, y = _synthetic(samples=103, source_features=9, target_features=6)
    boundaries = (0, 13, 52, 54, 103)

    def chunks():
        return [(x[a:b], y[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])]

    solution, statistics = fit_centered_ridge_streaming(chunks, ridge_lambda=0.01)
    mean_x = x.float().mean(dim=0)
    mean_y = y.float().mean(dim=0)
    xc = x.float() - mean_x
    yc = y.float() - mean_y

    assert statistics.n_samples == x.shape[0]
    torch.testing.assert_close(statistics.mean_x, mean_x, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(statistics.mean_y, mean_y, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(statistics.xtx, xc.T @ xc, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(statistics.xty, xc.T @ yc, rtol=2e-5, atol=2e-5)
    assert normal_equation_relative_residual(statistics, solution) < 2e-6


def test_bf16_inputs_are_accumulated_and_solved_in_fp32() -> None:
    x, y = _synthetic(samples=79)
    x_bf16, y_bf16 = x.bfloat16(), y.bfloat16()
    solution, statistics = fit_centered_ridge_streaming(
        lambda: ((x_bf16[:31], y_bf16[:31]), (x_bf16[31:], y_bf16[31:])),
        ridge_lambda=0.01,
    )

    for tensor in (
        statistics.mean_x,
        statistics.mean_y,
        statistics.xtx,
        statistics.xty,
        solution.weight,
        solution.bias,
    ):
        assert tensor.dtype == torch.float32


def test_joint_multi_head_solve_equals_independent_target_head_solves() -> None:
    # The production solve concatenates all target KV heads along the output
    # feature axis.  Since every output column shares X and the same ridge
    # system, it must equal solving each head independently.
    generator = torch.Generator().manual_seed(17)
    samples, source_features, heads, head_dim = 211, 11, 4, 3
    x = torch.randn(samples, source_features, generator=generator)
    y = torch.randn(samples, heads, head_dim, generator=generator)

    joint = fit_centered_ridge(x, y.reshape(samples, -1), ridge_lambda=0.07)
    independent = [
        fit_centered_ridge(x, y[:, head], ridge_lambda=0.07) for head in range(heads)
    ]
    expected_weight = torch.cat([result.weight for result in independent], dim=1)
    expected_bias = torch.cat([result.bias for result in independent])

    torch.testing.assert_close(joint.weight, expected_weight, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(joint.bias, expected_bias, rtol=2e-5, atol=2e-5)


def test_streaming_detects_mismatched_pass_sample_counts() -> None:
    x, y = _synthetic(samples=12)
    calls = 0

    def inconsistent_chunks():
        nonlocal calls
        calls += 1
        if calls == 1:
            return ((x, y),)
        return ((x[:-1], y[:-1]),)

    with pytest.raises(RidgeInputError, match="different sample counts"):
        fit_centered_ridge_streaming(inconsistent_chunks)


def test_accumulators_reject_feature_layout_changes() -> None:
    means = StreamingFeatureMeans()
    means.update(torch.ones(4, 3), torch.ones(4, 2))
    with pytest.raises(RidgeInputError, match="feature dimensions changed"):
        means.update(torch.ones(2, 4), torch.ones(2, 2))

    fixed_means = means.finalize()
    statistics = CenteredRidgeAccumulator(fixed_means)
    with pytest.raises(RidgeInputError, match="do not match"):
        statistics.update(torch.ones(4, 3), torch.ones(4, 5))


def test_cholesky_failure_is_explicit_and_never_silently_regularized() -> None:
    # With lambda=0, two identical columns make the centered Gram singular.
    base = torch.arange(8, dtype=torch.float32).reshape(-1, 1)
    x = torch.cat([base, base], dim=1)
    y = base * 2
    with pytest.raises(RidgeSolveError) as error:
        fit_centered_ridge(x, y, ridge_lambda=0.0, solver="cholesky")
    assert error.value.cholesky_info is not None


def test_predict_validates_feature_count() -> None:
    x, y = _synthetic(samples=20)
    solution = fit_centered_ridge(x, y)
    with pytest.raises(RidgeInputError, match="mapper expects"):
        solution.predict(torch.ones(3, x.shape[1] + 1))

