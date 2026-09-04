from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from kvcache_transfer.activation_store import write_bf16_activations
from kvcache_transfer.select_layers import select_from_head_scores


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/select_layers.py"
SPEC = importlib.util.spec_from_file_location("selection_pipeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PIPELINE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PIPELINE
SPEC.loader.exec_module(PIPELINE)


def test_sufficient_statistics_match_explicit_affine_ols_r2() -> None:
    generator = torch.Generator().manual_seed(101)
    samples, heads, dimension = 90, 2, 3
    x = torch.randn(samples, heads, dimension, generator=generator, dtype=torch.float32)
    coefficients = torch.randn(heads, dimension, dimension, generator=generator)
    bias = torch.randn(heads, dimension, generator=generator)
    y = torch.einsum("nhi,hio->nho", x, coefficients) + bias
    y += 0.02 * torch.randn(y.shape, generator=generator)

    x_mean = x.mean(0)
    y_mean = y.mean(0)
    xc = x - x_mean
    yc = y - y_mean
    gram = torch.einsum("nhi,nhj->hij", xc, xc)
    cross = torch.einsum("nhi,nhj->hij", xc, yc)
    weights, solver_codes = PIPELINE._solve_batched_ols(
        gram, cross, fallback="error"
    )
    target_sst = torch.sum((y.double() - y.double().mean(0)).square(), dim=0)
    target_fit_energy = torch.sum((y - y_mean).double().square(), dim=0)

    uniform, pooled, raw_minimum = PIPELINE.score_from_sufficient_statistics(
        gram, cross, weights, target_fit_energy, target_sst
    )
    prediction = torch.einsum("nhi,hio->nho", x, weights) + (
        y_mean - torch.einsum("hi,hio->ho", x_mean, weights)
    )
    error = (y.double() - prediction.double()).square().sum(0)
    expected_channels = 1.0 - error / target_sst
    expected_uniform = expected_channels.mean(1).numpy()
    expected_pooled = (1.0 - error.sum(1) / target_sst.sum(1)).numpy()

    np.testing.assert_allclose(uniform, expected_uniform, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(pooled, expected_pooled, rtol=2e-5, atol=2e-6)
    assert (solver_codes.cpu().numpy() == PIPELINE.SOLVER_CHOLESKY).all()
    assert raw_minimum.min() >= -1e-4


def test_selection_uses_uniform_scores_not_global_diagnostics() -> None:
    key_uniform = np.asarray([[[0.9], [0.8]]], dtype=np.float64)
    value_uniform = np.asarray([[[0.9], [0.8]]], dtype=np.float64)
    deliberately_opposite_global = np.asarray([[[0.0], [1.0]]], dtype=np.float64)
    del deliberately_opposite_global  # Never supplied to the ranking interface.
    selected = select_from_head_scores(key_uniform, value_uniform, k=1)
    assert selected.selected_layers == ((0,),)


def _write_tiny_store(
    root: Path,
    cache_type: str,
    layer: int,
    values: torch.Tensor,
    *,
    model_name: str,
) -> None:
    path = root / cache_type / f"layer_{layer:03d}.bf16.bin"
    write_bf16_activations(
        path,
        values.to(torch.bfloat16),
        provenance={
            "model_name": model_name,
            "token_sha256": "a" * 64,
            "split": "train",
            "stride": 4,
            "stride_start": 0,
            "sequences": 2,
            "cache_type": cache_type,
            "layer": layer,
        },
    )


def test_tiny_cli_end_to_end_and_resume(tmp_path: Path) -> None:
    samples, heads, dimension = 48, 2, 3
    generator = torch.Generator().manual_seed(9)
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_layers, target_layers = 3, 2
    source_values: dict[tuple[str, int], torch.Tensor] = {}
    for cache_type in PIPELINE.CACHE_TYPES:
        for layer in range(source_layers):
            values = torch.randn(samples, heads, dimension, generator=generator)
            source_values[(cache_type, layer)] = values
            _write_tiny_store(source_root, cache_type, layer, values, model_name="source")
        for layer in range(target_layers):
            # K target layer 0 comes from source 1 and target layer 1 from source
            # 2.  V swaps those, making the equal-K/V reduction deterministic.
            chosen = (layer + (1 if cache_type == "k_content" else 2)) % source_layers
            values = 1.5 * source_values[(cache_type, chosen)] + 0.1
            _write_tiny_store(target_root, cache_type, layer, values, model_name="target")

    output = tmp_path / "selection"
    statistics = tmp_path / "statistics"
    project = tmp_path / "project"
    project.mkdir()
    (project / "REPRODUCTION.md").write_text("# test\n", encoding="utf-8")
    command = [
        str(ROOT / ".venv/bin/python"),
        str(SCRIPT),
        "--project-root",
        str(project),
        "--source-activations",
        str(source_root),
        "--target-activations",
        str(target_root),
        "--output",
        str(output),
        "--statistics",
        str(statistics),
        "--devices",
        "cpu",
        "--source-layers",
        str(source_layers),
        "--target-layers",
        str(target_layers),
        "--kv-heads",
        str(heads),
        "--head-dim",
        str(dimension),
        "--top-k",
        "2",
        "--chunk-tokens",
        "11",
        "--debug-allow-nonpaper-architecture",
        "--no-ledger",
    ]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    assert '"status": "complete"' in first.stdout
    manifest = json.loads((output / "selected_layers.json").read_text())
    assert len(manifest["selected_source_layers_rank_order"]) == target_layers
    assert all(len(row) == 2 for row in manifest["selected_source_layers_rank_order"])
    assert manifest["selection_rule"]["benchmark_data_used"] is False
    with np.load(output / "scores.npz", allow_pickle=False) as scores:
        assert scores["key_head_uniform_r2"].shape == (
            target_layers,
            source_layers,
            heads,
        )
        assert scores["combined_equal_kv_uniform_r2"].shape == (
            target_layers,
            source_layers,
        )

    # A second invocation validates and skips every checkpoint.  It must not
    # silently overwrite immutable output artifacts.
    second = subprocess.run(command, check=True, capture_output=True, text=True)
    assert '"status": "complete"' in second.stdout
    run_statuses = [
        json.loads(path.read_text())
        for path in sorted((project / "runs").glob("*/status.json"))
    ]
    assert all(item["status"] == "complete" for item in run_statuses)
