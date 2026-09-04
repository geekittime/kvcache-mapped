from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "shrinkage_mapper.py"
SPEC = importlib.util.spec_from_file_location("shrinkage_mapper", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ShrinkageMathTests(unittest.TestCase):
    def test_analytic_alpha_recovers_known_mixture(self) -> None:
        rng = np.random.default_rng(17)
        d = rng.normal(size=(40, 8, 128))
        expected = np.linspace(0.0, 1.0, 8)
        e_dot_d = d * d * expected.reshape(1, 8, 1)
        d2 = d * d
        observed = MODULE.fit_alpha(e_dot_d, d2, list(range(40)))
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2e-15)

    def test_alpha_is_convexly_clipped(self) -> None:
        d2 = np.ones((3, 8, 128), dtype=np.float64)
        factors = np.asarray([-2.0, -0.1, 0.0, 0.2, 0.8, 1.0, 1.1, 4.0])
        e_dot_d = d2 * factors.reshape(1, 8, 1)
        observed = MODULE.fit_alpha(e_dot_d, d2, [0, 1, 2])
        np.testing.assert_allclose(
            observed, np.clip(factors, 0.0, 1.0), rtol=0.0, atol=2e-15
        )

    def test_weight_fusion_equals_functional_mixture(self) -> None:
        generator = torch.Generator().manual_seed(23)
        # Exercise the production dimensions because row embedding is the
        # scientific invariant that makes one-matmul deployment possible.
        w20 = torch.randn(20480, 1024, generator=generator)
        w4 = torch.randn(4096, 1024, generator=generator)
        b20 = torch.randn(1024, generator=generator)
        b4 = torch.randn(1024, generator=generator)
        alpha = np.linspace(0.0, 1.0, 8)
        fused_w = MODULE._fuse_component(w20, w4, alpha)
        fused_b = MODULE._fuse_bias(b20, b4, alpha)
        x = torch.randn(3, 20480, generator=generator)
        actual = torch.addmm(fused_b, x, fused_w)
        p20 = torch.addmm(b20, x, w20)
        p4 = torch.addmm(b4, x[:, :4096], w4)
        expanded = torch.as_tensor(alpha, dtype=torch.float32).repeat_interleave(128)
        expected = p4 + expanded * (p20 - p4)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-4)

    def test_metric_quadratic_matches_explicit_residual(self) -> None:
        rng = np.random.default_rng(29)
        sequences = 4
        samples = 7
        y = rng.normal(size=(sequences, samples, 8, 128))
        p4 = rng.normal(size=y.shape)
        p20 = rng.normal(size=y.shape)
        e = y - p4
        d = p20 - p4
        stats = {
            "e2": np.sum(e * e, axis=1),
            "e_dot_d": np.sum(e * d, axis=1),
            "d2": np.sum(d * d, axis=1),
            "target_mean": np.mean(y, axis=1),
            "target_m2": np.sum(np.square(y - np.mean(y, axis=1, keepdims=True)), axis=1),
            "flat_mean": np.mean(y.reshape(sequences, -1), axis=1),
            "flat_m2": np.sum(
                np.square(
                    y.reshape(sequences, -1)
                    - np.mean(y.reshape(sequences, -1), axis=1, keepdims=True)
                ),
                axis=1,
            ),
        }
        # The production helper has a fixed 256 observations per sequence.
        # Scale sufficient statistics and repeat samples to that cardinality.
        repeats = 256 // samples
        remainder = 256 - repeats * samples
        y256 = np.concatenate([np.tile(y, (1, repeats, 1, 1)), y[:, :remainder]], axis=1)
        p4_256 = np.concatenate([np.tile(p4, (1, repeats, 1, 1)), p4[:, :remainder]], axis=1)
        p20_256 = np.concatenate([np.tile(p20, (1, repeats, 1, 1)), p20[:, :remainder]], axis=1)
        e256 = y256 - p4_256
        d256 = p20_256 - p4_256
        flat = y256.reshape(sequences, -1)
        stats = {
            "e2": np.sum(e256 * e256, axis=1),
            "e_dot_d": np.sum(e256 * d256, axis=1),
            "d2": np.sum(d256 * d256, axis=1),
            "target_mean": np.mean(y256, axis=1),
            "target_m2": np.sum(
                np.square(y256 - np.mean(y256, axis=1, keepdims=True)), axis=1
            ),
            "flat_mean": np.mean(flat, axis=1),
            "flat_m2": np.sum(
                np.square(flat - np.mean(flat, axis=1, keepdims=True)), axis=1
            ),
        }
        alpha = np.linspace(0.1, 0.8, 8)
        observed = MODULE.metrics_for_alpha(stats, [0, 1, 2, 3], alpha, per_channel=True)
        explicit = e256 - alpha.reshape(1, 1, 8, 1) * d256
        self.assertAlmostEqual(observed["global_sse_fp64"], float(np.sum(explicit**2)), places=8)


class ProtocolTests(unittest.TestCase):
    def test_real_selection_is_exactly_embeddable(self) -> None:
        config = MODULE.load_config()
        k20 = MODULE._selection_rows(Path(config["upstream"]["k20_mapper"]))
        k4 = MODULE._selection_rows(Path(config["upstream"]["k4_mapper"]))
        self.assertEqual(len(k20), 40)
        self.assertTrue(all(row20[:4] == row4 for row20, row4 in zip(k20, k4, strict=True)))

    def test_split_is_deterministic_disjoint_and_balanced(self) -> None:
        config = MODULE.load_config()
        first = MODULE.build_split_manifest(config)
        second = MODULE.build_split_manifest(config)
        self.assertEqual(MODULE.canonical_bytes(first), MODULE.canonical_bytes(second))
        locked = first["locked_audit_sequence_ids"]
        folds = [row["validation_sequence_ids"] for row in first["cv_folds"]]
        tuning = [value for fold in folds for value in fold]
        self.assertEqual(len(locked), 10)
        self.assertEqual([len(fold) for fold in folds], [8] * 5)
        self.assertFalse(set(locked) & set(tuning))
        self.assertEqual(sorted(locked + tuning), list(range(50)))

    def test_output_paths_are_confined_to_candidate_directory(self) -> None:
        config = MODULE.load_config()
        root = MODULE.ROOT.resolve()
        for path in config["outputs"].values():
            Path(path).resolve().relative_to(root)


if __name__ == "__main__":
    unittest.main()
