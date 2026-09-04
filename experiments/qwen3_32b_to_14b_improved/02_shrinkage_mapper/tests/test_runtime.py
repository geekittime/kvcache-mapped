from __future__ import annotations

import sys
from pathlib import Path
import unittest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from runtime import (  # noqa: E402
    DEFAULT_MAPPER_MANIFEST,
    ShrinkageMapperBundle,
)


class RuntimeArtifactTests(unittest.TestCase):
    def test_bundle_and_one_layer_load_fail_closed(self) -> None:
        bundle = ShrinkageMapperBundle.open(
            DEFAULT_MAPPER_MANIFEST, verify_payloads=False
        )
        self.assertEqual(len(bundle.layers), 40)
        self.assertEqual(bundle.architecture.source_features, 20480)
        self.assertEqual(bundle.architecture.target_features, 1024)
        self.assertEqual(
            bundle.completion_receipt["status"],
            "complete_by_append_only_amendment",
        )
        layer = bundle.load_layer(0, device="cpu")
        self.assertEqual(tuple(layer.key_weight.shape), (20480, 1024))
        self.assertEqual(tuple(layer.value_weight.shape), (20480, 1024))
        self.assertEqual(layer.source_layers[:4], tuple(range(4)))
        self.assertEqual(len(layer.source_layers), 20)


if __name__ == "__main__":
    unittest.main()
