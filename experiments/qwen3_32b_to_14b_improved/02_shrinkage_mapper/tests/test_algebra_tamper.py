from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapper_algebra_audit import require_bitwise_equal  # noqa: E402


class AlgebraTamperTests(unittest.TestCase):
    def test_one_bit_tamper_fails_closed(self) -> None:
        expected = torch.tensor(
            [[1.0, -2.0], [3.5, 0.125]], dtype=torch.float32
        )
        require_bitwise_equal(expected.clone(), expected, "control")
        tampered = expected.clone()
        bits = tampered.view(torch.int32).reshape(-1)
        bits[0] = bits[0] ^ 1
        with self.assertRaisesRegex(ValueError, "not bitwise equal"):
            require_bitwise_equal(tampered, expected, "one-bit-tamper")


if __name__ == "__main__":
    unittest.main()

