"""Configuration loading and paper-specific invariants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ArchitectureSpec:
    source_layers: int = 40
    target_layers: int = 64
    source_kv_heads: int = 8
    target_kv_heads: int = 8
    source_head_dim: int = 128
    target_head_dim: int = 128
    selected_source_layers: int = 8

    @property
    def source_features(self) -> int:
        return (
            self.selected_source_layers
            * self.source_kv_heads
            * self.source_head_dim
        )

    @property
    def target_features(self) -> int:
        return self.target_kv_heads * self.target_head_dim

    @property
    def mapper_weight_count(self) -> int:
        return (
            2
            * self.target_layers
            * self.source_features
            * self.target_features
        )

    @property
    def mapper_bias_count(self) -> int:
        return 2 * self.target_layers * self.target_features


PAPER_ARCHITECTURE = ArchitectureSpec()


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without implicit project-global state."""
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(value).__name__}")
    return value


def assert_paper_architecture(
    config: Any,
    *,
    role: str | None = None,
) -> None:
    """Fail early unless a checkpoint is the intended Qwen3 paper endpoint.

    ``role=None`` checks the matched-KV invariants shared by both endpoints.
    Formal extraction passes ``source`` or ``target`` so layer count, query
    heads, and scale-specific dimensions cannot silently accept another Qwen3
    checkpoint that happens to use the same KV layout.
    """

    if role not in (None, "source", "target"):
        raise ValueError(f"role must be source, target, or None; got {role!r}")
    expected = {
        "model_type": "qwen3",
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rope_theta": 1_000_000,
        "use_sliding_window": False,
        "sliding_window": None,
    }
    if role == "source":
        expected.update(
            {
                "num_hidden_layers": 40,
                "num_attention_heads": 40,
                "hidden_size": 5120,
                "intermediate_size": 17408,
            }
        )
    elif role == "target":
        expected.update(
            {
                "num_hidden_layers": 64,
                "num_attention_heads": 64,
                "hidden_size": 5120,
                "intermediate_size": 25600,
            }
        )
    for field, wanted in expected.items():
        observed = getattr(config, field, None)
        if observed != wanted:
            raise ValueError(f"{field}: expected {wanted!r}, observed {observed!r}")
