from __future__ import annotations

from src.contracts import load_config
from src.runtime_adapter import validate_method_bindings


def test_real_candidate_and_reverse_runtime_bindings_are_content_addressed() -> None:
    config = load_config(verify_inputs=True)
    bindings = validate_method_bindings(config, deep_model_hash=False)
    assert len(bindings.digest) == 64
    assert bindings.baseline_manifest_ref.sha256 == (
        "382a1deb6f4fd33c824fe782d956f46bdfb2b876081cb36487616e4de92a8ee2"
    )
    assert bindings.shrinkage["state"] == "ready"
    assert bindings.tail_refresh["state"] == "ready"
    assert bindings.tail_refresh["allowed_replay_tokens"] == [8, 32]
