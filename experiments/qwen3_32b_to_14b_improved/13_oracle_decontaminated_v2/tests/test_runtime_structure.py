from __future__ import annotations

import ast
from pathlib import Path

from src.contracts import ROOT


def test_gpu_runner_has_no_eager_torch_or_transformers_import() -> None:
    path = ROOT / "src" / "gpu_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {"torch", "transformers", "kvcache_transfer"}
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(forbidden)


def test_runner_contains_one_explicit_k20_builder_call_and_pre_torch_deep_gate() -> None:
    source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
    assert source.count("mapped_cache = builder(prefix)") == 1
    assert source.index('"status": "complete_before_torch_import"') < source.index("_import_verified_legacy(config)")
    assert source.index("_deep_mapper_identity(config)") < source.index("_import_verified_legacy(config)")


def test_no_formal_config_or_gpu_result_is_checked_in() -> None:
    assert not list(ROOT.glob("config.formal*.json"))
    assert not list((ROOT / "runs").rglob("completion_receipt.json"))

