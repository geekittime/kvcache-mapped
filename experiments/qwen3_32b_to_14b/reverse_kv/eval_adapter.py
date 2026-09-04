"""Isolated evaluation adapter for source=32B, target=14B, and mapped 14B."""

from __future__ import annotations

import datetime as dt
import importlib
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml

from .eval_common import model_artifact_identity as reverse_model_artifact_identity
from .protocol import DIRECTION, EXPERIMENT_ROOT, REPOSITORY_ROOT, sha256_file
from .runtime import ReverseMappedPrefixCacheBuilder


FORMAL_HELLASWAG_TOPOLOGY = {
    "source": {
        "devices": (0, 1, 2, 3, 4),
        "layer_counts": (12, 13, 14, 13, 12),
        "model": (EXPERIMENT_ROOT / "../../../../models/Qwen3-32B").resolve(),
    },
    "target": {
        "devices": (5, 6, 7),
        "layer_counts": (13, 14, 13),
        "model": (EXPERIMENT_ROOT / "../../../../models/Qwen3-14B").resolve(),
    },
}


def _validate_formal_eval_topology() -> None:
    config_path = EXPERIMENT_ROOT / "config/eval.yaml"
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runtime = raw.get("runtime") if isinstance(raw, dict) else None
    paths = raw.get("paths") if isinstance(raw, dict) else None
    if not isinstance(runtime, dict) or not isinstance(paths, dict):
        raise ValueError("formal HellaSwag config lacks runtime/paths")
    for role, contract in FORMAL_HELLASWAG_TOPOLOGY.items():
        if tuple(runtime.get(f"{role}_devices", ())) != contract["devices"]:
            raise ValueError(f"formal HellaSwag {role} devices changed")
        if tuple(runtime.get(f"{role}_layer_counts", ())) != contract["layer_counts"]:
            raise ValueError(f"formal HellaSwag {role} layer counts changed")
        if Path(paths.get(f"{role}_model", "")).resolve() != contract["model"]:
            raise ValueError(f"formal HellaSwag {role} model changed")


def _load_implementation() -> Any:
    scripts = REPOSITORY_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    module = importlib.import_module("evaluate")
    expected = (scripts / "evaluate.py").resolve()
    if Path(module.__file__).resolve() != expected:
        raise ImportError(f"unexpected evaluation implementation: {module.__file__}")
    return module


def run_evaluation(extra_argv: Sequence[str] = ()) -> int:
    _validate_formal_eval_topology()
    implementation = _load_implementation()
    original_load = implementation.load_qwen3

    def reverse_load(model_path: str | Path, *, role: str | None = None, **kwargs: Any) -> Any:
        if role not in ("source", "target"):
            raise ValueError(f"invalid semantic role: {role!r}")
        contract = FORMAL_HELLASWAG_TOPOLOGY[role]
        if Path(model_path).resolve() != contract["model"]:
            raise ValueError(f"formal HellaSwag {role} model path changed")
        observed_devices = tuple(int(value) for value in kwargs.get("devices", ()))
        if observed_devices != contract["devices"]:
            raise ValueError(f"formal HellaSwag {role} devices changed")
        supplied_counts = kwargs.pop("layer_counts", None)
        if supplied_counts is not None and tuple(supplied_counts) != contract["layer_counts"]:
            raise ValueError(f"formal HellaSwag {role} layer counts changed")
        kwargs["layer_counts"] = contract["layer_counts"]
        native_role = {"source": "target", "target": "source"}[role]
        model = original_load(model_path, role=native_role, **kwargs)
        setattr(
            model,
            "_reverse_semantic_role_binding",
            {
                "semantic_role": role,
                "capture_native_forward_role": native_role,
                "devices": list(contract["devices"]),
                "layer_counts": list(contract["layer_counts"]),
                "explicit_exact_counts": True,
            },
        )
        return model

    def reverse_output(root: Path, supplied: Path | None) -> Path:
        if supplied is not None:
            output = supplied.resolve()
            try:
                output.relative_to(EXPERIMENT_ROOT)
            except ValueError as exc:
                raise ValueError("evaluation output must remain inside reverse experiment") from exc
            output.mkdir(parents=True, exist_ok=True)
            return output
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = root / "runs" / f"{stamp}_evaluation_{DIRECTION}"
        output.mkdir(parents=True, exist_ok=False)
        return output

    original_project_code = implementation.project_code_provenance

    def reverse_project_code(root: Path) -> dict[str, Any]:
        evidence = dict(original_project_code(REPOSITORY_ROOT))
        reverse_files = {}
        for path in sorted((EXPERIMENT_ROOT / "reverse_kv").glob("*.py")):
            reverse_files[str(path.relative_to(EXPERIMENT_ROOT))] = sha256_file(path)
        for path in sorted((EXPERIMENT_ROOT / "scripts").glob("*.py")):
            reverse_files[str(path.relative_to(EXPERIMENT_ROOT))] = sha256_file(path)
        evidence["reverse_direction_code"] = reverse_files
        evidence["reverse_direction"] = DIRECTION
        freeze = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
        if not freeze.is_file():
            raise FileNotFoundError("formal reverse evaluation requires code_freeze.json")
        evidence["reverse_code_freeze"] = {
            "path": str(freeze.resolve()),
            "sha256": sha256_file(freeze),
            "nbytes": freeze.stat().st_size,
        }
        return evidence

    implementation.load_qwen3 = reverse_load
    implementation.MappedPrefixCacheBuilder = ReverseMappedPrefixCacheBuilder
    # The shared forward evaluator intentionally uses a lightweight model
    # identity.  Formal reverse HellaSwag must instead bind every weight-shard
    # byte, matching CoQA and latency in this independent experiment.
    implementation.model_artifact_identity = reverse_model_artifact_identity
    implementation._make_output_directory = reverse_output
    implementation.project_code_provenance = reverse_project_code
    argv = [
        "--config",
        str(EXPERIMENT_ROOT / "config/eval.yaml"),
        *map(str, extra_argv),
    ]
    return int(implementation.main(argv))


__all__ = ["FORMAL_HELLASWAG_TOPOLOGY", "run_evaluation"]
