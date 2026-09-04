"""Reverse evaluator adapter with the extension overlay in the fingerprint."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Sequence

from .protocol import (
    EXPERIMENT_ROOT,
    EXTENSION_CODE_FREEZE,
    REPOSITORY_ROOT,
)

from reverse_kv.eval_adapter import (  # noqa: E402
    FORMAL_HELLASWAG_TOPOLOGY,
    _load_implementation,
    _validate_formal_eval_topology,
)
from reverse_kv.eval_common import model_artifact_identity as reverse_model_artifact_identity  # noqa: E402
from reverse_kv.protocol import DIRECTION, read_json, sha256_file  # noqa: E402
from reverse_kv.runtime import ReverseMappedPrefixCacheBuilder  # noqa: E402


def extension_project_code_identity(original_project_code: Any | None = None) -> dict[str, Any]:
    implementation = _load_implementation()
    builder = original_project_code or implementation.project_code_provenance
    evidence = dict(builder(REPOSITORY_ROOT))
    reverse_files: dict[str, str] = {}
    for path in sorted((EXPERIMENT_ROOT / "reverse_kv").glob("*.py")):
        reverse_files[str(path.relative_to(EXPERIMENT_ROOT))] = sha256_file(path)
    for path in sorted((EXPERIMENT_ROOT / "scripts").glob("*.py")):
        reverse_files[str(path.relative_to(EXPERIMENT_ROOT))] = sha256_file(path)
    evidence["reverse_direction_code"] = reverse_files
    evidence["reverse_direction"] = DIRECTION
    base_freeze = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
    evidence["reverse_code_freeze"] = {
        "path": str(base_freeze.resolve()),
        "sha256": sha256_file(base_freeze),
        "nbytes": base_freeze.stat().st_size,
    }
    overlay = read_json(EXTENSION_CODE_FREEZE)
    overlay_sidecar = EXTENSION_CODE_FREEZE.with_name(f"{EXTENSION_CODE_FREEZE.name}.sha256.json")
    evidence["five_task_extension_overlay"] = {
        "path": str(EXTENSION_CODE_FREEZE.resolve()),
        "sha256": sha256_file(EXTENSION_CODE_FREEZE),
        "nbytes": EXTENSION_CODE_FREEZE.stat().st_size,
        "sha256_sidecar": {
            "path": str(overlay_sidecar.resolve()),
            "sha256": sha256_file(overlay_sidecar),
            "nbytes": overlay_sidecar.stat().st_size,
        },
        "inventory_digest": overlay["inventory_digest"],
        "paper_bit_exact": False,
    }
    return evidence


def run_extension_evaluation(extra_argv: Sequence[str] = ()) -> int:
    """Run the shared evaluator with reverse semantics and overlay provenance."""

    _validate_formal_eval_topology()
    implementation = _load_implementation()
    original_load = implementation.load_qwen3
    original_project_code = implementation.project_code_provenance

    def reverse_load(model_path: str | Path, *, role: str | None = None, **kwargs: Any) -> Any:
        if role not in ("source", "target"):
            raise ValueError(f"invalid semantic role: {role!r}")
        contract = FORMAL_HELLASWAG_TOPOLOGY[role]
        if Path(model_path).resolve() != contract["model"]:
            raise ValueError(f"formal extension {role} model path changed")
        observed_devices = tuple(int(value) for value in kwargs.get("devices", ()))
        if observed_devices != contract["devices"]:
            raise ValueError(f"formal extension {role} devices changed")
        supplied_counts = kwargs.pop("layer_counts", None)
        if supplied_counts is not None and tuple(supplied_counts) != contract["layer_counts"]:
            raise ValueError(f"formal extension {role} layer counts changed")
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
                raise ValueError("extension output must remain inside reverse experiment") from exc
            output.mkdir(parents=True, exist_ok=True)
            return output
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
        output = root / "runs" / f"{stamp}_five_task_extension_{DIRECTION}"
        output.mkdir(parents=True, exist_ok=False)
        return output

    implementation.load_qwen3 = reverse_load
    implementation.MappedPrefixCacheBuilder = ReverseMappedPrefixCacheBuilder
    implementation.model_artifact_identity = reverse_model_artifact_identity
    implementation._make_output_directory = reverse_output
    implementation.project_code_provenance = (
        lambda _root: extension_project_code_identity(original_project_code)
    )
    argv = ["--config", str(EXPERIMENT_ROOT / "config/eval.yaml"), *map(str, extra_argv)]
    return int(implementation.main(argv))


__all__ = ["extension_project_code_identity", "run_extension_evaluation"]
