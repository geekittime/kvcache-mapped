#!/usr/bin/env python3
"""Append-only publication receipt for the completed NO-GO probe."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "probe_v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ref(path: Path, *, relative: bool = False) -> dict[str, Any]:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or path.is_symlink():
        raise RuntimeError(f"not a regular non-symlink file: {path}")
    return {
        "path": str(path.relative_to(ROOT)) if relative else str(path.resolve()),
        "nbytes": st.st_size,
        "sha256": sha256(path),
    }


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"not a JSON object: {path}")
    return value


def atomic(path: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    results = read(RUN / "results.json")
    completed = read(RUN / "COMPLETED.json")
    preflight = read(RUN / "input_preflight.json")
    if results.get("status") != "complete" or results.get("decision") != "NO_GO":
        raise RuntimeError("probe is not a completed NO-GO")
    if results.get("locked_activation_evaluations") != 0 or results.get("locked_rows_indexed") is not False:
        raise RuntimeError("locked activation boundary changed")
    if completed != {"status": "complete", "results": ref(RUN / "results.json")}:
        raise RuntimeError("completion marker no longer binds results")
    if preflight.get("status") != "pass" or preflight.get("opaque_hashing_does_not_index_locked_values") is not True:
        raise RuntimeError("input preflight no longer passes")

    implementation_paths = [ROOT / "README.md", ROOT / "RESULTS.md", ROOT / "config.json"]
    for directory in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in sorted(directory.iterdir()):
            if path.name == "__pycache__":
                raise RuntimeError("local bytecode cache is forbidden")
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"nonregular implementation entry: {path}")
            if path.suffix == ".py":
                implementation_paths.append(path)
    implementation = [ref(path, relative=True) for path in sorted(implementation_paths)]
    scientific_paths = [
        RUN / "run_definition.json",
        RUN / "input_preflight.json",
        RUN / "idle_gate.json",
        RUN / "results.json",
        RUN / "COMPLETED.json",
        *sorted((RUN / "stats").glob("*.npz")),
    ]
    scientific = [ref(path, relative=True) for path in scientific_paths]
    receipt = {
        "schema_version": 1,
        "status": "complete_no_go_by_append_only_receipt",
        "decision": "NO_GO",
        "immutable_run_definition_initial_status": "running",
        "terminal_markers": {
            "results": ref(RUN / "results.json", relative=True),
            "completed": ref(RUN / "COMPLETED.json", relative=True),
        },
        "implementation_inventory": implementation,
        "implementation_inventory_sha256": hashlib.sha256(canonical_bytes(implementation)).hexdigest(),
        "scientific_inventory": scientific,
        "scientific_inventory_sha256": hashlib.sha256(canonical_bytes(scientific)).hexdigest(),
        "input_preflight": ref(RUN / "input_preflight.json", relative=True),
        "locked_activation_evaluations": 0,
        "locked_rows_indexed": False,
        "gpu_execution_performed": True,
        "models_loaded": False,
        "downstream_evaluation_performed": False,
        "downstream_or_superiority_claim_permitted": False,
        "claim_boundary": "mapper-geometry ablation only; failed pre-registered GO gate",
    }
    destination = RUN / "completion_receipt_v1.json"
    payload = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if destination.exists() and destination.read_bytes() != payload:
        raise RuntimeError("append-only completion receipt already differs")
    if not destination.exists():
        atomic(destination, payload)
    marker = {"path": destination.name, "nbytes": destination.stat().st_size, "sha256": sha256(destination)}
    marker_path = destination.with_suffix(destination.suffix + ".sha256.json")
    marker_payload = json.dumps(marker, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if marker_path.exists() and marker_path.read_bytes() != marker_payload:
        raise RuntimeError("append-only receipt sidecar already differs")
    if not marker_path.exists():
        atomic(marker_path, marker_payload)
    print(json.dumps({"receipt": ref(destination), "sidecar": ref(marker_path)}, sort_keys=True))


if __name__ == "__main__":
    main()

