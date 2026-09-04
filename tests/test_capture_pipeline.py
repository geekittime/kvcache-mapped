from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kvcache_transfer.activation_store import (
    commit_existing_bf16_payload,
    open_bf16_activations,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "capture_activations_script", PROJECT_ROOT / "scripts/capture_activations.py"
)
assert SPEC is not None and SPEC.loader is not None
capture_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_script)


SEQUENCE_NBYTES = (
    capture_script.SAMPLED_PER_SEQUENCE * capture_script.ROW_NBYTES
)


def _sequence_blob(value: int) -> bytes:
    return bytes([value]) * SEQUENCE_NBYTES


def _write_all_payloads(stage: Path, layers: int, value: bytes) -> list[Path]:
    handles = capture_script._open_append_handles(stage, layers)
    try:
        for handle in handles:
            handle.write(value)
    finally:
        capture_script._close_handles(handles)
    return capture_script._expected_payloads(stage, layers)


def test_chunk_writer_completes_partial_element_writes() -> None:
    class PartialWriter:
        def __init__(self) -> None:
            self.value = bytearray()

        def write(self, value: memoryview) -> int:
            count = min(len(value), 64 * 1024)
            self.value.extend(value[:count])
            return count

    tensor = torch.zeros(
        1,
        capture_script.SAMPLED_PER_SEQUENCE,
        capture_script.KV_HEADS,
        capture_script.HEAD_DIM,
        dtype=torch.bfloat16,
    )
    writer = PartialWriter()
    capture_script._write_bf16_chunk(writer, tensor)
    assert len(writer.value) == SEQUENCE_NBYTES
    assert writer.value == bytes(SEQUENCE_NBYTES)


@pytest.mark.parametrize(
    ("layers", "expected_bytes"),
    [
        (40, 20_971_520_000),
        (64, 33_554_432_000),
    ],
)
def test_formal_capture_storage_estimate_is_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layers: int,
    expected_bytes: int,
) -> None:
    stage = tmp_path / ".capture.inprogress"
    stage.mkdir()
    monkeypatch.setattr(
        capture_script.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1 << 40),
    )

    estimate = capture_script._capture_storage_preflight(stage, layers, 500)

    assert estimate["payload_files"] == 2 * layers
    assert estimate["per_payload_bytes"] == 262_144_000
    assert estimate["estimated_final_payload_bytes"] == expected_bytes
    assert estimate["remaining_payload_bytes"] == expected_bytes


def test_capture_storage_preflight_counts_resume_bytes_and_gates_low_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / ".capture.inprogress"
    payloads = _write_all_payloads(stage, layers=1, value=_sequence_blob(0x12))
    final_bytes = 2 * 2 * SEQUENCE_NBYTES
    existing_bytes = sum(path.stat().st_size for path in payloads)
    required = final_bytes - existing_bytes + capture_script.MIN_CAPTURE_HEADROOM_BYTES
    monkeypatch.setattr(
        capture_script.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=required - 1),
    )

    with pytest.raises(OSError, match="insufficient free space"):
        capture_script._capture_storage_preflight(stage, layers=1, sequences=2)


def test_main_persists_failure_after_run_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_run_creation(_argv: object) -> int:
        capture_script._ACTIVE_RUN_PATH = tmp_path
        raise RuntimeError("synthetic capture failure")

    monkeypatch.setattr(capture_script, "_main", fail_after_run_creation)
    with pytest.raises(RuntimeError, match="synthetic capture failure"):
        capture_script.main([])

    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert "rerun the identical command" in status["resume"]
    assert capture_script._ACTIVE_RUN_PATH is None


def test_commit_existing_payload_is_a_verified_activation(tmp_path: Path) -> None:
    value = torch.randn(7, 2, 4).to(torch.bfloat16).contiguous()
    payload = tmp_path / "layer_000.bf16.bin"
    payload.write_bytes(value.view(torch.uint16).numpy().tobytes())
    metadata = commit_existing_bf16_payload(
        payload, value.shape, provenance={"purpose": "unit-test"}
    )
    assert metadata.shape == tuple(value.shape)
    with open_bf16_activations(payload) as mapped:
        torch.testing.assert_close(mapped.to_torch(), value)


def test_resume_truncates_every_payload_to_durable_progress_and_fsyncs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / ".capture.inprogress"
    layers = 2
    durable = _sequence_blob(0x11)
    payloads = _write_all_payloads(stage, layers, durable)
    for index, payload in enumerate(payloads):
        with payload.open("ab") as handle:
            # Cover a whole uncheckpointed sequence followed by a partial
            # element/sequence suffix.
            handle.write(_sequence_blob(0x20 + index))
            handle.write(b"partial-element")
        capture_script.sidecar_path(payload).write_text("stale", encoding="utf-8")
    (stage / "manifest.json").write_text("stale", encoding="utf-8")

    real_fsync = os.fsync
    fsynced_paths: set[Path] = set()

    def recording_fsync(descriptor: int) -> None:
        try:
            fsynced_paths.add(Path(os.readlink(f"/proc/self/fd/{descriptor}")).resolve())
        except OSError:
            pass
        real_fsync(descriptor)

    monkeypatch.setattr(capture_script.os, "fsync", recording_fsync)
    capture_script._validate_progress_payloads(stage, layers, 1)

    for payload in payloads:
        assert payload.read_bytes() == durable
        assert payload.resolve() in fsynced_paths
        assert not capture_script.sidecar_path(payload).exists()
    assert not (stage / "manifest.json").exists()


def test_resume_truncates_only_overlong_payloads_in_a_partial_sequence(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".capture.inprogress"
    layers = 2
    durable = _sequence_blob(0x31)
    payloads = _write_all_payloads(stage, layers, durable)
    suffixes = {payloads[0]: b"\x01", payloads[-1]: _sequence_blob(0x32)[:777]}
    for payload, suffix in suffixes.items():
        with payload.open("ab") as handle:
            handle.write(suffix)

    capture_script._validate_progress_payloads(stage, layers, 1)

    for payload in payloads:
        assert payload.read_bytes() == durable


def test_missing_progress_means_zero_durable_bytes(tmp_path: Path) -> None:
    stage = tmp_path / ".capture.inprogress"
    payloads = _write_all_payloads(stage, layers=1, value=b"partial-first-sequence")

    capture_script._validate_progress_payloads(stage, layers=1, completed_sequences=0)

    assert all(payload.stat().st_size == 0 for payload in payloads)


def test_resume_rejects_short_durable_payload_before_changing_siblings(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".capture.inprogress"
    layers = 1
    durable = _sequence_blob(0x41)
    payloads = _write_all_payloads(stage, layers, durable)
    payloads[0].write_bytes(durable[:-1])
    with payloads[1].open("ab") as handle:
        handle.write(b"uncommitted")
    marker = capture_script.sidecar_path(payloads[1])
    marker.write_text("stale", encoding="utf-8")
    overlong_size = payloads[1].stat().st_size

    with pytest.raises(ValueError, match="shorter than durable checkpoint"):
        capture_script._validate_progress_payloads(stage, layers, 1)

    assert payloads[0].stat().st_size == SEQUENCE_NBYTES - 1
    assert payloads[1].stat().st_size == overlong_size
    assert marker.exists()


def test_fsync_every_greater_than_one_resumes_without_duplicate_sequences(
    tmp_path: Path,
) -> None:
    stage = tmp_path / ".capture.inprogress"
    progress_path = stage / "progress.json"
    layers = 1
    fsync_every = 2
    handles = capture_script._open_append_handles(stage, layers)
    try:
        # Four sequences are planned.  Sequence 2 (zero based) reaches every
        # payload, then the process crashes before the next every-two checkpoint.
        for sequence_index in range(3):
            for handle in handles:
                handle.write(_sequence_blob(sequence_index))
            durable_count = sequence_index + 1
            if durable_count % fsync_every == 0:
                capture_script._durable_checkpoint(
                    handles,
                    progress_path,
                    {
                        "completed_sequences": durable_count,
                        "run_fingerprint": "f" * 64,
                    },
                )
    finally:
        capture_script._close_handles(handles)

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completed_sequences"] == 2
    capture_script._validate_progress_payloads(
        stage, layers, int(progress["completed_sequences"])
    )

    handles = capture_script._open_append_handles(stage, layers)
    try:
        for sequence_index in range(2, 4):
            for handle in handles:
                handle.write(_sequence_blob(sequence_index))
        capture_script._durable_checkpoint(
            handles,
            progress_path,
            {"completed_sequences": 4, "run_fingerprint": "f" * 64},
        )
    finally:
        capture_script._close_handles(handles)

    expected = b"".join(_sequence_blob(index) for index in range(4))
    capture_script._validate_progress_payloads(stage, layers, 4)
    for payload in capture_script._expected_payloads(stage, layers):
        assert payload.read_bytes() == expected


def test_partial_sidecar_commit_can_resume(tmp_path: Path) -> None:
    stage = tmp_path / ".capture.inprogress"
    output = tmp_path / "capture"
    model = {
        "path": "/model",
        "role": "source",
        "config_sha256": "a" * 64,
        "weight_index_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
        "architecture": {"num_hidden_layers": 1},
    }
    definition = {
        "schema_version": capture_script.SCHEMA_VERSION,
        "model_name": "qwen3_14b",
        "model": model,
        "calibration": {
            "token_path": "/tokens.npy",
            "token_sha256": "d" * 64,
            "manifest_sha256": "e" * 64,
            "dataset_fingerprint": "f" * 64,
            "split": "train",
        },
        "sequences": 1,
        "run_fingerprint": "1" * 64,
    }
    handles = capture_script._open_append_handles(stage, 1)
    try:
        zeros = torch.zeros(
            capture_script.SAMPLED_PER_SEQUENCE,
            capture_script.KV_HEADS,
            capture_script.HEAD_DIM,
            dtype=torch.bfloat16,
        )
        for handle in handles:
            handle.write(zeros.view(torch.uint16).numpy().tobytes())
    finally:
        capture_script._close_handles(handles)
    first = stage / capture_script._payload_relative("k_content", 0)
    commit_existing_bf16_payload(
        first,
        (capture_script.SAMPLED_PER_SEQUENCE, 8, 128),
        provenance=capture_script._activation_provenance(definition, "k_content", 0),
    )
    manifest = capture_script._commit_stage(stage, output, definition)
    assert len(manifest["entries"]) == 2
    capture_script.verify_capture_output(output, "1" * 64)
