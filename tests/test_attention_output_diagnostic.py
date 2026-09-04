from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvcache_transfer.attention_output_diagnostic import (
    AttentionDiagnosticError,
    LayerCosineAccumulator,
    TargetTrajectoryAttentionCosineProbe,
    aggregate_layer_accumulators,
    _apply_rotary,
    causal_attention_output_fp32,
    rowwise_cosine_fp64,
)


def test_causal_attention_matches_small_manual_reference() -> None:
    query = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32
    )
    key = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]], dtype=torch.float32
    )
    value = torch.tensor(
        [[[[2.0, 0.0], [0.0, 4.0], [6.0, 6.0]]]], dtype=torch.float32
    )
    output = causal_attention_output_fp32(
        query,
        key,
        value,
        query_positions=(0, 2),
        scaling=1.0,
        query_chunk=1,
    )
    torch.testing.assert_close(output[0, 0, 0], value[0, 0, 0])
    weights = torch.softmax(torch.tensor([1.0, 1.0, 2.0]), dim=0)
    expected_last = weights @ value[0, 0]
    torch.testing.assert_close(output[0, 0, 1], expected_last)
    assert output.dtype == torch.float32


def test_causal_mask_makes_future_kv_irrelevant() -> None:
    query = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    value = torch.tensor([[[[3.0, 5.0], [7.0, 11.0]]]])
    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[:, :, 1] = 1_000_000
    changed_value[:, :, 1] = -1_000_000
    baseline = causal_attention_output_fp32(
        query, key, value, query_positions=(0,), scaling=1.0
    )
    changed = causal_attention_output_fp32(
        query, changed_key, changed_value, query_positions=(0,), scaling=1.0
    )
    torch.testing.assert_close(baseline, changed)


def test_gqa_query_heads_follow_repeat_kv_ordering() -> None:
    # With zero Q/K, the last query is a uniform average of both positions.
    query = torch.zeros(1, 4, 2, 2)
    key = torch.zeros(1, 2, 2, 2)
    value = torch.tensor(
        [[[[2.0, 4.0], [6.0, 8.0]], [[10.0, 20.0], [30.0, 40.0]]]]
    )
    output = causal_attention_output_fp32(
        query, key, value, query_positions=(1,), scaling=1.0
    )
    expected_kv0 = torch.tensor([4.0, 6.0])
    expected_kv1 = torch.tensor([20.0, 30.0])
    torch.testing.assert_close(output[0, 0, 0], expected_kv0)
    torch.testing.assert_close(output[0, 1, 0], expected_kv0)
    torch.testing.assert_close(output[0, 2, 0], expected_kv1)
    torch.testing.assert_close(output[0, 3, 0], expected_kv1)


def test_binary_key_mask_is_combined_with_causal_mask() -> None:
    query = torch.zeros(1, 1, 3, 1)
    key = torch.zeros(1, 1, 3, 1)
    value = torch.tensor([[[[2.0], [100.0], [6.0]]]])
    output = causal_attention_output_fp32(
        query,
        key,
        value,
        query_positions=(2,),
        key_valid_mask=torch.tensor([[1, 0, 1]]),
    )
    torch.testing.assert_close(output, torch.tensor([[[[4.0]]]]))


def test_rowwise_cosine_fp64_has_explicit_zero_rules() -> None:
    reference = torch.tensor([[[[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]])
    candidate = torch.tensor([[[[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]]])
    cosine = rowwise_cosine_fp64(reference, candidate)
    assert cosine.dtype == torch.float64
    torch.testing.assert_close(cosine, torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float64))


def test_layer_and_global_aggregation_are_row_equal_weighted() -> None:
    first = LayerCosineAccumulator(query_heads=2)
    first.update(
        torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]]),
        torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
    )
    second = LayerCosineAccumulator(query_heads=2)
    second.update(
        torch.tensor([[[[1.0, 0.0]], [[1.0, 0.0]]]]),
        torch.tensor([[[[-1.0, 0.0]], [[1.0, 0.0]]]]),
    )
    result = aggregate_layer_accumulators((first, second))
    # Four cosines: 1,0,-1,1.
    assert result["mean_cosine_fp64"] == pytest.approx(0.25)
    assert result["per_target_layer_mean_cosine_fp64"] == pytest.approx([0.5, 0.0])
    assert result["per_query_head_mean_cosine_fp64"] == pytest.approx([0.0, 0.5])
    assert result["row_observations"] == 4
    assert result["headline_aggregation"] == "equal_arithmetic_mean_of_target_layer_means"
    assert result["row_observations_per_layer"] == 2
    assert result["row_observations_per_layer_per_query_head"] == 1
    assert all(result["aggregation_invariants"].values())
    assert result["nan_or_inf_count"] == 0


def test_layer_result_rejects_unequal_per_head_counts() -> None:
    layer = LayerCosineAccumulator(query_heads=2)
    layer.update(
        torch.tensor([[[[1.0]], [[1.0]]]]),
        torch.tensor([[[[1.0]], [[1.0]]]]),
    )
    layer.per_head_count[1] += 1
    layer.observations += 1
    with pytest.raises(AttentionDiagnosticError, match="unequal per-head"):
        layer.result(layer_index=0)


def test_global_aggregation_rejects_unequal_layer_counts() -> None:
    first = LayerCosineAccumulator(query_heads=2)
    second = LayerCosineAccumulator(query_heads=2)
    reference = torch.tensor([[[[1.0]], [[1.0]]]])
    for layer in (first, second):
        layer.update(reference, reference)
    second.update(reference, reference)
    with pytest.raises(AttentionDiagnosticError, match="target layers have unequal"):
        aggregate_layer_accumulators((first, second))


@pytest.mark.parametrize(
    "query,key,value,positions,error",
    [
        (
            torch.zeros(1, 3, 2, 4),
            torch.zeros(1, 2, 2, 4),
            torch.zeros(1, 2, 2, 4),
            (1,),
            "divisible",
        ),
        (
            torch.zeros(1, 2, 2, 4),
            torch.zeros(1, 1, 2, 4),
            torch.zeros(1, 1, 2, 4),
            (2,),
            "must lie",
        ),
    ],
)
def test_attention_input_contract_rejects_invalid_shapes(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: tuple[int, ...],
    error: str,
) -> None:
    with pytest.raises(AttentionDiagnosticError, match=error):
        causal_attention_output_fp32(query, key, value, query_positions=positions)


def _tiny_qwen3() -> torch.nn.Module:
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    model = transformers.Qwen3ForCausalLM(config)
    model.eval()
    return model


def test_explicit_attention_matches_native_qwen3_pre_o_proj_input() -> None:
    """Lock the diagnostic vector to Transformers' native pre-o_proj meaning."""

    torch.manual_seed(5)
    model = _tiny_qwen3()
    attention = model.model.layers[0].self_attn
    captured: dict[str, torch.Tensor] = {}

    def capture_explicit(
        module: torch.nn.Module,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        position_embeddings = kwargs.get(
            "position_embeddings", args[1] if len(args) > 1 else None
        )
        assert isinstance(hidden, torch.Tensor)
        assert isinstance(position_embeddings, tuple)
        cos, sin = position_embeddings
        hidden_shape = (*hidden.shape[:-1], -1, module.head_dim)
        query = module.q_norm(module.q_proj(hidden).view(hidden_shape)).transpose(1, 2)
        key = module.k_norm(module.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
        value = module.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        query = _apply_rotary(query, cos, sin)
        key = _apply_rotary(key, cos, sin)
        captured["explicit"] = causal_attention_output_fp32(
            query,
            key,
            value,
            query_positions=tuple(range(hidden.shape[1])),
            scaling=float(module.scaling),
        )

    def capture_native(
        module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        captured["native_flat"] = args[0].detach().clone()

    attention_handle = attention.register_forward_pre_hook(capture_explicit, with_kwargs=True)
    output_handle = attention.o_proj.register_forward_pre_hook(capture_native)
    input_ids = torch.tensor([[3, 5, 7, 11, 13]], dtype=torch.long)
    try:
        with torch.inference_mode():
            model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                position_ids=torch.arange(input_ids.shape[1]).unsqueeze(0),
                cache_position=torch.arange(input_ids.shape[1]),
                use_cache=False,
                return_dict=True,
            )
    finally:
        attention_handle.remove()
        output_handle.remove()
    explicit = captured["explicit"]
    native = captured["native_flat"].view(
        input_ids.shape[0], input_ids.shape[1], attention.config.num_attention_heads, attention.head_dim
    ).transpose(1, 2)
    torch.testing.assert_close(explicit, native, rtol=1e-5, atol=5e-8)
    assert float((explicit - native).abs().max().item()) < 5e-8


def test_target_trajectory_probe_exact_cache_is_one_and_removes_hooks() -> None:
    torch.manual_seed(7)
    model = _tiny_qwen3()
    input_ids = torch.tensor([[3, 5, 7, 11, 13]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    with torch.inference_mode():
        true_output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.arange(input_ids.shape[1]),
            use_cache=True,
            return_dict=True,
        )
    from transformers.cache_utils import DynamicCache

    true_cache = true_output.past_key_values
    assert isinstance(true_cache, DynamicCache)
    hook_counts_before = [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers]
    probe = TargetTrajectoryAttentionCosineProbe(
        model,
        true_cache,
        query_positions=(0, 2, 4),
        key_valid_mask=attention_mask,
        query_chunk=2,
    )
    with torch.inference_mode(), probe:
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.arange(input_ids.shape[1]),
            use_cache=False,
            return_dict=True,
        )
    probe.validate_forward()
    result = aggregate_layer_accumulators(probe.accumulators)
    assert result["mean_cosine_fp64"] == pytest.approx(1.0, abs=1e-12)
    assert result["row_observations"] == 2 * 4 * 3
    assert [len(layer.self_attn._forward_pre_hooks) for layer in model.model.layers] == (
        hook_counts_before
    )


def test_target_trajectory_probe_detects_perturbed_values() -> None:
    torch.manual_seed(11)
    model = _tiny_qwen3()
    input_ids = torch.tensor([[2, 3, 5, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    with torch.inference_mode():
        true_output = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.arange(input_ids.shape[1]),
            use_cache=True,
            return_dict=True,
        )
    perturbed = tuple(
        (key.clone(), torch.roll(value.clone(), shifts=1, dims=-1))
        for key, value in true_output.past_key_values
    )
    probe = TargetTrajectoryAttentionCosineProbe(
        model,
        perturbed,
        query_positions=(3,),
        key_valid_mask=attention_mask,
    )
    with torch.inference_mode(), probe:
        model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.arange(input_ids.shape[1]),
            use_cache=False,
            return_dict=True,
        )
    result = aggregate_layer_accumulators(probe.accumulators)
    assert math.isfinite(result["mean_cosine_fp64"])
    assert result["mean_cosine_fp64"] < 0.999


def _load_diagnostic_cli() -> object:
    path = Path(__file__).resolve().parents[1] / "scripts/diagnose_attention_output_cosine.py"
    name = "test_diagnose_attention_output_cosine_cli"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _assert_payload_marker(payload: Path) -> dict[str, object]:
    marker = payload.with_suffix(".sha256.json")
    value = json.loads(marker.read_text(encoding="utf-8"))
    observed = hashlib.sha256(payload.read_bytes()).hexdigest()
    assert value["sha256"] == observed
    assert value["nbytes"] == payload.stat().st_size
    assert Path(str(value["path"])) == payload.resolve()
    return value


def test_cli_preflight_binds_all_identity_scopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _load_diagnostic_cli()
    project = tmp_path / "project"
    mapper_manifest = project / "mapper/manifest.json"
    calibration = project / "calibration"
    output = project / "artifacts/result.json"
    source = tmp_path / "source"
    target = tmp_path / "target"
    for path in (mapper_manifest.parent, calibration, output.parent, source, target):
        path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(cli, "ROOT", project)
    fake_bundle = SimpleNamespace(
        manifest_path=mapper_manifest.resolve(),
        manifest_sha256="mapper-sha",
        fit_fingerprint="f" * 64,
        architecture=SimpleNamespace(target_layers=64, selected_source_layers=8),
    )
    monkeypatch.setattr(cli.MapperBundle, "open", lambda _root: fake_bundle)
    monkeypatch.setattr(
        cli,
        "_load_heldout_tokens",
        lambda *_args, **_kwargs: (
            np.zeros((4, 256), dtype=np.int64),
            {"heldout_token_file_sha256": "heldout-sha"},
        ),
    )
    monkeypatch.setattr(
        cli,
        "_model_identity",
        lambda path: {
            "path": str(Path(path).resolve()),
            "weight_shard_contents_hashed": False,
            "cryptographic_weight_content_identity": False,
            "weight_identity_limitation": "same-size shard changes are not detected",
        },
    )
    monkeypatch.setattr(
        cli,
        "_software_identity",
        lambda: {
            "torch": "torch-version",
            "transformers": "transformers-version",
            "accelerate": "accelerate-version",
            "flash_attention": "fa-version",
            "torch_cuda_build": "cuda-version",
            "qwen3_modeling_implementation": {"sha256": "qwen-code-sha"},
        },
    )
    monkeypatch.setattr(
        cli,
        "_code_identity",
        lambda: {"files_digest": "diagnostic-code", "files": {}},
    )
    monkeypatch.setattr(
        cli,
        "_formal_evaluation_code_identity",
        lambda: {
            "files_digest": "formal-code",
            "files": {},
            "diagnostic_code_files_disjoint": True,
        },
    )
    args = cli.parse_args(
        [
            "--source-model",
            str(source),
            "--target-model",
            str(target),
            "--mapper-manifest",
            str(mapper_manifest),
            "--calibration-dir",
            str(calibration),
            "--output",
            str(output),
        ]
    )
    tokens, identity = cli._preflight(args)
    assert tokens.shape == (4, 256)
    assert identity["mapper"]["manifest_sha256"] == "mapper-sha"
    assert identity["software_runtime"]["qwen3_modeling_implementation"]["sha256"] == (
        "qwen-code-sha"
    )
    assert identity["source_model"]["cryptographic_weight_content_identity"] is False
    assert identity["formal_evaluation_code_unchanged_guard"][
        "diagnostic_code_files_disjoint"
    ] is True
    without_fingerprint = dict(identity)
    fingerprint = without_fingerprint.pop("diagnostic_fingerprint")
    assert fingerprint == cli._canonical_digest(without_fingerprint)


def test_real_software_and_model_identity_state_scope_precisely(tmp_path: Path) -> None:
    cli = _load_diagnostic_cli()
    software = cli._software_identity()
    for field in (
        "torch",
        "transformers",
        "accelerate",
        "flash_attention",
        "torch_cuda_build",
        "cuda_available_at_preflight",
        "cuda_visible_device_count_at_preflight",
    ):
        assert field in software
    modeling = software["qwen3_modeling_implementation"]
    modeling_path = Path(modeling["path"])
    assert modeling_path.is_file()
    assert modeling["sha256"] == hashlib.sha256(modeling_path.read_bytes()).hexdigest()

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "shard-1.safetensors").write_bytes(b"not-a-real-weight")
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "shard-1.safetensors"}}),
        encoding="utf-8",
    )
    identity = cli._model_identity(model)
    assert identity["weight_shard_contents_hashed"] is False
    assert identity["cryptographic_weight_content_identity"] is False
    assert "not detected" in identity["weight_identity_limitation"]
    assert "sha256" not in identity["weight_shards"][0]
    assert identity["content_hashed_files"]["config.json"]["sha256"] == hashlib.sha256(
        (model / "config.json").read_bytes()
    ).hexdigest()


def test_cli_run_payloads_overwrite_and_failure_are_auditable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _load_diagnostic_cli()
    project = tmp_path / "project"
    project.mkdir()
    (project / "REPRODUCTION.md").write_text("# test\n", encoding="utf-8")
    output = project / "artifacts/attention.json"
    output.parent.mkdir(parents=True)
    mapper = project / "mapper/manifest.json"
    calibration = project / "calibration"
    mapper.parent.mkdir()
    calibration.mkdir()
    monkeypatch.setattr(cli, "ROOT", project)
    run_paths: list[Path] = []

    def make_run(_root: Path, _stage: str, _name: str) -> SimpleNamespace:
        path = project / "runs" / f"run_{len(run_paths):02d}"
        path.mkdir(parents=True)
        run_paths.append(path)
        return SimpleNamespace(path=path, run_id=path.name)

    monkeypatch.setattr(cli, "make_run_directory", make_run)
    monkeypatch.setattr(cli, "append_reproduction_log", lambda *_args, **_kwargs: None)
    preflight = {
        "schema_version": 1,
        "status": "passed",
        "diagnostic_fingerprint": "d" * 64,
        "software_runtime": {"torch": "test"},
    }
    preflight_calls = 0

    def fake_preflight(_args: object) -> tuple[np.ndarray, dict[str, object]]:
        nonlocal preflight_calls
        preflight_calls += 1
        return np.zeros((1, 2), dtype=np.int64), dict(preflight)

    diagnostic_calls = 0

    def fake_diagnostic(
        _args: object, _tokens: np.ndarray, supplied_preflight: dict[str, object]
    ) -> dict[str, object]:
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        return {
            "schema_version": 1,
            "status": "complete",
            "diagnostic_fingerprint": supplied_preflight["diagnostic_fingerprint"],
            "summary": {
                "mean_cosine_fp64": 0.8 + diagnostic_calls / 100.0,
                "row_observations": 4,
                "nan_or_inf_count": 0,
            },
        }

    monkeypatch.setattr(cli, "_preflight", fake_preflight)
    monkeypatch.setattr(cli, "_run_diagnostic", fake_diagnostic)
    common = [
        "--mapper-manifest",
        str(mapper),
        "--calibration-dir",
        str(calibration),
        "--output",
        str(output),
    ]

    # Preflight-only writes a complete identity payload+marker and never an aggregate.
    assert cli.main([*common, "--preflight-only"]) == 0
    preflight_run = run_paths[-1]
    assert json.loads((preflight_run / "result.json").read_text())["status"] == (
        "preflight-complete"
    )
    _assert_payload_marker(preflight_run / "preflight.json")
    assert not output.exists()
    assert diagnostic_calls == 0

    # First success persists a byte-identical run-local payload and aggregate.
    assert cli.main(common) == 0
    first_run = run_paths[-1]
    first_payload = first_run / "attention_output_cosine.json"
    first_bytes = first_payload.read_bytes()
    assert first_bytes == output.read_bytes()
    first_marker = _assert_payload_marker(first_payload)
    _assert_payload_marker(output)
    first_result = json.loads((first_run / "result.json").read_text())
    assert first_result["immutable_run_payload_sha256"] == first_marker["sha256"]
    assert first_result["aggregate_overwrite_requested"] is False

    # Existing aggregate without --overwrite fails before expensive preflight and is unchanged.
    aggregate_before_refusal = output.read_bytes()
    preflight_before_refusal = preflight_calls
    with pytest.raises(FileExistsError, match="--overwrite"):
        cli.main(common)
    refused_run = run_paths[-1]
    assert preflight_calls == preflight_before_refusal
    assert output.read_bytes() == aggregate_before_refusal
    refused_result = json.loads((refused_run / "result.json").read_text())
    assert refused_result["status"] == "failed"
    _assert_payload_marker(refused_run / "failure.json")

    # Explicit overwrite creates a new full run payload; the older payload is immutable.
    assert cli.main([*common, "--overwrite"]) == 0
    overwrite_run = run_paths[-1]
    overwrite_payload = overwrite_run / "attention_output_cosine.json"
    assert output.read_bytes() == overwrite_payload.read_bytes()
    assert output.read_bytes() != first_bytes
    assert first_payload.read_bytes() == first_bytes
    _assert_payload_marker(overwrite_payload)
    overwrite_result = json.loads((overwrite_run / "result.json").read_text())
    assert overwrite_result["aggregate_overwrite_requested"] is True

    # A compute failure has its own immutable traceback payload and marker.
    failure_output = project / "artifacts/failure_target.json"
    monkeypatch.setattr(
        cli,
        "_run_diagnostic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic failure")),
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        cli.main(
            [
                "--mapper-manifest",
                str(mapper),
                "--calibration-dir",
                str(calibration),
                "--output",
                str(failure_output),
            ]
        )
    failed_run = run_paths[-1]
    failure = json.loads((failed_run / "failure.json").read_text())
    assert failure["status"] == "failed"
    assert failure["error"] == "synthetic failure"
    _assert_payload_marker(failed_run / "failure.json")
    assert not failure_output.exists()
