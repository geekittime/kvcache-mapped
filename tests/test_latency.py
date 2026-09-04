from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from kvcache_transfer.latency import (
    COMPONENT_NAMES,
    ResidentMapperBundle,
    benchmark_boundary_decode,
    benchmark_callable,
    benchmark_mapper_end_to_end,
    benchmark_target_reprefill,
    configure_strict_latency_math,
    measure_synchronized,
    percentile,
    profile_transfer_once,
    summarize_latency,
)
from kvcache_transfer.mapper import LayerMapper
from kvcache_transfer.transfer import source_cache_to_content, transfer_source_cache


def tiny_qwen3(device: torch.device = torch.device("cpu")) -> Qwen3ForCausalLM:
    torch.manual_seed(20260813)
    config = Qwen3Config(
        vocab_size=101,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rope_theta=1_000_000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=None,
        _attn_implementation="eager",
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def identity_bundle(model: Qwen3ForCausalLM) -> ResidentMapperBundle:
    from kvcache_transfer.config import ArchitectureSpec

    config = model.config
    architecture = ArchitectureSpec(
        source_layers=config.num_hidden_layers,
        target_layers=config.num_hidden_layers,
        source_kv_heads=config.num_key_value_heads,
        target_kv_heads=config.num_key_value_heads,
        source_head_dim=config.head_dim,
        target_head_dim=config.head_dim,
        selected_source_layers=1,
    )
    device = next(model.parameters()).device
    features = architecture.target_features
    identity = torch.eye(features, dtype=torch.float32, device=device)
    return ResidentMapperBundle(
        architecture=architecture,
        layers=tuple(
            LayerMapper(
                source_layers=(layer,),
                key_weight=identity.clone(),
                key_bias=torch.zeros(features, device=device),
                value_weight=identity.clone(),
                value_bias=torch.zeros(features, device=device),
            )
            for layer in range(architecture.target_layers)
        ),
    )


def source_cache(model: Qwen3ForCausalLM, tokens: torch.Tensor):
    device = next(model.parameters()).device
    with torch.inference_mode():
        return model.model(
            input_ids=tokens.to(device),
            use_cache=True,
            return_dict=True,
        ).past_key_values


def test_percentile_and_summary_are_explicit_and_deterministic() -> None:
    samples = [1.0, 2.0, 3.0, 8.0]
    assert percentile(samples, 0.0) == 1.0
    assert percentile(samples, 0.5) == 2.5
    assert percentile(samples, 0.9) == pytest.approx(6.5)
    result = summarize_latency(samples)
    assert result["unit"] == "ms"
    assert result["count"] == 4
    assert result["mean"] == 3.5
    assert result["median"] == result["p50"] == 2.5
    assert result["p10"] == pytest.approx(1.3)
    assert result["p90"] == pytest.approx(6.5)
    assert result["std_population"] == pytest.approx(2.692582403567252)
    assert result["samples"] == samples
    with pytest.raises(ValueError, match="at least one"):
        summarize_latency([])
    with pytest.raises(ValueError, match="finite"):
        summarize_latency([float("nan")])


def test_benchmark_callable_excludes_prepare_and_honors_warmup_counts() -> None:
    calls = {"prepare": 0, "operation": 0, "release": 0}

    def prepare() -> int:
        calls["prepare"] += 1
        return calls["prepare"]

    def operation(value: int) -> int:
        calls["operation"] += 1
        return value + 1

    def release(output: int, state: int) -> None:
        assert output == state + 1
        calls["release"] += 1

    result = benchmark_callable(
        operation,
        prepare=prepare,
        warmup=2,
        repeats=3,
        release=release,
    )
    assert calls == {"prepare": 5, "operation": 5, "release": 5}
    assert result["warmup"] == 2
    assert result["timed_repeats"] == result["count"] == 3
    assert result["timing_backends"] == ["host_perf_counter"]
    assert result["cuda_peak_memory_collected"] is True
    assert result["cuda_peak_allocated_bytes"] == {}
    assert result["cuda_peak_reserved_bytes"] == {}


def test_benchmark_callable_can_preserve_a_caller_owned_outer_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kvcache_transfer.latency as latency_module

    cuda_device = torch.device("cuda:0")
    monkeypatch.setattr(
        latency_module,
        "_as_unique_devices",
        lambda _devices: (cuda_device,),
    )
    monkeypatch.setattr(latency_module, "synchronize_devices", lambda _devices: None)
    monkeypatch.setattr(
        latency_module,
        "measure_synchronized",
        lambda operation, **_kwargs: latency_module.TimedValue(
            value=operation(), milliseconds=1.0, timing_backend="mock"
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("nested CUDA peak counter access")

    monkeypatch.setattr(latency_module.torch.cuda, "reset_peak_memory_stats", forbidden)
    monkeypatch.setattr(latency_module.torch.cuda, "max_memory_allocated", forbidden)
    monkeypatch.setattr(latency_module.torch.cuda, "max_memory_reserved", forbidden)
    result = latency_module.benchmark_callable(
        lambda value: value + 1,
        prepare=lambda: 1,
        warmup=0,
        repeats=1,
        devices=(cuda_device,),
        collect_cuda_peak_memory=False,
    )
    assert result["cuda_peak_memory_collected"] is False
    assert result["cuda_peak_memory_disabled_reason"] == (
        "preserve_caller_owned_outer_cuda_peak_counters"
    )
    assert "cuda_peak_allocated_bytes" not in result
    assert "cuda_peak_reserved_bytes" not in result


def test_mapper_component_profile_can_preserve_a_caller_owned_outer_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kvcache_transfer.latency as latency_module

    cpu_tensor = torch.zeros(1)
    cuda_device = torch.device("cuda:0")
    monkeypatch.setattr(
        latency_module,
        "cache_to_legacy",
        lambda _cache: ((cpu_tensor, cpu_tensor),),
    )
    monkeypatch.setattr(
        latency_module,
        "qwen3_layer_devices",
        lambda _model: (cuda_device,),
    )
    monkeypatch.setattr(
        latency_module,
        "profile_transfer_once",
        lambda *_args, **_kwargs: (
            object(),
            {name: 1.0 for name in COMPONENT_NAMES},
            {name: ["mock"] for name in COMPONENT_NAMES},
        ),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("nested CUDA peak counter access")

    monkeypatch.setattr(latency_module.torch.cuda, "reset_peak_memory_stats", forbidden)
    monkeypatch.setattr(latency_module.torch.cuda, "max_memory_allocated", forbidden)
    monkeypatch.setattr(latency_module.torch.cuda, "max_memory_reserved", forbidden)
    result = latency_module.benchmark_mapper_components(
        object(),
        source_config=object(),
        target_model=object(),
        mapper_bundle=object(),  # type: ignore[arg-type]
        warmup=0,
        repeats=1,
        collect_cuda_peak_memory=False,
    )
    assert result["cuda_peak_memory_collected"] is False
    assert result["cuda_peak_memory_disabled_reason"] == (
        "preserve_caller_owned_outer_cuda_peak_counters"
    )
    assert "cuda_peak_allocated_bytes" not in result
    assert "cuda_peak_reserved_bytes" not in result


def test_strict_mapper_math_disables_tf32() -> None:
    original_matmul = torch.backends.cuda.matmul.allow_tf32
    original_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        result = configure_strict_latency_math()
        assert result["mapper_dtype"] == "float32"
        assert result["mapper_execution"] == "eager"
        assert result["allow_tf32_cuda_matmul"] is False
        assert result["allow_tf32_cudnn"] is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_matmul
        torch.backends.cudnn.allow_tf32 = original_cudnn


def test_target_reprefill_uses_transformer_body_and_excludes_lm_head() -> None:
    model = tiny_qwen3()

    class ForbiddenHead(torch.nn.Module):
        def forward(self, *_args, **_kwargs):
            raise AssertionError("LM head was called by target re-prefill")

    model.lm_head = ForbiddenHead()
    ids = torch.tensor([[2, 3, 5, 7]])
    result = benchmark_target_reprefill(model, ids, warmup=1, repeats=2)
    assert result["measurement_scope"] == "target_transformer_reprefill"
    assert result["lm_head_included"] is False
    assert result["kv_cache_output"] is True
    assert result["input_length"] == 4
    assert result["count"] == 2
    assert result["inference_mode_enabled"] is True
    assert result["autograd_graph_disabled"] is True


def test_target_reprefill_forward_runs_in_inference_mode_without_grad_graph() -> None:
    calls = 0

    class AssertingBody(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(32, 8)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8)])

        def forward(self, *, input_ids, use_cache, return_dict):
            nonlocal calls
            assert torch.is_inference_mode_enabled() is True
            assert torch.is_grad_enabled() is False
            assert use_cache is True
            assert return_dict is True
            output = self.layers[0](self.embed_tokens(input_ids))
            assert output.requires_grad is False
            calls += 1
            return output

    class Target(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = AssertingBody()

    result = benchmark_target_reprefill(
        Target(),
        torch.tensor([[2, 3, 5, 7]]),
        warmup=1,
        repeats=2,
    )
    assert calls == 3
    assert result["inference_mode_enabled"] is True
    assert result["autograd_graph_disabled"] is True


def test_mapper_timed_transfer_runs_in_inference_mode_without_grad_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kvcache_transfer.latency as latency_module

    source = torch.zeros(1)
    parameter = torch.nn.Parameter(torch.ones(1))
    calls = 0

    monkeypatch.setattr(
        latency_module,
        "cache_to_legacy",
        lambda _cache: ((source, source),),
    )
    monkeypatch.setattr(
        latency_module,
        "qwen3_layer_devices",
        lambda _model: (torch.device("cpu"),),
    )

    def asserting_transfer(*_args, **_kwargs):
        nonlocal calls
        assert torch.is_inference_mode_enabled() is True
        assert torch.is_grad_enabled() is False
        output = parameter * 2
        assert output.requires_grad is False
        calls += 1
        return output

    monkeypatch.setattr(latency_module, "transfer_source_cache", asserting_transfer)
    result = latency_module.benchmark_mapper_end_to_end(
        object(),
        source_config=object(),
        target_model=object(),
        mapper_bundle=object(),  # type: ignore[arg-type]
        warmup=1,
        repeats=2,
    )
    assert calls == 3
    assert result["inference_mode_enabled"] is True
    assert result["autograd_graph_disabled"] is True


def test_profiled_identity_transfer_matches_runtime_and_has_all_components() -> None:
    model = tiny_qwen3()
    bundle = identity_bundle(model)
    prefix = torch.tensor([[7, 5, 19, 2, 31, 11]])
    cache = source_cache(model, prefix)
    expected = transfer_source_cache(
        cache,
        source_config=model.config,
        target_model=model,
        mapper_bundle=bundle,  # type: ignore[arg-type]
        token_chunk=2,
    )
    observed, timings, backends = profile_transfer_once(
        cache,
        source_config=model.config,
        target_model=model,
        mapper_bundle=bundle,
        token_chunk=2,
    )
    assert tuple(timings) == COMPONENT_NAMES
    assert tuple(backends) == COMPONENT_NAMES
    assert all(value >= 0.0 for value in timings.values())
    assert all(values == ["host_perf_counter"] for values in backends.values())
    for layer in range(model.config.num_hidden_layers):
        torch.testing.assert_close(
            observed.key_cache[layer], expected.key_cache[layer], atol=3e-6, rtol=3e-6
        )
        torch.testing.assert_close(
            observed.value_cache[layer], expected.value_cache[layer]
        )


def test_mapper_and_boundary_metrics_keep_source_prefill_and_cache_clone_outside() -> (
    None
):
    model = tiny_qwen3()
    bundle = identity_bundle(model)
    prefix = torch.tensor([[7, 5, 19, 2, 31, 11]])
    cache = source_cache(model, prefix)
    mapper = benchmark_mapper_end_to_end(
        cache,
        source_config=model.config,
        target_model=model,
        mapper_bundle=bundle,
        token_chunk=2,
        warmup=1,
        repeats=2,
    )
    assert mapper["source_prefill_included"] is False
    assert mapper["includes_source_cache_device_handoff"] is True
    assert mapper["count"] == 2

    mapped = transfer_source_cache(
        cache,
        source_config=model.config,
        target_model=model,
        mapper_bundle=bundle,  # type: ignore[arg-type]
        token_chunk=2,
    )
    original_length = mapped.get_seq_length()
    boundary = benchmark_boundary_decode(
        model,
        torch.tensor([[3]]),
        mapped,
        warmup=1,
        repeats=2,
        include_lm_head=False,
    )
    assert boundary["query_tokens"] == 1
    assert boundary["lm_head_included"] is False
    assert boundary["cache_clone_included"] is False
    assert boundary["prefix_length"] == original_length
    assert mapped.get_seq_length() == original_length


def test_boundary_uses_a_shallow_read_only_prefix_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kvcache_transfer.latency as latency_module

    model = tiny_qwen3()
    bundle = identity_bundle(model)
    cache = source_cache(model, torch.tensor([[7, 5, 19, 2]]))
    mapped = transfer_source_cache(
        cache,
        source_config=model.config,
        target_model=model,
        mapper_bundle=bundle,  # type: ignore[arg-type]
    )
    original = latency_module.build_dynamic_cache
    clone_arguments: list[bool] = []

    def wrapped(*args, **kwargs):
        clone_arguments.append(bool(kwargs.get("clone", True)))
        return original(*args, **kwargs)

    monkeypatch.setattr(latency_module, "build_dynamic_cache", wrapped)
    before = tuple((key.data_ptr(), value.data_ptr()) for key, value in mapped)
    result = benchmark_boundary_decode(
        model,
        torch.tensor([[3]]),
        mapped,
        warmup=0,
        repeats=1,
    )
    after = tuple((key.data_ptr(), value.data_ptr()) for key, value in mapped)
    assert clone_arguments == [False]
    assert before == after
    assert result["prefix_tensors_shared_read_only"] is True
    assert result["decode_cache_concatenation_included"] is True


def test_source_content_strict_validation_is_safe_by_default_and_explicitly_skippable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kvcache_transfer.transfer as transfer_module

    model = tiny_qwen3()
    cache = source_cache(model, torch.tensor([[7, 5, 19, 2]]))
    original = transfer_module.remove_rope
    observed_validate: list[bool] = []

    def wrapped(*args, **kwargs):
        observed_validate.append(bool(kwargs.get("validate", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(transfer_module, "remove_rope", wrapped)
    source_cache_to_content(cache, model.config)
    assert observed_validate == [True] * model.config.num_hidden_layers
    observed_validate.clear()
    source_cache_to_content(cache, model.config, strict_validation=False)
    assert observed_validate == [False] * model.config.num_hidden_layers


def test_latency_cli_defaults_are_paper_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(sys, "argv", ["benchmark_latency.py"])
    args = cli.parse_args()
    assert args.mode == "production"
    assert args.lengths == cli.PAPER_LENGTHS
    assert args.source_devices == (4, 5)
    assert args.target_devices == (0, 1, 2, 3)
    assert args.source_layer_counts is None
    assert args.target_layer_counts is None
    assert args.warmup == 50
    assert args.repeats == 30
    assert args.mapper_token_chunk == 1024
    assert args.attention == "flash_attention_2"


def test_backend_identity_binds_python_cuda_driver_and_runtime_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(cli.torch.cuda.nccl, "version", lambda: (2, 25, 1))
    monkeypatch.setattr(cli.torch.backends.cudnn, "version", lambda: 90501)
    monkeypatch.setattr(
        cli,
        "_command_output",
        lambda command: {
            "command": list(command),
            "returncode": 0,
            "stdout": "570.00\n" * 8,
            "stderr": "",
        },
    )
    for name in (
        "PYTORCH_CUDA_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "NVIDIA_TF32_OVERRIDE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "")
    first = cli.backend_identity("flash_attention_2")
    assert first["python_version"] == sys.version
    assert first["python_executable"] == str(Path(sys.executable).resolve())
    assert first["cudnn_version"] == 90501
    assert first["nccl_version"] == [2, 25, 1]
    assert first["nvidia_smi_driver_version"]["stdout"] == "570.00\n" * 8
    assert first["runtime_environment"] == {
        "PYTORCH_CUDA_ALLOC_CONF": None,
        "PYTORCH_ALLOC_CONF": None,
        "CUBLAS_WORKSPACE_CONFIG": "",
        "NVIDIA_TF32_OVERRIDE": None,
    }
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    second = cli.backend_identity("flash_attention_2")
    assert second["runtime_environment"] != first["runtime_environment"]
    assert cli._canonical_digest(second) != cli._canonical_digest(first)


def test_latency_cli_accepts_explicit_cuda_zero_but_lengths_remain_positive() -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(
        ["--source-devices", "4,5", "--target-devices", "0,1,2,3", "--lengths", "64"]
    )
    assert args.source_devices == (4, 5)
    assert args.target_devices == (0, 1, 2, 3)
    assert args.lengths == (64,)

    with pytest.raises(SystemExit):
        cli.parse_args(["--target-devices", "-1,0"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--lengths", "0,64"])


def test_latency_cli_accepts_and_fingerprints_weighted_layer_sharding() -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(
        [
            "--source-devices",
            "5,6,7",
            "--target-devices",
            "0,1,2,3,4",
            "--source-layer-counts",
            "14,13,13",
            "--target-layer-counts",
            "8,15,14,14,13",
        ]
    )
    assert args.source_layer_counts == (14, 13, 13)
    assert args.target_layer_counts == (8, 15, 14, 14, 13)
    assert cli._cli_identity(args)["source_layer_counts"] == [14, 13, 13]
    assert cli._cli_identity(args)["target_layer_counts"] == [8, 15, 14, 14, 13]
    assert cli.production_layer_sharding_contract(args) == {
        "source": {
            "mode": "explicit_exact_counts",
            "devices": [5, 6, 7],
            "declared_layer_counts": [14, 13, 13],
            "resolved_layer_counts": [14, 13, 13],
            "total_layers": 40,
        },
        "target": {
            "mode": "explicit_exact_counts",
            "devices": [0, 1, 2, 3, 4],
            "declared_layer_counts": [8, 15, 14, 14, 13],
            "resolved_layer_counts": [8, 15, 14, 14, 13],
            "total_layers": 64,
        },
    }

    with pytest.raises(SystemExit):
        cli.parse_args(["--target-layer-counts", "8,,15,14,14,13"])
    with pytest.raises(SystemExit):
        cli.parse_args(["--source-layer-counts", "14,13,0"])


def test_latency_weighted_layer_sharding_validation_rejects_wrong_shape_and_sum() -> (
    None
):
    import scripts.benchmark_latency as cli

    wrong_shape = cli.parse_args(
        ["--source-devices", "5,6,7", "--source-layer-counts", "20,20"]
    )
    with pytest.raises(ValueError, match="length"):
        cli.validate_args(wrong_shape)

    wrong_sum = cli.parse_args(
        [
            "--target-devices",
            "0,1,2,3,4",
            "--target-layer-counts",
            "8,14,14,14,13",
        ]
    )
    with pytest.raises(ValueError, match="sum exactly"):
        cli.validate_args(wrong_sum)


def test_run_definition_binds_declared_and_resolved_weighted_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    gib = 1 << 30
    monkeypatch.setattr(
        cli,
        "model_artifact_identity",
        lambda path: {
            "path": str(path),
            "weight_shard_total_nbytes": 28 * gib if "14B" in str(path) else 62 * gib,
        },
    )
    monkeypatch.setattr(
        cli, "mapper_artifact_identity", lambda path: {"root": str(path)}
    )
    monkeypatch.setattr(
        cli,
        "project_code_provenance",
        lambda: {"files": {}, "files_digest": "code-digest"},
    )
    monkeypatch.setattr(
        cli, "backend_identity", lambda attention: {"requested": attention}
    )
    hardware = {
        "devices": [{"total_memory_bytes": 24 * gib}] * 8,
        "selected_devices": [5, 6, 7, 0, 1, 2, 3, 4],
    }

    def definition(target_counts: str):
        args = cli.parse_args(
            [
                "--source-devices",
                "5,6,7",
                "--target-devices",
                "0,1,2,3,4",
                "--source-layer-counts",
                "14,13,13",
                "--target-layer-counts",
                target_counts,
            ]
        )
        return cli.build_run_definition(
            args,
            lengths=(32768,),
            profile_warmup=50,
            profile_repeats=30,
            hardware=hardware,
        )

    observed = definition("8,15,14,14,13")
    assert observed["cli"]["source_layer_counts"] == [14, 13, 13]
    assert observed["cli"]["target_layer_counts"] == [8, 15, 14, 14, 13]
    assert observed["resolved"]["layer_sharding"]["source"][
        "resolved_layer_counts"
    ] == [14, 13, 13]
    assert observed["resolved"]["layer_sharding"]["target"][
        "resolved_layer_counts"
    ] == [8, 15, 14, 14, 13]
    assert observed["analytical_memory"]["layer_sharding"]["target"][
        "resolved_layer_counts"
    ] == [8, 15, 14, 14, 13]
    alternative = definition("9,14,14,14,13")
    assert cli._canonical_digest(observed) != cli._canonical_digest(alternative)


def test_tiny_latency_rejects_unused_explicit_layer_counts() -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(["--mode", "tiny", "--source-layer-counts", "20,20"])
    with pytest.raises(ValueError, match="only in production"):
        cli.validate_args(args)


def test_production_cli_rejects_a_non_fa2_backend() -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(["--attention", "eager"])
    with pytest.raises(ValueError, match="flash_attention_2"):
        cli.validate_args(args)


def test_eager_tiny_attention_dispatch_contract_is_observable() -> None:
    import scripts.benchmark_latency as cli

    model = tiny_qwen3()
    contract = cli.assert_attention_backend(
        model,
        role="tiny",
        expected="eager",
        maximum_requested_length=16,
    )
    assert contract["status"] == "passed"
    assert contract["config_attn_implementation"] == "eager"
    assert contract["dispatcher_verified_for_every_layer"] is True


def test_production_memory_contract_has_exact_32k_cache_arithmetic() -> None:
    import scripts.benchmark_latency as cli

    gib = 1 << 30
    value = cli.analytical_memory_contract(
        maximum_length=32768,
        token_chunk=1024,
        artifacts={
            "source_model": {"weight_shard_total_nbytes": 28 * gib},
            "target_model": {"weight_shard_total_nbytes": 62 * gib},
        },
        hardware={"devices": [{"total_memory_bytes": 24 * gib}] * 6},
        source_device_count=2,
        target_device_count=4,
        supplemental_diagnostics=False,
    )
    assert value["source_cache_bf16_bytes"] == 5 * gib
    assert value["mapped_target_cache_bf16_bytes"] == 8 * gib
    assert value["source_inverse_content_keys_fp32_bytes"] == 5 * gib
    assert value["resident_mapper_fp32_bytes"] == 4 * gib + 512 * 1024
    assert value["boundary_prefix_deep_copy_bytes"] == 0
    assert value["paper_main_cell_only"] is True
    assert value["supplemental_diagnostics_enabled"] is False
    assert value["boundary_parent_plus_decode_output_cache_bytes"] == 0
    assert value["target_cache_copies_in_analytic_cell_bound"] == 1
    assert (
        value["formal_acceptance_requires_per_length_cuda_peak_memory_evidence"] is True
    )

    supplemental = cli.analytical_memory_contract(
        maximum_length=32768,
        token_chunk=1024,
        artifacts={
            "source_model": {"weight_shard_total_nbytes": 28 * gib},
            "target_model": {"weight_shard_total_nbytes": 62 * gib},
        },
        hardware={"devices": [{"total_memory_bytes": 24 * gib}] * 6},
        source_device_count=2,
        target_device_count=4,
        supplemental_diagnostics=True,
    )
    assert supplemental["paper_main_cell_only"] is False
    assert supplemental["boundary_parent_plus_decode_output_cache_bytes"] == 16 * gib
    assert supplemental["target_cache_copies_in_analytic_cell_bound"] == 2


def test_production_memory_contract_records_weighted_layer_fraction_estimates() -> None:
    import scripts.benchmark_latency as cli

    gib = 1 << 30
    value = cli.analytical_memory_contract(
        maximum_length=32768,
        token_chunk=1024,
        artifacts={
            "source_model": {"weight_shard_total_nbytes": 28 * gib},
            "target_model": {"weight_shard_total_nbytes": 62 * gib},
        },
        hardware={"devices": [{"total_memory_bytes": 24 * gib}] * 8},
        source_device_count=3,
        target_device_count=5,
        supplemental_diagnostics=False,
        source_layer_counts=(14, 13, 13),
        target_layer_counts=(8, 15, 14, 14, 13),
    )
    assert value["layer_sharding"] == {
        "source": {
            "mode": "explicit_exact_counts",
            "resolved_layer_counts": [14, 13, 13],
            "total_layers": 40,
        },
        "target": {
            "mode": "explicit_exact_counts",
            "resolved_layer_counts": [8, 15, 14, 14, 13],
            "total_layers": 64,
        },
    }
    source_estimates = value[
        "approx_source_per_device_lower_bound_bytes_by_layer_fraction"
    ]
    target_estimates = value[
        "approx_target_cell_per_device_lower_bound_bytes_by_layer_fraction"
    ]
    assert len(source_estimates) == 3
    assert len(target_estimates) == 5
    assert source_estimates[0] > source_estimates[1] == source_estimates[2]
    assert target_estimates[0] < target_estimates[4]
    assert target_estimates[4] < target_estimates[2] == target_estimates[3]
    assert target_estimates[2] < target_estimates[1]
    assert value["approx_source_per_device_lower_bound_bytes"] == max(source_estimates)
    assert value["approx_target_cell_per_device_lower_bound_bytes"] == max(
        target_estimates
    )


def test_staged_v9_topology_and_exact_cpu_cache_totals_are_fingerprinted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3,4,5,6,7,0,1")
    args = cli.parse_args(
        [
            "--stage-source-caches-on-cpu-before-cells",
            "--source-prefill-devices",
            "5,6,7",
            "--source-prefill-layer-counts",
            "14,13,13",
            "--source-devices",
            "6,7",
            "--source-layer-counts",
            "20,20",
            "--target-devices",
            "0,1,2,3,4,5",
            "--target-layer-counts",
            "10,11,11,11,11,10",
        ]
    )
    topology = cli.staged_runtime_topology(args)
    assert topology["source_model_prefill"] == {
        "logical_devices": [5, 6, 7],
        "physical_devices": [7, 0, 1],
        "layer_counts": [14, 13, 13],
    }
    assert topology["source_cache_staging"] == {
        "logical_devices": [6, 7],
        "physical_devices": [0, 1],
        "layer_counts": [20, 20],
    }
    assert topology["target_timed"] == {
        "logical_devices": [0, 1, 2, 3, 4, 5],
        "physical_devices": [2, 3, 4, 5, 6, 7],
        "layer_counts": [10, 11, 11, 11, 11, 10],
    }
    assert topology["source_prefill_target_logical_overlap"] == [5]
    assert topology["source_prefill_target_physical_overlap"] == [7]
    assert sum(cli.source_cache_exact_bytes(x) for x in cli.PAPER_LENGTHS) == (
        10_726_932_480
    )
    assert sum(cli.source_cache_exact_bytes(x) for x in (64, 32768)) == (
        5_379_194_880
    )


def test_staged_v9_analytic_timed_source_bound_excludes_source_weights() -> None:
    import scripts.benchmark_latency as cli

    gib = 1 << 30
    value = cli.analytical_memory_contract(
        maximum_length=32768,
        token_chunk=1024,
        artifacts={
            "source_model": {"weight_shard_total_nbytes": 28 * gib},
            "target_model": {"weight_shard_total_nbytes": 62 * gib},
        },
        hardware={"devices": [{"total_memory_bytes": 24 * gib}] * 8},
        source_device_count=2,
        target_device_count=6,
        supplemental_diagnostics=False,
        source_layer_counts=(20, 20),
        target_layer_counts=(10, 11, 11, 11, 11, 10),
        staged_source_lifecycle=True,
        source_prefill_device_count=3,
        source_prefill_layer_counts=(14, 13, 13),
        staged_prefix_lengths=cli.PAPER_LENGTHS,
    )
    assert value["all_prefix_lengths_cpu_staging_exact_bytes"] == 10_726_932_480
    assert value["source_model_live_during_timed_cells"] is False
    assert value["timed_mapper_source_includes_source_model_weights"] is False
    assert value["source_lower_bound_phase"] == (
        "timed_mapper_source_cache_plus_inverse_without_source_model"
    )
    # 32K raw cache (5 GiB) + inverse keys (5 GiB), split 20/20.
    assert value["approx_source_per_device_lower_bound_bytes_by_layer_fraction"] == [
        5 * gib,
        5 * gib,
    ]
    assert value["layer_sharding"]["source_model_prefill"][
        "resolved_layer_counts"
    ] == [14, 13, 13]


def test_staged_peak_scope_is_distinct_and_run_one_length_has_no_source_model() -> (
    None
):
    import inspect
    import scripts.benchmark_latency as cli

    fingerprint = "e" * 64
    state = cli.begin_cuda_peak_memory_measurement(
        (),
        prefix_length=64,
        request_fingerprint=fingerprint,
        measurement_scope=cli.STAGED_CUDA_PEAK_MEASUREMENT_SCOPE,
    )
    evidence = cli.finish_cuda_peak_memory_measurement(state)
    cli.verify_cuda_peak_memory_evidence(
        evidence,
        expected_device_ids=(),
        expected_prefix_length=64,
        expected_request_fingerprint=fingerprint,
        expected_measurement_scope=cli.STAGED_CUDA_PEAK_MEASUREMENT_SCOPE,
    )
    parameters = inspect.signature(cli.run_one_length).parameters
    assert "source_model" not in parameters
    assert {"source_cache", "source_config", "prefix_ids", "boundary_ids"} <= set(
        parameters
    )


def test_source_release_clears_cublas_workspace_in_fail_closed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    events: list[str] = []

    class DeviceContext:
        def __init__(self, device: int):
            self.device = device

        def __enter__(self):
            events.append(f"enter:{self.device}")

        def __exit__(self, *_args):
            events.append(f"exit:{self.device}")

    monkeypatch.setattr(
        cli, "synchronize_devices", lambda devices: events.append(f"sync:{tuple(devices)}")
    )
    monkeypatch.setattr(
        cli.torch._C,
        "_cuda_clearCublasWorkspaces",
        lambda: events.append("clear_cublas"),
    )
    monkeypatch.setattr(cli.gc, "collect", lambda: events.append("gc"))
    monkeypatch.setattr(cli.torch.cuda, "device", DeviceContext)
    monkeypatch.setattr(
        cli.torch.cuda, "empty_cache", lambda: events.append("empty_cache")
    )
    evidence = cli.clear_cuda_workspaces_and_allocator_cache((5, 6, 7))
    assert evidence["call_order"] == [
        "synchronize_before_cublas_workspace_clear",
        "torch._C._cuda_clearCublasWorkspaces",
        "gc_collect_after_cublas_workspace_clear",
        "torch.cuda.empty_cache_each_selected_device",
        "synchronize_after_empty_cache",
    ]
    assert events == [
        "sync:(5, 6, 7)",
        "clear_cublas",
        "gc",
        "enter:5",
        "empty_cache",
        "exit:5",
        "enter:6",
        "empty_cache",
        "exit:6",
        "enter:7",
        "empty_cache",
        "exit:7",
        "sync:(5, 6, 7)",
    ]
    monkeypatch.setattr(cli.torch._C, "_cuda_clearCublasWorkspaces", None)
    with pytest.raises(RuntimeError, match="lacks _cuda_clearCublasWorkspaces"):
        cli.clear_cuda_workspaces_and_allocator_cache((5, 6, 7))


def test_staged_peak_baseline_and_release_are_cross_bound() -> None:
    import scripts.benchmark_latency as cli

    cache_bytes = cli.source_cache_exact_bytes(64)
    topology = {
        "source_cache_staging": {
            "logical_devices": [6, 7],
            "layer_counts": [20, 20],
        }
    }
    lifecycle = {
        "cache_total_bytes": cache_bytes,
        "restaged_cache_release": {
            "allocator_driver_after_release": {
                "6": {"allocated_bytes": 0, "reserved_bytes": 0},
                "7": {"allocated_bytes": 0, "reserved_bytes": 0},
            }
        },
    }
    half = cache_bytes // 2
    peak = {
        "devices": {
            "6": {
                "allocated_bytes_before": half,
                "reserved_bytes_before": half,
                "allocated_bytes_after": 0,
                "reserved_bytes_after": 0,
            },
            "7": {
                "allocated_bytes_before": half,
                "reserved_bytes_before": half,
                "allocated_bytes_after": 0,
                "reserved_bytes_after": 0,
            },
        }
    }
    cli.verify_staged_peak_lifecycle_binding(
        lifecycle=lifecycle,
        peak_evidence=peak,
        expected_runtime_topology=topology,
    )
    resigned_bad_baseline = json.loads(json.dumps(peak))
    resigned_bad_baseline["devices"]["6"]["allocated_bytes_before"] += 512
    with pytest.raises(ValueError, match="does not equal its raw cache"):
        cli.verify_staged_peak_lifecycle_binding(
            lifecycle=lifecycle,
            peak_evidence=resigned_bad_baseline,
            expected_runtime_topology=topology,
        )
    resigned_bad_after = json.loads(json.dumps(peak))
    resigned_bad_after["devices"]["7"]["reserved_bytes_after"] = 512
    with pytest.raises(ValueError, match="reserved_bytes_after is nonzero"):
        cli.verify_staged_peak_lifecycle_binding(
            lifecycle=lifecycle,
            peak_evidence=resigned_bad_after,
            expected_runtime_topology=topology,
        )


def test_latency_model_identity_hashes_every_weight_shard_and_detects_same_size_change(
    tmp_path: Path,
) -> None:
    import scripts.benchmark_latency as cli

    model = tmp_path / "model"
    model.mkdir()
    for name in ("config.json", "tokenizer.json"):
        (model / name).write_text("{}", encoding="utf-8")
    index = {
        "weight_map": {
            "layer.0.weight": "model-00001-of-00002.safetensors",
            "layer.1.weight": "model-00002-of-00002.safetensors",
        }
    }
    (model / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    first_shard = model / "model-00001-of-00002.safetensors"
    second_shard = model / "model-00002-of-00002.safetensors"
    first_shard.write_bytes(b"first-shard")
    second_shard.write_bytes(b"second-shard")

    first = cli.model_artifact_identity(model)
    assert first["weight_shard_contents_hashed"] is True
    assert first["weight_shard_count"] == 2
    assert all(len(item["sha256"]) == 64 for item in first["weight_shards"])
    first_digest = first["weight_shards_digest"]

    first_shard.write_bytes(b"FIRST-SHARD")
    assert first_shard.stat().st_size == len(b"first-shard")
    second = cli.model_artifact_identity(model)
    assert second["weight_shards_digest"] != first_digest
    assert second["weight_shards"][0]["nbytes"] == first["weight_shards"][0]["nbytes"]
    assert second["weight_shards"][0]["sha256"] != first["weight_shards"][0]["sha256"]


def test_post_hash_preload_gpu_idle_gate_captures_double_sandwich_streak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(
        [
            "--preload-idle-samples",
            "2",
            "--preload-idle-sample-interval-seconds",
            "0",
            "--preload-idle-timeout-seconds",
            "10",
        ]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    memory = [
        {"physical_index": 4, "gpu_uuid": "GPU-four", "memory_used_mib": 3},
        {"physical_index": 5, "gpu_uuid": "GPU-five", "memory_used_mib": 4},
    ]
    calls = {"memory": 0, "apps": 0}

    def memory_rows():
        calls["memory"] += 1
        return memory

    def compute_rows():
        calls["apps"] += 1
        return []

    monkeypatch.setattr(cli, "_preload_gpu_memory_rows", memory_rows)
    monkeypatch.setattr(cli, "_preload_gpu_compute_rows", compute_rows)
    evidence = cli.wait_for_preload_gpu_idle_gate(args, (0, 1))
    assert evidence["status"] == "passed"
    assert evidence["attempt_count"] == 2
    assert evidence["rejected_attempt_count"] == 0
    assert [item["logical_index"] for item in evidence["selected_gpu_bindings"]] == [
        0,
        1,
    ]
    assert len(evidence["final_consecutive_sandwich_samples"]) == 2
    assert calls == {"memory": 4, "apps": 4}
    assert (
        cli.verify_preload_gpu_idle_gate_evidence(
            evidence,
            expected_policy=cli.preload_gpu_idle_gate_policy(args, (0, 1)),
        )
        == evidence
    )


def test_post_hash_preload_gpu_idle_gate_resets_streak_on_compute_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(
        [
            "--preload-idle-samples",
            "2",
            "--preload-idle-sample-interval-seconds",
            "0",
            "--preload-idle-timeout-seconds",
            "10",
        ]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    memory = [{"physical_index": 4, "gpu_uuid": "GPU-four", "memory_used_mib": 3}]
    app = {
        "gpu_uuid": "GPU-four",
        "pid": 123,
        "process_name": "intruder",
        "memory_used_mib": 1,
    }
    app_snapshots = iter([[app], [], [], [], [], []])
    monkeypatch.setattr(cli, "_preload_gpu_memory_rows", lambda: memory)
    monkeypatch.setattr(cli, "_preload_gpu_compute_rows", lambda: next(app_snapshots))
    evidence = cli.wait_for_preload_gpu_idle_gate(args, (0,))
    assert evidence["attempt_count"] == 3
    assert evidence["rejected_attempt_count"] == 1
    assert [
        item["sequence"] for item in evidence["final_consecutive_sandwich_samples"]
    ] == [2, 3]


def test_preload_gate_keeps_not_found_compute_row_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(
        cli,
        "_strict_nvidia_smi_rows",
        lambda **_kwargs: [["GPU-four", "123", "[Not Found]", "[Not Found]"]],
    )
    assert cli._preload_gpu_compute_rows() == [
        {
            "gpu_uuid": "GPU-four",
            "pid": 123,
            "process_name": "[Not Found]",
            "memory_used_mib": None,
        }
    ]


def test_preload_gate_evidence_rejects_resigned_binding_and_sequence_tamper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(
        [
            "--preload-idle-samples",
            "2",
            "--preload-idle-sample-interval-seconds",
            "0",
            "--preload-idle-timeout-seconds",
            "10",
        ]
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    memory = [{"physical_index": 4, "gpu_uuid": "GPU-four", "memory_used_mib": 3}]
    monkeypatch.setattr(cli, "_preload_gpu_memory_rows", lambda: memory)
    monkeypatch.setattr(cli, "_preload_gpu_compute_rows", lambda: [])
    evidence = cli.wait_for_preload_gpu_idle_gate(args, (0,))
    policy = cli.preload_gpu_idle_gate_policy(args, (0,))

    changed_binding = json.loads(json.dumps(evidence))
    changed_binding["final_consecutive_sandwich_samples"][0]["memory_before"][0][
        "physical_index"
    ] = 5
    with pytest.raises(ValueError, match="binding/limit"):
        cli.verify_preload_gpu_idle_gate_evidence(
            changed_binding, expected_policy=policy
        )

    changed_sequence = json.loads(json.dumps(evidence))
    changed_sequence["final_consecutive_sandwich_samples"][0]["sequence"] = 0
    with pytest.raises(ValueError, match="sequence"):
        cli.verify_preload_gpu_idle_gate_evidence(
            changed_sequence, expected_policy=policy
        )


def test_tiny_mode_does_not_enforce_production_preload_gate_policy() -> None:
    import scripts.benchmark_latency as cli

    args = cli.parse_args(["--mode", "tiny", "--preload-idle-samples", "1"])
    cli.validate_args(args)


def test_cuda_peak_memory_evidence_resets_after_sync_and_covers_selected_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    phase = {"value": "before"}
    synchronized: list[tuple[int, ...]] = []
    reset: list[int] = []
    allocated = {
        "before": {2: 100, 0: 50},
        "after": {2: 120, 0: 60},
    }
    reserved = {
        "before": {2: 200, 0: 80},
        "after": {2: 220, 0: 90},
    }
    driver = {
        "before": {2: (700, 1000), 0: (800, 1000)},
        "after": {2: (600, 1000), 0: (750, 1000)},
    }
    peaks = {2: (300, 400), 0: (150, 200)}

    class Properties:
        total_memory = 1000

    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        cli,
        "synchronize_devices",
        lambda devices: synchronized.append(tuple(int(device) for device in devices)),
    )
    monkeypatch.setattr(
        cli.torch.cuda, "get_device_properties", lambda _index: Properties()
    )
    monkeypatch.setattr(
        cli.torch.cuda,
        "mem_get_info",
        lambda index: driver[phase["value"]][int(index)],
    )
    monkeypatch.setattr(
        cli.torch.cuda,
        "memory_allocated",
        lambda index: allocated[phase["value"]][int(index)],
    )
    monkeypatch.setattr(
        cli.torch.cuda,
        "memory_reserved",
        lambda index: reserved[phase["value"]][int(index)],
    )
    monkeypatch.setattr(
        cli.torch.cuda,
        "reset_peak_memory_stats",
        lambda index: reset.append(int(index)),
    )
    monkeypatch.setattr(
        cli.torch.cuda,
        "max_memory_allocated",
        lambda index: (
            allocated["before"][int(index)]
            if phase["value"] == "before"
            else peaks[int(index)][0]
        ),
    )
    monkeypatch.setattr(
        cli.torch.cuda,
        "max_memory_reserved",
        lambda index: (
            reserved["before"][int(index)]
            if phase["value"] == "before"
            else peaks[int(index)][1]
        ),
    )

    fingerprint = "a" * 64
    state = cli.begin_cuda_peak_memory_measurement(
        (2, 0),
        prefix_length=32768,
        request_fingerprint=fingerprint,
    )
    assert synchronized == [(2, 0)]
    assert reset == [2, 0]
    phase["value"] = "after"
    evidence = cli.finish_cuda_peak_memory_measurement(state)
    assert synchronized == [(2, 0), (2, 0)]
    assert evidence["status"] == "captured"
    assert evidence["selected_device_ids"] == [2, 0]
    assert evidence["devices"]["2"]["peak_allocated_bytes"] == 300
    assert evidence["devices"]["2"]["peak_reserved_bytes"] == 400
    assert evidence["devices"]["2"]["capacity_headroom_at_peak_reserved_bytes"] == 600
    assert evidence["devices"]["2"]["driver_free_bytes_before"] == 700
    assert evidence["devices"]["2"]["driver_free_bytes_after"] == 600
    cli.verify_cuda_peak_memory_evidence(
        evidence,
        expected_device_ids=(2, 0),
        expected_prefix_length=32768,
        expected_request_fingerprint=fingerprint,
    )
    with pytest.raises(ValueError, match="selected devices"):
        cli.verify_cuda_peak_memory_evidence(
            evidence,
            expected_device_ids=(0, 2),
            expected_prefix_length=32768,
            expected_request_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="prefix_length"):
        cli.verify_cuda_peak_memory_evidence(
            evidence,
            expected_device_ids=(2, 0),
            expected_prefix_length=64,
            expected_request_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="request_fingerprint"):
        cli.verify_cuda_peak_memory_evidence(
            evidence,
            expected_device_ids=(2, 0),
            expected_prefix_length=32768,
            expected_request_fingerprint="b" * 64,
        )
    bad_schema = json.loads(json.dumps(evidence))
    bad_schema["schema_version"] = True
    with pytest.raises(ValueError, match="unsupported schema"):
        cli.verify_cuda_peak_memory_evidence(
            bad_schema,
            expected_device_ids=(2, 0),
            expected_prefix_length=32768,
            expected_request_fingerprint=fingerprint,
        )
    bad_devices = json.loads(json.dumps(evidence))
    bad_devices["selected_device_ids"] = [True, 0]
    with pytest.raises(ValueError, match="exact integers"):
        cli.verify_cuda_peak_memory_evidence(
            bad_devices,
            expected_device_ids=(2, 0),
            expected_prefix_length=32768,
            expected_request_fingerprint=fingerprint,
        )
    bad_scope = json.loads(json.dumps(evidence))
    bad_scope["measurement_scope"] = "shorter_scope"
    with pytest.raises(ValueError, match="measurement_scope"):
        cli.verify_cuda_peak_memory_evidence(
            bad_scope,
            expected_device_ids=(2, 0),
            expected_prefix_length=32768,
            expected_request_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="exact integers"):
        cli.begin_cuda_peak_memory_measurement(
            (True,),
            prefix_length=32768,
            request_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="exact integers"):
        cli.begin_cuda_peak_memory_measurement(
            (1.0,),  # type: ignore[arg-type]
            prefix_length=32768,
            request_fingerprint=fingerprint,
        )

    malformed = json.loads(json.dumps(evidence))
    del malformed["devices"]["0"]["peak_reserved_bytes"]
    with pytest.raises(ValueError, match="peak_reserved_bytes"):
        cli.verify_cuda_peak_memory_evidence(
            malformed,
            expected_device_ids=(2, 0),
            expected_prefix_length=32768,
            expected_request_fingerprint=fingerprint,
        )


def test_cpu_length_cell_has_explicit_not_applicable_cuda_memory_evidence() -> None:
    import scripts.benchmark_latency as cli

    fingerprint = "c" * 64
    state = cli.begin_cuda_peak_memory_measurement(
        (),
        prefix_length=4,
        request_fingerprint=fingerprint,
    )
    evidence = cli.finish_cuda_peak_memory_measurement(state)
    assert evidence["status"] == "not_applicable"
    assert evidence["selected_device_ids"] == []
    assert evidence["devices"] == {}
    cli.verify_cuda_peak_memory_evidence(
        evidence,
        expected_device_ids=(),
        expected_prefix_length=4,
        expected_request_fingerprint=fingerprint,
    )
    with pytest.raises(ValueError, match="lacks CUDA peak-memory evidence"):
        cli.verify_cuda_peak_memory_evidence(
            None,
            expected_device_ids=(),
            expected_prefix_length=4,
            expected_request_fingerprint=fingerprint,
        )
    missing_cpu_flag = json.loads(json.dumps(evidence))
    del missing_cpu_flag["synchronized_after_cell"]
    with pytest.raises(ValueError, match="CPU latency checkpoint"):
        cli.verify_cuda_peak_memory_evidence(
            missing_cpu_flag,
            expected_device_ids=(),
            expected_prefix_length=4,
            expected_request_fingerprint=fingerprint,
        )


def test_successful_cell_peak_finish_failure_is_persisted_explicitly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = tmp_path / "peak-finish-failure"

    def fail_finish(_state):
        raise RuntimeError("intentional peak finish failure")

    monkeypatch.setattr(cli, "finish_cuda_peak_memory_measurement", fail_finish)
    argv = [
        "--mode",
        "tiny",
        "--tiny-device",
        "cpu",
        "--lengths",
        "4",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--profile-warmup",
        "0",
        "--profile-repeats",
        "1",
        "--output-dir",
        str(output),
        "--no-ledger",
    ]
    with pytest.raises(RuntimeError, match="intentional peak finish failure"):
        cli.main(argv)
    failure = json.loads((output / "result.json").read_text(encoding="utf-8"))
    evidence = failure["cuda_peak_memory"]
    assert failure["prefix_length"] == 4
    assert evidence["schema_version"] == cli.CUDA_PEAK_MEMORY_SCHEMA_VERSION
    assert evidence["status"] == "capture_failed"
    assert evidence["capture_failure_stage"] == "after_successful_cell"
    assert evidence["prefix_length"] == 4
    assert evidence["request_fingerprint"] == failure["request_fingerprint"]


def test_tiny_cli_writes_json_csv_checkpoints_and_reproduction_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    (tmp_path / "REPRODUCTION.md").write_text("# test ledger\n", encoding="utf-8")
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_latency.py",
            "--mode",
            "tiny",
            "--tiny-device",
            "cpu",
            "--lengths",
            "4",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--profile-warmup",
            "0",
            "--profile-repeats",
            "1",
            "--mapper-token-chunk",
            "2",
        ],
    )
    assert cli.main() == 0
    run_directories = list((tmp_path / "runs").iterdir())
    assert len(run_directories) == 1
    run = run_directories[0]
    for name in (
        "command.sh",
        "config.json",
        "environment.json",
        "hardware.json",
        "mapper_setup.json",
        "length_00004.json",
        "metrics.json",
        "metrics.csv",
        "result.json",
        "status.json",
    ):
        assert (run / name).is_file(), name
    metrics = json.loads((run / "metrics.json").read_text())
    assert metrics["status"] == "complete"
    assert metrics["protocol"]["source_prefill_included"] is False
    assert metrics["protocol"]["absolute_latency_directly_comparable_to_paper"] is False
    assert metrics["length_results"][0]["prefix_length"] == 4
    assert metrics["length_results"][0]["target_reprefill"]["count"] == 1
    assert (
        metrics["length_results"][0]["cuda_peak_memory"]["status"] == "not_applicable"
    )
    length_result = metrics["length_results"][0]
    assert length_result["nested_cuda_peak_memory_contract"] == {
        "outer_cell_peak_measurement_active": True,
        "internal_metric_peak_collection_enabled": False,
        "internal_peak_counter_resets_disabled": True,
    }
    assert length_result["latency_protocol"] == {
        "name": "paper_main_only",
        "paper_main_metrics": ["target_reprefill", "mapper_handoff_end_to_end"],
        "supplemental_diagnostics_enabled": False,
        "component_profile_in_cell": False,
        "boundary_decode_in_cell": False,
        "protocol_complete_in_cell": False,
        "paper_headline_speedup_uses_supplemental_metrics": False,
    }
    for metric in ("target_reprefill", "mapper_handoff_end_to_end"):
        summary = length_result[metric]
        assert summary["cuda_peak_memory_collected"] is False
        assert "cuda_peak_allocated_bytes" not in summary
        assert "cuda_peak_reserved_bytes" not in summary
        assert summary["inference_mode_enabled"] is True
        assert summary["autograd_graph_disabled"] is True
    for field in (
        "mapper_synchronized_component_profile",
        "mapper_kernel_only_profile",
        "boundary_one_token_decode",
        "protocol_complete",
        "median_speedup_target_reprefill_over_protocol_complete",
    ):
        assert field not in length_result
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["metric"] for row in rows} == {
        "target_reprefill",
        "mapper_handoff_end_to_end",
    }
    ledger = (tmp_path / "REPRODUCTION.md").read_text(encoding="utf-8")
    assert "P8-LATENCY" in ledger
    assert "source prefill 未计时" in ledger


def test_tiny_cli_supplemental_diagnostics_are_explicit_and_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = tmp_path / "supplemental-latency"
    argv = [
        "--mode",
        "tiny",
        "--tiny-device",
        "cpu",
        "--lengths",
        "4",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--profile-warmup",
        "0",
        "--profile-repeats",
        "1",
        "--mapper-token-chunk",
        "2",
        "--supplemental-diagnostics",
        "--output-dir",
        str(output),
        "--no-ledger",
    ]
    assert cli.main(argv) == 0
    value = json.loads((output / "length_00004.json").read_text(encoding="utf-8"))
    assert value["latency_protocol"]["name"] == "paper_main_plus_supplemental"
    assert value["latency_protocol"]["supplemental_diagnostics_enabled"] is True
    for field in (
        "mapper_synchronized_component_profile",
        "mapper_kernel_only_profile",
        "boundary_one_token_decode",
        "protocol_complete",
        "median_speedup_target_reprefill_over_protocol_complete",
    ):
        assert field in value
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert config["latency_protocol"] == "paper_main_plus_supplemental"
    assert config["supplemental_diagnostics_enabled"] is True


def test_tiny_cli_exact_fingerprint_resume_and_hash_tamper_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = tmp_path / "fixed-latency-run"
    argv = [
        "--mode",
        "tiny",
        "--tiny-device",
        "cpu",
        "--lengths",
        "4",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--profile-warmup",
        "0",
        "--profile-repeats",
        "1",
        "--mapper-token-chunk",
        "2",
        "--output-dir",
        str(output),
        "--no-ledger",
    ]
    assert cli.main(argv) == 0
    checkpoint = output / "length_00004.json"
    before_sha = cli.sha256_file(checkpoint)
    assert cli.main(argv) == 0
    assert cli.sha256_file(checkpoint) == before_sha
    value = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert value["strict_transfer_preflight"]["status"] == "passed"
    assert (
        value["timed_transfer_fast_path"][
            "diagnostic_rope_and_finite_reductions_in_timed_region"
        ]
        is False
    )
    assert (output / "results.manifest.json").is_file()
    assert (output / "metrics.json.sha256.json").is_file()
    assert (output / "metrics.csv.sha256.json").is_file()

    without_memory = json.loads(json.dumps(value))
    del without_memory["cuda_peak_memory"]
    checkpoint.write_text(json.dumps(without_memory), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="lacks CUDA peak-memory evidence"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    invalid_preflight = json.loads(json.dumps(value))
    invalid_preflight["strict_transfer_preflight"]["mapped_cache_length"] = 5
    checkpoint.write_text(json.dumps(invalid_preflight), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="invalid mapped_cache_length"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    invalid_wall = json.loads(json.dumps(value))
    invalid_wall["strict_transfer_preflight"]["wall_ms_diagnostic_not_latency"] = float(
        "nan"
    )
    checkpoint.write_text(json.dumps(invalid_wall), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="invalid diagnostic wall time"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    invalid_layers = json.loads(json.dumps(value))
    invalid_layers["strict_transfer_preflight"]["validated_target_layers"] = 1
    checkpoint.write_text(json.dumps(invalid_layers), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="invalid validated_target_layers"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    invalid_dtype = json.loads(json.dumps(value))
    invalid_dtype["strict_transfer_preflight"]["mapped_cache_dtypes"] = ["garbage"]
    checkpoint.write_text(json.dumps(invalid_dtype), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="invalid mapped_cache_dtypes"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    nested_reset = json.loads(json.dumps(value))
    nested_reset["nested_cuda_peak_memory_contract"][
        "internal_peak_counter_resets_disabled"
    ] = False
    checkpoint.write_text(json.dumps(nested_reset), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="did not disable nested CUDA peak resets"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    contaminated = json.loads(json.dumps(value))
    contaminated["protocol_complete"] = contaminated["mapper_handoff_end_to_end"]
    checkpoint.write_text(json.dumps(contaminated), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="contains supplemental fields"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    invalid_inference = json.loads(json.dumps(value))
    invalid_inference["target_reprefill"]["inference_mode_enabled"] = False
    checkpoint.write_text(json.dumps(invalid_inference), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="lacks inference-mode evidence"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    invalid_autograd = json.loads(json.dumps(value))
    invalid_autograd["mapper_handoff_end_to_end"]["autograd_graph_disabled"] = False
    checkpoint.write_text(json.dumps(invalid_autograd), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    with pytest.raises(ValueError, match="lacks disabled-autograd evidence"):
        cli.main(argv)

    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    cli.write_sha_sidecar(checkpoint)
    value["prefix_length"] = 5
    checkpoint.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 sidecar mismatch"):
        cli.main(argv)


def test_latency_resume_rejects_any_cli_fingerprint_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = tmp_path / "fixed-latency-run"
    common = [
        "--mode",
        "tiny",
        "--lengths",
        "4",
        "--warmup",
        "0",
        "--profile-warmup",
        "0",
        "--profile-repeats",
        "1",
        "--output-dir",
        str(output),
        "--no-ledger",
    ]
    assert cli.main([*common, "--repeats", "1"]) == 0
    with pytest.raises(ValueError, match="fingerprint/definition mismatch"):
        cli.main([*common, "--repeats", "2"])


def test_latency_fingerprint_binds_cuda_visible_devices_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = tmp_path / "visibility-bound-run"
    argv = [
        "--mode",
        "tiny",
        "--tiny-device",
        "cpu",
        "--lengths",
        "4",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--profile-warmup",
        "0",
        "--profile-repeats",
        "1",
        "--output-dir",
        str(output),
        "--no-ledger",
    ]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7,0,1")
    assert cli.main(argv) == 0
    configuration = json.loads((output / "config.json").read_text())
    definition = json.loads((output / "run_definition.json").read_text())
    assert configuration["cuda_visible_devices_environment"] == "4,5,6,7,0,1"
    assert definition["topology"]["cuda_visible_devices_environment"] == "4,5,6,7,0,1"

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5")
    with pytest.raises(ValueError, match="fingerprint/definition mismatch"):
        cli.main(argv)


def test_partial_latency_run_is_not_complete_and_resumes_only_missing_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.benchmark_latency as cli

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    output = tmp_path / "partial-latency-run"
    argv = [
        "--mode",
        "tiny",
        "--lengths",
        "4,8",
        "--warmup",
        "0",
        "--repeats",
        "1",
        "--profile-warmup",
        "0",
        "--profile-repeats",
        "1",
        "--output-dir",
        str(output),
        "--no-ledger",
    ]
    original = cli.run_one_length
    first_calls: list[int] = []

    def fail_on_second(**kwargs):
        length = int(kwargs["length"])
        first_calls.append(length)
        if length == 8:
            raise RuntimeError("intentional interruption")
        return original(**kwargs)

    monkeypatch.setattr(cli, "run_one_length", fail_on_second)
    with pytest.raises(RuntimeError, match="intentional interruption"):
        cli.main(argv)
    assert first_calls == [4, 8]
    assert (output / "length_00004.json").is_file()
    assert not (output / "length_00008.json").exists()
    failure = json.loads((output / "result.json").read_text())
    assert failure["status"] == "failed"
    assert failure["prefix_length"] == 8
    assert failure["cuda_peak_memory"]["status"] == "not_applicable"
    assert not (output / "results.manifest.json").exists()

    resumed_calls: list[int] = []

    def record_resume(**kwargs):
        resumed_calls.append(int(kwargs["length"]))
        return original(**kwargs)

    monkeypatch.setattr(cli, "run_one_length", record_resume)
    assert cli.main(argv) == 0
    assert resumed_calls == [8]
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["reused_checkpoint_lengths"] == [4]
    assert [item["prefix_length"] for item in metrics["length_results"]] == [4, 8]
    assert json.loads((output / "result.json").read_text())["status"] == "complete"


@pytest.mark.cuda
def test_single_cuda_device_uses_synchronized_cuda_events() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda:0")
    tensor = torch.randn((128, 128), device=device)
    timed = measure_synchronized(lambda: tensor @ tensor, devices=(device,))
    assert timed.timing_backend == "cuda_event"
    assert timed.milliseconds >= 0.0
    assert timed.value.device == device


@pytest.mark.cuda
def test_two_cuda_devices_use_host_clock_not_cross_device_events() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices unavailable")
    left = torch.device("cuda:0")
    right = torch.device("cuda:1")
    timed = measure_synchronized(
        lambda: (torch.ones(1, device=left), torch.ones(1, device=right)),
        devices=(left, right),
    )
    assert timed.timing_backend == "host_perf_counter_with_multi_cuda_synchronize"
    assert timed.milliseconds >= 0.0


@pytest.mark.cuda
def test_real_cuda_peak_memory_evidence_observes_allocator_growth() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    import scripts.benchmark_latency as cli

    device_index = 0
    fingerprint = "d" * 64
    state = cli.begin_cuda_peak_memory_measurement(
        (device_index,),
        prefix_length=8,
        request_fingerprint=fingerprint,
    )
    baseline = state["devices"][str(device_index)]["allocated_bytes_before"]
    tensor = torch.empty(
        8 * 1024 * 1024, dtype=torch.uint8, device=f"cuda:{device_index}"
    )
    evidence = cli.finish_cuda_peak_memory_measurement(state)
    cli.verify_cuda_peak_memory_evidence(
        evidence,
        expected_device_ids=(device_index,),
        expected_prefix_length=8,
        expected_request_fingerprint=fingerprint,
    )
    assert evidence["devices"][str(device_index)]["peak_allocated_bytes"] >= (
        baseline + tensor.numel()
    )
    del tensor
    torch.cuda.empty_cache()
