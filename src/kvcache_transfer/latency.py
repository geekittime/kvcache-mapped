"""Latency primitives for the Qwen3-14B -> Qwen3-32B reproduction.

The paper compares target-model re-prefill with cache mapping *after* the
source cache already exists.  This module deliberately keeps source prefill
outside every timed region.  It exposes two complementary measurements:

* an unsplit, end-to-end mapper call with only start/end synchronization;
* an explicitly synchronized diagnostic profile which attributes time to
  inverse RoPE, source-cache handoff, FP32 affine GEMMs, target RoPE, and
  output casting/cache construction.

The diagnostic profile changes scheduling and therefore is not a replacement
for the end-to-end number.  Its component sum is reported only as a profile,
never as paper-style serving latency.

CUDA events are used only when the complete timed region runs on one CUDA
device.  A CUDA event belongs to one device and cannot correctly bracket a
pipeline spanning several GPUs.  Multi-GPU measurements consequently use a
host monotonic clock bracketed by explicit synchronization of every involved
device.  This distinction is persisted with each result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, TypeVar

import torch
from transformers.cache_utils import Cache, DynamicCache

from .cache_adapter import (
    build_dynamic_cache,
    cache_sequence_length,
    cache_to_legacy,
    qwen3_layer_devices,
)
from .config import ArchitectureSpec
from .mapper import LayerMapper
from .rope import apply_rope, qwen3_rope_cos_sin_from_config
from .transfer import (
    MapperBundle,
    TransferValidationError,
    source_cache_to_content,
    target_boundary_replay,
    transfer_source_cache,
)


T = TypeVar("T")


COMPONENT_NAMES = (
    "inverse_source_rope_fp32",
    "source_cache_cross_device_handoff",
    "feature_pack_and_fp32_cast",
    "mapper_fp32_matmul_and_bias",
    "target_rope_fp32_and_bf16_cast",
    "value_bf16_cast_and_cache_build",
)


def configure_strict_latency_math() -> dict[str, Any]:
    """Disable TF32 and record the eager/precision contract used by the run."""

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "model_forward_dtype": "bfloat16",
        "mapper_dtype": "float32",
        "allow_tf32_cuda_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mapper_execution": "eager",
        "torch_compile": False,
        "cuda_graph": False,
    }


def _as_unique_devices(devices: Sequence[torch.device | str | int]) -> tuple[torch.device, ...]:
    result: list[torch.device] = []
    seen: set[tuple[str, int | None]] = set()
    for value in devices:
        if isinstance(value, int):
            device = torch.device("cuda", value)
        else:
            device = torch.device(value)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        key = (device.type, device.index)
        if key not in seen:
            seen.add(key)
            result.append(device)
    return tuple(result)


def synchronize_devices(devices: Sequence[torch.device | str | int]) -> None:
    """Synchronize each distinct CUDA device, without changing peer settings."""

    for device in _as_unique_devices(devices):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@dataclass(frozen=True)
class TimedValue:
    value: Any
    milliseconds: float
    timing_backend: str


def measure_synchronized(
    operation: Callable[[], T],
    *,
    devices: Sequence[torch.device | str | int] = (),
    force_host_clock: bool = False,
) -> TimedValue:
    """Measure one operation with a valid synchronization boundary.

    A single CUDA device uses events and synchronizes the end event before
    reading ``elapsed_time``.  CPU and multi-GPU regions use ``perf_counter``
    with every CUDA device synchronized both before and after the operation.
    ``force_host_clock`` is useful for Python-heavy cache construction and
    host-to-device setup, where CUDA events would intentionally omit host work.
    """

    resolved = _as_unique_devices(devices)
    cuda_devices = tuple(device for device in resolved if device.type == "cuda")
    if len(cuda_devices) == 1 and not force_host_clock:
        device = cuda_devices[0]
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(device))
            value = operation()
            end.record(torch.cuda.current_stream(device))
            # Reading an event before it completes yields invalid or implicit
            # timing.  Synchronizing the end event is the required boundary.
            end.synchronize()
            milliseconds = float(start.elapsed_time(end))
        return TimedValue(value, milliseconds, "cuda_event")

    synchronize_devices(resolved)
    started = time.perf_counter_ns()
    value = operation()
    synchronize_devices(resolved)
    milliseconds = (time.perf_counter_ns() - started) / 1_000_000.0
    backend = (
        "host_perf_counter_with_multi_cuda_synchronize"
        if len(cuda_devices) > 1
        else "host_perf_counter_with_cuda_synchronize"
        if len(cuda_devices) == 1
        else "host_perf_counter"
    )
    return TimedValue(value, float(milliseconds), backend)


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear-interpolated percentile with deterministic one-sample behavior."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a percentile of no samples")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_latency(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Return the complete persisted latency summary for millisecond samples."""

    samples = tuple(float(value) for value in samples_ms)
    if not samples:
        raise ValueError("at least one timed sample is required")
    if not all(math.isfinite(value) and value >= 0.0 for value in samples):
        raise ValueError("latency samples must be finite and non-negative")
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    return {
        "unit": "ms",
        "count": len(samples),
        "mean": float(mean),
        "median": float(median),
        "p10": float(percentile(samples, 0.10)),
        "p50": float(percentile(samples, 0.50)),
        "p90": float(percentile(samples, 0.90)),
        "std_population": float(statistics.pstdev(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "samples": list(samples),
    }


def benchmark_callable(
    operation: Callable[[T], Any],
    *,
    prepare: Callable[[], T],
    warmup: int = 50,
    repeats: int = 30,
    devices: Sequence[torch.device | str | int] = (),
    force_host_clock: bool = False,
    release: Callable[[Any, T], None] | None = None,
    collect_cuda_peak_memory: bool = True,
) -> dict[str, Any]:
    """Warm up and benchmark an operation, optionally preparing state untimed.

    ``prepare`` runs before the start synchronization.  This is required for
    one-token cache decode: every iteration receives a fresh cloned prefix,
    but cloning the entire cache is not charged to the decode kernel.

    ``collect_cuda_peak_memory=False`` is required when a caller owns a wider
    peak-memory measurement window.  In that mode this function neither resets
    nor reads the process-global CUDA allocator peak counters, and it omits the
    per-metric peak fields rather than relabeling the caller's outer peak.
    """

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    resolved_devices = _as_unique_devices(devices)
    cuda_devices = tuple(device for device in resolved_devices if device.type == "cuda")
    for _ in range(warmup):
        state = prepare()
        output = operation(state)
        synchronize_devices(resolved_devices)
        if release is not None:
            release(output, state)
        del output, state

    if collect_cuda_peak_memory:
        for device in cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    backends: set[str] = set()
    for _ in range(repeats):
        state = prepare()
        timed = measure_synchronized(
            lambda: operation(state),
            devices=resolved_devices,
            force_host_clock=force_host_clock,
        )
        samples.append(timed.milliseconds)
        backends.add(timed.timing_backend)
        if release is not None:
            release(timed.value, state)
        del timed, state
    summary = summarize_latency(samples)
    summary.update(
        {
            "warmup": warmup,
            "timed_repeats": repeats,
            "timing_backends": sorted(backends),
            "synchronize_each_iteration": True,
            "cuda_peak_memory_collected": bool(collect_cuda_peak_memory),
        }
    )
    if collect_cuda_peak_memory:
        summary["cuda_peak_allocated_bytes"] = {
            str(device): int(torch.cuda.max_memory_allocated(device)) for device in cuda_devices
        }
        summary["cuda_peak_reserved_bytes"] = {
            str(device): int(torch.cuda.max_memory_reserved(device)) for device in cuda_devices
        }
    else:
        summary["cuda_peak_memory_disabled_reason"] = (
            "preserve_caller_owned_outer_cuda_peak_counters"
        )
    return summary


def _move_layer_mapper(mapper: LayerMapper, device: torch.device) -> LayerMapper:
    return LayerMapper(
        source_layers=mapper.source_layers,
        key_weight=mapper.key_weight.to(device=device, dtype=torch.float32),
        key_bias=mapper.key_bias.to(device=device, dtype=torch.float32),
        value_weight=mapper.value_weight.to(device=device, dtype=torch.float32),
        value_bias=mapper.value_bias.to(device=device, dtype=torch.float32),
    )


@dataclass(frozen=True)
class ResidentMapperBundle:
    """MapperBundle-compatible container with all FP32 layers GPU-resident."""

    architecture: ArchitectureSpec
    layers: tuple[LayerMapper, ...]
    manifest_sha256: str | None = None
    fit_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if len(self.layers) != self.architecture.target_layers:
            raise ValueError("resident mapper must contain every target layer")
        for mapper in self.layers:
            mapper.validate(
                source_heads=self.architecture.source_kv_heads,
                head_dim=self.architecture.source_head_dim,
            )
            if {
                mapper.key_weight.dtype,
                mapper.key_bias.dtype,
                mapper.value_weight.dtype,
                mapper.value_bias.dtype,
            } != {torch.float32}:
                raise ValueError("resident mapper tensors must all be FP32")

    @property
    def resident_parameter_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for mapper in self.layers
            for tensor in (
                mapper.key_weight,
                mapper.key_bias,
                mapper.value_weight,
                mapper.value_bias,
            )
        )

    def iter_layers(
        self, layer_devices: Sequence[torch.device | str]
    ):
        if len(layer_devices) != len(self.layers):
            raise ValueError("target device count does not match resident mapper")
        for target_layer, (mapper, expected_value) in enumerate(
            zip(self.layers, layer_devices, strict=True)
        ):
            expected = torch.device(expected_value)
            if mapper.key_weight.device != expected:
                raise ValueError(
                    f"resident mapper layer {target_layer} is on "
                    f"{mapper.key_weight.device}, expected {expected}"
                )
            yield target_layer, mapper


def materialize_mapper_on_target(
    bundle: MapperBundle,
    layer_devices: Sequence[torch.device | str],
) -> tuple[ResidentMapperBundle, dict[str, Any]]:
    """Load mapper files on CPU, then place each complete layer on its target GPU.

    These are one-time setup measurements and are not included in per-request
    mapper latency.  Host file/hash/deserialization and host-to-device transfer
    are reported separately.
    """

    devices = tuple(torch.device(value) for value in layer_devices)
    if len(devices) != bundle.architecture.target_layers:
        raise ValueError("target layer-device list does not match mapper architecture")
    resident: list[LayerMapper] = []
    host_load_samples: list[float] = []
    device_transfer_samples: list[float] = []
    transfer_backends: set[str] = set()
    per_layer: list[dict[str, Any]] = []
    for target_layer, device in enumerate(devices):
        loaded = measure_synchronized(
            lambda target_layer=target_layer: bundle.load_layer(target_layer, device="cpu"),
            force_host_clock=True,
        )
        moved = measure_synchronized(
            lambda loaded=loaded, device=device: _move_layer_mapper(loaded.value, device),
            devices=(device,),
            force_host_clock=True,
        )
        mapper = moved.value
        mapper.validate(
            source_heads=bundle.architecture.source_kv_heads,
            head_dim=bundle.architecture.source_head_dim,
        )
        host_load_samples.append(loaded.milliseconds)
        device_transfer_samples.append(moved.milliseconds)
        transfer_backends.add(moved.timing_backend)
        per_layer.append(
            {
                "target_layer": target_layer,
                "device": str(device),
                "host_load_and_hash_ms": loaded.milliseconds,
                "weight_host_to_device_ms": moved.milliseconds,
            }
        )
        resident.append(mapper)
    return (
        ResidentMapperBundle(
            architecture=bundle.architecture,
            layers=tuple(resident),
            manifest_sha256=bundle.manifest_sha256,
            fit_fingerprint=bundle.fit_fingerprint,
        ),
        {
            "scope": "one_time_setup_excluded_from_per_request_latency",
            "host_load_and_hash": summarize_latency(host_load_samples),
            "weight_host_to_device": {
                **summarize_latency(device_transfer_samples),
                "timing_backends": sorted(transfer_backends),
            },
            "per_layer": per_layer,
        },
    )


def _record_component(
    samples: dict[str, float],
    backends: dict[str, set[str]],
    name: str,
    timed: TimedValue,
) -> Any:
    samples[name] += timed.milliseconds
    backends[name].add(timed.timing_backend)
    return timed.value


@torch.inference_mode()
def profile_transfer_once(
    source_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    source_config: Any,
    target_model: torch.nn.Module,
    mapper_bundle: ResidentMapperBundle,
    output_dtype: torch.dtype | None = None,
    token_chunk: int = 1024,
    strict_validation: bool = True,
) -> tuple[DynamicCache, dict[str, float], dict[str, list[str]]]:
    """Run one mapper conversion with synchronized diagnostic attribution."""

    if token_chunk <= 0:
        raise ValueError("token_chunk must be positive")
    architecture = mapper_bundle.architecture
    target_devices = qwen3_layer_devices(target_model)
    if len(target_devices) != architecture.target_layers:
        raise TransferValidationError("target model layer count does not match mapper")
    source_pairs = cache_to_legacy(source_cache)
    source_devices = tuple(key.device for key, _ in source_pairs)
    target_layers = getattr(getattr(target_model, "model", target_model), "layers")
    target_dtypes = tuple(next(layer.parameters()).dtype for layer in target_layers)
    if output_dtype is None:
        if len(set(target_dtypes)) != 1:
            raise TransferValidationError("target decoder layers have mixed dtypes")
        resolved_dtype = target_dtypes[0]
    else:
        resolved_dtype = output_dtype
        if any(dtype != resolved_dtype for dtype in target_dtypes):
            raise TransferValidationError("mapped cache dtype differs from target layers")

    timings = {name: 0.0 for name in COMPONENT_NAMES}
    backends = {name: set() for name in COMPONENT_NAMES}
    inverted = measure_synchronized(
        lambda: source_cache_to_content(
            source_cache,
            source_config,
            expected_layers=architecture.source_layers,
            expected_heads=architecture.source_kv_heads,
            expected_head_dim=architecture.source_head_dim,
            strict_validation=strict_validation,
        ),
        devices=source_devices,
    )
    content_keys, source_values, positions = _record_component(
        timings, backends, "inverse_source_rope_fp32", inverted
    )

    batch_size = content_keys[0].shape[0]
    sequence_length = content_keys[0].shape[2]
    heads = architecture.source_kv_heads
    head_dim = architecture.source_head_dim
    target_features = architecture.target_features
    result = DynamicCache()

    for target_layer, mapper in mapper_bundle.iter_layers(target_devices):
        device = target_devices[target_layer]
        position_source = positions.device
        local_positions_timed = measure_synchronized(
            lambda: positions.to(device),
            devices=(position_source, device),
            force_host_clock=position_source != device,
        )
        local_positions = _record_component(
            timings,
            backends,
            "source_cache_cross_device_handoff",
            local_positions_timed,
        )
        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        for begin in range(0, sequence_length, token_chunk):
            end = min(begin + token_chunk, sequence_length)

            def handoff() -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
                key_pieces = tuple(
                    content_keys[index][:, :, begin:end, :].to(device=device)
                    for index in mapper.source_layers
                )
                value_pieces = tuple(
                    source_values[index][:, :, begin:end, :].to(device=device)
                    for index in mapper.source_layers
                )
                return key_pieces, value_pieces

            handoff_devices = (device,) + tuple(
                content_keys[index].device for index in mapper.source_layers
            ) + tuple(source_values[index].device for index in mapper.source_layers)
            handed = measure_synchronized(
                handoff,
                devices=handoff_devices,
                # On this 4090 host peer access is unavailable; host staging
                # and Python dispatch are part of the handoff, so an event on
                # the destination stream alone would under-count it.
                force_host_clock=True,
            )
            handed_keys, handed_values = _record_component(
                timings,
                backends,
                "source_cache_cross_device_handoff",
                handed,
            )

            def pack_features() -> tuple[torch.Tensor, torch.Tensor]:
                return (
                    torch.cat(
                        tuple(
                            value.permute(0, 2, 1, 3).flatten(2).float()
                            for value in handed_keys
                        ),
                        dim=-1,
                    ),
                    torch.cat(
                        tuple(
                            value.permute(0, 2, 1, 3).flatten(2).float()
                            for value in handed_values
                        ),
                        dim=-1,
                    ),
                )

            x_key, x_value = _record_component(
                timings,
                backends,
                "feature_pack_and_fp32_cast",
                measure_synchronized(pack_features, devices=(device,)),
            )

            def affine() -> tuple[torch.Tensor, torch.Tensor]:
                return (
                    x_key @ mapper.key_weight + mapper.key_bias,
                    x_value @ mapper.value_weight + mapper.value_bias,
                )

            affine_result = measure_synchronized(affine, devices=(device,))
            key_part, value_part = _record_component(
                timings,
                backends,
                "mapper_fp32_matmul_and_bias",
                affine_result,
            )
            key_parts.append(key_part)
            value_parts.append(value_part)

        def target_rope() -> torch.Tensor:
            content_key = (
                torch.cat(key_parts, dim=1)
                .unflatten(-1, (heads, head_dim))
                .transpose(1, 2)
            )
            cos, sin = qwen3_rope_cos_sin_from_config(
                local_positions,
                target_model.config,
                device=device,
                dtype=torch.float32,
            )
            return apply_rope(
                content_key,
                cos,
                sin,
                output_dtype=resolved_dtype,
            )

        target_key = _record_component(
            timings,
            backends,
            "target_rope_fp32_and_bf16_cast",
            measure_synchronized(
                target_rope,
                devices=(device,),
            ),
        )

        def cast_value_and_build() -> torch.Tensor:
            target_value = (
                torch.cat(value_parts, dim=1)
                .unflatten(-1, (heads, head_dim))
                .transpose(1, 2)
                .to(resolved_dtype)
            )
            expected = (batch_size, heads, sequence_length, head_dim)
            if tuple(target_key.shape) != expected or tuple(target_value.shape) != expected:
                raise TransferValidationError(
                    f"profiled mapper target layer {target_layer} has invalid shape"
                )
            result.update(target_key.detach(), target_value.detach(), target_layer)
            return target_value

        _record_component(
            timings,
            backends,
            "value_bf16_cast_and_cache_build",
            measure_synchronized(
                cast_value_and_build,
                devices=(device,),
                # DynamicCache.update is Python work and must not disappear
                # from an event-only measurement.
                force_host_clock=True,
            ),
        )
    if len(result) != architecture.target_layers:
        raise TransferValidationError("profiled mapped cache is missing target layers")
    if cache_sequence_length(result) != sequence_length:
        raise TransferValidationError("profiled mapper changed the cache sequence length")
    return result, timings, {name: sorted(values) for name, values in backends.items()}


def benchmark_mapper_components(
    source_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    source_config: Any,
    target_model: torch.nn.Module,
    mapper_bundle: ResidentMapperBundle,
    output_dtype: torch.dtype | None = None,
    token_chunk: int = 1024,
    warmup: int = 50,
    repeats: int = 30,
    strict_validation: bool = True,
    collect_cuda_peak_memory: bool = True,
) -> dict[str, Any]:
    """Profile mapper components after warming the exact end-to-end runtime.

    Disable per-profile peak collection when a wider caller-owned CUDA peak
    interval is active; this prevents a nested reset from erasing that peak.
    """

    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    all_devices = tuple(key.device for key, _ in cache_to_legacy(source_cache)) + tuple(
        qwen3_layer_devices(target_model)
    )
    for _ in range(warmup):
        mapped = transfer_source_cache(
            source_cache,
            source_config=source_config,
            target_model=target_model,
            mapper_bundle=mapper_bundle,  # type: ignore[arg-type]
            output_dtype=output_dtype,
            token_chunk=token_chunk,
            strict_validation=strict_validation,
        )
        synchronize_devices(all_devices)
        del mapped

    cuda_devices = tuple(
        device for device in _as_unique_devices(all_devices) if device.type == "cuda"
    )
    if collect_cuda_peak_memory:
        for device in cuda_devices:
            torch.cuda.reset_peak_memory_stats(device)
    samples = {name: [] for name in COMPONENT_NAMES}
    observed_backends = {name: set() for name in COMPONENT_NAMES}
    for _ in range(repeats):
        mapped, current, backends = profile_transfer_once(
            source_cache,
            source_config=source_config,
            target_model=target_model,
            mapper_bundle=mapper_bundle,
            output_dtype=output_dtype,
            token_chunk=token_chunk,
            strict_validation=strict_validation,
        )
        for name in COMPONENT_NAMES:
            samples[name].append(current[name])
            observed_backends[name].update(backends[name])
        del mapped
    components = {
        name: {
            **summarize_latency(values),
            "warmup": warmup,
            "timed_repeats": repeats,
            "timing_backends": sorted(observed_backends[name]),
        }
        for name, values in samples.items()
    }
    component_sums = [
        sum(samples[name][index] for name in COMPONENT_NAMES)
        for index in range(repeats)
    ]
    result: dict[str, Any] = {
        "measurement_scope": "synchronized_diagnostic_profile",
        "is_end_to_end_serving_latency": False,
        "components_are_additive_serving_estimate": False,
        "reason": (
            "Profiling synchronizes between components and changes scheduling; use "
            "mapper_handoff_end_to_end for paper-style latency."
        ),
        "components": components,
        "synchronized_component_sum": summarize_latency(component_sums),
        "cuda_peak_memory_collected": bool(collect_cuda_peak_memory),
    }
    if collect_cuda_peak_memory:
        result["cuda_peak_allocated_bytes"] = {
            str(device): int(torch.cuda.max_memory_allocated(device)) for device in cuda_devices
        }
        result["cuda_peak_reserved_bytes"] = {
            str(device): int(torch.cuda.max_memory_reserved(device)) for device in cuda_devices
        }
    else:
        result["cuda_peak_memory_disabled_reason"] = (
            "preserve_caller_owned_outer_cuda_peak_counters"
        )
    return result


@torch.inference_mode()
def benchmark_mapper_end_to_end(
    source_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    source_config: Any,
    target_model: torch.nn.Module,
    mapper_bundle: ResidentMapperBundle,
    output_dtype: torch.dtype | None = None,
    token_chunk: int = 1024,
    warmup: int = 50,
    repeats: int = 30,
    strict_validation: bool = True,
    collect_cuda_peak_memory: bool = True,
) -> dict[str, Any]:
    """Measure source-cache handoff plus eager mapper conversion, no prefill."""

    inference_mode_enabled = bool(torch.is_inference_mode_enabled())
    autograd_graph_disabled = not bool(torch.is_grad_enabled())
    if not inference_mode_enabled or not autograd_graph_disabled:
        raise RuntimeError("mapper latency must run with inference mode enabled")

    devices = tuple(key.device for key, _ in cache_to_legacy(source_cache)) + tuple(
        qwen3_layer_devices(target_model)
    )
    summary = benchmark_callable(
        lambda _: transfer_source_cache(
            source_cache,
            source_config=source_config,
            target_model=target_model,
            mapper_bundle=mapper_bundle,  # type: ignore[arg-type]
            output_dtype=output_dtype,
            token_chunk=token_chunk,
            strict_validation=strict_validation,
        ),
        prepare=lambda: None,
        warmup=warmup,
        repeats=repeats,
        devices=devices,
        collect_cuda_peak_memory=collect_cuda_peak_memory,
    )
    summary.update(
        {
            "measurement_scope": "source_cache_to_mapped_target_cache",
            "source_prefill_included": False,
            "mapper_weights_resident_on_target_devices": True,
            "includes_source_cache_device_handoff": True,
            "includes_inverse_and_target_rope": True,
            "includes_cache_build": True,
            "inference_mode_enabled": inference_mode_enabled,
            "autograd_graph_disabled": autograd_graph_disabled,
        }
    )
    return summary


@torch.inference_mode()
def benchmark_target_reprefill(
    target_model: torch.nn.Module,
    prefix_input_ids: torch.Tensor,
    *,
    warmup: int = 50,
    repeats: int = 30,
    collect_cuda_peak_memory: bool = True,
) -> dict[str, Any]:
    """Measure target transformer-body prefill with KV output, excluding LM head."""

    inference_mode_enabled = bool(torch.is_inference_mode_enabled())
    autograd_graph_disabled = not bool(torch.is_grad_enabled())
    if not inference_mode_enabled or not autograd_graph_disabled:
        raise RuntimeError("target re-prefill latency must run with inference mode enabled")

    if prefix_input_ids.ndim != 2 or prefix_input_ids.shape[1] <= 0:
        raise ValueError("prefix_input_ids must be [batch, nonempty_sequence]")
    base = getattr(target_model, "model", target_model)
    embedding = getattr(base, "embed_tokens", None)
    if embedding is None:
        raise ValueError("target model has no model.embed_tokens")
    ids = prefix_input_ids.to(embedding.weight.device)
    devices = qwen3_layer_devices(target_model)
    summary = benchmark_callable(
        lambda _: base(
            input_ids=ids,
            use_cache=True,
            return_dict=True,
        ),
        prepare=lambda: None,
        warmup=warmup,
        repeats=repeats,
        devices=devices,
        collect_cuda_peak_memory=collect_cuda_peak_memory,
    )
    summary.update(
        {
            "measurement_scope": "target_transformer_reprefill",
            "lm_head_included": False,
            "kv_cache_output": True,
            "input_length": int(ids.shape[1]),
            "batch_size": int(ids.shape[0]),
            "inference_mode_enabled": inference_mode_enabled,
            "autograd_graph_disabled": autograd_graph_disabled,
        }
    )
    return summary


def benchmark_boundary_decode(
    target_model: torch.nn.Module,
    boundary_input_ids: torch.Tensor,
    mapped_prefix_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    warmup: int = 50,
    repeats: int = 30,
    include_lm_head: bool = False,
    collect_cuda_peak_memory: bool = True,
) -> dict[str, Any]:
    """Measure one target boundary decode from a fresh shallow cache branch.

    ``DynamicCache.update`` replaces each branch entry with a concatenated
    tensor; it does not mutate the shared prefix tensor in-place.  A fresh
    container can therefore share the immutable mapped prefix and avoid an
    untimed 8-GiB deep copy at 32K.  The decode's required cache concatenation
    remains inside the timed operation and the parent cache stays unchanged.
    """

    if boundary_input_ids.ndim != 2 or boundary_input_ids.shape[1] != 1:
        raise ValueError("boundary_input_ids must be [batch,1]")
    target_devices = qwen3_layer_devices(target_model)

    def prepare() -> DynamicCache:
        return build_dynamic_cache(
            mapped_prefix_cache,
            model=target_model,
            clone=False,
            detach=True,
        )

    def decode(cache: DynamicCache) -> Any:
        if include_lm_head:
            return target_boundary_replay(
                target_model,
                boundary_input_ids,
                cache,
                clone_cache=False,
            )
        base = getattr(target_model, "model", target_model)
        embedding = getattr(base, "embed_tokens", None)
        if embedding is None:
            raise ValueError("target model has no model.embed_tokens")
        device = embedding.weight.device
        prefix_length = cache_sequence_length(cache)
        boundary = boundary_input_ids.to(device)
        return base(
            input_ids=boundary,
            past_key_values=cache,
            attention_mask=torch.ones(
                (boundary.shape[0], prefix_length + 1),
                device=device,
                dtype=torch.long,
            ),
            position_ids=torch.full(
                (boundary.shape[0], 1),
                prefix_length,
                device=device,
                dtype=torch.long,
            ),
            cache_position=torch.tensor([prefix_length], device=device, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )

    summary = benchmark_callable(
        decode,
        prepare=prepare,
        warmup=warmup,
        repeats=repeats,
        devices=target_devices,
        collect_cuda_peak_memory=collect_cuda_peak_memory,
    )
    summary.update(
        {
            "measurement_scope": "one_token_target_boundary_decode",
            "query_tokens": 1,
            "lm_head_included": include_lm_head,
            "cache_clone_included": False,
            "cache_branch_container_prepare_included": False,
            "prefix_tensor_copy_included": False,
            "prefix_tensors_shared_read_only": True,
            "decode_cache_concatenation_included": True,
            "prefix_length": cache_sequence_length(mapped_prefix_cache),
        }
    )
    return summary


def benchmark_protocol_complete(
    source_cache: Cache | Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    source_config: Any,
    target_model: torch.nn.Module,
    boundary_input_ids: torch.Tensor,
    mapper_bundle: ResidentMapperBundle,
    output_dtype: torch.dtype | None = None,
    token_chunk: int = 1024,
    warmup: int = 50,
    repeats: int = 30,
    include_lm_head: bool = False,
    strict_validation: bool = True,
    collect_cuda_peak_memory: bool = True,
) -> dict[str, Any]:
    """Measure mapper conversion followed immediately by one boundary decode."""

    source_devices = tuple(key.device for key, _ in cache_to_legacy(source_cache))
    target_devices = qwen3_layer_devices(target_model)

    def operation(_: None) -> Any:
        cache = transfer_source_cache(
            source_cache,
            source_config=source_config,
            target_model=target_model,
            mapper_bundle=mapper_bundle,  # type: ignore[arg-type]
            output_dtype=output_dtype,
            token_chunk=token_chunk,
            strict_validation=strict_validation,
        )
        if include_lm_head:
            return target_boundary_replay(
                target_model,
                boundary_input_ids,
                cache,
                clone_cache=False,
            )
        base = getattr(target_model, "model", target_model)
        device = base.embed_tokens.weight.device
        prefix_length = cache_sequence_length(cache)
        boundary = boundary_input_ids.to(device)
        return base(
            input_ids=boundary,
            past_key_values=cache,
            attention_mask=torch.ones(
                (boundary.shape[0], prefix_length + 1),
                device=device,
                dtype=torch.long,
            ),
            position_ids=torch.full(
                (boundary.shape[0], 1), prefix_length, device=device, dtype=torch.long
            ),
            cache_position=torch.tensor([prefix_length], device=device, dtype=torch.long),
            use_cache=True,
            return_dict=True,
        )

    summary = benchmark_callable(
        operation,
        prepare=lambda: None,
        warmup=warmup,
        repeats=repeats,
        devices=source_devices + tuple(target_devices),
        collect_cuda_peak_memory=collect_cuda_peak_memory,
    )
    summary.update(
        {
            "measurement_scope": "mapper_handoff_plus_one_token_boundary_decode",
            "source_prefill_included": False,
            "lm_head_included_for_boundary": include_lm_head,
        }
    )
    return summary


__all__ = [
    "COMPONENT_NAMES",
    "ResidentMapperBundle",
    "TimedValue",
    "benchmark_boundary_decode",
    "benchmark_callable",
    "benchmark_mapper_components",
    "benchmark_mapper_end_to_end",
    "benchmark_protocol_complete",
    "benchmark_target_reprefill",
    "configure_strict_latency_math",
    "materialize_mapper_on_target",
    "measure_synchronized",
    "percentile",
    "profile_transfer_once",
    "summarize_latency",
    "synchronize_devices",
]
