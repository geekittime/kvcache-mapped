"""Strict cache-composition and paired functional metrics for A1 oracles.

This module does not load a model and never initializes CUDA.  It defines the
runtime boundary that a later, separately authorized executor must satisfy:

1. tokenize one frozen prompt ``C``;
2. construct native-14B and mapped-32B->14B caches for ``C[:-1]``;
3. compose K and V according to one preregistered intervention;
4. replay the identical boundary token ``C[-1]`` through native 14B; and
5. compare that branch with native-K/native-V at the same causal boundary.

The cache contract deliberately names the RoPE space.  Mixing a content-space
key with a post-RoPE key is therefore rejected before a model forward.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import string
from typing import Any, Iterable, Mapping, Sequence

import torch


TARGET_LAYERS = 40
TARGET_KV_HEADS = 8
TARGET_QUERY_HEADS = 40
TARGET_HEAD_DIM = 128
ROPE_THETA = 1_000_000.0
KEY_SPACE = "post_target_qwen3_rope_cache"
VALUE_SPACE = "target_cache_value_space"
POSITION_SCHEME = "contiguous_zero_based_prefix"


class OracleContractError(RuntimeError):
    """Raised when an oracle input could make a comparison non-causal."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OracleContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise OracleContractError("token IDs must be non-negative integers")
        normalized.append(value)
    return canonical_sha256(normalized)


def positions_sha256(prefix_length: int) -> str:
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, int) or prefix_length < 1:
        raise OracleContractError("prefix length must be a positive integer")
    return token_ids_sha256(list(range(prefix_length)))


@dataclass(frozen=True)
class CacheContext:
    """Semantic identity shared by the native and mapped prefix caches."""

    schema_version: int
    dataset: str
    sample_id: str
    serialized_prompt_sha256: str
    prompt_token_ids_sha256: str
    prefix_token_ids_sha256: str
    boundary_token_id: int
    prefix_length: int
    batch_size: int
    position_ids_sha256: str
    target_config_sha256: str
    target_layers: int = TARGET_LAYERS
    target_kv_heads: int = TARGET_KV_HEADS
    target_query_heads: int = TARGET_QUERY_HEADS
    target_head_dim: int = TARGET_HEAD_DIM
    key_space: str = KEY_SPACE
    value_space: str = VALUE_SPACE
    position_scheme: str = POSITION_SCHEME
    rope_theta: float = ROPE_THETA
    rope_scaling: str = "none"
    sliding_window: str = "none"

    @classmethod
    def from_prompt(
        cls,
        *,
        dataset: str,
        sample_id: str,
        serialized_prompt: str,
        token_ids: Sequence[int],
        target_config_sha256: str,
    ) -> "CacheContext":
        ids = tuple(token_ids)
        if len(ids) < 2:
            raise OracleContractError("a cache oracle prompt needs at least two tokens")
        if not isinstance(serialized_prompt, str) or not serialized_prompt:
            raise OracleContractError("serialized prompt must be a non-empty string")
        if not isinstance(dataset, str) or not dataset or not isinstance(sample_id, str) or not sample_id:
            raise OracleContractError("dataset and sample ID must be non-empty strings")
        token_ids_sha256(ids)
        if not re.fullmatch(r"[0-9a-f]{64}", target_config_sha256):
            raise OracleContractError("target config SHA-256 is malformed")
        prefix = ids[:-1]
        return cls(
            schema_version=1,
            dataset=dataset,
            sample_id=sample_id,
            serialized_prompt_sha256=hashlib.sha256(serialized_prompt.encode("utf-8")).hexdigest(),
            prompt_token_ids_sha256=token_ids_sha256(ids),
            prefix_token_ids_sha256=token_ids_sha256(prefix),
            boundary_token_id=int(ids[-1]),
            prefix_length=len(prefix),
            batch_size=1,
            position_ids_sha256=positions_sha256(len(prefix)),
            target_config_sha256=target_config_sha256,
        )

    def validate(self) -> None:
        if self.schema_version != 1:
            raise OracleContractError("unsupported cache-context schema")
        if not self.dataset or not self.sample_id:
            raise OracleContractError("cache context has an empty dataset/sample identity")
        for name in (
            "serialized_prompt_sha256",
            "prompt_token_ids_sha256",
            "prefix_token_ids_sha256",
            "position_ids_sha256",
            "target_config_sha256",
        ):
            if re.fullmatch(r"[0-9a-f]{64}", str(getattr(self, name))) is None:
                raise OracleContractError(f"cache context {name} is malformed")
        if isinstance(self.boundary_token_id, bool) or self.boundary_token_id < 0:
            raise OracleContractError("boundary token ID is invalid")
        if self.prefix_length < 1 or self.batch_size != 1:
            raise OracleContractError("oracle caches require one non-empty unpadded sequence")
        if (
            self.target_layers,
            self.target_kv_heads,
            self.target_query_heads,
            self.target_head_dim,
        ) != (TARGET_LAYERS, TARGET_KV_HEADS, TARGET_QUERY_HEADS, TARGET_HEAD_DIM):
            raise OracleContractError("target cache geometry drifted from Qwen3-14B")
        if self.target_query_heads % self.target_kv_heads:
            raise OracleContractError("target query heads are not divisible by KV heads")
        if self.key_space != KEY_SPACE or self.value_space != VALUE_SPACE:
            raise OracleContractError("cache is not in target post-RoPE K / target V space")
        if self.position_scheme != POSITION_SCHEME:
            raise OracleContractError("unsupported cache position scheme")
        if not math.isclose(float(self.rope_theta), ROPE_THETA, rel_tol=0.0, abs_tol=0.0):
            raise OracleContractError("RoPE theta drift")
        if self.rope_scaling != "none" or self.sliding_window != "none":
            raise OracleContractError("scaled RoPE or sliding-window cache is unsupported")
        if self.position_ids_sha256 != positions_sha256(self.prefix_length):
            raise OracleContractError("position-ID commitment is not contiguous 0..L-2")


@dataclass(frozen=True)
class CacheProvenance:
    role: str
    producer: str
    producer_sha256: str
    mapper_manifest_sha256: str | None = None

    def validate(self) -> None:
        if self.role not in {"native", "mapped"}:
            raise OracleContractError("cache role must be native or mapped")
        if not self.producer:
            raise OracleContractError("cache producer must be named")
        if re.fullmatch(r"[0-9a-f]{64}", self.producer_sha256) is None:
            raise OracleContractError("cache producer SHA-256 is malformed")
        if self.role == "native" and self.mapper_manifest_sha256 is not None:
            raise OracleContractError("native cache may not claim mapper provenance")
        if self.role == "mapped":
            if self.mapper_manifest_sha256 is None or re.fullmatch(
                r"[0-9a-f]{64}", self.mapper_manifest_sha256
            ) is None:
                raise OracleContractError("mapped cache requires a mapper-manifest SHA-256")


LegacyCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass(frozen=True)
class CacheEnvelope:
    context: CacheContext
    provenance: CacheProvenance
    layers: LegacyCache


def _legacy_pairs(cache: Any) -> LegacyCache:
    if isinstance(cache, torch.Tensor):
        raise OracleContractError("cache must be per-layer K/V pairs, not a single tensor")
    try:
        pairs = tuple((pair[0], pair[1]) for pair in cache)
    except (TypeError, IndexError, KeyError) as exc:
        raise OracleContractError("cache is not iterable as per-layer K/V pairs") from exc
    if not pairs:
        raise OracleContractError("cache contains no layers")
    return pairs


def _validate_cache_layers(
    pairs: LegacyCache,
    context: CacheContext,
    *,
    finite_check: bool,
) -> None:
    if len(pairs) != context.target_layers:
        raise OracleContractError(
            f"cache has {len(pairs)} layers, expected {context.target_layers}"
        )
    expected = (
        context.batch_size,
        context.target_kv_heads,
        context.prefix_length,
        context.target_head_dim,
    )
    for layer_index, (key, value) in enumerate(pairs):
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise OracleContractError(f"cache layer {layer_index} is not a tensor pair")
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise OracleContractError(
                f"cache layer {layer_index} shape drift: K={tuple(key.shape)}, "
                f"V={tuple(value.shape)}, expected={expected}"
            )
        if not key.is_floating_point() or not value.is_floating_point():
            raise OracleContractError(f"cache layer {layer_index} is not floating point")
        if key.dtype != value.dtype:
            raise OracleContractError(f"cache layer {layer_index} K/V dtype mismatch")
        if key.device != value.device:
            raise OracleContractError(f"cache layer {layer_index} K/V device mismatch")
        if key.layout != torch.strided or value.layout != torch.strided:
            raise OracleContractError(f"cache layer {layer_index} must use strided tensors")
        if finite_check and (
            not bool(torch.isfinite(key).all()) or not bool(torch.isfinite(value).all())
        ):
            raise OracleContractError(f"cache layer {layer_index} contains NaN or Inf")


def wrap_cache(
    cache: Any,
    *,
    context: CacheContext,
    provenance: CacheProvenance,
    finite_check: bool = True,
) -> CacheEnvelope:
    """Validate and label a target-format prefix cache without copying it."""

    context.validate()
    provenance.validate()
    pairs = _legacy_pairs(cache)
    _validate_cache_layers(pairs, context, finite_check=finite_check)
    return CacheEnvelope(context=context, provenance=provenance, layers=pairs)


@dataclass(frozen=True)
class InterventionSpec:
    id: str
    default_k: str
    default_v: str
    native_kv_layers: tuple[int, ...] = ()

    def validate(self) -> None:
        if not self.id or self.default_k not in {"native", "mapped"} or self.default_v not in {
            "native",
            "mapped",
        }:
            raise OracleContractError("malformed intervention")
        if len(set(self.native_kv_layers)) != len(self.native_kv_layers):
            raise OracleContractError(f"intervention {self.id} repeats a layer")
        if tuple(sorted(self.native_kv_layers)) != self.native_kv_layers:
            raise OracleContractError(f"intervention {self.id} layers must be sorted")
        if any(layer < 0 or layer >= TARGET_LAYERS for layer in self.native_kv_layers):
            raise OracleContractError(f"intervention {self.id} contains an invalid layer")
        if self.native_kv_layers and (self.default_k, self.default_v) != ("mapped", "mapped"):
            raise OracleContractError("layer replacement must use mapped/mapped as its base")

    def sources_for_layer(self, layer_index: int) -> tuple[str, str]:
        self.validate()
        if layer_index < 0 or layer_index >= TARGET_LAYERS:
            raise OracleContractError("target layer is out of range")
        if layer_index in self.native_kv_layers:
            return "native", "native"
        return self.default_k, self.default_v

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["native_kv_layers"] = list(self.native_kv_layers)
        return value


def intervention_catalog() -> tuple[InterventionSpec, ...]:
    values = (
        InterventionSpec("native_k_native_v", "native", "native"),
        InterventionSpec("mapped_k_native_v", "mapped", "native"),
        InterventionSpec("native_k_mapped_v", "native", "mapped"),
        InterventionSpec("mapped_k_mapped_v", "mapped", "mapped"),
        InterventionSpec("native_kv_layers_0_9", "mapped", "mapped", tuple(range(0, 10))),
        InterventionSpec("native_kv_layers_10_19", "mapped", "mapped", tuple(range(10, 20))),
        InterventionSpec("native_kv_layers_20_29", "mapped", "mapped", tuple(range(20, 30))),
        InterventionSpec("native_kv_layers_30_39", "mapped", "mapped", tuple(range(30, 40))),
        InterventionSpec("native_kv_layer_34", "mapped", "mapped", (34,)),
        InterventionSpec("native_kv_layer_35", "mapped", "mapped", (35,)),
        InterventionSpec("native_kv_layers_34_35", "mapped", "mapped", (34, 35)),
    )
    if len({value.id for value in values}) != len(values):
        raise OracleContractError("intervention IDs are not unique")
    for value in values:
        value.validate()
    return values


def intervention_by_id(value: str) -> InterventionSpec:
    matches = [item for item in intervention_catalog() if item.id == value]
    if len(matches) != 1:
        raise OracleContractError(f"unknown intervention: {value!r}")
    return matches[0]


@dataclass(frozen=True)
class CompositionReceipt:
    schema_version: int
    intervention_id: str
    context_sha256: str
    mapper_manifest_sha256: str
    k_sources: tuple[str, ...]
    v_sources: tuple[str, ...]
    cloned: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["k_sources"] = list(self.k_sources)
        value["v_sources"] = list(self.v_sources)
        return value


def _validate_pair(native: CacheEnvelope, mapped: CacheEnvelope) -> None:
    native.context.validate()
    mapped.context.validate()
    native.provenance.validate()
    mapped.provenance.validate()
    if native.provenance.role != "native" or mapped.provenance.role != "mapped":
        raise OracleContractError("cache envelopes were passed in reversed roles")
    if native.context != mapped.context:
        raise OracleContractError("native and mapped caches do not describe the exact same prefix")
    _validate_cache_layers(native.layers, native.context, finite_check=True)
    _validate_cache_layers(mapped.layers, mapped.context, finite_check=True)
    for layer_index, ((native_k, native_v), (mapped_k, mapped_v)) in enumerate(
        zip(native.layers, mapped.layers, strict=True)
    ):
        for component, left, right in (
            ("K", native_k, mapped_k),
            ("V", native_v, mapped_v),
        ):
            if left.shape != right.shape:
                raise OracleContractError(f"layer {layer_index} native/mapped {component} shape mismatch")
            if left.dtype != right.dtype:
                raise OracleContractError(f"layer {layer_index} native/mapped {component} dtype mismatch")
            if left.device != right.device:
                raise OracleContractError(f"layer {layer_index} native/mapped {component} device mismatch")


def compose_legacy_cache(
    native: CacheEnvelope,
    mapped: CacheEnvelope,
    intervention: InterventionSpec | str,
    *,
    clone: bool = True,
) -> tuple[LegacyCache, CompositionReceipt]:
    """Select K/V sources and return an isolated legacy cache branch."""

    _validate_pair(native, mapped)
    spec = intervention_by_id(intervention) if isinstance(intervention, str) else intervention
    spec.validate()
    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    key_sources: list[str] = []
    value_sources: list[str] = []
    for layer_index in range(native.context.target_layers):
        key_source, value_source = spec.sources_for_layer(layer_index)
        key = (native if key_source == "native" else mapped).layers[layer_index][0]
        value = (native if value_source == "native" else mapped).layers[layer_index][1]
        key = key.detach()
        value = value.detach()
        if clone:
            key, value = key.clone(), value.clone()
        result.append((key, value))
        key_sources.append(key_source)
        value_sources.append(value_source)
    pairs = tuple(result)
    _validate_cache_layers(pairs, native.context, finite_check=True)
    receipt = CompositionReceipt(
        schema_version=1,
        intervention_id=spec.id,
        context_sha256=canonical_sha256(asdict(native.context)),
        mapper_manifest_sha256=str(mapped.provenance.mapper_manifest_sha256),
        k_sources=tuple(key_sources),
        v_sources=tuple(value_sources),
        cloned=bool(clone),
    )
    return pairs, receipt


def build_dynamic_cache(pairs: LegacyCache) -> Any:
    """Build a fresh Transformers DynamicCache, importing Transformers lazily."""

    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as exc:  # pragma: no cover - the production environment pins Transformers.
        raise OracleContractError("Transformers is required to materialize a DynamicCache") from exc
    result = DynamicCache()
    for layer_index, (key, value) in enumerate(pairs):
        result.update(key, value, layer_index)
    if len(result) != len(pairs):
        raise OracleContractError("DynamicCache materialization dropped a target layer")
    return result


def boundary_replay_inputs(
    *,
    context: CacheContext,
    boundary_input_ids: torch.Tensor,
    composed_cache: LegacyCache,
    clone_cache: bool = True,
) -> dict[str, Any]:
    """Create the exact one-token target replay inputs for ``C[-1]``.

    No padding is accepted in A1.  The caller must put the boundary token on
    the target embedding device; individual cache layers remain model-sharded.
    """

    context.validate()
    _validate_cache_layers(composed_cache, context, finite_check=True)
    if not isinstance(boundary_input_ids, torch.Tensor) or boundary_input_ids.shape != (
        context.batch_size,
        1,
    ):
        raise OracleContractError("boundary_input_ids must have shape [1,1]")
    if boundary_input_ids.dtype not in (torch.int32, torch.int64):
        raise OracleContractError("boundary token must use an integer dtype")
    if int(boundary_input_ids.detach().to(device="cpu").item()) != context.boundary_token_id:
        raise OracleContractError("boundary token differs from the frozen prompt")
    if clone_cache:
        pairs = tuple((key.detach().clone(), value.detach().clone()) for key, value in composed_cache)
    else:
        pairs = composed_cache
    attention_mask = torch.ones(
        (context.batch_size, context.prefix_length + 1),
        dtype=torch.long,
        device=boundary_input_ids.device,
    )
    position_ids = torch.full(
        (context.batch_size, 1),
        context.prefix_length,
        dtype=torch.long,
        device=boundary_input_ids.device,
    )
    cache_position = torch.tensor(
        [context.prefix_length], dtype=torch.long, device=boundary_input_ids.device
    )
    return {
        "input_ids": boundary_input_ids,
        "past_key_values": build_dynamic_cache(pairs),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "use_cache": True,
        "return_dict": True,
    }


def validate_qwen3_config(config: Mapping[str, Any] | Any, *, source: bool) -> None:
    def read(name: str) -> Any:
        return config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)

    expected_layers = 64 if source else TARGET_LAYERS
    expected_query_heads = 64 if source else TARGET_QUERY_HEADS
    observed = (
        read("model_type"),
        read("num_hidden_layers"),
        read("num_attention_heads"),
        read("num_key_value_heads"),
        read("head_dim"),
    )
    expected = ("qwen3", expected_layers, expected_query_heads, TARGET_KV_HEADS, TARGET_HEAD_DIM)
    if observed != expected:
        raise OracleContractError(f"Qwen3 model geometry mismatch: {observed} != {expected}")
    if float(read("rope_theta")) != ROPE_THETA or read("rope_scaling") not in (None, {}):
        raise OracleContractError("Qwen3 model uses an unsupported RoPE convention")
    if read("sliding_window") is not None:
        raise OracleContractError("sliding-window cache is unsupported")
    if int(read("max_position_embeddings")) < 40_960:
        raise OracleContractError("model context limit is below the frozen protocol")


@dataclass(frozen=True)
class AttentionReadout:
    output: torch.Tensor
    probabilities: torch.Tensor
    query_sha256_fp32: str
    query_absolute_positions: tuple[int, ...]
    query_policy: str = "fixed_native14_boundary_query"
    boundary_kv_policy: str = "selected_prefix_plus_fixed_native14_boundary_kv"
    output_space: str = "pre_o_proj_per_query_head"
    kernel: str = "explicit_causal_gqa_fp32"


def _tensor_sha256_fp32(value: torch.Tensor) -> str:
    cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def attention_readout_fp32(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_absolute_positions: Sequence[int],
    key_valid_mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> AttentionReadout:
    """Explicit causal GQA readout before ``o_proj`` using FP32 softmax.

    Q must already be target post-RoPE.  K must be target post-RoPE and V must
    be in target cache value space.  For the boundary diagnostic, pass the
    one-token boundary Q and full prefix-plus-boundary K/V with absolute query
    position equal to ``prefix_length``.
    """

    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4 or not tensor.is_floating_point():
            raise OracleContractError(f"{name} must be a floating [B,H,T,D] tensor")
    if key.shape != value.shape:
        raise OracleContractError("attention K/V shapes differ")
    batch, query_heads, query_length, head_dim = query.shape
    key_batch, kv_heads, key_length, key_head_dim = key.shape
    if (batch, query_heads, head_dim) != (key_batch, TARGET_QUERY_HEADS, key_head_dim):
        raise OracleContractError("attention query geometry is not target Qwen3-14B")
    if kv_heads != TARGET_KV_HEADS or head_dim != TARGET_HEAD_DIM:
        raise OracleContractError("attention KV geometry is not target Qwen3-14B")
    if query.device != key.device or key.device != value.device:
        raise OracleContractError("attention Q/K/V must share one device")
    if query.dtype != key.dtype or key.dtype != value.dtype:
        raise OracleContractError("attention Q/K/V must share one dtype")
    if query_length < 1 or key_length < 1 or query_heads % kv_heads:
        raise OracleContractError("invalid attention lengths or GQA grouping")
    positions = torch.as_tensor(query_absolute_positions, dtype=torch.long, device=query.device)
    if positions.ndim != 1 or positions.numel() != query_length:
        raise OracleContractError("one absolute position is required for every query row")
    if int(torch.unique(positions).numel()) != query_length:
        raise OracleContractError("query positions must not repeat")
    if bool(torch.any(positions < 0)) or bool(torch.any(positions >= key_length)):
        raise OracleContractError("query absolute positions are outside the supplied cache")
    if key_valid_mask is None:
        valid = torch.ones((batch, key_length), dtype=torch.bool, device=query.device)
    else:
        if not isinstance(key_valid_mask, torch.Tensor) or key_valid_mask.shape != (
            batch,
            key_length,
        ):
            raise OracleContractError("key_valid_mask shape mismatch")
        if key_valid_mask.dtype != torch.bool and not bool(
            torch.all((key_valid_mask == 0) | (key_valid_mask == 1))
        ):
            raise OracleContractError("key_valid_mask must be Boolean or binary")
        valid = key_valid_mask.to(device=query.device, dtype=torch.bool)
    scale = 1.0 / math.sqrt(head_dim) if scaling is None else float(scaling)
    if not math.isfinite(scale) or scale <= 0:
        raise OracleContractError("attention scaling must be finite and positive")
    groups = query_heads // kv_heads
    q = query.to(torch.float32).reshape(batch, kv_heads, groups, query_length, head_dim)
    k = key.to(torch.float32)
    v = value.to(torch.float32)
    scores = torch.einsum("bkgqd,bkmd->bkgqm", q, k) * scale
    key_positions = torch.arange(key_length, device=query.device)
    causal = key_positions.unsqueeze(0) <= positions.unsqueeze(1)
    allowed = valid[:, None, None, None, :] & causal[None, None, None, :, :]
    if bool(torch.any(~allowed.any(dim=-1))):
        raise OracleContractError("an attention query has no visible key")
    scores.masked_fill_(~allowed, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.einsum("bkgqm,bkmd->bkgqd", probabilities, v).reshape(
        batch, query_heads, query_length, head_dim
    )
    probabilities = probabilities.reshape(batch, query_heads, query_length, key_length)
    if not bool(torch.isfinite(output).all()) or not bool(torch.isfinite(probabilities).all()):
        raise OracleContractError("attention readout contains NaN or Inf")
    return AttentionReadout(
        output=output,
        probabilities=probabilities,
        query_sha256_fp32=_tensor_sha256_fp32(query),
        query_absolute_positions=tuple(int(value) for value in positions.detach().cpu().tolist()),
    )


def _rowwise_cosine(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if reference.shape != candidate.shape or reference.ndim != 4:
        raise OracleContractError("attention output shapes differ")
    left = reference.detach().to(device="cpu", dtype=torch.float64)
    right = candidate.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise OracleContractError("attention output contains NaN or Inf")
    numerator = (left * right).sum(dim=-1)
    left_norm = torch.linalg.vector_norm(left, dim=-1)
    right_norm = torch.linalg.vector_norm(right, dim=-1)
    denominator = left_norm * right_norm
    result = torch.zeros_like(numerator)
    nonzero = denominator > 0
    result[nonzero] = numerator[nonzero] / denominator[nonzero]
    result[(left_norm == 0) & (right_norm == 0)] = 1.0
    return result.clamp(-1.0, 1.0)


def attention_pair_metrics(
    reference: AttentionReadout,
    candidate: AttentionReadout,
) -> dict[str, float]:
    expected_policy = (
        "fixed_native14_boundary_query",
        "selected_prefix_plus_fixed_native14_boundary_kv",
        "pre_o_proj_per_query_head",
        "explicit_causal_gqa_fp32",
    )
    if (
        reference.query_policy,
        reference.boundary_kv_policy,
        reference.output_space,
        reference.kernel,
    ) != expected_policy or (
        candidate.query_policy,
        candidate.boundary_kv_policy,
        candidate.output_space,
        candidate.kernel,
    ) != expected_policy:
        raise OracleContractError("attention readout provenance is not the frozen fixed-Q protocol")
    if (
        reference.query_sha256_fp32 != candidate.query_sha256_fp32
        or reference.query_absolute_positions != candidate.query_absolute_positions
    ):
        raise OracleContractError("paired attention readouts did not use the identical native query")
    if re.fullmatch(r"[0-9a-f]{64}", reference.query_sha256_fp32) is None:
        raise OracleContractError("attention query commitment is malformed")
    if reference.output.shape != candidate.output.shape:
        raise OracleContractError("attention outputs are not paired")
    if reference.probabilities.shape != candidate.probabilities.shape:
        raise OracleContractError("attention probabilities are not paired")
    cosine = _rowwise_cosine(reference.output, candidate.output)
    left = reference.output.detach().to(device="cpu", dtype=torch.float64)
    right = candidate.output.detach().to(device="cpu", dtype=torch.float64)
    mse = (left - right).square().mean(dim=-1)
    reference_energy = left.square().mean(dim=-1)
    normalized_mse = torch.where(reference_energy > 0, mse / reference_energy, mse)
    p = reference.probabilities.detach().to(device="cpu", dtype=torch.float64)
    q = candidate.probabilities.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(q).all()):
        raise OracleContractError("attention probabilities contain NaN or Inf")
    mixture = 0.5 * (p + q)
    p_term = torch.where(p > 0, p * (torch.log(p) - torch.log(mixture)), torch.zeros_like(p))
    q_term = torch.where(q > 0, q * (torch.log(q) - torch.log(mixture)), torch.zeros_like(q))
    js = 0.5 * (p_term.sum(dim=-1) + q_term.sum(dim=-1))
    return {
        "attention_output_cosine": float(cosine.mean().item()),
        "attention_output_normalized_mse": float(normalized_mse.mean().item()),
        "attention_probability_js": float(js.mean().item()),
    }


def _paired_logits(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise OracleContractError("logits must be tensors")
    if reference.shape != candidate.shape or reference.ndim not in (1, 2):
        raise OracleContractError("paired first-token logits must have equal [V] or [B,V] shapes")
    if reference.ndim == 1:
        reference, candidate = reference.unsqueeze(0), candidate.unsqueeze(0)
    if reference.shape[0] != 1 or reference.shape[1] < 2:
        raise OracleContractError("A1 first-token metrics require batch one and a non-trivial vocabulary")
    reference = reference.detach().to(device="cpu", dtype=torch.float64)
    candidate = candidate.detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(reference).all()) or not bool(torch.isfinite(candidate).all()):
        raise OracleContractError("first-token logits contain NaN or Inf")
    return reference, candidate


def answer_margin(logits: torch.Tensor, gold_first_token_ids: Sequence[int]) -> float:
    values = logits.detach().to(device="cpu", dtype=torch.float64)
    if values.ndim == 2:
        if values.shape[0] != 1:
            raise OracleContractError("answer margin requires batch one")
        values = values[0]
    if values.ndim != 1 or values.numel() < 2 or not bool(torch.isfinite(values).all()):
        raise OracleContractError("answer margin requires finite vocabulary logits")
    gold = sorted(set(gold_first_token_ids))
    if not gold or any(isinstance(item, bool) or not isinstance(item, int) for item in gold):
        raise OracleContractError("gold first-token IDs must be non-empty integers")
    if gold[0] < 0 or gold[-1] >= values.numel() or len(gold) == values.numel():
        raise OracleContractError("gold first-token IDs are outside the vocabulary")
    gold_index = torch.tensor(gold, dtype=torch.long)
    gold_best = values.index_select(0, gold_index).max()
    non_gold = torch.ones(values.numel(), dtype=torch.bool)
    non_gold[gold_index] = False
    distractor_best = values[non_gold].max()
    return float((gold_best - distractor_best).item())


def first_token_pair_metrics(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    gold_first_token_ids: Sequence[int],
) -> dict[str, float | int | bool]:
    reference, candidate = _paired_logits(reference_logits, candidate_logits)
    reference_logp = torch.log_softmax(reference, dim=-1)
    candidate_logp = torch.log_softmax(candidate, dim=-1)
    reference_p = reference_logp.exp()
    kl = (reference_p * (reference_logp - candidate_logp)).sum(dim=-1)
    reference_margin = answer_margin(reference, gold_first_token_ids)
    candidate_margin = answer_margin(candidate, gold_first_token_ids)
    reference_decision = int(reference.argmax(dim=-1).item())
    candidate_decision = int(candidate.argmax(dim=-1).item())
    return {
        "first_token_kl_native_to_candidate": float(kl.item()),
        "native_answer_margin": reference_margin,
        "candidate_answer_margin": candidate_margin,
        "answer_margin_delta_candidate_minus_native": candidate_margin - reference_margin,
        "native_first_token_id": reference_decision,
        "candidate_first_token_id": candidate_decision,
        "first_token_agreement": reference_decision == candidate_decision,
    }


def normalize_answer(value: str) -> str:
    lowered = str(value).lower()
    without_punctuation = "".join(
        character for character in lowered if character not in set(string.punctuation)
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation, flags=re.UNICODE)
    return " ".join(without_articles.split())


def answer_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted or not gold:
        return float(predicted == gold)
    common = sum((Counter(predicted) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision = common / len(predicted)
    recall = common / len(gold)
    return 2.0 * precision * recall / (precision + recall)


def generation_score(prediction: str, references: Sequence[str]) -> dict[str, float]:
    if not isinstance(prediction, str) or not references or not all(
        isinstance(value, str) and value for value in references
    ):
        raise OracleContractError("generation score requires text and non-empty answer references")
    return {
        "answer_em": float(
            max(normalize_answer(prediction) == normalize_answer(reference) for reference in references)
        ),
        "answer_f1": max(answer_f1(prediction, reference) for reference in references),
    }


def generation_pair_metrics(
    native_prediction: str,
    candidate_prediction: str,
    references: Sequence[str],
) -> dict[str, float | bool]:
    native = generation_score(native_prediction, references)
    candidate = generation_score(candidate_prediction, references)
    return {
        "native_answer_em": native["answer_em"],
        "candidate_answer_em": candidate["answer_em"],
        "answer_em_delta_candidate_minus_native": candidate["answer_em"] - native["answer_em"],
        "native_answer_f1": native["answer_f1"],
        "candidate_answer_f1": candidate["answer_f1"],
        "answer_f1_delta_candidate_minus_native": candidate["answer_f1"] - native["answer_f1"],
        "normalized_answer_agreement": normalize_answer(native_prediction)
        == normalize_answer(candidate_prediction),
    }


def aggregate_attention_layers(
    native: Mapping[int, AttentionReadout],
    candidate: Mapping[int, AttentionReadout],
) -> dict[str, Any]:
    expected = set(range(TARGET_LAYERS))
    if set(native) != expected or set(candidate) != expected:
        raise OracleContractError("attention observations must cover exactly target layers 0..39")
    layers: list[dict[str, Any]] = []
    for layer_index in range(TARGET_LAYERS):
        layers.append({"layer": layer_index, **attention_pair_metrics(native[layer_index], candidate[layer_index])})
    return {
        "attention_layers": layers,
        "attention_output_cosine_uniform_layer_mean": sum(
            row["attention_output_cosine"] for row in layers
        )
        / TARGET_LAYERS,
        "attention_output_normalized_mse_uniform_layer_mean": sum(
            row["attention_output_normalized_mse"] for row in layers
        )
        / TARGET_LAYERS,
        "attention_probability_js_uniform_layer_mean": sum(
            row["attention_probability_js"] for row in layers
        )
        / TARGET_LAYERS,
    }


def _validate_boundary_attention_observation(
    context: CacheContext,
    observation: Mapping[int, AttentionReadout],
    *,
    label: str,
) -> None:
    if set(observation) != set(range(TARGET_LAYERS)):
        raise OracleContractError(f"{label} attention must cover exactly layers 0..39")
    expected_output = (
        context.batch_size,
        context.target_query_heads,
        1,
        context.target_head_dim,
    )
    expected_probabilities = (
        context.batch_size,
        context.target_query_heads,
        1,
        context.prefix_length + 1,
    )
    for layer_index in range(TARGET_LAYERS):
        readout = observation[layer_index]
        if not isinstance(readout, AttentionReadout):
            raise OracleContractError(f"{label} layer {layer_index} is not an AttentionReadout")
        if readout.query_absolute_positions != (context.prefix_length,):
            raise OracleContractError(
                f"{label} layer {layer_index} is not the frozen boundary query position"
            )
        if tuple(readout.output.shape) != expected_output:
            raise OracleContractError(f"{label} layer {layer_index} attention-output shape drift")
        if tuple(readout.probabilities.shape) != expected_probabilities:
            raise OracleContractError(
                f"{label} layer {layer_index} attention-probability shape drift"
            )


@dataclass(frozen=True)
class OracleObservation:
    first_token_logits: torch.Tensor
    generated_text: str
    generated_token_count: int
    stop_reason: str
    attention: Mapping[int, AttentionReadout]


def score_paired_observations(
    *,
    context: CacheContext,
    intervention: InterventionSpec | str,
    references: Sequence[str],
    gold_first_token_ids: Sequence[int],
    native: OracleObservation,
    candidate: OracleObservation,
) -> dict[str, Any]:
    """Return one complete paired record; all required metrics fail closed."""

    context.validate()
    spec = intervention_by_id(intervention) if isinstance(intervention, str) else intervention
    spec.validate()
    if native.generated_token_count < 0 or candidate.generated_token_count < 0:
        raise OracleContractError("generated-token counts must be non-negative")
    if native.generated_token_count > 256 or candidate.generated_token_count > 256:
        raise OracleContractError("generation exceeded the frozen cap of 256 tokens")
    valid_stop_reasons = {"eos", "cap"}
    if native.stop_reason not in valid_stop_reasons or candidate.stop_reason not in valid_stop_reasons:
        raise OracleContractError("stop reason must be eos or cap")
    _validate_boundary_attention_observation(context, native.attention, label="native")
    _validate_boundary_attention_observation(context, candidate.attention, label="candidate")
    return {
        "schema_version": 1,
        "dataset": context.dataset,
        "sample_id": context.sample_id,
        "intervention_id": spec.id,
        "context_sha256": canonical_sha256(asdict(context)),
        **first_token_pair_metrics(
            native.first_token_logits,
            candidate.first_token_logits,
            gold_first_token_ids=gold_first_token_ids,
        ),
        **generation_pair_metrics(native.generated_text, candidate.generated_text, references),
        **aggregate_attention_layers(native.attention, candidate.attention),
        "native_generated_tokens": native.generated_token_count,
        "candidate_generated_tokens": candidate.generated_token_count,
        "native_stop_reason": native.stop_reason,
        "candidate_stop_reason": candidate.stop_reason,
        "native_cap": native.stop_reason == "cap",
        "candidate_cap": candidate.stop_reason == "cap",
    }


__all__ = [
    "AttentionReadout",
    "CacheContext",
    "CacheEnvelope",
    "CacheProvenance",
    "CompositionReceipt",
    "InterventionSpec",
    "OracleContractError",
    "OracleObservation",
    "aggregate_attention_layers",
    "answer_f1",
    "answer_margin",
    "attention_pair_metrics",
    "attention_readout_fp32",
    "boundary_replay_inputs",
    "build_dynamic_cache",
    "canonical_sha256",
    "compose_legacy_cache",
    "first_token_pair_metrics",
    "generation_pair_metrics",
    "generation_score",
    "intervention_by_id",
    "intervention_catalog",
    "normalize_answer",
    "positions_sha256",
    "score_paired_observations",
    "token_ids_sha256",
    "validate_qwen3_config",
    "wrap_cache",
]
