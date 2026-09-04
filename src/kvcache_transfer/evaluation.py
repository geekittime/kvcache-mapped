"""Completion-mode evaluation for transferred Qwen3 KV caches.

The central invariant in this module is the causal boundary protocol.  For a
context ``C`` and continuation ``Y`` we cache ``C[:-1]`` and query the target
with ``C[-1] + Y[:-1]``.  The resulting logits score every token in ``Y``,
including its first token.  Prefix caches are immutable parents: every score
or generation request receives a deep cache branch before target execution.

The implementation deliberately supports batch size one only.  Padding and
batched candidate caches introduce position/mask ambiguity and are outside the
paper reproduction protocol.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Protocol

import torch

from .cache_adapter import (
    boundary_replay_inputs,
    cache_sequence_length,
    continuation_scoring_inputs,
    full_attention_mask,
    qwen3_layer_devices,
    replay_cache_position,
    replay_position_ids,
)
from .latency import ResidentMapperBundle, materialize_mapper_on_target
from .transfer import MapperBundle, transfer_source_cache


EVALUATION_MODES = (
    "source_standalone",
    "target_standalone",
    "target_oracle",
    "mapped",
)
STANDALONE_MODES = ("source_standalone", "target_standalone")


class PrefixCacheBuilder(Protocol):
    """Build an immutable batch-one prefix cache from token IDs."""

    def __call__(self, prefix_token_ids: Sequence[int]) -> Any: ...


@dataclass(frozen=True)
class ContinuationScore:
    loglikelihood: float
    is_greedy: bool
    token_count: int


@dataclass(frozen=True)
class GenerationResult:
    """Auditable result of one deterministic generation request.

    ``generate_text`` intentionally keeps its historical string-only API for
    the frozen paper evaluation.  New long-context evaluations use this richer
    result so a token-cap stop cannot be mistaken for a model EOS.
    """

    text: str
    generated_token_ids: tuple[int, ...]
    stop_reason: str
    cap_hit: bool
    requested_max_tokens: int
    original_context_tokens: int
    used_context_tokens: int
    truncation: str

    @property
    def generated_token_count(self) -> int:
        return len(self.generated_token_ids)


@dataclass(frozen=True)
class PrefixPerplexityResult:
    loglikelihood: float
    scored_tokens: int
    chunks: int
    perplexity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _input_device(model: torch.nn.Module) -> torch.device:
    embeddings = model.get_input_embeddings()
    try:
        device = next(embeddings.parameters()).device
    except StopIteration as exc:  # pragma: no cover - malformed model
        raise ValueError("target input embedding has no parameters") from exc
    if device.type == "meta":
        raise ValueError("target input embedding is still on the meta device")
    return device


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("evaluation tokenizer unexpectedly returned a batch")
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _tokenizer_decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(list(token_ids))


def _model_max_length(model: torch.nn.Module) -> int:
    config = model.config
    for name in ("max_position_embeddings", "n_positions", "n_ctx"):
        value = getattr(config, name, None)
        if isinstance(value, int) and 1 < value < 10**9:
            return value
    return 2048


def _prefix_token_id(tokenizer: Any) -> int:
    for name in ("bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, name, None)
        if value is not None:
            return int(value)
    raise ValueError("tokenizer must define bos_token_id or eos_token_id")


def head_tail_truncate_token_ids(
    token_ids: Sequence[int], budget: int
) -> tuple[list[int], dict[str, int | str | bool]]:
    """Keep both prompt instructions and the final query under a token budget.

    LongBench recommends removing the middle of over-length prompts because
    task instructions commonly occur at the beginning and the query at the
    end.  The split is deterministic: the odd token, if any, is assigned to
    the head.  Token IDs are never decode/re-encoded.
    """

    if budget <= 0:
        raise ValueError("token budget must be positive")
    values = [int(token) for token in token_ids]
    original = len(values)
    if original <= budget:
        return values, {
            "policy": "head_tail",
            "truncated": False,
            "original_tokens": original,
            "used_tokens": original,
            "removed_tokens": 0,
            "head_tokens": original,
            "tail_tokens": 0,
        }
    head = (budget + 1) // 2
    tail = budget - head
    truncated = values[:head] + (values[-tail:] if tail else [])
    return truncated, {
        "policy": "head_tail",
        "truncated": True,
        "original_tokens": original,
        "used_tokens": len(truncated),
        "removed_tokens": original - len(truncated),
        "head_tokens": head,
        "tail_tokens": tail,
    }


def encode_context_continuation(
    tokenizer: Any,
    context: str,
    continuation: str,
    *,
    prefix_token_id: int,
) -> tuple[list[int], list[int]]:
    """Match lm-eval's causal context/continuation boundary tokenization."""

    if not isinstance(context, str) or not isinstance(continuation, str):
        raise TypeError("context and continuation must be strings")
    if continuation == "":
        raise ValueError("continuation must not be empty")

    if context == "":
        continuation_ids = _tokenizer_encode(tokenizer, continuation)
        if not continuation_ids:
            raise ValueError("continuation tokenized to zero tokens")
        if continuation_ids[0] == prefix_token_id:
            if len(continuation_ids) == 1:
                raise ValueError("continuation contains only the prefix token")
            return continuation_ids[:1], continuation_ids[1:]
        return [prefix_token_id], continuation_ids

    # This is TemplateLM._encode_pair's trailing-space rule.  Encoding the
    # concatenated string is necessary for tokenizers whose BPE boundary
    # changes when the continuation begins.
    trailing_spaces = len(context) - len(context.rstrip())
    if trailing_spaces:
        continuation = context[-trailing_spaces:] + continuation
        context = context[:-trailing_spaces]
    whole_ids = _tokenizer_encode(tokenizer, context + continuation)
    context_ids = _tokenizer_encode(tokenizer, context)
    continuation_ids = whole_ids[len(context_ids) :]
    if not context_ids:
        context_ids = [prefix_token_id]
        continuation_ids = whole_ids
    if not continuation_ids:
        raise ValueError("continuation tokenized to zero tokens at the context boundary")
    return context_ids, continuation_ids


def rolling_token_windows(
    token_ids: Sequence[int],
    *,
    prefix_token_id: int,
    max_length: int,
) -> Iterable[tuple[list[int], list[int]]]:
    """Yield lm-eval-compatible disjoint windows with every token scored once."""

    tokens = list(map(int, token_ids))
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if not tokens:
        return
    # This is get_rolling_token_windows(context_len=1), followed by
    # make_disjoint_window.  Keeping it local makes the core usable before the
    # separately vendored harness has been installed into the environment.
    first_length = min(max_length, len(tokens))
    yield [prefix_token_id], tokens[:first_length]
    predicted = first_length
    prediction_window = max_length
    while predicted < len(tokens):
        prediction_length = min(len(tokens) - predicted, prediction_window)
        window_end = predicted + prediction_length
        overlapping_input = tokens[
            window_end - max_length - 1 : window_end - 1
        ]
        prediction = tokens[window_end - prediction_length : window_end]
        context = overlapping_input[: len(overlapping_input) - (len(prediction) - 1)]
        if not context:
            raise AssertionError("rolling construction produced an empty context")
        yield context, prediction
        predicted += prediction_length


class MappedPrefixCacheBuilder:
    """Map one source prefill cache into a complete target ``DynamicCache``.

    Mapper tensors are validated and materialized exactly once during
    construction, then remain FP32-resident beside their target decoder
    layers for every benchmark request.  The one-time host load/hash and
    device-transfer setup is explicitly reported in :attr:`provenance` and is
    not part of per-request mapping.
    """

    def __init__(
        self,
        *,
        source_model: torch.nn.Module,
        target_model: torch.nn.Module,
        mapper_manifest: str | Path,
        token_chunk: int = 1024,
    ) -> None:
        self.source_model = source_model
        self.target_model = target_model
        self.token_chunk = int(token_chunk)
        if self.token_chunk <= 0:
            raise ValueError("token_chunk must be positive")
        manifest_path = Path(mapper_manifest).resolve()
        mapper_root = manifest_path if manifest_path.is_dir() else manifest_path.parent
        lazy_bundle = MapperBundle.open(mapper_root)
        self.manifest_path = lazy_bundle.manifest_path

        source_observed = (
            int(source_model.config.num_hidden_layers),
            int(source_model.config.num_key_value_heads),
            int(source_model.config.head_dim),
        )
        target_observed = (
            int(target_model.config.num_hidden_layers),
            int(target_model.config.num_key_value_heads),
            int(target_model.config.head_dim),
        )
        architecture = lazy_bundle.architecture
        if source_observed != (
            architecture.source_layers,
            architecture.source_kv_heads,
            architecture.source_head_dim,
        ):
            raise ValueError(f"source model architecture does not match mapper: {source_observed}")
        if target_observed != (
            architecture.target_layers,
            architecture.target_kv_heads,
            architecture.target_head_dim,
        ):
            raise ValueError(f"target model architecture does not match mapper: {target_observed}")

        target_layer_devices = qwen3_layer_devices(target_model)
        resident, materialization = materialize_mapper_on_target(
            lazy_bundle,
            target_layer_devices,
        )
        self.mapper_bundle: ResidentMapperBundle = resident
        self.materialization = materialization
        self.target_layer_devices = tuple(str(device) for device in target_layer_devices)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "manifest_sha256": self.mapper_bundle.manifest_sha256,
            "fit_fingerprint": self.mapper_bundle.fit_fingerprint,
            "layers": len(self.mapper_bundle.layers),
            "token_chunk": self.token_chunk,
            "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
            "target_layer_devices": list(self.target_layer_devices),
            "resident_parameter_bytes": self.mapper_bundle.resident_parameter_bytes,
            "materialization": self.materialization,
        }

    @torch.inference_mode()
    def __call__(self, prefix_token_ids: Sequence[int]) -> Any:
        prefix = [int(token) for token in prefix_token_ids]
        if not prefix:
            raise ValueError("mapped prefix must contain at least one token")
        source_base = getattr(self.source_model, "model", self.source_model)
        source_device = _input_device(self.source_model)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=source_device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(len(prefix), device=source_device).unsqueeze(0)
        source_output = source_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=torch.arange(len(prefix), device=source_device),
            use_cache=True,
            return_dict=True,
        )
        cache = transfer_source_cache(
            source_output.past_key_values,
            source_config=self.source_model.config,
            target_model=self.target_model,
            mapper_bundle=self.mapper_bundle,
            position_ids=position_ids,
            # Infer the cache dtype from the target decoder layers.  Formal
            # Qwen3-32B is BF16; tiny FP32 integration models remain valid.
            output_dtype=None,
            token_chunk=self.token_chunk,
        )
        if cache_sequence_length(cache) != len(prefix):
            raise AssertionError("mapped target cache has an incorrect sequence length")
        return cache


class TransferEvaluationEngine:
    """Batch-one scorer/generator for one of the three evaluation paths."""

    def __init__(
        self,
        *,
        target_model: torch.nn.Module,
        tokenizer: Any,
        mode: str,
        mapped_prefix_builder: PrefixCacheBuilder | None = None,
        max_length: int | None = None,
        max_gen_tokens: int = 256,
        prefix_token_id: int | None = None,
    ) -> None:
        if mode not in EVALUATION_MODES:
            raise ValueError(f"mode must be one of {EVALUATION_MODES}, got {mode!r}")
        if mode == "mapped" and mapped_prefix_builder is None:
            raise ValueError("mapped mode requires mapped_prefix_builder")
        if mode != "mapped" and mapped_prefix_builder is not None:
            raise ValueError("mapped_prefix_builder is valid only in mapped mode")
        if target_model.training:
            raise ValueError("target model must be in eval mode")
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.mode = mode
        self.mapped_prefix_builder = mapped_prefix_builder
        self.max_length = int(max_length or _model_max_length(target_model))
        self.max_gen_tokens = int(max_gen_tokens)
        self.prefix_token_id = int(
            _prefix_token_id(tokenizer) if prefix_token_id is None else prefix_token_id
        )
        if self.max_length <= 1 or self.max_gen_tokens <= 0:
            raise ValueError("max_length must exceed one and max_gen_tokens must be positive")
        self.batch_size = 1
        self.device = _input_device(target_model)
        self._memo_key: tuple[str, tuple[int, ...]] | None = None
        self._memo_cache: Any = None

    def clear_prefix_cache(self) -> None:
        self._memo_key = None
        self._memo_cache = None

    def tok_encode(self, text: str) -> list[int]:
        return _tokenizer_encode(self.tokenizer, text)

    def _trim_score_pair(
        self, context_ids: Sequence[int], continuation_ids: Sequence[int]
    ) -> tuple[list[int], list[int]]:
        context = list(map(int, context_ids))
        continuation = list(map(int, continuation_ids))
        if not context or not continuation:
            raise ValueError("scoring requires non-empty context and continuation token IDs")
        if len(continuation) > self.max_length:
            raise ValueError(
                f"continuation has {len(continuation)} tokens, exceeding max_length={self.max_length}"
            )
        # C + Y[:-1] is the model input and must fit max_length.
        context_budget = self.max_length - len(continuation) + 1
        context = context[-context_budget:]
        if not context:
            raise AssertionError("score truncation removed the entire context")
        return context, continuation

    @torch.inference_mode()
    def _target_prefix_cache(self, prefix: Sequence[int]) -> Any:
        ids = torch.tensor([list(prefix)], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(ids)
        output = self.target_model(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=torch.arange(ids.shape[1], device=self.device).unsqueeze(0),
            cache_position=torch.arange(ids.shape[1], device=self.device),
            use_cache=True,
            return_dict=True,
        )
        return output.past_key_values

    def _parent_prefix_cache(self, prefix: Sequence[int]) -> Any:
        if not prefix:
            raise ValueError("a reusable prefix cache cannot be empty")
        key = (self.mode, tuple(map(int, prefix)))
        if key == self._memo_key:
            return self._memo_cache
        if self.mode == "target_oracle":
            cache = self._target_prefix_cache(prefix)
        elif self.mode == "mapped":
            assert self.mapped_prefix_builder is not None
            cache = self.mapped_prefix_builder(prefix)
        else:
            raise AssertionError("standalone mode has no parent prefix cache")
        if cache_sequence_length(cache) != len(prefix):
            raise ValueError(
                f"prefix cache length {cache_sequence_length(cache)} != token length {len(prefix)}"
            )
        self._memo_key, self._memo_cache = key, cache
        return cache

    @staticmethod
    def _reduce_logits(logits: torch.Tensor, labels: torch.Tensor) -> ContinuationScore:
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError(f"expected batch-one logits [1,T,V], got {tuple(logits.shape)}")
        if labels.ndim != 2 or labels.shape[0] != 1:
            raise ValueError(f"expected batch-one labels [1,T], got {tuple(labels.shape)}")
        if logits.shape[1] != labels.shape[1]:
            raise ValueError(
                f"logit/label lengths differ: {logits.shape[1]} != {labels.shape[1]}"
            )
        labels = labels.to(logits.device)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        selected = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return ContinuationScore(
            loglikelihood=float(selected.double().sum().item()),
            is_greedy=bool(torch.equal(logits.argmax(-1), labels)),
            token_count=int(labels.numel()),
        )

    @torch.inference_mode()
    def score_token_ids(
        self, context_ids: Sequence[int], continuation_ids: Sequence[int]
    ) -> ContinuationScore:
        context, continuation = self._trim_score_pair(context_ids, continuation_ids)
        labels = torch.tensor([continuation], dtype=torch.long, device=self.device)
        if self.mode in STANDALONE_MODES or len(context) == 1:
            query = torch.tensor(
                [context + continuation[:-1]], dtype=torch.long, device=self.device
            )
            output = self.target_model(
                input_ids=query,
                attention_mask=torch.ones_like(query),
                position_ids=torch.arange(query.shape[1], device=self.device).unsqueeze(0),
                cache_position=torch.arange(query.shape[1], device=self.device),
                use_cache=False,
                return_dict=True,
            )
            begin = len(context) - 1
            logits = output.logits[:, begin : begin + len(continuation), :]
            return self._reduce_logits(logits, labels)

        parent = self._parent_prefix_cache(context[:-1])
        boundary = torch.tensor([[context[-1]]], dtype=torch.long, device=self.device)
        model_inputs, labels = continuation_scoring_inputs(
            boundary,
            labels,
            parent,
            clone_cache=True,
        )
        output = self.target_model(**model_inputs, return_dict=True)
        return self._reduce_logits(output.logits, labels)

    def loglikelihood_text(self, context: str, continuation: str) -> ContinuationScore:
        context_ids, continuation_ids = encode_context_continuation(
            self.tokenizer,
            context,
            continuation,
            prefix_token_id=self.prefix_token_id,
        )
        return self.score_token_ids(context_ids, continuation_ids)

    def rolling_loglikelihood_text(self, text: str) -> float:
        tokens = self.tok_encode(text)
        total = 0.0
        for context, continuation in rolling_token_windows(
            tokens,
            prefix_token_id=self.prefix_token_id,
            max_length=self.max_length,
        ):
            total += self.score_token_ids(context, continuation).loglikelihood
        return total

    def _generation_kwargs(self, raw: Mapping[str, Any]) -> tuple[int, list[str]]:
        kwargs = dict(raw)
        unknown = set(kwargs) - {
            "until",
            "max_gen_toks",
            "max_new_tokens",
            "max_length",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
        }
        if unknown:
            raise ValueError(f"unsupported generation kwargs: {sorted(unknown)}")
        if bool(kwargs.get("do_sample", False)):
            raise ValueError("paper evaluation uses greedy decoding; do_sample must be false")
        if kwargs.get("temperature") not in (None, 0, 0.0):
            raise ValueError("paper evaluation requires temperature=0/None")
        if kwargs.get("top_p") not in (None, 1, 1.0):
            raise ValueError("top_p is unsupported for greedy paper evaluation")
        if kwargs.get("top_k") not in (None, 0):
            raise ValueError("top_k is unsupported for greedy paper evaluation")
        max_tokens = kwargs.get("max_gen_toks", kwargs.get("max_new_tokens", self.max_gen_tokens))
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            raise ValueError("max_gen_toks must be positive")
        until = kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        if until is None:
            until = []
        if not isinstance(until, Sequence) or any(not isinstance(item, str) for item in until):
            raise TypeError("until must be a string or sequence of strings")
        if any(item == "" for item in until):
            raise ValueError("empty stop sequences are invalid")
        return max_tokens, list(until)

    @torch.inference_mode()
    def generate_token_ids(
        self,
        context_token_ids: Sequence[int],
        gen_kwargs: Mapping[str, Any],
        *,
        truncation: str = "suffix",
    ) -> GenerationResult:
        """Generate from exact prompt token IDs and report how generation stopped.

        ``suffix`` is the legacy lm-eval behavior.  ``head_tail`` implements
        LongBench's middle-removal convention without a lossy decode/re-encode
        cycle.  ``error`` is useful for protocols that forbid truncation.
        """

        max_tokens, stop_strings = self._generation_kwargs(gen_kwargs)
        if max_tokens >= self.max_length:
            raise ValueError("max_gen_toks must be smaller than max_length")
        if truncation not in {"suffix", "head_tail", "error"}:
            raise ValueError("truncation must be suffix, head_tail, or error")
        context_ids = [int(token) for token in context_token_ids]
        original_context_tokens = len(context_ids)
        if not context_ids:
            context_ids = [self.prefix_token_id]
        context_budget = self.max_length - max_tokens
        if len(context_ids) > context_budget:
            if truncation == "suffix":
                context_ids = context_ids[-context_budget:]
            elif truncation == "head_tail":
                context_ids, _ = head_tail_truncate_token_ids(context_ids, context_budget)
            else:
                raise ValueError(
                    f"generation context has {len(context_ids)} tokens, exceeding "
                    f"budget={context_budget}"
                )

        if self.mode in STANDALONE_MODES or len(context_ids) == 1:
            ids = torch.tensor([context_ids], dtype=torch.long, device=self.device)
            output = self.target_model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                position_ids=torch.arange(ids.shape[1], device=self.device).unsqueeze(0),
                cache_position=torch.arange(ids.shape[1], device=self.device),
                use_cache=True,
                return_dict=True,
            )
        else:
            parent = self._parent_prefix_cache(context_ids[:-1])
            boundary = torch.tensor([[context_ids[-1]]], dtype=torch.long, device=self.device)
            inputs = boundary_replay_inputs(boundary, parent, clone_cache=True)
            output = self.target_model(**inputs, return_dict=True)

        generated: list[int] = []
        eos_value = getattr(self.tokenizer, "eos_token_id", None)
        eos_ids = {int(eos_value)} if eos_value is not None else set()
        generation_eos = getattr(
            getattr(self.target_model, "generation_config", None), "eos_token_id", None
        )
        if isinstance(generation_eos, int):
            eos_ids.add(generation_eos)
        elif generation_eos is not None:
            eos_ids.update(int(value) for value in generation_eos)
        stop_reason = "max_tokens"
        for step in range(max_tokens):
            next_token = int(output.logits[:, -1, :].argmax(-1).item())
            reached_eos = next_token in eos_ids
            if not reached_eos:
                generated.append(next_token)
            # Avoid quadratic decode work for long generations with no stop
            # strings (the extended runner uses exact token caps).  Legacy
            # stop-string behavior still decodes and checks every step.
            if stop_strings:
                text = _tokenizer_decode(self.tokenizer, generated)
                stop_at = min(
                    (text.find(stop) for stop in stop_strings if stop in text),
                    default=-1,
                )
                if stop_at >= 0:
                    return GenerationResult(
                        text=text[:stop_at],
                        generated_token_ids=tuple(generated),
                        stop_reason="stop_sequence",
                        cap_hit=False,
                        requested_max_tokens=max_tokens,
                        original_context_tokens=original_context_tokens,
                        used_context_tokens=len(context_ids),
                        truncation=truncation,
                    )
            if reached_eos:
                return GenerationResult(
                    text=_tokenizer_decode(self.tokenizer, generated),
                    generated_token_ids=tuple(generated),
                    stop_reason="eos",
                    cap_hit=False,
                    requested_max_tokens=max_tokens,
                    original_context_tokens=original_context_tokens,
                    used_context_tokens=len(context_ids),
                    truncation=truncation,
                )
            if step + 1 == max_tokens:
                break

            past = output.past_key_values
            prefix_length = cache_sequence_length(past)
            token = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            attention_mask = full_attention_mask(
                prefix_length, 1, batch_size=1, device=self.device
            )
            output = self.target_model(
                input_ids=token,
                past_key_values=past,
                attention_mask=attention_mask,
                position_ids=replay_position_ids(attention_mask, 1),
                cache_position=replay_cache_position(prefix_length, 1, device=self.device),
                use_cache=True,
                return_dict=True,
            )
        return GenerationResult(
            text=_tokenizer_decode(self.tokenizer, generated),
            generated_token_ids=tuple(generated),
            stop_reason=stop_reason,
            cap_hit=True,
            requested_max_tokens=max_tokens,
            original_context_tokens=original_context_tokens,
            used_context_tokens=len(context_ids),
            truncation=truncation,
        )

    @torch.inference_mode()
    def generate_text(self, context: str, gen_kwargs: Mapping[str, Any]) -> str:
        """Legacy string-only API used by the frozen paper evaluation."""

        return self.generate_token_ids(
            self.tok_encode(context), gen_kwargs, truncation="suffix"
        ).text


class LMEvalRequestAdapter:
    """Method implementation mixed with ``lm_eval.api.model.LM`` at runtime."""

    def _adapter_init(self, engine: TransferEvaluationEngine) -> None:
        self.engine = engine
        self.tokenizer = engine.tokenizer
        self.batch_size = 1
        self.max_length = engine.max_length
        self.max_gen_toks = engine.max_gen_tokens
        self._device = engine.device
        self.config = engine.target_model.config

    def get_model_info(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "evaluation_mode": self.engine.mode,
            "completion_mode": True,
            "apply_chat_template": False,
            "boundary_replay": True,
            "batch_size": 1,
        }
        builder = self.engine.mapped_prefix_builder
        provenance = getattr(builder, "provenance", None)
        if isinstance(provenance, Mapping):
            value["mapper"] = dict(provenance)
        return value

    @property
    def tokenizer_name(self) -> str:
        return str(
            getattr(self.engine.tokenizer, "name_or_path", type(self.engine.tokenizer).__name__)
        )

    def chat_template(self, chat_template: bool | str = False) -> str:
        if chat_template:
            raise ValueError("this reproduction is completion-only; chat templates are forbidden")
        return ""

    def apply_chat_template(self, *args: Any, **kwargs: Any) -> str:
        raise ValueError("this reproduction is completion-only; chat templates are forbidden")

    def loglikelihood(self, requests: Sequence[Any], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = request.args
            score = self.engine.loglikelihood_text(context, continuation)
            value = (score.loglikelihood, score.is_greedy)
            results.append(value)
            hook = getattr(self, "cache_hook", None)
            if hook is not None:
                hook.add_partial("loglikelihood", request.args, value)
        return results

    def loglikelihood_rolling(self, requests: Sequence[Any], disable_tqdm: bool = False) -> list[float]:
        results: list[float] = []
        for request in requests:
            (text,) = request.args
            value = self.engine.rolling_loglikelihood_text(text)
            results.append(value)
            hook = getattr(self, "cache_hook", None)
            if hook is not None:
                hook.add_partial("loglikelihood_rolling", request.args, value)
        return results

    def generate_until(self, requests: Sequence[Any], disable_tqdm: bool = False) -> list[str]:
        results: list[str] = []
        for request in requests:
            context, kwargs = request.args
            value = self.engine.generate_text(context, kwargs)
            results.append(value)
            hook = getattr(self, "cache_hook", None)
            if hook is not None:
                hook.add_partial("generate_until", request.args, value)
        return results


def make_lm_eval_adapter(engine: TransferEvaluationEngine) -> Any:
    """Return an initialized harness ``LM`` without importing it at module load."""

    try:
        from lm_eval.api.model import LM
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "lm-evaluation-harness is not installed; install the pinned vendor checkout "
            "and its task dependencies before benchmark execution"
        ) from exc

    class TransferCacheLM(LMEvalRequestAdapter, LM):
        def __init__(self, value: TransferEvaluationEngine) -> None:
            LM.__init__(self)
            self._adapter_init(value)

    return TransferCacheLM(engine)


def evaluate_prefix_perplexity(
    engine: TransferEvaluationEngine,
    token_ids: Sequence[int],
    *,
    chunk_length: int = 2048,
    prefix_length: int = 1024,
    scored_length: int = 1024,
    max_chunks: int | None = None,
) -> PrefixPerplexityResult:
    """Score non-overlapping WikiText-2 chunks with a fixed cached prefix."""

    if min(chunk_length, prefix_length, scored_length) <= 0:
        raise ValueError("WikiText chunk lengths must be positive")
    if prefix_length + scored_length != chunk_length:
        raise ValueError("paper protocol requires prefix_length + scored_length == chunk_length")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError("max_chunks must be positive when supplied")
    tokens = list(map(int, token_ids))
    total_loglikelihood = 0.0
    total_tokens = 0
    chunks = 0
    for begin in range(0, len(tokens) - chunk_length + 1, chunk_length):
        if max_chunks is not None and chunks >= max_chunks:
            break
        chunk = tokens[begin : begin + chunk_length]
        score = engine.score_token_ids(
            chunk[:prefix_length],
            chunk[prefix_length : prefix_length + scored_length],
        )
        if score.token_count != scored_length:
            raise AssertionError("WikiText chunk scored the wrong number of tokens")
        total_loglikelihood += score.loglikelihood
        total_tokens += score.token_count
        chunks += 1
    if chunks == 0:
        raise ValueError(f"token stream has no complete {chunk_length}-token chunk")
    perplexity = math.exp(-total_loglikelihood / total_tokens)
    return PrefixPerplexityResult(
        loglikelihood=total_loglikelihood,
        scored_tokens=total_tokens,
        chunks=chunks,
        perplexity=perplexity,
    )


def concatenate_document_tokens(
    tokenizer: Any,
    documents: Iterable[str],
    *,
    separator: str = "\n\n",
) -> list[int]:
    """Tokenize one explicitly separated document stream without chat formatting."""

    values = [document for document in documents if isinstance(document, str) and document]
    if not values:
        raise ValueError("dataset supplied no non-empty text documents")
    return _tokenizer_encode(tokenizer, separator.join(values))


__all__ = [
    "ContinuationScore",
    "EVALUATION_MODES",
    "GenerationResult",
    "LMEvalRequestAdapter",
    "MappedPrefixCacheBuilder",
    "PrefixPerplexityResult",
    "STANDALONE_MODES",
    "TransferEvaluationEngine",
    "concatenate_document_tokens",
    "encode_context_continuation",
    "evaluate_prefix_perplexity",
    "head_tail_truncate_token_ids",
    "make_lm_eval_adapter",
    "rolling_token_windows",
]
