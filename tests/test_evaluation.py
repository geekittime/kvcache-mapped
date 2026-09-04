from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import kvcache_transfer.evaluation as evaluation_module
from kvcache_transfer.cache_adapter import build_dynamic_cache, cache_to_legacy
from kvcache_transfer.config import ArchitectureSpec
from kvcache_transfer.evaluation import (
    LMEvalRequestAdapter,
    MappedPrefixCacheBuilder,
    TransferEvaluationEngine,
    encode_context_continuation,
    evaluate_prefix_perplexity,
    head_tail_truncate_token_ids,
    rolling_token_windows,
)
from kvcache_transfer.mapper import LayerMapper
from kvcache_transfer.transfer import transfer_source_cache


class CharacterTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    name_or_path = "character-tokenizer"

    def __init__(self, vocabulary_size: int = 32) -> None:
        self.vocabulary_size = vocabulary_size

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [3 + (ord(character) % (self.vocabulary_size - 3)) for character in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        values = []
        for token in token_ids:
            if skip_special_tokens and token in (self.bos_token_id, self.eos_token_id, self.pad_token_id):
                continue
            values.append(chr(97 + (int(token) - 3) % 26))
        return "".join(values)

    def get_vocab(self) -> dict[str, int]:
        return {str(index): index for index in range(self.vocabulary_size)}


class CountingQwen3(Qwen3ForCausalLM):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.forward_records: list[dict[str, object]] = []

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids")
        past = kwargs.get("past_key_values")
        self.forward_records.append(
            {
                "input_ids": input_ids.detach().cpu().clone() if input_ids is not None else None,
                "past_id": id(past) if past is not None else None,
                # Holding a reference prevents Python from reusing the first
                # branch object's id after that candidate forward returns.
                "past": past,
                "attention_shape": tuple(kwargs["attention_mask"].shape)
                if kwargs.get("attention_mask") is not None
                else None,
            }
        )
        return super().forward(*args, **kwargs)


@pytest.fixture()
def tiny_model() -> CountingQwen3:
    torch.manual_seed(17)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        use_sliding_window=False,
        sliding_window=None,
        max_window_layers=0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    model = CountingQwen3(config).eval()
    return model


class CountingIdentityMapperBundle:
    """Tiny lazy MapperBundle double whose loads stand in for file/hash work."""

    def __init__(self, architecture: ArchitectureSpec) -> None:
        self.architecture = architecture
        self.manifest_path = Path("/tiny/mapper/manifest.json")
        self.manifest_sha256 = "b" * 64
        self.fit_fingerprint = "c" * 64
        self.layers = tuple(range(architecture.target_layers))
        self.load_calls: list[int] = []

    def load_layer(self, target_layer: int, *, device="cpu") -> LayerMapper:
        self.load_calls.append(target_layer)
        features = self.architecture.target_features
        identity = torch.eye(features, dtype=torch.float32, device=device)
        zeros = torch.zeros(features, dtype=torch.float32, device=device)
        return LayerMapper(
            source_layers=(target_layer,),
            key_weight=identity.clone(),
            key_bias=zeros.clone(),
            value_weight=identity.clone(),
            value_bias=zeros.clone(),
        )

    def iter_layers(self, layer_devices):
        for target_layer, device in enumerate(layer_devices):
            yield target_layer, self.load_layer(target_layer, device=device)


def test_mapped_builder_materializes_once_and_matches_lazy_transfer(
    tiny_model, monkeypatch
):
    architecture = ArchitectureSpec(
        source_layers=2,
        target_layers=2,
        source_kv_heads=2,
        target_kv_heads=2,
        source_head_dim=8,
        target_head_dim=8,
        selected_source_layers=1,
    )
    lazy = CountingIdentityMapperBundle(architecture)
    open_calls: list[Path] = []

    def open_once(_cls, directory, **_kwargs):
        open_calls.append(Path(directory))
        return lazy

    monkeypatch.setattr(evaluation_module.MapperBundle, "open", classmethod(open_once))
    builder = MappedPrefixCacheBuilder(
        source_model=tiny_model,
        target_model=tiny_model,
        mapper_manifest=Path("/tiny/mapper/manifest.json"),
        token_chunk=2,
    )
    # MapperBundle.open performs the complete bundle/hash verification in the
    # real implementation.  Every layer is then loaded exactly once during
    # construction, not once per prefix.
    assert len(open_calls) == 1
    assert lazy.load_calls == [0, 1]
    expected_bytes = 2 * (2 * 16 * 16 + 2 * 16) * 4
    assert builder.provenance["resident_parameter_bytes"] == expected_bytes
    assert builder.provenance["materialization"]["scope"].startswith("one_time_setup")

    prefix = [7, 5, 19, 2, 11]
    input_ids = torch.tensor([prefix])
    positions = torch.arange(len(prefix)).unsqueeze(0)
    with torch.inference_mode():
        source_output = tiny_model.model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=positions,
            cache_position=torch.arange(len(prefix)),
            use_cache=True,
            return_dict=True,
        )
    reference_lazy = CountingIdentityMapperBundle(architecture)
    expected = transfer_source_cache(
        source_output.past_key_values,
        source_config=tiny_model.config,
        target_model=tiny_model,
        mapper_bundle=reference_lazy,  # type: ignore[arg-type]
        position_ids=positions,
        token_chunk=2,
    )
    first = builder(prefix)
    second = builder([3, 4, 5, 6])
    assert lazy.load_calls == [0, 1]
    for layer in range(architecture.target_layers):
        torch.testing.assert_close(first.key_cache[layer], expected.key_cache[layer])
        torch.testing.assert_close(first.value_cache[layer], expected.value_cache[layer])
    assert second.get_seq_length() == 4


def _oracle_builder(model: Qwen3ForCausalLM):
    device = next(model.parameters()).device

    @torch.inference_mode()
    def build(prefix_ids: list[int]):
        ids = torch.tensor([prefix_ids], device=device)
        output = model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=torch.arange(ids.shape[1], device=device).unsqueeze(0),
            cache_position=torch.arange(ids.shape[1], device=device),
            use_cache=True,
            return_dict=True,
        )
        # Return a fresh cache detached from the model output so this test
        # exercises the same injected-DynamicCache path as a mapped builder.
        return build_dynamic_cache(cache_to_legacy(output.past_key_values), clone=True)

    return build


def _engine(model, tokenizer, mode, mapped_builder=None, max_length=64):
    return TransferEvaluationEngine(
        target_model=model,
        tokenizer=tokenizer,
        mode=mode,
        mapped_prefix_builder=mapped_builder,
        max_length=max_length,
        max_gen_tokens=8,
    )


def test_head_tail_truncation_is_exact_and_deterministic():
    tokens = list(range(10))
    truncated, metadata = head_tail_truncate_token_ids(tokens, 5)
    assert truncated == [0, 1, 2, 8, 9]
    assert metadata == {
        "policy": "head_tail",
        "truncated": True,
        "original_tokens": 10,
        "used_tokens": 5,
        "removed_tokens": 5,
        "head_tokens": 3,
        "tail_tokens": 2,
    }
    unchanged, unchanged_metadata = head_tail_truncate_token_ids(tokens, 10)
    assert unchanged == tokens
    assert unchanged_metadata["truncated"] is False
    with pytest.raises(ValueError, match="token budget"):
        head_tail_truncate_token_ids(tokens, 0)


def test_encode_pair_matches_lm_eval_trailing_space_rule():
    tokenizer = CharacterTokenizer()
    context, continuation = encode_context_continuation(
        tokenizer,
        "abc ",
        "def",
        prefix_token_id=tokenizer.bos_token_id,
    )
    assert context == tokenizer.encode("abc")
    assert context + continuation == tokenizer.encode("abc def")

    empty_context, empty_continuation = encode_context_continuation(
        tokenizer,
        "",
        "xyz",
        prefix_token_id=tokenizer.bos_token_id,
    )
    assert empty_context == [tokenizer.bos_token_id]
    assert empty_continuation == tokenizer.encode("xyz")


def test_standalone_oracle_and_identity_mapped_scores_match(tiny_model):
    tokenizer = CharacterTokenizer()
    source_standalone = _engine(tiny_model, tokenizer, "source_standalone")
    standalone = _engine(tiny_model, tokenizer, "target_standalone")
    oracle = _engine(tiny_model, tokenizer, "target_oracle")
    mapped = _engine(
        tiny_model,
        tokenizer,
        "mapped",
        mapped_builder=_oracle_builder(tiny_model),
    )
    context = [4, 5, 6, 7, 8]
    continuation = [9, 10, 11]
    expected = standalone.score_token_ids(context, continuation)
    source_score = source_standalone.score_token_ids(context, continuation)
    oracle_score = oracle.score_token_ids(context, continuation)
    mapped_score = mapped.score_token_ids(context, continuation)

    assert source_score.loglikelihood == pytest.approx(expected.loglikelihood, abs=1e-5)
    assert oracle_score.loglikelihood == pytest.approx(expected.loglikelihood, abs=1e-5)
    assert mapped_score.loglikelihood == pytest.approx(expected.loglikelihood, abs=1e-5)
    assert oracle_score.is_greedy == expected.is_greedy
    assert mapped_score.is_greedy == expected.is_greedy
    assert expected.token_count == 3


def test_boundary_replay_scores_first_continuation_token(tiny_model):
    tokenizer = CharacterTokenizer()
    engine = _engine(tiny_model, tokenizer, "target_oracle")
    tiny_model.forward_records.clear()
    score = engine.score_token_ids([3, 4, 5, 6], [7, 8, 9])

    assert score.token_count == 3
    # Prefill C[:-1], then query C[-1] + Y[:-1].  There must be three logits
    # for the three continuation labels; omitting replay would produce two.
    assert tiny_model.forward_records[-2]["input_ids"].shape[1] == 3
    assert tiny_model.forward_records[-1]["input_ids"].tolist() == [[6, 7, 8]]
    assert tiny_model.forward_records[-1]["attention_shape"] == (1, 6)


def test_multiple_choice_candidates_reuse_parent_but_receive_isolated_branches(tiny_model):
    tokenizer = CharacterTokenizer()
    engine = _engine(tiny_model, tokenizer, "target_oracle")
    tiny_model.forward_records.clear()
    first = engine.score_token_ids([3, 4, 5, 6], [7, 8])
    second = engine.score_token_ids([3, 4, 5, 6], [9, 10])

    assert first.token_count == second.token_count == 2
    # One parent prefill only.  Each candidate then gets a distinct cloned
    # DynamicCache object, so candidate one cannot append into candidate two.
    assert len(tiny_model.forward_records) == 3
    first_branch = tiny_model.forward_records[1]["past"]
    second_branch = tiny_model.forward_records[2]["past"]
    assert first_branch is not None and second_branch is not None
    assert first_branch is not second_branch
    assert tiny_model.forward_records[1]["attention_shape"] == (1, 5)
    assert tiny_model.forward_records[2]["attention_shape"] == (1, 5)


def test_rolling_windows_score_each_token_exactly_once():
    tokens = list(range(13))
    windows = list(rolling_token_windows(tokens, prefix_token_id=99, max_length=4))
    assert [token for _, continuation in windows for token in continuation] == tokens
    assert all(context for context, _ in windows)
    assert all(len(context) + len(continuation) - 1 <= 4 for context, continuation in windows)


def test_rolling_engine_matches_manual_window_sum(tiny_model):
    tokenizer = CharacterTokenizer()
    engine = _engine(tiny_model, tokenizer, "target_oracle", max_length=5)
    text = "abcdefghijk"
    expected = sum(
        engine.score_token_ids(context, continuation).loglikelihood
        for context, continuation in rolling_token_windows(
            tokenizer.encode(text), prefix_token_id=tokenizer.bos_token_id, max_length=5
        )
    )
    engine.clear_prefix_cache()
    assert engine.rolling_loglikelihood_text(text) == pytest.approx(expected, abs=1e-6)


def test_generation_oracle_and_identity_mapped_match_standalone(tiny_model):
    tokenizer = CharacterTokenizer()
    source_standalone = _engine(tiny_model, tokenizer, "source_standalone")
    standalone = _engine(tiny_model, tokenizer, "target_standalone")
    oracle = _engine(tiny_model, tokenizer, "target_oracle")
    mapped = _engine(
        tiny_model,
        tokenizer,
        "mapped",
        mapped_builder=_oracle_builder(tiny_model),
    )
    kwargs = {"max_gen_toks": 5, "do_sample": False, "until": ["zzzz"]}
    expected = standalone.generate_text("abcdef", kwargs)
    assert source_standalone.generate_text("abcdef", kwargs) == expected
    assert oracle.generate_text("abcdef", kwargs) == expected
    assert mapped.generate_text("abcdef", kwargs) == expected

    rich = standalone.generate_token_ids(
        [3 + (index % 29) for index in range(100)], kwargs, truncation="head_tail"
    )
    assert isinstance(rich.text, str)
    assert rich.original_context_tokens == 100
    assert rich.used_context_tokens == standalone.max_length - 5
    assert rich.truncation == "head_tail"
    assert rich.stop_reason in {"eos", "stop_sequence", "max_tokens"}
    assert rich.cap_hit is (rich.stop_reason == "max_tokens")

    with pytest.raises(ValueError, match="exceeding budget"):
        standalone.generate_token_ids(
            [3 + (index % 29) for index in range(100)], kwargs, truncation="error"
        )

    with pytest.raises(ValueError, match="greedy"):
        standalone.generate_text("abcdef", {"do_sample": True})


class RecordingScoreEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[list[int], list[int]]] = []

    def score_token_ids(self, context, continuation):
        context, continuation = list(context), list(continuation)
        self.calls.append((context, continuation))
        return SimpleNamespace(
            loglikelihood=-float(len(continuation)),
            token_count=len(continuation),
            is_greedy=True,
        )


def test_wikitext_prefix_protocol_is_nonoverlap_and_scores_only_second_half():
    engine = RecordingScoreEngine()
    result = evaluate_prefix_perplexity(
        engine,
        list(range(18)),
        chunk_length=6,
        prefix_length=3,
        scored_length=3,
    )
    assert engine.calls == [
        ([0, 1, 2], [3, 4, 5]),
        ([6, 7, 8], [9, 10, 11]),
        ([12, 13, 14], [15, 16, 17]),
    ]
    assert result.chunks == 3
    assert result.scored_tokens == 9
    assert result.loglikelihood == -9
    assert result.perplexity == pytest.approx(math.e)


def test_engine_rejects_batch_or_chat_adjacent_protocols(tiny_model):
    tokenizer = CharacterTokenizer()
    with pytest.raises(ValueError, match="requires mapped_prefix_builder"):
        _engine(tiny_model, tokenizer, "mapped")
    engine = _engine(tiny_model, tokenizer, "target_standalone")
    with pytest.raises(ValueError, match="continuation"):
        engine.loglikelihood_text("abc", "")
    with pytest.raises(ValueError, match="unsupported generation kwargs"):
        engine.generate_text("abc", {"num_beams": 2})


def test_lm_eval_request_methods_preserve_order_and_cardinality(tiny_model):
    tokenizer = CharacterTokenizer()
    engine = _engine(tiny_model, tokenizer, "target_oracle", max_length=8)

    class Adapter(LMEvalRequestAdapter):
        def __init__(self):
            self._adapter_init(engine)
            self.cache_hook = SimpleNamespace(add_partial=lambda *args: None)

    adapter = Adapter()
    likelihood_requests = [
        SimpleNamespace(args=("abc", "d")),
        SimpleNamespace(args=("abc", "ef")),
    ]
    likelihoods = adapter.loglikelihood(likelihood_requests)
    assert len(likelihoods) == len(likelihood_requests)
    assert all(isinstance(value[0], float) and isinstance(value[1], bool) for value in likelihoods)

    rolling_requests = [SimpleNamespace(args=("abc",)), SimpleNamespace(args=("defg",))]
    rolling = adapter.loglikelihood_rolling(rolling_requests)
    assert len(rolling) == 2 and all(isinstance(value, float) for value in rolling)

    generation_requests = [
        SimpleNamespace(args=("abc", {"max_gen_toks": 1, "do_sample": False})),
        SimpleNamespace(args=("def", {"max_gen_toks": 2, "do_sample": False})),
    ]
    generations = adapter.generate_until(generation_requests)
    assert len(generations) == 2 and all(isinstance(value, str) for value in generations)
    assert adapter.get_model_info()["evaluation_mode"] == "target_oracle"
    with pytest.raises(ValueError, match="completion-only"):
        adapter.apply_chat_template([])
