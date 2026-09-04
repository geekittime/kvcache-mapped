from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cache_oracle import (  # noqa: E402
    AttentionReadout,
    CacheContext,
    CacheProvenance,
    OracleContractError,
    OracleObservation,
    answer_f1,
    attention_pair_metrics,
    attention_readout_fp32,
    boundary_replay_inputs,
    compose_legacy_cache,
    first_token_pair_metrics,
    generation_pair_metrics,
    intervention_catalog,
    normalize_answer,
    validate_qwen3_config,
    wrap_cache,
)
from oracle_diagnostics import (  # noqa: E402
    DEFAULT_OUTPUT,
    PreparationError,
    hotpot_prompt,
    iter_json_array,
    load_config,
    musique_prompt,
    validate_mapper_manifest,
    validate_prepared,
)
from runtime_harness import OracleHarness, SamplePlan  # noqa: E402


def context(token_ids: tuple[int, ...] = (11, 12, 13)) -> CacheContext:
    return CacheContext.from_prompt(
        dataset="hotpotqa_train",
        sample_id="sample-1",
        serialized_prompt="serialized prompt",
        token_ids=token_ids,
        target_config_sha256="a" * 64,
    )


def layers(value: float, *, ctx: CacheContext | None = None, dtype: torch.dtype = torch.float32):
    ctx = context() if ctx is None else ctx
    return tuple(
        (
            torch.full(
                (1, 8, ctx.prefix_length, 128), value + layer_index,
                dtype=dtype,
            ),
            torch.full(
                (1, 8, ctx.prefix_length, 128), value + 100 + layer_index,
                dtype=dtype,
            ),
        )
        for layer_index in range(40)
    )


def envelopes(*, ctx: CacheContext | None = None):
    ctx = context() if ctx is None else ctx
    native = wrap_cache(
        layers(1.0, ctx=ctx),
        context=ctx,
        provenance=CacheProvenance("native", "native14-prefix", "b" * 64),
    )
    mapped = wrap_cache(
        layers(-1000.0, ctx=ctx),
        context=ctx,
        provenance=CacheProvenance("mapped", "k20-runtime", "c" * 64, "d" * 64),
    )
    return native, mapped


def test_intervention_catalog_is_exact_and_ordered() -> None:
    values = intervention_catalog()
    assert [value.id for value in values] == [
        "native_k_native_v",
        "mapped_k_native_v",
        "native_k_mapped_v",
        "mapped_k_mapped_v",
        "native_kv_layers_0_9",
        "native_kv_layers_10_19",
        "native_kv_layers_20_29",
        "native_kv_layers_30_39",
        "native_kv_layer_34",
        "native_kv_layer_35",
        "native_kv_layers_34_35",
    ]
    assert values[4].native_kv_layers == tuple(range(10))
    assert values[7].native_kv_layers == tuple(range(30, 40))
    assert values[-1].native_kv_layers == (34, 35)


@pytest.mark.parametrize(
    ("intervention", "key_source", "value_source"),
    [
        ("native_k_native_v", "native", "native"),
        ("mapped_k_native_v", "mapped", "native"),
        ("native_k_mapped_v", "native", "mapped"),
        ("mapped_k_mapped_v", "mapped", "mapped"),
    ],
)
def test_global_factorial_selects_components(
    intervention: str, key_source: str, value_source: str
) -> None:
    native, mapped = envelopes()
    pairs, receipt = compose_legacy_cache(native, mapped, intervention)
    expected_key = native.layers[0][0] if key_source == "native" else mapped.layers[0][0]
    expected_value = native.layers[0][1] if value_source == "native" else mapped.layers[0][1]
    assert torch.equal(pairs[0][0], expected_key)
    assert torch.equal(pairs[0][1], expected_value)
    assert pairs[0][0].data_ptr() != expected_key.data_ptr()
    assert pairs[0][1].data_ptr() != expected_value.data_ptr()
    assert receipt.k_sources[0] == key_source
    assert receipt.v_sources[0] == value_source
    assert receipt.cloned is True


@pytest.mark.parametrize(
    ("intervention", "native_layers"),
    [
        ("native_kv_layers_0_9", set(range(0, 10))),
        ("native_kv_layers_10_19", set(range(10, 20))),
        ("native_kv_layers_20_29", set(range(20, 30))),
        ("native_kv_layers_30_39", set(range(30, 40))),
        ("native_kv_layer_34", {34}),
        ("native_kv_layer_35", {35}),
        ("native_kv_layers_34_35", {34, 35}),
    ],
)
def test_layer_replacements_are_native_both_only(intervention: str, native_layers: set[int]) -> None:
    native, mapped = envelopes()
    pairs, receipt = compose_legacy_cache(native, mapped, intervention)
    for layer_index, (key, value) in enumerate(pairs):
        expected = native if layer_index in native_layers else mapped
        assert torch.equal(key, expected.layers[layer_index][0])
        assert torch.equal(value, expected.layers[layer_index][1])
        assert receipt.k_sources[layer_index] == ("native" if layer_index in native_layers else "mapped")
        assert receipt.v_sources[layer_index] == ("native" if layer_index in native_layers else "mapped")


def test_cache_pair_fails_on_prompt_rope_shape_dtype_and_finiteness() -> None:
    native, mapped = envelopes()
    different = context((11, 12, 14))
    mapped_other = wrap_cache(
        layers(-1.0, ctx=different),
        context=different,
        provenance=CacheProvenance("mapped", "k20", "c" * 64, "d" * 64),
    )
    with pytest.raises(OracleContractError, match="exact same prefix"):
        compose_legacy_cache(native, mapped_other, "mapped_k_mapped_v")

    with pytest.raises(OracleContractError, match="RoPE theta"):
        replace(context(), rope_theta=10_000.0).validate()

    malformed = list(layers(0.0))
    malformed[7] = (torch.zeros(1, 8, 1, 128), torch.zeros(1, 8, 1, 128))
    with pytest.raises(OracleContractError, match="shape drift"):
        wrap_cache(
            malformed,
            context=context(),
            provenance=CacheProvenance("native", "native", "b" * 64),
        )

    mapped_fp16 = wrap_cache(
        layers(-1.0, dtype=torch.float16),
        context=context(),
        provenance=CacheProvenance("mapped", "k20", "c" * 64, "d" * 64),
    )
    with pytest.raises(OracleContractError, match="dtype mismatch"):
        compose_legacy_cache(native, mapped_fp16, "mapped_k_mapped_v")

    nonfinite = list(layers(0.0))
    nonfinite[0][0][0, 0, 0, 0] = float("nan")
    with pytest.raises(OracleContractError, match="NaN or Inf"):
        wrap_cache(
            nonfinite,
            context=context(),
            provenance=CacheProvenance("native", "native", "b" * 64),
        )


def test_boundary_replay_inputs_are_exact_causal_boundary() -> None:
    native, mapped = envelopes()
    composed, _ = compose_legacy_cache(native, mapped, "mapped_k_native_v")
    inputs = boundary_replay_inputs(
        context=context(),
        boundary_input_ids=torch.tensor([[13]], dtype=torch.long),
        composed_cache=composed,
    )
    assert inputs["attention_mask"].tolist() == [[1, 1, 1]]
    assert inputs["position_ids"].tolist() == [[2]]
    assert inputs["cache_position"].tolist() == [2]
    assert len(inputs["past_key_values"]) == 40
    assert inputs["past_key_values"][0][0].shape == (1, 8, 2, 128)
    with pytest.raises(OracleContractError, match="differs from the frozen prompt"):
        boundary_replay_inputs(
            context=context(),
            boundary_input_ids=torch.tensor([[99]], dtype=torch.long),
            composed_cache=composed,
        )


def test_attention_readout_and_metrics_distinguish_k_and_v() -> None:
    query = torch.zeros(1, 40, 1, 128)
    key = torch.zeros(1, 8, 2, 128)
    native_value = torch.ones(1, 8, 2, 128)
    mapped_value = torch.full((1, 8, 2, 128), 2.0)
    native = attention_readout_fp32(
        query, key, native_value, query_absolute_positions=[1]
    )
    mapped = attention_readout_fp32(
        query, key, mapped_value, query_absolute_positions=[1]
    )
    assert native.output.shape == (1, 40, 1, 128)
    assert native.probabilities.shape == (1, 40, 1, 2)
    assert torch.allclose(native.probabilities, torch.full_like(native.probabilities, 0.5))
    metrics = attention_pair_metrics(native, mapped)
    assert metrics["attention_output_cosine"] == pytest.approx(1.0)
    assert metrics["attention_output_normalized_mse"] == pytest.approx(1.0)
    assert metrics["attention_probability_js"] == pytest.approx(0.0)

    with pytest.raises(OracleContractError, match="identical native query"):
        attention_pair_metrics(
            native,
            AttentionReadout(
                output=mapped.output,
                probabilities=mapped.probabilities,
                query_sha256_fp32="f" * 64,
                query_absolute_positions=mapped.query_absolute_positions,
            ),
        )

    changed_key = key.clone()
    changed_key[:, :, 0, 0] = -10.0
    changed_key[:, :, 1, 0] = 10.0
    changed_query = query.clone()
    changed_query[:, :, :, 0] = 1.0
    key_branch = attention_readout_fp32(
        changed_query, changed_key, native_value, query_absolute_positions=[1]
    )
    assert not torch.allclose(key_branch.probabilities, native.probabilities)


def test_attention_contract_rejects_content_space_like_wrong_geometry() -> None:
    with pytest.raises(OracleContractError, match="target Qwen3-14B"):
        attention_readout_fp32(
            torch.zeros(1, 32, 1, 128),
            torch.zeros(1, 8, 2, 128),
            torch.zeros(1, 8, 2, 128),
            query_absolute_positions=[1],
        )
    with pytest.raises(OracleContractError, match="outside"):
        attention_readout_fp32(
            torch.zeros(1, 40, 1, 128),
            torch.zeros(1, 8, 2, 128),
            torch.zeros(1, 8, 2, 128),
            query_absolute_positions=[2],
        )


def test_first_token_kl_and_answer_margin() -> None:
    native = torch.tensor([0.0, 3.0, 1.0, -1.0])
    identical = first_token_pair_metrics(native, native.clone(), gold_first_token_ids=[1])
    assert identical["first_token_kl_native_to_candidate"] == pytest.approx(0.0, abs=1e-15)
    assert identical["native_answer_margin"] == pytest.approx(2.0)
    assert identical["answer_margin_delta_candidate_minus_native"] == pytest.approx(0.0)
    candidate = torch.tensor([4.0, 1.0, 0.0, -1.0])
    changed = first_token_pair_metrics(native, candidate, gold_first_token_ids=[1])
    assert changed["first_token_kl_native_to_candidate"] > 0
    assert changed["candidate_answer_margin"] == pytest.approx(-3.0)
    assert changed["first_token_agreement"] is False


def test_generation_f1_interface_matches_normalized_token_f1() -> None:
    assert normalize_answer("The, Eiffel Tower!") == "eiffel tower"
    assert answer_f1("Eiffel Tower Paris", "the Eiffel Tower") == pytest.approx(0.8)
    metrics = generation_pair_metrics(
        "the Eiffel Tower", "Eiffel Tower Paris", ["Eiffel Tower", "Tour Eiffel"]
    )
    assert metrics["native_answer_f1"] == 1.0
    assert metrics["candidate_answer_f1"] == pytest.approx(0.8)
    assert metrics["answer_f1_delta_candidate_minus_native"] == pytest.approx(-0.2)


def test_qwen_config_and_train_only_config() -> None:
    config = load_config()
    for role in ("source", "target"):
        model_config = json.loads(Path(config["models"][role]["config"]["path"]).read_text())
        validate_qwen3_config(model_config, source=role == "source")
    assert config["execution_policy"]["gpu_execution_authorized_by_this_config"] is False
    assert config["execution_policy"]["locked_or_dev_split_access"] is False
    for specification in config["method_dev_data"].values():
        assert specification["tier"] == "method-dev"
        assert "train" in specification["split"]
        assert all(term not in specification["path"].lower().split("/")[-1] for term in ("dev", "test", "locked"))


def test_streaming_json_array(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"id": 1, "text": "x" * 1000}, {"id": 2}]))
    assert list(iter_json_array(path, chunk_size=17)) == [
        (0, {"id": 1, "text": "x" * 1000}),
        (1, {"id": 2}),
    ]
    path.write_text('[{"id":1}')
    with pytest.raises(PreparationError):
        list(iter_json_array(path, chunk_size=3))


def test_prompt_outer_template_is_identical() -> None:
    hotpot = {
        "_id": "h",
        "question": "Who?",
        "answer": "A",
        "context": [["Title", ["Sentence."]]],
        "supporting_facts": [["Title", 0]],
        "level": "easy",
        "type": "bridge",
    }
    musique = {
        "id": "m",
        "question": "Who?",
        "answer": "A",
        "answer_aliases": [],
        "answerable": True,
        "question_decomposition": [],
        "paragraphs": [
            {"idx": 0, "title": "Title", "paragraph_text": "Sentence.", "is_supporting": True}
        ],
    }
    assert hotpot_prompt(hotpot) == musique_prompt(musique)


def test_k20_manifest_is_strict_without_payload_read() -> None:
    receipt = validate_mapper_manifest(load_config(), deep_layer_hashes=False)
    assert receipt["layer_count"] == 40
    assert receipt["deep_layer_hashes"] is False
    assert receipt["layer_payload_bytes_hashed"] == 0


def _attention_map() -> dict[int, AttentionReadout]:
    readout = AttentionReadout(
        output=torch.ones(1, 40, 1, 128),
        probabilities=torch.full((1, 40, 1, 3), 1.0 / 3.0),
        query_sha256_fp32="e" * 64,
        query_absolute_positions=(2,),
    )
    return {layer: readout for layer in range(40)}


def test_dependency_injected_harness_runs_all_paired_interventions() -> None:
    ctx = context()
    native, mapped = envelopes(ctx=ctx)

    def native_provider(*, context: CacheContext, prefix_token_ids: torch.Tensor):
        assert context == ctx and prefix_token_ids.tolist() == [[11, 12]]
        return native

    def mapped_provider(*, context: CacheContext, prefix_token_ids: torch.Tensor):
        assert context == ctx and prefix_token_ids.tolist() == [[11, 12]]
        return mapped

    seen: list[str] = []

    def executor(
        *,
        context: CacheContext,
        boundary_token_id: int,
        composed_prefix_cache,
        max_new_tokens: int,
        intervention_id: str,
    ) -> OracleObservation:
        assert context == ctx and boundary_token_id == 13 and max_new_tokens == 256
        assert len(composed_prefix_cache) == 40
        seen.append(intervention_id)
        return OracleObservation(
            first_token_logits=torch.tensor([0.0, 3.0, 1.0]),
            generated_text="answer",
            generated_token_count=1,
            stop_reason="eos",
            attention=_attention_map(),
        )

    harness = OracleHarness(
        native_cache_provider=native_provider,
        mapped_cache_provider=mapped_provider,
        boundary_decode_executor=executor,
    )
    result = harness.run(
        SamplePlan(
            context=ctx,
            prompt_token_ids=(11, 12, 13),
            references=("answer",),
            gold_first_token_ids=(1,),
        )
    )
    assert seen == [value.id for value in intervention_catalog()]
    assert len(result.paired_records) == len(result.compositions) == 11
    assert all(row["first_token_kl_native_to_candidate"] == 0 for row in result.paired_records)
    assert all(row["candidate_answer_f1"] == 1 for row in result.paired_records)


def test_prepared_artifacts_are_deeply_self_consistent() -> None:
    result = validate_prepared(DEFAULT_OUTPUT)
    assert result["record_count"] == 512
    assert result["paired_plan_count"] == 512 * 11


def test_tests_never_initialize_cuda() -> None:
    assert torch.cuda.is_initialized() is False
