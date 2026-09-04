"""Controlled adapter from the shared long-context runner to reverse KV mapping.

The 2,300-line shared runner remains the single implementation of prompt
construction, causal-boundary replay, reference scoring, durable checkpoints,
and summary validation.  This module changes only the scientific direction,
formal topology, frozen sample scopes, and MultiNews metric surface.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from . import metrics as extension_metrics
from . import selection as extension_selection
from .protocol import (
    ATTENTION_IMPLEMENTATION,
    BASE_RUNNER,
    CODE_IDENTITY_FILES,
    CONFIG_PATH,
    DIALOG_REFERENCE_SCORE_CAP_TOKENS,
    DIRECTION,
    EXPECTED_CAPS,
    EXPECTED_DATA_IDENTITIES,
    EXPECTED_MAPPER_FIT_FINGERPRINT,
    EXPECTED_MAPPER_MANIFEST_SHA256,
    EXPECTED_RESULT_COUNT,
    EXPECTED_SAMPLE_COUNTS,
    EXPECTED_SMOKE_RESULT_COUNT,
    EXPECTED_TARGET_VOCAB_SIZE,
    EXPERIMENT_ROOT,
    EXTENSION_ROOT,
    EXTENSION_VERSION,
    FORMAL_TOPOLOGY,
    LMSYS_SELECTION_SEED,
    MAPPER_MANIFEST,
    MAPPER_TOKEN_CHUNK,
    MAX_LENGTH,
    MULTINEWS_SELECTION_SEED,
    PAPER_BIT_EXACT,
    PAPER_SCOPE_NOTE,
    PROTOCOL_NAME,
    REPOSITORY_ROOT,
    REVERSE_RUNTIME_CODE_FILES,
    SHAREGPT4_SELECTION_SEED,
    SMOKE_ENDPOINTS,
    SMOKE_SAMPLE_INDICES,
    SOURCE_MODEL,
    TARGET_MODEL,
    TASKS,
    output_within_experiment,
)

if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from reverse_kv.eval_common import (  # noqa: E402
    mapper_manifest_identity as reverse_mapper_manifest_identity,
    model_artifact_identity as reverse_model_artifact_identity,
)
from reverse_kv.protocol import read_json, sha256_file  # noqa: E402
from reverse_kv.runtime import ReverseMappedPrefixCacheBuilder  # noqa: E402


_BASE_MODULE_NAME = "qwen3_reverse_long_context_shared_runner"


def _load_base_runner() -> Any:
    """Load the exact shared runner without relying on ambiguous module names."""

    expected = BASE_RUNNER.resolve()
    if not expected.is_file():
        raise FileNotFoundError(expected)
    spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, expected)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared long-context runner: {expected}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses inspect their defining module through sys.modules.
    sys.modules[_BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != expected:
        raise ImportError(f"unexpected shared runner: {module.__file__}")
    return module


def _resolved(value: Any) -> Path:
    return Path(str(value)).resolve()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"reverse long-context config requires {label} mapping")
    return value


def _validate_raw_extension_contract(raw: Mapping[str, Any], args: Any) -> None:
    """Reject any drift before adapting the shared forward-specific validator."""

    if tuple(args.tasks) != TASKS:
        raise ValueError(
            "reverse long-context extension requires exactly --tasks "
            + ",".join(TASKS)
        )
    if _resolved(raw.get("project_root", "")) != EXPERIMENT_ROOT.resolve():
        raise ValueError("reverse long-context project_root changed")
    if raw.get("direction") != DIRECTION or raw.get("protocol") != PROTOCOL_NAME:
        raise ValueError("reverse long-context direction/protocol changed")
    if raw.get("paper_bit_exact") is not PAPER_BIT_EXACT:
        raise ValueError("paper_bit_exact disclosure changed")

    paths = _require_mapping(raw.get("paths"), "paths")
    expected_paths = {
        "source_model": SOURCE_MODEL.resolve(),
        "target_model": TARGET_MODEL.resolve(),
        "mapper_manifest": MAPPER_MANIFEST.resolve(),
    }
    for name, expected in expected_paths.items():
        if _resolved(paths.get(name, "")) != expected:
            raise ValueError(f"reverse long-context {name} changed")
    target_model_config = read_json(TARGET_MODEL / "config.json")
    if target_model_config.get("vocab_size") != EXPECTED_TARGET_VOCAB_SIZE:
        raise ValueError("reverse long-context target vocab_size changed")

    data = _require_mapping(raw.get("data"), "data")
    data_root = REPOSITORY_ROOT / "data"
    expected_data_paths = {
        "hotpotqa_dev_distractor": (
            data_root / "HotpotQA/raw/hotpot_dev_distractor_v1.json"
        ).resolve(),
        "multi_news_source": (
            data_root / "MultiNews/data/test.src.cleaned"
        ).resolve(),
        "multi_news_target": (data_root / "MultiNews/data/test.tgt").resolve(),
        "lmsys_parquet_dir": (data_root / "LMSYS/data").resolve(),
        "sharegpt_jsonl": (
            data_root / "ShareGPT4/sharegpt_gpt4.jsonl"
        ).resolve(),
    }
    for name, expected in expected_data_paths.items():
        if _resolved(data.get(name, "")) != expected:
            raise ValueError(f"reverse long-context data path {name} changed")
    expected_data_values = {
        "multi_news_sample_count": EXPECTED_SAMPLE_COUNTS["multi_news"],
        "multi_news_selection_seed": MULTINEWS_SELECTION_SEED,
        "lmsys_sample_count": EXPECTED_SAMPLE_COUNTS["lmsys"],
        "lmsys_selection_seed": LMSYS_SELECTION_SEED,
        "sharegpt_sample_count": EXPECTED_SAMPLE_COUNTS["sharegpt4"],
        "sharegpt_selection_seed": SHAREGPT4_SELECTION_SEED,
    }
    for name, expected in expected_data_values.items():
        if data.get(name) != expected:
            raise ValueError(f"reverse long-context data contract {name} changed")

    runtime = _require_mapping(raw.get("runtime"), "runtime")
    for role, contract in FORMAL_TOPOLOGY.items():
        if tuple(runtime.get(f"{role}_devices", ())) != contract["devices"]:
            raise ValueError(f"reverse long-context {role} devices changed")
        if tuple(runtime.get(f"{role}_layer_counts", ())) != contract["layer_counts"]:
            raise ValueError(f"reverse long-context {role} layer counts changed")
    expected_runtime = {
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "mapper_token_chunk": MAPPER_TOKEN_CHUNK,
        "max_length": MAX_LENGTH,
        "dialog_reference_score_cap_tokens": DIALOG_REFERENCE_SCORE_CAP_TOKENS,
    }
    for name, expected in expected_runtime.items():
        if runtime.get(name) != expected:
            raise ValueError(f"reverse long-context runtime {name} changed")

    tasks = _require_mapping(raw.get("tasks"), "tasks")
    if tuple(tasks) != TASKS:
        raise ValueError("reverse long-context task config order/content changed")
    for task in TASKS:
        config = _require_mapping(tasks.get(task), f"tasks.{task}")
        if int(config.get("max_gen_tokens", -1)) != EXPECTED_CAPS[task]:
            raise ValueError(f"reverse long-context {task} cap changed")
        if config.get("prompt_mode") != "qwen_chat_no_think":
            raise ValueError(f"reverse long-context {task} prompt mode changed")


def _compatible_forward_validator_raw(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Make only the fields hard-coded by the shared validator compatible.

    The result is private and is never written to a manifest.  The returned
    validated configuration is restored to the reverse contract immediately
    afterward.  This retains every generic validation rule without changing
    the shared runner that existing forward receipts hash.
    """

    compatible = dict(raw)
    data = dict(_require_mapping(raw.get("data"), "data"))
    data["lmsys_sample_count"] = 200
    data["sharegpt_sample_count"] = 200
    compatible["data"] = data
    runtime = dict(_require_mapping(raw.get("runtime"), "runtime"))
    runtime.update(
        {
            "source_devices": [6, 7],
            "source_layer_counts": [20, 20],
            "target_devices": [0, 1, 2, 3, 4, 5],
            "target_layer_counts": [10, 11, 11, 11, 11, 10],
        }
    )
    compatible["runtime"] = runtime
    return compatible


def _strict_mapper_identity(path: str | Path) -> dict[str, Any]:
    identity = reverse_mapper_manifest_identity(path)
    manifest = Path(str(identity["manifest"])).resolve()
    if manifest != MAPPER_MANIFEST.resolve():
        raise ValueError("reverse long-context mapper path changed")
    if identity.get("manifest_sha256") != EXPECTED_MAPPER_MANIFEST_SHA256:
        raise ValueError("reverse long-context mapper manifest SHA256 changed")
    if identity.get("fit_fingerprint") != EXPECTED_MAPPER_FIT_FINGERPRINT:
        raise ValueError("reverse long-context mapper fit fingerprint changed")
    if identity.get("target_layer_count") != 40:
        raise ValueError("reverse long-context mapper target coverage changed")
    return identity


def _validate_smoke_endpoint_identities(
    task: str,
    samples: Sequence[Any],
    manifest: Mapping[str, Any] | None = None,
) -> None:
    by_index = {int(sample.index): sample for sample in samples}
    source_indices: dict[int, int] = {}
    if task in ("lmsys", "sharegpt4"):
        if not isinstance(manifest, Mapping) or not isinstance(
            manifest.get("selected"), list
        ):
            raise ValueError(f"{task} smoke endpoint source manifest is missing")
        source_indices = {
            int(row["evaluation_index"]): int(row["source_order"])
            for row in manifest["selected"]
            if isinstance(row, Mapping)
        }
    for endpoint in SMOKE_ENDPOINTS[task]:
        index = int(endpoint["evaluation_index"])
        sample = by_index.get(index)
        if sample is None:
            raise ValueError(f"frozen smoke endpoint is absent: {task}/{index}")
        if (
            str(sample.sample_id) != endpoint["sample_id"]
            or str(sample.record_sha256) != endpoint["record_sha256"]
        ):
            raise ValueError(f"frozen smoke endpoint identity changed: {task}/{index}")
        observed_source = source_indices.get(index, index)
        if observed_source != endpoint["source_index"]:
            raise ValueError(
                f"frozen smoke endpoint source index changed: {task}/{index}"
            )


def validate_generated_token_evidence(
    payload: Mapping[str, Any],
    *,
    tokenizer: Any,
    expected_cap: int,
    token_ids_sha256: Any,
) -> None:
    """Validate the exact continuation tokens persisted in one checkpoint."""

    generation = payload.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("checkpoint lacks generation evidence")
    expected_keys = {
        "requested_max_tokens",
        "generated_tokens",
        "stop_reason",
        "cap_hit",
        "generated_token_ids",
        "generated_token_ids_sha256",
    }
    if set(generation) != expected_keys:
        raise ValueError("generation evidence schema changed")
    token_ids = generation.get("generated_token_ids")
    if not isinstance(token_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) for token in token_ids
    ):
        raise ValueError("generated_token_ids must be an exact integer list")
    # Qwen3-14B's output projection has 151,936 rows while its tokenizer lists
    # 151,669 ordinary entries.  IDs in that reserved model-output tail are
    # legal even when they decode to an empty string, so the model config --
    # not ``len(tokenizer)`` -- is the frozen upper bound.
    if any(token < 0 or token >= EXPECTED_TARGET_VOCAB_SIZE for token in token_ids):
        raise ValueError("generated token ID is outside the target model vocabulary")
    if generation.get("requested_max_tokens") != expected_cap:
        raise ValueError("generation cap changed")
    if generation.get("generated_tokens") != len(token_ids):
        raise ValueError("generated token count/list length changed")
    if generation.get("generated_token_ids_sha256") != token_ids_sha256(token_ids):
        raise ValueError("generated token IDs disagree with their SHA256")
    stop_reason = generation.get("stop_reason")
    if stop_reason not in {"eos", "max_tokens"}:
        raise ValueError("until=[] permits only eos or max_tokens")
    cap_hit = generation.get("cap_hit")
    if not isinstance(cap_hit, bool):
        raise ValueError("generation cap_hit must be boolean")
    if stop_reason == "max_tokens":
        if len(token_ids) != expected_cap or cap_hit is not True:
            raise ValueError("max_tokens stop requires an exact full-cap token list")
    elif len(token_ids) >= expected_cap or cap_hit is not False:
        raise ValueError("eos stop requires a shorter-than-cap token list")
    try:
        decoded = str(tokenizer.decode(token_ids, skip_special_tokens=True))
    except TypeError:
        decoded = str(tokenizer.decode(token_ids))
    if payload.get("prediction") != decoded:
        raise ValueError("prediction is not the exact decode of generated_token_ids")


def _extension_code_provenance(original_builder: Any) -> dict[str, Any]:
    evidence = dict(original_builder())
    reverse_files: dict[str, dict[str, Any]] = {}
    for relative in REVERSE_RUNTIME_CODE_FILES:
        path = EXPERIMENT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        reverse_files[relative] = {
            "sha256": sha256_file(path),
            "nbytes": path.stat().st_size,
        }
    extension_files: dict[str, dict[str, Any]] = {}
    for relative in CODE_IDENTITY_FILES:
        path = EXPERIMENT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        extension_files[relative] = {
            "sha256": sha256_file(path),
            "nbytes": path.stat().st_size,
        }
    evidence["reverse_direction"] = DIRECTION
    evidence["reverse_runtime_files"] = reverse_files
    evidence["long_context_extension_files"] = extension_files
    base_freeze = EXPERIMENT_ROOT / "artifacts/inputs/code_freeze.json"
    if not base_freeze.is_file():
        raise FileNotFoundError(base_freeze)
    evidence["reverse_base_code_freeze"] = {
        "path": str(base_freeze.resolve()),
        "sha256": sha256_file(base_freeze),
        "nbytes": base_freeze.stat().st_size,
    }
    # Local import avoids a module cycle while making the overlay mandatory.
    from .freeze import validate_code_freeze

    frozen = validate_code_freeze()
    overlay = EXTENSION_ROOT / "code_freeze.json"
    sidecar = overlay.with_name(f"{overlay.name}.sha256.json")
    evidence["long_context_extension_code_freeze"] = {
        "path": str(overlay.resolve()),
        "sha256": sha256_file(overlay),
        "nbytes": overlay.stat().st_size,
        "sidecar": {
            "path": str(sidecar.resolve()),
            "sha256": sha256_file(sidecar),
            "nbytes": sidecar.stat().st_size,
        },
        "inventory_digest": frozen["inventory_digest"],
    }
    return evidence


def configured_base_runner() -> Any:
    """Return the shared runner after applying every sealed reverse override."""

    base = _load_base_runner()
    original_validate = base.validate_configuration
    original_load = base.load_qwen3
    original_build_definition = base.build_run_definition
    original_code_provenance = base.code_provenance
    original_metric_provenance = base.metric_implementation_provenance
    original_evaluate_one = base.evaluate_one
    original_checkpoint_complete = base.checkpoint_complete
    original_load_hotpotqa = base.load_hotpotqa_samples
    original_load_multi_news = base.load_multi_news_samples
    original_load_lmsys = base.load_lmsys_samples
    original_load_sharegpt = base.load_sharegpt_samples
    selection_base = SimpleNamespace(
        load_multi_news_samples=original_load_multi_news,
        load_lmsys_samples=original_load_lmsys,
        load_sharegpt_samples=original_load_sharegpt,
    )

    # Several checkpoint validators consult these module constants directly.
    base.DEFAULT_CONFIG = CONFIG_PATH
    base.TASKS = TASKS
    base.EXPECTED_CAPS = dict(EXPECTED_CAPS)

    def validate_reverse(raw: Mapping[str, Any], args: Any) -> dict[str, Any]:
        _validate_raw_extension_contract(raw, args)
        configuration = original_validate(
            _compatible_forward_validator_raw(raw), args
        )
        configuration.update(
            {
                "source_devices": FORMAL_TOPOLOGY["source"]["devices"],
                "source_layer_counts": FORMAL_TOPOLOGY["source"]["layer_counts"],
                "target_devices": FORMAL_TOPOLOGY["target"]["devices"],
                "target_layer_counts": FORMAL_TOPOLOGY["target"]["layer_counts"],
                "lmsys_sample_count": EXPECTED_SAMPLE_COUNTS["lmsys"],
                "sharegpt_sample_count": EXPECTED_SAMPLE_COUNTS["sharegpt4"],
                "multi_news_sample_count": EXPECTED_SAMPLE_COUNTS["multi_news"],
                "multi_news_selection_seed": MULTINEWS_SELECTION_SEED,
                "direction": DIRECTION,
            }
        )
        # Prove that the private compatibility shim cannot leak into execution.
        for role, contract in FORMAL_TOPOLOGY.items():
            if tuple(configuration[f"{role}_devices"]) != contract["devices"]:
                raise AssertionError(f"{role} device restoration failed")
            if tuple(configuration[f"{role}_layer_counts"]) != contract["layer_counts"]:
                raise AssertionError(f"{role} layer-count restoration failed")
        return configuration

    def reverse_load(
        model_path: str | Path, *, role: str | None = None, **kwargs: Any
    ) -> Any:
        if role not in ("source", "target"):
            raise ValueError(f"invalid semantic role: {role!r}")
        contract = FORMAL_TOPOLOGY[role]
        if Path(model_path).resolve() != Path(contract["model"]).resolve():
            raise ValueError(f"reverse long-context {role} model changed")
        if tuple(int(value) for value in kwargs.get("devices", ())) != contract["devices"]:
            raise ValueError(f"reverse long-context {role} devices changed")
        supplied_counts = kwargs.pop("layer_counts", None)
        if supplied_counts is not None and tuple(supplied_counts) != contract["layer_counts"]:
            raise ValueError(f"reverse long-context {role} layer counts changed")
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

    def reverse_output(_root: Path, supplied: Path | None) -> Path:
        if supplied is None:
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
            output = EXPERIMENT_ROOT / "runs" / f"{stamp}_{EXTENSION_VERSION}"
            output.mkdir(parents=True, exist_ok=False)
            return output
        output = output_within_experiment(supplied)
        output.mkdir(parents=True, exist_ok=True)
        return output

    def load_multi_news(
        source_path: str | Path, target_path: str | Path
    ) -> tuple[list[Any], dict[str, Any]]:
        samples, manifest = extension_selection.load_multi_news_bottom500(
            selection_base, source_path, target_path
        )
        _validate_smoke_endpoint_identities("multi_news", samples, manifest)
        return samples, manifest

    def load_lmsys(
        parquet_dir: str | Path, *, sample_count: int, selection_seed: str
    ) -> tuple[list[Any], dict[str, Any]]:
        samples, manifest = extension_selection.load_lmsys_bottom500(
            selection_base,
            parquet_dir,
            sample_count=sample_count,
            selection_seed=selection_seed,
        )
        _validate_smoke_endpoint_identities("lmsys", samples, manifest)
        return samples, manifest

    def load_sharegpt(
        jsonl_path: str | Path, *, sample_count: int, selection_seed: str
    ) -> tuple[list[Any], dict[str, Any]]:
        samples, manifest = extension_selection.load_sharegpt_bottom500(
            selection_base,
            jsonl_path,
            sample_count=sample_count,
            selection_seed=selection_seed,
        )
        _validate_smoke_endpoint_identities("sharegpt4", samples, manifest)
        return samples, manifest

    def load_hotpotqa(
        json_path: str | Path,
    ) -> tuple[list[Any], dict[str, Any]]:
        samples, manifest = original_load_hotpotqa(json_path)
        expected = EXPECTED_DATA_IDENTITIES["hotpotqa"]
        if len(samples) != EXPECTED_SAMPLE_COUNTS["hotpotqa"]:
            raise ValueError("HotpotQA sample count changed")
        if manifest.get("rows") != EXPECTED_SAMPLE_COUNTS["hotpotqa"]:
            raise ValueError("HotpotQA manifest row count changed")
        if manifest.get("sha256") != expected["file_sha256"]:
            raise ValueError("HotpotQA file SHA256 changed")
        if (
            manifest.get("sample_identity_sha256")
            != expected["sample_identity_sha256"]
        ):
            raise ValueError("HotpotQA sample identity changed")
        _validate_smoke_endpoint_identities("hotpotqa", samples, manifest)
        return samples, dict(manifest)

    def sample_metrics(
        task: str, prediction: str, references: Sequence[str]
    ) -> dict[str, Any]:
        return extension_metrics.sample_metrics(base, task, prediction, references)

    def evaluate_one_with_tokens(**kwargs: Any) -> dict[str, Any]:
        engine = kwargs["engine"]
        original_generate = engine.generate_token_ids
        captured: list[Any] = []

        def capture_generate(*args: Any, **generation_kwargs: Any) -> Any:
            result = original_generate(*args, **generation_kwargs)
            captured.append(result)
            return result

        had_instance_override = "generate_token_ids" in getattr(engine, "__dict__", {})
        previous_override = getattr(engine, "__dict__", {}).get("generate_token_ids")
        engine.generate_token_ids = capture_generate
        try:
            value = original_evaluate_one(**kwargs)
        finally:
            if had_instance_override:
                engine.generate_token_ids = previous_override
            else:
                delattr(engine, "generate_token_ids")
        if len(captured) != 1:
            raise AssertionError("evaluate_one must perform exactly one generation")
        generated = captured[0]
        token_ids = [int(token) for token in generated.generated_token_ids]
        value["generation"]["generated_token_ids"] = token_ids
        validate_generated_token_evidence(
            value,
            tokenizer=kwargs["tokenizer"],
            expected_cap=int(kwargs["task_config"]["max_gen_tokens"]),
            token_ids_sha256=base.token_ids_sha256,
        )
        return value

    def checkpoint_complete_with_tokens(*args: Any, **kwargs: Any) -> bool:
        if not original_checkpoint_complete(*args, **kwargs):
            return False
        path = Path(args[0] if args else kwargs["path"])
        try:
            payload = read_json(path)
            validate_generated_token_evidence(
                payload,
                tokenizer=kwargs["tokenizer"],
                expected_cap=EXPECTED_CAPS[str(kwargs["task"])],
                token_ids_sha256=base.token_ids_sha256,
            )
            target_checkpoint = kwargs.get("target_checkpoint")
            if target_checkpoint is not None:
                validate_generated_token_evidence(
                    target_checkpoint,
                    tokenizer=kwargs["tokenizer"],
                    expected_cap=EXPECTED_CAPS[str(kwargs["task"])],
                    token_ids_sha256=base.token_ids_sha256,
                )
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True

    def metric_provenance() -> dict[str, Any]:
        metric_base = SimpleNamespace(
            metric_implementation_provenance=original_metric_provenance
        )
        return extension_metrics.metric_implementation_provenance(metric_base)

    def build_reverse_definition(**kwargs: Any) -> dict[str, Any]:
        definition = original_build_definition(**kwargs)
        args = kwargs["args"]
        formal = args.limit is None and args.sample_indices is None
        smoke = bool(
            args.limit is None
            and args.sample_indices is not None
            and {
                task: tuple(indices)
                for task, indices in args.sample_indices.items()
            }
            == SMOKE_SAMPLE_INDICES
        )
        if formal:
            sample_scope = (
                "hotpotqa_full7405_multinews_bottom500_lmsys_bottom500_"
                "sharegpt4_bottom500"
            )
            expected_counts = dict(EXPECTED_SAMPLE_COUNTS)
            expected_results = EXPECTED_RESULT_COUNT
        elif smoke:
            sample_scope = "frozen_explicit_two_indices_per_task_not_formal"
            expected_counts = {
                task: len(indices) for task, indices in SMOKE_SAMPLE_INDICES.items()
            }
            expected_results = EXPECTED_SMOKE_RESULT_COUNT
        else:
            sample_scope = "debug_subset_not_formal"
            expected_counts = None
            expected_results = None
        definition.update(
            {
                "protocol": PROTOCOL_NAME,
                "direction": DIRECTION,
                "paper_bit_exact": PAPER_BIT_EXACT,
                "paper_scope_note": PAPER_SCOPE_NOTE,
                "sample_scope": sample_scope,
                "expected_sample_counts": expected_counts,
                "expected_atomic_result_count": expected_results,
                "frozen_smoke_sample_indices": {
                    task: list(indices)
                    for task, indices in SMOKE_SAMPLE_INDICES.items()
                },
                "frozen_smoke_endpoints": {
                    task: [dict(endpoint) for endpoint in endpoints]
                    for task, endpoints in SMOKE_ENDPOINTS.items()
                },
                "target_generation_vocabulary_size": EXPECTED_TARGET_VOCAB_SIZE,
                "semantic_modes": {
                    "target_standalone": "Qwen3-14B standalone",
                    "mapped": "Qwen3-32B prefix KV mapped through production k20 into Qwen3-14B",
                },
                "formal_topology": {
                    role: {
                        "model": str(contract["model"]),
                        "devices": list(contract["devices"]),
                        "layer_counts": list(contract["layer_counts"]),
                    }
                    for role, contract in FORMAL_TOPOLOGY.items()
                },
                "frozen_data_expectations": EXPECTED_DATA_IDENTITIES,
                "hotpotqa_cap_change": {
                    "max_gen_tokens": EXPECTED_CAPS["hotpotqa"],
                    "previous_forward_cap_for_context_only": 512,
                    "reason": "user requested a larger cap to reduce unfinished-answer truncation",
                },
            }
        )
        definition["metrics"].update(
            {
                "summarization": (
                    "rouge==1.0.1 summary-level ROUGE-1/ROUGE-2/ROUGE-L F1 "
                    "against the official MultiNews test reference; ROUGE-L is primary"
                ),
                "multinews_not_accuracy": (
                    "MultiNews is abstractive summarization and has a gold reference, "
                    "but no classification-style accuracy metric"
                ),
            }
        )
        return definition

    base.validate_configuration = validate_reverse
    base.load_qwen3 = reverse_load
    base.MappedPrefixCacheBuilder = ReverseMappedPrefixCacheBuilder
    base.model_artifact_identity = reverse_model_artifact_identity
    base.mapper_artifact_identity = _strict_mapper_identity
    base.load_hotpotqa_samples = load_hotpotqa
    base.load_multi_news_samples = load_multi_news
    base.load_lmsys_samples = load_lmsys
    base.load_sharegpt_samples = load_sharegpt
    base._sample_metrics = sample_metrics
    base.evaluate_one = evaluate_one_with_tokens
    base.checkpoint_complete = checkpoint_complete_with_tokens
    base.metric_implementation_provenance = metric_provenance
    base.code_provenance = lambda: _extension_code_provenance(
        original_code_provenance
    )
    base.build_run_definition = build_reverse_definition
    base._make_output_directory = reverse_output
    return base


def run_extension_evaluation(extra_argv: Sequence[str] = ()) -> int:
    """Run the shared evaluator with sealed reverse-direction semantics."""

    base = configured_base_runner()
    argv = ["--config", str(CONFIG_PATH), *map(str, extra_argv)]
    return int(base.main(argv))


__all__ = [
    "_load_base_runner",
    "_strict_mapper_identity",
    "_validate_smoke_endpoint_identities",
    "_validate_raw_extension_contract",
    "configured_base_runner",
    "run_extension_evaluation",
    "validate_generated_token_evidence",
]
