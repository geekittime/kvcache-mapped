"""Auditable result-level numeric re-scoring for completed GSM8K evaluations.

This module deliberately does *not* replace the canonical lm-eval GSM8K
metrics.  The paper reports only ``acc`` and does not publish its exact answer
filter.  Consequently, the metric implemented here is a separately named
diagnostic: extract one final numeric answer without consulting the gold
target, then compare the extracted and gold values by exact rational equality.

The input is the task checkpoint written by :mod:`scripts.evaluate`.  Strict
validation preserves the canonical strict/flexible rows and independently
recomputes both published lm-eval values before the diagnostic is allowed to
run.  Reports retain input checksums, request identity, per-sample extraction
evidence, and checksums for every output artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata

from .io import atomic_write_json, sha256_file


SCHEMA_VERSION = 1
METRIC_NAME = "exact_match,numeric-final-result-level-diagnostic-v1"
TASK_NAME = "gsm8k_cot"
STRICT_FILTER = "strict-match"
FLEXIBLE_FILTER = "flexible-extract"
EXPECTED_SAMPLE_COUNT = 1_319
EXPECTED_MAX_GEN_TOKENS = 1_024
EXPECTED_GENERATION_KWARGS = {
    "do_sample": False,
    "until": ["Q:", "</s>", "<|im_end|>"],
}
EXPECTED_STRICT_FILTER = {
    "name": STRICT_FILTER,
    "filter": [
        {
            "function": "regex",
            "regex_pattern": r"The answer is (\-?[0-9\.\,]+).",
        },
        {"function": "take_first"},
    ],
}
EXPECTED_FLEXIBLE_FILTER = {
    "name": FLEXIBLE_FILTER,
    "filter": [
        {
            "function": "regex",
            "group_select": -1,
            "regex_pattern": r"(-?[$0-9.,]{2,})|(-?[0-9]+)",
        },
        {"function": "take_first"},
    ],
}
EXPECTED_METRIC = {
    "aggregation": "mean",
    "higher_is_better": True,
    "ignore_case": True,
    "ignore_punctuation": False,
    "metric": "exact_match",
    "regexes_to_ignore": [",", r"\$", r"(?s).*#### ", r"\.$"],
}
PAPER_DISCLOSURE = (
    "The paper reports GSM8K accuracy but does not publish its exact answer "
    "filter. This local numeric-final metric is a result-level alignment "
    "diagnostic; it does not replace lm-eval strict-match or flexible-extract "
    "and is not asserted to be the authors' hidden filter."
)


class GSM8KNumericDiagnosticError(ValueError):
    """Raised when an input checkpoint cannot be audited without ambiguity."""


# These two patterns are the complete frozen v1 extraction grammar.  They are
# intentionally identical to the rule used for the pre-implementation audit
# of all 1,319 mapped responses.  Any extension requires a new metric version.
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|:|=)\s*"
    r"(?:\*\*)?(?:\\boxed\{)?(?:\$\s*)?"
    r"(?P<number>\-?[0-9][0-9.,]*)",
    flags=re.IGNORECASE,
)
_FALLBACK_NUMBER_RE = re.compile(r"-?[$0-9.,]{2,}|-?[0-9]+")

# Exact local transcription of the pinned lm-eval v0.4.12 GSM8K filters.  The
# implementation below is intentionally tiny, stable, and regression-compared
# with the persisted filtered responses for every sample.
_LM_EVAL_STRICT_RE = re.compile(r"The answer is (\-?[0-9\.\,]+).")
_LM_EVAL_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


@dataclass(frozen=True)
class NumericValue:
    """One exact rational value parsed from model or target text."""

    raw: str
    canonical: str
    numerator: int
    denominator: int

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class Extraction:
    """Gold-independent extraction evidence for a single completion."""

    status: str
    strategy: str | None
    value: NumericValue | None
    match_start: int | None
    match_end: int | None
    matched_text: str | None
    context: str | None


def _normalise_model_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\N{MINUS SIGN}", "-")
    return value


def _decimal_fraction(text: str) -> Fraction:
    cleaned = (
        text.replace(",", "")
        .replace("$", "")
        .replace("**", "")
        .strip()
        .rstrip(".}")
        .strip()
    )
    try:
        decimal = Decimal(cleaned)
    except InvalidOperation as exc:
        raise GSM8KNumericDiagnosticError(f"unsupported decimal token: {text!r}") from exc
    if not decimal.is_finite():
        raise GSM8KNumericDiagnosticError(f"non-finite decimal token: {text!r}")
    return Fraction(decimal)


def parse_numeric_value(text: str) -> NumericValue:
    """Parse one supported numeric token into an exact rational value.

    This follows the frozen audit rule: remove commas, dollar/Markdown markers,
    whitespace, and a terminal sentence period/box brace, then compare finite
    ``Decimal`` values.  Fractions, expressions, and units are not evaluated.
    """

    raw = text
    value = _normalise_model_text(text).strip()
    fraction = _decimal_fraction(value)
    canonical = (
        str(fraction.numerator)
        if fraction.denominator == 1
        else f"{fraction.numerator}/{fraction.denominator}"
    )
    return NumericValue(
        raw=raw,
        canonical=canonical,
        numerator=fraction.numerator,
        denominator=fraction.denominator,
    )


def _context(text: str, start: int, end: int, radius: int = 96) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _extraction_from_match(
    text: str, *, start: int, end: int, matched: str, strategy: str
) -> Extraction:
    try:
        value = parse_numeric_value(matched)
    except GSM8KNumericDiagnosticError:
        # The deliberately broad frozen fallback regex can capture punctuation
        # such as ``$.``.  Preserve it as per-sample evidence and score it
        # incorrect; one malformed model candidate must not invalidate the
        # otherwise auditable checkpoint.
        return Extraction(
            status="invalid_numeric",
            strategy=strategy,
            value=None,
            match_start=start,
            match_end=end,
            matched_text=matched,
            context=_context(text, start, end),
        )
    return Extraction(
        status="extracted",
        strategy=strategy,
        value=value,
        match_start=start,
        match_end=end,
        matched_text=matched,
        context=_context(text, start, end),
    )


def extract_final_numeric_answer(response: str) -> Extraction:
    """Extract one final numeric answer without access to the gold target.

    The last explicit ``[the] [final] answer is/:/=`` numeric marker is used if
    present.  Otherwise, the final token matched by the frozen lm-eval-like
    fallback number regex is used.  Gold data is never supplied to or inspected
    by this function.
    """

    if not isinstance(response, str):
        raise GSM8KNumericDiagnosticError("response must be a string")
    text = _normalise_model_text(response)
    explicit = list(_EXPLICIT_ANSWER_RE.finditer(text))
    if explicit:
        match = explicit[-1]
        start, end = match.span("number")
        matched = match.group("number")
        return _extraction_from_match(
            text,
            start=start,
            end=end,
            matched=matched,
            strategy="last_explicit_answer_marker",
        )

    fallback = list(_FALLBACK_NUMBER_RE.finditer(text))
    if fallback:
        match = fallback[-1]
        start, end = match.span()
        matched = match.group(0)
        return _extraction_from_match(
            text,
            start=start,
            end=end,
            matched=matched,
            strategy="fallback_last_numeric_match",
        )
    return Extraction(
        status="missing",
        strategy=None,
        value=None,
        match_start=None,
        match_end=None,
        matched_text=None,
        context=None,
    )


def canonical_lm_eval_filter(response: str, filter_name: str) -> str:
    """Re-run the pinned lm-eval v0.4.12 GSM8K extraction filter.

    This mirrors ``RegexFilter`` followed by ``TakeFirstFilter`` for a single
    deterministic response repeat.  Strict chooses the first regex match;
    flexible uses ``group_select=-1`` and therefore chooses the final match.
    """

    if not isinstance(response, str):
        response = ""
    if filter_name == STRICT_FILTER:
        matches = _LM_EVAL_STRICT_RE.findall(response)
        return matches[0].strip() if matches else "[invalid]"
    if filter_name == FLEXIBLE_FILTER:
        matches = _LM_EVAL_FLEXIBLE_RE.findall(response)
        if not matches:
            return "[invalid]"
        selected = matches[-1]
        if isinstance(selected, tuple):
            nonempty = [item for item in selected if item]
            return nonempty[0].strip() if nonempty else "[invalid]"
        return str(selected).strip()
    raise GSM8KNumericDiagnosticError(f"unsupported lm-eval filter {filter_name!r}")


def canonical_lm_eval_exact_match(prediction: str, target: str) -> int:
    """Re-run the pinned per-sample exact-match normalization and comparison."""

    if not isinstance(prediction, str) or not isinstance(target, str):
        raise GSM8KNumericDiagnosticError("lm-eval exact-match inputs must be strings")
    normalized_prediction = prediction
    normalized_target = target
    for pattern in EXPECTED_METRIC["regexes_to_ignore"]:
        normalized_prediction = re.sub(pattern, "", normalized_prediction)
        normalized_target = re.sub(pattern, "", normalized_target)
    return int(normalized_prediction.lower() == normalized_target.lower())


def _canonical_digest(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GSM8KNumericDiagnosticError(f"cannot canonicalize JSON value: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GSM8KNumericDiagnosticError(f"{label} must be a JSON object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise GSM8KNumericDiagnosticError(f"{label} must be a JSON list")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(value)


def _read_strict_json(path: Path, label: str = "JSON") -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise GSM8KNumericDiagnosticError(f"cannot read strict JSON {path}: {exc}") from exc
    return _mapping(value, label)


def _raw_response(sample: Mapping[str, Any], label: str) -> str:
    resps = _sequence(sample.get("resps"), f"{label}.resps")
    if len(resps) != 1:
        raise GSM8KNumericDiagnosticError(f"{label}.resps must have one repeat")
    repeat = _sequence(resps[0], f"{label}.resps[0]")
    if len(repeat) != 1 or not isinstance(repeat[0], str):
        raise GSM8KNumericDiagnosticError(f"{label}.resps[0] must contain one string")
    return repeat[0]


def _validate_pair(
    strict: Mapping[str, Any], flexible: Mapping[str, Any], *, doc_id: int
) -> None:
    for field in (
        "doc_id",
        "doc_hash",
        "prompt_hash",
        "target_hash",
        "target",
        "doc",
        "arguments",
        "resps",
    ):
        if strict.get(field) != flexible.get(field):
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} strict/flexible mismatch in {field}"
            )
    if strict.get("filter") != STRICT_FILTER:
        raise GSM8KNumericDiagnosticError(f"doc_id={doc_id} missing strict row")
    if flexible.get("filter") != FLEXIBLE_FILTER:
        raise GSM8KNumericDiagnosticError(f"doc_id={doc_id} missing flexible row")


def _metric_bit(sample: Mapping[str, Any], label: str) -> int:
    value = sample.get("exact_match")
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise GSM8KNumericDiagnosticError(f"{label}.exact_match must be finite 0/1")
    if float(value) not in (0.0, 1.0):
        raise GSM8KNumericDiagnosticError(f"{label}.exact_match must be 0 or 1")
    return int(value)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validated_run_provenance(
    checkpoint_path: Path, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    run_dir: Path | None = None
    required_names = ("run_definition.json", "status.json", "summary.json")
    for parent in checkpoint_path.parents:
        if not (parent / "run_definition.json").is_file():
            continue
        run_dir = parent.resolve()
        break
    if run_dir is None:
        raise GSM8KNumericDiagnosticError(
            "checkpoint must be inside a run directory containing run_definition.json"
        )
    missing = [name for name in required_names if not (run_dir / name).is_file()]
    if missing:
        raise GSM8KNumericDiagnosticError(f"run provenance files are missing: {missing}")

    run_definition_path = run_dir / "run_definition.json"
    status_path = run_dir / "status.json"
    summary_path = run_dir / "summary.json"
    run_definition = _read_strict_json(run_definition_path, "run_definition")
    status = _read_strict_json(status_path, "status")
    summary = _read_strict_json(summary_path, "summary")

    declared_fingerprint = run_definition.get("request_fingerprint")
    if not isinstance(declared_fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", declared_fingerprint
    ):
        raise GSM8KNumericDiagnosticError("run_definition request_fingerprint is invalid")
    fingerprint_input = dict(run_definition)
    fingerprint_input.pop("request_fingerprint", None)
    recomputed_fingerprint = _canonical_digest(fingerprint_input)
    if recomputed_fingerprint != declared_fingerprint:
        raise GSM8KNumericDiagnosticError(
            "run_definition request_fingerprint does not match canonical content"
        )
    if checkpoint.get("request_fingerprint") != declared_fingerprint:
        raise GSM8KNumericDiagnosticError(
            "checkpoint request_fingerprint does not match run_definition"
        )

    mode = checkpoint.get("mode")
    if mode not in {"source_standalone", "target_standalone", "target_oracle", "mapped"}:
        raise GSM8KNumericDiagnosticError(f"unsupported checkpoint mode {mode!r}")
    modes = _sequence(run_definition.get("modes"), "run_definition.modes")
    tasks = _sequence(run_definition.get("tasks"), "run_definition.tasks")
    if mode not in modes or TASK_NAME not in tasks:
        raise GSM8KNumericDiagnosticError("checkpoint mode/task is absent from run_definition")
    required_run_fields = {
        "schema_version": 1,
        "max_gen_tokens": EXPECTED_MAX_GEN_TOKENS,
        "limit": None,
        "log_samples": True,
        "completion_mode": True,
        "apply_chat_template": False,
        "batch_size": 1,
    }
    for field, expected in required_run_fields.items():
        if run_definition.get(field) != expected:
            raise GSM8KNumericDiagnosticError(
                f"run_definition.{field}={run_definition.get(field)!r}, expected {expected!r}"
            )
    benchmarks = _mapping(run_definition.get("benchmarks"), "run_definition.benchmarks")
    benchmark = _mapping(benchmarks.get(TASK_NAME), f"benchmarks.{TASK_NAME}")
    expected_benchmark = {
        "metric": "exact_match",
        "metric_filter": STRICT_FILTER,
        "mode": "greedy_generation_cot",
        "num_fewshot": 8,
    }
    if dict(benchmark) != expected_benchmark:
        raise GSM8KNumericDiagnosticError("run_definition GSM8K benchmark contract mismatch")

    config_path_value = run_definition.get("config_path")
    config_sha = run_definition.get("config_sha256")
    if not isinstance(config_path_value, str) or not isinstance(config_sha, str):
        raise GSM8KNumericDiagnosticError("run_definition config identity is invalid")
    config_path = Path(config_path_value).resolve()
    if not config_path.is_file() or sha256_file(config_path) != config_sha:
        raise GSM8KNumericDiagnosticError(
            "current config file is missing or disagrees with run_definition SHA256"
        )

    if status.get("status") != "complete" or status.get("failures") != 0:
        raise GSM8KNumericDiagnosticError("run status must be complete with zero failures")
    status_summary = _mapping(status.get("summary"), "status.summary")
    if _canonical_digest(status_summary) != _canonical_digest(summary):
        raise GSM8KNumericDiagnosticError("status embedded summary disagrees with summary.json")
    if summary.get("request_fingerprint") != declared_fingerprint:
        raise GSM8KNumericDiagnosticError("summary request_fingerprint mismatch")
    summary_metrics = _mapping(summary.get("metrics"), "summary.metrics")
    mode_metrics = _mapping(summary_metrics.get(str(mode)), f"summary.metrics.{mode}")
    checkpoint_metric = checkpoint.get("primary_metric_value")
    if mode_metrics.get(TASK_NAME) != checkpoint_metric:
        raise GSM8KNumericDiagnosticError(
            "summary GSM8K metric disagrees with checkpoint primary metric"
        )

    files = {
        name: {
            "path": str((run_dir / name).resolve()),
            "sha256": sha256_file(run_dir / name),
        }
        for name in required_names
    }
    return {
        "run_dir": str(run_dir),
        "request_fingerprint_declared": declared_fingerprint,
        "request_fingerprint_recomputed": recomputed_fingerprint,
        "request_fingerprint_verified": True,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "config_sha256_verified": True,
        "mode_task_contract_verified": True,
        "max_gen_tokens": EXPECTED_MAX_GEN_TOKENS,
        "full_dataset_limit": None,
        "files": files,
    }


def _validate_resolved_task_contract(checkpoint: Mapping[str, Any]) -> None:
    if checkpoint.get("schema_version") != 1:
        raise GSM8KNumericDiagnosticError("checkpoint schema_version must be 1")
    if checkpoint.get("primary_metric") != "exact_match":
        raise GSM8KNumericDiagnosticError("checkpoint primary_metric must be exact_match")
    resolved = _mapping(checkpoint.get("resolved_task_config"), "resolved_task_config")
    configs = _mapping(resolved.get("configs"), "resolved_task_config.configs")
    declared_config_digest = resolved.get("canonical_configs_sha256")
    recomputed_config_digest = _canonical_digest(configs)
    if declared_config_digest != recomputed_config_digest:
        raise GSM8KNumericDiagnosticError(
            "resolved task canonical_configs_sha256 does not match configs"
        )
    raw_result = _mapping(checkpoint.get("result"), "checkpoint.result")
    raw_configs = _mapping(raw_result.get("configs"), "checkpoint.result.configs")
    raw_n_shot = _mapping(raw_result.get("n-shot"), "checkpoint.result.n-shot")
    raw_versions = _mapping(raw_result.get("versions"), "checkpoint.result.versions")
    if _canonical_digest(raw_configs) != recomputed_config_digest:
        raise GSM8KNumericDiagnosticError(
            "resolved task configs disagree with checkpoint.result.configs"
        )
    config = _mapping(configs.get(TASK_NAME), f"resolved configs.{TASK_NAME}")
    required = {
        "num_fewshot": 8,
        "output_type": "generate_until",
        "repeats": 1,
        "test_split": "test",
        "generation_kwargs": EXPECTED_GENERATION_KWARGS,
        "filter_list": [EXPECTED_STRICT_FILTER, EXPECTED_FLEXIBLE_FILTER],
        "metric_list": [EXPECTED_METRIC],
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise GSM8KNumericDiagnosticError(
                f"resolved GSM8K config field {field!r} does not match pinned contract"
            )
    n_shot = _mapping(resolved.get("n_shot"), "resolved_task_config.n_shot")
    versions = _mapping(resolved.get("versions"), "resolved_task_config.versions")
    if n_shot.get(TASK_NAME) != 8 or versions.get(TASK_NAME) != 3.0:
        raise GSM8KNumericDiagnosticError("resolved GSM8K n-shot/version contract mismatch")
    if dict(raw_n_shot) != dict(n_shot) or dict(raw_versions) != dict(versions):
        raise GSM8KNumericDiagnosticError(
            "resolved task n-shot/version disagree with checkpoint.result"
        )


def _validate_result_sample_count(result: Mapping[str, Any], sample_count: int) -> None:
    if sample_count != EXPECTED_SAMPLE_COUNT:
        raise GSM8KNumericDiagnosticError(
            f"full GSM8K diagnostic requires {EXPECTED_SAMPLE_COUNT} samples, got {sample_count}"
        )
    n_samples = _mapping(result.get("n-samples"), "checkpoint.result.n-samples")
    task_counts = _mapping(n_samples.get(TASK_NAME), f"n-samples.{TASK_NAME}")
    if task_counts.get("effective") != EXPECTED_SAMPLE_COUNT or task_counts.get(
        "original"
    ) != EXPECTED_SAMPLE_COUNT:
        raise GSM8KNumericDiagnosticError("result n-samples contract mismatch")


def _stored_filtered_response(sample: Mapping[str, Any], label: str) -> str:
    values = _sequence(sample.get("filtered_resps"), f"{label}.filtered_resps")
    if len(values) != 1 or not isinstance(values[0], str):
        raise GSM8KNumericDiagnosticError(
            f"{label}.filtered_resps must contain exactly one string"
        )
    return values[0]


def _validate_sample_identity(
    sample: Mapping[str, Any], *, doc_id: int
) -> tuple[str, str]:
    label = f"doc_id={doc_id}"
    doc = _mapping(sample.get("doc"), f"{label}.doc")
    question = doc.get("question")
    answer = doc.get("answer")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise GSM8KNumericDiagnosticError(f"{label}.doc question/answer must be strings")
    if set(doc) != {"question", "answer"}:
        raise GSM8KNumericDiagnosticError(f"{label}.doc schema mismatch")
    expected_target = answer.split("####")[-1].strip()
    target = sample.get("target")
    if target != expected_target:
        raise GSM8KNumericDiagnosticError(f"{label}.target disagrees with canonical doc answer")

    expected_doc_hash = _sha256_text(
        # The dataset feature order at evaluation time is question, answer.
        # atomic_write_json later sorted the persisted object keys, so rebuild
        # that original order before reproducing evaluator.py's doc_hash.
        json.dumps(
            {"question": question, "answer": answer},
            indent=2,
            ensure_ascii=False,
        )
    )
    expected_target_hash = _sha256_text(str(target))
    if sample.get("doc_hash") != expected_doc_hash:
        raise GSM8KNumericDiagnosticError(f"{label}.doc_hash mismatch")
    if sample.get("target_hash") != expected_target_hash:
        raise GSM8KNumericDiagnosticError(f"{label}.target_hash mismatch")

    arguments = _sequence(sample.get("arguments"), f"{label}.arguments")
    if len(arguments) != 1:
        raise GSM8KNumericDiagnosticError(f"{label}.arguments must contain one request")
    request = _sequence(arguments[0], f"{label}.arguments[0]")
    if len(request) != 2 or not isinstance(request[0], str):
        raise GSM8KNumericDiagnosticError(f"{label}.request must be [prompt, kwargs]")
    prompt = request[0]
    request_kwargs = _mapping(request[1], f"{label}.generation_kwargs")
    if dict(request_kwargs) != EXPECTED_GENERATION_KWARGS:
        raise GSM8KNumericDiagnosticError(f"{label}.generation kwargs mismatch")
    if sample.get("prompt_hash") != _sha256_text(prompt):
        raise GSM8KNumericDiagnosticError(f"{label}.prompt_hash mismatch")
    if not prompt.endswith(f"Q: {question}\nA:"):
        raise GSM8KNumericDiagnosticError(f"{label}.prompt does not end in its canonical query")
    metrics = _sequence(sample.get("metrics"), f"{label}.metrics")
    if list(metrics) != ["exact_match"]:
        raise GSM8KNumericDiagnosticError(f"{label}.metrics mismatch")
    return str(target), prompt


def _implementation_provenance() -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    project_root = module_path.parents[2]
    relative_files = (
        "src/kvcache_transfer/gsm8k_numeric_diagnostic.py",
        "scripts/rescore_gsm8k_numeric.py",
        "vendor/lm-evaluation-harness/lm_eval/filters/extraction.py",
        "vendor/lm-evaluation-harness/lm_eval/api/metrics.py",
        "vendor/lm-evaluation-harness/lm_eval/evaluator.py",
        "vendor/lm-evaluation-harness/lm_eval/utils.py",
    )
    files: dict[str, dict[str, Any]] = {}
    for relative in relative_files:
        path = project_root / relative
        if not path.is_file():
            raise GSM8KNumericDiagnosticError(
                f"required scorer provenance source is missing: {path}"
            )
        files[relative] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "metric_name": METRIC_NAME,
        "schema_version": SCHEMA_VERSION,
        "project_root": str(project_root),
        "files": files,
        "files_digest": _canonical_digest(files),
        "pinned_lm_eval_contract_locally_transcribed_and_samplewise_verified": True,
    }


def score_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Validate and re-score one completed GSM8K task checkpoint."""

    path = Path(checkpoint_path).resolve()
    checkpoint = _read_strict_json(path)
    if checkpoint.get("status") != "complete":
        raise GSM8KNumericDiagnosticError("checkpoint status must be complete")
    if checkpoint.get("task") != TASK_NAME:
        raise GSM8KNumericDiagnosticError(f"checkpoint task must be {TASK_NAME!r}")
    _validate_resolved_task_contract(checkpoint)
    run_provenance = _validated_run_provenance(path, checkpoint)

    result = _mapping(checkpoint.get("result"), "checkpoint.result")
    result_metrics = _mapping(result.get("results"), "checkpoint.result.results")
    task_metrics = _mapping(result_metrics.get(TASK_NAME), f"results.{TASK_NAME}")
    samples_by_task = _mapping(result.get("samples"), "checkpoint.result.samples")
    samples = _sequence(samples_by_task.get(TASK_NAME), f"samples.{TASK_NAME}")
    if not samples:
        raise GSM8KNumericDiagnosticError("GSM8K samples are empty")

    paired: dict[int, dict[str, Mapping[str, Any]]] = {}
    for index, raw_sample in enumerate(samples):
        sample = _mapping(raw_sample, f"samples[{index}]")
        doc_id = sample.get("doc_id")
        if not isinstance(doc_id, int) or isinstance(doc_id, bool) or doc_id < 0:
            raise GSM8KNumericDiagnosticError(f"samples[{index}].doc_id is invalid")
        filter_name = sample.get("filter")
        if filter_name not in (STRICT_FILTER, FLEXIBLE_FILTER):
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} has unexpected filter {filter_name!r}"
            )
        bucket = paired.setdefault(doc_id, {})
        if filter_name in bucket:
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} has duplicate {filter_name} row"
            )
        bucket[str(filter_name)] = sample

    expected_ids = list(range(len(paired)))
    if sorted(paired) != expected_ids:
        raise GSM8KNumericDiagnosticError("doc_id values must be contiguous from zero")
    _validate_result_sample_count(result, len(paired))

    strict_correct = 0
    flexible_correct = 0
    diagnostic_correct = 0
    missing = 0
    invalid_numeric = 0
    strategies: dict[str, int] = {}
    per_sample: list[dict[str, Any]] = []
    for doc_id in expected_ids:
        bucket = paired[doc_id]
        if set(bucket) != {STRICT_FILTER, FLEXIBLE_FILTER}:
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} must have exactly strict and flexible rows"
            )
        strict = bucket[STRICT_FILTER]
        flexible = bucket[FLEXIBLE_FILTER]
        _validate_pair(strict, flexible, doc_id=doc_id)
        target_raw, _ = _validate_sample_identity(strict, doc_id=doc_id)
        # Pair equality above makes one full identity recomputation sufficient;
        # the flexible row must contain byte-for-byte equal identity fields.
        response = _raw_response(strict, f"doc_id={doc_id}.strict")
        strict_filtered = canonical_lm_eval_filter(response, STRICT_FILTER)
        flexible_filtered = canonical_lm_eval_filter(response, FLEXIBLE_FILTER)
        stored_strict_filtered = _stored_filtered_response(
            strict, f"doc_id={doc_id}.strict"
        )
        stored_flexible_filtered = _stored_filtered_response(
            flexible, f"doc_id={doc_id}.flexible"
        )
        if stored_strict_filtered != strict_filtered:
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} strict filtered response disagrees with raw-response recomputation"
            )
        if stored_flexible_filtered != flexible_filtered:
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} flexible filtered response disagrees with raw-response recomputation"
            )
        strict_bit = canonical_lm_eval_exact_match(strict_filtered, target_raw)
        flexible_bit = canonical_lm_eval_exact_match(flexible_filtered, target_raw)
        stored_strict_bit = _metric_bit(strict, f"doc_id={doc_id}.strict")
        stored_flexible_bit = _metric_bit(flexible, f"doc_id={doc_id}.flexible")
        if strict_bit != stored_strict_bit:
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} strict exact_match bit disagrees with raw-response recomputation"
            )
        if flexible_bit != stored_flexible_bit:
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} flexible exact_match bit disagrees with raw-response recomputation"
            )
        strict_correct += strict_bit
        flexible_correct += flexible_bit

        try:
            target = parse_numeric_value(target_raw)
        except GSM8KNumericDiagnosticError as exc:
            raise GSM8KNumericDiagnosticError(
                f"doc_id={doc_id} target is not one supported numeric value: {exc}"
            ) from exc
        extraction = extract_final_numeric_answer(response)
        if extraction.value is None:
            correct = False
            if extraction.status == "missing":
                missing += 1
                strategy_key = "missing"
            else:
                invalid_numeric += 1
                strategy_key = f"{extraction.strategy}:invalid_numeric"
        else:
            correct = extraction.value.fraction == target.fraction
            strategy_key = str(extraction.strategy)
        diagnostic_correct += int(correct)
        strategies[strategy_key] = strategies.get(strategy_key, 0) + 1
        per_sample.append(
            {
                "doc_id": doc_id,
                "doc_hash": strict.get("doc_hash"),
                "prompt_hash": strict.get("prompt_hash"),
                "target_hash": strict.get("target_hash"),
                "response_sha256": _sha256_text(response),
                "target": asdict(target),
                "extraction": asdict(extraction),
                "numeric_final_correct": correct,
                "lm_eval": {
                    "strict_match_correct": bool(strict_bit),
                    "flexible_extract_correct": bool(flexible_bit),
                    "strict_filtered_response": strict_filtered,
                    "flexible_filtered_response": flexible_filtered,
                    "raw_response_filter_recomputed": True,
                    "persisted_filter_and_bit_matched": True,
                },
            }
        )

    total = len(expected_ids)
    strict_value = strict_correct / total
    flexible_value = flexible_correct / total
    diagnostic_value = diagnostic_correct / total
    embedded_strict = task_metrics.get("exact_match,strict-match")
    embedded_flexible = task_metrics.get("exact_match,flexible-extract")
    for label, embedded, recomputed in (
        ("strict-match", embedded_strict, strict_value),
        ("flexible-extract", embedded_flexible, flexible_value),
    ):
        if not isinstance(embedded, (int, float)) or not math.isfinite(float(embedded)):
            raise GSM8KNumericDiagnosticError(f"embedded {label} metric is not finite")
        if not math.isclose(float(embedded), recomputed, rel_tol=0.0, abs_tol=1e-15):
            raise GSM8KNumericDiagnosticError(
                f"embedded {label}={embedded!r} != recomputed {recomputed!r}"
            )
    sample_len = task_metrics.get("sample_len")
    if sample_len is not None and sample_len != total:
        raise GSM8KNumericDiagnosticError(
            f"embedded sample_len={sample_len!r} != paired sample count {total}"
        )
    primary = checkpoint.get("primary_metric_value")
    if not isinstance(primary, (int, float)) or not math.isclose(
        float(primary), strict_value, rel_tol=0.0, abs_tol=1e-15
    ):
        raise GSM8KNumericDiagnosticError(
            "checkpoint primary_metric_value does not preserve lm-eval strict-match"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "task": TASK_NAME,
        "metric_name": METRIC_NAME,
        "metric_role": "result_level_alignment_diagnostic_only",
        "paper_disclosure": PAPER_DISCLOSURE,
        "implementation": _implementation_provenance(),
        "definition": {
            "gold_independent_extraction": True,
            "precedence": [
                "last case-insensitive [the] [final] answer is/:/= numeric marker",
                "last match of -?[$0-9.,]{2,}|-?[0-9]+ in the completion",
            ],
            "numeric_equivalence": "finite Decimal normalization followed by exact equality",
            "accepted_forms": [
                "signed integer",
                "signed finite decimal",
                "dollar-prefixed integer or finite decimal",
                "comma-grouped thousands",
                "Markdown-bold numeric value",
                "terminal sentence period or closing box brace",
            ],
            "not_supported": [
                "fractions",
                "general arithmetic-expression evaluation",
                "unit conversion",
                "word-number conversion",
                "target-aware candidate selection",
            ],
        },
        "input": {
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "request_fingerprint": checkpoint.get("request_fingerprint"),
            "mode": checkpoint.get("mode"),
            "finished_utc": checkpoint.get("finished_utc"),
            "run_provenance": run_provenance,
        },
        "assurance": {
            "sample_identity_hashes_recomputed": True,
            "raw_response_filters_recomputed": True,
            "per_sample_exact_match_recomputed": True,
            "persisted_filtered_responses_and_bits_verified": True,
            "run_fingerprint_recomputed": True,
            "run_status_summary_config_and_protocol_verified": True,
            "expected_full_sample_count": EXPECTED_SAMPLE_COUNT,
            "expected_max_gen_tokens": EXPECTED_MAX_GEN_TOKENS,
        },
        "counts": {
            "samples": total,
            "raw_filter_rows": len(samples),
            "missing_numeric_prediction": missing,
            "invalid_numeric_prediction": invalid_numeric,
            "extraction_strategies": strategies,
        },
        "preserved_lm_eval_metrics": {
            "strict_match": {
                "correct": strict_correct,
                "total": total,
                "accuracy": strict_value,
                "embedded_accuracy": float(embedded_strict),
                "independently_recomputed": True,
            },
            "flexible_extract": {
                "correct": flexible_correct,
                "total": total,
                "accuracy": flexible_value,
                "embedded_accuracy": float(embedded_flexible),
                "independently_recomputed": True,
            },
        },
        "numeric_final_diagnostic": {
            "correct": diagnostic_correct,
            "total": total,
            "accuracy": diagnostic_value,
        },
        "samples": per_sample,
    }


def _strict_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise GSM8KNumericDiagnosticError(f"report is not strict JSON: {exc}") from exc


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _markdown(report: Mapping[str, Any]) -> str:
    strict = report["preserved_lm_eval_metrics"]["strict_match"]
    flexible = report["preserved_lm_eval_metrics"]["flexible_extract"]
    numeric = report["numeric_final_diagnostic"]
    return "\n".join(
        [
            "# GSM8K final-numeric result-level diagnostic",
            "",
            f"> {report['paper_disclosure']}",
            "",
            f"- Input: `{report['input']['checkpoint_path']}`",
            f"- Input SHA256: `{report['input']['checkpoint_sha256']}`",
            f"- Samples: `{numeric['total']}`",
            f"- lm-eval strict-match (preserved): `{strict['correct']}/{strict['total']} = {100 * strict['accuracy']:.4f}%`",
            f"- lm-eval flexible-extract (preserved): `{flexible['correct']}/{flexible['total']} = {100 * flexible['accuracy']:.4f}%`",
            f"- Local numeric-final diagnostic: `{numeric['correct']}/{numeric['total']} = {100 * numeric['accuracy']:.4f}%`",
            f"- Missing numeric prediction: `{report['counts']['missing_numeric_prediction']}`",
            "",
            "The extractor never receives the gold target. It selects the last explicit answer marker when present and otherwise the last fallback numeric match. Comparison uses exact finite-decimal equality; it does not execute arithmetic or convert units.",
            "",
            "Per-sample extraction evidence is stored in `samples.jsonl`.",
            "",
        ]
    )


def _render_report_files(report: Mapping[str, Any]) -> dict[str, str]:
    samples = _sequence(report.get("samples"), "report.samples")
    summary = dict(report)
    summary.pop("samples", None)
    return {
        "report.json": _strict_json_text(report),
        "summary.json": _strict_json_text(summary),
        "samples.jsonl": "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in samples
        ),
        "report.md": _markdown(report),
    }


def verify_report_directory(
    output_dir: str | Path,
    *,
    expected_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the report commit marker, every declared file, and optional content."""

    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise GSM8KNumericDiagnosticError(f"report directory does not exist: {output}")
    manifest_path = output / "sha256_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise GSM8KNumericDiagnosticError("report manifest is missing or is a symlink")
    manifest = _read_strict_json(manifest_path, "report manifest")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise GSM8KNumericDiagnosticError("report manifest is not a complete schema-v1 commit")
    declared_files = _mapping(manifest.get("files"), "report manifest.files")
    input_checkpoint_sha = manifest.get("input_checkpoint_sha256")
    if not isinstance(input_checkpoint_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", input_checkpoint_sha
    ):
        raise GSM8KNumericDiagnosticError("report manifest input checkpoint SHA256 is invalid")
    expected_names = {"report.json", "summary.json", "samples.jsonl", "report.md"}
    if set(declared_files) != expected_names:
        raise GSM8KNumericDiagnosticError("report manifest file set mismatch")
    actual_names = {item.name for item in output.iterdir()}
    if actual_names != expected_names | {"sha256_manifest.json"}:
        raise GSM8KNumericDiagnosticError("report directory contains missing or unexpected files")

    for name in sorted(expected_names):
        metadata = _mapping(declared_files[name], f"report manifest.files.{name}")
        candidate = output / name
        if candidate.is_symlink() or not candidate.is_file():
            raise GSM8KNumericDiagnosticError(f"report file is missing or is a symlink: {name}")
        expected_path = str(candidate.resolve())
        if metadata.get("path") != expected_path:
            raise GSM8KNumericDiagnosticError(f"manifest path mismatch for {name}")
        declared_sha = metadata.get("sha256")
        declared_size = metadata.get("size_bytes")
        if not isinstance(declared_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", declared_sha):
            raise GSM8KNumericDiagnosticError(f"manifest SHA256 is invalid for {name}")
        if not isinstance(declared_size, int) or isinstance(declared_size, bool):
            raise GSM8KNumericDiagnosticError(f"manifest size is invalid for {name}")
        if candidate.stat().st_size != declared_size or sha256_file(candidate) != declared_sha:
            raise GSM8KNumericDiagnosticError(f"report file checksum/size mismatch: {name}")

    if expected_report is not None:
        expected_files = _render_report_files(expected_report)
        for name, contents in expected_files.items():
            if (output / name).read_bytes() != contents.encode("utf-8"):
                raise GSM8KNumericDiagnosticError(
                    f"existing report content disagrees with requested result: {name}"
                )
        expected_input_sha = expected_report["input"]["checkpoint_sha256"]
        if manifest.get("input_checkpoint_sha256") != expected_input_sha:
            raise GSM8KNumericDiagnosticError(
                "report manifest input checkpoint identity mismatch"
            )
    return dict(manifest)


def write_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Publish atomically, strictly reuse identical output, or refuse conflicts."""

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.iterdir())
    if existing and not overwrite:
        try:
            manifest = verify_report_directory(output, expected_report=report)
        except GSM8KNumericDiagnosticError as exc:
            raise GSM8KNumericDiagnosticError(
                f"output directory already exists but is not the identical committed report: {exc}; "
                "pass --overwrite to replace the known report files"
            ) from exc
        return {**manifest, "publication_status": "reused_complete"}

    allowed_names = {
        "report.json",
        "summary.json",
        "samples.jsonl",
        "report.md",
        "sha256_manifest.json",
    }
    unexpected = sorted(item.name for item in existing if item.name not in allowed_names)
    if unexpected:
        raise GSM8KNumericDiagnosticError(
            f"refusing to overwrite directory with unexpected entries: {unexpected}"
        )
    # Invalidate the old commit marker before an explicitly requested rewrite.
    # A crash can then leave only an uncommitted directory, never a stale
    # manifest that appears complete.
    (output / "sha256_manifest.json").unlink(missing_ok=True)
    files = _render_report_files(report)
    for name, contents in files.items():
        _atomic_write_text(output / name, contents)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_checkpoint_sha256": report["input"]["checkpoint_sha256"],
        "files": {
            name: {
                "path": str((output / name).resolve()),
                "sha256": sha256_file(output / name),
                "size_bytes": (output / name).stat().st_size,
            }
            for name in sorted(files)
        },
    }
    # The manifest is the final commit marker.  Consumers must require its
    # status and verify each listed checksum before accepting the directory.
    atomic_write_json(output / "sha256_manifest.json", manifest)
    verified = verify_report_directory(output, expected_report=report)
    return {**verified, "publication_status": "overwritten" if existing else "created"}


__all__ = [
    "FLEXIBLE_FILTER",
    "GSM8KNumericDiagnosticError",
    "METRIC_NAME",
    "PAPER_DISCLOSURE",
    "SCHEMA_VERSION",
    "STRICT_FILTER",
    "extract_final_numeric_answer",
    "parse_numeric_value",
    "score_checkpoint",
    "verify_report_directory",
    "write_report",
]
