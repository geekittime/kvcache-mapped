"""Sealed HellaSwag-domain mechanism diagnostics for Qwen3 32B -> 14B.

The paper defines the diagnostics in section 4.5, but it does not publish the
Qwen3 32B->14B scalar values, sample IDs, or aggregation implementation.  This
module therefore implements a deterministic local completion of the disclosed
formulas and records that boundary in every final artifact.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .eval_common import (
    assert_evaluation_output,
    canonical_jsonl_bytes,
    code_identity,
    hashed_json,
    hashed_jsonl,
    immutable_hashed_json,
    mapper_manifest_identity,
    model_artifact_identity,
    token_ids_sha256,
    verify_hashed_json,
)
from .hellaswag_eval import FORMAL_RECEIPT as HELLASWAG_RECEIPT
from .hellaswag_eval import _validate_existing_output as validate_hellaswag_output
from .hellaswag_eval import _validate_gpu_idle_gate as validate_gpu_idle_gate
from .protocol import (
    BIAS_COUNT,
    DIRECTION,
    EXPERIMENT_ROOT,
    REPOSITORY_ROOT,
    TOP_K,
    WEIGHT_COUNT,
    canonical_sha256,
    atomic_write_json,
    read_json,
    sha256_file,
)
from .reverse_latency import acquire_gpu_lock, wait_for_idle_gpus


DEFAULT_CONFIG = EXPERIMENT_ROOT / "config/mechanism.yaml"
RESULT_NAME = "mechanism_result.json"
RECEIPT_NAME = "formal_mechanism_receipt.json"
EXPECTED_VALIDATION_SHA256 = (
    "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd"
)
EXPECTED_ROWS = 10_042
TARGET_LAYERS = 40
KV_HEADS = 8
QUERY_HEADS = 40
HEAD_DIM = 128
CHANNELS = KV_HEADS * HEAD_DIM
REQUESTS = 400

CODE_FILES = (
    "reverse_kv/__init__.py",
    "reverse_kv/mechanism.py",
    "reverse_kv/eval_common.py",
    "reverse_kv/hellaswag_eval.py",
    "reverse_kv/provenance.py",
    "reverse_kv/protocol.py",
    "reverse_kv/reverse_latency.py",
    "reverse_kv/runtime.py",
    "scripts/evaluate_mechanism.py",
    "config/mechanism.yaml",
    "artifacts/inputs/code_freeze.json",
)
SHARED_CODE_FILES = (
    "src/kvcache_transfer/__init__.py",
    "src/kvcache_transfer/artifacts.py",
    "src/kvcache_transfer/cache_adapter.py",
    "src/kvcache_transfer/config.py",
    "src/kvcache_transfer/evaluation.py",
    "src/kvcache_transfer/io.py",
    "src/kvcache_transfer/latency.py",
    "src/kvcache_transfer/mapper.py",
    "src/kvcache_transfer/modeling.py",
    "src/kvcache_transfer/rope.py",
    "src/kvcache_transfer/transfer.py",
)
HARNESS_CODE_FILES = (
    "vendor/lm-evaluation-harness/lm_eval/api/task.py",
    "vendor/lm-evaluation-harness/lm_eval/tasks/hellaswag/hellaswag.yaml",
    "vendor/lm-evaluation-harness/lm_eval/tasks/hellaswag/utils.py",
)


def _resolve(base: Path, value: Any) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_mechanism_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the immutable local completion and reject protocol drift."""

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("mechanism config must be a mapping")
    if raw.get("schema_version") != 1 or raw.get("direction") != DIRECTION:
        raise ValueError("mechanism config direction/schema changed")
    paper = raw.get("paper_reference")
    dataset = raw.get("dataset")
    protocol = raw.get("local_protocol")
    paths = raw.get("paths")
    runtime = raw.get("runtime")
    if not all(isinstance(row, Mapping) for row in (paper, dataset, protocol, paths, runtime)):
        raise ValueError("mechanism config lacks a required mapping")
    if dict(paper) != {
        "paper": "arXiv:2608.03893v1",
        "section": "4.5",
        "evaluation_domain": "HellaSwag tokens",
        "published_cross_pair_evaluations": 12,
        "published_attention_cosine_retention_pearson_r": 0.57,
        "published_calibration_r2k_retention_pearson_r": -0.20,
        "qwen3_32b_to_14b_single_pair_values_published": False,
    }:
        raise ValueError("paper mechanism disclosure boundary changed")
    dataset_expected = {
        "split": "validation",
        "rows": EXPECTED_ROWS,
        "sha256": EXPECTED_VALIDATION_SHA256,
        "canonical_repo": "Rowan/hellaswag",
        "canonical_revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
    }
    for field, expected in dataset_expected.items():
        if dataset.get(field) != expected:
            raise ValueError(f"mechanism dataset {field} changed")
    protocol_expected = {
        "documents": 100,
        "choices_per_document": 4,
        "scoring_sequences": REQUESTS,
        "selection": "sha256_rank_without_gold_label",
        "selection_seed": "qwen3_32b_to_14b_hellaswag_mechanism_v1",
        "completion_mode": True,
        "apply_chat_template": False,
        "num_fewshot": 0,
        "input_tokens": "exact_lm_eval_scoring_forward_input_context_plus_continuation_except_final_label_token",
        "cache_construction": "direct_source_and_target_prefill_then_full_sequence_mapper_diagnostic",
        "max_sequence_tokens": 512,
        "truncation": "forbidden",
        "r2_key_space": "both_target_rope_stripped_content_primary_and_target_rope_cache_diagnostic",
        "r2_value_space": "raw_value_space",
        "attention_query_source": "target_standalone_same_sequence_layer_input",
        "attention_output_space": "pre_o_proj_per_query_head",
        "concentration_key_space": "target_rope_cache_space",
        "grouped_query_policy": "aggregate_five_target_query_heads_that_share_each_kv_head",
        "r2_reduction": "uniform_channel_then_uniform_target_layer",
        "concentration_reduction": "per_scoring_sequence_then_uniform_kv_head_then_uniform_target_layer",
        "attention_cosine_reduction": "per_query_token_and_query_head_then_uniform_target_layer",
        "concentration_normalization": "isotropic_error_equals_one",
        "r2_and_scalar_aggregation_dtype": "float64",
        "attention_and_concentration_math_dtype": "float32",
        "checkpoint_every_requests": 10,
    }
    for field, expected in protocol_expected.items():
        if protocol.get(field) != expected:
            raise ValueError(f"mechanism local protocol {field} changed")
    if not isinstance(protocol.get("disclosure"), str) or "does not disclose" not in protocol["disclosure"]:
        raise ValueError("mechanism local-protocol disclosure is missing")
    runtime_expected = {
        "source_devices": [0, 1, 2, 3, 4],
        "source_layer_counts": [12, 13, 14, 13, 12],
        "target_devices": [5, 6, 7],
        "target_layer_counts": [13, 14, 13],
        "attention_implementation": "flash_attention_2",
        "mapper_token_chunk": 1024,
        "require_cuda_visible_devices": "0,1,2,3,4,5,6,7",
        "idle_memory_limit_mib": 64,
        "idle_consecutive_samples": 12,
        "idle_sample_interval_seconds": 5,
        "idle_timeout_seconds": 86_400,
    }
    for field, expected in runtime_expected.items():
        if runtime.get(field) != expected:
            raise ValueError(f"mechanism runtime {field} changed")
    resolved_paths = {
        "source_model": _resolve(EXPERIMENT_ROOT, paths.get("source_model")),
        "target_model": _resolve(EXPERIMENT_ROOT, paths.get("target_model")),
        "mapper_manifest": _resolve(EXPERIMENT_ROOT, paths.get("mapper_manifest")),
        "heldout_result": _resolve(EXPERIMENT_ROOT, paths.get("heldout_result")),
        "hellaswag_run": _resolve(EXPERIMENT_ROOT, paths.get("hellaswag_run")),
        "output": _resolve(EXPERIMENT_ROOT, paths.get("output")),
    }
    dataset_path = _resolve(EXPERIMENT_ROOT, dataset.get("path"))
    for label in ("mapper_manifest", "heldout_result", "hellaswag_run", "output"):
        try:
            resolved_paths[label].relative_to(EXPERIMENT_ROOT)
        except ValueError as exc:
            raise ValueError(f"mechanism {label} escaped the reverse experiment") from exc
    return {
        "config_path": config_path,
        "config_sha256": sha256_file(config_path),
        "paper_reference": dict(paper),
        "dataset": {**dict(dataset), "path": dataset_path},
        "protocol": dict(protocol),
        "paths": resolved_paths,
        "runtime": dict(runtime),
    }


def preprocess_hellaswag(text: str) -> str:
    """Exact text cleanup used by the pinned lm-eval HellaSwag task."""

    value = str(text).strip().replace(" [title]", ". ")
    value = re.sub(r"\[.*?\]", "", value)
    return value.replace("  ", " ")


def _document_rank(seed: str, row_index: int) -> str:
    return hashlib.sha256(f"{seed}\0{row_index}".encode("utf-8")).hexdigest()


def _validate_document(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    endings = row.get("endings")
    if hasattr(endings, "tolist"):
        endings = endings.tolist()
    if not isinstance(endings, Sequence) or isinstance(endings, (str, bytes)) or len(endings) != 4:
        raise ValueError(f"HellaSwag row {row_index} does not have four endings")
    try:
        gold = int(row.get("label"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"HellaSwag row {row_index} has no integer label") from exc
    if gold not in range(4):
        raise ValueError(f"HellaSwag row {row_index} label is out of range")
    ctx_b = str(row.get("ctx_b", ""))
    ctx = str(row.get("ctx_a", "")) + " " + ctx_b.capitalize()
    query = preprocess_hellaswag(str(row.get("activity_label", "")) + ": " + ctx)
    choices = tuple(preprocess_hellaswag(str(value)) for value in endings)
    if not query or any(not choice for choice in choices):
        raise ValueError(f"HellaSwag row {row_index} has empty processed text")
    return {
        "row_index": row_index,
        "ind": int(row.get("ind")),
        "source_id": str(row.get("source_id", "")),
        "query": query,
        "choices": choices,
        "gold_choice": gold,
    }


def select_documents(
    rows: Sequence[Mapping[str, Any]], *, count: int, seed: str
) -> list[dict[str, Any]]:
    """Select a deterministic gold-independent SHA-ranked document subset."""

    if count <= 0 or count > len(rows) or not seed:
        raise ValueError("invalid mechanism document selection request")
    ranked = sorted(
        ((_document_rank(seed, index), index) for index in range(len(rows))),
        key=lambda pair: (pair[0], pair[1]),
    )[:count]
    result = []
    for selection_index, (rank, row_index) in enumerate(ranked):
        document = _validate_document(rows[row_index], row_index)
        result.append(
            {
                **document,
                "selection_index": selection_index,
                "rank_sha256": rank,
            }
        )
    return result


def load_hellaswag_rows(dataset: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(dataset["path"])
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    if sha256_file(path) != EXPECTED_VALIDATION_SHA256:
        raise ValueError("HellaSwag validation payload SHA256 changed")
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=[
            "ind", "activity_label", "ctx_a", "ctx_b", "endings",
            "source_id", "label",
        ],
    )
    if table.num_rows != EXPECTED_ROWS:
        raise ValueError("HellaSwag validation row count changed")
    return table.to_pylist()


def _prefix_token_id(tokenizer: Any) -> int:
    for field in ("bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, field, None)
        if value is not None:
            return int(value)
    raise ValueError("tokenizer has no BOS/EOS token for lm-eval boundary encoding")


def build_token_plan(
    documents: Sequence[Mapping[str, Any]], tokenizer: Any, *, max_tokens: int
) -> list[dict[str, Any]]:
    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.evaluation import encode_context_continuation

    prefix = _prefix_token_id(tokenizer)
    result: list[dict[str, Any]] = []
    for document in documents:
        choices = document["choices"]
        for choice_index, choice in enumerate(choices):
            continuation = " " + str(choice)
            context_ids, continuation_ids = encode_context_continuation(
                tokenizer,
                str(document["query"]),
                continuation,
                prefix_token_id=prefix,
            )
            input_ids = [*context_ids, *continuation_ids[:-1]]
            if not input_ids or len(input_ids) > max_tokens:
                raise ValueError(
                    f"mechanism request length {len(input_ids)} is outside 1..{max_tokens}"
                )
            result.append(
                {
                    "request_index": len(result),
                    "selection_index": int(document["selection_index"]),
                    "row_index": int(document["row_index"]),
                    "rank_sha256": str(document["rank_sha256"]),
                    "ind": int(document["ind"]),
                    "source_id": str(document["source_id"]),
                    "choice_index": choice_index,
                    "gold_choice": int(document["gold_choice"]),
                    "query_sha256": hashlib.sha256(
                        str(document["query"]).encode("utf-8")
                    ).hexdigest(),
                    "choice_sha256": hashlib.sha256(
                        str(choice).encode("utf-8")
                    ).hexdigest(),
                    "input_token_ids": input_ids,
                    "input_token_count": len(input_ids),
                    "input_token_ids_sha256": token_ids_sha256(input_ids),
                }
            )
    if len(result) != REQUESTS or [row["request_index"] for row in result] != list(range(REQUESTS)):
        raise ValueError("mechanism token plan is not exactly 400 ordered requests")
    return result


def _persist_immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = canonical_jsonl_bytes(rows)
    if path.exists():
        observed = path.read_bytes()
        if observed != expected:
            raise ValueError(f"immutable mechanism token plan changed: {path}")
        sidecar = read_json(path.with_name(f"{path.name}.sha256.json"))
        exact = {
            "schema_version": 1,
            "artifact": path.name,
            "sha256": hashlib.sha256(expected).hexdigest(),
            "nbytes": len(expected),
            "rows": len(rows),
        }
        if sidecar != exact:
            raise ValueError("mechanism token-plan SHA sidecar changed")
        return sidecar
    return hashed_jsonl(path, rows)


class StreamingR2State:
    """Serializable parallel-Welford state for layer/channel R²."""

    def __init__(self, layers: int, channels: int) -> None:
        if layers <= 0 or channels <= 0:
            raise ValueError("R2 state dimensions must be positive")
        self.layers = int(layers)
        self.channels = int(channels)
        self.n = np.zeros(self.layers, dtype=np.int64)
        self.mean = np.zeros((self.layers, self.channels), dtype=np.float64)
        self.m2 = np.zeros_like(self.mean)
        self.sse = np.zeros_like(self.mean)

    def update(self, layer: int, target: torch.Tensor, prediction: torch.Tensor) -> None:
        if not 0 <= layer < self.layers or target.shape != prediction.shape:
            raise ValueError("invalid R2 update layer or tensor shape")
        if target.ndim != 2 or target.shape[1] != self.channels or target.shape[0] <= 0:
            raise ValueError("R2 update must be non-empty [observations,channels]")
        true = target.detach().to(device="cpu", dtype=torch.float64).numpy()
        pred = prediction.detach().to(device="cpu", dtype=torch.float64).numpy()
        if not np.isfinite(true).all() or not np.isfinite(pred).all():
            raise ValueError("R2 update contains NaN/Inf")
        chunk_n = int(true.shape[0])
        old_n = int(self.n[layer])
        total = old_n + chunk_n
        chunk_mean = true.mean(axis=0)
        centered = true - chunk_mean
        chunk_m2 = np.square(centered).sum(axis=0)
        delta = chunk_mean - self.mean[layer]
        self.m2[layer] += chunk_m2 + np.square(delta) * (old_n * chunk_n / total)
        self.mean[layer] += delta * (chunk_n / total)
        self.sse[layer] += np.square(true - pred).sum(axis=0)
        self.n[layer] = total

    def finalize_layer(self, layer: int) -> dict[str, Any]:
        if not 0 <= layer < self.layers or int(self.n[layer]) <= 0:
            raise ValueError("cannot finalize an empty R2 layer")
        sst = self.m2[layer]
        sse = self.sse[layer]
        score = np.empty(self.channels, dtype=np.float64)
        nonconstant = sst != 0
        score[nonconstant] = 1.0 - sse[nonconstant] / sst[nonconstant]
        score[~nonconstant] = np.where(sse[~nonconstant] == 0, 1.0, 0.0)
        global_sse = float(sse.sum())
        global_sst = float(sst.sum())
        global_r2 = 1.0 - global_sse / global_sst if global_sst else (1.0 if global_sse == 0 else 0.0)
        return {
            "observations": int(self.n[layer]),
            "channels": self.channels,
            "uniform_channel_r2_fp64": float(score.mean()),
            "global_r2_fp64": float(global_r2),
            "mse_fp64": global_sse / (int(self.n[layer]) * self.channels),
            "global_sse_fp64": global_sse,
            "global_sst_fp64": global_sst,
        }

    def state_dict(self, prefix: str) -> dict[str, np.ndarray]:
        return {
            f"{prefix}_n": self.n,
            f"{prefix}_mean": self.mean,
            f"{prefix}_m2": self.m2,
            f"{prefix}_sse": self.sse,
        }

    @classmethod
    def from_state_dict(
        cls, values: Mapping[str, np.ndarray], prefix: str, *, layers: int, channels: int
    ) -> "StreamingR2State":
        result = cls(layers, channels)
        expected = {
            "n": ((layers,), np.dtype("int64")),
            "mean": ((layers, channels), np.dtype("float64")),
            "m2": ((layers, channels), np.dtype("float64")),
            "sse": ((layers, channels), np.dtype("float64")),
        }
        for field, (shape, dtype) in expected.items():
            key = f"{prefix}_{field}"
            if key not in values:
                raise ValueError(f"mechanism checkpoint lacks {key}")
            array = np.asarray(values[key])
            if array.shape != shape or array.dtype != dtype:
                raise ValueError(f"mechanism checkpoint {key} has wrong shape/dtype")
            if field != "n" and not np.isfinite(array).all():
                raise ValueError(f"mechanism checkpoint {key} contains NaN/Inf")
            setattr(result, field, array.copy())
        if np.any(result.n < 0) or np.any(result.m2 < -1e-9) or np.any(result.sse < -1e-9):
            raise ValueError("mechanism checkpoint has invalid R2 sufficient statistics")
        return result


def _validate_layer_tensors(
    query: torch.Tensor,
    gt_key: torch.Tensor,
    gt_value: torch.Tensor,
    mapped_key: torch.Tensor,
    mapped_value: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if gt_key.shape != gt_value.shape or gt_key.shape != mapped_key.shape or gt_key.shape != mapped_value.shape:
        raise ValueError("GT/mapped K/V tensors must have one shape")
    if gt_key.ndim != 4 or query.ndim != 4:
        raise ValueError("mechanism tensors must use [batch,heads,tokens,head_dim]")
    batch, kv_heads, tokens, head_dim = gt_key.shape
    if query.shape[0] != batch or query.shape[2:] != (tokens, head_dim):
        raise ValueError("query shape is incompatible with K/V")
    query_heads = int(query.shape[1])
    if batch != 1 or tokens <= 0 or query_heads % kv_heads:
        raise ValueError("mechanism diagnostics require batch one and grouped-query heads")
    if not all(tensor.is_floating_point() for tensor in (query, gt_key, gt_value, mapped_key, mapped_value)):
        raise TypeError("mechanism tensors must be floating point")
    return batch, kv_heads, query_heads, tokens, head_dim


@torch.inference_mode()
def layer_mechanism_metrics(
    query: torch.Tensor,
    gt_key: torch.Tensor,
    gt_value: torch.Tensor,
    mapped_key: torch.Tensor,
    mapped_value: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the paper's attention-sensitive diagnostics for one sequence.

    K-concentration is ``d*tr(Q'Q E'E)/(tr(Q'Q)tr(E'E))`` for every KV
    head, algebraically identical to weighting K-error energy in the right
    singular-vector basis of the target queries.  V-concentration uses the
    squared ground-truth attention read weight at each cache position.  Both
    are normalized so isotropic/uniform error has value one.
    """

    batch, kv_heads, query_heads, tokens, head_dim = _validate_layer_tensors(
        query, gt_key, gt_value, mapped_key, mapped_value
    )
    device = query.device
    values = [tensor.to(device=device, dtype=torch.float32) for tensor in (
        query, gt_key, gt_value, mapped_key, mapped_value
    )]
    query32, gt_key32, gt_value32, mapped_key32, mapped_value32 = values
    groups = query_heads // kv_heads

    def repeated(value: torch.Tensor) -> torch.Tensor:
        return (
            value[:, :, None, :, :]
            .expand(batch, kv_heads, groups, tokens, head_dim)
            .reshape(batch, query_heads, tokens, head_dim)
        )

    causal = torch.ones((tokens, tokens), dtype=torch.bool, device=device).triu(1)

    def attention(key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query32, repeated(key).transpose(-2, -1)) / math.sqrt(head_dim)
        scores.masked_fill_(causal[None, None, :, :], -torch.inf)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.matmul(weights, repeated(value)), weights

    gt_output, gt_attention = attention(gt_key32, gt_value32)
    mapped_output, _ = attention(mapped_key32, mapped_value32)
    cosine = F.cosine_similarity(gt_output, mapped_output, dim=-1, eps=1e-8)
    cosine_sum = cosine.sum(dim=(0, 2)).to(dtype=torch.float64, device="cpu")
    cosine_count = torch.full(
        (query_heads,), batch * tokens, dtype=torch.int64, device="cpu"
    )

    grouped_query = query32.reshape(batch, kv_heads, groups, tokens, head_dim)
    qmat = grouped_query.permute(1, 0, 2, 3, 4).reshape(kv_heads, -1, head_dim)
    key_error = (mapped_key32 - gt_key32).permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)
    qcov = torch.bmm(qmat.transpose(1, 2), qmat)
    ecov = torch.bmm(key_error.transpose(1, 2), key_error)
    qtrace = qcov.diagonal(dim1=-2, dim2=-1).sum(-1)
    etrace = ecov.diagonal(dim1=-2, dim2=-1).sum(-1)
    key_denominator = qtrace * etrace
    if torch.any(key_denominator <= 0):
        raise ValueError("K concentration is undefined for zero query/error energy")
    key_concentration = (
        head_dim * (qcov * ecov).sum(dim=(-2, -1)) / key_denominator
    )

    grouped_attention = gt_attention.reshape(batch, kv_heads, groups, tokens, tokens)
    read_weight = grouped_attention.square().sum(dim=(2, 3))
    value_error_energy = (mapped_value32 - gt_value32).square().sum(-1)
    weighted = (read_weight * value_error_energy).sum(dim=(0, 2))
    read_total = read_weight.sum(dim=(0, 2))
    error_total = value_error_energy.sum(dim=(0, 2))
    value_denominator = read_total * error_total
    if torch.any(value_denominator <= 0):
        raise ValueError("V concentration is undefined for zero attention/error energy")
    value_concentration = tokens * weighted / value_denominator

    outputs = {
        "key_concentration_per_kv_head": key_concentration.to(dtype=torch.float64, device="cpu"),
        "value_concentration_per_kv_head": value_concentration.to(dtype=torch.float64, device="cpu"),
        "attention_cosine_sum_per_query_head": cosine_sum,
        "attention_cosine_count_per_query_head": cosine_count,
    }
    if any(not torch.isfinite(value).all() for value in outputs.values()):
        raise ValueError("mechanism layer metrics contain NaN/Inf")
    return outputs


class MechanismAccumulator:
    def __init__(self) -> None:
        self.key_r2 = StreamingR2State(TARGET_LAYERS, CHANNELS)
        self.key_rope_r2 = StreamingR2State(TARGET_LAYERS, CHANNELS)
        self.value_r2 = StreamingR2State(TARGET_LAYERS, CHANNELS)
        self.key_concentration_sum = np.zeros((TARGET_LAYERS, KV_HEADS), dtype=np.float64)
        self.key_concentration_count = np.zeros((TARGET_LAYERS, KV_HEADS), dtype=np.int64)
        self.value_concentration_sum = np.zeros((TARGET_LAYERS, KV_HEADS), dtype=np.float64)
        self.value_concentration_count = np.zeros((TARGET_LAYERS, KV_HEADS), dtype=np.int64)
        self.cosine_sum = np.zeros((TARGET_LAYERS, QUERY_HEADS), dtype=np.float64)
        self.cosine_count = np.zeros((TARGET_LAYERS, QUERY_HEADS), dtype=np.int64)

    def update_metrics(self, layer: int, metrics: Mapping[str, torch.Tensor]) -> None:
        rows = (
            ("key_concentration", KV_HEADS),
            ("value_concentration", KV_HEADS),
            ("cosine", QUERY_HEADS),
        )
        for stem, width in rows:
            values = metrics[
                "attention_cosine_sum_per_query_head"
                if stem == "cosine"
                else f"{stem}_per_kv_head"
            ].numpy()
            counts = (
                metrics["attention_cosine_count_per_query_head"].numpy()
                if stem == "cosine"
                else np.ones(width, dtype=np.int64)
            )
            if values.shape != (width,) or counts.shape != (width,):
                raise ValueError("mechanism accumulator metric width changed")
            getattr(self, f"{stem}_sum")[layer] += values
            getattr(self, f"{stem}_count")[layer] += counts

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            **self.key_r2.state_dict("key_r2"),
            **self.key_rope_r2.state_dict("key_rope_r2"),
            **self.value_r2.state_dict("value_r2"),
            "key_concentration_sum": self.key_concentration_sum,
            "key_concentration_count": self.key_concentration_count,
            "value_concentration_sum": self.value_concentration_sum,
            "value_concentration_count": self.value_concentration_count,
            "cosine_sum": self.cosine_sum,
            "cosine_count": self.cosine_count,
        }

    @classmethod
    def from_state_dict(cls, values: Mapping[str, np.ndarray]) -> "MechanismAccumulator":
        result = cls()
        result.key_r2 = StreamingR2State.from_state_dict(
            values, "key_r2", layers=TARGET_LAYERS, channels=CHANNELS
        )
        result.key_rope_r2 = StreamingR2State.from_state_dict(
            values, "key_rope_r2", layers=TARGET_LAYERS, channels=CHANNELS
        )
        result.value_r2 = StreamingR2State.from_state_dict(
            values, "value_r2", layers=TARGET_LAYERS, channels=CHANNELS
        )
        shapes = {
            "key_concentration_sum": ((TARGET_LAYERS, KV_HEADS), "float64"),
            "key_concentration_count": ((TARGET_LAYERS, KV_HEADS), "int64"),
            "value_concentration_sum": ((TARGET_LAYERS, KV_HEADS), "float64"),
            "value_concentration_count": ((TARGET_LAYERS, KV_HEADS), "int64"),
            "cosine_sum": ((TARGET_LAYERS, QUERY_HEADS), "float64"),
            "cosine_count": ((TARGET_LAYERS, QUERY_HEADS), "int64"),
        }
        for field, (shape, dtype) in shapes.items():
            if field not in values:
                raise ValueError(f"mechanism checkpoint lacks {field}")
            array = np.asarray(values[field])
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise ValueError(f"mechanism checkpoint {field} has wrong shape/dtype")
            if dtype == "float64" and not np.isfinite(array).all():
                raise ValueError(f"mechanism checkpoint {field} contains NaN/Inf")
            if dtype == "int64" and np.any(array < 0):
                raise ValueError(f"mechanism checkpoint {field} has negative counts")
            setattr(result, field, array.copy())
        return result


def _plan_prefix_digest(plan: Sequence[Mapping[str, Any]], completed: int) -> str:
    if not 0 <= completed <= len(plan):
        raise ValueError("invalid completed mechanism request count")
    return canonical_sha256(
        [
            {
                "request_index": row["request_index"],
                "input_token_ids_sha256": row["input_token_ids_sha256"],
            }
            for row in plan[:completed]
        ]
    )


CHECKPOINT_SCALARS = (
    "schema_version", "completed_requests", "request_fingerprint", "plan_prefix_sha256"
)


def save_checkpoint(
    root: Path,
    accumulator: MechanismAccumulator,
    *,
    completed: int,
    fingerprint: str,
    plan: Sequence[Mapping[str, Any]],
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"state_{completed:04d}.npz"
    values = {
        **accumulator.state_dict(),
        "schema_version": np.asarray([1], dtype=np.int64),
        "completed_requests": np.asarray([completed], dtype=np.int64),
        "request_fingerprint": np.asarray([fingerprint], dtype="U64"),
        "plan_prefix_sha256": np.asarray([_plan_prefix_digest(plan, completed)], dtype="U64"),
    }
    if path.exists():
        loaded, observed_completed = load_checkpoint(path, fingerprint=fingerprint, plan=plan)
        if observed_completed != completed:
            raise ValueError("existing mechanism checkpoint index changed")
        for key, value in values.items():
            if not np.array_equal(loaded[key], value):
                raise ValueError(f"immutable mechanism checkpoint changed {key}")
        _ensure_checkpoint_sidecar(
            path, completed=completed, fingerprint=fingerprint, plan=plan
        )
        return path
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=root, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(handle, **values)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _ensure_checkpoint_sidecar(
        path, completed=completed, fingerprint=fingerprint, plan=plan
    )
    return path


def _ensure_checkpoint_sidecar(
    path: Path,
    *,
    completed: int,
    fingerprint: str,
    plan: Sequence[Mapping[str, Any]],
) -> Path:
    sidecar = path.with_name(f"{path.name}.sha256.json")
    value = {
        "schema_version": 1,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "completed_requests": completed,
        "request_fingerprint": fingerprint,
        "plan_prefix_sha256": _plan_prefix_digest(plan, completed),
    }
    if sidecar.exists():
        if read_json(sidecar) != value:
            raise ValueError("mechanism checkpoint SHA sidecar changed")
    else:
        atomic_write_json(sidecar, value)
    return sidecar


def load_checkpoint(
    path: Path, *, fingerprint: str, plan: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, np.ndarray], int]:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    for key in CHECKPOINT_SCALARS:
        if key not in values or values[key].shape != (1,):
            raise ValueError(f"mechanism checkpoint lacks scalar {key}")
    if values["schema_version"].dtype != np.dtype("int64") or int(values["schema_version"][0]) != 1:
        raise ValueError("mechanism checkpoint schema changed")
    completed = int(values["completed_requests"][0])
    match = re.fullmatch(r"state_(\d{4})\.npz", path.name)
    if match is None or int(match.group(1)) != completed or not 0 < completed <= REQUESTS:
        raise ValueError("mechanism checkpoint filename/count mismatch")
    if str(values["request_fingerprint"][0]) != fingerprint:
        raise ValueError("mechanism checkpoint fingerprint changed")
    if str(values["plan_prefix_sha256"][0]) != _plan_prefix_digest(plan, completed):
        raise ValueError("mechanism checkpoint token-plan prefix changed")
    # Full shape/dtype/finite validation.
    MechanismAccumulator.from_state_dict(values)
    sidecar = path.with_name(f"{path.name}.sha256.json")
    if not sidecar.is_file():
        raise ValueError("mechanism checkpoint SHA sidecar is missing")
    expected_sidecar = {
        "schema_version": 1,
        "artifact": path.name,
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "completed_requests": completed,
        "request_fingerprint": fingerprint,
        "plan_prefix_sha256": _plan_prefix_digest(plan, completed),
    }
    if read_json(sidecar) != expected_sidecar:
        raise ValueError("mechanism checkpoint SHA sidecar changed")
    return values, completed


def resume_accumulator(
    root: Path, *, fingerprint: str, plan: Sequence[Mapping[str, Any]]
) -> tuple[MechanismAccumulator, int, Path | None]:
    paths = sorted(root.glob("state_*.npz")) if root.is_dir() else []
    latest: tuple[dict[str, np.ndarray], int, Path] | None = None
    seen: set[int] = set()
    for path in paths:
        values, completed = load_checkpoint(path, fingerprint=fingerprint, plan=plan)
        _ensure_checkpoint_sidecar(
            path, completed=completed, fingerprint=fingerprint, plan=plan
        )
        if completed in seen:
            raise ValueError("duplicate mechanism checkpoint count")
        if completed % 10 and completed != REQUESTS:
            raise ValueError("mechanism checkpoint violates frozen cadence")
        seen.add(completed)
        if latest is None or completed > latest[1]:
            latest = (values, completed, path)
    if latest is None:
        return MechanismAccumulator(), 0, None
    return MechanismAccumulator.from_state_dict(latest[0]), latest[1], latest[2]


def _input_device(model: torch.nn.Module) -> torch.device:
    return next(model.get_input_embeddings().parameters()).device


@contextmanager
def capture_target_attention_inputs(model: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None or len(layers) != TARGET_LAYERS:
        raise ValueError("target model does not expose 40 Qwen3 layers")
    captured: list[torch.Tensor | None] = [None] * TARGET_LAYERS
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], *, slot: int = index) -> None:
            hidden = kwargs.get("hidden_states") if kwargs else (args[0] if args else None)
            if not isinstance(hidden, torch.Tensor) or captured[slot] is not None:
                raise RuntimeError(f"invalid/duplicate target query capture at layer {slot}")
            captured[slot] = hidden.detach()

        handles.append(layer.self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        typed: list[torch.Tensor] = []
        yield typed
        if any(value is None for value in captured):
            raise RuntimeError("target query capture did not cover all 40 layers")
        typed.extend(value for value in captured if value is not None)
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def target_prefill_with_query_inputs(
    target: torch.nn.Module, token_ids: Sequence[int]
) -> tuple[Any, tuple[torch.Tensor, ...]]:
    base = getattr(target, "model", target)
    device = _input_device(target)
    ids = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=device)
    positions = torch.arange(ids.shape[1], device=device)
    with capture_target_attention_inputs(target) as captures:
        output = base(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            use_cache=True,
            return_dict=True,
        )
    if len(captures) != TARGET_LAYERS:
        raise RuntimeError("target query capture returned incomplete layers")
    return output.past_key_values, tuple(captures)


@torch.inference_mode()
def target_query_states(
    target: torch.nn.Module, layer: int, hidden_states: torch.Tensor
) -> torch.Tensor:
    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.rope import apply_rope, qwen3_rope_cos_sin_from_config

    base = getattr(target, "model", target)
    attention = base.layers[layer].self_attn
    batch, tokens, _ = hidden_states.shape
    query = attention.q_proj(hidden_states).view(
        batch, tokens, QUERY_HEADS, HEAD_DIM
    )
    query = attention.q_norm(query).transpose(1, 2)
    positions = torch.arange(tokens, device=query.device).unsqueeze(0)
    cos, sin = qwen3_rope_cos_sin_from_config(
        positions, target.config, device=query.device, dtype=torch.float32
    )
    return apply_rope(query, cos, sin, output_dtype=torch.float32)


def _update_one_request(
    accumulator: MechanismAccumulator,
    *,
    target: torch.nn.Module,
    mapped_cache: Any,
    target_cache: Any,
    query_inputs: Sequence[torch.Tensor],
) -> None:
    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.cache_adapter import cache_to_legacy
    from kvcache_transfer.rope import qwen3_rope_cos_sin_from_config, remove_rope

    gt_layers = cache_to_legacy(target_cache)
    mapped_layers = cache_to_legacy(mapped_cache)
    if len(gt_layers) != TARGET_LAYERS or len(mapped_layers) != TARGET_LAYERS:
        raise ValueError("mechanism cache target-layer coverage changed")
    for layer, ((gt_key, gt_value), (mapped_key, mapped_value), hidden) in enumerate(
        zip(gt_layers, mapped_layers, query_inputs, strict=True)
    ):
        if gt_key.shape != mapped_key.shape or tuple(gt_key.shape[1:3])[:1] != (KV_HEADS,):
            raise ValueError(f"mechanism mapped/GT cache shape changed at layer {layer}")
        tokens = int(gt_key.shape[-2])
        positions = torch.arange(tokens, device=gt_key.device).unsqueeze(0)
        cos, sin = qwen3_rope_cos_sin_from_config(
            positions, target.config, device=gt_key.device, dtype=torch.float32
        )
        gt_content = remove_rope(gt_key, cos, sin, output_dtype=torch.float32)
        mapped_content = remove_rope(mapped_key, cos, sin, output_dtype=torch.float32)
        accumulator.key_r2.update(
            layer,
            gt_content.permute(0, 2, 1, 3).reshape(-1, CHANNELS),
            mapped_content.permute(0, 2, 1, 3).reshape(-1, CHANNELS),
        )
        accumulator.key_rope_r2.update(
            layer,
            gt_key.permute(0, 2, 1, 3).reshape(-1, CHANNELS),
            mapped_key.permute(0, 2, 1, 3).reshape(-1, CHANNELS),
        )
        accumulator.value_r2.update(
            layer,
            gt_value.permute(0, 2, 1, 3).reshape(-1, CHANNELS),
            mapped_value.permute(0, 2, 1, 3).reshape(-1, CHANNELS),
        )
        query = target_query_states(target, layer, hidden)
        metrics = layer_mechanism_metrics(
            query, gt_key, gt_value, mapped_key, mapped_value
        )
        accumulator.update_metrics(layer, metrics)


def summarize_accumulator(accumulator: MechanismAccumulator) -> dict[str, Any]:
    layers = []
    expected_observations: int | None = None
    for layer in range(TARGET_LAYERS):
        key_count = accumulator.key_concentration_count[layer]
        value_count = accumulator.value_concentration_count[layer]
        cosine_count = accumulator.cosine_count[layer]
        if (
            np.any(key_count != REQUESTS)
            or np.any(value_count != REQUESTS)
            or np.any(cosine_count <= 0)
        ):
            raise ValueError(f"mechanism metric coverage is incomplete at layer {layer}")
        key_concentration = accumulator.key_concentration_sum[layer] / key_count
        value_concentration = accumulator.value_concentration_sum[layer] / value_count
        cosine = accumulator.cosine_sum[layer] / cosine_count
        if not all(np.isfinite(value).all() for value in (key_concentration, value_concentration, cosine)):
            raise ValueError("mechanism final metrics contain NaN/Inf")
        key_r2 = accumulator.key_r2.finalize_layer(layer)
        key_rope_r2 = accumulator.key_rope_r2.finalize_layer(layer)
        value_r2 = accumulator.value_r2.finalize_layer(layer)
        observations = int(key_r2["observations"])
        if not (
            observations
            == int(key_rope_r2["observations"])
            == int(value_r2["observations"])
        ):
            raise ValueError(f"K-content/K-RoPE/V R2 observation counts differ at layer {layer}")
        if expected_observations is None:
            expected_observations = observations
        elif observations != expected_observations:
            raise ValueError("R2 token coverage differs across target layers")
        layers.append(
            {
                "target_layer": layer,
                "key_content_r2": key_r2,
                "key_rope_cache_r2": key_rope_r2,
                "value_r2": value_r2,
                "key_concentration": {
                    "per_kv_head": key_concentration.tolist(),
                    "mean_across_kv_heads": float(key_concentration.mean()),
                    "sequences_per_head": REQUESTS,
                },
                "value_concentration": {
                    "per_kv_head": value_concentration.tolist(),
                    "mean_across_kv_heads": float(value_concentration.mean()),
                    "sequences_per_head": REQUESTS,
                },
                "attention_output_cosine": {
                    "per_query_head": cosine.tolist(),
                    "mean_across_query_heads": float(cosine.mean()),
                    "token_head_observations": int(cosine_count.sum()),
                },
            }
        )
    key_r2 = [row["key_content_r2"]["uniform_channel_r2_fp64"] for row in layers]
    key_rope_r2 = [row["key_rope_cache_r2"]["uniform_channel_r2_fp64"] for row in layers]
    value_r2 = [row["value_r2"]["uniform_channel_r2_fp64"] for row in layers]
    key_concentration = [row["key_concentration"]["mean_across_kv_heads"] for row in layers]
    value_concentration = [row["value_concentration"]["mean_across_kv_heads"] for row in layers]
    cosine = [row["attention_output_cosine"]["mean_across_query_heads"] for row in layers]
    return {
        "layers": layers,
        "scoring_input_tokens": expected_observations,
        "headline": {
            "hellaswag_eval_domain_r2_k_content": float(np.mean(key_r2)),
            "hellaswag_eval_domain_r2_k_rope_cache": float(np.mean(key_rope_r2)),
            "hellaswag_eval_domain_r2_v": float(np.mean(value_r2)),
            "key_error_concentration": float(np.mean(key_concentration)),
            "value_error_concentration": float(np.mean(value_concentration)),
            "attention_output_cosine": float(np.mean(cosine)),
        },
    }


def _hellaswag_binding(root: Path) -> dict[str, Any]:
    validate_hellaswag_output(root)
    receipt_path = root / HELLASWAG_RECEIPT
    if not receipt_path.is_file():
        raise FileNotFoundError("mechanism run requires completed formal HellaSwag first")
    verify_hashed_json(receipt_path)
    receipt = read_json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("task") != "hellaswag":
        raise ValueError("formal HellaSwag receipt changed")
    by_mode = {
        str(row.get("mode")): row
        for row in receipt.get("results", ())
        if isinstance(row, Mapping)
    }
    if set(by_mode) != {"target_standalone", "mapped"}:
        raise ValueError("formal HellaSwag modes changed")
    target = float(by_mode["target_standalone"]["acc_norm"])
    mapped = float(by_mode["mapped"]["acc_norm"])
    if not (0 < target <= 1 and 0 <= mapped <= 1):
        raise ValueError("formal HellaSwag accuracy is invalid")
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "receipt_nbytes": receipt_path.stat().st_size,
        "request_fingerprint": receipt.get("request_fingerprint"),
        "target_standalone_acc_norm": target,
        "mapped_acc_norm": mapped,
        "mapped_over_target_retention_percent": mapped / target * 100.0,
    }


def _heldout_binding(path: Path) -> dict[str, Any]:
    """Bind the earlier local FineWeb held-out audit without relabelling it."""

    if not path.is_file():
        raise FileNotFoundError("mechanism run requires completed held-out reconstruction first")
    sidecar = path.with_suffix(".sha256.json")
    value = read_json(path)
    marker = read_json(sidecar)
    if marker.get("sha256") != sha256_file(path):
        raise ValueError("held-out reconstruction SHA marker changed")
    layers = value.get("layers")
    if (
        value.get("status") != "complete"
        or value.get("direction") != DIRECTION
        or not isinstance(layers, list)
        or len(layers) != TARGET_LAYERS
    ):
        raise ValueError("held-out reconstruction protocol changed")
    key = []
    val = []
    for expected_layer, row in enumerate(layers):
        if not isinstance(row, Mapping) or row.get("target_layer") != expected_layer:
            raise ValueError("held-out reconstruction layer order changed")
        key.append(float(row["key"]["uniform_channel_r2_fp64"]))
        val.append(float(row["value"]["uniform_channel_r2_fp64"]))
    if not all(math.isfinite(number) for number in (*key, *val)):
        raise ValueError("held-out reconstruction contains non-finite R2")
    key_mean = sum(key) / TARGET_LAYERS
    value_mean = sum(val) / TARGET_LAYERS
    summary = value.get("summary")
    if not isinstance(summary, Mapping) or not math.isclose(
        float(summary.get("key_uniform_channel_r2_fp64_mean_across_layers")),
        key_mean,
        rel_tol=0,
        abs_tol=1e-12,
    ) or not math.isclose(
        float(summary.get("value_uniform_channel_r2_fp64_mean_across_layers")),
        value_mean,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("held-out reconstruction summary disagrees with its layers")
    return {
        "scope": "local_independent_fineweb_heldout_not_paper_disclosed_calibration_scalar",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
        "sha256_sidecar": {
            "path": str(sidecar.resolve()),
            "sha256": sha256_file(sidecar),
            "nbytes": sidecar.stat().st_size,
        },
        "key_content_uniform_channel_r2_mean_across_layers": key_mean,
        "value_uniform_channel_r2_mean_across_layers": value_mean,
    }


def _definition(
    configuration: Mapping[str, Any],
    token_plan_identity: Mapping[str, Any],
    hellaswag: Mapping[str, Any],
    heldout: Mapping[str, Any],
) -> dict[str, Any]:
    paths = configuration["paths"]
    source = model_artifact_identity(paths["source_model"])
    target = model_artifact_identity(paths["target_model"])
    source_tokenizer = source["content_hashed_files"]["tokenizer.json"]["sha256"]
    target_tokenizer = target["content_hashed_files"]["tokenizer.json"]["sha256"]
    if source_tokenizer != target_tokenizer:
        raise ValueError("source/target tokenizer.json bytes differ")
    return {
        "schema_version": 1,
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism_local_completion",
        "direction": DIRECTION,
        "paper_reference": configuration["paper_reference"],
        "local_protocol": configuration["protocol"],
        "dataset": {
            "path": str(configuration["dataset"]["path"]),
            "rows": EXPECTED_ROWS,
            "sha256": EXPECTED_VALIDATION_SHA256,
            "canonical_repo": configuration["dataset"]["canonical_repo"],
            "canonical_revision": configuration["dataset"]["canonical_revision"],
        },
        "token_plan": dict(token_plan_identity),
        "artifacts": {
            "source_model": source,
            "target_model": target,
            "mapper": mapper_manifest_identity(paths["mapper_manifest"]),
            "local_heldout_reconstruction": dict(heldout),
            "formal_hellaswag": dict(hellaswag),
        },
        "runtime": configuration["runtime"],
        "code": {
            "reverse_experiment": code_identity(CODE_FILES),
            "shared_read_only_runtime": code_identity(
                SHARED_CODE_FILES, root=REPOSITORY_ROOT
            ),
            "vendored_hellaswag_protocol": code_identity(
                HARNESS_CODE_FILES, root=REPOSITORY_ROOT
            ),
        },
        "metric_formulas": {
            "r2": (
                "per-channel 1-SSE/SST in target RoPE-stripped content-K "
                "(primary), target RoPE-cache K (diagnostic), and raw-V; "
                "arithmetic mean channels then layers"
            ),
            "key_concentration": "d*trace((Q^T Q)(E_K^T E_K))/(trace(Q^T Q)*trace(E_K^T E_K))",
            "value_concentration": "T*sum_t(read_weight_t*||E_V,t||^2)/(sum_t read_weight_t*sum_t ||E_V,t||^2)",
            "value_read_weight": "sum over grouped target query heads and causal query positions of squared GT attention probability",
            "attention_output_cosine": "cosine(mapped-attention-output,GT-attention-output) before o_proj",
        },
        "comparability": {
            "paper_qwen3_32b_to_14b_single_pair_reference_available": False,
            "paper_table4_contains_this_direction": False,
            "published_pearson_correlations_require_all_12_pair_directions": True,
            "single_pair_correlation_claimed": False,
            "bit_exact_paper_reproduction_claimed": False,
        },
    }


def _checkpoint_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "nbytes": path.stat().st_size,
    }


def _expanded_cuda_devices(
    devices: Sequence[int], layer_counts: Sequence[int]
) -> list[str]:
    if len(devices) != len(layer_counts):
        raise ValueError("mechanism runtime device/count lengths differ")
    return [
        f"cuda:{int(device)}"
        for device, count in zip(devices, layer_counts, strict=True)
        for _ in range(int(count))
    ]


def _validate_runtime_sharding(
    value: Any,
    *,
    devices: Sequence[int],
    layer_counts: Sequence[int],
    native_role: str,
    label: str,
) -> None:
    expected_layer_devices = [
        int(device)
        for device, count in zip(devices, layer_counts, strict=True)
        for _ in range(int(count))
    ]
    expected_keys = {
        "status",
        "role",
        "mode",
        "devices",
        "declared_layer_counts",
        "expected_layer_counts",
        "actual_layer_counts",
        "total_layers",
        "layer_devices",
        "contiguous_ranges",
        "actual_parameters_inspected_for_every_layer",
        "parameter_tensors_inspected",
        "auxiliary_modules",
        "auxiliary_parameter_tensors_inspected",
        "auxiliary_parameters_inspected_for_every_module",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"mechanism runtime {label} sharding schema changed")
    fixed = {
        "status": "passed",
        "role": native_role,
        "mode": "explicit_exact_counts",
        "devices": list(devices),
        "declared_layer_counts": list(layer_counts),
        "expected_layer_counts": list(layer_counts),
        "actual_layer_counts": list(layer_counts),
        "total_layers": sum(int(count) for count in layer_counts),
        "layer_devices": expected_layer_devices,
        "actual_parameters_inspected_for_every_layer": True,
        "auxiliary_parameters_inspected_for_every_module": True,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(
                f"mechanism runtime {label} sharding {field} changed"
            )
    ranges = []
    start = 0
    for device, count in zip(devices, layer_counts, strict=True):
        count = int(count)
        ranges.append(
            {
                "device": int(device),
                "start_layer": start,
                "end_layer_exclusive": start + count,
                "count": count,
            }
        )
        start += count
    if value.get("contiguous_ranges") != ranges:
        raise ValueError(
            f"mechanism runtime {label} contiguous sharding ranges changed"
        )
    inspected = value.get("parameter_tensors_inspected")
    auxiliary_inspected = value.get("auxiliary_parameter_tensors_inspected")
    if (
        type(inspected) is not int
        or inspected <= 0
        or type(auxiliary_inspected) is not int
        or auxiliary_inspected <= 0
    ):
        raise ValueError(
            f"mechanism runtime {label} sharding inspection counts are invalid"
        )
    auxiliary = value.get("auxiliary_modules")
    expected_auxiliary_devices = {
        "model.embed_tokens": int(devices[0]),
        "model.norm": int(devices[-1]),
        "lm_head": int(devices[-1]),
    }
    if not isinstance(auxiliary, Mapping) or set(auxiliary) != set(
        expected_auxiliary_devices
    ):
        raise ValueError(
            f"mechanism runtime {label} auxiliary sharding schema changed"
        )
    recomputed_auxiliary = 0
    for module_name, expected_device in expected_auxiliary_devices.items():
        row = auxiliary[module_name]
        if not isinstance(row, Mapping) or set(row) != {
            "expected_device",
            "actual_device",
            "parameter_tensors_inspected",
        }:
            raise ValueError(
                f"mechanism runtime {label} auxiliary module schema changed"
            )
        count = row.get("parameter_tensors_inspected")
        if (
            row.get("expected_device") != expected_device
            or row.get("actual_device") != expected_device
            or type(count) is not int
            or count <= 0
        ):
            raise ValueError(
                f"mechanism runtime {label} auxiliary module placement changed"
            )
        recomputed_auxiliary += count
    if auxiliary_inspected != recomputed_auxiliary:
        raise ValueError(
            f"mechanism runtime {label} auxiliary inspection total changed"
        )


def _model_instance_identity(artifact: Any, *, label: str) -> dict[str, str]:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"mechanism definition lacks {label} model identity")
    files = artifact.get("content_hashed_files")
    if not isinstance(files, Mapping):
        raise ValueError(f"mechanism definition {label} model metadata changed")
    expected = {
        "model_path": artifact.get("path"),
        "model_config_sha256": files.get("config.json", {}).get("sha256"),
        "model_tokenizer_sha256": files.get("tokenizer.json", {}).get("sha256"),
        "model_weight_index_sha256": files.get(
            "model.safetensors.index.json", {}
        ).get("sha256"),
    }
    for field, observed in expected.items():
        if field == "model_path":
            if not isinstance(observed, str) or not observed:
                raise ValueError(
                    f"mechanism definition {label} model path is invalid"
                )
        elif not isinstance(observed, str) or re.fullmatch(
            r"[0-9a-f]{64}", observed
        ) is None:
            raise ValueError(
                f"mechanism definition {label} model hash is invalid"
            )
    return expected


def _current_runtime_mapper_identity(
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the current mapper and rehash every committed layer payload."""

    from .runtime import ReverseMapperBundle

    manifest = Path(configuration["paths"]["mapper_manifest"]).resolve()
    bundle = ReverseMapperBundle.open(
        manifest if manifest.is_dir() else manifest.parent
    )
    return {
        "manifest": str(bundle.manifest_path),
        "manifest_sha256": bundle.manifest_sha256,
        "fit_fingerprint": bundle.fit_fingerprint,
        "layers": len(bundle.layers),
        "selected_source_layers": bundle.architecture.selected_source_layers,
        "source_capture_model_identity": dict(bundle.source_model_identity),
        "target_capture_model_identity": dict(bundle.target_model_identity),
    }


def _validate_materialization(
    value: Any, *, target_layer_devices: Sequence[str]
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "scope",
        "host_load_and_hash",
        "weight_host_to_device",
        "per_layer",
    }:
        raise ValueError("mechanism runtime mapper materialization schema changed")
    if value.get("scope") != "one_time_setup_excluded_from_per_request_latency":
        raise ValueError("mechanism runtime mapper materialization scope changed")
    rows = value.get("per_layer")
    if not isinstance(rows, list) or len(rows) != len(target_layer_devices):
        raise ValueError("mechanism runtime mapper materialization coverage changed")
    host_samples: list[float] = []
    transfer_samples: list[float] = []
    for target_layer, (row, expected_device) in enumerate(
        zip(rows, target_layer_devices, strict=True)
    ):
        if not isinstance(row, Mapping) or set(row) != {
            "target_layer",
            "device",
            "host_load_and_hash_ms",
            "weight_host_to_device_ms",
        }:
            raise ValueError(
                "mechanism runtime mapper materialization row schema changed"
            )
        host = row.get("host_load_and_hash_ms")
        transfer = row.get("weight_host_to_device_ms")
        if (
            row.get("target_layer") != target_layer
            or row.get("device") != expected_device
            or type(host) not in (int, float)
            or not math.isfinite(float(host))
            or float(host) < 0
            or type(transfer) not in (int, float)
            or not math.isfinite(float(transfer))
            or float(transfer) < 0
        ):
            raise ValueError(
                "mechanism runtime mapper materialization row changed"
            )
        host_samples.append(float(host))
        transfer_samples.append(float(transfer))
    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.latency import summarize_latency

    if value.get("host_load_and_hash") != summarize_latency(host_samples):
        raise ValueError(
            "mechanism runtime mapper host-load summary changed"
        )
    expected_transfer = {
        **summarize_latency(transfer_samples),
        "timing_backends": ["host_perf_counter_with_cuda_synchronize"],
    }
    if value.get("weight_host_to_device") != expected_transfer:
        raise ValueError(
            "mechanism runtime mapper device-transfer summary changed"
        )


def _validate_runtime_setup(
    value: Any,
    *,
    fingerprint: str,
    configuration: Mapping[str, Any],
    definition: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "status",
        "request_fingerprint",
        "source_sharding",
        "target_sharding",
        "mapper",
    }:
        raise ValueError("mechanism runtime setup schema changed")
    if (
        value.get("schema_version") != 1
        or value.get("status") != "ready"
        or value.get("request_fingerprint") != fingerprint
    ):
        raise ValueError("mechanism runtime setup identity changed")
    runtime = configuration["runtime"]
    _validate_runtime_sharding(
        value.get("source_sharding"),
        devices=runtime["source_devices"],
        layer_counts=runtime["source_layer_counts"],
        native_role="target",
        label="source",
    )
    _validate_runtime_sharding(
        value.get("target_sharding"),
        devices=runtime["target_devices"],
        layer_counts=runtime["target_layer_counts"],
        native_role="source",
        label="target",
    )
    mapper = value.get("mapper")
    mapper_keys = {
        "direction",
        "manifest",
        "manifest_sha256",
        "fit_fingerprint",
        "layers",
        "selected_source_layers",
        "token_chunk",
        "execution",
        "target_layer_devices",
        "resident_parameter_bytes",
        "materialization",
        "source_model_artifact",
        "target_model_artifact",
        "source_capture_model_identity",
        "target_capture_model_identity",
    }
    if not isinstance(mapper, Mapping) or set(mapper) != mapper_keys:
        raise ValueError("mechanism runtime mapper schema changed")
    artifacts = definition.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("mechanism definition artifact identity changed")
    source_model = _model_instance_identity(
        artifacts.get("source_model"), label="source"
    )
    target_model = _model_instance_identity(
        artifacts.get("target_model"), label="target"
    )
    current_mapper = _current_runtime_mapper_identity(configuration)
    target_layer_devices = _expanded_cuda_devices(
        runtime["target_devices"], runtime["target_layer_counts"]
    )
    fixed = {
        "direction": DIRECTION,
        "manifest": current_mapper["manifest"],
        "manifest_sha256": current_mapper["manifest_sha256"],
        "fit_fingerprint": current_mapper["fit_fingerprint"],
        "layers": TARGET_LAYERS,
        "selected_source_layers": TOP_K,
        "token_chunk": runtime["mapper_token_chunk"],
        "execution": "fp32_mapper_weights_resident_on_target_layer_devices",
        "target_layer_devices": target_layer_devices,
        "resident_parameter_bytes": (WEIGHT_COUNT + BIAS_COUNT) * 4,
        "source_model_artifact": source_model,
        "target_model_artifact": target_model,
        "source_capture_model_identity": current_mapper[
            "source_capture_model_identity"
        ],
        "target_capture_model_identity": current_mapper[
            "target_capture_model_identity"
        ],
    }
    for field, expected in fixed.items():
        if mapper.get(field) != expected:
            raise ValueError(f"mechanism runtime mapper {field} changed")
    if (
        current_mapper.get("layers") != TARGET_LAYERS
        or current_mapper.get("selected_source_layers") != TOP_K
    ):
        raise ValueError("mechanism current mapper architecture changed")
    definition_mapper = artifacts.get("mapper")
    if (
        not isinstance(definition_mapper, Mapping)
        or definition_mapper.get("manifest") != current_mapper["manifest"]
        or definition_mapper.get("manifest_sha256")
        != current_mapper["manifest_sha256"]
        or definition_mapper.get("fit_fingerprint")
        != current_mapper["fit_fingerprint"]
        or definition_mapper.get("target_layer_count") != TARGET_LAYERS
    ):
        raise ValueError(
            "mechanism runtime mapper differs from current definition"
        )
    _validate_materialization(
        mapper.get("materialization"),
        target_layer_devices=target_layer_devices,
    )


def validate_mechanism_result(path: str | Path, *, fingerprint: str) -> dict[str, Any]:
    destination = Path(path)
    verify_hashed_json(destination)
    value = read_json(destination)
    fixed = {
        "schema_version": 1,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism",
        "direction": DIRECTION,
        "request_fingerprint": fingerprint,
        "documents": 100,
        "scoring_sequences": REQUESTS,
        "target_layers": TARGET_LAYERS,
    }
    for field, expected in fixed.items():
        if value.get(field) != expected:
            raise ValueError(f"mechanism result {field} changed")
    layers = value.get("layers")
    if not isinstance(layers, list) or len(layers) != TARGET_LAYERS:
        raise ValueError("mechanism result does not contain 40 target layers")
    scoring_tokens = value.get("scoring_input_tokens")
    if type(scoring_tokens) is not int or scoring_tokens <= 0:
        raise ValueError("mechanism result has invalid scoring-token coverage")
    fields = (
        ("hellaswag_eval_domain_r2_k_content", "key_content_r2", "uniform_channel_r2_fp64"),
        ("hellaswag_eval_domain_r2_k_rope_cache", "key_rope_cache_r2", "uniform_channel_r2_fp64"),
        ("hellaswag_eval_domain_r2_v", "value_r2", "uniform_channel_r2_fp64"),
        ("key_error_concentration", "key_concentration", "mean_across_kv_heads"),
        ("value_error_concentration", "value_concentration", "mean_across_kv_heads"),
        ("attention_output_cosine", "attention_output_cosine", "mean_across_query_heads"),
    )
    headline = value.get("headline")
    if not isinstance(headline, Mapping):
        raise ValueError("mechanism result lacks headline")
    for headline_field, section, scalar in fields:
        per_layer = []
        for expected_layer, row in enumerate(layers):
            if not isinstance(row, Mapping) or row.get("target_layer") != expected_layer:
                raise ValueError("mechanism target-layer order changed")
            part = row.get(section)
            if not isinstance(part, Mapping):
                raise ValueError(f"mechanism layer lacks {section}")
            observed = float(part.get(scalar))
            if not math.isfinite(observed):
                raise ValueError(f"mechanism layer {section} is non-finite")
            per_layer.append(observed)
            if section in {"key_content_r2", "key_rope_cache_r2", "value_r2"} and part.get("observations") != scoring_tokens:
                raise ValueError("mechanism result R2 token coverage changed")
        recomputed = sum(per_layer) / TARGET_LAYERS
        if not math.isclose(float(headline.get(headline_field)), recomputed, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"mechanism headline {headline_field} disagrees with layers")
    if value.get("choices_per_document") != 4 or value.get("metric_scope") != (
        "deterministic local completion of paper section 4.5 on HellaSwag scoring tokens"
    ):
        raise ValueError("mechanism result local-completion scope changed")
    comparability = value.get("paper_comparison")
    expected_comparability = {
        "qwen3_32b_to_14b_published_reference": None,
        "reason": "paper Table 4 omits Qwen3 32B->14B single-pair mechanism scalars",
        "published_attention_cosine_retention_pearson_r": 0.57,
        "published_calibration_r2k_retention_pearson_r": -0.20,
        "published_correlations_scope": "12 matched-KV pair-direction evaluations",
        "single_pair_correlation_reproduction": "not mathematically defined",
    }
    if not isinstance(comparability, Mapping) or dict(comparability) != expected_comparability:
        raise ValueError("mechanism paper-comparison boundary changed")
    return {
        "path": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "nbytes": destination.stat().st_size,
        "headline": dict(headline),
    }


def validate_completed_output(output: Path) -> dict[str, Any] | None:
    receipt_path = output / RECEIPT_NAME
    if not receipt_path.exists():
        return None
    verify_hashed_json(receipt_path)
    receipt = read_json(receipt_path)
    definition_path = output / "run_definition.json"
    verify_hashed_json(definition_path)
    definition = read_json(definition_path)
    configuration = load_mechanism_config()
    expected_definition_fields = {
        "schema_version": 1,
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism_local_completion",
        "direction": DIRECTION,
        "paper_reference": configuration["paper_reference"],
        "local_protocol": configuration["protocol"],
        "runtime": configuration["runtime"],
    }
    for field, expected in expected_definition_fields.items():
        if definition.get(field) != expected:
            raise ValueError(f"mechanism run definition {field} changed")
    fingerprint = str(definition.get("request_fingerprint", ""))
    unsigned = dict(definition)
    unsigned.pop("request_fingerprint", None)
    if len(fingerprint) != 64 or canonical_sha256(unsigned) != fingerprint:
        raise ValueError("mechanism run definition fingerprint changed")
    result = validate_mechanism_result(output / RESULT_NAME, fingerprint=fingerprint)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "complete"
        or receipt.get("kind")
        != "qwen3_32b_to_14b_hellaswag_mechanism_receipt"
        or receipt.get("request_fingerprint") != fingerprint
        or receipt.get("result") != result
        or receipt.get("paper_reference_boundary")
        != {
            "single_pair_reverse_values_published": False,
            "correlations_are_cross_pair_n12_only": True,
        }
    ):
        raise ValueError("mechanism completion receipt changed")
    validate_gpu_idle_gate(output)
    checkpoint = receipt.get("final_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("mechanism receipt lacks final checkpoint")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    if (
        not checkpoint_path.is_file()
        or checkpoint_path.stat().st_size != checkpoint.get("nbytes")
        or sha256_file(checkpoint_path) != checkpoint.get("sha256")
    ):
        raise ValueError("mechanism final checkpoint changed")
    checkpoint_sidecar = checkpoint_path.with_name(
        f"{checkpoint_path.name}.sha256.json"
    )
    if not checkpoint_sidecar.is_file():
        raise ValueError("mechanism final checkpoint SHA sidecar is missing")
    recorded_checkpoint_sidecar = receipt.get("final_checkpoint_sha256_sidecar")
    if (
        not isinstance(recorded_checkpoint_sidecar, Mapping)
        or Path(str(recorded_checkpoint_sidecar.get("path", ""))).resolve()
        != checkpoint_sidecar.resolve()
        or checkpoint_sidecar.stat().st_size
        != recorded_checkpoint_sidecar.get("nbytes")
        or sha256_file(checkpoint_sidecar)
        != recorded_checkpoint_sidecar.get("sha256")
    ):
        raise ValueError("mechanism final checkpoint sidecar binding changed")
    token_plan_path = output / "token_plan.jsonl"
    token_plan_sidecar_path = token_plan_path.with_name(
        f"{token_plan_path.name}.sha256.json"
    )
    token_plan_sidecar = read_json(
        token_plan_sidecar_path
    )
    if (
        token_plan_sidecar.get("sha256") != sha256_file(token_plan_path)
        or token_plan_sidecar.get("nbytes") != token_plan_path.stat().st_size
        or token_plan_sidecar.get("rows") != REQUESTS
    ):
        raise ValueError("mechanism token-plan commit marker changed")
    plan = [
        json.loads(line)
        for line in token_plan_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(plan) != REQUESTS or [row.get("request_index") for row in plan] != list(range(REQUESTS)):
        raise ValueError("mechanism token plan is incomplete or reordered")
    for row in plan:
        token_ids = row.get("input_token_ids")
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(token) is not int or token < 0 for token in token_ids)
            or row.get("input_token_count") != len(token_ids)
            or row.get("input_token_ids_sha256") != token_ids_sha256(token_ids)
        ):
            raise ValueError("mechanism token plan token payload changed")
    plan_identity = {
        "path": str(token_plan_path.resolve()),
        **token_plan_sidecar,
        "token_count_min": min(int(row["input_token_count"]) for row in plan),
        "token_count_max": max(int(row["input_token_count"]) for row in plan),
        "token_count_total": sum(int(row["input_token_count"]) for row in plan),
    }
    selection_path = output / "selection.json"
    verify_hashed_json(selection_path)
    selection = read_json(selection_path)
    selection_rows = selection.get("selected_documents")
    if (
        selection.get("schema_version") != 1
        or selection.get("method") != "sha256_rank_without_gold_label"
        or selection.get("seed")
        != configuration["protocol"]["selection_seed"]
        or selection.get("dataset_sha256") != EXPECTED_VALIDATION_SHA256
        or not isinstance(selection_rows, list)
        or len(selection_rows) != 100
        or selection.get("selection_sha256") != canonical_sha256(selection_rows)
    ):
        raise ValueError("mechanism document selection changed")
    selection_by_index = {
        row.get("selection_index"): row
        for row in selection_rows
        if isinstance(row, Mapping)
    }
    if set(selection_by_index) != set(range(100)):
        raise ValueError("mechanism selection indices changed")
    for request_index, row in enumerate(plan):
        selected = selection_by_index.get(row.get("selection_index"))
        if (
            row.get("selection_index") != request_index // 4
            or
            row.get("choice_index") != request_index % 4
            or not isinstance(selected, Mapping)
            or any(
                row.get(field) != selected.get(field)
                for field in ("row_index", "rank_sha256", "ind", "source_id")
            )
        ):
            raise ValueError("mechanism token plan/selection binding changed")
    state_values, completed_requests = load_checkpoint(
        checkpoint_path, fingerprint=fingerprint, plan=plan
    )
    if completed_requests != REQUESTS:
        raise ValueError("mechanism final checkpoint is not request 400")
    recomputed = summarize_accumulator(
        MechanismAccumulator.from_state_dict(state_values)
    )
    result_value = read_json(output / RESULT_NAME)
    if (
        result_value.get("scoring_input_tokens")
        != recomputed["scoring_input_tokens"]
        or
        result_value.get("layers") != recomputed["layers"]
        or result_value.get("headline") != recomputed["headline"]
        or recomputed["scoring_input_tokens"]
        != sum(int(row["input_token_count"]) for row in plan)
    ):
        raise ValueError("mechanism result disagrees with final sufficient statistics")
    for field, expected_path in (
        ("run_definition", output / "run_definition.json"),
        ("token_plan", token_plan_path),
        ("selection", selection_path),
        ("runtime_setup", output / "runtime_setup.json"),
        (
            "runtime_setup_sha256_sidecar",
            output / "runtime_setup.json.sha256.json",
        ),
        ("gpu_idle_gate", output / "gpu_idle_gate.json"),
    ):
        row = receipt.get(field)
        if not isinstance(row, Mapping):
            raise ValueError(f"mechanism receipt lacks {field}")
        if (
            not expected_path.is_file()
            or Path(str(row.get("path", ""))).resolve() != expected_path.resolve()
            or expected_path.stat().st_size != row.get("nbytes")
            or sha256_file(expected_path) != row.get("sha256")
        ):
            raise ValueError(f"mechanism receipt {field} artifact changed")
    runtime_setup_path = output / "runtime_setup.json"
    verify_hashed_json(runtime_setup_path)
    hellaswag = receipt.get("formal_hellaswag_receipt")
    if not isinstance(hellaswag, Mapping):
        raise ValueError("mechanism receipt lacks formal HellaSwag binding")
    hellaswag_path = Path(str(hellaswag.get("path", "")))
    if (
        not hellaswag_path.is_file()
        or hellaswag_path.stat().st_size != hellaswag.get("nbytes")
        or sha256_file(hellaswag_path) != hellaswag.get("sha256")
    ):
        raise ValueError("mechanism formal HellaSwag binding changed")
    live_hellaswag = _hellaswag_binding(hellaswag_path.parent)
    definition_artifacts = definition.get("artifacts")
    if (
        not isinstance(definition_artifacts, Mapping)
        or definition_artifacts.get("formal_hellaswag") != live_hellaswag
        or result_value.get("hellaswag") != live_hellaswag
        or hellaswag.get("path") != live_hellaswag["receipt"]
        or hellaswag.get("sha256") != live_hellaswag["receipt_sha256"]
        or hellaswag.get("nbytes") != live_hellaswag["receipt_nbytes"]
    ):
        raise ValueError("mechanism HellaSwag semantic binding changed")
    heldout = receipt.get("local_heldout_reconstruction")
    result_heldout = result_value.get("local_heldout_reconstruction")
    if not isinstance(heldout, Mapping) or not isinstance(result_heldout, Mapping):
        raise ValueError("mechanism receipt/result lacks local held-out binding")
    heldout_path = Path(str(heldout.get("path", "")))
    if (
        heldout.get("path") != result_heldout.get("path")
        or heldout.get("sha256") != result_heldout.get("sha256")
        or heldout.get("nbytes") != result_heldout.get("nbytes")
        or not heldout_path.is_file()
        or heldout_path.stat().st_size != heldout.get("nbytes")
        or sha256_file(heldout_path) != heldout.get("sha256")
    ):
        raise ValueError("mechanism local held-out binding changed")
    live_heldout = _heldout_binding(heldout_path)
    if (
        result_heldout != live_heldout
        or definition_artifacts.get("local_heldout_reconstruction")
        != live_heldout
    ):
        raise ValueError("mechanism local held-out semantic binding changed")
    current_unsigned = _definition(
        configuration, plan_identity, live_hellaswag, live_heldout
    )
    if unsigned != current_unsigned:
        raise ValueError(
            "mechanism run definition no longer matches current "
            "code/model/mapper/freeze identities"
        )
    _validate_runtime_setup(
        read_json(runtime_setup_path),
        fingerprint=fingerprint,
        configuration=configuration,
        definition=current_unsigned,
    )
    return receipt


def run_mechanism(configuration: Mapping[str, Any], output: Path) -> dict[str, Any]:
    completed = validate_completed_output(output)
    if completed is not None:
        return completed
    output = assert_evaluation_output(output)
    rows = load_hellaswag_rows(configuration["dataset"])
    selected = select_documents(
        rows,
        count=int(configuration["protocol"]["documents"]),
        seed=str(configuration["protocol"]["selection_seed"]),
    )
    selection_rows = [
        {
            "selection_index": row["selection_index"],
            "row_index": row["row_index"],
            "rank_sha256": row["rank_sha256"],
            "ind": row["ind"],
            "source_id": row["source_id"],
        }
        for row in selected
    ]
    immutable_hashed_json(
        output / "selection.json",
        {
            "schema_version": 1,
            "method": "sha256_rank_without_gold_label",
            "seed": configuration["protocol"]["selection_seed"],
            "dataset_sha256": EXPECTED_VALIDATION_SHA256,
            "selected_documents": selection_rows,
            "selection_sha256": canonical_sha256(selection_rows),
        },
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        configuration["paths"]["target_model"], local_files_only=True, use_fast=True
    )
    plan = build_token_plan(
        selected, tokenizer, max_tokens=int(configuration["protocol"]["max_sequence_tokens"])
    )
    plan_sidecar = _persist_immutable_jsonl(output / "token_plan.jsonl", plan)
    plan_identity = {
        "path": str((output / "token_plan.jsonl").resolve()),
        **plan_sidecar,
        "token_count_min": min(row["input_token_count"] for row in plan),
        "token_count_max": max(row["input_token_count"] for row in plan),
        "token_count_total": sum(row["input_token_count"] for row in plan),
    }
    heldout = _heldout_binding(configuration["paths"]["heldout_result"])
    hellaswag = _hellaswag_binding(configuration["paths"]["hellaswag_run"])
    definition = _definition(configuration, plan_identity, hellaswag, heldout)
    fingerprint = canonical_sha256(definition)
    definition = {**definition, "request_fingerprint": fingerprint}
    immutable_hashed_json(output / "run_definition.json", definition)

    accumulator, start, prior_checkpoint = resume_accumulator(
        output / "checkpoints", fingerprint=fingerprint, plan=plan
    )
    status = {
        "schema_version": 1,
        "status": "running",
        "request_fingerprint": fingerprint,
        "completed_requests": start,
        "expected_requests": REQUESTS,
        "resumed_from": str(prior_checkpoint) if prior_checkpoint else None,
        "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    hashed_json(output / "status.json", status)

    src_root = REPOSITORY_ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from kvcache_transfer.modeling import load_qwen3
    from .runtime import ReverseMappedPrefixCacheBuilder

    runtime = configuration["runtime"]
    target = load_qwen3(
        configuration["paths"]["target_model"],
        devices=runtime["target_devices"],
        layer_counts=runtime["target_layer_counts"],
        role="source",
        attention_implementation=runtime["attention_implementation"],
    )
    source = load_qwen3(
        configuration["paths"]["source_model"],
        devices=runtime["source_devices"],
        layer_counts=runtime["source_layer_counts"],
        role="target",
        attention_implementation=runtime["attention_implementation"],
    )
    builder = ReverseMappedPrefixCacheBuilder(
        source_model=source,
        target_model=target,
        mapper_manifest=configuration["paths"]["mapper_manifest"],
        token_chunk=int(runtime["mapper_token_chunk"]),
    )
    hashed_json(
        output / "runtime_setup.json",
        {
            "schema_version": 1,
            "status": "ready",
            "request_fingerprint": fingerprint,
            "source_sharding": getattr(source, "_kvcache_transfer_layer_sharding_evidence", None),
            "target_sharding": getattr(target, "_kvcache_transfer_layer_sharding_evidence", None),
            "mapper": builder.provenance,
        },
    )

    latest = prior_checkpoint
    for index in range(start, REQUESTS):
        token_ids = plan[index]["input_token_ids"]
        target_cache, query_inputs = target_prefill_with_query_inputs(target, token_ids)
        mapped_cache = builder(token_ids)
        _update_one_request(
            accumulator,
            target=target,
            mapped_cache=mapped_cache,
            target_cache=target_cache,
            query_inputs=query_inputs,
        )
        del target_cache, mapped_cache, query_inputs
        completed_requests = index + 1
        if completed_requests % int(configuration["protocol"]["checkpoint_every_requests"]) == 0 or completed_requests == REQUESTS:
            latest = save_checkpoint(
                output / "checkpoints",
                accumulator,
                completed=completed_requests,
                fingerprint=fingerprint,
                plan=plan,
            )
            hashed_json(
                output / "status.json",
                {
                    **status,
                    "completed_requests": completed_requests,
                    "latest_checkpoint": _checkpoint_artifact(latest),
                    "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
    if latest is None or latest.name != "state_0400.npz":
        raise RuntimeError("mechanism final checkpoint was not committed")
    summary = summarize_accumulator(accumulator)
    if summary["scoring_input_tokens"] != sum(
        int(row["input_token_count"]) for row in plan
    ):
        raise RuntimeError("mechanism checkpoint token coverage disagrees with token plan")
    result_value = {
        "schema_version": 1,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism",
        "direction": DIRECTION,
        "request_fingerprint": fingerprint,
        "documents": 100,
        "choices_per_document": 4,
        "scoring_sequences": REQUESTS,
        "target_layers": TARGET_LAYERS,
        "metric_scope": "deterministic local completion of paper section 4.5 on HellaSwag scoring tokens",
        **summary,
        "local_heldout_reconstruction": heldout,
        "hellaswag": hellaswag,
        "paper_comparison": {
            "qwen3_32b_to_14b_published_reference": None,
            "reason": "paper Table 4 omits Qwen3 32B->14B single-pair mechanism scalars",
            "published_attention_cosine_retention_pearson_r": 0.57,
            "published_calibration_r2k_retention_pearson_r": -0.20,
            "published_correlations_scope": "12 matched-KV pair-direction evaluations",
            "single_pair_correlation_reproduction": "not mathematically defined",
        },
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    hashed_json(output / RESULT_NAME, result_value)
    result_identity = validate_mechanism_result(
        output / RESULT_NAME, fingerprint=fingerprint
    )
    hashed_json(
        output / "status.json",
        {
            **status,
            "status": "complete",
            "completed_requests": REQUESTS,
            "latest_checkpoint": _checkpoint_artifact(latest),
            "result": result_identity,
            "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "kind": "qwen3_32b_to_14b_hellaswag_mechanism_receipt",
        "request_fingerprint": fingerprint,
        "result": result_identity,
        "run_definition": _checkpoint_artifact(output / "run_definition.json"),
        "token_plan": _checkpoint_artifact(output / "token_plan.jsonl"),
        "selection": _checkpoint_artifact(output / "selection.json"),
        "runtime_setup": _checkpoint_artifact(output / "runtime_setup.json"),
        "runtime_setup_sha256_sidecar": _checkpoint_artifact(
            output / "runtime_setup.json.sha256.json"
        ),
        "final_checkpoint": _checkpoint_artifact(latest),
        "final_checkpoint_sha256_sidecar": _checkpoint_artifact(
            latest.with_name(f"{latest.name}.sha256.json")
        ),
        "formal_hellaswag_receipt": {
            "path": hellaswag["receipt"],
            "sha256": hellaswag["receipt_sha256"],
            "nbytes": hellaswag["receipt_nbytes"],
        },
        "local_heldout_reconstruction": {
            "path": heldout["path"],
            "sha256": heldout["sha256"],
            "nbytes": heldout["nbytes"],
        },
        "gpu_idle_gate": _checkpoint_artifact(output / "gpu_idle_gate.json"),
        "paper_reference_boundary": {
            "single_pair_reverse_values_published": False,
            "correlations_are_cross_pair_n12_only": True,
        },
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    hashed_json(output / RECEIPT_NAME, receipt)
    validated = validate_completed_output(output)
    if validated is None:
        raise RuntimeError("mechanism receipt validation failed")
    return validated


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configuration = load_mechanism_config()
    output = (
        configuration["paths"]["output"]
        if args.output_dir is None
        else args.output_dir.resolve()
    )
    try:
        output.relative_to(EXPERIMENT_ROOT)
    except ValueError as exc:
        raise ValueError("mechanism output escaped the reverse experiment") from exc
    if args.dry_run:
        rows = load_hellaswag_rows(configuration["dataset"])
        selected = select_documents(
            rows,
            count=int(configuration["protocol"]["documents"]),
            seed=str(configuration["protocol"]["selection_seed"]),
        )
        print(json.dumps({
            "status": "dry_run",
            "documents": len(selected),
            "requests": len(selected) * 4,
            "dataset_sha256": EXPECTED_VALIDATION_SHA256,
            "paper_single_pair_reference_available": False,
        }, ensure_ascii=False, indent=2))
        return 0
    completed = validate_completed_output(output) if output.exists() else None
    if completed is not None:
        print(json.dumps(completed, ensure_ascii=False, indent=2))
        return 0
    lock = acquire_gpu_lock(blocking=True)
    try:
        output.mkdir(parents=True, exist_ok=True)
        idle = wait_for_idle_gpus({"runtime": configuration["runtime"]})
        hashed_json(output / "gpu_idle_gate.json", idle)
        result = run_mechanism(configuration, output)
    finally:
        lock.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "MechanismAccumulator",
    "StreamingR2State",
    "build_token_plan",
    "layer_mechanism_metrics",
    "load_mechanism_config",
    "main",
    "parse_args",
    "preprocess_hellaswag",
    "resume_accumulator",
    "save_checkpoint",
    "select_documents",
    "summarize_accumulator",
    "validate_completed_output",
    "validate_mechanism_result",
]
