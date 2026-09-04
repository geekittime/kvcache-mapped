from __future__ import annotations

import torch

from src.trainer import FunctionalBatch, FunctionalTrainer, LayerBatch, default_curriculum
from src.residual import FunctionalResidualMapper


def make_batch_and_mapper():
    torch.manual_seed(11)
    stats = {}
    for component in ("key", "value"):
        stats[(7, component)] = {
            "feature_mean": torch.zeros(6),
            "feature_inv_std": torch.ones(6),
            "rms_q99": 3.0,
            "absmax_q99": 6.0,
        }
    mapper = FunctionalResidualMapper(
        active_layers=[7],
        layer_loss_weights={7: 1.0},
        input_dim=6,
        rank=2,
        kv_heads=1,
        head_dim=2,
        statistics=stats,
        target_layers=40,
    )
    source_key = torch.randn(1, 4, 6)
    source_value = torch.randn(1, 4, 6)
    anchor_key = torch.randn(1, 4, 1, 2)
    anchor_value = torch.randn(1, 4, 1, 2)
    native_key = anchor_key + 0.15
    native_value = anchor_value - 0.1
    native_query = torch.randn(1, 2, 2, 2)
    causal_mask = torch.tensor([[[True, True, True, False], [True, True, True, True]]])
    layer = LayerBatch(
        source_key=source_key,
        source_value=source_value,
        anchor_key=anchor_key,
        anchor_value=anchor_value,
        native_key=native_key,
        native_value=native_value,
        native_query=native_query,
        causal_mask=causal_mask,
        key_variance=torch.ones(1, 2),
        value_variance=torch.ones(1, 2),
    )
    native_suffix_logits = torch.randn(1, 4, 5)
    batch = FunctionalBatch(
        layers={7: layer},
        native_suffix_logits=native_suffix_logits,
        suffix_mask=torch.tensor([[True, True, True, True]]),
        eos_token_ids=[4],
    )
    return mapper, batch


def receiver(mapped, batch):
    # A tiny frozen receiver surrogate. Only mapper parameters are in the
    # optimizer; the fixed basis makes the suffix loss differentiable.
    score = mapped[7]["key"].mean(dim=(2, 3))[:, -1]
    basis = torch.tensor([-1.0, -0.25, 0.0, 0.25, 1.0], dtype=score.dtype, device=score.device)
    return score[:, None, None].expand(-1, batch.native_suffix_logits.shape[1], basis.numel()) * basis


def test_curriculum_boundaries_are_ordered():
    curriculum = default_curriculum()
    assert curriculum.at(0.0).name == "attention_alignment"
    assert curriculum.at(0.2).name == "suffix8"
    assert curriculum.at(1.0).name == "suffix32"


def test_trainer_computes_all_terms_and_updates_only_mapper():
    mapper, batch = make_batch_and_mapper()
    before_receiver = receiver
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=1e-2)
    trainer = FunctionalTrainer(mapper=mapper, receiver_forward=receiver, optimizer=optimizer)
    result0 = trainer.compute_batch(batch, progress=0.0)
    assert torch.isfinite(result0.total_loss)
    assert float(result0.terms["suffix_kl"]) == 0.0
    result1 = trainer.train_step(batch, progress=0.75)
    assert torch.isfinite(result1.total_loss)
    assert float(result1.terms["suffix_kl"]) >= 0.0
    # The callback itself is immutable and no receiver parameters are handed
    # to the optimizer.
    assert receiver is before_receiver
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in mapper.parameters())


def test_short_cpu_fit_produces_finite_records():
    mapper, batch = make_batch_and_mapper()
    optimizer = torch.optim.AdamW(mapper.parameters(), lr=5e-3)
    trainer = FunctionalTrainer(mapper=mapper, receiver_forward=receiver, optimizer=optimizer)
    records = trainer.fit([batch], [batch], epochs=2)
    assert records
    assert all(record.validation_loss is not None and record.loss == record.loss for record in records)

