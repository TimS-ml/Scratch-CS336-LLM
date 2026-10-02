"""Held-out loss is the token mean over the selected windows, whatever the dp layout."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.eval import evaluate_loss
from scratch_cs336.models import PRESETS, TransformerLM
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize

MODEL = dataclasses.replace(PRESETS["cs336-tiny"], vocab_size=64)
SEQ = 8


class ArraySource:
    def __init__(self, tokens: np.ndarray) -> None:
        self.tokens = tokens

    def num_windows(self, seq_len: int) -> int:
        return (len(self.tokens) - 1) // seq_len

    def window(self, i: int, seq_len: int) -> np.ndarray:
        return self.tokens[i * seq_len : i * seq_len + seq_len + 1]


SOURCE = ArraySource(np.random.default_rng(0).integers(0, MODEL.vocab_size, 12 * SEQ + 1))


def _model() -> TransformerLM:
    model = TransformerLM(MODEL)
    model.init_weights(0)
    return model


def _reference(n: int) -> float:
    windows = torch.from_numpy(np.stack([SOURCE.window(i, SEQ) for i in range(n)]).astype(np.int64))
    with torch.no_grad():
        logits = _model()(windows[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), windows[:, 1:].flatten()).item()


def _evaluate(env: DistEnv, strategy: Strategy, backend: Backend, batch_size: int, max_batches: int | None) -> dict:
    cfg = ParallelConfig(MeshConfig(replicate=1, shard=env.world_size, tensor=1), strategy, backend)
    mesh = build_mesh(cfg.mesh, env)
    pmodel = parallelize(_model(), mesh, cfg)
    return evaluate_loss(pmodel, SOURCE, SEQ, batch_size, mesh, max_batches)


@pytest.mark.parametrize(
    ("strategy", "backend", "batch_size", "max_batches", "windows"),
    [
        (Strategy.FSDP, Backend.SCRATCH, 3, 2, 6),  # 3 windows per rank in chunks of 2: the last forward has 1 row
        (Strategy.FSDP, Backend.NATIVE, 2, None, 12),
        (Strategy.FSDP, Backend.SCRATCH, 1, 5, 5),  # uneven split: rank 0 runs a padding forward
        (Strategy.DDP, Backend.SCRATCH, 8, None, 12),
    ],
)
def test_dp2_loss_matches_single_process_mean(strategy, backend, batch_size, max_batches, windows) -> None:
    results = run_distributed(_evaluate, 2, strategy, backend, batch_size, max_batches)
    assert results[0] == results[1]
    assert results[0]["tokens"] == windows * SEQ
    assert results[0]["loss"] == pytest.approx(_reference(windows), rel=1e-5)
    assert results[0]["perplexity"] == pytest.approx(math.exp(results[0]["loss"]))
