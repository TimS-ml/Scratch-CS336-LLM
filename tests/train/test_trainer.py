"""The generic Trainer: layout-independent training and exact resume.

Batches carry a per-row label mask, so ranks and micro-batches hold different numbers of loss tokens; only a
global token-weighted mean makes a dp=2 run with grad accumulation match the single-process run.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.eval import load_exported
from scratch_cs336.models import PRESETS, TransformerLM
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize
from scratch_cs336.train import Batch, Trainer, TrainerConfig
from scratch_cs336.train.optim import OptimizerConfig, ScheduleConfig, build_parallel_optimizer
from scratch_cs336.train.pretrain import IGNORE_INDEX, lm_loss

MODEL = dataclasses.replace(PRESETS["cs336-tiny"], vocab_size=96)
SEQ, GLOBAL_BATCH, MICRO = 16, 8, 2
STEPS = 6
CFG = TrainerConfig(
    num_steps=STEPS,
    global_batch_size=GLOBAL_BATCH,
    micro_batch_size=MICRO,
    optimizer=OptimizerConfig(lr=3e-3, weight_decay=0.1),
    schedule=ScheduleConfig(warmup_steps=2, total_steps=STEPS),
    max_grad_norm=0.5,
    log_every=1,
)


class MaskedTokens:
    """Random tokens; row ``j`` of step ``s`` keeps a pseudo-random number of its labels (the rest are ignored)."""

    def __init__(self, dp_rank: int, dp_size: int) -> None:
        self.rows = GLOBAL_BATCH // dp_size
        self.lo = dp_rank * self.rows

    def batch(self, step: int) -> Batch:
        g = torch.Generator().manual_seed(1000 + step)
        tokens = torch.randint(0, MODEL.vocab_size, (GLOBAL_BATCH, SEQ + 1), generator=g)
        keep = torch.randint(1, SEQ + 1, (GLOBAL_BATCH, 1), generator=g)
        labels = tokens[:, 1:].masked_fill(torch.arange(SEQ) >= keep, IGNORE_INDEX)
        rows = slice(self.lo, self.lo + self.rows)
        return {"input_ids": tokens[rows, :-1], "labels": labels[rows]}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        pass


class Crash(Exception):
    pass


def _model() -> TransformerLM:
    model = TransformerLM(MODEL)
    model.init_weights(0)
    return model


def _fit(env: DistEnv, parallel: ParallelConfig, cfg: TrainerConfig, out: Path, crash_after: int | None = None):
    mesh = build_mesh(parallel.mesh, env)
    pmodel = parallelize(_model(), mesh, parallel)
    optimizer = build_parallel_optimizer(pmodel, parallel, cfg.optimizer)

    def on_step_end(step: int) -> None:
        if step == crash_after:
            raise Crash

    trainer = Trainer(cfg, pmodel, optimizer, MaskedTokens(mesh.dp_rank, mesh.dp_size), lm_loss, mesh, out,
                      on_step_end=on_step_end)  # fmt: skip
    try:
        trainer.fit()
    except Crash:
        trainer.checkpointer.wait()  # the checkpoint taken before the crash is committed


def _run_layouts(env: DistEnv, cases: list[tuple[str, ParallelConfig]], root: Path) -> None:
    for name, parallel in cases:
        _fit(env, parallel, CFG, root / name)


def _final(out: Path) -> dict[str, torch.Tensor]:
    return load_file(out / "final" / "model.safetensors")


def _records(out: Path, key: str) -> list[float]:
    records = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    return [r[key] for r in records if key in r]


def _losses(out: Path) -> list[float]:
    return _records(out, "loss")


def _parallel(world: int, strategy: Strategy, backend: Backend) -> ParallelConfig:
    return ParallelConfig(MeshConfig(replicate=1, shard=world, tensor=1), strategy, backend)


DP2_CASES = {
    "fsdp-scratch": (Strategy.FSDP, Backend.SCRATCH),
    "fsdp-native": (Strategy.FSDP, Backend.NATIVE),
    "zero1-scratch": (Strategy.ZERO1, Backend.SCRATCH),
}


@functools.cache
def _parity_runs(root: Path) -> Path:
    run_distributed(_run_layouts, 1, [("reference", _parallel(1, Strategy.DDP, Backend.SCRATCH))], root)
    cases = [(name, _parallel(2, *sb)) for name, sb in DP2_CASES.items()]
    run_distributed(_run_layouts, 2, cases, root)
    return root


@pytest.fixture(scope="module")
def parity_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _parity_runs(tmp_path_factory.mktemp("parity"))


@pytest.mark.parametrize("case", list(DP2_CASES))
def test_dp2_with_accumulation_and_uneven_masks_matches_single_process(parity_root: Path, case: str) -> None:
    ref, got = parity_root / "reference", parity_root / case
    assert len(_losses(ref)) == STEPS
    torch.testing.assert_close(_losses(got), _losses(ref), rtol=1e-5, atol=1e-6)
    # Pre-clip norms pin the gradient scale itself (clipping + Adam would hide a wrong normalization in the params).
    torch.testing.assert_close(_records(got, "grad_norm"), _records(ref, "grad_norm"), rtol=1e-5, atol=1e-6)
    ref_state, got_state = _final(ref), _final(got)
    assert got_state.keys() == ref_state.keys()
    for k in ref_state:
        torch.testing.assert_close(got_state[k], ref_state[k], rtol=1e-4, atol=1e-5, msg=k)
    exported = load_exported(got / "final").state_dict()  # config.json + weights load into a fresh model
    assert all(torch.equal(exported[k], got_state[k]) for k in exported)


def _crash_and_resume(env: DistEnv, root: Path, backend: Backend) -> None:
    parallel = _parallel(2, Strategy.FSDP, backend)
    cfg = dataclasses.replace(CFG, checkpoint_every=3)
    _fit(env, parallel, cfg, root / "uninterrupted")
    _fit(env, parallel, cfg, root / "interrupted", crash_after=4)


def _resume(env: DistEnv, root: Path, backend: Backend) -> None:
    _fit(env, _parallel(2, Strategy.FSDP, backend), dataclasses.replace(CFG, checkpoint_every=3), root / "interrupted")


@pytest.mark.parametrize("backend", list(Backend))
def test_killed_run_resumes_bitwise_from_last_checkpoint(tmp_path: Path, backend: Backend) -> None:
    run_distributed(_crash_and_resume, 2, tmp_path, backend)
    assert not (tmp_path / "interrupted" / "final").exists()
    assert len(_losses(tmp_path / "interrupted")) == 4
    run_distributed(_resume, 2, tmp_path, backend)  # fresh processes, same output_dir

    full, resumed = _losses(tmp_path / "uninterrupted"), _losses(tmp_path / "interrupted")
    assert resumed[4:] == full[3:]  # steps 4..6 re-run from the step-3 checkpoint
    a, b = _final(tmp_path / "uninterrupted"), _final(tmp_path / "interrupted")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
