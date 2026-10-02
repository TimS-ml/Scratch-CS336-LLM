"""Generic data-parallel training loop: any model, any loss, any parallel strategy.

Per optimizer step every rank takes its slice of the global batch, splits it into micro-batches and backpropagates
the *sum* of per-unit losses of each one (gradient communication only after the last). The units (tokens, sequences,
...) are counted globally, so ``grad * dp_size / global_weight`` (DP averages grads over ranks) is the exact gradient
of the global mean loss, however unevenly the units fall across ranks and micro-batches.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from scratch_cs336.checkpoint import Checkpointer, export_full
from scratch_cs336.distributed.mesh import Mesh
from scratch_cs336.parallel.api import ParallelModel
from scratch_cs336.tracking import Tracker, build_tracker
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.optim import lr_at

CHECKPOINT_DIR = "checkpoints"
FINAL_DIR = "final"
MODEL_CONFIG_FILE = "config.json"

# Dense bf16 tensor-core peak (FP32 accumulate) per device, matched by substring of the CUDA device name, in order.
PEAK_BF16_FLOPS: tuple[tuple[str, float], ...] = (
    ("B200", 2.25e15),
    ("H200", 989e12),
    ("H100 PCIe", 756e12),
    ("H100", 989e12),
    ("A100", 312e12),
    ("L40S", 362e12),
    ("RTX 4090", 165.2e12),
)

Batch = dict[str, Tensor]


class BatchSource(Protocol):
    """Deterministic and random-access by optimizer step; ``batch(step)`` is this rank's slice of the global batch."""

    def batch(self, step: int) -> Batch: ...
    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...


@dataclass
class LossOutput:
    loss_sum: Tensor  # scalar with grad: SUM of per-unit losses in this micro-batch
    weight: Tensor  # scalar: number of units in this micro-batch
    metrics: dict[str, Tensor] = field(default_factory=dict)  # extra sums, reported as global_sum / global_weight


LossFn = Callable[[ParallelModel, Batch], LossOutput]
Evaluator = Callable[[ParallelModel], dict[str, float]]


def peak_flops(device: torch.device) -> float | None:
    """Peak dense bf16 FLOP/s of ``device``; None when unknown (CPU, unlisted GPUs)."""
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device)
    return next((flops for key, flops in PEAK_BF16_FLOPS if key in name), None)


def _rng_state() -> dict[str, Tensor]:
    state = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _set_rng_state(state: dict[str, Tensor]) -> None:
    torch.set_rng_state(state["cpu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


class Trainer:
    """Collective: construct and ``fit`` on every rank.

    ``output_dir`` receives ``checkpoints/`` (sharded, for resume), ``final/`` (full export + model ``config.json``)
    and ``metrics.jsonl``. ``on_step_end(n)`` runs after ``n`` optimizer steps have completed.
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        pmodel: ParallelModel,
        optimizer: torch.optim.Optimizer,
        train: BatchSource,
        loss_fn: LossFn,
        mesh: Mesh,
        output_dir: Path,
        evaluators: Mapping[str, Evaluator] = {},  # noqa: B006 - read-only
        flops_per_token: float | None = None,
        on_step_end: Callable[[int], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.pmodel = pmodel
        self.optimizer = optimizer
        self.train = train
        self.loss_fn = loss_fn
        self.mesh = mesh
        self.output_dir = Path(output_dir)
        self.evaluators = dict(evaluators)
        self.flops_per_token = flops_per_token
        self.on_step_end = on_step_end
        self.grad_accum = cfg.grad_accum(mesh.dp_size)
        self.rows_per_rank = cfg.global_batch_size // mesh.dp_size
        self.checkpointer = Checkpointer(
            self.output_dir / CHECKPOINT_DIR, mesh, cfg.keep_last, cfg.permanent_every, cfg.async_checkpoint
        )
        self.step = 0

    def fit(self) -> None:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        self._resume()
        tracker = build_tracker(cfg.tracker, self.output_dir, self.mesh.env.is_main)
        try:
            tracker.log_config(dataclasses.asdict(cfg))
            self.pmodel.module.train()
            window_start, window_steps, window_tokens = time.perf_counter(), 0, 0
            while self.step < cfg.num_steps:
                metrics, tokens = self._train_step(self.step)
                self.step += 1
                window_steps += 1
                window_tokens += tokens
                last = self.step == cfg.num_steps
                if last or self.step % cfg.log_every == 0:
                    record = {k: v.item() if isinstance(v, Tensor) else v for k, v in metrics.items()}
                    elapsed = time.perf_counter() - window_start  # .item() above synchronized the device
                    record |= self._throughput(elapsed, window_steps, window_tokens)
                    self._log(tracker, record)
                    window_start, window_steps, window_tokens = time.perf_counter(), 0, 0
                if self.evaluators and (last or (cfg.eval_every and self.step % cfg.eval_every == 0)):
                    self._log(tracker, self.evaluate())
                    window_start, window_steps, window_tokens = time.perf_counter(), 0, 0
                if last or (cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0):
                    self.save_checkpoint()
                if self.on_step_end is not None:
                    self.on_step_end(self.step)
            self.checkpointer.wait()
            self._export()
        finally:
            tracker.finish()

    def _train_step(self, step: int) -> tuple[dict[str, Tensor | float], int]:
        """One optimizer step; returns (metrics, global tokens in the batch or 0)."""
        batch = self.train.batch(step)
        rows = {k: v.shape[0] for k, v in batch.items()}
        if set(rows.values()) != {self.rows_per_rank}:
            raise ValueError(f"step {step}: expected {self.rows_per_rank} rows per rank in every field, got {rows}")
        lr = lr_at(step, self.cfg.optimizer.lr, self.cfg.schedule)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        device, mb = self.mesh.env.device, self.cfg.micro_batch_size
        loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        weight = torch.zeros((), dtype=torch.float64, device=device)
        extra: dict[str, Tensor] = {}
        for i in range(self.grad_accum):
            micro = {k: v[i * mb : (i + 1) * mb].to(device, non_blocking=True) for k, v in batch.items()}
            with self.pmodel.no_sync() if i < self.grad_accum - 1 else nullcontext():
                out = self.loss_fn(self.pmodel, micro)
                out.loss_sum.backward()
            loss_sum += out.loss_sum.detach().double()
            weight += out.weight.detach().double()
            for k, v in out.metrics.items():
                extra[k] = extra.get(k, 0.0) + v.detach().double()
        self.pmodel.finish_grad_sync()

        names = sorted(extra)  # every rank must report the same metric names
        totals = torch.stack([loss_sum, weight, *(extra[k] for k in names)])
        dist.all_reduce(totals, group=self.mesh.dp_group)
        global_weight = totals[1].item()
        scale = self.mesh.dp_size / global_weight if global_weight > 0 else 0.0
        with torch.no_grad():
            for p in self.pmodel.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
        max_norm = self.cfg.max_grad_norm if self.cfg.max_grad_norm is not None else math.inf
        grad_norm = self.pmodel.clip_grad_norm_(max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        denom = max(global_weight, 1e-12)
        metrics: dict[str, Tensor | float] = {"loss": totals[0] / denom, "lr": lr, "grad_norm": grad_norm}
        metrics |= {k: totals[2 + i] / denom for i, k in enumerate(names)}
        metrics["weight"] = global_weight
        tokens = batch["input_ids"].numel() * self.mesh.dp_size if "input_ids" in batch else 0
        return metrics, tokens

    def _throughput(self, elapsed: float, steps: int, tokens: int) -> dict[str, float]:
        out = {"step_time": elapsed / steps}
        if tokens:
            tokens_per_s = tokens / elapsed
            out["tokens_per_s"] = tokens_per_s
            peak = peak_flops(self.mesh.env.device)
            if self.flops_per_token is not None and peak is not None:
                out["mfu"] = self.flops_per_token * tokens_per_s / (peak * self.mesh.env.world_size)
        return out

    def _log(self, tracker: Tracker, record: dict[str, float]) -> None:
        tracker.log(record, self.step)
        if self.mesh.env.is_main:
            shown = " ".join(f"{k}={v:.4g}" for k, v in record.items())
            print(f"step {self.step}: {shown}", flush=True)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run every evaluator (collective); keys are ``<evaluator>/<metric>``."""
        self.pmodel.module.eval()
        try:
            return {f"{name}/{k}": v for name, fn in self.evaluators.items() for k, v in fn(self.pmodel).items()}
        finally:
            self.pmodel.module.train()

    def save_checkpoint(self) -> None:
        state = {
            "model": self.pmodel.sharded_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train": self.train.state_dict(),
            "rng": _rng_state(),
            "step": self.step,
        }
        self.checkpointer.save(self.step, state)

    def _resume(self) -> None:
        latest = [self.checkpointer.latest_step()]
        dist.broadcast_object_list(latest, src=0)  # every rank must restore the same step
        if latest[0] is None:
            return
        state = self.checkpointer.load(latest[0])
        self.pmodel.load_sharded_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.train.load_state_dict(state["train"])
        _set_rng_state(state["rng"])
        self.step = int(state["step"])
        if self.mesh.env.is_main:
            print(f"resumed from {self.checkpointer.step_dir(self.step)}", flush=True)

    def _export(self) -> None:
        final = self.output_dir / FINAL_DIR
        export_full(self.pmodel, final)
        model_cfg = getattr(self.pmodel.module, "cfg", None)
        if self.mesh.env.is_main and dataclasses.is_dataclass(model_cfg):
            (final / MODEL_CONFIG_FILE).write_text(json.dumps(dataclasses.asdict(model_cfg), indent=2))
