"""Direct Preference Optimization entry: ``torchrun ... -m scratch_cs336.posttrain.dpo --config_path <yaml>``.

The reference is a frozen copy of the initial policy, parallelized with the same mesh and run forward-only inside
the batch source: every step's batch carries the reference log-probs of its pairs. Compared with precomputing them
for the whole dataset this needs no extra pass, no cache to invalidate, and keeps the source a pure function of the
step (resume restores nothing); it costs one sharded forward per step and the reference's sharded fp32 weights
(no gradients or optimizer state).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import draccus
import torch

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh, destroy_distributed, init_distributed
from scratch_cs336.parallel import Backend, ParallelConfig, ParallelModel, Strategy, parallelize
from scratch_cs336.posttrain.data import PreferenceSource, load_rows
from scratch_cs336.posttrain.losses import dpo_loss, sequence_log_probs
from scratch_cs336.posttrain.policy import PolicyConfig, load_checked_tokenizer, load_policy, policy_config
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.optim import build_parallel_optimizer
from scratch_cs336.train.trainer import Batch, BatchSource, LossOutput, Trainer


@dataclass(frozen=True)
class DPOConfig:
    output_dir: str = ""
    policy: PolicyConfig = field(default_factory=PolicyConfig)  # also the frozen reference
    train_data: str = ""  # jsonl rows {"prompt": str | messages, "chosen": str, "rejected": str}
    beta: float = 0.1
    seq_len: int = 512
    parallel: ParallelConfig = field(
        default_factory=lambda: ParallelConfig(MeshConfig(), Strategy.FSDP, Backend.SCRATCH)
    )
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(num_steps=1000, global_batch_size=32, micro_batch_size=2)
    )


def pair_log_probs(model: ParallelModel, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequence log-probs of the chosen and rejected responses, from one forward over both."""
    input_ids = torch.cat([batch["chosen_input_ids"], batch["rejected_input_ids"]])
    labels = torch.cat([batch["chosen_labels"], batch["rejected_labels"]])
    return sequence_log_probs(model(input_ids), labels).chunk(2)


class ReferenceScored:
    """Adds ``ref_chosen_logps`` / ``ref_rejected_logps`` ``[rows]`` to a preference source's batches.

    Collective (the reference may be sharded): every rank calls :meth:`batch` for the same steps, as the Trainer does.
    """

    def __init__(self, source: BatchSource, reference: ParallelModel, device: torch.device, micro_batch_size: int):
        self.source = source
        self.reference = reference
        self.device = device
        self.micro_batch_size = micro_batch_size

    @torch.no_grad()
    def batch(self, step: int) -> Batch:
        batch = self.source.batch(step)
        rows = next(iter(batch.values())).shape[0]
        chosen, rejected = [], []
        for start in range(0, rows, self.micro_batch_size):
            micro = {k: v[start : start + self.micro_batch_size].to(self.device) for k, v in batch.items()}
            c, r = pair_log_probs(self.reference, micro)
            chosen.append(c.cpu())
            rejected.append(r.cpu())
        return batch | {"ref_chosen_logps": torch.cat(chosen), "ref_rejected_logps": torch.cat(rejected)}

    def state_dict(self) -> dict[str, Any]:
        return self.source.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.source.load_state_dict(state)


def dpo_batch_loss(pmodel: ParallelModel, batch: Batch, beta: float) -> LossOutput:
    """Sum of per-pair DPO losses; weight = pairs. Metrics are per-pair implicit rewards and their margin."""
    chosen, rejected = pair_log_probs(pmodel, batch)
    loss, meta = dpo_loss(chosen, rejected, batch["ref_chosen_logps"], batch["ref_rejected_logps"], beta)
    margin = meta["chosen_reward"] - meta["rejected_reward"]
    metrics = {
        "chosen_reward": meta["chosen_reward"].sum(),
        "rejected_reward": meta["rejected_reward"].sum(),
        "reward_margin": margin.sum(),
        "reward_accuracy": (margin > 0).float().sum(),
    }
    return LossOutput(loss.sum(), torch.tensor(float(loss.numel()), device=loss.device), metrics)


def run(cfg: DPOConfig, env: DistEnv) -> None:
    if not cfg.output_dir or not cfg.train_data:
        raise ValueError("output_dir and train_data are required")
    mesh = build_mesh(cfg.parallel.mesh, env)
    policy_config(cfg.policy).check_tensor_parallel(mesh.tp_size)
    model = load_policy(cfg.policy, cfg.trainer.seed)
    tokenizer = load_checked_tokenizer(cfg.policy, model)
    pairs = PreferenceSource.from_rows(
        load_rows(cfg.train_data), tokenizer, cfg.seq_len, cfg.trainer.global_batch_size, mesh.dp_rank, mesh.dp_size,
        cfg.trainer.seed,
    )  # fmt: skip
    reference = parallelize(copy.deepcopy(model).to(env.device).requires_grad_(False), mesh, cfg.parallel)
    reference.module.eval()
    pmodel = parallelize(model.to(env.device), mesh, cfg.parallel)
    optimizer = build_parallel_optimizer(pmodel, cfg.parallel, cfg.trainer.optimizer)
    source = ReferenceScored(pairs, reference, env.device, cfg.trainer.micro_batch_size)
    Trainer(
        cfg.trainer, pmodel, optimizer, source, partial(dpo_batch_loss, beta=cfg.beta), mesh, Path(cfg.output_dir)
    ).fit()


def main() -> None:
    cfg = draccus.parse(config_class=DPOConfig)
    env = init_distributed()
    run(cfg, env)
    destroy_distributed()


if __name__ == "__main__":
    main()
