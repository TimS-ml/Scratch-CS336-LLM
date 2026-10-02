"""Supervised fine-tuning entry: ``torchrun ... -m scratch_cs336.posttrain.sft --config_path <yaml>``.

Chats (``messages`` / ``prompt,response`` / ``question,answer`` rows) are rendered as ChatML and trained with
token-mean cross entropy on assistant tokens only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import draccus

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh, destroy_distributed, init_distributed
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize
from scratch_cs336.posttrain.data import PromptFormat, SFTSource, load_rows
from scratch_cs336.posttrain.policy import PolicyConfig, load_checked_tokenizer, load_policy, policy_config
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.optim import build_parallel_optimizer
from scratch_cs336.train.pretrain import lm_loss
from scratch_cs336.train.trainer import Trainer


@dataclass(frozen=True)
class SFTConfig:
    output_dir: str = ""
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train_data: str = ""  # load_rows spec: jsonl(.gz) path or "gsm8k:<split>"
    prompt_format: PromptFormat = PromptFormat.CHAT  # raw formats need single-turn prompt/response rows
    seq_len: int = 512
    parallel: ParallelConfig = field(
        default_factory=lambda: ParallelConfig(MeshConfig(), Strategy.FSDP, Backend.SCRATCH)
    )
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(num_steps=1000, global_batch_size=32, micro_batch_size=4)
    )


def run(cfg: SFTConfig, env: DistEnv) -> None:
    if not cfg.output_dir or not cfg.train_data:
        raise ValueError("output_dir and train_data are required")
    mesh = build_mesh(cfg.parallel.mesh, env)
    policy_config(cfg.policy).check_tensor_parallel(mesh.tp_size)
    model = load_policy(cfg.policy, cfg.trainer.seed)
    tokenizer = load_checked_tokenizer(cfg.policy, model)
    source = SFTSource.from_rows(
        load_rows(cfg.train_data), tokenizer, cfg.seq_len, cfg.trainer.global_batch_size, mesh.dp_rank, mesh.dp_size,
        cfg.trainer.seed, cfg.prompt_format,
    )  # fmt: skip
    flops_per_token = model.flops_per_token(cfg.seq_len)
    pmodel = parallelize(model.to(env.device), mesh, cfg.parallel)
    optimizer = build_parallel_optimizer(pmodel, cfg.parallel, cfg.trainer.optimizer)
    Trainer(
        cfg.trainer, pmodel, optimizer, source, lm_loss, mesh, Path(cfg.output_dir), flops_per_token=flops_per_token
    ).fit()


def main() -> None:
    cfg = draccus.parse(config_class=SFTConfig)
    env = init_distributed()
    run(cfg, env)
    destroy_distributed()


if __name__ == "__main__":
    main()
