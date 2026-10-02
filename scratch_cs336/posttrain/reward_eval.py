"""Held-out reward / accuracy of a policy on verifiable prompts (e.g. GSM8K test), greedy decoding:
``torchrun ... -m scratch_cs336.posttrain.reward_eval --config_path <yaml>``; writes ``<output_dir>/eval.json``.

The policy is parallelized like in training and handed to the rollout engine through ``sync_weights``, so the
torch and vLLM engines evaluate exactly as GRPO's in-training evaluator does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import draccus
import torch

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh, destroy_distributed, init_distributed
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize
from scratch_cs336.posttrain.data import PromptFormat, load_rows
from scratch_cs336.posttrain.grpo import (
    EngineConfig,
    RewardKind,
    SamplingConfig,
    build_engine,
    build_reward,
    held_out_evaluator,
    sampling_params,
)
from scratch_cs336.posttrain.policy import PolicyConfig, load_checked_tokenizer, load_policy, policy_config

RESULT_FILE = "eval.json"


@dataclass(frozen=True)
class RewardEvalConfig:
    output_dir: str = ""
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    data: str = "gsm8k:test"  # load_rows spec; rows {"question", "ground_truth" | GSM8K "answer"}
    max_examples: int | None = None  # first N rows; None = all
    prompt_format: PromptFormat = PromptFormat.R1_ZERO
    reward: RewardKind = RewardKind.R1_ZERO
    target_token: int | None = None
    max_prompt_len: int = 512
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig(stop=("</answer>",)))
    engine: EngineConfig = field(default_factory=EngineConfig)
    parallel: ParallelConfig = field(
        default_factory=lambda: ParallelConfig(MeshConfig(), Strategy.FSDP, Backend.SCRATCH)
    )


@torch.no_grad()
def run(cfg: RewardEvalConfig, env: DistEnv) -> dict[str, float]:
    if not cfg.output_dir:
        raise ValueError("output_dir is required")
    mesh = build_mesh(cfg.parallel.mesh, env)
    policy_config(cfg.policy).check_tensor_parallel(mesh.tp_size)
    model = load_policy(cfg.policy, seed=0)
    tokenizer = load_checked_tokenizer(cfg.policy, model)
    engine = build_engine(cfg.engine, cfg.parallel, model.cfg, tokenizer, env)
    pmodel = parallelize(model.to(env.device), mesh, cfg.parallel)
    pmodel.module.eval()
    rows = load_rows(cfg.data)[: cfg.max_examples]
    evaluator = held_out_evaluator(
        rows, tokenizer, cfg.prompt_format, cfg.max_prompt_len, engine,
        build_reward(cfg.reward, cfg.target_token), sampling_params(cfg.sampling, cfg.prompt_format, tokenizer), mesh,
    )  # fmt: skip
    metrics = evaluator(pmodel)
    if env.is_main:
        out = Path(cfg.output_dir) / RESULT_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics), flush=True)
    return metrics


def main() -> None:
    cfg = draccus.parse(config_class=RewardEvalConfig)
    env = init_distributed()
    run(cfg, env)
    destroy_distributed()


if __name__ == "__main__":
    main()
