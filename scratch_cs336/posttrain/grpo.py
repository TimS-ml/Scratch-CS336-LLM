"""On-policy GRPO (and Dr. GRPO / RFT / MaxRL / off-policy GRPO-clip / GSPO) on the generic Trainer:
``torchrun ... -m scratch_cs336.posttrain.grpo --config_path <yaml>``.

Rollouts are a :class:`BatchSource`. Rollout batch ``k`` (``n_prompts_per_rollout`` prompts x ``group_size``
samples) is generated from the current policy when the Trainer asks for its first optimizer step and serves
``steps_per_rollout = epochs_per_rollout_batch * rollout_rows / global_batch_size`` consecutive steps, each on the
next ``global_batch_size`` slice. Every dp rank samples, scores and normalizes its own prompt shard (groups never
straddle ranks). The current rollout batch is part of the source state, so resuming mid-rollout replays the exact
same data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import draccus
import torch
import torch.distributed as dist
from torch import Tensor

from scratch_cs336.data.chat import IM_END
from scratch_cs336.data.permutation import derive_seed
from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh, destroy_distributed, init_distributed
from scratch_cs336.distributed.mesh import Mesh
from scratch_cs336.models import ModelConfig
from scratch_cs336.parallel import Backend, ParallelConfig, ParallelModel, Strategy, parallelize
from scratch_cs336.posttrain import losses as L
from scratch_cs336.posttrain.data import PromptFormat, PromptSource, load_rows, pack_prompt_response, rl_rows
from scratch_cs336.posttrain.policy import PolicyConfig, load_checked_tokenizer, load_policy, policy_config
from scratch_cs336.posttrain.rewards import REWARD_FNS
from scratch_cs336.posttrain.rollout import (
    RolloutEngine,
    SamplingParams,
    TorchRolloutEngine,
    VLLMRolloutEngine,
    VLLMServerConfig,
)
from scratch_cs336.tokenizer import Tokenizer
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.optim import build_parallel_optimizer
from scratch_cs336.train.trainer import Batch, LossOutput, Trainer

# (response token ids, decoded response, ground truth) -> {"reward", "format_reward", "answer_reward"}
RolloutRewardFn = Callable[[list[int], str, str], dict[str, float]]


class RewardKind(StrEnum):
    R1_ZERO = "r1_zero"
    BOXED = "boxed"
    LAST_NUMBER = "last_number"
    TARGET_TOKEN = "target_token"  # toy verifiable task: fraction of response tokens equal to `target_token`


class EngineKind(StrEnum):
    TORCH = "torch"
    VLLM = "vllm"


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    stop: tuple[str, ...] = ()  # e.g. ("</answer>",) for r1_zero; EOS (and <|im_end|> for chat) always stop


@dataclass(frozen=True)
class EngineConfig:
    kind: EngineKind = EngineKind.TORCH
    max_batch_size: int = 64  # torch engine: sequences sampled together
    vllm: VLLMServerConfig = field(default_factory=lambda: VLLMServerConfig(model_id=""))


@dataclass(frozen=True)
class GRPOAlgorithm:
    group_size: int = 8
    n_prompts_per_rollout: int = 32  # global; rollout rows = n_prompts_per_rollout * group_size
    epochs_per_rollout_batch: int = 1
    baseline: L.Baseline = L.Baseline.MEAN
    advantage_normalizer: L.AdvantageNormalizer = L.AdvantageNormalizer.STD
    advantage_eps: float = 1e-6
    importance_reweighting: L.ImportanceReweighting = L.ImportanceReweighting.NONE
    cliprange: float = 0.2
    loss_normalization: L.LossNormalization = L.LossNormalization.SEQUENCE
    normalization_constant: float | None = None  # loss_normalization=constant: total response tokens are / this

    @property
    def rollout_rows(self) -> int:
        return self.n_prompts_per_rollout * self.group_size

    def steps_per_rollout(self, global_batch_size: int) -> int:
        if self.rollout_rows % global_batch_size:
            raise ValueError(f"rollout rows {self.rollout_rows} not divisible by global_batch_size {global_batch_size}")
        return self.epochs_per_rollout_batch * self.rollout_rows // global_batch_size


class RolloutBatches:
    """``BatchSource`` of policy rollouts. Fields (rows = this rank's slice of the step's global batch):
    ``input_ids`` / ``labels`` / ``response_mask`` ``[rows, T]``, ``advantages`` / ``rewards`` ``[rows]`` and, when
    importance reweighting is on, ``old_log_probs`` ``[rows, T]`` from the policy that sampled them.

    Collective: rollouts sync weights from and (for ``old_log_probs``) run the sharded policy on every rank.
    """

    def __init__(
        self,
        prompts: PromptSource,
        engine: RolloutEngine,
        pmodel: ParallelModel,
        tokenizer: Tokenizer,
        reward_fn: RolloutRewardFn,
        sampling: SamplingParams,
        algo: GRPOAlgorithm,
        global_batch_size: int,
        mesh: Mesh,
        micro_batch_size: int,
        seed: int,
    ):
        self.prompts = prompts
        self.engine = engine
        self.pmodel = pmodel
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.sampling = sampling
        self.algo = algo
        self.mesh = mesh
        self.micro_batch_size = micro_batch_size
        self.seed = seed
        self.steps_per_rollout = algo.steps_per_rollout(global_batch_size)
        self.rows_per_step = global_batch_size // mesh.dp_size
        self.slices_per_rollout = algo.rollout_rows // global_batch_size
        self.rollout_index: int | None = None
        self.rollout: Batch = {}

    def batch(self, step: int) -> Batch:
        k, position = divmod(step, self.steps_per_rollout)
        if self.rollout_index != k:
            self.rollout = self._generate(k)
            self.rollout_index = k
        start = (position % self.slices_per_rollout) * self.rows_per_step
        return {name: t[start : start + self.rows_per_step] for name, t in self.rollout.items()}

    def _generate(self, k: int) -> Batch:
        algo = self.algo
        indices = self.prompts.batch(k)["example_index"].tolist()
        prompts = [self.prompts.prompt(i) for i in indices]
        self.engine.sync_weights(self.pmodel)
        params = replace(self.sampling, seed=derive_seed(self.seed, k, self.mesh.dp_rank) & 0x7FFF_FFFF)
        completions = self.engine.generate(prompts, algo.group_size, params)
        repeated_prompts = [p for p in prompts for _ in range(algo.group_size)]
        responses = [c for group in completions for c in group]
        truths = [self.prompts.ground_truth(i) for i in indices for _ in range(algo.group_size)]
        scores = [self.reward_fn(r, self.tokenizer.decode(r), gt) for r, gt in zip(responses, truths, strict=True)]
        rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)
        advantages, _ = L.compute_group_normalized_rewards(
            rewards, algo.group_size, algo.baseline, algo.advantage_eps, algo.advantage_normalizer
        )
        rollout = pack_prompt_response(repeated_prompts, responses)
        rollout |= {"advantages": advantages, "rewards": rewards}
        if L.ImportanceReweighting(algo.importance_reweighting) is not L.ImportanceReweighting.NONE:
            rollout["old_log_probs"] = self._log_probs(rollout)
        return rollout

    @torch.no_grad()
    def _log_probs(self, rollout: Batch) -> Tensor:
        out = []
        device = self.mesh.env.device
        for start in range(0, rollout["input_ids"].shape[0], self.micro_batch_size):
            rows = slice(start, start + self.micro_batch_size)
            logits = self.pmodel(rollout["input_ids"][rows].to(device))
            out.append(L.token_log_probs(logits, rollout["labels"][rows].to(device)).cpu())
        return torch.cat(out)

    def state_dict(self) -> dict[str, Any]:
        return {"prompts": self.prompts.state_dict(), "rollout_index": self.rollout_index, "rollout": self.rollout}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.prompts.load_state_dict(state["prompts"])
        self.rollout_index = state["rollout_index"]
        self.rollout = dict(state["rollout"])


def grpo_loss(pmodel: ParallelModel, batch: Batch, algo: GRPOAlgorithm, global_batch_size: int) -> LossOutput:
    """``sequence``: sum of per-sequence masked means, weight = sequences (global mean over sequences).
    ``constant``: masked token sum, weight = ``Z * rows / G`` so the global weight is the constant ``Z``.
    Metrics are per-sequence means: reward, response length, token entropy, clip fraction."""
    logits = pmodel(batch["input_ids"])
    mask = batch["response_mask"]
    log_probs = L.token_log_probs(logits, batch["labels"])
    per_token, meta = L.compute_policy_gradient_loss(
        batch["advantages"], log_probs, algo.importance_reweighting, batch.get("old_log_probs"), algo.cliprange, mask
    )
    rows = mask.shape[0]
    if L.LossNormalization(algo.loss_normalization) is L.LossNormalization.SEQUENCE:
        loss_sum = L.masked_mean(per_token, mask, dim=-1).sum()
        weight = float(rows)
    else:
        if algo.normalization_constant is None:
            raise ValueError("loss_normalization=constant requires normalization_constant")
        loss_sum = L.masked_normalize(per_token, mask, 1.0)
        weight = algo.normalization_constant * rows / global_batch_size
    per_row = weight / rows  # metric sums / global weight = per-sequence means
    with torch.no_grad():
        metrics = {
            "reward": batch["rewards"].sum() * per_row,
            "response_len": mask.sum() * per_row,
            "entropy": L.masked_mean(L.token_entropy(logits), mask, dim=-1).sum() * per_row,
        }
        if "clipped" in meta:
            metrics["clip_fraction"] = L.masked_mean(meta["clipped"], mask, dim=-1).sum() * per_row
    return LossOutput(loss_sum, torch.tensor(weight, device=loss_sum.device), metrics)


class RolloutEvaluator:
    """Mean reward of one sample per held-out prompt (collective; each rank scores every dp-th prompt)."""

    def __init__(
        self,
        prompts: Sequence[list[int]],
        ground_truths: Sequence[str],
        engine: RolloutEngine,
        tokenizer: Tokenizer,
        reward_fn: RolloutRewardFn,
        sampling: SamplingParams,
        mesh: Mesh,
    ):
        self.prompts = list(prompts)[mesh.dp_rank :: mesh.dp_size]
        self.ground_truths = list(ground_truths)[mesh.dp_rank :: mesh.dp_size]
        self.engine = engine
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.sampling = sampling
        self.mesh = mesh

    def __call__(self, pmodel: ParallelModel) -> dict[str, float]:
        self.engine.sync_weights(pmodel)
        completions = self.engine.generate(self.prompts, 1, self.sampling) if self.prompts else []
        keys = ("reward", "format_reward", "answer_reward")
        totals = torch.zeros(len(keys) + 2, dtype=torch.float64, device=self.mesh.env.device)
        for (response,), truth in zip(completions, self.ground_truths, strict=True):
            score = self.reward_fn(response, self.tokenizer.decode(response), truth)
            row = [score.get(k, 0.0) for k in keys] + [len(response), 1.0]
            totals += torch.tensor(row, dtype=torch.float64, device=totals.device)
        dist.all_reduce(totals, group=self.mesh.dp_group)
        n = max(totals[-1].item(), 1.0)
        means = {k: totals[i].item() / n for i, k in enumerate(keys)}
        return {
            "reward": means["reward"],
            "format_reward": means["format_reward"],
            "accuracy": means["answer_reward"],
            "response_len": totals[-2].item() / n,
            "examples": totals[-1].item(),
        }


def text_reward(fn: L.RewardFn) -> RolloutRewardFn:
    return lambda ids, text, truth: fn(text, truth)


def target_token_reward(target: int) -> RolloutRewardFn:
    def score(ids: list[int], text: str, truth: str) -> dict[str, float]:
        reward = sum(t == target for t in ids) / max(len(ids), 1)
        return {"reward": reward, "format_reward": 1.0, "answer_reward": reward}

    return score


@dataclass(frozen=True)
class GRPOConfig:
    output_dir: str = ""
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train_data: str = "gsm8k:train"  # load_rows spec; rows {"question", "ground_truth" | GSM8K "answer"}
    eval_data: str | None = None
    eval_prompts: int = 256  # first N eval rows
    prompt_format: PromptFormat = PromptFormat.R1_ZERO
    reward: RewardKind = RewardKind.R1_ZERO
    target_token: int | None = None  # reward=target_token
    max_prompt_len: int = 512
    sampling: SamplingConfig = field(default_factory=lambda: SamplingConfig(stop=("</answer>",)))
    algo: GRPOAlgorithm = field(default_factory=GRPOAlgorithm)
    engine: EngineConfig = field(default_factory=EngineConfig)
    parallel: ParallelConfig = field(
        default_factory=lambda: ParallelConfig(MeshConfig(), Strategy.FSDP, Backend.SCRATCH)
    )
    # global_batch_size = rollout rows per optimizer step (on-policy: n_prompts_per_rollout * group_size).
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(num_steps=200, global_batch_size=256, micro_batch_size=2)
    )


def build_reward(reward: RewardKind, target_token: int | None = None) -> RolloutRewardFn:
    if RewardKind(reward) is RewardKind.TARGET_TOKEN:
        if target_token is None:
            raise ValueError("reward=target_token requires target_token")
        return target_token_reward(target_token)
    return text_reward(REWARD_FNS[reward])


def sampling_params(sampling: SamplingConfig, fmt: PromptFormat, tokenizer: Tokenizer) -> SamplingParams:
    stop_ids = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    if PromptFormat(fmt) is PromptFormat.CHAT:
        stop_ids += tokenizer.encode(IM_END)
    return SamplingParams(
        sampling.max_tokens, sampling.temperature, sampling.top_p, tuple(stop_ids), tuple(sampling.stop)
    )


def build_engine(
    cfg: EngineConfig, parallel: ParallelConfig, model_cfg: ModelConfig, tokenizer: Tokenizer, env: DistEnv
) -> RolloutEngine:
    if EngineKind(cfg.kind) is EngineKind.VLLM:
        return VLLMRolloutEngine(cfg.vllm, model_cfg, env.device)
    dtype = parallel.mixed_precision_dtype or torch.float32
    return TorchRolloutEngine(model_cfg, env.device, dtype, cfg.max_batch_size, tokenizer)


def held_out_evaluator(
    rows: list[dict[str, Any]],
    tokenizer: Tokenizer,
    fmt: PromptFormat,
    max_prompt_len: int,
    engine: RolloutEngine,
    reward_fn: RolloutRewardFn,
    sampling: SamplingParams,
    mesh: Mesh,
) -> RolloutEvaluator:
    """Greedy decoding on ``{"question", "ground_truth" | "answer"}`` rows."""
    held_out = PromptSource.from_rows(rl_rows(rows), tokenizer, fmt, max_prompt_len, 1, 0, 1, seed=0)
    greedy = replace(sampling, temperature=0.0)
    return RolloutEvaluator(held_out.prompts, held_out.ground_truths, engine, tokenizer, reward_fn, greedy, mesh)


def run(cfg: GRPOConfig, env: DistEnv) -> None:
    if not cfg.output_dir:
        raise ValueError("output_dir is required")
    mesh = build_mesh(cfg.parallel.mesh, env)
    policy_config(cfg.policy).check_tensor_parallel(mesh.tp_size)
    model = load_policy(cfg.policy, cfg.trainer.seed)
    tokenizer = load_checked_tokenizer(cfg.policy, model)
    prompts = PromptSource.from_rows(
        rl_rows(load_rows(cfg.train_data)), tokenizer, cfg.prompt_format, cfg.max_prompt_len,
        cfg.algo.n_prompts_per_rollout, mesh.dp_rank, mesh.dp_size, cfg.trainer.seed,
    )  # fmt: skip
    reward_fn = build_reward(cfg.reward, cfg.target_token)
    sampling = sampling_params(cfg.sampling, cfg.prompt_format, tokenizer)
    engine = build_engine(cfg.engine, cfg.parallel, model.cfg, tokenizer, env)
    pmodel = parallelize(model.to(env.device), mesh, cfg.parallel)
    optimizer = build_parallel_optimizer(pmodel, cfg.parallel, cfg.trainer.optimizer)
    source = RolloutBatches(
        prompts, engine, pmodel, tokenizer, reward_fn, sampling, cfg.algo, cfg.trainer.global_batch_size, mesh,
        cfg.trainer.micro_batch_size, cfg.trainer.seed,
    )  # fmt: skip
    evaluators = {}
    if cfg.eval_data is not None:
        evaluators["eval"] = held_out_evaluator(
            load_rows(cfg.eval_data)[: cfg.eval_prompts], tokenizer, cfg.prompt_format, cfg.max_prompt_len, engine,
            reward_fn, sampling, mesh,
        )  # fmt: skip
    loss_fn = partial(grpo_loss, algo=cfg.algo, global_batch_size=cfg.trainer.global_batch_size)
    Trainer(cfg.trainer, pmodel, optimizer, source, loss_fn, mesh, Path(cfg.output_dir), evaluators=evaluators).fit()


def main() -> None:
    cfg = draccus.parse(config_class=GRPOConfig)
    env = init_distributed()
    run(cfg, env)
    destroy_distributed()


if __name__ == "__main__":
    main()
