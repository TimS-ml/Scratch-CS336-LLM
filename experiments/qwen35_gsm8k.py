"""Qwen3.5-0.8B on GSM8K, CS336 hw5 setup: gsm8k -> SFT on train solutions -> GRPO (r1_zero reward) from the SFT
export -> greedy reward / accuracy on GSM8K test for both the SFT and the GRPO policy.

Prompt format: r1_zero for SFT, RL and eval. The reward checks this exact format (``</think> <answer> N </answer>``),
so SFT on r1_zero-formatted solutions teaches the format the reward needs and GRPO starts from a policy with
non-zero format reward; the raw-text prompt also avoids depending on a chat template (our ChatML rendering is not
Qwen3.5's thinking template).

python -m experiments.qwen35_gsm8k                                    # print the plan (cpu smoke)
python -m experiments.qwen35_gsm8k --device cpu --run --root /tmp/exp-qwen
python -m experiments.qwen35_gsm8k --device 4090x4 --run --root runs
"""

from __future__ import annotations

from dataclasses import dataclass

from scratch_cs336.distributed import MeshConfig
from scratch_cs336.launch import Resources
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy
from scratch_cs336.pipeline import InputPath, Step, experiment_main
from scratch_cs336.pipeline.builders import (
    GSM8K_SFT,
    GSM8K_TEST,
    GSM8K_TRAIN,
    final_model,
    grpo,
    gsm8k,
    reward_eval,
    sft,
)
from scratch_cs336.posttrain.data import PromptFormat
from scratch_cs336.posttrain.grpo import EngineConfig, EngineKind, GRPOAlgorithm, SamplingConfig
from scratch_cs336.posttrain.policy import PolicyConfig
from scratch_cs336.posttrain.rollout import VLLMServerConfig
from scratch_cs336.train import TrainerConfig
from scratch_cs336.train.optim import OptimizerConfig, ScheduleConfig

BASE = "Qwen/Qwen3.5-0.8B"
TOKENIZER = f"hf:{BASE}"
FORMAT = PromptFormat.R1_ZERO
STOP = ("</answer>",)


@dataclass(frozen=True)
class Device:
    # SFT and GRPO / eval can use different process layouts: with vLLM, rollouts get dedicated GPUs.
    sft_resources: Resources
    sft_parallel: ParallelConfig
    rl_resources: Resources
    rl_parallel: ParallelConfig
    engine: EngineConfig
    policy: PolicyConfig  # what SFT starts from
    sft_batch: int
    sft_micro: int
    sft_steps: int
    sft_lr: float
    seq_len: int
    prompts_per_rollout: int  # x group_size rollouts per optimizer step (on-policy)
    group_size: int
    rl_micro: int
    rl_steps: int
    grpo_lr: float
    max_tokens: int  # rollout / eval generation budget
    max_train: int | None = None  # GSM8K rows per split; None = all (7473 train, 1319 test)
    max_test: int | None = None
    eval_prompts: int = 256  # held-out prompts evaluated during GRPO


def _parallel(strategy: Strategy, backend: Backend, compute_dtype: str) -> ParallelConfig:
    return ParallelConfig(MeshConfig(), strategy, backend, compute_dtype)


DEVICES: dict[str, Device] = {
    # Pipeline smoke: random-init 4-layer hybrid with the real vocabulary and tokenizer, tiny subsets and budgets.
    "cpu": Device(
        sft_resources=Resources(nproc_per_node=2),
        sft_parallel=_parallel(Strategy.FSDP, Backend.SCRATCH, "float32"),
        rl_resources=Resources(nproc_per_node=2),
        rl_parallel=_parallel(Strategy.FSDP, Backend.SCRATCH, "float32"),
        engine=EngineConfig(EngineKind.TORCH, max_batch_size=16),
        policy=PolicyConfig(model="qwen3.5-tiny", tokenizer=TOKENIZER),
        sft_batch=8,
        sft_micro=1,
        sft_steps=20,
        sft_lr=3e-3,  # random init
        seq_len=256,
        prompts_per_rollout=4,
        group_size=4,
        rl_micro=2,
        rl_steps=4,
        grpo_lr=1e-3,
        max_tokens=16,
        max_train=64,
        max_test=8,
        eval_prompts=8,
    ),
    # 4 x 24 GB: everything on the 4 training ranks; rollouts by the torch engine (no KV cache: slow but exact).
    "4090x4": Device(
        sft_resources=Resources(nproc_per_node=4, gpus_per_node=4),
        sft_parallel=_parallel(Strategy.FSDP, Backend.SCRATCH, "bfloat16"),
        rl_resources=Resources(nproc_per_node=4, gpus_per_node=4),
        rl_parallel=_parallel(Strategy.FSDP, Backend.SCRATCH, "bfloat16"),
        engine=EngineConfig(EngineKind.TORCH, max_batch_size=64),
        policy=PolicyConfig(init_from=BASE, tokenizer=TOKENIZER),
        sft_batch=64,
        sft_micro=2,
        sft_steps=234,  # ~2 epochs of 7473 solutions
        sft_lr=1e-5,
        seq_len=512,
        prompts_per_rollout=32,
        group_size=8,
        rl_micro=2,
        rl_steps=200,
        grpo_lr=1e-5,
        max_tokens=512,
    ),
    # 8 x H100: SFT on all 8; GRPO and eval train on GPUs 0-6 (7 ranks) and a vLLM server owns GPU 7 (rollouts
    # over HTTP, NCCL weight sync from rank 0), so batch sizes are multiples of 7.
    "h100x8": Device(
        sft_resources=Resources(nproc_per_node=8, gpus_per_node=8),
        sft_parallel=_parallel(Strategy.FSDP, Backend.NATIVE, "bfloat16"),
        rl_resources=Resources(nproc_per_node=7, gpus_per_node=8),
        rl_parallel=_parallel(Strategy.FSDP, Backend.NATIVE, "bfloat16"),
        engine=EngineConfig(EngineKind.VLLM, vllm=VLLMServerConfig(model_id=BASE, gpu=7, gpu_memory_utilization=0.85)),
        policy=PolicyConfig(init_from=BASE, tokenizer=TOKENIZER),
        sft_batch=64,
        sft_micro=8,
        sft_steps=234,
        sft_lr=1e-5,
        seq_len=512,
        prompts_per_rollout=28,
        group_size=8,
        rl_micro=4,
        rl_steps=200,
        grpo_lr=1e-5,
        max_tokens=512,
    ),
}


@dataclass(frozen=True)
class Options:
    device: str = "cpu"
    sft_lr: float | None = None  # override the device's learning rates and step budgets
    grpo_lr: float | None = None
    sft_steps: int | None = None
    grpo_steps: int | None = None


def _trainer(steps: int, batch: int, micro: int, lr: float, warmup: int, evals: int) -> TrainerConfig:
    return TrainerConfig(
        num_steps=steps,
        global_batch_size=batch,
        micro_batch_size=micro,
        optimizer=OptimizerConfig(lr=lr, betas=(0.9, 0.95), weight_decay=0.0),
        schedule=ScheduleConfig(warmup_steps=warmup, total_steps=steps, min_lr_ratio=0.1),
        max_grad_norm=1.0,
        log_every=max(1, min(10, steps // 20)),
        eval_every=max(1, steps // evals),
        checkpoint_every=max(1, steps // 5),
    )


def build(opts: Options) -> list[Step]:
    if opts.device not in DEVICES:
        raise SystemExit(f"--device must be one of {sorted(DEVICES)}, got {opts.device!r}")
    device = DEVICES[opts.device]
    sft_steps = opts.sft_steps or device.sft_steps
    grpo_steps = opts.grpo_steps or device.rl_steps
    sft_lr = opts.sft_lr or device.sft_lr
    grpo_lr = opts.grpo_lr or device.grpo_lr

    data = gsm8k("gsm8k", FORMAT, max_train=device.max_train, max_test=device.max_test)
    sft_step = sft(
        "qwen35-gsm8k-sft",
        policy=device.policy,
        train=InputPath(data, GSM8K_SFT),
        prompt_format=FORMAT,
        seq_len=device.seq_len,
        parallel=device.sft_parallel,
        trainer=_trainer(sft_steps, device.sft_batch, device.sft_micro, sft_lr, max(1, sft_steps // 20), 1),
        resources=device.sft_resources,
    )
    sft_policy = PolicyConfig(init_from=final_model(sft_step), tokenizer=TOKENIZER)
    sampling = SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=device.max_tokens, stop=STOP)
    rollouts = device.prompts_per_rollout * device.group_size
    rl = grpo(
        "qwen35-gsm8k-grpo",
        policy=sft_policy,
        train=InputPath(data, GSM8K_TRAIN),
        eval_data=InputPath(data, GSM8K_TEST),
        eval_prompts=device.eval_prompts,
        prompt_format=FORMAT,
        sampling=sampling,
        algo=GRPOAlgorithm(group_size=device.group_size, n_prompts_per_rollout=device.prompts_per_rollout),
        engine=device.engine,
        parallel=device.rl_parallel,
        trainer=_trainer(grpo_steps, rollouts, device.rl_micro, grpo_lr, 0, 4),
        resources=device.rl_resources,
    )

    def held_out(name: str, policy: PolicyConfig) -> Step:
        return reward_eval(
            name,
            policy=policy,
            data=InputPath(data, GSM8K_TEST),
            prompt_format=FORMAT,
            sampling=sampling,
            engine=device.engine,
            parallel=device.rl_parallel,
            resources=device.rl_resources,
        )

    return [
        held_out("qwen35-gsm8k-sft-eval", sft_policy),
        held_out("qwen35-gsm8k-grpo-eval", PolicyConfig(init_from=final_model(rl), tokenizer=TOKENIZER)),
    ]


if __name__ == "__main__":
    experiment_main(build, options=Options)
