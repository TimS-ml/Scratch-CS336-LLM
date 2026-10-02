import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.models.presets import PRESETS
from scratch_cs336.models.transformer import TransformerLM
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize
from scratch_cs336.posttrain import losses as L
from scratch_cs336.posttrain.data import PromptFormat, PromptSource, tokenize_prompt_and_output
from scratch_cs336.posttrain.grpo import GRPOAlgorithm, RolloutBatches, grpo_loss, target_token_reward
from scratch_cs336.posttrain.rollout import SamplingParams, TorchRolloutEngine
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.optim import OptimizerConfig, ScheduleConfig, build_parallel_optimizer
from scratch_cs336.train.trainer import Trainer
from tests.posttrain.char_tokenizer import CharTokenizer
from tests.posttrain.test_losses import FIXTURES, WORDS, WordTokenizer, assert_snapshot

# ---- hw5 grpo_train_step snapshots: grpo_loss + the Trainer's normalization (grad / global weight) ----

PROMPTS = ["Hello", "Hello", "This is", "This is"]
RESPONSES = ["world", "test", "a test", "another test"]
REWARDS = torch.tensor([1.0, 0.0, 0.5, 0.0])


def _tiny_gpt2() -> torch.nn.Module:
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(
        vocab_size=len(WORDS), n_positions=16, n_ctx=16, n_embd=8, n_layer=1, n_head=2, n_inner=16,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, use_cache=False, bos_token_id=1, eos_token_id=1, pad_token_id=0,
    )  # fmt: skip
    model = transformers.GPT2LMHeadModel(cfg)
    # hw5's initial weights (transformers' GPT-2 init changed across versions, so they are stored, not re-drawn).
    init = np.load(FIXTURES / "tiny_gpt2_init.npz")
    model.load_state_dict({k: torch.from_numpy(init[k]) for k in init.files}, strict=False)
    return model.train()


def _train_step(model: torch.nn.Module, algo: GRPOAlgorithm, old_log_probs: torch.Tensor | None = None) -> torch.Tensor:
    advantages, _ = L.compute_group_normalized_rewards(
        REWARDS, algo.group_size, algo.baseline, algo.advantage_eps, algo.advantage_normalizer
    )
    batch = tokenize_prompt_and_output(PROMPTS, RESPONSES, WordTokenizer(), pad_token_id=0)
    batch |= {"advantages": advantages, "rewards": REWARDS}
    if old_log_probs is not None:
        batch["old_log_probs"] = old_log_probs
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_sum, weight = torch.zeros(()), torch.zeros(())
    for start in (0, 2):  # two micro-batches of two rollouts
        micro = {k: v[start : start + 2] for k, v in batch.items()}
        out = grpo_loss(lambda ids: model(ids).logits, micro, algo, global_batch_size=4)
        out.loss_sum.backward()
        loss_sum, weight = loss_sum + out.loss_sum.detach(), weight + out.weight
    for p in model.parameters():
        p.grad /= weight
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss_sum / weight


def _assert_train_step(name: str, model: torch.nn.Module, loss: torch.Tensor) -> None:
    assert_snapshot(name, {"loss": loss, **{f"param.{k}": p for k, p in model.named_parameters()}}, atol=1e-6)


def test_grpo_train_step_standard_on_policy():
    model = _tiny_gpt2()
    loss = _train_step(model, GRPOAlgorithm(group_size=2))
    _assert_train_step("test_grpo_train_step_standard_on_policy", model, loss)


@pytest.mark.parametrize(
    ("variant", "baseline", "normalizer"),
    [("grpo_constant", "mean", "std"), ("dr_grpo", "mean", "none"), ("rft", "none", "none"), ("maxrl", "mean", "mean")],
)
def test_grpo_train_step_variants(variant, baseline, normalizer):
    model = _tiny_gpt2()
    algo = GRPOAlgorithm(
        group_size=2, baseline=baseline, advantage_normalizer=normalizer, loss_normalization="constant",
        normalization_constant=32,
    )  # fmt: skip
    _assert_train_step(f"test_grpo_train_step_variants_on_policy[{variant}]", model, _train_step(model, algo))


@pytest.mark.parametrize("method", ["noclip", "grpo", "gspo"])
def test_grpo_train_step_off_policy(method):
    model = _tiny_gpt2()
    old = torch.linspace(-1.5, 0.5, steps=12).reshape(4, 3)
    loss = _train_step(model, GRPOAlgorithm(group_size=2, importance_reweighting=method, cliprange=0.1), old)
    _assert_train_step(f"test_grpo_train_step_off_policy[{method}]", model, loss)


# ---- the rollout source under the real Trainer: crash mid-rollout, resume bitwise ----

MODEL = PRESETS["cs336-tiny"]
TARGET = CharTokenizer().encode("e")[0]
ALGO = GRPOAlgorithm(
    group_size=4, n_prompts_per_rollout=4, epochs_per_rollout_batch=2, importance_reweighting="grpo", cliprange=0.2
)
CFG = TrainerConfig(
    num_steps=6, global_batch_size=16, micro_batch_size=4, log_every=1, checkpoint_every=3, async_checkpoint=False,
    optimizer=OptimizerConfig(lr=1e-2, weight_decay=0.0), schedule=ScheduleConfig(name="constant"),
)  # fmt: skip


class Crash(Exception):
    pass


def _fit(env: DistEnv, out: Path, crash_after: int | None = None) -> None:
    parallel = ParallelConfig(MeshConfig(shard=2), Strategy.FSDP, Backend.SCRATCH)
    mesh = build_mesh(parallel.mesh, env)
    tok = CharTokenizer()
    model = TransformerLM(MODEL)
    model.init_weights(0)
    rows = [{"question": f"tell me {i}", "ground_truth": ""} for i in range(8)]
    prompts = PromptSource.from_rows(rows, tok, PromptFormat.BOXED, 64, 4, mesh.dp_rank, mesh.dp_size, seed=0)
    pmodel = parallelize(model, mesh, parallel)
    source = RolloutBatches(
        prompts, TorchRolloutEngine(MODEL, tokenizer=tok), pmodel, tok, target_token_reward(TARGET),
        SamplingParams(max_tokens=6, stop_token_ids=(tok.eos_token_id,)), ALGO, CFG.global_batch_size, mesh,
        CFG.micro_batch_size, seed=0,
    )  # fmt: skip

    def on_step_end(step: int) -> None:
        if step == crash_after:
            raise Crash

    loss_fn = lambda pm, batch: grpo_loss(pm, batch, ALGO, CFG.global_batch_size)  # noqa: E731
    optimizer = build_parallel_optimizer(pmodel, parallel, CFG.optimizer)
    trainer = Trainer(CFG, pmodel, optimizer, source, loss_fn, mesh, out, on_step_end=on_step_end)
    try:
        trainer.fit()
    except Crash:
        pass


def _crash(env: DistEnv, root: Path) -> None:
    _fit(env, root / "uninterrupted")
    _fit(env, root / "interrupted", crash_after=4)


def _resume(env: DistEnv, root: Path) -> None:
    _fit(env, root / "interrupted")


def _records(out: Path, key: str) -> list[float]:
    return [r[key] for r in map(json.loads, (out / "metrics.jsonl").read_text().splitlines()) if key in r]


def test_resume_mid_rollout_replays_the_same_rollouts(tmp_path: Path):
    run_distributed(_crash, 2, tmp_path)
    run_distributed(_resume, 2, tmp_path)
    full, resumed = tmp_path / "uninterrupted", tmp_path / "interrupted"
    # Step 4 (the second step on rollout batch 1) resumes from the step-3 checkpoint, after rollout 1 was sampled.
    assert _records(resumed, "loss")[4:] == _records(full, "loss")[3:]
    assert _records(resumed, "reward")[4:] == _records(full, "reward")[3:]
    # The first step on each rollout batch is on-policy: importance ratios are exactly 1, nothing is clipped.
    assert [_records(full, "clip_fraction")[i] for i in (0, 2, 4)] == [0.0, 0.0, 0.0]
    a, b = load_file(full / "final" / "model.safetensors"), load_file(resumed / "final" / "model.safetensors")
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
