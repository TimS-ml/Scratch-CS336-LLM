"""Pretraining entry point: config.yaml round trip and model/tokenizer validation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import draccus
import pytest
from transformers import Qwen3_5TextConfig, Qwen3Config, Qwen3ForCausalLM

import scratch_cs336.pipeline.runner  # noqa: F401 - registers the yaml encoding the runner uses
from scratch_cs336.distributed import DistEnv, MeshConfig
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy
from scratch_cs336.posttrain.policy import PolicyConfig, load_checked_tokenizer, load_policy
from scratch_cs336.posttrain.sft import SFTConfig
from scratch_cs336.posttrain.sft import run as sft_run
from scratch_cs336.tokenizer import BPETokenizer
from scratch_cs336.train import TrainerConfig
from scratch_cs336.train.optim import OptimizerConfig, OptimizerName, ScheduleConfig, ScheduleName
from scratch_cs336.train.pretrain import MixtureComponent, PretrainConfig, build_model
from scratch_cs336.train.pretrain import run as pretrain_run


def test_dumped_config_round_trips_with_cli_overrides(tmp_path: Path) -> None:
    cfg = PretrainConfig(
        output_dir=str(tmp_path / "out"),
        vocab_size=2000,
        mixture=(MixtureComponent("/a", 0.7), MixtureComponent("/b", 0.3)),
        val_cache=None,
        seq_len=64,
        parallel=ParallelConfig(MeshConfig(replicate=2, shard=-1), Strategy.ZERO1, Backend.NATIVE, "bfloat16"),
        trainer=TrainerConfig(
            num_steps=7,
            global_batch_size=16,
            micro_batch_size=2,
            optimizer=OptimizerConfig(OptimizerName.TORCH_ADAMW, betas=(0.8, 0.99)),
            schedule=ScheduleConfig(ScheduleName.WSD, warmup_steps=1, total_steps=7, decay_steps=2),
            max_grad_norm=None,
            eval_every=3,
        ),
    )
    path = tmp_path / "config.yaml"
    with path.open("w") as f:
        draccus.dump(cfg, f)
    parsed = draccus.parse(
        PretrainConfig, args=["--config_path", str(path), "--trainer.num_steps", "9", "--parallel.strategy", "fsdp"]
    )
    expected = dataclasses.replace(
        cfg,
        trainer=dataclasses.replace(cfg.trainer, num_steps=9),
        parallel=dataclasses.replace(cfg.parallel, strategy=Strategy.FSDP),
    )
    assert parsed == expected


def test_hf_init_requires_a_tokenizer_with_the_checkpoints_vocabulary(tmp_path: Path) -> None:
    hf_dir = tmp_path / "hf"
    hf_config = Qwen3Config(
        vocab_size=512, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, max_position_embeddings=256,
    )  # fmt: skip
    Qwen3ForCausalLM(hf_config).save_pretrained(hf_dir)
    BPETokenizer({i: bytes([i]) for i in range(256)}, [], ["<|endoftext|>"]).save(tmp_path / "bpe")
    spec = f"bpe:{tmp_path / 'bpe'}"  # 257 tokens: fits a 512-row embedding, but is not the checkpoint's tokenizer

    assert build_model(PretrainConfig(model="cs336-tiny", vocab_size=512, tokenizer=spec)).cfg.vocab_size == 512
    with pytest.raises(ValueError, match="HF checkpoint"):
        build_model(PretrainConfig(init_from_hf=str(hf_dir), tokenizer=spec))
    policy = PolicyConfig(init_from=str(hf_dir), tokenizer=spec)
    with pytest.raises(ValueError, match="HF checkpoint"):
        load_checked_tokenizer(policy, load_policy(policy, seed=0))


def _run_entries_with_tensor_4(env: DistEnv, root: Path) -> list[str]:
    parallel = ParallelConfig(MeshConfig(replicate=1, shard=1, tensor=4), Strategy.FSDP, Backend.SCRATCH)
    out = str(root / "out")
    messages = []
    with pytest.raises(ValueError) as e:
        pretrain_run(PretrainConfig(out, init_from_hf=str(root), train_cache="unused", parallel=parallel), env)
    messages.append(str(e.value))
    with pytest.raises(ValueError) as e:
        sft_run(SFTConfig(out, PolicyConfig(init_from=str(root)), train_data="unused", parallel=parallel), env)
    messages.append(str(e.value))
    return messages


def test_tensor_parallel_degree_the_model_cannot_shard_fails_before_loading_weights(tmp_path: Path) -> None:
    # Qwen3.5-0.8B head counts; only config.json exists, so reaching the weights would fail differently.
    Qwen3_5TextConfig(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=8,
        num_key_value_heads=2, head_dim=16, linear_num_key_heads=16, linear_num_value_heads=16,
    ).to_json_file(tmp_path / "config.json")  # fmt: skip
    for message in run_distributed(_run_entries_with_tensor_4, 4, tmp_path)[0]:
        assert "tensor=4 must divide n_kv_heads=2" in message and "{1, 2}" in message
