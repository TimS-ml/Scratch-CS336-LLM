"""Reusable step builders: the library owns *how* and *where* (run functions, entry points, output layout), the
experiment owns *what* (names, data, model, hyperparameters).

CPU work (download, tokenizer training, tokenization) runs in-process as ``run`` steps; torchrun work (training,
evaluation) is an ``entrypoint`` step launched with the step's ``Resources``. Every config is a frozen dataclass, so
a step's output path is a hash of what it computes.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import get_session, hf_hub_download, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from scratch_cs336.data.tokenize import DEFAULT_SHARD_BYTES, DOCUMENT_SEPARATOR, build_token_cache
from scratch_cs336.eval.perplexity import PerplexityConfig
from scratch_cs336.launch.resources import Resources
from scratch_cs336.parallel import ParallelConfig
from scratch_cs336.pipeline.step import InputPath, Step
from scratch_cs336.posttrain.data import PromptFormat, gsm8k_sft_rows, load_gsm8k
from scratch_cs336.posttrain.dpo import DPOConfig
from scratch_cs336.posttrain.grpo import EngineConfig, GRPOAlgorithm, GRPOConfig, RewardKind, SamplingConfig
from scratch_cs336.posttrain.policy import PolicyConfig
from scratch_cs336.posttrain.reward_eval import RewardEvalConfig
from scratch_cs336.posttrain.sft import SFTConfig
from scratch_cs336.tokenizer import BPETokenizer, train_bpe
from scratch_cs336.train.config import TrainerConfig
from scratch_cs336.train.pretrain import PretrainConfig
from scratch_cs336.train.trainer import FINAL_DIR

PathLike = str | InputPath

# --- download


@dataclass(frozen=True)
class HFDownloadConfig:
    repo_id: str
    filenames: tuple[str, ...]
    repo_type: str = "dataset"
    revision: str | None = None
    max_bytes: int | None = None  # per file: keep only the documents within the first max_bytes (smoke runs)


def truncate_at_boundary(data: bytes, filename: str) -> bytes:
    """``data`` up to and including its last document boundary: a newline for jsonl, ``<|endoftext|>`` otherwise."""
    sep = b"\n" if filename.endswith(".jsonl") else DOCUMENT_SEPARATOR.encode()
    end = data.rfind(sep)
    if end < 0:
        raise ValueError(f"no document boundary {sep!r} within the first {len(data)} bytes of {filename}")
    return data[: end + len(sep)]


def _download_prefix(cfg: HFDownloadConfig, filename: str, out: Path) -> None:
    if filename.endswith(".gz"):
        raise ValueError(f"max_bytes cannot truncate compressed file {filename}")
    url = hf_hub_url(cfg.repo_id, filename, repo_type=cfg.repo_type, revision=cfg.revision)
    data, complete = bytearray(), True
    with get_session().stream("GET", url, headers=build_hf_headers(), follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            data += chunk
            if len(data) >= cfg.max_bytes:
                complete = False
                break
    payload = bytes(data) if complete else truncate_at_boundary(bytes(data[: cfg.max_bytes]), filename)
    target = out / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, target)


def _download(cfg: HFDownloadConfig, out: Path) -> None:
    for filename in cfg.filenames:
        if cfg.max_bytes is None:
            hf_hub_download(cfg.repo_id, filename, repo_type=cfg.repo_type, revision=cfg.revision, local_dir=out)
        else:
            _download_prefix(cfg, filename, out)


def hf_download(
    name: str,
    repo_id: str,
    filenames: Sequence[str],
    repo_type: str = "dataset",
    max_bytes: int | None = None,
    revision: str | None = None,
) -> Step[HFDownloadConfig]:
    """Files of a Hub repo at ``<out>/<filename>``; reference one with ``InputPath(step, filename)``."""
    return Step(name, HFDownloadConfig(repo_id, tuple(filenames), repo_type, revision, max_bytes), run=_download)


# --- tokenizer


@dataclass(frozen=True)
class TrainTokenizerConfig:
    corpus: PathLike
    vocab_size: int
    special_tokens: tuple[str, ...] = (DOCUMENT_SEPARATOR,)


def _train_tokenizer(cfg: TrainTokenizerConfig, out: Path) -> None:
    vocab, merges = train_bpe(cfg.corpus, cfg.vocab_size, list(cfg.special_tokens))
    BPETokenizer(vocab, merges, list(cfg.special_tokens)).save(out)


def train_tokenizer(
    name: str, corpus: InputPath, vocab_size: int, special_tokens: Sequence[str] = (DOCUMENT_SEPARATOR,)
) -> Step[TrainTokenizerConfig]:
    """Byte-level BPE trained on ``corpus``; ``InputPath(step)`` is a tokenizer spec (a BPE directory)."""
    return Step(name, TrainTokenizerConfig(corpus, vocab_size, tuple(special_tokens)), run=_train_tokenizer)


# --- token cache


@dataclass(frozen=True)
class TokenizeConfig:
    files: tuple[PathLike, ...]
    tokenizer: PathLike  # load_tokenizer spec or BPE directory


def _tokenize(cfg: TokenizeConfig, out: Path) -> None:
    files = [Path(f) for f in cfg.files]
    workers = os.cpu_count() or 1
    # Enough shards to keep every worker busy; the token stream does not depend on the split.
    shard_bytes = min(DEFAULT_SHARD_BYTES, max(1 << 20, math.ceil(sum(f.stat().st_size for f in files) / workers)))
    build_token_cache(files, str(cfg.tokenizer), out, num_workers=workers, shard_bytes=shard_bytes)


def tokenize(name: str, files: PathLike | Sequence[PathLike], tokenizer: PathLike) -> Step[TokenizeConfig]:
    """Token cache of ``files``; ``InputPath(step)`` is the cache directory."""
    files = (files,) if isinstance(files, str | InputPath) else tuple(files)
    return Step(name, TokenizeConfig(files, tokenizer), run=_tokenize)


# --- training and evaluation


def pretrain(
    name: str,
    *,
    model: str,
    train: PathLike,
    val: PathLike | None,
    seq_len: int,
    parallel: ParallelConfig,
    trainer: TrainerConfig,
    resources: Resources,
    tokenizer: PathLike | None = None,
    vocab_size: int | None = None,
    eval_batches: int = 8,
) -> Step[PretrainConfig]:
    """``scratch_cs336.train.pretrain`` on token caches; :func:`final_model` points at its export."""
    cfg = PretrainConfig(
        model=model,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        train_cache=train,
        val_cache=val,
        seq_len=seq_len,
        parallel=parallel,
        trainer=trainer,
        eval_batches=eval_batches,
    )
    return Step(name, cfg, entrypoint="scratch_cs336.train.pretrain", resources=resources)


def final_model(train_step: Step) -> InputPath:
    """The full export (``model.safetensors`` + ``config.json``) a training step leaves at the end."""
    return InputPath(train_step, FINAL_DIR)


def evaluate(
    name: str,
    model_dir: InputPath,
    cache: PathLike,
    *,
    seq_len: int,
    batch_size: int,
    parallel: ParallelConfig,
    resources: Resources,
    max_batches: int | None = None,
) -> Step[PerplexityConfig]:
    """Loss / perplexity of an exported model on a token cache; writes ``<out>/eval.json``."""
    cfg = PerplexityConfig(
        model_dir=model_dir,
        cache=cache,
        seq_len=seq_len,
        batch_size=batch_size,
        max_batches=max_batches,
        parallel=parallel,
    )
    return Step(name, cfg, entrypoint="scratch_cs336.eval.perplexity", resources=resources)


# --- post-training

GSM8K_TRAIN = "train.jsonl"  # {"question", "answer"} rows as in openai/gsm8k
GSM8K_TEST = "test.jsonl"
GSM8K_SFT = "sft_train.jsonl"  # {"prompt", "response"} train solutions in the SFT prompt format's answer convention


@dataclass(frozen=True)
class GSM8KConfig:
    sft_format: PromptFormat = PromptFormat.R1_ZERO
    max_train: int | None = None  # first N rows per split (smoke runs); None = all
    max_test: int | None = None


def _write_jsonl(rows: Sequence[dict], path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def _gsm8k(cfg: GSM8KConfig, out: Path) -> None:
    train = load_gsm8k("train")[: cfg.max_train]
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train, out / GSM8K_TRAIN)
    _write_jsonl(load_gsm8k("test")[: cfg.max_test], out / GSM8K_TEST)
    _write_jsonl(gsm8k_sft_rows(train, cfg.sft_format), out / GSM8K_SFT)


def gsm8k(
    name: str,
    sft_format: PromptFormat = PromptFormat.R1_ZERO,
    max_train: int | None = None,
    max_test: int | None = None,
) -> Step[GSM8KConfig]:
    """``openai/gsm8k`` (needs the ``datasets`` extra) as local jsonl: :data:`GSM8K_TRAIN`, :data:`GSM8K_TEST`,
    :data:`GSM8K_SFT`; reference one with ``InputPath(step, GSM8K_TRAIN)``."""
    return Step(name, GSM8KConfig(sft_format, max_train, max_test), run=_gsm8k)


def sft(
    name: str,
    *,
    policy: PolicyConfig,
    train: PathLike,
    seq_len: int,
    parallel: ParallelConfig,
    trainer: TrainerConfig,
    resources: Resources,
    prompt_format: PromptFormat = PromptFormat.CHAT,
) -> Step[SFTConfig]:
    """``scratch_cs336.posttrain.sft``; ``policy.init_from`` may be a :func:`final_model` reference."""
    cfg = SFTConfig(
        policy=policy, train_data=train, prompt_format=prompt_format, seq_len=seq_len, parallel=parallel,
        trainer=trainer,
    )  # fmt: skip
    return Step(name, cfg, entrypoint="scratch_cs336.posttrain.sft", resources=resources)


def dpo(
    name: str,
    *,
    policy: PolicyConfig,
    train: PathLike,
    beta: float,
    seq_len: int,
    parallel: ParallelConfig,
    trainer: TrainerConfig,
    resources: Resources,
) -> Step[DPOConfig]:
    """``scratch_cs336.posttrain.dpo`` on ``{"prompt", "chosen", "rejected"}`` rows; the reference is the initial
    policy."""
    cfg = DPOConfig(policy=policy, train_data=train, beta=beta, seq_len=seq_len, parallel=parallel, trainer=trainer)
    return Step(name, cfg, entrypoint="scratch_cs336.posttrain.dpo", resources=resources)


def grpo(
    name: str,
    *,
    policy: PolicyConfig,
    train: PathLike,
    algo: GRPOAlgorithm,
    sampling: SamplingConfig,
    engine: EngineConfig,
    parallel: ParallelConfig,
    trainer: TrainerConfig,
    resources: Resources,
    prompt_format: PromptFormat = PromptFormat.R1_ZERO,
    reward: RewardKind = RewardKind.R1_ZERO,
    max_prompt_len: int = 512,
    eval_data: PathLike | None = None,
    eval_prompts: int = 256,
) -> Step[GRPOConfig]:
    """``scratch_cs336.posttrain.grpo``; ``trainer.global_batch_size`` counts rollouts per optimizer step."""
    cfg = GRPOConfig(
        policy=policy, train_data=train, eval_data=eval_data, eval_prompts=eval_prompts, prompt_format=prompt_format,
        reward=reward, max_prompt_len=max_prompt_len, sampling=sampling, algo=algo, engine=engine,
        parallel=parallel, trainer=trainer,
    )  # fmt: skip
    return Step(name, cfg, entrypoint="scratch_cs336.posttrain.grpo", resources=resources)


def reward_eval(
    name: str,
    *,
    policy: PolicyConfig,
    data: PathLike,
    sampling: SamplingConfig,
    engine: EngineConfig,
    parallel: ParallelConfig,
    resources: Resources,
    prompt_format: PromptFormat = PromptFormat.R1_ZERO,
    reward: RewardKind = RewardKind.R1_ZERO,
    max_examples: int | None = None,
    max_prompt_len: int = 512,
) -> Step[RewardEvalConfig]:
    """Greedy reward / accuracy of ``policy`` on held-out prompts; writes ``<out>/eval.json``."""
    cfg = RewardEvalConfig(
        policy=policy, data=data, max_examples=max_examples, prompt_format=prompt_format, reward=reward,
        max_prompt_len=max_prompt_len, sampling=sampling, engine=engine, parallel=parallel,
    )  # fmt: skip
    return Step(name, cfg, entrypoint="scratch_cs336.posttrain.reward_eval", resources=resources)
