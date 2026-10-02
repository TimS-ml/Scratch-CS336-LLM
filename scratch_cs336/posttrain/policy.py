"""Policy initialization shared by the post-training entries: HF checkpoint, exported ``final/`` dir, or a preset."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from scratch_cs336.checkpoint.export import MODEL_FILE
from scratch_cs336.eval.perplexity import exported_config, load_exported
from scratch_cs336.models import PRESETS, ModelConfig, TransformerLM
from scratch_cs336.models.hf import load_hf_config, load_hf_pretrained
from scratch_cs336.tokenizer import Tokenizer, check_tokenizer_vocab, load_tokenizer
from scratch_cs336.train.trainer import MODEL_CONFIG_FILE


@dataclass(frozen=True)
class PolicyConfig:
    # HF repo / snapshot dir, or a trainer-exported ``final/`` dir (e.g. SFT output for DPO / GRPO).
    # None: the ``model`` preset with deterministic random init (smoke tests).
    init_from: str | None = None
    model: str = "qwen3.5-tiny"
    vocab_size: int | None = None  # overrides the preset's vocabulary
    tokenizer: str = "hf:Qwen/Qwen3.5-0.8B"  # load_tokenizer spec


def is_exported_dir(path: Path) -> bool:
    """A trainer export has our ``ModelConfig`` json (no HF ``model_type``) next to ``model.safetensors``."""
    config = path / MODEL_CONFIG_FILE
    return (path / MODEL_FILE).exists() and config.exists() and "model_type" not in json.loads(config.read_text())


def policy_config(cfg: PolicyConfig) -> ModelConfig:
    """The policy's :class:`ModelConfig` without loading (or downloading) weights; validate it before ``load_policy``."""
    if cfg.init_from is None:
        if cfg.model not in PRESETS:
            raise ValueError(f"unknown model preset {cfg.model!r}; choose from {sorted(PRESETS)}")
        model_cfg = PRESETS[cfg.model]
        if cfg.vocab_size is not None:
            model_cfg = dataclasses.replace(model_cfg, vocab_size=cfg.vocab_size)
        return model_cfg
    if is_exported_dir(Path(cfg.init_from)):
        return exported_config(cfg.init_from)
    return load_hf_config(cfg.init_from)


def load_policy(cfg: PolicyConfig, seed: int) -> TransformerLM:
    """Full fp32 model on CPU."""
    if cfg.init_from is None:
        model = TransformerLM(policy_config(cfg))
        model.init_weights(seed)
        return model
    if is_exported_dir(Path(cfg.init_from)):
        return load_exported(cfg.init_from).float()
    model_cfg, state = load_hf_pretrained(cfg.init_from, dtype=torch.float32)
    model = TransformerLM(model_cfg)
    model.load_state_dict(state)
    return model


def load_checked_tokenizer(cfg: PolicyConfig, model: TransformerLM) -> Tokenizer:
    """``cfg.tokenizer``, validated against the policy ``load_policy(cfg)`` built (see ``check_tokenizer_vocab``)."""
    tokenizer = load_tokenizer(cfg.tokenizer)
    hf_checkpoint = None if cfg.init_from is None or is_exported_dir(Path(cfg.init_from)) else cfg.init_from
    check_tokenizer_vocab(cfg.tokenizer, tokenizer, model.cfg.vocab_size, hf_checkpoint)
    return tokenizer
