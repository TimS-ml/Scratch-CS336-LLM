"""Transformer LM family: CS336 basics (Llama-style), Qwen3, Qwen3.5 text (hybrid Gated DeltaNet)."""

from scratch_cs336.models.config import LayerType, ModelConfig
from scratch_cs336.models.presets import PRESETS
from scratch_cs336.models.transformer import TransformerLM

__all__ = ["PRESETS", "LayerType", "ModelConfig", "TransformerLM"]
