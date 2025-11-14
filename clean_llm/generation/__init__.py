"""
Generation utilities for LLM inference and text generation.
LLM推理和文本生成的实用工具。

This package provides utilities for:
- Multi-turn conversation context management / 多轮对话上下文管理
- Math reasoning with Program-of-Thought (PoT) / 基于程序思维的数学推理
- Custom logits processors for generation control / 用于生成控制的自定义logits处理器
- Streaming text generation / 流式文本生成
"""

from .utils import make_context, parse_pot_no_stream
from .processors import OutputRepetitionPenaltyLogitsProcessor
from .streaming import TextIterStreamer

__all__ = [
    "make_context",
    "parse_pot_no_stream",
    "OutputRepetitionPenaltyLogitsProcessor",
    "TextIterStreamer",
]

__version__ = "1.0.0"
