"""
Tokenizer Module / 分词器模块

This module provides comprehensive tokenizer functionality including:
- Training custom tokenizers (especially for Chinese)
- Merging vocabularies from different tokenizers
- Expanding model embeddings for new vocabularies

该模块提供全面的分词器功能，包括：
- 训练自定义分词器（特别是中文）
- 合并不同分词器的词汇表
- 为新词汇表扩展模型嵌入

Author: LLM-from-Scratch Team
License: MIT
"""

# Import from merge_vocab module / 从merge_vocab模块导入
from .merge_vocab import (
    merge_tokenizers,
    validate_merged_tokenizer,
    export_vocabulary,
)

# Import from expand_embedding module / 从expand_embedding模块导入
from .expand_embedding import (
    expand_embeddings,
    expand_model_for_tokenizer,
    verify_embedding_expansion,
)

# Import from train_chinese module / 从train_chinese模块导入
from .train_chinese import (
    train_chinese_tokenizer,
    test_tokenizer,
    analyze_tokenizer,
    convert_to_huggingface_tokenizer,
)

__all__ = [
    # Vocabulary merging / 词汇表合并
    "merge_tokenizers",
    "validate_merged_tokenizer",
    "export_vocabulary",

    # Embedding expansion / 嵌入扩展
    "expand_embeddings",
    "expand_model_for_tokenizer",
    "verify_embedding_expansion",

    # Tokenizer training / 分词器训练
    "train_chinese_tokenizer",
    "test_tokenizer",
    "analyze_tokenizer",
    "convert_to_huggingface_tokenizer",
]

__version__ = "1.0.0"
