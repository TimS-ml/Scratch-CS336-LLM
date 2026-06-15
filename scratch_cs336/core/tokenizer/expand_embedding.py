"""
Embedding Layer Expansion Module / 嵌入层扩展模块

This module provides functionality to expand model embedding layers when the
vocabulary size increases. It preserves weights for existing tokens and properly
initializes new token embeddings.

该模块提供了在词汇表大小增加时扩展模型嵌入层的功能。它保留现有tokens的权重，
并正确初始化新token的嵌入。

Author: LLM-from-Scratch Team
License: MIT
"""

from typing import Union, Optional, Tuple, Dict, Any
from pathlib import Path
import torch
import torch.nn as nn

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer
    )
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Please run: "
        "pip install transformers torch"
    ) from e


def expand_embeddings(
    model: PreTrainedModel,
    new_vocab_size: int,
    pad_to_multiple_of: Optional[int] = None,
    initialization_strategy: str = "normal",
    mean: float = 0.0,
    std: float = 0.02,
    verbose: bool = True
) -> PreTrainedModel:
    """
    Expand model embedding and output layers to accommodate new vocabulary size.
    扩展模型嵌入层和输出层以适应新的词汇表大小。

    This function resizes both the input embedding layer and the output (lm_head)
    layer of a language model. Weights for existing tokens are preserved, and
    new token embeddings are initialized using the specified strategy.

    该函数调整语言模型的输入嵌入层和输出层（lm_head）的大小。保留现有tokens的权重，
    并使用指定策略初始化新token的嵌入。

    Args:
        model: Pre-trained language model / 预训练语言模型
        new_vocab_size: Target vocabulary size / 目标词汇表大小
        pad_to_multiple_of: Pad vocab size to multiple of this value (for efficiency)
                          将词汇表大小填充到此值的倍数（为了效率）
        initialization_strategy: Strategy for initializing new embeddings
                               ("normal", "uniform", "xavier", "kaiming")
                               新嵌入的初始化策略
        mean: Mean for normal initialization / 正态分布初始化的均值
        std: Standard deviation for normal initialization / 正态分布初始化的标准差
        verbose: Print progress information / 打印进度信息

    Returns:
        PreTrainedModel: Model with expanded embeddings / 扩展嵌入后的模型

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("model_path")
        >>> model = expand_embeddings(model, new_vocab_size=50000)
        >>> print(f"New embedding size: {model.get_input_embeddings().weight.shape}")
    """
    current_vocab_size = model.config.vocab_size

    if verbose:
        print(f"Current vocabulary size: {current_vocab_size}")
        print(f"Target vocabulary size: {new_vocab_size}")

    # 验证新词汇表大小 / Validate new vocabulary size
    if new_vocab_size <= 0:
        raise ValueError(f"Invalid vocabulary size: {new_vocab_size}")

    if new_vocab_size < current_vocab_size:
        raise ValueError(
            f"New vocabulary size ({new_vocab_size}) cannot be smaller than "
            f"current size ({current_vocab_size})"
        )

    if new_vocab_size == current_vocab_size:
        if verbose:
            print("Vocabulary size unchanged, no expansion needed")
        return model

    # 可选：填充到指定倍数 / Optional: Pad to multiple
    if pad_to_multiple_of is not None:
        remainder = new_vocab_size % pad_to_multiple_of
        if remainder != 0:
            new_vocab_size += pad_to_multiple_of - remainder
            if verbose:
                print(f"Padded vocabulary size to: {new_vocab_size}")

    # 获取当前嵌入层 / Get current embeddings
    old_embeddings = model.get_input_embeddings()
    old_num_tokens, embedding_dim = old_embeddings.weight.shape

    if verbose:
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Tokens to add: {new_vocab_size - current_vocab_size}")

    # 调整token嵌入大小 / Resize token embeddings
    model.resize_token_embeddings(new_vocab_size)

    # 更新配置 / Update config
    model.config.vocab_size = new_vocab_size

    # 获取新的嵌入层 / Get new embeddings
    new_embeddings = model.get_input_embeddings()

    # 初始化新添加的token嵌入 / Initialize new token embeddings
    if new_vocab_size > old_num_tokens:
        with torch.no_grad():
            if initialization_strategy == "normal":
                # 正态分布初始化 / Normal distribution initialization
                nn.init.normal_(
                    new_embeddings.weight[old_num_tokens:],
                    mean=mean,
                    std=std
                )
            elif initialization_strategy == "uniform":
                # 均匀分布初始化 / Uniform distribution initialization
                nn.init.uniform_(
                    new_embeddings.weight[old_num_tokens:],
                    a=-std,
                    b=std
                )
            elif initialization_strategy == "xavier":
                # Xavier初始化 / Xavier initialization
                nn.init.xavier_normal_(new_embeddings.weight[old_num_tokens:])
            elif initialization_strategy == "kaiming":
                # Kaiming初始化 / Kaiming initialization
                nn.init.kaiming_normal_(new_embeddings.weight[old_num_tokens:])
            else:
                raise ValueError(
                    f"Unknown initialization strategy: {initialization_strategy}"
                )

        if verbose:
            print(f"Initialized {new_vocab_size - old_num_tokens} new token embeddings "
                  f"using '{initialization_strategy}' strategy")

    if verbose:
        print("Embedding expansion completed successfully")

    return model


def expand_model_for_tokenizer(
    model_path: Union[str, Path],
    tokenizer_path: Union[str, Path],
    output_dir: Union[str, Path],
    pad_to_multiple_of: Optional[int] = 64,
    initialization_strategy: str = "normal",
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    verbose: bool = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer, expand embeddings, and save.
    加载模型和分词器，扩展嵌入，并保存。

    This is a convenience function that handles the complete workflow:
    1. Load model and tokenizer
    2. Expand model embeddings to match tokenizer vocabulary
    3. Save both model and tokenizer

    这是一个便捷函数，处理完整的工作流程：
    1. 加载模型和分词器
    2. 扩展模型嵌入以匹配分词器词汇表
    3. 保存模型和分词器

    Args:
        model_path: Path to original model / 原始模型路径
        tokenizer_path: Path to new tokenizer (with expanded vocabulary)
                       新分词器路径（具有扩展的词汇表）
        output_dir: Output directory for expanded model / 扩展模型的输出目录
        pad_to_multiple_of: Pad vocab size to multiple of this value
                          将词汇表大小填充到此值的倍数
        initialization_strategy: Initialization strategy for new embeddings
                               新嵌入的初始化策略
        device_map: Device mapping for model loading / 模型加载的设备映射
        torch_dtype: Data type for model weights / 模型权重的数据类型
        verbose: Print progress information / 打印进度信息

    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: Expanded model and tokenizer
                                                     扩展后的模型和分词器

    Example:
        >>> model, tokenizer = expand_model_for_tokenizer(
        ...     model_path="original_model",
        ...     tokenizer_path="merged_tokenizer",
        ...     output_dir="expanded_model"
        ... )
    """
    model_path = Path(model_path)
    tokenizer_path = Path(tokenizer_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    if verbose:
        print(f"Loading model from: {model_path}")

    # 加载模型 / Load model
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    if verbose:
        print(f"Loading tokenizer from: {tokenizer_path}")

    # 加载分词器 / Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True
    )

    new_vocab_size = len(tokenizer)

    if verbose:
        print(f"\nModel info:")
        print(f"  - Original vocab size: {model.config.vocab_size}")
        print(f"  - Tokenizer vocab size: {new_vocab_size}")

    # 扩展嵌入 / Expand embeddings
    model = expand_embeddings(
        model=model,
        new_vocab_size=new_vocab_size,
        pad_to_multiple_of=pad_to_multiple_of,
        initialization_strategy=initialization_strategy,
        verbose=verbose
    )

    # 保存模型和分词器 / Save model and tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving expanded model to: {output_dir}")

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    if verbose:
        print("Model and tokenizer saved successfully")

    return model, tokenizer


def verify_embedding_expansion(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_token_ids: Optional[list] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Verify that embedding expansion was successful.
    验证嵌入扩展是否成功。

    Args:
        model: Expanded model / 扩展后的模型
        tokenizer: Corresponding tokenizer / 对应的分词器
        test_token_ids: Optional list of token IDs to test
                       可选的要测试的token ID列表
        verbose: Print verification results / 打印验证结果

    Returns:
        Dict[str, Any]: Verification results / 验证结果

    Example:
        >>> results = verify_embedding_expansion(model, tokenizer)
        >>> print(results["status"])
    """
    results = {
        "status": "success",
        "model_vocab_size": model.config.vocab_size,
        "tokenizer_vocab_size": len(tokenizer),
        "embedding_shape": None,
        "lm_head_shape": None,
        "issues": []
    }

    # 检查词汇表大小匹配 / Check vocabulary size match
    if model.config.vocab_size != len(tokenizer):
        results["status"] = "warning"
        results["issues"].append(
            f"Vocabulary size mismatch: model={model.config.vocab_size}, "
            f"tokenizer={len(tokenizer)}"
        )

    # 检查嵌入层形状 / Check embedding layer shape
    embeddings = model.get_input_embeddings()
    results["embedding_shape"] = tuple(embeddings.weight.shape)

    if embeddings.weight.shape[0] != model.config.vocab_size:
        results["status"] = "error"
        results["issues"].append(
            f"Embedding layer size ({embeddings.weight.shape[0]}) does not match "
            f"config vocab size ({model.config.vocab_size})"
        )

    # 检查输出层形状 / Check output layer shape
    if hasattr(model, "lm_head"):
        lm_head = model.lm_head
        results["lm_head_shape"] = tuple(lm_head.weight.shape)

        if lm_head.weight.shape[0] != model.config.vocab_size:
            results["status"] = "error"
            results["issues"].append(
                f"LM head size ({lm_head.weight.shape[0]}) does not match "
                f"config vocab size ({model.config.vocab_size})"
            )

    # 测试特定token ID的嵌入 / Test embeddings for specific token IDs
    if test_token_ids:
        with torch.no_grad():
            for token_id in test_token_ids:
                if token_id >= embeddings.weight.shape[0]:
                    results["status"] = "error"
                    results["issues"].append(
                        f"Token ID {token_id} out of range (max: {embeddings.weight.shape[0] - 1})"
                    )
                else:
                    embedding_vector = embeddings.weight[token_id]
                    if torch.isnan(embedding_vector).any():
                        results["status"] = "error"
                        results["issues"].append(f"NaN values in embedding for token {token_id}")

    if verbose:
        print("\n=== Embedding Expansion Verification ===")
        print(f"Status: {results['status'].upper()}")
        print(f"Model vocab size: {results['model_vocab_size']}")
        print(f"Tokenizer vocab size: {results['tokenizer_vocab_size']}")
        print(f"Embedding shape: {results['embedding_shape']}")
        if results['lm_head_shape']:
            print(f"LM head shape: {results['lm_head_shape']}")

        if results["issues"]:
            print("\nIssues found:")
            for issue in results["issues"]:
                print(f"  - {issue}")
        else:
            print("\n✓ All checks passed")

    return results


if __name__ == "__main__":
    # Example usage / 使用示例
    import argparse

    parser = argparse.ArgumentParser(
        description="Expand model embeddings for new tokenizer / 为新分词器扩展模型嵌入"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to original model / 原始模型路径"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to new tokenizer / 新分词器路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory / 输出目录"
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=64,
        help="Pad vocabulary size to multiple of this value / 填充词汇表大小到此值的倍数"
    )
    parser.add_argument(
        "--init-strategy",
        type=str,
        default="normal",
        choices=["normal", "uniform", "xavier", "kaiming"],
        help="Initialization strategy / 初始化策略"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify expansion after completion / 完成后验证扩展"
    )

    args = parser.parse_args()

    # 扩展模型 / Expand model
    model, tokenizer = expand_model_for_tokenizer(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        pad_to_multiple_of=args.pad_to_multiple_of,
        initialization_strategy=args.init_strategy,
        verbose=True
    )

    # 可选验证 / Optional verification
    if args.verify:
        verify_embedding_expansion(model, tokenizer, verbose=True)
