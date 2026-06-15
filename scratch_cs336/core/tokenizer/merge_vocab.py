"""
Vocabulary Merging Module / 词汇表合并模块

This module provides functionality to merge different tokenizer vocabularies,
particularly for combining base models (e.g., LLaMA) with additional language-specific tokens.

该模块提供了合并不同分词器词汇表的功能，特别适用于将基础模型（如LLaMA）与额外的特定语言token组合。

Author: LLM-from-Scratch Team
License: MIT
"""

import os
from typing import Optional, List, Dict, Union
from pathlib import Path
import json

try:
    from transformers import LlamaTokenizer, AutoTokenizer, PreTrainedTokenizer
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
    import sentencepiece as spm
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Please run: "
        "pip install transformers sentencepiece"
    ) from e


def merge_tokenizers(
    base_tokenizer_path: Union[str, Path],
    additional_sp_model_path: Union[str, Path],
    output_dir: Union[str, Path],
    special_tokens: Optional[List[str]] = None,
    tokenizer_type: str = "llama",
    keep_duplicate_scores: bool = False,
    verbose: bool = True
) -> PreTrainedTokenizer:
    """
    Merge a base tokenizer with additional SentencePiece tokens.
    合并基础分词器与额外的SentencePiece tokens。

    This function takes a base tokenizer (e.g., LLaMA) and merges it with tokens
    from an additional SentencePiece model (e.g., Chinese tokens). The merged
    tokenizer is saved in Hugging Face format.

    该函数接受一个基础分词器（如LLaMA）并将其与额外SentencePiece模型中的tokens合并
    （如中文tokens）。合并后的分词器以Hugging Face格式保存。

    Args:
        base_tokenizer_path: Path to base tokenizer directory
                           基础分词器目录路径
        additional_sp_model_path: Path to additional SentencePiece model file (.model)
                                 额外的SentencePiece模型文件路径（.model）
        output_dir: Output directory for merged tokenizer
                   合并后分词器的输出目录
        special_tokens: List of special tokens to add (e.g., ["<|system|>", "<|user|>"])
                       要添加的特殊token列表（如["<|system|>", "<|user|>"]）
        tokenizer_type: Type of base tokenizer ("llama" or "auto")
                       基础分词器类型（"llama"或"auto"）
        keep_duplicate_scores: If True, keep scores from additional model for duplicates
                              如果为True，保留额外模型中重复token的分数
        verbose: Print progress information / 打印进度信息

    Returns:
        PreTrainedTokenizer: Merged tokenizer / 合并后的分词器

    Example:
        >>> tokenizer = merge_tokenizers(
        ...     base_tokenizer_path="llama2_tokenizer",
        ...     additional_sp_model_path="chinese_spm.model",
        ...     output_dir="merged_tokenizer",
        ...     special_tokens=["<|system|>", "<|user|>", "<|assistant|>"]
        ... )
        >>> print(f"Merged tokenizer has {len(tokenizer)} tokens")
    """
    # 设置环境变量以使用纯Python实现的protobuf / Set env var for pure Python protobuf
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

    # 验证输入路径 / Validate input paths
    base_path = Path(base_tokenizer_path)
    additional_path = Path(additional_sp_model_path)
    output_path = Path(output_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base tokenizer path not found: {base_path}")
    if not additional_path.exists():
        raise FileNotFoundError(f"Additional model path not found: {additional_path}")

    # 加载基础分词器 / Load base tokenizer
    if verbose:
        print(f"Loading base tokenizer from: {base_path}")

    if tokenizer_type.lower() == "llama":
        base_tokenizer = LlamaTokenizer.from_pretrained(str(base_path))
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(str(base_path))

    # 加载额外的SentencePiece模型 / Load additional SentencePiece model
    if verbose:
        print(f"Loading additional SentencePiece model from: {additional_path}")

    additional_sp = spm.SentencePieceProcessor()
    additional_sp.Load(str(additional_path))

    # 将分词器加载为protobuf对象 / Load tokenizers as protobuf objects
    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(base_tokenizer.sp_model.serialized_model_proto())

    additional_spm = sp_pb2_model.ModelProto()
    additional_spm.ParseFromString(additional_sp.serialized_model_proto())

    # 打印基本信息 / Print basic information
    base_token_count = len(base_tokenizer)
    additional_token_count = len(additional_sp)

    if verbose:
        print(f"Base tokenizer tokens: {base_token_count}")
        print(f"Additional SentencePiece tokens: {additional_token_count}")

    # 合并词汇表 / Merge vocabularies
    base_tokens_set = set(p.piece for p in base_spm.pieces)
    added_count = 0

    if verbose:
        print("Merging vocabularies...")

    for piece_obj in additional_spm.pieces:
        token_text = piece_obj.piece

        if token_text not in base_tokens_set:
            # 创建新的SentencePiece对象 / Create new SentencePiece object
            new_piece = sp_pb2_model.ModelProto().SentencePiece()
            new_piece.piece = token_text
            new_piece.score = piece_obj.score if keep_duplicate_scores else 0
            base_spm.pieces.append(new_piece)
            added_count += 1

    if verbose:
        print(f"Added {added_count} new tokens to base vocabulary")

    # 保存合并后的模型 / Save merged model
    temp_sp_dir = output_path / "temp_sp_model"
    temp_sp_dir.mkdir(parents=True, exist_ok=True)

    temp_model_file = temp_sp_dir / "tokenizer.model"

    if verbose:
        print(f"Saving merged SentencePiece model to: {temp_model_file}")

    with open(temp_model_file, 'wb') as f:
        f.write(base_spm.SerializeToString())

    # 使用合并后的vocab初始化新的分词器 / Initialize new tokenizer with merged vocab
    if tokenizer_type.lower() == "llama":
        merged_tokenizer = LlamaTokenizer(vocab_file=str(temp_model_file), legacy=True)
    else:
        merged_tokenizer = AutoTokenizer.from_pretrained(
            str(base_path),
            vocab_file=str(temp_model_file)
        )

    # 添加特殊tokens / Add special tokens
    if special_tokens:
        if verbose:
            print(f"Adding {len(special_tokens)} special tokens")

        for token in special_tokens:
            merged_tokenizer.add_tokens(token)

    # 保存为Hugging Face格式 / Save in Hugging Face format
    output_path.mkdir(parents=True, exist_ok=True)
    merged_tokenizer.save_pretrained(str(output_path))

    final_token_count = len(merged_tokenizer)

    if verbose:
        print(f"\nMerged tokenizer statistics:")
        print(f"  - Final token count: {final_token_count}")
        print(f"  - Original base tokens: {base_token_count}")
        print(f"  - Tokens added from additional model: {added_count}")
        print(f"  - Special tokens added: {len(special_tokens) if special_tokens else 0}")
        print(f"  - Saved to: {output_path}")

    return merged_tokenizer


def validate_merged_tokenizer(
    tokenizer_path: Union[str, Path],
    test_texts: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Dict[str, List[int]]:
    """
    Validate a merged tokenizer by testing encoding/decoding.
    验证合并后的分词器，通过测试编码/解码功能。

    Args:
        tokenizer_path: Path to merged tokenizer directory
                       合并后分词器的目录路径
        test_texts: Dictionary of test texts {name: text}
                   测试文本字典 {名称: 文本}
        verbose: Print validation results / 打印验证结果

    Returns:
        Dict[str, List[int]]: Dictionary mapping text names to token IDs
                             将文本名称映射到token ID的字典

    Example:
        >>> results = validate_merged_tokenizer(
        ...     "merged_tokenizer",
        ...     test_texts={
        ...         "chinese": "你好，世界！",
        ...         "english": "Hello, world!",
        ...         "mixed": "Hello 你好 World 世界"
        ...     }
        ... )
    """
    tokenizer_path = Path(tokenizer_path)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")

    # 加载分词器 / Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

    if verbose:
        print(f"Loaded tokenizer from: {tokenizer_path}")
        print(f"Vocabulary size: {len(tokenizer)}")

    # 默认测试文本 / Default test texts
    if test_texts is None:
        test_texts = {
            "chinese": "你好，世界！这是一个测试。",
            "english": "Hello, world! This is a test.",
            "mixed": "Hello 你好 World 世界",
            "special_tokens": "<|system|> You are a helpful assistant. <|user|> Hi! <|assistant|>"
        }

    results = {}

    for name, text in test_texts.items():
        if verbose:
            print(f"\n--- Testing: {name} ---")
            print(f"Input text: {text}")

        # 编码 / Encode
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        if verbose:
            print(f"Token IDs: {token_ids}")
            print(f"Token count: {len(token_ids)}")

        # 解码 / Decode
        decoded_text = tokenizer.decode(token_ids)

        if verbose:
            print(f"Decoded text: {decoded_text}")

            # 检查往返一致性 / Check round-trip consistency
            if text == decoded_text:
                print("✓ Round-trip encoding/decoding successful")
            else:
                print("⚠ Warning: Decoded text differs from original")

        results[name] = token_ids

    return results


def export_vocabulary(
    tokenizer_path: Union[str, Path],
    output_file: Union[str, Path],
    format: str = "json",
    include_scores: bool = False
) -> None:
    """
    Export tokenizer vocabulary to a file for inspection.
    导出分词器词汇表到文件以供检查。

    Args:
        tokenizer_path: Path to tokenizer directory / 分词器目录路径
        output_file: Output file path / 输出文件路径
        format: Output format ("json", "txt", or "csv") / 输出格式（"json"、"txt"或"csv"）
        include_scores: Include token scores (for SentencePiece) / 包含token分数（用于SentencePiece）

    Example:
        >>> export_vocabulary(
        ...     "merged_tokenizer",
        ...     "vocab.json",
        ...     format="json"
        ... )
    """
    tokenizer_path = Path(tokenizer_path)
    output_file = Path(output_file)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    vocab = tokenizer.get_vocab()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    elif format == "txt":
        # 按token ID排序 / Sort by token ID
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        with open(output_file, 'w', encoding='utf-8') as f:
            for token, idx in sorted_vocab:
                f.write(f"{idx}\t{token}\n")

    elif format == "csv":
        import csv
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Token ID", "Token"])
            for token, idx in sorted_vocab:
                writer.writerow([idx, token])

    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json', 'txt', or 'csv'")

    print(f"Vocabulary exported to: {output_file}")
    print(f"Total tokens: {len(vocab)}")


if __name__ == "__main__":
    # Example usage / 使用示例
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge tokenizer vocabularies / 合并分词器词汇表"
    )
    parser.add_argument(
        "--base-tokenizer",
        type=str,
        required=True,
        help="Path to base tokenizer directory / 基础分词器目录路径"
    )
    parser.add_argument(
        "--additional-model",
        type=str,
        required=True,
        help="Path to additional SentencePiece model / 额外SentencePiece模型路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged tokenizer / 合并后分词器输出目录"
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=None,
        help="Special tokens to add / 要添加的特殊tokens"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate merged tokenizer / 验证合并后的分词器"
    )

    args = parser.parse_args()

    # 合并分词器 / Merge tokenizers
    tokenizer = merge_tokenizers(
        base_tokenizer_path=args.base_tokenizer,
        additional_sp_model_path=args.additional_model,
        output_dir=args.output_dir,
        special_tokens=args.special_tokens,
        verbose=True
    )

    # 可选验证 / Optional validation
    if args.validate:
        validate_merged_tokenizer(args.output_dir, verbose=True)
