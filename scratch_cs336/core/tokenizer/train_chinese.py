"""
Chinese SentencePiece Tokenizer Training Module / 中文SentencePiece分词器训练模块

This module provides functionality to train SentencePiece tokenizers optimized
for Chinese text, with support for large corpora and various configuration options.

该模块提供了训练针对中文文本优化的SentencePiece分词器的功能，
支持大型语料库和各种配置选项。

Author: LLM-from-Scratch Team
License: MIT
"""

import os
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
import glob

try:
    import sentencepiece as spm
except ImportError as e:
    raise ImportError(
        "sentencepiece not installed. Please run: pip install sentencepiece"
    ) from e


def train_chinese_tokenizer(
    input_files: Union[str, Path, List[Union[str, Path]]],
    vocab_size: int,
    output_dir: Union[str, Path],
    model_prefix: str = "chinese_tokenizer",
    model_type: str = "bpe",
    character_coverage: float = 0.9995,
    split_digits: bool = True,
    byte_fallback: bool = True,
    normalization_rule_name: str = "identity",
    max_sentence_length: int = 16384,
    num_threads: Optional[int] = None,
    user_defined_symbols: Optional[List[str]] = None,
    control_symbols: Optional[List[str]] = None,
    unk_surface: str = r" \342\201\207 ",
    train_extremely_large_corpus: bool = False,
    verbose: bool = True,
    **kwargs
) -> str:
    """
    Train a SentencePiece tokenizer optimized for Chinese text.
    训练针对中文文本优化的SentencePiece分词器。

    This function trains a SentencePiece model with settings optimized for
    Chinese language processing. It supports both BPE and unigram models.

    该函数使用针对中文语言处理优化的设置训练SentencePiece模型。
    支持BPE和unigram模型。

    Args:
        input_files: Path(s) to input text file(s) or directory containing .txt files
                    输入文本文件路径或包含.txt文件的目录
        vocab_size: Target vocabulary size / 目标词汇表大小
        output_dir: Output directory for trained model / 训练模型的输出目录
        model_prefix: Prefix for output model files / 输出模型文件的前缀
        model_type: Model type ("bpe", "unigram", "char", or "word")
                   模型类型（"bpe"、"unigram"、"char"或"word"）
        character_coverage: Character coverage ratio (0.9995 recommended for Chinese)
                          字符覆盖率（中文推荐0.9995）
        split_digits: Split digits into individual tokens / 将数字拆分为单独的tokens
        byte_fallback: Use byte fallback for unknown characters / 对未知字符使用字节回退
        normalization_rule_name: Text normalization rule / 文本规范化规则
        max_sentence_length: Maximum sentence length in bytes / 最大句子长度（字节）
        num_threads: Number of threads (default: CPU count) / 线程数（默认：CPU数量）
        user_defined_symbols: User-defined symbols to preserve / 要保留的用户定义符号
        control_symbols: Control symbols (e.g., special tokens) / 控制符号（如特殊tokens）
        unk_surface: Surface representation for unknown tokens / 未知token的表面表示
        train_extremely_large_corpus: Enable for very large corpora (>10M sentences)
                                     为非常大的语料库启用（>1000万句）
        verbose: Print training progress / 打印训练进度
        **kwargs: Additional SentencePiece training arguments / 额外的SentencePiece训练参数

    Returns:
        str: Path to trained model file / 训练模型文件的路径

    Example:
        >>> model_path = train_chinese_tokenizer(
        ...     input_files="chinese_corpus",
        ...     vocab_size=20000,
        ...     output_dir="tokenizer_output",
        ...     model_prefix="chinese_sp"
        ... )
        >>> print(f"Model saved to: {model_path}")
    """
    # 处理输入文件 / Process input files
    if isinstance(input_files, (str, Path)):
        input_path = Path(input_files)

        if input_path.is_dir():
            # 如果是目录，查找所有.txt文件 / If directory, find all .txt files
            text_files = sorted(glob.glob(str(input_path / "*.txt")))
            if not text_files:
                raise FileNotFoundError(f"No .txt files found in directory: {input_path}")
        elif input_path.is_file():
            # 单个文件 / Single file
            text_files = [str(input_path)]
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")
    else:
        # 文件列表 / List of files
        text_files = [str(Path(f)) for f in input_files]

    # 验证所有文件存在 / Validate all files exist
    for file_path in text_files:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    if verbose:
        print(f"Training SentencePiece tokenizer with the following settings:")
        print(f"  - Input files: {len(text_files)} file(s)")
        for i, file_path in enumerate(text_files[:5], 1):
            print(f"    {i}. {file_path}")
        if len(text_files) > 5:
            print(f"    ... and {len(text_files) - 5} more files")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Model type: {model_type}")
        print(f"  - Character coverage: {character_coverage}")

    # 创建输出目录 / Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 完整的模型前缀路径 / Full model prefix path
    full_model_prefix = str(output_path / model_prefix)

    # 设置线程数 / Set thread count
    if num_threads is None:
        num_threads = os.cpu_count()

    # 构建训练参数 / Build training arguments
    train_args = {
        "input": text_files,
        "model_prefix": full_model_prefix,
        "model_type": model_type,
        "vocab_size": vocab_size,
        "character_coverage": character_coverage,
        "num_threads": num_threads,
        "split_digits": split_digits,
        "byte_fallback": byte_fallback,
        "normalization_rule_name": normalization_rule_name,
        "max_sentence_length": max_sentence_length,
        "unk_surface": unk_surface,
        "input_format": "text",
        "self_test_sample_size": 0,
        "train_extremely_large_corpus": train_extremely_large_corpus,
    }

    # 添加可选参数 / Add optional arguments
    if user_defined_symbols:
        train_args["user_defined_symbols"] = user_defined_symbols

    if control_symbols:
        train_args["control_symbols"] = control_symbols

    # 允许覆盖任何参数 / Allow overriding any argument
    train_args.update(kwargs)

    # 训练模型 / Train model
    if verbose:
        print("\nTraining tokenizer... This may take a while for large corpora.")

    try:
        spm.SentencePieceTrainer.train(**train_args)
    except Exception as e:
        raise RuntimeError(f"SentencePiece training failed: {e}") from e

    model_file = f"{full_model_prefix}.model"
    vocab_file = f"{full_model_prefix}.vocab"

    if verbose:
        print(f"\n✓ Training completed successfully!")
        print(f"  - Model file: {model_file}")
        print(f"  - Vocab file: {vocab_file}")

    return model_file


def test_tokenizer(
    model_path: Union[str, Path],
    test_texts: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Test a trained SentencePiece tokenizer.
    测试训练好的SentencePiece分词器。

    Args:
        model_path: Path to trained .model file / 训练好的.model文件路径
        test_texts: Dictionary of test texts {name: text} / 测试文本字典 {名称: 文本}
        verbose: Print test results / 打印测试结果

    Returns:
        Dict[str, List[str]]: Dictionary mapping text names to token lists
                             将文本名称映射到token列表的字典

    Example:
        >>> results = test_tokenizer(
        ...     "output/chinese_sp.model",
        ...     test_texts={"example": "你好，世界！"}
        ... )
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载模型 / Load model
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    if verbose:
        print(f"Loaded SentencePiece model from: {model_path}")
        print(f"Vocabulary size: {sp.vocab_size()}")

    # 默认测试文本 / Default test texts
    if test_texts is None:
        test_texts = {
            "chinese_simple": "你好，世界！",
            "chinese_complex": "翻译下面的句子为英文：有朋自远方来，不亦乐乎",
            "mixed": "Hello 你好 World 世界",
            "numbers": "2024年1月1日，温度是25.5℃",
            "punctuation": "这是一个测试，包含：逗号、冒号、句号。",
        }

    results = {}

    for name, text in test_texts.items():
        # 分词 / Tokenize
        pieces = sp.encode_as_pieces(text)
        ids = sp.encode_as_ids(text)

        if verbose:
            print(f"\n--- Test: {name} ---")
            print(f"Input: {text}")
            print(f"Tokens: {pieces}")
            print(f"Token count: {len(pieces)}")
            print(f"Token IDs: {ids}")

            # 解码测试 / Decode test
            decoded = sp.decode(ids)
            print(f"Decoded: {decoded}")

            if decoded == text:
                print("✓ Round-trip successful")
            else:
                print("⚠ Round-trip differs from original")

        results[name] = pieces

    return results


def analyze_tokenizer(
    model_path: Union[str, Path],
    top_n: int = 50,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze a trained SentencePiece tokenizer.
    分析训练好的SentencePiece分词器。

    Args:
        model_path: Path to trained .model file / 训练好的.model文件路径
        top_n: Number of top tokens to show / 显示的顶部token数量
        verbose: Print analysis results / 打印分析结果

    Returns:
        Dict[str, Any]: Analysis results / 分析结果

    Example:
        >>> analysis = analyze_tokenizer("output/chinese_sp.model")
        >>> print(f"Vocab size: {analysis['vocab_size']}")
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 加载模型 / Load model
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))

    vocab_size = sp.vocab_size()

    # 收集分析数据 / Collect analysis data
    analysis = {
        "vocab_size": vocab_size,
        "model_type": sp.model_type(),
        "unk_id": sp.unk_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "pad_id": sp.pad_id(),
        "top_tokens": []
    }

    # 获取top N tokens / Get top N tokens
    for i in range(min(top_n, vocab_size)):
        token = sp.id_to_piece(i)
        score = sp.get_score(i)
        analysis["top_tokens"].append({
            "id": i,
            "token": token,
            "score": score
        })

    if verbose:
        print(f"\n=== Tokenizer Analysis ===")
        print(f"Vocabulary size: {analysis['vocab_size']}")
        print(f"Model type: {analysis['model_type']}")
        print(f"Special token IDs:")
        print(f"  - UNK: {analysis['unk_id']}")
        print(f"  - BOS: {analysis['bos_id']}")
        print(f"  - EOS: {analysis['eos_id']}")
        print(f"  - PAD: {analysis['pad_id']}")

        print(f"\nTop {len(analysis['top_tokens'])} tokens:")
        for item in analysis["top_tokens"][:20]:
            token_repr = repr(item['token'])
            print(f"  {item['id']:6d}: {token_repr:20s} (score: {item['score']:.4f})")

        if len(analysis["top_tokens"]) > 20:
            print(f"  ... and {len(analysis['top_tokens']) - 20} more tokens")

    return analysis


def convert_to_huggingface_tokenizer(
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    tokenizer_class: str = "llama",
    special_tokens: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> str:
    """
    Convert SentencePiece model to Hugging Face tokenizer format.
    将SentencePiece模型转换为Hugging Face分词器格式。

    Args:
        model_path: Path to .model file / .model文件路径
        output_dir: Output directory / 输出目录
        tokenizer_class: Tokenizer class to use ("llama" or others)
                        要使用的分词器类（"llama"或其他）
        special_tokens: Dictionary of special tokens to add
                       要添加的特殊tokens字典
        verbose: Print conversion progress / 打印转换进度

    Returns:
        str: Path to output directory / 输出目录路径

    Example:
        >>> output_path = convert_to_huggingface_tokenizer(
        ...     "chinese_sp.model",
        ...     "chinese_tokenizer_hf",
        ...     special_tokens={
        ...         "bos_token": "<s>",
        ...         "eos_token": "</s>",
        ...         "unk_token": "<unk>"
        ...     }
        ... )
    """
    try:
        from transformers import LlamaTokenizer, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers not installed. Please run: pip install transformers"
        ) from e

    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if verbose:
        print(f"Converting SentencePiece model to Hugging Face format")
        print(f"  - Input: {model_path}")
        print(f"  - Output: {output_dir}")

    # 创建输出目录 / Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载分词器 / Load tokenizer
    if tokenizer_class.lower() == "llama":
        tokenizer = LlamaTokenizer(vocab_file=str(model_path), legacy=True)
    else:
        # 其他分词器类型 / Other tokenizer types
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path.parent),
            vocab_file=str(model_path)
        )

    # 添加特殊tokens / Add special tokens
    if special_tokens:
        if verbose:
            print(f"  - Adding {len(special_tokens)} special tokens")

        for key, value in special_tokens.items():
            setattr(tokenizer, key, value)

    # 保存 / Save
    tokenizer.save_pretrained(str(output_dir))

    if verbose:
        print(f"✓ Conversion completed")
        print(f"  - Saved to: {output_dir}")
        print(f"  - Vocabulary size: {len(tokenizer)}")

    return str(output_dir)


if __name__ == "__main__":
    # Example usage / 使用示例
    import argparse

    parser = argparse.ArgumentParser(
        description="Train Chinese SentencePiece tokenizer / 训练中文SentencePiece分词器"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input text file(s) or directory / 输入文本文件或目录"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Vocabulary size / 词汇表大小"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory / 输出目录"
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="chinese_tokenizer",
        help="Model prefix / 模型前缀"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bpe",
        choices=["bpe", "unigram", "char", "word"],
        help="Model type / 模型类型"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test tokenizer after training / 训练后测试分词器"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze tokenizer after training / 训练后分析分词器"
    )

    args = parser.parse_args()

    # 训练分词器 / Train tokenizer
    model_path = train_chinese_tokenizer(
        input_files=args.input,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        model_type=args.model_type,
        verbose=True
    )

    # 可选测试 / Optional testing
    if args.test:
        test_tokenizer(model_path, verbose=True)

    # 可选分析 / Optional analysis
    if args.analyze:
        analyze_tokenizer(model_path, verbose=True)
