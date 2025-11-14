#!/usr/bin/env python3
"""
Merge Tokenizers Script / 合并分词器脚本

This script demonstrates how to merge vocabularies from different tokenizers,
particularly useful for combining base models (e.g., LLaMA) with language-specific
tokens (e.g., Chinese).

该脚本演示如何合并不同分词器的词汇表，特别适用于将基础模型（如LLaMA）
与特定语言的tokens（如中文）组合。

Usage / 使用方法:
    python scripts/merge_tokenizers.py \\
        --base-tokenizer path/to/llama/tokenizer \\
        --chinese-model path/to/chinese.model \\
        --output-dir merged_tokenizer

Author: LLM-from-Scratch Team
License: MIT
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Add project root to path / 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from clean_llm.tokenizer import (
        merge_tokenizers,
        validate_merged_tokenizer,
        export_vocabulary
    )
except ImportError as e:
    print(f"Error importing clean_llm modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


def main():
    """Main function / 主函数"""
    parser = argparse.ArgumentParser(
        description="Merge tokenizer vocabularies / 合并分词器词汇表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:

1. Basic merge / 基础合并:
   python scripts/merge_tokenizers.py \\
       --base-tokenizer llama2_tokenizer \\
       --chinese-model chinese_sp.model \\
       --output-dir merged_tokenizer

2. Merge with custom special tokens / 使用自定义特殊tokens合并:
   python scripts/merge_tokenizers.py \\
       --base-tokenizer llama2_tokenizer \\
       --chinese-model chinese_sp.model \\
       --output-dir merged_tokenizer \\
       --special-tokens "<|system|>" "<|user|>" "<|assistant|>" \\
       --validate \\
       --export-vocab

3. Merge and export vocabulary / 合并并导出词汇表:
   python scripts/merge_tokenizers.py \\
       --base-tokenizer llama2_tokenizer \\
       --chinese-model chinese_sp.model \\
       --output-dir merged_tokenizer \\
       --export-vocab \\
       --vocab-format json
        """
    )

    # Required arguments / 必需参数
    parser.add_argument(
        "--base-tokenizer",
        type=str,
        required=True,
        help="Path to base tokenizer directory (e.g., LLaMA tokenizer)\n"
             "基础分词器目录路径（如LLaMA分词器）"
    )

    parser.add_argument(
        "--chinese-model",
        type=str,
        required=True,
        help="Path to Chinese SentencePiece model file (.model)\n"
             "中文SentencePiece模型文件路径（.model）"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged tokenizer\n"
             "合并后分词器的输出目录"
    )

    # Optional arguments / 可选参数
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=None,
        help="Special tokens to add (e.g., <|system|> <|user|> <|assistant|>)\n"
             "要添加的特殊tokens（如<|system|> <|user|> <|assistant|>）"
    )

    parser.add_argument(
        "--tokenizer-type",
        type=str,
        default="llama",
        choices=["llama", "auto"],
        help="Type of base tokenizer (default: llama)\n"
             "基础分词器类型（默认：llama）"
    )

    parser.add_argument(
        "--keep-duplicate-scores",
        action="store_true",
        help="Keep scores from additional model for duplicate tokens\n"
             "保留额外模型中重复token的分数"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate merged tokenizer after creation\n"
             "创建后验证合并的分词器"
    )

    parser.add_argument(
        "--export-vocab",
        action="store_true",
        help="Export vocabulary to file\n"
             "导出词汇表到文件"
    )

    parser.add_argument(
        "--vocab-format",
        type=str,
        default="json",
        choices=["json", "txt", "csv"],
        help="Vocabulary export format (default: json)\n"
             "词汇表导出格式（默认：json）"
    )

    parser.add_argument(
        "--vocab-output",
        type=str,
        default=None,
        help="Custom path for vocabulary export (default: output_dir/vocab.{format})\n"
             "词汇表导出的自定义路径（默认：output_dir/vocab.{format}）"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages\n"
             "抑制进度消息"
    )

    args = parser.parse_args()

    # Convert paths to Path objects / 转换路径为Path对象
    base_tokenizer_path = Path(args.base_tokenizer)
    chinese_model_path = Path(args.chinese_model)
    output_dir = Path(args.output_dir)

    verbose = not args.quiet

    # Display configuration / 显示配置
    if verbose:
        print("=" * 70)
        print("Tokenizer Vocabulary Merge / 分词器词汇表合并")
        print("=" * 70)
        print(f"\nConfiguration / 配置:")
        print(f"  Base tokenizer / 基础分词器: {base_tokenizer_path}")
        print(f"  Chinese model / 中文模型: {chinese_model_path}")
        print(f"  Output directory / 输出目录: {output_dir}")
        print(f"  Tokenizer type / 分词器类型: {args.tokenizer_type}")

        if args.special_tokens:
            print(f"  Special tokens / 特殊tokens: {args.special_tokens}")

        print("\n" + "-" * 70 + "\n")

    # Step 1: Merge tokenizers / 步骤1：合并分词器
    try:
        if verbose:
            print("Step 1: Merging tokenizer vocabularies...")
            print("步骤1：合并分词器词汇表...\n")

        merged_tokenizer = merge_tokenizers(
            base_tokenizer_path=base_tokenizer_path,
            additional_sp_model_path=chinese_model_path,
            output_dir=output_dir,
            special_tokens=args.special_tokens,
            tokenizer_type=args.tokenizer_type,
            keep_duplicate_scores=args.keep_duplicate_scores,
            verbose=verbose
        )

        if verbose:
            print("\n✓ Tokenizers merged successfully!")
            print("✓ 分词器合并成功！\n")

    except Exception as e:
        print(f"\n✗ Error merging tokenizers: {e}")
        print(f"✗ 合并分词器时出错：{e}")
        sys.exit(1)

    # Step 2: Validate (if requested) / 步骤2：验证（如果请求）
    if args.validate:
        try:
            if verbose:
                print("-" * 70)
                print("\nStep 2: Validating merged tokenizer...")
                print("步骤2：验证合并后的分词器...\n")

            # Define test texts / 定义测试文本
            test_texts = {
                "chinese": "你好，世界！这是一个中文测试。",
                "english": "Hello, world! This is an English test.",
                "mixed": "Hello 你好 World 世界！Let's test 测试 mixed 混合 text.",
            }

            # Add special token test if special tokens were added
            # 如果添加了特殊tokens，添加特殊token测试
            if args.special_tokens:
                special_text = " ".join(args.special_tokens)
                test_texts["special_tokens"] = f"Testing special tokens: {special_text}"

            results = validate_merged_tokenizer(
                tokenizer_path=output_dir,
                test_texts=test_texts,
                verbose=verbose
            )

            if verbose:
                print("\n✓ Validation completed!")
                print("✓ 验证完成！\n")

        except Exception as e:
            print(f"\n⚠ Warning: Validation failed: {e}")
            print(f"⚠ 警告：验证失败：{e}")

    # Step 3: Export vocabulary (if requested) / 步骤3：导出词汇表（如果请求）
    if args.export_vocab:
        try:
            if verbose:
                print("-" * 70)
                print("\nStep 3: Exporting vocabulary...")
                print("步骤3：导出词汇表...\n")

            # Determine output file / 确定输出文件
            if args.vocab_output:
                vocab_file = Path(args.vocab_output)
            else:
                vocab_file = output_dir / f"vocab.{args.vocab_format}"

            export_vocabulary(
                tokenizer_path=output_dir,
                output_file=vocab_file,
                format=args.vocab_format
            )

            if verbose:
                print(f"\n✓ Vocabulary exported to: {vocab_file}")
                print(f"✓ 词汇表已导出到：{vocab_file}\n")

        except Exception as e:
            print(f"\n⚠ Warning: Vocabulary export failed: {e}")
            print(f"⚠ 警告：词汇表导出失败：{e}")

    # Final summary / 最终摘要
    if verbose:
        print("=" * 70)
        print("Merge Completed Successfully! / 合并成功完成！")
        print("=" * 70)
        print(f"\nMerged tokenizer saved to / 合并后的分词器保存到: {output_dir}")
        print(f"Vocabulary size / 词汇表大小: {len(merged_tokenizer)}")
        print("\nYou can now use this tokenizer with:")
        print("现在可以使用此分词器：")
        print(f"\n  from transformers import AutoTokenizer")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
        print()


if __name__ == "__main__":
    main()
