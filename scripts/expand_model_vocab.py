#!/usr/bin/env python3
"""
Expand Model Vocabulary Script / 扩展模型词汇表脚本

This script expands a language model's embedding layers to accommodate a new
tokenizer with an expanded vocabulary. It preserves weights for existing tokens
and properly initializes new token embeddings.

该脚本扩展语言模型的嵌入层以适应具有扩展词汇表的新分词器。
它保留现有tokens的权重并正确初始化新token的嵌入。

Usage / 使用方法:
    python scripts/expand_model_vocab.py \\
        --model-path path/to/original/model \\
        --tokenizer-path path/to/merged/tokenizer \\
        --output-dir expanded_model

Author: LLM-from-Scratch Team
License: MIT
"""

import sys
import argparse
from pathlib import Path
import torch


try:
    from scratch_cs336.core.tokenizer import (
        expand_model_for_tokenizer,
        verify_embedding_expansion
    )
except ImportError as e:
    print(f"Error importing scratch_cs336 modules: {e}")
    print("Please ensure you're running from the project root directory.")
    sys.exit(1)


def parse_dtype(dtype_str: str) -> torch.dtype:
    """
    Parse dtype string to torch.dtype.
    解析dtype字符串为torch.dtype。

    Args:
        dtype_str: Dtype string (e.g., "float32", "float16", "bfloat16")

    Returns:
        torch.dtype: Corresponding torch dtype
    """
    dtype_map = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
    }

    dtype_str = dtype_str.lower()
    if dtype_str not in dtype_map:
        raise ValueError(
            f"Invalid dtype: {dtype_str}. "
            f"Valid options: {', '.join(dtype_map.keys())}"
        )

    return dtype_map[dtype_str]


def main():
    """Main function / 主函数"""
    parser = argparse.ArgumentParser(
        description="Expand model embeddings for new tokenizer / 为新分词器扩展模型嵌入",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:

1. Basic expansion / 基础扩展:
   python scripts/expand_model_vocab.py \\
       --model-path original_model \\
       --tokenizer-path merged_tokenizer \\
       --output-dir expanded_model

2. Expansion with custom settings / 使用自定义设置扩展:
   python scripts/expand_model_vocab.py \\
       --model-path original_model \\
       --tokenizer-path merged_tokenizer \\
       --output-dir expanded_model \\
       --init-strategy xavier \\
       --pad-to-multiple-of 128 \\
       --dtype float16 \\
       --verify

3. Expansion with CPU-only mode / 仅使用CPU模式扩展:
   python scripts/expand_model_vocab.py \\
       --model-path original_model \\
       --tokenizer-path merged_tokenizer \\
       --output-dir expanded_model \\
       --device-map cpu \\
       --verify
        """
    )

    # Required arguments / 必需参数
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to original model directory\n"
             "原始模型目录路径"
    )

    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to new tokenizer directory (with expanded vocabulary)\n"
             "新分词器目录路径（具有扩展的词汇表）"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for expanded model\n"
             "扩展模型的输出目录"
    )

    # Model loading options / 模型加载选项
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device mapping for model loading (default: auto)\n"
             "模型加载的设备映射（默认：auto）\n"
             "Options: 'auto', 'cpu', 'cuda', 'cuda:0', etc."
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type for model weights (default: model's original dtype)\n"
             "模型权重的数据类型（默认：模型的原始dtype）\n"
             "Options: float32, float16, bfloat16, etc."
    )

    # Expansion options / 扩展选项
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=64,
        help="Pad vocabulary size to multiple of this value (default: 64)\n"
             "将词汇表大小填充到此值的倍数（默认：64）\n"
             "Set to 0 to disable padding / 设置为0以禁用填充"
    )

    parser.add_argument(
        "--init-strategy",
        type=str,
        default="normal",
        choices=["normal", "uniform", "xavier", "kaiming"],
        help="Initialization strategy for new embeddings (default: normal)\n"
             "新嵌入的初始化策略（默认：normal）\n"
             "  - normal: Normal distribution / 正态分布\n"
             "  - uniform: Uniform distribution / 均匀分布\n"
             "  - xavier: Xavier/Glorot initialization / Xavier初始化\n"
             "  - kaiming: Kaiming/He initialization / Kaiming初始化"
    )

    # Verification options / 验证选项
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify embedding expansion after completion\n"
             "完成后验证嵌入扩展"
    )

    parser.add_argument(
        "--test-token-ids",
        type=int,
        nargs="*",
        default=None,
        help="Specific token IDs to test during verification\n"
             "验证期间要测试的特定token ID"
    )

    # Output options / 输出选项
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages\n"
             "抑制进度消息"
    )

    args = parser.parse_args()

    # Convert paths to Path objects / 转换路径为Path对象
    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)
    output_dir = Path(args.output_dir)

    verbose = not args.quiet

    # Parse dtype if provided / 如果提供则解析dtype
    torch_dtype = None
    if args.dtype:
        try:
            torch_dtype = parse_dtype(args.dtype)
            if verbose:
                print(f"Using dtype: {torch_dtype}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Handle padding option / 处理填充选项
    pad_to_multiple_of = args.pad_to_multiple_of if args.pad_to_multiple_of > 0 else None

    # Display configuration / 显示配置
    if verbose:
        print("=" * 70)
        print("Model Vocabulary Expansion / 模型词汇表扩展")
        print("=" * 70)
        print(f"\nConfiguration / 配置:")
        print(f"  Model path / 模型路径: {model_path}")
        print(f"  Tokenizer path / 分词器路径: {tokenizer_path}")
        print(f"  Output directory / 输出目录: {output_dir}")
        print(f"  Device mapping / 设备映射: {args.device_map}")

        if torch_dtype:
            print(f"  Data type / 数据类型: {torch_dtype}")

        print(f"  Initialization strategy / 初始化策略: {args.init_strategy}")

        if pad_to_multiple_of:
            print(f"  Pad to multiple of / 填充到倍数: {pad_to_multiple_of}")
        else:
            print(f"  Padding / 填充: disabled / 禁用")

        print("\n" + "-" * 70 + "\n")

    # Step 1: Expand model / 步骤1：扩展模型
    try:
        if verbose:
            print("Step 1: Loading model and tokenizer, expanding embeddings...")
            print("步骤1：加载模型和分词器，扩展嵌入...\n")

        model, tokenizer = expand_model_for_tokenizer(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            pad_to_multiple_of=pad_to_multiple_of,
            initialization_strategy=args.init_strategy,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            verbose=verbose
        )

        if verbose:
            print("\n✓ Model expansion completed successfully!")
            print("✓ 模型扩展成功完成！\n")

    except Exception as e:
        print(f"\n✗ Error expanding model: {e}")
        print(f"✗ 扩展模型时出错：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Verify (if requested) / 步骤2：验证（如果请求）
    if args.verify:
        try:
            if verbose:
                print("-" * 70)
                print("\nStep 2: Verifying embedding expansion...")
                print("步骤2：验证嵌入扩展...\n")

            results = verify_embedding_expansion(
                model=model,
                tokenizer=tokenizer,
                test_token_ids=args.test_token_ids,
                verbose=verbose
            )

            if results["status"] == "success":
                if verbose:
                    print("\n✓ Verification passed!")
                    print("✓ 验证通过！\n")
            elif results["status"] == "warning":
                if verbose:
                    print("\n⚠ Verification completed with warnings")
                    print("⚠ 验证完成但有警告\n")
            else:
                print("\n✗ Verification failed!")
                print("✗ 验证失败！")
                print("\nIssues / 问题:")
                for issue in results["issues"]:
                    print(f"  - {issue}")
                sys.exit(1)

        except Exception as e:
            print(f"\n⚠ Warning: Verification failed: {e}")
            print(f"⚠ 警告：验证失败：{e}")
            import traceback
            traceback.print_exc()

    # Final summary / 最终摘要
    if verbose:
        print("=" * 70)
        print("Expansion Completed Successfully! / 扩展成功完成！")
        print("=" * 70)
        print(f"\nExpanded model saved to / 扩展后的模型保存到: {output_dir}")
        print(f"Vocabulary size / 词汇表大小: {len(tokenizer)}")
        print(f"Embedding shape / 嵌入形状: {model.get_input_embeddings().weight.shape}")

        if hasattr(model, "lm_head"):
            print(f"LM head shape / LM头形状: {model.lm_head.weight.shape}")

        print("\nYou can now use this model with:")
        print("现在可以使用此模型：")
        print(f"\n  from transformers import AutoModelForCausalLM, AutoTokenizer")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
        print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
        print()

        print("Next steps / 下一步:")
        print("  1. Fine-tune the model on your target dataset")
        print("     在目标数据集上微调模型")
        print("  2. The new token embeddings will be updated during training")
        print("     新token嵌入将在训练期间更新")
        print()


if __name__ == "__main__":
    main()
