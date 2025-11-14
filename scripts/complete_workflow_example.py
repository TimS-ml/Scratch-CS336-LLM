#!/usr/bin/env python3
"""
Complete Tokenizer Workflow Example / 完整的分词器工作流程示例

This script demonstrates the complete workflow for:
1. Training a Chinese tokenizer
2. Merging it with a base tokenizer (e.g., LLaMA)
3. Expanding a model's embeddings to use the new vocabulary

这个脚本演示了完整的工作流程：
1. 训练中文分词器
2. 将其与基础分词器（如LLaMA）合并
3. 扩展模型的嵌入以使用新词汇表

Author: LLM-from-Scratch Team
License: MIT
"""

import sys
from pathlib import Path

# Add project root to path / 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clean_llm.tokenizer import (
    train_chinese_tokenizer,
    test_tokenizer,
    merge_tokenizers,
    validate_merged_tokenizer,
    expand_model_for_tokenizer,
    verify_embedding_expansion,
)


def complete_workflow_example():
    """
    Demonstrate complete workflow / 演示完整工作流程
    """
    print("=" * 80)
    print("Complete Tokenizer Workflow Example")
    print("完整的分词器工作流程示例")
    print("=" * 80)

    # ===== Step 1: Train Chinese Tokenizer =====
    # ===== 步骤1：训练中文分词器 =====
    print("\n" + "=" * 80)
    print("STEP 1: Train Chinese SentencePiece Tokenizer")
    print("步骤1：训练中文SentencePiece分词器")
    print("=" * 80 + "\n")

    # Example: Train on Chinese text files
    # 示例：在中文文本文件上训练
    chinese_corpus_dir = "data/chinese_corpus"  # Update with your path / 更新为您的路径
    chinese_vocab_size = 20000
    chinese_output_dir = "outputs/chinese_tokenizer"

    print(f"Training Chinese tokenizer...")
    print(f"训练中文分词器...")
    print(f"  - Corpus directory: {chinese_corpus_dir}")
    print(f"  - Vocabulary size: {chinese_vocab_size}")
    print(f"  - Output directory: {chinese_output_dir}\n")

    try:
        chinese_model_path = train_chinese_tokenizer(
            input_files=chinese_corpus_dir,
            vocab_size=chinese_vocab_size,
            output_dir=chinese_output_dir,
            model_prefix="chinese_sp",
            model_type="bpe",
            verbose=True
        )

        print(f"\n✓ Chinese tokenizer trained successfully!")
        print(f"✓ 中文分词器训练成功！")
        print(f"  Model saved to: {chinese_model_path}")

        # Test the Chinese tokenizer / 测试中文分词器
        print("\nTesting Chinese tokenizer...")
        print("测试中文分词器...")
        test_tokenizer(chinese_model_path, verbose=True)

    except Exception as e:
        print(f"\n✗ Error training Chinese tokenizer: {e}")
        print(f"✗ 训练中文分词器时出错：{e}")
        print("\nNote: Make sure you have Chinese text files in the corpus directory.")
        print("注意：确保语料库目录中有中文文本文件。")
        print("\nSkipping to next step with placeholder paths...")
        print("跳到下一步，使用占位符路径...")
        chinese_model_path = "path/to/chinese_sp.model"

    # ===== Step 2: Merge Vocabularies =====
    # ===== 步骤2：合并词汇表 =====
    print("\n" + "=" * 80)
    print("STEP 2: Merge LLaMA Tokenizer with Chinese Tokens")
    print("步骤2：合并LLaMA分词器与中文Tokens")
    print("=" * 80 + "\n")

    base_tokenizer_path = "path/to/llama2_tokenizer"  # Update with your path / 更新为您的路径
    merged_output_dir = "outputs/merged_tokenizer"

    # Special tokens for chat format / 聊天格式的特殊tokens
    special_tokens = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|im_start|>",
        "<|im_end|>"
    ]

    print(f"Merging tokenizers...")
    print(f"合并分词器...")
    print(f"  - Base tokenizer: {base_tokenizer_path}")
    print(f"  - Chinese model: {chinese_model_path}")
    print(f"  - Special tokens: {len(special_tokens)} tokens")
    print(f"  - Output directory: {merged_output_dir}\n")

    try:
        merged_tokenizer = merge_tokenizers(
            base_tokenizer_path=base_tokenizer_path,
            additional_sp_model_path=chinese_model_path,
            output_dir=merged_output_dir,
            special_tokens=special_tokens,
            tokenizer_type="llama",
            verbose=True
        )

        print(f"\n✓ Tokenizers merged successfully!")
        print(f"✓ 分词器合并成功！")
        print(f"  Final vocabulary size: {len(merged_tokenizer)}")

        # Validate merged tokenizer / 验证合并后的分词器
        print("\nValidating merged tokenizer...")
        print("验证合并后的分词器...")
        validate_merged_tokenizer(merged_output_dir, verbose=True)

    except Exception as e:
        print(f"\n✗ Error merging tokenizers: {e}")
        print(f"✗ 合并分词器时出错：{e}")
        print("\nNote: Make sure you have the base tokenizer (e.g., LLaMA) available.")
        print("注意：确保您有基础分词器（如LLaMA）可用。")
        print("\nSkipping to next step...")
        print("跳到下一步...")

    # ===== Step 3: Expand Model Embeddings =====
    # ===== 步骤3：扩展模型嵌入 =====
    print("\n" + "=" * 80)
    print("STEP 3: Expand Model Embeddings for New Vocabulary")
    print("步骤3：为新词汇表扩展模型嵌入")
    print("=" * 80 + "\n")

    original_model_path = "path/to/original_model"  # Update with your path / 更新为您的路径
    expanded_model_output = "outputs/expanded_model"

    print(f"Expanding model embeddings...")
    print(f"扩展模型嵌入...")
    print(f"  - Original model: {original_model_path}")
    print(f"  - New tokenizer: {merged_output_dir}")
    print(f"  - Output directory: {expanded_model_output}\n")

    try:
        expanded_model, tokenizer = expand_model_for_tokenizer(
            model_path=original_model_path,
            tokenizer_path=merged_output_dir,
            output_dir=expanded_model_output,
            pad_to_multiple_of=64,
            initialization_strategy="normal",
            device_map="auto",
            verbose=True
        )

        print(f"\n✓ Model embeddings expanded successfully!")
        print(f"✓ 模型嵌入扩展成功！")
        print(f"  New vocabulary size: {len(tokenizer)}")
        print(f"  Embedding shape: {expanded_model.get_input_embeddings().weight.shape}")

        # Verify expansion / 验证扩展
        print("\nVerifying embedding expansion...")
        print("验证嵌入扩展...")
        verify_embedding_expansion(expanded_model, tokenizer, verbose=True)

    except Exception as e:
        print(f"\n✗ Error expanding model: {e}")
        print(f"✗ 扩展模型时出错：{e}")
        print("\nNote: Make sure you have the original model available.")
        print("注意：确保您有原始模型可用。")

    # ===== Workflow Complete =====
    # ===== 工作流程完成 =====
    print("\n" + "=" * 80)
    print("Workflow Complete! / 工作流程完成！")
    print("=" * 80)
    print("\nNext steps / 下一步:")
    print("  1. Fine-tune the expanded model on your Chinese dataset")
    print("     在您的中文数据集上微调扩展后的模型")
    print("  2. The new token embeddings will be learned during training")
    print("     新token嵌入将在训练期间学习")
    print("  3. Evaluate model performance on Chinese tasks")
    print("     在中文任务上评估模型性能")
    print()


def simple_usage_example():
    """
    Simple usage example for quick reference / 简单使用示例供快速参考
    """
    print("\n" + "=" * 80)
    print("Simple Usage Example / 简单使用示例")
    print("=" * 80 + "\n")

    print("# Import modules / 导入模块")
    print("from clean_llm.tokenizer import (")
    print("    train_chinese_tokenizer,")
    print("    merge_tokenizers,")
    print("    expand_model_for_tokenizer")
    print(")")

    print("\n# Step 1: Train Chinese tokenizer / 步骤1：训练中文分词器")
    print('model_path = train_chinese_tokenizer(')
    print('    input_files="chinese_corpus/",')
    print('    vocab_size=20000,')
    print('    output_dir="chinese_tokenizer"')
    print(')')

    print("\n# Step 2: Merge with base tokenizer / 步骤2：与基础分词器合并")
    print('merged_tokenizer = merge_tokenizers(')
    print('    base_tokenizer_path="llama2_tokenizer/",')
    print('    additional_sp_model_path=model_path,')
    print('    output_dir="merged_tokenizer/",')
    print('    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"]')
    print(')')

    print("\n# Step 3: Expand model embeddings / 步骤3：扩展模型嵌入")
    print('model, tokenizer = expand_model_for_tokenizer(')
    print('    model_path="original_model/",')
    print('    tokenizer_path="merged_tokenizer/",')
    print('    output_dir="expanded_model/"')
    print(')')

    print("\n# Use the expanded model / 使用扩展后的模型")
    print('# Now fine-tune on your Chinese dataset!')
    print('# 现在在您的中文数据集上微调！')
    print()


def main():
    """Main function / 主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Complete tokenizer workflow example / 完整的分词器工作流程示例"
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Show simple usage example only / 仅显示简单使用示例"
    )

    args = parser.parse_args()

    if args.simple:
        simple_usage_example()
    else:
        complete_workflow_example()


if __name__ == "__main__":
    main()
