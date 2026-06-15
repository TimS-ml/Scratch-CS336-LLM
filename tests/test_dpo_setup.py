#!/usr/bin/env python3
"""
DPO Training Setup Test
DPO 训练设置测试

This script tests the DPO training setup by creating a minimal example dataset
and running a quick sanity check.

此脚本通过创建最小示例数据集并运行快速完整性检查来测试 DPO 训练设置。

Usage / 用法:
    python scripts/test_dpo_setup.py

Author: wdndev
Date: 2025-11-14
"""

import os
import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent


def create_test_data():
    """
    Create minimal test dataset
    创建最小测试数据集
    """
    print("=" * 80)
    print("Creating test dataset...")
    print("创建测试数据集...")
    print("=" * 80)

    # Create test data directory / 创建测试数据目录
    test_data_dir = project_root / "data" / "dpo_test"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    # Example training data / 示例训练数据
    train_data = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris, which is also its largest city and a major European center of art, culture, and history.",
            "rejected": "Paris."
        },
        {
            "prompt": "How do I make a sandwich?",
            "chosen": "Here's a simple way to make a sandwich:\n1. Take two slices of bread\n2. Add your favorite ingredients (lettuce, tomato, cheese, meat)\n3. Add condiments if desired\n4. Put the slices together\n5. Cut diagonally for easier eating\nEnjoy your meal!",
            "rejected": "Put stuff between bread."
        },
        {
            "prompt": "Explain photosynthesis",
            "chosen": "Photosynthesis is the process by which plants convert light energy (usually from the sun) into chemical energy stored in glucose. During this process, plants take in carbon dioxide from the air and water from the soil, and use sunlight to convert these into glucose and oxygen. The oxygen is released as a byproduct.",
            "rejected": "Plants make food from sunlight."
        },
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention. Common applications include image recognition, natural language processing, and recommendation systems.",
            "rejected": "ML is when computers learn stuff."
        },
        {
            "prompt": "How to stay healthy?",
            "chosen": "Maintaining good health involves several key practices:\n1. Regular exercise (at least 30 minutes daily)\n2. Balanced diet with fruits, vegetables, and whole grains\n3. Adequate sleep (7-9 hours per night)\n4. Stress management through meditation or hobbies\n5. Regular health checkups\n6. Staying hydrated\n7. Avoiding harmful substances like tobacco and excessive alcohol",
            "rejected": "Exercise and eat good food."
        },
    ]

    # Duplicate to create more samples / 复制以创建更多样本
    train_data = train_data * 10  # 50 samples total

    # Save training data / 保存训练数据
    train_file = test_data_dir / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created training data: {train_file}")
    print(f"创建训练数据: {train_file}")
    print(f"Number of samples: {len(train_data)}")
    print(f"样本数量: {len(train_data)}")

    # Create evaluation data (smaller) / 创建评估数据（较小）
    eval_data = train_data[:5]
    eval_file = test_data_dir / "eval.jsonl"
    with open(eval_file, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Created evaluation data: {eval_file}")
    print(f"创建评估数据: {eval_file}")
    print(f"Number of samples: {len(eval_data)}")
    print(f"样本数量: {len(eval_data)}")
    print()

    return train_file, eval_file


def create_test_config(train_file, eval_file):
    """
    Create test configuration file
    创建测试配置文件
    """
    print("=" * 80)
    print("Creating test configuration...")
    print("创建测试配置...")
    print("=" * 80)

    config = {
        "model": {
            "model_name_or_path": "Qwen/Qwen2.5-0.5B",
            "trust_remote_code": True,
            "use_cache": False,
            "torch_dtype": "auto",
        },
        "data": {
            "train_dataset_path": str(train_file),
            "eval_dataset_path": str(eval_file),
            "max_length": 512,
            "max_prompt_length": 256,
            "sanity_check": True,
            "num_proc": 2,
            "system_prompt": "你是一个有帮助的AI助手。",
        },
        "training": {
            "output_dir": str(project_root / "outputs" / "dpo_test"),
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 5e-5,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "optim": "adamw_torch",
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,
            "bf16": False,
            "fp16": False,
            "logging_dir": str(project_root / "outputs" / "dpo_test" / "logs"),
            "logging_strategy": "steps",
            "logging_steps": 5,
            "evaluation_strategy": "steps",
            "eval_steps": 10,
            "save_strategy": "steps",
            "save_steps": 10,
            "save_total_limit": 2,
            "report_to": "none",
            "remove_unused_columns": False,
            "seed": 42,
        },
        "dpo": {
            "beta": 0.1,
            "loss_type": "sigmoid",
            "label_smoothing": 0.0,
            "generate_during_eval": False,
        },
        "lora": {
            "use_lora": False,
        }
    }

    # Save config / 保存配置
    config_file = project_root / "configs" / "dpo_test.yaml"
    import yaml
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"Created test config: {config_file}")
    print(f"创建测试配置: {config_file}")
    print()

    return config_file


def check_data_loading():
    """
    Test dataset loading
    测试数据集加载
    """
    print("=" * 80)
    print("Testing dataset loading...")
    print("测试数据集加载...")
    print("=" * 80)

    try:
        from scratch_cs336.training.datasets import load_dpo_dataset

        train_file = project_root / "data" / "dpo_test" / "train.jsonl"

        dataset = load_dpo_dataset(
            data_path=str(train_file),
            max_length=512,
            sanity_check=True,
            num_proc=1,
            system="你是一个有帮助的AI助手。",
        )

        print(f"✓ Successfully loaded dataset")
        print(f"✓ 成功加载数据集")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  数据集大小: {len(dataset)}")
        print(f"  Sample: {dataset[0]}")
        print()

        return True

    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print(f"✗ 加载数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_loading():
    """
    Test model and tokenizer loading
    测试模型和分词器加载
    """
    print("=" * 80)
    print("Testing model loading (this may take a while)...")
    print("测试模型加载（这可能需要一些时间）...")
    print("=" * 80)

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_name = "Qwen/Qwen2.5-0.5B"

        print(f"Loading tokenizer from {model_name}...")
        print(f"从 {model_name} 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"✓ Successfully loaded tokenizer")
        print(f"✓ 成功加载分词器")
        print()

        print(f"Loading model from {model_name}...")
        print(f"从 {model_name} 加载模型...")
        print("Note: This will download ~500MB if not cached")
        print("注意: 如果未缓存，这将下载约 500MB")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype="auto",
        )

        print(f"✓ Successfully loaded model")
        print(f"✓ 成功加载模型")
        print()

        return True

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print(f"✗ 加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main test function
    主测试函数
    """
    print("\n")
    print("=" * 80)
    print("DPO Training Setup Test")
    print("DPO 训练设置测试")
    print("=" * 80)
    print("\n")

    # Test 1: Create test data / 测试 1: 创建测试数据
    try:
        train_file, eval_file = create_test_data()
    except Exception as e:
        print(f"✗ Failed to create test data: {e}")
        print(f"✗ 创建测试数据失败: {e}")
        return

    # Test 2: Create test config / 测试 2: 创建测试配置
    try:
        config_file = create_test_config(train_file, eval_file)
    except Exception as e:
        print(f"✗ Failed to create test config: {e}")
        print(f"✗ 创建测试配置失败: {e}")
        return

    # Test 3: Test dataset loading / 测试 3: 测试数据集加载
    if not check_data_loading():
        print("\n✗ Dataset loading test failed")
        print("✗ 数据集加载测试失败")
        return

    # Test 4: Test model loading / 测试 4: 测试模型加载
    print("Do you want to test model loading? (downloads ~500MB) [y/N]: ", end='')
    response = input().strip().lower()
    if response in ['y', 'yes']:
        if not check_model_loading():
            print("\n✗ Model loading test failed")
            print("✗ 模型加载测试失败")
            return
    else:
        print("Skipping model loading test")
        print("跳过模型加载测试")

    # Summary / 摘要
    print("\n")
    print("=" * 80)
    print("Setup Test Complete!")
    print("设置测试完成!")
    print("=" * 80)
    print("\nYou can now run DPO training with:")
    print("您现在可以运行 DPO 训练:")
    print()
    print(f"  python scripts/train_dpo.py --config {config_file}")
    print()
    print("Note: The test uses a very small model (Qwen2.5-0.5B) for quick testing.")
    print("For production, use a larger model.")
    print("注意: 测试使用非常小的模型（Qwen2.5-0.5B）进行快速测试。")
    print("生产环境请使用更大的模型。")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
