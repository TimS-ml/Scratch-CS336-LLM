#!/usr/bin/env python3
"""
GPTQ Quantization Example / GPTQ 量化示例

This script demonstrates how to use the GPTQ quantization module.
此脚本演示如何使用 GPTQ 量化模块。

This is a complete end-to-end example showing:
这是一个完整的端到端示例，展示：
1. Creating a quantization configuration / 创建量化配置
2. Initializing the quantizer / 初始化量化器
3. Preparing calibration data / 准备校准数据
4. Running quantization / 运行量化
5. Saving the quantized model / 保存量化模型
6. Loading and using the quantized model / 加载和使用量化模型

Author: LLM-from-Scratch Project
License: MIT
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

# Configure logging / 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def create_sample_calibration_data(output_path: str, num_samples: int = 100) -> None:
    """
    Create sample calibration data for demonstration / 创建示例校准数据用于演示

    Args:
        output_path: Path to save calibration data / 保存校准数据的路径
        num_samples: Number of samples to generate / 生成的样本数量
    """
    logger.info(f"Creating {num_samples} sample calibration data points")
    logger.info(f"创建 {num_samples} 个示例校准数据点")

    # Sample questions and answers / 示例问题和答案
    sample_data = [
        {
            "input": "What is machine learning?",
            "target": "Machine learning is a subset of artificial intelligence that focuses on developing systems that can learn from and make decisions based on data."
        },
        {
            "input": "Explain neural networks.",
            "target": "Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process and transform data."
        },
        {
            "input": "What is deep learning?",
            "target": "Deep learning is a subset of machine learning that uses multi-layered neural networks to progressively extract higher-level features from raw input."
        },
        {
            "input": "How does gradient descent work?",
            "target": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters in the direction of steepest descent to minimize a loss function."
        },
        {
            "input": "What is the attention mechanism?",
            "target": "The attention mechanism allows models to focus on different parts of the input when producing each part of the output, improving performance on sequence tasks."
        },
    ]

    # Generate calibration data / 生成校准数据
    calibration_data = []
    for i in range(num_samples):
        # Cycle through sample data / 循环使用示例数据
        sample = sample_data[i % len(sample_data)]
        calibration_data.append(sample)

    # Save to JSONL file / 保存到 JSONL 文件
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in calibration_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    logger.info(f"Calibration data saved to {output_path}")
    logger.info(f"校准数据已保存到 {output_path}")


def example_quantization():
    """
    Example: Quantize a model using GPTQ / 示例：使用 GPTQ 量化模型

    This function demonstrates the complete quantization workflow.
    此函数演示完整的量化工作流。
    """
    from scratch_cs336.core.quantize import GPTQConfig, GPTQQuantizer, prepare_calibration_data

    logger.info("=" * 80)
    logger.info("GPTQ Quantization Example / GPTQ 量化示例")
    logger.info("=" * 80)

    # Step 1: Create quantization configuration / 步骤 1：创建量化配置
    logger.info("\nStep 1: Creating quantization configuration")
    logger.info("步骤 1：创建量化配置")

    config = GPTQConfig(
        bits=4,                      # 4-bit quantization / 4 位量化
        group_size=128,              # Group size / 组大小
        damp_percent=0.1,            # Damping percentage / 阻尼百分比
        desc_act=False,              # False for faster inference / False 以加快推理
        max_input_length=2048,       # Maximum sequence length / 最大序列长度
        batch_size=1,                # Batch size / 批次大小
        cache_examples_on_gpu=False, # Cache on GPU / GPU 缓存
        use_triton=False,            # Use Triton / 使用 Triton
    )

    logger.info(f"Configuration: {config.bits}-bit, group_size={config.group_size}")
    logger.info(f"配置：{config.bits} 位，组大小={config.group_size}")

    # Step 2: Initialize quantizer / 步骤 2：初始化量化器
    logger.info("\nStep 2: Initializing quantizer")
    logger.info("步骤 2：初始化量化器")

    # Note: Replace with your actual model path / 注意：替换为您的实际模型路径
    model_path = "Qwen/Qwen2.5-0.5B"  # Example model / 示例模型
    output_dir = "output/quantized_model_example"

    quantizer = GPTQQuantizer(
        model_path=model_path,
        config=config,
        num_gpus=1,
        max_memory_per_gpu=20,
        trust_remote_code=True,
    )

    logger.info(f"Quantizer initialized for model: {model_path}")
    logger.info(f"已为模型初始化量化器：{model_path}")

    # Step 3: Prepare calibration data / 步骤 3：准备校准数据
    logger.info("\nStep 3: Preparing calibration data")
    logger.info("步骤 3：准备校准数据")

    # Create sample calibration data / 创建示例校准数据
    calibration_file = "data/calibration_example.jsonl"
    create_sample_calibration_data(calibration_file, num_samples=128)

    calibration_data = prepare_calibration_data(
        data_path=calibration_file,
        tokenizer=quantizer.tokenizer,
        max_length=config.max_input_length,
        max_samples=128,  # Use 128 samples for demonstration / 使用 128 个样本演示
    )

    logger.info(f"Prepared {len(calibration_data)} calibration examples")
    logger.info(f"准备了 {len(calibration_data)} 个校准样本")

    # Step 4: Run quantization / 步骤 4：运行量化
    logger.info("\nStep 4: Running quantization (this may take a while...)")
    logger.info("步骤 4：运行量化（这可能需要一些时间...）")

    quantizer.quantize(calibration_data)

    logger.info("Quantization completed successfully!")
    logger.info("量化成功完成！")

    # Step 5: Save quantized model / 步骤 5：保存量化模型
    logger.info("\nStep 5: Saving quantized model")
    logger.info("步骤 5：保存量化模型")

    quantizer.save(
        output_dir=output_dir,
        use_safetensors=True,
        save_tokenizer=True,
    )

    logger.info(f"Quantized model saved to: {output_dir}")
    logger.info(f"量化模型已保存到：{output_dir}")

    # Get model size information / 获取模型大小信息
    size_info = quantizer.get_model_size_info()
    logger.info("\nModel Size Information / 模型大小信息:")
    for key, value in size_info.items():
        logger.info(f"  {key}: {value}")

    logger.info("=" * 80)
    logger.info("Quantization example completed! / 量化示例完成！")
    logger.info("=" * 80)


def example_load_and_inference():
    """
    Example: Load and use a quantized model / 示例：加载和使用量化模型

    This function demonstrates how to load and use a quantized model for inference.
    此函数演示如何加载和使用量化模型进行推理。
    """
    from scratch_cs336.core.quantize import load_quantized_model

    logger.info("=" * 80)
    logger.info("Load and Inference Example / 加载和推理示例")
    logger.info("=" * 80)

    # Load quantized model / 加载量化模型
    logger.info("Loading quantized model...")
    logger.info("加载量化模型...")

    model_path = "output/quantized_model_example"

    try:
        model, tokenizer = load_quantized_model(
            model_path=model_path,
            device_map="auto",
            trust_remote_code=True,
        )

        logger.info("Model loaded successfully!")
        logger.info("模型加载成功！")

        # Example inference / 示例推理
        logger.info("\nRunning inference example...")
        logger.info("运行推理示例...")

        prompt = "What is artificial intelligence?"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        logger.info(f"\nPrompt / 提示: {prompt}")
        logger.info(f"Response / 响应: {response}")

        logger.info("=" * 80)
        logger.info("Inference example completed! / 推理示例完成！")
        logger.info("=" * 80)

    except FileNotFoundError:
        logger.error(f"Quantized model not found at: {model_path}")
        logger.error(f"未找到量化模型：{model_path}")
        logger.info("Please run example_quantization() first to create the model.")
        logger.info("请先运行 example_quantization() 以创建模型。")


def example_batch_inference():
    """
    Example: Batch inference with quantized model / 示例：使用量化模型批量推理

    This function demonstrates batch inference with a quantized model.
    此函数演示使用量化模型进行批量推理。
    """
    from scratch_cs336.core.quantize import load_quantized_model

    logger.info("=" * 80)
    logger.info("Batch Inference Example / 批量推理示例")
    logger.info("=" * 80)

    model_path = "output/quantized_model_example"

    try:
        model, tokenizer = load_quantized_model(model_path)

        # Multiple prompts / 多个提示
        prompts = [
            "What is machine learning?",
            "Explain deep learning.",
            "What is natural language processing?",
        ]

        logger.info(f"Processing {len(prompts)} prompts in batch...")
        logger.info(f"批量处理 {len(prompts)} 个提示...")

        # Batch tokenization / 批量分词
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Batch generation / 批量生成
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9,
        )

        # Decode responses / 解码响应
        logger.info("\nBatch Results / 批量结果:")
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            response = tokenizer.decode(output, skip_special_tokens=True)
            logger.info(f"\n{i+1}. Prompt / 提示: {prompt}")
            logger.info(f"   Response / 响应: {response}")

        logger.info("=" * 80)

    except FileNotFoundError:
        logger.error(f"Quantized model not found at: {model_path}")
        logger.error(f"未找到量化模型：{model_path}")


def main():
    """
    Main function / 主函数

    Run different examples based on user input.
    根据用户输入运行不同的示例。
    """
    print("\nGPTQ Quantization Examples / GPTQ 量化示例")
    print("=" * 80)
    print("\nAvailable examples / 可用示例:")
    print("1. Full quantization workflow / 完整量化工作流")
    print("2. Load and inference / 加载和推理")
    print("3. Batch inference / 批量推理")
    print("4. Create sample calibration data / 创建示例校准数据")
    print("5. Run all examples / 运行所有示例")
    print("\nNote: Examples 2-3 require running example 1 first.")
    print("注意：示例 2-3 需要先运行示例 1。")
    print("=" * 80)

    choice = input("\nSelect example (1-5) / 选择示例 (1-5): ").strip()

    if choice == "1":
        example_quantization()
    elif choice == "2":
        example_load_and_inference()
    elif choice == "3":
        example_batch_inference()
    elif choice == "4":
        create_sample_calibration_data("data/calibration_example.jsonl", num_samples=128)
        logger.info("Sample calibration data created!")
        logger.info("示例校准数据已创建！")
    elif choice == "5":
        logger.info("Running all examples...")
        logger.info("运行所有示例...")
        example_quantization()
        example_load_and_inference()
        example_batch_inference()
    else:
        logger.error("Invalid choice. Please select 1-5.")
        logger.error("无效选择。请选择 1-5。")


if __name__ == "__main__":
    main()
