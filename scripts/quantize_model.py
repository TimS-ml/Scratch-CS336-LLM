#!/usr/bin/env python3
"""
Model Quantization Script / 模型量化脚本

This script provides a command-line interface for quantizing language models using GPTQ.
此脚本提供了使用 GPTQ 量化语言模型的命令行界面。

Usage / 用法:
    # Basic usage / 基本用法
    python scripts/quantize_model.py \\
        --model_path Qwen/Qwen2.5-0.5B \\
        --data_path data/calibration.jsonl \\
        --output_dir output/quantized_model \\
        --bits 4

    # Using config file / 使用配置文件
    python scripts/quantize_model.py --config configs/quantization.yaml

    # Advanced usage / 高级用法
    python scripts/quantize_model.py \\
        --model_path Qwen/Qwen2.5-1.5B \\
        --data_path data/calibration.jsonl \\
        --output_dir output/qwen2.5-1.5b-gptq-4bit \\
        --bits 4 \\
        --group_size 128 \\
        --num_gpus 2 \\
        --max_samples 256

Author: LLM-from-Scratch Project
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Add project root to path / 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clean_llm.quantize import (
    GPTQConfig,
    GPTQQuantizer,
    prepare_calibration_data,
)

# Configure logging / 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments / 解析命令行参数

    Returns:
        argparse.Namespace: Parsed arguments / 解析的参数
    """
    parser = argparse.ArgumentParser(
        description="Quantize language models using GPTQ / 使用 GPTQ 量化语言模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config file / 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides other arguments) / YAML 配置文件路径（覆盖其他参数）"
    )

    # Model parameters / 模型参数
    model_group = parser.add_argument_group("Model Parameters / 模型参数")
    model_group.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path or HuggingFace model ID / 路径或 HuggingFace 模型 ID"
    )
    model_group.add_argument(
        "--output_dir",
        type=str,
        required=False,
        help="Output directory for quantized model / 量化模型输出目录"
    )

    # Data parameters / 数据参数
    data_group = parser.add_argument_group("Data Parameters / 数据参数")
    data_group.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to calibration data (JSONL format) / 校准数据路径（JSONL 格式）"
    )
    data_group.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of calibration samples (None for all) / 最大校准样本数（None 表示全部）"
    )
    data_group.add_argument(
        "--input_field",
        type=str,
        default="input",
        help="Field name for input text in JSONL / JSONL 中输入文本的字段名"
    )
    data_group.add_argument(
        "--target_field",
        type=str,
        default="target",
        help="Field name for target text in JSONL / JSONL 中目标文本的字段名"
    )

    # Quantization parameters / 量化参数
    quant_group = parser.add_argument_group("Quantization Parameters / 量化参数")
    quant_group.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Number of bits for quantization (4 or 8) / 量化位数（4 或 8）"
    )
    quant_group.add_argument(
        "--group_size",
        type=int,
        default=128,
        help="Group size for quantization (32, 64, 128, -1) / 量化组大小"
    )
    quant_group.add_argument(
        "--damp_percent",
        type=float,
        default=0.1,
        help="Damping percentage for quantization / 量化阻尼百分比"
    )
    quant_group.add_argument(
        "--desc_act",
        action="store_true",
        help="Use descending activation order (slower but may improve quality) / 使用降序激活顺序"
    )
    quant_group.add_argument(
        "--no_sym",
        action="store_true",
        help="Use asymmetric quantization / 使用非对称量化"
    )
    quant_group.add_argument(
        "--max_input_length",
        type=int,
        default=8192,
        help="Maximum input sequence length / 最大输入序列长度"
    )
    quant_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for quantization / 量化批次大小"
    )

    # Hardware parameters / 硬件参数
    hw_group = parser.add_argument_group("Hardware Parameters / 硬件参数")
    hw_group.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use / 使用的 GPU 数量"
    )
    hw_group.add_argument(
        "--max_memory_per_gpu",
        type=int,
        default=20,
        help="Maximum memory per GPU in GB / 每个 GPU 的最大内存（GB）"
    )
    hw_group.add_argument(
        "--cache_examples_on_gpu",
        action="store_true",
        help="Cache calibration examples on GPU / 在 GPU 上缓存校准样本"
    )
    hw_group.add_argument(
        "--use_triton",
        action="store_true",
        help="Use Triton kernels (faster but requires Triton) / 使用 Triton 内核"
    )

    # Other parameters / 其他参数
    other_group = parser.add_argument_group("Other Parameters / 其他参数")
    other_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model / 加载模型时信任远程代码"
    )
    other_group.add_argument(
        "--use_safetensors",
        action="store_true",
        default=True,
        help="Save model in safetensors format / 以 safetensors 格式保存模型"
    )

    args = parser.parse_args()

    # Load config file if specified / 如果指定则加载配置文件
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        logger.info(f"从 {args.config} 加载配置")
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Override args with config values / 用配置值覆盖参数
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # Validate required arguments / 验证必需参数
    if not args.model_path:
        parser.error("--model_path is required / 需要 --model_path")
    if not args.data_path:
        parser.error("--data_path is required / 需要 --data_path")
    if not args.output_dir:
        parser.error("--output_dir is required / 需要 --output_dir")

    return args


def validate_environment() -> None:
    """
    Validate the environment and dependencies / 验证环境和依赖项

    Raises:
        ImportError: If required packages are not installed
    """
    try:
        import auto_gptq
        logger.info(f"auto-gptq version: {auto_gptq.__version__}")
    except ImportError:
        logger.error(
            "auto-gptq is not installed. Install it with: "
            "pip install auto-gptq"
        )
        raise

    try:
        import transformers
        logger.info(f"transformers version: {transformers.__version__}")
    except ImportError:
        logger.error(
            "transformers is not installed. Install it with: "
            "pip install transformers"
        )
        raise

    # Check CUDA availability / 检查 CUDA 可用性
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Quantization may be very slow. "
            "CUDA 不可用。量化可能会非常慢。"
        )
    else:
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"CUDA 可用：{torch.cuda.device_count()} 个 GPU")


def print_summary(args: argparse.Namespace, config: GPTQConfig) -> None:
    """
    Print summary of quantization configuration / 打印量化配置摘要

    Args:
        args: Command line arguments / 命令行参数
        config: GPTQ configuration / GPTQ 配置
    """
    logger.info("=" * 80)
    logger.info("Quantization Configuration Summary / 量化配置摘要")
    logger.info("=" * 80)
    logger.info(f"Model Path / 模型路径: {args.model_path}")
    logger.info(f"Data Path / 数据路径: {args.data_path}")
    logger.info(f"Output Directory / 输出目录: {args.output_dir}")
    logger.info("-" * 80)
    logger.info(f"Quantization Bits / 量化位数: {config.bits}")
    logger.info(f"Group Size / 组大小: {config.group_size}")
    logger.info(f"Damp Percent / 阻尼百分比: {config.damp_percent}")
    logger.info(f"Descending Activation / 降序激活: {config.desc_act}")
    logger.info(f"Symmetric Quantization / 对称量化: {config.sym}")
    logger.info("-" * 80)
    logger.info(f"Max Input Length / 最大输入长度: {config.max_input_length}")
    logger.info(f"Batch Size / 批次大小: {config.batch_size}")
    logger.info(f"Max Samples / 最大样本数: {args.max_samples or 'All / 全部'}")
    logger.info("-" * 80)
    logger.info(f"Number of GPUs / GPU 数量: {args.num_gpus}")
    logger.info(f"Max Memory per GPU / 每个 GPU 最大内存: {args.max_memory_per_gpu}GB")
    logger.info(f"Cache on GPU / 在 GPU 上缓存: {config.cache_examples_on_gpu}")
    logger.info(f"Use Triton / 使用 Triton: {config.use_triton}")
    logger.info("=" * 80)


def main():
    """
    Main function for model quantization / 模型量化主函数
    """
    try:
        # Parse arguments / 解析参数
        args = parse_args()

        # Validate environment / 验证环境
        logger.info("Validating environment / 验证环境")
        validate_environment()

        # Create GPTQ config / 创建 GPTQ 配置
        config = GPTQConfig(
            bits=args.bits,
            group_size=args.group_size,
            damp_percent=args.damp_percent,
            desc_act=args.desc_act,
            sym=not args.no_sym,
            max_input_length=args.max_input_length,
            batch_size=args.batch_size,
            cache_examples_on_gpu=args.cache_examples_on_gpu,
            use_triton=args.use_triton,
        )

        # Print summary / 打印摘要
        print_summary(args, config)

        # Initialize quantizer / 初始化量化器
        logger.info("Initializing quantizer / 初始化量化器")
        quantizer = GPTQQuantizer(
            model_path=args.model_path,
            config=config,
            num_gpus=args.num_gpus,
            max_memory_per_gpu=args.max_memory_per_gpu,
            trust_remote_code=args.trust_remote_code,
        )

        # Prepare calibration data / 准备校准数据
        logger.info("Preparing calibration data / 准备校准数据")
        calibration_data = prepare_calibration_data(
            data_path=args.data_path,
            tokenizer=quantizer.tokenizer,
            max_length=config.max_input_length,
            max_samples=args.max_samples,
            input_field=args.input_field,
            target_field=args.target_field,
        )

        if not calibration_data:
            logger.error("No calibration data available / 没有可用的校准数据")
            sys.exit(1)

        # Run quantization / 运行量化
        logger.info("Starting quantization process / 开始量化过程")
        quantizer.quantize(calibration_data)

        # Get model size info / 获取模型大小信息
        size_info = quantizer.get_model_size_info()
        logger.info(f"Model size info / 模型大小信息: {size_info}")

        # Save quantized model / 保存量化模型
        logger.info("Saving quantized model / 保存量化模型")
        quantizer.save(
            output_dir=args.output_dir,
            use_safetensors=args.use_safetensors,
        )

        logger.info("=" * 80)
        logger.info("Quantization completed successfully! / 量化成功完成！")
        logger.info(f"Quantized model saved to: {args.output_dir}")
        logger.info(f"量化模型已保存到: {args.output_dir}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\nQuantization interrupted by user / 用户中断量化")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Quantization failed / 量化失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
