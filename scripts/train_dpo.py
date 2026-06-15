#!/usr/bin/env python3
"""
DPO Training Entry Point
DPO 训练入口点

This script serves as the main entry point for DPO (Direct Preference Optimization) training.
It handles argument parsing and delegates to the main training module.

此脚本作为 DPO（直接偏好优化）训练的主入口点。
它处理参数解析并委托给主训练模块。

Usage / 用法:
    # Using YAML config / 使用 YAML 配置:
    python scripts/train_dpo.py --config configs/dpo_training.yaml

    # Using command-line arguments / 使用命令行参数:
    python scripts/train_dpo.py \
        --model_name_or_path /path/to/model \
        --train_dataset_path /path/to/train.jsonl \
        --output_dir ./output \
        --num_train_epochs 3

    # Combining both (CLI overrides YAML) / 结合两者（CLI 覆盖 YAML）:
    python scripts/train_dpo.py \
        --config configs/dpo_training.yaml \
        --num_train_epochs 5 \
        --learning_rate 1e-5

    # Resume from checkpoint / 从检查点恢复:
    python scripts/train_dpo.py \
        --config configs/dpo_training.yaml \
        --resume_from_checkpoint ./output/checkpoint-1000

Author: wdndev
Date: 2025-11-14
"""

import os
import sys
import argparse
from pathlib import Path


from scratch_cs336.training.dpo import main


def parse_arguments():
    """
    Parse command-line arguments
    解析命令行参数

    Returns:
        Parsed arguments / 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="Train a language model using DPO (Direct Preference Optimization)\n"
                   "使用 DPO（直接偏好优化）训练语言模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:

  1. Train with YAML config / 使用 YAML 配置训练:
     python scripts/train_dpo.py --config configs/dpo_training.yaml

  2. Train with command-line args / 使用命令行参数训练:
     python scripts/train_dpo.py \\
         --model_name_or_path meta-llama/Llama-2-7b-hf \\
         --train_dataset_path data/dpo_train.jsonl \\
         --output_dir ./outputs/dpo \\
         --num_train_epochs 3 \\
         --per_device_train_batch_size 4 \\
         --learning_rate 5e-5

  3. Train with LoRA / 使用 LoRA 训练:
     python scripts/train_dpo.py \\
         --config configs/dpo_training.yaml \\
         --use_lora \\
         --lora_r 8 \\
         --lora_alpha 16

  4. Resume from checkpoint / 从检查点恢复:
     python scripts/train_dpo.py \\
         --config configs/dpo_training.yaml \\
         --resume_from_checkpoint ./outputs/dpo/checkpoint-500

For full list of arguments, see the configuration dataclasses in scratch_cs336/training/dpo.py
有关完整参数列表，请参见 scratch_cs336/training/dpo.py 中的配置数据类
        """
    )

    # Main arguments / 主要参数
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file / YAML 配置文件路径"
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from / 要恢复训练的检查点路径"
    )

    # Parse known args, pass rest to main training script
    # 解析已知参数，将其余参数传递给主训练脚本
    args, remaining_args = parser.parse_known_args()

    return args, remaining_args


def main_entry():
    """
    Main entry point
    主入口点
    """
    # Parse arguments / 解析参数
    args, remaining_args = parse_arguments()

    # Print header / 打印标题
    print("=" * 80)
    print("DPO Training Script")
    print("DPO 训练脚本")
    print("=" * 80)
    print()

    # Validate config file if provided / 如果提供了配置文件，验证其存在性
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Configuration file not found: {args.config}")
            print(f"错误: 找不到配置文件: {args.config}")
            sys.exit(1)
        print(f"Using configuration file: {args.config}")
        print(f"使用配置文件: {args.config}")
    else:
        print("No configuration file provided, using command-line arguments only")
        print("未提供配置文件，仅使用命令行参数")

    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            print(f"Warning: Checkpoint directory not found: {args.resume_from_checkpoint}")
            print(f"警告: 找不到检查点目录: {args.resume_from_checkpoint}")
        else:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
            print(f"从检查点恢复: {args.resume_from_checkpoint}")

    print()

    # Call main training function / 调用主训练函数
    try:
        main(
            config_path=args.config,
            cli_args=remaining_args if remaining_args else None,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("Training interrupted by user")
        print("训练被用户中断")
        print("=" * 80)
        sys.exit(0)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Training failed with error: {e}")
        print(f"训练失败，错误: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main_entry()
