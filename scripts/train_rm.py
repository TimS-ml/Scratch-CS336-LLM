"""
Reward Model Training Entry Point
奖励模型训练入口点

This script serves as the main entry point for training reward models.
It loads a plain YAML config file and provides a clean interface
for launching RM training jobs.

此脚本作为训练奖励模型的主入口点。
它加载普通 YAML 配置文件，并提供一个简洁的接口来启动 RM 训练任务。

Usage / 用法:
    # Train with default config / 使用默认配置训练
    python scripts/train_rm.py

    # Train with custom config / 使用自定义配置训练
    python scripts/train_rm.py --config configs/rm_training_custom.yaml

    # Override specific parameters / 覆盖特定参数
    python scripts/train_rm.py learning_rate=1e-5 num_epochs=3

    # Use different model / 使用不同模型
    python scripts/train_rm.py model_path=/path/to/model

Example / 示例:
    python scripts/train_rm.py \\
        model_path=models/sft_checkpoint \\
        data_path=data/rm_train/rm_data.jsonl \\
        output_dir=outputs/reward_model \\
        num_epochs=3 \\
        learning_rate=1e-5

Author: Your Name
Date: 2025-11-14
"""

import os
import sys
import logging
import torch
from omegaconf import DictConfig, OmegaConf

from scratch_cs336.training.rm import (
    ModelArguments,
    DataArguments,
    RMTrainingArguments,
    train_reward_model,
)
from scratch_cs336.utils import load_config

# Setup logger / 设置日志记录器
logger = logging.getLogger(__name__)


def validate_config(cfg: DictConfig) -> None:
    """
    Validate configuration parameters
    验证配置参数

    Checks that all required parameters are present and have valid values.
    检查所有必需的参数都存在且具有有效值。

    Args:
        cfg: config object / 配置对象

    Raises:
        ValueError: If configuration is invalid / 如果配置无效
    """
    # Check required paths exist / 检查必需的路径是否存在
    if not os.path.exists(cfg.model_path):
        raise ValueError(
            f"Model path does not exist: {cfg.model_path}\n"
            f"模型路径不存在: {cfg.model_path}"
        )

    if not os.path.exists(cfg.data_path):
        raise ValueError(
            f"Data path does not exist: {cfg.data_path}\n"
            f"数据路径不存在: {cfg.data_path}"
        )

    # Check numeric parameters are valid / 检查数值参数是否有效
    if cfg.learning_rate <= 0:
        raise ValueError(
            f"Learning rate must be positive, got: {cfg.learning_rate}\n"
            f"学习率必须为正数，得到: {cfg.learning_rate}"
        )

    if cfg.num_epochs <= 0:
        raise ValueError(
            f"Number of epochs must be positive, got: {cfg.num_epochs}\n"
            f"训练轮数必须为正数，得到: {cfg.num_epochs}"
        )

    if cfg.max_length <= 0:
        raise ValueError(
            f"Max length must be positive, got: {cfg.max_length}\n"
            f"最大长度必须为正数，得到: {cfg.max_length}"
        )

    # Validate loss type / 验证损失类型
    valid_loss_types = ["sigmoid", "hinge"]
    if cfg.get("loss_type", "sigmoid") not in valid_loss_types:
        raise ValueError(
            f"Invalid loss_type: {cfg.loss_type}. "
            f"Must be one of {valid_loss_types}\n"
            f"无效的 loss_type: {cfg.loss_type}. "
            f"必须是 {valid_loss_types} 之一"
        )

    logger.info("Configuration validation passed ✓")
    logger.info("配置验证通过 ✓")


def setup_output_directories(cfg: DictConfig) -> None:
    """
    Setup output directories for training
    设置训练输出目录

    Creates necessary directories for checkpoints, logs, and saved models.
    创建检查点、日志和保存模型所需的目录。

    Args:
        cfg: config object / 配置对象
    """
    # Create output directories / 创建输出目录
    os.makedirs(cfg.output_dir, exist_ok=True)

    if hasattr(cfg, 'checkpoint_dir'):
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoint directory: {cfg.checkpoint_dir}")

    logger.info(f"Output directory: {cfg.output_dir}")
    logger.info(f"输出目录: {cfg.output_dir}")


def log_training_info(cfg: DictConfig) -> None:
    """
    Log training configuration and environment information
    记录训练配置和环境信息

    Args:
        cfg: config object / 配置对象
    """
    logger.info("=" * 80)
    logger.info("Reward Model Training Configuration")
    logger.info("奖励模型训练配置")
    logger.info("=" * 80)

    # Log environment info / 记录环境信息
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")

    # Log key configuration / 记录关键配置
    logger.info("\nKey Configuration / 关键配置:")
    logger.info(f"  Model path / 模型路径: {cfg.model_path}")
    logger.info(f"  Data path / 数据路径: {cfg.data_path}")
    logger.info(f"  Output directory / 输出目录: {cfg.output_dir}")
    logger.info(f"  Max length / 最大长度: {cfg.max_length}")
    logger.info(f"  Learning rate / 学习率: {cfg.learning_rate}")
    logger.info(f"  Batch size / 批次大小: {cfg.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation / 梯度累积: {cfg.gradient_accumulation_steps}")
    logger.info(f"  Number of epochs / 训练轮数: {cfg.num_epochs}")
    logger.info(f"  Loss type / 损失类型: {cfg.get('loss_type', 'sigmoid')}")
    logger.info(f"  Margin / 边距: {cfg.get('margin', 0.0)}")
    logger.info("=" * 80 + "\n")


def main(cfg: DictConfig) -> None:
    """
    Main training function
    主训练函数

    This function:
        1. Validates the configuration
        2. Sets up output directories
        3. Converts config to training argument objects
        4. Launches the training process

    此函数：
        1. 验证配置
        2. 设置输出目录
        3. 将配置转换为训练参数对象
        4. 启动训练过程

    Args:
        cfg: config object loaded from YAML
            从 YAML 加载的配置对象
    """
    # Print full configuration for debugging / 打印完整配置以供调试
    logger.info("Full configuration / 完整配置:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Validate configuration / 验证配置
    try:
        validate_config(cfg)
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.error(f"配置验证失败: {e}")
        sys.exit(1)

    # Setup directories / 设置目录
    setup_output_directories(cfg)

    # Log training info / 记录训练信息
    log_training_info(cfg)

    # ========================================================================
    # Convert config to training arguments / 将配置转换为训练参数
    # ========================================================================

    # Model arguments / 模型参数
    model_args = ModelArguments(
        model_name_or_path=cfg.model_path,
        model_type=cfg.get("model_type", "auto"),
        trust_remote_code=cfg.get("trust_remote_code", True),
        use_cache=False,  # Always False for training
    )

    # Data arguments / 数据参数
    data_args = DataArguments(
        data_path=cfg.data_path,
        max_length=cfg.max_length,
        system_prompt=cfg.get("system_prompt", "你是由wdndev开发的个人助手。"),
        eval_split_ratio=cfg.get("eval_split_ratio", 0.01),
        preprocessing_num_workers=cfg.get("preprocessing_num_workers", None),
    )

    # Training arguments / 训练参数
    training_args = RMTrainingArguments(
        # Output / 输出
        output_dir=cfg.output_dir,
        overwrite_output_dir=cfg.get("overwrite_output_dir", True),

        # Training / 训练
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size",
                                          cfg.per_device_train_batch_size),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),

        # Optimizer / 优化器
        optim=cfg.get("optim", "adamw_torch"),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),

        # Evaluation / 评估
        evaluation_strategy=cfg.get("evaluation_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 500),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 500),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "accuracy"),
        greater_is_better=True,

        # Logging / 日志
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_strategy=cfg.get("logging_strategy", "steps"),
        logging_steps=cfg.get("logging_steps", 10),
        report_to=cfg.get("report_to", ["tensorboard"]),

        # Hardware / 硬件
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=cfg.get("dataloader_pin_memory", True),

        # Reproducibility / 可重复性
        seed=cfg.get("seed", 42),

        # RM-specific / RM 特定
        margin=cfg.get("margin", 0.0),
        loss_type=cfg.get("loss_type", "sigmoid"),

        # Checkpointing / 检查点
        resume_from_checkpoint=cfg.get("resume_from_checkpoint", None),
    )

    # ========================================================================
    # Start Training / 开始训练
    # ========================================================================

    try:
        logger.info("Starting reward model training...")
        logger.info("开始奖励模型训练...")

        train_reward_model(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
        )

        logger.info("Training completed successfully! ✓")
        logger.info("训练成功完成！ ✓")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        logger.error(f"训练失败，错误: {e}")
        raise


if __name__ == "__main__":
    main(load_config("configs/rm_training.yaml"))
