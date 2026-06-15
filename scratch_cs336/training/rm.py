"""
Reward Model Training Module
奖励模型训练模块

This module provides a complete implementation for training reward models (RM) in the RLHF pipeline.
Reward models learn to predict human preferences by comparing pairs of responses to the same prompt.

本模块提供 RLHF 流程中训练奖励模型（RM）的完整实现。
奖励模型通过比较同一提示的两个回答来学习预测人类偏好。

The training follows the InstructGPT approach:
    - Uses pairwise ranking loss (Bradley-Terry model)
    - Trains a model to output a scalar reward score
    - Optimizes for higher scores on preferred responses

训练遵循 InstructGPT 方法：
    - 使用成对排序损失（Bradley-Terry 模型）
    - 训练模型输出标量奖励分数
    - 优化以在偏好回答上获得更高分数

References:
    - InstructGPT Paper: https://arxiv.org/abs/2203.02155
    - Training language models to follow instructions with human feedback

Author: Your Name
Date: 2025-11-14
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score

import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from torch.utils.data import DataLoader, random_split

# Import the RMDataset from rlhf_datasets
# 从 rlhf_datasets 导入 RMDataset
from scratch_cs336.training.datasets import RMDataset

# Setup logger / 设置日志记录器
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Arguments / 参数数据类
# ============================================================================

@dataclass
class ModelArguments:
    """
    Arguments for model configuration
    模型配置参数

    These arguments define the model architecture and loading behavior.
    Support both custom TinyLLM models and standard HuggingFace models.

    这些参数定义模型架构和加载行为。
    支持自定义 TinyLLM 模型和标准 HuggingFace 模型。
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models\n"
                    "预训练模型路径或 huggingface.co/models 的模型标识符"
        }
    )

    model_type: str = field(
        default="auto",
        metadata={
            "help": "Model type: 'auto', 'tinyllm', or specific HF model type\n"
                    "模型类型：'auto'、'tinyllm' 或特定的 HF 模型类型"
        }
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whether to trust remote code when loading models\n"
                    "加载模型时是否信任远程代码"
        }
    )

    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether to use KV cache (should be False for training)\n"
                    "是否使用 KV 缓存（训练时应为 False）"
        }
    )


@dataclass
class DataArguments:
    """
    Arguments for data loading and preprocessing
    数据加载和预处理参数

    These arguments control how the preference data is loaded and processed.

    这些参数控制偏好数据的加载和处理方式。
    """

    data_path: str = field(
        metadata={
            "help": "Path to the training data (JSONL format with prompt/chosen/rejected)\n"
                    "训练数据路径（包含 prompt/chosen/rejected 的 JSONL 格式）"
        }
    )

    max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length for tokenization\n"
                    "分词的最大序列长度"
        }
    )

    system_prompt: str = field(
        default="你是由wdndev开发的个人助手。",
        metadata={
            "help": "System prompt for the conversation\n"
                    "对话的系统提示"
        }
    )

    eval_split_ratio: float = field(
        default=0.01,
        metadata={
            "help": "Ratio of data to use for evaluation (e.g., 0.01 = 1%)\n"
                    "用于评估的数据比例（例如，0.01 = 1%）"
        }
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of workers for data preprocessing\n"
                    "数据预处理的工作进程数"
        }
    )


@dataclass
class RMTrainingArguments(TrainingArguments):
    """
    Extended training arguments for Reward Model training
    奖励模型训练的扩展训练参数

    Inherits from HuggingFace TrainingArguments and adds RM-specific options.

    继承自 HuggingFace TrainingArguments 并添加 RM 特定选项。
    """

    margin: float = field(
        default=0.0,
        metadata={
            "help": "Margin for ranking loss (if using margin-based loss)\n"
                    "排序损失的边距（如果使用基于边距的损失）"
        }
    )

    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "Type of ranking loss: 'sigmoid' (InstructGPT style) or 'hinge'\n"
                    "排序损失类型：'sigmoid'（InstructGPT 风格）或 'hinge'"
        }
    )


# ============================================================================
# Custom Reward Model Trainer / 自定义奖励模型训练器
# ============================================================================

class RMTrainer(Trainer):
    """
    Custom Trainer for Reward Model training
    自定义奖励模型训练器

    This trainer implements the pairwise ranking loss used in InstructGPT.
    For each training example, we have:
        - A prompt
        - A chosen response (preferred by human)
        - A rejected response (not preferred)

    此训练器实现 InstructGPT 中使用的成对排序损失。
    对于每个训练样本，我们有：
        - 一个提示
        - 一个选择的回答（人类偏好）
        - 一个拒绝的回答（不偏好）

    The model learns to assign higher scores to chosen responses.
    模型学习为选择的回答分配更高的分数。

    Loss Function / 损失函数:
        InstructGPT uses: loss = -log(sigmoid(reward_chosen - reward_rejected))
        This is equivalent to cross-entropy loss for binary classification.

        InstructGPT 使用：loss = -log(sigmoid(reward_chosen - reward_rejected))
        这等价于二分类的交叉熵损失。

    Args:
        margin (float): Optional margin for the loss (default 0.0)
                       损失的可选边距（默认 0.0）
        loss_type (str): Type of loss ('sigmoid' or 'hinge')
                        损失类型（'sigmoid' 或 'hinge'）
    """

    def __init__(
        self,
        margin: float = 0.0,
        loss_type: str = "sigmoid",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.margin = margin
        self.loss_type = loss_type

        logger.info(f"Initialized RMTrainer with loss_type={loss_type}, margin={margin}")
        logger.info(f"使用 loss_type={loss_type}, margin={margin} 初始化 RMTrainer")

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the pairwise ranking loss for reward model training
        计算奖励模型训练的成对排序损失

        This implements the InstructGPT pairwise ranking loss:
            loss = -log(sigmoid(reward_j - reward_k))

        where:
            - reward_j is the score for the chosen/preferred response
            - reward_k is the score for the rejected response

        实现 InstructGPT 成对排序损失：
            loss = -log(sigmoid(reward_j - reward_k))

        其中：
            - reward_j 是选择/偏好回答的分数
            - reward_k 是拒绝回答的分数

        Args:
            model: The reward model (outputs scalar scores)
                  奖励模型（输出标量分数）
            inputs: Dictionary containing:
                   包含以下内容的字典：
                - input_ids_j: Tokenized chosen response
                              分词后的选择回答
                - attention_mask_j: Attention mask for chosen response
                                   选择回答的注意力掩码
                - input_ids_k: Tokenized rejected response
                              分词后的拒绝回答
                - attention_mask_k: Attention mask for rejected response
                                   拒绝回答的注意力掩码
            return_outputs: Whether to return model outputs along with loss
                          是否返回模型输出和损失

        Returns:
            loss: Scalar tensor representing the ranking loss
                 表示排序损失的标量张量
            outputs (optional): Dictionary with reward scores if return_outputs=True
                              如果 return_outputs=True，则包含奖励分数的字典
        """
        # Forward pass for chosen response / 选择回答的前向传播
        # Model outputs logits of shape (batch_size, num_labels=1)
        # 模型输出形状为 (batch_size, num_labels=1) 的 logits
        outputs_j = model(
            input_ids=inputs["input_ids_j"],
            attention_mask=inputs["attention_mask_j"]
        )
        rewards_j = outputs_j.logits.squeeze(-1)  # Shape: (batch_size,)

        # Forward pass for rejected response / 拒绝回答的前向传播
        outputs_k = model(
            input_ids=inputs["input_ids_k"],
            attention_mask=inputs["attention_mask_k"]
        )
        rewards_k = outputs_k.logits.squeeze(-1)  # Shape: (batch_size,)

        # Compute pairwise ranking loss / 计算成对排序损失
        if self.loss_type == "sigmoid":
            # InstructGPT style: -log(sigmoid(r_chosen - r_rejected))
            # Equivalent to: cross_entropy([r_chosen, r_rejected], target=0)
            # InstructGPT 风格：-log(sigmoid(r_chosen - r_rejected))
            # 等价于：cross_entropy([r_chosen, r_rejected], target=0)
            loss = -nn.functional.logsigmoid(rewards_j - rewards_k - self.margin).mean()

        elif self.loss_type == "hinge":
            # Hinge loss: max(0, margin - (r_chosen - r_rejected))
            # Forces r_chosen to be at least margin higher than r_rejected
            # Hinge 损失：max(0, margin - (r_chosen - r_rejected))
            # 强制 r_chosen 至少比 r_rejected 高 margin
            loss = nn.functional.relu(self.margin - (rewards_j - rewards_k)).mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Return outputs if requested / 如果需要则返回输出
        if return_outputs:
            return loss, {
                "rewards_chosen": rewards_j,
                "rewards_rejected": rewards_k,
            }

        return loss


# ============================================================================
# Metrics and Evaluation / 指标和评估
# ============================================================================

def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for reward model
    计算奖励模型的评估指标

    The key metric is accuracy: how often does the model assign a higher
    score to the chosen response compared to the rejected response?

    关键指标是准确率：模型为选择的回答分配高于拒绝回答的分数的频率是多少？

    Args:
        eval_preds: EvalPrediction object containing:
                   包含以下内容的 EvalPrediction 对象：
            - predictions: Array of shape (batch_size, 2) where:
                          形状为 (batch_size, 2) 的数组，其中：
                predictions[:, 0] = rewards for chosen responses
                predictions[:, 1] = rewards for rejected responses
            - label_ids: Not used (we don't have explicit labels)
                        未使用（我们没有显式标签）

    Returns:
        Dictionary with metrics / 包含指标的字典:
            - accuracy: Percentage of times reward_chosen > reward_rejected
                       reward_chosen > reward_rejected 的百分比
            - margin_mean: Average margin (reward_chosen - reward_rejected)
                          平均边距（reward_chosen - reward_rejected）
            - margin_std: Standard deviation of margins
                         边距的标准差
    """
    predictions = eval_preds.predictions

    # predictions shape: (num_samples, 2)
    # predictions[:, 0] = rewards_chosen
    # predictions[:, 1] = rewards_rejected

    # For accuracy, check if reward_chosen > reward_rejected
    # 对于准确率，检查 reward_chosen > reward_rejected
    # We use argmax along axis=1: returns 0 if first value is larger
    # 我们沿 axis=1 使用 argmax：如果第一个值较大则返回 0
    preds = np.argmax(predictions, axis=1)  # 0 if chosen > rejected

    # Ground truth: chosen should always be better (label = 0)
    # 真实标签：选择的应该总是更好（标签 = 0）
    labels = np.zeros(preds.shape, dtype=int)

    # Calculate accuracy / 计算准确率
    accuracy = accuracy_score(labels, preds)

    # Calculate margin statistics / 计算边距统计
    margins = predictions[:, 0] - predictions[:, 1]  # reward_chosen - reward_rejected
    margin_mean = float(np.mean(margins))
    margin_std = float(np.std(margins))

    return {
        "accuracy": float(accuracy),
        "margin_mean": margin_mean,
        "margin_std": margin_std,
    }


# ============================================================================
# Main Training Function / 主训练函数
# ============================================================================

def train_reward_model(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: RMTrainingArguments,
) -> None:
    """
    Main function to train a reward model
    训练奖励模型的主函数

    This function orchestrates the entire reward model training pipeline:
        1. Setup logging and random seeds
        2. Load tokenizer and model
        3. Load and split dataset
        4. Initialize custom RMTrainer
        5. Train and evaluate
        6. Save final model

    此函数协调整个奖励模型训练流程：
        1. 设置日志和随机种子
        2. 加载分词器和模型
        3. 加载和拆分数据集
        4. 初始化自定义 RMTrainer
        5. 训练和评估
        6. 保存最终模型

    Args:
        model_args: Model configuration arguments / 模型配置参数
        data_args: Data loading arguments / 数据加载参数
        training_args: Training configuration arguments / 训练配置参数
    """

    # ========================================================================
    # 1. Setup Logging / 设置日志
    # ========================================================================

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set log level for transformers / 设置 transformers 的日志级别
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log basic info / 记录基本信息
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"fp16 training: {training_args.fp16}, "
        f"bf16 training: {training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters: {training_args}")

    # ========================================================================
    # 2. Set Random Seed / 设置随机种子
    # ========================================================================

    set_seed(training_args.seed)
    logger.info(f"Set random seed to {training_args.seed}")
    logger.info(f"设置随机种子为 {training_args.seed}")

    # ========================================================================
    # 3. Load Tokenizer / 加载分词器
    # ========================================================================

    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}")
    logger.info(f"从 {model_args.model_name_or_path} 加载分词器")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Ensure tokenizer has pad token / 确保分词器有填充 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
        logger.info("将 pad_token 设置为 eos_token")

    # ========================================================================
    # 4. Load Model / 加载模型
    # ========================================================================

    logger.info(f"Loading model from {model_args.model_name_or_path}")
    logger.info(f"从 {model_args.model_name_or_path} 加载模型")

    # Load model configuration / 加载模型配置
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Configure for reward modeling / 配置奖励建模
    config.use_cache = model_args.use_cache
    config.num_labels = 1  # Output a single scalar reward
    config.pad_token_id = tokenizer.pad_token_id

    # Load model for sequence classification (outputs scalar score)
    # 加载用于序列分类的模型（输出标量分数）
    if model_args.model_type == "tinyllm":
        # Import TinyLLM specific model if needed
        # 如果需要，导入 TinyLLM 特定模型
        try:
            from tiny_llm_zh.modeling_tinyllm import TinyllmForSequenceClassification
            logger.info("Using TinyllmForSequenceClassification")
            logger.info("使用 TinyllmForSequenceClassification")

            model = TinyllmForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                trust_remote_code=model_args.trust_remote_code,
            )
        except ImportError:
            logger.warning("Could not import TinyllmForSequenceClassification, "
                         "falling back to AutoModelForSequenceClassification")
            logger.warning("无法导入 TinyllmForSequenceClassification，"
                         "回退到 AutoModelForSequenceClassification")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                trust_remote_code=model_args.trust_remote_code,
            )
    else:
        # Use standard HuggingFace model / 使用标准 HuggingFace 模型
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            trust_remote_code=model_args.trust_remote_code,
        )

    # Log model statistics / 记录模型统计信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    logger.info(f"总参数: {total_params:,} ({total_params/1e6:.2f}M)")
    logger.info(f"可训练参数: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    # ========================================================================
    # 5. Load Dataset / 加载数据集
    # ========================================================================

    logger.info(f"Loading dataset from {data_args.data_path}")
    logger.info(f"从 {data_args.data_path} 加载数据集")

    # Create full dataset / 创建完整数据集
    full_dataset = RMDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        max_length=data_args.max_length,
        system=data_args.system_prompt,
    )

    # Split into train and eval / 拆分为训练和评估集
    total_len = len(full_dataset)
    eval_size = max(1, int(data_args.eval_split_ratio * total_len))
    train_size = total_len - eval_size

    logger.info(f"Total dataset size: {total_len}")
    logger.info(f"Train size: {train_size}, Eval size: {eval_size}")
    logger.info(f"总数据集大小: {total_len}")
    logger.info(f"训练集大小: {train_size}，评估集大小: {eval_size}")

    train_dataset, eval_dataset = random_split(
        full_dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(training_args.seed)
    )

    # ========================================================================
    # 6. Initialize Trainer / 初始化训练器
    # ========================================================================

    logger.info("Initializing RMTrainer")
    logger.info("初始化 RMTrainer")

    trainer = RMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        margin=training_args.margin,
        loss_type=training_args.loss_type,
    )

    # ========================================================================
    # 7. Train Model / 训练模型
    # ========================================================================

    logger.info("Starting training...")
    logger.info("开始训练...")

    # Check if resuming from checkpoint / 检查是否从检查点恢复
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {checkpoint}")
        logger.info(f"从检查点恢复: {checkpoint}")

    # Train / 训练
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Log training metrics / 记录训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ========================================================================
    # 8. Evaluate Model / 评估模型
    # ========================================================================

    if eval_dataset is not None:
        logger.info("Running final evaluation...")
        logger.info("运行最终评估...")

        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # ========================================================================
    # 9. Save Model / 保存模型
    # ========================================================================

    logger.info("Saving final model...")
    logger.info("保存最终模型...")

    # Save model state dict / 保存模型状态字典
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save as PyTorch state dict / 保存为 PyTorch 状态字典
    state_dict_path = os.path.join(output_dir, "last_model.pth")
    torch.save(model.state_dict(), state_dict_path)
    logger.info(f"Saved state dict to {state_dict_path}")

    # Save as HuggingFace format / 保存为 HuggingFace 格式
    final_model_dir = os.path.join(output_dir, "final_reward_model")
    os.makedirs(final_model_dir, exist_ok=True)

    model.save_pretrained(final_model_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_model_dir)

    logger.info(f"Saved final model to {final_model_dir}")
    logger.info(f"最终模型已保存到 {final_model_dir}")

    logger.info("Training completed successfully!")
    logger.info("训练成功完成！")


# ============================================================================
# Command-line Interface / 命令行接口
# ============================================================================

def main():
    """
    Main entry point for command-line training
    命令行训练的主入口点

    Parses arguments and starts training.
    解析参数并开始训练。
    """
    # Parse arguments / 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments, RMTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load from JSON file / 从 JSON 文件加载
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        # Parse from command line / 从命令行解析
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Start training / 开始训练
    train_reward_model(model_args, data_args, training_args)


if __name__ == "__main__":
    main()
