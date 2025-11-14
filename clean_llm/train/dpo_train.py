"""
DPO Training Module
DPO 训练模块

This module implements Direct Preference Optimization (DPO) training for language models.
DPO is a reinforcement learning approach that directly optimizes a policy using preference data
without requiring a separate reward model or value function.

本模块实现了用于语言模型的直接偏好优化（DPO）训练。
DPO 是一种强化学习方法，直接使用偏好数据优化策略，
无需单独的奖励模型或价值函数。

Key Features / 主要特性:
    - Modern implementation using HuggingFace TRL DPOTrainer
      使用 HuggingFace TRL DPOTrainer 的现代实现
    - Support for LoRA (Low-Rank Adaptation) fine-tuning
      支持 LoRA（低秩适应）微调
    - Configurable via YAML files or command-line arguments
      可通过 YAML 文件或命令行参数配置
    - Comprehensive evaluation and metrics tracking
      全面的评估和指标跟踪
    - Production-ready error handling and logging
      生产就绪的错误处理和日志记录

References / 参考资料:
    - DPO Paper: https://arxiv.org/abs/2305.18290
    - TRL Documentation: https://huggingface.co/docs/trl/main/en/dpo_trainer
    - Example: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py

Author: wdndev
Date: 2025-11-14
"""

import os
import sys
import yaml
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Union, Any, List
from pathlib import Path

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig
from datasets import Dataset

# Add parent directory to path for imports / 将父目录添加到路径以导入模块
sys.path.append(str(Path(__file__).parent.parent))

from train.rlhf_datasets import load_dpo_dataset


# ============================================================================
# Logging Configuration / 日志配置
# ============================================================================

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Dataclasses / 配置数据类
# ============================================================================

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/tokenizer we are going to fine-tune.
    与我们要微调的模型/分词器相关的参数。
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
                   " / 预训练模型路径或来自 huggingface.co/models 的模型标识符"
        }
    )

    base_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the base model (if using LoRA, this is the original model)"
                   " / 基础模型路径（如果使用 LoRA，这是原始模型）"
        }
    )

    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if different from model"
                   " / 预训练分词器名称或路径（如果与模型不同）"
        }
    )

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust remote code when loading models"
                   " / 加载模型时是否信任远程代码"
        }
    )

    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether to use model cache (should be False for training)"
                   " / 是否使用模型缓存（训练时应为 False）"
        }
    )

    torch_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Override the default torch.dtype (e.g., 'float16', 'bfloat16', 'float32')"
                   " / 覆盖默认的 torch.dtype（例如 'float16'、'bfloat16'、'float32'）"
        }
    )

    device_map: Optional[str] = field(
        default=None,
        metadata={
            "help": "Device map for model parallelism (e.g., 'auto', 'balanced')"
                   " / 模型并行的设备映射（例如 'auto'、'balanced'）"
        }
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    与我们将为模型训练和评估输入的数据相关的参数。
    """

    train_dataset_path: str = field(
        metadata={
            "help": "Path to training dataset (JSONL format with prompt/chosen/rejected)"
                   " / 训练数据集路径（JSONL 格式，包含 prompt/chosen/rejected）"
        }
    )

    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to evaluation dataset (JSONL format with prompt/chosen/rejected)"
                   " / 评估数据集路径（JSONL 格式，包含 prompt/chosen/rejected）"
        }
    )

    max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum total sequence length (prompt + response)"
                   " / 最大总序列长度（提示 + 回答）"
        }
    )

    max_prompt_length: int = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt"
                   " / 提示的最大长度"
        }
    )

    sanity_check: bool = field(
        default=False,
        metadata={
            "help": "If True, only use 1000 samples for quick testing"
                   " / 如果为 True，仅使用 1000 个样本进行快速测试"
        }
    )

    num_proc: int = field(
        default=4,
        metadata={
            "help": "Number of processes for parallel data preprocessing"
                   " / 并行数据预处理的进程数"
        }
    )

    system_prompt: str = field(
        default="你是由wdndev开发的个人助手。",
        metadata={
            "help": "System prompt for conversations"
                   " / 对话的系统提示"
        }
    )


@dataclass
class DPOTrainingArguments:
    """
    Arguments specific to DPO training.
    DPO 训练特定的参数。
    """

    beta: float = field(
        default=0.1,
        metadata={
            "help": "The beta parameter for DPO loss (controls strength of KL penalty)"
                   " / DPO 损失的 beta 参数（控制 KL 惩罚的强度）"
        }
    )

    loss_type: str = field(
        default="sigmoid",
        metadata={
            "help": "Type of DPO loss ('sigmoid', 'hinge', 'ipo', 'kto')"
                   " / DPO 损失类型（'sigmoid'、'hinge'、'ipo'、'kto'）"
        }
    )

    label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Label smoothing factor for DPO loss"
                   " / DPO 损失的标签平滑因子"
        }
    )

    generate_during_eval: bool = field(
        default=False,
        metadata={
            "help": "Whether to generate samples during evaluation"
                   " / 是否在评估期间生成样本"
        }
    )


@dataclass
class LoRAArguments:
    """
    Arguments for LoRA (Low-Rank Adaptation) configuration.
    LoRA（低秩适应）配置的参数。
    """

    use_lora: bool = field(
        default=False,
        metadata={
            "help": "Whether to use LoRA for parameter-efficient fine-tuning"
                   " / 是否使用 LoRA 进行参数高效微调"
        }
    )

    lora_r: int = field(
        default=8,
        metadata={
            "help": "LoRA attention dimension (rank)"
                   " / LoRA 注意力维度（秩）"
        }
    )

    lora_alpha: float = field(
        default=16,
        metadata={
            "help": "LoRA alpha parameter (scaling factor)"
                   " / LoRA alpha 参数（缩放因子）"
        }
    )

    lora_dropout: float = field(
        default=0.05,
        metadata={
            "help": "LoRA dropout probability"
                   " / LoRA dropout 概率"
        }
    )

    lora_target_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of module names to apply LoRA to (e.g., ['q_proj', 'v_proj'])"
                   " / 要应用 LoRA 的模块名称列表（例如 ['q_proj', 'v_proj']）"
        }
    )

    lora_bias: str = field(
        default="none",
        metadata={
            "help": "Bias type for LoRA ('none', 'all', 'lora_only')"
                   " / LoRA 的偏置类型（'none'、'all'、'lora_only'）"
        }
    )


# ============================================================================
# DPO Trainer Class / DPO 训练器类
# ============================================================================

class DPOTrainingPipeline:
    """
    Complete DPO training pipeline
    完整的 DPO 训练流程

    This class orchestrates the entire DPO training process including:
    - Model and tokenizer loading
    - Dataset preparation
    - Training configuration
    - Training execution
    - Model saving

    此类编排整个 DPO 训练过程，包括：
    - 模型和分词器加载
    - 数据集准备
    - 训练配置
    - 训练执行
    - 模型保存

    Attributes:
        model_args: Model configuration arguments / 模型配置参数
        data_args: Data configuration arguments / 数据配置参数
        training_args: Training configuration arguments / 训练配置参数
        dpo_args: DPO-specific arguments / DPO 特定参数
        lora_args: LoRA configuration arguments / LoRA 配置参数
    """

    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        dpo_args: DPOTrainingArguments,
        lora_args: LoRAArguments,
    ):
        """
        Initialize the DPO training pipeline
        初始化 DPO 训练流程

        Args:
            model_args: Model configuration / 模型配置
            data_args: Data configuration / 数据配置
            training_args: Training configuration / 训练配置
            dpo_args: DPO-specific configuration / DPO 特定配置
            lora_args: LoRA configuration / LoRA 配置
        """
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.dpo_args = dpo_args
        self.lora_args = lora_args

        # Set random seed for reproducibility / 设置随机种子以确保可重现性
        if training_args.seed is not None:
            set_seed(training_args.seed)
            logger.info(f"Random seed set to {training_args.seed}")
            logger.info(f"随机种子设置为 {training_args.seed}")

    def load_tokenizer(self) -> transformers.PreTrainedTokenizer:
        """
        Load and configure tokenizer
        加载并配置分词器

        Returns:
            Configured tokenizer / 配置好的分词器
        """
        logger.info("Loading tokenizer...")
        logger.info("加载分词器...")

        tokenizer_path = (
            self.model_args.tokenizer_name_or_path
            if self.model_args.tokenizer_name_or_path
            else self.model_args.model_name_or_path
        )

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=self.model_args.trust_remote_code,
            )

            # Set pad token if not set / 如果未设置填充 token，则设置
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
                logger.info(f"将 pad_token 设置为 eos_token: {tokenizer.eos_token}")

            # Ensure special tokens are properly configured
            # 确保正确配置特殊 token
            if tokenizer.bos_token is None:
                tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
                tokenizer.bos_token_id = tokenizer.eos_token_id
                logger.info("Added bos_token")
                logger.info("添加了 bos_token")

            logger.info(f"Tokenizer loaded successfully from {tokenizer_path}")
            logger.info(f"成功从 {tokenizer_path} 加载分词器")

            return tokenizer

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            logger.error(f"加载分词器失败: {e}")
            raise

    def load_model(self) -> transformers.PreTrainedModel:
        """
        Load and configure model
        加载并配置模型

        Returns:
            Configured model / 配置好的模型
        """
        logger.info("Loading model...")
        logger.info("加载模型...")

        # Determine torch dtype / 确定 torch 数据类型
        torch_dtype = self.model_args.torch_dtype
        if torch_dtype == "auto":
            torch_dtype = None
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        try:
            # Load model / 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                self.model_args.model_name_or_path,
                trust_remote_code=self.model_args.trust_remote_code,
                torch_dtype=torch_dtype,
                device_map=self.model_args.device_map,
            )

            # Disable caching for training / 训练时禁用缓存
            model.config.use_cache = self.model_args.use_cache

            # Enable gradient checkpointing if specified
            # 如果指定，启用梯度检查点
            if self.training_args.gradient_checkpointing:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
                logger.info("已启用梯度检查点")

            # Fix for DDP issues with boolean buffers
            # 修复 DDP 与布尔缓冲区的问题
            if hasattr(model, '_ddp_params_and_buffers_to_ignore'):
                model._ddp_params_and_buffers_to_ignore = [
                    name for name, buffer in model.named_buffers()
                    if buffer.dtype == torch.bool
                ]

            logger.info(f"Model loaded successfully from {self.model_args.model_name_or_path}")
            logger.info(f"成功从 {self.model_args.model_name_or_path} 加载模型")

            # Apply LoRA if specified / 如果指定，应用 LoRA
            if self.lora_args.use_lora:
                model = self._apply_lora(model)

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(f"加载模型失败: {e}")
            raise

    def _apply_lora(
        self,
        model: transformers.PreTrainedModel
    ) -> transformers.PreTrainedModel:
        """
        Apply LoRA to the model
        为模型应用 LoRA

        Args:
            model: Base model / 基础模型

        Returns:
            Model with LoRA adapters / 带有 LoRA 适配器的模型
        """
        logger.info("Applying LoRA configuration...")
        logger.info("应用 LoRA 配置...")

        # Determine target modules if not specified
        # 如果未指定目标模块，则确定目标模块
        target_modules = self.lora_args.lora_target_modules
        if target_modules is None:
            # Default target modules for common architectures
            # 常见架构的默认目标模块
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            logger.info(f"Using default target modules: {target_modules}")
            logger.info(f"使用默认目标模块: {target_modules}")

        # Create LoRA config / 创建 LoRA 配置
        lora_config = LoraConfig(
            r=self.lora_args.lora_r,
            lora_alpha=self.lora_args.lora_alpha,
            lora_dropout=self.lora_args.lora_dropout,
            target_modules=target_modules,
            bias=self.lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        # Apply LoRA / 应用 LoRA
        model = get_peft_model(model, lora_config)

        # Print trainable parameters / 打印可训练参数
        trainable_params, all_params = self._count_parameters(model)
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_params:,} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )
        logger.info(
            f"可训练参数: {trainable_params:,} || "
            f"总参数: {all_params:,} || "
            f"可训练占比: {100 * trainable_params / all_params:.2f}%"
        )

        return model

    @staticmethod
    def _count_parameters(model: transformers.PreTrainedModel) -> tuple:
        """
        Count trainable and total parameters
        统计可训练和总参数数量

        Args:
            model: Model to count parameters for / 要统计参数的模型

        Returns:
            Tuple of (trainable_params, all_params) / (可训练参数, 总参数) 的元组
        """
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        all_params = sum(p.numel() for p in model.parameters())
        return trainable_params, all_params

    def load_datasets(self) -> tuple:
        """
        Load and prepare datasets
        加载并准备数据集

        Returns:
            Tuple of (train_dataset, eval_dataset) / (训练数据集, 评估数据集) 的元组
        """
        logger.info("Loading datasets...")
        logger.info("加载数据集...")

        try:
            # Load training dataset / 加载训练数据集
            logger.info(f"Loading training dataset from {self.data_args.train_dataset_path}")
            logger.info(f"从 {self.data_args.train_dataset_path} 加载训练数据集")

            train_dataset = load_dpo_dataset(
                data_path=self.data_args.train_dataset_path,
                max_length=self.data_args.max_length,
                sanity_check=self.data_args.sanity_check,
                num_proc=self.data_args.num_proc,
                system=self.data_args.system_prompt,
            )

            logger.info(f"Training dataset loaded: {len(train_dataset)} samples")
            logger.info(f"训练数据集已加载: {len(train_dataset)} 个样本")

            # Load evaluation dataset if provided / 如果提供，加载评估数据集
            eval_dataset = None
            if self.data_args.eval_dataset_path:
                logger.info(f"Loading evaluation dataset from {self.data_args.eval_dataset_path}")
                logger.info(f"从 {self.data_args.eval_dataset_path} 加载评估数据集")

                eval_dataset = load_dpo_dataset(
                    data_path=self.data_args.eval_dataset_path,
                    max_length=self.data_args.max_length,
                    sanity_check=self.data_args.sanity_check,
                    num_proc=self.data_args.num_proc,
                    system=self.data_args.system_prompt,
                )

                logger.info(f"Evaluation dataset loaded: {len(eval_dataset)} samples")
                logger.info(f"评估数据集已加载: {len(eval_dataset)} 个样本")
            else:
                logger.info("No evaluation dataset provided")
                logger.info("未提供评估数据集")

            return train_dataset, eval_dataset

        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            logger.error(f"加载数据集失败: {e}")
            raise

    def create_trainer(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
    ) -> DPOTrainer:
        """
        Create DPO trainer
        创建 DPO 训练器

        Args:
            model: Model to train / 要训练的模型
            tokenizer: Tokenizer / 分词器
            train_dataset: Training dataset / 训练数据集
            eval_dataset: Evaluation dataset (optional) / 评估数据集（可选）

        Returns:
            Configured DPO trainer / 配置好的 DPO 训练器
        """
        logger.info("Creating DPO trainer...")
        logger.info("创建 DPO 训练器...")

        try:
            # Create DPO trainer / 创建 DPO 训练器
            trainer = DPOTrainer(
                model=model,
                ref_model=None,  # Will create reference model automatically
                                # 将自动创建参考模型
                args=self.training_args,
                beta=self.dpo_args.beta,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                max_length=self.data_args.max_length,
                max_prompt_length=self.data_args.max_prompt_length,
                loss_type=self.dpo_args.loss_type,
                label_smoothing=self.dpo_args.label_smoothing,
                generate_during_eval=self.dpo_args.generate_during_eval,
            )

            logger.info("DPO trainer created successfully")
            logger.info("DPO 训练器创建成功")

            return trainer

        except Exception as e:
            logger.error(f"Failed to create trainer: {e}")
            logger.error(f"创建训练器失败: {e}")
            raise

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Execute the complete training pipeline
        执行完整的训练流程

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
                                   要恢复的检查点路径（可选）
        """
        logger.info("="*80)
        logger.info("Starting DPO Training Pipeline")
        logger.info("开始 DPO 训练流程")
        logger.info("="*80)

        try:
            # Step 1: Load tokenizer / 步骤 1: 加载分词器
            tokenizer = self.load_tokenizer()

            # Step 2: Load model / 步骤 2: 加载模型
            model = self.load_model()

            # Step 3: Load datasets / 步骤 3: 加载数据集
            train_dataset, eval_dataset = self.load_datasets()

            # Step 4: Create trainer / 步骤 4: 创建训练器
            trainer = self.create_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
            )

            # Step 5: Train / 步骤 5: 训练
            logger.info("Starting training...")
            logger.info("开始训练...")

            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            logger.info("Training completed!")
            logger.info("训练完成!")

            # Step 6: Save model / 步骤 6: 保存模型
            logger.info("Saving final model...")
            logger.info("保存最终模型...")

            output_dir = os.path.join(self.training_args.output_dir, "final_model")
            os.makedirs(output_dir, exist_ok=True)

            # Save model / 保存模型
            trainer.save_model(output_dir)

            # Save tokenizer / 保存分词器
            tokenizer.save_pretrained(output_dir)

            # Save training metrics / 保存训练指标
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

            logger.info(f"Model saved to {output_dir}")
            logger.info(f"模型已保存到 {output_dir}")

            logger.info("="*80)
            logger.info("DPO Training Pipeline Completed Successfully!")
            logger.info("DPO 训练流程成功完成!")
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            logger.error(f"训练流程失败: {e}")
            raise


# ============================================================================
# Configuration Loading / 配置加载
# ============================================================================

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    从 YAML 文件加载配置

    Args:
        config_path: Path to YAML configuration file
                    YAML 配置文件路径

    Returns:
        Configuration dictionary / 配置字典
    """
    logger.info(f"Loading configuration from {config_path}")
    logger.info(f"从 {config_path} 加载配置")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.info("Configuration loaded successfully")
        logger.info("配置加载成功")

        return config

    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.error(f"加载配置失败: {e}")
        raise


def merge_configs(
    yaml_config: Optional[Dict[str, Any]],
    cli_args: Optional[List[str]] = None
) -> tuple:
    """
    Merge YAML configuration with command-line arguments
    合并 YAML 配置和命令行参数

    Command-line arguments take precedence over YAML configuration.
    命令行参数优先于 YAML 配置。

    Args:
        yaml_config: Configuration from YAML file / 来自 YAML 文件的配置
        cli_args: Command-line arguments / 命令行参数

    Returns:
        Tuple of (ModelArguments, DataArguments, TrainingArguments,
                  DPOTrainingArguments, LoRAArguments)
        (模型参数, 数据参数, 训练参数, DPO训练参数, LoRA参数) 的元组
    """
    # Create parser / 创建解析器
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        DPOTrainingArguments,
        LoRAArguments,
    ))

    # If YAML config provided, convert to command-line format
    # 如果提供了 YAML 配置，转换为命令行格式
    if yaml_config is not None:
        # Flatten YAML config into command-line arguments
        # 将 YAML 配置展平为命令行参数
        yaml_args = []
        for section, params in yaml_config.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    if value is not None:
                        if isinstance(value, bool):
                            if value:
                                yaml_args.append(f"--{key}")
                        elif isinstance(value, list):
                            yaml_args.append(f"--{key}")
                            yaml_args.extend([str(v) for v in value])
                        else:
                            yaml_args.append(f"--{key}")
                            yaml_args.append(str(value))

        # Parse YAML args first, then override with CLI args
        # 首先解析 YAML 参数，然后用 CLI 参数覆盖
        if cli_args:
            yaml_args.extend(cli_args)

        model_args, data_args, training_args, dpo_args, lora_args = (
            parser.parse_args_into_dataclasses(args=yaml_args)
        )
    else:
        # Parse only CLI args / 仅解析 CLI 参数
        model_args, data_args, training_args, dpo_args, lora_args = (
            parser.parse_args_into_dataclasses(args=cli_args)
        )

    return model_args, data_args, training_args, dpo_args, lora_args


# ============================================================================
# Main Function / 主函数
# ============================================================================

def main(
    config_path: Optional[str] = None,
    cli_args: Optional[List[str]] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Main entry point for DPO training
    DPO 训练的主入口点

    Args:
        config_path: Path to YAML configuration file (optional)
                    YAML 配置文件路径（可选）
        cli_args: Command-line arguments (optional)
                 命令行参数（可选）
        resume_from_checkpoint: Path to checkpoint to resume from (optional)
                               要恢复的检查点路径（可选）
    """
    # Load configuration / 加载配置
    yaml_config = None
    if config_path:
        yaml_config = load_config_from_yaml(config_path)

    # Merge configurations / 合并配置
    model_args, data_args, training_args, dpo_args, lora_args = merge_configs(
        yaml_config=yaml_config,
        cli_args=cli_args,
    )

    # Log configuration / 记录配置
    logger.info("Configuration / 配置:")
    logger.info(f"Model: {model_args}")
    logger.info(f"Data: {data_args}")
    logger.info(f"Training: {training_args}")
    logger.info(f"DPO: {dpo_args}")
    logger.info(f"LoRA: {lora_args}")

    # Create training pipeline / 创建训练流程
    pipeline = DPOTrainingPipeline(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        dpo_args=dpo_args,
        lora_args=lora_args,
    )

    # Execute training / 执行训练
    pipeline.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    """
    Example usage / 使用示例:

    # Using YAML config / 使用 YAML 配置:
    python dpo_train.py --config configs/dpo_training.yaml

    # Using command-line arguments / 使用命令行参数:
    python dpo_train.py \
        --model_name_or_path /path/to/model \
        --train_dataset_path /path/to/train.jsonl \
        --output_dir ./output \
        --num_train_epochs 3

    # Combining YAML and CLI (CLI overrides YAML) / 结合 YAML 和 CLI（CLI 覆盖 YAML）:
    python dpo_train.py \
        --config configs/dpo_training.yaml \
        --num_train_epochs 5
    """
    main()
