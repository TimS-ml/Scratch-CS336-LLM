# DPO Training Guide
# DPO 训练指南

This guide explains how to use the DPO (Direct Preference Optimization) training system.

本指南说明如何使用 DPO（直接偏好优化）训练系统。

## Table of Contents / 目录

1. [Overview / 概述](#overview)
2. [Installation / 安装](#installation)
3. [Data Format / 数据格式](#data-format)
4. [Quick Start / 快速开始](#quick-start)
5. [Configuration / 配置](#configuration)
6. [Advanced Usage / 高级用法](#advanced-usage)
7. [Troubleshooting / 故障排除](#troubleshooting)
8. [References / 参考资料](#references)

---

## Overview / 概述

### What is DPO? / 什么是 DPO？

Direct Preference Optimization (DPO) is a reinforcement learning approach that directly optimizes a language model using preference data, without requiring a separate reward model. It's simpler and more stable than traditional RLHF methods like PPO.

直接偏好优化（DPO）是一种强化学习方法，直接使用偏好数据优化语言模型，无需单独的奖励模型。它比传统的 RLHF 方法（如 PPO）更简单、更稳定。

### Key Features / 主要特性

- ✅ Modern implementation using HuggingFace TRL
- ✅ Support for LoRA (Low-Rank Adaptation) fine-tuning
- ✅ Configurable via YAML files or command-line arguments
- ✅ Comprehensive evaluation and metrics tracking
- ✅ Production-ready error handling and logging
- ✅ Bilingual documentation (English + Chinese)

### File Structure / 文件结构

```
scratch_cs336/training/
├── dpo.py                    # Main DPO training module / 主 DPO 训练模块
└── datasets.py               # Dataset loading utilities / 数据集加载工具

scripts/
└── train_dpo.py              # Entry point script / 入口脚本

configs/
└── dpo_training.yaml         # Example configuration / 配置示例

docs/
└── DPO_TRAINING.md           # This file / 本文件
```

---

## Installation / 安装

### Requirements / 依赖项

```bash
pip install torch transformers datasets peft trl pyyaml accelerate
```

### Recommended Versions / 推荐版本

```
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
trl>=0.7.0
```

---

## Data Format / 数据格式

### Training Data / 训练数据

DPO requires preference data in JSONL format with three fields:

DPO 需要 JSONL 格式的偏好数据，包含三个字段：

```jsonl
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris, which is also its largest city and a major European center of art, culture, and history.", "rejected": "Paris."}
{"prompt": "How do I make a cake?", "chosen": "Here's a simple recipe:\n1. Preheat oven to 350°F\n2. Mix flour, sugar, eggs, and butter\n3. Pour into a pan\n4. Bake for 30 minutes", "rejected": "Mix ingredients and bake."}
```

### Field Descriptions / 字段说明

| Field / 字段 | Description / 描述 |
|-------------|-------------------|
| `prompt` | User question or instruction / 用户问题或指令 |
| `chosen` | Preferred/better response / 偏好的/更好的回答 |
| `rejected` | Rejected/worse response / 拒绝的/较差的回答 |

### Data Preparation Tips / 数据准备建议

1. **Quality over Quantity** / 质量重于数量
   - Focus on high-quality preference pairs
   - 专注于高质量的偏好对

2. **Clear Preferences** / 明确的偏好
   - The difference between chosen and rejected should be clear
   - 选择和拒绝之间的差异应该明确

3. **Diversity** / 多样性
   - Include diverse types of prompts and responses
   - 包含多样化的提示和回答类型

4. **Length** / 长度
   - Keep total length (prompt + response) under `max_length`
   - 保持总长度（提示 + 回答）在 `max_length` 以下

---

## Quick Start / 快速开始

### 1. Prepare Your Data / 准备数据

Create your training data file:

创建训练数据文件：

```bash
# Example: data/dpo_train/train.jsonl
mkdir -p data/dpo_train
cat > data/dpo_train/train.jsonl << 'EOF'
{"prompt": "What is AI?", "chosen": "Artificial Intelligence (AI) refers to computer systems that can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.", "rejected": "AI is computers that are smart."}
{"prompt": "Explain machine learning", "chosen": "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.", "rejected": "Machine learning is when computers learn stuff."}
EOF
```

### 2. Basic Training / 基本训练

Using the example configuration:

使用示例配置：

```bash
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml
```

### 3. Custom Training / 自定义训练

Override configuration with command-line arguments:

使用命令行参数覆盖配置：

```bash
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --train_dataset_path data/dpo_train/train.jsonl \
    --output_dir ./outputs/my_dpo_model \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5
```

### 4. Training with LoRA / 使用 LoRA 训练

For parameter-efficient fine-tuning:

用于参数高效微调：

```bash
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05
```

---

## Configuration / 配置

### Configuration File Structure / 配置文件结构

The YAML configuration file has five main sections:

YAML 配置文件包含五个主要部分：

1. **model**: Model configuration / 模型配置
2. **data**: Dataset configuration / 数据集配置
3. **training**: Training hyperparameters / 训练超参数
4. **dpo**: DPO-specific parameters / DPO 特定参数
5. **lora**: LoRA configuration / LoRA 配置

### Key Parameters / 关键参数

#### Model Parameters / 模型参数

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-0.5B"  # Model to fine-tune
  trust_remote_code: true                   # For custom models
  torch_dtype: "auto"                       # "float16", "bfloat16", "float32"
```

#### Data Parameters / 数据参数

```yaml
data:
  train_dataset_path: "data/dpo_train/train.jsonl"
  max_length: 1024              # Max total sequence length
  max_prompt_length: 512        # Max prompt length
  system_prompt: "你是由wdndev开发的个人助手。"
```

#### Training Parameters / 训练参数

```yaml
training:
  output_dir: "./outputs/dpo_training"
  num_train_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 5.0e-5
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.1
  gradient_checkpointing: true
  bf16: false                   # Set to true if GPU supports it
  fp16: true
```

#### DPO Parameters / DPO 参数

```yaml
dpo:
  beta: 0.1                     # KL penalty strength (0.1-0.5 typical)
  loss_type: "sigmoid"          # "sigmoid", "hinge", "ipo", "kto_pair"
  label_smoothing: 0.0
```

#### LoRA Parameters / LoRA 参数

```yaml
lora:
  use_lora: false
  lora_r: 8                     # Rank (8, 16, 32, 64)
  lora_alpha: 16                # Typically 2 * lora_r
  lora_dropout: 0.05
  lora_target_modules: null     # Auto-detect if null
```

---

## Advanced Usage / 高级用法

### Multi-GPU Training / 多 GPU 训练

Using PyTorch Distributed:

使用 PyTorch 分布式：

```bash
torchrun --nproc_per_node=4 scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
```

### Resume from Checkpoint / 从检查点恢复

```bash
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --resume_from_checkpoint ./outputs/dpo_training/checkpoint-500
```

### Monitoring Training / 监控训练

#### TensorBoard

```bash
tensorboard --logdir ./outputs/dpo_training/logs
```

#### Weights & Biases

Modify configuration:

修改配置：

```yaml
training:
  report_to: "wandb"
```

Then set your W&B API key:

然后设置您的 W&B API 密钥：

```bash
export WANDB_API_KEY=your_api_key
python scripts/train_dpo.py --config configs/dpo_training.yaml
```

### Memory Optimization / 内存优化

For large models or limited GPU memory:

对于大模型或有限的 GPU 内存：

```yaml
model:
  torch_dtype: "float16"        # or "bfloat16" if supported
  device_map: "auto"            # Enable model parallelism

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true

lora:
  use_lora: true                # Significantly reduces memory
  lora_r: 8
```

### Custom Target Modules for LoRA / LoRA 自定义目标模块

For LLaMA/Qwen models:

对于 LLaMA/Qwen 模型：

```yaml
lora:
  use_lora: true
  lora_target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
```

For GPT models:

对于 GPT 模型：

```yaml
lora:
  use_lora: true
  lora_target_modules:
    - c_attn
    - c_proj
    - c_fc
```

---

## Troubleshooting / 故障排除

### Common Issues / 常见问题

#### 1. CUDA Out of Memory / CUDA 内存不足

**Error / 错误:**
```
RuntimeError: CUDA out of memory
```

**Solutions / 解决方案:**

1. Reduce batch size:
   ```yaml
   training:
     per_device_train_batch_size: 1
   ```

2. Increase gradient accumulation:
   ```yaml
   training:
     gradient_accumulation_steps: 16
   ```

3. Enable gradient checkpointing:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

4. Use LoRA:
   ```yaml
   lora:
     use_lora: true
   ```

#### 2. Slow Training / 训练缓慢

**Solutions / 解决方案:**

1. Increase batch size if memory allows
2. Use mixed precision training (bf16/fp16)
3. Disable `generate_during_eval`
4. Reduce evaluation frequency
5. Use multiple GPUs

#### 3. Loss Not Decreasing / 损失不下降

**Solutions / 解决方案:**

1. Check data quality
2. Adjust learning rate
3. Modify beta parameter
4. Increase training epochs
5. Check for data preprocessing issues

#### 4. Model Generating Poor Outputs / 模型生成质量差

**Solutions / 解决方案:**

1. Ensure high-quality preference data
2. Increase training epochs
3. Adjust beta parameter (try 0.1, 0.3, 0.5)
4. Use a better base model
5. Collect more diverse training data

---

## Best Practices / 最佳实践

### Data Quality / 数据质量

1. **Clear Preferences** / 明确的偏好
   - Chosen responses should be noticeably better
   - 选择的回答应该明显更好

2. **Diversity** / 多样性
   - Cover various topics and styles
   - 涵盖各种主题和风格

3. **Consistency** / 一致性
   - Maintain consistent quality standards
   - 保持一致的质量标准

### Hyperparameter Tuning / 超参数调优

1. **Beta Parameter** / Beta 参数
   - Start with 0.1
   - Increase if model deviates too much from base
   - Decrease if learning is too conservative
   - 从 0.1 开始
   - 如果模型偏离基础模型太多则增加
   - 如果学习太保守则减少

2. **Learning Rate** / 学习率
   - Start with 5e-5 for full fine-tuning
   - Use 1e-4 to 5e-4 for LoRA
   - 全量微调从 5e-5 开始
   - LoRA 使用 1e-4 到 5e-4

3. **Batch Size** / 批次大小
   - Larger effective batch size (batch_size × accumulation_steps) is better
   - Aim for 32-128 effective batch size
   - 更大的有效批次大小（批次大小 × 累积步数）更好
   - 目标是 32-128 的有效批次大小

### Evaluation / 评估

1. Use a separate evaluation dataset
2. Monitor both loss and generation quality
3. Compare with base model outputs
4. Use human evaluation for final assessment

### Production Deployment / 生产部署

1. Save final model in HuggingFace format
2. Test thoroughly before deployment
3. Monitor inference performance
4. Keep base model for fallback

---

## Example Workflows / 示例工作流程

### Workflow 1: Quick Experiment / 快速实验

```bash
# 1. Prepare small test dataset
head -100 data/full_dpo_data.jsonl > data/test_dpo.jsonl

# 2. Run quick test
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --train_dataset_path data/test_dpo.jsonl \
    --sanity_check \
    --num_train_epochs 1 \
    --eval_steps 10

# 3. Check results
ls outputs/dpo_training/
```

### Workflow 2: Full Training with LoRA / 完整 LoRA 训练

```bash
# 1. Prepare data
# Ensure train.jsonl and eval.jsonl are ready

# 2. Train with LoRA
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --model_name_or_path Qwen/Qwen2.5-7B \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --num_train_epochs 3 \
    --output_dir ./outputs/qwen_dpo_lora

# 3. Monitor with tensorboard
tensorboard --logdir ./outputs/qwen_dpo_lora/logs

# 4. Merge LoRA weights (optional)
# Use merge_lora_weights.py script
```

### Workflow 3: Multi-GPU Production Training / 多 GPU 生产训练

```bash
# 1. Set up environment
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 2. Run distributed training
torchrun --nproc_per_node=4 scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --model_name_or_path meta-llama/Llama-2-13b-hf \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --bf16 \
    --output_dir ./outputs/llama2_dpo_production

# 3. Evaluate final model
python scripts/evaluate_dpo.py \
    --model_path ./outputs/llama2_dpo_production/final_model \
    --test_data data/dpo_test.jsonl
```

---

## References / 参考资料

### Papers / 论文

1. **DPO Paper**: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
2. **LoRA Paper**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

### Documentation / 文档

1. [HuggingFace TRL Documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer)
2. [PEFT Documentation](https://huggingface.co/docs/peft)
3. [Transformers Documentation](https://huggingface.co/docs/transformers)

### Tutorials / 教程

1. [TRL DPO Example](https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/dpo_llama2.py)
2. [HuggingFace Blog: DPO](https://huggingface.co/blog/dpo-trl)

---

## Support / 支持

For issues and questions:

如有问题：

- Check this documentation / 查看本文档
- Review configuration examples / 查看配置示例
- Check HuggingFace TRL issues / 查看 HuggingFace TRL 问题
- Contact: wdndev

---

**Author / 作者**: wdndev
**Date / 日期**: 2025-11-14
**Version / 版本**: 1.0
