# Reward Model Training Guide
# 奖励模型训练指南

This guide explains how to train reward models (RM) for RLHF using the clean, well-documented training scripts.

本指南说明如何使用干净、文档完善的训练脚本为 RLHF 训练奖励模型（RM）。

---

## Table of Contents / 目录

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Data Preparation](#data-preparation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

---

## Overview

### What is a Reward Model? / 什么是奖励模型？

A Reward Model (RM) is a key component in Reinforcement Learning from Human Feedback (RLHF). It learns to predict human preferences by comparing pairs of responses to the same prompt.

奖励模型（RM）是基于人类反馈的强化学习（RLHF）的关键组件。它通过比较同一提示的两个回答来学习预测人类偏好。

**How it works / 工作原理:**
- Input: A prompt + two responses (chosen vs rejected)
- Output: A scalar reward score for each response
- Training: Learn to assign higher scores to preferred responses

**Training objective / 训练目标:**
```
loss = -log(sigmoid(reward_chosen - reward_rejected))
```

This is the InstructGPT pairwise ranking loss (Bradley-Terry model).

这是 InstructGPT 成对排序损失（Bradley-Terry 模型）。

### Training Pipeline / 训练流程

```
Pretrained Model → SFT Model → Reward Model → RL-tuned Model
                                    ↑
                                你在这里
```

**Important:** Always start RM training from an SFT (Supervised Fine-Tuning) model, not a pretrained-only model!

**重要：**始终从 SFT（监督微调）模型开始 RM 训练，而不是仅预训练的模型！

---

## Quick Start

### Installation / 安装

```bash
# Install required packages
pip install torch transformers datasets scikit-learn omegaconf

# Optional: Install for better logging
pip install tensorboard wandb mlflow
```

### Basic Training Command / 基本训练命令

```bash
# Train with default configuration
python scripts/train_rm.py

# Train with custom model and data
python scripts/train_rm.py \
    model_path=/path/to/sft/model \
    data_path=data/rm_train/my_data.jsonl \
    output_dir=outputs/my_reward_model
```

---

## Data Preparation

### Required Format / 所需格式

Your training data should be a JSONL file where each line contains:

您的训练数据应该是一个 JSONL 文件，其中每行包含：

```json
{
    "prompt": "用户问题或指令",
    "chosen": "人类偏好的回答",
    "rejected": "人类不偏好的回答"
}
```

### Example Data / 示例数据

```json
{"prompt": "什么是机器学习？", "chosen": "机器学习是人工智能的一个分支，通过算法让计算机从数据中学习并改进性能，而无需显式编程。", "rejected": "机器学习就是让机器学习。"}
{"prompt": "写一首关于春天的诗", "chosen": "春风拂面花满园，\n绿柳成荫鸟语欢。\n万物复苏展新颜，\n大地处处是诗篇。", "rejected": "春天很美。"}
```

### Data Collection Tips / 数据收集提示

**High-quality preference data is crucial! / 高质量的偏好数据至关重要！**

1. **Consistent Preferences / 一致的偏好**
   - Ensure chosen responses are genuinely better
   - 确保选择的回答真正更好
   - Avoid ambiguous or subjective pairs
   - 避免模糊或主观的配对

2. **Diverse Prompts / 多样化的提示**
   - Cover various task types (QA, creative writing, reasoning, etc.)
   - 涵盖各种任务类型（问答、创意写作、推理等）
   - Include different difficulty levels
   - 包含不同的难度级别

3. **Balanced Distribution / 平衡的分布**
   - Mix easy and hard comparisons
   - 混合简单和困难的比较
   - Avoid too many trivial examples
   - 避免过多琐碎的示例

4. **Dataset Size / 数据集大小**
   - Minimum: ~1,000 pairs
   - Recommended: 10,000+ pairs for good performance
   - 最低：约 1,000 对
   - 推荐：10,000+ 对以获得良好性能

### Data Processing Script / 数据处理脚本

If you have multiple data sources, you can use the processing script:

如果您有多个数据源，可以使用处理脚本：

```python
# See tiny-llm-zh/utils/rm_train_process.py for reference
# 参考 tiny-llm-zh/utils/rm_train_process.py
```

---

## Configuration

### Configuration File / 配置文件

Edit `configs/rm_training.yaml`:

编辑 `configs/rm_training.yaml`：

```yaml
# Model settings
model_path: models/sft_checkpoint
model_type: auto

# Data settings
data_path: data/rm_train/rm_data.jsonl
max_length: 512
system_prompt: "你是由wdndev开发的个人助手。"

# Training hyperparameters
num_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-5

# RM-specific
loss_type: sigmoid  # or 'hinge'
margin: 0.0
```

### Key Parameters / 关键参数

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `learning_rate` | Learning rate | 1e-6 to 1e-4 |
| `num_epochs` | Training epochs | 1-5 |
| `max_length` | Max sequence length | 256-1024 |
| `per_device_train_batch_size` | Batch size per GPU | 2-8 |
| `gradient_accumulation_steps` | Gradient accumulation | 2-8 |
| `loss_type` | Loss function | sigmoid, hinge |

### Effective Batch Size / 有效批次大小

```
Effective Batch Size = per_device_train_batch_size ×
                       gradient_accumulation_steps ×
                       num_gpus
```

**Example / 示例:**
```
4 × 4 × 2 = 32
```

---

## Training

### Start Training / 开始训练

```bash
# Basic training
python scripts/train_rm.py

# With custom config
python scripts/train_rm.py --config-name my_rm_config

# Override parameters
python scripts/train_rm.py \
    learning_rate=2e-5 \
    num_epochs=5 \
    output_dir=outputs/rm_v2
```

### Monitor Training / 监控训练

#### Using TensorBoard / 使用 TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir outputs/reward_model/logs

# Open browser to http://localhost:6006
```

#### Key Metrics to Watch / 要监控的关键指标

1. **Training Loss / 训练损失**
   - Should decrease over time
   - 应随时间减少
   - Typical range: 0.3-0.7 at start → 0.1-0.3 at end
   - 典型范围：开始时 0.3-0.7 → 结束时 0.1-0.3

2. **Accuracy / 准确率**
   - How often reward_chosen > reward_rejected
   - reward_chosen > reward_rejected 的频率
   - Target: >0.60, Good: >0.70, Excellent: >0.80
   - 目标：>0.60，良好：>0.70，优秀：>0.80

3. **Margin Mean / 平均边距**
   - Average difference: reward_chosen - reward_rejected
   - 平均差异：reward_chosen - reward_rejected
   - Should be positive and stable/increasing
   - 应为正值且稳定/增加

4. **Margin Std / 边距标准差**
   - Variation in reward differences
   - 奖励差异的变化
   - Too high: inconsistent predictions
   - 太高：预测不一致

### Expected Training Time / 预期训练时间

| Dataset Size | GPU | Time (approx) |
|--------------|-----|---------------|
| 10K pairs | V100 | 2-3 hours |
| 50K pairs | V100 | 8-12 hours |
| 10K pairs | A100 | 1-1.5 hours |

---

## Evaluation

### During Training / 训练期间

The trainer automatically evaluates on the validation set and logs metrics.

训练器会自动在验证集上评估并记录指标。

### After Training / 训练后

Check the final model performance:

检查最终模型性能：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load trained reward model
model = AutoModelForSequenceClassification.from_pretrained(
    "outputs/reward_model/final_reward_model"
)
tokenizer = AutoTokenizer.from_pretrained(
    "outputs/reward_model/final_reward_model"
)

def get_reward(text):
    """Get reward score for a text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        reward = model(**inputs).logits.squeeze().item()
    return reward

# Test
prompt = "什么是AI？"
response1 = "AI是人工智能，让机器模拟人类智能。"
response2 = "不知道。"

text1 = f"<|system|>\n你是助手\n<|user|>\n{prompt}\n<|assistant|>\n{response1}"
text2 = f"<|system|>\n你是助手\n<|user|>\n{prompt}\n<|assistant|>\n{response2}"

r1 = get_reward(text1)
r2 = get_reward(text2)

print(f"Response 1 reward: {r1:.4f}")
print(f"Response 2 reward: {r2:.4f}")
print(f"Better response: {'Response 1' if r1 > r2 else 'Response 2'}")
```

### Good RM Characteristics / 良好 RM 的特征

✅ **Good / 好:**
- Consistently prefers high-quality responses
  始终偏好高质量的回答
- Reasonable margin between chosen and rejected
  选择和拒绝之间有合理的边距
- Generalizes to new prompts
  泛化到新提示

❌ **Bad / 坏:**
- Overfits to training data
  过拟合训练数据
- Gives similar scores to all responses
  给所有回答相似的分数
- Prefers longer responses regardless of quality
  无论质量如何都偏好较长的回答

---

## Troubleshooting

### Common Issues / 常见问题

#### 1. Low Accuracy (<0.6) / 低准确率 (<0.6)

**Possible causes / 可能原因:**
- Poor data quality (ambiguous preferences)
  数据质量差（偏好模糊）
- Learning rate too high or too low
  学习率太高或太低
- Insufficient training
  训练不足

**Solutions / 解决方案:**
```bash
# Try lower learning rate
python scripts/train_rm.py learning_rate=5e-6

# Train longer
python scripts/train_rm.py num_epochs=5

# Check data quality
# Manually review some training samples
```

#### 2. Training Loss Not Decreasing / 训练损失不减少

**Solutions / 解决方案:**
- Reduce learning rate: `learning_rate=1e-6`
- Check data format is correct
- Ensure using SFT model, not pretrained-only

#### 3. Overfitting (train >> eval accuracy) / 过拟合

**Solutions / 解决方案:**
```yaml
# Increase weight decay
weight_decay: 0.1

# Reduce epochs
num_epochs: 2

# Use larger eval set
eval_split_ratio: 0.05
```

#### 4. Out of Memory (OOM) / 内存不足

**Solutions / 解决方案:**
```yaml
# Option 1: Reduce batch size / 选项 1：减少批次大小
per_device_train_batch_size: 2
gradient_accumulation_steps: 8  # Increase to maintain effective batch size

# Option 2: Reduce sequence length / 选项 2：减少序列长度
max_length: 256

# Option 3: Use mixed precision / 选项 3：使用混合精度
bf16: true  # For A100/H100
# or
fp16: true  # For V100/T4

# Option 4: Enable DeepSpeed ZeRO (Recommended for multi-GPU)
# 选项 4：启用 DeepSpeed ZeRO（推荐用于多 GPU）
deepspeed: configs/deepspeed/zero2.json  # For 2-8 GPUs
# or
deepspeed: configs/deepspeed/zero3.json  # For very large models
bf16: true
```

#### 5. All Predictions Same / 所有预测相同

**Possible causes / 可能原因:**
- Model collapsed to always predict same score
  模型崩溃，总是预测相同的分数
- Learning rate too high
  学习率太高

**Solutions / 解决方案:**
- Restart with lower learning rate: `1e-6`
- Check data has sufficient diversity
- Ensure model initialized properly from SFT checkpoint

---

## Advanced Usage

### Custom Loss Functions / 自定义损失函数

Edit `scratch_cs336/training/rm.py` to add custom loss:

编辑 `scratch_cs336/training/rm.py` 添加自定义损失：

```python
class RMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Your custom loss here
        # 在这里添加自定义损失
        pass
```

### Multi-GPU Training / 多 GPU 训练

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 scripts/train_rm.py

# Using accelerate
accelerate launch scripts/train_rm.py
```

### Distributed Training / 分布式训练

```bash
# Multi-node training
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_rm.py
```

### DeepSpeed Training / DeepSpeed 训练

DeepSpeed ZeRO enables efficient training of large models with reduced memory usage.

DeepSpeed ZeRO 能够以降低的内存使用高效训练大型模型。

**Quick Start / 快速开始:**

```bash
# Step 1: Install DeepSpeed / 步骤 1：安装 DeepSpeed
pip install deepspeed

# Step 2: Update config file / 步骤 2：更新配置文件
# Edit configs/rm_training.yaml:
# deepspeed: configs/deepspeed/zero2.json
# bf16: true  # or fp16: true

# Step 3: Run training / 步骤 3：运行训练
torchrun --nproc_per_node=4 scripts/train_rm.py
```

**Which ZeRO Stage? / 使用哪个 ZeRO 阶段？**

```bash
# ZeRO-2: For models up to ~7B on multi-GPU
# ZeRO-2：适用于多 GPU 上最多约 7B 的模型
deepspeed: configs/deepspeed/zero2.json

# ZeRO-3: For very large models (7B+) or limited memory
# ZeRO-3：适用于非常大的模型（7B+）或内存有限的情况
deepspeed: configs/deepspeed/zero3.json
```

**Configuration Example / 配置示例:**

```yaml
# In configs/rm_training.yaml
model_path: models/sft_7b
data_path: data/rm_train/rm_data.jsonl
output_dir: outputs/reward_model_7b

# Training settings
num_epochs: 3
per_device_train_batch_size: 2  # Smaller batch for large models
gradient_accumulation_steps: 8   # Maintain effective batch size
learning_rate: 1e-5

# Enable mixed precision (required for DeepSpeed)
bf16: true
fp16: false

# DeepSpeed configuration
deepspeed: configs/deepspeed/zero2.json
```

**Run with DeepSpeed / 使用 DeepSpeed 运行:**

```bash
# Option 1: Using torchrun (recommended)
torchrun --nproc_per_node=4 scripts/train_rm.py

# Option 2: Using DeepSpeed launcher
deepspeed --num_gpus=4 scripts/train_rm.py

# Option 3: Multi-node with DeepSpeed
deepspeed \
    --num_gpus=8 \
    --num_nodes=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_rm.py
```

**See Also / 另请参阅:**
- Full DeepSpeed guide: `configs/deepspeed/README.md`
- Configuration files: `configs/deepspeed/zero*.json`

### Resume from Checkpoint / 从检查点恢复

```yaml
# In config file
resume_from_checkpoint: outputs/reward_model/checkpoint-1000
```

Or via command line:

或通过命令行：

```bash
python scripts/train_rm.py \
    resume_from_checkpoint=outputs/reward_model/checkpoint-1000
```

### Custom Data Preprocessing / 自定义数据预处理

Modify `RMDataset` in `scratch_cs336/training/datasets.py`:

修改 `scratch_cs336/training/datasets.py` 中的 `RMDataset`：

```python
class RMDataset(Dataset):
    def preprocessing(self, example, debug=False):
        # Custom preprocessing logic
        # 自定义预处理逻辑
        pass
```

---

## Best Practices / 最佳实践

### 1. Data Quality / 数据质量
- **Most important factor for RM success!**
  **RM 成功的最重要因素！**
- Review samples manually
  手动审查样本
- Ensure clear preference distinctions
  确保明确的偏好区分

### 2. Model Selection / 模型选择
- Always use SFT model as base
  始终使用 SFT 模型作为基础
- Model should already follow instructions well
  模型应该已经很好地遵循指令

### 3. Hyperparameters / 超参数
- Start conservative: `lr=1e-5`, `epochs=3`
  保守开始：`lr=1e-5`，`epochs=3`
- Monitor eval metrics closely
  密切监控评估指标
- Don't overtrain (causes reward hacking later)
  不要过度训练（稍后会导致奖励破解）

### 4. Evaluation / 评估
- Use held-out test set
  使用保留的测试集
- Test on diverse prompts
  在不同的提示上测试
- Check for biases (length, format, etc.)
  检查偏差（长度、格式等）

### 5. Monitoring / 监控
- Watch for overfitting early
  及早注意过拟合
- Track accuracy, margin, and loss
  跟踪准确率、边距和损失
- Save best checkpoint by accuracy
  按准确率保存最佳检查点

---

## File Structure / 文件结构

```
LLM-from-Scratch/
├── scratch_cs336/
│   └── training/
│       ├── rm.py                 # Main RM training module
│       └── datasets.py           # Dataset classes
├── scripts/
│   └── train_rm.py               # Entry point script
├── configs/
│   └── rm_training.yaml          # Configuration file
└── outputs/
    └── reward_model/             # Training outputs
        ├── checkpoint-*/         # Saved checkpoints
        ├── logs/                 # TensorBoard logs
        └── final_reward_model/   # Final model
```

---

## References / 参考

1. **InstructGPT Paper**
   - Training language models to follow instructions with human feedback
   - https://arxiv.org/abs/2203.02155

2. **Reward Modeling Resources**
   - OpenAI Alignment Research
   - Anthropic Constitutional AI

3. **Implementation References**
   - HuggingFace TRL library
   - DeepSpeed-Chat

---

## Support / 支持

For questions or issues:

如有问题或疑问：

- Check the troubleshooting section
  查看故障排除部分
- Review code comments in `scratch_cs336/training/rm.py`
  查看 `scratch_cs336/training/rm.py` 中的代码注释
- Open an issue on GitHub
  在 GitHub 上提出问题

---

**Happy Training! / 祝训练愉快！** 🚀
