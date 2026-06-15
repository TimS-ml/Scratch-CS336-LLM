# DPO Training - Quick Start Guide
# DPO 训练 - 快速入门指南

> For the detailed reference, see [DPO_TRAINING.md](./DPO_TRAINING.md).

## What Was Created / 创建了什么

### Core Files / 核心文件

1. **`/home/user/LLM-from-Scratch/scratch_cs336/training/dpo.py`** (31 KB)
   - Main DPO training module with production-ready implementation
   - 主 DPO 训练模块，生产就绪的实现
   - Features: Type hints, bilingual comments, error handling, LoRA support
   - 特性: 类型提示、双语注释、错误处理、LoRA 支持

2. **`/home/user/LLM-from-Scratch/scripts/train_dpo.py`** (5.7 KB)
   - Entry point script for training
   - 训练的入口脚本
   - Handles argument parsing and configuration
   - 处理参数解析和配置

3. **`/home/user/LLM-from-Scratch/configs/dpo_training.yaml`** (9.9 KB)
   - Comprehensive configuration template
   - 全面的配置模板
   - Includes all parameters with detailed comments
   - 包含所有参数及详细注释

4. **`/home/user/LLM-from-Scratch/scripts/test_dpo_setup.py`** (12 KB)
   - Setup testing script
   - 设置测试脚本
   - Creates test data and validates configuration
   - 创建测试数据并验证配置

5. **`/home/user/LLM-from-Scratch/docs/DPO_TRAINING.md`** (15 KB)
   - Comprehensive documentation
   - 全面的文档
   - Includes examples, troubleshooting, and best practices
   - 包含示例、故障排除和最佳实践

6. **`/home/user/LLM-from-Scratch/scratch_cs336/training/__init__.py`** (506 B)
   - Package initialization
   - 包初始化

---

## Quick Start / 快速开始

### 1. Test Your Setup / 测试设置

```bash
cd /home/user/LLM-from-Scratch
python tests/test_dpo_setup.py
```

This will:
- Create test dataset in `data/dpo_test/`
- Create test configuration in `configs/dpo_test.yaml`
- Verify dataset loading works
- Optionally test model loading

这将:
- 在 `data/dpo_test/` 中创建测试数据集
- 在 `configs/dpo_test.yaml` 中创建测试配置
- 验证数据集加载正常工作
- 可选地测试模型加载

### 2. Prepare Your Data / 准备数据

Create a JSONL file with your preference data:

创建包含偏好数据的 JSONL 文件:

```jsonl
{"prompt": "What is AI?", "chosen": "Detailed and helpful response", "rejected": "Short unhelpful response"}
{"prompt": "Explain ML", "chosen": "Comprehensive explanation", "rejected": "Brief answer"}
```

### 3. Configure Training / 配置训练

Edit `configs/dpo_training.yaml`:

编辑 `configs/dpo_training.yaml`:

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-0.5B"  # Change to your model

data:
  train_dataset_path: "data/dpo_train/train.jsonl"  # Your data
  eval_dataset_path: "data/dpo_train/eval.jsonl"    # Optional

training:
  output_dir: "./outputs/my_dpo_model"
  num_train_epochs: 3
```

### 4. Start Training / 开始训练

#### Basic Training / 基础训练

```bash
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml
```

#### With LoRA / 使用 LoRA

```bash
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --use_lora \
    --lora_r 8 \
    --lora_alpha 16
```

#### Multi-GPU / 多 GPU

```bash
torchrun --nproc_per_node=4 scripts/train_dpo.py \
    --config configs/dpo_training.yaml
```

---

## Key Features / 主要特性

### 1. Production-Ready / 生产就绪

- ✅ Comprehensive error handling / 全面的错误处理
- ✅ Detailed logging / 详细的日志记录
- ✅ Progress tracking / 进度跟踪
- ✅ Checkpoint management / 检查点管理

### 2. Flexible Configuration / 灵活的配置

- ✅ YAML configuration files / YAML 配置文件
- ✅ Command-line override / 命令行覆盖
- ✅ Multiple configuration presets / 多个配置预设

### 3. Advanced Features / 高级特性

- ✅ LoRA support for parameter-efficient training / 参数高效训练的 LoRA 支持
- ✅ Mixed precision training (FP16/BF16) / 混合精度训练
- ✅ Gradient checkpointing / 梯度检查点
- ✅ Multi-GPU distributed training / 多 GPU 分布式训练

### 4. Comprehensive Documentation / 全面的文档

- ✅ Bilingual comments (English + Chinese) / 双语注释
- ✅ Type hints throughout / 全面的类型提示
- ✅ Detailed README with examples / 带示例的详细 README
- ✅ Troubleshooting guide / 故障排除指南

---

## File Structure / 文件结构

```
LLM-from-Scratch/
├── scratch_cs336/
│   └── training/
│       ├── __init__.py                    # Package init
│       ├── dpo.py                         # Main training module ⭐
│       └── datasets.py                    # Dataset utilities
├── scripts/
│   └── train_dpo.py                       # Entry point ⭐
├── tests/
│   └── test_dpo_setup.py                  # Setup tester ⭐
├── configs/
│   └── dpo_training.yaml                  # Config template ⭐
└── docs/
    ├── DPO_QUICK_START.md                 # This file ⭐
    └── DPO_TRAINING.md                    # Comprehensive docs ⭐
```

---

## Architecture Overview / 架构概述

### Main Components / 主要组件

```
train_dpo.py (Entry Point)
    ↓
dpo.py (Main Module)
    ├── DPOTrainingPipeline
    │   ├── load_tokenizer()
    │   ├── load_model()
    │   ├── load_datasets() → datasets.load_dpo_dataset()
    │   ├── create_trainer() → TRL DPOTrainer
    │   └── train()
    │
    ├── Configuration Classes
    │   ├── ModelArguments
    │   ├── DataArguments
    │   ├── TrainingArguments (HF)
    │   ├── DPOTrainingArguments
    │   └── LoRAArguments
    │
    └── Config Management
        ├── load_config_from_yaml()
        └── merge_configs()
```

---

## Example Workflows / 示例工作流程

### Workflow 1: Quick Test (5 minutes) / 快速测试（5分钟）

```bash
# 1. Test setup
python tests/test_dpo_setup.py

# 2. Run quick training
python scripts/train_dpo.py \
    --config configs/dpo_test.yaml
```

### Workflow 2: Full Training with Your Data / 使用您的数据进行完整训练

```bash
# 1. Prepare your data
# Create train.jsonl and eval.jsonl

# 2. Edit config
vim configs/dpo_training.yaml

# 3. Train
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml

# 4. Monitor
tensorboard --logdir outputs/dpo_training/logs
```

### Workflow 3: Production Training with LoRA / 使用 LoRA 的生产训练

```bash
# Large model with memory optimization
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --gradient_checkpointing \
    --bf16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32
```

---

## Configuration Examples / 配置示例

### Minimal Configuration / 最小配置

```yaml
model:
  model_name_or_path: "Qwen/Qwen2.5-0.5B"

data:
  train_dataset_path: "data/train.jsonl"

training:
  output_dir: "./outputs/dpo"
  num_train_epochs: 3
```

### Memory-Optimized Configuration / 内存优化配置

```yaml
model:
  torch_dtype: "float16"
  device_map: "auto"

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  gradient_checkpointing: true
  fp16: true

lora:
  use_lora: true
  lora_r: 8
```

### Multi-GPU Configuration / 多 GPU 配置

```bash
# Run with torchrun
torchrun --nproc_per_node=4 scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4
```

---

## Common Commands / 常用命令

### Training / 训练

```bash
# Basic training
python scripts/train_dpo.py --config configs/dpo_training.yaml

# Override specific parameters
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --num_train_epochs 5 \
    --learning_rate 1e-5

# Training with LoRA
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --use_lora
```

### Monitoring / 监控

```bash
# TensorBoard
tensorboard --logdir outputs/dpo_training/logs

# Check logs
tail -f outputs/dpo_training/logs/*.log
```

### Resume Training / 恢复训练

```bash
python scripts/train_dpo.py \
    --config configs/dpo_training.yaml \
    --resume_from_checkpoint outputs/dpo_training/checkpoint-500
```

---

## Next Steps / 下一步

1. **Read the Full Documentation** / 阅读完整文档
   - See `docs/DPO_TRAINING.md`

2. **Test Your Setup** / 测试设置
   - Run `python tests/test_dpo_setup.py`

3. **Prepare Your Data** / 准备数据
   - Create JSONL files with prompt/chosen/rejected

4. **Configure Training** / 配置训练
   - Edit `configs/dpo_training.yaml`

5. **Start Training** / 开始训练
   - Run `python scripts/train_dpo.py --config ...`

6. **Monitor and Evaluate** / 监控和评估
   - Use TensorBoard or W&B
   - Evaluate generated outputs

---

## Support Resources / 支持资源

- **Full Documentation**: `docs/DPO_TRAINING.md`
- **Configuration Template**: `configs/dpo_training.yaml`
- **Test Script**: `tests/test_dpo_setup.py`
- **Example Data**: Created by test script in `data/dpo_test/`

---

## Tips for Success / 成功提示

1. **Start Small** / 从小开始
   - Use test script to verify setup
   - Train on small dataset first
   - 使用测试脚本验证设置
   - 先在小数据集上训练

2. **Monitor Training** / 监控训练
   - Watch loss curves in TensorBoard
   - Check sample generations
   - 在 TensorBoard 中观察损失曲线
   - 检查生成样本

3. **Optimize Hyperparameters** / 优化超参数
   - Start with beta=0.1
   - Adjust learning rate
   - Use LoRA for large models
   - 从 beta=0.1 开始
   - 调整学习率
   - 大模型使用 LoRA

4. **Evaluate Thoroughly** / 彻底评估
   - Use separate test set
   - Compare with base model
   - Manual quality assessment
   - 使用单独的测试集
   - 与基础模型比较
   - 人工质量评估

---

**Author / 作者**: wdndev
**Date / 日期**: 2025-11-14
**Version / 版本**: 1.0

---

**Ready to start? Run this command:**
**准备开始？运行此命令:**

```bash
cd /home/user/LLM-from-Scratch
python tests/test_dpo_setup.py
```
