# DeepSpeed Configuration Guide
# DeepSpeed 配置指南

This directory contains production-ready DeepSpeed ZeRO configurations for efficient distributed training.

此目录包含用于高效分布式训练的生产就绪 DeepSpeed ZeRO 配置。

---

## Table of Contents / 目录

1. [What is DeepSpeed ZeRO?](#what-is-deepspeed-zero)
2. [Available Configurations](#available-configurations)
3. [Which Configuration to Use?](#which-configuration-to-use)
4. [How to Use](#how-to-use)
5. [Configuration Details](#configuration-details)
6. [Performance Tips](#performance-tips)
7. [Troubleshooting](#troubleshooting)
8. [References](#references)

---

## What is DeepSpeed ZeRO?
## 什么是 DeepSpeed ZeRO？

**DeepSpeed** is a deep learning optimization library developed by Microsoft that makes distributed training easy, efficient, and effective. **ZeRO (Zero Redundancy Optimizer)** is DeepSpeed's main optimization technique that eliminates memory redundancy in data-parallel training.

**DeepSpeed** 是微软开发的深度学习优化库，使分布式训练变得简单、高效和有效。**ZeRO（零冗余优化器）**是 DeepSpeed 的主要优化技术，它消除了数据并行训练中的内存冗余。

### Key Benefits / 主要优势

- **Reduced Memory Usage** / **减少内存使用**: Train larger models with the same hardware
- **Faster Training** / **更快的训练**: Efficient communication and computation overlap
- **Easy to Use** / **易于使用**: Minimal code changes required
- **Scalability** / **可扩展性**: Scale from single GPU to multi-node clusters

### ZeRO Stages / ZeRO 阶段

| Stage | Partitions | Memory Savings | Use Case |
|-------|-----------|----------------|----------|
| **ZeRO-1** | Optimizer States | ~4x | Deprecated, use ZeRO-2 |
| **ZeRO-2** | Optimizer + Gradients | ~8x | Most common, good balance |
| **ZeRO-3** | Optimizer + Gradients + Parameters | ~Linear with GPUs | Very large models |

| 阶段 | 分片内容 | 内存节省 | 使用场景 |
|------|---------|---------|---------|
| **ZeRO-1** | 优化器状态 | ~4倍 | 已弃用，使用 ZeRO-2 |
| **ZeRO-2** | 优化器 + 梯度 | ~8倍 | 最常见，平衡良好 |
| **ZeRO-3** | 优化器 + 梯度 + 参数 | ~与GPU数量线性相关 | 非常大的模型 |

---

## Available Configurations
## 可用配置

### `zero2.json` - ZeRO Stage 2

**Best for / 最适合**:
- Multi-GPU training (2-8 GPUs) / 多 GPU 训练（2-8 GPU）
- Models up to ~7B parameters / 最多约 7B 参数的模型
- Standard training workloads / 标准训练工作负载

**Features / 特性**:
- Optimizer state partitioning / 优化器状态分片
- Gradient partitioning / 梯度分片
- FP16/BF16 mixed precision / FP16/BF16 混合精度
- Communication-computation overlap / 通信-计算重叠
- Optional CPU offloading / 可选的 CPU 卸载

### `zero3.json` - ZeRO Stage 3

**Best for / 最适合**:
- Very large models (7B+ parameters) / 非常大的模型（7B+ 参数）
- Limited GPU memory / GPU 内存有限
- Multi-node training / 多节点训练

**Features / 特性**:
- Full model sharding / 完全模型分片
- CPU/NVMe offloading / CPU/NVMe 卸载
- Activation checkpointing / 激活检查点
- Maximum memory efficiency / 最大内存效率

---

## Which Configuration to Use?
## 使用哪个配置？

### Decision Tree / 决策树

```
Does your model fit in GPU memory with standard training?
您的模型能否在标准训练中放入 GPU 内存？
│
├─ YES → Use standard DDP (no DeepSpeed needed)
│        使用标准 DDP（不需要 DeepSpeed）
│
└─ NO → Do you have multiple GPUs?
        您有多个 GPU 吗？
        │
        ├─ YES (2-8 GPUs) → Use zero2.json
        │                    使用 zero2.json
        │
        └─ YES (8+ GPUs or very large model) → Use zero3.json
                                                使用 zero3.json
```

### By Model Size / 按模型大小

| Model Size | Recommended Config | Notes |
|------------|-------------------|-------|
| < 1B params | Standard DDP | DeepSpeed not needed |
| 1B - 7B | `zero2.json` | Good balance of speed and memory |
| 7B - 13B | `zero2.json` or `zero3.json` | Depends on GPU memory |
| 13B+ | `zero3.json` | May need CPU offloading |
| 70B+ | `zero3.json` with offloading | Requires many GPUs or offloading |

| 模型大小 | 推荐配置 | 备注 |
|---------|---------|------|
| < 1B 参数 | 标准 DDP | 不需要 DeepSpeed |
| 1B - 7B | `zero2.json` | 速度和内存的良好平衡 |
| 7B - 13B | `zero2.json` 或 `zero3.json` | 取决于 GPU 内存 |
| 13B+ | `zero3.json` | 可能需要 CPU 卸载 |
| 70B+ | `zero3.json` 带卸载 | 需要多个 GPU 或卸载 |

### By Hardware / 按硬件

**Single GPU** / **单 GPU**:
- DeepSpeed ZeRO not beneficial / DeepSpeed ZeRO 没有优势
- Consider CPU offloading if model doesn't fit / 如果模型放不下，考虑 CPU 卸载

**2-8 GPUs (e.g., 8x A100 40GB)** / **2-8 个 GPU（例如，8x A100 40GB）**:
- Use `zero2.json` for most cases / 大多数情况下使用 `zero2.json`
- Fast training with good memory savings / 快速训练，良好的内存节省

**8+ GPUs or Multi-Node** / **8+ 个 GPU 或多节点**:
- Use `zero3.json` for maximum efficiency / 使用 `zero3.json` 以获得最大效率
- Especially important for 13B+ models / 对于 13B+ 模型特别重要

---

## How to Use
## 如何使用

### Method 1: With Transformers Trainer (Recommended)
### 方法 1：使用 Transformers Trainer（推荐）

The training scripts in this repository use HuggingFace Transformers Trainer, which has built-in DeepSpeed support.

此仓库中的训练脚本使用 HuggingFace Transformers Trainer，它具有内置的 DeepSpeed 支持。

#### Step 1: Install DeepSpeed
#### 步骤 1：安装 DeepSpeed

```bash
pip install deepspeed
```

#### Step 2: Modify Your Training Config (e.g., `configs/rm_training.yaml`)
#### 步骤 2：修改训练配置（例如，`configs/rm_training.yaml`）

Add DeepSpeed configuration path:

添加 DeepSpeed 配置路径：

```yaml
# Add to your training config / 添加到训练配置
deepspeed: scripts/configs/deepspeed/zero2.json  # or zero3.json

# Make sure these are set to 'auto' / 确保这些设置为 'auto'
gradient_accumulation_steps: auto  # or specific number
per_device_train_batch_size: auto  # or specific number
```

#### Step 3: Run Training with DeepSpeed
#### 步骤 3：使用 DeepSpeed 运行训练

**Single Node / 单节点**:

```bash
# For Reward Model training / 奖励模型训练
python scripts/train_rm.py

# The deepspeed config will be automatically loaded from the YAML
# DeepSpeed 配置将从 YAML 自动加载
```

**Multi-GPU with torchrun** / **使用 torchrun 的多 GPU**:

```bash
torchrun --nproc_per_node=4 scripts/train_rm.py
```

**Multi-Node** / **多节点**:

```bash
# On each node / 在每个节点上
torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train_rm.py
```

### Method 2: With DeepSpeed Launcher
### 方法 2：使用 DeepSpeed 启动器

Alternatively, use DeepSpeed's launcher directly:

或者，直接使用 DeepSpeed 的启动器：

```bash
deepspeed --num_gpus=4 scripts/train_rm.py \
    --deepspeed scripts/configs/deepspeed/zero2.json \
    <other training arguments>
```

### Method 3: With Hydra Override
### 方法 3：使用 Hydra 覆盖

```bash
# Override DeepSpeed config at runtime / 在运行时覆盖 DeepSpeed 配置
python scripts/train_rm.py \
    deepspeed=scripts/configs/deepspeed/zero3.json
```

---

## Configuration Details
## 配置详细信息

### Key Parameters Explained / 关键参数解释

#### 1. Batch Size Settings / 批次大小设置

```json
{
  "train_batch_size": "auto",  // Total batch size across all GPUs / 所有 GPU 的总批次大小
  "train_micro_batch_size_per_gpu": "auto",  // Batch size per GPU / 每个 GPU 的批次大小
  "gradient_accumulation_steps": "auto"  // Number of accumulation steps / 累积步数
}
```

**"auto"** means DeepSpeed will read these from your Trainer arguments.

**"auto"** 意味着 DeepSpeed 将从您的 Trainer 参数中读取这些值。

**Calculation / 计算**:
```
total_batch_size = micro_batch_size_per_gpu × num_gpus × gradient_accumulation_steps
```

#### 2. Mixed Precision / 混合精度

**FP16** (Older GPUs: V100, T4):
```json
{
  "fp16": {"enabled": true},
  "bf16": {"enabled": false},
  "data_types": {"grad_accum_dtype": "fp16"}
}
```

**BF16** (Newer GPUs: A100, H100):
```json
{
  "fp16": {"enabled": false},
  "bf16": {"enabled": true},
  "data_types": {"grad_accum_dtype": "fp32"}
}
```

**Why BF16?** / **为什么使用 BF16？**
- Better numerical stability / 更好的数值稳定性
- No loss scaling needed / 不需要损失缩放
- Recommended for modern GPUs / 推荐用于现代 GPU

#### 3. CPU Offloading / CPU 卸载

**When to enable** / **何时启用**:
- Model doesn't fit in GPU memory / 模型放不进 GPU 内存
- You have sufficient CPU RAM (>100GB) / 您有足够的 CPU 内存（>100GB）
- You can tolerate slower training / 您可以容忍较慢的训练

**Zero2 - Optimizer Offloading** / **Zero2 - 优化器卸载**:
```json
{
  "offload_optimizer": {
    "device": "cpu",  // Change from "none" to "cpu" / 从 "none" 改为 "cpu"
    "pin_memory": true
  }
}
```

**Zero3 - Full Offloading** / **Zero3 - 完全卸载**:
```json
{
  "offload_optimizer": {"device": "cpu"},
  "offload_param": {"device": "cpu"}  // Also offload parameters / 同时卸载参数
}
```

#### 4. Activation Checkpointing / 激活检查点

**Recommended for ZeRO-3** / **推荐用于 ZeRO-3**:

```json
{
  "activation_checkpointing": {
    "partition_activations": true,
    "contiguous_memory_optimization": true
  }
}
```

**Trade-off / 权衡**:
- Saves memory / 节省内存
- Increases computation time (~20-30%) / 增加计算时间（约 20-30%）

---

## Performance Tips
## 性能技巧

### 1. Choose the Right Precision / 选择正确的精度

- **V100 or T4 GPUs**: Use FP16 / 使用 FP16
- **A100 or H100 GPUs**: Use BF16 / 使用 BF16
- **Numerical instability**: Switch to BF16 or FP32 / 切换到 BF16 或 FP32

### 2. Optimize Batch Size / 优化批次大小

```python
# Find optimal batch size / 找到最佳批次大小
# Start with small batch and increase until OOM / 从小批次开始，增加直到 OOM

# Example configurations / 示例配置：
# - 7B model on 8x A100 40GB with ZeRO-2: micro_batch=4, gas=8
# - 13B model on 8x A100 40GB with ZeRO-3: micro_batch=2, gas=16
```

### 3. Enable Communication Overlap / 启用通信重叠

Already enabled in our configs:
我们的配置中已启用：

```json
{
  "overlap_comm": true,
  "contiguous_gradients": true
}
```

### 4. Monitor Training / 监控训练

Enable profiling temporarily to diagnose issues:
临时启用分析以诊断问题：

```json
{
  "wall_clock_breakdown": true,  // Shows time breakdown / 显示时间分解
  "flops_profiler": {
    "enabled": true,
    "profile_step": 30
  }
}
```

### 5. Multi-Node Optimization / 多节点优化

- Use high-bandwidth network (InfiniBand preferred) / 使用高带宽网络（首选 InfiniBand）
- Adjust `allgather_bucket_size` and `reduce_bucket_size` / 调整桶大小参数
- Ensure uniform hardware across nodes / 确保节点间硬件一致

---

## Troubleshooting
## 故障排除

### Out of Memory (OOM) / 内存不足

**Symptoms / 症状**:
```
RuntimeError: CUDA out of memory
```

**Solutions / 解决方案**:

1. **Reduce micro batch size** / **减少微批次大小**:
   ```yaml
   per_device_train_batch_size: 2  # Reduce from 4 to 2 / 从 4 减少到 2
   gradient_accumulation_steps: 8  # Increase to maintain total batch / 增加以维持总批次
   ```

2. **Switch to ZeRO-3** / **切换到 ZeRO-3**:
   ```yaml
   deepspeed: scripts/configs/deepspeed/zero3.json
   ```

3. **Enable CPU offloading** / **启用 CPU 卸载**:
   Edit config to set `"device": "cpu"` in offload sections.
   编辑配置，在卸载部分设置 `"device": "cpu"`。

4. **Enable activation checkpointing** / **启用激活检查点**:
   ```python
   # In your model initialization / 在模型初始化中
   model.gradient_checkpointing_enable()
   ```

5. **Reduce sequence length** / **减少序列长度**:
   ```yaml
   max_length: 512  # Reduce from 1024 / 从 1024 减少
   ```

### Slow Training / 训练缓慢

**Symptoms / 症状**:
- Very slow iteration time / 迭代时间非常慢
- Low GPU utilization / GPU 利用率低

**Solutions / 解决方案**:

1. **Disable unnecessary offloading** / **禁用不必要的卸载**:
   If you have enough GPU memory, keep data on GPU.
   如果有足够的 GPU 内存，将数据保留在 GPU 上。

2. **Increase batch size** / **增加批次大小**:
   Larger batches improve GPU utilization.
   更大的批次提高 GPU 利用率。

3. **Use ZeRO-2 instead of ZeRO-3** / **使用 ZeRO-2 而不是 ZeRO-3**:
   ZeRO-3 is slower due to parameter gathering.
   ZeRO-3 由于参数收集而较慢。

4. **Check CPU-GPU communication** / **检查 CPU-GPU 通信**:
   ```json
   "pin_memory": true  // Should be enabled / 应该启用
   ```

### NaN Loss / NaN 损失

**Symptoms / 症状**:
```
Loss is NaN
```

**Solutions / 解决方案**:

1. **Use BF16 instead of FP16** / **使用 BF16 而不是 FP16**:
   ```json
   {"fp16": {"enabled": false}, "bf16": {"enabled": true}}
   ```

2. **Adjust loss scaling (FP16 only)** / **调整损失缩放（仅限 FP16）**:
   ```json
   {
     "fp16": {
       "loss_scale": 0,  // Dynamic scaling / 动态缩放
       "initial_scale_power": 10  // Reduce from 12 / 从 12 减少
     }
   }
   ```

3. **Reduce learning rate** / **降低学习率**:
   ```yaml
   learning_rate: 1e-6  # Reduce from 1e-5 / 从 1e-5 减少
   ```

4. **Enable gradient clipping** / **启用梯度裁剪**:
   ```json
   {"gradient_clipping": 1.0}
   ```

### DeepSpeed Not Found / 找不到 DeepSpeed

**Symptoms / 症状**:
```
ModuleNotFoundError: No module named 'deepspeed'
```

**Solution / 解决方案**:
```bash
pip install deepspeed

# If compilation fails, try pre-built wheels / 如果编译失败，尝试预构建的轮子
pip install deepspeed --no-build-isolation
```

### Incompatible DeepSpeed Version / DeepSpeed 版本不兼容

**Symptoms / 症状**:
```
AttributeError: module 'deepspeed' has no attribute 'xxx'
```

**Solution / 解决方案**:
```bash
# Upgrade DeepSpeed / 升级 DeepSpeed
pip install --upgrade deepspeed

# Check version / 检查版本
python -c "import deepspeed; print(deepspeed.__version__)"
# Recommended: 0.10.0 or later / 推荐：0.10.0 或更高版本
```

### NCCL Errors / NCCL 错误

**Symptoms / 症状**:
```
NCCL error: unhandled system error
```

**Solutions / 解决方案**:

1. **Set NCCL environment variables** / **设置 NCCL 环境变量**:
   ```bash
   export NCCL_DEBUG=INFO
   export NCCL_IB_DISABLE=0
   export NCCL_NET_GDR_LEVEL=2
   ```

2. **Check network connectivity** / **检查网络连接**:
   ```bash
   # Test communication between nodes / 测试节点间通信
   ping <other_node_ip>
   ```

3. **Use TCP instead of InfiniBand** / **使用 TCP 而不是 InfiniBand**:
   ```bash
   export NCCL_IB_DISABLE=1
   export NCCL_SOCKET_IFNAME=eth0  # Your network interface / 您的网络接口
   ```

---

## Example Training Scripts
## 示例训练脚本

### Complete Example: Reward Model Training with ZeRO-2
### 完整示例：使用 ZeRO-2 的奖励模型训练

**Step 1**: Update `configs/rm_training.yaml`:
**步骤 1**：更新 `configs/rm_training.yaml`：

```yaml
# Add DeepSpeed configuration / 添加 DeepSpeed 配置
deepspeed: scripts/configs/deepspeed/zero2.json

# Model and data / 模型和数据
model_path: models/sft_checkpoint
data_path: data/rm_train/rm_data.jsonl

# Training settings / 训练设置
num_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-5

# Use BF16 if you have A100/H100 / 如果有 A100/H100 使用 BF16
bf16: true
fp16: false
```

**Step 2**: Update `zero2.json` for your hardware:
**步骤 2**：根据硬件更新 `zero2.json`：

```json
{
  "bf16": {"enabled": true},  // Match your YAML config / 匹配 YAML 配置
  "fp16": {"enabled": false},
  "data_types": {"grad_accum_dtype": "fp32"}  // fp32 for bf16 / bf16 使用 fp32
}
```

**Step 3**: Run training:
**步骤 3**：运行训练：

```bash
# Single node, 4 GPUs / 单节点，4 个 GPU
torchrun --nproc_per_node=4 scripts/train_rm.py

# Or with DeepSpeed launcher / 或使用 DeepSpeed 启动器
deepspeed --num_gpus=4 scripts/train_rm.py
```

### Complete Example: DPO Training with ZeRO-3
### 完整示例：使用 ZeRO-3 的 DPO 训练

For very large models (13B+):
对于非常大的模型（13B+）：

**Step 1**: Update `configs/dpo_training.yaml`:
**步骤 1**：更新 `configs/dpo_training.yaml`：

```yaml
deepspeed: scripts/configs/deepspeed/zero3.json

model_path: models/13b_model
per_device_train_batch_size: 1  // Very small for ZeRO-3 / ZeRO-3 使用很小的值
gradient_accumulation_steps: 32  // Large accumulation / 大的累积

bf16: true
```

**Step 2**: Enable gradient checkpointing in `train_dpo.py`:
**步骤 2**：在 `train_dpo.py` 中启用梯度检查点：

```python
# After model loading / 模型加载后
model.gradient_checkpointing_enable()
```

**Step 3**: Run with 8 GPUs:
**步骤 3**：使用 8 个 GPU 运行：

```bash
torchrun --nproc_per_node=8 scripts/train_dpo.py
```

---

## Monitoring DeepSpeed Training
## 监控 DeepSpeed 训练

### Check DeepSpeed is Active / 检查 DeepSpeed 是否激活

Look for these logs:
查找这些日志：

```
[DeepSpeed] Initializing DeepSpeed ZeRO-2
[DeepSpeed] WORLD_SIZE = 4
[DeepSpeed] fp16 is enabled
```

### Monitor GPU Memory / 监控 GPU 内存

```bash
# In another terminal / 在另一个终端
watch -n 1 nvidia-smi
```

### TensorBoard Logs / TensorBoard 日志

DeepSpeed logs are automatically saved to TensorBoard:
DeepSpeed 日志自动保存到 TensorBoard：

```bash
tensorboard --logdir outputs/reward_model/logs
```

Key metrics to monitor:
要监控的关键指标：

- `train/loss`: Should decrease steadily / 应稳定下降
- `train/learning_rate`: Check LR schedule / 检查学习率调度
- `train/epoch`: Training progress / 训练进度

---

## Advanced: Custom DeepSpeed Configs
## 高级：自定义 DeepSpeed 配置

### Create a Custom Config / 创建自定义配置

You can copy and modify existing configs:
您可以复制并修改现有配置：

```bash
# Copy zero2.json / 复制 zero2.json
cp scripts/configs/deepspeed/zero2.json scripts/configs/deepspeed/my_config.json

# Edit as needed / 根据需要编辑
nano scripts/configs/deepspeed/my_config.json
```

### Common Customizations / 常见自定义

**1. Adjust bucket sizes for faster communication / 调整桶大小以加快通信**:
```json
{
  "allgather_bucket_size": 1e9,  // Larger for faster networks / 更大用于更快的网络
  "reduce_bucket_size": 1e9
}
```

**2. Enable ZeRO-Offload to NVMe / 启用 ZeRO-Offload 到 NVMe**:
```json
{
  "offload_optimizer": {
    "device": "nvme",
    "nvme_path": "/local_nvme",
    "pin_memory": true
  }
}
```

**3. Customize optimizer / 自定义优化器**:
```json
{
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-5,
      "betas": [0.9, 0.98],
      "eps": 1e-6,
      "weight_decay": 0.1
    }
  }
}
```

---

## References
## 参考资料

### Official Documentation / 官方文档
- [DeepSpeed Official Website](https://www.deepspeed.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- [HuggingFace DeepSpeed Integration](https://huggingface.co/docs/transformers/main/en/main_classes/deepspeed)

### Tutorials / 教程
- [DeepSpeed Tutorial](https://www.deepspeed.ai/tutorials/)
- [ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [HuggingFace + DeepSpeed](https://huggingface.co/docs/transformers/main_classes/deepspeed)

### Community / 社区
- [DeepSpeed Issues](https://github.com/microsoft/DeepSpeed/issues)
- [HuggingFace Forum](https://discuss.huggingface.co/)

---

## FAQ

### Q: Do I need to change my training code?
### Q: 我需要更改训练代码吗？

**A**: No! If you're using HuggingFace Trainer, just add the `deepspeed` parameter to your config. The Trainer handles everything automatically.

**A**: 不需要！如果您使用 HuggingFace Trainer，只需在配置中添加 `deepspeed` 参数。Trainer 会自动处理一切。

### Q: Can I use DeepSpeed with single GPU?
### Q: 我可以在单个 GPU 上使用 DeepSpeed 吗？

**A**: Yes, but ZeRO stages 1-2 won't provide benefits. You can use ZeRO-3 with CPU offloading for very large models on single GPU.

**A**: 可以，但 ZeRO 阶段 1-2 不会提供优势。您可以在单个 GPU 上使用 ZeRO-3 和 CPU 卸载来训练非常大的模型。

### Q: Which is faster, ZeRO-2 or ZeRO-3?
### Q: ZeRO-2 和 ZeRO-3 哪个更快？

**A**: ZeRO-2 is generally faster because it doesn't partition model parameters. Use ZeRO-3 only when you need the extra memory savings.

**A**: ZeRO-2 通常更快，因为它不对模型参数进行分片。只有在需要额外的内存节省时才使用 ZeRO-3。

### Q: Can I mix DeepSpeed configs?
### Q: 我可以混合使用 DeepSpeed 配置吗？

**A**: You can create hybrid configs by combining features from both configs. Just make sure the settings are compatible.

**A**: 您可以通过组合两个配置的功能来创建混合配置。只需确保设置兼容即可。

### Q: How do I know if DeepSpeed is working?
### Q: 我如何知道 DeepSpeed 是否工作？

**A**: Check the logs at the start of training. You should see DeepSpeed initialization messages. Also check GPU memory usage - it should be lower than without DeepSpeed.

**A**: 检查训练开始时的日志。您应该看到 DeepSpeed 初始化消息。还要检查 GPU 内存使用情况 - 它应该比不使用 DeepSpeed 时更低。

---

## Contributing / 贡献

If you find issues with these configs or have improvements, please:
如果您发现这些配置有问题或有改进建议，请：

1. Open an issue / 提出问题
2. Submit a pull request / 提交拉取请求
3. Share your experience / 分享您的经验

---

## License / 许可证

These configurations are provided as-is under the MIT License.
这些配置按原样提供，采用 MIT 许可证。

---

**Happy Training! / 训练愉快！**

For questions or support, please open an issue in the repository.
如有问题或需要支持，请在仓库中提出问题。
