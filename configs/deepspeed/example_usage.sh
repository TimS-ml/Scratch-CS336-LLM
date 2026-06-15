#!/bin/bash
# ============================================================================
# DeepSpeed Training Example Script
# DeepSpeed 训练示例脚本
# ============================================================================
#
# This script demonstrates how to use DeepSpeed ZeRO configurations
# for training reward models and DPO models.
#
# 此脚本演示如何使用 DeepSpeed ZeRO 配置来训练奖励模型和 DPO 模型。
#
# Usage / 用法:
#   bash configs/deepspeed/example_usage.sh
#
# ============================================================================

set -e  # Exit on error / 遇到错误时退出

# ============================================================================
# Configuration / 配置
# ============================================================================

# Project root / 项目根目录
PROJECT_ROOT="/home/user/LLM-from-Scratch"
cd $PROJECT_ROOT

# Number of GPUs / GPU 数量
NUM_GPUS=4

# DeepSpeed config to use / 要使用的 DeepSpeed 配置
# Options: zero2.json, zero3.json
DS_CONFIG="zero2.json"

# Training script / 训练脚本
TRAINING_SCRIPT="scripts/train_rm.py"  # or train_dpo.py

# ============================================================================
# Example 1: Reward Model Training with DeepSpeed ZeRO-2
# 示例 1：使用 DeepSpeed ZeRO-2 训练奖励模型
# ============================================================================

echo "============================================================================"
echo "Example 1: Reward Model Training with DeepSpeed ZeRO-2"
echo "示例 1：使用 DeepSpeed ZeRO-2 训练奖励模型"
echo "============================================================================"

# First, update the config file to enable DeepSpeed
# 首先，更新配置文件以启用 DeepSpeed
cat > /tmp/rm_deepspeed_config.yaml <<EOF
# Temporary RM config with DeepSpeed enabled
# 启用 DeepSpeed 的临时 RM 配置

model_path: models/sft_checkpoint
data_path: data/rm_train/rm_data.jsonl
output_dir: outputs/reward_model_deepspeed

# Training settings
num_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-5

# Mixed precision (required for DeepSpeed)
bf16: true
fp16: false

# DeepSpeed configuration
deepspeed: configs/deepspeed/${DS_CONFIG}

# Other settings
max_length: 512
seed: 42
EOF

echo "Config file created at: /tmp/rm_deepspeed_config.yaml"
echo "配置文件已创建于: /tmp/rm_deepspeed_config.yaml"
echo ""

# Run training with torchrun (recommended)
# 使用 torchrun 运行训练（推荐）
echo "Running training command / 运行训练命令:"
echo "torchrun --nproc_per_node=${NUM_GPUS} ${TRAINING_SCRIPT}"
echo ""

# Uncomment to actually run training / 取消注释以实际运行训练
# torchrun --nproc_per_node=${NUM_GPUS} ${TRAINING_SCRIPT}

echo "To run training, uncomment the torchrun command above"
echo "要运行训练，请取消注释上面的 torchrun 命令"
echo ""

# ============================================================================
# Example 2: DPO Training with DeepSpeed ZeRO-3 (Large Models)
# 示例 2：使用 DeepSpeed ZeRO-3 训练 DPO（大型模型）
# ============================================================================

echo "============================================================================"
echo "Example 2: DPO Training with DeepSpeed ZeRO-3 (Large Models)"
echo "示例 2：使用 DeepSpeed ZeRO-3 训练 DPO（大型模型）"
echo "============================================================================"

# For very large models, use ZeRO-3
# 对于非常大的模型，使用 ZeRO-3
DS_CONFIG_LARGE="zero3.json"

cat > /tmp/dpo_deepspeed_config.yaml <<EOF
# Temporary DPO config with DeepSpeed ZeRO-3
# 启用 DeepSpeed ZeRO-3 的临时 DPO 配置

model:
  model_name_or_path: models/large_model_7b
  trust_remote_code: true
  use_cache: false

data:
  train_dataset_path: data/dpo_train.jsonl
  eval_dataset_path: data/dpo_eval.jsonl
  max_length: 512
  max_prompt_length: 256

training:
  output_dir: outputs/dpo_deepspeed
  num_train_epochs: 3
  per_device_train_batch_size: 1  # Very small for ZeRO-3
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 32  # Large accumulation to compensate
  learning_rate: 5e-7

  # Mixed precision
  bf16: true
  fp16: false

  # DeepSpeed ZeRO-3 with offloading
  deepspeed: configs/deepspeed/${DS_CONFIG_LARGE}

  # Other settings
  save_strategy: steps
  save_steps: 500
  eval_strategy: steps
  eval_steps: 500
  logging_steps: 10
  seed: 42

dpo:
  beta: 0.1
  loss_type: sigmoid
EOF

echo "Config file created at: /tmp/dpo_deepspeed_config.yaml"
echo "配置文件已创建于: /tmp/dpo_deepspeed_config.yaml"
echo ""

# ============================================================================
# Example 3: Multi-Node Training with DeepSpeed
# 示例 3：使用 DeepSpeed 进行多节点训练
# ============================================================================

echo "============================================================================"
echo "Example 3: Multi-Node Training with DeepSpeed"
echo "示例 3：使用 DeepSpeed 进行多节点训练"
echo "============================================================================"

cat > /tmp/run_multinode_deepspeed.sh <<'EOF'
#!/bin/bash
# Multi-node DeepSpeed training script
# 多节点 DeepSpeed 训练脚本

# Set these environment variables on each node
# 在每个节点上设置这些环境变量
export MASTER_ADDR="10.0.0.1"  # IP of node 0
export MASTER_PORT="29500"
export WORLD_SIZE=2  # Total number of nodes
export RANK=0  # Node rank (0 for master, 1, 2, ... for workers)

# Number of GPUs per node
GPUS_PER_NODE=8

# Run training
torchrun \
    --nnodes=${WORLD_SIZE} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_rm.py

# Alternative: Using DeepSpeed launcher
# 或者：使用 DeepSpeed 启动器
# deepspeed \
#     --num_gpus=${GPUS_PER_NODE} \
#     --num_nodes=${WORLD_SIZE} \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     scripts/train_rm.py
EOF

chmod +x /tmp/run_multinode_deepspeed.sh

echo "Multi-node script created at: /tmp/run_multinode_deepspeed.sh"
echo "多节点脚本已创建于: /tmp/run_multinode_deepspeed.sh"
echo ""

# ============================================================================
# Example 4: Monitoring DeepSpeed Training
# 示例 4：监控 DeepSpeed 训练
# ============================================================================

echo "============================================================================"
echo "Example 4: Monitoring DeepSpeed Training"
echo "示例 4：监控 DeepSpeed 训练"
echo "============================================================================"

echo "1. Check GPU usage in real-time:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "2. Monitor TensorBoard logs:"
echo "   tensorboard --logdir outputs/reward_model_deepspeed/logs"
echo ""
echo "3. Check DeepSpeed initialization in logs:"
echo "   grep -i 'deepspeed' outputs/reward_model_deepspeed/train.log"
echo ""

# ============================================================================
# Tips and Best Practices / 提示和最佳实践
# ============================================================================

echo "============================================================================"
echo "Tips and Best Practices / 提示和最佳实践"
echo "============================================================================"

cat <<EOF

1. Choose the right ZeRO stage / 选择正确的 ZeRO 阶段:
   - ZeRO-2: Models up to ~7B, faster training
   - ZeRO-3: Models 7B+, maximum memory savings

2. Always enable mixed precision / 始终启用混合精度:
   - Use bf16 for A100/H100 GPUs
   - Use fp16 for V100/T4 GPUs

3. Adjust batch size / 调整批次大小:
   - Start small and increase until OOM
   - Use gradient accumulation to maintain effective batch size

4. Monitor memory usage / 监控内存使用:
   - nvidia-smi during training
   - Check for GPU memory fragmentation

5. Troubleshooting / 故障排除:
   - If training is slow: Check if CPU offloading is enabled
   - If OOM: Switch to ZeRO-3 or reduce batch size
   - If NaN loss: Use bf16 instead of fp16

6. See full documentation / 查看完整文档:
   - configs/deepspeed/README.md

EOF

echo "============================================================================"
echo "Example script completed successfully!"
echo "示例脚本已成功完成！"
echo "============================================================================"
