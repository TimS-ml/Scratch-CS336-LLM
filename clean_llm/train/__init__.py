"""
Training Modules
训练模块

This package contains training scripts for various stages of LLM training:
本包包含 LLM 训练各个阶段的训练脚本:

- Pretraining (pretrain.py) / 预训练
- Supervised Fine-Tuning (sft.py) / 监督微调
- Reward Model Training (rm_train.py) / 奖励模型训练
- DPO Training (dpo_train.py) / DPO 训练
- RLHF Datasets (rlhf_datasets.py) / RLHF 数据集
"""

__all__ = [
    "dpo_train",
    "rm_train",
    "sft",
    "pretrain",
    "rlhf_datasets",
]
