"""
Training Modules
训练模块

This package contains training scripts for various stages of LLM training:
本包包含 LLM 训练各个阶段的训练脚本:

- Pretraining (pretrain.py) / 预训练
- Supervised Fine-Tuning (sft.py) / 监督微调
- Reward Model Training (rm.py) / 奖励模型训练
- DPO Training (dpo.py) / DPO 训练
- RLHF Datasets (datasets.py) / RLHF 数据集
- Adapters / test helpers (adapters.py) / 适配器
"""

__all__ = [
    "pretrain",
    "sft",
    "rm",
    "dpo",
    "datasets",
    "adapters",
]
