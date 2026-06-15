"""
Data Processors for LLM Training
数据处理器模块

This module provides processors for different types of LLM training data:
本模块提供不同类型的LLM训练数据处理器：

- sft_processor: Supervised Fine-Tuning data processing / 监督微调数据处理
- pretrain_processor: Pre-training data processing / 预训练数据处理
- rm_processor: Reward Model data processing / 奖励模型数据处理
"""

from .sft_processor import (
    SFTDataProcessor,
    process_belle_dataset,
    process_firefly_dataset,
    process_tigerbot_sft_dataset,
)

from .pretrain_processor import (
    PretrainDataProcessor,
    process_wikipedia_dataset,
    process_tigerbot_pretrain_dataset,
    process_baidu_baike_dataset,
    merge_binary_files,
)

from .rm_processor import (
    RMDataProcessor,
    process_cvalues_dataset,
    process_rlhf_dataset,
    process_zhihu_rlhf_dataset,
)

__all__ = [
    # SFT processors / SFT处理器
    "SFTDataProcessor",
    "process_belle_dataset",
    "process_firefly_dataset",
    "process_tigerbot_sft_dataset",
    # Pretrain processors / 预训练处理器
    "PretrainDataProcessor",
    "process_wikipedia_dataset",
    "process_tigerbot_pretrain_dataset",
    "process_baidu_baike_dataset",
    "merge_binary_files",
    # RM processors / 奖励模型处理器
    "RMDataProcessor",
    "process_cvalues_dataset",
    "process_rlhf_dataset",
    "process_zhihu_rlhf_dataset",
]
