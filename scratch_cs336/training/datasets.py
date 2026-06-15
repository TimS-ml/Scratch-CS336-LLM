"""
RLHF Datasets Module
RLHF 数据集模块

This module contains dataset classes for RLHF (Reinforcement Learning from Human Feedback) training,
including pretraining, reward modeling, and reinforcement learning (PPO/DPO) datasets.

本模块包含用于 RLHF（基于人类反馈的强化学习）训练的数据集类，
包括预训练、奖励模型和强化学习（PPO/DPO）数据集。

Classes:
    - PTMDatasetMap: Memory-mapped pretraining dataset for efficient large-scale data loading
                     基于内存映射的预训练数据集，用于高效加载大规模数据
    - RMDataset: Reward model dataset for training preference-based reward models
                 奖励模型数据集，用于训练基于偏好的奖励模型
    - RLDataset: Reinforcement learning dataset for PPO training
                 强化学习数据集，用于 PPO 训练

Functions:
    - load_dpo_dataset: Load and preprocess dataset for DPO (Direct Preference Optimization) training
                        加载并预处理用于 DPO（直接偏好优化）训练的数据集
    - load_ppo_dataset: Load and preprocess dataset for PPO (Proximal Policy Optimization) training
                        加载并预处理用于 PPO（近端策略优化）训练的数据集
"""

import os
import random
import hashlib
import jsonlines
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any
from torch.utils.data import Dataset
from datasets import load_dataset
import datasets


class PTMDatasetMap(Dataset):
    """
    Memory-mapped Pretraining Dataset
    基于内存映射的预训练数据集

    This dataset class uses memory mapping to handle large pretraining data files efficiently,
    minimizing memory usage while providing fast random access to training samples.

    该数据集类使用内存映射技术高效处理大型预训练数据文件，
    在提供快速随机访问训练样本的同时最小化内存使用。

    Features / 特性:
        - Memory-efficient: Only loads required data chunks into memory
          内存高效：仅加载所需的数据块到内存
        - Supports multiple data files: Can handle multiple binary files seamlessly
          支持多文件：可以无缝处理多个二进制文件
        - Random shuffling: Provides shuffled access to samples for better training
          随机打乱：为更好的训练提供打乱的样本访问

    Args:
        data_path_list (List[str]): List of paths to binary data files (.bin format)
                                     二进制数据文件路径列表（.bin 格式）
        max_length (int): Maximum sequence length for each sample
                         每个样本的最大序列长度

    Data Format / 数据格式:
        - Input files should be binary files containing uint16 tokens
          输入文件应为包含 uint16 token 的二进制文件
        - Each file is memory-mapped as a 2D array of shape (num_samples, max_length)
          每个文件被映射为形状为 (样本数, 最大长度) 的二维数组

    Returns:
        Dictionary containing / 返回包含以下内容的字典:
            - input_ids (torch.LongTensor): Input token IDs / 输入 token ID
            - labels (torch.LongTensor): Target token IDs (copy of input_ids for causal LM)
                                        目标 token ID（用于因果语言模型的 input_ids 副本）

    Example / 示例:
        >>> # Prepare binary data files / 准备二进制数据文件
        >>> data_files = [
        ...     '/path/to/pretrain_data_part1.bin',
        ...     '/path/to/pretrain_data_part2.bin'
        ... ]
        >>>
        >>> # Create dataset / 创建数据集
        >>> dataset = PTMDatasetMap(
        ...     data_path_list=data_files,
        ...     max_length=512
        ... )
        >>>
        >>> # Use with DataLoader / 配合 DataLoader 使用
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(
        ...     dataset,
        ...     batch_size=32,
        ...     shuffle=False,  # Dataset handles internal shuffling
        ...     num_workers=4
        ... )
        >>>
        >>> # Iterate over batches / 遍历批次
        >>> for batch in dataloader:
        ...     input_ids = batch['input_ids']  # Shape: (batch_size, max_length)
        ...     labels = batch['labels']        # Shape: (batch_size, max_length)
        ...     # Training code here / 训练代码
    """

    def __init__(self, data_path_list: List[str], max_length: int = 512):
        super(PTMDatasetMap, self).__init__()

        # Initialize data structures / 初始化数据结构
        self.data: List[np.memmap] = []          # Store memory-mapped data for each file
                                                  # 存储每个文件的内存映射数据
        self.index_map: Dict[int, tuple] = {}    # Map global index to (file_idx, local_idx)
                                                  # 将全局索引映射到 (文件索引, 局部索引)
        self.token_size: int = 0                 # Total number of tokens across all files
                                                  # 所有文件的 token 总数
        self.data_size: int = 0                  # Total number of samples
                                                  # 样本总数

        # Process each data file / 处理每个数据文件
        for idx, file_path in enumerate(data_path_list):
            # Get file size in bytes and convert to number of tokens
            # 获取文件字节大小并转换为 token 数量
            with open(file_path, 'r') as f:
                nbytes = f.seek(0, 2)  # Seek to end of file / 定位到文件末尾
                flen = f.tell() // np.dtype('uint16').itemsize  # Number of tokens

            # Update statistics and index mapping
            # 更新统计信息和索引映射
            self.token_size += flen
            num_samples_in_file = flen // max_length

            # Create index mapping: global_idx -> (file_idx, sample_idx_in_file)
            # 创建索引映射：全局索引 -> (文件索引, 文件内样本索引)
            self.index_map.update({
                self.data_size + i: (idx, i)
                for i in range(num_samples_in_file)
            })
            self.data_size += num_samples_in_file

            # Memory-map the file for efficient access
            # 对文件进行内存映射以实现高效访问
            mmap_array = np.memmap(
                file_path,
                dtype=np.dtype('uint16'),
                shape=(num_samples_in_file, max_length)
            )
            self.data.append(mmap_array)

        print(f'Total token size: {self.token_size:,} tokens, '
              f'Data sample size: [{self.data_size:,}, {max_length}]')
        print(f'总 token 数量: {self.token_size:,} tokens, '
              f'数据样本规模: [{self.data_size:,}, {max_length}]')

        # Initialize shuffled indices for random access
        # 初始化打乱的索引以实现随机访问
        self.shuffled_indices = list(self.index_map.keys())
        random.shuffle(self.shuffled_indices)

    def __len__(self) -> int:
        """Return the total number of samples / 返回样本总数"""
        return self.data_size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index / 根据索引获取样本

        Args:
            index (int): Sample index (will be mapped to shuffled index)
                        样本索引（将被映射到打乱后的索引）

        Returns:
            Dictionary with input_ids and labels / 包含 input_ids 和 labels 的字典
        """
        # Map shuffled index to actual data location
        # 将打乱的索引映射到实际数据位置
        real_index = self.shuffled_indices[index]
        file_idx, sample_idx = self.index_map[real_index]

        # Retrieve sample from memory-mapped array
        # 从内存映射数组中检索样本
        sample = self.data[file_idx][sample_idx]
        X = np.array(sample).astype(np.int64)
        input_ids = torch.LongTensor(X)

        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),  # For causal LM, labels = input_ids shifted
                                          # 对于因果语言模型，labels = 移位的 input_ids
        }


class RMDataset(Dataset):
    """
    Reward Model Dataset
    奖励模型数据集

    Dataset for training reward models using paired preference data (chosen vs rejected responses).
    This is a key component in RLHF for learning human preferences.

    用于使用成对偏好数据（选择的回答 vs 拒绝的回答）训练奖励模型的数据集。
    这是 RLHF 中学习人类偏好的关键组件。

    Features / 特性:
        - Paired comparisons: Each sample contains a prompt with both chosen and rejected responses
          成对比较：每个样本包含一个提示和选择/拒绝的回答
        - Automatic padding: Handles variable-length sequences with proper padding
          自动填充：使用适当的填充处理可变长度序列
        - Tokenizer agnostic: Works with both ChatGLM and HuggingFace tokenizers
          分词器无关：适用于 ChatGLM 和 HuggingFace 分词器

    Args:
        data_path (str): Path to JSONL file containing preference data
                        包含偏好数据的 JSONL 文件路径
        tokenizer: Tokenizer instance (ChatGLM or HuggingFace)
                  分词器实例（ChatGLM 或 HuggingFace）
        max_length (int): Maximum sequence length
                         最大序列长度
        system (str): System prompt for the conversation
                     对话的系统提示

    Data Format / 数据格式:
        Each line in JSONL should contain / JSONL 中每行应包含:
        {
            "prompt": "User question / 用户问题",
            "chosen": "Preferred response / 偏好回答",
            "rejected": "Rejected response / 拒绝的回答"
        }

    Returns:
        Dictionary containing / 返回包含以下内容的字典:
            - input_ids_j (torch.LongTensor): Token IDs for chosen response
                                             选择回答的 token ID
            - attention_mask_j (torch.LongTensor): Attention mask for chosen response
                                                   选择回答的注意力掩码
            - input_ids_k (torch.LongTensor): Token IDs for rejected response
                                             拒绝回答的 token ID
            - attention_mask_k (torch.LongTensor): Attention mask for rejected response
                                                   拒绝回答的注意力掩码

    Example / 示例:
        >>> from transformers import AutoTokenizer
        >>>
        >>> # Using HuggingFace tokenizer / 使用 HuggingFace 分词器
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>>
        >>> # Or using ChatGLM tokenizer / 或使用 ChatGLM 分词器
        >>> # from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
        >>> # tokenizer = ChatGLMTokenizer(vocab_file='path/to/tokenizer.model')
        >>>
        >>> # Create dataset / 创建数据集
        >>> dataset = RMDataset(
        ...     data_path='data/reward_model/rm_data.jsonl',
        ...     tokenizer=tokenizer,
        ...     max_length=512,
        ...     system="You are a helpful assistant."
        ... )
        >>>
        >>> # Use with DataLoader / 配合 DataLoader 使用
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        >>>
        >>> # Training loop / 训练循环
        >>> for batch in dataloader:
        ...     # batch contains paired comparisons
        ...     # batch 包含成对的比较数据
        ...     chosen_ids = batch['input_ids_j']
        ...     rejected_ids = batch['input_ids_k']
        ...     # Compute reward scores / 计算奖励分数

    Notes / 注意事项:
        - For ChatGLM tokenizer: Uses special tokens like <|system|>, <|user|>, <|assistant|>
          对于 ChatGLM 分词器：使用特殊 token 如 <|system|>, <|user|>, <|assistant|>
        - For HuggingFace tokenizer: May need to adjust special token formatting
          对于 HuggingFace 分词器：可能需要调整特殊 token 格式
        - Samples exceeding max_length are skipped automatically
          超过最大长度的样本会被自动跳过
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 256,
        system: str = "你是由wdndev开发的个人助手。",
    ):
        super(RMDataset, self).__init__()

        # Load data from JSONL file / 从 JSONL 文件加载数据
        self.data = []
        with jsonlines.open(data_path) as reader:
            for obj in reader:
                self.data.append(obj)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system = system

        # Display first 5 samples for verification / 显示前 5 个样本以验证
        for ex in self.data[:5]:
            self.preprocessing(ex, debug=True)

        print(f"RM Data loading is completed. Data length: {len(self.data)}")
        print(f"RM 数据加载完成。数据长度: {len(self.data)}")

    def preprocessing(
        self,
        example: Dict[str, str],
        debug: bool = False
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Preprocess a single example into tokenized format
        将单个样本预处理为分词格式

        Creates two complete sequences: one with chosen response, one with rejected response.
        Both sequences follow the format:
            [gMASK]sop <|system|>
            {system_prompt}
            <|user|>
            {user_prompt}
            <|assistant|>
            {response}

        创建两个完整序列：一个包含选择的回答，一个包含拒绝的回答。
        两个序列都遵循以下格式:
            [gMASK]sop <|system|>
            {系统提示}
            <|user|>
            {用户提示}
            <|assistant|>
            {回答}

        Args:
            example: Dictionary with 'prompt', 'chosen', 'rejected'
                    包含 'prompt', 'chosen', 'rejected' 的字典
            debug: Whether to print debug information
                  是否打印调试信息

        Returns:
            Tokenized dictionary or None if sequence too long
            分词后的字典，如果序列过长则返回 None
        """
        prompt_txt = self.system
        user_txt = example["prompt"]
        assistant_chosen_txt = example["chosen"]
        assistant_rejected_txt = example["rejected"]

        # Format instruction part (same for both chosen and rejected)
        # 格式化指令部分（选择和拒绝的回答共用）
        instruction_text = "\n".join([
            "<|system|>", prompt_txt.strip(),
            "<|user|>", user_txt.strip(),
            "<|assistant|>"
        ]).strip() + "\n"

        instruction = self.tokenizer.encode(
            text=instruction_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )

        # Process chosen response / 处理选择的回答
        response_j = self.tokenizer.encode(
            assistant_chosen_txt.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        input_ids_j = instruction + response_j + [self.tokenizer.eos_token_id]

        # Pad to max_length / 填充到最大长度
        pad_len_j = self.max_length - len(input_ids_j)
        input_ids_j += [self.tokenizer.pad_token_id] * pad_len_j

        # Process rejected response / 处理拒绝的回答
        response_k = self.tokenizer.encode(
            assistant_rejected_txt.strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length
        )
        input_ids_k = instruction + response_k + [self.tokenizer.eos_token_id]

        # Pad to max_length / 填充到最大长度
        pad_len_k = self.max_length - len(input_ids_k)
        input_ids_k += [self.tokenizer.pad_token_id] * pad_len_k

        # Debug output / 调试输出
        if debug:
            print("Chosen response / 选择的回答:")
            print(self.tokenizer.decode(input_ids_j))
            print("******************")
            print("Rejected response / 拒绝的回答:")
            print(self.tokenizer.decode(input_ids_k))
            print("-------------------------------")

        # Skip if either sequence exceeds max_length
        # 如果任一序列超过最大长度则跳过
        if len(input_ids_j) > self.max_length or len(input_ids_k) > self.max_length:
            return None

        # Convert to tensors and create attention masks
        # 转换为张量并创建注意力掩码
        input_ids_j = torch.LongTensor(input_ids_j)
        attention_mask_j = input_ids_j.ne(self.tokenizer.pad_token_id)

        input_ids_k = torch.LongTensor(input_ids_k)
        attention_mask_k = input_ids_k.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids_j": input_ids_j,
            "attention_mask_j": attention_mask_j,
            "input_ids_k": input_ids_k,
            "attention_mask_k": attention_mask_k,
        }

    def __len__(self) -> int:
        """Return the total number of samples / 返回样本总数"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample by index, skipping invalid samples
        根据索引获取样本，跳过无效样本

        If a sample is too long, automatically moves to the next valid sample.
        如果样本过长，自动移动到下一个有效样本。
        """
        processed_example = self.preprocessing(self.data[idx])
        while processed_example is None:
            idx = (idx + 1) % len(self.data)  # Move to next sample / 移动到下一个样本
            processed_example = self.preprocessing(self.data[idx])

        return processed_example


class RLDataset(Dataset):
    """
    Reinforcement Learning Dataset (for PPO)
    强化学习数据集（用于 PPO）

    Dataset for reinforcement learning training, specifically designed for PPO (Proximal Policy Optimization).
    Contains only prompts without responses, as responses will be generated by the model during training.

    用于强化学习训练的数据集，专为 PPO（近端策略优化）设计。
    仅包含提示而不包含回答，因为回答将在训练期间由模型生成。

    Features / 特性:
        - Deduplication: Automatically removes duplicate prompts using hash-based filtering
          去重：使用基于哈希的过滤自动删除重复提示
        - Query generation: Creates formatted query strings for model generation
          查询生成：创建用于模型生成的格式化查询字符串
        - Flexible tokenization: Works with various tokenizer implementations
          灵活的分词：适用于各种分词器实现

    Args:
        data_path (str): Path to JSONL file containing prompts
                        包含提示的 JSONL 文件路径
        tokenizer: Tokenizer instance (ChatGLM or HuggingFace)
                  分词器实例（ChatGLM 或 HuggingFace）
        max_length (int): Maximum sequence length
                         最大序列长度
        system (str): System prompt for the conversation
                     对话的系统提示

    Data Format / 数据格式:
        Each line in JSONL should contain / JSONL 中每行应包含:
        {
            "prompt": "User question for RL training / 用于 RL 训练的用户问题"
        }

    Returns:
        Dictionary containing / 返回包含以下内容的字典:
            - query (str): Formatted query string for text generation
                          用于文本生成的格式化查询字符串
            - input_ids (torch.LongTensor): Tokenized query (without padding)
                                           分词后的查询（不含填充）

    Example / 示例:
        >>> from transformers import AutoTokenizer
        >>>
        >>> # Create tokenizer / 创建分词器
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>>
        >>> # Create RL dataset / 创建 RL 数据集
        >>> dataset = RLDataset(
        ...     data_path='data/rl_train/prompts.jsonl',
        ...     tokenizer=tokenizer,
        ...     max_length=256,
        ...     system="You are a helpful assistant."
        ... )
        >>>
        >>> # Use in PPO training / 在 PPO 训练中使用
        >>> from torch.utils.data import DataLoader
        >>> dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        >>>
        >>> # PPO training loop / PPO 训练循环
        >>> for batch in dataloader:
        ...     queries = batch['query']  # Text queries for generation
        ...     query_tensors = batch['input_ids']  # Tokenized queries
        ...
        ...     # Generate responses using the policy model
        ...     # 使用策略模型生成回答
        ...     # responses = model.generate(query_tensors)
        ...
        ...     # Compute rewards using reward model
        ...     # 使用奖励模型计算奖励
        ...     # rewards = reward_model(query_tensors, responses)
        ...
        ...     # PPO update
        ...     # PPO 更新

    Notes / 注意事项:
        - Prompts are deduplicated using MD5 hashing
          使用 MD5 哈希对提示进行去重
        - No padding is applied to input_ids (padding handled by collator)
          input_ids 不进行填充（由 collator 处理填充）
        - Query strings include full conversation template
          查询字符串包含完整的对话模板
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 256,
        system: str = "你是由wdndev开发的个人助手。",
    ):
        super(RLDataset, self).__init__()

        # Load data with deduplication / 加载数据并去重
        # Since we only need prompts, there may be many duplicates
        # 因为只需要提示，所以可能有很多重复
        self.data = []
        seen_hashes = set()

        with jsonlines.open(data_path) as reader:
            for obj in reader:
                prompt = obj["prompt"]
                # Generate hash for deduplication / 生成哈希用于去重
                prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

                # Skip if already seen / 如果已经见过则跳过
                if prompt_hash in seen_hashes:
                    continue

                self.data.append(obj)
                seen_hashes.add(prompt_hash)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system = system

        # Display first 5 samples for verification / 显示前 5 个样本以验证
        for ex in self.data[:5]:
            self.preprocessing(ex, debug=True)

        print(f"RL Data loading is completed. Data length: {len(self.data)}")
        print(f"RL 数据加载完成。数据长度: {len(self.data)}")

    def preprocessing(
        self,
        example: Dict[str, str],
        debug: bool = False
    ) -> Optional[Dict[str, Union[str, torch.Tensor]]]:
        """
        Preprocess a single prompt into query format
        将单个提示预处理为查询格式

        Creates a formatted query string and tokenizes it for model input.
        The query ends with <|assistant|> token, ready for the model to generate a response.

        创建格式化的查询字符串并将其分词以供模型输入。
        查询以 <|assistant|> token 结尾，准备让模型生成回答。

        Args:
            example: Dictionary with 'prompt'
                    包含 'prompt' 的字典
            debug: Whether to print debug information
                  是否打印调试信息

        Returns:
            Dictionary with query string and tokenized input_ids
            包含查询字符串和分词后的 input_ids 的字典
        """
        prompt_txt = self.system
        user_txt = example["prompt"]

        # Format query string for generation
        # 格式化用于生成的查询字符串
        # Note: For ChatGLM, this includes [gMASK]sop special token
        # 注意：对于 ChatGLM，这包括 [gMASK]sop 特殊 token
        query = "[gMASK]sop <|system|>\n" + prompt_txt + "\n" + \
                "<|user|>\n" + user_txt.strip() + "\n" + "<|assistant|>\n"

        # Tokenize the instruction part
        # 对指令部分进行分词
        instruction_text = "\n".join([
            "<|system|>", prompt_txt.strip(),
            "<|user|>", user_txt.strip(),
            "<|assistant|>"
        ]).strip() + "\n"

        instruction = self.tokenizer.encode(
            text=instruction_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length
        )

        if debug:
            print("Query string / 查询字符串:", query)
            token_query = self.tokenizer.decode(instruction)
            print("Tokenized query / 分词后的查询:", token_query)
            print(f"Original length / 原始长度: {len(query)}")
            print(f"Tokenized length / 分词后长度: {len(token_query)}")
            print("-------------------------------")

        # Convert to tensor (no padding for PPO)
        # 转换为张量（PPO 不需要填充）
        input_ids = torch.LongTensor(instruction)

        return {
            "query": query,           # Text query for reference / 用于参考的文本查询
            "input_ids": input_ids,   # Tokenized input / 分词后的输入
        }

    def __len__(self) -> int:
        """Return the total number of samples / 返回样本总数"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """
        Get a sample by index
        根据索引获取样本

        Unlike RMDataset, RLDataset samples should rarely be None since they're shorter.
        Still includes fallback logic for consistency.

        与 RMDataset 不同，RLDataset 样本很少为 None，因为它们更短。
        仍然包含回退逻辑以保持一致性。
        """
        processed_example = self.preprocessing(self.data[idx])
        while processed_example is None:
            idx = (idx + 1) % len(self.data)  # Move to next sample / 移动到下一个样本
            processed_example = self.preprocessing(self.data[idx])

        return processed_example


def load_dpo_dataset(
    data_path: str,
    max_length: int = 256,
    sanity_check: bool = False,
    num_proc: int = 24,
    system: str = "你是由wdndev开发的个人助手。",
) -> datasets.Dataset:
    """
    Load and preprocess dataset for DPO (Direct Preference Optimization) training
    加载并预处理用于 DPO（直接偏好优化）训练的数据集

    DPO is an alternative to PPO that directly optimizes the policy using preference data
    without requiring a separate reward model or value function.

    DPO 是 PPO 的替代方案，直接使用偏好数据优化策略，
    无需单独的奖励模型或价值函数。

    This function loads a JSONL dataset and formats it for DPO training using HuggingFace datasets.
    The dataset is processed in parallel for efficiency.

    此函数加载 JSONL 数据集并使用 HuggingFace datasets 将其格式化为 DPO 训练格式。
    数据集并行处理以提高效率。

    Args:
        data_path (str): Path to JSONL file with preference data
                        包含偏好数据的 JSONL 文件路径
        max_length (int): Maximum total sequence length (prompt + response)
                         最大总序列长度（提示 + 回答）
        sanity_check (bool): If True, only use first 1000 samples for quick testing
                            如果为 True，仅使用前 1000 个样本进行快速测试
        num_proc (int): Number of processes for parallel data processing
                       并行数据处理的进程数
        system (str): System prompt for conversations
                     对话的系统提示

    Data Format / 数据格式:
        Input JSONL format / 输入 JSONL 格式:
        {
            "prompt": "User question / 用户问题",
            "chosen": "Preferred response / 偏好回答",
            "rejected": "Rejected response / 拒绝的回答"
        }

        Output dataset format / 输出数据集格式:
        {
            "prompt": "Formatted prompt with system and user / 包含系统和用户的格式化提示",
            "chosen": "Chosen response / 选择的回答",
            "rejected": "Rejected response / 拒绝的回答"
        }

    Returns:
        datasets.Dataset: Preprocessed HuggingFace Dataset ready for DPO training
                         预处理后的 HuggingFace 数据集，可用于 DPO 训练

    Example / 示例:
        >>> # Load DPO dataset / 加载 DPO 数据集
        >>> dpo_dataset = load_dpo_dataset(
        ...     data_path='data/dpo_train/preference_data.jsonl',
        ...     max_length=512,
        ...     sanity_check=False,
        ...     num_proc=8,
        ...     system="You are a helpful AI assistant."
        ... )
        >>>
        >>> # Use with DPO trainer / 配合 DPO trainer 使用
        >>> from trl import DPOTrainer
        >>>
        >>> trainer = DPOTrainer(
        ...     model=model,
        ...     ref_model=ref_model,
        ...     train_dataset=dpo_dataset,
        ...     tokenizer=tokenizer,
        ...     # ... other arguments
        ... )
        >>>
        >>> # Train / 训练
        >>> trainer.train()

    Notes / 注意事项:
        - Samples where prompt+chosen or prompt+rejected exceed max_length are filtered out
          超过 max_length 的提示+选择或提示+拒绝样本将被过滤掉
        - The dataset is NOT set to torch format (compatible with DPOTrainer)
          数据集未设置为 torch 格式（与 DPOTrainer 兼容）
        - Parallel processing significantly speeds up preprocessing for large datasets
          并行处理显著加快大型数据集的预处理速度
    """

    # Load dataset from JSON file / 从 JSON 文件加载数据集
    dataset = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )
    original_columns = dataset.column_names

    # For quick testing, use only subset / 用于快速测试，仅使用子集
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
        print(f"Sanity check mode: using {len(dataset)} samples")
        print(f"完整性检查模式：使用 {len(dataset)} 个样本")

    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Preprocess a batch of examples
        预处理一批样本

        Formats prompts with system message and user query.
        格式化提示，包含系统消息和用户查询。

        Args:
            examples: Batch of examples from dataset
                     来自数据集的样本批次

        Returns:
            Dictionary with formatted prompts and responses
            包含格式化提示和回答的字典
        """
        prompt_list = []

        # Format each prompt with conversation template
        # 使用对话模板格式化每个提示
        for question in examples["prompt"]:
            formatted_prompt = "\n".join([
                "<|system|>", system.strip(),
                "<|user|>", question.strip(),
                "<|assistant|>"
            ]).strip() + "\n"
            prompt_list.append(formatted_prompt)

        return {
            "prompt": prompt_list,
            "chosen": examples["chosen"],
            "rejected": examples["rejected"],
        }

    # Apply preprocessing in parallel / 并行应用预处理
    print(f"Preprocessing dataset with {num_proc} processes...")
    print(f"使用 {num_proc} 个进程预处理数据集...")

    dataset_map = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    # Filter out samples that are too long
    # 过滤掉太长的样本
    print("Filtering samples by length...")
    print("按长度过滤样本...")

    dataset_map = dataset_map.filter(
        lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_length
        and len(x["prompt"]) + len(x["rejected"]) <= max_length
    )

    print(f"Dataset size after filtering: {len(dataset_map)}")
    print(f"过滤后的数据集大小: {len(dataset_map)}")

    # Note: Do NOT set format to torch for DPO trainer
    # 注意：不要为 DPO trainer 设置 torch 格式
    # dataset_map.set_format(type="torch")

    return dataset_map


def load_ppo_dataset(
    data_path: str,
    tokenizer: Any,
    max_length: int = 256,
    sanity_check: bool = False,
    num_proc: int = 24,
    system: str = "你是由wdndev开发的个人助手。",
) -> datasets.Dataset:
    """
    Load and preprocess dataset for PPO (Proximal Policy Optimization) training
    加载并预处理用于 PPO（近端策略优化）训练的数据集

    PPO is a reinforcement learning algorithm that uses policy gradient methods with
    a clipped objective to ensure stable training.

    PPO 是一种强化学习算法，使用带有截断目标的策略梯度方法来确保稳定训练。

    This function loads prompts and tokenizes them for use in PPO training.
    Unlike DPO, PPO only needs prompts (no chosen/rejected responses).

    此函数加载提示并将其分词以用于 PPO 训练。
    与 DPO 不同，PPO 只需要提示（无需选择/拒绝的回答）。

    Args:
        data_path (str): Path to JSONL file with prompts
                        包含提示的 JSONL 文件路径
        tokenizer: Tokenizer instance for encoding text
                  用于编码文本的分词器实例
        max_length (int): Maximum sequence length for tokenized prompts
                         分词后提示的最大序列长度
        sanity_check (bool): If True, only use first 1000 samples for quick testing
                            如果为 True，仅使用前 1000 个样本进行快速测试
        num_proc (int): Number of processes for parallel data processing
                       并行数据处理的进程数
        system (str): System prompt for conversations
                     对话的系统提示

    Data Format / 数据格式:
        Input JSONL format / 输入 JSONL 格式:
        {
            "prompt": "User question for PPO training / 用于 PPO 训练的用户问题"
        }

        Output dataset format / 输出数据集格式:
        {
            "query": "Formatted query string / 格式化的查询字符串",
            "input_ids": [List of token IDs / token ID 列表]
        }

    Returns:
        datasets.Dataset: Preprocessed HuggingFace Dataset with torch format
                         预处理后的 HuggingFace 数据集，使用 torch 格式

    Example / 示例:
        >>> from transformers import AutoTokenizer
        >>>
        >>> # Create tokenizer / 创建分词器
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>>
        >>> # Load PPO dataset / 加载 PPO 数据集
        >>> ppo_dataset = load_ppo_dataset(
        ...     data_path='data/ppo_train/prompts.jsonl',
        ...     tokenizer=tokenizer,
        ...     max_length=256,
        ...     sanity_check=False,
        ...     num_proc=8,
        ...     system="You are a helpful AI assistant."
        ... )
        >>>
        >>> # Use with PPO trainer / 配合 PPO trainer 使用
        >>> from trl import PPOTrainer, PPOConfig
        >>>
        >>> ppo_config = PPOConfig(
        ...     batch_size=16,
        ...     # ... other config
        ... )
        >>>
        >>> ppo_trainer = PPOTrainer(
        ...     config=ppo_config,
        ...     model=model,
        ...     ref_model=ref_model,
        ...     tokenizer=tokenizer,
        ...     dataset=ppo_dataset,
        ... )
        >>>
        >>> # Training loop / 训练循环
        >>> for batch in ppo_trainer.dataloader:
        ...     query_tensors = batch['input_ids']
        ...     # Generate responses / 生成回答
        ...     response_tensors = ppo_trainer.generate(query_tensors)
        ...     # Compute rewards / 计算奖励
        ...     rewards = reward_model(query_tensors, response_tensors)
        ...     # PPO step / PPO 步骤
        ...     stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

    Notes / 注意事项:
        - Only prompts are needed for PPO (responses generated during training)
          PPO 只需要提示（回答在训练期间生成）
        - Samples exceeding max_length are filtered out
          超过 max_length 的样本将被过滤掉
        - Dataset is set to torch format for compatibility with PPOTrainer
          数据集设置为 torch 格式以与 PPOTrainer 兼容
        - Query strings include full conversation template
          查询字符串包含完整的对话模板
    """

    # Load dataset from JSON file / 从 JSON 文件加载数据集
    dataset = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )
    original_columns = dataset.column_names

    # For quick testing, use only subset / 用于快速测试，仅使用子集
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
        print(f"Sanity check mode: using {len(dataset)} samples")
        print(f"完整性检查模式：使用 {len(dataset)} 个样本")

    def preprocess_function(examples: Dict[str, List[str]]) -> Dict[str, List]:
        """
        Preprocess a batch of examples
        预处理一批样本

        Formats and tokenizes prompts for PPO training.
        格式化并分词提示以用于 PPO 训练。

        Args:
            examples: Batch of examples from dataset
                     来自数据集的样本批次

        Returns:
            Dictionary with query strings and tokenized input_ids
            包含查询字符串和分词后的 input_ids 的字典
        """
        new_examples = {
            "query": [],
            "input_ids": [],
        }

        # Format and tokenize each prompt
        # 格式化并分词每个提示
        for question in examples["prompt"]:
            # Create formatted query / 创建格式化的查询
            query = "\n".join([
                "<|system|>", system.strip(),
                "<|user|>", question.strip(),
                "<|assistant|>"
            ]).strip() + "\n"

            # Tokenize query / 分词查询
            tokenized_query = tokenizer(query, truncation=True)

            new_examples["query"].append(query)
            new_examples["input_ids"].append(tokenized_query["input_ids"])

        return new_examples

    # Apply preprocessing in parallel / 并行应用预处理
    print(f"Preprocessing dataset with {num_proc} processes...")
    print(f"使用 {num_proc} 个进程预处理数据集...")

    dataset_map = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

    # Filter out samples that are too long
    # 过滤掉太长的样本
    print("Filtering samples by length...")
    print("按长度过滤样本...")

    dataset_map = dataset_map.filter(
        lambda x: len(x["input_ids"]) <= max_length
    )

    print(f"Dataset size after filtering: {len(dataset_map)}")
    print(f"过滤后的数据集大小: {len(dataset_map)}")

    # Set format to torch for PPO trainer
    # 为 PPO trainer 设置 torch 格式
    dataset_map.set_format(type="torch")

    return dataset_map


# ============================================================================
# Usage Examples / 使用示例
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of the RLHF datasets
    RLHF 数据集的使用示例
    """

    # Example 1: Using ChatGLM tokenizer / 示例 1：使用 ChatGLM 分词器
    # from utils.chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
    # tokenizer = ChatGLMTokenizer(vocab_file='path/to/tokenizer.model')

    # Example 2: Using HuggingFace tokenizer / 示例 2：使用 HuggingFace 分词器
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # ========================================================================
    # PTMDatasetMap Example / PTMDatasetMap 示例
    # ========================================================================
    print("\n" + "="*80)
    print("PTMDatasetMap Example / PTMDatasetMap 示例")
    print("="*80 + "\n")

    # data_path_list = [
    #     'data/pretrain/pretrain_data_part1.bin',
    #     'data/pretrain/pretrain_data_part2.bin'
    # ]
    # ptm_dataset = PTMDatasetMap(data_path_list, max_length=512)
    #
    # ptm_loader = torch.utils.data.DataLoader(
    #     ptm_dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     num_workers=4
    # )
    #
    # for batch in ptm_loader:
    #     print(f"Batch shape: {batch['input_ids'].shape}")
    #     break

    # ========================================================================
    # RMDataset Example / RMDataset 示例
    # ========================================================================
    print("\n" + "="*80)
    print("RMDataset Example / RMDataset 示例")
    print("="*80 + "\n")

    # rm_dataset = RMDataset(
    #     data_path='data/reward_model/rm_data.jsonl',
    #     tokenizer=tokenizer,
    #     max_length=512,
    #     system="You are a helpful assistant."
    # )
    #
    # rm_loader = torch.utils.data.DataLoader(
    #     rm_dataset,
    #     batch_size=4,
    #     shuffle=True
    # )
    #
    # for batch in rm_loader:
    #     print(f"Chosen shape: {batch['input_ids_j'].shape}")
    #     print(f"Rejected shape: {batch['input_ids_k'].shape}")
    #     break

    # ========================================================================
    # RLDataset Example / RLDataset 示例
    # ========================================================================
    print("\n" + "="*80)
    print("RLDataset Example / RLDataset 示例")
    print("="*80 + "\n")

    # rl_dataset = RLDataset(
    #     data_path='data/rl_train/prompts.jsonl',
    #     tokenizer=tokenizer,
    #     max_length=256,
    #     system="You are a helpful assistant."
    # )
    #
    # rl_loader = torch.utils.data.DataLoader(
    #     rl_dataset,
    #     batch_size=16,
    #     shuffle=True
    # )
    #
    # for batch in rl_loader:
    #     print(f"Query example: {batch['query'][0]}")
    #     print(f"Input shape: {batch['input_ids'].shape}")
    #     break

    # ========================================================================
    # load_dpo_dataset Example / load_dpo_dataset 示例
    # ========================================================================
    print("\n" + "="*80)
    print("load_dpo_dataset Example / load_dpo_dataset 示例")
    print("="*80 + "\n")

    # dpo_dataset = load_dpo_dataset(
    #     data_path='data/dpo_train/preference_data.jsonl',
    #     max_length=512,
    #     sanity_check=True,  # Use small subset for testing
    #     num_proc=4,
    #     system="You are a helpful assistant."
    # )
    #
    # print(f"DPO Dataset size: {len(dpo_dataset)}")
    # print(f"Sample: {dpo_dataset[0]}")

    # ========================================================================
    # load_ppo_dataset Example / load_ppo_dataset 示例
    # ========================================================================
    print("\n" + "="*80)
    print("load_ppo_dataset Example / load_ppo_dataset 示例")
    print("="*80 + "\n")

    # ppo_dataset = load_ppo_dataset(
    #     data_path='data/ppo_train/prompts.jsonl',
    #     tokenizer=tokenizer,
    #     max_length=256,
    #     sanity_check=True,  # Use small subset for testing
    #     num_proc=4,
    #     system="You are a helpful assistant."
    # )
    #
    # print(f"PPO Dataset size: {len(ppo_dataset)}")
    # print(f"Sample: {ppo_dataset[0]}")

    print("\nAll examples completed! / 所有示例完成！")
