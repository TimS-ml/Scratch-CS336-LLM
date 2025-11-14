"""
Custom logits processors for fine-grained generation control.
用于细粒度生成控制的自定义logits处理器。
"""

import torch
from typing import Tuple
from transformers.utils import logging, add_start_docstrings
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)


class OutputRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    """
    Custom logits processor that applies repetition penalty only to output tokens.
    仅对输出token应用重复惩罚的自定义logits处理器。

    This processor prevents repetition in generated text through three types of penalties:
    1. Repetition penalty: Penalizes tokens that have already appeared
    2. Frequency penalty: Penalizes based on token frequency
    3. Presence penalty: Penalizes based on token presence (binary)

    此处理器通过三种类型的惩罚来防止生成文本中的重复：
    1. 重复惩罚：惩罚已出现的token
    2. 频率惩罚：基于token频率进行惩罚
    3. 存在惩罚：基于token存在性进行惩罚（二进制）

    Unlike standard repetition penalty which applies to the entire sequence including
    the prompt, this processor separates prompt tokens from output tokens and applies
    penalties more selectively.

    与应用于包括提示在内的整个序列的标准重复惩罚不同，此处理器将提示token与
    输出token分开，并更有选择性地应用惩罚。

    Args:
        input_length (int): Length of the input prompt (tokens before generation starts).
                           输入提示的长度（生成开始前的token数）。
        presence_penalties (float, optional): Penalty for token presence. Range: [-2, 2].
                                             1.0 means no penalty. Defaults to 1.0.
                                             token存在性惩罚。范围：[-2, 2]。
                                             1.0表示无惩罚。默认为1.0。
        frequency_penalties (float, optional): Penalty based on token frequency. Range: [-2, 2].
                                              0 means no penalty. Defaults to 0.
                                              基于token频率的惩罚。范围：[-2, 2]。
                                              0表示无惩罚。默认为0。
        repetition_penalties (float, optional): Main repetition penalty factor.
                                               Must be > 0. 1.0 means no penalty.
                                               Above 1.0 penalizes, below 1.0 rewards.
                                               Defaults to 0 (but must be set > 0).
                                               主要重复惩罚因子。必须> 0。1.0表示无惩罚。
                                               大于1.0为惩罚，小于1.0为奖励。
                                               默认为0（但必须设置> 0）。

    Raises:
        ValueError: If penalty parameters are outside valid ranges.
                   如果惩罚参数超出有效范围。

    Example:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("model_name")
        >>> model = AutoModelForCausalLM.from_pretrained("model_name")
        >>>
        >>> # Prepare input
        >>> prompt = "Once upon a time"
        >>> input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        >>> input_length = input_ids.shape[1]
        >>>
        >>> # Create processor with moderate repetition penalty
        >>> processor = OutputRepetitionPenaltyLogitsProcessor(
        ...     input_length=input_length,
        ...     repetition_penalties=1.2,  # Penalize repetition
        ...     frequency_penalties=0.5,   # Light frequency penalty
        ...     presence_penalties=0.0     # No presence penalty
        ... )
        >>>
        >>> # Generate with processor
        >>> output = model.generate(
        ...     input_ids,
        ...     logits_processor=[processor],
        ...     max_new_tokens=50
        ... )

    Notes:
        Penalty Meanings / 惩罚含义:
        - Repetition penalty > 1.0: Reduce probability of repeated tokens / 降低重复token概率
        - Repetition penalty < 1.0: Increase probability of repeated tokens / 增加重复token概率
        - Frequency penalty: Subtracts penalty proportional to frequency / 减去与频率成比例的惩罚
        - Presence penalty: Binary penalty if token appeared at all / 如果token出现则应用二进制惩罚

        Based on OpenAI API parameter definitions:
        https://platform.openai.com/docs/api-reference/parameter-details
        基于OpenAI API参数定义。

    References:
        Original repetition penalty paper: https://arxiv.org/pdf/1909.05858.pdf
        Suggested penalty value: ~1.2 for good balance between truthfulness and diversity
        建议的惩罚值：约1.2，以在真实性和多样性之间取得良好平衡
    """

    def __init__(
        self,
        input_length: int,
        presence_penalties: float = 1.0,
        frequency_penalties: float = 0.0,
        repetition_penalties: float = 1.0,
    ):
        # Validate repetition penalty
        # 验证重复惩罚参数
        if not (repetition_penalties > 0):
            raise ValueError(
                f"`repetition_penalties` has to be a strictly positive float, "
                f"but is {repetition_penalties}"
            )

        # Validate frequency penalty range
        # 验证频率惩罚范围
        if not ((frequency_penalties >= -2) and (frequency_penalties <= 2)):
            raise ValueError(
                f"`frequency_penalties` has to be in [-2, 2], "
                f"but is {frequency_penalties}"
            )

        # Validate presence penalty range
        # 验证存在性惩罚范围
        if not ((presence_penalties >= -2) and (presence_penalties <= 2)):
            raise ValueError(
                f"`presence_penalties` has to be in [-2, 2], "
                f"but is {presence_penalties}"
            )

        self.repetition_penalties = repetition_penalties
        self.frequency_penalties = frequency_penalties
        self.presence_penalties = presence_penalties
        self.input_length = input_length

    def _get_bin_counts_and_mask(
        self,
        tokens: torch.Tensor,
        vocab_size: int,
        num_seqs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute token frequency counts and presence mask.
        计算token频率计数和存在性掩码。

        Args:
            tokens: Token tensor of shape [num_seqs, seq_len].
                   形状为[num_seqs, seq_len]的token张量。
            vocab_size: Size of the vocabulary.
                       词汇表大小。
            num_seqs: Number of sequences in the batch.
                     批次中的序列数。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - bin_counts: Token frequency counts [num_seqs, vocab_size]
                             token频率计数 [num_seqs, vocab_size]
                - mask: Binary mask for token presence [num_seqs, vocab_size]
                       token存在性的二进制掩码 [num_seqs, vocab_size]
        """
        # Initialize bin counts with extra column for padding
        # 使用额外的列初始化bin计数（用于填充）
        bin_counts = torch.zeros(
            (num_seqs, vocab_size + 1),
            dtype=torch.long,
            device=tokens.device
        )

        # Count occurrences of each token
        # 统计每个token的出现次数
        bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))

        # Remove padding column
        # 移除填充列
        bin_counts = bin_counts[:, :vocab_size]

        # Create binary presence mask
        # 创建二进制存在性掩码
        mask = bin_counts > 0

        return bin_counts, mask

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self,
        input_ids: torch.LongTensor,
        logits: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to apply repetition penalties.
        处理logits以应用重复惩罚。

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len].
                      形状为[batch_size, seq_len]的token ID。
            logits: Raw logits from model of shape [batch_size, vocab_size].
                   来自模型的原始logits，形状为[batch_size, vocab_size]。

        Returns:
            torch.FloatTensor: Processed logits with penalties applied.
                              应用惩罚后的处理后logits。
        """
        # Split input into prompt and output tokens
        # 将输入分为提示token和输出token
        prompt_tokens_tensor = input_ids[:, :self.input_length + 1]
        output_tokens_tensor = input_ids[:, self.input_length + 1:]

        num_seqs, vocab_size = logits.shape

        # Get token statistics for prompt and output
        # 获取提示和输出的token统计信息
        _, prompt_mask = self._get_bin_counts_and_mask(
            prompt_tokens_tensor, vocab_size, num_seqs
        )
        output_bin_counts, output_mask = self._get_bin_counts_and_mask(
            output_tokens_tensor, vocab_size, num_seqs
        )

        # Convert penalty values to tensors
        # 将惩罚值转换为张量
        repetition_penalties = torch.tensor(
            [self.repetition_penalties],
            device=logits.device
        )
        frequency_penalties = torch.tensor(
            [self.frequency_penalties],
            device=logits.device
        )
        presence_penalties = torch.tensor(
            [self.presence_penalties],
            device=logits.device
        )

        # Apply repetition penalty
        # 应用重复惩罚
        repetition_penalties = repetition_penalties[:, None].repeat(1, vocab_size)
        # Only penalize tokens that appeared in prompt or output
        # 仅惩罚在提示或输出中出现的token
        repetition_penalties[~(prompt_mask | output_mask)] = 1.0

        # Apply penalty: divide if logit > 0, multiply if logit < 0
        # 应用惩罚：如果logit > 0则除以惩罚，如果logit < 0则乘以惩罚
        logits = torch.where(
            logits > 0,
            logits / repetition_penalties,
            logits * repetition_penalties
        )

        # Apply frequency penalty (OpenAI API definition)
        # 应用频率惩罚（OpenAI API定义）
        # Subtract penalty proportional to how many times token appeared
        # 减去与token出现次数成比例的惩罚
        logits -= frequency_penalties.unsqueeze_(dim=1) * output_bin_counts

        # Apply presence penalty (OpenAI API definition)
        # 应用存在性惩罚（OpenAI API定义）
        # Subtract penalty if token appeared at all (binary)
        # 如果token出现过则减去惩罚（二进制）
        logits -= presence_penalties.unsqueeze_(dim=1) * output_mask

        return logits
