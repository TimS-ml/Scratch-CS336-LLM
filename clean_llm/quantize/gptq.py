"""
GPTQ Quantization Implementation / GPTQ 量化实现

This module implements GPTQ (Gradient-based Post-Training Quantization) for language models.
此模块为语言模型实现 GPTQ（基于梯度的训练后量化）。

GPTQ is a one-shot weight quantization method that uses a second-order information (Hessian)
to quantize weights while minimizing the reconstruction error.
GPTQ 是一种单次权重量化方法，使用二阶信息（Hessian）来量化权重，同时最小化重建误差。

References / 参考:
    - Paper: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
    - GitHub: https://github.com/AutoGPTQ/AutoGPTQ

Author: LLM-from-Scratch Project
License: MIT
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer, PreTrainedTokenizer

# Configure logging / 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    """
    Configuration for GPTQ quantization / GPTQ 量化配置

    This class holds all the parameters needed for GPTQ quantization.
    此类包含 GPTQ 量化所需的所有参数。

    Attributes / 属性:
        bits (int): Number of bits for quantization (4 or 8) / 量化位数（4 或 8）
        group_size (int): Group size for quantization (32, 64, 128, -1 for no grouping)
                         / 量化组大小（32、64、128，-1 表示不分组）
        damp_percent (float): Damping percentage for quantization (0.01-0.1)
                             / 量化阻尼百分比（0.01-0.1）
        desc_act (bool): Whether to use descending activation order
                        False speeds up inference but may affect perplexity
                        / 是否使用降序激活顺序，False 可加速推理但可能影响困惑度
        static_groups (bool): Whether to use static groups / 是否使用静态组
        sym (bool): Whether to use symmetric quantization / 是否使用对称量化
        true_sequential (bool): Whether to use true sequential quantization
                               / 是否使用真正的顺序量化
        max_input_length (int): Maximum input sequence length / 最大输入序列长度
        batch_size (int): Batch size for quantization / 量化批次大小
        cache_examples_on_gpu (bool): Cache calibration examples on GPU for faster processing
                                     / 在 GPU 上缓存校准样本以加快处理速度
        use_triton (bool): Use Triton kernels for quantization (faster but requires Triton)
                          / 使用 Triton 内核进行量化（更快但需要 Triton）

    Example / 示例:
        >>> # 4-bit quantization with default settings / 4 位量化，默认设置
        >>> config = GPTQConfig(bits=4, group_size=128)
        >>>
        >>> # 8-bit quantization for better quality / 8 位量化以获得更好质量
        >>> config = GPTQConfig(bits=8, group_size=128, damp_percent=0.01)
    """

    # Quantization parameters / 量化参数
    bits: int = field(
        default=4,
        metadata={"help": "Number of bits for quantization (4 or 8) / 量化位数（4 或 8）"}
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Group size for quantization (32, 64, 128, -1) / 量化组大小"}
    )
    damp_percent: float = field(
        default=0.1,
        metadata={"help": "Damping percentage for quantization / 量化阻尼百分比"}
    )
    desc_act: bool = field(
        default=False,
        metadata={"help": "Use descending activation order (False for faster inference) / 使用降序激活顺序"}
    )
    static_groups: bool = field(
        default=False,
        metadata={"help": "Use static groups / 使用静态组"}
    )
    sym: bool = field(
        default=True,
        metadata={"help": "Use symmetric quantization / 使用对称量化"}
    )
    true_sequential: bool = field(
        default=True,
        metadata={"help": "Use true sequential quantization / 使用真正的顺序量化"}
    )

    # Training parameters / 训练参数
    max_input_length: int = field(
        default=8192,
        metadata={"help": "Maximum input sequence length / 最大输入序列长度"}
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for quantization / 量化批次大小"}
    )
    cache_examples_on_gpu: bool = field(
        default=False,
        metadata={"help": "Cache examples on GPU / 在 GPU 上缓存样本"}
    )
    use_triton: bool = field(
        default=False,
        metadata={"help": "Use Triton kernels / 使用 Triton 内核"}
    )

    def __post_init__(self):
        """Validate configuration parameters / 验证配置参数"""
        if self.bits not in [4, 8]:
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")

        if self.group_size not in [-1, 32, 64, 128]:
            logger.warning(
                f"group_size={self.group_size} is not a standard value. "
                f"Recommended values are -1, 32, 64, or 128."
            )

        if not 0 < self.damp_percent < 1:
            raise ValueError(f"damp_percent must be between 0 and 1, got {self.damp_percent}")

    def to_base_config(self) -> BaseQuantizeConfig:
        """
        Convert to AutoGPTQ BaseQuantizeConfig / 转换为 AutoGPTQ BaseQuantizeConfig

        Returns:
            BaseQuantizeConfig: AutoGPTQ quantization configuration
        """
        return BaseQuantizeConfig(
            bits=self.bits,
            group_size=self.group_size,
            damp_percent=self.damp_percent,
            desc_act=self.desc_act,
            static_groups=self.static_groups,
            sym=self.sym,
            true_sequential=self.true_sequential,
        )


class GPTQQuantizer:
    """
    GPTQ Quantizer for Language Models / 语言模型 GPTQ 量化器

    This class handles the complete GPTQ quantization workflow including:
    此类处理完整的 GPTQ 量化工作流，包括：
    - Loading the pre-trained model / 加载预训练模型
    - Preparing calibration data / 准备校准数据
    - Running quantization / 运行量化
    - Saving the quantized model / 保存量化模型

    Example / 示例:
        >>> from clean_llm.quantize import GPTQQuantizer, GPTQConfig
        >>>
        >>> # Initialize quantizer / 初始化量化器
        >>> config = GPTQConfig(bits=4, group_size=128)
        >>> quantizer = GPTQQuantizer(
        ...     model_path="Qwen/Qwen2.5-0.5B",
        ...     config=config,
        ...     num_gpus=1,
        ...     max_memory_per_gpu=20
        ... )
        >>>
        >>> # Prepare calibration data / 准备校准数据
        >>> calibration_data = prepare_calibration_data(
        ...     data_path="data/calibration.jsonl",
        ...     tokenizer=quantizer.tokenizer,
        ...     max_length=config.max_input_length
        ... )
        >>>
        >>> # Quantize the model / 量化模型
        >>> quantizer.quantize(calibration_data)
        >>>
        >>> # Save quantized model / 保存量化模型
        >>> quantizer.save("output/quantized_model")
    """

    def __init__(
        self,
        model_path: str,
        config: GPTQConfig,
        num_gpus: int = 1,
        max_memory_per_gpu: int = 20,
        trust_remote_code: bool = True,
    ):
        """
        Initialize GPTQ Quantizer / 初始化 GPTQ 量化器

        Args:
            model_path (str): Path or HuggingFace model ID / 路径或 HuggingFace 模型 ID
            config (GPTQConfig): GPTQ configuration / GPTQ 配置
            num_gpus (int): Number of GPUs to use / 使用的 GPU 数量
            max_memory_per_gpu (int): Maximum memory per GPU in GB / 每个 GPU 的最大内存（GB）
            trust_remote_code (bool): Trust remote code when loading model / 加载模型时信任远程代码
        """
        self.model_path = model_path
        self.config = config
        self.num_gpus = num_gpus
        self.max_memory_per_gpu = max_memory_per_gpu
        self.trust_remote_code = trust_remote_code

        logger.info(f"Initializing GPTQ Quantizer / 初始化 GPTQ 量化器")
        logger.info(f"Model: {model_path}")
        logger.info(f"Quantization config: {config.bits}-bit, group_size={config.group_size}")

        # Load tokenizer / 加载分词器
        self._load_tokenizer()

        # Load model / 加载模型
        self._load_model()

        self._is_quantized = False

    def _load_tokenizer(self) -> None:
        """Load tokenizer from model path / 从模型路径加载分词器"""
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        logger.info("Tokenizer loaded successfully / 分词器加载成功")

    def _load_model(self) -> None:
        """Load model for quantization / 加载模型进行量化"""
        logger.info(f"Loading model from {self.model_path}")

        # Configure GPU memory / 配置 GPU 内存
        max_memory = {i: f"{self.max_memory_per_gpu}GB" for i in range(self.num_gpus)}
        logger.info(f"GPU memory configuration: {max_memory}")

        # Load model with quantization config / 使用量化配置加载模型
        self.model = AutoGPTQForCausalLM.from_pretrained(
            self.model_path,
            quantize_config=self.config.to_base_config(),
            max_memory=max_memory,
            trust_remote_code=self.trust_remote_code,
        )

        logger.info("Model loaded successfully / 模型加载成功")

    def quantize(
        self,
        calibration_data: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Quantize the model using calibration data / 使用校准数据量化模型

        Args:
            calibration_data (List[Dict[str, torch.Tensor]]): List of calibration examples
                Each example should contain 'input_ids' and 'attention_mask'
                / 校准样本列表，每个样本应包含 'input_ids' 和 'attention_mask'

        Raises:
            ValueError: If calibration_data is empty or invalid
        """
        if not calibration_data:
            raise ValueError("calibration_data cannot be empty / 校准数据不能为空")

        logger.info(f"Starting quantization with {len(calibration_data)} calibration examples")
        logger.info(f"使用 {len(calibration_data)} 个校准样本开始量化")

        try:
            self.model.quantize(
                calibration_data,
                batch_size=self.config.batch_size,
                cache_examples_on_gpu=self.config.cache_examples_on_gpu,
                use_triton=self.config.use_triton,
            )
            self._is_quantized = True
            logger.info("Quantization completed successfully / 量化成功完成")

        except Exception as e:
            logger.error(f"Quantization failed / 量化失败: {str(e)}")
            raise

    def save(
        self,
        output_dir: Union[str, Path],
        use_safetensors: bool = True,
        save_tokenizer: bool = True,
    ) -> None:
        """
        Save quantized model to disk / 保存量化模型到磁盘

        Args:
            output_dir (Union[str, Path]): Output directory / 输出目录
            use_safetensors (bool): Use safetensors format (recommended) / 使用 safetensors 格式（推荐）
            save_tokenizer (bool): Also save tokenizer / 同时保存分词器

        Raises:
            RuntimeError: If model has not been quantized yet
        """
        if not self._is_quantized:
            raise RuntimeError(
                "Model must be quantized before saving. Call quantize() first. "
                "/ 保存前必须先量化模型。请先调用 quantize()。"
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving quantized model to {output_dir}")
        logger.info(f"保存量化模型到 {output_dir}")

        try:
            # Save model / 保存模型
            self.model.save_quantized(
                str(output_dir),
                use_safetensors=use_safetensors,
            )

            # Save tokenizer / 保存分词器
            if save_tokenizer:
                self.tokenizer.save_pretrained(str(output_dir))
                logger.info("Tokenizer saved / 分词器已保存")

            logger.info("Model saved successfully / 模型保存成功")

        except Exception as e:
            logger.error(f"Failed to save model / 保存模型失败: {str(e)}")
            raise

    def get_model_size_info(self) -> Dict[str, Any]:
        """
        Get information about model size / 获取模型大小信息

        Returns:
            Dict[str, Any]: Dictionary containing size information
        """
        if not self._is_quantized:
            logger.warning("Model has not been quantized yet / 模型尚未量化")

        # Calculate approximate size reduction / 计算近似的大小减少
        original_bits = 16  # Assume FP16
        quantized_bits = self.config.bits
        compression_ratio = original_bits / quantized_bits

        return {
            "quantization_bits": quantized_bits,
            "original_precision": f"FP{original_bits}",
            "estimated_compression_ratio": f"{compression_ratio}x",
            "is_quantized": self._is_quantized,
        }


def prepare_calibration_data(
    data_path: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    max_samples: Optional[int] = None,
    input_field: str = "input",
    target_field: str = "target",
    use_chat_template: bool = True,
) -> List[Dict[str, torch.Tensor]]:
    """
    Prepare calibration data for GPTQ quantization / 为 GPTQ 量化准备校准数据

    Args:
        data_path (Union[str, Path]): Path to JSONL data file / JSONL 数据文件路径
        tokenizer (PreTrainedTokenizer): Tokenizer to use / 使用的分词器
        max_length (int): Maximum sequence length / 最大序列长度
        max_samples (Optional[int]): Maximum number of samples to use (None for all)
                                    / 使用的最大样本数（None 表示全部）
        input_field (str): Field name for input text / 输入文本字段名
        target_field (str): Field name for target text / 目标文本字段名
        use_chat_template (bool): Use chat template for formatting / 使用对话模板格式化

    Returns:
        List[Dict[str, torch.Tensor]]: List of tokenized calibration examples
                                       / 分词后的校准样本列表

    Example / 示例:
        >>> calibration_data = prepare_calibration_data(
        ...     data_path="data/calibration.jsonl",
        ...     tokenizer=tokenizer,
        ...     max_length=2048,
        ...     max_samples=128
        ... )
    """
    import json

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading calibration data from {data_path}")
    logger.info(f"从 {data_path} 加载校准数据")

    # Read JSONL file / 读取 JSONL 文件
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data_list.append(json.loads(line))

    # Limit samples if specified / 如果指定则限制样本数
    if max_samples is not None and len(data_list) > max_samples:
        logger.info(f"Limiting to {max_samples} samples out of {len(data_list)}")
        data_list = data_list[:max_samples]

    logger.info(f"Processing {len(data_list)} calibration examples")

    # Process data / 处理数据
    calibration_data = []
    for idx, item in enumerate(data_list):
        try:
            input_text = item.get(input_field, "")
            target_text = item.get(target_field, "")

            if not input_text:
                logger.warning(f"Empty input in example {idx}, skipping")
                continue

            # Format text / 格式化文本
            if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input_text},
                ]
                if target_text:
                    messages.append({"role": "assistant", "content": target_text})

                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            else:
                text = input_text
                if target_text:
                    text += f"\n{target_text}"

            # Tokenize / 分词
            model_inputs = tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors="pt"
            )

            calibration_data.append({
                "input_ids": model_inputs["input_ids"].squeeze(0),
                "attention_mask": model_inputs["attention_mask"].squeeze(0),
            })

        except Exception as e:
            logger.warning(f"Failed to process example {idx}: {str(e)}")
            continue

    logger.info(f"Successfully prepared {len(calibration_data)} calibration examples")
    logger.info(f"成功准备了 {len(calibration_data)} 个校准样本")

    return calibration_data


def load_quantized_model(
    model_path: Union[str, Path],
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> tuple[AutoGPTQForCausalLM, PreTrainedTokenizer]:
    """
    Load a quantized GPTQ model / 加载量化的 GPTQ 模型

    Args:
        model_path (Union[str, Path]): Path to quantized model / 量化模型路径
        device_map (str): Device map for model placement / 模型放置的设备映射
        trust_remote_code (bool): Trust remote code / 信任远程代码

    Returns:
        tuple: (model, tokenizer) / (模型, 分词器)

    Example / 示例:
        >>> model, tokenizer = load_quantized_model("output/quantized_model")
        >>> outputs = model.generate(**tokenizer("Hello", return_tensors="pt"))
    """
    logger.info(f"Loading quantized model from {model_path}")
    logger.info(f"从 {model_path} 加载量化模型")

    # Load model / 加载模型
    model = AutoGPTQForCausalLM.from_quantized(
        model_path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )

    # Load tokenizer / 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
    )

    logger.info("Model and tokenizer loaded successfully")
    logger.info("模型和分词器加载成功")

    return model, tokenizer


# Example usage / 使用示例
if __name__ == "__main__":
    """
    Example script demonstrating GPTQ quantization / 演示 GPTQ 量化的示例脚本

    This example shows how to:
    此示例展示如何：
    1. Create a quantization configuration / 创建量化配置
    2. Initialize the quantizer / 初始化量化器
    3. Prepare calibration data / 准备校准数据
    4. Run quantization / 运行量化
    5. Save the quantized model / 保存量化模型
    """

    # Configure logging / 配置日志
    logging.basicConfig(level=logging.INFO)

    # Example configuration / 示例配置
    config = GPTQConfig(
        bits=4,
        group_size=128,
        damp_percent=0.1,
        desc_act=False,  # False for faster inference / False 以加快推理速度
        max_input_length=2048,
    )

    print("GPTQ Configuration / GPTQ 配置:")
    print(f"  Bits / 位数: {config.bits}")
    print(f"  Group size / 组大小: {config.group_size}")
    print(f"  Damp percent / 阻尼百分比: {config.damp_percent}")
    print(f"  Max input length / 最大输入长度: {config.max_input_length}")

    print("\nTo use this module / 使用此模块:")
    print("1. Create a GPTQConfig / 创建 GPTQConfig")
    print("2. Initialize GPTQQuantizer with your model / 使用您的模型初始化 GPTQQuantizer")
    print("3. Prepare calibration data / 准备校准数据")
    print("4. Call quantizer.quantize(calibration_data) / 调用 quantizer.quantize(calibration_data)")
    print("5. Save with quantizer.save(output_dir) / 使用 quantizer.save(output_dir) 保存")
