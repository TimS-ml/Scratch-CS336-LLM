"""
GPTQ Quantization Package / GPTQ 量化包

This package provides GPTQ quantization support for large language models.
此包为大型语言模型提供 GPTQ 量化支持。

Modules:
    gptq: GPTQ quantization implementation / GPTQ 量化实现

Example / 示例:
    >>> from clean_llm.quantize import GPTQQuantizer, GPTQConfig
    >>> config = GPTQConfig(bits=4, group_size=128)
    >>> quantizer = GPTQQuantizer(model_path="path/to/model", config=config)
    >>> quantizer.quantize(calibration_data)
"""

from .gptq import (
    GPTQConfig,
    GPTQQuantizer,
    prepare_calibration_data,
    load_quantized_model,
)

__all__ = [
    "GPTQConfig",
    "GPTQQuantizer",
    "prepare_calibration_data",
    "load_quantized_model",
]

__version__ = "1.0.0"
