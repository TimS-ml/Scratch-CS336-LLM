# GPTQ Quantization Module / GPTQ 量化模块

Production-ready GPTQ quantization implementation for large language models.
用于大型语言模型的生产就绪 GPTQ 量化实现。

## Features / 特性

- ✅ **Clean Implementation** / **清晰实现**: Well-structured, maintainable code
- ✅ **Bilingual Documentation** / **双语文档**: English + Chinese comments and docstrings
- ✅ **Type Hints** / **类型提示**: Full type annotations for better IDE support
- ✅ **Comprehensive Logging** / **全面日志**: Detailed logging for debugging and monitoring
- ✅ **Flexible Configuration** / **灵活配置**: Support for 4-bit and 8-bit quantization
- ✅ **Production Ready** / **生产就绪**: Error handling, validation, and best practices

## Installation / 安装

```bash
# Install required dependencies / 安装所需依赖
pip install torch transformers auto-gptq

# Optional: Install Triton for faster quantization / 可选：安装 Triton 以加快量化
pip install triton
```

## Quick Start / 快速开始

### 1. Command Line Usage / 命令行使用

#### Basic Usage / 基本用法

```bash
python scripts/quantize_model.py \
  --model_path Qwen/Qwen2.5-0.5B \
  --data_path data/calibration.jsonl \
  --output_dir output/quantized_model \
  --bits 4 \
  --group_size 128
```

#### Using Configuration File / 使用配置文件

```bash
python scripts/quantize_model.py --config scripts/configs/quantization.yaml
```

#### Override Config Parameters / 覆盖配置参数

```bash
python scripts/quantize_model.py \
  --config scripts/configs/quantization.yaml \
  --bits 8 \
  --max_samples 512
```

### 2. Python API Usage / Python API 使用

```python
from clean_llm.quantize import GPTQConfig, GPTQQuantizer, prepare_calibration_data

# Create quantization configuration / 创建量化配置
config = GPTQConfig(
    bits=4,                    # 4-bit quantization / 4 位量化
    group_size=128,            # Group size / 组大小
    damp_percent=0.1,          # Damping percentage / 阻尼百分比
    desc_act=False,            # Descending activation (False for speed) / 降序激活
    max_input_length=2048,     # Maximum sequence length / 最大序列长度
    batch_size=1,              # Batch size / 批次大小
)

# Initialize quantizer / 初始化量化器
quantizer = GPTQQuantizer(
    model_path="Qwen/Qwen2.5-0.5B",
    config=config,
    num_gpus=1,
    max_memory_per_gpu=20,
)

# Prepare calibration data / 准备校准数据
calibration_data = prepare_calibration_data(
    data_path="data/calibration.jsonl",
    tokenizer=quantizer.tokenizer,
    max_length=config.max_input_length,
    max_samples=128,
)

# Run quantization / 运行量化
quantizer.quantize(calibration_data)

# Save quantized model / 保存量化模型
quantizer.save("output/quantized_model")

print("Quantization complete! / 量化完成！")
```

### 3. Loading Quantized Model / 加载量化模型

```python
from clean_llm.quantize import load_quantized_model

# Load the quantized model / 加载量化模型
model, tokenizer = load_quantized_model("output/quantized_model")

# Use the model for inference / 使用模型进行推理
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Configuration Parameters / 配置参数

### Quantization Parameters / 量化参数

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bits` | int | 4 | Number of bits (4 or 8) / 位数 |
| `group_size` | int | 128 | Group size (-1, 32, 64, 128) / 组大小 |
| `damp_percent` | float | 0.1 | Damping percentage (0.01-0.1) / 阻尼百分比 |
| `desc_act` | bool | False | Descending activation order / 降序激活 |
| `sym` | bool | True | Symmetric quantization / 对称量化 |
| `static_groups` | bool | False | Static groups / 静态组 |
| `true_sequential` | bool | True | Sequential quantization / 顺序量化 |

### Training Parameters / 训练参数

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_input_length` | int | 8192 | Maximum sequence length / 最大序列长度 |
| `batch_size` | int | 1 | Batch size / 批次大小 |
| `cache_examples_on_gpu` | bool | False | Cache on GPU / GPU 缓存 |
| `use_triton` | bool | False | Use Triton kernels / 使用 Triton |

## Calibration Data Format / 校准数据格式

The calibration data should be in JSONL format (one JSON object per line):
校准数据应为 JSONL 格式（每行一个 JSON 对象）：

```jsonl
{"input": "What is the capital of France?", "target": "Paris"}
{"input": "Explain quantum computing.", "target": "Quantum computing..."}
{"input": "Write a poem about AI.", "target": "In silicon dreams..."}
```

Each line should contain:
每行应包含：
- `input`: User's question or prompt / 用户问题或提示
- `target`: Expected response (optional) / 预期响应（可选）

## Example Configurations / 示例配置

### Maximum Quality / 最高质量

```python
config = GPTQConfig(
    bits=8,
    group_size=32,
    damp_percent=0.01,
    desc_act=True,
)
```

### Balanced / 平衡

```python
config = GPTQConfig(
    bits=4,
    group_size=128,
    damp_percent=0.1,
    desc_act=False,
)
```

### Maximum Speed / 最高速度

```python
config = GPTQConfig(
    bits=4,
    group_size=-1,  # No grouping / 不分组
    damp_percent=0.1,
    desc_act=False,
)
```

## Model Size Examples / 模型大小示例

### Small Models (< 1B) / 小模型（< 10 亿）

```bash
python scripts/quantize_model.py \
  --model_path Qwen/Qwen2.5-0.5B \
  --data_path data/calibration.jsonl \
  --output_dir output/qwen2.5-0.5b-gptq-4bit \
  --bits 4 \
  --group_size 128 \
  --max_memory_per_gpu 10 \
  --max_samples 128
```

### Medium Models (1B-7B) / 中等模型（10-70 亿）

```bash
python scripts/quantize_model.py \
  --model_path Qwen/Qwen2.5-1.5B \
  --data_path data/calibration.jsonl \
  --output_dir output/qwen2.5-1.5b-gptq-4bit \
  --bits 4 \
  --group_size 128 \
  --max_memory_per_gpu 20 \
  --max_samples 256
```

### Large Models (7B-13B) / 大模型（70-130 亿）

```bash
python scripts/quantize_model.py \
  --model_path Qwen/Qwen2.5-7B \
  --data_path data/calibration.jsonl \
  --output_dir output/qwen2.5-7b-gptq-4bit \
  --bits 4 \
  --group_size 128 \
  --num_gpus 2 \
  --max_memory_per_gpu 40 \
  --max_samples 512
```

## Performance Tips / 性能提示

### Memory Optimization / 内存优化

1. **Monitor GPU memory**: Use `nvidia-smi` to monitor GPU usage
   **监控 GPU 内存**：使用 `nvidia-smi` 监控 GPU 使用情况

2. **Reduce memory usage**: If OOM, reduce these parameters:
   **减少内存使用**：如果内存不足，减少这些参数：
   - `max_memory_per_gpu`
   - `batch_size`
   - `max_input_length`

3. **Multi-GPU**: Use multiple GPUs for large models
   **多 GPU**：对大模型使用多个 GPU
   ```bash
   --num_gpus 2 --max_memory_per_gpu 40
   ```

### Speed Optimization / 速度优化

1. **Cache on GPU**: Enable for faster processing (if memory allows)
   **GPU 缓存**：启用以加快处理（如果内存允许）
   ```bash
   --cache_examples_on_gpu
   ```

2. **Use Triton**: Install and enable Triton kernels
   **使用 Triton**：安装并启用 Triton 内核
   ```bash
   pip install triton
   python scripts/quantize_model.py --use_triton ...
   ```

3. **Reduce calibration samples**: Fewer samples = faster quantization
   **减少校准样本**：更少样本 = 更快量化
   ```bash
   --max_samples 128
   ```

### Quality Optimization / 质量优化

1. **Use 8-bit quantization**: Better quality, less compression
   **使用 8 位量化**：更好质量，更少压缩
   ```bash
   --bits 8
   ```

2. **Smaller group size**: Better quality, larger model size
   **更小的组大小**：更好质量，更大模型
   ```bash
   --group_size 64
   ```

3. **More calibration samples**: Better quality, slower quantization
   **更多校准样本**：更好质量，更慢量化
   ```bash
   --max_samples 512
   ```

## Troubleshooting / 故障排除

### Out of Memory (OOM) / 内存不足

**Problem**: CUDA out of memory error
**问题**：CUDA 内存不足错误

**Solution**:
**解决方案**：
```bash
# Reduce GPU memory allocation
--max_memory_per_gpu 10

# Reduce batch size
--batch_size 1

# Reduce max input length
--max_input_length 2048
```

### Quantization Too Slow / 量化太慢

**Problem**: Quantization takes too long
**问题**：量化时间过长

**Solution**:
**解决方案**：
```bash
# Reduce calibration samples
--max_samples 128

# Enable GPU caching
--cache_examples_on_gpu

# Use Triton (if available)
--use_triton
```

### Poor Model Quality / 模型质量差

**Problem**: Quantized model performs poorly
**问题**：量化模型性能差

**Solution**:
**解决方案**：
```bash
# Use 8-bit quantization
--bits 8

# Smaller group size
--group_size 64

# More calibration samples
--max_samples 512

# Enable descending activation
--desc_act
```

### Model Not Loading / 模型无法加载

**Problem**: Cannot load model or tokenizer
**问题**：无法加载模型或分词器

**Solution**:
**解决方案**：
```bash
# Ensure trust_remote_code is enabled
--trust_remote_code

# Check model path is correct
--model_path Qwen/Qwen2.5-0.5B

# Verify you have internet connection for HuggingFace downloads
```

## API Reference / API 参考

### GPTQConfig

Configuration class for GPTQ quantization.
GPTQ 量化配置类。

```python
config = GPTQConfig(
    bits: int = 4,
    group_size: int = 128,
    damp_percent: float = 0.1,
    desc_act: bool = False,
    static_groups: bool = False,
    sym: bool = True,
    true_sequential: bool = True,
    max_input_length: int = 8192,
    batch_size: int = 1,
    cache_examples_on_gpu: bool = False,
    use_triton: bool = False,
)
```

### GPTQQuantizer

Main quantizer class.
主要量化器类。

```python
quantizer = GPTQQuantizer(
    model_path: str,
    config: GPTQConfig,
    num_gpus: int = 1,
    max_memory_per_gpu: int = 20,
    trust_remote_code: bool = True,
)

# Quantize model / 量化模型
quantizer.quantize(calibration_data: List[Dict[str, torch.Tensor]])

# Save quantized model / 保存量化模型
quantizer.save(
    output_dir: Union[str, Path],
    use_safetensors: bool = True,
    save_tokenizer: bool = True,
)

# Get model size info / 获取模型大小信息
info = quantizer.get_model_size_info()
```

### Helper Functions / 辅助函数

```python
# Prepare calibration data / 准备校准数据
calibration_data = prepare_calibration_data(
    data_path: Union[str, Path],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 8192,
    max_samples: Optional[int] = None,
    input_field: str = "input",
    target_field: str = "target",
    use_chat_template: bool = True,
)

# Load quantized model / 加载量化模型
model, tokenizer = load_quantized_model(
    model_path: Union[str, Path],
    device_map: str = "auto",
    trust_remote_code: bool = True,
)
```

## Advanced Usage / 高级用法

### Custom Data Processing / 自定义数据处理

```python
from clean_llm.quantize import GPTQQuantizer, GPTQConfig
import torch

# Create custom calibration data / 创建自定义校准数据
calibration_data = []
for text in custom_texts:
    inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)
    calibration_data.append({
        "input_ids": inputs["input_ids"].squeeze(0),
        "attention_mask": inputs["attention_mask"].squeeze(0),
    })

# Quantize with custom data / 使用自定义数据量化
quantizer.quantize(calibration_data)
```

### Multi-GPU Quantization / 多 GPU 量化

```python
quantizer = GPTQQuantizer(
    model_path="Qwen/Qwen2.5-7B",
    config=config,
    num_gpus=4,  # Use 4 GPUs / 使用 4 个 GPU
    max_memory_per_gpu=40,  # 40GB per GPU / 每个 GPU 40GB
)
```

### Batch Inference with Quantized Model / 使用量化模型批量推理

```python
from clean_llm.quantize import load_quantized_model

model, tokenizer = load_quantized_model("output/quantized_model")

# Batch inference / 批量推理
texts = ["Question 1?", "Question 2?", "Question 3?"]
inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100)

for i, output in enumerate(outputs):
    response = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Response {i+1}: {response}")
```

## License / 许可证

MIT License

## References / 参考文献

- [GPTQ Paper](https://arxiv.org/abs/2210.17323): "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
- [AutoGPTQ GitHub](https://github.com/AutoGPTQ/AutoGPTQ): Official AutoGPTQ implementation
- [Transformers](https://github.com/huggingface/transformers): HuggingFace Transformers library

## Support / 支持

For issues and questions:
问题和疑问：

- GitHub Issues: [Project Issues](https://github.com/your-repo/issues)
- Documentation: See this README and code comments
- Examples: Check `scripts/configs/quantization.yaml`

## Contributing / 贡献

Contributions are welcome! Please:
欢迎贡献！请：

1. Fork the repository / 复刻仓库
2. Create a feature branch / 创建功能分支
3. Add tests for new features / 为新功能添加测试
4. Submit a pull request / 提交拉取请求

---

**Note**: This implementation requires `auto-gptq`, `torch`, and `transformers` libraries.
**注意**：此实现需要 `auto-gptq`、`torch` 和 `transformers` 库。

For production use, always validate quantized models on your specific use cases.
用于生产时，请始终在您的特定用例上验证量化模型。
