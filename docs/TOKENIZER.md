# Tokenizer Module / 分词器模块

Comprehensive tokenizer functionality for training, merging, and expanding language model vocabularies.

用于训练、合并和扩展语言模型词汇表的综合分词器功能。

---

## Features / 功能特性

### 1. **Chinese Tokenizer Training** / **中文分词器训练**
- Train custom SentencePiece tokenizers optimized for Chinese text
- 训练针对中文文本优化的自定义SentencePiece分词器
- Support for BPE, Unigram, Character, and Word models
- 支持BPE、Unigram、字符和词模型
- Configurable vocabulary size and character coverage
- 可配置的词汇表大小和字符覆盖率

### 2. **Vocabulary Merging** / **词汇表合并**
- Merge base tokenizers (e.g., LLaMA) with language-specific tokens
- 合并基础分词器（如LLaMA）与特定语言tokens
- Preserve original vocabulary while adding new tokens
- 保留原始词汇表同时添加新tokens
- Support for custom special tokens
- 支持自定义特殊tokens

### 3. **Embedding Expansion** / **嵌入扩展**
- Expand model embeddings to accommodate new vocabulary
- 扩展模型嵌入以适应新词汇表
- Preserve weights for existing tokens
- 保留现有tokens的权重
- Multiple initialization strategies for new embeddings
- 为新嵌入提供多种初始化策略

---

## Installation / 安装

```bash
# Install required dependencies / 安装所需依赖
pip install transformers sentencepiece torch
```

---

## Quick Start / 快速开始

### Complete Workflow / 完整工作流程

```python
from scratch_cs336.core.tokenizer import (
    train_chinese_tokenizer,
    merge_tokenizers,
    expand_model_for_tokenizer
)

# Step 1: Train Chinese tokenizer / 步骤1：训练中文分词器
chinese_model = train_chinese_tokenizer(
    input_files="chinese_corpus/",
    vocab_size=20000,
    output_dir="chinese_tokenizer"
)

# Step 2: Merge with base tokenizer / 步骤2：与基础分词器合并
merged_tokenizer = merge_tokenizers(
    base_tokenizer_path="llama2_tokenizer/",
    additional_sp_model_path=chinese_model,
    output_dir="merged_tokenizer/",
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"]
)

# Step 3: Expand model embeddings / 步骤3：扩展模型嵌入
model, tokenizer = expand_model_for_tokenizer(
    model_path="original_model/",
    tokenizer_path="merged_tokenizer/",
    output_dir="expanded_model/"
)

# Ready for fine-tuning! / 准备好进行微调！
```

---

## Module Overview / 模块概述

### 1. `train_chinese.py`

Train SentencePiece tokenizers optimized for Chinese text.

训练针对中文文本优化的SentencePiece分词器。

#### Main Functions / 主要函数

##### `train_chinese_tokenizer()`

```python
model_path = train_chinese_tokenizer(
    input_files="chinese_corpus/",      # Input text files or directory
    vocab_size=20000,                   # Target vocabulary size
    output_dir="tokenizer_output/",     # Output directory
    model_prefix="chinese_sp",          # Model file prefix
    model_type="bpe",                   # Model type: bpe, unigram, char, word
    character_coverage=0.9995,          # Character coverage (0.9995 for Chinese)
    split_digits=True,                  # Split digits into tokens
    byte_fallback=True,                 # Use byte fallback
    verbose=True                        # Print progress
)
```

##### `test_tokenizer()`

```python
results = test_tokenizer(
    model_path="tokenizer.model",
    test_texts={
        "example1": "你好，世界！",
        "example2": "Hello, world!"
    },
    verbose=True
)
```

##### `analyze_tokenizer()`

```python
analysis = analyze_tokenizer(
    model_path="tokenizer.model",
    top_n=50,                          # Number of top tokens to show
    verbose=True
)
print(f"Vocab size: {analysis['vocab_size']}")
```

---

### 2. `merge_vocab.py`

Merge vocabularies from different tokenizers.

合并不同分词器的词汇表。

#### Main Functions / 主要函数

##### `merge_tokenizers()`

```python
merged_tokenizer = merge_tokenizers(
    base_tokenizer_path="llama2_tokenizer/",
    additional_sp_model_path="chinese_sp.model",
    output_dir="merged_tokenizer/",
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    tokenizer_type="llama",             # or "auto"
    keep_duplicate_scores=False,
    verbose=True
)
```

##### `validate_merged_tokenizer()`

```python
results = validate_merged_tokenizer(
    tokenizer_path="merged_tokenizer/",
    test_texts={
        "chinese": "你好，世界！",
        "english": "Hello, world!",
        "mixed": "Hello 你好 World 世界"
    },
    verbose=True
)
```

##### `export_vocabulary()`

```python
export_vocabulary(
    tokenizer_path="merged_tokenizer/",
    output_file="vocab.json",
    format="json",                      # json, txt, or csv
    include_scores=False
)
```

---

### 3. `expand_embedding.py`

Expand model embeddings for new vocabularies.

为新词汇表扩展模型嵌入。

#### Main Functions / 主要函数

##### `expand_embeddings()`

```python
expanded_model = expand_embeddings(
    model=model,
    new_vocab_size=50000,
    pad_to_multiple_of=64,              # Pad for efficiency
    initialization_strategy="normal",    # normal, uniform, xavier, kaiming
    mean=0.0,
    std=0.02,
    verbose=True
)
```

##### `expand_model_for_tokenizer()`

```python
model, tokenizer = expand_model_for_tokenizer(
    model_path="original_model/",
    tokenizer_path="merged_tokenizer/",
    output_dir="expanded_model/",
    pad_to_multiple_of=64,
    initialization_strategy="normal",
    device_map="auto",                  # Device mapping
    torch_dtype=None,                   # Data type (or torch.float16, etc.)
    verbose=True
)
```

##### `verify_embedding_expansion()`

```python
results = verify_embedding_expansion(
    model=expanded_model,
    tokenizer=tokenizer,
    test_token_ids=[0, 100, 50000],
    verbose=True
)
print(f"Status: {results['status']}")
```

---

## Command-Line Scripts / 命令行脚本

### 1. `merge_tokenizers.py`

Merge tokenizer vocabularies from the command line.

从命令行合并分词器词汇表。

```bash
python scripts/merge_tokenizers.py \
    --base-tokenizer llama2_tokenizer/ \
    --chinese-model chinese_sp.model \
    --output-dir merged_tokenizer/ \
    --special-tokens "<|system|>" "<|user|>" "<|assistant|>" \
    --validate \
    --export-vocab
```

**Options / 选项:**
- `--base-tokenizer`: Base tokenizer directory / 基础分词器目录
- `--chinese-model`: Chinese SentencePiece model file / 中文SentencePiece模型文件
- `--output-dir`: Output directory / 输出目录
- `--special-tokens`: Special tokens to add / 要添加的特殊tokens
- `--tokenizer-type`: Type of tokenizer (llama/auto) / 分词器类型
- `--validate`: Validate merged tokenizer / 验证合并后的分词器
- `--export-vocab`: Export vocabulary to file / 导出词汇表到文件
- `--vocab-format`: Export format (json/txt/csv) / 导出格式

---

### 2. `expand_model_vocab.py`

Expand model embeddings from the command line.

从命令行扩展模型嵌入。

```bash
python scripts/expand_model_vocab.py \
    --model-path original_model/ \
    --tokenizer-path merged_tokenizer/ \
    --output-dir expanded_model/ \
    --init-strategy normal \
    --pad-to-multiple-of 64 \
    --dtype float16 \
    --verify
```

**Options / 选项:**
- `--model-path`: Original model directory / 原始模型目录
- `--tokenizer-path`: New tokenizer directory / 新分词器目录
- `--output-dir`: Output directory / 输出目录
- `--device-map`: Device mapping (auto/cpu/cuda) / 设备映射
- `--dtype`: Data type (float32/float16/bfloat16) / 数据类型
- `--pad-to-multiple-of`: Padding multiple / 填充倍数
- `--init-strategy`: Initialization strategy / 初始化策略
- `--verify`: Verify after expansion / 扩展后验证

---

### 3. `examples/tokenizer_workflow.py`

Complete workflow demonstration.

完整工作流程演示。

```bash
# Run complete workflow / 运行完整工作流程
python examples/tokenizer_workflow.py

# Show simple usage only / 仅显示简单用法
python examples/tokenizer_workflow.py --simple
```

---

## Advanced Usage / 高级用法

### Custom Initialization Strategies / 自定义初始化策略

```python
# Normal distribution (default) / 正态分布（默认）
model = expand_embeddings(model, new_vocab_size=50000, initialization_strategy="normal")

# Uniform distribution / 均匀分布
model = expand_embeddings(model, new_vocab_size=50000, initialization_strategy="uniform")

# Xavier initialization / Xavier初始化
model = expand_embeddings(model, new_vocab_size=50000, initialization_strategy="xavier")

# Kaiming initialization / Kaiming初始化
model = expand_embeddings(model, new_vocab_size=50000, initialization_strategy="kaiming")
```

### Training Large Corpora / 训练大型语料库

```python
# For very large corpora (>10M sentences) / 用于非常大的语料库（>1000万句）
model_path = train_chinese_tokenizer(
    input_files="large_corpus/",
    vocab_size=50000,
    output_dir="large_tokenizer/",
    train_extremely_large_corpus=True,  # Enable for large datasets
    max_sentence_length=32768,          # Increase for long sentences
    num_threads=32                      # Use more threads
)
```

### Mixed-Precision Training / 混合精度训练

```python
import torch

# Use float16 for faster inference / 使用float16以加快推理
model, tokenizer = expand_model_for_tokenizer(
    model_path="original_model/",
    tokenizer_path="merged_tokenizer/",
    output_dir="expanded_model/",
    torch_dtype=torch.float16
)

# Use bfloat16 for training / 使用bfloat16进行训练
model, tokenizer = expand_model_for_tokenizer(
    model_path="original_model/",
    tokenizer_path="merged_tokenizer/",
    output_dir="expanded_model/",
    torch_dtype=torch.bfloat16
)
```

---

## Best Practices / 最佳实践

### 1. **Vocabulary Size Selection** / **词汇表大小选择**

- **Chinese-only**: 15,000 - 25,000 tokens / 仅中文：15,000 - 25,000 tokens
- **Bilingual (EN+ZH)**: 40,000 - 60,000 tokens / 双语（英中）：40,000 - 60,000 tokens
- **Multilingual**: 60,000+ tokens / 多语言：60,000+ tokens

### 2. **Character Coverage** / **字符覆盖率**

- **Chinese**: 0.9995 (recommended) / 中文：0.9995（推荐）
- **English**: 1.0 / 英语：1.0
- **Mixed**: 0.9995 / 混合：0.9995

### 3. **Model Type Selection** / **模型类型选择**

- **BPE**: Best for most cases / 大多数情况下最佳
- **Unigram**: Good for morphologically rich languages / 适合词形丰富的语言
- **Character**: For very small models / 适用于非常小的模型
- **Word**: Rarely used for neural models / 很少用于神经模型

### 4. **Special Tokens** / **特殊Tokens**

Always add task-specific special tokens during merging:
始终在合并期间添加特定于任务的特殊tokens：

```python
special_tokens = [
    "<|system|>",      # System message marker
    "<|user|>",        # User input marker
    "<|assistant|>",   # Assistant response marker
    "<|im_start|>",    # Instruction start
    "<|im_end|>"       # Instruction end
]
```

### 5. **Embedding Padding** / **嵌入填充**

Pad vocabulary to multiples of 64 or 128 for GPU efficiency:
为GPU效率将词汇表填充到64或128的倍数：

```python
expand_model_for_tokenizer(
    ...,
    pad_to_multiple_of=64  # or 128 for larger models
)
```

---

## Troubleshooting / 故障排除

### Issue: Import Error / 问题：导入错误

```
ImportError: cannot import name 'train_chinese_tokenizer'
```

**Solution / 解决方案:**
```bash
# Make sure you're in the project root / 确保您在项目根目录
cd /path/to/LLM-from-Scratch

# Install in development mode / 以开发模式安装
pip install -e .
```

---

### Issue: protobuf Error / 问题：protobuf错误

```
TypeError: Descriptors cannot not be created directly
```

**Solution / 解决方案:**
```bash
# Downgrade protobuf / 降级protobuf
pip install protobuf==3.20.0

# Or set environment variable (already handled in code)
# 或设置环境变量（代码中已处理）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

---

### Issue: Out of Memory / 问题：内存不足

```
RuntimeError: CUDA out of memory
```

**Solution / 解决方案:**
```python
# Use CPU for expansion / 使用CPU进行扩展
expand_model_for_tokenizer(
    ...,
    device_map="cpu"
)

# Or use float16 / 或使用float16
import torch
expand_model_for_tokenizer(
    ...,
    torch_dtype=torch.float16
)
```

---

### Issue: Vocabulary Mismatch / 问题：词汇表不匹配

```
Warning: Vocabulary size mismatch: model=50000, tokenizer=49958
```

**Solution / 解决方案:**

This is usually fine due to padding. Use `pad_to_multiple_of` to align:
这通常是由于填充造成的。使用`pad_to_multiple_of`对齐：

```python
expand_model_for_tokenizer(
    ...,
    pad_to_multiple_of=64  # Ensures alignment
)
```

---

## File Structure / 文件结构

```
scratch_cs336/core/tokenizer/
├── __init__.py              # Module exports / 模块导出
├── train_chinese.py         # Chinese tokenizer training / 中文分词器训练
├── merge_vocab.py           # Vocabulary merging / 词汇表合并
└── expand_embedding.py      # Embedding expansion / 嵌入扩展

scripts/
├── merge_tokenizers.py          # Merge tokenizers CLI / 合并分词器命令行
└── expand_model_vocab.py        # Expand embeddings CLI / 扩展嵌入命令行

examples/
└── tokenizer_workflow.py        # Complete workflow / 完整工作流程

docs/
└── TOKENIZER.md             # This file / 本文件
```

---

## Examples / 示例

### Example 1: Train and Test Chinese Tokenizer / 示例1：训练和测试中文分词器

```python
from scratch_cs336.core.tokenizer import train_chinese_tokenizer, test_tokenizer

# Train / 训练
model_path = train_chinese_tokenizer(
    input_files="chinese_wiki/",
    vocab_size=20000,
    output_dir="chinese_tokenizer/"
)

# Test / 测试
test_tokenizer(
    model_path,
    test_texts={
        "greeting": "你好，世界！",
        "sentence": "翻译下面的句子为英文：有朋自远方来，不亦乐乎"
    }
)
```

---

### Example 2: Merge LLaMA with Chinese / 示例2：合并LLaMA与中文

```python
from scratch_cs336.core.tokenizer import merge_tokenizers, validate_merged_tokenizer

# Merge / 合并
tokenizer = merge_tokenizers(
    base_tokenizer_path="llama2-7b-hf/",
    additional_sp_model_path="chinese_sp_20k.model",
    output_dir="llama2_chinese/",
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"]
)

# Validate / 验证
validate_merged_tokenizer("llama2_chinese/")
```

---

### Example 3: Expand Model for New Vocabulary / 示例3：为新词汇表扩展模型

```python
from scratch_cs336.core.tokenizer import expand_model_for_tokenizer, verify_embedding_expansion

# Expand / 扩展
model, tokenizer = expand_model_for_tokenizer(
    model_path="llama2-7b-hf/",
    tokenizer_path="llama2_chinese/",
    output_dir="llama2_chinese_expanded/",
    initialization_strategy="normal"
)

# Verify / 验证
verify_embedding_expansion(model, tokenizer)
```

---

## API Reference / API参考

See individual module docstrings for detailed API documentation:
有关详细的API文档，请参阅各个模块的文档字符串：

- `train_chinese.py`: Training and analysis functions / 训练和分析函数
- `merge_vocab.py`: Merging and validation functions / 合并和验证函数
- `expand_embedding.py`: Expansion and verification functions / 扩展和验证函数

All functions include:
所有函数都包括：
- Type hints / 类型提示
- Bilingual docstrings (EN/ZH) / 双语文档字符串（英/中）
- Usage examples / 使用示例
- Error handling / 错误处理

---

## Contributing / 贡献

Contributions are welcome! Please ensure:
欢迎贡献！请确保：

1. Code follows existing style / 代码遵循现有风格
2. Bilingual comments and docstrings / 双语注释和文档字符串
3. Type hints for all functions / 所有函数的类型提示
4. Comprehensive error handling / 全面的错误处理
5. Example usage in docstrings / 文档字符串中的示例用法

---

## License / 许可证

MIT License

---

## Authors / 作者

LLM-from-Scratch Team

---

## Acknowledgments / 致谢

- Based on work from tiny-llm-zh / 基于tiny-llm-zh的工作
- SentencePiece library by Google / Google的SentencePiece库
- Transformers library by Hugging Face / Hugging Face的Transformers库
