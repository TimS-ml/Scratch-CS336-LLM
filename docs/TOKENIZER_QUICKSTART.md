# Tokenizer Quick Start Guide / 分词器快速入门指南

This guide provides quick examples for merging Chinese tokenization into your LLM.

本指南提供将中文分词合并到LLM的快速示例。

---

## 📋 Table of Contents / 目录

1. [Installation](#installation--安装)
2. [Three-Step Workflow](#three-step-workflow--三步工作流程)
3. [Command Line Usage](#command-line-usage--命令行使用)
4. [Python API Usage](#python-api-usage--python-api使用)
5. [File Locations](#file-locations--文件位置)

---

## 🔧 Installation / 安装

```bash
# Install dependencies / 安装依赖
pip install transformers sentencepiece torch

# Verify installation / 验证安装
python -c "from scratch_cs336.core.tokenizer import train_chinese_tokenizer; print('✓ Ready!')"
```

---

## 🚀 Three-Step Workflow / 三步工作流程

### Step 1️⃣: Train Chinese Tokenizer / 训练中文分词器

**Command Line / 命令行:**
```bash
python -m scratch_cs336.core.tokenizer.train_chinese \
    --input chinese_corpus/ \
    --vocab-size 20000 \
    --output-dir outputs/chinese_tokenizer
```

**Python API:**
```python
from scratch_cs336.core.tokenizer import train_chinese_tokenizer

model_path = train_chinese_tokenizer(
    input_files="chinese_corpus/",
    vocab_size=20000,
    output_dir="outputs/chinese_tokenizer"
)
```

---

### Step 2️⃣: Merge with Base Tokenizer / 与基础分词器合并

**Command Line / 命令行:**
```bash
python scripts/merge_tokenizers.py \
    --base-tokenizer path/to/llama2_tokenizer \
    --chinese-model outputs/chinese_tokenizer/chinese_sp.model \
    --output-dir outputs/merged_tokenizer \
    --special-tokens "<|system|>" "<|user|>" "<|assistant|>" \
    --validate
```

**Python API:**
```python
from scratch_cs336.core.tokenizer import merge_tokenizers

tokenizer = merge_tokenizers(
    base_tokenizer_path="llama2_tokenizer/",
    additional_sp_model_path="outputs/chinese_tokenizer/chinese_sp.model",
    output_dir="outputs/merged_tokenizer/",
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"]
)
```

---

### Step 3️⃣: Expand Model Embeddings / 扩展模型嵌入

**Command Line / 命令行:**
```bash
python scripts/expand_model_vocab.py \
    --model-path path/to/original_model \
    --tokenizer-path outputs/merged_tokenizer \
    --output-dir outputs/expanded_model \
    --verify
```

**Python API:**
```python
from scratch_cs336.core.tokenizer import expand_model_for_tokenizer

model, tokenizer = expand_model_for_tokenizer(
    model_path="original_model/",
    tokenizer_path="outputs/merged_tokenizer/",
    output_dir="outputs/expanded_model/"
)
```

---

## 💻 Command Line Usage / 命令行使用

### Complete Workflow in One Go / 一次性完成整个工作流程

```bash
# See the complete example / 查看完整示例
python examples/tokenizer_workflow.py --simple

# Run the complete workflow (with your paths) / 运行完整工作流程（使用您的路径）
python examples/tokenizer_workflow.py
```

### Individual Commands / 单独命令

#### Merge Tokenizers / 合并分词器
```bash
python scripts/merge_tokenizers.py \
    --base-tokenizer llama2_tokenizer/ \
    --chinese-model chinese_sp.model \
    --output-dir merged_tokenizer/ \
    --special-tokens "<|system|>" "<|user|>" "<|assistant|>" \
    --validate \
    --export-vocab \
    --vocab-format json
```

#### Expand Model / 扩展模型
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

---

## 🐍 Python API Usage / Python API使用

### Import Everything / 导入所有功能

```python
from scratch_cs336.core.tokenizer import (
    # Training / 训练
    train_chinese_tokenizer,
    test_tokenizer,
    analyze_tokenizer,

    # Merging / 合并
    merge_tokenizers,
    validate_merged_tokenizer,
    export_vocabulary,

    # Expansion / 扩展
    expand_embeddings,
    expand_model_for_tokenizer,
    verify_embedding_expansion,
)
```

### Example: Complete Workflow / 示例：完整工作流程

```python
# 1. Train Chinese tokenizer / 训练中文分词器
chinese_model = train_chinese_tokenizer(
    input_files="data/chinese_corpus",
    vocab_size=20000,
    output_dir="outputs/chinese_tokenizer",
    model_type="bpe"
)

# 2. Merge vocabularies / 合并词汇表
merged_tokenizer = merge_tokenizers(
    base_tokenizer_path="models/llama2_tokenizer",
    additional_sp_model_path=chinese_model,
    output_dir="outputs/merged_tokenizer",
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"]
)

# 3. Expand model embeddings / 扩展模型嵌入
model, tokenizer = expand_model_for_tokenizer(
    model_path="models/llama2-7b",
    tokenizer_path="outputs/merged_tokenizer",
    output_dir="outputs/llama2_chinese"
)

# 4. Verify / 验证
verify_embedding_expansion(model, tokenizer, verbose=True)

print("✓ Ready for fine-tuning on Chinese data!")
print("✓ 准备在中文数据上微调！")
```

### Example: Advanced Options / 示例：高级选项

```python
import torch

# Train with custom settings / 使用自定义设置训练
chinese_model = train_chinese_tokenizer(
    input_files=["corpus1.txt", "corpus2.txt", "corpus3.txt"],
    vocab_size=30000,
    output_dir="outputs/tokenizer",
    model_type="bpe",
    character_coverage=0.9995,
    split_digits=True,
    byte_fallback=True,
    max_sentence_length=32768,
    num_threads=16
)

# Merge with validation / 合并并验证
merged_tokenizer = merge_tokenizers(
    base_tokenizer_path="llama2_tokenizer",
    additional_sp_model_path=chinese_model,
    output_dir="merged_tokenizer",
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"],
    keep_duplicate_scores=False,
    verbose=True
)

# Test merged tokenizer / 测试合并后的分词器
validate_merged_tokenizer(
    "merged_tokenizer",
    test_texts={
        "chinese": "你好，世界！这是一个测试。",
        "english": "Hello, world! This is a test.",
        "mixed": "Hello 你好 World 世界！"
    }
)

# Expand with mixed precision / 使用混合精度扩展
model, tokenizer = expand_model_for_tokenizer(
    model_path="llama2-7b",
    tokenizer_path="merged_tokenizer",
    output_dir="llama2_chinese",
    pad_to_multiple_of=128,
    initialization_strategy="xavier",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
```

---

## 📁 File Locations / 文件位置

### Core Modules / 核心模块
```
scratch_cs336/core/tokenizer/
├── __init__.py              # Module exports
├── train_chinese.py         # Chinese tokenizer training
├── merge_vocab.py           # Vocabulary merging
└── expand_embedding.py      # Embedding expansion
```

### Scripts / 脚本
```
scripts/
├── merge_tokenizers.py          # CLI for merging tokenizers
└── expand_model_vocab.py        # CLI for expanding embeddings

examples/
└── tokenizer_workflow.py        # Complete workflow demo
```

### Source Files (from tiny-llm-zh) / 源文件（来自tiny-llm-zh）
```
tiny-llm-zh/tokenizer/
├── expend_tokenizer.py      # Original merge implementation
├── expend_embedding.py      # Original embedding expansion
└── train_chinese_sp.py      # Original Chinese training
```

---

## 🎯 Common Use Cases / 常见用例

### Use Case 1: Add Chinese to LLaMA / 为LLaMA添加中文

```python
from scratch_cs336.core.tokenizer import *

# Step 1: Train Chinese tokenizer (20K tokens)
chinese_model = train_chinese_tokenizer(
    input_files="chinese_wiki/",
    vocab_size=20000,
    output_dir="chinese_tok"
)

# Step 2: Merge with LLaMA (32K → 49K tokens)
merge_tokenizers(
    base_tokenizer_path="llama-2-7b-hf/",
    additional_sp_model_path=f"{chinese_model}",
    output_dir="llama_chinese_tok/",
    special_tokens=["<|system|>", "<|user|>", "<|assistant|>"]
)

# Step 3: Expand model
expand_model_for_tokenizer(
    model_path="llama-2-7b-hf/",
    tokenizer_path="llama_chinese_tok/",
    output_dir="llama_chinese_model/"
)
```

---

### Use Case 2: Bilingual Model / 双语模型

```python
# Larger Chinese vocabulary for better coverage
chinese_model = train_chinese_tokenizer(
    input_files=["chinese_wiki/", "chinese_books/"],
    vocab_size=30000,  # Larger vocab
    output_dir="chinese_tok_30k"
)

# Merge and expand
merge_tokenizers(
    base_tokenizer_path="llama-2-13b-hf/",
    additional_sp_model_path=chinese_model,
    output_dir="llama_bilingual_tok/"
)

expand_model_for_tokenizer(
    model_path="llama-2-13b-hf/",
    tokenizer_path="llama_bilingual_tok/",
    output_dir="llama_bilingual_model/",
    pad_to_multiple_of=128  # Better GPU efficiency
)
```

---

### Use Case 3: Custom Special Tokens / 自定义特殊Tokens

```python
# Define custom tokens for your application
special_tokens = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|code|>",      # For code sections
    "<|math|>",      # For math expressions
    "<|think|>",     # For chain-of-thought
]

merge_tokenizers(
    base_tokenizer_path="llama-2-7b-hf/",
    additional_sp_model_path="chinese_sp.model",
    output_dir="custom_tokenizer/",
    special_tokens=special_tokens
)
```

---

## 📊 Expected Vocabulary Sizes / 预期词汇表大小

| Base Model | Original Vocab | Chinese Tokens | Final Vocab |
|------------|----------------|----------------|-------------|
| LLaMA-2    | 32,000        | 20,000         | ~49,000     |
| LLaMA-3    | 128,000       | 20,000         | ~145,000    |
| Qwen       | 151,936       | -              | 151,936     |

*Note: Duplicates are removed, so final size may be smaller*

*注意：会删除重复项，因此最终大小可能更小*

---

## 🔍 Validation and Testing / 验证和测试

### Validate Merged Tokenizer / 验证合并后的分词器

```python
from scratch_cs336.core.tokenizer import validate_merged_tokenizer

results = validate_merged_tokenizer(
    "merged_tokenizer/",
    test_texts={
        "chinese": "你好，世界！",
        "english": "Hello, world!",
        "mixed": "Hello 你好"
    }
)
```

### Verify Model Expansion / 验证模型扩展

```python
from scratch_cs336.core.tokenizer import verify_embedding_expansion

results = verify_embedding_expansion(
    model=model,
    tokenizer=tokenizer,
    test_token_ids=[0, 1000, 40000, 48000]
)

if results["status"] == "success":
    print("✓ Expansion successful!")
```

---

## ⚙️ Configuration Options / 配置选项

### Tokenizer Training / 分词器训练

| Parameter | Default | Description |
|-----------|---------|-------------|
| vocab_size | - | Vocabulary size / 词汇表大小 |
| model_type | "bpe" | bpe, unigram, char, word |
| character_coverage | 0.9995 | Character coverage (0-1) / 字符覆盖率 |
| split_digits | True | Split numbers / 拆分数字 |
| byte_fallback | True | Handle unknown chars / 处理未知字符 |

### Embedding Expansion / 嵌入扩展

| Parameter | Default | Description |
|-----------|---------|-------------|
| initialization_strategy | "normal" | normal, uniform, xavier, kaiming |
| pad_to_multiple_of | 64 | Padding for efficiency / 填充以提高效率 |
| device_map | "auto" | Device placement / 设备放置 |
| torch_dtype | None | Data type / 数据类型 |

---

## 🐛 Troubleshooting / 故障排除

### Problem: Module not found / 问题：找不到模块
```bash
# Solution: Install in development mode / 解决方案：以开发模式安装
cd /path/to/LLM-from-Scratch
pip install -e .
```

### Problem: protobuf error / 问题：protobuf错误
```bash
# Solution: Use compatible version / 解决方案：使用兼容版本
pip install protobuf==3.20.0
```

### Problem: Out of memory / 问题：内存不足
```python
# Solution: Use CPU or mixed precision / 解决方案：使用CPU或混合精度
expand_model_for_tokenizer(..., device_map="cpu")
# or / 或
import torch
expand_model_for_tokenizer(..., torch_dtype=torch.float16)
```

---

## 📚 Full Documentation / 完整文档

For detailed documentation, see:
有关详细文档，请参阅：

- **Module Documentation**: `docs/TOKENIZER.md`
- **Script Help**: `python scripts/merge_tokenizers.py --help`
- **Examples**: `python examples/tokenizer_workflow.py`

---

## 🚀 Next Steps / 下一步

After expanding your model:
扩展模型后：

1. **Fine-tune on Chinese data** / **在中文数据上微调**
   ```bash
   python scripts/train_sft.py --model outputs/expanded_model --data chinese_dataset
   ```

2. **Test performance** / **测试性能**
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer

   model = AutoModelForCausalLM.from_pretrained("outputs/expanded_model")
   tokenizer = AutoTokenizer.from_pretrained("outputs/expanded_model")

   inputs = tokenizer("你好，世界！", return_tensors="pt")
   outputs = model.generate(**inputs, max_length=50)
   print(tokenizer.decode(outputs[0]))
   ```

3. **Evaluate on benchmarks** / **在基准测试上评估**
   - C-Eval (Chinese knowledge)
   - CMMLU (Chinese understanding)
   - Your custom evaluation tasks

---

## 📞 Support / 支持

- **GitHub Issues**: Report bugs and request features
- **Documentation**: See README files in each module
- **Examples**: Check `examples/tokenizer_workflow.py`

---

**Happy training! / 祝训练顺利！** 🎉
