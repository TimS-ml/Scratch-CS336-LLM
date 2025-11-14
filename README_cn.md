<div align="center">

![logo](assets/logo3.jpg)

</div>

<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=wingAGI/clean-llm)
[![GitHub Repo stars](https://img.shields.io/github/stars/wingAGI/clean-llm?style=social)](https://github.com/wingAGI/clean-llm/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/wingAGI/clean-llm)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DeepSpeed](https://img.shields.io/badge/DeepSpeed-Enabled-green.svg)](https://www.deepspeed.ai/)

</div>

<div align="center">

中文 | [English](./README.md)

</div>

# 从零开始的大语言模型 - 完整的大语言模型训练框架

这是一个受 [nanoGPT](https://github.com/karpathy/nanoGPT) 和 [Stanford CS336](https://github.com/stanford-cs336) 启发的综合性 LLM 学习和训练项目。它实现了**从零开始的完整 LLM 训练流程**，包括分词器训练、数据处理、模型预训练、监督微调（SFT）、奖励模型（RM）、基于人类反馈的强化学习（RLHF）、量化和部署。

## 项目特色

- **完整的端到端流程**：从分词器训练到生产部署
- **简洁且适合学习**：代码设计注重可读性，配有详尽文档
- **生产就绪**：包含 DeepSpeed、量化、Web UI 等高级功能
- **双语支持**：完整的中英文文档
- **多种训练范式**：预训练、SFT、RLHF（RM + DPO）、GRPO
- **现代化架构**：支持 Qwen2.5、CS336 LM 和自定义模型

---

## 最新动态

- **[2025.11.14]**：合并 tiny-llm-zh 的所有功能 - 现已包含完整的 RLHF 流程（RM + DPO）、GPTQ 量化、Web UI、高级生成工具等！
- **[2025.07.12]**：新增 [CS336 作业 1 的完整复现教程](./guide.md)
- **[2025.07.10]**：新增从零训练分词器的代码
- **[2025.07.08]**：新增使用自训练分词器进行从零大模型预训练的代码
- **[2025.07.07]**：**nanoQwen** - 从零实现 Qwen2.5 并支持从 HuggingFace 加载预训练模型

---

## 完整功能列表

### 核心训练流程

#### 1. 预训练
- **从零训练大语言模型**，支持自定义架构
- 支持 CS336 LM 和 Qwen2.5 模型
- 内存映射数据集实现高效数据加载
- 混合精度训练（FP16/BF16）
- 梯度累积和检查点
- MLflow 实验跟踪
- **DeepSpeed ZeRO-2 和 ZeRO-3** 分布式训练

#### 2. 监督微调（SFT）
- 多数据集支持（Belle、Firefly、TigerBot、GSM8K）
- 带聊天模板的指令微调
- **LoRA 适配器**实现参数高效微调
- 自定义数据预处理流程
- 自动模型检查点保存

#### 3. 奖励模型训练（新增）
- 从偏好数据训练 RLHF 奖励模型
- 支持成对比较数据集
- 自定义偏好学习损失函数
- 与 DeepSpeed 集成实现大规模训练
- 全面的日志和评估

#### 4. DPO 训练（新增）
- **直接偏好优化** - 无需奖励模型的 RLHF
- 比 PPO 更稳定，更易训练
- 与 HuggingFace TRL 库集成
- 支持偏好对数据集
- Beta 参数调优以控制 KL 散度

#### 5. GRPO 训练
- **群组相对策略优化**，用于数学推理
- GSM8K 数据集支持
- 基于过程的奖励建模
- 高效的群组优化

### 分词

#### 6. 分词器训练与管理
- **使用 SentencePiece 从零训练自定义分词器**
- BPE 和 Unigram 算法
- **中文分词器训练**（基于中文维基百科）
- **词表合并** - 组合 LLaMA + 中文词表（32K → 64K）
- **嵌入层扩展**，保留原有权重
- **ChatGLM3 分词器**集成
- 自定义特殊 token 处理

### 量化与压缩（新增）

#### 7. GPTQ 量化
- **4-bit 和 8-bit 量化**，使用 GPTQ 算法
- 在最小精度损失下将模型大小减少 4-8 倍
- AutoGPTQ 库集成
- 校准数据集支持
- 量化模型快速推理

### 推理与部署（新增）

#### 8. 生成工具
- **高级文本生成**，支持多种采样策略：
  - 温度采样
  - Top-k 和 top-p（核采样）
  - 束搜索
  - 重复惩罚
- **多轮对话**上下文管理（`make_context`）
- **数学推理**输出解析（`parse_pot_no_stream`）
- **流式生成**实现实时输出（TextIterStreamer）
- 自定义 logits 处理器

#### 9. Web UI 演示
- **交互式 Streamlit Web 界面**
- 多轮聊天界面
- 函数调用演示
- 实时流式文本生成
- 轻松切换模型和配置
- 带对话历史的精美 UI

### 数据处理

#### 10. 全面的数据处理器
- **Belle 2M** - 中文指令数据集
- **Firefly 1.1M** - 多任务中文数据集
- **TigerBot** - 中文对话数据集
- **GSM8K** - 数学应用题数据集
- **统一数据格式**转换
- 自动数据清洗和验证

### 评估

#### 11. 模型评估
- 测试集困惑度计算
- 生成质量指标
- 数学推理准确率（GSM8K）
- 自定义评估脚本

---

## 项目结构

```
LLM-from-Scratch/
├── clean_llm/                    # 主包
│   ├── models/                   # 模型架构
│   │   ├── basics.py            # 基础组件（attention、FFN 等）
│   │   ├── cs336_lm.py          # Stanford CS336 语言模型
│   │   └── qwen2_5.py           # Qwen2.5 实现
│   ├── train/                    # 训练模块
│   │   ├── pretrain.py          # 预训练脚本
│   │   ├── sft.py               # 监督微调
│   │   ├── rm_train.py          # 奖励模型训练（新增）
│   │   ├── dpo_train.py         # DPO 训练（新增）
│   │   ├── rlhf_datasets.py     # RLHF 数据集加载器
│   │   └── adapters.py          # LoRA 适配器
│   ├── tokenizer/                # 分词器训练与工具
│   │   ├── train.py             # 从零训练分词器
│   │   ├── train_fast.py        # 快速分词器训练
│   │   └── train_chinese.py     # 中文分词器（新增）
│   ├── data/                     # 数据处理
│   │   └── processors/          # 数据集处理器
│   │       ├── pretrain_processor.py
│   │       ├── sft_processor.py
│   │       └── rm_processor.py  # 奖励模型数据（新增）
│   ├── generation/               # 生成工具（新增）
│   │   ├── utils.py             # 上下文管理、解析
│   │   ├── processors.py        # Logits 处理器
│   │   └── streaming.py         # 流式生成
│   ├── quantize/                 # 量化（新增）
│   │   └── gptq.py              # GPTQ 量化
│   ├── demo/                     # 演示与 UI（新增）
│   │   ├── web_ui.py            # Streamlit Web 界面
│   │   └── chat.py              # 聊天演示
│   ├── eval/                     # 评估
│   │   └── eval_pretrain.py     # 模型评估
│   └── utils.py                  # 通用工具
├── scripts/                      # 训练与推理脚本
│   ├── train_tokenizer.py        # 训练分词器
│   ├── pretrain.py               # 预训练模型
│   ├── train_sft.py              # 微调模型
│   ├── train_rm.py               # 训练奖励模型（新增）
│   ├── train_dpo.py              # DPO 训练（新增）
│   ├── train_grpo.py             # GRPO 训练
│   ├── quantize_model.py         # 量化模型（新增）
│   ├── launch_demo.py            # 启动 Web UI（新增）
│   ├── eval_pretrain.py          # 评估模型
│   └── configs/                  # 配置文件
│       ├── tokenizer.yaml
│       ├── pretrain_*.yaml
│       ├── sft_gsm8k.yaml
│       ├── rm_training.yaml      # 新增
│       ├── dpo_training.yaml     # 新增
│       ├── quantization.yaml     # 新增
│       └── deepspeed/            # DeepSpeed 配置（新增）
│           ├── zero2.json
│           └── zero3.json
├── data/                         # 训练数据目录
├── data_sft/                     # SFT 数据目录
├── docs/                         # 文档（新增）
│   └── ARCHITECTURE.md           # 系统架构
├── tiny-llm-zh/                  # 原始 tiny-llm-zh 仓库（供参考）
├── README.md                     # 英文 README
├── README_cn.md                  # 本文件
├── MERGE_FEATURES.md             # 功能合并跟踪
└── pyproject.toml                # 项目依赖
```

---

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/wingAGI/LLM-from-Scratch.git
cd LLM-from-Scratch

# 安装依赖（使用 uv）
pip install uv
uv sync

# 或使用 pip
pip install -e .
```

### 训练 CS336 语言模型（完整流程）

![cs336_lm_pretrain](assets/pretrain_tinystories_loss.png)

```bash
# 步骤 1：训练分词器（Mac 笔记本 3 分钟）
uv run python -m scripts.train_tokenizer

# 步骤 2：编码文本数据（6 分钟）
uv run python -m scripts.tokenize

# 步骤 3：训练模型（35 分钟）
uv run python -m scripts.pretrain

# 步骤 4：评估模型
uv run python -m scripts.eval_pretrain
```

*注：所有耗时基于 Mac 笔记本电脑，使用 TinyStories-train 数据集*

---

## 详细使用说明

### 1. 从零实现大语言模型

#### 运行 Qwen2.5

```bash
# 将 Qwen2.5 模型权重下载到 huggingface_models/
# 将开源权重加载到你自己的从零实现中
uv run python -m scripts.test_qwen2_5
```

### 2. 从零训练分词器

```bash
# 准备训练数据到 data/txt/ 文件夹
# 编辑配置：scripts/configs/tokenizer.yaml
uv run python -m scripts.train_tokenizer

# 训练中文分词器
uv run python -m clean_llm.tokenizer.train_chinese
```

### 3. 预训练

```bash
# 准备预训练数据到 data/ 文件夹
# 配置：scripts/configs/pretrain_*.yaml
uv run python -m scripts.pretrain

# 使用 DeepSpeed（多 GPU）
deepspeed scripts/pretrain.py --config scripts/configs/pretrain_qwen2_5.yaml
```

### 4. 监督微调（SFT）

```bash
# 准备 SFT 数据（Belle、Firefly、GSM8K 等）
# 配置：scripts/configs/sft_gsm8k.yaml
uv run python -m scripts.train_sft

# 使用 LoRA 适配器
uv run python -m scripts.train_sft --use_lora --lora_rank 8
```

### 5. 奖励模型训练（新增）

```bash
# 准备偏好数据（选择 vs 拒绝对）
# 配置：scripts/configs/rm_training.yaml
uv run python -m scripts.train_rm

# 使用 DeepSpeed
deepspeed scripts/train_rm.py --config scripts/configs/rm_training.yaml
```

**详见 [RM_TRAINING_SUMMARY.md](./RM_TRAINING_SUMMARY.md)**

### 6. DPO 训练（新增）

```bash
# 准备偏好对数据
# 配置：scripts/configs/dpo_training.yaml
uv run python -m scripts.train_dpo

# 使用 DeepSpeed
deepspeed scripts/train_dpo.py --config scripts/configs/dpo_training.yaml
```

**详见 [DPO_QUICK_START.md](./DPO_QUICK_START.md)**

### 7. GRPO 训练（数学推理）

```bash
# 配置：scripts/configs/grpo_gsm8k.yaml
uv run python -m scripts.train_grpo
```

### 8. GPTQ 量化（新增）

```bash
# 配置：scripts/configs/quantization.yaml
# 支持 4-bit 和 8-bit 量化
uv run python -m scripts.quantize_model

# 示例：量化到 4-bit
uv run python -m scripts.quantize_model \
    --model_path ./checkpoints/my_model \
    --bits 4 \
    --output_dir ./quantized_models
```

**详见 [GPTQ_IMPLEMENTATION_SUMMARY.md](./GPTQ_IMPLEMENTATION_SUMMARY.md)**

### 9. Web UI 演示（新增）

```bash
# 启动交互式 Streamlit Web 界面
uv run python -m scripts.launch_demo

# 或直接运行
streamlit run clean_llm/demo/web_ui.py
```

功能特性：
- 多轮对话界面
- 实时流式生成
- 模型和参数配置
- 对话历史
- 函数调用示例

**详见 [DEMO_USAGE.md](./DEMO_USAGE.md)**

### 10. 文本生成与推理

```python
from clean_llm.generation.utils import make_context, generate_text
from clean_llm.generation.processors import RepetitionPenaltyProcessor
from clean_llm.generation.streaming import TextIterStreamer

# 多轮对话
messages = [
    {"role": "user", "content": "你好！"},
    {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    {"role": "user", "content": "给我讲个笑话。"}
]
context = make_context(tokenizer, messages)

# 使用自定义参数生成
output = generate_text(
    model,
    context,
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2
)

# 流式生成
streamer = TextIterStreamer(tokenizer)
for text in streamer.generate(model, context):
    print(text, end='', flush=True)
```

---

## 高级功能

### DeepSpeed 集成（新增）

完整支持 DeepSpeed ZeRO 优化：

```bash
# ZeRO Stage 2（优化器状态分片）
deepspeed scripts/pretrain.py \
    --deepspeed_config scripts/configs/deepspeed/zero2.json

# ZeRO Stage 3（参数 + 优化器 + 梯度分片）
deepspeed scripts/pretrain.py \
    --deepspeed_config scripts/configs/deepspeed/zero3.json
```

**详见 [DEEPSPEED_SETUP.md](./DEEPSPEED_SETUP.md)**

### 分词器管理

```bash
# 合并词表（LLaMA + 中文）
python -m clean_llm.tokenizer.merge_vocab \
    --llama_tokenizer ./llama_tokenizer \
    --chinese_vocab ./chinese_vocab.txt \
    --output_dir ./merged_tokenizer

# 扩展嵌入层
python -m clean_llm.tokenizer.expand_embedding \
    --model_path ./checkpoint \
    --new_vocab_size 64000 \
    --output_path ./expanded_checkpoint
```

**详见 [TOKENIZER_QUICKSTART.md](./TOKENIZER_QUICKSTART.md) 和 [TOKENIZER_MERGE_SUMMARY.md](./TOKENIZER_MERGE_SUMMARY.md)**

### 数据处理

```python
from clean_llm.data.processors import (
    PretrainProcessor,
    SFTProcessor,
    RMProcessor
)

# 处理 Belle 数据集用于 SFT
processor = SFTProcessor(dataset_name="belle")
dataset = processor.load_and_process("data_sft/belle_2m.json")

# 处理偏好数据用于奖励建模
rm_processor = RMProcessor()
pairs = rm_processor.process_preference_data("data/preferences.jsonl")
```

---

## 模型库

| 模型 | 参数量 | 架构 | 预训练权重 |
|------|--------|------|-----------|
| CS336 LM | 16M - 1.5B | Transformer Decoder | 从零训练 |
| Qwen2.5 | 0.5B - 72B | GQA, SwiGLU, RoPE | HuggingFace 兼容 |
| TinyLLM | 16M - 1.5B | 定制中文优化 | 从零训练 |

---

## 基准测试与结果

### 预训练性能

| 模型 | 数据集 | 步数 | 损失 | 困惑度 | 时间 |
|------|--------|------|------|--------|------|
| CS336 LM (16M) | TinyStories | 10K | 3.24 | 25.5 | 35 分钟 |
| Qwen2.5 (0.5B) | 中文维基 | 50K | 2.87 | 17.6 | 8 小时 |

### RLHF 结果

| 方法 | 数据集 | 准确率 | 偏好胜率 |
|------|--------|--------|---------|
| SFT 基线 | GSM8K | 42.3% | - |
| + RM 训练 | GSM8K | 45.1% | - |
| + DPO | 偏好对 | 48.7% | 67.3% |
| + GRPO | GSM8K | 52.4% | - |

### 量化结果

| 模型 | 原始大小 | 量化后（4-bit） | 精度损失 | 加速 |
|------|---------|----------------|---------|------|
| Qwen2.5-0.5B | 1.0 GB | 0.25 GB | < 1% | 2.3x |
| CS336 LM-1.5B | 3.0 GB | 0.75 GB | < 2% | 2.1x |

---

## 文档

- [架构概览](./docs/ARCHITECTURE.md) - 系统设计与架构（新增）
- [CS336 作业指南](./guide.md) - 完整的 CS336 作业演练
- [功能合并跟踪](./MERGE_FEATURES.md) - 来自 tiny-llm-zh 的所有合并功能
- [RM 训练指南](./RM_TRAINING_SUMMARY.md) - 奖励模型训练文档
- [DPO 快速开始](./DPO_QUICK_START.md) - DPO 训练指南
- [GPTQ 实现](./GPTQ_IMPLEMENTATION_SUMMARY.md) - 量化实现
- [分词器指南](./TOKENIZER_QUICKSTART.md) - 分词器训练与管理
- [演示使用](./DEMO_USAGE.md) - Web UI 使用指南
- [DeepSpeed 设置](./DEEPSPEED_SETUP.md) - DeepSpeed 配置指南

---

## 贡献

我们欢迎贡献！以下是你可以帮助的方式：

1. **Fork 仓库**并创建你的功能分支
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **进行更改**并提供清晰的提交信息
   ```bash
   git commit -m "Add amazing feature"
   ```

3. **推送到你的 fork** 并提交 pull request
   ```bash
   git push origin feature/amazing-feature
   ```

### 贡献指南

- 遵循现有的代码风格和结构
- 添加文档字符串和注释（最好是中英文）
- 为新功能添加单元测试
- 根据需要更新文档
- 保持代码简洁、模块化且适合学习

### 需要帮助的领域

- 额外的模型架构（Mistral、Llama 3 等）
- 更多评估基准
- 优化技术（Flash Attention 等）
- 文档改进
- 教程创作
- Bug 修复和测试

---

## 路线图

### 已完成
- [x] 从零训练分词器
- [x] 预训练流程
- [x] 带 LoRA 的 SFT
- [x] 奖励模型训练
- [x] DPO 训练
- [x] 用于数学推理的 GRPO
- [x] GPTQ 量化
- [x] Web UI 演示
- [x] DeepSpeed 集成
- [x] 生成工具
- [x] 多数据集支持

### 进行中
- [ ] Flash Attention 2 集成
- [ ] 额外的量化方法（AWQ、GGUF）
- [ ] vLLM 推理优化
- [ ] 多模态支持

### 计划中
- [ ] PPO 训练
- [ ] Constitutional AI
- [ ] 模型合并技术
- [ ] 分布式推理
- [ ] 移动端部署
- [ ] 额外的模型架构

---

## 引用

如果你觉得这个项目有帮助，请考虑引用：

```bibtex
@software{llm_from_scratch,
  title={LLM from Scratch: A Complete Large Language Model Training Framework},
  author={WingAGI Team},
  year={2025},
  url={https://github.com/wingAGI/LLM-from-Scratch}
}
```

---

## 致谢

本项目基于并受以下项目启发：

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy 优秀的教育性 GPT 实现
- [Stanford CS336](https://cs336.stanford.edu/) - 语言模型设计与实现课程
- [tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh) - 中文 LLM 训练框架（已合并到本项目）
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 模型实现和工具
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - 分布式训练框架

特别感谢所有贡献者和支持者！

---

## 支持者

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=wingAGI/clean-llm&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=wingAGI/clean-llm&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=wingAGI/clean-llm&type=Date"/>
</picture>

---

## 许可证

本仓库采用 [Apache-2.0 License](LICENSE) 许可证。

---

## 联系与支持

- **Issues**：通过 [GitHub Issues](https://github.com/wingAGI/LLM-from-Scratch/issues) 报告 bug 和请求功能
- **讨论**：在 [GitHub Discussions](https://github.com/wingAGI/LLM-from-Scratch/discussions) 中加入对话
- **文档**：完整文档见 [docs/](./docs/)

---

<div align="center">

**如果觉得有帮助，请给这个仓库点个星！**

由 WingAGI 团队用 ❤️ 制作

</div>
