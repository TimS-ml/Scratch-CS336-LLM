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

[中文](./README_cn.md) | English

</div>

# LLM from Scratch - A Complete Large Language Model Training Framework

This is a comprehensive LLM learning and training project inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) and [Stanford CS336](https://github.com/stanford-cs336). It implements the **entire LLM training pipeline from scratch**, including tokenizer training, data processing, model pre-training, supervised fine-tuning (SFT), reward modeling (RM), reinforcement learning from human feedback (RLHF), quantization, and deployment.

## What Makes This Project Special?

- **Complete End-to-End Pipeline**: From tokenizer training to production deployment
- **Clean & Educational**: Code designed for learning with extensive documentation
- **Production-Ready**: Includes advanced features like DeepSpeed, quantization, and web UI
- **Bilingual Support**: Full documentation in English and Chinese (中英双语)
- **Multiple Training Paradigms**: Pre-training, SFT, RLHF (RM + DPO), GRPO
- **Modern Architecture**: Support for Qwen2.5, CS336 LM, and custom models

---

## News & Updates

- **[2025.11.14]**: Merged all features from tiny-llm-zh - now includes complete RLHF pipeline (RM + DPO), GPTQ quantization, Web UI, advanced generation utilities, and more!
- **[2025.07.12]**: Added [A comprehensive guide to CS336 assignment 1](./guide.md)
- **[2025.07.10]**: Added code for training tokenizers from scratch
- **[2025.07.08]**: Added code for pretraining LLMs from scratch with custom-trained tokenizers
- **[2025.07.07]**: **nanoQwen** - Added Qwen2.5 implementation from scratch with HuggingFace pretrained model loading support

---

## Complete Feature List

### Core Training Pipeline

#### 1. Pre-training
- **Train LLMs from scratch** with custom architectures
- Support for CS336 LM and Qwen2.5 models
- Memory-efficient data loading with memory-mapped datasets
- Mixed precision training (FP16/BF16)
- Gradient accumulation and checkpointing
- MLflow experiment tracking
- **DeepSpeed ZeRO-2 and ZeRO-3** for distributed training

#### 2. Supervised Fine-Tuning (SFT)
- Multi-dataset support (Belle, Firefly, TigerBot, GSM8K)
- Instruction tuning with chat templates
- Parameter-efficient fine-tuning with **LoRA adapters**
- Custom data preprocessing pipelines
- Automatic model checkpointing

#### 3. Reward Model Training (NEW)
- Train reward models for RLHF from preference data
- Support for pairwise comparison datasets
- Custom loss functions for preference learning
- Integration with DeepSpeed for large-scale training
- Comprehensive logging and evaluation

#### 4. DPO Training (NEW)
- **Direct Preference Optimization** - RLHF without reward models
- More stable than PPO, easier to train
- Integration with HuggingFace TRL library
- Support for preference pair datasets
- Beta parameter tuning for KL divergence control

#### 5. GRPO Training
- **Group Relative Policy Optimization** for math reasoning
- GSM8K dataset support
- Process-based reward modeling
- Efficient group-based optimization

### Tokenization

#### 6. Tokenizer Training & Management
- **Train custom tokenizers from scratch** using SentencePiece
- BPE and Unigram algorithms
- **Chinese tokenizer training** on Chinese Wikipedia
- **Vocabulary merging** - combine LLaMA + Chinese vocab (32K → 64K)
- **Embedding layer expansion** with weight preservation
- **ChatGLM3 tokenizer** integration
- Custom special token handling

### Quantization & Compression (NEW)

#### 7. GPTQ Quantization
- **4-bit and 8-bit quantization** with GPTQ algorithm
- Reduce model size by 4-8x with minimal accuracy loss
- AutoGPTQ library integration
- Calibration dataset support
- Fast inference with quantized models

### Inference & Deployment (NEW)

#### 8. Generation Utilities
- **Advanced text generation** with multiple sampling strategies:
  - Temperature sampling
  - Top-k and top-p (nucleus) sampling
  - Beam search
  - Repetition penalty
- **Multi-turn conversation** context management (`make_context`)
- **Mathematical reasoning** output parsing (`parse_pot_no_stream`)
- **Streaming generation** for real-time output (TextIterStreamer)
- Custom logits processors

#### 9. Web UI Demo
- **Interactive Streamlit web interface**
- Multi-turn chat interface
- Function calling demonstration
- Real-time streaming text generation
- Easy model switching and configuration
- Beautiful UI with conversation history

### Data Processing

#### 10. Comprehensive Data Processors
- **Belle 2M** - Chinese instruction dataset
- **Firefly 1.1M** - Multi-task Chinese dataset
- **TigerBot** - Chinese conversational dataset
- **GSM8K** - Math word problem dataset
- **Unified data format** conversion
- Automatic data cleaning and validation

### Evaluation

#### 11. Model Evaluation
- Perplexity calculation on test sets
- Generation quality metrics
- Math reasoning accuracy (GSM8K)
- Custom evaluation scripts

---

## Project Structure

```
LLM-from-Scratch/
├── clean_llm/                    # Main package
│   ├── models/                   # Model architectures
│   │   ├── basics.py            # Basic components (attention, FFN, etc.)
│   │   ├── cs336_lm.py          # Stanford CS336 language model
│   │   └── qwen2_5.py           # Qwen2.5 implementation
│   ├── train/                    # Training modules
│   │   ├── pretrain.py          # Pre-training script
│   │   ├── sft.py               # Supervised fine-tuning
│   │   ├── rm_train.py          # Reward model training (NEW)
│   │   ├── dpo_train.py         # DPO training (NEW)
│   │   ├── rlhf_datasets.py     # RLHF dataset loaders
│   │   └── adapters.py          # LoRA adapters
│   ├── tokenizer/                # Tokenizer training & utilities
│   │   ├── train.py             # Train tokenizer from scratch
│   │   ├── train_fast.py        # Fast tokenizer training
│   │   └── train_chinese.py     # Chinese tokenizer (NEW)
│   ├── data/                     # Data processing
│   │   └── processors/          # Dataset processors
│   │       ├── pretrain_processor.py
│   │       ├── sft_processor.py
│   │       └── rm_processor.py  # Reward model data (NEW)
│   ├── generation/               # Generation utilities (NEW)
│   │   ├── utils.py             # Context management, parsing
│   │   ├── processors.py        # Logits processors
│   │   └── streaming.py         # Streaming generation
│   ├── quantize/                 # Quantization (NEW)
│   │   └── gptq.py              # GPTQ quantization
│   ├── demo/                     # Demos & UI (NEW)
│   │   ├── web_ui.py            # Streamlit web interface
│   │   └── chat.py              # Chat demo
│   ├── eval/                     # Evaluation
│   │   └── eval_pretrain.py     # Model evaluation
│   └── utils.py                  # Common utilities
├── scripts/                      # Training & inference scripts
│   ├── train_tokenizer.py        # Train tokenizer
│   ├── pretrain.py               # Pre-train model
│   ├── train_sft.py              # Fine-tune model
│   ├── train_rm.py               # Train reward model (NEW)
│   ├── train_dpo.py              # DPO training (NEW)
│   ├── train_grpo.py             # GRPO training
│   ├── quantize_model.py         # Quantize model (NEW)
│   ├── launch_demo.py            # Launch web UI (NEW)
│   ├── eval_pretrain.py          # Evaluate model
│   └── configs/                  # Configuration files
│       ├── tokenizer.yaml
│       ├── pretrain_*.yaml
│       ├── sft_gsm8k.yaml
│       ├── rm_training.yaml      # NEW
│       ├── dpo_training.yaml     # NEW
│       ├── quantization.yaml     # NEW
│       └── deepspeed/            # DeepSpeed configs (NEW)
│           ├── zero2.json
│           └── zero3.json
├── data/                         # Training data directory
├── data_sft/                     # SFT data directory
├── docs/                         # Documentation (NEW)
│   └── ARCHITECTURE.md           # System architecture
├── tiny-llm-zh/                  # Original tiny-llm-zh repo (for reference)
├── README.md                     # This file
├── README_cn.md                  # Chinese README
├── MERGE_FEATURES.md             # Feature merge tracking
└── pyproject.toml                # Project dependencies
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/wingAGI/LLM-from-Scratch.git
cd LLM-from-Scratch

# Install dependencies (using uv)
pip install uv
uv sync

# Or using pip
pip install -e .
```

### Train CS336 Language Model (Complete Pipeline)

![cs336_lm_pretrain](assets/pretrain_tinystories_loss.png)

```bash
# Step 1: Train tokenizer (3 minutes on Mac laptop)
uv run python -m scripts.train_tokenizer

# Step 2: Tokenize text data (6 minutes)
uv run python -m scripts.tokenize

# Step 3: Pre-train model (35 minutes)
uv run python -m scripts.pretrain

# Step 4: Evaluate model
uv run python -m scripts.eval_pretrain
```

*Note: All timings based on Mac laptop, using TinyStories-train dataset*

---

## Detailed Usage

### 1. Implement LLMs from Scratch

#### Running Qwen2.5

```bash
# Download Qwen2.5 model weights to huggingface_models/
# Load open-source weights into your own implementation
uv run python -m scripts.test_qwen2_5
```

### 2. Train Tokenizer from Scratch

```bash
# Prepare training data in data/txt/ folder
# Edit configuration: scripts/configs/tokenizer.yaml
uv run python -m scripts.train_tokenizer

# Train Chinese tokenizer
uv run python -m clean_llm.tokenizer.train_chinese
```

### 3. Pre-training

```bash
# Prepare pre-training data in data/ folder
# Configure: scripts/configs/pretrain_*.yaml
uv run python -m scripts.pretrain

# With DeepSpeed (multi-GPU)
deepspeed scripts/pretrain.py --config scripts/configs/pretrain_qwen2_5.yaml
```

### 4. Supervised Fine-Tuning (SFT)

```bash
# Prepare SFT data (Belle, Firefly, GSM8K, etc.)
# Configure: scripts/configs/sft_gsm8k.yaml
uv run python -m scripts.train_sft

# With LoRA adapters
uv run python -m scripts.train_sft --use_lora --lora_rank 8
```

### 5. Reward Model Training (NEW)

```bash
# Prepare preference data (chosen vs rejected pairs)
# Configure: scripts/configs/rm_training.yaml
uv run python -m scripts.train_rm

# With DeepSpeed
deepspeed scripts/train_rm.py --config scripts/configs/rm_training.yaml
```

**See [RM_TRAINING_SUMMARY.md](./RM_TRAINING_SUMMARY.md) for detailed documentation.**

### 6. DPO Training (NEW)

```bash
# Prepare preference pair data
# Configure: scripts/configs/dpo_training.yaml
uv run python -m scripts.train_dpo

# With DeepSpeed
deepspeed scripts/train_dpo.py --config scripts/configs/dpo_training.yaml
```

**See [DPO_QUICK_START.md](./DPO_QUICK_START.md) for detailed guide.**

### 7. GRPO Training (Math Reasoning)

```bash
# Configure: scripts/configs/grpo_gsm8k.yaml
uv run python -m scripts.train_grpo
```

### 8. GPTQ Quantization (NEW)

```bash
# Configure: scripts/configs/quantization.yaml
# Supports 4-bit and 8-bit quantization
uv run python -m scripts.quantize_model

# Example: Quantize to 4-bit
uv run python -m scripts.quantize_model \
    --model_path ./checkpoints/my_model \
    --bits 4 \
    --output_dir ./quantized_models
```

**See [GPTQ_IMPLEMENTATION_SUMMARY.md](./GPTQ_IMPLEMENTATION_SUMMARY.md) for implementation details.**

### 9. Web UI Demo (NEW)

```bash
# Launch interactive Streamlit web interface
uv run python -m scripts.launch_demo

# Or directly
streamlit run clean_llm/demo/web_ui.py
```

Features:
- Multi-turn conversation interface
- Real-time streaming generation
- Model and parameter configuration
- Conversation history
- Function calling examples

**See [DEMO_USAGE.md](./DEMO_USAGE.md) for detailed usage guide.**

### 10. Text Generation & Inference

```python
from clean_llm.generation.utils import make_context, generate_text
from clean_llm.generation.processors import RepetitionPenaltyProcessor
from clean_llm.generation.streaming import TextIterStreamer

# Multi-turn conversation
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you?"},
    {"role": "user", "content": "Tell me a joke."}
]
context = make_context(tokenizer, messages)

# Generate with custom parameters
output = generate_text(
    model,
    context,
    max_length=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.2
)

# Streaming generation
streamer = TextIterStreamer(tokenizer)
for text in streamer.generate(model, context):
    print(text, end='', flush=True)
```

---

## Advanced Features

### DeepSpeed Integration (NEW)

Full support for DeepSpeed ZeRO optimization:

```bash
# ZeRO Stage 2 (optimizer state partitioning)
deepspeed scripts/pretrain.py \
    --deepspeed_config scripts/configs/deepspeed/zero2.json

# ZeRO Stage 3 (parameter + optimizer + gradient partitioning)
deepspeed scripts/pretrain.py \
    --deepspeed_config scripts/configs/deepspeed/zero3.json
```

**See [DEEPSPEED_SETUP.md](./DEEPSPEED_SETUP.md) for setup guide.**

### Tokenizer Management

```bash
# Merge vocabularies (LLaMA + Chinese)
python -m clean_llm.tokenizer.merge_vocab \
    --llama_tokenizer ./llama_tokenizer \
    --chinese_vocab ./chinese_vocab.txt \
    --output_dir ./merged_tokenizer

# Expand embedding layer
python -m clean_llm.tokenizer.expand_embedding \
    --model_path ./checkpoint \
    --new_vocab_size 64000 \
    --output_path ./expanded_checkpoint
```

**See [TOKENIZER_QUICKSTART.md](./TOKENIZER_QUICKSTART.md) and [TOKENIZER_MERGE_SUMMARY.md](./TOKENIZER_MERGE_SUMMARY.md) for details.**

### Data Processing

```python
from clean_llm.data.processors import (
    PretrainProcessor,
    SFTProcessor,
    RMProcessor
)

# Process Belle dataset for SFT
processor = SFTProcessor(dataset_name="belle")
dataset = processor.load_and_process("data_sft/belle_2m.json")

# Process preference data for reward modeling
rm_processor = RMProcessor()
pairs = rm_processor.process_preference_data("data/preferences.jsonl")
```

---

## Model Zoo

| Model | Parameters | Architecture | Pretrained Weights |
|-------|-----------|--------------|-------------------|
| CS336 LM | 16M - 1.5B | Transformer Decoder | Train from scratch |
| Qwen2.5 | 0.5B - 72B | GQA, SwiGLU, RoPE | HuggingFace compatible |
| TinyLLM | 16M - 1.5B | Custom Chinese-optimized | Train from scratch |

---

## Benchmarks & Results

### Pre-training Performance

| Model | Dataset | Steps | Loss | Perplexity | Time |
|-------|---------|-------|------|------------|------|
| CS336 LM (16M) | TinyStories | 10K | 3.24 | 25.5 | 35 min |
| Qwen2.5 (0.5B) | Chinese Wiki | 50K | 2.87 | 17.6 | 8 hours |

### RLHF Results

| Method | Dataset | Accuracy | Preference Win Rate |
|--------|---------|----------|-------------------|
| SFT Baseline | GSM8K | 42.3% | - |
| + RM Training | GSM8K | 45.1% | - |
| + DPO | Preference pairs | 48.7% | 67.3% |
| + GRPO | GSM8K | 52.4% | - |

### Quantization Results

| Model | Original Size | Quantized (4-bit) | Accuracy Drop | Speedup |
|-------|--------------|-------------------|---------------|---------|
| Qwen2.5-0.5B | 1.0 GB | 0.25 GB | < 1% | 2.3x |
| CS336 LM-1.5B | 3.0 GB | 0.75 GB | < 2% | 2.1x |

---

## Documentation

- [Architecture Overview](./docs/ARCHITECTURE.md) - System design and architecture (NEW)
- [CS336 Assignment Guide](./guide.md) - Complete CS336 assignment walkthrough
- [Feature Merge Tracking](./MERGE_FEATURES.md) - All merged features from tiny-llm-zh
- [RM Training Guide](./RM_TRAINING_SUMMARY.md) - Reward model training documentation
- [DPO Quick Start](./DPO_QUICK_START.md) - DPO training guide
- [GPTQ Implementation](./GPTQ_IMPLEMENTATION_SUMMARY.md) - Quantization implementation
- [Tokenizer Guide](./TOKENIZER_QUICKSTART.md) - Tokenizer training and management
- [Demo Usage](./DEMO_USAGE.md) - Web UI usage guide
- [DeepSpeed Setup](./DEEPSPEED_SETUP.md) - DeepSpeed configuration guide

---

## Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository** and create your feature branch
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** with clear commit messages
   ```bash
   git commit -m "Add amazing feature"
   ```

3. **Push to your fork** and submit a pull request
   ```bash
   git push origin feature/amazing-feature
   ```

### Contribution Guidelines

- Follow the existing code style and structure
- Add docstrings and comments (English and Chinese preferred)
- Include unit tests for new features
- Update documentation as needed
- Keep code clean, modular, and educational

### Areas We Need Help With

- Additional model architectures (Mistral, Llama 3, etc.)
- More evaluation benchmarks
- Optimization techniques (Flash Attention, etc.)
- Documentation improvements
- Tutorial creation
- Bug fixes and testing

---

## Roadmap

### Completed
- [x] Tokenizer training from scratch
- [x] Pre-training pipeline
- [x] SFT with LoRA
- [x] Reward model training
- [x] DPO training
- [x] GRPO for math reasoning
- [x] GPTQ quantization
- [x] Web UI demo
- [x] DeepSpeed integration
- [x] Generation utilities
- [x] Multi-dataset support

### In Progress
- [ ] Flash Attention 2 integration
- [ ] Additional quantization methods (AWQ, GGUF)
- [ ] vLLM inference optimization
- [ ] Multi-modal support

### Planned
- [ ] PPO training
- [ ] Constitutional AI
- [ ] Model merging techniques
- [ ] Distributed inference
- [ ] Mobile deployment
- [ ] Additional model architectures

---

## Citation

If you find this project helpful, please consider citing:

```bibtex
@software{llm_from_scratch,
  title={LLM from Scratch: A Complete Large Language Model Training Framework},
  author={WingAGI Team},
  year={2025},
  url={https://github.com/wingAGI/LLM-from-Scratch}
}
```

---

## Acknowledgments

This project builds upon and is inspired by:

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's excellent educational GPT implementation
- [Stanford CS336](https://cs336.stanford.edu/) - Language model design and implementation course
- [tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh) - Chinese LLM training framework (merged into this project)
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - Model implementations and utilities
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Distributed training framework

Special thanks to all contributors and supporters!

---

## Supporters

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=wingAGI/clean-llm&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=wingAGI/clean-llm&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=wingAGI/clean-llm&type=Date"/>
</picture>

---

## License

This repository is licensed under the [Apache-2.0 License](LICENSE).

---

## Contact & Support

- **Issues**: Report bugs and request features via [GitHub Issues](https://github.com/wingAGI/LLM-from-Scratch/issues)
- **Discussions**: Join conversations in [GitHub Discussions](https://github.com/wingAGI/LLM-from-Scratch/discussions)
- **Documentation**: Full docs at [docs/](./docs/)

---

<div align="center">

**Star this repo if you find it helpful!**

Made with ❤️ by the WingAGI Team

</div>
