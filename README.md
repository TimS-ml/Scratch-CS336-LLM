# LLM from Scratch
This is a comprehensive LLM learning and training project inspired by:
- [nanoGPT](https://github.com/karpathy/nanoGPT)
- [nanochat](https://github.com/karpathy/nanochat/tree/master)
- [Stanford CS336](https://github.com/stanford-cs336)
  - [assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
- [clean-llm](https://github.com/wingAGI/clean-llm)
- [tiny-llm-zh](https://github.com/wdndev/tiny-llm-zh/tree/main)

It implements the **entire LLM training pipeline from scratch**, including tokenizer training, data processing, model pre-training, supervised fine-tuning (SFT), reward modeling (RM), reinforcement learning from human feedback (RLHF), quantization, and deployment.

## What Makes This Project Special?
- **Complete End-to-End Pipeline**: From tokenizer training to production deployment
- **Clean & Educational**: Code designed for learning with extensive documentation
- **Production-Ready**: Includes advanced features like DeepSpeed, quantization, and web UI
- **Multiple Training Paradigms**: Pre-training, SFT, RLHF (RM + DPO), GRPO
- **Modern Architecture**: Support for Qwen2.5, CS336 LM, and custom models

## Feature List
### Core Training Pipeline
- **Pre-training**: Train LLMs from scratch with custom architectures, memory-efficient data loading, mixed precision training, gradient accumulation, and DeepSpeed ZeRO-2/ZeRO-3 support
- **Supervised Fine-Tuning (SFT)**: Multi-dataset support (Belle, Firefly, TigerBot, GSM8K), instruction tuning with chat templates, LoRA adapters for parameter-efficient fine-tuning
- **Reward Model Training**: Train reward models for RLHF from preference data with pairwise comparison datasets
- **DPO Training**: Direct Preference Optimization - RLHF without reward models, more stable than PPO
- **GRPO Training**: Group Relative Policy Optimization for math reasoning with GSM8K dataset

### Tokenization
- **Train custom tokenizers from scratch** using SentencePiece (BPE and Unigram algorithms)
- **Chinese tokenizer training** on Chinese Wikipedia
- **Vocabulary merging** - combine LLaMA + Chinese vocab (32K → 64K)
- **Embedding layer expansion** with weight preservation
- Custom special token handling

### Quantization & Compression
- **GPTQ Quantization**: 4-bit and 8-bit quantization with GPTQ algorithm
- Reduce model size by 4-8x with minimal accuracy loss
- AutoGPTQ library integration with calibration dataset support

### Inference & Deployment
- **Advanced text generation** with temperature sampling, top-k/top-p sampling, beam search, and repetition penalty
- **Multi-turn conversation** context management
- **Streaming generation** for real-time output
- **Interactive Streamlit web interface** with multi-turn chat and function calling

### Data Processing
- Comprehensive data processors for Belle 2M, Firefly 1.1M, TigerBot, GSM8K
- Unified data format conversion
- Automatic data cleaning and validation

### Evaluation
- Perplexity calculation on test sets
- Generation quality metrics
- Math reasoning accuracy (GSM8K)

## Quick Start
### Installation
```bash
# Clone the repository
git clone https://github.com/TimS-ml/LLM-from-Scratch.git
cd LLM-from-Scratch

# Install dependencies (using uv)
pip install uv
uv sync

# Or using pip
pip install -e .
```

### Train CS336 Language Model (Complete Pipeline)
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

*All timings based on Mac laptop, using TinyStories-train dataset*

## Basic Usage
### Pre-training
```bash
# Configure: scripts/configs/pretrain_*.yaml
uv run python -m scripts.pretrain

# With DeepSpeed (multi-GPU)
deepspeed scripts/pretrain.py --config scripts/configs/pretrain_qwen2_5.yaml
```

### Supervised Fine-Tuning
```bash
# Configure: scripts/configs/sft_gsm8k.yaml
uv run python -m scripts.train_sft

# With LoRA adapters
uv run python -m scripts.train_sft --use_lora --lora_rank 8
```

### Reward Model Training
```bash
# Configure: scripts/configs/rm_training.yaml
uv run python -m scripts.train_rm

# With DeepSpeed
deepspeed scripts/train_rm.py --config scripts/configs/rm_training.yaml
```

### DPO Training
```bash
# Configure: scripts/configs/dpo_training.yaml
uv run python -m scripts.train_dpo

# With DeepSpeed
deepspeed scripts/train_dpo.py --config scripts/configs/dpo_training.yaml
```

### GPTQ Quantization
```bash
# Configure: scripts/configs/quantization.yaml
uv run python -m scripts.quantize_model

# Example: Quantize to 4-bit
uv run python -m scripts.quantize_model \
    --model_path ./checkpoints/my_model \
    --bits 4 \
    --output_dir ./quantized_models
```

### Web UI Demo
```bash
# Launch interactive Streamlit web interface
uv run python -m scripts.launch_demo

# Or directly
streamlit run clean_llm/demo/web_ui.py
```

### Text Generation
```python
from clean_llm.generation.utils import make_context, generate_text

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
```

## Project Structure
```
LLM-from-Scratch/
├── clean_llm/                    # Main package
│   ├── models/                   # Model architectures (CS336 LM, Qwen2.5)
│   ├── train/                    # Training modules (pretrain, SFT, RM, DPO)
│   ├── tokenizer/                # Tokenizer training & utilities
│   ├── data/                     # Data processing
│   ├── generation/               # Generation utilities
│   ├── quantize/                 # GPTQ quantization
│   ├── demo/                     # Web UI and demos
│   ├── eval/                     # Evaluation
│   └── utils.py                  # Common utilities
├── scripts/                      # Training & inference scripts
│   └── configs/                  # Configuration files
├── data/                         # Training data
├── data_sft/                     # SFT data
└── docs/                         # Documentation
```

