# System Architecture - LLM from Scratch

## Table of Contents

1. [Overview](#overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Diagram](#component-diagram)
4. [Data Flow](#data-flow)
5. [Module Descriptions](#module-descriptions)
6. [Training Pipeline](#training-pipeline)
7. [Design Decisions](#design-decisions)
8. [Technology Stack](#technology-stack)

---

## Overview

LLM from Scratch is a comprehensive framework for training large language models from the ground up. The architecture is designed with the following principles:

- **Modularity**: Each component is self-contained and can be used independently
- **Extensibility**: Easy to add new models, datasets, and training methods
- **Clarity**: Code is educational and easy to understand
- **Production-Ready**: Includes advanced features for real-world deployment

The system supports the complete LLM lifecycle:
1. Data preparation and tokenization
2. Model pre-training
3. Supervised fine-tuning (SFT)
4. Reinforcement learning from human feedback (RLHF)
5. Model quantization and compression
6. Deployment and inference

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LLM from Scratch                             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Data       │  │   Model      │  │   Training   │              │
│  │  Processing  │─▶│Architecture  │◀─│   Pipeline   │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                  │                      │
│         ▼                 ▼                  ▼                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Tokenizer   │  │ Generation   │  │ Evaluation   │              │
│  │   Training   │  │   Utils      │  │   Metrics    │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│         │                 │                  │                      │
│         └─────────────────┴──────────────────┘                      │
│                           │                                         │
│                           ▼                                         │
│  ┌────────────────────────────────────────────────────┐             │
│  │         Deployment & Inference Layer               │             │
│  │  (Quantization, Web UI, API Serving)              │             │
│  └────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Diagram

### 1. Core Modules

```
clean_llm/
│
├── models/                      # Model Architectures
│   ├── basics.py               # Core components (attention, FFN, embeddings)
│   ├── cs336_lm.py            # CS336 Language Model implementation
│   └── qwen2_5.py             # Qwen2.5 model with GQA
│
├── tokenizer/                   # Tokenization Layer
│   ├── train.py               # SentencePiece tokenizer training
│   ├── train_fast.py          # Fast BPE implementation
│   └── train_chinese.py       # Chinese-optimized tokenizer
│
├── data/                        # Data Processing Pipeline
│   └── processors/
│       ├── pretrain_processor.py    # Pre-training data
│       ├── sft_processor.py         # SFT data (Belle, Firefly, etc.)
│       └── rm_processor.py          # Preference data for RM/DPO
│
├── train/                       # Training Orchestration
│   ├── pretrain.py            # Pre-training loop
│   ├── sft.py                 # Supervised fine-tuning
│   ├── rm_train.py            # Reward model training
│   ├── dpo_train.py           # Direct Preference Optimization
│   ├── rlhf_datasets.py       # RLHF data loaders
│   └── adapters.py            # LoRA/PEFT adapters
│
├── generation/                  # Text Generation
│   ├── utils.py               # Context management, parsing
│   ├── processors.py          # Logits processors (repetition penalty, etc.)
│   └── streaming.py           # Streaming generation
│
├── quantize/                    # Model Compression
│   └── gptq.py                # GPTQ quantization (4-bit, 8-bit)
│
├── demo/                        # User Interfaces
│   ├── web_ui.py              # Streamlit web interface
│   └── chat.py                # Command-line chat
│
└── eval/                        # Evaluation Framework
    └── eval_pretrain.py       # Perplexity and metrics
```

### 2. Scripts Layer

```
scripts/
│
├── train_tokenizer.py          # Entry point: Tokenizer training
├── pretrain.py                 # Entry point: Pre-training
├── train_sft.py                # Entry point: SFT
├── train_rm.py                 # Entry point: Reward model
├── train_dpo.py                # Entry point: DPO
├── train_grpo.py               # Entry point: GRPO
├── quantize_model.py           # Entry point: Quantization
├── launch_demo.py              # Entry point: Web UI
├── eval_pretrain.py            # Entry point: Evaluation
│
└── configs/                     # Configuration Management
    ├── tokenizer.yaml
    ├── pretrain_*.yaml
    ├── sft_gsm8k.yaml
    ├── rm_training.yaml
    ├── dpo_training.yaml
    ├── quantization.yaml
    └── deepspeed/
        ├── zero2.json
        └── zero3.json
```

---

## Data Flow

### Pre-Training Pipeline

```
┌─────────────┐
│ Raw Text    │
│ Data (TXT)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Tokenizer Training  │
│ (SentencePiece/BPE) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Text Tokenization   │
│ (Encode to IDs)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Data Loader         │
│ (Memory-mapped)     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Pre-training Loop   │
│ - Forward pass      │
│ - Loss calculation  │
│ - Backward pass     │
│ - Optimizer step    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Model Checkpoint    │
│ (Saved to disk)     │
└─────────────────────┘
```

### RLHF Pipeline

```
┌─────────────┐
│ SFT Model   │
│ (Baseline)  │
└──────┬──────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
┌──────────────┐      ┌──────────────┐
│ Method 1:    │      │ Method 2:    │
│ RM Training  │      │ DPO Training │
│              │      │ (Direct)     │
│ ┌──────────┐ │      │              │
│ │Preference│ │      │ ┌──────────┐ │
│ │   Data   │ │      │ │Preference│ │
│ └────┬─────┘ │      │ │   Pairs  │ │
│      │       │      │ └────┬─────┘ │
│      ▼       │      │      │       │
│ ┌──────────┐ │      │      ▼       │
│ │  Reward  │ │      │ ┌──────────┐ │
│ │  Model   │ │      │ │   DPO    │ │
│ └────┬─────┘ │      │ │  Loss    │ │
│      │       │      │ └────┬─────┘ │
│      ▼       │      │      │       │
│ ┌──────────┐ │      │      ▼       │
│ │   PPO/   │ │      │ ┌──────────┐ │
│ │   GRPO   │ │      │ │ Aligned  │ │
│ └────┬─────┘ │      │ │  Model   │ │
└──────┼──────┘      └──────┼──────┘
       │                    │
       └────────┬───────────┘
                ▼
        ┌──────────────┐
        │ Final Aligned│
        │    Model     │
        └──────────────┘
```

### Inference Pipeline

```
┌─────────────┐
│  User Input │
│   (Text)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│   Tokenization      │
│   (Text → IDs)      │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Context Building   │
│  (Multi-turn chat)  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Model Forward      │
│  (Get logits)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Logits Processing   │
│ - Temperature       │
│ - Top-k/Top-p       │
│ - Repetition penalty│
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│    Sampling         │
│ (Select next token) │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Detokenization     │
│  (IDs → Text)       │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  Output Display     │
│  (Streaming/Batch)  │
└─────────────────────┘
```

---

## Module Descriptions

### 1. Model Architectures (`clean_llm/models/`)

**Purpose**: Define neural network architectures for language models.

**Key Components**:

- **basics.py**:
  - `RMSNorm`: Root Mean Square Layer Normalization
  - `RotaryPositionalEmbedding`: RoPE for position encoding
  - `MultiHeadAttention`: Standard attention mechanism
  - `GroupedQueryAttention`: Memory-efficient attention (GQA)
  - `FeedForward`: MLP layers (standard and SwiGLU variants)

- **cs336_lm.py**:
  - `CS336LanguageModel`: Stanford CS336 transformer decoder
  - Configurable depth, width, attention heads
  - Supports different normalization strategies

- **qwen2_5.py**:
  - `Qwen2_5ForCausalLM`: Modern LLM with GQA
  - Compatible with HuggingFace weights
  - SwiGLU activation, RoPE embeddings

**Design Pattern**: Factory pattern for model instantiation, configuration-driven architecture.

---

### 2. Tokenizer (`clean_llm/tokenizer/`)

**Purpose**: Convert text to token IDs and vice versa.

**Key Components**:

- **train.py**:
  - SentencePiece BPE/Unigram training
  - Configurable vocabulary size
  - Special token handling

- **train_fast.py**:
  - Fast BPE implementation
  - Optimized for large corpora

- **train_chinese.py**:
  - Chinese-optimized tokenizer
  - Trained on Chinese Wikipedia
  - Better segmentation for CJK characters

**Design Pattern**: Strategy pattern for different tokenization algorithms.

---

### 3. Data Processing (`clean_llm/data/processors/`)

**Purpose**: Load, clean, and prepare datasets for training.

**Key Components**:

- **pretrain_processor.py**:
  - Raw text processing
  - Document chunking
  - Memory-mapped dataset creation

- **sft_processor.py**:
  - Instruction-response pair formatting
  - Multi-dataset support (Belle, Firefly, TigerBot, GSM8K)
  - Chat template application

- **rm_processor.py**:
  - Preference pair loading
  - Chosen vs. rejected formatting
  - Reward signal computation

**Design Pattern**: Template method pattern with dataset-specific implementations.

---

### 4. Training (`clean_llm/train/`)

**Purpose**: Orchestrate the training loop for different paradigms.

**Key Components**:

- **pretrain.py**:
  - Next-token prediction training
  - Gradient accumulation
  - DeepSpeed ZeRO integration
  - MLflow experiment tracking

- **sft.py**:
  - Instruction-following fine-tuning
  - LoRA adapter support
  - Multi-dataset batching

- **rm_train.py**:
  - Pairwise preference learning
  - Margin-based loss function
  - Reward accuracy metrics

- **dpo_train.py**:
  - Direct Preference Optimization
  - Reference model frozen
  - Beta parameter for KL control

- **adapters.py**:
  - LoRA (Low-Rank Adaptation)
  - Parameter-efficient fine-tuning
  - Adapter merging

**Design Pattern**: Strategy pattern for different training objectives, observer pattern for logging.

---

### 5. Generation (`clean_llm/generation/`)

**Purpose**: Generate text from trained models.

**Key Components**:

- **utils.py**:
  - `make_context()`: Multi-turn conversation context
  - `parse_pot_no_stream()`: Math reasoning parsing
  - Beam search implementation

- **processors.py**:
  - `RepetitionPenaltyProcessor`: Prevent repetition
  - `TemperatureProcessor`: Control randomness
  - `TopKTopPProcessor`: Nucleus sampling

- **streaming.py**:
  - `TextIterStreamer`: Real-time token streaming
  - Async generation support
  - WebSocket compatibility

**Design Pattern**: Chain of responsibility for logits processing, iterator pattern for streaming.

---

### 6. Quantization (`clean_llm/quantize/`)

**Purpose**: Compress models for efficient deployment.

**Key Components**:

- **gptq.py**:
  - GPTQ algorithm implementation
  - 4-bit and 8-bit quantization
  - Layer-wise calibration
  - AutoGPTQ integration

**Design Pattern**: Decorator pattern for model wrapping.

---

### 7. Demo & UI (`clean_llm/demo/`)

**Purpose**: User-facing interfaces for model interaction.

**Key Components**:

- **web_ui.py**:
  - Streamlit web interface
  - Multi-turn chat UI
  - Parameter tuning controls
  - Conversation history

- **chat.py**:
  - Command-line interface
  - Simple query-response loop

**Design Pattern**: MVC (Model-View-Controller) pattern.

---

### 8. Evaluation (`clean_llm/eval/`)

**Purpose**: Assess model performance.

**Key Components**:

- **eval_pretrain.py**:
  - Perplexity calculation
  - Loss on validation set
  - Generation quality metrics

**Design Pattern**: Strategy pattern for different evaluation metrics.

---

## Training Pipeline

### Stage 1: Pre-training

```python
# Pseudocode for pre-training

1. Load tokenizer
2. Prepare data:
   - Tokenize raw text
   - Create memory-mapped dataset
3. Initialize model from config
4. Setup optimizer (AdamW) and scheduler
5. Training loop:
   for epoch in epochs:
       for batch in dataloader:
           # Forward pass
           logits = model(batch.input_ids)
           loss = cross_entropy(logits, batch.labels)

           # Backward pass
           loss.backward()

           # Gradient accumulation
           if (step + 1) % grad_accum_steps == 0:
               optimizer.step()
               optimizer.zero_grad()

           # Logging
           log_metrics(loss, lr, step)

           # Checkpointing
           if step % checkpoint_interval == 0:
               save_checkpoint(model, optimizer, step)
```

### Stage 2: Supervised Fine-Tuning (SFT)

```python
# Pseudocode for SFT

1. Load pre-trained model checkpoint
2. Prepare instruction dataset:
   - Format: {"instruction": "...", "response": "..."}
   - Apply chat template
3. Optional: Add LoRA adapters
4. Training loop (similar to pre-training):
   - Mask instruction tokens in loss (only learn from response)
   - Lower learning rate than pre-training
   - Fewer epochs (typically 1-3)
```

### Stage 3: RLHF (Reward Model + DPO)

```python
# Option 1: Reward Model + PPO/GRPO

1. Train Reward Model:
   - Input: (chosen_text, rejected_text) pairs
   - Output: scalar reward for each
   - Loss: margin_loss(reward_chosen - reward_rejected)

2. RL Training (GRPO):
   - Generate completions from policy model
   - Score with reward model
   - Update policy to maximize reward
   - KL penalty to prevent drift from SFT model

# Option 2: Direct Preference Optimization (DPO)

1. Load SFT model as reference (frozen)
2. Initialize policy model from SFT
3. Training loop:
   for (chosen, rejected) in preference_data:
       # Compute log probabilities
       log_pi_chosen = policy_model.log_prob(chosen)
       log_pi_rejected = policy_model.log_prob(rejected)
       log_ref_chosen = ref_model.log_prob(chosen)
       log_ref_rejected = ref_model.log_prob(rejected)

       # DPO loss
       loss = -log_sigmoid(
           beta * (log_pi_chosen - log_ref_chosen) -
           beta * (log_pi_rejected - log_ref_rejected)
       )

       loss.backward()
       optimizer.step()
```

### Stage 4: Quantization

```python
# GPTQ Quantization

1. Load full-precision model
2. Prepare calibration dataset (small subset)
3. For each layer:
   - Collect activation statistics
   - Compute optimal quantization parameters
   - Quantize weights to 4-bit or 8-bit
4. Save quantized model
```

---

## Design Decisions

### 1. Configuration-Driven Architecture

**Decision**: Use YAML files for all hyperparameters and settings.

**Rationale**:
- Easier to experiment with different configurations
- Version control for experiments
- Reproducibility
- No code changes needed for hyperparameter tuning

**Example**:
```yaml
# pretrain_config.yaml
model:
  name: cs336_lm
  n_layers: 12
  n_heads: 8
  d_model: 512

training:
  batch_size: 32
  learning_rate: 3e-4
  max_steps: 100000
```

### 2. Modular Component Design

**Decision**: Separate concerns into distinct modules (models, data, training, etc.).

**Rationale**:
- Easy to understand and maintain
- Can swap out components (e.g., different model architectures)
- Facilitates unit testing
- Enables code reuse

### 3. Memory-Mapped Datasets

**Decision**: Use memory mapping for large datasets instead of loading into RAM.

**Rationale**:
- Supports datasets larger than available RAM
- Fast random access
- Shared memory across processes in distributed training
- OS-level caching

### 4. DeepSpeed Integration

**Decision**: Support DeepSpeed ZeRO for distributed training.

**Rationale**:
- Train models that don't fit on a single GPU
- Efficient memory usage with ZeRO optimizer
- Automatic mixed precision
- Industry-standard tool

### 5. LoRA for Fine-Tuning

**Decision**: Implement LoRA adapters for parameter-efficient fine-tuning.

**Rationale**:
- Dramatically reduces memory requirements
- Faster training with fewer parameters to update
- Can maintain multiple task-specific adapters
- Proven effective in research and practice

### 6. Multiple RLHF Methods

**Decision**: Support both RM+PPO and DPO for alignment.

**Rationale**:
- DPO is simpler and more stable (no RL training loop)
- Traditional RM+PPO provides more control and interpretability
- Different use cases benefit from different methods
- Educational value in understanding both approaches

### 7. Streaming Generation

**Decision**: Implement streaming text generation for UI.

**Rationale**:
- Better user experience (see output as it generates)
- Lower perceived latency
- Essential for production chatbots
- Compatible with modern web frameworks

### 8. Bilingual Documentation

**Decision**: Provide full documentation in both English and Chinese.

**Rationale**:
- Accessibility to wider audience
- Chinese LLM community is very active
- Educational project should be inclusive
- Merged features from Chinese codebase (tiny-llm-zh)

---

## Technology Stack

### Core Framework
- **PyTorch 2.0+**: Deep learning framework
- **Python 3.10+**: Programming language

### Training Infrastructure
- **DeepSpeed**: Distributed training and optimization
- **MLflow**: Experiment tracking and model registry
- **Weights & Biases** (optional): Advanced experiment tracking

### Tokenization
- **SentencePiece**: Subword tokenization
- **tiktoken**: Fast BPE tokenizer
- **HuggingFace Tokenizers**: Fast tokenizer library

### Data Processing
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **datasets** (HuggingFace): Dataset loading utilities

### Quantization
- **AutoGPTQ**: GPTQ quantization
- **bitsandbytes** (optional): 8-bit optimizers

### Deployment & UI
- **Streamlit**: Web UI framework
- **FastAPI** (optional): REST API serving
- **vLLM** (planned): High-performance inference

### Development Tools
- **uv**: Fast Python package manager
- **pytest**: Testing framework
- **black**: Code formatting
- **mypy**: Type checking

---

## Performance Considerations

### Memory Optimization

1. **Gradient Accumulation**: Simulate larger batch sizes
2. **Mixed Precision (FP16/BF16)**: Reduce memory footprint
3. **Gradient Checkpointing**: Trade compute for memory
4. **DeepSpeed ZeRO**: Partition optimizer states and parameters

### Training Speed

1. **Data Loading**: Memory-mapped datasets with prefetching
2. **Compiled Models**: PyTorch 2.0 compilation
3. **Flash Attention** (planned): Faster attention computation
4. **Distributed Training**: Multi-GPU and multi-node

### Inference Optimization

1. **Quantization**: 4-bit/8-bit weights
2. **KV-Cache**: Cache attention keys/values
3. **Batching**: Process multiple requests together
4. **vLLM** (planned): PagedAttention for efficiency

---

## Security Considerations

1. **Model Checkpoint Validation**: Verify checkpoints before loading
2. **Input Sanitization**: Clean user inputs in web UI
3. **Resource Limits**: Prevent excessive memory/compute usage
4. **Access Control** (for deployment): Authentication and authorization

---

## Testing Strategy

### Unit Tests
- Individual component testing (models, tokenizers, data loaders)
- Mock external dependencies

### Integration Tests
- End-to-end training pipeline
- Data processing → Training → Evaluation

### Performance Tests
- Benchmarking training speed
- Memory usage profiling
- Inference latency measurement

---

## Future Enhancements

1. **Flash Attention 2**: Faster and more memory-efficient attention
2. **Multi-Modal Support**: Vision-language models
3. **Distributed Inference**: Tensor parallelism for large models
4. **Additional Quantization**: AWQ, GGUF formats
5. **Model Merging**: Combine multiple fine-tuned models
6. **Constitutional AI**: Advanced alignment techniques

---

## References

- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal GPT implementation
- [Stanford CS336](https://cs336.stanford.edu/) - Language model course
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer paper
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [DPO](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization
- [GPTQ](https://arxiv.org/abs/2210.17323) - Quantization paper
- [DeepSpeed](https://www.deepspeed.ai/) - Distributed training framework

---

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Maintained By**: WingAGI Team
