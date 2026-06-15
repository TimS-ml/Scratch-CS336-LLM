# LLM Demo Usage Guide / LLM演示使用指南

This guide explains how to use the newly created demo interfaces for the LLM project.
本指南介绍如何使用LLM项目新创建的演示界面。

## 📁 Created Files / 创建的文件

### Demo Package / 演示包
- `/home/user/LLM-from-Scratch/scratch_cs336/serve/__init__.py` - Package initialization / 包初始化
- `/home/user/LLM-from-Scratch/scratch_cs336/serve/web_ui.py` - Streamlit web interface / Streamlit Web界面
- `/home/user/LLM-from-Scratch/scratch_cs336/serve/chat.py` - CLI chat interface / CLI聊天界面

### Scripts / 脚本
- `/home/user/LLM-from-Scratch/scripts/launch_demo.py` - Demo launcher script / 演示启动脚本

## 🚀 Quick Start / 快速开始

### Prerequisites / 先决条件

```bash
# Install required packages / 安装必需的包
pip install torch transformers

# For web UI, also install Streamlit / 对于Web UI，还需安装Streamlit
pip install streamlit
```

## 💻 Usage Examples / 使用示例

### 1. Web UI (Streamlit) / Web界面 (Streamlit)

#### Using launch script / 使用启动脚本
```bash
# Launch with default model / 使用默认模型启动
python scripts/launch_demo.py web

# Launch with custom model / 使用自定义模型启动
python scripts/launch_demo.py web --model_path outputs/ckpt/tiny_llm_sft_92m

# Launch on custom port / 在自定义端口上启动
python scripts/launch_demo.py web --port 8080 --model_path wdndev/tiny_llm_sft_92m
```

#### Direct usage / 直接使用
```bash
streamlit run scratch_cs336/serve/web_ui.py -- --model_path wdndev/tiny_llm_sft_92m
```

### 2. CLI Chat / CLI聊天

#### Using launch script / 使用启动脚本
```bash
# Launch with default settings / 使用默认设置启动
python scripts/launch_demo.py cli

# Launch with custom model / 使用自定义模型启动
python scripts/launch_demo.py cli --model_path outputs/ckpt/tiny_llm_sft_92m

# Launch with custom parameters / 使用自定义参数启动
python scripts/launch_demo.py cli \
    --model_path wdndev/tiny_llm_sft_92m \
    --max_new_tokens 256 \
    --temperature 0.5 \
    --top_p 0.9
```

#### Direct usage / 直接使用
```bash
python -m scratch_cs336.serve.chat --model_path wdndev/tiny_llm_sft_92m

# Or run directly / 或直接运行
python scratch_cs336/serve/chat.py --model_path wdndev/tiny_llm_sft_92m
```

## ⚙️ Configuration Options / 配置选项

### Web UI Parameters / Web UI参数

All parameters can be configured through the sidebar in the web interface:
所有参数可以通过Web界面的侧边栏配置：

- **System Prompt** / 系统提示: Guide model behavior / 引导模型行为
- **Max New Tokens** / 最大新token数: Maximum tokens to generate (1-2048) / 生成的最大token数
- **Temperature** / 温度: Controls randomness (0.0-2.0) / 控制随机性
- **Top P** / 核采样: Nucleus sampling threshold (0.0-1.0) / 核采样阈值
- **Top K**: Top-k sampling (0-100, 0=disabled) / Top-k采样（0=禁用）
- **Enable Sampling** / 启用采样: Use sampling vs greedy decoding / 使用采样而非贪婪解码
- **Enable Streaming** / 启用流式输出: Stream tokens in real-time / 实时流式输出token

### CLI Chat Parameters / CLI聊天参数

```bash
python scripts/launch_demo.py cli --help

Options:
  --model_path PATH          Path to model or HuggingFace model ID
  --system_prompt TEXT       System prompt to guide model behavior
  --max_new_tokens INT       Maximum tokens to generate (default: 512)
  --temperature FLOAT        Sampling temperature (default: 0.7)
  --top_p FLOAT             Nucleus sampling (default: 0.8)
  --top_k INT               Top-k sampling (default: 0)
  --no_sample               Disable sampling (use greedy decoding)
  --no_streaming            Disable streaming output
```

### CLI Commands (in chat) / CLI命令（在聊天中）

Once in the chat interface, you can use:
进入聊天界面后，可以使用：

- `/clear` - Clear conversation history / 清除对话历史
- `/exit` or `/quit` - Exit chat / 退出聊天
- `Ctrl+C` - Interrupt and exit / 中断并退出

## 🎨 Features / 特性

### Web UI Features / Web UI特性
- ✅ Multi-turn conversation support / 多轮对话支持
- ✅ Real-time streaming output / 实时流式输出
- ✅ Configurable generation parameters / 可配置的生成参数
- ✅ Clean, responsive UI / 简洁的响应式UI
- ✅ Bilingual interface (EN/ZH) / 双语界面（英文/中文）
- ✅ Conversation history management / 对话历史管理
- ✅ Model caching for fast reloading / 模型缓存以快速重新加载

### CLI Chat Features / CLI聊天特性
- ✅ Interactive command-line interface / 交互式命令行界面
- ✅ Conversation history / 对话历史
- ✅ Streaming output support / 流式输出支持
- ✅ Colored terminal output / 彩色终端输出
- ✅ Simple commands for history management / 简单的历史管理命令
- ✅ Comprehensive parameter configuration / 全面的参数配置

## 🏗️ Architecture / 架构

Both interfaces use the generation utilities from `scratch_cs336.core.generation`:
两个界面都使用来自 `scratch_cs336.core.generation` 的生成工具：

- `make_context()` - Build conversation context / 构建对话上下文
- `TextIterStreamer` - Stream tokens in real-time / 实时流式传输token
- Proper multi-turn conversation handling / 正确的多轮对话处理
- Compatible with TinyLLM and HuggingFace models / 兼容TinyLLM和HuggingFace模型

## 📝 Code Quality / 代码质量

All code includes:
所有代码包括：

- ✅ Bilingual comments (Chinese + English) / 双语注释（中文+英文）
- ✅ Type hints / 类型提示
- ✅ Comprehensive docstrings / 全面的文档字符串
- ✅ Error handling / 错误处理
- ✅ Production-ready code / 生产就绪的代码
- ✅ Clean and maintainable structure / 简洁且可维护的结构

## 🔧 Troubleshooting / 故障排除

### Missing Dependencies / 缺少依赖

If you get import errors:
如果出现导入错误：

```bash
# Install core dependencies / 安装核心依赖
pip install torch transformers

# For web UI / 对于Web UI
pip install streamlit
```

### Model Not Found / 找不到模型

Make sure the model path is correct:
确保模型路径正确：

```bash
# Use HuggingFace model ID / 使用HuggingFace模型ID
--model_path wdndev/tiny_llm_sft_92m

# Use local path / 使用本地路径
--model_path outputs/ckpt/tiny_llm_sft_92m
```

### Port Already in Use / 端口已被占用

If the default port (8501) is in use:
如果默认端口（8501）已被占用：

```bash
python scripts/launch_demo.py web --port 8502
```

## 📚 Additional Resources / 其他资源

- Streamlit Documentation: https://docs.streamlit.io
- Transformers Documentation: https://huggingface.co/docs/transformers
- PyTorch Documentation: https://pytorch.org/docs

---

**Created by:** Claude Code
**Date:** 2025-11-14
