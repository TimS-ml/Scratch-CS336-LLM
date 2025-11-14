"""
Streamlit-based web interface for LLM chat demo.
基于Streamlit的LLM聊天演示Web界面。

This module provides an interactive web UI for chatting with language models,
supporting both TinyLLM and standard HuggingFace models.

此模块提供了一个交互式Web UI用于与语言模型聊天，
支持TinyLLM和标准HuggingFace模型。

Usage:
    streamlit run web_ui.py -- --model_path <path_to_model>

Example:
    streamlit run web_ui.py -- --model_path outputs/ckpt/tiny_llm_sft_92m
"""

import os
import sys
import json
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Add parent directory to path for imports
# 将父目录添加到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clean_llm.generation import make_context, TextIterStreamer


# ============================================================================
# Configuration / 配置
# ============================================================================

DEFAULT_SYSTEM_PROMPT_EN = "You are a helpful AI assistant."
DEFAULT_SYSTEM_PROMPT_ZH = "你是一个有帮助的AI助手。"


# ============================================================================
# Model Loading / 模型加载
# ============================================================================

@st.cache_resource
def load_model_and_tokenizer(
    model_path: str,
    device_map: str = "auto",
    trust_remote_code: bool = True
) -> tuple:
    """
    Load model, tokenizer, and generation config with caching.
    使用缓存加载模型、分词器和生成配置。

    Args:
        model_path: Path to the model directory or HuggingFace model ID.
                   模型目录路径或HuggingFace模型ID。
        device_map: Device mapping strategy for model loading.
                   模型加载的设备映射策略。
        trust_remote_code: Whether to trust remote code in model config.
                          是否信任模型配置中的远程代码。

    Returns:
        tuple: (model, tokenizer, generation_config)
              (模型, 分词器, 生成配置)
    """
    try:
        # Load tokenizer / 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            trust_remote_code=trust_remote_code
        )

        # Load model / 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Load generation config / 加载生成配置
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code
            )
        except Exception as e:
            st.warning(f"Could not load generation config, using defaults: {e}")
            st.warning(f"无法加载生成配置，使用默认值: {e}")
            generation_config = GenerationConfig()

        return model, tokenizer, generation_config

    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error(f"加载模型失败: {e}")
        raise


# ============================================================================
# Session State Management / 会话状态管理
# ============================================================================

def init_session_state():
    """
    Initialize Streamlit session state variables.
    初始化Streamlit会话状态变量。
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT_EN


def clear_chat_history():
    """
    Clear conversation history.
    清除对话历史。
    """
    st.session_state.messages = []


# ============================================================================
# UI Components / UI组件
# ============================================================================

def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar with generation parameters.
    渲染侧边栏及生成参数。

    Returns:
        dict: Generation parameters / 生成参数
    """
    st.sidebar.title("⚙️ Settings / 设置")

    # System prompt / 系统提示
    st.sidebar.subheader("System Prompt / 系统提示")
    system_prompt = st.sidebar.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=100,
        help="System prompt to guide the model's behavior / 用于引导模型行为的系统提示"
    )
    st.session_state.system_prompt = system_prompt

    st.sidebar.divider()

    # Generation parameters / 生成参数
    st.sidebar.subheader("Generation Parameters / 生成参数")

    max_new_tokens = st.sidebar.slider(
        "Max New Tokens / 最大新token数",
        min_value=1,
        max_value=2048,
        value=512,
        step=1,
        help="Maximum number of tokens to generate / 最大生成token数"
    )

    temperature = st.sidebar.slider(
        "Temperature / 温度",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.01,
        help="Controls randomness. Lower = more focused, Higher = more creative / "
             "控制随机性。较低=更集中，较高=更有创意"
    )

    top_p = st.sidebar.slider(
        "Top P / 核采样",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
        help="Nucleus sampling threshold / 核采样阈值"
    )

    top_k = st.sidebar.slider(
        "Top K",
        min_value=0,
        max_value=100,
        value=0,
        step=1,
        help="Top-k sampling (0 = disabled) / Top-k采样（0=禁用）"
    )

    do_sample = st.sidebar.checkbox(
        "Enable Sampling / 启用采样",
        value=True,
        help="Use sampling instead of greedy decoding / 使用采样而不是贪婪解码"
    )

    use_streaming = st.sidebar.checkbox(
        "Enable Streaming / 启用流式输出",
        value=True,
        help="Stream tokens as they are generated / 在生成时流式输出token"
    )

    st.sidebar.divider()

    # Clear button / 清除按钮
    if st.sidebar.button("🗑️ Clear Chat / 清除对话", use_container_width=True):
        clear_chat_history()
        st.rerun()

    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "do_sample": do_sample,
        "use_streaming": use_streaming,
    }


def render_chat_history():
    """
    Render conversation history.
    渲染对话历史。
    """
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(content)


# ============================================================================
# Generation Functions / 生成函数
# ============================================================================

def generate_response(
    model: Any,
    tokenizer: Any,
    generation_config: GenerationConfig,
    messages: List[Dict[str, str]],
    params: Dict[str, Any],
    placeholder: Any
) -> str:
    """
    Generate model response for given messages.
    为给定消息生成模型响应。

    Args:
        model: Language model instance / 语言模型实例
        tokenizer: Tokenizer instance / 分词器实例
        generation_config: Generation configuration / 生成配置
        messages: Conversation messages / 对话消息
        params: Generation parameters / 生成参数
        placeholder: Streamlit placeholder for output / 用于输出的Streamlit占位符

    Returns:
        str: Generated response / 生成的响应
    """
    # Update generation config with UI parameters
    # 使用UI参数更新生成配置
    generation_config.max_new_tokens = params["max_new_tokens"]
    generation_config.temperature = params["temperature"]
    generation_config.top_p = params["top_p"]
    generation_config.top_k = params["top_k"] if params["top_k"] > 0 else None
    generation_config.do_sample = params["do_sample"]

    try:
        if params["use_streaming"]:
            # Streaming generation / 流式生成
            return generate_with_streaming(
                model, tokenizer, generation_config, messages, placeholder
            )
        else:
            # Non-streaming generation / 非流式生成
            return generate_without_streaming(
                model, tokenizer, generation_config, messages, placeholder
            )
    except Exception as e:
        error_msg = f"Generation failed: {e}\n生成失败: {e}"
        placeholder.error(error_msg)
        return error_msg


def generate_with_streaming(
    model: Any,
    tokenizer: Any,
    generation_config: GenerationConfig,
    messages: List[Dict[str, str]],
    placeholder: Any
) -> str:
    """
    Generate response with streaming output.
    使用流式输出生成响应。
    """
    # Create streamer / 创建流式处理器
    streamer = TextIterStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        use_pot=False  # Disable PoT for chat / 聊天模式禁用PoT
    )

    # Build input context / 构建输入上下文
    input_ids = make_context(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        system=st.session_state.system_prompt,
        max_new_tokens=generation_config.max_new_tokens
    )

    # Prepare generation kwargs / 准备生成参数
    generation_kwargs = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "streamer": streamer,
        "return_dict_in_generate": True,
    }

    # Start generation in separate thread / 在单独线程中启动生成
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream output / 流式输出
    generated_text = ""
    for new_text in streamer:
        generated_text = new_text
        placeholder.markdown(generated_text)

    thread.join()

    return generated_text


def generate_without_streaming(
    model: Any,
    tokenizer: Any,
    generation_config: GenerationConfig,
    messages: List[Dict[str, str]],
    placeholder: Any
) -> str:
    """
    Generate response without streaming (all at once).
    不使用流式输出生成响应（一次性输出）。
    """
    # Build input context / 构建输入上下文
    input_ids = make_context(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        system=st.session_state.system_prompt,
        max_new_tokens=generation_config.max_new_tokens
    )

    # Generate / 生成
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True
        )

    # Decode output / 解码输出
    output_ids = generated_ids.sequences[0][len(input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)

    # Display / 显示
    placeholder.markdown(response)

    return response


# ============================================================================
# Main Application / 主应用
# ============================================================================

def main():
    """
    Main application entry point.
    主应用入口点。
    """
    # Page config / 页面配置
    st.set_page_config(
        page_title="LLM Chat Demo / LLM聊天演示",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Title / 标题
    st.title("🤖 LLM Chat Demo / LLM聊天演示")

    # Parse command line arguments / 解析命令行参数
    parser = argparse.ArgumentParser(description="LLM Chat Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", "wdndev/tiny_llm_sft_92m"),
        help="Path to model or HuggingFace model ID"
    )

    # Get model path from session state or args
    # 从会话状态或参数获取模型路径
    if "model_path" not in st.session_state:
        try:
            args = parser.parse_args()
            st.session_state.model_path = args.model_path
        except:
            st.session_state.model_path = "wdndev/tiny_llm_sft_92m"

    # Display model info / 显示模型信息
    st.sidebar.info(f"**Model / 模型:** `{st.session_state.model_path}`")

    # Initialize session state / 初始化会话状态
    init_session_state()

    # Load model / 加载模型
    try:
        with st.spinner("Loading model... / 正在加载模型..."):
            model, tokenizer, generation_config = load_model_and_tokenizer(
                st.session_state.model_path
            )
        st.success("Model loaded successfully! / 模型加载成功！")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.error(f"模型加载失败: {e}")
        st.stop()

    # Render sidebar and get parameters / 渲染侧边栏并获取参数
    params = render_sidebar()

    # Show welcome message / 显示欢迎消息
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            welcome_msg = (
                "Hello! I'm an AI assistant. How can I help you today?\n\n"
                "你好！我是AI助手。有什么可以帮助你的吗？"
            )
            st.markdown(welcome_msg)

    # Render chat history / 渲染对话历史
    render_chat_history()

    # Chat input / 聊天输入
    if prompt := st.chat_input("Type your message here... / 在此输入消息..."):
        # Add user message / 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message / 显示用户消息
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Generate assistant response / 生成助手响应
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()

            # Prepare messages for model / 为模型准备消息
            messages = [{"role": "system", "content": st.session_state.system_prompt}]
            messages.extend(st.session_state.messages)

            # Generate response / 生成响应
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                messages=messages,
                params=params,
                placeholder=placeholder
            )

        # Add assistant message / 添加助手消息
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Log for debugging / 调试日志
        print(f"User: {prompt}")
        print(f"Assistant: {response}")
        print(f"用户: {prompt}")
        print(f"助手: {response}")


if __name__ == "__main__":
    main()
