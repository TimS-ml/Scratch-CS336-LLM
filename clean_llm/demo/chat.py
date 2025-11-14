"""
Command-line chat interface for LLM interaction.
用于LLM交互的命令行聊天界面。

This module provides a simple CLI for chatting with language models,
supporting conversation history and configurable generation parameters.

此模块提供了一个简单的CLI用于与语言模型聊天，
支持对话历史和可配置的生成参数。

Usage:
    python -m clean_llm.demo.chat --model_path <path_to_model>

Example:
    python -m clean_llm.demo.chat --model_path outputs/ckpt/tiny_llm_sft_92m
"""

import os
import sys
import argparse
import threading
from pathlib import Path
from typing import List, Dict, Optional, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Add parent directory to path for imports
# 将父目录添加到路径以便导入
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from clean_llm.generation import make_context, TextIterStreamer


# ============================================================================
# Colors for terminal output / 终端输出的颜色
# ============================================================================

class Colors:
    """ANSI color codes for terminal output / 终端输出的ANSI颜色代码"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


# ============================================================================
# Chat Interface / 聊天界面
# ============================================================================

class ChatInterface:
    """
    Command-line chat interface for LLM interaction.
    用于LLM交互的命令行聊天界面。

    This class manages conversation history, model interaction, and provides
    a user-friendly CLI for chatting with language models.

    此类管理对话历史、模型交互，并提供用户友好的CLI用于与语言模型聊天。

    Args:
        model_path: Path to model or HuggingFace model ID.
                   模型路径或HuggingFace模型ID。
        system_prompt: System prompt to guide model behavior.
                      用于引导模型行为的系统提示。
        max_new_tokens: Maximum tokens to generate per response.
                       每次响应生成的最大token数。
        temperature: Sampling temperature (higher = more random).
                    采样温度（越高=越随机）。
        top_p: Nucleus sampling threshold.
              核采样阈值。
        top_k: Top-k sampling (0 = disabled).
              Top-k采样（0=禁用）。
        do_sample: Whether to use sampling instead of greedy decoding.
                  是否使用采样而不是贪婪解码。
        use_streaming: Whether to stream output token by token.
                      是否逐token流式输出。
        device_map: Device mapping strategy.
                   设备映射策略。
        trust_remote_code: Whether to trust remote code.
                          是否信任远程代码。
    """

    def __init__(
        self,
        model_path: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 0,
        do_sample: bool = True,
        use_streaming: bool = True,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.use_streaming = use_streaming

        # Initialize conversation history / 初始化对话历史
        self.messages: List[Dict[str, str]] = []

        # Load model, tokenizer, and config / 加载模型、分词器和配置
        print(f"{Colors.CYAN}Loading model from: {model_path}{Colors.RESET}")
        print(f"{Colors.CYAN}正在加载模型: {model_path}{Colors.RESET}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=False,
                trust_remote_code=trust_remote_code
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )

            try:
                self.generation_config = GenerationConfig.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code
                )
            except Exception as e:
                print(f"{Colors.YELLOW}Warning: Could not load generation config, using defaults{Colors.RESET}")
                print(f"{Colors.YELLOW}警告: 无法加载生成配置，使用默认值{Colors.RESET}")
                self.generation_config = GenerationConfig()

            # Update generation config / 更新生成配置
            self.generation_config.max_new_tokens = max_new_tokens
            self.generation_config.temperature = temperature
            self.generation_config.top_p = top_p
            self.generation_config.top_k = top_k if top_k > 0 else None
            self.generation_config.do_sample = do_sample

            print(f"{Colors.GREEN}Model loaded successfully!{Colors.RESET}")
            print(f"{Colors.GREEN}模型加载成功！{Colors.RESET}\n")

        except Exception as e:
            print(f"{Colors.RED}Failed to load model: {e}{Colors.RESET}")
            print(f"{Colors.RED}模型加载失败: {e}{Colors.RESET}")
            raise

    def clear_history(self):
        """
        Clear conversation history.
        清除对话历史。
        """
        self.messages = []
        print(f"{Colors.YELLOW}Conversation history cleared.{Colors.RESET}")
        print(f"{Colors.YELLOW}对话历史已清除。{Colors.RESET}")

    def generate_response(self, user_input: str) -> str:
        """
        Generate model response for user input.
        为用户输入生成模型响应。

        Args:
            user_input: User's message / 用户消息

        Returns:
            str: Model's response / 模型响应
        """
        # Add user message to history / 将用户消息添加到历史
        self.messages.append({"role": "user", "content": user_input})

        # Prepare messages with system prompt / 准备带系统提示的消息
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.messages)

        try:
            if self.use_streaming:
                response = self._generate_with_streaming(messages)
            else:
                response = self._generate_without_streaming(messages)

            # Add assistant response to history / 将助手响应添加到历史
            self.messages.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            error_msg = f"Generation failed: {e}"
            print(f"{Colors.RED}{error_msg}{Colors.RESET}")
            return error_msg

    def _generate_with_streaming(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response with streaming output.
        使用流式输出生成响应。
        """
        # Create streamer / 创建流式处理器
        streamer = TextIterStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            use_pot=False
        )

        # Build input context / 构建输入上下文
        input_ids = make_context(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=messages,
            system=self.system_prompt,
            max_new_tokens=self.generation_config.max_new_tokens
        )

        # Prepare generation kwargs / 准备生成参数
        generation_kwargs = {
            "input_ids": input_ids,
            "generation_config": self.generation_config,
            "streamer": streamer,
            "return_dict_in_generate": True,
        }

        # Start generation in separate thread / 在单独线程中启动生成
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output / 流式输出
        print(f"{Colors.GREEN}Assistant: {Colors.RESET}", end="", flush=True)
        print(f"{Colors.GREEN}助手: {Colors.RESET}", end="", flush=True)

        generated_text = ""
        for new_text in streamer:
            # Print only new content / 仅打印新内容
            if len(new_text) > len(generated_text):
                print(new_text[len(generated_text):], end="", flush=True)
            generated_text = new_text

        print()  # New line after generation / 生成后换行
        thread.join()

        return generated_text

    def _generate_without_streaming(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response without streaming.
        不使用流式输出生成响应。
        """
        # Build input context / 构建输入上下文
        input_ids = make_context(
            model=self.model,
            tokenizer=self.tokenizer,
            messages=messages,
            system=self.system_prompt,
            max_new_tokens=self.generation_config.max_new_tokens
        )

        # Generate / 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                return_dict_in_generate=True
            )

        # Decode output / 解码输出
        output_ids = generated_ids.sequences[0][len(input_ids[0]):]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        print(f"{Colors.GREEN}Assistant: {Colors.RESET}{response}")
        print(f"{Colors.GREEN}助手: {Colors.RESET}{response}")

        return response

    def run(self):
        """
        Run the interactive chat loop.
        运行交互式聊天循环。
        """
        # Print welcome message / 打印欢迎消息
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}LLM Chat Interface / LLM聊天界面{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.RESET}\n")

        print(f"{Colors.YELLOW}Commands / 命令:{Colors.RESET}")
        print(f"  {Colors.WHITE}/clear{Colors.RESET}  - Clear conversation history / 清除对话历史")
        print(f"  {Colors.WHITE}/exit{Colors.RESET}   - Exit chat / 退出聊天")
        print(f"  {Colors.WHITE}/quit{Colors.RESET}   - Exit chat / 退出聊天\n")

        print(f"{Colors.CYAN}System Prompt / 系统提示:{Colors.RESET} {self.system_prompt}\n")
        print(f"{Colors.GREEN}Ready to chat! Type your message and press Enter.{Colors.RESET}")
        print(f"{Colors.GREEN}准备就绪！输入消息并按回车键。{Colors.RESET}\n")

        # Main chat loop / 主聊天循环
        while True:
            try:
                # Get user input / 获取用户输入
                user_input = input(f"{Colors.BLUE}You: {Colors.RESET}").strip()
                print(f"{Colors.BLUE}你: {Colors.RESET}{user_input}")

                # Handle empty input / 处理空输入
                if not user_input:
                    continue

                # Handle commands / 处理命令
                if user_input.lower() == "/clear":
                    self.clear_history()
                    print()
                    continue

                if user_input.lower() in ["/exit", "/quit"]:
                    print(f"\n{Colors.CYAN}Goodbye! / 再见！{Colors.RESET}\n")
                    break

                # Generate response / 生成响应
                self.generate_response(user_input)
                print()

            except KeyboardInterrupt:
                print(f"\n\n{Colors.CYAN}Interrupted. Goodbye! / 已中断。再见！{Colors.RESET}\n")
                break

            except Exception as e:
                print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                print(f"{Colors.RED}错误: {e}{Colors.RESET}\n")


# ============================================================================
# Main Entry Point / 主入口点
# ============================================================================

def main():
    """
    Main entry point for CLI chat.
    CLI聊天的主入口点。
    """
    parser = argparse.ArgumentParser(
        description="Command-line chat interface for LLM / LLM命令行聊天界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  %(prog)s --model_path wdndev/tiny_llm_sft_92m
  %(prog)s --model_path outputs/ckpt/tiny_llm_sft_92m --max_new_tokens 256
  %(prog)s --model_path my_model --system_prompt "You are a helpful coding assistant."
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=os.getenv("MODEL_PATH", "wdndev/tiny_llm_sft_92m"),
        help="Path to model or HuggingFace model ID / 模型路径或HuggingFace模型ID"
    )

    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt to guide model behavior / 用于引导模型行为的系统提示"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate / 生成的最大token数"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature / 采样温度"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Nucleus sampling threshold / 核采样阈值"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling (0 = disabled) / Top-k采样（0=禁用）"
    )

    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling (use greedy decoding) / 禁用采样（使用贪婪解码）"
    )

    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming output / 禁用流式输出"
    )

    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy / 设备映射策略"
    )

    args = parser.parse_args()

    # Create and run chat interface / 创建并运行聊天界面
    try:
        chat = ChatInterface(
            model_path=args.model_path,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.no_sample,
            use_streaming=not args.no_streaming,
            device_map=args.device_map,
        )

        chat.run()

    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.RESET}")
        print(f"{Colors.RED}致命错误: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
