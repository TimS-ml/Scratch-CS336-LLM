#!/usr/bin/env python3
"""
Launcher script for LLM demo applications.
LLM演示应用的启动脚本。

This script provides a convenient entry point for launching either the
web UI or CLI chat interface with proper configuration and error handling.

此脚本提供了一个方便的入口点，用于启动Web UI或CLI聊天界面，
具有适当的配置和错误处理。

Usage:
    # Launch web UI / 启动Web UI
    python scripts/launch_demo.py web --model_path <path>

    # Launch CLI chat / 启动CLI聊天
    python scripts/launch_demo.py cli --model_path <path>

    # Show help / 显示帮助
    python scripts/launch_demo.py --help
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


# ============================================================================
# Configuration / 配置
# ============================================================================

DEFAULT_MODEL_PATH = "wdndev/tiny_llm_sft_92m"
DEFAULT_PORT = 8501


# ============================================================================
# Utility Functions / 实用函数
# ============================================================================

def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.
    检查是否安装了所需的依赖项。

    Returns:
        bool: True if all dependencies are available, False otherwise.
             如果所有依赖项都可用则返回True，否则返回False。
    """
    try:
        import torch
        import transformers
        return True
    except ImportError as e:
        print(f"Error: Missing required dependency: {e}")
        print(f"错误: 缺少必需的依赖项: {e}")
        print("\nPlease install required packages:")
        print("请安装必需的软件包:")
        print("  pip install torch transformers")
        return False


def check_streamlit() -> bool:
    """
    Check if Streamlit is installed.
    检查是否安装了Streamlit。

    Returns:
        bool: True if Streamlit is available, False otherwise.
             如果Streamlit可用则返回True，否则返回False。
    """
    try:
        import streamlit
        return True
    except ImportError:
        print("Error: Streamlit is not installed.")
        print("错误: 未安装Streamlit。")
        print("\nPlease install Streamlit:")
        print("请安装Streamlit:")
        print("  pip install streamlit")
        return False


def get_project_root() -> Path:
    """
    Get the project root directory.
    获取项目根目录。

    Returns:
        Path: Path to project root.
             项目根目录的路径。
    """
    # Assume script is in scripts/ directory
    # 假设脚本在scripts/目录中
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    return project_root


# ============================================================================
# Launch Functions / 启动函数
# ============================================================================

def launch_web_ui(
    model_path: str,
    port: int = DEFAULT_PORT,
    host: str = "0.0.0.0",
    extra_args: Optional[List[str]] = None
):
    """
    Launch Streamlit web UI.
    启动Streamlit Web UI。

    Args:
        model_path: Path to model or HuggingFace model ID.
                   模型路径或HuggingFace模型ID。
        port: Port number for web server.
             Web服务器的端口号。
        host: Host address to bind to.
             要绑定的主机地址。
        extra_args: Additional arguments to pass to Streamlit.
                   要传递给Streamlit的额外参数。
    """
    # Check dependencies / 检查依赖项
    if not check_dependencies() or not check_streamlit():
        sys.exit(1)

    # Get paths / 获取路径
    project_root = get_project_root()
    web_ui_path = project_root / "clean_llm" / "demo" / "web_ui.py"

    if not web_ui_path.exists():
        print(f"Error: Web UI file not found: {web_ui_path}")
        print(f"错误: 未找到Web UI文件: {web_ui_path}")
        sys.exit(1)

    # Set environment variable for model path / 设置模型路径的环境变量
    os.environ["MODEL_PATH"] = model_path

    # Build streamlit command / 构建streamlit命令
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(web_ui_path),
        "--server.port",
        str(port),
        "--server.address",
        host,
        "--",
        "--model_path",
        model_path,
    ]

    # Add extra arguments / 添加额外参数
    if extra_args:
        cmd.extend(extra_args)

    # Print launch info / 打印启动信息
    print("=" * 70)
    print("Launching LLM Web UI...")
    print("启动LLM Web UI...")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"模型: {model_path}")
    print(f"\nServer will be available at:")
    print(f"服务器将在以下地址可用:")
    print(f"  Local URL: http://localhost:{port}")
    print(f"  Network URL: http://{host}:{port}")
    print(f"\nPress Ctrl+C to stop the server.")
    print(f"按Ctrl+C停止服务器。\n")
    print("=" * 70)
    print()

    # Launch / 启动
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        print("正在关闭服务器...")
    except Exception as e:
        print(f"\nError launching web UI: {e}")
        print(f"启动Web UI时出错: {e}")
        sys.exit(1)


def launch_cli_chat(
    model_path: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 0,
    no_sample: bool = False,
    no_streaming: bool = False,
    extra_args: Optional[List[str]] = None
):
    """
    Launch CLI chat interface.
    启动CLI聊天界面。

    Args:
        model_path: Path to model or HuggingFace model ID.
                   模型路径或HuggingFace模型ID。
        system_prompt: System prompt to guide model behavior.
                      用于引导模型行为的系统提示。
        max_new_tokens: Maximum tokens to generate.
                       生成的最大token数。
        temperature: Sampling temperature.
                    采样温度。
        top_p: Nucleus sampling threshold.
              核采样阈值。
        top_k: Top-k sampling.
              Top-k采样。
        no_sample: Disable sampling.
                  禁用采样。
        no_streaming: Disable streaming.
                     禁用流式输出。
        extra_args: Additional arguments.
                   额外参数。
    """
    # Check dependencies / 检查依赖项
    if not check_dependencies():
        sys.exit(1)

    # Get paths / 获取路径
    project_root = get_project_root()
    chat_path = project_root / "clean_llm" / "demo" / "chat.py"

    if not chat_path.exists():
        print(f"Error: Chat file not found: {chat_path}")
        print(f"错误: 未找到Chat文件: {chat_path}")
        sys.exit(1)

    # Build command / 构建命令
    cmd = [
        sys.executable,
        str(chat_path),
        "--model_path",
        model_path,
        "--max_new_tokens",
        str(max_new_tokens),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(top_k),
    ]

    if system_prompt:
        cmd.extend(["--system_prompt", system_prompt])

    if no_sample:
        cmd.append("--no_sample")

    if no_streaming:
        cmd.append("--no_streaming")

    # Add extra arguments / 添加额外参数
    if extra_args:
        cmd.extend(extra_args)

    # Launch / 启动
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\nExiting...")
        print("退出中...")
    except Exception as e:
        print(f"\nError launching CLI chat: {e}")
        print(f"启动CLI聊天时出错: {e}")
        sys.exit(1)


# ============================================================================
# Main Entry Point / 主入口点
# ============================================================================

def main():
    """
    Main entry point for demo launcher.
    演示启动器的主入口点。
    """
    parser = argparse.ArgumentParser(
        description="Launch LLM demo applications / 启动LLM演示应用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  # Launch web UI with default model / 使用默认模型启动Web UI
  %(prog)s web

  # Launch web UI with custom model / 使用自定义模型启动Web UI
  %(prog)s web --model_path outputs/ckpt/tiny_llm_sft_92m

  # Launch web UI on custom port / 在自定义端口上启动Web UI
  %(prog)s web --port 8080

  # Launch CLI chat / 启动CLI聊天
  %(prog)s cli --model_path wdndev/tiny_llm_sft_92m

  # Launch CLI chat with custom parameters / 使用自定义参数启动CLI聊天
  %(prog)s cli --max_new_tokens 256 --temperature 0.5
        """
    )

    subparsers = parser.add_subparsers(
        dest="mode",
        help="Demo mode to launch / 要启动的演示模式"
    )

    # Web UI subcommand / Web UI子命令
    web_parser = subparsers.add_parser(
        "web",
        help="Launch web UI / 启动Web UI"
    )
    web_parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model (default: {DEFAULT_MODEL_PATH}) / 模型路径"
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port number (default: {DEFAULT_PORT}) / 端口号"
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address (default: 0.0.0.0) / 主机地址"
    )

    # CLI chat subcommand / CLI聊天子命令
    cli_parser = subparsers.add_parser(
        "cli",
        help="Launch CLI chat / 启动CLI聊天"
    )
    cli_parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to model (default: {DEFAULT_MODEL_PATH}) / 模型路径"
    )
    cli_parser.add_argument(
        "--system_prompt",
        type=str,
        help="System prompt / 系统提示"
    )
    cli_parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512) / 生成的最大token数"
    )
    cli_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7) / 采样温度"
    )
    cli_parser.add_argument(
        "--top_p",
        type=float,
        default=0.8,
        help="Nucleus sampling (default: 0.8) / 核采样"
    )
    cli_parser.add_argument(
        "--top_k",
        type=int,
        default=0,
        help="Top-k sampling (default: 0) / Top-k采样"
    )
    cli_parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Disable sampling / 禁用采样"
    )
    cli_parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming / 禁用流式输出"
    )

    args = parser.parse_args()

    # Check if mode was provided / 检查是否提供了模式
    if not args.mode:
        parser.print_help()
        print("\nError: Please specify a mode (web or cli)")
        print("错误: 请指定模式（web或cli）")
        sys.exit(1)

    # Launch appropriate demo / 启动适当的演示
    if args.mode == "web":
        launch_web_ui(
            model_path=args.model_path,
            port=args.port,
            host=args.host
        )
    elif args.mode == "cli":
        launch_cli_chat(
            model_path=args.model_path,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            no_sample=args.no_sample,
            no_streaming=args.no_streaming
        )


if __name__ == "__main__":
    main()
