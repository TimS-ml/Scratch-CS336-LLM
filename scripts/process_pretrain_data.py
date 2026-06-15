#!/usr/bin/env python3
"""
Process Pretrain Data - CLI for Pre-training Data Processing
处理预训练数据 - 预训练数据处理命令行工具

This script processes pre-training datasets and converts them into
tokenized binary format for efficient loading during training.
该脚本处理预训练数据集并将其转换为token化的二进制格式，以便在训练期间高效加载。

Usage / 用法:
    # Process Wikipedia dataset / 处理维基百科数据集
    python scripts/process_pretrain_data.py \\
        --wikipedia corpus/wiki_cn.json \\
        --tokenizer tokenizer.model \\
        --output data/pretrain/

    # Process JSONL dataset / 处理JSONL数据集
    python scripts/process_pretrain_data.py \\
        --jsonl corpus/webnovel.jsonl \\
        --text-field text \\
        --tokenizer tokenizer.model \\
        --output data/pretrain/

    # Process Parquet directory / 处理Parquet目录
    python scripts/process_pretrain_data.py \\
        --parquet-dir corpus/tigerbot/ \\
        --text-field content \\
        --tokenizer tokenizer.model \\
        --output data/pretrain/

    # Process Baidu Baike / 处理百度百科
    python scripts/process_pretrain_data.py \\
        --baidu-baike corpus/baidubaike.jsonl \\
        --tokenizer tokenizer.model \\
        --output data/pretrain/

    # Merge binary files / 合并二进制文件
    python scripts/process_pretrain_data.py \\
        --merge data/pretrain/*.bin \\
        --output data/pretrain/merged.bin
"""

import sys
import argparse
import glob
from pathlib import Path
from typing import List


from scratch_cs336.data.processors.pretrain_processor import (
    PretrainDataProcessor,
    merge_binary_files
)
import logging

# Configure logging / 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Parse command line arguments
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="Process pre-training datasets / 处理预训练数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  # Process Wikipedia / 处理维基百科
  %(prog)s --wikipedia corpus/wiki.json --tokenizer tokenizer.model --output data/pretrain/

  # Process JSONL files / 处理JSONL文件
  %(prog)s --jsonl corpus/text.jsonl --text-field content --tokenizer tokenizer.model --output data/pretrain/

  # Process Parquet directory / 处理Parquet目录
  %(prog)s --parquet-dir corpus/data/ --text-field content --tokenizer tokenizer.model --output data/pretrain/

  # Merge binary files / 合并二进制文件
  %(prog)s --merge data/pretrain/part_*.bin --output data/pretrain/merged.bin
        """
    )

    # Input arguments / 输入参数
    input_group = parser.add_argument_group('Input sources / 输入源')
    input_group.add_argument(
        '--wikipedia',
        type=str,
        help='Path to Wikipedia JSON file / 维基百科JSON文件路径'
    )
    input_group.add_argument(
        '--jsonl',
        type=str,
        help='Path to JSONL file / JSONL文件路径'
    )
    input_group.add_argument(
        '--jsonl-dir',
        type=str,
        help='Directory containing JSONL files / 包含JSONL文件的目录'
    )
    input_group.add_argument(
        '--parquet',
        type=str,
        help='Path to Parquet file / Parquet文件路径'
    )
    input_group.add_argument(
        '--parquet-dir',
        type=str,
        help='Directory containing Parquet files / 包含Parquet文件的目录'
    )
    input_group.add_argument(
        '--baidu-baike',
        type=str,
        help='Path to Baidu Baike JSONL file / 百度百科JSONL文件路径'
    )
    input_group.add_argument(
        '--merge',
        type=str,
        nargs='+',
        help='Binary files to merge (supports glob patterns) / 要合并的二进制文件（支持glob模式）'
    )

    # Tokenizer arguments / 分词器参数
    tokenizer_group = parser.add_argument_group('Tokenizer options / 分词器选项')
    tokenizer_group.add_argument(
        '--tokenizer',
        type=str,
        help='Path to tokenizer model file / 分词器模型文件路径'
    )
    tokenizer_group.add_argument(
        '--tokenizer-type',
        type=str,
        default='huggingface',
        choices=['sentencepiece', 'huggingface'],
        help='Tokenizer type / 分词器类型 (default: huggingface)'
    )

    # Processing options / 处理选项
    proc_group = parser.add_argument_group('Processing options / 处理选项')
    proc_group.add_argument(
        '--text-field',
        type=str,
        default='text',
        help='Field name containing text in JSON/Parquet / JSON/Parquet中包含文本的字段名 (default: text)'
    )
    proc_group.add_argument(
        '--batch-size',
        type=int,
        default=1000000,
        help='Batch size for processing / 处理的批大小 (default: 1000000)'
    )
    proc_group.add_argument(
        '--max-tokens-per-file',
        type=int,
        default=10000000,
        help='Maximum tokens per output file / 每个输出文件的最大token数 (default: 10000000)'
    )
    proc_group.add_argument(
        '--output-prefix',
        type=str,
        default='pretrain_part',
        help='Prefix for output filenames / 输出文件名前缀 (default: pretrain_part)'
    )

    # Output arguments / 输出参数
    output_group = parser.add_argument_group('Output options / 输出选项')
    output_group.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help='Output directory or file path / 输出目录或文件路径'
    )

    args = parser.parse_args()

    # Validation / 验证
    if args.merge:
        # Merging mode doesn't need tokenizer / 合并模式不需要分词器
        return args

    if not any([args.wikipedia, args.jsonl, args.jsonl_dir, args.parquet, args.parquet_dir, args.baidu_baike]):
        parser.error("At least one input source is required / 至少需要一个输入源")

    if not args.tokenizer:
        parser.error("Tokenizer path is required for processing / 处理需要分词器路径")

    return args


def load_tokenizer(tokenizer_path: str, tokenizer_type: str):
    """
    Load tokenizer based on type
    根据类型加载分词器

    Args:
        tokenizer_path: Path to tokenizer file / 分词器文件路径
        tokenizer_type: Type of tokenizer / 分词器类型

    Returns:
        Loaded tokenizer / 加载的分词器
    """
    logger.info(f"Loading {tokenizer_type} tokenizer from {tokenizer_path}")

    try:
        if tokenizer_type == 'sentencepiece':
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.load(tokenizer_path)
            return sp

        elif tokenizer_type == 'huggingface':
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(tokenizer_path)

        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    except ImportError as e:
        logger.error(f"Failed to import required library: {e}")
        logger.error("Please install the required package for the tokenizer type")
        raise
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def main():
    """
    Main function
    主函数
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Pre-training Data Processing / 预训练数据处理")
    logger.info("=" * 60)

    try:
        # Merge mode / 合并模式
        if args.merge:
            logger.info("Merging binary files / 合并二进制文件")

            # Expand glob patterns / 展开glob模式
            file_paths: List[str] = []
            for pattern in args.merge:
                expanded = glob.glob(pattern)
                if not expanded:
                    logger.warning(f"No files found matching pattern: {pattern}")
                file_paths.extend(expanded)

            if not file_paths:
                logger.error("No files to merge / 没有要合并的文件")
                return 1

            logger.info(f"Merging {len(file_paths)} files")
            output_path = Path(args.output)

            # If output is a directory, create default filename / 如果输出是目录，创建默认文件名
            if output_path.is_dir() or not output_path.suffix:
                output_path = output_path / "pretrain_merged.bin"

            merge_binary_files(file_paths, output_path)
            logger.info(f"Merged file saved to: {output_path}")
            return 0

        # Processing mode / 处理模式
        # Load tokenizer / 加载分词器
        tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_type)

        # Initialize processor / 初始化处理器
        processor = PretrainDataProcessor(
            tokenizer=tokenizer,
            output_dir=args.output,
            batch_size=args.batch_size
        )

        # Process Wikipedia / 处理维基百科
        if args.wikipedia:
            logger.info(f"Processing Wikipedia dataset: {args.wikipedia}")
            processor.process_wikipedia(args.wikipedia)

        # Process JSONL file / 处理JSONL文件
        if args.jsonl:
            logger.info(f"Processing JSONL file: {args.jsonl}")
            processor.process_jsonl(args.jsonl, text_field=args.text_field)

        # Process JSONL directory / 处理JSONL目录
        if args.jsonl_dir:
            logger.info(f"Processing JSONL directory: {args.jsonl_dir}")
            processor.process_directory(
                args.jsonl_dir,
                pattern="*.jsonl",
                text_field=args.text_field
            )

        # Process Parquet file / 处理Parquet文件
        if args.parquet:
            logger.info(f"Processing Parquet file: {args.parquet}")
            processor.process_parquet(
                args.parquet,
                text_field=args.text_field,
                max_tokens_per_file=args.max_tokens_per_file
            )

        # Process Parquet directory / 处理Parquet目录
        if args.parquet_dir:
            logger.info(f"Processing Parquet directory: {args.parquet_dir}")
            processor.process_parquet_directory(
                args.parquet_dir,
                text_field=args.text_field,
                output_prefix=args.output_prefix,
                max_tokens_per_file=args.max_tokens_per_file
            )

        # Process Baidu Baike / 处理百度百科
        if args.baidu_baike:
            logger.info(f"Processing Baidu Baike dataset: {args.baidu_baike}")
            processor.process_baidu_baike(
                args.baidu_baike,
                output_prefix=args.output_prefix,
                batch_size=args.batch_size
            )

        logger.info("=" * 60)
        logger.info("Processing complete! / 处理完成！")
        logger.info(f"Output directory / 输出目录: {args.output}")
        logger.info("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found / 文件未找到: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error during processing / 处理期间出错: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
