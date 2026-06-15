#!/usr/bin/env python3
"""
Process SFT Data - CLI for SFT Data Processing
处理SFT数据 - SFT数据处理命令行工具

This script processes supervised fine-tuning datasets from various sources
into a standardized format.
该脚本将来自各种来源的监督微调数据集处理为标准格式。

Usage / 用法:
    # Process Belle dataset / 处理Belle数据集
    python scripts/process_sft_data.py --belle data/belle.json --output data/sft/

    # Process Firefly dataset / 处理Firefly数据集
    python scripts/process_sft_data.py --firefly data/firefly.jsonl --output data/sft/

    # Process TigerBot dataset / 处理TigerBot数据集
    python scripts/process_sft_data.py --tigerbot data/tigerbot/ --output data/sft/

    # Process and merge all datasets / 处理并合并所有数据集
    python scripts/process_sft_data.py \\
        --belle data/belle.json \\
        --firefly data/firefly.jsonl \\
        --tigerbot data/tigerbot/ \\
        --output data/sft/ \\
        --output-name merged_sft.jsonl
"""

import sys
import argparse
from pathlib import Path


from scratch_cs336.data.processors.sft_processor import SFTDataProcessor
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
        description="Process SFT datasets / 处理SFT数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  # Process single dataset / 处理单个数据集
  %(prog)s --belle corpus/belle.json --output data/sft/

  # Merge multiple datasets / 合并多个数据集
  %(prog)s --belle corpus/belle.json --firefly corpus/firefly.jsonl \\
           --tigerbot corpus/tigerbot/ --output data/sft/ \\
           --output-name merged_sft.jsonl

  # Process with custom output name / 使用自定义输出名称处理
  %(prog)s --belle corpus/belle.json --output data/sft/ --output-name belle_processed.jsonl
        """
    )

    # Input arguments / 输入参数
    input_group = parser.add_argument_group('Input sources / 输入源')
    input_group.add_argument(
        '--belle',
        type=str,
        help='Path to Belle dataset JSON file / Belle数据集JSON文件路径'
    )
    input_group.add_argument(
        '--firefly',
        type=str,
        help='Path to Firefly dataset JSONL file / Firefly数据集JSONL文件路径'
    )
    input_group.add_argument(
        '--tigerbot',
        type=str,
        help='Path to TigerBot dataset directory / TigerBot数据集目录路径'
    )

    # Output arguments / 输出参数
    output_group = parser.add_argument_group('Output options / 输出选项')
    output_group.add_argument(
        '--output',
        '-o',
        type=str,
        required=True,
        help='Output directory path / 输出目录路径'
    )
    output_group.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Output filename (default: auto-generated) / 输出文件名（默认：自动生成）'
    )

    # Processing options / 处理选项
    proc_group = parser.add_argument_group('Processing options / 处理选项')
    proc_group.add_argument(
        '--merge',
        action='store_true',
        default=True,
        help='Merge all datasets into single file / 将所有数据集合并为单个文件 (default: True)'
    )
    proc_group.add_argument(
        '--separate',
        action='store_true',
        help='Save each dataset separately / 分别保存每个数据集'
    )

    args = parser.parse_args()

    # Validation / 验证
    if not any([args.belle, args.firefly, args.tigerbot]):
        parser.error("At least one input source is required / 至少需要一个输入源")

    return args


def main():
    """
    Main function
    主函数
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("SFT Data Processing / SFT数据处理")
    logger.info("=" * 60)

    # Initialize processor / 初始化处理器
    processor = SFTDataProcessor(output_dir=args.output)

    try:
        if args.separate:
            # Process each dataset separately / 分别处理每个数据集
            logger.info("Processing datasets separately / 分别处理数据集")

            if args.belle:
                logger.info(f"Processing Belle dataset: {args.belle}")
                belle_data = processor.process_belle(args.belle)
                output_name = args.output_name if args.output_name else "belle_processed.jsonl"
                processor.save_data(belle_data, output_name)

            if args.firefly:
                logger.info(f"Processing Firefly dataset: {args.firefly}")
                firefly_data = processor.process_firefly(args.firefly)
                output_name = args.output_name if args.output_name else "firefly_processed.jsonl"
                processor.save_data(firefly_data, output_name)

            if args.tigerbot:
                logger.info(f"Processing TigerBot dataset: {args.tigerbot}")
                tigerbot_data = processor.process_tigerbot_sft(args.tigerbot)
                output_name = args.output_name if args.output_name else "tigerbot_processed.jsonl"
                processor.save_data(tigerbot_data, output_name)

        else:
            # Merge all datasets / 合并所有数据集
            logger.info("Processing and merging datasets / 处理并合并数据集")

            output_name = args.output_name if args.output_name else "sft_merged.jsonl"
            merged_data = processor.process_and_merge(
                belle_path=args.belle,
                firefly_path=args.firefly,
                tigerbot_dir=args.tigerbot,
                output_filename=output_name
            )

            logger.info(f"Total merged examples: {len(merged_data)}")

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
