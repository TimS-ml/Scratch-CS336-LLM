#!/usr/bin/env python3
"""
Process RM Data - CLI for Reward Model Data Processing
处理RM数据 - 奖励模型数据处理命令行工具

This script processes reward model (preference) datasets from various sources
into a standardized format and splits them into training and evaluation sets.
该脚本将来自各种来源的奖励模型（偏好）数据集处理为标准格式，并将其拆分为训练集和评估集。

Usage / 用法:
    # Process CValues dataset / 处理CValues数据集
    python scripts/process_rm_data.py \\
        --cvalues corpus/cvalues.jsonl \\
        --output data/rm/ \\
        --eval-size 2000

    # Process RLHF Parquet dataset / 处理RLHF Parquet数据集
    python scripts/process_rm_data.py \\
        --rlhf corpus/rlhf.parquet \\
        --output data/rm/ \\
        --eval-size 0.1

    # Process Zhihu RLHF TSV dataset / 处理知乎RLHF TSV数据集
    python scripts/process_rm_data.py \\
        --zhihu corpus/zhihu_rlhf.tsv \\
        --output data/rm/ \\
        --eval-size 500

    # Process entire directory / 处理整个目录
    python scripts/process_rm_data.py \\
        --directory corpus/rm_train/ \\
        --output data/rm/ \\
        --eval-size 2000 \\
        --shuffle
"""

import sys
import argparse
from pathlib import Path

# Add project root to path / 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clean_llm.data.processors.rm_processor import RMDataProcessor
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
        description="Process reward model (preference) datasets / 处理奖励模型（偏好）数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  # Process single dataset / 处理单个数据集
  %(prog)s --cvalues corpus/cvalues.jsonl --output data/rm/ --eval-size 2000

  # Process directory with multiple formats / 处理包含多种格式的目录
  %(prog)s --directory corpus/rm_train/ --output data/rm/ --eval-size 0.1 --shuffle

  # Process without splitting / 处理但不拆分
  %(prog)s --rlhf corpus/rlhf.parquet --output data/rm/ --no-split

  # Custom output filenames / 自定义输出文件名
  %(prog)s --directory corpus/rm/ --output data/rm/ \\
           --train-name my_train.jsonl --eval-name my_eval.jsonl
        """
    )

    # Input arguments / 输入参数
    input_group = parser.add_argument_group('Input sources / 输入源')
    input_group.add_argument(
        '--cvalues',
        type=str,
        help='Path to CValues JSONL file / CValues JSONL文件路径'
    )
    input_group.add_argument(
        '--rlhf',
        type=str,
        help='Path to RLHF Parquet file / RLHF Parquet文件路径'
    )
    input_group.add_argument(
        '--zhihu',
        type=str,
        help='Path to Zhihu RLHF TSV file / 知乎RLHF TSV文件路径'
    )
    input_group.add_argument(
        '--directory',
        '--dir',
        type=str,
        help='Directory containing RM datasets (JSONL, Parquet, TSV) / 包含RM数据集的目录（JSONL、Parquet、TSV）'
    )

    # Processing options / 处理选项
    proc_group = parser.add_argument_group('Processing options / 处理选项')
    proc_group.add_argument(
        '--eval-size',
        type=str,
        default='2000',
        help='Evaluation set size (int for count, float for fraction) / 评估集大小（整数表示数量，浮点数表示比例）(default: 2000)'
    )
    proc_group.add_argument(
        '--shuffle',
        action='store_true',
        default=True,
        help='Shuffle data before splitting / 拆分前打乱数据 (default: True)'
    )
    proc_group.add_argument(
        '--no-shuffle',
        action='store_false',
        dest='shuffle',
        help='Do not shuffle data before splitting / 拆分前不打乱数据'
    )
    proc_group.add_argument(
        '--no-split',
        action='store_true',
        help='Do not split into train/eval, save all as single file / 不拆分为训练/评估，保存为单个文件'
    )
    proc_group.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for shuffling / 打乱的随机种子 (default: 42)'
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
        '--train-name',
        type=str,
        default='rm_train.jsonl',
        help='Training data filename / 训练数据文件名 (default: rm_train.jsonl)'
    )
    output_group.add_argument(
        '--eval-name',
        type=str,
        default='rm_eval.jsonl',
        help='Evaluation data filename / 评估数据文件名 (default: rm_eval.jsonl)'
    )
    output_group.add_argument(
        '--all-name',
        type=str,
        default='rm_all.jsonl',
        help='All data filename (when --no-split) / 所有数据文件名（当使用--no-split时）(default: rm_all.jsonl)'
    )

    # Format selection / 格式选择
    format_group = parser.add_argument_group('Format selection (for directory mode) / 格式选择（目录模式）')
    format_group.add_argument(
        '--formats',
        nargs='+',
        choices=['jsonl', 'parquet', 'tsv'],
        default=['jsonl', 'parquet', 'tsv'],
        help='File formats to process from directory / 从目录处理的文件格式 (default: all)'
    )

    args = parser.parse_args()

    # Validation / 验证
    if not any([args.cvalues, args.rlhf, args.zhihu, args.directory]):
        parser.error("At least one input source is required / 至少需要一个输入源")

    # Convert eval_size to appropriate type / 将eval_size转换为适当的类型
    try:
        if '.' in args.eval_size:
            args.eval_size = float(args.eval_size)
        else:
            args.eval_size = int(args.eval_size)
    except ValueError:
        parser.error(f"Invalid eval-size value: {args.eval_size}")

    return args


def main():
    """
    Main function
    主函数
    """
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Reward Model Data Processing / 奖励模型数据处理")
    logger.info("=" * 60)

    # Initialize processor / 初始化处理器
    processor = RMDataProcessor(output_dir=args.output)

    try:
        all_data = []

        # Process directory / 处理目录
        if args.directory:
            logger.info(f"Processing directory: {args.directory}")
            # Convert format list to extension format / 将格式列表转换为扩展名格式
            formats = [f'.{fmt}' for fmt in args.formats]
            data = processor.process_directory(args.directory, formats=formats)
            all_data.extend(data)

        # Process CValues / 处理CValues
        if args.cvalues:
            logger.info(f"Processing CValues dataset: {args.cvalues}")
            data = processor.process_cvalues(args.cvalues)
            all_data.extend(data)

        # Process RLHF Parquet / 处理RLHF Parquet
        if args.rlhf:
            logger.info(f"Processing RLHF Parquet dataset: {args.rlhf}")
            data = processor.process_rlhf_parquet(args.rlhf)
            all_data.extend(data)

        # Process Zhihu RLHF / 处理知乎RLHF
        if args.zhihu:
            logger.info(f"Processing Zhihu RLHF dataset: {args.zhihu}")
            data = processor.process_zhihu_rlhf_tsv(args.zhihu)
            all_data.extend(data)

        logger.info(f"Total examples processed: {len(all_data)}")

        if not all_data:
            logger.warning("No data was processed / 没有处理任何数据")
            return 0

        # Save data / 保存数据
        if args.no_split:
            # Save all data without splitting / 保存所有数据不拆分
            logger.info("Saving all data without splitting / 保存所有数据不拆分")
            output_path = Path(args.output) / args.all_name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in all_data:
                    outfile.write(line)

            logger.info(f"Saved {len(all_data)} examples to {output_path}")

        else:
            # Split and save / 拆分并保存
            logger.info("Splitting data into train and eval sets / 将数据拆分为训练集和评估集")
            train_data, eval_data = processor.split_train_eval(
                all_data,
                eval_size=args.eval_size,
                shuffle=args.shuffle,
                random_seed=args.random_seed
            )

            logger.info(f"Train examples: {len(train_data)}")
            logger.info(f"Eval examples: {len(eval_data)}")

            train_path, eval_path = processor.save_data(
                train_data,
                eval_data,
                train_filename=args.train_name,
                eval_filename=args.eval_name
            )

            logger.info(f"Training data saved to: {train_path}")
            logger.info(f"Evaluation data saved to: {eval_path}")

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
