"""
RM Data Processor - Reward Model Data Processing
奖励模型数据处理器

This module provides utilities for processing reward model (preference) datasets
from various sources into a standardized format.
本模块提供从各种来源处理奖励模型（偏好）数据集为标准格式的工具。

Supported datasets / 支持的数据集:
- CValues-Comparison (iic/CValues-Comparison)
- RLHF Reward Single Round (beyond/rlhf-reward-single-round-trans_chinese)
- Zhihu RLHF 3k (liyucheng/zhihu_rlhf_3k)

Standard format / 标准格式:
{
    "prompt": "用户提示 / User prompt",
    "chosen": "选中的回答 / Chosen response",
    "rejected": "拒绝的回答 / Rejected response"
}
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import logging

# Configure logging / 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RMDataProcessor:
    """
    Processor for Reward Model (preference) datasets
    奖励模型（偏好）数据集处理器

    This class provides methods to process various RM datasets into a unified format.
    该类提供将各种RM数据集处理为统一格式的方法。

    Attributes:
        output_dir (Path): Directory to save processed data / 保存处理后数据的目录

    Example / 示例:
        >>> processor = RMDataProcessor(output_dir="data/rm_processed")
        >>> data = processor.process_cvalues("cvalues.jsonl")
        >>> processor.save_data(data, "rm_train.jsonl", "rm_eval.jsonl")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the RM data processor
        初始化RM数据处理器

        Args:
            output_dir: Output directory path / 输出目录路径
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def process_cvalues(file_path: Union[str, Path]) -> List[str]:
        """
        Process CValues-Comparison dataset
        处理CValues-Comparison数据集

        CValues format / CValues格式:
        {
            "prompt": "提示 / Prompt",
            "pos_resp": "正面回答 / Positive response",
            "neg_resp": "负面回答 / Negative response"
        }

        Args:
            file_path: Path to CValues JSONL file / CValues JSONL文件路径

        Returns:
            List of processed JSONL strings / 处理后的JSONL字符串列表

        Example / 示例:
            >>> data = RMDataProcessor.process_cvalues("cvalues.jsonl")
            >>> print(len(data))
            10000
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing CValues dataset from {file_path}")
        total_lines = []

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            for line in tqdm(lines, desc="Processing CValues"):
                try:
                    json_obj = json.loads(line)

                    prompt_text = json_obj.get("prompt", "")
                    chosen_text = json_obj.get("pos_resp", "")
                    rejected_text = json_obj.get("neg_resp", "")

                    # Skip empty entries / 跳过空条目
                    if not prompt_text.strip() or not chosen_text.strip() or not rejected_text.strip():
                        continue

                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }

                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n'
                    total_lines.append(processed_line)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue
                except KeyError as e:
                    logger.warning(f"Missing key in data: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing CValues dataset: {e}")
            raise

        logger.info(f"Processed {len(total_lines)} examples from CValues dataset")
        return total_lines

    @staticmethod
    def process_rlhf_parquet(file_path: Union[str, Path]) -> List[str]:
        """
        Process RLHF dataset in Parquet format
        处理Parquet格式的RLHF数据集

        RLHF Parquet format / RLHF Parquet格式:
        Columns: prompt, chosen, rejected

        Args:
            file_path: Path to RLHF Parquet file / RLHF Parquet文件路径

        Returns:
            List of processed JSONL strings / 处理后的JSONL字符串列表

        Example / 示例:
            >>> data = RMDataProcessor.process_rlhf_parquet("rlhf.parquet")
            >>> print(len(data))
            50000
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for processing Parquet files")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing RLHF Parquet dataset from {file_path}")
        total_lines = []

        try:
            df = pd.read_parquet(file_path)

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing RLHF Parquet"):
                try:
                    prompt_text = str(row.get('prompt', ''))
                    chosen_text = str(row.get('chosen', ''))
                    rejected_text = str(row.get('rejected', ''))

                    # Skip empty entries / 跳过空条目
                    if not prompt_text.strip() or not chosen_text.strip() or not rejected_text.strip():
                        continue

                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }

                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n'
                    total_lines.append(processed_line)

                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing RLHF Parquet dataset: {e}")
            raise

        logger.info(f"Processed {len(total_lines)} examples from RLHF Parquet dataset")
        return total_lines

    @staticmethod
    def process_zhihu_rlhf_tsv(file_path: Union[str, Path]) -> List[str]:
        """
        Process Zhihu RLHF dataset in TSV format
        处理TSV格式的知乎RLHF数据集

        Zhihu RLHF format / 知乎RLHF格式:
        TSV with columns: prompt, chosen, rejected

        Args:
            file_path: Path to Zhihu RLHF TSV file / 知乎RLHF TSV文件路径

        Returns:
            List of processed JSONL strings / 处理后的JSONL字符串列表

        Example / 示例:
            >>> data = RMDataProcessor.process_zhihu_rlhf_tsv("zhihu_rlhf.tsv")
            >>> print(len(data))
            3000
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for processing TSV files")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing Zhihu RLHF TSV dataset from {file_path}")
        total_lines = []

        try:
            df = pd.read_csv(file_path, sep='\t')

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Zhihu RLHF"):
                try:
                    prompt_text = str(row.get('prompt', ''))
                    chosen_text = str(row.get('chosen', ''))
                    rejected_text = str(row.get('rejected', ''))

                    # Skip empty entries / 跳过空条目
                    if not prompt_text.strip() or not chosen_text.strip() or not rejected_text.strip():
                        continue

                    data_dict = {
                        "prompt": prompt_text,
                        "chosen": chosen_text,
                        "rejected": rejected_text
                    }

                    processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n'
                    total_lines.append(processed_line)

                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing Zhihu RLHF TSV dataset: {e}")
            raise

        logger.info(f"Processed {len(total_lines)} examples from Zhihu RLHF dataset")
        return total_lines

    def process_directory(
        self,
        input_dir: Union[str, Path],
        formats: Optional[List[str]] = None
    ) -> List[str]:
        """
        Process all supported files in directory
        处理目录中所有支持的文件

        Args:
            input_dir: Input directory path / 输入目录路径
            formats: List of formats to process (default: all) / 要处理的格式列表（默认：全部）

        Returns:
            List of processed JSONL strings / 处理后的JSONL字符串列表

        Supported formats / 支持的格式:
        - .jsonl (CValues format)
        - .parquet (RLHF format)
        - .tsv (Zhihu RLHF format)

        Example / 示例:
            >>> processor = RMDataProcessor()
            >>> data = processor.process_directory("corpus/rm_train/")
            >>> print(len(data))
            60000
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        if formats is None:
            formats = ['.jsonl', '.parquet', '.tsv']

        logger.info(f"Processing directory {input_dir} for formats: {formats}")
        total_lines = []

        try:
            for subdir, dirs, files in os.walk(input_dir):
                for file in files:
                    file_path = Path(subdir) / file

                    # Process JSONL files / 处理JSONL文件
                    if file.endswith('.jsonl') and '.jsonl' in formats:
                        logger.info(f"Processing JSONL: {file_path}")
                        data = self.process_cvalues(file_path)
                        total_lines.extend(data)

                    # Process Parquet files / 处理Parquet文件
                    elif file.endswith('.parquet') and '.parquet' in formats:
                        logger.info(f"Processing Parquet: {file_path}")
                        data = self.process_rlhf_parquet(file_path)
                        total_lines.extend(data)

                    # Process TSV files / 处理TSV文件
                    elif file.endswith('.tsv') and '.tsv' in formats:
                        logger.info(f"Processing TSV: {file_path}")
                        data = self.process_zhihu_rlhf_tsv(file_path)
                        total_lines.extend(data)

        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            raise

        logger.info(f"Processed {len(total_lines)} total examples from {input_dir}")
        return total_lines

    @staticmethod
    def split_train_eval(
        data: List[str],
        eval_size: Union[int, float] = 2000,
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[List[str], List[str]]:
        """
        Split data into training and evaluation sets
        将数据拆分为训练集和评估集

        Args:
            data: List of JSONL strings / JSONL字符串列表
            eval_size: Number or fraction of eval examples / 评估样本数量或比例
            shuffle: Whether to shuffle before splitting / 是否在拆分前打乱
            random_seed: Random seed for reproducibility / 用于可重现性的随机种子

        Returns:
            Tuple of (train_data, eval_data) / (训练数据, 评估数据)元组

        Example / 示例:
            >>> train, eval = RMDataProcessor.split_train_eval(data, eval_size=0.1)
            >>> print(f"Train: {len(train)}, Eval: {len(eval)}")
            Train: 54000, Eval: 6000
        """
        if not data:
            raise ValueError("Data list is empty")

        # Set random seed / 设置随机种子
        random.seed(random_seed)

        # Calculate eval size / 计算评估集大小
        if isinstance(eval_size, float):
            if not 0 < eval_size < 1:
                raise ValueError("eval_size as float must be between 0 and 1")
            eval_count = int(len(data) * eval_size)
        else:
            eval_count = min(eval_size, len(data))

        # Shuffle if requested / 如果需要则打乱
        data_copy = data.copy()
        if shuffle:
            random.shuffle(data_copy)

        # Split data / 拆分数据
        eval_data = data_copy[:eval_count]
        train_data = data_copy[eval_count:]

        logger.info(f"Split data: {len(train_data)} train, {len(eval_data)} eval")
        return train_data, eval_data

    def save_data(
        self,
        train_data: List[str],
        eval_data: List[str],
        train_filename: str = "rm_train.jsonl",
        eval_filename: str = "rm_eval.jsonl",
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[Path, Path]:
        """
        Save training and evaluation data to files
        保存训练和评估数据到文件

        Args:
            train_data: Training data / 训练数据
            eval_data: Evaluation data / 评估数据
            train_filename: Training data filename / 训练数据文件名
            eval_filename: Evaluation data filename / 评估数据文件名
            output_dir: Output directory (overrides instance output_dir) / 输出目录（覆盖实例output_dir）

        Returns:
            Tuple of (train_path, eval_path) / (训练路径, 评估路径)元组

        Example / 示例:
            >>> processor = RMDataProcessor(output_dir="data/rm")
            >>> train_path, eval_path = processor.save_data(train_data, eval_data)
        """
        if output_dir is None:
            if self.output_dir is None:
                raise ValueError("No output directory specified")
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save training data / 保存训练数据
            train_path = output_dir / train_filename
            with open(train_path, 'w', encoding='utf-8') as outfile:
                for line in tqdm(train_data, desc="Writing train data"):
                    outfile.write(line)
            logger.info(f"Saved {len(train_data)} training examples to {train_path}")

            # Save evaluation data / 保存评估数据
            eval_path = output_dir / eval_filename
            with open(eval_path, 'w', encoding='utf-8') as outfile:
                for line in tqdm(eval_data, desc="Writing eval data"):
                    outfile.write(line)
            logger.info(f"Saved {len(eval_data)} evaluation examples to {eval_path}")

            return train_path, eval_path

        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def process_and_split(
        self,
        input_dir: Union[str, Path],
        eval_size: Union[int, float] = 2000,
        train_filename: str = "rm_train.jsonl",
        eval_filename: str = "rm_eval.jsonl",
        shuffle: bool = True,
        random_seed: int = 42
    ) -> Tuple[Path, Path]:
        """
        Process directory and split into train/eval sets
        处理目录并拆分为训练/评估集

        Args:
            input_dir: Input directory path / 输入目录路径
            eval_size: Number or fraction of eval examples / 评估样本数量或比例
            train_filename: Training data filename / 训练数据文件名
            eval_filename: Evaluation data filename / 评估数据文件名
            shuffle: Whether to shuffle before splitting / 是否在拆分前打乱
            random_seed: Random seed for reproducibility / 用于可重现性的随机种子

        Returns:
            Tuple of (train_path, eval_path) / (训练路径, 评估路径)元组

        Example / 示例:
            >>> processor = RMDataProcessor(output_dir="data/rm")
            >>> train_path, eval_path = processor.process_and_split("corpus/rm_train/")
        """
        # Process all data / 处理所有数据
        all_data = self.process_directory(input_dir)

        # Split into train/eval / 拆分为训练/评估
        train_data, eval_data = self.split_train_eval(
            all_data,
            eval_size=eval_size,
            shuffle=shuffle,
            random_seed=random_seed
        )

        # Save data / 保存数据
        train_path, eval_path = self.save_data(
            train_data,
            eval_data,
            train_filename=train_filename,
            eval_filename=eval_filename
        )

        return train_path, eval_path


# Convenience functions / 便捷函数
def process_cvalues_dataset(file_path: Union[str, Path]) -> List[str]:
    """Convenience function to process CValues dataset / 处理CValues数据集的便捷函数"""
    return RMDataProcessor.process_cvalues(file_path)


def process_rlhf_dataset(file_path: Union[str, Path]) -> List[str]:
    """Convenience function to process RLHF dataset / 处理RLHF数据集的便捷函数"""
    return RMDataProcessor.process_rlhf_parquet(file_path)


def process_zhihu_rlhf_dataset(file_path: Union[str, Path]) -> List[str]:
    """Convenience function to process Zhihu RLHF dataset / 处理知乎RLHF数据集的便捷函数"""
    return RMDataProcessor.process_zhihu_rlhf_tsv(file_path)


if __name__ == "__main__":
    # Example usage / 使用示例
    print("RM Data Processor - Example Usage")
    print("奖励模型数据处理器 - 使用示例")
    print("-" * 60)

    # Initialize processor / 初始化处理器
    processor = RMDataProcessor(output_dir="data/rm_train")

    # Example: Process individual datasets / 示例：处理单个数据集
    # cvalues_data = processor.process_cvalues("corpus/rm_train/cvalues.jsonl")
    # rlhf_data = processor.process_rlhf_parquet("corpus/rm_train/rlhf.parquet")
    # zhihu_data = processor.process_zhihu_rlhf_tsv("corpus/rm_train/zhihu_rlhf.tsv")

    # Example: Process entire directory and split / 示例：处理整个目录并拆分
    # train_path, eval_path = processor.process_and_split(
    #     input_dir="corpus/rm_train/",
    #     eval_size=2000,
    #     shuffle=True
    # )

    print("\nTo use this processor, uncomment the example code above")
    print("要使用此处理器，请取消注释上面的示例代码")
