"""
SFT Data Processor - Supervised Fine-Tuning Data Processing
SFT数据处理器 - 监督微调数据处理

This module provides utilities for processing supervised fine-tuning datasets
from various sources into a standardized format.
本模块提供从各种来源处理监督微调数据集为标准格式的工具。

Supported datasets / 支持的数据集:
- Belle (BelleGroup/train_2M_CN)
- Firefly (YeungNLP/firefly-train-1.1M)
- TigerBot (TigerResearch/sft_zh)

Standard format / 标准格式:
{
    "question": "用户问题 / User question",
    "answer": "模型回答 / Model answer"
}
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from tqdm import tqdm
import logging

# Configure logging / 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SFTDataProcessor:
    """
    Processor for Supervised Fine-Tuning datasets
    监督微调数据集处理器

    This class provides methods to process various SFT datasets into a unified format.
    该类提供将各种SFT数据集处理为统一格式的方法。

    Attributes:
        output_dir (Path): Directory to save processed data / 保存处理后数据的目录

    Example / 示例:
        >>> processor = SFTDataProcessor(output_dir="data/processed")
        >>> data = processor.process_belle("data/raw/belle.json")
        >>> processor.save_data(data, "belle_processed.jsonl")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the SFT data processor
        初始化SFT数据处理器

        Args:
            output_dir: Output directory path / 输出目录路径
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def process_belle(file_path: Union[str, Path]) -> List[str]:
        """
        Process Belle dataset
        处理Belle数据集

        Belle format / Belle格式:
        {
            "instruction": "指令 / Instruction",
            "input": "输入 / Input",
            "output": "输出 / Output"
        }

        Args:
            file_path: Path to Belle JSONL file / Belle JSONL文件路径

        Returns:
            List of processed JSONL strings / 处理后的JSONL字符串列表

        Example / 示例:
            >>> data = SFTDataProcessor.process_belle("belle.json")
            >>> print(len(data))
            2000000
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing Belle dataset from {file_path}")
        total_lines = []

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            for line in tqdm(lines, desc="Processing Belle"):
                try:
                    json_obj = json.loads(line)

                    instruction = json_obj.get("instruction", "")
                    input_str = json_obj.get("input", "")
                    answer = json_obj.get("output", "")

                    # Combine instruction and input as question
                    # 将指令和输入组合为问题
                    question = instruction + input_str

                    # Skip empty entries / 跳过空条目
                    if not question.strip() or not answer.strip():
                        continue

                    data_dict = {
                        "question": question,
                        "answer": answer
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
            logger.error(f"Error processing Belle dataset: {e}")
            raise

        logger.info(f"Processed {len(total_lines)} examples from Belle dataset")
        return total_lines

    @staticmethod
    def process_firefly(file_path: Union[str, Path]) -> List[str]:
        """
        Process Firefly dataset
        处理Firefly数据集

        Firefly format / Firefly格式:
        {
            "input": "输入问题 / Input question",
            "target": "目标回答 / Target answer"
        }

        Args:
            file_path: Path to Firefly JSONL file / Firefly JSONL文件路径

        Returns:
            List of processed JSONL strings / 处理后的JSONL字符串列表

        Example / 示例:
            >>> data = SFTDataProcessor.process_firefly("firefly.jsonl")
            >>> print(len(data))
            1100000
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing Firefly dataset from {file_path}")
        total_lines = []

        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            for line in tqdm(lines, desc="Processing Firefly"):
                try:
                    json_obj = json.loads(line)

                    question = json_obj.get("input", "")
                    answer = json_obj.get("target", "")

                    # Skip empty entries / 跳过空条目
                    if not question.strip() or not answer.strip():
                        continue

                    data_dict = {
                        "question": question,
                        "answer": answer
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
            logger.error(f"Error processing Firefly dataset: {e}")
            raise

        logger.info(f"Processed {len(total_lines)} examples from Firefly dataset")
        return total_lines

    @staticmethod
    def process_tigerbot_sft(input_dir: Union[str, Path]) -> List[str]:
        """
        Process TigerBot SFT dataset
        处理TigerBot SFT数据集

        TigerBot format / TigerBot格式:
        {
            "instruction": "指令 / Instruction",
            "input": "输入 / Input",
            "output": "输出 / Output"
        }

        Args:
            input_dir: Directory containing TigerBot JSON files / 包含TigerBot JSON文件的目录

        Returns:
            List of processed JSONL strings / 处理后的JSONL字符串列表

        Example / 示例:
            >>> data = SFTDataProcessor.process_tigerbot_sft("tigerbot/")
            >>> print(len(data))
            500000
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        logger.info(f"Processing TigerBot SFT dataset from {input_dir}")
        total_lines = []

        try:
            # Recursively find all JSON files / 递归查找所有JSON文件
            json_files = list(input_dir.rglob("*.json"))

            if not json_files:
                logger.warning(f"No JSON files found in {input_dir}")
                return total_lines

            for file_path in json_files:
                logger.info(f"Processing file: {file_path}")

                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        lines = infile.readlines()

                    for line in tqdm(lines, desc=f"Processing {file_path.name}"):
                        try:
                            json_obj = json.loads(line)

                            instruction = json_obj.get("instruction", "")
                            input_str = json_obj.get("input", "")
                            answer = json_obj.get("output", "")

                            # Combine instruction and input as question
                            # 将指令和输入组合为问题
                            question = instruction + input_str

                            # Skip empty entries / 跳过空条目
                            if not question.strip() or not answer.strip():
                                continue

                            data_dict = {
                                "question": question,
                                "answer": answer
                            }

                            processed_line = json.dumps(data_dict, ensure_ascii=False) + '\n'
                            total_lines.append(processed_line)

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line in {file_path}: {e}")
                            continue
                        except KeyError as e:
                            logger.warning(f"Missing key in {file_path}: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error processing TigerBot dataset: {e}")
            raise

        logger.info(f"Processed {len(total_lines)} examples from TigerBot SFT dataset")
        return total_lines

    def save_data(
        self,
        data: List[str],
        output_filename: str,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save processed data to file
        保存处理后的数据到文件

        Args:
            data: List of JSONL strings to save / 要保存的JSONL字符串列表
            output_filename: Output filename / 输出文件名
            output_dir: Output directory (overrides instance output_dir) / 输出目录（覆盖实例output_dir）

        Returns:
            Path to saved file / 保存文件的路径

        Example / 示例:
            >>> processor = SFTDataProcessor(output_dir="data/processed")
            >>> path = processor.save_data(data, "processed.jsonl")
            >>> print(f"Saved to {path}")
        """
        if output_dir is None:
            if self.output_dir is None:
                raise ValueError("No output directory specified")
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as outfile:
                for line in tqdm(data, desc="Writing data"):
                    outfile.write(line)

            logger.info(f"Saved {len(data)} examples to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving data to {output_path}: {e}")
            raise

    def process_and_merge(
        self,
        belle_path: Optional[Union[str, Path]] = None,
        firefly_path: Optional[Union[str, Path]] = None,
        tigerbot_dir: Optional[Union[str, Path]] = None,
        output_filename: str = "sft_merged.jsonl"
    ) -> List[str]:
        """
        Process and merge multiple SFT datasets
        处理并合并多个SFT数据集

        Args:
            belle_path: Path to Belle dataset / Belle数据集路径
            firefly_path: Path to Firefly dataset / Firefly数据集路径
            tigerbot_dir: Directory containing TigerBot dataset / 包含TigerBot数据集的目录
            output_filename: Output filename / 输出文件名

        Returns:
            Merged data list / 合并后的数据列表

        Example / 示例:
            >>> processor = SFTDataProcessor(output_dir="data/processed")
            >>> data = processor.process_and_merge(
            ...     belle_path="belle.json",
            ...     firefly_path="firefly.jsonl",
            ...     tigerbot_dir="tigerbot/"
            ... )
            >>> print(f"Total: {len(data)} examples")
        """
        all_data = []

        if belle_path:
            logger.info("Processing Belle dataset...")
            belle_data = self.process_belle(belle_path)
            all_data.extend(belle_data)
            logger.info(f"Belle: {len(belle_data)} examples")

        if firefly_path:
            logger.info("Processing Firefly dataset...")
            firefly_data = self.process_firefly(firefly_path)
            all_data.extend(firefly_data)
            logger.info(f"Firefly: {len(firefly_data)} examples")

        if tigerbot_dir:
            logger.info("Processing TigerBot dataset...")
            tigerbot_data = self.process_tigerbot_sft(tigerbot_dir)
            all_data.extend(tigerbot_data)
            logger.info(f"TigerBot: {len(tigerbot_data)} examples")

        logger.info(f"Total merged: {len(all_data)} examples")

        if self.output_dir and output_filename:
            self.save_data(all_data, output_filename)

        return all_data


# Convenience functions / 便捷函数
def process_belle_dataset(file_path: Union[str, Path]) -> List[str]:
    """Convenience function to process Belle dataset / 处理Belle数据集的便捷函数"""
    return SFTDataProcessor.process_belle(file_path)


def process_firefly_dataset(file_path: Union[str, Path]) -> List[str]:
    """Convenience function to process Firefly dataset / 处理Firefly数据集的便捷函数"""
    return SFTDataProcessor.process_firefly(file_path)


def process_tigerbot_sft_dataset(input_dir: Union[str, Path]) -> List[str]:
    """Convenience function to process TigerBot SFT dataset / 处理TigerBot SFT数据集的便捷函数"""
    return SFTDataProcessor.process_tigerbot_sft(input_dir)


if __name__ == "__main__":
    # Example usage / 使用示例
    print("SFT Data Processor - Example Usage")
    print("SFT数据处理器 - 使用示例")
    print("-" * 60)

    # Initialize processor / 初始化处理器
    processor = SFTDataProcessor(output_dir="data/sft_processed")

    # Example: Process individual datasets / 示例：处理单个数据集
    # belle_data = processor.process_belle("corpus/sft_train/belle_2m/train_2M_CN.json")
    # firefly_data = processor.process_firefly("corpus/sft_train/firefly/firefly-train-1.1M.jsonl")
    # tigerbot_data = processor.process_tigerbot_sft("corpus/sft_train/tigerbot")

    # Example: Process and merge all datasets / 示例：处理并合并所有数据集
    # merged_data = processor.process_and_merge(
    #     belle_path="corpus/sft_train/belle_2m/train_2M_CN.json",
    #     firefly_path="corpus/sft_train/firefly/firefly-train-1.1M.jsonl",
    #     tigerbot_dir="corpus/sft_train/tigerbot",
    #     output_filename="sft_merged.jsonl"
    # )

    print("\nTo use this processor, uncomment the example code above")
    print("要使用此处理器，请取消注释上面的示例代码")
