"""
Pretrain Data Processor - Pre-training Data Processing
预训练数据处理器

This module provides utilities for processing pre-training datasets into
tokenized binary format for efficient loading during training.
本模块提供将预训练数据集处理为token化二进制格式的工具，以便在训练期间高效加载。

Supported datasets / 支持的数据集:
- Wikipedia Chinese (pleisto/wikipedia-cn-20230720-filtered)
- TigerBot Pretrain (TigerResearch/pretrain_zh)
- Baidu Baike (xuqinyang/BaiduBaike-5.63M)
- Zhihu KOL (wangrui6/Zhihu-KOL)
- Web novels and other text corpora

Output format / 输出格式:
- Binary files (.bin) with uint16 token IDs / 包含uint16 token ID的二进制文件
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import numpy as np
from tqdm import tqdm
import logging

# Configure logging / 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PretrainDataProcessor:
    """
    Processor for pre-training datasets
    预训练数据集处理器

    This class tokenizes text data and saves it as binary files for efficient loading.
    该类将文本数据token化并保存为二进制文件以供高效加载。

    Attributes:
        tokenizer: Tokenizer object with encode() method / 具有encode()方法的分词器对象
        output_dir (Path): Directory to save processed binary files / 保存处理后二进制文件的目录
        batch_size (int): Number of examples to process before saving / 保存前处理的样本数量

    Example / 示例:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
        >>> processor = PretrainDataProcessor(tokenizer, output_dir="data/pretrain")
        >>> processor.process_wikipedia("wiki_cn.json")
    """

    def __init__(
        self,
        tokenizer: Any,
        output_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 1000000
    ):
        """
        Initialize the pretrain data processor
        初始化预训练数据处理器

        Args:
            tokenizer: Tokenizer with encode() method / 具有encode()方法的分词器
            output_dir: Output directory path / 输出目录路径
            batch_size: Batch size for processing / 处理的批大小
        """
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        # Verify tokenizer has required methods / 验证分词器具有所需方法
        if not hasattr(tokenizer, 'encode'):
            raise ValueError("Tokenizer must have an 'encode' method")

    def _get_eos_token(self) -> int:
        """
        Get EOS token ID from tokenizer
        从分词器获取EOS token ID

        Returns:
            EOS token ID / EOS token ID
        """
        if hasattr(self.tokenizer, 'special_tokens') and '<eos>' in self.tokenizer.special_tokens:
            return self.tokenizer.special_tokens['<eos>']
        elif hasattr(self.tokenizer, 'eos_token_id'):
            return self.tokenizer.eos_token_id
        else:
            logger.warning("Could not find EOS token, using default value 2")
            return 2

    def _tokenize_text(self, text: str, min_length: int = 5) -> Optional[List[int]]:
        """
        Tokenize text and add EOS token
        对文本进行token化并添加EOS token

        Args:
            text: Text to tokenize / 要token化的文本
            min_length: Minimum token length to keep / 保留的最小token长度

        Returns:
            List of token IDs or None if too short / token ID列表，如果太短则返回None
        """
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            tokens.append(self._get_eos_token())

            if len(tokens) > min_length:
                return tokens
            return None

        except Exception as e:
            logger.warning(f"Error tokenizing text: {e}")
            return None

    def _save_tokens(
        self,
        tokens: List[int],
        output_filename: str,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save tokens to binary file
        保存tokens到二进制文件

        Args:
            tokens: List of token IDs / token ID列表
            output_filename: Output filename / 输出文件名
            output_dir: Output directory / 输出目录

        Returns:
            Path to saved file / 保存文件的路径
        """
        if output_dir is None:
            output_dir = self.output_dir

        output_path = output_dir / output_filename

        try:
            arr = np.array(tokens, dtype=np.uint16)
            with open(output_path, 'wb') as f:
                f.write(arr.tobytes())

            logger.info(f"Saved {len(tokens)} tokens to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving tokens to {output_path}: {e}")
            raise

    def process_wikipedia(
        self,
        file_path: Union[str, Path],
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Process Wikipedia Chinese dataset
        处理中文维基百科数据集

        Wikipedia format / 维基百科格式:
        [
            {
                "completion": "文章内容 / Article content",
                ...
            }
        ]

        Args:
            file_path: Path to Wikipedia JSON file / 维基百科JSON文件路径
            output_filename: Output filename (auto-generated if None) / 输出文件名（如果为None则自动生成）

        Returns:
            Path to output binary file / 输出二进制文件路径

        Example / 示例:
            >>> processor.process_wikipedia("wiki_cn.json")
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing Wikipedia dataset from {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            all_tokens = []
            for item in tqdm(data, desc="Processing Wikipedia"):
                text = item.get('completion', '')
                tokens = self._tokenize_text(text)
                if tokens:
                    all_tokens.extend(tokens)

            # Generate output filename / 生成输出文件名
            if output_filename is None:
                output_filename = file_path.stem + '.bin'

            output_path = self._save_tokens(all_tokens, output_filename)
            logger.info(f"Wikipedia processing complete: {len(all_tokens)} total tokens")
            return output_path

        except Exception as e:
            logger.error(f"Error processing Wikipedia dataset: {e}")
            raise

    def process_jsonl(
        self,
        file_path: Union[str, Path],
        text_field: str = 'text',
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Process JSONL format dataset
        处理JSONL格式数据集

        Args:
            file_path: Path to JSONL file / JSONL文件路径
            text_field: Field name containing text / 包含文本的字段名
            output_filename: Output filename (auto-generated if None) / 输出文件名（如果为None则自动生成）

        Returns:
            Path to output binary file / 输出二进制文件路径

        Example / 示例:
            >>> processor.process_jsonl("webnovel.jsonl", text_field="content")
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing JSONL dataset from {file_path}")

        try:
            all_tokens = []
            with open(file_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

            for line in tqdm(lines, desc=f"Processing {file_path.name}"):
                try:
                    json_obj = json.loads(line)
                    text = json_obj.get(text_field, '')
                    tokens = self._tokenize_text(text)
                    if tokens:
                        all_tokens.extend(tokens)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {e}")
                    continue

            # Generate output filename / 生成输出文件名
            if output_filename is None:
                output_filename = file_path.stem + '.bin'

            output_path = self._save_tokens(all_tokens, output_filename)
            logger.info(f"JSONL processing complete: {len(all_tokens)} total tokens")
            return output_path

        except Exception as e:
            logger.error(f"Error processing JSONL dataset: {e}")
            raise

    def process_directory(
        self,
        input_dir: Union[str, Path],
        pattern: str = "*.jsonl",
        text_field: str = 'text'
    ) -> List[Path]:
        """
        Process all files in directory matching pattern
        处理目录中所有匹配模式的文件

        Args:
            input_dir: Input directory path / 输入目录路径
            pattern: File pattern to match / 要匹配的文件模式
            text_field: Field name containing text / 包含文本的字段名

        Returns:
            List of output file paths / 输出文件路径列表

        Example / 示例:
            >>> processor.process_directory("webnovels/", pattern="*.jsonl")
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        logger.info(f"Processing directory {input_dir} with pattern {pattern}")

        output_paths = []
        files = list(input_dir.rglob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern {pattern} in {input_dir}")
            return output_paths

        for file_path in files:
            logger.info(f"Processing file: {file_path}")
            try:
                output_path = self.process_jsonl(file_path, text_field=text_field)
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        logger.info(f"Processed {len(output_paths)} files from {input_dir}")
        return output_paths

    def process_parquet(
        self,
        file_path: Union[str, Path],
        text_field: str = 'content',
        output_filename: Optional[str] = None,
        max_tokens_per_file: Optional[int] = None
    ) -> Union[Path, List[Path]]:
        """
        Process Parquet format dataset
        处理Parquet格式数据集

        Args:
            file_path: Path to Parquet file / Parquet文件路径
            text_field: Field name containing text / 包含文本的字段名
            output_filename: Output filename / 输出文件名
            max_tokens_per_file: Split output if exceeds this many tokens / 如果超过此token数则拆分输出

        Returns:
            Path(s) to output binary file(s) / 输出二进制文件路径

        Example / 示例:
            >>> processor.process_parquet("dataset.parquet", text_field="RESPONSE")
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for processing Parquet files")

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Processing Parquet dataset from {file_path}")

        try:
            df = pd.read_parquet(file_path)
            texts = df[text_field]

            all_tokens = []
            for text in tqdm(texts, desc=f"Processing {file_path.name}"):
                tokens = self._tokenize_text(str(text))
                if tokens:
                    all_tokens.extend(tokens)

            # Generate output filename / 生成输出文件名
            if output_filename is None:
                output_filename = file_path.stem + '.bin'

            # Split into multiple files if needed / 如果需要则拆分为多个文件
            if max_tokens_per_file and len(all_tokens) > max_tokens_per_file:
                output_paths = []
                num_files = (len(all_tokens) + max_tokens_per_file - 1) // max_tokens_per_file

                for i in range(num_files):
                    start_idx = i * max_tokens_per_file
                    end_idx = min((i + 1) * max_tokens_per_file, len(all_tokens))
                    tokens_chunk = all_tokens[start_idx:end_idx]

                    chunk_filename = f"{Path(output_filename).stem}_{i}{Path(output_filename).suffix}"
                    output_path = self._save_tokens(tokens_chunk, chunk_filename)
                    output_paths.append(output_path)

                logger.info(f"Parquet processing complete: {len(all_tokens)} total tokens in {len(output_paths)} files")
                return output_paths
            else:
                output_path = self._save_tokens(all_tokens, output_filename)
                logger.info(f"Parquet processing complete: {len(all_tokens)} total tokens")
                return output_path

        except Exception as e:
            logger.error(f"Error processing Parquet dataset: {e}")
            raise

    def process_parquet_directory(
        self,
        input_dir: Union[str, Path],
        text_field: str = 'content',
        output_prefix: str = "pretrain_part",
        max_tokens_per_file: int = 10000000
    ) -> List[Path]:
        """
        Process all Parquet files in directory
        处理目录中所有Parquet文件

        Args:
            input_dir: Input directory path / 输入目录路径
            text_field: Field name containing text / 包含文本的字段名
            output_prefix: Prefix for output filenames / 输出文件名前缀
            max_tokens_per_file: Maximum tokens per output file / 每个输出文件的最大token数

        Returns:
            List of output file paths / 输出文件路径列表

        Example / 示例:
            >>> processor.process_parquet_directory("tigerbot/", text_field="content")
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for processing Parquet files")

        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Directory not found: {input_dir}")

        logger.info(f"Processing Parquet directory {input_dir}")

        all_tokens = []
        output_paths = []
        file_idx = 0

        parquet_files = list(input_dir.glob("*.parquet"))

        if not parquet_files:
            logger.warning(f"No Parquet files found in {input_dir}")
            return output_paths

        for parquet_file in parquet_files:
            logger.info(f"Processing file: {parquet_file}")

            try:
                df = pd.read_parquet(parquet_file)
                texts = df[text_field]

                for text in tqdm(texts, desc=f"Processing {parquet_file.name}"):
                    tokens = self._tokenize_text(str(text))
                    if tokens:
                        all_tokens.extend(tokens)

                    # Save batch if reached limit / 如果达到限制则保存批次
                    if len(all_tokens) >= max_tokens_per_file:
                        output_filename = f"{output_prefix}_{file_idx}.bin"
                        output_path = self._save_tokens(all_tokens, output_filename)
                        output_paths.append(output_path)

                        all_tokens = []
                        file_idx += 1

            except Exception as e:
                logger.error(f"Error processing {parquet_file}: {e}")
                continue

        # Save remaining tokens / 保存剩余的tokens
        if all_tokens:
            output_filename = f"{output_prefix}_{file_idx}.bin"
            output_path = self._save_tokens(all_tokens, output_filename)
            output_paths.append(output_path)

        logger.info(f"Processed {len(parquet_files)} Parquet files into {len(output_paths)} binary files")
        return output_paths

    def process_baidu_baike(
        self,
        file_path: Union[str, Path],
        output_prefix: str = "baidubaike",
        batch_size: Optional[int] = None
    ) -> List[Path]:
        """
        Process Baidu Baike dataset
        处理百度百科数据集

        Baidu Baike format / 百度百科格式:
        {
            "title": "标题 / Title",
            "summary": "摘要 / Summary",
            "sections": [
                {
                    "title": "章节标题 / Section title",
                    "content": "章节内容 / Section content"
                }
            ]
        }

        Args:
            file_path: Path to Baidu Baike JSONL file / 百度百科JSONL文件路径
            output_prefix: Prefix for output filenames / 输出文件名前缀
            batch_size: Number of examples per batch / 每批次样本数

        Returns:
            List of output file paths / 输出文件路径列表

        Example / 示例:
            >>> processor.process_baidu_baike("baidubaike.jsonl")
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if batch_size is None:
            batch_size = self.batch_size

        logger.info(f"Processing Baidu Baike dataset from {file_path}")

        output_paths = []
        doc_ids = []
        cnt = 0
        batch_cnt = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f1:
                for line in tqdm(f1, desc="Processing Baidu Baike"):
                    try:
                        data = json.loads(line)
                        text = ''

                        # Process title and summary / 处理标题和摘要
                        if 'title' in data and 'summary' in data:
                            text += data['title'] + '：' + data['summary']

                        # Process sections / 处理章节
                        if 'sections' in data:
                            for section in data['sections']:
                                text += section.get('title', '') + '：' + section.get('content', '') + '。'

                        tokens = self._tokenize_text(text)
                        if tokens:
                            doc_ids.extend(tokens)

                        cnt += 1

                        # Save batch / 保存批次
                        if cnt % batch_size == 0:
                            batch_cnt += 1
                            output_filename = f"{output_prefix}_{batch_cnt}.bin"
                            output_path = self._save_tokens(doc_ids, output_filename)
                            output_paths.append(output_path)
                            doc_ids = []

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing entry: {e}")
                        continue

            # Save remaining / 保存剩余
            if doc_ids:
                batch_cnt += 1
                output_filename = f"{output_prefix}_{batch_cnt}.bin"
                output_path = self._save_tokens(doc_ids, output_filename)
                output_paths.append(output_path)

            logger.info(f"Processed {cnt} Baidu Baike entries into {len(output_paths)} files")
            return output_paths

        except Exception as e:
            logger.error(f"Error processing Baidu Baike dataset: {e}")
            raise


def merge_binary_files(
    file_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    dtype: type = np.uint16
) -> Path:
    """
    Merge multiple binary files into one
    合并多个二进制文件为一个

    Args:
        file_paths: List of binary file paths to merge / 要合并的二进制文件路径列表
        output_path: Output file path / 输出文件路径
        dtype: Data type of tokens / token的数据类型

    Returns:
        Path to merged file / 合并文件的路径

    Example / 示例:
        >>> files = ["part_0.bin", "part_1.bin", "part_2.bin"]
        >>> merge_binary_files(files, "merged.bin")
    """
    logger.info(f"Merging {len(file_paths)} binary files")

    try:
        data_arr = []
        for file_path in tqdm(file_paths, desc="Loading files"):
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}, skipping")
                continue

            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=dtype)
                data_arr.append(data)

        # Concatenate all arrays / 连接所有数组
        merged_arr = np.concatenate(data_arr)
        logger.info(f"Merged array shape: {merged_arr.shape}")

        # Save merged file / 保存合并文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(merged_arr.tobytes())

        logger.info(f"Saved merged file to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error merging binary files: {e}")
        raise


# Convenience functions / 便捷函数
def process_wikipedia_dataset(
    tokenizer: Any,
    file_path: Union[str, Path],
    output_dir: Union[str, Path]
) -> Path:
    """Convenience function to process Wikipedia dataset / 处理维基百科数据集的便捷函数"""
    processor = PretrainDataProcessor(tokenizer, output_dir=output_dir)
    return processor.process_wikipedia(file_path)


def process_tigerbot_pretrain_dataset(
    tokenizer: Any,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path]
) -> List[Path]:
    """Convenience function to process TigerBot pretrain dataset / 处理TigerBot预训练数据集的便捷函数"""
    processor = PretrainDataProcessor(tokenizer, output_dir=output_dir)
    return processor.process_parquet_directory(input_dir, text_field='content')


def process_baidu_baike_dataset(
    tokenizer: Any,
    file_path: Union[str, Path],
    output_dir: Union[str, Path]
) -> List[Path]:
    """Convenience function to process Baidu Baike dataset / 处理百度百科数据集的便捷函数"""
    processor = PretrainDataProcessor(tokenizer, output_dir=output_dir)
    return processor.process_baidu_baike(file_path)


if __name__ == "__main__":
    # Example usage / 使用示例
    print("Pretrain Data Processor - Example Usage")
    print("预训练数据处理器 - 使用示例")
    print("-" * 60)

    # Note: You need to provide a tokenizer / 注意：您需要提供一个分词器
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')

    # processor = PretrainDataProcessor(tokenizer, output_dir="data/pretrain")

    # Example: Process Wikipedia / 示例：处理维基百科
    # processor.process_wikipedia("corpus/wiki_cn.json")

    # Example: Process JSONL files / 示例：处理JSONL文件
    # processor.process_jsonl("corpus/webnovel.jsonl", text_field="text")

    # Example: Process Parquet files / 示例：处理Parquet文件
    # processor.process_parquet_directory("corpus/tigerbot/", text_field="content")

    # Example: Process Baidu Baike / 示例：处理百度百科
    # processor.process_baidu_baike("corpus/baidubaike.jsonl")

    # Example: Merge binary files / 示例：合并二进制文件
    # files = ["part_0.bin", "part_1.bin", "part_2.bin"]
    # merge_binary_files(files, "merged.bin")

    print("\nTo use this processor, uncomment the example code above")
    print("要使用此处理器，请取消注释上面的示例代码")
