"""
Streaming utilities for real-time text generation output.
用于实时文本生成输出的流式处理工具。
"""

import torch
from queue import Queue
from typing import Optional, Iterator
from .utils import parse_pot_no_stream


class TextIterStreamer:
    """
    Iterator-based streamer for real-time text generation output.
    基于迭代器的流式文本生成输出器。

    This class enables streaming of generated text token-by-token or chunk-by-chunk,
    allowing for real-time display of model outputs rather than waiting for complete
    generation. It's particularly useful for chat applications and interactive demos.

    此类支持逐token或逐块流式传输生成的文本，允许实时显示模型输出而不是等待
    完整生成。它特别适用于聊天应用程序和交互式演示。

    Features / 特性:
    - Real-time streaming of generated tokens / 生成token的实时流式传输
    - Optional prompt skipping / 可选的提示跳过
    - Special token filtering / 特殊token过滤
    - Program-of-Thought (PoT) post-processing / 程序思维(PoT)后处理
    - Queue-based buffering for smooth iteration / 基于队列的缓冲以实现流畅迭代

    Args:
        tokenizer: HuggingFace tokenizer for decoding tokens.
                  用于解码token的HuggingFace tokenizer。
        skip_prompt (bool, optional): If True, skip outputting the input prompt tokens.
                                     如果为True，则跳过输出输入提示token。
                                     Defaults to False.
                                     默认为False。
        skip_special_tokens (bool, optional): If True, remove special tokens from output.
                                             如果为True，则从输出中删除特殊token。
                                             Defaults to False.
                                             默认为False。
        use_pot (bool, optional): If True, apply Program-of-Thought post-processing.
                                 如果为True，则应用程序思维后处理。
                                 Defaults to True.
                                 默认为True。

    Attributes:
        tokens (list): Accumulated list of generated token IDs.
                      累积的生成token ID列表。
        text_queue (Queue): Queue for buffering decoded text chunks.
                           用于缓冲解码文本块的队列。
        next_tokens_are_prompt (bool): Flag to track if next tokens are from prompt.
                                      标记下一个token是否来自提示的标志。

    Example:
        Basic usage with model generation:
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM
        >>> tokenizer = AutoTokenizer.from_pretrained("model_name")
        >>> model = AutoModelForCausalLM.from_pretrained("model_name")
        >>>
        >>> # Create streamer
        >>> streamer = TextIterStreamer(
        ...     tokenizer,
        ...     skip_prompt=True,
        ...     skip_special_tokens=True
        ... )
        >>>
        >>> # Generate with streaming
        >>> input_ids = tokenizer("Hello, how are", return_tensors="pt").input_ids
        >>> generation_kwargs = dict(
        ...     input_ids=input_ids,
        ...     max_new_tokens=50,
        ...     streamer=streamer
        ... )
        >>>
        >>> # Start generation in separate thread
        >>> import threading
        >>> thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        >>> thread.start()
        >>>
        >>> # Stream output in real-time
        >>> for text in streamer:
        ...     print(text, end="", flush=True)
        >>> thread.join()

    Example:
        With math reasoning (PoT enabled):
        >>> streamer = TextIterStreamer(tokenizer, skip_prompt=True, use_pot=True)
        >>> # If model generates: "The answer is <<x = 5 + 3>>, so x equals 8"
        >>> # Streamer will output: "The answer is , so 8 equals 8"
        >>> for text in streamer:
        ...     print(text)

    Notes:
        - This streamer is designed to work with HuggingFace transformers' generate() method
          此流式处理器设计用于HuggingFace transformers的generate()方法
        - The generate() call should be run in a separate thread for true streaming
          generate()调用应在单独的线程中运行以实现真正的流式传输
        - Call end() when generation is complete to signal iteration termination
          生成完成时调用end()以发出迭代终止信号
        - The streamer maintains state between calls, create new instance for each generation
          流式处理器在调用之间保持状态，为每次生成创建新实例

    Thread Safety:
        This class uses Queue which is thread-safe for communication between
        generation thread and output iteration thread.
        此类使用Queue，它对于生成线程和输出迭代线程之间的通信是线程安全的。
    """

    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        skip_special_tokens: bool = False,
        use_pot: bool = True,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.use_pot = use_pot

        # Storage for accumulated tokens
        # 用于存储累积token的列表
        self.tokens = []

        # Queue for buffering text chunks
        # 使用队列来缓存生成的文本片段，以便于逐块输出
        self.text_queue = Queue()

        # Flag to track prompt tokens
        # 用于跟踪提示token的标志
        self.next_tokens_are_prompt = True

    def put(self, value: torch.Tensor) -> None:
        """
        Add new tokens to the stream and decode them.
        向流中添加新token并解码它们。

        This method is called by the model's generation function for each new
        batch of tokens. It accumulates tokens and decodes them into text.

        此方法由模型的生成函数为每批新token调用。它累积token并将其解码为文本。

        Args:
            value: Tensor of token IDs, shape [batch_size, seq_len] or [seq_len].
                  token ID张量，形状为[batch_size, seq_len]或[seq_len]。

        Notes:
            - If skip_prompt is True, the first call (prompt tokens) is ignored
              如果skip_prompt为True，则忽略第一次调用（提示token）
            - Tokens are accumulated for proper decoding of multi-token sequences
              累积token以正确解码多token序列
            - Decoded text is post-processed with PoT if use_pot is True
              如果use_pot为True，则使用PoT对解码文本进行后处理
        """
        # Skip prompt tokens if requested
        # 如果请求跳过提示，则跳过提示token
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Handle batch dimension
        # 处理批次维度
        if len(value.shape) > 1:
            value = value[0]

        # Accumulate tokens
        # 累积token
        self.tokens.extend(value.tolist())

        # Decode accumulated tokens to text
        # 将累积的token解码为文本
        tokens_str = self.tokenizer.decode(
            self.tokens,
            skip_special_tokens=self.skip_special_tokens,
            errors='ignore'
        )

        # Apply Program-of-Thought post-processing if enabled
        # 如果启用，则应用程序思维后处理
        if self.use_pot:
            tokens_str = parse_pot_no_stream(tokens_str)

        # Add to output queue
        # 添加到输出队列
        self.text_queue.put(tokens_str)

    def end(self) -> None:
        """
        Signal end of generation by putting None in queue.
        通过在队列中放入None来发出生成结束的信号。

        This method should be called when generation is complete to signal
        the iterator to stop iteration.

        此方法应在生成完成时调用，以通知迭代器停止迭代。

        Note:
            The model's generate() method typically calls this automatically
            if the streamer is passed as a parameter.
            如果将流式处理器作为参数传递，则模型的generate()方法通常会自动调用此方法。
        """
        self.text_queue.put(None)

    def __iter__(self) -> Iterator[str]:
        """
        Return iterator object (self).
        返回迭代器对象（self）。

        Returns:
            Iterator[str]: The streamer instance itself.
                          流式处理器实例本身。
        """
        return self

    def __next__(self) -> str:
        """
        Get next text chunk from the stream.
        从流中获取下一个文本块。

        Returns:
            str: Next decoded text chunk.
                下一个解码的文本块。

        Raises:
            StopIteration: When generation is complete (None received from queue).
                          当生成完成时（从队列接收到None）。

        Notes:
            This method blocks until new text is available in the queue.
            此方法会阻塞，直到队列中有新文本可用。
        """
        # Get next value from queue (blocking)
        # 从队列中获取并返回文本，或在无更多内容时抛出StopIteration异常
        value = self.text_queue.get()

        # None signals end of generation
        # None表示生成结束
        if value is None:
            raise StopIteration()

        return value

    def reset(self) -> None:
        """
        Reset the streamer state for reuse.
        重置流式处理器状态以便重用。

        Clears accumulated tokens and queue. Use this if you want to reuse
        the same streamer instance for multiple generations (not recommended).

        清除累积的token和队列。如果要为多个生成重用同一个流式处理器实例，
        请使用此方法（不推荐）。

        Note:
            It's generally better to create a new streamer instance for each
            generation rather than reusing.
            通常最好为每次生成创建新的流式处理器实例而不是重用。
        """
        self.tokens = []
        # Clear queue
        # 清空队列
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except:
                break
        self.next_tokens_are_prompt = True
