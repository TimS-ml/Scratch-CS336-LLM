"""ChatML rendering (Qwen format) and assistant-only loss masks."""

from __future__ import annotations

from scratch_cs336.tokenizer import Tokenizer

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

Message = dict[str, str]  # {"role": ..., "content": ...}


def _header(role: str) -> str:
    return f"{IM_START}{role}\n"


def render_chat(messages: list[Message], add_generation_prompt: bool = False) -> str:
    """``<|im_start|>{role}\\n{content}<|im_end|>\\n`` per message, plus an open assistant turn when requested."""
    text = "".join(f"{_header(m['role'])}{m['content']}{IM_END}\n" for m in messages)
    return text + _header("assistant") if add_generation_prompt else text


def tokenize_chat(messages: list[Message], tokenizer: Tokenizer) -> tuple[list[int], list[int]]:
    """Token ids of ``render_chat(messages)`` and a 0/1 loss mask: 1 on assistant content and its ``<|im_end|>``.

    Messages start and end at special tokens, so encoding them one by one equals encoding the whole rendering.
    If assistant content starts with a character that merges with the header's trailing newline, the merged token
    counts as content.
    """
    im_end = tokenizer.encode(IM_END)
    if len(im_end) != 1:
        raise ValueError(f"{IM_END} must be a single token in this tokenizer, got {im_end}")
    trailer = tokenizer.encode("\n")  # text between this message's <|im_end|> and the next special
    ids: list[int] = []
    mask: list[int] = []
    for message in messages:
        message_ids = tokenizer.encode(f"{_header(message['role'])}{message['content']}{IM_END}\n")
        trained = [0] * len(message_ids)
        if message["role"] == "assistant":
            header_ids = tokenizer.encode(_header("assistant"))
            content_start = 0
            while content_start < len(header_ids) and message_ids[content_start] == header_ids[content_start]:
                content_start += 1
            for i in range(content_start, len(message_ids) - len(trailer)):
                trained[i] = 1
        ids.extend(message_ids)
        mask.extend(trained)
    return ids, mask
