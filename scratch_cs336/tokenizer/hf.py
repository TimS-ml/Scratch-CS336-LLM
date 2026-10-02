"""`tokenizers`-backed tokenizer (Qwen, GPT-2, ...) exposing the project's Tokenizer protocol."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from tokenizers import Tokenizer

_FALLBACK_EOS = "<|endoftext|>"


class HFTokenizer:
    def __init__(self, tokenizer: Tokenizer, eos_token_id: int | None = None):
        self._tok = tokenizer
        self._eos_token_id = eos_token_id
        added = [t.content for t in tokenizer.get_added_tokens_decoder().values() if t.special]
        added.sort(key=len, reverse=True)
        self._special_re = regex.compile("|".join(regex.escape(t) for t in added)) if added else None
        self._max_special_len = max((len(t) for t in added), default=1)

    @classmethod
    def from_pretrained(cls, repo_or_path: str | os.PathLike) -> HFTokenizer:
        """Load from a hub repo id, a directory, or a ``tokenizer.json`` path. EOS comes from tokenizer_config.json."""
        path = Path(repo_or_path)
        if path.is_dir():
            tokenizer_file, config_file = path / "tokenizer.json", path / "tokenizer_config.json"
        elif path.is_file():
            tokenizer_file, config_file = path, path.with_name("tokenizer_config.json")
        else:
            tokenizer_file = Path(hf_hub_download(str(repo_or_path), "tokenizer.json"))
            try:
                config_file = Path(hf_hub_download(str(repo_or_path), "tokenizer_config.json"))
            except EntryNotFoundError:
                config_file = tokenizer_file.with_name("tokenizer_config.json")
        tokenizer = Tokenizer.from_file(str(tokenizer_file))
        eos = None
        if config_file.exists():
            eos = json.loads(config_file.read_text(encoding="utf-8")).get("eos_token")
            if isinstance(eos, dict):
                eos = eos.get("content")
        eos_id = tokenizer.token_to_id(eos) if eos else None
        if eos_id is None:
            eos_id = tokenizer.token_to_id(_FALLBACK_EOS)
        return cls(tokenizer, eos_id)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    def token_to_id(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode(list(ids), skip_special_tokens=False)

    def encode_iterable(self, pieces: Iterable[str]) -> Iterator[int]:
        """Lazily encode text pieces; equivalent to ``encode("".join(pieces))``.

        Only the part of the buffer whose pretoken boundaries are final is encoded; the rest waits for more input.
        Assumes the normalizer (if any) does not change string length around a cut.
        """
        if self._tok.pre_tokenizer is None:
            raise ValueError("streaming encode needs a pre-tokenizer to find safe cut points")
        buffer = ""
        for piece in pieces:
            buffer += piece
            cut = self._stable_cut(buffer)
            if cut:
                yield from self.encode(buffer[:cut])
                buffer = buffer[cut:]
        if buffer:
            yield from self.encode(buffer)

    def _stable_cut(self, buffer: str) -> int:
        """Largest offset such that ``encode(buffer[:cut])`` is a prefix of ``encode(buffer + anything)``."""
        limit = len(buffer) - self._max_special_len  # anything closer to the end may be a partial special token
        if limit <= 0:
            return 0
        matches = list(self._special_re.finditer(buffer)) if self._special_re else []
        last_end = max((m.end() for m in matches if m.end() <= limit), default=0)
        region_end = next((m.start() for m in matches if m.end() > limit), len(buffer))
        region = buffer[last_end:region_end]
        spans = self._tok.pre_tokenizer.pre_tokenize_str(region)
        # Cutting at a pretoken start is safe only if the prefix pretokenizes identically on its own: a
        # look-ahead such as `\s+(?!\S)` can split a whitespace run differently once text follows it.
        for k in range(len(spans) - 1, 0, -1):
            start = spans[k][1][0]
            if last_end + start <= limit and self._tok.pre_tokenizer.pre_tokenize_str(region[:start]) == spans[:k]:
                return last_end + start
        return last_end
