"""Character-level tokenizer with ChatML specials (ids < 512, so it pairs with the ``cs336-tiny`` preset)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
_SPLIT = regex.compile("(" + "|".join(regex.escape(s) for s in SPECIALS) + ")")
_OFFSET = len(SPECIALS)


class CharTokenizer:
    vocab_size = 512
    eos_token_id = 0

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in _SPLIT.split(text):
            if piece in SPECIALS:
                ids.append(SPECIALS.index(piece))
            else:
                ids.extend(ord(c) + _OFFSET for c in piece if ord(c) + _OFFSET < self.vocab_size)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(SPECIALS[i] if i < _OFFSET else chr(i - _OFFSET) for i in ids)

    def encode_iterable(self, pieces: Iterable[str]) -> Iterator[int]:
        yield from self.encode("".join(pieces))
