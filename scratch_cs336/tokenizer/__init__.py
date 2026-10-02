"""Tokenizers behind one protocol: from-scratch byte-level BPE and HF `tokenizers`."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from scratch_cs336.tokenizer.bpe import BPETokenizer, train_bpe
from scratch_cs336.tokenizer.hf import HFTokenizer


class Tokenizer(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Iterable[int]) -> str: ...

    def encode_iterable(self, pieces: Iterable[str]) -> Iterator[int]: ...


def load_tokenizer(spec: str) -> Tokenizer:
    """``"bpe:<dir>"`` loads a saved :class:`BPETokenizer`; ``"hf:<repo_or_path>"`` an :class:`HFTokenizer`.

    A bare existing directory is a saved :class:`BPETokenizer` (what a pipeline tokenizer step produces).
    """
    if Path(spec).is_dir():
        return BPETokenizer.load(spec)
    kind, sep, target = spec.partition(":")
    if not sep or not target:
        raise ValueError(f"tokenizer spec must be 'bpe:<dir>', 'hf:<repo_or_path>' or a BPE directory, got {spec!r}")
    match kind:
        case "bpe":
            return BPETokenizer.load(target)
        case "hf":
            return HFTokenizer.from_pretrained(target)
        case _:
            raise ValueError(f"unknown tokenizer kind {kind!r} in {spec!r}")


def check_tokenizer_vocab(spec: str, tokenizer: Tokenizer, model_vocab: int, hf_checkpoint: str | None) -> None:
    """Raise unless ``tokenizer`` (loaded from ``spec``) matches a model with ``model_vocab`` embedding rows.

    A model initialized from scratch or from our own export (``hf_checkpoint=None``) only needs a row per token id.
    HF weights were trained with one tokenizer: ``spec`` must be the checkpoint's own (``hf:<hf_checkpoint>``, whose
    vocabulary may be smaller than the padded embedding) or have exactly ``model_vocab`` tokens.
    """
    vocab = tokenizer.vocab_size
    if hf_checkpoint is None:
        if vocab > model_vocab:
            raise ValueError(f"tokenizer {spec!r} has {vocab} tokens, the model only {model_vocab}")
        return
    kind, _, target = spec.partition(":")
    own = kind == "hf" and Path(target) == Path(hf_checkpoint)
    if not own and vocab != model_vocab:
        raise ValueError(
            f"tokenizer {spec!r} has {vocab} tokens but HF checkpoint {hf_checkpoint!r} has {model_vocab}; "
            f"use the checkpoint's own tokenizer 'hf:{hf_checkpoint}' or one with exactly {model_vocab} tokens"
        )


__all__ = ["BPETokenizer", "HFTokenizer", "Tokenizer", "check_tokenizer_vocab", "load_tokenizer", "train_bpe"]
