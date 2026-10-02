"""Sharded on-disk token cache with a ledger.

Layout::

    <dir>/shard_00000.bin    raw little-endian tokens (uint16 when vocab_size <= 65535, else uint32)
    <dir>/shard_00000.json   per-shard ledger entry, written after the .bin is in place (resume marker)
    <dir>/ledger.json        {"dtype", "shards": [{"file", "num_tokens"}], "finished"}; written last

Every file is written to ``<name>.tmp`` and renamed, so a crash never leaves a readable partial file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

LEDGER_NAME = "ledger.json"


def token_dtype(vocab_size: int) -> np.dtype:
    return np.dtype(np.uint16 if vocab_size <= 65535 else np.uint32)


def shard_name(index: int) -> str:
    return f"shard_{index:05d}.bin"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True)
class ShardEntry:
    source: str  # identifies what the shard was built from; a changed source invalidates the entry
    dtype: str
    num_tokens: int


def read_shard_entry(directory: Path, index: int) -> ShardEntry | None:
    """The shard's ledger entry, or None unless both the entry and a correctly sized .bin exist."""
    entry_path = directory / Path(shard_name(index)).with_suffix(".json")
    shard_path = directory / shard_name(index)
    if not entry_path.exists() or not shard_path.exists():
        return None
    entry = ShardEntry(**json.loads(entry_path.read_text(encoding="utf-8")))
    if shard_path.stat().st_size != entry.num_tokens * np.dtype(entry.dtype).itemsize:
        return None
    return entry


def write_shard(directory: Path, index: int, chunks: Iterable[np.ndarray], dtype: np.dtype, source: str) -> ShardEntry:
    """Stream ``chunks`` into shard ``index`` and publish its ledger entry. Returns the entry."""
    directory.mkdir(parents=True, exist_ok=True)
    shard_path = directory / shard_name(index)
    tmp = shard_path.with_name(shard_path.name + ".tmp")
    num_tokens = 0
    with open(tmp, "wb") as f:
        for chunk in chunks:
            chunk.astype(dtype, copy=False).tofile(f)
            num_tokens += len(chunk)
    os.replace(tmp, shard_path)
    entry = ShardEntry(source=source, dtype=dtype.name, num_tokens=num_tokens)
    _atomic_write_text(shard_path.with_suffix(".json"), json.dumps(entry.__dict__))
    return entry


def commit_ledger(directory: Path, dtype: np.dtype, num_shards: int) -> None:
    """Consolidate the per-shard entries into ``ledger.json`` with ``finished=true``. Call after all shards exist."""
    shards = []
    for i in range(num_shards):
        entry = read_shard_entry(directory, i)
        if entry is None or entry.dtype != dtype.name:
            raise FileNotFoundError(f"shard {i} is missing or inconsistent in {directory}")
        shards.append({"file": shard_name(i), "num_tokens": entry.num_tokens})
    ledger = {"dtype": dtype.name, "shards": shards, "finished": True}
    _atomic_write_text(directory / LEDGER_NAME, json.dumps(ledger, indent=2))


class TokenCache:
    """Read side: the shards viewed as one contiguous token stream (a ``WindowSource``)."""

    def __init__(self, directory: Path, dtype: np.dtype, shards: list[np.ndarray]):
        self.directory = directory
        self.dtype = dtype
        self._shards = shards
        # _starts[i] = global offset of shard i's first token; _starts[-1] = num_tokens
        self._starts = np.concatenate([[0], np.cumsum([len(s) for s in shards])]).astype(np.int64)

    @classmethod
    def open(cls, directory: str | os.PathLike) -> TokenCache:
        root = Path(directory)
        ledger = json.loads((root / LEDGER_NAME).read_text(encoding="utf-8"))
        if not ledger["finished"]:
            raise ValueError(f"token cache at {root} is not finished")
        dtype = np.dtype(ledger["dtype"])
        shards = []
        for shard in ledger["shards"]:
            count = shard["num_tokens"]
            # np.memmap cannot map an empty file
            shards.append(np.memmap(root / shard["file"], dtype=dtype, mode="r") if count else np.empty(0, dtype))
            if len(shards[-1]) != count:
                raise ValueError(f"{shard['file']} holds {len(shards[-1])} tokens, ledger says {count}")
        return cls(root, dtype, shards)

    @property
    def num_tokens(self) -> int:
        return int(self._starts[-1])

    @property
    def shard_tokens(self) -> list[int]:
        return [len(s) for s in self._shards]

    def read(self, start: int, length: int) -> np.ndarray:
        """Tokens ``[start, start + length)`` of the concatenated stream (a copy)."""
        if start < 0 or length < 0 or start + length > self.num_tokens:
            raise IndexError(f"range [{start}, {start + length}) outside [0, {self.num_tokens})")
        out = np.empty(length, dtype=self.dtype)
        first = int(np.searchsorted(self._starts, start, side="right")) - 1
        pos, shard = start, first
        while pos < start + length:
            shard_start = int(self._starts[shard])
            take_to = min(start + length, int(self._starts[shard + 1]))
            out[pos - start : take_to - start] = self._shards[shard][pos - shard_start : take_to - shard_start]
            pos, shard = take_to, shard + 1
        return out

    def num_windows(self, seq_len: int) -> int:
        """Non-overlapping windows of ``seq_len + 1`` tokens at stride ``seq_len`` (inputs and shifted targets)."""
        return max(self.num_tokens - 1, 0) // seq_len

    def window(self, i: int, seq_len: int) -> np.ndarray:
        if not 0 <= i < self.num_windows(seq_len):
            raise IndexError(f"window {i} outside [0, {self.num_windows(seq_len)})")
        return self.read(i * seq_len, seq_len + 1)
