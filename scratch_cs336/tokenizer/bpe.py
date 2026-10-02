"""Byte-level BPE: parallel training and a streaming tokenizer (GPT-2 pretokenization, CS336 hw1 contract)."""

from __future__ import annotations

import heapq
import json
import multiprocessing as mp
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path
from typing import BinaryIO

import regex

GPT2_PATTERN = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

_MIN_PARALLEL_BYTES = 8 << 20  # below this per worker, process start-up costs more than it saves
_MAX_CHUNK_BYTES = 64 << 20  # bounds the memory a pretokenization worker holds
_PRETOKEN_CACHE_SIZE = 200_000


@cache
def gpt2_bytes_to_unicode() -> dict[int, str]:
    """GPT-2's reversible byte -> printable-unicode map, used by the on-disk vocab/merges format."""
    printable = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    codepoints = printable[:]
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(printable, codepoints, strict=True)}


def _special_pattern(special_tokens: Iterable[str]) -> regex.Pattern[str] | None:
    """Alternation of the specials, longest first so overlapping tokens resolve to the longest."""
    ordered = sorted(set(special_tokens), key=len, reverse=True)
    return regex.compile("|".join(regex.escape(t) for t in ordered)) if ordered else None


# ----------------------------------------------------------------------------- training


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """Byte offsets splitting ``file`` into independent chunks, each (but the first) starting at a split token.

    May return fewer chunks than requested when boundaries coincide.
    """
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    chunk_size = max(file_size // desired_num_chunks, 1)
    boundaries = [min(i * chunk_size, file_size) for i in range(desired_num_chunks + 1)]
    boundaries[-1] = file_size
    mini_chunk_size = 4096
    for bi in range(1, len(boundaries) - 1):
        position = boundaries[bi]
        file.seek(position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if not mini_chunk:
                boundaries[bi] = file_size
                break
            found = mini_chunk.find(split_special_token)
            if found != -1:
                boundaries[bi] = position + found
                break
            position += mini_chunk_size
    return sorted(set(boundaries))


def _count_pretokens(args: tuple[str, int, int, tuple[str, ...]]) -> Counter[bytes]:
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    splitter = _special_pattern(special_tokens)
    counts: Counter[str] = Counter()
    for part in splitter.split(text) if splitter else [text]:
        counts.update(m.group() for m in GPT2_PATTERN.finditer(part))
    return Counter({k.encode("utf-8"): v for k, v in counts.items()})


def _pretoken_counts(path: Path, special_tokens: tuple[str, ...], num_workers: int | None) -> Counter[bytes]:
    size = path.stat().st_size
    workers = num_workers if num_workers is not None else min(os.cpu_count() or 1, max(1, size // _MIN_PARALLEL_BYTES))
    num_chunks = max(workers, -(-size // _MAX_CHUNK_BYTES))
    if special_tokens and num_chunks > 1:
        with open(path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, special_tokens[0].encode("utf-8"))
    else:
        boundaries = [0, size]  # without a split token there is no safe place to cut
    jobs = [(str(path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:], strict=True)]
    total: Counter[bytes] = Counter()
    if workers <= 1 or len(jobs) == 1:
        for job in jobs:
            total.update(_count_pretokens(job))
        return total
    with mp.get_context("spawn").Pool(min(workers, len(jobs))) as pool:
        for partial in pool.imap_unordered(_count_pretokens, jobs):
            total.update(partial)
    return total


class _Descending:
    """Heap key that orders byte-pairs from lexicographically greatest to least."""

    __slots__ = ("key",)

    def __init__(self, key: tuple[bytes, bytes]):
        self.key = key

    def __lt__(self, other: _Descending) -> bool:
        return self.key > other.key


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_workers: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train byte-level BPE. Special tokens split the corpus (never merged across) and take the last ids.

    ``num_workers=None`` picks a worker count from the file size; the result never depends on it.
    Ties between equally frequent pairs go to the lexicographically greatest ``(bytes, bytes)`` pair.
    """
    specials = tuple(dict.fromkeys(special_tokens))
    num_merges = vocab_size - 256 - len(specials)
    if num_merges < 0:
        raise ValueError(f"vocab_size={vocab_size} is smaller than 256 bytes + {len(specials)} special tokens")

    counts = _pretoken_counts(Path(input_path), specials, num_workers)
    words = [list(word) for word in counts]
    freqs = list(counts.values())
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    where: dict[tuple[int, int], set[int]] = defaultdict(set)  # superset of words containing the pair
    for idx, word in enumerate(words):
        for pair in zip(word, word[1:], strict=False):
            pair_counts[pair] += freqs[idx]
            where[pair].add(idx)

    def entry(pair: tuple[int, int]) -> tuple[int, _Descending, tuple[int, int]]:
        return (-pair_counts[pair], _Descending((vocab[pair[0]], vocab[pair[1]])), pair)

    heap = [entry(p) for p in pair_counts]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        pair = None
        while heap:
            neg_count, _, candidate = heapq.heappop(heap)
            if pair_counts.get(candidate, 0) == -neg_count:  # otherwise a stale entry; a fresh one was pushed
                pair = candidate
                break
        if pair is None:
            break
        a, b = pair
        new_id = 256 + len(merges)
        vocab[new_id] = vocab[a] + vocab[b]
        merges.append((vocab[a], vocab[b]))

        touched: set[tuple[int, int]] = set()
        for idx in where.pop(pair):
            word = words[idx]
            merged = _merge_word(word, a, b, new_id)
            if len(merged) == len(word):
                continue
            freq = freqs[idx]
            for p in zip(word, word[1:], strict=False):
                pair_counts[p] -= freq
                touched.add(p)
            for p in zip(merged, merged[1:], strict=False):
                pair_counts[p] += freq
                where[p].add(idx)
                touched.add(p)
            words[idx] = merged
        del pair_counts[pair]
        touched.discard(pair)
        for p in touched:
            if pair_counts[p] > 0:
                heapq.heappush(heap, entry(p))
            else:
                del pair_counts[p]

    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")
    return vocab, merges


def _merge_word(word: list[int], a: int, b: int, new_id: int) -> list[int]:
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


# ----------------------------------------------------------------------------- tokenizer


class BPETokenizer:
    """Byte-level BPE tokenizer. ``encode_iterable`` streams with memory bounded by the largest input piece."""

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        self._token_to_id = {token: i for i, token in self.vocab.items()}
        for special in self.special_tokens:
            encoded = special.encode("utf-8")
            if encoded not in self._token_to_id:
                new_id = max(self.vocab, default=-1) + 1
                self.vocab[new_id] = encoded
                self._token_to_id[encoded] = new_id
        self._ranks = {(self._token_to_id[a], self._token_to_id[b]): rank for rank, (a, b) in enumerate(self.merges)}
        self._merged_id = {
            (self._token_to_id[a], self._token_to_id[b]): self._token_to_id[a + b] for a, b in self.merges
        }
        self._special_ids = {s: self._token_to_id[s.encode("utf-8")] for s in self.special_tokens}
        self._splitter = _special_pattern(self.special_tokens)
        self._capturing_splitter = regex.compile(f"({self._splitter.pattern})") if self._splitter else None
        self._pretoken_cache: dict[str, tuple[int, ...]] = {}

    # --- properties

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eos_token_id(self) -> int | None:
        if "<|endoftext|>" in self._special_ids:
            return self._special_ids["<|endoftext|>"]
        return next(iter(self._special_ids.values()), None)

    # --- encode

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        cached = self._pretoken_cache.get(pretoken)
        if cached is not None:
            return cached
        ids = [self._token_to_id[bytes([b])] for b in pretoken.encode("utf-8")]
        while len(ids) > 1:
            best = min(zip(ids, ids[1:], strict=False), key=lambda p: self._ranks.get(p, len(self._ranks)))
            if best not in self._ranks:
                break
            ids = _merge_word(ids, best[0], best[1], self._merged_id[best])
        result = tuple(ids)
        if len(self._pretoken_cache) >= _PRETOKEN_CACHE_SIZE:
            self._pretoken_cache.clear()
        self._pretoken_cache[pretoken] = result
        return result

    def _encode_plain(self, text: str) -> Iterator[int]:
        for match in GPT2_PATTERN.finditer(text):
            yield from self._encode_pretoken(match.group())

    def _hold_back(self, text: str) -> int:
        """Length of the longest suffix of ``text`` that could still grow into a longer special token.

        A suffix that starts strictly inside an already complete special token cannot start a new one.
        """
        if self._splitter is None:
            return 0
        spans = [m.span() for m in self._splitter.finditer(text)]
        longest = 0
        for special in self.special_tokens:
            for k in range(min(len(special) - 1, len(text)), longest, -1):
                start = len(text) - k
                if text.endswith(special[:k]) and not any(s < start < e for s, e in spans):
                    longest = k
                    break
        return longest

    def _encode_stream(self, text: str, final: bool) -> tuple[list[int], str]:
        """Encode the stable prefix of ``text``; return (ids, unconsumed suffix). ``final`` consumes everything."""
        hold = 0 if final else self._hold_back(text)
        body = text[: len(text) - hold]
        parts = self._capturing_splitter.split(body) if self._capturing_splitter else [body]
        ids: list[int] = []
        rest = ""
        # re.split with a capture group puts special-token matches at odd indices
        for k, part in enumerate(parts):
            if k % 2 == 1:
                ids.append(self._special_ids[part])
            elif k == len(parts) - 1 and not final:
                pretokens = [m.group() for m in GPT2_PATTERN.finditer(part)]
                for pretoken in pretokens[:-1]:
                    ids.extend(self._encode_pretoken(pretoken))
                rest = pretokens[-1] if pretokens else ""
            else:
                ids.extend(self._encode_plain(part))
        return ids, rest + text[len(text) - hold :]

    def encode(self, text: str) -> list[int]:
        return self._encode_stream(text, final=True)[0]

    def encode_iterable(self, pieces: Iterable[str]) -> Iterator[int]:
        """Lazily encode text pieces (e.g. file lines). Equivalent to ``encode("".join(pieces))``."""
        buffer = ""
        for piece in pieces:
            buffer += piece
            ids, buffer = self._encode_stream(buffer, final=False)
            yield from ids
        yield from self._encode_stream(buffer, final=True)[0]

    def decode(self, ids: Iterable[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # --- persistence

    def save(self, directory: str | os.PathLike) -> None:
        """GPT-2 layout: ``vocab.json`` + ``merges.txt`` (byte-unicode encoded) + ``special_tokens.json``."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        to_unicode = gpt2_bytes_to_unicode()

        def encode_bytes(token: bytes) -> str:
            return "".join(to_unicode[b] for b in token)

        vocab = {encode_bytes(token): i for i, token in self.vocab.items()}
        (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        lines = [f"{encode_bytes(a)} {encode_bytes(b)}" for a, b in self.merges]
        (out / "merges.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (out / "special_tokens.json").write_text(json.dumps(self.special_tokens), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | os.PathLike) -> BPETokenizer:
        root = Path(directory)
        specials_path = root / "special_tokens.json"
        specials = json.loads(specials_path.read_text(encoding="utf-8")) if specials_path.exists() else []
        return cls.from_gpt2_files(root / "vocab.json", root / "merges.txt", specials)

    @classmethod
    def from_gpt2_files(
        cls, vocab_path: str | os.PathLike, merges_path: str | os.PathLike, special_tokens: list[str] | None = None
    ) -> BPETokenizer:
        from_unicode = {c: b for b, c in gpt2_bytes_to_unicode().items()}

        def decode_bytes(token: str) -> bytes:
            return bytes(from_unicode[c] for c in token)

        with open(vocab_path, encoding="utf-8") as f:
            vocab = {i: decode_bytes(token) for token, i in json.load(f).items()}
        merges = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split(" ")
                if len(fields) == 2 and all(fields):  # skips the "#version" header
                    merges.append((decode_bytes(fields[0]), decode_bytes(fields[1])))
        return cls(vocab, merges, special_tokens)
