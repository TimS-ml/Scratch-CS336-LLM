"""Parallel, resumable text -> token cache build."""

from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing as mp
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scratch_cs336.data.cache import (
    LEDGER_NAME,
    TokenCache,
    commit_ledger,
    read_shard_entry,
    token_dtype,
    write_shard,
)
from scratch_cs336.tokenizer import Tokenizer, load_tokenizer
from scratch_cs336.tokenizer.bpe import find_chunk_boundaries

DOCUMENT_SEPARATOR = "<|endoftext|>"
DEFAULT_SHARD_BYTES = 32 << 20  # uncompressed text per shard for splittable inputs; bounds worker memory
_DOC_BATCH_TOKENS = 1 << 16


@dataclass(frozen=True)
class _Job:
    index: int
    path: Path
    start: int  # byte range of the input; (0, -1) = the whole file (gzip cannot be split)
    end: int
    tokenizer_spec: str
    dtype: str


def _source_id(job: _Job) -> str:
    """Path, byte range and a blake2b digest of those bytes: editing the input in place invalidates the shard."""
    digest = hashlib.blake2b(digest_size=16)
    with open(job.path, "rb") as f:
        f.seek(job.start)
        remaining = None if job.end < 0 else job.end - job.start
        while remaining is None or remaining > 0:
            block = f.read(1 << 20 if remaining is None else min(1 << 20, remaining))
            if not block:
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return f"{job.path.resolve()}:{job.start}-{job.end}:blake2b={digest.hexdigest()}"


def _is_jsonl(path: Path) -> bool:
    return path.suffix == ".jsonl" or path.name.endswith(".jsonl.gz")


def _plan_jobs(input_files: list[Path], shard_bytes: int) -> list[tuple[Path, int, int]]:
    """(path, start, end) per shard; splittable files are cut at document boundaries into ~shard_bytes pieces."""
    ranges: list[tuple[Path, int, int]] = []
    for path in input_files:
        size = path.stat().st_size
        if path.name.endswith(".gz") or size <= shard_bytes:
            ranges.append((path, 0, -1))
            continue
        separator = b"\n" if _is_jsonl(path) else DOCUMENT_SEPARATOR.encode()
        with open(path, "rb") as f:
            bounds = find_chunk_boundaries(f, -(-size // shard_bytes), separator)
        ranges.extend((path, s, e) for s, e in zip(bounds[:-1], bounds[1:], strict=True))
    return ranges


def _documents(job: _Job) -> Iterator[str]:
    path = job.path
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from _jsonl_texts(f)
        return
    with open(path, "rb") as f:
        f.seek(job.start)
        raw = f.read() if job.end < 0 else f.read(job.end - job.start)
    text = raw.decode("utf-8", errors="ignore")
    if _is_jsonl(path):
        yield from _jsonl_texts(text.splitlines())
    else:
        yield from (doc for doc in text.split(DOCUMENT_SEPARATOR) if doc)


def _jsonl_texts(lines) -> Iterator[str]:
    for line in lines:
        if line.strip():
            yield json.loads(line)["text"]


_WORKER_TOKENIZERS: dict[str, Tokenizer] = {}


def _tokenizer(spec: str) -> Tokenizer:
    if spec not in _WORKER_TOKENIZERS:
        _WORKER_TOKENIZERS[spec] = load_tokenizer(spec)
    return _WORKER_TOKENIZERS[spec]


def _token_chunks(job: _Job) -> Iterator[np.ndarray]:
    tokenizer = _tokenizer(job.tokenizer_spec)
    eos = tokenizer.eos_token_id
    batch: list[int] = []
    for doc in _documents(job):
        batch.extend(tokenizer.encode(doc))
        batch.append(eos)
        if len(batch) >= _DOC_BATCH_TOKENS:
            yield np.asarray(batch, dtype=np.int64)
            batch = []
    if batch:
        yield np.asarray(batch, dtype=np.int64)


def _run_job(args: tuple[_Job, Path]) -> int:
    job, out_dir = args
    source = _source_id(job)
    entry = read_shard_entry(out_dir, job.index)
    if entry is not None and entry.source == source and entry.dtype == job.dtype:
        return entry.num_tokens
    return write_shard(out_dir, job.index, _token_chunks(job), np.dtype(job.dtype), source).num_tokens


def build_token_cache(
    input_files: list[Path],
    tokenizer_spec: str,
    out_dir: str | os.PathLike,
    num_workers: int,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
) -> TokenCache:
    """Tokenize ``input_files`` into a sharded cache at ``out_dir`` and return it.

    ``.txt`` inputs hold documents separated by ``<|endoftext|>``; ``.jsonl`` / ``.jsonl.gz`` inputs hold one JSON
    object per line with a ``text`` field. The tokenizer's EOS id is appended after every document.
    Shards whose ledger entry already matches (same source range, same content digest of that range, same dtype)
    are not recomputed, so an interrupted build resumes where it stopped; ``ledger.json`` with ``finished=true``
    is written last.
    """
    out = Path(out_dir)
    tokenizer = load_tokenizer(tokenizer_spec)
    if tokenizer.eos_token_id is None:
        raise ValueError(f"tokenizer {tokenizer_spec!r} has no EOS token to separate documents")
    dtype = token_dtype(tokenizer.vocab_size)
    del tokenizer

    jobs = [
        _Job(i, path, start, end, tokenizer_spec, dtype.name)
        for i, (path, start, end) in enumerate(_plan_jobs([Path(p) for p in input_files], shard_bytes))
    ]
    out.mkdir(parents=True, exist_ok=True)
    (out / LEDGER_NAME).unlink(missing_ok=True)  # a rebuild is unfinished until its last shard lands
    work = [(job, out) for job in jobs]
    if num_workers <= 1 or len(jobs) <= 1:
        for item in work:
            _run_job(item)
    else:
        with mp.get_context("spawn").Pool(min(num_workers, len(jobs))) as pool:
            for _ in pool.imap_unordered(_run_job, work):
                pass
    commit_ledger(out, dtype, len(jobs))
    return TokenCache.open(out)
