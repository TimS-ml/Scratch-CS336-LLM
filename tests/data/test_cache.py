from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from scratch_cs336.data.cache import TokenCache, commit_ledger, read_shard_entry, token_dtype, write_shard
from scratch_cs336.data.tokenize import DOCUMENT_SEPARATOR, build_token_cache
from scratch_cs336.tokenizer import BPETokenizer, load_tokenizer, train_bpe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EOT = "<|endoftext|>"


def _write_cache(directory: Path, shards: list[np.ndarray], dtype: np.dtype | None = None) -> TokenCache:
    dtype = dtype or np.dtype(np.uint16)
    for i, shard in enumerate(shards):
        write_shard(directory, i, [shard], dtype, source=f"s{i}")
    commit_ledger(directory, dtype, len(shards))
    return TokenCache.open(directory)


def test_reads_across_shard_boundaries_equal_the_concatenated_stream(tmp_path: Path):
    rng = np.random.default_rng(0)
    sizes = [5, 0, 17, 1, 0, 0, 30, 2]
    shards = [rng.integers(0, 60000, n).astype(np.uint16) for n in sizes]
    stream = np.concatenate(shards)
    cache = _write_cache(tmp_path, shards)
    assert cache.num_tokens == len(stream) and cache.shard_tokens == sizes
    for start in range(len(stream) + 1):
        for length in (0, 1, 2, 7, 25, len(stream) - start):
            if start + length <= len(stream):
                assert np.array_equal(cache.read(start, length), stream[start : start + length])
    with pytest.raises(IndexError):
        cache.read(len(stream) - 1, 2)


def test_windows_stride_by_seq_len_with_one_token_overlap(tmp_path: Path):
    cache = _write_cache(tmp_path, [np.arange(0, 10, dtype=np.uint16), np.arange(10, 23, dtype=np.uint16)])
    assert cache.num_tokens == 23
    assert cache.num_windows(4) == 5  # window 4 ends at token index 20 < 23; a sixth would need index 24
    assert cache.num_windows(11) == 2
    assert cache.num_windows(22) == 1 and cache.num_windows(23) == 0
    for i in range(5):
        assert cache.window(i, 4).tolist() == list(range(4 * i, 4 * i + 5))
    with pytest.raises(IndexError):
        cache.window(5, 4)


def test_dtype_widens_when_ids_do_not_fit_uint16(tmp_path: Path):
    assert token_dtype(65535) == np.uint16 and token_dtype(65536) == np.uint32
    big = np.array([0, 70000, 2**20], dtype=np.int64)
    cache = _write_cache(tmp_path, [big], dtype=token_dtype(2**20 + 1))
    assert cache.read(0, 3).tolist() == big.tolist()


def test_unfinished_cache_cannot_be_opened(tmp_path: Path):
    write_shard(tmp_path, 0, [np.arange(4)], np.dtype(np.uint16), source="s")
    with pytest.raises(FileNotFoundError):
        TokenCache.open(tmp_path)  # no ledger.json yet
    (tmp_path / "ledger.json").write_text(json.dumps({"dtype": "uint16", "shards": [], "finished": False}))
    with pytest.raises(ValueError):
        TokenCache.open(tmp_path)


def test_truncated_shard_is_not_a_finished_entry(tmp_path: Path):
    write_shard(tmp_path, 0, [np.arange(8)], np.dtype(np.uint16), source="s")
    assert read_shard_entry(tmp_path, 0).num_tokens == 8
    (tmp_path / "shard_00000.bin").write_bytes(b"\x00" * 6)
    assert read_shard_entry(tmp_path, 0) is None


# --- build_token_cache


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory) -> str:
    corpus = tmp_path_factory.mktemp("bpe") / "corpus.txt"
    corpus.write_text((FIXTURES / "tinystories_sample.txt").read_text(encoding="utf-8"), encoding="utf-8")
    vocab, merges = train_bpe(corpus, 330, [EOT])
    directory = corpus.parent / "tok"
    BPETokenizer(vocab, merges, [EOT]).save(directory)
    return str(directory)


DOCS = ["Once upon a time.", "héllo wörld\n\nsecond line", "  leading space", "x", "Another story, with 3 numbers 123."]


def _expected_stream(spec: str, docs: list[str]) -> list[int]:
    tokenizer = load_tokenizer(spec)
    return [t for doc in docs for t in (*tokenizer.encode(doc), tokenizer.eos_token_id)]


def _inputs(directory: Path) -> tuple[list[Path], list[str]]:
    txt = directory / "a.txt"
    txt.write_text(DOCUMENT_SEPARATOR.join(DOCS[:2]) + DOCUMENT_SEPARATOR, encoding="utf-8")
    lines = [json.dumps({"text": d, "id": i}) for i, d in enumerate(DOCS[2:4])]
    jsonl = directory / "b.jsonl"
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    gz = directory / "c.jsonl.gz"
    with gzip.open(gz, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"text": DOCS[4]}) + "\n")
    return [txt, jsonl, gz], DOCS


@pytest.mark.parametrize("num_workers", [1, 2])
def test_build_tokenizes_every_input_type_with_eos_after_each_document(tmp_path: Path, tokenizer_dir: str, num_workers):
    files, docs = _inputs(tmp_path)
    spec = f"bpe:{tokenizer_dir}"
    cache = build_token_cache(files, spec, tmp_path / "cache", num_workers=num_workers)
    expected = _expected_stream(spec, docs)
    assert cache.num_tokens == len(expected)
    assert cache.read(0, cache.num_tokens).tolist() == expected
    assert len(cache.shard_tokens) == 3


def test_large_files_are_split_into_shards_at_document_boundaries(tmp_path: Path, tokenizer_dir: str):
    spec = f"bpe:{tokenizer_dir}"
    docs = [f"document number {i} " * (1 + i % 5) for i in range(60)]
    txt = tmp_path / "big.txt"
    txt.write_text(DOCUMENT_SEPARATOR.join(docs), encoding="utf-8")
    jsonl = tmp_path / "big.jsonl"
    jsonl.write_text("".join(json.dumps({"text": d}) + "\n" for d in docs), encoding="utf-8")
    for path in (txt, jsonl):
        cache = build_token_cache([path], spec, tmp_path / f"cache_{path.suffix[1:]}", num_workers=1, shard_bytes=400)
        assert len(cache.shard_tokens) > 3
        assert cache.read(0, cache.num_tokens).tolist() == _expected_stream(spec, docs)


def test_interrupted_build_resumes_without_redoing_finished_shards(tmp_path: Path, tokenizer_dir: str):
    files, docs = _inputs(tmp_path)
    spec = f"bpe:{tokenizer_dir}"
    out = tmp_path / "cache"
    build_token_cache(files, spec, out, num_workers=1)
    # Simulate a crash after shard 0: no final ledger, shards 1 and 2 never finished.
    (out / "ledger.json").unlink()
    for i in (1, 2):
        (out / f"shard_0000{i}.json").unlink()
    # Tamper with the finished shard's content so a recompute would be detectable.
    marker = np.full(3, 7, dtype=np.uint16)
    marker.tofile(out / "shard_00000.bin")
    entry = json.loads((out / "shard_00000.json").read_text())
    entry["num_tokens"] = 3
    (out / "shard_00000.json").write_text(json.dumps(entry))

    cache = build_token_cache(files, spec, out, num_workers=1)
    tail = _expected_stream(spec, docs[2:])
    assert cache.read(0, cache.num_tokens).tolist() == [7, 7, 7, *tail]


def test_shard_built_from_a_different_source_is_recomputed(tmp_path: Path, tokenizer_dir: str):
    files, docs = _inputs(tmp_path)
    spec = f"bpe:{tokenizer_dir}"
    out = tmp_path / "cache"
    build_token_cache(files, spec, out, num_workers=1)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"text": "changed"}) + "\n", encoding="utf-8")
    cache = build_token_cache([files[0], other, files[2]], spec, out, num_workers=1)
    expected = _expected_stream(spec, [*docs[:2], "changed", docs[4]])
    assert cache.read(0, cache.num_tokens).tolist() == expected


def test_input_edited_in_place_with_same_length_is_retokenized(tmp_path: Path, tokenizer_dir: str):
    files, docs = _inputs(tmp_path)
    spec = f"bpe:{tokenizer_dir}"
    out = tmp_path / "cache"
    build_token_cache(files, spec, out, num_workers=1)
    size = files[0].stat().st_size
    files[0].write_text(files[0].read_text(encoding="utf-8").replace("Once", "Then"), encoding="utf-8")
    assert files[0].stat().st_size == size
    cache = build_token_cache(files, spec, out, num_workers=1)
    assert cache.read(0, cache.num_tokens).tolist() == _expected_stream(spec, ["Then upon a time.", *docs[1:]])
