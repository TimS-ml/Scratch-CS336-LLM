from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest

from scratch_cs336.tokenizer import BPETokenizer, load_tokenizer, train_bpe
from scratch_cs336.tokenizer.bpe import gpt2_bytes_to_unicode

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
EOT = "<|endoftext|>"


def _gpt2(special_tokens: list[str] | None = None) -> BPETokenizer:
    return BPETokenizer.from_gpt2_files(FIXTURES / "gpt2_vocab.json", FIXTURES / "gpt2_merges.txt", special_tokens)


def _reference_gpt2():
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer

    try:
        root = snapshot_download("gpt2", allow_patterns=["tokenizer.json"], local_files_only=True)
    except Exception:
        pytest.skip("gpt2 tokenizer not in the local HF cache")
    return Tokenizer.from_file(str(Path(root) / "tokenizer.json"))


def _fixture_texts() -> list[str]:
    names = [
        "address.txt",
        "german.txt",
        "tinystories_sample.txt",
        "special_token_trailing_newlines.txt",
        "special_token_double_newlines_non_whitespace.txt",
    ]
    texts = [(FIXTURES / n).read_text(encoding="utf-8") for n in names]
    texts += ["", "s", "héllo 🙃 wörld", f"a {EOT} b{EOT}{EOT}c", "  \n\n \t x  \n"]
    return texts


# --- training


def test_train_matches_hw1_reference_and_is_fast():
    start = time.time()
    vocab, merges = train_bpe(FIXTURES / "corpus.en", 500, [EOT])
    assert time.time() - start < 1.5

    decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    def to_bytes(token: str) -> bytes:
        return bytes(decoder[c] for c in token)

    with open(FIXTURES / "train-bpe-reference-merges.txt", encoding="utf-8") as f:
        reference_merges = [tuple(to_bytes(t) for t in line.rstrip().split(" ")) for line in f]
    assert merges == reference_merges

    reference_vocab = json.loads((FIXTURES / "train-bpe-reference-vocab.json").read_text(encoding="utf-8"))
    assert set(vocab) == set(reference_vocab.values())
    assert set(vocab.values()) == {to_bytes(t) for t in reference_vocab}


def test_train_parallel_equals_serial(tmp_path: Path):
    paragraphs = (FIXTURES / "corpus.en").read_text(encoding="utf-8").split("\n\n")
    path = tmp_path / "corpus.txt"
    path.write_text(f"\n{EOT}".join(paragraphs * 3), encoding="utf-8")
    serial = train_bpe(path, 400, [EOT], num_workers=1)
    assert train_bpe(path, 400, [EOT], num_workers=3) == serial


def test_special_tokens_are_never_merged_across(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text(f"ab{EOT}ab<|x|>ab" * 50, encoding="utf-8")
    vocab, merges = train_bpe(path, 300, [EOT, "<|x|>"])
    assert list(vocab.values())[-2:] == [EOT.encode(), b"<|x|>"]
    assert all(b"<|" not in token for token in list(vocab.values())[:-2])
    assert merges[0] == (b"a", b"b")


# --- tokenizer parity with HF GPT-2


@pytest.mark.parametrize("text", _fixture_texts())
def test_encode_matches_hf_gpt2_and_round_trips(text: str):
    tokenizer = _gpt2([EOT])
    ids = tokenizer.encode(text)
    assert ids == _reference_gpt2().encode(text).ids
    assert tokenizer.decode(ids) == text


def test_overlapping_special_tokens_resolve_longest_first():
    tokenizer = _gpt2([EOT, EOT + EOT])
    ids = tokenizer.encode(f"x{EOT}{EOT}{EOT}")
    double, single = tokenizer.vocab_size - 1, tokenizer.eos_token_id
    assert ids[1:] == [double, single]
    assert tokenizer.decode(ids) == f"x{EOT}{EOT}{EOT}"


# --- streaming


@pytest.mark.parametrize("specials", [[EOT], [EOT, EOT + EOT]])
def test_encode_iterable_equals_encode_for_any_chunking(specials: list[str]):
    tokenizer = _gpt2(specials)
    text = "".join(_fixture_texts()) + f"{EOT}{EOT}  tail {EOT}"
    expected = tokenizer.encode(text)
    rng = random.Random(0)
    for max_piece in (1, 3, 17, 500):
        pieces, i = [], 0
        while i < len(text):
            step = rng.randint(1, max_piece)
            pieces.append(text[i : i + step])
            i += step
        assert list(tokenizer.encode_iterable(pieces)) == expected


def test_encode_iterable_on_file_lines():
    tokenizer = _gpt2([EOT])
    path = FIXTURES / "tinystories_sample.txt"
    with open(path, encoding="utf-8") as f:
        streamed = list(tokenizer.encode_iterable(f))
    assert streamed == tokenizer.encode(path.read_text(encoding="utf-8"))


def test_encode_iterable_buffers_a_bounded_amount():
    tokenizer = _gpt2([EOT])
    consumed = 0

    def pieces():
        nonlocal consumed
        for _ in range(2000):
            consumed += 1
            yield "word " * 4

    first = next(iter(tokenizer.encode_iterable(pieces())))
    assert first is not None and consumed <= 2


# --- persistence


def test_save_load_round_trip(tmp_path: Path):
    path = tmp_path / "corpus.txt"
    path.write_text((FIXTURES / "tinystories_sample.txt").read_text(encoding="utf-8"), encoding="utf-8")
    vocab, merges = train_bpe(path, 400, [EOT])
    tokenizer = BPETokenizer(vocab, merges, [EOT])
    tokenizer.save(tmp_path / "tok")
    tok = tmp_path / "tok"
    for loaded in (BPETokenizer.load(tok), load_tokenizer(f"bpe:{tok}"), load_tokenizer(str(tok))):
        assert loaded.vocab == tokenizer.vocab and loaded.merges == tokenizer.merges
        assert loaded.eos_token_id == tokenizer.eos_token_id == 400 - 1
        text = f"héllo {EOT} wörld, once upon a time"
        assert loaded.encode(text) == tokenizer.encode(text)


def test_load_tokenizer_rejects_unknown_spec():
    with pytest.raises(ValueError):
        load_tokenizer("tiktoken:gpt2")
