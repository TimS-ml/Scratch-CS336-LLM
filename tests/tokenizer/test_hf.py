from __future__ import annotations

import random
from pathlib import Path

import pytest

from scratch_cs336.tokenizer import load_tokenizer
from scratch_cs336.tokenizer.hf import HFTokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _snapshot(repo: str) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, allow_patterns=["tokenizer*.json"], local_files_only=True)
    except Exception:
        pytest.skip(f"{repo} not in the local HF cache")


def test_gpt2_eos_and_vocab():
    tokenizer = HFTokenizer.from_pretrained(_snapshot("gpt2"))
    assert tokenizer.eos_token_id == 50256
    assert tokenizer.vocab_size == 50257
    assert tokenizer.encode("<|endoftext|>") == [50256]


def test_qwen_eos_comes_from_tokenizer_config():
    tokenizer = HFTokenizer.from_pretrained(_snapshot("Qwen/Qwen3.5-0.8B"))
    assert tokenizer.decode([tokenizer.eos_token_id]) == "<|im_end|>"
    ids = tokenizer.encode("<|im_start|>user\nhi<|im_end|>")
    assert ids[0] == tokenizer.token_to_id("<|im_start|>") and ids[-1] == tokenizer.eos_token_id


@pytest.mark.parametrize("repo", ["gpt2", "Qwen/Qwen3.5-0.8B"])
def test_encode_iterable_equals_encode_for_any_chunking(repo: str):
    tokenizer = load_tokenizer(f"hf:{_snapshot(repo)}")
    parts = [
        (FIXTURES / n).read_text(encoding="utf-8") for n in ("address.txt", "german.txt", "tinystories_sample.txt")
    ]
    text = "<|endoftext|>".join(parts) + "  \n\n x<|endoftext|><|endoftext|>\n\n  end"
    expected = tokenizer.encode(text)
    assert tokenizer.decode(expected) == text
    rng = random.Random(1)
    for max_piece in (1, 5, 40, 2000):
        pieces, i = [], 0
        while i < len(text):
            step = rng.randint(1, max_piece)
            pieces.append(text[i : i + step])
            i += step
        assert list(tokenizer.encode_iterable(pieces)) == expected
