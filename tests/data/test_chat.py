from __future__ import annotations

from pathlib import Path

import pytest

from scratch_cs336.data.chat import IM_END, IM_START, render_chat, tokenize_chat
from scratch_cs336.tokenizer import BPETokenizer, HFTokenizer, train_bpe

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

CONVERSATION = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "And times three?"},
    {"role": "assistant", "content": "12, héllo\n  end\n"},
]


@pytest.fixture(scope="module")
def tiny_bpe(tmp_path_factory) -> BPETokenizer:
    corpus = tmp_path_factory.mktemp("chat") / "corpus.txt"
    corpus.write_text((FIXTURES / "tinystories_sample.txt").read_text(encoding="utf-8"), encoding="utf-8")
    specials = ["<|endoftext|>", IM_START, IM_END]
    vocab, merges = train_bpe(corpus, 400, specials)
    return BPETokenizer(vocab, merges, specials)


def test_render_chatml_and_generation_prompt():
    text = render_chat(CONVERSATION[:2])
    assert text == "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
    assert render_chat(CONVERSATION[:2], add_generation_prompt=True) == text + "<|im_start|>assistant\n"


def _check(tokenizer) -> None:
    ids, mask = tokenize_chat(CONVERSATION, tokenizer)
    assert ids == tokenizer.encode(render_chat(CONVERSATION))
    assert len(mask) == len(ids) and set(mask) <= {0, 1}
    # Contiguous trained runs decode to exactly the assistant content plus its <|im_end|>.
    runs, current = [], []
    for token, m in zip(ids, mask, strict=True):
        if m:
            current.append(token)
        elif current:
            runs.append(current)
            current = []
    assert [tokenizer.decode(r) for r in runs] == ["4" + IM_END, "12, héllo\n  end\n" + IM_END]
    # Nothing else is trained: system/user turns and role headers have mask 0.
    assert sum(mask) == sum(len(r) for r in runs)


def test_mask_covers_only_assistant_content_and_im_end_bpe(tiny_bpe: BPETokenizer):
    _check(tiny_bpe)


def test_mask_with_qwen_tokenizer():
    from huggingface_hub import snapshot_download

    try:
        root = snapshot_download("Qwen/Qwen3.5-0.8B", allow_patterns=["tokenizer*.json"], local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not in the local HF cache")
    _check(HFTokenizer.from_pretrained(root))
