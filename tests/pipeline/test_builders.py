from pathlib import Path

import pytest

from scratch_cs336.data.cache import TokenCache
from scratch_cs336.pipeline import InputPath, run_steps
from scratch_cs336.pipeline.builders import tokenize, train_tokenizer, truncate_at_boundary
from scratch_cs336.tokenizer import load_tokenizer
from tests.pipeline.test_runner import NoLauncher

FIXTURES = Path(__file__).parents[1] / "fixtures"
EOT = "<|endoftext|>"


def test_truncation_keeps_whole_documents_only():
    data = f"one{EOT}two{EOT}thr".encode()
    assert truncate_at_boundary(data, "x.txt") == f"one{EOT}two{EOT}".encode()
    assert truncate_at_boundary(b'{"text": 1}\n{"te', "x.jsonl") == b'{"text": 1}\n'
    with pytest.raises(ValueError, match="no document boundary"):
        truncate_at_boundary(b"a partial document", "x.txt")


def test_tokenizer_and_cache_steps_chain_through_input_paths(tmp_path):
    corpus = str(FIXTURES / "tinystories_sample.txt")
    bpe = train_tokenizer("bpe", corpus, 300, [EOT])
    cache = tokenize("cache", corpus, InputPath(bpe))
    paths = run_steps([cache], tmp_path, NoLauncher(), dry_run=False)

    tokens = TokenCache.open(paths["cache"])
    tokenizer = load_tokenizer(str(paths["bpe"]))
    assert tokenizer.vocab_size == 300
    text = tokenizer.decode(tokens.read(0, tokens.num_tokens).tolist())
    docs = [d for d in Path(corpus).read_text().split(EOT) if d]
    assert text == "".join(d + EOT for d in docs)
