from collections import Counter

import pytest
import torch

from scratch_cs336.data.chat import IM_END
from scratch_cs336.posttrain.data import (
    PreferenceSource,
    PromptFormat,
    PromptSource,
    SFTSource,
    format_prompt,
    gsm8k_sft_rows,
)
from scratch_cs336.posttrain.losses import IGNORE_INDEX
from scratch_cs336.posttrain.rewards import boxed_reward, gsm8k_ground_truth, r1_zero_reward
from tests.posttrain.char_tokenizer import CharTokenizer

TOK = CharTokenizer()
ROWS = [{"prompt": f"q{i}?" + "x" * i, "response": f"answer {i}"} for i in range(10)]


def _labelled_text(labels: torch.Tensor) -> str:
    return TOK.decode(t for t in labels.tolist() if t != IGNORE_INDEX)


def test_sft_labels_are_exactly_the_next_assistant_tokens():
    source = SFTSource.from_rows(ROWS, TOK, seq_len=64, global_batch_size=10, dp_rank=0, dp_size=1, seed=0)
    batch = source.batch(0)
    for row in range(10):
        ids, labels = batch["input_ids"][row], batch["labels"][row]
        assert _labelled_text(labels) in {f"answer {i}{IM_END}" for i in range(10)}
        valid = labels != IGNORE_INDEX
        # Each label is the following input token (shifted by one).
        assert torch.equal(labels[:-1][valid[:-1]], ids[1:][valid[:-1]])
        assert torch.equal(batch["loss_mask"][row], valid)
    assert sorted(_labelled_text(batch["labels"][r]) for r in range(10)) == sorted(
        f"answer {i}{IM_END}" for i in range(10)
    )


def test_sft_truncation_drops_chats_without_assistant_tokens():
    long_prompt = [{"prompt": "x" * 100, "response": "y"}]
    source = SFTSource.from_rows(ROWS + long_prompt, TOK, seq_len=40, global_batch_size=2, dp_rank=0, dp_size=1, seed=0)
    assert len(source.examples) == len(ROWS)
    assert source.batch(0)["input_ids"].shape[1] <= 40


def _rank_rows(dp_size: int, step: int) -> list[str]:
    rows = []
    for rank in range(dp_size):
        source = SFTSource.from_rows(ROWS, TOK, 64, global_batch_size=4, dp_rank=rank, dp_size=dp_size, seed=7)
        labels = source.batch(step)["labels"]
        assert labels.shape[0] == 4 // dp_size
        rows += [_labelled_text(r) for r in labels]
    return rows


def test_sft_global_batch_is_independent_of_dp_size_and_epochs_cover_each_example_once():
    for step in range(5):
        assert _rank_rows(1, step) == _rank_rows(2, step) == _rank_rows(4, step)
    # 10 examples, 4 per step: steps 0..4 are 20 samples = exactly two epochs.
    counts = Counter(text for step in range(5) for text in _rank_rows(2, step))
    assert set(counts.values()) == {2} and len(counts) == 10


def test_resume_guard_rejects_different_data():
    a = SFTSource.from_rows(ROWS, TOK, 64, 2, 0, 1, seed=0)
    b = SFTSource.from_rows(ROWS[:5], TOK, 64, 2, 0, 1, seed=0)
    a.load_state_dict(SFTSource.from_rows(ROWS, TOK, 64, 2, 0, 1, seed=0).state_dict())
    with pytest.raises(ValueError):
        a.load_state_dict(b.state_dict())


def test_preference_pairs_label_only_the_final_responses():
    rows = [
        {
            "prompt": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "earlier turn"},
                {"role": "user", "content": f"pick {i}"},
            ],
            "chosen": f"good {i}",
            "rejected": f"bad answer {i}",
        }
        for i in range(4)
    ]
    batch = PreferenceSource.from_rows(rows, TOK, 128, 4, 0, 1, seed=0).batch(0)
    width = batch["chosen_input_ids"].shape[1]
    assert all(batch[k].shape == (4, width) for k in batch)
    chosen = sorted(_labelled_text(r) for r in batch["chosen_labels"])
    rejected = sorted(_labelled_text(r) for r in batch["rejected_labels"])
    assert chosen == [f"good {i}{IM_END}" for i in range(4)]
    assert rejected == [f"bad answer {i}{IM_END}" for i in range(4)]
    # Row r of chosen and rejected is the same pair.
    for c, r in zip(batch["chosen_labels"], batch["rejected_labels"], strict=True):
        assert _labelled_text(c).split()[1] == _labelled_text(r).split()[2]


@pytest.mark.parametrize("fmt", list(PromptFormat))
def test_prompt_source_rows_match_their_ground_truth(fmt):
    rows = [{"question": f"what is {i} + {i}?", "ground_truth": str(2 * i)} for i in range(6)]
    source = PromptSource.from_rows(
        rows, TOK, fmt, max_prompt_len=1024, global_batch_size=6, dp_rank=0, dp_size=1, seed=3
    )
    batch = source.batch(0)
    for ids, n, idx in zip(batch["prompt_ids"], batch["prompt_lens"], batch["example_index"], strict=True):
        i = int(idx)
        assert ids[:n].tolist() == source.prompt(i)
        assert TOK.decode(source.prompt(i)) == format_prompt(rows[i]["question"], fmt)
        assert source.ground_truth(i) == str(2 * i)
    assert sorted(batch["example_index"].tolist()) == list(range(6))


GSM8K_ROW = {"question": "Natalia sold 48 clips?", "answer": "She sold 48/2 = <<48/2=24>>24.\n#### 1,072"}


@pytest.mark.parametrize(
    ("fmt", "reward"), [(PromptFormat.R1_ZERO, r1_zero_reward), (PromptFormat.BOXED, boxed_reward)]
)
def test_gsm8k_sft_responses_earn_full_reward_and_train_only_response_and_eos(fmt, reward):
    rows = gsm8k_sft_rows([GSM8K_ROW], fmt)
    assert "<<" not in rows[0]["response"]
    assert reward(rows[0]["response"], gsm8k_ground_truth(GSM8K_ROW["answer"]))["reward"] == 1.0
    source = SFTSource.from_rows(rows, TOK, 1024, 1, 0, 1, seed=0, fmt=fmt)
    ids, mask = source.examples[0]
    prompt = TOK.encode(format_prompt(GSM8K_ROW["question"], fmt))
    assert ids[: len(prompt)] == prompt and not any(mask[: len(prompt)])
    assert all(mask[len(prompt) :]) and ids[-1] == TOK.eos_token_id
    assert TOK.decode(ids[len(prompt) : -1]) == rows[0]["response"]
