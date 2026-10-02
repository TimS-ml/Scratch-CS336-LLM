"""Post-training data: SFT chats, preference pairs and RL prompts as deterministic, random-access batch sources.

Every source follows the pretraining loader's scheme: global sample ``g = step * G + j`` is example
``perm_{seed, g // N}(g % N)`` and dp rank ``r`` owns ``j in [r * G / dp, (r + 1) * G / dp)``. Batches are
right-padded to the longest row of the rank's slice; padded positions carry ``IGNORE_INDEX`` labels.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import re
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scratch_cs336.data.chat import Message, render_chat, tokenize_chat
from scratch_cs336.data.permutation import FeistelPermutation, derive_seed
from scratch_cs336.posttrain.losses import IGNORE_INDEX
from scratch_cs336.posttrain.rewards import gsm8k_ground_truth
from scratch_cs336.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

Row = dict[str, Any]
Batch = dict[str, Tensor]
Example = tuple[list[int], list[int]]  # token ids, 0/1 loss mask over the same positions

R1_ZERO_PROMPT = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant "
    "first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning "
    "process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>"
)
BOXED_PROMPT = "{question} Please put your final answer within \\boxed{{}}."
_CALCULATOR = re.compile(r"<<[^>]*>>")


class PromptFormat(StrEnum):
    R1_ZERO = "r1_zero"  # raw text, pairs with rewards.r1_zero_reward and stop string "</answer>"
    BOXED = "boxed"  # raw text (hw5 question_only), pairs with rewards.boxed_reward
    CHAT = "chat"  # ChatML user turn + open assistant turn


def format_prompt(question: str, fmt: PromptFormat) -> str:
    match PromptFormat(fmt):
        case PromptFormat.R1_ZERO:
            return R1_ZERO_PROMPT.format(question=question)
        case PromptFormat.BOXED:
            return BOXED_PROMPT.format(question=question)
        case PromptFormat.CHAT:
            return render_chat([{"role": "user", "content": question}], add_generation_prompt=True)


def pack_prompt_response(
    prompt_ids: Sequence[Sequence[int]], response_ids: Sequence[Sequence[int]], pad_token_id: int = 0
) -> Batch:
    """Concatenate each prompt with its response, right-pad, and shift (hw5 ``tokenize_prompt_and_output``).

    Returns ``input_ids`` / ``labels`` / ``response_mask`` of shape ``[B, max_len - 1]``; ``response_mask`` is True
    where the label is a response token. Padded labels are ``pad_token_id``.
    """
    rows = [list(p) + list(r) for p, r in zip(prompt_ids, response_ids, strict=True)]
    width = max(len(r) for r in rows)
    tokens = torch.full((len(rows), width), pad_token_id, dtype=torch.long)
    in_response = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, (row, prompt) in enumerate(zip(rows, prompt_ids, strict=True)):
        tokens[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        in_response[i, len(prompt) : len(row)] = True
    return {"input_ids": tokens[:, :-1], "labels": tokens[:, 1:], "response_mask": in_response[:, 1:]}


def tokenize_prompt_and_output(
    prompt_strs: Sequence[str], output_strs: Sequence[str], tokenizer: Tokenizer, pad_token_id: int = 0
) -> Batch:
    """Prompts and outputs are encoded separately, so the response boundary is a token boundary."""
    return pack_prompt_response(
        [tokenizer.encode(p) for p in prompt_strs], [tokenizer.encode(o) for o in output_strs], pad_token_id
    )


def messages_from_row(row: Row) -> list[Message]:
    """``messages`` as-is, or a single user/assistant exchange from ``prompt/response`` or ``question/answer``."""
    if "messages" in row:
        return list(row["messages"])
    for user_key, assistant_key in (("prompt", "response"), ("question", "answer")):
        if user_key in row and assistant_key in row:
            return [{"role": "user", "content": row[user_key]}, {"role": "assistant", "content": row[assistant_key]}]
    raise KeyError(f"row has neither 'messages' nor prompt/response or question/answer keys: {sorted(row)}")


def read_jsonl(path: str | os.PathLike) -> list[Row]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gsm8k(split: str, path: str | os.PathLike | None = None) -> list[Row]:
    """GSM8K rows ``{"question", "answer"}`` from a local jsonl, else HF ``openai/gsm8k`` via ``datasets``."""
    if path is not None:
        return read_jsonl(path)
    try:
        import datasets
    except ImportError as e:
        raise ImportError(
            "loading openai/gsm8k from the Hub needs the `datasets` extra (pip install 'scratch-cs336[datasets]'); "
            "alternatively pass a local jsonl path with question/answer rows"
        ) from e
    return [dict(row) for row in datasets.load_dataset("openai/gsm8k", "main", split=split)]


def load_rows(spec: str) -> list[Row]:
    """``"gsm8k:<split>"`` (HF ``openai/gsm8k``) or a local jsonl / jsonl.gz path."""
    kind, sep, split = spec.partition(":")
    if sep and kind == "gsm8k":
        return load_gsm8k(split)
    return read_jsonl(spec)


def rl_rows(rows: Sequence[Row]) -> list[Row]:
    """``{"question", "ground_truth"}`` rows; GSM8K ``answer`` fields contribute their final ``#### ...`` value."""
    return [
        {
            "question": r["question"],
            "ground_truth": r["ground_truth"] if "ground_truth" in r else gsm8k_ground_truth(r["answer"]),
        }
        for r in rows
    ]


def gsm8k_sft_rows(rows: Sequence[Row], fmt: PromptFormat) -> list[Row]:
    """``{"prompt": question, "response": solution}`` with the final answer in ``fmt``'s reward convention
    (``</think> <answer> x </answer>`` for r1_zero, ``\\boxed{x}`` for boxed, a closing sentence for chat).
    Calculator annotations ``<<48/2=24>>`` are dropped."""
    out = []
    for row in rows:
        reasoning = _CALCULATOR.sub("", row["answer"].rpartition("####")[0]).strip()
        final = gsm8k_ground_truth(row["answer"])
        match PromptFormat(fmt):
            case PromptFormat.R1_ZERO:
                response = f" {reasoning} </think> <answer> {final} </answer>"
            case PromptFormat.BOXED:
                response = f"{reasoning}\nThe answer is \\boxed{{{final}}}."
            case PromptFormat.CHAT:
                response = f"{reasoning}\nThe answer is {final}."
        out.append({"prompt": row["question"], "response": response})
    return out


def _sft_example(messages: list[Message], tokenizer: Tokenizer, fmt: PromptFormat) -> Example:
    if fmt is PromptFormat.CHAT:
        return tokenize_chat(messages, tokenizer)
    roles = [m["role"] for m in messages]
    if roles != ["user", "assistant"]:
        raise ValueError(f"prompt format {fmt} needs one user and one assistant turn, got roles {roles}")
    if tokenizer.eos_token_id is None:
        raise ValueError("raw prompt formats end responses with EOS; the tokenizer has none")
    prompt = tokenizer.encode(format_prompt(messages[0]["content"], fmt))
    response = tokenizer.encode(messages[1]["content"]) + [tokenizer.eos_token_id]
    return prompt + response, [0] * len(prompt) + [1] * len(response)


class _ShuffledSlices:
    """Rank-local example indices for a step; each epoch is an independent Feistel permutation of the examples."""

    def __init__(self, num_examples: int, global_batch_size: int, dp_rank: int, dp_size: int, seed: int):
        if num_examples < 1:
            raise ValueError("no examples")
        if global_batch_size % dp_size:
            raise ValueError(f"global_batch_size {global_batch_size} is not divisible by dp_size {dp_size}")
        if not 0 <= dp_rank < dp_size:
            raise ValueError(f"dp_rank {dp_rank} outside [0, {dp_size})")
        self.num_examples = num_examples
        self.global_batch_size = global_batch_size
        self.local_batch_size = global_batch_size // dp_size
        self.dp_rank = dp_rank
        self.seed = seed
        self._perms: dict[int, FeistelPermutation] = {}

    def example_index(self, g: int) -> int:
        epoch, position = divmod(g, self.num_examples)
        if epoch not in self._perms:
            if len(self._perms) > 2:
                self._perms.clear()
            self._perms[epoch] = FeistelPermutation(self.num_examples, derive_seed(self.seed, epoch))
        return self._perms[epoch](position)

    def indices(self, step: int) -> list[int]:
        first = step * self.global_batch_size + self.dp_rank * self.local_batch_size
        return [self.example_index(g) for g in range(first, first + self.local_batch_size)]

    def state_dict(self) -> dict[str, int]:
        return {"num_examples": self.num_examples, "seed": self.seed}

    def load_state_dict(self, state: dict[str, int]) -> None:
        # Batches are a pure function of the step; the state only guards against resuming on different data.
        if state != self.state_dict():
            raise ValueError(f"resuming with a different dataset or seed: saved {state}, current {self.state_dict()}")


def _shifted(examples: Sequence[Example], width: int, pad_token_id: int) -> tuple[Tensor, Tensor]:
    """``input_ids = ids[:-1]`` and ``labels = ids[1:]`` (``IGNORE_INDEX`` where the mask is 0), padded to ``width``."""
    input_ids = torch.full((len(examples), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, mask) in enumerate(examples):
        n = len(ids) - 1
        input_ids[i, :n] = torch.tensor(ids[:-1], dtype=torch.long)
        targets = torch.tensor(ids[1:], dtype=torch.long)
        labels[i, :n] = torch.where(torch.tensor(mask[1:], dtype=torch.bool), targets, IGNORE_INDEX)
    return input_ids, labels


def _truncate(example: Example, max_tokens: int) -> Example | None:
    """First ``max_tokens`` tokens, or None if no trainable target remains (position 0 is never a target)."""
    ids, mask = example[0][:max_tokens], example[1][:max_tokens]
    return (ids, mask) if any(mask[1:]) else None


class SFTSource:
    """Batch: ``input_ids``, ``labels`` (``IGNORE_INDEX`` outside assistant turns), ``loss_mask`` — ``[G / dp, T]``
    with ``T <= seq_len`` the longest row of the slice."""

    def __init__(
        self,
        examples: Sequence[Example],
        seq_len: int,
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
        pad_token_id: int = 0,
    ):
        self.examples = [e for e in (_truncate(e, seq_len + 1) for e in examples) if e is not None]
        if dropped := len(examples) - len(self.examples):
            logger.warning(
                "dropped %d/%d chats with no assistant tokens within %d tokens", dropped, len(examples), seq_len
            )
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self._slices = _ShuffledSlices(len(self.examples), global_batch_size, dp_rank, dp_size, seed)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Row],
        tokenizer: Tokenizer,
        seq_len: int,
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
        fmt: PromptFormat = PromptFormat.CHAT,
    ) -> SFTSource:
        """``CHAT`` renders whole conversations as ChatML; the raw formats render the single user turn with
        :func:`format_prompt` and train on the assistant text followed by EOS, matching how rollouts are scored."""
        examples = [_sft_example(messages_from_row(r), tokenizer, PromptFormat(fmt)) for r in rows]
        return cls(examples, seq_len, global_batch_size, dp_rank, dp_size, seed)

    def batch(self, step: int) -> Batch:
        rows = [self.examples[i] for i in self._slices.indices(step)]
        input_ids, labels = _shifted(rows, max(len(ids) for ids, _ in rows) - 1, self.pad_token_id)
        return {"input_ids": input_ids, "labels": labels, "loss_mask": labels != IGNORE_INDEX}

    def state_dict(self) -> dict[str, int]:
        return self._slices.state_dict()

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._slices.load_state_dict(state)


def _preference_conversations(row: Row) -> tuple[list[Message], list[Message]]:
    prompt = row["prompt"]
    context = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    return (
        [*context, {"role": "assistant", "content": row["chosen"]}],
        [*context, {"role": "assistant", "content": row["rejected"]}],
    )


class PreferenceSource:
    """Batch: ``chosen_input_ids`` / ``chosen_labels`` / ``rejected_input_ids`` / ``rejected_labels``, all
    ``[G / dp, T]`` with one ``T`` so a loss can run chosen and rejected through the model as one ``[2B, T]`` batch.
    Labels are ``IGNORE_INDEX`` outside the final assistant response."""

    def __init__(
        self,
        pairs: Sequence[tuple[Example, Example]],
        seq_len: int,
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
        pad_token_id: int = 0,
    ):
        self.pairs = []
        for chosen, rejected in pairs:
            c, r = _truncate(chosen, seq_len + 1), _truncate(rejected, seq_len + 1)
            if c is not None and r is not None:
                self.pairs.append((c, r))
        if dropped := len(pairs) - len(self.pairs):
            logger.warning(
                "dropped %d/%d preference pairs with an empty response within %d tokens", dropped, len(pairs), seq_len
            )
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self._slices = _ShuffledSlices(len(self.pairs), global_batch_size, dp_rank, dp_size, seed)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Row],
        tokenizer: Tokenizer,
        seq_len: int,
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
    ) -> PreferenceSource:
        """Rows ``{"prompt": str | messages, "chosen": str, "rejected": str}``; only the final turn is trained."""
        pairs = []
        for row in rows:
            chosen, rejected = _preference_conversations(row)
            c_ids, c_mask = tokenize_chat(chosen, tokenizer)
            r_ids, r_mask = tokenize_chat(rejected, tokenizer)
            pairs.append((_last_turn_only(c_ids, c_mask), _last_turn_only(r_ids, r_mask)))
        return cls(pairs, seq_len, global_batch_size, dp_rank, dp_size, seed)

    def batch(self, step: int) -> Batch:
        rows = [self.pairs[i] for i in self._slices.indices(step)]
        width = max(max(len(c[0]), len(r[0])) for c, r in rows) - 1
        chosen_ids, chosen_labels = _shifted([c for c, _ in rows], width, self.pad_token_id)
        rejected_ids, rejected_labels = _shifted([r for _, r in rows], width, self.pad_token_id)
        return {
            "chosen_input_ids": chosen_ids,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_ids,
            "rejected_labels": rejected_labels,
        }

    def state_dict(self) -> dict[str, int]:
        return self._slices.state_dict()

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._slices.load_state_dict(state)


def _last_turn_only(ids: list[int], mask: list[int]) -> Example:
    """Keep the loss mask only on the last contiguous run of trained tokens (the final assistant turn)."""
    out = [0] * len(mask)
    i = len(mask) - 1
    while i >= 0 and not mask[i]:
        i -= 1
    while i >= 0 and mask[i]:
        out[i] = 1
        i -= 1
    return ids, out


class PromptSource:
    """RL prompts. Batch: ``prompt_ids`` ``[G / dp, P]`` right-padded, ``prompt_lens`` and ``example_index``
    ``[G / dp]``; :meth:`prompt` and :meth:`ground_truth` look rows up by ``example_index``.
    ``global_batch_size`` counts prompts (each later expands into a group of rollouts)."""

    def __init__(
        self,
        prompts: Sequence[list[int]],
        ground_truths: Sequence[str],
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
        pad_token_id: int = 0,
    ):
        if len(prompts) != len(ground_truths):
            raise ValueError(f"{len(prompts)} prompts vs {len(ground_truths)} ground truths")
        self.prompts = [list(p) for p in prompts]
        self.ground_truths = list(ground_truths)
        self.pad_token_id = pad_token_id
        self._slices = _ShuffledSlices(len(self.prompts), global_batch_size, dp_rank, dp_size, seed)

    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Row],
        tokenizer: Tokenizer,
        fmt: PromptFormat,
        max_prompt_len: int,
        global_batch_size: int,
        dp_rank: int,
        dp_size: int,
        seed: int,
    ) -> PromptSource:
        """Rows ``{"question", "ground_truth"}``; prompts longer than ``max_prompt_len`` tokens are dropped."""
        prompts, truths = [], []
        for row in rows:
            ids = tokenizer.encode(format_prompt(row["question"], fmt))
            if len(ids) <= max_prompt_len:
                prompts.append(ids)
                truths.append(str(row["ground_truth"]))
        if dropped := len(rows) - len(prompts):
            logger.warning("dropped %d/%d prompts longer than %d tokens", dropped, len(rows), max_prompt_len)
        return cls(prompts, truths, global_batch_size, dp_rank, dp_size, seed)

    def prompt(self, index: int) -> list[int]:
        return self.prompts[index]

    def ground_truth(self, index: int) -> str:
        return self.ground_truths[index]

    def batch(self, step: int) -> Batch:
        indices = self._slices.indices(step)
        rows = [self.prompts[i] for i in indices]
        prompt_ids = torch.full((len(rows), max(len(r) for r in rows)), self.pad_token_id, dtype=torch.long)
        for i, row in enumerate(rows):
            prompt_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        return {
            "prompt_ids": prompt_ids,
            "prompt_lens": torch.tensor([len(r) for r in rows], dtype=torch.long),
            "example_index": torch.tensor(indices, dtype=torch.long),
        }

    def state_dict(self) -> dict[str, int]:
        return self._slices.state_dict()

    def load_state_dict(self, state: dict[str, int]) -> None:
        self._slices.load_state_dict(state)
