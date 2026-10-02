"""Parity with the CS336 hw5 (sp26) reference snapshots, using the hw5 fixtures' inputs."""

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from scratch_cs336.posttrain import losses as L
from scratch_cs336.posttrain.data import tokenize_prompt_and_output

FIXTURES = Path(__file__).parent.parent / "fixtures" / "posttrain"
WORDS = ["<pad>", "<eos>", "<unk>", "Hello", "world", "This", "is", "a", "test", "another", "Question", "Answer"]
WORDS += ["Instruction", "Response", "###"]


def assert_snapshot(name: str, actual: torch.Tensor | dict[str, torch.Tensor], atol: float = 1e-2) -> None:
    expected = dict(np.load(FIXTURES / f"{name}.npz"))
    actual = actual if isinstance(actual, dict) else {"array": actual}
    assert set(actual) == set(expected)
    for key, value in actual.items():
        np.testing.assert_allclose(value.detach().numpy(), expected[key], rtol=1e-4, atol=atol, err_msg=key)


class WordTokenizer:
    def __init__(self) -> None:
        self._tok = Tokenizer(WordLevel(vocab={w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
        self._tok.pre_tokenizer = Whitespace()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids


def _seeded(fn):
    torch.manual_seed(42)
    return fn()


@pytest.fixture
def log_probs():
    return _seeded(lambda: torch.randn(2, 10))


@pytest.fixture
def old_log_probs(log_probs):
    return _seeded(lambda: log_probs + torch.randn_like(log_probs))


@pytest.fixture
def rewards():
    return _seeded(lambda: torch.rand(2, 1))


@pytest.fixture
def response_mask(log_probs):
    return _seeded(lambda: torch.rand_like(log_probs) > 0.5)


def test_tokenize_prompt_and_output_masks_only_response_labels():
    prompts = ["Hello, world!", "This is a test.", "This is another test."]
    out = tokenize_prompt_and_output(prompts, prompts, WordTokenizer(), pad_token_id=0)
    assert_snapshot("test_tokenize_prompt_and_output", out, atol=0)


def test_entropy():
    logits = _seeded(lambda: torch.randn(2, 10, 100))
    assert_snapshot("test_compute_entropy", L.token_entropy(logits))


def test_log_probs_match_log_softmax_and_ignore_negative_labels():
    logits = _seeded(lambda: torch.randn(2, 5, 7) * 3)
    labels = torch.tensor([[1, 2, 3, -100, 0], [6, -100, 5, 4, 3]])
    reference = torch.log_softmax(logits, -1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    expected = torch.where(labels >= 0, reference, 0.0)
    torch.testing.assert_close(L.token_log_probs(logits, labels), expected)
    torch.testing.assert_close(L.sequence_log_probs(logits, labels), expected.sum(-1))


@pytest.mark.parametrize("dim", [0, 1, -1, None])
def test_masked_normalize(dim):
    tensor = _seeded(lambda: torch.randn(2, 10, 100))
    mask = _seeded(lambda: torch.rand_like(tensor) > 0.5)
    name = {0: "dim0", 1: "dim1", -1: "dimlast", None: "dimNone"}[dim]
    assert_snapshot(f"test_masked_normalize_{name}", L.masked_normalize(tensor, mask, 42.0, dim))


def test_rollout_rewards():
    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
        r = (int(hashlib.sha256(response.encode()).hexdigest(), 16) % 10) / 10.0
        return {"reward": r, "format_reward": r, "answer_reward": r}

    responses = [f"hmm I think ths answer is {i}" for i in range(8)]
    rewards, meta = L.compute_rollout_rewards(reward_fn, responses, ["42"] * 8)
    assert_snapshot("test_compute_rollout_rewards", rewards)
    assert meta["reward_mean"] == pytest.approx(rewards.mean().item())


@pytest.mark.parametrize(
    ("name", "baseline", "normalizer", "key"),
    [
        ("grpo", "mean", "std", "advantages"),
        ("drgrpo", "mean", "none", "advantages"),
        ("drgrpo", "none", "none", "no_baseline_advantages"),
        ("maxrl", "mean", "mean", "array"),
    ],
)
def test_group_normalized_rewards(name, baseline, normalizer, key):
    advantages, _ = L.compute_group_normalized_rewards(
        torch.tensor([1.0, 0.0, 0.0, 1.0]), 2, baseline, 1e-6, normalizer
    )
    expected = np.load(FIXTURES / f"test_compute_group_normalized_rewards_{name}.npz")[key]
    np.testing.assert_allclose(advantages.numpy(), expected, rtol=1e-4, atol=1e-6)


def test_policy_gradient_loss_on_policy(rewards, log_probs):
    loss, _ = L.compute_policy_gradient_loss(rewards, log_probs, "none")
    assert_snapshot("test_compute_policy_gradient_loss_on_policy", loss)


def test_policy_gradient_loss_off_policy(rewards, log_probs, old_log_probs):
    noclip, _ = L.compute_policy_gradient_loss(rewards, log_probs, "noclip", old_log_probs)
    clipped, meta = L.compute_policy_gradient_loss(rewards, log_probs, "grpo", old_log_probs, cliprange=0.1)
    assert_snapshot("test_compute_policy_gradient_loss_off_policy", {"noclip_loss": noclip, "clipped_loss": clipped})
    # Rewards are positive here, so exactly the tokens whose ratio exceeds 1 + eps are clipped.
    assert torch.equal(meta["clipped"].bool(), torch.exp(log_probs - old_log_probs) > 1.1)


def test_policy_gradient_loss_gspo(rewards, log_probs, old_log_probs, response_mask):
    loss, _ = L.compute_policy_gradient_loss(rewards, log_probs, "gspo", old_log_probs, 0.1, response_mask)
    assert_snapshot("test_compute_policy_gradient_loss_off_policy_gspo", loss)


def test_gspo_ignores_log_ratios_outside_the_response(rewards, log_probs, old_log_probs, response_mask):
    perturbed = torch.where(response_mask, log_probs, log_probs + 5.0)
    a, _ = L.compute_policy_gradient_loss(rewards, log_probs, "gspo", old_log_probs, 0.1, response_mask)
    b, _ = L.compute_policy_gradient_loss(rewards, perturbed, "gspo", old_log_probs, 0.1, response_mask)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize(("normalization", "constant"), [("sequence", None), ("constant", 42)])
def test_aggregate_loss(log_probs, response_mask, normalization, constant):
    loss = L.aggregate_loss_across_microbatch(log_probs, response_mask, normalization, constant)
    assert_snapshot(f"test_aggregate_loss_across_microbatch_{normalization}", loss)


def test_dpo_loss_hand_computed():
    beta = 0.5
    loss, meta = L.dpo_loss(
        torch.tensor([-3.0]), torch.tensor([-5.0]), torch.tensor([-4.0]), torch.tensor([-4.5]), beta
    )
    margin = beta * ((-3.0 + 4.0) - (-5.0 + 4.5))  # 0.75
    assert loss.item() == pytest.approx(math.log1p(math.exp(-margin)))
    assert meta["chosen_reward"].item() == pytest.approx(0.5)
    assert meta["rejected_reward"].item() == pytest.approx(-0.25)
