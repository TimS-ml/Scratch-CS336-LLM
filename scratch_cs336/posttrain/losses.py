"""Post-training objectives as plain tensor functions (CS336 hw5 semantics): log-probs, entropy, group-normalized
advantages, policy-gradient losses with importance reweighting (GRPO / GSPO), loss aggregation, DPO."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor

IGNORE_INDEX = -100

RewardFn = Callable[[str, str], dict[str, float]]  # (response, ground_truth) -> {"reward", "format_reward", ...}


class Baseline(StrEnum):
    MEAN = "mean"  # subtract the group mean reward
    NONE = "none"


class AdvantageNormalizer(StrEnum):
    STD = "std"  # GRPO
    NONE = "none"  # Dr. GRPO, RFT
    MEAN = "mean"  # MaxRL


class ImportanceReweighting(StrEnum):
    NONE = "none"  # on-policy REINFORCE with advantages
    NOCLIP = "noclip"  # token-level ratio, no clipping
    GRPO = "grpo"  # token-level PPO/GRPO clipped ratio
    GSPO = "gspo"  # sequence-level (geometric-mean) clipped ratio


class LossNormalization(StrEnum):
    SEQUENCE = "sequence"  # mean over each sequence's response tokens, then mean over sequences
    CONSTANT = "constant"  # sum over all response tokens divided by a fixed constant


def token_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """``log p(labels[b, t] | logits[b, t])`` in fp32, shape ``[B, T]``; positions with ``labels < 0`` are 0."""
    logits = logits.float()
    valid = labels >= 0
    picked = logits.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (picked - torch.logsumexp(logits, dim=-1)) * valid


def token_entropy(logits: Tensor) -> Tensor:
    """Entropy of the next-token distribution at every position, shape ``logits.shape[:-1]``."""
    logits = logits.float()
    return torch.logsumexp(logits, dim=-1) - (torch.softmax(logits, dim=-1) * logits).sum(-1)


def masked_mean(tensor: Tensor, mask: Tensor, dim: int | None = None) -> Tensor:
    """Mean over elements where ``mask`` is set (all of them when ``dim is None``, else along ``dim``)."""
    mask = mask.to(tensor.dtype)
    if dim is None:
        return (tensor * mask).sum() / mask.sum()
    return (tensor * mask).sum(dim) / mask.sum(dim)


def masked_normalize(tensor: Tensor, mask: Tensor, normalize_constant: float, dim: int | None = None) -> Tensor:
    """Sum over masked elements divided by ``normalize_constant``."""
    masked = tensor * mask.to(tensor.dtype)
    return (masked.sum() if dim is None else masked.sum(dim)) / normalize_constant


def compute_rollout_rewards(
    reward_fn: RewardFn, rollout_responses: list[str], repeated_ground_truths: list[str]
) -> tuple[Tensor, dict[str, float]]:
    """Raw rewards ``[n]`` plus the batch means of every reward component."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(f"{len(rollout_responses)} responses vs {len(repeated_ground_truths)} ground truths")
    scores = [reward_fn(r, gt) for r, gt in zip(rollout_responses, repeated_ground_truths, strict=True)]
    rewards = torch.tensor([s["reward"] for s in scores], dtype=torch.float32)
    metadata = {f"{key}_mean": sum(s[key] for s in scores) / len(scores) for key in scores[0]} if scores else {}
    return rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: Tensor,
    group_size: int,
    baseline: Baseline = Baseline.MEAN,
    advantage_eps: float = 1e-6,
    advantage_normalizer: AdvantageNormalizer = AdvantageNormalizer.STD,
) -> tuple[Tensor, dict[str, float]]:
    """Per-rollout advantages from rewards laid out as consecutive groups of ``group_size`` (same prompt).

    ``std`` divides by the unbiased group std + eps (GRPO), ``mean`` by the group mean + eps (MaxRL).
    """
    baseline, advantage_normalizer = Baseline(baseline), AdvantageNormalizer(advantage_normalizer)
    if raw_rewards.numel() % group_size:
        raise ValueError(f"{raw_rewards.numel()} rewards do not form groups of {group_size}")
    grouped = raw_rewards.float().reshape(-1, group_size)
    mean = grouped.mean(-1, keepdim=True)
    advantages = grouped - mean if baseline is Baseline.MEAN else grouped
    match advantage_normalizer:
        case AdvantageNormalizer.STD:
            advantages = advantages / (grouped.std(-1, keepdim=True) + advantage_eps)
        case AdvantageNormalizer.MEAN:
            advantages = advantages / (mean + advantage_eps)
        case AdvantageNormalizer.NONE:
            pass
    metadata = {
        "reward_mean": grouped.mean().item(),
        "reward_std": grouped.std().item() if grouped.numel() > 1 else 0.0,
        "reward_max": grouped.max().item(),
        "reward_min": grouped.min().item(),
        "group_reward_std_mean": grouped.std(-1).mean().item() if group_size > 1 else 0.0,
        "frac_zero_advantage": (advantages == 0).float().mean().item(),
    }
    return advantages.reshape(-1), metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: Tensor,
    policy_log_probs: Tensor,
    importance_reweighting_method: ImportanceReweighting = ImportanceReweighting.NONE,
    old_log_probs: Tensor | None = None,
    cliprange: float | None = None,
    response_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-token loss ``[B, T]`` = negative (possibly importance-weighted, clipped) objective.

    ``grpo``: ``-min(w A, clip(w, 1±eps) A)`` with token ratio ``w = exp(logp - old)``.
    ``gspo``: same with the sequence ratio ``s = exp(mean_t (logp - old))`` over ``response_mask`` tokens,
    broadcast to every position. Metadata holds a per-token ``clipped`` 0/1 tensor for the clipped methods.
    """
    method = ImportanceReweighting(importance_reweighting_method)
    advantages = raw_rewards_or_advantages.reshape(-1, 1).to(policy_log_probs.dtype)
    if method is ImportanceReweighting.NONE:
        return -advantages * policy_log_probs, {}
    if old_log_probs is None:
        raise ValueError(f"importance_reweighting_method={method} requires old_log_probs")
    log_ratio = policy_log_probs - old_log_probs
    if method is ImportanceReweighting.NOCLIP:
        return -torch.exp(log_ratio) * advantages, {}
    if cliprange is None:
        raise ValueError(f"importance_reweighting_method={method} requires cliprange")
    if method is ImportanceReweighting.GSPO:
        if response_mask is None:
            raise ValueError("gspo requires response_mask")
        # exp of the masked mean log-ratio is the geometric mean of token ratios, without overflow.
        log_ratio = masked_mean(log_ratio, response_mask, dim=-1).unsqueeze(-1).expand_as(policy_log_probs)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - cliprange, 1.0 + cliprange) * advantages
    loss = -torch.minimum(unclipped, clipped)
    return loss, {"clipped": (clipped < unclipped).float()}


def aggregate_loss_across_microbatch(
    per_token_policy_gradient_loss: Tensor,
    mask: Tensor,
    loss_normalization: LossNormalization = LossNormalization.SEQUENCE,
    normalization_constant: float | None = None,
) -> Tensor:
    """Scalar loss: mean of per-sequence masked means (``sequence``) or masked sum / constant (``constant``)."""
    match LossNormalization(loss_normalization):
        case LossNormalization.SEQUENCE:
            return masked_mean(per_token_policy_gradient_loss, mask, dim=-1).mean()
        case LossNormalization.CONSTANT:
            if normalization_constant is None:
                raise ValueError("loss_normalization='constant' requires normalization_constant")
            return masked_normalize(per_token_policy_gradient_loss, mask, normalization_constant)


def sequence_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    """Summed log-prob of every sequence's labelled tokens (``labels >= 0``), shape ``[B]``."""
    return token_log_probs(logits, labels).sum(-1)


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    ref_chosen_logps: Tensor,
    ref_rejected_logps: Tensor,
    beta: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Per-pair DPO loss ``-log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))`` on sequence log-probs.

    Metadata: implicit rewards ``beta * (pi - ref)`` for chosen / rejected (detached), per pair.
    """
    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps)
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss, {"chosen_reward": chosen_reward.detach(), "rejected_reward": rejected_reward.detach()}
