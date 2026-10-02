"""Building blocks: Linear, Embedding, RMSNorm, rotary embedding, SwiGLU.

Parameters are allocated uninitialized; ``reset_parameters(generator)`` fills them so that a model can be built on
the meta device and initialized deterministically in one pass (``TransformerLM.init_weights``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def trunc_normal_(weight: Tensor, std: float, generator: torch.Generator | None) -> None:
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator)


class Linear(nn.Module):
    """Bias-free linear layer, ``weight [d_out, d_in]``."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        d_out, d_in = self.weight.shape
        trunc_normal_(self.weight, math.sqrt(2.0 / (d_in + d_out)), generator)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        trunc_normal_(self.weight, 1.0, generator)

    def forward(self, ids: Tensor) -> Tensor:
        return F.embedding(ids, self.weight)


class RMSNorm(nn.Module):
    """RMSNorm over the last dim, normalized in fp32.

    ``zero_centered=False``: ``w * x̂`` with ``w`` init 1 (Llama/Qwen3: x̂ is cast back before the multiply).
    ``zero_centered=True``: ``(1 + w) * x̂`` with ``w`` init 0 (Qwen3.5: the multiply happens in fp32).
    """

    def __init__(self, dim: int, eps: float, zero_centered: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.zero_centered = zero_centered
        self.weight = nn.Parameter(torch.empty(dim))

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        nn.init.constant_(self.weight, 0.0 if self.zero_centered else 1.0)

    def forward(self, x: Tensor) -> Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.zero_centered:
            return (h * (1.0 + self.weight.float())).to(x.dtype)
        return self.weight * h.to(x.dtype)


def rotary_cos_sin(position_ids: Tensor, rotary_dim: int, theta: float, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """``cos, sin [B, T, rotary_dim]`` for half-split (``rotate_half``) rotary embedding, computed in fp32."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device=position_ids.device) / rotary_dim)
    )
    freqs = position_ids.float()[..., None] * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the first ``cos.shape[-1]`` features of ``x [B, H, T, head_dim]``; the rest pass through."""
    rot = cos.shape[-1]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x1, x2 = x_rot[..., : rot // 2], x_rot[..., rot // 2 :]
    rotated = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat((rotated, x_pass), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff)
        self.up_proj = Linear(d_model, d_ff)
        self.down_proj = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
