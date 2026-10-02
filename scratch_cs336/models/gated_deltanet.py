"""Gated DeltaNet token mixer (Qwen3.5 ``linear_attention`` layers).

Per value head, with state ``S [dk, dv]``, l2-normalized ``q_t, k_t`` and ``q_t`` scaled by ``dk^-1/2``::

    S_t = exp(g_t) * S_{t-1}
    S_t = S_t + beta_t * k_t (v_t - S_t^T k_t)^T
    o_t = S_t^T q_t

Layout differs from HF on purpose: q, k, v have separate projections and separate depthwise convs (HF fuses them
into ``in_proj_qkv`` / one ``conv1d``), and ``A_log`` / ``dt_bias`` live in their own ``decay`` module, so every
per-head parameter sits in a module that a tensor-parallel plan can shard by heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_cs336.models.config import ModelConfig
from scratch_cs336.models.layers import Linear

CHUNK_SIZE = 64


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def chunk_gated_delta_rule(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor, chunk_size: int = CHUNK_SIZE
) -> Tensor:
    """Chunked gated delta rule (WY / UT-transform form), fp32 math, zero initial state.

    ``q, k [B, T, H, dk]`` (already normalized and scaled), ``v [B, T, H, dv]``, ``g`` (log decay, <= 0) and
    ``beta`` ``[B, T, H]``. Returns ``o [B, T, H, dv]`` in fp32.
    """
    seq = q.shape[1]
    q, k, v, g, beta = (t.transpose(1, 2).float() for t in (q, k, v, g, beta))
    pad = (-seq) % chunk_size
    if pad:
        # Zero-padding at the end cannot influence earlier (causal) outputs.
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
        g, beta = (F.pad(t, (0, pad)) for t in (g, beta))
    batch, heads, padded, dk = k.shape
    dv = v.shape[-1]
    n = padded // chunk_size
    q, k, v = (t.reshape(batch, heads, n, chunk_size, -1) for t in (q, k, v))
    g, beta = (t.reshape(batch, heads, n, chunk_size) for t in (g, beta))

    cum = g.cumsum(-1)  # log decay from chunk start through position i
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril()
    pair_decay = (cum[..., :, None] - cum[..., None, :]).masked_fill(~causal, float("-inf")).exp()

    k_beta = k * beta[..., None]
    # Within a chunk the delta rule is the unit lower-triangular system (I + tril(beta K K^T ⊙ D, -1)) X = RHS;
    # unitriangular=True ignores the diagonal of the matrix we pass.
    system = (k_beta @ k.transpose(-1, -2)) * pair_decay
    u = torch.linalg.solve_triangular(system, v * beta[..., None], upper=False, unitriangular=True)
    w = torch.linalg.solve_triangular(system, k_beta * cum.exp()[..., None], upper=False, unitriangular=True)
    intra = (q @ k.transpose(-1, -2)) * pair_decay

    q_read = q * cum.exp()[..., None]
    k_write = k * (cum[..., -1:] - cum).exp()[..., None]
    chunk_decay = cum[..., -1].exp()[..., None, None]

    state = q.new_zeros(batch, heads, dk, dv)
    outs = []
    for i in range(n):
        v_new = u[:, :, i] - w[:, :, i] @ state
        outs.append(q_read[:, :, i] @ state + intra[:, :, i] @ v_new)
        state = state * chunk_decay[:, :, i] + k_write[:, :, i].transpose(-1, -2) @ v_new
    out = torch.stack(outs, dim=2).reshape(batch, heads, padded, dv)[:, :, :seq]
    return out.transpose(1, 2)


def recurrent_gated_delta_rule(q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> Tensor:
    """Token-by-token reference for :func:`chunk_gated_delta_rule` (same contract)."""
    q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
    batch, seq, heads, dk = k.shape
    state = q.new_zeros(batch, heads, dk, v.shape[-1])
    outs = []
    for t in range(seq):
        state = state * g[:, t].exp()[..., None, None]
        k_t = k[:, t]
        delta = (v[:, t] - torch.einsum("bhkv,bhk->bhv", state, k_t)) * beta[:, t, :, None]
        state = state + k_t[..., :, None] * delta[..., None, :]
        outs.append(torch.einsum("bhkv,bhk->bhv", state, q[:, t]))
    return torch.stack(outs, dim=1)


class CausalDepthwiseConv(nn.Module):
    """Depthwise causal conv + SiLU over ``[B, T, C]``; ``weight [C, kernel]``."""

    def __init__(self, channels: int, kernel: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, kernel))

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        bound = 1.0 / math.sqrt(self.weight.shape[1])
        nn.init.uniform_(self.weight, -bound, bound, generator=generator)

    def forward(self, x: Tensor) -> Tensor:
        channels, kernel = self.weight.shape
        y = F.conv1d(x.transpose(1, 2), self.weight.unsqueeze(1), padding=kernel - 1, groups=channels)
        return F.silu(y[..., : x.shape[1]]).transpose(1, 2)


class DeltaDecay(nn.Module):
    """Per-value-head log decay ``g = -exp(A_log) * softplus(a + dt_bias)`` (Mamba-2 style discretization)."""

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(n_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_heads))

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(0.01, 16, generator=generator).log())
        nn.init.ones_(self.dt_bias)

    def forward(self, a: Tensor) -> Tensor:
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)


class GatedRMSNorm(nn.Module):
    """``w * rmsnorm(x) * silu(gate)`` over the head dim (weight shared by all heads)."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))

    def reset_parameters(self, generator: torch.Generator | None = None) -> None:
        nn.init.ones_(self.weight)

    def forward(self, x: Tensor, gate: Tensor) -> Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(x.dtype)
        return (h * F.silu(gate.float())).to(x.dtype)


class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.k_head_dim = cfg.linear_k_head_dim
        self.v_head_dim = cfg.linear_v_head_dim
        key_dim = cfg.linear_n_k_heads * cfg.linear_k_head_dim
        value_dim = cfg.linear_n_v_heads * cfg.linear_v_head_dim
        self.q_proj = Linear(cfg.d_model, key_dim)
        self.k_proj = Linear(cfg.d_model, key_dim)
        self.v_proj = Linear(cfg.d_model, value_dim)
        self.z_proj = Linear(cfg.d_model, value_dim)
        self.a_proj = Linear(cfg.d_model, cfg.linear_n_v_heads)
        self.b_proj = Linear(cfg.d_model, cfg.linear_n_v_heads)
        self.q_conv = CausalDepthwiseConv(key_dim, cfg.linear_conv_kernel)
        self.k_conv = CausalDepthwiseConv(key_dim, cfg.linear_conv_kernel)
        self.v_conv = CausalDepthwiseConv(value_dim, cfg.linear_conv_kernel)
        self.decay = DeltaDecay(cfg.linear_n_v_heads)
        self.norm = GatedRMSNorm(cfg.linear_v_head_dim, cfg.norm_eps)
        self.out_proj = Linear(value_dim, cfg.d_model)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        dk, dv = self.k_head_dim, self.v_head_dim
        # Head counts come from (possibly tensor-parallel sharded) weight shapes.
        q = self.q_conv(self.q_proj(x)).view(batch, seq, -1, dk)
        k = self.k_conv(self.k_proj(x)).view(batch, seq, -1, dk)
        v = self.v_conv(self.v_proj(x)).view(batch, seq, -1, dv)
        beta = torch.sigmoid(self.b_proj(x))
        g = self.decay(self.a_proj(x))
        # Value head j reads key head j // ratio, so contiguous head shards stay self-contained.
        ratio = v.shape[2] // k.shape[2]
        if ratio > 1:
            q, k = q.repeat_interleave(ratio, dim=2), k.repeat_interleave(ratio, dim=2)
        q = l2norm(q.float()) * dk**-0.5
        k = l2norm(k.float())
        core = chunk_gated_delta_rule(q, k, v, g, beta).to(x.dtype)
        z = self.z_proj(x).view(batch, seq, -1, dv)
        return self.out_proj(self.norm(core, z).reshape(batch, seq, -1))
