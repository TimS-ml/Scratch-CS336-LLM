"""Causal grouped-query attention with optional QK-norm (Qwen3) and sigmoid output gate (Qwen3.5)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_cs336.models.config import ModelConfig
from scratch_cs336.models.layers import Linear, RMSNorm, apply_rotary


class Attention(nn.Module):
    """``q_proj`` rows are laid out per head as ``[query_h | gate_h]`` when gated (HF layout), so sharding any
    projection along dim 0 in whole heads keeps every head's parameters together."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.output_gate = cfg.attn_output_gate
        q_width = cfg.n_heads * cfg.head_dim * (2 if cfg.attn_output_gate else 1)
        self.q_proj = Linear(cfg.d_model, q_width)
        self.k_proj = Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim)
        self.v_proj = Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim)
        self.o_proj = Linear(cfg.n_heads * cfg.head_dim, cfg.d_model)
        if cfg.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, cfg.zero_centered_norm)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.norm_eps, cfg.zero_centered_norm)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, seq, _ = x.shape
        hd = self.head_dim
        # Head counts come from (possibly tensor-parallel sharded) weight shapes.
        q = self.q_proj(x).view(batch, seq, -1, hd * (2 if self.output_gate else 1))
        if self.output_gate:
            q, gate = q.chunk(2, dim=-1)
        k = self.k_proj(x).view(batch, seq, -1, hd)
        v = self.v_proj(x).view(batch, seq, -1, hd)
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=q.shape[1] != k.shape[1])
        out = out.transpose(1, 2).reshape(batch, seq, -1)
        if self.output_gate:
            out = out * torch.sigmoid(gate.reshape(batch, seq, -1))
        return self.o_proj(out)
