"""Pre-norm decoder-only LM: CS336 basics / Qwen3 (all attention) and Qwen3.5 (hybrid Gated DeltaNet)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_cs336.models.attention import Attention
from scratch_cs336.models.config import LayerType, ModelConfig
from scratch_cs336.models.gated_deltanet import GatedDeltaNet
from scratch_cs336.models.layers import Embedding, Linear, RMSNorm, SwiGLU, rotary_cos_sin, trunc_normal_
from scratch_cs336.parallel.plan import TPStyle


class Block(nn.Module):
    """``x + mixer(norm(x))`` then ``x + mlp(norm(x))``; the mixer is ``attn`` or ``linear_attn``."""

    def __init__(self, cfg: ModelConfig, layer_type: LayerType) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.zero_centered_norm)
        match layer_type:
            case LayerType.ATTENTION:
                self.attn = Attention(cfg)
            case LayerType.GATED_DELTANET:
                self.linear_attn = GatedDeltaNet(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.zero_centered_norm)
        self.mlp = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        h = self.attn_norm(x)
        if self.layer_type is LayerType.ATTENTION:
            x = x + self.attn(h, cos, sin)
        else:
            x = x + self.linear_attn(h)
        return x + self.mlp(self.mlp_norm(x))


class TransformerLM(nn.Module):
    """Parameters are uninitialized after construction; call :meth:`init_weights` or load a state dict."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, t) for t in cfg.resolved_layer_types)
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.zero_centered_norm)
        if not cfg.tie_embeddings:
            self.lm_head = Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, input_ids: Tensor, position_ids: Tensor | None = None) -> Tensor:
        return self.logits(self.hidden_states(input_ids, position_ids))

    def hidden_states(self, input_ids: Tensor, position_ids: Tensor | None = None) -> Tensor:
        """Final-normed hidden states ``[B, T, d_model]``; lets callers project only the positions they need."""
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        x = self.embed(input_ids)
        cos, sin = rotary_cos_sin(position_ids, self.cfg.rotary_dim, self.cfg.rope_theta, x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.final_norm(x)

    def logits(self, hidden: Tensor) -> Tensor:
        head = self.embed.weight if self.cfg.tie_embeddings else self.lm_head.weight
        return F.linear(hidden, head)

    @torch.no_grad()
    def init_weights(self, seed: int) -> None:
        generator = torch.Generator(device=self.embed.weight.device).manual_seed(seed)
        for module in self.modules():
            if module is not self and hasattr(module, "reset_parameters"):
                module.reset_parameters(generator)
        if self.cfg.tie_embeddings:
            # A tied table is also the output projection: std 1 would give logits of scale sqrt(d_model).
            trunc_normal_(self.embed.weight, math.sqrt(2.0 / (self.cfg.vocab_size + self.cfg.d_model)), generator)

    def num_params(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if exclude_embedding else n

    def flops_per_token(self, seq_len: int) -> float:
        """Training FLOPs per token: ``6 N`` for every weight matmul (incl. the output head, tied or not) plus
        the sequence-mixing terms (PaLM convention, no causal discount): ``12 * H * hd * T`` per attention layer
        and, per Gated DeltaNet layer, chunked state read/write ``~18 * Hv * dk * dv`` plus intra-chunk
        attention ``12 * Hv * (dk + dv) / 2 * chunk``."""
        cfg = self.cfg
        matmul_params = self.num_params(exclude_embedding=True) + cfg.vocab_size * cfg.d_model
        flops = 6.0 * matmul_params
        for layer_type in cfg.resolved_layer_types:
            if layer_type is LayerType.ATTENTION:
                flops += 12.0 * cfg.n_heads * cfg.head_dim * seq_len
            else:
                hv, dk, dv = cfg.linear_n_v_heads, cfg.linear_k_head_dim, cfg.linear_v_head_dim
                chunk = min(seq_len, 64)
                flops += 18.0 * hv * dk * dv + 6.0 * hv * (dk + dv) * chunk
        return flops

    def tp_plan(self) -> dict[str, TPStyle]:
        """fnmatch patterns over module FQNs. REPLICATE marks replicated modules that run on head-sharded
        activations inside the TP region (their grads are summed over ``tp``); unlisted modules are replicated
        and used outside it."""
        plan = {
            "blocks.*.mlp.gate_proj": TPStyle.COLWISE,
            "blocks.*.mlp.up_proj": TPStyle.COLWISE,
            "blocks.*.mlp.down_proj": TPStyle.ROWWISE,
        }
        layer_types = set(self.cfg.resolved_layer_types)
        if LayerType.ATTENTION in layer_types:
            plan |= {
                "blocks.*.attn.q_proj": TPStyle.COLWISE,
                "blocks.*.attn.k_proj": TPStyle.COLWISE,
                "blocks.*.attn.v_proj": TPStyle.COLWISE,
                "blocks.*.attn.o_proj": TPStyle.ROWWISE,
            }
            if self.cfg.qk_norm:
                plan |= {"blocks.*.attn.q_norm": TPStyle.REPLICATE, "blocks.*.attn.k_norm": TPStyle.REPLICATE}
        if LayerType.GATED_DELTANET in layer_types:
            plan |= {f"blocks.*.linear_attn.{p}_proj": TPStyle.COLWISE for p in "qkvzab"}
            plan |= {f"blocks.*.linear_attn.{p}_conv": TPStyle.HEADWISE for p in "qkv"}
            plan |= {
                "blocks.*.linear_attn.decay": TPStyle.HEADWISE,
                "blocks.*.linear_attn.out_proj": TPStyle.ROWWISE,
                "blocks.*.linear_attn.norm": TPStyle.REPLICATE,
            }
        return plan
