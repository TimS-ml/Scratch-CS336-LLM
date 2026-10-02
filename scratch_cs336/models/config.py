"""Model configuration shared by CS336 basics (Llama-style), Qwen3 and Qwen3.5 text models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class LayerType(StrEnum):
    ATTENTION = "full_attention"
    GATED_DELTANET = "linear_attention"


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    max_seq_len: int
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 1.0
    norm_eps: float = 1e-5
    qk_norm: bool = False
    attn_output_gate: bool = False
    zero_centered_norm: bool = False
    tie_embeddings: bool = False
    # None means every layer is full attention.
    layer_types: tuple[LayerType, ...] | None = None
    linear_n_k_heads: int = 0
    linear_n_v_heads: int = 0
    linear_k_head_dim: int = 0
    linear_v_head_dim: int = 0
    linear_conv_kernel: int = 0

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"n_heads={self.n_heads} must be a multiple of n_kv_heads={self.n_kv_heads}")
        if self.rotary_dim % 2:
            raise ValueError(f"rotary dim {self.rotary_dim} must be even")
        if self.layer_types is not None:
            if len(self.layer_types) != self.n_layers:
                raise ValueError(f"len(layer_types)={len(self.layer_types)} != n_layers={self.n_layers}")
            # Normalize so configs parsed from yaml/json (plain strings) compare and hash equal to typed ones.
            object.__setattr__(self, "layer_types", tuple(LayerType(t) for t in self.layer_types))
        if self.has_gated_deltanet:
            dims = (self.linear_n_k_heads, self.linear_n_v_heads, self.linear_k_head_dim, self.linear_v_head_dim)
            if min(*dims, self.linear_conv_kernel) <= 0:
                raise ValueError("Gated DeltaNet layers need positive linear_* dims")
            if self.linear_n_v_heads % self.linear_n_k_heads:
                raise ValueError("linear_n_v_heads must be a multiple of linear_n_k_heads")

    @property
    def resolved_layer_types(self) -> tuple[LayerType, ...]:
        return self.layer_types or (LayerType.ATTENTION,) * self.n_layers

    @property
    def has_gated_deltanet(self) -> bool:
        return LayerType.GATED_DELTANET in self.resolved_layer_types

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    def check_tensor_parallel(self, tp: int) -> None:
        """Tensor parallelism shards whole heads and the MLP hidden dim, so ``tp`` must divide each of them."""
        sizes = {"d_ff": self.d_ff}
        if LayerType.ATTENTION in self.resolved_layer_types:
            sizes |= {"n_heads": self.n_heads, "n_kv_heads": self.n_kv_heads}
        if self.has_gated_deltanet:
            sizes |= {"linear_n_k_heads": self.linear_n_k_heads, "linear_n_v_heads": self.linear_n_v_heads}
        bad = [f"{name}={size}" for name, size in sizes.items() if size % tp]
        if bad:
            g = math.gcd(*sizes.values())
            supported = ", ".join(str(t) for t in range(1, g + 1) if g % t == 0)
            raise ValueError(
                f"tensor={tp} must divide {', '.join(bad)} (tensor parallelism shards whole heads); "
                f"this model supports tensor in {{{supported}}}"
            )
