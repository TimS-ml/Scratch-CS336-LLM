"""Named model configurations."""

from __future__ import annotations

from scratch_cs336.models.config import LayerType, ModelConfig

_A, _L = LayerType.ATTENTION, LayerType.GATED_DELTANET

PRESETS: dict[str, ModelConfig] = {
    # Smoke-test size; pairs with a small trained BPE (vocab <= 512).
    "cs336-tiny": ModelConfig(
        vocab_size=512, d_model=64, n_layers=2, n_heads=4, n_kv_heads=4, head_dim=16, d_ff=192, max_seq_len=128
    ),
    # CS336 hw1 TinyStories model.
    "cs336-17m": ModelConfig(
        vocab_size=10000, d_model=512, n_layers=4, n_heads=16, n_kv_heads=16, head_dim=32, d_ff=1344, max_seq_len=256
    ),
    # CS336 hw4 data-filtering model (GPT-2 tokenizer).
    "cs336-hw4": ModelConfig(
        vocab_size=50257, d_model=1024, n_layers=24, n_heads=16, n_kv_heads=16, head_dim=64, d_ff=3072, max_seq_len=512
    ),
    "qwen3-0.6b": ModelConfig(
        vocab_size=151936,
        d_model=1024,
        n_layers=28,
        n_heads=16,
        n_kv_heads=8,
        head_dim=128,
        d_ff=3072,
        max_seq_len=40960,
        rope_theta=1e6,
        norm_eps=1e-6,
        qk_norm=True,
        tie_embeddings=True,
    ),
    "qwen3.5-0.8b": ModelConfig(
        vocab_size=248320,
        d_model=1024,
        n_layers=24,
        n_heads=8,
        n_kv_heads=2,
        head_dim=256,
        d_ff=3584,
        max_seq_len=262144,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        norm_eps=1e-6,
        qk_norm=True,
        attn_output_gate=True,
        zero_centered_norm=True,
        tie_embeddings=True,
        layer_types=(_L, _L, _L, _A) * 6,
        linear_n_k_heads=16,
        linear_n_v_heads=16,
        linear_k_head_dim=128,
        linear_v_head_dim=128,
        linear_conv_kernel=4,
    ),
    # Tiny hybrid with the real Qwen3.5 vocabulary so it runs with the Qwen3.5 tokenizer in post-training smokes.
    "qwen3.5-tiny": ModelConfig(
        vocab_size=248320,
        d_model=64,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        max_seq_len=1024,
        rope_theta=1e7,
        partial_rotary_factor=0.25,
        norm_eps=1e-6,
        qk_norm=True,
        attn_output_gate=True,
        zero_centered_norm=True,
        tie_embeddings=True,
        layer_types=(_L, _L, _L, _A),
        linear_n_k_heads=2,
        linear_n_v_heads=4,
        linear_k_head_dim=16,
        linear_v_head_dim=16,
        linear_conv_kernel=4,
    ),
}
