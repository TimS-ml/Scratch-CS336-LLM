"""Load Hugging Face Qwen3 / Qwen3.5 (text) checkpoints into :class:`TransformerLM` layout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from torch import Tensor

from scratch_cs336.models.config import LayerType, ModelConfig

_SUPPORTED = ("qwen3", "qwen3_5_text")
_SKIPPED_PREFIXES = ("mtp.", "model.visual.", "visual.")
_TEXT_PREFIXES = ("model.language_model.", "model.")

_LAYER_RENAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "mlp_norm.weight",
    **{
        f"self_attn.{p}.weight": f"attn.{p}.weight"
        for p in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")
    },
    **{f"mlp.{p}.weight": f"mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")},
    "linear_attn.in_proj_z.weight": "linear_attn.z_proj.weight",
    "linear_attn.in_proj_a.weight": "linear_attn.a_proj.weight",
    "linear_attn.in_proj_b.weight": "linear_attn.b_proj.weight",
    "linear_attn.A_log": "linear_attn.decay.A_log",
    "linear_attn.dt_bias": "linear_attn.decay.dt_bias",
    "linear_attn.norm.weight": "linear_attn.norm.weight",
    "linear_attn.out_proj.weight": "linear_attn.out_proj.weight",
}


def config_from_hf(hf_config: dict[str, Any]) -> ModelConfig:
    """Accepts a Qwen3 config or a Qwen3.5 config (top-level with ``text_config``, or the text config itself)."""
    text = hf_config.get("text_config") or hf_config
    model_type = text["model_type"]
    if model_type not in _SUPPORTED:
        raise ValueError(f"unsupported model_type {model_type!r}; expected one of {_SUPPORTED}")
    qwen35 = model_type == "qwen3_5_text"
    if text.get("use_sliding_window") or text.get("rope_scaling"):
        raise ValueError("sliding-window attention and rope scaling are not supported")
    rope = text.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"unsupported rope_type {rope['rope_type']!r}")
    # Qwen3.5 text uses mrope; with identical position ids for every section it is exactly standard rope.
    partial = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25 if qwen35 else 1.0))
    n_layers = text["num_hidden_layers"]
    layer_types = None
    if qwen35:
        names = text.get("layer_types") or [
            "full_attention" if (i + 1) % text.get("full_attention_interval", 4) == 0 else "linear_attention"
            for i in range(n_layers)
        ]
        layer_types = tuple(LayerType(n) for n in names)
    return ModelConfig(
        vocab_size=text["vocab_size"],
        d_model=text["hidden_size"],
        n_layers=n_layers,
        n_heads=text["num_attention_heads"],
        n_kv_heads=text["num_key_value_heads"],
        head_dim=text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"],
        d_ff=text["intermediate_size"],
        max_seq_len=text["max_position_embeddings"],
        rope_theta=float(rope.get("rope_theta") or text.get("rope_theta") or 10000.0),
        partial_rotary_factor=float(partial),
        norm_eps=text["rms_norm_eps"],
        qk_norm=True,
        attn_output_gate=qwen35 and text.get("attn_output_gate", True),
        zero_centered_norm=qwen35,
        # In composite (vision-language) configs the top-level flag is the one HF applies.
        tie_embeddings=hf_config.get("tie_word_embeddings", text.get("tie_word_embeddings", False)),
        layer_types=layer_types,
        linear_n_k_heads=text.get("linear_num_key_heads", 0) if qwen35 else 0,
        linear_n_v_heads=text.get("linear_num_value_heads", 0) if qwen35 else 0,
        linear_k_head_dim=text.get("linear_key_head_dim", 0) if qwen35 else 0,
        linear_v_head_dim=text.get("linear_value_head_dim", 0) if qwen35 else 0,
        linear_conv_kernel=text.get("linear_conv_kernel_dim", 0) if qwen35 else 0,
    )


def _text_key(key: str) -> str | None:
    """HF key relative to the text model (``layers.0...``, ``lm_head.weight``), or None if not part of it."""
    if key.startswith(_SKIPPED_PREFIXES):
        return None
    for prefix in _TEXT_PREFIXES:
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return key


def convert_hf_state_dict(hf_sd: dict[str, Tensor], cfg: ModelConfig) -> dict[str, Tensor]:
    """Map HF text weights to our names, splitting the fused GatedDeltaNet ``in_proj_qkv`` and ``conv1d``."""
    key_dim = cfg.linear_n_k_heads * cfg.linear_k_head_dim
    qkv_split = [key_dim, key_dim, cfg.linear_n_v_heads * cfg.linear_v_head_dim]
    out: dict[str, Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        key = _text_key(hf_key)
        if key is None:
            continue
        if key == "embed_tokens.weight":
            out["embed.weight"] = tensor
        elif key == "norm.weight":
            out["final_norm.weight"] = tensor
        elif key == "lm_head.weight":
            if not cfg.tie_embeddings:
                out["lm_head.weight"] = tensor
        elif key.startswith("layers."):
            _, idx, rest = key.split(".", 2)
            prefix = f"blocks.{idx}."
            if rest == "linear_attn.in_proj_qkv.weight":
                for name, part in zip("qkv", tensor.split(qkv_split, dim=0), strict=True):
                    out[f"{prefix}linear_attn.{name}_proj.weight"] = part
            elif rest == "linear_attn.conv1d.weight":
                for name, part in zip("qkv", tensor.squeeze(1).split(qkv_split, dim=0), strict=True):
                    out[f"{prefix}linear_attn.{name}_conv.weight"] = part
            elif rest in _LAYER_RENAMES:
                out[prefix + _LAYER_RENAMES[rest]] = tensor
            else:
                raise KeyError(f"unexpected HF weight {hf_key!r}")
        else:
            raise KeyError(f"unexpected HF weight {hf_key!r}")
    return {k: v.contiguous() for k, v in out.items()}


def to_hf_state_dict(sd: dict[str, Tensor], cfg: ModelConfig) -> dict[str, Tensor]:
    """Inverse of :func:`convert_hf_state_dict`: HF checkpoint names (Qwen3.5 under ``model.language_model.``)."""
    prefix = "model.language_model." if cfg.layer_types is not None else "model."
    renames = {ours: theirs for theirs, ours in _LAYER_RENAMES.items()}
    fused = {"proj.weight": "in_proj_qkv.weight", "conv.weight": "conv1d.weight"}
    out: dict[str, Tensor] = {}
    for key, tensor in sd.items():
        if key == "embed.weight":
            out[prefix + "embed_tokens.weight"] = tensor
        elif key == "final_norm.weight":
            out[prefix + "norm.weight"] = tensor
        elif key == "lm_head.weight":
            out["lm_head.weight"] = tensor
        elif key.startswith("blocks."):
            _, idx, rest = key.split(".", 2)
            layer = f"{prefix}layers.{idx}."
            name, _, suffix = rest.removeprefix("linear_attn.").partition("_")
            if rest.startswith("linear_attn.") and name in ("q", "k", "v") and suffix in fused:
                if name == "q":  # k and v are fused into the same HF tensor
                    parts = torch.cat([sd[f"blocks.{idx}.linear_attn.{n}_{suffix}"] for n in "qkv"], dim=0)
                    out[layer + "linear_attn." + fused[suffix]] = (
                        parts.unsqueeze(1) if suffix == "conv.weight" else parts
                    )
            elif rest in renames:
                out[layer + renames[rest]] = tensor
            else:
                raise KeyError(f"unexpected weight {key!r}")
        else:
            raise KeyError(f"unexpected weight {key!r}")
    return out


def load_hf_config(repo_or_path: str | Path) -> ModelConfig:
    """Only ``config.json`` of a local snapshot directory or Hub repo: cheap validation before loading weights."""
    path = Path(repo_or_path)
    config = path / "config.json" if path.is_dir() else Path(hf_hub_download(str(repo_or_path), "config.json"))
    return config_from_hf(json.loads(config.read_text()))


def load_hf_pretrained(
    repo_or_path: str | Path, dtype: torch.dtype = torch.bfloat16
) -> tuple[ModelConfig, dict[str, Tensor]]:
    """Read a local snapshot directory or download ``repo_or_path`` from the Hub (json + safetensors only)."""
    path = Path(repo_or_path)
    if not path.is_dir():
        path = Path(snapshot_download(str(repo_or_path), allow_patterns=["*.json", "*.safetensors"]))
    cfg = load_hf_config(path)
    index = path / "model.safetensors.index.json"
    if index.exists():
        files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    else:
        files = ["model.safetensors"]
    hf_sd: dict[str, Tensor] = {}
    for file in files:
        with safe_open(path / file, framework="pt") as handle:
            for key in handle.keys():
                if _text_key(key) is not None:
                    hf_sd[key] = handle.get_tensor(key).to(dtype)
    return cfg, convert_hf_state_dict(hf_sd, cfg)
