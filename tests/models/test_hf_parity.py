import pytest
import torch
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from scratch_cs336.models.hf import config_from_hf, convert_hf_state_dict, load_hf_pretrained
from scratch_cs336.models.transformer import TransformerLM

VOCAB = 97
SEQ = 80  # > Gated DeltaNet chunk size (64), not a multiple of it


def _qwen3_config(tie: bool) -> Qwen3Config:
    return Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=tie,
    )


def _qwen35_text_config(tie: bool) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=256,
        tie_word_embeddings=tie,
    )


def _randomize(model: torch.nn.Module) -> torch.nn.Module:
    # HF init leaves norms at identity and gates trivial; perturb everything so each weight matters.
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1 * torch.randn(p.shape, generator=gen))
    return model.eval()


def _ours(hf_model: torch.nn.Module) -> TransformerLM:
    cfg = config_from_hf(hf_model.config.to_dict())
    model = TransformerLM(cfg)
    model.load_state_dict(convert_hf_state_dict(hf_model.state_dict(), cfg), strict=True)
    return model.eval()


def _input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (2, SEQ), generator=torch.Generator().manual_seed(1))


@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize(
    ("hf_cls", "make_config"), [(Qwen3ForCausalLM, _qwen3_config), (Qwen3_5ForCausalLM, _qwen35_text_config)]
)
def test_logits_match_hf(hf_cls, make_config, tie):
    torch.manual_seed(0)
    hf = _randomize(hf_cls(make_config(tie)))
    ours = _ours(hf)
    ids = _input_ids()
    positions = torch.arange(SEQ).expand(2, SEQ) + torch.tensor([[0], [7]])
    with torch.no_grad():
        expected = hf(input_ids=ids, position_ids=positions).logits
        actual = ours(ids, positions)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_load_qwen35_conditional_generation_checkpoint(tmp_path):
    """Composite checkpoint (vision tower + ``model.language_model.*`` keys) loaded from disk; text-only forward
    of HF's multimodal model uses mrope with identical sections, which must equal our plain rope."""
    torch.manual_seed(0)
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=4)
    config = Qwen3_5Config(
        text_config=_qwen35_text_config(tie=True).to_dict(), vision_config=vision, tie_word_embeddings=True
    )
    hf = _randomize(Qwen3_5ForConditionalGeneration(config))
    hf.save_pretrained(tmp_path)

    cfg, state = load_hf_pretrained(tmp_path, dtype=torch.float32)
    ours = TransformerLM(cfg)
    ours.load_state_dict(state, strict=True)
    ids = _input_ids()
    with torch.no_grad():
        expected = hf(input_ids=ids).logits
        actual = ours.eval()(ids)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
