import torch

from scratch_cs336.models.generate import generate
from scratch_cs336.models.presets import PRESETS
from scratch_cs336.models.transformer import TransformerLM


def _model() -> TransformerLM:
    model = TransformerLM(PRESETS["cs336-tiny"])
    model.init_weights(seed=0)
    return model.eval()


def test_vanishing_top_p_sampling_is_greedy():
    model = _model()
    greedy = generate(model, [1, 2, 3], max_new_tokens=8, temperature=0)
    sampled = generate(model, [1, 2, 3], max_new_tokens=8, top_p=1e-6, generator=torch.Generator().manual_seed(0))
    assert sampled == greedy


def test_stops_after_eos():
    model = _model()
    greedy = generate(model, [5], max_new_tokens=8, temperature=0)
    eos = greedy[2]
    out = generate(model, [5], max_new_tokens=8, temperature=0, eos_token_id=eos)
    assert out == greedy[: greedy.index(eos) + 1]


def test_init_weights_is_deterministic_per_seed():
    a, b, c = (TransformerLM(PRESETS["qwen3.5-tiny"]) for _ in range(3))
    a.init_weights(seed=3)
    b.init_weights(seed=3)
    c.init_weights(seed=4)
    for (name, pa), pb in zip(a.named_parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb), name
    assert not torch.equal(a.blocks[0].linear_attn.q_proj.weight, c.blocks[0].linear_attn.q_proj.weight)
