import pytest
import torch
import torch.nn.functional as F

from scratch_cs336.models.gated_deltanet import chunk_gated_delta_rule, l2norm, recurrent_gated_delta_rule


def _inputs(seq: int, heads: int = 3, dk: int = 8, dv: int = 6, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    q = l2norm(torch.randn(2, seq, heads, dk, generator=gen)) * dk**-0.5
    k = l2norm(torch.randn(2, seq, heads, dk, generator=gen))
    v = torch.randn(2, seq, heads, dv, generator=gen)
    g = -F.softplus(torch.randn(2, seq, heads, generator=gen))
    beta = torch.rand(2, seq, heads, generator=gen)
    return q, k, v, g, beta


@pytest.mark.parametrize("seq", [1, 15, 16, 37, 64])
@pytest.mark.parametrize("chunk_size", [16, 64])
def test_chunked_matches_recurrence(seq, chunk_size):
    inputs = _inputs(seq)
    expected = recurrent_gated_delta_rule(*inputs)
    actual = chunk_gated_delta_rule(*inputs, chunk_size=chunk_size)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_chunked_is_causal():
    q, k, v, g, beta = _inputs(40)
    out = chunk_gated_delta_rule(q, k, v, g, beta, chunk_size=16)
    v2 = v.clone()
    v2[:, 25:] += 10.0
    out2 = chunk_gated_delta_rule(q, k, v2, g, beta, chunk_size=16)
    torch.testing.assert_close(out[:, :25], out2[:, :25])
    assert not torch.allclose(out[:, 25:], out2[:, 25:])
