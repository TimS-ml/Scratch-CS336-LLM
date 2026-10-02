"""Autoregressive sampling (full recompute per token; no KV cache)."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: list[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Returns only the new tokens (stops after emitting ``eos_token_id``). ``temperature == 0`` is greedy."""
    device = next(model.parameters()).device
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new: list[int] = []
    for _ in range(max_new_tokens):
        logits = model(ids)[0, -1].float()
        if temperature == 0:
            token = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, order = probs.sort(descending=True)
                # Keep the smallest prefix whose mass reaches top_p (always at least the top token).
                drop = sorted_probs.cumsum(-1) - sorted_probs >= top_p
                sorted_probs[drop] = 0.0
                probs = torch.zeros_like(probs).scatter_(0, order, sorted_probs)
            token = int(torch.multinomial(probs, 1, generator=generator))
        new.append(token)
        if token == eos_token_id:
            break
        ids = torch.cat([ids, ids.new_tensor([[token]])], dim=1)
    return new
