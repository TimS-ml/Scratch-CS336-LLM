"""Toy LM following the TP rule, plus the deterministic training loop shared by the parallel tests.

It contains every case the parallel code must handle: an embedding (tied to the output head in one variant),
attention with COLWISE q/k/v (q with a bias) and ROWWISE o, a per-head norm used inside the TP region (REPLICATE),
a per-head gate (HEADWISE), a SwiGLU MLP (COLWISE up/gate, ROWWISE down) whose up-projection bias is frozen,
RMSNorms, and a frozen replicated parameter in the root.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from scratch_cs336.parallel import TPStyle

VOCAB, D_MODEL, N_HEADS, HEAD_DIM, D_FF, N_LAYERS = 48, 24, 4, 6, 40, 2
SEQ, GLOBAL_BATCH, ACCUM = 8, 8, 2
OPTIM = {"lr": 3e-3, "betas": (0.9, 0.95), "weight_decay": 0.1, "eps": 1e-8}
MAX_NORM = 0.5


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(dim))

    def forward(self, x: Tensor) -> Tensor:
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (h * self.weight.float()).to(x.dtype)


class HeadGate(nn.Module):
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_heads))
        self.bias = nn.Parameter(0.1 * torch.randn(n_heads))

    def forward(self, x: Tensor) -> Tensor:  # [B, T, H, hd]
        return x * torch.sigmoid(self.weight)[:, None] + self.bias[:, None]


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=True)
        self.k = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False)
        self.o = nn.Linear(N_HEADS * HEAD_DIM, D_MODEL, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.gate = HeadGate(N_HEADS)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        h = self.q.weight.shape[0] // HEAD_DIM
        q = self.q_norm(self.q(x).view(b, t, h, HEAD_DIM)).transpose(1, 2)
        k = self.k(x).view(b, t, h, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(b, t, h, HEAD_DIM).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)
        return self.o(self.gate(y).reshape(b, t, h * HEAD_DIM))


class SwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(D_MODEL, D_FF, bias=True)
        self.up.bias.requires_grad_(False)
        self.gate = nn.Linear(D_MODEL, D_FF, bias=False)
        self.down = nn.Linear(D_FF, D_MODEL, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(D_MODEL)
        self.attn = Attention()
        self.mlp_norm = RMSNorm(D_MODEL)
        self.mlp = SwiGLU()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class ToyLM(nn.Module):
    def __init__(self, tied: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList(Block() for _ in range(N_LAYERS))
        self.final_norm = RMSNorm(D_MODEL)
        self.final_scale = nn.Parameter(torch.full((D_MODEL,), 1.5), requires_grad=False)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
        if tied:
            self.lm_head.weight = self.embed.weight

    def forward(self, ids: Tensor) -> Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x) * self.final_scale)

    def tp_plan(self) -> dict[str, TPStyle]:
        return {
            "blocks.*.attn.[qkv]": TPStyle.COLWISE,
            "blocks.*.attn.o": TPStyle.ROWWISE,
            "blocks.*.attn.q_norm": TPStyle.REPLICATE,
            "blocks.*.attn.gate": TPStyle.HEADWISE,
            "blocks.*.mlp.up": TPStyle.COLWISE,
            "blocks.*.mlp.gate": TPStyle.COLWISE,
            "blocks.*.mlp.down": TPStyle.ROWWISE,
        }


def make_model(tied: bool = False) -> ToyLM:
    torch.manual_seed(0)
    return ToyLM(tied)


def make_batch(step: int, vocab: int = VOCAB) -> Tensor:
    """[ACCUM, GLOBAL_BATCH, SEQ + 1] token ids, a pure function of the step."""
    g = torch.Generator().manual_seed(1234 + step)
    return torch.randint(0, vocab, (ACCUM, GLOBAL_BATCH, SEQ + 1), generator=g)


def _loss(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits.float().flatten(0, -2), targets.flatten())


def reference_run(model: nn.Module, steps: int, vocab: int = VOCAB) -> tuple[list[float], list[float], dict]:
    """Single-process AdamW + clipping + grad accumulation over the global batch."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], **OPTIM)
    losses, norms = [], []
    for step in range(steps):
        total = 0.0
        for micro in make_batch(step, vocab):
            loss = _loss(model(micro[:, :-1]), micro[:, 1:]) / ACCUM
            loss.backward()
            total += loss.item()
        norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM).item())
        opt.step()
        opt.zero_grad()
        losses.append(total)
    return losses, norms, {k: v.detach().clone() for k, v in model.state_dict().items()}


def train(pmodel, opt, mesh, start: int, stop: int, vocab: int = VOCAB) -> tuple[list[float], list[float]]:
    """Steps ``[start, stop)`` on this rank's rows of the global batch; returns dp-averaged losses and norms."""
    rows = GLOBAL_BATCH // mesh.dp_size
    lo = mesh.dp_rank * rows
    losses, norms = [], []
    for step in range(start, stop):
        batch = make_batch(step, vocab)
        total = torch.zeros(())
        for i, micro in enumerate(batch[:, lo : lo + rows]):
            with pmodel.no_sync() if i < ACCUM - 1 else nullcontext():
                loss = _loss(pmodel(micro[:, :-1]), micro[:, 1:]) / ACCUM
                loss.backward()
            total += loss.detach()
        pmodel.finish_grad_sync()
        norms.append(pmodel.clip_grad_norm_(MAX_NORM))
        opt.step()
        opt.zero_grad()
        dist.all_reduce(total, group=mesh.dp_group)
        losses.append(total.item() / mesh.dp_size)
    return losses, norms
