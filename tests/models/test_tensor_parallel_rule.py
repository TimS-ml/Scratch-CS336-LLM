"""TP rule: slicing weights per ``tp_plan()`` is enough to make the model tensor-parallel.

Each of ``tp`` shard copies runs in its own thread; communication is emulated in lockstep, in one autograd graph:
a COLWISE module's input is replaced by rank 0's input (the TP-region entry: identity forward, all-reduce of input
grads backward), a ROWWISE module's output is the sum of every rank's partial output. Backward runs from rank 0.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fnmatch import fnmatch

import pytest
import torch
from torch import nn

from scratch_cs336.models.config import LayerType, ModelConfig
from scratch_cs336.models.presets import PRESETS
from scratch_cs336.models.transformer import TransformerLM
from scratch_cs336.parallel.plan import TPStyle

TP = 2
SEQ = 70

CONFIGS = {
    "cs336-mha": replace(PRESETS["cs336-tiny"], n_layers=1),
    "qwen3-gqa": ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_ff=48,
        max_seq_len=128,
        qk_norm=True,
        tie_embeddings=True,
    ),  # fmt: skip
    "qwen3.5-hybrid": replace(
        PRESETS["qwen3.5-tiny"], vocab_size=64, n_layers=2, layer_types=(LayerType.GATED_DELTANET, LayerType.ATTENTION)
    ),
}


def _style(plan: dict[str, TPStyle], name: str) -> TPStyle | None:
    styles = {style for pattern, style in plan.items() if fnmatch(name, pattern)}
    assert len(styles) <= 1, f"{name} matched conflicting styles {styles}"
    return styles.pop() if styles else None


def _shard_dim(style: TPStyle | None) -> int | None:
    match style:
        case TPStyle.COLWISE | TPStyle.HEADWISE:
            return 0
        case TPStyle.ROWWISE:
            return 1
        case _:
            return None


class _Lockstep:
    def __init__(self, world: int) -> None:
        self.world = world
        self.barrier = threading.Barrier(world, timeout=60)
        self.lock = threading.Lock()
        self.slots: dict[str, list] = {}

    def exchange(self, key: str, rank: int, value: torch.Tensor) -> list[torch.Tensor]:
        with self.lock:
            self.slots.setdefault(key, [None] * self.world)[rank] = value
        self.barrier.wait()
        return self.slots[key]


def _make_shard(full: TransformerLM, plan: dict[str, TPStyle], rank: int, comm: _Lockstep) -> TransformerLM:
    shard = TransformerLM(full.cfg)
    shard.load_state_dict(full.state_dict())
    for name, module in shard.named_modules():
        style = _style(plan, name)
        for pname, param in list(module.named_parameters(recurse=False)):
            dim = _shard_dim(style)
            if dim is not None:
                setattr(module, pname, nn.Parameter(param.detach().chunk(TP, dim=dim)[rank].clone()))
        if style is TPStyle.COLWISE:
            module.register_forward_pre_hook(lambda m, args, key=name: (comm.exchange(key, rank, args[0])[0],))
        elif style is TPStyle.ROWWISE:
            module.register_forward_hook(lambda m, args, out, key=name: sum(comm.exchange(key, rank, out)))
    return shard


@pytest.mark.parametrize("name", list(CONFIGS))
def test_sharded_forward_and_grads_match_unsharded(name):
    cfg = CONFIGS[name]
    full = TransformerLM(cfg)
    full.init_weights(seed=0)
    plan = full.tp_plan()
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, cfg.vocab_size, (2, SEQ), generator=gen)
    probe = torch.randn(2, SEQ, cfg.vocab_size, generator=gen)

    expected = full(ids)
    (expected * probe).sum().backward()

    comm = _Lockstep(TP)
    shards = [_make_shard(full, plan, r, comm) for r in range(TP)]
    with ThreadPoolExecutor(TP) as pool:
        outputs = list(pool.map(lambda s: s(ids), shards))
    (outputs[0] * probe).sum().backward()

    torch.testing.assert_close(outputs[0], expected, atol=1e-5, rtol=1e-5)
    for module_name, module in full.named_modules():
        style = _style(plan, module_name)
        for pname, param in module.named_parameters(recurse=False):
            fqn = f"{module_name}.{pname}"
            grads = [s.get_parameter(fqn).grad for s in shards]
            dim = _shard_dim(style)
            if dim is not None:
                actual = torch.cat(grads, dim=dim)
            elif style is TPStyle.REPLICATE:
                actual = sum(grads)
            else:
                # Used outside the TP region: every rank already holds the full gradient.
                actual = grads[0]
            torch.testing.assert_close(actual, param.grad, atol=1e-4, rtol=1e-3, msg=lambda m, f=fqn: f"{f}: {m}")
