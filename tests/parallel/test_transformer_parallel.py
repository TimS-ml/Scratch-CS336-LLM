"""Integration: the real ``TransformerLM`` (CS336 MHA and the Qwen3.5 hybrid with Gated DeltaNet, qk-norm and
gated attention) under TP x FSDP/ZeRO/DDP trains like a single process.

Runs in float64: the hybrid's fp32 gradients carry ~1e-5 relative rounding noise that AdamW amplifies on
near-zero gradients, which would hide real sharding errors behind a loose tolerance. A bf16-compute smoke run of
the hybrid (fp32 master weights) checks the mixed-precision path under TP against the fp32 single process.
"""

from __future__ import annotations

import functools
from dataclasses import replace

import pytest
import torch
from toy_lm import OPTIM, reference_run, train

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.models.config import ModelConfig
from scratch_cs336.models.presets import PRESETS
from scratch_cs336.models.transformer import TransformerLM
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, build_optimizer, parallelize

STEPS = 3
VOCAB = 256
MESH = MeshConfig(replicate=1, shard=2, tensor=2)
MODELS = {name: replace(PRESETS[name], vocab_size=VOCAB) for name in ("cs336-tiny", "qwen3.5-tiny")}
CASES = [(m, s, b) for m in MODELS for s in Strategy for b in Backend]
BF16_MODEL = "qwen3.5-tiny"
# Single-process FSDP with bf16 compute vs the fp32 reference (same model, seed, batches) deviates by up to 0.021 in
# loss over these 3 steps (0.0033 at step 0, before any update; 0.084 by step 5 as AdamW amplifies the noise); the
# TP x FSDP bf16 runs measured 0.026. 0.05 leaves ~2x headroom over the inherent bf16 gap.
BF16_LOSS_ATOL = 0.05


def _model(cfg: ModelConfig, dtype: torch.dtype = torch.float64) -> TransformerLM:
    model = TransformerLM(cfg)
    model.init_weights(seed=0)
    return model.to(dtype)


def _run(env: DistEnv) -> dict:
    mesh = build_mesh(MESH, env)
    out = {}
    for name, strategy, backend in CASES:
        cfg = ParallelConfig(MESH, strategy, backend)
        pmodel = parallelize(_model(MODELS[name]), mesh, cfg)
        opt = build_optimizer(pmodel, cfg, torch.optim.AdamW, **OPTIM)
        losses, norms = train(pmodel, opt, mesh, 0, STEPS, VOCAB)
        out[name, strategy, backend] = (losses, norms, pmodel.full_state_dict())
    for backend in Backend:
        cfg = ParallelConfig(MESH, Strategy.FSDP, backend, "bfloat16")
        pmodel = parallelize(_model(MODELS[BF16_MODEL], torch.float32), mesh, cfg)
        opt = build_optimizer(pmodel, cfg, torch.optim.AdamW, **OPTIM)
        out["bf16", backend] = train(pmodel, opt, mesh, 0, STEPS, VOCAB)[0]
    return out


@functools.cache
def _results() -> list[dict]:
    return run_distributed(_run, 4)


@functools.cache
def _reference(name: str) -> tuple:
    return reference_run(_model(MODELS[name]), STEPS, VOCAB)


@pytest.mark.parametrize(("name", "strategy", "backend"), CASES, ids=["-".join(c) for c in CASES])
def test_tp_x_dp_matches_single_process(name: str, strategy: Strategy, backend: Backend) -> None:
    ref_losses, ref_norms, ref_state = _reference(name)
    for rank in _results():
        losses, norms, state = rank[name, strategy, backend]
        torch.testing.assert_close(losses, ref_losses, atol=1e-5, rtol=0)
        torch.testing.assert_close(norms, ref_norms, atol=1e-5, rtol=0)
        assert list(state) == list(ref_state)
        for key, ref in ref_state.items():
            torch.testing.assert_close(state[key], ref, atol=1e-5, rtol=0, msg=lambda m, k=key: f"{k}: {m}")


@pytest.mark.parametrize("backend", list(Backend))
def test_qwen35_bf16_compute_under_tp_stays_close_to_fp32_single_process(backend: Backend) -> None:
    ref_losses = reference_run(_model(MODELS[BF16_MODEL], torch.float32), STEPS, VOCAB)[0]
    for rank in _results():
        losses = torch.tensor(rank["bf16", backend])
        assert torch.isfinite(losses).all()
        torch.testing.assert_close(losses, torch.tensor(ref_losses), atol=BF16_LOSS_ATOL, rtol=0)
