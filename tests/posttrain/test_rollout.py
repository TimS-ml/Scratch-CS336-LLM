from dataclasses import replace

import pytest
import torch

from scratch_cs336.distributed import DistEnv, MeshConfig, build_mesh
from scratch_cs336.distributed.spawn import run_distributed
from scratch_cs336.models.generate import generate
from scratch_cs336.models.hf import convert_hf_state_dict, to_hf_state_dict
from scratch_cs336.models.presets import PRESETS
from scratch_cs336.models.transformer import TransformerLM
from scratch_cs336.parallel import Backend, ParallelConfig, Strategy, parallelize
from scratch_cs336.posttrain.rollout import SamplingParams, TorchRolloutEngine
from tests.posttrain.char_tokenizer import CharTokenizer

MODELS = {name: replace(PRESETS[name], vocab_size=512) for name in ("cs336-tiny", "qwen3.5-tiny")}
PROMPTS = [[5, 6, 7], [9], [11, 12, 13, 14, 15, 16, 17], [20, 21]]


def _model(name: str) -> TransformerLM:
    model = TransformerLM(MODELS[name])
    model.init_weights(seed=0)
    return model.eval()


def _engine(name: str, **kw) -> TorchRolloutEngine:
    engine = TorchRolloutEngine(MODELS[name], max_batch_size=3, **kw)
    engine.load_state_dict(_model(name).state_dict())
    return engine


@pytest.mark.parametrize("name", list(MODELS))
def test_batched_greedy_matches_unbatched_generation(name):
    """Right padding must not leak into attention or the GatedDeltaNet recurrence of shorter prompts."""
    model = _model(name)
    out = _engine(name).generate(PROMPTS, n=2, params=SamplingParams(max_tokens=6, temperature=0))
    for prompt, completions in zip(PROMPTS, out, strict=True):
        expected = generate(model, prompt, max_new_tokens=6, temperature=0)
        assert completions == [expected, expected]


def test_each_sequence_stops_at_its_own_stop_token():
    name = "cs336-tiny"
    model = _model(name)
    free = [generate(model, p, max_new_tokens=8, temperature=0) for p in PROMPTS]
    stop = free[0][2]
    out = _engine(name).generate(PROMPTS, 1, SamplingParams(max_tokens=8, temperature=0, stop_token_ids=(stop,)))
    for completions, unstopped in zip(out, free, strict=True):
        cut = unstopped.index(stop) + 1 if stop in unstopped else len(unstopped)
        assert completions[0] == unstopped[:cut]


def test_stop_strings_end_generation_after_the_string():
    name = "cs336-tiny"
    tok = CharTokenizer()
    free = _engine(name).generate([PROMPTS[0]], 1, SamplingParams(max_tokens=8, temperature=0))[0][0]
    stop_text = tok.decode(free[2:4])
    out = _engine(name, tokenizer=tok).generate(
        [PROMPTS[0]], 1, SamplingParams(max_tokens=8, temperature=0, stop=(stop_text,))
    )[0][0]
    text = tok.decode(free)
    assert tok.decode(out) == text[: text.index(stop_text) + len(stop_text)]


def test_sampling_is_seeded():
    params = SamplingParams(max_tokens=5, temperature=1.0, top_p=0.9, seed=3)
    engine = _engine("cs336-tiny")
    a, b = engine.generate(PROMPTS, 3, params), engine.generate(PROMPTS, 3, params)
    assert a == b
    assert a != engine.generate(PROMPTS, 3, replace(params, seed=4))


def test_hf_state_dict_round_trip_qwen35():
    model = _model("qwen3.5-tiny")
    sd = model.state_dict()
    hf = to_hf_state_dict(sd, model.cfg)
    assert "model.language_model.layers.0.linear_attn.in_proj_qkv.weight" in hf
    back = convert_hf_state_dict(hf, model.cfg)
    assert back.keys() == sd.keys()
    for k, v in sd.items():
        assert torch.equal(back[k], v), k


def _sync_from_fsdp(env: DistEnv) -> list[list[list[int]]]:
    mesh = build_mesh(MeshConfig(shard=2), env)
    pmodel = parallelize(
        _model("qwen3.5-tiny"), mesh, ParallelConfig(MeshConfig(shard=2), Strategy.FSDP, Backend.SCRATCH)
    )
    engine = TorchRolloutEngine(MODELS["qwen3.5-tiny"])
    engine.sync_weights(pmodel)
    return engine.generate(PROMPTS[mesh.dp_rank :: 2], 1, SamplingParams(max_tokens=4, temperature=0))


def test_sync_weights_from_sharded_policy():
    model = _model("qwen3.5-tiny")
    rank0, rank1 = run_distributed(_sync_from_fsdp, 2)
    for prompt, completions in zip(PROMPTS[0::2] + PROMPTS[1::2], rank0 + rank1, strict=True):
        assert completions == [generate(model, prompt, max_new_tokens=4, temperature=0)]
