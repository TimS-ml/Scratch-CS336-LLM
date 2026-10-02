"""Parity against Hugging Face on the real Qwen/Qwen3.5-0.8B weights (opt-in: slow, needs the local HF cache).

The bf16 reference for "how close can bf16 get" is HF's own bf16 forward: both are compared against HF fp32.
"""

import gc

import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from scratch_cs336.models.generate import generate
from scratch_cs336.models.hf import load_hf_pretrained
from scratch_cs336.models.transformer import TransformerLM

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

REPO = "Qwen/Qwen3.5-0.8B"
# > 64 tokens so the Gated DeltaNet chunk recurrence carries state across a chunk boundary.
TEXT = (
    "The Gated DeltaNet layer keeps a fixed-size memory that is updated one token at a time with the delta rule, "
    "while every fourth layer of the network uses ordinary softmax attention over the whole prefix. Training runs "
    "the recurrence in chunks of sixty-four tokens, so a long enough prompt exercises the state that is carried "
    "from one chunk to the next."
)
GREEDY_PROMPT = "The capital of France is"
GREEDY_TOKENS = 8
FP32_ATOL = 1e-4
# Measured: ours bf16 RMS error 1.01x and max error 0.84x HF bf16's; HF's own bf16 sdpa and eager forwards differ
# from each other by ~5% in RMS error against fp32.
BF16_RMS_FACTOR = 1.1
BF16_MAX_FACTOR = 1.25


@pytest.fixture(scope="module")
def snapshot() -> str:
    try:
        return snapshot_download(REPO, allow_patterns=["*.json", "*.safetensors"], local_files_only=True)
    except LocalEntryNotFoundError:
        pytest.skip(f"{REPO} is not in the local Hugging Face cache")


@pytest.fixture(scope="module")
def prompts(snapshot: str) -> dict[str, list[int]]:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(f"{snapshot}/tokenizer.json")
    ids = tok.encode(TEXT).ids
    assert len(ids) > 64
    return {"text": ids, "greedy": tok.encode(GREEDY_PROMPT).ids}


@pytest.fixture(scope="module")
def hf_reference(snapshot: str, prompts: dict[str, list[int]]) -> dict[str, object]:
    """HF logits in fp32 and bf16 plus bf16 greedy tokens; each HF model is freed before the next loads."""
    from transformers import Qwen3_5ForCausalLM

    ref: dict[str, object] = {}
    ids = torch.tensor([prompts["text"]])
    for dtype in (torch.float32, torch.bfloat16):
        model = Qwen3_5ForCausalLM.from_pretrained(snapshot, dtype=dtype, attn_implementation="sdpa").eval()
        with torch.no_grad():
            ref[dtype] = model(input_ids=ids).logits[0].float()
            if dtype is torch.bfloat16:
                greedy = torch.tensor([prompts["greedy"]])
                out = model.generate(greedy, max_new_tokens=GREEDY_TOKENS, do_sample=False)
                ref["greedy"] = out[0, greedy.shape[1] :].tolist()
        del model
        gc.collect()
    return ref


def _rms(err: torch.Tensor) -> float:
    return err.pow(2).mean().sqrt().item()


def _ours(snapshot: str, dtype: torch.dtype) -> TransformerLM:
    cfg, sd = load_hf_pretrained(snapshot, dtype=dtype)
    with torch.device("meta"):
        model = TransformerLM(cfg)
    # assign=True keeps the checkpoint dtype; a plain load_state_dict would copy into fp32 parameters.
    model.load_state_dict(sd, strict=True, assign=True)
    return model.eval()


def test_fp32_logits_match_hf_at_every_position(snapshot, prompts, hf_reference):
    model = _ours(snapshot, torch.float32)
    with torch.no_grad():
        logits = model(torch.tensor([prompts["text"]]))[0]
    torch.testing.assert_close(logits, hf_reference[torch.float32], rtol=0, atol=FP32_ATOL)


def test_bf16_is_as_accurate_as_hf_bf16_and_greedy_agrees(snapshot, prompts, hf_reference):
    model = _ours(snapshot, torch.bfloat16)
    assert next(model.parameters()).dtype is torch.bfloat16
    with torch.no_grad():
        logits = model(torch.tensor([prompts["text"]]))[0].float()
    exact = hf_reference[torch.float32]
    ours_err = (logits - exact).abs()
    hf_err = (hf_reference[torch.bfloat16] - exact).abs()
    assert _rms(ours_err) <= BF16_RMS_FACTOR * _rms(hf_err), (_rms(ours_err), _rms(hf_err))
    assert ours_err.max() <= BF16_MAX_FACTOR * hf_err.max(), (ours_err.max().item(), hf_err.max().item())

    assert generate(model, prompts["greedy"], GREEDY_TOKENS, temperature=0) == hf_reference["greedy"]
