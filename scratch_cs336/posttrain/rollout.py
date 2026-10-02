"""Rollout engines: sample completions from the current policy.

``TorchRolloutEngine`` runs on every dp rank with an unsharded local copy of the policy (rebuilt from
``full_state_dict()`` on :meth:`sync_weights`) and samples its own prompt shard. ``VLLMRolloutEngine`` ports the
CS336 hw5 staff ``vllm_utils``: a ``vllm serve`` process on dedicated GPUs, HTTP completions, NCCL weight broadcast
from global rank 0.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.distributed as dist
from torch import Tensor

from scratch_cs336.models.config import ModelConfig
from scratch_cs336.models.hf import to_hf_state_dict
from scratch_cs336.models.transformer import TransformerLM
from scratch_cs336.parallel.api import ParallelModel
from scratch_cs336.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    temperature: float = 1.0  # 0 = greedy
    top_p: float = 1.0
    stop_token_ids: tuple[int, ...] = ()  # generation ends after emitting one of these (kept in the output)
    stop: tuple[str, ...] = ()  # or once the decoded completion contains one of these strings (kept)
    seed: int = 0


class RolloutEngine(Protocol):
    def generate(self, prompts: list[list[int]], n: int, params: SamplingParams) -> list[list[list[int]]]:
        """``n`` completions (token ids, without the prompt) for every prompt, in prompt order."""
        ...

    def sync_weights(self, pmodel: ParallelModel) -> None:
        """Collective over all ranks: make the engine sample from ``pmodel``'s current weights."""
        ...


def sample_tokens(logits: Tensor, temperature: float, top_p: float, generator: torch.Generator | None) -> Tensor:
    """One token per row of ``logits [N, V]``: greedy at temperature 0, else nucleus sampling."""
    if temperature == 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, order = probs.sort(dim=-1, descending=True)
        # Keep the smallest prefix whose mass reaches top_p (always at least the top token).
        sorted_probs[sorted_probs.cumsum(-1) - sorted_probs >= top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, order, sorted_probs)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


class TorchRolloutEngine:
    """Batched sampling with right padding and full recompute per token (no KV cache).

    Right padding keeps every sequence's positions ``0..len-1``, so causal attention and the causal GatedDeltaNet
    recurrence never see padding; finished sequences leave the active batch. Only the last position of each active
    sequence is projected to the vocabulary.
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        max_batch_size: int = 64,
        tokenizer: Tokenizer | None = None,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.tokenizer = tokenizer
        self.model = TransformerLM(model_cfg).to(device=self.device, dtype=dtype).eval().requires_grad_(False)

    def sync_weights(self, pmodel: ParallelModel) -> None:
        self.load_state_dict(pmodel.full_state_dict())

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> None:
        self.model.load_state_dict({k: v.to(self.dtype) for k, v in state_dict.items()})

    @torch.inference_mode()
    def generate(self, prompts: list[list[int]], n: int, params: SamplingParams) -> list[list[list[int]]]:
        if params.stop and self.tokenizer is None:
            raise ValueError("stop strings need a tokenizer; pass one to TorchRolloutEngine")
        generator = torch.Generator(device=self.device).manual_seed(params.seed)
        flat = [p for p in prompts for _ in range(n)]
        completions: list[list[int]] = []
        for start in range(0, len(flat), self.max_batch_size):
            completions.extend(self._generate_batch(flat[start : start + self.max_batch_size], params, generator))
        return [completions[i * n : (i + 1) * n] for i in range(len(prompts))]

    def _generate_batch(
        self, prompts: list[list[int]], params: SamplingParams, generator: torch.Generator
    ) -> list[list[int]]:
        if any(not p for p in prompts):
            raise ValueError("empty prompt")
        stop_ids = set(params.stop_token_ids)
        stop_window = max((len(s.encode()) for s in params.stop), default=0)  # a token decodes to >= 1 byte
        lens = [len(p) for p in prompts]
        tokens = torch.zeros(len(prompts), max(lens) + params.max_tokens, dtype=torch.long, device=self.device)
        for i, p in enumerate(prompts):
            tokens[i, : len(p)] = torch.tensor(p, dtype=torch.long)
        new: list[list[int]] = [[] for _ in prompts]
        active = list(range(len(prompts)))
        for _ in range(params.max_tokens):
            if not active:
                break
            rows = torch.tensor(active, device=self.device)
            cur = torch.tensor([lens[i] for i in active], device=self.device)
            hidden = self.model.hidden_states(tokens[rows, : int(cur.max())])
            last = hidden[torch.arange(len(active), device=self.device), cur - 1]
            sampled = sample_tokens(self.model.logits(last), params.temperature, params.top_p, generator).tolist()
            still = []
            for i, token in zip(active, sampled, strict=True):
                tokens[i, lens[i]] = token
                lens[i] += 1
                new[i].append(token)
                done = token in stop_ids
                if not done and stop_window:
                    tail = self.tokenizer.decode(new[i][-stop_window:])
                    done = any(s in tail for s in params.stop)
                if not done:
                    still.append(i)
            active = still
        return new


# ---- vLLM (port of the CS336 hw5 staff vllm_utils; needs Linux + CUDA + vllm) ----


@dataclass(frozen=True)
class VLLMServerConfig:
    model_id: str  # HF repo or local HF-format directory the server loads first
    host: str = "127.0.0.1"
    port: int = 8000
    gpu: int = 1  # CUDA device of the server, distinct from the trainer's devices
    seed: int = 0
    load_format: str = "auto"
    logging_level: str = "ERROR"
    gpu_memory_utilization: float = 0.9
    launch_server: bool = True  # False: connect to an already running server
    startup_timeout: int = 600
    shutdown_timeout: int = 30
    request_batch_size: int | None = None  # prompts per HTTP request; None sends a rank's shard at once


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


def _kill_existing_server(port: int) -> None:
    pattern = f"vllm serve .* --port {port}"
    try:
        if subprocess.run(["pkill", "-TERM", "-f", pattern], check=False).returncode == 0:
            time.sleep(2)
            subprocess.run(["pkill", "-KILL", "-f", pattern], check=False)
    except FileNotFoundError:
        pass


def _start_server(cfg: VLLMServerConfig) -> subprocess.Popen:
    env = os.environ.copy()  # the child process's environment, not configuration of this program
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu)
    env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes /pause, /update_weights, /init_weight_transfer_engine
    env["VLLM_LOGGING_LEVEL"] = cfg.logging_level
    command = [
        "vllm", "serve", cfg.model_id,
        "--host", cfg.host,
        "--port", str(cfg.port),
        "--dtype", "bfloat16",
        "--enable-prefix-caching",
        "--gpu-memory-utilization", str(cfg.gpu_memory_utilization),
        "--seed", str(cfg.seed),
        "--tensor-parallel-size", "1",
        "--weight-transfer-config", json.dumps({"backend": "nccl"}),
        "--load-format", cfg.load_format,
    ]  # fmt: skip
    logger.info("starting vLLM server: %s", " ".join(command))
    return subprocess.Popen(command, env=env, start_new_session=True)


def _wait_for_server(base_url: str, process: subprocess.Popen | None, timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"timed out waiting for vLLM server at {base_url}")


def _stop_server(process: subprocess.Popen | None, timeout: int) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _import_vllm_nccl() -> Any:
    try:
        from vllm.distributed.weight_transfer import nccl_engine
    except ImportError as e:
        raise ImportError(
            "VLLMRolloutEngine needs vllm with NCCL weight transfer (Linux + CUDA; pip install 'scratch-cs336[vllm]'). "
            "Use TorchRolloutEngine elsewhere."
        ) from e
    return nccl_engine


class VLLMRolloutEngine:
    """Every rank sends its own prompt shard to one shared server; global rank 0 owns the server process and the
    NCCL weight-transfer group (rank 0 of a ``1 + inference_world_size`` group, vLLM workers at offset 1)."""

    def __init__(self, server: VLLMServerConfig, model_cfg: ModelConfig, policy_device: torch.device | str):
        self.server = server
        self.model_cfg = model_cfg
        self.policy_device = torch.device(policy_device)
        self.base_url = f"http://{server.host}:{server.port}"
        self.is_owner = dist.get_rank() == 0
        self.process: subprocess.Popen | None = None
        self.weight_sync_group: Any = None
        if self.is_owner:
            nccl_engine = _import_vllm_nccl()
            if server.launch_server:
                _kill_existing_server(server.port)
                self.process = _start_server(server)
                atexit.register(self.stop)
        _wait_for_server(self.base_url, self.process, server.startup_timeout)
        if self.is_owner:
            self.weight_sync_group = self._init_weight_sync(nccl_engine)
        dist.barrier()

    def stop(self) -> None:
        _stop_server(self.process, self.server.shutdown_timeout)

    def _init_weight_sync(self, nccl_engine: Any) -> Any:
        from vllm.utils.network_utils import get_ip, get_open_port

        inference_world_size = _http_json("GET", f"{self.base_url}/get_world_size", timeout=10)["world_size"]
        world_size = inference_world_size + 1
        master_address, master_port = get_ip(), get_open_port()
        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        torch.cuda.set_device(self.policy_device)
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(
                _http_json, "POST", f"{self.base_url}/init_weight_transfer_engine", {"init_info": init_info}, 60
            )
            group = nccl_engine.NCCLWeightTransferEngine.trainer_init(
                {"master_address": master_address, "master_port": master_port, "world_size": world_size}
            )
            pending.result()
        return group

    def sync_weights(self, pmodel: ParallelModel) -> None:
        """Gathers the full policy on every rank (collective), then rank 0 broadcasts HF-named bf16 weights."""
        full = pmodel.full_state_dict()
        if self.is_owner:
            nccl_engine = _import_vllm_nccl()
            weights = [
                (name, t.to(self.policy_device, torch.bfloat16))
                for name, t in to_hf_state_dict(full, self.model_cfg).items()
            ]
            update_info = {
                "names": [name for name, _ in weights],
                "dtype_names": [str(t.dtype).split(".")[-1] for _, t in weights],
                "shapes": [list(t.shape) for _, t in weights],
                "packed": True,
            }
            torch.cuda.set_device(self.policy_device)
            _http_json("POST", f"{self.base_url}/pause", timeout=60)
            with ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(
                    _http_json, "POST", f"{self.base_url}/update_weights", {"update_info": update_info}, 300
                )
                nccl_engine.NCCLWeightTransferEngine.trainer_send_weights(
                    iterator=iter(weights),
                    trainer_args=nccl_engine.NCCLTrainerSendWeightsArgs(group=self.weight_sync_group, packed=True),
                )
                pending.result()
            _http_json("POST", f"{self.base_url}/reset_prefix_cache", timeout=60)
            _http_json("POST", f"{self.base_url}/resume", timeout=60)
        dist.barrier()  # no rank samples before the server holds the new weights

    def generate(self, prompts: list[list[int]], n: int, params: SamplingParams) -> list[list[list[int]]]:
        size = self.server.request_batch_size or max(len(prompts), 1)
        out: list[list[list[int]]] = []
        for start in range(0, len(prompts), size):
            chunk = prompts[start : start + size]
            payload: dict[str, Any] = {
                "model": self.server.model_id,
                "prompt": chunk,  # token-id lists: no re-tokenization on the server
                "temperature": params.temperature,
                "top_p": params.top_p,
                "max_tokens": params.max_tokens,
                "n": n,
                "seed": params.seed + start,  # callers derive params.seed per (step, rank)
                "return_token_ids": True,
                "stop_token_ids": list(params.stop_token_ids),
            }
            if params.stop:
                payload["stop"] = list(params.stop)
                payload["include_stop_str_in_output"] = True
            response = _http_json("POST", f"{self.base_url}/v1/completions", payload, timeout=3600)
            choices = sorted(response["choices"], key=lambda c: c["index"])  # index = prompt * n + sample
            ids = [list(c.get("token_ids") or []) for c in choices]
            out.extend(ids[i * n : (i + 1) * n] for i in range(len(chunk)))
        return out
