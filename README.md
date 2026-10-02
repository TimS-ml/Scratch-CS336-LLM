# Scratch-CS336-LLM

A PyTorch codebase for pretraining and post-training LLMs, written from scratch and distributed-first. It follows the
Stanford CS336 (Spring 2026) assignment contracts (BPE, DDP / ZeRO-1 / FSDP, token pipelines, SFT / DPO / GRPO) and
organizes them the way Marin organizes large runs: a declarative device mesh, sharded state with deterministic resume,
sharded token caches, and an experiment DAG whose steps own hashed output paths. Marin is JAX; every idea here is
ported to `torch.distributed` (`DeviceMesh`, process groups, DTensor).

Distributed-first means world size is never special. Every entry point calls `init_distributed()`, so a plain
`python -m ...` run is a world of one with a real process group, and the same code runs on 1 CPU process, a multi-4090
box, or a Slurm cluster. Tests exercise the multi-rank paths on CPU with gloo, so parallelism bugs surface without GPUs.

## Features

- **Models** (`scratch_cs336.models`): one `TransformerLM` covering CS336 basics (Llama-style), Qwen3, and Qwen3.5
  text (hybrid Gated DeltaNet + gated attention). HF Qwen3 / Qwen3.5 checkpoints load directly and export back.
  On the real Qwen/Qwen3.5-0.8B weights, our Qwen3.5 forward matches Hugging Face's within 1e-4 on every logit in fp32
  (bitwise identical on a short prompt), its bf16 logits are as close to the fp32 reference as HF's own bf16 forward is,
  and bf16 greedy decoding produces the same tokens. See `tests/models/test_qwen35_real.py` (opt-in, `slow`). Random-weight
  configs are tested to 1e-4 in `tests/models/test_hf_parity.py`.
- **Parallelism** (`scratch_cs336.parallel`): DDP, ZeRO-1, FSDP and HSDP, each implemented from scratch
  (`backend=scratch`) and with torch-native primitives (`backend=native`: `DistributedDataParallel`,
  `ZeroRedundancyOptimizer`, FSDP2 `fully_shard`). Tensor parallelism composes underneath, driven by the model's
  `tp_plan()` with four styles: `colwise`, `rowwise`, `headwise`, `replicate`. bf16 mixed precision with fp32 master
  weights; activation checkpointing; tests assert every strategy x backend trains like a single process.
- **Checkpoints** (`scratch_cs336.checkpoint`): each rank writes its own shard, rank 0 commits `metadata.json` last;
  async save, retention, deterministic bitwise resume. A full unsharded `model.safetensors` export is the path to eval
  and across mesh layouts.
- **Data**: from-scratch parallel byte-level BPE trainer and tokenizer plus an HF `tokenizers` wrapper; resumable
  sharded token caches with a ledger; a deterministic distributed loader (each rank's rows are a pure function of
  `(seed, step, dp_rank)`); exact-proportion mixtures; ChatML rendering with assistant-only loss masks.
- **Trainer** (`scratch_cs336.train`): one generic loop for any model, loss and strategy; exact global-mean gradients
  under uneven masks and gradient accumulation; AdamW from scratch; cosine / WSD / constant schedules; MFU logging;
  jsonl and wandb tracking.
- **Post-training** (`scratch_cs336.posttrain`): SFT (assistant-token loss), DPO (sharded frozen reference), and GRPO
  with variants (Dr. GRPO, RFT, MaxRL, GRPO-clip, GSPO), GSM8K rewards, and rollout engines (torch, or a vLLM server
  with NCCL weight sync).
- **Pipeline** (`scratch_cs336.pipeline`): `Step`s form a DAG; a step's output directory is `<name>-<hash8>`, a hash of
  its config and its dependencies' hashes. Re-running is a cache hit; changing a hyperparameter forks a new path.
- **Launchers** (`scratch_cs336.launch`): `torchrun --standalone` for one machine, `sbatch` + `srun torchrun` for Slurm.

## Install

Python 3.12 or 3.13 (`.python-version` pins 3.12), torch >= 2.13 (the version vLLM pins).

```bash
uv sync                      # core + dev group (pytest, ruff, transformers for parity tests)
uv sync --extra wandb        # wandb tracker
uv sync --extra vllm         # vLLM rollout engine (Linux only)
uv sync --extra datasets     # GSM8K download (experiments/qwen35_gsm8k.py)
uv sync --no-dev             # without the dev group
```

Extras combine (`uv sync --extra wandb --extra datasets`). Commands below use `python`; with uv, prefix `uv run` or
use `.venv/bin/python`.

## Quickstart

`experiments/` holds Marin-style experiment files. Each prints its plan by default and executes it with `--run`.

```bash
python -m experiments.tinystories                              # print the plan, run nothing
python -m experiments.tinystories --device cpu --run           # CPU smoke: 2 gloo ranks, 10 MB of data, 100 steps
python -m experiments.tinystories --device 4090x4 --run --root runs
```

`tinystories` is the CS336 hw1 setup: download TinyStoriesV2-GPT4 (needs network) -> train a 10k BPE -> tokenize train
and validation caches -> pretrain `cs336-17m` (context 256) -> evaluate. Device presets:

| experiment | `--device` presets |
| --- | --- |
| `experiments.tinystories` | `cpu`, `4090x1` (DDP), `4090x4` (FSDP), `h100x8` (native FSDP) |
| `experiments.qwen35_gsm8k` | `cpu`, `4090x4`, `h100x8` (vLLM server on GPU 7, 7 training ranks; vLLM path untested on real hardware) |

`experiments.qwen35_gsm8k` is the hw5 setup on Qwen3.5-0.8B: GSM8K -> SFT -> GRPO from the SFT export -> greedy GSM8K
test accuracy of the SFT and GRPO policies. Its `cpu` preset is a pipeline smoke on a random-init `qwen3.5-tiny`.

Experiment flags (all experiments): `--root DIR` (default `runs`), `--run`, `--launcher {local,slurm}`,
`--partition`, `--account`, `--qos`, `--force NAME` (repeatable: rerun one step). Experiment-specific flags come from
its `Options` dataclass, e.g. `--device`, `--lr`, `--steps` (tinystories) or `--sft-lr`, `--grpo-lr`, `--sft-steps`,
`--grpo-steps` (qwen35_gsm8k). `python -m experiments.<name> --help` lists them.

Each step writes `<root>/<name>-<hash8>/` with `.status.json` (`RUNNING | SUCCESS | FAILED`), `.step.json`
(provenance), and for torchrun steps `config.yaml` and `launch.log`. Training steps also write `checkpoints/`, `final/`
(`model.safetensors` + `config.json`) and `metrics.jsonl`; evaluation steps write `eval.json`. Steps already at
`SUCCESS` are skipped; re-running a step that was killed or failed resumes it from its last committed checkpoint.

## Running one entry point directly

Every torchrun entry point is `python -m <module> --config_path <yaml> [--a.b value ...]` (draccus). The pipeline
already wrote the config for you at `<root>/<step>/config.yaml`; copy it or write your own.

| module | config class |
| --- | --- |
| `scratch_cs336.train.pretrain` | `PretrainConfig` |
| `scratch_cs336.posttrain.sft` / `.dpo` / `.grpo` | `SFTConfig` / `DPOConfig` / `GRPOConfig` |
| `scratch_cs336.posttrain.reward_eval` | `RewardEvalConfig` |
| `scratch_cs336.eval.perplexity` | `PerplexityConfig` |

```yaml
# pretrain.yaml
output_dir: runs/demo
model: cs336-17m
tokenizer: runs/tinystories-bpe-10k-<hash>       # BPE directory, "bpe:<dir>" or "hf:<repo>"
train_cache: runs/tinystories-train-tokens-<hash>
val_cache: runs/tinystories-valid-tokens-<hash>
seq_len: 256
parallel:
  mesh: {replicate: 1, shard: -1, tensor: 1}
  strategy: fsdp                                 # ddp | zero1 | fsdp
  backend: scratch                               # scratch | native
  compute_dtype: bfloat16
trainer:
  num_steps: 1000
  global_batch_size: 128
  micro_batch_size: 32
  optimizer: {lr: 0.003}
  schedule: {warmup_steps: 20, total_steps: 1000}
```

```bash
torchrun --standalone --nproc-per-node 4 -m scratch_cs336.train.pretrain \
  --config_path pretrain.yaml --trainer.num_steps 500 --trainer.schedule.total_steps 500 --parallel.backend native
```

Dotted flags follow the dataclass tree and override the YAML. Without `--config_path`, a nested config must be given
all its required fields on the command line (`parallel.strategy` and `parallel.backend`; `trainer.num_steps`,
`trainer.global_batch_size` and `trainer.micro_batch_size`), so prefer a YAML plus overrides. The schedule length is
its own field (`trainer.schedule.total_steps`); change it together with `num_steps`. On macOS add
`--local-addr 127.0.0.1` to `torchrun`. `--help` on any entry point lists every config field.

Post-training runs the same way, e.g. `-m scratch_cs336.posttrain.sft --config_path sft.yaml`. `policy.init_from` takes
an HF repo or snapshot directory, or a trainer-exported `final/` directory (SFT output feeding DPO or GRPO).

## Slurm

```bash
python -m experiments.qwen35_gsm8k --device h100x8 --run --launcher slurm --partition gpu --account lab --qos high
```

Each torchrun step becomes one `sbatch` job (`--nodes`, `--gpus-per-node`, `--time`, ... from the step's `Resources`;
one task per node), and `srun torchrun --rdzv-backend c10d` starts `nproc_per_node` ranks on every node. The runner
polls `sacct` until the job ends and `scancel`s it on Ctrl-C, so keep it alive (tmux) on the login node. CPU steps
(download, tokenizer, tokenization) run in that process. Rendered scripts are kept at `<step>/job.sbatch`, logs at
`<step>/slurm-<jobid>.out`. Node count and GPUs per node are part of the experiment's `Resources`, for example
`Resources(nnodes=2, nproc_per_node=4, gpus_per_node=4)`. For `module load` / venv lines or extra `#SBATCH` options,
call `run_steps(steps, root, SlurmLauncher(SlurmConfig(setup=(...), extra_sbatch=(...))), dry_run=False)` yourself.

Without the pipeline, run `torchrun` on every node with the same rendezvous:

```bash
torchrun --nnodes 2 --nproc-per-node 4 --rdzv-backend c10d --rdzv-id job1 --rdzv-endpoint node0:29500 \
  -m scratch_cs336.train.pretrain --config_path pretrain.yaml
```

## Single machine (RunPod, vast.ai, multi-4090 box)

`--launcher local` (the default) runs `torchrun --standalone --nproc-per-node <Resources.nproc_per_node>` on the
machine, so a pod needs only the install and `python -m experiments.<name> --device 4090x4 --run`. Add a device preset
to the experiment's `DEVICES` dict for another GPU count or strategy.

## Mesh

The device mesh has three named dims, outermost to innermost:

```
("dp_replicate", "dp_shard", "tp")        MeshConfig(replicate=1, shard=-1, tensor=1)
```

- `tp`: tensor-parallel peers, innermost so they share the fastest links (one node).
- `dp_shard`: FSDP / ZeRO shards of parameters, gradients and optimizer state.
- `dp_replicate`: pure replicas (DDP) or, together with `dp_shard > 1`, the "H" in HSDP.
- The data-parallel group is `dp_replicate x dp_shard`: the loader and the loss normalization use it.
- One dim may be `-1` and absorbs the remaining ranks; the product must equal the world size.

Ranks are laid out row-major, so with `nproc_per_node=4` the innermost dims fall within a node. Two nodes x four GPUs
(world size 8):

| goal | `parallel.mesh` | `strategy` |
| --- | --- | --- |
| FSDP over all 8 ranks | `{replicate: 1, shard: -1, tensor: 1}` (default) | `fsdp` |
| HSDP: shard in a node, replicate across nodes | `{replicate: 2, shard: 4, tensor: 1}` | `fsdp` |
| TP in a node, FSDP across nodes | `{replicate: 1, shard: 2, tensor: 4}` | `fsdp` |
| plain DDP | `{replicate: 1, shard: -1, tensor: 1}` | `ddp` |

TP needs the model's head counts and widths divisible by `tensor`; every entry point checks this against the model config
right after building the mesh, before loading weights. Qwen3.5-0.8B (`n_kv_heads=2`) supports `tensor` ∈ {1, 2}.
A checkpoint resumes only on the same world size and mesh; to change layouts load the full export (`final/`) instead.

## Layout

```
scratch_cs336/
  distributed/   init_distributed, DistEnv, MeshConfig / Mesh, run_distributed (spawn gloo ranks for tests)
  parallel/      ParallelConfig, parallelize, build_optimizer; ddp.py zero.py fsdp.py tp.py (scratch), native.py
  checkpoint/    Checkpointer (per-rank shards + commit marker), export_full
  models/        ModelConfig, presets, TransformerLM, attention, gated_deltanet, hf (HF weights), generate
  tokenizer/     bpe.py (parallel trainer + tokenizer), hf.py, load_tokenizer
  data/          cache, tokenize (parallel build), permutation, loader, mixture, chat
  train/         TrainerConfig, Trainer, optim (AdamW, schedules), pretrain entry
  posttrain/     data, losses, rewards, rollout, policy; sft / dpo / grpo / reward_eval entries
  eval/          perplexity (entry + evaluate_loss)
  pipeline/      step (Step, InputPath, hashing), runner (status, locks), builders, cli (experiment_main)
  launch/        Resources, LocalLauncher (torchrun), SlurmLauncher (sbatch + srun)
  tracking.py    jsonl + wandb trackers, rank 0 only
experiments/     tinystories.py, qwen35_gsm8k.py
tests/           mirrors the packages; multi-rank tests use run_distributed
docs/DESIGN.md   design and contracts
```

## Testing

```bash
.venv/bin/python -m pytest -q          # ~305 tests, ~90-170 s on an 8-core laptop; CPU, multi-rank tests on gloo
.venv/bin/python -m pytest tests/parallel -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Multi-rank tests spawn real process groups (`run_distributed`), so they cover the same code paths as a cluster run.
Tests that need the Qwen tokenizer skip when it is not in the local HF cache. `tests/models/test_qwen35_real.py` is
opt-in (`@pytest.mark.slow`): it checks the real Qwen/Qwen3.5-0.8B weights against Hugging Face, skips when the
checkpoint is not in the local HF cache, and `-m 'not slow'` deselects it.

## Acknowledgements

- Stanford CS336 (Language Modeling from Scratch) for the assignment contracts and reference behaviors.
- [Marin](https://github.com/marin-community/marin) and Levanter for the architecture this ports from JAX: the mesh,
  sharded state, token caches, and the step DAG.
- nanoGPT and nanochat for the minimal-training-loop sensibility.
- minimind for showing a small from-scratch LLM stack end to end.

## License

Apache-2.0, see `LICENSE`.
