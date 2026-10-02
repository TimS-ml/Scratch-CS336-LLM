# Design: distributed-first LLM training from scratch

PyTorch reimplementation of the CS336 (Spring 2026) pipeline, organized the way Marin/Levanter organize
large-scale training. Two sources, two roles:

- **CS336 sp26** gives the *contracts*: what a from-scratch DDP / ZeRO-1 / FSDP / BPE / SFT / DPO / GRPO must do
  (hw1 basics, hw2 systems, hw4 data, hw5 alignment). We implement them ourselves.
- **Marin** gives the *architecture*: a declarative mesh, sharded state with deterministic resume,
  sharded token caches with ledgers, and an experiment DAG whose steps own versioned output paths.
  Marin is JAX; every idea here is ported to `torch.distributed` (`DeviceMesh`, process groups, DTensor).

This document records why the pieces look the way they do and the contracts between them. Usage is in `README.md`,
conventions for contributors in `AGENTS.md`.

## Principles

1. **World size is never special.** Every entry point calls `init_distributed()`. A plain `python -m` run is a world
   of one with a real process group; tests run world 2/4 on CPU with gloo. There is no single-process code path to
   drift from the distributed one.
2. **Parallelism is configuration.** `MeshConfig(replicate, shard, tensor)` + `ParallelConfig(strategy, backend)`
   decide layout. Model code never mentions ranks; trainer code never mentions strategies. A model reaches parallelism
   only through `tp_plan()`, the trainer only through the `ParallelModel` protocol.
3. **Every rank reads only its data, as a pure function of the step.** Loaders compute a rank's rows of a global batch
   from `(seed, step, dp_rank)`. Resume restores `step` (plus whatever state a source genuinely carries).
4. **State is sharded on disk the way it is sharded in memory.** Each rank writes its own shard; rank 0 commits
   metadata last. A full, unsharded export exists for eval, HF and vLLM, and is the only path across mesh layouts.
5. **Scratch first, native as reference.** Each strategy has a from-scratch implementation (`backend=scratch`) and a
   torch-native one (`backend=native`: DDP, `ZeroRedundancyOptimizer`, FSDP2 `fully_shard`, DTensor TP). Tests assert
   both equal a single-process run.
6. **Pipelines are data.** Experiments compose `Step`s; a step's output path is a hash of its config and its
   dependencies, so re-running is a cache hit and changing a hyperparameter forks a new path.
7. **Global-weight normalization.** A loss is a sum plus a weight; the trainer divides by the global weight. Results do
   not depend on dp size, accumulation or how masked tokens fall across ranks.

## Layout

```
scratch_cs336/
  distributed/   env.py (init_distributed, DistEnv), mesh.py (MeshConfig, Mesh, build_mesh), spawn.py (run_distributed)
  parallel/      plan.py (TPStyle), api.py (ParallelConfig, Strategy, Backend, parallelize, build_optimizer,
                 ParallelModel), ddp.py, zero.py, fsdp.py, tp.py (scratch), native.py (torch-native)
  checkpoint/    sharded.py (Checkpointer), export.py (export_full)
  models/        config.py (ModelConfig, LayerType), presets.py (PRESETS), layers.py, attention.py,
                 gated_deltanet.py, transformer.py (TransformerLM), hf.py (HF weights), generate.py
  tokenizer/     bpe.py (train_bpe, BPETokenizer), hf.py (HFTokenizer), __init__ (Tokenizer, load_tokenizer)
  data/          cache.py (TokenCache + ledger), tokenize.py (build_token_cache), permutation.py, loader.py
                 (PretrainLoader, WindowSource), mixture.py (MixtureSource), chat.py (ChatML + masks)
  train/         config.py (TrainerConfig), optim.py (AdamW, schedules, param groups), trainer.py (Trainer,
                 BatchSource, LossOutput), pretrain.py (entry)
  eval/          perplexity.py (evaluate_loss, load_exported, entry)
  posttrain/     data.py (SFT / preference / prompt sources), losses.py, rewards.py, rollout.py (engines),
                 policy.py (PolicyConfig, load_policy), sft.py, dpo.py, grpo.py, reward_eval.py (entries)
  pipeline/      step.py (Step, InputPath, hashing), runner.py (status, locks, plan), builders.py, cli.py
                 (experiment_main)
  launch/        resources.py (Resources), local.py (torchrun), slurm.py (sbatch + srun torchrun)
  tracking.py    Tracker protocol: jsonl + wandb, rank 0 only
experiments/     tinystories.py (hw1), qwen35_gsm8k.py (hw5: SFT -> GRPO on Qwen3.5-0.8B)
tests/           mirrors the packages; multi-rank tests use run_distributed(world_size=2|4)
```

Dependency direction: `distributed` -> `parallel` -> `models` / `checkpoint` -> `train` -> `eval` -> `posttrain` ->
`pipeline` -> `experiments`, with `tokenizer` -> `data` -> `train`, `tracking` -> `train` and `launch` -> `pipeline`.
`models` imports only `parallel.plan` (for `TPStyle`). `eval.perplexity` reads two constants from `train.trainer`
and the `train.pretrain` entry point imports `eval`; the generic loop itself never imports `eval`.

## Mesh

```
("dp_replicate", "dp_shard", "tp")   # outermost -> innermost;  MeshConfig(replicate=1, shard=-1, tensor=1)
```

- `tp` innermost: TP peers are adjacent ranks and share the fastest links (one node).
- `dp_shard`: FSDP/ZeRO shards. `dp_replicate`: pure replicas; with `dp_shard > 1` this is HSDP (replicate across
  nodes, shard within).
- `Mesh.dp_group` is the flattened `dp_replicate x dp_shard`; the loader (`dp_rank`, `dp_size`) and loss normalization
  use it. `Mesh.shard_group`, `replicate_group`, `tp_group` serve the strategies.
- At most one dim may be `-1`; it absorbs the remaining ranks (Levanter's `data: -1`). The default puts every rank in
  `dp_shard`. The product must equal the world size.
- Example, 2 nodes x 4 GPUs: `{replicate: 2, shard: 4}` is HSDP with shards inside a node; `{shard: 2, tensor: 4}` is TP
  inside a node with FSDP across nodes.

Target hardware: a single node (multi-4090 box, RunPod/vast.ai pods) via `torchrun`, multi-node via Slurm.

## Contracts

### Models (`scratch_cs336/models`)

- `ModelConfig` (frozen dataclass) covers CS336 basics, Qwen3 and Qwen3.5 text: `vocab_size, d_model, n_layers, n_heads,
  n_kv_heads, head_dim, d_ff, max_seq_len, rope_theta, partial_rotary_factor, norm_eps, qk_norm, attn_output_gate,
  zero_centered_norm, tie_embeddings, layer_types` plus Gated DeltaNet dims (`linear_n_k_heads, linear_n_v_heads,
  linear_k_head_dim, linear_v_head_dim, linear_conv_kernel`). `layer_types` mixes `full_attention` and
  `linear_attention` layers, which is how Qwen3.5 interleaves three Gated DeltaNet layers with one gated attention layer.
  `PRESETS` names the configurations (`cs336-tiny`, `cs336-17m`, `cs336-hw4`, `qwen3-0.6b`, `qwen3.5-0.8b`, `qwen3.5-tiny`).
- `TransformerLM(cfg)`: `forward(input_ids [B,T], position_ids [B,T] | None) -> logits [B,T,V]`; `hidden_states` and
  `logits` are split so callers can project only the positions they need. Submodules `embed`, `blocks: nn.ModuleList`,
  `final_norm`, `lm_head` (absent when tied). Parameters are allocated uninitialized; `init_weights(seed)` is
  deterministic and runs on the full model *before* sharding. `flops_per_token(seq_len)` feeds MFU.
- **Gated DeltaNet layout** differs from HF on purpose: q, k, v have separate projections and separate depthwise convs,
  and `A_log` / `dt_bias` live in their own `decay` module, so every per-head parameter sits in a module a TP plan can
  shard by heads. `hf.py` converts both ways (`convert_hf_state_dict`, `to_hf_state_dict`). Attention packs the output
  gate per head (`[query_h | gate_h]`, HF layout) so a dim-0 shard keeps whole heads together. The chunked delta rule is
  tested against its token-by-token recurrence.
- **TP rule:** forward derives local head counts from parameter shapes, never from cached ints, so sharding weights
  along the head dimension is enough to make a module tensor-parallel.
- `TransformerLM.tp_plan() -> dict[str, TPStyle]`: fnmatch patterns over module FQNs. Four styles:
  - `COLWISE`: `weight [out,in]` (and `bias`) sharded on dim 0; the module input enters the TP region (identity forward,
    all-reduce of its gradient over `tp`).
  - `ROWWISE`: `weight` sharded on dim 1, no bias; the partial output is all-reduced over `tp` (identity backward).
  - `HEADWISE`: every parameter sharded on dim 0, no communication (depthwise convs, per-head decay vectors).
  - `REPLICATE`: parameters stay whole on every `tp` rank but are used inside the region on head-sharded activations
    (per-head q/k norm, the Gated DeltaNet output norm). Each rank sees only a partial gradient, so the parameter itself
    enters the region and its gradient is summed over `tp`.
  Embedding, final norm and `lm_head` are unlisted: replicated and used outside the region. A tied parameter cannot be
  tp-sharded (`check_shardable` rejects it), which is why the tied embedding stays unlisted. The scratch backend shards in place and installs hooks (`tp.py`); the native
  backend uses DTensor `parallelize_module` but stores plain local shards, so shapes seen by model code are identical in
  both.
- `ModelConfig.check_tensor_parallel(tp)` requires `tp` to divide `d_ff` and every head count (attention q/kv, Gated
  DeltaNet k/v), since TP shards whole heads. Entry points call it on the config (`pretrain.model_config`,
  `policy.policy_config`, `perplexity.exported_config`, all weight-free) right after building the mesh.
- `hf.load_hf_pretrained(repo_or_path, dtype) -> (ModelConfig, state_dict)` maps HF Qwen3 / Qwen3.5 text keys (also from
  the multimodal Qwen3.5 checkpoint's `model.language_model.*`); `hf.load_hf_config` reads only `config.json`.
- `generate.generate(...)` is a reference sampler (full recompute, no KV cache).

### Parallel (`scratch_cs336/parallel`)

```python
class Strategy(StrEnum): DDP, ZERO1, FSDP
class Backend(StrEnum):  SCRATCH, NATIVE
@dataclass(frozen=True)
class ParallelConfig:
    mesh: MeshConfig; strategy: Strategy; backend: Backend      # strategy / backend have no defaults
    compute_dtype: str = "float32"      # "float32" | "bfloat16"; params, grads, optimizer state stay fp32
    activation_checkpointing: bool = False

def parallelize(model, mesh, cfg) -> ParallelModel
def build_optimizer(pmodel, cfg, optimizer_cls, params=None, **kw) -> torch.optim.Optimizer

class ParallelModel(Protocol):
    module: nn.Module                                   # local (TP-sharded) module
    def __call__(*a, **kw)                              # forward; owns mixed precision, no autocast in the trainer
    def parameters() -> list[nn.Parameter]              # what the optimizer owns (local shards)
    def no_sync() -> ContextManager                     # skip grad comm on accumulation micro-steps
    def finish_grad_sync() -> None                      # after the last backward, before clipping / step
    def clip_grad_norm_(max_norm) -> float              # global L2 norm over every dp and tp shard, each element once
    def sharded_state_dict() / load_sharded_state_dict(sd)   # per rank, torch.save-able
    def full_state_dict() -> dict[str, Tensor]          # collective: unsharded, on every rank, CPU
```

`parallelize` is collective and does, in order: broadcast rank 0's parameters and buffers (all strategies start
identical); activation checkpointing of each `blocks[i]`; TP from `tp_plan()` (only if `tp > 1`); the data-parallel
strategy on the local module.

- **DDP**: all-reduce over the flattened dp group, bucketed and overlapped with backward; `finish_grad_sync` launches
  stragglers and averages.
- **ZeRO-1**: DDP gradients; optimizer state sharded over `dp_shard` by greedy size-balanced ownership, owners broadcast
  updated parameters. `build_optimizer` returns the sharded optimizer.
- **FSDP / HSDP**: units are the children of `model.blocks` plus the root; each unit flattens its parameters into one fp32
  master buffer, shards it over `dp_shard`, all-gathers in `compute_dtype` per forward and re-gathers for backward,
  reduce-scatters gradients, and with `dp_replicate > 1` all-reduces across replicas. Gradients are always averaged over dp.
- Mixed precision lives here: FSDP casts at all-gather, DDP/ZeRO run the forward under `torch.autocast`. TP composes first,
  then DP/FSDP act on the TP-local parameters.
- `full_state_dict` gathers over dp, then over tp, so `export_full` and the rollout engines see one unsharded model.

### Trainer (`scratch_cs336/train`)

`Trainer(cfg, pmodel, optimizer, train: BatchSource, loss_fn, mesh, output_dir, evaluators, flops_per_token, on_step_end)`
is the only training loop; pretraining, SFT, DPO and GRPO differ only in the `BatchSource` and `LossFn` they pass.

```python
class BatchSource(Protocol):                    # deterministic and random-access by optimizer step
    def batch(step) -> dict[str, Tensor]        # THIS rank's slice of the global batch; same row count in every field
    def state_dict() -> dict; def load_state_dict(state) -> None

@dataclass
class LossOutput:
    loss_sum: Tensor     # scalar with grad: SUM of per-unit losses in this micro-batch
    weight: Tensor       # scalar: number of units (tokens, sequences, pairs) in this micro-batch
    metrics: dict[str, Tensor]   # extra sums, reported as global_sum / global_weight

LossFn = Callable[[ParallelModel, Batch], LossOutput]
```

Per optimizer step:

1. `batch(step)` gives `global_batch_size / dp_size` rows; they are split into `grad_accum` micro-batches of
   `micro_batch_size` rows (`global_batch_size` must divide by `micro_batch_size * dp_size`).
2. Each micro-batch backpropagates `loss_sum` (a sum, never a mean); `no_sync()` covers all but the last.
3. After `finish_grad_sync()`, `loss_sum`, `weight` and the metric sums are all-reduced over dp, and gradients are scaled by
   `dp_size / sum(weight)`. The parallel model has already averaged over dp (divided by `dp_size`), so the result is exactly
   `sum(grad) / global_weight`: the gradient of the global mean loss, however unevenly masked units fall across ranks and
   micro-batches. Losses therefore define their own unit: tokens for LM and SFT, pairs for DPO, sequences or a fixed constant
   for GRPO.
4. `clip_grad_norm_`, optimizer step, learning rate from `lr_at(step)`, then metrics (`loss`, `lr`, `grad_norm`, `weight`,
   throughput, MFU on known GPUs) to the tracker.

Evaluators (`name -> fn(pmodel) -> metrics`) run collectively on `eval_every` and at the end. A checkpoint holds `model`,
`optimizer`, `train` (the source's `state_dict`), `rng` and `step`, so a killed run resumes bitwise. At the end the trainer
writes `final/` (`model.safetensors` + our `ModelConfig` as `config.json`), the input of eval, SFT -> DPO/GRPO hand-off and
vLLM sync. `OptimizerConfig` / `ScheduleConfig` select AdamW (scratch or torch) and cosine / WSD / constant;
`param_groups` decays matrices only.

### Checkpoint (`scratch_cs336/checkpoint`)

`<run>/checkpoints/step_000100/rank_00000.pt ... + metadata.json`. Each rank writes its own file; the ranks then
all-reduce a failure count, and only if every shard landed does rank 0 write `metadata.json` (step, world size, mesh
shape) and apply retention. Any failure makes every rank raise (from `wait()`/the next `save()` when async) instead of
hanging. A directory without `metadata.json` is incomplete and ignored.
`Checkpointer(root, mesh, keep_last, permanent_every, async_save)`: `save(step, state)` snapshots tensors to CPU
synchronously and (by default) writes on a thread coordinated over a dedicated gloo group; `wait()`, `latest_step()`,
`load(step)`. Loading rejects another world size or mesh; changing layout goes through `export_full`.

### Data (`scratch_cs336/data`, `scratch_cs336/tokenizer`)

- `Tokenizer` protocol: `vocab_size, eos_token_id, encode, decode, encode_iterable`; `load_tokenizer("bpe:<dir>" |
  "hf:<repo_or_path>" | <bpe dir>)`. `train_bpe` is parallel over document-aligned chunks; `HFTokenizer` serves Qwen/GPT-2.
- Token cache: `<dir>/shard_00000.bin` (little-endian uint16, or uint32 when the vocabulary needs it), per-shard
  `shard_00000.json` written after the `.bin` (resume marker) and `ledger.json` (`dtype`, shards, `finished`) written last.
  Every file lands via temp-and-rename. `build_token_cache` tokenizes shards in parallel and skips shards whose entry matches
  the same source range, the same blake2b digest of that range's bytes, and dtype; an unfinished cache cannot be opened.
- `WindowSource` (`num_windows(seq_len)`, `window(i, seq_len) -> np.ndarray[seq_len+1]`) is implemented by `TokenCache` and
  `MixtureSource`.
- `PretrainLoader(source, seq_len, global_batch_size, dp_rank, dp_size, seed)`: global sample `g = step * G + j` is window
  `perm_{seed, g // N}(g % N)` (stateless Feistel permutation per epoch, O(1) memory); rank `r` owns
  `j in [r*G/dp, (r+1)*G/dp)`. Every window appears once per epoch.
- `MixtureSource(sources, weights, seed, block_size)`: every block holds exactly the rounded per-source counts, shuffled
  within the block; it ends when the scarcest source runs out.
- `chat.py`: ChatML rendering (Qwen format) and a 0/1 mask on assistant content and its `<|im_end|>`.

### Post-training (`scratch_cs336/posttrain`)

All three entries are `python -m scratch_cs336.posttrain.<sft|dpo|grpo>` and build a `Trainer`; they reuse
`train.pretrain.lm_loss` or define a `LossFn`. `PolicyConfig(init_from, model, vocab_size, tokenizer)` initializes the policy
from an HF repo/snapshot, a trainer `final/` export, or a preset with random init. Scratch and exported models need the
tokenizer vocabulary to fit the model; HF weights need their own tokenizer (`hf:<init_from>`) or one with exactly the
checkpoint's vocabulary (`check_tokenizer_vocab`, also used by pretraining's `init_from_hf`).

- **Data sources** follow the pretraining loader's scheme (global sample `g = step*G + j` is example `perm(g mod N)`, rank
  `r` owns a contiguous slice), right-pad to the longest row of the rank's slice, and mark padding with `IGNORE_INDEX`:
  `SFTSource` (`input_ids, labels, loss_mask`; chats from `messages`, `prompt/response`, or `question/answer` rows),
  `PreferenceSource` (chosen / rejected input ids and labels; only the final response is trained), `PromptSource`
  (RL prompts plus ground truths). Their state only pins data identity: resuming on different data is an error.
- **SFT**: token-mean cross entropy on assistant tokens, i.e. `lm_loss` with weight = trained tokens.
- **DPO**: the reference is a frozen copy of the initial policy parallelized with the same mesh and run forward-only inside
  the batch source (`ReferenceScored`), so each batch carries reference log-probs. No precompute pass, nothing to invalidate,
  the source stays a pure function of the step; the price is one sharded forward per step and the reference's fp32 weights.
  Loss = sum of per-pair DPO losses, weight = pairs; metrics are implicit rewards, margin and accuracy.
- **GRPO** (and Dr. GRPO, RFT, MaxRL, GRPO-clip, GSPO through `GRPOAlgorithm` enums: baseline, advantage normalizer,
  importance reweighting, loss normalization): rollouts are a `BatchSource` (`RolloutBatches`). Rollout batch `k`
  (`n_prompts_per_rollout x group_size` samples) is generated lazily when the Trainer first asks for it, from the current
  policy via `RolloutEngine.sync_weights(pmodel)` then `generate`, and serves
  `epochs_per_rollout_batch * rollout_rows / global_batch_size` consecutive optimizer steps. Each dp rank samples, scores
  and group-normalizes its own prompt shard, so groups never straddle ranks. The rollout is part of the source state, so
  a resume mid-rollout replays identical data. `trainer.global_batch_size` counts rollouts per step. Loss weight is
  sequences (`sequence` normalization) or the constant `Z * rows / G` (`constant`), so the global weight is fixed.
  `old_log_probs` are computed once per rollout, only when importance reweighting is on.
- **Rollout engines** (`RolloutEngine`: `generate(prompts, n, params)`, collective `sync_weights(pmodel)`):
  `TorchRolloutEngine` samples on every dp rank with an unsharded local copy rebuilt from `full_state_dict()` (batched,
  right-padded, full recompute per token: exact but slow). `VLLMRolloutEngine` ports the hw5 staff `vllm_utils`: rank 0
  starts a `vllm serve` process on a dedicated GPU, every rank sends its prompt shard over HTTP, and `sync_weights` gathers
  the full policy and broadcasts HF-named bf16 weights over NCCL from rank 0. Untested on real hardware. It needs Linux, CUDA and the `vllm` extra,
  and the server owns a GPU that the training ranks do not use.
- **Rewards** (`rewards.py`): GSM8K-style answers compared as exact rationals; `r1_zero` (format + answer),
  `boxed`, `last_number`; a toy `target_token` reward lives in `grpo.py` for tests. Prompt formats (`r1_zero`, `boxed`, `chat`)
  pair with a reward and a stop string.
- **`reward_eval`**: greedy reward / accuracy of a policy on held-out prompts through the same engines as GRPO's in-training
  evaluator (`RolloutEvaluator`); writes `eval.json`.
- `losses.py` holds the hw5 objectives as plain tensor functions; they carry no distributed code.

### Pipeline / launch

- `Step(name, config, run=fn | entrypoint="module", resources=Resources(...))`. `InputPath(step, subpath)` inside a config
  references another step's output; the output path is `<root>/<name>-<hash8>` where the hash covers the canonical config
  with references replaced by dependency names and hashes. Configs must be canonicalizable (dataclasses, lists, tuples,
  `dict[str, ...]`, enums, paths, primitives) and the root never enters the hash.
- Runner (`run_steps`): topological order, `.status.json` (`RUNNING | SUCCESS | FAILED`), exclusive `.lock` (a lock left by
  a dead local pid is taken over, any other is an error), `.step.json` provenance, skip on `SUCCESS`, `--force NAME` reruns only
  that step, and the plan is printed unless `--run`.
- Entry-point steps: the runner writes `<out>/config.yaml` (draccus, enums as values) with `output_dir` filled in and calls
  `Launcher.run(module, config_path, resources, log_dir)`. `LocalLauncher` runs `torchrun --standalone --local-addr 127.0.0.1`;
  `SlurmLauncher` submits one `sbatch` job per step whose script runs `srun torchrun --rdzv-backend c10d` on every node
  (head node from `scontrol`), polls `sacct`, and `scancel`s on interrupt. `run=` steps execute in the runner process.
- `pipeline/builders.py` owns *how and where*: `hf_download`, `train_tokenizer`, `tokenize`, `pretrain`, `evaluate`, `gsm8k`,
  `sft`, `dpo`, `grpo`, `reward_eval`, plus `final_model(step)` (the `final/` export as an `InputPath`). Experiments own *what*:
  names, data, model, hyperparameters. `sft(policy=PolicyConfig(init_from=final_model(...)))` chains stages through hashes.
- `experiment_main(build, options=Options)` is the CLI: `--root`, `--run`, `--launcher {local,slurm}`, `--partition`,
  `--account`, `--qos`, `--force NAME`, plus one `--field-name` flag per field of the `Options` dataclass (bools become
  on/off switches, enums choices). Without `--run` it only prints the plan.
- Experiments define a `DEVICES` preset table (`Resources`, `ParallelConfig`, batch sizes) selected by `--device`.
  `tinystories` is hw1 (`cs336-17m`, 10k BPE); `qwen35_gsm8k` is hw5 (SFT on train solutions in the `r1_zero` format, then GRPO,
  then greedy test accuracy of both policies; SFT and GRPO may use different layouts because vLLM takes a GPU from training).

## Testing

- CPU + gloo, `run_distributed(fn, world_size)`; parity tests compare every strategy x backend, and the 2-D layouts (TP x DP,
  HSDP), to a single-process reference over several optimizer steps with accumulation and clipping.
- Model parity against HF `transformers` on small random configs (no downloads), plus the chunked Gated DeltaNet against
  its recurrence. `tests/models/test_qwen35_real.py` (`slow`, skipped without the checkpoint in the local HF cache) checks the
  real Qwen3.5-0.8B weights: fp32 logits within 1e-4, bf16 as close to the fp32 reference as HF's bf16, same greedy tokens.
- Invariant tests for determinism and resume: loader purity, bitwise resume from a killed trainer and from mid-rollout GRPO,
  interrupted cache builds, hash stability across processes.
- Behavior tests only: no tests of wiring or incidental defaults.
