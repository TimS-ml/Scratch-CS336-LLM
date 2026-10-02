# Agent guidelines

Conventions for coding agents working in this repository. Architecture and contracts are in `docs/DESIGN.md`; usage is
in `README.md`. Adapted from Marin's `AGENTS.md` / `TESTING.md` to a single-package PyTorch codebase.

## Commands

```bash
uv sync                                        # install (dev group included)
.venv/bin/python -m pytest -q                  # whole suite (~305 tests), CPU + gloo, ~90-170 s
.venv/bin/python -m pytest tests/parallel -q   # one directory / file / test while editing
.venv/bin/ruff check . && .venv/bin/ruff format .
```

- Python >= 3.12, line length 120, ruff rules `E F I UP B`. Every module starts with `from __future__ import annotations`.
- Do not replace the default pytest options or add `-m` expressions that hide failures; `slow` and `gpu` markers exist
  for tests that cannot run on CPU gloo.
- macOS: a hand-run `torchrun` needs `--local-addr 127.0.0.1` (hostname resolution can hang). `LocalLauncher` already
  passes it.
- Multi-rank code is not verified by importing it. Run it: `run_distributed(fn, world_size)` for a test, or the CPU
  preset of an experiment (`python -m experiments.tinystories --device cpu --run --root /tmp/exp`).

## Architecture rules

### Dependency direction

```
distributed -> parallel -> models, checkpoint -> train -> eval -> posttrain -> pipeline -> experiments
tokenizer -> data -> train                 tracking -> train                 launch -> pipeline
```

A package may import only from packages to its left. `models` imports exactly one thing from `parallel`
(`parallel.plan.TPStyle`); `data` knows only `tokenizer`; `launch` knows nothing but its own package; `pipeline` is the
only package that sees both `launch` and the entry points; `experiments/` only composes `pipeline.builders`. Known
wrinkle: `eval.perplexity` imports constants from `train.trainer`, and the `train.pretrain` entry point imports
`eval`; the generic loop (`train.trainer`, `train.config`, `train.optim`) itself never imports `eval`. Do not add reverse
edges; break cycles with a Protocol or by moving the shared constant down.

### World size is never special

- Every entry point is `cfg = draccus.parse(...)`, `env = init_distributed()`, `run(cfg, env)`, `destroy_distributed()`.
  A single-process run is a world of one with a real process group. Never write `if world_size == 1:` shortcuts or a
  separate non-distributed path.
- Functions that must be called on every rank say `Collective` in their docstring (`Trainer`, `Checkpointer`,
  `parallelize`, `full_state_dict`, rollout `sync_weights`). A collective reached on some ranks only is a hang, so
  conditionals around collectives must depend on rank-independent values (config, `step`, `mesh` sizes).
- Rank 0 does only I/O that other ranks do not need (tracker, commit marker, printing, `export_full`'s file). Compute
  is identical on all ranks.

### Models never mention ranks; the trainer never mentions strategies

- `models/` has no `torch.distributed`, no rank, no group. Parallelism reaches a model only through `tp_plan()`.
- `train/trainer.py` talks to a `ParallelModel` (`__call__`, `no_sync`, `finish_grad_sync`, `clip_grad_norm_`,
  `sharded_state_dict`, ...) and a `Mesh`; it never imports `Strategy` or `Backend`. Adding a strategy touches
  `parallel/` only.
- Losses return a `LossOutput(loss_sum, weight, metrics)`: a sum with gradient plus a weight (token or sequence count),
  never a per-micro-batch mean. The trainer normalizes by the global weight (`grad * dp_size / sum(weight)`), which is
  what makes uneven masks, accumulation and `dp_size` irrelevant to the result. Do not average inside a loss.
- Batch sources are `BatchSource`s: `batch(step)` is a pure, random-access function of the step (plus the state in
  `state_dict()`), returning this rank's rows. Never consume a stateful iterator.

### Tensor-parallel rule

Model forward code derives local head counts from weight shapes (`view(batch, seq, -1, head_dim)`), never from
cached `cfg.n_heads`-style ints. Sharding the weights along the head dimension is then all a module needs to be
tensor-parallel. `TransformerLM.tp_plan()` maps fnmatch patterns over module names to a `TPStyle`:

| style | parameters | communication |
| --- | --- | --- |
| `colwise` | `weight [out, in]` (+ `bias`) sharded on dim 0 | input enters the TP region: all-reduce of its gradient over `tp` |
| `rowwise` | `weight` sharded on dim 1, no bias | output all-reduced over `tp` |
| `headwise` | every parameter sharded on dim 0 | none (depthwise conv, per-head decay) |
| `replicate` | whole on every `tp` rank, used inside the region (per-head q/k norm) | parameter gradient summed over `tp` |

Lay out fused projections per head (`[query_h | gate_h]`) so a dim-0 shard keeps whole heads together. Embedding, final
norm and `lm_head` stay replicated and outside the region. A new module type is TP-ready when the sharded-forward-equals-
unsharded test passes for it (`tests/models/test_tensor_parallel_rule.py`, `tests/parallel/test_transformer_parallel.py`).
Each strategy has a scratch and a native implementation (`Backend.SCRATCH` / `Backend.NATIVE`); a behavior change in one
must keep both passing the same parity tests.

### Checkpoints and state

- State is stored the way it is sharded: `<run>/checkpoints/step_000100/rank_00000.pt ...` plus `metadata.json`
  (step, world size, mesh) that rank 0 writes last. A step directory without `metadata.json` does not exist.
- Files are written to a temporary name and renamed. The same rule holds for token-cache shards, ledgers, status and
  config files.
- Per-rank state must be `torch.save`/`weights_only=True`-loadable: tensors, numbers, strings, dicts, lists. Resume must
  be bitwise (model, optimizer, data source state, RNG, step). Resume requires the same world size and mesh; changing
  layout goes through `export_full` (`final/model.safetensors` + `config.json`).
- Anything that makes a run non-reproducible from `(config, seed, step)` is a bug, e.g. wall-clock seeds, iteration
  over unordered sets, rank-dependent RNG without `derive_seed`.

### Configs

- Configs are frozen dataclasses parsed by draccus. Nested configs compose by embedding, not inheritance. Fields have
  type annotations and, for entry-point configs, defaults; critical choices (`ParallelConfig.strategy` / `backend`,
  `TrainerConfig` sizes) have none and must be explicit.
- Enumerations are `StrEnum`s (`Strategy`, `Backend`, `TPStyle`, `RewardKind`, ...), not bare strings. YAML and CLI take
  the value (`fsdp`), and `__post_init__` or `Enum(value)` rejects unknown values early.
- No environment variables as configuration. The only reads of `os.environ` are rank discovery in
  `distributed/env.py`, which mirrors what torchrun / Slurm set, and passing a child process its environment. Put new
  knobs in a config dataclass.
- Step configs must be canonicalizable: dataclasses, lists, tuples, `dict[str, ...]`, enums, paths, `None`, bool, int,
  float, str. An entry-point step's config must have an `output_dir` field. Paths of other steps are `InputPath`, never
  hard-coded strings, so hashes and dependencies stay correct.
- draccus uses field comments as `--help` text and argparse %-formats it: a literal `%` in a field comment breaks `--help`.

### No backward compatibility

Change every call site; do not leave aliases, deprecated parameters, re-exports, `hasattr` probes or compatibility
shims. Delete code that a change makes dead, including its comments and docs. Resolve environment-dependent defaults once
and fail fast with `ValueError` on unknown input; do not catch exceptions to continue silently.

## Code style

- Prefer small functions and plain dataclasses; add abstraction only under real pressure. Protocols (`Tokenizer`,
  `WindowSource`, `BatchSource`, `Launcher`, `ParallelModel`, `Tracker`) decouple packages.
- Comments and docstrings explain contracts and non-obvious reasons, not what the next line does. Module docstrings state
  the layout/contract (see `data/cache.py`, `parallel/fsdp.py`). Delete stale comments on sight.
- All imports at the top of the file; local imports only for optional dependencies (`wandb`, `vllm`, `datasets`).
- No `*_utils.py`; name modules after what they hold. Top-level constants for magic strings and numbers.
- Compute dtypes are explicit: parameters and optimizer state stay fp32, `ParallelConfig.compute_dtype` owns mixed
  precision (the trainer uses no `autocast`), losses are computed in fp32.
- Avoid avoidable copies and host syncs in hot paths: no `.item()` / `.cpu()` per micro-step; the trainer syncs once per
  optimizer step.

## Testing

Read the Marin testing policy in spirit: a test must fail when behavior is wrong, and not when an implementation detail
changes. Tests here check **behavior and parity**.

- Parity against an independent reference: every strategy x backend (and `tp x dp`, HSDP) equals a single-process run
  over several optimizer steps; model logits equal HF `transformers`; `AdamW` equals `torch.optim.AdamW`; the chunked
  Gated DeltaNet equals the token-by-token recurrence; BPE equals the hw1 reference.
- Invariants and transitions: a batch is a pure function of the step; ranks partition the global batch; each window
  appears once per epoch; a killed run resumes bitwise; an interrupted token-cache build resumes; changing a config forks
  every downstream step hash; a lock is never stolen from a live process.
- Round trips through public APIs: save then load, dump a config then parse it, export then load strictly.
- Multi-rank tests use `scratch_cs336.distributed.spawn.run_distributed(fn, world_size, *args)` on gloo: `fn` is a
  module-level function `fn(env, *args)` returning picklable per-rank results, and the test asserts on those results in
  the parent. Put the case grid inside one spawned world when startup dominates. Use world size 2 or 4.
- Each test finishes in under 60 s (`pyproject.toml` caps at 120 s); the suite stays near 90 s. Use tiny models and a few
  steps. No `time.sleep`; inject fakes for clocks, `sbatch` / `sacct` (`SlurmLauncher(runner=..., sleep=...)`), and HTTP.
- Do not weaken tolerances to make a test pass; find the numerical cause. Tests needing a downloaded checkpoint or
  tokenizer skip when it is not in the local HF cache; they never download.
- Do not write wiring tests (a function forwards its arguments), tautologies, tests of private state, default values,
  log text, command-line text, or configuration echoes. A validation guard needs a test only for a reported regression.
  Delete tests that pin implementation details instead of re-pinning them. Probes of your own change belong in throwaway
  scripts, not in `tests/`.
- Put a test in the file that mirrors the code (`tests/parallel/...` for `parallel/`), use top-level `def test_*`
  functions named for subject and expected behavior, and extend an existing file before creating one.

## Experiments and pipeline

- An experiment file under `experiments/` defines `Options` (frozen dataclass, every field defaulted), a `DEVICES` preset
  table, `build(opts) -> list[Step]`, and ends with `experiment_main(build, options=Options)`. Hardware belongs in presets
  (`Resources`, `ParallelConfig`, batch sizes), not in step builders.
- `pipeline/builders.py` owns how and where a step runs; experiments own what runs. CPU work is a `run=` step in the
  runner process, torchrun work is an `entrypoint=` step launched with its `Resources`.
- Changing anything in a step's config (including defaults of its dataclasses) changes its hash and forks its output
  path and everything downstream: that is the intended cache invalidation, so never edit a config dataclass default
  casually.
- Run outputs (`runs/`, `outputs/`, `checkpoints/`, `data/`, `wandb/`) are git-ignored; never commit them.

## Docs

Keep `README.md` (usage) and `docs/DESIGN.md` (why and contracts) in step with code in the same change. Documented
commands and flags must exist; verify them against the argparse / draccus definitions. No emojis, no marketing prose.
Do not create new planning or status documents in the repository.
