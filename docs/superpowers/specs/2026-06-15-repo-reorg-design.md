# Repo Reorganization Design — `clean_llm` → `scratch_cs336`

Date: 2026-06-15
Status: Approved (full overhaul)

## 1. Goal

Reorganize this LLM-from-scratch repo from its current "dirty" state into a clean,
consistent layout. This is a **full overhaul**: redesign package boundaries, unify the
data pipeline, drop Hydra in favor of plain YAML, prune docs, and rename the package
`clean_llm` → `scratch_cs336` (pyproject `name = "scratch_cs336"`).

Clean break: **no backward-compatible `clean_llm` aliases**. All imports, scripts, and
docs are updated. Use `git mv` for moves to preserve history.

## 2. Decisions (locked)

| # | Decision |
|---|----------|
| 1 | Scope: **full overhaul** |
| 2 | Entry points: **thin `scripts/`** + top-level `configs/`; tests → `tests/` |
| 3 | Package internals: **Approach B** — library vs. pipeline layers (`core/`, `training/`, `data/`, `eval/`, `serve/`) |
| 4 | Data: **single top-level `data/`** holding data only; all processing **code** moves into `scratch_cs336/data/`; delete broken `data_sft/preprocess.py` stub |
| 5 | Config: **drop Hydra everywhere**; load plain YAML via `OmegaConf.load`; remove `hydra-core` dep |
| 6 | Docs: **consolidate + prune** (delete AI-artifact reports, fold in-package READMEs into `docs/`) |
| 7 | Examples: **keep** in `examples/` |
| 8 | `pyproject.toml`: rename + add `[build-system]` (hatchling) + package discovery; remove `hydra-core`, add `omegaconf` explicitly |
| 9 | No `clean_llm` compatibility shims (clean break) |

## 3. Target structure

```
Boring-CS336-LLM-from-Scratch-Extend/
├── scratch_cs336/                # renamed from clean_llm/, regrouped (Approach B)
│   ├── __init__.py
│   ├── utils.py                  # device + mlflow/omegaconf helpers (was clean_llm/utils.py)
│   ├── core/                     # reusable primitives
│   │   ├── __init__.py
│   │   ├── models/               # basics.py, cs336_lm.py, qwen2_5.py  (+ NEW __init__.py)
│   │   ├── tokenizer/            # tokenizer.py, train.py, train_fast.py, train_chinese.py,
│   │   │                         #   merge_vocab.py, expand_embedding.py, utils.py, __init__.py
│   │   ├── generation/           # processors.py, streaming.py, utils.py, __init__.py
│   │   └── quantize/             # gptq.py, __init__.py   (example.py → examples/)
│   ├── training/                 # training pipeline
│   │   ├── __init__.py
│   │   ├── pretrain.py   sft.py   adapters.py
│   │   ├── rm.py                 # was rm_train.py
│   │   ├── dpo.py                # was dpo_train.py
│   │   └── datasets.py           # was rlhf_datasets.py
│   ├── data/                     # data PROCESSING code
│   │   ├── __init__.py
│   │   ├── preprocess.py         # real impl (replaces broken data_sft stub)
│   │   └── processors/           # pretrain_processor.py, rm_processor.py, sft_processor.py, __init__.py
│   ├── eval/                     # + NEW __init__.py
│   │   ├── __init__.py
│   │   └── pretrain.py           # was eval/eval_pretrain.py
│   └── serve/                    # was demo/
│       ├── __init__.py
│       ├── chat.py   web_ui.py
├── scripts/                      # THIN entry points only (plain YAML, no Hydra, no sys.path hacks)
│   ├── pretrain.py   tokenize.py   train_tokenizer.py   eval_pretrain.py
│   ├── train_sft.py   train_rm.py   train_dpo.py   train_grpo.py
│   ├── quantize_model.py   launch_demo.py
│   ├── merge_tokenizers.py   expand_model_vocab.py
│   └── process_pretrain_data.py   process_rm_data.py   process_sft_data.py
├── configs/                      # moved from scripts/configs/ (+ deepspeed/), Hydra-isms removed
├── data/                         # DATA only (no code)
│   ├── README.md                 # documents committed vs generated/downloaded
│   ├── download/                 # download_owt.sh, download_ts.sh
│   ├── pretrain/                 # gitignored (owt, TinyStories)
│   └── sft/gsm8k/                # committed parquet (from data_sft/openai/gsm8k/)
├── tests/                        # test_dpo_setup.py, test_qwen2_5.py (de-scripted)
├── examples/                     # tokenizer_workflow.py, quantize_example.py
├── docs/                         # pruned + consolidated
├── pyproject.toml   pipeline.sh   README.md   LICENSE   uv.lock
```

## 4. File moves (use `git mv`)

### 4.1 Package: `clean_llm/` → `scratch_cs336/`

| From | To |
|------|----|
| `clean_llm/__init__.py` | `scratch_cs336/__init__.py` |
| `clean_llm/utils.py` | `scratch_cs336/utils.py` |
| `clean_llm/models/{basics,cs336_lm,qwen2_5}.py` | `scratch_cs336/core/models/` (+ new `__init__.py`) |
| `clean_llm/tokenizer/*` (code) | `scratch_cs336/core/tokenizer/` |
| `clean_llm/generation/*` | `scratch_cs336/core/generation/` |
| `clean_llm/quantize/gptq.py`, `__init__.py` | `scratch_cs336/core/quantize/` |
| `clean_llm/quantize/example.py` | `examples/quantize_example.py` |
| `clean_llm/train/pretrain.py` | `scratch_cs336/training/pretrain.py` |
| `clean_llm/train/sft.py` | `scratch_cs336/training/sft.py` |
| `clean_llm/train/adapters.py` | `scratch_cs336/training/adapters.py` |
| `clean_llm/train/rm_train.py` | `scratch_cs336/training/rm.py` |
| `clean_llm/train/dpo_train.py` | `scratch_cs336/training/dpo.py` |
| `clean_llm/train/rlhf_datasets.py` | `scratch_cs336/training/datasets.py` |
| `clean_llm/data/processors/*` | `scratch_cs336/data/processors/` |
| `clean_llm/eval/eval_pretrain.py` | `scratch_cs336/eval/pretrain.py` (+ new `__init__.py`) |
| `clean_llm/demo/{chat,web_ui}.py`, `__init__.py` | `scratch_cs336/serve/` |

New files: `scratch_cs336/core/__init__.py`, `core/models/__init__.py`, `eval/__init__.py`,
`data/__init__.py`, `data/preprocess.py` (real, replacing stub).

### 4.2 In-package READMEs → `docs/`

| From | To |
|------|----|
| `clean_llm/tokenizer/README.md` | `docs/TOKENIZER.md` |
| `clean_llm/quantize/README.md` | `docs/QUANTIZATION.md` |
| `clean_llm/train/DPO_TRAINING_README.md` | merge into `docs/DPO_QUICK_START.md` |
| `scripts/README_RM_TRAINING.md` | `docs/RM_TRAINING.md` |

(Update the `from clean_llm...` snippets inside these docs to new paths.)

### 4.3 scripts / configs / tests / examples / data

| From | To |
|------|----|
| `scripts/configs/*.yaml` | `configs/` |
| `scripts/configs/deepspeed/` | `configs/deepspeed/` |
| `scripts/test_dpo_setup.py` | `tests/test_dpo_setup.py` |
| `scripts/test_qwen2_5.py` | `tests/test_qwen2_5.py` |
| `scripts/complete_workflow_example.py` | `examples/tokenizer_workflow.py` |
| `data/download_owt.sh`, `data/download_ts.sh` | `data/download/` |
| `data_sft/openai/gsm8k/` | `data/sft/gsm8k/` |

### 4.4 Deletions

- `scripts/train_sft_v0.py` (versioned cruft)
- `data_sft/preprocess.py` (broken `pdb.set_trace()` stub) + empty `data_sft/`
- `clean_llm/` (after all moves)
- AI-artifact docs: `docs/MERGE_COMPLETION_REPORT.md`, `docs/MERGE_FEATURES.md`,
  `docs/GPTQ_IMPLEMENTATION_SUMMARY.md`, `docs/RM_TRAINING_SUMMARY.md`,
  `docs/TOKENIZER_MERGE_SUMMARY.md`, `docs/guide_archive.md`
  (only after confirming any unique setup/usage info has been folded into kept docs)

Kept docs: `ARCHITECTURE.md`, `DEEPSPEED_SETUP.md`, `DEMO_USAGE.md`,
`DPO_QUICK_START.md`, `TOKENIZER_QUICKSTART.md`, plus the folded-in `TOKENIZER.md`,
`QUANTIZATION.md`, `RM_TRAINING.md`.

## 5. Import remap (per-module, not blind replace)

| Old | New |
|-----|-----|
| `clean_llm.models.*` | `scratch_cs336.core.models.*` |
| `clean_llm.tokenizer` / `.tokenizer.*` | `scratch_cs336.core.tokenizer` / `.tokenizer.*` |
| `clean_llm.generation` / `.generation.*` | `scratch_cs336.core.generation` / `.generation.*` |
| `clean_llm.quantize` / `.quantize.*` | `scratch_cs336.core.quantize` / `.quantize.*` |
| `clean_llm.train.pretrain` | `scratch_cs336.training.pretrain` |
| `clean_llm.train.sft` | `scratch_cs336.training.sft` |
| `clean_llm.train.adapters` | `scratch_cs336.training.adapters` |
| `clean_llm.train.rm_train` | `scratch_cs336.training.rm` |
| `clean_llm.train.dpo_train` | `scratch_cs336.training.dpo` |
| `clean_llm.train.rlhf_datasets` | `scratch_cs336.training.datasets` |
| `clean_llm.data.processors.*` | `scratch_cs336.data.processors.*` |
| `clean_llm.eval.eval_pretrain` | `scratch_cs336.eval.pretrain` |
| `clean_llm.demo.*` | `scratch_cs336.serve.*` |
| `clean_llm.utils` | `scratch_cs336.utils` |

Known intra-package couplings to fix:
- `serve/chat.py`, `serve/web_ui.py`: `from clean_llm.generation import ...`
  → `from scratch_cs336.core.generation import ...` (public `__init__` re-exports preserved).
- `training/rm.py`: `from clean_llm.train.rlhf_datasets import RMDataset`
  → `from scratch_cs336.training.datasets import RMDataset`.
- `examples/quantize_example.py`: `from clean_llm.quantize import ...`
  → `from scratch_cs336.core.quantize import ...`.

Non-import string references:
- `scripts/launch_demo.py`: path strings `clean_llm/demo/web_ui.py`, `clean_llm/demo/chat.py`
  → `scratch_cs336/serve/web_ui.py`, `scratch_cs336/serve/chat.py`.
- `README.md`: project-structure block + `from clean_llm.generation.utils import ...` example
  + `streamlit run clean_llm/demo/web_ui.py` → `scratch_cs336/serve/web_ui.py`.

## 6. Drop Hydra — entry-point pattern

Every `scripts/*.py` replaces `@hydra.main(...)` with:

```python
import argparse
from omegaconf import OmegaConf

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/<name>.yaml")
    args, overrides = p.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:  # optional dotlist overrides, e.g. training.batch_size=8
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    # ... existing body unchanged (cfg.attr access still works) ...

if __name__ == "__main__":
    main()
```

Rationale: `OmegaConf.load` preserves the existing `cfg.attr` access and in-place mutation
(`model_config.vocab_size = ...`), so module bodies stay nearly identical. No Hydra app
framework: no `config_path`/`config_name`, no cwd switching, no auto output dirs.
`scratch_cs336/utils.py:log_params_from_omegaconf_dict` keeps working (OmegaConf, not Hydra).

### 6.1 Config edits (remove Hydra interpolations)

`configs/*.yaml` use `${hydra:...}` interpolations that must be replaced (plain `${var}`
interpolations are kept — `OmegaConf.load` resolves them on access):

- `root_dir: ${hydra:runtime.cwd}` → `root_dir: .` (scripts run from repo root; no cwd switch)
- `${hydra:run.dir}/...` (in `rm_training.yaml`, `sft_gsm8k.yaml`, `grpo_gsm8k.yaml`)
  → `outputs/...` (e.g. `output_dir: outputs/reward_model/`, `checkpoint_dir: outputs/.../checkpoint/`).
  `outputs/` is already gitignored.
- Remove dead `# hydra:` comment headers.

Affected files: `pretrain_cs336_lm.yaml`, `pretrain_qwen2_5.yaml`, `evaluate_cs336_lm.yaml`,
`evaluate_qwen2_5.yaml`, `tokenizer.yaml`, `rm_training.yaml`, `sft_gsm8k.yaml`, `grpo_gsm8k.yaml`.

## 7. pyproject.toml

- `name = "scratch_cs336"` (was `clean-llm`).
- Remove `hydra-core>=1.3.2`.
- Add `omegaconf` (was transitive via hydra) and `pyyaml` if needed.
- Add build backend + package discovery so `pip install -e .` resolves the package:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["scratch_cs336"]
```

## 8. .gitignore updates

- Replace `data/owt/`, `data/TinyStories/` with `data/pretrain/` (new pretrain data home).
- Keep `data/sft/gsm8k/` **committed** (do not ignore).
- Leave unrelated nanoGPT-style entries as-is unless they conflict.

## 9. data/README.md (new)

Short doc stating:
- `data/download/` — scripts to fetch raw pretrain corpora.
- `data/pretrain/` — generated/downloaded tokenized data (gitignored).
- `data/sft/gsm8k/` — committed sample SFT data (parquet).
- Processing code lives in `scratch_cs336/data/`.

## 10. Verification

Cheap checks (must pass):
1. `rg -n "clean_llm" --glob '!docs/superpowers/**'` → no hits in code/scripts/configs/README.
2. `rg -n "hydra" --glob '!docs/superpowers/**'` → no hits in code/scripts/configs/pyproject.
3. Smoke-import every submodule:
   `uv run python -c "import scratch_cs336, scratch_cs336.core.models.qwen2_5, scratch_cs336.core.models.cs336_lm, scratch_cs336.core.tokenizer, scratch_cs336.core.generation, scratch_cs336.core.quantize, scratch_cs336.training.pretrain, scratch_cs336.training.sft, scratch_cs336.training.rm, scratch_cs336.training.dpo, scratch_cs336.training.datasets, scratch_cs336.data.processors, scratch_cs336.eval.pretrain, scratch_cs336.serve.chat, scratch_cs336.utils"`
4. `uv run python -c "from omegaconf import OmegaConf; OmegaConf.load('configs/pretrain_cs336_lm.yaml')"` resolves.
5. `pipeline.sh` references (`scripts.tokenize`, `scripts.pretrain`) still resolve as modules.

Heavy checks (document if skipped — require data/model/network/GPU):
- Full `pipeline.sh` run, actual training, vllm-dependent imports (`serve`, `quantize` may
  import heavy deps; if an import fails purely due to optional heavy deps not installed,
  note it rather than treating as a reorg regression).

## 11. Risks & mitigations

- **Heavy optional deps** (vllm, autogptq, streamlit) may make some smoke-imports fail for
  reasons unrelated to the reorg. Mitigation: distinguish "module path resolves" from
  "third-party import error"; verify path correctness via `python -c "import importlib,ast"`
  or by importing leaf modules that lack heavy deps first.
- **OmegaConf interpolation differences** vs Hydra (`${hydra:...}`) — handled in §6.1.
- **`git mv` of a tracked dir into a new nested layout** — do moves in dependency-safe order;
  fix imports immediately after moving so verification can run.
- **Docs deletion losing info** — fold unique content before deleting (§4.4).

## 12. Out of scope

- Implementing missing GRPO logic (`train_grpo.py` reward_fn is a stub) — keep as-is.
- Rewriting model/training internals beyond import/path/config changes.
- Restructuring `reference/` (vendored CS336 assignments, gitignored).
