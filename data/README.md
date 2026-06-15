# data/

Data lives here; processing **code** lives in `scratch_cs336/data/`.

| Path | Contents | Tracked in git? |
|------|----------|-----------------|
| `data/download/` | Shell scripts to fetch raw pretrain corpora (OpenWebText, TinyStories) | yes |
| `data/pretrain/` | Downloaded / tokenized pretrain data (`.dat`, raw text) | no (gitignored) |
| `data/sft/gsm8k/` | Committed GSM8K sample data (parquet) used by SFT/GRPO | yes |

## Usage

```bash
# download raw pretrain corpora
bash data/download/download_ts.sh      # TinyStories
bash data/download/download_owt.sh     # OpenWebText

# load GSM8K from the committed parquet
uv run python -m scratch_cs336.data.preprocess
```
