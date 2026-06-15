"""GSM8K dataset loading helpers.

Replaces the old ``data_sft/preprocess.py`` debug stub. Loads the committed
parquet under ``data/sft/gsm8k/`` and falls back to the HuggingFace hub.
"""

from pathlib import Path

from datasets import load_dataset

# repo_root/scratch_cs336/data/preprocess.py -> parents[2] == repo root
GSM8K_DIR = Path(__file__).resolve().parents[2] / "data" / "sft" / "gsm8k"


def load_gsm8k(split: str = "main"):
    """Load GSM8K (``"main"`` or ``"socratic"``).

    Prefers the committed local parquet; falls back to ``openai/gsm8k`` on the hub.
    Returns a ``datasets.DatasetDict`` with ``train`` / ``test`` splits.
    """
    subset_dir = GSM8K_DIR / split
    train_pq = subset_dir / "train-00000-of-00001.parquet"
    test_pq = subset_dir / "test-00000-of-00001.parquet"
    if train_pq.exists() and test_pq.exists():
        return load_dataset(
            "parquet",
            data_files={"train": str(train_pq), "test": str(test_pq)},
        )
    return load_dataset("openai/gsm8k", split)


if __name__ == "__main__":
    ds = load_gsm8k("main")
    for name, split in ds.items():
        print(f"{name}: {len(split)} examples")
    print(ds["train"][0])
