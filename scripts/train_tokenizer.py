import os
import pickle

from omegaconf import DictConfig
# from scratch_cs336.core.tokenizer.train import run_train_bpe       # slow version
from scratch_cs336.core.tokenizer.train_fast import run_train_bpe    # fast version
from scratch_cs336.utils import load_config


def main(cfg: DictConfig):
    vocab, merges = run_train_bpe(
        input_path=cfg.input_path,
        vocab_size=cfg.vocab_size,
        special_tokens=cfg.special_tokens,
        num_chunks=cfg.num_chunks,
        num_processes=cfg.num_processes
    )

    os.makedirs(cfg.tokenizer_dir, exist_ok=True)
    with open(cfg.vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(cfg.merges_path, "wb") as f:
        pickle.dump(merges, f)

    # 统计最长token
    longest_token = max(vocab.values(), key=len)
    print("最长token:", longest_token, "长度:", len(longest_token))


if __name__ == "__main__":
    main(load_config("configs/tokenizer.yaml"))
