"""Entry point used by launcher tests: every rank records what it saw into ``output_dir``."""

import json
from dataclasses import dataclass
from pathlib import Path

import draccus

from scratch_cs336.distributed import destroy_distributed, init_distributed


@dataclass
class DummyConfig:
    output_dir: str = ""
    data: str = ""  # path of an upstream artifact, read by every rank
    fail: bool = False


def main() -> None:
    cfg = draccus.parse(config_class=DummyConfig)
    env = init_distributed()
    if cfg.fail:
        raise SystemExit(3)
    payload = {
        "rank": env.rank,
        "world_size": env.world_size,
        "local_rank": env.local_rank,
        "data": Path(cfg.data).read_text() if cfg.data else None,
    }
    Path(cfg.output_dir, f"rank_{env.rank}.json").write_text(json.dumps(payload))
    destroy_distributed()


if __name__ == "__main__":
    main()
