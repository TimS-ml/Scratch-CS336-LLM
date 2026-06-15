import argparse

import torch
import mlflow
from omegaconf import DictConfig, ListConfig, OmegaConf


def load_config(default_path):
    """Load a plain YAML config (no Hydra) and apply optional CLI dotlist overrides.

    Usage in an entry point::

        if __name__ == "__main__":
            main(load_config("configs/pretrain_cs336_lm.yaml"))

    Pass ``--config path/to.yaml`` to use a different file, and append
    ``key=value`` pairs (e.g. ``training.batch_size=8``) to override fields.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_path)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)

def _to_device_and_compile(model, device=None):
    if not device:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    model = model.to(device)
    return model, device
