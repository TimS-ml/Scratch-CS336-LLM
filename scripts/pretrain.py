import mlflow
from omegaconf import DictConfig
import warnings
warnings.filterwarnings("ignore")

from scratch_cs336.core.models.qwen2_5 import Qwen2_5
from scratch_cs336.core.models.cs336_lm import BasicsTransformerLM
from scratch_cs336.training.pretrain import train
from scratch_cs336.core.tokenizer.tokenizer import get_custom_tokenizer
from scratch_cs336.utils import _to_device_and_compile, log_params_from_omegaconf_dict, load_config




def main(cfg: DictConfig):
    mlflow.set_experiment(cfg.exp_name)
    mlflow.start_run()

    if cfg.model_type == "qwen2_5":
        model_config, training_config, tokenizer_config = cfg.model, cfg.training, cfg.tokenizer
        tokenizer = get_custom_tokenizer(**tokenizer_config)
        model_config.vocab_size = tokenizer.vocab_size
        model_config.eos_token_id = tokenizer.eos_token_id
        model = Qwen2_5.from_config(model_config)
    elif cfg.model_type == "cs336_lm":
        model_config, training_config = cfg.model, cfg.training
        model = BasicsTransformerLM(**model_config)

    model, device = _to_device_and_compile(model)

    train(model, device, training_config)

    mlflow.end_run()

if __name__ == '__main__':
    main(load_config("configs/pretrain_cs336_lm.yaml"))
