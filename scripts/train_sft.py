import torch
from omegaconf import DictConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from scratch_cs336.training.sft import run_tokenize_prompt_and_output
from scratch_cs336.training.sft import run_get_response_log_probs
from scratch_cs336.training.sft import run_sft_microbatch_train_step
from scratch_cs336.utils import load_config


def main(cfg: DictConfig):
    model = AutoModelForCausalLM.from_pretrained(
    cfg.model_path,
    torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    dataset = load_dataset(cfg.dataset_path, 'main')
    
    train_prompt_strs = [example['question'] for example in dataset['train']]
    train_output_strs = [example['answer'] for example in dataset['train']]
    test_prompt_strs = [example['question'] for example in dataset['test']]
    test_output_strs = [example['answer'] for example in dataset['test']]
    
    print(f"Num of train examples = {len(train_prompt_strs)}")
    
    
    train_prompt_strs = train_prompt_strs[:3]
    train_output_strs = train_output_strs[:3]
    
    train_inputs = run_tokenize_prompt_and_output(train_prompt_strs, train_output_strs, tokenizer)
    test_inputs = run_tokenize_prompt_and_output(test_prompt_strs, test_output_strs, tokenizer)
    
    # import pdb; pdb.set_trace()
    
    res = run_get_response_log_probs(model, 
                               train_inputs['input_ids'],
                               train_inputs['labels'],
                               return_token_entropy=True)
    policy_log_probs = res['log_probs']
    token_entropy = res['token_entropy']
    
    loss, metadata = run_sft_microbatch_train_step(policy_log_probs,
                                                   train_inputs['response_mask'],
                                                   gradient_accumulation_steps=1,
                                                   normalize_constant=1)
    
    print('loss', loss)
    
    
    
if __name__ == '__main__':
    main(load_config("configs/sft_gsm8k.yaml"))