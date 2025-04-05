import torch
from transformers import PreTrainedTokenizerBase
from constants.model_constants import PREPARE_INPUT_MAP, DERIVE_STEP_REWARDS_MAP, DERIVE_STEP_REWARDS_VLLM_MAP

def prepare_input(model_name: str, 
                  problem: str, 
                  steps: list[str], 
                  tokenizer: PreTrainedTokenizerBase,
                  device="cuda"):
    prepare_input_fn = PREPARE_INPUT_MAP[model_name]
    input_ids, token_masks = prepare_input_fn(problem, steps, tokenizer)
    return input_ids.to(device), token_masks.to(device)

def derive_step_rewards(model_name: str, logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    derive_step_rewards_fn = DERIVE_STEP_REWARDS_MAP[model_name]
    return derive_step_rewards_fn(logits, token_masks, tokenizer)

def derive_step_rewards_vllm(model_name, logits, token_masks, tokenizer):
    derive_step_rewards_fn = DERIVE_STEP_REWARDS_VLLM_MAP[model_name]
    return derive_step_rewards_fn(logits, token_masks, tokenizer)

def prepare_batch_input_for_model(input_ids, reward_flags, pad_token_id):
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(ids) for ids in input_ids], 
        batch_first=True,
        padding_value=pad_token_id
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor([1] * len(ids)) for ids in input_ids], 
        batch_first=True,
        padding_value=0
    )
    padded_reward_flags = torch.nn.utils.rnn.pad_sequence(
        [torch.LongTensor(reward_flag) for reward_flag in reward_flags], 
        batch_first=True,
        padding_value=0
    )
    return padded_input_ids, padded_attention_mask, padded_reward_flags