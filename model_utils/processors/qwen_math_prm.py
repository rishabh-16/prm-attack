import torch
import numpy as np
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

STEP_SEP_TOKEN = "<extra_0>"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

def prepare_input(problem: str, steps: list[str], tokenizer: PreTrainedTokenizerBase):
    """
    This function prepares the input for the PRM model.
    It takes a problem and a list of steps, and returns the input ids and the token masks.
    """
    ## Generate input ids
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},  
        {"role": "user", "content": problem},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    input_ids = tokenizer.encode(conversation_str, return_tensors="pt")

    ## Calculate token masks
    step_sep_id = tokenizer.encode(STEP_SEP_TOKEN, add_special_tokens=False)[0]
    token_masks = (np.array(input_ids) == step_sep_id)

    token_masks = torch.from_numpy(token_masks)
    return input_ids, token_masks

def derive_step_rewards(logits: torch.Tensor, token_masks: torch.Tensor, tokenizer: PreTrainedTokenizerBase):
    """
    This function derives the step rewards for the PRM model.
    It takes the logits, the token masks, and the tokenizer, and returns the step rewards.
    """
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    batch_size = probabilities.size(0)
    
    all_scores_res = []
    for i in range(batch_size):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res