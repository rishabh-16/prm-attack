import numpy as np
import torch.nn.functional as F

STEP_SEP_TOKEN = "+"
CANDIDATE_TOKENS = ["+", "-"]

def prepare_input(problem: str, steps: list[str], tokenizer):
    """
    This function prepares the input for the PRM model.
    It takes a problem and a list of steps, and returns the input ids and the token masks.
    """
    ## Generate input ids
    messages = [
        {"role": "user", "content": f"{problem} {steps[0]}"},
        {"role": "assistant", "content": STEP_SEP_TOKEN},
    ]
    for step in steps[1:]:
        messages.append({"role": "user", "content": f"{step}"})
        messages.append({"role": "assistant", "content": STEP_SEP_TOKEN})

    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    input_ids = tokenizer.encode(conversation_str)
    
    ## Calculate token masks for each response
    step_sep_id = tokenizer.encode(STEP_SEP_TOKEN, add_special_tokens=False)[0]
    
    token_masks = np.zeros(len(input_ids), dtype=bool)

    current_position = 0
    for message in messages:
        tokenized_input = tokenizer.apply_chat_template([message])
        if message["role"] == "assistant":
            for i, token_id in enumerate(tokenized_input):
                if token_id == step_sep_id:
                    token_masks[current_position + i] = True
        current_position += len(tokenized_input)

    return input_ids, token_masks

def derive_step_rewards(rewards, token_masks, tokenizer):
    """
    This function derives the step rewards for the PRM model.
    It takes the rewards, the token masks, and the tokenizer, and returns the step rewards.
    """
    candidate_tokens = tokenizer.encode(f"{CANDIDATE_TOKENS[0]} {CANDIDATE_TOKENS[1]}", add_special_tokens=False)

    probabilities = F.softmax(rewards[..., candidate_tokens], dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, 2
    batch_size = probabilities.size(0)

    all_scores_res = []
    for i in range(batch_size):
        sample = probabilities[i] # seq_len, 2
        positive_probs = sample[sample != 0].view(-1, 2)[:, 0]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res