import torch.nn.functional as F

STEP_SEP_TOKEN = "\n"

def prepare_input(problem, steps, tokenizer):
    """
    This function prepares the input for the PRM model.
    It takes a problem and a list of steps, and returns the input ids and the reward flags.
    """
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + STEP_SEP_TOKEN)
    response_ids = []
    reward_flags = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(STEP_SEP_TOKEN)[-1]
    for _, step in enumerate(steps):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []
        step_ids += [step_token_id]
        step = step + STEP_SEP_TOKEN
        flag = [0] * len(step_ids)
        flag[-1] = 1
        response_ids.extend(step_ids)
        reward_flags.extend(flag)
    input_ids = prompt_ids + response_ids
    return input_ids, reward_flags

def derive_step_rewards(rewards, token_masks, tokenizer):
    """
    This function derives the step rewards for the PRM model.
    It takes the rewards, the token masks, and the tokenizer, and returns the step rewards.
    """
    probabilities = F.sigmoid(rewards)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, 1
    batch_size = probabilities.size(0)

    all_scores_res = []
    for i in range(batch_size):
        sample = probabilities[i] # seq_len, 1
        positive_probs = sample[sample != 0].view(-1, 1)
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res
