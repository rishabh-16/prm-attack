import re
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.prompt_utils import get_augmentation_prompt, get_equivalence_prompt
from utils.io_utils import prepare_input, prepare_batch_input_for_model, derive_step_rewards_vllm

def augmentor(df, task_text, client, model):
    prompts = []
    tokenizer = AutoTokenizer.from_pretrained(model)

    for index_row in df.iterrows():
        _, row = index_row
        question, steps = row["problem"], row["steps"]
        prompt          = get_augmentation_prompt(question, steps, task_text)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    # call OpenAI
    responses = []
    batch_size = 256
    for i in tqdm(range(0, len(prompts), batch_size), desc="Augmenting"):
        batch = prompts[i:i + batch_size]
        resp = client.completions.create(
            model=model,
            prompt=batch,
            max_tokens=4000,
            temperature=0.9,
        ).choices
        responses.extend(resp)
    resp = responses
    contents = sorted(resp, key=lambda x: int(x.index))
    contents = [content.text for content in contents]
    list_aug_questions = []
    list_aug_steps = []

    for content in contents:

        m = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        if not m:
            list_aug_questions.append("")
            list_aug_steps.append([])
            continue

        body = m.group(1).strip()

        # extract question
        q_m = re.search(r"<question>(.*?)</question>", body, re.DOTALL)
        aug_question = q_m.group(1).strip() if q_m else ""
        list_aug_questions.append(aug_question)

        # extract steps: find all <step#>…</step#>
        aug_steps = re.findall(r"<step\d+>(.*?)</step\d+>", body, re.DOTALL)
        aug_steps = [s.strip() for s in aug_steps]
        list_aug_steps.append(aug_steps)

    return {"aug_problem": list_aug_questions, "aug_steps": list_aug_steps}

def equivalence_check(df_org, df_aug, client, model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    qAs, stepsAs = df_org["problem"], df_org["steps"]
    qBs, stepsBs = df_aug["aug_problem"], df_aug["aug_steps"]

    prompts = []

    for qA, stepsA, qB, stepsB in zip(qAs, stepsAs, qBs, stepsBs):
        prompt = get_equivalence_prompt(qA, stepsA, qB, stepsB)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    responses = []
    batch_size = 256
    for i in tqdm(range(0, len(prompts), batch_size), desc="Checking equivalence"):
        batch = prompts[i:i + batch_size]
        resp = client.completions.create(
            model=model,
            prompt=batch,
            max_tokens=4000,
        ).choices
        responses.extend(resp)
    resp = responses
    contents = sorted(resp, key=lambda x: int(x.index))
    contents = [content.text for content in contents]
    equivalence_results = []
    body_equivalence_results = []

    for content in contents:

        # extract the <response>…</response> block
        m = re.search(r"<response>(.*?)</response>", content, re.DOTALL)
        if not m:
            equivalence_results.append(False)
            body_equivalence_results.append(content)
            continue

        body = m.group(1)
        body_equivalence_results.append(body)
        # grab question flag
        q_m = re.search(r"<question>\s*([YN])\s*</question>", body)
        question_flag = q_m.group(1) if q_m else "N"

        # grab all step flags into a list
        step_flags = re.findall(r"<step\d+>\s*([YN])\s*</step\d+>", body)

        # final check: question + every step must be "Y"
        all_flags = [question_flag] + step_flags
        all_flags = all(f == "Y" for f in all_flags)
        equivalence_results.append(all_flags)
    return {"equivalence": equivalence_results, "body_equivalence_results": body_equivalence_results}

def prm_scorer(questions, steps, client, model, batch_size=32):
    num_samples = len(questions)
    tokenizer = AutoTokenizer.from_pretrained(model)

    input_ids_all = []
    token_mask_all = []
    all_rewards = []

    for (question, step) in tqdm(zip(questions, steps), total=len(questions), desc="[PRM] Preparing input"):
        input_ids, token_mask = prepare_input(
                                model, 
                                problem=question, 
                                steps=step, 
                                tokenizer=tokenizer,
                                convert_to_list=True
        )
        input_ids_all.append(input_ids)
        token_mask_all.append(token_mask)


    for start_idx in tqdm(range(0, num_samples, batch_size), desc="[PRM] Scoring"):
        end_idx = start_idx + batch_size
    
        batch_input_ids = input_ids_all[start_idx:end_idx]
        batch_token_masks = token_mask_all[start_idx:end_idx]
    
        batch_input_ids, batch_token_masks = prepare_batch_input_for_model(batch_input_ids, batch_token_masks, pad_token_id=0)
    
        batch_logits = client.embeddings.create(
            input=batch_input_ids.cpu().tolist(),
            model=model,
        )
    
        rewards = derive_step_rewards_vllm(
            model,
            batch_logits,
            batch_token_masks,
            tokenizer
        )
    
        all_rewards.extend(rewards)
    return all_rewards