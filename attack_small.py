import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
import re
import sys
import os

# Add project root to path if needed
sys.path.append(os.getcwd())

from constants.model_constants import MODEL_CLASS_MAP
from utils.io_utils import prepare_input, derive_step_rewards

def calculate_reward(model_path, problem, steps):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    model = MODEL_CLASS_MAP[model_path].from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16)
    model = model.eval()
    input_ids, token_masks = prepare_input(
                            model_path, 
                            problem=problem, 
                            steps=steps, 
                            tokenizer=tokenizer
                        )
    with torch.inference_mode():
        logits = model(input_ids).logits

    rewards = derive_step_rewards(
        model_path, 
        logits, 
        token_masks, 
        tokenizer
    )
    return rewards

def extract_boxed_answer(text):
    """Extract the answer from \\boxed{} notation"""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None

def normalize_answer(answer):
    """Normalize answer for comparison"""
    if answer is None:
        return None
    # Remove spaces and convert to lowercase
    answer = answer.replace(' ', '').lower()
    # Remove common formatting
    answer = answer.replace(',', '')
    return answer

def main():
    # Constants
    SKYWORK_MODEL_PATH = "/home/rishabhtiwari/hf_cache/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-1.5B"
    
    # Load Dataset
    print("Loading dataset...")
    aime_2025_dataset = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
    aime_2025_data = aime_2025_dataset[:]
    
    # Example Data Setup
    data = {
        "problem": "On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$.",
        "steps": [
            "To determine the total weight of all Cindy's books, we need to calculate the weight of each book individually and then sum these weights.", 
            "First, for the math and science books:\\n- Each math book weighs 2 pounds.\\n- Each science book weighs 2 pounds.\\n- Cindy has 2 math books and 2 science books.\\n- Total weight of math books: \\(2 \\text{ books} \\times 2 \\text{ pounds/book} = 4 \\text{ pounds}\\).\\n- Total weight of science books: \\(2 \\text{ books} \\times 2 \\text{ pounds/book} = 4 \\text{ pounds}\\).\\n- Combined weight of math and science books: \\(4 \\text{ pounds} + 4 \\text{ pounds} = 8 \\text{ pounds}\\).", 
            "Second, for the French book:\\n- The French book weighs 4 pounds.",
            "Third, for the English book:\\n- The English book weighs 3 pounds.", 
            "Fourth, for the history book:\\n- The history book weighs twice as much as the English book.\\n- Weight of the history book: \\(2 \\times 3 \\text{ pounds} = 6 \\text{ pounds}\\).",
            "Finally, for the total weight:\\n- Sum of the weights of all the books: \\[ 8 \\text{ pounds} \\text{ (math and science)} + 4 \\text{ pounds} \\text{ (French)} + 3 \\text{ pounds} \\text{ (English)} + 6 \\text{ pounds} \\text{ (history)} = 21 \\text{ pounds} \\]",
            "Therefore, the total weight of the books Cindy is carrying is \\(\\boxed{21}\\) pounds."
        ],
        "answer": "588"
    }

    # Skywork Model Evaluation
    print("Evaluating with Skywork model...")
    tokenizer = AutoTokenizer.from_pretrained(SKYWORK_MODEL_PATH)
    
    # Note: Using cuda:0 for Skywork as in notebook
    model = MODEL_CLASS_MAP[SKYWORK_MODEL_PATH].from_pretrained(
        SKYWORK_MODEL_PATH, 
        torch_dtype=torch.bfloat16,
        device_map={"":"cuda:0"}, 
        low_cpu_mem_usage=False 
    )
    model = model.eval()
    
    input_ids, token_masks = prepare_input(
                            SKYWORK_MODEL_PATH, 
                            problem=data["problem"], 
                            steps=data["steps"], 
                            tokenizer=tokenizer,
                            device=torch.device("cuda:0")
                        )

    with torch.inference_mode():
        logits = model(input_ids.view(1, -1))[-1]

    rewards = derive_step_rewards(
        SKYWORK_MODEL_PATH, 
        logits, 
        token_masks, 
        tokenizer
    )
    print(f"Rewards: {rewards}")

    # Clean up Skywork model to free memory if needed, though we have 2 GPUs
    del model
    torch.cuda.empty_cache()

    # Generation with vLLM
    print("Generating answers with vLLM...")
    # Note: Using cuda:1 for vLLM as in notebook
    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        device="cuda:1"
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        n=100  # Generate 100 answers
    )

    prompt = f"""Problem: {data['problem']}

Please solve this problem step by step. Separate each step with a blank line (\\n\\n). Provide your final answer in \\boxed{{}}.

Solution:"""

    outputs = llm.generate([prompt], sampling_params)
    generated_answers = [output.text for output in outputs[0].outputs]

    print(f"Generated {len(generated_answers)} answers")
    
    # Evaluation
    ground_truth = data['answer']
    ground_truth_normalized = normalize_answer(ground_truth)
    
    print(f"Ground truth answer: {ground_truth}")
    
    predicted_answers = []
    correctness = []
    
    for i, answer in enumerate(generated_answers):
        predicted = extract_boxed_answer(answer)
        predicted_normalized = normalize_answer(predicted)
        
        is_correct = 1 if predicted_normalized == ground_truth_normalized else 0
        
        predicted_answers.append(predicted)
        correctness.append(is_correct)
        
        # Print first few for verification
        if i < 5:
            print(f"Answer {i+1}: {predicted} -> {'Correct' if is_correct else 'Incorrect'}")
            
    print(f"\nCorrectness list: {correctness}")
    accuracy = sum(correctness)/len(correctness) if correctness else 0
    print(f"Accuracy: {sum(correctness)}/{len(correctness)} = {accuracy:.2%}")

if __name__ == "__main__":
    main()
