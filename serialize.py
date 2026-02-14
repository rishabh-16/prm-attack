# Takes around 10 mins to run

from transformers import AutoModelForCausalLM
import torch

from constants.model_constants import MODEL_CLASS_MAP

path = "/p/vast1/tomar1/.cache/huggingface/checkpoints"
Skywork_o1_Open_PRM_Qwen_2_5_7B = f"{path}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-7B"
Skywork_o1_Open_PRM_Qwen_2_5_1_5B = f"{path}/Skywork--Skywork-o1-Open-PRM-Qwen-2.5-1.5B"

for model_path in [Skywork_o1_Open_PRM_Qwen_2_5_7B, Skywork_o1_Open_PRM_Qwen_2_5_1_5B]:
    # Load the model
    print("Loading model...")
    model = MODEL_CLASS_MAP[model_path].from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu"  # Keep on CPU for conversion
    )

    # Save with safetensors format
    print("Saving in safetensors format...")
    model.save_pretrained(
        model_path,
        safe_serialization=True  # This enables safetensors format
    )

    print("Conversion complete!")