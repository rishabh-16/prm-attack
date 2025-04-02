from model_utils import processors

Qwen2_5_Math_PRM_7B = "Qwen/Qwen2.5-Math-PRM-7B"
Math_Shepherd_Mistral_7B_PRM = "peiyi9979/math-shepherd-mistral-7b-prm"
Llama3_1_8B_PRM_Mistral_Data = "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
Skywork_o1_Open_PRM_Qwen_2_5_7B = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"

PREPARE_INPUT_MAP = {
    Qwen2_5_Math_PRM_7B: processors.qwen_math_prm.prepare_input,
    Math_Shepherd_Mistral_7B_PRM: processors.math_shepherd_prm.prepare_input,
    Llama3_1_8B_PRM_Mistral_Data: processors.rlhflow_math_prm.prepare_input,
    Skywork_o1_Open_PRM_Qwen_2_5_7B: processors.skywork_o1_open_prm.prepare_input,
}

DERIVE_STEP_REWARDS_MAP = {
    Qwen2_5_Math_PRM_7B: processors.qwen_math_prm.derive_step_rewards,
    Math_Shepherd_Mistral_7B_PRM: processors.math_shepherd_prm.derive_step_rewards,
    Llama3_1_8B_PRM_Mistral_Data: processors.rlhflow_math_prm.derive_step_rewards,
    Skywork_o1_Open_PRM_Qwen_2_5_7B: processors.skywork_o1_open_prm.derive_step_rewards,
}