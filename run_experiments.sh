#!/bin/bash
# PRM Attack Experiments
# Duplicate and modify these commands as needed
#
# Create logs directory first: mkdir -p logs
#
# Key arguments:
#   --prm_model: 1.5B, 7B (Skywork PRMs), or Qwen-7B (Qwen Math PRM)
#   --experiment: single or batched
#   --continuous: add this flag for continuous optimization (default is discrete)
#   --num_adv_tokens: number of adversarial tokens (default: 1)
#   --adv_position: position of adv tokens - "end" (after solution) or "middle" (after question) (default: end)
#   --num_iterations: optimization iterations (default: 1000)
#   --num_train_trajectories: limit training trajectories (default: all)
#   --num_eval_trajectories: limit eval trajectories (default: all)
#   --run_transfer: run transfer evaluation on AIME2025
#   --run_plots: generate training curve plots
#   --run_3d_landscape: generate 3D reward landscape
#   --landscape_grid_size: grid resolution for landscape (default: 25)
#   --force_rerun: force rerun even if cache exists
#
# PRM Types:
#   - Skywork (1.5B, 7B): Optimizes final step reward
#   - Qwen (Qwen-7B): Optimizes minimum step reward (locates first incorrect step)

mkdir -p logs
# ============================================
# 1.5B Model - Batched - Continuous (1 token)
# ============================================
python prm_attack_experiments.py \
    --prm_model 1.5B \
    --experiment batched \
    --num_adv_tokens 1 \
    --adv_position end \
    --continuous \
    --num_iterations 1000 \
    --num_train_trajectories 8 \
    --num_eval_trajectories 8 \
    --run_transfer \
    --run_plots \
    --run_3d_landscape \
    --batch_chunk_size 2 \
    --distributed \
    2>&1 | tee logs/batched_1.5B_continuous_1tok_end.log

# # ============================================
# # 1.5B Model - Batched - Discrete (1 token)
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model 1.5B \
#     --experiment batched \
#     --num_adv_tokens 1 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 2 \
#     --distributed \
#     2>&1 | tee logs/batched_1.5B_discrete_1tok_end.log

# # ============================================
# # 1.5B Model - Batched - Discrete (50 tokens)
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model 1.5B \
#     --experiment batched \
#     --num_adv_tokens 50 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 2 \
#     --distributed \
#     2>&1 | tee logs/batched_1.5B_discrete_50tok_end.log

# # ============================================
# # 1.5B Model - Batched - Discrete (100 tokens)
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model 1.5B \
#     --experiment batched \
#     --num_adv_tokens 100 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 2 \
#     --distributed \
#     2>&1 | tee logs/batched_1.5B_discrete_100tok_end.log

# ============================================
# 1.5B Model - Batched - Discrete (500 tokens)
# ============================================
# python prm_attack_experiments.py \
#     --prm_model 1.5B \
#     --experiment batched \
#     --num_adv_tokens 500 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 2 \
#     --distributed \
#     2>&1 | tee logs/batched_1.5B_discrete_500tok_end.log


# # ============================================
# # 7B Model - Batched - Continuous (1 token)
# # ============================================
# # python prm_attack_experiments.py \
# #     --prm_model 7B \
# #     --experiment batched \
# #     --num_adv_tokens 1 \
# #     --adv_position end \
# #     --continuous \
# #     --num_iterations 1000 \
# #     --num_train_trajectories 8 \
# #     --num_eval_trajectories 8 \
# #     --run_transfer \
# #     --run_plots \
# #     --run_3d_landscape \
# #     --batch_chunk_size 1 \
# #     --distributed \
# #     2>&1 | tee logs/batched_7B_continuous_1tok_end.log


# # ============================================
# # 7B Model - Batched - Discrete (1 token)
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model 7B \
#     --experiment batched \
#     --num_adv_tokens 1 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/batched_7B_discrete_1tok_end.log

# # ============================================
# # 7B Model - Batched - Discrete (50 tokens)
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model 7B \
#     --experiment batched \
#     --num_adv_tokens 50 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/batched_7B_discrete_50tok_end.log

# # ============================================
# # 7B Model - Batched - Discrete (100 tokens)
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model 7B \
#     --experiment batched \
#     --num_adv_tokens 100 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/batched_7B_discrete_100tok_end.log

# ============================================
# 7B Model - Batched - Discrete (500 tokens)
# ============================================
# python prm_attack_experiments.py \
#     --prm_model 7B \
#     --experiment batched \
#     --num_adv_tokens 500 \
#     --adv_position end \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/batched_7B_discrete_500tok_end.log


# ============================================
# ============================================
# QWEN MATH PRM EXPERIMENTS
# Qwen PRM locates first incorrect step - we optimize the minimum step reward
# ============================================
# ============================================

# # ============================================
# # Qwen-7B Model - Batched - Discrete (1 token) - Middle Position
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model Qwen-7B \
#     --experiment batched \
#     --num_adv_tokens 1 \
#     --adv_position middle \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_1tok_middle.log

# # ============================================
# # Qwen-7B Model - Batched - Discrete (50 tokens) - Middle Position
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model Qwen-7B \
#     --experiment batched \
#     --num_adv_tokens 50 \
#     --adv_position middle \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_50tok_middle.log

# # ============================================
# # Qwen-7B Model - Batched - Discrete (100 tokens) - Middle Position
# # ============================================
# python prm_attack_experiments.py \
#     --prm_model Qwen-7B \
#     --experiment batched \
#     --num_adv_tokens 100 \
#     --adv_position middle \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_100tok_middle.log

# ============================================
# Qwen-7B Model - Batched - Discrete (500 tokens) - Middle Position
# ============================================
# python prm_attack_experiments.py \
#     --prm_model Qwen-7B \
#     --experiment batched \
#     --num_adv_tokens 500 \
#     --adv_position middle \
#     --num_iterations 1000 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_500tok_middle.log

# ============================================
# Qwen-7B Model - Batched - Discrete (100 tokens) - Middle Position
# ============================================
# python -u prm_attack_experiments.py \
#     --prm_model Qwen-7B \
#     --experiment batched \
#     --num_adv_tokens 100 \
#     --adv_position middle \
#     --num_iterations 1000 \
#     --entropy_weight_start 0.00001 \
#     --entropy_weight_end 0.001 \
#     --num_train_trajectories 8 \
#     --num_eval_trajectories 8 \
#     --run_transfer \
#     --run_plots \
#     --run_3d_landscape \
#     --batch_chunk_size 1 \
#     --distributed \
#     2>&1 | tee logs/qwen_batched_Qwen-7B_discrete_100tok_middle.log