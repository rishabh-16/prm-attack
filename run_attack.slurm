#!/bin/bash
#SBATCH --job-name=prm_attack
#SBATCH --output=logs/attack_%j.out
#SBATCH --error=logs/attack_%j.err
#SBATCH --time=4:00:00
#SBATCH --constraint=gpu
#SBATCH --account=m4790_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate prm_bias

# Run the script
python attack_small.py
