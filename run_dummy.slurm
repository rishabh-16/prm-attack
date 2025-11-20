#!/bin/bash
#SBATCH --job-name=dummy_test
#SBATCH --output=logs/dummy_%j.out
#SBATCH --error=logs/dummy_%j.err
#SBATCH --time=0:10:00
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment
source ~/.bashrc
conda activate prm_bias

# Run the script
python dummy_test.py
