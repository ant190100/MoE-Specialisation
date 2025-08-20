#!/bin/bash
#SBATCH --job-name=vlm_stage1_training
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=../out_slurm/train_stage_1/train_stage_1_%j.out
#SBATCH --error=../err_slurm/train_stage_1/train_stage_1_%j.err

# --- Environment Setup ---
module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0

# Initialize conda properly
source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe_project

# Change to project directory  
cd ~/MoE-Specialisation

# Add project to Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# --- Execute the Main Training Script ---
echo "Starting VLM Stage 1 training script..."
srun python training_scripts/train_stage_1.py

echo "Job finished."