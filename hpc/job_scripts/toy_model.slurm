#!/bin/bash

#SBATCH --job-name=toy_model_analysis
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=~/out_slurm/toy_model/toy_model_%j.out
#SBATCH --error=~/err_slurm/toy_model/toy_model_%j.err

echo "=========================================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "=========================================================="

# Load modules
module purge
module load python/3.11.3
module load cuda/11.7.1

# Activate virtual environment (adjust path as needed)
source ~/MoE-Specialisation/moe_env/bin/activate

# Change to project directory
cd ~/MoE-Specialisation

# Run toy model experiment
echo "Running toy model training and analysis..."
python -m hpc.experiments.toy_model_experiment

echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="
