#!/bin/bash

# --- Slurm Job Configuration ---
#SBATCH --job-name=tinymistral_analysis # Name of the job
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                      # Run on a single CPU core
#SBATCH --cpus-per-task=4               # Request 4 CPUs
#SBATCH --mem=32G                       # Request 32GB of memory
#SBATCH --time=00:30:00                 # Job time limit (1 hour)
#SBATCH --output=~/out_slurm/tinymistral/tinymistral_%j.out     # Standard output file
#SBATCH --error=~/err_slurm/tinymistral/tinymistral_%j.err      # Standard error file

# --- Job Execution ---
echo "=========================================================="
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "=========================================================="

# 1. Load necessary modules
module purge
module load python/3.11.3
module load cuda/11.7.1

# 2. Run your new Python analysis script
echo "Running TinyMistral analysis script..."
python analyze_tinymistral.py

echo "=========================================================="
echo "Job finished at: $(date)"
echo "=========================================================="
