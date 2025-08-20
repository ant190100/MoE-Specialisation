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

# Load dependencies in correct order (CRITICAL!)
module load GCCcore/11.3.0
module load Python/3.10.4
module load CUDA/11.7.0

echo "Modules loaded successfully"
echo "Loaded modules: $(module list 2>&1)"

# Initialize conda
source $HOME/.bashrc
conda activate moe_project

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python version: $(python --version)"
echo "Python location: $(which python)"

# Verify PyTorch is available
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Change to project directory
cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Starting VLM Stage 1 training script..."
srun python training_scripts/train_stage_1.py

echo "Job finished."