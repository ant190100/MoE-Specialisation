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

# Load modules
module purge
module load GCC/11.3.0
module load OpenMPI/4.1.4
module load CUDA/11.7.0
module load PyTorch/2.1.2-CUDA-11.7.0

# Activate virtual environment (NOT conda)
source ~/venv_moe_project/bin/activate

echo "=== VERIFICATION ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Rest of your script...
cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Starting training script..."
python training_scripts/train_stage_1.py

echo "Job finished."