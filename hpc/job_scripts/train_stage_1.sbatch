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
module load CUDA/11.7.0

# Activate virtual environment
source ~/venv_moe_project/bin/activate

# Completely override system Python paths
unset PYTHONHOME
export PYTHONPATH="/home/aticinovic/venv_moe_project/lib/python3.10/site-packages"

echo "=== VERIFICATION ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import google.protobuf; print(f'Protobuf version: {google.protobuf.__version__}')"

cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:/home/aticinovic/venv_moe_project/lib/python3.10/site-packages"

echo "Starting training script..."
python training_scripts/train_stage_1.py

echo "Job finished."