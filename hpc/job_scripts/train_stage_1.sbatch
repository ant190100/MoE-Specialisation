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

# Load only CUDA, not Python (let conda handle Python)
module load CUDA/11.7.0
module load OpenSSL/1.1
module load GCCcore/11.3.0


# Initialize conda
source $HOME/.bashrc
conda activate moe_project

# Force the conda environment Python to be used
export PATH="/home/aticinovic/.conda/envs/moe_project/bin:$PATH"

echo "=== VERIFICATION ==="
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
echo "Python version: $(python --version)"

# Test packages
python -c "import yaml; print('✅ YAML available')"
python -c "import torch; print(f'✅ PyTorch {torch.__version__} available, CUDA: {torch.cuda.is_available()}')"

cd $HOME/MoE-Specialisation
export PYTHONPATH="${PWD}:${PYTHONPATH}"

echo "Starting VLM Stage 1 training script..."
srun python training_scripts/train_stage_1.py

echo "Job finished."