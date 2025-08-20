#!/bin/bash
#SBATCH --job-name=debug_test
#SBATCH -p gpu-a100-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=../out_slurm/debug_%j.out
#SBATCH --error=../err_slurm/debug_%j.err

echo "=== Debug Test Starting ==="
echo "Current directory: $(pwd)"
echo "Home directory: $HOME"

module purge
module load Python/3.10.4-GCCcore-11.3.0
module load CUDA/11.7.0
echo "Modules loaded successfully"

# Test conda activation
source ~/.bashrc
echo "Bashrc sourced"

conda info --envs
echo "Available conda environments:"

conda activate moe_project
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# Test directory change
cd ~/MoE-Specialisation
echo "Changed to directory: $(pwd)"
ls -la
echo "Directory contents listed"

# Test Python imports
python -c "import sys; print('Python path:'); [print(p) for p in sys.path]"
python -c "import torch; print('PyTorch imported successfully')"
python -c "from models.vl_connector import VisionLanguageConnector; print('VisionLanguageConnector imported successfully')"

echo "=== Debug Test Complete ==="