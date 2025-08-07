"""
Quick test script to verify the repository structure works.
"""

import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.tiny_moe import TinyMoEForClassification
from src.utils.config import load_config


def test_model_creation():
    """Test basic model creation."""
    print("Testing model creation...")

    # Load config
    config = load_config("configs/toy_model_config.yaml")

    # Create a simple model
    model = TinyMoEForClassification(
        vocab_size=1000,  # Small vocab for testing
        embed_dim=config["EMBED_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        num_experts=config["NUM_EXPERTS"],
        top_k=config["TOP_K"],
        num_classes=config["NUM_CLASSES"],
    )

    # Test forward pass
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    with torch.no_grad():
        outputs = model(input_ids)

    print(f"Model created successfully!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Number of experts: {config['NUM_EXPERTS']}")
    print(f"Top-k routing: {config['TOP_K']}")

    return True


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")

    config = load_config("configs/toy_model_config.yaml")

    print("Configuration loaded successfully:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    return True


def main():
    """Run all tests."""
    print("🧪 Testing MoE Specialisation Repository Structure\n")

    try:
        test_config_loading()
        test_model_creation()

        print("\n✅ All tests passed! Repository structure is working correctly.")
        print("\n📝 Next steps:")
        print("  1. Install datasets: pip install datasets")
        print("  2. Run full local test: python analysis_scripts/test_local.py")
        print("  3. Configure HPC paths in hpc/configs/")
        print("  4. Submit HPC jobs: sbatch hpc/job_scripts/toy_model.slurm")
        print("  5. Analyze results: python analysis_scripts/local_analysis.py")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    main()
