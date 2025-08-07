"""
Simple script to test the toy model locally.
"""

import torch
from transformers import AutoTokenizer
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.tiny_moe import TinyMoEForClassification
from src.data.loaders import create_data_loaders
from src.analysis.ablation import run_ablation_analysis
from src.analysis.visualization import (
    plot_ablation_heatmap,
    run_full_visualization_analysis,
)
from src.utils.config import load_config


def main():
    """Run a quick test of the toy model."""
    print("Loading configuration...")
    config = load_config("configs/toy_model_config.yaml")

    # Override for quick testing
    config["NUM_EPOCHS"] = 1
    config["BATCH_SIZE"] = 16
    config["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {config['DEVICE']}")

    # Load tokenizer and data
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained(config["TOKENIZER_NAME"])
    train_loader, test_loader, class_names = create_data_loaders(
        dataset_name=config["DATASET"],
        batch_size=config["BATCH_SIZE"],
        tokenizer=tokenizer,
    )

    # Initialize model
    print("Initializing model...")
    model = TinyMoEForClassification(
        vocab_size=len(tokenizer),
        embed_dim=config["EMBED_DIM"],
        hidden_dim=config["HIDDEN_DIM"],
        num_experts=config["NUM_EXPERTS"],
        top_k=config["TOP_K"],
        num_classes=config["NUM_CLASSES"],
    ).to(config["DEVICE"])

    # Quick training
    print("Training model (1 epoch)...")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"])

    model.train()
    total_loss = 0
    num_batches = min(50, len(train_loader))  # Limit to 50 batches for quick test

    for i, batch in enumerate(train_loader):
        if i >= num_batches:
            break

        input_ids = batch["input_ids"].to(config["DEVICE"])
        attention_mask = batch["attention_mask"].to(config["DEVICE"])
        labels = batch["labels"].to(config["DEVICE"])

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if i % 10 == 0:
            print(f"Batch {i}/{num_batches}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    print(f"Average training loss: {avg_loss:.4f}")

    # Run analysis
    print("\nRunning ablation analysis...")
    ablation_results = run_ablation_analysis(
        model,
        test_loader,
        config["DEVICE"],
        config["NUM_CLASSES"],
        config["NUM_EXPERTS"],
        class_names,
    )

    # Visualizations
    print("\nGenerating visualizations...")
    plot_ablation_heatmap(ablation_results)

    run_full_visualization_analysis(
        model, test_loader, config["DEVICE"], config, tokenizer
    )

    print("\nTest completed successfully!")


if __name__ == "__main__":
    main()
