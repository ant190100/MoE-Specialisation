"""
Toy model training and analysis experiment for local execution.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.tiny_moe import TinyMoEForClassification
from src.data.loaders import create_data_loaders
from src.analysis.ablation import run_ablation_analysis
from src.utils.config import load_config, setup_experiment_paths
import logging
from datetime import datetime
from pathlib import Path


class ToyModelExperiment:
    """Local experiment for training and analyzing toy MoE model."""

    def __init__(self, config_path):
        self.config = load_config(config_path)

        # Auto-detect device (CUDA, MPS for Apple Silicon, or CPU)
        if torch.cuda.is_available():
            self.config["DEVICE"] = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.config["DEVICE"] = "mps"  # Apple Silicon GPU
        else:
            self.config["DEVICE"] = "cpu"

        self.experiment_name = "toy_model_analysis"
        self.setup_paths()
        self.setup_logging()

    def setup_paths(self):
        """Setup experiment directory structure."""
        self.paths = setup_experiment_paths(self.experiment_name, "./results")

    def setup_logging(self):
        """Setup logging for the experiment."""
        log_file = (
            self.paths["logs_dir"]
            / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Using device: {self.config['DEVICE']}")

    def setup_model_and_data(self):
        """Initialize model, tokenizer, and data loaders."""
        self.logger.info("Setting up model and data...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["TOKENIZER_NAME"])

        # Create data loaders
        self.train_loader, self.test_loader, self.class_names = create_data_loaders(
            dataset_name=self.config["DATASET"],
            batch_size=self.config["BATCH_SIZE"],
            tokenizer=self.tokenizer,
        )

        # Initialize model
        self.model = TinyMoEForClassification(
            vocab_size=len(self.tokenizer),
            embed_dim=self.config["EMBED_DIM"],
            hidden_dim=self.config["HIDDEN_DIM"],
            num_experts=self.config["NUM_EXPERTS"],
            top_k=self.config["TOP_K"],
            num_classes=self.config["NUM_CLASSES"],
        ).to(self.config["DEVICE"])

        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config["LEARNING_RATE"]
        )

    def train_model(self):
        """Train the toy MoE model."""
        self.logger.info("Starting model training...")

        for epoch in range(self.config["NUM_EPOCHS"]):
            self.model.train()
            total_loss = 0

            for batch in tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{self.config['NUM_EPOCHS']}"
            ):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.config["DEVICE"])
                attention_mask = batch["attention_mask"].to(self.config["DEVICE"])
                labels = batch["labels"].to(self.config["DEVICE"])

                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch+1} | Average Training Loss: {avg_loss:.4f}")

        # Save trained model
        model_path = self.paths["models_dir"] / "trained_model.pt"
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    def run_analysis(self):
        """Run comprehensive analysis on the trained model."""
        self.logger.info("Running comprehensive analysis...")

        results = {}

        # 1. Ablation analysis
        self.logger.info("Running ablation analysis...")
        results["ablation_results"] = run_ablation_analysis(
            self.model,
            self.test_loader,
            self.config["DEVICE"],
            self.config["NUM_CLASSES"],
            self.config["NUM_EXPERTS"],
            self.class_names,
        )

        # Add metadata
        results["config"] = self.config
        results["class_names"] = self.class_names

        self.logger.info("Analysis complete!")
        return results

    def save_results(self, results, filename):
        """Save experiment results."""
        output_path = self.paths["analysis_dir"] / filename
        torch.save(results, output_path)
        self.logger.info(f"Results saved to {output_path}")

    def run(self):
        """Run the complete toy model experiment."""
        self.logger.info(f"Starting experiment: {self.experiment_name}")

        self.setup_model_and_data()
        self.train_model()

        # Run analysis
        results = self.run_analysis()

        # Save complete results
        self.save_results(results, "analysis_results.pt")

        # Save individual analysis components
        if "ablation_results" in results:
            results["ablation_results"].to_csv(
                self.paths["analysis_dir"] / "ablation_results.csv"
            )

        self.logger.info("Experiment complete!")
        return results


if __name__ == "__main__":
    config_path = "configs/toy_model_config.yaml"

    experiment = ToyModelExperiment(config_path)
    results = experiment.run()
