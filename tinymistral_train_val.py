"""
TinyMistral Training Pipeline
A script for training TinyMistral MoE model for classification.
"""

import torch
import torch.nn as nn
import yaml
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.tinymistral_classifier import (
    TinyMistralForClassification,
    load_tinymistral_tokenizer,
)
from src.data.loaders import create_data_loaders


class TinyMistralPipeline:
    """Training pipeline for TinyMistral classification."""

    def __init__(self, config_path="configs/tinymistral_training.yaml"):
        """Initialize the pipeline with configuration."""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.setup_paths()
        self.setup_logging()
        self.setup_device()

    def setup_device(self):
        """Setup the best available device."""
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.logger.info(f"Using device: {self.device}")

    def setup_paths(self):
        """Setup experiment directory structure."""
        self.results_dir = Path("results") / "tinymistral"
        for subdir in ["models", "logs", "analysis", "checkpoints"]:
            (self.results_dir / subdir).mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """Setup logging."""
        log_file = (
            self.results_dir
            / "logs"
            / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def setup_model_and_data(self):
        """Initialize model, tokenizer, and data loaders."""
        self.logger.info("Setting up TinyMistral model and data...")

        # Load tokenizer and data
        self.tokenizer = load_tinymistral_tokenizer(self.config["MODEL_NAME"])
        self.train_loader, self.val_loader, self.class_names = create_data_loaders(
            dataset_name=self.config["DATASET"],
            batch_size=self.config["BATCH_SIZE"],
            tokenizer=self.tokenizer,
        )

        # Initialize model
        self.model = TinyMistralForClassification(
            model_name=self.config["MODEL_NAME"], num_classes=len(self.class_names)
        ).to(self.device)

        # Freeze base model, only train classifier
        trainable_params = 0
        for name, param in self.model.named_parameters():
            param.requires_grad = "classifier" in name
            if param.requires_grad:
                trainable_params += param.numel()

        self.logger.info(f"Trainable parameters: {trainable_params:,}")

        # Setup training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config["LEARNING_RATE"],
            weight_decay=self.config.get("WEIGHT_DECAY", 0.01),
        )

        self.logger.info(f"Model setup complete. Classes: {self.class_names}")

    def _process_batch(self, batch):
        """Move batch to device and extract tensors."""
        return (
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
            batch["labels"].to(self.device),
        )

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        max_batches = min(
            self.config.get("MAX_BATCHES_PER_EPOCH", float("inf")),
            len(self.train_loader),
        )
        progress_bar = tqdm(
            enumerate(self.train_loader), total=max_batches, desc=f"Epoch {epoch+1}"
        )

        for batch_idx, batch in progress_bar:
            if batch_idx >= max_batches:
                break

            input_ids, attention_mask, labels = self._process_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            loss.backward()

            if self.config.get("GRADIENT_CLIPPING", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config["GRADIENT_CLIPPING"]
                )

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Acc": f"{100 * correct / total:.2f}%"}
            )

        return total_loss / max_batches, 100 * correct / total

    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = {name: 0 for name in self.class_names}
        class_total = {name: 0 for name in self.class_names}

        with torch.no_grad():
            max_val_batches = self.config.get("MAX_VAL_BATCHES", float("inf"))
            progress_bar = tqdm(
                enumerate(self.val_loader),
                total=min(max_val_batches, len(self.val_loader)),
                desc="Validation",
            )

            for batch_idx, batch in progress_bar:
                if batch_idx >= max_val_batches:
                    break

                input_ids, attention_mask, labels = self._process_batch(batch)
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{100 * correct / total:.2f}%",
                    }
                )

                # Per-class accuracy
                for i, class_name in enumerate(self.class_names):
                    class_mask = labels == i
                    if class_mask.sum() > 0:
                        class_total[class_name] += class_mask.sum().item()
                        class_correct[class_name] += (
                            (predicted[class_mask] == labels[class_mask]).sum().item()
                        )

        avg_loss = total_loss / min(max_val_batches, len(self.val_loader))
        accuracy = 100 * correct / total

        class_accuracies = {
            name: (
                100 * class_correct[name] / class_total[name]
                if class_total[name] > 0
                else 0.0
            )
            for name in self.class_names
        }

        return avg_loss, accuracy, class_accuracies

    def save_checkpoint(self, epoch, accuracy, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
            "config": self.config,
            "class_names": self.class_names,
        }

        # Save regular checkpoint
        checkpoint_path = (
            self.results_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            torch.save(checkpoint, self.results_dir / "models" / "best_model.pt")
            self.logger.info(f"New best model saved: {accuracy:.2f}%")

        return checkpoint_path

    def train(self):
        """Complete training loop."""
        self.logger.info("Starting TinyMistral training...")

        best_accuracy = 0.0
        train_losses = []
        val_accuracies = []

        for epoch in range(self.config["NUM_EPOCHS"]):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)

            # Validation
            val_loss, val_acc, class_accs = self.evaluate()
            val_accuracies.append(val_acc)

            # Logging
            self.logger.info(
                f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )

            # Save checkpoint
            is_best = val_acc > best_accuracy
            if is_best:
                best_accuracy = val_acc
            self.save_checkpoint(epoch, val_acc, is_best)

        self.logger.info(f"Training complete! Best accuracy: {best_accuracy:.2f}%")

        return {
            "best_accuracy": best_accuracy,
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "final_class_accuracies": class_accs,
        }

    def load_trained_model(self):
        """Load a previously trained model."""
        best_model_path = self.results_dir / "models" / "best_model.pt"

        if not best_model_path.exists():
            raise FileNotFoundError(f"No trained model found at {best_model_path}")

        checkpoint = torch.load(best_model_path, map_location=self.device)

        # Verify required keys
        required_keys = ["model_state_dict", "accuracy", "class_names"]
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            raise KeyError(f"Checkpoint missing required keys: {missing_keys}")

        # Load class names and model state
        self.class_names = checkpoint["class_names"]
        state_dict = checkpoint["model_state_dict"]

        # Fix key mismatch if needed
        if any(key.startswith("base_model.") for key in state_dict.keys()):
            state_dict = {
                (
                    key.replace("base_model.", "model.")
                    if key.startswith("base_model.")
                    else key
                ): value
                for key, value in state_dict.items()
            }

        self.model.load_state_dict(state_dict)
        self.logger.info(f"Model loaded successfully: {checkpoint['accuracy']:.2f}%")

        return {"best_accuracy": checkpoint["accuracy"]}

    def run_training_pipeline(self, skip_training=False):
        """Run the complete training pipeline."""
        self.logger.info("Starting TinyMistral Training Pipeline")

        # Setup
        self.setup_model_and_data()

        # Training or loading
        if skip_training:
            performance_metrics = self.load_trained_model()
        else:
            performance_metrics = self.train()

        self.logger.info("TinyMistral training pipeline completed successfully!")
        return performance_metrics


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="TinyMistral Training Pipeline")
    parser.add_argument(
        "--config",
        default="configs/tinymistral_training.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and load existing model",
    )

    args = parser.parse_args()

    # Initialize and run pipeline
    pipeline = TinyMistralPipeline(args.config)
    pipeline.run_training_pipeline(skip_training=args.skip_training)


if __name__ == "__main__":
    main()
