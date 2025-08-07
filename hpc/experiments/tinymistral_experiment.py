"""
TinyMistral analysis experiment for HPC.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from .base_experiment import BaseExperiment


class TinyMistralExperiment(BaseExperiment):
    """Experiment for analyzing TinyMistral model."""

    def __init__(self, config_path):
        super().__init__(config_path, "tinymistral_analysis")

    def load_model_and_data(self):
        """Load the TinyMistral model and dataset."""
        self.logger.info(f"Loading model from: {self.config['MODEL_PATH']}")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["MODEL_PATH"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["MODEL_PATH"], torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

        # Load dataset
        self.logger.info("Loading 20 Newsgroups dataset...")
        dataset = load_dataset("SetFit/20_newsgroups", split="test")

        def collate_fn(batch):
            texts = [item["text"] for item in batch]
            labels = [item["label"] for item in batch]
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["MAX_LENGTH"],
            )
            inputs["labels"] = torch.tensor(labels)
            return inputs

        self.test_loader = DataLoader(
            dataset, batch_size=self.config["BATCH_SIZE"], collate_fn=collate_fn
        )

    def evaluate_per_class_accuracy(self):
        """Evaluate per-class accuracy for the model."""
        class_correct = [0] * self.config["NUM_CLASSES"]
        class_totals = [0] * self.config["NUM_CLASSES"]

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.config["DEVICE"])
                attention_mask = batch["attention_mask"].to(self.config["DEVICE"])
                labels = batch["labels"].to(self.config["DEVICE"])

                # Get model outputs - this is a causal LM
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits[:, -1, :]  # Last token logits

                # Simple classification head (in practice, you'd want something more sophisticated)
                classifier_head = torch.nn.Linear(
                    logits.size(-1), self.config["NUM_CLASSES"]
                ).to(self.config["DEVICE"])
                class_logits = classifier_head(logits)

                _, predicted = torch.max(class_logits.data, 1)

                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_totals[label] += 1

        accuracies = [
            (class_correct[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
            for i in range(self.config["NUM_CLASSES"])
        ]
        return accuracies

    def run_ablation_analysis(self):
        """Run ablation analysis on MoE experts."""
        self.logger.info("Starting ablation analysis...")

        # Calculate baseline accuracy
        baseline_accuracies = self.evaluate_per_class_accuracy()
        self.logger.info("Baseline accuracies calculated")

        # Store results
        ablation_results = []

        # Ablate each expert in the first MoE layer
        moe_layer_index = 0  # First MoE layer

        for expert_to_ablate in range(self.config["NUM_EXPERTS"]):
            self.logger.info(f"Ablating Expert {expert_to_ablate}...")

            # Get the expert path
            expert_path = self.model.model.layers[
                moe_layer_index
            ].block_sparse_moe.experts[expert_to_ablate]

            # Store original weights
            original_weights = [p.clone().detach() for p in expert_path.parameters()]

            # Zero out the expert
            with torch.no_grad():
                for param in expert_path.parameters():
                    param.data.fill_(0)

            # Evaluate
            ablated_accuracies = self.evaluate_per_class_accuracy()
            ablation_results.append(ablated_accuracies)

            # Restore weights
            with torch.no_grad():
                for i, param in enumerate(expert_path.parameters()):
                    param.data.copy_(original_weights[i])

            self.logger.info(f"Expert {expert_to_ablate} restored")

        # Calculate accuracy drops
        accuracy_drops = [
            [
                base - ablated
                for base, ablated in zip(baseline_accuracies, expert_accuracies)
            ]
            for expert_accuracies in ablation_results
        ]

        # Create results dataframe
        class_names_short = [name[:15] for name in self.config["CLASS_NAMES"]]
        results_df = pd.DataFrame(
            accuracy_drops,
            columns=class_names_short,
            index=[f"Ablate Expert {i}" for i in range(self.config["NUM_EXPERTS"])],
        )

        return {
            "baseline_accuracies": baseline_accuracies,
            "ablation_results": results_df,
            "accuracy_drops": accuracy_drops,
        }

    def run(self):
        """Run the complete TinyMistral analysis."""
        self.load_model_and_data()

        # Run ablation analysis
        results = self.run_ablation_analysis()

        # Save results
        self.save_results(results, "ablation_results.pt")

        # Save results as CSV for easy reading
        results["ablation_results"].to_csv(
            self.paths["analysis_dir"] / "ablation_results.csv"
        )

        self.logger.info("Analysis complete!")
        return results


if __name__ == "__main__":
    config_path = "hpc/configs/tinymistral_config.yaml"

    with TinyMistralExperiment(config_path) as experiment:
        results = experiment.run()
