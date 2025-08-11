#!/usr/bin/env python3
"""
TinyMistral Simple Accuracy-Based Ablation Analysis
Ultra-lightweight using direct token prediction accuracy instead of classification.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TinyMistralSimpleAccuracyAblation:
    """Ultra-simple accuracy-based ablation using next token prediction."""

    def __init__(self, model_name="M4-ai/TinyMistral-6x248M", max_samples=200):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.max_samples = max_samples
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Simple token prediction categories
        self.category_tokens = {
            "World": ["world", "international", "country", "nation", "global"],
            "Sports": ["sports", "game", "team", "player", "match"],
            "Business": ["business", "company", "market", "stock", "economic"],
            "Sci/Tech": ["technology", "science", "computer", "research", "tech"],
        }

    def load_model(self):
        """Load TinyMistral model and tokenizer."""
        logger.info(f"🔄 Loading TinyMistral model: {self.model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32, device_map=self.device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get model info
        self.num_layers = len(self.model.model.layers)
        first_moe = self.model.model.layers[0].block_sparse_moe
        self.num_experts = first_moe.num_experts

        logger.info(
            f"✅ Model loaded: {self.num_layers} layers, {self.num_experts} experts per layer"
        )
        logger.info(f"📊 Device: {self.device}")

    def prepare_simple_data(self):
        """Prepare simple token completion tasks from AG News."""
        logger.info(
            f"📚 Creating simple completion tasks (max {self.max_samples} samples)..."
        )

        # Load dataset
        dataset = load_dataset("ag_news", split="test")

        samples = []
        samples_per_class = self.max_samples // 4
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for text, label in zip(dataset["text"], dataset["label"]):
            if class_counts[label] >= samples_per_class:
                continue

            # Create simple completion task
            words = text.split()
            if len(words) > 20:
                prompt = " ".join(words[:15])  # Take first 15 words
                target = words[15].lower()  # Predict 16th word

                samples.append(
                    {
                        "prompt": prompt,
                        "target": target,
                        "label": label,
                        "category": ["World", "Sports", "Business", "Sci/Tech"][label],
                    }
                )
                class_counts[label] += 1

            if sum(class_counts.values()) >= self.max_samples:
                break

        logger.info(f"✅ Dataset prepared: {len(samples)} completion tasks")
        for i, (cat, count) in enumerate(
            zip(["World", "Sports", "Business", "Sci/Tech"], class_counts.values())
        ):
            logger.info(f"  {cat}: {count} samples")

        return samples

    def evaluate_completion_accuracy(self, samples, description="Evaluating"):
        """Evaluate next-token prediction accuracy."""
        self.model.eval()

        class_correct = {i: 0 for i in range(4)}
        class_total = {i: 0 for i in range(4)}

        with torch.no_grad():
            for sample in tqdm(samples, desc=description, leave=False):
                try:
                    prompt = sample["prompt"]
                    target_word = sample["target"]
                    label = sample["label"]

                    # Tokenize prompt
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=50,  # Short context
                        truncation=True,
                    ).to(self.device)

                    # Forward pass
                    outputs = self.model(**inputs)

                    # Get next token predictions
                    next_token_logits = outputs.logits[
                        0, -1, :
                    ]  # Last token predictions
                    predicted_token_id = torch.argmax(next_token_logits).item()
                    predicted_word = (
                        self.tokenizer.decode([predicted_token_id]).strip().lower()
                    )

                    # Check if prediction is correct
                    class_total[label] += 1
                    if (
                        predicted_word == target_word
                        or predicted_word in target_word
                        or target_word in predicted_word
                    ):
                        class_correct[label] += 1

                except Exception as e:
                    logger.warning(f"Error processing sample: {e}")
                    continue

        # Calculate accuracy per class
        class_accuracies = {}
        for class_id in range(4):
            if class_total[class_id] > 0:
                accuracy = (class_correct[class_id] / class_total[class_id]) * 100
                class_accuracies[class_id] = accuracy
            else:
                class_accuracies[class_id] = 0

        return class_accuracies

    def ablate_expert_and_measure_accuracy(
        self, samples, target_layer, expert_to_ablate
    ):
        """Ablate one expert and measure accuracy impact."""

        # Get expert module
        expert_module = self.model.model.layers[target_layer].block_sparse_moe.experts[
            expert_to_ablate
        ]

        # Store original weights
        original_weights = {}
        for name, param in expert_module.named_parameters():
            original_weights[name] = param.data.clone()

        # Zero out expert weights
        with torch.no_grad():
            for param in expert_module.parameters():
                param.data.fill_(0)

        # Measure accuracy with ablated expert
        ablated_accuracies = self.evaluate_completion_accuracy(
            samples, f"Ablating L{target_layer}-E{expert_to_ablate}"
        )

        # Restore original weights
        with torch.no_grad():
            for name, param in expert_module.named_parameters():
                param.data.copy_(original_weights[name])

        return ablated_accuracies

    def run_layer_ablation_analysis(self, samples, target_layer=0):
        """Run accuracy-based ablation analysis on a specific layer."""
        logger.info(
            f"🔬 Starting simple accuracy ablation analysis on Layer {target_layer}"
        )
        logger.info(f"⚡ Using {len(samples)} completion tasks")

        # Measure baseline accuracy
        logger.info("📊 Measuring baseline accuracy...")
        baseline_accuracies = self.evaluate_completion_accuracy(samples, "Baseline")

        logger.info("Baseline Next-Token Accuracy by Class (%):")
        baseline_data = []
        class_names = ["World", "Sports", "Business", "Sci/Tech"]
        for class_id in range(4):
            acc = baseline_accuracies[class_id]
            logger.info(f"  {class_names[class_id]}: {acc:.1f}%")
            baseline_data.append(acc)

        # Run ablation for each expert
        logger.info(f"🎯 Ablating {self.num_experts} experts...")

        accuracy_drops = []
        expert_impacts = []

        for expert_idx in range(self.num_experts):
            # Measure with expert ablated
            ablated_accuracies = self.ablate_expert_and_measure_accuracy(
                samples, target_layer, expert_idx
            )

            # Calculate accuracy drops (baseline - ablated)
            drops = []
            total_drop = 0

            for class_id in range(4):
                baseline = baseline_accuracies[class_id]
                ablated = ablated_accuracies[class_id]
                drop = baseline - ablated  # Positive = performance drop
                drops.append(drop)
                total_drop += drop

            accuracy_drops.append(drops)
            expert_impacts.append(total_drop)

            avg_drop = total_drop / 4
            logger.info(f"  Expert {expert_idx}: Avg accuracy drop {avg_drop:.2f}%")

        # Create results dataframe
        results_df = pd.DataFrame(
            accuracy_drops,
            columns=[name[:10] for name in class_names],
            index=[f"L{target_layer}-E{i}" for i in range(self.num_experts)],
        )

        baseline_df = pd.DataFrame(
            [baseline_data],
            columns=[name[:10] for name in class_names],
            index=["Baseline"],
        )

        logger.info("\n🎯 Ablation Results (Accuracy Drop %):")
        logger.info(f"\n{results_df.round(2)}")

        return results_df, baseline_df, expert_impacts

    def create_visualizations(
        self, results_df, baseline_df, output_dir="results/tinymistral_simple_accuracy"
    ):
        """Create simple accuracy-based visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"📊 Creating accuracy ablation visualizations...")

        # Combine baseline and results for full view
        combined_df = pd.concat([baseline_df, results_df])

        # 1. Accuracy Change Table (Baseline + Drops)
        plt.figure(figsize=(12, 8))

        # Create heatmap showing both baseline and drops
        sns.heatmap(
            combined_df,
            annot=True,
            fmt=".1f",
            cmap="RdYlBu_r",  # Red for drops, blue for good performance
            cbar_kws={"label": "Accuracy (%) / Drop (%)"},
            center=0,
            vmin=-50,
            vmax=50,
        )
        plt.title(
            "TinyMistral Next-Token Prediction: Baseline vs Expert Ablation",
            fontweight="bold",
        )
        plt.xlabel("News Categories")
        plt.ylabel("Condition")
        plt.tight_layout()

        heatmap_path = output_path / "accuracy_change_table.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Accuracy change table saved: {heatmap_path}")

        # 2. Expert Impact Ranking (only drops)
        expert_total_impact = results_df.sum(axis=1).sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(expert_total_impact)), expert_total_impact.values)
        plt.title("Expert Impact Ranking (Total Accuracy Drop %)", fontweight="bold")
        plt.xlabel("Expert (Ranked by Impact)")
        plt.ylabel("Total Accuracy Drop (%)")
        plt.xticks(
            range(len(expert_total_impact)), expert_total_impact.index, rotation=45
        )

        # Color bars by impact level
        max_impact = max(expert_total_impact.max(), 1)  # Avoid division by zero
        for bar, impact in zip(bars, expert_total_impact.values):
            normalized_impact = max(
                0, impact / max_impact
            )  # Normalize positive impacts
            bar.set_color(plt.cm.Reds(min(1.0, normalized_impact)))

        plt.tight_layout()

        impact_path = output_path / "expert_impact_ranking.png"
        plt.savefig(impact_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Impact ranking saved: {impact_path}")

        # 3. Save detailed results
        combined_results_path = output_path / "baseline_and_ablation_results.csv"
        combined_df.to_csv(combined_results_path)

        impact_summary = pd.DataFrame(
            {
                "Expert": expert_total_impact.index,
                "Total_Accuracy_Drop": expert_total_impact.values,
                "Avg_Accuracy_Drop": expert_total_impact.values / 4,
            }
        ).reset_index(drop=True)

        impact_summary_path = output_path / "expert_impact_summary.csv"
        impact_summary.to_csv(impact_summary_path, index=False)

        logger.info(f"✅ Results saved to {output_path}")

        # Print summary
        logger.info("\n📊 **ACCURACY ANALYSIS SUMMARY**")
        logger.info("\nBaseline Performance:")
        for col in baseline_df.columns:
            baseline_acc = baseline_df.loc["Baseline", col]
            logger.info(f"  {col}: {baseline_acc:.1f}%")

        logger.info("\n🏆 Top 3 Most Impactful Experts:")
        for i, (expert, impact) in enumerate(expert_total_impact.head(3).items()):
            logger.info(f"  {i+1}. {expert}: -{impact:.2f}% total accuracy drop")

        return output_path


def main():
    """Main simple accuracy ablation analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="TinyMistral Simple Accuracy Ablation Analysis"
    )
    parser.add_argument(
        "--samples", type=int, default=80, help="Max samples to use (default: 80)"
    )
    parser.add_argument(
        "--single-layer", type=int, help="Analyze only specific layer (0-11)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/tinymistral_simple_accuracy",
        help="Output directory",
    )

    args = parser.parse_args()

    logger.info("🚀 Starting TinyMistral Simple Accuracy Ablation Analysis")
    logger.info(f"⚡ Ultra-lightweight with {args.samples} completion tasks")

    # Initialize analyzer
    analyzer = TinyMistralSimpleAccuracyAblation(max_samples=args.samples)
    analyzer.load_model()

    # Prepare data
    samples = analyzer.prepare_simple_data()

    # Run analysis
    if args.single_layer is not None:
        logger.info(f"🎯 Single layer analysis: Layer {args.single_layer}")
        results_df, baseline_df, expert_impacts = analyzer.run_layer_ablation_analysis(
            samples, target_layer=args.single_layer
        )

        # Create visualizations
        output_path = analyzer.create_visualizations(
            results_df, baseline_df, args.output_dir
        )

        logger.info("🎉 Simple accuracy ablation analysis completed!")
        logger.info(f"📁 Results saved to: {output_path}")
        logger.info(f"🔍 **ACCURACY TABLE**: {output_path}/accuracy_change_table.png")
    else:
        logger.info("❗ Multi-layer analysis not implemented in simple version")
        logger.info("   Use --single-layer 0 to analyze a specific layer")


if __name__ == "__main__":
    main()
