"""
Local analysis script for processing results from HPC runs.
"""

import argparse
import torch
import pandas as pd
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.analysis.visualization import (
    plot_ablation_heatmap,
    create_analysis_report,
    plot_expert_utilization_bar,
    plot_routing_entropy,
    run_full_visualization_analysis,
)
from src.utils.config import load_config, setup_experiment_paths
from src.models.tiny_moe import TinyMoEForClassification
from src.data.loaders import create_data_loaders
from transformers import AutoTokenizer


def analyze_toy_model_results(results_path, output_dir=None):
    """Analyze toy model results with comprehensive visualizations."""
    print(f"Loading results from {results_path}")
    results = torch.load(results_path, map_location="cpu")

    if output_dir is None:
        # Use the same directory where results are stored
        output_dir = Path(results_path).parent

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract results and config
    ablation_df = results["ablation_results"]
    config = results["config"]

    print("\nAblation Analysis Results:")
    print(ablation_df.round(2))

    # Create ablation visualization
    plot_ablation_heatmap(ablation_df, save_path=output_dir / "ablation_heatmap.png")

    # Load model and data for token distribution and entropy analysis
    print("\nLoading model for token distribution and entropy analysis...")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["TOKENIZER_NAME"])

        # Recreate data loaders
        _, test_loader, class_names = create_data_loaders(
            dataset_name=config["DATASET"],
            batch_size=config["BATCH_SIZE"],
            tokenizer=tokenizer,
        )

        # Recreate model
        model = TinyMoEForClassification(
            vocab_size=len(tokenizer),
            embed_dim=config["EMBED_DIM"],
            hidden_dim=config["HIDDEN_DIM"],
            num_experts=config["NUM_EXPERTS"],
            top_k=config["TOP_K"],
            num_classes=config["NUM_CLASSES"],
        ).to(config["DEVICE"])

        # Load model weights
        model_weights_path = (
            Path(results_path).parent.parent / "models" / "trained_model.pt"
        )
        if model_weights_path.exists():
            model.load_state_dict(
                torch.load(model_weights_path, map_location=config["DEVICE"])
            )
            print("✅ Model weights loaded successfully")

            # Generate token distribution and entropy visualizations
            print("\nGenerating token distribution analysis...")
            plot_expert_utilization_bar(
                model,
                test_loader,
                config["DEVICE"],
                config["EMBED_DIM"],
                config["TOP_K"],
                config["NUM_EXPERTS"],
                save_path=output_dir / "token_distribution_by_expert.png",
            )

            print("\nGenerating routing entropy analysis...")
            plot_routing_entropy(
                model,
                test_loader,
                config["DEVICE"],
                config["EMBED_DIM"],
                config["NUM_EXPERTS"],
                tokenizer,
                save_path=output_dir / "routing_entropy_distribution.png",
            )

        else:
            print(f"⚠️  Model weights not found at {model_weights_path}")
            print("   Skipping token distribution and entropy analysis")

    except Exception as e:
        print(f"⚠️  Could not generate token/entropy visualizations: {e}")
        print("   Only ablation analysis will be available")

    # Save summary statistics
    summary = {
        "mean_accuracy_drop": ablation_df.mean(axis=1),
        "max_accuracy_drop": ablation_df.max(axis=1),
        "min_accuracy_drop": ablation_df.min(axis=1),
        "std_accuracy_drop": ablation_df.std(axis=1),
    }

    summary_df = pd.DataFrame(summary)
    print("\nSummary Statistics:")
    print(summary_df.round(2))

    # Save to CSV
    summary_df.to_csv(output_dir / "ablation_summary.csv")
    ablation_df.to_csv(output_dir / "ablation_detailed.csv")

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return ablation_df, summary_df


def analyze_tinymistral_results(results_path, output_dir=None):
    """Analyze TinyMistral results."""
    print(f"Loading TinyMistral results from {results_path}")
    results = torch.load(results_path, map_location="cpu")

    if output_dir is None:
        output_dir = Path(results_path).parent / "analysis"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract results
    ablation_df = results["ablation_results"]
    baseline_accuracies = results["baseline_accuracies"]

    print("\nBaseline Accuracies:")
    print(f"Mean: {sum(baseline_accuracies)/len(baseline_accuracies):.2f}%")

    print("\nAblation Analysis Results:")
    print(ablation_df.round(2))

    # Create visualizations
    plot_ablation_heatmap(
        ablation_df, save_path=output_dir / "tinymistral_ablation_heatmap.png"
    )

    # Expert specialization analysis
    expert_specialization = ablation_df.idxmax(axis=1)
    print("\nExpert Specialization (class with highest impact when ablated):")
    for expert, specialized_class in expert_specialization.items():
        max_drop = ablation_df.loc[expert, specialized_class]
        print(f"{expert}: {specialized_class} ({max_drop:.2f}% drop)")

    # Save results
    ablation_df.to_csv(output_dir / "tinymistral_ablation_detailed.csv")

    specialization_df = pd.DataFrame(
        {
            "Expert": expert_specialization.index,
            "Specialized_Class": expert_specialization.values,
            "Max_Impact": [
                ablation_df.loc[expert, class_name]
                for expert, class_name in expert_specialization.items()
            ],
        }
    )
    specialization_df.to_csv(output_dir / "expert_specialization.csv", index=False)

    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return ablation_df, specialization_df


def compare_experiments(experiment_dirs, output_dir="./results/comparisons"):
    """Compare results across multiple experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Comparing experiments...")

    all_results = {}
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        exp_name = exp_path.name

        # Look for results files
        results_files = list(exp_path.glob("**/*results.pt"))
        if results_files:
            results = torch.load(results_files[0], map_location="cpu")
            all_results[exp_name] = results
            print(f"Loaded {exp_name}: {results_files[0]}")

    if len(all_results) < 2:
        print("Need at least 2 experiments to compare")
        return

    # Create comparison plots and tables
    comparison_data = {}
    for exp_name, results in all_results.items():
        if "ablation_results" in results:
            df = results["ablation_results"]
            comparison_data[exp_name] = {
                "mean_impact": df.mean(axis=1).mean(),
                "max_impact": df.max().max(),
                "num_experts": len(df),
                "num_classes": len(df.columns),
            }

    comparison_df = pd.DataFrame(comparison_data).T
    print("\nExperiment Comparison:")
    print(comparison_df.round(2))

    comparison_df.to_csv(output_dir / "experiment_comparison.csv")
    print(f"Comparison saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE experiment results")
    parser.add_argument(
        "--experiment", required=True, help="Experiment name or path to results"
    )
    parser.add_argument(
        "--type",
        choices=["toy_model", "tinymistral", "auto"],
        default="auto",
        help="Type of experiment",
    )
    parser.add_argument("--output-dir", help="Output directory for analysis")
    parser.add_argument(
        "--compare", nargs="+", help="Compare multiple experiment directories"
    )

    args = parser.parse_args()

    if args.compare:
        compare_experiments(args.compare, args.output_dir or "./results/comparisons")
        return

    # Single experiment analysis
    experiment_path = Path(args.experiment)

    if experiment_path.is_dir():
        # Look for results files in directory - check analysis subdirectory first
        results_files = list(experiment_path.glob("**/analysis/*results.pt"))
        if not results_files:
            # Fallback to any results files
            results_files = list(experiment_path.glob("**/*results.pt"))
        if not results_files:
            print(f"No results files found in {experiment_path}")
            return
        results_path = results_files[0]
    else:
        results_path = experiment_path

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return

    # Auto-detect experiment type if not specified
    if args.type == "auto":
        if "toy_model" in str(results_path) or "toy" in str(results_path):
            experiment_type = "toy_model"
        elif "tinymistral" in str(results_path) or "mistral" in str(results_path):
            experiment_type = "tinymistral"
        else:
            print("Could not auto-detect experiment type. Please specify --type")
            return
    else:
        experiment_type = args.type

    # Run appropriate analysis
    if experiment_type == "toy_model":
        analyze_toy_model_results(results_path, args.output_dir)
    elif experiment_type == "tinymistral":
        analyze_tinymistral_results(results_path, args.output_dir)
    else:
        print(f"Unknown experiment type: {experiment_type}")


if __name__ == "__main__":
    main()
