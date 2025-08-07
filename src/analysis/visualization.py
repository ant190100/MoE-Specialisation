"""
Visualization utilities for MoE analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from tqdm import tqdm


def plot_ablation_heatmap(accuracy_drops_df, save_path=None):
    """Generates a heatmap to visualize the accuracy drop."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        accuracy_drops_df, annot=True, fmt=".1f", cmap="viridis", linewidths=0.5
    )
    plt.title("Expert Specialization: Accuracy Drop (%) After Ablation", fontsize=16)
    plt.xlabel("News Category", fontsize=12)
    plt.ylabel("Ablated Expert", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Ablation heatmap saved to {save_path}")
        plt.close()  # Close the figure to free memory
    else:
        plt.show()


def plot_expert_utilization_bar(
    model, data_loader, device, embed_dim, top_k, num_experts, save_path=None
):
    """
    Generates a bar chart of expert utilization for a sample batch.
    """
    print("\n--- Calculating Expert Utilization for a Sample Batch ---")
    model.eval()

    with torch.no_grad():
        sample_batch = next(iter(data_loader))
        input_ids = sample_batch["input_ids"].to(device)

        embedded = model.embedding(input_ids)
        x_flat = embedded.view(-1, embed_dim)
        router_logits = model.moe_layer.router(x_flat)

        _, chosen_expert_indices = torch.topk(router_logits, top_k, dim=-1)

        expert_counts = torch.bincount(
            chosen_expert_indices.flatten(), minlength=num_experts
        )

        labels = [f"Expert {i}" for i in range(num_experts)]
        counts = expert_counts.cpu().numpy()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=counts)
        plt.title("Expert Utilization for a Sample Batch", fontsize=16)
        plt.xlabel("Expert", fontsize=12)
        plt.ylabel("Number of Tokens Routed", fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Token distribution plot saved to {save_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()


def plot_routing_entropy(
    model, data_loader, device, embed_dim, num_experts, tokenizer, save_path=None
):
    """
    Calculates and plots the distribution of routing entropy for a sample batch,
    and shows the text of the most and least confident token routings.
    """
    print("\n--- Calculating Routing Entropy for a Sample Batch ---")
    model.eval()

    with torch.no_grad():
        sample_batch = next(iter(data_loader))
        input_ids = sample_batch["input_ids"].to(device)

        embedded = model.embedding(input_ids)
        x_flat = embedded.view(-1, embed_dim)
        router_logits = model.moe_layer.router(x_flat)

        probs = F.softmax(router_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-9), dim=-1)

        max_entropy = np.log2(num_experts)
        uncertainty_score = (entropy / max_entropy * 100).cpu().numpy()

        fig, ax = plt.subplots(figsize=(12, 6))
        n_bins = 30
        n, bins, patches = ax.hist(uncertainty_score, bins=n_bins)

        cmap = plt.get_cmap("RdYlGn_r")
        for i, p in enumerate(patches):
            plt.setp(p, "facecolor", cmap(i / n_bins))

        avg_uncertainty = np.mean(uncertainty_score)
        ax.axvline(
            avg_uncertainty,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Avg. Uncertainty ({avg_uncertainty:.1f}%)",
        )

        plt.suptitle("Router Confidence Analysis", fontsize=16, fontweight="bold")
        plt.title(
            "Distribution of Routing Uncertainty Score for a Sample Batch", fontsize=12
        )
        plt.xlabel("Routing Uncertainty Score (%)", fontsize=12)
        plt.ylabel("Token Count", fontsize=12)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ Routing entropy plot saved to {save_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

        # Qualitative Analysis
        sorted_indices = np.argsort(uncertainty_score)
        input_ids_flat = input_ids.cpu().numpy().flatten()
        probs_np = probs.cpu().numpy()

        print("\n--- Qualitative Routing Analysis ---")

        # Most Confident Token
        confident_idx = sorted_indices[0]
        token_id = input_ids_flat[confident_idx]
        token_text = tokenizer.decode([token_id])
        token_probs = probs_np[confident_idx]

        print(f"\nMOST CONFIDENT Routing (Lowest Entropy):")
        print(f"  - Token: '{token_text}'")
        print(f"  - Uncertainty Score: {uncertainty_score[confident_idx]:.1f}%")
        print(f"  - Expert Probabilities:")
        for i, p in enumerate(token_probs):
            print(f"    - Expert {i}: {p*100:.1f}%")

        # Most Uncertain Token
        uncertain_idx = sorted_indices[-1]
        token_id = input_ids_flat[uncertain_idx]
        token_text = tokenizer.decode([token_id])
        token_probs = probs_np[uncertain_idx]

        print(f"\nMOST UNCERTAIN Routing (Highest Entropy):")
        print(f"  - Token: '{token_text}'")
        print(f"  - Uncertainty Score: {uncertainty_score[uncertain_idx]:.1f}%")
        print(f"  - Expert Probabilities:")
        for i, p in enumerate(token_probs):
            print(f"    - Expert {i}: {p*100:.1f}%")


def create_analysis_report(results, output_dir, experiment_name="analysis"):
    """Create a comprehensive analysis report with multiple visualizations."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    if "ablation_results" in results:
        plot_ablation_heatmap(
            results["ablation_results"],
            save_path=os.path.join(
                output_dir, f"{experiment_name}_ablation_heatmap.png"
            ),
        )

    print(f"Analysis report saved to {output_dir}")


def run_full_visualization_analysis(model, test_loader, device, config, tokenizer):
    """
    Performs and visualizes a full suite of diagnostic tests.
    """
    # Expert Utilization Analysis
    plot_expert_utilization_bar(
        model,
        test_loader,
        device,
        config["EMBED_DIM"],
        config["TOP_K"],
        config["NUM_EXPERTS"],
    )

    # Routing Entropy Analysis
    plot_routing_entropy(
        model,
        test_loader,
        device,
        config["EMBED_DIM"],
        config["NUM_EXPERTS"],
        tokenizer,
    )
