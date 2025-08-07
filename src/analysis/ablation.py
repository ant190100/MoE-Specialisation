"""
Ablation analysis for MoE models.
"""

import torch
import pandas as pd
from tqdm import tqdm


def evaluate_per_class_accuracy(model, test_loader, device, num_classes):
    """Evaluate per-class accuracy for a model."""
    class_correct = [0] * num_classes
    class_totals = [0] * num_classes

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()
                class_totals[label] += 1

    accuracies = [
        (class_correct[i] / class_totals[i]) * 100 if class_totals[i] > 0 else 0
        for i in range(num_classes)
    ]
    return accuracies


def run_ablation_analysis(
    model, test_loader, device, num_classes, num_experts, class_names
):
    """
    Run ablation analysis by systematically zeroing out each expert
    and measuring the impact on per-class accuracy.
    """
    print("\n--- Starting Ablation Analysis ---")

    # Calculate baseline accuracy
    print("Calculating baseline accuracy...")
    baseline_accuracies = evaluate_per_class_accuracy(
        model, test_loader, device, num_classes
    )

    print("\nBaseline Per-Class Accuracy (%):")
    baseline_df = pd.DataFrame(
        [baseline_accuracies], columns=class_names, index=["Baseline Acc."]
    )
    print(baseline_df.round(2))

    ablation_results = []

    # Run ablation for each expert
    for expert_to_ablate in range(num_experts):
        print(f"\nAblating Expert {expert_to_ablate}...")

        # Store original weights
        original_weights = [
            p.clone().detach()
            for p in model.moe_layer.experts[expert_to_ablate].parameters()
        ]

        # Zero out the expert
        with torch.no_grad():
            for param in model.moe_layer.experts[expert_to_ablate].parameters():
                param.data.fill_(0)

        # Evaluate with ablated expert
        ablated_accuracies = evaluate_per_class_accuracy(
            model, test_loader, device, num_classes
        )
        ablation_results.append(ablated_accuracies)

        # Restore original weights
        with torch.no_grad():
            for i, param in enumerate(
                model.moe_layer.experts[expert_to_ablate].parameters()
            ):
                param.data.copy_(original_weights[i])

        print(f"Restored Expert {expert_to_ablate}.")

    # Calculate accuracy drops
    accuracy_drops = [
        [
            base - ablated
            for base, ablated in zip(baseline_accuracies, expert_accuracies)
        ]
        for expert_accuracies in ablation_results
    ]

    # Create results dataframe
    results_df = pd.DataFrame(
        accuracy_drops,
        columns=class_names,
        index=[f"Ablate Expert {i}" for i in range(num_experts)],
    )

    print("\n--- Ablation Analysis Results ---")
    print("Accuracy Drop (%) After Ablating Each Expert:")
    print(results_df.round(2))

    return results_df
