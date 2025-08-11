#!/usr/bin/env python3
"""
Generate TinyMistral ablation heatmap using existing visualization functions.
"""

import torch
import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.analysis.visualization import plot_ablation_heatmap

def generate_tinymistral_ablation_graph():
    """Generate the ablation heatmap for TinyMistral using existing visualization."""
    
    print("🚀 Generating TinyMistral Ablation Heatmap")
    print("=" * 50)
    
    # Load the lightweight results
    results_path = "results/tinymistral_lightweight/analysis/lightweight_results.pt"
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        return
    
    print(f"📊 Loading results from {results_path}")
    results = torch.load(results_path, map_location='cpu')
    
    # Get expert analysis data (6 experts × 4 classes)
    expert_analysis = results.get('expert_analysis')
    class_names = results.get('class_names', ['World', 'Sports', 'Business', 'Sci/Tech'])
    
    if expert_analysis is None:
        print("❌ No expert_analysis data found")
        return
    
    print(f"✅ Expert analysis shape: {expert_analysis.shape}")
    
    # Convert to pandas DataFrame for the heatmap
    expert_labels = [f"Expert_{i}" for i in range(expert_analysis.shape[0])]
    ablation_df = pd.DataFrame(
        expert_analysis.numpy(),
        index=expert_labels,
        columns=class_names
    )
    
    print("\n📈 Expert Analysis Data:")
    print(ablation_df.round(3))
    
    # Create output directory
    output_dir = Path("results/tinymistral_lightweight/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate the ablation heatmap
    save_path = output_dir / "tinymistral_ablation_heatmap.png"
    
    print(f"\n🎨 Generating ablation heatmap...")
    plot_ablation_heatmap(ablation_df, save_path=save_path)
    
    # Also save detailed statistics
    summary_stats = {
        'mean_per_expert': ablation_df.mean(axis=1),
        'max_per_expert': ablation_df.max(axis=1),
        'min_per_expert': ablation_df.min(axis=1),
        'std_per_expert': ablation_df.std(axis=1),
        'mean_per_class': ablation_df.mean(axis=0),
        'max_per_class': ablation_df.max(axis=0),
    }
    
    print("\n📊 Expert Statistics:")
    print("Mean utilization per expert:")
    for expert, mean_val in summary_stats['mean_per_expert'].items():
        max_val = summary_stats['max_per_expert'][expert]
        print(f"  {expert}: {mean_val:.3f} (max: {max_val:.3f})")
    
    print("\nMean utilization per class:")
    for class_name, mean_val in summary_stats['mean_per_class'].items():
        max_val = summary_stats['max_per_class'][class_name]
        print(f"  {class_name}: {mean_val:.3f} (max: {max_val:.3f})")
    
    # Find expert specialization
    expert_specialization = ablation_df.idxmax(axis=1)
    print("\n🎯 Expert Specialization (highest impact class):")
    for expert, specialized_class in expert_specialization.items():
        impact = ablation_df.loc[expert, specialized_class]
        print(f"  {expert}: {specialized_class} ({impact:.3f})")
    
    # Save detailed results
    ablation_df.to_csv(output_dir / "tinymistral_ablation_detailed.csv")
    
    specialization_df = pd.DataFrame({
        'Expert': expert_specialization.index,
        'Specialized_Class': expert_specialization.values,
        'Max_Impact': [ablation_df.loc[expert, class_name] 
                      for expert, class_name in expert_specialization.items()]
    })
    specialization_df.to_csv(output_dir / "expert_specialization.csv", index=False)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📁 Results saved in {output_dir}:")
    print(f"   📊 tinymistral_ablation_heatmap.png (Your ablation graph!)")
    print(f"   📄 tinymistral_ablation_detailed.csv")
    print(f"   🎯 expert_specialization.csv")
    
    return ablation_df

if __name__ == "__main__":
    generate_tinymistral_ablation_graph()
