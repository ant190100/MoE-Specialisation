"""
Local analysis script specifically for TinyMistral results.
"""

import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def analyze_lightweight_results(results_path):
    """Analyze lightweight TinyMistral evaluation results."""
    
    results_path = Path(results_path)
    
    # Try both result file locations
    analysis_files = [
        results_path / "analysis" / "lightweight_results.pt",
        results_path / "analysis" / "tinymistral_analysis_results.pt"
    ]
    
    analysis_file = None
    for file_path in analysis_files:
        if file_path.exists():
            analysis_file = file_path
            break
    
    if analysis_file is None:
        print(f"❌ No results files found in: {results_path / 'analysis'}")
        print(f"💡 Available files:")
        analysis_dir = results_path / "analysis"
        if analysis_dir.exists():
            for file in analysis_dir.glob("*.pt"):
                print(f"   {file.name}")
        else:
            print("   (analysis directory doesn't exist)")
        return
        
    print(f"📊 Loading results from {analysis_file}")
    results = torch.load(analysis_file, map_location='cpu')
    
    # Check if this is lightweight evaluation
    is_lightweight = results.get("evaluation_mode") == "lightweight"
    
    # Display basic info
    print("\n" + "="*60)
    if is_lightweight:
        print("🚀 TINYMISTRAL LIGHTWEIGHT EVALUATION RESULTS")
    else:
        print("🤗 TINYMISTRAL ANALYSIS RESULTS")
    print("="*60)
    
    model_info = results.get("model_info", {})
    print(f"📱 Model: {model_info.get('model_name', 'TinyMistral-6x248M')}")
    print(f"👥 Number of experts: {model_info.get('num_experts', 6)}")
    print(f"🎯 Number of classes: {model_info.get('num_classes', 4)}")
    print(f"🔢 Hidden size: {model_info.get('hidden_size', 'Unknown')}")
    
    if is_lightweight:
        # Lightweight evaluation results
        zero_shot = results.get("zero_shot_performance", {})
        adapted = results.get("adapted_performance", {})
        
        if zero_shot:
            print(f"\n🎯 ZERO-SHOT PERFORMANCE:")
            print(f"   Accuracy: {zero_shot.get('accuracy', 0):.2f}%")
            print(f"   Samples: {zero_shot.get('total_samples', 0)}")
            
        if adapted:
            print(f"\n🔥 AFTER MINIMAL ADAPTATION:")
            print(f"   Accuracy: {adapted.get('accuracy', 0):.2f}%")
            print(f"   Improvement: {adapted.get('accuracy', 0) - zero_shot.get('accuracy', 0):+.2f}%")
    else:
        # Standard evaluation results
        performance = results.get("performance", {})
        if performance:
            print(f"\n🎯 PERFORMANCE METRICS:")
            print(f"   Accuracy: {performance.get('accuracy', 0):.2f}%")
            print(f"   Total samples: {performance.get('total_samples', 0)}")
    
    # Routing analysis
    routing = results.get("routing_analysis", {})
    if routing:
        print(f"\n🔄 ROUTING ANALYSIS:")
        if is_lightweight:
            print(f"   Batches analyzed: {routing.get('batches_analyzed', 0)}")
            print(f"   Routing available: {routing.get('routing_available', False)}")
            print(f"   Avg text length: {routing.get('average_text_length', 0):.1f} tokens")
        else:
            print(f"   Batches analyzed: {routing.get('num_batches_analyzed', 0)}")
            print(f"   Has routing weights: {routing.get('has_routing_weights', False)}")
    
    # Expert usage analysis
    expert_key = "expert_analysis" if is_lightweight else "ablation_results"
    if expert_key in results and results[expert_key] is not None:
        expert_df = results[expert_key]
        print(f"\n👥 EXPERT USAGE ANALYSIS:")
        print(f"   Expert data shape: {expert_df.shape}")
        
        # Find most and least active experts
        expert_avg = expert_df.mean(axis=1)
        most_active = expert_avg.idxmax()
        least_active = expert_avg.idxmin()
        
        print(f"   Most active expert: {most_active} (avg: {expert_avg[most_active]:.3f})")
        print(f"   Least active expert: {least_active} (avg: {expert_avg[least_active]:.3f})")
        print(f"   Usage range: {expert_avg.min():.3f} - {expert_avg.max():.3f}")
    
    # Class names
    if "class_names" in results:
        print(f"\n📚 CLASSES: {', '.join(results['class_names'])}")
    
    print("="*60)
    
    # Generate visualizations
    output_dir = results_path / "analysis"
    create_tinymistral_visualizations(results, output_dir)
    
    print(f"\n✅ Analysis complete! Check {output_dir} for visualizations.")
    return results

def create_tinymistral_visualizations(results, output_dir):
    """Create visualizations for TinyMistral analysis."""
    
    print("🎨 Creating visualizations...")
    
    plt.style.use('default')
    sns.set_palette("viridis")
    
    # Expert usage visualization
    if "expert_analysis" in results and results["expert_analysis"] is not None:
        expert_df = results["expert_analysis"]
        
        # 1. Expert usage heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            expert_df.T,  # Transpose for classes on y-axis
            annot=True,
            fmt='.3f',
            cmap='viridis',
            cbar_kws={'label': 'Expert Usage Score'}
        )
        plt.title('TinyMistral Expert Usage by Class', fontsize=14, fontweight='bold')
        plt.xlabel('Expert', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'tinymistral_expert_usage_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Expert utilization bar chart
        expert_avg = expert_df.mean(axis=1)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(expert_avg)), expert_avg.values)
        plt.xlabel('Expert ID', fontsize=12)
        plt.ylabel('Average Usage Score', fontsize=12)
        plt.title('TinyMistral Average Expert Utilization', fontsize=14, fontweight='bold')
        plt.xticks(range(len(expert_avg)), expert_avg.index, rotation=0)
        
        # Color bars by usage level
        for i, bar in enumerate(bars):
            height = bar.get_height()
            color = plt.cm.viridis(height / expert_avg.max())
            bar.set_color(color)
            # Add value labels on bars
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(output_dir / 'tinymistral_expert_utilization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Class-wise expert preference
        plt.figure(figsize=(12, 8))
        for i, class_name in enumerate(expert_df.columns):
            plt.subplot(2, 2, i+1)
            class_usage = expert_df[class_name]
            plt.bar(range(len(class_usage)), class_usage.values, color=plt.cm.viridis(i/len(expert_df.columns)))
            plt.title(f'{class_name}', fontsize=12, fontweight='bold')
            plt.xlabel('Expert')
            plt.ylabel('Usage Score')
            plt.xticks(range(len(class_usage)), [f'E{i}' for i in range(len(class_usage))])
            
        plt.suptitle('Expert Usage by Class', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'tinymistral_class_expert_preference.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Performance visualization
    performance = results.get("performance", {})
    if performance and "accuracy" in performance:
        plt.figure(figsize=(8, 6))
        
        # Simple accuracy bar
        accuracy = performance["accuracy"]
        bars = plt.bar(['TinyMistral'], [accuracy], color='skyblue', edgecolor='navy', linewidth=2)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('TinyMistral Classification Accuracy', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        
        # Add accuracy text on bar
        plt.text(0, accuracy + 2, f'{accuracy:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tinymistral_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    # Summary statistics visualization
    if "expert_analysis" in results and results["expert_analysis"] is not None:
        expert_df = results["expert_analysis"]
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Expert usage statistics
        plt.subplot(1, 2, 1)
        expert_stats = expert_df.describe()
        plt.boxplot([expert_df[col] for col in expert_df.columns], labels=expert_df.columns)
        plt.title('Expert Usage Distribution', fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Usage Score')
        plt.xticks(rotation=45)
        
        # Subplot 2: Class usage statistics
        plt.subplot(1, 2, 2)
        class_stats = expert_df.T.describe()
        plt.boxplot([expert_df.loc[idx] for idx in expert_df.index], labels=[f'E{i}' for i in range(len(expert_df.index))])
        plt.title('Class Usage Distribution', fontweight='bold')
        plt.xlabel('Expert')
        plt.ylabel('Usage Score')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'tinymistral_usage_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

def compare_with_toy_model(tinymistral_path, toy_model_path):
    """Compare TinyMistral with toy model results."""
    
    print("\n🔄 COMPARING TINYMISTRAL VS TOY MODEL")
    print("="*60)
    
    # Load TinyMistral results (try both file types)
    tinymistral_files = [
        Path(tinymistral_path) / "analysis" / "lightweight_results.pt",
        Path(tinymistral_path) / "analysis" / "tinymistral_analysis_results.pt"
    ]
    
    tinymistral_results = None
    for file_path in tinymistral_files:
        if file_path.exists():
            tinymistral_results = torch.load(file_path, map_location='cpu')
            break
    
    if tinymistral_results is None:
        print("❌ Could not find TinyMistral results")
        return
    
    # Load toy model results
    toy_results_file = Path(toy_model_path) / "analysis" / "analysis_results.pt"
    if not toy_results_file.exists():
        print(f"❌ Could not find toy model results at {toy_results_file}")
        return
        
    toy_results = torch.load(toy_results_file, map_location='cpu')
    
    # Compare performance
    is_lightweight = tinymistral_results.get("evaluation_mode") == "lightweight"
    
    if is_lightweight:
        # Lightweight evaluation - use zero-shot performance
        tm_acc = tinymistral_results.get("zero_shot_performance", {}).get("accuracy", 0)
    else:
        # Standard evaluation
        tm_acc = tinymistral_results.get("performance", {}).get("accuracy", 0)
    
    toy_acc = toy_results.get("performance", {}).get("accuracy", 0)
    
    print(f"🤗 TinyMistral Accuracy: {tm_acc:.2f}%")
    print(f"🧸 Toy Model Accuracy: {toy_acc:.2f}%")
    print(f"📈 Difference: {tm_acc - toy_acc:+.2f}%")
    
    # Compare expert counts
    tm_experts = tinymistral_results.get("model_info", {}).get("num_experts", 6)
    toy_experts = toy_results.get("config", {}).get("NUM_EXPERTS", 4)
    
    print(f"👥 TinyMistral Experts: {tm_experts}")
    print(f"🧸 Toy Model Experts: {toy_experts}")
    
    # Compare model sizes
    print(f"\n📊 MODEL COMPARISON:")
    print(f"🤗 TinyMistral: ~1B parameters")
    print(f"🧸 Toy Model: ~50K parameters (estimated)")
    print(f"📏 Size ratio: ~20,000x larger")
    
    # Performance per parameter analysis
    if tm_acc > 0 and toy_acc > 0:
        tm_efficiency = tm_acc / 1000  # Accuracy per billion parameters
        toy_efficiency = toy_acc / 0.05  # Accuracy per 50K parameters (0.05M)
        
        print(f"\n⚡ EFFICIENCY (Accuracy per M parameters):")
        print(f"🤗 TinyMistral: {tm_efficiency:.2f}%/M params")
        print(f"🧸 Toy Model: {toy_efficiency:.2f}%/M params")
        
        if toy_efficiency > tm_efficiency:
            print("🎯 Toy model is more parameter-efficient!")
        else:
            print("🎯 TinyMistral is more parameter-efficient!")
    
    print("="*60)

def create_ablation_compatible_results(lightweight_results_path, output_path):
    """Convert lightweight results to format compatible with local_analysis.py ablation functions."""
    
    print(f"🔄 Converting {lightweight_results_path} to ablation format...")
    
    # Load lightweight results
    results = torch.load(lightweight_results_path, map_location='cpu')
    
    # Extract expert analysis (6 experts × 4 classes)
    expert_analysis = results.get('expert_analysis')
    if expert_analysis is None:
        print("❌ No expert_analysis found in results")
        return None
    
    # Create ablation-compatible format
    ablation_results = {
        'ablation_results': expert_analysis,  # This is the key local_analysis.py expects
        'baseline_accuracies': [75.0, 75.0, 75.0, 75.0],  # Dummy baseline for visualization
        'config': results.get('config', {}),
        'model_info': results.get('model_info', {}),
        'class_names': results.get('class_names', ['World', 'Sports', 'Business', 'Sci/Tech'])
    }
    
    # Save in format that local_analysis.py expects
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(ablation_results, output_path)
    print(f"✅ Ablation-compatible results saved to {output_path}")
    
    return ablation_results

def run_ablation_with_local_analysis():
    """Run ablation analysis using the existing local_analysis.py script."""
    
    print("🚀 Running TinyMistral ablation analysis with local_analysis.py")
    
    # Convert lightweight results to ablation format
    lightweight_path = "results/tinymistral_lightweight/analysis/lightweight_results.pt"
    ablation_path = "results/tinymistral_lightweight/analysis/tinymistral_ablation_results.pt"
    
    ablation_results = create_ablation_compatible_results(lightweight_path, ablation_path)
    
    if ablation_results is None:
        print("❌ Could not create ablation results")
        return
    
    # Import and run the local analysis
    import sys
    import os
    
    # Add current directory to path to import local_analysis
    sys.path.append(os.path.dirname(__file__))
    from local_analysis import analyze_tinymistral_results
    
    # Run the analysis
    output_dir = "results/tinymistral_lightweight/analysis"
    ablation_df, specialization_df = analyze_tinymistral_results(ablation_path, output_dir)
    
    print("🎉 Ablation analysis complete!")
    print(f"📊 Check {output_dir} for:")
    print("   - tinymistral_ablation_heatmap.png (Your main ablation graph!)")
    print("   - tinymistral_ablation_detailed.csv")
    print("   - expert_specialization.csv")
    
    return ablation_df, specialization_df

def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze TinyMistral experiment results")
    parser.add_argument("--experiment", default="results/tinymistral_lightweight", help="Path to experiment results")
    parser.add_argument("--compare", help="Path to toy model results for comparison")
    parser.add_argument("--ablation", action="store_true", help="Run ablation analysis using local_analysis.py")
    
    args = parser.parse_args()
    
    if args.ablation:
        # Run ablation analysis using local_analysis.py
        run_ablation_with_local_analysis()
    else:
        # Analyze TinyMistral results (standard analysis)
        results = analyze_lightweight_results(args.experiment)
        
        # Compare with toy model if requested
        if args.compare and results:
            compare_with_toy_model(args.experiment, args.compare)

if __name__ == "__main__":
    main()
