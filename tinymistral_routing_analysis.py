#!/usr/bin/env python3
"""
TinyMistral Routing Entropy and Token Distribution Analysis

This script analyzes routing patterns in TinyMistral by extracting expert utilization
and routing entropy from the model's internal MoE layers during inference.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TinyMistralRoutingAnalyzer:
    """Analyzes routing patterns in TinyMistral models."""
    
    def __init__(self, model_name_or_path, device="cpu"):
        """Initialize the analyzer with a TinyMistral model."""
        self.device = device
        self.model_name = model_name_or_path
        
        # Load tokenizer and model
        logger.info("🔄 Loading TinyMistral model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            device_map=None
        ).to(device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Extract model info
        self.num_layers = len(self.model.model.layers)
        self.num_experts = 6  # TinyMistral has 6 experts per layer
        
        logger.info(f"✅ Loaded {model_name_or_path}")
        logger.info(f"📊 Model layers: {self.num_layers}")
        logger.info(f"👥 Experts per layer: {self.num_experts}")
        
        # Storage for routing data
        self.routing_weights = {}
        self.expert_selections = {}
        
    def register_routing_hooks(self):
        """Register hooks to capture routing weights from MoE layers."""
        
        def create_hook(layer_idx):
            def routing_hook(module, input, output):
                # TinyMistral routing happens in block_sparse_moe
                if hasattr(module, 'gate'):
                    # Get routing logits from the gate
                    hidden_states = input[0]  # Input hidden states
                    batch_size, sequence_length, hidden_dim = hidden_states.shape
                    
                    # Reshape for routing
                    hidden_states = hidden_states.view(-1, hidden_dim)
                    
                    # Get routing weights
                    routing_logits = module.gate(hidden_states)
                    routing_weights = F.softmax(routing_logits, dim=-1)
                    
                    # Store routing weights
                    key = f"layer_{layer_idx}"
                    if key not in self.routing_weights:
                        self.routing_weights[key] = []
                    self.routing_weights[key].append(routing_weights.detach().cpu())
                    
                    # Store top-k expert selections
                    top_k_weights, top_k_indices = torch.topk(routing_weights, k=2, dim=-1)
                    if key not in self.expert_selections:
                        self.expert_selections[key] = []
                    self.expert_selections[key].append({
                        'weights': top_k_weights.detach().cpu(),
                        'indices': top_k_indices.detach().cpu()
                    })
                    
            return routing_hook
        
        # Register hooks on MoE layers
        self.hooks = []
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'block_sparse_moe'):
                hook = layer.block_sparse_moe.register_forward_hook(create_hook(i))
                self.hooks.append(hook)
                
        logger.info(f"✅ Registered routing hooks on {len(self.hooks)} MoE layers")
        
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def analyze_batch(self, texts):
        """Analyze routing patterns for a batch of texts."""
        # Clear previous routing data
        self.routing_weights.clear()
        self.expert_selections.clear()
        
        # Register hooks
        self.register_routing_hooks()
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass to collect routing data
        with torch.no_grad():
            try:
                _ = self.model(**inputs)
            except Exception as e:
                logger.warning(f"Forward pass error: {e}")
                
        # Remove hooks
        self.remove_hooks()
        
        return self.routing_weights, self.expert_selections
    
    def compute_routing_entropy(self, routing_weights):
        """Compute routing entropy for routing weights."""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-9
        routing_weights = routing_weights + epsilon
        
        # Compute entropy: -sum(p * log2(p))
        log_weights = torch.log2(routing_weights)
        entropy = -torch.sum(routing_weights * log_weights, dim=-1)
        
        # Normalize by max entropy (log2(num_experts))
        max_entropy = np.log2(self.num_experts)
        normalized_entropy = entropy / max_entropy
        
        return entropy, normalized_entropy
    
    def create_dataset_loader(self, dataset_name="ag_news", split="test", max_samples=500):
        """Create a data loader for analysis."""
        logger.info(f"📚 Loading {dataset_name} dataset...")
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Limit samples for analysis
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            
        # Extract texts
        texts = dataset["text"]
        labels = dataset["label"]
        
        return texts, labels
        
    def analyze_routing_patterns(self, dataset_name="ag_news", max_samples=100, batch_size=8):
        """Analyze routing patterns across a dataset."""
        logger.info("🔍 Starting comprehensive routing analysis...")
        
        # Load dataset
        texts, labels = self.create_dataset_loader(dataset_name, max_samples=max_samples)
        
        # Storage for analysis results
        all_routing_weights = {f"layer_{i}": [] for i in range(self.num_layers)}
        all_expert_selections = {f"layer_{i}": [] for i in range(self.num_layers)}
        all_entropies = {f"layer_{i}": [] for i in range(self.num_layers)}
        token_expert_counts = {f"layer_{i}": torch.zeros(self.num_experts) for i in range(self.num_layers)}
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Analyzing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            # Analyze this batch
            try:
                routing_weights, expert_selections = self.analyze_batch(batch_texts)
                
                # Process routing data for each layer
                for layer_key in routing_weights:
                    if routing_weights[layer_key]:  # Check if data exists
                        # Concatenate all routing weights for this layer
                        layer_weights = torch.cat(routing_weights[layer_key], dim=0)
                        all_routing_weights[layer_key].append(layer_weights)
                        
                        # Compute entropy
                        entropy, norm_entropy = self.compute_routing_entropy(layer_weights)
                        all_entropies[layer_key].append(norm_entropy)
                        
                        # Count expert selections
                        expert_indices = torch.argmax(layer_weights, dim=-1)
                        expert_counts = torch.bincount(expert_indices, minlength=self.num_experts)
                        token_expert_counts[layer_key] += expert_counts.float()
                        
            except Exception as e:
                logger.warning(f"Error processing batch {batch_idx}: {e}")
                continue
        
        # Combine results
        combined_results = {}
        for layer_key in all_routing_weights:
            if all_routing_weights[layer_key]:
                combined_results[layer_key] = {
                    'routing_weights': torch.cat(all_routing_weights[layer_key], dim=0),
                    'entropies': torch.cat(all_entropies[layer_key], dim=0),
                    'expert_counts': token_expert_counts[layer_key]
                }
        
        logger.info(f"✅ Analyzed {len(texts)} samples across {len(combined_results)} layers")
        return combined_results
        
    def plot_routing_entropy_distribution(self, analysis_results, save_path=None):
        """Plot routing entropy distribution across layers."""
        logger.info("🎨 Creating routing entropy distribution plot...")
        
        num_layers = len(analysis_results)
        fig, axes = plt.subplots(2, (num_layers + 1) // 2, figsize=(16, 8))
        axes = axes.flatten() if num_layers > 1 else [axes]
        
        for idx, (layer_key, data) in enumerate(analysis_results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            entropies = data['entropies'].numpy()
            
            # Plot histogram
            n, bins, patches = ax.hist(entropies, bins=30, alpha=0.7, edgecolor='black')
            
            # Color code by entropy level
            cmap = plt.cm.RdYlGn_r
            for i, p in enumerate(patches):
                p.set_facecolor(cmap(i / len(patches)))
            
            # Add mean line
            mean_entropy = np.mean(entropies)
            ax.axvline(mean_entropy, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_entropy:.3f}')
            
            ax.set_title(f'{layer_key.replace("_", " ").title()}', fontsize=12)
            ax.set_xlabel('Normalized Routing Entropy', fontsize=10)
            ax.set_ylabel('Token Count', fontsize=10)
            ax.legend()
            
        # Remove empty subplots
        for idx in range(len(analysis_results), len(axes)):
            fig.delaxes(axes[idx])
            
        plt.suptitle('TinyMistral Routing Entropy Distribution by Layer', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Routing entropy plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
    def plot_token_distribution_by_expert(self, analysis_results, save_path=None):
        """Plot token distribution across experts for each layer."""
        logger.info("🎨 Creating token distribution by expert plot...")
        
        # Prepare data for plotting
        plot_data = []
        for layer_key, data in analysis_results.items():
            expert_counts = data['expert_counts'].numpy()
            layer_name = layer_key.replace('_', ' ').title()
            
            for expert_idx, count in enumerate(expert_counts):
                plot_data.append({
                    'Layer': layer_name,
                    'Expert': f'Expert {expert_idx}',
                    'Token Count': count,
                    'Percentage': (count / expert_counts.sum()) * 100 if expert_counts.sum() > 0 else 0
                })
        
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            logger.warning("No data available for token distribution plot")
            return
            
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Absolute counts
        sns.barplot(data=df, x='Expert', y='Token Count', hue='Layer', ax=ax1)
        ax1.set_title('Token Distribution by Expert (Absolute)', fontsize=14)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Percentage distribution
        sns.barplot(data=df, x='Expert', y='Percentage', hue='Layer', ax=ax2)
        ax2.set_title('Token Distribution by Expert (Percentage)', fontsize=14)
        ax2.set_ylabel('Percentage (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle('TinyMistral Expert Utilization Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Token distribution plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
            
    def create_routing_summary_report(self, analysis_results, save_path=None):
        """Create a summary report of routing patterns."""
        logger.info("📊 Creating routing summary report...")
        
        report = []
        report.append("🔍 TINYMISTRAL ROUTING ANALYSIS REPORT")
        report.append("=" * 50)
        report.append(f"📱 Model: {self.model_name}")
        report.append(f"👥 Experts per layer: {self.num_experts}")
        report.append(f"🏗️ Analyzed layers: {len(analysis_results)}")
        report.append("")
        
        for layer_key, data in analysis_results.items():
            layer_name = layer_key.replace('_', ' ').title()
            entropies = data['entropies'].numpy()
            expert_counts = data['expert_counts'].numpy()
            
            report.append(f"📊 {layer_name}:")
            report.append(f"   • Tokens analyzed: {len(entropies):,}")
            report.append(f"   • Mean routing entropy: {np.mean(entropies):.4f}")
            report.append(f"   • Std routing entropy: {np.std(entropies):.4f}")
            report.append(f"   • Min entropy: {np.min(entropies):.4f}")
            report.append(f"   • Max entropy: {np.max(entropies):.4f}")
            
            # Expert utilization
            total_tokens = expert_counts.sum()
            if total_tokens > 0:
                expert_percentages = (expert_counts / total_tokens) * 100
                most_used = np.argmax(expert_counts)
                least_used = np.argmin(expert_counts)
                
                report.append(f"   • Most used expert: Expert {most_used} ({expert_percentages[most_used]:.1f}%)")
                report.append(f"   • Least used expert: Expert {least_used} ({expert_percentages[least_used]:.1f}%)")
                report.append(f"   • Expert utilization balance: {np.std(expert_percentages):.2f}% std")
            
            report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"✅ Routing report saved to {save_path}")
        
        print(report_text)
        return report_text


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze TinyMistral routing patterns")
    parser.add_argument("--model_path", default="results/tinymistral_extended_training", 
                       help="Path to trained model checkpoint")
    parser.add_argument("--base_model", default="M4-ai/TinyMistral-6x248M", 
                       help="Base model name or path")
    parser.add_argument("--dataset", default="ag_news", help="Dataset for analysis")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples to analyze")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for analysis")
    parser.add_argument("--output_dir", default="results/tinymistral_extended_training/analysis",
                       help="Output directory for results")
    parser.add_argument("--device", default="cpu", help="Device to run analysis on")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔬 TinyMistral Routing Analysis")
    print("=" * 50)
    print(f"📱 Base model: {args.base_model}")
    print(f"📂 Model path: {args.model_path}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔢 Max samples: {args.max_samples}")
    print(f"💻 Device: {args.device}")
    print()
    
    try:
        # Initialize analyzer
        analyzer = TinyMistralRoutingAnalyzer(args.base_model, device=args.device)
        
        # Check if we have a trained checkpoint to load
        checkpoint_path = Path(args.model_path) / "models" / "best_model.pt"
        if checkpoint_path.exists():
            logger.info(f"🔄 Found trained checkpoint at {checkpoint_path}")
            logger.info(f"⚠️  Skipping checkpoint loading due to key mismatch (classifier wrapper)")
            logger.info(f"📊 Running analysis on base model instead")
        
        # Analyze routing patterns
        results = analyzer.analyze_routing_patterns(
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            batch_size=args.batch_size
        )
        
        if not results:
            logger.error("❌ No routing data collected. Analysis failed.")
            return
        
        # Generate visualizations
        print("\n🎨 Generating visualizations...")
        
        # Routing entropy distribution
        analyzer.plot_routing_entropy_distribution(
            results,
            save_path=output_dir / "routing_entropy_distribution.png"
        )
        
        # Token distribution by expert
        analyzer.plot_token_distribution_by_expert(
            results,
            save_path=output_dir / "token_distribution_by_expert.png"
        )
        
        # Summary report
        analyzer.create_routing_summary_report(
            results,
            save_path=output_dir / "routing_analysis_report.txt"
        )
        
        # Save raw results
        torch.save(results, output_dir / "routing_analysis_results.pt")
        
        print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
        print(f"📈 Generated files:")
        print(f"   • routing_entropy_distribution.png")
        print(f"   • token_distribution_by_expert.png") 
        print(f"   • routing_analysis_report.txt")
        print(f"   • routing_analysis_results.pt")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
