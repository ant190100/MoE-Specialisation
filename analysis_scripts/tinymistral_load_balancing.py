#!/usr/bin/env python3
"""
TinyMistral Load Balancing Analysis
Analyzes token distribution to experts across layers to understand load balancing.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
import sys
import os
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.tinymistral_classifier import TinyMistralForClassification, load_tinymistral_tokenizer
from src.data.loaders import create_data_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TinyMistralLoadBalancingAnalyzer:
    """Analyzes expert load balancing in TinyMistral MoE layers."""
    
    def __init__(self, model_path=None, config_path="configs/tinymistral_training.yaml"):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.tokenizer = None
        self.class_names = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """Load trained TinyMistral model."""
        logger.info("Loading TinyMistral model...")
        
        # Load tokenizer
        self.tokenizer = load_tinymistral_tokenizer("M4-ai/TinyMistral-6x248M")
        
        # Create data loaders to get class names
        _, _, self.class_names = create_data_loaders(
            dataset_name="ag_news",
            batch_size=16,
            tokenizer=self.tokenizer,
        )
        
        # Initialize model
        self.model = TinyMistralForClassification(
            model_name="M4-ai/TinyMistral-6x248M", 
            num_classes=len(self.class_names)
        ).to(self.device)
        
        # Load trained weights if provided
        if self.model_path:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint['model_state_dict']
            
            # Fix key mismatch if needed
            if any(key.startswith("base_model.") for key in state_dict.keys()):
                state_dict = {
                    key.replace("base_model.", "model.") if key.startswith("base_model.") else key: value
                    for key, value in state_dict.items()
                }
            
            self.model.load_state_dict(state_dict)
            self.class_names = checkpoint.get('class_names', self.class_names)
            logger.info(f"Loaded trained model with {checkpoint.get('accuracy', 0):.2f}% accuracy")
        
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = len(self.model.model.model.layers)
        first_moe = self.model.model.model.layers[0].block_sparse_moe
        self.num_experts = first_moe.num_experts
        self.top_k = first_moe.top_k
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.num_experts} experts per layer, top-{self.top_k}")
        
    def extract_expert_assignments(self, input_ids, attention_mask):
        """Extract expert assignments from all MoE layers."""
        expert_assignments = []
        
        def routing_hook(layer_idx):
            def hook(module, input, output):
                # For TinyMistral, the gate module outputs router logits
                router_logits = output  # Router logits from gate
                if isinstance(router_logits, tuple):
                    router_logits = router_logits[0]
                
                # Get top-k experts for each token
                top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
                top_k_probs = torch.softmax(top_k_weights, dim=-1)
                
                expert_assignments.append({
                    'layer': layer_idx,
                    'top_k_indices': top_k_indices.detach().cpu(),
                    'top_k_probs': top_k_probs.detach().cpu(),
                    'router_probs': torch.softmax(router_logits, dim=-1).detach().cpu()
                })
            return hook
        
        # Register hooks
        hooks = []
        for i, layer in enumerate(self.model.model.model.layers):
            if hasattr(layer, 'block_sparse_moe'):
                hook = layer.block_sparse_moe.gate.register_forward_hook(routing_hook(i))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids, attention_mask)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return expert_assignments
            
        return expert_assignments
    
    def analyze_load_balancing(self, num_batches=10, batch_size=16, max_length=128):
        """Analyze expert load balancing across multiple batches."""
        logger.info(f"Analyzing load balancing across {num_batches} batches...")
        
        # Load validation data
        _, val_loader, _ = create_data_loaders(
            dataset_name="ag_news",
            batch_size=batch_size,
            tokenizer=self.tokenizer,
        )
        
        # Collect expert usage statistics
        expert_usage = defaultdict(lambda: defaultdict(int))  # expert_usage[layer][expert_id] = count
        expert_load_per_category = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # [layer][expert][category]
        total_tokens = 0
        total_tokens_per_layer = defaultdict(int)
        
        batch_results = []
        
        val_iter = iter(val_loader)
        
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            try:
                batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)  # Reset iterator
                batch = next(val_iter)
            
            input_ids = batch["input_ids"][:batch_size, :max_length].to(self.device)
            attention_mask = batch["attention_mask"][:batch_size, :max_length].to(self.device)
            labels = batch["labels"][:batch_size].to(self.device)
            
            # Extract expert assignments
            assignments = self.extract_expert_assignments(input_ids, attention_mask)
            
            # Process each layer's assignments
            for assign_info in assignments:
                layer_idx = assign_info['layer']
                top_k_indices = assign_info['top_k_indices']
                top_k_probs = assign_info['top_k_probs']
                router_probs = assign_info['router_probs']
                
                # Handle different tensor shapes
                if top_k_indices.dim() == 2:  # (seq_len, top_k) - single sequence
                    top_k_indices = top_k_indices.unsqueeze(0)  # Make it (1, seq_len, top_k)
                    top_k_probs = top_k_probs.unsqueeze(0)
                    router_probs = router_probs.unsqueeze(0)
                
                batch_tokens = 0
                batch_size_actual = min(top_k_indices.shape[0], input_ids.shape[0])
                
                for b_idx in range(batch_size_actual):
                    category = self.class_names[labels[b_idx].item()]
                    
                    seq_len = min(top_k_indices.shape[1], attention_mask.shape[1])
                    for t_idx in range(seq_len):
                        if attention_mask[b_idx, t_idx].item() == 1:  # Valid token
                            batch_tokens += 1
                            total_tokens += 1
                            total_tokens_per_layer[layer_idx] += 1
                            
                            # Count expert usage (top-k experts get weighted by their probability)
                            for k in range(min(self.top_k, top_k_indices.shape[2])):
                                expert_idx = top_k_indices[b_idx, t_idx, k].item()
                                prob_weight = top_k_probs[b_idx, t_idx, k].item()
                                
                                expert_usage[layer_idx][expert_idx] += prob_weight
                                expert_load_per_category[layer_idx][expert_idx][category] += prob_weight
                
                # Store batch-level statistics
                batch_results.append({
                    'batch_idx': batch_idx,
                    'layer': layer_idx,
                    'tokens': batch_tokens,
                    'expert_usage': dict(expert_usage[layer_idx]),
                    'router_probs_mean': router_probs.mean(dim=(0, 1)).numpy(),  # Average prob per expert
                    'router_probs_std': router_probs.std(dim=(0, 1)).numpy(),
                })
        
        # Convert to DataFrames
        usage_data = []
        for layer_idx in range(self.num_layers):
            layer_total = total_tokens_per_layer[layer_idx]
            
            for expert_idx in range(self.num_experts):
                usage_count = expert_usage[layer_idx][expert_idx]
                usage_percentage = (usage_count / layer_total * 100) if layer_total > 0 else 0
                
                # Category breakdown
                category_breakdown = {}
                for category in self.class_names:
                    cat_usage = expert_load_per_category[layer_idx][expert_idx][category]
                    category_breakdown[category] = cat_usage
                
                usage_data.append({
                    'layer': layer_idx,
                    'expert': expert_idx,
                    'usage_count': usage_count,
                    'usage_percentage': usage_percentage,
                    'total_layer_tokens': layer_total,
                    **{f'{cat}_usage': category_breakdown[cat] for cat in self.class_names}
                })
        
        usage_df = pd.DataFrame(usage_data)
        
        logger.info(f"Processed {total_tokens:,} tokens across {num_batches} batches")
        
        return usage_df, batch_results
    
    def visualize_load_balancing(self, usage_df, output_dir="results/tinymistral_load_balancing"):
        """Create load balancing visualizations - simplified to show only load distribution per layer."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating load distribution visualization...")
        
        # Load Distribution per Layer
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for layer_idx in range(self.num_layers):
            layer_data = usage_df[usage_df['layer'] == layer_idx]
            
            ax = axes[layer_idx]
            bars = ax.bar(layer_data['expert'], layer_data['usage_percentage'])
            ax.set_title(f'Layer {layer_idx} Load Distribution', fontweight='bold')
            ax.set_xlabel('Expert Index')
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(0, max(20, usage_df['usage_percentage'].max() * 1.1))
            
            # Color bars by usage level
            max_usage = layer_data['usage_percentage'].max()
            for bar, usage in zip(bars, layer_data['usage_percentage']):
                normalized_usage = usage / max_usage if max_usage > 0 else 0
                bar.set_color(plt.cm.YlOrRd(normalized_usage))
            
            # Add ideal line (perfect balance)
            ideal_usage = 100 / self.num_experts
            ax.axhline(y=ideal_usage, color='red', linestyle='--', alpha=0.7, 
                      label=f'Perfect Balance ({ideal_usage:.1f}%)')
            ax.legend()
        
        plt.tight_layout()
        
        load_distribution_path = output_path / "load_distribution_per_layer.png"
        plt.savefig(load_distribution_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save basic data
        usage_df.to_csv(output_path / "expert_usage_detailed.csv", index=False)
        
        # Calculate and display basic summary statistics
        logger.info("\n📊 **LOAD BALANCING ANALYSIS SUMMARY**")
        logger.info(f"Total experts analyzed: {self.num_layers * self.num_experts}")
        logger.info(f"Ideal usage per expert: {100 / self.num_experts:.1f}%")
        
        logger.info("\n⚖️ Balance Quality by Layer (Coefficient of Variation):")
        for layer_idx in range(self.num_layers):
            layer_data = usage_df[usage_df['layer'] == layer_idx]['usage_percentage']
            cv = layer_data.std() / layer_data.mean() if layer_data.mean() > 0 else float('inf')
            quality = "Good" if cv < 0.5 else "Fair" if cv < 1.0 else "Poor"
            logger.info(f"   Layer {layer_idx:2d}: CV={cv:.3f} ({quality})")
        
        logger.info("\n🎯 Most/Least Used Experts:")
        overall_usage = usage_df.groupby(['layer', 'expert'])['usage_percentage'].first().reset_index()
        most_used = overall_usage.loc[overall_usage['usage_percentage'].idxmax()]
        least_used = overall_usage.loc[overall_usage['usage_percentage'].idxmin()]
        
        logger.info(f"   Most used: Layer {int(most_used['layer'])}-Expert {int(most_used['expert'])} ({most_used['usage_percentage']:.1f}%)")
        logger.info(f"   Least used: Layer {int(least_used['layer'])}-Expert {int(least_used['expert'])} ({least_used['usage_percentage']:.1f}%)")
        
        logger.info(f"\n✅ Visualization saved to: {output_path}")
        
        return output_path


def main():
    """Main load balancing analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyMistral Load Balancing Analysis")
    parser.add_argument("--model-path", help="Path to trained model checkpoint")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches to analyze")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for analysis")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--output-dir", default="results/tinymistral_load_balancing", help="Output directory")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting TinyMistral Load Balancing Analysis")
    
    # Initialize analyzer
    analyzer = TinyMistralLoadBalancingAnalyzer(
        model_path=args.model_path,
    )
    
    # Load model
    analyzer.load_model()
    
    # Analyze load balancing
    usage_df, batch_results = analyzer.analyze_load_balancing(
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Create visualizations
    output_path = analyzer.visualize_load_balancing(usage_df, args.output_dir)
    
    logger.info("🎉 Load balancing analysis completed!")
    logger.info(f"📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
