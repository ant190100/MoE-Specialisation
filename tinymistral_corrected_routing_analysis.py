#!/usr/bin/env python3
"""
Corrected TinyMistral Routing Analysis - Fixed for top-k=2 routing
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
from pathlib import Path
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TinyMistralRoutingAnalyzer:
    """Analyze routing patterns in TinyMistral with proper top-k=2 handling."""
    
    def __init__(self, model_name="M4-ai/TinyMistral-6x248M"):
        """Initialize the analyzer."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.hooks = []
        
        # Storage for routing data
        self.routing_data = defaultdict(list)
        
    def load_model(self):
        """Load the TinyMistral model and tokenizer."""
        logger.info(f"🔄 Loading TinyMistral model: {self.model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get model info
        self.num_layers = len(self.model.model.layers)
        first_moe = self.model.model.layers[0].block_sparse_moe
        self.num_experts = first_moe.num_experts
        self.top_k = first_moe.top_k
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"📊 Layers: {self.num_layers}, Experts per layer: {self.num_experts}, Top-k: {self.top_k}")
        
    def register_routing_hooks(self):
        """Register hooks to capture routing data with proper top-k handling."""
        
        def create_hook(layer_idx):
            def routing_hook(module, input, output):
                if hasattr(module, 'gate'):
                    # Get input hidden states
                    hidden_states = input[0]
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    
                    # Flatten for routing
                    hidden_flat = hidden_states.view(-1, hidden_dim)
                    
                    # Get routing logits and probabilities
                    routing_logits = module.gate(hidden_flat)
                    routing_probs = F.softmax(routing_logits, dim=-1)
                    
                    # Get top-k selections (what actually gets routed)
                    top_k_weights, top_k_indices = torch.topk(routing_probs, k=self.top_k, dim=-1)
                    
                    # Store comprehensive routing data
                    self.routing_data[f"layer_{layer_idx}"].append({
                        'routing_probs': routing_probs.detach().cpu(),  # All probabilities
                        'top_k_weights': top_k_weights.detach().cpu(),  # Weights of selected experts
                        'top_k_indices': top_k_indices.detach().cpu(),  # Indices of selected experts
                        'batch_size': batch_size,
                        'seq_len': seq_len
                    })
                    
            return routing_hook
        
        # Register hooks on all MoE layers
        self.hooks = []
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'block_sparse_moe'):
                hook = layer.block_sparse_moe.register_forward_hook(create_hook(i))
                self.hooks.append(hook)
                
        logger.info(f"✅ Registered hooks on {len(self.hooks)} MoE layers")
        
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("🧹 Removed all hooks")
        
    def analyze_texts(self, texts, max_length=128):
        """Analyze routing patterns for given texts."""
        logger.info(f"🔍 Analyzing {len(texts)} texts...")
        
        # Clear previous data
        self.routing_data.clear()
        
        # Tokenize
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        )
        
        logger.info(f"📝 Input shape: {inputs['input_ids'].shape}")
        
        # Register hooks
        self.register_routing_hooks()
        
        try:
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            logger.info("✅ Forward pass completed")
            
        finally:
            # Always remove hooks
            self.remove_hooks()
            
        return self._compute_routing_statistics()
    
    def _compute_routing_statistics(self):
        """Compute comprehensive routing statistics."""
        logger.info("📊 Computing routing statistics...")
        
        stats = {}
        
        for layer_name, layer_data in self.routing_data.items():
            layer_stats = {
                'expert_utilization': np.zeros(self.num_experts),  # Count of times each expert was selected
                'expert_load': np.zeros(self.num_experts),         # Total weight each expert received
                'routing_entropy': [],                             # Entropy for each token
                'total_tokens': 0
            }
            
            # Process all batches for this layer
            for batch_data in layer_data:
                routing_probs = batch_data['routing_probs']      # [num_tokens, num_experts]
                top_k_indices = batch_data['top_k_indices']      # [num_tokens, top_k]
                top_k_weights = batch_data['top_k_weights']      # [num_tokens, top_k]
                
                num_tokens = routing_probs.shape[0]
                layer_stats['total_tokens'] += num_tokens
                
                # Compute routing entropy for each token
                for token_idx in range(num_tokens):
                    probs = routing_probs[token_idx].numpy()
                    # Add small epsilon to avoid log(0)
                    entropy = -np.sum(probs * np.log(probs + 1e-8))
                    layer_stats['routing_entropy'].append(entropy)
                
                # Count expert utilization (how many times each expert was selected)
                for token_idx in range(num_tokens):
                    selected_experts = top_k_indices[token_idx].numpy()
                    selected_weights = top_k_weights[token_idx].numpy()
                    
                    for expert_idx, weight in zip(selected_experts, selected_weights):
                        layer_stats['expert_utilization'][expert_idx] += 1
                        layer_stats['expert_load'][expert_idx] += weight
            
            # Convert to percentages and averages
            total_selections = np.sum(layer_stats['expert_utilization'])
            layer_stats['expert_utilization_pct'] = (layer_stats['expert_utilization'] / total_selections) * 100
            layer_stats['average_entropy'] = np.mean(layer_stats['routing_entropy'])
            
            stats[layer_name] = layer_stats
            
        logger.info("✅ Statistics computed")
        return stats
    
    def create_visualizations(self, stats, output_dir="results/tinymistral_corrected_routing"):
        """Create comprehensive visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Creating visualizations in {output_path}")
        
        # 1. Expert Utilization Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("TinyMistral Routing Analysis (Corrected for Top-k=2)", fontsize=16)
        
        # Plot utilization for first 4 layers
        layers_to_plot = [f"layer_{i}" for i in range(min(4, len(stats)))]
        
        for idx, layer_name in enumerate(layers_to_plot):
            ax = axes[idx // 2, idx % 2]
            layer_stats = stats[layer_name]
            
            expert_ids = range(self.num_experts)
            utilization = layer_stats['expert_utilization_pct']
            
            bars = ax.bar(expert_ids, utilization, alpha=0.7, 
                         color=plt.cm.tab10(np.arange(self.num_experts)))
            
            ax.set_title(f"{layer_name.replace('_', ' ').title()} - Expert Utilization")
            ax.set_xlabel("Expert ID")
            ax.set_ylabel("Utilization (%)")
            ax.set_ylim(0, max(utilization) * 1.1)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, utilization):
                if pct > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path / "expert_utilization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Routing Entropy by Layer
        fig, ax = plt.subplots(figsize=(12, 6))
        
        layer_names = []
        avg_entropies = []
        
        for layer_name in sorted(stats.keys(), key=lambda x: int(x.split('_')[1])):
            layer_names.append(layer_name.replace('layer_', 'L'))
            avg_entropies.append(stats[layer_name]['average_entropy'])
        
        bars = ax.bar(layer_names, avg_entropies, alpha=0.7, color='steelblue')
        ax.set_title("Average Routing Entropy by Layer", fontsize=14)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Average Entropy")
        ax.set_ylim(0, max(avg_entropies) * 1.1)
        
        # Add value labels
        for bar, entropy in zip(bars, avg_entropies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{entropy:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / "routing_entropy.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Expert Load Distribution Heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create matrix of expert utilization percentages
        utilization_matrix = []
        layer_labels = []
        
        for layer_name in sorted(stats.keys(), key=lambda x: int(x.split('_')[1])):
            utilization_matrix.append(stats[layer_name]['expert_utilization_pct'])
            layer_labels.append(layer_name.replace('layer_', 'Layer '))
        
        utilization_matrix = np.array(utilization_matrix)
        
        sns.heatmap(utilization_matrix, 
                   xticklabels=[f'Expert {i}' for i in range(self.num_experts)],
                   yticklabels=layer_labels,
                   annot=True, fmt='.1f', cmap='viridis',
                   cbar_kws={'label': 'Utilization %'})
        
        ax.set_title("Expert Utilization Across All Layers (%)", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path / "expert_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Visualizations saved")
        
    def save_detailed_report(self, stats, output_dir="results/tinymistral_corrected_routing"):
        """Save detailed routing analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create detailed report
        report = {
            'model_info': {
                'model_name': self.model_name,
                'num_layers': self.num_layers,
                'num_experts': self.num_experts,
                'top_k': self.top_k
            },
            'analysis_summary': {},
            'layer_details': {}
        }
        
        # Overall statistics
        total_selections = sum(np.sum(stats[layer]['expert_utilization']) 
                              for layer in stats)
        all_entropies = []
        for layer_stats in stats.values():
            all_entropies.extend(layer_stats['routing_entropy'])
        
        report['analysis_summary'] = {
            'total_routing_decisions': int(total_selections),
            'average_entropy_across_all_tokens': float(np.mean(all_entropies)),
            'entropy_std': float(np.std(all_entropies)),
            'note': f'Each token is routed to exactly {self.top_k} experts (top-k routing)'
        }
        
        # Layer details
        for layer_name, layer_stats in stats.items():
            report['layer_details'][layer_name] = {
                'total_tokens': int(layer_stats['total_tokens']),
                'total_selections': int(np.sum(layer_stats['expert_utilization'])),
                'average_entropy': float(layer_stats['average_entropy']),
                'expert_utilization_counts': layer_stats['expert_utilization'].tolist(),
                'expert_utilization_percentages': layer_stats['expert_utilization_pct'].tolist(),
                'expert_average_load': (layer_stats['expert_load'] / np.maximum(layer_stats['expert_utilization'], 1)).tolist()
            }
        
        # Save report
        report_path = output_path / "routing_analysis_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save human-readable summary
        summary_path = output_path / "routing_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("TinyMistral Routing Analysis Summary (CORRECTED)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Architecture: {self.num_layers} layers, {self.num_experts} experts per layer\n")
            f.write(f"Routing: Top-{self.top_k} experts per token\n\n")
            
            f.write("Key Findings:\n")
            f.write(f"- Total routing decisions: {total_selections:,}\n")
            f.write(f"- Average routing entropy: {np.mean(all_entropies):.4f} ± {np.std(all_entropies):.4f}\n")
            f.write(f"- Each token is routed to exactly {self.top_k} experts\n\n")
            
            f.write("Expert Utilization by Layer:\n")
            f.write("-" * 30 + "\n")
            
            for layer_name in sorted(stats.keys(), key=lambda x: int(x.split('_')[1])):
                layer_stats = stats[layer_name]
                f.write(f"\n{layer_name.replace('_', ' ').title()}:\n")
                
                for expert_id in range(self.num_experts):
                    count = int(layer_stats['expert_utilization'][expert_id])
                    pct = layer_stats['expert_utilization_pct'][expert_id]
                    f.write(f"  Expert {expert_id}: {count:,} selections ({pct:.1f}%)\n")
        
        logger.info(f"✅ Detailed report saved to {output_path}")


def main():
    """Main analysis function."""
    logger.info("🚀 Starting TinyMistral Routing Analysis (CORRECTED)")
    
    # Initialize analyzer
    analyzer = TinyMistralRoutingAnalyzer()
    analyzer.load_model()
    
    # Load test data
    logger.info("📚 Loading test data...")
    dataset = load_dataset("ag_news", split="test")
    test_texts = dataset["text"][:20]  # Analyze 20 samples
    
    # Run analysis
    stats = analyzer.analyze_texts(test_texts)
    
    # Create visualizations and report
    analyzer.create_visualizations(stats)
    analyzer.save_detailed_report(stats)
    
    logger.info("🎉 Analysis completed successfully!")
    logger.info("📁 Results saved in results/tinymistral_corrected_routing/")

if __name__ == "__main__":
    main()
