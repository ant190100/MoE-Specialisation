#!/usr/bin/env python3
"""
TinyMistral Routing Entropy Analysis
Visualizes routing entropy across layers to understand expert selection patterns.
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

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.models.tinymistral_classifier import TinyMistralForClassification, load_tinymistral_tokenizer
from src.data.loaders import create_data_loaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TinyMistralRoutingEntropyAnalyzer:
    """Analyzes routing entropy patterns in TinyMistral MoE layers."""
    
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
        
        logger.info(f"Model loaded: {self.num_layers} layers, {self.num_experts} experts per layer")
        
    def extract_routing_weights(self, input_ids, attention_mask):
        """Extract routing weights from all MoE layers."""
        routing_weights = []
        
        def routing_hook(layer_idx):
            def hook(module, input, output):
                # For TinyMistral, the gate module outputs router logits
                # We need to capture the router logits from the gate forward pass
                if hasattr(module, 'weight'):  # This is the gate/router linear layer
                    # Get the input to the gate (which are the router logits)
                    router_logits = output  # The output of the gate is the router logits
                    if isinstance(router_logits, tuple):
                        router_logits = router_logits[0]
                    
                    routing_weights.append({
                        'layer': layer_idx,
                        'logits': router_logits.detach().cpu(),
                        'probs': torch.softmax(router_logits, dim=-1).detach().cpu()
                    })
            return hook
        
        # Register hooks on the gate modules
        hooks = []
        for i, layer in enumerate(self.model.model.model.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                hook = layer.block_sparse_moe.gate.register_forward_hook(routing_hook(i))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return routing_weights
    
    def calculate_entropy(self, probs):
        """Calculate Shannon entropy for routing probabilities."""
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        probs = probs + eps
        
        # Ensure we have the right dimensions (batch_size, seq_len, num_experts)
        if probs.dim() == 2:  # If only (seq_len, num_experts), add batch dim
            probs = probs.unsqueeze(0)
        
        # Calculate Shannon entropy: H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return entropy
    
    def analyze_batch_entropy(self, batch_size=8, max_length=128):
        """Analyze routing entropy for a batch of samples."""
        logger.info(f"Analyzing routing entropy for batch of {batch_size} samples...")
        
        # Load validation data
        _, val_loader, _ = create_data_loaders(
            dataset_name="ag_news",
            batch_size=batch_size,
            tokenizer=self.tokenizer,
        )
        
        # Get one batch
        batch = next(iter(val_loader))
        input_ids = batch["input_ids"][:batch_size, :max_length].to(self.device)
        attention_mask = batch["attention_mask"][:batch_size, :max_length].to(self.device)
        labels = batch["labels"][:batch_size].to(self.device)
        
        # Extract routing weights
        routing_data = self.extract_routing_weights(input_ids, attention_mask)
        
        # Calculate entropy for each layer
        entropy_results = []
        
        if not routing_data:
            print("Warning: No routing data captured")
            return pd.DataFrame(), input_ids, labels
        
        print(f"Processing {len(routing_data)} layers")
        
        for route_info in routing_data:
            layer_idx = route_info['layer']
            probs = route_info['probs']  
            
            print(f"Layer {layer_idx} - probs shape: {probs.shape}")
            
            # Calculate entropy for each token
            entropy = self.calculate_entropy(probs)  
            print(f"Layer {layer_idx} - entropy shape: {entropy.shape}")
            
            # Handle different tensor shapes
            if entropy.dim() == 1:  # (seq_len,) - single sequence
                batch_size_actual = 1
                seq_len = entropy.shape[0]
                entropy = entropy.unsqueeze(0)  # Make it (1, seq_len)
            elif entropy.dim() == 2:  # (batch_size, seq_len)
                batch_size_actual, seq_len = entropy.shape
            else:
                print(f"Unexpected entropy shape: {entropy.shape}, skipping layer {layer_idx}")
                continue
            
            # Ensure probs matches entropy dimensions
            if probs.dim() == 2:  # (seq_len, num_experts) - single sequence
                probs = probs.unsqueeze(0)  # Make it (1, seq_len, num_experts)
            
            # Store results
            for batch_idx in range(min(batch_size_actual, input_ids.shape[0])):
                for token_idx in range(min(seq_len, attention_mask.shape[1])):
                    if attention_mask[batch_idx, token_idx].item() == 1:  # Valid token
                        entropy_results.append({
                            'layer': layer_idx,
                            'batch_idx': batch_idx,
                            'token_idx': token_idx,
                            'entropy': entropy[batch_idx, token_idx].item(),
                            'category': self.class_names[labels[batch_idx].item()],
                            'max_prob': probs[batch_idx, token_idx].max().item(),
                            'top_expert': probs[batch_idx, token_idx].argmax().item()
                        })
        
        return pd.DataFrame(entropy_results), input_ids, labels
    
    def visualize_entropy_patterns(self, entropy_df, output_dir="results/tinymistral_entropy", target_layer=6):
        """Create simplified entropy visualizations - entropy by layer and single layer distribution."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating entropy visualizations...")
        
        # 1. Mean Entropy by Layer (from the entropy_layer_statistics top subplot)
        layer_stats = entropy_df.groupby('layer')['entropy'].agg([
            'mean', 'std', 'min', 'max', 'median'
        ]).round(3)
        
        plt.figure(figsize=(12, 6))
        plt.plot(layer_stats.index, layer_stats['mean'], 'o-', linewidth=2, markersize=8, color='steelblue')
        plt.fill_between(layer_stats.index, 
                        layer_stats['mean'] - layer_stats['std'],
                        layer_stats['mean'] + layer_stats['std'], 
                        alpha=0.3, color='steelblue')
        plt.title('Mean Routing Entropy by Layer (±1 std)', fontweight='bold', fontsize=14)
        plt.xlabel('Layer Index')
        plt.ylabel('Entropy (bits)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        entropy_by_layer_path = output_path / "entropy_by_layer.png"
        plt.savefig(entropy_by_layer_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Single Layer Entropy Distribution
        if target_layer < self.num_layers:
            layer_data = entropy_df[entropy_df['layer'] == target_layer]
            
            plt.figure(figsize=(10, 6))
            plt.hist(layer_data['entropy'], bins=30, alpha=0.7, color='coral', edgecolor='black')
            plt.axvline(layer_data['entropy'].mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {layer_data["entropy"].mean():.3f}')
            plt.axvline(layer_data['entropy'].median(), color='blue', linestyle='--', linewidth=2, 
                       label=f'Median: {layer_data["entropy"].median():.3f}')
            plt.title(f'Layer {target_layer} Routing Entropy Distribution', fontweight='bold', fontsize=14)
            plt.xlabel('Shannon Entropy (bits)')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            layer_distribution_path = output_path / f"layer_{target_layer}_entropy_distribution.png"
            plt.savefig(layer_distribution_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            logger.warning(f"Target layer {target_layer} is out of range (0-{self.num_layers-1})")
        
        # Save basic statistics
        layer_stats.to_csv(output_path / "entropy_layer_statistics.csv")
        
        # Print summary
        logger.info("\n📊 **ENTROPY ANALYSIS SUMMARY**")
        logger.info(f"Total tokens analyzed: {len(entropy_df):,}")
        logger.info(f"Average entropy across all layers: {entropy_df['entropy'].mean():.3f} bits")
        logger.info(f"Entropy range: {entropy_df['entropy'].min():.3f} - {entropy_df['entropy'].max():.3f} bits")
        
        logger.info("\n🔄 Layer-wise Average Entropy:")
        for layer, stats in layer_stats.iterrows():
            logger.info(f"  Layer {layer:2d}: {stats['mean']:.3f} ± {stats['std']:.3f} bits")
        
        if target_layer < self.num_layers:
            target_stats = layer_stats.loc[target_layer]
            logger.info(f"\n🎯 Target Layer {target_layer} Statistics:")
            logger.info(f"  Mean: {target_stats['mean']:.3f} ± {target_stats['std']:.3f} bits")
            logger.info(f"  Range: {target_stats['min']:.3f} - {target_stats['max']:.3f} bits")
            logger.info(f"  Median: {target_stats['median']:.3f} bits")
        
        logger.info(f"\n✅ Visualizations saved to: {output_path}")
        
        return output_path


def main():
    """Main entropy analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinyMistral Routing Entropy Analysis")
    parser.add_argument("--model-path", help="Path to trained model checkpoint")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for analysis")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--target-layer", type=int, default=6, help="Target layer for detailed entropy distribution")
    parser.add_argument("--output-dir", default="results/tinymistral_entropy", help="Output directory")
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting TinyMistral Routing Entropy Analysis")
    
    # Initialize analyzer
    analyzer = TinyMistralRoutingEntropyAnalyzer(
        model_path=args.model_path,
    )
    
    # Load model
    analyzer.load_model()
    
    # Analyze entropy
    entropy_df, input_ids, labels = analyzer.analyze_batch_entropy(
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Create visualizations
    output_path = analyzer.visualize_entropy_patterns(entropy_df, args.output_dir, args.target_layer)
    
    logger.info("🎉 Routing entropy analysis completed!")
    logger.info(f"📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
