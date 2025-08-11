#!/usr/bin/env python3
"""
TinyMistral Ablation Analysis - Measuring Expert Impact on Routing Patterns
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import original visualization functions
from src.analysis.visualization import plot_ablation_heatmap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TinyMistralAblationAnalyzer:
    """Ablation analysis measuring expert impact on routing and model behavior."""
    
    def __init__(self, model_name="M4-ai/TinyMistral-6x248M"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        self.num_classes = len(self.class_names)
        
    def load_model(self):
        """Load TinyMistral model and tokenizer."""
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
        
        logger.info(f"✅ Model loaded: {self.num_layers} layers, {self.num_experts} experts per layer")
        
    def prepare_data(self, max_samples=200):
        """Prepare AG News dataset by class."""
        logger.info("📚 Loading AG News dataset...")
        
        # Load dataset
        dataset = load_dataset("ag_news", split="test")
        
        # Group samples by class for balanced analysis
        class_samples = {i: [] for i in range(self.num_classes)}
        
        for idx, (text, label) in enumerate(zip(dataset["text"], dataset["label"])):
            if len(class_samples[label]) < max_samples // self.num_classes:
                class_samples[label].append((text, label))
        
        # Combine all samples
        all_samples = []
        for class_id, samples in class_samples.items():
            all_samples.extend(samples)
        
        logger.info(f"✅ Dataset prepared: {len(all_samples)} samples")
        return all_samples
        
    def measure_baseline_metrics(self, samples):
        """Measure baseline routing patterns and perplexity."""
        logger.info("📊 Measuring baseline metrics...")
        
        self.model.eval()
        
        # Group samples by class
        class_metrics = {i: {'perplexity': [], 'routing_entropy': []} for i in range(self.num_classes)}
        
        with torch.no_grad():
            for text, label in tqdm(samples, desc="Baseline evaluation"):
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Calculate perplexity
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()
                
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_losses = token_losses.view(shift_labels.shape)
                
                # Average perplexity for this sample
                valid_tokens = (shift_labels != self.tokenizer.pad_token_id).sum()
                if valid_tokens > 0:
                    avg_loss = token_losses.sum() / valid_tokens
                    perplexity = torch.exp(avg_loss).item()
                    class_metrics[label]['perplexity'].append(perplexity)
        
        # Calculate average metrics per class
        baseline_metrics = {}
        for class_id in range(self.num_classes):
            baseline_metrics[class_id] = {
                'avg_perplexity': np.mean(class_metrics[class_id]['perplexity']) if class_metrics[class_id]['perplexity'] else 0,
                'std_perplexity': np.std(class_metrics[class_id]['perplexity']) if class_metrics[class_id]['perplexity'] else 0
            }
        
        return baseline_metrics
    
    def measure_ablated_metrics(self, samples, target_layer, expert_to_ablate):
        """Measure metrics with one expert ablated."""
        
        # Get expert path
        expert_path = self.model.model.layers[target_layer].block_sparse_moe.experts[expert_to_ablate]
        
        # Store and zero out weights
        original_weights = [p.clone().detach() for p in expert_path.parameters()]
        
        with torch.no_grad():
            for param in expert_path.parameters():
                param.data.fill_(0)
        
        # Measure metrics with ablated expert
        self.model.eval()
        class_metrics = {i: {'perplexity': []} for i in range(self.num_classes)}
        
        with torch.no_grad():
            for text, label in tqdm(samples, desc=f"Ablating Expert {expert_to_ablate}", leave=False):
                # Tokenize
                inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Calculate perplexity
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()
                
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                token_losses = token_losses.view(shift_labels.shape)
                
                # Average perplexity for this sample
                valid_tokens = (shift_labels != self.tokenizer.pad_token_id).sum()
                if valid_tokens > 0:
                    avg_loss = token_losses.sum() / valid_tokens
                    perplexity = torch.exp(avg_loss).item()
                    class_metrics[label]['perplexity'].append(perplexity)
        
        # Restore weights
        with torch.no_grad():
            for i, param in enumerate(expert_path.parameters()):
                param.data.copy_(original_weights[i])
        
        # Calculate average metrics per class
        ablated_metrics = {}
        for class_id in range(self.num_classes):
            ablated_metrics[class_id] = {
                'avg_perplexity': np.mean(class_metrics[class_id]['perplexity']) if class_metrics[class_id]['perplexity'] else 0
            }
        
        return ablated_metrics
    
    def run_ablation_analysis(self, samples, target_layer=0):
        """Run comprehensive ablation analysis."""
        logger.info(f"🔬 Starting ablation analysis on Layer {target_layer}")
        
        # Calculate baseline metrics
        baseline_metrics = self.measure_baseline_metrics(samples)
        
        logger.info("📊 Baseline Metrics by Class:")
        baseline_df_data = []
        for class_id in range(self.num_classes):
            avg_perp = baseline_metrics[class_id]['avg_perplexity']
            logger.info(f"  {self.class_names[class_id]}: Perplexity = {avg_perp:.2f}")
            baseline_df_data.append(avg_perp)
        
        baseline_df = pd.DataFrame(
            [baseline_df_data], 
            columns=[name[:12] for name in self.class_names], 
            index=['Baseline']
        )
        
        # Run ablation for each expert
        ablation_results = []
        perplexity_increases = []
        
        for expert_to_ablate in range(self.num_experts):
            logger.info(f"🎯 Analyzing Expert {expert_to_ablate}...")
            
            # Measure with expert ablated
            ablated_metrics = self.measure_ablated_metrics(samples, target_layer, expert_to_ablate)
            
            # Calculate perplexity increases
            expert_increases = []
            for class_id in range(self.num_classes):
                baseline_perp = baseline_metrics[class_id]['avg_perplexity']
                ablated_perp = ablated_metrics[class_id]['avg_perplexity']
                increase = ablated_perp - baseline_perp
                expert_increases.append(increase)
            
            perplexity_increases.append(expert_increases)
            ablation_results.append(ablated_metrics)
        
        # Create results dataframe
        results_df = pd.DataFrame(
            perplexity_increases,
            columns=[name[:12] for name in self.class_names],
            index=[f"Expert {i}" for i in range(self.num_experts)]
        )
        
        logger.info("\n🎯 Ablation Analysis Results:")
        logger.info("Perplexity Increase After Ablating Each Expert:")
        logger.info(f"\n{results_df.round(3)}")
        
        return results_df, baseline_df
    
    def create_visualizations(self, results_df, output_dir="results/tinymistral_ablation_original"):
        """Create ablation visualizations using original functions."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 Creating ablation heatmap...")
        
        # Create the ablation heatmap using original function
        heatmap_path = output_path / "tinymistral_ablation_heatmap.png"
        plot_ablation_heatmap(results_df, save_path=heatmap_path)
        
        # Save detailed results
        results_path = output_path / "ablation_results.csv"
        results_df.to_csv(results_path)
        logger.info(f"✅ Results saved to {results_path}")
        
        # Create expert impact analysis
        expert_impact = results_df.max(axis=1)  # Maximum impact across all classes
        most_impactful_class = results_df.idxmax(axis=1)  # Class with highest impact
        
        impact_summary = []
        for expert in range(self.num_experts):
            expert_name = f"Expert {expert}"
            max_impact = expert_impact.loc[expert_name]
            impacted_class = most_impactful_class.loc[expert_name]
            impact_summary.append({
                'Expert': expert_name,
                'Most_Impacted_Class': impacted_class,
                'Max_Perplexity_Increase': max_impact
            })
        
        impact_df = pd.DataFrame(impact_summary)
        impact_path = output_path / "expert_impact_analysis.csv"
        impact_df.to_csv(impact_path, index=False)
        
        logger.info("\n🎯 Expert Impact Analysis:")
        logger.info("(Perplexity increase when expert is ablated)")
        for _, row in impact_df.iterrows():
            logger.info(f"  {row['Expert']}: {row['Most_Impacted_Class']} (+{row['Max_Perplexity_Increase']:.3f})")
        
        # Identify inactive experts
        max_impacts = impact_df['Max_Perplexity_Increase']
        inactive_threshold = 0.1  # Very small impact
        inactive_experts = impact_df[max_impacts < inactive_threshold]['Expert'].tolist()
        
        if inactive_experts:
            logger.info(f"\n⚠️  Potentially Inactive Experts (impact < {inactive_threshold}):")
            for expert in inactive_experts:
                logger.info(f"    {expert}")
        else:
            logger.info(f"\n✅ All experts show significant impact (≥ {inactive_threshold})")
        
        logger.info(f"✅ All results saved to {output_path}")
        return output_path

def main():
    """Main analysis function."""
    logger.info("🚀 Starting TinyMistral Ablation Analysis with Original Visualization")
    
    # Initialize analyzer
    analyzer = TinyMistralAblationAnalyzer()
    analyzer.load_model()
    
    # Prepare balanced data
    samples = analyzer.prepare_data(max_samples=200)  # 50 per class
    
    # Run ablation analysis on first MoE layer
    results_df, baseline_df = analyzer.run_ablation_analysis(samples, target_layer=0)
    
    # Create visualizations using original functions
    output_path = analyzer.create_visualizations(results_df)
    
    logger.info("🎉 Ablation analysis completed!")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info(f"🔍 **ABLATION HEATMAP**: {output_path}/tinymistral_ablation_heatmap.png")

if __name__ == "__main__":
    main()
