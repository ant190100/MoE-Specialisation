#!/usr/bin/env python3
"""
Comprehensive model diagnostics for trained TinyMistral model.
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import time

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "experiments"))

from tinymistral_extended_training import TinyMistralExtendedTraining

class ModelDiagnostics:
    """Comprehensive diagnostics for trained TinyMistral model."""
    
    def __init__(self, config_path="configs/tinymistral_training.yaml"):
        # Create trainer instance for loading model
        self.trainer = TinyMistralExtendedTraining(config_path)
        self.device = self.trainer.device
        self.results_dir = self.trainer.results_dir
        
        # Load trained model
        self.epoch, self.accuracy, self.train_losses, self.val_accuracies = self.trainer.load_from_checkpoint()
        self.model = self.trainer.model
        self.val_loader = self.trainer.val_loader
        self.class_names = self.trainer.class_names
        self.config = self.trainer.config
        
    def analyze_model_parameters(self):
        """Analyze model parameters and architecture."""
        print(f"\n{'='*60}")
        print("📊 MODEL PARAMETER ANALYSIS")
        print(f"{'='*60}")
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {frozen_params:,}")
        print(f"Training ratio: {trainable_params/total_params*100:.4f}%")
        
        # Analyze parameter distributions
        print(f"\n📈 Trainable layer statistics:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                stats = {
                    'shape': list(param.shape),
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'min': param.data.min().item(),
                    'max': param.data.max().item(),
                    'num_params': param.numel()
                }
                print(f"  {name}:")
                print(f"    Shape: {stats['shape']}, Params: {stats['num_params']:,}")
                print(f"    Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                print(f"    Range: [{stats['min']:.6f}, {stats['max']:.6f}]")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params
        }
    
    def analyze_validation_performance(self):
        """Run detailed validation analysis."""
        print(f"\n{'='*60}")
        print("🎯 VALIDATION PERFORMANCE ANALYSIS")
        print(f"{'='*60}")
        
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        
        correct = 0
        total = 0
        val_loss = 0
        num_batches = 0
        class_correct = {i: 0 for i in range(self.config["NUM_CLASSES"])}
        class_total = {i: 0 for i in range(self.config["NUM_CLASSES"])}
        
        # Run validation with progress tracking
        max_val_batches = self.config.get("MAX_VAL_BATCHES", 50)
        if self.device == "cpu":
            max_val_batches = min(max_val_batches, 25)
            
        print(f"Running validation on {max_val_batches} batches...")
        
        start_time = time.time()
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), total=max_val_batches, desc="Validation")
            
            for i, batch in progress_bar:
                if i >= max_val_batches:
                    break
                    
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class accuracy
                for j in range(labels.size(0)):
                    label = labels[j].item()
                    class_total[label] += 1
                    if predicted[j] == labels[j]:
                        class_correct[label] += 1
                
                # Update progress
                current_acc = 100 * correct / total if total > 0 else 0
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'Samples': f'{total}'
                })
        
        val_time = time.time() - start_time
        accuracy = 100 * correct / total
        avg_loss = val_loss / num_batches
        
        # Calculate per-class accuracies
        class_accuracies = {}
        for i in range(self.config["NUM_CLASSES"]):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                class_accuracies[self.class_names[i]] = class_acc
        
        # Print results
        print(f"\n📊 VALIDATION RESULTS:")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Validation Time: {val_time:.1f}s")
        print(f"Processed: {num_batches} batches, {total} samples")
        
        print(f"\n📋 Per-Class Accuracies:")
        sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
        for class_name, class_acc in sorted_classes:
            print(f"  {class_name}: {class_acc:.2f}%")
        
        # Identify best and worst
        best_class = max(class_accuracies.items(), key=lambda x: x[1])
        worst_class = min(class_accuracies.items(), key=lambda x: x[1])
        
        print(f"\n🏆 Best performing class: {best_class[0]} ({best_class[1]:.2f}%)")
        print(f"📉 Worst performing class: {worst_class[0]} ({worst_class[1]:.2f}%)")
        print(f"📊 Performance gap: {best_class[1] - worst_class[1]:.2f}%")
        
        return {
            'overall_accuracy': accuracy,
            'avg_loss': avg_loss,
            'class_accuracies': class_accuracies,
            'validation_time': val_time,
            'samples_processed': total
        }
    
    def analyze_prediction_confidence(self):
        """Analyze model prediction confidence patterns."""
        print(f"\n{'='*60}")
        print("🔍 PREDICTION CONFIDENCE ANALYSIS")
        print(f"{'='*60}")
        
        self.model.eval()
        confidence_by_class = {name: [] for name in self.class_names}
        correct_confidences = []
        incorrect_confidences = []
        
        max_batches = 15  # Limit for detailed analysis
        
        with torch.no_grad():
            progress_bar = tqdm(enumerate(self.val_loader), total=max_batches, desc="Confidence Analysis")
            
            for i, batch in progress_bar:
                if i >= max_batches:
                    break
                    
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                _, predicted = torch.max(outputs, dim=1)
                
                # Collect confidence statistics
                for j in range(labels.size(0)):
                    true_class = self.class_names[labels[j].item()]
                    confidence = confidences[j].item()
                    is_correct = predicted[j] == labels[j]
                    
                    confidence_by_class[true_class].append(confidence)
                    
                    if is_correct:
                        correct_confidences.append(confidence)
                    else:
                        incorrect_confidences.append(confidence)
        
        # Analyze results
        print(f"\n📊 Confidence Statistics:")
        overall_avg_confidence = np.mean(correct_confidences + incorrect_confidences)
        correct_avg_confidence = np.mean(correct_confidences) if correct_confidences else 0
        incorrect_avg_confidence = np.mean(incorrect_confidences) if incorrect_confidences else 0
        
        print(f"Overall average confidence: {overall_avg_confidence:.3f}")
        print(f"Correct predictions confidence: {correct_avg_confidence:.3f}")
        print(f"Incorrect predictions confidence: {incorrect_avg_confidence:.3f}")
        print(f"Confidence gap: {correct_avg_confidence - incorrect_avg_confidence:.3f}")
        
        print(f"\n📈 Per-Class Confidence:")
        for class_name in self.class_names:
            if confidence_by_class[class_name]:
                avg_conf = np.mean(confidence_by_class[class_name])
                std_conf = np.std(confidence_by_class[class_name])
                samples = len(confidence_by_class[class_name])
                print(f"  {class_name}: {avg_conf:.3f} ± {std_conf:.3f} ({samples} samples)")
        
        return {
            'overall_confidence': overall_avg_confidence,
            'correct_confidence': correct_avg_confidence,
            'incorrect_confidence': incorrect_avg_confidence,
            'confidence_by_class': {k: np.mean(v) if v else 0 for k, v in confidence_by_class.items()}
        }
    
    def run_full_diagnostics(self):
        """Run all diagnostic analyses."""
        print(f"🔬 COMPREHENSIVE MODEL DIAGNOSTICS")
        print(f"Model loaded from epoch {self.epoch+1} with {self.accuracy:.2f}% accuracy")
        
        # Run all analyses
        param_analysis = self.analyze_model_parameters()
        val_analysis = self.analyze_validation_performance()
        confidence_analysis = self.analyze_prediction_confidence()
        
        # Save results
        diagnostic_results = {
            'timestamp': datetime.now().isoformat(),
            'model_checkpoint': f'epoch_{self.epoch+1}',
            'loaded_accuracy': self.accuracy,
            'parameter_analysis': param_analysis,
            'validation_analysis': val_analysis,
            'confidence_analysis': confidence_analysis,
            'config': self.config
        }
        
        # Save to file
        results_path = self.results_dir / "analysis" / f"diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(diagnostic_results, results_path)
        
        print(f"\n💾 Diagnostic results saved to: {results_path}")
        print(f"\n{'='*60}")
        print("🎉 DIAGNOSTICS COMPLETE!")
        print(f"{'='*60}")
        
        return diagnostic_results

def main():
    """Run model diagnostics."""
    try:
        diagnostics = ModelDiagnostics()
        results = diagnostics.run_full_diagnostics()
        return results
    except Exception as e:
        print(f"❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
