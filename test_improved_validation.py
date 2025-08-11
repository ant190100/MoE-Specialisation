#!/usr/bin/env python3
"""
Quick test of the improved validation with progress tracking.
"""

import torch
import torch.nn as nn
import yaml
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time

sys.path.append(os.path.dirname(__file__))

from src.models.tinymistral_classifier import TinyMistralForClassification, load_tinymistral_tokenizer
from src.data.loaders import create_data_loaders

def setup_device():
    """Setup the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🎯 Using CUDA: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("💻 Using CPU")
    return device

def load_config():
    """Load the training configuration."""
    config_path = "configs/tinymistral_training.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def improved_validate(model, val_loader, class_names, criterion, device, config):
    """Test the improved validation function."""
    print("🧪 Testing Improved Validation Function")
    print("="*50)
    
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    num_batches = 0
    class_correct = {i: 0 for i in range(config["NUM_CLASSES"])}
    class_total = {i: 0 for i in range(config["NUM_CLASSES"])}
    
    # Determine validation batch limit based on device
    max_val_batches = config.get("MAX_VAL_BATCHES", 50)
    if device == "cpu":
        max_val_batches = min(max_val_batches, 15)  # Very limited for CPU
        print(f"🎯 Running validation (CPU optimized: {max_val_batches} batches)...")
    else:
        print(f"🎯 Running validation ({max_val_batches} batches)...")
    
    # Add progress bar for validation
    val_progress = tqdm(
        enumerate(val_loader), 
        total=max_val_batches,
        desc="Validation",
        leave=False
    )
    
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in val_progress:
            if i >= max_val_batches:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            num_batches += 1
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar with current stats
            current_acc = 100 * correct / total if total > 0 else 0
            val_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Samples': f'{total}'
            })
            
            # Per-class accuracy
            for j in range(labels.size(0)):
                label = labels[j].item()
                class_total[label] += 1
                if predicted[j] == labels[j]:
                    class_correct[label] += 1
    
    val_time = time.time() - start_time            
    accuracy = 100 * correct / total
    avg_val_loss = val_loss / num_batches
    
    # Per-class accuracy logging
    class_accuracies = {}
    for i in range(config["NUM_CLASSES"]):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            class_accuracies[class_names[i]] = class_acc
    
    print(f"\n✅ Validation completed in {val_time:.1f}s ({num_batches} batches, {total} samples)")
    print(f"📊 Results: Loss={avg_val_loss:.4f}, Accuracy={accuracy:.2f}%")
    print(f"⏱️  Average time per batch: {val_time/num_batches:.1f}s")
    print(f"📈 Estimated time for full validation (950 batches): {(val_time/num_batches)*950/60:.1f} minutes")
    
    if config.get("DETAILED_LOGGING", False):
        print("\n📋 Per-class accuracies:")
        for class_name, class_acc in class_accuracies.items():
            print(f"   {class_name}: {class_acc:.2f}%")
    
    return accuracy, avg_val_loss, class_accuracies

def main():
    print("🧪 Testing Improved Validation")
    print("="*50)
    
    # Setup
    device = setup_device()
    config = load_config()
    
    print(f"📊 Configuration:")
    print(f"   Model: {config['MODEL_NAME']}")
    print(f"   Dataset: {config['DATASET']}")
    print(f"   Batch size: {config['BATCH_SIZE']}")
    print(f"   Max validation batches: {config.get('MAX_VAL_BATCHES', 50)}")
    
    try:
        # Setup model and data
        print("\n🔄 Setting up model and data...")
        tokenizer = load_tinymistral_tokenizer(config["MODEL_NAME"])
        
        model = TinyMistralForClassification(
            model_name=config["MODEL_NAME"],
            num_classes=config["NUM_CLASSES"]
        ).to(device)
        
        # Freeze base model parameters
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        train_loader, val_loader, class_names = create_data_loaders(
            dataset_name=config["DATASET"],
            batch_size=config["BATCH_SIZE"],
            tokenizer=tokenizer,
        )
        
        criterion = nn.CrossEntropyLoss()
        
        print(f"✅ Setup complete!")
        
        # Test improved validation
        accuracy, val_loss, class_accuracies = improved_validate(
            model, val_loader, class_names, criterion, device, config
        )
        
        print(f"\n🎉 Test completed! The improved validation is much faster.")
        print(f"💡 Key improvements:")
        print(f"   - Limited to {config.get('MAX_VAL_BATCHES', 15)} batches on CPU")
        print(f"   - Added progress bar with live updates")
        print(f"   - Shows time estimates")
        print(f"   - Much more responsive user experience")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
