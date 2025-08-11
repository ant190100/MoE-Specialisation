#!/usr/bin/env python3
"""
Debug script to investigate validation issues with trained TinyMistral model.
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

def setup_model_and_data(config, device):
    """Setup model and data loaders."""
    print("🔄 Setting up model and data...")
    
    # Load tokenizer and model (fresh, not from checkpoint since training just completed)
    print("📝 Loading TinyMistral tokenizer...")
    tokenizer = load_tinymistral_tokenizer(config["MODEL_NAME"])
    
    print("🔄 Loading TinyMistral model...")
    model = TinyMistralForClassification(
        model_name=config["MODEL_NAME"],
        num_classes=config["NUM_CLASSES"]
    ).to(device)
    
    # Freeze base model parameters (same as training)
    print("❄️ Freezing base model parameters...")
    trainable_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            param.requires_grad = False
            frozen_params += param.numel()
    
    print(f"📊 Trainable parameters: {trainable_params:,}")
    print(f"❄️ Frozen parameters: {frozen_params:,}")
    
    # Create data loaders
    print("📚 Creating data loaders...")
    train_loader, val_loader, class_names = create_data_loaders(
        dataset_name=config["DATASET"],
        batch_size=config["BATCH_SIZE"],
        tokenizer=tokenizer,
    )
    
    criterion = nn.CrossEntropyLoss()
    
    return model, val_loader, class_names, criterion, tokenizer

def debug_validation_step_by_step(model, val_loader, class_names, criterion, device, max_batches=5):
    """Debug validation step by step to identify bottleneck."""
    print(f"\n🔍 DEBUGGING VALIDATION - Testing first {max_batches} batches")
    print("="*60)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
                
            print(f"\nBatch {i+1}/{max_batches}")
            print("-" * 30)
            
            start_time = time.time()
            
            # Step 1: Move to device
            print("1. Moving data to device...")
            step_start = time.time()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            print(f"   ✅ Data moved in {time.time() - step_start:.2f}s")
            print(f"   📊 Batch shape: {input_ids.shape}")
            
            # Step 2: Forward pass
            print("2. Running forward pass...")
            step_start = time.time()
            outputs = model(input_ids, attention_mask)
            forward_time = time.time() - step_start
            print(f"   ✅ Forward pass completed in {forward_time:.2f}s")
            print(f"   📊 Output shape: {outputs.shape}")
            
            if forward_time > 30:  # If a single batch takes more than 30 seconds
                print("   ⚠️  SLOW FORWARD PASS DETECTED!")
                break
            
            # Step 3: Loss calculation
            print("3. Calculating loss...")
            step_start = time.time()
            loss = criterion(outputs, labels)
            print(f"   ✅ Loss calculated in {time.time() - step_start:.2f}s")
            print(f"   📊 Loss: {loss.item():.4f}")
            
            # Step 4: Predictions
            print("4. Getting predictions...")
            step_start = time.time()
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = 100 * correct / labels.size(0)
            print(f"   ✅ Predictions calculated in {time.time() - step_start:.2f}s")
            print(f"   📊 Batch accuracy: {accuracy:.2f}%")
            
            total_time = time.time() - start_time
            print(f"\n⏱️  Total batch time: {total_time:.2f}s")
            
            if total_time > 60:  # If a single batch takes more than 60 seconds
                print("⚠️  VERY SLOW BATCH PROCESSING!")
                print("This suggests the validation is hanging due to slow processing.")
                break

def debug_validation_memory_usage(model, val_loader, device, max_batches=3):
    """Debug memory usage during validation."""
    print(f"\n🧠 DEBUGGING MEMORY USAGE - Testing first {max_batches} batches")
    print("="*60)
    
    model.eval()
    
    if device == "cuda":
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
                
            print(f"\nBatch {i+1}/{max_batches}")
            
            # Before processing
            if device == "cuda":
                mem_before = torch.cuda.memory_allocated() / 1024**3
                print(f"Memory before batch: {mem_before:.2f} GB")
            
            # Process batch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            if device == "cuda":
                mem_after_move = torch.cuda.memory_allocated() / 1024**3
                print(f"Memory after moving to device: {mem_after_move:.2f} GB")
            
            outputs = model(input_ids, attention_mask)
            
            if device == "cuda":
                mem_after_forward = torch.cuda.memory_allocated() / 1024**3
                print(f"Memory after forward pass: {mem_after_forward:.2f} GB")
                
                if mem_after_forward > 8.0:  # More than 8GB
                    print("⚠️  HIGH MEMORY USAGE DETECTED!")

def quick_validation_test(model, val_loader, class_names, criterion, device):
    """Run a quick validation test with limited batches."""
    print(f"\n🎯 QUICK VALIDATION TEST - Limited to 10 batches")
    print("="*60)
    
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    num_batches = 0
    max_batches = 10  # Very limited for testing
    
    start_time = time.time()
    
    with torch.no_grad():
        progress_bar = tqdm(enumerate(val_loader), total=max_batches, desc="Validation")
        
        for i, batch in progress_bar:
            if i >= max_batches:
                break
                
            # Check if this batch is taking too long
            batch_start = time.time()
            
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
            
            batch_time = time.time() - batch_start
            current_acc = 100 * correct / total
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'Time/batch': f'{batch_time:.1f}s'
            })
            
            if batch_time > 30:  # If batch takes more than 30 seconds
                print(f"\n⚠️  Batch {i+1} took {batch_time:.1f}s - this is the bottleneck!")
                break
    
    total_time = time.time() - start_time
    accuracy = 100 * correct / total if total > 0 else 0
    avg_loss = val_loss / num_batches if num_batches > 0 else 0
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Processed batches: {num_batches}")
    print(f"   Total samples: {total}")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Average time per batch: {total_time/num_batches:.1f}s")
    
    if total_time / num_batches > 15:  # More than 15 seconds per batch
        print("⚠️  SLOW VALIDATION DETECTED - Each batch takes >15 seconds")
        print("This explains why validation appears to hang.")

def main():
    print("🔍 TinyMistral Validation Debug Tool")
    print("="*60)
    
    # Setup
    device = setup_device()
    config = load_config()
    
    print(f"📊 Configuration:")
    print(f"   Model: {config['MODEL_NAME']}")
    print(f"   Dataset: {config['DATASET']}")
    print(f"   Batch size: {config['BATCH_SIZE']}")
    print(f"   Max length: {config.get('MAX_LENGTH', 64)}")
    
    # Load model and data
    try:
        model, val_loader, class_names, criterion, tokenizer = setup_model_and_data(config, device)
        print(f"✅ Setup complete!")
        
        print(f"\n📚 Validation dataset info:")
        print(f"   Total batches in val_loader: {len(val_loader)}")
        print(f"   Samples per batch: {config['BATCH_SIZE']}")
        print(f"   Estimated total validation samples: {len(val_loader) * config['BATCH_SIZE']}")
        
        # Run debug tests
        print(f"\n" + "="*60)
        print("STARTING VALIDATION DEBUGGING")
        print("="*60)
        
        # Test 1: Step-by-step debugging
        debug_validation_step_by_step(model, val_loader, class_names, criterion, device)
        
        # Test 2: Memory usage (if CUDA)
        if device == "cuda":
            debug_validation_memory_usage(model, val_loader, device)
        
        # Test 3: Quick validation
        quick_validation_test(model, val_loader, class_names, criterion, device)
        
        print(f"\n" + "="*60)
        print("🎉 DEBUGGING COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
